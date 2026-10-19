// ExpertStore core: construction (hot upload, the pinned cold copy and its background page-locking), the stream
// mode of prompt chunks (two slots, prefetch two layers ahead), the compute-side acquire / release / signal.
// Sizing and the plan: expert_store_plan.cu; the decode ring and admission: expert_store_ring.cu.
#include "runtime/expert_store.h"

#include "core/cuda_check.h"
#include "runtime/expert_store_util.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <thread>

namespace truss::runtime {

using namespace store_detail;

ExpertStore::ExpertStore(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z,
                         std::vector<std::pair<int, int>> fill)
{
    fill_ = std::move(fill);
    const int L = (int) layers.size(), half_l = (L + 1) / 2;
    if ((int) hot.size() != L) throw std::invalid_argument("ExpertStore: hot set has the wrong layer count");
    layers_.resize(L);
    TRUSS_CUDA(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
    for (int s = 0; s < 2; ++s) {
        TRUSS_CUDA(cudaEventCreateWithFlags(&copied_[s], cudaEventDisableTiming));
        TRUSS_CUDA(cudaEventCreateWithFlags(&released_[s], cudaEventDisableTiming));
    }
    TRUSS_CUDA(cudaEventCreateWithFlags(&compute_mark_, cudaEventDisableTiming));
    // Every event the driver thread records is created here, on the constructing thread: cuEventCreate takes the
    // context's write lock, which a host thread blocked in cudaLaunchKernel (its queue full behind a doorbell spin
    // that waits for the driver) holds as a reader: the driver then never writes the plan (TRACKER #115).
    for (cudaEvent_t * e : { &demand_ev_, &piece_ev_, &fence_ev_, &adm_ev_ })
        TRUSS_CUDA(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    ev_pool_.resize(LANDING_EVENTS);
    for (cudaEvent_t & e : ev_pool_) TRUSS_CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    if (std::getenv("TRUSS_STREAM_PROFILE") && std::atoi(std::getenv("TRUSS_STREAM_PROFILE"))) {
        stream_prof_ = true;   // measurement build only: timing events for the per-chunk stream profile
        prof_pool_.resize(1024);
        for (cudaEvent_t & e : prof_pool_) TRUSS_CUDA(cudaEventCreate(&e));
    }
    TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&signal_src_), sizeof(int) * SIGNAL_RING, cudaHostAllocDefault));
    pinned_.push_back(signal_src_);

    size_t hot_lo = 0, hot_hi = 0;
    for (int l = 0; l < L; ++l) {
        Layer & Y = layers_[l];
        const int E = layers[l][0]->n_expert;
        Y.n_expert = E;
        Y.bytes.resize(E);
        for (int p = 0; p < 3; ++p) Y.part[p].resize(E);
        for (int e = 0; e < E; ++e) {
            Y.part[0][e] = 0;
            Y.part[1][e] = proj_bytes(*layers[l][0], e);
            Y.part[2][e] = Y.part[1][e] + proj_bytes(*layers[l][1], e);
            Y.bytes[e] = expert_bytes(layers[l], e);
            (hot[l][e] ? (l < half_l ? hot_lo : hot_hi) : Y.cold) += Y.bytes[e];
        }
    }
    ring_ = ring_size(layers, hot, z);
    vend_ = (int64_t) ring_;
    // E7: the pinned refill may take everything but Sizes::pin_fill, which stays free for the FIFO (0 = off)
    pin_budget_ = z.pin_fill && ring_ > z.pin_fill ? ring_ - z.pin_fill : 0;
    ring_hi_ = (int64_t) ring_;
    for (const Layer & Y : layers_) slot_ = std::max(slot_, align_up(Y.cold));
    auto * arena = static_cast<uint8_t *>(device_alloc(hot_lo + ring_ + hot_hi, device_));
    base_ = arena + hot_lo;
    spare_ = z.stream_extra ? base_ + 2 * slot_ : nullptr;
    stream_end_ = (int64_t) align_up(2 * slot_ + z.stream_extra);
    device_bytes_ = hot_lo + ring_ + hot_hi;

    // Every layer's cold copy lives in one anonymous mapping, filled from the file (pread) by TRUSS_LOAD_THREADS
    // workers (16) and page-locked afterwards by start_pinning(): cudaHostAlloc per layer cost ~0.6 s/GB on this box,
    // an anonymous fill ~0.1 s/GB and cudaHostRegister ~0.08 s/GB. Hot experts go to the device through per-worker
    // pinned staging buffers in 32 MB batches (was one pageable cudaMemcpy per expert projection, ~14K of them).
    // X3.1: this constructor 39.6 -> 3.6 s, the Forward 43 -> 6.7 s, the server ready ~2 min -> 8.3 s.
    std::vector<size_t> region_at(L, 0);
    for (int l = 0; l < L; ++l) {
        region_at[l] = host_region_bytes_;
        host_region_bytes_ += align_up(layers_[l].cold, 4096);
        cold_total_ += layers_[l].cold;
    }
    if (host_region_bytes_) {
        void * m = mmap(nullptr, host_region_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m == MAP_FAILED) throw std::runtime_error("ExpertStore: mmap of the cold copy failed");
        madvise(m, host_region_bytes_, MADV_NOHUGEPAGE);   // THP faults compact this fragmented box: 8 GB took 31 s
        host_region_ = static_cast<uint8_t *>(m);
    }
    struct Hot {   // a layer's hot experts: one contiguous device run in expert order
        uint8_t * dev = nullptr;
        std::vector<int> experts;
    };
    std::vector<Hot> hot_run(L);
    std::vector<std::vector<int32_t>> smeta_all(L);
    uint8_t * next_hot[2] = { arena, arena + hot_lo + ring_ };
    for (int l = 0; l < L; ++l) {
        Layer & Y = layers_[l];
        const int E = Y.n_expert;
        if (Y.cold) Y.host = host_region_ + region_at[l];
        TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&Y.ring_meta_host), sizeof(int32_t) * 3 * 2 * E,
                                 cudaHostAllocDefault));
        pinned_.push_back(Y.ring_meta_host);
        std::vector<int32_t> & smeta = smeta_all[l];
        smeta.assign(3 * 2 * E, 0);
        Y.cold_off.assign(E, -1);
        Y.ring_at.assign(E, -1);
        Y.pend.assign(E, 0);
        Y.ref.assign(E, 0);
        uint8_t * slot_dst = base_ + (l % 2) * slot_;
        size_t cold_pos = 0;
        for (int e = 0; e < E; ++e) {
            uint8_t * at;
            if (hot[l][e]) {
                uint8_t *& h = next_hot[l < half_l ? 0 : 1];
                if (hot_run[l].experts.empty()) hot_run[l].dev = h;
                hot_run[l].experts.push_back(e);
                at = h;
                h += Y.bytes[e];
            } else {
                Y.cold_off[e] = (int64_t) cold_pos;
                at = slot_dst + cold_pos;
                cold_pos += Y.bytes[e];
            }
            for (int p = 0; p < 3; ++p) {
                const formats::ExpertTable & t = *layers[l][p];
                int32_t * sm = smeta.data() + (size_t) p * 2 * E, * rm = Y.ring_meta_host + (size_t) p * 2 * E;
                sm[2 * e] = rm[2 * e] = t.k[e];
                sm[2 * e + 1] = unit_offset(at + Y.part[p][e], base_, l);
                rm[2 * e + 1] = hot[l][e] ? sm[2 * e + 1] : 0;   // cold: set when fetched
            }
        }
    }
    {
        // jobs: (layer, 0) its cold experts -> the host copy, (layer, 1) its hot run -> the device. A layer's file
        // pages are dropped by whichever of its two jobs finishes last (the load's resident peak otherwise counts the
        // pack twice, pinned + page cache, ~69 GB measured, beside renters holding 58-77 GB).
        constexpr size_t STAGE = 32u << 20;
        const int n_jobs = 2 * L;
        const unsigned hw = std::thread::hardware_concurrency();
        const int T = std::max(1, std::min<int>(std::getenv("TRUSS_LOAD_THREADS") ? std::atoi(std::getenv("TRUSS_LOAD_THREADS")) : 16,
                                                 hw ? (int) hw : 16));
        std::atomic<int> next{ 0 };
        std::vector<std::atomic<int>> layer_left(L);
        for (auto & a : layer_left) a = 2;
        std::vector<std::string> errs(T);
        auto copy_part = [&](int l, int e, uint8_t * dst) {   // expert e of layer l as [gate | up | down]
            for (int p = 0; p < 3; ++p) {
                const formats::ExpertTable & t = *layers[l][p];
                const uint8_t * from = reinterpret_cast<const uint8_t *>(t.trellis->data) + t.offset[e] * 2;
                // pread from the shard: no faults, no file pages mapped (through the mapping, 16 workers had ~16
                // layers of file pages resident at once)
                if (!gguf::read_mapped(from, proj_bytes(t, e), dst + layers_[l].part[p][e])) {
                    std::memcpy(dst + layers_[l].part[p][e], from, proj_bytes(t, e));
                    release_range(from, proj_bytes(t, e));
                }
            }
        };
        auto worker = [&](int w) {
            cudaStream_t st = nullptr;
            uint8_t * stage[2] = {};
            cudaEvent_t done[2] = {};
            try {
                for (;;) {
                    const int j = next.fetch_add(1);
                    if (j >= n_jobs) break;
                    const int l = j / 2;
                    const Layer & Y = layers_[l];
                    if (j % 2 == 0) {
                        for (int e = 0; e < Y.n_expert; ++e)
                            if (Y.cold_off[e] >= 0) copy_part(l, e, host_region_ + region_at[l] + Y.cold_off[e]);
                    } else if (!hot_run[l].experts.empty()) {
                        if (!st) {
                            TRUSS_CUDA(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
                            for (int b = 0; b < 2; ++b) {
                                TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&stage[b]), STAGE, cudaHostAllocDefault));
                                TRUSS_CUDA(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming | cudaEventBlockingSync));
                            }
                        }
                        int b = 0;
                        size_t fill = 0, dev_off = 0;
                        auto flush = [&] {
                            if (!fill) return;
                            TRUSS_CUDA(cudaMemcpyAsync(hot_run[l].dev + dev_off, stage[b], fill, cudaMemcpyHostToDevice, st));
                            TRUSS_CUDA(cudaEventRecord(done[b], st));
                            dev_off += fill, fill = 0, b ^= 1;
                            TRUSS_CUDA(cudaEventSynchronize(done[b]));   // the other buffer's last copy has left it
                        };
                        for (int e : hot_run[l].experts) {
                            if (Y.bytes[e] > STAGE) throw std::runtime_error("ExpertStore: expert larger than the load stage");
                            if (fill + Y.bytes[e] > STAGE) flush();
                            copy_part(l, e, stage[b] + fill);
                            fill += Y.bytes[e];
                        }
                        flush();
                        // the last copy may still read its stage buffer, and the next job restarts at stage[0]: without
                        // this wait it overwrote an in-flight batch (random hot experts corrupt at load, lead 10-07)
                        TRUSS_CUDA(cudaStreamSynchronize(st));
                    }
                    if (layer_left[l].fetch_sub(1) == 1)
                        for (int p = 0; p < 3; ++p) release_pages(layers[l][p]->trellis);
                }
                if (st) TRUSS_CUDA(cudaStreamSynchronize(st));
            } catch (const std::exception & ex) {
                errs[w] = ex.what();
                next = n_jobs;
                if (st) cudaStreamSynchronize(st);
            }
            for (int b = 0; b < 2; ++b) {
                if (stage[b]) cudaFreeHost(stage[b]);
                if (done[b]) cudaEventDestroy(done[b]);
            }
            if (st) cudaStreamDestroy(st);
        };
        const auto t0 = std::chrono::steady_clock::now();
        int dev = 0;
        TRUSS_CUDA(cudaGetDevice(&dev));
        std::vector<std::thread> pool;
        for (int w = 0; w < T; ++w)
            pool.emplace_back([&, w] {
                cudaSetDevice(dev);
                worker(w);
            });
        for (auto & th : pool) th.join();
        for (const std::string & m : errs)
            if (!m.empty()) throw std::runtime_error("ExpertStore load: " + m);
        if (std::getenv("TRUSS_LOAD_TIMES"))
            std::fprintf(stderr, "load ExpertStore: %d workers copied %.2f GB cold + uploaded the hot set in %.2f s\n", T,
                         cold_total_ / 1e9, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    for (int l = 0; l < L; ++l) {
        Layer & Y = layers_[l];
        const int E = Y.n_expert;
        const std::vector<int32_t> & smeta = smeta_all[l];
        for (int p = 0; p < 3; ++p) {
            const formats::ExpertTable & t = *layers[l][p];
            Y.stream_meta[p] = static_cast<int32_t *>(device_alloc(sizeof(int32_t) * 2 * E, device_));
            Y.ring_meta[p] = static_cast<int32_t *>(device_alloc(sizeof(int32_t) * 2 * E, device_));
            TRUSS_CUDA(cudaMemcpy(Y.stream_meta[p], smeta.data() + (size_t) p * 2 * E, sizeof(int32_t) * 2 * E,
                                  cudaMemcpyHostToDevice));
            TRUSS_CUDA(cudaMemcpy(Y.ring_meta[p], Y.ring_meta_host + (size_t) p * 2 * E, sizeof(int32_t) * 2 * E,
                                  cudaMemcpyHostToDevice));
            auto * suh = static_cast<half *>(device_alloc(t.suh->bytes, device_));
            auto * svh = static_cast<half *>(device_alloc(t.svh->bytes, device_));
            TRUSS_CUDA(cudaMemcpy(suh, t.suh->data, t.suh->bytes, cudaMemcpyHostToDevice));
            TRUSS_CUDA(cudaMemcpy(svh, t.svh->data, t.svh->bytes, cudaMemcpyHostToDevice));
            Y.suh[p] = suh, Y.svh[p] = svh;
            device_bytes_ += 2 * sizeof(int32_t) * 2 * E + t.suh->bytes + t.svh->bytes;
        }
    }}

void ExpertStore::start_pinning()
{
    // Page-locking the cold copy (~0.08 s/GB, serialized in the driver: ~3 s here) runs in the background, layer by
    // layer; the owner starts it after its own setup, since it holds the driver lock that cudaMalloc / cudaMemcpy /
    // cudaHostAlloc also take. Until a layer is registered its copies are pageable (synchronous on the issuing
    // thread, slower), never wrong; wait_pinned() blocks until all are done (the benches call it before timing).
    if (reg_thread_.joinable() || reg_ok_) return;
    const int L = (int) layers_.size();
    reg_ok_.reset(new std::atomic<uint8_t>[L]);
    for (int l = 0; l < L; ++l) reg_ok_[l] = 0;
    int dev = 0;
    TRUSS_CUDA(cudaGetDevice(&dev));
    reg_thread_ = std::thread([this, dev, L] {
        cudaSetDevice(dev);
        const auto r0 = std::chrono::steady_clock::now();
        for (int l = 0; l < L && !reg_stop_; ++l) {
            const Layer & Y = layers_[l];
            if (!Y.cold) continue;
            const cudaError_t e = cudaHostRegister(const_cast<uint8_t *>(Y.host), align_up(Y.cold, 4096),
                                                   cudaHostRegisterDefault);
            if (e != cudaSuccess) {   // the copies stay pageable: slower, still right
                std::fprintf(stderr, "ExpertStore: cudaHostRegister of layer %d failed (%s); its copies stay pageable\n", l,
                             cudaGetErrorString(e));
                continue;
            }
            reg_ok_[l] = 1;
        }
        if (std::getenv("TRUSS_LOAD_TIMES"))
            std::fprintf(stderr, "load ExpertStore: cold copy page-locked in the background in %.2f s\n",
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - r0).count());
    });
}

ExpertStore::~ExpertStore()
{
    reg_stop_ = true;
    if (reg_thread_.joinable()) reg_thread_.join();
    for (cudaEvent_t & e : prof_pool_) cudaEventDestroy(e);
    if (copy_) cudaStreamSynchronize(copy_);
    for (void * p : device_) cudaFree(p);
    for (void * p : pinned_) cudaFreeHost(p);
    if (host_region_) {
        for (size_t l = 0; l < layers_.size(); ++l)
            if (reg_ok_ && reg_ok_[l]) cudaHostUnregister(const_cast<uint8_t *>(layers_[l].host));
        munmap(host_region_, host_region_bytes_);
    }
    for (int s = 0; s < 2; ++s) {
        if (copied_[s]) cudaEventDestroy(copied_[s]);
        if (released_[s]) cudaEventDestroy(released_[s]);
    }
    if (compute_mark_) cudaEventDestroy(compute_mark_);
    if (adm_ev_) cudaEventDestroy(adm_ev_);
    for (auto & x : landing_) cudaEventDestroy(x.first);
    for (cudaEvent_t e : ev_pool_) cudaEventDestroy(e);
    if (piece_ev_) cudaEventDestroy(piece_ev_);
    if (demand_ev_) cudaEventDestroy(demand_ev_);
    if (fence_ev_) cudaEventDestroy(fence_ev_);
    if (copy_) cudaStreamDestroy(copy_);
}

moe::Weights ExpertStore::weights(int l) const
{
    const Layer & Y = layers_.at(l);
    moe::Weights w{};
    for (int p = 0; p < 3; ++p) {
        w.proj[p] = { reinterpret_cast<const uint16_t *>(base_), mode_ == Mode::STREAM ? Y.stream_meta[p] : Y.ring_meta[p],
                      Y.suh[p], Y.svh[p] };
        w.proj[p].shift = SHIFT;
    }
    w.n_expert = Y.n_expert;
    return w;
}

void ExpertStore::wait_compute(cudaStream_t compute)
{
    TRUSS_CUDA(cudaEventRecord(compute_mark_, compute));
    TRUSS_CUDA(cudaStreamWaitEvent(copy_, compute_mark_, 0));
}

void ExpertStore::begin_stream(cudaStream_t compute)
{
    prof_harvest();
    std::lock_guard<std::mutex> g(adm_mu_);
    wait_compute(compute);   // decode kernels may still read the ring the slots overwrite
    // The slots and the chunk's buffers take [0, stream_end_): only the entries there go. The rest of the ring and the
    // lent KV segments keep theirs for the next decode (TRACKER #115: an agent turn's prompt used to empty the ring).
    // The head moves past the stream area, so address order from the head is still FIFO order.
    for (auto it = fifo_.begin(); it != fifo_.end();)
        if (it->off < stream_end_) drop(*it), it = fifo_.erase(it);
        else ++it;
    if (head_ < stream_end_) head_ = stream_end_;
    hinted_.clear();
    adm_.clear();
    claims_.clear(), adm_backlog_ = 0;   // landing_ events stay: their claims are gone from the ring (checked by offset)
    // E7: pinned residents leave with the ring; the next begin_ring() refills
    if (pin_active_.load(std::memory_order_acquire)) pin_report(" (partial: the prompt dropped the rest)");
    for (const RingEntry & r : pins_) drop(r);
    pins_.clear();
    pin_active_.store(false, std::memory_order_release);
    pin_queued_ = false;
    ring_hi_ = (int64_t) ring_;
    released_recorded_[0] = released_recorded_[1] = false;
    mode_ = Mode::STREAM;
}

void ExpertStore::prefetch(int l)
{
    if (mode_ != Mode::STREAM) throw std::logic_error("ExpertStore::prefetch outside a begin_stream() chunk");
    const Layer & Y = layers_.at(l);
    const int s = l % 2;
    cudaEvent_t pb0 = nullptr, pb1 = nullptr;
    if (stream_prof_) { pb0 = prof_take(); pb1 = prof_take(); TRUSS_CUDA(cudaEventRecord(pb0, copy_)); }
    if (released_recorded_[s]) TRUSS_CUDA(cudaStreamWaitEvent(copy_, released_[s], 0));
    if (Y.cold) TRUSS_CUDA(cudaMemcpyAsync(base_ + s * slot_, Y.host, Y.cold, cudaMemcpyHostToDevice, copy_));
    TRUSS_CUDA(cudaEventRecord(copied_[s], copy_));
    if (pb1) { TRUSS_CUDA(cudaEventRecord(pb1, copy_)); busy_ev_.push_back({pb0, pb1}); }
}

size_t ExpertStore::prefetch_some(int l, const int * ids, int n)
{
    if (mode_ != Mode::STREAM) throw std::logic_error("ExpertStore::prefetch_some outside a begin_stream() chunk");
    const Layer & Y = layers_.at(l);
    const int s = l % 2;
    cudaEvent_t pb0 = nullptr, pb1 = nullptr;
    if (stream_prof_) { pb0 = prof_take(); pb1 = prof_take(); TRUSS_CUDA(cudaEventRecord(pb0, copy_)); }
    if (released_recorded_[s]) TRUSS_CUDA(cudaStreamWaitEvent(copy_, released_[s], 0));
    std::vector<uint8_t> want(Y.n_expert, 0);
    for (int i = 0; i < n; ++i)
        if (ids[i] >= 0 && Y.cold_off[ids[i]] >= 0) want[ids[i]] = 1;
    uint8_t * slot = base_ + s * slot_;
    size_t host_bytes = 0;
    // host copies coalesced over experts adjacent in the layer's cold layout (expert order)
    int64_t run0 = -1, run1 = -1;
    auto flush = [&] {
        if (run0 < 0) return;
        TRUSS_CUDA(cudaMemcpyAsync(slot + run0, Y.host + run0, run1 - run0, cudaMemcpyHostToDevice, copy_));
        host_bytes += run1 - run0;
        run0 = run1 = -1;
    };
    for (int e = 0; e < Y.n_expert; ++e) {
        if (!want[e]) continue;
        const int64_t o = Y.cold_off[e], b = (int64_t) Y.bytes[e];
        if (Y.ring_at[e] >= 0 && !Y.pend[e]) {   // outside [0, stream_end_) (begin_stream dropped the rest)
            TRUSS_CUDA(cudaMemcpyAsync(slot + o, real(Y.ring_at[e]), b, cudaMemcpyDeviceToDevice, copy_));
            continue;
        }
        if (run1 != o) flush();
        if (run0 < 0) run0 = o;
        run1 = o + b;
    }
    flush();
    TRUSS_CUDA(cudaEventRecord(copied_[s], copy_));
    if (pb1) { TRUSS_CUDA(cudaEventRecord(pb1, copy_)); busy_ev_.push_back({pb0, pb1}); }
    return host_bytes;
}

// FIFO allocation: evict the oldest entries overlapping [head, head + bytes) (wrapping to 0 when the tail is too
// short), copy the expert, point the ring meta at it.
void ExpertStore::signal(int l, int * flag, int seq)
{
    (void) l;
    int * src = signal_src_ + signal_next_;
    signal_next_ = (signal_next_ + 1) % SIGNAL_RING;
    *src = seq;
    TRUSS_CUDA(cudaMemcpyAsync(flag, src, sizeof(int), cudaMemcpyHostToDevice, copy_));
}

void ExpertStore::upload(void * dst, const void * src, size_t bytes)
{
    TRUSS_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, copy_));
}

void ExpertStore::acquire(int l, cudaStream_t compute)
{
    cudaEvent_t pw0 = nullptr;
    if (stream_prof_ && mode_ == Mode::STREAM) { pw0 = prof_take(); TRUSS_CUDA(cudaEventRecord(pw0, compute)); }
    if (mode_ == Mode::RING)
    {   // a put that landed after fetch()'s upload (admit_step during the CPU wait): the table goes up here, and
        // copied_ re-records behind it, so the kernel sees (meta, data) consistent (1.A)
        Layer & Y = layers_[l];
        if (Y.meta_dirty)
        {
            for (int p = 0; p < 3; ++p)
                TRUSS_CUDA(cudaMemcpyAsync(Y.ring_meta[p], Y.ring_meta_host + (size_t) p * 2 * Y.n_expert,
                                           sizeof(int32_t) * 2 * Y.n_expert, cudaMemcpyHostToDevice, copy_));
            Y.meta_dirty = false;
            TRUSS_CUDA(cudaEventRecord(copied_[l % 2], copy_));
        }
    }
    TRUSS_CUDA(cudaStreamWaitEvent(compute, copied_[l % 2], 0));
    if (pw0) { const cudaEvent_t pw1 = prof_take(); TRUSS_CUDA(cudaEventRecord(pw1, compute)); wait_ev_.push_back({pw0, pw1}); }
}

void ExpertStore::prof_harvest()
{
    if (!stream_prof_) return;
    for (auto & p : busy_ev_) {
        cudaEventSynchronize(p.second);
        float ms = 0;
        cudaEventElapsedTime(&ms, p.first, p.second);
        prof_busy_ms_ += ms;
    }
    for (auto & p : wait_ev_) {
        cudaEventSynchronize(p.second);
        float ms = 0;
        cudaEventElapsedTime(&ms, p.first, p.second);
        prof_wait_ms_ += ms;
    }
    busy_ev_.clear(); wait_ev_.clear(); prof_used_ = 0;   // all pairs synced: the pool is free again
}

void ExpertStore::stream_stats(double & busy_ms, double & wait_ms, bool reset) const
{
    busy_ms = prof_busy_ms_; wait_ms = prof_wait_ms_;
    if (reset) prof_busy_ms_ = prof_wait_ms_ = 0;
}

void ExpertStore::release(int l, cudaStream_t compute)
{
    TRUSS_CUDA(cudaEventRecord(released_[l % 2], compute));
    released_recorded_[l % 2] = true;
}

void ExpertStore::wait_pinned()
{
    start_pinning();
    if (reg_thread_.joinable()) reg_thread_.join();
}

}  // namespace truss::runtime

