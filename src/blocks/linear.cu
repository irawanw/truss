#include "blocks/linear.h"

#include "engine/limits.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

namespace truss::blocks {

namespace {

// A tensor's file pages are dead weight once it has been copied to the device. MADV_DONTNEED only drops them: the
// mapping stays PROT_READ|MAP_SHARED, so a later touch re-faults from disk and nothing can break. Done per tensor to
// keep the peak RSS down (upload used to fault ~48 GB at once, TRACKER #72).
void release_tensor(const gguf::Tensor * t)
{
    long ps = sysconf(_SC_PAGESIZE);
    if (!t || !t->data || !t->bytes || ps <= 0) return;
    const uintptr_t a = (uintptr_t) t->data, m = (uintptr_t) ps - 1;
    const uintptr_t lo = a & ~m, hi = (a + (uintptr_t) t->bytes + m) & ~m;
    madvise((void *) lo, (size_t) (hi - lo), MADV_DONTNEED);
}

// The tensors' file pages, faulted in by 16 threads (MADV_POPULATE_READ, 64 MB pieces) before the uploads read them:
// the pageable copies otherwise fault every 4 KB page on one thread (~2.6 s for ~5 GB).
void prefault(const std::vector<T> & a, const std::vector<T> & b, const std::vector<T> & c)
{
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
    const long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) return;
    constexpr size_t PIECE = 64u << 20;
    std::vector<std::pair<uintptr_t, size_t>> pieces;
    for (const auto * v : { &a, &b, &c })
        for (T t : *v) {
            if (!t || !t->data || !t->bytes) continue;
            const uintptr_t m = (uintptr_t) ps - 1, lo = (uintptr_t) t->data & ~m,
                            hi = ((uintptr_t) t->data + t->bytes + m) & ~m;
            for (uintptr_t x = lo; x < hi; x += PIECE) pieces.push_back({ x, std::min<size_t>(PIECE, hi - x) });
        }
    std::atomic<size_t> next{ 0 };
    std::vector<std::thread> th;
    for (int i = 0; i < 16; ++i)
        th.emplace_back([&] {
            for (size_t j; (j = next.fetch_add(1)) < pieces.size();)
                madvise((void *) pieces[j].first, pieces[j].second, MADV_POPULATE_READ);   // best effort
        });
    for (auto & t : th) t.join();
}

}  // namespace

Linear::~Linear()
{
    if (host_q_) cudaFreeHost(host_q_);
    if (host_d_) cudaFreeHost(host_d_);
}

void Linear::add(T t)
{
    if (!t) return;
    if (t->trellis) { trellis_list_.push_back(t); return; }
    (t->type == gguf::Type::Q8_0 ? q8_list_ : plain_).push_back(t);
}

// The trellis-coded matrices: the words and the two scale vectors go to the device as they are (the window GEMV
// reads the tiles in place), the scratch is sized for the widest list the calls can build.
void Linear::upload_trellis()
{
    if (trellis_list_.empty()) return;
    int dev = g_.device >= 0 ? g_.device : 0;
    TRUSS_CUDA(cudaDeviceGetAttribute(&td_sms_, cudaDevAttrMultiProcessorCount, dev));

    size_t words = 0, suh = 0, svh = 0, max_in = 0, max_out = 0;
    for (T t : trellis_list_) {
        const gguf::TrellisGroup * gr = t->trellis;
        words += (size_t) gr->trellis->elements();
        suh += (size_t) gr->in;
        svh += (size_t) gr->out;
        max_in = std::max<size_t>(max_in, (size_t) gr->in);
        max_out = std::max<size_t>(max_out, (size_t) gr->out);
    }
    uint16_t * B = g_.alloc<uint16_t>(words);
    half * su = g_.alloc<half>(suh);
    half * sv = g_.alloc<half>(svh);
    size_t bw = 0, bsu = 0, bsv = 0;
    for (T t : trellis_list_) {
        const gguf::TrellisGroup * gr = t->trellis;
        TRUSS_CUDA(cudaMemcpy(B + bw, gr->trellis->data, gr->trellis->bytes, cudaMemcpyHostToDevice));
        TRUSS_CUDA(cudaMemcpy(su + bsu, gr->suh->data, gr->suh->bytes, cudaMemcpyHostToDevice));
        TRUSS_CUDA(cudaMemcpy(sv + bsv, gr->svh->data, gr->svh->bytes, cudaMemcpyHostToDevice));
        release_tensor(gr->trellis);
        release_tensor(gr->suh);
        release_tensor(gr->svh);
        trellis_[t] = { B + bw, su + bsu, sv + bsv, gr->k, (int) gr->in, (int) gr->out };
        bw += (size_t) gr->trellis->elements();
        bsu += (size_t) gr->in;
        bsv += (size_t) gr->out;
    }

    // the window GEMV scratch: the staged activations of a TD_MAX_PARTS list plus the partial sums of the
    // TD_MAX_PARTS largest matrices (a list never holds more)
    const size_t a_bytes = trellis_dense::align256((size_t) trellis_dense::TD_MAX_PARTS * trellis_dense::TD_MAX_N *
                                                     max_in * sizeof(half));
    std::vector<size_t> p;
    for (T t : trellis_list_) {
        const trellis_dense::Matrix & M = trellis_.at(t);
        p.push_back((size_t) trellis_dense::kparts(M.out, M.in, td_sms_) * trellis_dense::TD_MAX_N * M.out *
                   sizeof(float));
    }
    std::sort(p.begin(), p.end(), std::greater<size_t>());
    size_t p_bytes = 0;
    for (size_t i = 0; i < p.size() && i < trellis_dense::TD_MAX_PARTS; ++i) p_bytes += p[i];
    td_scratch_bytes_ = a_bytes + p_bytes;
    td_scratch_ = g_.alloc<char>(td_scratch_bytes_);

    // the prompt path: the matrix is reconstructed a row chunk at a time, so the buffer stays small
    const size_t chunk = std::min<size_t>(1024, max_out);
    td_recon_bytes_ = chunk * max_in * sizeof(half);
    td_recon_ = g_.alloc<half>(chunk * max_in);
}

void Linear::upload(T host_rows, const std::vector<T> & no_w16)
{
    prefault(plain_, q8_list_, trellis_list_);
    engine::load_mark("dense weights: file pages faulted in");
    dev_ = std::make_unique<DeviceTensors>(plain_);
    for (T t : plain_) release_tensor(t);

    size_t biggest = 0, total_q = 0, total_d = 0;
    for (T t : q8_list_) {
        biggest = std::max<size_t>(biggest, t->bytes);
        if (t == host_rows) continue;
        total_q += (size_t) t->elements();
        total_d += (size_t) t->elements() / 32;
    }
    int8_t * qs = g_.alloc<int8_t>(total_q);
    half * ds = g_.alloc<half>(total_d);
    void * stage = engine::dmalloc<unsigned char>(biggest, false);
    size_t max_w = 0;
    for (T t : q8_list_) {
        if (q8_.count(t) || t == host_rows) continue;
        const int in = (int) t->shape[0], out = (int) t->elements() / in;
        TRUSS_CUDA(cudaMemcpy(stage, t->data, t->bytes, cudaMemcpyHostToDevice));
        release_tensor(t);
        dense::q8_repack(stage, in, out, qs, ds, g_.s);
        q8_[t] = { qs, ds, in, out };
        qs += (size_t) in * out;
        ds += (size_t) in * out / 32;
        if (std::find(no_w16.begin(), no_w16.end(), t) == no_w16.end()) max_w = std::max(max_w, (size_t) in * out);
    }
    if (host_rows) {   // repack on the device through temporaries, keep the result in pinned host memory
        const T t = host_rows;
        const int in = (int) t->shape[0], out = (int) t->elements() / in;
        const size_t nq = (size_t) in * out, nd = nq / 32;
        int8_t * tq = engine::dmalloc<int8_t>(nq, false);
        half * td = engine::dmalloc<half>(nd, false);
        TRUSS_CUDA(cudaMemcpy(stage, t->data, t->bytes, cudaMemcpyHostToDevice));
        release_tensor(t);
        dense::q8_repack(stage, in, out, tq, td, g_.s);
        TRUSS_CUDA(cudaStreamSynchronize(g_.s));
        TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&host_q_), nq, cudaHostAllocMapped));
        TRUSS_CUDA(cudaHostAlloc(reinterpret_cast<void **>(&host_d_), nd * sizeof(half), cudaHostAllocMapped));
        TRUSS_CUDA(cudaMemcpy(host_q_, tq, nq, cudaMemcpyDeviceToHost));
        TRUSS_CUDA(cudaMemcpy(host_d_, td, nd * sizeof(half), cudaMemcpyDeviceToHost));
        cudaFree(tq);
        cudaFree(td);
        q8_[t] = { host_q_, host_d_, in, out };   // unified addressing: the kernels read the host pointers directly
    }
    TRUSS_CUDA(cudaStreamSynchronize(g_.s));
    cudaFree(stage);
    w16_elems_ = max_w;
    w16_ = g_.alloc<half>(max_w);
    upload_trellis();
}

Linear::Act Linear::quant(const half * x, int rows, int in)
{
    Act a{ x };
    if (act_ == Activations::Q8_1) {
        a.q = g_.sc->alloc<int8_t>((size_t) rows * in);
        a.d = g_.sc->alloc<half>((size_t) rows * in / 32);
        dense::q8_quantize_act(x, rows, in, a.q, a.d, g_.s);
    }
    return a;
}

void Linear::lin(T t, const half * x, int rows, float * y)
{
    const size_t m = g_.sc->mark();
    const int in = trellis_.count(t) ? trellis_.at(t).in : q8_.at(t).in;
    lin(t, quant(x, rows, in), rows, y);
    g_.sc->release(m);
}

// A trellis-coded matrix: the window GEMV at decode sizes (the weights are read in place, never converted), the
// reconstructed matrix and a plain GEMM for the prompt rows.
void Linear::lin_trellis(const trellis_dense::Matrix & M, const Act & a, int rows, float * y)
{
    if (rows <= trellis_dense::TD_MAX_N) {
        trellis_dense::List L;
        trellis_dense::list_begin(L, M.in, rows, M.out);
        trellis_dense::list_add(L, M.B, M.suh, M.svh, M.out, y, M.out, td_sms_);
        trellis_dense::list_bind_scratch(L, td_scratch_);
        trellis_dense::matvec(L, M.k, a.x, g_.s);
        return;
    }
    const int chunk = (int) (td_recon_bytes_ / (sizeof(half) * (size_t) M.in));
    for (int r0 = 0; r0 < M.out; r0 += chunk) {
        const int nr = std::min(chunk, M.out - r0);
        trellis_dense::reconstruct(M, td_recon_, r0, nr, g_.s);
        const float one = 1.f, zero = 0.f;
        TRUSS_CUBLAS(cublasSetStream(g_.blas, g_.s));
        // column-major view: W^T is (in x nr, ld in), x^T (in x rows, ld in), y^T (nr x rows, ld out); the chunk
        // holds the output columns [r0, r0 + nr), so the output pointer moves by r0 (the row stride is out)
        TRUSS_CUBLAS(cublasGemmEx(g_.blas, CUBLAS_OP_T, CUBLAS_OP_N, nr, rows, M.in, &one, td_recon_, CUDA_R_16F, M.in,
                                  a.x, CUDA_R_16F, M.in, &zero, y + r0, CUDA_R_32F, M.out, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT));
    }
}

// Several trellis matrices of one input in one window GEMV (the parts of a list write to their own outputs).
void Linear::lin_trellis_multi(const std::vector<Proj> & ps, const Act & a, int rows)
{
    std::vector<const trellis_dense::Matrix *> Ms;
    std::vector<float *> ys;
    for (const Proj & p : ps) {
        const auto it = trellis_.find(p.first);
        if (it == trellis_.end()) continue;
        Ms.push_back(&it->second);
        ys.push_back(p.second);
    }
    if (Ms.empty()) return;
    if (rows > trellis_dense::TD_MAX_N) {   // the prompt path: one matrix at a time
        for (size_t i = 0; i < Ms.size(); ++i) lin_trellis(*Ms[i], a, rows, ys[i]);
        return;
    }
    size_t i = 0;
    while (i < Ms.size()) {
        size_t j = i;   // the run with the same input size and rate
        while (j < Ms.size() && Ms[j]->in == Ms[i]->in && Ms[j]->k == Ms[i]->k) ++j;
        for (size_t s = i; s < j; s += trellis_dense::TD_MAX_PARTS) {
            const size_t e = std::min(j, s + trellis_dense::TD_MAX_PARTS);
            trellis_dense::List L;
            trellis_dense::list_begin(L, Ms[s]->in, rows, 0);
            for (size_t c = s; c < e; ++c)
                trellis_dense::list_add(L, Ms[c]->B, Ms[c]->suh, Ms[c]->svh, Ms[c]->out, ys[c], Ms[c]->out, td_sms_);
            trellis_dense::list_bind_scratch(L, td_scratch_);
            trellis_dense::matvec(L, Ms[s]->k, a.x, g_.s);
        }
        i = j;
    }
}

void Linear::lin(T t, const Act & a, int rows, float * y)
{
    if (const auto it = trellis_.find(t); it != trellis_.end()) {
        lin_trellis(it->second, a, rows, y);
        return;
    }
    const dense::Q8Matrix & W = q8_.at(t);
    if (act_ == Activations::FP16 && rows <= dense::GEMV_ROWS) dense::q8_gemv_a16(W, a.x, rows, y, g_.s);
    else if (act_ == Activations::FP16) dense::q8_gemm_a16(W, a.x, rows, y, w16_, g_.blas, g_.s);
    else if (rows <= dense::GEMV_ROWS) dense::q8_gemv(W, a.q, a.d, rows, y, g_.s);
    else dense::q8_gemm(W, a.q, a.d, rows, y, g_.s);
}

void Linear::lin32(T t, const float * x, int rows, float * y)
{
    const DTensor & W = d(t);
    const int in = (int) W.ne[0], out = (int) W.ne[1];
    if (rows <= engine::FETCH_ROWS) {   // decode sizes: row-invariant (routing must not depend on the window size)
        dense::f32_gemv(W.as<float>(), in, out, x, rows, y, g_.s);
        return;
    }
    const float one = 1.f, zero = 0.f;
    TRUSS_CUBLAS(cublasSetStream(g_.blas, g_.s));
    TRUSS_CUBLAS(cublasSgemm(g_.blas, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &one, W.as<float>(), in, x, in, &zero,
                             y, out));
}

void Linear::lin_multi(const std::vector<Proj> & ps, const Act & a, int rows)
{
    lin_trellis_multi(ps, a, rows);
    // the Q8 path: the projections that are not trellis-coded
    std::vector<Proj> q8ps;
    for (const Proj & p : ps)
        if (!trellis_.count(p.first)) q8ps.push_back(p);
    if (q8ps.empty()) return;
    if (rows <= dense::GEMV_ROWS && q8ps.size() > 1 && q8ps.size() <= (size_t) dense::MULTI_MAX) {
        const dense::Q8Matrix * W[dense::MULTI_MAX];
        float * y[dense::MULTI_MAX];
        for (size_t i = 0; i < q8ps.size(); ++i) W[i] = &q8_.at(q8ps[i].first), y[i] = q8ps[i].second;
        if (act_ == Activations::FP16) dense::q8_gemv_multi_a16(W, y, (int) q8ps.size(), a.x, rows, g_.s);
        else dense::q8_gemv_multi(W, y, (int) q8ps.size(), a.q, a.d, rows, g_.s);
        return;
    }
    for (const Proj & p : q8ps) lin(p.first, a, rows, p.second);
}

void Linear::lin_multi_silu(const std::vector<Proj> & ps, half * y16, float scale, const Act & a, int rows)
{
    const int n = (int) ps.size();
    const dense::Q8Matrix * W[dense::MULTI_MAX];
    float * y[dense::MULTI_MAX];
    half * o16[dense::MULTI_MAX];
    float s16[dense::MULTI_MAX];
    for (int i = 0; i < n; ++i) W[i] = &q8_.at(ps[i].first), y[i] = ps[i].second, o16[i] = nullptr, s16[i] = 0.f;
    o16[0] = y16, s16[0] = scale;
    dense::q8_gemv_multi_a16_silu(W, y, o16, s16, n, a.x, rows, g_.s);
}

void Linear::lin32_multi(const std::vector<Proj> & ps, const float * x, int rows)
{
    if (rows <= engine::FETCH_ROWS && ps.size() > 1 && ps.size() <= (size_t) dense::MULTI_MAX) {
        const float * W[dense::MULTI_MAX];
        float * y[dense::MULTI_MAX];
        int out[dense::MULTI_MAX];
        const int in = (int) d(ps[0].first).ne[0];
        for (size_t i = 0; i < ps.size(); ++i) {
            const DTensor & t = d(ps[i].first);
            if ((int) t.ne[0] != in) throw std::runtime_error("lin32_multi: inputs differ");
            W[i] = t.as<float>(), out[i] = (int) t.ne[1], y[i] = ps[i].second;
        }
        dense::f32_gemv_multi(W, out, y, (int) ps.size(), in, x, rows, g_.s);
        return;
    }
    for (const Proj & p : ps) lin32(p.first, x, rows, p.second);
}

}  // namespace truss::blocks
