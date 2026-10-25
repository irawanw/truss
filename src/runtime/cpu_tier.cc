// MoeExec's CPU tier loaders: which cold experts of each store layer the host computes, and their copies.
//   q4s     (Options::cpu_dir): 4-bit copies from cpu_dir/L<nn>.q4s (flashnext_truss_cpu_q4.py), the rarest
//           non-resident experts up to cpu_share of the layer's non-resident routing mass
//   trellis (Options::cpu_trellis): the pack's own trellis bytes, from the ExpertStore's pinned cold copy (no extra
//           RAM; TRACKER #73); static set as above, or every cold expert with cpu_dynamic
// The trellis tier takes any expert shape (the pool is sized from MoeKernels); the q4s copies exist for the Flash-Next
// shape (cpu::D_MODEL / D_FF) only and fail loudly for another.
#include "runtime/moe_exec_state.h"

#include "engine/limits.h"

#include <algorithm>
#include <numeric>

namespace truss::runtime {

using engine::require;

void MoeExec::load_cpu_tier_trellis(const ExpertStore::HotSet & hot, const std::vector<float> & usage)
{
    require(!usage.empty() || o_.cpu_dynamic, "the static CPU tier needs expert_usage (it picks the rarest)");
    auto t = std::make_unique<CpuTier>();
    t->trellis = true;
    const int E = E_;
    t->is_cpu.assign(n_store(), std::vector<uint8_t>(E, 0));
    t->tview.resize(n_store());
    size_t n_cpu = 0, n_layers = 0;
    for (int l = 0; l < n_store(); ++l) {
        if (!layers_[l].cpu_ok) continue;
        ++n_layers;
        const ExpertLayer & tab = layers_[l].tables;
        t->tview[l].resize(E);
        std::vector<int> cold;
        double mass = 0;
        for (int e = 0; e < E; ++e)
            if (!hot[l][e] && !store_->is_far(l, e)) {   // far: no pinned bytes for the CPU to read
                cold.push_back(e);
                if (!usage.empty()) mass += usage[(size_t) l * E + e];
                cpu::TrellisMat m[3];
                for (int p = 0; p < 3; ++p) {
                    const formats::ExpertTable & T = *tab[p];
                    require((T.k[e] >= 1 && T.k[e] <= 6) || T.k[e] == 25 || T.k[e] == 35,
                            "CPU tier: trellis K out of 1..6, 25, 35");
                    m[p] = { reinterpret_cast<const uint32_t *>(store_->host_part(l, e, p)), T.k[e], (int) T.in,
                             (int) T.out, reinterpret_cast<const uint16_t *>(T.suh->data) + (size_t) e * T.in,
                             reinterpret_cast<const uint16_t *>(T.svh->data) + (size_t) e * T.out };
                }
                t->tview[l][e] = { m[0], m[1], m[2] };
            }
        if (o_.cpu_dynamic) {
            for (int e : cold) t->is_cpu[l][e] = 1;
            n_cpu += cold.size();
            continue;
        }
        std::stable_sort(cold.begin(), cold.end(),
                         [&](int a, int b) { return usage[(size_t) l * E + a] < usage[(size_t) l * E + b]; });
        double acc = 0;
        for (int e : cold) {
            const double u = usage[(size_t) l * E + e];
            if (acc + u > o_.cpu_share * mass) break;
            acc += u;
            t->is_cpu[l][e] = 1;
            ++n_cpu;
        }
    }
    std::fprintf(stderr, "CPU tier (trellis, pinned pack bytes): %zu experts eligible (%.1f per layer), %s\n", n_cpu,
                 (double) n_cpu / std::max<size_t>(1, n_layers), o_.cpu_dynamic ? "dynamic split" : "static set");
    const size_t rows = std::max({ engine::FETCH_ROWS, cpu::MAX_ROWS, o_.split_rows });   // E7: a split chunk's rows
    TRUSS_CUDA(cudaMallocHost(&t->x, sizeof(float) * rows * k_.d_model));
    TRUSS_CUDA(cudaMallocHost(&t->w, sizeof(float) * rows * K_));
    TRUSS_CUDA(cudaMallocHost(&t->y, sizeof(float) * rows * k_.d_model));
    TRUSS_CUDA(cudaMallocHost(&t->ids, sizeof(int) * rows * K_));
    cpu_tier_ = std::move(t);
}

// With cpu_dir/L<nn>.ids (int32 expert ids in file order) the file holds only those experts (an NVMe subset of the
// full archive, flashnext_truss_cpu_subset.py) and the CPU set is chosen among them.
void MoeExec::load_cpu_tier(const ExpertStore::HotSet & hot, const std::vector<float> & usage)
{
    require(!usage.empty(), "the CPU tier needs expert_usage (it picks the rarest experts)");
    require(k_.d_model == cpu::D_MODEL && k_.d_ff == cpu::D_FF, "CPU tier shape");
    auto t = std::make_unique<CpuTier>();
    const int E = E_, NS = n_store();
    t->is_cpu.assign(NS, std::vector<uint8_t>(E, 0));
    t->data.resize(NS);
    t->fscales.resize(NS);
    t->view.resize(NS);
    std::vector<std::vector<int>> pick_of(NS);
    std::vector<std::vector<int64_t>> slot_of(NS);
    auto base_of = [&](int l) {
        const int ml = layers_[l].model_layer;
        return o_.cpu_dir + "/L" + (ml < 10 ? "0" : "") + std::to_string(ml);
    };
    for (int l = 0; l < NS; ++l) {
        if (!layers_[l].cpu_ok) continue;
        const std::string base = base_of(l);
        std::vector<int64_t> & sl = slot_of[l];
        sl.assign(E, -1);   // expert -> position in the q4s file
        if (FILE * fi = std::fopen((base + ".ids").c_str(), "rb")) {
            int32_t e;
            for (int64_t k = 0; std::fread(&e, 4, 1, fi) == 1; ++k) {
                require(e >= 0 && e < E, base + ".ids: expert id out of range");
                sl[e] = k;
            }
            std::fclose(fi);
        } else {
            for (int e = 0; e < E; ++e) sl[e] = e;
        }
        std::vector<int> cold;
        double mass = 0;
        for (int e = 0; e < E; ++e)
            if (!hot[l][e]) {
                mass += usage[(size_t) l * E + e];
                if (sl[e] >= 0) cold.push_back(e);
            }
        std::stable_sort(cold.begin(), cold.end(),
                         [&](int a, int b) { return usage[(size_t) l * E + a] < usage[(size_t) l * E + b]; });
        std::vector<int> pick;
        double acc = 0;
        for (int e : cold) {
            const double u = usage[(size_t) l * E + e];
            if (acc + u > o_.cpu_share * mass) break;
            acc += u;
            pick.push_back(e);
        }
        std::sort(pick.begin(), pick.end());
        pick_of[l] = pick;
        t->data[l].resize(pick.size() * cpu::EXPERT_BYTES);
        t->fscales[l].resize(pick.size() * cpu::SCALES_PER_EXPERT);
        t->view[l].resize(E);
    }
    // Read the experts in .q4s slot order, not pick order: the file is ordered rarest-first, so pick order scatters
    // into independent random 2.7 MB reads (~18 min for ~15000 of them). In slot order the reads run forward, and the
    // layers are independent, so they go in parallel too (TRACKER #71).
    const int nth = std::max(1, std::min(NS, (int) std::thread::hardware_concurrency()));
    std::vector<std::thread> workers;
    std::mutex mu;
    std::string err;
    std::atomic<size_t> bytes{ 0 };
    for (int w = 0; w < nth; ++w)
        workers.emplace_back([&, w] {
            for (int l = w; l < NS; l += nth) {
                if (!layers_[l].cpu_ok) continue;
                const std::vector<int> & pick = pick_of[l];
                const std::vector<int64_t> & sl = slot_of[l];
                const std::string path = base_of(l) + ".q4s";
                FILE * f = std::fopen(path.c_str(), "rb");
                if (!f) {
                    std::lock_guard<std::mutex> g(mu);
                    err = "cannot open " + path;
                    return;
                }
                std::vector<int> order(pick.size());
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(), [&](int a, int b) { return sl[pick[a]] < sl[pick[b]]; });
                for (int i : order) {
                    uint8_t * dst = t->data[l].data() + (size_t) i * cpu::EXPERT_BYTES;
                    const bool ok = std::fseek(f, (long) ((size_t) sl[pick[i]] * cpu::EXPERT_BYTES), SEEK_SET) == 0 &&
                                    std::fread(dst, 1, cpu::EXPERT_BYTES, f) == cpu::EXPERT_BYTES;
                    if (!ok) {
                        std::lock_guard<std::mutex> g(mu);
                        err = "short read of expert " + std::to_string(pick[i]) + " from " + path;
                        std::fclose(f);
                        return;
                    }
                    t->view[l][pick[i]] = cpu::expert_view(dst);
                    cpu::preconvert_scales(t->view[l][pick[i]], t->fscales[l].data() + (size_t) i * cpu::SCALES_PER_EXPERT);
                    t->is_cpu[l][pick[i]] = 1;
                }
                std::fclose(f);
                bytes += t->data[l].size();
            }
        });
    for (auto & th : workers) th.join();
    require(err.empty(), err);
    t->bytes = bytes;
    size_t n_cpu = 0, n_layers = 0;
    for (int l = 0; l < NS; ++l) n_cpu += pick_of[l].size(), n_layers += layers_[l].cpu_ok;
    std::fprintf(stderr, "CPU tier: %zu experts (%.1f per layer), %.2f GB of q4s in RAM\n", n_cpu,
                 (double) n_cpu / std::max<size_t>(1, n_layers), t->bytes / 1e9);
    const size_t rows = std::max({ engine::FETCH_ROWS, cpu::MAX_ROWS, o_.split_rows });   // E7: a split chunk's rows
    TRUSS_CUDA(cudaMallocHost(&t->x, sizeof(float) * rows * k_.d_model));
    TRUSS_CUDA(cudaMallocHost(&t->w, sizeof(float) * rows * K_));
    TRUSS_CUDA(cudaMallocHost(&t->y, sizeof(float) * rows * k_.d_model));
    TRUSS_CUDA(cudaMallocHost(&t->ids, sizeof(int) * rows * K_));
    cpu_tier_ = std::move(t);
}

}  // namespace truss::runtime
