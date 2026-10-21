// ExpertStore sizing and the hot-set plan (static members): how many experts of which layers stay resident in a
// VRAM budget, and the usage file reader. Also the background page-locking of the cold copy (start/wait_pinned).
#include "runtime/expert_store.h"
#include "runtime/expert_store_util.h"

#include "core/cuda_check.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <thread>
#include <sys/mman.h>

namespace truss::runtime {

using namespace store_detail;

size_t ExpertStore::ring_size(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z)
{
    size_t slot = 0;
    for (size_t l = 0; l < layers.size(); ++l) {
        size_t cold = 0;
        for (int e = 0; e < layers[l][0]->n_expert; ++e)
            if (!hot[l][e]) cold += expert_bytes(layers[l], e);
        slot = std::max(slot, align_up(cold));
    }
    return align_up(std::max(z.ring_bytes, 2 * slot + z.stream_extra));
}

size_t ExpertStore::device_bytes(const std::vector<ExpertLayer> & layers, const HotSet & hot, const Sizes & z)
{
    size_t total = ring_size(layers, hot, z);
    for (size_t l = 0; l < layers.size(); ++l) {
        total += fixed_bytes(layers[l]);
        for (int e = 0; e < layers[l][0]->n_expert; ++e)
            if (hot[l][e]) total += expert_bytes(layers[l], e);
    }
    return total;
}

ExpertStore::HotSet ExpertStore::plan(const std::vector<ExpertLayer> & layers, size_t budget, const Sizes & z,
                                      const std::vector<float> & usage)
{
    const int L = (int) layers.size(), n_expert = layers.empty() ? 0 : layers[0][0]->n_expert;
    HotSet hot(L, std::vector<uint8_t>(n_expert, 0));
    if (device_bytes(layers, hot, z) > budget)
        throw std::runtime_error("ExpertStore::plan: " + std::to_string(budget >> 20) + " MiB cannot hold even the ring (" +
                                 std::to_string(device_bytes(layers, hot, z) >> 20) + " MiB)");
    if (usage.empty()) {   // the same count in every layer, index order
        for (int n = 0; n < n_expert; ++n) {
            for (auto & h : hot) h[n] = 1;
            if (device_bytes(layers, hot, z) > budget) {
                for (auto & h : hot) h[n] = 0;
                break;
            }
        }
        return hot;
    }
    if (usage.size() != (size_t) L * n_expert)
        throw std::invalid_argument("ExpertStore::plan: usage has " + std::to_string(usage.size()) + " values, expected " +
                                    std::to_string((size_t) L * n_expert));
    size_t fixed = 0;
    std::vector<size_t> size(L * (size_t) n_expert), cold_all(L);
    for (int l = 0; l < L; ++l) {
        fixed += fixed_bytes(layers[l]);
        for (int e = 0; e < n_expert; ++e) {
            size[(size_t) l * n_expert + e] = expert_bytes(layers[l], e);
            cold_all[l] += size[(size_t) l * n_expert + e];
        }
    }
    std::vector<double> score(size.size());
    for (size_t i = 0; i < size.size(); ++i) score[i] = usage[i] / std::pow((double) size[i], z.plan_alpha);
    auto better = [&](int a, int b) { return score[a] > score[b]; };
    std::vector<int> order(size.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int) i;
    std::stable_sort(order.begin(), order.end(), better);
    std::vector<std::vector<int>> by_layer(L);
    for (int i : order) by_layer[i / n_expert].push_back(i);

    auto fill = [&](int floor, HotSet & h) -> double {   // the hot usage, or -1 if the floor does not fit
        std::vector<size_t> cold = cold_all;
        for (auto & x : h) std::fill(x.begin(), x.end(), 0);
        auto total = [&](size_t hot_bytes) {
            size_t slot = 0;
            for (int l = 0; l < L; ++l) slot = std::max(slot, align_up(cold[l]));
            return fixed + hot_bytes + align_up(std::max(z.ring_bytes, 2 * slot + z.stream_extra));
        };
        size_t hot_bytes = 0;
        double used = 0;
        auto take = [&](int i) {
            cold[i / n_expert] -= size[i];
            hot_bytes += size[i];
            h[i / n_expert][i % n_expert] = 1;
            used += usage[i];
        };
        for (int l = 0; l < L; ++l)
            for (int j = 0; j < floor; ++j) take(by_layer[l][j]);
        if (total(hot_bytes) > budget) return -1;
        for (int i : order) {
            const int l = i / n_expert;
            if (h[l][i % n_expert]) continue;
            cold[l] -= size[i];
            const bool fits = total(hot_bytes + size[i]) <= budget;
            cold[l] += size[i];
            if (fits) take(i);
        }
        return used;
    };
    double best = -1;
    HotSet trial = hot;
    for (int floor = 0; floor <= n_expert; floor += 8) {
        const double used = fill(floor, trial);
        if (used < 0) break;
        if (used > best) best = used, hot = trial;
    }
    return hot;
}

// E7: the cold experts in plan()'s order: usage / bytes^alpha, descending; index order without a usage profile.
std::vector<std::pair<int, int>> ExpertStore::fill_order(const std::vector<ExpertLayer> & layers, const HotSet & hot,
                                                         const std::vector<float> & usage, double alpha)
{
    const int L = (int) layers.size(), n_expert = layers.empty() ? 0 : layers[0][0]->n_expert;
    if (usage.size() != 0 && usage.size() != (size_t) L * n_expert)
        throw std::invalid_argument("ExpertStore::fill_order: usage has " + std::to_string(usage.size()) +
                                    " values, expected " + std::to_string((size_t) L * n_expert));
    std::vector<int> all(size_t(L) * n_expert);
    for (size_t i = 0; i < all.size(); ++i) all[i] = (int) i;
    if (!usage.empty()) {
        std::vector<double> score(all.size());
        for (int i : all) score[i] = usage[i] / std::pow((double) expert_bytes(layers[i / n_expert], i % n_expert), alpha);
        std::stable_sort(all.begin(), all.end(), [&](int a, int b) { return score[a] > score[b]; });
    }
    std::vector<std::pair<int, int>> out;
    out.reserve(all.size());
    for (int i : all)
        if (!hot[i / n_expert][i % n_expert]) out.push_back({ i / n_expert, i % n_expert });
    return out;
}

std::vector<float> ExpertStore::load_usage(const std::string & path, int n_layer, int n_expert)
{
    std::vector<float> u((size_t) n_layer * n_expert);
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("ExpertStore::load_usage: cannot open " + path);
    const size_t got = std::fread(u.data(), sizeof(float), u.size(), f);
    const bool extra = std::fgetc(f) != EOF;
    std::fclose(f);
    if (got != u.size() || extra)
        throw std::runtime_error("ExpertStore::load_usage: " + path + " is not " + std::to_string(n_layer) + " x " +
                                 std::to_string(n_expert) + " float32");
    return u;
}

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
            // a kept copy (cold_keep.h) is filled but has no page tables in this process: populating them first,
            // outside the driver lock, took the register of 61 GB 17.9 -> 7.5 s, which a first request waits behind
            if (kept_) madvise(const_cast<uint8_t *>(Y.host), align_up(Y.cold, 4096), 23 /* MADV_POPULATE_WRITE */);
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

void ExpertStore::wait_pinned()
{
    start_pinning();
    if (reg_thread_.joinable()) reg_thread_.join();
}

}  // namespace truss::runtime
