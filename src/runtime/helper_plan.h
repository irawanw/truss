// Helper-expert placement (PLAN Phase 1): every GPU keeps about half of every layer's hot experts and holds about
// half of the OTHER stage's, so that in a decode round the idle GPU can compute the other stage's share of its
// layers. Pure host code — no CUDA, no model — so tests/unit/helper_plan_test.cu runs it on any machine.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace truss::runtime {

// One stage's hot plan, as ExpertStore::plan produced it, plus what the split needs in order to rank and cost it.
struct HotPlan {
    int n_expert = 0;
    double alpha = 0.0;                          // the plan's own score exponent (Sizes::plan_alpha)
    std::vector<std::vector<uint8_t>> hot;       // [store layer][expert]: resident on this GPU
    std::vector<float> usage;                    // [layer * n_expert], ExpertStore::plan's layout
    std::vector<size_t> bytes;                   // [layer * n_expert] resident bytes of (layer, expert)
};

// The owner / helper split of one stage's hot set. owner | give == hot and the two are disjoint.
struct HelperSplit {
    std::vector<std::vector<uint8_t>> owner, give;               // [store layer][expert]
    std::vector<std::pair<int, int>> give_ranked;                // give, best score first: the peer's fill order
    size_t owner_bytes = 0, give_bytes = 0;
    long owner_count = 0, give_count = 0;
    double owner_mass = 0, give_mass = 0;                        // routed usage of each side, all layers
    double worst_imbalance = 0;                                  // max over layers |m_o - m_g| / (m_o + m_g)
};

// Split one stage's hot set. Per layer: rank its hot experts by the plan's own score (usage / bytes^alpha,
// descending, stable in expert order) and walk them; an expert goes to the helper while the helper's routed mass is
// below its share, m_g * (1 - frac) < m_o * frac. With frac 0.5 that is alternation plus a mass correction: plain
// alternation hands rank 0 to the owner and never looks back, so it drifts whenever a layer's usage is skewed.
// Ranking per layer, and balancing per layer, is the point: both GPUs work on every layer, so a layer whose mass
// sits on one side is a layer whose round time does not improve.
inline HelperSplit helper_split(const HotPlan & p, double frac)
{
    HelperSplit s;
    const int L = (int) p.hot.size(), E = p.n_expert;
    const double f = std::min(0.95, std::max(0.0, frac));
    s.owner.assign(L, std::vector<uint8_t>(E, 0));
    s.give.assign(L, std::vector<uint8_t>(E, 0));
    std::vector<int> order;
    auto score = [&](int l, int e) {
        const size_t b = p.bytes[(size_t) l * E + e];
        return (double) p.usage[(size_t) l * E + e] / std::pow((double) (b ? b : 1), p.alpha);
    };
    for (int l = 0; l < L; ++l) {
        order.clear();
        for (int e = 0; e < E; ++e)
            if (p.hot[l][e]) order.push_back(e);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score(l, a) > score(l, b); });
        double mo = 0, mg = 0;
        for (int e : order) {
            const size_t b = p.bytes[(size_t) l * E + e];
            const double u = p.usage[(size_t) l * E + e];
            const bool to_helper = mg * (1.0 - f) < mo * f;
            (to_helper ? s.give[l][e] : s.owner[l][e]) = 1;
            (to_helper ? mg : mo) += u;
            (to_helper ? s.give_mass : s.owner_mass) += u;
            (to_helper ? s.give_bytes : s.owner_bytes) += b;
            ++(to_helper ? s.give_count : s.owner_count);
        }
        if (mo + mg > 0) s.worst_imbalance = std::max(s.worst_imbalance, std::abs(mo - mg) / (mo + mg));
    }
    // The fill order across layers is one global score order: a fill that runs out of room drops its tail, which is
    // then the least-routed experts overall (they stay cold on both GPUs, as any other cold expert).
    s.give_ranked.reserve((size_t) s.give_count);
    for (int l = 0; l < L; ++l)
        for (int e = 0; e < E; ++e)
            if (s.give[l][e]) s.give_ranked.push_back({ l, e });
    std::stable_sort(s.give_ranked.begin(), s.give_ranked.end(), [&](const std::pair<int, int> & a,
                                                                     const std::pair<int, int> & b) {
        return score(a.first, a.second) > score(b.first, b.second);
    });
    return s;
}

// The experts of one stage's give-away that fit the peer's reserved region: first fit in fill order, so the ones
// left out are the least-routed that did not fit. They stay resident nowhere, so the owner fetches them over PCIe
// / admits them to its ring exactly as it does any other cold expert. `bytes` is the giving stage's table.
inline std::vector<std::pair<int, int>> helper_fill(const std::vector<std::pair<int, int>> & give_ranked,
                                                    const std::vector<size_t> & bytes, int n_expert, size_t budget,
                                                    size_t * used = nullptr, long * dropped = nullptr)
{
    std::vector<std::pair<int, int>> out;
    size_t b = 0;
    long drop = 0;
    for (const auto & it : give_ranked) {
        const size_t sz = bytes[(size_t) it.first * n_expert + it.second];
        if (b + sz > budget) { ++drop; continue; }   // keep walking: a later expert may still fit
        b += sz;
        out.push_back(it);
    }
    if (used) *used = b;
    if (dropped) *dropped = drop;
    return out;
}

}  // namespace truss::runtime
