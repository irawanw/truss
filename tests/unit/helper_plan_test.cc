// runtime::helper_split / helper_fill (host, no GPU): the owner / helper placement of Phase 1. What the gate cares
// about, checked directly: the two sides partition the hot set (so an expert is resident on exactly one GPU), every
// layer's routed mass lands within 10% of evenly (an unbalanced layer is a layer whose round time does not improve),
// the fill stays inside the peer's reserved region, nests as the region grows, and never puts an expert on both GPUs.
// usage: helper_plan_test [usage.f32]  (default data/usage_strata_rank.f32, the real 48 x 512 profile)
#include "runtime/helper_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace truss::runtime;

namespace {

constexpr int E = 512;
int fails = 0, checks = 0;

void check(bool ok, const std::string & what)
{
    ++checks;
    if (!ok) { ++fails; std::printf("  FAIL %s\n", what.c_str()); }
}

// The hot set ExpertStore::plan produces with plan_alpha 0: the top `per_layer` experts of every layer by usage.
// Resident sizes imitate the trellis rates (1.0-4.0x of ~1.7 MB), deterministic in (layer, expert).
HotPlan make_plan(const std::vector<float> & usage, int L, int per_layer)
{
    HotPlan p;
    p.n_expert = E;
    p.alpha = 0.0;
    p.usage = usage;
    p.hot.assign(L, std::vector<uint8_t>(E, 0));
    p.bytes.assign((size_t) L * E, 0);
    std::vector<int> order(E);
    for (int l = 0; l < L; ++l) {
        for (int e = 0; e < E; ++e) order[e] = e;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return usage[(size_t) l * E + a] > usage[(size_t) l * E + b];
        });
        for (int j = 0; j < per_layer; ++j) p.hot[l][order[j]] = 1;
        std::mt19937 rng(1234u + (unsigned) l);
        for (int e = 0; e < E; ++e)
            p.bytes[(size_t) l * E + e] = (size_t) ((1.0 + rng() / (double) rng.max() * 3.0) * 1.7e6);
    }
    return p;
}

// owner and give partition the hot set exactly, and the returned totals agree with it
void check_partition(const HotPlan & p, const HelperSplit & s, const char * tag)
{
    long n_hot = 0, both = 0, neither = 0, misplaced = 0;
    size_t hot_bytes = 0, side_bytes = 0, mass = 0;
    for (size_t l = 0; l < p.hot.size(); ++l)
        for (int e = 0; e < E; ++e) {
            const bool h = p.hot[l][e], o = s.owner[l][e], g = s.give[l][e];
            n_hot += h;
            both += o && g;
            neither += h && !o && !g;
            misplaced += !h && (o || g);
            hot_bytes += h ? p.bytes[l * E + e] : 0;
            side_bytes += (o || g) ? p.bytes[l * E + e] : 0;
            if (h) mass += (long) p.usage[l * E + e];
        }
    check(both == 0, std::string(tag) + ": an expert is on both sides");
    check(neither == 0, std::string(tag) + ": a hot expert is on neither side");
    check(misplaced == 0, std::string(tag) + ": a cold expert was placed");
    check(s.owner_count + s.give_count == n_hot, std::string(tag) + ": expert count");
    check(s.owner_bytes + s.give_bytes == hot_bytes && side_bytes == hot_bytes, std::string(tag) + ": byte total");
    check(std::abs(s.owner_mass + s.give_mass - (double) mass) < 1e-3 * (double) mass,
          std::string(tag) + ": routed mass total");
}

// how far a layer's mass sits from what frac asks for, and the overall share
void check_balance(const HelperSplit & s, double tol, double frac, const char * tag)
{
    // a split at frac leaves the two sides |1 - 2*frac| apart by definition; the check is that no layer drifts from
    // that, which is what makes a layer's round time actually improve
    const double want = std::abs(1.0 - 2.0 * frac);
    check(std::abs(s.worst_imbalance - want) <= tol,
          std::string(tag) + ": worst per-layer imbalance " + std::to_string(100 * s.worst_imbalance) + "% against " +
              std::to_string(100 * want) + "% +- " + std::to_string(100 * tol) + "%");
    const double share = s.give_mass / (s.owner_mass + s.give_mass);
    check(std::abs(share - frac) < 0.01, std::string(tag) + ": helper mass share " + std::to_string(share) +
                                             " is not " + std::to_string(frac));
    std::printf("  %s: mass %.1f/%.1f, worst imbalance %.3f%% (ideal %.1f%%)\n", tag,
                100 * s.owner_mass / (s.owner_mass + s.give_mass), 100 * s.give_mass / (s.owner_mass + s.give_mass),
                100 * s.worst_imbalance, 100 * want);
}

}  // namespace

int main(int argc, char ** argv)
{
    const std::string path = argc > 1 ? argv[1] : "data/usage_strata_rank.f32";
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) { std::printf("helper_plan_test: cannot open %s\n", path.c_str()); return 1; }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f) / 4;
    std::fseek(f, 0, SEEK_SET);
    std::vector<float> usage;
    usage.resize((size_t) n);
    if (std::fread(usage.data(), 4, (size_t) n, f) != (size_t) n) { std::printf("helper_plan_test: short read\n"); return 1; }
    std::fclose(f);
    const int L = (int) n / E;
    std::printf("helper_plan_test: %ld values = %d layers x %d experts\n", n, L, E);

    constexpr int HOT_PER_LAYER = 300;               // what the served budget makes resident (~60% of a layer)
    const int La = L / 2, Lb = L - La;
    const HotPlan pa = make_plan(std::vector<float>(usage.begin(), usage.begin() + (size_t) La * E), La, HOT_PER_LAYER);
    const HotPlan pb = make_plan(std::vector<float>(usage.begin() + (size_t) La * E, usage.end()), Lb, HOT_PER_LAYER);

    for (double frac : {0.3, 0.5, 0.6}) {
        const HelperSplit sa = helper_split(pa, frac), sb = helper_split(pb, frac);
        char tag[64];
        std::snprintf(tag, sizeof tag, "frac %.1f stage A", frac);
        check_partition(pa, sa, tag);
        check_balance(sa, 0.02, frac, tag);
        std::snprintf(tag, sizeof tag, "frac %.1f stage B", frac);
        check_partition(pb, sb, tag);
        check_balance(sb, 0.02, frac, tag);
        // deterministic: the same plan splits the same way, so the load log is reproducible
        const HelperSplit again = helper_split(pa, frac);
        check(sa.owner == again.owner && sa.give == again.give && sa.give_ranked == again.give_ranked,
              "the split is not deterministic");
        if (frac != 0.5) continue;

        // Each GPU's region is its own give-away. Filling A's region from B's give-away, and the other way round,
        // must keep every expert on exactly one GPU and keep the resident count near the unsplit plan.
        size_t used_a = 0, used_b = 0;
        long drop_a = 0, drop_b = 0;
        const auto fill_a = helper_fill(sb.give_ranked, pb.bytes, E, sa.give_bytes, &used_a, &drop_a);
        const auto fill_b = helper_fill(sa.give_ranked, pa.bytes, E, sb.give_bytes, &used_b, &drop_b);
        check(used_a <= sa.give_bytes && used_b <= sb.give_bytes, "a helper region was overrun");
        for (const auto & it : fill_a) check(!sb.owner[it.first][it.second], "stage B: an expert is on both GPUs");
        for (const auto & it : fill_b) check(!sa.owner[it.first][it.second], "stage A: an expert is on both GPUs");
        long given = 0;
        for (const auto & it : fill_a) given += sb.give[it.first][it.second];
        check(given == (long) fill_a.size(), "the fill took an expert that was not given away");
        check(drop_a == sb.give_count - (long) fill_a.size(), "the dropped count disagrees with the fill");

        const long unsplit = sa.owner_count + sa.give_count + sb.owner_count + sb.give_count;
        const long resident = sa.owner_count + (long) fill_a.size() + sb.owner_count + (long) fill_b.size();
        std::printf("  frac 0.5: A %ld own + %ld helper, B %ld own + %ld helper = %ld resident against %ld unsplit "
                    "(%+.2f%%), worst imbalance %.2f%% / %.2f%%\n",
                    sa.owner_count, (long) fill_a.size(), sb.owner_count, (long) fill_b.size(), resident, unsplit,
                    100.0 * (resident - unsplit) / (double) unsplit, 100 * sa.worst_imbalance, 100 * sb.worst_imbalance);
        check(resident <= unsplit, "more experts resident than the unsplit plan (nothing was freed for the helper set)");
        check(resident >= unsplit * 0.98, "more than 2% of the resident experts were lost to the split");
        // the fill nests: nothing fits an empty region, the full region holds the whole give-away, and a smaller
        // region only ever takes a subset of what a bigger one does
        check(helper_fill(sb.give_ranked, pb.bytes, E, 0).empty(), "a zero-size region still filled");
        long drop_full = -1;
        const auto full = helper_fill(sb.give_ranked, pb.bytes, E, sb.give_bytes, nullptr, &drop_full);
        check(full.size() == (size_t) sb.give_count && drop_full == 0,
              "the full region did not hold the whole give-away");
        const auto half = helper_fill(sb.give_ranked, pb.bytes, E, sb.give_bytes / 2);
        long nested = 0;
        for (const auto & it : half) nested += std::find(full.begin(), full.end(), it) != full.end();
        check(nested == (long) half.size(), "the half-size region took an expert the full region left out");
    }
    std::printf("helper_plan_test: %d checks, %d failed  %s\n", checks, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
