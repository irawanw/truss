#include "engine/options.h"

#include <cstdlib>
#include <string>
#include <type_traits>

namespace truss::engine {

void apply_env(Options & o)
{
    auto f = [](const char * k, auto & v) {
        if (const char * e = std::getenv(k)) v = (std::remove_reference_t<decltype(v)>) std::atof(e);
    };
    runtime::MoeOptions & m = o.moe;
    f("TRUSS_CPU_SHARE", m.cpu_share);
    f("TRUSS_CPU_THREADS", m.cpu_threads);
    f("TRUSS_CPU_DYNAMIC", m.cpu_dynamic);     // Strata's split
    f("TRUSS_CPU_TRELLIS", m.cpu_trellis);     // CPU tier from the pack's own bytes
    f("TRUSS_PCIE_GBPS", m.pcie_gbps);
    f("TRUSS_PCIE_FRAC", m.pcie_frac);         // fixed PCIe share of the misses
    f("TRUSS_ADAPT_EVERY", m.adapt_every);     // adaptive tier
    f("TRUSS_ADAPT_SWAPS", m.adapt_swaps);
    f("TRUSS_ADAPT_MIN", m.adapt_min);
    f("TRUSS_ADAPT_DECAY", m.adapt_decay);
    f("TRUSS_HINT_K", m.hint_k);               // pre-gated prefetch width
    f("TRUSS_HINT_MIN_ROWS", m.hint_min_rows);
    f("TRUSS_TAIL_SKIP", m.tail_skip);         // route tail skip rank (0 off; CPU tier decode only, #192)
    f("TRUSS_SKIP_WEIGHT", m.skip_weight);     // weight-share skip: missed expert under this share of its row's K weights, every row (PolyStrata #34, D10)
    f("TRUSS_ADMIT_IDLE", m.admit_idle);       // admission on the idle link, experts per pass
    f("TRUSS_ADMIT_MIN", m.admit_min);
    f("TRUSS_ADMIT_DECAY", m.admit_decay);
    f("TRUSS_PLAN_ALPHA", m.plan_alpha);       // hot-set ranking: usage / bytes^alpha
    f("TRUSS_SPLIT_ROWS", m.split_rows);       // E7: short prompt chunks split cold experts CPU / PCIe
    f("TRUSS_SPLIT_T", m.split_t);
    f("TRUSS_SPLIT_MS_ROW", m.split_ms_row);
    f("TRUSS_KV_LEND", o.kv_lend);             // E6
    f("TRUSS_PREFILL_ROWS", o.prefill_rows);
    f("TRUSS_KV_INT8", o.kv_int8);             // Strata's int8 KV
    if (const char * e = std::getenv("TRUSS_ACT"))   // dense activations: fp16 (W8A16, KL 0.040) or else q8_1 (0.065)
        o.act = std::string(e) == "fp16" ? blocks::Activations::FP16 : blocks::Activations::Q8_1;
    f("TRUSS_LAYER_SPLIT", o.layer_split);             // two devices: layers [0, N) on TRUSS_LAYER_SPLIT_DEVICE
    f("TRUSS_LAYER_SPLIT_DEVICE", o.layer_split_device);
    f("TRUSS_LAYER_SPLIT_FRAC", o.layer_split_frac);
    f("TRUSS_SLOTS", o.slots);                 // sequence slots
    if (const char * e = std::getenv("TRUSS_PLE_DIRECT"); e && !std::atoi(e)) o.direct_tables = false;   // A/B
    if (const char * e = std::getenv("TRUSS_RING_GB")) o.ring_bytes_override = (size_t) (std::atof(e) * (1ull << 30));
    // a fixed VRAM budget for the routed experts instead of "all free memory": a test beside another GPU job (#118)
    if (const char * e = std::getenv("TRUSS_EXPERT_GB")) m.expert_budget = (size_t) (std::atof(e) * (1ull << 30));
    // E7 (E6 option c, D8): GiB of the ring kept as a FIFO cache; the rest is refilled with the plan's next cold
    // experts as pinned residents after a prompt. Unset = 0 = off: the ring is FIFO-only, byte-identical to before.
    if (const char * e = std::getenv("TRUSS_PIN_FILL")) m.pin_fill = (size_t) (std::atof(e) * (1ull << 30));
}

}  // namespace truss::engine
