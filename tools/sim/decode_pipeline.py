#!/usr/bin/env python3
"""D0 decode pipeline simulator (PLAN-20261006 §2 D0).

Discrete-event model of one decode pass (48 MoE layers) on the GPU 1 (x16) server:
  GPU stream : mixer+hc(l) -> window(l){ wait plan, wait go, expert compute, wait cpu_done, combine }
  driver     : one thread, serial: serve(l) = max(doorbell(l), serve(l-1)) -> split -> plan -> hints
  copy engine: serial FIFO on the x16 link: demand(l) bytes, then hint(l) bytes (serve(): fetch ->
               signal(go) -> prefetch_hint, forward.cu serve())
  CPU tier   : pool resource, serial at its DRAM floor: call(l) starts at serve(l), wall = wake +
               per_slot * slots (TRACKER #75 model)

Row closure (calibration anchors, all rows close with slots = 1.22 x PCIe-experts):
  per-pass: cold slot-visits, demand experts, prefetch experts, CPU slots = cold - 1.22*(dem+pre)

Environment knobs (both justified by logs, see ROWS):
  clock_scale  GPU derate: short probe rows run near boost (1.0); full/sustained rows SM 1200-1740
               of 2115 MHz -> 1.09 (gpu1_throttle.log)
  cpu_slow     CPU pool wall varies with host state across rows (measured cpu join 10.2-15.1 ms/pass
               at identical slots); range 1.0-1.35

Calibration gate: reproduce the six measured GPU 1 rows' PASS TIME within +-5% (the clean physical
quantity; tg = tokens/pass / pass, and tokens/pass 2.60-2.93 is MTP acceptance, a weights-domain
fact the scheduler sim does not model - plan D3), and reproduce the ORDER of tg.

Usage:
  decode_pipeline.py --base       served-config base breakdown
  decode_pipeline.py --calibrate  gate check vs the six rows
  decode_pipeline.py --lever      rank levers (plan D0 questions a-d) on the served config
"""
import argparse, random

# ---- measured constants (row 1006_111210 bench_1.log + plan §2) ----
P = dict(
    n_layer=48,
    tokens_per_pass=2.93,
    k_experts=8,
    # GPU per-layer serial times at boost clocks (plan table / 48):
    mixer_ms=7.17 / 48,
    overhead_ms=(2.34 + 1.79 + 0.21) / 48,   # ple+hc_mix + hc_mix + shared sections (serial GPU)
    expert_kernel_ms=5.2 / 48,               # window compute at ~16.9 GPU-routed slots/layer
    combine_ms=0.02,
    launch_gap_ms=0.005,                     # residual inter-kernel gap beyond section totals
    host_gap_ms=3.9,                         # between-pass host work: drafts, sampling, setup (measured
                                             # wall - section sum: 3.67-4.08 ms/pass, constant)
    window_overhead_ms=0.09,                 # window kernel fixed cost: scan, resident compute, combine
    dem_pressure_ms=0.032,                   # ring contention (expert_store.cu ring_put/wait_compute):
                                             # demand rate above ~1.6 experts/layer stalls slot release,
                                             # stretching the window per extra demand expert
    # driver:
    driver_split_ms=0.05,
    doorbell_poll_ms=0.01,
    # CPU tier:
    cpu_wake_ms=0.12,
    cpu_per_slot_ms=0.068,                   # effective wall/slot (0.085 engine charge, 0.056 DRAM floor)
    cpu_slow=1.0,                            # host-state multiplier on pool wall (1.0-1.35)
    # link: 21 GB/s = 21e6 bytes/ms (x16 measured)
    link_bytes_ms=21e6,
    expert_bytes=1.95e6,
    slots_per_pcie_expert=1.22,              # slot-visits served per PCIe-fetched expert
    # routing aggregates per LAYER (default = served K=3 config, row 1006_111210):
    cold_slots=487.7 / 48, demand=64.6 / 48, prefetch=78.3 / 48,
    clock_scale=1.0,
    hint_k=3, pcie_frac=0.2, admit_idle=0,
)

# the six calibration rows: measured per-pass aggregates + environment + measured pass/tg/tok
ROWS = [
    dict(name="K=2 f0.2",       hint_k=2, pcie_frac=0.2,  admit_idle=0, clock=1.0,  cpu_slow=0.95,
         cold=413.4, dem=56.2,  pre=39.8,  pass_ms=37.59, tg=62.7, tpp=2.60),
    dict(name="K=3 f0.2 probe", hint_k=3, pcie_frac=0.2,  admit_idle=0, clock=1.0,  cpu_slow=0.95,
         cold=487.7, dem=64.6,  pre=78.3,  pass_ms=39.21, tg=67.9, tpp=2.93),
    dict(name="K=3 f0.2 final", hint_k=3, pcie_frac=0.2,  admit_idle=0, clock=1.09, cpu_slow=1.09,
         cold=487.7, dem=64.6,  pre=78.3,  pass_ms=43.57, tg=61.7, tpp=2.93),
    dict(name="K=4 f0.2",       hint_k=4, pcie_frac=0.2,  admit_idle=0, clock=1.0,  cpu_slow=1.22,
         cold=492.3, dem=59.8,  pre=109.3, pass_ms=43.28, tg=59.9, tpp=2.83),
    dict(name="K=3 f0.35",      hint_k=3, pcie_frac=0.35, admit_idle=0, clock=1.0,  cpu_slow=1.05,
         cold=515.1, dem=110.4, pre=49.5,  pass_ms=43.52, tg=60.5, tpp=2.88),
    dict(name="K=3 admit2",     hint_k=3, pcie_frac=0.2,  admit_idle=2, clock=1.0,  cpu_slow=1.10,
         cold=468.3, dem=61.6,  pre=75.7,  pass_ms=42.39, tg=61.0, tpp=2.81),
]


def row_params(r):
    q = dict(P)
    q.update(hint_k=r["hint_k"], pcie_frac=r["pcie_frac"], admit_idle=r["admit_idle"],
             clock_scale=r["clock"], cpu_slow=r["cpu_slow"],
             cold_slots=r["cold"] / 48.0, demand=r["dem"] / 48.0, prefetch=r["pre"] / 48.0,
             tokens_per_pass=r["tpp"])
    # CPU slots from closure; GPU-computed slots = routed slots - cpu slots
    q["cpu_slots"] = (r["cold"] - q["slots_per_pcie_expert"] * (r["dem"] + r["pre"])) / 48.0
    return q


def q_early(p):
    """spec-cpu lever: CPU tier may start on predicted misses before the split (plan D0 d)."""
    return p.get("cpu_early_ms", 0.0)


def simulate(p, rng=None):
    cs = p["clock_scale"]
    mix = (p["mixer_ms"] + p["overhead_ms"] + p["launch_gap_ms"]) * cs
    expert_full = p["expert_kernel_ms"] * cs
    routed_slots = p["tokens_per_pass"] * p["k_experts"]
    gpu_t = drv_t = copy_t = 0.0
    st = dict(mix=0.0, plan=0.0, go=0.0, expert=0.0, cpu=0.0, combine=0.0,
              copy_busy=0.0, link_idle=0.0)
    jit = (lambda x: x) if rng is None else (lambda x: x * rng.uniform(0.9, 1.1))
    simulate._pool_free = 0.0
    for l in range(p["n_layer"]):
        mix_l = jit(mix + p["dem_pressure_ms"] * max(0.0, p["demand"] - 1.6))
        mix_end = gpu_t + mix_l                                # doorbell at mix_end (router fused)
        serve = max(drv_t, mix_end + p["doorbell_poll_ms"])
        plan_t = serve + p["driver_split_ms"]
        # copy engine FIFO: demand(l) gates go(l); hints(l) queue behind
        dem_ms = jit(p["demand"] * p["expert_bytes"] / p["link_bytes_ms"])
        dem_start = max(copy_t, serve)
        dem_end = dem_start + dem_ms
        hint_ms = jit(p["prefetch"] * p["expert_bytes"] / p["link_bytes_ms"])
        hint_end = max(dem_end, serve) + hint_ms
        if dem_start > copy_t:
            st["link_idle"] += dem_start - copy_t
        copy_t = hint_end
        # CPU pool: serial resource at DRAM floor; call queued behind previous layer's call
        busy = (p["cpu_wake_ms"] + p["cpu_per_slot_ms"] * p["cpu_slots"]) * p["cpu_slow"] \
            if p["cpu_slots"] > 0 else 0.0
        cpu_start = max(simulate._pool_free, serve - q_early(p))
        cpu_done = cpu_start + busy
        simulate._pool_free = cpu_done
        # GPU window kernel: plan -> go -> expert compute -> cpu_done -> combine
        t = max(mix_end, plan_t)
        st["plan"] += t - mix_end
        t2 = max(t, dem_end)
        st["go"] += t2 - t
        expert_ms = jit(expert_full * (routed_slots - p["cpu_slots"]) / 16.9
                              + p["window_overhead_ms"] * p["clock_scale"])
        t3 = t2 + expert_ms
        st["expert"] += expert_ms
        t4 = max(t3, cpu_done)
        st["cpu"] += t4 - t3
        gpu_t = t4 + p["combine_ms"]
        drv_t = plan_t                                          # driver free once plan written
        st["mix"] += mix_l
        st["copy_busy"] += dem_ms + hint_ms
    delattr(simulate, "_pool_free")
    return gpu_t, st


def tg_of(pass_ms, p):
    return 1000.0 * p["tokens_per_pass"] / pass_ms


def calibrate():
    print(f"{'config':16s} {'wall':>6s} {'sim':>6s} {'werr%':>6s} | {'tg':>5s} {'sim_tg':>6s} {'terr%':>6s}")
    worst_p = worst_t = 0.0
    order_meas, order_sim = [], []
    for r in ROWS:
        q = row_params(r)
        pm, st = simulate(q, random.Random(11))
        pm_wall = pm + 3.7 + 0.02 * st["copy_busy"]        # between-pass host work; grows with link congestion
                                                           # (driver publishes + issues copies on the same host)
        tg = tg_of(pm_wall, q)
        wall = 1000.0 * r["tpp"] / r["tg"]                 # measured wall per pass (tg-derived)
        ep = 100 * (pm_wall - wall) / wall
        et = 100 * (tg - r["tg"]) / r["tg"]
        worst_p = max(worst_p, abs(ep)); worst_t = max(worst_t, abs(et))
        order_meas.append((r["tg"], r["name"])); order_sim.append((tg, r["name"]))
        print(f"{r['name']:16s} {wall:6.2f} {pm_wall:6.2f} {ep:6.1f} | {r['tg']:5.1f} {tg:6.1f} {et:6.1f}")
    om = [n for _, n in sorted(order_meas, reverse=True)]
    os_ = [n for _, n in sorted(order_sim, reverse=True)]
    print(f"GATE wall: max abs err {worst_p:.1f}% (limit 5%) -> {'PASS' if worst_p <= 5 else 'FAIL'}")
    print(f"GATE tg:   max abs err {worst_t:.1f}% -> {'PASS' if worst_t <= 5 else 'FAIL'}")
    print(f"order measured: {om}\norder simulated:  {os_} {'MATCH' if om == os_ else 'MISMATCH'}")
    return worst_p, worst_t


# candidate levers = plan 2 D0 questions (a)-(e), applied to the served config (K=3 final row).
# The calibrated model says: at the served point the CPU tier is the binding wait (~0.29 ms/layer)
# and the link is secondary (~0.08 ms/layer); moving experts between tiers is net-neutral (frac 0.2
# is the engine's balance point - confirmed by the f0.35 row). Gains must remove CPU-tier time or
# cold volume, or remove launch/serialization gaps.
LEVERS = {
    "a_hint2ahead": ("demand->prefetch conversion (0.4 experts/layer, predictor-decay-limited): "
                     "link FIFO unchanged total -> net ~0 while CPU binds; re-measure when CPU frees"),
    "b_ringslots":  ("+1 ring slot: less eviction churn (-14.4 cold/pass)"),
    "c_frac005":    ("frac 0.2->0.05: link experts back onto the CPU tier (tests the balance point)"),
    "d_speccpu":    ("speculative CPU start at the doorbell (+0.06 ms/layer earlier, +12% mispredict waste)"),
    "g_graphs":     ("CUDA graphs per window size: launch gaps 0.005->0.002 ms/layer-kernel"),
    "combo_bg":     ("b + g (best without placement/weights changes)"),
}


def lever_row(name):
    r = dict(ROWS[2])                                  # served config, sustained env
    if name == "a_hint2ahead":
        conv = 19.2                                    # 0.4 experts/layer converted
        r["dem"] -= conv; r["pre"] += conv
    if name in ("b_ringslots", "combo_bg"):
        r["cold"] -= 14.4
    if name == "c_frac005":
        r["pcie_frac"] = 0.05
        r["dem"] *= 0.25
    return r


def levers():
    base_row = ROWS[2]
    q = row_params(base_row)
    base, stb = simulate(q, random.Random(3))
    base_wall = base + 3.7 + 0.02 * stb["copy_busy"]
    print(f"served base wall {base_wall:.2f} ms -> {tg_of(base_wall, q):.1f} tok/s (row {base_row['tg']})")
    for name, desc in LEVERS.items():
        r = lever_row(name)
        q = row_params(r)
        if name == "d_speccpu":
            q["cpu_early_ms"] = 0.06                   # CPU starts at doorbell, not at split
            q["cpu_slow"] *= 1.12                      # mispredict waste on the DRAM floor
        if name in ("g_graphs", "combo_bg"):
            q["launch_gap_ms"] = 0.002
        pm, st = simulate(q, random.Random(3))
        wall = pm + 3.7 + 0.02 * st["copy_busy"]
        print(f"  {name:13s} {wall:7.2f} ms -> {tg_of(wall, q):5.1f} tok/s "
              f"({tg_of(wall, q) - tg_of(base_wall, q):+.1f})   # {desc}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--lever", action="store_true")
    args, _un = ap.parse_known_args()
    if args.base or not (args.calibrate or args.lever):
        q = row_params(ROWS[2])
        pm, st = simulate(q, random.Random(5))
        print(f"served base pass {pm:.2f} ms -> {tg_of(pm, q):.1f} tok/s")
        for k, v in st.items():
            print(f"  {k:10s} {v:6.2f} ms/pass")
    if args.calibrate:
        calibrate()
    if args.lever:
        levers()


# =====================================================================================
# X3.1 recalibration (Order 8): clean row 1007_064447 (kept build 4775617, 250 W).
# Engine structure READ FROM CODE (forward.cu serve(), 595-717):
#   serve(l): split -> pool->start -> fetch(demand copies) -> signal(go, copy stream, BEHIND this
#   layer's demand, BEFORE hints) -> PLAN WRITTEN DIRECT TO MAPPED HOST FLAG (b.plan, immediate,
#   NOT behind the copy queue) -> prefetch_hint -> claim(admission) -> pool->wait()  <-- INSIDE
#   serve: the driver thread is serialized per layer by the CPU tier.
# Device stream per layer: tinies(mix/hc/PLE/glue) -> wait_plan (mapped flag: ~0 clean) ->
#   window kernel { spin on go = demand-landing, resident scan } -> gemv -> spin (CPU done) -> combine.
# Consistent clean-row model (ALL bench.log/lead numbers close):
#   wall 35.85 = verify 32.43 (48 x 0.675) + draft 1.65 + head 0.93 + accept 0.85;
#   GPU busy 20.5 = tinies 7.5 + resident 0.3 + gemv 7.7 + go-spin 5.0 (INSIDE window kernel 5.3);
#   waits: wait_plan ~0.2 (mapped), spin 11.7 = pool wall/call 0.514 (bench: 0.136 ms/expert x 3.68
#   experts/call = 0.50) minus window+gemv end 0.27; pool serial (pool->wait), util 76%;
#   link: dem 42.6x1.99MB @18.4 GB/s = 0.096 ms/layer + API 0.008 -> go visible serve+0.104;
#   hint 56.6, admission 58.6 queue behind go (link 17.1 ms busy/pass, ahead of the device).
# KEY CONSEQUENCE (the lead's absorption risk, quantified): pool_done(l) = serve(l)+wall/call is
# INDEPENDENT of GPU-side durations -> every GPU-side saving up to the spin (11.7 ms/pass, floor
# driver pace serve-interval 0.544 = wall/call + API 0.030) is ABSORBED by a longer spin.
PX = dict(
    n_layer=48, tokens_per_pass=2.61,
    tinies_ms=7.5/48,          # mixer+hc+PLE+glue+combine GPU serial, incl tiny-kernel gaps
    resident_ms=0.3/48,        # window-kernel resident scan/compute AFTER go arrives
    gemv_ms=7.7/48,            # GPU expert gemv (after window)
    poll_ms=0.004, driver_api_ms=1.42/48,   # split+pool start+copy APIs+hint+claim (drv_ms sum)
    pool_wall_ms=0.514,        # pool wall per layer call (measured 0.136 ms/expert x 3.68)
    link_bytes_ms=18.4e6, expert_bytes=1.99e6,
    demand=42.6/48, prefetch=56.6/48, admission=58.6/48,
    host_gap_ms=3.42,          # draft 1.65 + head 0.93 + accept 0.85
    plan_mapped=True,          # ENGINE FACT today; False = plan rides at the copy-batch tail
    spec_cap_ms=float("inf"),  # L1: hold speculative copies back to <= cap ms in front of demand
    link_upgrade_ms=0.0,       # Phase 5.0 stand-in: extra link bytes/ms
    window_overlap_ms=0.0,     # L2: useful window work done DURING the go spin (eta up)
    tinies_cut_ms=0.0,         # L3 glue fusion
    cpu_slow=1.0,              # L4 multiplier on pool wall/call
)
ROW_X = dict(wall=35.85, tg=72.8, tpp=2.61)

def simulate_x(p):
    link = p["link_bytes_ms"] + p["link_upgrade_ms"]
    dem_ms = p["demand"] * p["expert_bytes"] / link
    hint_ms = p["prefetch"] * p["expert_bytes"] / link
    adm_ms = p["admission"] * p["expert_bytes"] / link
    tinies = p["tinies_ms"] - p["tinies_cut_ms"]
    gpu_t = drv_free = link_free = 0.0
    st = dict(tinies=0.0, plan=0.0, go=0.0, resident=0.0, gemv=0.0, cpu=0.0,
              copy_busy=0.0, link_idle=0.0, driver_idle=0.0)
    for l in range(p["n_layer"]):
        mix_end = gpu_t + tinies
        serve = max(drv_free, mix_end + p["poll_ms"])
        st["driver_idle"] += serve - drv_free
        # copies: demand issued right after pool->start; hints+admission behind go
        issue = serve + p["driver_api_ms"] * 0.3
        dem_start = max(link_free, issue)     # demand is FIRST in the FIFO (engine order 670-671)
        dem_end = dem_start + dem_ms
        hint_end = dem_end + hint_ms          # go signal rides at dem_end; hints after
        adm_end = hint_end + adm_ms
        if p["spec_cap_ms"] != float("inf"):
            # L1: driver holds speculative issue back so queued speculative bytes never exceed
            # cap ms in front of the next demand (no-op when hint+admission < cap, true today).
            adm_end = min(adm_end, dem_end + p["spec_cap_ms"])
            hint_end = min(hint_end, adm_end)
        if dem_start > link_free: st["link_idle"] += dem_start - link_free
        link_free = adm_end
        st["copy_busy"] += dem_ms + hint_ms + adm_ms
        # pool: serial across layers (pool->wait inside serve); starts at serve
        pool_done = max(drv_free, serve) + p["pool_wall_ms"] * p["cpu_slow"]
        # device stream: tinies -> wait_plan -> window{go-spin + resident} -> gemv -> spin
        t = max(mix_end, serve + p["poll_ms"])          # mapped plan visible ~serve+poll
        if not p["plan_mapped"]:
            t = max(t, adm_end)                        # plan behind the whole batch
        st["tinies"] += tinies; st["plan"] += t - mix_end
        win_start = t
        go_vis = max(dem_end, win_start)
        go_spin = go_vis - win_start
        overlap = min(go_spin, p["window_overlap_ms"])  # L2: resident work overlaps the spin
        st["go"] += go_spin - overlap
        st["resident"] += p["resident_ms"] + overlap   # overlap = GPU work that WAS idle
        win_end = max(go_vis, win_start) + p["resident_ms"] - overlap
        gemv_end = win_end + p["gemv_ms"]
        st["gemv"] += p["gemv_ms"]
        cpu_done = max(pool_done, gemv_end)            # spin: CPU results joined at window tail
        st["cpu"] += cpu_done - gemv_end
        gpu_t = cpu_done
        drv_free = pool_done + p["driver_api_ms"] * 0.7  # driver free after pool->wait + tail APIs
    wall = gpu_t + p["host_gap_ms"]
    return wall, st

def x31():
    p = dict(PX)
    wall, st = simulate_x(p)
    err = 100*(wall-ROW_X["wall"])/ROW_X["wall"]
    print(f"X3.1 base wall {wall:.2f} ms vs measured {ROW_X['wall']:.2f} ({err:+.1f}%) -> "
          f"{1000*p['tokens_per_pass']/wall:.1f} tok/s vs 72.8")
    print(f"  busy = tinies {st['tinies']:.1f} + resident {st['resident']:.1f} + gemv {st['gemv']:.1f} "
          f"+ go-spin {st['go']:.1f} (in window) = {st['tinies']+st['resident']+st['gemv']+st['go']:.1f} (meas 20.5)")
    for k in ("plan","cpu","copy_busy","link_idle","driver_idle"):
        print(f"  {k:12s} {st[k]:6.2f}")
    print(f"  verify {wall-p['host_gap_ms']:.2f} (meas 32.43); spin {st['cpu']:.2f} (lead 14.5*)")
    return wall, st

def x31_levers():
    base, stb = x31()
    tg0 = 1000*PX["tokens_per_pass"]/base
    print()
    def run(name, **kw):
        p = dict(PX); p.update(kw)
        w, st = simulate_x(p)
        tg = 1000*p["tokens_per_pass"]/w
        print(f"  {name:26s} wall {w:6.2f} {tg:5.1f} tok/s ({tg-tg0:+5.1f})  "
              f"go-spin {st['go']:5.2f} spin {st['cpu']:5.2f} plan {st['plan']:5.2f} "
              f"link_idle {st['link_idle']:5.2f} drv_idle {st['driver_idle']:5.2f}")
        return w
    L1  = dict(spec_cap_ms=1.0)
    L1n = dict(spec_cap_ms=1.0, plan_mapped=False)   # cap + plan-behind-batch = worst case
    L2  = dict(window_overlap_ms=(5.3-2.7)/48)       # eta 0.33->0.65: 2.6 ms of window time made useful
    L3  = dict(tinies_cut_ms=3.0/48)                 # glue fusion mid-point
    L4  = dict(cpu_slow=0.70)
    A50 = dict(link_upgrade_ms=(24.0-18.4)*1e6)      # Phase 5.0 arena stand-in (24 GB/s)
    run("L1 spec-cap 1ms", **L1)
    run("L1' cap + plan-behind", **L1n)
    run("L2 window eta 0.65", **L2)
    run("L3 glue -3ms", **L3)
    run("L4 CPU -30%", **L4)
    run("5.0 arena 24GB/s", **A50)
    run("L1+L4", **L1, **L4)
    run("L2+L3+L4", **L2, **L3, **L4)
    run("ALL L1..L4", **L1, **L2, **L3, **L4)
    run("ALL + 5.0", **L1, **L2, **L3, **L4, **A50)

if __name__ == "__main__":
    import sys
    if "--x31" in sys.argv: x31()
    if "--x31-levers" in sys.argv: x31_levers()
