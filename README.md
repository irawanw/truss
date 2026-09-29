# trellis-kernel — "Loom"

A decode engine for **EXL3-format trellis models** (PAW X3 GGUF and EXL3 safetensors) on
**RTX 3090 / RTX 3060 (Ampere, sm_86)**, and nothing else. Status: **plan only**, 2026-09-27.

| file | what |
|---|---|
| [PLAN.md](PLAN.md) | phases, gates, kill criteria, calendar, risks |
| [docs/01-evidence-the-wall.md](docs/01-evidence-the-wall.md) | everything we have measured: where the time goes, what is closed |
| [docs/02-landscape.md](docs/02-landscape.md) | exllamav3, sglang-exl3, TensorFold, llama-paw: speeds and what each does |
| [docs/03-levers.md](docs/03-levers.md) | every speed lever that keeps accuracy, with evidence and exactness class |
| [docs/04-architecture.md](docs/04-architecture.md) | engine design, code layout, tests, how it grows |
| [docs/05-flashnext-1gpu-2gpu.md](docs/05-flashnext-1gpu-2gpu.md) | Flash-Next: Strata result, the 2× rule, 1-GPU pack S1 vs 2-GPU pack D2 (PLAN Phase 7) |

## Verdict

**The wall.** Our trellis matmul is limited by instructions per weight, not by memory. Deleting
all weight traffic still leaves 73–88% of its time. Specifically:

- the kernel skeleton is about half the time;
- 3-bit and K3.5 window extraction is expensive;
- there are 397 launches per token.

Around the matmul sits about 7 ms/token of framework cost (other kernels, idle gaps), plus
6.7 ms of host time per speculative round. The new X3.1 model (K3.5) **has no fast kernels in
llama-paw at all**, and its speed has never been measured.

**The competition is ahead, and it proves the headroom is real.**

- SGLang with the sglang-exl3 plugin decodes a 3.0 bpw EXL3 27B at **52.4 tok/s** without a
  drafter, and **225 tok/s** on code with DFlash2, on one 3090.
- Its linears take 13.3 ms against exllamav3's 19.2. That is roughly 686 GB/s at 3 bits, where
  our 3-bit shapes run at 428–471.
- llama-paw does 40.7 (B3.5) and 100.45 with DFlash2.

**Why a DGX Spark reaches ~100 tok/s when a 3090 has 3.7× its bandwidth**
(`docs/02-landscape.md`):

- TensorFold's Spark result is a large **speculative multiplier**, not more hardware: 12.9 serial
  → 49.6 on standard coding replies, and up to 126.9 on long structured prompts. It comes from
  three things:
  - verifying 12 rows costs only 1.09× one row on a Spark;
  - draft trees plus copy chains commit 4.7–12 tokens per round;
  - the host is hidden.
- On our 3090 the same row count costs 1.52×, we use a chain (3.48 tokens/round), and 6.7 ms of
  host time per round is exposed.
- **My earlier plan played it safe by treating 3.48 tokens/round as the drafter's limit. It is
  not: trees, copies and a flat verify curve are engine features.**

**What a new engine can and cannot do (arithmetic, in `PLAN.md` §2):**

- **Serial decode:** no engine can exceed **74.9 tok/s** on X3.1 on a 3090 (the read ceiling).
  Loom's realistic range is 50–58 tok/s, +5% to +25% over the best existing stack at equal bits.
- **Speculative decode, where the big gain is:** a TensorFold-class design on a 3090, on our
  **trellis** file, points to **~150–175 tok/s on ordinary coding replies** (190–200 only in the
  best case) and roughly 1.5–2.5× that on long structured output and file edits. Compare
  llama-paw's 100.45 and sglang-exl3's 225 on its own code prompts. The trellis sensitivity
  table is in `PLAN.md` §2.
- The key unknowns are whether the 3090's verify curve flattens (≤ 1.2× at 12–16 rows is
  physically available but unproven) and whether our Q2_K drafter's candidates are good enough.
  Phase 0 measures both, including running TensorFold itself on a 3090.
- If raw serial speed were the only goal, the cheapest path would be to export X3.1 to EXL3 and
  run it in exllamav3. Phase 0 measures that too.

**Why our own engine**, if Phase 0 agrees:

1. trellis K3.5 kernels built for flat verify, which no engine has;
2. exact tree speculation with GDN tree recurrence on our formats;
3. the RTX 3060;
4. the Flash-Next MoE on 2×3090;
5. escaping a 19.5k-line llama.cpp overlay that broke silently on the last upstream port.

Phase 1 has a hard kill gate. If the new kernel is not ≥ 1.10× the best existing trellis kernel
after three iterations, the engine stops and the kernel work goes into llama-paw instead.

## First action

`PLAN.md` Phase 0 (2–4 days, measurement only):

1. X3.1 speed in llama-paw;
2. X3.1 exported to EXL3 and measured in exllamav3;
3. a per-shape kernel baseline table at 1–16 rows;
4. platform probes;
5. **TensorFold itself on a 3090**, with its round trace;
6. the tree ceiling of our own drafter, offline.

No engine code until decision D0 is written.
