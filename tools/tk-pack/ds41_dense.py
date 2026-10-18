#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 -> TRUSS dense GGUF (`general.architecture = deepseek41`), everything but the routed
experts (ds41_experts.py writes those as an X3.1 overlay) and the two Engram tables (203 GB fp8; their own file).

Only the dense tensors are downloaded: each shard's safetensors header, then one HTTP range per tensor (~9 GB of the
~330 GB checkpoint). Tensor names and metadata keys follow mmnga/Strata `deepseek` src/deepseek41/model.cpp, so its
reader's layout checks apply to this file too; routed experts are renumbered to the pack's kept set (keep256: the
kept original ids in ascending order -> 0..255; `deepseek41.expert_keep` records them).

Formats: FP8 e4m3 weights (ue8m0 scale per 32 x 32 block) and BF16 matrices -> Q8_0; norms, router, router bias,
hyper-connection projections, attention sinks, pooling-compressor weights (the reference runs them in fp32) and the
Engram q/k weights -> F32. Vision, image tokens, the VL router bias and DSpark (mtp.*) are left out.

Run with the exl3 venv (transformers + sympy for the Engram hash layout) and llama-paw's gguf-py:
  PYTHONPATH=~/llama-paw/gguf-py:<poc>/scripts/dsref ~/src/venvs/exl3new/bin/python ds41_dense.py \
      --model-dir ~/models_nvme/dsv41flash_1017 --keep <poc>/data/prune/keep256.pt --out <dir>/ds41-dense.gguf
"""
import argparse
import json
import os
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import gguf

BASE = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"
ARCH = "deepseek41"
E4M3 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float().numpy()


def fetch(shard, lo, hi, tries=8):
    """bytes [lo, hi) of a shard"""
    for t in range(tries):
        try:
            r = urllib.request.Request(BASE + shard, headers={"Range": f"bytes={lo}-{hi - 1}"})
            b = urllib.request.urlopen(r, timeout=300).read()
            if len(b) == hi - lo:
                return b
        except Exception as e:  # noqa: BLE001 - HF drops streams; retry
            print(f"{shard} {lo}: {e} (try {t + 1})", flush=True)
        time.sleep(5 * (t + 1))
    raise SystemExit(f"{shard}: range {lo}-{hi} failed")


def header(shard):
    n = struct.unpack("<Q", fetch(shard, 0, 8))[0]
    return 8 + n, json.loads(fetch(shard, 8, 8 + n))


def to_f32(raw, dtype, shape):
    if dtype == "F32":
        return np.frombuffer(raw, "<f4").reshape(shape).copy()
    if dtype == "BF16":
        return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
    raise SystemExit(f"unexpected dtype {dtype}")


def fp8_deq(w, s, shape):
    """[N, K] e4m3 + [ceil(N/32), ceil(K/32)] e8m0 -> fp32 (kernel_torch.fp8_dequant)"""
    n, k = shape
    v = E4M3[np.frombuffer(w, np.uint8)].reshape(n, k)
    sc = np.exp2(np.frombuffer(s, np.uint8).astype(np.float32) - 127).reshape((n + 31) // 32, (k + 31) // 32)
    return v * np.repeat(np.repeat(sc, 32, 0)[:n], 32, 1)[:, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="config.json, model.safetensors.index.json, tokenizer.json")
    ap.add_argument("--keep", required=True, help="keep256.pt: {'keep': bool [layers][384]}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="first N layers only (a test slice); 0 = all")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.model_dir, "config.json")))["text_config"]
    wmap = json.load(open(os.path.join(a.model_dir, "model.safetensors.index.json")))["weight_map"]
    keep = torch.load(a.keep)["keep"]
    n_layer = a.layers or cfg["num_hidden_layers"]
    kept = [np.nonzero(keep[l].numpy())[0] for l in range(n_layer)]
    n_exp = len(kept[0])
    assert all(len(k) == n_exp for k in kept), "every layer keeps the same number of experts"

    # (gguf name, [hf names], how): how = q8 | f32 | fp8q8 (weight + .scale) ; rows: router row subset
    plan = [("token_embd.weight", ["embed.weight"], "q8", None)]
    for l in range(n_layer):
        b, h = f"blk.{l}.", f"layers.{l}."
        def add(name, hf, how, rows=None):
            if hf in wmap or hf + ".weight" in wmap:
                plan.append((b + name, [hf], how, rows))
        add("attn_norm.weight", h + "attn_norm.weight", "f32")
        add("ffn_norm.weight", h + "ffn_norm.weight", "f32")
        add("attn_sinks.weight", h + "attn.attn_sink", "f32")
        for p, hp in (("attn_q_a", "wq_a"), ("attn_q_b", "wq_b"), ("attn_kv", "wkv"), ("attn_output_a", "wo_a"),
                      ("attn_output_b", "wo_b")):
            add(p + ".weight", h + f"attn.{hp}", "fp8q8")
        add("attn_q_a_norm.weight", h + "attn.q_norm.weight", "f32")
        add("attn_kv_a_norm.weight", h + "attn.kv_norm.weight", "f32")
        for s in ("attn", "ffn"):
            add(f"hc_{s}_fn.weight", h + f"hc_{s}_fn", "f32")
            add(f"hc_{s}_base.weight", h + f"hc_{s}_base", "f32")
            add(f"hc_{s}_scale.weight", h + f"hc_{s}_scale", "f32")
        add("ffn_gate_inp.weight", h + "ffn.gate.weight", "f32", kept[l])
        add("exp_probs_b.bias", h + "ffn.gate.bias", "f32", kept[l])
        for p, w in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            add(f"ffn_{p}_shexp.weight", h + f"ffn.shared_experts.{w}", "fp8q8")
        ratio = cfg["compress_ratios"][l]
        add("attn_compressor_kv.weight", h + "attn.compressor.wkv.weight", "f32" if ratio > 1 else "q8")
        add("attn_compressor_gate.weight", h + "attn.compressor.wgate.weight", "f32")
        add("attn_compressor_norm.weight", h + "attn.compressor.norm.weight", "f32")
        add("indexer.attn_k.weight", h + "attn.indexer.wk.weight", "q8")
        add("indexer.k_norm.weight", h + "attn.indexer.k_norm.weight", "f32")
        add("indexer.attn_q_b.weight", h + "attn.indexer.wq_b", "fp8q8")
        add("indexer.proj.weight", h + "attn.indexer.weights_proj.weight", "q8")
        add("engram_wkv.weight", h + "engram.wkv", "fp8q8")
        add("engram_q.weight", h + "engram.q_weight", "f32")
        add("engram_k.weight", h + "engram.k_weight", "f32")
    if n_layer == cfg["num_hidden_layers"]:
        plan += [("output_norm.weight", ["norm.weight"], "f32", None), ("output.weight", ["head.weight"], "q8", None)]

    def hf_names(hf, how):
        return [hf + ".weight", hf + ".scale"] if how == "fp8q8" else [hf]

    hdrs = {}
    shards = sorted({wmap[n] for _, hfs, how, _ in plan for n in hf_names(hfs[0], how)})
    with ThreadPoolExecutor(a.threads) as ex:
        for s, r in zip(shards, ex.map(header, shards)):
            hdrs[s] = r
    total = sum(hdrs[wmap[n]][1][n]["data_offsets"][1] - hdrs[wmap[n]][1][n]["data_offsets"][0]
                for _, hfs, how, _ in plan for n in hf_names(hfs[0], how))
    print(f"{len(plan)} tensors from {len(shards)} shards, {total / 1e9:.2f} GB to download", flush=True)

    def raw(n):
        base, hd = hdrs[wmap[n]]
        lo, hi = hd[n]["data_offsets"]
        return fetch(wmap[n], base + lo, base + hi), hd[n]["dtype"], hd[n]["shape"]

    def convert(item):
        name, hfs, how, rows = item
        if how == "fp8q8":
            (w, dt, shape), (s, _, _) = raw(hfs[0] + ".weight"), raw(hfs[0] + ".scale")
            assert dt == "F8_E4M3", (name, dt)
            x = fp8_deq(w, s, shape)
        else:
            x = to_f32(*raw(hfs[0]))
        if rows is not None:
            x = x[rows]
        if how == "f32" or x.shape[-1] % 32:
            return name, x.astype(np.float32), gguf.GGMLQuantizationType.F32
        return name, gguf.quants.quantize(x.astype(np.float32), gguf.GGMLQuantizationType.Q8_0), \
            gguf.GGMLQuantizationType.Q8_0

    w = gguf.GGUFWriter(a.out + ".tmp", ARCH)
    k = lambda s: f"{ARCH}.{s}"   # noqa: E731
    u32 = [("block_count", n_layer), ("embedding_length", cfg["hidden_size"]),
           ("attention.head_count", cfg["num_attention_heads"]), ("attention.head_count_kv", cfg["num_key_value_heads"]),
           ("attention.key_length", cfg["head_dim"]), ("attention.value_length", cfg["head_dim"]),
           ("rope.dimension_count", cfg["qk_rope_head_dim"]), ("attention.q_lora_rank", cfg["q_lora_rank"]),
           ("attention.output_lora_rank", cfg["o_lora_rank"]), ("attention.output_group_count", cfg["o_groups"]),
           ("attention.sliding_window", cfg["sliding_window"]), ("expert_count", n_exp),
           ("expert_used_count", cfg["num_experts_per_tok"]), ("expert_feed_forward_length", cfg["moe_intermediate_size"]),
           ("expert_shared_count", cfg["n_shared_experts"]), ("expert_gating_func", 4),
           ("hyper_connection.count", cfg["hc_mult"]), ("hyper_connection.sinkhorn_iterations", cfg["hc_sinkhorn_iters"]),
           ("attention.indexer.head_count", cfg["index_n_heads"]), ("attention.indexer.key_length", cfg["index_head_dim"]),
           ("attention.indexer.top_k", cfg["index_topk"]), ("context_length", cfg["max_position_embeddings"]),
           ("rope.scaling.original_context_length", cfg["rope_scaling"]["original_max_position_embeddings"]),
           ("attention.candidate_source_layer", cfg["candidate_source_layer_id"]),
           ("attention.candidate_top_k_blocks", cfg["candidate_topk_blocks"]),
           ("attention.candidate_block_size", cfg["candidate_block_size"]),
           ("engram.head_count", cfg["engram_n_heads"]), ("engram.key_length", cfg["engram_head_dim"]),
           ("engram.max_ngram_size", cfg["engram_max_ngram_size"])]
    for s, v in u32:
        w.add_uint32(k(s), v)
    for s, v in (("expert_weights_scale", cfg["routed_scaling_factor"]), ("hyper_connection.epsilon", cfg["hc_eps"]),
                 ("attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"]), ("rope.freq_base", cfg["rope_theta"]),
                 ("attention.compress_rope_freq_base", cfg["compress_rope_theta"]),
                 ("rope.scaling.factor", cfg["rope_scaling"]["factor"]),
                 ("rope.scaling.yarn_beta_fast", cfg["rope_scaling"]["beta_fast"]),
                 ("rope.scaling.yarn_beta_slow", cfg["rope_scaling"]["beta_slow"])):
        w.add_float32(k(s), float(v))
    w.add_bool(k("expert_weights_norm"), bool(cfg["norm_topk_prob"]))
    w.add_string(k("rope.scaling.type"), cfg["rope_scaling"]["rope_type"])
    w.add_array(k("attention.compress_ratios"), [int(r) for r in cfg["compress_ratios"][:n_layer]])
    w.add_array(k("attention.kv_source_layers"), [int(r) for r in cfg["kv_source_layer_ids"]])
    w.add_array(k("attention.index_source_layers"), [int(r) for r in cfg["index_source_layer_ids"]])
    w.add_array(k("swiglu_clamp_exp"), [float(cfg["swiglu_limit"])] * n_layer)
    w.add_array(k("expert_keep"), [int(e) for kk in kept for e in kk])

    # Engram hash layout (dsref engram.py: the reference NgramHashState; asserts the compressed vocab size)
    from transformers import PreTrainedTokenizerFast
    import load as L
    from engram import EngramLayout, NgramHashState, compute_hash_multipliers
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(a.model_dir, "tokenizer.json"))
    args = L.make_args(1, 16)
    layout = EngramLayout.from_args(args)
    hs = NgramHashState(args, layout, tok)
    w.add_uint32(k("engram.pad_id"), int(hs.pad_id))
    w.add_array(k("engram.layer_ids"), [int(x) for x in layout.layer_ids])
    u64 = lambda s, v: w.add_key_value(k(s), [int(x) for x in v], gguf.GGUFValueType.ARRAY,   # noqa: E731
                                       sub_type=gguf.GGUFValueType.UINT64)
    u64("engram.multipliers", hs.multipliers.flatten().tolist())          # up to 2^62: int32 would overflow
    u64("engram.primes", hs.primes.flatten().tolist())
    u64("engram.offsets", hs.offsets.flatten().tolist())
    tmap = [int(x) for x in hs.token_map.tolist()]
    tmap += [int(hs.pad_id)] * (cfg["vocab_size"] - len(tmap))       # ids past the tokenizer: the pad n-gram slot
    w.add_array(k("engram.token_map"), tmap)
    u64("engram.table_rows", layout.num_embeddings)
    vocab = [tok.convert_ids_to_tokens(i) for i in range(cfg["vocab_size"])]
    w.add_token_list([t if t is not None else f"<unused{i}>" for i, t in enumerate(vocab)])
    w.add_bos_token_id(cfg.get("bos_token_id", 0))
    w.add_eos_token_id(cfg.get("eos_token_id", 1))

    t0, done = time.time(), 0
    with ThreadPoolExecutor(a.threads) as ex:
        for name, data, qt in ex.map(convert, plan):
            w.add_tensor(name, data, raw_dtype=qt)
            done += 1
            if done % 50 == 0:
                print(f"{done}/{len(plan)} tensors, {time.time() - t0:.0f}s", flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e9:.2f} GB, {len(plan)} tensors, {time.time() - t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
