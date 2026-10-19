#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 DSpark drafter (the `mtp.0-2.*` namespace) -> TRUSS drafter GGUF (`general.architecture
= deepseek41`), dense side only: unlike ds41_dense.py this reads a LOCAL model dir (the mtp shards are fetched to
/mnt/hdd/.../src_mtp) and covers only the drafter, not the main model.

The routed experts (`mtp.<i>.ffn.experts.<e>`) are NOT here: they are X3.1-encoded (K3, all 128 kept) by
`20261019_truss_port/scripts/a3_encode_dspark.py` and packed by `ds41_experts.py --prefix dspark` into the overlay
`ds41-dspark-experts.gguf` (`TRUSS_DSPARK_EXPERT_OVERLAY`). The main model's embed / head / norm are NOT copied either (the drafter reuses the
main file's, exactly as Flash-Next's MTP does).

Formats match ds41_dense.py: FP8 e4m3 (ue8m0 32 x 32 scale) and BF16 matrices -> Q8_0; norms, routers + router
bias, hyper-connection projections and attention sinks -> F32.

usage: PYTHONPATH=~/llama-paw/gguf-py ~/src/venvs/exl3new/bin/python ds41_dspark.py \
           --src /mnt/hdd/ml/dsv41flash/20261019_truss_port/src_mtp --out <drafter>/ds41-dspark-dense.gguf
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import gguf

ARCH = "deepseek41"
E4M3 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float().numpy()


def to_f32(raw, dtype, shape):
    if dtype == "F32":
        return np.frombuffer(raw, "<f4").reshape(shape).copy()
    if dtype == "BF16":
        return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
    raise SystemExit(f"unexpected dtype {dtype}")


def fp8_deq(w, s, shape):
    """[N, K] e4m3 + [ceil(N/32), ceil(K/32)] e8m0 -> fp32 (ds41_dense.fp8_deq)"""
    n, k = shape
    v = E4M3[np.frombuffer(w, np.uint8)].reshape(n, k)
    sc = np.exp2(np.frombuffer(s, np.uint8).astype(np.float32) - 127).reshape((n + 31) // 32, (k + 31) // 32)
    return v * np.repeat(np.repeat(sc, 32, 0)[:n], 32, 1)[:, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="local model dir with config.json, index and the mtp shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="first N drafter layers (test); 0 = num_nextn_predict_layers")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg = cfg.get("text_config", cfg)
    wmap = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))["weight_map"]
    n_mtp = a.layers or cfg["num_nextn_predict_layers"]
    assert cfg.get("dspark_block_size"), "config has no dspark_block_size"

    # (gguf name, hf name, how); how = q8 | f32 | fp8q8 ; hf may be a bare tensor (norms) or a `.weight`
    plan = []
    for i in range(n_mtp):
        b, h = f"dspark.{i}.", f"mtp.{i}."
        for p, hp in (("attn_q_a", "attn.wq_a"), ("attn_q_b", "attn.wq_b"), ("attn_kv", "attn.wkv"),
                      ("attn_output_a", "attn.wo_a"), ("attn_output_b", "attn.wo_b")):
            plan.append((b + p + ".weight", h + hp, "fp8q8"))
        for p, w in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
            plan.append((b + f"ffn_{p}_shexp.weight", h + f"ffn.shared_experts.{w}", "fp8q8"))
        plan += [(b + "attn_norm.weight", h + "attn_norm.weight", "f32"),
                 (b + "ffn_norm.weight", h + "ffn_norm.weight", "f32"),
                 (b + "attn_q_a_norm.weight", h + "attn.q_norm.weight", "f32"),
                 (b + "attn_kv_a_norm.weight", h + "attn.kv_norm.weight", "f32"),
                 (b + "attn_sinks.weight", h + "attn.attn_sink", "f32"),
                 (b + "ffn_gate_inp.weight", h + "ffn.gate.weight", "f32"),
                 (b + "ffn_gate_inp.bias", h + "ffn.gate.bias", "f32"),
                 (b + "ffn_gate_inp.bias_vl", h + "ffn.gate.bias_vl", "f32"),
                 (b + "hc_attn_fn.weight", h + "hc_attn_fn", "f32"),
                 (b + "hc_attn_base.weight", h + "hc_attn_base", "f32"),
                 (b + "hc_attn_scale.weight", h + "hc_attn_scale", "f32"),
                 (b + "hc_ffn_fn.weight", h + "hc_ffn_fn", "f32"),
                 (b + "hc_ffn_base.weight", h + "hc_ffn_base", "f32"),
                 (b + "hc_ffn_scale.weight", h + "hc_ffn_scale", "f32")]
        if i == 0:
            plan += [(b + "main_proj.weight", h + "main_proj", "fp8q8"),
                     (b + "main_norm.weight", h + "main_norm.weight", "f32")]
        if i == n_mtp - 1:
            plan += [(b + "norm.weight", h + "norm.weight", "f32"),
                     (b + "markov_embed.weight", h + "markov_head.embed.weight", "q8"),
                     (b + "markov_head.weight", h + "markov_head.head.weight", "q8"),
                     (b + "confidence_proj.weight", h + "confidence_head.proj.weight", "f32")]

    def hf_names(hf, how):
        return [hf + ".weight", hf + ".scale"] if how == "fp8q8" else [hf]

    for _, hf, how in plan:
        for n in hf_names(hf, how):
            if n not in wmap:
                raise SystemExit(f"missing in the index: {n}")

    shards = sorted({wmap[n] for _, hf, how in plan for n in hf_names(hf, how)})
    from safetensors import safe_open
    handles = {s: safe_open(os.path.join(a.src, s), framework="pt", device="cpu") for s in shards}

    def raw(name):
        t = handles[wmap[name]].get_tensor(name)
        return t

    def convert(item):
        name, hf, how = item
        if how == "fp8q8":
            w, s = raw(hf + ".weight"), raw(hf + ".scale")
            assert w.dtype == torch.float8_e4m3fn, (name, w.dtype)
            x = fp8_deq(w.view(torch.uint8).numpy().tobytes(), s.view(torch.uint8).numpy().tobytes(), tuple(w.shape))
        else:
            t = raw(hf)
            x = t.float().numpy() if t.dtype == torch.bfloat16 else t.numpy()
        if how == "f32" or x.shape[-1] % 32:
            return name, np.ascontiguousarray(x.astype(np.float32)), gguf.GGMLQuantizationType.F32
        return name, gguf.quants.quantize(np.ascontiguousarray(x.astype(np.float32)), gguf.GGMLQuantizationType.Q8_0), \
            gguf.GGMLQuantizationType.Q8_0

    w = gguf.GGUFWriter(a.out + ".tmp", ARCH)
    k = lambda s: f"{ARCH}.{s}"   # noqa: E731
    w.add_uint32(k("block_count"), n_mtp)
    w.add_uint32(k("dspark.n_layer"), n_mtp)
    w.add_uint32(k("dspark.block_size"), int(cfg["dspark_block_size"]))
    w.add_uint32(k("dspark.window"), 128)
    w.add_uint32(k("dspark.markov_rank"), int(cfg["dspark_markov_rank"]))
    w.add_uint32(k("dspark.noise_token"), int(cfg["dspark_noise_token_id"]))
    w.add_uint32(k("dspark.n_expert"), int(cfg["dspark_n_routed_experts"]))
    w.add_uint32(k("dspark.n_expert_used"), int(cfg["dspark_num_experts_per_tok"]))
    w.add_array(k("dspark.target_layers"), [int(x) for x in cfg["dspark_target_layer_ids"]])
    # the drafter embeds/heads/norms reuse the main file's, so the shared hyperparameters ride along for the binder
    for s, v in (("embedding_length", cfg["hidden_size"]), ("attention.head_count", cfg["num_attention_heads"]),
                 ("attention.key_length", cfg["head_dim"]), ("attention.value_length", cfg["head_dim"]),
                 ("rope.dimension_count", cfg["qk_rope_head_dim"]), ("attention.q_lora_rank", cfg["q_lora_rank"]),
                 ("attention.output_lora_rank", cfg["o_lora_rank"]), ("attention.output_group_count", cfg["o_groups"]),
                 ("attention.sliding_window", 128), ("expert_feed_forward_length", cfg["moe_intermediate_size"]),
                 ("hyper_connection.count", cfg["hc_mult"])):
        w.add_uint32(k(s), v)
    for s, v in (("attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"]),
                 ("hyper_connection.epsilon", cfg["hc_eps"]), ("rope.freq_base", cfg["rope_theta"])):
        w.add_float32(k(s), float(v))

    t0 = time.time()
    total = 0
    for name, data, qt in map(convert, plan):
        w.add_tensor(name, data, raw_dtype=qt)
        total += data.nbytes if hasattr(data, "nbytes") else len(data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e6:.1f} MB, {len(plan)} tensors, {time.time() - t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
