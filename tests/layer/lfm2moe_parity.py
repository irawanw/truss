#!/usr/bin/env python3
"""lfm2moe parity: TRUSS (libtruss, the C API) vs HF transformers' Lfm2Moe layers on THE SAME weights.

The reference is HF's own decoder layers (modeling_lfm2_moe, fp32 math) loaded from the pack TRUSS runs: Q8_0
tensors dequantized with gguf-py, trellis experts reconstructed with exllamav3 (the encoder's own decoder,
reconstruct_had_slice), F32 tensors as stored. So the difference measures TRUSS's implementation (fp16 activations,
kernels), not quantization. Tied head (token_embd) after the final norm.

Checks, on a prompt of real tokens (the corpus held-out ids):
  prompt   the last row's logits after one prompt (TRUSS chunked by its max_chunk)        KL, top-1
  decode   N single-token steps after it (TRUSS decode path: gemv, moe_window, split attention)   KL per step
Gate: mean KL <= 2e-3 (fp16 activations in TRUSS vs fp32 here). Top-1 is reported, not gated: over ~10 rows one
The reference streams layers (one at a time, one causal forward over all ids) so it fits next to TRUSS; run with TRUSS_EXPERT_GB=7.
near-tie flips it (2026-10-14, 4-layer K2 slice: prompts 120 / 128 / 200 -> mean KL 2.3e-4 / 5.3e-4 / 5.4e-4, top-1
7/7, 8/9, 7/7; no step-dependent drift).

Run in the exllamav3 venv (torch + transformers lfm2_moe + exllamav3), GPU via CUDA_VISIBLE_DEVICES:
  lfm2moe_parity.py <pack.gguf> <libtruss.so> <ids.pt> [--steps 16] [--prompt 256]
"""
import argparse
import ctypes
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/llama-paw/gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def tensors(r):
    return {t.name: t for t in r.tensors}


def deq(t):
    if t.tensor_type in (GGMLQuantizationType.F32, GGMLQuantizationType.F16):
        a = np.array(t.data, dtype=np.float32)
    else:
        a = dequantize(t.data, t.tensor_type).astype(np.float32)
    return torch.from_numpy(a.reshape([int(x) for x in reversed(t.shape)]).copy())


def experts(T, L, k_of):
    """gate_up [E][2F][D], down [E][D][F] fp32 from the m3 tensors (exllamav3 reconstruct, as the encoder checks)."""
    import exllamav3_ext as ext
    out = {}
    for p in ("gate", "up", "down"):
        b = f"blk.{L}.ffn_{p}_exps."
        meta = np.array(T[b + "m3_meta"].data).reshape(-1, 2)
        suh = torch.from_numpy(np.array(T[b + "m3_suh"].data).reshape(meta.shape[0], -1).copy()).cuda()
        svh = torch.from_numpy(np.array(T[b + "m3_svh"].data).reshape(meta.shape[0], -1).copy()).cuda()
        tr = np.array(T[b + "m3_trellis"].data).view(np.int16)
        fin, fout = suh.shape[1], svh.shape[1]
        mats = []
        for e in range(meta.shape[0]):
            k = k_of(int(meta[e, 0]))
            words = (fin // 16) * (fout // 16) * int(round(16 * k))
            t = torch.from_numpy(tr[meta[e, 1]: meta[e, 1] + words].copy()).cuda().view(fin // 16, fout // 16, -1)
            w = torch.empty((fin, fout), dtype=torch.float16, device="cuda")
            ext.reconstruct_had_slice(w, t, suh[e], svh[e], k if k != int(k) else int(k), False, True, 0)
            mats.append(w.float().T.contiguous())   # HF [out][in]
        out[p] = torch.stack(mats)
    return torch.cat([out["gate"], out["up"]], dim=1), out["down"]


def layer_weights(cfg, T, f, hkv, L, k_of):
    """HF state dict of layer L from the pack."""
    p = f"blk.{L}."
    sd = {"operator_norm.weight": deq(T[p + "attn_norm.weight"]), "ffn_norm.weight": deq(T[p + "ffn_norm.weight"])}
    if hkv[L]:
        for hf, gg in (("q_proj", "attn_q"), ("k_proj", "attn_k"), ("v_proj", "attn_v"), ("out_proj", "attn_output")):
            sd[f"self_attn.{hf}.weight"] = deq(T[p + gg + ".weight"])
        sd["self_attn.q_layernorm.weight"] = deq(T[p + "attn_q_norm.weight"])
        sd["self_attn.k_layernorm.weight"] = deq(T[p + "attn_k_norm.weight"])
    else:
        sd["conv.conv.weight"] = deq(T[p + "shortconv.conv.weight"]).unsqueeze(1)
        sd["conv.in_proj.weight"] = deq(T[p + "shortconv.in_proj.weight"])
        sd["conv.out_proj.weight"] = deq(T[p + "shortconv.out_proj.weight"])
    if L < cfg.num_dense_layers:
        for hf, gg in (("w1", "ffn_gate"), ("w3", "ffn_up"), ("w2", "ffn_down")):
            sd[f"feed_forward.{hf}.weight"] = deq(T[p + gg + ".weight"])
    else:
        sd["feed_forward.gate.weight"] = deq(T[p + "ffn_gate_inp.weight"])
        sd["feed_forward.expert_bias"] = deq(T[p + "exp_probs_b.bias"])
        gu, dn = experts(T, L, k_of)
        sd["feed_forward.experts.gate_up_proj"], sd["feed_forward.experts.down_proj"] = gu, dn
    return sd


def reference(pack, ids, n_steps):
    from transformers.models.lfm2_moe.configuration_lfm2_moe import Lfm2MoeConfig
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeDecoderLayer, Lfm2MoeRotaryEmbedding
    r = GGUFReader(pack)
    T = tensors(r)
    f = {k: v.contents() for k, v in r.fields.items() if k.startswith("lfm2moe.")}
    n_layer = int(f["lfm2moe.block_count"])
    hkv = list(f["lfm2moe.attention.head_count_kv"])
    cfg = Lfm2MoeConfig(vocab_size=128000, hidden_size=int(f["lfm2moe.embedding_length"]),
                        intermediate_size=int(f["lfm2moe.feed_forward_length"]),
                        moe_intermediate_size=int(f["lfm2moe.expert_feed_forward_length"]), num_hidden_layers=n_layer,
                        num_attention_heads=int(f["lfm2moe.attention.head_count"]),
                        num_key_value_heads=max(hkv), num_experts=int(f["lfm2moe.expert_count"]),
                        num_experts_per_tok=int(f["lfm2moe.expert_used_count"]),
                        num_dense_layers=int(f["lfm2moe.leading_dense_block_count"]),
                        norm_eps=float(f["lfm2moe.attention.layer_norm_rms_epsilon"]),
                        conv_L_cache=int(f["lfm2moe.shortconv.l_cache"]),
                        layer_types=["full_attention" if h else "conv" for h in hkv],
                        rope_parameters={"rope_theta": float(f["lfm2moe.rope.freq_base"]), "rope_type": "default"},
                        use_expert_bias=True, norm_topk_prob=True)
    cfg._attn_implementation = "eager"
    k_of = lambda c: {25: 2.5, 35: 3.5}.get(c, c)
    emb, fnorm = deq(T["token_embd.weight"]).cuda(), deq(T["token_embd_norm.weight"]).cuda()
    rot = Lfm2MoeRotaryEmbedding(config=cfg).cuda()
    n = len(ids)
    # Causal: row n'-1 of one forward over all ids == the last row of the n'-prefix, so ONE pass gives every step's
    # logits, and layers are built, run and freed one at a time (fp32 experts are ~2.4 GB per layer).
    with torch.inference_mode():
        x = emb[torch.tensor(ids, device="cuda")].unsqueeze(0)
        xs = [x, x.clone()]   # [0] fp32 reference, [1] the same with the residual rounded to fp16 after every layer
        pos = torch.arange(n, device="cuda").unsqueeze(0)
        pe = rot(x, pos)
        # eager attention adds attention_mask and applies NO causal mask of its own: every row needs one (conv layers
        # read attention_mask as a padding mask: None there)
        mask = torch.full((n, n), torch.finfo(torch.float32).min, device="cuda").triu(1)[None, None]
        for L in range(n_layer):
            layer = Lfm2MoeDecoderLayer(cfg, L).float().cuda().eval()
            layer.load_state_dict({k: v.cuda() for k, v in layer_weights(cfg, T, f, hkv, L, k_of).items()}, strict=True)
            for i in range(2):
                xs[i] = layer(xs[i], position_embeddings=pe, attention_mask=mask if hkv[L] else None, position_ids=pos,
                              past_key_values=None)
            xs[1] = xs[1].half().float()
            del layer
            torch.cuda.empty_cache()
        out = [[], []]
        for i in range(2):
            for m in range(n - n_steps, n + 1):
                h = xs[i][0, m - 1]
                h = h * torch.rsqrt(h.square().mean() + cfg.norm_eps) * fnorm
                out[i].append((h @ emb.T).float().cpu())
    return out


def truss(pack, lib, ids, n_steps):
    L = ctypes.CDLL(lib)
    L.truss_open.restype = ctypes.c_void_p
    L.truss_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    L.truss_n_vocab.argtypes = [ctypes.c_void_p]
    L.truss_eval.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    L.truss_last_error.restype = ctypes.c_char_p
    m = L.truss_open(pack.encode(), 4096, 256, None)
    if not m:
        raise SystemExit("truss_open: " + L.truss_last_error().decode())
    V = L.truss_n_vocab(m)
    out = []
    lg = (ctypes.c_float * V)()
    p = ids[: len(ids) - n_steps]
    arr = (ctypes.c_int32 * len(p))(*p)
    if L.truss_eval(m, arr, len(p), lg) < 0:
        raise SystemExit("truss_eval: " + L.truss_last_error().decode())
    out.append(torch.tensor(list(lg)))
    for t in ids[len(ids) - n_steps:]:
        one = (ctypes.c_int32 * 1)(t)
        if L.truss_eval(m, one, 1, lg) < 0:
            raise SystemExit("truss_eval: " + L.truss_last_error().decode())
        out.append(torch.tensor(list(lg)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack"), ap.add_argument("lib"), ap.add_argument("ids")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--prompt", type=int, default=256)
    a = ap.parse_args()
    ids = torch.load(a.ids)[-1][: a.prompt + a.steps].tolist()
    got = truss(a.pack, a.lib, ids, a.steps)
    ref, ref16 = reference(a.pack, ids, a.steps)
    kl = lambda g, r: float(F.kl_div(F.log_softmax(g, -1), F.log_softmax(r, -1), log_target=True, reduction="sum"))
    kls, floor, top1 = [], [], 0
    for i, (g, r, r16) in enumerate(zip(got, ref, ref16)):
        kls.append(kl(g, r))
        floor.append(kl(r16, r))
        same = int(g.argmax()) == int(r.argmax())
        top1 += same
        print(f"{'prompt' if i == 0 else f'step {i:2d}'}: KL {kls[-1]:.2e} (fp16-residual reference {floor[-1]:.2e}) "
              f"top-1 {'same' if same else 'DIFF'} rel {float((g - r).norm() / r.norm()):.2e}")
    mean_kl, mean_floor = sum(kls) / len(kls), sum(floor) / len(floor)
    ok = mean_kl <= 2e-3
    print(f"mean KL {mean_kl:.2e} (fp16-residual reference vs fp32: {mean_floor:.2e}), top-1 {top1}/{len(kls)}  "
          f"{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
