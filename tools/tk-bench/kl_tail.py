"""Tail statistics of the arms bench KL task (docs/PLAN-20261022-expert-k.md: the mean KL does not see the damage a
pruned / low-rate pack does to a model; 10-23, user: skip 0.10 reads KL ~0.06, Q4 level, yet the model is clearly hurt).

Per teacher-forced step the task records KL(FP4 reference || engine). The mean hides a few very wrong steps, so the
report adds the upper tail: p99, p99.9, and the share of steps with KL over 0.5 (the reference's distribution is
mostly lost there). The sequences are the 1024-token held-out set, so this says nothing about positions past ~1K.
"""
import numpy as np


def summary(rec):
    """rec: the task record {kl: [per step], agree: top-1 hits, n: steps} -> the table fields."""
    kl = np.asarray(rec["kl"])
    return dict(kl=float(kl.mean()), top1=100 * rec["agree"] / rec["n"], p99=float(np.percentile(kl, 99)),
                p999=float(np.percentile(kl, 99.9)), over=100 * float((kl > 0.5).mean()))


def fmt(S, base_kl):
    """The table fragment: mean KL (vs the first arm), top-1, upper tail."""
    if "kl" not in S:
        return ""
    return (f"  KL {S['kl']:.4f} ({S['kl'] - base_kl:+.4f}) top-1 {S['top1']:.1f}%"
            f"  tail p99 {S['p99']:.2f} p99.9 {S['p999']:.2f} >0.5: {S['over']:.2f}%")
