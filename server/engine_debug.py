"""Engine diagnostics (a mixin of server.engine.Engine).
"""
import json
import os
import time
import numpy as np


class EngineDebug:
    """Engine's diagnostics: request dumps, stop logs, saved requests, the /debug/probe distribution check."""

    def dump(self, prompt_ids, text, finish, sampling, keep=20):
        """--dump-dir: one JSON per request (prompt tail, generated text, finish, sampling), the newest `keep` kept;
        for diagnosing replies after the fact (e.g. a 32K-token reply that hit the client's cap)."""
        if not self.dump_dir:
            return
        try:
            os.makedirs(self.dump_dir, exist_ok=True)
            path = os.path.join(self.dump_dir, time.strftime("%Y%m%d-%H%M%S") + f"-{len(prompt_ids)}.json")
            with open(path, "w") as f:
                json.dump({"n_prompt": len(prompt_ids), "prompt_tail": self.chat.tok.decode(prompt_ids[-4000:]),
                           "finish": finish, "sampling": sampling, "text": text}, f, ensure_ascii=False)
            for old in sorted(os.listdir(self.dump_dir))[:-keep]:
                os.remove(os.path.join(self.dump_dir, old))
        except OSError as e:
            print(f"dump failed: {e}", flush=True)

    def log_stop(self, slot, tok, n_prompt, n_gen, rounds):
        """A reply ended with an end-of-turn token: where it came from (the round's sampled first token, i.e. the
        previous round's verified sample, or an accepted draft) and the text before it, for stall diagnosis."""
        src = "plain step"
        pos = n_gen - 1   # index of the stop token among the generated tokens
        for start, emitted, after in reversed(rounds):
            if start <= pos < start + len(emitted):
                i = pos - start
                src = (f"round at gen {start}: emitted[{i}] of {len(emitted)} "
                       f"({'the verified sample fed as nxt' if i == 0 else 'an ACCEPTED DRAFT'})")
                break
        before = self.chat.tok.decode(slot.cached[n_prompt:n_prompt + n_gen - 1][-40:], skip_special_tokens=False)
        if before.rstrip().endswith("</tool_call>"):
            return   # the normal end of an agent step: not logged
        self.log(f"{self.log_tag}: slot {slot.idx} STOP without a tool call at gen {n_gen} ({src}): ...{before[-120:]!r}")

    def save_request(self, prompt_ids, gen_ids, finish, n_gen, meta):
        """TRUSS_REQ_LOG_DIR: the last 12 requests as .npz (prompt ids, generated ids as held by the engine, meta),
        so a stopped reply can be replayed by /debug/probe."""
        d = os.environ.get("TRUSS_REQ_LOG_DIR")
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
            name = time.strftime("%Y%m%d-%H%M%S") + f"-{finish}-{n_gen}"
            np.savez(os.path.join(d, name + ".npz"), prompt=np.asarray(prompt_ids, dtype=np.int32),
                     gen=np.asarray(gen_ids[:n_gen], dtype=np.int32), meta=json.dumps(dict(meta, finish=finish)))
            for old in sorted(os.listdir(d))[:-12]:
                os.remove(os.path.join(d, old))
        except OSError as e:
            print(f"save_request failed: {e}", flush=True)

    def probe(self, name, window=400):
        """Replay a saved request on the plain path (truss_eval, no drafts) and measure, at each of the last
        `window` generated positions, the model's probability of the token it produced and of the stop tokens,
        raw (T 1) and under the request's sampling filters. The engine's sequence is lost (the next request reads
        its prompt again)."""
        d = os.environ.get("TRUSS_REQ_LOG_DIR")
        z = np.load(os.path.join(d, name), allow_pickle=False)
        prompt, gen = [int(x) for x in z["prompt"]], [int(x) for x in z["gen"]]
        meta = json.loads(str(z["meta"]))
        stops = sorted(self.stop_ids)
        T, tp, tk, mp = meta["temperature"], meta["top_p"], meta["top_k"], meta["min_p"]

        def dist(logits):
            x = logits.astype(np.float64)
            raw = np.exp(x - x.max()); raw /= raw.sum()
            if T <= 0:
                return raw, raw
            y = x / T
            q = np.exp(y - y.max()); q /= q.sum()
            keep = np.ones_like(q, dtype=bool)
            if tk and tk > 0:
                keep &= y >= np.sort(y)[-tk]
            if mp and mp > 0:
                keep &= q >= mp * q.max()
            if tp and 0 < tp < 1:
                o = np.argsort(-q)
                c = np.cumsum(q[o])
                kp = np.zeros_like(keep); kp[o[:int(np.searchsorted(c, tp) + 1)]] = True
                keep &= kp
            f = np.where(keep, q, 0.0); f /= f.sum()
            return raw, f

        # probe is a request like any other: it takes a sequence (its replay destroys it, so the next request on
        # that slot reads its prompt again) and every call goes through Engine.call.
        slot = self.acquire_slot(prompt, f"probe: {name}")
        try:
            self.call(slot, self.model.reset)
            slot.cached, slot.ck_tokens = [], []
            n = len(gen)
            j0 = max(0, n - window)
            logits = self.call(slot, self.model.eval, prompt + gen[:j0])
            rows = []
            think_closed = bool(self.think_close in gen[:j0])
            for j in range(j0, n):
                raw, f = dist(logits)
                t = gen[j]
                rows.append(dict(j=j, tok=int(t), text=self.chat.tok.decode([t]), p_tok=float(f[t]), p_tok_raw=float(raw[t]),
                                 p_stop=float(sum(f[s] for s in stops)), p_stop_raw=float(sum(raw[s] for s in stops)),
                                 think_closed=think_closed,
                                 top=[(self.chat.tok.decode([int(i)]), round(float(f[i]), 4)) for i in np.argsort(-f)[:6]]))
                think_closed = think_closed or t == self.think_close
                if j + 1 < n:
                    logits = self.call(slot, self.model.eval, [t])
            self.call(slot, self.model.reset)
            slot.cached = []
        finally:
            self.release_slot(slot)
        last = rows[-1]
        risky = [r for r in rows if r["p_stop"] > 0.001]
        return dict(meta=meta, n_prompt=len(prompt), n_gen=n, last=last,
                    positions_with_p_stop_over_0_001=[{k: r[k] for k in ("j", "text", "p_stop", "p_stop_raw", "think_closed")}
                                                      for r in risky],
                    tail_text=self.chat.tok.decode(gen[-60:], skip_special_tokens=False))
