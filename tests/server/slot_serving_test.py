"""Host-only test (no GPU) of the server's multi-stream serving layer: Engine with two fake sequences.

Checks, through Engine.generate:
  1. two requests in flight take different slots, and each slot's sequence stays exactly its own prompt + reply;
  2. a third request waits for a free sequence, runs once one is released, and gets 503 rather than waiting forever;
  3. a request goes to the slot whose sequence or checkpoint is the longest prefix of its prompt and, when none
     matches, to the free slot holding the fewest cached tokens (ties: least recently used) - so a short side
     request cannot evict a long conversation - and every request logs its longest common prefix with each slot;
  4. while another slot is busy the prompt read is split into TRUSS_PREFILL_PIECE pieces, and produces the same ids
     (greedy and sampled, penalty history included) as the single-call read it replaces;
  5. the slot comes back when the request raises, and Engine.call refuses to nest (that would deadlock the lock).

  python3 tests/server/slot_serving_test.py
"""
import os
import sys
import threading
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from server import app  # noqa: E402
from server.truss_ctypes import TrussError  # noqa: E402

IM = 99999   # <|im_start|>


class FakeSlots:
    """An engine with `slots` sequences of its own. Every call is recorded with the slot that was selected for it,
    so a call made on the wrong sequence is caught instead of inferred."""

    def __init__(self, slots=2, n_ctx=100000, round_ms=0.0, fail=None):
        self.n_ctx, self.slots, self.drafts = n_ctx, slots, 0
        self.n_vocab = 1_000_000   # the real Model exposes this; the raw-prompt route checks client ids on it (F1)
        self.seq, self.ck = [[] for _ in range(slots)], [None] * slots
        self.selected, self.calls, self.evals = 0, [], 0
        self.round_s, self.fail = round_ms / 1e3, fail   # decode cost per round; call number that raises
        self.n_calls, self.hist = 0, []   # hist: what each sampled call was given as the penalty history

    @property
    def position(self):
        return len(self.seq[self.selected])

    def select(self, slot):
        self.selected = slot

    def rec(self, name, n=0):
        self.n_calls += 1
        self.calls.append((self.selected, name, n))
        if self.fail == self.n_calls:
            raise TrussError(f"injected failure at call {self.fail}")

    def reset(self):
        self.seq[self.selected] = []
        self.rec("reset")

    def checkpoint(self):
        self.ck[self.selected] = list(self.seq[self.selected])
        self.rec("checkpoint")

    def restore(self):
        ck = self.ck[self.selected]
        self.rec("restore")
        if ck is None:
            return -1
        self.seq[self.selected] = list(ck)
        return len(ck)

    def eval(self, tokens):
        self.evals += 1
        self.seq[self.selected] += list(tokens)
        self.rec("eval", len(tokens))

    def eval_argmax(self, tokens):
        time.sleep(self.round_s)   # a decode round takes real time; the tests that need overlap rely on it
        self.seq[self.selected] += list(tokens)
        self.rec("eval_argmax", len(tokens))
        return 5000 + len(self.seq[self.selected])

    def eval_sample(self, tokens, sp):
        """The sampled path: records the penalty history it was given, which must not depend on where the prompt
        read was split."""
        self.seq[self.selected] += list(tokens)
        self.rec("eval_sample", len(tokens))
        self.hist.append(sp.n_history)
        return 1000 + len(self.seq[self.selected])


def engine(model, piece=4096):
    """A bare Engine over the fake model, the way the host-only tests build one."""
    e = app.Engine.__new__(app.Engine)
    e.model = model
    e.chat = types.SimpleNamespace(tok=types.SimpleNamespace(decode=lambda ids, **k: "".join(f"<{i}>" for i in ids)))
    e.stop_ids, e.log_tag, e.log_path, e.dump_dir = set(), "test", None, None
    e.log, e.im_start, e.progress_lines = (lambda line: None), IM, []
    e.progress = lambda line: e.progress_lines.append(line)
    os.environ["TRUSS_PREFILL_PIECE"] = str(piece)   # read by init_sequences
    e.init_sequences()
    return e


def run(e, prompt, n_gen=4, temperature=0.0):
    return [t for t, _, _ in e.generate(prompt, n_gen, temperature, 0.95, 40, 7, [], 0.05) if t is not None]


def prompt(tag, n, tail=True):
    """n distinct ids (tag*100000+i) plus the generation prompt, which is where the checkpoint goes."""
    return [tag * 100000 + i for i in range(1, n + 1)] + ([IM, 7, 8] if tail else [])


ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")


def test_two_streams():
    """Two requests at once: different sequences, and neither sees the other's tokens."""
    m = FakeSlots(round_ms=4.0)
    e = engine(m)
    pa, pb = prompt(1, 300), prompt(2, 400)
    out, both_busy = {}, []

    def go(key, p):
        out[key] = run(e, p, n_gen=40)

    ta, tb = threading.Thread(target=go, args=("a", pa)), threading.Thread(target=go, args=("b", pb))
    t0 = time.time()
    ta.start()
    tb.start()
    while time.time() - t0 < 5.0:
        if all(s.busy for s in e.slots):   # the two requests were in flight together
            both_busy.append([s.idx for s in e.slots])
            break
        time.sleep(0.005)
    ta.join()
    tb.join()

    check("two streams ran at once", both_busy, f"both sequences busy: {both_busy}")
    check("one sequence each", all(s.cached for s in e.slots) and e.slots[0].cached != e.slots[1].cached,
          f"slot 0 {len(e.slots[0].cached)} tokens, slot 1 {len(e.slots[1].cached)}")
    a = 0 if e.slots[0].cached[:3] == pa[:3] else 1
    check("no cross-talk in the sequences",
          all(e.slots[i].cached == m.seq[i] for i in range(2))
          and e.slots[a].cached[:len(pa)] == pa and e.slots[1 - a].cached[:len(pb)] == pb,
          f"slot {a} = A ({len(e.slots[a].cached)} tokens), slot {1 - a} = B ({len(e.slots[1 - a].cached)})")
    # the ids encode their request (tag*100000+i), so a token that leaked from one sequence into the other shows up
    tags = [{i // 100000 for i in s.cached if i >= 100000} for s in e.slots]
    check("no tokens from the other stream", all(len(t) == 1 for t in tags) and tags[0] != tags[1],
          f"slot 0 holds request {tags[0]}, slot 1 holds request {tags[1]}")
    check("both streams free again", not any(s.busy for s in e.slots) and e.slot_gate.permits == e.n_slots,
          f"permits {e.slot_gate.permits}")


def test_third_waits():
    """A third request queues (FIFO) until a sequence is free."""
    m = FakeSlots(round_ms=6.0)
    e = engine(m)
    started, pc = {}, prompt(3, 200)

    def go(key, p):
        for t, _, _ in e.generate(p, 40, 0.0, 1.0, 0, None, []):
            if t is not None:   # the first token out is when this request got a sequence
                started.setdefault(key, time.time())

    ts = [threading.Thread(target=go, args=(k, prompt(i, 200))) for k, i in (("a", 1), ("b", 2))]
    for t in ts:
        t.start()
    t0 = time.time()
    while time.time() - t0 < 5.0 and not all(s.busy for s in e.slots):
        time.sleep(0.005)
    tc = threading.Thread(target=go, args=("c", pc))
    tc.start()
    time.sleep(0.15)   # both are still decoding: c must not have got a token
    check("the third request waited", "c" not in started and all(s.busy for s in e.slots),
          f"started {sorted(started)}, busy {[s.idx for s in e.slots if s.busy]}")
    for t in ts + [tc]:
        t.join()
    check("and then ran", "c" in started and not any(s.busy for s in e.slots),
          f"c got its first token {round(started['c'] - started['a'], 2)} s after a")
    check("on a real sequence", any(s.cached[:len(pc)] == pc for s in e.slots),
          f"{m.n_calls} calls over {e.n_slots} sequences")


def test_assignment():
    """Longest cached/checkpoint prefix wins; with no match anywhere, the slot with the fewest cached tokens."""
    m = FakeSlots()
    e = engine(m)
    pa, pb = prompt(1, 120), prompt(2, 120)
    run(e, pa)   # both free and equally old: slot 0
    check("first prompt takes slot 0", e.slots[0].cached[:3] == pa[:3], f"slot 0 has {len(e.slots[0].cached)} tokens")
    run(e, pb)   # slot 0's sequence is not a prefix of pb: the other one
    check("an unrelated prompt takes the other", e.slots[1].cached[:3] == pb[:3],
          f"slot 1 has {len(e.slots[1].cached)} tokens")
    now = time.time()
    e.slots[0].last_used, e.slots[1].last_used = now + 50, now + 100   # a tie would pick slot 0; only a match can
    mark = len(m.calls)
    pc = pa + [4001, 4002, IM, 7, 8]   # only a prefix match can put this on slot 0, which is the one holding A
    run(e, pc)
    new = m.calls[mark:]
    first = next(i for i, c in enumerate(new) if c[1] == "eval_argmax")
    check("a prompt extending slot 0 goes to slot 0",
          e.slots[0].cached[:len(pa)] == pa and e.slots[1].cached[:3] == pb[:3],
          f"slot 0 {len(e.slots[0].cached)} tokens (A + continuation), slot 1 untouched at {len(e.slots[1].cached)}")
    # restored to the checkpoint (120 tokens) and read the 8 tokens after it; a reset would have re-read all 128
    fed = sum(c[2] for c in new[:first + 1] if c[1] in ("eval", "eval_argmax"))
    check("and read only its tail", [c[1] for c in new[:first]] == ["restore", "eval", "checkpoint"]
          and fed == len(pc) - 120, f"{new[:first + 1]}, {fed} tokens fed of {len(pc)}")
    pd = prompt(4, 60)   # matches neither: the free slot with the fewest cached tokens, not the LRU one
    run(e, pd)
    check("a prompt matching neither takes the slot with the least cached", e.slots[1].cached[:3] == pd[:3],
          f"slot 1 holds {len(e.slots[1].cached)} of D, slot 0 keeps {len(e.slots[0].cached)} of A")


def test_side_request():
    """A1: a short request that matches nothing must not evict a long conversation while a slot with less cached is
    free, whichever of the two was used last. The old rule was pure LRU, and in the live two-agent test a 55-token
    side request took a 126K conversation's sequence (3 of its 16 full reads) and made that conversation read 126K
    again on its next turn."""
    for recent in (0, 1):   # the long conversation as the older sequence, then as the newer one
        m = FakeSlots()
        e = engine(m)
        run(e, prompt(1, 5000))      # slot 0: the conversation, checkpointed before its generation prompt
        run(e, prompt(2, 40))        # slot 1: a short exchange
        now = time.time()
        e.slots[recent].last_used, e.slots[1 - recent].last_used = now + 100, now + 50
        side = [900000 + i for i in range(55)] + [IM, 7, 8]   # shares nothing with either sequence
        run(e, side)
        check(f"a 55-token side request spares the conversation (slot {recent} used last)",
              len(e.slots[0].cached) > 4000 and e.slots[1].cached[:3] == side[:3],
              f"slot 0 keeps {len(e.slots[0].cached)} tokens, slot 1 holds {len(e.slots[1].cached)}")
        cont, n_gen = e.slots[0].cached + [4001, IM, 7, 8], 4
        mark, cached = len(m.calls), len(e.slots[0].cached)
        run(e, cont, n_gen)
        fed = sum(c[2] for c in m.calls[mark:] if c[1].startswith("eval"))
        check("and that conversation resumes on its own sequence",
              fed == (len(cont) - cached) + (n_gen - 1),   # the prompt's last token is what yields the first reply
              f"{fed} tokens fed: {len(cont) - cached} new + {n_gen - 1} more rounds, {cached} reused")


def test_lcp_log():
    """A1: every request logs the longest common prefix with every slot's cache and checkpoint, reused or not -
    the prefill line only ever says 0 or all of it, which is what hid the reads."""
    m = FakeSlots()
    e = engine(m)
    pa = prompt(1, 300)
    run(e, pa)      # both sequences empty when it was picked
    check("an empty server logs a zero lcp", any("lcp s0 0/ck 0, s1 0/ck 0" in l for l in e.progress_lines),
          next((l for l in e.progress_lines if "lcp " in l), "no lcp line"))
    e.progress_lines.clear()
    # slot 0 holds 303 prompt tokens + its reply, checkpointed at 300: the new prompt shares the 303, not the reply
    run(e, pa + [4001, 4002, IM, 7, 8])
    line = next((l for l in e.progress_lines if "lcp " in l), "")
    check("the line says what each slot shared with the prompt", "lcp s0 303/ck 300, s1 0/ck 0" in line, line)


def test_slot_wait_503():
    """Review F3: the queue for a free sequence is bounded. TRUSS_ENGINE_CALL_TIMEOUT covers the engine lock, which
    the live configuration (engine_slots) never takes, so a wedged slot used to queue every later request forever."""
    m = FakeSlots()
    os.environ["TRUSS_SLOT_WAIT_TIMEOUT"] = "0.2"   # read by init_sequences
    try:
        e = engine(m)
        check("the slot gate carries the timeout", e.slot_gate.timeout == 0.2, f"{e.slot_gate.timeout} s")
        for _ in range(e.n_slots):
            e.acquire_slot(prompt(1, 20))   # every permit taken and none given back: the wedged-slot case
        try:
            e.acquire_slot(prompt(2, 20))
            check("a request behind a full queue fails instead of hanging", False, "acquire_slot returned")
        except app.HTTPException as x:
            check("a request behind a full queue gets 503", x.status_code == 503 and "slot gate" in str(x.detail),
                  f"{x.status_code}: {x.detail}")
        e.release_slot(e.slots[0])   # one sequence comes free: the waiter that timed out must have left no ticket
        t = time.time()
        s = e.acquire_slot(prompt(3, 20))
        check("and the queue still works for the next request", s.idx == 0 and time.time() - t < 0.2,
              f"slot {s.idx} after {time.time() - t:.2f} s")
    finally:
        os.environ.pop("TRUSS_SLOT_WAIT_TIMEOUT", None)


def test_piecewise_prefill():
    """The split read: pieces while the other slot decodes, one call when it is idle, same ids either way."""
    piece, n = 256, 1000
    p = prompt(1, n, tail=False)   # no IM_START: the whole prompt goes through step(), like /v1/completions

    m1 = FakeSlots()
    e1 = engine(m1, piece=piece)
    ids_alone = run(e1, p, n_gen=3)
    # nothing else is running, so the prompt is read the way it always was: one call that also samples the token
    check("idle: the prompt is one call", m1.calls[:2] == [(0, "reset", 0), (0, "eval_argmax", len(p))],
          f"{m1.calls[:2]}")

    m2 = FakeSlots()
    e2 = engine(m2, piece=piece)
    e2.slots[1].busy = True   # the other stream is decoding
    ids_split = run(e2, p, n_gen=3)
    e2.slots[1].busy = False
    # the calls up to the one that samples the first token are the prompt read
    first = next(i for i, c in enumerate(m2.calls) if c[1] == "eval_argmax")
    reads = [c[2] for c in m2.calls[:first] if c[1] == "eval"]
    check("busy: read in pieces", reads == [256, 256, 256, 231], f"{reads} then {m2.calls[first]}")
    check("the step that samples is the last call of the read", m2.calls[first] == (0, "eval_argmax", 1),
          f"{m2.calls[first]}")
    check("split read gives the same ids", ids_split == ids_alone, f"{ids_alone} vs {ids_split}")
    check("and the same sequence", e2.slots[0].cached == e1.slots[0].cached == p + ids_alone[:-1],
          f"{len(e2.slots[0].cached)} tokens")
    check("piece progress lines name the slot", any("slot 0 prefill" in ln for ln in e2.progress_lines),
          f"{e2.progress_lines[:1]}")

    # the served config samples with penalties: the split read must not move the history either
    m3, m4 = FakeSlots(), FakeSlots()
    e3, e4 = engine(m3, piece=piece), engine(m4, piece=piece)
    e4.slots[1].busy = True

    def sampled(e):
        return [t for t, _, _ in e.generate(p, 3, 0.8, 0.95, 40, 7, [], 0.05, presence_penalty=1.5,
                                            penalty_last_n=64) if t is not None]

    ids_c, ids_d = sampled(e3), sampled(e4)
    e4.slots[1].busy = False
    check("split read keeps the sampled path identical", ids_c == ids_d and m3.hist == m4.hist,
          f"ids {ids_c}, penalty history sizes {m3.hist} vs {m4.hist}")


def test_released_on_error():
    """A request that dies still gives its sequence back."""
    m = FakeSlots(fail=2)   # the prompt read raises
    e = engine(m)
    raised = None
    try:
        run(e, prompt(1, 50))
    except TrussError as exc:
        raised = str(exc)
    check("the error reached the caller", raised is not None, f"{raised}")
    check("the sequence came back", not any(s.busy for s in e.slots) and e.slot_gate.permits == e.n_slots,
          f"permits {e.slot_gate.permits}")
    check("its stale record was dropped", e.slots[0].cached == [], f"slot 0 holds {len(e.slots[0].cached)} tokens")
    check("and the engine is usable again", len(run(e, prompt(2, 50))) == 4, "next request completed")


def test_no_nesting():
    """Engine.call must not be re-entrant: a call from inside a call would deadlock the 1-permit lock."""
    e = engine(FakeSlots())
    boom = []

    def nested(*_a):
        try:
            e.call(e.slots[0], e.model.reset)
        except RuntimeError as exc:
            boom.append(str(exc))

    e.model.eval_argmax = nested
    try:
        list(e.generate(prompt(1, 10), 1, 0.0, 1.0, 0, None, []))
    except Exception:   # noqa: BLE001 - the fake returns no token, so generate may end early
        pass
    check("a nested engine call raises instead of deadlocking", boom, f"{boom[:1]}")


def test_fair_order():
    """FairGate serves arrivals in order, so neither stream can starve behind the other."""
    gate = app.FairGate(1, "test")
    order, served = [], []
    gate.acquire()   # held while the waiters queue up behind it
    for i in range(5):
        order.append(i)
        threading.Thread(target=lambda i=i: (gate.acquire(), served.append(i), time.sleep(0.01), gate.release())).start()
        time.sleep(0.03)   # so the arrival order is the launch order
    gate.release()
    time.sleep(0.4)
    check("waiters were served in arrival order", served == order, f"served {served}, arrived {order}")


def main():
    for t in (test_two_streams, test_third_waits, test_assignment, test_side_request, test_lcp_log,
              test_slot_wait_503, test_piecewise_prefill, test_released_on_error, test_no_nesting,
              test_fair_order):
        print(f"-- {t.__name__}")
        t()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
