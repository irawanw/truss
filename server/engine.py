"""The server's engine: the model, its sequence slots and the fair gates (generate() and the diagnostics are mixins:
server/generate.py, server/engine_debug.py).
"""
import os
import threading
import time
from fastapi import HTTPException
from server.chat import Chat
from server.vision import Tower
from server.truss_ctypes import Model
from server.engine_debug import EngineDebug
from server.generate import EngineGenerate
from server.seq import FairGate, Slot, lcp, prefix_len


class Engine(EngineGenerate, EngineDebug):
    """The model plus its sequences: TRUSS_SLOTS of them, one before that flag existed. generate() is the only way
    in; it owns a slot for the whole request and holds the engine lock one call at a time, so two requests take
    turns per round instead of per request."""

    def __init__(self, a):
        # the tokenizer (transformers + torch imports, ~2.3 s) loads on a thread while the model loads: the ctypes
        # call releases the GIL
        chat = {}
        tok_thread = threading.Thread(target=lambda: chat.setdefault("c", Chat(a.tokenizer, None)), daemon=True)
        tok_thread.start()
        self.model = Model(a.model, a.n_ctx, a.chunk, a.expert_usage, mtp=a.mtp, drafts=a.drafts, cpu_dir=a.cpu_dir,
                           cpu_share=a.cpu_share, cpu_threads=a.cpu_threads, draft_vocab=a.draft_vocab,
                           draft_min_p=a.draft_min_p)
        tok_thread.join()
        if "c" not in chat:
            raise RuntimeError(f"tokenizer {a.tokenizer} failed to load (see the thread's traceback above)")
        self.chat = chat["c"]
        self.chat.template = self.model.meta_string("tokenizer.chat_template")
        # DeepSeek-V4.1 has no GGUF chat template: its prompt and reply format come from the official encoder
        # (server/encoding.py, MIT), driven by server/ds41.py. The family decides, not a flag.
        self.ds41 = None
        if self.model.meta_string("general.architecture") == "deepseek41":
            from server.ds41 import Ds41Chat
            self.ds41 = Ds41Chat(self.chat.tok)
        self.stop_ids = {i for i in (self.model.meta_int("tokenizer.ggml.eos_token_id"),
                                     self.chat.token_id("<|im_end|>"), self.chat.token_id("<|endoftext|>"),
                                     self.chat.token_id("<｜end▁of▁sentence｜>"))
                         if i is not None and i >= 0}
        self.name, self.log_path, self.log_tag = a.name, a.log, a.log_tag
        self.allow_local_images = getattr(a, "allow_local_images", False)
        self.vision = Tower(a.mmproj, a.mmproj_device) if getattr(a, "mmproj", None) else None
        self.dump_dir = a.dump_dir
        self.init_sequences(a.chunk)   # one record per sequence + the two fair gates (see below)
        self.im_start = self.chat.token_id("<|im_start|>")
        if self.ds41:   # DeepSeek's turn marker: the prompt-end checkpoint sits before the last <｜Assistant｜>
            from server import encoding as ds_enc
            self.im_start = self.chat.token_id(ds_enc.ASSISTANT_SP_TOKEN)
        self.think_close = self.chat.token_id("</think>")

    def log(self, line: str):
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    def progress(self, line: str):
        """Live progress for the console / pm2 log only (the --log file keeps one summary line per request)."""
        print(line, flush=True)

    # ------------------------------------------------------------------ the sequences

    def init_sequences(self, chunk=8192):
        """One record per engine sequence plus the two fair gates. __init__ calls it after the model loads; the
        host-only tests call it on a bare Engine whose model is a fake."""
        self.n_slots = max(1, getattr(self.model, "slots", 1))   # truss_slots(), 1 without TRUSS_SLOTS
        self.slots = [Slot(i) for i in range(self.n_slots)]
        # TRUSS_ENGINE_CALL_TIMEOUT: a request that cannot get the engine within this fails with 503 instead of
        # hanging behind a wedged call (H3). Generous: the lock is held for one engine call, and a full-chunk prefill
        # of a long prompt is the longest normal one.
        self.lock = FairGate(1, "engine lock", timeout=float(os.environ.get("TRUSS_ENGINE_CALL_TIMEOUT", 600)))
        # One permit per FREE slot: a 3rd request queues. TRUSS_SLOT_WAIT_TIMEOUT (review F3) bounds that queue:
        # this is the gate the live configuration actually waits on (with the engine running its own slots, the
        # engine lock is never taken), so without a limit one wedged slot queues every later request behind it.
        self.slot_gate = FairGate(self.n_slots, "slot gate", thread_bound=False,
                                  timeout=float(os.environ.get("TRUSS_SLOT_WAIT_TIMEOUT", 600)))
        # TRUSS_PREFILL_PIECE: how big a prompt piece is while another stream decodes. It has to be a multiple of
        # --chunk: measured on the served model (GATE 2, reports/LOG.md), the same 25,170-token prompt cut at 4096
        # produced a different first generated token than one call did, while cut at 8192 or 16384 it produced the
        # same 64 tokens — a piece smaller than an engine chunk moves the row-count boundaries the kernels tile on.
        # 0 = never split, which is how it behaved before slots.
        # the engine runs the slots itself (Forward::prompt / round with per-stage locks) when the binding has the
        # slot-explicit calls: then the server splits nothing (the engine interleaves at its chunks)
        self.engine_slots = self.n_slots > 1 and hasattr(self.model, "spec_step_sampled_s")
        self.prefill_piece = 0 if self.engine_slots else int(os.environ.get("TRUSS_PREFILL_PIECE", chunk))
        self.last_rounds = 0

    @property
    def cached(self):
        """Slot 0's sequence. With TRUSS_SLOTS=1 that is the engine's sequence, which is what scripts/replay.py and
        the host-only tests read; with 2 slots, use the slot record."""
        return self.slots[0].cached

    @cached.setter
    def cached(self, ids):
        self.slots[0].cached = ids

    @property
    def ck_tokens(self):
        return self.slots[0].ck_tokens

    @ck_tokens.setter
    def ck_tokens(self, ids):
        self.slots[0].ck_tokens = ids

    @property
    def n_rounds(self):
        """Decode rounds of the request that finished last (scripts/replay.py's tokens-per-round check)."""
        return self.last_rounds

    def call(self, slot, fn, *args):
        """Every engine call goes through here, so no call can run on another slot's sequence. With several slots the
        engine's slot-explicit call (`<name>_s`) runs WITHOUT a server lock: the engine holds each GPU stage only while
        this slot's work is enqueued there, so two requests' rounds overlap on the two GPUs, and a long prompt gives
        way to the other slot's rounds (TRUSS_DECODE_SHARE). One slot: the fair lock and select(), as before."""
        if self.engine_slots:
            out = getattr(self.model, fn.__name__ + "_s")(slot.idx, *args)
            slot.pos = self.model.position_s(slot.idx)
            return out
        try:
            with self.lock:
                self.model.select(slot.idx)
                out = fn(*args)
                slot.pos = self.model.position
                return out
        except TimeoutError as e:
            raise HTTPException(503, str(e))

    def other_busy(self, slot):
        """Another slot has a request in flight: the prompt read then goes in pieces so that stream keeps its rounds."""
        return any(s.busy and s is not slot for s in self.slots)

    def will_split(self, n, slot):
        return self.prefill_piece > 0 and n > self.prefill_piece and self.other_busy(slot)

    def read(self, slot, ids):
        """eval() on this slot: one call when nothing else is running (full prefill speed), otherwise pieces of
        TRUSS_PREFILL_PIECE with the lock released between them, so the other stream gets a round per piece. The
        head runs on the last row of every call, so where the pieces fall cannot change the result. Returns the
        last piece's logits row. truss_progress() is per model, not per slot, so this logs its own progress."""
        piece = self.prefill_piece
        if not self.will_split(len(ids), slot):
            return self.call(slot, self.model.eval, ids)
        n, t0, out = len(ids), time.time(), None
        for i in range(0, n, piece):
            out = self.call(slot, self.model.eval, ids[i:i + piece])
            done = min(i + piece, n)
            rate = done / max(time.time() - t0, 1e-9)
            self.progress(f"{self.log_tag}: slot {slot.idx} prefill {done}/{n} tokens ({100.0 * done / n:.0f}%), "
                          f"{rate:.0f} tok/s in {piece}-token pieces, "
                          f"ETA {(n - done) / max(rate, 1e-9):.0f} s")
        return out

    def pick_slot(self, prompt_ids):
        """The free slot whose sequence, or checkpoint, is the longest prefix of this prompt; when nothing matches,
        the free slot holding the FEWEST cached tokens (ties: least recently used). Called while holding a
        slot_gate permit, which is what makes a free slot guaranteed.

        A1 (LOG.md "v1.6 merged build live"): the LRU-only rule gave a prompt that matched nothing whatever had
        sat idle longest, which is how a 55-token side request evicted a 126K conversation (3 of that test's 16
        full reads) and made its next turn read 126K again. Least-cached puts the side request on the slot with the
        least to lose while the conversation that is mid-exchange stays free for its own next turn."""
        best, best_rank = None, None
        for s in self.slots:
            if s.busy:
                continue
            match = max(prefix_len(prompt_ids, s.cached), prefix_len(prompt_ids, s.ck_tokens))
            # a slot that matches beats one that does not, then the longest match; with no match anywhere, the slot
            # with the fewest cached tokens to throw away. -last_used keeps the old tie-break.
            rank = (1 if match else 0, match if match else -len(s.cached), -s.last_used)
            if best_rank is None or rank > best_rank:
                best, best_rank = s, rank
        return best

    def acquire_slot(self, prompt_ids, note=""):
        """Take a sequence for a request, waiting FIFO while every slot is busy, and say which one it got. The
        request line from the route arrives in `note` and is logged here, because only now is the slot known. The
        lcp field is the longest common prefix with every slot's cache and checkpoint, reused or not (A1): the
        reuse number in the prefill line only ever says 0 or all of it, which is what hid the reads in the first
        place."""
        t = time.time()
        try:
            self.slot_gate.acquire()
        except TimeoutError as e:   # the same shape the engine lock's timeout already answers with
            raise HTTPException(503, str(e))
        waited = time.time() - t
        slot = self.pick_slot(prompt_ids)
        slot.busy, slot.taken = True, t
        lcp_line = "lcp " + ", ".join(f"s{o.idx} {lcp(prompt_ids, o.cached)}/ck {lcp(prompt_ids, o.ck_tokens)}"
                                      for o in self.slots)
        wait = f", waited {waited:.1f} s for a free sequence" if waited >= 1.0 else ""
        if note:
            self.log(f"{self.log_tag}: {note}, slot {slot.idx}, {lcp_line}{wait}")
        else:
            self.progress(f"{self.log_tag}: slot {slot.idx}, {lcp_line}{wait}")
        return slot

    def release_slot(self, slot):
        """generate()'s finally: the sequence is free for the next request again (client disconnect included), and
        the gate permit goes back so a queued request can take it."""
        slot.busy, slot.last_used = False, time.time()
        self.slot_gate.release()

    def watch_prefill(self, n_read, base, t0, stop, slot=-1):
        """Prints one line per finished prompt chunk while a blocking eval runs on the request thread. Only started
        when that eval holds the engine to itself, since truss_progress() is per model, not per slot."""
        seen = 0
        while not stop.wait(0.5):
            done, total = (self.model.progress_s(slot) if self.engine_slots and slot >= 0
                           else self.model.progress())
            if done < seen:   # a second eval call (the prompt's tail after the checkpoint) restarted the counter
                base, seen = base + seen, 0
            if done <= seen or total <= 0:
                continue
            seen = done
            run = base + done
            rate = run / max(time.time() - t0, 1e-9)
            self.progress(f"{self.log_tag}: slot {slot} prefill {run}/{n_read} tokens ({100.0 * run / n_read:.0f}%), "
                          f"{rate:.0f} tok/s, ETA {(n_read - run) / max(rate, 1e-9):.0f} s")
