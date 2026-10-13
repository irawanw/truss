"""Host-only test (no GPU) of the raw token-id `prompt` field, review F1.

`{"prompt": [...]}` (a598f0e) lets a client supply the token ids itself, which moves the trust boundary from the
tokenizer to the network: the ids reach `rows_kernel`, which takes the embedding row at `(size_t) id` with no bound,
so one id >= n_vocab reads past the end of the table and a negative one becomes an address of ~1.8e19. That is an
illegal address, a CUDA context does not recover from one, and both slots go down until a restart - so the request
has to be refused here, before a slot is taken and before a response starts.

  python3 tests/server/raw_prompt_test.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
from server import app  # noqa: E402
from slot_serving_test import FakeSlots  # noqa: E402

m = FakeSlots()          # the fake engine: raw_ids takes its n_vocab exactly as the route passes engine.model.n_vocab
V = m.n_vocab
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{name}: {detail}  {'PASS' if cond else 'FAIL'}")


def refuse(raw, why):
    """raw_ids must answer ValueError - the app maps that to a 400. Anything else (or acceptance) is the bug."""
    try:
        got = app.raw_ids(raw, V)
    except ValueError as e:
        return "" if why in str(e) else f"wrong message: {e}"
    except Exception as e:   # noqa: BLE001 - a TypeError/500 is exactly what F1 turns a bad request into
        return f"{type(e).__name__} instead of ValueError: {e}"
    return f"accepted as {got}"


def test_accepted():
    """The ids a real client sends must come through unchanged, as ints."""
    ids = [0, 1, 99999, V - 1]
    check("a valid id list passes through", app.raw_ids(ids, V) == ids, f"[0, 1, 99999, n_vocab-1] -> {ids}")
    check("and stays a list of plain ints", all(type(x) is int for x in app.raw_ids([5, 7], V)),
          f"{[type(x).__name__ for x in app.raw_ids([5, 7], V)]}")
    long_ids = list(range(50000))   # a real conversation's prompt: the check must not change it
    check("a 50K-token prompt is unchanged", app.raw_ids(long_ids, V) == long_ids, "same 50,000 ids")


def test_refused():
    """Every one of these used to reach the embedding table."""
    for raw, why in (([], "must not be empty"),                       # nothing to sample; the engine said 500
                     ([99999999], f"in [0, {V})"),                    # past the end of the table
                     ([V], f"in [0, {V})"),                           # the first id out of range, exactly
                     ([-1], f"in [0, {V})"),                          # (size_t) -1 is ~1.8e19
                     ([1, 2, -5], f"in [0, {V})"),                    # one bad id among good ones
                     ([1.5], "flat list of ints"),                    # int() used to truncate this to 1
                     (["7"], "flat list of ints"),                    # a string id
                     ([True], "flat list of ints"),                   # bool is not an id
                     ([1, None], "flat list of ints"),                # a hole in the prompt
                     ([[1, 2]], "flat list of ints"),                 # nested, not a prompt
                     ({"tokens": [1]}, "flat list of ints")):         # an object where a list belongs
        detail = refuse(raw, why)
        check(f"refuses {raw!r}", not detail, detail or f"ValueError mentioning {why!r}")


def main():
    for t in (test_accepted, test_refused):
        print(f"-- {t.__name__}")
        t()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
