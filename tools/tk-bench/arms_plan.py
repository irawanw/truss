"""The arms bench's plan task (tools/tk-bench/arms.py --tasks plan): how rich the model's opening plan is on the voxel
scene request, per arm. User 10-24: the shallow pack's simple scenes show in its early thinking ("if the model lists
many items it will create, it is probably fixed"), so this replaces the 85-min r1 + loop pass as the pack screen.

Prompt: an R1 turn (.npz from scripts/r1_build.py: prompt ids ending in <think>, the API's reference reply as target);
default the voxel Eiffel session's first reasoning turn (3f8837ea: the API thinks 1,726 tokens and lists ~30 scene
items). Per prompt the prompt is read once and checkpointed; per arm, --plan-k samples of up to --plan-tokens tokens
with the served sampling (arms_loop.sample: temperature 1.0, min_p 0.05), each stopped at </think> or a loop.
Per --plan-temps temperature (default 1.0, the served one) one set of samples. Score per sample: `items` = distinct scene concepts named in the thinking (SCENE below, word-boundary match, plural
allowed), `anim` = distinct animation / lighting techniques (ANIM), `bullets` = list lines. The API reference's
first --plan-tokens target tokens are scored the same way and printed first: an arm near it plans like the API.
Every sample's text goes to --plan-out/<arm>.json (read them: the counts are a proxy).
"""
import json
import os
import re
import time

import numpy as np

from arms_loop import sample
from server.truss_ctypes import Sampling

R1 = os.path.expanduser("~/ML_projects/dsv41flash/20261019_truss_port/data/r1")
SCENE = ("tower", "leg", "arch", "platform", "spire", "beacon", "antenna", "lattice", "garden", "hedge", "path",
         "flower", "fountain", "tree", "bush", "bench", "lamp", "lantern", "grass", "lawn", "pond", "river", "seine",
         "boat", "bridge", "building", "window", "roof", "chimney", "street", "road", "car", "traffic", "arc",
         "trocadero", "champ", "sky", "star", "moon", "cloud", "firework", "aurora", "fog", "spotlight", "searchlight",
         "sparkle", "firefl", "bird", "butterfl", "people", "pedestrian", "carousel", "ferris", "cafe", "kiosk",
         "statue", "gate", "fence", "water", "reflection", "shadow", "camera", "skyline", "dome", "church", "hill",
         "park", "petal", "leaf", "rose", "tulip", "vine", "ivy", "rock", "stone", "mountain", "ground", "terrain")
ANIM = ("rotat", "flicker", "twinkl", "orbit", "sway", "ripple", "drift", "pulse", "bloom", "glow", "emissive",
        "instanc", "shader", "particle", "fade", "wave", "spin", "sparkl", "shimmer", "fly", "float", "day", "night")


def add_args(ap):
    ap.add_argument("--plan-prompts", default=os.path.join(R1, "3f8837ea.npz"), help="comma list of R1 .npz turns")
    ap.add_argument("--plan-k", type=int, default=8)
    ap.add_argument("--plan-tokens", type=int, default=1800)
    ap.add_argument("--plan-out", default="", help="dir for the sample texts (<arm>.json); empty: not saved")
    ap.add_argument("--plan-temps", default="1.0", help="comma list of sampling temperatures, each a full --plan-k set "
                    "(10-24: does a lower temperature cut the loops of a low-bit pack?); min_p stays 0.05")


def score(text):
    t = text.lower()
    hit = lambda words: sorted({w for w in words if re.search(r"\b" + w + r"[a-z]*", t)})
    items, anim = hit(SCENE), hit(ANIM)
    bullets = len(re.findall(r"^\s*(?:[-*]|\d+[.)])\s+\S", text, re.M))
    return dict(items=len(items), anim=len(anim), bullets=bullets, item_list=items)


def plan_task(b, a):
    close = b.tok.token_id("</think>") if b.tok else -1
    dec = b.tok.tok.decode
    for f in a.plan_prompts.split(","):
        d = np.load(f, allow_pickle=False)
        ids, name = [int(x) for x in d["prompt"]], os.path.basename(f)
        ref = score(dec([int(x) for x in d["target"][:a.plan_tokens]]))
        print(f"task plan {name}: {len(ids)} prompt tokens, {a.plan_k} samples x <= {a.plan_tokens} tokens; API "
              f"reference: items {ref['items']}, anim {ref['anim']}, bullets {ref['bullets']}", flush=True)
        b.m.reset()
        t = time.time()
        b.m.eval(ids[:-1])
        b.m.checkpoint()
        print(f"  read {time.time() - t:.1f} s", flush=True)
        for arm in b.arms:
            b.apply(arm)
            spec = arm.get("spec", True)
            for temp in [float(x) for x in a.plan_temps.split(",")]:
                tag, rows, t = arm["name"] + ("" if temp == 1.0 else f"@t{temp:g}"), [], time.time()
                for k in range(a.plan_k):
                    b.m.restore()
                    sp = Sampling(temp, 1.0, 0, 0.05, (hash(name) + 7919 * k) % (1 << 63))
                    out, lp = sample(b.m, ids[-1], sp, spec, a.plan_tokens, b.stop, close)
                    text = dec(out)
                    rows.append(dict(score(text), tokens=len(out), looped=bool(lp), closed=close in out, text=text))
                m = {k: float(np.mean([r[k] for r in rows])) for k in ("items", "anim", "bullets", "tokens")}
                rec = b.res[arm["name"]]["tasks"].setdefault("plan", dict(ref=ref, plans=[]))   # not "runs": arms.table() sums those as speed runs
                rec["plans"].append(dict(prompt=name, temp=temp, **m, looped=sum(r["looped"] for r in rows), k=a.plan_k))
                empty = sum(r["tokens"] < 32 for r in rows)
                print(f"  {tag:>14}: items {m['items']:.1f} [{min(r['items'] for r in rows)}-"
                      f"{max(r['items'] for r in rows)}] (API {ref['items']}), anim {m['anim']:.1f} (API {ref['anim']}), "
                      f"bullets {m['bullets']:.1f}, {m['tokens']:.0f} tokens per sample, empty {empty}, "
                      f"looped {sum(r['looped'] for r in rows)} of {a.plan_k} ({time.time() - t:.0f} s, "
                      f"{sum(r['tokens'] for r in rows) / (time.time() - t):.1f} tok/s)", flush=True)
                if a.plan_out:
                    os.makedirs(a.plan_out, exist_ok=True)
                    with open(os.path.join(a.plan_out, f"{tag}_{name[:-4]}.json"), "w") as fh:
                        json.dump(dict(ref=ref, temp=temp, samples=rows), fh, indent=1)
