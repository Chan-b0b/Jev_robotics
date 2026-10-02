"""Accuracy per task and question of a served model (base or LoRA) on lora_data.py rows.

Reads each row as closed_loop.py does (one token, softmax over the option letters)
and compares the argmax with the row's letter. Image rows send their PNG frames.
"""
import argparse
import asyncio
import json
from collections import defaultdict

import numpy as np
from PIL import Image

import vision
from closed_loop import Qwen
from tasks import LETTERS


async def main(args):
    rows = [json.loads(l) for l in open(args.data)]
    qwen = Qwen(args.url, args.model, args.concurrency)
    await qwen.setup()
    n_opts = lambda r: len(r["soft"]) if "soft" in r else 2 if r["key"] == "grip" else 3  # ML10's grip has 3
    user = lambda r: (vision.for_vllm(r["messages"][1]["content"], [np.array(Image.open(f)) for f in r["images"]])
                      if "images" in r else r["messages"][1]["content"])
    probs = []
    for i in range(0, len(rows), args.concurrency):  # in chunks, so no request waits long for a connection
        probs += await asyncio.gather(*[qwen.read(user(r), n_opts(r), r["messages"][0]["content"])
                                        for r in rows[i:i + args.concurrency]])
    soft3 = {(r["task"], r["key"]) for r in rows if len(r.get("soft", ())) == 3}
    hits, conf = defaultdict(list), defaultdict(lambda: np.zeros((3, 3), int))
    for r, p in zip(rows, probs):
        truth = LETTERS.index(r["messages"][2]["content"])
        conf[(r["task"], r["key"])][truth, p.argmax()] += 1
        ok = int(p.argmax() == truth)
        hits[(r["task"], r["key"])].append(ok)
        hits[(r["task"], "all")].append(ok)
    print(f"model {args.model} on {args.data} ({len(rows)} rows): overall {np.mean([h for k, v in hits.items() if k[1] != 'all' for h in v]):.4f}")
    for task in dict.fromkeys(r["task"] for r in rows):
        keys = [k for t, k in hits if t == task and k != "all"] + ["all"]
        print(f"  {task:16s} " + "  ".join(f"{k} {np.mean(hits[(task, k)]):.3f}" for k in keys))
    if args.confusion:
        print("confusion, rows = label, columns = answer (A B C = dec hold inc, or close open)")
        for (task, key), c in conf.items():
            n = 3 if (task, key) in soft3 else 2 if key == "grip" else 3
            print(f"  {task:16s} {key:4s} " + " | ".join(" ".join(f"{v:5d}" for v in row[:n]) for row in c[:n]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/lora/multi/val.jsonl")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--confusion", action="store_true", help="also print the label x answer counts")
    args = ap.parse_args()
    asyncio.run(main(args))
