"""Stage 2 eval: can Qwen read the right label from coordinates given as text?

Each state in data.jsonl becomes four questions (dx, dy, dz, grip). Each question
is one chat call with max_tokens=1, and the answer is the softmax over the
option letters' logprobs at that token, as JevK5 reads a choice. Reports, per
task and question: accuracy against the label, the majority-label baseline,
NLL, ECE and mean confidence.

Run against a vLLM server: python stage2_eval.py [--limit N]
"""
import argparse
import asyncio
import json
import math
import os
from collections import defaultdict

import httpx
import numpy as np

from stage2_data import DEADBAND, MAG, PROCEDURES, subgoal

SYSTEM = ("You control a robot gripper in a tabletop simulation. Positions are (x, y, z) in meters; "
          "z is height. Each step, the gripper moves {mag} m along each axis you choose to move, "
          "or holds that axis.").format(mag=MAG / 100)
RULE = ("On each axis, move toward the target: decrease if the target's coordinate is smaller by more than "
        f"{DEADBAND}, increase if it is larger by more than {DEADBAND}, otherwise hold.")
OBJECT = {"reach-v3": None, "push-v3": "puck", "pick-place-v3": "puck", "drawer-open-v3": "drawer handle"}
DIFF_RULE = (f"For each axis, look at target minus gripper on that axis: below -{DEADBAND} means decrease, "
             f"above +{DEADBAND} means increase, between -{DEADBAND} and +{DEADBAND} means hold.")
MM = round(DEADBAND * 1000)
DIFF_MM_RULE = (f"For each axis, look at target minus gripper on that axis, in millimeters: below -{MM} means decrease, "
                f"above +{MM} means increase, from -{MM} to +{MM} means hold.")
AXES = {"dx": "x", "dy": "y", "dz": "z"}
LETTERS = ["A", "B", "C"]


# Prompt variants for the axis questions, to find where the base prompt fails:
# base; reversed (option order flipped); target (the subgoal is given, no
# procedure); diff (target minus gripper is given). Grip questions use base.
# goal: the task's goal and the state only, no procedure, for all four questions;
# the labels then follow a strategy the model is not told, so closed-loop success
# (stage2_closed.py) is the measure and label agreement is only a reference.
VARIANTS = ("base", "reversed", "target", "diff", "diff_mm", "goal")
GOALS = {
    "reach-v3": "Move the gripper to the goal position.",
    "push-v3": "Push the puck along the table to the goal position.",
    "pick-place-v3": "Pick up the puck and place it at the goal position.",
    "drawer-open-v3": "Open the drawer by pulling its handle. The drawer opens toward -y.",
}


def target_of(row):
    s = {"hand": np.array(row["hand"]), "opening": row["opening"], "obj": np.array(row["obj"]),
         "goal": np.array(row["goal"])}
    return subgoal(row["task"], s)[0]


def user_text(row, question, variant="base"):
    if variant == "goal":
        lines = [f"Task: {GOALS[row['task']]}", "", "State:", f"gripper position: {tuple(row['hand'])}",
                 f"gripper opening: {row['opening']} (1.0 is fully open, 0.0 is closed)"]
        if OBJECT[row["task"]]:
            lines.append(f"{OBJECT[row['task']]} position: {tuple(row['obj'])}")
        if row["task"] != "drawer-open-v3":
            lines.append(f"goal position: {tuple(row['goal'])}")
        return "\n".join(lines + ["", question, "Reply with only the letter."])
    if variant == "diff_mm":
        d = (target_of(row) - np.array(row["hand"])) * 1000
        return "\n".join(["Task: Move the gripper toward the target.", "", DIFF_MM_RULE, "", "State:",
                          "target minus gripper (mm): (" + ", ".join(f"{v:+.0f}" for v in d) + ")",
                          "", question, "Reply with only the letter."])
    if variant in ("target", "diff"):
        target = target_of(row)
        lines = ["Task: Move the gripper toward the target.", "", DIFF_RULE if variant == "diff" else RULE, "", "State:",
                 f"gripper position: {tuple(row['hand'])}"]
        if variant == "target":
            lines.append(f"target position: {tuple(round(float(v), 4) for v in target)}")
        else:
            lines.append("target minus gripper: ("
                         + ", ".join(f"{v:+.4f}" for v in target - np.array(row["hand"])) + ")")
        lines += ["", question, "Reply with only the letter."]
        return "\n".join(lines)
    lines = [f"Task: {PROCEDURES[row['task']]}", "", RULE, "", "State:",
             f"gripper position: {tuple(row['hand'])}", f"gripper opening: {row['opening']}"]
    if OBJECT[row["task"]]:
        lines.append(f"{OBJECT[row['task']]} position: {tuple(row['obj'])}")
    if row["task"] != "drawer-open-v3":
        lines.append(f"goal position: {tuple(row['goal'])}")
    lines += ["", question, "Reply with only the letter."]
    return "\n".join(lines)


def questions(row, variant="base"):
    """(key, text, options, true option index, order, variant) for one state. ``order``
    lists the semantic option shown at each letter; options and the true index are
    semantic (decrease, hold, increase / close, open)."""
    out = []
    for key, axis in AXES.items():
        opts = [f"decrease {axis}", f"hold {axis}", f"increase {axis}"]
        order = [2, 1, 0] if variant == "reversed" else [0, 1, 2]
        text = f"Question: What should the gripper do along {axis}?\n" + "\n".join(
            f"{l}: {opts[i]}" for l, i in zip(LETTERS, order))
        out.append((key, text, opts, row[key] + 1, order, variant))
    opts = ["close the gripper", "open the gripper"]
    text = "Question: What should the gripper do with its fingers?\n" + "\n".join(
        f"{l}: {o}" for l, o in zip(LETTERS, opts))
    out.append(("grip", text, opts, 0 if row["grip"] else 1, [0, 1], "goal" if variant == "goal" else "base"))
    return out


class Reader:
    def __init__(self, url, model, concurrency):
        self.http = httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(120.0, connect=5.0),
                                      limits=httpx.Limits(max_connections=concurrency))
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)

    async def letter_ids(self):
        ids = []
        for letter in LETTERS:
            r = await self.http.post("/tokenize", json={"model": self.model, "prompt": letter,
                                                        "add_special_tokens": False})
            toks = r.json()["tokens"]
            if len(toks) != 1:
                raise ValueError(f"letter {letter!r} is {len(toks)} tokens")
            ids.append(toks[0])
        self.ids = ids
        return ids

    async def read(self, user, n):
        ids = self.ids[:n]
        async with self.sem:
            r = await self.http.post("/v1/chat/completions", json={
                "model": self.model,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
                "logprob_token_ids": ids, "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        top = {int(t["token"].split(":")[1]): t["logprob"]
               for t in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        missing = [i for i in ids if i not in top]
        if missing:
            raise ValueError(f"label ids {missing} missing from the returned logprobs")
        lp = np.array([top[i] for i in ids])
        p = np.exp(lp - lp.max())
        # mass on the letters, before renormalising: how much the model wanted to answer at all
        return p / p.sum(), float(np.exp(lp).sum())


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return total


async def main(args):
    rows = [json.loads(l) for l in open(args.data)]
    if args.limit:
        rng = np.random.default_rng(0)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.limit, replace=False))]
    reader = Reader(args.url, args.model, args.concurrency)
    print("letter ids", await reader.letter_ids())

    async def one(row, q):
        key, text, opts, true, order, variant = q
        shown, mass = await reader.read(user_text(row, text, variant), len(opts))
        p = np.zeros(len(opts))
        p[order] = shown  # back to semantic order
        return {"task": row["task"], "seed": row["seed"], "t": row["t"], "q": key, "true": true,
                "probs": p.round(4).tolist(), "letter": int(shown.argmax()), "mass": round(mass, 4)}

    preds = await asyncio.gather(*[one(r, q) for r in rows for q in questions(r, args.variant)])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    groups = defaultdict(list)
    for p in preds:
        groups[(p["task"], p["q"])].append(p)
        groups[("all", p["q"])].append(p)
    axis = [p for p in preds if p["q"] != "grip"]
    print(f"\n{len(rows)} states, {len(preds)} reads, variant {args.variant} -> {args.out}")
    print("axis answers by letter A/B/C:", "/".join(f"{v:.2f}" for v in np.bincount([p["letter"] for p in axis], minlength=3) / len(axis)),
          "  by meaning dec/hold/inc:", "/".join(f"{v:.2f}" for v in np.bincount([int(np.argmax(p["probs"])) for p in axis], minlength=3) / len(axis)))
    print(f"{'task':16s} {'q':4s} {'acc':>5s} {'base':>5s} {'nll':>5s} {'ece':>5s} {'conf':>5s} {'mass':>5s}  predicted (per true label)")
    for (task, q), ps in sorted(groups.items(), key=lambda kv: (kv[0][0] == "all", kv[0])):
        probs = [np.array(p["probs"]) for p in ps]
        true = np.array([p["true"] for p in ps])
        pred = np.array([pr.argmax() for pr in probs])
        conf = [pr.max() for pr in probs]
        base = np.bincount(true).max() / len(true)
        nll = -np.mean([math.log(max(pr[t], 1e-9)) for pr, t in zip(probs, true)])
        n = len(probs[0])
        confusion = " ".join("/".join(str(int(((true == t) & (pred == k)).sum())) for k in range(n))
                             for t in range(n))
        print(f"{task:16s} {q:4s} {np.mean(pred == true):5.2f} {base:5.2f} {nll:5.2f} "
              f"{ece(conf, pred == true):5.2f} {np.mean(conf):5.2f} {np.mean([p['mass'] for p in ps]):5.2f}  {confusion}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/stage2/data.jsonl")
    ap.add_argument("--out", default="out/stage2/preds.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="random subset of states")
    ap.add_argument("--variant", choices=VARIANTS, default="base")
    asyncio.run(main(ap.parse_args()))
