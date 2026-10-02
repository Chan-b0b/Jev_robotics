"""Action rows with an instruction line in L2's words (lang_data.motion_words), to train a policy that
does what the instruction says.

Each picked state of the mt50 train rows gets one of: no instruction (the row as it was), the state's own
expert action as the instruction (the label stays the expert's), or a random instruction (random moves,
strongest first, and a random grip), whose label is the instruction's action: quickly = 1, slightly = 0.5
split over decrease / hold / increase as the soft labels are. The random ones do not match the scene, so
a model only gets them right by reading the instruction.

  python instr_data.py train --per-task 2000 --out out/mt50instr/data/train.jsonl
  python instr_data.py follow --data out/mt50/data/val_comp_a.jsonl --out out/mt50instr/data/val_comp_follow.jsonl
"""
import argparse
import copy
import json

import numpy as np

import prompts_mt50
from lang_data import motion_words

MIX = (0.3, 0.35, 0.35)  # no instruction, the expert's, a random one
LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
LEVEL_P = np.array([0.15, 0.15, 0.4, 0.15, 0.15])


def soft_of(v, grip):
    """Soft labels per key for a move v (3,) and a grip option."""
    out = {k: [max(-x, 0.0), 1 - abs(x), max(x, 0.0)] for k, x in zip(("dx", "dy", "dz"), v)}
    out["grip"] = np.eye(3)[grip].tolist()
    return out


def random_instruction(rng):
    v = rng.choice(LEVELS, 3, p=LEVEL_P)
    grip = int(rng.integers(3))
    soft = soft_of(v, grip)
    return motion_words(soft), soft


def with_instruction(r, instruction, soft=None):
    r = copy.deepcopy(r)
    part = r["messages"][1]["content"][-1]
    i = part["text"].index("\n\nQuestion")
    part["text"] = part["text"][:i] + f"\nInstruction: {instruction}" + part["text"][i:]
    if soft is not None:
        r["soft"] = [round(x, 4) for x in soft[r["key"]]]
        r["messages"][2]["content"] = "ABC"[int(np.argmax(r["soft"]))]
    r["instruction"] = instruction
    return r


def by_state(path):
    rows = [json.loads(l) for l in open(path)]
    return [rows[i:i + 4] for i in range(0, len(rows), 4)]  # dx, dy, dz, grip of one state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["train", "follow"])
    ap.add_argument("--per-task", type=int, default=2000)
    ap.add_argument("--data", help="follow: validation rows to give random instructions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out, kinds = [], np.zeros(3, int)
    if args.kind == "train":
        for t in prompts_mt50.TRAIN:
            states = by_state(f"out/mt50/data/{t}_train.jsonl")
            for i in rng.choice(len(states), min(args.per_task, len(states)), replace=False):
                s, k = states[i], rng.choice(3, p=MIX)
                kinds[k] += 1
                if k == 0:
                    out += s
                elif k == 1:
                    words = motion_words({r["key"]: r["soft"] for r in s})
                    out += [with_instruction(r, words) for r in s]
                else:
                    words, soft = random_instruction(rng)
                    out += [with_instruction(r, words, soft) for r in s]
        rng.shuffle(out)
    else:
        for s in by_state(args.data):
            words, soft = random_instruction(rng)
            out += [with_instruction(r, words, soft) for r in s]
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"{len(out)} rows -> {args.out}" + (f"  (states: none {kinds[0]}, expert {kinds[1]}, random {kinds[2]})"
                                               if args.kind == "train" else ""))
    r = out[0]
    print(r["messages"][1]["content"][-1]["text"].split("\n\nQuestion")[0], "| key", r["key"], "soft", r["soft"])
