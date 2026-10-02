"""Stage 2 closed loop: does the task get done when Qwen's reads drive the robot?

Each step reads the four questions of a prompt variant (stage2_eval.py) on the text
state, moves each axis by (P(increase) - P(decrease)) * MAG, and closes the gripper
when P(close) > 0.5, as demo_video.py does. Episodes run concurrently.

Reports per task: success rate, steps, the closest the gripper came to the object
(the goal, for reach), and how often the reads matched the stage 2 label.
"""
import argparse
import asyncio
import json
import os

import numpy as np

from stage0_expert import TASKS, make
from stage2_data import MAG, labels, parse
from stage2_eval import VARIANTS, Reader, questions, user_text


async def episode(reader, task, seed, variant, max_steps):
    env = make(task, seed)
    obs, _ = env.reset(seed=seed)
    closest, matched, reads, success = np.inf, 0, 0, False
    for t in range(max_steps):
        s = parse(obs)
        xyz, close = labels(task, s)
        row = {"task": task, "hand": s["hand"].round(4).tolist(), "opening": round(s["opening"], 3),
               "obj": s["obj"].round(4).tolist(), "goal": s["goal"].round(4).tolist(),
               "dx": int(xyz[0]), "dy": int(xyz[1]), "dz": int(xyz[2]), "grip": close}
        qs = questions(row, variant)
        shown = await asyncio.gather(*[reader.read(user_text(row, text, v), len(opts))
                                       for _, text, opts, _, _, v in qs])
        probs = {}
        for (key, _, opts, true, order, _), (p, _) in zip(qs, shown):
            probs[key] = np.zeros(len(opts))
            probs[key][order] = p
            matched += int(probs[key].argmax() == true)
            reads += 1
        move = [(probs[k][2] - probs[k][0]) * MAG for k in ("dx", "dy", "dz")]
        action = np.array([*move, 1 if probs["grip"][0] > 0.5 else -1], np.float32)
        ref = s["goal"] if task == "reach-v3" else s["obj"]
        closest = min(closest, float(np.linalg.norm(s["hand"] - ref)))
        obs, _, terminated, truncated, info = env.step(action)
        if info["success"]:
            success = True
            break
        if terminated or truncated:
            break
    env.close()
    return {"task": task, "seed": seed, "success": success, "steps": t + 1, "closest": closest,
            "match": matched / reads}


async def main(args):
    reader = Reader(args.url, args.model, args.concurrency)
    await reader.letter_ids()
    tasks = args.tasks or TASKS
    results = await asyncio.gather(*[episode(reader, task, args.seed + i, args.variant, args.max_steps)
                                     for task in tasks for i in range(args.episodes)])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"variant {args.variant}, {args.episodes} episodes per task, at most {args.max_steps} steps -> {args.out}")
    print(f"{'task':16s} {'success':>7s} {'steps':>6s} {'closest (m)':>12s} {'label match':>12s}")
    for task in tasks:
        rs = [r for r in results if r["task"] == task]
        print(f"{task:16s} {np.mean([r['success'] for r in rs]):7.2f} {np.mean([r['steps'] for r in rs]):6.0f} "
              f"{np.median([r['closest'] for r in rs]):12.3f} {np.mean([r['match'] for r in rs]):12.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="goal")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"out/stage2/closed_{args.variant}.jsonl"
    asyncio.run(main(args))
