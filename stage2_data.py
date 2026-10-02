"""Stage 2 data: (state, label) pairs from a labelled expert, with labels a model can be told.

Stage 1 thresholded the expert's command, but each expert scales its error by a
different gain, so the same threshold meant a different distance per task. Here
each task's expert is restated as a subgoal procedure (the same rules as
metaworld.policies), and a label is decrease / hold / increase of the error to
the subgoal, with hold inside DEADBAND. The procedure text is what the model
reads in stage 2, so the labels follow rules the model is given.

`--check` replays the labels (as in stage 1) to confirm they still solve the task.
"""
import argparse
import json
import os

import numpy as np

from stage0_expert import TASKS, make

DEADBAND = 0.005  # m: an axis within this of its subgoal is "hold"
MAG = 0.5  # action per non-hold label: 5 mm, as stage 1 chose

# Each expert, restated: (subgoal, close_gripper) from the parsed state, and the
# same rules in words for the model. Offsets and thresholds are the experts'.
PROCEDURES = {
    "reach-v3": "Move the gripper to the goal position. Keep the gripper open.",
    "push-v3": (
        "Push the puck to the goal. Let P be the puck position shifted by -0.005 in x.\n"
        "1. If the gripper's xy distance to P is more than 0.02, the target is P raised by 0.20 in z.\n"
        "2. Otherwise, if the gripper's z differs from P's z by more than 0.04, the target is P raised by 0.03 in z.\n"
        "3. Otherwise, the target is the goal.\n"
        "Close the gripper when its xy distance to the puck is at most 0.02 and its z is within 0.10 of the "
        "puck's z; otherwise keep it open."),
    "pick-place-v3": (
        "Pick up the puck and place it at the goal. Let P be the puck position shifted by -0.005 in x.\n"
        "1. If the gripper's xy distance to P is more than 0.02, the target is P raised by 0.10 in z.\n"
        "2. Otherwise, if the gripper's z differs from P's z by more than 0.05 and P's z is below 0.04, "
        "the target is P raised by 0.03 in z.\n"
        "3. Otherwise, if the gripper opening is more than 0.73, the target is the gripper's current "
        "position (wait for the gripper to close).\n"
        "4. Otherwise, the target is the goal.\n"
        "Close the gripper when its distance to the puck is less than 0.07; otherwise keep it open."),
    "drawer-open-v3": (
        "Open the drawer by pulling its handle. Let H be the handle position lowered by 0.02 in z.\n"
        "1. If the gripper's xy distance to H is more than 0.06, the target is H raised by 0.30 in z.\n"
        "2. Otherwise, if the gripper's z differs from H's z by more than 0.04, the target is H.\n"
        "3. Otherwise, the target is H moved by -0.06 in y.\n"
        "Keep the gripper open."),
}


def parse(obs):
    return {"hand": obs[:3], "opening": float(obs[3]), "obj": obs[4:7], "goal": obs[-3:]}


def subgoal(task, s):
    hand, obj, goal = s["hand"], s["obj"], s["goal"]
    xy = lambda p: np.linalg.norm(hand[:2] - p[:2])
    if task == "reach-v3":
        return goal, False
    if task == "push-v3":
        p = obj + [-0.005, 0, 0]
        close = xy(obj) <= 0.02 and abs(hand[2] - obj[2]) <= 0.10
        if xy(p) > 0.02:
            return p + [0, 0, 0.2], close
        if abs(hand[2] - p[2]) > 0.04:
            return p + [0, 0, 0.03], close
        return goal, close
    if task == "pick-place-v3":
        p = obj + [-0.005, 0, 0]
        close = np.linalg.norm(hand - obj) < 0.07
        if xy(p) > 0.02:
            return p + [0, 0, 0.1], close
        if abs(hand[2] - p[2]) > 0.05 and p[2] < 0.04:
            return p + [0, 0, 0.03], close
        if s["opening"] > 0.73:
            return hand, close
        return goal, close
    if task == "drawer-open-v3":
        h = obj + [0, 0, -0.02]
        if xy(h) > 0.06:
            return h + [0, 0, 0.3], False
        if abs(hand[2] - h[2]) > 0.04:
            return h, False
        return h + [0, -0.06, 0], False
    raise KeyError(task)


def labels(task, s):
    """[-1, 0, 1] per axis and 1 (close) or 0 (open)."""
    target, close = subgoal(task, s)
    err = target - s["hand"]
    return np.where(np.abs(err) <= DEADBAND, 0, np.sign(err)).astype(int), int(close)


def rollout(task, seed, record=None):
    """Run the labels as the policy. Returns (success, steps)."""
    env = make(task, seed)
    obs, _ = env.reset(seed=seed)
    for t in range(env.unwrapped.max_path_length):
        s = parse(obs)
        xyz, close = labels(task, s)
        if record is not None:
            record.append({"task": task, "seed": seed, "t": t,
                           "hand": s["hand"].round(4).tolist(), "opening": round(s["opening"], 3),
                           "obj": s["obj"].round(4).tolist(), "goal": s["goal"].round(4).tolist(),
                           "dx": int(xyz[0]), "dy": int(xyz[1]), "dz": int(xyz[2]), "grip": close})
        obs, _, terminated, truncated, info = env.step(np.array([*(xyz * MAG), 1 if close else -1], np.float32))
        if info["success"]:
            env.close()
            return True, t + 1
        if terminated or truncated:
            break
    env.close()
    return False, t + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=50, help="replay episodes per task")
    ap.add_argument("--episodes", type=int, default=10, help="recorded episodes per task")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="out/stage2/data.jsonl")
    args = ap.parse_args()
    for task in TASKS:
        runs = [rollout(task, args.seed + 10000 + i) for i in range(args.check)]
        print(f"{task:16s} replay success {np.mean([r[0] for r in runs]):.2f}  steps {np.mean([r[1] for r in runs]):.0f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    for task in TASKS:
        for i in range(args.episodes):
            rollout(task, args.seed + i, rows)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows in {args.out}")
    for task in TASKS:
        tr = [r for r in rows if r["task"] == task]
        dist = {k: np.bincount([r[k] + (k != "grip") for r in tr], minlength=3 if k != "grip" else 2) / len(tr)
                for k in ("dx", "dy", "dz", "grip")}
        print(f"  {task:16s} " + "  ".join(f"{k} " + "/".join(f"{v:.2f}" for v in d) for k, d in dist.items()))
