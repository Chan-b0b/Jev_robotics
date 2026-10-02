"""LoRA data: states labelled by tasks.labels, as the chat rows closed_loop.py sends.

reach-v3: labels depend only on the fingertip and goal positions, so states are
drawn directly instead of from rollouts. The goal is uniform over reach-v3's goal
range. Per axis, the fingertips are either uniform over the box around the start
and goal ranges, or within NEAR of the goal, so hold and the deadband edges are common.

The other tasks: the puck, the opening and the fingertips move together (the puck
is only in the air when held), so states come from rollouts of the labels as the
policy, with a random action instead on a fraction EPS of steps, so that the data
also holds states off the labels' path (a dropped puck, a missed grasp).

Positions are rounded as the prompt shows them before labelling. Each state gives
one row per question, with the true letter as the assistant reply.

--vision: the prompt shows camera frames instead of coordinates (vision.py), so
every task, reach included, comes from rollouts; the frames are saved as PNG
next to the jsonl and the rows name them (--cameras picks the cameras). Reach's
rollouts run VISION_STEPS steps without stopping at success, and each starts the
hand at a uniform point of HAND_LOW-HAND_HIGH instead of reach-v3's fixed
(0, 0.6, 0.2): from the fixed start and stopped at success, the gripper never
reaches the goal's y (the goals are at y 0.8-0.9), so every dy label is increase.
The other tasks' rollouts are those of the text data.

--suite ml10 (with --vision): the tasks of tasks.ML10, labelled softly by their experts
(each row also keeps its soft label), and each row's task text drawn from prompts.py:
a description, plus a procedure half the time (--procedure mix), never (none) or always (all).
--suite mt50 does the same with prompts_mt50.py and tasks.SYSTEM_IMAGE; --look train|test
gives an episode a random look (looks.py) with probability --look-p.
"""
import argparse
import json
import os

import numpy as np

from PIL import Image

import looks
import prompts
import prompts_mt50
import vision
from stage0_expert import make
from tasks import GRIP_LEVELS, LETTERS, SUITES, SYSTEMS, TASKS, action_of, labels, questions, state_of, user_text

GOAL_LOW, GOAL_HIGH = np.array([-0.1, 0.8, 0.05]), np.array([0.1, 0.9, 0.3])  # reach-v3
TIP_LOW, TIP_HIGH = np.array([-0.2, 0.5, 0.03]), np.array([0.2, 1.0, 0.35])  # start (0, 0.6, 0.15) and goals, with margin
NEAR = 0.03  # m
P_NEAR = 0.5  # per axis
EPS = 0.3  # rollouts: fraction of random steps
MAX_STEPS = 300
VISION_STEPS = 100
PROMPTS = {"ml10": prompts, "mt50": prompts_mt50}
HAND_LOW, HAND_HIGH = np.array([-0.2, 0.5, 0.08]), np.array([0.2, 1.0, 0.35])  # --vision: start of the hand


def sample(rng):
    goal = rng.uniform(GOAL_LOW, GOAL_HIGH)
    tip = np.where(rng.random(3) < P_NEAR, goal + rng.uniform(-NEAR, NEAR, 3), rng.uniform(TIP_LOW, TIP_HIGH))
    return {"tip": tip.round(4), "goal": goal.round(4)}


def shown(s):
    return {"tip": s["tip"].round(4), "opening": round(s["opening"], 3), "obj": s["obj"].round(4),
            "goal": s["goal"].round(4), "obs": s["obs"]}


def rollout_states(task, n, seed, rng, frames=None, max_steps=MAX_STEPS, stop_at_success=True, random_start=False,
                   cameras=vision.DEFAULT, look=None, look_p=1.0):
    """States from noisy rollouts until there are n; with a frames list, also each state's camera frames."""
    out, qs = [], questions(task)
    while len(out) < n:
        env = make(task.name, seed)
        if random_start:
            env.unwrapped.hand_init_pos = rng.uniform(HAND_LOW, HAND_HIGH)
        obs, _ = env.reset(seed=seed)
        if look and rng.random() < look_p:
            looks.randomize(env, rng, look)
        cams = None if frames is None else vision.Cameras(env, cameras)
        for _ in range(max_steps):
            s = shown(state_of(env, obs))
            out.append(s)
            if cams:
                frames.append(cams.frames())
            soft = getattr(task, "soft", False)
            if rng.random() < EPS:
                action = np.array([*rng.uniform(-1, 1, 3), rng.choice(GRIP_LEVELS if soft else [-1, 1])], np.float32)
            elif soft:
                action = action_of(task, task.soft_labels(s))
            else:
                truth = labels(task, s)
                action = action_of(task, {k: np.eye(m)[truth[k]] for k, _, m in qs})
            obs, _, terminated, truncated, info = env.step(action)
            if (stop_at_success and info["success"]) or terminated or truncated or len(out) >= n:
                break
        if cams:
            cams.close()
        env.close()
        seed += 1
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), default="default", help="task registry of tasks.py")
    ap.add_argument("--task", choices=sorted({n for t in SUITES.values() for n in t}), default="reach-v3")
    ap.add_argument("--procedure", choices=["mix", "none", "all"], default="mix", help="--suite ml10/mt50: procedures in the text")
    ap.add_argument("--look", choices=looks.SPLITS, default=None, help="random looks (looks.py) of this split")
    ap.add_argument("--look-p", type=float, default=1.0, help="the fraction of episodes given a random look")
    ap.add_argument("--states", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/lora/reach_train.jsonl")
    ap.add_argument("--vision", action="store_true", help="camera frames instead of coordinates")
    ap.add_argument("--cameras", nargs="+", default=list(vision.DEFAULT), choices=list(vision.RES))
    args = ap.parse_args()
    task, rng = SUITES[args.suite][args.task], np.random.default_rng(args.seed)
    soft = getattr(task, "soft", False)
    assert args.vision or not soft, "soft (ml10) tasks have image prompts only"
    text_rng = np.random.default_rng(args.seed + 12345)
    frames = [] if args.vision else None
    if args.task == "reach-v3" and not args.vision:
        states = [sample(rng) for _ in range(args.states)]
    else:
        vision_reach = args.vision and args.task == "reach-v3"
        states = rollout_states(task, args.states, 100000 + 1000 * args.seed, rng, frames,
                                *((VISION_STEPS, False, True) if vision_reach else ()), cameras=args.cameras,
                                look=args.look, look_p=args.look_p)
    qs = questions(task)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img_dir = os.path.splitext(args.out)[0] + "_img"
    if args.vision:
        os.makedirs(img_dir, exist_ok=True)
    counts = {k: np.zeros(n) for k, _, n in qs}
    with open(args.out, "w") as f:
        for i, s in enumerate(states):
            truth = labels(task, s)
            extra = {}
            if args.vision:
                extra["images"] = [os.path.join(img_dir, f"{i:06d}_{cam}.png") for cam in args.cameras]
                for path, frame in zip(extra["images"], frames[i]):
                    Image.fromarray(frame).save(path, compress_level=1)
            soft_labels = task.soft_labels(s) if soft else None
            for key, text, _ in qs:
                counts[key] += soft_labels[key] if soft else np.eye(len(counts[key]))[truth[key]]
                if soft:
                    extra["soft"] = soft_labels[key].round(4).tolist()
                    procedure = {"mix": text_rng.random() < 0.5, "none": False, "all": True}[args.procedure]
                    user = vision.user_content(task.name, text, args.cameras, PROMPTS[args.suite].text(task.name, text_rng, procedure))
                elif args.vision:
                    user = vision.user_content(task.name, text, args.cameras)
                else:
                    user = user_text(task, s, text)
                f.write(json.dumps({"task": task.name, "key": key,
                                    **{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()},
                                    **extra, **({"cameras": args.cameras} if args.vision else {}), "messages": [
                    {"role": "system", "content": SYSTEMS[args.suite]},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": LETTERS[truth[key]]}]}) + "\n")
    print(f"{len(states) * len(qs)} rows in {args.out}")
    for k, c in counts.items():
        print(f"  {k} " + "/".join(f"{v:.2f}" for v in c / c.sum()))
