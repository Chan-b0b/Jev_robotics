"""Stage 1: can three labels per axis still do the task?

The expert's continuous action becomes, per axis, decrease / hold / increase
(|a| <= tau is hold), and the gripper becomes close / open (effort > 0 is close).
The labels are turned back into an action of magnitude `mag` and executed, with
the expert relabelling every step. If this fails, the action representation is
the problem, before any model is involved.
"""
import argparse
import itertools

import numpy as np
from metaworld.policies import ENV_POLICY_MAP

from stage0_expert import TASKS, make


def to_labels(action, tau):
    """[-1, 0, 1] per axis, and 1 (close) or -1 (open) for the gripper."""
    xyz = np.where(np.abs(action[:3]) <= tau, 0, np.sign(action[:3])).astype(int)
    return xyz, 1 if action[3] > 0 else -1


def run(task, episodes, seed, tau=None, mag=1.0):
    """Success rate and mean steps. tau=None runs the expert unchanged."""
    env = make(task, seed)
    policy = ENV_POLICY_MAP[task]()
    successes, lengths, counts = 0, [], np.zeros((3, 3), int)  # axis x {-1, 0, 1}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(env.unwrapped.max_path_length):
            action = policy.get_action(obs)
            if tau is not None:
                xyz, grip = to_labels(action, tau)
                counts[np.arange(3), xyz + 1] += 1
                action = np.array([*(xyz * mag), grip], dtype=np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            if info["success"]:
                successes += 1
                break
            if terminated or truncated:
                break
        lengths.append(t + 1)
    env.close()
    return successes / episodes, float(np.mean(lengths)), counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--mags", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    args = ap.parse_args()
    for task in TASKS:
        rate, steps, _ = run(task, args.episodes, args.seed)
        print(f"\n{task}  expert: success {rate:.2f}, steps {steps:.0f}")
        print("   tau   mag  success  steps   hold% (x y z)")
        for tau, mag in itertools.product(args.taus, args.mags):
            rate, steps, counts = run(task, args.episodes, args.seed, tau, mag)
            hold = 100 * counts[:, 1] / counts.sum(1)
            print(f"  {tau:4.2f}  {mag:4.2f}  {rate:7.2f}  {steps:5.0f}   "
                  + " ".join(f"{h:3.0f}" for h in hold))
