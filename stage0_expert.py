"""Stage 0: Meta-World expert policies succeed, and the cameras show gripper and object.

Runs each task's scripted policy for a few episodes, reports the success rate,
and saves the first frame of each task from every candidate camera.
"""
import argparse
import os

import gymnasium as gym
import metaworld  # noqa: F401  (registers Meta-World/MT1)
import numpy as np
from metaworld.policies import ENV_POLICY_MAP
from PIL import Image

TASKS = ["reach-v3", "push-v3", "pick-place-v3", "drawer-open-v3"]
CAMERAS = ["corner", "corner2", "corner3", "topview", "behindGripper", "gripperPOV"]


def make(task, seed, camera=None):
    return gym.make("Meta-World/MT1", env_name=task, seed=seed,
                    render_mode="rgb_array" if camera else None, camera_name=camera,
                    width=320, height=320)


def run_expert(task, episodes, seed):
    env = make(task, seed)
    policy = ENV_POLICY_MAP[task]()
    successes, lengths = 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for t in range(env.unwrapped.max_path_length):
            obs, _, terminated, truncated, info = env.step(policy.get_action(obs))
            if info["success"]:
                successes += 1
                break
            if terminated or truncated:
                break
        lengths.append(t + 1)
    env.close()
    return successes / episodes, float(np.mean(lengths))


def save_frames(task, seed, out):
    for cam in CAMERAS:
        env = make(task, seed, cam)
        env.reset(seed=seed)
        # Meta-World's offscreen render comes out rotated 180 degrees
        Image.fromarray(np.rot90(env.render(), 2)).save(os.path.join(out, f"{task}_{cam}.png"))
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/stage0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for task in TASKS:
        rate, steps = run_expert(task, args.episodes, args.seed)
        print(f"{task:16s} success {rate:.2f}  mean steps {steps:.0f}")
    for task in TASKS:
        save_frames(task, args.seed, args.out)
    print(f"frames in {args.out}/")
