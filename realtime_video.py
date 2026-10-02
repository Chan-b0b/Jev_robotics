"""Real-time versions of the closed_loop.py videos, recorded on the sim clock as a
camera on a real robot would.

While Qwen answers a step's questions (sent in parallel, as in closed_loop.py),
the sim runs on with the arm holding still for as long as the requests took;
then the answer moves the arm for one step. Episodes run one at a time, so the
latency is that of a single robot. After success the policy keeps acting and
the recording goes on for --after seconds. Writes <task>_qwen_<seed>_realtime.mp4.
"""
import argparse
import asyncio
import os
import time

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from closed_loop import FONT, Qwen, panel
from stage0_expert import make
from tasks import TASKS, action_of, labels, questions, state_of, user_text


def lift_limits(env):
    """Waiting steps are sim time, not decisions: lift the 500-step episode limits."""
    w = env
    while hasattr(w, "env"):
        if isinstance(w, gym.wrappers.TimeLimit):
            w._max_episode_steps = 10 ** 6
        w = w.env
    w.max_path_length = 10 ** 6


async def episode(task, seed, qwen, max_steps, video, fps, after, camera="corner"):
    env = make(task.name, seed, camera)
    lift_limits(env)
    obs, _ = env.reset(seed=seed)
    qs = questions(task)
    dt, sim_steps, frames = env.unwrapped.dt, 0, []
    shown = {key: np.zeros(n) for key, _, n in qs}  # the reads behind the last executed action
    grip, success_at, latencies = -1.0, None, []
    font = ImageFont.truetype(FONT, 18)

    def sim_step(a, t, truth, status, color):
        """One sim step, and a video frame whenever the sim clock passes the next one."""
        nonlocal obs, sim_steps, success_at
        obs, _, terminated, truncated, info = env.step(a)
        sim_steps += 1
        if success_at is None and info["success"]:
            success_at = sim_steps * dt
        if success_at is not None:
            status, color = f"success at {success_at:.2f} s", "#2a7"
        while len(frames) < sim_steps * dt * fps:
            frame = np.ascontiguousarray(np.rot90(env.render(), 2))  # Meta-World renders rotated 180 degrees
            img = Image.fromarray(panel(frame, f"{task.name} (qwen)", t, shown, truth, a))
            d = ImageDraw.Draw(img)
            text = f"t {sim_steps * dt:5.2f} s  {status}"
            d.rectangle([0, 0, 10 + d.textlength(text, font=font), 30], fill="white")
            d.text((5, 5), text, fill=color, font=font)
            frames.append(np.array(img))
        return terminated or truncated or (success_at is not None and sim_steps * dt >= success_at + after)

    for t in range(max_steps):
        s = state_of(env, obs)
        truth = labels(task, s)
        start = time.perf_counter()
        ps = await asyncio.gather(*[qwen.read(user_text(task, s, text), n) for _, text, n in qs])
        latency = time.perf_counter() - start
        latencies.append(latency)
        probs = {key: p for (key, _, _), p in zip(qs, ps)}
        action = action_of(task, probs)
        hold = np.array([0, 0, 0, grip], np.float32)
        for k in range(int(latency / dt)):
            done = sim_step(hold, t, truth, f"waiting {(k + 1) * dt * 1000:.0f}/{latency * 1000:.0f} ms", "#c44")
            if done:
                break
        else:
            shown, grip = probs, action[3]
            done = sim_step(action, t, truth, "act", "#2a7")
        if done:
            break
    env.close()
    imageio.mimsave(video, frames, fps=fps, macro_block_size=1)
    return success_at, t + 1, sim_steps * dt, np.median(latencies)


async def main(args):
    qwen = Qwen(args.url, args.model, 16)
    await qwen.setup()
    await qwen.read("Reply with A.", 2)  # warm up: the first request can take seconds
    for name in args.tasks:
        for seed in range(args.seed, args.seed + args.episodes):
            video = os.path.join(args.outdir, f"{name}_qwen_{seed}_realtime.mp4")
            success_at, steps, sim_time, latency = await episode(TASKS[name], seed, qwen, args.max_steps, video,
                                                                 args.fps, args.after)
            success = "no" if success_at is None else f"at {success_at:.1f} s"
            print(f"{name:16s} seed {seed}: success {success}, {steps} steps, {sim_time:.1f} s sim time, "
                  f"median latency {latency * 1000:.0f} ms -> {video}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--after", type=float, default=3.0, help="seconds to keep going after success")
    ap.add_argument("--url", default="http://127.0.0.1:8002")
    ap.add_argument("--model", default="multi")
    ap.add_argument("--outdir", default="out/lora/eval_multi_multi")
    asyncio.run(main(ap.parse_args()))
