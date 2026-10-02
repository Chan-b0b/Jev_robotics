"""Per-step inference time of a served image model, one episode at a time on an idle server.

Each step, as closed_loop.py --vision does: render the cameras, encode the frames as
base64 PNG, and send the step's questions at once; the answers drive the robot. Times
each part per step, and also one question at a time (a single request's latency).
"""
import argparse
import asyncio
import time

import numpy as np

import vision
from closed_loop import Qwen
from stage0_expert import make
from tasks import TASKS, action_of, questions


async def episode(qwen, task, seed, steps, cameras, args):
    env = make(task.name, seed)
    obs, _ = env.reset(seed=seed)
    cams = vision.Cameras(env, cameras)
    qs = questions(task)
    t_render, t_encode, t_step, t_single = [], [], [], []
    for t in range(steps):
        a = time.perf_counter()
        shots = cams.frames()
        b = time.perf_counter()
        urls = [vision.encode(f) for f in shots] if args.encode_once else shots
        users = [vision.for_vllm(vision.user_content(task.name, q, cameras), urls) for _, q, _ in qs]
        c = time.perf_counter()
        if args.first_alone:  # the first question fills the prefix cache, the rest then share it
            ps = [await qwen.read(users[0], qs[0][2])]
            ps += await asyncio.gather(*[qwen.read(u, n) for u, (_, _, n) in zip(users[1:], qs[1:])])
        else:
            ps = await asyncio.gather(*[qwen.read(u, n) for u, (_, _, n) in zip(users, qs)])
        d = time.perf_counter()
        t_render.append(b - a), t_encode.append(c - b), t_step.append(d - c)
        if t % 10 == 5:  # one question alone, now and then
            e = time.perf_counter()
            await qwen.read(users[0], qs[0][2])
            t_single.append(time.perf_counter() - e)
        obs, _, terminated, truncated, info = env.step(action_of(task, {k: p for (k, _, _), p in zip(qs, ps)}))
        if info["success"] or terminated or truncated:
            break
    cams.close()
    env.close()
    return len(qs), t_render, t_encode, t_step, t_single


async def main(args):
    qwen = Qwen(args.url, args.model, 16)
    await qwen.setup()
    await episode(qwen, TASKS[args.tasks[0]], args.seed, 3, args.cameras, args)  # warm up
    print(f"model {args.model} at {args.url}, cameras {' '.join(args.cameras)}, frames encoded "
          f"{'once per step' if args.encode_once else 'per question'}, questions "
          f"{'first alone, then the rest at once' if args.first_alone else 'all at once'}, one episode per task, "
          f"times in ms (median / p90)")
    print(f"{'task':16s} {'questions':>9s} {'steps':>5s} {'render':>11s} {'encode':>11s} {'server':>11s} "
          f"{'step total':>11s} {'1 question':>11s} {'Hz':>5s}")
    ms = lambda x: f"{np.median(x) * 1e3:5.0f}/{np.percentile(x, 90) * 1e3:5.0f}"
    for name in args.tasks:
        n, r, e, s, one = await episode(qwen, TASKS[name], args.seed, args.steps, args.cameras, args)
        total = np.array(r) + np.array(e) + np.array(s)
        print(f"{name:16s} {n:9d} {len(s):5d} {ms(r)} {ms(e)} {ms(s)} {ms(total)} {ms(one)} {1 / np.median(total):5.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8004")
    ap.add_argument("--model", default="vmulti")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--cameras", nargs="+", default=list(vision.DEFAULT), choices=list(vision.RES))
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--encode-once", action="store_true", help="encode the frames once per step, not per question")
    ap.add_argument("--first-alone", action="store_true", help="send the first question alone, then the rest at once")
    asyncio.run(main(ap.parse_args()))
