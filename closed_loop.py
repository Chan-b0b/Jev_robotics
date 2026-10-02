"""Closed loop with the per-task prompts of tasks.py.

--policy labels acts on the labels (to check that they solve the task);
--policy qwen reads each question from Qwen and acts on the probabilities.
Every step rebuilds the prompt from the simulator's current state. Episodes run
concurrently. --video N saves the first N episodes as mp4 with the reads beside
the frame.

Reports: success rate, steps, final fingertip-to-goal distance, label agreement.
Every step (state, target error, probabilities, labels) goes to steps_<policy>.jsonl.
--after-success N keeps acting N steps past success, so a video shows the rest.
--condition changes the scene's look (vision.apply) for a robustness check.
--instruct oracle adds the expert's action in words (instr_data.py's instruction line) each step.
--suite ml10 --vision: each episode's task text is drawn from prompts.py (by its seed), a
description only (--procedure none) or with a procedure (all).
--vision asks with the camera frames of vision.py instead of the coordinates.
--realtime records the video on the sim clock (30 fps), as demo_video.py does: while
the step's reads are made (rendering, encoding and the requests, timed), the sim runs
on with the arm holding still for that long, then the answer moves it for one step.
Run one episode at a time for it, or the requests of other episodes add to the wait.
"""
import argparse
import asyncio
import json
import os
import time

import gymnasium as gym
import httpx
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import looks
import prompts
import prompts_mt50
from lang_data import motion_words
import vision
from stage0_expert import make
from tasks import LETTERS, SUITES, SYSTEM, SYSTEMS, TASKS, action_of, labels, questions, state_of, user_text

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
OPTION_NAMES = {3: ["dec", "hold", "inc"], 2: ["close", "open"]}
SOFT_GRIP_NAMES = ["open", "grasp", "tight"]  # a soft task's 3 grip options
RETRIES = 3


class Qwen:
    def __init__(self, url, model, concurrency):
        self.http = httpx.AsyncClient(base_url=url, timeout=httpx.Timeout(120.0, connect=5.0),
                                      limits=httpx.Limits(max_connections=concurrency))
        self.model = model
        self.retries = 0
        self.system = SYSTEM  # main sets its suite's

    async def setup(self):
        self.ids = []
        for letter in LETTERS:
            r = await self.http.post("/tokenize", json={"model": self.model, "prompt": letter, "add_special_tokens": False})
            toks = r.json()["tokens"]
            assert len(toks) == 1, f"letter {letter!r} is {len(toks)} tokens"
            self.ids.append(toks[0])

    async def read(self, user, n, system=None):
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system or self.system}, {"role": "user", "content": user}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
                "logprob_token_ids": self.ids[:n], "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        for attempt in range(RETRIES + 1):
            try:
                r = await self.http.post("/v1/chat/completions", json=body)
                break
            except httpx.TransportError:  # a dropped connection, seen with 150+ concurrent image episodes
                if attempt == RETRIES:
                    raise
                self.retries += 1
                await asyncio.sleep(1 + attempt)
        r.raise_for_status()
        top = {int(t["token"].split(":")[1]): t["logprob"]
               for t in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        lp = np.array([top[i] for i in self.ids[:n]])
        p = np.exp(lp - lp.max())
        return p / p.sum()


def panel(frame, title, t, probs, truth, action, status=None):
    h = 480
    w = round(frame.shape[1] * h / frame.shape[0])
    img = Image.new("RGB", (w + 360, h), "white")
    img.paste(Image.fromarray(frame).resize((w, h)), (0, 0))
    d = ImageDraw.Draw(img)
    big, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 14)
    x0, y = w + 15, 64
    d.text((x0, 10), f"{title}  step {t}", fill="black", font=big)
    if status:
        d.text((x0, 36), status[0], fill=status[1], font=small)
    for key, p in probs.items():
        d.text((x0, y), key, fill="black", font=big)
        y += 24
        for k, name in enumerate(SOFT_GRIP_NAMES if key == "grip" and len(p) == 3 else OPTION_NAMES[len(p)]):
            color = "#2a7" if k == truth[key] else "#c44"
            d.text((x0, y), f"{name:>5s}", fill="black", font=small)
            d.rectangle([x0 + 50, y + 2, x0 + 50 + int(200 * p[k]), y + 14], fill=color)
            d.text((x0 + 258, y), f"{p[k]:.2f}", fill="black", font=small)
            if k == truth[key]:
                d.rectangle([x0 + 48, y, x0 + 252, y + 16], outline="black")
            y += 18
        y += 8
    d.text((x0, y + 4), "action (m): " + " ".join(f"{a / 100:+.4f}" for a in action[:3]), fill="black", font=small)
    d.text((x0, y + 22), f"grip: {'close' if action[3] > 0 else 'open'}", fill="black", font=small)
    d.text((x0, h - 40), "green = label (boxed)", fill="#2a7", font=small)
    d.text((x0, h - 22), "red = other options", fill="#c44", font=small)
    return np.array(img)


async def episode(task, seed, policy, qwen, max_steps, video=None, camera="corner", use_vision=False,
                  cameras=vision.DEFAULT, upside_down=vision.UPSIDE_DOWN, after_success=0,
                  condition=None, realtime_fps=None, procedure="none", suite="default", look=None, instruct=None):
    env = make(task.name, seed, camera if video else None)
    if realtime_fps:  # waiting steps are sim time, not decisions: lift the step limits
        w = env
        while hasattr(w, "env"):
            if isinstance(w, gym.wrappers.TimeLimit):
                w._max_episode_steps = 10 ** 6
            w = w.env
        w.max_path_length = 10 ** 6
    obs, _ = env.reset(seed=seed)
    text = None
    if use_vision and getattr(task, "soft", False):
        text = {"ml10": prompts, "mt50": prompts_mt50}[suite].text(task.name, np.random.default_rng(seed), procedure == "all")
    if condition:
        text, obs = vision.apply(env, task.name, condition, obs, seed)
    if look:
        looks.randomize(env, np.random.default_rng(seed + 777), look)
    goal = state_of(env, obs)["goal"].round(4).tolist()  # where the episode's goal started
    cams = vision.Cameras(env, cameras, upside_down) if use_vision else None
    qs = questions(task)
    frames, trace, matched, reads, success = [], [], 0, 0, False
    end, extra = None, None  # the results at the first success; steps left after it
    title, dt, sim_steps = f"{task.name} ({policy})", env.unwrapped.dt, 0
    # the reads behind the last act, its labels, its step and its grip: what the video shows while waiting
    shown = ({key: np.zeros(n) for key, _, n in qs}, labels(task, state_of(env, obs)), 0, -1.0)

    def display():
        if cams:  # the model's cameras, side by side
            fs = cams.frames()
            side = max(f.shape[0] for f in fs)
            return np.concatenate([np.array(Image.fromarray(f).resize((side, side))) for f in fs], axis=1)
        env.unwrapped.mujoco_renderer._get_viewer("rgb_array").make_context_current()  # envs render concurrently
        return np.ascontiguousarray(np.rot90(env.render(), 2))  # Meta-World renders rotated 180 degrees

    def advance(a, status, color):
        """--realtime: one sim step, and a video frame whenever the sim clock passes the next one."""
        nonlocal sim_steps
        out = env.step(a)
        sim_steps += 1
        while video and len(frames) < sim_steps * dt * realtime_fps:
            frames.append(panel(display(), title, shown[2], shown[0], shown[1], a,
                                (f"t {sim_steps * dt:5.2f} s  {status}", color)))
        return out

    for t in range(max_steps + after_success):
        if t >= max_steps and extra is None:
            break
        start = time.perf_counter()
        s = state_of(env, obs)
        truth = labels(task, s)
        if policy == "qwen":
            if cams:
                shots = cams.frames()
                urls = [vision.encode(f) for f in shots]  # once per step, shared by its questions
                step_text = text
                if instruct == "oracle":  # the expert's action in L2's words (instr_data.py), every step
                    step_text = f"{text}\nInstruction: {motion_words(task.soft_labels(s))}"
                users = [vision.for_vllm(vision.user_content(task.name, q, cameras, step_text), urls) for _, q, _ in qs]
            else:
                users = [user_text(task, s, text) for _, text, _ in qs]
            ps = await asyncio.gather(*[qwen.read(u, n) for u, (_, _, n) in zip(users, qs)])
            probs = {key: p for (key, _, _), p in zip(qs, ps)}
        else:
            probs = (task.soft_labels(s) if getattr(task, "soft", False)
                     else {key: np.eye(n)[truth[key]] for key, _, n in qs})
        matched += sum(int(probs[k].argmax() == truth[k]) for k in probs)
        reads += len(probs)
        action = action_of(task, probs)
        trace.append({"task": task.name, "seed": seed, "t": t, "tip": s["tip"].round(4).tolist(),
                      "err": (task.target(s)[0] - s["tip"]).round(4).tolist(),
                      "probs": {k: p.round(4).tolist() for k, p in probs.items()}, "truth": truth})
        if realtime_fps:
            latency = time.perf_counter() - start
            hold, hit = np.array([0, 0, 0, shown[3]], np.float32), False
            for k in range(int(latency / dt)):
                _, _, _, _, info = advance(hold, f"waiting {(k + 1) * dt * 1000:.0f}/{latency * 1000:.0f} ms", "#c44")
                hit |= bool(info["success"])
            shown = (probs, truth, t, float(action[3]))
            obs, _, terminated, truncated, info = advance(action, "act", "#2a7")
            info = {"success": hit or info["success"]}
        else:
            if video and cams and policy == "qwen":  # the frames the model was shown, side by side
                side = max(f.shape[0] for f in shots)
                frame = np.concatenate([np.array(Image.fromarray(f).resize((side, side))) for f in shots], axis=1)
                frames.append(panel(frame, title, t, probs, truth, action))
            elif video:
                frames.append(panel(display(), title, t, probs, truth, action))
            obs, _, terminated, truncated, info = env.step(action)
        if extra is not None:
            extra -= 1
            if extra == 0 or terminated or truncated:
                break
            continue
        if info["success"]:
            success = True
            end = (t, state_of(env, obs), matched, reads)
            if not after_success:
                break
            extra = after_success
            continue
        if terminated or truncated:
            break
    t, s, matched, reads = end or (t, state_of(env, obs), matched, reads)
    if cams:
        cams.close()
    env.close()
    if video:
        imageio.mimsave(video, frames, fps=realtime_fps or 10, macro_block_size=1)
    return {"task": task.name, "seed": seed, "goal": goal, "success": success, "steps": t + 1,
            "final_dist": float(np.linalg.norm(s["tip"] - task.target(s)[0])), "match": matched / reads}, trace


async def main(args):
    qwen = None
    if args.policy == "qwen":
        qwen = Qwen(args.url, args.model, args.concurrency)
        qwen.system = SYSTEMS[args.suite]
        await qwen.setup()
    os.makedirs(args.outdir, exist_ok=True)
    runs = []
    for name in args.tasks:
        for i in range(args.episodes):
            video = os.path.join(args.outdir, f"{name}_{args.policy}_{args.seed + i}.mp4") if i < args.video else None
            runs.append(episode(SUITES[args.suite][name], args.seed + i, args.policy, qwen, args.max_steps, video,
                                use_vision=args.vision, cameras=args.cameras,
                                upside_down=args.upside_down, after_success=args.after_success,
                                condition=args.condition, realtime_fps=30 if args.realtime else None,
                                procedure=args.procedure, suite=args.suite, look=args.look, instruct=args.instruct))
    done = await asyncio.gather(*runs)
    results = [r for r, _ in done]
    out = os.path.join(args.outdir, f"closed_{args.policy}.jsonl")
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.outdir, f"steps_{args.policy}.jsonl"), "w") as f:
        for _, trace in done:
            for row in trace:
                f.write(json.dumps(row) + "\n")
    print(f"policy {args.policy}, {args.episodes} episodes per task, at most {args.max_steps} steps -> {out}"
          + (f" ({qwen.retries} requests retried)" if qwen else ""))
    print(f"{'task':16s} {'success':>7s} {'steps':>6s} {'final dist':>10s} {'label match':>11s}")
    for name in args.tasks:
        rs = [r for r in results if r["task"] == name]
        print(f"{name:16s} {np.mean([r['success'] for r in rs]):7.2f} {np.mean([r['steps'] for r in rs]):6.0f} "
              f"{np.median([r['final_dist'] for r in rs]):10.3f} {np.mean([r['match'] for r in rs]):11.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), default="default", help="task registry of tasks.py")
    ap.add_argument("--tasks", nargs="+", default=["reach-v3"], choices=sorted({n for t in SUITES.values() for n in t}))
    ap.add_argument("--policy", choices=["labels", "qwen"], default="qwen")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--video", type=int, default=0, help="save the first N episodes per task as mp4")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--outdir", default="out/tasks")
    ap.add_argument("--vision", action="store_true", help="ask with camera frames instead of coordinates")
    ap.add_argument("--condition", choices=vision.CONDITIONS, help="--vision: change the scene's look (vision.apply)")
    ap.add_argument("--procedure", choices=["none", "all"], default="none", help="--suite ml10/mt50: procedure in the text")
    ap.add_argument("--look", choices=looks.SPLITS, default=None, help="--vision: a random look (looks.py) of this split")
    ap.add_argument("--instruct", choices=["oracle"], default=None,
                    help="--suite ml10/mt50 --vision: add an instruction line each step (oracle: the expert's action)")
    ap.add_argument("--realtime", action="store_true", help="video on the sim clock, the waits for the reads included")
    ap.add_argument("--after-success", type=int, default=0,
                    help="keep acting this many steps after success (for videos); results stay those at success")
    ap.add_argument("--cameras", nargs="+", default=list(vision.DEFAULT), choices=list(vision.RES))
    ap.add_argument("--upside-down", nargs="*", default=sorted(vision.UPSIDE_DOWN), choices=list(vision.RES),
                    help="cameras rotated 180 degrees (out/vision/reach_r16 was trained with corner gripperPOV)")
    asyncio.run(main(ap.parse_args()))
