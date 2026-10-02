"""Watch a policy act: an mp4 of one episode, with each step's reads beside the frame.

--policy qwen: every step asks Qwen the stage 2 questions (text state) and moves
each axis by (P(increase) - P(decrease)) * MAG; the gripper closes when
P(close) > 0.5. --policy labels: the stage 2 labels act, for comparison.
The panel shows the model's probabilities and the label it should have given.

--realtime records on the sim clock, as a camera on a real robot would: while
Qwen answers, the sim runs on with the arm holding still for as long as the
requests took, then the answer moves the arm for one step.
"""
import argparse
import os
import time

import gymnasium as gym
import httpx
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stage0_expert import make
from stage2_data import MAG, labels, parse
from stage2_eval import LETTERS, SYSTEM, VARIANTS, questions, user_text

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
NAMES = {"dx": ["dec", "hold", "inc"], "dy": ["dec", "hold", "inc"], "dz": ["dec", "hold", "inc"],
         "grip": ["close", "open"]}


class Qwen:
    def __init__(self, url, model):
        self.http = httpx.Client(base_url=url, timeout=60)
        self.model = model
        self.ids = [self.http.post("/tokenize", json={"model": model, "prompt": l, "add_special_tokens": False})
                    .json()["tokens"][0] for l in LETTERS]

    def read(self, user, n):
        r = self.http.post("/v1/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
            "logprob_token_ids": self.ids[:n], "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}})
        r.raise_for_status()
        top = {int(t["token"].split(":")[1]): t["logprob"]
               for t in r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        lp = np.array([top[i] for i in self.ids[:n]])
        p = np.exp(lp - lp.max())
        return p / p.sum()


def panel(frame, task, t, probs, truth, action, policy, status=None):
    """The camera frame with a side panel: per question, a bar per option, the label boxed."""
    h = 480
    img = Image.new("RGB", (h + 360, h), "white")
    img.paste(Image.fromarray(frame).resize((h, h)), (0, 0))
    d = ImageDraw.Draw(img)
    big, small = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 14)
    x0 = h + 15
    d.text((x0, 10), f"{task}  step {t}", fill="black", font=big)
    d.text((x0, 34), f"policy: {policy}", fill="gray", font=small)
    y = 64
    for key in ("dx", "dy", "dz", "grip"):
        d.text((x0, y), key, fill="black", font=big)
        y += 24
        for k, name in enumerate(NAMES[key]):
            p = probs[key][k] if probs else (1.0 if k == truth[key] else 0.0)
            color = "#2a7" if k == truth[key] else "#c44"
            d.text((x0, y), f"{name:>5s}", fill="black", font=small)
            d.rectangle([x0 + 50, y + 2, x0 + 50 + int(200 * p), y + 14], fill=color)
            d.text((x0 + 258, y), f"{p:.2f}", fill="black", font=small)
            if k == truth[key]:
                d.rectangle([x0 + 48, y, x0 + 252, y + 16], outline="black")
            y += 18
        y += 8
    d.text((x0, y + 4), "action (m): " + " ".join(f"{a / 100:+.4f}" for a in action[:3]), fill="black", font=small)
    d.text((x0, y + 22), f"grip: {'close' if action[3] > 0 else 'open'}", fill="black", font=small)
    if status:  # on the camera frame, top left
        text, color = status
        d.rectangle([0, 0, 10 + d.textlength(text, font=big), 30], fill="white")
        d.text((5, 5), text, fill=color, font=big)
    d.text((x0, h - 40), "green = correct label (boxed)", fill="#2a7", font=small)
    d.text((x0, h - 22), "red = other options", fill="#c44", font=small)
    return np.array(img)


def main(args):
    env = make(args.task, args.seed, args.camera)
    qwen = Qwen(args.url, args.model) if args.policy == "qwen" else None
    obs, _ = env.reset(seed=args.seed)
    frames, success, correct, total = [], False, 0, 0
    policy = args.policy if args.policy == "labels" else f"qwen ({args.variant} prompt)"
    if args.realtime:  # waiting steps are sim time, not decisions: lift the 500-step limits
        w = env
        while hasattr(w, "env"):
            if isinstance(w, gym.wrappers.TimeLimit):
                w._max_episode_steps = 10 ** 6
            w = w.env
        w.max_path_length = 10 ** 6
    sim_steps, dt = 0, env.unwrapped.dt
    shown = {k: np.zeros(len(v)) for k, v in NAMES.items()}  # the reads behind the last executed action
    grip = -1

    def sim_step(a, t, truth, status, color):
        """One sim step, and a video frame whenever the sim clock passes the next one."""
        nonlocal obs, sim_steps
        obs, _, terminated, truncated, info = env.step(a)
        sim_steps += 1
        while len(frames) < sim_steps * dt * args.fps:
            frame = np.ascontiguousarray(np.rot90(env.render(), 2))
            frames.append(panel(frame, args.task, t, shown, truth, a, policy,
                                (f"t {sim_steps * dt:5.2f} s  {status}", color)))
        return info["success"], terminated or truncated

    for t in range(args.steps):
        start = time.perf_counter()
        s = parse(obs)
        xyz, close = labels(args.task, s)
        truth = {"dx": xyz[0] + 1, "dy": xyz[1] + 1, "dz": xyz[2] + 1, "grip": 0 if close else 1}
        if qwen:
            row = {"task": args.task, "hand": s["hand"].round(4).tolist(), "opening": round(s["opening"], 3),
                   "obj": s["obj"].round(4).tolist(), "goal": s["goal"].round(4).tolist(),
                   "dx": 0, "dy": 0, "dz": 0, "grip": 0}
            probs = {}
            for key, text, opts, _, order, v in questions(row, args.variant):
                probs[key] = np.zeros(len(opts))
                probs[key][order] = qwen.read(user_text(row, text, v), len(opts))
            move = [(probs[k][2] - probs[k][0]) * MAG for k in ("dx", "dy", "dz")]
            action = np.array([*move, 1 if probs["grip"][0] > 0.5 else -1], np.float32)
            correct += sum(int(probs[k].argmax() == truth[k]) for k in truth)
            total += 4
        else:
            probs = None
            action = np.array([*(xyz * MAG), 1 if close else -1], np.float32)
        if args.realtime:
            latency = time.perf_counter() - start
            hold = np.array([0, 0, 0, grip], np.float32)
            for k in range(int(latency / dt)):
                success, done = sim_step(hold, t, truth,
                                         f"waiting {(k + 1) * dt * 1000:.0f}/{latency * 1000:.0f} ms", "#c44")
                if success or done:
                    break
            else:
                shown, grip = probs, action[3]
                success, done = sim_step(action, t, truth, "act", "#2a7")
            if success or done:
                break
            continue
        frame = np.rot90(env.render(), 2)  # Meta-World renders rotated 180 degrees
        frames.append(panel(np.ascontiguousarray(frame), args.task, t, probs, truth, action, policy))
        obs, _, terminated, truncated, info = env.step(action)
        if info["success"]:
            success = True
            break
        if terminated or truncated:
            break
    env.close()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, macro_block_size=1)
    acc = f", argmax matched the label on {correct}/{total} reads" if total else ""
    print(f"{args.task} {args.policy}: success={success} after {t + 1} steps{acc} -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="reach-v3")
    ap.add_argument("--policy", choices=["qwen", "labels"], default="qwen")
    ap.add_argument("--variant", choices=VARIANTS, default="base", help="prompt variant (stage2_eval.py)")
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--camera", default="corner")
    ap.add_argument("--fps", type=int, default=None, help="default 10, or 30 with --realtime")
    ap.add_argument("--realtime", action="store_true", help="record on the sim clock, waits for Qwen included")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.fps = args.fps or (30 if args.realtime else 10)
    args.out = args.out or (f"out/demo/{args.task}_{args.policy}{'' if args.policy == 'labels' else '_' + args.variant}"
                            f"{'_realtime' if args.realtime else ''}.mp4")
    main(args)
