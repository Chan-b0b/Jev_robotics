"""Language rows to train alongside the action rows (co-training), from the mt50 train data's states.

L2, motion in words: "which way should the gripper move now?", answered from the state's soft
labels (the expert's action): the axes it moves along, strongest first, in words and +/-x, y, z,
and the gripper's setting. L1, scene descriptions: the base model's own answer to "describe the
scene" for the same frames (self-distillation), so that training keeps the base model's way of
describing a scene. Rows have kind "lm" and the whole assistant text as the answer; lora_train.py
trains on its tokens.

  python lang_data.py l2 --per-task 4000 --out out/mt50/data/lang_l2.jsonl
  python lang_data.py l1 --per-task 1050 --urls http://127.0.0.1:8051 ... --out out/mt50/data/lang_l1.jsonl
"""
import argparse
import asyncio
import json

import httpx
import numpy as np
from PIL import Image

import prompts_mt50
import vision

L1_QUESTIONS = [
    "Describe the scene: what objects are on the table, and where is the gripper?",
    "What do the cameras show? Name the objects on the table and say where the gripper is.",
    "Describe what you see on the table and the position of the gripper.",
]
L2_QUESTIONS = [
    "To do the task, which way should the gripper move now, and how should the fingers be set?",
    "What should the gripper do next? Say the direction to move and whether the fingers are open or closed.",
    "Which direction should the gripper move at this moment, and should the fingers be open or closed?",
    "Describe the gripper's next motion and its fingers for this task.",
]
WORDS = {("x", 1): "to the robot's right (+x)", ("x", -1): "to the robot's left (-x)",
         ("y", 1): "away from the robot (+y)", ("y", -1): "toward the robot (-y)",
         ("z", 1): "up (+z)", ("z", -1): "down (-z)"}
GRIP = ["Keep the fingers open.", "Close the fingers to grasp.", "Close the fingers tightly."]


def states(task, n, rng):
    """n states of the task's train rows, spread over the file: (the dx row, {key: soft label})."""
    rows = [json.loads(l) for l in open(f"out/mt50/data/{task}_train.jsonl")]
    by_state = [rows[i:i + 4] for i in range(0, len(rows), 4)]  # 4 questions per state, in order
    pick = rng.choice(len(by_state), size=min(n, len(by_state)), replace=False)
    return [(by_state[i][0], {r["key"]: r["soft"] for r in by_state[i]}) for i in sorted(pick)]


def motion_words(soft):
    """The expert's action in words: axes moving more than 0.15, strongest first."""
    v = {a: soft[k][2] - soft[k][0] for a, k in (("x", "dx"), ("y", "dy"), ("z", "dz"))}
    moving = sorted((a for a in v if abs(v[a]) > 0.15), key=lambda a: -abs(v[a]))
    if moving:
        parts = [("quickly " if abs(v[a]) > 0.6 else "slightly ") + WORDS[(a, int(np.sign(v[a])))] for a in moving]
        move = "Move " + (parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]) + "."
    else:
        move = "Hold the gripper still."
    return f"{move} {GRIP[int(np.argmax(soft['grip']))]}"


def lm_row(r, question, answer):
    content = r["messages"][1]["content"]
    task_text = content[-1]["text"].split("\n\nQuestion")[0]
    user = content[:-1] + [{"type": "text", "text": f"{task_text}\n\n{question}"}]
    return {"task": r["task"], "kind": "lm", "images": r["images"], "cameras": r["cameras"], "messages": [
        r["messages"][0], {"role": "user", "content": user}, {"role": "assistant", "content": answer}]}


async def describe(urls, items, concurrency=64):
    """The base model's answers (greedy, 150 tokens) for (row, question) items, spread over the servers."""
    clients = [httpx.AsyncClient(base_url=u, timeout=600) for u in urls]
    sem = asyncio.Semaphore(concurrency * len(urls))

    async def one(i, r, q):
        async with sem:
            row = lm_row(r, q, "")
            frames = [np.array(Image.open(p)) for p in r["images"]]
            body = {"model": "qwen", "max_tokens": 150, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [row["messages"][0], {"role": "user", "content": vision.for_vllm(row["messages"][1]["content"], frames)}]}
            resp = await clients[i % len(clients)].post("/v1/chat/completions", json=body)
            return resp.json()["choices"][0]["message"]["content"].strip()
    return await asyncio.gather(*[one(i, r, q) for i, (r, q) in enumerate(items)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["l1", "l2"])
    ap.add_argument("--per-task", type=int, required=True)
    ap.add_argument("--urls", nargs="+", default=["http://127.0.0.1:8051"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed + (args.kind == "l1"))
    picked = [s for t in prompts_mt50.TRAIN for s in states(t, args.per_task, rng)]
    if args.kind == "l2":
        rows = [lm_row(r, rng.choice(L2_QUESTIONS), motion_words(soft)) for r, soft in picked]
    else:
        items = [(r, rng.choice(L1_QUESTIONS)) for r, _ in picked]
        answers = asyncio.run(describe(args.urls, items))
        rows = [lm_row(r, q, a) for (r, q), a in zip(items, answers)]
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} {args.kind} rows -> {args.out}")
    for r in rows[:3]:
        print(" Q:", r["messages"][1]["content"][-1]["text"].split("\n\n")[-1], "\n A:", r["messages"][2]["content"][:200])
