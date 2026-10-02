"""Free-text probe: do base Qwen and the mt50 adapter describe the scene and plan the motion in words?
Each task's first validation state (three camera frames), three questions, greedy, 150 tokens."""
import asyncio, json, sys
import httpx, numpy as np
from PIL import Image
import vision
from tasks import SYSTEM_IMAGE

TASKS = {"drawer-close-v3": "val_comp_a", "door-close-v3": "val_comp_a", "handle-pull-side-v3": "val_comp_a",
         "faucet-open-v3": "val_obj_a", "faucet-close-v3": "val_obj_a", "drawer-open-v3": "val_id"}
QUESTIONS = [
    "Describe the scene: what objects are on the table, and where is the gripper?",
    "To do the task, which part of which object should the gripper go to, and then in which direction should it "
    "move? Give the directions as +x/-x (the robot's right/left), +y/-y (away from/toward the robot), +z/-z (up/down).",
    "Should the gripper's fingers be open or closed for this task? Why?",
]

def first_row(task, f):
    for line in open(f"out/mt50/data/{f}.jsonl"):
        r = json.loads(line)
        if r["task"] == task:
            return r

async def ask(http, model, row, q):
    content = row["messages"][1]["content"]
    task_text = content[-1]["text"].split("\n\nQuestion")[0]
    frames = [np.array(Image.open(p)) for p in row["images"]]
    user = vision.for_vllm(content[:-1], frames) + [{"type": "text", "text": f"{task_text}\n\n{q}"}]
    r = await http.post("/v1/chat/completions", json={
        "model": model, "max_tokens": 150, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": SYSTEM_IMAGE}, {"role": "user", "content": user}]})
    return task_text, r.json()["choices"][0]["message"]["content"].strip()

async def main(url):
    async with httpx.AsyncClient(base_url=url, timeout=300) as http:
        out = []
        for task, f in TASKS.items():
            row = first_row(task, f)
            for qi, q in enumerate(QUESTIONS):
                res = await asyncio.gather(*[ask(http, m, row, q) for m in MODELS])
                out.append({"task": task, "task_text": res[0][0], "q": qi + 1, **{m: a for m, (_, a) in zip(MODELS, res)}})
        json.dump(out, open(OUT, "w"), indent=1)
        for o in out:
            print(f"\n##### {o['task']}  Q{o['q']}   [{o['task_text']}]")
            for m in MODELS:
                print(f"  {m.upper()}: {o[m][:400]}")

# usage: probe.py <server url> [models, comma-separated; "qwen" is the base] [answers json]
MODELS = (sys.argv[2] if len(sys.argv) > 2 else "qwen,mt50").split(",")
OUT = sys.argv[3] if len(sys.argv) > 3 else "out/mt50/probe/answers.json"
asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8041"))
