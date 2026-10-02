"""Language path vs action path on the same validation states (mt50co): the action path's v = P(inc) - P(dec)
per axis from the three axis questions and the grip question's most likely option; the language path's v from
the model's own answer to an L2 question (lang_data.py), read back (quickly = 1, slightly = 0.5, an axis not
named = 0; the grip from its sentence). Both against the expert's v (the soft labels): per axis decrease /
hold / increase with |v| > 0.15 (L2's threshold for naming an axis), the grip, and the mean |v error|."""
import asyncio, json, re, sys
import httpx, numpy as np
from PIL import Image
import vision
from closed_loop import Qwen
from lang_data import L2_QUESTIONS, lm_row
from tasks import SYSTEM_IMAGE

AX = {"x": 0, "y": 1, "z": 2}

def parse(text):
    v = np.zeros(3)
    for adv, sign, ax in re.findall(r"(quickly|slightly)[^()]*?\(([+-])([xyz])\)", text):
        v[AX[ax]] = (1.0 if adv == "quickly" else 0.5) * (1 if sign == "+" else -1)
    grip = 2 if "tightly" in text else 1 if "grasp" in text else 0
    return v, grip

def cls(v):
    return np.where(v > 0.15, 2, np.where(v < -0.15, 0, 1))

async def main(path, n_states, url, model):
    rows = [json.loads(l) for l in open(path)]
    states = [rows[i:i + 4] for i in range(0, len(rows), 4)]
    if n_states:
        states = [states[i] for i in np.linspace(0, len(states) - 1, n_states).astype(int)]
    qwen = Qwen(url, model, 128); qwen.system = SYSTEM_IMAGE; await qwen.setup()
    gen = httpx.AsyncClient(base_url=url, timeout=600, limits=httpx.Limits(max_connections=128))
    sem = asyncio.Semaphore(96)

    async def one(i, s):
        async with sem:
            frames = [np.array(Image.open(p)) for p in s[0]["images"]]
            urls = [vision.encode(f) for f in frames]
            soft = {r["key"]: np.array(r["soft"]) for r in s}
            v_exp = np.array([soft[k][2] - soft[k][0] for k in ("dx", "dy", "dz")]); g_exp = int(np.argmax(soft["grip"]))
            ps = await asyncio.gather(*[qwen.read(vision.for_vllm(r["messages"][1]["content"], urls), 3) for r in s])
            p = {r["key"]: q for r, q in zip(s, ps)}
            v_act = np.array([p[k][2] - p[k][0] for k in ("dx", "dy", "dz")]); g_act = int(np.argmax(p["grip"]))
            row = lm_row(s[0], L2_QUESTIONS[i % len(L2_QUESTIONS)], "")
            body = {"model": model, "max_tokens": 60, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [row["messages"][0], {"role": "user", "content": vision.for_vllm(row["messages"][1]["content"], urls)}]}
            text = (await gen.post("/v1/chat/completions", json=body)).json()["choices"][0]["message"]["content"].strip()
            v_lang, g_lang = parse(text)
            return {"task": s[0]["task"], "v_exp": v_exp.tolist(), "g_exp": g_exp, "v_act": v_act.tolist(), "g_act": g_act,
                    "v_lang": v_lang.tolist(), "g_lang": g_lang, "text": text}
    res = await asyncio.gather(*[one(i, s) for i, s in enumerate(states)])
    out = path.split("/")[-1].replace(".jsonl", "")
    json.dump(res, open(f"out/mt50co/langact/{model}_{out}.json", "w"), indent=0)
    print(f"== {model} on {out}: {len(res)} states   (axis = decrease/hold/increase agreement, |v err| = mean abs error)")
    print(f"  {'task':26s} {'axis act':>9s} {'axis lang':>10s} {'grip act':>9s} {'grip lang':>10s} {'|v err| act':>12s} {'|v err| lang':>13s}")
    for t in list(dict.fromkeys(r["task"] for r in res)) + ["ALL"]:
        rs = [r for r in res if t in ("ALL", r["task"])]
        e, a, l = (np.array([r[k] for r in rs]) for k in ("v_exp", "v_act", "v_lang"))
        print(f"  {t:26s} {np.mean(cls(a) == cls(e)):9.3f} {np.mean(cls(l) == cls(e)):10.3f} "
              f"{np.mean([r['g_act'] == r['g_exp'] for r in rs]):9.3f} {np.mean([r['g_lang'] == r['g_exp'] for r in rs]):10.3f} "
              f"{np.abs(a - e).mean():12.3f} {np.abs(l - e).mean():13.3f}")

asyncio.run(main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "mt50co"))
