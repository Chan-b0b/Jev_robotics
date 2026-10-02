"""Text probes on the mt50 validation rows (images unchanged): train-task rows with another train task's
description (swapped); test-task rows with an instruction line in L2's words (lang_data.motion_words), either
the state's own expert action (instr) or another state's of the same task (wrong)."""
import copy, json, numpy as np
import prompts_mt50 as P
from lang_data import motion_words
D, O = "out/mt50/data", "out/mt50co/textprobe"
rng = np.random.default_rng(0)

def task_text(r):
    return r["messages"][1]["content"][-1]["text"].split("\n\nQuestion")[0]

def set_text(r, new):
    """A copy of row r with its task text replaced."""
    r = copy.deepcopy(r)
    part = r["messages"][1]["content"][-1]
    part["text"] = new + part["text"][part["text"].index("\n\nQuestion"):]
    return r

rows = [json.loads(l) for l in open(f"{D}/val_id.jsonl")]
with open(f"{O}/val_id_swapped.jsonl", "w") as f:
    for r in rows:
        other = rng.choice([t for t in P.TRAIN if t != r["task"]])
        f.write(json.dumps(set_text(r, "Task: " + rng.choice(P.DESCRIPTIONS[other]))) + "\n")

for name in ("val_comp_a", "val_obj_a"):
    rows = [json.loads(l) for l in open(f"{D}/{name}.jsonl")]
    states = [rows[i:i + 4] for i in range(0, len(rows), 4)]  # dx, dy, dz, grip of one state
    words = [motion_words({r["key"]: r["soft"] for r in s}) for s in states]
    by_task = {}
    for i, s in enumerate(states):
        by_task.setdefault(s[0]["task"], []).append(i)
    for kind in ("instr", "wrong"):
        with open(f"{O}/{name}_{kind}.jsonl", "w") as f:
            for i, s in enumerate(states):
                j = i if kind == "instr" else rng.choice([k for k in by_task[s[0]["task"]] if words[k] != words[i]] or [i])
                for r in s:
                    f.write(json.dumps(set_text(r, f"{task_text(r)}\nInstruction: {words[j]}")) + "\n")
    print(name, len(states), "states")
