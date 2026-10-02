#!/bin/bash
# mt50 split data (prompts_mt50.py, looks.py), three cameras: train tasks 5k states each (20k rows)
# with train looks on 80% of episodes; validation of 60 states per train task with train looks (T-id)
# and with test looks on all episodes (T-vis); and 250 states per test task (T-comp, T-obj) with train
# looks, once without (a) and once with (b) procedures. At most 40 generators at once (EGL).
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
D=out/mt50/data
G=".venv/bin/python lora_data.py --suite mt50 --vision --cameras corner gripperPOV behindGripper"
jobs=$(.venv/bin/python - <<'PY'
import prompts_mt50 as P
D = "out/mt50/data"
for t in P.TRAIN:
    print(f"--task {t} --states 5000 --seed 0 --look train --look-p 0.8 --out {D}/{t}_train.jsonl")
    print(f"--task {t} --states 60 --seed 1 --look train --look-p 0.8 --out {D}/{t}_val_id.jsonl")
    print(f"--task {t} --states 60 --seed 2 --look test --look-p 1.0 --out {D}/{t}_val_vis.jsonl")
for t in P.T_COMP + P.T_OBJ:
    for v, p in (("a", "none"), ("b", "all")):
        print(f"--task {t} --states 250 --seed 1 --look train --look-p 0.8 --procedure {p} --out {D}/{t}_val_{v}.jsonl")
PY
)
echo "$jobs" | xargs -P 40 -I{} sh -c "$G {} > \$(echo '{}' | sed 's/.*--out //; s/.jsonl$/.log/') 2>&1"
.venv/bin/python - <<'PY'
import random, prompts_mt50 as P
D = "out/mt50/data"
def rows(p): return open(p).read().splitlines()
def write(p, rs, shuffle=True):
    if shuffle: random.Random(0).shuffle(rs)
    open(p, "w").write("\n".join(rs) + "\n"); print(p, len(rs))
write(f"{D}/train.jsonl", [r for t in P.TRAIN for r in rows(f"{D}/{t}_train.jsonl")])
write(f"{D}/val_id.jsonl", [r for t in P.TRAIN for r in rows(f"{D}/{t}_val_id.jsonl")], False)
write(f"{D}/val_vis.jsonl", [r for t in P.TRAIN for r in rows(f"{D}/{t}_val_vis.jsonl")], False)
for v in "ab":
    write(f"{D}/val_comp_{v}.jsonl", [r for t in P.T_COMP for r in rows(f"{D}/{t}_val_{v}.jsonl")], False)
    write(f"{D}/val_obj_{v}.jsonl", [r for t in P.T_OBJ for r in rows(f"{D}/{t}_val_{v}.jsonl")], False)
PY
echo done
