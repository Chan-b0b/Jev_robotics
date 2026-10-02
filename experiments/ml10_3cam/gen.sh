#!/bin/bash
# ML10 image data, three cameras (corner, gripperPOV, behindGripper) (tasks.ML10, prompts.py): 10k states per train task (40k rows; the 20k-row set
# is each task's first 5k states), and 250 validation states per task: train tasks with procedures
# mixed in, test tasks once without (a) and once with (b) procedures.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
D=out/ml10_3cam/data
TRAIN="reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 sweep-v3 basketball-v3"
TEST="drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3"
G=".venv/bin/python lora_data.py --suite ml10 --vision --cameras corner gripperPOV behindGripper"
for t in $TRAIN; do
  $G --task $t --states 10000 --seed 0 --out $D/${t}_train.jsonl > $D/${t}_train.log 2>&1 &
  $G --task $t --states 250 --seed 1 --out $D/${t}_val.jsonl > $D/${t}_val.log 2>&1 &
done
for t in $TEST; do
  $G --task $t --states 250 --seed 1 --procedure none --out $D/${t}_val_a.jsonl > $D/${t}_val_a.log 2>&1 &
  $G --task $t --states 250 --seed 1 --procedure all --out $D/${t}_val_b.jsonl > $D/${t}_val_b.log 2>&1 &
done
wait
.venv/bin/python - <<'PY'
import json, random
D = "out/ml10_3cam/data"
TRAIN = "reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 sweep-v3 basketball-v3".split()
TEST = "drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3".split()
def rows(p): return open(p).read().splitlines()
def write(p, rs, shuffle=True):
    if shuffle: random.Random(0).shuffle(rs)
    open(p, "w").write("\n".join(rs) + "\n"); print(p, len(rs))
full = {t: rows(f"{D}/{t}_train.jsonl") for t in TRAIN}
write(f"{D}/train_40k.jsonl", [r for t in TRAIN for r in full[t]])
write(f"{D}/train_20k.jsonl", [r for t in TRAIN for r in full[t][:20000]])  # rows are 4 per state, in order
write(f"{D}/val_train.jsonl", [r for t in TRAIN for r in rows(f"{D}/{t}_val.jsonl")], False)
for v in "ab":
    write(f"{D}/val_test_{v}.jsonl", [r for t in TEST for r in rows(f"{D}/{t}_val_{v}.jsonl")], False)
PY
echo done
