#!/bin/bash
# More mt50 videos with another seed (default 6000): the same sets as eval.sh's, saved next to them
# as out/demo/mt50/<set>/<task>_seed<seed>.mp4; at most 8 at once next to the running evaluation.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
SEED=${1:-6000}
jobs=$(SEED=$SEED .venv/bin/python - <<'PY'
import os, prompts_mt50 as P
seed = os.environ["SEED"]
CL = (".venv/bin/python closed_loop.py --suite mt50 --vision --cameras corner gripperPOV behindGripper "
      f"--policy qwen --model mt50 --max-steps 500 --episodes 1 --video 1 --after-success 40 --seed {seed}")
E, i = "out/mt50/eval", 0
def job(name, task, look, proc):
    global i
    port = 8041 + i % 4; i += 1
    print(f"{CL} --url http://127.0.0.1:{port} --look {look} --procedure {proc} --tasks {task} "
          f"--outdir {E}/{name}_s{seed} > {E}/{name}_s{seed}.txt 2>&1")
for t in P.T_COMP + P.T_OBJ:
    for v, proc in (("a", "none"), ("b", "all")):
        job(f"video_test_{v}_{t}", t, "train", proc)
for t in P.TRAIN:
    job(f"video_id_{t}", t, "train", "none")
    job(f"video_vis_{t}", t, "test", "none")
PY
)
echo "$jobs" | xargs -P 8 -I{} sh -c "{}"
E=out/mt50/eval
for d in $E/video_*_s$SEED/; do b=$(basename $d _s$SEED); t=${b##*_}; f=$d/${t}_qwen_$SEED.mp4; [ -f $f ] || continue
  case $b in video_test_a_*) s=test_a;; video_test_b_*) s=test_b;; video_id_*) s=id;; video_vis_*) s=vis;; esac
  cp $f out/demo/mt50/$s/${t}_seed$SEED.mp4
done
echo done
