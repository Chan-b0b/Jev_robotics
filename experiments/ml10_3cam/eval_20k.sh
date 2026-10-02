#!/bin/bash
# 50-episode closed loops of the 20k 3-camera adapter on the servers already up: test tasks (a) and
# (b) on GPU 0 (port 8021), train tasks (a) on GPU 1 (port 8012). Waits on the clients only.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
TEST="drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3"
TRAIN="reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 sweep-v3 basketball-v3"
CL=".venv/bin/python closed_loop.py --suite ml10 --vision --cameras corner gripperPOV behindGripper --policy qwen --model ml10_20k --max-steps 500 --episodes 50"
E=out/ml10_3cam/eval; pids=()
for t in $TEST; do for pr in none all; do
  $CL --url http://127.0.0.1:8021 --procedure $pr --tasks $t --outdir $E/20k_${pr}_$t > $E/20k_${pr}_$t.txt 2>&1 & pids+=($!)
done; done
for t in $TRAIN; do
  $CL --url http://127.0.0.1:8012 --procedure none --tasks $t --outdir $E/20k_none_$t > $E/20k_none_$t.txt 2>&1 & pids+=($!)
done
wait "${pids[@]}"
echo done
