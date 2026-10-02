#!/bin/bash
# Re-record the ML10 3-camera videos after the panel fix (the grip options were shown as dec/hold/inc):
# same episodes (seed 5000), 20k on the servers on GPU 0 (8021, test tasks) and GPU 1 (8012, train tasks),
# 40k on GPU 2 (8031). Replaces out/demo/ml10_3cam_{20k,40k}/*.mp4.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
TEST="drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3"
TRAIN="reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 sweep-v3 basketball-v3"
pids=()
for m in 20k 40k; do
  V=out/ml10_3cam/redemo_$m; mkdir -p $V
  url() { if [ $m = 40k ]; then echo http://127.0.0.1:8031; elif [[ " $TEST " == *" $1 "* ]]; then echo http://127.0.0.1:8021; else echo http://127.0.0.1:8012; fi; }
  CL=".venv/bin/python closed_loop.py --suite ml10 --vision --cameras corner gripperPOV behindGripper --policy qwen --model ml10_$m --max-steps 500 --episodes 1 --video 1 --after-success 40"
  for t in $TEST $TRAIN; do $CL --url $(url $t) --procedure none --tasks $t --outdir $V/a_$t > $V/a_$t.txt 2>&1 & pids+=($!); done
  for t in $TEST; do $CL --url $(url $t) --procedure all --tasks $t --outdir $V/b_$t > $V/b_$t.txt 2>&1 & pids+=($!); done
done
wait "${pids[@]}"
for m in 20k 40k; do V=out/ml10_3cam/redemo_$m
  for t in $TEST $TRAIN; do cp $V/a_$t/${t}_qwen_5000.mp4 out/demo/ml10_3cam_$m/$t.mp4; done
  for t in $TEST; do cp $V/b_$t/${t}_qwen_5000.mp4 out/demo/ml10_3cam_$m/${t}_procedure.mp4; done
done
echo done
