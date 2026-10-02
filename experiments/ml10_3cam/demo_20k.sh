#!/bin/bash
# One video per ML10 task with the 20k 3-camera adapter, as soon as eval_lora.sh's servers serve it:
# out/demo/ml10_3cam_20k/<task>.mp4 (seed 5000, description only, 40 steps past success).
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
for p in 8011 8012; do until curl -s localhost:$p/v1/models | grep -q ml10_20k; do sleep 30; done; done
D=out/ml10_3cam/demo_20k; mkdir -p $D out/demo/ml10_3cam_20k
CL=".venv/bin/python closed_loop.py --suite ml10 --vision --cameras corner gripperPOV behindGripper --policy qwen --model ml10_20k --procedure none --episodes 1 --video 1 --after-success 40 --max-steps 500"
for t in drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3; do
  $CL --url http://127.0.0.1:8011 --tasks $t --outdir $D/$t > $D/$t.txt 2>&1 &
done
for t in reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 sweep-v3 basketball-v3; do
  $CL --url http://127.0.0.1:8012 --tasks $t --outdir $D/$t > $D/$t.txt 2>&1 &
done
wait
for t in $(ls $D | grep -v txt); do cp $D/$t/${t}_qwen_5000.mp4 out/demo/ml10_3cam_20k/$t.mp4; done
echo done
