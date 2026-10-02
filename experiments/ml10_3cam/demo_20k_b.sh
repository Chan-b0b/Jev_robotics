#!/bin/bash
# Test-task videos with the 20k 3-camera adapter and a procedure in the query (b):
# out/demo/ml10_3cam_20k/<task>_procedure.mp4 (seed 5000, 40 steps past success).
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
until curl -s localhost:8011/v1/models | grep -q ml10_20k; do sleep 30; done
D=out/ml10_3cam/demo_20k_b; mkdir -p $D out/demo/ml10_3cam_20k
for t in drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3; do
  .venv/bin/python closed_loop.py --suite ml10 --vision --cameras corner gripperPOV behindGripper --policy qwen \
    --url http://127.0.0.1:8011 --model ml10_20k --procedure all --episodes 1 --video 1 --after-success 40 --max-steps 500 \
    --tasks $t --outdir $D/$t > $D/$t.txt 2>&1 &
done
wait
for t in drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3; do cp $D/$t/${t}_qwen_5000.mp4 out/demo/ml10_3cam_20k/${t}_procedure.mp4; done
echo done
