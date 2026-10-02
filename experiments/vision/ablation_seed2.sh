#!/bin/bash
# A second video per ablation condition and task, with another seed (other object and goal
# placements): out/demo/all_2cam/ablation/<condition>/<task>_seed<N>.mp4, on the server on GPU 0.
# goal_far uses 6004, whose goal is on the other side from seed 5000's (behind the arm in the corner view).
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
ALL="reach-v3 push-v3 pick-place-v3 drawer-open-v3"
for spec in goal_color:6000 table_color:6000 background:6000 goal_far:6004; do
  c=${spec%:*} seed=${spec#*:} d=out/vision_multi/ablation/${spec%:*}
  until [ -f out/demo/all_2cam/ablation/$c/drawer-open-v3.mp4 ]; do sleep 30; done
  .venv/bin/python closed_loop.py --vision --policy qwen --url http://127.0.0.1:8004 --model vmulti --condition $c \
    --seed $seed --episodes 1 --video 1 --after-success 40 --max-steps 500 --tasks $ALL --outdir $d/demo_seed$seed > $d/demo_seed$seed.txt 2>&1
  for t in $ALL; do cp $d/demo_seed$seed/${t}_qwen_$seed.mp4 out/demo/all_2cam/ablation/$c/${t}_seed$seed.mp4; done
done
echo done
