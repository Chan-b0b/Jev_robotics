#!/bin/bash
# goal_far for all_2cam (vmulti): 50 episodes per task, one client per task on its own server
# (one client for all four tasks is CPU-bound: 200 episodes of simulation, rendering and PNG
# encoding in one process), merged into eval/closed_qwen.jsonl; then one video per task.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
ALL="reach-v3 push-v3 pick-place-v3 drawer-open-v3"
c=goal_far d=out/vision_multi/ablation/goal_far
mkdir -p $d
for spec in reach-v3:8001 push-v3:8002 pick-place-v3:8003 drawer-open-v3:8004; do t=${spec%:*}
  .venv/bin/python closed_loop.py --vision --policy qwen --url http://127.0.0.1:${spec#*:} --model vmulti --condition $c \
    --episodes 50 --max-steps 500 --tasks $t --outdir $d/eval_$t > $d/closed_$t.txt 2>&1 &
done
wait
mkdir -p $d/eval
cat $d/eval_*/closed_qwen.jsonl > $d/eval/closed_qwen.jsonl
cat $d/eval_*/steps_qwen.jsonl > $d/eval/steps_qwen.jsonl
grep -hE "^(reach|push|pick|drawer)" $d/closed_*-v3.txt > $d/closed.txt
.venv/bin/python closed_loop.py --vision --policy qwen --url http://127.0.0.1:8001 --model vmulti --condition $c --episodes 1 --video 1 \
  --after-success 40 --max-steps 500 --tasks $ALL --outdir $d/demo > $d/demo.txt 2>&1
mkdir -p out/demo/all_2cam/ablation/$c
for t in $ALL; do cp $d/demo/${t}_qwen_5000.mp4 out/demo/all_2cam/ablation/$c/$t.mp4; done
echo done
