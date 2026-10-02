#!/bin/bash
# Robustness of all_2cam (vmulti): each condition of vision.apply on its own server, 50 episodes
# per task, then one video per task in out/demo/all_2cam/ablation/<condition>/<task>.mp4
# (seed 5000, 40 steps past success). goal_color_nocolor runs only if goal_color's mean success < 0.9.
# Also reruns the default-look all_2cam and base evaluations that dropped connections before.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
ALL="reach-v3 push-v3 pick-place-v3 drawer-open-v3"
run() {  # run <condition> <port>
  local c=$1 u=http://127.0.0.1:$2 d=out/vision_multi/ablation/$1
  mkdir -p $d
  .venv/bin/python closed_loop.py --vision --policy qwen --url $u --model vmulti --condition $c --episodes 50 --max-steps 500 \
    --tasks $ALL --outdir $d/eval > $d/closed.txt 2>&1
  .venv/bin/python closed_loop.py --vision --policy qwen --url $u --model vmulti --condition $c --episodes 1 --video 1 \
    --after-success 40 --max-steps 500 --tasks $ALL --outdir $d/demo > $d/demo.txt 2>&1
  mkdir -p out/demo/all_2cam/ablation/$c
  for t in $ALL; do cp $d/demo/${t}_qwen_5000.mp4 out/demo/all_2cam/ablation/$c/$t.mp4; done
}
CL=".venv/bin/python closed_loop.py --vision --policy qwen --episodes 50 --max-steps 500"
$CL --url http://127.0.0.1:8004 --model vmulti --video 2 --tasks $ALL --outdir out/vision_multi/eval_vmulti \
  > out/vision_multi/closed_vmulti.txt 2>&1 &  # the default look, for comparison
run goal_color 8001 &
run table_color 8003 &
(run background 8002
 $CL --url http://127.0.0.1:8002 --model qwen --tasks push-v3 pick-place-v3 drawer-open-v3 --outdir out/vision_multi/eval_qwen \
   > out/vision_multi/closed_qwen.txt 2>&1) &
wait
low=$(.venv/bin/python -c "
import json, numpy as np
rs = [json.loads(l) for l in open('out/vision_multi/ablation/goal_color/eval/closed_qwen.jsonl')]
print(int(np.mean([r['success'] for r in rs]) < 0.9))")
[ "$low" = 1 ] && run goal_color_nocolor 8001
echo done
