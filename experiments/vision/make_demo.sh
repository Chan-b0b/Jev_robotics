#!/bin/bash
# One video per model and task, out/demo/<model>/<task>.mp4: the seed-5000 episode, run on
# for 40 steps after success so the video shows the gripper settling at the goal.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
ALL="reach-v3 push-v3 pick-place-v3 drawer-open-v3"
demo() {  # demo <model name> <served model> <tasks> [closed_loop options...]
  local name=$1 model=$2 tasks=$3; shift 3
  local dir=out/demo_runs/$name
  .venv/bin/python closed_loop.py --vision --policy qwen --url http://127.0.0.1:8001 --model $model --episodes 1 \
    --video 1 --after-success 40 --max-steps 500 --tasks $tasks --outdir $dir "$@" > $dir.txt 2>&1
  mkdir -p out/demo/$name
  for t in $tasks; do cp "$dir/${t}_qwen_5000.mp4" "out/demo/$name/$t.mp4"; done
}
mkdir -p out/demo_runs
demo base_2cam qwen "$ALL"
demo reach_2cam_flipped v0 reach-v3 --upside-down corner gripperPOV
demo reach_2cam v2 reach-v3
demo reach_3cam v3 reach-v3 --cameras corner gripperPOV behindGripper
demo all_2cam vmulti "$ALL"
find out/demo -mindepth 2 -name "*.mp4" | sort
echo done
