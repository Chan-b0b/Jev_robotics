#!/bin/bash
# Evaluate the four vision adapters served on port 8001 (val accuracy with confusion, then closed loop).
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
U=http://127.0.0.1:8001
.venv/bin/python lora_eval.py --url $U --model v0 --data out/vision/reach_val.jsonl --confusion > out/vision/val_v0.txt 2>&1
.venv/bin/python lora_eval.py --url $U --model v2 --data out/vision2/reach_val.jsonl --confusion > out/vision2/val_v2.txt 2>&1
.venv/bin/python lora_eval.py --url $U --model v3 --data out/vision3/reach_val.jsonl --confusion > out/vision3/val_v3.txt 2>&1
.venv/bin/python lora_eval.py --url $U --model vmulti --data out/vision_multi/val.jsonl --confusion > out/vision_multi/val_vmulti.txt 2>&1
CL=".venv/bin/python closed_loop.py --vision --policy qwen --url $U --episodes 50 --video 2"
$CL --model v0 --upside-down corner gripperPOV --tasks reach-v3 --outdir out/vision/eval_v0 > out/vision/closed_v0.txt 2>&1 &
$CL --model v2 --tasks reach-v3 --outdir out/vision2/eval_v2 > out/vision2/closed_v2.txt 2>&1 &
$CL --model v3 --cameras corner gripperPOV behindGripper --tasks reach-v3 --outdir out/vision3/eval_v3 > out/vision3/closed_v3.txt 2>&1 &
$CL --model vmulti --max-steps 500 --tasks reach-v3 push-v3 pick-place-v3 drawer-open-v3 --outdir out/vision_multi/eval_vmulti > out/vision_multi/closed_vmulti.txt 2>&1 &
$CL --model qwen --max-steps 500 --video 0 --tasks push-v3 pick-place-v3 drawer-open-v3 --outdir out/vision_multi/eval_qwen > out/vision_multi/closed_qwen.txt 2>&1 &
wait
echo done
