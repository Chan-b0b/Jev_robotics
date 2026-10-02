#!/bin/bash
# The evaluations that timed out when eval_all.sh ran five clients on one server,
# split over the servers on GPUs 2 (port 8002) and 3 (port 8003).
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
(
  .venv/bin/python closed_loop.py --vision --policy qwen --url http://127.0.0.1:8002 --episodes 50 --max-steps 500 --model qwen \
    --tasks push-v3 pick-place-v3 drawer-open-v3 --outdir out/vision_multi/eval_qwen > out/vision_multi/closed_qwen.txt 2>&1
  .venv/bin/python lora_eval.py --url http://127.0.0.1:8002 --model v0 --data out/vision/reach_val.jsonl --confusion > out/vision/val_v0.txt 2>&1
  .venv/bin/python lora_eval.py --url http://127.0.0.1:8002 --model v2 --data out/vision2/reach_val.jsonl --confusion > out/vision2/val_v2.txt 2>&1
) &
(
  .venv/bin/python lora_eval.py --url http://127.0.0.1:8003 --model vmulti --data out/vision_multi/val.jsonl --confusion > out/vision_multi/val_vmulti.txt 2>&1
  .venv/bin/python lora_eval.py --url http://127.0.0.1:8003 --model v3 --data out/vision3/reach_val.jsonl --confusion > out/vision3/val_v3.txt 2>&1
) &
wait
echo done
