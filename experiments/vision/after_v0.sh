#!/bin/bash
# After out/vision/reach_r16 (v0) is saved: train the 2-camera upright run on GPU 0,
# serve v0 on GPU 1, and evaluate v0 with its training orientation (corner and gripperPOV rotated).
cd /home/maverick/Jev/jev-manip
until grep -qE "adapter ->|Traceback" logs/vision_train.log; do sleep 30; done
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 nohup .venv-vllm/bin/python lora_train.py --train out/vision2/reach_train.jsonl \
  --val out/vision2/reach_val.jsonl --out out/vision2/reach_r16 --eval-every 250 > logs/vision2_train.log 2>&1 &
pid=$(ps -eo pid,args | awk '$2 ~ /vllm$/ && /--port 8001/ {print $1}'); [ -n "$pid" ] && kill $pid
while nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{exit !($1>1000)}'; do sleep 5; done
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen --port 8001 --host 127.0.0.1 --gpu-memory-utilization 0.3 \
  --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 --attention-backend FLASH_ATTN \
  --enable-lora --max-lora-rank 16 --max-loras 2 --lora-modules v0=out/vision/reach_r16 multi=out/lora/multi_r16 > logs/vllm_lora.log 2>&1 &
until grep -q "Application startup complete" logs/vllm_lora.log; do sleep 5; done
export MUJOCO_GL=egl
.venv/bin/python lora_eval.py --model v0 --data out/vision/reach_val.jsonl --confusion > out/vision/eval_v0_val.txt 2>&1
.venv/bin/python closed_loop.py --vision --upside-down corner gripperPOV --policy qwen --model v0 --url http://127.0.0.1:8001 \
  --tasks reach-v3 --episodes 50 --video 2 --outdir out/vision/eval_v0 > out/vision/eval_v0_closed.txt 2>&1
echo done
