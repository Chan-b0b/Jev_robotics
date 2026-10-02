#!/bin/bash
# Base Qwen (no LoRA) on the ML10 test tasks, three cameras, (a) description only and (b) with a
# procedure: 50 episodes each, one client per task and prompt, on a server on GPU 1 (port 8010; 0.3 of its memory, as a first run on GPU 0 ran out when another process took GPU memory).
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen --port 8010 --host 127.0.0.1 --gpu-memory-utilization 0.3 \
  --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 --attention-backend FLASH_ATTN \
  > logs/vllm_ml10_base.log 2>&1 &
until grep -q "Application startup complete" logs/vllm_ml10_base.log; do sleep 5; done
for t in drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3; do for p in none all; do
  .venv/bin/python closed_loop.py --suite ml10 --vision --cameras corner gripperPOV behindGripper --policy qwen \
    --url http://127.0.0.1:8010 --model qwen --procedure $p --episodes 50 --max-steps 500 --tasks $t \
    --outdir out/ml10_3cam/eval/base_${p}_$t > out/ml10_3cam/eval/base_${p}_$t.txt 2>&1 &
done; done
wait
echo done
