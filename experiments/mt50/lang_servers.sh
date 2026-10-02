#!/bin/bash
# Base Qwen servers (no LoRA) on GPUs 4-7, ports 8051-8054, one at a time, for lang_data.py l1.
cd /home/maverick/Jev/jev-manip
for g in 4 5 6 7; do p=$((8047 + g))
  ss -ltn | grep -q ":$p " && { echo "port $p is taken"; exit 1; }
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
    --served-model-name qwen --port $p --host 127.0.0.1 --gpu-memory-utilization 0.5 \
    --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --attention-backend FLASH_ATTN > logs/vllm_lang_$g.log 2>&1 &
  s=$!
  until curl -s localhost:$p/v1/models | grep -q qwen; do sleep 10; kill -0 $s 2>/dev/null || { echo "server on $p failed"; exit 1; }; done
  echo "server on GPU $g up (port $p, pid $s)"
done
echo done
