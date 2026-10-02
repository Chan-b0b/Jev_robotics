#!/bin/bash
# vLLM for Qwen3.5-4B on GPU 7 (shared server: other jobs use GPUs 4-7, so cap our share at 30%).
# No nvcc here: FlashInfer's sampler and its sm103 TRT-LLM attention both JIT-compile, so neither is used.
cd "$(dirname "$0")"
mkdir -p logs
CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 exec .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen --port 8000 --host 127.0.0.1 --gpu-memory-utilization 0.3 \
  --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 --attention-backend FLASH_ATTN
