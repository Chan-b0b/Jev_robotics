#!/bin/bash
# LoRA on the mt50 train tasks (760k rows) on GPUs 0-3, with fla kernels; the adapter goes to
# out/mt50/r16 (on /data001 through the out/mt50 link). Validation on the T-id rows.
cd /home/maverick/Jev/jev-manip
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 .venv-vllm/bin/torchrun --standalone --nproc_per_node 4 lora_train.py \
  --train out/mt50/data/train.jsonl --val out/mt50/data/val_id.jsonl --out out/mt50/r16 --eval-every 2500
echo done
