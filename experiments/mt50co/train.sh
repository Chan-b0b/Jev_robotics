#!/bin/bash
# Co-training on GPUs 4-7: the mt50 action rows (760k) with language rows, L1 scene descriptions from
# base Qwen (40k) and L2 motion in words from the experts (152k), lang_data.py. Adapter: out/mt50co/r16.
cd /home/maverick/Jev/jev-manip
D=out/mt50/data
CUDA_VISIBLE_DEVICES=4,5,6,7 HF_HUB_OFFLINE=1 .venv-vllm/bin/torchrun --standalone --nproc_per_node 4 lora_train.py \
  --train $D/train.jsonl --lm $D/lang_l1.jsonl,$D/lang_l2.jsonl --val $D/val_id.jsonl --out out/mt50co/r16 --eval-every 2500
echo done
