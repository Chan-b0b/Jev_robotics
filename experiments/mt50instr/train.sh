#!/bin/bash
# Instruction-following executor: from the co-trained adapter (out/mt50co/r16), action rows with no, the
# expert's or a random instruction (instr_data.py) and some language rows, on GPUs 0-3. Validation: the
# T-comp states with random instructions, labelled by the instruction (so accuracy = following).
cd /home/maverick/Jev/jev-manip
D=out/mt50instr/data
CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1 .venv-vllm/bin/torchrun --standalone --nproc_per_node 4 lora_train.py \
  --init out/mt50co/r16 --train $D/train.jsonl --lm $D/lang_l1_sub.jsonl,$D/lang_l2_sub.jsonl \
  --val $D/val_comp_follow.jsonl --out out/mt50instr/r16 --eval-every 2000
echo done
