#!/bin/bash
# After gen.sh: LoRA on the ML10 train tasks with three cameras, 20k rows per task on GPU 2 and
# 40k on GPU 3 (batches of 32 as two micro-batches of 16: three cameras run out of memory at 32).
cd /home/maverick/Jev/jev-manip
until grep -q "^done" logs/ml10_3cam_gen.log; do sleep 30; done
D=out/ml10_3cam/data
export HF_HUB_OFFLINE=1
for spec in 20k:2 40k:3; do n=${spec%:*} g=${spec#*:}
  CUDA_VISIBLE_DEVICES=$g nohup .venv-vllm/bin/python lora_train.py --train $D/train_$n.jsonl --val $D/val_train.jsonl \
    --out out/ml10_3cam/r16_$n --eval-every 1000 --accum 2 > logs/ml10_3cam_train_$n.log 2>&1 &
done
wait
echo done
