#!/bin/bash
# After gen.sh: LoRA on the ML10 train tasks, 20k rows per task on GPU 0 and 40k on GPU 1.
cd /home/maverick/Jev/jev-manip
until grep -q "^done" logs/ml10_gen.log; do sleep 30; done
D=out/ml10/data
export HF_HUB_OFFLINE=1
for spec in 20k:0 40k:1; do n=${spec%:*} g=${spec#*:}
  CUDA_VISIBLE_DEVICES=$g nohup .venv-vllm/bin/python lora_train.py --train $D/train_$n.jsonl --val $D/val_train.jsonl \
    --out out/ml10/r16_$n --eval-every 1000 > logs/ml10_train_$n.log 2>&1 &
done
wait
echo done
