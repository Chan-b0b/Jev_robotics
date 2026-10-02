#!/bin/bash
# Text-swap probe: train-task validation rows with the task text replaced by another train task's
# description (val_train_swapped.jsonl), for the 20k (8021) and 40k (8031) adapters.
cd /home/maverick/Jev/jev-manip
D=out/ml10_3cam/data O=out/ml10_3cam/val; pids=()
for spec in 20k:8021 40k:8031; do m=${spec%:*} p=${spec#*:}
  .venv/bin/python lora_eval.py --url http://127.0.0.1:$p --model ml10_$m --data $D/val_train_swapped.jsonl --confusion > $O/${m}_train_swapped.txt 2>&1 & pids+=($!)
  .venv/bin/python lora_eval.py --url http://127.0.0.1:$p --model ml10_$m --data $D/val_train.jsonl > $O/${m}_train_fixed.txt 2>&1 & pids+=($!)
done
wait "${pids[@]}"
echo done
