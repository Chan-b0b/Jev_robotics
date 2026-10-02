#!/bin/bash
# Per-question accuracy and confusion of the 20k (port 8021) and 40k (port 8031) adapters on the
# validation rows: test tasks with (a) and (b), and train tasks. Waits on the clients only.
cd /home/maverick/Jev/jev-manip
D=out/ml10_3cam/data O=out/ml10_3cam/val; pids=()
for spec in 20k:8021 40k:8031; do m=${spec%:*} p=${spec#*:}
  for v in test_a test_b train; do
    .venv/bin/python lora_eval.py --url http://127.0.0.1:$p --model ml10_$m --data $D/val_$v.jsonl --confusion \
      > $O/${m}_$v.txt 2>&1 & pids+=($!)
  done
done
wait "${pids[@]}"
echo done
