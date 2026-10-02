#!/bin/bash
# The text probes for mt50co and mt50 on eval.sh's servers (ports 8061-8064) once all four are up.
cd /home/maverick/Jev/jev-manip
O=out/mt50co/textprobe
for p in 8061 8062 8063 8064; do until curl -s localhost:$p/v1/models | grep -q mt50co; do sleep 20; done; done
i=0; pids=()
for m in mt50co mt50; do for f in val_id_swapped val_comp_a_instr val_comp_a_wrong val_obj_a_instr val_obj_a_wrong; do
  p=$((8061 + i % 4)); i=$((i + 1))
  .venv/bin/python lora_eval.py --url http://127.0.0.1:$p --model $m --data $O/$f.jsonl > $O/${m}_$f.txt 2>&1 & pids+=($!)
done; done
wait "${pids[@]}"
echo done
