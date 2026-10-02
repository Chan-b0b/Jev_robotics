#!/bin/bash
# Closed loop of an ML10 3-camera adapter once it is saved: test tasks with (a) no procedure and
# (b) a procedure, and train tasks with (a), 50 episodes each, one client per task and prompt.
# Test tasks on a server on GPU 0 (port 8011), train tasks on GPU 1 (port 8012).
# Usage: eval_lora.sh <name, e.g. 20k>
cd /home/maverick/Jev/jev-manip
n=$1 A=out/ml10_3cam/r16_$n E=out/ml10_3cam/eval
until [ -f $A/adapter_model.safetensors ]; do sleep 60; done
export MUJOCO_GL=egl
for spec in 0:8011 1:8012; do g=${spec%:*} p=${spec#*:}
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
    --served-model-name qwen --port $p --host 127.0.0.1 --gpu-memory-utilization 0.3 \
    --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 --attention-backend FLASH_ATTN \
    --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules ml10_$n=$A > logs/vllm_ml10_$n_$g.log 2>&1 &
done
for p in 8011 8012; do until curl -s localhost:$p/v1/models | grep -q ml10_$n; do sleep 10; done; done
CL=".venv/bin/python closed_loop.py --suite ml10 --vision --cameras corner gripperPOV behindGripper --policy qwen --model ml10_$n --episodes 50 --max-steps 500"
for t in drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3; do for pr in none all; do
  $CL --url http://127.0.0.1:8011 --procedure $pr --tasks $t --outdir $E/${n}_${pr}_$t > $E/${n}_${pr}_$t.txt 2>&1 &
done; done
for t in reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 sweep-v3 basketball-v3; do
  $CL --url http://127.0.0.1:8012 --procedure none --tasks $t --outdir $E/${n}_none_$t > $E/${n}_none_$t.txt 2>&1 &
done
wait
echo done
