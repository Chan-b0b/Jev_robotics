#!/bin/bash
# After the 40k 3-camera adapter is saved: videos (out/demo/ml10_3cam_40k/<task>.mp4 for (a), and
# <task>_procedure.mp4 for the test tasks with (b)), then 50-episode closed loops like the 20k's.
# One server on GPU 2 (port 8031, checked free first); waits on the clients only.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
A=out/ml10_3cam/r16_40k P=8031
until [ -f $A/adapter_model.safetensors ]; do sleep 60; done
ss -ltn | grep -q ":$P " && { echo "port $P is taken"; exit 1; }
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen --port $P --host 127.0.0.1 --gpu-memory-utilization 0.5 \
  --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 --attention-backend FLASH_ATTN \
  --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules ml10_40k=$A > logs/vllm_ml10_40k.log 2>&1 &
server=$!
until curl -s localhost:$P/v1/models | grep -q ml10_40k; do
  sleep 10; kill -0 $server 2>/dev/null || { echo "server failed, see logs/vllm_ml10_40k.log"; exit 1; }
done
echo "server up"
TEST="drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3"
TRAIN="reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 sweep-v3 basketball-v3"
CL=".venv/bin/python closed_loop.py --suite ml10 --vision --cameras corner gripperPOV behindGripper --policy qwen --model ml10_40k --max-steps 500 --url http://127.0.0.1:$P"
V=out/ml10_3cam/demo_40k; mkdir -p $V out/demo/ml10_3cam_40k; pids=()
for t in $TEST $TRAIN; do $CL --procedure none --episodes 1 --video 1 --after-success 40 --tasks $t --outdir $V/a_$t > $V/a_$t.txt 2>&1 & pids+=($!); done
for t in $TEST; do $CL --procedure all --episodes 1 --video 1 --after-success 40 --tasks $t --outdir $V/b_$t > $V/b_$t.txt 2>&1 & pids+=($!); done
wait "${pids[@]}"
for t in $TEST $TRAIN; do cp $V/a_$t/${t}_qwen_5000.mp4 out/demo/ml10_3cam_40k/$t.mp4; done
for t in $TEST; do cp $V/b_$t/${t}_qwen_5000.mp4 out/demo/ml10_3cam_40k/${t}_procedure.mp4; done
echo "videos done"
E=out/ml10_3cam/eval; pids=()
for t in $TEST; do for pr in none all; do
  $CL --procedure $pr --episodes 50 --tasks $t --outdir $E/40k_${pr}_$t > $E/40k_${pr}_$t.txt 2>&1 & pids+=($!)
done; done
for t in $TRAIN; do $CL --procedure none --episodes 50 --tasks $t --outdir $E/40k_none_$t > $E/40k_none_$t.txt 2>&1 & pids+=($!); done
wait "${pids[@]}"
echo done
