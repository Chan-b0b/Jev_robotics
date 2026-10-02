#!/bin/bash
# The 20k 3-camera adapter: videos, then 50-episode closed loops. Test tasks on a server on GPU 0
# (port 8021; 8011 is taken by another user's process), train tasks on eval_lora.sh's server on GPU 1 (8012).
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
TEST="drawer-open-v3 door-close-v3 shelf-place-v3 sweep-into-v3 lever-pull-v3"
TRAIN="reach-v3 push-v3 pick-place-v3 door-open-v3 drawer-close-v3 button-press-topdown-v3 peg-insert-side-v3 window-open-v3 sweep-v3 basketball-v3"
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
  --served-model-name qwen --port 8021 --host 127.0.0.1 --gpu-memory-utilization 0.3 \
  --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 --attention-backend FLASH_ATTN \
  --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules ml10_20k=out/ml10_3cam/r16_20k > logs/vllm_ml10_20k_gpu0.log 2>&1 &
for p in 8021 8012; do until curl -s localhost:$p/v1/models | grep -q ml10_20k; do sleep 10; done; done
url() { [[ " $TEST " == *" $1 "* ]] && echo http://127.0.0.1:8021 || echo http://127.0.0.1:8012; }
CL=".venv/bin/python closed_loop.py --suite ml10 --vision --cameras corner gripperPOV behindGripper --policy qwen --model ml10_20k --max-steps 500"
# videos: (a) every task, (b) test tasks
V=out/ml10_3cam/demo_20k; mkdir -p $V out/demo/ml10_3cam_20k
for t in $TEST $TRAIN; do $CL --url $(url $t) --procedure none --episodes 1 --video 1 --after-success 40 --tasks $t --outdir $V/a_$t > $V/a_$t.txt 2>&1 & done
for t in $TEST; do $CL --url $(url $t) --procedure all --episodes 1 --video 1 --after-success 40 --tasks $t --outdir $V/b_$t > $V/b_$t.txt 2>&1 & done
wait
for t in $TEST $TRAIN; do cp $V/a_$t/${t}_qwen_5000.mp4 out/demo/ml10_3cam_20k/$t.mp4; done
for t in $TEST; do cp $V/b_$t/${t}_qwen_5000.mp4 out/demo/ml10_3cam_20k/${t}_procedure.mp4; done
echo videos done
# 50 episodes: test tasks (a) and (b), train tasks (a)
E=out/ml10_3cam/eval
for t in $TEST; do for pr in none all; do $CL --url $(url $t) --procedure $pr --episodes 50 --tasks $t --outdir $E/20k_${pr}_$t > $E/20k_${pr}_$t.txt 2>&1 & done; done
for t in $TRAIN; do $CL --url $(url $t) --procedure none --episodes 50 --tasks $t --outdir $E/20k_none_$t > $E/20k_none_$t.txt 2>&1 & done
wait
echo done
