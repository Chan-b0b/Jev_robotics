#!/bin/bash
# Evaluate the mt50 adapter (out/mt50/r16): servers on GPUs 0-3 (ports 8041-8044, checked free), then
# per-question validation accuracy, and closed loops with videos, at most 24 clients at once (EGL):
# T-comp and T-obj with (a) and (b), train looks, 50 episodes; T-id (train looks) and T-vis (test looks)
# on the train tasks with (a), 30 episodes; one video per (set, task, prompt) in out/demo/mt50/.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
E=out/mt50/eval D=out/mt50/data
servers=()
for g in 0 1 2 3; do p=$((8041 + g))
  ss -ltn | grep -q ":$p " && { echo "port $p is taken"; exit 1; }
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
    --served-model-name qwen --port $p --host 127.0.0.1 --gpu-memory-utilization 0.5 \
    --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 --attention-backend FLASH_ATTN \
    --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules mt50=out/mt50/r16 > logs/vllm_mt50_$g.log 2>&1 &
  servers+=($!)
  # one server at a time: the GPU driver hung once just after four started together
  until curl -s localhost:$p/v1/models | grep -q mt50; do sleep 10; kill -0 ${servers[$g]} 2>/dev/null || { echo "server on $p failed"; exit 1; }; done
  echo "server on GPU $g up"
done
echo "servers up"
# validation accuracy, one file per server
v=0; vp=()
for f in val_id val_vis val_comp_a val_comp_b val_obj_a val_obj_b; do p=$((8041 + v % 4)); v=$((v + 1))
  .venv/bin/python lora_eval.py --url http://127.0.0.1:$p --model mt50 --data $D/$f.jsonl --confusion > $E/$f.txt 2>&1 & vp+=($!)
done
wait "${vp[@]}"
echo "val done"
# closed loops and videos
jobs=$(.venv/bin/python - <<'PY'
import prompts_mt50 as P
CL = (".venv/bin/python closed_loop.py --suite mt50 --vision --cameras corner gripperPOV behindGripper "
      "--policy qwen --model mt50 --max-steps 500")
E, i = "out/mt50/eval", 0
def job(name, task, look, proc, eps, video):
    global i
    port = 8041 + i % 4; i += 1
    extra = "--video 1 --after-success 40" if video else ""
    print(f"{CL} --url http://127.0.0.1:{port} --look {look} --procedure {proc} --episodes {eps} {extra} "
          f"--tasks {task} --outdir {E}/{name} > {E}/{name}.txt 2>&1")
for t in P.T_COMP + P.T_OBJ:
    for v, proc in (("a", "none"), ("b", "all")):
        job(f"video_test_{v}_{t}", t, "train", proc, 1, True)
for t in P.TRAIN:
    job(f"video_id_{t}", t, "train", "none", 1, True)
    job(f"video_vis_{t}", t, "test", "none", 1, True)
for t in P.T_COMP + P.T_OBJ:
    for v, proc in (("a", "none"), ("b", "all")):
        job(f"test_{v}_{t}", t, "train", proc, 50, False)
for t in P.TRAIN:
    job(f"id_{t}", t, "train", "none", 30, False)
    job(f"vis_{t}", t, "test", "none", 30, False)
PY
)
echo "$jobs" | xargs -P 24 -I{} sh -c "{}"
for d in $E/video_*/; do n=$(basename $d); t=${n##*_}
  case $n in video_test_a_*) o=out/demo/mt50/test_a/$t.mp4;; video_test_b_*) o=out/demo/mt50/test_b/$t.mp4;;
             video_id_*) o=out/demo/mt50/id/$t.mp4;; video_vis_*) o=out/demo/mt50/vis/$t.mp4;; esac
  mkdir -p $(dirname $o); cp $d/${t}_qwen_5000.mp4 $o 2>/dev/null
done
echo done
