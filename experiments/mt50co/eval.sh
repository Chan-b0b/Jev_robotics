#!/bin/bash
# Evaluate the co-trained adapter (out/mt50co/r16) as out/mt50/eval.sh did the mt50 one: servers on GPUs 0-3
# (ports 8061-8064, one at a time, with both adapters), the language probe (base, mt50, mt50co), validation
# accuracy, then videos (seed 5000) and closed loops. Each client renders on the GPU of its server
# (MUJOCO_EGL_DEVICE_ID) and at most 12 run at once: the mt50 run's clients all rendered on GPU 0 and
# ran out of framebuffers.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
E=out/mt50co/eval D=out/mt50/data
for g in 0 1 2 3; do p=$((8061 + g))
  ss -ltn | grep -q ":$p " && { echo "port $p is taken"; exit 1; }
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
    --served-model-name qwen --port $p --host 127.0.0.1 --gpu-memory-utilization 0.5 \
    --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 --attention-backend FLASH_ATTN \
    --enable-lora --max-lora-rank 16 --max-loras 2 --lora-modules mt50co=out/mt50co/r16 mt50=out/mt50/r16 > logs/vllm_mt50co_$g.log 2>&1 &
  s=$!
  until curl -s localhost:$p/v1/models | grep -q mt50co; do sleep 10; kill -0 $s 2>/dev/null || { echo "server on $p failed"; exit 1; }; done
  echo "server on GPU $g up"
done
PYTHONPATH=. .venv/bin/python out/mt50/probe/probe.py http://127.0.0.1:8061 qwen,mt50,mt50co $E/probe.json > $E/probe.txt 2>&1 &
pr=$!
v=0; vp=()
for f in val_id val_vis val_comp_a val_comp_b val_obj_a val_obj_b; do p=$((8062 + v % 3)); v=$((v + 1))
  .venv/bin/python lora_eval.py --url http://127.0.0.1:$p --model mt50co --data $D/$f.jsonl --confusion > $E/$f.txt 2>&1 & vp+=($!)
done
wait $pr "${vp[@]}"
echo "probe and val done"
jobs=$(.venv/bin/python - <<'PY'
import prompts_mt50 as P
CL = (".venv/bin/python closed_loop.py --suite mt50 --vision --cameras corner gripperPOV behindGripper "
      "--policy qwen --model mt50co --max-steps 500")
E, i = "out/mt50co/eval", 0
def job(name, task, look, proc, eps, video):
    global i
    g = i % 4; i += 1
    extra = "--video 1 --after-success 40" if video else ""
    print(f"MUJOCO_EGL_DEVICE_ID={g} {CL} --url http://127.0.0.1:{8061 + g} --look {look} --procedure {proc} "
          f"--episodes {eps} {extra} --tasks {task} --outdir {E}/{name} > {E}/{name}.txt 2>&1")
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
copy_videos() {
  for d in $E/video_*/; do n=$(basename $d); t=${n##*_}; f=$d/${t}_qwen_5000.mp4; [ -f $f ] || continue
    case $n in video_test_a_*) o=out/demo/mt50co/test_a/$t.mp4;; video_test_b_*) o=out/demo/mt50co/test_b/$t.mp4;;
               video_id_*) o=out/demo/mt50co/id/$t.mp4;; video_vis_*) o=out/demo/mt50co/vis/$t.mp4;; esac
    mkdir -p $(dirname $o); cp $f $o
  done
}
echo "$jobs" | head -96 | xargs -P 12 -I{} sh -c "{}"
copy_videos
echo "videos done"
echo "$jobs" | tail -n +97 | xargs -P 12 -I{} sh -c "{}"
echo done
