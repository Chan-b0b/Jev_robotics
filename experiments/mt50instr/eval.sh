#!/bin/bash
# Evaluate the instruction-following adapter (out/mt50instr/r16): servers on GPUs 0-3 (ports 8071-8074, one at
# a time), validation accuracy without, with the expert's, with a wrong and with a random instruction, then
# videos and 50-episode closed loops on the test tasks with the expert's instruction each step (oracle) and
# without. Clients render on their server's GPU, at most 12 at once.
cd /home/maverick/Jev/jev-manip
export MUJOCO_GL=egl
E=out/mt50instr/eval
for g in 0 1 2 3; do p=$((8071 + g))
  ss -ltn | grep -q ":$p " && { echo "port $p is taken"; exit 1; }
  VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=$g HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B \
    --served-model-name qwen --port $p --host 127.0.0.1 --gpu-memory-utilization 0.5 \
    --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 --attention-backend FLASH_ATTN \
    --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules mt50instr=out/mt50instr/r16 > logs/vllm_mt50instr_$g.log 2>&1 &
  s=$!
  until curl -s localhost:$p/v1/models | grep -q mt50instr; do sleep 10; kill -0 $s 2>/dev/null || { echo "server on $p failed"; exit 1; }; done
  echo "server on GPU $g up"
done
i=0; vp=()
for f in out/mt50/data/val_comp_a out/mt50/data/val_obj_a out/mt50/data/val_id \
         out/mt50co/textprobe/val_comp_a_instr out/mt50co/textprobe/val_comp_a_wrong \
         out/mt50co/textprobe/val_obj_a_instr out/mt50co/textprobe/val_obj_a_wrong \
         out/mt50instr/data/val_comp_follow out/mt50instr/data/val_obj_follow out/mt50instr/data/val_id_follow; do
  p=$((8071 + i % 4)); i=$((i + 1))
  .venv/bin/python lora_eval.py --url http://127.0.0.1:$p --model mt50instr --data $f.jsonl --confusion > $E/$(basename $f).txt 2>&1 & vp+=($!)
done
wait "${vp[@]}"
echo "val done"
jobs() { .venv/bin/python - "$1" "$2" <<'PY'
import sys, prompts_mt50 as P
eps, video = int(sys.argv[1]), sys.argv[2] == "video"
CL = (".venv/bin/python closed_loop.py --suite mt50 --vision --cameras corner gripperPOV behindGripper "
      "--policy qwen --model mt50instr --max-steps 500 --look train --procedure none")
for i, (t, ins) in enumerate((t, ins) for t in P.T_COMP + P.T_OBJ for ins in ("oracle", "none")):
    g, name = i % 4, f"{'video_' if video else ''}{ins}_{t}"
    opt = "--instruct oracle" if ins == "oracle" else ""
    extra = "--video 1 --after-success 40" if video else ""
    print(f"MUJOCO_EGL_DEVICE_ID={g} {CL} {opt} --url http://127.0.0.1:{8071 + g} --episodes {eps} {extra} "
          f"--tasks {t} --outdir out/mt50instr/eval/{name} > out/mt50instr/eval/{name}.txt 2>&1")
PY
}
jobs 1 video | xargs -P 12 -I{} sh -c "{}"
for d in $E/video_*/; do n=$(basename $d); ins=${n#video_}; ins=${ins%%_*}; t=${n##*_}; f=$d/${t}_qwen_5000.mp4
  [ -f $f ] && mkdir -p out/demo/mt50instr/$ins && cp $f out/demo/mt50instr/$ins/$t.mp4; done
echo "videos done"
jobs 50 run | xargs -P 12 -I{} sh -c "{}"
echo done
