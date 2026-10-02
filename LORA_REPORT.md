# Qwen3.5-4B LoRA: Meta-World 4개 task (2026-09-28)

## 요약
- **Reach만 학습한 LoRA**: Reach closed-loop 성공률 0.70 → **1.00**, 평균 step 108 → 48
- **4개 task 통합 LoRA**: 전 task closed-loop 성공률 **1.00** (각 50 에피소드)
  - base 모델은 push, pick-place, drawer-open 성공률 0.00
  - val 정확도(task·질문별 한 번 읽기) 0.489 → **0.999**
- 모델이 따라가는 라벨 규칙(task별 procedure)은 프롬프트에 글로 주어집니다. 따라서 이 결과는 "주어진 규칙을 좌표에 정확히 적용하는 능력"을 학습한 것입니다. 규칙 없이 스스로 전략을 세우는 능력은 측정하지 않았습니다.

## 1. 결과

### Closed loop (`closed_loop.py --policy qwen`, seed 5000–5049, 50 에피소드)
| task | 라벨 정책 (상한, 100 ep) | base Qwen | Reach LoRA | 통합 LoRA |
|---|---|---|---|---|
| reach-v3 | 1.00 / 50 step | 0.70 / 108 step | **1.00 / 48** | **1.00 / 48** |
| push-v3 | 1.00 / 114 | 0.00 / 500 | – | **1.00 / 112** |
| pick-place-v3 | 1.00 / 74 | 0.00 / 500 | – | **1.00 / 73** |
| drawer-open-v3 | 1.00 / 101 | 0.00 / 500 | – | **1.00 / 101** |

- 표의 값은 "성공률 / 평균 step"입니다.
- 최대 step: Reach는 200, 나머지는 500입니다. 통합 LoRA는 전 task를 500으로 돌렸습니다.
- 라벨 일치율: 통합 LoRA는 전 task 1.00, Reach LoRA는 reach에서 0.98, base는 reach 0.71 / push 0.27 / pick-place 0.03 / drawer-open 0.12입니다.
- 통합 LoRA의 step 수는 라벨 정책과 거의 같습니다. 즉 라벨 정책을 사실상 그대로 재현합니다.

### Val 정확도 (`lora_eval.py`, `out/lora/multi/val.jsonl` 4,004행)
| 모델 | reach | push | pick-place | drawer-open | 전체 |
|---|---|---|---|---|---|
| base | 0.784 | 0.351 | 0.499 | 0.320 | 0.489 |
| Reach LoRA | 0.993 | 0.606 | 0.767 | 0.673 | 0.760 |
| 통합 LoRA | 1.000 | 0.998 | 0.999 | 0.999 | **0.999** |

- 통합 LoRA의 오답은 4,004개 중 4개입니다: push dz 0.992, pick-place dx 0.996, drawer dz 0.997.
- Reach만 학습했는데도 다른 task의 정확도가 올랐습니다(0.489 → 0.760). 좌표 비교 능력 자체가 전이된 것으로 보입니다.

## 2. 데이터

### Reach: state 직접 샘플링
- Reach 라벨은 (fingertip, goal)만으로 정해지는 함수라서 sim 없이 state를 직접 뽑습니다.
- goal: reach-v3 goal 범위 x[-0.1, 0.1], y[0.8, 0.9], z[0.05, 0.3]에서 균등하게 뽑습니다.
- fingertip: 축마다 50% 확률로 goal ±0.03 안에서 뽑고, 나머지는 박스 x[-0.2, 0.2], y[0.5, 1.0], z[0.03, 0.35]에서 균등하게 뽑습니다.
- 결과적으로 hold가 축마다 약 10%입니다.

### push / pick-place / drawer-open: 노이즈 섞은 rollout
- 이 세 task는 무작위 샘플링을 쓰지 않았습니다. 퍽 위치, 그리퍼 열림 정도, fingertip 위치가 물리적으로 묶여 있어서(퍽은 잡혀 있을 때만 공중에 있음) 무작위로 뽑으면 실제로 나올 수 없는 state가 생깁니다.
- 대신 라벨 정책으로 rollout하면서 30% step은 무작위 행동(xyz 균등, grip 무작위)으로 바꿨습니다. 이렇게 하면 퍽을 떨어뜨리거나 grasp에 실패하는 등 경로에서 벗어난 state도 포함됩니다.
- 라벨은 어느 state에서나 규칙으로 정확히 계산됩니다.
- env seed: train은 100000번대, val은 101000번대, 평가는 5000번대로 서로 겹치지 않습니다.

### 규칙을 fingertip 기준으로 옮김 ([tasks.py](tasks.py))
- [stage2_data.py](stage2_data.py)의 procedure는 hand(obs[:3]) 기준입니다. `tasks.py`는 fingertip(tcp_center) 기준입니다.
- 측정해보니 hand는 fingertip보다 z로 0.045 높았습니다(전 task에서 동일, std 0.0002).
- 기준점 P와 H를 0.045 내리고 임계값은 그대로 두는 방식으로 옮겼습니다.
- push 3단계는 처음에 "fingertip → goal"로 옮겼더니 replay 성공률이 0.92~0.94였습니다. 원래 규칙과 똑같이 "goal을 0.045 내린 지점"으로 바꾸니 1.00이 됐습니다. 이 때문에 push의 dz 라벨은 91%가 decrease입니다(테이블을 누르는 상태가 계속되기 때문).
- 네 task 모두 라벨 replay 성공률 1.00입니다(각 100 에피소드, `out/lora/labels_check/`).

### 규모
| 파일 | 행 수 | 구성 |
|---|---|---|
| `out/lora/reach_train.jsonl` / `reach_val.jsonl` | 60,000 / 3,000 | Reach 전용 (state 20k × 질문 3개) |
| `out/lora/multi/train.jsonl` | 160,004 | task당 약 40k행. push·pick-place는 state 10k × 질문 4개, reach·drawer-open은 13.3k × 3개 |
| `out/lora/multi/val.jsonl` | 4,004 | task당 약 1k행 |

## 3. 학습 설정 ([lora_train.py](lora_train.py))
- 모델: Qwen3.5-4B (`Qwen3_5ForConditionalGeneration`, bf16), GPU는 B300 한 장을 씁니다.
- LoRA: r=16, alpha=32, dropout 0
- 대상 layer: language model의 `q/k/v/o_proj`, linear-attention의 `out_proj`, MLP의 `gate/up/down_proj`. vision tower와 MTP head는 건드리지 않았습니다.
- 학습 파라미터는 23.8M(0.52%)입니다.
- loss: 답 글자 한 토큰에만 CE를 겁니다. 프롬프트는 추론 때와 같은 chat template(`enable_thinking=False`)입니다. linear attention이 recurrent 구조라서 right padding을 쓰고, 마지막 프롬프트 위치의 logit을 사용합니다.
- 최적화: AdamW, lr 1e-4, warmup 5% 후 cosine decay, batch 32, 1 epoch
- Reach 학습: 1,875 step, 0.9s/step, 약 30분. val 곡선 0.792 → 0.984 (250 step) → 0.991 (500) → 0.993
- 통합 학습: 5,001 step, 1.8s/step, 약 2.5시간. 프롬프트 최대 489 토큰. val 곡선 0.489 → 0.961 (500) → 0.990 (1500) → 0.9975 (2500) → 0.999
- `fla`와 `causal_conv1d`는 설치하지 않고 PyTorch 기본 구현으로 돌렸습니다. 설치하면 더 빨라질 수 있습니다.
- vLLM 호환성: smoke adapter로 먼저 확인했습니다. vLLM과 HF에서 잰 정확도가 일치했습니다(base 0.777 vs 0.775, LoRA 0.959 vs 0.961).

## 4. 환경 변경
- `.venv-vllm`에 peft 0.21.0, trl 1.14.0과 의존 패키지(accelerate, datasets 등)를 설치했습니다.
- torch 2.13.0, transformers 5.17.0, vllm 0.30.0은 버전을 고정해서 그대로입니다. fsspec만 2026.9.0 → 2026.6.0으로 바뀌었습니다.
- 학습 스크립트에서 trl은 쓰지 않았습니다(직접 짠 루프).
- `tasks.py`에 Push, PickPlace, DrawerOpen 클래스를 추가했습니다. Reach 클래스는 수정하지 않았습니다. `closed_loop.py`의 기본 `--tasks`는 여전히 reach-v3입니다.
- GPU 7의 기존 vLLM 서버(port 8000)는 건드리지 않았습니다. 평가용 서버(GPU 1, port 8001)는 평가 후 내렸습니다.

## 5. 재현
```bash
# 데이터
.venv/bin/python lora_data.py                                        # reach 60k
.venv/bin/python lora_data.py --task push-v3 --states 10000 --out out/lora/multi/push-v3_train.jsonl
#   (나머지 task·val은 --task/--states/--seed 1로 같은 방식. multi/train.jsonl은 4개 파일을 합쳐서 섞은 것)

# 학습
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv-vllm/bin/python lora_train.py      # -> out/lora/reach_r16
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 .venv-vllm/bin/python lora_train.py \
  --train out/lora/multi/train.jsonl --val out/lora/multi/val.jsonl --out out/lora/multi_r16 --eval-every 500

# 서빙 (serve_qwen.sh 옵션 + LoRA). keep-alive를 늘려야 50 에피소드 동시 실행 중 httpx.ReadError가 안 남
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600 CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  .venv-vllm/bin/vllm serve Qwen/Qwen3.5-4B --served-model-name qwen --port 8001 --host 127.0.0.1 \
  --gpu-memory-utilization 0.3 --max-model-len 8192 --max-num-seqs 256 --enable-prefix-caching --max-logprobs 20 \
  --attention-backend FLASH_ATTN --enable-lora --max-lora-rank 16 --max-loras 2 \
  --lora-modules reach=out/lora/reach_r16 multi=out/lora/multi_r16

# 평가
.venv/bin/python lora_eval.py --model multi
MUJOCO_GL=egl .venv/bin/python closed_loop.py --policy qwen --model multi --url http://127.0.0.1:8001 \
  --tasks reach-v3 push-v3 pick-place-v3 drawer-open-v3 --episodes 50 --max-steps 500 --video 2 --outdir out/lora/eval_multi_multi
```

## 6. 산출물
- adapter: `out/lora/reach_r16/`, `out/lora/multi_r16/` (각 91MB, `log.jsonl`에 val 곡선)
- 평가 결과
  - `out/lora/eval_reach_{qwen,reach}/`: reach closed loop, base vs Reach LoRA
  - `out/lora/eval_multi_qwen/`: base, 3개 task
  - `out/lora/eval_multi_multi/`: 통합 LoRA, 4개 task, task당 mp4 2개
- 로그: `logs/lora_train.log`, `logs/lora_train_multi.log`, `logs/vllm_lora.log`

## 7. 한계와 다음 단계
- **규칙이 프롬프트에 있습니다.** 모델은 procedure에 좌표를 대입하는 것을 배웠습니다. 규칙 없이 목표만 주는 설정(stage2의 `goal` variant)에서도 되는지는 따로 실험해야 합니다.
- **분포 밖 일반화는 측정하지 않았습니다.** 평가는 학습과 같은 env 초기 분포이고 seed만 다릅니다. 확인할 만한 것:
  - 학습에 없던 task
  - 다른 임계값이나 오프셋을 쓴 규칙
  - 좌표 범위 밖의 goal
- val과 closed-loop 모두 거의 포화 상태입니다. 쉽게 해볼 수 있는 방향:
  - 더 적은 데이터나 더 작은 rank로 효율 비교 (Reach는 250 step, 약 8k 샘플에서 이미 0.984)
  - 시각 입력(카메라 프레임)으로 확장
