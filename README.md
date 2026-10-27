# Local LLM pipeline and sovereign model experiments

RTX 3090 한 대에서 진행한 로컬 LLM 생성·학습·평가·양자화 실험의 소스 스냅샷입니다. Qwen3 계열의 4B/8B 모델, QLoRA SFT, reward-ranked fine-tuning(RAFT), DPO, GGUF 변환과 llama.cpp 서빙 스크립트를 포함합니다.

## 포함된 실험

- `server/model-build/`: 3090 서버의 생성, QLoRA, RAFT, 평가, GGUF 빌드 스크립트.
- `server/home-scripts/`: 홈에 따로 있던 생성·서빙·복구 스크립트.
- `mac/model-build/`: 맥에 남아 있던 별도 실험 사본. 서버 사본과 같은 이름이어도 내용이 다를 수 있습니다.
- `mac/model-build/ton3079/`: 소형 모델의 보안 분류·앙상블 평가 코드.
- `mac/sovereign-neg-signal/`: 부정 신호를 사용한 DPO 실험.

소버린 AI 실험은 에이전트 작업 문맥을 읽고 다음 행동·작업 상태를 출력하는 모델을 학습·평가하는 작업입니다. 독립적인 범용 에이전트 제품이나 검증된 배포 모델을 뜻하지 않습니다. LLM 파이프라인과 학습·평가 코드를 공유하므로 한 레포에 함께 보존했습니다.

## 주요 진입점

| 단계 | 파일 예시 |
| --- | --- |
| 로컬 모델 응답 생성 | `server/model-build/gen_v3_safe.py`, `gen_v3_resumable.py`, `gen_27b.py` |
| QLoRA SFT | `server/model-build/sft_train.py`, `run_sft_8b.sh` |
| RAFT rollout → reward → select | `server/model-build/raft.py`, `run_raft_8b.sh` |
| DPO | `server/model-build/dpo_train.py`, `build_dpo_pairs.py` |
| 평가 | `server/model-build/eval_harness.py`, `eval_v2_gen.py`, `eval_v2_local/` |
| GGUF 변환·양자화 | `server/model-build/build_gguf_v0.sh`, `build_gguf_iter0.sh`, `build_gguf_iter1.sh` |
| llama.cpp 서빙 | `server/model-build/serve_v0.sh`, `serve_v1.sh`, `server/home-scripts/start-llama.sh` |

## 재사용 범위

원래의 경로·서비스명·모델 경로를 사용하는 연구용 스크립트입니다. 설치 패키지나 처음부터 끝까지 자동 실행되는 배포판으로 정리한 것은 아닙니다. 실행 전에 스크립트의 경로와 서비스 제어 동작을 본인 환경에 맞게 검토해야 합니다.

학습에는 Python, CUDA/PyTorch, transformers, datasets, peft, trl, bitsandbytes가 사용되며, GGUF 빌드·서빙에는 별도 llama.cpp 체크아웃이 필요합니다. 재현 가능한 의존성 잠금 파일은 원본에서 확인하지 못했습니다. 학습 데이터·모델 가중치·어댑터는 제공하지 않으므로 사용자가 준비해야 합니다.

`sft_train.py --dry-run`은 설정만 출력하는 진입점입니다. `--validate-data`는 데이터와 토크나이저가 필요합니다. 학습·배포 셸 스크립트는 서비스 정지·재시작이나 GPU 작업을 포함할 수 있습니다.

## 공개에서 제외한 자료

자격증명 및 의심되는 자격증명 할당이 있는 파일, 사설 네트워크 주소가 있는 배포·운영 파일, 개인정보 후보가 있는 파일, 원본 데이터셋과 모델 출력, 가중치·로그·백업·DB·캐시·기존 Git 이력은 포함하지 않았습니다. 데이터 추출·구성 스크립트 일부도 파일 단위 제외 대상이므로 학습 데이터는 별도로 준비해야 합니다.

원본 데이터에는 합성 응답뿐 아니라 실제 작업 문맥에서 만든 학습 자료도 섞여 있었습니다. 해당 자료는 공개하지 않습니다. 이 스냅샷에는 과거 실험의 성능이나 성공 여부를 보증하는 주장이 없습니다.

공개 전 점검은 비밀 탐지와 Python/셸 구문 검사입니다. GPU 학습, 외부 API 호출, 모델 다운로드, 서비스 배포는 실행하지 않았습니다.
