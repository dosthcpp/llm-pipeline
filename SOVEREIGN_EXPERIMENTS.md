# 소버린 AI 실험

목표는 작은 로컬 모델이 에이전트의 작업 문맥을 읽고 다음 행동과 작업 상태를 출력하도록 학습·평가하는 것이었습니다. Qwen3 계열의 4B/8B QLoRA 학습과 RAFT, DPO 실험이 LLM 파이프라인과 같은 소스를 사용합니다.

주요 파일은 각 사본의 `sft_train.py`, `raft.py`, `eval_v2_gen.py`, `eval_v2_judge_prep.py`, `eval_v2_local/local_teacher_judge.py`입니다. 맥 사본의 `ton3079/`에는 보안 분류·앙상블 평가 코드가 있고, `mac/sovereign-neg-signal/`에는 별도 DPO 실험이 있습니다.

공개 범위는 실험 코드입니다. 실제 작업 로그로 구성한 학습 자료, 생성 응답, 모델 가중치 및 어댑터는 포함하지 않습니다. 완성된 범용 에이전트나 Rewind 검색 모델로 제시하지 않습니다. 과거 실험 결과를 이번 공개를 통해 재검증한 것은 아닙니다.
