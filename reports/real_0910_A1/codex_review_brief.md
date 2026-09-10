# Codex(gpt-6-astra) 검토 의뢰 — 독립 반박 리뷰 (읽기 전용, 파일 수정 금지)

대상: `reports/real_0910_A1/plan_next_test.md` (다음 실물 시험 계획), 근거 데이터 `reports/real_0910_A1/trial{1,2,3}/summary_bb.md`, 블랙박스 `reports/real_0910_A1/trial*/rl_bridge_*.jsonl`(필드 설명은 `docs/A1_stand_hang_v5_실행카드.md` 7절), 시뮬 지지 모델 `rl_walking/mdp_hang.py`(hang_support_reset/_apply/_ratio), 러너 관절 순서 `rl_walking/deploy/policy_runner.py` JOINT_ORDER(좌우 교대).

역할: 계획의 결론을 **반박**하라. 특히
1. "실패 = 팽팽한 로프의 뒤꿈치 강체 회전" 주장에 대한 대안 설명이 있는가? (예: 발 슬립, 운영자 재인장, 발목롤 자세불일치, 로프 앞뒤 2줄 기하) 블랙박스로 구분 가능한 검증 방법을 제시.
2. 절차 A1 "60% 고원에서 2초 안에 20~30 cm 한 번에 슬랙"의 위험(급방출 시 정책 분포 밖·낙하 충격·양발 이탈 가드)을 시뮬 release_range(2~8 s)와 대조해 평가.
3. 시뮬 케이블 모델(B1) 설계의 빠진 항: 앵커 기하, 길이 스케줄, 재인장, 탄성, 부분접촉 관측 — 무엇을 랜덤화해야 v5 재현→v6 일반화가 되는가.
4. 관절 라벨 정정(JOINT_ORDER 좌우 교대)이 맞는지 `policy_runner.py`와 `summary_bb.md`를 대조해 확인.
출력: 반박 항목별 근거(파일·시각·수치)와 판정(동의/반박/보류), 300~600단어, 파일은 만들지 말고 터미널에 출력.
