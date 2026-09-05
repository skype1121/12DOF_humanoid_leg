# 0906 야간 자율 작업에 쓴 판정·학습·패키지 스크립트 (보존용 사본)

원본은 세션 스크래치(`/tmp/claude-1000/.../scratchpad`)에 있었고 경로가 하드코딩돼 있다(SCRATCH·REPO). 재사용 시 상단 경로만 고칠 것.

| 스크립트 | 역할 |
|---|---|
| `judge_gen.sh <태그> <run> <ckpt> <task> <usd>` | 세대 판정 전체(사다리 0~3 DR OFF/ON + 배포조건 old/new·com0/16·dr_on + 명목 + V6 고정환경) → `judge_<태그>.txt` |
| `judge_gen_h3.sh` | 배포 경로(h3)만 7런 + 대조군 old_h2. `EXTRA="--ablate gains"` 로 h3g |
| `judge_gen_h3s.sh` | h3 + 정착 1s + ±3° 게이트(3런). `EXTRA` 동일 |
| `train_gen.sh <태그> <task> <warm_run> <warm_ckpt> <iters>` | 웜스타트 학습 런치(4096env) |
| `train_chain_h3.sh` / `train_chain_v20.sh` / `h3_chain.sh` | 0906 체인(v18c→v19 / v20 / h3 재판정) — 10:00 이후 새 학습 생략 규칙 포함 |
| `finalize_gen.sh <tag> <name> <task> <warm_desc>` | export → gen_report → register_model → register_bridge → selftest → deploy_package |
| `register_bridge.py <name> <run>` | 브리지 CHECKPOINTS/CKPT_GAIT_FREQ 멱등 삽입 |
| `pkg_verify.sh` / `repro_jetson_selftest.sh` | 패키지 조립 → 젯슨형 트리(하드웨어맵+기배포 세대, stand_v14 없음) 풀기 → md5 → selftest 3종 |
