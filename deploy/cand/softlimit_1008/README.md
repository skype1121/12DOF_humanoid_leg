# softlimit_1008 — 시뮬 대조 평가용 한계 표 (실물 제안 아님 · 미배포)

- `joint_limits_12dof.json`: 현 표에서 힙 굴곡 soft 를 60°(양쪽), 오른 힙 외전 soft 를 34° 로 넓힌 것. X17b·X17c 의 한 팔.
- `joint_limits_ankle_only_SIMONLY.json`: 힙·무릎 soft 를 hard 까지 풀고 발목만 현 표대로 둔 것. X17c 의 원인 분리용 팔.
- **결론(2026-10-08 06:06, `reports/gen_w3/TRANSITION_W3KE.md` §5.20f)**: 큰 표본(팔당 860 env)에서 현 실물 표는 회복을 깎지 않는다. 넓힐 근거가 없으므로 **이 표들을 실물에 쓰자고 제안하지 않는다.**
  실물의 `config/joint_limits_12dof.json` 은 손대지 않았다.
