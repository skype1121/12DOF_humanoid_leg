#!/bin/bash
# v5 기준선 평가 배터리 (2026-09-11) — reports/real_prep_0911/promotion_criteria.md 의 조건 A~H 재현.
#
# 사용:  bash rl_walking/scripts/run_v5_eval_battery.sh [출력 로그 디렉토리]
#   기본 로그 디렉토리 = log_picture/eval_logs_<stamp>/ (stdout). 판정 산출물은 항상 log_picture/hang2stand_<tag>.txt.
#
# 원칙 (CLAUDE.md 규칙 5 · 0911 실측):
#   - DR-on 조건은 앱 종료 시 네이티브로 죽어 stdout 이 유실된다 → 성공 여부는 **결과 파일**(핵심 지표 줄)로만 판정.
#   - python.sh -u(무버퍼)로 실행. isaaclab.sh 는 exit 0 을 삼킨다.
#   - 같은 설정은 결정론적으로 동일 결과(seed=agent_cfg 고정). 독립 seed 는 --seed.
#   - 후보 비교는 CK/T 만 바꿔 같은 배터리를 돌린다. 시뮬 생존율을 실물 성공률로 쓰지 않는다.
set -u
cd "$(dirname "$0")/../.."
OUT=${1:-log_picture/eval_logs_$(date +%Y%m%d_%H%M)}; mkdir -p "$OUT"
CK=${CK:-"--load_run 2026-09-09_19-15-42 --checkpoint model_139987.pt"}   # v5
T=${T:-Biped12-HangStand-S8ac4-v0}
N=${N:-256}
P=${P:-v5}          # 결과 파일 태그 접두어 — 후보 평가 시 P=v5d 등으로 바꿔 v5 결과를 덮어쓰지 않게 한다
PY=${PY:-$HOME/IsaacLab/_isaac_sim/python.sh}

run() {  # run <tag> [eval args...]   (tag 의 선행 'v5_' 는 접두어 $P 로 치환)
  local tag="${P}_${1#v5_}"; shift
  rm -f "log_picture/hang2stand_$tag.txt"
  PYTHONPATH="$PWD" "$PY" -u rl_walking/scripts/eval_hang2stand.py --headless --task "$T" $CK --num_envs "$N" "$@" --out_tag "$tag" > "$OUT/$tag.log" 2>&1
  local F="log_picture/hang2stand_$tag.txt"
  if [ -s "$F" ] && grep -q "핵심 지표" "$F"; then
    printf "%-22s %s | %s | %s\n" "$tag" "$(grep '핵심 지표' "$F" | grep -oE '[0-9]+/[0-9]+ \([0-9.]+%\)')" \
      "$(grep '낙상(조기' "$F" | grep -oE '뒤 [0-9]+ · 앞 [0-9]+ · 왼 [0-9]+ · 오른 [0-9]+')" "$(grep -oE '발 들기 [0-9.]+회/env' "$F")"
  else
    printf "%-22s 결과파일 없음 (%s)\n" "$tag" "$(grep -E 'Error|Traceback' "$OUT/$tag.log" | tail -1)"
  fi
}

echo "# v5 평가 배터리 — task $T, env $N, 결과: log_picture/hang2stand_<tag>.txt, stdout: $OUT/"
# A 명목 / B 현실 DR / 진단 대조군
run v5_nom_256      --nominal
run v5_dr_256
run v5_dr_nopush_nonoise --no_push --no_obs_noise
# C 접촉 obs 교란 (actor 관측만, 정답·물리 불변)
run v5_nom_cReal    --nominal --contact_perturb realistic
run v5_nom_cHold    --nominal --contact_perturb hold
run v5_nom_cChatter --nominal --contact_perturb chatter
run v5_nom_cOne     --nominal --contact_perturb onefoot
run v5_nom_cBoth0   --nominal --contact_perturb both0
run v5_nom_cBoth1   --nominal --contact_perturb both1
run v5_dr_cReal     --contact_perturb realistic
# D 측면 CoM (COMY_FIX 반영판: y 만 고정, x·z 는 env 범위)
run v5_dr_comy_n02  --com_fix_y -0.02
run v5_dr_comy_p02  --com_fix_y  0.02
run v5_dr_comy_n04  --com_fix_y -0.04
run v5_dr_comy_p04  --com_fix_y  0.04
run v5_nom_comy_n02 --nominal --com_fix_y -0.02
run v5_nom_comy_p02 --nominal --com_fix_y  0.02
# E 독립 seed
for s in 1 2; do
  run v5_nom_s$s        --nominal --seed $s
  run v5_dr_s$s         --seed $s
  run v5_nom_cReal_s$s  --nominal --contact_perturb realistic --seed $s --perturb_seed $s
done
# F 30 s 유지 / G 느린 방출 / H 지연 민감도(실측 중앙 6.3틱·p95 8.9)
run v5_nom_30s      --nominal --seconds 45 --stand_s 30
run v5_dr_30s       --seconds 45 --stand_s 30
run v5_nom_slowrel  --nominal --release_s 20,40 --seconds 60
run v5_dr_slowrel   --release_s 20,40 --seconds 60
run v5_nom_delay7   --nominal --delay_steps 7,7
run v5_nom_delay9   --nominal --delay_steps 9,9
run v5_nom_delay12  --nominal --delay_steps 12,12
run v5_dr_delay9    --delay_steps 9,9
run v5_nom_delay69  --nominal --delay_steps 6,9
run v5_dr_delay69   --delay_steps 6,9
echo "# 완료. 표 갱신: reports/real_prep_0911/sim_eval_v5.md"
