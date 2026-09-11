#!/bin/bash
# 실물 시도 로그 자동 수집 + 사후 분석 (PC 에서 실행, 젯슨은 읽기만 — scp). 2026-09-11.
# 사용: bash jetson/scripts/collect_trial_logs.sh <YYYYMMDD_HHMM 블랙박스 stamp> [세션 디렉토리명] [젯슨 host]
#   예: bash jetson/scripts/collect_trial_logs.sh 20260912_1030 session_20260912_1028 172.16.35.89
# 산출: reports/real_<날짜>/<stamp>/ 에 rl_bridge_<stamp>.jsonl, session/timeline.jsonl, pads_*.jsonl, stiff_*.jsonl 사본 +
#       analysis.md (analyze_hang2stand_trial --timeline, replay_bridge_log, analyze_fall, analyze_cop_transfer[진단만]).
set -uo pipefail
cd "$(dirname "$0")/../.."
STAMP=${1:?블랙박스 stamp (예 20260912_1030)}; SESS=${2:-}; HOST=${3:-172.16.35.89}; USER_=${JETSON_USER:-mama}
DAY=${STAMP:0:8}; OUT="reports/real_${DAY}/${STAMP}"; mkdir -p "$OUT"
S="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10"
echo "# 수집 → $OUT"
scp -q -o StrictHostKeyChecking=no "$USER_@$HOST:~/logs/rl_bridge_${STAMP}.jsonl" "$OUT/" && echo "블랙박스 OK" || echo "블랙박스 실패(경로/stamp 확인)"
if [ -z "$SESS" ]; then SESS=$($S "$USER_@$HOST" "ls -td ~/logs/sessions/session_* | head -1" 2>/dev/null | xargs -r basename); fi
[ -n "$SESS" ] && { mkdir -p "$OUT/$SESS"; scp -q -o StrictHostKeyChecking=no "$USER_@$HOST:~/logs/sessions/$SESS/timeline.jsonl" "$OUT/$SESS/" && echo "타임라인 OK ($SESS)" || echo "타임라인 실패"; }
for pat in "pads_inc_*.jsonl" "pads_cop_*.jsonl" "stiff_cop_*.jsonl"; do
  f=$($S "$USER_@$HOST" "ls -t ~/logs/$pat 2>/dev/null | head -1"); [ -n "$f" ] && scp -q -o StrictHostKeyChecking=no "$USER_@$HOST:$f" "$OUT/" && echo "$(basename "$f") OK"
done
BB="$OUT/rl_bridge_${STAMP}.jsonl"; TL=$(ls "$OUT"/session_*/timeline.jsonl 2>/dev/null | head -1)
{
  echo "# 시도 분석 $STAMP ($(date '+%F %T'))"; echo
  echo "## 1. 구간·순서·운영자 이벤트 (analyze_hang2stand_trial)"; echo '```'
  python3 jetson/scripts/analyze_hang2stand_trial.py "$BB" ${TL:+--timeline "$TL"} 2>&1; echo '```'
  echo "## 2. 로그 재생 계약 검증 (replay_bridge_log — 출력·계약, 폐루프 성능 아님)"; echo '```'
  PYTHONPATH=. python3 rl_walking/deploy/replay_bridge_log.py "$BB" --model models/stand_hang_v5 --gait_freq 1.1 2>&1 | tail -6; echo '```'
  echo "## 3. 낙하 등급 (analyze_fall — 성공 판정 아님)"
  echo "> ⚠ 이 등급은 지지 줄·보조 여부를 모른다. 0910 trial1(윈치에 매달린 채 260 s 안정)도 'A급 완전 성공'으로 나온다. 낙하 여부·자세 진단으로만 읽고, 자립 성공은 §끝의 규칙(모든 줄 느슨·무보조 확인 이후 관측 + 영상)으로만 판정한다."; echo '```'
  python3 jetson/scripts/analyze_fall.py "$BB" 2>&1 | tail -25; echo '```'
  P=$(ls "$OUT"/pads_*.jsonl 2>/dev/null | head -1)
  if [ -n "$P" ]; then echo "## 4. 패드 진단 (analyze_cop_transfer — 미보정, 원인·성공 판정 금지)"; echo '```'; python3 jetson/scripts/analyze_cop_transfer.py "$P" "$BB" --every 5 2>&1 | tail -30; echo '```'; fi
  echo; echo "판정 규칙: 자립 성공은 운영자 '모든 줄 느슨·무보조 확인' 이후 관측 시간 + 영상. 가설 판별표: reports/real_prep_0911/hypotheses_and_field_checks.md §B."
} > "$OUT/analysis.md"
echo "분석 → $OUT/analysis.md"
