#!/bin/bash
# G16 후보 체크포인트 → <run>/exported/{policy.pt,policy.onnx} (play.py 의 내보내기 경로 그대로).
# play.py 는 내보낸 뒤 무한 재생하므로, 새 파일이 생기면 **내가 띄운 프로세스 그룹만** 명시 PGID 로 내린다(pkill 금지).
# 사용: reports/gen_g16/export_ckpt.sh <run 디렉토리 이름> <model_N.pt> <task 예: Biped12-HangStand-G16o-v0>
set -euo pipefail
cd /home/ryu/humanoid_leg_test1
RUN=$1; CK=$2; TASK=$3
D=logs/rsl_rl/biped12_stage4/$RUN
[ -f "$D/$CK" ] || { echo "체크포인트 없음 $D/$CK"; exit 1; }
t0=$(date +%s)
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
LOG=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/export_${RUN}_${CK%.pt}.log
setsid /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/play.py --headless --task "$TASK" --num_envs 1 \
  --checkpoint "$PWD/$D/$CK" > "$LOG" 2>&1 &
PG=$!
for i in $(seq 1 120); do
  sleep 5
  if [ -f "$D/exported/policy.onnx" ] && [ $(stat -c %Y "$D/exported/policy.onnx") -ge $t0 ] \
     && [ -f "$D/exported/policy.pt" ] && [ $(stat -c %Y "$D/exported/policy.pt") -ge $t0 ]; then
    sleep 5; break
  fi
  kill -0 $PG 2>/dev/null || { echo "play.py 가 먼저 끝남 — $LOG 확인"; exit 1; }
done
kill -TERM -- -$PG 2>/dev/null || true
# Kit 의 python3 는 TERM 을 무시하고 남는다(그룹 리더 bash 만 죽음 — 1001 확인) → 그룹 전체에 KILL (리더가 없어도 그룹은 남아 있다)
sleep 5; kill -KILL -- -$PG 2>/dev/null || true
sleep 1; ps -eo pgid= | grep -qx " *$PG" && { echo "⚠ 프로세스 그룹 $PG 잔존 — 확인 필요"; ps -eo pid,pgid,args | awk -v g=$PG '$2==g'; }
[ $(stat -c %Y "$D/exported/policy.pt") -ge $t0 ] || { echo "내보내기 실패(시간 초과)"; exit 1; }
echo "$CK" > "$D/exported/SOURCE_CHECKPOINT"
# 대조: 체크포인트 actor vs 내보낸 jit — 무작위 관측 256개에서 출력 최대 차이
/home/ryu/IsaacLab/_isaac_sim/python.sh - "$D" "$CK" <<'EOF'
import sys, torch
d, ck = sys.argv[1], sys.argv[2]
sd = torch.load(f'{d}/{ck}', map_location='cpu', weights_only=False)['model_state_dict']
jit = torch.jit.load(f'{d}/exported/policy.pt', map_location='cpu')
ks = sorted({k.split('.')[1] for k in sd if k.startswith('actor.') and k.endswith('.weight')}, key=int)
din = sd[f'actor.{ks[0]}.weight'].shape[1]
x = torch.randn(256, din)
h = x
for i, k in enumerate(ks):
    h = h @ sd[f'actor.{k}.weight'].T + sd[f'actor.{k}.bias']
    if i < len(ks) - 1:
        h = torch.nn.functional.elu(h)
with torch.no_grad():
    y = jit(x)
print(f'내보내기 대조: 관측 {din} → 출력 {tuple(y.shape)} · 최대 차이 {float((y - h).abs().max()):.2e}')
assert float((y - h).abs().max()) < 1e-4
EOF
echo "완료: $D/exported ($CK)"
