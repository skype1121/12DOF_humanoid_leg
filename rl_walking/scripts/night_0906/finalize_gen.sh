#!/bin/bash
# 세대 마무리: finalize_gen.sh <tag> <name> <task> <warm_desc>
#   chain_<X>.out 의 "체인 종료" 이후에 실행. <tag>_pick.txt(run ckpt) → 익스포트(eval_policy) → gen_report →
#   register_model → 브리지 CHECKPOINTS/CKPT_GAIT_FREQ 등록(멱등) → selftest → 배포 패키지 재조립.
#   젯슨 복사는 하지 않는다(승윤님 결정).
set -u
TAG=$1; NAME=$2; TASK=$3; WARM=${4:-}
S=/tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad
cd /home/ryu/humanoid_leg_test1
export PYTHONPATH=/home/ryu/humanoid_leg_test1 OMNI_KIT_ACCEPT_EULA=YES
read RUN CKPT < $S/${TAG}_pick.txt
echo "[$(date +%T)] finalize $TAG → $NAME (run $RUN ckpt $CKPT)"
EXP=logs/rsl_rl/biped12_stage4/$RUN/exported
if [ ! -f "$EXP/policy.pt" ]; then
  /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_policy.py --headless --task $TASK --load_run $RUN --checkpoint $CKPT \
      --steps 100 --num_envs 4 --cmd_x 0.0 --out_tag ${TAG}_export > $S/export_$TAG.log 2>&1
fi
if [ ! -f "$EXP/policy.pt" ]; then
  echo "익스포트 실패 — $EXP/policy.pt 없음"; grep -m3 -E "Traceback|Error" $S/export_$TAG.log; exit 1
fi
echo "  policy.pt md5 $(md5sum $EXP/policy.pt | cut -c1-8)…"
python3 rl_walking/scripts/gen_report.py $TAG --title "$NAME" --run $RUN --ckpt $CKPT --task $TASK --usd rl_walking/assets/usd_final3/biped12.usd | tail -1
python3 rl_walking/scripts/register_model.py $NAME --run $RUN --ckpt $CKPT --task $TASK --usd rl_walking/assets/usd_final3/biped12.usd \
    --warm "$WARM" --summary "chain 판정 reports/gen_$TAG/README.md" | tail -2
python3 $S/register_bridge.py $NAME $RUN
timeout 900 python3 jetson/scripts/rl_bridge_node.py --selftest 2>&1 | tail -1
GENS="stand_v15 stand_v16 stand_v17 stand_v18 stand_v18b" bash jetson/deploy_package.sh 2>&1 | tail -2
echo "[$(date +%T)] finalize $TAG 완료"
