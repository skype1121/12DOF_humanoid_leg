#!/bin/bash
# 1003 러너↔학습 env 동치 검사 실행기. 인자: tag task model act_beta jvel_beta lpf jvel_ema_mode sim_restart2
# Kit 종료가 멈추므로 결과 파일이 생기면 그 kit python PID 만 종료(명시 PID).
cd /home/ryu/humanoid_leg_test1; export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
POSE='[0.5282, -0.5282, 0.0, 0.0, 0.0, 0.0, -0.672, 0.672, 0.4887, -0.4887, 0.0, 0.0]'
OUT=reports/gen_g18/runner_equiv_$1.txt; rm -f $OUT
/home/ryu/IsaacLab/_isaac_sim/python.sh -u reports/gen_g18/runner_equiv.py --task $2 --model $3 --act_beta $4 --jvel_beta $5 --lpf $6 \
  --jvel_ema_mode $7 --sim_restart2 $8 --pose "$POSE" --steps ${STEPS:-600} --out $OUT > /tmp/claude-1000/-home-ryu-humanoid-leg-test1/e53dc80b-9c62-45a7-a400-3cd12071b2f1/scratchpad/eq_$1.l 2>&1 &
for i in $(seq 1 120); do sleep 5; [ -s $OUT ] && break; done
PID=$(ps -eo pid,cmd | grep "runner_equiv.py" | grep "$OUT" | grep kit/python | grep -v grep | awk '{print $1}')
[ -n "$PID" ] && kill $PID
cat $OUT 2>/dev/null || echo "결과 없음: $1"
