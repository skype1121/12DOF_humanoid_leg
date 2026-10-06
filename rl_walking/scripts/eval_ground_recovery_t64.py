"""1007 parity 감사 — eval_ground_recovery.py 를 **감지기 타이머만 float64**(배포 c5_switch 현행 IMU 경로와 같은 정밀도)로 바꿔 실행.

원본 파일은 건드리지 않는다: 소스를 읽어 아래 두 줄의 텐서 dtype 만 float64 로 바꾼 뒤 같은 __file__ 로 exec 한다(인자·출력 규약 그대로).
  sw_t_on·sw_calm(calm 1.0 s·dwell 0.4 s)  ·  early_hold(조기 복귀 hold 0.2 s)
시뮬 float32 누적은 0.02 × 50 = 0.99999994 < 1.0 이라 51 틱째 채워지고, 배포 float64 는 50 틱째 채워진다(STOPEST §9: 꺼짐 1 틱 이름).
같은 --seed 로 원본과 짝 실행하면 첫 타이머 경계까지 궤적이 같아 차이는 타이머에서만 온다(짝 비교).
사용: isaaclab.sh -p rl_walking/scripts/eval_ground_recovery_t64.py <eval_ground_recovery.py 와 같은 인자>
"""
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_ground_recovery.py")
src = open(SRC, encoding="utf-8").read()
REPL = [
    ("sw_t_on = torch.zeros(N, device=dev); sw_calm = torch.zeros(N, device=dev)",
     "sw_t_on = torch.zeros(N, device=dev, dtype=torch.float64); sw_calm = torch.zeros(N, device=dev, dtype=torch.float64)"),
    ("\nearly_hold = torch.zeros(N, device=dev)\n", "\nearly_hold = torch.zeros(N, device=dev, dtype=torch.float64)\n"),
]
for a, b in REPL:
    assert src.count(a) == 1, f"패치 대상이 1 곳이 아님: {a!r} ({src.count(a)})"
    src = src.replace(a, b)
print("[eval_ground_recovery_t64] 감지기 타이머 float64(sw_t_on·sw_calm·early_hold) — 배포 c5_switch 현행 IMU 경로 정밀도", flush=True)
g = {"__name__": "__main__", "__file__": SRC}
exec(compile(src, SRC, "exec"), g)
