#!/usr/bin/env python3
"""w3_parity_record.py 사본 w3_parity_record_nb.py 를 만든다(1010 새 몸 걷기 parity 기록). 원본 불변, 앵커 불일치면 assert.
더하는 것: (1) RL_WALKING_EXTRA_TASKS 모듈 읽기 (2) 새 몸 태스크의 관측 감싸개(nb_obs_fixed_zero)가 항상 더하는 잡음을 끔 — 원본이
enable_corruption False·joint_vel_ema 내장 잡음 0 으로 하는 것과 같은 뜻(감싸개 인자 noise None·안쪽 함수 noise 0)."""
import os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src = open(os.path.join(R, "rl_walking/scripts/w3_parity_record.py"), encoding="utf-8").read()


def rep(s, a, b):
    assert s.count(a) == 1, (s.count(a), a)
    return s.replace(a, b)


A = "import rl_walking  # noqa: F401,E402\n"
src = rep(src, A, A + """import importlib as _nb_il  # 1010 새 몸: 학습 전용 추가 태스크 등록
for _nb_m in [x for x in os.environ.get("RL_WALKING_EXTRA_TASKS", "").split(",") if x.strip()]:
    _nb_il.import_module(_nb_m.strip())
""")
A = 'pol.joint_vel_ema.params["noise"] = 0.0\n'
src = rep(src, A, """_nb_wrapped = [n for n in ("joint_pos", "joint_vel", "joint_vel_ema") if getattr(getattr(pol, n).func, "__name__", "") == "nb_obs_fixed_zero"]
if _nb_wrapped:      # 새 몸 태스크: 감싸개 잡음 끔 + 안쪽 joint_vel_ema 내장 잡음 0
    for _n in _nb_wrapped:
        getattr(pol, _n).params["noise"] = None
    pol.joint_vel_ema.params["params"] = dict(pol.joint_vel_ema.params["params"], noise=0.0)
    print(f"[parity_nb] 관측 감싸개 잡음 끔: {_nb_wrapped}", flush=True)
else:
    pol.joint_vel_ema.params["noise"] = 0.0
""")
dst = os.path.join(R, "rl_walking/scripts/w3_parity_record_nb.py")
open(dst, "w", encoding="utf-8").write(src); print("wrote", dst)
