"""랜덤화 범위 전수 감사 — 실측 추정치가 학습 분포 안에 있는가.

**왜 필요한가 (0810).** CoM 에서 이 결함이 실제로 났다. 실측 역산 −31.6mm
인데 v7~v9 의 학습 범위가 −30~+20mm 라 **참값이 분포 밖**이었다. 정책은
있지도 않은 몸을 균형잡는 법을 배웠고, 정책을 아무리 갈아도 그 갭은 남았다.

같은 결함이 다른 파라미터에도 있는지 한 번에 본다. 파라미터별로
 ① 세대별 학습 범위  ② 실측·추정치와 그 출처  ③ 포함 여부
를 표로 낸다. 포함 안 되는 항목이 나오면 그것이 다음 세대의 교정 대상이다.

실행 (시뮬은 안 돌리지만 pxr 때문에 앱 기동은 필요):
  PYTHONPATH=/home/ryu/humanoid_leg_test1 \
      ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/audit_dr_coverage.py
"""
import os
import sys

from isaaclab.app import AppLauncher

_app = AppLauncher(headless=True).app

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.join(REPO, "log_picture", "랜덤화범위감사.txt")

#: 실측·추정치 대장 — (하한, 상한, 출처). 범위인 이유는 대부분 구간 추정이라서다.
#: 여기에 없는 항목은 "잰 적이 없다"는 뜻이고, 그 자체가 결함 후보다.
MEASURED = {
    "base_com.x": (-0.059, -0.018,
                   "0810 낙하 역산. 물성 가정은 ±1.7mm 밖에 안 흔들지만 "
                   "놓임각 ±2°가 25mm를 흔든다 (eval_com_fit 8조건)"),
    "add_base_mass": (1.5, 3.5,
                      "URDF base 10.491kg. payload판 12.001kg(=+1.51). "
                      "승윤님 추정 실물 ~13kg(=+2.5). 저울 실측 대기"),
    "action_delay_steps": (4, 5,
                           "0803 블랙박스 상호상관 실측 (stand_v6 5틱, v3 4틱)"),
    "imu_bias_rad": (-0.087, 0.087,
                     "미측정. imu_mount_calib 이 '수직 매달림'을 영점으로 쓰는데 "
                     "줄에 매단 몸은 연직이 아니다 → 편향이 1:1로 박힌다. "
                     "0810 A: −4.5°가 실물 전방추세·떨림을 동시 재현"),
    "joint_zero_bias_rad": (-0.052, 0.052,
                            "미측정. 로터절대 엔코더가 전원사이클마다 36°/10° "
                            "배수로 시프트(m8 1건 실측). zero_kinematic_check 미실행"),
    "stiffness_scale": (1.0, 1.0,
                        "kp150 명목. 구조 컴플라이언스로 실효는 더 낮을 수 있으나 "
                        "0810 SEA 검증에서 관측측 기여는 부차적으로 판명"),
    "damping_scale": (1.0, 1.0, "kd5 = MIT 프로토콜 상한 (실물 고정값)"),
}

#: 감사 대상 세대 (등록 태스크 id)
TASKS = [
    ("stand_v5", "Biped12-Velocity-Stage4V2-Stand-V5-v0"),
    ("stand_v6", "Biped12-Velocity-Stage4V2-Stand-V6-v0"),
    ("stand_v7", "Biped12-Velocity-Stage4V2-Stand-V7-v0"),
    ("stand_v8", "Biped12-Velocity-Stage4V2-Stand-V8-v0"),
    ("stand_v9", "Biped12-Velocity-Stage4V2-Stand-V9-v0"),
    ("stand_v10", "Biped12-Velocity-Stage4V2-Stand-V10-v0"),
    ("stand_v13", "Biped12-Velocity-Stage4V2-Stand-V13-v0"),
]


def extract(cfg):
    """env cfg 에서 랜덤화 범위를 뽑는다. 없으면 None (= 랜덤화 안 함)."""
    ev = cfg.events
    out = {}

    def ev_param(name, key):
        t = getattr(ev, name, None)
        return None if t is None else t.params.get(key)

    com = ev_param("base_com", "com_range")
    out["base_com.x"] = tuple(com["x"]) if com else None
    out["add_base_mass"] = ev_param("add_base_mass", "mass_distribution_params")
    out["stiffness_scale"] = ev_param("actuator_gains",
                                      "stiffness_distribution_params")
    out["damping_scale"] = ev_param("actuator_gains",
                                    "damping_distribution_params")
    out["joint_friction"] = ev_param("joint_friction",
                                     "friction_distribution_params")
    a = cfg.actions.joint_pos
    out["action_delay_steps"] = (getattr(a, "min_delay_steps", None),
                                 getattr(a, "max_delay_steps", None))
    cb = getattr(ev, "calib_bias", None)
    out["joint_zero_bias_rad"] = cb.params.get("joint_bias_rad") if cb else None
    out["imu_bias_rad"] = cb.params.get("imu_tilt_rad") if cb else None
    out["push_robot"] = "있음" if getattr(ev, "push_robot", None) else "없음"
    return out


def covers(rng, lo, hi):
    """학습 범위가 실측 구간을 덮는가 → 'O' 완전 / '△' 부분 / 'X' 전무."""
    if rng is None or rng[0] is None:
        return "X", "랜덤화 없음"
    a, b = min(rng), max(rng)
    if a <= lo and hi <= b:
        return "O", ""
    if b < lo or hi < a:
        return "X", "**분포 밖**"
    return "△", "일부만"


def main():
    import gymnasium  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    sys.path.insert(0, REPO)
    import rl_walking  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    cols, data = [], {}
    for name, task in TASKS:
        try:
            data[name] = extract(load_cfg_from_registry(task, "env_cfg_entry_point"))
            cols.append(name)
        except Exception as exc:                       # noqa: BLE001
            print(f"[warn] {name}: {exc}")

    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P("=" * 100)
    P("랜덤화 범위 전수 감사 — 실측 추정치가 학습 분포 안에 있는가")
    P("O=완전히 덮음  △=일부만  X=분포 밖 또는 랜덤화 없음")
    P("=" * 100)

    keys = ["base_com.x", "add_base_mass", "action_delay_steps",
            "joint_zero_bias_rad", "imu_bias_rad", "stiffness_scale",
            "damping_scale", "joint_friction", "push_robot"]
    for k in keys:
        P("")
        m = MEASURED.get(k)
        P(f"■ {k}")
        if m:
            P(f"   실측·추정: {m[0]} ~ {m[1]}")
            P(f"   출처: {m[2]}")
        else:
            P("   실측·추정: **없음 — 잰 적이 없다**")
        for c in cols:
            rng = data[c].get(k)
            if isinstance(rng, str):
                P(f"     {c:<12} {rng}")
                continue
            mark = ""
            if m and rng is not None:
                sym, note = covers(rng, m[0], m[1])
                mark = f"   [{sym}] {note}"
            txt = "랜덤화 없음" if rng is None or rng[0] is None else str(tuple(rng))
            P(f"     {c:<12} {txt:<26}{mark}")

    P("")
    P("=" * 100)
    P("■ 결론 후보 — [X] 가 붙은 칸이 다음 세대의 교정 대상이다.")
    P("  실측·추정이 '없음'인 항목은 더 나쁘다: 덮었는지 아닌지조차 모른다.")
    P("=" * 100)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n리포트: {OUT}")


main()
_app.close()
