"""sim_walking 오프라인 자동 테스트 (Isaac 불필요 — 순수 파이썬 궤적 검증).

실행: python3 -m pytest tests/test_sim_walking.py -v
"""
import json
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sim_walking.gait_static import StaticGait, StaticGaitParams  # noqa: E402
from sim_walking.gait_fsm import GaitFSM, GaitParams              # noqa: E402
from sim_walking.sim_session import ANAT_SIGN, anat               # noqa: E402

R2D = 180.0 / math.pi

JOINT_NAMES = [
    "left_hip_f_joint", "left_hip_a_joint", "left_hip_r_joint",
    "left_knee_joint", "left_ankle_f_joint", "left_ankle_r_joint",
    "right_hip_f_joint", "right_hip_a_joint", "right_hip_r_joint",
    "right_knee_joint", "right_ankle_f_joint", "right_ankle_r_joint",
]


def _limits():
    with open(os.path.join(REPO, "config", "sim_joint_limits.json")) as f:
        return json.load(f)["limits_deg"]


def _demo_gait():
    with open(os.path.join(REPO, "config", "sim_walk_params.json")) as f:
        cfg = json.load(f)
    return StaticGait(StaticGaitParams(**cfg["gait"]))


def test_all_12_joints_emitted():
    """궤적이 12관절 전부(하드웨어맵과 같은 이름)를 출력한다."""
    g = _demo_gait()
    tg = g.targets(3.3)
    assert sorted(tg.keys()) == sorted(JOINT_NAMES)


def test_static_gait_within_limits():
    """데모 파라미터 궤적이 시뮬 관절 리밋을 위반하지 않는다."""
    g = _demo_gait()
    lim = _limits()
    for i in range(0, 3000):
        tg = g.targets(i * 0.01)
        for k, v in tg.items():
            lo, hi = lim[k]["lower"], lim[k]["upper"]
            assert lo - 1e-6 <= v * R2D <= hi + 1e-6, \
                f"{k}={v*R2D:.2f} out of [{lo},{hi}] at t={i*0.01:.2f}"


def test_static_gait_continuity():
    """10ms당 관절 점프가 3° 미만(급격한 명령 없음 = 과속 방지)."""
    g = _demo_gait()
    prev = None
    for i in range(0, 3000):
        tg = g.targets(i * 0.01)
        if prev:
            for k, v in tg.items():
                assert abs(v - prev[k]) * R2D < 3.0, f"{k} jump at t={i*0.01}"
        prev = tg


def test_static_gait_starts_at_neutral():
    """t=0 궤적은 직립 트림 자세(힙/무릎 스텝 성분 0)여야 한다."""
    g = _demo_gait()
    tg = g.targets(0.0)
    assert abs(tg["left_hip_f_joint"]) < 1e-9
    assert abs(tg["right_hip_f_joint"]) < 1e-9


def test_anat_sign_mapping_consistency():
    """해부학 매핑: 좌우 hip/knee/ankle_f는 반대 부호, ankle_r만 동일 부호."""
    for action in ("hip_flexion", "hip_abduction", "hip_ext_rot",
                   "knee_flexion", "dorsiflexion"):
        assert ANAT_SIGN[action]["left"] == -ANAT_SIGN[action]["right"], action
    assert ANAT_SIGN["eversion"]["left"] == ANAT_SIGN["eversion"]["right"]


def test_anat_helper():
    j, v = anat("left", "knee_flexion", 0.5)
    assert j == "left_knee_joint" and v == -0.5
    j, v = anat("right", "knee_flexion", 0.5)
    assert j == "right_knee_joint" and v == +0.5


def test_periodic_gait_smoke():
    """(구) 주기 로킹 FSM도 리밋 내 유지 — 회귀 방지."""
    f = GaitFSM(GaitParams())
    lim = _limits()
    for i in range(0, 1000):
        tg = f.targets(i * 0.01)
        for k, v in tg.items():
            lo, hi = lim[k]["lower"], lim[k]["upper"]
            assert lo - 1e-6 <= v * R2D <= hi + 1e-6


def test_lat_ref_bounded():
    """의도 롤 기준값은 파라미터 진폭을 넘지 않는다."""
    g = _demo_gait()
    amp = g.p.lean_roll_deg + 1e-9
    for i in range(0, 2000):
        assert abs(g.lat_ref(i * 0.01)) <= amp
