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


def test_stop_sequence_continuity_and_neutral():
    """num_steps 지정 시: 정지 전환 연속(<3°/10ms), 종료 후 직립 유지."""
    with open(os.path.join(REPO, "config", "sim_walk_params.json")) as f:
        cfg = json.load(f)
    g = StaticGait(StaticGaitParams(**cfg["gait"]), num_steps=5)
    ts = g._t_stop()
    assert ts is not None
    prev = None
    for i in range(0, int((ts + g.stop_dur + 2.0) * 100)):
        tg = g.targets(i * 0.01)
        if prev:
            for k, v in tg.items():
                assert abs(v - prev[k]) * R2D < 3.0, f"{k} jump at t={i*0.01:.2f}"
        prev = tg
    # 정지 시작점 = LEAN 종료 → 양 힙이 같은 해부학 각(-S) = 양발 나란
    # (raw 부호는 좌우 반대이므로 합이 0이면 나란)
    base = g._targets_raw(ts - 1e-6)
    assert abs(base["left_hip_f_joint"] + base["right_hip_f_joint"]) < 1e-6
    # 종료 후: 전 관절 중립 (발 모아 직립 — 걷기 재시작 가능 상태)
    final = g.targets(ts + g.stop_dur + 1.0)
    assert abs(final["left_hip_f_joint"]) < 1e-6
    assert abs(final["right_hip_f_joint"]) < 1e-6
    assert abs(final["left_hip_a_joint"]) < 1e-6
    assert abs(final["right_ankle_r_joint"]) < 1e-6
    assert abs(g.lat_ref(ts + g.stop_dur + 1.0)) < 1e-6


def test_request_stop_mid_walk():
    """걷는 중 정지 요청: 진행 중 스윙 완료 후 LEAN 종료에 정지, 연속성 유지."""
    with open(os.path.join(REPO, "config", "sim_walk_params.json")) as f:
        cfg = json.load(f)
    g = StaticGait(StaticGaitParams(**cfg["gait"]))
    # 오른스윙 한가운데(t = lean + step/2)에서 정지 요청
    t_req = g.p.lean_dur + g.p.step_dur * 0.5
    ts = g.request_stop(t_req)
    assert ts > t_req
    # 스윙은 완료돼야 함: ts >= 스윙 종료 시각
    assert ts >= g.p.lean_dur + g.p.step_dur
    prev = None
    for i in range(0, int((ts + g.stop_dur + 1.0) * 100)):
        tg = g.targets(i * 0.01)
        if prev:
            for k, v in tg.items():
                assert abs(v - prev[k]) * R2D < 3.0, f"{k} jump at {i*0.01:.2f}"
        prev = tg
    final = g.targets(ts + g.stop_dur + 0.5)
    assert abs(final["left_hip_f_joint"]) < 1e-6


def test_step_counter_synthetic():
    """카운터: 스윙 창 안의 리프트만 집계, 창 밖 롤링 범프·미복귀 상승 제외."""
    from sim_walking.runner import count_steps

    def mkrow(t, zl, zr):
        return {"t": t, "foot_z": {"left": zl, "right": zr}}

    rows = []
    for i in range(200):           # 20초, 0.1s 간격
        t = round(i * 0.1, 1)
        zl = zr = 0.050
        if 2.0 <= t <= 3.0:        # 오른발 스윙 창 내 리프트 (스텝)
            zr = 0.050 + 0.020
        if 6.0 <= t <= 7.0:        # 왼발 스윙 창 내 리프트 (스텝)
            zl = 0.050 + 0.020
        if 10.0 <= t <= 10.6:      # 창 밖 왼발 롤링 범프 (제외돼야 함)
            zl = 0.050 + 0.014
        if t >= 15.0:              # 미복귀 상승 (정지 자세 굴림 — 제외)
            zr = 0.050 + 0.014
        rows.append(mkrow(t, zl, zr))

    def swing_fn(t):
        if 1.9 <= t <= 3.1:
            return "right"
        if 5.9 <= t <= 7.1:
            return "left"
        return None

    s = count_steps(rows, swing_fn=swing_fn)
    assert s == {"left": 1, "right": 1}, s
    # swing_fn 없이도 미복귀 상승은 제외
    s2 = count_steps(rows)
    assert s2["right"] == 1, s2   # 15s+ 상승은 미복귀라 미집계


def test_real_bridge_tape_valid():
    """sim→real tape: 데모 구성에서 위반 0, 페이로드가 Stage8 스키마 준수."""
    from sim_walking.real_bridge import BridgeConfig, build_tape
    with open(os.path.join(REPO, "config", "sim_walk_params.json")) as f:
        cfg = json.load(f)
    g = StaticGait(StaticGaitParams(**cfg["gait"]), num_steps=4)
    tape, stats = build_tape(g, g._t_stop() + g.stop_dur + 1.0,
                             BridgeConfig(stream_hz=12.5))
    assert stats["violations"] == []
    assert stats["commands"] == len(tape) > 100
    for rec in tape[:50] + tape[-50:]:
        p = rec["payload"]
        assert p["command"] == "SET_JOINT_TARGET"
        assert p["joint"] in JOINT_NAMES
        assert isinstance(p["target_deg"], float)
        assert "joints" not in p and "joints_deg" not in p  # 멀티관절 금지 준수
    # 시간 단조 증가
    ts = [r["t"] for r in tape]
    assert ts == sorted(ts)


def test_real_bridge_applies_hw_sign(monkeypatch):
    """하드웨어맵 sign/direction 캘리브레이션이 tape에 자동 반영된다."""
    import sim_walking.real_bridge as RB
    hw = RB._load_hw_map()
    flipped = {k: dict(v) for k, v in hw.items()}
    flipped["left_knee_joint"]["sign"] = -1
    monkeypatch.setattr(RB, "_load_hw_map", lambda: flipped)
    with open(os.path.join(REPO, "config", "sim_walk_params.json")) as f:
        cfg = json.load(f)
    g = StaticGait(StaticGaitParams(**cfg["gait"]), num_steps=2)
    # sign 반전 시 left_knee 리밋 검증은 반전각 기준으로 걸리므로 sim limit 끄고 확인
    tape, stats = RB.build_tape(g, 5.0, RB.BridgeConfig(use_sim_limits=False))
    knee = [r["payload"]["target_deg"] for r in tape
            if r["payload"]["joint"] == "left_knee_joint"]
    # 시뮬 left_knee 스탠스 -6°(무릎굽힘 raw -) → sign -1 이면 +6° 로 반전
    assert any(v > 4.0 for v in knee), knee[:5]


def test_real_bridge_delta_guard():
    """스트림 주기 대비 과속 명령은 위반으로 잡힌다 (아주 낮은 hz에서)."""
    from sim_walking.real_bridge import BridgeConfig, build_tape
    with open(os.path.join(REPO, "config", "sim_walk_params.json")) as f:
        cfg = json.load(f)
    gp = dict(cfg["gait"]); gp["knee_lift_deg"] = 60.0   # 과격한 스윙
    g = StaticGait(StaticGaitParams(**gp))
    _, stats = build_tape(g, 8.0, BridgeConfig(stream_hz=2.0))
    assert len(stats["violations"]) > 0
