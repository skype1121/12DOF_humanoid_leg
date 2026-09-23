"""pitch_comp (앞뒤 균형 복원항) 단위 검증 — ROS·CAN 없이. 0923.

핵심은 부호다. 틀리면 복원이 아니라 낙상 가속이 된다.
기준 사실(0922 3회차 실물, 브리지 rp_deg 로 확인):
  · pitch + = 앞기울임
  · 앞으로 기울 때 정책 발목 명령은 평발 기하선(−1.00 °/°)을 따라 왼발목 ↓ / 오른발목 ↑(0 쪽)
  · 버티려면 그 선보다 왼발목은 더 +, 오른발목은 더 − 로 가야 한다(시뮬 잔차 +0.35~0.64)
"""
import importlib.util
import math
import sys
import types

REPO = "/home/ryu/humanoid_leg_test1"


def _load():
    """rl_bridge_node 를 ROS 의존 없이 로드 — 모듈 상단 임포트만 스텁한다."""
    for name in ("rclpy", "rclpy.node", "rclpy.qos", "std_msgs", "std_msgs.msg",
                 "sensor_msgs", "sensor_msgs.msg", "geometry_msgs", "geometry_msgs.msg"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
    sys.modules["rclpy.node"].Node = type("Node", (), {"__init__": lambda self, *a, **k: None})
    for mod, cls in (("std_msgs.msg", "String"), ("sensor_msgs.msg", "Imu"),
                     ("sensor_msgs.msg", "JointState"), ("geometry_msgs.msg", "Vector3")):
        setattr(sys.modules[mod], cls, type(cls, (), {"__init__": lambda self, *a, **k: None}))
    sys.modules["rclpy.qos"].QoSProfile = type("QoSProfile", (), {"__init__": lambda self, *a, **k: None})
    sys.modules["rclpy.qos"].ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=0, RELIABLE=1)
    sys.modules["rclpy.qos"].HistoryPolicy = types.SimpleNamespace(KEEP_LAST=0)
    sys.modules["rclpy.qos"].DurabilityPolicy = types.SimpleNamespace(VOLATILE=0)
    spec = importlib.util.spec_from_file_location("rlbn", REPO + "/jetson/scripts/rl_bridge_node.py")
    m = importlib.util.module_from_spec(spec)
    sys.argv = ["rl_bridge_node.py", "--selftest"]
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


M = _load()


def _grav_for_pitch(deg):
    """pitch = atan2(g_x, −g_z) 를 만족하는 단위 투영중력 벡터."""
    th = math.radians(deg)
    return (math.sin(th), 0.0, -math.cos(th))


def _cfg(K=0.35, tau=0.06, limit=15.0, slew=2.0):
    return M._validate_roll_comp({
        "enabled": True, "axis": "pitch", "joints": "ankle_f", "K": K, "tau_s": tau,
        "limit_deg": limit, "sign": 1, "lr": [1, -1], "gate": "any",
        "debounce_s": 0.3, "slew_deg_per_tick": slew, "roll_clamp_deg": 30.0}, name="pitch_comp")


def test_validator_axis_and_envelope():
    c = _cfg()
    assert c["axis"] == "pitch" and c["joints"] == "ankle_f"
    assert abs(c["K"] - 0.35) < 1e-9
    # roll 축은 종전 봉투 유지 — τ 0.06 은 roll 에서 거부되어야 한다
    try:
        M._validate_roll_comp({"enabled": True, "axis": "roll", "tau_s": 0.06})
        raise AssertionError("roll 축 τ 봉투가 느슨해졌다")
    except ValueError:
        pass
    # ankle_f 한계는 15° 까지, 그 위는 거부
    try:
        _cfg(limit=20.0)
        raise AssertionError("ankle_f limit 봉투 초과가 통과됐다")
    except ValueError:
        pass
    # 오타 키는 조용히 무시되지 않는다
    try:
        M._validate_roll_comp({"enabled": True, "axis": "pitch", "Kp": 1.0}, name="pitch_comp")
        raise AssertionError("오타 키가 통과됐다")
    except ValueError:
        pass


def test_pitch_sign_forward_lean():
    """앞기울임(+) → 왼발목 +, 오른발목 −. 이 부호가 '버티는' 방향이다."""
    rc = M.RollComp(_cfg())
    off = 0.0
    for _ in range(400):                       # 정착까지
        off, gate = rc.update(_grav_for_pitch(10.0), (1.0, 1.0), True, True)
    assert gate
    assert off > 0, "앞기울임에서 오프셋이 +가 아니다"
    assert abs(math.degrees(off) - 0.35 * 10.0) < 0.15, math.degrees(off)
    tgt = [0.0] * 12
    out = rc.add_to(tgt, off)
    li = M.JOINT_ORDER.index("left_ankle_f_joint")
    ri = M.JOINT_ORDER.index("right_ankle_f_joint")
    assert out[li] > 0 and out[ri] < 0, (out[li], out[ri])
    assert abs(out[li] + out[ri]) < 1e-12, "좌우가 거울이 아니다"
    assert sum(abs(v) for i, v in enumerate(out) if i not in (li, ri)) == 0, "다른 관절이 건드려졌다"


def test_pitch_sign_backward_lean():
    """뒤기울임(−) → 반대 부호. 같은 이득 하나가 앞뒤를 다 덮는다."""
    rc = M.RollComp(_cfg())
    off = 0.0
    for _ in range(400):
        off, _ = rc.update(_grav_for_pitch(-10.0), (1.0, 1.0), True, True)
    assert off < 0 and abs(math.degrees(off) + 0.35 * 10.0) < 0.15, math.degrees(off)


def test_limit_and_slew_hold():
    """큰 기울기에서도 limit 을 넘지 않고, 틱당 변화는 slew 이내."""
    rc = M.RollComp(_cfg(K=1.0, limit=5.0, slew=0.25))
    prev, off = 0.0, 0.0
    for _ in range(600):
        off, _ = rc.update(_grav_for_pitch(25.0), (1.0, 1.0), True, True)
        assert abs(off - prev) <= math.radians(0.25) + 1e-12, "슬루 초과"
        prev = off
    assert abs(math.degrees(off)) <= 5.0 + 1e-9, math.degrees(off)


def test_gate_off_decays_without_step():
    """접촉이 끊겨도 오프셋은 계단으로 0 이 되지 않는다(슬루로 감쇠)."""
    rc = M.RollComp(_cfg())
    off = 0.0
    for _ in range(400):
        off, _ = rc.update(_grav_for_pitch(10.0), (1.0, 1.0), True, True)
    prev = off
    for _ in range(200):                       # 양발 무접촉 (gate=any 라 raw_gate 해제)
        off, _ = rc.update(_grav_for_pitch(10.0), (0.0, 0.0), True, True)
        assert abs(off - prev) <= math.radians(2.0) + 1e-12
        prev = off
    assert abs(off) < abs(math.radians(0.35 * 10.0)), "게이트 해제 뒤에도 안 줄었다"


def test_inactive_is_hard_reset():
    rc = M.RollComp(_cfg())
    for _ in range(400):
        rc.update(_grav_for_pitch(10.0), (1.0, 1.0), True, True)
    off, gate = rc.update(_grav_for_pitch(10.0), (1.0, 1.0), False, True)
    assert off == 0.0 and not gate


def test_disabled_is_noop():
    rc = M.RollComp(None)
    assert not rc.enabled
    assert rc.update((0.1, 0.0, -1.0), (1.0, 1.0), True, True) == (0.0, False)
    t = [1.0] * 12
    assert rc.add_to(t, 0.0) is t


def test_roll_comp_unchanged():
    """종전 roll 설정이 그대로 통과하고 roll 축으로 동작한다(회귀 방지)."""
    c = M._validate_roll_comp({"enabled": True, "joints": "ankle_r", "K": 1.0, "tau_s": 1.0,
                               "limit_deg": 5.0, "sign": 1, "lr": [1, 1], "gate": "both",
                               "debounce_s": 0.2, "slew_deg_per_tick": 0.25, "roll_clamp_deg": 15.0})
    assert c["axis"] == "roll"
    rc = M.RollComp(c)
    off = 0.0
    for _ in range(3000):
        off, _ = rc.update((0.0, math.sin(math.radians(5.0)), -math.cos(math.radians(5.0))),
                           (1.0, 1.0), True, True)
    assert abs(math.degrees(off) - 5.0) < 0.2, math.degrees(off)   # roll 은 g_y 를 본다
    # pitch 성분은 roll 축에 섞이지 않는다
    rc2 = M.RollComp(c)
    off2 = 0.0
    for _ in range(3000):
        off2, _ = rc2.update(_grav_for_pitch(10.0), (1.0, 1.0), True, True)
    assert abs(math.degrees(off2)) < 0.05, math.degrees(off2)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print("\n%d/%d 통과" % (len(fns), len(fns)))
