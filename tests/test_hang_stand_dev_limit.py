"""hang_stand 관절별 이탈 한계 단위 검증 — CAN 없이. 0928 (승윤님 승인: 무릎만 따로, 지속조건 없음)."""
import importlib.util
import sys
import types

REPO = "/home/ryu/humanoid_leg_test1"


def _load():
    sys.path.insert(0, REPO + "/jetson/scripts")
    sys.path.insert(0, REPO + "/protocols"); sys.path.insert(0, REPO)
    for nm in ("can",):
        if nm not in sys.modules:
            m = types.ModuleType(nm); m.__path__ = []
            m.Bus = type("Bus", (), {"__init__": lambda s, *a, **k: None})
            m.Message = type("Message", (), {"__init__": lambda s, *a, **k: None})
            sys.modules[nm] = m
    spec = importlib.util.spec_from_file_location("hs", REPO + "/jetson/scripts/hang_stand.py")
    m = importlib.util.module_from_spec(spec)
    _a, sys.argv = sys.argv, ["hang_stand.py"]
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    finally:
        sys.argv = _a
    return m


M = _load()
KNEES = (4, 10)
HIPS_F = (1, 7)


def test_knee_limit_is_25_others_15():
    for mid in range(1, 13):
        lim = M.dev_limit_deg(mid)
        want = 25.0 if mid in KNEES + HIPS_F else 15.0
        assert lim == want, (mid, lim, want)


def test_knee_covers_measured_sag_with_margin():
    """실측 정적 처짐 12.6° + 좌우 차 1.5° 를 넘고, 그 2배(25.2°) 부근이어야 한다."""
    measured, spread = 12.6, 1.5
    for mid in KNEES:
        lim = M.dev_limit_deg(mid)
        assert lim > measured + spread, (mid, lim)        # 정상 처짐에서 안 터진다
        assert lim <= 2.0 * measured + 1.0, (mid, lim)    # 무한정 열지는 않는다


def test_non_knee_unchanged():
    """힙 좌우·회전·발목은 15° 그대로 — 완화가 다른 관절로 새지 않는다(힙 앞뒤는 0929 승윤님 지시로 25°)."""
    for mid in (2, 3, 5, 6, 8, 9, 11, 12):
        assert M.dev_limit_deg(mid) == M.DEV_ABORT_DEG == 15.0, mid


def test_unknown_motor_falls_back():
    assert M.dev_limit_deg(99) == M.DEV_ABORT_DEG


def test_limit_inside_soft_band():
    """스탠바이 무릎 ∓38.5° 에서 한계만큼 벗어나도 소프트 밴드 안이어야 한다."""
    import json
    lim_cfg = json.load(open(REPO + "/config/joint_limits_12dof.json"))["limits_deg"]
    for mid, joint, standby in ((4, "left_knee_joint", -38.5), (10, "right_knee_joint", +38.5)):
        d = M.dev_limit_deg(mid)
        lo, hi = lim_cfg[joint]["soft_min"], lim_cfg[joint]["soft_max"]
        assert lo <= standby - d and standby + d <= hi, (joint, standby, d, lo, hi)


def test_guard_still_trips_on_runaway():
    """폭주는 여전히 잡는다 — 한계의 두 배 이탈은 어느 관절이든 초과 판정."""
    for mid in range(1, 13):
        lim = M.dev_limit_deg(mid)
        assert abs(2.0 * lim) > lim


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print("PASS", fn.__name__)
    print("\n%d/%d 통과" % (len(fns), len(fns)))
