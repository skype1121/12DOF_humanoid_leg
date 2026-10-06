"""w3_contingency.py 단위 시험(순수 numpy, ROS·GPU 불필요) — python3 rl_walking/deploy/selftest_w3_contingency.py

브리지 통합(BridgeCore 비트 동일·감독기 거부·정지 래치)은 jetson/scripts/rl_bridge_node.py --selftest 의 'W3 감시(1007)' 항목.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
from rl_walking.deploy import w3_contingency as C  # noqa: E402

DT = 0.02
G0 = (0.0, 0.0, -1.0)


def grav_rp(roll_deg, pitch_deg):
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    # 롤 = atan2(g_y, −g_z), 피치 = atan2(g_x, −g_z) 가 되도록
    g = np.array([math.tan(p), math.tan(r), -1.0])
    return tuple(g / np.linalg.norm(g))


def walk_contacts(k):
    """1.1 Hz 교대: 왼 스윙 u∈(0.05,0.45), 오른 스윙 u∈(0.55,0.95)."""
    u = (k * DT * 1.1) % 1.0
    return (0.0 if 0.05 < u < 0.45 else 1.0, 0.0 if 0.55 < u < 0.95 else 1.0)


def t_parse():
    assert C.parse_w3_monitor(None) is None and C.parse_w3_monitor("off") is None and C.parse_w3_monitor("") is None
    assert C.parse_w3_monitor("on") == C.W3MonitorConfig()
    assert C.parse_w3_monitor("note").mode == "note"
    c = C.parse_w3_monitor('{"lat_max_ticks": 8, "chatter_n": 5, "mode": "note"}')
    assert c.lat_max_ticks == 8.0 and c.chatter_n == 5 and isinstance(c.chatter_n, int) and c.mode == "note"
    for bad in ('{"foo": 1}', '{"walk_tilt_deg": 40}', '{"chatter_n": 2.5}', '{"walk_w": true}', '{"mode": "x"}', '[1]',
                '{"contact_max_age_s": 0.5}', '{"hl_wz_translate": 0.5}'):
        try:
            C.parse_w3_monitor(bad)
        except ValueError:
            continue
        raise AssertionError(f"거부 안 됨: {bad}")
    return "형식·범위·정수·mode 검증 8 종 거부"


def run(m, n, *, walking, contact_fn=None, grav_fn=None, gyro_fn=None, yaw_fn=None, k0=0):
    for k in range(k0, k0 + n):
        m.tick(walking=walking(k) if callable(walking) else walking,
               contact2=(contact_fn or (lambda k: (1.0, 1.0)))(k),
               gyro=(gyro_fn or (lambda k: (0.01, 0.0, 0.0)))(k),
               grav=(grav_fn or (lambda k: G0))(k), yaw=(yaw_fn or (lambda k: 0.3))(k))
    return k0 + n


def t_benign():
    """정상 걷기(교대 접촉·작은 흔들림) 60 s → 사건 0·정지 요청 0·켜기 허용."""
    m = C.W3WalkMonitor(C.W3MonitorConfig(walk_max_s=0.0))
    k = run(m, 100, walking=False)
    k = run(m, 3000, walking=True, contact_fn=walk_contacts, k0=k,
            grav_fn=lambda k: grav_rp(2.0 * math.sin(k * DT * 2 * math.pi * 0.55), 1.5 * math.sin(k * DT * 2 * math.pi * 1.1)),
            gyro_fn=lambda k: (0.4 * math.sin(k * DT * 7), 0.3 * math.cos(k * DT * 5), 0.0))
    assert not m.events and m.take_stop() is None, m.events
    k = run(m, 100, walking=False, k0=k)
    assert m.enable_gate() == (True, "ok"), m.enable_gate()
    return "정상 걷기 60 s: 사건 0·정지 0·켜기 허용"


def t_tilt_w():
    out = []
    for kind, kw in (("tilt", dict(grav_fn=lambda k: grav_rp(0, 2.0 if k < 150 else 14.0))),
                     ("w_xy", dict(gyro_fn=lambda k: (0.1 if k < 150 else 2.0, 0.0, 0.0)))):
        for mode in ("act", "note"):
            m = C.W3WalkMonitor(C.W3MonitorConfig(mode=mode))
            run(m, 50, walking=False)
            run(m, 250, walking=True, contact_fn=walk_contacts, k0=50, **kw)
            ev = [e for e in m.events if e["kind"] == kind]
            assert len(ev) == 1, (kind, mode, m.events)
            st = m.take_stop()
            if mode == "act":
                assert st is not None and st[0] == kind and ev[0]["action"] == "stop_walk", st
            else:
                assert st is None and ev[0]["action"] == "note:stop_walk"
            # 정지 시각: 위반 시작(틱 150 → t 3.02 s, 걷기 시작 틱 50) + 저역 도달 + 지속 0.2 s
            cfg = C.W3MonitorConfig()
            if kind == "tilt":     # 서기 0° 기준, 걷는 중 2° → 14° 계단(EMA ≈ 2°): 편차 14 − 12·e^{−t/τ} > 8
                t_cross = -cfg.walk_tau_s * math.log((14.0 - cfg.walk_tilt_deg) / 12.0)
            else:                  # |ω| 0.1 → 2.0: RMS² EMA 0.01 → 4: √(4 − 3.99 e^{−t/τ}) > 1.8
                t_cross = -cfg.walk_tau_s * math.log((4.0 - cfg.walk_w ** 2) / 3.99)
            assert abs(ev[0]["t"] - (151 * DT + t_cross + 0.2)) < 0.06, (ev[0], t_cross)
        out.append(kind)
    # 기울기는 걷기 전 서기 평균 기준: 서기 6° 로 시작하면 6°+7.5° 까지 허용(문턱 8°)
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    run(m, 200, walking=False, grav_fn=lambda k: grav_rp(0, 6.0))
    run(m, 300, walking=True, contact_fn=walk_contacts, k0=200, grav_fn=lambda k: grav_rp(0, 6.0 if k < 202 else 13.5))
    assert not m.events, m.events
    # 걸음 롤 흔들림 ±12°(1.1 Hz)·|ω_x| 진폭 1.5 rad/s 는 저역 뒤 문턱 아래(순간값이면 트립)
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    run(m, 200, walking=False)
    run(m, 1500, walking=True, contact_fn=walk_contacts, k0=200,
        grav_fn=lambda k: grav_rp(12.0 * math.sin(2 * math.pi * 1.1 * k * DT), 0.0),
        gyro_fn=lambda k: (1.5 * math.cos(2 * math.pi * 1.1 * k * DT), 0.0, 0.0))
    assert not m.events, m.events
    return "저역 기울기·|ω_xy| RMS 경계 → act 정지 요청/note 기록, 시각 = 위반 + 저역 도달 + 0.2 s, 기준 = 걷기 전 서기 평균, 걸음 롤 ±12° 무트립"


def t_contact():
    # 한 발 오래 들기
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    run(m, 50, walking=False)
    run(m, 100, walking=True, k0=50, contact_fn=lambda k: (0.0, 1.0) if k >= 80 else (1.0, 1.0))
    assert m.take_stop()[0] == "swing_long"
    # 양발 무접촉
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    run(m, 50, walking=False)
    run(m, 100, walking=True, k0=50, contact_fn=lambda k: (0.0, 0.0) if k >= 80 else (1.0, 1.0))
    assert m.take_stop()[0] == "both_off"
    # 걷는 중 채터(오른발 2 틱마다 토글)
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    run(m, 50, walking=False)
    run(m, 100, walking=True, k0=50, contact_fn=lambda k: (1.0, float((k // 2) % 2)) if k >= 80 else (1.0, 1.0))
    assert m.take_stop()[0] == "chatter"
    # 서기 중 채터 → 켜기 거부, 2 s 지나면 해제
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    k = run(m, 100, walking=False, contact_fn=lambda k: (1.0, 0.0) if 40 <= k < 45 else (1.0, 1.0))
    ok, why = m.enable_gate()
    assert not ok and "채터" in why, why
    run(m, 120, walking=False, k0=k)
    assert m.enable_gate()[0]
    # 센서 나이(발·피드백·stage8 명령 처리) — 0.3 s 지속해야 판정, 정상이면 즉시 해제
    m = C.W3WalkMonitor(C.W3MonitorConfig())         # 발 센서 나이는 즉시(지속 요구 없음), 0.2 s(1 펄스 결손)는 통과
    run(m, 5, walking=False)
    m.health(foot_age_s=0.2)
    assert m.enable_gate()[0]
    m.health(foot_age_s=0.3)
    assert not m.enable_gate()[0]
    for kw in ({"fb_age_s": 0.12}, {"s8_p50_ms": 120.0}):
        m = C.W3WalkMonitor(C.W3MonitorConfig())
        run(m, 10, walking=False)
        m.health(foot_age_s=0.05, fb_age_s=0.01, s8_p50_ms=1.0)
        assert m.enable_gate()[0]
        for k in range(10, 40):
            run(m, 1, walking=False, k0=k)
            m.health(**dict({"foot_age_s": 0.05, "fb_age_s": 0.01, "s8_p50_ms": 1.0}, **kw))
            if k == 20:
                assert m.enable_gate()[0], "0.3 s 전 판정"
        assert not m.enable_gate()[0], kw
        run(m, 5, walking=True, k0=40, contact_fn=walk_contacts)
        assert m.take_stop()[0] == "health"
        m.health(foot_age_s=0.05)
        assert m.enable_gate()[0]
    # 한 틱 튐은 무시
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    for k in range(40):
        run(m, 1, walking=False, k0=k)
        m.health(s8_p50_ms=150.0 if k % 10 == 0 else 2.0)
    assert m.enable_gate()[0] and not m.events
    # note 모드는 켜기를 막지 않는다
    m = C.W3WalkMonitor(C.W3MonitorConfig(mode="note"))
    for k in range(30):
        run(m, 1, walking=False, k0=k)
        m.health(foot_age_s=0.3)
    assert m.enable_gate()[0] and any(e["kind"] == "health" for e in m.events)
    return "한발 0.8 s·양발 0.2 s·채터 → 정지 / 서기 채터·센서 나이 3 종(0.3 s 지속) → 켜기 거부(해제 확인)·한 틱 튐 무시 / note 는 거부 안 함"


def t_yaw():
    # 정지 표류 4°/min → 20 s 뒤 고리 금지
    rate = math.radians(4.0) / 60.0
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    run(m, 1500, walking=False, yaw_fn=lambda k: 0.3 + rate * k * DT)
    assert m.hl_allowed()[0] is False and abs(m.yaw_rate_dpm - 4.0) < 0.05, (m.yaw_rate_dpm, m.hl_ban)
    # 1.5°/min → 허용, 타임아웃 = min(120, 60·5/1.5 = 200) = 120 / 예산 2° → 80 s
    rate = math.radians(1.5) / 60.0
    m = C.W3WalkMonitor(C.W3MonitorConfig(yaw_budget_deg=2.0))
    run(m, 1500, walking=False, yaw_fn=lambda k: 3.1 + rate * k * DT)     # ±π 경계 통과(펼침 확인)
    assert m.hl_allowed()[0] and abs(m.yaw_rate_dpm - 1.5) < 0.05, m.yaw_rate_dpm
    assert abs(m.hl_timeout_s(120.0) - 80.0) < 1.0, m.hl_timeout_s(120.0)
    # 조용하지 않은 서기(|ω| 큼) 표본은 쓰지 않는다
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    run(m, 1500, walking=False, yaw_fn=lambda k: 0.3 + math.radians(10.0) / 60.0 * k * DT, gyro_fn=lambda k: (0.3, 0.0, 0.0))
    assert m.yaw_rate_dpm is None and m.hl_allowed()[0]
    # 점프: 한 틱 3° (자이로 0) → 금지, 고리 걷는 중이면 정지
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    run(m, 50, walking=False)
    for k in range(50, 80):
        m.tick(walking=True, contact2=walk_contacts(k), gyro=(0.0, 0.0, 0.0), grav=G0,
               yaw=0.3 + (math.radians(5.0) if k >= 70 else 0.0), hl_latched=True)
    assert not m.hl_allowed()[0] and m.take_stop()[0] == "yaw_jump"
    # 실제 회전(자이로가 설명하는 변화)은 점프 아님
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    for k in range(200):
        m.tick(walking=True, contact2=walk_contacts(k), gyro=(0.0, 0.0, 0.2), grav=G0, yaw=C._wrap(0.2 * k * DT), hl_latched=True)
    assert m.hl_allowed()[0] and m.take_stop() is None, m.events
    return "표류 4°/min 금지·1.5°/min 허용(타임아웃 80 s)·±π 펼침·비정숙 표본 제외·점프 금지+정지·실회전 무해"


def t_cap_walkmax_settle():
    m = C.W3WalkMonitor(C.W3MonitorConfig(hl_wz_translate=0.1))
    assert m.hl_wz_cap(0.06, 0.0) == 0.1 and m.hl_wz_cap(0.0, 0.0) is None and m.hl_wz_cap(0.0, -0.03) == 0.1
    assert C.W3WalkMonitor(C.W3MonitorConfig()).hl_wz_cap(0.06, 0.0) is None
    assert C.W3WalkMonitor(C.W3MonitorConfig(mode="note", hl_wz_translate=0.1)).hl_wz_cap(0.06, 0.0) is None
    m = C.W3WalkMonitor(C.W3MonitorConfig(walk_max_s=10.0))
    run(m, 10, walking=False)
    run(m, 600, walking=True, contact_fn=walk_contacts, k0=10)
    ev = [e for e in m.events if e["kind"] == "walk_max"]
    assert len(ev) == 1 and abs(ev[0]["t"] - (11 * DT + 10.0)) < 0.03, ev
    # 정지 정착: 정지 뒤 |ω| 큰 채 3 s → note 1 회 / 조용하면 0
    for noisy, want in ((True, 1), (False, 0)):
        m = C.W3WalkMonitor(C.W3MonitorConfig())
        k = run(m, 100, walking=True, contact_fn=walk_contacts)
        run(m, 250, walking=False, k0=k, gyro_fn=(lambda k: (0.8, 0.0, 0.0)) if noisy else None)
        assert sum(e["kind"] == "settle" for e in m.events) == want, m.events
    return "병진 중 고리 wz 상한(act 만)·연속 걷기 상한 10 s·정지 뒤 미정착 기록"


def t_latency():
    rng = np.random.default_rng(0)
    T = 101
    base = np.cumsum(rng.normal(0, 0.01, (T + 20, 4)), axis=0) + 0.2 * np.sin(np.arange(T + 20)[:, None] * 2 * math.pi * 1.1 * DT + np.arange(4))
    got = []
    for L in range(0, 12):
        tgt = base[20:20 + T]
        enc = base[20 - L:20 - L + T] + rng.normal(0, 0.0005, (T, 4))
        Lh, c, nj = C.lag_xcorr(tgt, enc, maxlag=15)
        got.append(Lh)
        assert Lh == L and nj == 4, (L, Lh, c)
    # 들뜸 없음(정지) → None
    assert C.lag_xcorr(np.zeros((101, 4)), np.zeros((101, 4)))[0] is None
    # 감시기: 지연 8 틱이 3 회 연속 → lat_bad, 켜기 거부, 걷는 중이면 정지
    m = C.W3WalkMonitor(C.W3MonitorConfig(lat_max_ticks=6), joint_order=None)
    m.lat_idx = [0, 1, 2, 3]
    sig = np.cumsum(rng.normal(0, 0.01, (2000, 4)), axis=0) + 0.2 * np.sin(np.arange(2000)[:, None] * 0.14 + np.arange(4))
    for k in range(20, 400):
        m.tick(walking=True, contact2=walk_contacts(k), gyro=(0, 0, 0), grav=G0, yaw=0.0)
        m.track(np.r_[sig[k], np.zeros(8)], np.r_[sig[k - 8], np.zeros(8)])
    assert m.lat_bad and m.lat_est == 8, (m.lat_est, m.lat_bad)
    assert m.take_stop()[0] == "latency"
    for k in range(400, 420):                        # 서기로 돌아와도 켜기 거부 유지(서기에선 추정 안 함)
        m.tick(walking=False, contact2=(1, 1), gyro=(0, 0, 0), grav=G0, yaw=0.0)
        m.track(np.r_[sig[k], np.zeros(8)], np.r_[sig[k - 8], np.zeros(8)])
    assert not m.enable_gate()[0] and len(m._lat_buf) == 0
    # 서기 신호만으로는 추정하지 않는다(lat_walk_only)
    m = C.W3WalkMonitor(C.W3MonitorConfig(lat_max_ticks=6))
    m.lat_idx = [0, 1, 2, 3]
    for k in range(20, 400):
        m.tick(walking=False, contact2=(1, 1), gyro=(0, 0, 0), grav=G0, yaw=0.0)
        m.track(np.r_[sig[k], np.zeros(8)], np.r_[sig[k - 8], np.zeros(8)])
    assert m.lat_est is None and m.enable_gate()[0]
    # 들뜸 관절 수 부족(2 < 3) → 추정 없음
    sig2 = sig.copy(); sig2[:, 2:] = 0.0
    m = C.W3WalkMonitor(C.W3MonitorConfig(lat_max_ticks=6))
    m.lat_idx = [0, 1, 2, 3]
    for k in range(20, 400):
        m.tick(walking=True, contact2=walk_contacts(k), gyro=(0, 0, 0), grav=G0, yaw=0.0)
        m.track(np.r_[sig2[k], np.zeros(8)], np.r_[sig2[k - 8], np.zeros(8)])
    assert m.lat_est is None
    # 지연 3 틱이면 정상
    m = C.W3WalkMonitor(C.W3MonitorConfig(lat_max_ticks=6))
    m.lat_idx = [0, 1, 2, 3]
    for k in range(20, 400):
        m.tick(walking=True, contact2=walk_contacts(k), gyro=(0, 0, 0), grav=G0, yaw=0.0)
        m.track(np.r_[sig[k], np.zeros(8)], np.r_[sig[k - 3], np.zeros(8)])
    assert not m.lat_bad and m.lat_est == 3 and m.enable_gate()[0]
    # 꺼짐(lat_max_ticks 0) 은 버퍼도 안 쌓는다
    m = C.W3WalkMonitor(C.W3MonitorConfig())
    m.track(np.zeros(12), np.zeros(12))
    assert len(m._lat_buf) == 0
    return "교차상관 지연 0–11 틱 정확 추정·정지 None·걷는 중 8 틱 3 회 → 정지·서기 뒤 켜기 거부 유지·3 틱 정상·서기 신호 무추정·관절 수 부족 무추정·꺼짐 무연산"


def t_overtime():
    o = C.RecoveryOvertime(max_s=6.0, action="note")
    evs = [o.update(1) for _ in range(400)]
    fired = [i for i, e in enumerate(evs) if e is not None]
    assert fired == [299], fired
    assert o.update(0) is None and o.t_in == 0.0
    assert all(o.update(1) is None for _ in range(299)) and o.update(1) is not None
    for bad in ("x:6", "note:0.5", "freeze", "note:40"):
        try:
            C.parse_c5_overtime(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert C.parse_c5_overtime(None) is None and C.parse_c5_overtime("freeze:5").action == "freeze"
    return "회복 모드 6 s → 사건 1 회(틱 300)·모드 0 초기화·형식 거부 4 종"


def main():
    tests = [("설정 파싱", t_parse), ("정상 걷기 무해", t_benign), ("자세 경계", t_tilt_w), ("접촉 이상", t_contact),
             ("요 표류·점프", t_yaw), ("고리 상한·걷기 상한·정착", t_cap_walkmax_settle), ("실효 지연", t_latency),
             ("회복 체류 상한", t_overtime)]
    bad = 0
    for name, fn in tests:
        try:
            print(f"[PASS] {name}: {fn()}")
        except Exception as e:  # noqa: BLE001
            bad += 1
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print("selftest_w3_contingency:", "ALL PASS" if bad == 0 else f"{bad} FAILED")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
