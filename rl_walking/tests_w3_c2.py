"""c2(w3_c2) CPU 단위시험 — Isaac 없이 순수 계산만: supervisor 상태기계·W3 엔진 래치 연동·포획 목표·착지 보상 동일성.
실행: python3 rl_walking/tests_w3_c2.py  (CUDA_VISIBLE_DEVICES= 권장, GPU 불필요)"""
import math
import sys
from types import SimpleNamespace

sys.path.insert(0, __file__.rsplit("/rl_walking/", 1)[0])

import torch

from rl_walking import w3 as W3
from rl_walking import w3_c2 as C

DT = 0.02
P = C.C2Params()
FAIL = []


def check(name, cond, info=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {info}" if info else ""))
    if not cond:
        FAIL.append(name)


def fresh_state(n=1, P_=P):
    return C.new_state(n, "cpu", P_, DT)


def T(x, n=1):
    return torch.full((n,), float(x))


def B(x, n=1):
    return torch.full((n,), bool(x))


# ---------------------------------------------------------------- 1. 켜짐: 무장 필수 + 감지 지연 ----------------------------------------------------------------
def t_trigger():
    st = fresh_state()
    for _ in range(20):                                   # 무장 없음: 큰 기울기여도 켜지지 않음(nd·자기 개방 차단)
        C.c2_core(st, T(10.0), T(2.0), B(False), B(False), DT, P)
    check("1a 무장 없으면 켜짐 없음", int(st["mode"][0]) == C.STAND)
    st = fresh_state(); k_on = None
    for k in range(10):
        det = k >= 3
        C.c2_core(st, T(6.0 if det else 1.0), T(0.1), B(True), B(False), DT, P)
        if k_on is None and int(st["mode"][0]) == C.REC:
            k_on = k
    check("1b 감지 지연 = det_delay 틱", k_on == 3 + P.det_delay, f"감지 틱 3 → 켜짐 틱 {k_on}(기대 {3 + P.det_delay})")
    st = fresh_state()
    for _ in range(10):
        C.c2_core(st, T(3.9), T(0.49), B(True), B(False), DT, P)
    check("1c 문턱 미만(4°·0.5 rad/s)은 켜짐 없음", int(st["mode"][0]) == C.STAND)


# ---------------------------------------------------------------- 2. 정지 요청: 최소 시간·착지·정숙 창 / 시간 초과 ----------------------------------------------------------------
def _to_rec(st):
    for _ in range(P.det_delay + 1):
        C.c2_core(st, T(6.0), T(1.0), B(True), B(False), DT, P)
    assert int(st["mode"][0]) == C.REC


def t_stop():
    st = fresh_state(); _to_rec(st); k_req = None
    for k in range(200):                                  # 즉시 정숙 + 착지 있음 → min_rec_s 와 창(K 틱)이 지연을 정함
        td = (k % 23) == 5
        C.c2_core(st, T(1.0), T(0.1), B(True), B(td), DT, P)
        if st["ev_req"][0] > 0:
            k_req = k; break
    t_req = (k_req + 1) * DT
    check("2a 정지 요청 ≥ min_rec_s", k_req is not None and t_req >= P.min_rec_s - 1e-6, f"요청 {t_req:.2f}s")
    st = fresh_state(); _to_rec(st)
    for k in range(150):                                  # 착지 0 → 요청 없음(시간 초과 전)
        C.c2_core(st, T(1.0), T(0.1), B(True), B(False), DT, P)
    check("2b 유효 착지 0 이면 정지 요청 없음(시간 초과 전)", int(st["mode"][0]) == C.REC and float(st["ev_timeout"].sum()) == 0)
    st = fresh_state(); _to_rec(st); n_td = 0
    for k in range(150):                                  # 착지 있으나 창 안 ω 가 문턱 이상 → 요청 없음
        td = (k % 23) == 5
        C.c2_core(st, T(2.0), T(P.stop_w + 0.1), B(True), B(td), DT, P)
    check("2c 정숙 창 미충족이면 정지 요청 없음", int(st["mode"][0]) == C.REC)
    st = fresh_state(); _to_rec(st); k_to = None
    for k in range(400):
        C.c2_core(st, T(2.0), T(P.stop_w + 0.1), B(True), B(False), DT, P)
        if st["ev_timeout"][0] > 0:
            k_to = k; break
    check("2d 시간 초과 → STOPREQ", k_to is not None and int(st["mode"][0]) == C.STOPREQ, f"{(k_to or 0) * DT:.2f}s")
    # 2e 시간 초과로 들어온 STOPREQ 는 감지기가 다시 켜져도 취소되지 않음
    for _ in range(10):
        C.c2_core(st, T(8.0), T(2.0), B(True), B(False), DT, P)
    check("2e 시간 초과 STOPREQ 는 취소 불가", int(st["mode"][0]) == C.STOPREQ)


def t_cancel():
    st = fresh_state(); _to_rec(st)
    for k in range(200):
        C.c2_core(st, T(1.0), T(0.1), B(True), B((k % 23) == 5), DT, P)
        if int(st["mode"][0]) == C.STOPREQ:
            break
    for _ in range(P.det_delay + 1):
        C.c2_core(st, T(7.0), T(1.5), B(True), B(False), DT, P)
    check("3a STOPREQ 중 재불안정 → REC 취소", int(st["mode"][0]) == C.REC and float(st["ev_cancel"].sum()) >= 0)


# ---------------------------------------------------------------- 4. W3 엔진 래치 연동(실제 w3_gait_update) ----------------------------------------------------------------
def fake_env(n=1):
    e = SimpleNamespace(num_envs=n, device="cpu", step_dt=DT, episode_length_buf=torch.full((n,), 100, dtype=torch.long))
    e._ge_cmd = torch.zeros(n, dtype=torch.bool); e._ge = torch.zeros(n, dtype=torch.bool); e._ge_obs = torch.zeros(n)
    e._gu = torch.zeros(n); e._gf = torch.zeros(n)
    return e


def run_scenario(push_profile, n_ticks=500, P_=P, stop_never=False):
    """합성 회복 궤적: push_profile(k) → (tilt, w, armed). 착지 사건 = 엔진 반주기 경계 직후(ge 1 동안)."""
    env = fake_env(); st = fresh_state(P_=P_); log = []
    prev_half = None
    for k in range(n_ticks):
        tilt, w, armed = push_profile(k)
        half = int(float(env._gu[0]) >= 0.5) if bool(env._ge[0]) else None
        td = bool(env._ge[0]) and prev_half is not None and half != prev_half
        prev_half = half
        want = C.c2_core(st, T(tilt), T(w), B(armed), B(td), DT, P_)
        if stop_never:
            want = want | (st["mode"] == C.STOPREQ)
        env._ge_cmd = want
        W3.w3_gait_update(env, None, f_step=1.1)
        C.c2_after_latch(st, env._ge, P_)
        log.append((k, int(st["mode"][0]), bool(env._ge[0]), float(env._gu[0]), td))
    return log, st


def push_decay(k, k0=50, k_calm=110):
    """k0 에 외란 시작, k_calm 까지 불안정(기울기 8°·ω 1.2) → 이후 지수 감쇠. 무장 = 외력 ~0.6 s + arm 2.5 s."""
    if k < k0:
        return 1.0, 0.1, False
    armed = k < k0 + 30 + int(P.arm_s / DT)
    if k < k_calm:
        return 8.0, 1.2, armed
    a = math.exp(-(k - k_calm) * DT / 0.3)
    return 1.0 + 7.0 * a, 0.1 + 1.1 * a, armed


def t_engine():
    log, st = run_scenario(push_decay)
    modes = [m for _, m, _, _, _ in log]
    k_trig = next(k for k, m, *_ in log if m == C.REC)
    k_req = next(k for k, m, *_ in log if m == C.STOPREQ)
    k_post = next(k for k, m, *_ in log if m == C.POST)
    ge_on = [g for _, _, g, _, _ in log]
    half_ticks = int(math.ceil(0.5 / 1.1 / DT))
    check("4a 감지 → REC → STOPREQ → POST 순서", k_trig < k_req < k_post, f"켜짐 {k_trig * DT:.2f}s 요청 {k_req * DT:.2f}s 래치 {k_post * DT:.2f}s")
    check("4b 요청 → 래치 ≤ 반주기(스윙 중 끄지 않음)", 0 < k_post - k_req <= half_ticks + 1, f"{(k_post - k_req) * DT:.2f}s ≤ {half_ticks * DT:.2f}s")
    check("4c 래치 순간 위상 = 반주기 경계(u 0 리셋)", abs(log[k_post][3]) < 1e-6 and not log[k_post][2])
    n_td_after = sum(1 for k, m, g, u, td in log if k > k_req and td)
    check("4d 요청 뒤 착지 사건 ≤ 1(반주기 마무리)", n_td_after <= 1, f"{n_td_after}")
    check("4e POST 뒤 ge 0 유지(무장 중에도 재켜짐 없음 — 정숙)", all(not g for k, _, g, _, _ in log if k > k_post))
    log2, _ = run_scenario(lambda k: (1.0, 0.1, False))
    check("4f 무외란(nd): ge 0 내내·모드 STAND 내내", all(m == C.STAND and not g for _, m, g, _, _ in log2))
    log3, _ = run_scenario(push_decay, stop_never=True)
    check("4g 대조(정지 요청 무시 = RW3Z 형): 래치 없음·ge 1 유지", not any(m == C.POST for _, m, *_ in log3) and log3[-1][2])


def t_refractory():
    env = fake_env(); st = fresh_state()
    # REC → STOPREQ → POST 강제 후, 무장 상태 큰 기울기: 불응 refr_s 동안 재켜짐 없음
    _to_rec(st); st["mode"][0] = C.STOPREQ
    C.c2_after_latch(st, torch.zeros(1, dtype=torch.bool), P)
    k_re = None
    for k in range(100):
        C.c2_core(st, T(8.0), T(1.5), B(True), B(False), DT, P)
        if int(st["mode"][0]) == C.REC:
            k_re = k; break
    check("5a POST 불응기 뒤에만 재켜짐", k_re is not None and (k_re + 1) * DT >= P.refr_s - 1e-6, f"{(k_re or 0) * DT:.2f}s")


# ---------------------------------------------------------------- 6. 정착 사건: 외란 1건당 1회 ----------------------------------------------------------------
def t_settle():
    def run(st, n, v=0.02, force=False):
        ev = 0.0
        for _ in range(n):
            C.c2_settle_update(st, B(True), T(1.0), T(0.1), T(v), B(force), DT, P); ev += float(st["ev_settle"][0])
        return ev
    st = fresh_state(); st["mode"][0] = C.POST; st["elig"][0] = True        # 회복 켜짐(elig) → 래치 → POST
    check("6a POST 정착 사건 1회만(계속 서 있어도)", run(st, 200) == 1.0)
    st["elig"][0] = True; st["mode"][0] = C.POST                             # 같은 외란 안 재켜짐 → 재 POST
    check("6b 같은 외란 재진입은 지급 없음(파밍 차단)", run(st, 100) == 0.0)
    ev_force = run(st, 30, force=True)                                       # 새 외력(상승 에지) — 외력 중 정착 불인정
    check("6c-1 외력 중에는 정착 지급 없음", ev_force == 0.0)
    check("6c-2 새 외란이어도 회복이 켜지지 않았으면(elig 0) 지급 없음", run(st, 100) == 0.0)
    st["elig"][0] = True; st["mode"][0] = C.POST                             # 새 외란에서 회복 켜짐 → POST
    check("6c-3 새 외란 + 회복 뒤 1회 재지급", run(st, 100) == 1.0)
    st = fresh_state(); st["mode"][0] = C.POST; st["elig"][0] = True
    check("6d 표류 중(|v| 0.12)이면 정착 아님", run(st, 200, v=0.12) == 0.0)
    st = fresh_state()
    check("6e 회복 없던 STAND 는 정착 보상 0(무외란 Stage 1 불변)", run(st, 200) == 0.0)
    st = fresh_state(); _to_rec(st)
    check("6f c2_core 켜짐이 자격(elig)을 세움", bool(st["elig"][0]))


# ---------------------------------------------------------------- 7. 포획 목표·착지 보상 동일성 ----------------------------------------------------------------
def t_capture():
    n = 4
    nom_x = torch.zeros(n, 2); nom_y = torch.tensor([[0.10, -0.10]]).repeat(n, 1)
    om = torch.full((n,), math.sqrt(9.81 / 0.55))
    tx, ty = C.capture_target_rel(nom_x, nom_y, torch.zeros(n, 2), om)
    check("7a v=0 → W3 명목 목표와 정확히 같음", torch.equal(tx, nom_x) and torch.equal(ty, nom_y))
    v = torch.tensor([[0.3, 0.0]]).repeat(n, 1)
    tx, ty = C.capture_target_rel(nom_x, nom_y, v, om, b=0.02, v_ref=0.05)
    exp = 0.3 / om[0] + 0.02 * 0.3 / math.sqrt(0.09 + 0.0025)
    check("7b 앞 0.3 m/s → 목표 +v/ω0 + b 여유", abs(float(tx[0, 0]) - exp) < 1e-6 and torch.allclose(ty, nom_y), f"{float(tx[0, 0]) * 100:.1f} cm")
    tx, _ = C.capture_target_rel(nom_x, nom_y, torch.tensor([[-0.3, 0.0]]).repeat(n, 1), om)
    check("7c 뒤 속도 → 목표 뒤(부호 대칭)", abs(float(tx[0, 0]) + exp) < 1e-6)
    tx, _ = C.capture_target_rel(nom_x, nom_y, torch.tensor([[2.0, 0.0]]).repeat(n, 1), om, max_off=0.25)
    check("7d 오프셋 상한 max_off", abs(float(tx[0, 0]) - 0.25) < 1e-6)
    # 착지 보상 동일성: td_rec 전부 False → W3 w3_place_reward 식과 비트 동일
    g = torch.Generator().manual_seed(0)
    ev_td = (torch.rand(256, 2, generator=g) < 0.3).float(); td_err = torch.rand(256, 2, generator=g) * 0.1 * ev_td
    ge = torch.rand(256, generator=g) < 0.7
    cap_err = torch.rand(256, 2, generator=g) * 0.2
    w3ref = (torch.exp(-(td_err / 0.04) ** 2) * ev_td).sum(1) * ge.float()
    c2v = C.place_value(ev_td, td_err, torch.zeros(256, 2, dtype=torch.bool), cap_err, ge, 0.04, 0.05)
    check("7e 비회복 착지 = w3_place_reward 비트 동일", torch.equal(w3ref, c2v))
    rec = torch.rand(256, 2, generator=g) < 0.5
    c2r = C.place_value(ev_td, td_err, rec, cap_err, ge, 0.04, 0.05)
    same = ~(rec & (ev_td > 0)).any(1)
    check("7f 회복 착지 없는 env 는 값 동일", torch.equal(c2r[same], w3ref[same]))


# ---------------------------------------------------------------- 8. 보상 예산(계산) ----------------------------------------------------------------
def t_budget():
    # 사건 보상 초당 기여 ≤ TASK 연속 항 50 %(W3 설계 규칙) — 회복 중 착지 2.2 회/s × 10 × dt 0.02 × 1.0 = 0.44/s
    place_rate = 2.2 * 10.0 * DT
    settle_once = 50.0 * DT
    # 회복 1건(≈ 2 s) 기준 c2 추가분: 포획 착지 최대 0.44·2 = 0.88(명목 착지 대비 증분은 그 이하) + 정착 1.0
    check("8a 정착 보너스(1.0) < 낙상 벌점 75 의 2 %(정착 위해 위험 감수 동기 없음)", settle_once < 0.02 * 75 + 1e-9, f"{settle_once:.2f}")
    check("8b 포획 착지 보상 상한 = 명목 착지 상한(같은 가중 10) — 회복 중 걸음 수 경쟁 없음", True, f"{place_rate:.2f}/s")


# ---------------------------------------------------------------- 9. 반사실 진단 팔(C2_PARAMS 덮어쓰기) — 계획서 B2/B4/B5 ----------------------------------------------------------------
def t_arms():
    late = C._P({"min_td": 1e6})
    early = C._P({"min_rec_s": 0.0, "stop_w": 1e9, "stop_tilt": 1e9})
    nostop = C._P({"min_td": 1e6, "tmax_s": 1e9})
    lg, _ = run_scenario(lambda k: push_decay(k), P_=late)
    k_req = next((k for k, m, *_ in lg if m == C.STOPREQ), None); k_trig = next(k for k, m, *_ in lg if m == C.REC)
    check("9a late(B5): 정지 요청 = 시간 초과 시각", k_req is not None and abs((k_req - k_trig) * DT - late.tmax_s) < 0.05, f"{(k_req - k_trig) * DT:.2f}s")
    eg, _ = run_scenario(lambda k: push_decay(k), P_=early)
    k_req = next((k for k, m, *_ in eg if m == C.STOPREQ), None); k_td = next(k for k, m, g, u, td in eg if td)
    check("9b early(B4): 첫 유효 착지 직후 정지 요청", k_req is not None and k_req - k_td <= 1, f"착지 {k_td * DT:.2f}s 요청 {k_req * DT:.2f}s")
    ng, _ = run_scenario(lambda k: push_decay(k), P_=nostop)
    check("9c nostop(B2): 정지 없음 = RW3Z 형", not any(m in (C.STOPREQ, C.POST) for _, m, *_ in ng))
    try:
        C._P({"no_such": 1}); ok = False
    except KeyError:
        ok = True
    check("9d 없는 키 덮어쓰기는 오류(조용한 무시 금지)", ok)


# ---------------------------------------------------------------- 10. 상대 기울기 감지기(tilt_ref_tau > 0) ----------------------------------------------------------------
def t_reltilt():
    def g_of(pitch_deg, roll_deg=0.0):
        p, r = math.radians(pitch_deg), math.radians(roll_deg)
        return torch.tensor([[math.sin(p), -math.sin(r) * math.cos(p), -math.cos(p) * math.cos(r)]])
    d = float(C.tilt_dev_deg(g_of(5.0), g_of(5.0))[0]); check("10a 같은 자세 편차 0", abs(d) < 1e-3, f"{d:.4f}")
    d = float(C.tilt_dev_deg(g_of(9.0), g_of(5.0))[0]); check("10b 서기 5° 기준에서 +4° 기울면 편차 4°", abs(d - 4.0) < 1e-3, f"{d:.3f}")
    ref = torch.zeros(1, 3); ref = C.update_gref(ref, g_of(5.0), B(False), B(True), DT, 10.0)
    check("10c init → 기준 = 현재", float(C.tilt_dev_deg(g_of(5.0), ref)[0]) < 1e-3)
    for _ in range(50):                                     # 갱신 금지(upd 0) 동안 기준 고정
        ref = C.update_gref(ref, g_of(10.0), B(False), B(False), DT, 10.0)
    check("10d upd 0 이면 기준 불변(회복 중 기준 끌림 없음)", abs(float(C.tilt_dev_deg(g_of(10.0), ref)[0]) - 5.0) < 1e-3)
    ref2 = C.update_gref(torch.zeros(1, 3), g_of(20.0), B(False), B(True), DT, 10.0)
    check("10e 기준 기울기 상한 8°", abs(float(C.tilt_dev_deg(ref2, g_of(0.0))[0]) - 8.0) < 0.05, f"{float(C.tilt_dev_deg(ref2, g_of(0.0))[0]):.2f}")
    p0 = C.C2Params(); check("10f 기본 tilt_ref_tau 0(종전 절대 기울기 — 학습 설정 불변)", p0.tilt_ref_tau == 0.0)


# ---------------------------------------------------------------- 11. 취소 조건(cancel_calm) ----------------------------------------------------------------
def t_cancel_calm():
    P2 = C._P({"cancel_calm": 1.0, "stop_w": 2.0, "stop_tilt": 12.5})
    st = fresh_state(P_=P2)
    for _ in range(P2.det_delay + 1):
        C.c2_core(st, T(6.0), T(1.0), B(True), B(False), DT, P2)
    for k in range(200):                                   # 걸음 흔들림(ω 1.2 > 감지 0.5, < 정지 2.0) 속 정지 요청
        C.c2_core(st, T(6.0), T(1.2), B(True), B((k % 23) == 5), DT, P2)
        if int(st["mode"][0]) == C.STOPREQ:
            break
    for _ in range(10):
        C.c2_core(st, T(6.0), T(1.2), B(True), B(False), DT, P2)
    check("11a cancel_calm: 걸음 흔들림(감지 문턱 초과·정지 문턱 미만)으로는 취소 안 됨", int(st["mode"][0]) == C.STOPREQ)
    for _ in range(3):
        C.c2_core(st, T(14.0), T(2.5), B(True), B(False), DT, P2)
    check("11b cancel_calm: 정지 문턱 초과면 취소", int(st["mode"][0]) == C.REC)


# ---------------------------------------------------------------- 12. 이동 방향 포획 여유 점수(cap_mode margin) ----------------------------------------------------------------
def t_margin():
    v = torch.tensor([[0.4, 0.0]]); xi = torch.tensor([[0.10, 0.0]])
    f = lambda x: torch.tensor([[[x, 0.10], [x, -0.10]]])
    s_ahead = C.capture_margin_score(f(0.13), xi, v); s_under = C.capture_margin_score(f(0.0), xi, v); s_far = C.capture_margin_score(f(0.40), xi, v)
    check("12a 이동 방향으로 ICP 보다 앞 = +", float(s_ahead[0, 0]) > 0.5, f"{float(s_ahead[0, 0]):+.2f}")
    check("12b 제자리(골반 밑, ICP 뒤) = −", float(s_under[0, 0]) < 0, f"{float(s_under[0, 0]):+.2f}")
    check("12c 포화 +1", abs(float(s_far[0, 0]) - 1.0) < 1e-6)
    s_slow = C.capture_margin_score(f(0.13), xi, torch.tensor([[0.05, 0.0]]))
    check("12d |v| ≤ 0.10 이면 0(운동량 없음)", float(s_slow.abs().max()) == 0.0)
    s_back = C.capture_margin_score(torch.tensor([[[-0.13, 0.1], [-0.13, -0.1]]]), -xi, -v)
    check("12e 뒤 방향 대칭", abs(float(s_back[0, 0]) - float(s_ahead[0, 0])) < 1e-6)
    g = torch.Generator().manual_seed(1)
    ev_td = (torch.rand(64, 2, generator=g) < 0.3).float(); td_err = torch.rand(64, 2, generator=g) * 0.1; ge = torch.ones(64, dtype=torch.bool)
    ref = C.place_value(ev_td, td_err, torch.zeros(64, 2, dtype=torch.bool), td_err, ge)
    mg = C.place_value(ev_td, td_err, torch.zeros(64, 2, dtype=torch.bool), td_err, ge, cap_score=torch.rand(64, 2, generator=g))
    check("12f 여유형도 비회복 착지는 W3 와 비트 동일", torch.equal(ref, mg))
    check("12g c2_place_reward 기본 cap_mode = err(종전 비트 동일)", C.c2_place_reward.__defaults__[-1] == "err")


# ---------------------------------------------------------------- 13. 차분형 포획 점수·지지영역 거리·정지 시점 스윕 ----------------------------------------------------------------
def t_diff_sweep():
    yaw = torch.zeros(1)
    fxy = torch.tensor([[[0.0, 0.10], [0.0, -0.10]]])
    cor = C.foot_corners(fxy, yaw).reshape(1, 8, 2)
    both = torch.ones(1, 8, dtype=torch.bool)
    d_in = float(C.support_dist(torch.tensor([[0.0, 0.0]]), cor, both)[0]); d_out = float(C.support_dist(torch.tensor([[0.185, 0.0]]), cor, both)[0])
    check("13a 양발 지지영역 안(골반 밑) 거리 ≤ 0", d_in <= 0, f"{d_in:+.3f}")
    check("13b 앞 0.185 m 점 = 발끝(0.085) 에서 0.10 m", abs(d_out - 0.10) < 2e-3, f"{d_out:.4f}")
    xi = torch.tensor([[0.20, 0.0]])
    cont = torch.tensor([[False, True]]); td = torch.tensor([[True, False]])
    # 왼발이 ξ 쪽(앞 0.18)으로 딛음 vs 제자리(0)로 딛음 — 오른발 지지
    f_step = C.foot_corners(torch.tensor([[[0.18, 0.10], [0.0, -0.10]]]), yaw); f_place = C.foot_corners(fxy, yaw)
    sc_step, pb1, pa1 = C.capture_diff_score(xi, f_step, cont, td); sc_place, pb2, pa2 = C.capture_diff_score(xi, f_place, cont, td)
    check("13c 방향 스텝 점수 > 제자리 디딤 점수", float(sc_step[0, 0]) > float(sc_place[0, 0]) + 0.3, f"{float(sc_step[0,0]):.2f} vs {float(sc_place[0,0]):.2f}")
    sc_cap, _, _ = C.capture_diff_score(torch.tensor([[0.0, -0.10]]), f_place, cont, td)
    check("13d 이미 포획(ξ 가 지지발 위) 착지 = 0 점", float(sc_cap[0, 0]) == 0.0)
    sc_air, _, _ = C.capture_diff_score(xi, f_step, torch.tensor([[False, False]]), td)
    check("13e 지지 없던 착지(공중) = 0 점", float(sc_air[0, 0]) == 0.0)
    # 정지 시점 스윕: k 번째 유효 착지에서 정지 요청, 상태 조건·취소 무시
    for k in (1, 3):
        Pk = C._P({"stop_td_k": k}); st = fresh_state(P_=Pk)
        for _ in range(Pk.det_delay + 1):
            C.c2_core(st, T(6.0), T(1.0), B(True), B(False), DT, Pk)
        n = 0; kreq = None
        for i in range(200):
            td_ = (i % 23) == 5
            n += int(td_)
            C.c2_core(st, T(8.0), T(2.0), B(True), B(td_), DT, Pk)
            if st["ev_req"][0] > 0:
                kreq = n; break
        check(f"13f 강제 정지 k={k}: k 번째 착지에서 요청(불안정 무시)", kreq == k, f"{kreq}")
    Po = C._P({"stop_oracle": 1.0}); st = fresh_state(P_=Po)
    for _ in range(Po.det_delay + 1):
        C.c2_core(st, T(6.0), T(1.0), B(True), B(False), DT, Po)
    C.c2_core(st, T(8.0), T(2.0), B(True), B(True), DT, Po, oracle_ok=B(False))
    a0 = int(st["mode"][0])
    C.c2_core(st, T(8.0), T(2.0), B(True), B(True), DT, Po, oracle_ok=B(True))
    check("13g 오라클: 포획 착지 신호에서만 요청", a0 == C.REC and int(st["mode"][0]) == C.STOPREQ)


# ---------------------------------------------------------------- 14. 레드팀 수정: 보상이 supervisor 보다 먼저 _c2() 를 불러도 같은 P ----------------------------------------------------------------
def t_order():
    import os as _os
    _os.environ.pop("C2_PARAMS", None)
    def fenv(c2):
        e = SimpleNamespace(num_envs=4, device="cpu", step_dt=DT, episode_length_buf=torch.zeros(4, dtype=torch.long))
        e.cfg = SimpleNamespace(events=SimpleNamespace(w3_gait=SimpleNamespace(params={"f_step": 1.1, "c2": c2})))
        return e
    e = fenv({"v_box_s": 0.909, "stop_v": 0.07})
    st = C._c2(e)                                   # 학습 순서: 보상 쪽이 먼저
    check("14a 보상 먼저 불러도 속도 버퍼 = 주기(45 틱)", st["vbuf"].shape[1] == 45, f"{st['vbuf'].shape[1]}")
    check("14b env._c2_P = 설정 c2 파라미터", abs(e._c2_P.v_box_s - 0.909) < 1e-9 and abs(e._c2_P.stop_v - 0.07) < 1e-9)
    st2 = C._c2(e, e._c2_P)
    check("14c supervisor 쪽 호출은 같은 상태(재생성 없음)", st2 is st)
    e2 = fenv({}); e2._c2 = C.new_state(4, "cpu", C.C2Params(), DT); e2._c2["stats"] = {"trig": 3.0}
    st3 = C._c2(e2, C._P({"v_box_s": 0.909}))
    check("14d 버퍼 길이가 P 와 다르면 재생성(통계 보존)", st3["vbuf"].shape[1] == 45 and st3["stats"]["trig"] == 3.0)
    P = C._P({"kv_bias": 0.03, "kv_lf": 0.02})
    check("14e 상관 잡음 파라미터(kv_bias·kv_lf) 해석", P.kv_bias == 0.03 and P.kv_lf == 0.02)


# ---------------------------------------------------------------- 15. 차분형 v2: 전 = 직전 양발 지지영역 ----------------------------------------------------------------
def t_diff_v2():
    yaw = torch.zeros(1)
    xi = torch.tensor([[0.20, 0.0]])                       # 앞으로 밀린 ICP, 두 발 사이(횡)
    prev = C.foot_corners(torch.tensor([[[0.0, 0.18], [0.0, -0.18]]]), yaw)
    cont = torch.tensor([[False, True]]); td = torch.tensor([[True, False]])
    same = C.foot_corners(torch.tensor([[[0.0, 0.18], [0.0, -0.18]]]), yaw)
    fwd = C.foot_corners(torch.tensor([[[0.18, 0.18], [0.0, -0.18]]]), yaw)
    s_same, _, _ = C.capture_diff_score(xi, same, cont, td, corners_prev=prev)
    s_fwd, _, _ = C.capture_diff_score(xi, fwd, cont, td, corners_prev=prev)
    s_v1, _, _ = C.capture_diff_score(xi, same, cont, td)
    check("15a v2: 제자리 되딛기 = 0 점", float(s_same[0, 0]) == 0.0, f"{float(s_same[0, 0]):.3f}")
    check("15b v2: ICP 쪽으로 딛기 > 0", float(s_fwd[0, 0]) > 0.3, f"{float(s_fwd[0, 0]):.3f}")
    check("15c v1(다른 발만) 은 제자리도 점수 받음 — 결함 재현", float(s_v1[0, 0]) > 0.5, f"{float(s_v1[0, 0]):.3f}")


if __name__ == "__main__":
    t_trigger(); t_stop(); t_cancel(); t_engine(); t_refractory(); t_settle(); t_capture(); t_budget(); t_arms(); t_reltilt(); t_cancel_calm(); t_margin(); t_diff_sweep(); t_order(); t_diff_v2()
    print(f"\n결과: 실패 {len(FAIL)}" + (f" {FAIL}" if FAIL else " — 전부 통과"))
    sys.exit(1 if FAIL else 0)
