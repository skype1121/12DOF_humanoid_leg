"""W3 Stage 2(옆걸음) CPU 단위시험 (1006) — torch CPU 만(Isaac·GPU 불필요).

  python3 rl_walking/scripts/unit_w3_s2.py [--ref 6a91c303]

rl_walking/w3.py 를 가짜 mdp_hang(스텁)과 함께 올리고, --ref 커밋의 w3.py(git show)를 같은 방식으로 올려 비교한다.
  [A] Stage 0/1 비트 동일: install_w3_cmd(stage 0·1, S1b/S1c/S1d 인자)·w3_gait_update(lead 기본 0)·w3_step_tracker+w3_stride(vs w3_stride2, vy 0)
      — 같은 난수열·같은 합성 궤적에서 출력 텐서 torch.equal. 기존 함수 소스(AST) 불변 확인(install_w3_cmd·w3_gait_update 제외).
  [B] 반사실 보상(합성 tracker 상태 = 실제 tracker 를 합성 발 궤적으로 구동): vy ∈ {±0.03, ±0.05, ±0.08} 에서
      같은 방향 ≫ 0(제자리) ≫ 반대, 기울임만(착지 없음) = 0, 제자리 디딤 = 0, 폭 위반·교차 = 0 점 + 벌점, 요 180° 스폰에서도 부호 유지.
  [C] 리드 발: vy < 0 → STEP 시작 위상 0.5(오른발), vy ≥ 0 → 0.
  [D] Stage 2 명령 분포(STAND 30 %·제자리 0.3·앞뒤 0.3·옆 0.4, |vy| 범위·부호 반반).
  [E] vy 커리큘럼 진급 규칙(부호 ≥ 0.9·폭 위반 < 0.02·체류).
  [F] Isaac Lab UniformVelocityCommand._update_command(heading → wz, 서기 → 0; AST 추출) 를 리샘플 사이 매 틱 돌려도 vy 가 남는지.
종료 코드 0 = 전부 통과. 결과 표는 stdout.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DT = 0.02
FAILS = []


def check(cond, msg):
    print(("  [통과] " if cond else "  [실패] ") + msg)
    if not cond:
        FAILS.append(msg)


# ------------------------------------------------------------------ 모듈 적재(스텁 mdp_hang)
def _stub_h():
    H = types.ModuleType("w3stub.mdp_hang")
    H.G = 9.81
    H._base_cfg = lambda: types.SimpleNamespace(name="robot")
    H._g21_bids = lambda env: [0, 1]
    H._feet_body_ids = lambda env: [0, 1]
    H._resolve_nested = lambda env, p: dict(p or {})
    return H


def _load(src, tag):
    pkg = f"w3stub_{tag}"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg); m.__path__ = []; sys.modules[pkg] = m
        h = _stub_h(); sys.modules[f"{pkg}.mdp_hang"] = h; m.mdp_hang = h
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(src); tmp.close()
    try:
        spec = importlib.util.spec_from_file_location(f"{pkg}.w3", tmp.name)
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.unlink(tmp.name)
    return mod


# ------------------------------------------------------------------ 가짜 env
class _Term:
    def __init__(self, N):
        self.vel_command_b = torch.zeros(N, 3)
        self.is_standing_env = torch.zeros(N, dtype=torch.bool)

    def _resample_command(self, ids):                     # 원래 리샘플(난수 소비) 흉내
        n = len(ids) if not isinstance(ids, slice) else self.vel_command_b.shape[0]
        self.vel_command_b[ids] = torch.empty(n, 3).uniform_(-0.2, 0.2)


class _CM:
    def __init__(self, N):
        self.t = _Term(N)

    def get_term(self, name):
        return self.t

    def get_command(self, name):
        return self.t.vel_command_b


class _Env:
    def __init__(self, N, yaw=None):
        self.num_envs = N; self.device = torch.device("cpu"); self.step_dt = DT
        self.episode_length_buf = torch.full((N,), 10, dtype=torch.long); self.common_step_counter = 0
        self.command_manager = _CM(N)
        self._ge_obs = torch.zeros(N)                    # install_w3_cmd 가 만드는 상태(시험에선 직접)
        yaw =torch.zeros(N) if yaw is None else yaw
        q = torch.stack([torch.cos(yaw / 2), torch.zeros(N), torch.zeros(N), torch.sin(yaw / 2)], 1)
        self.data = types.SimpleNamespace(root_quat_w=q, root_pos_w=torch.zeros(N, 3), body_pos_w=torch.zeros(N, 2, 3),
                                          default_mass=torch.full((N, 1), 17.6), root_lin_vel_w=torch.zeros(N, 3),
                                          body_lin_vel_w=torch.zeros(N, 2, 3))
        self.sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=torch.zeros(N, 2, 3)), body_names=["l", "r"])
        self.scene = _Scene(self)


class _Scene:
    def __init__(self, env):
        self.env = env; self.sensors = {"contact_forces": env.sensor}

    def __getitem__(self, k):                            # env.scene["robot"].data
        return self.env


def _yaw_rot(yaw, v):                                    # 요 프레임 → 월드 (N,2)
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)


def _scenario(mod, N, cmd, foot, dpos, lean=None, yaw=None, half_w=0.144, use_new=True, lead=0.0, swing=True, ticks=60, other_off=None):
    """한 발(foot 0 왼 / 1 오)이 tick 5 이륙 → 공중 12 틱(정점 5 cm) → 요 프레임 변위 dpos 만큼 옮겨 착지. 골반은 lean(요 프레임) 만큼 서서히 이동.
    반환: (stride 보상 합, stride2 보상 합, 폭 벌점 합, 착지 td_dy, 상태). use_new=False 면 stride2/폭 없음(구 모듈)."""
    env = _Env(N, yaw)
    yaw = torch.zeros(N) if yaw is None else yaw
    env.command_manager.t.vel_command_b[:] = cmd
    env._ge_cmd = torch.ones(N, dtype=torch.bool)
    feet0 = torch.stack([torch.zeros(N), torch.full((N,), half_w)], 1), torch.stack([torch.zeros(N), torch.full((N,), -half_w)], 1)
    feet0 = list(feet0)
    if other_off is not None:                             # 지지발(스윙 아닌 발) 시작 위치 이동(요 프레임) — 벌어진 자세에서 따라오는 스텝
        feet0[1 - foot] = feet0[1 - foot] + other_off
    feet_w = [_yaw_rot(yaw, feet0[0]), _yaw_rot(yaw, feet0[1])]
    dp_w = _yaw_rot(yaw, dpos)
    lean_w = _yaw_rot(yaw, lean) if lean is not None else torch.zeros(N, 2)
    r1 = torch.zeros(N); r2 = torch.zeros(N); pw = torch.zeros(N); tddy = torch.zeros(N)
    mg = 17.6 * 9.81
    for t in range(ticks):
        env.episode_length_buf += 1; env.common_step_counter += 1
        a = min(1.0, t / 30.0)
        env.data.root_pos_w[:, :2] = lean_w * a
        F = torch.zeros(N, 2, 3); F[:, :, 2] = mg / 2
        bp = torch.zeros(N, 2, 3)
        for k in (0, 1):
            bp[:, k, :2] = feet_w[k]
        if swing:
            if 5 <= t < 17:
                F[:, foot, 2] = 0.0; F[:, 1 - foot, 2] = mg
                v = (t - 5) / 12.0
                bp[:, foot, 2] = 0.05 * math.sin(math.pi * min(1.0, v + 1 / 12))
                bp[:, foot, :2] = feet_w[foot] + dp_w * v
            elif t >= 17:
                bp[:, foot, :2] = feet_w[foot] + dp_w
        env.data.body_pos_w = bp; env.sensor.data.net_forces_w = F
        if use_new:
            mod.w3_gait_update(env, None, f_step=1.1, lead_vy_cmin=lead)
        else:
            mod.w3_gait_update(env, None, f_step=1.1)
        mod.w3_step_tracker(env, None, nominal_half_width=0.10)
        r1 += mod.w3_stride(env)
        if use_new:
            mod.w3_lat_tracker(env, None, nominal_half_width=half_w, width_margin=0.05)
            r2 += mod.w3_stride2(env); pw += mod.w3_width_pen(env)
            tddy += env._w3lat["td_dy"].sum(1)
    return r1, r2, pw, tddy, env


# ------------------------------------------------------------------ [A] 비트 동일
def test_A(old, new):
    print("[A] Stage 0/1 비트 동일")
    src_o = ast.parse(old.__src__); src_n = ast.parse(new.__src__)
    fo = {n.name: ast.dump(n) for n in src_o.body if isinstance(n, ast.FunctionDef)}
    fn = {n.name: ast.dump(n) for n in src_n.body if isinstance(n, ast.FunctionDef)}
    changed = sorted(k for k in fo if fo[k] != fn.get(k))
    check(changed == ["install_w3_cmd", "w3_gait_update"], f"기존 함수 중 바뀐 것 = install_w3_cmd·w3_gait_update 뿐 (실제 {changed})")
    for label, kw in (("S0", {"stage": 0}), ("S1", {"stage": 1}), ("S1b/c", {"stage": 1, "vx_back_range": (0.03, 0.06)}),
                      ("S1d", {"stage": 1, "vx_range": (0.03, 0.06), "vx_back_range": (0.03, 0.06)})):
        outs = []
        for mod in (old, new):
            torch.manual_seed(7); env = _Env(512)
            mod.install_w3_cmd(env, None, p_step=0.7, **kw)
            seq = []
            for i in range(40):
                ids = torch.randperm(512)[: 1 + (i * 37) % 300]
                env.command_manager.t._resample_command(ids)
                seq.append((env.command_manager.t.vel_command_b.clone(), env._ge_cmd.clone()))
            outs.append(seq)
        ok = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(*outs))
        check(ok, f"install_w3_cmd {label}: 40 회 리샘플 명령·ge_cmd 비트 동일")
    # w3_gait_update: 무작위 ge 명령·리셋·vy ≠ 0 명령(새 기본값은 무시해야)
    outs = []
    for mod in (old, new):
        torch.manual_seed(3); env = _Env(256)
        env.command_manager.t.vel_command_b[:, 1] = torch.empty(256).uniform_(-0.08, 0.08)
        tr = []
        for t in range(400):
            env.episode_length_buf += 1
            rs = torch.rand(256) < 0.01; env.episode_length_buf[rs] = 1
            env._ge_cmd = torch.rand(256) < (0.7 if (t // 37) % 2 == 0 else 0.2)
            mod.w3_gait_update(env, None, f_step=1.1)
            tr.append(torch.cat([env._gu, env._gf, env._ge.float(), env._ge_obs]))
        outs.append(torch.stack(tr))
    check(torch.equal(outs[0], outs[1]), "w3_gait_update(기본 lead 0): 400 틱 위상·주파수·ge·ge_obs 비트 동일(vy ≠ 0 명령 포함)")
    # tracker + stride(구) vs stride2(신, vy 0): 같은 합성 궤적
    for vx in (0.0, 0.06, -0.04):
        for foot in (0, 1):
            cmd = torch.tensor([vx, 0.0, 0.0])
            dpos = torch.stack([torch.linspace(-0.08, 0.08, 64), torch.linspace(-0.03, 0.03, 64)], 1)
            yaw = torch.linspace(-math.pi, math.pi, 64)
            r_old, _, _, _, eo = _scenario(old, 64, cmd, foot, dpos, yaw=yaw, use_new=False)
            r_o2, r_new, _, _, en = _scenario(new, 64, cmd, foot, dpos, yaw=yaw, use_new=True)
            same_state = all(torch.equal(eo._w3[k], en._w3[k]) for k in eo._w3)
            check(torch.equal(r_old, r_new) and torch.equal(r_old, r_o2) and same_state,
                  f"vx {vx:+.2f}·발 {foot}: 구 w3_stride = 신 w3_stride2(vy 0) 비트 동일, 추적기 상태 동일 (합 {float(r_old.sum()):+.4f})")


# ------------------------------------------------------------------ [B] 반사실 보상
def test_B(new):
    print("[B] 반사실 보상 — 착지 사건당 w3_stride2(가중 전 raw), 폭 벌점 raw")
    rows = []
    for vy in (0.03, 0.05, 0.08, -0.03, -0.05, -0.08):
        D = vy / 1.1; sgn = 1.0 if vy > 0 else -1.0
        lead_foot = 0 if vy > 0 else 1                       # 가는 쪽 발(바깥으로 나가는 발 = 폭 증가)
        cmd = torch.tensor([0.0, vy, 0.0])
        cases = {"같은 방향(리드 발)": (lead_foot, (0.0, D), None, True),
                 "같은 방향(따라오는 발, 벌어진 자세에서)": (1 - lead_foot, (0.0, D), None, True),
                 "좁히는 스텝(명목 자세에서 따라오는 발)": (1 - lead_foot, (0.0, D), None, True),
                 "제자리 디딤": (lead_foot, (0.0, 0.0), None, True),
                 "반대 방향": (lead_foot, (0.0, -D), None, True),
                 "기울임만(골반 +y·착지 없음)": (lead_foot, (0.0, 0.0), (0.0, D), False),
                 "같은 방향 2배": (lead_foot, (0.0, 2 * D), None, True),
                 "같은 방향 + 앞 3 cm": (lead_foot, (0.03, D), None, True)}
        res = {}
        for name, (foot, dp, lean, swing) in cases.items():
            yaw = torch.tensor([0.0, math.pi, 0.7])           # 요 0·180°(스폰 실제)·40°
            dpos = torch.tensor(dp).repeat(3, 1); ln = torch.tensor(lean).repeat(3, 1) if lean else None
            oo = torch.tensor([0.0, D]).repeat(3, 1) if name.startswith("같은 방향(따라오는") else None
            _, r2, pw, tddy, _ = _scenario(new, 3, cmd, foot, dpos, lean=ln, yaw=yaw, swing=swing, other_off=oo)
            res[name] = (r2, pw, tddy)
            rows.append((vy, name, float(r2[0]), float(r2[1]), float(r2[2]), float(pw[0]), float(tddy[0])))
        same = res["같은 방향(리드 발)"][0]; zero = res["제자리 디딤"][0]; opp = res["반대 방향"][0]; lean = res["기울임만(골반 +y·착지 없음)"][0]
        trail = res["같은 방향(따라오는 발, 벌어진 자세에서)"][0]; narrow = res["좁히는 스텝(명목 자세에서 따라오는 발)"][0]
        check(bool((same > 0.4).all()) and bool((zero.abs() < 1e-6).all()) and bool((opp < -0.3).all()),
              f"vy {vy:+.2f}: 같은 {float(same[0]):+.3f} ≫ 제자리 {float(zero[0]):+.3f} ≫ 반대 {float(opp[0]):+.3f} (요 0/180/40° 모두)")
        check(bool((lean.abs() < 1e-9).all()), f"vy {vy:+.2f}: 기울임만 = 0 ({float(lean[0]):+.4f})")
        check(bool(((same - same[0]).abs() < 1e-5).all()), f"vy {vy:+.2f}: 요 180°·40° 스폰에서도 같은 점수(요 프레임)")
        check(bool((res["같은 방향(리드 발)"][1] == 0).all()), f"vy {vy:+.2f}: 정상 리드 스텝 폭 벌점 0")
        check(bool((trail > 0.4).all()) and bool((res["같은 방향(따라오는 발, 벌어진 자세에서)"][1] == 0).all()),
              f"vy {vy:+.2f}: 정상 옆걸음 따라오는 스텝(폭 {2*0.144+abs(D):.3f} → 0.288 m) 점수 {float(trail[0]):+.3f}, 폭 벌점 0")
        # 명목 자세에서 안쪽으로 좁히면 폭 0.288 − |D|: |D| > 0.05(vy > 0.055) 면 문턱 0.238 아래 → 0 점 + 벌점
        exp_bad = (2 * 0.144 - abs(D)) < (2 * 0.144 - 0.05)
        check(bool((narrow > 0.4).all()) != exp_bad, f"vy {vy:+.2f}: 좁히는 스텝(폭 {2*0.144-abs(D):.3f} m) 점수 {float(narrow[0]):+.3f}"
              + (" — 폭 위반으로 0" if exp_bad else " — 문턱 안, 정상"))
    # 폭 위반·교차(따라오는 발이 지나치게 안쪽으로): vy +0.05, 오른발이 +y 로 6 cm / 32 cm
    cmd = torch.tensor([0.0, 0.05, 0.0])
    for name, dy, exp_pen in (("폭 위반(오른발 안쪽 6 cm → W 0.228)", 0.06, 1.0), ("교차(오른발 +32 cm → W −0.032)", 0.32, 2.0)):
        _, r2, pw, _, _ = _scenario(new, 1, cmd, 1, torch.tensor([[0.0, dy]]))
        rows.append((0.05, name, float(r2[0]), float("nan"), float("nan"), float(pw[0]), dy))
        check(float(r2[0]) == 0.0 and abs(float(pw[0]) - exp_pen) < 1e-6, f"{name}: stride2 {float(r2[0]):+.3f} = 0, 폭 벌점 raw {float(pw[0]):.1f}")
    # 제자리 디딤(vy 0) 폭 벌점 0
    _, _, pw, _, _ = _scenario(new, 1, torch.tensor([0.0, 0.0, 0.0]), 0, torch.tensor([[0.0, 0.0]]))
    check(float(pw[0]) == 0.0, "명령 0 제자리 디딤(W 0.288): 폭 벌점 0")
    print("\n  | vy | 경우 | stride2 raw(요 0) | 요 180° | 요 40° | 폭 벌점 raw | 착지 Δy [m] |")
    print("  |---|---|---|---|---|---|---|")
    for r in rows:
        print(f"  | {r[0]:+.2f} | {r[1]} | {r[2]:+.3f} | {r[3]:+.3f} | {r[4]:+.3f} | {r[5]:.1f} | {r[6]:+.4f} |")
    print()


# ------------------------------------------------------------------ [C] 리드 발
def test_C(new):
    print("[C] 리드 발 위상")
    env = _Env(4)
    env.command_manager.t.vel_command_b[:, 1] = torch.tensor([0.05, -0.05, 0.0, -0.01])
    env._ge_cmd = torch.ones(4, dtype=torch.bool); env.episode_length_buf[:] = 1
    new.w3_gait_update(env, None, f_step=1.1, lead_vy_cmin=0.02)
    check(env._gu.tolist() == [0.0, 0.5, 0.0, 0.0], f"에피소드 시작: vy +0.05→0 · −0.05→0.5 · 0→0 · −0.01(문턱 아래)→0 (실제 {env._gu.tolist()})")
    env.episode_length_buf[:] = 5; env._ge_cmd[:] = False
    for _ in range(40):
        env.episode_length_buf += 1; new.w3_gait_update(env, None, f_step=1.1, lead_vy_cmin=0.02)
    check(not bool(env._ge.any()), "끄기 요청 → 반주기 경계에서 ge 0")
    env._ge_cmd[:] = True; env.episode_length_buf += 1
    new.w3_gait_update(env, None, f_step=1.1, lead_vy_cmin=0.02)
    check(env._gu.tolist() == [0.0, 0.5, 0.0, 0.0], f"중간 0→1 켜기도 같은 규칙 (실제 {env._gu.tolist()})")
    u1 = env._gu.clone(); env.episode_length_buf += 1
    new.w3_gait_update(env, None, f_step=1.1, lead_vy_cmin=0.02)
    check(torch.allclose(env._gu - u1, torch.full((4,), 1.1 * DT)), "켠 뒤엔 정상 누적(리드 재설정 없음)")


# ------------------------------------------------------------------ [D] 명령 분포
def test_D(new):
    print("[D] Stage 2 명령 분포")
    N = 200000; torch.manual_seed(11); env = _Env(N)
    new.install_w3_cmd(env, None, p_step=0.7, stage=2, vx_back_range=(0.03, 0.06), mix2=(0.3, 0.3, 0.4), vy_range=(0.03, 0.04))
    env.command_manager.t._resample_command(torch.arange(N))
    c = env.command_manager.t.vel_command_b; ge = env._ge_cmd
    st = ge.float().mean(); g = c[ge]
    inpl = ((g[:, 0] == 0) & (g[:, 1] == 0)).float().mean(); fb = (g[:, 0] != 0).float().mean(); lat = (g[:, 1] != 0).float().mean()
    vy = g[g[:, 1] != 0, 1]
    check(abs(float(st) - 0.7) < 0.01, f"STEP 비율 {float(st):.3f} ≈ 0.7")
    check(abs(float(inpl) - 0.3) < 0.01 and abs(float(fb) - 0.3) < 0.01 and abs(float(lat) - 0.4) < 0.01,
          f"STEP 중 제자리 {float(inpl):.3f} · 앞뒤 {float(fb):.3f} · 옆 {float(lat):.3f}")
    check(bool(((vy.abs() >= 0.03) & (vy.abs() <= 0.04)).all()) and abs(float((vy > 0).float().mean()) - 0.5) < 0.01,
          f"|vy| ∈ [0.03, 0.04], 양수 비율 {float((vy > 0).float().mean()):.3f}")
    check(bool((c[~ge] == 0).all()) and bool((g[:, 2] == 0).all()) and bool(((g[:, 0] != 0) & (g[:, 1] != 0)).sum() == 0),
          "STAND 명령 0 · wz 0 · 대각(vx·vy 동시) 없음")
    env._w3_vy_range = (0.03, 0.08); env.command_manager.t._resample_command(torch.arange(N))
    vy = env.command_manager.t.vel_command_b[:, 1]; vy = vy[vy != 0]
    check(float(vy.abs().max()) > 0.075, "커리큘럼 상한(env._w3_vy_range) 변경이 다음 리샘플에 반영")


# ------------------------------------------------------------------ [E] 커리큘럼
def test_E(new):
    print("[E] vy 커리큘럼")
    env = _Env(8); L = new._lat(env)
    seq = []
    for it, (sign, bad) in enumerate([(0.85, 0.0), (0.95, 0.03), (0.95, 0.01), (0.95, 0.01), (0.95, 0.01), (0.95, 0.01)]):
        env.common_step_counter = (it + 1) * 120 * 24
        L["n_lat_td"] += 4000; L["n_lat_ok"] += 4000 * sign; L["n_td"] += 10000; L["n_bad"] += 10000 * bad
        out = new.w3_vy_curriculum(env, None)
        seq.append(out["vy_max"])
    check(seq == [0.04, 0.04, 0.06, 0.08, 0.08, 0.08], f"상한 순서 {seq} (부호 0.85 정체 · 폭 0.03 정체 · 이후 0.06 → 0.08, 최대 유지)")


# ------------------------------------------------------------------ [F] 명령 항 매 틱 후처리가 vy 를 덮지 않는가
ISAAC_VEL = "/home/ryu/IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py"


def test_F(new):
    """W3 태스크 명령 항 = UniformVelocityCommand 계열(heading_command=True·ang_vel_z (0,0) — 직접 넣은 wz 는 매 틱 0 으로 덮임, 측정 사실).
    그 _update_command(매 틱 후처리: heading → wz, 서기 env → 전 성분 0)를 Isaac Lab 소스 그대로(AST 추출) 돌려
    install_w3_cmd stage 2 가 넣은 vy 가 리샘플 사이 중간 시점에도 남는지 본다."""
    print("[F] 명령 항 후처리(heading·서기 0) 뒤 vy 보존")
    if not os.path.exists(ISAAC_VEL):
        check(False, f"Isaac Lab 소스 없음: {ISAAC_VEL}"); return
    tree = ast.parse(open(ISAAC_VEL).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UniformVelocityCommand")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_update_command")
    g = {"torch": torch, "math_utils": types.SimpleNamespace(wrap_to_pi=lambda x: torch.atan2(torch.sin(x), torch.cos(x)))}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), ISAAC_VEL, "exec"), g)
    upd = g["_update_command"]
    N = 4000; torch.manual_seed(5); env = _Env(N)
    t = env.command_manager.t
    t.cfg = types.SimpleNamespace(heading_command=True, heading_control_stiffness=0.5, ranges=types.SimpleNamespace(ang_vel_z=(0.0, 0.0)))
    t.is_heading_env = torch.ones(N, dtype=torch.bool); t.heading_target = torch.empty(N).uniform_(-3, 3)
    t.robot = types.SimpleNamespace(data=types.SimpleNamespace(heading_w=torch.empty(N).uniform_(-3, 3)))
    orig = t._resample_command

    def _orig_like_isaac(ids):                           # 원래 리샘플: 서기 env 를 무작위로 표시(rel_standing 0.5 가정 — 최악 쪽)
        orig(ids); t.is_standing_env[ids] = torch.rand(len(ids)) < 0.5
    t._resample_command = _orig_like_isaac
    new.install_w3_cmd(env, None, p_step=0.7, stage=2, vx_back_range=(0.03, 0.06), mix2=(0.3, 0.3, 0.4), vy_range=(0.03, 0.08))
    t._resample_command(torch.arange(N))
    vy0 = t.vel_command_b[:, 1].clone(); vx0 = t.vel_command_b[:, 0].clone()
    for k in range(50):                                  # 리샘플 사이 50 틱(1 s) 매 틱 후처리
        upd(t)
        if k == 24:
            vy_mid = t.vel_command_b[:, 1].clone()
    lat = vy0 != 0
    check(int(lat.sum()) > 0 and torch.equal(vy_mid, vy0) and torch.equal(t.vel_command_b[:, 1], vy0) and torch.equal(t.vel_command_b[:, 0], vx0),
          f"옆 명령 {int(lat.sum())} env: 중간(25 틱)·끝(50 틱) vel_command_b[:,1] = 리샘플 값 그대로(vx 도), 서기 표시 0 개(install 이 덮음: {int(t.is_standing_env.sum())})")
    check(bool((t.vel_command_b[:, 2] == 0).all()), "wz 는 heading·(0,0) 범위로 0(측정 사실과 일치 — Stage 2 는 wz 미사용)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="6a91c303")
    a = ap.parse_args()
    src_old = subprocess.run(["git", "-C", REPO, "show", f"{a.ref}:rl_walking/w3.py"], check=True, capture_output=True, text=True).stdout
    src_new = open(os.path.join(REPO, "rl_walking", "w3.py")).read()
    old = _load(src_old, "old"); old.__src__ = src_old
    new = _load(src_new, "new"); new.__src__ = src_new
    test_A(old, new); test_B(new); test_C(new); test_D(new); test_E(new); test_F(new)
    print(f"\n결과: 실패 {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
