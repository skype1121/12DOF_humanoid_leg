"""W3D(걷는 중 외란 → 회복 스텝 → WALK/STAND 복귀) CPU 반사실 단위시험 (1007) — torch CPU 만(Isaac·GPU 불필요).

  python3 rl_walking/scripts/unit_w3d.py

rl_walking/w3.py·w3_c2.py·w3_dist.py 를 스텁 mdp_hang(외란 힘 함수 _dist_profile/_dist_force_w 는 실제 소스를 AST 로 떼어 씀)과 함께 올린다.
  [A] 무외란 불변: w3d_gait_update ≡ w3_gait_update(정지 대상 없음, 무작위 ge_cmd 400 틱) · 같은 스윙 궤적에서 W3D 이벤트를 함께 돌려도
      W3 상태(_w3·_w3lat)·기존 보상 값 같고 새 보상 5종 = 0 · p_dist 0 리셋 = 난수 소비 0 · 발별 stride2 합 ≡ w3_stride2(앞·뒤·옆·0 명령).
  [B] 반사실(같은 충격·같은 몸 운동, 착지 위치만 다름): 사건 보상 합(가중 곱, W3S3d 가중 + W3D 새 항)
      방향 회복 스텝 ≫ 원래 보폭 ≫ 반대 — 앞(왼발 스윙)·뒤·왼쪽, 걷기 명령 vx 0.06 / 제자리 0. 얼어붙기(스윙 안 함) < 원래 보폭.
  [C] 숨기·이른 정지·파밍 점검: 외란 없이 일부러 ICP 를 밀어낸 착지 = 새 항 0, 회복 창 밖(시간·착지 수 초과) 착지 = 0,
      정지 시나리오에서 명령 리샘플이 ge_cmd 를 다시 켜도 엔진 전에 덮임, 외란이 STAND 에서 발생하면 취소·정지 명령 없음.
  [D] 외란 스케줄 분포(비율·방향·충격/램프·크기 단계·간격·정지 비율). [E] 크기 커리큘럼 진급 규칙.
종료 코드 0 = 전부 통과.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DT = 0.02
F_STEP = 1.1
FAILS = []
W_HALF = 0.144
MASS = (0.4, 0.4, 16.8)
WEIGHTS = {"w3_valid": 15.0, "w3_place": 10.0, "w3_stride": 10.0, "w3_miss": -10.0, "w3_wrong": -5.0, "w3_periodic": -1.5, "w3_height": 3.0, "w3_width": -10.0,
           "w3d_cap": 10.0, "w3d_rplace": 10.0, "w3d_rstride": 10.0, "w3d_ret": 20.0, "w3d_sret": 20.0}


def check(cond, msg):
    print(("  [통과] " if cond else "  [실패] ") + msg)
    if not cond:
        FAILS.append(msg)


# ------------------------------------------------------------------ 모듈 적재
def _stub_h():
    H = types.ModuleType("w3dstub.mdp_hang")
    H.G = 9.81; H._DST_K = 3
    H._base_cfg = lambda: types.SimpleNamespace(name="robot")
    H._g21_bids = lambda env: [0, 1]
    H._feet_body_ids = lambda env: [0, 1]
    H._resolve_nested = lambda env, p: dict(p or {})
    src = open(os.path.join(REPO, "rl_walking", "mdp_hang.py")).read()
    tree = ast.parse(src)
    code = "import torch\n" + "\n\n".join(ast.get_source_segment(src, n) for n in tree.body
                                          if isinstance(n, ast.FunctionDef) and n.name in ("_dist_profile", "_dist_force_w"))
    exec(code, H.__dict__)

    def _state(env):
        st = getattr(env, "_hang", None)
        if st is None:
            N, K = env.num_envs, 3
            z = lambda *sh: torch.zeros(*sh)
            st = {"t": z(N), "dst_kind": z(N, K), "dst_t0": z(N, K), "dst_rise": torch.ones(N, K), "dst_hold": z(N, K), "dst_F": z(N, K),
                  "dst_dir": z(N, K, 2), "dst_F_now": z(N), "dst_last_end": torch.full((N,), -1.0), "dst_pending": z(N), "dst_quiet_run": z(N), "dst_bonus": z(N)}
            env._hang = st
        return st
    H._state = _state
    return H


def _load():
    pkg = "w3dstub"
    m = types.ModuleType(pkg); m.__path__ = []; sys.modules[pkg] = m
    h = _stub_h(); sys.modules[f"{pkg}.mdp_hang"] = h; m.mdp_hang = h
    mods = {}
    for name in ("w3", "w3_c2", "w3_dist", "w3_cur2"):
        spec = importlib.util.spec_from_file_location(f"{pkg}.{name}", os.path.join(REPO, "rl_walking", f"{name}.py"))
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; setattr(m, name, mod)
        spec.loader.exec_module(mod); mods[name] = mod
    return h, mods["w3"], mods["w3_c2"], mods["w3_dist"], mods["w3_cur2"]


H, W3, C2, WD, WC2 = _load()


# ------------------------------------------------------------------ 가짜 env
def qz(a):
    return torch.stack([torch.cos(a / 2), torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2)], -1)


class _Term:
    def __init__(self, N):
        self.vel_command_b = torch.zeros(N, 3); self.is_standing_env = torch.zeros(N, dtype=torch.bool)


class _CM:
    def __init__(self, N):
        self.t = _Term(N)

    def get_term(self, name):
        return self.t

    def get_command(self, name):
        return self.t.vel_command_b


class _Env:
    def __init__(self, N):
        self.num_envs = N; self.device = torch.device("cpu"); self.step_dt = DT
        self.episode_length_buf = torch.zeros(N, dtype=torch.long); self.common_step_counter = 0
        self.command_manager = _CM(N); self._ge_obs = torch.zeros(N)
        self.data = types.SimpleNamespace(root_quat_w=qz(torch.zeros(N)), root_pos_w=torch.zeros(N, 3), body_pos_w=torch.zeros(N, 3, 3),
                                          default_mass=torch.tensor(MASS).repeat(N, 1), root_lin_vel_w=torch.zeros(N, 3), root_ang_vel_b=torch.zeros(N, 3),
                                          projected_gravity_b=torch.tensor([0.0, 0.0, -1.0]).repeat(N, 1),
                                          body_lin_vel_w=torch.zeros(N, 3, 3), body_com_pos_w=torch.zeros(N, 3, 3), body_com_lin_vel_w=torch.zeros(N, 3, 3))
        self.sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=torch.zeros(N, 2, 3)))
        self.scene = types.SimpleNamespace(sensors={"contact_forces": self.sensor}, __getitem__=None)
        self.scene = _Scene(self)
        self.termination_manager = types.SimpleNamespace(terminated=torch.zeros(N, dtype=torch.bool))


class _Scene:
    def __init__(self, env):
        self.env = env; self.sensors = {"contact_forces": env.sensor}

    def __getitem__(self, k):
        return self.env


def _set_body(env, foot_xyz, com_xy, com_v, com_z=0.55):
    """발(N,2,3)·CoM(몸통 질량점 위치·속도로 CoM 이 com_xy 가 되도록) 설정. 골반 root = CoM xy."""
    N = env.num_envs
    bp = torch.zeros(N, 3, 3); bp[:, :2] = foot_xyz
    m = torch.tensor(MASS); mt = m.sum()
    feet_m = (foot_xyz[..., :2] * m[:2, None]).sum(1)
    torso_xy = (com_xy * mt - feet_m) / m[2]
    bp[:, 2, :2] = torso_xy; bp[:, 2, 2] = (com_z * mt - (foot_xyz[..., 2] * m[:2]).sum(1)) / m[2]
    env.data.body_pos_w = bp; env.data.body_com_pos_w = bp.clone()
    bv = torch.zeros(N, 3, 3); bv[:, 2, :2] = com_v * mt / m[2]
    env.data.body_com_lin_vel_w = bv; env.data.body_lin_vel_w = torch.zeros(N, 3, 3)
    env.data.root_pos_w = torch.cat([com_xy, torch.full((N, 1), com_z)], 1)
    env.data.root_lin_vel_w = torch.cat([com_v, torch.zeros(N, 1)], 1)


def _rewards(env, out):
    vals = {"w3_valid": W3.w3_value(env, "ev_valid"), "w3_place": W3.w3_place_reward(env), "w3_stride": W3.w3_stride2(env),
            "w3_miss": W3.w3_value(env, "ev_miss"), "w3_wrong": W3.w3_value(env, "ev_wrong"), "w3_periodic": W3.w3_periodic_contact(env),
            "w3_height": W3.w3_swing_height(env), "w3_width": W3.w3_width_pen(env)}
    if out.get("wd", True):
        vals.update({"w3d_cap": WD.w3d_cap(env), "w3d_rplace": WD.w3d_relief_place(env), "w3d_rstride": WD.w3d_relief_stride(env),
                     "w3d_ret": WD.w3d_value(env, "ev_ret"), "w3d_sret": WD.w3d_value(env, "ev_sret")})
    for k, v in vals.items():
        out.setdefault(k, torch.zeros(env.num_envs)); out[k] = out[k] + v * WEIGHTS[k] * DT


# ------------------------------------------------------------------ 한 번 스윙 시나리오(왼발, 첫 반주기)
LIFT, LAND, TAIL = 4, 18, 12


def swing(cmd, land_off, push=None, xi_off=None, wd=True, freeze=False, ticks=None, t0=0.02, sched=True, win_s=1.2, max_td=3):
    """왼발(0) LIFT 틱 이륙 → LAND 틱 착지(명목 착지 + land_off(N,2)), 오른발 고정. 몸: 골반·CoM 이 명령 속도로 진행.
    push = (N,2) 단위 방향(월드, 몸통 요 0)·충격 18 N 0.3 s(t0) — 회복 창 무장용. xi_off(N,2) = 착지 순간 ICP − CoM(동역학 발산 결과를 직접 지정).
    freeze = 스윙 없음(양발 고정). 반환 = 항별 가중 합(dict), env."""
    N = land_off.shape[0]
    env = _Env(N)
    env.command_manager.t.vel_command_b[:] = torch.tensor(cmd)
    env._ge_cmd = torch.ones(N, dtype=torch.bool)
    hs = H._state(env)
    if push is not None and sched:
        hs["dst_kind"][:, 0] = 2.0; hs["dst_t0"][:, 0] = t0; hs["dst_hold"][:, 0] = 0.3; hs["dst_F"][:, 0] = 18.0
        WD._st(env)["dir_b"][:, 0] = push
    else:
        WD._st(env)
    vx = cmd[0]; tx = vx / F_STEP
    tgt_land = torch.tensor([vx / (4 * F_STEP), W_HALF])                     # 착지 순간 골반 기준 명목 목표
    t_land = LAND * DT
    pel_land = torch.tensor([vx * t_land, 0.0])
    P_land = (pel_land + tgt_land).expand(N, 2) + land_off
    P_lift = (pel_land + tgt_land - torch.tensor([tx, 0.0])).expand(N, 2)
    P_R = torch.tensor([vx * t_land * 0.5, -W_HALF]).expand(N, 2)
    xi_off = torch.zeros(N, 2) if xi_off is None else xi_off
    out = {"wd": wd}
    mg = sum(MASS) * 9.81
    T = ticks or (LAND + TAIL)
    for k in range(T):
        env.episode_length_buf += 1; env.common_step_counter += 1
        hs["t"] += DT
        tt = hs["t"].unsqueeze(1)
        prof = ((hs["dst_kind"] == 2) & (tt >= hs["dst_t0"]) & (tt < hs["dst_t0"] + hs["dst_hold"])).float()
        hs["dst_F_now"] = ((hs["dst_F"] * prof).unsqueeze(-1) * hs["dst_dir"]).sum(1).norm(dim=-1)
        s = min(1.0, max(0.0, (k - LIFT) / (LAND - LIFT))) if not freeze else 0.0
        fx = torch.zeros(N, 2, 3); fx[:, 0, :2] = P_lift + s * (P_land - P_lift); fx[:, 1, :2] = P_R
        F = torch.zeros(N, 2, 3); F[:, :, 2] = mg / 2
        if not freeze and LIFT <= k < LAND:
            F[:, 0, 2] = 0.0; F[:, 1, 2] = mg; fx[:, 0, 2] = 0.05 * math.sin(math.pi * s)
        elif not freeze and k >= LAND:
            F[:, 0, 2] = 0.6 * mg; F[:, 1, 2] = 0.4 * mg
        pel = torch.tensor([vx * min(k, LAND) * DT, 0.0]).expand(N, 2)
        om = math.sqrt(9.81 / 0.55)
        com_v = torch.tensor([vx, 0.0]).expand(N, 2) + (xi_off * om if k >= LAND - 1 else torch.zeros(N, 2))
        _set_body(env, fx, pel.clone(), com_v.clone())
        env.sensor.data.net_forces_w = F
        if wd:
            WD.w3d_gait_update(env, None, f_step=F_STEP, lead_vy_cmin=0.02, lead_wz_cmin=0.08)
        else:
            W3.w3_gait_update(env, None, f_step=F_STEP, lead_vy_cmin=0.02, lead_wz_cmin=0.08)
        W3.w3_step_tracker(env, None)
        W3.w3_lat_tracker(env, None, nominal_half_width=W_HALF, width_margin=0.05)
        if wd:
            WD.w3d_dist_step(env, None, win_s=win_s, max_td=max_td)
            WD.w3d_tracker(env, None)
        _rewards(env, out)
    out.pop("wd"); out["env"] = env
    return out


def total(o, keys=None):
    return sum(v for k, v in o.items() if k != "env" and (keys is None or k in keys))


# ------------------------------------------------------------------ [A]
def test_A():
    print("[A] 무외란 불변")
    torch.manual_seed(3)
    N = 64
    e1, e2 = _Env(N), _Env(N)
    for e in (e1, e2):
        e._ge_cmd = torch.zeros(N, dtype=torch.bool); e.command_manager.t.vel_command_b[:, 1] = torch.linspace(-0.08, 0.08, N)
    WD._st(e2)
    ok = True
    for k in range(400):
        want = torch.rand(N) < (0.5 + 0.4 * math.sin(k / 30))
        for e in (e1, e2):
            e.episode_length_buf += 1; e.common_step_counter += 1; e._ge_cmd = want.clone()
        W3.w3_gait_update(e1, None, f_step=F_STEP, lead_vy_cmin=0.02, lead_wz_cmin=0.08)
        WD.w3d_gait_update(e2, None, f_step=F_STEP, lead_vy_cmin=0.02, lead_wz_cmin=0.08)
        ok &= torch.equal(e1._ge, e2._ge) and torch.equal(e1._gu, e2._gu) and torch.equal(e1._ge_obs, e2._ge_obs)
    check(ok, "w3d_gait_update ≡ w3_gait_update (정지 대상 없음, 400 틱 ge·위상·관측 램프 비트 동일)")
    for cmd in ((0.06, 0.0, 0.0), (-0.04, 0.0, 0.0), (0.0, 0.05, 0.0), (0.0, -0.05, 0.0), (0.0, 0.0, 0.0)):
        off = torch.randn(16, 2) * 0.03
        a = swing(cmd, off, wd=False); b = swing(cmd, off, wd=True)
        sa = all(torch.equal(a["env"]._w3[k], b["env"]._w3[k]) for k in a["env"]._w3) and all(torch.equal(a["env"]._w3lat[k], b["env"]._w3lat[k]) for k in a["env"]._w3lat)
        eq = all(torch.equal(a[k], b[k]) for k in a if k != "env")
        z = all(float(b[k].abs().max()) == 0.0 for k in ("w3d_cap", "w3d_rplace", "w3d_rstride", "w3d_ret", "w3d_sret"))
        check(sa and eq and z, f"명령 {cmd}: W3D 이벤트 동반 시 W3 상태·기존 보상 비트 동일, 새 항 5종 = 0 (기존 합 {float(total(b)[0]):+.3f})")
        env = b["env"]; r, smax = WD.stride2_per_foot(env)
        check(torch.equal(r.sum(1) * W3._ge(env).float(), W3.w3_stride2(env)) or torch.allclose(r.sum(1) * W3._ge(env).float(), W3.w3_stride2(env), atol=0, rtol=0),
              f"명령 {cmd}: 발별 stride2 합 ≡ w3_stride2 (마지막 틱)")
    # 발별 합 ≡ w3_stride2 를 착지 틱에서
    for cmd in ((0.06, 0.0, 0.0), (0.0, 0.05, 0.0), (-0.04, 0.0, 0.0)):
        env = _Env(8)
        o = swing(cmd, torch.randn(8, 2) * 0.03, ticks=LAND + 1)
        env = o["env"]; r, _ = WD.stride2_per_foot(env)
        d = (r.sum(1) * W3._ge(env).float() - W3.w3_stride2(env)).abs().max()
        check(float(d) < 1e-6 and float(env._w3["ev_td"].sum()) > 0, f"명령 {cmd} 착지 틱: Σ발 stride2_per_foot = w3_stride2 (최대 차 {float(d):.1e})")
    env = _Env(32); WD._st(env)
    st0 = torch.random.get_rng_state()
    WD.w3d_dist_reset(env, torch.arange(32), p_dist=0.0)
    check(torch.equal(st0, torch.random.get_rng_state()) and float(H._state(env)["dst_kind"].abs().sum()) == 0.0, "p_dist 0 리셋: 난수 소비 0·스케줄 없음")


# ------------------------------------------------------------------ [B]
def test_B():
    print("[B] 반사실 회복 착지(같은 충격·같은 몸 운동) — 사건 보상 합(가중×dt)")
    N = 1
    rows = []
    for cmd in ((0.06, 0.0, 0.0), (0.0, 0.0, 0.0)):
        for name, d in (("앞", (1.0, 0.0)), ("뒤", (-1.0, 0.0)), ("왼", (0.0, 1.0))):
            dv = torch.tensor([d]); X = 0.13 if name != "왼" else 0.22
            xi = dv * X
            res = {}
            for lab, off in (("방향 회복", dv * 0.08), ("원래 보폭", torch.zeros(1, 2)), ("반대", -dv * 0.06)):
                o = swing(cmd, off, push=dv, xi_off=xi)
                res[lab] = (float(total(o)[0]), float(o["w3d_cap"][0]), float(o["env"]._w3d["phi_b"].abs().sum()))
            fz = swing(cmd, torch.zeros(1, 2), push=dv, xi_off=xi, freeze=True, ticks=LAND + TAIL)
            res["얼어붙기"] = (float(total(fz)[0]), 0.0, 0.0)
            rows.append((cmd, name, res))
            a, b_, c = res["방향 회복"][0], res["원래 보폭"][0], res["반대"][0]
            print(f"    명령 vx {cmd[0]:+.2f} 밀기 {name}: 방향 {a:+.3f}(cap {res['방향 회복'][1]:.3f}) · 원래 {b_:+.3f}(cap {res['원래 보폭'][1]:.3f}) · 반대 {c:+.3f} · 얼어붙기 {res['얼어붙기'][0]:+.3f}")
            check(a > b_ + 0.03 and b_ > c + 0.03, f"vx {cmd[0]:+.2f}·{name}: 방향 회복 > 원래 보폭 > 반대 (차 ≥ 0.03 = 착지 사건 가중 1.5 상당)")
            check(res["얼어붙기"][0] < b_, f"vx {cmd[0]:+.2f}·{name}: 얼어붙기 < 원래 보폭(숨기 불리)")
    # 손해 보전 확인: 방향 회복 착지의 기존 place+stride 손실이 보전되는가
    dv = torch.tensor([[1.0, 0.0]])
    o_dir = swing((0.06, 0.0, 0.0), dv * 0.08, push=dv, xi_off=dv * 0.13)
    o_nom = swing((0.06, 0.0, 0.0), torch.zeros(1, 2), push=dv, xi_off=dv * 0.13)
    base = ("w3_place", "w3_stride")
    print(f"    앞 vx 0.06: 기존 place+stride 방향 {float(total(o_dir, base)[0]):+.3f} vs 원래 {float(total(o_nom, base)[0]):+.3f} · 보전 포함 "
          f"{float(total(o_dir, base + ('w3d_rplace', 'w3d_rstride'))[0]):+.3f} vs {float(total(o_nom, base + ('w3d_rplace', 'w3d_rstride'))[0]):+.3f}")
    check(float(total(o_dir, base + ("w3d_rplace", "w3d_rstride"))[0]) >= float(total(o_nom, base + ("w3d_rplace", "w3d_rstride"))[0]) - 0.02,
          "보전 포함 착지·보폭 점수: 방향 회복 ≥ 원래 보폭 − 0.02 (회복 쪽이 손해 보지 않음)")
    o_wd = swing((0.06, 0.0, 0.0), dv * 0.08, push=dv, xi_off=dv * 0.13)
    keys_old = ("w3_valid", "w3_place", "w3_stride", "w3_miss", "w3_wrong", "w3_periodic", "w3_height", "w3_width")
    print(f"    참고: 새 항 없는 W3S3d 채점 — 방향 {float(total(o_dir, keys_old)[0]):+.3f} vs 원래 {float(total(o_nom, keys_old)[0]):+.3f} (기존 보상은 회복 스텝을 벌한다 = W3D 필요 근거)")
    return rows


# ------------------------------------------------------------------ [C]
def test_C():
    print("[C] 숨기·이른 정지·파밍 점검")
    dv = torch.tensor([[1.0, 0.0]])
    o = swing((0.06, 0.0, 0.0), dv * 0.08, push=dv, xi_off=dv * 0.13, sched=False)
    check(all(float(o[k].abs().max()) == 0 for k in ("w3d_cap", "w3d_rplace", "w3d_rstride")), "외란 없이 ICP 를 밀어낸 착지(자기 불안정화) = 새 항 0")
    o = swing((0.06, 0.0, 0.0), dv * 0.08, push=dv, xi_off=dv * 0.13, win_s=0.2)
    check(all(float(o[k].abs().max()) == 0 for k in ("w3d_cap", "w3d_rplace", "w3d_rstride")), "회복 창(0.2 s) 밖 착지 = 새 항 0")
    o = swing((0.06, 0.0, 0.0), dv * 0.08, push=dv, xi_off=dv * 0.13, max_td=0)
    check(all(float(o[k].abs().max()) == 0 for k in ("w3d_cap", "w3d_rplace", "w3d_rstride")), "회복 착지 수 상한 초과 = 새 항 0")
    # 정지 시나리오: 리샘플이 ge_cmd 를 켜도 덮임
    N = 8; env = _Env(N); env._ge_cmd = torch.ones(N, dtype=torch.bool); env.command_manager.t.vel_command_b[:, 0] = 0.06
    s = WD._st(env); hs = H._state(env)
    s["stop_want"][:4] = True; s["t_stop"][:4] = 0.5; s["applied"][:4, 0] = True
    s["applied"][4:6, 0] = False; s["stop_want"][4:6] = True; s["t_stop"][4:6] = 0.5
    ge_hist = []
    for k in range(120):
        env.episode_length_buf += 1; env.common_step_counter += 1; hs["t"] += DT
        env._ge_cmd[:] = True; env.command_manager.t.vel_command_b[:, 0] = 0.06          # 리샘플이 매 틱 다시 켜는 최악
        WD.w3d_gait_update(env, None, f_step=F_STEP)
        ge_hist.append(env._ge.clone())
    g = torch.stack(ge_hist)
    late = g[60:]
    check(not bool(late[:, :4].any()) and bool(late[:, 4:].all()), "정지 시각 뒤 ge 0 유지(리샘플 덮기 포함, 반주기 래치 뒤) · 외란이 실제로 안 걸린 env 는 정지 명령 없음")
    check(float(env.command_manager.t.vel_command_b[:4].abs().sum()) == 0.0, "정지 env 명령 0")
    # STAND 에서 발생 → 취소
    env = _Env(4); env._ge = torch.tensor([True, False, True, False]); env._ge_cmd = env._ge.clone()
    s = WD._st(env); hs = H._state(env)
    hs["dst_kind"][:, 0] = 2.0; hs["dst_t0"][:, 0] = 0.1; hs["dst_hold"][:, 0] = 0.3; hs["dst_F"][:, 0] = 15.0; s["dir_b"][:, 0] = torch.tensor([1.0, 0.0])
    for k in range(10):
        env.episode_length_buf += 1; env.common_step_counter += 1; hs["t"] += DT
        WD.w3d_dist_step(env, None)
    check(s["applied"][:, 0].tolist() == [True, False, True, False] and hs["dst_kind"][:, 0].tolist() == [2.0, 0.0, 2.0, 0.0],
          "발생 직전 STAND env 의 외란 취소(걷는 env 만 적용)")
    check(bool(s["w_on"][0]) and not bool(s["w_on"][1]), "충격 적용 env 만 회복 창 무장")
    # 방향 = 발생 순간 몸통 요(요 90°)
    env = _Env(2); env._ge = torch.ones(2, dtype=torch.bool); env.data.root_quat_w = qz(torch.tensor([math.pi / 2, 0.0]))
    s = WD._st(env); hs = H._state(env)
    hs["dst_kind"][:, 0] = 2.0; hs["dst_t0"][:, 0] = 0.1; hs["dst_hold"][:, 0] = 0.3; hs["dst_F"][:, 0] = 15.0; s["dir_b"][:, 0] = torch.tensor([1.0, 0.0])
    for k in range(3):
        env.episode_length_buf += 1; hs["t"] += DT; WD.w3d_dist_step(env, None)
    d = hs["dst_dir"][:, 0]
    check(torch.allclose(d[0], torch.tensor([0.0, 1.0]), atol=1e-6) and torch.allclose(d[1], torch.tensor([1.0, 0.0]), atol=1e-6), "외란 방향 = 몸통 요 기준(앞 = 요 90° 에서 월드 +y)")


# ------------------------------------------------------------------ [D]
def test_D():
    print("[D] 외란 스케줄 분포")
    torch.manual_seed(11)
    N = 20000; env = _Env(N); s = WD._st(env); hs = H._state(env)
    WD.w3d_dist_reset(env, torch.arange(N))
    k = hs["dst_kind"]; act = (k > 0).any(1)
    st = s["stop_want"]
    n1 = ((k > 0).sum(1) == 1) & act; n2 = ((k > 0).sum(1) == 2)
    imp = (k == 2); rmp = (k == 1)
    Fi = hs["dst_F"][imp]; Fr = hs["dst_F"][rmp]; di = s["dir_i"][k > 0]
    frac = [float((di == j).float().mean()) for j in range(4)]
    gap_ok = bool(((hs["dst_t0"][:, 1] - (hs["dst_t0"][:, 0] + torch.where(k[:, 0] == 2, hs["dst_hold"][:, 0], hs["dst_rise"][:, 0] + hs["dst_hold"][:, 0])))[n2] >= 3.0 - 1e-5).all())
    print(f"    외란 에피소드 {float(act.float().mean()):.3f} · 1회/2회 {float(n1.float().mean()):.3f}/{float(n2.float().mean()):.3f} · 충격 비 {float(imp.sum() / (imp.sum() + rmp.sum())):.3f}"
          f" · 방향 f/b/l/r {frac[0]:.3f}/{frac[1]:.3f}/{frac[2]:.3f}/{frac[3]:.3f} · 충격 F {float(Fi.min()):.1f}–{float(Fi.max()):.1f} · 램프 F {float(Fr.min()):.1f}–{float(Fr.max()):.1f}"
          f" · 정지 {float(st.float().mean()):.3f}")
    check(abs(float(act.float().mean()) - 0.5) < 0.02 and abs(float(imp.sum() / (imp.sum() + rmp.sum())) - 0.7) < 0.02, "외란 에피소드 ≈ 50 %·충격 ≈ 70 %")
    check(max(abs(a - b) for a, b in zip(frac, (0.35, 0.25, 0.2, 0.2))) < 0.02, "방향 비 f .35·b .25·l .2·r .2")
    check(float(Fi.min()) >= 6.0 and float(Fi.max()) <= 10.0 and float(Fr.max()) <= 6.0 + 1e-6, "단계 0: 충격 6–10 N·램프 3–6 N")
    check(gap_ok and float(hs["dst_t0"][:, 0][act].min()) >= 3.0 and float(hs["dst_t0"][:, 0][act].max()) <= 15.0, "첫 외란 3–15 s·간격 ≥ 3 s")
    check(abs(float(st.float().mean()) - 0.15) < 0.015 and not bool((st & ~act).any()), "정지 시나리오 = 외란 에피소드의 30 %(전체 15 %), 무외란 env 정지 없음")
    ph = torch.remainder(hs["dst_t0"][:, 0][act] * F_STEP, 1.0)
    hist = torch.histc(ph, bins=4, min=0, max=1) / ph.numel()
    check(float(hist.max() - hist.min()) < 0.03, f"발생 보행 위상(1.1 Hz 시간 기준) 4 구간 균일 {[round(float(x), 3) for x in hist]}")
    s["lvl"] = 2; WD.w3d_dist_reset(env, torch.arange(N)); Fi = hs["dst_F"][hs["dst_kind"] == 2]
    check(float(Fi.min()) >= 6.0 and float(Fi.max()) <= 18.0 and float(Fi.max()) > 17.5, "단계 2: 충격 6–18 N")


# ------------------------------------------------------------------ [E]
def test_E():
    print("[E] 크기 커리큘럼 진급 규칙")
    N = 4000; env = _Env(N); s = WD._st(env)
    lv = []
    for it in range(400):
        env.common_step_counter = it * 24
        ids = torch.arange(N)[(torch.arange(N) + it) % 40 == 0]                      # 매 반복 100 env 리셋
        s["ep_imp"][ids] = True
        env.termination_manager.terminated = torch.rand(N) < (0.2 if s["lvl"] < 2 else 0.4)
        WD.w3d_dist_reset(env, ids)
        out = WD.w3d_curriculum(env, None)
        lv.append(int(out["level"]))
    check(lv[90] == 0 and max(lv) == 2 and lv[-1] == 2, f"낙상 20 % 단계는 체류 100 반복 뒤 진급, 40 % 단계에서 정지 (단계 이력 {sorted(set(lv))}, 100 반복 {lv[100]}·끝 {lv[-1]})")


def test_F():
    print("[F] vy 커리큘럼 대안(w3_cur2): 주기 평균 가로 이득 판정·전환 직후 제외")
    for gain_true, expect in ((0.8, 2), (0.3, 0)):
        N = 256; env = _Env(N); env._ge_cmd = torch.ones(N, dtype=torch.bool)
        vy = torch.where(torch.arange(N) % 2 == 0, 0.035, -0.035)
        env.command_manager.t.vel_command_b[:, 1] = vy
        W3._lat(env)
        pos = torch.zeros(N, 2)
        for k in range(2400 * 24 // 10):
            env.episode_length_buf += 1; env.common_step_counter += 10
            W3.w3_gait_update(env, None, f_step=F_STEP)
            pos[:, 1] += gain_true * vy * DT
            env.data.root_pos_w = torch.cat([pos, torch.zeros(N, 1)], 1)
            WC2.w3_lat_gain_tracker(env, None)
            out = WC2.w3_vy_curriculum2(env, None, min_ticks=20000.0)
        check(int(round(out["vy_max"] / 0.02)) - 2 == expect, f"실제 이득 {gain_true}: 단계 {out['vy_max']:.2f}(기대 단계 {expect}) · 블록 이득 {out['lat_gain']:.3f}")


def main():
    test_A(); test_B(); test_C(); test_D(); test_E(); test_F()
    print(f"\n결과: 실패 {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
