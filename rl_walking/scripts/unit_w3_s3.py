"""W3 Stage 3(제자리 회전, wz) CPU 단위시험 (1006) — torch CPU 만(Isaac·GPU 불필요).

  python3 rl_walking/scripts/unit_w3_s3.py [--ref 68493c11]

rl_walking/w3.py 를 가짜 mdp_hang(스텁)과 함께 올리고, --ref 커밋의 w3.py(git show)와 비교한다.
  [A] Stage 0–2 비트 동일: 바뀐 기존 함수 = install_w3_cmd·w3_gait_update 뿐(AST), install_w3_cmd stage 0/1/S1b/S1d/2 의 40 회 리샘플,
      w3_gait_update(기본 인자) 400 틱, Stage 2 합성 궤적의 tracker·stride·stride2·폭 벌점 — 그리고 **회전 추적기·회전 보상을 함께 돌려도**
      (명령 wz 0) 기존 상태·보상이 같고 새 보상은 정확히 0.
  [B] 반사실 발 요 착지 보상(w3_turn_stride raw/사건): wz ±0.1/0.2/0.3/0.4 × 요 스폰 0/180/40° × 발 링크 축 오프셋(CAD 식 x 축 90°):
      같은 방향 ≫ 0(제자리 디딤) ≫ 반대, 몸통만 비틀기 = 0, 지지발 피벗만 = 0(피벗 감시 지표엔 잡힘), 2배 회전, 이동+회전.
  [C] 순 요 추종(w3_yaw_track_rel 발 연결선 vs 골반) + w3_yaw_net: 가만히 0 · 정회전 + · 역회전 − · **톱니 몸통 비틀기**(발 고정).
  [D] Isaac Lab UniformVelocityCommand 의 _resample_command·_update_command(AST 추출, 원본 그대로)를 매 틱 돌려
      heading_command=False 면 stage 3 의 wz 가 유지되고, True·(0,0) 이면 0 으로 덮임(1006 측정 사실 재현).
  [E] Stage 3 명령 분포(S3·S3s2 혼합비, |wz| 범위·부호, 대각 없음, 헤딩 고리 표시 비율, W3_CMD 덮어쓰기 → wz₀).
  [F] 외부 헤딩 고리 w3_heading_loop(시작 요 목표·감싸기·포화·비대상 env 불변·STAND 불변·W3_HLOOP=0 끔).
  [G] wz 커리큘럼 진급 규칙. [H] 리드 발(wz < 0 → 위상 0.5).
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
F_STEP = 1.1


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
    pkg = f"w3s3stub_{tag}"
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
def qz(a):
    """요 a(텐서) → 쿼터니언 (w,x,y,z)."""
    return torch.stack([torch.cos(a / 2), torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2)], -1)


def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


Q_OFF = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])    # 발 링크 축 오프셋: x 축 90°(CAD Y-up 식) — 프레임 독립성 확인


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
        self._ge_obs = torch.zeros(N)
        yaw = torch.zeros(N) if yaw is None else yaw
        self.data = types.SimpleNamespace(root_quat_w=qz(yaw), root_pos_w=torch.zeros(N, 3), body_pos_w=torch.zeros(N, 2, 3),
                                          body_quat_w=qmul(qz(yaw).unsqueeze(1).repeat(1, 2, 1), Q_OFF.expand(N, 2, 4)),
                                          default_mass=torch.full((N, 1), 17.6), root_lin_vel_w=torch.zeros(N, 3),
                                          body_lin_vel_w=torch.zeros(N, 2, 3))
        self.sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=torch.zeros(N, 2, 3)), body_names=["l", "r"])
        self.scene = _Scene(self)


class _Scene:
    def __init__(self, env):
        self.env = env; self.sensors = {"contact_forces": env.sensor}

    def __getitem__(self, k):
        return self.env


def _rot(yaw, v):                                        # (N,) 요로 (N,2) 회전
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)


def _turn_scenario(mod, N, cmd, foot, swing_rot, yaw0, twist=None, pivot=None, half_w=0.144, ticks=60, s2=False, turn=True, move=None):
    """한 발(foot) tick 5 이륙 → 12 틱 공중(정점 5 cm) → 착지. 스윙발은 공중에서 골반 중심 기준 swing_rot(N,) 만큼 공전·자전.
    twist(N,): 같은 구간 골반 요만 그만큼(발 고정) / pivot(N,): 지지발이 접촉한 채 그만큼 자전 / move(N,2): 스윙발 요 프레임 추가 이동.
    반환 dict: turn_stride 합·pivot 누적(지지발)·td_dpsi 합·(구 경로) stride·stride2·폭 벌점."""
    env = _Env(N, yaw0)
    env.command_manager.t.vel_command_b[:] = cmd
    env._ge_cmd = torch.ones(N, dtype=torch.bool)
    p0 = [torch.stack([torch.zeros(N), torch.full((N,), half_w)], 1), torch.stack([torch.zeros(N), torch.full((N,), -half_w)], 1)]
    zero = torch.zeros(N)
    twist = zero if twist is None else twist; pivot = zero if pivot is None else pivot; move = torch.zeros(N, 2) if move is None else move
    out = {k: torch.zeros(N) for k in ("turn", "pivot", "dpsi", "stride", "stride2", "width")}
    mg = 17.6 * 9.81
    for t in range(ticks):
        env.episode_length_buf += 1; env.common_step_counter += 1
        v = min(1.0, max(0.0, (t - 5) / 12.0))
        F = torch.zeros(N, 2, 3); F[:, :, 2] = mg / 2
        bp = torch.zeros(N, 2, 3); fy = [zero.clone(), zero.clone()]
        for k in (0, 1):
            bp[:, k, :2] = _rot(yaw0, p0[k])
        sw_a = swing_rot * v; pv_a = pivot * v
        bp[:, foot, :2] = _rot(yaw0 + sw_a, p0[foot]) + _rot(yaw0, move * v)
        fy[foot] = sw_a; fy[1 - foot] = pv_a
        if 5 <= t < 17:
            F[:, foot, 2] = 0.0; F[:, 1 - foot, 2] = mg
            bp[:, foot, 2] = 0.05 * math.sin(math.pi * min(1.0, v + 1 / 12))
        env.data.body_pos_w = bp; env.sensor.data.net_forces_w = F
        env.data.body_quat_w = qmul(torch.stack([qz(yaw0 + fy[0]), qz(yaw0 + fy[1])], 1), Q_OFF.expand(N, 2, 4))
        env.data.root_quat_w = qz(yaw0 + twist * v)
        mod.w3_gait_update(env, None, f_step=F_STEP)
        mod.w3_step_tracker(env, None, nominal_half_width=0.10)
        out["stride"] += mod.w3_stride(env)
        if s2:
            mod.w3_lat_tracker(env, None, nominal_half_width=half_w, width_margin=0.05)
            out["stride2"] += mod.w3_stride2(env); out["width"] += mod.w3_width_pen(env)
        if turn:
            mod.w3_turn_tracker(env, None)
            out["turn"] += mod.w3_turn_stride(env)
            out["pivot"] += env._w3turn["pivot"][:, 1 - foot]; out["dpsi"] += env._w3turn["td_dpsi"].sum(1)
    out["env"] = env
    return out


# ------------------------------------------------------------------ [A] 비트 동일
def test_A(old, new):
    print("[A] Stage 0–2 비트 동일 + 명령 wz 0 에서 회전 항 정확히 0")
    fo = {n.name: ast.dump(n) for n in ast.parse(old.__src__).body if isinstance(n, ast.FunctionDef)}
    fn = {n.name: ast.dump(n) for n in ast.parse(new.__src__).body if isinstance(n, ast.FunctionDef)}
    changed = sorted(k for k in fo if fo[k] != fn.get(k))
    check(changed == ["install_w3_cmd", "w3_gait_update"], f"기존 함수 중 바뀐 것 = install_w3_cmd·w3_gait_update 뿐 (실제 {changed})")
    for label, kw in (("S0", {"stage": 0}), ("S1", {"stage": 1}), ("S1b/c", {"stage": 1, "vx_back_range": (0.03, 0.06)}),
                      ("S1d", {"stage": 1, "vx_range": (0.03, 0.06), "vx_back_range": (0.03, 0.06)}),
                      ("S2", {"stage": 2, "vx_back_range": (0.03, 0.06), "mix2": (0.3, 0.3, 0.4), "vy_range": (0.03, 0.04)})):
        outs = []
        for mod in (old, new):
            torch.manual_seed(7); env = _Env(512)
            mod.install_w3_cmd(env, None, p_step=0.7, **kw)
            seq = []
            for i in range(40):
                ids = torch.randperm(512)[: 1 + (i * 37) % 300]
                env.command_manager.t._resample_command(ids)
                seq.append((env.command_manager.t.vel_command_b.clone(), env._ge_cmd.clone()))
            outs.append((seq, hasattr(env, "_w3_wz0")))
        ok = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(outs[0][0], outs[1][0]))
        check(ok and not outs[1][1], f"install_w3_cmd {label}: 40 회 리샘플 명령·ge_cmd 비트 동일, Stage 3 상태 속성 없음")
    outs = []
    for mod in (old, new):
        torch.manual_seed(3); env = _Env(256)
        env.command_manager.t.vel_command_b[:, 1] = torch.empty(256).uniform_(-0.08, 0.08)
        env.command_manager.t.vel_command_b[:, 2] = torch.empty(256).uniform_(-0.4, 0.4)
        tr = []
        for t in range(400):
            env.episode_length_buf += 1
            rs = torch.rand(256) < 0.01; env.episode_length_buf[rs] = 1
            env._ge_cmd = torch.rand(256) < (0.7 if (t // 37) % 2 == 0 else 0.2)
            mod.w3_gait_update(env, None, f_step=1.1, lead_vy_cmin=(0.02 if t >= 200 else 0.0))
            tr.append(torch.cat([env._gu, env._gf, env._ge.float(), env._ge_obs]))
        outs.append(torch.stack(tr))
    check(torch.equal(outs[0], outs[1]), "w3_gait_update(lead_wz 기본 0, lead_vy 0/0.02): 400 틱 위상·주파수·ge·ge_obs 비트 동일(vy·wz ≠ 0 명령 포함)")
    for vx, vy in ((0.0, 0.0), (0.06, 0.0), (-0.04, 0.0), (0.0, 0.05), (0.0, -0.03)):
        for foot in (0, 1):
            cmd = torch.tensor([vx, vy, 0.0])
            yaw = torch.linspace(-math.pi, math.pi, 32)
            rot = torch.linspace(-0.3, 0.3, 32)
            ro = _turn_scenario(old, 32, cmd, foot, rot, yaw, s2=True, turn=False)
            rn = _turn_scenario(new, 32, cmd, foot, rot, yaw, s2=True, turn=True)
            same = all(torch.equal(ro["env"]._w3[k], rn["env"]._w3[k]) for k in ro["env"]._w3) and \
                all(torch.equal(ro["env"]._w3lat[k], rn["env"]._w3lat[k]) for k in ro["env"]._w3lat)
            check(same and torch.equal(ro["stride"], rn["stride"]) and torch.equal(ro["stride2"], rn["stride2"]) and torch.equal(ro["width"], rn["width"])
                  and bool((rn["turn"] == 0).all()),
                  f"(vx {vx:+.2f}, vy {vy:+.2f})·발 {foot}: 회전 추적기 동반 실행에도 _w3·_w3lat·stride·stride2·폭 벌점 비트 동일, 회전 착지 보상 = 0"
                  f" (발 요 −17~+17° 회전 착지 포함)")
    # 회전 추종 개선분: 명령 wz 0 → 정확히 0 (몸이 돌고 있어도)
    env = _Env(16, torch.linspace(-3, 3, 16)); env._ge = torch.ones(16, dtype=torch.bool)
    for t in range(80):
        env.episode_length_buf += 1; env.common_step_counter += 1
        a = torch.linspace(-3, 3, 16) + 0.4 * t * DT
        env.data.root_quat_w = qz(a)
        env.data.body_pos_w[:, 0, :2] = _rot(a, torch.tensor([[0.0, 0.144]]).repeat(16, 1)); env.data.body_pos_w[:, 1, :2] = -env.data.body_pos_w[:, 0, :2]
        r1 = new.w3_yaw_track_rel(env); r2 = new.w3_yaw_track_rel(env, src="pelvis")
        if not (bool((r1 == 0).all()) and bool((r2 == 0).all())):
            break
    check(bool((r1 == 0).all()) and bool((r2 == 0).all()), "명령 wz 0: w3_yaw_track_rel(발·골반) 정확히 0(몸이 0.4 rad/s 로 돌아도)")


# ------------------------------------------------------------------ [B] 반사실 발 요 착지 보상
def test_B(new):
    print("[B] 반사실 발 요 착지 보상 — 사건당 w3_turn_stride raw(가중 10 곱하기 전)")
    rows = []
    yaw = torch.tensor([0.0, math.pi, 0.7])
    for wz in (0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4):
        D = wz / F_STEP
        cmd = torch.tensor([0.0, 0.0, wz])
        lead = 0 if wz > 0 else 1
        cases = {"같은 방향(리드 발)": dict(foot=lead, swing_rot=D),
                 "같은 방향(따라오는 발)": dict(foot=1 - lead, swing_rot=D),
                 "제자리 디딤": dict(foot=lead, swing_rot=0.0),
                 "반대 방향": dict(foot=lead, swing_rot=-D),
                 "몸통만 비틀기(발 요 불변)": dict(foot=lead, swing_rot=0.0, twist=D),
                 "지지발 피벗만(스윙발 요 불변)": dict(foot=lead, swing_rot=0.0, pivot=D),
                 "같은 방향 2배": dict(foot=lead, swing_rot=2 * D),
                 "같은 방향 절반": dict(foot=lead, swing_rot=0.5 * D),
                 "같은 방향 + 앞 3 cm": dict(foot=lead, swing_rot=D, move=(0.03, 0.0))}
        res = {}
        for name, kw in cases.items():
            sr = torch.full((3,), kw["swing_rot"])
            tw = torch.full((3,), kw.get("twist", 0.0)); pv = torch.full((3,), kw.get("pivot", 0.0))
            mv = torch.tensor(kw["move"]).repeat(3, 1) if "move" in kw else None
            o = _turn_scenario(new, 3, cmd, kw["foot"], sr, yaw, twist=tw, pivot=pv, move=mv)
            res[name] = o
            rows.append((wz, name, float(o["turn"][0]), float(o["turn"][1]), float(o["turn"][2]), math.degrees(float(o["dpsi"][0])), math.degrees(float(o["pivot"][0]))))
        same, zero, opp = res["같은 방향(리드 발)"]["turn"], res["제자리 디딤"]["turn"], res["반대 방향"]["turn"]
        check(bool((same > 0.3).all()) and bool((zero.abs() < 1e-6).all()) and bool((opp < -0.3).all()),
              f"wz {wz:+.1f}(Δψ* {math.degrees(D):.1f}°): 같은 {float(same[0]):+.3f} ≫ 제자리 {float(zero[0]):+.3f} ≫ 반대 {float(opp[0]):+.3f} (요 0/180/40°)")
        check(bool(((same - same[0]).abs() < 1e-4).all()), f"wz {wz:+.1f}: 요 스폰 180°·40° + 발 링크 축 오프셋에서도 같은 점수")
        check(bool((res["몸통만 비틀기(발 요 불변)"]["turn"].abs() < 1e-6).all()), f"wz {wz:+.1f}: 몸통만 비틀기 = 0 ({float(res['몸통만 비틀기(발 요 불변)']['turn'][0]):+.4f})")
        pv_o = res["지지발 피벗만(스윙발 요 불변)"]
        check(bool((pv_o["turn"].abs() < 1e-6).all()) and bool(((pv_o["pivot"] - D).abs() < 1e-3).all()),
              f"wz {wz:+.1f}: 지지발 피벗만 = 0 점, 피벗 감시 지표 = {math.degrees(float(pv_o['pivot'][0])):+.1f}°(= Δψ*)")
        check(bool((res["같은 방향(따라오는 발)"]["turn"] > 0.3).all()), f"wz {wz:+.1f}: 따라오는 발도 같은 방향이면 + ({float(res['같은 방향(따라오는 발)']['turn'][0]):+.3f})")
        check(bool((res["같은 방향 절반"]["turn"] > 0).all()), f"wz {wz:+.1f}: 절반 회전도 + ({float(res['같은 방향 절반']['turn'][0]):+.3f}) — 단조 접근 기울기")
    print("\n  | wz | 경우 | raw(요 0) | 요 180° | 요 40° | 착지 Δψ [°] | 지지발 피벗 [°] |")
    print("  |---|---|---|---|---|---|---|")
    for r in rows:
        print(f"  | {r[0]:+.1f} | {r[1]} | {r[2]:+.3f} | {r[3]:+.3f} | {r[4]:+.3f} | {r[5]:+.1f} | {r[6]:+.1f} |")
    print()


# ------------------------------------------------------------------ [C] 순 요 추종
def _yaw_track_run(new, wz, pelvis_fn, feet_fn, seconds=12.0, warm=2.0, N=4):
    env = _Env(N); env._ge = torch.ones(N, dtype=torch.bool)
    env.command_manager.t.vel_command_b[:, 2] = wz
    yaw0 = torch.tensor([0.0, math.pi, 0.7, -2.0])[:N]
    acc = {"feet": torch.zeros(N), "pelvis": torch.zeros(N), "net": torch.zeros(N)}; n = 0
    for i in range(int(seconds / DT)):
        t = (i + 1) * DT
        env.episode_length_buf += 1; env.common_step_counter += 1
        a = yaw0 + pelvis_fn(t); b = yaw0 + feet_fn(t)
        env.data.root_quat_w = qz(a)
        pl = _rot(b, torch.tensor([[0.0, 0.144]]).repeat(N, 1))
        env.data.body_pos_w = torch.zeros(N, 2, 3); env.data.body_pos_w[:, 0, :2] = pl; env.data.body_pos_w[:, 1, :2] = -pl
        f = new.w3_yaw_track_rel(env, src="feet"); p = new.w3_yaw_track_rel(env, src="pelvis"); y = new.w3_yaw_net(env)
        if t > warm:
            acc["feet"] += f; acc["pelvis"] += p; acc["net"] += y; n += 1
    return {k: float((v / n).mean()) for k, v in acc.items()}, {k: float((v / n).std()) for k, v in acc.items()}


def test_C(new):
    print("[C] 순 요 추종 — 초당 평균 raw(w3_yaw_track_rel 발 연결선 / 골반, w3_yaw_net 벌점 크기), 12 s(앞 2 s 제외), 요 스폰 4 종")
    saw = lambda t: 0.3 * (t % 1.8) if (t % 1.8) < 1.5 else 0.45 * (1.0 - ((t % 1.8) - 1.5) / 0.3)   # 1.5 s 동안 +0.3 rad/s 비틀기 → 0.3 s 복귀(±26°)
    rows = []
    for wz in (0.3, -0.3, 0.1):
        sg = 1.0 if wz > 0 else -1.0
        cases = {"가만히": (lambda t: 0.0, lambda t: 0.0),
                 "정회전": (lambda t, w=wz: w * t, lambda t, w=wz: w * t),
                 "역회전": (lambda t, w=wz: -w * t, lambda t, w=wz: -w * t),
                 "절반 회전": (lambda t, w=wz: 0.5 * w * t, lambda t, w=wz: 0.5 * w * t),
                 "몸통 톱니 비틀기(발 고정)": (lambda t, s=sg: s * saw(t), lambda t: 0.0)}
        out = {}
        for name, (pf, ff) in cases.items():
            m, sd = _yaw_track_run(new, wz, pf, ff)
            out[name] = m; rows.append((wz, name, m["feet"], m["pelvis"], m["net"]))
        still, good, rev, half, saw_ = (out[k] for k in ("가만히", "정회전", "역회전", "절반 회전", "몸통 톱니 비틀기(발 고정)"))
        check(abs(still["feet"]) < 1e-6 and good["feet"] > 0.2 and rev["feet"] < 0 and 0 < half["feet"] < good["feet"],
              f"wz {wz:+.1f}: 발 추종 가만히 {still['feet']:+.3f} · 정 {good['feet']:+.3f} · 절반 {half['feet']:+.3f} · 역 {rev['feet']:+.3f}")
        check(abs(saw_["feet"]) < 1e-6, f"wz {wz:+.1f}: 몸통 톱니 비틀기 → 발 추종 = 0 ({saw_['feet']:+.4f}); 골반 기반이면 {saw_['pelvis']:+.3f}")
        check(saw_["net"] >= abs(wz) - 0.02 and still["net"] >= abs(wz) - 1e-6 and good["net"] < 0.02,
              f"wz {wz:+.1f}: w3_yaw_net(골반 선형) 가만히 {still['net']:.3f} · 톱니 {saw_['net']:.3f} ≥ |wz| · 정회전 {good['net']:.3f}(톱니로 벌점 못 줄임)")
    # 총합(가중 3·추종, −4·순 요): 톱니 비틀기는 가만히보다 나으면 안 됨 — 발 기반 채택 근거
    print("\n  | wz | 행동 | 발 추종 raw | 골반 추종 raw(기각안) | yaw_net 벌점 | 총 +3·발 −4·net [/s] | (골반안) +3·골반 −4·net |")
    print("  |---|---|---|---|---|---|---|")
    for r in rows:
        print(f"  | {r[0]:+.1f} | {r[1]} | {r[2]:+.3f} | {r[3]:+.3f} | {r[4]:.3f} | {3 * r[2] - 4 * r[4]:+.3f} | {3 * r[3] - 4 * r[4]:+.3f} |")
    print()
    tot = {(r[0], r[1]): (3 * r[2] - 4 * r[4], 3 * r[3] - 4 * r[4]) for r in rows}
    for wz in (0.3, -0.3, 0.1):
        s_f, s_p = tot[(wz, "가만히")]; w_f, w_p = tot[(wz, "몸통 톱니 비틀기(발 고정)")]
        check(w_f <= s_f + 1e-6, f"wz {wz:+.1f}: 채택안(발) 총합 톱니 {w_f:+.3f} ≤ 가만히 {s_f:+.3f}"
              + (f" · 기각안(골반) 톱니 {w_p:+.3f} vs 가만히 {s_p:+.3f}" + (" → 골반안은 비틀기 해킹 가능" if w_p > s_p else "")))


# ------------------------------------------------------------------ [D] Isaac Lab 명령 항 후처리
ISAAC_VEL = "/home/ryu/IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py"


def test_D(new):
    print("[D] Isaac Lab UniformVelocityCommand 리샘플·후처리(원본 AST) 뒤 wz 보존")
    if not os.path.exists(ISAAC_VEL):
        check(False, f"Isaac Lab 소스 없음: {ISAAC_VEL}"); return
    tree = ast.parse(open(ISAAC_VEL).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UniformVelocityCommand")
    g = {"torch": torch, "math_utils": types.SimpleNamespace(wrap_to_pi=lambda x: torch.atan2(torch.sin(x), torch.cos(x))), "Sequence": list}
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_resample_command", "_update_command")]
    exec(compile(ast.Module(body=fns, type_ignores=[]), ISAAC_VEL, "exec"), g)
    for heading in (False, True):
        N = 4000; torch.manual_seed(5); env = _Env(N)
        t = env.command_manager.t
        t.device = "cpu"
        t.cfg = types.SimpleNamespace(heading_command=heading, heading_control_stiffness=0.5, rel_heading_envs=1.0, rel_standing_envs=0.5,
                                      ranges=types.SimpleNamespace(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-math.pi, math.pi)))
        t.is_heading_env = torch.zeros(N, dtype=torch.bool); t.heading_target = torch.zeros(N)
        t.robot = types.SimpleNamespace(data=types.SimpleNamespace(heading_w=torch.empty(N).uniform_(-3, 3)))
        t._resample_command = types.MethodType(g["_resample_command"], t)
        upd = types.MethodType(g["_update_command"], t)
        new.install_w3_cmd(env, None, p_step=0.7, stage=3, vx_back_range=(0.03, 0.06), mix3=(0.3, 0.3, 0.0, 0.4), wz_range=(0.1, 0.4))
        t._resample_command(torch.arange(N)); upd()
        wz0 = env._w3_wz0.clone(); turn = wz0 != 0
        mids = []
        for k in range(50):
            upd()
            if k == 24:
                mids.append(t.vel_command_b[:, 2].clone())
        if not heading:
            check(int(turn.sum()) > 0 and torch.equal(mids[0][turn], wz0[turn]) and torch.equal(t.vel_command_b[:, 2], wz0)
                  and int(t.is_standing_env.sum()) == 0,
                  f"heading_command False: 회전 명령 {int(turn.sum())} env 의 wz 가 25·50 틱 뒤에도 리샘플 값 그대로, 서기 표시 0(install 이 덮음)")
        else:
            check(bool((t.vel_command_b[:, 2] == 0).all()) and int(turn.sum()) > 0,
                  f"heading_command True·ang_vel_z (0,0): 같은 회전 명령 {int(turn.sum())} env 의 wz 가 첫 후처리에서 0 으로 덮임(1006 측정 사실 재현 → Stage 3 은 끔 필수)")


# ------------------------------------------------------------------ [E] 명령 분포
def test_E(new):
    print("[E] Stage 3 명령 분포")
    for label, mix3, p_hl in (("W3S3", (0.3, 0.3, 0.0, 0.4), 0.0), ("W3S3h", (0.3, 0.3, 0.0, 0.4), 0.5), ("W3S3s2", (0.25, 0.2, 0.25, 0.3), 0.0)):
        N = 200000; torch.manual_seed(11); env = _Env(N)
        new.install_w3_cmd(env, None, p_step=0.7, stage=3, vx_back_range=(0.03, 0.06), vy_range=(0.03, 0.04), mix3=mix3, wz_range=(0.1, 0.2), p_hloop=p_hl)
        env.command_manager.t._resample_command(torch.arange(N))
        c = env.command_manager.t.vel_command_b; ge = env._ge_cmd; g = c[ge]
        fr = lambda m: float(m.float().mean())
        inpl = (g == 0).all(1); fb = g[:, 0] != 0; lat = g[:, 1] != 0; turn = g[:, 2] != 0
        check(abs(fr(ge) - 0.7) < 0.01, f"{label}: STEP 비율 {fr(ge):.3f}")
        check(all(abs(x - y) < 0.01 for x, y in zip((fr(inpl), fr(fb), fr(lat), fr(turn)), mix3)),
              f"{label}: STEP 중 제자리 {fr(inpl):.3f} · 앞뒤 {fr(fb):.3f} · 옆 {fr(lat):.3f} · 회전 {fr(turn):.3f} (목표 {mix3})")
        w = g[turn, 2]
        check(bool(((w.abs() >= 0.1) & (w.abs() <= 0.2)).all()) and abs(fr(w > 0) - 0.5) < 0.01, f"{label}: |wz| ∈ [0.1, 0.2], 양수 비율 {fr(w > 0):.3f}")
        multi = ((g != 0).sum(1) > 1).sum()
        check(int(multi) == 0 and bool((c[~ge] == 0).all()), f"{label}: 대각·이동+회전 동시 없음, STAND 명령 0")
        check(torch.equal(env._w3_wz0, c[:, 2]), f"{label}: env._w3_wz0 = 리샘플 wz")
        hl = env._w3_hl
        exp_hl = p_hl * float((ge & (c[:, 2] == 0)).float().mean())
        check(abs(fr(hl) - exp_hl) < 0.01 and not bool((hl & ~ge).any()) and not bool((hl & (c[:, 2] != 0)).any()),
              f"{label}: 헤딩 고리 표시 {fr(hl):.3f}(기대 {exp_hl:.3f}) — STAND·회전 명령에는 없음")
    env._w3_wz_range = (0.1, 0.4); env.command_manager.t._resample_command(torch.arange(N))
    w = env.command_manager.t.vel_command_b[:, 2]; w = w[w != 0]
    check(float(w.abs().max()) > 0.39, "커리큘럼 상한(env._w3_wz_range) 변경이 다음 리샘플에 반영")
    os.environ["W3_CMD"] = "0,0,-0.3"
    try:
        env = _Env(64); new.install_w3_cmd(env, None, stage=3); env.command_manager.t._resample_command(torch.arange(64))
        check(bool((env._w3_wz0 == -0.3).all()) and bool((env.command_manager.t.vel_command_b[:, 2] == -0.3).all()), "W3_CMD='0,0,-0.3' 고정 → 명령·wz₀ = −0.3")
    finally:
        del os.environ["W3_CMD"]


# ------------------------------------------------------------------ [F] 외부 헤딩 고리
def test_F(new):
    print("[F] 외부 헤딩 고리 w3_heading_loop(kp 0.5, ±0.4)")
    N = 6
    env = _Env(N); new.install_w3_cmd(env, None, stage=3)
    t = env.command_manager.t
    env._w3_hl = torch.tensor([True, True, True, False, True, True]); env._w3_wz0 = torch.tensor([0.0, 0.0, 0.0, 0.25, 0.0, 0.2])
    env._ge = torch.tensor([True, True, True, True, False, True])
    psi0 = torch.tensor([0.0, 3.0, -1.0, 0.5, 0.2, 0.0])
    env.data.root_quat_w = qz(psi0); env.episode_length_buf[:] = 1
    t.vel_command_b[:, 2] = env._w3_wz0
    new.w3_heading_loop(env, None)
    check(torch.allclose(t.vel_command_b[:, 2], torch.tensor([0.0, 0.0, 0.0, 0.25, 0.0, 0.2])), f"에피소드 시작: 목표 = 현재 요 → 고리 출력 wz₀ (실제 {t.vel_command_b[:, 2].tolist()})")
    env.episode_length_buf[:] = 5
    psi1 = torch.tensor([0.2, -3.0, -1.0 - 1.5, 0.9, 0.6, 0.0])                        # env1: 3.0 → −3.0(±π 넘어 +0.283 rad 회전)
    env.data.root_quat_w = qz(psi1)
    new.w3_heading_loop(env, None)
    w = t.vel_command_b[:, 2]
    e1 = -(2 * math.pi - 6.0)                                                          # wrap(3.0 − (−3.0)) = −0.283
    check(abs(float(w[0]) - (-0.1)) < 1e-5, f"요 +0.2 rad 벗어남 → wz = −0.1 (실제 {float(w[0]):+.4f})")
    check(abs(float(w[1]) - 0.5 * e1) < 1e-4, f"±π 감싸기: 목표 3.0·현재 −3.0 → wz = {0.5 * e1:+.4f} (실제 {float(w[1]):+.4f})")
    check(abs(float(w[2]) - 0.4) < 1e-6, f"큰 오차(1.5 rad) → 포화 +0.4 (실제 {float(w[2]):+.4f})")
    check(abs(float(w[3]) - 0.25) < 1e-6 and abs(float(w[4]) - 0.0) < 1e-6, "고리 대상 아님(wz₀ 0.25 유지)·STAND(ge 0) env 는 wz₀ 그대로")
    exp5 = 0.2 + 0.5 * (0.2 * DT)
    check(abs(float(w[5]) - exp5) < 1e-5, f"wz₀ 0.2 를 목표 적분(ψ* += wz₀·dt): wz = {exp5:+.4f} (실제 {float(w[5]):+.4f})")
    os.environ["W3_HLOOP"] = "0"
    try:
        t.vel_command_b[:, 2] = 0.123; new.w3_heading_loop(env, None)
        check(bool((t.vel_command_b[:, 2] == 0.123).all()), "W3_HLOOP=0: 고리 끔(명령 손대지 않음)")
    finally:
        del os.environ["W3_HLOOP"]
    # 닫힌 고리 모의: 요 1차 플랜트(ψ̇ = 실제 wz 의 70 %·편향 −0.02 rad/s) → 정상상태 헤딩 오차 = 편향/(kp·0.7) 수준으로 유한
    env = _Env(1); new.install_w3_cmd(env, None, stage=3)
    env._w3_hl = torch.tensor([True]); env._w3_wz0 = torch.zeros(1); env._ge = torch.tensor([True]); psi = torch.zeros(1)
    env.episode_length_buf[:] = 1; env.data.root_quat_w = qz(psi); new.w3_heading_loop(env, None)
    for k in range(1500):
        env.episode_length_buf += 1
        psi = psi + DT * (0.7 * env.command_manager.t.vel_command_b[:, 2] - 0.02)
        env.data.root_quat_w = qz(psi); new.w3_heading_loop(env, None)
    ss = math.degrees(float(psi[0])); exp_ss = math.degrees(-0.02 / (0.5 * 0.7))
    check(abs(ss - exp_ss) < 0.5, f"닫힌 고리(이득 0.7·편향 −0.02 rad/s, 30 s): 헤딩 오차 {ss:+.2f}° ≈ 이론 {exp_ss:+.2f}°(무고리면 −34°)")


# ------------------------------------------------------------------ [G] 커리큘럼
def test_G(new):
    print("[G] wz 커리큘럼")
    env = _Env(8); T_ = new._turn(env)
    seq = []
    for it, (sign, gain) in enumerate([(0.85, 0.8), (0.95, 0.3), (0.95, 0.6), (0.95, 0.6), (0.95, 0.6), (0.95, 0.6)]):
        env.common_step_counter = (it + 1) * 120 * 24
        T_["n_turn_td"] += 4000; T_["n_turn_ok"] += 4000 * sign; T_["gain_den"] += 1000.0; T_["gain_num"] += 1000.0 * gain
        T_["swing_sgn"] += 100.0; T_["piv_sgn"] += 25.0; T_["piv_abs"] += 900.0      # |·| 떨림은 비율에 안 들어가야
        out = new.w3_wz_curriculum(env, None)
        seq.append(out["wz_max"])
    check(seq == [0.2, 0.2, 0.3, 0.4, 0.4, 0.4], f"상한 순서 {seq} (부호 0.85 정체 · 이득 0.3 정체 · 이후 0.3 → 0.4, 최대 유지)")
    check(abs(out["pivot_frac"] - 0.2) < 1e-6, f"피벗 비율 로그(부호 기여) {out['pivot_frac']:.3f} = 25/(25+100) — |·| 떨림 900 무시")


# ------------------------------------------------------------------ [H] 리드 발
def test_H(new):
    print("[H] 리드 발 위상(lead_wz_cmin 0.08)")
    env = _Env(5)
    env.command_manager.t.vel_command_b[:, 2] = torch.tensor([0.3, -0.3, -0.05, -0.3, 0.0])
    env.command_manager.t.vel_command_b[:, 1] = torch.tensor([0.0, 0.0, 0.0, 0.05, 0.0])
    env._ge_cmd = torch.ones(5, dtype=torch.bool); env.episode_length_buf[:] = 1
    new.w3_gait_update(env, None, f_step=1.1, lead_vy_cmin=0.02, lead_wz_cmin=0.08)
    check(env._gu.tolist() == [0.0, 0.5, 0.0, 0.0, 0.0],
          f"wz +0.3→0 · −0.3→0.5 · −0.05(문턱 아래)→0 · 옆 +0.05 와 wz −0.3 동시(옆 규칙 우선)→0 · 0→0 (실제 {env._gu.tolist()})")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="68493c11")
    a = ap.parse_args()
    src_old = subprocess.run(["git", "-C", REPO, "show", f"{a.ref}:rl_walking/w3.py"], check=True, capture_output=True, text=True).stdout
    src_new = open(os.path.join(REPO, "rl_walking", "w3.py")).read()
    old = _load(src_old, "old"); old.__src__ = src_old
    new = _load(src_new, "new"); new.__src__ = src_new
    torch.set_default_dtype(torch.float32)
    test_A(old, new); test_B(new); test_C(new); test_D(new); test_E(new); test_F(new); test_G(new); test_H(new)
    print(f"\n결과: 실패 {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
