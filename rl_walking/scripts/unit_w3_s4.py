"""W3 Stage 4(혼합 명령 vx·vy·wz, 곡선 보행) CPU 단위시험 (1006) — torch CPU 만(Isaac·GPU 불필요).

  python3 rl_walking/scripts/unit_w3_s4.py [--ref 2aeb91cd]

rl_walking/w3.py 를 가짜 mdp_hang(스텁)과 함께 올리고 --ref 커밋(W3 Stage 3 구현)의 w3.py 와 비교한다. 가짜 env·스텁은 unit_w3_s3.py 의 것을 재사용.
  [A] Stage 0–3 비트 동일: 바뀐 기존 함수 = install_w3_cmd 뿐(AST). install_w3_cmd S0·S1·S1b/c·S1d·S2·S3·S3h·S3s2 40 회 리샘플(명령·ge_cmd·wz₀·고리 표시) 같음,
      Stage 4 속성 없음. 단일 축·제자리 회전·명령 0 궤적(곡선 몸 운동 포함)에서 혼합 추적기를 함께 돌려도 기존 상태(_w3·_w3lat·_w3turn) 같고
      w3_stride4 ≡ w3_stride2, w3_track_rel4 ≡ w3_track_rel, w3_drift_net4 ≡ w3_drift_net(torch.equal, 틱마다).
  [B] 반사실 곡선 착지(사건당, 가중 10 곱한 w3_stride4 + w3_turn_stride): 몸이 원호(본체 속도 (vx, vy)·wz)를 따라가는 정상 상태 강체 보행에서
      맞는 곡선 착지(원호 적분으로 독립 계산) ≫ 단일 축만 맞춘 착지(직선 병진 · 직선 병진 + 발 요 · 제자리 회전 · 회전 방향만 반대) ≫ 반대.
      목표식 변형 비교: 채택(직전 착지 골반 요 프레임 + R(θ/2)·sinc + 지레) vs 브리프 문자식(이륙 골반 요 + R(θ/2), 지레 없음) vs 지레 없음 vs 이륙 프레임 정확식.
  [C] 곡선 병진 추종(w3_track_rel4·w3_drift_net4 vs 기존): 원호 운동 12 s — 기존 식의 체계 오차, 새 식은 정확한 원호에서 오차 0·순위 유지.
  [D] Stage 4 명령 분포(STEP 0.7, 범주 mix4, 혼합 조합 mix_pat, 축별 부호 독립, 크기 커리큘럼, 뒤 범위, 고리 표시, W3_CMD).
  [E] 리드 발 충돌 해소(옆 우선) — w3_gait_update(코드 불변) 결과 표 + 근거 수치(잘못된 리드의 첫 걸음 발 간격 vs 회전 리드 위반 비용).
  [F] 혼합 크기 커리큘럼 진급 규칙. [G] 혼합 추적기 축별 부호 지표(맞는 곡선 100 %, 반대 0 %).
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

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("unit_w3_s3_lib", os.path.join(REPO, "rl_walking", "scripts", "unit_w3_s3.py"))
S3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(S3)
_Env, qz, qmul, Q_OFF, DT, F_STEP = S3._Env, S3.qz, S3.qmul, S3.Q_OFF, S3.DT, S3.F_STEP
FAILS = []
W_HALF = 0.144
T_CYC = 1.0 / F_STEP


def check(cond, msg):
    print(("  [통과] " if cond else "  [실패] ") + msg)
    if not cond:
        FAILS.append(msg)


def rot(a, v):
    c, s = torch.cos(a), torch.sin(a)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def arc(psi0, vb, w, t):
    """원호 적분(독립 계산): 본체 속도 vb(N,2)·요 속도 w(N) 로 t 초 — 변위 X(N,2)(월드), 요 ψ(N). ∫R(ψ0 + w s) vb ds 의 닫힌 꼴."""
    wt = w * t
    small = w.abs() < 1e-9
    ws = torch.where(small, torch.ones_like(w), w)
    a11 = torch.where(small, torch.full_like(w, t), torch.sin(wt) / ws); a12 = torch.where(small, torch.zeros_like(w), (torch.cos(wt) - 1.0) / ws)
    a21 = torch.where(small, torch.zeros_like(w), (1.0 - torch.cos(wt)) / ws); a22 = a11
    loc = torch.stack([a11 * vb[:, 0] + a12 * vb[:, 1], a21 * vb[:, 0] + a22 * vb[:, 1]], -1)
    return rot(psi0, loc), psi0 + wt


# ------------------------------------------------------------------ 한 번 스윙 곡선 시나리오
LIFT, LAND, TAIL = 23, 45, 4


def curve_swing(mod, cmd, foot, vb, w, psi0, D=None, dpsi=None, s4=True, variants=None):
    """에피소드 시작(틱 1 = fresh, 직전 착지 = 시작 자세) → 몸은 원호(vb, w) — 틱 k 의 주기 시간 τ = k·T/LAND(착지 = 정확히 한 주기).
    스윙발 foot: LIFT 틱 이륙 → LAND 틱 착지. 착지 변위 D(N,2, 시작 요 프레임; None = 정상 상태 강체 착지 = X(T) + R(ψ_T)p − R(ψ0)p),
    발 요 변화 dpsi(N; None = w·T). 다른 발 = 반주기 전에 정상 상태로 놓인 위치 X(T/2) + R(ψ_{T/2})p 에 고정.
    반환: 보상 합(stride4·turn·stride2·track4/track·drift4/drift — 틱 합), 변형 stride4 합, 혼합 추적기·기존 상태."""
    N = vb.shape[0]
    env = _Env(N, psi0)
    env.episode_length_buf[:] = 0
    env.command_manager.t.vel_command_b[:] = cmd
    env._ge_cmd = torch.ones(N, dtype=torch.bool)
    pL = torch.tensor([0.0, W_HALF]).expand(N, 2); pR = torch.tensor([0.0, -W_HALF]).expand(N, 2)
    p = [pL, pR]
    XT, psiT = arc(psi0, vb, w, T_CYC)
    Xh, psih = arc(psi0, vb, w, 0.5 * T_CYC)
    P0 = rot(psi0, p[foot])
    if D is None:
        Pland = XT + rot(psiT, p[foot])
    else:
        Pland = P0 + rot(psi0, D)
    dps = w * T_CYC if dpsi is None else dpsi
    Pother = Xh + rot(psih, p[1 - foot]); yaw_other = psih
    out = {k: torch.zeros(N) for k in ("s4", "turn", "s2", "tr4", "tr", "dr4", "dr", "W", "bad")}
    vout = {k: torch.full((N,), float("nan")) for k in list(variants or {}) + (["채택식"] if variants else [])}
    mg = 17.6 * 9.81
    for k in range(LAND + TAIL):
        env.episode_length_buf += 1; env.common_step_counter += 1
        tau = min(k, LAND) * T_CYC / LAND
        X, psi = arc(psi0, vb, w, tau)
        env.data.root_quat_w = qz(psi)
        env.data.root_pos_w = torch.cat([X, torch.zeros(N, 1)], 1)
        v_w = rot(psi, vb); env.data.root_lin_vel_w = torch.cat([v_w, torch.zeros(N, 1)], 1)
        bp = torch.zeros(N, 2, 3); F = torch.zeros(N, 2, 3); F[:, :, 2] = mg / 2
        s = min(1.0, max(0.0, (k - LIFT) / (LAND - LIFT)))
        bp[:, foot, :2] = P0 + s * (Pland - P0); bp[:, 1 - foot, :2] = Pother
        fy = [None, None]; fy[foot] = psi0 + s * dps; fy[1 - foot] = yaw_other
        if LIFT <= k < LAND:
            F[:, foot, 2] = 0.0; F[:, 1 - foot, 2] = mg
            bp[:, foot, 2] = 0.05 * math.sin(math.pi * s)
        env.data.body_pos_w = bp; env.sensor.data.net_forces_w = F
        env.data.body_quat_w = qmul(torch.stack([qz(fy[0]), qz(fy[1])], 1), Q_OFF.expand(N, 2, 4))
        mod.w3_gait_update(env, None, f_step=F_STEP)
        mod.w3_step_tracker(env, None, nominal_half_width=0.10)
        mod.w3_lat_tracker(env, None, nominal_half_width=W_HALF, width_margin=0.05)
        mod.w3_turn_tracker(env, None)
        if s4:
            mod.w3_mix_tracker(env, None, nominal_half_width=W_HALF)
            out["s4"] += mod.w3_stride4(env, nominal_half_width=W_HALF)
            out["tr4"] += mod.w3_track_rel4(env); out["dr4"] += mod.w3_drift_net4(env)
            tdk = env._w3["ev_td"][:, foot] > 0.5
            for name, kw in list((variants or {}).items()) + ([("채택식", {})] if variants else []):
                kw2 = dict(frame="td", rot_frac=0.5, lever=True, sinc=True); kw2.update(kw)
                Tt = mod.w3_curve_target(env.command_manager.t.vel_command_b, env._gf, W_HALF, rot_frac=kw2["rot_frac"], lever=kw2["lever"], sinc=kw2["sinc"])
                yf = env._w3mix["psi_ref"] if kw2["frame"] == "td" else env._w3mix["psi_lo"]
                e = (rot(-yf[:, foot], env._w3mix["td_d"][:, foot]) - Tt[:, foot]).norm(dim=-1)   # 착지 목표 오차[m] — 기준선 항 없이 식의 정확도만
                vout[name] = torch.where(tdk, e, vout[name])
        out["turn"] += mod.w3_turn_stride(env)
        out["s2"] += mod.w3_stride2(env)
        out["tr"] += mod.w3_track_rel(env); out["dr"] += mod.w3_drift_net(env)
        out["W"] += env._w3lat["td_width"].sum(1); out["bad"] += env._w3lat["width_bad"].sum(1)
    out["env"] = env; out["var"] = vout
    return out


# ------------------------------------------------------------------ [A] 비트 동일
def _resample_seq(mod, kw, N=512):
    torch.manual_seed(7); env = _Env(N)
    mod.install_w3_cmd(env, None, p_step=0.7, **kw)
    seq = []
    for i in range(40):
        ids = torch.randperm(N)[: 1 + (i * 37) % 300]
        env.command_manager.t._resample_command(ids)
        seq.append((env.command_manager.t.vel_command_b.clone(), env._ge_cmd.clone(),
                    getattr(env, "_w3_wz0", torch.zeros(1)).clone(), getattr(env, "_w3_hl", torch.zeros(1)).clone()))
    return seq, hasattr(env, "_w3_mix4")


def test_A(old, new):
    print("[A] Stage 0–3 비트 동일")
    fo = {n.name: ast.dump(n) for n in ast.parse(old.__src__).body if isinstance(n, ast.FunctionDef)}
    fn = {n.name: ast.dump(n) for n in ast.parse(new.__src__).body if isinstance(n, ast.FunctionDef)}
    changed = sorted(k for k in fo if fo[k] != fn.get(k))
    check(changed == ["install_w3_cmd"], f"기존 함수 중 바뀐 것 = install_w3_cmd 뿐 (실제 {changed}) — w3_gait_update(리드 발 규칙) 불변")
    S3a = {"stage": 3, "vx_back_range": (0.03, 0.06), "mix3": (0.3, 0.3, 0.0, 0.4), "wz_range": (0.1, 0.2)}
    for label, kw in (("S0", {"stage": 0}), ("S1", {"stage": 1}), ("S1b/c", {"stage": 1, "vx_back_range": (0.03, 0.06)}),
                      ("S1d", {"stage": 1, "vx_range": (0.03, 0.06), "vx_back_range": (0.03, 0.06)}),
                      ("S2", {"stage": 2, "vx_back_range": (0.03, 0.06), "mix2": (0.3, 0.3, 0.4), "vy_range": (0.03, 0.04)}),
                      ("S3", S3a), ("S3h", dict(S3a, p_hloop=0.5)),
                      ("S3s2", dict(S3a, mix3=(0.25, 0.2, 0.25, 0.3), vy_range=(0.03, 0.08)))):
        so, _ = _resample_seq(old, kw); sn, has4 = _resample_seq(new, kw)
        ok = all(all(torch.equal(x, y) for x, y in zip(a, b)) for a, b in zip(so, sn))
        check(ok and not has4, f"install_w3_cmd {label}: 40 회 리샘플 명령·ge_cmd·wz₀·고리 표시 비트 동일, Stage 4 속성 없음")
    # 단일 축·제자리 회전·명령 0 궤적: 혼합 추적기 동반 + 새 보상 3종 ≡ 기존(몸은 실제로 원호를 돌고 있어도 — 명령이 단일 축이면 기존 값)
    yaw = torch.linspace(-math.pi, math.pi, 16)
    cases = [((0.0, 0.0, 0.0), (0.0, 0.0), 0.0), ((0.06, 0.0, 0.0), (0.06, 0.0), 0.0), ((-0.04, 0.0, 0.0), (-0.04, 0.0), 0.15),
             ((0.0, 0.05, 0.0), (0.0, 0.05), -0.2), ((0.0, -0.08, 0.0), (0.0, -0.08), 0.0), ((0.0, 0.0, 0.3), (0.0, 0.0), 0.3),
             ((0.0, 0.0, -0.2), (0.02, 0.0), -0.2), ((0.01, 0.0, 0.3), (0.01, 0.0), 0.3), ((0.06, 0.0, 0.05), (0.06, 0.0), 0.05)]
    for cmd, vb_, w_ in cases:
        for foot in (0, 1):
            vb = torch.tensor(vb_).repeat(16, 1); w = torch.full((16,), w_)
            ro = curve_swing(old, torch.tensor(cmd), foot, vb, w, yaw, s4=False)
            rn = curve_swing(new, torch.tensor(cmd), foot, vb, w, yaw, s4=True)
            same = all(torch.equal(ro["env"]._w3[k], rn["env"]._w3[k]) for k in ro["env"]._w3) and \
                all(torch.equal(ro["env"]._w3lat[k], rn["env"]._w3lat[k]) for k in ro["env"]._w3lat) and \
                all(torch.equal(ro["env"]._w3turn[k], rn["env"]._w3turn[k]) for k in ro["env"]._w3turn if ro["env"]._w3turn[k] is not None)
            eq = torch.equal(rn["s4"], ro["s2"]) and torch.equal(rn["tr4"], ro["tr"]) and torch.equal(rn["dr4"], ro["dr"]) and torch.equal(rn["turn"], ro["turn"])
            check(same and eq, f"명령 {cmd}·몸 (vb {vb_}, w {w_:+.2f})·발 {foot}: 기존 상태 동일, stride4≡stride2 · track_rel4≡track_rel · drift_net4≡drift_net"
                  f" (합 {float(rn['s4'][0]):+.3f}/{float(rn['tr4'][0]):+.3f}/{float(rn['dr4'][0]):+.3f})")


# ------------------------------------------------------------------ [B] 반사실 곡선 착지
VARS = {"브리프 문자식(이륙 골반 요·R(θ/2)·지레 없음)": dict(frame="lo", rot_frac=0.5, lever=False, sinc=False),
        "지레 없음(직전 착지 프레임·R(θ/2)·sinc)": dict(frame="td", rot_frac=0.5, lever=False, sinc=True),
        "이륙 골반 프레임 정확식(회전 0·지레)": dict(frame="lo", rot_frac=0.0, lever=True, sinc=True)}


def test_B(new):
    print("[B] 반사실 곡선 착지 — 사건 합(가중 10·stride4 + 10·turn_stride), 요 스폰 0/180/40°; 판정은 한 주기(리드 + 따라오는 발) 합")
    yaw = torch.tensor([0.0, math.pi, 0.7])
    rows = []; vrows = []
    cmds = [(0.06, 0.0, 0.2), (0.06, 0.0, -0.2), (0.10, 0.0, 0.4), (-0.04, 0.0, 0.2), (0.0, 0.05, 0.2), (0.0, -0.05, 0.3),
            (0.06, 0.04, -0.3), (-0.04, -0.05, -0.2), (0.03, 0.0, 0.4)]
    singles = ["직선 병진만(발 요 0)", "직선 병진 + 발 요(축 단순 합)", "제자리 회전(병진 0)"]
    for cmd in cmds:
        vx, vy, wz = cmd
        lead = 1 if (vy < -0.02 or (abs(vy) < 0.02 and wz < -0.08)) else 0
        cyc = {}
        for foot, fname in ((lead, "리드"), (1 - lead, "따라옴")):
            vb = torch.tensor([vx, vy]).repeat(3, 1); w = torch.full((3,), wz); c = torch.tensor(cmd)
            th = wz * T_CYC
            straight = torch.tensor([vx, vy]) * T_CYC
            pf = torch.tensor([0.0, W_HALF if foot == 0 else -W_HALF]).expand(3, 2)
            XTm, psTm = arc(torch.zeros(3), -vb, -w, T_CYC)                     # 반대 = 전 축 거울 정상 상태 착지
            Dopp = XTm + rot(psTm, pf) - pf
            XTr, psTr = arc(torch.zeros(3), vb, -w, T_CYC)
            Drev = XTr + rot(psTr, pf) - pf
            cases = {"맞는 곡선(강체 정상 상태)": dict(),
                     "직선 병진만(발 요 0)": dict(D=straight.repeat(3, 1), dpsi=torch.zeros(3)),
                     "직선 병진 + 발 요(축 단순 합)": dict(D=straight.repeat(3, 1)),
                     "제자리 회전(병진 0)": dict(D=torch.zeros(3, 2)),
                     "회전 방향만 반대": dict(D=Drev, dpsi=torch.full((3,), -th)),
                     "제자리 디딤": dict(D=torch.zeros(3, 2), dpsi=torch.zeros(3)),
                     "반대(전 축)": dict(D=Dopp, dpsi=torch.full((3,), -th))}
            for name, kw in cases.items():
                o = curve_swing(new, c, foot, vb, w, yaw, variants=VARS if name.startswith("맞는") else None, **kw)
                tot = 10 * o["s4"] + 10 * o["turn"]
                cyc[name] = cyc.get(name, 0) + tot
                rows.append((cmd, fname, name, float(o["s4"][0]), float(o["turn"][0]), float(tot[0]), float(tot.std()), float(o["W"][0]), float(o["bad"][0])))
                if name.startswith("맞는"):
                    vrows.append((cmd, fname, 100 * float(o["var"]["채택식"][0]), *[100 * float(o["var"][k][0]) for k in VARS]))
                    n2 = float((new.w3_curve_target(torch.tensor([cmd]), torch.tensor([F_STEP]), W_HALF)[0, foot] ** 2).sum())
                    ideal = 1.0 - math.exp(-n2 / 0.03 ** 2)
                    check(float(o["bad"].sum()) == 0.0 and bool((o["s4"] >= 0.97 * ideal - 1e-4).all()) and bool(((tot - tot[0]).abs() < 1e-3).all()),
                          f"{cmd}·{fname}: 맞는 곡선 착지 — 폭 위반 0(W {float(o['W'][0]):.3f} m) · stride4 raw {float(o['s4'][0]):+.3f} ≈ 상한 {ideal:+.3f} · 요 스폰 무관")
                if name == "제자리 디딤":
                    check(bool((tot.abs() < 1e-6).all()), f"{cmd}·{fname}: 제자리 디딤 = 0")
        good = cyc["맞는 곡선(강체 정상 상태)"]
        best_single = torch.stack([cyc[k] for k in singles]).max(0).values
        worst_single = torch.stack([cyc[k] for k in singles]).min(0).values
        opp = cyc["반대(전 축)"]; rev = cyc["회전 방향만 반대"]
        check(bool((good - best_single >= 2.0).all()) and bool((worst_single - opp >= 2.0).all()) and bool((good - rev >= 2.0).all()),
              f"{cmd} 한 주기: 맞는 곡선 {float(good[0]):+.2f} ≫ 단일 축만 맞춤 최고 {float(best_single[0]):+.2f} · 최저 {float(worst_single[0]):+.2f}"
              f" ≫ 반대 {float(opp[0]):+.2f} (여유 ≥ 2); 회전 방향만 반대 {float(rev[0]):+.2f}")
    print("\n  | 명령 (vx,vy,wz) | 발 | 착지 | stride4 raw | turn raw | 합(가중) | 요 스폰 표준편차 | 착지 발 간격 W [m] | 폭 위반 |")
    print("  |---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"  | {r[0]} | {r[1]} | {r[2]} | {r[3]:+.3f} | {r[4]:+.3f} | {r[5]:+.2f} | {r[6]:.1e} | {r[7]:.3f} | {r[8]:.0f} |")
    print("\n  목표식 변형 — 맞는 곡선(강체 정상 상태) 착지에 대한 목표 오차 ‖Δp − ΔP*‖ [cm] (σ 3 cm; 0 = 그 식이 정확한 착지를 목표로 삼음)")
    print("  | 명령 | 발 | 채택식 | " + " | ".join(VARS) + " |")
    print("  |---|---|---|" + "---|" * len(VARS))
    for r in vrows:
        print(f"  | {r[0]} | {r[1]} | {r[2]:.2f} | " + " | ".join(f"{x:.2f}" for x in r[3:]) + " |")
    check(all(r[2] < 0.1 for r in vrows), "채택식 목표 오차 < 0.1 cm(모든 경우 — 원호 적분과 독립 일치)")
    big = [r for r in vrows if abs(r[0][2]) >= 0.3]
    check(all(r[3] >= 1.0 for r in big), "|wz| ≥ 0.3 에서 브리프 문자식 오차 ≥ 1 cm(지레 누락 + 이륙 골반 프레임 이중 회전) — "
          + ", ".join(f"{r[0]}·{r[1]} {r[3]:.1f}" for r in big))


# ------------------------------------------------------------------ [C] 곡선 병진 추종
def test_C(new):
    print("[C] 곡선 병진 추종 — 12 s 원호 운동, 초당 평균 raw(앞 2 s 제외), 요 스폰 4 종")
    yaw0 = torch.tensor([0.0, math.pi, 0.7, -2.0])
    rows = []
    for cmd in ((0.06, 0.0, 0.3), (0.10, 0.0, 0.4), (0.06, 0.04, -0.3), (-0.04, 0.0, 0.2), (0.0, 0.05, 0.2)):
        vx, vy, wz = cmd
        behs = {"정확한 원호": ((vx, vy), wz), "직선(회전 없음)": ((vx, vy), 0.0), "제자리 회전": ((0.0, 0.0), wz),
                "병진 반대·회전 맞음": ((-vx, -vy), wz), "가만히": ((0.0, 0.0), 0.0)}
        res = {}
        for name, (vb_, w_) in behs.items():
            N = 4; env = _Env(N, yaw0); env._ge = torch.ones(N, dtype=torch.bool)
            env.command_manager.t.vel_command_b[:] = torch.tensor(cmd); env.episode_length_buf[:] = 0
            vb = torch.tensor(vb_).repeat(N, 1); w = torch.full((N,), w_)
            acc = {k: torch.zeros(N) for k in ("tr4", "tr", "dr4", "dr", "yn")}; n = 0
            for i in range(int(12.0 / DT)):
                t = (i + 1) * DT
                env.episode_length_buf += 1; env.common_step_counter += 1
                X, psi = arc(yaw0, vb, w, t)
                env.data.root_quat_w = qz(psi); env.data.root_pos_w = torch.cat([X, torch.zeros(N, 1)], 1)
                env.data.root_lin_vel_w = torch.cat([rot(psi, vb), torch.zeros(N, 1)], 1)
                a = {"tr4": new.w3_track_rel4(env), "tr": new.w3_track_rel(env), "dr4": new.w3_drift_net4(env), "dr": new.w3_drift_net(env), "yn": new.w3_yaw_net(env)}
                if t > 2.0:
                    for k in acc:
                        acc[k] += a[k]
                    n += 1
            res[name] = {k: float((v / n).mean()) for k, v in acc.items()}
            res[name]["tot"] = 3 * res[name]["tr4"] - 3 * res[name]["dr4"] - 4 * res[name]["yn"]
            rows.append((cmd, name, res[name]["tr4"], res[name]["tr"], res[name]["dr4"], res[name]["dr"], res[name]["tot"]))
        ex = res["정확한 원호"]
        cn = math.hypot(vx, vy); ideal = 1.0 - math.exp(-(cn / 0.05) ** 2)
        check(ex["dr4"] < 2e-3 and abs(ex["tr4"] - ideal) < 5e-3,
              f"{cmd}: 정확한 원호 새 식 순 표류 {ex['dr4']:.4f} ≈ 0 · 추종 {ex['tr4']:+.3f} ≈ 상한 {ideal:+.3f} (기존 식 표류 {ex['dr']:.4f} · 추종 {ex['tr']:+.3f})")
        st, ip, op = res["직선(회전 없음)"], res["제자리 회전"], res["병진 반대·회전 맞음"]
        check(abs(st["tr4"] - ex["tr4"]) < 5e-3 and abs(ip["tr4"]) < 0.02 and op["tr4"] < 0 and abs(res["가만히"]["tr4"]) < 1e-6,
              f"{cmd}: 병진 추종은 회전과 분리 — 원호 {ex['tr4']:+.3f} = 직선 {st['tr4']:+.3f}(실제 요 경로 기준), 제자리 회전 {ip['tr4']:+.4f}(≈ 0; sinc 기준선 차), 병진 반대 {op['tr4']:+.3f} < 0")
        check(ex["tot"] > max(st["tot"], ip["tot"]) + 0.3 and min(st["tot"], ip["tot"]) > op["tot"] + 0.3,
              f"{cmd}: 합(+3·추종 −3·표류 −4·순 요) 원호 {ex['tot']:+.2f} ≫ 단일 축(직선 {st['tot']:+.2f} · 제자리 회전 {ip['tot']:+.2f}) ≫ 병진 반대 {op['tot']:+.2f}")
    print("\n  | 명령 | 몸 운동 | track_rel4 | track_rel(기존) | drift_net4 [m/s] | drift_net(기존) | 합 +3·tr4 −3·dr4 −4·yaw_net [/s] |")
    print("  |---|---|---|---|---|---|---|")
    for r in rows:
        print(f"  | {r[0]} | {r[1]} | {r[2]:+.3f} | {r[3]:+.3f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:+.3f} |")


# ------------------------------------------------------------------ [D] 명령 분포
def test_D(new):
    print("[D] Stage 4 명령 분포")
    mix4 = (0.2, 0.15, 0.15, 0.2, 0.3); pat = (0.4, 0.2, 0.15, 0.25)
    R = {"vx": (0.03, 0.10), "vb": (0.03, 0.06), "vy": (0.03, 0.08), "wz": (0.1, 0.4)}
    for s in (0.0, 0.5, 1.0):
        N = 300000; torch.manual_seed(13); env = _Env(N)
        new.install_w3_cmd(env, None, p_step=0.7, stage=4, vx_range=R["vx"], vx_back_range=R["vb"], vy_range=R["vy"], wz_range=R["wz"],
                           mix4=mix4, mix_pat=pat, mix_scale=s, p_hloop=0.5)
        env.command_manager.t._resample_command(torch.arange(N))
        c = env.command_manager.t.vel_command_b; ge = env._ge_cmd; g = c[ge]
        fr = lambda m: float(m.float().mean())
        nz = g != 0; k = nz.sum(1)
        inpl = k == 0; single = k == 1; multi = k >= 2
        cat = (fr(inpl), fr(single & nz[:, 0]), fr(single & nz[:, 1]), fr(single & nz[:, 2]), fr(multi))
        check(abs(fr(ge) - 0.7) < 0.01 and all(abs(x - y) < 0.01 for x, y in zip(cat, mix4)) and bool((c[~ge] == 0).all()),
              f"s={s}: STEP {fr(ge):.3f} · 제자리/앞뒤/옆/회전/혼합 " + "/".join(f"{x:.3f}" for x in cat) + f" (목표 {mix4}), STAND 명령 0")
        m = g[multi]; mz = m != 0
        pats = (fr(mz[:, 0] & ~mz[:, 1] & mz[:, 2]), fr(mz[:, 0] & mz[:, 1] & ~mz[:, 2]), fr(~mz[:, 0] & mz[:, 1] & mz[:, 2]), fr(mz.all(1)))
        check(all(abs(x - y) < 0.01 for x, y in zip(pats, pat)), f"s={s}: 혼합 조합 앞뒤+회전/앞뒤+옆/옆+회전/셋 다 " + "/".join(f"{x:.3f}" for x in pats) + f" (목표 {pat})")
        sx = m[mz[:, 0], 0]; sy = m[mz[:, 1], 1]; sw = m[mz[:, 2], 2]
        both = mz[:, 0] & mz[:, 2]
        corr = float((torch.sign(m[both, 0]) * torch.sign(m[both, 2])).mean())
        check(all(abs(fr(x > 0) - 0.5) < 0.01 for x in (sx, sy, sw)) and abs(corr) < 0.02,
              f"s={s}: 혼합 부호 양수 비율 x {fr(sx > 0):.3f} · y {fr(sy > 0):.3f} · wz {fr(sw > 0):.3f}, vx·wz 부호 상관 {corr:+.3f}(독립)")
        hi = lambda rg: rg[0] + s * (rg[1] - rg[0])
        okr = bool(((sx[sx > 0] >= R["vx"][0] - 1e-6) & (sx[sx > 0] <= hi(R["vx"]) + 1e-6)).all()) and \
            bool(((-sx[sx < 0] >= R["vb"][0] - 1e-6) & (-sx[sx < 0] <= hi(R["vb"]) + 1e-6)).all()) and \
            bool(((sy.abs() >= R["vy"][0] - 1e-6) & (sy.abs() <= hi(R["vy"]) + 1e-6)).all()) and \
            bool(((sw.abs() >= R["wz"][0] - 1e-6) & (sw.abs() <= hi(R["wz"]) + 1e-6)).all())
        check(okr, f"s={s}: 혼합 크기 상한 = lo + s·(hi − lo): 앞 ≤ {hi(R['vx']):.3f} · 뒤 ≤ {hi(R['vb']):.3f}(뒤 범위) · |vy| ≤ {hi(R['vy']):.3f} · |wz| ≤ {hi(R['wz']):.3f}"
              f" (실제 최대 {float(sx.max()):.3f}/{float(-sx.min()):.3f}/{float(sy.abs().max()):.3f}/{float(sw.abs().max()):.3f})")
        sg1 = g[single]
        okS = bool(((sg1[:, 2].abs() == 0) | ((sg1[:, 2].abs() >= 0.1) & (sg1[:, 2].abs() <= 0.4))).all())
        check(okS, f"s={s}: 단일 축 범주는 크기 커리큘럼과 무관하게 단일 축 범위 전체(Stage 3 식)")
        hl = env._w3_hl
        exp_hl = 0.5 * fr(ge & (c[:, 2] == 0))
        check(abs(fr(hl) - exp_hl) < 0.01 and not bool((hl & (c[:, 2] != 0)).any()) and not bool((hl & ~ge).any()) and torch.equal(env._w3_wz0, c[:, 2]),
              f"s={s}: 고리 표시 {fr(hl):.3f}(기대 {exp_hl:.3f}) — wz ≠ 0·STAND 에는 없음, wz₀ = 리샘플 wz")
    env._w3_mix_scale = 0.25; env.command_manager.t._resample_command(torch.arange(N))
    g = env.command_manager.t.vel_command_b[env._ge_cmd]; m = g[(g != 0).sum(1) >= 2]
    check(float(m[:, 2].abs().max()) <= 0.1 + 0.25 * 0.3 + 1e-6, f"env._w3_mix_scale 변경(0.25)이 다음 리샘플 혼합 크기에 반영(|wz| 최대 {float(m[:, 2].abs().max()):.3f})")
    os.environ["W3_CMD"] = "0.06,0.03,-0.2"
    try:
        env = _Env(64); new.install_w3_cmd(env, None, stage=4); env.command_manager.t._resample_command(torch.arange(64))
        check(bool((env.command_manager.t.vel_command_b == torch.tensor([0.06, 0.03, -0.2])).all()) and bool((env._w3_wz0 == -0.2).all()),
              "W3_CMD='0.06,0.03,-0.2' 고정 → 명령·wz₀ 반영")
    finally:
        del os.environ["W3_CMD"]


# ------------------------------------------------------------------ [E] 리드 발 충돌
def test_E(new):
    print("[E] 리드 발 충돌 해소 — 옆 우선(w3_gait_update 코드 불변, lead_vy 0.02·lead_wz 0.08)")
    combos = [((0.06, 0.0, 0.3), 0.0, "앞+왼 회전 → 왼(회전 규칙)"), ((0.06, 0.0, -0.3), 0.5, "앞+오른 회전 → 오른(회전 규칙)"),
              ((0.0, 0.05, 0.3), 0.0, "왼 옆+왼 회전 → 왼(일치)"), ((0.0, -0.05, -0.3), 0.5, "오른 옆+오른 회전 → 오른(일치)"),
              ((0.0, 0.05, -0.3), 0.0, "왼 옆+오른 회전 → 왼(충돌: 옆 우선)"), ((0.0, -0.05, 0.3), 0.5, "오른 옆+왼 회전 → 오른(충돌: 옆 우선)"),
              ((0.06, 0.04, -0.3), 0.0, "앞+왼 옆+오른 회전 → 왼(충돌: 옆 우선)"), ((-0.04, -0.03, 0.2), 0.5, "뒤+오른 옆+왼 회전 → 오른(충돌: 옆 우선)"),
              ((0.06, 0.01, -0.3), 0.5, "옆 문턱(0.02) 아래 + 오른 회전 → 오른")]
    env = _Env(len(combos))
    env.command_manager.t.vel_command_b[:] = torch.tensor([c for c, _, _ in combos])
    env._ge_cmd = torch.ones(len(combos), dtype=torch.bool); env.episode_length_buf[:] = 1
    new.w3_gait_update(env, None, f_step=F_STEP, lead_vy_cmin=0.02, lead_wz_cmin=0.08)
    for (c, exp, lab), u in zip(combos, env._gu.tolist()):
        check(u == exp, f"{c}: 시작 위상 {u} — {lab}")
    # 근거 수치: 옆 명령에서 리드를 거꾸로(따라오는 발 먼저) 하면 첫 걸음이 발 간격을 |vy|/f 만큼 좁힘(문턱 2·0.144 − 0.05 = 0.238 m)
    print("\n  | |vy| | 옆 리드 지킴: 첫 착지 간격 | 거꾸로: 첫 착지 간격 | 거꾸로 폭 위반 | 회전만의 횡 성분 w(1 − cos θ) @ wz 0.4 |")
    print("  |---|---|---|---|---|")
    lat_rot = W_HALF * (1 - math.cos(0.4 * T_CYC))
    viol = []
    for vy in (0.03, 0.05, 0.06, 0.08):
        d = vy * T_CYC
        good, bad = 2 * W_HALF + d, 2 * W_HALF - d
        viol.append((vy, bad < 0.238))
        print(f"  | {vy:.2f} | {good:.3f} m | {bad:.3f} m | {'위반' if bad < 0.238 else '—'} | {lat_rot * 100:.2f} cm |")
    check([v for _, v in viol] == [False, False, True, True], f"거꾸로 리드 폭 위반: |vy| ≥ 0.06 에서 (첫 걸음 간격 < 0.238 m) — 옆 리드 위반 비용은 크기에 비례")
    check(lat_rot < 0.03 * T_CYC, f"회전 리드 위반 비용: 회전이 만드는 횡 변위 {lat_rot * 100:.2f} cm < 최소 옆 보폭 {0.03 * T_CYC * 100:.2f} cm, "
          "발 요 착지 보상은 따라오는 발도 같은 방향이면 + (unit_w3_s3 [B]) → 회전은 어느 발로도 시작 가능, 옆은 아님")


# ------------------------------------------------------------------ [F] 커리큘럼
def test_F(new):
    print("[F] 혼합 크기 커리큘럼")
    env = _Env(8); M = new._mix(env)
    seq = []
    blocks = [(0.85, 0.95, 0.95, 0.0), (0.95, 0.95, 0.95, 0.03), (0.95, 0.92, 0.91, 0.01), (0.95, 0.95, 0.95, 0.0), (0.99, 0.99, 0.99, 0.0)]
    for it, (sx, sy, sw, bad) in enumerate(blocks):
        env.common_step_counter = (it + 1) * 120 * 24
        M["n_td"] += 4000; M["n_x"] += 3000; M["ok_x"] += 3000 * sx; M["n_y"] += 1500; M["ok_y"] += 1500 * sy
        M["n_w"] += 2500; M["ok_w"] += 2500 * sw; M["n_bad"] += 4000 * bad
        out = new.w3_mix_curriculum(env, None)
        seq.append(out["mix_scale"])
    check(seq == [0.0, 0.0, 0.5, 1.0, 1.0], f"크기 순서 {seq} (x 부호 0.85 정체 · 폭 위반 0.03 정체 · 이후 0.5 → 1, 최대 유지)")
    env = _Env(8); M = new._mix(env); env.common_step_counter = 5000
    M["n_td"] += 100
    out = new.w3_mix_curriculum(env, None, start_stage=1)
    check(out["mix_scale"] == 0.5 and env._w3_mix_scale == 0.5, "start_stage 1 → 0.5, 블록 미달(혼합 착지 < 3000)이면 판정 없음")


# ------------------------------------------------------------------ [G] 혼합 추적기 부호 지표
def test_G(new):
    print("[G] 혼합 추적기 축별 부호 지표(지레·회전 보정 뒤)")
    yaw = torch.tensor([0.0, math.pi, 0.7])
    for cmd in ((0.06, 0.0, 0.2), (0.03, 0.0, 0.4), (0.06, 0.04, -0.3), (0.0, -0.05, 0.3), (-0.04, -0.05, -0.2)):
        vx, vy, wz = cmd
        vb = torch.tensor([vx, vy]).repeat(3, 1); w = torch.full((3,), wz)
        tot = {}
        for name, kw in (("맞는 곡선", {}), ("반대(전 축)", "opp")):
            agg = {k: 0.0 for k in ("n_td", "n_x", "ok_x", "n_y", "ok_y", "n_w", "ok_w")}
            for foot in (0, 1):
                if kw == "opp":
                    pf = torch.tensor([0.0, W_HALF if foot == 0 else -W_HALF]).expand(3, 2)
                    XTm, psTm = arc(torch.zeros(3), -vb, -w, T_CYC)
                    o = curve_swing(new, torch.tensor(cmd), foot, vb, w, yaw, D=XTm + rot(psTm, pf) - pf, dpsi=torch.full((3,), -wz * T_CYC))
                else:
                    o = curve_swing(new, torch.tensor(cmd), foot, vb, w, yaw)
                for k in agg:
                    agg[k] += float(o["env"]._w3mix[k].sum())
            tot[name] = agg
        g, b = tot["맞는 곡선"], tot["반대(전 축)"]
        rate = lambda d, a: d["ok_" + a] / d["n_" + a] if d["n_" + a] > 0 else float("nan")
        gr = [rate(g, a) for a in "xyw"]; br = [rate(b, a) for a in "xyw"]
        act = [abs(vx) >= 0.02, abs(vy) >= 0.02, abs(wz) >= 0.08]
        check(g["n_td"] == 6 and all((r == 1.0) if a else (r != r) for r, a in zip(gr, act)) and all((r == 0.0) if a else (r != r) for r, a in zip(br, act)),
              f"{cmd}: 맞는 곡선 부호 x/y/요 " + "/".join(f"{r:.2f}" for r in gr) + " · 반대 " + "/".join(f"{r:.2f}" for r in br)
              + " (안쪽 발이 뒤로 가는 작은 반경에서도 지레 보정으로 x 부호 정답)")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default="2aeb91cd")
    a = ap.parse_args()
    src_old = subprocess.run(["git", "-C", REPO, "show", f"{a.ref}:rl_walking/w3.py"], check=True, capture_output=True, text=True).stdout
    src_new = open(os.path.join(REPO, "rl_walking", "w3.py")).read()
    old = S3._load(src_old, "s4old"); old.__src__ = src_old
    new = S3._load(src_new, "s4new"); new.__src__ = src_new
    torch.set_default_dtype(torch.float32)
    test_A(old, new); test_B(new); test_C(new); test_D(new); test_E(new); test_F(new); test_G(new)
    print(f"\n결과: 실패 {len(FAILS)}")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
