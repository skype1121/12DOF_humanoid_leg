"""1007 병목 2 odic 배포 연결 동치 검증 — PC 전용(torch CPU 필요, GPU·Isaac 불필요). 젯슨 브리지 selftest 와 별개.

  python3 rl_walking/deploy/selftest_odic_equiv.py [--ref f44e8701]

검사:
  [A] 기본(imu) 경로 비트 동일: --ref 커밋의 c5_switch.py(`git show`)를 별도 모듈로 올려, 같은 입력열(켜짐·조기 복귀·불응기·cue 포함)에서
      SwitchDetector 모드·카운터·기준 자세와 C5Runner 목표각·관측·블랙박스를 전 틱 비트 비교(KE·krF·C5 RV4 설정 3 종).
  [B] 시뮬 소스 = 배포 구현: eval_ground_recovery.py 의 감지 블록(`if policy2 is not None and i >= STEPS_PRE:` ~
      `sw_mode = (sw_mode | _new_on) & ~_swoff`)을 **소스 그대로** 잘라 torch CPU 로 실행(가짜 robot·uenv·obs, 다른 기능 플래그 꺼짐)하고,
      같은 입력 시계열을 배포 SwitchDetector(settle=odic) N 개에 넣는다. 입력열 = 조용한 서기 → 기울기 펄스(켜짐) → 다리 운동(회복 스텝 모사:
      힙·무릎 진동 + 접촉 교대 + 무릎 토크 변동) → 정착, 펄스 크기·시각·운동 길이 env 별 랜덤 + τ 잡음(첫 틱 0).
      대조: 매 틱 모드(전 env), |v̄|(float32 비트), 유효비, 양발 연속 시간, 조기 복귀 수, 켜짐 수.
      팔 2 개: odic(주입 42 + 보상 42, 기준 = 인계 토크 → 배포 tau_ref 'handover') · odoc(주입 없음 + 보상 42 → 시뮬은 절대 τ/k̂ → 'absolute').
      민감도 확인(1007, 수동): t_on 문턱의 −1e-6 제거·τ 부호 반대 뮤턴트를 [B] 가 각각 2 건·9584 건 불일치로 잡는다.
  [C] 브리지 코어 기본 경로(--c5-settle 생략) = --ref 커밋 브리지와 비트 동일(실모델 C5 kr·C5·G21a, 기울기 펄스로 전환 포함).
  [B'] (1007 odicw) 시뮬 --settle_w 0.4 = 배포 settle w_thr 0.4: 입력열에 '다리 정지 + 몸통 진동'(양발 지지 진동 모사, 평균 0 인 자이로 3–6 Hz
      0.5–2 rad/s) 구간을 더해 |ω_xy| 조건이 실제로 조기 복귀를 막는 틱(n_w_block > 0)을 덮는다. 진동 난수는 별도 RNG(기존 [B] 입력열 불변).
  [D] (1007) odic(w_thr 0) 배포 경로 = --ref-odic 커밋(28657c3f, odicw 도입 전) c5_switch 의 odic 와 비트 동일(진동 입력열 포함, 모드·|v̄|·블랙박스).
      민감도 확인(1007, 수동): odicw 를 필터 전 |ω| 로 바꾼 뮤턴트를 [B'] 가 잡는지 — STOPEST §11 기록.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import textwrap
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rl_walking.deploy import c5_switch as C5  # noqa: E402
from rl_walking.deploy.policy_runner import DEFAULT_POSE_RAD, JOINT_ORDER  # noqa: E402

KE = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5, off_contact="any",
          early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33, early_gate=(5.0, 1.0))
#: settle_ab_run.sh BASE 의 감지 인자와 같은 값(--sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 --sw_off_contact any
#: --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33)
SIM_ARGS = dict(det_from_obs=True, det_filt=0.33, contact_drop="", sw_ref_tau=10.0, sw_ref_init_s=1.5, settle_judge="odo", settle_v=0.07,
                settle_ds=0.3, settle_min=0.9, sw_off_contact="any", sw_early_imu="0.6,6,0.4,0.2", early_gate="5,1.0", sw_refrac="1.5,6,1.0",
                sw_early="", ho_state="", sw_guard="", ho_at_load="", sw_oracle=False, settle_w=0.0)
EVAL = os.path.join(REPO, "rl_walking", "scripts", "eval_ground_recovery.py")
BEGIN = "        if policy2 is not None and i >= STEPS_PRE:\n"
END = "            swc_n += (_new_on & alive).float(); sw_mode = (sw_mode | _new_on) & ~_swoff\n"


def _git_show(ref, path):
    return subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"], check=True, capture_output=True, text=True).stdout


# ------------------------------------------------------------------ [A] 기본(imu) 경로 비트 동일
def _load_ref_c5(ref, expect_odic=False):
    src = _git_show(ref, "rl_walking/deploy/c5_switch.py")
    if expect_odic:
        assert "OdicSettle" in src and "w_thr" not in src, f"{ref} 의 c5_switch 가 odic 有·odicw 無 커밋이 아니다"
    else:
        assert "OdicSettle" not in src, f"{ref} 의 c5_switch 에 이미 odic 가 있다 — 기준 커밋 확인"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=os.path.join(REPO, "rl_walking", "deploy"))
    tmp.write(src); tmp.close()
    try:
        name = "rl_walking.deploy." + os.path.splitext(os.path.basename(tmp.name))[0]
        spec = importlib.util.spec_from_file_location(name, tmp.name)
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.unlink(tmp.name)
    return mod


def _imu_inputs(rng, k, pulses):
    tilt = 0.6 + 0.3 * math.sin(0.05 * k)
    for t0, amp, dur in pulses:
        if t0 <= k < t0 + dur:
            tilt += amp * math.sin(math.pi * (k - t0) / dur)
    th = math.radians(tilt)
    grav = np.array([math.sin(th), 0.02 * math.sin(0.11 * k), -math.cos(th)], np.float32)
    grav /= np.linalg.norm(grav)
    gyro = rng.normal(0, 0.05, 3).astype(np.float32)
    for t0, amp, dur in pulses:
        if t0 <= k < t0 + dur:
            gyro[1] += np.float32(0.15 * amp * math.cos(math.pi * (k - t0) / dur))
    contact = np.array([1.0, 1.0 if (k // 23) % 5 else 0.0], np.float32)
    qpos = (np.asarray(DEFAULT_POSE_RAD) + rng.normal(0, 0.02, 12)).astype(np.float32)
    qvel = rng.normal(0, 0.1, 12).astype(np.float32)
    return gyro, grav, qpos, qvel, contact


def check_default_bitident(ref):
    old = _load_ref_c5(ref)
    W = np.random.default_rng(11).normal(0, 0.05, (12, 320)).astype(np.float32)
    stand = lambda o: np.tanh(W @ o)            # noqa: E731
    rec = lambda o: 0.3 - np.tanh(W @ o)        # noqa: E731
    cfgs = {"KE(c5kr)": (dict(KE), dict(cue_vx=0.03, cue_s=1.0, cue_dir_imu=True)),
            "C5 RV4": (dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5,
                            off_contact="any"), {}),
            "절대·조기": (dict(early=(0.6, 6.0, 0.4, 0.2)), {})}
    pulses = [(120, 7.0, 25), (300, 4.5, 12), (330, 9.0, 30), (600, 6.0, 40), (900, 12.0, 20)]
    summ = []
    for nm, (dk, rk) in cfgs.items():
        kw = dict(obs_dim=320, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
        a = C5.C5Runner(stand, rec, detector=C5.SwitchDetector(**dk), **rk, **kw)
        b = old.C5Runner(stand, rec, detector=old.SwitchDetector(**dk), **rk, **kw)
        ra, rb = np.random.default_rng(3), np.random.default_rng(3)
        n_sw = 0
        for k in range(1200):
            if k == 700:
                a.reset(); b.reset()
            ga, gra, qa, va, ca = _imu_inputs(ra, k, pulses); gb, grb, qb, vb, cb = _imu_inputs(rb, k, pulses)
            ya = a.step(ga, gra, (0.0, 0.0, 0.0), qa, va, ca)
            yb = b.step(gb, grb, (0.0, 0.0, 0.0), qb, vb, cb)
            assert np.array_equal(np.asarray(ya).view(np.uint32), np.asarray(yb).view(np.uint32)), f"[A] {nm} 틱 {k} 목표각 불일치"
            assert np.array_equal(a.core.last_obs, b.core.last_obs), f"[A] {nm} 틱 {k} 관측 불일치"
            assert a.blackbox() == b.blackbox(), f"[A] {nm} 틱 {k} 블랙박스 불일치"
            n_sw += a.last_switch != ""
        da, db = a.det, b.det
        assert (da.n_on, da.n_early, da.mode) == (db.n_on, db.n_early, db.mode) and "odic" not in a.blackbox()
        assert not a.needs_knee_tau
        summ.append(f"{nm} 전환 {n_sw}·조기 {da.n_early}")
    return f"[A] 기본(imu) 경로 = {ref} c5_switch 비트 동일 × 1200 틱(중간 리셋): " + " · ".join(summ)


# ------------------------------------------------------------------ [B] 시뮬 소스 = 배포 구현
def _sim_block():
    lines = open(EVAL, encoding="utf-8").read().splitlines(keepends=True)
    i0 = lines.index(BEGIN); i1 = lines.index(END, i0)
    return compile(textwrap.dedent("".join(lines[i0:i1 + 1])), EVAL + f":{i0 + 1}-{i1 + 1}", "exec"), (i0 + 1, i1 + 1)


class _NS(types.SimpleNamespace):
    pass


def _scenario(N, T, seed, osc=False):
    """env 별 입력 시계열(넘파이 float32): 기울기·자이로·접촉·엔코더(링크각 + 편향)·무릎 적용 토크·τ 잡음.
    osc(1007 odicw): 사건 뒤 다리 정지 구간에 몸통 진동(자이로 x·y 3–6 Hz, 0.5–2 rad/s, 0.5–2 s)을 별도 RNG 로 더한다."""
    rng = np.random.default_rng(seed)
    rng_o = np.random.default_rng(seed + 1000)
    q0 = np.asarray(DEFAULT_POSE_RAD, np.float64)
    J = {n: JOINT_ORDER.index(n) for n in JOINT_ORDER}
    G = np.zeros((T, N, 3), np.float32); W = np.zeros((T, N, 3), np.float32); Cc = np.ones((T, N, 2), np.float32)
    Q = np.zeros((T, N, 12), np.float32); TAU = np.zeros((T, N, 12), np.float32)
    for e in range(N):
        n_ev = rng.integers(1, 4); t_ev = sorted(rng.integers(90, T - 200, n_ev)); amp = rng.uniform(3.0, 12.0, n_ev)
        mv = rng.uniform(0.3, 2.5, n_ev) * (rng.random(n_ev) > 0.4)      # 다리 운동 길이 [s] — 40 % 는 0(제자리 버팀: 켜진 뒤 0.9 s 최소 시간 경계를 덮는다)
        fr = rng.uniform(1.0, 2.0); bias = rng.normal(0, 0.03, 12)
        tilt = np.zeros(T); roll = np.zeros(T); gy = rng.normal(0, 0.04, (T, 3)); moving = np.zeros(T, bool)
        for t0, a, m in zip(t_ev, amp, mv):
            d = int(rng.integers(8, 30))
            k = np.arange(t0, min(T, t0 + d)); tilt[k] += a * np.sin(np.pi * (k - t0) / d)
            gy[k, 1] += 0.012 * a * np.pi / (d * 0.02) * np.cos(np.pi * (k - t0) / d) * 0.6
            moving[t0 + d // 2:min(T, t0 + d // 2 + int(m * 50))] = True
            k2 = np.arange(t0 + d, min(T, t0 + d + 40)); tilt[k2] += rng.uniform(-2.5, 2.5) * np.exp(-(k2 - t0 - d) / 15.0)
            if rng.random() < 0.6:      # 회복 뒤 잔류 기울기 1.6–2.8°(조용함 1.5° 초과 → calm 경로 막힘, 켜기 3° 미만) — odic 조기 복귀 경로를 덮는다
                k3 = np.arange(t0 + d, min(T, t0 + d + int(rng.uniform(1.0, 3.0) * 50))); tilt[k3] += rng.uniform(1.6, 2.8)
        if osc:      # 양발 지지 진동 모사: 평균 속도 0(다리 정지) + 큰 |ω| — odic 는 '정착', odicw 는 막아야 하는 상태
            for t0 in t_ev:
                if rng_o.random() < 0.8:
                    s0 = int(t0 + rng_o.integers(40, 90)); L = int(rng_o.uniform(0.5, 2.0) * 50)
                    k4 = np.arange(s0, min(T, s0 + L)); f_o = rng_o.uniform(3.0, 6.0); a_o = rng_o.uniform(0.5, 2.0); ph0 = rng_o.uniform(0, 2 * np.pi)
                    gy[k4, 0] += a_o * np.sin(2 * np.pi * f_o * 0.02 * (k4 - s0) + ph0)
                    gy[k4, 1] += 0.7 * a_o * np.cos(2 * np.pi * f_o * 0.02 * (k4 - s0) + ph0)
        tilt += 0.4 * np.sin(0.03 * np.arange(T) + e); roll += 0.5 * np.sin(0.021 * np.arange(T) + 2 * e)
        th, ph = np.radians(tilt), np.radians(roll)
        g = np.stack([np.sin(th), np.sin(ph) * np.cos(th), -np.cos(th) * np.cos(ph)], -1) + rng.normal(0, 0.004, (T, 3))
        G[:, e] = g; W[:, e] = gy
        ph_t = 2 * np.pi * fr * 0.02 * np.cumsum(moving)
        q = np.tile(q0, (T, 1)) + rng.normal(0, 0.0015, (T, 12))
        amp_h = np.where(moving, 0.18, 0.0)
        q[:, J["left_hip_f_joint"]] += amp_h * np.sin(ph_t); q[:, J["right_hip_f_joint"]] += amp_h * np.sin(ph_t)
        q[:, J["left_knee_joint"]] += np.where(moving, -0.15, 0.0) * np.maximum(0, np.sin(ph_t))
        q[:, J["right_knee_joint"]] += np.where(moving, 0.15, 0.0) * np.maximum(0, -np.sin(ph_t))
        q[:, J["left_hip_r_joint"]] += 0.004 * np.sin(0.05 * np.arange(T))
        Q[:, e] = q + bias
        lift_l = moving & (np.sin(ph_t) > 0.5); lift_r = moving & (np.sin(ph_t) < -0.5)
        Cc[:, e, 0] = ~lift_l; Cc[:, e, 1] = ~lift_r
        drop = rng.random(T) < 0.01; Cc[drop, e, int(rng.integers(0, 2))] = 0.0                 # 짧은 접촉 누락
        tl = 6.0 + np.where(moving, 4.0 * np.sin(ph_t + 0.4), 0.0) + rng.normal(0, 0.2, T)
        tr = -6.0 - np.where(moving, 4.0 * np.sin(ph_t + 0.4 + np.pi), 0.0) + rng.normal(0, 0.2, T)
        TAU[:, e, J["left_knee_joint"]] = tl; TAU[:, e, J["right_knee_joint"]] = tr
        TAU[:, e, :] += rng.normal(0, 0.5, (T, 12)) * (np.arange(12) % 6 != 3)
        Q[:, e] += 0.0
    BIAS = rng.normal(0, 0.02, (N, 12)).astype(np.float32)
    NOISE = rng.normal(0, 0.3, (T, N, 2)); NOISE[0] = 0.0                 # 첫 틱 0: 배포 τ 기준(첫 측정) = 시뮬 _KI_T0(무잡음 인계 토크)
    return G, W, Cc, Q, TAU, BIAS, NOISE


def check_sim_equiv(N=48, T=900, seed=7, inject=True, w=0.0, osc=False):
    import torch
    code, (l0, l1) = _sim_block()
    G, W, Cc, Q, TAU, BIAS, NOISE = _scenario(N, T, seed, osc=osc)
    dev = "cpu"
    ik = [JOINT_ORDER.index("left_knee_joint"), JOINT_ORDER.index("right_knee_joint")]
    # 가짜 Isaac 객체(블록이 읽는 속성만)
    data = _NS(joint_names=list(JOINT_ORDER), applied_torque=None, projected_gravity_b=torch.zeros(N, 3), root_ang_vel_b=torch.zeros(N, 3))
    robot = _NS(data=data)
    om = _NS(active_terms={"policy": ["base_ang_vel", "projected_gravity", "feet_contact"]},
             group_obs_term_dim={"policy": [(3,), (3,), (2,)]})
    uenv = _NS(observation_manager=om, _default_pose_bias=torch.as_tensor(BIAS))
    enc_box = {}
    mh = types.ModuleType("rl_walking.mdp_hang"); mh.encoder_joint_pos_v2 = lambda u: enc_box["enc"]
    import rl_walking
    _saved = (sys.modules.get("rl_walking.mdp_hang"), getattr(rl_walking, "mdp_hang", None))
    sys.modules["rl_walking.mdp_hang"] = mh; rl_walking.mdp_hang = mh

    class _FakeRng:      # 시뮬 _KC_RNG.normal(0, 0.3, size) 자리 — 미리 뽑은 잡음열을 순서대로(배포에도 같은 값이 측정 τ 로 들어간다)
        def __init__(self):
            self.k = 0

        def normal(self, mu, sd, size):
            assert (mu, sd) == (0.0, 0.3) and tuple(size) == (N, 2)
            v = NOISE[self.k]; self.k += 1; return v

    args = _NS(**dict(SIM_ARGS, settle_knee_inject=(42.0 if inject else 0.0), settle_knee_comp=42.0, settle_w=float(w)))
    ns = dict(torch=torch, math=math, args_cli=args, robot=robot, uenv=uenv, dev=dev, N=N, STEPS_PRE=0, dists=[], policy2=object(),
              _cs=_NS(data=_NS(net_forces_w=torch.zeros(N, 2, 3))), _fl=0, _fr=1,
              _rp=lambda: (torch.zeros(N), torch.zeros(N)), alive=torch.ones(N, dtype=torch.bool),
              _SW_ON=[3.0, 0.5], _SW_OFF=[1.5, 0.3, 1.0, 0.4],
              sw_mode=torch.zeros(N, dtype=torch.bool), sw_t_on=torch.zeros(N), sw_calm=torch.zeros(N), early_hold=torch.zeros(N),
              swc_n=torch.zeros(N), _KC_IDX=ik, _KC_RNG=_FakeRng())
    _st = dict(tau_ref=("handover" if inject else "absolute"))
    if w > 0.0:
        _st["w_thr"] = float(w)
    dets = [C5.SwitchDetector(**KE, settle=_st) for _ in range(N)]
    # 주입 플랜트(시뮬 줄 그대로, float32 torch): 엔코더 = 링크 + 편향 − 편향 + (τ − τ(인계)) / k_env — 배포 qpos 는 이 값
    KI_K = torch.as_tensor(42.0 * np.random.default_rng(777).uniform(0.8, 1.25, size=(N, 1)), dtype=torch.float32)
    T0 = torch.as_tensor(TAU[0][:, ik])
    n_cmp = 0; n_mode1 = 0; eoffs = 0; mism = []
    try:
        for i in range(T):
            obs = torch.as_tensor(np.concatenate([W[i], G[i], Cc[i]], -1))
            data.applied_torque = torch.as_tensor(TAU[i])
            enc_box["enc"] = torch.as_tensor(Q[i])
            ns.update(i=i, t_rel=(i - 0) / 50.0, obs={"policy": obs})
            exec(code, ns)
            # 배포 입력 재구성(시뮬 블록과 같은 float32 연산)
            e_d = torch.as_tensor(Q[i]) - uenv._default_pose_bias
            if inject:
                e_d = e_d.clone(); e_d[:, ik] = e_d[:, ik] + (data.applied_torque[:, ik] - T0) / KI_K
            tau_meas = (data.applied_torque[:, ik] + torch.as_tensor(NOISE[i], dtype=torch.float32)).numpy()
            for e in range(N):
                m = dets[e].update(G[i, e], W[i, e], Cc[i, e], qpos_rad=e_d[e].numpy(), tau_knee_nm=tau_meas[e])
                st = dets[e].settle
                ok = (m == int(ns["sw_mode"][e])) and (np.float32(ns["odo_spd"][e]).view(np.uint32) == st.spd.view(np.uint32)) \
                    and (np.float32(ns["odo_frac"][e]) == st.frac) and (np.float32(ns["odo_ds"][e]) == st.ds)
                if not ok:
                    mism.append((i, e, m, int(ns["sw_mode"][e]), float(ns["odo_spd"][e]), float(st.spd)))
                n_cmp += 1; n_mode1 += m
    finally:
        if _saved[0] is None:
            sys.modules.pop("rl_walking.mdp_hang", None)
        else:
            sys.modules["rl_walking.mdp_hang"] = _saved[0]
        if _saved[1] is None:
            if hasattr(rl_walking, "mdp_hang"):
                delattr(rl_walking, "mdp_hang")
        else:
            rl_walking.mdp_hang = _saved[1]
    assert not mism, f"[B] 시뮬≠배포 {len(mism)} 건, 첫 5: {mism[:5]}"
    n_on = int(ns["swc_n"].sum()); n_on_d = sum(d.n_on for d in dets); eoffs = sum(d.n_early for d in dets)
    assert n_on == n_on_d and n_on >= N, f"[B] 켜짐 수 시뮬 {n_on} · 배포 {n_on_d}"
    assert eoffs >= N // 2, f"[B] 조기 복귀가 너무 적어 판정 경로를 덮지 못함({eoffs})"
    # 시각 대조: 조기 복귀 틱 목록(배포) — 위 모드 전 틱 일치로 같은 시각이 보장된다
    tag = "odic(주입 42·보상 42·기준 인계 τ)" if inject else "odoc(주입 없음·보상 42·절대 τ)"
    extra = ""
    if w > 0.0:
        n_wb = sum(d.settle.n_w_block for d in dets)
        assert n_wb > 0, "[B'] |ω_xy| 조건이 막은 틱 0 — 진동 입력열이 odicw 경로를 덮지 못함"
        tag = f"odicw(w {w}) + " + tag; extra = f" · |ω| 차단 틱 {n_wb}"
    if osc:
        tag += " · 진동 입력열"
    return (f"[B{chr(39) if w > 0.0 else ''}] 시뮬 소스(eval_ground_recovery.py:{l0}-{l1}) = 배포 SwitchDetector(settle) — {tag}: {N} env × {T} 틱 = {n_cmp} 비교, "
            f"모드·|v̄|(float32 비트)·유효비·양발 시간 불일치 0 · 켜짐 {n_on} · 조기 복귀 {eoffs} · 회복 모드 틱 {n_mode1}{extra}")


# ------------------------------------------------------------------ [D] odic(w_thr 0) = odicw 도입 전 커밋 odic 비트 동일
def check_odic_prev(ref, N=24, T=900, seed=11):
    old = _load_ref_c5(ref, expect_odic=True)
    G, W, Cc, Q, TAU, BIAS, NOISE = _scenario(N, T, seed, osc=True)
    ik = [JOINT_ORDER.index("left_knee_joint"), JOINT_ORDER.index("right_knee_joint")]
    a = [C5.SwitchDetector(**KE, settle={}) for _ in range(N)]
    b = [old.SwitchDetector(**KE, settle={}) for _ in range(N)]
    a0 = [C5.SwitchDetector(**KE, settle={"w_thr": 0.0}) for _ in range(N)]      # 명시적 0 = 끔
    n = 0
    for i in range(T):
        for e in range(N):
            tau = (TAU[i, e, ik] + NOISE[i, e]).astype(np.float32)
            q = (Q[i, e] - BIAS[e]).astype(np.float32)
            ma = a[e].update(G[i, e], W[i, e], Cc[i, e], qpos_rad=q, tau_knee_nm=tau)
            mb = b[e].update(G[i, e], W[i, e], Cc[i, e], qpos_rad=q, tau_knee_nm=tau)
            m0 = a0[e].update(G[i, e], W[i, e], Cc[i, e], qpos_rad=q, tau_knee_nm=tau)
            assert ma == mb == m0, f"[D] 틱 {i} env {e} 모드 {ma}/{mb}/{m0}"
            assert a[e].settle.spd.view(np.uint32) == b[e].settle.spd.view(np.uint32) or (np.isnan(a[e].settle.spd) and np.isnan(b[e].settle.spd))
            assert a[e].settle.blackbox() == b[e].settle.blackbox() == a0[e].settle.blackbox(), f"[D] 틱 {i} env {e} 블랙박스"
            n += 1
    ne = sum(d.n_early for d in a)
    assert ne == sum(d.n_early for d in b) and ne >= N // 2, f"[D] 조기 복귀 {ne}"
    return f"[D] odic(w_thr 0·생략) = {ref} odic 비트 동일: {N} env × {T} 틱(진동 입력열) = {n} 비교, 조기 복귀 {ne}"


# ------------------------------------------------------------------ [C] 브리지 코어 기본 경로 비트 동일(실모델)
def check_bridge_off(ref):
    src = _git_show(ref, "jetson/scripts/rl_bridge_node.py")
    assert "C5_SETTLE" not in src, f"{ref} 브리지에 이미 odic 플래그가 있다"
    tmp = os.path.join(REPO, "jetson", "scripts", "_ref_bridge_odic_tmp.py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        spec = importlib.util.spec_from_file_location("_ref_bridge_odic_tmp", tmp)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    finally:
        os.unlink(tmp)
    spec = importlib.util.spec_from_file_location("_new_bridge_odic", os.path.join(REPO, "jetson", "scripts", "rl_bridge_node.py"))
    new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
    assert new.C5_SETTLE is None and set(new.CHECKPOINTS) == set(old.CHECKPOINTS) and new.C5_EXTRA == old.C5_EXTRA
    done = []
    pulses = [(80, 7.0, 25), (260, 9.0, 30)]
    for name in ("stand_c5kr_g21a_g24ra", "stand_c5_g21a_g24ra", "stand_crouch_g21a"):
        d = old.CHECKPOINTS.get(name)
        if d is None or not os.path.exists(os.path.join(d, "policy.pt")) and not os.path.exists(os.path.join(d, "policy.onnx")):
            continue
        a, b = new.BridgeCore(name), old.BridgeCore(name)
        for c in (a, b):
            c.reset(); c.cmd = (0.0, 0.0, 0.0)
        ra, rb = np.random.default_rng(9), np.random.default_rng(9)
        n_sw = 0
        for k in range(400):
            ga, gra, qa, va, ca = _imu_inputs(ra, k, pulses); gb, grb, qb, vb, cb = _imu_inputs(rb, k, pulses)
            th = math.atan2(float(gra[0]), -float(gra[2]))
            quat = (math.cos(th / 2), 0.0, math.sin(th / 2), 0.0)
            oa = a.tick(ga, quat, qa, va, ca); ob = b.tick(gb, quat, qb, vb, cb)
            assert np.array_equal(np.asarray(oa).view(np.uint32), np.asarray(ob).view(np.uint32)), f"[C] {name} 틱 {k} 불일치"
            if isinstance(a.runner, C5.C5Runner):
                assert a.runner.blackbox() == b.runner.blackbox(), f"[C] {name} 틱 {k} 블랙박스 불일치"
                n_sw += a.runner.last_switch != ""
        done.append(f"{name}(전환 {n_sw})")
    assert done, "[C] 비교할 실모델 없음(logs/ 확인)"
    return f"[C] 브리지 코어 기본(--c5-settle 생략) = {ref} 브리지 비트 동일: {', '.join(done)} × 400 틱"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="f44e8701")
    ap.add_argument("--ref-odic", default="28657c3f", help="odicw 도입 전 odic 커밋([D])")
    a = ap.parse_args()
    fails = 0
    for f, kw in ((check_default_bitident, dict(ref=a.ref)), (check_bridge_off, dict(ref=a.ref)),
                  (check_sim_equiv, dict(inject=True)), (check_sim_equiv, dict(inject=False, seed=8)),
                  (check_sim_equiv, dict(inject=True, osc=True, seed=9)),
                  (check_sim_equiv, dict(inject=True, w=0.4, osc=True, seed=9)),
                  (check_sim_equiv, dict(inject=False, w=0.4, osc=True, seed=10)),
                  (check_odic_prev, dict(ref=a.ref_odic))):
        try:
            print("[PASS]", f(**kw), flush=True)
        except Exception as ex:  # noqa: BLE001
            fails += 1
            print("[FAIL]", f.__name__, f"{type(ex).__name__}: {ex}", flush=True)
    print("odic equiv: ALL PASS" if fails == 0 else f"odic equiv: {fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
