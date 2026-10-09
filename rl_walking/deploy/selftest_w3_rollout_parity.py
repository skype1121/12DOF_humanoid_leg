"""W3S3d 롤아웃 parity 시험(1007 parity 감사) — 시뮬 기록 원천을 배포 러너·브리지 코어에 열린 고리로 넣어 관측 320·행동 대조.

  python3 rl_walking/deploy/selftest_w3_rollout_parity.py [--npz log_picture/w3parity_rec_w3s3d_15800_d3.npz] [--report out.txt]

기록: rl_walking/scripts/w3_parity_record.py(GPU 1 회, 관측 잡음 끔·지연/편향 DR 켬, 명령열 서기→제자리→앞→옆→회전→앞+헤딩 고리→정지→옆−(오른발 리드)→정지).
CPU·torch 만 필요(onnxruntime 있으면 onnx 도 비교). **계약 검증이지 폐루프 성능 증명이 아니다**(CLAUDE.md 6).

검사(허용오차는 실행 전에 고정 — 아래 TOL):
  [P1] 러너 직접·교사 강제(직전 행동 = 시뮬 행동): 원천 = 시뮬 관측의 최신 프레임(자이로·중력·명령·관절각+env 기본자세·접촉) + 시뮬 엔코더 속도.
       러너가 만든 관측 320 의 항별 최대 |Δ| ≤ TOL_OBS, W3 클록 ge·u·ge_obs = 시뮬(env._ge·_gu·_ge_obs) 비트, 행동 |Δ| ≤ TOL_ACT_TF.
  [P2] 같은 원천·자유(러너 자신의 행동으로 직전 행동 EMA) — 실배포 열린 고리: 관측 ≤ TOL_FREE, 행동 ≤ TOL_FREE.
  [P3] 브리지 코어(BridgeCore.tick, W3S3d 키, heading_hold OFF = stand 가족 경로) — 명령은 코어가 만든다(헤딩 고리 W3HeadingLoop: 같은 틱 IMU 요),
       qvel 은 브리지 _tick 의 유한차분식, 기본자세 = 브리지 CROUCH_V5. 중력은 시뮬 관측 원천을 넣는다(IMU 지연·장착 편향 DR 포함 — 코어의 쿼터니언→중력식은 [P5]).
       대조: 명령 관측 wz(고리 출력)·관측 320·행동.
  [P4] 행동 체인: 러너 슬루 → 브리지 LPF(lpf_step) vs 시뮬 액션텀(지연 d → 슬루 → LPF) 상태(_lpf_state) — d+1 틱 이동 비교.
       브리지 LPF 첫 값 = 슬루 출력(배포) / 실측 qpos(시뮬과 같은 초기화) 두 가지. 백래시·온셋·소프트 클램프는 시뮬 물리 모형이라 대상 밖.
  [P5] 공식: 브리지 projected_gravity(쿼터니언) = 시뮬 projected_gravity_b(참값), 요 식 = 시뮬 _yaw_of.
  [P6] 민감도(뮤턴트가 잡히는지): 클록 1 틱 늦춤 · jvel β 0.27 · 행동 EMA 끔 · 헤딩 고리 한 틱 전 요(eval_w3 방식) · 접촉 좌우 뒤바꿈.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rl_walking.deploy import w3_deploy as W  # noqa: E402
from rl_walking.deploy.deploy_chain import lpf_step  # noqa: E402
from rl_walking.deploy.policy_runner import JOINT_ORDER, SLEW_RAD  # noqa: E402
from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4, yaw_from_quat_wxyz  # noqa: E402

NPZ_DEFAULT = os.path.join(REPO, "log_picture", "w3parity_rec_w3s3d_15800_d3.npz")
EXPORT = os.path.join(REPO, W.W3S3D_RUN_REL, "exported")
FIXED = []                      # 1010: --mech-fixed 관절 열(없으면 종전과 같은 계산)
KEY = W.W3S3D_CKPT_KEY          # 1010: --key 로 바꿀 수 있다(기본 = 종전 W3S3d-15800 — 인자 없이는 종전과 같은 실행)
TERMS = (("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3), ("joint_pos", 12), ("joint_vel", 12), ("actions", 12),
         ("gait_clock", 2), ("feet_contact", 2), ("gyro_ema", 3), ("joint_vel_ema", 12))
OFF = {}
_o = 0
for _n, _d in TERMS:
    OFF[_n] = slice(_o, _o + 5 * _d)
    _o += 5 * _d
assert _o == 320
#: 허용오차(실행 전 고정). 관측: float32 연산 순서(GPU torch vs CPU numpy)·sin/cos ulp. 행동: 같은 네트워크 GPU vs CPU 곱셈 순서.
TOL_OBS = 1e-5
TOL_ACT_TF = 2e-4
TOL_FREE = 1e-3
TOL_CMD = 1e-6
TOL_CHAIN = 1e-5
TOL_GRAV = 2e-6
SS = 150          # '정상 구간' 시작 틱(3 s): 시작 과도(EMA 0.9ⁿ·LPF 0.73ⁿ)가 1e-6 아래로 감쇠한 뒤


def load(npz):
    z = np.load(npz, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z, meta


def _bridge_mod():
    spec = importlib.util.spec_from_file_location("_bridge_parity", os.path.join(REPO, "jetson", "scripts", "rl_bridge_node.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def valid_len(z, e):
    t = np.nonzero(z["term"][:, e])[0]
    return int(t[0]) if len(t) else z["obs"].shape[0]


def term_diffs(a, b):
    return {n: float(np.abs(a[..., OFF[n]].astype(np.float64) - b[..., OFF[n]].astype(np.float64)).max()) for n, _ in TERMS}


def mk_runner(meta, e, default=None, w3cfg=W.W3S3D_CONFIG, jvel_beta=0.25, act_beta=0.27, backend="torch", ema_fresh=1):
    jn = meta["joint_names"]
    assert jn == list(JOINT_ORDER), f"관절 순서 다름: 시뮬 {jn} vs 러너 {JOINT_ORDER}"
    dp = np.asarray(meta["default_joint_pos"][e], np.float32) if default is None else default
    return PolicyRunnerStage4(EXPORT, backend=backend, obs_dim=320, gait_freq=1.1, default_pose_rad=dp, w3=w3cfg,
                              act_obs_ema_beta=act_beta, jvel_obs_ema_beta=jvel_beta, **({"ema_fresh_ticks": ema_fresh} if ema_fresh != 1 else {}))


def run_runner(z, meta, e, *, teacher=True, mut=None, backend="torch"):
    """[P1]/[P2](+뮤턴트). 반환 dict(관측 최대차 항별, 행동 최대차, 클록 불일치 틱, 러너 출력 열)."""
    mut = mut or {}
    r = mk_runner(meta, e, jvel_beta=mut.get("jvel_beta", 0.25), act_beta=mut.get("act_beta", 0.27), backend=backend,
                  ema_fresh=mut.get("ema_fresh", 2))
    r.reset()
    T = valid_len(z, e)
    dp = r.default_pose
    obs_run, act_run, emitted = [], [], []
    clk_bad = []
    for n in range(T):
        want = bool(z["want"][n, e])
        if mut.get("clock_lag") and n > 0:
            want = bool(z["want"][n - 1, e])
        r.set_gait_enable(want)
        if teacher and n > 0:
            r.last_action = z["act"][n - 1, e].astype(np.float32).copy()
        contact = z["contact"][n, e]
        if mut.get("swap_contact"):
            contact = contact[::-1].copy()
        out = r.step(z["gyro"][n, e], z["grav"][n, e], z["cmd"][n, e], z["qpos_rel"][n, e] + dp, z["qvel"][n, e], contact)
        obs_run.append(r.last_obs.copy())
        act_run.append(r.last_action.copy())
        emitted.append(out.copy())
        c = r._w3_clock
        if (c.ge != bool(z["ge"][n, e]) and n > 0) or (n > 0 and np.float32(c.u).view(np.uint32) != np.float32(z["gu"][n, e]).view(np.uint32)) \
                or (n > 0 and np.float32(c.ge_obs).view(np.uint32) != np.float32(z["ge_obs"][n, e]).view(np.uint32)):
            clk_bad.append(n)
    obs_run = np.stack(obs_run)
    act_run = np.stack(act_run)
    return {"T": T, "obs": term_diffs(obs_run, z["obs"][:T, e]), "act": float(np.abs(act_run - z["act"][:T, e]).max()),
            "obs_ss": term_diffs(obs_run[SS:], z["obs"][SS:T, e]), "act_ss": float(np.abs(act_run[SS:] - z["act"][SS:T, e]).max()),
            "act_argmax_t": int(np.abs(act_run - z["act"][:T, e]).max(1).argmax()), "clk_bad": clk_bad,
            "emitted": np.stack(emitted), "act_run": act_run, "obs_run": obs_run}


def bridge_qvel(qpos_seq, tick_s):
    """rl_bridge_node._tick 의 qvel 식 그대로(관절별 '값이 머문 틱 수' 보정, 캡 12, 첫 틱 0)."""
    prev, hold, out = None, [1] * 12, []
    for qpos in qpos_seq:
        if prev is None:
            qv = [0.0] * 12
            hold = [1] * 12
        else:
            qv = []
            for k, (a_, b_) in enumerate(zip(qpos, prev)):
                if a_ == b_:
                    hold[k] = min(hold[k] + 1, 12)
                    qv.append(0.0)
                else:
                    qv.append((a_ - b_) / (tick_s * hold[k]))
                    hold[k] = 1
        prev = qpos
        out.append(qv)
    return out


def run_bridge(z, meta, e, B, *, stale_yaw=False, qvel_mode="bridge"):
    """[P3] BridgeCore.tick 경로(러너 기본 설정 = 실제 브리지 _make_runner). 중력은 시뮬 원천을 넣는다(모듈 함수 교체 — 쿼터니언 식은 [P5] 별도).
    qvel_mode: "bridge" = 브리지 _tick 의 유한차분(같은 값 반복 시 '머문 틱 수'로 나눔) / "plain" = 단순 유한차분(시뮬 encoder_joint_vel_v2 식).
    헤딩 고리 고장(오차 15°·120 s)이 나면 코어가 걷기를 멈춘다(배포 안전 장치 — 시뮬 기록에는 없음) → 그 틱 전까지만 비교하고 사건을 돌려준다."""
    key = KEY
    core = B.BridgeCore(key)
    core.reset()
    core.heading_hold = False                     # stand 가족 경로(브리지 stand 시작이 강제)
    assert core.w3 and core.w3cmd and core.w3hl is not None
    T = valid_len(z, e)
    dp = core.runner.default_pose
    qpos_seq = [[float(v) for v in (z["qpos_rel"][n, e] + dp)] for n in range(T)]
    if qvel_mode == "bridge":
        qv = bridge_qvel(qpos_seq, B.TICK_SEC)
    else:
        qv = [[0.0] * 12] + [[(a_ - b_) / B.TICK_SEC for a_, b_ in zip(qpos_seq[n], qpos_seq[n - 1])] for n in range(1, T)]
    g_now = {"g": None}
    orig_pg, orig_cg = B.projected_gravity, B.correct_gravity
    B.projected_gravity = lambda q: g_now["g"]
    B.correct_gravity = lambda g, bias_deg=None: g
    obs_run, act_run, cmd_run = [], [], []
    fault = None
    try:
        for n in range(T):
            want = bool(z["want"][n, e])
            bc = tuple(float(v) for v in z["base_cmd"][n, e])
            core.cmd = bc if want else (0.0, 0.0, 0.0)
            core.w3hl_on = bool(z["hl_on"][n, e])
            core.set_gait_enable(want)
            g_now["g"] = tuple(float(v) for v in z["grav"][n, e])
            qn = n - 1 if (stale_yaw and n > 0) else n
            quat = tuple(float(v) for v in z["quat"][qn, e])
            core.tick(tuple(float(v) for v in z["gyro"][n, e]), quat, qpos_seq[n], qv[n],
                      [float(v) for v in z["contact"][n, e]])
            if core.w3hl_events:
                fault = (n, core.w3hl_events[0]["why"])
                break
            obs_run.append(core.runner.last_obs.copy())
            act_run.append(core.runner.last_action.copy())
            cmd_run.append(core.runner.last_obs[OFF["velocity_commands"]][-3:].copy())
    finally:
        B.projected_gravity, B.correct_gravity = orig_pg, orig_cg
    Tc = len(obs_run)
    obs_run = np.stack(obs_run)
    act_run = np.stack(act_run)
    cmd_run = np.stack(cmd_run)
    hl = z["hl_on"][:Tc, e].astype(bool)
    return {"T": Tc, "fault": fault, "obs": term_diffs(obs_run, z["obs"][:Tc, e]), "act": float(np.abs(act_run - z["act"][:Tc, e]).max()),
            "obs_ss": term_diffs(obs_run[SS:], z["obs"][SS:Tc, e]), "act_ss": float(np.abs(act_run[SS:] - z["act"][SS:Tc, e]).max()),
            "cmd": float(np.abs(cmd_run.astype(np.float64) - z["cmd"][:Tc, e].astype(np.float64)).max()),
            "cmd_hl": float(np.abs(cmd_run[hl, 2].astype(np.float64) - z["cmd"][:Tc, e][hl, 2].astype(np.float64)).max()) if hl.any() else 0.0,
            "n_hl": int(hl.sum()), "wz_hl": cmd_run[hl, 2].astype(np.float64)}


def run_chain(z, meta, e, init):
    """[P4] 시뮬 원시 행동 → 러너 슬루식(target = 기본자세 + 0.25·raw, ±4°/틱, 첫 기준 = 실측 qpos) → 브리지 lpf_step
    vs 시뮬 액션텀 _lpf_state[n + d + 1](지연 d → 슬루 → LPF). init = "deploy"(브리지: 첫 LPF 값 = 첫 슬루 출력) | "sim"(첫 값 = 실측 qpos)."""
    d = int(meta["delay_env"][e])
    beta = float(meta["lpf_beta_env"][e])
    T = valid_len(z, e)
    dp = np.asarray(meta["default_joint_pos"][e], np.float32)
    q0 = (z["qpos_rel"][0, e] + dp).astype(np.float32)
    prev = q0.copy()
    st = None if init == "deploy" else [float(v) for v in q0]
    outs = []
    for n in range(T):
        target = dp + np.float32(0.25) * z["act"][n, e].astype(np.float32)
        prev = prev + np.clip(target - prev, -np.float32(SLEW_RAD), np.float32(SLEW_RAD))      # 러너 step 과 같은 float32 식
        st = lpf_step(st, [float(v) for v in prev], beta)
        outs.append(st)
    outs = np.asarray(outs)
    sim = z["lpf_state"][:, e].astype(np.float64)
    k = np.arange(T) + d + 1
    ok = k < T
    dall = np.abs(outs[ok] - sim[k[ok]])
    live = [j for j in range(dall.shape[1]) if j not in FIXED]      # 기구 고정 관절은 따로: 시뮬 LPF 는 실제 고정각에서 출발하고(원천은 영점), 그 목표는 브리지가 0° 로 덮는다
    diff = dall[:, live].max(1)
    return {"max": float(diff.max()), "first5": [float(x) for x in diff[:5]], "after50": float(diff[50:].max()) if len(diff) > 50 else float("nan"),
            "d": d, "beta": beta, "fixed_max": float(dall[:, FIXED].max()) if FIXED else None, "fixed_after50": float(dall[50:, FIXED].max()) if FIXED and len(dall) > 50 else None}


def check_formulas(z, meta, B):
    q = z["quat"].reshape(-1, 4).astype(np.float64)
    gt = z["grav_true"].reshape(-1, 3).astype(np.float64)
    gb = np.array([B.projected_gravity(tuple(v)) for v in q])
    yaw_d = np.array([yaw_from_quat_wxyz(*v) for v in q])
    w, x, y, zz = q.T
    yaw_s = np.arctan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz))
    return float(np.abs(gb - gt).max()), float(np.abs(yaw_d - yaw_s).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ_DEFAULT)
    ap.add_argument("--report", default="")
    ap.add_argument("--envs", default="", help="쉼표 env 목록(기본 전부)")
    ap.add_argument("--key", default=W.W3S3D_CKPT_KEY, help="1010: 대조할 W3 키(w3_deploy.W3_SPECS) — 정책 폴더 = <run_rel>/exported. 기본 = 종전")
    ap.add_argument("--mech-fixed", default="", help="1010: 기구 고정 관절(쉼표) — stage8 합성 피드백 계약대로 원천의 그 관절 상대각·속도를 0 으로 넣는다(기본 없음 = 종전)")
    a = ap.parse_args()
    global EXPORT, KEY
    if a.key != W.W3S3D_CKPT_KEY:
        assert a.key in W.W3_SPECS and W.W3_SPECS[a.key][3] == W.W3S3D_CONFIG, f"--key {a.key}: W3_SPECS 에 없거나 위상 설정이 W3S3d 와 다름"
        KEY = a.key; EXPORT = os.path.join(REPO, W.W3_SPECS[a.key][0], "exported")
    if not os.path.isfile(a.npz):
        print(f"[SKIP] 기록 없음: {a.npz} — rl_walking/scripts/w3_parity_record.py 로 먼저 기록")
        return 0
    if not os.path.isdir(EXPORT):
        print(f"[SKIP] export 없음: {EXPORT}")
        return 0
    z, meta = load(a.npz)
    if a.mech_fixed:
        _fx = [list(JOINT_ORDER).index(n.strip()) for n in a.mech_fixed.split(",") if n.strip()]
        FIXED[:] = _fx
        z = {k: z[k] for k in z.files}
        _pre = (float(np.abs(z["qpos_rel"][..., _fx]).max()), float(np.abs(z["qvel"][..., _fx]).max()))
        z["qpos_rel"] = z["qpos_rel"].copy(); z["qvel"] = z["qvel"].copy()
        z["qpos_rel"][..., _fx] = 0.0; z["qvel"][..., _fx] = 0.0
        print(f"[mech-fixed] 관절 {a.mech_fixed} 원천 = 영점·속도 0 (바꾸기 전 |상대각| 최대 {_pre[0]:.3e} rad · |속도| 최대 {_pre[1]:.3e} rad/s)")
    N = int(meta["num_envs"])
    envs = [int(x) for x in a.envs.split(",")] if a.envs else list(range(N))
    B = _bridge_mod()
    B.enable_experimental_w3()
    lines = [f"# W3S3d 롤아웃 parity — {os.path.basename(a.npz)}", f"- ckpt {meta['ckpt']} · 지연 env {meta['delay_env']} · LPF β {meta['lpf_beta_env']} · "
             f"자이로 지연 {meta['gyro_lag_env']} · IMU 지연 {meta['imu_lag_env']} · 보조 α {meta['w3_alpha']}/{meta['assist_env']} assist max {meta['assist_last_max']}",
             f"- 허용오차: 관측 {TOL_OBS} · 행동(교사) {TOL_ACT_TF} · 자유 {TOL_FREE} · 명령 {TOL_CMD} · 체인 {TOL_CHAIN} · 중력식 {TOL_GRAV}"]
    fails = []

    def chk(cond, msg):
        lines.append(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    eg, ey = check_formulas(z, meta, B)
    chk(eg <= TOL_GRAV and ey <= 1e-9, f"[P5] 브리지 projected_gravity(쿼터니언) vs 시뮬 projected_gravity_b 최대 {eg:.2e} · 요 식 {ey:.1e}")
    for e in envs:
        T = valid_len(z, e)
        lines.append(f"## env {e}: 유효 {T} 틱(낙상 {'없음' if T == z['obs'].shape[0] else f'틱 {T}'}) · ge 1 {int(z['ge'][:T, e].sum())} 틱 · "
                     f"리드 오른발(u 0.5 시작) {int(((z['ge'][1:T, e] == 1) & (z['ge'][:T - 1, e] == 0) & (np.abs(z['gu'][1:T, e] - 0.5) < 1e-7)).sum())}회")
        P1 = run_runner(z, meta, e, teacher=True)                       # ema_fresh_ticks 2 = 시뮬 _ema_state 계약(제안 옵션)
        mo = max(P1["obs"].values())
        chk(mo <= TOL_OBS and not P1["clk_bad"], f"[P1] 교사 강제·ema_fresh 2 관측 최대 |Δ| {mo:.2e} (항별 " + ", ".join(f"{k} {v:.1e}" for k, v in P1["obs"].items())
            + f") · W3 클록(ge·u·ge_obs) 불일치 틱 {len(P1['clk_bad'])}")
        chk(P1["act"] <= TOL_ACT_TF, f"[P1] 교사 강제·ema_fresh 2 행동 최대 |Δ| {P1['act']:.2e} (틱 {P1['act_argmax_t']}) — 같은 네트워크 GPU(시뮬) vs CPU(export)")
        P1d = run_runner(z, meta, e, teacher=True, mut={"ema_fresh": 1})  # 현행 배포 기본(틱 0 에서만 EMA 시작)
        chk(max(P1d["obs_ss"].values()) <= TOL_OBS and P1d["act_ss"] <= TOL_ACT_TF,
            f"[P1] 교사 강제·현행 기본(ema_fresh 1) 정상 구간(틱 ≥ {SS}) 관측 {max(P1d['obs_ss'].values()):.2e} · 행동 {P1d['act_ss']:.2e}")
        lines.append(f"INFO [P1] 현행 기본 시작 과도(전 구간): gyro_ema {P1d['obs']['gyro_ema']:.2e} · joint_vel_ema {P1d['obs']['joint_vel_ema']:.2e} · "
                     f"행동 {P1d['act']:.2e}(틱 {P1d['act_argmax_t']}) — 시뮬 _ema_state 는 틱 0·1 둘 다 재시작, 러너는 틱 0 만")
        P2 = run_runner(z, meta, e, teacher=False)
        mo2 = max(P2["obs"].values())
        chk(mo2 <= TOL_FREE and P2["act"] <= TOL_FREE, f"[P2] 자유(러너 자신 행동)·ema_fresh 2 관측 {mo2:.2e} · 행동 {P2['act']:.2e} · actions 항 {P2['obs']['actions']:.1e}")
        P3p = run_bridge(z, meta, e, B, qvel_mode="plain")
        P3 = run_bridge(z, meta, e, B, qvel_mode="bridge")
        wz = P3p["wz_hl"]
        small = int(((np.abs(wz) > 0) & (np.abs(wz) < 0.1)).sum())
        if P3p["fault"] is not None:
            lines.append(f"INFO [P3] 배포 헤딩 고리 고장 → 걷기 정지: 틱 {P3p['fault'][0]}({P3p['fault'][0] * 0.02:.2f} s) '{P3p['fault'][1]}' — 시뮬 기록(고장 장치 없음)은 계속 걸음. 비교는 그 전까지")
        chk(P3p["cmd"] <= TOL_CMD, f"[P3] 브리지 코어(헤딩 고리 W3HeadingLoop) 명령 관측 최대 |Δ| {P3p['cmd']:.2e}(고리 구간 {P3p['n_hl']} 틱 wz {P3p['cmd_hl']:.2e}) · "
            f"고리 wz 범위 [{wz.min() if len(wz) else 0:+.4f}, {wz.max() if len(wz) else 0:+.4f}], 0<|wz|<0.1 {small}/{len(wz)} 틱(학습 밖 크기)")
        chk(max(P3p["obs_ss"].values()) <= TOL_FREE and P3p["act_ss"] <= TOL_FREE,
            f"[P3] 브리지 코어·단순 유한차분 qvel 정상 구간 관측 {max(P3p['obs_ss'].values()):.2e} · 행동 {P3p['act_ss']:.2e}")
        lines.append(f"INFO [P3] 브리지 _tick qvel(반복값 '머문 틱' 보정) 정상 구간: joint_vel {P3['obs_ss']['joint_vel']:.2e} · joint_vel_ema {P3['obs_ss']['joint_vel_ema']:.2e} · "
                     f"행동 {P3['act_ss']:.2e} — 시뮬 엔코더(발목 유격 안 = 명령값)가 정확히 같은 값을 반복할 때 시뮬 계약(단순 차분)과 다름")
        C1 = run_chain(z, meta, e, init="sim")
        C2 = run_chain(z, meta, e, init="deploy")
        chk(C1["max"] <= TOL_CHAIN, f"[P4] 체인(시뮬 원시 행동 → 러너 슬루 → 브리지 LPF β{C1['beta']:.2f}) vs 시뮬(지연 {C1['d']} → 슬루 → LPF), LPF 첫 값=실측 qpos: 최대 |Δ| {C1['max']:.2e}"
            + (f" (기구 고정 관절 제외 — 그 관절은 최대 {C1['fixed_max']:.2e} rad·50 틱 뒤 {C1['fixed_after50']:.2e}: 시뮬은 실제 고정각에서 출발)" if FIXED else ""))
        lines.append(f"INFO [P4] 배포 LPF 첫 값(=첫 슬루 출력) 최대 |Δ| {C2['max']:.2e} rad · 첫 5 틱 {['%.1e' % x for x in C2['first5']]} · 50 틱 뒤 {C2['after50']:.1e}"
                     " — 시작 과도만((1−β)ⁿ 감쇠)")
        if e == envs[0]:
            # [P6] 민감도: 뮤턴트가 허용오차를 넘어야 한다
            for nm, kw in (("클록 1 틱 늦춤(want 한 틱 지연)", {"clock_lag": True}), ("jvel β 0.27", {"jvel_beta": 0.27}),
                           ("행동 EMA 끔(β 1)", {"act_beta": 1.0}), ("접촉 좌우 뒤바꿈", {"swap_contact": True})):
                M = run_runner(z, meta, e, teacher=True, mut=kw)
                mm = max(M["obs"].values())
                chk(mm > TOL_OBS or M["clk_bad"], f"[P6] 뮤턴트 '{nm}' 검출: 관측 최대 |Δ| {mm:.2e} · 클록 불일치 {len(M['clk_bad'])} 틱")
            S = run_bridge(z, meta, e, B, stale_yaw=True)
            chk(S["cmd_hl"] > TOL_CMD, f"[P6] 뮤턴트 '헤딩 고리 한 틱 전 요(eval_w3 방식)' 검출: 고리 wz 최대 |Δ| {S['cmd_hl']:.2e} rad/s · 행동 {S['act']:.2e}")
    lines.append(f"w3 rollout parity: {'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
    txt = "\n".join(lines)
    print(txt)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
