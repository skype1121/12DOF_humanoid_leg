"""1007 parity 감사: KE 현행 IMU 조기 복귀 감지기의 float64 타이머 vs 시뮬 float32 — 배포 옵션 imu_f32 동치 시험. PC 전용(torch CPU).

  python3 rl_walking/deploy/selftest_c5_imu_f32.py [--ref HEAD]

검사:
  [A] imu_f32 생략(기본) = --ref 커밋 c5_switch 비트 동일: 같은 입력열(selftest_odic_equiv._scenario, 진동 포함)에서 KE 감지기 모드·카운터·
      calm/t_on/hold 값 전 틱 비교 + C5Runner(스텁 정책) 목표각·관측·블랙박스 비교.
  [B] 시뮬 소스(eval_ground_recovery.py 감지 블록, --settle_judge 생략 = imu 조기 복귀)를 그대로 exec 한 모드열 vs 배포 SwitchDetector:
      imu_f32=True → 불일치 0 이어야 함 · 기본(float64) → 불일치 틱·꺼짐 시각 차(틱) 분포를 보고(STOPEST §9: 82 꺼짐 중 46 건 −1 틱 재현).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rl_walking.deploy import c5_switch as C5  # noqa: E402
from rl_walking.deploy import selftest_odic_equiv as OE  # noqa: E402

KE = dict(OE.KE)


def _load_ref(ref):
    src = subprocess.run(["git", "-C", REPO, "show", f"{ref}:rl_walking/deploy/c5_switch.py"], check=True, capture_output=True, text=True).stdout
    assert "imu_f32" not in src, f"{ref} 의 c5_switch 에 이미 imu_f32 가 있다 — 기준 커밋 확인"
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


def check_default(ref, N=24, T=900, seed=21):
    old = _load_ref(ref)
    G, W, Cc, Q, TAU, BIAS, NOISE = OE._scenario(N, T, seed, osc=True)
    a = [C5.SwitchDetector(**KE) for _ in range(N)]
    b = [old.SwitchDetector(**KE) for _ in range(N)]
    n = 0
    for i in range(T):
        for e in range(N):
            ma = a[e].update(G[i, e], W[i, e], Cc[i, e]); mb = b[e].update(G[i, e], W[i, e], Cc[i, e])
            assert ma == mb and (a[e].calm, a[e].t_on, a[e].early_hold, a[e].n_on, a[e].n_early) == (b[e].calm, b[e].t_on, b[e].early_hold, b[e].n_on, b[e].n_early), \
                f"[A] 틱 {i} env {e}"
            n += 1
    # 러너 래퍼(스텁)도 비트 동일
    Wm = np.random.default_rng(5).normal(0, 0.05, (12, 320)).astype(np.float32)
    stand = lambda o: np.tanh(Wm @ o)            # noqa: E731
    rec = lambda o: 0.3 - np.tanh(Wm @ o)        # noqa: E731
    kw = dict(obs_dim=320, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
    ra = C5.C5Runner(stand, rec, detector=C5.SwitchDetector(**KE), cue_vx=0.03, cue_s=1.0, cue_dir_imu=True, **kw)
    rb = old.C5Runner(stand, rec, detector=old.SwitchDetector(**KE), cue_vx=0.03, cue_s=1.0, cue_dir_imu=True, **kw)
    for i in range(T):
        qa = Q[i, 0]
        ya = ra.step(W[i, 0], G[i, 0], (0.0, 0.0, 0.0), qa, np.zeros(12, np.float32), Cc[i, 0])
        yb = rb.step(W[i, 0], G[i, 0], (0.0, 0.0, 0.0), qa, np.zeros(12, np.float32), Cc[i, 0])
        assert np.array_equal(np.asarray(ya).view(np.uint32), np.asarray(yb).view(np.uint32)) and ra.blackbox() == rb.blackbox(), f"[A] 러너 틱 {i}"
    ne = sum(d.n_early for d in a)
    return f"[A] imu_f32 생략 = {ref} c5_switch 비트 동일: 감지기 {N} env × {T} 틱 = {n} 비교(조기 복귀 {ne}) + C5Runner {T} 틱"


def sim_vs_deploy(imu_f32, N=48, T=900, seed=7):
    """시뮬 감지 블록(imu 조기 복귀) 모드열 vs 배포. 반환 (불일치 틱 수, 꺼짐 시각 차 목록[틱: 배포 − 시뮬], 시뮬 꺼짐 수)."""
    import torch
    code, (l0, l1) = OE._sim_block()
    G, W, Cc, Q, TAU, BIAS, NOISE = OE._scenario(N, T, seed, osc=True)
    data = OE._NS(joint_names=list(OE.JOINT_ORDER), applied_torque=None, projected_gravity_b=torch.zeros(N, 3), root_ang_vel_b=torch.zeros(N, 3))
    robot = OE._NS(data=data)
    om = OE._NS(active_terms={"policy": ["base_ang_vel", "projected_gravity", "feet_contact"]}, group_obs_term_dim={"policy": [(3,), (3,), (2,)]})
    uenv = OE._NS(observation_manager=om)
    args = OE._NS(**dict(OE.SIM_ARGS, settle_judge="", settle_knee_inject=0.0, settle_knee_comp=0.0))
    ns = dict(torch=torch, math=__import__("math"), args_cli=args, robot=robot, uenv=uenv, dev="cpu", N=N, STEPS_PRE=0, dists=[], policy2=object(),
              _cs=OE._NS(data=OE._NS(net_forces_w=torch.zeros(N, 2, 3))), _fl=0, _fr=1,
              _rp=lambda: (torch.zeros(N), torch.zeros(N)), alive=torch.ones(N, dtype=torch.bool),
              _SW_ON=[3.0, 0.5], _SW_OFF=[1.5, 0.3, 1.0, 0.4],
              sw_mode=torch.zeros(N, dtype=torch.bool), sw_t_on=torch.zeros(N), sw_calm=torch.zeros(N), early_hold=torch.zeros(N), swc_n=torch.zeros(N))
    dets = [C5.SwitchDetector(**KE, imu_f32=imu_f32) for _ in range(N)]
    ms = np.zeros((T, N), np.int8); md = np.zeros((T, N), np.int8)
    for i in range(T):
        obs = torch.as_tensor(np.concatenate([W[i], G[i], Cc[i]], -1))
        ns.update(i=i, t_rel=i / 50.0, obs={"policy": obs})
        exec(code, ns)
        ms[i] = ns["sw_mode"].numpy().astype(np.int8)
        for e in range(N):
            md[i, e] = dets[e].update(G[i, e], W[i, e], Cc[i, e])
    bad = int((ms != md).sum())
    # 꺼짐 사건(1→0) 시각 대응: 시뮬 꺼짐마다 같은 env 의 가장 가까운 배포 꺼짐(±5 틱)과의 차
    dd = []
    n_off = 0
    for e in range(N):
        so = np.nonzero((ms[:-1, e] == 1) & (ms[1:, e] == 0))[0] + 1
        do = np.nonzero((md[:-1, e] == 1) & (md[1:, e] == 0))[0] + 1
        n_off += len(so)
        for t in so:
            if len(do):
                k = do[np.abs(do - t).argmin()]
                if abs(k - t) <= 5:
                    dd.append(int(k - t))
    return bad, dd, n_off, (l0, l1), N * T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="HEAD", help="imu_f32 도입 전 c5_switch 커밋")
    a = ap.parse_args()
    fails = 0
    try:
        print("[PASS]", check_default(a.ref))
    except Exception as ex:  # noqa: BLE001
        import traceback; traceback.print_exc(); fails += 1; print("[FAIL] [A]", ex)
    for f32 in (True, False):
        bad, dd, n_off, (l0, l1), n = sim_vs_deploy(f32)
        hist = {k: dd.count(k) for k in sorted(set(dd))}
        msg = (f"[B] 시뮬 감지 블록(eval_ground_recovery.py:{l0}-{l1}, imu 조기 복귀 KE) vs 배포 imu_f32={f32}: 모드 불일치 {bad}/{n} 틱 · "
               f"시뮬 꺼짐 {n_off} 건의 배포 꺼짐 시각 차(틱) {hist}")
        ok = (bad == 0) if f32 else True
        if not ok:
            fails += 1
        print("[PASS]" if ok else "[FAIL]", msg + ("" if f32 else " (INFO: 현행 기본 float64 타이머)"))
    print("c5 imu_f32:", "ALL PASS" if fails == 0 else f"{fails} FAILED")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
