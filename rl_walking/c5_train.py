"""PA1 (1006): 파이프라인 정렬 회복 학습 — 학습 루프를 배포 회복 파이프라인 KE 와 같게 만든다.

배경: KE = 서기 G21a-11300 + 회복 G24R-a-21793 + 감지기(RV4·det_from_obs·det_filt 0.33·cue 0.03/1 s·조기 복귀·불응기·early_gate).
회복 정책을 G24R-a 설정 그대로 이어학습하면 KE 평가가 나빠진다 — 학습 목표가 '감지기가 켜진 순간부터, G21a 가 남긴
액션 이력·cue 입력 아래에서의 회복'을 재지 않기 때문(가설). 그래서 학습 env 일부(파이프라인 env)를 배포 파이프라인과 같게 돌린다.

구성
- TorchSwitchDetector: rl_walking/deploy/c5_switch.SwitchDetector(+C5Runner.cue_update)의 torch 벡터판(N env, float64).
  시간 누적은 정수 틱 카운터 + '배포 모듈 float64 누적이 문턱에 닿는 틱 수'로 바꿔 배포 모듈과 결정이 비트 단위로 같다
  (KE 재생 대조에서 본 float32/64 1틱 경계 차 제거). selftest_detector() 가 같은 입력열 재생으로 검증한다.
- C5PipelineWrapper: RslRlVecEnvWrapper 바깥 래퍼. 매 틱
    관측 o_t(원본) → 감지기(관측 최신 프레임, 입력 EMA) → mode_t, cue_t
    rsl_rl 에 넘기는 관측 = o_t + (cue 면 policy 그룹 velocity_commands vx 5 프레임 ← 0.03) + obs["c5"] = [mask, mode, cue, pipe]
    step(a_pi): a_s = G21a(o_t 원본), a_exec = where(mask_t, a_pi, a_s) → env.step(a_exec)
  관측의 직전 액션 항(last_action_lpf)은 env.action_manager.action = a_exec 를 읽으므로 '직전 액션 = 실제 실행 액션' 계약이 자동 유지.
  파이프라인 env(비율 frac): 명령 서기 고정(vx=0) — 명령 항의 is_standing_env 를 리샘플마다 True 로 덮는다.
  앵커 env(1−frac): 원래 G24R-a 학습 그대로(명령·cue 없음, mask=1 상시).
- MaskedPPO: compute_returns 뒤 mask=1 표본으로만 advantage 정규화, mask=0 표본 advantage = 0(정책 기울기 0).
  GAE·가치 학습은 전 틱(가치 = 파이프라인 전체 행동의 가치). 선택: critic 예열 N 반복(전 mask 0 + 액터 고정).

자가검증(CPU, 시뮬 불필요):  /home/ryu/IsaacLab/isaaclab.sh -p rl_walking/c5_train.py --selftest
"""
from __future__ import annotations

import math
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))

# ── KE 배포 감지기 설정(eval_ground_recovery 인자 = reports/gen_c5/replay_kr.py·log_picture/*adk1* 실행 인자와 같음) ──
#   --sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 --sw_off_contact any
#   --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 --det_from_obs --det_filt 0.33
#   --sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu
KE_DET = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4,
              ref_tau_s=10.0, ref_cap_deg=8.0, ref_init_s=1.5, off_contact="any",
              early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33, early_gate=(5.0, 1.0))
KE_CUE = dict(cue_vx=0.03, cue_s=1.0, cue_dir_imu=True)
DT = 1.0 / 50.0
C5_KEY = "c5"          # 관측 TensorDict 의 학습 전용 그룹(obs_groups 밖 → 액터·크리틱 입력 아님)
C5_COLS = ("mask", "mode", "cue", "pipe")


def _import_switch():
    p = os.path.join(_HERE, "deploy")
    if p not in sys.path:
        sys.path.insert(0, p)
    import c5_switch  # noqa: E402
    return c5_switch


def _ticks_to_reach(thr: float, dt: float = DT, cap: int = 10 ** 6) -> int:
    """배포 모듈의 float64 누적 s_k = (((0+dt)+dt)+…) 가 처음 s_k ≥ thr 가 되는 k. (s_k < thr ⇔ k < 반환값)"""
    s = 0.0; k = 0
    while s < thr:
        s += dt; k += 1
        if k > cap:
            raise ValueError(thr)
    return k


# =====================================================================================================
# 감지기 (torch, N env)
# =====================================================================================================
class TorchSwitchDetector:
    """c5_switch.SwitchDetector + C5Runner.cue_update 의 벡터판. 상태: mode 0 = 서기(G21a), 1 = 회복.
    update 순서·조건은 배포 모듈 그대로(아래 주석의 줄 = c5_switch.py SwitchDetector.update).
    배포 모듈과 다른 점(평가 eval_ground_recovery 와의 차이 포함):
      - 시간 누적은 정수 틱(문턱 틱 수는 배포 float64 누적과 동일하게 산출 → 결정 비트 동일).
      - 조기 복귀와 calm 복귀가 같은 틱이면 불응기를 시작하지 않는다(배포 모듈 규약; 평가기는 이 경우에도 불응기 시작 — 드문 경계)."""

    def __init__(self, num_envs: int, device="cpu", dt: float = DT, on_tilt_deg=4.0, on_w=0.5, off_tilt_deg=2.5, off_w=0.3,
                 calm_s=1.0, dwell_s=0.4, ref_tau_s=0.0, ref_cap_deg=8.0, ref_init_s=0.0, off_contact="both",
                 early=None, refrac=None, in_filt=0.0, early_gate=None, cue_vx=None, cue_s=1.0, cue_dir_imu=True):
        self.N, self.dev, self.dt = int(num_envs), torch.device(device), float(dt)
        self.on_tilt, self.on_w, self.off_tilt, self.off_w = float(on_tilt_deg), float(on_w), float(off_tilt_deg), float(off_w)
        self.ref_tau, self.ref_cap, self.in_filt = float(ref_tau_s), float(ref_cap_deg), float(in_filt)
        if off_contact not in ("both", "any", "none"):
            raise ValueError(off_contact)
        self.off_contact = off_contact
        self.early = None if early is None else tuple(float(x) for x in early)
        self.refrac = None if refrac is None else tuple(float(x) for x in refrac)
        self.early_gate = None if early_gate is None else tuple(float(x) for x in early_gate)
        self.K_calm, self.K_dwell = _ticks_to_reach(float(calm_s), dt), _ticks_to_reach(float(dwell_s), dt)
        self.K_ref = _ticks_to_reach(float(ref_init_s), dt) if float(ref_init_s) > 0 else 0
        self.K_tmin = _ticks_to_reach(self.early[0], dt) if self.early else 0
        self.K_hold = _ticks_to_reach(self.early[3], dt) if self.early else 0
        self.K_rs = _ticks_to_reach(self.refrac[0], dt) if self.refrac else 0
        self.cue_vx = None if cue_vx is None else float(cue_vx)
        self.K_cue = _ticks_to_reach(float(cue_s) - 1e-9, dt)
        self.cue_dir_imu = bool(cue_dir_imu)
        self._BIG = 1 << 40
        N, d = self.N, self.dev
        z = lambda dt_=torch.long: torch.zeros(N, dtype=dt_, device=d)   # noqa: E731
        self.mode = z(torch.bool); self.n_calm = z(); self.n_ton = z(); self.n_eh = z(); self.n_ts = z()
        self.n_se = torch.full((N,), self._BIG, dtype=torch.long, device=d)
        self.has_ref = z(torch.bool); self.ref = torch.zeros(N, 2, dtype=torch.float64, device=d)
        self.has_f = z(torch.bool); self.fg = torch.zeros(N, 3, dtype=torch.float64, device=d); self.fw = torch.zeros_like(self.fg)
        self.pk_t = torch.zeros(N, dtype=torch.float64, device=d); self.pk_w = torch.zeros_like(self.pk_t)
        self.cue = z(torch.bool); self.n_cue_t = z()
        self.last_w = torch.zeros(N, 3, dtype=torch.float64, device=d)
        # 누적 통계(리셋과 무관)
        self.cnt_on = z(); self.cnt_early = z(); self.cnt_cue = z()

    def reset(self, ids_mask: torch.Tensor) -> None:
        m = ids_mask.to(self.dev).bool()
        if not bool(m.any()):
            return
        for t in (self.n_calm, self.n_ton, self.n_eh, self.n_ts, self.n_cue_t):
            t[m] = 0
        self.n_se[m] = self._BIG
        for t in (self.mode, self.has_ref, self.has_f, self.cue):
            t[m] = False
        self.pk_t[m] = 0.0; self.pk_w[m] = 0.0

    def update(self, gravity, gyro, contact2, count_mask=None):
        """(N,3) 중력 단위벡터, (N,3) 자이로, (N,2) 접촉 0/1 → (mode bool, cue bool). 입력은 float64 로 처리.
        count_mask: 통계 카운터(cnt_*) 누적 대상 env(None = 전부) — 결정에는 영향 없음."""
        cm = None if count_mask is None else count_mask.to(self.dev).bool()
        g = gravity.to(self.dev, torch.float64); w3 = gyro.to(self.dev, torch.float64)
        c = contact2.to(self.dev, torch.float64) > 0.5
        # filtered(): 첫 틱 = 입력, 이후 EMA; 중력은 필터 뒤 정규화
        if self.in_filt > 0.0:
            nf = ~self.has_f
            self.fg = torch.where(nf[:, None], g, self.fg); self.fw = torch.where(nf[:, None], w3, self.fw)
            self.has_f |= True
            self.fg = self.fg + self.in_filt * (g - self.fg); self.fw = self.fw + self.in_filt * (w3 - self.fw)
            g, w3 = self.fg, self.fw
        g = g / torch.clamp(torch.sqrt(g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1] + g[:, 2] * g[:, 2]), min=1e-6)[:, None]
        self.last_w = w3
        tilt = torch.rad2deg(torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)))
        w = torch.sqrt(w3[:, 0] * w3[:, 0] + w3[:, 1] * w3[:, 1])
        both = c[:, 0] & c[:, 1]
        mode = self.mode
        if self.ref_tau > 0.0:
            rp = torch.rad2deg(torch.stack([torch.atan2(g[:, 1], -g[:, 2]), torch.atan2(g[:, 0], -g[:, 2])], -1))
            init = (~self.has_ref) | (self.n_ts < self.K_ref)
            self.ref = torch.where(init[:, None], rp, self.ref); self.has_ref |= True
            upd = (~mode) & (w < 0.1) & both
            self.ref = torch.where(upd[:, None], torch.clamp(self.ref + (rp - self.ref) * (self.dt / self.ref_tau), -self.ref_cap, self.ref_cap), self.ref)
            dv = rp - self.ref
            tilt = torch.sqrt(dv[:, 0] * dv[:, 0] + dv[:, 1] * dv[:, 1])
        cq = {"both": both, "any": c[:, 0] | c[:, 1], "none": torch.ones_like(both)}[self.off_contact]
        quiet = (tilt < self.off_tilt) & (w < self.off_w) & cq
        self.n_calm = torch.where(mode & quiet, self.n_calm + 1, torch.zeros_like(self.n_calm))
        self.n_ton = torch.where(mode, self.n_ton + 1, torch.zeros_like(self.n_ton))
        off = mode & (self.n_calm >= self.K_calm) & (self.n_ton >= self.K_dwell)
        early_now = torch.zeros_like(mode)
        if self.early is not None:
            _, _tt, _tw, _ = self.early
            ec = mode & (self.n_ton >= self.K_tmin) & (tilt < _tt) & (w < _tw) & both
            if self.early_gate is not None:
                zero = torch.zeros_like(self.pk_t)
                self.pk_t = torch.where(mode, torch.maximum(self.pk_t, tilt), zero)
                self.pk_w = torch.where(mode, torch.maximum(self.pk_w, w), zero)
                ec = ec & ((self.pk_t >= self.early_gate[0]) | (self.pk_w >= self.early_gate[1]))
            self.n_eh = torch.where(ec, self.n_eh + 1, torch.zeros_like(self.n_eh))
            early_now = mode & (self.n_eh >= self.K_hold) & ~off
            off = off | early_now
            self.cnt_early += (early_now if cm is None else early_now & cm).long()
        # t_since_early: 조기 복귀 틱 = 0, 이후 +1
        self.n_se = torch.where(early_now, torch.zeros_like(self.n_se), torch.clamp(self.n_se + 1, max=self._BIG))
        on = (~mode) & ((tilt > self.on_tilt) | (w > self.on_w))
        if self.refrac is not None:
            in_ref = self.n_se < self.K_rs
            on = on & (~in_ref | (tilt > self.refrac[1]) | (w > self.refrac[2]))
        self.n_ts = self.n_ts + 1
        self.cnt_on += (on if cm is None else on & cm).long()
        new_mode = (mode | on) & ~off
        # C5Runner.cue_update(mode, prev)
        if self.cue_vx is not None:
            start = new_mode & ~mode
            fwd = (self.last_w[:, 1] > 0.0) if self.cue_dir_imu else torch.ones_like(start)
            cont = new_mode & mode & self.cue
            n_next = self.n_cue_t + 1
            cue = torch.where(start, fwd, torch.where(cont, n_next < self.K_cue, self.cue))
            self.n_cue_t = torch.where(start, torch.zeros_like(n_next), torch.where(cont, n_next, self.n_cue_t))
            cue = cue & new_mode
            self.cnt_cue += ((start & fwd) if cm is None else (start & fwd & cm)).long()
            self.cue = cue
        self.mode = new_mode
        return new_mode.clone(), self.cue.clone()


def selftest_detector(n_env: int = 24, T: int = 700, seed: int = 0, det_kw=None, cue_kw=None, verbose: bool = True) -> dict:
    """같은 합성 입력열(float32, 여러 env, 무작위 리셋 포함)을 배포 모듈(env 별 SwitchDetector + C5Runner.cue_update)과
    TorchSwitchDetector 에 넣어 매 틱 mode·cue 가 비트 단위로 같은지 본다. 불일치 시 AssertionError."""
    C = _import_switch()
    det_kw = dict(KE_DET if det_kw is None else det_kw); cue_kw = dict(KE_CUE if cue_kw is None else cue_kw)
    rng = np.random.default_rng(seed)
    # 합성 입력: 느린 기준 기울기 + 외란 사건(지수 감쇠 피치/롤 편차, ω = 미분 + 잡음) + 접촉 깜빡임·한 발
    G = np.zeros((T, n_env, 3), np.float32); W = np.zeros((T, n_env, 3), np.float32); Cc = np.ones((T, n_env, 2), np.float32)
    RST = np.zeros((T, n_env), bool)
    for e in range(n_env):
        p0, r0 = rng.normal(0, 2.0), rng.normal(0, 1.5)
        dp = np.zeros(T); dr = np.zeros(T)
        t = int(rng.integers(20, 120))
        while t < T:
            amp_p = rng.choice([-1, 1]) * rng.uniform(1.0, 12.0); amp_r = rng.normal(0, 3.0)
            tau = rng.uniform(0.15, 1.5) / DT; rise = int(rng.integers(3, 15))
            for k in range(t, T):
                x = (k - t) / rise if k - t < rise else math.exp(-(k - t - rise) / tau)
                dp[k] += amp_p * x; dr[k] += amp_r * x
            t += int(rng.integers(60, 260))
        pit = p0 + dp + np.cumsum(rng.normal(0, 0.03, T)); rol = r0 + dr + np.cumsum(rng.normal(0, 0.02, T))
        wy = np.gradient(np.radians(pit)) / DT + rng.normal(0, 0.06, T); wx = np.gradient(np.radians(rol)) / DT + rng.normal(0, 0.06, T)
        wz = rng.normal(0, 0.05, T)
        pr, rr = np.radians(pit), np.radians(rol)
        gx, gy, gz = np.tan(pr), np.tan(rr), -np.ones(T)
        nrm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        g = np.stack([gx / nrm, gy / nrm, gz / nrm], -1) + rng.normal(0, 0.01, (T, 3))
        G[:, e] = g; W[:, e] = np.stack([wx, wy, wz], -1)
        big = np.abs(dp) > 4.0
        for k in range(T):
            if big[k] and rng.random() < 0.3:
                Cc[k, e, int(rng.integers(0, 2))] = 0.0
            if rng.random() < 0.01:
                Cc[k, e, :] = 0.0
        for k in rng.integers(50, T, size=int(rng.integers(0, 3))):
            RST[k, e] = True
    # 배포 모듈
    dets = [C.SwitchDetector(**det_kw) for _ in range(n_env)]
    runs = []
    for d in dets:
        r = C.C5Runner.__new__(C.C5Runner); r.det = d; r.cue_vx = cue_kw["cue_vx"]; r.cue_s = cue_kw["cue_s"]; r.cue_dir_imu = cue_kw["cue_dir_imu"]
        r.cue_active = False; r.cue_t = 0.0; r.n_cue = 0; runs.append(r)
    prev = [0] * n_env
    tdet = TorchSwitchDetector(n_env, "cpu", **det_kw, **cue_kw)
    mism_m = mism_c = 0; first = None; n_mode1 = n_cue1 = 0; acc_on = acc_early = 0
    for k in range(T):
        rs = torch.as_tensor(RST[k])
        tdet.reset(rs)
        tm, tc = tdet.update(torch.as_tensor(G[k]), torch.as_tensor(W[k]), torch.as_tensor(Cc[k]))
        for e in range(n_env):
            if RST[k, e]:
                acc_on += dets[e].n_on; acc_early += dets[e].n_early      # 배포 reset() 은 n_on·n_early 를 0 으로 — 리셋 전 누적
                dets[e].reset(); runs[e].cue_active = False; runs[e].cue_t = 0.0; prev[e] = 0
            m = dets[e].update(G[k, e], W[k, e], Cc[k, e])
            cu = runs[e].cue_update(m, prev[e], W[k, e]); prev[e] = m
            n_mode1 += m; n_cue1 += int(cu)
            if m != int(tm[e]):
                mism_m += 1; first = first or (k, e, "mode", m, int(tm[e]))
            if int(cu) != int(tc[e]):
                mism_c += 1; first = first or (k, e, "cue", int(cu), int(tc[e]))
    res = dict(ticks=T * n_env, mode_mismatch=mism_m, cue_mismatch=mism_c, mode1_ticks=n_mode1, cue_ticks=n_cue1,
               on=acc_on + int(sum(d.n_on for d in dets)), early=acc_early + int(sum(d.n_early for d in dets)), cue_starts=int(sum(r.n_cue for r in runs)),
               t_on=int(tdet.cnt_on.sum()), t_early=int(tdet.cnt_early.sum()), t_cue_starts=int(tdet.cnt_cue.sum()), resets=int(RST.sum()))
    if verbose:
        print(f"[C5 selftest] 감지기 재생 대조: {res}" + (f" 첫 불일치 {first}" if first else ""), flush=True)
    assert mism_m == 0 and mism_c == 0, f"감지기 비트 불일치 {first}"
    assert res["on"] == res["t_on"] and res["early"] == res["t_early"] and res["cue_starts"] == res["t_cue_starts"], res
    assert res["on"] > 0 and res["early"] > 0 and res["cue_starts"] > 0 and n_mode1 > 0, f"합성 입력이 경로를 못 덮음 {res}"
    return res


def replay_eval_dump(tag: str, root: str = "/home/ryu/humanoid_leg_test1/log_picture", verbose: bool = True) -> dict:
    """KE 평가 덤프(recbb_<tag>_env*.jsonl 의 par_g·par_w·par_c = 감지기 원 입력, par_mode·par_cue = 평가기 결정)를 TorchSwitchDetector 로 재생해
    평가기(eval_ground_recovery, float32·GPU)와 틱 단위 비교. 로그 재생 = 계약 검증(폐루프 성능 증명 아님)."""
    import glob
    import json
    files = sorted(glob.glob(os.path.join(root, f"recbb_{tag}_env*.jsonl")))
    tot = mm = mc = 0; env_m = 0; first = None
    for fp in files:
        d = TorchSwitchDetector(1, "cpu", **KE_DET, **KE_CUE); bad = 0
        for line in open(fp):
            r = json.loads(line)
            if "par_mode" not in r:
                continue
            m, c = d.update(torch.tensor([r["par_g"]]), torch.tensor([r["par_w"]]), torch.tensor([r["par_c"]], dtype=torch.float64))
            tot += 1
            if int(m[0]) != int(r["par_mode"]):
                mm += 1; bad += 1; first = first or (os.path.basename(fp), r.get("t_rel"), "mode")
            if int(c[0]) != int(r["par_cue"]):
                mc += 1; bad += 1; first = first or (os.path.basename(fp), r.get("t_rel"), "cue")
        env_m += int(bad > 0)
    res = dict(tag=tag, envs=len(files), ticks=tot, mode_mismatch=mm, cue_mismatch=mc, envs_with_mismatch=env_m, first=first)
    if verbose:
        print(f"[C5 replay] 평가 덤프 재생: {res}", flush=True)
    return res


# =====================================================================================================
# 관측 레이아웃·계약 확인
# =====================================================================================================
def obs_layout(uenv, group: str = "policy") -> dict:
    """{항: (시작, 끝, 프레임 차원, 이력)} — 이력 평탄화는 [프레임0(最古) … 프레임L-1(최신)] (symmetry.py 주석·eval 과 같은 가정)."""
    om = uenv.observation_manager
    names = list(om.active_terms[group]); dims = om.group_obs_term_dim[group]
    cfgs = om._group_obs_term_cfgs[group]
    out = {}; o = 0
    for n, d, cf in zip(names, dims, cfgs):
        sz = 1
        for v in (d if isinstance(d, (tuple, list)) else (d,)):
            sz *= int(v)
        L = int(cf.history_length or 1)
        out[n] = (o, o + sz, sz // L, L); o += sz
    return out


def _yaml_obs_terms(env_yaml: str, group: str = "policy") -> list:
    """params/env.yaml 의 observations.<group> 항 (이름, func, history_length) 목록(텍스트 파서 — python 태그 회피)."""
    txt = open(env_yaml).read()
    i = txt.index("\nobservations:\n"); i = txt.index(f"\n  {group}:\n", i)
    m = re.search(r"\n  \S", txt[i + 1:]); seg = txt[i + 1: i + 1 + m.start()] if m else txt[i + 1:]
    terms = []; cur = None
    for line in seg.splitlines():
        mt = re.match(r"^    (\w+):\s*$", line)
        if mt:
            cur = [mt.group(1), None, None]; terms.append(cur); continue
        if cur is not None:
            mf = re.match(r"^      func: (\S+)", line)
            if mf:
                cur[1] = mf.group(1)
            mh = re.match(r"^      history_length: (\S+)", line)
            if mh:
                cur[2] = None if mh.group(1) == "null" else int(mh.group(1))
    return [tuple(t) for t in terms]


def _live_obs_terms(uenv, group: str = "policy") -> list:
    om = uenv.observation_manager
    out = []
    for n, cf in zip(om.active_terms[group], om._group_obs_term_cfgs[group]):
        f = cf.func
        fn = f"{f.__module__}:{f.__name__}" if hasattr(f, "__name__") else f"{type(f).__module__}:{type(f).__name__}"
        out.append((n, fn, cf.history_length))
    return out


def check_obs_contract(uenv, ckpts: dict) -> dict:
    """ckpts = {이름: 체크포인트 경로}. 각 런의 params/env.yaml 관측 항(이름·func·이력)이 현재 env 와 같고,
    체크포인트 액터/크리틱 입력 차원이 현재 관측 차원과 같은지 assert. velocity_commands 레이아웃도 확인."""
    live = {g: _live_obs_terms(uenv, g) for g in ("policy", "critic")}
    lay = obs_layout(uenv, "policy")
    pdim = max(b for _, b, _, _ in lay.values()); cdim = max(b for _, b, _, _ in obs_layout(uenv, "critic").values())
    rep = {"policy_dim": pdim, "critic_dim": cdim}
    for nm, ck in ckpts.items():
        ey = os.path.join(os.path.dirname(ck), "params", "env.yaml")
        for g in ("policy", "critic"):
            y = _yaml_obs_terms(ey, g)
            assert y == live[g], f"[C5] {nm} 관측 계약 불일치({g}):\n  yaml {y}\n  live {live[g]}"
        sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
        a_in = sd["actor.0.weight"].shape[1]; c_in = sd["critic.0.weight"].shape[1]
        assert a_in == pdim, f"[C5] {nm} 액터 입력 {a_in} ≠ policy {pdim}"
        assert c_in in (pdim + cdim, cdim), f"[C5] {nm} 크리틱 입력 {c_in} ≠ policy+critic {pdim + cdim}"
        rep[nm] = dict(actor_in=a_in, critic_in=c_in, terms=len(live["policy"]))
    vc = lay["velocity_commands"]
    assert vc[2] == 3, f"velocity_commands 프레임 차원 {vc[2]} ≠ 3"
    for k, d in (("base_ang_vel", 3), ("projected_gravity", 3), ("feet_contact", 2)):
        assert lay[k][2] == d, f"{k} 프레임 차원 {lay[k][2]} ≠ {d}"
    rep["velocity_commands"] = vc[:2]
    rep["vx_cols"] = [vc[0] + 3 * k for k in range(vc[3])]
    return rep


# =====================================================================================================
# 파이프라인 래퍼
# =====================================================================================================
def patch_standing_commands(uenv, pipe: torch.Tensor, command_name: str = "base_velocity") -> None:
    """파이프라인 env 의 속도 명령을 서기(0)로 고정: 리샘플 뒤 is_standing_env=True·명령 0. _update_command 가 매 틱 0 으로 유지."""
    term = uenv.command_manager.get_term(command_name)
    orig = term._resample_command
    pipe = pipe.to(term.vel_command_b.device)

    def _resample_pipe_standing(env_ids):
        orig(env_ids)
        if isinstance(env_ids, slice):
            ids = torch.arange(pipe.numel(), device=pipe.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=pipe.device, dtype=torch.long).reshape(-1)
        sel = ids[pipe[ids]]
        if sel.numel():
            term.is_standing_env[sel] = True
            term.vel_command_b[sel] = 0.0

    term._resample_command = _resample_pipe_standing
    term.is_standing_env[pipe] = True
    term.vel_command_b[pipe] = 0.0


class C5PipelineWrapper:
    """RslRlVecEnvWrapper 바깥 래퍼(위 모듈 설명 참고). rsl_rl 이 쓰는 속성(num_envs·num_actions·max_episode_length·
    episode_length_buf·device·cfg·get_observations·step)을 제공한다."""

    def __init__(self, env, stand_policy, frac: float = 0.75, seed: int = 0, layout: dict | None = None, arm_s: float = 1.0,
                 det_kw=None, cue_kw=None, patch_commands: bool = True, log_every: int = 0):
        self.env = env
        self.stand_policy = stand_policy
        N = int(env.num_envs); dev = torch.device(env.device)
        self.N, self.dev = N, dev
        if layout is None:
            u = env.unwrapped
            assert abs(float(u.step_dt) - DT) < 1e-9, f"정책 주기 {u.step_dt} ≠ {DT}"
            layout = {"policy": obs_layout(u, "policy"), "critic": obs_layout(u, "critic")}
        lp = layout["policy"]
        self.sl_w = slice(lp["base_ang_vel"][1] - 3, lp["base_ang_vel"][1])
        self.sl_g = slice(lp["projected_gravity"][1] - 3, lp["projected_gravity"][1])
        self.sl_c = slice(lp["feet_contact"][1] - 2, lp["feet_contact"][1])
        self.vx_cols = {}
        for grp, lg in layout.items():
            if "velocity_commands" in lg:
                a, _, fd, L = lg["velocity_commands"]
                self.vx_cols[grp] = torch.tensor([a + fd * k for k in range(L)], device=dev)   # vx = 프레임 첫 채널
        gen = torch.Generator(device="cpu"); gen.manual_seed(int(seed) + 7919)
        n_pipe = int(round(float(frac) * N))
        pipe = torch.zeros(N, dtype=torch.bool); pipe[torch.randperm(N, generator=gen)[:n_pipe]] = True
        self.pipe = pipe.to(dev)
        self.det = TorchSwitchDetector(N, dev, **dict(KE_DET if det_kw is None else det_kw), **dict(KE_CUE if cue_kw is None else cue_kw))
        self.cue_vx = float(self.det.cue_vx) if self.det.cue_vx is not None else None
        if patch_commands and n_pipe > 0:
            patch_standing_commands(env.unwrapped, self.pipe)
        self._raw = None; self._obs = None; self._mask = None
        self.stats = {}; self._reset_stats()
        self.log_every = int(log_every); self._nstep = 0
        # 감지기 무장 지연: 에피소드 시작 뒤 arm_s 동안은 mode 0(G21a) 고정·감지기 초기화 유지, 무장 틱에 첫 update(평가기의 인계 시점 초기화와 같은 규약).
        # 근거(스모크 1006): 무장 지연 없이 스폰 과도에서 파이프라인 env 49/48 켜짐 → 첫 반복 mode1 99.5 %(평가 KE 는 precheck 뒤 인계 시점에 감지기 시작).
        self.K_arm = int(round(float(arm_s) / DT)); self.ep_ticks = torch.zeros(N, dtype=torch.long, device=dev)
        _vx = {k: v.tolist() for k, v in self.vx_cols.items()}
        print(f"[C5] 파이프라인 env {n_pipe}/{N} (frac {frac}) · 앵커(정책 상시) {N - n_pipe} · vx 열 {_vx}"
              f" · 감지 입력 자이로 {self.sl_w} 중력 {self.sl_g} 접촉 {self.sl_c}", flush=True)

    # ── rsl_rl VecEnv 인터페이스 ──
    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def num_actions(self):
        return self.env.num_actions

    @property
    def max_episode_length(self):
        return self.env.max_episode_length

    @property
    def device(self):
        return self.env.device

    @property
    def cfg(self):
        return self.env.cfg

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def __getattr__(self, name):
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)

    def close(self):
        return self.env.close()

    def _reset_stats(self):
        self.stats = dict(pipe_ticks=0, pipe_mode1=0, anchor_ticks=0, cue_ticks=0, mask_ticks=0, ticks=0)
        self._cnt0 = (int(self.det.cnt_on[self.pipe].sum()), int(self.det.cnt_early[self.pipe].sum()), int(self.det.cnt_cue[self.pipe].sum()))

    def pop_stats(self) -> dict:
        s = dict(self.stats)
        on = int(self.det.cnt_on[self.pipe].sum()) - self._cnt0[0]
        ea = int(self.det.cnt_early[self.pipe].sum()) - self._cnt0[1]
        cu = int(self.det.cnt_cue[self.pipe].sum()) - self._cnt0[2]
        out = {"c5_pipe_mode1_frac": s["pipe_mode1"] / max(1, s["pipe_ticks"]), "c5_mask_frac_env": s["mask_ticks"] / max(1, s["ticks"]),
               "c5_cue_frac_pipe": s["cue_ticks"] / max(1, s["pipe_ticks"]), "c5_on_events": float(on), "c5_early_off": float(ea),
               "c5_cue_starts": float(cu)}
        self._reset_stats()
        return out

    def _process(self, raw, done_mask):
        po = raw["policy"]
        done_mask = done_mask.to(self.dev).bool()
        self.ep_ticks = torch.where(done_mask, torch.zeros_like(self.ep_ticks), self.ep_ticks + 1) if self._obs is not None else torch.zeros_like(self.ep_ticks)
        armed = self.ep_ticks >= self.K_arm
        self.det.reset(done_mask | (self.ep_ticks <= self.K_arm))      # 무장 틱(=K_arm)까지 초기화 → 무장 틱 update 가 첫 틱
        mode, cue = self.det.update(po[:, self.sl_g], po[:, self.sl_w], po[:, self.sl_c], count_mask=armed & self.pipe)
        mode = mode.to(self.dev) & armed; cue = cue.to(self.dev) & armed
        pipe = self.pipe
        mode_p = mode & pipe                     # 파이프라인 env 의 실행 모드
        cue_p = cue & mode_p
        mask = mode_p | ~pipe                    # 앵커 env 는 정책 상시
        obs = raw.clone()
        if self.cue_vx is not None and bool(cue_p.any()):
            for grp, cols in self.vx_cols.items():
                x = obs[grp].clone()
                sub = x[:, cols]
                x[:, cols] = torch.where(cue_p[:, None], torch.full_like(sub, self.cue_vx), sub)
                obs[grp] = x
        obs[C5_KEY] = torch.stack([mask, mode_p, cue_p, pipe], -1).to(po.dtype)
        self._raw, self._obs, self._mask = raw, obs, mask
        st = self.stats
        st["ticks"] += self.N; st["pipe_ticks"] += int(pipe.sum()); st["anchor_ticks"] += int((~pipe).sum())
        st["pipe_mode1"] += int(mode_p.sum()); st["cue_ticks"] += int(cue_p.sum()); st["mask_ticks"] += int(mask.sum())
        return obs

    def get_observations(self):
        if self._obs is None:
            raw = self.env.get_observations()
            self._process(raw, torch.ones(self.N, dtype=torch.bool, device=self.dev))
        return self._obs

    def step(self, actions):
        with torch.inference_mode():
            a_s = self.stand_policy(self._raw)               # G21a 는 원본 관측(cue 없음)
        a_exec = torch.where(self._mask[:, None], actions.to(a_s.dtype), a_s)
        self.last_a_exec = a_exec
        obs, rew, dones, extras = self.env.step(a_exec)
        self._process(obs, dones.to(self.dev) > 0)
        self._nstep += 1
        if self.log_every and self._nstep % self.log_every == 0:
            print(f"[C5] step {self._nstep} mode1(pipe) {int((self._obs[C5_KEY][:, 1] > 0.5).sum())}/{int(self.pipe.sum())}", flush=True)
        return self._obs, rew, dones, extras


# =====================================================================================================
# MaskedPPO
# =====================================================================================================
def make_masked_ppo(base):
    """base(PPO 또는 SmoothPPO 등) 위에 마스크 advantage 를 얹은 하위 클래스를 만든다."""

    class MaskedPPO(base):
        WARMUP = 0            # critic 예열 반복 수(이 동안 mask 전부 0 + 액터 고정 + 엔트로피·미러 손실 0 + 학습률 고정)
        RESCALE = True        # mask=1 advantage 에 (전 표본/mask 표본) 배율 — 손실 평균이 전 표본이라 생기는 희석 보정(상한 RESCALE_CAP)
        RESCALE_CAP = 10.0
        WRAPPER = None        # C5PipelineWrapper(통계 출력용)
        VERBOSE = True

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if self.normalize_advantage_per_mini_batch:
                raise ValueError("MaskedPPO: normalize_advantage_per_mini_batch=True 는 mask 0 표본을 다시 정규화해 계약을 깬다")
            self._c5_it = 0; self._c5_stats = {}
            self._c5_critic_ids = {id(p) for p in self.policy.critic.parameters()}
            opt_step = self.optimizer.step

            def step_c5(*aa, **kk):
                if self._c5_warm():
                    for p in self.policy.parameters():
                        if id(p) not in self._c5_critic_ids:
                            p.grad = None              # Adam 이 액터 파라미터를 건너뜀(모멘텀 이동도 없음)
                return opt_step(*aa, **kk)

            self.optimizer.step = step_c5

        def _c5_warm(self) -> bool:
            return self._c5_it < int(self.WARMUP)

        def compute_returns(self, obs):
            last_values = self.policy.evaluate(obs).detach()
            self.storage.compute_returns(last_values, self.gamma, self.lam, normalize_advantage=False)
            adv = self.storage.advantages
            m = self.storage.observations[C5_KEY][..., 0:1] > 0.5
            if self._c5_warm():
                m = torch.zeros_like(m)
            n = int(m.sum()); tot = m.numel()
            st = {"c5_mask_n": float(n), "c5_mask_frac": n / max(1, tot), "c5_adv_raw_all_mean": float(adv.mean()),
                  "c5_adv_raw_all_std": float(adv.std())}
            if n >= 2:
                am = adv[m]
                mu, sd = am.mean(), am.std()
                scale = min(tot / n, float(self.RESCALE_CAP)) if self.RESCALE else 1.0
                a = (adv - mu) / (sd + 1e-8) * scale
                st.update(c5_adv_raw_mask_mean=float(mu), c5_adv_raw_mask_std=float(sd), c5_adv_scale=float(scale))
            else:
                a = torch.zeros_like(adv)
                st.update(c5_adv_raw_mask_mean=0.0, c5_adv_raw_mask_std=0.0, c5_adv_scale=0.0)
            self.storage.advantages = torch.where(m, a, torch.zeros_like(a))
            fa = self.storage.advantages[m] if n else torch.zeros(1, device=adv.device)
            st.update(c5_adv_mask_mean=float(fa.mean()), c5_adv_mask_std=float(fa.std()) if n > 1 else 0.0,
                      c5_adv_zero_n=float(int((self.storage.advantages == 0).sum())), c5_warmup=float(self._c5_warm()))
            if self.WRAPPER is not None:
                st.update(self.WRAPPER.pop_stats())
            self._c5_stats = st
            if self.VERBOSE:
                print("[C5] it {} | mask 표본 {}/{} ({:.3f}) | 파이프라인 mode1 비율 {:.4f} · cue 비율 {:.4f} · 켜짐 {} · 조기복귀 {} · cue 시작 {} | "
                      "adv 원(전) {:+.4f}±{:.4f} 원(mask) {:+.4f}±{:.4f} → mask 정규화 {:+.4f}±{:.4f} (배율 {:.3f}) · 0 표본 {}{}".format(
                          self._c5_it, n, tot, st["c5_mask_frac"], st.get("c5_pipe_mode1_frac", float("nan")), st.get("c5_cue_frac_pipe", float("nan")),
                          int(st.get("c5_on_events", -1)), int(st.get("c5_early_off", -1)), int(st.get("c5_cue_starts", -1)),
                          st["c5_adv_raw_all_mean"], st["c5_adv_raw_all_std"], st["c5_adv_raw_mask_mean"], st["c5_adv_raw_mask_std"],
                          st["c5_adv_mask_mean"], st["c5_adv_mask_std"], st["c5_adv_scale"], int(st["c5_adv_zero_n"]),
                          " · critic 예열" if self._c5_warm() else ""), flush=True)

        def update(self):
            warm = self._c5_warm()
            saved = None
            if warm:
                saved = (self.entropy_coef, self.schedule, (self.symmetry or {}).get("use_mirror_loss") if self.symmetry else None)
                self.entropy_coef = 0.0; self.schedule = "fixed"
                if self.symmetry:
                    self.symmetry["use_mirror_loss"] = False
            try:
                d = super().update()
            finally:
                if saved is not None:
                    self.entropy_coef, self.schedule = saved[0], saved[1]
                    if self.symmetry:
                        self.symmetry["use_mirror_loss"] = saved[2]
            d.update(self._c5_stats)
            self._c5_it += 1
            return d

    MaskedPPO.__name__ = f"Masked{base.__name__}"
    return MaskedPPO


# =====================================================================================================
# train.py 연결
# =====================================================================================================
def resolve_ckpt(spec: str, log_root: str) -> str:
    if os.path.isfile(spec):
        return os.path.abspath(spec)
    run, ck = spec.split(":")
    from isaaclab_tasks.utils import get_checkpoint_path
    return get_checkpoint_path(log_root, run, ck)


def load_stand_policy(env, agent_cfg, ckpt: str):
    """고정 서기 정책(G21a)을 두 번째 OnPolicyRunner 로 로드 — eval_ground_recovery.py(policy 로드) 방식. 추론 함수 반환."""
    from rsl_rl.runners import OnPolicyRunner
    r = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    r.load(ckpt)
    pol = r.get_inference_policy(device=env.unwrapped.device)
    for p in r.alg.policy.parameters():
        p.requires_grad_(False)
    r.alg.storage = None          # 롤아웃 버퍼 해제(추론 전용)
    return pol, r


def install(env, agent_cfg, args, log_root: str, resume_path: str | None):
    """train.py 에서 호출: 사전 확인 → G21a 로드 → 래퍼 → MaskedPPO 주입. 반환 = 래핑된 env."""
    import rsl_rl.runners.on_policy_runner as _opr
    print("[C5] PA1 파이프라인 정렬 학습 — 사전 확인", flush=True)
    selftest_detector()                                     # 감지기 재생 비트 일치(CPU)
    stand_ckpt = resolve_ckpt(args.c5_stand, log_root)
    ck = {"stand": stand_ckpt}
    if resume_path:
        ck["resume"] = resume_path
    rep = check_obs_contract(env.unwrapped, ck)
    print(f"[C5] 관측 계약 OK: {rep}", flush=True)
    assert rep["velocity_commands"] == (30, 45), f"velocity_commands 위치 {rep['velocity_commands']} (기대 30:45 — KE 평가 레이아웃)"
    if agent_cfg.algorithm.class_name != "PPO":
        raise ValueError(f"C5: 알고리즘 {agent_cfg.algorithm.class_name} 미지원")
    stand_pol, _r = load_stand_policy(env, agent_cfg, stand_ckpt)
    print(f"[C5] 고정 서기 정책 = {stand_ckpt}", flush=True)
    wenv = C5PipelineWrapper(env, stand_pol, frac=float(args.c5_frac), seed=int(agent_cfg.seed or 0), arm_s=float(getattr(args, "c5_arm_s", 1.0)))
    M = make_masked_ppo(_opr.PPO)
    M.WARMUP = int(args.c5_critic_warmup); M.RESCALE = not bool(getattr(args, "c5_no_rescale", False)); M.WRAPPER = wenv
    _opr.PPO = M
    print(f"[C5] {M.__name__} 주입 (critic 예열 {M.WARMUP} 반복, advantage 배율 {'켬' if M.RESCALE else '끔'})", flush=True)
    wenv._stand_runner = _r
    return wenv


# =====================================================================================================
# CPU 자가검증
# =====================================================================================================
class _FakeVecEnv:
    """레이아웃만 맞춘 가짜 VecEnv(시뮬 없음) — 래퍼 배선 검증용. 관측 = 내부 상태에서 만든 320+15 차원."""

    def __init__(self, N=8, seed=0):
        from tensordict import TensorDict
        self.TD = TensorDict
        self.num_envs = N; self.num_actions = 12; self.max_episode_length = 1000; self.device = "cpu"
        self.episode_length_buf = torch.zeros(N, dtype=torch.long); self.cfg = None; self.unwrapped = self
        self.rng = np.random.default_rng(seed); self.t = 0; self.actions_log = []
        names = [("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3), ("joint_pos", 12), ("joint_vel", 12), ("actions", 12),
                 ("gait_clock", 2), ("feet_contact", 2), ("gyro_ema", 3), ("joint_vel_ema", 12)]
        lay = {}; o = 0
        for n, d in names:
            lay[n] = (o, o + 5 * d, d, 5); o += 5 * d
        self.layout = {"policy": lay, "critic": {"base_lin_vel": (0, 15, 3, 5)}}
        self.pitch = np.zeros(N)

    def _obs(self):
        N = self.num_envs
        x = torch.zeros(N, 320); L = self.layout["policy"]
        for n, (a, b, d, h) in L.items():
            x[:, a:b] = torch.as_tensor(self.rng.normal(0, 0.05, (N, b - a)), dtype=torch.float32)
        p = np.radians(self.pitch)
        g = np.stack([np.sin(p), np.zeros(N), -np.cos(p)], -1)
        for k in range(5):
            a = L["projected_gravity"][0] + 3 * k; x[:, a:a + 3] = torch.as_tensor(g, dtype=torch.float32)
            a = L["feet_contact"][0] + 2 * k; x[:, a:a + 2] = 1.0
            a = L["velocity_commands"][0] + 3 * k; x[:, a:a + 3] = 0.0
            a = L["base_ang_vel"][0] + 3 * k; x[:, a + 1] = torch.as_tensor(self.wy, dtype=torch.float32) if hasattr(self, "wy") else 0.0
        return self.TD({"policy": x, "critic": torch.zeros(N, 15)}, batch_size=[N])

    def get_observations(self):
        self.wy = np.zeros(self.num_envs)
        return self._obs()

    def step(self, a):
        self.actions_log.append(a.clone()); self.t += 1
        N = self.num_envs
        old = self.pitch.copy()
        self.pitch = np.where((self.t % 120 > 20) & (self.t % 120 < 35), self.pitch + 0.6, self.pitch * 0.9)   # 주기적 앞 기울임
        self.wy = np.radians(self.pitch - old) / DT
        dones = torch.zeros(N, dtype=torch.long)
        if self.t % 97 == 0:
            dones[0] = 1; self.pitch[0] = 0.0
        return self._obs(), torch.zeros(N), dones, {}


def selftest_wrapper() -> str:
    """가짜 env 로 배선 확인: a_exec = mask ? a_pi : a_s, 앵커 env 상시 a_pi, cue 는 파이프라인·mode1·cue 틱의 vx 열에만 0.03,
    G21a 는 원본 관측(cue 없음)을 받는다, c5 그룹 = [mask, mode, cue, pipe]."""
    env = _FakeVecEnv(N=8)
    seen = []

    def stand(o):
        seen.append(o["policy"].clone()); return torch.full((o["policy"].shape[0], 12), -1.0)

    w = C5PipelineWrapper(env, stand, frac=0.75, seed=0, layout=env.layout, patch_commands=False, arm_s=0.5)
    vx = w.vx_cols["policy"]
    o = w.get_observations()
    n_cue = n_m1 = 0
    for t in range(400):
        c5 = o[C5_KEY]; mask, mode, cue, pipe = (c5[:, i] > 0.5 for i in range(4))
        assert torch.equal(pipe, w.pipe) and torch.equal(mask, mode | ~pipe) and not bool((cue & ~mode).any())
        assert not bool((mode & (w.ep_ticks < w.K_arm)).any()), "무장 전 mode 1"
        raw = w._raw["policy"]
        # cue 열만 바뀌고, 바뀐 곳은 cue env 뿐
        diff = (o["policy"] != raw)
        cols = torch.zeros(320, dtype=torch.bool); cols[vx] = True
        assert not bool(diff[:, ~cols].any()), "cue 가 vx 열 밖을 바꿈"
        assert torch.equal(diff.any(1), cue), "cue env 와 관측 변경 env 불일치"
        if bool(cue.any()):
            assert torch.allclose(o["policy"][cue][:, vx], torch.full_like(o["policy"][cue][:, vx], 0.03))
        n_cue += int(cue.sum()); n_m1 += int(mode.sum())
        a_pi = torch.full((8, 12), 1.0)
        o2, _, _, _ = w.step(a_pi)
        assert torch.equal(seen[-1], raw), "G21a 가 원본 관측을 받지 않음"
        a_exec = env.actions_log[-1]
        exp = torch.where(mask[:, None], a_pi, torch.full_like(a_pi, -1.0))
        assert torch.equal(a_exec, exp), "a_exec ≠ where(mask, a_pi, a_s)"
        o = o2
    assert n_m1 > 0 and n_cue > 0, (n_m1, n_cue)
    return f"wrapper OK (mode1 틱 {n_m1}, cue 틱 {n_cue}, 파이프라인 env {int(w.pipe.sum())}/8)"


def selftest_masked_ppo() -> str:
    """CPU 소형 ActorCritic + MaskedPPO: mask 0 advantage = 0, mask 1 정규화(평균 0·표준편차 = 배율), update 동작,
    critic 예열 중 액터 파라미터 불변·크리틱 변화."""
    from rsl_rl.algorithms import PPO
    from rsl_rl.modules import ActorCritic
    from tensordict import TensorDict
    torch.manual_seed(0)
    N, T = 16, 24
    mk = lambda: TensorDict({"policy": torch.randn(N, 10), "critic": torch.randn(N, 3), C5_KEY: (torch.rand(N, 4) < 0.3).float()}, batch_size=[N])  # noqa: E731
    obs0 = mk()
    out = []
    for warm in (0, 1):
        M = make_masked_ppo(PPO); M.WARMUP = warm; M.VERBOSE = True
        pol = ActorCritic(obs0, {"policy": ["policy"], "critic": ["policy", "critic"]}, 4, actor_hidden_dims=[16], critic_hidden_dims=[16])
        alg = M(pol, num_learning_epochs=2, num_mini_batches=2,
                symmetry_cfg=None)
        alg.init_storage("rl", N, T, obs0, [4])
        a0 = [p.detach().clone() for n, p in pol.named_parameters() if not n.startswith("critic")]
        c0 = [p.detach().clone() for p in pol.critic.parameters()]
        for t in range(T):
            ob = mk()
            alg.act(ob)
            alg.process_env_step(mk(), torch.randn(N), (torch.rand(N) < 0.05).long(), {"time_outs": torch.zeros(N)})
        alg.compute_returns(mk())
        m = alg.storage.observations[C5_KEY][..., 0:1] > 0.5
        adv = alg.storage.advantages
        if warm:
            assert bool((adv == 0).all()), "예열 중 advantage ≠ 0"
        else:
            assert bool((adv[~m] == 0).all()), "mask 0 advantage ≠ 0"
            sc = alg._c5_stats["c5_adv_scale"]
            assert abs(float(adv[m].mean())) < 1e-4 and abs(float(adv[m].std()) - sc) < 1e-3 * sc, (float(adv[m].mean()), float(adv[m].std()), sc)
        d = alg.update()
        a1 = [p.detach() for n, p in pol.named_parameters() if not n.startswith("critic")]
        c1 = [p.detach() for p in pol.critic.parameters()]
        a_same = all(torch.equal(x, y) for x, y in zip(a0, a1)); c_same = all(torch.equal(x, y) for x, y in zip(c0, c1))
        if warm:
            assert a_same and not c_same, f"예열: 액터 불변 {a_same} · 크리틱 변화 {not c_same}"
        else:
            assert not a_same and not c_same
        assert "c5_mask_frac" in d
        out.append(f"warm={warm}: mask {int(d['c5_mask_n'])}/{N * T} 액터 불변 {a_same} 크리틱 불변 {c_same}")
    return "MaskedPPO OK (" + "; ".join(out) + ")"


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        r = selftest_detector()
        selftest_detector(n_env=32, T=900, seed=1)
        # 절대 기울기·필터 없음·조기 복귀 없음 변형(배포 모듈 기본값)도 비트 일치 — 경로 일반성 확인
        selftest_detector(n_env=16, T=600, seed=2, det_kw=dict(on_tilt_deg=4.0, on_w=0.5, off_tilt_deg=2.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, early=(0.6, 6.0, 0.4, 0.2)),
                          cue_kw=dict(cue_vx=0.05, cue_s=1.0, cue_dir_imu=False))
        for _tg in ("kepar_nd_s3", "kepar_sf15_s3", "kepar_b18_s3"):      # 평가기(KE 인자) 결정과의 대조 — 정보용(float32 경계 차 가능)
            if os.path.exists(f"/home/ryu/humanoid_leg_test1/log_picture/recbb_{_tg}_env0.jsonl"):
                replay_eval_dump(_tg)
        print(selftest_wrapper(), flush=True)
        print(selftest_masked_ppo(), flush=True)
        print("C5 train selftest PASS", flush=True)
