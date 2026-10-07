"""W3 걷기 회복 supervisor 코어(1007, 학습 없음·정책 밖) — 실물 관측만 사용: IMU 피치·롤(중력 방향), 자이로 |ω_xy|.
설계·사전 기준: reports/gen_w3/WD_DESIGN.md §10. 평가 eval_w3 --rsup, 배포 rl_walking/deploy/w3_deploy.py(기본 꺼짐) 가 같은 식을 쓴다.

상태(env 별):
  기준(base)      = 피치·롤·|ω| EMA(τ_base), **감지 없음(QUIET)일 때만** 갱신 — 외란 시각을 모름(시뮬 참값 불사용).
  편차 devs       = (피치, 롤) − 기준, 짧은 EMA(τ_dev).
  감지            = |devs| > th_on 이 n_on 틱 연속 & 불응 끝 → ACTIVE(오프셋 창 off_T 초) → RECOVER(정숙 확인까지).
  정숙(quiet)     = 주기 EMA(τ_cyc) 피치·롤이 기준 ± q_ang, |ω| 주기 EMA ≤ q_wk·기준 + q_wa, q_hold 초 유지 → QUIET.
(a) 정지 래치 지연: 정지 요청이 와도 QUIET 이 아니면 ge 0 요청을 미룬다(최대 stop_tmax 초, 넘으면 그대로 정지).
(b) 포획점 명령 오프셋: ACTIVE 동안 명령 (vx, vy) += A·(devs 방향 단위벡터 — 피치 + = 앞 기울임 → +vx, 롤 − = 왼 기울임 → +vy),
    결과를 학습 범위 [vx_lo, vx_hi]·[−vy_max, vy_max] 로 클립. 방향은 감지 순간 값으로 고정.
모든 연산은 torch 텐서(N,) — 배포는 N = 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

QUIET, ACTIVE, RECOVER = 0, 1, 2


@dataclass
class RSupParams:
    th_on: float = 4.0        # [deg] 감지 문턱(무외란 걷기 보정 실행으로 사전 등록)
    n_on: int = 2             # 연속 틱
    tau_base: float = 2.0
    tau_dev: float = 0.1
    tau_cyc: float = 0.91
    q_ang: float = 2.0
    q_wk: float = 1.5
    q_wa: float = 0.05
    q_hold: float = 0.45
    off_T: float = 0.91       # 오프셋 창(≈ 2 보)
    refr: float = 2.0         # 감지 불응(오프셋 창 시작 기준)
    A: float = 0.04           # 오프셋 크기 [m/s]
    vx_lo: float = -0.06
    vx_hi: float = 0.10
    vy_max: float = 0.06
    stop_tmax: float = 3.0
    use_offset: bool = False  # (b)
    use_stopgate: bool = False  # (a)
    roll_sign: float = -1.0   # vy = roll_sign·롤 편차(보정 실행으로 부호 확인)


class RSup:
    def __init__(self, N: int, dev, P: RSupParams, dt: float):
        self.P, self.dt, self.N = P, dt, N
        z = lambda: torch.zeros(N, device=dev)
        self.base = torch.zeros(N, 3, device=dev); self.binit = torch.zeros(N, dtype=torch.bool, device=dev)
        self.devs = torch.zeros(N, 2, device=dev); self.cyc = torch.zeros(N, 3, device=dev)
        self.state = torch.zeros(N, dtype=torch.long, device=dev)
        self.cnt = z(); self.t_act = z(); self.t_q = z(); self.t_ref = torch.full((N,), 1e9, device=dev)
        self.dirv = torch.zeros(N, 2, device=dev)
        self.stop_wait = z(); self.stop_req = torch.zeros(N, dtype=torch.bool, device=dev); self.stop_latched = torch.zeros(N, dtype=torch.bool, device=dev)
        self.n_trig = z()                        # 감지 횟수(오발동 집계용)

    def reset(self, mask):
        for x in (self.cnt, self.t_act, self.t_q, self.stop_wait):
            x[mask] = 0.0
        self.t_ref[mask] = 1e9; self.state[mask] = QUIET; self.binit[mask] = False
        self.stop_req[mask] = False; self.stop_latched[mask] = False; self.n_trig[mask] = 0.0

    def step(self, pitch, roll, wxy, walking):
        """한 틱 갱신(정책 스텝 뒤 관측). pitch·roll [deg], wxy [rad/s], walking(bool, ge). 반환 없음 — offset()/stop_ok() 로 읽는다."""
        P, dt = self.P, self.dt
        x = torch.stack([pitch, roll, wxy], -1)
        init = ~self.binit
        self.base = torch.where(init.unsqueeze(1), x, self.base); self.cyc = torch.where(init.unsqueeze(1), x, self.cyc)
        self.binit = self.binit | init
        a_c = min(1.0, dt / P.tau_cyc); self.cyc = self.cyc + a_c * (x - self.cyc)
        a_d = min(1.0, dt / P.tau_dev); self.devs = self.devs + a_d * ((x[:, :2] - self.base[:, :2]) - self.devs)
        q = self.state == QUIET
        a_b = min(1.0, dt / P.tau_base)
        self.base = torch.where(q.unsqueeze(1), self.base + a_b * (x - self.base), self.base)
        mag = self.devs.norm(dim=-1)
        self.cnt = torch.where(mag > P.th_on, self.cnt + 1, torch.zeros_like(self.cnt))
        self.t_ref = self.t_ref + dt
        trig = (self.cnt >= P.n_on) & (self.t_ref >= P.refr) & walking & ((self.state == QUIET) | (self.state == RECOVER))
        self.n_trig = self.n_trig + trig.float()
        d = torch.stack([self.devs[:, 0], P.roll_sign * self.devs[:, 1]], -1)
        self.dirv = torch.where(trig.unsqueeze(1), d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6), self.dirv)
        self.t_ref = torch.where(trig, torch.zeros_like(self.t_ref), self.t_ref)
        self.t_act = torch.where(trig, torch.zeros_like(self.t_act), self.t_act + dt)
        st = torch.where(trig, torch.full_like(self.state, ACTIVE), self.state)
        st = torch.where((st == ACTIVE) & (self.t_act >= P.off_T), torch.full_like(st, RECOVER), st)
        calm = ((self.cyc[:, :2] - self.base[:, :2]).abs() <= P.q_ang).all(-1) & (self.cyc[:, 2] <= P.q_wk * self.base[:, 2] + P.q_wa)
        self.t_q = torch.where((st == RECOVER) & calm, self.t_q + dt, torch.zeros_like(self.t_q))
        st = torch.where((st == RECOVER) & (self.t_q >= P.q_hold), torch.full_like(st, QUIET), st)
        self.state = st

    def offset(self, cmd_xy):
        """(b) ACTIVE env 의 명령 (N,2) 에 오프셋 더하고 학습 범위로 클립. use_offset 꺼짐이면 입력 그대로(같은 텐서)."""
        P = self.P
        if not P.use_offset:
            return cmd_xy
        act = (self.state == ACTIVE).unsqueeze(1)
        c = cmd_xy + P.A * self.dirv
        c = torch.stack([c[:, 0].clamp(P.vx_lo, P.vx_hi), c[:, 1].clamp(-P.vy_max, P.vy_max)], -1)
        return torch.where(act, c, cmd_xy)

    def stop_ok(self, req):
        """(a) 정지 요청(bool N) → 실제 정지(래치) 허용. 꺼짐이면 req 그대로. 한 번 래치되면 유지."""
        P = self.P
        if not P.use_stopgate:
            return req
        self.stop_wait = torch.where(req & ~self.stop_latched, self.stop_wait + self.dt, self.stop_wait)
        ok = req & ((self.state == QUIET) | (self.stop_wait >= P.stop_tmax))
        self.stop_latched = self.stop_latched | ok
        return self.stop_latched & req
