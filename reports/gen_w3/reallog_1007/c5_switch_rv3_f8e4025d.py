"""C5 (1004 사용자 승인 구조, 실물 미배포): 서기 정책(G21a) + 회복 정책(G24R 계열)을 IMU 감지기로 전환하는 러너 래퍼.

시뮬 평가 `eval_ground_recovery.py --switch_to` 와 같은 규칙을 배포 러너 위에 옮긴 것이다.
- 관측 조립(이력·EMA·직전 액션)은 PolicyRunnerStage4 하나가 공유 — 전환 순간 관측 불연속 없음
  (1004 handoff 분석: 시뮬에서 전환 틱 |Δobs| = 평소 틱).
- 매 틱 감지기가 어느 네트워크의 출력을 '실행'할지 고른다. 관측의 직전 액션 = 실제로 실행된 액션(어느 정책이든).
- 감지기 입력은 정책 관측과 같은 신호(IMU 중력 단위벡터·자이로·발 접촉 0/1) — 정책 관측 계약 불변.

감지기(1004 사용자 지정 초기값):
  켜기  tilt > on_tilt_deg(4.0) 또는 |ω_xy| > on_w(0.5 rad/s)
  끄기  tilt < off_tilt_deg(2.5) & |ω_xy| < off_w(0.3) & 양발 접촉 이 calm_s(1.0) 연속 & 켜진 뒤 ≥ dwell_s(0.4)

이 파일은 로컬 모듈이다. 젯슨 브리지·서비스 연결은 별도 승인 사항(배포 체인 변경).
자가검증: python3 rl_walking/deploy/c5_switch.py --selftest (CPU, GPU 불필요)
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

try:      # 패키지 경로(브리지: rl_walking.deploy.c5_switch) — 러너 모듈을 한 번만 로드
    from .policy_runner_stage4 import (ACT_DIM, ACTION_SCALE, SLEW_RAD, PolicyRunner,  # noqa: F401
                                       PolicyRunnerStage4, _load_backend)
except ImportError:   # 단독 실행(--selftest)
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from policy_runner_stage4 import (ACT_DIM, ACTION_SCALE, SLEW_RAD, PolicyRunner,  # noqa: E402
                                      PolicyRunnerStage4, _load_backend)

RATE_HZ = 50.0


class SwitchDetector:
    """IMU 히스테리시스 감지기 — 상태: 0 = 서기 정책, 1 = 회복 정책."""

    def __init__(self, on_tilt_deg: float = 4.0, on_w: float = 0.5, off_tilt_deg: float = 2.5, off_w: float = 0.3,
                 calm_s: float = 1.0, dwell_s: float = 0.4, dt: float = 1.0 / RATE_HZ, ref_tau_s: float = 0.0, ref_cap_deg: float = 8.0, ref_init_s: float = 0.0):
        """ref_tau_s > 0: 기울기를 **기준 자세 대비 편차**로(1004 실물 로그 재생: 실물 서기 기울기 3–4° → 절대 기준이면 회복 모드 70–99 % 체류).
        기준 = 첫 update 의 (롤, 피치), 서기 모드·|ω|<0.1·양발일 때 EMA(τ) 갱신, |ref| ≤ ref_cap_deg. 0 = 절대 기울기(시뮬 SW-1 정의)."""
        self.on_tilt, self.on_w, self.off_tilt, self.off_w = float(on_tilt_deg), float(on_w), float(off_tilt_deg), float(off_w)
        self.calm_s, self.dwell_s, self.dt = float(calm_s), float(dwell_s), float(dt)
        self.ref_tau_s, self.ref_cap, self.ref_init_s = float(ref_tau_s), float(ref_cap_deg), float(ref_init_s)
        self.reset()

    def reset(self) -> None:
        self.mode = 0
        self.t_on = 0.0
        self.calm = 0.0
        self.n_on = 0
        self.ref = None
        self.t_since = 0.0

    @staticmethod
    def tilt_deg(gravity_unit) -> float:
        g = np.asarray(gravity_unit, dtype=np.float64).reshape(3)
        return math.degrees(math.acos(max(-1.0, min(1.0, -float(g[2])))))

    @staticmethod
    def roll_pitch_deg(gravity_unit) -> np.ndarray:
        """시뮬 eval _rp 와 같은 정의: 피치 = atan2(g_x, −g_z), 롤 = atan2(g_y, −g_z) [deg]."""
        g = np.asarray(gravity_unit, dtype=np.float64).reshape(3)
        return np.degrees(np.array([math.atan2(g[1], -g[2]), math.atan2(g[0], -g[2])]))

    def update(self, gravity_unit, gyro_rad_s, contact2) -> int:
        tilt = self.tilt_deg(gravity_unit)
        w = float(np.linalg.norm(np.asarray(gyro_rad_s, dtype=np.float64).reshape(3)[:2]))
        c = np.asarray(contact2, dtype=np.float64).reshape(2) > 0.5
        if self.ref_tau_s > 0.0:
            rp = self.roll_pitch_deg(gravity_unit)
            if self.ref is None or self.t_since < self.ref_init_s:     # 인계 블렌딩 과도 동안 기준 = 현재 자세(편차 0)
                self.ref = rp.copy()
            if self.mode == 0 and w < 0.1 and bool(c[0] and c[1]):
                self.ref = np.clip(self.ref + (rp - self.ref) * (self.dt / self.ref_tau_s), -self.ref_cap, self.ref_cap)
            tilt = float(np.linalg.norm(rp - self.ref))
        quiet = (tilt < self.off_tilt) and (w < self.off_w) and bool(c[0] and c[1])
        # 시뮬과 같은 순서: calm·t_on 갱신 → 끄기 판정 → 켜기 판정
        self.calm = self.calm + self.dt if (self.mode == 1 and quiet) else 0.0
        self.t_on = self.t_on + self.dt if self.mode == 1 else 0.0
        off = self.mode == 1 and self.calm >= self.calm_s and self.t_on >= self.dwell_s
        on = self.mode == 0 and ((tilt > self.on_tilt) or (w > self.on_w))
        self.t_since += self.dt
        if on:
            self.n_on += 1
        self.mode = 1 if (self.mode == 1 or on) and not off else 0
        return self.mode


class C5Runner:
    """PolicyRunnerStage4(서기 모델) + 회복 네트워크 + SwitchDetector.

    step() 은 PolicyRunnerStage4.step 과 같은 인자·반환(슬루 적용 목표각 [rad]).
    last_mode·last_switch(이번 틱 켜짐/꺼짐 이벤트)·last_raw_stand·last_raw_rec 를 남긴다(블랙박스 기록용).
    """

    def __init__(self, stand_model, rec_model, backend: str = "auto", detector: SwitchDetector | None = None, **runner_kw):
        self.core = PolicyRunnerStage4(stand_model, backend=backend, **runner_kw)
        rb, self._rec_net = _load_backend(rec_model, backend)
        self.rec_backend = rb
        self.det = detector or SwitchDetector()
        self.last_mode = 0
        self.last_switch = ""
        self.last_raw_stand = None
        self.last_raw_rec = None

    def __getattr__(self, name):
        """브리지가 읽는 러너 속성(last_obs·last_action·act_obs_ema_beta·jvel_ema_mode·last_jvel_ema·_ema …)은 공유 코어에 위임."""
        core = self.__dict__.get("core")
        if core is None:
            raise AttributeError(name)
        return getattr(core, name)

    def blackbox(self) -> dict:
        """블랙박스 C5 필드(기록 전용)."""
        d = self.det
        return {"mode": int(self.last_mode), "switch": self.last_switch, "n_on": int(d.n_on),
                "ref_deg": (None if d.ref is None else [round(float(v), 3) for v in d.ref]),
                "raw_stand": (None if self.last_raw_stand is None else [round(float(v), 4) for v in self.last_raw_stand]),
                "raw_rec": (None if self.last_raw_rec is None else [round(float(v), 4) for v in self.last_raw_rec])}

    def reset(self) -> None:
        self.core.reset()
        self.det.reset()
        self.last_mode = 0
        self.last_switch = ""

    def step(self, gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s, contact2) -> np.ndarray:
        core = self.core
        qpos = PolicyRunner._vec(qpos_rad, 12, "qpos_rad")
        obs = core.assemble_obs(gyro_rad_s, gravity_unit, cmd, qpos, qvel_rad_s, contact2)
        core.last_obs = obs
        core._tick += 1
        prev = self.last_mode
        mode = self.det.update(gravity_unit, gyro_rad_s, contact2)
        raw_s = np.asarray(core._net(obs), dtype=np.float32).reshape(-1)
        raw_r = np.asarray(self._rec_net(obs), dtype=np.float32).reshape(-1)     # 매 틱 계산(지연 일정·기록용)
        for r in (raw_s, raw_r):
            if r.shape != (ACT_DIM,):
                raise RuntimeError(f"정책 출력 shape 이상: {r.shape}")
        raw = raw_r if mode == 1 else raw_s
        self.last_raw_stand, self.last_raw_rec = raw_s, raw_r
        self.last_switch = "on" if (mode == 1 and prev == 0) else ("off" if (mode == 0 and prev == 1) else "")
        self.last_mode = mode
        core.last_action = raw.copy()                 # 관측의 직전 액션 = 실제 실행된 액션
        target = core.default_pose + ACTION_SCALE * raw
        if core._prev_emitted is None:
            core._prev_emitted = qpos.copy()
        emitted = core._prev_emitted + np.clip(target - core._prev_emitted, -SLEW_RAD, SLEW_RAD)
        core._prev_emitted = emitted
        return emitted.copy()


# ---------------------------------------------------------------------------
# 자가검증
# ---------------------------------------------------------------------------
def _inputs(k: int, tilt_deg: float = 0.0, w: float = 0.0, both: bool = True):
    rng = np.random.default_rng(k)
    th = math.radians(tilt_deg)
    grav = np.array([math.sin(th), 0.0, -math.cos(th)], dtype=np.float32)
    gyro = np.array([0.0, w, 0.0], dtype=np.float32)
    return dict(gyro_rad_s=gyro, gravity_unit=grav, cmd=np.zeros(3, np.float32),
                qpos_rad=rng.normal(0, 0.02, 12).astype(np.float32), qvel_rad_s=rng.normal(0, 0.1, 12).astype(np.float32),
                contact2=np.array([1.0, 1.0 if both else 0.0], np.float32))


def _selftest_detector() -> str:
    d = SwitchDetector()
    seq = [d.update(_inputs(0, 1.0)["gravity_unit"], [0, 0, 0], [1, 1]) for _ in range(10)]
    assert max(seq) == 0, "조용한 서기에서 켜짐"
    assert d.update(_inputs(0, 4.5)["gravity_unit"], [0, 0, 0], [1, 1]) == 1, "tilt 4.5° 에서 안 켜짐"
    # 바로 조용해져도 calm 1.0 s(50 틱) 전에는 유지
    m = [d.update(_inputs(0, 1.0)["gravity_unit"], [0, 0, 0], [1, 1]) for _ in range(49)]
    assert min(m) == 1, "calm 1.0 s 전에 꺼짐"
    assert d.update(_inputs(0, 1.0)["gravity_unit"], [0, 0, 0], [1, 1]) == 0, "calm 1.0 s 뒤 안 꺼짐"
    # 한 발 접촉이면 calm 누적 안 됨
    d.reset(); d.update(_inputs(0, 5.0)["gravity_unit"], [0, 0, 0], [1, 1])
    m = [d.update(_inputs(0, 1.0)["gravity_unit"], [0, 0, 0], [1, 0]) for _ in range(80)]
    assert min(m) == 1, "한 발 접촉 중 꺼짐"
    # ω 로 켜짐
    d.reset(); assert d.update(_inputs(0, 0.0)["gravity_unit"], [0.0, 0.6, 0.0], [1, 1]) == 1, "ω 0.6 에서 안 켜짐"
    # dwell: calm 을 0.2 s 로 줄이면 dwell 0.4 s 가 지배
    d2 = SwitchDetector(calm_s=0.2); d2.update(_inputs(0, 5.0)["gravity_unit"], [0, 0, 0], [1, 1])
    m = [d2.update(_inputs(0, 1.0)["gravity_unit"], [0, 0, 0], [1, 1]) for _ in range(30)]
    first_off = m.index(0) + 1
    assert 19 <= first_off <= 21, f"dwell 0.4 s 아님 (꺼진 틱 {first_off})"
    # 상대 기울기: 정적 3.5° 기울기에서 시작하면 켜지지 않고, 그 위에 +5° 편차면 켜짐
    d3 = SwitchDetector(ref_tau_s=10.0)
    m = [d3.update(_inputs(0, 3.5)["gravity_unit"], [0, 0, 0], [1, 1]) for _ in range(100)]
    assert max(m) == 0, "상대 감지기가 정적 3.5° 에서 켜짐"
    assert d3.update(_inputs(0, 8.6)["gravity_unit"], [0, 0, 0], [1, 1]) == 1, "상대 편차 5.1° 에서 안 켜짐"
    m = [d3.update(_inputs(0, 3.5)["gravity_unit"], [0, 0, 0], [1, 1]) for _ in range(60)]
    assert m[-1] == 0, "상대 감지기가 기준 자세 복귀 뒤 안 꺼짐"
    da = SwitchDetector(); da.update(_inputs(0, 4.5)["gravity_unit"], [0, 0, 0], [1, 1])
    m = [da.update(_inputs(0, 3.5)["gravity_unit"], [0, 0, 0], [1, 1]) for _ in range(200)]
    assert min(m) == 1, "절대 감지기 확인: 정적 3.5° 에서 꺼지면 안 됨(실물 로그 재생 현상 재현)"
    return "detector OK (절대·상대)"


def _selftest_equivalence() -> str:
    """회복이 한 번도 켜지지 않으면 C5Runner == PolicyRunnerStage4(서기 모델) 비트 동일."""
    W = np.random.default_rng(1).normal(0, 0.05, (12, 320)).astype(np.float32)
    stand = lambda o: np.tanh(W @ o)          # noqa: E731
    rec = lambda o: -np.tanh(W @ o)           # noqa: E731
    kw = dict(obs_dim=320, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
    a = PolicyRunnerStage4(stand, **kw); b = C5Runner(stand, rec, **kw)
    for k in range(120):
        x = _inputs(k, 1.0)
        ya = a.step(**x); yb = b.step(**x)
        assert np.array_equal(ya, yb), f"틱 {k} 출력 불일치"
        assert np.array_equal(a.last_obs, b.core.last_obs), f"틱 {k} 관측 불일치"
    assert b.det.n_on == 0
    return "equivalence(no switch) OK"


def _selftest_continuity() -> str:
    """전환 뒤 관측의 직전 액션 = 직전 틱에 실제 실행된 원시 액션(회복 정책 출력)."""
    W = np.random.default_rng(2).normal(0, 0.05, (12, 320)).astype(np.float32)
    stand = lambda o: np.tanh(W @ o)          # noqa: E731
    rec = lambda o: 0.5 - np.tanh(W @ o)      # noqa: E731
    b = C5Runner(stand, rec, obs_dim=320)     # 직전 액션 EMA 없음 → 관측 actions 최신 프레임 = 직전 raw 그대로
    for k in range(20):
        b.step(**_inputs(k, 1.0))
    b.step(**_inputs(20, 6.0)); assert b.last_mode == 1 and b.last_switch == "on"
    executed = b.core.last_action.copy(); assert np.array_equal(executed, b.last_raw_rec)
    b.step(**_inputs(21, 6.0))
    off = 0; sl = None
    for name, dim in b.core.terms:
        L = b.core._hist[name]._buf.shape[0]
        if name == "actions":
            sl = (off, off + dim * L, dim); break
        off += dim * L
    assert sl is not None, "관측에 actions 항 없음"
    a0, a1, dim = sl
    newest = b.core.last_obs[a1 - dim:a1]
    assert np.allclose(newest, executed, atol=1e-6), "관측 직전 액션 ≠ 실행된 회복 액션"
    return "continuity OK"


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        for f in (_selftest_detector, _selftest_equivalence, _selftest_continuity):
            print(f())
        print("C5 selftest PASS")
