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
    from .policy_runner_stage4 import (ACT_DIM, ACTION_SCALE, JOINT_ORDER, SLEW_RAD, PolicyRunner,  # noqa: F401
                                       PolicyRunnerStage4, _load_backend)
except ImportError:   # 단독 실행(--selftest)
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from policy_runner_stage4 import (ACT_DIM, ACTION_SCALE, JOINT_ORDER, SLEW_RAD, PolicyRunner,  # noqa: E402
                                      PolicyRunnerStage4, _load_backend)

RATE_HZ = 50.0

#: 1007 병목 2 odic 기본 설정(reports/gen_w3/STOPEST.md §8–§9 승격 후보 = 시뮬 --settle_judge odo --settle_v 0.07 --settle_ds 0.3
#: --settle_min 0.9 --settle_knee_comp 42 [+ --settle_knee_inject 42 플랜트]). k_hat = 0919 무릎 처짐 역산 k 35–45 의 중앙(가정값 —
#: 실물 k 식별 전). tau_ref "handover" = 시뮬 odic 과 같은 기준(인계 첫 틱 토크), "absolute" = 처짐 τ/k̂ 전부(엔코더 = 링크 + τ/k 모형).
ODIC_DEFAULT = dict(v_thr=0.07, ds_s=0.3, min_s=0.9, min_frac=0.5, window_s=0.9, k_hat=42.0, tau_ref="handover")
#: 1007 odicw = odic + 감지기 |ω_xy|(감지 EMA 뒤 — 현행 imu 조기 복귀와 같은 신호) < w_thr [rad/s] (시뮬 --settle_w 0.4).
#: 실물 재생에서 odic 이 양발 지지 진동(평균 속도 0) 중 조기 복귀한 3 건 차단용(STOPEST §10–§11). w_thr 0 = 끔(odic 그대로).
ODICW_DEFAULT = dict(ODIC_DEFAULT, w_thr=0.4)


def _leg_odometry_cls():
    """leg_odometry 지연 import — odic 를 켜지 않으면 import 자체가 일어나지 않는다(기본 경로 불변)."""
    try:
        from .leg_odometry import LegOdometry
    except ImportError:
        from leg_odometry import LegOdometry  # type: ignore
    return LegOdometry


class OdicSettle:
    """1007 병목 2: KE 조기 복귀 판정 odic — 다리 운동학 오도메트리 0.9 s 주기 평균 속도 + 무릎 토크 처짐 보상.

    시뮬 eval_ground_recovery.py `--settle_judge odo --settle_knee_comp k̂` 블록(odo_spd·odo_frac·odo_ds)의 배포판.
    수치 계약(시뮬과 같은 판정·같은 시각 — selftest_odic_equiv.py 가 시뮬 소스를 그대로 실행해 대조):
      enc = qpos(float32, JOINT_ORDER, URDF 관절 절대각) ; 무릎 enc −= (τ − τ_ref) / k̂ (float32) ;
      LegOdometry.step(enc, 자이로 원값, 중력 원값(필터 전), 접촉 0/1) → |v̄|·유효비를 float32 로 ; 양발 연속 시간 float32 누적.
    입력 τ = 무릎 2 개 관절 프레임 토크 [N·m] (왼, 오른) — 브리지가 tau_est_nm(모터 프레임)×direction 으로 바꾸고, 미수신·신선도 초과면 None.
    고장 처리(조용한 대체 금지): τ 가 None/비유한이면 이번 틱 '판정 불가'(ok False → 조기 복귀 막힘, calm 경로·불응기는 그대로)이고
    오도메트리에는 마지막 유효 τ 를 유지해 넣는다(창 연속성). 유효 τ 를 한 번도 못 받았으면 보상 없이 넣고 판정 불가.
    k_hat ≤ 0 → 보상 끔(= 시뮬 odo/odi 팔, τ 불필요).
    """

    def __init__(self, v_thr=0.07, ds_s=0.3, min_s=0.9, min_frac=0.5, window_s=0.9, k_hat=42.0, tau_ref="handover",
                 dt: float = 1.0 / RATE_HZ, joint_order=None, w_thr: float = 0.0):
        if tau_ref not in ("handover", "absolute"):
            raise ValueError(f"tau_ref {tau_ref!r} — handover|absolute")
        for nm, v, lo, hi in (("v_thr", v_thr, 0.01, 0.3), ("ds_s", ds_s, 0.0, 2.0), ("min_s", min_s, 0.0, 5.0), ("min_frac", min_frac, 0.0, 1.0),
                              ("window_s", window_s, 0.2, 3.0), ("k_hat", k_hat, 0.0, 1000.0), ("w_thr", w_thr, 0.0, 5.0)):
            if not (math.isfinite(float(v)) and lo <= float(v) <= hi):
                raise ValueError(f"odic {nm}={v} 범위 [{lo}, {hi}] 밖")
        self.v_thr, self.ds_s, self.min_s, self.min_frac = float(v_thr), float(ds_s), float(min_s), float(min_frac)
        self.k_hat, self.tau_ref, self.dt = float(k_hat), tau_ref, float(dt)
        self.w_thr = float(w_thr)      # 1007 odicw: > 0 이면 ok() 에 감지기 |ω_xy| < w_thr (0 = 끔 → odic 와 비트 동일)
        self.joint_order = list(JOINT_ORDER if joint_order is None else joint_order)
        self.ik = [self.joint_order.index("left_knee_joint"), self.joint_order.index("right_knee_joint")]
        self.odo = _leg_odometry_cls()(self.joint_order, n=1, dt=self.dt, window_s=float(window_s))
        self.reset()

    def reset(self) -> None:
        self.odo.reset()
        self.ds = np.float32(0.0); self.spd = np.float32(np.nan); self.frac = np.float32(0.0)
        self.tau0 = None; self.tau_last = None; self.tau_ok = self.k_hat <= 0.0
        self.n_tau_bad = 0
        self.w_block = False; self.n_w_block = 0      # odicw: 오도메트리는 정착인데 |ω_xy| 로 막힌 틱(진단)

    @property
    def needs_tau(self) -> bool:
        return self.k_hat > 0.0

    def update(self, qpos_rad, gyro_raw, grav_raw, contact2, tau_knee_nm=None) -> None:
        enc = np.asarray(qpos_rad, dtype=np.float32).reshape(1, len(self.joint_order)).copy()
        if self.needs_tau:
            tk = None
            if tau_knee_nm is not None:
                try:
                    tk = np.asarray([float(v) for v in tau_knee_nm], dtype=np.float32).reshape(2)
                except (TypeError, ValueError):
                    tk = None
                if tk is not None and not np.all(np.isfinite(tk)):
                    tk = None
            self.tau_ok = tk is not None
            if tk is None:
                self.n_tau_bad += 1
                tk = self.tau_last
            if tk is not None:
                self.tau_last = tk
                if self.tau0 is None:
                    self.tau0 = tk.copy()
                tq = (tk - self.tau0) if self.tau_ref == "handover" else tk
                enc[0, self.ik] = enc[0, self.ik] - tq / self.k_hat
        c = np.asarray(contact2, dtype=np.float32).reshape(1, 2)
        out = self.odo.step(enc, np.asarray(gyro_raw, dtype=np.float32).reshape(1, 3),
                            np.asarray(grav_raw, dtype=np.float32).reshape(1, 3), c)
        self.spd = np.float32(out["speed_bar"][0]); self.frac = np.float32(out["frac"][0])
        both = bool(c[0, 0] > 0.5) and bool(c[0, 1] > 0.5)
        self.ds = np.float32(self.ds + np.float32(self.dt)) if both else np.float32(0.0)

    def ok(self, t_on, w_det=None) -> bool:
        """시뮬 _ec2(odo) 의 판정부(모드·early_gate 제외): 켜진 뒤 ≥ min_s & |v̄| < v_thr & 유효비 ≥ min_frac & 양발 ≥ ds_s & τ 유효
        [& odicw: 감지기 |ω_xy|(float32, 감지 EMA 뒤) < w_thr — 시뮬 `_ec2 & (_w < settle_w)`]. w_thr > 0 인데 w_det 없음 → 판정 불가(False)."""
        r = bool(self.tau_ok and (np.float32(t_on) >= np.float32(self.min_s - 1e-6)) and (self.spd < np.float32(self.v_thr))
                 and (self.frac >= np.float32(self.min_frac)) and (self.ds >= np.float32(self.ds_s - 1e-6)))
        if self.w_thr > 0.0:
            self.w_block = bool(r and not (w_det is not None and np.isfinite(w_det) and np.float32(w_det) < np.float32(self.w_thr)))
            if self.w_block:
                self.n_w_block += 1
            r = r and not self.w_block
        return r

    def blackbox(self) -> dict:
        return {"spd": (None if not np.isfinite(self.spd) else round(float(self.spd), 4)), "frac": round(float(self.frac), 3),
                "ds": round(float(self.ds), 3), "tau_ok": int(self.tau_ok), "n_tau_bad": int(self.n_tau_bad),
                "k_hat": self.k_hat, "tau_ref": self.tau_ref,
                **({"w_thr": self.w_thr, "w_block": int(self.w_block), "n_w_block": int(self.n_w_block)} if self.w_thr > 0.0 else {})}


class SwitchDetector:
    """IMU 히스테리시스 감지기 — 상태: 0 = 서기 정책, 1 = 회복 정책."""

    def __init__(self, on_tilt_deg: float = 4.0, on_w: float = 0.5, off_tilt_deg: float = 2.5, off_w: float = 0.3,
                 calm_s: float = 1.0, dwell_s: float = 0.4, dt: float = 1.0 / RATE_HZ, ref_tau_s: float = 0.0, ref_cap_deg: float = 8.0, ref_init_s: float = 0.0, off_contact: str = "both",
                 early=None, refrac=None, in_filt: float = 0.0, early_gate=None, settle=None, imu_f32: bool = False, requiet=None):
        """ref_tau_s > 0: 기울기를 **기준 자세 대비 편차**로(1004 실물 로그 재생: 실물 서기 기울기 3–4° → 절대 기준이면 회복 모드 70–99 % 체류).
        기준 = 첫 update 의 (롤, 피치), 서기 모드·|ω|<0.1·양발일 때 EMA(τ) 갱신, |ref| ≤ ref_cap_deg. 0 = 절대 기울기(시뮬 SW-1 정의)."""
        self.on_tilt, self.on_w, self.off_tilt, self.off_w = float(on_tilt_deg), float(on_w), float(off_tilt_deg), float(off_w)
        self.calm_s, self.dwell_s, self.dt = float(calm_s), float(dwell_s), float(dt)
        self.ref_tau_s, self.ref_cap, self.ref_init_s = float(ref_tau_s), float(ref_cap_deg), float(ref_init_s)
        if off_contact not in ("both", "any", "none"):
            raise ValueError(off_contact)
        # 1005 ki: 상태 기반 조기 복귀 (tmin, tilt, w, hold) — 켜진 뒤 tmin s 지나고 기울기<tilt·|ω_xy|<w·양발 접촉(0/1)이 hold s 유지 → 끄기.
        # 시뮬 eval_ground_recovery --sw_early_imu 와 같은 정의(IMU·접촉 0/1 만, 참 속도·접촉 힘 미사용).
        self.early = None if early is None else tuple(float(x) for x in early)
        # 1005 kr: 조기 복귀 뒤 불응기 (s, tilt, w) — s 초 동안은 기울기>tilt 또는 |ω_xy|>w 일 때만 재진입(모드 왕복 46 % → 10–18 %). 시뮬 --sw_refrac 와 같은 정의.
        self.refrac = None if refrac is None else tuple(float(x) for x in refrac)
        # 1005 krF: 감지 입력(중력 단위벡터·자이로) EMA α (0 = 끔). 정책 입력은 불변 — 감지 전용. 시뮬 --det_filt 와 같은 정의(첫 틱 = 입력값으로 초기화).
        self.in_filt = float(in_filt)
        # 1005 KE: 조기 복귀 허용 조건 (tilt, w) — 이번 활성화의 최대 기울기 편차·|ω_xy| 가 문턱 이상이었을 때만(작은 오탐은 calm 경로로만 끔). 시뮬 --early_gate 와 같은 정의.
        self.early_gate = None if early_gate is None else tuple(float(x) for x in early_gate)
        self.off_contact = off_contact   # 1004 밤: 끄기 조건 접촉 요구 — 실물 패드 0(무응답·무부하)로 회복 모드 고착(시험 4, 검토 7회차)
        # 1007 병목 2: 조기 복귀 판정기. None(기본) = 현행 IMU 규칙(아래 update 경로 그대로 — 비트 동일).
        # dict/OdicSettle = odic: 조기 복귀 조건을 오도메트리 판정으로 바꾸고, 감지기 한 틱 전체를 시뮬 eval_ground_recovery 의
        # float32 순서로 계산한다(_update_odic). early 의 hold(4 번째 값)·early_gate·불응기·calm 경로·켜기 규칙은 그대로.
        self.settle = None
        if settle is not None:
            if self.early is None:
                raise ValueError("odic 는 조기 복귀(early) 설정이 있는 감지기에만(유지 시간 = early[3]) — 시뮬 --sw_early_imu 필수와 같음")
            if self.ref_tau_s <= 0.0:
                raise ValueError("odic 경로는 상대 기울기 감지기(ref_tau_s > 0, 시뮬 --sw_ref_tau)만 지원")
            self.settle = settle if isinstance(settle, OdicSettle) else OdicSettle(dt=self.dt, **dict(ODIC_DEFAULT, **settle))
        # 1007 parity 감사: imu_f32=True(기본 False = 종전 비트 동일) — 현행 IMU 조기 복귀 규칙을 odic 와 같은 float32 경로(_update_odic,
        # 시뮬 eval_ground_recovery 감지 블록 순서·정밀도)로 계산한다. 종전 float64 타이머는 calm 1.0 s·hold 0.2 s 를 시뮬보다 1 틱 일찍
        # 채운다(STOPEST §9 부수 측정: 82 꺼짐 중 46 건 −1 틱). odic 와 같은 전제(상대 기울기 ref_tau_s > 0, early 설정)에서만.
        # 1007 X5(**기본 None = 종전 비트 동일 · 실물 미검증**): 회복 모드 갇힘 해소 (T_s, w) — 모드 중 감지 |ω_xy| < w ∧ 양발 접촉이 T s 이어지면
        # 기준 자세를 현재 자세로 다시 잡고 모드를 끈다. 배경: 밀린 뒤 서는 자세가 기준과 off_tilt 넘게 달라지면 평온 경로가 영영 안 채워지고
        # (모드 중 기준 미갱신), 가벼운 활성화는 조기 복귀 자격(early_gate)도 없어 조용히 서 있는 채로 회복 모드에 갇힌다
        # (reports/gen_w3/TRANSITION_W3KE.md §5.6f–§5.6h). 시뮬 eval_transition_dev.py --sw_requiet 와 같은 정의·순서.
        # 세 번째 값(선택) = 접촉 요구: "both"(기본, 양발) | "off"(이 감지기의 끄기 접촉 요구 off_contact 를 그대로 — KE 는 '한 발 이상').
        # 실물에서 한쪽 패드가 0 을 내는 고장(1004 시험 4)에서는 "both" 가 발동하지 않는다(reports/gen_w3/TRANSITION_W3KE.md §5.13).
        self.requiet = None; self.requiet_contact = "both"
        if requiet is not None:
            rq = tuple(requiet)
            if len(rq) == 3:
                if rq[2] not in ("both", "off"):
                    raise ValueError(f"requiet 접촉 요구는 'both' | 'off' (받음 {rq[2]!r})")
                self.requiet_contact = str(rq[2]); rq = rq[:2]
            if len(rq) != 2 or float(rq[0]) <= 0.0 or float(rq[1]) <= 0.0:
                raise ValueError(f"requiet 는 (T_s > 0, w > 0[, 'both'|'off']) (받음 {requiet!r})")
            self.requiet = (float(rq[0]), float(rq[1]))
        self.imu_f32 = bool(imu_f32)
        if self.imu_f32:
            if self.settle is not None:
                raise ValueError("imu_f32 는 settle(odic) 와 함께 쓰지 않는다 — odic 경로는 이미 float32")
            if self.early is None or self.ref_tau_s <= 0.0:
                raise ValueError("imu_f32 는 조기 복귀(early)·상대 기울기(ref_tau_s > 0) 감지기(KE 계열)에만")
        self.reset()

    def reset(self) -> None:
        self.mode = 0
        self.t_on = 0.0
        self.calm = 0.0
        self.n_on = 0
        self.ref = None
        self.t_since = 0.0
        self.early_hold = 0.0
        self.n_early = 0
        self.t_since_early = 1e9
        self._fg = None; self._fw = None
        self.pk_tilt = 0.0; self.pk_w = 0.0
        self.rq_run = 0.0; self.n_requiet = 0
        if getattr(self, "settle", None) is not None or getattr(self, "imu_f32", False):      # odic·imu_f32 상태(시뮬 float32 텐서와 같은 정밀도)
            if self.settle is not None:
                self.settle.reset()
            self._k = 0
            self._f32 = dict(calm=np.float32(0), t_on=np.float32(0), hold=np.float32(0), pk_t=np.float32(0), pk_w=np.float32(0),
                             refr_t=np.float32(-1e9))

    @staticmethod
    def tilt_deg(gravity_unit) -> float:
        g = np.asarray(gravity_unit, dtype=np.float64).reshape(3)
        return math.degrees(math.acos(max(-1.0, min(1.0, -float(g[2])))))

    @staticmethod
    def roll_pitch_deg(gravity_unit) -> np.ndarray:
        """시뮬 eval _rp 와 같은 정의: 피치 = atan2(g_x, −g_z), 롤 = atan2(g_y, −g_z) [deg]."""
        g = np.asarray(gravity_unit, dtype=np.float64).reshape(3)
        return np.degrees(np.array([math.atan2(g[1], -g[2]), math.atan2(g[0], -g[2])]))

    def filtered(self, gravity_unit, gyro_rad_s):
        """감지 입력 EMA(in_filt > 0) — 중력은 필터 뒤 단위벡터로 정규화. cue 방향(ω_y)도 이 값을 쓴다."""
        g = np.asarray(gravity_unit, dtype=np.float64).reshape(3); w = np.asarray(gyro_rad_s, dtype=np.float64).reshape(3)
        if self.in_filt > 0.0:
            if self._fg is None:
                self._fg, self._fw = g.copy(), w.copy()
            self._fg = self._fg + self.in_filt * (g - self._fg); self._fw = self._fw + self.in_filt * (w - self._fw)
            g, w = self._fg, self._fw
        return g / max(float(np.linalg.norm(g)), 1e-6), w

    @property
    def needs_odic_inputs(self) -> bool:
        return self.settle is not None

    def update(self, gravity_unit, gyro_rad_s, contact2, qpos_rad=None, tau_knee_nm=None) -> int:
        if self.settle is not None or self.imu_f32:
            return self._update_odic(gravity_unit, gyro_rad_s, contact2, qpos_rad, tau_knee_nm)
        gravity_unit, gyro_rad_s = self.filtered(gravity_unit, gyro_rad_s)
        self.last_w = gyro_rad_s
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
        cq = {"both": bool(c[0] and c[1]), "any": bool(c[0] or c[1]), "none": True}[self.off_contact]
        quiet = (tilt < self.off_tilt) and (w < self.off_w) and cq
        # 시뮬과 같은 순서: calm·t_on 갱신 → 끄기 판정 → 켜기 판정
        self.calm = self.calm + self.dt if (self.mode == 1 and quiet) else 0.0
        self.t_on = self.t_on + self.dt if self.mode == 1 else 0.0
        off = self.mode == 1 and self.calm >= self.calm_s and self.t_on >= self.dwell_s
        if self.early is not None:
            _tm, _tt, _tw, _th = self.early
            ec = self.mode == 1 and self.t_on >= _tm and tilt < _tt and w < _tw and bool(c[0] and c[1])
            if self.early_gate is not None:
                self.pk_tilt = max(self.pk_tilt, tilt) if self.mode == 1 else 0.0
                self.pk_w = max(self.pk_w, w) if self.mode == 1 else 0.0
                ec = ec and (self.pk_tilt >= self.early_gate[0] or self.pk_w >= self.early_gate[1])
            self.early_hold = self.early_hold + self.dt if ec else 0.0
            if self.mode == 1 and self.early_hold >= _th and not off:
                off = True; self.n_early += 1; self.t_since_early = -self.dt   # 아래에서 +dt → 이번 틱 0
        self.t_since_early += self.dt
        on = self.mode == 0 and ((tilt > self.on_tilt) or (w > self.on_w))
        if on and self.refrac is not None and self.t_since_early < self.refrac[0]:
            on = (tilt > self.refrac[1]) or (w > self.refrac[2])
        if self.requiet is not None:          # 갇힘 해소: 조용한 양발 서기 T s → 기준 := 현재 자세, 모드 끔(다른 꺼짐이 없을 때만 집계)
            _rqc = cq if self.requiet_contact == "off" else bool(c[0] and c[1])
            self.rq_run = self.rq_run + self.dt if (self.mode == 1 and w < self.requiet[1] and _rqc) else 0.0
            if self.mode == 1 and self.rq_run >= self.requiet[0] - 1e-9 and not off:
                if self.ref_tau_s > 0.0:
                    self.ref = rp.copy()
                off = True; self.n_requiet += 1
        self.t_since += self.dt
        if on:
            self.n_on += 1
        self.mode = 1 if (self.mode == 1 or on) and not off else 0
        return self.mode


    def _update_odic(self, gravity_unit, gyro_rad_s, contact2, qpos_rad, tau_knee_nm) -> int:
        """odic 감지기 한 틱 — eval_ground_recovery.py 의 감지 블록(--det_from_obs --det_filt --sw_ref_tau --sw_early_imu
        --settle_judge odo [--early_gate] [--sw_refrac])을 같은 순서·float32 로 옮긴 것. 시각 t_rel = 인계(첫 update) 뒤 k/50 s.
        순서: 감지 입력 EMA → 상대 기울기(기준 초기화·갱신·±cap) → 오도메트리 → 켜기/조용함 → calm·t_on → calm 끄기 →
        odic 조기 복귀(·early_gate) → hold → 불응기 → 모드."""
        if qpos_rad is None and self.settle is not None:
            raise ValueError("odic 감지기: qpos_rad(엔코더 12) 필수")
        f32 = np.float32; S = self._f32; dt = f32(self.dt)
        t_rel = self._k / (1.0 / self.dt)            # 시뮬 (i − STEPS_PRE) / 50.0 과 같은 반올림
        g_raw = np.asarray(gravity_unit, dtype=np.float32).reshape(3); w_raw = np.asarray(gyro_rad_s, dtype=np.float32).reshape(3)
        if self.in_filt > 0.0:
            if self._fg is None or self._k == 0:
                self._fg, self._fw = g_raw.copy(), w_raw.copy()
            a = f32(self.in_filt)
            self._fg = (self._fg + a * (g_raw - self._fg)).astype(np.float32); self._fw = (self._fw + a * (w_raw - self._fw)).astype(np.float32)
            g, wv = self._fg, self._fw
        else:
            g, wv = g_raw, w_raw
        g = (g / max(f32(np.linalg.norm(g)), f32(1e-6))).astype(np.float32)
        self.last_w = wv
        c = np.asarray(contact2, dtype=np.float32).reshape(2) > 0.5
        w = f32(np.linalg.norm(wv[:2]))
        rp = np.degrees(np.array([np.arctan2(g[1], -g[2]), np.arctan2(g[0], -g[2])], dtype=np.float32)).astype(np.float32)
        if self.ref is None or self._k == 0 or t_rel < self.ref_init_s:
            self.ref = rp.copy()
        if self.mode == 0 and w < f32(0.1) and bool(c[0] and c[1]):
            self.ref = (self.ref + (rp - self.ref) * f32(self.dt / self.ref_tau_s)).astype(np.float32)
        self.ref = np.clip(self.ref, -self.ref_cap, self.ref_cap).astype(np.float32)
        tilt = f32(np.linalg.norm(rp - self.ref))
        self.last_tilt = float(tilt)      # 진단(재생 보고용) — 판정에 쓰지 않음
        if self.settle is not None:
            self.settle.update(qpos_rad, w_raw, g_raw, contact2, tau_knee_nm)      # 오도메트리 = 필터 전 원값(시뮬 _gy_raw·_pg_raw)
        on = bool((tilt > f32(self.on_tilt)) or (w > f32(self.on_w)))
        cq = {"both": bool(c[0] and c[1]), "any": bool(c[0] or c[1]), "none": True}[self.off_contact]
        quiet = bool((tilt < f32(self.off_tilt)) and (w < f32(self.off_w)) and cq)
        m = self.mode == 1
        S["calm"] = f32(S["calm"] + dt) if (m and quiet) else f32(0)
        S["t_on"] = f32(S["t_on"] + dt) if m else f32(0)
        off = bool(m and S["calm"] >= f32(self.calm_s) and S["t_on"] >= f32(self.dwell_s))
        if self.settle is not None:
            ec = bool(m and self.settle.ok(S["t_on"], w))      # odicw(w_thr > 0): w = 감지 EMA 뒤 |ω_xy| float32 — 시뮬 _w 와 같은 값
        else:                                               # imu_f32: 시뮬 --sw_early_imu 규칙(_ec2) float32
            _tm, _tt, _tw, _th = self.early
            ec = bool(m and S["t_on"] >= f32(_tm) and tilt < f32(_tt) and w < f32(_tw) and bool(c[0] and c[1]))
        if self.early_gate is not None:
            S["pk_t"] = f32(max(S["pk_t"], tilt)) if m else f32(0); S["pk_w"] = f32(max(S["pk_w"], w)) if m else f32(0)
            ec = ec and bool((S["pk_t"] >= f32(self.early_gate[0])) or (S["pk_w"] >= f32(self.early_gate[1])))
        S["hold"] = f32(S["hold"] + dt) if ec else f32(0)
        eoff = bool(m and S["hold"] >= f32(self.early[3]))
        if eoff:
            self.n_early += 1
        off = off or eoff
        if self.refrac is not None:
            if eoff:
                S["refr_t"] = f32(t_rel)
            in_ref = bool(f32(f32(t_rel) - S["refr_t"]) < f32(self.refrac[0]))
            on = on and ((not in_ref) or bool(tilt > f32(self.refrac[1])) or bool(w > f32(self.refrac[2])))
        if self.requiet is not None:          # 갇힘 해소(시뮬 --sw_requiet 와 같은 순서: 모든 꺼짐 판정 뒤, 켜기 반영 전)
            _rqc = cq if self.requiet_contact == "off" else bool(c[0] and c[1])
            self.rq_run = float(f32(f32(self.rq_run) + dt)) if (m and w < f32(self.requiet[1]) and _rqc) else 0.0
            if m and f32(self.rq_run) >= f32(self.requiet[0] - 1e-6) and not off:
                self.ref = rp.copy()
                off = True; self.n_requiet += 1
        new_on = (not m) and on
        if new_on:
            self.n_on += 1
        self.mode = 1 if ((m or new_on) and not off) else 0
        # 기존 필드(블랙박스·cue) 호환
        self.calm, self.t_on, self.early_hold = float(S["calm"]), float(S["t_on"]), float(S["hold"])
        self.pk_tilt, self.pk_w = float(S["pk_t"]), float(S["pk_w"])
        self._k += 1
        self.t_since = self._k / (1.0 / self.dt)
        return self.mode


class C5Runner:
    """PolicyRunnerStage4(서기 모델) + 회복 네트워크 + SwitchDetector.

    step() 은 PolicyRunnerStage4.step 과 같은 인자·반환(슬루 적용 목표각 [rad]).
    last_mode·last_switch(이번 틱 켜짐/꺼짐 이벤트)·last_raw_stand·last_raw_rec 를 남긴다(블랙박스 기록용).
    """

    def __init__(self, stand_model, rec_model, backend: str = "auto", detector: SwitchDetector | None = None,
                 cue_vx: float | None = None, cue_s: float = 1.0, cue_dir_imu: bool = True, **runner_kw):
        """cue_vx(1005 ki): 감지기가 켜지는 틱에 IMU ω_y > 0(앞으로 기우는 중)이면 cue_s 동안 **회복 네트워크 입력만** 명령 vx 를 cue_vx 로
        (관측 velocity_commands 이력 전 프레임; 서기 네트워크·공유 이력·외부 명령 불변). 시뮬 --sw_rec_vx/--sw_rec_vx_s/--cue_dir_imu 와 같은 정의."""
        self.core = PolicyRunnerStage4(stand_model, backend=backend, **runner_kw)
        self.cue_vx = None if cue_vx is None else float(cue_vx)
        self.cue_s = float(cue_s); self.cue_dir_imu = bool(cue_dir_imu)
        self.cue_active = False; self.cue_t = 0.0; self.n_cue = 0
        self._vc_slices = None
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
        bb = self._blackbox_base(d)
        if getattr(d, "settle", None) is not None:      # 1007 odic 진단(odic 감지기에서만 키 추가 — imu 경로 스키마 불변)
            bb["odic"] = d.settle.blackbox()
        if getattr(d, "requiet", None) is not None:     # 1007 갇힘 해소(옵션 켠 감지기에서만 키 추가 — 기본 스키마 불변)
            bb["requiet"] = {"n": int(d.n_requiet), "run": round(float(d.rq_run), 2)}
        return bb

    def _blackbox_base(self, d) -> dict:
        return {"mode": int(self.last_mode), "switch": self.last_switch, "n_on": int(d.n_on),
                "cue": int(self.cue_active), "n_cue": int(self.n_cue), "n_early": int(getattr(d, "n_early", 0)),
                "ref_deg": (None if d.ref is None else [round(float(v), 3) for v in d.ref]),
                "raw_stand": (None if self.last_raw_stand is None else [round(float(v), 4) for v in self.last_raw_stand]),
                "raw_rec": (None if self.last_raw_rec is None else [round(float(v), 4) for v in self.last_raw_rec])}

    def reset(self) -> None:
        self.core.reset()
        self.det.reset()
        self.last_mode = 0
        self.last_switch = ""
        self.cue_active = False; self.cue_t = 0.0

    def cue_update(self, mode: int, prev: int, gyro_rad_s) -> bool:
        """cue 상태기(로그 재생 대조에서도 단독 호출). 켜짐 틱에 ω_y>0 이면 시작, cue_s 지나면 종료, 서기 모드면 해제."""
        if mode == 1 and prev == 0:          # 이번 틱 켜짐 → 방향 조건 만족 시 cue 시작(활성화당 1회)
            _wd = getattr(self.det, "last_w", None)          # 1005 krF: 감지기가 쓴(필터된) 자이로 — 시뮬 _gy_det 과 같다
            _wd = _wd if _wd is not None else np.asarray(gyro_rad_s, dtype=np.float64).reshape(3)
            fwd = (float(_wd[1]) > 0.0) if self.cue_dir_imu else True
            self.cue_active = bool(fwd); self.cue_t = 0.0
            if fwd:
                self.n_cue += 1
        elif mode == 1 and self.cue_active:
            self.cue_t += self.det.dt
            if self.cue_t >= self.cue_s - 1e-9:
                self.cue_active = False
        if mode == 0:
            self.cue_active = False
        return self.cue_active

    def _vx_slices(self):
        """관측에서 velocity_commands 각 프레임의 vx 위치(이력 L 프레임, 프레임당 3) — 코어 항 목록에서 계산."""
        if self._vc_slices is None:
            off = 0; out = None
            for name, dim in self.core.terms:
                L = self.core._hist[name]._buf.shape[0]
                if name == "velocity_commands":
                    out = [off + k * dim for k in range(L)]; break
                off += dim * L
            if out is None:
                raise RuntimeError("관측에 velocity_commands 항 없음 — cue 사용 불가")
            self._vc_slices = out
        return self._vc_slices

    @property
    def needs_knee_tau(self) -> bool:
        """1007 odic(무릎 보상 k̂ > 0) 감지기면 True — 브리지가 tau_knee_nm(관절 프레임 무릎 2 개)을 step 에 넘긴다."""
        st = getattr(self.det, "settle", None)
        return st is not None and st.needs_tau

    def step(self, gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s, contact2, tau_knee_nm=None) -> np.ndarray:
        """tau_knee_nm(1007 odic 전용): (왼, 오른) 무릎 관절 프레임 토크 [N·m] 또는 None(미수신·낡음). imu 감지기는 받지도 쓰지도 않는다."""
        core = self.core
        qpos = PolicyRunner._vec(qpos_rad, 12, "qpos_rad")
        obs = core.assemble_obs(gyro_rad_s, gravity_unit, cmd, qpos, qvel_rad_s, contact2)
        core.last_obs = obs
        core._tick += 1
        prev = self.last_mode
        if getattr(self.det, "settle", None) is not None:
            mode = self.det.update(gravity_unit, gyro_rad_s, contact2, qpos_rad=qpos, tau_knee_nm=tau_knee_nm)
        else:
            mode = self.det.update(gravity_unit, gyro_rad_s, contact2)
        raw_s = np.asarray(core._net(obs), dtype=np.float32).reshape(-1)
        obs_r = obs
        if self.cue_vx is not None:
            self.cue_update(mode, prev, gyro_rad_s)
            if self.cue_active:
                obs_r = obs.copy()
                for i0 in self._vx_slices():
                    obs_r[i0] = self.cue_vx
        raw_r = np.asarray(self._rec_net(obs_r), dtype=np.float32).reshape(-1)     # 매 틱 계산(지연 일정·기록용)
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


def _selftest_cue_early() -> str:
    """1005 ki: cue 는 켜짐 틱 ω_y>0 일 때만, 정확히 cue_s(50 틱) 동안 회복 입력의 vx 만 바꾼다. 조기 복귀는 tmin·hold 뒤 끈다."""
    W = np.random.default_rng(3).normal(0, 0.05, (12, 320)).astype(np.float32)
    seen = []
    stand = lambda o: np.tanh(W @ o)                                  # noqa: E731
    def rec(o):
        seen.append(o.copy()); return -np.tanh(W @ o)
    kw = dict(obs_dim=320, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
    # (1) cue 없음 설정은 종전 C5 와 비트 동일
    a = C5Runner(stand, lambda o: -np.tanh(W @ o), **kw); b = C5Runner(stand, lambda o: -np.tanh(W @ o), cue_vx=None, **kw)
    for k in range(60):
        x = _inputs(k, 6.0 if 20 <= k < 25 else 1.0)
        assert np.array_equal(a.step(**x), b.step(**x)), "cue 없음 설정이 종전과 다름"
    # (2) 앞으로 기울며 켜짐(ω_y>0) → 50 틱 cue
    c = C5Runner(stand, rec, cue_vx=0.03, cue_s=1.0, **kw)
    for k in range(10):
        c.step(**_inputs(k, 1.0))
    seen.clear(); vx = c._vx_slices()
    x = _inputs(10, 6.0, w=0.6); c.step(**x)
    assert c.cue_active and c.n_cue == 1, "ω_y>0 켜짐에서 cue 미시작"
    n_cue_ticks = 1
    for k in range(11, 120):
        c.step(**_inputs(k, 6.0))
        n_cue_ticks += int(c.cue_active)
    assert n_cue_ticks == 50, f"cue 지속 틱 {n_cue_ticks} ≠ 50"
    assert all(abs(seen[0][i] - 0.03) < 1e-7 for i in vx), "cue 중 회복 입력 vx ≠ 0.03"
    assert all(abs(c.core.last_obs[i]) < 1e-7 for i in vx), "공유 관측(서기 입력)의 vx 가 바뀜"
    # (3) 뒤로 기울며 켜짐(ω_y<0) → cue 없음
    d = C5Runner(stand, lambda o: -np.tanh(W @ o), cue_vx=0.03, **kw)
    for k in range(10):
        d.step(**_inputs(k, 1.0))
    d.step(**_inputs(10, 6.0, w=-0.6)); assert d.last_mode == 1 and not d.cue_active and d.n_cue == 0, "뒤 기울기에서 cue 시작"
    # (4) 조기 복귀: tmin 0.6 s 전에는 유지, 이후 hold 0.2 s 조용하면 끔(calm 1.0 s 보다 먼저)
    e = SwitchDetector(on_tilt_deg=3.0, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, early=(0.6, 6.0, 0.4, 0.2))
    e.update(_inputs(0, 5.0)["gravity_unit"], [0, 0, 0], [1, 1])
    m = [e.update(_inputs(0, 2.0)["gravity_unit"], [0, 0, 0], [1, 1]) for _ in range(60)]
    first_off = m.index(0) + 1
    assert 39 <= first_off <= 41 and e.n_early == 1, f"조기 복귀 시점 {first_off} 틱(기대 ~40 = tmin 0.6 s + hold 0.2 s)"
    # (5) 불응기: 조기 복귀 직후 1 s 동안 4° 기울기로는 재진입하지 않고 7° 면 재진입
    f = SwitchDetector(on_tilt_deg=3.0, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, early=(0.6, 6.0, 0.4, 0.2), refrac=(1.0, 6.0, 1.0))
    f.update(_inputs(0, 5.0)["gravity_unit"], [0, 0, 0], [1, 1])
    for _ in range(60):
        if f.update(_inputs(0, 2.0)["gravity_unit"], [0, 0, 0], [1, 1]) == 0:
            break
    assert f.mode == 0 and f.n_early == 1
    m = [f.update(_inputs(0, 4.0)["gravity_unit"], [0, 0, 0], [1, 1]) for _ in range(10)]
    assert max(m) == 0, "불응기 중 4° 로 재진입"
    assert f.update(_inputs(0, 7.0)["gravity_unit"], [0, 0, 0], [1, 1]) == 1, "불응기 중 7° 로 재진입 안 됨"
    g = SwitchDetector(on_tilt_deg=3.0, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, early=(0.6, 6.0, 0.4, 0.2), refrac=(1.0, 6.0, 1.0))
    g.update(_inputs(0, 5.0)["gravity_unit"], [0, 0, 0], [1, 1])
    for _ in range(60):
        if g.update(_inputs(0, 2.0)["gravity_unit"], [0, 0, 0], [1, 1]) == 0:
            break
    m = [g.update(_inputs(0, 2.0)["gravity_unit"], [0, 0, 0], [1, 1]) for _ in range(55)]
    assert g.update(_inputs(0, 4.0)["gravity_unit"], [0, 0, 0], [1, 1]) == 1, "불응기(1 s) 뒤 4° 로 재진입 안 됨"
    return "cue·early·refrac OK"


def _selftest_odic() -> str:
    """1007 odic 감지기 단위 동작(시뮬 소스 대조는 selftest_odic_equiv.py): 설정 검증 · 정지 자세면 켜진 뒤 0.9 s + 유지 0.2 s(float32 11 틱)에
    조기 복귀 · τ 미수신이면 조기 복귀 막힘(모드 유지) · 다리가 움직이면 막힘 · 기본(settle 없음)은 qpos/τ 를 받지 않는다."""
    ke = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5, off_contact="any",
              early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33, early_gate=(5.0, 1.0))
    for bad in (dict(ke, early=None), dict(ke, ref_tau_s=0.0)):
        try:
            SwitchDetector(**bad, settle={}); raise AssertionError("odic 설정 검증 누락")
        except ValueError:
            pass
    try:
        OdicSettle(k_hat=-1.0); raise AssertionError("k_hat 범위 검증 누락")
    except ValueError:
        pass
    from policy_runner import DEFAULT_POSE_RAD as _DP  # noqa: E402
    q0 = np.asarray(_DP, np.float64)

    def run(tau_fn, q_fn, n=200):
        d = SwitchDetector(**ke, settle={})
        g = lambda deg: np.array([math.sin(math.radians(deg)), 0.0, -math.cos(math.radians(deg))])   # noqa: E731
        for k in range(100):
            d.update(g(0.0), [0, 0, 0], [1, 1], qpos_rad=q_fn(k), tau_knee_nm=tau_fn(k))
        assert d.mode == 0
        # 6° 6 틱(감지 입력 EMA α0.33 → 필터 기울기 최대 ≥ 5° = early_gate 충족) → 2°(켜기 3° 미만·조용함 1.5° 초과 = calm 경로 없음)
        ms = [d.update(g(6.0 if j < 6 else 2.0), [0, 0, 0], [1, 1], qpos_rad=q_fn(100 + j), tau_knee_nm=tau_fn(100 + j)) for j in range(n)]
        k_on = ms.index(1)
        return d, (ms.index(0, k_on) - k_on if 0 in ms[k_on:] else None)      # 켜진 틱 = 0

    d, k_off = run(lambda k: (3.0, -3.0), lambda k: q0)
    assert d.n_early == 1 and k_off == 55, f"정지 자세 조기 복귀 틱 {k_off}(기대 55 = 켜진 뒤 t_on float32 ≥ 0.9−1e-6 인 45 번째 갱신부터 유지 0.2 s float32 11 틱)"
    d, k_off = run(lambda k: None, lambda k: q0)
    assert k_off is None and d.n_early == 0 and d.settle.n_tau_bad == 300, "τ 미수신인데 조기 복귀"
    d, k_off = run(lambda k: (3.0, -3.0) if k < 130 else None, lambda k: q0)
    assert k_off is None and d.n_early == 0, "τ 끊긴 뒤 조기 복귀"
    iL = JOINT_ORDER.index("left_hip_f_joint"); iR = JOINT_ORDER.index("right_hip_f_joint")

    def q_mv(k):      # 골반이 앞으로 0.15 m/s 쯤 움직이는 양발 지지(두 힙 피치 램프)
        q = q0.copy(); q[iL] += 0.006 * k * 0.5; q[iR] -= 0.006 * k * 0.5; return q
    d, k_off = run(lambda k: (3.0, -3.0), q_mv)
    assert k_off is None and d.n_early == 0 and float(d.settle.spd) > 0.07, f"움직이는 중 조기 복귀(|v̄| {float(d.settle.spd):.3f})"
    d0 = SwitchDetector(**ke)
    assert d0.settle is None and not d0.needs_odic_inputs
    return "odic 감지기 OK (정지 55 틱 조기 복귀·τ 미수신/끊김 막힘·움직임 막힘·기본 경로 settle 없음)"


def _selftest_requiet() -> str:
    """갇힘 해소(1007): 기본(None)은 종전과 틱 단위 동일 · 기준과 2.2° 다른 자세로 조용히 서면 종전은 갇히고 옵션은 2.0 s 뒤 재기준·꺼짐 ·
    흔들리는 동안(|ω| ≥ w)·한발 접촉에서는 발동하지 않음 · float32 경로도 같은 틱(±1)."""
    kw = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5, off_contact="any",
              early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33, early_gate=(5.0, 1.0))

    def g(p_deg, r_deg=0.0):
        p, r = math.radians(p_deg), math.radians(r_deg)
        v = np.array([math.sin(p) * math.cos(r), math.sin(r), -math.cos(p) * math.cos(r)]); return v / np.linalg.norm(v)

    def run(det, seq):
        out = []
        for (p, w, c) in seq:
            out.append(det.update(g(p), np.array([0.0, w, 0.0]), np.array(c, dtype=np.float32)))
        return out
    both, one = (1.0, 1.0), (1.0, 0.0)
    # 서기 2 s(기준 0°) → 가벼운 밀림(|ω| 0.6, 0.3 s: 최대 편차 < 5°·|ω| < 1 → 조기 복귀 자격 없음) → 2.2° 어긋난 자세로 조용히 6 s
    seq = [(0.0, 0.0, both)] * 100 + [(1.5, 0.6, both)] * 15 + [(2.2, 0.02, both)] * 300
    a = run(SwitchDetector(**kw), seq); b = run(SwitchDetector(requiet=None, **kw), seq)
    assert a == b, "requiet None 이 종전과 다름"
    assert a[-1] == 1 and sum(a[115:]) == 300, "종전 감지기는 이 자세에서 갇혀야 함(전제 깨짐)"
    d = SwitchDetector(requiet=(2.0, 0.3), **kw); c_ = run(d, seq)
    k_off = next(i for i in range(115, len(c_)) if c_[i] == 0)
    assert c_[:k_off] == a[:k_off] and d.n_requiet == 1 and c_[-1] == 0 and sum(c_[k_off:]) == 0, "재기준 뒤 꺼짐·재켜짐 없음"
    t_off = (k_off - 115) * 0.02
    assert 1.9 <= t_off <= 2.7, f"조용해진 뒤 약 2 s 에 꺼져야 함: {t_off:.2f}s"       # 감지 입력 EMA(0.33)로 |ω| 가 문턱 아래로 내려오는 틱 포함
    assert abs(float(d.ref[1]) - 2.2) < 0.15, f"기준이 현재 자세로 재설정돼야 함: {d.ref}"
    # 흔들림 유지(|ω| 0.35 ≥ w)·한발 접촉에서는 발동 없음
    for bad in ([(2.2, 0.35, both)] * 300, [(2.2, 0.02, one)] * 300):
        d2 = SwitchDetector(requiet=(2.0, 0.3), **kw); run(d2, [(0.0, 0.0, both)] * 100 + [(1.5, 0.6, both)] * 15 + bad)
        assert d2.n_requiet == 0 and d2.mode == 1, "조용한 양발 서기가 아닌데 재기준 발동"
    # 접촉 요구 "off"(= off_contact any): 한쪽 접촉 비트가 0 인 채 조용히 서도(패드 무응답 모사) 발동 — "both" 는 위에서 본 대로 미발동
    d4 = SwitchDetector(requiet=(2.0, 0.3, "off"), **kw); f_ = run(d4, [(0.0, 0.0, both)] * 100 + [(1.5, 0.6, both)] * 15 + [(2.2, 0.02, one)] * 300)
    assert d4.n_requiet == 1 and f_[-1] == 0, "접촉 요구 off(any)인데 한발 비트에서 미발동"
    d5 = SwitchDetector(requiet=(2.0, 0.3, "off"), **kw); assert run(d5, seq) == c_, "양발 접촉 입력에서는 both 와 off 가 같아야 함"
    try:
        SwitchDetector(requiet=(2.0, 0.3, "any"), **kw); raise AssertionError("잘못된 접촉 요구를 받아들임")
    except ValueError:
        pass
    # float32 경로(imu_f32) — 같은 규칙, 꺼짐 틱 ±1
    d3 = SwitchDetector(requiet=(2.0, 0.3), imu_f32=True, **kw); e_ = run(d3, seq)
    k3 = next(i for i in range(115, len(e_)) if e_[i] == 0)
    assert abs(k3 - k_off) <= 1 and d3.n_requiet == 1 and e_[-1] == 0, f"float32 경로 꺼짐 틱 {k3} vs {k_off}"
    try:
        SwitchDetector(requiet=(0.0, 0.3), **kw); raise AssertionError("잘못된 requiet 를 받아들임")
    except ValueError:
        pass
    return f"갇힘 해소 OK (기본 None = 종전 동일 · 종전은 갇힘 · 옵션은 조용해진 뒤 {t_off:.2f} s 에 재기준·꺼짐 · 흔들림/한발에서는 미발동 · 접촉 요구 off 는 한발 비트에서도 발동 · float32 경로 ±1 틱)"


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        for f in (_selftest_detector, _selftest_equivalence, _selftest_continuity, _selftest_cue_early, _selftest_odic, _selftest_requiet):
            print(f())
        print("C5 selftest PASS")
