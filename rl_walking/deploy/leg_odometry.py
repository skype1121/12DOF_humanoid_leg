"""다리 운동학 오도메트리 + 회복 종료(정착) 판정기 — 순수 numpy, 배포 러너용(기본 미연결).

배경(reports/night_1005/LOG.md C2CAL, reports/gen_w3/C2_DESIGN.md §7): IMU 창 문턱(|ω|·기울기)은 제자리 걸음 흔들림에 묻혀
회복 중 '아직 움직임'을 9–35 % 만 잡는다. 분리되는 신호는 한 보행 주기 평균 속도뿐이다. 실물에는 속도 추정기가 없으므로
배포 가능한 입력(관절 엔코더·IMU 자이로/중력·발 접촉 0/1)만으로 골반 수평 속도를 추정하고 한 주기 평균을 낸다.

추정식(지지발 고정 가정):  v_base^w = −R_wb (ω_b × p_b + ṗ_b),  p_b = FK(엔코더) = 몸통 프레임의 지지발 바닥점.
  - R_wb = Rz(ψ̂)·R_rp(중력 관측) — ψ̂ 는 자이로 적분(드리프트는 1 주기 평균에 무의미). 회전항은 자이로를 직접 쓴다(중력 미분 안 함).
  - ṗ_b = 엔코더 FK 의 유한차분. 지지 판정 = 접촉 비트가 이번·이전 틱 모두 1 인 다리(착지 첫 틱의 스윙 속도 혼입 방지).
  - 두 다리 지지 = 평균, 지지 없음 = 무효(창 평균에서 제외).
  - 발목 유격·무릎 처짐처럼 엔코더가 못 보는 링크 이동은 그대로 오차가 된다(오프라인 표에서 측정).

입력 계약(틱 50 Hz): enc[12](rad, 로봇 절대 관절각 — 정책 상대각 + DEFAULT, 이름 순서는 joint_names 로 지정),
gyro_b[3](rad/s, 몸통), grav_b[3](projected gravity, 단위 무관 — 정규화), contact[2](왼·오 0/1).
모든 메서드는 배치 (N, …) 를 받는다(시뮬 오프라인 평가 = N env, 배포 = N 1).

권장 규칙(1006 오프라인 측정, reports/gen_w3/STOPEST.md — 시뮬 G11a·배포 신호 계약 C1, 실물 미검증):
  - 서기형 회복(KE: 회복 끝나면 서기 정책): settled = |v̄| < 0.07 m/s(창 0.9 s, 유효비 ≥ 0.5) & 양발 접촉 연속 ≥ 0.3 s,
    회복 켜진 뒤 ≥ 0.9 s, 0.2 s 유지 → 거짓정착 0.6–1.3 %·미선언 0·지연 중앙 +0.14–0.36 s (KE 현행 IMU 규칙 3.3–11.9 %).
  - 걸으며 정착(W3/c2v): 양발 조건 불가(제자리 걸음) → SettleJudge 만(θ 0.07). 단, 엔코더가 못 보는 무릎 하중 처짐이
    동적이면(가설, 실물 무릎 τ 변동 5–8 N·m) 미선언 75–90 % → 무릎 토크 기반 보상 없이는 W3 정지 판정에 쓰지 말 것.
  - z_gate(발 높이 게이트)는 효과 없음(측정) — 기본 None.
"""
from __future__ import annotations

import numpy as np

# ── biped12_final3.urdf 다리 사슬(부모 → 자식 원점 xyz, 회전축). base→pelvis 고정 rpy(π/2, 0, π/2). 다른 관절 rpy 전부 0. ──
# selftest: rl_walking/tests_leg_odometry.py 가 URDF 재파싱 값과 비교한다.
_BASE_TO_PELVIS_RPY = (np.pi / 2, 0.0, np.pi / 2)
LEG_CHAIN = {
    "left": [
        ("left_hip_f_joint", (0.071, 0.0, 0.0), (-1, 0, 0)),
        ("left_hip_a_joint", (0.0725, 0.0, 0.019625), (0, 0, -1)),
        ("left_hip_r_joint", (0.0, -0.12405, -0.02), (0, 1, 0)),
        ("left_knee_joint", (-0.01985, -0.15827, 0.00035), (-1, 0, 0)),
        ("left_ankle_f_joint", (0.0, -0.236, 0.0), (-1, 0, 0)),
        ("left_ankle_r_joint", (0.02, -0.08245, -0.0105), (0, 0, 1)),
    ],
    "right": [
        ("right_hip_f_joint", (-0.071, 0.0, 0.0), (1, 0, 0)),
        ("right_hip_a_joint", (-0.0725, 0.0, 0.019625), (0, 0, -1)),
        ("right_hip_r_joint", (0.0, -0.12405, -0.02), (0, 1, 0)),
        ("right_knee_joint", (0.01985, -0.158275, -0.00035), (1, 0, 0)),
        ("right_ankle_f_joint", (0.0, -0.236, 0.0), (1, 0, 0)),
        ("right_ankle_r_joint", (-0.01955, -0.08245, -0.0115), (0, 0, -1)),
    ],
}
# 발 링크(ankle_r) 프레임의 바닥점 = RA30P 패드 4 개 원점의 중심(패드 두께 2.5 mm 아래). 발 법선 = 발 링크 +Y.
SOLE_POINT = (0.000125, -0.0561, 0.009834)


def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, th):
    """(N,) 각 → (N,3,3) 회전. axis 는 단위 정수축."""
    ax = np.asarray(axis, float); c = np.cos(th); s = np.sin(th); C = 1 - c
    x, y, z = ax
    R = np.empty(th.shape + (3, 3))
    R[..., 0, 0] = c + x * x * C; R[..., 0, 1] = x * y * C - z * s; R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s; R[..., 1, 1] = c + y * y * C; R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s; R[..., 2, 1] = z * y * C + x * s; R[..., 2, 2] = c + z * z * C
    return R


_R_BP = _rpy_to_R(*_BASE_TO_PELVIS_RPY)


def fk_sole(q_side: np.ndarray, side: str, sole=SOLE_POINT):
    """q_side (N,6) = 사슬 순서(hip_f, hip_a, hip_r, knee, ankle_f, ankle_r) 관절각 → (발바닥점 (N,3), 발 회전 (N,3,3)) 몸통(base) 프레임."""
    q_side = np.atleast_2d(q_side)
    N = q_side.shape[0]
    R = np.broadcast_to(_R_BP, (N, 3, 3)).copy(); p = np.zeros((N, 3))
    for k, (_, xyz, ax) in enumerate(LEG_CHAIN[side]):
        p = p + R @ np.asarray(xyz, float)
        R = R @ _axis_angle(ax, q_side[:, k])
    return p + R @ np.asarray(sole, float), R


def quat_to_R(qwxyz):
    w, x, y, z = (qwxyz[..., i] for i in range(4))
    R = np.empty(qwxyz.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _rpy_batch(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty(r.shape + (3, 3))
    R[..., 0, 0] = cy * cp; R[..., 0, 1] = cy * sp * sr - sy * cr; R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp; R[..., 1, 1] = sy * sp * sr + cy * cr; R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp; R[..., 2, 1] = cp * sr; R[..., 2, 2] = cp * cr
    return R


class LegOdometry:
    """골반 수평 속도 추정 + 한 주기 창 평균. step() 은 틱마다 1 회(50 Hz).

    반환(dict): v_inst (N,2) 이번 틱 추정(무효 틱 = 직전 값), valid (N,) bool, v_bar (N,2) 창 평균(유효 틱만),
                speed_bar (N,) |v_bar|, frac (N,) 창 안 유효 비율.
    """

    def __init__(self, joint_names, n=1, dt=0.02, window_s=0.9, sole=SOLE_POINT, need_prev_contact=True, z_gate=None):
        names = list(joint_names)
        self.idx = {s: [names.index(j) for j, _, _ in LEG_CHAIN[s]] for s in ("left", "right")}
        self.n = n; self.dt = float(dt); self.W = max(1, int(round(window_s / dt))); self.sole = sole
        self.need_prev = need_prev_contact
        self.z_gate = z_gate      # None = 접촉 비트만 / 값[m] = 접촉 ∧ 발 높이(월드, FK·IMU) ≤ 낮은 발 + z_gate (유지·지연된 접촉 비트의 스윙 혼입 차단)
        self.reset()

    def reset(self, ids=None):
        n, W = self.n, self.W
        if ids is None or not hasattr(self, "yaw"):
            self.yaw = np.zeros(n); self.p_prev = None; self.c_prev = np.zeros((n, 2), bool)
            self.buf = np.zeros((W, n, 2)); self.bvalid = np.zeros((W, n), bool); self.k = 0; self.v_last = np.zeros((n, 2))
        else:
            self.yaw[ids] = 0; self.c_prev[ids] = False; self.buf[:, ids] = 0; self.bvalid[:, ids] = False; self.v_last[ids] = 0
            if self.p_prev is not None:
                self.p_prev[:, ids] = np.nan

    def step(self, enc, gyro_b, grav_b, contact):
        enc = np.atleast_2d(np.asarray(enc, float)); gyro_b = np.atleast_2d(np.asarray(gyro_b, float))
        grav_b = np.atleast_2d(np.asarray(grav_b, float)); c = np.atleast_2d(np.asarray(contact)) > 0.5
        R = _rpy_batch(*self._rp(grav_b), self.yaw)
        p = np.stack([fk_sole(enc[:, self.idx[s]], s, self.sole)[0] for s in ("left", "right")], 0)    # (2,N,3)
        # 요 적분: 월드 z 각속도 = (R ω_b)_z
        wz = np.einsum("nj,nj->n", R[:, 2, :], gyro_b)
        self.yaw = self.yaw + wz * self.dt
        v = np.zeros((self.n, 2)); wsum = np.zeros(self.n)
        if self.z_gate is not None:
            zw = np.einsum("nj,lnj->ln", R[:, 2, :], p)                    # (2,N) 발바닥점 월드 높이(몸통 기준)
            low = zw <= (zw.min(0, keepdims=True) + self.z_gate)
            c = c & low.T
        if self.p_prev is not None:
            for li in range(2):
                ok = c[:, li] & (self.c_prev[:, li] if self.need_prev else True) & np.isfinite(self.p_prev[li]).all(-1)
                pdot = (p[li] - self.p_prev[li]) / self.dt
                vb = -(np.cross(gyro_b, p[li]) + pdot)                   # 몸통 프레임
                vw = np.einsum("nij,nj->ni", R, vb)[:, :2]
                v += np.where(ok[:, None], vw, 0.0); wsum += ok
        valid = wsum > 0
        v = np.where(valid[:, None], v / np.maximum(wsum, 1)[:, None], self.v_last)
        self.v_last = v; self.p_prev = p; self.c_prev = c
        self.buf[self.k] = v; self.bvalid[self.k] = valid; self.k = (self.k + 1) % self.W
        nv = self.bvalid.sum(0)
        vbar = (self.buf * self.bvalid[..., None]).sum(0) / np.maximum(nv, 1)[:, None]
        return {"v_inst": v, "valid": valid, "v_bar": vbar, "speed_bar": np.linalg.norm(vbar, axis=-1), "frac": nv / self.W}

    @staticmethod
    def _rp(grav_b):
        g = grav_b / np.maximum(np.linalg.norm(grav_b, axis=-1, keepdims=True), 1e-9)
        zb = -g
        return np.arctan2(zb[:, 1], zb[:, 2]), np.arcsin(np.clip(-zb[:, 0], -1, 1))


class SettleJudge:
    """판정기(a): 주기 평균 속도 기반 정착. settled = |v̄| < v_thr & 창 유효 비율 ≥ min_frac 가 hold_s 연속 & 활성 뒤 ≥ min_s.
    재무장: arm() 호출(회복 켜짐 순간) 시 창·타이머 초기화하지 않음(오도메트리는 상시) — 창이 활성 전 정지 구간을 품지 않도록 min_s ≥ 창 길이 권장."""

    def __init__(self, n=1, dt=0.02, v_thr=0.07, hold_s=0.2, min_s=0.9, min_frac=0.5):
        self.n, self.dt = n, dt; self.v_thr, self.hold_s, self.min_s, self.min_frac = v_thr, hold_s, min_s, min_frac
        self.t_on = np.zeros(n); self.t_ok = np.zeros(n)

    def arm(self, ids):
        self.t_on[ids] = 0.0; self.t_ok[ids] = 0.0

    def step(self, odo_out, active):
        active = np.asarray(active, bool)
        ok = (odo_out["speed_bar"] < self.v_thr) & (odo_out["frac"] >= self.min_frac)
        self.t_on = np.where(active, self.t_on + self.dt, 0.0)
        self.t_ok = np.where(active & ok, self.t_ok + self.dt, 0.0)
        return active & (self.t_on >= self.min_s - 1e-9) & (self.t_ok >= self.hold_s - 1e-9)


class ContactImuJudge:
    """판정기(b) 대안: 접촉 패턴 + IMU 저주파. settled = 양발 접촉 연속 ≥ ds_s(이륙 없음) & LPF |ω_xy| < w_thr & |기울기 변화율 LPF| 작음.
    걸음을 계속하는 정책(W3 제자리 걸음)에서는 양발 지속 조건이 성립하지 않아 원리적으로 '걸으며 정착'을 판정할 수 없다(서기 정책 회복 전용)."""

    def __init__(self, n=1, dt=0.02, ds_s=0.5, w_thr=0.3, tau_s=0.2, min_s=0.9):
        self.n, self.dt, self.ds_s, self.w_thr, self.min_s = n, dt, ds_s, w_thr, min_s
        self.a = dt / max(tau_s, dt)
        self.t_ds = np.zeros(n); self.w_lp = np.zeros(n); self.t_on = np.zeros(n)

    def step(self, gyro_b, contact, active):
        gyro_b = np.atleast_2d(gyro_b); c = np.atleast_2d(np.asarray(contact)) > 0.5; active = np.asarray(active, bool)
        both = c[:, 0] & c[:, 1]
        self.t_ds = np.where(both, self.t_ds + self.dt, 0.0)
        self.w_lp = self.w_lp + self.a * (np.linalg.norm(gyro_b[:, :2], axis=-1) - self.w_lp)
        self.t_on = np.where(active, self.t_on + self.dt, 0.0)
        return active & (self.t_on >= self.min_s - 1e-9) & (self.t_ds >= self.ds_s - 1e-9) & (self.w_lp < self.w_thr)
