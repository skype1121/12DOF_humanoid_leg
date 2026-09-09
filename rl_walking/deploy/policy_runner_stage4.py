"""RL 보행 정책 실행기 Stage4판 (실물/단독 PC용) — isaaclab 의존 없음.

체크포인트: logs/rsl_rl/biped12_stage4/2026-07-26_14-53-03/exported/{policy.onnx, policy.pt}
정책: 245차원 관측 → 12차원 raw 액션 MLP (50Hz 정책틱).
기존 45차원판(policy_runner.PolicyRunner)은 그대로 두고 확장한 별도 클래스.

관측 245 = 단일 프레임 49 항목 × 이력 T=5 (stage4_env_cfg.Stage4ObservationsCfg /
symmetry.py 레이아웃 상수와 1:1). isaaclab observation_manager는 항별로
CircularBuffer.buffer (N, T, D; T축 index 0=最古)를 (N, T·D)로 reshape 후 항 순서대로
concat한다 — 즉 '항 블록' 구조이며 프레임 인터리브가 아니다:

  항(D)                블록 구간      프레임 내부 배치
  base_ang_vel(3)      [  0: 15)    [프레임0(最古) 3 | 프레임1 3 | ... | 프레임4(最新) 3]
  projected_gravity(3) [ 15: 30)    (이하 동일 — 항별 T=5 프레임, index0=最古)
  velocity_commands(3) [ 30: 45)
  joint_pos(12)        [ 45:105)    절대각 − DEFAULT_POSE_RAD
  joint_vel(12)        [105:165)
  actions(12)          [165:225)    직전 틱 raw 정책출력 (첫 틱 0)
  gait_clock(2)        [225:235)    [sin φ, cos φ], φ = 2π·1.4Hz·tick·0.02s
  feet_contact(2)      [235:245)    [왼발, 오른발] 접촉 이진 (RA30P >5N → 1.0)

이력 백필: isaaclab CircularBuffer.append (circular_buffer.py L136-139)는 리셋 후
'첫' append에서 버퍼 전체를 그 프레임으로 채운다 (num_pushes==0 → 전 슬롯 = data).
이 실행기의 링버퍼도 동일 — 리셋 직후 첫 step()의 관측은 5프레임 전부 현재값이다.

게이트 클록: 학습(mdp_stage4._gait_phase)과 동일 공식 φ = 2π·GAIT_FREQ_HZ·t,
t = tick·0.02s. 리셋 시 tick=0 (env의 episode_length_buf 리셋과 동일) — 외부 입력
없이 내부 틱 카운터로 재현하므로 실물에서도 50Hz 호출만 지키면 된다.

발 접촉 입력(contact2): 실물 RA30P 발바닥 힘센서 → 법선힘 노름 > 5N이면 1.0,
아니면 0.0. 순서 [왼발, 오른발]. 학습 관측(mdp_stage4.feet_contact_binary)과 동일
정의라 sim2real 갭 최소. 0/1 이외 값(예: 생힘 [N])이 들어오면 즉시 에러 —
매핑 실수를 조립 단계에서 잡는다.

액션 파이프라인은 45차원판과 동일 (아핀 ×0.25+기본자세 → 슬루 ±4°/틱 →
PD kp150/kd5 위치명령). last_action 관측은 슬루 前 raw — 반환값 되먹임 금지.

시뮬 전용 — 이 모듈은 CAN/모터에 아무것도 보내지 않는다.

사용:
    from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4
    r = PolicyRunnerStage4()                 # exported/ 자동 로드 (onnx 우선)
    r.reset()
    q_cmd = r.step(gyro, gravity, cmd, qpos, qvel, contact2)   # 50Hz 루프

자가검증 (GPU 불필요):
    /home/ryu/IsaacLab/_isaac_sim/python.sh \
        rl_walking/deploy/policy_runner_stage4.py --selftest
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

if __package__:
    from .policy_runner import (
        ACT_DIM, ACTION_SCALE, DEFAULT_POSE_RAD, JOINT_ORDER, RATE_HZ, SLEW_RAD,
        PolicyRunner, _load_backend,
    )
else:  # 파일 직접 실행 (--selftest)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from policy_runner import (  # type: ignore
        ACT_DIM, ACTION_SCALE, DEFAULT_POSE_RAD, JOINT_ORDER, RATE_HZ, SLEW_RAD,
        PolicyRunner, _load_backend,
    )

__all__ = ["PolicyRunnerStage4", "JOINT_ORDER"]

# ---------------------------------------------------------------------------
# Stage4 확정 상수 (stage4_env_cfg / mdp_stage4 / symmetry.py와 동기 — 변경 금지)
# ---------------------------------------------------------------------------

GAIT_FREQ_HZ = 1.4                       # mdp_stage4.GAIT_FREQ_HZ
POLICY_DT = 1.0 / RATE_HZ                # 0.02s — env.step_dt (200Hz×dec4)
HISTORY_LEN = 5                          # symmetry.HISTORY_LEN
FRAME_DIM = 49                           # symmetry.FRAME_POLICY_DIM
OBS_DIM_STAGE4 = 245                     # symmetry.POLICY_OBS_DIM = 49×5
CONTACT_FORCE_THRESHOLD_N = 5.0          # RA30P 이진화 문턱 (참고 — 이진화는 호출자)

#: 관측 항 (이름, 프레임 차원 D) — Stage4ObservationsCfg.PolicyCfg 선언 순서 그대로.
#: 245 조립은 이 튜플 순서로 항 블록(T·D)을 concat한다.
STAGE4_TERMS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
    ("gait_clock", 2),
    ("feet_contact", 2),
)
assert sum(d for _, d in STAGE4_TERMS) == FRAME_DIM
assert FRAME_DIM * HISTORY_LEN == OBS_DIM_STAGE4

#: S8a (0909) raw+EMA 확장 계약 — policy 그룹 **끝에** 두 항 추가. hang_stand_env_cfg._add_s8_obs /
#: symmetry.policy_s8a / expand_checkpoint.py 와 동기. 245 모델은 그대로(레이아웃은 모델 입력 차원으로 자동 선택).
STAGE4_TERMS_S8A = STAGE4_TERMS + (("gyro_ema", 3), ("joint_vel_ema", 12))
OBS_DIM_S8A = HISTORY_LEN * sum(d for _, d in STAGE4_TERMS_S8A)      # 320
EMA_ALPHA_S8A = 0.1                       # mdp_hang.gyro_ema/joint_vel_ema alpha (τ≈10틱=200ms)
_TERMS_BY_DIM = {OBS_DIM_STAGE4: STAGE4_TERMS, OBS_DIM_S8A: STAGE4_TERMS_S8A}


def _probe_obs_dim(net, candidates=(OBS_DIM_STAGE4, OBS_DIM_S8A)) -> int:
    """모델이 받는 관측 차원 검출 — 0 벡터 추론을 후보 차원(245 → 320)으로 시도. 전부 실패 = 계약 밖 모델."""
    errs = []
    for d in candidates:
        try:
            out = np.asarray(net(np.zeros(d, np.float32)), dtype=np.float32).reshape(-1)
            if out.shape == (ACT_DIM,):
                return d
            errs.append(f"{d}: 출력 {out.shape}")
        except Exception as e:  # noqa: BLE001 — 차원 불일치 예외는 백엔드마다 다르다
            errs.append(f"{d}: {type(e).__name__}")
    raise RuntimeError(f"관측 차원 검출 실패 (후보 {candidates}): {errs}")

#: 기본 모델 경로 (exported/ — policy.onnx 우선, 없으면 policy.pt)
DEFAULT_MODEL_DIR_STAGE4 = (
    "/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_stage4/"
    "2026-07-26_14-53-03/exported"
)


def gait_clock_stage4(tick: int, freq: float = GAIT_FREQ_HZ) -> np.ndarray:
    """게이트 클록 [sin φ, cos φ], φ = 2π·freq·(tick·0.02) [rad] — float32.

    mdp_stage4._gait_phase와 동일 공식·동일 float32 연산 순서:
    t = fl32(tick)·fl32(0.02), φ = fl32(2π·freq)·t (torch 스칼라 승격 규칙 재현).
    freq는 체크포인트별 학습 클록과 일치해야 함 (v5부터 1.1Hz — 기본 1.4Hz는
    기존 체크포인트 전부의 계약이므로 불변).
    """
    t = np.float32(tick) * np.float32(POLICY_DT)
    phase = np.float32(2.0 * math.pi * freq) * t
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


class HeadingHold:
    """상위 헤딩 폐루프 — IMU 요를 보고 wz 명령을 생성해 직진 드리프트를 상쇄.

    정책은 개루프 재생 시 요 편향이 누적된다 (sim2sim 실측: 직진 20s에 −48.8°,
    log_picture/sim2sim_R2.txt). 정책 재학습 대신 명령 계층에서 잡는 게 정석:
    wz = clip(kp·wrap(ref − yaw), ±wz_limit). kp=1.0이면 드리프트율 0.043rad/s
    기준 정상상태 오차 ≈ 2.4°.

    실물에서는 iAHRS 요(rad)를, 시뮬 검증에서는 골반 쿼터니언 요를 넣는다.
    wz_limit은 학습 명령 분포(±0.6) 이내로 제한 — 분포 밖 명령 방지.
    회전 명령 구간에서는 update를 부르지 말고 set_ref로 목표 헤딩만 갱신할 것.
    """

    def __init__(self, kp: float = 1.0, wz_limit: float = 0.6, ref: float = 0.0):
        self.kp = float(kp)
        self.wz_limit = float(wz_limit)
        self.ref = float(ref)

    def set_ref(self, yaw: float) -> None:
        """목표 헤딩 지정 (보행 시작 순간의 현재 요를 넣는 게 안전)."""
        self.ref = float(yaw)

    def update(self, yaw: float) -> float:
        """현재 요(rad) → wz 명령(rad/s). 각도 랩어라운드(±π) 처리 포함."""
        err = (self.ref - float(yaw) + math.pi) % (2.0 * math.pi) - math.pi
        wz = self.kp * err
        return max(-self.wz_limit, min(self.wz_limit, wz))


def yaw_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
    """쿼터니언(w,x,y,z) → 요(rad). isaaclab euler_xyz_from_quat의 요 성분과 동일식."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


class PositionHold:
    """상위 위치 유지 폐루프 — 제자리 걷기(march) 드리프트 상쇄.

    march 실측(2026-07-26): 정책 관측에 위치·선속도가 없어(IMU·엔코더만)
    블라인드 정책 단독으로는 드리프트를 못 잡는다 (앵커 벌점 학습 2.0→1.8m 한계).
    HeadingHold와 동일 패턴의 명령 계층이 정공법:
      cmd_xy(몸 기준) = clip(-kp·(현위치-기준점, 몸 프레임 회전), ±v_limit)
    정책은 이 소속도 명령을 잘 추종하므로(학습 분포 내) 재학습 불필요.

    위치 입력: 시뮬=루트 위치, 실물=다리 오도메트리(엔코더+접촉) 또는 외부 측위.
    v_limit 기본 0.15 m/s — 제자리 유지 목적의 저속 보정 (학습 분포 내).
    """

    def __init__(self, kp: float = 0.8, v_limit: float = 0.15,
                 ref_xy: tuple = (0.0, 0.0)):
        self.kp = float(kp)
        self.v_limit = float(v_limit)
        self.ref = (float(ref_xy[0]), float(ref_xy[1]))

    def set_ref(self, x: float, y: float) -> None:
        """기준점 지정 (march 시작 순간의 현재 위치 권장)."""
        self.ref = (float(x), float(y))

    def update(self, x: float, y: float, yaw: float) -> tuple:
        """현 위치(월드 xy)·요 → (vx_cmd, vy_cmd) 몸 프레임 [m/s]."""
        ex, ey = self.ref[0] - float(x), self.ref[1] - float(y)  # 월드 오차
        c, s = math.cos(yaw), math.sin(yaw)
        bx = c * ex + s * ey      # 몸 프레임 (요 회전 역변환)
        by = -s * ex + c * ey
        vx = max(-self.v_limit, min(self.v_limit, self.kp * bx))
        vy = max(-self.v_limit, min(self.v_limit, self.kp * by))
        return vx, vy


class _TermHistory:
    """isaaclab CircularBuffer(batch=1) 동작 재현 — 항별 (T, D) 링버퍼.

    - append: 리셋 후 첫 push는 버퍼 전체 백필 (circular_buffer.py L136-139와 동일),
      이후는 한 칸 밀고 맨 뒤(最新)에 기록.
    - flat(): CircularBuffer.buffer.reshape(-1)와 동일 — index0=最古, 끝=最新.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._buf = np.zeros((HISTORY_LEN, dim), dtype=np.float32)
        self._pushes = 0

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._pushes = 0

    def append(self, frame: np.ndarray) -> None:
        f = np.asarray(frame, dtype=np.float32).reshape(self.dim)
        if self._pushes == 0:
            self._buf[:] = f              # 첫 push 백필
        else:
            self._buf = np.roll(self._buf, -1, axis=0)
            self._buf[-1] = f
        self._pushes += 1

    def flat(self) -> np.ndarray:
        return self._buf.reshape(-1)      # (T·D,) — 프레임0(最古)부터


# ---------------------------------------------------------------------------
# 실행기 본체
# ---------------------------------------------------------------------------

class PolicyRunnerStage4:
    """Stage4: 관측 245 조립(이력 5·클록·접촉) → 추론 → 아핀+슬루 → 목표각 12 [rad].

    50Hz(RATE_HZ) 호출은 호출자 책임 — 클록이 tick·0.02s로 시간을 재구성하므로
    주기가 틀어지면 학습 위상과 어긋난다. reset() 시 tick=0, 이력·last_action 초기화,
    슬루 기준점은 리셋 후 '첫 step()의 qpos_rad'로 재초기화 (45차원판과 동일).

    step() 직후 self.last_obs에 이번 틱 245차원 관측이 남는다 (시뮬 검증용).
    """

    def __init__(self, model_path=DEFAULT_MODEL_DIR_STAGE4, backend: str = "auto",
                 gait_freq: float = GAIT_FREQ_HZ, obs_dim: int | None = None):
        """model_path: exported/ 디렉토리 또는 .onnx/.pt 파일 경로 (callable=스텁).
        backend: 'auto'(onnx 우선→torch 폴백) | 'onnx' | 'torch'.
        gait_freq: 클록 주파수 [Hz] — 반드시 체크포인트의 학습 클록과 일치
        (v5=1.1, 그 외=1.4). 불일치 = 위상 어긋난 관측 → 보행 파괴.
        obs_dim: 245(기존) | 320(S8a raw+EMA). None = 모델에서 검출 (스텁은 245 기본).
        """
        self.gait_freq = float(gait_freq)
        self.backend, self._net = _load_backend(model_path, backend)
        if obs_dim is None:
            obs_dim = OBS_DIM_STAGE4 if self.backend == "stub" else _probe_obs_dim(self._net)
        if int(obs_dim) not in _TERMS_BY_DIM:
            raise ValueError(f"지원하지 않는 관측 차원 {obs_dim} (245 | 320)")
        self.obs_dim = int(obs_dim)
        self.terms = _TERMS_BY_DIM[self.obs_dim]
        self._hist = {name: _TermHistory(dim) for name, dim in self.terms}
        self._ema = {}                   # S8a: 항 → 현재 EMA (리셋 시 비움; 첫 틱 = 현재값 — mdp_hang._ema_state 와 동일)
        # 절대각 소프트 리밋 로드 (joint_limits_12dof.json — 표준lib 로더).
        # 실패 시 클램프 비활성 (기존 동작 유지) — 실물 배포에서는 반드시 활성 확인.
        try:
            _root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from robot_runtime.joint_limits import get_absolute_limits_deg
            _lims = get_absolute_limits_deg()
            self._limits_lo = np.array(
                [math.radians(_lims[j]["soft_min"]) for j in JOINT_ORDER],
                dtype=np.float32)
            self._limits_hi = np.array(
                [math.radians(_lims[j]["soft_max"]) for j in JOINT_ORDER],
                dtype=np.float32)
        except Exception:
            self._limits_lo = None
            self._limits_hi = None
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.last_obs: np.ndarray | None = None
        self._prev_emitted = None        # 슬루 기준점 — 첫 step()에서 qpos로 초기화
        self._tick = 0                   # 클록 틱 (env episode_length_buf 대응)

    def reset(self) -> None:
        """에피소드/세션 초기화 — tick=0, 이력 비움(다음 step에서 백필), last_action=0."""
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.last_obs = None
        self._prev_emitted = None
        self._tick = 0
        self._ema = {}
        for h in self._hist.values():
            h.reset()

    # ---------- 관측 ----------

    @staticmethod
    def _contact2(x) -> np.ndarray:
        """접촉 이진 [L, R] 검증 — 0.0/1.0(또는 bool)만 허용, 생힘[N] 입력 차단."""
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        if v.shape != (2,):
            raise ValueError(f"contact2: 길이 2 필요, 받은 shape {np.shape(x)}")
        if not np.all((v == 0.0) | (v == 1.0)):
            raise ValueError(
                f"contact2 값은 0/1 이진이어야 함 (받음: {v.tolist()}) — "
                "RA30P 생힘[N]이면 (힘>5N)로 먼저 이진화할 것"
            )
        return v

    def assemble_obs(self, gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s,
                     contact2) -> np.ndarray:
        """이번 틱 프레임을 이력에 push하고 245차원 관측을 조립 (상태 변경 있음!).

        step()이 내부에서 호출한다 — 단독 호출은 시뮬 검증/테스트 전용.
        틱당 정확히 1회만 불러야 한다 (이력·클록이 어긋남).
        """
        vec = PolicyRunner._vec
        frames = {
            "base_ang_vel": vec(gyro_rad_s, 3, "gyro_rad_s"),
            "projected_gravity": vec(gravity_unit, 3, "gravity_unit"),
            "velocity_commands": vec(cmd, 3, "cmd"),
            "joint_pos": vec(qpos_rad, 12, "qpos_rad") - DEFAULT_POSE_RAD,
            "joint_vel": vec(qvel_rad_s, 12, "qvel_rad_s"),
            "actions": self.last_action,
            "gait_clock": gait_clock_stage4(self._tick, self.gait_freq),
            "feet_contact": self._contact2(contact2),
        }
        if self.obs_dim == OBS_DIM_S8A:
            # raw+EMA(α0.1): 첫 틱은 현재값으로 시작, 이후 (1−α)·prev + α·x — 시뮬 mdp_hang._ema_state 와 같은 수열
            for name, src in (("gyro_ema", "base_ang_vel"), ("joint_vel_ema", "joint_vel")):
                x = np.asarray(frames[src], dtype=np.float32)
                prev = self._ema.get(name)
                self._ema[name] = x.copy() if prev is None else (1.0 - EMA_ALPHA_S8A) * prev + EMA_ALPHA_S8A * x
                frames[name] = self._ema[name]
        for name, _dim in self.terms:
            self._hist[name].append(frames[name])
        obs = np.concatenate([self._hist[name].flat() for name, _ in self.terms])
        assert obs.shape == (self.obs_dim,) and obs.dtype == np.float32
        return obs

    # ---------- 스텝 ----------

    def step(self, gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s,
             contact2) -> np.ndarray:
        """1 정책틱 → 슬루 적용된 목표관절각 12개 [rad] (JOINT_ORDER 순).

        contact2: 발 접촉 이진 [왼발, 오른발] — RA30P 법선힘 > 5N이면 1.0.
        관측의 actions 최신 프레임에는 '이전 틱' raw가 들어가고 (첫 틱 0),
        이번 틱 raw는 반환 전에 내부 저장 — 반환값(슬루 후)을 되먹이지 말 것.
        """
        qpos = PolicyRunner._vec(qpos_rad, 12, "qpos_rad")
        obs = self.assemble_obs(
            gyro_rad_s, gravity_unit, cmd, qpos, qvel_rad_s, contact2)
        self.last_obs = obs
        # env 등가성: env는 step() 안에서 episode_length_buf를 '먼저' +1 하고 obs를
        # 계산하지만(manager_based_rl_env.py L201→L238), 리셋 직후 첫 obs는 reset()이
        # buf=0으로 계산한다 — 즉 n번째 정책호출이 보는 위상은 항상 φ(n·dt), n=0부터.
        # 이 러너는 'tick으로 조립 후 +1'로 같은 수열을 재현한다. 순서 변경 금지
        # (조립 전에 +1하면 위상이 1틱 앞서 학습 분포와 어긋난다).
        self._tick += 1
        raw = np.asarray(self._net(obs), dtype=np.float32).reshape(-1)
        if raw.shape != (ACT_DIM,):
            raise RuntimeError(f"정책 출력 shape 이상: {raw.shape} (기대 ({ACT_DIM},))")
        self.last_action = raw.copy()
        # 아핀 → 슬루. ⚠ 타깃 클램프 금지 (2026-08-02 회귀 수정 — 0726 밤
        # 22c0385가 soft 리밋 클램프를 여기 추가했으나 폐루프 재검증 없이 배포됨):
        # 학습 env(SlewJointPositionAction)는 타깃을 클램프하지 않는다. 정책 타깃은
        # '위치'가 아니라 슬루가 방향·속도로 해석하는 지시값 (실측: 정상 보행 중
        # 무릎 타깃 -158°~+206° 등 리밋 밖이 일상) — 클램프하면 학습과 1:1이
        # 깨져 실물 발진 사고의 근원이 됐다 (폐루프 검증 목표각 오차 3.6°).
        # 위치 안전 울타리는 Stage8 노드의 절대각(soft/hard) 검사가 '방출 명령'
        # 기준으로 담당한다 — cansend 사고 후속 보호는 그 계층에서 유지됨.
        target = DEFAULT_POSE_RAD + ACTION_SCALE * raw
        if self._prev_emitted is None:
            self._prev_emitted = qpos.copy()
        emitted = self._prev_emitted + np.clip(
            target - self._prev_emitted, -SLEW_RAD, SLEW_RAD)
        self._prev_emitted = emitted
        return emitted.copy()


# ---------------------------------------------------------------------------
# 자가검증 (--selftest) — GPU 불필요, CPU만
# ---------------------------------------------------------------------------

def _term_offsets() -> dict:
    """항 이름 → (블록 시작 오프셋, D). 골든테스트의 독립 인덱스 산술용."""
    out, off = {}, 0
    for name, dim in STAGE4_TERMS:
        out[name] = (off, dim)
        off += HISTORY_LEN * dim
    assert off == OBS_DIM_STAGE4
    return out


def _mk_inputs(k: int):
    """스텝 k의 결정적 원신호 세트 (골든/백필 테스트 공용)."""
    gyro = np.float32([0.1, -0.2, 0.3]) + np.float32(0.01) * k
    grav = np.float32([0.02, -0.05, -0.9985]) - np.float32(0.001) * k
    cmd = np.float32([0.5, 0.0, -0.1]) + np.float32(0.02) * k
    qpos = DEFAULT_POSE_RAD + np.float32(0.01) * (np.arange(12, dtype=np.float32) + k)
    qvel = np.float32(-0.02) * (np.arange(12, dtype=np.float32) - k)
    contact = np.float32([k % 2, 1 - (k % 2)])
    return gyro, grav, cmd, qpos, qvel, contact


def _selftest_golden() -> str:
    """(a) 245 조립 골든: 5스텝 구동 후 전 항×전 프레임을 독립 인덱스 산술로 대조."""
    raws = []

    def stub(obs):
        raw = np.float32(0.1) * (np.arange(12, dtype=np.float32) - len(raws))
        raws.append(raw)
        return raw

    r = PolicyRunnerStage4(model_path=stub)
    r.reset()
    per_step = []
    for k in range(HISTORY_LEN):
        gyro, grav, cmd, qpos, qvel, contact = _mk_inputs(k)
        r.step(gyro, grav, cmd, qpos, qvel, contact)
        per_step.append({
            "base_ang_vel": gyro,
            "projected_gravity": grav,
            "velocity_commands": cmd,
            "joint_pos": qpos - DEFAULT_POSE_RAD,
            "joint_vel": qvel,
            # actions 프레임 k = 이전 스텝 raw (k=0은 0)
            "actions": np.zeros(12, np.float32) if k == 0 else raws[k - 1],
            "gait_clock": gait_clock_stage4(k),
            "feet_contact": contact,
        })
    obs = r.last_obs
    assert obs is not None and obs.shape == (OBS_DIM_STAGE4,) and obs.dtype == np.float32

    # 기대값을 concat이 아닌 '오프셋 직접 기입'으로 독립 구성 (항 블록·프레임 순서 검증)
    exp = np.zeros(OBS_DIM_STAGE4, np.float32)
    for name, (off, dim) in _term_offsets().items():
        for t in range(HISTORY_LEN):      # 5 push 후: 프레임 t = 스텝 t (index0=最古)
            exp[off + t * dim: off + (t + 1) * dim] = per_step[t][name]
    assert np.array_equal(obs, exp), \
        f"골든 불일치 max|Δ|={np.abs(obs - exp).max()} @ {np.abs(obs - exp).argmax()}"
    # 블록 경계 개별 확인: joint_pos 최신 프레임의 무릎 오프셋, feet_contact 마지막 원소
    off_jp, _ = _term_offsets()["joint_pos"]
    assert abs(obs[off_jp + 4 * 12 + 6]
               - (per_step[4]["joint_pos"][6])) < 1e-7
    assert obs[OBS_DIM_STAGE4 - 1] == per_step[4]["feet_contact"][1]
    return "245 조립 골든 OK (항 블록 순서·프레임 T축·오프셋 전수 일치)"


def _selftest_backfill() -> str:
    """(b) 이력 백필: 첫 push 전체 백필 + 롤링 + 리셋 후 재백필 (CircularBuffer 정합)."""
    stub = lambda obs: np.zeros(ACT_DIM, np.float32)  # noqa: E731
    r = PolicyRunnerStage4(model_path=stub)
    r.reset()
    offs = _term_offsets()

    # 1스텝: 모든 항의 5프레임 전부 = 현재 프레임 (백필)
    g0 = _mk_inputs(0)
    r.step(*g0)
    obs = r.last_obs
    for name, (off, dim) in offs.items():
        blk = obs[off: off + HISTORY_LEN * dim].reshape(HISTORY_LEN, dim)
        assert np.array_equal(blk, np.tile(blk[-1], (HISTORY_LEN, 1))), \
            f"{name}: 첫 스텝 백필 실패 (프레임들이 서로 다름)"
    # 백필 값 자체 검증 (base_ang_vel = gyro0)
    off_av, _ = offs["base_ang_vel"]
    assert np.array_equal(obs[off_av: off_av + 3], g0[0]), "백필 값이 첫 프레임이 아님"

    # 3스텝까지: [f0, f0, f0, f1, f2] — 롤링과 백필 잔존이 CircularBuffer와 동일
    g1, g2 = _mk_inputs(1), _mk_inputs(2)
    r.step(*g1)
    r.step(*g2)
    blk = r.last_obs[off_av: off_av + 15].reshape(HISTORY_LEN, 3)
    exp = np.stack([g0[0], g0[0], g0[0], g1[0], g2[0]])
    assert np.array_equal(blk, exp), f"3 push 롤링 불일치:\n{blk}\nvs\n{exp}"

    # 리셋 후 새 값으로 재백필 — 이전 에피소드 잔존 없음
    r.reset()
    g9 = _mk_inputs(9)
    r.step(*g9)
    blk = r.last_obs[off_av: off_av + 15].reshape(HISTORY_LEN, 3)
    assert np.array_equal(blk, np.tile(g9[0], (HISTORY_LEN, 1))), "리셋 후 잔존 이력"

    # actions 이력: 첫 틱 0 백필 → 2틱째 최신 프레임 = 1틱 raw
    raws = []

    def stub2(obs):
        raw = np.full(ACT_DIM, 0.5 + 0.5 * len(raws), np.float32)
        raws.append(raw)
        return raw

    r2 = PolicyRunnerStage4(model_path=stub2)
    r2.step(*_mk_inputs(0))
    off_a, _ = offs["actions"]
    assert np.array_equal(
        r2.last_obs[off_a: off_a + 60], np.zeros(60, np.float32)), "첫 틱 actions != 0"
    r2.step(*_mk_inputs(1))
    blk = r2.last_obs[off_a: off_a + 60].reshape(HISTORY_LEN, 12)
    assert np.array_equal(blk[-1], raws[0]) and np.array_equal(blk[0], np.zeros(12)), \
        "actions 이력이 '이전 raw'가 아님"
    return "이력 백필 OK (첫 push 전체 백필·롤링·리셋 격리 — CircularBuffer L136-139 정합)"


def _selftest_clock() -> str:
    """(c) 클록 공식: φ = 2π·1.4·tick·0.02, reset()시 tick=0 — float64 독립 계산 대조."""
    stub = lambda obs: np.zeros(ACT_DIM, np.float32)  # noqa: E731
    r = PolicyRunnerStage4(model_path=stub)
    off_c, _ = _term_offsets()["gait_clock"]

    r.reset()
    for k in range(60):                   # 1.2s > 1주기(1/1.4s) — 위상 랩 포함
        r.step(*_mk_inputs(k))
        newest = r.last_obs[off_c + 4 * 2: off_c + 5 * 2]
        phi = 2.0 * math.pi * GAIT_FREQ_HZ * (k * POLICY_DT)   # float64 독립 계산
        assert abs(newest[0] - math.sin(phi)) < 1e-5, f"tick{k} sin 불일치"
        assert abs(newest[1] - math.cos(phi)) < 1e-5, f"tick{k} cos 불일치"
    # 첫 틱 정확값: φ=0 → (sin, cos) = (0, 1) 정확히 + 백필로 5프레임 동일
    r.reset()
    r.step(*_mk_inputs(0))
    blk = r.last_obs[off_c: off_c + 10].reshape(HISTORY_LEN, 2)
    assert np.array_equal(blk, np.tile(np.float32([0.0, 1.0]), (HISTORY_LEN, 1))), \
        "reset 후 첫 틱 클록 != (0,1) 백필"
    # 2틱째 이력: [f0×4, f1] — 클록 프레임도 이력 규약을 따름
    r.step(*_mk_inputs(1))
    blk = r.last_obs[off_c: off_c + 10].reshape(HISTORY_LEN, 2)
    assert np.array_equal(blk[3], np.float32([0.0, 1.0]))
    assert np.array_equal(blk[4], gait_clock_stage4(1))
    return "클록 공식 OK (2π·1.4·tick·0.02, reset→0, 랩 포함 60틱 float64 대조)"


def _selftest_slew() -> str:
    """(d) 아핀+슬루+contact 검증이 45차원판과 동일하게 동작하는지 (스텁)."""
    zeros3 = np.zeros(3)
    grav = np.float32([0.0, 0.0, -1.0])
    qpos0 = DEFAULT_POSE_RAD.copy()
    both = np.float32([1.0, 1.0])

    r = PolicyRunnerStage4(model_path=lambda o: np.full(ACT_DIM, 1.0, np.float32))
    out1 = r.step(zeros3, grav, zeros3, qpos0, np.zeros(12), both)
    assert np.allclose(out1, qpos0 + SLEW_RAD, atol=1e-7), "1틱 슬루 포화 불일치"
    out2 = r.step(zeros3, grav, zeros3, qpos0, np.zeros(12), both)
    assert np.allclose(out2, qpos0 + 2 * SLEW_RAD, atol=1e-7), "2틱 슬루 누적 불일치"

    # 리셋 후 슬루 기준점 = 새 측정각 (default_pose 아님)
    r.reset()
    qposX = DEFAULT_POSE_RAD + 0.5
    outr = r.step(zeros3, grav, zeros3, qposX, np.zeros(12), both)
    assert np.allclose(outr, qposX - SLEW_RAD, atol=1e-6), "리셋 후 슬루 기준점 오류"

    # contact 검증: 생힘[N]/이진 아님 → 즉시 에러
    try:
        r.step(zeros3, grav, zeros3, qpos0, np.zeros(12), np.float32([42.0, 0.0]))
        raise AssertionError("비이진 contact가 통과됨")
    except ValueError:
        pass
    return "슬루/리셋 기준점/contact 이진 검증 OK"


def _selftest_model() -> str:
    """(e) 실제 Stage4 체크포인트 로드 + 1회 추론 (245 입력 수용·유한성·슬루 한도)."""
    r = PolicyRunnerStage4(DEFAULT_MODEL_DIR_STAGE4, backend="auto")
    r.reset()
    out = r.step(
        gyro_rad_s=np.zeros(3),
        gravity_unit=np.float32([0.0, 0.0, -1.0]),
        cmd=np.zeros(3),                 # 시작은 (0,0,0) 스탠딩부터
        qpos_rad=DEFAULT_POSE_RAD.copy(),
        qvel_rad_s=np.zeros(12),
        contact2=np.float32([1.0, 1.0]),  # 직립 양발 접지
    )
    assert out.shape == (ACT_DIM,) and out.dtype == np.float32, out.shape
    assert np.all(np.isfinite(out)), "추론 출력에 NaN/Inf"
    assert np.all(np.abs(out - DEFAULT_POSE_RAD) <= SLEW_RAD + 1e-6), \
        "첫 틱 출력이 슬루 한도(기준점=qpos)를 벗어남"
    assert r.last_obs.shape == (OBS_DIM_STAGE4,), "관측 245 아님"
    return (f"모델 로드+추론 OK (백엔드={r.backend}, 관측 245→출력 12 유한, "
            f"첫 틱 |Δ|max={np.abs(out - DEFAULT_POSE_RAD).max():.4f} rad ≤ 슬루)")


def _selftest_s8a() -> str:
    """(f) S8a 320 조립: 앞 245 는 245판과 비트 동일, 뒤 75 = EMA(α0.1) 이력(자이로 3·관절속도 12 × 5틀) 독립 산술 대조,
    리셋 후 EMA 가 현재값으로 재시작. 모델 차원 검출(_probe_obs_dim)은 245/320 스텁으로 확인."""
    def stub(obs):
        return np.zeros(ACT_DIM, np.float32)

    r245 = PolicyRunnerStage4(model_path=stub)
    r320 = PolicyRunnerStage4(model_path=stub, obs_dim=OBS_DIM_S8A)
    r245.reset(); r320.reset()
    eg = ej = None; hist_g, hist_j = [], []
    for k in range(7):
        ins = _mk_inputs(k)
        r245.step(*ins); r320.step(*ins)
        g = np.asarray(ins[0], np.float32); j = np.asarray(ins[4], np.float32)
        eg = g.copy() if eg is None else 0.9 * eg + 0.1 * g
        ej = j.copy() if ej is None else 0.9 * ej + 0.1 * j
        hist_g.append(eg.copy()); hist_j.append(ej.copy())
    o = r320.last_obs
    assert o.shape == (OBS_DIM_S8A,) and o.dtype == np.float32
    assert np.array_equal(o[:OBS_DIM_STAGE4], r245.last_obs), "앞 245 가 245판과 다름"
    exp = np.concatenate([np.stack(hist_g[-HISTORY_LEN:]).reshape(-1), np.stack(hist_j[-HISTORY_LEN:]).reshape(-1)])
    assert np.allclose(o[OBS_DIM_STAGE4:], exp, atol=1e-6), "EMA 블록 불일치"
    r320.reset(); ins9 = _mk_inputs(9); r320.step(*ins9)
    o2 = r320.last_obs
    assert np.allclose(o2[OBS_DIM_STAGE4:OBS_DIM_STAGE4 + 15], np.tile(np.asarray(ins9[0], np.float32), HISTORY_LEN)), \
        "리셋 후 EMA 가 현재값으로 재시작하지 않음"
    # 차원 검출: 320 만 받는 스텁 → 320, 245 만 받는 스텁 → 245
    def only(d):
        def f(obs):
            if obs.shape != (d,):
                raise ValueError("dim")
            return np.zeros(ACT_DIM, np.float32)
        return f
    assert _probe_obs_dim(only(OBS_DIM_S8A)) == OBS_DIM_S8A and _probe_obs_dim(only(OBS_DIM_STAGE4)) == OBS_DIM_STAGE4
    return "S8a 320 조립 OK (앞 245 비트동일·EMA α0.1 이력 일치·리셋 재시작·차원 검출 245/320)"


def _run_selftest() -> int:
    tests = [("조립 골든", _selftest_golden),
             ("이력 백필", _selftest_backfill),
             ("클록 공식", _selftest_clock),
             ("슬루/입력검증", _selftest_slew),
             ("모델 추론", _selftest_model),
             ("S8a 320 조립", _selftest_s8a)]
    failed = 0
    for name, fn in tests:
        try:
            msg = fn()
            print(f"[PASS] {name}: {msg}")
        except Exception as e:  # noqa: BLE001 — selftest는 전 항목 보고가 목적
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print("selftest:", "ALL PASS" if failed == 0 else f"{failed} FAILED")
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="RL 보행 정책 실행기 Stage4판 (시뮬 전용)")
    ap.add_argument("--selftest", action="store_true", help="자가검증 실행 (GPU 불필요)")
    args = ap.parse_args()
    if args.selftest:
        return _run_selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
