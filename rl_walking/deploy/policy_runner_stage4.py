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


def _hturn2_mod():
    """1006 HTURN2 실험 모듈 지연 import — 플래그가 꺼져 있으면 import 자체가 일어나지 않는다."""
    if __package__:
        from . import hturn2
    else:  # 파일 직접 실행 (--selftest)
        import hturn2  # type: ignore
    return hturn2


def _w3_mod():
    """1006 W3 실험 모듈 지연 import — 플래그가 꺼져 있으면 import 자체가 일어나지 않는다."""
    if __package__:
        from . import w3_deploy
    else:  # 파일 직접 실행 (--selftest)
        import w3_deploy  # type: ignore
    return w3_deploy

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
#: G13 (0930) — S8a 끝에 last_action **16틀** 이력 추가(실물 명령→반응 지연 ~16틱 동안 보낸 명령을 정책이 보게).
#: hang_stand_env_cfg.Biped12HangStandG13EnvCfg / symmetry.policy_g13 와 동기. 이 항만 이력 길이가 다르다.
G13_ACT_HIST = 16
STAGE4_TERMS_G13 = STAGE4_TERMS_S8A + (("action_hist16", 12),)
TERM_HISTORY = {"action_hist16": G13_ACT_HIST}        # 없는 항 = HISTORY_LEN
OBS_DIM_G13 = OBS_DIM_S8A + G13_ACT_HIST * 12                     # 512
_TERMS_BY_DIM = {OBS_DIM_STAGE4: STAGE4_TERMS, OBS_DIM_S8A: STAGE4_TERMS_S8A, OBS_DIM_G13: STAGE4_TERMS_G13}


def _probe_obs_dim(net, candidates=(OBS_DIM_STAGE4, OBS_DIM_S8A, OBS_DIM_G13)) -> int:
    """모델이 받는 관측 차원 검출 — 0 벡터 추론을 후보 차원(245 → 320 → 512)으로 시도. 전부 실패 = 계약 밖 모델."""
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
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 0910: PC 절대경로 → 저장소 상대(젯슨 ~/humanoid_repo 에서도 유효). 브리지는 항상 명시 경로를 넘기므로 런타임 영향 없음(selftest 기본값).
DEFAULT_MODEL_DIR_STAGE4 = os.path.join(
    _REPO_ROOT, "logs/rsl_rl/biped12_stage4/"
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

    def __init__(self, dim: int, length: int = HISTORY_LEN):
        self.dim = dim
        self._buf = np.zeros((int(length), dim), dtype=np.float32)
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
                 gait_freq: float = GAIT_FREQ_HZ, obs_dim: int | None = None,
                 default_pose_rad=None, jvel_ema_mode: str = "v2",
                 act_obs_ema_beta: float | None = None, jvel_obs_ema_beta: float | None = None,
                 hturn2=None, w3=None):
        """model_path: exported/ 디렉토리 또는 .onnx/.pt 파일 경로 (callable=스텁).
        backend: 'auto'(onnx 우선→torch 폴백) | 'onnx' | 'torch'.
        gait_freq: 클록 주파수 [Hz] — 반드시 체크포인트의 학습 클록과 일치
        (v5=1.1, 그 외=1.4). 불일치 = 위상 어긋난 관측 → 보행 파괴.
        obs_dim: 245(기존) | 320(S8a raw+EMA). None = 모델에서 검출 (스텁은 245 기본).
        jvel_ema_mode: 'v2'(기본, 기존 동작) = joint_vel_ema 에 실제 관절속도의 EMA |
            'ema0' = joint_vel_ema 프레임을 0 으로 (1001: G4~G15 는 학습 중 이 채널이 0+잡음이었다 — mdp_hang 구버전
            encoder_joint_vel_ema 결함. 그 정책들의 학습 계약 복원용. EMA 상태는 진단용으로 계속 계산해 last_jvel_ema 에 둔다).
        act_obs_ema_beta: (1003 G18f~G21a 계약) 관측 'actions' = 원시 액션의 EMA(β) — 학습 env mdp_hang.last_action_lpf.
            None = 원시 직전 액션(종전, G12 등). 배포 β = 브리지 stand LPF 0.27.
        jvel_obs_ema_beta: (1003 G19v~G21a 계약) 관측 'joint_vel' = 관절속도의 EMA(β) — 학습 env mdp_hang.encoder_joint_vel_lpf.
            None = 원시 관절속도(종전). ⚠ joint_vel_ema(α0.1) 채널은 여전히 **원시** 속도에서 계산(학습 encoder_joint_vel_ema_v2 와 동일).
        두 상태는 _ema 에 있고 reset() 이 비운다(정책 전환 시 BridgeCore 가 러너를 새로 만든다).
        hturn2: (1006 **실험 플래그, 기본 None = 종전과 비트 동일**) rl_walking.deploy.hturn2.Hturn2Config — 켜면
            ① 관측 gait_clock = 명령 연동 cadence 위상 누적(W2q 계약, gait_freq 무시) ② 러너 target 직후·슬루 앞에
            회전 피드포워드 hip_r 잔차(install_turn_assist 식). 브리지가 --experimental-hturn2 + HTURN2 키일 때만 넘긴다.
        w3: (1006 **실험 플래그, 기본 None = 종전과 비트 동일**) rl_walking.deploy.w3_deploy.W3Config — 켜면 관측 gait_clock =
            W3 계약([sin 2πu, cos 2πu] × ge_obs, gait_enable 래치·0.2 s 램프, gait_freq 무시). ge 는 set_gait_enable() 로 준다
            (reset 뒤 기본 False = 서기). 행동 잔차는 없다(보조·참조는 배포에 없음). hturn2 와 동시 사용 불가.
        """
        if hturn2 is not None and w3 is not None:
            raise ValueError("hturn2 와 w3 는 동시에 켤 수 없다")
        for _b in (act_obs_ema_beta, jvel_obs_ema_beta):
            if _b is not None and not (0.0 < float(_b) <= 1.0):
                raise ValueError(f"관측 EMA β 는 (0,1] (받음 {_b})")
        self.act_obs_ema_beta = None if act_obs_ema_beta is None else float(act_obs_ema_beta)
        self.jvel_obs_ema_beta = None if jvel_obs_ema_beta is None else float(jvel_obs_ema_beta)
        if jvel_ema_mode not in ("v2", "ema0"):
            raise ValueError(f"jvel_ema_mode 는 'v2' | 'ema0' (받음 {jvel_ema_mode!r})")
        self.jvel_ema_mode = jvel_ema_mode
        self.last_jvel_ema = None        # 이번 틱 실제 속도 EMA (ema0 에서도 계산 — 블랙박스 진단용)
        self.gait_freq = float(gait_freq)
        # 0918: 체크포인트별 기본자세(S9c 크라우치 등). None = 학습 공통 DEFAULT_POSE_RAD. 관측 joint_pos·액션 오프셋 둘 다 이 값.
        self.default_pose = (np.asarray(default_pose_rad, dtype=np.float32).reshape(12)
                             if default_pose_rad is not None else DEFAULT_POSE_RAD.copy())
        self.backend, self._net = _load_backend(model_path, backend)
        if obs_dim is None:
            obs_dim = OBS_DIM_STAGE4 if self.backend == "stub" else _probe_obs_dim(self._net)
        if int(obs_dim) not in _TERMS_BY_DIM:
            raise ValueError(f"지원하지 않는 관측 차원 {obs_dim} (245 | 320 | 512)")
        self.obs_dim = int(obs_dim)
        self.terms = _TERMS_BY_DIM[self.obs_dim]
        if self.jvel_ema_mode == "ema0" and "joint_vel_ema" not in dict(self.terms):
            raise ValueError(f"jvel_ema_mode='ema0' 은 joint_vel_ema 항이 있는 모델(320/512)에만 — 이 모델 {self.obs_dim}")
        self._hist = {name: _TermHistory(dim, TERM_HISTORY.get(name, HISTORY_LEN)) for name, dim in self.terms}
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
        # 1006 HTURN2 실험 플래그 — None 이면 아래 상태는 전혀 쓰이지 않는다(종전 경로 그대로).
        self.hturn2 = hturn2
        self._h2_clock = None
        self.last_ff = None              # 이번 틱 피드포워드 잔차 (left_hip_r, right_hip_r) [rad] — 블랙박스 진단용
        if hturn2 is not None:
            self._h2_clock = _hturn2_mod().CadenceClock(hturn2)
            self._h2_iL = JOINT_ORDER.index("left_hip_r_joint")
            self._h2_iR = JOINT_ORDER.index("right_hip_r_joint")
            self._h2_cmd = (0.0, 0.0, 0.0)
        # 1006 W3 실험 플래그 — None 이면 아래 상태는 전혀 쓰이지 않는다(종전 경로 그대로).
        self.w3 = w3
        self._w3_clock = _w3_mod().W3GaitClock(w3) if w3 is not None else None

    def reset(self) -> None:
        """에피소드/세션 초기화 — tick=0, 이력 비움(다음 step에서 백필), last_action=0."""
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self.last_obs = None
        self._prev_emitted = None
        self._tick = 0
        self._ema = {}
        self.last_jvel_ema = None
        for h in self._hist.values():
            h.reset()
        if self._h2_clock is not None:   # 1006 HTURN2 (실험 플래그 켜짐에만)
            self._h2_clock.reset()
            self.last_ff = None
        if self._w3_clock is not None:   # 1006 W3 (실험 플래그 켜짐에만) — want 도 False(서기)로
            self._w3_clock.reset()

    def set_gait_enable(self, want: bool) -> None:
        """1006 W3: 명령이 원하는 gait_enable(시뮬 env._ge_cmd). 실제 ge 는 다음 관측 조립 때 래치 규칙으로 바뀐다."""
        if self._w3_clock is None:
            raise ValueError("W3 러너가 아님 — gait_enable 입력 불가")
        self._w3_clock.set_want(want)

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
        qvel_raw = vec(qvel_rad_s, 12, "qvel_rad_s")
        frames = {
            "base_ang_vel": vec(gyro_rad_s, 3, "gyro_rad_s"),
            "projected_gravity": vec(gravity_unit, 3, "gravity_unit"),
            "velocity_commands": vec(cmd, 3, "cmd"),
            "joint_pos": vec(qpos_rad, 12, "qpos_rad") - self.default_pose,
            "joint_vel": qvel_raw,
            "actions": self.last_action,
            "gait_clock": (self._h2_clock_frame(cmd) if self._h2_clock is not None
                           else self._w3_clock_frame() if self._w3_clock is not None
                           else gait_clock_stage4(self._tick, self.gait_freq)),
            "feet_contact": self._contact2(contact2),
        }
        # 1003 관측 필터 계약(체크포인트별, 학습 env.yaml 로 결정) — 첫 틱 = 현재값(학습 env 의 fresh 초기화와 같은 수열)
        for name, src, beta in (("act_obs", "actions", self.act_obs_ema_beta), ("jvel_obs", "joint_vel", self.jvel_obs_ema_beta)):
            if beta is None:
                continue
            x = np.asarray(frames[src], dtype=np.float32)
            prev = self._ema.get(name)
            self._ema[name] = x.copy() if prev is None else (prev + np.float32(beta) * (x - prev)).astype(np.float32)
            frames[src] = self._ema[name]
        if "gyro_ema" in self._hist:         # S8a·G13
            # raw+EMA(α0.1): 첫 틱은 현재값으로 시작, 이후 (1−α)·prev + α·x — 시뮬 mdp_hang._ema_state 와 같은 수열
            for name, src in (("gyro_ema", "base_ang_vel"), ("joint_vel_ema", "joint_vel")):
                x = np.asarray(qvel_raw if src == "joint_vel" else frames[src], dtype=np.float32)   # 1003: joint_vel_ema 는 원시 속도에서
                prev = self._ema.get(name)
                self._ema[name] = x.copy() if prev is None else (1.0 - EMA_ALPHA_S8A) * prev + EMA_ALPHA_S8A * x
                frames[name] = self._ema[name]
            self.last_jvel_ema = self._ema["joint_vel_ema"].copy()
            if self.jvel_ema_mode == "ema0":   # 1001: legacy 학습 계약(이 채널 = 0+잡음)의 평균값
                frames["joint_vel_ema"] = np.zeros(12, dtype=np.float32)
        if "action_hist16" in self._hist:    # G13: 같은 last_action 을 16틀로 따로 쌓는다
            frames["action_hist16"] = self.last_action
        for name, _dim in self.terms:
            self._hist[name].append(frames[name])
        obs = np.concatenate([self._hist[name].flat() for name, _ in self.terms])
        assert obs.shape == (self.obs_dim,) and obs.dtype == np.float32
        return obs

    def _h2_clock_frame(self, cmd) -> np.ndarray:
        """1006 HTURN2: 이번 틱 명령으로 cadence 위상을 한 틱 누적하고 클록 관측을 낸다(조립 전 tick = env episode_length_buf)."""
        c = PolicyRunner._vec(cmd, 3, "cmd")
        self._h2_cmd = (float(c[0]), float(c[1]), float(c[2]))
        self._h2_clock.update(self._tick, c[0])
        return self._h2_clock.clock_obs()

    def _w3_clock_frame(self) -> np.ndarray:
        """1006 W3: 래치·위상·램프를 한 틱 갱신(조립 전 tick = env episode_length_buf)하고 게이트된 클록 관측을 낸다."""
        self._w3_clock.update(self._tick)
        return self._w3_clock.clock_obs()

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
        target = self.default_pose + ACTION_SCALE * raw
        if self._h2_clock is not None:
            # 1006 HTURN2 회전 피드포워드 — 시뮬 pre_delay_hook 위치(target 직후, 지연·슬루·LPF 앞) = 러너 슬루 앞.
            vx, _vy, wz = self._h2_cmd
            rl, rr = _hturn2_mod().turn_ff_residual(self._h2_clock.u, self._h2_clock.f, vx, wz, self.hturn2)
            target[self._h2_iL] = target[self._h2_iL] + rl
            target[self._h2_iR] = target[self._h2_iR] + rr
            self.last_ff = (float(rl), float(rr))
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
    model_dir = DEFAULT_MODEL_DIR_STAGE4
    if not os.path.isdir(model_dir):
        # 젯슨엔 walk 기본 정책이 미배포일 수 있다 — 배포된 exported 중 최신(정렬 마지막)으로 대체 (0910 A1 배포에서 FAIL 난 경로)
        import glob
        cands = sorted(glob.glob(os.path.join(_REPO_ROOT, "logs/rsl_rl/biped12_stage4/*/exported/policy.pt")))
        if not cands:
            raise FileNotFoundError(f"배포된 exported/policy.pt 없음 (기본 {model_dir})")
        model_dir = os.path.dirname(cands[-1])
    r = PolicyRunnerStage4(model_dir, backend="auto")
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
    assert r.last_obs.shape == (r.obs_dim,), f"관측 {r.obs_dim} 아님: {r.last_obs.shape}"
    return (f"모델 로드+추론 OK (백엔드={r.backend}, {os.path.basename(os.path.dirname(model_dir))} 관측 {r.obs_dim}→출력 12 유한, "
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


def _selftest_g13() -> str:
    """(g) G13 512 조립: 앞 320 은 S8a판과 비트 동일, 뒤 192 = 직전 액션 16틀(index0=最古, 첫 push 백필 — CircularBuffer 규약)을
    독립 산술로 대조. 5틀 actions 블록의 마지막 5틀과도 일치해야 한다. 리셋 뒤 백필 재시작·차원 검출 512 확인."""
    outs = []

    def stub(obs):
        a = np.full(ACT_DIM, 0.01 * (len(outs) + 1), np.float32); a[0] = -a[0]; outs.append(a.copy())
        return a

    def stub0(obs):
        return np.zeros(ACT_DIM, np.float32)
    r320 = PolicyRunnerStage4(model_path=stub0, obs_dim=OBS_DIM_S8A)
    r512 = PolicyRunnerStage4(model_path=stub, obs_dim=OBS_DIM_G13)
    r320.reset(); r512.reset()
    N = 21
    acts_seen = []                    # 각 틱에 관측으로 들어간 last_action
    for k in range(N):
        ins = _mk_inputs(k)
        acts_seen.append(r512.last_action.copy())
        r512.step(*ins)
        r320.last_action = acts_seen[-1].copy(); r320.step(*ins)   # 같은 last_action 을 넣어 앞 320 비교
    o = r512.last_obs
    assert o.shape == (OBS_DIM_G13,) and o.dtype == np.float32
    assert np.array_equal(o[:OBS_DIM_S8A], r320.last_obs), "앞 320 이 S8a판과 다름"
    exp = np.stack(acts_seen[-G13_ACT_HIST:]).reshape(-1)
    assert np.array_equal(o[OBS_DIM_S8A:], exp), "16틀 액션 이력 불일치"
    off_a = _term_offsets()["actions"][0]
    assert np.array_equal(o[off_a: off_a + 60].reshape(HISTORY_LEN, 12), np.stack(acts_seen[-HISTORY_LEN:])), "5틀 actions 블록과 불일치"
    r512.reset(); r512.step(*_mk_inputs(3))
    assert np.array_equal(r512.last_obs[OBS_DIM_S8A:], np.zeros(G13_ACT_HIST * 12, np.float32)), "리셋 뒤 16틀 백필(0) 아님"

    def only(d):
        def f(obs):
            if obs.shape != (d,):
                raise ValueError("dim")
            return np.zeros(ACT_DIM, np.float32)
        return f
    assert _probe_obs_dim(only(OBS_DIM_G13)) == OBS_DIM_G13
    return "G13 512 조립 OK (앞 320 비트동일·16틀 액션 이력 最古→最新·5틀 블록 정합·리셋 백필·차원 검출 512)"


def _selftest_ema0() -> str:
    """(h) 1001 jvel_ema_mode='ema0': 320·512 에서 joint_vel_ema 블록[260:320] 만 0, 나머지는 v2 모드와 비트 동일.
    EMA 상태는 계속 계산(last_jvel_ema = v2 의 최신 EMA 프레임). 잘못된 모드·245 모델의 ema0 은 거부."""
    def stub(obs):
        return np.zeros(ACT_DIM, np.float32)
    off, dim = 260, 60
    out = []
    for od in (OBS_DIM_S8A, OBS_DIM_G13):
        rv = PolicyRunnerStage4(model_path=stub, obs_dim=od)
        rz = PolicyRunnerStage4(model_path=stub, obs_dim=od, jvel_ema_mode="ema0")
        rv.reset(); rz.reset()
        for k in range(9):
            ins = _mk_inputs(k)
            rv.step(*ins); rz.step(*ins)
            ov, oz = rv.last_obs, rz.last_obs
            assert np.array_equal(ov[:off], oz[:off]) and np.array_equal(ov[off + dim:], oz[off + dim:]), f"{od}: EMA 밖 블록이 다름"
            assert not np.any(oz[off:off + dim]), f"{od}: ema0 인데 EMA 블록이 0 이 아님"
            assert np.any(ov[off:off + dim]), f"{od}: v2 EMA 블록이 0 (시험 입력 결함)"
            assert np.array_equal(ov[off + dim - 12:off + dim], rz.last_jvel_ema), f"{od}: last_jvel_ema ≠ v2 최신 EMA 프레임"
        out.append(od)
    for od in (OBS_DIM_S8A, OBS_DIM_G13):     # 오프셋 260 을 항 목록에서 독립 산출
        o = 0
        for name, d in _TERMS_BY_DIM[od]:
            if name == "joint_vel_ema":
                assert (o, d * TERM_HISTORY.get(name, HISTORY_LEN)) == (off, dim), f"{od}: joint_vel_ema 오프셋 {o}"
            o += d * TERM_HISTORY.get(name, HISTORY_LEN)
        assert o == od
    for bad in (dict(jvel_ema_mode="zero"), dict(obs_dim=OBS_DIM_STAGE4, jvel_ema_mode="ema0")):
        try:
            PolicyRunnerStage4(model_path=stub, **bad)
        except ValueError:
            continue
        raise AssertionError(f"거부되지 않음: {bad}")
    return f"ema0 OK ({'/'.join(map(str, out))}: [260:320] 만 0·나머지 비트동일·EMA 진단값 유지·잘못된 모드/245 거부)"



def _selftest_w3() -> str:
    """(1006 W3 실험) 독립 float64 기준: 래치(0→1 즉시·1→0 반주기 경계)·위상·0.2 s 램프·클록 = [sin,cos]×ge_obs, 꺼짐 러너 무영향.
    시뮬 원본 소스와의 비트 동치는 rl_walking/deploy/selftest_w3_equiv.py(torch)."""
    stub = lambda obs: np.zeros(ACT_DIM, np.float32)  # noqa: E731
    w3m = _w3_mod()
    r = PolicyRunnerStage4(model_path=stub, obs_dim=320, gait_freq=1.1, w3=w3m.W3_DEFAULT)
    try:
        PolicyRunnerStage4(model_path=stub, w3=w3m.W3_DEFAULT, hturn2=object())
        raise AssertionError("hturn2+w3 동시 허용")
    except ValueError:
        pass
    r.reset()
    want_at = lambda n: (60 <= n < 113) or (150 <= n < 151) or (200 <= n < 330)   # noqa: E731 — 반주기 중간·경계·1틱 펄스
    ge, u, go = False, 0.0, 0.0
    flips = []
    q = DEFAULT_POSE_RAD.copy()
    for n in range(400):
        w = want_at(n)
        r.set_gait_enable(w)
        r.step((0, 0, 0), (0, 0, -1), (0, 0, 0), q, np.zeros(12), (1.0, 1.0))
        fresh = n <= 1
        un = u + 1.1 * 0.02
        cross = (u < 0.5 and un >= 0.5) or un >= 1.0
        on = w and not ge
        off = (not w) and ge and cross
        g2 = w if fresh else (True if on else (False if off else ge))
        if g2 != ge:
            flips.append((n, g2))
        u = 0.0 if (fresh or on or not g2) else un % 1.0
        go = (1.0 if g2 else 0.0) if fresh else min(max(go + (0.1 if g2 else -0.1), 0.0), 1.0)
        ge = g2
        c = r._w3_clock
        assert c.ge == ge, f"틱 {n} ge {c.ge} vs 기준 {ge}"
        assert abs(float(c.u) - u) < 1e-5 and abs(float(c.ge_obs) - go) < 1e-5, f"틱 {n} u/ge_obs {c.u},{c.ge_obs} vs {u},{go}"
        clk = r.last_obs[225 + 4 * 2: 225 + 5 * 2]
        assert abs(clk[0] - math.sin(2 * math.pi * u) * go) < 1e-5 and abs(clk[1] - math.cos(2 * math.pi * u) * go) < 1e-5
    assert [f for f, _ in flips] == [t for t, _ in c.events] and len(flips) >= 4, flips
    off_ticks = [t for t, g in flips if not g]
    # 끄기는 반주기 경계에서만: 끈 틱 직전 위상 + du 가 0.5 또는 1 을 넘는다(기준식 cross) — 위 일치로 이미 보장, 여기선 지연만 기록
    r2 = PolicyRunnerStage4(model_path=stub, obs_dim=320, gait_freq=1.1)
    r2.reset()
    r2.step((0, 0, 0), (0, 0, -1), (0, 0, 0), q, np.zeros(12), (1.0, 1.0))
    try:
        r2.set_gait_enable(True)
        raise AssertionError("W3 아닌 러너가 gait_enable 을 받음")
    except ValueError:
        pass
    return f"래치 전이 {flips} (끄기 틱 {off_ticks}) 기준식 일치·클록=[sin,cos]×ge_obs·램프 0.2 s·hturn2 동시 거부"


def _selftest_obsfilter() -> str:
    """(i) 1003 관측 필터 계약: β 없음 = 종전과 비트 동일(G12 무영향) / β 있음 = actions·joint_vel 블록이 EMA 수열,
    joint_vel_ema 는 원시 속도 EMA(α0.1), 나머지 블록 비트 동일 / reset() 이 필터 상태를 비움."""
    rng = np.random.default_rng(3)
    def stub(obs):
        return np.tanh(np.asarray(obs[:12], dtype=np.float32) * 0.7 + 0.1).astype(np.float32)
    ra = PolicyRunnerStage4(model_path=stub, obs_dim=OBS_DIM_S8A)
    rb = PolicyRunnerStage4(model_path=stub, obs_dim=OBS_DIM_S8A)
    rf = PolicyRunnerStage4(model_path=stub, obs_dim=OBS_DIM_S8A, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
    off, o = {}, 0
    for name, dim in rf.terms:
        off[name] = (o, dim); o += rf._hist[name]._buf.shape[0] * dim
    ya = yv = None; yema = None; acts = []
    for k in range(9):
        g, gr, c, q, v, ct = _mk_inputs(k)
        v = (v + rng.normal(0, 0.3, 12)).astype(np.float32)
        ea = ra.step(g, gr, c, q, v, ct); eb = rb.step(g, gr, c, q, v, ct)
        assert np.array_equal(ra.last_obs, rb.last_obs) and np.array_equal(ea, eb), "β 없음 러너가 결정적이지 않음"
        prev_act = rf.last_action.copy()
        rf.step(g, gr, c, q, v, ct)
        ya = prev_act.copy() if ya is None else (ya + np.float32(0.27) * (prev_act - ya)).astype(np.float32)
        yv = v.copy() if yv is None else (yv + np.float32(0.25) * (v - yv)).astype(np.float32)
        yema = v.copy() if yema is None else ((1.0 - EMA_ALPHA_S8A) * yema + EMA_ALPHA_S8A * v).astype(np.float32)
        for name, want in (("actions", ya), ("joint_vel", yv), ("joint_vel_ema", yema)):
            a0, d = off[name]; L = rf._hist[name]._buf.shape[0]
            got = rf.last_obs[a0 + (L - 1) * d: a0 + L * d]
            assert np.allclose(got, want, atol=1e-6), f"{name} 최신 프레임 불일치 (k={k})"
    # 필터 외 블록은 같은 입력이면 같다(무필터 러너와 액션 이력만 다름 → 같은 입력 1틱으로 비교)
    r1 = PolicyRunnerStage4(model_path=stub, obs_dim=OBS_DIM_S8A)
    r2 = PolicyRunnerStage4(model_path=stub, obs_dim=OBS_DIM_S8A, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
    g, gr, c, q, v, ct = _mk_inputs(0); r1.step(g, gr, c, q, v, ct); r2.step(g, gr, c, q, v, ct)
    assert np.array_equal(r1.last_obs, r2.last_obs), "첫 틱(EMA = 현재값)은 무필터와 같아야 함"
    rf.reset(); assert not rf._ema, "reset() 뒤 필터 상태가 남음"
    for bad in (0.0, 1.5, -0.1):
        try:
            PolicyRunnerStage4(model_path=stub, obs_dim=OBS_DIM_S8A, act_obs_ema_beta=bad); raise AssertionError(f"β={bad} 허용됨")
        except ValueError:
            pass
    return "관측 필터 OK (β 없음 결정적·종전 경로 / actions·joint_vel EMA 수열 일치 / joint_vel_ema 원시 속도 / 첫 틱 동일 / reset 비움 / 잘못된 β 거부)"

def _run_selftest() -> int:
    tests = [("조립 골든", _selftest_golden),
             ("이력 백필", _selftest_backfill),
             ("클록 공식", _selftest_clock),
             ("슬루/입력검증", _selftest_slew),
             ("모델 추론", _selftest_model),
             ("S8a 320 조립", _selftest_s8a),
             ("G13 512 조립", _selftest_g13),
             ("EMA0 계약 모드", _selftest_ema0),
             ("관측 필터 계약(1003)", _selftest_obsfilter),
             ("W3 클록 실험 플래그(1006)", _selftest_w3)]
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
