"""RL 보행 정책 실행기 (실물/단독 PC용) — isaaclab 의존 없음, numpy + (onnxruntime | torch) 만.

체크포인트: logs/rsl_rl/biped12_flat/2026-07-25_21-59-19/exported/{policy.onnx, policy.pt}
정책: 45차원 관측 → 12차원 raw 액션 MLP (50Hz 정책틱).

관측 조립 (params/env.yaml 실측 순서, concatenate_terms=true — 인덱스 고정):
  [0:3]   base_ang_vel        베이스 프레임 각속도 [rad/s] — iAHRS 자이로 (축 정렬 전제)
  [3:6]   projected_gravity   중력 '단위벡터'의 베이스 프레임 투영 — 직립 (0,0,-1).
                              가속도계 원시값 아님! IMU 자세 쿼터니언에서 R^T@(0,0,-1)로 계산
  [6:9]   velocity_commands   (vx[m/s], vy[m/s], wz[rad/s]) 베이스 프레임 목표속도.
                              베이스 프레임 = Z-up, +X 전방 (RL 자산 base 링크 기준)
  [9:21]  joint_pos_rel       엔코더 절대각 − DEFAULT_POSE_RAD [rad], JOINT_ORDER 순 12개
  [21:33] joint_vel_rel       관절 각속도 [rad/s] (default_joint_vel=0이라 그냥 각속도)
  [33:45] last_action         직전 틱의 '정책 raw 출력' (스케일/오프셋/슬루 전부 적용 前 값).
                              리셋/시작 첫 틱은 0 벡터

학습 시 관측 노이즈(Unoise)는 corruption 전용 — 실행기는 생측정값을 그대로 넣는다.

액션 파이프라인 (rl_walking/actions.py SlewJointPositionAction과 동일 수식):
  raw a_t → target = DEFAULT_POSE_RAD + 0.25 * a_t  (clip 없음)
          → emitted = prev + clamp(target − prev, ±4°/틱)  (슬루)
          → emitted를 위치명령으로 PD 추종 (kp150/kd5, AK70 25Nm / 발목롤 AK45-36 24Nm)
  prev(슬루 기준점)는 리셋/시작 시 '현재 측정 관절각'으로 초기화 (actions.py _needs_init).
  학습 DR의 액션지연(max_delay_steps=1)은 실물 자연 루프지연(~15-19ms+서보 ≈1틱@50Hz)이
  이미 대응 — 이 실행기에는 인위적 지연 버퍼를 넣지 않는다.

슬루 이중 적용에 대해 (실물 Stage8 노드에도 NODE_SLEW_DEG_PER_TICK=4°가 있음):
  본 실행기의 출력은 이미 틱당 |Δ|≤4°로 제한되어 있으므로, 같은 50Hz 틱에서 같은
  기준점(현재 관절각)으로 시작하는 노드 슬루는 항등 통과가 된다 — 클램프는 한도 이내
  입력에 대해 항등이므로 이중 적용은 무해하고, 시작점만 같으면 두 경로의 출력이 정확히
  일치한다. 러너 내부 슬루를 유지하는 이유: (1) 학습 파이프라인과 1:1 수식 정합,
  (2) 노드 없이 시뮬/로그 재생 시에도 동일 거동 보장.

시뮬 전용 — 이 모듈은 CAN/모터에 아무것도 보내지 않는다. 실물 송신 연결은 별도 결정 사항.

사용:
    from rl_walking.deploy.policy_runner import PolicyRunner
    r = PolicyRunner()                      # exported/ 자동 로드 (onnx 우선, 없으면 torch)
    r.reset()
    q_cmd = r.step(gyro, gravity, (0,0,0), qpos, qvel)   # 50Hz 루프에서 호출

자가검증 (GPU 불필요):
    python3 policy_runner.py --selftest
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 확정 상수 (학습 run 2026-07-25_21-59-19 params/env.yaml 실측값과 동일)
# ---------------------------------------------------------------------------

#: Isaac Lab BFS 순서 (좌우 인터리브) — 하드웨어맵 motor_id(좌1-6/우7-12)와 다름.
#: 실물 연결 시 반드시 '관절명' 기준으로 매핑할 것 (docs/RL보행_이식가이드.md §2).
JOINT_ORDER = [
    "left_hip_f_joint", "right_hip_f_joint",
    "left_hip_a_joint", "right_hip_a_joint",
    "left_hip_r_joint", "right_hip_r_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_f_joint", "right_ankle_f_joint",
    "left_ankle_r_joint", "right_ankle_r_joint",
]

#: 학습 기본자세 [rad], JOINT_ORDER 순 (biped12_cfg.BIPED12_DEFAULT_JOINT_POS)
DEFAULT_POSE_RAD = np.array([
    0.0, 0.0,
    0.0, 0.0,
    0.0, 0.0,
    -0.10471975511965977, 0.10471975511965977,
    0.0244, -0.0244,
    0.0, 0.0,
], dtype=np.float32)

ACTION_SCALE = 0.25                      # target = default + 0.25 * raw
SLEW_RAD = 0.06981317007977318           # 4°/정책틱 = 실물 NODE_SLEW_DEG_PER_TICK
RATE_HZ = 50                             # 정책틱 주기 (step()을 이 주기로 호출)
OBS_DIM = 45
ACT_DIM = 12

#: 기본 모델 경로 (exported/ 디렉토리 — policy.onnx 우선, 없으면 policy.pt)
DEFAULT_MODEL_DIR = (
    "/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_flat/"
    "2026-07-25_21-59-19/exported"
)


# ---------------------------------------------------------------------------
# 추론 백엔드 — onnxruntime 가능 시 ONNX, 아니면 torch.jit (자동 선택)
# ---------------------------------------------------------------------------

class _OnnxBackend:
    """ONNX Runtime CPU 추론. 입력명은 세션에서 조회 (하드코딩 금지)."""

    name = "onnx"

    def __init__(self, path: str):
        import onnxruntime as ort  # 지연 임포트 — 없으면 torch로 폴백
        self._sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = self._sess.run(None, {self._input: obs[None, :]})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)


class _TorchBackend:
    """torch.jit(policy.pt) CPU 추론."""

    name = "torch"

    def __init__(self, path: str):
        import torch  # 지연 임포트
        self._torch = torch
        self._net = torch.jit.load(path, map_location="cpu")
        self._net.eval()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            a = self._net(self._torch.from_numpy(obs[None, :]))[0].numpy()
        return np.asarray(a, dtype=np.float32).reshape(-1)


def _load_backend(model_path, backend: str = "auto"):
    """(백엔드명, 호출객체) 반환. model_path가 callable이면 그대로 사용 (selftest용 스텁)."""
    if callable(model_path):
        return "stub", model_path
    p = str(model_path)
    if os.path.isdir(p):
        onnx_p = os.path.join(p, "policy.onnx")
        pt_p = os.path.join(p, "policy.pt")
    elif p.endswith(".onnx"):
        onnx_p, pt_p = p, None
    elif p.endswith(".pt"):
        onnx_p, pt_p = None, p
    else:
        raise FileNotFoundError(f"모델 경로 인식 불가 (.onnx/.pt/디렉토리만): {p}")

    errors = []
    if backend in ("auto", "onnx"):
        if onnx_p and os.path.isfile(onnx_p):
            try:
                return "onnx", _OnnxBackend(onnx_p)
            except ImportError as e:
                errors.append(f"onnx 불가({e})")
                if backend == "onnx":
                    raise
        elif backend == "onnx":
            raise FileNotFoundError(f"policy.onnx 없음: {onnx_p}")
    if backend in ("auto", "torch"):
        if pt_p and os.path.isfile(pt_p):
            try:
                return "torch", _TorchBackend(pt_p)
            except ImportError as e:
                errors.append(f"torch 불가({e})")
                if backend == "torch":
                    raise
        elif backend == "torch":
            raise FileNotFoundError(f"policy.pt 없음: {pt_p}")
    raise RuntimeError(
        "사용 가능한 추론 백엔드 없음 — onnxruntime 또는 torch를 설치할 것. "
        f"시도 내역: {errors or '파일 없음'} (경로: {p})"
    )


# ---------------------------------------------------------------------------
# 실행기 본체
# ---------------------------------------------------------------------------

class PolicyRunner:
    """관측 조립 → 정책 추론 → 아핀 + 슬루 → 목표관절각 12개 [rad] 반환.

    50Hz(RATE_HZ)로 step()을 호출하는 것이 호출자 책임이다 — 내부에 타이머 없음.
    last_action(직전 raw 출력)은 내부에서 관리하며 reset() 시 0으로 초기화된다.
    슬루 기준점은 리셋 후 '첫 step()의 qpos_rad'로 초기화된다 (actions.py _needs_init
    와 동일 — default_pose가 아님에 주의).

    반환값은 슬루 적용 후 목표각이므로 그대로 위치명령(PD kp150/kd5)으로 쓴다.
    실물 Stage8 노드에 동일한 4°/틱 슬루가 또 있어도 무해 — 모듈 docstring 참조.
    """

    def __init__(self, model_path=DEFAULT_MODEL_DIR, backend: str = "auto"):
        """model_path: exported/ 디렉토리 또는 .onnx/.pt 파일 경로.
        backend: 'auto'(onnx 우선→torch 폴백) | 'onnx' | 'torch'.
        """
        self.backend, self._net = _load_backend(model_path, backend)
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._prev_emitted = None        # 슬루 기준점 — 첫 step()에서 qpos로 초기화

    def reset(self) -> None:
        """에피소드/세션 초기화 — last_action=0, 슬루 기준점 재초기화 예약."""
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._prev_emitted = None

    # ---------- 관측 ----------

    @staticmethod
    def _vec(x, n: int, name: str) -> np.ndarray:
        v = np.asarray(x, dtype=np.float32).reshape(-1)
        if v.shape != (n,):
            raise ValueError(f"{name}: 길이 {n} 필요, 받은 shape {np.shape(x)}")
        return v

    @staticmethod
    def assemble_obs(gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s,
                     last_action) -> np.ndarray:
        """45차원 관측 벡터 조립 (obs_layout 인덱스와 1:1 — 골든테스트 대상).

        gyro_rad_s(3): 베이스 프레임 각속도 [rad/s]
        gravity_unit(3): 중력 단위벡터의 베이스 프레임 투영 (직립 (0,0,-1))
        cmd(3): (vx, vy, wz) 목표속도 [m/s, m/s, rad/s]
        qpos_rad(12): 관절 절대각 [rad], JOINT_ORDER 순
        qvel_rad_s(12): 관절 각속도 [rad/s], JOINT_ORDER 순
        last_action(12): 직전 틱 raw 정책출력 (무차원)
        """
        obs = np.empty(OBS_DIM, dtype=np.float32)
        obs[0:3] = PolicyRunner._vec(gyro_rad_s, 3, "gyro_rad_s")
        obs[3:6] = PolicyRunner._vec(gravity_unit, 3, "gravity_unit")
        obs[6:9] = PolicyRunner._vec(cmd, 3, "cmd")
        obs[9:21] = PolicyRunner._vec(qpos_rad, 12, "qpos_rad") - DEFAULT_POSE_RAD
        obs[21:33] = PolicyRunner._vec(qvel_rad_s, 12, "qvel_rad_s")
        obs[33:45] = PolicyRunner._vec(last_action, 12, "last_action")
        return obs

    # ---------- 스텝 ----------

    def step(self, gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s) -> np.ndarray:
        """1 정책틱 실행 → 슬루 적용된 목표관절각 12개 [rad] (JOINT_ORDER 순).

        관측의 last_action 자리에는 '이전 틱' raw 출력이 들어가고 (첫 틱은 0),
        이번 틱 raw 출력은 반환 전에 내부 저장된다 — 슬루 후 목표각을 되먹이면
        학습 분포와 어긋나므로 절대 반환값을 last_action으로 쓰지 말 것.
        """
        qpos = self._vec(qpos_rad, 12, "qpos_rad")
        obs = self.assemble_obs(
            gyro_rad_s, gravity_unit, cmd, qpos, qvel_rad_s, self.last_action)
        raw = np.asarray(self._net(obs), dtype=np.float32).reshape(-1)
        if raw.shape != (ACT_DIM,):
            raise RuntimeError(f"정책 출력 shape 이상: {raw.shape} (기대 ({ACT_DIM},))")
        # last_action 관측 = 아핀/슬루 '전' raw (action_manager 저장 시점과 동일)
        self.last_action = raw.copy()
        # (1) 아핀: clip 없음 — 안전은 슬루 + 노드 SafetyFilter + 관절리밋 담당
        target = DEFAULT_POSE_RAD + ACTION_SCALE * raw
        # (2) 슬루 기준점 초기화: 리셋 후 첫 틱은 '현재 측정 관절각'에서 시작
        if self._prev_emitted is None:
            self._prev_emitted = qpos.copy()
        # (3) 슬루 클램프 ±4°/틱
        emitted = self._prev_emitted + np.clip(
            target - self._prev_emitted, -SLEW_RAD, SLEW_RAD)
        self._prev_emitted = emitted
        return emitted.copy()


# ---------------------------------------------------------------------------
# 자가검증 (--selftest) — GPU 불필요, CPU만
# ---------------------------------------------------------------------------

def _selftest_obs() -> str:
    """(a) 관측 조립 골든테스트: 알려진 입력 → 기대 벡터 (인덱스 단위 대조)."""
    gyro = np.array([0.1, -0.2, 0.3], np.float32)
    grav = np.array([0.02, -0.05, -0.9985], np.float32)
    cmd = np.array([0.5, 0.0, -0.1], np.float32)
    qpos = DEFAULT_POSE_RAD + 0.01 * np.arange(12, dtype=np.float32)
    qvel = -0.02 * np.arange(12, dtype=np.float32)
    last_a = 0.1 * (np.arange(12, dtype=np.float32) - 6.0)

    obs = PolicyRunner.assemble_obs(gyro, grav, cmd, qpos, qvel, last_a)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32, obs.shape

    # 기대 벡터를 concatenate가 아닌 '인덱스 직접 기입'으로 독립 구성 (레이아웃 검증)
    exp = np.zeros(OBS_DIM, np.float32)
    exp[0:3] = gyro
    exp[3:6] = grav
    exp[6:9] = cmd
    exp[9:21] = qpos - DEFAULT_POSE_RAD      # joint_pos_rel = 절대각 − 기본자세
    exp[21:33] = qvel                        # default_joint_vel=0 → 그냥 각속도
    exp[33:45] = last_a
    assert np.array_equal(obs, exp), f"골든 불일치 max|Δ|={np.abs(obs - exp).max()}"
    # 무릎(-0.1047/+0.1047)·발목F(0.0244/-0.0244) 오프셋이 실제로 빠졌는지 개별 확인
    assert abs(obs[9 + 6] - (qpos[6] - (-0.10471975511965977))) < 1e-7
    assert abs(obs[9 + 9] - (qpos[9] - (-0.0244))) < 1e-7
    return "obs 골든테스트 OK (45차원, 인덱스/오프셋 일치)"


def _selftest_slew() -> str:
    """(b) 슬루 클램프 + last_action 관리 + 리셋 기준점 검증 (모델 없이 스텁으로)."""
    seen_obs = []

    def stub(obs):
        seen_obs.append(obs.copy())
        return np.full(ACT_DIM, 1.0, np.float32)   # target = default + 0.25 → 슬루 포화

    r = PolicyRunner(model_path=stub)
    r.reset()
    zeros3 = np.zeros(3)
    grav = np.array([0.0, 0.0, -1.0])
    qpos0 = DEFAULT_POSE_RAD.copy()

    # 1틱: 기준점 = qpos0(현재 관절각), Δ=0.25 → 정확히 +SLEW_RAD만 이동
    out1 = r.step(zeros3, grav, zeros3, qpos0, np.zeros(12))
    assert np.allclose(out1, qpos0 + SLEW_RAD, atol=1e-7), "1틱 슬루 포화 불일치"
    assert np.array_equal(seen_obs[0][33:45], np.zeros(12)), "첫 틱 last_action != 0"

    # 2틱: 기준점은 emitted(측정각 아님) → +2*SLEW_RAD, obs의 last_action = 이전 raw
    out2 = r.step(zeros3, grav, zeros3, qpos0, np.zeros(12))
    assert np.allclose(out2, qpos0 + 2 * SLEW_RAD, atol=1e-7), "2틱 슬루 누적 불일치"
    assert np.array_equal(seen_obs[1][33:45], np.full(12, 1.0)), \
        "last_action이 raw(슬루 前)가 아님"
    # 허용오차 1e-6: float32 ulp(≈7.5e-9 @0.07rad) 누적 대비 — 물리적으로 무의미한 크기
    assert np.all(np.abs(out2 - out1) <= SLEW_RAD + 1e-6), "틱당 변화량 초과"

    # 음방향 포화 + 한도 이내 항등 (이중 슬루 무해성의 근거)
    r2 = PolicyRunner(model_path=lambda o: np.full(ACT_DIM, -1.0, np.float32))
    outn = r2.step(zeros3, grav, zeros3, qpos0, np.zeros(12))
    assert np.allclose(outn, qpos0 - SLEW_RAD, atol=1e-7), "음방향 슬루 불일치"
    small = 0.1  # 0.25*0.1=0.025rad < SLEW → 클램프 항등 통과
    r3 = PolicyRunner(model_path=lambda o: np.full(ACT_DIM, small, np.float32))
    outs = r3.step(zeros3, grav, zeros3, qpos0, np.zeros(12))
    assert np.allclose(outs, DEFAULT_POSE_RAD + ACTION_SCALE * small, atol=1e-7), \
        "한도 이내 입력이 항등 통과하지 않음"

    # 리셋 후 기준점이 '새 측정각'으로 재초기화되는지 (default_pose 아님)
    r.reset()
    assert np.array_equal(r.last_action, np.zeros(12)), "reset 후 last_action != 0"
    qposX = DEFAULT_POSE_RAD + 0.5      # 기본자세에서 먼 자세
    outr = r.step(zeros3, grav, zeros3, qposX, np.zeros(12))
    assert np.allclose(outr, qposX - SLEW_RAD, atol=1e-6), \
        "리셋 후 슬루 기준점이 현재 측정각이 아님"
    return "슬루 클램프/last_action/리셋 기준점 OK (±4°/틱 정확)"


def _selftest_model() -> str:
    """(c) 실제 체크포인트 로드 + 1회 추론 shape/유한성 검증."""
    r = PolicyRunner(DEFAULT_MODEL_DIR, backend="auto")
    r.reset()
    out = r.step(
        gyro_rad_s=np.zeros(3),
        gravity_unit=np.array([0.0, 0.0, -1.0]),
        cmd=np.zeros(3),                     # 시작은 (0,0,0) 스탠딩부터
        qpos_rad=DEFAULT_POSE_RAD.copy(),
        qvel_rad_s=np.zeros(12),
    )
    assert out.shape == (ACT_DIM,) and out.dtype == np.float32, out.shape
    assert np.all(np.isfinite(out)), "추론 출력에 NaN/Inf"
    assert np.all(np.abs(out - DEFAULT_POSE_RAD) <= SLEW_RAD + 1e-6), \
        "첫 틱 출력이 슬루 한도(기준점=qpos)를 벗어남"
    assert r.last_action.shape == (ACT_DIM,), "last_action 내부 저장 shape 이상"
    return (f"모델 로드+추론 OK (백엔드={r.backend}, 출력 12차원 유한, "
            f"첫 틱 |Δ|max={np.abs(out - DEFAULT_POSE_RAD).max():.4f} rad ≤ 슬루)")


def _run_selftest() -> int:
    tests = [("obs 골든", _selftest_obs),
             ("슬루", _selftest_slew),
             ("모델 추론", _selftest_model)]
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
    ap = argparse.ArgumentParser(description="RL 보행 정책 실행기 (시뮬 전용)")
    ap.add_argument("--selftest", action="store_true", help="자가검증 실행 (GPU 불필요)")
    args = ap.parse_args()
    if args.selftest:
        return _run_selftest()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
