"""좌우 미러 대칭 증강 — rsl_rl symmetry_cfg용 data_augmentation_func.

설계 근거:
- Yu et al. 2018 "Learning Symmetric and Low-Energy Locomotion": 미러 손실로
  좌우 대칭 걸음을 학습 초기부터 형성 (Stage3 대칭 벌점 2회 실패의 대안).
- Mittal et al. 2024 (rsl_rl symmetry 구현 논문): [원본; 미러] 스택 규약.

rsl_rl(이 설치본: _isaac_sim/.../site-packages/rsl_rl) 호출 규약 (algorithms/ppo.py):
- obs는 tensordict.TensorDict (storage/rollout_storage.py가 TensorDict로 저장,
  키 = 관측 그룹명 "policy"/"critic"), actions는 (B, 12) 텐서.
- data_augmentation_func(obs=..., actions=..., env=...) → (증강obs, 증강actions).
  반환 배치 = [원본; 미러] (원본이 첫 절반 — ppo.py가 mean_actions[:B]를 원본 취급).
  obs 또는 actions가 None이면 해당 반환도 None.

레이아웃 (stage4_env_cfg.Stage4ObservationsCfg와 1:1 — 항 순서/이력 변경 시 여기도 갱신!):
- 단일 프레임 49 = ang_vel(3)+gravity(3)+cmd(3)+qpos(12)+qvel(12)+act(12)+clock(2)+contact(2)
- 이력 T=5, flatten_history_dim=True: isaaclab observation_manager.compute_group()이
  항별 CircularBuffer.buffer (N, T, D; T축 index 0=최고(最古))를 (N, T·D)로 reshape 후
  항 순서대로 concat → 항별 [프레임0(D), 프레임1(D), ... 프레임4(D)] 블록 구조.
  따라서 미러는 항 블록 안에서 프레임별(D 단위)로 동일 변환을 반복 적용한다.
- "policy" 그룹 = 245 (49×5) / "critic" 그룹 = 15 (base_lin_vel 3×5).
  크리틱 네트 입력 = policy 245 + critic 15 = 260 (actor_critic.get_critic_obs concat).

미러 변환 (y→−y 반사; 부호 근거 log_picture/05 실측 프로브 — 해부학적 동일 동작에
좌/우 명령 부호가 반대인 관절 = hip_f, hip_a, hip_r, knee, ankle_f. ankle_r만 동부호):
- base_ang_vel (wx,wy,wz) → (−wx, wy, −wz)
- projected_gravity (gx,gy,gz) → (gx, −gy, gz)
- velocity_commands (vx,vy,wz) → (vx, −vy, −wz)
- 관절량(qpos/qvel/action): [L,R] 인터리브 6쌍(hip_f,hip_a,hip_r,knee,ankle_f,ankle_r —
  log_picture/03 실측 순서)에서 쌍 내 L↔R 스왑 후 부호 s 곱.
  s=−1: hip_f, hip_a, hip_r, knee, ankle_f 쌍 / s=+1: ankle_r 쌍만.
- gait_clock (s,c) → (−s,−c)  [φ→φ+π: 좌우 발 역할 교대]
- feet_contact (L,R) → (R,L)
- base_lin_vel (vx,vy,vz) → (vx, −vy, vz)

전 변환이 '부호 있는 순열'이므로 perm/sign 벡터 한 쌍으로 O(1) 적용. 이중 미러 = 항등
(단위테스트로 검증). 시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
from __future__ import annotations

import torch

try:  # tensordict는 rsl_rl 학습 환경에만 존재 — CPU 단위테스트에서는 dict로 대체 가능
    from tensordict import TensorDict
except ImportError:  # pragma: no cover
    TensorDict = None

# ── 레이아웃 상수 (stage4_env_cfg와 동기) ──────────────────────────────────
NUM_JOINTS = 12
HISTORY_LEN = 5
FRAME_POLICY_DIM = 49          # 45(기존) + clock 2 + contact 2
POLICY_OBS_DIM = 245           # 49 × 5
CRITIC_GROUP_DIM = 15          # base_lin_vel 3 × 5 ("critic" 그룹 단독 차원)
CRITIC_OBS_DIM = 260           # 크리틱 네트 총 입력 = 245 + 15

# 쌍 순서: hip_f, hip_a, hip_r, knee, ankle_f, ankle_r — 쌍 내 [L, R]
_PAIR_SIGN = (-1.0, -1.0, -1.0, -1.0, -1.0, +1.0)  # ankle_r만 +1 (log_picture/05)


def _joint_perm_sign() -> tuple[list[int], list[float]]:
    """12관절 미러 = 쌍 내 L↔R 스왑 + 쌍별 부호."""
    perm: list[int] = []
    sign: list[float] = []
    for k, s in enumerate(_PAIR_SIGN):
        perm += [2 * k + 1, 2 * k]  # L 슬롯 ← R 값, R 슬롯 ← L 값
        sign += [s, s]
    return perm, sign


_JOINT_PERM, _JOINT_SIGN = _joint_perm_sign()

# 항별 (이름, 프레임 차원 D, 프레임 내 perm, sign) — 관측 그룹 선언 순서 그대로
_POLICY_TERMS = (
    ("base_ang_vel", 3, (0, 1, 2), (-1.0, 1.0, -1.0)),
    ("projected_gravity", 3, (0, 1, 2), (1.0, -1.0, 1.0)),
    ("velocity_commands", 3, (0, 1, 2), (1.0, -1.0, -1.0)),
    ("joint_pos", 12, tuple(_JOINT_PERM), tuple(_JOINT_SIGN)),
    ("joint_vel", 12, tuple(_JOINT_PERM), tuple(_JOINT_SIGN)),
    ("actions", 12, tuple(_JOINT_PERM), tuple(_JOINT_SIGN)),
    ("gait_clock", 2, (0, 1), (-1.0, -1.0)),
    ("feet_contact", 2, (1, 0), (1.0, 1.0)),
)
_CRITIC_TERMS = (("base_lin_vel", 3, (0, 1, 2), (1.0, -1.0, 1.0)),)


def build_perm_sign(terms, history: int) -> tuple[torch.Tensor, torch.Tensor]:
    """항 블록×이력 평탄화 레이아웃의 전역 perm/sign 벡터 생성 (CPU, 테스트에서도 사용)."""
    perm: list[int] = []
    sign: list[float] = []
    offset = 0
    for _name, dim, p, s in terms:
        assert len(p) == dim and len(s) == dim
        for t in range(history):
            base = offset + t * dim
            perm += [base + i for i in p]
            sign += list(s)
        offset += history * dim
    return torch.tensor(perm, dtype=torch.long), torch.tensor(sign, dtype=torch.float32)


# 그룹키 → (terms, 이력, 기대 차원). rollout 미니배치 TensorDict의 키와 일치해야 한다.
_GROUP_SPECS = {
    "policy": (_POLICY_TERMS, HISTORY_LEN, POLICY_OBS_DIM),
    "critic": (_CRITIC_TERMS, HISTORY_LEN, CRITIC_GROUP_DIM),
    "actions": (
        (("actions", NUM_JOINTS, tuple(_JOINT_PERM), tuple(_JOINT_SIGN)),),
        1,
        NUM_JOINTS,
    ),
}

_cache: dict = {}  # (그룹키, device) → (perm, sign)


def _perm_sign_for(kind: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (kind, device)
    if key not in _cache:
        terms, history, dim = _GROUP_SPECS[kind]
        perm, sign = build_perm_sign(terms, history)
        assert perm.numel() == dim, f"{kind}: perm {perm.numel()} != 기대 {dim}"
        _cache[key] = (perm.to(device), sign.to(device))
    return _cache[key]


def _mirror(x: torch.Tensor, kind: str) -> torch.Tensor:
    """부호 있는 순열 적용 — 마지막 차원에 대해 (배치/시간 등 앞 차원 불문)."""
    terms, history, dim = _GROUP_SPECS[kind]
    if x.shape[-1] != dim:
        raise ValueError(
            f"'{kind}' 미러: 마지막 차원 {x.shape[-1]} != 기대 {dim} — "
            "관측 레이아웃이 바뀌었으면 symmetry.py 상수를 갱신할 것"
        )
    perm, sign = _perm_sign_for(kind, x.device)
    return x.index_select(-1, perm) * sign.to(x.dtype)


def _obs_kind(key: str) -> str:
    if key not in ("policy", "critic"):
        raise ValueError(
            f"미러 규칙이 없는 관측 그룹 '{key}' — symmetry.py._GROUP_SPECS에 추가할 것"
        )
    return key


def mirror_biped12(env=None, obs=None, actions=None):
    """rsl_rl data_augmentation_func — [원본; 미러] 스택 반환 (원본이 첫 절반).

    Args:
        env: rsl_rl이 넘기는 환경 핸들 (미사용 — 레이아웃은 상수로 고정).
        obs: TensorDict/dict(그룹키→(B, D) 텐서) 또는 (B, 245) 정책 텐서. None 가능.
        actions: (B, 12) 액션 텐서. None 가능.

    Returns:
        (증강 obs, 증강 actions) — 각각 입력이 None이면 None, 아니면 배치 2B.
    """
    obs_out = None
    actions_out = None

    if obs is not None:
        if TensorDict is not None and isinstance(obs, TensorDict):
            mirrored = obs.clone()
            for key in list(obs.keys()):
                mirrored[key] = _mirror(obs[key], _obs_kind(key))
            obs_out = torch.cat([obs, mirrored], dim=0)
        elif isinstance(obs, dict):  # 단위테스트/타 러너 호환
            obs_out = {
                key: torch.cat([val, _mirror(val, _obs_kind(key))], dim=0)
                for key, val in obs.items()
            }
        else:  # 평탄 텐서 — 정책 관측으로 해석
            obs_out = torch.cat([obs, _mirror(obs, "policy")], dim=0)

    if actions is not None:
        actions_out = torch.cat([actions, _mirror(actions, "actions")], dim=0)

    return obs_out, actions_out
