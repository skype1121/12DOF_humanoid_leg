"""symmetry.mirror_biped12 CPU 단위테스트 — 이중미러=항등 + 프레임 의미 검증."""
import sys

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")

import torch

from rl_walking.symmetry import (
    CRITIC_GROUP_DIM,
    CRITIC_OBS_DIM,
    FRAME_POLICY_DIM,
    HISTORY_LEN,
    NUM_JOINTS,
    POLICY_OBS_DIM,
    _CRITIC_TERMS,
    _POLICY_TERMS,
    build_perm_sign,
    mirror_biped12,
)

torch.manual_seed(0)
B = 7
PASS = []


def check(name, cond):
    assert cond, f"FAIL: {name}"
    PASS.append(name)
    print(f"  [PASS] {name}")


print("== 0) 레이아웃 상수 ==")
check("정책 245 = 49*5", POLICY_OBS_DIM == FRAME_POLICY_DIM * HISTORY_LEN == 245)
check("크리틱그룹 15 / 크리틱입력 260", CRITIC_GROUP_DIM == 15 and CRITIC_OBS_DIM == 260)
p_perm, p_sign = build_perm_sign(_POLICY_TERMS, HISTORY_LEN)
c_perm, c_sign = build_perm_sign(_CRITIC_TERMS, HISTORY_LEN)
check("perm은 순열(전단사)", sorted(p_perm.tolist()) == list(range(245))
      and sorted(c_perm.tolist()) == list(range(15)))
check("sign은 ±1만", set(p_sign.abs().tolist()) == {1.0} and set(c_sign.abs().tolist()) == {1.0})

print("== 1) dict(그룹키) 배치: 스택 규약 + 이중미러=항등 ==")
obs = {"policy": torch.randn(B, POLICY_OBS_DIM), "critic": torch.randn(B, CRITIC_GROUP_DIM)}
act = torch.randn(B, NUM_JOINTS)
obs2, act2 = mirror_biped12(env=None, obs={k: v.clone() for k, v in obs.items()}, actions=act.clone())
check("배치 2B", obs2["policy"].shape == (2 * B, 245) and obs2["critic"].shape == (2 * B, 15)
      and act2.shape == (2 * B, 12))
check("첫 절반 = 원본", torch.equal(obs2["policy"][:B], obs["policy"])
      and torch.equal(obs2["critic"][:B], obs["critic"]) and torch.equal(act2[:B], act))
# 미러 절반을 다시 미러 → 원본 복원 (이중미러 = 항등)
obs_m = {k: v[B:] for k, v in obs2.items()}
obs3, act3 = mirror_biped12(env=None, obs=obs_m, actions=act2[B:])
check("이중미러=항등 (policy)", torch.allclose(obs3["policy"][B:], obs["policy"], atol=1e-6))
check("이중미러=항등 (critic)", torch.allclose(obs3["critic"][B:], obs["critic"], atol=1e-6))
check("이중미러=항등 (actions)", torch.allclose(act3[B:], act, atol=1e-6))
check("미러는 원본과 다름 (자명해 방지)", not torch.equal(obs2["policy"][B:], obs["policy"]))

print("== 2) obs=None / actions=None 경로 (rsl_rl 미러손실 호출 규약) ==")
o, a = mirror_biped12(env=None, obs={k: v.clone() for k, v in obs.items()}, actions=None)
check("actions=None → None", a is None and o is not None)
o, a = mirror_biped12(env=None, obs=None, actions=act.clone())
check("obs=None → None", o is None and a.shape == (2 * B, 12))

print("== 3) 단일 프레임(45+4=49차원) 의미 검증 — 이력 T=1 perm ==")
f_perm, f_sign = build_perm_sign(_POLICY_TERMS, 1)
check("단일 프레임 차원 49", f_perm.numel() == 49)

def mirror_frame(x):
    return x[..., f_perm] * f_sign

# 항 오프셋 (T=1): ang_vel 0, gravity 3, cmd 6, qpos 9, qvel 21, act 33, clock 45, contact 47
fr = torch.zeros(49)
fr[0:3] = torch.tensor([0.1, 0.2, 0.3])        # ang_vel (wx,wy,wz)
fr[3:6] = torch.tensor([0.04, 0.05, -0.98])    # gravity
fr[6:9] = torch.tensor([0.5, 0.1, -0.4])       # cmd (vx,vy,wz)
fr[9] = 0.30                                   # qpos left_hip_f (쌍0 L)
fr[16] = 0.40                                  # qpos right_knee (쌍3 R → idx 9+7)
fr[19] = 0.20                                  # qpos left_ankle_r (쌍5 L → idx 9+10)
fr[45:47] = torch.tensor([0.6, -0.8])          # clock (sin, cos)
fr[47:49] = torch.tensor([1.0, 0.0])           # contact (L, R)
m = mirror_frame(fr)
check("ang_vel (wx,wy,wz)→(-wx,wy,-wz)", torch.allclose(m[0:3], torch.tensor([-0.1, 0.2, -0.3])))
check("gravity (gx,gy,gz)→(gx,-gy,gz)", torch.allclose(m[3:6], torch.tensor([0.04, -0.05, -0.98])))
check("cmd (vx,vy,wz)→(vx,-vy,-wz)", torch.allclose(m[6:9], torch.tensor([0.5, -0.1, 0.4])))
def near(a, b):
    return abs(a - b) < 1e-6

check("qpos left_hip_f=+0.3 → right_hip_f=-0.3, left=0",
      near(m[10].item(), -0.30) and near(m[9].item(), 0.0))
check("qpos right_knee=+0.4 → left_knee=-0.4", near(m[15].item(), -0.40) and near(m[16].item(), 0.0))
check("qpos left_ankle_r=+0.2 → right_ankle_r=+0.2 (동부호, log_picture/05)",
      near(m[20].item(), +0.20) and near(m[19].item(), 0.0))
check("clock (s,c)→(-s,-c)", torch.allclose(m[45:47], torch.tensor([-0.6, 0.8])))
check("contact (L,R)→(R,L)", torch.allclose(m[47:49], torch.tensor([0.0, 1.0])))
check("프레임 이중미러=항등", torch.allclose(mirror_frame(m), fr, atol=1e-6))

print("== 4) 이력 평탄화 블록 배치 검증 (항별 [프레임0..4] 블록, 프레임별 동일 변환) ==")
# joint_pos 블록: [45,105), 프레임 t 시작 = 45 + 12t. 프레임2의 left_knee(관절 6)=0.5
x = torch.zeros(1, 245)
x[0, 45 + 2 * 12 + 6] = 0.5
xm = mirror_biped12(env=None, obs=x.clone(), actions=None)[0][1]  # 평탄 텐서 경로
check("이력 프레임2 left_knee=0.5 → 같은 프레임2 right_knee=-0.5",
      near(xm[45 + 2 * 12 + 7].item(), -0.5) and near(xm[45 + 2 * 12 + 6].item(), 0.0)
      and near(xm.abs().sum().item(), 0.5))
# feet_contact 블록: [235,245), 프레임 t 시작 = 235 + 2t. 프레임4 (L=1,R=0)
x = torch.zeros(1, 245)
x[0, 235 + 2 * 4 + 0] = 1.0
xm = mirror_biped12(env=None, obs=x.clone(), actions=None)[0][1]
check("이력 프레임4 contact L→R 스왑 (같은 프레임 위치)",
      near(xm[235 + 2 * 4 + 1].item(), 1.0) and near(xm.abs().sum().item(), 1.0))
# critic: base_lin_vel 프레임3의 vy → -vy (같은 프레임 위치)
x = torch.zeros(1, 15)
x[0, 3 * 3 + 1] = 0.7
xm = mirror_biped12(env=None, obs={"critic": x.clone()}, actions=None)[0]["critic"][1]
check("critic 이력 프레임3 vy → -vy", near(xm[3 * 3 + 1].item(), -0.7) and near(xm.abs().sum().item(), 0.7))

print("== 5) 액션 미러 = 관절 미러 규칙 동일 ==")
a = torch.zeros(1, 12)
a[0, 0] = 1.0   # left_hip_f
a[0, 11] = 2.0  # right_ankle_r
am = mirror_biped12(env=None, obs=None, actions=a)[1][1]
check("action left_hip_f=1 → right_hip_f=-1", near(am[1].item(), -1.0))
check("action right_ankle_r=2 → left_ankle_r=+2", near(am[10].item(), 2.0))

print("== 6) TensorDict 경로 (rsl_rl 실제 컨테이너) ==")
try:
    from tensordict import TensorDict
    td = TensorDict(
        {"policy": torch.randn(B, 245), "critic": torch.randn(B, 15)}, batch_size=[B]
    )
    td2, _ = mirror_biped12(env=None, obs=td, actions=None)
    check("TensorDict 배치 2B + 첫 절반 원본",
          td2.batch_size[0] == 2 * B
          and torch.equal(td2["policy"][:B], td["policy"])
          and torch.equal(td2["critic"][:B], td["critic"]))
    td3, _ = mirror_biped12(env=None, obs=td2[B:], actions=None)
    check("TensorDict 이중미러=항등",
          torch.allclose(td3["policy"][B:], td["policy"], atol=1e-6)
          and torch.allclose(td3["critic"][B:], td["critic"], atol=1e-6))
except ImportError:
    print("  [SKIP] tensordict 미설치 — dict 경로로 갈음")

print("== 7) 차원 불일치 방어 ==")
try:
    mirror_biped12(env=None, obs={"policy": torch.zeros(2, 244)}, actions=None)
    raise AssertionError("FAIL: 244차원이 통과됨")
except ValueError:
    check("잘못된 차원 → ValueError", True)
try:
    mirror_biped12(env=None, obs={"unknown": torch.zeros(2, 10)}, actions=None)
    raise AssertionError("FAIL: 미지 그룹키가 통과됨")
except ValueError:
    check("미지 그룹키 → ValueError", True)

print(f"\nALL {len(PASS)} CHECKS PASSED")
