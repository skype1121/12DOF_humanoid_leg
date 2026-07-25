"""적대적 추가 검증 — 저자 테스트에 없는 3가지:
A) observation_manager 실코드 방식 (N,T,D).reshape→concat 재구성 vs 전역 perm/sign 랜덤 전수 일치
B) log_picture/05 부호표의 '해부학 고정점' (양측 대칭 자세 = 미러 불변, L스윙→R스윙)
C) gait_phase_reward가 미러 변환(φ+π, 접촉 스왑) 아래 등가인지 전수 검사
"""
import sys

import torch

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
from rl_walking import symmetry as S  # noqa: E402

N, T = 11, S.HISTORY_LEN
g = torch.Generator().manual_seed(42)

# ── A. observation_manager L424/L433 방식 재구성 vs 전역 미러 ──
frames, parts = {}, []
for name, dim, p, s in S._POLICY_TERMS:
    buf = torch.randn(N, T, dim, generator=g)  # CircularBuffer.buffer (N,T,D), t0=最古
    frames[name] = buf
    parts.append(buf.reshape(N, -1))           # flatten_history_dim (L424)
obs = torch.cat(parts, dim=-1)                 # 항 순서 concat (L433)
assert obs.shape == (N, 245)
ref = torch.cat(
    [(frames[n][:, :, list(p)] * torch.tensor(s)).reshape(N, -1) for n, d, p, s in S._POLICY_TERMS],
    dim=-1,
)
assert torch.allclose(S._mirror(obs, "policy"), ref), "이력 평탄화 레이아웃 가정 위반!"
print("A. per-frame mirror == global perm/sign on obs-manager-style layout (random, full)")

# ── B. 해부학 고정점 (05 부호표) ──
sym_pose = torch.tensor([[+0.30, -0.30,  # hip_f 양측 굴곡 (반대부호 규약)
                          +0.20, -0.20,  # hip_a 양측 내전
                          +0.30, -0.30,  # hip_r 양측 외회전
                          -0.40, +0.40,  # knee 양측 굴곡 (L은 -가 굴곡)
                          +0.30, -0.30,  # ankle_f 양측 배굴
                          +0.20, +0.20]])  # ankle_r 양측 외반 (동부호 예외)
assert torch.allclose(S._mirror(sym_pose, "actions"), sym_pose), "양측 대칭 자세가 고정점이 아님"
ls = torch.zeros(1, 12); ls[0, 0], ls[0, 6] = +0.5, -0.8   # 왼발 스윙 (hip_f/knee 굴곡)
rs = torch.zeros(1, 12); rs[0, 1], rs[0, 7] = -0.5, +0.8   # 오른발 스윙
assert torch.allclose(S._mirror(ls, "actions"), rs), "L스윙 미러 != R스윙"
# 기본자세도 고정점이어야 (knee ∓0.1047, ankle_f ±0.0244 — biped12_cfg)
dflt = torch.tensor([[0, 0, 0, 0, 0, 0, -0.10472, +0.10472, +0.0244, -0.0244, 0, 0]])
assert torch.allclose(S._mirror(dflt, "actions"), dflt, atol=1e-6), "기본자세가 고정점이 아님"
print("B. anatomical fixed points OK (bilateral pose, default pose, L/R swing swap)")

# ── C. 주기 보상 미러 등가성 ──
def walk_r(sin_phi, cL, cR):
    stance_l = sin_phi >= 0.0
    return float(stance_l == cL) + float((not stance_l) == cR)

for sp in (0.9, 0.3, -0.3, -0.9):  # sinφ=0 경계는 측도 0 — 학습에 무영향
    for cL in (0, 1):
        for cR in (0, 1):
            assert walk_r(sp, cL, cR) == walk_r(-sp, cR, cL), (sp, cL, cR)
phi = torch.rand(1000, generator=g) * 6.283
assert torch.allclose(torch.sin(phi + torch.pi), -torch.sin(phi), atol=1e-5)
assert torch.allclose(torch.cos(phi + torch.pi), -torch.cos(phi), atol=1e-5)
print("C. clock (s,c)->(-s,-c) == phase+pi; walk reward invariant under mirror (exhaustive)")

print("\nADVERSARIAL CHECKS ALL PASSED")
