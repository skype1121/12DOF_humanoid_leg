"""배포 경로 단일 원본 — 브리지(jetson/scripts/rl_bridge_node.py)와 판정 스크립트(eval_drift_dr --deploy old)가
같은 상수·같은 순서를 쓰게 한다 (0905 배포 계약 감사 B1·B2: 판정 래퍼가 LPF→슬루 순서였고, 관측 actions 가
래퍼 출력으로 오염돼 있었다 — 5개 축이 동일 발견).

실물 경로 (rl_bridge_node.py _tick, 0906 코드 확인):
  raw(정책) → [러너 policy_runner_stage4.step] target = DEFAULT_POSE + 0.25·raw
             → 슬루 ±4°/틱 (기준 = 직전 방출값, 첫 틱은 실측 qpos)
           → [브리지, stand 계열] EMA LPF β0.27 (첫 틱 = 슬루 출력으로 초기화, 점프 없음)
           → [브리지] 온셋 블렌딩 1.5s 스무스스텝: 시작점(stage8 스트리밍 명령 = 착지 자세) → 타깃
           → 퍼블리시 → (전송 지연 4~8틱) → stage8 노드 슬루/클램프 → 모터
관측의 'actions' 는 언제나 raw (러너 step 주석: 반환값을 되먹이지 말 것).

시뮬 판정: DeployChain.step(raw) 을 액션텀 훅에 넣어 raw → 체인 출력으로 바꾸고(액션텀의 지연·슬루가 그 뒤에
stage8 몫으로 붙는다), env.step 에는 raw 를 넘겨 관측 actions 가 raw 로 남게 한다.

검증: python3 rl_walking/deploy/deploy_chain.py --selftest   (torch 만, GPU·Isaac 불필요)
"""
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from rl_walking.deploy.policy_runner import ACTION_SCALE, RATE_HZ, SLEW_RAD  # noqa: E402

BLEND_S = 1.5          # 브리지 온셋 블렌딩 [s] — stage8 인수 크로스페이드 (rl_bridge_node.BLEND_S 의 원본)
STAND_LPF_BETA = 0.27  # stand 계열 EMA 저역필터 (차단 ≈2.5Hz@50Hz) (rl_bridge_node.STAND_LPF_BETA 의 원본)
TICK_S = 1.0 / RATE_HZ


def blend_alpha(elapsed_s, dur_s=BLEND_S):
    """온셋 블렌딩 가중치 — 스무스스텝 0→1 (양 끝 기울기 0, 단조증가). dur_s<=0 이면 즉시 1.0."""
    if dur_s <= 0.0:
        return 1.0
    a = min(1.0, max(0.0, elapsed_s / dur_s))
    return a * a * (3.0 - 2.0 * a)


def lpf_step(state, x, beta=STAND_LPF_BETA):
    """EMA 1차 저역필터 한 스텝 — state None 이면 x 로 초기화 (점프 없음). 리스트 입출력(브리지용)."""
    if state is None:
        return list(x)
    return [s + beta * (v - s) for s, v in zip(state, x)]


class DeployChain:
    """실물 배포 경로의 벡터화 재현 (torch, N env × J 관절): 러너 슬루 → stand LPF → 온셋 블렌딩.

    step(raw (N,J)) → 관절 타깃 (N,J) [rad, env 관절 순서]. def_pose/land_pose 는 env 관절 순서의 (N,J).
    관측 actions 는 raw 를 써야 한다(호출자 책임). 에피소드 리셋 env 는 reset(ids, qpos) 로 재초기화.
    """

    def __init__(self, def_pose, land_pose, qpos0=None, *, lpf_beta=STAND_LPF_BETA, blend_s=BLEND_S,
                 slew_rad=SLEW_RAD, action_scale=ACTION_SCALE, tick_s=TICK_S, stand=True):
        import torch
        self.def_pose = def_pose
        self.land = land_pose.clone()
        self.prev = (qpos0 if qpos0 is not None else land_pose).clone()   # 러너 _prev_emitted 초기값 = 실측 qpos
        self.filt = torch.zeros_like(self.prev)
        self.filt_ok = torch.zeros(self.prev.shape[0], dtype=torch.bool, device=self.prev.device)
        self.t = torch.zeros(self.prev.shape[0], dtype=torch.long, device=self.prev.device)
        self.lpf_beta = float(lpf_beta) if stand else 1.0
        self.blend_s = float(blend_s)
        self.slew = float(slew_rad)
        self.scale = float(action_scale)
        self.tick_s = float(tick_s)

    def reset(self, ids, qpos):
        """리셋된 env: 슬루 기준점 = 현재 qpos, LPF 재초기화, 블렌딩 시계 0."""
        self.prev[ids] = qpos[ids]
        self.filt_ok[ids] = False
        self.t[ids] = 0

    def step(self, raw):
        import torch
        target = self.def_pose + self.scale * raw
        emitted = self.prev + torch.clamp(target - self.prev, -self.slew, self.slew)
        self.prev = emitted
        if self.lpf_beta < 1.0:
            init = ~self.filt_ok
            if torch.any(init):
                self.filt[init] = emitted[init]
                self.filt_ok[init] = True
            self.filt = self.filt + self.lpf_beta * (emitted - self.filt)
            out = self.filt
        else:
            out = emitted
        if self.blend_s > 0.0:
            a = torch.clamp(self.t.to(out.dtype) * self.tick_s / self.blend_s, 0.0, 1.0)
            alpha = (a * a * (3.0 - 2.0 * a)).unsqueeze(1)
            out = self.land * (1.0 - alpha) + out * alpha
        self.t = self.t + 1
        return out.clone()


# ---------------------------------------------------------------------------
# 자가검증 — 벡터화 체인 vs 스칼라 참조(브리지 함수 lpf_step/blend_alpha + 러너 슬루식)
# ---------------------------------------------------------------------------

def _reference(raws, def_pose, land, qpos0, lpf_beta, blend_s, slew, scale, tick_s, reset_at=None):
    """env 1개·관절 J 의 스칼라 참조 구현. raws: [T][J] 리스트. reset_at: (틱, qpos 리스트)."""
    prev = list(qpos0)
    filt = None
    t0 = 0
    outs = []
    for i, raw in enumerate(raws):
        if reset_at is not None and i == reset_at[0]:
            prev = list(reset_at[1]); filt = None; t0 = i
        target = [d + scale * r for d, r in zip(def_pose, raw)]
        emitted = [p + max(-slew, min(slew, tg - p)) for p, tg in zip(prev, target)]
        prev = emitted
        if lpf_beta < 1.0:
            filt = lpf_step(filt, emitted, lpf_beta)
            out = filt
        else:
            out = emitted
        al = blend_alpha((i - t0) * tick_s, blend_s)
        out = [ld * (1.0 - al) + o * al for ld, o in zip(land, out)]
        outs.append(list(out))
    return outs


def selftest():
    import torch
    torch.manual_seed(0)
    N, J, T = 3, 12, 220
    def_pose = torch.randn(N, J) * 0.2
    land = torch.zeros(N, J)
    qpos0 = torch.randn(N, J) * 0.05
    raws = torch.randn(T, N, J) * 1.5                      # 목표가 슬루 포화를 자주 넘도록 크게
    reset_tick, reset_q = 90, torch.randn(N, J) * 0.05
    for lpf_beta, blend_s in ((STAND_LPF_BETA, BLEND_S), (1.0, 0.0), (STAND_LPF_BETA, 0.0), (1.0, BLEND_S)):
        ch = DeployChain(def_pose, land, qpos0, lpf_beta=lpf_beta, blend_s=blend_s)
        outs = []
        for i in range(T):
            if i == reset_tick:                             # env 1 만 리셋
                ch.reset(torch.tensor([1]), reset_q)
            outs.append(ch.step(raws[i]))
        outs = torch.stack(outs)                            # (T,N,J)
        for n in range(N):
            ref = _reference([raws[i, n].tolist() for i in range(T)], def_pose[n].tolist(), land[n].tolist(),
                             qpos0[n].tolist(), lpf_beta, blend_s, SLEW_RAD, ACTION_SCALE, TICK_S,
                             reset_at=(reset_tick, reset_q[n].tolist()) if n == 1 else None)
            err = (outs[:, n] - torch.tensor(ref)).abs().max().item()
            assert err < 1e-5, f"벡터화 ≠ 참조 (env {n}, lpf {lpf_beta}, blend {blend_s}): {err:.2e}"
        if blend_s == 0.0 and lpf_beta == 1.0:              # 순수 러너 슬루: 틱당 |Δ| ≤ 4°
            d = (outs[1:] - outs[:-1]).abs()
            d[reset_tick - 1, 1] = 0.0                      # 리셋 점프는 제외
            assert d.max().item() <= SLEW_RAD + 1e-6, f"슬루 초과 {math.degrees(d.max().item()):.3f}°"
        if blend_s > 0.0:                                   # 첫 틱은 정확히 착지 자세(α=0)
            assert torch.allclose(outs[0], land, atol=1e-7), "첫 틱 α≠0"
            assert torch.allclose(outs[reset_tick, 1], land[1], atol=1e-7), "리셋 env 첫 틱 α≠0"
    # 순서 감수성: 슬루→LPF 와 LPF→슬루(h2 래퍼) 는 포화 구간에서 다르다 — 정합 작업의 존재 이유
    step_raw = torch.full((1, J), 2.0)                      # 목표 0.5rad ≈ 28.6° 스텝
    ch = DeployChain(torch.zeros(1, J), torch.zeros(1, J), torch.zeros(1, J), blend_s=0.0)
    slew_lpf = [ch.step(step_raw)[0, 0].item() for _ in range(10)]
    f = None; prev = 0.0; lpf_slew = []
    for _ in range(10):                                     # h2: LPF 먼저, 그 다음 슬루
        f = 0.5 if f is None else f + STAND_LPF_BETA * (0.5 - f)
        prev = prev + max(-SLEW_RAD, min(SLEW_RAD, f - prev)); lpf_slew.append(prev)
    diff = max(abs(a - b) for a, b in zip(slew_lpf, lpf_slew))
    assert diff > 0.01, f"순서 차이가 없다? {diff}"
    print(f"[selftest] deploy_chain: 벡터화=참조(4구성×{N}env×{T}틱, 리셋 포함) · 슬루 ≤4°/틱 · 첫 틱 α=0 · "
          f"순서 감수성 {math.degrees(diff):.2f}° (슬루→LPF vs LPF→슬루, 28.6° 스텝 10틱) — ALL PASS")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)
