"""W3S3t*(걷기 토크 예산 후보, reports/gen_w3/TORQUE_W3T_DESIGN.md) CPU 단위시험 (1007) — torch CPU 만(Isaac·GPU 불필요).

  python3 rl_walking/scripts/unit_w3t.py [--lp <log_picture 디렉터리>]

rl_walking/w3.py 를 스텁 mdp_hang 과 함께 올리고, mdp_stage4.torque_saturation 은 실제 소스를 AST 로 떼어 쓴다(Isaac import 없이 — unit_w3d.py 방식).
  [A] torque_saturation 식·부호·크기(손 계산 대조) + 한계에서의 평탄 구간(적용 토크는 잘린 값 — 요구가 한계를 넘는 동안은 값이 변하지 않음).
  [B] W3S3t 의 새 항 = w3_step_gated(torque_saturation): 걷기(ge 1)만 값·서기(ge 0) 정확히 0, 기존 서기 항(w3_stand_gated, 가중 −2)과 서로소,
      서기 env 의 두 항 합은 종전과 비트 동일, 가중 곱은 음수(벌점).
  [C] 실제 걷기 토크 재생(X12 추적 npz 가 있을 때만): 새 항 평균 = 설계 문서 수치(0.0076, ±10 %)·numpy 식과 일치, 가중 −40/−80 의 초당 크기, KE 서기 ≈ 0.
  [D] 토크 한계 배율과 겹칠 때(W3S3f 의 effort_scale): 분모가 그 env 의 한계(data.joint_effort_limits)라 같은 토크의 벌점이 커진다.
  [E] W3S3tD 스윙 창 반사실: 같은 두 궤적(긴 스윙 v .12–.88 / 짧은 스윙 v .25–.75, 둘 다 5 cm 반사인)의 반주기 보상 합(w3_height 3·w3_periodic −1.5)이
      기본 창에서는 긴 > 짧은, 새 창 (0.25, 0.75) 에서는 짧은 > 긴 · 새 창에서 창 밖 양발 지지는 벌점 0 · 가만히 서기는 두 창 모두 음수.
  [F] 설정 정적 점검(AST): 새 클래스 3 개의 부모·상수, 건드리는 속성이 보상뿐(관측·행동·명령·이벤트·장면 불변 = 계약 불변), 등록 이름·중복 없음.
종료 코드 0 = 전부 통과.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import math
import os
import sys
import types

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DT = 0.02
F_STEP = 1.1
FAILS = []
LIM = torch.tensor([25.0] * 10 + [24.0] * 2)                    # biped12_cfg: AK70 25 · 발목 롤 24 N·m (관절 순서 LhF,RhF,LhA,RhA,LhR,RhR,Lkn,Rkn,LaF,RaF,LaR,RaR)
MASS = (0.4, 0.4, 16.8)
W_HEIGHT, W_PERIODIC = 3.0, -1.5                                 # Biped12HangStandW3BaseEnvCfg 의 가중([F] 에서 소스와 대조)
W_STAND_SAT = -2.0                                               # 기존 서기 항 가중(stage4_v2_env_cfg V15)


def check(cond, msg):
    print(("  [통과] " if cond else "  [실패] ") + msg)
    if not cond:
        FAILS.append(msg)


# ------------------------------------------------------------------ 모듈 적재
def _load():
    pkg = "w3tstub"
    m = types.ModuleType(pkg); m.__path__ = []; sys.modules[pkg] = m
    H = types.ModuleType(f"{pkg}.mdp_hang")
    H.G = 9.81
    H._base_cfg = lambda: types.SimpleNamespace(name="robot")
    H._g21_bids = lambda env: [0, 1]
    H._feet_body_ids = lambda env: [0, 1]
    H._resolve_nested = lambda env, p: dict(p or {})
    sys.modules[f"{pkg}.mdp_hang"] = H; m.mdp_hang = H
    spec = importlib.util.spec_from_file_location(f"{pkg}.w3", os.path.join(REPO, "rl_walking", "w3.py"))
    w3 = importlib.util.module_from_spec(spec); sys.modules[spec.name] = w3; m.w3 = w3
    spec.loader.exec_module(w3)
    src = open(os.path.join(REPO, "rl_walking", "mdp_stage4.py"), encoding="utf-8").read()
    seg = [ast.get_source_segment(src, n) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "torque_saturation"]
    assert len(seg) == 1, "mdp_stage4.torque_saturation 을 찾지 못함"
    ns = {"torch": torch, "SceneEntityCfg": lambda name, **kw: types.SimpleNamespace(name=name)}
    exec(seg[0], ns)
    return w3, ns["torque_saturation"]


W3, TSAT = _load()


# ------------------------------------------------------------------ 가짜 env
class _Scene:
    def __init__(self, env):
        self.env = env; self.sensors = {"contact_forces": env.sensor}

    def __getitem__(self, k):
        return self.env


class _Env:
    def __init__(self, N):
        self.num_envs = N; self.device = torch.device("cpu"); self.step_dt = DT
        self.episode_length_buf = torch.full((N,), 10, dtype=torch.long); self.common_step_counter = 0
        self._ge = torch.zeros(N, dtype=torch.bool); self._gu = torch.zeros(N); self._gf = torch.full((N,), F_STEP)
        self.data = types.SimpleNamespace(applied_torque=torch.zeros(N, 12), joint_effort_limits=LIM.repeat(N, 1).clone(),
                                          default_mass=torch.tensor(MASS).repeat(N, 1), body_pos_w=torch.zeros(N, 2, 3), body_lin_vel_w=torch.zeros(N, 2, 3))
        self.sensor = types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=torch.zeros(N, 2, 3)))
        self.scene = _Scene(self)


def np_term(tq, lim, thresh):
    r = tq.abs() / (lim + 1e-6)
    return torch.clamp(r - thresh, min=0.0).pow(2).mean(dim=-1)


# ------------------------------------------------------------------ [A]
def test_a():
    print("[A] torque_saturation 식·부호·크기")
    env = _Env(8)
    tq = torch.zeros(8, 12)
    tq[1, 2] = 25.0                         # 왼 힙 외전 한계
    tq[2, 2] = -25.0                        # 부호 반대
    tq[3, 6] = 18.5                         # 왼 무릎(걷기 지지 평균)
    tq[4, 10] = 24.0                        # 발목 롤 한계(24)
    tq[5, 6] = 15.0                         # 문턱 정확히
    tq[6, 2] = 25.0; tq[6, 7] = 23.0        # 힙 외전 포화 + 반대쪽 무릎
    tq[7] = LIM                             # 전 관절 한계
    env.data.applied_torque = tq
    v = TSAT(env, thresh=0.6)
    check(float(v[0]) == 0.0, f"토크 0 → 0 (실제 {float(v[0])})")
    check(abs(float(v[1]) - 0.16 / 12) < 1e-6, f"한 관절 한계(25 N·m) → 0.4²/12 = {0.16 / 12:.6f} (실제 {float(v[1]):.6f})")
    check(float(v[2]) == float(v[1]), "부호 무관(|τ|)")
    check(abs(float(v[3]) - (18.5 / 25 - 0.6) ** 2 / 12) < 1e-6, f"무릎 18.5 N·m → (0.74 − 0.6)²/12 = {(18.5 / 25 - 0.6) ** 2 / 12:.6f} (실제 {float(v[3]):.6f})")
    check(abs(float(v[4]) - 0.16 / 12) < 1e-6, "발목 롤은 자기 한계(24 N·m)로 정규화")
    check(float(v[5]) < 1e-9, f"문턱(15 N·m) 이하 = 0 (실제 {float(v[5]):.2e})")
    check(abs(float(v[6]) - (0.16 + (23 / 25 - 0.6) ** 2) / 12) < 1e-6, "두 관절 초과분 제곱의 합 / 12")
    check(abs(float(v[7]) - 0.16) < 1e-5, f"전 관절 한계 = 상한 0.16 (실제 {float(v[7]):.5f})")
    check(bool((v >= 0).all()), "값 ≥ 0(벌점 크기) — 음수 가중과 곱해 벌점")
    check(bool(torch.allclose(v, np_term(tq, LIM, 0.6), atol=1e-7)), "설계 도구(torque_phase.sat_term)와 같은 식")
    # 평탄 구간: PD 요구 kp·오차 가 한계를 넘으면 적용 토크는 25 로 잘려 값이 같다
    env2 = _Env(4)
    err = torch.deg2rad(torch.tensor([5.0, 9.6, 10.0, 14.1]))                    # 지지 힙 외전 추종 오차(자료: 이륙 구간 평균 14.1°)
    tq2 = torch.zeros(4, 12); tq2[:, 2] = torch.clamp(150.0 * err, -25.0, 25.0)
    env2.data.applied_torque = tq2
    v2 = TSAT(env2, thresh=0.6)
    check(float(v2[0]) == 0.0 and float(v2[1]) > 0, f"오차 5°(13.1 N·m) = 0, 9.6°(25 N·m) > 0 (값 {[round(float(x), 5) for x in v2]})")
    check(float(v2[2]) == float(v2[3]) == float(v2[1]), "평탄 구간: 오차 9.6°·10°·14.1° 는 적용 토크가 같아(25) 벌점도 같다 — 요구를 한계 아래로 내려야 벌점이 준다")
    xs = torch.linspace(15.0, 25.0, 11); env3 = _Env(11); t3 = torch.zeros(11, 12); t3[:, 6] = xs; env3.data.applied_torque = t3
    v3 = TSAT(env3, thresh=0.6)
    check(bool((v3[1:] > v3[:-1]).all()), "15–25 N·m 에서 단조 증가(토크를 줄이면 벌점이 준다)")


# ------------------------------------------------------------------ [B]
def test_b():
    print("[B] w3_step_gated(torque_saturation) — 걷기만·서기 0·기존 서기 항과 서로소")
    env = _Env(6)
    env._ge = torch.tensor([True, False, True, False, True, False])
    tq = torch.zeros(6, 12); tq[:, 2] = 25.0; tq[:, 7] = 20.0
    env.data.applied_torque = tq
    raw = TSAT(env, thresh=0.6)
    step = W3.w3_step_gated(env, TSAT, {"thresh": 0.6})
    stand = W3.w3_stand_gated(env, TSAT, {"thresh": 0.6})
    check(bool((step[env._ge] == raw[env._ge]).all()) and bool((step[~env._ge] == 0).all()), "새 항: ge 1 = 원값, ge 0 = 정확히 0")
    check(bool((stand[~env._ge] == raw[~env._ge]).all()) and bool((stand[env._ge] == 0).all()), "기존 서기 항: ge 0 = 원값, ge 1 = 정확히 0")
    check(bool((step * stand == 0).all()) and bool(torch.equal(step + stand, raw)), "두 항은 서로소이고 합 = 원값")
    for w in (-40.0, -80.0):
        r_old = W_STAND_SAT * stand * DT
        r_new = W_STAND_SAT * stand * DT + w * step * DT
        check(bool(torch.equal(r_new[~env._ge], r_old[~env._ge])), f"가중 {w:g}: 서기 env 의 틱 보상은 종전과 비트 동일")
        check(bool((r_new[env._ge] < 0).all()) and bool((r_old[env._ge] == 0).all()), f"가중 {w:g}: 걷기 env 는 종전 0 → 새 벌점 {float(r_new[0]) / DT:+.3f}/s(이 토크에서)")
    sig = inspect.signature(W3.w3_step_gated)
    check(list(sig.parameters)[:3] == ["env", "func", "params"], "w3_step_gated(env, func, params) 서명")


# ------------------------------------------------------------------ [C]
def test_c(lp):
    print("[C] 실제 걷기 토크 재생(X12)")
    import numpy as np
    p = os.path.join(lp, "transall_x12_w3load_cmds_d2_s21.npz")
    if not os.path.exists(p):
        print(f"  [건너뜀] {p} 없음 — --lp 로 log_picture 위치를 주면 실행"); return
    z = np.load(p, allow_pickle=True); pre = z["pre_pass"].astype(bool); K = lambda x: int(round(x / DT))   # noqa: E731
    seg = [(3, 8), (9, 14), (15, 20), (21, 26), (32, 37), (38, 43), (49, 54), (60, 65)]
    walk = np.concatenate([z["tq"][K(a):K(b)].astype(np.float32)[(z["alive"][K(a):K(b)] & z["ge"][K(a):K(b)] & pre[None])] for a, b in seg])
    stand = np.concatenate([z["tq"][K(a):K(b)].astype(np.float32)[(z["alive"][K(a):K(b)] & z["ke"][K(a):K(b)] & ~z["ge"][K(a):K(b)] & pre[None])] for a, b in ((29.5, 31), (46.5, 48), (57.8, 59))])
    env = _Env(len(walk)); env.data.applied_torque = torch.from_numpy(walk); env._ge[:] = True
    v = W3.w3_step_gated(env, TSAT, {"thresh": 0.6})
    mean = float(v.mean())
    check(0.0076 * 0.9 <= mean <= 0.0076 * 1.1, f"걷기 {len(walk)} 틱 평균 {mean:.5f} — 설계 문서 0.0076 의 ±10 % 안")
    ref = (np.clip(np.abs(walk) / (LIM.numpy() + 1e-6) - 0.6, 0, None) ** 2).mean(-1).mean()
    check(abs(mean - float(ref)) < 1e-6, f"numpy 식과 일치({float(ref):.6f})")
    print(f"    가중별 걷기 초당: −40 → {-40 * mean:+.3f} · −80 → {-80 * mean:+.3f} (걷기 순 보상 +1.28/s 대비 {40 * mean / 1.284:.0%} · {80 * mean / 1.284:.0%})")
    check(-40 * mean < -0.2 and -80 * mean > -1.284, "−40 은 기존 걷기 벌점(−0.14…−0.46/s)과 같은 급, −80 도 걷기 순 보상(+1.28/s)보다 작다")
    env2 = _Env(len(stand)); env2.data.applied_torque = torch.from_numpy(stand)
    check(float(TSAT(env2, thresh=0.6).mean()) < 1e-5, f"KE 서기 {len(stand)} 틱의 원값 ≈ 0 ({float(TSAT(env2, thresh=0.6).mean()):.2e}) — 서기 토크는 문턱 아래")
    env2._ge[:] = False
    check(float(W3.w3_step_gated(env2, TSAT, {"thresh": 0.6}).abs().max()) == 0.0, "서기(ge 0)에서 새 항 정확히 0")
    hA = np.abs(walk[:, [2, 3]]); kn = np.abs(walk[:, [6, 7]])
    print(f"    참고: 힙 외전 RMS {np.sqrt((hA ** 2).mean()):.1f}·포화 {100 * (hA >= 24.5).mean():.1f} % · 무릎 RMS {np.sqrt((kn ** 2).mean()):.1f}·포화 {100 * (kn >= 24.5).mean():.1f} %")


# ------------------------------------------------------------------ [D]
def test_d():
    print("[D] 토크 한계 배율(effort_scale)과 겹칠 때")
    env = _Env(3)
    tq = torch.zeros(3, 12); tq[:, 6] = 15.0; tq[2, 6] = 17.5
    env.data.applied_torque = tq
    env.data.joint_effort_limits[1:] = LIM * 0.7
    v = TSAT(env, thresh=0.6)
    check(float(v[0]) < 1e-9, "명목 한계: 15 N·m = 문턱 → 0")
    check(abs(float(v[1]) - (15 / 17.5 - 0.6) ** 2 / 12) < 1e-6, f"한계 0.7 배(17.5): 같은 15 N·m → (0.857 − 0.6)²/12 = {(15 / 17.5 - 0.6) ** 2 / 12:.5f}")
    check(abs(float(v[2]) - 0.16 / 12) < 1e-6, "한계 0.7 배에서 17.5 N·m = 포화 → 0.4²/12 (분모 = 그 env 의 한계)")


# ------------------------------------------------------------------ [E]
def _half_cycle(env, traj, win):
    """왼발 스윙 반주기(u 0 → 0.5) 한 번의 보상 합(가중·dt 곱): (w3_height, w3_periodic). traj(v) → (스윙발 높이 m, 스윙발 공중?)."""
    mg = 9.81 * sum(MASS)
    rh = rp = 0.0
    n = int(round(0.5 / (F_STEP * DT)))                                          # 22.7 → 23 틱
    for k in range(n):
        u = (k + 0.5) / n * 0.5; v = 2.0 * u
        z, air = traj(v)
        env._gu = torch.full((env.num_envs,), u)
        bp = torch.zeros(env.num_envs, 2, 3); bp[:, 0, 2] = z; env.data.body_pos_w = bp
        F = torch.zeros(env.num_envs, 2, 3)
        F[:, 0, 2] = 0.0 if air else 0.5 * mg; F[:, 1, 2] = mg if air else 0.5 * mg
        env.sensor.data.net_forces_w = F
        rh += W_HEIGHT * float(W3.w3_swing_height(env, win=win)[0]) * DT
        rp += W_PERIODIC * float(W3.w3_periodic_contact(env, win=win)[0]) * DT
    return rh, rp


def test_e():
    print("[E] W3S3tD 스윙 창 반사실(반주기 보상 합 = w3_height·3 + w3_periodic·(−1.5))")
    env = _Env(1); env._ge[:] = True
    long_ = lambda v: (0.05 * math.sin(math.pi * (v - 0.12) / 0.76), True) if 0.12 <= v < 0.88 else (0.0, False)     # noqa: E731
    short = lambda v: (0.05 * math.sin(math.pi * (v - 0.25) / 0.50), True) if 0.25 <= v < 0.75 else (0.0, False)     # noqa: E731
    stand = lambda v: (0.0, False)                                                                                    # noqa: E731
    old, new = (0.12, 0.88), (0.25, 0.75)
    R = {}
    for nm, tr in (("긴 스윙", long_), ("짧은 스윙", short), ("가만히 서기", stand)):
        for wn, w in (("기본 창", old), ("새 창", new)):
            R[nm, wn] = _half_cycle(env, tr, w)
            print(f"    {nm:6s} · {wn}: 높이 {R[nm, wn][0]:+.3f} · 주기 접촉 {R[nm, wn][1]:+.3f} · 합 {sum(R[nm, wn]):+.3f}")
    tot = lambda a, b: sum(R[a, b])                                                                                   # noqa: E731
    d_old = tot("긴 스윙", "기본 창") - tot("짧은 스윙", "기본 창"); d_new = tot("짧은 스윙", "새 창") - tot("긴 스윙", "새 창")
    check(d_old > 0.3, f"기본 창 (0.12, 0.88): 긴 스윙이 짧은 스윙보다 높다(반주기 +{d_old:.3f} = 초당 +{2 * F_STEP * d_old:.2f}) — 현 보상이 긴 한발 지지를 고른다")
    check(d_new > 0.1, f"새 창 (0.25, 0.75): 순서가 뒤집힌다 — 짧은 스윙이 높다(반주기 +{d_new:.3f} = 초당 +{2 * F_STEP * d_new:.2f}; 긴 스윙도 새 창 보상의 "
                       f"{tot('긴 스윙', '새 창') / tot('짧은 스윙', '새 창'):.0%} 는 받는다 — 끌림은 완만)")
    check(R["짧은 스윙", "기본 창"][1] < -0.05 and R["짧은 스윙", "새 창"][1] == 0.0, "창 밖 양발 지지: 기본 창에서는 벌점, 새 창에서는 0")
    check(abs(R["긴 스윙", "기본 창"][0] - W_HEIGHT * 0.76 * 0.5 / F_STEP) < 0.08, f"긴 스윙·기본 창 높이 보상 ≈ 3 × 0.76 × 0.4545 s = {W_HEIGHT * 0.76 * 0.5 / F_STEP:.3f}")
    check(abs(R["짧은 스윙", "새 창"][0] - W_HEIGHT * 0.50 * 0.5 / F_STEP) < 0.08, f"짧은 스윙·새 창 높이 보상 ≈ 3 × 0.50 × 0.4545 s = {W_HEIGHT * 0.50 * 0.5 / F_STEP:.3f}(걷기 최대 보상이 줄어든다 — 문서의 알려진 부작용)")
    check(tot("가만히 서기", "기본 창") < 0 and tot("가만히 서기", "새 창") < 0, "가만히 서기는 두 창 모두 음수(스윙 창 안 스윙발 하중 벌점) — 서기 해킹이 생기지 않는다")
    check(tot("짧은 스윙", "새 창") > tot("가만히 서기", "새 창") + 0.5, "새 창에서도 걷기(짧은 스윙) ≫ 가만히 서기")
    for f in (W3.w3_periodic_contact, W3.w3_swing_height):
        p = inspect.signature(f).parameters
        check("win" in p and tuple(p["win"].default) == (0.12, 0.88), f"{f.__name__} 의 win 인자 기본값 (0.12, 0.88) — 기존 태스크는 그대로")
    env._ge[:] = False
    check(sum(_half_cycle(env, long_, new)) == 0.0, "서기(ge 0)에서는 두 항 모두 0(새 창도 걷기 env 에만 작용)")


# ------------------------------------------------------------------ [F]
def test_f():
    print("[F] 설정 정적 점검(AST)")
    src = open(os.path.join(REPO, "rl_walking", "hang_stand_env_cfg.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    cls = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    want = {"Biped12HangStandW3S3tEnvCfg": "Biped12HangStandW3S3dEnvCfg", "Biped12HangStandW3S3tHEnvCfg": "Biped12HangStandW3S3tEnvCfg",
            "Biped12HangStandW3S3tDEnvCfg": "Biped12HangStandW3S3dEnvCfg"}
    for name, base in want.items():
        ok = name in cls and [b.id for b in cls[name].bases if isinstance(b, ast.Name)] == [base]
        check(ok, f"{name}({base})")
        if not ok:
            return

    def consts(c):
        return {t.id: ast.literal_eval(n.value) for n in c.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    check(consts(cls["Biped12HangStandW3S3tEnvCfg"]) == {"W_TSAT": -40.0, "TSAT_THRESH": 0.6}, f"W3S3t 상수 {consts(cls['Biped12HangStandW3S3tEnvCfg'])}")
    check(consts(cls["Biped12HangStandW3S3tHEnvCfg"]) == {"W_TSAT": -80.0}, f"W3S3tH 상수 {consts(cls['Biped12HangStandW3S3tHEnvCfg'])}")
    check(consts(cls["Biped12HangStandW3S3tDEnvCfg"]) == {"SWING_WIN": (0.25, 0.75)}, f"W3S3tD 상수 {consts(cls['Biped12HangStandW3S3tDEnvCfg'])}")

    def touched(c):
        """__post_init__ 에서 대입하는 속성 경로(self.rewards.x, rw.x.params …)."""
        out = []
        for n in ast.walk(c):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Attribute):
                        out.append(ast.unparse(t))
        return sorted(out)
    check(touched(cls["Biped12HangStandW3S3tEnvCfg"]) == ["self.rewards.w3_torque_sat"], f"W3S3t 가 바꾸는 것: {touched(cls['Biped12HangStandW3S3tEnvCfg'])} (보상 항 하나 추가)")
    check(touched(cls["Biped12HangStandW3S3tHEnvCfg"]) == [], "W3S3tH 는 상수만 다르다")
    check(touched(cls["Biped12HangStandW3S3tDEnvCfg"]) == ["rw.w3_height.params", "rw.w3_periodic.params"], f"W3S3tD 가 바꾸는 것: {touched(cls['Biped12HangStandW3S3tDEnvCfg'])} (두 보상 항의 인자)")
    body = "\n".join(ast.unparse(cls[n]) for n in want)
    check(not any(k in body for k in ("self.observations", "self.actions", "self.commands", "self.events", "self.scene", "self.curriculum", "self.terminations")),
          "새 클래스는 관측·행동·명령·이벤트·장면·커리큘럼·종료를 건드리지 않는다(배포 계약 불변)")
    check("rw.w3_periodic = RewTerm(func=_w3.w3_periodic_contact, weight=-1.5)" in src and "rw.w3_height = RewTerm(func=_w3.w3_swing_height, weight=3.0)" in src,
          "W3Base 의 w3_periodic −1.5·w3_height 3.0 (이 시험 [E] 의 가중과 같음)")
    isrc = open(os.path.join(REPO, "rl_walking", "__init__.py"), encoding="utf-8").read()
    names = []
    for n in ast.walk(ast.parse(isrc)):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "_stage" and isinstance(n.iter, ast.Tuple):
            names += [e.value for e in n.iter.elts if isinstance(e, ast.Constant)]
        elif isinstance(n, ast.For) and isinstance(n.target, ast.Tuple) and isinstance(n.iter, ast.Tuple):
            names += [e.elts[0].value for e in n.iter.elts if isinstance(e, ast.Tuple) and e.elts and isinstance(e.elts[0], ast.Constant)]
    check(all(s in names for s in ("W3S3t", "W3S3tH", "W3S3tD")), "등록 목록에 W3S3t·W3S3tH·W3S3tD")
    dup = sorted({s for s in names if names.count(s) > 1})
    check(not dup, f"등록 이름 중복 없음(전체 {len(names)} 개){' — 중복 ' + str(dup) if dup else ''}")
    check(all(f"Biped12HangStand{s}EnvCfg" in cls for s in names if s.startswith("W3S3")), "W3S3* 등록 이름마다 설정 클래스가 있다")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lp", default=os.path.join(REPO, "log_picture")); a = ap.parse_args()
    test_a(); test_b(); test_c(a.lp); test_d(); test_e(); test_f()
    print(f"\n결과: {'전부 통과' if not FAILS else str(len(FAILS)) + ' 건 실패'}")
    for f in FAILS:
        print("  실패:", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
