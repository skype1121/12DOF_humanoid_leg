"""HTURN2 실험 플래그 동치 검증 (1006) — PC 전용(torch 필요, GPU·Isaac 불필요). 젯슨 브리지 selftest 와 별개.

  python3 rl_walking/deploy/selftest_hturn2_equiv.py [--ref a9bdc83c]

검사:
  [A] 플래그 꺼짐 = 기존 러너와 비트 동일: --ref 커밋의 policy_runner_stage4.py(`git show`)를 별도 모듈로 올려,
      같은 결정적 스텁 정책(245/320/512 관측, 관측 필터 β 유무, 리셋 포함)과 등록 실모델(있으면)로
      반환 목표각·관측(last_obs) 전 틱 비트 비교.
  [B] 플래그 켜짐 = 시뮬 함수와 같은 출력: rl_walking/mdp_hang.py 의 cadence_of_cmd·gait_phase_update·gait_u·
      install_turn_assist 와 mdp_stage4.py 의 _gait_phase·gait_clock 을 **소스 그대로**(AST 추출) torch CPU 로 실행 —
      ManagerBasedRLEnv.step 순서(process_action→buf+1→명령→interval 이벤트→관측)를 흉내 낸 1-env 가짜 env 에서
      같은 명령열(서기·걷기·문턱 0.03·속도 포화·음수 vx·wz ±0.3·중간 리셋)로 관측 클록과 hip_r 잔차를 대조.
  [C] HeadingHold(kp 1, 상한 0.3) = eval_walk.py 식(atan2 랩·clamp).
  [D] 브리지 코어(BridgeCore.tick) 플래그 꺼짐 = --ref 커밋 브리지와 비트 동일(실모델, HeadingHold 활성).
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import math
import os
import subprocess
import sys
import tempfile
import types

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rl_walking.deploy import hturn2 as H  # noqa: E402
from rl_walking.deploy.policy_runner import DEFAULT_POSE_RAD, JOINT_ORDER  # noqa: E402
from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4  # noqa: E402


# ------------------------------------------------------------------ [A] 플래그 꺼짐 비트 동일
def _load_ref_runner(ref):
    src = subprocess.run(["git", "-C", REPO, "show", f"{ref}:rl_walking/deploy/policy_runner_stage4.py"],
                         check=True, capture_output=True, text=True).stdout
    assert "hturn2" not in src, f"{ref} 의 러너에 이미 hturn2 가 있다 — 기준 커밋 확인"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tmp.write(src); tmp.close()
    spec = importlib.util.spec_from_file_location("rl_walking.deploy._ref_runner_head", tmp.name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    os.unlink(tmp.name)
    return mod.PolicyRunnerStage4


def _stub(dim, seed):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((12, dim)).astype(np.float32) * np.float32(0.05)

    def net(obs):
        return np.tanh(W @ np.asarray(obs, np.float32)).astype(np.float32) * np.float32(2.0)
    return net


def _inputs(k, rng):
    gyro = rng.standard_normal(3).astype(np.float32) * np.float32(0.2)
    grav = np.float32([0.02, -0.03, -0.999]) + rng.standard_normal(3).astype(np.float32) * np.float32(0.01)
    vx = float(np.float32([0.0, 0.05, 0.1, 0.15, 0.03, 0.029][(k // 37) % 6]))
    cmd = (vx, 0.0, float(np.float32(0.3 * math.sin(k * 0.05))))
    qpos = DEFAULT_POSE_RAD + rng.standard_normal(12).astype(np.float32) * np.float32(0.05)
    qvel = rng.standard_normal(12).astype(np.float32) * np.float32(0.5)
    contact = np.float32([float(k % 7 != 0), float(k % 5 != 0)])
    return gyro, grav, cmd, qpos, qvel, contact


def check_flag_off(ref):
    Ref = _load_ref_runner(ref)
    cases = []
    for dim in (245, 320, 512):
        for kw in ({}, {"act_obs_ema_beta": 0.27, "jvel_obs_ema_beta": 0.25}, {"gait_freq": 1.1}):
            cases.append((f"stub{dim}{'+filt' if 'act_obs_ema_beta' in kw else ''}{'+1.1Hz' if 'gait_freq' in kw else ''}",
                          dict(model_path=_stub(dim, dim), obs_dim=dim, **kw)))
    # 실모델(있으면): g21a export(320, 관측 필터 계약) — 브리지 등록 경로와 같은 인자
    g21a = os.path.join(REPO, "logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported")
    if os.path.isdir(g21a):
        cases.append(("g21a(실모델)", dict(model_path=g21a, gait_freq=1.1, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)))
    n_ticks = 0
    for name, kw in cases:
        a, b = PolicyRunnerStage4(**kw), Ref(**kw)
        a.reset(); b.reset()
        rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
        for k in range(400):
            if k == 211:
                a.reset(); b.reset()
            oa = a.step(*_inputs(k, rng_a)); ob = b.step(*_inputs(k, rng_b))
            assert oa.dtype == ob.dtype and np.array_equal(oa.view(np.uint32), ob.view(np.uint32)), f"[A] {name} 틱 {k}: 목표각 비트 불일치"
            assert np.array_equal(a.last_obs.view(np.uint32), b.last_obs.view(np.uint32)), f"[A] {name} 틱 {k}: 관측 비트 불일치"
            assert a.last_ff is None, f"[A] {name}: 플래그 꺼짐인데 피드포워드 기록"
            n_ticks += 1
    return f"[A] 플래그 꺼짐 비트 동일: {len(cases)} 경우 × 400틱(중간 리셋) = {n_ticks}틱, 목표각·관측 전 비트 일치 (기준 {ref})"


# ------------------------------------------------------------------ [B] 플래그 켜짐 = 시뮬 함수
def _extract(path, names):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in out}
    assert not missing, f"{path}: 함수 없음 {missing}"
    return ast.unparse(ast.Module(body=out, type_ignores=[]))


def _sim_namespace():
    import torch
    ns = {"torch": torch, "math": math, "os": os, "GAIT_FREQ_HZ": 1.4,
          "_base_cfg": lambda: types.SimpleNamespace(name="robot"),
          "_walk_mode": lambda *a, **k: (_ for _ in ()).throw(AssertionError("omni 경로 호출 — W2q 는 비-omni"))}
    exec(_extract(os.path.join(REPO, "rl_walking/mdp_hang.py"),
                  ["gait_u", "cadence_of_cmd", "gait_phase_update", "install_turn_assist"]), ns)
    exec(_extract(os.path.join(REPO, "rl_walking/mdp_stage4.py"), ["_gait_phase", "gait_clock"]), ns)
    return ns


class _FakeEnv:
    def __init__(self, torch):
        self.num_envs, self.device, self.step_dt = 1, "cpu", 0.005 * 4
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self._cmd = torch.zeros(1, 3)
        term = types.SimpleNamespace(pre_delay_hook=None)
        self.action_manager = types.SimpleNamespace(get_term=lambda n: term)
        self.command_manager = types.SimpleNamespace(get_command=lambda n: self._cmd)
        asset = types.SimpleNamespace(joint_names=list(JOINT_ORDER),
                                      data=types.SimpleNamespace(root_ang_vel_w=torch.zeros(1, 3)))
        self.scene = {"robot": asset}
        self.term = term


def _cmd_seq(n):
    rng = np.random.default_rng(3)
    seq = []
    for k in range(n):
        seg = (k // 45) % 9
        vx = [0.0, 0.05, 0.10, 0.15, 0.03, 0.029, 0.30, -0.10, 0.12][seg]
        wz = float(rng.uniform(-0.3, 0.3)) if seg != 0 else float(rng.uniform(-0.05, 0.05))
        seq.append((float(np.float32(vx)), 0.0, float(np.float32(wz))))
    return seq


def check_flag_on():
    import torch
    ns = _sim_namespace()
    os.environ.pop("TURN_ALPHA", None)
    cfg = H.HTURN2_DEFAULT
    env = _FakeEnv(torch)
    ns["install_turn_assist"](env, None, alpha0=cfg.alpha, kappa=cfg.kappa, lead_ticks=cfg.lead_ticks, walk_min=cfg.walk_min)
    hook = env.term.pre_delay_hook
    zero = torch.zeros(1, 12)
    r = PolicyRunnerStage4(model_path=lambda o: np.zeros(12, np.float32), obs_dim=245, gait_freq=1.1, hturn2=cfg)
    r.reset()
    seq = _cmd_seq(900)
    reset_at = {400}
    iL, iR = JOINT_ORDER.index("left_hip_r_joint"), JOINT_ORDER.index("right_hip_r_joint")
    q = DEFAULT_POSE_RAD.copy()
    prev_ff = None
    n_ff = n_nz = 0
    max_clk = 0.0
    clk_bit = 0
    for n, cmd in enumerate(seq):
        # --- 시뮬: (n>0) 직전 관측의 행동 처리 = hook → buf+1(또는 리셋 0) → 명령 → interval → 관측
        if n > 0:
            tgt = hook(zero.clone())
            sim_ff = (tgt[0, iL].item(), tgt[0, iR].item())
            others = torch.cat([tgt[0, :iL], tgt[0, iL + 1:iR], tgt[0, iR + 1:]])
            assert torch.all(others == 0), "[B] 시뮬 hook 이 hip_r 밖을 건드림"
            assert prev_ff is not None and np.float32(sim_ff[0]) == np.float32(prev_ff[0]) and np.float32(sim_ff[1]) == np.float32(prev_ff[1]), \
                f"[B] 틱 {n - 1} 피드포워드 불일치: 시뮬 {sim_ff} vs 러너 {prev_ff}"
            n_ff += 1
            n_nz += int(sim_ff[0] != 0.0 or sim_ff[1] != 0.0)
            env.episode_length_buf += 1
        if n in reset_at:
            env.episode_length_buf[:] = 0
            r.reset()
        env._cmd = torch.tensor([cmd], dtype=torch.float32)
        ns["gait_phase_update"](env, None)
        sim_clk = ns["gait_clock"](env, 1.1)[0].numpy()
        # --- 러너: 같은 명령으로 1틱
        out = r.step((0, 0, 0), (0, 0, -1), cmd, q, np.zeros(12), (1.0, 1.0))
        run_clk = r.last_obs[225 + 4 * 2: 225 + 5 * 2]           # gait_clock 블록 최신 프레임
        assert np.float32(r._h2_clock.u) == np.float32(env._gu[0].item()), f"[B] 틱 {n} 위상 u 불일치 {r._h2_clock.u} vs {env._gu[0].item()}"
        assert np.float32(r._h2_clock.f) == np.float32(env._gf[0].item()), f"[B] 틱 {n} cadence f 불일치"
        d = float(np.abs(run_clk.astype(np.float64) - sim_clk.astype(np.float64)).max())
        max_clk = max(max_clk, d)
        clk_bit += int(np.array_equal(run_clk.view(np.uint32), sim_clk.astype(np.float32).view(np.uint32)))
        assert d <= 2.4e-7, f"[B] 틱 {n} 클록 관측 차 {d:.3e} > 2 ulp"
        prev_ff = r.last_ff
        q = out
    assert n_nz > 300, f"[B] 0 아닌 피드포워드 표본이 적다({n_nz}) — 명령열 점검"
    return (f"[B] 플래그 켜짐 = 시뮬 소스 실행: {len(seq)}틱(중간 리셋 1회) 위상 u·cadence f 비트 일치, "
            f"hip_r 잔차 {n_ff}틱 비트 일치(0 아님 {n_nz}), 클록 관측 비트 일치 {clk_bit}/{len(seq)}·최대 차 {max_clk:.1e}")


# ------------------------------------------------------------------ [C] HeadingHold
def check_heading():
    import torch
    from rl_walking.deploy.policy_runner_stage4 import HeadingHold
    cfg = H.HTURN2_DEFAULT
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(5000):
        ref, yaw = rng.uniform(-3.1, 3.1, 2)
        if abs(abs(ref - yaw) - math.pi) < 1e-3:
            continue
        hh = HeadingHold(kp=cfg.hh_kp, wz_limit=cfg.hh_wz_max, ref=ref)
        a = hh.update(yaw)
        e = torch.tensor(ref - yaw, dtype=torch.float64)
        b = float((cfg.hh_kp * torch.atan2(torch.sin(e), torch.cos(e))).clamp(-cfg.hh_wz_max, cfg.hh_wz_max))
        worst = max(worst, abs(a - b))
        assert abs(a) <= cfg.hh_wz_max + 1e-12
    assert worst < 1e-12, f"[C] HeadingHold 차 {worst}"
    return f"[C] HeadingHold(kp {cfg.hh_kp}, |wz|≤{cfg.hh_wz_max}) = eval_walk 식: 5000표본 최대 차 {worst:.1e}"


# ------------------------------------------------------------------ [D] 브리지 코어 플래그 꺼짐 비트 동일
def check_bridge_off(ref):
    """--ref 커밋의 rl_bridge_node.py 를 같은 디렉터리 임시 파일로 올려(REPO 경로 동일) BridgeCore.tick 출력 비교.
    러너는 양쪽 모두 현재 파일(플래그 꺼짐) — 러너 동치는 [A] 가 따로 본다. 실모델이 있는 키만."""
    src = subprocess.run(["git", "-C", REPO, "show", f"{ref}:jetson/scripts/rl_bridge_node.py"],
                         check=True, capture_output=True, text=True).stdout
    assert "HTURN2" not in src
    tmp = os.path.join(REPO, "jetson", "scripts", "_ref_bridge_head_tmp.py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        spec = importlib.util.spec_from_file_location("_ref_bridge_head_tmp", tmp)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    finally:
        os.unlink(tmp)
    spec = importlib.util.spec_from_file_location("_new_bridge", os.path.join(REPO, "jetson", "scripts", "rl_bridge_node.py"))
    new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
    assert not new.EXPERIMENTAL_HTURN2 and set(new.CHECKPOINTS) == set(old.CHECKPOINTS)
    done = []
    for name in ("stand_crouch_g21a", "walk", "drv2", "stand_v18b"):
        d = old.CHECKPOINTS.get(name)
        if d is None or not (os.path.exists(os.path.join(d, "policy.pt")) or os.path.exists(os.path.join(d, "policy.onnx"))):
            continue
        a, b = new.BridgeCore(name), old.BridgeCore(name)
        for c in (a, b):
            c.reset(); c.cmd = (0.1, 0.0, 0.0)
        rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
        for k in range(300):
            if k == 150:
                for c in (a, b):
                    c.cmd = (0.05, 0.0, 0.2)
            yaw = 0.4 + 0.01 * k
            quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
            ia, ib = _inputs(k, rng_a), _inputs(k, rng_b)
            oa = a.tick(ia[0], quat, ia[3], ia[4], ia[5]); ob = b.tick(ib[0], quat, ib[3], ib[4], ib[5])
            assert np.array_equal(np.asarray(oa).view(np.uint32), np.asarray(ob).view(np.uint32)), f"[D] {name} 틱 {k} 불일치"
            assert np.array_equal(a.runner.last_obs.view(np.uint32), b.runner.last_obs.view(np.uint32))
        done.append(name)
    assert done, "[D] 비교할 실모델 없음(logs/ 확인)"
    return f"[D] 브리지 코어 플래그 꺼짐 비트 동일: {', '.join(done)} × 300틱(HeadingHold 활성·wz 명령 전환) 목표각·관측 일치 (기준 {ref})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="a9bdc83c", help="플래그 꺼짐 비교 기준 커밋(HTURN2 이전 러너)")
    a = ap.parse_args()
    failed = 0
    for fn in (lambda: check_flag_off(a.ref), check_flag_on, check_heading, lambda: check_bridge_off(a.ref)):
        try:
            print("[PASS]", fn())
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("[FAIL]", type(e).__name__, e)
    print("hturn2 equiv:", "ALL PASS" if failed == 0 else f"{failed} FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
