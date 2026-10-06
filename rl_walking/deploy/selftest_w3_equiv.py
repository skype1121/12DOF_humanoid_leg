"""W3 실험 플래그 동치 검증 (1006) — PC 전용(torch 필요, GPU·Isaac 불필요). 젯슨 브리지 selftest 와 별개.

  python3 rl_walking/deploy/selftest_w3_equiv.py [--ref 35b69b4a]

검사:
  [A] 플래그 꺼짐 = 기존 러너와 비트 동일: --ref 커밋의 policy_runner_stage4.py(`git show`)를 별도 모듈로 올려 같은 결정적 스텁
      (245/320/512·관측 필터 β 유무·1.1 Hz)·HTURN2 켜짐 경로·등록 실모델(g21a, 있으면)로 목표각·관측 전 틱 비트 비교(중간 리셋 포함).
  [B] 플래그 켜짐 = 시뮬 함수와 같은 출력: rl_walking/w3.py 의 _ge·w3_gait_update 와 mdp_stage4.py 의 _gait_phase·gait_clock 을
      **소스 그대로**(AST 추출) torch CPU 로 실행 — ManagerBasedRLEnv.step 순서(buf+1 → 리셋 → 명령(_ge_cmd) → interval → 관측)를 흉내 낸
      1-env 가짜 env 와 러너에 같은 gait_enable 명령열을 넣는다. 명령열 = 켜짐 길이 1~50틱 전수(반주기 중간·경계·직전·직후 끄기를 모두 덮음)
      × 꺼짐 길이 {1,2,3,11,30}틱 + 에피소드 첫 틱(fresh) 켜짐 + 켜진 채 리셋. 위상 u·주파수 f·ge·ge_obs 비트, 래치 전이 틱 목록,
      관측 클록(sin/cos ≤ 2 ulp)을 대조.
  [C] 브리지 코어(BridgeCore.tick) 플래그 꺼짐 = --ref 커밋 브리지와 비트 동일(실모델).
  [D] 서기(G21a) → W3 상태 이어붙이기(carry_runner_state, 브리지 미연결): 이어받은 직후 관측이 원 러너를 계속 돌린 관측과
      클록 블록만 빼고 비트 동일, 클록 블록 = (0,0), 레이아웃·기본자세·필터 불일치 거부.
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

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from rl_walking.deploy import hturn2 as H2  # noqa: E402
from rl_walking.deploy import w3_deploy as W  # noqa: E402
from rl_walking.deploy.policy_runner import DEFAULT_POSE_RAD  # noqa: E402
from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4  # noqa: E402

G21A = os.path.join(REPO, "logs/rsl_rl/biped12_stage4/2026-10-02_21-23-08_g21a/exported")
CLK = slice(225 + 4 * 2, 225 + 5 * 2)            # gait_clock 블록 최신 프레임(245/320/512 공통 오프셋)
CLK_BLOCK = slice(225, 235)


def _git_show(ref, path):
    return subprocess.run(["git", "-C", REPO, "show", f"{ref}:{path}"], check=True, capture_output=True, text=True).stdout


# ------------------------------------------------------------------ [A] 플래그 꺼짐 비트 동일
def _load_ref_runner(ref):
    src = _git_show(ref, "rl_walking/deploy/policy_runner_stage4.py")
    assert "_w3_mod" not in src, f"{ref} 의 러너에 이미 W3 가 있다 — 기준 커밋 확인"
    tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=os.path.join(REPO, "rl_walking", "deploy"))
    tmp.write(src); tmp.close()
    try:
        name = "rl_walking.deploy." + os.path.splitext(os.path.basename(tmp.name))[0]
        spec = importlib.util.spec_from_file_location(name, tmp.name)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.unlink(tmp.name)
    return mod.PolicyRunnerStage4


def _stub(dim, seed):
    rng = np.random.default_rng(seed)
    Wm = rng.standard_normal((12, dim)).astype(np.float32) * np.float32(0.05)

    def net(obs):
        return np.tanh(Wm @ np.asarray(obs, np.float32)).astype(np.float32) * np.float32(2.0)
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
    cases.append(("stub320+hturn2", dict(model_path=_stub(320, 9), obs_dim=320, gait_freq=1.1, hturn2=H2.HTURN2_DEFAULT)))
    if os.path.isdir(G21A):
        cases.append(("g21a(실모델)", dict(model_path=G21A, gait_freq=1.1, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)))
    n_ticks = 0
    for name, kw in cases:
        a, b = PolicyRunnerStage4(**kw), Ref(**kw)
        assert a.w3 is None and a._w3_clock is None
        a.reset(); b.reset()
        rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
        for k in range(400):
            if k == 211:
                a.reset(); b.reset()
            oa = a.step(*_inputs(k, rng_a)); ob = b.step(*_inputs(k, rng_b))
            assert oa.dtype == ob.dtype and np.array_equal(oa.view(np.uint32), ob.view(np.uint32)), f"[A] {name} 틱 {k}: 목표각 비트 불일치"
            assert np.array_equal(a.last_obs.view(np.uint32), b.last_obs.view(np.uint32)), f"[A] {name} 틱 {k}: 관측 비트 불일치"
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
    ns = {"torch": torch, "math": math, "os": os, "GAIT_FREQ_HZ": 1.4}
    exec(_extract(os.path.join(REPO, "rl_walking/w3.py"), ["_ge", "w3_gait_update"]), ns)
    exec(_extract(os.path.join(REPO, "rl_walking/mdp_stage4.py"), ["_gait_phase", "gait_clock"]), ns)
    return ns


class _FakeEnv:
    """install_w3_cmd 가 만드는 상태(_ge·_ge_cmd·_ge_obs) + w3_gait_update 가 읽는 것만."""

    def __init__(self, torch):
        self.num_envs, self.device = 1, "cpu"
        self.step_dt = 0.005 * 4                        # ManagerBasedEnv.step_dt = sim.dt × decimation
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self._ge = torch.zeros(1, dtype=torch.bool)
        self._ge_cmd = torch.zeros(1, dtype=torch.bool)
        self._ge_obs = torch.zeros(1)


def _want_seq():
    """(want, reset_here) 목록. 켜짐 길이 1~50 전수(1.1 Hz 반주기 = 22.7틱 → 끄기 요청이 반주기의 모든 위치에 온다)."""
    seq = [(True, True)]                                  # 에피소드 첫 틱 켜짐(fresh: 즉시 반영)
    seq += [(True, False)] * 40
    offs = (1, 2, 3, 11, 30)
    for i, L1 in enumerate(range(1, 51)):
        seq += [(False, False)] * offs[i % len(offs)]
        seq += [(True, False)] * L1
        if L1 == 25:
            seq += [(True, True)] + [(True, False)] * 5   # 켜진 채 리셋(fresh 규칙 재적용)
        if L1 == 40:
            seq += [(False, True)] + [(False, False)] * 3  # 꺼짐 요청 + 리셋
    seq += [(False, False)] * 60
    return seq


def check_flag_on():
    import torch
    ns = _sim_namespace()
    os.environ.pop("W3_CLOCK_DIAG", None)
    cfg = W.W3_DEFAULT
    env = _FakeEnv(torch)
    r = PolicyRunnerStage4(model_path=lambda o: np.zeros(12, np.float32), obs_dim=320, gait_freq=1.1, w3=cfg)
    seq = _want_seq()
    q = DEFAULT_POSE_RAD.copy()
    sim_ev, max_clk, clk_bit, n_on = [], 0.0, 0, 0
    prev_ge = None
    for n, (want, rst) in enumerate(seq):
        # 시뮬: buf+1 → (리셋) → 명령 리샘플(_ge_cmd) → interval(w3_gait_update) → 관측(gait_clock)
        if n > 0:
            env.episode_length_buf += 1
        if rst:
            env.episode_length_buf[:] = 0
            r.reset()
        env._ge_cmd = torch.tensor([want])
        ns["w3_gait_update"](env, None, f_step=cfg.f_step, ramp_s=cfg.ramp_s)
        sim_clk = ns["gait_clock"](env, 1.1)[0].numpy()
        ge_sim = bool(env._ge[0])
        if prev_ge is not None and ge_sim != prev_ge and not rst:
            sim_ev.append((n, "on" if ge_sim else "off"))
        prev_ge = ge_sim
        # 러너
        r.set_gait_enable(want)
        out = r.step((0, 0, 0), (0, 0, -1), (0, 0, 0), q, np.zeros(12), (1.0, 1.0))
        c = r._w3_clock
        assert c.ge == ge_sim, f"[B] 틱 {n} ge {c.ge} vs 시뮬 {ge_sim}"
        assert np.float32(c.u).view(np.uint32) == np.float32(env._gu[0].item()).view(np.uint32), f"[B] 틱 {n} u {c.u} vs {env._gu[0].item()}"
        assert np.float32(c.f) == np.float32(env._gf[0].item()), f"[B] 틱 {n} f"
        assert np.float32(c.ge_obs).view(np.uint32) == np.float32(env._ge_obs[0].item()).view(np.uint32), \
            f"[B] 틱 {n} ge_obs {c.ge_obs} vs {env._ge_obs[0].item()}"
        run_clk = r.last_obs[CLK]
        d = float(np.abs(run_clk.astype(np.float64) - sim_clk.astype(np.float64)).max())
        max_clk = max(max_clk, d)
        clk_bit += int(np.array_equal(run_clk.view(np.uint32), sim_clk.astype(np.float32).view(np.uint32)))
        assert d <= 2.4e-7, f"[B] 틱 {n} 클록 관측 차 {d:.3e} > 2 ulp"
        n_on += int(ge_sim)
        q = out
    # 래치 전이: 러너 진단 기록(리셋 구간 포함)과 시뮬 전이(리셋 틱 제외)가 같아야 한다 — 리셋 틱의 fresh 반영은 러너도 전이로 적는다
    run_ev = _collect_events(seq)
    assert sim_ev == run_ev, f"[B] 래치 전이 다름\n시뮬 {sim_ev[:10]}\n러너 {run_ev[:10]}"
    offs = [n for n, k in sim_ev if k == "off"]
    # 끄기는 반드시 반주기 경계 틱: 그 틱의 직전 위상 u_prev 와 u_prev+du 가 0.5 또는 1 을 건넌 틱이어야 함 → 시뮬 비트 일치로 보장
    return (f"[B] 플래그 켜짐 = 시뮬 소스 실행: {len(seq)}틱(리셋 {sum(1 for _, x in seq if x)}회, ge 1 {n_on}틱) "
            f"u·f·ge·ge_obs 비트 일치, 래치 전이 {len(sim_ev)}건 동일(끄기 {len(offs)}건 전부 반주기 경계), "
            f"클록 관측 비트 일치 {clk_bit}/{len(seq)}·최대 차 {max_clk:.1e}")


def _collect_events(seq):
    """러너만 따로 같은 명령열로 돌려 진단 이벤트(리셋 사이) 수집 — 리셋 틱 자체의 fresh 반영은 시뮬 쪽과 같이 제외."""
    r = PolicyRunnerStage4(model_path=lambda o: np.zeros(12, np.float32), obs_dim=320, gait_freq=1.1, w3=W.W3_DEFAULT)
    ev, q = [], DEFAULT_POSE_RAD.copy()
    prev = None
    for n, (want, rst) in enumerate(seq):
        if rst:
            r.reset()
        r.set_gait_enable(want)
        r.step((0, 0, 0), (0, 0, -1), (0, 0, 0), q, np.zeros(12), (1.0, 1.0))
        ge = r._w3_clock.ge
        if prev is not None and ge != prev and not rst:
            ev.append((n, "on" if ge else "off"))
        prev = ge
    return ev


# ------------------------------------------------------------------ [C] 브리지 코어 플래그 꺼짐 비트 동일
def check_bridge_off(ref):
    src = _git_show(ref, "jetson/scripts/rl_bridge_node.py")
    assert "EXPERIMENTAL_W3" not in src
    tmp = os.path.join(REPO, "jetson", "scripts", "_ref_bridge_w3_tmp.py")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(src)
    try:
        spec = importlib.util.spec_from_file_location("_ref_bridge_w3_tmp", tmp)
        old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    finally:
        os.unlink(tmp)
    spec = importlib.util.spec_from_file_location("_new_bridge_w3", os.path.join(REPO, "jetson", "scripts", "rl_bridge_node.py"))
    new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
    assert not new.EXPERIMENTAL_W3 and not new.W3_CKPTS and set(new.CHECKPOINTS) == set(old.CHECKPOINTS)
    done = []
    for name in ("stand_crouch_g21a", "stand_c5kr_g21a_g24ra", "walk", "stand_v18b"):
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
            assert np.array_equal(np.asarray(oa).view(np.uint32), np.asarray(ob).view(np.uint32)), f"[C] {name} 틱 {k} 불일치"
        done.append(name)
    assert done, "[C] 비교할 실모델 없음(logs/ 확인)"
    return f"[C] 브리지 코어 플래그 꺼짐 비트 동일: {', '.join(done)} × 300틱 (기준 {ref})"


# ------------------------------------------------------------------ [D] G21a → W3 상태 이어붙이기
def check_carry():
    model = G21A if os.path.isdir(G21A) else _stub(320, 3)
    kw = dict(obs_dim=320, gait_freq=1.1, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
    src = PolicyRunnerStage4(model_path=model, **kw)
    # dst 정책 자리는 W3 exported 가 없어 같은 G21a(또는 스텁)를 쓴다 — 상태 이전 경로 검증이지 정책 성능이 아님
    dst = PolicyRunnerStage4(model_path=model, w3=W.W3_DEFAULT, **kw)
    src.reset(); dst.reset()
    rng = np.random.default_rng(13)
    for k in range(100):
        x = _inputs(k, rng)
        src.step(x[0], x[1], (0, 0, 0), x[3], x[4], x[5])
    W.carry_runner_state(src, dst)
    assert np.all(dst._hist["gait_clock"]._buf == 0.0) and not dst._w3_clock.ge and dst._tick == src._tick
    worst_other = 0
    for k in range(100, 103):
        x = _inputs(k, rng)
        src.step(x[0], x[1], (0, 0, 0), x[3], x[4], x[5])
        dst.set_gait_enable(False)
        dst.step(x[0], x[1], (0, 0, 0), x[3], x[4], x[5])
        a, b = src.last_obs.copy(), dst.last_obs.copy()
        assert np.all(b[CLK_BLOCK] == 0.0), "[D] W3 서기 클록 블록이 (0,0) 아님"
        a[CLK_BLOCK] = 0.0; b[CLK_BLOCK] = 0.0
        if k == 100:
            assert np.array_equal(a.view(np.uint32), b.view(np.uint32)), "[D] 이어받은 첫 관측이 클록 밖에서 다름"
        worst_other = max(worst_other, int(np.count_nonzero(a != b)))
    # 거부 경로
    bad = [PolicyRunnerStage4(model_path=_stub(245, 1), obs_dim=245, w3=W.W3_DEFAULT),
           PolicyRunnerStage4(model_path=model, w3=W.W3_DEFAULT, default_pose_rad=DEFAULT_POSE_RAD + np.float32(0.01), **kw),
           PolicyRunnerStage4(model_path=model, w3=W.W3_DEFAULT, obs_dim=320, gait_freq=1.1),
           PolicyRunnerStage4(model_path=model, **kw)]
    for d in bad:
        try:
            W.carry_runner_state(src, d)
            raise AssertionError("[D] 계약 다른 dst 이어받기 허용")
        except ValueError:
            pass
    return (f"[D] G21a→W3 이어받기({'g21a 실모델' if model is G21A else '스텁'}): 첫 관측 클록 밖 비트 동일·클록 (0,0)·ge 0·틱 유지, "
            f"이후 2틱 차이 원소 ≤{worst_other}(정책 출력이 클록 차로 갈라짐 — 의도), 레이아웃·기본자세·필터·비 W3 거부 4종")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="35b69b4a", help="플래그 꺼짐 비교 기준 커밋(W3 플래그 이전)")
    a = ap.parse_args()
    failed = 0
    for fn in (lambda: check_flag_off(a.ref), check_flag_on, lambda: check_bridge_off(a.ref), check_carry):
        try:
            print("[PASS]", fn())
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print("[FAIL]", type(e).__name__, e)
    print("w3 equiv:", "ALL PASS" if failed == 0 else f"{failed} FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
