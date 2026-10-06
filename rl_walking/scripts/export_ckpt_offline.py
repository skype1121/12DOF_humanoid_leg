"""체크포인트 → <run>/exported/{policy.pt, policy.onnx, SOURCE_CHECKPOINT} — **시뮬(GPU) 없이** play.py 와 같은 내보내기.

play.py 는 env 를 띄운 뒤 IsaacLab isaaclab_rl.rsl_rl.exporter 의 export_policy_as_jit/onnx(policy=runner.alg.policy,
normalizer=actor_obs_normalizer) 를 부른다. 여기서는 같은 exporter 파일을 경로로 직접 읽고(패키지 __init__ 은 pxr 를 요구해
Kit 밖에서 import 불가), 같은 rsl_rl ActorCritic 을 run 의 params/agent.yaml 로 만들어 체크포인트를 strict 로 싣는다.
GPU 가 학습으로 꽉 찼을 때(1007: play.py Warp OOM) 쓰는 대체 경로 — 산출 형식은 같다(검증: 기존 exported 와 jit 구조 비교).

사용: python.sh rl_walking/scripts/export_ckpt_offline.py <run 디렉토리> <model_N.pt> [--ref <기존 exported/policy.pt>]
검증: 체크포인트 actor(float64 수동 MLP) vs jit vs onnx — 무작위 관측 256개 최대 차이 < 1e-4. 결과 파일 md5 출력.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys

import torch
import yaml


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("ckpt")
    ap.add_argument("--ref", default=None, help="형식 비교용 기존 exported/policy.pt")
    ap.add_argument("--exporter", default="/home/ryu/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py")
    a = ap.parse_args()
    d = os.path.abspath(a.run_dir)
    ck = os.path.join(d, a.ckpt)
    agent = yaml.safe_load(open(os.path.join(d, "params", "agent.yaml")))
    pc = dict(agent["policy"])
    assert pc.pop("class_name") == "ActorCritic", pc
    assert not pc.get("actor_obs_normalization") and not pc.get("critic_obs_normalization"), "정규화 있는 run 은 미지원(play.py 경로 사용)"
    sd = torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"]
    n_act = sd["actor.6.weight"].shape[0] if "actor.6.weight" in sd else None
    din = sd["actor.0.weight"].shape[1]
    dcr = sd["critic.0.weight"].shape[1]
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    groups = agent["obs_groups"]
    assert groups["policy"] == ["policy"] and groups["critic"] == ["policy", "critic"], groups
    obs = TensorDict({"policy": torch.zeros(1, din), "critic": torch.zeros(1, dcr - din)}, batch_size=[1])
    pol = ActorCritic(obs, groups, n_act, **pc)
    pol.load_state_dict(sd, strict=True)
    pol.eval()
    spec = importlib.util.spec_from_file_location("_il_exporter", a.exporter)
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    out = os.path.join(d, "exported")
    normalizer = getattr(pol, "actor_obs_normalizer", None)
    E.export_policy_as_jit(pol, normalizer=normalizer, path=out, filename="policy.pt")
    E.export_policy_as_onnx(pol, normalizer=normalizer, path=out, filename="policy.onnx")
    with open(os.path.join(out, "SOURCE_CHECKPOINT"), "w") as f:
        f.write(a.ckpt + "\n")
    # ---- 대조 ----
    ks = sorted({k.split(".")[1] for k in sd if k.startswith("actor.") and k.endswith(".weight")}, key=int)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, din, generator=g)
    def mlp(dt):
        h = x.to(dt)
        for i, k in enumerate(ks):
            h = h @ sd[f"actor.{k}.weight"].to(dt).T + sd[f"actor.{k}.bias"].to(dt)
            if i < len(ks) - 1:
                h = torch.nn.functional.elu(h)
        return h.double()
    h, h64 = mlp(torch.float32), mlp(torch.float64)
    jit = torch.jit.load(os.path.join(out, "policy.pt"), map_location="cpu")
    with torch.no_grad():
        yj = jit(x)
    ej = float((yj.double() - h).abs().max())
    try:
        import onnxruntime as ort
    except ImportError:          # 로컬에 onnxruntime 없음(1007) — 구조 검사만, 수치 대조는 미검증으로 표시
        ort = None
    if ort is not None:
        s = ort.InferenceSession(os.path.join(out, "policy.onnx"), providers=["CPUExecutionProvider"])
        nm = s.get_inputs()[0].name
        yo = torch.from_numpy(s.run(None, {nm: x.numpy()})[0])
        eo = float((yo.double() - h).abs().max())
        on = f"onnx {eo:.2e} · onnx 입력 {nm} {s.get_inputs()[0].shape}"
    else:                        # 로컬에 onnxruntime 없음(1007) — onnx 순수 파이썬 참조 실행기로 수치 대조(느림: 16개)
        import onnx
        from onnx.reference import ReferenceEvaluator
        m = onnx.load(os.path.join(out, "policy.onnx"))
        onnx.checker.check_model(m)
        nm = m.graph.input[0].name
        ev = ReferenceEvaluator(m)
        eo = max(float(abs(ev.run(None, {nm: x[i:i + 1].numpy()})[0] - yj[i:i + 1].numpy()).max()) for i in range(16))
        dims = [d_.dim_value or d_.dim_param for d_ in m.graph.input[0].type.tensor_type.shape.dim]
        on = f"onnx(참조 실행기, 16개) 대 jit {eo:.2e} · 입력 {nm} {dims}"
    e64 = float((yj.double() - h64).abs().max())
    print(f"관측 {din} → 행동 {tuple(yj.shape)} (|a| 최대 {float(yj.abs().max()):.2f}) · 수동 MLP(float32) 대비 최대 차이 jit {ej:.2e} "
          f"(float64 기준 {e64:.2e} = float32 반올림) · {on}")
    tol = 1e-4 * max(1.0, float(yj.abs().max()) / 10.0)   # 무작위 N(0,1) 관측은 분포 밖이라 |a| 가 수백 — float32 상대 오차 기준
    assert ej < tol and eo < tol, (ej, eo, tol)
    if a.ref:
        r = torch.jit.load(a.ref, map_location="cpu")
        ns = lambda m: sorted((n, tuple(p.shape)) for n, p in m.named_parameters())
        same = [n for n, _ in ns(r)] == [n for n, _ in ns(jit)] and type(r.normalizer).__name__ == type(jit.normalizer).__name__
        print(f"기존 exported 와 jit 구조(매개변수 이름·정규화기 종류) 동일: {same} — 기준 {ns(r)[:1]}… 새 {ns(jit)[:1]}…")
        assert same
    for fn in ("policy.pt", "policy.onnx", "SOURCE_CHECKPOINT"):
        print(f"{_md5(os.path.join(out, fn))}  {os.path.join(out, fn)}")
    print(f"{_md5(ck)}  {ck}")


if __name__ == "__main__":
    sys.exit(main())
