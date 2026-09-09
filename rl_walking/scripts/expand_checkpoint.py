#!/usr/bin/env python3
"""rsl_rl 체크포인트의 관측 입력을 확장한다 (S8: 245→325, 크리틱 260→340) — 새 입력 열은 0 으로 초기화.

새 관측이 0 이면 확장 전과 **완전히 같은 출력**을 내므로 v4 의 능력을 그대로 물려받고 새 채널만 학습된다.
크리틱 입력은 [policy | critic] 순이라 새 열을 policy 블록 끝(=245 뒤)에 끼운다. 옵티마이저 상태는 비운다(Adam 이 새로 만듦).
사용: python3 rl_walking/scripts/expand_checkpoint.py --src <run>/model_N.pt --dst <run>/model_N_s8exp.pt --extra 80 --policy_dim 245 --critic_extra 15
"""
import argparse
import torch


def expand(sd, policy_dim, extra, critic_extra):
    out = dict(sd)
    for key, insert_at, total_old in (("actor.0.weight", policy_dim, policy_dim), ("critic.0.weight", policy_dim, policy_dim + critic_extra)):
        w = sd[key]
        assert w.shape[1] == total_old, (key, tuple(w.shape), total_old)
        z = torch.zeros(w.shape[0], extra, dtype=w.dtype)
        out[key] = torch.cat([w[:, :insert_at], z, w[:, insert_at:]], dim=1)
    return out


def selftest():
    sd = {"actor.0.weight": torch.randn(8, 245), "critic.0.weight": torch.randn(8, 260), "actor.0.bias": torch.randn(8)}
    o = expand(sd, 245, 80, 15)
    assert tuple(o["actor.0.weight"].shape) == (8, 325) and tuple(o["critic.0.weight"].shape) == (8, 340)
    assert torch.equal(o["actor.0.weight"][:, :245], sd["actor.0.weight"]) and torch.equal(o["critic.0.weight"][:, :245], sd["critic.0.weight"][:, :245])
    assert torch.equal(o["critic.0.weight"][:, 325:], sd["critic.0.weight"][:, 245:])
    # 새 입력 0 → 출력 동일
    x_old = torch.randn(3, 245); x_new = torch.cat([x_old, torch.zeros(3, 80)], 1)
    assert torch.allclose(x_old @ sd["actor.0.weight"].T, x_new @ o["actor.0.weight"].T)
    print("[selftest] expand_checkpoint: 245→325·260→340, 새 입력 0 이면 출력 동일 — PASS")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src"); ap.add_argument("--dst"); ap.add_argument("--extra", type=int, default=80)
    ap.add_argument("--policy_dim", type=int, default=245); ap.add_argument("--critic_extra", type=int, default=15); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    d = torch.load(a.src, map_location="cpu", weights_only=False)
    d["model_state_dict"] = expand(d["model_state_dict"], a.policy_dim, a.extra, a.critic_extra)
    if "optimizer_state_dict" in d:
        d["optimizer_state_dict"] = {"state": {}, "param_groups": d["optimizer_state_dict"]["param_groups"]}
    torch.save(d, a.dst)
    print(f"확장 저장: {a.dst}  actor.0 {tuple(d['model_state_dict']['actor.0.weight'].shape)}  critic.0 {tuple(d['model_state_dict']['critic.0.weight'].shape)}  iter {d.get('iter')}")


if __name__ == "__main__":
    main()
