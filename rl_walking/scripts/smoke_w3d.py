"""W3D 학습 환경 스모크 + 시작점 기준선(학습 없음, 1007).

  isaaclab.sh -p rl_walking/scripts/smoke_w3d.py --headless --ckpt <절대 경로 model_*.pt> [--num_envs 256 --level 2 --p_dist 1.0 --seconds 22]

W3D 태스크를 그대로(학습 외란·지연 1–5·DR 켬) 만들고 정책(기본 W3S3d-15800)을 넣어 첫 에피소드만 집계:
  - 환경 무결성: 보상 유한, 외란 발생/취소(STAND)/정지 명령/회복 창·회복 착지·복귀 사건 수, 무외란 env 의 새 항 합 = 0.
  - 기준선: 걷는 중 외란이 실제로 걸린 env vs 안 걸린 env 의 낙상률(충격/램프·방향별), 회복 착지 cap·gain 평균.
결과: stdout + log_picture/smoke_w3d_<tag>.json. 산출물로 판정(isaaclab.sh exit 0 신뢰 금지).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W3D-v0")
p.add_argument("--ckpt", required=True)
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--seconds", type=float, default=21.0)
p.add_argument("--level", type=int, default=2, help="충격 크기 단계 고정(0–3)")
p.add_argument("--p_dist", type=float, default=1.0)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--tag", default="s3d15800")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import torch, gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import rl_walking  # noqa: F401,E402
from rl_walking import w3_dist as WD, w3 as W3  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

env_cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
env_cfg.scene.num_envs = a.num_envs; env_cfg.seed = a.seed
env_cfg.curriculum = None                                   # 단계 고정(커리큘럼 진급 없음)
L = WD.LEVELS[a.level]
env_cfg.events.disturbance.params = dict(env_cfg.events.disturbance.params, p_dist=float(a.p_dist), levels=(L, L, L, L))
env = gym.make(a.task, cfg=env_cfg)
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
U = env.unwrapped; dev = U.device; N = U.num_envs; dt = U.step_dt
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(a.ckpt); policy = runner.get_inference_policy(device=dev)
obs = venv.get_observations()
obs = obs[0] if isinstance(obs, tuple) else obs

first = torch.ones(N, dtype=torch.bool, device=dev); fell = torch.zeros(N, dtype=torch.bool, device=dev)
newsum = torch.zeros(N, device=dev); nonfinite = 0
names = ("w3d_cap", "w3d_rplace", "w3d_rstride", "w3d_ret", "w3d_sret")
tid = [U.reward_manager.active_terms.index(n) for n in names]
applied_any = torch.zeros(N, dtype=torch.bool, device=dev); imp_any = torch.zeros(N, dtype=torch.bool, device=dev)
dir_first = torch.full((N,), -1, dtype=torch.long, device=dev); stop_iss = torch.zeros(N, dtype=torch.bool, device=dev)
ret_paid = torch.zeros(N, dtype=torch.bool, device=dev); sret_paid = torch.zeros(N, dtype=torch.bool, device=dev)
rec_td = torch.zeros(N, device=dev); cap_s = torch.zeros(N, device=dev); gain_s = torch.zeros(N, device=dev); canceled = torch.zeros(N, dtype=torch.bool, device=dev)
t0w = time.time()
DIAG = []; rrun_max = torch.zeros(N, device=dev)
steps = int(round(a.seconds / dt))
for i in range(steps):
    with torch.inference_mode():
        act = policy(obs)
    obs, rew, dones, _ = venv.step(act)
    obs = obs[0] if isinstance(obs, tuple) else obs
    nonfinite += int((~torch.isfinite(rew)).sum())
    s = U._w3d
    st = U.reward_manager._step_reward[:, tid].sum(1) if hasattr(U.reward_manager, "_step_reward") else torch.zeros(N, device=dev)
    newsum += st * first
    with torch.no_grad():
        applied_any |= s["applied"].any(1) & first
        imp_any |= s["ep_imp"] & first
        canceled |= s["cancel"].any(1) & first
        stop_iss |= s["stop_on"] & first
        ret_paid |= (s["ev_ret"] > 0) & first
        sret_paid |= (s["ev_sret"] > 0) & first
        rec_td += s["td_rec"].float().sum(1) * first
        cap_s += (s["cap_df"] * s["td_rec"].float()).sum(1) * first
        gain_s += (s["gain"] * s["td_rec"].float()).sum(1) * first
        k1 = torch.argmax(s["applied"].float(), 1)
        dir_first = torch.where((dir_first < 0) & s["applied"].any(1), s["dir_i"].gather(1, k1.unsqueeze(1)).squeeze(1), dir_first)
        hs_ = U._hang; tnow = hs_["t"]
        wk = s["r_on"] & ~s["stop_want"] & first & (tnow >= s["r_tend"] + 0.3) & (tnow <= s["r_t0"] + 4.0) & W3._ge(U)
        if bool(wk.any()):
            e_, _ = W3._v_err(U, box=True)
            pg_ = U.scene["robot"].data.projected_gravity_b
            tl_ = torch.rad2deg(torch.acos((-pg_[:, 2]).clamp(-1, 1))); w_ = U.scene["robot"].data.root_ang_vel_b[:, :2].norm(dim=1)
            DIAG.append(torch.stack([e_[wk], tl_[wk], w_[wk]], 1).cpu())
        rrun_max = torch.maximum(rrun_max, s["r_run"])
    d = dones.bool()
    term = U.termination_manager.terminated.bool()
    fell |= d & term & first
    first &= ~d
print(f"[smoke_w3d] {steps} 틱 {time.time() - t0w:.0f} s", flush=True)
ok_env = int((~first).sum())
dist_w = applied_any; nd = ~applied_any & ~canceled
res = {"task": a.task, "ckpt": a.ckpt, "level": a.level, "F": L, "p_dist": a.p_dist, "seed": a.seed, "N": N, "seconds": a.seconds,
       "reward_nonfinite": nonfinite, "applied_env": int(dist_w.sum()), "impulse_env": int(imp_any.sum()), "canceled_stand_env": int((canceled & ~applied_any).sum()),
       "fall_dist": [int((fell & dist_w).sum()), int(dist_w.sum())], "fall_nodist": [int((fell & nd).sum()), int(nd.sum())],
       "fall_by_dir": {WD.DIR_NAMES[j]: [int((fell & dist_w & (dir_first == j)).sum()), int((dist_w & (dir_first == j)).sum())] for j in range(4)},
       "fall_imp_env": [int((fell & imp_any).sum()), int(imp_any.sum())],
       "stop_issued": int(stop_iss.sum()), "sret_paid": int(sret_paid.sum()), "ret_paid": int(ret_paid.sum()),
       "rec_td": float(rec_td.sum()), "cap_mean": float(cap_s.sum() / rec_td.sum().clamp(min=1)), "gain_mean": float(gain_s.sum() / rec_td.sum().clamp(min=1)),
       "newterm_sum_nodist_absmax": float(newsum[~applied_any].abs().max()) if bool((~applied_any).any()) else 0.0,
       "newterm_sum_dist_mean": float(newsum[applied_any].mean()) if bool(applied_any.any()) else 0.0,
       "step_reward_available": hasattr(U.reward_manager, "_step_reward")}
if DIAG:
    D = torch.cat(DIAG)
    q = lambda c: [round(float(torch.quantile(D[:, c], x)), 3) for x in (0.1, 0.5, 0.9)]
    res["ret_window_ticks"] = int(D.shape[0]); res["ret_verr_p10_50_90"] = q(0); res["ret_tilt_p10_50_90"] = q(1); res["ret_w_p10_50_90"] = q(2)
    res["ret_each_ok_frac"] = [round(float((D[:, 0] <= 0.05).float().mean()), 3), round(float((D[:, 1] <= 4.0).float().mean()), 3), round(float((D[:, 2] <= 0.5).float().mean()), 3)]
res["r_run_max_p90"] = float(torch.quantile(rrun_max[applied_any], 0.9)) if bool(applied_any.any()) else 0.0
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True)
with open(os.path.join(REPO, "log_picture", f"smoke_w3d_{a.tag}.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
app.close()
