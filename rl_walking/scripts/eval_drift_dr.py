"""DR 분포에서 '전방 드리프트'를 env별로 측정 — 실물 낙하가 시뮬에 있는가.

배경: 세대 승격 판정(낙상률·직립도)은 전부 Play 태스크(DR 꺼짐)에서 나왔다.
학습은 CoM·강성·지연 랜덤화를 켜고 했는데, **그 랜덤화가 실제로 강건성을
만들었는지는 확인한 적이 없다.** 실물 5차 v6b는 초기 3s 전방 추세 +4.06°/s로
1.4초 만에 무너졌다. 그 거동이 DR 분포 안에 존재하는지, 존재한다면 어떤 CoM
값에서 나오는지를 env별로 본다.

실행:
  ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_drift_dr.py --headless \
      --task Biped12-Velocity-Stage4V2-Stand-V7-v0 \
      --load_run 2026-08-04_13-13-49 --checkpoint model_63898.pt \
      --num_envs 128 --steps 250 --out_tag stand_v7_DR
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Biped12-Velocity-Stage4V2-Stand-V7-v0")
parser.add_argument("--steps", type=int, default=250, help="50Hz — 250=5s")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--out_tag", type=str, default="")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import isaaclab_tasks            # noqa: F401,E402
import rl_walking                # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.join(REPO, "log_picture", f"드리프트DR_{args_cli.out_tag}.txt")

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = agent_cfg.seed
# 정지 서기 조건 — 실물 무개입 낙하와 동일하게 명령 전무
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
env_cfg.episode_length_s = 60.0

log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path)
policy = runner.get_inference_policy(device=env.unwrapped.device)

uenv = env.unwrapped
robot = uenv.scene["robot"]
N = args_cli.num_envs

# 각 env가 뽑은 base CoM 오프셋 — 드리프트와의 상관을 보려면 이게 있어야 한다.
com_x = None
try:
    coms = robot.root_physx_view.get_coms().to("cpu").numpy()   # (N*bodies, 7) 또는 (N,bodies,7)
    if coms.ndim == 3:
        com_x = coms[:, 0, 0].copy()
    else:
        com_x = coms.reshape(N, -1, coms.shape[-1])[:, 0, 0].copy()
except Exception as exc:      # 읽기 실패해도 본 측정(드리프트)은 계속
    print(f"[warn] CoM 읽기 실패: {exc}")

obs = env.get_observations()
pg_log = []          # (steps, N, 3)
alive = torch.ones(N, dtype=torch.bool, device=uenv.device)
death_step = torch.full((N,), args_cli.steps, dtype=torch.long, device=uenv.device)

for i in range(args_cli.steps):
    with torch.inference_mode():
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
    pg_log.append(robot.data.projected_gravity_b.clone().cpu().numpy())
    term = uenv.termination_manager.terminated
    newly_dead = alive & term
    death_step[newly_dead] = i
    alive &= ~term

pg = np.stack(pg_log)                       # (S, N, 3)
fwd = np.degrees(np.arctan2(pg[:, :, 0], -pg[:, :, 2]))   # 브리지와 동일 규약
dt = 1.0 / 50.0
t = np.arange(pg.shape[0]) * dt
death = death_step.cpu().numpy()

# env별 초기 3s 전방 추세 — 실물 판정기(analyze_fall.py)와 같은 지표
W = min(int(3.0 / dt), pg.shape[0])
slopes, r2s = np.full(N, np.nan), np.full(N, np.nan)
for e in range(N):
    n = min(W, max(death[e], 2))
    y = fwd[:n, e]
    x = t[:n]
    if n < 10:
        continue
    A = np.vstack([x, np.ones_like(x)]).T
    (s, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    slopes[e] = s
    tot = ((y - y.mean()) ** 2).sum()
    r2s[e] = (1 - res[0] / tot) if len(res) and tot > 1e-12 else 0.0

REAL = 4.06        # 실물 v6b 5차 실측 (analyze_fall.py 확정)
ok = ~np.isnan(slopes)
lines = []
def P(s=""):
    print(s)
    lines.append(s)

P("=" * 74)
P(f"DR 분포 전방 드리프트 — {args_cli.task}")
P(f"체크포인트: {resume_path}")
P(f"{N} env × {args_cli.steps/50:.1f}s, 명령 전무(cmd=0), DR 켜짐")
P("=" * 74)
P(f"낙상(조기 종료): {int((death < args_cli.steps).sum())}/{N} "
  f"({100.0*(death < args_cli.steps).sum()/N:.1f}%)")
P(f"초기 3s 전방 추세 °/s — 평균 {np.nanmean(slopes):+.2f}, "
  f"중앙 {np.nanmedian(slopes):+.2f}, "
  f"p5 {np.nanpercentile(slopes,5):+.2f}, p95 {np.nanpercentile(slopes,95):+.2f}, "
  f"최대 {np.nanmax(slopes):+.2f}")
P("")
P(f"■ 실물 기준선 {REAL:+.2f}°/s 이상으로 전방 드리프트한 env: "
  f"{int((slopes[ok] >= REAL).sum())}/{int(ok.sum())} "
  f"({100.0*(slopes[ok] >= REAL).sum()/max(1,ok.sum()):.1f}%)")
P("   → 이 비율이 0에 가까우면, 실물 거동은 DR 분포 밖이다(랜덤화로 못 덮음).")
P("")
if com_x is not None and ok.sum() > 5:
    c = com_x[:N][ok]
    s = slopes[ok]
    if np.std(c) > 1e-9:
        r = float(np.corrcoef(c, s)[0, 1])
        P(f"■ CoM x vs 전방추세 상관계수 r = {r:+.3f}  "
          f"(CoM x 범위 {c.min():+.4f}~{c.max():+.4f} m)")
        # 실물 추세를 내는 CoM 추정 (선형회귀 역산)
        A = np.vstack([c, np.ones_like(c)]).T
        (m, b), *_ = np.linalg.lstsq(A, s, rcond=None)
        if abs(m) > 1e-6:
            P(f"   선형 적합: 추세 = {m:+.1f}·com_x {b:+.2f}  →  "
              f"실물 {REAL:+.2f}°/s 를 내는 com_x ≈ {(REAL-b)/m:+.4f} m")
            P("   (학습 범위 밖 값이면, 실물 CoM이 학습 분포를 벗어나 있다는 뜻)")
    else:
        P("■ CoM x 분산 0 — 랜덤화가 실제로 적용되지 않았을 수 있음(점검 필요)")
else:
    P("■ CoM 상관 분석 생략 (CoM 읽기 실패 또는 표본 부족)")
P("=" * 74)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
np.savez(os.path.join(REPO, "demo_output", f"drift_dr_{args_cli.out_tag}.npz"),
         fwd=fwd, slopes=slopes, r2=r2s, death=death,
         com_x=(com_x if com_x is not None else np.zeros(0)))
print(f"\n리포트: {OUT}")
env.close()
simulation_app.close()
