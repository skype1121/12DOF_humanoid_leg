"""보행 중 발 들기 높이(clearance) 실측 — '바닥을 긁는다'를 수치로.

실물 관찰(승윤님, 0808): march 보행에서 발이 바닥을 긁는 느낌이 강하다.
코드를 보면 원인이 분명하다 — **발 높이를 보상하는 항이 어디에도 없다.**
보행은 feet_air_time(공중 시간, 문턱 0.4s)과 게이트 클록으로만 만들어지므로
정책은 '떠 있기'만 하면 되고 높이 들 이유가 없다. 최소한으로 들어 스치듯
지나가는 것이 보상상 최적이다.

이 스크립트는 스윙 구간의 발바닥 높이를 분포로 뽑아 그 사실을 확정한다.

실행:
  ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_foot_clearance.py --headless \
      --task Biped12-Velocity-Stage4V2-March-DRv5-v0 \
      --load_run <런> --checkpoint <ckpt> --cmd_x 0.3 --out_tag march_v5b
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Biped12-Velocity-Stage4V2-March-DRv5-v0")
parser.add_argument("--steps", type=int, default=750)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--cmd_x", type=float, default=0.0)
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
OUT = os.path.join(REPO, "log_picture", f"발들기_{args_cli.out_tag}.txt")
FEET = ("left_ankle_r_joint", "right_ankle_r_joint")
#: 발목 롤축에서 발바닥까지 (STL 실측, 0808): 52.5mm
SOLE_BELOW_ANKLE_M = 0.0525

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = agent_cfg.seed
env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.cmd_x, args_cli.cmd_x)
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
cs = uenv.scene.sensors["contact_forces"]
feet_body = [robot.body_names.index(n) for n in FEET]
feet_sensor = [cs.body_names.index(n) for n in FEET]

obs = env.get_observations()
h_log, c_log = [], []
for _ in range(args_cli.steps):
    with torch.inference_mode():
        obs, _, _, _ = env.step(policy(obs))
    # 발바닥 높이 = 발목 롤축 높이 − 52.5mm (지면 기준)
    h = (robot.data.body_pos_w[:, feet_body, 2]
         - uenv.scene.env_origins[:, 2:3]) - SOLE_BELOW_ANKLE_M
    f = torch.norm(cs.data.net_forces_w[:, feet_sensor, :], dim=-1)
    h_log.append(h.cpu().numpy())
    c_log.append((f > 5.0).cpu().numpy())

H = np.stack(h_log) * 1000.0       # mm, (S, N, 2)
C = np.stack(c_log)                # 접촉 여부
swing = ~C                         # 스윙 = 비접촉

lines = []
def P(s=""):
    print(s)
    lines.append(s)

P("=" * 70)
P(f"발 들기 높이 — {args_cli.task}")
P(f"체크포인트: {resume_path}")
P(f"{args_cli.num_envs} env × {args_cli.steps/50:.1f}s, cmd_x={args_cli.cmd_x} m/s")
P("=" * 70)
P(f"스윙 비율: {100.0*swing.mean():.1f}% (나머지는 접지)")
if swing.sum() < 50:
    P("스윙 표본이 거의 없다 — 제자리 서기이거나 발을 안 뗀다.")
else:
    sh = H[swing]
    P(f"스윙 중 발바닥 높이 [mm] — 평균 {sh.mean():6.1f}, 중앙 {np.median(sh):6.1f}")
    P(f"                          p90 {np.percentile(sh,90):6.1f}, "
      f"최대 {sh.max():6.1f}")
    # 스윙 사이클별 '최고점'이 실제 들기 높이다 (평균은 이착지에 희석된다)
    peaks = []
    for e in range(H.shape[1]):
        for k in range(2):
            s = swing[:, e, k]
            i = 0
            while i < len(s):
                if s[i]:
                    j = i
                    while j < len(s) and s[j]:
                        j += 1
                    if j - i >= 3:
                        peaks.append(H[i:j, e, k].max())
                    i = j
                else:
                    i += 1
    if peaks:
        pk = np.array(peaks)
        P("")
        P(f"■ 스윙 사이클별 최고점 [mm] ({len(pk)}회) — **이것이 '들기 높이'**")
        P(f"    평균 {pk.mean():.1f} · 중앙 {np.median(pk):.1f} · "
          f"p10 {np.percentile(pk,10):.1f} · 최소 {pk.min():.1f}")
        low = 100.0 * (pk < 20.0).mean()
        P(f"    20mm 미만으로 지나간 스윙: {low:.1f}%  "
          f"(사람 보행 clearance ~15~25mm, 실물 바닥 요철 고려 시 최소선)")
        P("")
        if np.median(pk) < 25.0:
            P("→ 긁는다. 발 높이를 보상하는 항이 없어 '최소로 들어 스치기'가")
            P("   보상상 최적이 된 결과다. clearance 보상 + 클록 완화가 처방.")
        else:
            P("→ 들기 높이는 확보돼 있다. 긁힘은 다른 원인(지형·자세)일 수 있다.")
P("=" * 70)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n리포트: {OUT}")
env.close()
simulation_app.close()
