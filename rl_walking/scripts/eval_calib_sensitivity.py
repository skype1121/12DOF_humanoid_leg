"""캘리브레이션 오차 민감도 — 영점·IMU가 틀어져도 정책이 버티는가.

**왜 이 측정이 실기 전에 필요한가.**
관측 노이즈는 전부 `Unoise`(매 스텝 새로 뽑히는 제로평균)다. 관절각 ±0.57°,
중력벡터 ±0.05. 그런데 **캘리브레이션 오차는 에피소드 내내 유지되는 상수
편향**이고, 그런 항은 학습 어디에도 없다. 제로평균 노이즈는 '걸러내는 법'을
가르치지 '치우친 기준에 적응하는 법'을 가르치지 않는다.

영점(stand_zero_deg)이 δ 틀어졌을 때 실제로 일어나는 일:
    joint = dir × (motor − stand_zero)  이므로
    관측은 δ 만큼 적게 읽히고, 명령은 δ 만큼 더 나간다.
    → 정책은 '목표에 있다'고 믿는데 로봇은 목표+δ 자세로 선다.
이 스크립트는 그 두 효과를 함께 주입한다(관측 −δ, 구동 +δ).
IMU 장착 오차는 중력벡터를 θ 만큼 회전시켜 흉내낸다.

실행:
  ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_calib_sensitivity.py --headless \
      --task Biped12-Velocity-Stage4V2-Stand-V8-v0 \
      --load_run 2026-08-08_21-24-48 --checkpoint model_69897.pt --out_tag v8
"""
import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Biped12-Velocity-Stage4V2-Stand-V8-v0")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--out_tag", type=str, default="")
parser.add_argument("--zero_deg", type=float, default=None,
                    help="이 값을 주면 단일 조건만 측정 (환경 재생성 회피)")
parser.add_argument("--imu_deg", type=float, default=0.0)
parser.add_argument("--sign", choices=("mixed", "pos", "neg"), default="mixed",
                    help="영점 오차 부호. 실물 재현 검증엔 한 방향으로 고정")
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
OUT = os.path.join(REPO, "log_picture", f"캘리민감도_{args_cli.out_tag}.txt")

#: 관측 레이아웃 (단일프레임 49 × 이력 5 = 245). stage4_env_cfg PolicyCfg 순서:
#: ang_vel3 · gravity3 · cmd3 · joint_pos12 · joint_vel12 · actions12 · clock2 · contact2
GRAV_SLICE = slice(15, 30)     # 3 × 5
QPOS_SLICE = slice(45, 105)    # 12 × 5

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

lines = []
def P(s=""):
    print(s)
    lines.append(s)


def run(zero_deg, imu_deg, seed=0):
    """영점 오차 zero_deg[°], IMU 장착 오차 imu_deg[°]에서 낙상률·드리프트."""
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = seed
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    env_cfg.episode_length_s = 60.0

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                            device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    N = args_cli.num_envs

    dz = math.radians(zero_deg)
    th = math.radians(imu_deg)
    # 영점 오차 부호는 env마다 반반 — 한쪽으로만 주면 최악/최선만 본다
    if args_cli.sign == "pos":
        sign = torch.ones(N, device=dev)
    elif args_cli.sign == "neg":
        sign = -torch.ones(N, device=dev)
    else:
        sign = torch.where(torch.arange(N, device=dev) % 2 == 0,
                           torch.ones(N, device=dev), -torch.ones(N, device=dev))
    dz_v = (dz * sign).unsqueeze(1)

    obs = env.get_observations()
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    death = torch.full((N,), args_cli.steps, dtype=torch.long, device=dev)
    pg_log = []
    def _pol(o):
        """관측이 TensorDict 로 올 수 있다 — 정책 텐서만 꺼낸다 (0809 실측)."""
        try:
            return o["policy"]
        except Exception:
            return o

    for _ in range(args_cli.steps):
        o = obs.clone()
        t = _pol(o)
        # ① 영점 오차: 관측이 δ 만큼 적게 읽힌다
        if dz != 0.0:
            t[:, QPOS_SLICE] = t[:, QPOS_SLICE] - dz_v
        # ② IMU 장착 오차: 중력벡터를 피치축으로 θ 회전
        if th != 0.0:
            g = t[:, GRAV_SLICE].reshape(N, 5, 3).clone()
            gx, gz = g[..., 0].clone(), g[..., 2].clone()
            g[..., 0] = gx * math.cos(th) - gz * math.sin(th)
            g[..., 2] = gx * math.sin(th) + gz * math.cos(th)
            t[:, GRAV_SLICE] = g.reshape(N, 15)
        with torch.inference_mode():
            a = policy(o)
            # ③ 영점 오차: 명령이 δ 만큼 더 나간다 (ACTION_SCALE 0.25 보정)
            if dz != 0.0:
                a = a + dz_v / 0.25
            obs, _, _, _ = env.step(a)
        pg_log.append(robot.data.projected_gravity_b.clone().cpu().numpy())
        term = uenv.termination_manager.terminated
        death[alive & term] = len(pg_log) - 1
        alive &= ~term

    pg = np.stack(pg_log)
    fwd = np.degrees(np.arctan2(pg[:, :, 0], -pg[:, :, 2]))
    d = death.cpu().numpy()
    fell = int((d < args_cli.steps).sum())
    # 초기 3s 전방 추세 — analyze_fall.py 와 같은 지표라 실물과 직접 비교된다
    W = min(150, fwd.shape[0])
    t = np.arange(W) / 50.0
    sl = []
    for e in range(N):
        n = min(W, max(int(d[e]), 2))
        if n < 10:
            continue
        A = np.vstack([t[:n], np.ones(n)]).T
        sl.append(np.linalg.lstsq(A, fwd[:n, e], rcond=None)[0][0])
    slope = float(np.mean(sl)) if sl else float("nan")
    env.close()
    return fell, float(np.nanmean(fwd[-1])), float(np.nanmax(np.abs(fwd[-1]))), slope


P("=" * 74)
P(f"캘리브레이션 오차 민감도 — {args_cli.task}")
P(f"체크포인트: {resume_path}")
P(f"{args_cli.num_envs} env × {args_cli.steps/50:.1f}s, 명령 전무")
P("=" * 74)
P("영점 오차 = 관측 −δ + 구동 +δ (실물에서 stand_zero가 틀어졌을 때의 실제 효과)")
P("IMU 오차 = 중력벡터 피치축 회전")
P("")
P(f"{'영점δ':>7}{'IMUθ':>7}{'낙상':>10}{'종료기울임':>14}{'최대':>12}{'초기3s추세':>13}")
P("(실물 5차 v6b 무개입 낙하 실측: 초기 3s 전방추세 +4.06°/s)")
P("-" * 60)
# 한 프로세스에서 환경을 여러 번 만들면 극단적으로 느리다(0809 실측: 조건 1개에
# 16분+). 조건당 프로세스 1개로 돌리는 것이 정공법 — --zero_deg 를 주면 단일 조건.
CASES = ([(args_cli.zero_deg, args_cli.imu_deg)] if args_cli.zero_deg is not None
         else [(0.0, 0.0), (2.0, 0.0), (3.0, 0.0), (0.0, 2.0), (0.0, 3.0)])
for zd, id_ in CASES:
    f, m, mx, sl = run(zd, id_)
    hit = "  ★실물재현" if 3.0 <= sl <= 5.5 else ""
    P(f"{zd:>6.1f}°{id_:>6.1f}°{f:>6}/{args_cli.num_envs:<4}"
      f"{m:>+13.1f}°{mx:>11.1f}°{sl:>+11.2f}{hit}")
P("-" * 60)
P("")
P("판정 기준: 실물 영점은 매달림 육안 정렬로 잡았고 기구학 검증을 한 적이 없다.")
P("2~3° 오차는 충분히 가능한 범위다. 그 구간에서 낙상이 발생하면 실기 전에")
P("영점 검증 또는 오프셋 랜덤화 재학습이 선결이다.")
P("=" * 74)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"\n리포트: {OUT}")
simulation_app.close()
