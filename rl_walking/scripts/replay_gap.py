"""실물 블랙박스 명령 리플레이 — 시뮬-실물 갭 정량 해부 (0803).

실물 첫 march(0802, 61틱)의 '실제 송신 명령'(targets_deg_joint)을 시뮬 플랜트에
그대로 주입하고 관절 응답을 실물 실측(qpos_rad)과 비교한다. 정책 없음 — 순수
플랜트 대조. 변형(variant)별로 돌려 어느 물리 갭이 실물 응답을 가장 잘
설명하는지 판정한다.

사용: isaaclab.sh -p rl_walking/scripts/replay_gap.py --bb <블랙박스.jsonl> \
        --variant nominal|soft|softer|friction_low|tilt_fwd
출력: /tmp/replay_<variant>.npz (q_sim) + 관절별 대조 표 stdout.
"""
import argparse
import json
import math

parser = argparse.ArgumentParser()
parser.add_argument("--bb", type=str, required=True)
parser.add_argument("--variant", type=str, default="nominal")
parser.add_argument("--task", type=str, default="Biped12-Velocity-Stage4V2-March-Play-v0")
from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import rl_walking  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rl_walking.deploy.policy_runner import JOINT_ORDER, ACTION_SCALE, DEFAULT_POSE_RAD  # noqa: E402

# ── 실물 데이터 ──
ticks = [json.loads(l) for l in open(args_cli.bb) if l.strip()]
data = [t for t in ticks if "qpos_rad" in t]
T = len(data)
q_real = np.array([t["qpos_rad"] for t in data])
tgt_real = np.array([[math.radians(t["targets_deg_joint"][j]) for j in JOINT_ORDER]
                     for t in data])
print(f"[리플레이] 실물 {T}틱, variant={args_cli.variant}", flush=True)

# ── env 구성 (1 env, 노이즈·외란 없음) ──
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = 1
env_cfg.observations.policy.enable_corruption = False
env_cfg.events.base_external_force_torque = None
env_cfg.events.push_robot = None
env_cfg.events.reset_base.params["pose_range"] = {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)}
env_cfg.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in
                                                      ("x", "y", "z", "roll", "pitch", "yaw")}

v = args_cli.variant
if v == "soft":       # 게인 60% (컴플라이언스 1차 근사)
    env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = (0.6, 0.6)
    env_cfg.events.actuator_gains.params["damping_distribution_params"] = (0.8, 0.8)
elif v == "softer":   # 게인 30% (실측 무릎 추종비 0.27 대응)
    env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = (0.3, 0.3)
    env_cfg.events.actuator_gains.params["damping_distribution_params"] = (0.6, 0.6)
elif v == "friction_low":
    env_cfg.events.physics_material.params["static_friction_range"] = (0.35, 0.35)
    env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.3, 0.3)
elif v == "tilt_fwd":  # 초기 전방 기울기 2.8° (forward = -Y 규약과 무관하게 pitch로 근사)
    env_cfg.events.reset_base.params["pose_range"]["pitch"] = (0.049, 0.049)

env = gym.make(args_cli.task, cfg=env_cfg)
obs, _ = env.reset()
uenv = env.unwrapped
robot = uenv.scene["robot"]

# 시뮬 관절 인덱스 ↔ JOINT_ORDER 매핑
sim_names = robot.data.joint_names
idx = [sim_names.index(j) for j in JOINT_ORDER]

# 초기 상태를 실물 t0 관절각으로
jp = robot.data.default_joint_pos.clone()
jp[0, idx] = torch.tensor(q_real[0], dtype=torch.float32, device=jp.device)
robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))

# 액션 역산: env가 DEFAULT + 0.25·act 를 적용하므로 act = (tgt − DEFAULT)/0.25
DEF = np.asarray(DEFAULT_POSE_RAD)
q_sim = np.zeros_like(q_real)
q_sim[0] = q_real[0]
for t in range(1, T):
    act = (tgt_real[t] - DEF) / ACTION_SCALE
    a = torch.tensor(act, dtype=torch.float32, device=uenv.device).unsqueeze(0)
    obs, _, term, trunc, _ = env.step(a)
    q_sim[t] = robot.data.joint_pos[0, idx].cpu().numpy()
    if bool(term[0]):
        print(f"[리플레이] {t}틱에 시뮬 종료(낙상) — 이후 데이터 무효", flush=True)
        q_sim = q_sim[:t + 1]
        break

Ts = q_sim.shape[0]
np.savez(f"/tmp/replay_{v}.npz", q_sim=q_sim, q_real=q_real[:Ts], tgt=tgt_real[:Ts])
print(f"\n{'관절':16s} {'실물RMS°':>9s} {'시뮬RMS°':>9s} {'실-시뮬 괴리RMS°':>15s}")
for i, j in enumerate(JOINT_ORDER):
    err_r = math.degrees(float(np.sqrt(np.mean((q_real[:Ts, i] - tgt_real[:Ts, i]) ** 2))))
    err_s = math.degrees(float(np.sqrt(np.mean((q_sim[:, i] - tgt_real[:Ts, i]) ** 2))))
    gap = math.degrees(float(np.sqrt(np.mean((q_real[:Ts, i] - q_sim[:, i]) ** 2))))
    print(f"{j.replace('_joint',''):16s} {err_r:9.2f} {err_s:9.2f} {gap:15.2f}")
env.close()
app.close()
