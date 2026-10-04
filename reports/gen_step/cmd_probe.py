import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); p.add_argument("--task", default="Biped12-HangStand-W2h-v0"); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True; app = AppLauncher(a).app
import torch, gymnasium as gym, json
sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
import isaaclab_tasks  # noqa
import rl_walking  # noqa
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=256)
env = gym.make(a.task, cfg=cfg); U = env.unwrapped
obs, _ = env.reset()
ct = U.command_manager.get_term("base_velocity")
names = list(U.observation_manager.active_terms["policy"]); dims = U.observation_manager.group_obs_term_dim["policy"]
off = {}; o = 0
for n, d in zip(names, dims):
    sz = 1
    for v in (d if isinstance(d, (tuple, list)) else (d,)): sz *= int(v)
    off[n] = (o, o + sz); o += sz
out = []
for i in range(600):
    act = torch.zeros(U.num_envs, U.action_manager.total_action_dim, device=U.device)
    obs, r, te, tr, ext = env.step(act)
    if i % 100 == 99:
        c = ct.vel_command_b; po = obs["policy"]; a0, b0 = off["velocity_commands"]
        vc = po[:, a0:b0]
        walk = c[:, 0].abs() >= 0.03
        out.append(dict(i=i, walk_frac=float(walk.float().mean()), wz_abs_walk=float(c[walk, 2].abs().mean()) if walk.any() else None,
                        wz_abs_all=float(c[:, 2].abs().mean()), standing_env=float(ct.is_standing_env.float().mean()),
                        heading_env=float(ct.is_heading_env.float().mean()) if hasattr(ct, "is_heading_env") else None,
                        obs_vc_shape=b0 - a0, obs_vc_last3_absmean=[float(x) for x in vc[:, -3:].abs().mean(0)],
                        obs_vc_first3_absmean=[float(x) for x in vc[:, :3].abs().mean(0)],
                        scale=str(getattr(U.observation_manager._group_obs_term_cfgs["policy"][names.index("velocity_commands")], "scale", None))))
open(os.path.join(os.environ.get("OUT", "/tmp"), "cmd_probe.json"), "w").write(json.dumps(out, indent=1))
os._exit(0)
