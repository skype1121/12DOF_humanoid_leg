# 행동 0(자세 고정) 매달림 수동성 대조: 같은 실행 안에서 env 묶음마다 지지 모형 설정만 바꾼다.
import argparse, sys
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
p = argparse.ArgumentParser(); p.add_argument("--num_envs", type=int, default=512); p.add_argument("--secs", type=float, default=30.0); p.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(p); a = p.parse_args(); app = AppLauncher(a).app
import gymnasium as gym, torch
import rl_walking, rl_walking.tasks_extra_1008  # noqa
from isaaclab_tasks.utils import parse_env_cfg
from rl_walking import mdp_hang
cfg = parse_env_cfg("Biped12-HangStand-G21h2-v0", device=a.device, num_envs=a.num_envs); cfg.seed = 3
cfg.events.hang_mix_reset.params.update({"p_hang": 1.0, "cable_frac": 0.5, "air_hold_range": (60.0, 60.0), "sway_v": 0.1})
cfg.episode_length_s = 100.0
env = gym.make("Biped12-HangStand-G21h2-v0", cfg=cfg); u = env.unwrapped; env.reset(); st = mdp_hang._state(u); rb = u.scene["robot"]
N = u.num_envs; idx = torch.arange(N, device=u.device); g = idx % 5
st["dst_t0"][:] = st["dst_t0"] + 1000.0
# 묶음: 0 = 종전(보정 없음)·피치 구속 / 1 = 보정·피치 구속 / 2 = 보정·피치+요 복원 / 3 = 보정·2 점 구속 없음 / 4 = 종전·2 점 구속 없음
names = ["종전 · 피치 구속", "고정 보정 · 피치 구속", "고정 보정 · 피치 + 요 복원", "고정 보정 · 2 점 구속 없음", "종전 · 2 점 구속 없음"]
st["mid"][:] = ((g == 1) | (g == 2) | (g == 3)).float()
st["tp_yaw_on"][:] = (g == 2).float(); st["yaw0"][:] = mdp_hang._yaw_of(rb.data.root_quat_w); st["tp_yaw_c"][:] = 0.15
st["tp_d"][(g == 3) | (g == 4)] = 0.0
cab = st["cbl_env"].clone()
act = torch.zeros(N, u.action_manager.total_action_dim, device=u.device); R = []; WZ = []; Z = []; alive = torch.ones(N, dtype=torch.bool, device=u.device)
T = int(a.secs / u.step_dt)
for i in range(T):
    _, _, term, trunc, _ = env.step(act); alive &= ~(term | trunc)
    pg = rb.data.projected_gravity_b; R.append(torch.rad2deg(torch.atan2(pg[:, 1], -pg[:, 2])).clone()); WZ.append(rb.data.root_ang_vel_b[:, 2].abs().clone()); Z.append(rb.data.root_pos_w[:, 2].clone())
R = torch.stack(R); WZ = torch.stack(WZ); Z = torch.stack(Z)
lines = [f"행동 0(자세 고정) 매달림 {a.secs:.0f} s · env {N} · 롤 진폭 = 2 s 창의 (최대 − 최소)/2 [°] 중앙 · p90 | 끝 창의 요속도 중앙 [rad/s] | 높이 변화 p90 [cm] | 30 s 안 종료"]
for model, cm in (("줄 길이", cab), ("힘", ~cab)):
    for k in range(5):
        m = cm & (g == k); n = int(m.sum()); row = []
        for t0 in (0, 4, 8, 14, 20, a.secs - 2):
            i0 = int(t0 / u.step_dt); seg = R[i0:i0 + 100][:, m]; amp = (seg.max(0).values - seg.min(0).values) / 2
            row.append(f"{amp.median():.1f}·{amp.quantile(0.9):.1f}")
        lines.append(f"| {model} | {names[k]} | {n} | " + " | ".join(row) + f" | {WZ[-100:][:, m].mean(0).median():.2f} | {((Z[:, m].max(0).values - Z[:, m].min(0).values).quantile(0.9) * 100):.1f} | {int((~alive[m]).sum())} / {n} |")
txt = "\n".join(lines); print(txt, flush=True)
if a.out: open(a.out, "w", encoding="utf-8").write(txt + "\n")
env.close(); app.close()
