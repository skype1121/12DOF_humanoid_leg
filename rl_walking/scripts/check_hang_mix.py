"""G21h(매달림 시작 혼합) 환경의 기전 확인 — 학습 전에 돌리는 반사실 단위시험 (1008, reports/gen_w3/AIR_START_1008.md §8).

정책 없이 액션 0(= 기본자세 명령)으로 첫 에피소드만 본다. 확인하는 것:
  ① 매달림 에피소드 비율 ≈ p_hang  ② 매달림 = 스폰이 높고·방출 전 발 접촉 없음·떠 있는 동안 높이와 기울기가 유지됨
  ③ 지면 에피소드 = 처음부터 양발 접촉  ④ 매달림 에피소드는 지지가 남은 동안 회복 게이트가 닫혀 있음  ⑤ 외란은 지지가 풀리고 1 s 뒤부터
산출물: log_picture/check_hang_mix_<tag>.txt (판정은 이 파일로 — isaaclab.sh 는 오류를 삼킨다).

  PYTHONPATH=$PWD ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/check_hang_mix.py --headless --num_envs 256 --out_tag g21h
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

REPO = "/home/ryu/humanoid_leg_test1"
sys.path.insert(0, REPO)
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Biped12-HangStand-G21h-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=14.0)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out_tag", type=str, default="g21h")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import rl_walking  # noqa: E402,F401
import rl_walking.tasks_extra_1008  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from rl_walking import mdp_hang  # noqa: E402

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
env = gym.make(args_cli.task, cfg=env_cfg)
u = env.unwrapped
obs, _ = env.reset()
st = mdp_hang._state(u)
robot = u.scene["robot"]
N, dev = u.num_envs, u.device
hang = st["mix_hang"].clone() > 0.5
t_hold = st["t_hold"].clone(); t_rel = st["t_rel"].clone(); z0 = robot.data.root_pos_w[:, 2].clone()
dst_kind = st["dst_kind"].clone(); dst_t0 = st["dst_t0"].clone()
pre = u.cfg.events.hang_mix_reset.params
first = torch.ones(N, dtype=torch.bool, device=dev)          # 첫 에피소드가 아직 진행 중
air_contact = torch.zeros(N, dtype=torch.bool, device=dev)   # 방출 시작 전 발 접촉(어느 발이든)
gate_open_sup = torch.zeros(N, dtype=torch.bool, device=dev)  # 지지가 남은 동안 게이트 열림
dist_early = torch.zeros(N, dtype=torch.bool, device=dev)    # 지지 풀리고 1 s 전에 외란 힘
zmin = z0.clone(); zmax = z0.clone(); tilt_max = torch.zeros(N, device=dev)
both_by = torch.full((N,), -1.0, device=dev)                 # 양발 첫 접촉 시각
died = torch.zeros(N, dtype=torch.bool, device=dev); t_die = torch.full((N,), -1.0, device=dev)
act = torch.zeros(N, u.action_manager.total_action_dim, device=dev)
steps = int(round(args_cli.seconds / u.step_dt))
fid = mdp_hang._feet_body_ids(u)
for i in range(steps):
    _, _, term, trunc, _ = env.step(act)
    t = (i + 1) * u.step_dt
    done = (term | trunc)
    newly = done & first
    died |= newly & term; t_die = torch.where(newly & term, torch.full_like(t_die, t), t_die)
    first &= ~done
    f = u.scene.sensors["contact_forces"].data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0
    anyc = f.any(1); both = f.all(1)
    in_air = first & hang & (t < t_hold - 0.05)
    air_contact |= in_air & anyc
    sup = first & hang & (st["released_t"] < 0.0)
    gate_open_sup |= sup & (st["rec_open"] > 0.5)
    dist_early |= first & hang & (t < t_hold + t_rel + float(pre["dist_shift_s"]) - 0.05) & (st["dst_F_now"] > 0.0)
    z = robot.data.root_pos_w[:, 2]
    zmin = torch.where(in_air, torch.minimum(zmin, z), zmin); zmax = torch.where(in_air, torch.maximum(zmax, z), zmax)
    pg = robot.data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos((-pg[:, 2]).clamp(-1, 1)))
    tilt_max = torch.where(in_air, torch.maximum(tilt_max, tilt), tilt_max)
    both_by = torch.where(first & both & (both_by < 0), torch.full_like(both_by, t), both_by)

g = ~hang
nh, ng = int(hang.sum()), int(g.sum())
dz = z0[hang] - z0[g].median()
q = lambda x, p: float(x.float().quantile(p)) if x.numel() else float("nan")
lines = [f"G21h 기전 확인 — {args_cli.task} env {N} seed {args_cli.seed} · 액션 0(기본자세 명령) · {args_cli.seconds:.0f} s · 설정 {pre}"]
ok = True


def chk(name, cond, detail):
    global ok
    ok &= bool(cond)
    lines.append(f"[{'PASS' if cond else 'FAIL'}] {name} — {detail}")


chk("① 매달림 에피소드 비율", abs(nh / N - float(pre["p_hang"])) <= 0.12, f"{nh} / {N} ({100 * nh / N:.1f} %, 설정 {100 * float(pre['p_hang']):.0f} %)")
chk("② 매달림 스폰 높이(지면 에피소드 중앙 대비)", nh > 0 and q(dz, 0.0) >= float(pre["clearance_range"][0]) - 0.02 and q(dz, 1.0) <= float(pre["clearance_range"][1]) + 0.02,
    f"+{q(dz, 0.0):.3f} ~ +{q(dz, 1.0):.3f} m (설정 {pre['clearance_range']}, 지면 스폰의 z 무작위 ±0.01 포함)")
chk("② 방출 시작 전 발 접촉 없음", nh > 0 and float(air_contact[hang].float().mean()) <= 0.02, f"접촉한 개체 {int(air_contact[hang].sum())} / {nh}")
chk("② 떠 있는 동안 높이 유지(최대 − 최소)", nh > 0 and q((zmax - zmin)[hang], 0.95) <= 0.04, f"중앙 {q((zmax - zmin)[hang], 0.5) * 100:.1f} cm · p95 {q((zmax - zmin)[hang], 0.95) * 100:.1f} cm")
chk("② 떠 있는 동안 기울기", nh > 0 and q(tilt_max[hang], 0.5) <= 5.0 and q(tilt_max[hang], 0.95) <= 10.0, f"최대 기울기 중앙 {q(tilt_max[hang], 0.5):.1f}° · p95 {q(tilt_max[hang], 0.95):.1f}°")
chk("③ 지면 에피소드는 0.5 s 안에 양발 접촉", ng > 0 and float(((both_by[g] >= 0) & (both_by[g] <= 0.5)).float().mean()) >= 0.95,
    f"{int(((both_by[g] >= 0) & (both_by[g] <= 0.5)).sum())} / {ng}")
chk("④ 지지가 남은 동안 회복 게이트 닫힘(매달림)", int(gate_open_sup[hang].sum()) == 0, f"열린 적 있는 개체 {int(gate_open_sup[hang].sum())} / {nh}")
k = dst_kind[hang] > 0
tmin = torch.where(k, dst_t0[hang], torch.full_like(dst_t0[hang], 1e9)).min(dim=1).values
need = t_hold[hang] + t_rel[hang] + float(pre["dist_shift_s"]) + 2.0 - 1e-3
chk("⑤ 외란 일정이 지지 풀린 뒤로 밀림(매달림)", bool((tmin >= need).all()) and int(dist_early[hang].sum()) == 0,
    f"첫 외란 시각 − (유지 + 방출 + 1 s) 최소 {float((tmin - (t_hold[hang] + t_rel[hang] + float(pre['dist_shift_s'])))[tmin < 1e8].min()) if bool((tmin < 1e8).any()) else float('nan'):.2f} s(원래 일정의 하한 2.0) · 일찍 걸린 개체 {int(dist_early[hang].sum())}")
td = both_by[hang]
lines.append(f"[참고] 매달림: 유지 {q(t_hold[hang], 0.0):.1f}–{q(t_hold[hang], 1.0):.1f} s · 방출 {q(t_rel[hang], 0.0):.1f}–{q(t_rel[hang], 1.0):.1f} s · 양발 첫 접촉 = 방출 시작 뒤 중앙 {q((td - t_hold[hang])[td >= 0], 0.5):.2f} s"
             f" · 액션 0 으로 첫 에피소드 종료(낙상) {int(died[hang].sum())} / {nh} (지면 {int(died[g].sum())} / {ng}) — 정책 없는 참고값")
lines.append("ALL PASS" if ok else "FAILED")
out = os.path.join(REPO, "log_picture", f"check_hang_mix_{args_cli.out_tag}.txt")
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
env.close()
app.close()
