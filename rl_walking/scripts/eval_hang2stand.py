"""Hanging-to-Stand 평가 — 지지 스케줄 아래서 고정 Kp 홀드 / 정책의 전이 성공률을 잰다 (0908 저녁).

핵심 지표 = **지지가 0 이 된 뒤 5초 생존율**. 그 외: 접지(양발 동시 접촉) 도달률, 지지 0 도달률(살아서),
낙상률·방향, env별 최대 |롤|·|피치| 평균, 최대 각속도, 종료 시 접촉 상태, 종료 시 관절각.
판정은 산출물 파일로 한다 (CLAUDE.md 5): log_picture/hang2stand_<tag>.txt (+ demo_output/hang2stand_<tag>.npz).

실행 (PYTHONPATH 필수 — eval_drift_dr 함정 0908):
  PYTHONPATH=/home/ryu/humanoid_leg_test1 ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/eval_hang2stand.py --headless \
     --task Biped12-HangStand-S1-v0 --hold --num_envs 64 --out_tag s1_hold            # 고정 Kp150 (액션 0)
  ... --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --out_tag s1_v18b   # 정책 (h3 배포체인)
  옵션: --nominal(DR 끔) --com_fix X --r0 lo,hi --hold_s lo,hi --release_s lo,hi --clearance lo,hi --no_chain
"""
import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip  # noqa: E402

REPO = "/home/ryu/humanoid_leg_test1"
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Biped12-HangStand-S1-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=None, help="기본 = env episode_length_s")
parser.add_argument("--hold", action="store_true", help="정책 없이 액션 0 = DEFAULT_POSE kp150 홀드 (고정 Kp 기준선)")
parser.add_argument("--no_chain", action="store_true", help="배포 체인(러너 슬루→LPF→블렌딩) 끔 — env 원생 액션")
parser.add_argument("--nominal", action="store_true", help="DR 전부 끔 (명목 몸체)")
parser.add_argument("--com_fix", type=float, default=None, help="base_com x 고정 [m]")
parser.add_argument("--usd", type=str, default="rl_walking/assets/usd_final3/biped12.usd")
parser.add_argument("--r0", type=str, default=None); parser.add_argument("--hold_s", type=str, default=None)
parser.add_argument("--release_s", type=str, default=None); parser.add_argument("--clearance", type=str, default=None)
parser.add_argument("--stand_s", type=float, default=5.0, help="지지 0 이후 생존 판정 시간 [s]")
parser.add_argument("--no_push", action="store_true", help="푸시 외란 끔 (전이 능력과 푸시 강건성 분리 — 실물 윈치 시험엔 푸시가 없다)")
parser.add_argument("--no_obs_noise", action="store_true", help="관측 노이즈(enable_corruption) 끔 — 정책 고유 정지도 측정용(학습 노이즈 ±0.2rad/s 자이로 등이 잔떨림을 만드는지 분리)")
parser.add_argument("--lpf_beta", type=float, default=None, help="배포 체인 stand LPF β 덮어쓰기 (기본 deploy_chain.STAND_LPF_BETA=0.27; 작을수록 더 부드러움) — 재학습 없이 스탠바이 흔들림을 잡는 배포측 레버 시험")
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


def _pair(s):
    lo, hi = (float(v) for v in s.split(","))
    return (lo, hi)


agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
env_cfg.seed = agent_cfg.seed
if args_cli.usd:
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd
use_chain = (not args_cli.hold) and (not args_cli.no_chain)
if use_chain and hasattr(env_cfg.actions.joint_pos, "lpf_beta_range"):
    env_cfg.actions.joint_pos.lpf_beta_range = (1.0, 1.0)     # 배포 판정: 필터는 체인 1개만
# 정지 서기 명령 · 요 고정 · 초기 속도 0 (실물 착지는 정지)
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
_vr = env_cfg.events.reset_base.params.get("velocity_range")
if isinstance(_vr, dict):
    for _k in list(_vr):
        _vr[_k] = (0.0, 0.0)
if getattr(env_cfg.events, "reset_robot_joints", None):
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
# 지지 스케줄 오버라이드
_hp = env_cfg.events.hang_support_reset.params
if args_cli.r0: _hp["r0_range"] = _pair(args_cli.r0)
if args_cli.hold_s: _hp["hold_range"] = _pair(args_cli.hold_s)
if args_cli.release_s: _hp["release_range"] = _pair(args_cli.release_s)
if args_cli.clearance: env_cfg.events.reset_base.params["pose_range"]["z"] = _pair(args_cli.clearance)
if args_cli.nominal:
    for _k in ("roll", "pitch"):
        if _k in env_cfg.events.reset_base.params["pose_range"]:
            env_cfg.events.reset_base.params["pose_range"][_k] = (0.0, 0.0)
    if getattr(env_cfg.events, "reset_robot_joints", None):
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
    env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
    env_cfg.events.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
    env_cfg.actions.joint_pos.min_delay_steps = 4
    env_cfg.actions.joint_pos.max_delay_steps = 4
    if hasattr(env_cfg.actions.joint_pos, "lpf_beta_range"):
        env_cfg.actions.joint_pos.lpf_beta_range = (1.0, 1.0)
    if hasattr(env_cfg.actions.joint_pos, "backlash_rad_range"):
        env_cfg.actions.joint_pos.backlash_rad_range = (0.0, 0.0)
    if getattr(env_cfg.events, "base_com", None):
        env_cfg.events.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    if getattr(env_cfg.events, "calib_bias", None):
        for _k in list(env_cfg.events.calib_bias.params):
            if _k.endswith("_rad") and env_cfg.events.calib_bias.params[_k] is not None:
                env_cfg.events.calib_bias.params[_k] = (0.0, 0.0)
        if "imu_lag_steps" in env_cfg.events.calib_bias.params:
            env_cfg.events.calib_bias.params["imu_lag_steps"] = (0, 0)
    for _ev in ("stance_asym", "effort_scale"):
        if getattr(env_cfg.events, _ev, None) is not None:
            setattr(env_cfg.events, _ev, None)
    if getattr(env_cfg.events, "add_base_mass", None):
        _op = env_cfg.events.add_base_mass.params.get("operation", "add")
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = ((1.0, 1.0) if _op == "scale" else (0.0, 0.0))
    if getattr(env_cfg.events, "joint_friction", None):
        env_cfg.events.joint_friction.params["friction_distribution_params"] = (0.0, 0.0)
    if getattr(env_cfg.events, "physics_material", None):
        env_cfg.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
    if getattr(env_cfg.events, "joint_armature", None):
        env_cfg.events.joint_armature.params["armature_distribution_params"] = (1.0, 1.0)
    env_cfg.events.push_robot = None
if args_cli.no_push:
    env_cfg.events.push_robot = None
if args_cli.no_obs_noise:
    env_cfg.observations.policy.enable_corruption = False
if args_cli.com_fix is not None and getattr(env_cfg.events, "base_com", None):
    env_cfg.events.base_com.params["com_range"] = {"x": (args_cli.com_fix, args_cli.com_fix), "y": (0.0, 0.0), "z": (0.0, 0.0)}
T_s = args_cli.seconds if args_cli.seconds else float(env_cfg.episode_length_s)
env_cfg.episode_length_s = T_s + 1.0      # 평가 창 안에서 time_out 이 끼지 않게

env = gym.make(args_cli.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
uenv = env.unwrapped
robot = uenv.scene["robot"]
N = args_cli.num_envs
dev = uenv.device
STEPS = int(round(T_s * 50))

policy = None
resume_path = "(hold: 액션 0)"
if not args_cli.hold:
    log_root = os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name)
    resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=dev)

obs = env.get_observations()
_DEF = robot.data.default_joint_pos.clone()
_LAND = torch.zeros_like(_DEF)
_ASCALE = 0.25
chain_desc = "없음(env 원생 액션)"
if use_chain:
    from rl_walking.deploy.deploy_chain import DeployChain, BLEND_S, STAND_LPF_BETA  # noqa: E402
    from rl_walking.deploy.policy_runner import SLEW_RAD  # noqa: E402
    _term = uenv.action_manager.get_term("joint_pos")
    _chain = DeployChain(_DEF, _LAND, qpos0=robot.data.joint_pos.clone(), stand=True, **({"lpf_beta": args_cli.lpf_beta} if args_cli.lpf_beta is not None else {}))
    _orig = _term.process_actions

    def _hooked(a):
        if torch.any(_term._needs_init):
            _ids = torch.nonzero(_term._needs_init).squeeze(-1)
            _chain.reset(_ids, robot.data.joint_pos)
        out = _chain.step(a)
        _orig((out - _DEF) / _ASCALE)
        _term._raw_actions[:] = a

    _term.process_actions = _hooked
    chain_desc = f"h3 chain: 러너 슬루 {math.degrees(SLEW_RAD):.1f}°/틱 → LPF β{args_cli.lpf_beta if args_cli.lpf_beta is not None else STAND_LPF_BETA} → 블렌딩 {BLEND_S}s"

# ── 기록 (첫 에피소드만 — 종료 후 자동 리셋된 env 의 새 에피소드는 무시) ──
alive = torch.ones(N, dtype=torch.bool, device=dev)
death_step = torch.full((N,), -1, dtype=torch.long, device=dev)
touch_t = torch.full((N,), -1.0, device=dev); rel_t = torch.full((N,), -1.0, device=dev)
max_roll = torch.zeros(N, device=dev); max_pitch = torch.zeros(N, device=dev); max_w = torch.zeros(N, device=dev)
pitch_at_rel = torch.full((N,), float("nan"), device=dev)
pitch_death = torch.full((N,), float("nan"), device=dev); roll_death = torch.full((N,), float("nan"), device=dev)
r_at_death = torch.full((N,), float("nan"), device=dev)
ratio_trace = []; contact_trace = []; z_trace = []; pitch_trace = []
last_q = torch.zeros_like(robot.data.joint_pos); last_contact = torch.zeros(N, 2, device=dev)
_cs = uenv.scene["contact_forces"]; _bn = list(_cs.body_names)
_fl = _bn.index("left_ankle_r_joint"); _fr = _bn.index("right_ankle_r_joint")
# 물리 실현성 (0908 승윤님 "접지 자세가 실물로 될까"): 지지 0 이후 생존 스텝의 관절 토크 피크·RMS·포화(≥95% 한계), 발 간격
_rb = list(robot.data.body_names); _bl, _br = _rb.index("left_ankle_r_joint"), _rb.index("right_ankle_r_joint")
_nj = robot.data.joint_pos.shape[1]; _eff = robot.data.joint_effort_limits
_tau_peak = torch.zeros(_nj, device=dev); _tau_rms2 = torch.zeros(_nj, device=dev); _tau_sat = torch.zeros(_nj, device=dev); _tau_n = 0
_stance_w = []; _stance_dx = []
# 정지도 (0909 승윤님 "착지 후 흔들림 불가"): 지지 0 이후 생존 스텝의 |ω_xy| 평균·|v_xy| 평균·관절속도 RMS·발 들기 횟수(양발 접촉 후 한 발 이탈 전이)
_st_w = torch.zeros(N, device=dev); _st_v = torch.zeros(N, device=dev); _st_qd2 = torch.zeros(N, device=dev); _st_n = torch.zeros(N, device=dev)
_lift_cnt = torch.zeros(N, device=dev); _prev_both = torch.zeros(N, dtype=torch.bool, device=dev); _ever_both = torch.zeros(N, dtype=torch.bool, device=dev)
_ss_w = torch.zeros(N, device=dev); _ss_v = torch.zeros(N, device=dev); _ss_qd2 = torch.zeros(N, device=dev); _ss_n = torch.zeros(N, device=dev); _ss_lift = torch.zeros(N, device=dev)
_SS_FROM = STEPS - 150   # 마지막 3s = 정상상태(스탠바이) 구간
t0 = time.time()
for i in range(STEPS):
    with torch.inference_mode():
        a = torch.zeros(N, 12, device=dev) if policy is None else policy(obs)
        obs, _, dones, _ = env.step(a)
    st = uenv._hang
    term = uenv.termination_manager.terminated
    pg = robot.data.projected_gravity_b
    pitch = torch.rad2deg(torch.atan2(pg[:, 0], -pg[:, 2])); roll = torch.rad2deg(torch.atan2(pg[:, 1], -pg[:, 2]))
    w = robot.data.root_ang_vel_b.norm(dim=-1)
    f = _cs.data.net_forces_w[:, [_fl, _fr], :].norm(dim=-1)
    contact = (f > 5.0).float()
    a_ = alive
    max_roll[a_] = torch.maximum(max_roll[a_], roll[a_].abs()); max_pitch[a_] = torch.maximum(max_pitch[a_], pitch[a_].abs())
    max_w[a_] = torch.maximum(max_w[a_], w[a_])
    touch_t = torch.where(a_ & (touch_t < 0) & (st["touch_t"] >= 0), st["touch_t"], touch_t)
    newly_rel = a_ & (rel_t < 0) & (st["released_t"] >= 0)
    rel_t = torch.where(newly_rel, st["released_t"], rel_t); pitch_at_rel[newly_rel] = pitch[newly_rel]
    newly_dead = a_ & term
    death_step[newly_dead] = i; pitch_death[newly_dead] = pitch[newly_dead]; roll_death[newly_dead] = roll[newly_dead]
    r_at_death[newly_dead] = st["r"][newly_dead]
    last_q[a_] = robot.data.joint_pos[a_]; last_contact[a_] = contact[a_]
    _post = a_ & (st["r"] <= 0.0)
    _both_now = (contact[:, 0] > 0) & (contact[:, 1] > 0)
    _ever_both |= (a_ & _both_now)
    _lift_evt = (a_ & _ever_both & _prev_both & ~_both_now).float()
    _lift_cnt += _lift_evt
    if i >= _SS_FROM:
        _ss = a_ & (st["r"] <= 0.0)
        _ss_w[_ss] += robot.data.root_ang_vel_b[_ss, :2].norm(dim=-1); _ss_v[_ss] += robot.data.root_lin_vel_b[_ss, :2].norm(dim=-1)
        _ss_qd2[_ss] += (robot.data.joint_vel[_ss] ** 2).mean(1); _ss_n[_ss] += 1.0; _ss_lift += _lift_evt
    _prev_both = _both_now.clone()
    if _post.any():
        _st_w[_post] += robot.data.root_ang_vel_b[_post, :2].norm(dim=-1); _st_v[_post] += robot.data.root_lin_vel_b[_post, :2].norm(dim=-1)
        _st_qd2[_post] += (robot.data.joint_vel[_post] ** 2).mean(1); _st_n[_post] += 1.0
        _t = robot.data.applied_torque[_post].abs()
        _tau_peak = torch.maximum(_tau_peak, _t.max(0).values); _tau_rms2 += (_t ** 2).sum(0)
        _tau_sat += (_t >= 0.95 * _eff[_post]).float().sum(0); _tau_n += int(_post.sum())
        _pw = robot.data.body_pos_w[_post][:, [_bl, _br], :]
        _stance_w.append((_pw[:, 0, 1] - _pw[:, 1, 1]).abs().mean().item()); _stance_dx.append((_pw[:, 0, 0] - _pw[:, 1, 0]).abs().mean().item())
    alive &= ~term
    ratio_trace.append(st["r"].mean().item()); contact_trace.append(contact[alive].mean().item() if alive.any() else float("nan"))
    z_trace.append(robot.data.root_pos_w[alive, 2].mean().item() if alive.any() else float("nan"))
    pitch_trace.append(pitch[alive].mean().item() if alive.any() else float("nan"))

# ── 지표 ──
fell = death_step >= 0
touched = touch_t >= 0
released = rel_t >= 0
t_final = STEPS / 50.0
surv_after = torch.where(fell, death_step.float() / 50.0, torch.full_like(rel_t, t_final)) - rel_t
stand5 = released & (~fell | (surv_after >= args_cli.stand_s))
def pct(x): return 100.0 * float(x.float().mean())
def dirs(p, r):
    m = ~torch.isnan(p)
    back = int(((p < 0) & (p.abs() >= r.abs()) & m).sum()); fwd = int(((p > 0) & (p.abs() >= r.abs()) & m).sum())
    left = int(((r > 0) & (r.abs() > p.abs()) & m).sum()); right = int(((r < 0) & (r.abs() > p.abs()) & m).sum())
    return back, fwd, left, right
b, fw, l, rgt = dirs(pitch_death, roll_death)
jn = list(robot.data.joint_names)
q_end = torch.rad2deg(last_q)
out_lines = []
P = out_lines.append
P("=" * 74); P(f"Hanging-to-Stand 평가 — {args_cli.task}"); P(f"인자: {' '.join(sys.argv[1:])}")
P(f"체크포인트: {resume_path}"); P(f"배포 경로: {chain_desc}"); P(f"env {N} × {T_s:.1f}s, DR {'OFF(명목)' if args_cli.nominal else 'ON'}, com_fix={args_cli.com_fix}, 푸시 {'OFF' if (args_cli.no_push or args_cli.nominal) else 'ON'}, 관측노이즈 {'OFF' if args_cli.no_obs_noise else 'ON'}")
P(f"지지 스케줄: r0 {_hp['r0_range']} 유지 {_hp['hold_range']}s 방출 {_hp['release_range']}s, 간격 z {env_cfg.events.reset_base.params['pose_range'].get('z')}")
P("=" * 74)
P(f"접지(양발 동시) 도달      : {int(touched.sum())}/{N} ({pct(touched):.1f}%)  평균 접지 시각 {float(touch_t[touched].mean()) if touched.any() else float('nan'):.2f}s")
P(f"지지 0 도달(살아서)        : {int(released.sum())}/{N} ({pct(released):.1f}%)  평균 방출완료 시각 {float(rel_t[released].mean()) if released.any() else float('nan'):.2f}s  그때 피치 {float(pitch_at_rel[released].mean()) if released.any() else float('nan'):+.2f}°")
P(f"**지지 0 이후 {args_cli.stand_s:.0f}s 생존**   : {int(stand5.sum())}/{N} ({pct(stand5):.1f}%)   ← 핵심 지표")
P(f"낙상(조기 종료)            : {int(fell.sum())}/{N} ({pct(fell):.1f}%)  방향 뒤 {b} · 앞 {fw} · 왼 {l} · 오른 {rgt}")
if fell.any():
    P(f"   낙상 시 지지비 평균 {float(r_at_death[fell].mean()):.2f}, 낙상 시각 평균 {float(death_step[fell].float().mean())/50:.2f}s, "
      f"방출 후 낙상 {int((fell & released).sum())} / 방출 전 낙상 {int((fell & ~released).sum())}")
P(f"env별 최대 |롤| 평균 {float(max_roll.mean()):.2f}°  최대 |피치| 평균 {float(max_pitch.mean()):.2f}°  최대 각속도 평균 {float(max_w.mean()):.2f} rad/s")
_lc = last_contact
P(f"종료 시 접촉: 양발 {int(((_lc[:,0]>0)&(_lc[:,1]>0)).sum())} · 왼만 {int(((_lc[:,0]>0)&(_lc[:,1]==0)).sum())} · 오른만 {int(((_lc[:,0]==0)&(_lc[:,1]>0)).sum())} · 무접촉 {int(((_lc[:,0]==0)&(_lc[:,1]==0)).sum())}")
P("종료 시 관절각 평균 [deg] (생존 env):")
_al = ~fell
if _al.any():
    for k, n in enumerate(jn):
        P(f"   {n:<22} {float(q_end[_al, k].mean()):+7.2f}")
else:
    P("   (생존 env 없음)")
if _tau_n > 0:
    P("물리 실현성 — 지지 0 이후 생존 스텝 (AK 한계 대비): 관절 τ피크 /한계 τRMS 포화%")
    for k, n in enumerate(jn):
        P(f"   {n:<22} {float(_tau_peak[k]):6.2f} {float(_eff[0, k]):5.1f} {float((_tau_rms2[k] / _tau_n).sqrt()):6.2f} {100.0 * float(_tau_sat[k]) / _tau_n:5.1f}")
    P(f"   발 간격(좌우, 발목롤 링크) 평균 {sum(_stance_w)/len(_stance_w)*1000:.0f} mm · 앞뒤 어긋남 {sum(_stance_dx)/len(_stance_dx)*1000:.0f} mm")
_ok = _st_n > 0
if _ok.any():
    P(f"정지도 — 지지 0 이후 생존 스텝 env 평균: |각속도xy| {float((_st_w[_ok]/_st_n[_ok]).mean()):.3f} rad/s · |수평속도| {float((_st_v[_ok]/_st_n[_ok]).mean()):.3f} m/s · "
      f"관절속도 RMS {float(((_st_qd2[_ok]/_st_n[_ok]).sqrt()).mean()):.3f} rad/s · 발 들기 {float(_lift_cnt.mean()):.2f}회/env (양발 접촉 후)")
_ok2 = _ss_n > 0
if _ok2.any():
    P(f"**정상상태(마지막 3s, 스탠바이)** env 평균: |각속도xy| {float((_ss_w[_ok2]/_ss_n[_ok2]).mean()):.3f} rad/s · |수평속도| {float((_ss_v[_ok2]/_ss_n[_ok2]).mean()):.3f} m/s · "
      f"관절속도 RMS {float(((_ss_qd2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.3f} rad/s · 발 들기 {float(_ss_lift.mean()):.2f}회/env")
P("시간축 (0.5s 간격): t  지지비  접촉비  base z  피치")
for k in range(0, STEPS, 25):
    P(f"   {k/50:5.1f}  {ratio_trace[k]:5.2f}  {contact_trace[k]:5.2f}  {z_trace[k]:5.3f}  {pitch_trace[k]:+6.2f}")
P(f"소요 {time.time()-t0:.0f}s")
txt = "\n".join(out_lines)
print(txt)
tag = args_cli.out_tag or time.strftime("%m%d_%H%M%S")
os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True); os.makedirs(os.path.join(REPO, "demo_output"), exist_ok=True)
with open(os.path.join(REPO, "log_picture", f"hang2stand_{tag}.txt"), "w", encoding="utf-8") as fh:
    fh.write(txt + "\n")
np.savez(os.path.join(REPO, "demo_output", f"hang2stand_{tag}.npz"), death_step=death_step.cpu().numpy(), touch_t=touch_t.cpu().numpy(),
         rel_t=rel_t.cpu().numpy(), max_roll=max_roll.cpu().numpy(), max_pitch=max_pitch.cpu().numpy(), max_w=max_w.cpu().numpy(),
         pitch_death=pitch_death.cpu().numpy(), roll_death=roll_death.cpu().numpy(), q_end=q_end.cpu().numpy(),
         ratio_trace=np.array(ratio_trace), contact_trace=np.array(contact_trace), z_trace=np.array(z_trace), pitch_trace=np.array(pitch_trace))
print(f"[산출물] log_picture/hang2stand_{tag}.txt")
env.close()
simulation_app.close()
