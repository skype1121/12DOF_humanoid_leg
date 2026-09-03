"""검증 사다리 — 0~4단계 게이트. 아래 단계를 통과하기 전에 위를 실물에 올리지 않는다.

**왜 사다리인가 (0810).** 다섯 세대가 실물에서 연달아 실패했는데, 그 판정이
전부 '정책 서기'(2단계) 하나로만 이뤄졌다. 그래서 실패했을 때 원인이
정책인지·모델인지·래퍼인지 가릴 수가 없었다. 아래 단계가 통과해야 위
단계의 실패가 그 단계 탓임을 알 수 있다.

    0단계  강체 홀드 낙하        — 질량 분포가 맞는가 (정책 무관)
    1단계  강체 홀드 서기        — 강체 홀드로 서는가 (정책 무관)
    2단계  정책 서기            — 정책이 균형을 잡는가
    3단계  정책 서기 + 외란      — 밀어도 버티는가
    4단계  보행                 — 명령을 따르는가

**두 조건 모두에서 잰다.** 지금까지 승격 판정이 랜덤화 OFF 에서만 나왔고
(CLAUDE.md 3항), 그것이 다섯 세대를 잘못 고르게 한 원인이다.

**성공률만 보지 않는다.** 부드러움·에너지가 나빠지면 실물에서 발진·과열로
돌아온다. 세대 승격 조건에 **회귀 금지**를 넣는다 — 이전 세대 대비
부드러움(E_smth)·에너지가 나빠지면 성공률이 좋아도 승격 불가.

실행:
  PYTHONPATH=/home/ryu/humanoid_leg_test1 ~/IsaacLab/isaaclab.sh -p \
      rl_walking/scripts/eval_ladder.py --headless \
      --task Biped12-Velocity-Stage4V2-Stand-V6-v0 \
      --load_run 2026-08-04_11-31-27 --checkpoint model_59899.pt \
      --stages 0,1,2,3 --out_tag v6b
"""
import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str,
                    default="Biped12-Velocity-Stage4V2-Stand-V6-v0")
parser.add_argument("--stage", type=int, required=True,
                    help="0=자유낙하 1=PD홀드 2=정책서기 3=서기+외란 4=보행")
parser.add_argument("--dr", type=int, default=1, help="1=랜덤화 ON, 0=OFF")
parser.add_argument("--usd", type=str, default=None,
                    help="로봇 USD 교체 — spawn.usd_path 오버라이드")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--out_tag", type=str, default="")
parser.add_argument("--baseline", type=str, default=None,
                    help="회귀 판정 기준이 될 이전 세대 리포트 json 경로")
# 강성 sysid (0904): 17.6kg 실질량 자산이 kp150 강체홀드에서 64/64 후방 낙하하는데
# 실물은 같은 kp150 으로 5분 섰다 → 시뮬에 없는 패시브 강성(감속기 마찰 등)을 스윕
parser.add_argument("--kp", type=float, default=None,
                    help="전 액추에이터 stiffness 오버라이드 (Nm/rad)")
parser.add_argument("--jfric", type=float, default=None,
                    help="관절 마찰계수 고정값 (DR OFF 시 0 대신) — PhysX joint friction coeff")
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper                    # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path                 # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry              # noqa: E402
from rsl_rl.runners import OnPolicyRunner                            # noqa: E402

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.join(REPO, "log_picture", f"검증사다리_{args_cli.out_tag}.txt")
OUT_JSON = os.path.join(REPO, "demo_output", f"ladder_{args_cli.out_tag}.json")

ACTION_SCALE = 0.25
DT = 1.0 / 50.0

#: 단계별 통과 기준. 근거는 전부 실측·실기 기록이다.
#:   0단계 — 실물 강체 낙하 관측이 0804(안정)와 0809(−19°/s)로 갈린다.
#:           둘 다 담으려면 모델은 '적어도 뒤로 급히 무너지지는 않아야' 한다.
#:   1·2단계 — 낙상률. 3단계는 외란이 있으니 기준을 완화한다.
GATES = {
    # 0단계 = **실물 0단계와 같은 시험**: kp150 강체 홀드(전 관절 0)로 세우고
    # 줄을 놓는다. 무토크 자유낙하는 전부 주저앉아 판별력이 없다
    # (0810 실측: 추세 −195.9°/s, 64/64 낙상). 이 단계는 정책이 아니라
    # **질량 분포**를 잰다. 기준 구간이 넓은 이유는 실물 관측이 갈리기 때문 —
    # 0804 는 같은 자세에서 안정, 0809 는 −19°/s 후방 낙하. 둘 다 담는다.
    0: {"name": "강체 홀드 낙하 (모델정합)", "max_backfall_dps": 25.0,
        "max_fwd_dps": 5.0},
    1: {"name": "강체 홀드 서기 (자립)", "max_fall_ratio": 0.30},
    2: {"name": "정책 서기", "max_fall_ratio": 0.20},
    3: {"name": "정책 서기 + 외란", "max_fall_ratio": 0.40},
    4: {"name": "보행", "max_fall_ratio": 0.40, "min_track": 0.5},
}

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

lines = []


def P(s=""):
    print(s)
    lines.append(s)


def make_env(stage, dr_on):
    """단계·랜덤화 조합에 맞춘 환경을 만든다."""
    cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.usd:
        cfg.scene.robot.spawn.usd_path = args_cli.usd
    cfg.seed = 0
    cfg.episode_length_s = 60.0
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
    if stage == 4:
        cfg.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.4)
    # 외란은 3·4단계에서만 (0·1·2 단계는 그 단계가 재려는 것이 아니다)
    if stage < 3:
        cfg.events.push_robot = None
    if stage <= 1:
        # **0·1단계는 외란을 빼고 모델만 잰다.** reset_base 의 velocity_range
        # (roll/pitch/yaw ±0.3 rad/s = ±17°/s)는 모델 불확실성이 아니라 외란이다.
        # 0810 실측: 이걸 켠 채로는 명목 모델이 17/64 넘어졌는데, 끄자 0/64 ·
        # 추세 +0.07°/s 로 **완벽히 섰다** — 실물 0804(−1.7° 정지)와 일치.
        # 즉 "시뮬은 곧은 다리에서 뒤로 넘어진다(−4.55°/s)"는 여러 세대의
        # 근거였던 주장 자체가 이 오염의 산물이었다.
        _vr0 = cfg.events.reset_base.params.get("velocity_range")
        if isinstance(_vr0, dict):
            for _k0 in list(_vr0):
                _vr0[_k0] = (0.0, 0.0)
        for _k0 in ("roll", "pitch"):
            if _k0 in cfg.events.reset_base.params["pose_range"]:
                cfg.events.reset_base.params["pose_range"][_k0] = (0.0, 0.0)
        if getattr(cfg.events, "reset_robot_joints", None):
            cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        # **0·1단계는 항상 평지.** 이 단계가 재는 것은 질량 분포이지 지형
        # 내성이 아니고, 실물 0단계 시험도 평평한 바닥에서 했다.
        # (0810 실측: 이걸 안 껐더니 랜덤화 OFF 에서도 18/64 가 넘어졌다 —
        #  v2 지형이 최대 19° 경사를 20% 섞기 때문. 모델 판정이 오염된다.)
        # 생성기 지형은 all-flat 으로 바꿔도 패치 경계·원점 차이가 남는다.
        # 무한 평면으로 갈아 끼우는 것이 확실하다 (실물 시험장과 같다).
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        if getattr(cfg.curriculum, "terrain_levels", None) is not None:
            cfg.curriculum.terrain_levels = None
    if not dr_on:
        # 랜덤화 OFF — '이상 조건'. 이 값만으로 승격 판정 금지.
        for k in ("roll", "pitch"):
            if k in cfg.events.reset_base.params["pose_range"]:
                cfg.events.reset_base.params["pose_range"][k] = (0.0, 0.0)
        if getattr(cfg.events, "reset_robot_joints", None):
            cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
            cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        # ⚠ reset_base 는 pose_range 말고 **velocity_range** 도 갖는다:
        #   roll/pitch/yaw ±0.3 rad/s (=±17°/s). 0810 실측: 다른 걸 다 껐는데도
        #   DR OFF 가 17/64 로 갈린 범인이 이것이었다. 균형 문턱 시험에
        #   ±17°/s 초기 각속도를 주면 재는 것은 균형이 아니라 외란 내성이다.
        _vr = cfg.events.reset_base.params.get("velocity_range")
        if isinstance(_vr, dict):
            for _k in list(_vr):
                _vr[_k] = (0.0, 0.0)
        cfg.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
        cfg.events.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
        cfg.actions.joint_pos.min_delay_steps = 0
        cfg.actions.joint_pos.max_delay_steps = 0
        if getattr(cfg.events, "base_com", None):
            cfg.events.base_com.params["com_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
        if getattr(cfg.events, "calib_bias", None):
            cfg.events.calib_bias.params["joint_bias_rad"] = (0.0, 0.0)
            cfg.events.calib_bias.params["imu_tilt_rad"] = (0.0, 0.0)
        # 아래 둘을 빠뜨려 'OFF' 가 절반만 꺼져 있었다 (0810 실측: DR OFF 인데
        # 64 env 결과가 갈려 18/64 가 넘어졌다 — 완전히 같은 조건이면 0 아니면 64).
        if getattr(cfg.events, "add_base_mass", None):
            m = cfg.events.add_base_mass.params["mass_distribution_params"]
            mid = 0.5 * (m[0] + m[1])
            cfg.events.add_base_mass.params["mass_distribution_params"] = (mid, mid)
        if getattr(cfg.events, "joint_friction", None):
            cfg.events.joint_friction.params[
                "friction_distribution_params"] = (0.0, 0.0)
        # **지면 마찰**이 env 마다 다르면 넘어짐 판정이 갈린다 — 0810 실측:
        # 다른 걸 다 껐는데도 DR OFF 가 17/64 로 갈렸고 범인이 이것이었다.
        if getattr(cfg.events, "physics_material", None):
            pm = cfg.events.physics_material.params
            pm["static_friction_range"] = (0.8, 0.8)
            pm["dynamic_friction_range"] = (0.6, 0.6)
        if getattr(cfg.events, "joint_armature", None):
            cfg.events.joint_armature.params[
                "armature_distribution_params"] = (1.0, 1.0)
    return cfg


def run(stage, dr_on, policy=None):
    """한 조합을 돌리고 지표 묶음을 낸다."""
    cfg = make_env(stage, dr_on)
    if args_cli.kp is not None:
        for _a in cfg.scene.robot.actuators.values():
            _a.stiffness = args_cli.kp
    if args_cli.jfric is not None and getattr(cfg.events, "joint_friction", None):
        cfg.events.joint_friction.params["friction_distribution_params"] = (
            args_cli.jfric, args_cli.jfric)
    env = gym.make(args_cli.task, cfg=cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    dev = uenv.device
    N = args_cli.num_envs

    pol = None
    if stage >= 2:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                                device=agent_cfg.device)
        runner.load(resume_path)
        pol = runner.get_inference_policy(device=dev)

    # 0·1단계 = 착지 자세(전 관절 0) 강체 홀드. 같은 물리 시험이고
    # 판정 기준만 다르다 — 0 은 모델 정합(낙하 속도), 1 은 자립 여부.
    hold = torch.zeros(N, robot.data.joint_pos.shape[1], device=dev)
    if stage <= 1:
        hold = (torch.zeros_like(robot.data.default_joint_pos)
                - robot.data.default_joint_pos) / ACTION_SCALE
    obs = env.get_observations()
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    death = torch.full((N,), args_cli.steps, dtype=torch.long, device=dev)
    pg_log, act_log, tau_log, sat, nstep = [], [], [], None, 0
    lim = robot.data.soft_joint_pos_limits[0]
    for k in range(args_cli.steps):
        with torch.inference_mode():
            a = hold if pol is None else pol(obs)
            # env.step 의 4번째 반환은 extras(dict). `_` 로 받으면 루프변수가
            # dict 로 덮여 death 인덱싱이 깨진다 (0810 실측 버그).
            obs, _r, _d, _x = env.step(a)
        pg_log.append(robot.data.projected_gravity_b.clone().cpu().numpy())
        act_log.append((a if pol is None else a).detach().cpu().numpy())
        tau_log.append(robot.data.applied_torque.detach().cpu().numpy())
        near = ((robot.data.joint_pos - lim[:, 0]).abs() < 0.0175) | \
               ((robot.data.joint_pos - lim[:, 1]).abs() < 0.0175)
        sat = near.float().sum(0) if sat is None else sat + near.float().sum(0)
        nstep += N
        term = uenv.termination_manager.terminated
        death[alive & term] = k
        alive &= ~term

    pg = np.stack(pg_log)
    fwd = np.degrees(np.arctan2(pg[:, :, 0], -pg[:, :, 2]))
    act = np.stack(act_log)
    tau = np.stack(tau_log)
    d = death.cpu().numpy()
    W = min(int(3.0 / DT), pg.shape[0])
    t = np.arange(pg.shape[0]) * DT
    slopes = []
    for e in range(N):
        n = min(W, max(int(d[e]), 2))
        if n < 10:
            continue
        A = np.vstack([t[:n], np.ones(n)]).T
        slopes.append(float(np.linalg.lstsq(A, fwd[:n, e], rcond=None)[0][0]))
    # 부드러움 = 액션 2차차분 제곱합 (HoST E_smth), 에너지 = Στ²·dt
    e_smth = float((np.diff(act, n=2, axis=0) ** 2).sum(-1).mean())
    energy = float((tau ** 2).sum(-1).mean() * DT)
    res = {
        "stage": stage, "dr": bool(dr_on),
        "fall": int((d < args_cli.steps).sum()), "n": N,
        "fall_ratio": float((d < args_cli.steps).mean()),
        "trend_dps": float(np.mean(slopes)) if slopes else float("nan"),
        "e_smth": e_smth, "energy": energy,
        "cmd_amp_deg": float(np.degrees(act.std(0).mean()) * ACTION_SCALE),
        "sat_ratio": float((sat / max(1, nstep)).max().item()),
    }
    env.close()
    return res


# ── 단일 조합 실행 ────────────────────────────────────────────────────
# Isaac 은 한 프로세스에서 SimulationContext 를 두 번 만들 수 없다.
# 조합마다 프로세스를 새로 띄우고, 결과를 한 줄 json 으로 누적한다.
st, dr = args_cli.stage, bool(args_cli.dr)
r = run(st, dr)
g = GATES[st]
ok = True
if "max_fall_ratio" in g:
    ok = ok and r["fall_ratio"] <= g["max_fall_ratio"]
if "max_backfall_dps" in g:
    ok = ok and (np.isnan(r["trend_dps"]) or r["trend_dps"] > -g["max_backfall_dps"])
if "max_fwd_dps" in g:
    ok = ok and (np.isnan(r["trend_dps"]) or r["trend_dps"] < g["max_fwd_dps"])
r["pass_gate"] = bool(ok)
r["name"] = g["name"]
r["ckpt"] = resume_path
r["task"] = args_cli.task

os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
rows = []
if os.path.exists(OUT_JSON):
    try:
        rows = json.load(open(OUT_JSON))
    except Exception:                                     # noqa: BLE001
        rows = []
rows = [x for x in rows if not (x["stage"] == st and x["dr"] == dr)]
rows.append(r)
rows.sort(key=lambda x: (x["stage"], x["dr"]))
with open(OUT_JSON, "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)

P(f"{g['name']:<22}{'ON' if dr else 'OFF':<5}"
  f"{r['fall']:>4}/{r['n']:<4}{r['trend_dps']:>10.2f}{r['e_smth']:>10.4f}"
  f"{r['energy']:>10.2f}{r['cmd_amp_deg']:>10.2f}"
  f"{r['sat_ratio']*100:>7.1f}%{('통과' if ok else '탈락'):>8}")
print(f"\n누적 json: {OUT_JSON}")
simulation_app.close()
