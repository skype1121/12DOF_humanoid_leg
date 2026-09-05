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
import math
import json
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
parser.add_argument("--com_fix", type=float, default=None,
                    help="base_com x 를 이 값으로 고정 [m] — CoM 인과 검증용")
parser.add_argument("--com_z", type=float, default=0.0,
                    help="--com_fix 와 함께 쓰는 base_com z 오프셋 [m] (구 하드코딩 0.08 → 기본 0)")
parser.add_argument("--mass_kg", type=float, default=None,
                    help="실물 총질량 [kg] 고정 — payload 를 역산해 넣는다 (URDF base 10.491)")
parser.add_argument("--deploy", type=str, default=None,
                    help="실배포 래퍼 프리셋: new(슬루 인수·0810 수술) | old(블렌딩 1.5s)")
parser.add_argument("--lpf", type=float, default=0.0,
                    help="브리지 stand LPF EMA beta (실물 0.27). 0=끔")
parser.add_argument("--blend", type=float, default=0.0,
                    help="브리지 온셋 블렌딩 [s] (실물 1.5). 0=끔")
parser.add_argument("--blend_mode", type=str, default="full",
                    help="full(현행: 정책 권한도 α로 깎임) | offset(기준자세만 램프) | slew(속도제한)")
parser.add_argument("--slew_deg", type=float, default=4.0,
                    help="slew 모드 틱당 상한 [°] (브리지 4°/틱)")
parser.add_argument("--blend_from", type=str, default="zero",
                    help="블렌딩 시작 자세: zero(트림0) | trim(a12.5/h6.4/k10.4) | eq(정책평형)")
parser.add_argument("--land", action="store_true",
                    help="착지 자세(전 관절 0)에서 시작 — 실물 인수 시점 재현")
parser.add_argument("--zero_deg", type=float, default=0.0,
                    help="영점 오차 [°] — 관측은 δ 적게, 명령은 δ 더 나간다")
parser.add_argument("--imu_deg", type=float, default=0.0,
                    help="IMU 장착 오차 [°] — 중력벡터를 피치축으로 회전 (앞뒤)")
parser.add_argument("--still", action="store_true",
                    help="초기 속도 랜덤화 차단 — reset_base velocity_range(±17°/s)와 관절 속도. "
                         "실물 인수는 정지 상태에서 시작하므로 실물 재현 런에서 권장")
parser.add_argument("--usd", type=str, default=None,
                    help="로봇 USD 교체 (발 기하 변형판 등) — spawn.usd_path 오버라이드")
parser.add_argument("--att_lag", type=int, default=0,
                    help="IMU 자세 랙 [틱] — 중력 채널만 N틱 지연 (자이로는 즉시). iAHRS 자세필터 랙 모사")
parser.add_argument("--gyro_bias_dps", type=float, default=0.0,
                    help="자이로 피치축 상수 편향 [°/s] — ang_vel y 채널 오염")
parser.add_argument("--imu_roll_deg", type=float, default=0.0,
                    help="IMU 롤 장착 오차 [°] — 중력벡터를 롤축으로 회전 (좌우)")
parser.add_argument("--sign", type=str, default="mixed",
                    help="영점 오차 부호: mixed|pos|neg")
parser.add_argument("--kp", type=float, default=None,
                    help="서보 강성 고정 [Nm/rad] — SEA 루프이득 kp/k_struct 검증용")
parser.add_argument("--sea", type=float, default=None,
                    help="구조 강성 [Nm/rad] — 로터-링크 SEA 관측 모델 (실측 43.5)")
parser.add_argument("--delay", type=str, default=None,
                    help="액션 지연 min,max [틱] — 떨림 재현 스윕용")
parser.add_argument("--contact_delay", type=int, default=None,
                    help="발접촉 관측 지연 [틱] — 실물 발압력 10Hz(5틱) 모사")
parser.add_argument("--mass_fix", type=float, default=None,
                    help="add_base_mass 를 이 값으로 고정 [kg]")
parser.add_argument("--nominal", action="store_true",
                    help="결정론적 명목 평가 — 모든 랜덤화 OFF(속도·관절리셋·게인·지연·CoM·영점·질량·마찰·재질·아마추어). "
                         "실물 nominal 모델 그대로의 정책 거동 (0904 승윤님 평가원칙). com_fix/imu_deg 는 그 위에 적용")
parser.add_argument("--keep_env_lpf", action="store_true",
                    help="nominal/deploy 에서도 env 액션텀 LPF 대역을 유지 (v15 발산 분리 실험용)")
parser.add_argument("--keep_delay", action="store_true",
                    help="nominal 에서도 env 액션 지연 대역을 유지 (분리 실험용)")
parser.add_argument("--ground_k", type=float, default=None,
                    help="지면 compliant contact 강성 [N/m, 접촉점당] — 패드 컴플라이언스 실험")
parser.add_argument("--ground_c", type=float, default=None,
                    help="지면 compliant contact 감쇠 [N·s/m]; 생략 시 2·sqrt(2k)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ── 실배포 프리셋 ────────────────────────────────────────────────────
# 세대 판정은 **실제로 배포되는 것**과 같은 조건에서 해야 한다. v3~v9 승격은
# 래퍼 없는 조건에서 결정됐는데, 실물 실패의 지배 요인이 그 래퍼였다.
if args_cli.deploy in ("new", "old"):
    args_cli.lpf = 0.27              # 브리지 STAND_LPF_BETA
    args_cli.land = True             # hang_stand 착지 자세에서 인수
    args_cli.blend_from = "zero"     # 트림0 착지
    if args_cli.deploy == "new":
        args_cli.blend_mode = "slew"     # stage8 _step_toward 4°/틱
        args_cli.slew_deg = 4.0
    else:
        args_cli.blend_mode = "full"     # 종전 크로스페이드
        args_cli.blend = 1.5

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym          # noqa: E402
import numpy as np               # noqa: E402
import torch                     # noqa: E402
import isaaclab_tasks            # noqa: F401,E402
import rl_walking                # noqa: F401,E402
from rl_walking import mdp_stage4 as mdp_stage4_mod  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.join(REPO, "log_picture", f"드리프트DR_{args_cli.out_tag}.txt")

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
if args_cli.usd:
    env_cfg.scene.robot.spawn.usd_path = args_cli.usd
if args_cli.deploy in ("new", "old") and hasattr(env_cfg.actions.joint_pos, "lpf_beta_range") and not args_cli.keep_env_lpf:
    # v15+ 는 브리지 LPF(β 0.27)를 학습 env 액션텀에 내장한다(β 0.2~1 랜덤). 배포 판정은
    # 래퍼가 브리지 LPF 를 따로 씌우므로 env 쪽 LPF 는 꺼야 실제 배포(필터 1개)와 같다.
    # 0904 실측: 둘 다 켜면 위상지연 2배 → v15 가 3.75Hz 로 발산(E_smth 8.7e10, 100% 낙상).
    env_cfg.actions.joint_pos.lpf_beta_range = (1.0, 1.0)
if args_cli.still:
    # 7f52452 가 이 파일만 빠뜨렸었다 (적대검증 지적). 실물 5차 인수는 줄에
    # 매달려 정지한 상태에서 시작한다 — ±17°/s 초기 각속도는 실물에 없는 외란.
    _vr = env_cfg.events.reset_base.params.get("velocity_range")
    if isinstance(_vr, dict):
        for _k in list(_vr):
            _vr[_k] = (0.0, 0.0)
    if getattr(env_cfg.events, "reset_robot_joints", None):
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
env_cfg.seed = agent_cfg.seed
# 정지 서기 조건 — 실물 무개입 낙하와 동일하게 명령 전무
env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
env_cfg.events.reset_base.params["pose_range"]["yaw"] = (0.0, 0.0)
env_cfg.episode_length_s = 60.0
if args_cli.nominal:
    # eval_ladder DR-OFF 블록과 동일 목록 (남은 것 목록으로 점검한 결과)
    _vr = env_cfg.events.reset_base.params.get("velocity_range")
    if isinstance(_vr, dict):
        for _k in list(_vr): _vr[_k] = (0.0, 0.0)
    for _k in ("roll", "pitch"):
        if _k in env_cfg.events.reset_base.params["pose_range"]:
            env_cfg.events.reset_base.params["pose_range"][_k] = (0.0, 0.0)
    if getattr(env_cfg.events, "reset_robot_joints", None):
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    env_cfg.events.actuator_gains.params["stiffness_distribution_params"] = (1.0, 1.0)
    env_cfg.events.actuator_gains.params["damping_distribution_params"] = (1.0, 1.0)
    if not args_cli.keep_delay:
        env_cfg.actions.joint_pos.min_delay_steps = 0
        env_cfg.actions.joint_pos.max_delay_steps = 0
    if hasattr(env_cfg.actions.joint_pos, "lpf_beta_range") and not args_cli.keep_env_lpf:
        env_cfg.actions.joint_pos.lpf_beta_range = (1.0, 1.0)
    if hasattr(env_cfg.actions.joint_pos, "backlash_rad_range"):
        env_cfg.actions.joint_pos.backlash_rad_range = (0.0, 0.0)
    if getattr(env_cfg.events, "base_com", None):
        env_cfg.events.base_com.params["com_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    if getattr(env_cfg.events, "calib_bias", None):
        for _k in ("joint_bias_rad", "imu_tilt_rad", "ankle_bias_rad", "knee_bias_rad"):
            if _k in env_cfg.events.calib_bias.params and env_cfg.events.calib_bias.params[_k] is not None:
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
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
        env_cfg.curriculum.terrain_levels = None
    # 0905 감사 확정: 아래 4개가 빠져 '명목'이 명목이 아니었다
    #  - link_mass_calib(전 링크 ×1.09~1.19, 10.49kg 시절 12kg 정합용) → 17.62kg 자산에 이중 적용
    #  - 관측 노이즈(enable_corruption), qvel 드롭아웃, 지면 restitution
    if getattr(env_cfg.events, "link_mass_calib", None):
        env_cfg.events.link_mass_calib.params["mass_distribution_params"] = (1.0, 1.0)
    env_cfg.observations.policy.enable_corruption = False
    _jv = getattr(env_cfg.observations.policy, "joint_vel", None)
    if _jv is not None and "zero_prob" in (_jv.params or {}):
        _jv.params["zero_prob"] = 0.0
    if getattr(env_cfg.events, "physics_material", None):
        env_cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
if args_cli.com_fix is not None:
    # CoM 인과 검증: 정책은 그대로 두고 **몸만** 실측값으로 바꾼다.
    # 지금까지 CoM 은 전부 정책 없는 시험에서만 다뤘다 — 정책이 루프에 있는
    # 조건에서 처음으로 검증하는 실험이다.
    # z 는 0905 감사 전까지 (0.08,0.08) 고정이었다 — 10.49kg+payload 시절 값. 실질량
    # 자산에선 근거 없는 +8cm 상향 → 기본 0, --com_z 로 명시.
    env_cfg.events.base_com.params["com_range"] = {
        "x": (args_cli.com_fix, args_cli.com_fix),
        "y": (0.0, 0.0), "z": (args_cli.com_z, args_cli.com_z)}
if args_cli.mass_kg is not None:
    # 총질량만 바꾸고 CoM 은 고정 — 질량이 정말 상쇄되는지 보는 대조.
    _pl = args_cli.mass_kg - 10.491        # base URDF 실측 전개값
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (_pl, _pl)
if args_cli.kp is not None and args_cli.sea is None:
    for _a in env_cfg.scene.robot.actuators.values():
        _a.stiffness = args_cli.kp
if args_cli.sea is not None:
    # ── 직렬 컴플라이언스(SEA) 양면 모델 ──────────────────────────────
    # 로터 관성이 작아 준정적으로 풀면 (kp: 서보, k: 구조):
    #   링크가 받는 실효 강성   k_eff = kp·k/(kp+k)
    #   엔코더(로터)가 읽는 값  θ_r = θ_l + τ_link/k
    # 구동측만 낮추면 '게인 스케일'(기존 근사)이고, 관측측만 넣으면
    # 토크가 과대해 폭발한다. 둘 다 해야 물리가 맞는다.
    _k = args_cli.sea
    for _a in env_cfg.scene.robot.actuators.values():
        _kp = args_cli.kp if args_cli.kp is not None else _a.stiffness
        _a.stiffness = _kp * _k / (_kp + _k)
    _jp = env_cfg.observations.policy.joint_pos
    _jp.func = mdp_stage4_mod.joint_pos_rel_sea
    _jp.params["k_struct"] = _k
if args_cli.delay is not None:
    # 실물 sysid 실측 루프지연은 4~8틱인데 학습 환경은 2~6틱이다.
    # 지연은 폐루프 발진의 대표 원인이므로 떨림 재현의 1순위 후보.
    _lo, _hi = (int(v) for v in args_cli.delay.split(","))
    env_cfg.actions.joint_pos.min_delay_steps = _lo
    env_cfg.actions.joint_pos.max_delay_steps = _hi
if args_cli.contact_delay is not None:
    # **관측 차원을 바꾸지 않는다** — history_length 를 늘리면 245 계약이 깨져
    # 브리지 조립·셀프테스트·기존 정책 전부와 호환이 사라진다(0810 발견).
    # 값의 시점만 늦춘다.
    env_cfg.observations.policy.feet_contact.params["delay_steps"] = \
        args_cli.contact_delay
if args_cli.mass_fix is not None:
    env_cfg.events.add_base_mass.params["mass_distribution_params"] = (
        args_cli.mass_fix, args_cli.mass_fix)
if args_cli.ground_k is not None:
    # 패드 컴플라이언스 실험 (0904): 실물 RA30P 패드(고무 캡) ↔ 강체 지면의 직렬 스프링을
    # 지면 재질의 compliant contact 로 모델링 (PhysX 5, 접촉점당 N/m · N·s/m).
    # 접촉 8점·모멘트암 60mm → 발 피치 회전강성 ≈ 8·k·0.06² = 0.029·k Nm/rad
    # (k=1e4 이면 ≈ 290 Nm/rad = 발목 kp150×2 와 동급 — 숨은 직렬 탄성).
    _pm = env_cfg.scene.terrain.physics_material
    _pm.compliant_contact_stiffness = args_cli.ground_k
    _pm.compliant_contact_damping = (args_cli.ground_c if args_cli.ground_c is not None
                                     else 2.0 * (args_cli.ground_k * 2.0) ** 0.5)

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

# ── 캘리브레이션 편향 주입 ────────────────────────────────────────────
# 관측 레이아웃 245 = 49×5: ang_vel3·gravity3·cmd3·qpos12·qvel12·act12·clock2·contact2
_GRAV, _QPOS = slice(15, 30), slice(45, 105)
_dz, _th = math.radians(args_cli.zero_deg), math.radians(args_cli.imu_deg)
_rl = math.radians(args_cli.imu_roll_deg)
_N = args_cli.num_envs
_dev = env.unwrapped.device
if args_cli.sign == "pos":
    _sg = torch.ones(_N, device=_dev)
elif args_cli.sign == "neg":
    _sg = -torch.ones(_N, device=_dev)
else:   # 한쪽으로만 주면 최악/최선만 본다 — env 반반
    _sg = torch.where(torch.arange(_N, device=_dev) % 2 == 0,
                      torch.ones(_N, device=_dev), -torch.ones(_N, device=_dev))
_dzv = (_dz * _sg).unsqueeze(1)


#: 관측 245 = 49×5 채널블록 (역사-major, 각 블록 reshape (5,ch)):
#: ang_vel 0:15 · gravity 15:30 · cmd 30:45 · qpos 45:105 · qvel · act · clock · contact
_ANGV = slice(0, 15)
_att_hist = []      # 자세 랙용 중력 블록 링버퍼


def _bias(o):
    """영점·IMU 편향·자세랙·자이로편향을 관측에 주입 (제자리 수정)."""
    t = o["policy"] if hasattr(o, "keys") and "policy" in o.keys() else o
    if args_cli.att_lag > 0:
        # iAHRS 자세필터 랙 — 중력(자세) 채널만 N틱 과거값. 자이로는 원시라 즉시.
        _att_hist.append(t[:, _GRAV].clone())
        if len(_att_hist) > args_cli.att_lag:
            t[:, _GRAV] = _att_hist.pop(0)
        else:
            t[:, _GRAV] = _att_hist[0]
    if args_cli.gyro_bias_dps != 0.0:
        av = t[:, _ANGV].reshape(-1, 5, 3).clone()
        av[..., 1] = av[..., 1] + math.radians(args_cli.gyro_bias_dps)
        t[:, _ANGV] = av.reshape(-1, 15)
    if _dz != 0.0:
        t[:, _QPOS] = t[:, _QPOS] - _dzv
    if _th != 0.0 or _rl != 0.0:
        g = t[:, _GRAV].reshape(-1, 5, 3).clone()
        if _th != 0.0:                      # 피치축 (앞뒤 장착 오차)
            gx, gz = g[..., 0].clone(), g[..., 2].clone()
            g[..., 0] = gx * math.cos(_th) - gz * math.sin(_th)
            g[..., 2] = gx * math.sin(_th) + gz * math.cos(_th)
        if _rl != 0.0:                      # 롤축 (좌우 장착 오차)
            gy, gz = g[..., 1].clone(), g[..., 2].clone()
            g[..., 1] = gy * math.cos(_rl) - gz * math.sin(_rl)
            g[..., 2] = gy * math.sin(_rl) + gz * math.cos(_rl)
        t[:, _GRAV] = g.reshape(-1, 15)
    return o


# ── 배포 래퍼 이식 (jetson/scripts/rl_bridge_node.py) ──────────────────
# 실물에만 있고 시뮬에 없던 두 가지. 다섯 세대가 전부 이 래퍼를 통과했다.
#   ① stand LPF  : EMA beta=0.27 (0803 발진 완화책, 차단 2.5Hz@50Hz)
#   ② 온셋 블렌딩: 착지 자세 → 정책 타깃 스무스스텝 크로스페이드 1.5s
# 순서는 브리지와 동일: 정책 → LPF → 블렌딩 (bridge 1181행 → 1227행)
_robot = env.unwrapped.scene["robot"]
_DEF = _robot.data.default_joint_pos.clone()
#: 착지 자세 후보 — hang_stand 가 스트리밍하던 명령이 블렌딩 시작점이 된다.
#: 트림 3채널 → 관절각 (하드웨어맵 direction·TRIM_CH gain 으로 확정된 변환)
_TRIM_POSE_DEG = {"left_ankle_f_joint": +12.5, "right_ankle_f_joint": -12.5,
                  "left_hip_f_joint": -6.4, "right_hip_f_joint": +6.4,
                  "left_knee_joint": -10.4, "right_knee_joint": +10.4}
_jn = list(_robot.data.joint_names)
if args_cli.blend_from == "trim":
    _LAND = torch.zeros_like(_DEF)
    for _i, _n in enumerate(_jn):
        _LAND[:, _i] = math.radians(_TRIM_POSE_DEG.get(_n, 0.0))
elif args_cli.blend_from == "eq":
    _LAND = _DEF.clone()                # 정책 평형에 가장 가까운 근사 = 기본자세
else:
    _LAND = torch.zeros_like(_DEF)      # 트림0 = 관절 프레임 전부 0
_ASCALE = 0.25
_tgt_filt = None
_cmd_prev = None


def _wrapper(a, step_i):
    """정책 액션 → 실제 명령. 브리지와 같은 순서로 LPF·블렌딩을 건다."""
    global _tgt_filt
    tgt = _DEF + _ASCALE * a                                   # 관절 프레임 타깃
    if args_cli.lpf > 0.0:
        _tgt_filt = tgt.clone() if _tgt_filt is None else \
            _tgt_filt + args_cli.lpf * (tgt - _tgt_filt)
        tgt = _tgt_filt
    if args_cli.blend_mode == "slew":
        # 속도제한 인수: 정책 권한 100%, 명령이 틱당 상한보다 빨리 못 움직인다.
        # 크로스페이드가 없으니 정책 보정이 처음부터 온전히 나간다.
        global _cmd_prev
        lim = math.radians(args_cli.slew_deg)
        _cmd_prev = _LAND.clone() if _cmd_prev is None else _cmd_prev
        tgt = _cmd_prev + torch.clamp(tgt - _cmd_prev, -lim, lim)
        _cmd_prev = tgt.clone()
    elif args_cli.blend > 0.0:
        u = min(1.0, max(0.0, (step_i / 50.0) / args_cli.blend))
        al = u * u * (3.0 - 2.0 * u)                           # 스무스스텝
        if args_cli.blend_mode == "offset":
            # 기준자세만 크로스페이드하고 정책 보정 0.25a 는 온전히 더한다.
            # 현행 full 은 (DEF+0.25a)·α 라 **정책의 보정량까지 α배로 깎인다** —
            # α=0.26(0.5초)이면 정책이 자기 보정의 26% 밖에 못 낸다.
            tgt = (_LAND * (1.0 - al) + _DEF * al) + (tgt - _DEF)
        else:
            tgt = _LAND * (1.0 - al) + tgt * al
    return (tgt - _DEF) / _ASCALE


if args_cli.land:
    # 실물은 hang_stand 트림0 자세로 착지한 뒤 정책이 인수한다.
    _robot.write_joint_state_to_sim(_LAND, torch.zeros_like(_LAND))
pg_log = []          # (steps, N, 3)
act_log = []         # (steps, N, 12) — 떨림 지표용
alive = torch.ones(N, dtype=torch.bool, device=uenv.device)
death_step = torch.full((N,), args_cli.steps, dtype=torch.long, device=uenv.device)

sat_hits = torch.zeros(robot.data.joint_pos.shape[1], device=robot.data.joint_pos.device)
sat_n = 0
# ── 물리 실현성 지표 (0904 승윤님 요청): 생존 env-step 에서 AK 모터 한계 대비 ─────
_dev = robot.data.joint_pos.device
_nj = robot.data.joint_pos.shape[1]
_eff_lim = robot.data.joint_effort_limits                      # (N, J)
_vel_lim = getattr(robot.data, "joint_velocity_limits", None)
if _vel_lim is None:                                           # 폴백: 액추에이터 cfg
    _vl = torch.full((_nj,), 30.0, device=_dev)
    for _k, _n in enumerate(robot.data.joint_names):
        if _n.endswith("_ankle_r_joint"): _vl[_k] = 5.0
    _vel_lim = _vl.unsqueeze(0).expand(N, -1)
_tau_peak = torch.zeros(_nj, device=_dev); _qd_peak = torch.zeros(_nj, device=_dev)
_tau_sat = torch.zeros(_nj, device=_dev); _qd_sat = torch.zeros(_nj, device=_dev)
_tau_rms2 = torch.zeros(_nj, device=_dev); _alive_n = 0
_roll_log = []                                                 # (steps, N)
_cs = uenv.scene["contact_forces"]
_bnames = list(_cs.body_names)
_fl = [k for k, n in enumerate(_bnames) if n == "left_ankle_r_joint"][0]
_fr = [k for k, n in enumerate(_bnames) if n == "right_ankle_r_joint"][0]
_rb = list(robot.data.body_names)
# ⚠ 이름 주의: _rl 은 위에서 IMU 롤 편향각(rad)으로 이미 쓰이는 전역 — 0904 저녁 이 이름을
# 발 바디 인덱스로 덮어써 _bias() 가 중력을 11rad 회전 → 06:15~19:30 사이 전 평가 무효 사고.
_bidx_l, _bidx_r = _rb.index("left_ankle_r_joint"), _rb.index("right_ankle_r_joint")
_asym_sum = 0.0; _asym_n = 0
_slip_sum = 0.0; _slip_hits = 0; _slip_n = 0
_land_dx = []; _land_dyaw = []
for i in range(args_cli.steps):
    with torch.inference_mode():
        actions = policy(_bias(obs))
        if _dz != 0.0:                       # 명령도 δ 만큼 더 나간다
            actions = actions + _dzv / 0.25
        obs, _, dones, _ = env.step(_wrapper(actions, i))
    pg_log.append(robot.data.projected_gravity_b.clone().cpu().numpy())
    act_log.append(actions.detach().cpu().numpy())
    # E: 관절 한계 도달 점검 — 한계에 정확히 붙는 것은 보상을 우회하는 해의 징후
    # 0904 맹점 수정: 소프트리밋(0.9×) 기준 1° 창은 하드리밋(±30°)에 붙은 관절을
    # 놓친다 (v14 hip_r ±30.0° 오리발 자세를 0.8% 로 보고). 하드리밋 기준 + 소프트리밋
    # 밖까지 '한계 도달'로 센다.
    _hlim = robot.data.joint_pos_limits[0]              # (n_j, 2) 하드 lo/hi
    _lim = robot.data.soft_joint_pos_limits[0]          # (n_j, 2) 소프트 lo/hi
    _q = robot.data.joint_pos
    _near = ((_q - _hlim[:, 0].unsqueeze(0)).abs() < 0.0175) | \
            ((_q - _hlim[:, 1].unsqueeze(0)).abs() < 0.0175) | \
            (_q < _lim[:, 0].unsqueeze(0)) | (_q > _lim[:, 1].unsqueeze(0))
    sat_hits = sat_hits + _near.float().sum(0)
    sat_n += args_cli.num_envs
    term = uenv.termination_manager.terminated
    newly_dead = alive & term
    death_step[newly_dead] = i
    alive &= ~term
    # 물리 실현성 누산 (생존 env 만)
    if alive.any():
        _a = alive
        _tau = robot.data.applied_torque[_a].abs()
        _qd = robot.data.joint_vel[_a].abs()
        _tau_peak = torch.maximum(_tau_peak, _tau.max(0).values)
        _qd_peak = torch.maximum(_qd_peak, _qd.max(0).values)
        _tau_sat += (_tau >= 0.95 * _eff_lim[_a]).float().sum(0)
        _qd_sat += (_qd >= 0.95 * _vel_lim[_a]).float().sum(0)
        _tau_rms2 += (_tau ** 2).sum(0)
        _alive_n += int(_a.sum())
        _pgv = robot.data.projected_gravity_b
        _roll_log.append(torch.rad2deg(torch.atan2(_pgv[:, 1], -_pgv[:, 2])).cpu().numpy())  # (N,) 매 스텝(alive 있을 때)
        _f = _cs.data.net_forces_w[:, [_fl, _fr], 2].clamp(min=0.0)     # (N,2) 수직력
        _both = _a & (_f[:, 0] > 5.0) & (_f[:, 1] > 5.0)
        if _both.any():
            _asym_sum += float(((_f[_both, 0] - _f[_both, 1]).abs() / (_f[_both].sum(1) + 1e-6)).sum())
            _asym_n += int(_both.sum())
        _v = robot.data.body_lin_vel_w[:, [_bidx_l, _bidx_r], :2].norm(dim=-1)     # (N,2) 발 수평속도
        _inc = _a.unsqueeze(1) & (_f > 20.0)
        if _inc.any():
            _slip_sum += float(_v[_inc].sum()); _slip_hits += int((_v[_inc] > 0.02).sum()); _slip_n += int(_inc.sum())
        if 50 <= i < 150:                                                    # 1~3s 정착 구간 착지 기하
            _px = robot.data.body_pos_w[:, [_bidx_l, _bidx_r], 0]
            _land_dx.append((_px[_a, 0] - _px[_a, 1]).cpu().numpy())
            _qy = robot.data.body_quat_w[:, [_bidx_l, _bidx_r], :]                  # (N,2,4) wxyz
            _yaw = torch.atan2(2 * (_qy[..., 0] * _qy[..., 3] + _qy[..., 1] * _qy[..., 2]),
                               1 - 2 * (_qy[..., 2] ** 2 + _qy[..., 3] ** 2))
            _land_dyaw.append(torch.rad2deg(_yaw[_a, 0] - _yaw[_a, 1]).cpu().numpy())

# 정착 구간 평균 관절각 = 이 정책의 '평형 자세'. 실물에서 정책 투입 전에
# 이 자세로 미리 옮겨놓으면(posture_probe) 정책이 움직일 거리가 0이 되어
# 블렌딩 중 하중을 잃을 이유가 사라진다 — 5차 붕괴의 직접 처방.
try:
    q = robot.data.joint_pos.detach().cpu().numpy()
    jn_names = list(robot.data.joint_names)
    eq_pose = {n: round(float(np.degrees(q[:, i].mean())), 2)
               for i, n in enumerate(jn_names)}
except Exception as _exc:
    eq_pose, jn_names = {}, []
    print(f"[warn] 평형 자세 산출 실패: {_exc}")

# ── 떨림 지표 (0809 실물에서 1급 판정 지표로 승격) ────────────────────
# 5차 실물 v6b 실측: 관절 1~4Hz 로 RMS 2~4° 진동, E_smth 4.40 (낙하 초기 7.01).
# "명령의 진폭이 실측보다 크다" = 정책이 떨림을 만든다. 같은 잣대로 시뮬을 잰다.
ACT = np.stack(act_log) if act_log else None
smth = domf = hi5 = None
if ACT is not None and ACT.shape[0] > 8:
    # E_smth = Σ‖a(t+2) − 2a(t+1) + a(t)‖²  — HoST 지표. 액션은 무차원이라
    # 실물(deg²)과 절대비교는 안 되고, **세대 간 상대비교**로 쓴다.
    d2 = np.diff(ACT, n=2, axis=0)
    # 0905 감사: 자동 리셋된 죽은 env 의 액션이 섞여 세대 간 비교 오염 → env 별 생존 구간만 평균
    _dn2 = death.cpu().numpy() if hasattr(death, "cpu") else death
    _per = []
    for _e in range(ACT.shape[1]):
        _n = int(min(d2.shape[0], max(_dn2[_e] - 2, 1)))
        _per.append(float(np.mean(np.sum(d2[:_n, _e, :] ** 2, axis=1))))
    smth = float(np.mean(_per))
    # 우세 주파수 — env·관절 평균 스펙트럼에서 (2차 추세 제거)
    fs_ = 50.0
    x = ACT - ACT.mean(axis=0, keepdims=True)
    n_ = x.shape[0]
    w = np.hanning(n_)[:, None, None]
    X = np.abs(np.fft.rfft(x * w, axis=0))
    f_ = np.fft.rfftfreq(n_, 1.0 / fs_)
    P_ = (X ** 2).mean(axis=(1, 2))
    m_ = f_ > 0.5
    if m_.any():
        domf = float(f_[m_][np.argmax(P_[m_])])
        hi5 = float(100.0 * P_[m_ & (f_ > 5.0)].sum() / max(P_[m_].sum(), 1e-12))

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

# E: 관절별 한계 도달 비율 (1° 이내에 머문 시간 비율)
_satr = (sat_hits / max(1, sat_n)).cpu().numpy() * 100.0
_jnames = list(robot.data.joint_names)

P("=" * 74)
P(f"DR 분포 전방 드리프트 — {args_cli.task}")
P("인자: " + " ".join(sys.argv[1:]))
P(f"체크포인트: {resume_path}")
P(f"{N} env × {args_cli.steps/50:.1f}s, 명령 전무(cmd=0), DR 켜짐")
P("=" * 74)
P(f"낙상(조기 종료): {int((death < args_cli.steps).sum())}/{N} "
  f"({100.0*(death < args_cli.steps).sum()/N:.1f}%)")
# 낙상 방향 분해 — 넘어진 순간의 몸통 기울기 부호로 전/후/좌/우 분류
_lat = np.degrees(np.arctan2(pg[:, :, 1], -pg[:, :, 2]))
_fell = death < args_cli.steps
_dir = {"뒤": 0, "앞": 0, "왼": 0, "오른": 0}
for e in np.where(_fell)[0]:
    # 0905 감사 확정: env.step 이 종료 판정 직후 같은 스텝에서 리셋하므로 death 인덱스의
    # 자세는 리셋 틸트(±4°)다. 넘어진 자세는 그 직전 프레임(임계 ~40° 직전).
    k = max(0, min(int(death[e]) - 1, pg.shape[0] - 1))
    f_, l_ = fwd[k, e], _lat[k, e]
    if abs(f_) >= abs(l_):
        _dir["앞" if f_ > 0 else "뒤"] += 1
    else:
        _dir["왼" if l_ > 0 else "오른"] += 1
P(f"   낙상 방향 — 뒤 {_dir['뒤']} · 앞 {_dir['앞']} · "
  f"왼 {_dir['왼']} · 오른 {_dir['오른']}")
# 몸통 피치 진동 진폭 — 추세를 뺀 나머지의 RMS (실물 떨림 2~4°)
_amp = []
for e in range(N):
    n = min(W, max(death[e], 2))
    if n < 10:
        continue
    y = fwd[:n, e]
    _amp.append(float(np.std(y - np.poly1d(np.polyfit(t[:n], y, 1))(t[:n]))))
P(f"   몸통 피치 진동 RMS(추세 제거) — 중앙 {np.median(_amp):.2f}° "
  f"p95 {np.percentile(_amp, 95):.2f}°" if _amp else "   진동 표본 부족")
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
if smth is not None:
    P("")
    P("■ 떨림 지표 (실물 5차 v6b: 1~4Hz · RMS 2~4° · E_smth 4.40)")
    P(f"    E_smth (액션 2차차분)  {smth:.5f}")
    P(f"    우세 주파수            {domf:.2f} Hz")
    P(f"    >5Hz 에너지 비율        {hi5:.1f} %")
P("")
P("■ 물리 실현성 (AK 모터 한계 대비, 생존 env-step) — 실물이 이 동작을 낼 수 있는가")
try:
    _jn = list(robot.data.joint_names)
    _n_as = max(1, _alive_n)
    _lim0 = _eff_lim[0].cpu().numpy(); _vl0 = _vel_lim[0].cpu().numpy()
    _tp = _tau_peak.cpu().numpy(); _qp = _qd_peak.cpu().numpy()
    _ts = 100 * _tau_sat.cpu().numpy() / _n_as; _qs = 100 * _qd_sat.cpu().numpy() / _n_as
    _trms = np.sqrt(_tau_rms2.cpu().numpy() / _n_as)
    P(f"    {'관절':<20} {'τ피크':>7} {'/한계':>6} {'τRMS':>6} {'τ포화%':>7} {'|qd|피크':>8} {'/한계':>6} {'qd포화%':>7}")
    for k, n in enumerate(_jn):
        P(f"    {n:<20} {_tp[k]:7.2f} {_lim0[k]:6.1f} {_trms[k]:6.2f} {_ts[k]:7.2f} {_qp[k]:8.2f} {_vl0[k]:6.1f} {_qs[k]:7.2f}")
    _worst_t = int(np.argmax(_ts)); _worst_q = int(np.argmax(_qs))
    P(f"    토크 포화 최대 {_jn[_worst_t]} {_ts[_worst_t]:.2f}% · 속도 포화 최대 {_jn[_worst_q]} {_qs[_worst_q]:.2f}% "
      f"· 피크/한계 최대 {np.max(_tp / _lim0) * 100:.0f}%")
    if _roll_log:
        # 0905 감사: 낙상·리셋 env 까지 섞여 p95 오염 → env 별 생존 구간(death 이전)만
        _r = np.stack(_roll_log)                                   # (S', N)
        _dn = death.cpu().numpy() if hasattr(death, "cpu") else death
        _rr_ = []
        for _e in range(_r.shape[1]):
            _n = int(min(_r.shape[0], max(_dn[_e], 2)))
            if _n >= 10:
                _y = _r[:_n, _e]; _rr_.append(float(np.std(_y - _y.mean())))
        if _rr_:
            P(f"    롤 진동 RMS(추세 제거, 생존 구간) — 중앙 {np.median(_rr_):.2f}° p95 {np.percentile(_rr_, 95):.2f}°")
    if _asym_n:
        P(f"    좌우 접촉력 비대칭 |FL−FR|/(FL+FR) 평균 {100 * _asym_sum / _asym_n:.1f}%")
    if _land_dx:
        _dx = np.concatenate(_land_dx) * 1000; _dy = np.concatenate(_land_dyaw)
        P(f"    착지 기하 비대칭(1~3s) — 발 전후차 |L−R| 중앙 {np.median(np.abs(_dx)):.1f}mm p95 {np.percentile(np.abs(_dx), 95):.1f}mm "
          f"· 발 요차 중앙 {np.median(np.abs(_dy)):.1f}° p95 {np.percentile(np.abs(_dy), 95):.1f}°")
    if _slip_n:
        P(f"    foot slip — 접촉 중 발 수평속도 평균 {100 * _slip_sum / _slip_n:.2f} cm/s · >2cm/s 비율 {100 * _slip_hits / _slip_n:.1f}%")
except Exception as _exc:                                     # noqa: BLE001
    P(f"    (산출 실패: {_exc})")
P("")
P("■ 관절 한계 도달 비율 (한계 1° 이내 체류 시간 %) — 5% 이상만 표기")
_bad = sorted(((float(_satr[k]), _jnames[k]) for k in range(len(_jnames))),
              reverse=True)
if _bad and _bad[0][0] >= 5.0:
    for v, n in _bad:
        if v >= 5.0:
            P(f"    {n:<24} {v:6.1f} %")
    P("    ⚠ 한계에 붙어 사는 관절 = 보상 우회 해 의심 (v10 힙롤 +30.00° 전례)")
elif _bad:
    P(f"    없음 — 최대 {_bad[0][1]} {_bad[0][0]:.1f}%")

if eq_pose:
    P("")
    P("■ 평형 자세 (정착 구간 평균, 관절 프레임 deg) — posture_probe 전이 목표")
    for n in jn_names:
        P(f"    {n:<22} {eq_pose[n]:+7.2f}")
    P("  POSE_DICT " + json.dumps(eq_pose, ensure_ascii=False))
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
