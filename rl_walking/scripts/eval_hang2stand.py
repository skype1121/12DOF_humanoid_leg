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
parser.add_argument("--noise_off", type=str, default="", help="관측 노이즈 채널 제거(쉼표): gyro,grav,jpos,jvel,contact,all — S8 원인 분해용")
parser.add_argument("--noise_scale", type=float, default=1.0, help="남은 관측 노이즈 대역 스케일(0.25/0.5/0.75/1.0)")
parser.add_argument("--no_obs_noise", action="store_true", help="관측 노이즈(enable_corruption) 끔 — 정책 고유 정지도 측정용(학습 노이즈 ±0.2rad/s 자이로 등이 잔떨림을 만드는지 분리)")
parser.add_argument("--lpf_beta", type=float, default=None, help="배포 체인 stand LPF β 덮어쓰기 (기본 deploy_chain.STAND_LPF_BETA=0.27; 작을수록 더 부드러움) — 재학습 없이 스탠바이 흔들림을 잡는 배포측 레버 시험")
parser.add_argument("--imu_roll_bias_deg", type=float, default=None, help="IMU 롤 장착 편향을 전 env 상수로 고정 [deg] (0910 실물 재현: 실물 콘솔 lat +1.0~1.4°)")
parser.add_argument("--com_fix_y", type=float, default=0.0, help="횡 CoM 고정 편심 [m] (0910 진단: 실물 편하중 재현)")
parser.add_argument("--contact_perturb", type=str, default="none",
                    help="actor 접촉 obs 교란(정답·물리·보상 불변, 0911 접촉 의존도 평가): "
                         "none/hold(10Hz S&H)/delay3(60ms)/realistic(S&H+한발누락)/onefoot(오른발 버스트0)/"
                         "chatter(랜덤 플립)/both1/both0(진단 고정 대조군)")
parser.add_argument("--perturb_seed", type=int, default=0, help="접촉 교란 RNG seed (독립 seed 평가용)")
parser.add_argument("--seed", type=int, default=None, help="env/torch seed 덮어쓰기 (독립 seed 평가용; 기본 agent_cfg.seed 고정 → 같은 설정은 결정론적 동일 결과)")
parser.add_argument("--calib_fixed", type=str, default="", help="관절 고정 편향 주입 'left_ankle_r_joint:3.6,right_hip_r_joint:3.4' [deg] — 실측 좌우 비대칭 재현")
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
if args_cli.seed is not None:   # 독립 seed 평가(0911): 기본은 agent_cfg.seed 고정 → 같은 설정은 결정론적 동일 결과
    env_cfg.seed = int(args_cli.seed); agent_cfg.seed = int(args_cli.seed)
    print(f"[seed] env/agent seed = {args_cli.seed}")
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
_off = {t.strip() for t in args_cli.noise_off.split(",") if t.strip()}
_P = env_cfg.observations.policy
for _nm, _term in (("gyro", _P.base_ang_vel), ("grav", _P.projected_gravity), ("jpos", _P.joint_pos), ("jvel", _P.joint_vel)):
    if _nm in _off or "all" in _off:
        _term.noise = None
    elif _term.noise is not None and args_cli.noise_scale != 1.0:
        _term.noise.n_min = float(_term.noise.n_min) * args_cli.noise_scale
        _term.noise.n_max = float(_term.noise.n_max) * args_cli.noise_scale
for _nm, _ch in (("gyro_ema", "gyro"), ("joint_vel_ema", "jvel")):          # S8 raw+EMA 항: 내부 노이즈도 같은 규칙
    _t = getattr(_P, _nm, None)
    if _t is not None and "noise" in _t.params:
        if _ch in _off or "all" in _off or args_cli.no_obs_noise:
            _t.params["noise"] = 0.0
        else:
            _t.params["noise"] = float(_t.params["noise"]) * args_cli.noise_scale
if ("contact" in _off or "all" in _off) and "threshold_range" in _P.feet_contact.params:
    _P.feet_contact.params["threshold_range"] = None
if _off or args_cli.noise_scale != 1.0:
    print(f"[noise] off={sorted(_off)} scale={args_cli.noise_scale}")
if args_cli.calib_fixed and getattr(env_cfg.events, "calib_bias", None):
    _fb = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in args_cli.calib_fixed.split(",") if ":" in kv}
    env_cfg.events.calib_bias.params["fixed_bias_deg"] = _fb
    print(f"[calib] 고정 편향 주입 {_fb}")
if args_cli.imu_roll_bias_deg is not None and getattr(env_cfg.events, "calib_bias", None):
    _rb = math.radians(args_cli.imu_roll_bias_deg)
    env_cfg.events.calib_bias.params["imu_roll_rad"] = (_rb, _rb)
    print(f"[imu] 롤 편향 고정 {args_cli.imu_roll_bias_deg:+.2f}°")
if getattr(env_cfg.events, "base_com", None) and (args_cli.com_fix is not None or args_cli.com_fix_y):
    if args_cli.com_fix is not None:
        # 종전 동작 유지: x 고정 + y 고정(com_fix_y, 기본 0) + z 0  (results_table 비교 호환)
        _cy = args_cli.com_fix_y or 0.0
        env_cfg.events.base_com.params["com_range"] = {"x": (args_cli.com_fix, args_cli.com_fix), "y": (_cy, _cy), "z": (0.0, 0.0)}
    else:
        # ★정정(0911, COMY_FIX): --com_fix_y 단독은 종전엔 --com_fix 없이는 통째로 무시됐다(죽은 인자 —
        # ±0.01/±0.02 4런이 기준선과 자릿수까지 동일해 발견). x·z 는 env 범위 유지, y 만 고정.
        _cr = dict(env_cfg.events.base_com.params.get("com_range", {}))
        _cr["y"] = (args_cli.com_fix_y, args_cli.com_fix_y)
        env_cfg.events.base_com.params["com_range"] = _cr
    print(f"[com] base_com com_range = {env_cfg.events.base_com.params['com_range']}")
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
_ss_ar2 = torch.zeros(N, device=dev); _ss_r2 = torch.zeros(N, device=dev); _ss_p2 = torch.zeros(N, device=dev); _ss_rpk = torch.zeros(N, device=dev); _ss_ppk = torch.zeros(N, device=dev); _prev_a = None
_ss_aa2 = torch.zeros(N, device=dev); _ss_pm = torch.zeros(N, device=dev); _prev_a2 = None   # 액션 2차차분(채터 겨냥 보상의 직접 지표)·정착 피치 부호 평균
_air_q = torch.zeros(N, 12, device=dev); _air_n = torch.zeros(N, device=dev)   # 0910: 공중(지지비>0.9) 관절각 평균 — 실물 공중 비대칭(왼 힙a +11 vs 오른 +6) 대조
_jn = list(robot.joint_names); _jidx = {n: i for i, n in enumerate(_jn)}
try:
    from rl_walking.mdp_hang import _feet_cfg as _fcfg, _support_gone_mask
    _fc = _fcfg(); _fc.resolve(env.unwrapped.scene); _FEET = _fc.name; _FEET_IDS = _fc.body_ids
except Exception:
    _FEET, _FEET_IDS = None, None

# ── 접촉 obs 교란 (0911 접촉 의존도 평가) — actor 관측의 feet_contact 슬라이스만 변조.
#    보상(feet_contact_gated)·종료·평가 정답은 센서를 직접 읽으므로 불변 = 교란은 actor 에만. ──
_cs_lo = _cs_hi = None
try:
    _om = uenv.observation_manager
    _terms = _om.active_terms["policy"]; _dims = _om.group_obs_term_dim["policy"]
    _off = 0; _layout = []
    for _nm, _dm in zip(_terms, _dims):
        _d = _dm[0] if isinstance(_dm, (list, tuple)) else int(_dm)
        _layout.append((_nm, _off, _d))
        if _nm == "feet_contact":
            _cs_lo, _cs_hi = _off, _off + _d
        _off += _d
    if args_cli.contact_perturb != "none":
        print(f"[obs 레이아웃] policy dim={_off}, feet_contact 슬라이스={_cs_lo}:{_cs_hi} "
              f"(항 {len(_terms)}개). 교란={args_cli.contact_perturb} seed={args_cli.perturb_seed}")
        if _cs_lo is None:
            print("[경고] feet_contact 항을 못 찾음 — 교란 미적용")
except Exception as _e:
    print(f"[경고] obs 레이아웃 조회 실패: {_e} — 교란 미적용")


class ContactPerturb:
    """actor 접촉 obs 교란기. obs[:, cs:ce] (L/R 이진 접촉)만 변조. 정답/물리/보상 불변."""
    def __init__(self, mode, N, dev, cs, ce, seed=0):
        self.mode, self.N, self.dev, self.cs, self.ce = mode, N, dev, cs, ce
        self.g = torch.Generator(device="cpu"); self.g.manual_seed(int(seed))
        self.held = None; self.buf = []; self.every = 5   # 50Hz→10Hz S&H

    def _rand(self, shape, p):
        return (torch.rand(shape, generator=self.g).to(self.dev) < p)

    def __call__(self, obs, i):
        if self.mode == "none" or self.cs is None:
            return obs
        # rsl_rl 래퍼는 TensorDict(키 'policy')를 준다 — 텐서 슬라이싱은 TensorDict 에 못 쓰므로
        # 'policy' 텐서만 꺼내 변조하고 다시 담는다 (critic 키 등은 불변).
        td = None
        t = obs
        if not torch.is_tensor(obs):
            td = obs
            t = obs["policy"]
        c = t[:, self.cs:self.ce]
        p = c.clone()
        if self.mode == "both1":
            p[:] = 1.0
        elif self.mode == "both0":
            p[:] = 0.0
        elif self.mode == "hold":
            if self.held is None or i % self.every == 0:
                self.held = c.clone()
            p = self.held.clone()
        elif self.mode == "delay3":
            self.buf.append(c.clone())
            if len(self.buf) > 3:
                p = self.buf.pop(0)
        elif self.mode == "onefoot":            # 오른발(idx1) 2s 주기·앞 0.3s 0 오판 (실물 오른발 언로드 모사)
            if (i % 100) < 15:
                p[:, 1] = 0.0
        elif self.mode == "chatter":            # 랜덤 단발 누락(양발 각 8%/스텝 0)
            p = p * (~self._rand(c.shape, 0.08)).float()
        elif self.mode == "realistic":          # 10Hz S&H + 지터 + 한발 짧은 누락(각 3%/스텝)
            if self.held is None or i % self.every == 0:
                self.held = c.clone()
            p = self.held.clone() * (~self._rand(c.shape, 0.03)).float()
        t = t.clone()
        t[:, self.cs:self.ce] = p
        if td is not None:
            td = td.clone()
            td["policy"] = t
            return td
        return t


_cperturb = ContactPerturb(args_cli.contact_perturb, N, dev, _cs_lo, _cs_hi, args_cli.perturb_seed)
t0 = time.time()
for i in range(STEPS):
    with torch.inference_mode():
        obs = _cperturb(obs, i)          # actor 접촉 obs 교란(정답/물리 불변) — none 이면 무변
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
    _post = a_ & _support_gone_mask(st)   # item 4c: 케이블이면 실제 장력<=eps (힘 모델은 r<=0 동일)
    _both_now = (contact[:, 0] > 0) & (contact[:, 1] > 0)
    _ever_both |= (a_ & _both_now)
    _lift_evt = (a_ & _ever_both & _prev_both & ~_both_now).float()
    _lift_cnt += _lift_evt
    _air = a_ & (st["r"] > 0.9)
    if _air.any():
        _air_q[_air] += robot.data.joint_pos[_air]; _air_n[_air] += 1.0
    if i >= _SS_FROM:
        _ss = a_ & _support_gone_mask(st)   # item 4c: 실제 지지 소멸 (스케줄 아님)
        _ss_w[_ss] += robot.data.root_ang_vel_b[_ss, :2].norm(dim=-1); _ss_v[_ss] += robot.data.root_lin_vel_b[_ss, :2].norm(dim=-1)
        _ss_qd2[_ss] += (robot.data.joint_vel[_ss] ** 2).mean(1); _ss_n[_ss] += 1.0; _ss_lift += _lift_evt
        if _prev_a is not None:
            _ss_ar2[_ss] += ((a - _prev_a)[_ss] ** 2).mean(1)
            if _prev_a2 is not None:
                _ss_aa2[_ss] += ((a - 2.0 * _prev_a + _prev_a2)[_ss] ** 2).mean(1)
        _ss_r2[_ss] += roll[_ss] ** 2; _ss_p2[_ss] += pitch[_ss] ** 2; _ss_pm[_ss] += pitch[_ss]
        _ss_rpk[_ss] = torch.maximum(_ss_rpk[_ss], roll[_ss].abs()); _ss_ppk[_ss] = torch.maximum(_ss_ppk[_ss], pitch[_ss].abs())
    _prev_a2 = None if _prev_a is None else _prev_a.clone(); _prev_a = a.clone()
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
# 0911 Codex 지적 수정: '안 넘어짐(~fell)'만으로 성공 처리하면 방출이 늦어 방출 후 stand_s 를 못 채운 env 도 성공으로 셈.
# 성공 = 방출 도달 & 방출 후 stand_s 이상 생존(넘어지지 않고). 방출 후 stand_s 미만은 '미검증'으로 성공 제외.
stand5 = released & (~fell) & (surv_after >= args_cli.stand_s)
_inconcl = int((released & (~fell) & (surv_after < args_cli.stand_s)).sum())  # 방출은 됐으나 관측창이 짧아 미검증
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
P(f"**지지 0 이후 {args_cli.stand_s:.0f}s 생존**(미검증 {_inconcl}: 방출 후 관측창 <{args_cli.stand_s:.0f}s)")
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
    P(f"   정상상태 추가: action-rate RMS {float(((_ss_ar2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.4f} · 롤 RMS {float(((_ss_r2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.2f}° 피크 {float(_ss_rpk[_ok2].mean()):.2f}° · "
      f"피치 RMS {float(((_ss_p2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.2f}° 피크 {float(_ss_ppk[_ok2].mean()):.2f}° · "
      f"액션 2차차분 RMS {float(((_ss_aa2[_ok2]/_ss_n[_ok2]).sqrt()).mean()):.4f} · 정착 피치 평균 {float((_ss_pm[_ok2]/_ss_n[_ok2]).mean()):+.2f}°")
    _flz = robot.data.net_forces_w if False else None  # (placeholder)
    try:
        _fs = env.unwrapped.scene.sensors[_FEET].data.net_forces_w[:, _FEET_IDS, 2].clamp(min=0.0)
        _fl = float(_fs[_ok2, 0].mean()); _fr = float(_fs[_ok2, 1].mean())
        P(f"정상상태 하중: 왼 {_fl:.0f}N / 오른 {_fr:.0f}N · 불균형 {abs(_fl-_fr)/max(_fl+_fr,1):.2f}")
    except Exception as _e:
        pass
_okA = _air_n > 0
if _okA.any():
    _mq = torch.rad2deg((_air_q[_okA] / _air_n[_okA, None]).mean(0))
    def _J(n): return float(_mq[_jidx[n]])
    P(f"공중 자세(지지비>0.9, 관절각 평균°): 힙a L {_J('left_hip_a_joint'):+.1f} / R {_J('right_hip_a_joint'):+.1f} · 힙f L {_J('left_hip_f_joint'):+.1f} / R {_J('right_hip_f_joint'):+.1f} · "
      f"힙r L {_J('left_hip_r_joint'):+.1f} / R {_J('right_hip_r_joint'):+.1f} · 무릎 L {_J('left_knee_joint'):+.1f} / R {_J('right_knee_joint'):+.1f} · 발목f L {_J('left_ankle_f_joint'):+.1f} / R {_J('right_ankle_f_joint'):+.1f}"
      f"   (실물 0910 공중: 힙a +11/+6 · 힙f −10/−0.5 · 힙r +3/+8 · 무릎 −0.6/−10)")
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
