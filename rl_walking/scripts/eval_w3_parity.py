"""W3 배포 parity 폐루프 평가기(1007 parity 감사) — 배포와 시뮬 사이 차이 항목을 env 그룹별로 바꿔 같은 실행 안에서 비교한다.

eval_w3.py 와 같은 평가 조건(학습 태스크·학습 DR·관측 잡음 유지, 커리큘럼·리샘플·외란 끔, LPF β 0.27, 보조 0)에서
그룹마다 아래 축을 바꾼다(env i → 그룹 i mod G, 교차 배치). 기존 평가기·환경 코드는 바꾸지 않는다(관측 함수 교체·훅은 이 스크립트 안).

그룹 키(그룹 문법 'name:key=val,key=val;name2:...'):
  cmd=vx/vy/wz          고정 명령(기본 0/0/0). seq=<이름>,dwell=<s> 이면 명령 순환(아래 SEQS) — 배포 최소 체류 1 s 시험용.
  hl=none|stale|same    외부 헤딩 P 고리(K --hl_kp, ±--hl_wzmax). stale = eval_w3 방식(step 전 요 = 한 틱 전), same = 배포 방식(관측 직전 요).
                        고리는 STEP(want 1)·wz₀ = 0 일 때만(배포 BridgeCore._w3_cmd_step 규약), 켜지는 틱에 ψ* 래치.
  contact=train|notilt|notilt_h1|train_dK|notilt_dK   발 접촉 관측: train = 학습 계약 v2(힘 ∧ 발 평평 + 10 Hz 유지 + 1 틱),
                        notilt = 평평 조건 없음(힘만, 유지·지연 같음), _h1 = 유지 없음(매 틱), _dK = 전송 지연 K 틱.
  delay=K               액션 지연 고정 K 틱(1–7; 학습 1–5).
  enc=raw|q|qb          관절 엔코더 관측: raw = 시뮬 계약(연속값·매 틱 단순 차분), q = MIT 16 bit 양자화(±12.5 rad → 3.81e-4 rad) + 단순 차분,
                        qb = 양자화 + 브리지 _tick qvel 식(같은 값 반복 시 0, 바뀌면 Δ/(dt·머문 틱), 캡 12) — 실물 블랙박스 1001–1004 반복 53–68 % 대조용.
  stand=s               서기 시간(기본 --stand_first) 뒤 STEP.   kp=K  그룹별 헤딩 고리 이득(기본 --hl_kp).
  clamp=train|s8        최종 관절 목표 클램프: train = 학습(config soft 를 발목 앞뒤만), s8 = stage8 노드와 같이 **전 관절** config soft 경계 클램프
                        (stage8_12axis_mit_control_node: soft 초과 → 경계 클램프, 25° 초과는 거부 — 여기선 클램프만).
  push=F                몸통 수평 밀기 F[N] 0.2 s 를 채점 구간 t = 서기+워밍업+{8, 23, 38, 53} s 에(60 s 기준), 방향은 env 마다 앞·뒤·왼·오 순환
                        (몸통 요 프레임 → 월드, eval_w3 --dist 와 같은 permanent_wrench_composer 경로). 낙상은 채점 낙상에 포함.
지표(채점 = 워밍업 뒤): 낙상(종료), 유효 스텝률(env._w3 유효/예정 반주기), 시작 요 프레임 평균 속도, 요 속도·요 표류, 기울기 RMS·최대,
  |ω_xy| RMS, 고리 wz 분포(활성 틱 중 0<|wz|<0.1 비율 = 학습 밖 크기). 판정은 산출물(.txt 끝 COMPLETE).
사용: isaaclab.sh -p rl_walking/scripts/eval_w3_parity.py --headless --load_run <run> --checkpoint model_N.pt --num_envs 1024 --groups '...' --out_tag <t>
"""
import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402

REPO = os.environ.get("W3EVAL_REPO", "/home/ryu/humanoid_leg_test1")
CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, CODE)

p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W3S3d-v0")
p.add_argument("--num_envs", type=int, default=512)
p.add_argument("--seconds", type=float, default=30.0, help="채점 구간[s]")
p.add_argument("--warmup", type=float, default=2.0, help="STEP 시작 뒤 채점 전 안정화[s]")
p.add_argument("--stand_first", type=float, default=1.0)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--hl_kp", type=float, default=0.5)
p.add_argument("--hl_wzmax", type=float, default=0.4)
p.add_argument("--groups", required=True)
p.add_argument("--out_tag", default="par")
p.add_argument("--out_dir", default=os.path.join(CODE, "log_picture"))
cli_args.add_rsl_rl_args(p)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
os.environ["STEP_ASSIST_ALPHA"] = "0.0"
os.environ["W3_GAIT_ENABLE"] = "0"
os.environ.setdefault("TURN_ALPHA", "0")
os.environ["W3_HLOOP"] = "0"
for _k in ("W3_CMD", "W3_REF_SCALE", "W3_CLOCK_DIAG", "FREEZE_GAIT_CLOCK"):
    os.environ.pop(_k, None)

SEQS = {
    # 배포 검증기가 허용하는 단일 축 명령만(학습 범위 안). 연속 명령이 늘 다르도록 배치.
    "mix7": [(0.06, 0.0, 0.0), (0.0, 0.035, 0.0), (0.0, 0.0, 0.15), (-0.04, 0.0, 0.0), (0.0, -0.035, 0.0), (0.0, 0.0, -0.15), (0.10, 0.0, 0.0)],
    "fb": [(0.10, 0.0, 0.0), (-0.06, 0.0, 0.0)],
    "tf": [(0.0, 0.0, 0.15), (0.06, 0.0, 0.0), (0.0, 0.0, -0.15), (0.06, 0.0, 0.0)],     # 회전 → 앞(헤딩 고리 래치 직후 잔여 회전)
}


def parse_groups(s):
    out = []
    for g in [x.strip() for x in s.split(";") if x.strip()]:
        name, _, kv = g.partition(":")
        d = {"cmd": (0.0, 0.0, 0.0), "hl": "none", "contact": "train", "delay": 3, "seq": None, "dwell": 0.0, "stand": a.stand_first, "enc": "raw", "kp": a.hl_kp, "clamp": "train", "push": 0.0}
        for tok in [t.strip() for t in kv.split(",") if t.strip()]:
            k, _, v = tok.partition("=")
            if k == "cmd":
                d["cmd"] = tuple(float(x) for x in v.split("/"))
                assert len(d["cmd"]) == 3
            elif k in ("hl", "contact", "seq", "enc", "clamp"):
                d[k] = v
            elif k == "delay":
                d[k] = int(v)
            elif k in ("dwell", "stand", "kp", "push"):
                d[k] = float(v)
            else:
                raise ValueError(f"모르는 그룹 키 {k}")
        assert d["hl"] in ("none", "stale", "same"), d
        assert d["enc"] in ("raw", "q", "qb"), d
        assert d["clamp"] in ("train", "s8"), d
        assert d["seq"] is None or d["seq"] in SEQS, d
        assert 1 <= d["delay"] <= 7
        out.append((name, d))
    return out


GROUPS = parse_groups(a.groups)
app = AppLauncher(a).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import rl_walking  # noqa: F401,E402
from rl_walking import mdp_hang  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

G = len(GROUPS)


def _contact_params(mode):
    """(원본 사용, 평평 조건, 유지 틱, 지연 틱)."""
    if mode == "train":
        return True, True, 5, 1
    tilt = not mode.startswith("notilt")
    hold, dly = 5, 1
    rest = mode.split("_")[1:]
    for r in rest:
        if r == "h1":
            hold = 1
        elif r.startswith("d"):
            dly = int(r[1:])
        elif r not in ("",):
            raise ValueError(mode)
    return False, tilt, hold, dly


_orig_fc = mdp_hang.feet_contact_real_v2


def fc_variant(env, sensor_cfg=None, threshold: float = 5.0, threshold_range=None, hold_ticks: int = 5, extra_delay: int = 1,
               tilt_off_deg=(6.0, 20.0), asset_cfg=None):
    """원본(학습 계약 v2)을 그대로 계산해 상태·난수 소비를 보존하고, 변형 그룹 env 만 같은 임계·위상으로 다른 계약을 낸다."""
    out = _orig_fc(env, sensor_cfg, threshold, threshold_range, hold_ticks, extra_delay, tilt_off_deg, asset_cfg)
    C = getattr(env, "_fcv", None)
    if C is None:
        return out
    step = int(env.common_step_counter)
    if C["step"] != step:
        st = env._fcr2_st
        sensor = env.scene.sensors[sensor_cfg.name]
        fn = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
        Nn = fn.shape[0]
        now = (fn > st["fthr"]).to(torch.float32)
        asset = env.scene["robot"]
        q = asset.data.body_quat_w[:, st["ids"], :]
        g = quat_apply_inverse(q.reshape(-1, 4), torch.tensor([0.0, 0.0, -1.0], device=fn.device).expand(Nn * 2, 3)).reshape(Nn, 2, 3)
        tilt = torch.acos((-g[..., 1]).clamp(-1.0, 1.0))
        bit = torch.where(C["tilt"].view(-1, 1), now * (tilt < st["thr"]).to(now.dtype), now)
        fresh = env.episode_length_buf == 0
        upd = ((step + st["phase"]) % C["hold"] == 0) | fresh
        C["held"] = torch.where(upd.view(-1, 1), bit, C["held"])
        C["q"] = torch.roll(C["q"], 1, 0)
        C["q"][0] = C["held"]
        if bool(fresh.any()):
            C["q"][:, fresh] = C["held"][fresh].unsqueeze(0)
        C["out"] = C["q"][C["dly"], torch.arange(Nn, device=fn.device)]
        C["step"] = step
    return torch.where(C["orig"].view(-1, 1), out, C["out"])


QSTEP = 25.0 / 65535.0          # MIT 프로토콜 위치 16 bit, ±12.5 rad
_orig_encpos = mdp_hang.encoder_joint_pos_v2
_orig_encvel = mdp_hang.encoder_joint_vel_v2


def encpos_q(env, asset_cfg=None, action_term: str = "joint_pos", play_rad=None):
    enc = _orig_encpos(env, asset_cfg, action_term, play_rad)
    Q = getattr(env, "_encq", None)
    if Q is None:
        return enc
    return torch.where(Q["q"].view(-1, 1), torch.round(enc / QSTEP) * QSTEP, enc)


def encvel_q(env, asset_cfg=None, action_term: str = "joint_pos", play_rad=None):
    v = _orig_encvel(env, asset_cfg, action_term, play_rad)          # 안에서 encoder_joint_pos_v2(= encpos_q) 를 부른다 → q env 는 양자화 단순 차분
    Q = getattr(env, "_encq", None)
    if Q is None or not bool(Q["b"].any()):
        return v
    step = int(env.common_step_counter)
    if Q["step"] != step:
        pos = encpos_q(env, asset_cfg, action_term, play_rad)
        fresh = (env.episode_length_buf == 0).view(-1, 1)
        if Q["prev"] is None:
            Q["prev"] = pos.clone(); Q["hold"] = torch.ones_like(pos)
        rep_ = pos == Q["prev"]
        vb = torch.where(rep_, torch.zeros_like(pos), (pos - Q["prev"]) / (float(env.step_dt) * Q["hold"]))
        Q["hold"] = torch.where(rep_, (Q["hold"] + 1.0).clamp(max=12.0), torch.ones_like(Q["hold"]))
        vb = torch.where(fresh, torch.zeros_like(vb), vb)
        Q["hold"] = torch.where(fresh, torch.ones_like(Q["hold"]), Q["hold"])
        Q["prev"] = pos.clone(); Q["vb"] = vb; Q["step"] = step
    return torch.where(Q["b"].view(-1, 1), Q["vb"], v)


env_cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, a)
env_cfg.scene.num_envs = a.num_envs
env_cfg.seed = a.seed
env_cfg.curriculum = None
c = env_cfg.commands.base_velocity
c.resampling_time_range = (1e6, 1e6)
c.rel_standing_envs = 0.0
for _ev in ("disturbance", "push_robot"):
    if getattr(env_cfg.events, _ev, None) is not None:
        setattr(env_cfg.events, _ev, None)
act = env_cfg.actions.joint_pos
act.lpf_beta_range = (0.27, 0.27)
act.min_delay_steps, act.max_delay_steps = 1, 7
if hasattr(act, "delay_probs"):
    act.delay_probs = ()
if any(d["contact"] != "train" for _, d in GROUPS):
    env_cfg.observations.policy.feet_contact.func = fc_variant
if any(d["enc"] != "raw" for _, d in GROUPS):
    mdp_hang.encoder_joint_pos_v2 = encpos_q
    mdp_hang.encoder_joint_vel_v2 = encvel_q
T_END = max(d["stand"] for _, d in GROUPS) + a.warmup + a.seconds
env_cfg.episode_length_s = T_END + 5.0
env = gym.make(a.task, cfg=env_cfg)
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
U = env.unwrapped
dev = U.device
N = U.num_envs
dt = U.step_dt
log_root = os.path.abspath(os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name))
ck = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ck)
policy = runner.get_inference_policy(device=dev)
robot = U.scene["robot"]
cmd_term = U.command_manager.get_term("base_velocity")
term = U.action_manager.get_term("joint_pos")
gid = torch.arange(N, device=dev) % G


def per_env(fn, dtype=torch.float32):
    return torch.tensor([fn(GROUPS[int(i)][1]) for i in gid.cpu()], device=dev, dtype=dtype)


term._delay[:] = per_env(lambda d: d["delay"], torch.long)
cp = [_contact_params(d["contact"]) for _, d in GROUPS]
if any(d["contact"] != "train" for _, d in GROUPS):
    Dm = max(x[3] for x in cp)
    U._fcv = {"orig": torch.tensor([cp[int(i)][0] for i in gid.cpu()], device=dev),
              "tilt": torch.tensor([cp[int(i)][1] for i in gid.cpu()], device=dev),
              "hold": torch.tensor([cp[int(i)][2] for i in gid.cpu()], device=dev, dtype=torch.long),
              "dly": torch.tensor([cp[int(i)][3] for i in gid.cpu()], device=dev, dtype=torch.long),
              "held": torch.zeros(N, 2, device=dev), "q": torch.zeros(Dm + 1, N, 2, device=dev), "out": torch.zeros(N, 2, device=dev), "step": None}
hl_mode = per_env(lambda d: {"none": 0, "stale": 1, "same": 2}[d["hl"]], torch.long)
hl_kp_env = per_env(lambda d: d["kp"]).to(torch.float64)        # 그룹 키 kp(기본 --hl_kp)
push_F = per_env(lambda d: d["push"])
push_dir = (torch.arange(N, device=dev) // G) % 4                 # 0 앞 1 뒤 2 왼 3 오
_bn = list(robot.body_names); PUSH_BODY = _bn.index("base") if "base" in _bn else 0
if any(d["clamp"] == "s8" for _, d in GROUPS):
    sys.path.insert(0, CODE)
    from robot_runtime.joint_limits import get_absolute_limits_deg as _gl
    _L = _gl(); _jn = [robot.joint_names[i] for i in (term._joint_ids if not isinstance(term._joint_ids, slice) else range(len(robot.joint_names)))]
    S8_LO = torch.tensor([math.radians(_L[j]["soft_min"]) for j in _jn], device=dev)
    S8_HI = torch.tensor([math.radians(_L[j]["soft_max"]) for j in _jn], device=dev)
    S8_M = per_env(lambda d: d["clamp"] == "s8", torch.bool).view(-1, 1)
    U._s8_clamp_n = torch.zeros(N, len(_jn), device=dev)
    _orig_pa = term.process_actions

    def _pa_s8(actions):
        _orig_pa(actions)
        x = term._processed_actions
        c = torch.maximum(torch.minimum(x, S8_HI), S8_LO)
        U._s8_clamp_n += (S8_M & (c != x)).float()
        term._processed_actions = torch.where(S8_M, c, x)
    term.process_actions = _pa_s8
if any(d["enc"] != "raw" for _, d in GROUPS):
    U._encq = {"q": per_env(lambda d: d["enc"] in ("q", "qb"), torch.bool), "b": per_env(lambda d: d["enc"] == "qb", torch.bool),
               "prev": None, "hold": None, "vb": None, "step": None}
stand_t = per_env(lambda d: d["stand"])
base_fixed = torch.stack([per_env(lambda d, k=k: d["cmd"][k]) for k in range(3)], -1)
seq_id = [d["seq"] for _, d in GROUPS]
dwell = per_env(lambda d: d["dwell"])


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def base_cmd_at(t):
    cm = base_fixed.clone()
    for gi, (_, d) in enumerate(GROUPS):
        if d["seq"] is None:
            continue
        sq = SEQS[d["seq"]]
        k = int(max(0.0, t - d["stand"]) // d["dwell"]) % len(sq)
        cm[gid == gi] = torch.tensor(sq[k], device=dev)
    return cm


HL = {"tgt": torch.zeros(N, device=dev, dtype=torch.float64), "lat": torch.zeros(N, dtype=torch.bool, device=dev),
      "act": torch.zeros(N, dtype=torch.bool, device=dev), "wz0": torch.zeros(N, device=dev, dtype=torch.float64),
      "wz": torch.zeros(N, device=dev, dtype=torch.float64)}


def hl_update(mask):
    """mask env 의 고리 갱신 — 배포 W3HeadingLoop 식(float64): 래치 틱 ψ* = ψ, 이후 ψ* ← wrap(ψ* + wz₀·dt), wz = clip(wz₀ + K·wrap(ψ* − ψ))."""
    if not bool(mask.any()):
        return
    psi = yaw_of(robot.data.root_quat_w).to(torch.float64)
    new = mask & ~HL["lat"]
    HL["tgt"] = torch.where(new, psi, torch.where(mask, wrap(HL["tgt"] + HL["wz0"] * dt), HL["tgt"]))
    HL["lat"] = HL["lat"] | mask
    wz = (HL["wz0"] + hl_kp_env * wrap(HL["tgt"] - psi)).clamp(-a.hl_wzmax, a.hl_wzmax)
    HL["wz"] = torch.where(mask, wz, HL["wz"])
    cmd_term.vel_command_b[mask, 2] = wz[mask].to(cmd_term.vel_command_b.dtype)


om = U.observation_manager
_orig_compute = om.compute


def _compute_hook(*args, **kw):
    hl_update(HL["act"] & (hl_mode == 2))
    return _orig_compute(*args, **kw)


om.compute = _compute_hook

steps = int(round(T_END / dt))
z = lambda *s: torch.zeros(*s, device=dev)  # noqa: E731
alive = torch.ones(N, dtype=torch.bool, device=dev)
t_fall = torch.full((N,), float("nan"), device=dev)
scored = torch.zeros(N, dtype=torch.bool, device=dev)
t_score0 = stand_t + a.warmup
p0 = z(N, 2); yaw0 = z(N); yaw_acc = z(N); yaw_prev = z(N)
nv0 = z(N); ns0 = z(N)
tilt_ss = z(N); tilt_max = z(N); w_ss = z(N); n_tick = z(N)
hl_n = z(N); hl_small = z(N); hl_big = z(N); hl_abs = z(N); hl_err_max = z(N)
obs = venv.get_observations()
wall0 = time.time()
for i in range(steps):
    t = (i + 1) * dt
    want = t > stand_t
    U._ge_cmd[:] = want
    cm = base_cmd_at(t)
    cm = torch.where(want.unsqueeze(1), cm, torch.zeros_like(cm))
    cmd_term.vel_command_b[:] = cm
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False
    HL["wz0"] = cm[:, 2].to(torch.float64)
    act_hl = (hl_mode > 0) & want & (cm[:, 2] == 0)
    HL["lat"] = HL["lat"] & act_hl                      # 고리 꺼지면 ψ* 해제(배포 release)
    HL["act"] = act_hl
    hl_update(act_hl & (hl_mode == 1))                  # stale: step 전(한 틱 전 물리) 요
    if bool((push_F > 0).any()):
        _ts = t - 0.5 * dt - (stand_t + a.warmup)
        _on = torch.zeros(N, dtype=torch.bool, device=dev)
        for _p0 in (8.0, 23.0, 38.0, 53.0):
            _on |= (_ts >= _p0) & (_ts < _p0 + 0.2)
        _on &= push_F > 0
        if bool(_on.any()):
            from isaaclab.utils.math import quat_apply, quat_apply_inverse
            _q = robot.data.root_quat_w
            _ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(N, 3); _ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(N, 3)
            _vx = quat_apply(_q, _ex); _vy = quat_apply(_q, _ey); _vx[:, 2] = 0; _vy[:, 2] = 0
            _vx = _vx / _vx.norm(dim=-1, keepdim=True).clamp(min=1e-6); _vy = _vy / _vy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            _d = torch.where((push_dir == 0).unsqueeze(1), _vx, torch.where((push_dir == 1).unsqueeze(1), -_vx, torch.where((push_dir == 2).unsqueeze(1), _vy, -_vy)))
            _fw = _d * (push_F * _on.float()).unsqueeze(1)
            _fb = quat_apply_inverse(_q, _fw).unsqueeze(1).contiguous()
            robot.permanent_wrench_composer.add_forces_and_torques(forces=_fb, torques=None, positions=None, body_ids=[PUSH_BODY], env_ids=None, is_global=False)
    with torch.inference_mode():
        act_ = policy(obs)
    obs, _, dones, _ = venv.step(act_)
    d = dones.bool()
    tm = getattr(U, "termination_manager", None)
    if tm is not None:
        d = d & tm.terminated.bool()
    newly = alive & d
    t_fall[newly] = t
    alive &= ~d
    q = robot.data.root_quat_w
    yaw = yaw_of(q)
    s = U._w3
    nv = s["n_valid"].sum(1)
    ns = s["n_sched"]
    start = (~scored) & (t >= t_score0)
    if bool(start.any()):
        p0[start] = robot.data.root_pos_w[start, :2]
        yaw0[start] = yaw[start]; yaw_prev[start] = yaw[start]; yaw_acc[start] = 0.0
        nv0[start] = nv[start]; ns0[start] = ns[start]
        scored |= start
    m = scored & alive & ~start
    yaw_acc = torch.where(m, yaw_acc + wrap(yaw - yaw_prev), yaw_acc)
    yaw_prev = torch.where(scored & alive, yaw, yaw_prev)
    g = robot.data.projected_gravity_b
    tl = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
    wxy = robot.data.root_ang_vel_b[:, :2].norm(dim=-1)
    mf = m.float()
    tilt_ss += mf * tl * tl; tilt_max = torch.where(m, torch.maximum(tilt_max, tl), tilt_max); w_ss += mf * wxy * wxy; n_tick += mf
    am = m & HL["act"]
    wzv = cmd_term.vel_command_b[:, 2].abs()
    hl_n += am.float(); hl_small += (am & (wzv > 0) & (wzv < 0.1)).float(); hl_big += (am & (wzv >= 0.1)).float(); hl_abs += am.float() * wzv
    _he = wrap(HL["tgt"] - yaw.to(torch.float64)).abs().to(torch.float32)          # 고리 오차(배포 고장 문턱 15°와 비교)
    hl_err_max = torch.where(am & HL["lat"], torch.maximum(hl_err_max, _he), hl_err_max)
    if i % 500 == 0:
        print(f"[eval_w3_parity] t {t:.1f}/{T_END:.1f} 생존 {int(alive.sum())}/{N} wall {time.time() - wall0:.0f}s", flush=True)
T_sc = (n_tick).clamp(min=1) * dt
pe = robot.data.root_pos_w[:, :2]
dp = pe - p0
c0, s0 = torch.cos(yaw0), torch.sin(yaw0)
vx = (dp[:, 0] * c0 + dp[:, 1] * s0) / T_sc
vy = (-dp[:, 0] * s0 + dp[:, 1] * c0) / T_sc
yrate = yaw_acc / T_sc
vr = (U._w3["n_valid"].sum(1) - nv0) / (U._w3["n_sched"] - ns0).clamp(min=1)
rows, J = [], {"ckpt": ck, "task": a.task, "seed": a.seed, "num_envs": N, "seconds": a.seconds, "warmup": a.warmup, "hl_kp": a.hl_kp,
               "hl_wzmax": a.hl_wzmax, "groups": {}}
hdr = (f"| 그룹 | 설정 | n | 낙상 | 유효 스텝률 | vx | vy | 요 속도 °/s | 요 표류 |Δψ−wz·T| ° 평균/최대 | 기울기 RMS/최대 ° | |ω_xy| RMS | "
       "고리 0<|wz|<0.1 · ≥0.1 · 평균|wz| | 고리 오차 최대 ° 중앙/최대 · >15° env |")
rows.append(hdr)
rows.append("|" + "---|" * 14)
for gi, (name, d) in enumerate(GROUPS):
    mg = gid == gi
    n = int(mg.sum())
    nf = int((mg & ~torch.isnan(t_fall)).sum())
    sv = mg & alive
    f = lambda x: float(x[sv].mean()) if bool(sv.any()) else float("nan")  # noqa: E731
    wz_c = d["cmd"][2] if d["seq"] is None else float("nan")
    drift = (yaw_acc - wz_c * T_sc).abs() if d["seq"] is None else torch.full_like(yaw_acc, float("nan"))
    dr_mean = float(torch.rad2deg(drift[sv]).mean()) if bool(sv.any()) else float("nan")
    dr_max = float(torch.rad2deg(drift[sv]).max()) if bool(sv.any()) else float("nan")
    hn = float(hl_n[sv].sum())
    hs = f"{float(hl_small[sv].sum()) / hn:.3f} · {float(hl_big[sv].sum()) / hn:.3f} · {float(hl_abs[sv].sum()) / hn:.4f}" if hn > 0 else "—"
    trms = float((tilt_ss[sv] / n_tick[sv].clamp(min=1)).sqrt().mean()) if bool(sv.any()) else float("nan")
    wrms = float((w_ss[sv] / n_tick[sv].clamp(min=1)).sqrt().mean()) if bool(sv.any()) else float("nan")
    tmax = float(tilt_max[sv].max()) if bool(sv.any()) else float("nan")
    desc = (f"cmd {d['cmd']}" if d["seq"] is None else f"seq {d['seq']} 체류 {d['dwell']}s") + f" · hl {d['hl']} · 접촉 {d['contact']} · 지연 {d['delay']}" + (f" · 엔코더 {d['enc']}" if d['enc'] != 'raw' else "") + (f" · K {d['kp']}" if d['hl'] != 'none' and d['kp'] != a.hl_kp else "") + (" · 클램프 s8" if d['clamp'] == 's8' else "") + (f" · 밀기 {d['push']:g} N" if d['push'] > 0 else "")
    he = (f"{float(torch.rad2deg(hl_err_max[sv]).median()):.1f} / {float(torch.rad2deg(hl_err_max[sv]).max()):.1f} · "
          f"{int((torch.rad2deg(hl_err_max[sv]) > 15.0).sum())}") if hn > 0 else "—"
    rows.append(f"| {name} | {desc} | {n} | {nf} ({100.0 * nf / max(n, 1):.1f} %) | {f(vr):.3f} | {f(vx):+.4f} | {f(vy):+.4f} | {math.degrees(f(yrate)):+.2f} | "
                f"{dr_mean:.2f} / {dr_max:.2f} | {trms:.2f} / {tmax:.1f} | {wrms:.3f} | {hs} | {he} |")
    J["groups"][name] = {"cfg": d, "n": n, "falls": nf, "valid_rate": f(vr), "vx": f(vx), "vy": f(vy), "yaw_rate_dps": math.degrees(f(yrate)),
                         "yaw_drift_deg_mean": dr_mean, "yaw_drift_deg_max": dr_max, "tilt_rms": trms, "tilt_max": tmax, "w_rms": wrms,
                         "hl_ticks": hn, "hl_small_frac": (float(hl_small[sv].sum()) / hn if hn > 0 else None),
                         "hl_big_frac": (float(hl_big[sv].sum()) / hn if hn > 0 else None), "hl_mean_abs": (float(hl_abs[sv].sum()) / hn if hn > 0 else None),
                         "hl_err_max_deg": ([round(float(x), 2) for x in torch.rad2deg(hl_err_max[sv]).cpu()] if hn > 0 else None),
                         "t_fall": [round(float(x), 2) for x in t_fall[mg & ~torch.isnan(t_fall)].cpu()],
                         "s8_clamp_frac_per_joint": ((U._s8_clamp_n[sv].sum(0) / (n_tick[sv].sum() + 1e-9)).cpu().tolist() if getattr(U, '_s8_clamp_n', None) is not None and d['clamp'] == 's8' and bool(sv.any()) else None)}
wall = time.time() - wall0
head = [f"# eval_w3_parity {a.out_tag}", f"- ckpt {ck}", f"- task {a.task} seed {a.seed} envs {N} (그룹 {G} × ~{N // G}) · 서기 {a.stand_first}s → STEP, 워밍업 {a.warmup}s, 채점 {a.seconds}s · "
        f"고리 K {a.hl_kp} ±{a.hl_wzmax} · LPF β 0.27 · 관측 잡음·학습 DR 유지 · wall {wall:.0f}s", ""]
txt = "\n".join(head + rows + ["", "COMPLETE"])
os.makedirs(a.out_dir, exist_ok=True)
open(os.path.join(a.out_dir, f"w3par_{a.out_tag}.txt"), "w", encoding="utf-8").write(txt + "\n")
json.dump(J, open(os.path.join(a.out_dir, f"w3par_{a.out_tag}.json"), "w"), ensure_ascii=False, indent=1, default=str)
print(txt, flush=True)
env.close()
app.close()
