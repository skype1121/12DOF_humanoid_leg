"""W3S3d 배포 parity 기록기(1007 parity 감사) — 시뮬 롤아웃의 매 틱 관측 원천·정책 관측 320·원시 행동을 npz 로 남긴다.

목적: rl_walking/deploy/selftest_w3_rollout_parity.py 가 같은 원천을 배포 러너(PolicyRunnerStage4 + W3 클록 + 헤딩 고리)에
열린 고리로 넣어 관측 320·행동이 시뮬과 같은지 본다(계약 검증 — 폐루프 성능 증명 아님).

환경 = 학습 태스크(Biped12-HangStand-W3S3d-v0) + eval_w3 와 같은 평가 조건(커리큘럼·리샘플 끔, LPF β 0.27·지연 고정, 보조 0),
단 **관측 잡음 끔**(enable_corruption False, gyro_ema·joint_vel_ema 내장 잡음 0): 시뮬은 잡음을 필터 **뒤**(ObsTerm) 또는 EMA 입력에
항별로 따로 뽑아 넣어 같은 원천에서 비트 재현이 불가능하다(잡음 구조 차이는 감사 문서에 분포 항목으로 따로 적는다).
지연·편향 DR(자이로 0–3·IMU 0–3 틱 지연, IMU 장착 편향, 기본자세 편향, 접촉 계약 v2)은 **켠 채** 기록한다 — 그 출력이 '센서 원천'이다.

명령열(t = 시작 기준 s, dt 0.02): 0–3 서기 → 3–9 제자리 → 9–15 앞 0.06 → 15–21 옆 +0.035 → 21–27 회전 +0.15 →
27–37 앞 0.06 + 헤딩 고리(배포 W3HeadingLoop 와 같은 식·같은 틱 요: 관측 계산 직전, 물리 뒤 요) → 37–39 정지(want 0·명령 0) →
39–42 옆 −0.035(오른발 리드 규칙) → 42–45 정지.

사용(GPU 1 회): isaaclab.sh -p rl_walking/scripts/w3_parity_record.py --headless --num_envs 3 --delay 3
출력: log_picture/w3parity_rec_<tag>.npz + .txt(요약). 판정은 산출물로(CLAUDE.md 5) — .txt 의 완주 틱·낙상·SHA 확인.
"""
import argparse
import hashlib
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
p.add_argument("--num_envs", type=int, default=3)
p.add_argument("--delay", type=int, default=3)
p.add_argument("--seed", type=int, default=11)
p.add_argument("--hl_kp", type=float, default=0.5)
p.add_argument("--hl_wzmax", type=float, default=0.4)
p.add_argument("--out_tag", default="w3s3d_15800")
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
app = AppLauncher(a).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import rl_walking  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

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
act.min_delay_steps = act.max_delay_steps = a.delay
if hasattr(act, "delay_probs"):
    act.delay_probs = ()
pol = env_cfg.observations.policy
pol.enable_corruption = False
pol.gyro_ema.params["noise"] = 0.0
pol.joint_vel_ema.params["noise"] = 0.0
T_END = 45.0
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
om = U.observation_manager
names = list(om.active_terms["policy"])
dims = [int(np.prod(d)) for d in om.group_obs_term_dim["policy"]]
assert names == ["base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions", "gait_clock",
                 "feet_contact", "gyro_ema", "joint_vel_ema"], names
OFF = {}
_o = 0
for nm, d in zip(names, dims):
    OFF[nm] = (_o, d)
    _o += d
assert _o == 320, _o


def newest(obs, nm):
    o, d = OFF[nm]
    D = d // 5
    return obs[:, o + 4 * D: o + 5 * D]


# ---- 명령열 ----
SEG = [(0.0, 3.0, False, (0.0, 0.0, 0.0), False), (3.0, 9.0, True, (0.0, 0.0, 0.0), False), (9.0, 15.0, True, (0.06, 0.0, 0.0), False),
       (15.0, 21.0, True, (0.0, 0.035, 0.0), False), (21.0, 27.0, True, (0.0, 0.0, 0.15), False), (27.0, 37.0, True, (0.06, 0.0, 0.0), True),
       (37.0, 39.0, False, (0.0, 0.0, 0.0), False), (39.0, 42.0, True, (0.0, -0.035, 0.0), False), (42.0, 45.0, False, (0.0, 0.0, 0.0), False)]


def seg_at(t):
    for t0, t1, w, cm, hl in SEG:
        if t0 <= t < t1:
            return w, cm, hl
    return False, (0.0, 0.0, 0.0), False


# ---- 배포와 같은 헤딩 고리(같은 틱 요: 관측 계산 직전) — W3HeadingLoop 와 같은 float64 식 ----
HL = {"on": False, "wz0": 0.0, "tgt": [None] * N, "wz": [None] * N}


def _wrap(x):
    return math.atan2(math.sin(x), math.cos(x))


_orig_compute = om.compute


def _compute_hook(*args, **kw):
    if HL["on"]:
        q = robot.data.root_quat_w.detach().cpu().numpy().astype(np.float32)
        for e in range(N):
            w_, x_, y_, z_ = (float(v) for v in q[e])
            psi = math.atan2(2.0 * (w_ * z_ + x_ * y_), 1.0 - 2.0 * (y_ * y_ + z_ * z_))
            tg = HL["tgt"][e]
            tg = psi if tg is None else _wrap(tg + HL["wz0"] * dt)
            HL["tgt"][e] = tg
            wz = max(-a.hl_wzmax, min(a.hl_wzmax, HL["wz0"] + a.hl_kp * _wrap(tg - psi)))
            HL["wz"][e] = wz
            cmd_term.vel_command_b[e, 2] = wz
    else:
        HL["tgt"] = [None] * N
        HL["wz"] = [None] * N
    return _orig_compute(*args, **kw)


om.compute = _compute_hook

steps = int(round(T_END / dt))
rec = {k: [] for k in ("obs", "act", "gyro", "grav", "cmd", "qpos_rel", "qvel", "contact", "want", "ge", "gu", "ge_obs", "elb",
                       "quat", "grav_true", "gyro_true", "q_true", "lpf_state", "proc", "term", "base_cmd", "hl_on", "fz", "ftilt")}
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
fid_ids, _ = robot.find_bodies(["left_ankle_r_joint", "right_ankle_r_joint"], preserve_order=True)
sensor = U.scene.sensors["contact_forces"]
sfid, _ = sensor.find_bodies(["left_ankle_r_joint", "right_ankle_r_joint"], preserve_order=True)


def snap(obs, want_now, base_cmd, hl_on, terminated):
    o = obs["policy"] if hasattr(obs, "keys") else obs
    rec["obs"].append(o.detach().cpu().numpy().astype(np.float32))
    rec["gyro"].append(newest(o, "base_ang_vel").cpu().numpy())
    rec["grav"].append(newest(o, "projected_gravity").cpu().numpy())
    rec["cmd"].append(newest(o, "velocity_commands").cpu().numpy())
    rec["qpos_rel"].append(newest(o, "joint_pos").cpu().numpy())
    rec["contact"].append(newest(o, "feet_contact").cpu().numpy())
    ev = getattr(U, "_enc_vel", None)
    rec["qvel"].append(ev.detach().cpu().numpy().astype(np.float32) if ev is not None else np.zeros((N, 12), np.float32))
    rec["want"].append(np.full(N, bool(want_now)))
    rec["ge"].append(U._ge.cpu().numpy().copy())
    gu = getattr(U, "_gu", None)
    rec["gu"].append(gu.cpu().numpy().copy() if gu is not None else np.zeros(N, np.float32))
    rec["ge_obs"].append(U._ge_obs.cpu().numpy().copy())
    rec["elb"].append(U.episode_length_buf.cpu().numpy().copy())
    rec["quat"].append(robot.data.root_quat_w.cpu().numpy().astype(np.float32))
    rec["grav_true"].append(robot.data.projected_gravity_b.cpu().numpy().astype(np.float32))
    rec["gyro_true"].append(robot.data.root_ang_vel_b.cpu().numpy().astype(np.float32))
    rec["q_true"].append(robot.data.joint_pos.cpu().numpy().astype(np.float32))
    rec["lpf_state"].append(term._lpf_state.cpu().numpy().astype(np.float32))
    rec["proc"].append(term.processed_actions.cpu().numpy().astype(np.float32))
    rec["term"].append(terminated.cpu().numpy().copy())
    rec["base_cmd"].append(np.tile(np.asarray(base_cmd, np.float32), (N, 1)))
    rec["hl_on"].append(np.full(N, bool(hl_on)))
    F = sensor.data.net_forces_w[:, sfid, :].norm(dim=-1)
    rec["fz"].append(F.cpu().numpy().astype(np.float32))
    qf = robot.data.body_quat_w[:, fid_ids, :]
    g = quat_apply_inverse(qf.reshape(-1, 4), torch.tensor([0.0, 0.0, -1.0], device=dev).expand(N * 2, 3)).reshape(N, 2, 3)
    rec["ftilt"].append(torch.acos((-g[..., 1]).clamp(-1.0, 1.0)).cpu().numpy().astype(np.float32))


obs = venv.get_observations()
want0, cmd0, hl0 = seg_at(0.0)
snap(obs, U._ge_cmd[0].item(), cmd0, False, torch.zeros(N, dtype=torch.bool, device=dev))
# 리셋 관측(틱 0)의 want 는 리셋 리샘플 값(W3_GAIT_ENABLE=0 → False). 이후 틱 n 의 want = 그 관측을 만든 step 직전에 넣은 값.
t_wall = time.time()
alive = torch.ones(N, dtype=torch.bool, device=dev)
for i in range(steps):
    t_next = (i + 1) * dt                       # 이 step 이 만드는 관측의 시각
    w, cm, hl = seg_at(t_next - 1e-9)
    U._ge_cmd[:] = bool(w)
    cmd_term.vel_command_b[:, 0] = cm[0]
    cmd_term.vel_command_b[:, 1] = cm[1]
    cmd_term.vel_command_b[:, 2] = cm[2]
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False
    HL["on"] = bool(hl and w)
    HL["wz0"] = float(cm[2])
    o = obs["policy"] if hasattr(obs, "keys") else obs
    with torch.inference_mode():
        act_ = policy(obs)
    rec["act"].append(act_.detach().cpu().numpy().astype(np.float32))
    obs, _, dones, _ = venv.step(act_)
    d = dones.bool()
    tm = getattr(U, "termination_manager", None)
    if tm is not None:
        d = d & tm.terminated.bool()
    alive &= ~d
    snap(obs, w, cm, HL["on"], d)
with torch.inference_mode():
    rec["act"].append(policy(obs).detach().cpu().numpy().astype(np.float32))
wall = time.time() - t_wall

out = {k: np.stack(v, 0) for k, v in rec.items()}           # (T+1, N, ...)
meta = {"task": a.task, "ckpt": ck, "seed": a.seed, "delay": a.delay, "num_envs": N, "dt": dt, "steps": steps, "T_END": T_END,
        "joint_names": list(robot.joint_names), "default_joint_pos": robot.data.default_joint_pos.cpu().numpy().tolist(),
        "delay_env": term._delay.cpu().numpy().tolist() if hasattr(term, "_delay") else None,
        "lpf_beta_env": term._beta.cpu().numpy().tolist() if hasattr(term, "_beta") else None,
        "gyro_lag_env": U._gyro_lag_st["lag"].cpu().numpy().tolist() if getattr(U, "_gyro_lag_st", None) else None,
        "imu_lag_env": U._imu_lag.cpu().numpy().tolist() if hasattr(U, "_imu_lag") else None,
        "w3_alpha": float(getattr(U, "_w3_alpha", float("nan"))), "assist_env": os.environ.get("STEP_ASSIST_ALPHA"),
        "assist_last_max": float(U._w3_assist_last.abs().max()) if getattr(U, "_w3_assist_last", None) is not None else None,
        "ref_last_max": float(U._w3_ref_last.abs().max()) if getattr(U, "_w3_ref_last", None) is not None else None,
        "segments": SEG, "hl_kp": a.hl_kp, "hl_wzmax": a.hl_wzmax, "obs_terms": names, "obs_dims": dims,
        "alive_end": alive.cpu().numpy().tolist(), "wall_s": wall,
        "lead_cfg": dict(env_cfg.events.w3_gait.params)}
os.makedirs(a.out_dir, exist_ok=True)
fn = os.path.join(a.out_dir, f"w3parity_rec_{a.out_tag}.npz")
np.savez_compressed(fn, meta=json.dumps(meta, default=str), **out)
sha = hashlib.sha256(open(fn, "rb").read()).hexdigest()[:16]
term_any = out["term"].any(0)
lines = [f"# W3 parity 기록 {a.out_tag}", f"- ckpt {ck}", f"- task {a.task} seed {a.seed} delay {a.delay} envs {N} steps {steps} (+리셋 관측 1) wall {wall:.1f}s",
         f"- env 별 지연 {meta['delay_env']} · LPF β {meta['lpf_beta_env']} · 자이로 지연 {meta['gyro_lag_env']} · IMU 지연 {meta['imu_lag_env']}",
         f"- 보조 α env {meta['w3_alpha']} (STEP_ASSIST_ALPHA={meta['assist_env']}) · assist_last max {meta['assist_last_max']} · ref_last max {meta['ref_last_max']}",
         f"- 낙상(종료) env: {np.nonzero(term_any)[0].tolist()} · 생존 끝 {meta['alive_end']}",
         f"- 헤딩 고리 켜짐 틱 {int(out['hl_on'][:, 0].sum())} · ge 1 틱(env0) {int(out['ge'][:, 0].sum())}",
         f"- npz {fn} sha256[:16] {sha} 크기 {os.path.getsize(fn) / 1e6:.1f} MB",
         "COMPLETE"]
open(os.path.join(a.out_dir, f"w3parity_rec_{a.out_tag}.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
env.close()
app.close()
