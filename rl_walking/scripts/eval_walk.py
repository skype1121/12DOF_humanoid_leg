"""W1(1003) 보행 평가 — 명령 속도별 추종·낙상·보행 위생 지표 + 걷기→정지 정착 + 무명령 서기 흔들림.

환경: W1 태스크 그대로(관측 계약 = G21a: 직전 액션 EMA·관절속도 EMA·접촉 실물 계약, 학습 DR 유지 — CLAUDE.md 3),
      외란 없음, 커리큘럼·명령 리샘플 끔, 액션 LPF β 0.27 고정, 액션 지연 고정(--delay, 실측 d≈1).
절차(env 마다): 0–T_STAND s 서기(명령 0) → T_STAND–T_STOP s 걷기(vx = 그룹 속도) → T_STOP–T_END s 정지(명령 0).
그룹: 속도 0(서기 대조) + --speeds (기본 0.05,0.10,0.15,0.20). env 를 그룹에 균등 배분.
지표(그룹별, 걷기 창은 명령 시작 2 s 뒤부터):
  낙상률 · |v_fwd − vx| 평균(요 프레임 전진속도) · 요 표류[°] · 옆 표류[cm] · 착지 수 · 왼 착지 비율 · 착지당 앞축 보폭[cm] ·
  공중 시간 평균[s] · 양발 지지 구간 평균[s] · 접촉 중 발 미끄럼 속도[m/s] · 교대 착지 비율 · 정지 뒤 정착 시간[s](|v|<0.05 & |ω|<0.3 가 0.5 s) ·
  서기 흔들림: 피치각속도 2–4.5 Hz 대역 RMS(정지 창·서기 그룹) · 정지 뒤 피치 p2p.
사용: isaaclab.sh -p rl_walking/scripts/eval_walk.py --headless --task Biped12-HangStand-W1-v0 --load_run <run> --checkpoint model_N.pt --out_tag w1_N
      (G21a 대조: --load_run 2026-10-02_21-23-08_g21a --checkpoint model_11300.pt — 같은 W1 환경·같은 명령)"""
import argparse, os, sys, math
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402
REPO = "/home/ryu/humanoid_leg_test1"
sys.path.insert(0, REPO)
p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W1-v0")
p.add_argument("--num_envs", type=int, default=250)
p.add_argument("--speeds", default="0.05,0.10,0.15,0.20")
p.add_argument("--delay", type=int, default=1)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--t_stand", type=float, default=3.0)
p.add_argument("--t_walk", type=float, default=8.0)
p.add_argument("--t_after", type=float, default=4.0)
p.add_argument("--torsion_r", type=float, default=None, help="1005 진단: 로봇 충돌 형상 비틀림 마찰 패치 반경[m](PhysX torsional_patch_radius) — ⚠1005 WTOR: 이 자산·GPU 경로에서 결과 비트 동일 = 효과 없음 확인, 쓰지 말 것"); p.add_argument("--hipf_deg", type=float, default=None, help="1005: 공유 힙 앞뒤 영점 편향 고정[deg](default_pose_bias.shared_hip_f_rad lo=hi) — 힙 편향 강건성"); p.add_argument("--keep_dist", action="store_true", help="1005: 학습 외란(disturbance·push_robot) 유지 — 걷기 중 외란 강건성"); p.add_argument("--wpush", default="", help="1005: 걷기 중 정의 밀기 'dir:F:dur[:ramp]' (dir f|b|l, 힘 N, 지속 s, ramp=램프 상승 s) — 걷기 시작 3 s 뒤 몸통 프레임으로 모든 env"); p.add_argument("--rec_monitor", action="store_true", help="1005 통합 진단: 감지기만 계산하고 전환하지 않음 — 걷기 중 |ω_xy|·기울기 편차 최대값 분포 기록"); p.add_argument("--rec_walk_f", type=float, default=0.0, help="1005: 걷기 중 감지 = 보행 주기 저역(τ 0.45 s) 기울기 편차 > 값[°] (0 = 원 문턱 사용). 걷기 1.5 s 뒤부터"); p.add_argument("--rec_walk_on", default="3.0,0.5", help="걷기 중 감지기 켜기 문턱 'tilt,w'"); p.add_argument("--rec_policy", default="", help="1005 통합: 'run:model_N.pt' 회복 정책 — 걷기 시작부터 감지기(RV4 상대 3°/0.5, 끄기 1.5°/0.3·calm 1 s·dwell 0.4·접촉 any) + ki cue(ω_y>0 → 1 s vx 0.03) + 조기 복귀(0.6,6,0.4,0.2). 회복 뒤 = 서기(G21a) 래치"); p.add_argument("--stand_policy", default="", help="1005 전환 평가: 'run:model_N.pt' 서기 정책(G21a) — 걷기 명령 전·정지 뒤 stand_after_s 이후에 사용(같은 관측 계약)"); p.add_argument("--stand_after_s", type=float, default=0.5); p.add_argument("--out_tag", default="walk"); p.add_argument("--friction_scale", type=float, default=1.0, help="진단(T4): 발 마찰 범위 배율(학습 설정 불변)"); p.add_argument("--heading_hold", type=float, default=0.0, help="배포 브리지 HeadingHold 모사: wz = clip(kp·(요0 − 요), ±0.6), kp 값(0=끔)"); p.add_argument("--wz_groups", default="", help="진단: 걷기 중 상수 요 명령을 env 묶음별로(예 -0.3,0,0.3) — 회전 명령 추종 확인")
cli_args.add_rsl_rl_args(p)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import numpy as np, torch, gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import rl_walking  # noqa: F401,E402
from rl_walking import mdp_hang  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

env_cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, a)
env_cfg.scene.num_envs = a.num_envs; env_cfg.seed = a.seed
env_cfg.curriculum = None
c = env_cfg.commands.base_velocity
c.resampling_time_range = (1e6, 1e6); c.rel_standing_envs = 1.0
if not a.keep_dist:
    env_cfg.events.disturbance = None
if a.friction_scale != 1.0 and getattr(env_cfg.events, "physics_material", None) is not None:
    pm = env_cfg.events.physics_material.params
    pm["static_friction_range"] = tuple(x * a.friction_scale for x in pm["static_friction_range"])
    pm["dynamic_friction_range"] = tuple(x * a.friction_scale for x in pm["dynamic_friction_range"])
if a.hipf_deg is not None:
    _r = math.radians(a.hipf_deg); env_cfg.events.default_pose_bias.params['shared_hip_f_rad'] = (_r, _r); print(f'[hipf] 공유 힙 편향 {a.hipf_deg}°', flush=True)
if a.torsion_r is not None:
    import isaaclab.sim as _su
    env_cfg.scene.robot.spawn.collision_props = _su.CollisionPropertiesCfg(torsional_patch_radius=a.torsion_r, min_torsional_patch_radius=a.torsion_r)
    print(f"[torsion] 비틀림 패치 반경 {a.torsion_r} m", flush=True)
if getattr(env_cfg.events, "push_robot", None) is not None and not a.keep_dist:
    env_cfg.events.push_robot = None
act = env_cfg.actions.joint_pos
act.lpf_beta_range = (0.27, 0.27); act.min_delay_steps = act.max_delay_steps = a.delay
if hasattr(act, "delay_probs"):
    act.delay_probs = ()
T_END = a.t_stand + a.t_walk + a.t_after
env_cfg.episode_length_s = T_END + 1.0
env = gym.make(a.task, cfg=env_cfg)
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
U = env.unwrapped; dev = U.device; N = U.num_envs; dt = U.step_dt
log_root = os.path.abspath(os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name))
ck = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ck); policy = runner.get_inference_policy(device=dev)
policy_st = None
policy_rec = None
if a.rec_policy:
    _r3, _c3 = a.rec_policy.split(':')
    ck_rec = get_checkpoint_path(log_root, _r3, _c3)
    runner_rec = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device); runner_rec.load(ck_rec); policy_rec = runner_rec.get_inference_policy(device=dev)
    print(f'[rec_policy] {ck_rec}', flush=True)
if a.stand_policy:
    _r2, _c2 = a.stand_policy.split(':')
    ck_st = get_checkpoint_path(log_root, _r2, _c2)
    runner_st = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device); runner_st.load(ck_st); policy_st = runner_st.get_inference_policy(device=dev)
    print(f'[stand_policy] {ck_st} · 정지 뒤 {a.stand_after_s}s 에 인계', flush=True)
speeds = [0.0] + [float(s) for s in a.speeds.split(",")]
grp = torch.arange(N, device=dev) % len(speeds)
vgrp = torch.tensor(speeds, device=dev)[grp]
robot = U.scene["robot"]; sensor = U.scene.sensors["contact_forces"]
fid = mdp_hang._feet_body_ids(U); bid = mdp_hang._g21_bids(U)
cmd_term = U.command_manager.get_term("base_velocity")
_bn_ = list(robot.data.body_names); _BASE_ID = _bn_.index("base") if "base" in _bn_ else 0

hh_ref = None
WZ = None
if a.wz_groups:
    _wz = [float(x) for x in a.wz_groups.split(',')]
    WZ = torch.tensor(_wz, device=dev)[(torch.arange(N, device=dev) // len(speeds)) % len(_wz)]
def set_cmd(vx):
    global hh_ref
    cmd_term.vel_command_b[:, 0] = vx; cmd_term.vel_command_b[:, 1] = 0.0; cmd_term.vel_command_b[:, 2] = 0.0
    if a.heading_hold > 0.0:
        yw = yaw_of(robot.data.root_quat_w)
        if hh_ref is None:
            hh_ref = yw.clone()
        err = torch.atan2(torch.sin(hh_ref - yw), torch.cos(hh_ref - yw))
        cmd_term.vel_command_b[:, 2] = (a.heading_hold * err).clamp(-0.6, 0.6)
    if WZ is not None:
        cmd_term.vel_command_b[:, 2] = torch.where(vx > 0, WZ, torch.zeros_like(WZ))
    cmd_term.is_standing_env[:] = False
    cmd_term.is_heading_env[:] = False

def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

obs = venv.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
steps = int(T_END / dt)
alive = torch.ones(N, dtype=torch.bool, device=dev); t_fall = torch.full((N,), float("nan"), device=dev)
z = lambda: torch.zeros(N, device=dev)
err_s, err_n = z(), z(); n_td = torch.zeros(N, 2, device=dev); step_len_s, step_n = z(), z()
air_s, air_n = z(), z(); ds_s, ds_n = z(), z(); slip_s, slip_n = z(), z(); alt_n = z()
had_lift = torch.zeros(N, 2, dtype=torch.bool, device=dev); prev_c = torch.ones(N, 2, device=dev); air_t = torch.zeros(N, 2, device=dev); ds_t = z(); lift_x = torch.zeros(N, 2, device=dev)
last_td = torch.full((N,), -1.0, device=dev)
yaw0 = None; pos0 = None; yaw_w = z(); lat_w = z()
settle_t = torch.full((N,), float("nan"), device=dev); quiet_run = z()
wy_hist = []; pitch_after = []
t_ws, t_we = a.t_stand, a.t_stand + a.t_walk
_bz = lambda: torch.zeros(N, dtype=torch.bool, device=dev)
r_mode, r_latch, r_cue, r_cdone = _bz(), _bz(), _bz(), _bz()
r_calm, r_ton, r_early, r_cue_t, r_n_on, r_n_on_walk_nodist = z(), z(), z(), z(), z(), z()
r_wmax, r_tmax = z(), z(); r_fwmax, r_ftmax = z(), z(); r_frp = torch.zeros(N, 2, device=dev); r_fw = z(); r_fref = torch.zeros(N, 2, device=dev); r_ref = torch.zeros(N, 2, device=dev); r_first_on = torch.full((N,), float('nan'), device=dev)
for i in range(steps):
    t = i * dt
    walking = (t >= t_ws) & (t < t_we)
    if policy_rec is not None:
        _walk_env = (~r_mode & ~r_latch) & torch.full((N,), bool(walking), device=dev)
        set_cmd(torch.where(_walk_env, vgrp, torch.zeros_like(vgrp)))
        if walking or t >= t_we:
            pass
    else:
        set_cmd(vgrp if walking else torch.zeros(N, device=dev))
    if a.wpush:
        _pd, _pF, _pdur, *_pr = a.wpush.split(":"); _pF = float(_pF); _pdur = float(_pdur); _pr = float(_pr[0]) if _pr else 0.0
        _t0p = t_ws + 3.0; _el = t - _t0p
        _mag = 0.0 if (_el < 0 or _el >= _pdur) else (_pF * min(1.0, _el / _pr) if _pr > 0 else _pF)
        _fv = torch.zeros(N, 1, 3, device=dev); _ax = {"f": (0, 1.0), "b": (0, -1.0), "l": (1, 1.0)}[_pd]
        _fv[:, 0, _ax[0]] = _ax[1] * _mag
        robot.set_external_force_and_torque(_fv, torch.zeros_like(_fv), body_ids=[_BASE_ID])
    with torch.inference_mode():
        _use_st = policy_st is not None and ((t < t_ws) or (t >= t_we + a.stand_after_s))
        if policy_rec is None:
            act_ = policy_st(obs) if _use_st else policy(obs)
        else:
            # ── 1005 통합 감지기(eval_ground_recovery RV4 + ki 와 같은 정의) ──
            _pg = robot.data.projected_gravity_b
            _rp = torch.rad2deg(torch.stack([torch.atan2(_pg[:, 1], -_pg[:, 2]), torch.atan2(_pg[:, 0], -_pg[:, 2])], -1))
            _w2 = robot.data.root_ang_vel_b[:, :2].norm(dim=1)
            _cf2 = sensor.data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0
            if t >= t_ws:
                if abs(t - t_ws) < dt / 2 or (t - t_ws) < 1.5:
                    r_ref = _rp.clone()
                _upd = (~r_mode) & (_w2 < 0.1) & _cf2[:, 0] & _cf2[:, 1]
                r_ref = torch.where(_upd.unsqueeze(1), r_ref + (_rp - r_ref) * (dt / 10.0), r_ref).clamp(-8.0, 8.0)
                _tl = (_rp - r_ref).norm(dim=-1)
                _won = [float(x) for x in a.rec_walk_on.split(',')]
                _on_tilt = torch.where(torch.full((N,), bool(walking), device=dev) & ~r_latch, torch.full_like(_tl, _won[0]), torch.full_like(_tl, 3.0))
                _on_w = torch.where(torch.full((N,), bool(walking), device=dev) & ~r_latch, torch.full_like(_tl, _won[1]), torch.full_like(_tl, 0.5))
                # 1005 걷기 감지 후보: 보행 주기(1/1.1 Hz≈0.9 s) 저역 — 롤·피치 EMA τ 0.45 s, |ω_xy| EMA τ 0.45 s; 기준 = 걷기 1.5 s 시점 필터값
                _al = dt / 0.45
                r_frp = r_frp + (_rp - r_frp) * _al; r_fw = r_fw + (_w2 - r_fw) * _al
                if abs(t - (t_ws + 1.5)) < dt / 2:
                    r_fref = r_frp.clone()
                if walking and t >= t_ws + 1.5:
                    _ftl = (r_frp - r_fref).norm(dim=-1)
                    r_fwmax = torch.where(alive & ~r_mode & ~r_latch, torch.maximum(r_fwmax, r_fw), r_fwmax); r_ftmax = torch.where(alive & ~r_mode & ~r_latch, torch.maximum(r_ftmax, _ftl), r_ftmax)
                    r_wmax = torch.where(alive & ~r_mode & ~r_latch, torch.maximum(r_wmax, _w2), r_wmax); r_tmax = torch.where(alive & ~r_mode & ~r_latch, torch.maximum(r_tmax, _tl), r_tmax)
                _on = (~r_mode) & ((_tl > _on_tilt) | (_w2 > _on_w))
                if a.rec_walk_f > 0.0:
                    _wk = torch.full((N,), bool(walking), device=dev) & ~r_latch
                    _ftl_now = (r_frp - r_fref).norm(dim=-1) if t >= t_ws + 1.5 else torch.zeros_like(_tl)
                    _on = torch.where(_wk, (~r_mode) & (_ftl_now > a.rec_walk_f), _on)
                if a.rec_monitor:
                    _on = torch.zeros_like(_on)
                _quiet = (_tl < 1.5) & (_w2 < 0.3) & (_cf2[:, 0] | _cf2[:, 1])
                r_calm = torch.where(r_mode & _quiet, r_calm + dt, torch.zeros_like(r_calm))
                r_ton = torch.where(r_mode, r_ton + dt, torch.zeros_like(r_ton))
                _off = r_mode & (r_calm >= 1.0) & (r_ton >= 0.4)
                _ec = r_mode & (r_ton >= 0.6) & (_tl < 6.0) & (_w2 < 0.4) & _cf2[:, 0] & _cf2[:, 1]
                r_early = torch.where(_ec, r_early + dt, torch.zeros_like(r_early))
                _off = _off | (r_mode & (r_early >= 0.2))
                _new = _on & alive
                r_n_on += _new.float()
                if a.wpush == "" and walking:
                    r_n_on_walk_nodist += _new.float()
                r_first_on = torch.where(_new & torch.isnan(r_first_on), torch.full_like(r_first_on, t - t_ws), r_first_on)
                _fwd = robot.data.root_ang_vel_b[:, 1] > 0.0
                _cs_ = _new & _fwd
                r_cue = (r_cue | _cs_) & ~_off; r_cue_t = torch.where(_cs_, torch.zeros_like(r_cue_t), r_cue_t + dt)
                r_cue = r_cue & (r_cue_t < 1.0 - 1e-6)
                r_latch = r_latch | _off
                r_mode = (r_mode | _new) & ~_off
            _a_w = policy(obs)
            _a_s = policy_st(obs) if policy_st is not None else _a_w
            _obs_r = obs.clone(); _po = _obs_r["policy"].clone()
            _vc = _po[:, 30:45:3]
            _po[:, 30:45:3] = torch.where((r_cue & r_mode).unsqueeze(1), torch.full_like(_vc, 0.03), torch.zeros_like(_vc))
            _obs_r["policy"] = _po
            _a_r = policy_rec(_obs_r)
            _stand_now = torch.full((N,), bool(_use_st), device=dev) | r_latch
            act_ = torch.where(r_mode.unsqueeze(1), _a_r, torch.where(_stand_now.unsqueeze(1), _a_s, _a_w))
    obs, _, dones, _ = venv.step(act_)
    if isinstance(obs, tuple):
        obs = obs[0]
    d = dones.bool() if torch.is_tensor(dones) else torch.as_tensor(dones, device=dev).bool()
    newly = alive & d
    t_fall[newly] = t; alive &= ~d
    q = robot.data.root_quat_w; yaw = yaw_of(q)
    hx = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
    vw = robot.data.root_lin_vel_w[:, :2]; v_fwd = (vw * hx).sum(-1)
    w = robot.data.root_ang_vel_b[:, :2].norm(dim=1)
    cc = (sensor.data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0).float()
    fxy = robot.data.body_pos_w[:, bid, :2]; fx = (fxy * hx.unsqueeze(1)).sum(-1)
    fv = robot.data.body_lin_vel_w[:, bid, :2].norm(dim=-1)
    if abs(t - t_ws) < dt / 2:
        yaw0 = yaw.clone(); pos0 = robot.data.root_pos_w[:, :2].clone(); h0 = hx.clone()
    win = walking & (t >= t_ws + 2.0)
    if win:
        m = alive & (vgrp > 0)
        err_s += torch.where(m, (v_fwd - vgrp).abs(), torch.zeros_like(v_fwd)); err_n += m.float()
        vf_s = (vf_s if 'vf_s' in dir() else torch.zeros_like(v_fwd)) + torch.where(m, v_fwd, torch.zeros_like(v_fwd))
    if walking:
        m = alive
        lifted = (prev_c > 0.5) & (cc < 0.5); landed = (prev_c < 0.5) & (cc > 0.5)
        lift_x = torch.where(lifted, fx, lift_x); had_lift |= lifted
        landed = landed & had_lift                      # 걷기 창 안에서 든 발의 착지만(창 전 이륙 제외)
        had_lift &= ~landed
        n_td += (landed & m.unsqueeze(1)).float()
        step_len_s += ((fx - lift_x) * (landed & m.unsqueeze(1)).float()).sum(1); step_n += (landed & m.unsqueeze(1)).float().sum(1)
        ended_air = landed & m.unsqueeze(1)
        air_s += (air_t * ended_air.float()).sum(1); air_n += ended_air.float().sum(1)
        foot = torch.where(landed[:, 0] & ~landed[:, 1], 0.0, torch.where(landed[:, 1] & ~landed[:, 0], 1.0, -1.0))
        has = (foot >= 0) & m
        alt_n += (has & (last_td >= 0) & (foot != last_td)).float(); last_td = torch.where(has, foot, last_td)
        both = cc.min(1).values > 0.5
        end_ds = (~both) & (ds_t > 0) & m
        ds_s += torch.where(end_ds, ds_t, torch.zeros_like(ds_t)); ds_n += end_ds.float()
        slip_s += ((fv * cc).sum(1) * m.float()); slip_n += (cc.sum(1) * m.float())
        air_t = torch.where(cc < 0.5, air_t + dt, torch.zeros_like(air_t)); ds_t = torch.where(both, ds_t + dt, torch.zeros_like(ds_t))
    prev_c = cc
    if abs(t - (t_we - dt)) < dt / 2 and yaw0 is not None:
        yaw_w = torch.rad2deg(torch.atan2(torch.sin(yaw - yaw0), torch.cos(yaw - yaw0)))
        dp = robot.data.root_pos_w[:, :2] - pos0; lat_w = (dp * torch.stack([-h0[:, 1], h0[:, 0]], -1)).sum(-1)
    if t >= t_we:
        ok = alive & (vw.norm(dim=1) < 0.05) & (w < 0.3)
        quiet_run = torch.where(ok, quiet_run + dt, torch.zeros_like(quiet_run))
        hit = torch.isnan(settle_t) & (quiet_run >= 0.5)
        settle_t[hit] = max(t + dt - 0.5 - t_we, 0.0)
        pg = robot.data.projected_gravity_b
        pitch_after.append(torch.rad2deg(torch.atan2(pg[:, 0], -pg[:, 2])).cpu())
    if t >= 1.0:
        wy_hist.append((robot.data.root_ang_vel_b[:, 1] * alive.float()).cpu())
WY = torch.stack(wy_hist).numpy()                  # (T, N)
from scipy.signal import butter, filtfilt  # noqa: E402
b_, a_ = butter(2, [2.0 / (0.5 / dt), 4.5 / (0.5 / dt)], btype="band")
band = filtfilt(b_, a_, WY, axis=0)
t_axis = 1.0 + np.arange(WY.shape[0]) * dt
PA = torch.stack(pitch_after).numpy() if pitch_after else None
L = [f"W1 보행 평가 · {a.task} · {ck}" + (f" · HeadingHold kp {a.heading_hold}" if a.heading_hold > 0 else "") + (f" · 마찰 ×{a.friction_scale}" if a.friction_scale != 1.0 else "") + (f" · 비틀림 r {a.torsion_r} m" if a.torsion_r is not None else "") + (f" · 힙 편향 {a.hipf_deg}°" if a.hipf_deg is not None else "") + (f" · 걷기 밀기 {a.wpush} (걷기 3 s 뒤)" if a.wpush else "") + (f" · 회복 정책 {a.rec_policy}(통합 감지기)" if a.rec_policy else "") + (f" · 서기 정책 {a.stand_policy} (정지 뒤 {a.stand_after_s}s 인계)" if a.stand_policy else ""), f"지연 {a.delay}틱 · LPF β 0.27 · seed {a.seed} · env {N} · 서기 {a.t_stand}s → 걷기 {a.t_walk}s → 정지 {a.t_after}s" + (" · 학습 외란 유지" if a.keep_dist else " · 외란 없음") + " · 학습 DR 유지",
     "| vx 명령 | n | 낙상 % | 추종 오차 m/s | 요 표류 ° | 옆 표류 cm | 착지 수/env | 왼 착지 % | 보폭 cm | 공중 s | 양발지지 s | 미끄럼 m/s | 교대 % | 정지 정착 s (중앙) | 정지 창 3Hz RMS | 정지 뒤 피치 p2p ° |",
     "|" + "---|" * 16]
gn = grp.cpu().numpy(); al = alive.cpu().numpy()
for k, v in enumerate(speeds):
    m = gn == k; mt = torch.as_tensor(m, device=dev)
    nn = int(m.sum()); fall = 100.0 * float((~alive[mt]).float().mean())
    f = lambda s_, n_: float(s_[mt].sum() / n_[mt].sum().clamp(min=1))
    td = n_td[mt]; tdn = float(td.sum())
    st = settle_t[mt]; st = st[~torch.isnan(st)]
    stop = (t_axis >= t_we) & (t_axis < T_END)
    rms = float(np.sqrt((band[np.ix_(stop, m & al)] ** 2).mean())) if (m & al).any() else float("nan")
    p2p = float(np.ptp(PA[:, m & al], axis=0).mean()) if PA is not None and (m & al).any() else float("nan")
    walk = v > 0
    L.append(f"| {v:.2f} | {nn} | {fall:.1f} | " + (f"{f(err_s, err_n):.3f}" if walk else "—") +
             f" | {float(yaw_w[mt][alive[mt]].abs().mean()) if alive[mt].any() else float('nan'):.1f} | {100*float(lat_w[mt][alive[mt]].abs().mean()) if alive[mt].any() else float('nan'):.1f}"
             f" | {tdn / max(nn, 1):.1f} | " + (f"{100 * float(td[:, 0].sum()) / max(tdn, 1):.0f} | {100 * f(step_len_s, step_n):.1f}" if tdn >= 10 else "— | —") + f" | {f(air_s, air_n):.2f} | {f(ds_s, ds_n):.2f}"
             f" | {f(slip_s, slip_n):.3f} | {100 * float(alt_n[mt].sum()) / max(tdn, 1):.0f} | " + (f"{float(st.median()):.2f} ({len(st)}/{int(alive[mt].sum())})" if len(st) else "—") +
             f" | {rms:.4f} | {p2p:.2f} |")
_tf = t_fall.cpu().numpy()
L.append("낙상 시점(1005): 그룹별 [서기 중 / 걷기 시작 0–1.5 s / 정상 걷기 / 정지 뒤] 건수 · 걷기 시작 기준 낙상 시각 목록")
for k, v in enumerate(speeds):
    m = gn == k; tf = _tf[m]; tf = tf[~np.isnan(tf)]
    ph = [int(((tf < t_ws)).sum()), int(((tf >= t_ws) & (tf < t_ws + 1.5)).sum()), int(((tf >= t_ws + 1.5) & (tf < t_we)).sum()), int((tf >= t_we).sum())]
    L.append(f"- vx {v:.2f}: {ph} · " + ", ".join(f"{x - t_ws:+.1f}" for x in sorted(tf)))
L.append("평균 전진속도(걷기 2 s 뒤, 생존 env-틱 평균, 1005): " + " · ".join(f"명령 {v:.2f} → {float(vf_s[torch.as_tensor(gn == k, device=dev)].sum() / err_n[torch.as_tensor(gn == k, device=dev)].sum().clamp(min=1)):.3f}" for k, v in enumerate(speeds) if v > 0) if "vf_s" in dir() else "평균 전진속도: —")
if policy_rec is not None:
    _fo = r_first_on.cpu().numpy()
    for k, v in enumerate(speeds):
        m_ = gn == k
        _wm = r_wmax[torch.as_tensor(m_, device=dev)].cpu().numpy(); _tm = r_tmax[torch.as_tensor(m_, device=dev)].cpu().numpy()
        _fwm = r_fwmax[torch.as_tensor(m_, device=dev)].cpu().numpy(); _ftm = r_ftmax[torch.as_tensor(m_, device=dev)].cpu().numpy()
        _fl = (~alive[torch.as_tensor(m_, device=dev)]).cpu().numpy()
        L.append(f"- [통합 감시·저역] vx {v:.2f}: 최대 저역 |ω| 중앙/90/99 % {np.percentile(_fwm,50):.2f}/{np.percentile(_fwm,90):.2f}/{np.percentile(_fwm,99):.2f} · 최대 저역 기울기 편차 {np.percentile(_ftm,50):.1f}/{np.percentile(_ftm,90):.1f}/{np.percentile(_ftm,99):.1f}° · 낙상 env 저역 편차 중앙 {float(np.median(_ftm[_fl])) if _fl.any() else float('nan'):.1f}° / 생존 {float(np.median(_ftm[~_fl])) if (~_fl).any() else float('nan'):.1f}°")
        L.append(f"- [통합 감시] vx {v:.2f}: 걷기(1.5 s 뒤) 최대 |ω_xy| 중앙/90/99 % {np.percentile(_wm,50):.2f}/{np.percentile(_wm,90):.2f}/{np.percentile(_wm,99):.2f} rad/s · 최대 기울기 편차 {np.percentile(_tm,50):.1f}/{np.percentile(_tm,90):.1f}/{np.percentile(_tm,99):.1f}°")
        L.append(f"- [통합] vx {v:.2f}: 회복 켜짐 env {int((~np.isnan(_fo[m_])).sum())}/{int(m_.sum())} · 무외란 걷기 중 켜짐 합 {int(r_n_on_walk_nodist[torch.as_tensor(m_, device=dev)].sum())} · 서기 래치 {int(r_latch[torch.as_tensor(m_, device=dev)].sum())} · 첫 켜짐(걷기 기준) 중앙 {float(np.nanmedian(_fo[m_])) if (~np.isnan(_fo[m_])).any() else float('nan'):.2f}s")
L.append("착지 총 10 회 미만 그룹은 왼 착지 %·보폭을 — 로 표시(표본 과소).")
L.append("추종 오차 = 걷기 시작 2 s 뒤부터 생존 env 의 |요 프레임 전진속도 − 명령|. 서기 그룹(0.00)은 '걷기' 창에도 명령 0 — 무명령 서기 대조.")
if WZ is not None:
    L.append("| 요 명령 rad/s | n(걷기·생존) | 실제 요 변화 ° (부호, 평균) | 기대 ° (명령×걷기시간) | 추종비 |")
    L.append("|---|---|---|---|---|")
    for wv in _wz:
        mw = (WZ == wv) & (vgrp > 0) & alive
        mean = float(yaw_w[mw].mean()) if mw.any() else float("nan")
        exp = math.degrees(wv * a.t_walk)
        L.append(f"| {wv:+.2f} | {int(mw.sum())} | {mean:+.1f} | {exp:+.1f} | " + (f"{mean / exp:.2f}" if abs(exp) > 1e-6 else "—") + " |")
txt = "\n".join(L); print(txt, flush=True)
os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True)
open(os.path.join(REPO, "log_picture", f"walk_eval_{a.out_tag}.txt"), "w").write(txt + "\n")
app.close()
