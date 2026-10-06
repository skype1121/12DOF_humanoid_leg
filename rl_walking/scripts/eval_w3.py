"""W3 Stage 0 자동 채점표(1006) — 제자리 좌우 교대 스텝 엔진 판정(reports/gen_w3/W3_DESIGN.md).

환경: 학습 태스크 그대로(관측 계약·학습 DR 유지 — CLAUDE.md 3), 외란·밀기 없음, 커리큘럼·명령 리샘플 끔,
      액션 LPF β 0.27 고정, 액션 지연 고정(--delay). 명령은 매 틱 --cmd 로 고정(서기 env 비율 0).
보조·진단 스위치는 env var 로만 전달(실제 반영은 환경 코드 몫): STEP_ASSIST_ALPHA(--assist_alpha, 기본 0),
      W3_REF_SCALE=0(--ref_off), W3_GAIT_ENABLE(--gait_enable). TURN_ALPHA 는 지정되지 않았으면 0(보조 0 평가).
발별 사건(왼·오른 각각, 접촉 = 발 링크 |F| > 5 N):
  이륙(접촉→비접촉) · 의미 있는 스윙(공중 ≥ 60 ms 그리고 발 높이 상승 ≥ 1.5 cm) · 착지(비접촉→접촉) ·
  하중 수용(착지 후 0.3 s 안 Fz > 0.5 mg 가 0.1 s 누적). 유효 스텝 = 넷 모두.
지표: 이륙 성공률(예정 스윙 반주기 중 그 발이 이륙한 비율; 위상 = env._gu 가 있으면 그것(φ<0.5 왼 스윙), 없으면 1.1 Hz 시간 기반),
  착지 성공률(이륙 대비), 하중 수용률(착지 대비), 교대율(연속 유효 스텝의 좌우 교대 비율), 유효 cadence[Hz](유효 스텝/env/s ÷ 2 = 보행 주기 Hz),
  수평 표류(시작 요 프레임, 명령 속도 대비 잔차 평균 속도·총 이동), 요 표류, 낙상률(종료 = 낙상), 장시간 붕괴(10 s 창별 유효 cadence,
  마지막 창 < 첫 창 × 50 % = 붕괴), 스텝 높이·보폭 분포, 정지 떨림(피치 RMS — gait_enable 0 일 때 판정, 1 이면 참고).
사용: isaaclab.sh -p rl_walking/scripts/eval_w3.py --headless --task <T> --load_run <run> --checkpoint model_N.pt --out_tag <tag> [--long]
출력: log_picture/w3eval_<tag>.txt(요약 표) + .json(수치)."""
import argparse, os, sys, math, json, time
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402
REPO = os.environ.get("W3EVAL_REPO", "/home/ryu/humanoid_leg_test1")      # 체크포인트(logs/)·출력(log_picture/) 위치
CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # rl_walking 코드는 이 스크립트가 있는 체크아웃에서
sys.path.insert(0, CODE)
p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W2qe-v0")
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--seconds", type=float, default=30.0, help="채점 구간 길이[s](워밍업 뒤)")
p.add_argument("--long", action="store_true", help="장시간 모드: 채점 구간 120 s")
p.add_argument("--warmup", type=float, default=2.0, help="채점 전 안정화[s](명령은 처음부터 적용)")
p.add_argument("--delay", type=int, default=1)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--assist_alpha", type=float, default=0.0, help="STEP_ASSIST_ALPHA 로 전달(기본 0 = 보조 없는 최종 평가)")
p.add_argument("--ref_off", action="store_true", help="진단: W3_REF_SCALE=0 세팅(참조 잔차 끄기 — W3 코드가 읽을 때만 반영)")
p.add_argument("--gait_enable", type=int, default=1, choices=(0, 1), help="W3_GAIT_ENABLE 로 전달(0 = STAND, 1 = STEP)")
p.add_argument("--ramp_off", action="store_true",
               help="1006 하네스 감사 D3: hang_support 롤 토크 램프 크기 0(난수 소비 유지 — 같은 seed 짝 유지). 기본은 학습과 같은 램프 켜짐")
p.add_argument("--stand_first", type=float, default=0.0,
               help="1006 진단: 처음 S 초는 STAND(ge 0) 후 STEP 전환(실물 순서 STAND→STEP). 워밍업 = S + 2 s 로 자동 연장")
p.add_argument("--stop_at", type=float, default=0.0,
               help="1006 closure 2 정지 품질: 시작 후 T 초에 ge 0(STAND) 요청 → 정지까지 시간·추가 이륙·표류·최종 정지·자세 측정(채점 지표는 그대로)")
p.add_argument("--cmd", default="0,0,0", help="'vx,vy,wz' 고정 명령")
p.add_argument("--heading_loop", type=float, default=0.0,
               help="1006 W3S3: > 0 이면 외부 헤딩 P 고리(배포·W3S3h 학습과 같은 식) wz = clip(wz_cmd + K·wrap(ψ* − ψ), ±--hl_wzmax), ψ* = 시작 요에서 wz_cmd 로 적분")
p.add_argument("--hl_wzmax", type=float, default=0.4)
p.add_argument("--turn_metrics", action="store_true", help="회전 지표 표 강제 출력(|wz| > 0 또는 --heading_loop 이면 자동)")
p.add_argument("--out_tag", default="w3")
cli_args.add_rsl_rl_args(p)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
if a.long:
    a.seconds = 120.0
if a.stand_first > 0:
    a.warmup = max(a.warmup, a.stand_first + 2.0)
# env var 는 환경 생성 전에(시작 이벤트가 읽는다)
os.environ["STEP_ASSIST_ALPHA"] = repr(float(a.assist_alpha))
os.environ["W3_GAIT_ENABLE"] = str(int(a.gait_enable))
if a.ref_off:
    os.environ["W3_REF_SCALE"] = "0"
os.environ.setdefault("TURN_ALPHA", "0")
os.environ["W3_HLOOP"] = "0"                    # 환경 내장 헤딩 고리(W3S3h) 끔 — 평가는 --heading_loop 로만(같은 식)
CMD = [float(x) for x in a.cmd.split(",")]
assert len(CMD) == 3, "--cmd 는 'vx,vy,wz'"
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
c.resampling_time_range = (1e6, 1e6); c.rel_standing_envs = 0.0
for _ev in ("disturbance", "push_robot"):
    if getattr(env_cfg.events, _ev, None) is not None:
        setattr(env_cfg.events, _ev, None)
act = env_cfg.actions.joint_pos
act.lpf_beta_range = (0.27, 0.27); act.min_delay_steps = act.max_delay_steps = a.delay
if hasattr(act, "delay_probs"):
    act.delay_probs = ()
if a.ramp_off and getattr(env_cfg.events, "hang_support_reset", None) is not None:
    _hp = env_cfg.events.hang_support_reset.params
    if _hp.get("ramp_rate_range") is not None:
        _hp["ramp_rate_range"] = (0.0, 0.0); _hp["ramp_cap_range"] = (0.0, 0.0)
T_END = a.warmup + a.seconds
env_cfg.episode_length_s = T_END + 2.0          # 시간 초과 종료가 채점 구간 안에 오지 않게
env = gym.make(a.task, cfg=env_cfg)
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
U = env.unwrapped; dev = U.device; N = U.num_envs; dt = U.step_dt
log_root = os.path.abspath(os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name))
ck = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ck); policy = runner.get_inference_policy(device=dev)

robot = U.scene["robot"]; sensor = U.scene.sensors["contact_forces"]
fid = mdp_hang._feet_body_ids(U); bid = mdp_hang._g21_bids(U)
cmd_term = U.command_manager.get_term("base_velocity")
MG = 9.81 * float(robot.data.default_mass[0].sum())
_scale = getattr(act, "scale", None)
ACT_SCALE = float(_scale) if isinstance(_scale, (int, float)) else None


HL = {"tgt": None, "wz": None}


def _wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def set_cmd():
    cmd_term.vel_command_b[:, 0] = CMD[0]; cmd_term.vel_command_b[:, 1] = CMD[1]; cmd_term.vel_command_b[:, 2] = CMD[2]
    if a.heading_loop > 0:                       # 외부 헤딩 P 고리(정책 밖) — rl_walking/w3.py w3_heading_loop 와 같은 식
        psi = yaw_of(robot.data.root_quat_w)
        HL["tgt"] = psi.clone() if HL["tgt"] is None else _wrap(HL["tgt"] + CMD[2] * dt)
        HL["wz"] = (CMD[2] + a.heading_loop * _wrap(HL["tgt"] - psi)).clamp(-a.hl_wzmax, a.hl_wzmax)
        cmd_term.vel_command_b[:, 2] = HL["wz"]
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _qmul(p_, q_):
    aw, ax, ay, az = p_.unbind(-1); bw, bx, by, bz = q_.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def rel_yaw(q_now, q_ref):
    """월드 상대 회전 R_now·R_refᵀ 의 요(링크 축 규약 무관) — rl_walking/w3.py _rel_yaw 와 같은 식."""
    return yaw_of(_qmul(q_now, q_ref * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q_ref.device)))


def phase_now():
    """(φ ∈ [0,1), 출처) — env._gu 우선, 없으면 1.1 Hz 에피소드 시간."""
    gu = getattr(U, "_gu", None)
    if gu is not None and gu.shape[0] == N:
        return gu.clone(), "env._gu"
    t = U.episode_length_buf.to(torch.float32) * dt
    return torch.remainder(1.1 * t, 1.0), "시간 1.1 Hz"


obs = venv.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
steps = int(round(T_END / dt))
fz = lambda *s: torch.zeros(*s, device=dev)
alive = torch.ones(N, dtype=torch.bool, device=dev); t_fall = torch.full((N,), float("nan"), device=dev)
# 발 상태(N,2) [L,R]
prev_c = None
st_xy0 = None; st_lifts = fz(N); st_last_lift = torch.full((N,), float("nan"), device=dev); st_qrun = fz(N); st_settle = torch.full((N,), float("nan"), device=dev)
st_xy_end = fz(N, 2); st_fm0 = None; st_fm_end = fz(N, 2); st_lat_lifts = fz(N); st_alive0 = None; st_com_off = fz(N, 2); st_tilt = fz(N); st_dq = fz(N); st_ge = torch.ones(N, device=dev); st_ge_off = torch.full((N,), float("nan"), device=dev)
in_air = torch.zeros(N, 2, dtype=torch.bool, device=dev); air_t = fz(N, 2); z0 = fz(N, 2); zmax = fz(N, 2); p0 = fz(N, 2, 2)
pend = torch.zeros(N, 2, dtype=torch.bool, device=dev); pend_t = fz(N, 2); pend_load = fz(N, 2); pend_swing = torch.zeros(N, 2, dtype=torch.bool, device=dev)
pend_rise = fz(N, 2); pend_len = fz(N, 2); pend_fwd = fz(N, 2); pend_lat = fz(N, 2); pend_w = fz(N, 2)
n_lift, n_swing, n_td, n_load, n_valid = fz(N, 2), fz(N, 2), fz(N, 2), fz(N, 2), fz(N, 2)
last_valid = torch.full((N,), -1, dtype=torch.long, device=dev); alt_pairs, alt_ok = fz(N), fz(N)
# 예정 스윙 반주기
prev_half = None; win_ok = torch.zeros(N, dtype=torch.bool, device=dev); win_lift = torch.zeros(N, 2, dtype=torch.bool, device=dev)
n_sched, n_sched_lift = fz(N, 2), fz(N, 2)
lift_phase = []   # 이륙 시 위상(그 발 스윙 시작 기준 주기 단위, −0.5~0.5)
# 장시간 창
NW = max(1, int(math.ceil(a.seconds / 10.0 - 1e-6)))
w_valid, w_time = np.zeros(NW), np.zeros(NW)
# 분포
rises, lens, fwds = [], [], []
lats, widths, foot_ids = [], [], []   # 1006 Stage 2: 착지 옆 변위·착지 순간 발 간격·발
roll_s, roll_n = fz(N), fz(N)
# 1006 Stage 3 회전 지표(프레임 독립: 발 링크 쿼터니언 월드 상대 회전의 요) — 착지 발 요 변위·지지발 피벗·몸통 비틀림·고리 명령
TURN_ON = a.turn_metrics or abs(CMD[2]) > 0 or a.heading_loop > 0
q_lift = robot.data.body_quat_w[:, bid, :].clone(); q_prevf = q_lift.clone(); pend_dpsi = fz(N, 2)
dpsis, dpsi_feet = [], []
piv_abs, piv_sgn, sw_abs, sw_sgn = fz(N), fz(N), fz(N), fz(N)
tw_s, tw_ss, hlwz_s, hlwz_ss = fz(N), fz(N), fz(N), fz(N)
# 자세
pit_s, pit_ss, pit_n = fz(N), fz(N), fz(N); wy_ss = fz(N)
pos0 = yaw0 = None
ref_ss, ref_n, act_ss, act_n = 0.0, 0, 0.0, 0
has_ref = False
phase_src = None
t0 = time.time()
for i in range(steps):
    t = (i + 1) * dt
    set_cmd()
    if a.stand_first > 0 and getattr(U, "_ge_cmd", None) is not None:
        U._ge_cmd[:] = bool(t > a.stand_first)
    if a.stop_at > 0 and getattr(U, "_ge_cmd", None) is not None and t > a.stop_at:
        U._ge_cmd[:] = False
        cmd_term.vel_command_b[:] = 0.0                                      # 정지 = 명령 0 + ge 0 (실물 stop 과 같은 계약)
    with torch.inference_mode():
        act_ = policy(obs)
    obs, _, dones, _ = venv.step(act_)
    if isinstance(obs, tuple):
        obs = obs[0]
    d = dones.bool() if torch.is_tensor(dones) else torch.as_tensor(dones, device=dev).bool()
    tm = getattr(U, "termination_manager", None)
    if tm is not None and hasattr(tm, "terminated"):
        d = d & tm.terminated.bool()          # 시간 초과가 아닌 종료만 낙상으로(에피소드 길이는 채점 구간보다 길게 잡음)
    newly = alive & d
    t_fall[newly] = t; alive &= ~d
    ev = t > a.warmup
    q = robot.data.root_quat_w; yaw = yaw_of(q)
    if ev and pos0 is None:
        pos0 = robot.data.root_pos_w[:, :2].clone(); yaw0 = yaw.clone(); yaw_acc = torch.zeros_like(yaw); yaw_prev = yaw.clone()
    cc = sensor.data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0
    fzv = sensor.data.net_forces_w[:, fid, 2]
    fpos = robot.data.body_pos_w[:, bid, :]
    phi, phase_src = phase_now()
    half = (phi >= 0.5).long()     # 0 = 왼 스윙 예정, 1 = 오른 스윙 예정
    if prev_c is None:
        prev_c = cc.clone(); prev_half = half.clone()
    m2 = (alive & torch.full((N,), bool(ev), device=dev)).unsqueeze(1).expand(N, 2)
    # ── 예정 반주기 창: 경계에서 직전 창 마감 ──
    bound = half != prev_half
    if ev:
        done_w = bound & win_ok & alive
        for k in (0, 1):
            mk = done_w & (prev_half == k)
            n_sched[:, k] += mk.float(); n_sched_lift[:, k] += (mk & win_lift[:, k]).float()
        win_ok = torch.where(bound, alive.clone(), win_ok & alive)
        win_lift = torch.where(bound.unsqueeze(1), torch.zeros_like(win_lift), win_lift)
    prev_half = half.clone()
    # ── 이륙 ──
    lifted = prev_c & ~cc & m2
    if ev:
        for k in (0, 1):
            win_lift[:, k] |= lifted[:, k] & (half == k)
        if lifted.any():
            for k in (0, 1):
                mk = lifted[:, k]
                if mk.any():
                    ph_rel = torch.remainder(phi[mk] - 0.5 * k + 0.5, 1.0) - 0.5
                    lift_phase.append(ph_rel.cpu())
    n_lift += lifted.float()
    in_air = in_air | lifted
    air_t = torch.where(lifted, torch.zeros_like(air_t), air_t)
    z0 = torch.where(lifted, fpos[..., 2], z0); zmax = torch.where(lifted, fpos[..., 2], zmax)
    p0 = torch.where(lifted.unsqueeze(-1), fpos[..., :2], p0)
    fq = robot.data.body_quat_w[:, bid, :]
    q_lift = torch.where(lifted.unsqueeze(-1), fq, q_lift)
    air_now = in_air & ~cc
    air_t = air_t + dt * air_now.float(); zmax = torch.where(air_now, torch.maximum(zmax, fpos[..., 2]), zmax)
    # ── 착지 ──
    landed = in_air & cc & m2
    in_air = in_air & ~cc
    if landed.any():
        rise = zmax - z0; dp = fpos[..., :2] - p0
        hx = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1).unsqueeze(1)
        sw = landed & (air_t >= 0.06) & (rise >= 0.015)
        n_td += landed.float(); n_swing += sw.float()
        pend = pend | landed; pend_t = torch.where(landed, torch.zeros_like(pend_t), pend_t)
        pend_load = torch.where(landed, torch.zeros_like(pend_load), pend_load)
        pend_swing = torch.where(landed, sw, pend_swing)
        pend_rise = torch.where(landed, rise, pend_rise); pend_len = torch.where(landed, dp.norm(dim=-1), pend_len)
        pend_fwd = torch.where(landed, (dp * hx).sum(-1), pend_fwd)
        hy = torch.stack([-torch.sin(yaw), torch.cos(yaw)], -1).unsqueeze(1)
        pend_lat = torch.where(landed, (dp * hy).sum(-1), pend_lat)
        _wd = ((fpos[:, 0, :2] - fpos[:, 1, :2]) * hy[:, 0]).sum(-1)          # 왼 − 오 (요 프레임 y, 양수 = 정상)
        pend_w = torch.where(landed, _wd.unsqueeze(1).expand(-1, 2), pend_w)
        pend_dpsi = torch.where(landed, rel_yaw(fq, q_lift), pend_dpsi)
    # ── 하중 수용 ──
    pend = pend & m2
    pend_load = pend_load + dt * (pend & (fzv > 0.5 * MG)).float()
    acc = pend & (pend_load >= 0.1 - 1e-6) & (pend_t <= 0.3 + 1e-6)
    pend_t = pend_t + dt * pend.float()
    pend = pend & ~acc & (pend_t <= 0.3 + 1e-6)
    n_load += acc.float()
    val = acc & pend_swing
    n_valid += val.float()
    if val.any():
        rises.append(pend_rise[val].cpu()); lens.append(pend_len[val].cpu()); fwds.append(pend_fwd[val].cpu())
        lats.append(pend_lat[val].cpu()); widths.append(pend_w[val].cpu())
        dpsis.append(pend_dpsi[val].cpu())
        sw_abs += (pend_dpsi.abs() * val.float()).sum(1); sw_sgn += (pend_dpsi * val.float()).sum(1)
        foot_ids.append(torch.arange(2, device=dev).unsqueeze(0).expand(N, 2)[val].cpu())
        for k in (0, 1):
            vk = val[:, k]
            has = vk & (last_valid >= 0)
            alt_pairs += has.float(); alt_ok += (has & (last_valid != k)).float()
            last_valid = torch.where(vk, torch.full_like(last_valid, k), last_valid)
        wi = min(NW - 1, int((t - a.warmup) / 10.0))
        w_valid[wi] += float(val.float().sum())
    if a.stop_at > 0:                                                      # 정지 품질 계측
        if st_xy0 is None and t > a.stop_at:
            st_xy0 = robot.data.root_pos_w[:, :2].clone()
            st_fm0 = robot.data.body_pos_w[:, bid, :2].mean(1).clone(); st_alive0 = alive.clone()
        if st_xy0 is not None:
            off_ = (prev_c & ~cc) & alive.unsqueeze(1)
            st_lifts += off_.float().sum(1)
            st_last_lift = torch.where(off_.any(1), torch.full_like(st_last_lift, t - a.stop_at), st_last_lift)
            wq = robot.data.root_ang_vel_b[:, :2].norm(dim=-1); vq = robot.data.root_lin_vel_w[:, :2].norm(dim=-1)
            qq = alive & cc.all(1) & (wq < 0.15) & (vq < 0.05)
            st_qrun = torch.where(qq, st_qrun + dt, torch.zeros_like(st_qrun))
            st_settle = torch.where(torch.isnan(st_settle) & (st_qrun >= 0.5), torch.full_like(st_settle, t - a.stop_at - 0.5), st_settle)
            st_xy_end = torch.where(alive.unsqueeze(1), robot.data.root_pos_w[:, :2], st_xy_end)
            _fm = robot.data.body_pos_w[:, bid, :2].mean(1)                      # 1006 정지 표류 분해: 발 중점(흔들림 위상 오프셋 제외)
            st_fm_end = torch.where(alive.unsqueeze(1), _fm, st_fm_end)
            st_com_off = torch.where(alive.unsqueeze(1), robot.data.root_pos_w[:, :2] - _fm, st_com_off)
            if getattr(U, "_ge", None) is not None:
                st_lat_lifts += (off_.float().sum(1) * (~U._ge.bool()).float())   # 래치 해제(STAND) 뒤 이륙
            pg_ = robot.data.projected_gravity_b; st_tilt = torch.rad2deg(torch.atan2(pg_[:, :2].norm(dim=-1), -pg_[:, 2]))
            st_dq = torch.rad2deg((robot.data.joint_pos - robot.data.default_joint_pos).abs()).mean(-1)
            st_ge = U._ge.clone() if getattr(U, "_ge", None) is not None else st_ge
            st_ge_off = torch.where(torch.isnan(st_ge_off) & ~st_ge.bool(), torch.full_like(st_ge_off, t - a.stop_at), st_ge_off)
    prev_cc_t = prev_c.clone()                    # 직전 틱 접촉(지지발 피벗 판정용) — prev_c 갱신 전에 보관
    prev_c = cc.clone()
    if ev:
        wi = min(NW - 1, int((t - a.warmup - 1e-9) / 10.0))
        w_time[wi] += float(alive.float().sum()) * dt
        pg = robot.data.projected_gravity_b
        pit = torch.rad2deg(torch.atan2(pg[:, 0], -pg[:, 2])); af = alive.float()
        pit_s += pit * af; pit_ss += pit * pit * af; pit_n += af
        rol = torch.rad2deg(torch.atan2(-pg[:, 1], -pg[:, 2])); roll_s += rol * af; roll_n += af
        if TURN_ON:
            stance_ = (cc & prev_cc_t).float() * af.unsqueeze(1)
            dq_ = rel_yaw(fq, q_prevf) * stance_
            piv_abs += dq_.abs().sum(1); piv_sgn += dq_.sum(1)
            _d = fpos[:, 0, :2] - fpos[:, 1, :2]
            tw = _wrap(yaw - (torch.atan2(_d[:, 1], _d[:, 0]) - 0.5 * math.pi))
            tw_s += tw * af; tw_ss += tw * tw * af
            if HL["wz"] is not None:
                hlwz_s += HL["wz"] * af; hlwz_ss += HL["wz"] * HL["wz"] * af
        wy_ss += robot.data.root_ang_vel_b[:, 1] ** 2 * af
        a_raw = act_.detach()
        am = alive.unsqueeze(1).float()
        act_ss += float((a_raw ** 2 * am).sum()); act_n += int(alive.sum()) * a_raw.shape[1]
        rl = getattr(U, "_w3_ref_last", None)
        if torch.is_tensor(rl) and rl.shape[0] == N:
            has_ref = True
            rr = rl.detach().reshape(N, -1).float()
            ref_ss += float((rr ** 2 * am).sum()); ref_n += int(alive.sum()) * rr.shape[1]
    if ev:
        yaw_end = yaw.clone(); pos_end = robot.data.root_pos_w[:, :2].clone()
        _dy = torch.atan2(torch.sin(yaw - yaw_prev), torch.cos(yaw - yaw_prev))   # 1006 감사 D4: 틱별 누적(±180° 감싸기 없음)
        yaw_acc = torch.where(alive, yaw_acc + _dy, yaw_acc); yaw_prev = yaw.clone()
    q_prevf = fq.clone()                          # 지지발 피벗용 직전 틱 발 자세(틱 끝에서 갱신)
print(f"[eval_w3] {steps} 틱 완료 · {time.time() - t0:.0f} s", flush=True)

# ───────────── 집계 ─────────────
al = alive; T = a.seconds
S = lambda x: float(x.sum())
lift_rate = S(n_sched_lift) / max(1.0, S(n_sched))
td_rate = S(n_td) / max(1.0, S(n_lift))
load_rate = S(n_load) / max(1.0, S(n_td))
swing_rate = S(n_swing) / max(1.0, S(n_td))
alt_rate = S(alt_ok) / max(1.0, S(alt_pairs))
alive_time = float(w_time.sum())
cad = S(n_valid) / max(1e-9, alive_time) / 2.0
fall_rate = float((~al).float().mean())
tf_ = t_fall.cpu() if torch.is_tensor(t_fall) else torch.as_tensor(t_fall)
_tfv = tf_[~al.cpu()].numpy(); _tfv = _tfv[_tfv == _tfv]
fall_t_hist = {f"{lo}-{hi}s": int(((_tfv > lo) & (_tfv <= hi)).sum()) for lo, hi in ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, 5), (5, 1e9))}
n_fall_warm = int(((~al.cpu()) & (tf_ <= a.warmup)).sum()); n_fall_score = int(((~al.cpu()) & (tf_ > a.warmup)).sum())
dpos = pos_end - pos0
c0, s0 = torch.cos(yaw0), torch.sin(yaw0)
dx = dpos[:, 0] * c0 + dpos[:, 1] * s0; dy = -dpos[:, 0] * s0 + dpos[:, 1] * c0
res_v = torch.stack([dx / T - CMD[0], dy / T - CMD[1]], -1).norm(dim=-1)
dyaw = torch.rad2deg(yaw_acc)                      # 감싸지 않은 누적 요(구버전은 ±180° 감싸기 → 장시간 과소추정)
mA = lambda x: float(x[al].mean()) if al.any() else float("nan")
drift_v = mA(res_v); disp = mA(dpos.norm(dim=-1)); fwd_v = mA(dx / T); lat_v = mA(dy / T)
yaw_drift = mA((dyaw - math.degrees(CMD[2] * T)).abs())
yaw_rate = mA(dyaw / T)
w_cad = [float(w_valid[k] / max(1e-9, w_time[k]) / 2.0) for k in range(NW)]
collapse = bool(NW >= 2 and w_cad[0] > 0 and w_cad[-1] < 0.5 * w_cad[0])
pm = pit_s / pit_n.clamp(min=1); pit_rms = (pit_ss / pit_n.clamp(min=1) - pm * pm).clamp(min=0).sqrt()
pitch_rms = mA(pit_rms); wy_rms = mA((wy_ss / pit_n.clamp(min=1)).sqrt())
cat = lambda L: torch.cat(L).numpy() if L else np.zeros(0)
R_, L_, F_ = cat(rises) * 100, cat(lens) * 100, cat(fwds) * 100
Y_, W_, FI_ = cat(lats) * 100, cat(widths) * 100, (torch.cat(foot_ids).numpy() if foot_ids else np.zeros(0))
_sg = np.sign(CMD[1]) if abs(CMD[1]) > 1e-6 else 0.0
lat_sign_rate = float((np.sign(Y_) == _sg).mean()) if (len(Y_) and _sg != 0) else float("nan")
width_viol = float((W_ < 23.8).mean()) if len(W_) else float("nan")
cross_rate = float((W_ < 0).mean()) if len(W_) else float("nan")
lr_ratio = float((FI_ == 0).sum() / max(1, (FI_ == 1).sum())) if len(FI_) else float("nan")
roll_mean = mA(roll_s / roll_n.clamp(min=1))


def turn_summary():
    """1006 Stage 3 회전 지표(별도 함수 — 기존 지표·판정 불변). 단위: 도, 도/s.
    - 착지 발 요 변위 Δψ(유효 스텝) 분포·명령 부호 일치율, 목표 Δψ* = wz/1.1 Hz
    - 요 이득 = 실제 평균 요 속도 / 명령 wz(감싸지 않은 누적)
    - 회전 출처: 스윙 착지 Δψ 합 vs 지지발 피벗(접촉 유지 중 발 요 변화) 합(부호 = 명령 방향), 피벗 몫 = 피벗 / (피벗 + 스윙)(부호 기여)
    - 몸통 비틀림 ψ − ψ_발연결선 평균·RMS, 헤딩 고리 명령 wz 평균·RMS(고리 켰을 때)"""
    D_ = (torch.cat(dpsis).numpy() if dpsis else np.zeros(0)) * 180.0 / math.pi
    sg = np.sign(CMD[2]) if abs(CMD[2]) > 1e-6 else 0.0
    sign_rate = float((np.sign(D_) == sg).mean()) if (len(D_) and sg != 0) else float("nan")
    at = alive_time if alive_time > 0 else 1e-9
    deg = 180.0 / math.pi
    sgn = sg if sg != 0 else 1.0
    piv_a = S(piv_abs[al]) * deg; sw_a = S(sw_abs[al]) * deg
    out = {"dpsi_deg_p10_50_90": pct(D_), "dpsi_target_deg": math.degrees(CMD[2] / 1.1), "dpsi_sign_rate": sign_rate,
           "yaw_gain": (yaw_rate / math.degrees(CMD[2])) if abs(CMD[2]) > 1e-6 else float("nan"),
           "swing_yaw_dps": sgn * S(sw_sgn[al]) * deg / at, "pivot_yaw_dps": sgn * S(piv_sgn[al]) * deg / at,
           "pivot_abs_dps": piv_a / at, "pivot_frac_abs": piv_a / max(piv_a + sw_a, 1e-9),
           "twist_mean_deg": mA(tw_s / pit_n.clamp(min=1)) * deg,
           "twist_rms_deg": mA((tw_ss / pit_n.clamp(min=1)).sqrt()) * deg,
           "hl_wz_mean": (mA(hlwz_s / pit_n.clamp(min=1)) if a.heading_loop > 0 else None),
           "hl_wz_rms": (mA((hlwz_ss / pit_n.clamp(min=1)).sqrt()) if a.heading_loop > 0 else None)}
    _ps, _ss = out["pivot_yaw_dps"], out["swing_yaw_dps"]
    out["pivot_share"] = _ps / (_ps + _ss) if abs(_ps + _ss) > 1e-9 else float("nan")     # 판정용: 명령 방향 부호 기여(|·| 는 지지발 떨림 지배)
    rows = ["", f"## 회전 지표(Stage 3 — 명령 wz {CMD[2]:+.2f}" + (f", 외부 헤딩 고리 K {a.heading_loop} ±{a.hl_wzmax}" if a.heading_loop > 0 else "") + ")",
            f"- 착지 발 요 변위 Δψ(유효 스텝, 10/50/90 %) " + "/".join(f"{x:+.1f}" for x in out["dpsi_deg_p10_50_90"])
            + f"° · 목표 {out['dpsi_target_deg']:+.1f}° · 명령 부호 일치 {100 * sign_rate:.1f} %",
            f"- 요 이득(실제/명령) {out['yaw_gain']:.2f} · 평균 요 속도 {yaw_rate:+.2f}°/s",
            f"- 회전 출처(명령 방향 +): 스윙 착지 {out['swing_yaw_dps']:+.2f}°/s · 지지발 피벗 {out['pivot_yaw_dps']:+.2f}°/s · 피벗 몫(부호) {out['pivot_share']:.2f} · |피벗| {out['pivot_abs_dps']:.2f}°/s(떨림 포함 참고)",
            f"- 몸통 비틀림(골반 − 발 연결선) 평균 {out['twist_mean_deg']:+.1f}° · RMS {out['twist_rms_deg']:.1f}°"]
    if a.heading_loop > 0:
        rows.append(f"- 헤딩 고리 명령 wz 평균 {out['hl_wz_mean']:+.3f} · RMS {out['hl_wz_rms']:.3f} rad/s(요 표류 {yaw_drift:.1f}° = 시작 요 대비)")
    return out, rows
pct = lambda x: [float(np.percentile(x, q)) for q in (10, 50, 90)] if len(x) else [float("nan")] * 3
LP = cat(lift_phase)
act_rms = math.sqrt(act_ss / max(1, act_n))
act_rms_deg = math.degrees(act_rms * ACT_SCALE) if ACT_SCALE is not None else None
ref_rms_deg = math.degrees(math.sqrt(ref_ss / max(1, ref_n))) if has_ref else None
per_foot = {nm: {"이륙": S(n_lift[:, k]), "예정창": S(n_sched[:, k]), "예정창_이륙": S(n_sched_lift[:, k]), "착지": S(n_td[:, k]),
                 "의미스윙": S(n_swing[:, k]), "하중수용": S(n_load[:, k]), "유효": S(n_valid[:, k])} for k, nm in ((0, "왼"), (1, "오른"))}
crit = [
    ("교대 > 90 %", alt_rate > 0.90 and S(alt_pairs) > 0, f"{100 * alt_rate:.1f} %"),
    ("이륙 > 90 %", lift_rate > 0.90, f"{100 * lift_rate:.1f} %"),
    ("착지 > 90 %", td_rate > 0.90 and S(n_lift) > 0, f"{100 * td_rate:.1f} %"),
    ("하중 ≥ 85 %", load_rate >= 0.85 and S(n_td) > 0, f"{100 * load_rate:.1f} %"),
    ("채점 구간 낙상 < 2 % (시작 과도 제외 — 1006 판정 정의)", n_fall_score / max(1, N - n_fall_warm) < 0.02, f"{n_fall_score}/{N - n_fall_warm}"),
    ("표류 ≤ 0.03 m/s", drift_v <= 0.03, f"{drift_v:.3f} m/s"),
    ("장시간 붕괴 없음", (not collapse) and w_cad[0] > 0, "창 " + "/".join(f"{x:.2f}" for x in w_cad) + " Hz"),
]
if a.gait_enable == 0:
    crit = [("낙상 < 5 %", fall_rate < 0.05, f"{100 * fall_rate:.1f} %"), ("표류 ≤ 0.03 m/s", drift_v <= 0.03, f"{drift_v:.3f} m/s"),
            ("스텝 0(유효 ≤ 0.05 /env/s)", S(n_valid) / max(1e-9, alive_time) <= 0.05, f"{S(n_valid) / max(1e-9, alive_time):.3f} /env/s")]
all_pass = all(ok for _, ok, _ in crit)

L = [f"# W3 Stage 0 채점표 · {a.out_tag}",
     f"- 태스크 {a.task} · 정책 {ck}",
     f"- env {N} · seed {a.seed} · 지연 {a.delay}틱 · LPF β 0.27 · 워밍업 {a.warmup}s + 채점 {T:.0f}s{' (장시간)' if a.long else ''} · 외란 없음 · 학습 DR 유지",
     f"- 명령 (vx,vy,wz) = ({CMD[0]:+.2f},{CMD[1]:+.2f},{CMD[2]:+.2f}) · gait_enable {a.gait_enable} · STEP_ASSIST_ALPHA {a.assist_alpha}"
     f" · W3_REF_SCALE {'0(--ref_off)' if a.ref_off else '미설정'} · TURN_ALPHA {os.environ.get('TURN_ALPHA')} · 위상 출처 {phase_src}"
     + (f" · env._assist_alpha {getattr(U, '_assist_alpha'):.3g}" if hasattr(U, "_assist_alpha") else " · 보조 훅 없음"),
     "",
     "| 지표 | 값 |", "|---|---|",
     f"| 이륙 성공률(예정 스윙 반주기 중 이륙) | {100 * lift_rate:.1f} % ({S(n_sched_lift):.0f}/{S(n_sched):.0f}) |",
     f"| 착지 성공률(이륙 대비) | {100 * td_rate:.1f} % ({S(n_td):.0f}/{S(n_lift):.0f}) |",
     f"| 의미 있는 스윙(착지 대비, ≥60 ms·≥1.5 cm) | {100 * swing_rate:.1f} % ({S(n_swing):.0f}) |",
     f"| 하중 수용률(착지 대비, 0.3 s 안 Fz>0.5 mg 0.1 s) | {100 * load_rate:.1f} % ({S(n_load):.0f}) |",
     f"| 유효 스텝 | {S(n_valid):.0f} (왼 {S(n_valid[:, 0]):.0f} / 오른 {S(n_valid[:, 1]):.0f}) · {S(n_valid) / max(1e-9, alive_time):.2f} /env/s |",
     f"| 교대율(연속 유효 스텝 좌우 교대) | {100 * alt_rate:.1f} % ({S(alt_ok):.0f}/{S(alt_pairs):.0f}) |",
     f"| 유효 cadence(보행 주기) | {cad:.3f} Hz |",
     f"| 수평 표류(명령 대비 잔차 속도) | {drift_v:.3f} m/s · 총 이동 {disp:.2f} m · 실제 앞 {fwd_v:+.3f} / 옆 {lat_v:+.3f} m/s |",
     f"| 요 표류(명령 대비 |Δ요|) | {yaw_drift:.1f}° · 평균 요 속도 {yaw_rate:+.2f}°/s |",
     f"| 낙상률 | {100 * fall_rate:.1f} % ({int((~al).sum())}/{N}) — 워밍업(시작 과도) {n_fall_warm} · 채점 구간 {n_fall_score} · 시각 " + " ".join(f"{k}:{v}" for k, v in fall_t_hist.items() if v) + " |",
     f"| 10 s 창별 유효 cadence | " + " → ".join(f"{x:.2f}" for x in w_cad) + f" Hz · {'붕괴' if collapse else '붕괴 없음'} |",
     f"| 스텝 높이 cm (10/50/90 %) | " + "/".join(f"{x:.1f}" for x in pct(R_)) + " |",
     f"| 옆 변위 cm (착지, 10/50/90 %) | " + "/".join(f"{x:+.1f}" for x in pct(Y_)) + f" · 명령 방향 부호 일치 {100 * lat_sign_rate:.1f} % |",
     f"| 착지 발 간격 cm (10/50/90 %) | " + "/".join(f"{x:.1f}" for x in pct(W_)) + f" · 23.8 미만 {100 * width_viol:.2f} % · 교차 {100 * cross_rate:.2f} % · 좌/우 유효 스텝 비 {lr_ratio:.2f} · 롤 평균 {roll_mean:+.2f}° |",
     f"| 보폭 cm (수평 변위 10/50/90 %) | " + "/".join(f"{x:.1f}" for x in pct(L_)) + " · 앞축 " + "/".join(f"{x:+.1f}" for x in pct(F_)) + " |",
     f"| 이륙 위상(예정 스윙 시작 대비 주기, 10/50/90 %) | " + ("/".join(f"{x:+.2f}" for x in pct(LP)) if len(LP) else "—") + " |",
     f"| 피치 RMS(정지 떨림{'' if a.gait_enable == 0 else ', 참고'}) | {pitch_rms:.2f}° · 피치각속도 RMS {wy_rms:.3f} rad/s |",
     f"| 정책 행동 RMS | {act_rms:.3f}" + (f" (×scale {ACT_SCALE} = {act_rms_deg:.2f}°)" if act_rms_deg is not None else "") + " |",
     f"| 참조 잔차 RMS(env._w3_ref_last) | " + (f"{ref_rms_deg:.2f}°" if ref_rms_deg is not None else "속성 없음 — 생략") + " |",
     "",
     "| 발 | 이륙 | 예정창 | 예정창 이륙 | 착지 | 의미 스윙 | 하중 수용 | 유효 |", "|---|---|---|---|---|---|---|---|"]
for nm, v in per_foot.items():
    L.append(f"| {nm} | {v['이륙']:.0f} | {v['예정창']:.0f} | {v['예정창_이륙']:.0f} | {v['착지']:.0f} | {v['의미스윙']:.0f} | {v['하중수용']:.0f} | {v['유효']:.0f} |")
if a.stop_at > 0 and st_xy0 is not None:
    mS = lambda x: float(torch.nanmedian(x[al]).item()) if al.any() else float("nan")
    q90 = lambda x: float(torch.quantile(x[al & ~torch.isnan(x)], 0.9).item()) if (al & ~torch.isnan(x)).any() else float("nan")
    settled = al & ~torch.isnan(st_settle)
    stop_drift = (st_xy_end - st_xy0).norm(dim=-1) * 100
    fm_drift = (st_fm_end - st_fm0).norm(dim=-1) * 100
    n_alive_at_stop = int(st_alive0.sum()); n_fall_after_stop = int((st_alive0 & ~al).sum())
    L += ["", f"## 정지 품질(ge 0 요청 @ {a.stop_at:.1f} s, 생존 env 기준)",
          f"- ge 래치 해제(반주기 경계) 중앙 {mS(st_ge_off):.2f} s",
          f"- 요청 뒤 이륙 수 중앙 {mS(st_lifts):.0f} · 90 % {q90(st_lifts):.0f} · 마지막 이륙 중앙 {mS(st_last_lift):.2f} s",
          f"- 정착(양발·|ω|<0.15·|v|<0.05 가 0.5 s) 도달 {100 * float(settled.sum()) / max(1, int(al.sum())):.1f} % · 시간 중앙 {mS(st_settle):.2f} s · 90 % {q90(st_settle):.2f} s",
          f"- 요청 뒤 표류 중앙 {mS(stop_drift):.1f} cm · 90 % {q90(stop_drift):.1f} cm",
          f"- 끝 자세: 기울기 중앙 {mS(st_tilt):.2f}° · 기본자세 대비 관절 평균 편차 중앙 {mS(st_dq):.2f}°",
          f"- (주 지표) 발 중점 이동 중앙 {mS(fm_drift):.1f} cm · 90 % {q90(fm_drift):.1f} cm (골반 표류는 요청 순간 좌우 흔들림 오프셋 포함 — 참고용)",
          f"- 래치 해제(STAND) 뒤 이륙 중앙 {mS(st_lat_lifts):.0f} · 0 회 env {100 * float(((st_lat_lifts == 0) & al).sum()) / max(1, int(al.sum())):.1f} %",
          f"- 정지 요청 뒤 낙상 {n_fall_after_stop}/{n_alive_at_stop} · 끝 골반−발 중점 오프셋 중앙 |{mS(st_com_off.norm(dim=-1) * 100):.1f}| cm"]
    J_stop = {"ge_off_med": mS(st_ge_off), "lifts_med": mS(st_lifts), "settle_rate": float(settled.sum()) / max(1, int(al.sum())), "settle_med": mS(st_settle),
              "settle_p90": q90(st_settle), "drift_med_cm": mS(stop_drift), "tilt_med": mS(st_tilt), "dq_med": mS(st_dq), "foot_mid_drift_med_cm": mS(fm_drift), "foot_mid_drift_p90_cm": q90(fm_drift),
              "lifts_after_latch_med": mS(st_lat_lifts), "falls_after_stop": n_fall_after_stop, "alive_at_stop": n_alive_at_stop}
else:
    J_stop = None
J_turn = None
if TURN_ON:
    J_turn, _rows = turn_summary(); L += _rows
L += ["", f"## Stage 0 사전 기준 판정({'STAND, gait_enable 0' if a.gait_enable == 0 else 'STEP'}): {'PASS' if all_pass else 'FAIL'}"]
L += [f"- {'PASS' if ok else 'FAIL'} · {nm} · {val}" for nm, ok, val in crit]
L += ["", "주: 접촉 = 발 링크 |F| > 5 N. 유효 cadence = 유효 스텝/(생존 env·s)/2. 표류 = 시작 요 프레임 변위/채점 시간 − 명령(vx,vy). "
      "낙상 env 는 낙상 전까지의 사건만 집계. 이 판정은 시뮬 채점이며 실물 성공을 뜻하지 않는다."]
txt = "\n".join(L); print(txt, flush=True)
out_dir = os.path.join(REPO, "log_picture"); os.makedirs(out_dir, exist_ok=True)
open(os.path.join(out_dir, f"w3eval_{a.out_tag}.txt"), "w").write(txt + "\n")
J = {"ramp_off": a.ramp_off, "turn": J_turn, "heading_loop": a.heading_loop, "hl_wzmax": a.hl_wzmax, "lateral": {"lat_sign_rate": lat_sign_rate, "width_viol": width_viol, "cross_rate": cross_rate, "lr_ratio": lr_ratio, "roll_mean": roll_mean,
               "lat_p50_cm": (float(np.median(Y_)) if len(Y_) else None), "width_p50_cm": (float(np.median(W_)) if len(W_) else None)}, "stop": J_stop, "fall_t_hist": fall_t_hist, "fall_t": [round(float(x), 2) for x in _tfv], "fall_warm": n_fall_warm, "fall_score": n_fall_score, "stand_first": a.stand_first, "task": a.task, "checkpoint": ck, "num_envs": N, "seed": a.seed, "delay": a.delay, "warmup_s": a.warmup, "seconds": T, "long": a.long,
     "cmd": CMD, "gait_enable": a.gait_enable, "assist_alpha": a.assist_alpha, "ref_off": a.ref_off, "turn_alpha": os.environ.get("TURN_ALPHA"),
     "phase_source": phase_src, "steps_done": steps,
     "lift_rate": lift_rate, "touchdown_rate": td_rate, "swing_rate": swing_rate, "load_rate": load_rate, "alternation_rate": alt_rate,
     "valid_steps": S(n_valid), "valid_per_env_s": S(n_valid) / max(1e-9, alive_time), "valid_cadence_hz": cad,
     "drift_res_mps": drift_v, "displacement_m": disp, "fwd_mps": fwd_v, "lat_mps": lat_v, "yaw_drift_deg": yaw_drift, "yaw_rate_dps": yaw_rate,
     "fall_rate": fall_rate, "window_cadence_hz": w_cad, "collapse": collapse,
     "step_height_cm_p10_50_90": pct(R_), "step_len_cm_p10_50_90": pct(L_), "step_fwd_cm_p10_50_90": pct(F_), "lift_phase_p10_50_90": pct(LP),
     "pitch_rms_deg": pitch_rms, "pitch_rate_rms": wy_rms, "action_rms": act_rms, "action_rms_deg": act_rms_deg, "ref_rms_deg": ref_rms_deg,
     "per_foot": per_foot, "criteria": [{"name": nm, "pass": bool(ok), "value": val} for nm, ok, val in crit], "pass": all_pass}
json.dump(J, open(os.path.join(out_dir, f"w3eval_{a.out_tag}.json"), "w"), ensure_ascii=False, indent=1)
print(f"[eval_w3] 저장: log_picture/w3eval_{a.out_tag}.txt/.json", flush=True)
sys.stdout.flush(); os._exit(0)   # app.close() 가 종료에서 멈추는 경우가 있어(1006 스모크: 300 s 타임아웃) 산출물 저장 뒤 즉시 종료
