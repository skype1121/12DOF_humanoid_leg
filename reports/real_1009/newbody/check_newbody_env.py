"""1009 새 몸 학습 환경 점검(학습 전 검증 a·b·c) — 학습 태스크를 학습 때와 같은 설정으로 만들고 기존 체크포인트를 굴려 본다.
  (a) 고정 관절 변위·한계·네 링크 질량/질량 중심/관성·전체 질량을 실행 중 읽음
  (b) 이벤트 목록(모드별·함수·인자)과 관측 항 목록을 파일로 남김 — 원 태스크 실행과 diff
  (c) 체크포인트를 N 스텝 굴려 스텝 평균 보상·종료 사유별 건수·보상 항별 평균
사용: RL_WALKING_EXTRA_TASKS=rl_walking.tasks_extra_1009 isaaclab.sh -p -u check_newbody_env.py --headless --task <태스크> --num_envs 1024 --seed 1 \
      --load_run <run> --checkpoint model_N.pt --steps 1500 --tag <이름>
판정은 out/check_<tag>.txt(끝줄 'DONE')로만. 학습 태스크는 관측 잡음·DR 이 켜진 상태다(평가기와 조건이 다름 — 방향만 본다)."""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True); p.add_argument("--num_envs", type=int, default=1024); p.add_argument("--seed", type=int, default=1)
p.add_argument("--load_run", required=True); p.add_argument("--checkpoint", required=True)
p.add_argument("--steps", type=int, default=1500); p.add_argument("--tag", required=True)
p.add_argument("--drop_event", default="", help="1010 대조: 이 이름의 이벤트 항을 끄고(None) 환경을 만든다(쉼표 구분). 학습 설정 파일은 안 바뀜")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args); simulation_app = app.app

import importlib  # noqa: E402
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402

R = "/home/ryu/humanoid_leg_test1"; sys.path.insert(0, R)
import rl_walking  # noqa: F401,E402
for _m in [x for x in os.environ.get("RL_WALKING_EXTRA_TASKS", "").split(",") if x.strip()]:
    importlib.import_module(_m.strip())

OUT = os.path.join(R, "reports/real_1009/newbody/out", f"check_{args.tag}.txt"); L = []


def P(s=""):
    print(s, flush=True); L.append(str(s))


agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"); env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
for _de in [x for x in args.drop_event.split(",") if x.strip()]:
    assert getattr(env_cfg.events, _de.strip(), None) is not None, f"이벤트 {_de} 없음"
    setattr(env_cfg.events, _de.strip(), None); print(f"[drop_event] {_de.strip()} 끔", flush=True)
env_cfg.scene.num_envs = args.num_envs; env_cfg.seed = args.seed; agent_cfg.seed = args.seed
env = gym.make(args.task, cfg=env_cfg); env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
u = env.unwrapped; robot = u.scene["robot"]; dev = u.device; N = args.num_envs
P(f"# drop_event = {args.drop_event or '(없음)'}")
P(f"# check_newbody_env · task {args.task} · envs {N} · seed {args.seed} · {args.load_run}/{args.checkpoint} · steps {args.steps}")

# ── (b) 이벤트·관측·보상 목록 ──
em = u.event_manager
for mode in em.active_terms:
    for nm, tc in zip(em.active_terms[mode], em._mode_term_cfgs[mode]):
        prm = {k: (getattr(v, "__name__", None) or str(v))[:200] if callable(v) else str(v)[:200] for k, v in (tc.params or {}).items()}
        P(f"[event] {mode} | {nm} | {tc.func.__module__}.{tc.func.__name__} | {prm}")
om = u.observation_manager
for g in om.active_terms:
    for nm, tc, dim in zip(om.active_terms[g], om._group_obs_term_cfgs[g], om.group_obs_term_dim[g]):
        prm = {k: (getattr(v, "__name__", None) or str(v))[:120] if callable(v) else str(v)[:120] for k, v in (tc.params or {}).items()}
        P(f"[obs] {g} | {nm} | {tc.func.__module__}.{tc.func.__name__} | dim {tuple(dim)} | noise {None if tc.noise is None else (tc.noise.n_min, tc.noise.n_max)} | clip {tc.clip} | {prm}")
P(f"[obs] 차원 policy {om.group_obs_dim['policy']} · 행동 {u.action_manager.total_action_dim}")
rm = u.reward_manager
for nm, tc in zip(rm.active_terms, rm._term_cfgs):
    P(f"[reward] {nm} | {tc.weight}")
P(f"[cfg] episode_length_s {env_cfg.episode_length_s} · decimation {env_cfg.decimation} · sim dt {env_cfg.sim.dt} · usd {env_cfg.scene.robot.spawn.usd_path}")
ac = env_cfg.actions.joint_pos
P(f"[cfg] action scale {ac.scale} · delay {ac.min_delay_steps}-{ac.max_delay_steps} · slew {ac.max_delta_per_step} · lpf {ac.lpf_beta_range} · backlash {ac.backlash_rad_range}")
for an, a in robot.actuators.items():
    P(f"[actuator] {an} | {type(a).__name__} | kp(env0) {[round(float(v), 2) for v in a.stiffness[0]]} | kd(env0) {[round(float(v), 3) for v in a.damping[0]]} | effort {[round(float(v), 1) for v in a.effort_limit[0]]}")

# ── (a) 질량·고정 ──
pv = robot.root_physx_view; bn = list(robot.body_names); jn = list(robot.data.joint_names)
m = pv.get_masses(); c = pv.get_coms(); I = pv.get_inertias()
P(f"[mass] 전체 질량 env 평균 {float(m.sum(-1).mean()):.4f} kg · 범위 {float(m.sum(-1).min()):.4f}–{float(m.sum(-1).max()):.4f} · base 질량 범위 {float(m[:, bn.index('base')].min()):.4f}–{float(m[:, bn.index('base')].max()):.4f} "
  f"· base 뺀 합(env 범위) {float((m.sum(-1) - m[:, bn.index('base')]).min()):.4f}–{float((m.sum(-1) - m[:, bn.index('base')]).max()):.4f} · default_mass 합 {float(robot.data.default_mass.sum(-1).mean()):.4f}")
for nm in ("left_hip_a_joint", "right_hip_a_joint", "left_hip_r_joint", "right_hip_r_joint"):
    b = bn.index(nm)
    P(f"[mass] {nm}: 질량 {float(m[0, b]):.5f} (env 범위 {float(m[:, b].min()):.5f}–{float(m[:, b].max()):.5f}) · com {[round(float(v), 6) for v in c[0, b, :3]]} · 관성 9 {[f'{float(v):.4e}' for v in I[0, b]]}")
lk = getattr(u, "_nb_lock", None); LID = [jn.index("left_hip_r_joint"), jn.index("right_hip_r_joint")]
lim = robot.data.joint_pos_limits[:, LID]
P(f"[lock] hip_r 한계(env0) {torch.rad2deg(lim[0]).cpu().tolist()}° · 한계 폭 최대 {float(torch.rad2deg((lim[..., 1] - lim[..., 0]).max())):.4f}° · 한계 중심 env 범위 {float(torch.rad2deg(lim.mean(-1).min())):.3f}–{float(torch.rad2deg(lim.mean(-1).max())):.3f}° "
  f"· 기본자세 hip_r(env 범위) {float(torch.rad2deg(robot.data.default_joint_pos[:, LID].min())):.3f}–{float(torch.rad2deg(robot.data.default_joint_pos[:, LID].max())):.3f}° · _nb_lock {'있음' if lk else '없음'}")
TGT = lk["tgt"].clone() if lk else torch.zeros(N, 2, device=dev)

# ── (c) 체크포인트 굴리기 ──
log_root = os.path.join(R, "logs", "rsl_rl", agent_cfg.experiment_name)
path = get_checkpoint_path(log_root, args.load_run, args.checkpoint)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device); runner.load(path)
policy = runner.get_inference_policy(device=dev); P(f"[policy] {path}")
tm = u.termination_manager; tnames = list(tm.active_terms)
term_n = {k: 0 for k in tnames}; rew_sum = 0.0; rterm = torch.zeros(len(rm.active_terms), device=dev)
dev_max = torch.zeros(N, device=dev); vel_max = 0.0; a_lock_s = 0.0; a_oth_s = 0.0; a_oth_max = 0.0; a_lock_big = 0.0; vel_s = 0.0; dev_s = 0.0; tau_lock_s = 0.0; tau_lock_max = 0.0; ep_len_done = []; obs_lock_abs = 0.0; obs_other_abs = 0.0; act_lock_abs = 0.0; n_fin = True
# 관측에서 고정 관절 열(항마다 이력 5 프레임 × 12 관절, 평탄화) 위치
off = 0; cols_lock = []; cols_other = []
for nm, dim in zip(om.active_terms["policy"], om.group_obs_term_dim["policy"]):
    d = int(dim[0]) if len(dim) == 1 else int(torch.tensor(dim).prod())
    if nm in ("joint_pos", "joint_vel", "joint_vel_ema"):
        assert d % 12 == 0
        for f in range(d // 12):
            for j in range(12):
                (cols_lock if j in LID else cols_other).append(off + f * 12 + j)
    off += d
obs = env.get_observations()
with torch.inference_mode():
    for k in range(args.steps):
        a = policy(obs)
        o = obs["policy"] if not isinstance(obs, torch.Tensor) else obs
        obs_lock_abs = max(obs_lock_abs, float(o[:, cols_lock].abs().max())); obs_other_abs = max(obs_other_abs, float(o[:, cols_other].abs().max()))
        act_lock_abs = max(act_lock_abs, float(a[:, LID].abs().max()))
        OID = [j for j in range(12) if j not in LID]
        a_lock_s += float(a[:, LID].abs().mean()); a_oth_s += float(a[:, OID].abs().mean()); a_oth_max = max(a_oth_max, float(a[:, OID].abs().max()))
        a_lock_big += float((a[:, LID].abs() > 4.0).float().mean())
        ep_before = u.episode_length_buf.clone()
        obs, rew, dones, _ = env.step(a)
        n_fin = n_fin and bool(torch.isfinite(rew).all()) and bool(torch.isfinite(a).all())
        rew_sum += float(rew.mean()); rterm += rm._step_reward.mean(0)
        for tn in tnames:
            term_n[tn] += int(tm.get_term(tn).sum())
        d = dones.bool()
        if bool(d.any()):
            ep_len_done.extend((ep_before[d] + 1).cpu().tolist())
        q = robot.data.joint_pos[:, LID]; live = ~d
        dev_max[live] = torch.maximum(dev_max[live], torch.rad2deg((q - TGT).abs().amax(-1))[live])
        vel_max = max(vel_max, float(robot.data.joint_vel[:, LID].abs().max())); vel_s += float(robot.data.joint_vel[:, LID].abs().mean())
        dev_s += float(torch.rad2deg((q - TGT).abs()).mean())
        _tq = robot.data.applied_torque[:, LID].abs(); tau_lock_s += float(_tq.mean()); tau_lock_max = max(tau_lock_max, float(_tq.max()))
T = args.steps * u.step_dt; envsec = N * T
P(f"[roll] {args.steps} 스텝({T:.1f} s) · 유한값 {'예' if n_fin else '아니오(NaN/Inf)!'} · 스텝 평균 보상 {rew_sum / args.steps:.5f} · 20 s 환산 {rew_sum / args.steps * 20.0 / u.step_dt:.2f}")
P("[roll] 종료 사유별 건수(100 env·s 당): " + " · ".join(f"{k} {v} ({100.0 * v / envsec:.3f})" for k, v in term_n.items()))
fall = sum(v for k, v in term_n.items() if k != "time_out")
P(f"[roll] 낙상류 종료(time_out 제외) {fall} 건 = 100 env·s 당 {100.0 * fall / envsec:.3f} · 끝난 에피소드 {len(ep_len_done)} 개 평균 길이 {(sum(ep_len_done) / max(1, len(ep_len_done))) * u.step_dt:.2f} s")
P("[roll] 보상 항별 스텝 평균(가중·dt 적용 뒤): " + " · ".join(f"{nm} {float(v) / args.steps:+.5f}" for nm, v in zip(rm.active_terms, rterm)))
P(f"[lock] 고정 관절 |q−θ| 최대: 전 env {float(dev_max.max()):.4f}° · env 중앙값 {float(dev_max.median()):.4f}° · 99 % {float(torch.quantile(dev_max, 0.99)):.4f}° · 관절 속도 최대 {vel_max:.3f} rad/s")
P(f"[obs] 정책 관측의 고정 관절 열(joint_pos·joint_vel·joint_vel_ema × 이력) |값| 최대 {obs_lock_abs:.3e} · 다른 관절 열 |값| 최대 {obs_other_abs:.3f} · 정책이 낸 고정 관절 행동 |a| 최대 {act_lock_abs:.3f}")
P(f"[lock] 평균: |q−θ| {dev_s / args.steps:.4f}° · |관절 속도| {vel_s / args.steps:.4f} rad/s · |PD 토크| {tau_lock_s / args.steps:.3f} N·m (최대 {tau_lock_max:.2f}) · 정책 행동 |a| 평균 고정 관절 {a_lock_s / args.steps:.3f} / 다른 관절 {a_oth_s / args.steps:.3f} · 다른 관절 |a| 최대 {a_oth_max:.1f} · 고정 관절 |a|>4 비율 {100 * a_lock_big / args.steps:.3f} %")
m2 = pv.get_masses()
P(f"[mass] 끝 재확인: 처음 읽음 대비 링크 질량 최대 차 {float((m2 - m).abs().max()):.2e} kg · 전체 질량 env 평균 {float(m2.sum(-1).mean()):.4f} kg")
P("DONE")
open(OUT, "w").write("\n".join(L) + "\n")
sys.stdout.flush()
os._exit(0)      # Kit 종료가 멈추는 일이 있어(1009 실측 10 분) 결과 파일을 쓴 뒤 바로 끝낸다
