"""1006 보상 지형 해부 — 반사실(counterfactual) 보상 재생.

같은 정책·같은 학습 환경(외란·DR·지연 분포 그대로)에서 env 를 (행동 명령 b, 보상 명령 r) 짝 그룹으로 나눈다.
  - 정책 관측의 명령 채널(velocity_commands 30:45, 5 프레임 × (vx,vy,wz))은 b 로 덮어쓴다 → 궤적(서기/앞 걷기/제자리 걸음)은 b 가 정한다.
  - env 명령 항(보상 함수·게이트·sb0 사건·cadence 클록이 읽는 값)은 r → 같은 궤적을 r 기준으로 채점.
  - 주의: cadence 클록·sb0 일정은 r 을 따른다(관측 클록도 r 의 cadence). 범주 게이트(_omni_cat)도 r 로 설정.
항별로 raw(=step_reward/weight), weight, 초당 기여(= raw×weight, Isaac Lab 은 틱마다 ×dt), 0 아닌 틱 비율(사건 빈도),
첫 종료 전 생존 구간 평균, 총합 대비 비율을 그룹별로 낸다. 출력: log_picture/landscape_<tag>.csv + .txt

사용 예:
  isaaclab.sh -p rl_walking/scripts/reward_landscape.py --headless --task Biped12-HangStand-W2q-v0 --load_run <run> --checkpoint model_25590.pt \
     --behav "STAND:0,0,0;FWD:0.10,0,0" --rcmd "ZERO:0,0,0;FWD:0.10,0,0;BACK:-0.10,0,0;LATL:0,0.08,0;LATR:0,-0.08,0;YAW:0,0,0.3" --seconds 12 --out_tag w2q
"""
import argparse, os, sys, math
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402
REPO = "/home/ryu/humanoid_leg_test1"                      # 로그·출력 위치(본 저장소)
CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # 코드(worktree 에서 돌리면 worktree 코드)
sys.path.insert(0, CODE)
p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--num_envs", type=int, default=0, help="0 = 그룹당 --per_group")
p.add_argument("--per_group", type=int, default=64)
p.add_argument("--behav", required=True, help="'LABEL:vx,vy,wz;...' 정책이 보는 명령")
p.add_argument("--rcmd", required=True, help="'LABEL:vx,vy,wz;...' 보상이 보는 명령")
p.add_argument("--seconds", type=float, default=12.0)
p.add_argument("--warmup", type=float, default=2.0, help="집계 제외 초기 구간[s]")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--no_dist", action="store_true", help="외란·속도 밀기 끔(기본 = 학습 환경 그대로)")
p.add_argument("--out_tag", default="landscape")
p.add_argument("--stand_obs", default="", help="쉼표 목록: 이 행동 그룹은 정책 관측의 gait_clock 블록을 (0,0)으로 덮는다(STEP 채점 아래 '서기' 궤적 대리)")
p.add_argument("--lat_force", default="", help="'LABEL:Fy;...' 행동 그룹별 몸통(base 링크 프레임) y 방향 일정 힘[N](시간 평균) — 옆 이동 대리")
cli_args.add_rsl_rl_args(p)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import torch, gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import rl_walking  # noqa: F401,E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def parse(s):
    out = []
    for g in s.split(";"):
        g = g.strip()
        if not g:
            continue
        lab, v = g.split(":")
        out.append((lab, tuple(float(x) for x in v.split(","))))
    return out


B = parse(a.behav); Rc = parse(a.rcmd)
combos = [(bi, ri) for bi in range(len(B)) for ri in range(len(Rc))]
N = a.num_envs or a.per_group * len(combos)
env_cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, a)
env_cfg.scene.num_envs = N; env_cfg.seed = a.seed
env_cfg.curriculum = None
c = env_cfg.commands.base_velocity
c.resampling_time_range = (1e6, 1e6); c.rel_standing_envs = 0.0
if hasattr(c, "rel_heading_envs"):
    c.rel_heading_envs = 0.0
c.heading_command = False
if a.no_dist:
    for k in ("disturbance", "push_robot"):
        if getattr(env_cfg.events, k, None) is not None:
            setattr(env_cfg.events, k, None)
env_cfg.episode_length_s = a.seconds + 2.0
env = gym.make(a.task, cfg=env_cfg)
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
U = env.unwrapped; dev = U.device; dt = U.step_dt
log_root = os.path.abspath(os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name))
ck = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ck); policy = runner.get_inference_policy(device=dev)
print(f"[landscape] {a.task} · {ck} · env {N} · 그룹 {len(combos)} · 외란 {'끔' if a.no_dist else '학습 그대로'}", flush=True)

gi = torch.arange(N, device=dev) % len(combos)
bvec = torch.tensor([B[combos[k][0]][1] for k in range(len(combos))], device=dev)[gi]     # (N,3)
rvec = torch.tensor([Rc[combos[k][1]][1] for k in range(len(combos))], device=dev)[gi]
cmd_term = U.command_manager.get_term("base_velocity")
RM = U.reward_manager; names = list(RM._term_names); W = torch.tensor([float(cf.weight) for cf in RM._term_cfgs], device=dev)
T = len(names)


def cat_of(v):   # install_omni_cmd 범주 규약: 0 앞 · 1 뒤 · 2 옆 · 3 회전 · 4 앞+회전
    vx, vy, wz = v[:, 0], v[:, 1], v[:, 2]
    cat = torch.zeros(v.shape[0], dtype=torch.long, device=dev)
    cat = torch.where(vy.abs() > 1e-6, torch.full_like(cat, 2), cat)
    cat = torch.where((vy.abs() <= 1e-6) & (vx < -1e-6), torch.full_like(cat, 1), cat)
    cat = torch.where((vy.abs() <= 1e-6) & (vx.abs() <= 1e-6) & (wz.abs() > 1e-6), torch.full_like(cat, 3), cat)
    cat = torch.where((vx > 1e-6) & (wz.abs() > 1e-6), torch.full_like(cat, 4), cat)
    return cat


def force_cmd():
    cmd_term.vel_command_b[:] = rvec
    cmd_term.time_left[:] = 1e9
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False
    U._omni_cat = cat_of(rvec)


force_cmd()
# 1006 W3S2 반사실 옵션: 서기 대리(클록 관측 0) · 옆 이동 대리(몸통 y 힘)
_stand_lab = {x.strip() for x in a.stand_obs.split(",") if x.strip()}
stand_mask = torch.tensor([B[combos[k][0]][0] in _stand_lab for k in range(len(combos))], device=dev)[gi]
_lf = {lab: v for lab, v in ((g.split(":")[0].strip(), float(g.split(":")[1])) for g in a.lat_force.split(";") if g.strip())}
fy = torch.tensor([_lf.get(B[combos[k][0]][0], 0.0) for k in range(len(combos))], device=dev)[gi]
_om = U.observation_manager; _off = 0; clk = None
for _n, _cf, _d in zip(_om._group_obs_term_names["policy"], _om._group_obs_term_cfgs["policy"], _om._group_obs_term_dim["policy"]):
    _w = int(torch.tensor(_d).prod())
    if getattr(_cf.func, "__name__", "") == "gait_clock":
        clk = slice(_off, _off + _w)
    _off += _w
print(f"[landscape] gait_clock 관측 {clk} · 서기 대리 그룹 {sorted(_stand_lab)} · 옆 힘 {_lf}", flush=True)
if bool(stand_mask.any()):
    assert clk is not None, "gait_clock 관측 항을 못 찾음"
_dec = int(env_cfg.decimation)
rob0 = U.scene["robot"]
w3s = {"n_td": torch.zeros(N, device=dev), "dy": torch.zeros(N, device=dev), "dx": torch.zeros(N, device=dev), "bad": torch.zeros(N, device=dev),
       "ok": torch.zeros(N, device=dev), "roll": torch.zeros(N, device=dev), "wsum": torch.zeros(N, device=dev), "wn": torch.zeros(N, device=dev)}
obs = venv.get_observations()
sum_c = torch.zeros(N, T, device=dev); nz = torch.zeros(N, T, device=dev); n_t = torch.zeros(N, device=dev)
alive = torch.ones(N, dtype=torch.bool, device=dev)
vb = torch.zeros(N, 3, device=dev)
steps = int(a.seconds / dt); w0 = int(a.warmup / dt)
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
for i in range(steps):
    force_cmd()
    if isinstance(obs, torch.Tensor):
        o2 = obs.clone(); o2[:, 30:45] = bvec.repeat(1, 5)
        if clk is not None and bool(stand_mask.any()):
            o2[stand_mask, clk] = 0.0
    else:
        o2 = obs.clone(); _po = o2["policy"].clone(); _po[:, 30:45] = bvec.repeat(1, 5)
        if clk is not None and bool(stand_mask.any()):
            _po[stand_mask, clk] = 0.0
        o2["policy"] = _po
    with torch.inference_mode():
        act = policy(o2)
    if bool((fy != 0).any()):                            # 순간 렌치: 첫 물리 서브스텝만 적용되고 지워짐 → ×decimation 으로 시간 평균 Fy
        Fb = torch.zeros(N, 1, 3, device=dev); Fb[:, 0, 1] = fy * _dec
        rob0.instantaneous_wrench_composer.add_forces_and_torques(forces=Fb, torques=None, positions=None, body_ids=[0], env_ids=None, is_global=False)
    obs, _, dones, _ = venv.step(act)
    sr = RM._step_reward.clone()                        # (N,T) = raw × weight [초당]
    m = alive & (i >= w0)
    _L = getattr(U, "_w3lat", None); _S = getattr(U, "_w3", None)
    if _L is not None and _S is not None:               # W3S2 추적기: 유효 착지 Δy·폭 위반·명령 부호 정답(보상 명령 기준)
        td = (_S["ev_td"] > 0.5) & m.unsqueeze(1)
        w3s["n_td"] += td.float().sum(1); w3s["dy"] += (_L["td_dy"] * td).sum(1); w3s["dx"] += (_L["td_dx"] * td).sum(1)
        w3s["bad"] += (_L["width_bad"] * td).sum(1)
        w3s["ok"] += ((_L["td_dy"] * torch.sign(rvec[:, 1]).unsqueeze(1) > 0) & td).float().sum(1)
        w3s["wsum"] += _L["width"] * m; w3s["wn"] += m.float()
    pgb = rob0.data.projected_gravity_b
    w3s["roll"] += torch.rad2deg(torch.atan2(pgb[:, 1], -pgb[:, 2])) * m
    sum_c += sr * m.unsqueeze(1); nz += (sr.abs() > 1e-9).float() * m.unsqueeze(1); n_t += m.float()
    rob = U.scene["robot"]
    v_b = quat_apply_inverse(rob.data.root_quat_w, rob.data.root_lin_vel_w)
    vb[:, 0] += v_b[:, 0] * m; vb[:, 1] += v_b[:, 1] * m; vb[:, 2] += rob.data.root_ang_vel_w[:, 2] * m
    alive &= ~dones.bool()

rows = []; L = [f"보상 지형 · {a.task} · {os.path.basename(ck)} · {a.seconds}s(앞 {a.warmup}s 제외) · 외란 {'끔' if a.no_dist else '학습 그대로'} · seed {a.seed}", ""]
for k, (bi, ri) in enumerate(combos):
    mk = gi == k; nt = n_t[mk].sum().clamp(min=1.0)
    contrib = sum_c[mk].sum(0) / nt                      # 그룹 평균 초당 기여
    freq = nz[mk].sum(0) / nt
    tot = float(contrib.sum()); pos = float(contrib.clamp(min=0).sum())
    surv = float(alive[mk].float().mean()); v = vb[mk].sum(0) / nt
    L.append(f"## 행동 {B[bi][0]}{B[bi][1]} × 보상명령 {Rc[ri][0]}{Rc[ri][1]} — 총 {tot:+.3f}/s · 생존 {100*surv:.0f} % · 실제 (vx,vy,wz) ({float(v[0]):+.3f},{float(v[1]):+.3f},{float(v[2]):+.3f})")
    ntd = float(w3s["n_td"][mk].sum())
    L.append(f"  [W3] 유효 착지 {ntd:.0f} · 평균 Δx {float(w3s['dx'][mk].sum())/max(ntd,1):+.4f} · Δy {float(w3s['dy'][mk].sum())/max(ntd,1):+.4f} m"
             f" · vy 부호 정답 {float(w3s['ok'][mk].sum())/max(ntd,1):.3f} · 폭 위반 {float(w3s['bad'][mk].sum())/max(ntd,1):.3f}"
             f" · 평균 발 간격 {float(w3s['wsum'][mk].sum())/max(float(w3s['wn'][mk].sum()),1):.3f} m · 몸통 롤 평균 {float(w3s['roll'][mk].sum())/float(nt):+.2f}°")
    order = torch.argsort(contrib.abs(), descending=True)
    for j in order.tolist():
        if abs(float(contrib[j])) < 1e-4:
            continue
        rw = float(contrib[j]) / float(W[j]) if float(W[j]) != 0 else float("nan")
        L.append(f"  {names[j]:<28} w {float(W[j]):+8.3g} · raw {rw:+.4f} · 기여 {float(contrib[j]):+.4f}/s · 비율(양합) {100*float(contrib[j])/max(pos,1e-9):+6.1f} % · 사건빈도 {100*float(freq[j]):5.1f} %")
        rows.append((B[bi][0], Rc[ri][0], names[j], float(W[j]), rw, float(contrib[j]), float(contrib[j]) * dt, 100 * float(contrib[j]) / max(pos, 1e-9), 100 * float(freq[j]), tot, surv))
    L.append("")
os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True)
with open(os.path.join(REPO, "log_picture", f"landscape_{a.out_tag}.csv"), "w") as f:
    f.write("behav,rcmd,term,weight,raw,contrib_per_s,contrib_per_tick,frac_of_pos_pct,nonzero_freq_pct,total_per_s,survival\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")
txt = "\n".join(L)
with open(os.path.join(REPO, "log_picture", f"landscape_{a.out_tag}.txt"), "w") as f:
    f.write(txt + "\n")
print(txt, flush=True)
env.close(); app.close()
