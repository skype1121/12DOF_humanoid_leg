"""W3 걷기 후보 failure envelope·민감도 평가기(1007, 평가 전용 — 학습·환경 코드 불변).

한 실행 안에서 **env 그룹 = 명령 × 섭동 격자**(env i → 그룹 i % G, 섞어 배치해 같은 seed 의 DR 표본이 그룹마다 고르게)로
1 축 스윙(--grid 1 개) 또는 2 축 격자(--grid 2 개)를 한 번에 잰다. 각 실행은 기준값(0·1 배) 그룹을 대조군으로 함께 포함해야 한다.
환경: 학습 태스크 그대로(학습 DR 켬 — CLAUDE.md 3), 외란·밀기·커리큘럼·명령 리샘플 끔, 액션 LPF β 0.27 고정(eval_w3 와 같은 하네스).
섭동은 학습 DR **위에** 얹는다(지연·CoM·유격·쿨롱은 예외: 그 축을 고르면 해당 DR 을 그 값으로 고정 — 아래 표).

축(값 의미) — 관측 계약(차원·순서) 불변, 접촉 오류는 **actor 관측에만**(실제 접촉 물리·채점 불변):
  delay      액션 지연 [틱] env 별 고정(학습 1–5 균등 대신). 기본 --base_delay
  hipf_sym / hipa_sym   힙 앞뒤 / 좌우 엔코더 영점 오프셋 [°], 해부학적 대칭(왼 +v, 오 −v 관절좌표 — symmetry.py 쌍 부호 −1)
  hipf_L / hipa_L       같은 오프셋을 왼쪽 관절에만(좌우 비대칭)
  hipf_anti / hipa_anti 관절좌표 같은 값(해부학적 반대 방향)
       ↳ 영점 오프셋 모델: 관측 q_obs = q + b, 실제 구동 목표 = 명령 − b (actions.pre_delay_hook) — 실물 엔코더 오프셋과 같은 식
  imu_pitch / imu_roll  IMU 자세 편향 [°] — 정책 중력벡터 관측을 회전(eval 피치·롤 식 기준 +v 로 읽힘). 학습 calib_bias DR 위에 더함
  imu_noise  추가 IMU 잡음 배수 k — 자이로 ±0.05k rad/s·중력 ±0.02k 균등(W3S3d 학습 대역의 k 배), gyro_ema 에는 같은 α 로 EMA 한 잡음
  yaw_drift  IMU 요 표류율 [°/s] — 헤딩 고리 측정 요 ψ+r·t, 자이로 z 관측(원시·EMA)에 같은 편향 r
  kp / kd    전 관절 PD 게인 배수(학습 actuator_gains DR 위에 곱) · kp_L / kd_L 왼다리만
  c_delay    접촉 관측 추가 지연 [틱] · c_hold 표본 유지 주기 [틱](env 별 무작위 위상) · c_drop 틱당 누락(0 강제 5 틱) 시작 확률 · c_chat 틱당 비트 반전 확률
  cmd_scale  그룹 명령 배수(학습 범위 밖 포함)
  com_x / com_y  base CoM 오프셋 [m] (base_com DR 의 그 성분을 0 으로 고정하고 이 값)
  play / coul    발목 유격·쿨롱 배수(기본 자산 값 × v; 이 축을 고르면 그 DR 대신 고정)
지표(채점 = 워밍업 뒤, 시작 과도 낙상 분리): 채점 낙상률(워밍업 생존 대비, Wilson 95 % CI), 교대율, 유효 스텝률, 명령 추종 이득 두 가지(본체 프레임 평균 속도/명령 ·
  채점 시작 요 프레임 직선 변위/(명령×시간) · 진행 방향 프레임(요 1 주기 EMA) 평균 속도/명령 — 판정은 진행 방향 프레임(본체 프레임은 걸음 흔들림과 요 흔들림의
  상관으로 편향, 직선 변위는 요 표류 cos 로 편향),
  회전은 감싸지 않은 누적 요/명령), 요 표류(누적 감싸기 없음, 비회전), 헤딩 고리 참 요 오차, 피치·롤 평균·RMS·최대.
붕괴 판정(그룹): 채점 낙상 > 3 % 또는 추종 이득 < 0.6(명령 축만).
사용: isaaclab.sh -p rl_walking/scripts/eval_w3_envelope.py --headless --task <T> --load_run <run> --checkpoint model_N.pt \\
      --cmds "f06:0.06,0,0|z:0,0,0" --grid "delay=1,2,3,4,5,6" [--grid ...] [--fix "kp=0.8"] --seed 1 --out_tag <tag>
출력: log_picture/w3env_<tag>.json(+ .txt 표)."""
import argparse, os, sys, math, json, time, itertools, functools
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402
REPO = os.environ.get("W3EVAL_REPO", "/home/ryu/humanoid_leg_test1")
CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, CODE)
p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W3S3d-v0")
p.add_argument("--per_group", type=int, default=256)
p.add_argument("--seconds", type=float, default=20.0, help="채점 구간[s](워밍업 뒤)")
p.add_argument("--warmup", type=float, default=3.0)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--base_delay", type=int, default=2)
p.add_argument("--cmds", default="z:0,0,0", help="'이름:vx,vy,wz|…' 명령 그룹")
p.add_argument("--grid", action="append", default=[], help="'축=v1,v2,…' (최대 2 회)")
p.add_argument("--fix", action="append", default=[], help="'축=v' 전 그룹 고정 섭동")
p.add_argument("--sweep", default="", help="여러 축 1 축 스윕을 한 실행에: '축=v1,v2;축2=…' → 그룹 = 명령 × (기준 1 + 각 축의 기준 아닌 수준). --grid 와 같이 쓰지 않음")
p.add_argument("--heading_loop", type=float, default=0.0, help="> 0: 비회전 그룹에 외부 헤딩 P 고리(배포 식, wz₀ ≠ 0 그룹은 꺼짐)")
p.add_argument("--hl_wzmax", type=float, default=0.4)
p.add_argument("--ramp_off", action="store_true")
p.add_argument("--out_tag", default="env")
cli_args.add_rsl_rl_args(p)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True

AX_DEFAULT = {"delay": float(a.base_delay), "hipf_sym": 0.0, "hipa_sym": 0.0, "hipf_L": 0.0, "hipa_L": 0.0, "hipf_anti": 0.0, "hipa_anti": 0.0,
              "imu_pitch": 0.0, "imu_roll": 0.0, "imu_noise": 0.0, "yaw_drift": 0.0, "kp": 1.0, "kd": 1.0, "kp_L": 1.0, "kd_L": 1.0,
              "c_delay": 0.0, "c_hold": 1.0, "c_drop": 0.0, "c_chat": 0.0, "cmd_scale": 1.0, "com_x": 0.0, "com_y": 0.0, "play": 1.0, "coul": 1.0}
CMDS = []
for tok in a.cmds.split("|"):
    nm, v = tok.split(":"); vv = [float(x) for x in v.split(",")]; assert len(vv) == 3; CMDS.append((nm.strip(), vv))
GRID = []
for g in a.grid:
    k, v = g.split("="); assert k in AX_DEFAULT, k; GRID.append((k, [float(x) for x in v.split(",")]))
assert len(GRID) <= 2
FIX = {}
for f in a.fix:
    for kv in f.split(";"):
        if kv.strip():
            k, v = kv.split("="); assert k in AX_DEFAULT, k; FIX[k] = float(v)
USED = set(FIX) | {k for k, _ in GRID}
SWEEP = []
if a.sweep.strip():
    assert not GRID, "--sweep 와 --grid 는 같이 쓰지 않는다"
    for kv in a.sweep.split(";"):
        if kv.strip():
            k, v = kv.split("="); assert k in AX_DEFAULT, k; SWEEP.append((k, [float(x) for x in v.split(",")]))
    USED |= {k for k, _ in SWEEP}
    GRID = [(k, sorted(set([AX_DEFAULT.get(k) if k not in FIX else FIX[k]] + vals))) for k, vals in SWEEP]   # 표 열(축 값) 용
GROUPS = []   # (cmd 이름, cmd, {축: 값})
for (cn, cv) in (CMDS if SWEEP else []):
    _b = dict(AX_DEFAULT); _b.update(FIX)
    GROUPS.append((cn, cv, dict(_b)))
    for k, vals in SWEEP:
        for v in vals:
            if v != _b[k]:
                ax = dict(_b); ax[k] = v; GROUPS.append((cn, cv, ax))
for (cn, cv) in (CMDS if not SWEEP else []):
    for combo in itertools.product(*[vals for _, vals in GRID]) if GRID else [()]:
        ax = dict(AX_DEFAULT); ax.update(FIX)
        for (k, _), v in zip(GRID, combo):
            ax[k] = v
        GROUPS.append((cn, cv, ax))
G = len(GROUPS); N_REQ = G * a.per_group
print(f"[env] 그룹 {G} × {a.per_group} = {N_REQ} env, 축 {[k for k, _ in GRID]} 고정 {FIX}", flush=True)

os.environ["STEP_ASSIST_ALPHA"] = "0.0"; os.environ["W3_GAIT_ENABLE"] = "1"; os.environ.setdefault("TURN_ALPHA", "0"); os.environ["W3_HLOOP"] = "0"
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
env_cfg.scene.num_envs = N_REQ; env_cfg.seed = a.seed; env_cfg.curriculum = None
c = env_cfg.commands.base_velocity
c.resampling_time_range = (1e6, 1e6); c.rel_standing_envs = 0.0
for _ev in ("disturbance", "push_robot"):
    if getattr(env_cfg.events, _ev, None) is not None:
        setattr(env_cfg.events, _ev, None)
act = env_cfg.actions.joint_pos
act.lpf_beta_range = (0.27, 0.27)
DMAX = int(max(g[2]["delay"] for g in GROUPS))
act.min_delay_steps = act.max_delay_steps = DMAX          # 버퍼 크기용 — env 별 실제 지연은 리셋 뒤 덮어씀
if hasattr(act, "delay_probs"):
    act.delay_probs = ()
if a.ramp_off and getattr(env_cfg.events, "hang_support_reset", None) is not None:
    _hp = env_cfg.events.hang_support_reset.params
    if _hp.get("ramp_rate_range") is not None:
        _hp["ramp_rate_range"] = (0.0, 0.0); _hp["ramp_cap_range"] = (0.0, 0.0)
_ev = env_cfg.events
if ("com_x" in USED or "com_y" in USED) and getattr(_ev, "base_com", None) is not None:
    _cr = dict(_ev.base_com.params["com_range"])
    if "com_x" in USED: _cr["x"] = (0.0, 0.0)
    if "com_y" in USED: _cr["y"] = (0.0, 0.0)
    _ev.base_com.params["com_range"] = _cr
_acts = env_cfg.scene.robot.actuators
for _k, _a in _acts.items():
    if "play" in USED and getattr(_a, "play_rad_range", None) is not None:
        _a.play_rad_range = None                           # 축이 곧 배수(기본 × v)
    if "coul" in USED and getattr(_a, "coulomb_range", None) is not None:
        _a.coulomb_range = None
T_END = a.warmup + a.seconds
env_cfg.episode_length_s = T_END + 2.0

# ── actor 관측 래퍼(함수 서명 보존: functools.wraps → 관리자 인자 검사 통과). 실제 값은 env 생성 뒤 PX 텐서가 채워지면 적용 ──
PX = {}
P = env_cfg.observations.policy


def _step_gate(env, key):
    """관측 함수가 한 정책 틱에 여러 번 불려도 상태 갱신은 한 번."""
    s = int(getattr(env, "common_step_counter", 0)); last = PX.get(key)
    PX[key] = s
    return last != s


def _wrap_obs(name, fn_mod):
    t = getattr(P, name, None)
    if t is None:
        return False
    orig = t.func

    @functools.wraps(orig)
    def f(env, *args, **kw):
        out = orig(env, *args, **kw)
        return fn_mod(env, out, kw) if PX.get("ready") else out
    t.func = f
    return True


def _jp_mod(env, out, kw):
    return out + PX["enc"][:, : out.shape[1]]


def _rot_g(g, th, ph):
    gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
    ct, st = torch.cos(th), torch.sin(th)
    gx1 = gx * ct - gz * st; gz1 = gz * ct + gx * st
    cp, sp = torch.cos(ph), torch.sin(ph)
    gy2 = gy * cp + gz1 * sp; gz2 = gz1 * cp - gy * sp
    return torch.stack([gx1, gy2, gz2], -1)


def _pg_mod(env, out, kw):
    g = _rot_g(out, PX["imu_th"], PX["imu_ph"])
    k = PX["imu_k"].unsqueeze(1)
    return g + (torch.rand_like(g) * 2 - 1) * 0.02 * k


def _gyro_mod(env, out, kw):
    k = PX["imu_k"].unsqueeze(1)
    o = out.clone(); o[:, 2] += PX["yaw_r"]
    return o + (torch.rand_like(o) * 2 - 1) * 0.05 * k


def _gyro_ema_mod(env, out, kw):
    al = float(kw.get("alpha", 0.1))
    if _step_gate(env, "_ge_step"):
        n = (torch.rand_like(out) * 2 - 1) * 0.05 * PX["imu_k"].unsqueeze(1)
        st = PX.get("gema")
        if st is None:
            st = n.clone()
        fresh = env.episode_length_buf <= 1
        st[fresh] = n[fresh]
        PX["gema"] = st * (1 - al) + al * n
    o = out + PX["gema"]
    o = o.clone(); o[:, 2] += PX["yaw_r"]
    return o


CBUF = 12


def _contact_mod(env, out, kw):
    if _step_gate(env, "_c_step"):
        n = out.shape[0]
        st = PX.get("cst")
        fresh = env.episode_length_buf <= 1
        if st is None:
            st = {"ring": out.unsqueeze(0).repeat(CBUF, 1, 1).clone(), "i": 0, "held": out.clone(), "drop": torch.zeros_like(out),
                  "ph": torch.randint(0, 10_000, (n,), device=out.device), "t": 0}
            PX["cst"] = st
        st["t"] += 1
        upd = ((st["t"] + st["ph"]) % PX["c_hold"].clamp(min=1)) == 0
        st["held"] = torch.where((upd | fresh).unsqueeze(1), out, st["held"])
        st["ring"][st["i"] % CBUF] = st["held"]
        st["ring"][:, fresh] = st["held"][fresh].unsqueeze(0)
        idx = (st["i"] - PX["c_delay"]) % CBUF
        o = st["ring"][idx, torch.arange(n, device=out.device)]
        st["i"] += 1
        start = torch.rand_like(o) < PX["c_drop"].unsqueeze(1)
        st["drop"] = torch.where(start, torch.full_like(o, 5.0), (st["drop"] - 1).clamp(min=0))
        st["drop"][fresh] = 0
        o = torch.where(st["drop"] > 0, torch.zeros_like(o), o)
        flip = torch.rand_like(o) < PX["c_chat"].unsqueeze(1)
        o = torch.where(flip, 1.0 - o, o)
        st["out"] = o
    return PX["cst"]["out"].clone()


WRAPPED = {"joint_pos": _wrap_obs("joint_pos", _jp_mod), "projected_gravity": _wrap_obs("projected_gravity", _pg_mod),
           "base_ang_vel": _wrap_obs("base_ang_vel", _gyro_mod), "gyro_ema": _wrap_obs("gyro_ema", _gyro_ema_mod),
           "feet_contact": _wrap_obs("feet_contact", _contact_mod)}
OBS_DESC = {k: (f"{getattr(getattr(P, k), 'func').__wrapped__.__module__}.{getattr(getattr(P, k), 'func').__wrapped__.__name__} params={getattr(P, k).params} "
                f"noise={getattr(P, k).noise} hist={getattr(P, k).history_length}") for k, w in WRAPPED.items() if w}
print("[env] 래핑한 actor 관측: " + json.dumps(OBS_DESC, ensure_ascii=False), flush=True)
EV_DESC = []
for k, v in vars(env_cfg.events).items():
    if not k.startswith("_") and v is not None and hasattr(v, "func"):
        EV_DESC.append(f"{k}[{v.mode}] {getattr(v.func, '__name__', v.func)} {v.params}")
print("[env] 이벤트 목록(CLAUDE.md 4): " + " || ".join(EV_DESC), flush=True)

env = gym.make(a.task, cfg=env_cfg)
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
U = env.unwrapped; dev = U.device; N = U.num_envs; dt = U.step_dt
assert N == N_REQ
log_root = os.path.abspath(os.path.join(REPO, "logs", "rsl_rl", agent_cfg.experiment_name))
ck = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ck); policy = runner.get_inference_policy(device=dev)
robot = U.scene["robot"]; sensor = U.scene.sensors["contact_forces"]
fid = mdp_hang._feet_body_ids(U); bid = mdp_hang._g21_bids(U)
cmd_term = U.command_manager.get_term("base_velocity")
MG = 9.81 * float(robot.data.default_mass[0].sum())
JN = list(robot.data.joint_names)
print(f"[env] 관절 순서 {JN}", flush=True)

# ── env 별 섭동 텐서 ──
gid = torch.arange(N, device=dev) % G
AX = {k: torch.tensor([g[2][k] for g in GROUPS], device=dev, dtype=torch.float32)[gid] for k in AX_DEFAULT}
CMD3 = torch.tensor([g[1] for g in GROUPS], device=dev, dtype=torch.float32)[gid] * AX["cmd_scale"].unsqueeze(1)


def _jidx(sub, side):
    """sub(hip_f·hip_a) 관절의 [왼, 오] 인덱스 — 이름에 l/r 표기(뒤 '_l_'·'left' 등) 탐색."""
    cand = [i for i, n in enumerate(JN) if sub in n]
    assert len(cand) == 2, (sub, cand, JN)
    def _isleft(n):
        nl = n.lower()
        return ("left" in nl) or nl.endswith("_l") or ("_l_" in nl) or nl.startswith("l_") or ("_l" in nl and "_r" not in nl)
    L = [i for i in cand if _isleft(JN[i])]; R = [i for i in cand if i not in L]
    assert len(L) == 1 and len(R) == 1, (sub, [JN[i] for i in cand])
    return L[0] if side == "L" else R[0]


enc = torch.zeros(N, len(JN), device=dev)
for jt, key in (("hip_f", "hipf"), ("hip_a", "hipa")):
    iL, iR = _jidx(jt, "L"), _jidx(jt, "R")
    vs, vl, va = (torch.deg2rad(AX[f"{key}_{s}"]) for s in ("sym", "L", "anti"))
    enc[:, iL] += vs + vl + va; enc[:, iR] += -vs + va
PX.update({"enc": enc, "imu_th": torch.deg2rad(AX["imu_pitch"]), "imu_ph": torch.deg2rad(AX["imu_roll"]), "imu_k": AX["imu_noise"],
           "yaw_r": torch.deg2rad(AX["yaw_drift"]), "c_delay": AX["c_delay"].long(), "c_hold": AX["c_hold"].long(), "c_drop": AX["c_drop"], "c_chat": AX["c_chat"]})
assert int(AX["c_delay"].max()) < CBUF
aterm = U.action_manager.get_term("joint_pos")
_aj = aterm._joint_ids if not isinstance(aterm._joint_ids, slice) else list(range(len(JN)))
_aj = torch.as_tensor(_aj, device=dev, dtype=torch.long)
if bool((enc != 0).any()):
    _enc_a = enc[:, _aj]
    aterm.pre_delay_hook = lambda tgt: tgt - _enc_a          # 실물: 엔코더 기준 명령 → 실제 관절 = 명령 − b
if aterm._max_delay > 0:
    aterm._delay[:] = AX["delay"].long()
else:
    assert int(AX["delay"].max()) == 0
# 게인: 액추에이터 텐서(N, j_act) 에 env 별 배수 곱(학습 DR 표본 위), 내재 드라이브는 sim 에 쓰기
_Lset = set()
for jt in ("hip_f", "hip_a", "hip_r", "knee", "ankle_f", "ankle_r"):
    try:
        _Lset.add(_jidx(jt, "L"))
    except AssertionError:
        pass
leftmask = torch.tensor([i in _Lset for i in range(len(JN))], device=dev)
GAIN_LOG = {}
for nm, A in robot.actuators.items():
    ji = A.joint_indices if not isinstance(A.joint_indices, slice) else list(range(len(JN)))
    ji_t = torch.as_tensor(ji, device=dev, dtype=torch.long)
    lm = leftmask[ji_t].unsqueeze(0).float()
    skp = AX["kp"].unsqueeze(1) * (1 + lm * (AX["kp_L"].unsqueeze(1) - 1))
    skd = AX["kd"].unsqueeze(1) * (1 + lm * (AX["kd_L"].unsqueeze(1) - 1))
    if bool((skp != 1).any()) or bool((skd != 1).any()):
        A.stiffness[:] = A.stiffness * skp; A.damping[:] = A.damping * skd
        if any("Implicit" in c.__name__ for c in type(A).__mro__):
            robot.write_joint_stiffness_to_sim(A.stiffness, joint_ids=ji_t); robot.write_joint_damping_to_sim(A.damping, joint_ids=ji_t)
    if hasattr(A, "_play_base") and "play" in USED:
        A._play[:] = A._play_base * AX["play"].unsqueeze(1)
    if hasattr(A, "_tau_c_base") and "coul" in USED:
        A._tau_c[:] = A._tau_c_base * AX["coul"].unsqueeze(1)
    GAIN_LOG[nm] = {"type": type(A).__name__, "joints": [JN[i] for i in ji], "kp0_med": float(A.stiffness.median()), "kd0_med": float(A.damping.median()),
                    "play_med_deg": (float(torch.rad2deg(A._play).median()) if hasattr(A, "_play") else None),
                    "coul_med": (float(A._tau_c.median()) if hasattr(A, "_tau_c") else None)}
print("[env] 액추에이터: " + json.dumps(GAIN_LOG, ensure_ascii=False), flush=True)
if "com_x" in USED or "com_y" in USED:
    _bn = list(robot.data.body_names); _bi = _bn.index("base") if "base" in _bn else 0
    coms = robot.root_physx_view.get_coms().clone()
    coms[:, _bi, 0] += AX["com_x"].cpu(); coms[:, _bi, 1] += AX["com_y"].cpu()
    robot.root_physx_view.set_coms(coms, torch.arange(N, device="cpu"))
PX["ready"] = True


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


HL_ON = (a.heading_loop > 0) & (CMD3[:, 2].abs() < 1e-6)
HL = {"tgt": None, "t": 0.0}


def set_cmd(t):
    cmd_term.vel_command_b[:] = CMD3
    if a.heading_loop > 0:
        psi_m = yaw_of(robot.data.root_quat_w) + PX["yaw_r"] * t          # IMU 요(표류 포함)
        HL["tgt"] = psi_m.clone() if HL["tgt"] is None else _wrap(HL["tgt"] + CMD3[:, 2] * dt)
        wz = (CMD3[:, 2] + a.heading_loop * _wrap(HL["tgt"] - psi_m)).clamp(-a.hl_wzmax, a.hl_wzmax)
        cmd_term.vel_command_b[:, 2] = torch.where(HL_ON, wz, CMD3[:, 2])
    if hasattr(cmd_term, "is_standing_env"):
        cmd_term.is_standing_env[:] = False
    if hasattr(cmd_term, "is_heading_env"):
        cmd_term.is_heading_env[:] = False


obs = venv.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
steps = int(round(T_END / dt))
fz = lambda *s: torch.zeros(*s, device=dev)
alive = torch.ones(N, dtype=torch.bool, device=dev); t_fall = torch.full((N,), float("nan"), device=dev)
prev_c = None
in_air = torch.zeros(N, 2, dtype=torch.bool, device=dev); air_t = fz(N, 2); z0 = fz(N, 2); zmax = fz(N, 2)
pend = torch.zeros(N, 2, dtype=torch.bool, device=dev); pend_t = fz(N, 2); pend_load = fz(N, 2); pend_swing = torch.zeros(N, 2, dtype=torch.bool, device=dev)
n_valid = fz(N, 2); last_valid = torch.full((N,), -1, dtype=torch.long, device=dev); alt_pairs, alt_ok = fz(N), fz(N)
vb_s = fz(N, 2); nn_ = fz(N); pit_s, pit_ss, rol_s, rol_ss = fz(N), fz(N), fz(N), fz(N); tilt_max = fz(N)
yaw_acc = fz(N); yaw_prev = None; yaw_un = fz(N); yaw_f = None; vp_s = fz(N, 2); pos_w0 = None; yaw_w0 = None; disp_end = fz(N, 2); hl_err_max = fz(N); hl_err_end = fz(N); wzc_s = fz(N); wzc_ss = fz(N)
t0w = time.time()
for i in range(steps):
    t = (i + 1) * dt
    set_cmd(t)
    with torch.inference_mode():
        act_ = policy(obs)
    obs, _, dones, _ = venv.step(act_)
    if isinstance(obs, tuple):
        obs = obs[0]
    d = dones.bool() if torch.is_tensor(dones) else torch.as_tensor(dones, device=dev).bool()
    tm = getattr(U, "termination_manager", None)
    if tm is not None and hasattr(tm, "terminated"):
        d = d & tm.terminated.bool()
    newly = alive & d; t_fall[newly] = t; alive &= ~d
    ev = t > a.warmup
    q = robot.data.root_quat_w; yaw = yaw_of(q)
    cc = sensor.data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0
    fzv = sensor.data.net_forces_w[:, fid, 2]; fpos = robot.data.body_pos_w[:, bid, :]
    if prev_c is None:
        prev_c = cc.clone()
    m2 = (alive & torch.full((N,), bool(ev), device=dev)).unsqueeze(1).expand(N, 2)
    lifted = prev_c & ~cc & m2
    in_air = in_air | lifted
    air_t = torch.where(lifted, torch.zeros_like(air_t), air_t)
    z0 = torch.where(lifted, fpos[..., 2], z0); zmax = torch.where(lifted, fpos[..., 2], zmax)
    air_now = in_air & ~cc
    air_t = air_t + dt * air_now.float(); zmax = torch.where(air_now, torch.maximum(zmax, fpos[..., 2]), zmax)
    landed = in_air & cc & m2; in_air = in_air & ~cc
    if landed.any():
        sw = landed & (air_t >= 0.06) & ((zmax - z0) >= 0.015)
        pend = pend | landed; pend_t = torch.where(landed, torch.zeros_like(pend_t), pend_t)
        pend_load = torch.where(landed, torch.zeros_like(pend_load), pend_load); pend_swing = torch.where(landed, sw, pend_swing)
    pend = pend & m2
    pend_load = pend_load + dt * (pend & (fzv > 0.5 * MG)).float()
    acc = pend & (pend_load >= 0.1 - 1e-6) & (pend_t <= 0.3 + 1e-6)
    pend_t = pend_t + dt * pend.float(); pend = pend & ~acc & (pend_t <= 0.3 + 1e-6)
    val = acc & pend_swing
    n_valid += val.float()
    if val.any():
        for k in (0, 1):
            vk = val[:, k]; has = vk & (last_valid >= 0)
            alt_pairs += has.float(); alt_ok += (has & (last_valid != k)).float()
            last_valid = torch.where(vk, torch.full_like(last_valid, k), last_valid)
    prev_c = cc.clone()
    if ev:
        af = alive.float()
        if yaw_prev is None:
            yaw_prev = yaw.clone(); pos_w0 = robot.data.root_pos_w[:, :2].clone(); yaw_w0 = yaw.clone()
        _dp = robot.data.root_pos_w[:, :2] - pos_w0; _c0, _s0 = torch.cos(yaw_w0), torch.sin(yaw_w0)
        disp_end = torch.where(alive.unsqueeze(1), torch.stack([_dp[:, 0] * _c0 + _dp[:, 1] * _s0, -_dp[:, 0] * _s0 + _dp[:, 1] * _c0], -1), disp_end)   # 시작 요 프레임 직선 변위
        _dyw = _wrap(yaw - yaw_prev); yaw_acc = torch.where(alive, yaw_acc + _dyw, yaw_acc); yaw_un = yaw_un + _dyw; yaw_prev = yaw.clone()
        yaw_f = yaw_un.clone() if yaw_f is None else yaw_f + (dt / 0.91) * (yaw_un - yaw_f)        # 보행 한 주기(1.1 Hz) 시정수 EMA = 진행 방향
        _vw = robot.data.root_lin_vel_w[:, :2]; _yf = yaw_f + yaw_w0
        vp_s += torch.stack([_vw[:, 0] * torch.cos(_yf) + _vw[:, 1] * torch.sin(_yf), -_vw[:, 0] * torch.sin(_yf) + _vw[:, 1] * torch.cos(_yf)], -1) * af.unsqueeze(1)
        vb_s += robot.data.root_lin_vel_b[:, :2] * af.unsqueeze(1); nn_ += af
        pg = robot.data.projected_gravity_b
        pit = torch.rad2deg(torch.atan2(pg[:, 0], -pg[:, 2])); rol = torch.rad2deg(torch.atan2(-pg[:, 1], -pg[:, 2]))
        pit_s += pit * af; pit_ss += pit * pit * af; rol_s += rol * af; rol_ss += rol * rol * af
        tilt_max = torch.where(alive, torch.maximum(tilt_max, torch.rad2deg(torch.atan2(pg[:, :2].norm(dim=-1), -pg[:, 2]))), tilt_max)
        wz_now = cmd_term.vel_command_b[:, 2]
        wzc_s += wz_now * af; wzc_ss += wz_now * wz_now * af
        if a.heading_loop > 0 and HL["tgt"] is not None:
            e_true = torch.rad2deg(_wrap(HL["tgt"] - PX["yaw_r"] * t - yaw)).abs()    # 참 요 기준 오차(목표 − 참 요; 표류분 제외)
            e_meas = torch.rad2deg(_wrap(HL["tgt"] - (yaw + PX["yaw_r"] * t))).abs()  # 고리가 보는 오차(고장 문턱 15° 판정용)
            hl_err_max = torch.where(alive, torch.maximum(hl_err_max, e_meas), hl_err_max)
            hl_err_end = torch.where(alive, e_true, hl_err_end)
    if (i + 1) % 500 == 0:
        print(f"[env] {t:.1f}/{T_END:.1f} s 생존 {int(alive.sum())}/{N} ({time.time() - t0w:.0f} s)", flush=True)
wall = time.time() - t0w


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n; den = 1 + z * z / n; ce = (ph + z * z / (2 * n)) / den; hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ce - hw), min(1.0, ce + hw))


tf = t_fall.cpu(); al = alive.cpu(); T_SC = a.seconds
RES = []
for g_ in range(G):
    m = (gid == g_).cpu()
    fw = int(((~al) & m & (tf <= a.warmup)).sum()); fs = int(((~al) & m & (tf > a.warmup)).sum()); n_st = int(m.sum()) - fw
    surv = (al & m).to(dev)
    s_ = surv.float(); ns = max(1, int(surv.sum()))
    nn_c = nn_.clamp(min=1)
    vb = vb_s / nn_c.unsqueeze(1)
    cm = GROUPS[g_][1]; cs = GROUPS[g_][2]["cmd_scale"]; cmv = [x * cs for x in cm]
    mean = lambda x: float((x * s_).sum() / ns)
    gx = mean(vb[:, 0]) / cmv[0] if abs(cmv[0]) > 1e-6 else None
    gy = mean(vb[:, 1]) / cmv[1] if abs(cmv[1]) > 1e-6 else None
    yaw_deg = torch.rad2deg(yaw_acc)
    gz = (mean(yaw_acc) / (cmv[2] * T_SC)) if abs(cmv[2]) > 1e-6 else None
    dx_ = mean(disp_end[:, 0]) / T_SC; dy_ = mean(disp_end[:, 1]) / T_SC
    gdx = dx_ / cmv[0] if abs(cmv[0]) > 1e-6 else None; gdy = dy_ / cmv[1] if abs(cmv[1]) > 1e-6 else None
    vp = vp_s / nn_c.unsqueeze(1); gpx = mean(vp[:, 0]) / cmv[0] if abs(cmv[0]) > 1e-6 else None; gpy = mean(vp[:, 1]) / cmv[1] if abs(cmv[1]) > 1e-6 else None
    gains = [x for x in (gpx, gpy, gz) if x is not None]          # 판정 = 진행 방향 프레임(본체 프레임은 흔들림-요 상관 편향, 직선 변위는 요 표류 cos 편향)
    pm = pit_s / nn_c; rm = rol_s / nn_c
    prms = (pit_ss / nn_c - pm ** 2).clamp(min=0).sqrt(); rrms = (rol_ss / nn_c - rm ** 2).clamp(min=0).sqrt()
    yd = yaw_deg[surv].cpu().numpy() if bool(surv.any()) else np.array([float("nan")])
    rate = float((n_valid.sum(1) * s_).sum() / ns / T_SC)
    altr = float((alt_ok * s_).sum() / (alt_pairs * s_).sum().clamp(min=1))
    fr = fs / max(1, n_st); ci = wilson(fs, n_st)
    r = {"g": g_, "cmd": GROUPS[g_][0], "cmd_v": cmv, "ax": {k: GROUPS[g_][2][k] for k in [kk for kk, _ in GRID]}, "n": int(m.sum()), "fall_warm": fw,
         "fall_score": fs, "n_score": n_st, "fall_rate": fr, "fall_ci": ci, "alt": altr, "step_rate": rate,
         "gain_x": gx, "gain_y": gy, "gain_yaw": gz, "gain_disp_x": gdx, "gain_disp_y": gdy, "gain_path_x": gpx, "gain_path_y": gpy, "disp_v": [dx_, dy_], "gain_min": (min(gains) if gains else None),
         "vb": [mean(vb[:, 0]), mean(vb[:, 1])], "yaw_mean_deg": float(np.nanmean(yd)), "yaw_absp50_deg": float(np.nanmedian(np.abs(yd))),
         "yaw_absp95_deg": float(np.nanpercentile(np.abs(yd), 95)),
         "pitch_mean": mean(pm), "roll_mean": mean(rm), "pitch_rms": mean(prms), "roll_rms": mean(rrms),
         "tilt_max_p95": float(np.nanpercentile(tilt_max[surv].cpu().numpy(), 95)) if bool(surv.any()) else float("nan"),
         "wz_cmd_mean": mean(wzc_s / nn_c), "wz_cmd_rms": mean((wzc_ss / nn_c).sqrt())}
    if a.heading_loop > 0 and abs(cmv[2]) < 1e-6:
        hm = hl_err_max[surv].cpu().numpy() if bool(surv.any()) else np.array([float("nan")])
        he = hl_err_end[surv].cpu().numpy() if bool(surv.any()) else np.array([float("nan")])
        r.update({"hl_err_end_p50": float(np.nanmedian(he)), "hl_err_end_p95": float(np.nanpercentile(he, 95)),
                  "hl_err_max_p95": float(np.nanpercentile(hm, 95)), "hl_fault15_frac": float((hm > 15.0).mean())})
    r["collapse"] = bool(fr > 0.03 or (r["gain_min"] is not None and r["gain_min"] < 0.6))
    RES.append(r)

os.makedirs(os.path.join(REPO, "log_picture"), exist_ok=True)
base = os.path.join(REPO, "log_picture", f"w3env_{a.out_tag}")
J = {"task": a.task, "checkpoint": ck, "seed": a.seed, "per_group": a.per_group, "num_envs": N, "warmup": a.warmup, "seconds": a.seconds,
     "steps_done": steps, "completed": True, "base_delay": a.base_delay, "heading_loop": a.heading_loop, "ramp_on": (not a.ramp_off), "sweep": a.sweep, "grid": GRID, "fix": FIX, "cmds": CMDS,
     "events": EV_DESC, "wrapped_obs": OBS_DESC, "actuators": GAIN_LOG, "wall_s": wall, "groups": RES}
json.dump(J, open(base + ".json", "w"), ensure_ascii=False, indent=1)
L = [f"# w3env {a.out_tag} — {a.task} {os.path.basename(os.path.dirname(ck))}/{os.path.basename(ck)} seed {a.seed} · {a.per_group} env/그룹 · 워밍업 {a.warmup} s + 채점 {a.seconds} s · 기본 지연 {a.base_delay} · 고리 {a.heading_loop} · 롤 램프 {'끔' if a.ramp_off else '켬'} · 고정 {FIX}",
     "| 명령 | " + " | ".join(k for k, _ in GRID) + " | 과도 낙상 | 채점 낙상 (95 % CI) | 교대 | 스텝/s | 이득 진행 x/y/요 · 본체 x/y · 직선 x/y | 요 누적 평균·|p95| ° | 피치/롤 평균 | 피치/롤 RMS | 붕괴 |",
     "|---|" + "---|" * len(GRID) + "---|---|---|---|---|---|---|---|---|"]
fmt = lambda x: "—" if x is None else f"{x:.2f}"
for r in RES:
    L.append(f"| {r['cmd']} | " + " | ".join(f"{r['ax'][k]:g}" for k, _ in GRID) + f" | {r['fall_warm']} | {r['fall_score']}/{r['n_score']} {100 * r['fall_rate']:.1f} % ({100 * r['fall_ci'][0]:.1f}–{100 * r['fall_ci'][1]:.1f}) | {r['alt']:.3f} | {r['step_rate']:.2f} | "
             f"{fmt(r['gain_path_x'])}/{fmt(r['gain_path_y'])}/{fmt(r['gain_yaw'])} · {fmt(r['gain_x'])}/{fmt(r['gain_y'])} · {fmt(r['gain_disp_x'])}/{fmt(r['gain_disp_y'])} | {r['yaw_mean_deg']:+.1f}·{r['yaw_absp95_deg']:.1f} | {r['pitch_mean']:+.1f}/{r['roll_mean']:+.1f} | {r['pitch_rms']:.2f}/{r['roll_rms']:.2f} | "
             + ("**붕괴**" if r["collapse"] else "") + (f" 고리 끝오차 p95 {r['hl_err_end_p95']:.1f}° 15°초과 {100 * r['hl_fault15_frac']:.0f} %" if "hl_err_end_p95" in r else "") + " |")
open(base + ".txt", "w").write("\n".join(L) + "\n")
print("\n".join(L), flush=True)
print(f"[env] 완료 {wall:.0f} s → {base}.json", flush=True)
env.close(); app.close()
