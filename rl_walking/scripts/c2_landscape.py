"""C2UNIT (1006) — c2 반사실 보상 단위시험 한 팔(행동 궤적) 실행: 학습 env(W3C2, 외란·DR·지연 분포 그대로) 에서 행동 팔 하나로 굴리고,
같은 궤적을 보상 조건 R0(W3S1c)·R1(c2)·R1a(c2 − 포획 착지)·R1b(c2 − 정착)으로 채점한다(reports/gen_w3/C2_DESIGN.md §4).

행동 팔: B1 STAND(C2_ARM=stand) · B2 NOSTOP · B3 C2 · B4 EARLY · B5 LATE · B6 CUE(B3 + 회복 중 actor 관측 velocity_commands vx ±0.06, 외란 범주 방향 — 보상·env 명령은 0).
채점: 보상 매니저의 항별 초당 값(raw×weight) 중 w3_place·c2_settle 를 뺀 나머지(공통) + 이 스크립트가 같은 틱에 계산한
  명목 착지(W3 식) / 포획 착지(c2 식) / 정착 사건 — 세 흐름은 같은 시점(이벤트 뒤)에 계산되어 서로 정렬된다(보상 매니저 흐름과는 1 틱 어긋남, 합계 비교엔 무관).
창: env 별 첫 외란 시작 t0 ~ t0 + 8 s(무외란 범주도 표본 t0 사용), 첫 에피소드만, 창 전 낙상 env 는 제외 표시.
출력: log_picture/c2unit_<tag>.npz(+ .txt 요약). 판정은 reports/gen_w3/c2_unit_judge.py 가 B1–B6 을 모아 한다.
사용: isaaclab.sh -p rl_walking/scripts/c2_landscape.py --headless --load_run <run> --checkpoint model_N.pt --arm B3 --seed 1 --out_tag <tag>"""
import argparse, os, sys, math
from isaaclab.app import AppLauncher
sys.path.insert(0, "/home/ryu/IsaacLab/scripts/reinforcement_learning/rsl_rl")
import cli_args  # noqa: E402
CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.environ.get("EVAL_DATA_REPO", CODE)
sys.path.insert(0, CODE)
p = argparse.ArgumentParser()
p.add_argument("--task", default="Biped12-HangStand-W3C2-v0")
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--arm", required=True, choices=("B1", "B2", "B3", "B4", "B5", "B6", "X"))
p.add_argument("--c2_params", default="", help="arm X: C2_PARAMS(예 stop_td_k=2 / stop_oracle=1) — 정지 시점 스윕")
p.add_argument("--cue", action="store_true", help="arm X: 회복 중 B6 형 관측 cue")
p.add_argument("--window", type=float, default=8.0)
p.add_argument("--seconds", type=float, default=19.0, help="시뮬 길이(첫 외란 t0 ≤ 9 s + 창 8 s 를 덮게)")
p.add_argument("--cue_vx", type=float, default=0.06)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--out_tag", default="c2unit")
cli_args.add_rsl_rl_args(p)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
# 1006 밤: B2/B5 는 stop_td_k=0 을 명시(고정 k 설정에서도 의미 유지), B4 = 첫 회복 착지에서 정지(stop_td_k 1 — 종전 정의와 같은 의미, 강제·취소 없음)
ARM_ENV = {"B1": ("stand", ""), "B2": ("", "stop_td_k=0,stop_oracle=0,min_td=1e6,tmax_s=1e9"), "B3": ("", ""), "B4": ("", "stop_td_k=1"),
           "B5": ("", "stop_td_k=0,stop_oracle=0,min_td=1e6"), "B6": ("", ""), "X": ("", a.c2_params)}[a.arm]
CUE = a.arm == "B6" or (a.arm == "X" and a.cue)
for k in ("C2_ARM", "C2_PARAMS"):
    os.environ.pop(k, None)
if ARM_ENV[0]:
    os.environ["C2_ARM"] = ARM_ENV[0]
if ARM_ENV[1]:
    os.environ["C2_PARAMS"] = ARM_ENV[1]
os.environ["STEP_ASSIST_ALPHA"] = "0"          # 커리큘럼을 끄므로 보조 α 를 명시 0(W3S1c 학습 끝 상태와 같음)
os.environ.setdefault("TURN_ALPHA", "0")
app = AppLauncher(a).app

import numpy as np, torch, gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import rl_walking  # noqa: F401,E402
from rl_walking import w3 as W3, w3_c2 as C, mdp_hang as H  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

env_cfg = load_cfg_from_registry(a.task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(a.task, "rsl_rl_cfg_entry_point")
agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, a)
env_cfg.scene.num_envs = a.num_envs; env_cfg.seed = a.seed
env_cfg.curriculum = None
env_cfg.episode_length_s = a.seconds + 2.0
env = gym.make(a.task, cfg=env_cfg)
venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
U = env.unwrapped; dev = U.device; N = U.num_envs; dt = U.step_dt
ck = get_checkpoint_path(os.path.join(DATA, "logs", "rsl_rl", agent_cfg.experiment_name), agent_cfg.load_run, agent_cfg.load_checkpoint)
runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(ck); policy = runner.get_inference_policy(device=dev)
RM = U.reward_manager; names = list(RM._term_names)
i_place = names.index("w3_place"); i_settle = names.index("c2_settle")
other_mask = torch.ones(len(names), dtype=torch.bool, device=dev); other_mask[i_place] = False; other_mask[i_settle] = False
W_PLACE = float(RM._term_cfgs[i_place].weight); W_SETTLE = float(RM._term_cfgs[i_settle].weight)
_pp = RM._term_cfgs[i_place].params; SIG, SIG_CAP = float(_pp.get("sig", 0.04)), float(_pp.get("sig_cap", 0.05)); CAP_MODE = str(_pp.get("cap_mode", "err"))
# velocity_commands 위치(B6)
_nm = list(U.observation_manager.active_terms["policy"]); _dm = U.observation_manager.group_obs_term_dim["policy"]
_o = 0; LAY = {}
for n_, d_ in zip(_nm, _dm):
    sz = 1
    for v_ in (d_ if isinstance(d_, (tuple, list)) else (d_,)):
        sz *= int(v_)
    LAY[n_] = (_o, _o + sz); _o += sz
VC = LAY["velocity_commands"]
SB_TERMS = {}
for _nm in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp", "survive_after_release"):
    _cf = RM._term_cfgs[names.index(_nm)]
    SB_TERMS[_nm] = (_cf.params["func"], dict(_cf.params.get("params") or {}), float(_cf.weight))
print(f"[c2unit] {a.task} · {ck} · 팔 {a.arm} {ARM_ENV} · env {N} · seed {a.seed} · 보상 항 {len(names)} · 관측 {_o} · velocity_commands {VC}", flush=True)

z = lambda *s: torch.zeros(*s, device=dev)
first_ep = torch.ones(N, dtype=torch.bool, device=dev)       # 첫 에피소드 진행 중
fell = torch.zeros(N, dtype=torch.bool, device=dev); t_fall = torch.full((N,), float("nan"), device=dev)
S = {k: z(N) for k in ("other_w", "nom_w", "cap_w", "capm_w", "capd_w", "standbase_w", "settle_w", "nom_all", "cap_all", "settle_all", "rm_place_all",
                       "n_rec_td", "cap_raw_rec", "nom_raw_rec", "mg_raw_rec", "df_raw_rec", "n_td_req", "n_td_w", "n_valid_w", "trig_w", "req_w", "timeout_w", "cancel_w", "latch_w",
                       "settle_ev_w", "trig_all", "rec_ticks_w", "fell_w", "fell_pre", "drift_w", "to_after_req", "req_first_t", "latch_first_t")}
S["req_first_t"][:] = float("nan"); S["latch_first_t"][:] = float("nan")
xy0 = torch.full((N, 2), float("nan"), device=dev); xy_req = torch.full((N, 2), float("nan"), device=dev); xy_last = z(N, 2)
cat = None; kind0 = None; t0 = None; kind1 = None; t01 = None; T_STEPS = int(round(a.seconds / dt))

obs = venv.get_observations()
for i in range(T_STEPS):
    if isinstance(obs, torch.Tensor):
        po = obs
    else:
        po = obs["policy"]
    if CUE and getattr(U, "_c2", None) is not None:
        g = H._g22(U)["cat"]; md = U._c2["mode"]
        on = ((md == C.REC) | (md == C.STOPREQ)) & (g > 0.5)
        vx = torch.where(g > 1.5, -a.cue_vx, a.cue_vx)
        po = po.clone()
        cols = po[:, VC[0]:VC[1]:3]
        po[:, VC[0]:VC[1]:3] = torch.where(on.unsqueeze(1), vx.unsqueeze(1).expand_as(cols), cols)
        if not isinstance(obs, torch.Tensor):
            obs = obs.clone(); obs["policy"] = po
        else:
            obs = po
    with torch.inference_mode():
        act = policy(obs)
    obs, _, dones, extras = venv.step(act)
    with torch.inference_mode():
        if cat is None:          # 첫 리셋 뒤 범주·스케줄 고정(첫 에피소드)
            cat = H._g22(U)["cat"].clone(); hs = H._state(U); kind0 = hs["dst_kind"][:, 0].clone(); t0 = hs["dst_t0"][:, 0].clone(); kind1 = hs["dst_kind"][:, 1].clone(); t01 = hs["dst_t0"][:, 1].clone()
        t_ep = U.episode_length_buf.float() * dt
        done = dones.bool(); to = extras.get("time_outs", torch.zeros_like(done)).bool() if isinstance(extras, dict) else torch.zeros_like(done)
        live = first_ep.clone()
        t_cur = (i + 1) * dt                                         # 첫 에피소드 시간(리셋 전)
        inw = live & (t_cur >= t0) & (t_cur < t0 + a.window)
        sr = RM._step_reward
        oth = (sr * other_mask.float()).sum(1)
        s3 = W3._st(U); st = C._c2(U); ge = W3._ge(U)
        nom = W3.w3_place_reward(U, sig=SIG) * W_PLACE
        cap = C.c2_place_reward(U, sig=SIG, sig_cap=SIG_CAP, cap_mode=CAP_MODE) * W_PLACE
        stl = st["ev_settle"] * W_SETTLE
        capm = C.c2_place_reward(U, sig=SIG, sig_cap=SIG_CAP, cap_mode="margin") * W_PLACE     # 여유형 흐름(설정과 무관하게 함께 채점)
        capd = C.c2_place_reward(U, sig=SIG, sig_cap=SIG_CAP, cap_mode="diff") * W_PLACE       # 차분형 흐름
        # R1c(숨기 유인 해소 반사실): 회복 ge 1 동안에도 서기 기본 보상(상태 읽기만 하는 항 — 명령 추종 2개·생존)을 유지했다면
        sb = torch.zeros(N, device=dev)
        rec_on = W3._ge(U) & (H._g22(U)["cat"] > 0.5)
        for _nm, (_f, _p, _w) in SB_TERMS.items():
            sb = sb + _f(U, **H._resolve_nested(U, _p)) * _w
        sb = sb * rec_on.float()
        m = inw.float()
        S["other_w"] += oth * m * dt; S["nom_w"] += nom * m * dt; S["cap_w"] += cap * m * dt; S["capm_w"] += capm * m * dt; S["capd_w"] += capd * m * dt; S["standbase_w"] += sb * m * dt; S["settle_w"] += stl * m * dt
        lv = live.float()
        S["nom_all"] += nom * lv * dt; S["cap_all"] += cap * lv * dt; S["settle_all"] += stl * lv * dt; S["rm_place_all"] += sr[:, i_place] * lv * dt
        S["trig_all"] += st["ev_trig"] * lv
        tdr = st["td_rec"] & (s3["ev_td"] > 0.5)
        S["n_rec_td"] += tdr.float().sum(1) * m
        S["cap_raw_rec"] += (torch.exp(-(st["cap_err"] / SIG_CAP) ** 2) * tdr.float()).sum(1) * m
        S["nom_raw_rec"] += (torch.exp(-(s3["td_err"] / SIG) ** 2) * tdr.float()).sum(1) * m
        S["mg_raw_rec"] += (st["cap_mg"] * tdr.float()).sum(1) * m              # 여유형 점수(설정과 무관하게 항상 기록)
        S["df_raw_rec"] += (st["cap_df"] * tdr.float()).sum(1) * m              # 차분형 점수
        S["n_td_w"] += s3["ev_td"].sum(1) * m; S["n_valid_w"] += s3["ev_valid"].sum(1) * m
        for k_, e_ in (("trig_w", "ev_trig"), ("req_w", "ev_req"), ("timeout_w", "ev_timeout"), ("cancel_w", "ev_cancel"), ("latch_w", "ev_latch"), ("settle_ev_w", "ev_settle")):
            S[k_] += st[e_] * m
        S["rec_ticks_w"] += ((st["mode"] == C.REC) | (st["mode"] == C.STOPREQ)).float() * m
        rq = ((st["ev_req"] + st["ev_timeout"]) > 0.5) & inw & torch.isnan(S["req_first_t"])
        S["req_first_t"] = torch.where(rq, t_cur - t0, S["req_first_t"])
        S["n_td_req"] = torch.where(rq, st["n_td"], S["n_td_req"])               # 첫 정지 요청 때 회복 착지 수(스윕 k 확인)
        lt = (st["ev_latch"] > 0.5) & inw & torch.isnan(S["latch_first_t"])
        S["latch_first_t"] = torch.where(lt, t_cur - t0, S["latch_first_t"])
        S["to_after_req"] += s3["ev_takeoff"].sum(1) * (inw & ~torch.isnan(S["req_first_t"])).float()
        xy = U.scene["robot"].data.root_pos_w[:, :2]
        xy0 = torch.where((inw & torch.isnan(xy0[:, 0])).unsqueeze(1), xy, xy0)
        xy_req = torch.where((rq).unsqueeze(1), xy, xy_req)
        xy_last = torch.where((inw & ~done).unsqueeze(1), xy, xy_last)
        f_ = done & ~to & live
        S["fell_w"] += (f_ & inw).float(); S["fell_pre"] += (f_ & (t_cur < t0)).float()
        t_fall = torch.where(f_, torch.full_like(t_fall, t_cur), t_fall)
        first_ep = first_ep & ~done
S["drift_w"] = torch.where(torch.isnan(xy0[:, 0]), torch.full_like(S["drift_w"], float("nan")), 100.0 * (xy_last - xy0).norm(dim=-1))
S["drift_after_req"] = torch.where(torch.isnan(xy_req[:, 0]), torch.full((N,), float("nan"), device=dev), 100.0 * (xy_last - xy_req).norm(dim=-1))
out = {k: v.cpu().numpy() for k, v in S.items()}
out.update(cat=cat.cpu().numpy(), kind0=kind0.cpu().numpy(), t0=t0.cpu().numpy(), kind1=kind1.cpu().numpy(), t01=t01.cpu().numpy(), t_fall=t_fall.cpu().numpy(), arm=a.arm, c2_params=a.c2_params, cue=CUE, seed=a.seed, ckpt=str(ck),
           w_place=W_PLACE, w_settle=W_SETTLE, cap_mode=CAP_MODE, window=a.window)
os.makedirs(os.path.join(DATA, "log_picture"), exist_ok=True)
np.savez_compressed(os.path.join(DATA, "log_picture", f"c2unit_{a.out_tag}.npz"), **out)
L = [f"C2UNIT 팔 {a.arm} · {ck} · seed {a.seed} · env {N}"]
cn = out["cat"]; valid = out["fell_pre"] < 0.5
for cv, lab in ((0, "무외란"), (1, "앞"), (2, "뒤")):
    mk = (cn == cv) & valid
    if mk.sum() == 0:
        continue
    r0 = out["other_w"] + out["nom_w"]; r1 = out["other_w"] + out["cap_w"] + out["settle_w"]
    L.append(f"[{lab}] n {int(mk.sum())} · 창 낙상 {100*out['fell_w'][mk].mean():.1f} % · R0 {r0[mk].mean():+.3f} · R1 {r1[mk].mean():+.3f} · 켜짐 {out['trig_w'][mk].mean():.2f}"
             f" · 정지 요청 {out['req_w'][mk].mean():.2f}(시간 초과 {out['timeout_w'][mk].mean():.2f}) · 래치 {out['latch_w'][mk].mean():.2f} · 정착 {out['settle_ev_w'][mk].mean():.2f}"
             f" · 회복 착지 {out['n_rec_td'][mk].mean():.2f}(포획 raw/착지 {out['cap_raw_rec'][mk].sum()/max(out['n_rec_td'][mk].sum(),1):.3f}, 명목 raw {out['nom_raw_rec'][mk].sum()/max(out['n_rec_td'][mk].sum(),1):.3f})"
             f" · 표류 {np.nanmedian(out['drift_w'][mk]):.1f} cm")
L.append(f"창 전 낙상(스폰 과도, 제외) {int((~valid).sum())} · 보상 매니저 착지 합 vs 계산 포획 착지 합(첫 에피소드) {out['rm_place_all'].sum():.3f} vs {out['cap_all'].sum():.3f}")
txt = "\n".join(L); print(txt, flush=True)
open(os.path.join(DATA, "log_picture", f"c2unit_{a.out_tag}.txt"), "w").write(txt + "\n")
env.close(); app.close()
