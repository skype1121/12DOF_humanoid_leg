"""병목 2(회복 종료·정지 판정) 기록기 — eval_ground_recovery.py 를 **수정하지 않고** 그대로 실행하면서 틱별 상태를 npz 로 남긴다.

방식: 평가기 소스를 읽어 (1) `env.step(a)` 바로 뒤에 기록 훅, (2) `env.close()` 바로 앞에 저장 훅을 문자열로 끼운 사본을
같은 프로세스에서 exec 한다(원본 파일 불변, 치환 지점이 정확히 1개씩인지 assert). 평가 인자는 그대로 넘기고 `--rec_out` 만 이 스크립트가 쓴다.

기록(틱별, 인계 전 홀드 구간 포함):
  참값  : 골반 pos/quat/선속도(월드)/각속도(몸통), CoM pos/vel(질량 가중), 발 링크 pos/quat, 발 접촉 힘 크기
  배포 신호: 관절 링크각 q, 엔코더각(mdp_hang.encoder_joint_pos_v2 = 링크 + 유격 클램프, 로터측), env 별 기본자세 편향,
           정책 관측 최신 프레임의 자이로·중력(지연·잡음 포함)·접촉 0/1(v2 계약 = 정책이 실제로 받은 값)
  상태  : 회복 모드(sw_mode, KE 전환 또는 c2 ge), 종료 플래그
"""
import os, sys, re

SRC = "/home/ryu/humanoid_leg_test1/.claude/worktrees/agent-ab24ab30531b0f378/rl_walking/scripts/eval_ground_recovery.py"
if os.environ.get("STOPEST_SRC"):
    SRC = os.environ["STOPEST_SRC"]
argv = sys.argv[1:]
k = argv.index("--rec_out"); REC_OUT = argv[k + 1]; del argv[k:k + 2]
sys.argv = [SRC] + argv

src = open(SRC, encoding="utf-8").read()
A = "        obs, _, dones, _ = env.step(a)\n"
B = "env.close()\n"
assert src.count(A) == 1, "env.step 치환 지점이 1개가 아님"
assert src.count(B) == 1, "env.close 치환 지점이 1개가 아님"
src = src.replace(A, A + "        _REC_HOOK(globals(), i)\n").replace(B, "_REC_SAVE(globals())\n" + B)

_R = {}


def _REC_HOOK(G, i):
    import torch
    robot = G["robot"]; uenv = G["uenv"]; obs = G["obs"]; d = robot.data
    if "off" not in _R:
        from rl_walking import mdp_hang
        _R["mh"] = mdp_hang
        nm = list(uenv.observation_manager.active_terms["policy"]); dm = uenv.observation_manager.group_obs_term_dim["policy"]
        off = {}; o = 0
        for n, dd in zip(nm, dm):
            s = 1
            for v in (dd if isinstance(dd, (tuple, list)) else (dd,)): s *= int(v)
            off[n] = (o, o + s); o += s
        _R["off"] = off
        _R["mass"] = robot.root_physx_view.get_masses().to(d.root_pos_w.device)      # (N,B)
        _R["buf"] = {}
        cs = uenv.scene["contact_forces"]; cbn = list(cs.body_names)
        _R["cid"] = [next(j for j, n in enumerate(cbn) if n.startswith(s)) for s in ("left_ankle_r", "right_ankle_r")]
        bn = list(d.body_names); _R["fid"] = [bn.index("left_ankle_r_joint"), bn.index("right_ankle_r_joint")]
        print("[stopest_rec] obs 배치", {k_: off[k_] for k_ in ("base_ang_vel", "projected_gravity", "feet_contact") if k_ in off}, flush=True)
    off = _R["off"]; po = obs["policy"]
    m = _R["mass"]; M = m.sum(1, keepdim=True)
    com_p = (d.body_com_pos_w * m.unsqueeze(-1)).sum(1) / M
    com_v = (d.body_com_lin_vel_w * m.unsqueeze(-1)).sum(1) / M
    cs = uenv.scene["contact_forces"]
    fn = cs.data.net_forces_w[:, _R["cid"], :].norm(dim=-1)
    enc = _R["mh"].encoder_joint_pos_v2(uenv)
    pb = getattr(uenv, "_default_pose_bias", None)
    mode = G.get("sw_mode", None)
    if mode is None:
        mode = torch.zeros(enc.shape[0], dtype=torch.bool, device=enc.device)
    rec = {
        "pos": d.root_pos_w, "quat": d.root_quat_w, "lv": d.root_lin_vel_w, "av": d.root_ang_vel_b, "pg": d.projected_gravity_b,
        "com": com_p, "comv": com_v, "fpos": d.body_pos_w[:, _R["fid"], :], "fquat": d.body_quat_w[:, _R["fid"], :], "fn": fn,
        "q": d.joint_pos, "enc": enc, "pbias": (pb if pb is not None else torch.zeros_like(enc)),
        "o_gyro": po[:, off["base_ang_vel"][1] - 3:off["base_ang_vel"][1]],
        "o_pg": po[:, off["projected_gravity"][1] - 3:off["projected_gravity"][1]],
        "o_cf": po[:, off["feet_contact"][1] - 2:off["feet_contact"][1]],
        "mode": mode, "term": uenv.termination_manager.terminated,
    }
    for k_, v in rec.items():
        _R["buf"].setdefault(k_, []).append(v.detach().clone().cpu())


def _REC_SAVE(G):
    import numpy as np, torch, json
    out = {}
    for k_, v in _R["buf"].items():
        x = torch.stack(v, 0)
        out[k_] = x.numpy() if x.dtype == torch.bool else x.float().numpy()
    robot = G["robot"]; a = G["args_cli"]
    meta = {"argv": sys.argv[1:], "jn": list(robot.data.joint_names), "steps_pre": int(G["STEPS_PRE"]), "dt": 0.02,
            "dists": [(d_.kind, d_.dir, d_.t0, d_.F, d_.t_end) for d_ in G["dists"]], "ckpt": str(G.get("resume_path")),
            "switch_to": getattr(a, "switch_to", ""), "c2": bool(getattr(a, "c2", False)), "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
            "mass_total": float(_R["mass"].sum(1).mean())}
    out["pre_pass"] = G["pre_pass"].cpu().numpy() if "pre_pass" in G else np.ones(out["q"].shape[1], bool)
    out["default_joint_pos"] = robot.data.default_joint_pos.cpu().numpy()
    np.savez_compressed(REC_OUT, meta=json.dumps(meta, ensure_ascii=False), **out)
    print("[stopest_rec] 저장", REC_OUT, {k_: v.shape for k_, v in out.items() if hasattr(v, "shape")}, flush=True)


G = {"__name__": "__main__", "__file__": SRC, "_REC_HOOK": _REC_HOOK, "_REC_SAVE": _REC_SAVE, "__builtins__": __builtins__}
sys.path.insert(0, os.path.dirname(SRC))
exec(compile(src, SRC, "exec"), G)
