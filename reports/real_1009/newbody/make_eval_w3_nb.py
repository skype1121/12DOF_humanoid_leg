#!/usr/bin/env python3
"""eval_w3.py 사본 eval_w3_nb.py 를 만든다(1010, 새 몸 걷기 평가). 원본은 고치지 않는다. 앵커가 안 맞으면 assert.
더하는 것: (1) RL_WALKING_EXTRA_TASKS 모듈 읽기(새 몸 태스크 등록) (2) 토크 포화 틱 비율(전 관절 / 굳은 관절 hip_r 뺀 것, 관절별),
굳은 관절 변위·명령각, 전체 질량 → JSON 'nb' 와 txt 끝 줄. 채점 로직·기존 출력은 그대로."""
import os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src = open(os.path.join(R, "rl_walking/scripts/eval_w3.py"), encoding="utf-8").read()


def rep(s, a, b):
    assert s.count(a) == 1, (s.count(a), a)
    return s.replace(a, b)


A = "import rl_walking  # noqa: F401,E402\n"
src = rep(src, A, A + """import importlib as _nb_il  # 1010 새 몸: 학습 전용 추가 태스크 등록
for _nb_m in [x for x in os.environ.get("RL_WALKING_EXTRA_TASKS", "").split(",") if x.strip()]:
    _nb_il.import_module(_nb_m.strip())
""")
A = "alive = torch.ones(N, dtype=torch.bool, device=dev); t_fall = torch.full((N,), float(\"nan\"), device=dev)\n"
src = rep(src, A, A + """# 1010 새 몸 계측
_nb_jn = list(robot.data.joint_names); _nb_lock = [_nb_jn.index(n) for n in ("left_hip_r_joint", "right_hip_r_joint")]
_nb_oth = [j for j in range(len(_nb_jn)) if j not in _nb_lock]; _nb_st = getattr(U, "_nb_lock", None)
_nb_tgt = _nb_st["tgt"] if _nb_st is not None else torch.zeros(N, 2, device=dev)
_nb = {"n": 0.0, "sat_any": 0.0, "sat_oth": 0.0, "sat_j": torch.zeros(len(_nb_jn), device=dev), "dev_max": 0.0, "hipr_abs": [], "tau_lock": 0.0}
""")
A = "    obs, _, dones, _ = venv.step(act_)\n"
src = rep(src, A, A + """    if bool(alive.any()):      # 1010 새 몸 계측(생존 env·전 구간)
        _nbq = robot.data.applied_torque.abs()[alive]; _nbs = _nbq >= 0.95 * robot.data.joint_effort_limits[alive]
        _nb["n"] += float(alive.sum()); _nb["sat_any"] += float(_nbs.any(1).sum()); _nb["sat_oth"] += float(_nbs[:, _nb_oth].any(1).sum()); _nb["sat_j"] += _nbs.float().sum(0)
        _nbd = (robot.data.joint_pos[:, _nb_lock] - _nb_tgt).abs()[alive & ~dones.bool()]
        if _nbd.numel():
            _nb["dev_max"] = max(_nb["dev_max"], float(torch.rad2deg(_nbd.max())))
            _nb["hipr_abs"].append(torch.rad2deg(_nbd.amax(1)).float().cpu())
        _nb["tau_lock"] += float(_nbq[:, _nb_lock].mean(1).sum())
""")
A = "J[\"fall_env_ids\"] = [int(e) for e in torch.nonzero(~alive).flatten().cpu().tolist()]   # 1007: 짝 비교(같은 seed A/B)용\n"
src = rep(src, A, A + """_nbn = max(_nb["n"], 1.0); _nbh = torch.cat(_nb["hipr_abs"]) if _nb["hipr_abs"] else torch.zeros(1)
_nbm = robot.root_physx_view.get_masses().sum(-1)
J["nb"] = {"locked": _nb_st is not None, "sat_any_pct": 100 * _nb["sat_any"] / _nbn, "sat_excl_hipr_pct": 100 * _nb["sat_oth"] / _nbn,
           "sat_joint_pct": {n: round(100 * float(v) / _nbn, 3) for n, v in zip(_nb_jn, _nb["sat_j"])},
           "hipr_dev_max_deg": _nb["dev_max"], "hipr_abs_from_lock_p50_p99_deg": [float(_nbh.median()), float(_nbh.quantile(0.99))],
           "hipr_tau_mean_nm": _nb["tau_lock"] / _nbn, "mass_mean_kg": float(_nbm.mean()), "mass_min_max_kg": [float(_nbm.min()), float(_nbm.max())]}
_nbl = (f"[새 몸 계측] 고정 {'있음' if _nb_st is not None else '없음'} · 전체 질량 {J['nb']['mass_mean_kg']:.4f} kg · 토크 포화 틱(생존 env, ≥95 %): 전 관절 {J['nb']['sat_any_pct']:.2f} % · hip_r 뺀 {J['nb']['sat_excl_hipr_pct']:.2f} % · "
        f"hip_r |각 − 고정각| 최대 {_nb['dev_max']:.3f}° 중앙 {J['nb']['hipr_abs_from_lock_p50_p99_deg'][0]:.3f}° 99 % {J['nb']['hipr_abs_from_lock_p50_p99_deg'][1]:.3f}° · hip_r |토크| 평균 {J['nb']['hipr_tau_mean_nm']:.2f} N·m · 관절별 포화 % {J['nb']['sat_joint_pct']}")
print(_nbl, flush=True); open(os.path.join(out_dir, f"w3eval_{a.out_tag}.txt"), "a").write(_nbl + "\\n")
""")
dst = os.path.join(R, "rl_walking/scripts/eval_w3_nb.py")
open(dst, "w", encoding="utf-8").write(src); print("wrote", dst)
