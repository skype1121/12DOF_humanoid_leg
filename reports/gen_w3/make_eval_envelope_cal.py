"""eval_w3_envelope.py → eval_w3_envelope_cal.py (1007 claude2, ROBUST_DESIGN.md §5 CAL): 정지 보정 개념 시험용 사본.
추가: --stand_first S(처음 S 초 STAND) · 발목 앞뒤 토크 합 S_tq = τ_L·sL + τ_R·sR(잡음 σ) 기록 · --cop_trim(서 있는 동안 S_tq → 기준값으로 가도록 IMU 피치 **관측** 오프셋을 적분).
원본은 수정하지 않는다. 기본(--stand_first 0, --cop_trim 없음)은 원본과 같은 경로."""
import os
SRC = "rl_walking/scripts/eval_w3_envelope.py"; DST = os.environ.get("CAL_DST", "rl_walking/scripts/eval_w3_envelope_cal.py")
src = open(SRC, encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    n = src.count(old); assert n == count, f"앵커 {n}회: {old[:90]!r}"
    src = src.replace(old, new)

rep('p.add_argument("--out_tag", default="env")', '''p.add_argument("--out_tag", default="env")
p.add_argument("--stand_first", type=float, default=0.0, help="CAL: 처음 이 시간[s] 동안 STAND(ge 0·명령 0), 그 뒤 걷기. 채점은 stand_first + 2 s 뒤부터")
p.add_argument("--cop_trim", default="", help="CAL: 'k,S_ref,max_deg[,t0[,sL,sR]]' — STAND 구간 t ≥ t0(기본 1.5 s)부터 조용한 양발 서기일 때 trim[°] += k·(S_f − S_ref)·dt "
               "(S_f = 발목 앞뒤 토크 합의 저역 τ 0.3 s, k [°/(N·m·s)] 부호 포함), |trim| ≤ max_deg. trim 은 정책이 받는 중력 벡터의 피치 회전(IMU 편향과 같은 경로). 빈 값 = 끔")
p.add_argument("--cop_noise", type=float, default=0.3, help="CAL: 토크 추정 잡음 σ [N·m](관절·틱별)")
p.add_argument("--cop_sign", default="1,-1", help="CAL: S_tq = sL·τ_L + sR·τ_R")''')
rep('a = p.parse_args(); a.headless = True', '''a = p.parse_args(); a.headless = True
if a.stand_first > 0:
    a.warmup = max(a.warmup, a.stand_first + 2.0)
CT = None
if a.cop_trim.strip():
    _v = [float(x) for x in a.cop_trim.split(",")]
    CT = {"k": _v[0], "ref": _v[1], "max": _v[2], "t0": (_v[3] if len(_v) > 3 else 1.5)}
    assert a.stand_first > CT["t0"] + 0.5, "--cop_trim 은 --stand_first > t0 + 0.5 필요"
_SL, _SR = (float(x) for x in a.cop_sign.split(","))''')
rep('    g = _rot_g(out, PX["imu_th"], PX["imu_ph"])', '    g = _rot_g(out, PX["imu_th"] + PX["trim"], PX["imu_ph"])')
rep('PX["ready"] = True', '''PX["trim"] = torch.zeros(N, device=dev)          # CAL: IMU 피치 관측 트림 [rad]
_IAL = JN.index("left_ankle_f_joint"); _IAR = JN.index("right_ankle_f_joint")
CAL = {k: torch.zeros(N, device=dev) for k in ("S_pre", "n_pre", "S_post", "n_post", "p_pre", "p_post", "tl_pre", "tr_pre", "S_f")}
PX["ready"] = True''')
rep('''    cmd_term.vel_command_b[:] = CMD3
    if a.heading_loop > 0:''', '''    cmd_term.vel_command_b[:] = CMD3 if t > a.stand_first else torch.zeros_like(CMD3)
    if a.heading_loop > 0:''')
rep('''    set_cmd(t)
    with torch.inference_mode():
        act_ = policy(obs)''', '''    set_cmd(t)
    if a.stand_first > 0 and getattr(U, "_ge_cmd", None) is not None:
        U._ge_cmd[:] = bool(t > a.stand_first)
    with torch.inference_mode():
        act_ = policy(obs)''')
rep('    newly = alive & d; t_fall[newly] = t; alive &= ~d', '''    newly = alive & d; t_fall[newly] = t; alive &= ~d
    if a.stand_first > 0 and t <= a.stand_first:          # ── CAL: 정지 구간 계측·트림 ──
        with torch.inference_mode():
            _tq = robot.data.applied_torque[:, [_IAL, _IAR]]
            if a.cop_noise > 0:
                _tq = _tq + torch.randn_like(_tq) * a.cop_noise
            _S = _SL * _tq[:, 0] + _SR * _tq[:, 1]
            CAL["S_f"] = torch.where(torch.full((N,), i == 0, device=dev), _S, CAL["S_f"] + (dt / 0.3) * (_S - CAL["S_f"]))
            _pg0 = robot.data.projected_gravity_b; _pit0 = torch.rad2deg(torch.atan2(_pg0[:, 0], -_pg0[:, 2]))
            _both0 = (sensor.data.net_forces_w[:, fid, :].norm(dim=-1) > 5.0).all(dim=1); _w0 = robot.data.root_ang_vel_b[:, :2].norm(dim=-1)
            _af = alive.float()
            if 1.0 <= t < 1.5:
                CAL["S_pre"] += _S * _af; CAL["n_pre"] += _af; CAL["p_pre"] += _pit0 * _af; CAL["tl_pre"] += _tq[:, 0] * _af; CAL["tr_pre"] += _tq[:, 1] * _af
            if t > a.stand_first - 0.5:
                CAL["S_post"] += _S * _af; CAL["n_post"] += _af; CAL["p_post"] += _pit0 * _af
            if CT is not None and t >= CT["t0"]:
                _q = alive & _both0 & (_w0 < 0.3)
                _d = torch.deg2rad(torch.tensor(CT["k"], device=dev)) * (CAL["S_f"] - CT["ref"]) * dt
                PX["trim"] = torch.where(_q, (PX["trim"] + _d).clamp(-math.radians(CT["max"]), math.radians(CT["max"])), PX["trim"])''')
rep('    r["collapse"] = bool(fr > 0.03 or', '''    if a.stand_first > 0:
        _np_ = CAL["n_pre"].clamp(min=1); _nq_ = CAL["n_post"].clamp(min=1); _tr_ = torch.rad2deg(PX["trim"])
        _mm = lambda x: float((x * s_).sum() / ns)
        r["cal"] = {"S_pre": _mm(CAL["S_pre"] / _np_), "S_post": _mm(CAL["S_post"] / _nq_), "pitch_pre": _mm(CAL["p_pre"] / _np_), "pitch_post": _mm(CAL["p_post"] / _nq_),
                    "tauL_pre": _mm(CAL["tl_pre"] / _np_), "tauR_pre": _mm(CAL["tr_pre"] / _np_), "trim_mean_deg": _mm(_tr_), "trim_sd_deg": float(_tr_[surv].std()) if int(surv.sum()) > 1 else 0.0,
                    "S_pre_sd": float((CAL["S_pre"] / _np_)[surv].std()) if int(surv.sum()) > 1 else 0.0, "S_post_sd": float((CAL["S_post"] / _nq_)[surv].std()) if int(surv.sum()) > 1 else 0.0}
    r["collapse"] = bool(fr > 0.03 or''')
rep('"events": EV_DESC, "wrapped_obs": OBS_DESC,', '"stand_first": a.stand_first, "cop_trim": a.cop_trim, "cop_noise": a.cop_noise, "cop_sign": a.cop_sign, "events": EV_DESC, "wrapped_obs": OBS_DESC,')
open(DST + ".tmp", "w", encoding="utf-8").write(src); os.replace(DST + ".tmp", DST); print("written", DST, len(src.splitlines()))
