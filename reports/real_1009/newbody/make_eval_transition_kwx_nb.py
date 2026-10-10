#!/usr/bin/env python3
"""eval_transition_kwx.py(합성 러너 폐루프 전환 평가기) 사본 eval_transition_kwx_nb.py 를 만든다(1010 새 몸). 원본 불변, 앵커 불일치면 assert.
더하는 것: --lock_joints/--lock_band_deg/--link_urdf/--mass_delta(= eval_ground_recovery_lock.py 의 블록 그대로) · --lock_obs_zero
(합성 러너에 주는 굳은 관절의 관절각 = 기본자세, 속도 = 0 — stage8 합성 피드백 계약) · 결과 파일의 [질량]·[고정]·[굳은 관절 제외] 줄."""
import os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src = open(os.path.join(R, "rl_walking/scripts/eval_transition_kwx.py"), encoding="utf-8").read()
lock = open(os.path.join(R, "rl_walking/scripts/eval_ground_recovery_lock.py"), encoding="utf-8").read()


def rep(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:80])
    return s.replace(a, b)


A1 = 'parser.add_argument("--out_tag", type=str, default="rec")\n'
src = rep(src, A1, A1 +
          'parser.add_argument("--lock_joints", type=str, default="")\n'
          'parser.add_argument("--lock_band_deg", type=float, default=0.05)\n'
          'parser.add_argument("--link_urdf", type=str, default="")\n'
          'parser.add_argument("--mass_delta", type=str, default="")\n'
          'parser.add_argument("--lock_obs_zero", action="store_true", default=False, help="1010: 합성 러너에 주는 굳은 관절의 관절각 = 기본자세·속도 = 0")\n')
# 질량·고정 블록: 잠금 평가기에서 그대로 가져온다
i0 = lock.index("# ── 1009 링크 질량 변화(모터 제거) ──"); i1 = lock.index("\npolicy = None; resume_path", i0)
A2 = "N = args_cli.num_envs\n\npolicy = None; resume_path"
src = rep(src, A2, "N = args_cli.num_envs\n" + lock[i0:i1] + "\n_nbx_sat = torch.zeros(N, device=dev); _nbx_n = torch.zeros(N, device=dev)\n\npolicy = None; resume_path")
A3 = "    term = uenv.termination_manager.terminated\n    pitch, roll = _rp()\n"
src = rep(src, A3, A3 + """    if _LOCK_IDS:      # 고정 확인(살아 있는 env, 종료 틱 제외)
        _ldv = torch.rad2deg((robot.data.joint_pos[:, _LOCK_IDS] - _LOCK_TGT).abs()).amax(-1)
        _lok = alive & ~term
        lock_dev_max = torch.where(_lok, torch.maximum(lock_dev_max, _ldv), lock_dev_max)
""")
A5 = "        jv = (_KW_RAWJV[0] if _KW_RAWJV[0] is not None else po[:, _KW_JV1 - 12:_KW_JV1]).cpu().numpy().astype(_npk.float32)\n"
src = rep(src, A5, A5 + """        if args_cli.lock_obs_zero:      # 1010: stage8 기구 고정 합성 피드백(위치 = 영점 = 기본자세, 속도 0)
            assert _LOCK_IDS, "--lock_obs_zero 는 --lock_joints 와 함께"
            qp[:, _LOCK_IDS] = _KW_POSE[_LOCK_IDS]; jv[:, _LOCK_IDS] = 0.0
""")
A7 = "        tq = robot.data.applied_torque.abs(); sat[a_] += (tq[a_] >= 0.95 * _eff[a_]).any(1).float()\n"
src = rep(src, A7, A7 + """        if _LOCK_IDS:
            _nbo = [j for j in range(tq.shape[1]) if j not in _LOCK_IDS]
            _nbx_sat[a_] += (tq[a_][:, _nbo] >= 0.95 * _eff[a_][:, _nbo]).any(1).float(); _nbx_n[a_] += 1.0
""")
A4 = 'P("시간축 (인계 기준 s | 생존 env 평균 피치 | 생존 비율 | 외란 힘 N | 양발접촉 비율):")\n'
src = rep(src, A4, """if _MD:
    _mE = robot.root_physx_view.get_masses(); _keep = float((_mE - _MD_M1).abs().max())
    P(f"[질량] {_MD_INFO}")
    P(f"[질량] 실행 끝 재확인: 설정값 대비 링크 질량 최대 차 {_keep:.2e} kg · 전체 질량 env 평균 {float(_mE.sum(-1).mean()):.4f} kg → {'유지(덮어쓰기 없음)' if _keep < 1e-5 else '바뀜 — 무효!'}")
else:
    P(f"[질량] 변화 없음 · 전체 질량 env 평균 {float(robot.root_physx_view.get_masses().sum(-1).mean()):.4f} kg")
if _LOCK_IDS:
    _lm = float(lock_dev_max.max()); _gx = pre_pass & (_nbx_n > 0)
    P(f"[고정] 관절 {','.join(_LOCK_NAMES)} · 한계 = 기본자세 ± {args_cli.lock_band_deg}° · 최대 변위: 전 env {_lm:.4f}° · 판정(<0.3°) {'유효' if _lm < 0.3 else '무효!'} · 러너 입력 고정 관절 = {'기본자세·속도 0(--lock_obs_zero)' if args_cli.lock_obs_zero else '시뮬 값'}")
    P(f"[굳은 관절 제외] 토크 포화(사전검사 통과 env 평균, 인계 뒤, ≥95 %): 굳은 관절 뺀 어느 관절이든 {100 * float((_nbx_sat[_gx] / _nbx_n[_gx]).mean()) if _gx.any() else float('nan'):.2f} % 틱")
else:
    P("[고정] 없음")
""" + A4)
# 1010 빠른 걸음: --kw_w3_fstep F(> 0) — 합성 러너의 걷기 시계(W3_SPECS 의 W3Config.f_step)와 평가 env 의 위상 시계를 같은 값으로. 0(기본) = 종전과 같은 코드 경로.
src = rep(src, A1, A1 + 'parser.add_argument("--kw_w3_fstep", type=float, default=0.0, help="1010: 걷기 시계 주파수 덮어쓰기[Hz](러너·평가 env 둘 다). 0 = 등록값 1.1")\n')
A8 = "    _BR.enable_experimental_w3()\n"
src = rep(src, A8, """    if args_cli.kw_w3_fstep > 0:
        import dataclasses as _dc_
        _sp_ = _W3D.W3_SPECS["stand_w3s3d_15800"]
        _W3D.W3_SPECS["stand_w3s3d_15800"] = _sp_[:3] + (_dc_.replace(_sp_[3], f_step=float(args_cli.kw_w3_fstep)),) + _sp_[4:]
        print(f"[runner_loop] ★ 걷기 시계 덮어쓰기(평가 전용): f_step {_sp_[3].f_step} → {args_cli.kw_w3_fstep} Hz (러너 W3Config + 평가 env)", flush=True)
""" + A8)
A9 = '_evc.w3_gait = _ETC(func=_tr_gait_update, mode="interval", interval_range_s=(0.0, 0.0), params={"f_step": 1.1})'
src = rep(src, A9, '_evc.w3_gait = _ETC(func=_tr_gait_update, mode="interval", interval_range_s=(0.0, 0.0), params={"f_step": float(args_cli.kw_w3_fstep) if args_cli.kw_w3_fstep > 0 else 1.1})')
# 1010 힙 외전 한계 대조: --kw_effort_joints S — --kw_effort_scales 배율을 이름에 S 가 든 관절에만 건다('' = 종전: 전 관절).
src = rep(src, A1, A1 + 'parser.add_argument("--kw_effort_joints", type=str, default="", help="1010: --kw_effort_scales 를 이름에 이 문자열이 든 관절에만 적용")\n')
A10 = "_nom_ = robot.data.joint_effort_limits.clone(); _new_ = _nom_ * _sc_\n"
src = rep(src, A10, """_nom_ = robot.data.joint_effort_limits.clone(); _new_ = _nom_ * _sc_
                if args_cli.kw_effort_joints:
                    _ejm_ = torch.tensor([args_cli.kw_effort_joints in n_ for n_ in robot.joint_names], device=dev)
                    assert bool(_ejm_.any()), "--kw_effort_joints: 맞는 관절 없음"
                    _new_ = torch.where(_ejm_.unsqueeze(0), _new_, _nom_)
                    print(f"[runner_loop] 토크 한계 배율은 {[n_ for n_ in robot.joint_names if args_cli.kw_effort_joints in n_]} 에만", flush=True)
""")
# 1011 걷기 켜는 순간 완화(시뮬 시험용): --kw_w3_soft "T,d0" — KE→W3 전환 뒤 T 초 동안 **시뮬 액션 항**의 틱당 목표 변화 상한을 d0° 에서 4° 까지 선형으로(env 별).
#   --runner_loop 경로에서 PD 목표를 만드는 것은 시뮬 액션 항(지연 → 슬루)이므로 여기에 건다. '' (기본) = 종전(상수 4°/틱, 같은 코드 경로).
#   배포 대응(러너·stage8 쪽 슬루)은 효과가 확인되면 따로 만든다.
src = rep(src, A1, A1 + 'parser.add_argument("--kw_w3_soft", type=str, default="", help="1011: 걷기 전환 뒤 T 초 동안 틱당 목표 변화 상한 d0° → 4° 선형(\'T,d0\'). \'\' = 끔")\n')
A11 = "            act[e] = R.last_step_action; kind[e] = R.last_step_kind; want_v[e] = bool(clk.want)\n"
src = rep(src, A11, A11 + """            if args_cli.kw_w3_soft:
                if "_SOFT_TK" not in globals():
                    globals()["_SOFT_TK"] = [10 ** 6] * N
                _SOFT_TK[e] = int(R.ticks_in_mode) if kind[e] == 2 else 10 ** 6
""")
A12 = "        obs, _, dones, _ = env.step(a)\n"
src = rep(src, A12, """        if args_cli.kw_w3_soft and "_SOFT_TK" in globals():
            _sT_, _sd0_ = [float(x) for x in args_cli.kw_w3_soft.split(",")]
            _st_term_ = uenv.action_manager.get_term("joint_pos")
            if "_SOFT_MAX" not in globals():
                globals()["_SOFT_MAX"] = float(_st_term_._max_delta)
            _fr_ = (torch.tensor(_SOFT_TK, device=dev, dtype=torch.float32) * 0.02 / max(_sT_, 1e-6)).clamp(0.0, 1.0)
            _st_term_._max_delta = (math.radians(_sd0_) + (_SOFT_MAX - math.radians(_sd0_)) * _fr_).unsqueeze(1)
""" + A12)
dst = os.path.join(R, "rl_walking/scripts/eval_transition_kwx_nb.py")
open(dst, "w", encoding="utf-8").write(src); print("wrote", dst)
