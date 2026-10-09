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
dst = os.path.join(R, "rl_walking/scripts/eval_transition_kwx_nb.py")
open(dst, "w", encoding="utf-8").write(src); print("wrote", dst)
