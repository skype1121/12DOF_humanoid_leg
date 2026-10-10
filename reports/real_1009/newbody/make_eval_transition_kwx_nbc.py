#!/usr/bin/env python3
"""eval_transition_kwx_nb.py(새 몸 합성 러너 전환 평가기) 사본 eval_transition_kwx_nbc.py 를 만든다(1010 실물 전 점검 NEWBODY_CHECKS.md).
원본·eval_transition_kwx_nb.py 불변, 앵커 불일치면 assert. 더하는 것(전부 기본 꺼짐 — 꺼진 채로는 출력 동일):
  --lock_offset_deg "a,b"   고정 관절(--lock_joints 순서)의 고정각 = 기본자세 + 이 값[°]. 러너 입력(--lock_obs_zero)은 그대로 기본자세(0°) — stage8 합성 피드백 계약
  --trace_qd                transall npz 에 관절 속도 qd(rad/s) 추가(기계 출력 |τ·ω| 계산용, 기록 전용)
  --kw_contact_fix "B:1"    합성 러너 쪽이 보는 접촉 비트를 고정(L|R|B : 0|1). 물리·채점·env 관측(transall 의 cont) 불변
  --kw_contact_layer all|kepol|w3pol|sup
        all   = 러너·감독기에 들어가는 비트 전부
        kepol = 서기·회복 정책 관측(C5Runner.core.assemble_obs)에만
        w3pol = 걷기 정책 관측(W3 러너 assemble_obs)에만
        sup   = 정책 관측을 뺀 나머지(걷기 감독기 켜기 조건·정지 정착 판정·회복 감지기)에만"""
import os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src = open(os.path.join(R, "rl_walking/scripts/eval_transition_kwx_nb.py"), encoding="utf-8").read()


def rep(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:80])
    return s.replace(a, b)


A1 = 'parser.add_argument("--lock_obs_zero", action="store_true", default=False, help="1010: 합성 러너에 주는 굳은 관절의 관절각 = 기본자세·속도 = 0")\n'
src = rep(src, A1, A1 +
          'parser.add_argument("--lock_offset_deg", type=str, default="")\n'
          'parser.add_argument("--trace_qd", action="store_true", default=False)\n'
          'parser.add_argument("--kw_contact_fix", type=str, default="")\n'
          'parser.add_argument("--kw_contact_layer", type=str, default="all", choices=("all", "kepol", "w3pol", "sup"))\n')

A2 = "    _LOCK_TGT = robot.data.default_joint_pos[:, _LOCK_IDS].clone()\n"
src = rep(src, A2, A2 + """    if args_cli.lock_offset_deg:      # 1010 점검: 출력물 공차 모사 — 고정각 오차
        _lofs = [float(x) for x in args_cli.lock_offset_deg.split(",")]; assert len(_lofs) == len(_LOCK_IDS), (_lofs, _LOCK_NAMES)
        _LOCK_TGT = _LOCK_TGT + torch.deg2rad(torch.tensor(_lofs, device=_LOCK_TGT.device, dtype=_LOCK_TGT.dtype)).unsqueeze(0)
""")

A3 = "    _lm = float(lock_dev_max.max()); _gx = pre_pass & (_nbx_n > 0)\n"
src = rep(src, A3, """    if args_cli.lock_offset_deg:
        P(f"[고정각 오차] 고정 목표 = 기본자세 + {args_cli.lock_offset_deg}° ({','.join(_LOCK_NAMES)}) · 실행 끝 실제 관절각(env 평균, 기본자세 대비) {[round(float(v), 3) for v in torch.rad2deg((robot.data.joint_pos[:, _LOCK_IDS] - robot.data.default_joint_pos[:, _LOCK_IDS]).mean(0))]}° · 아래 변위는 이 목표 대비")
    if args_cli.kw_contact_fix:
        P(f"[접촉 고정] 러너 쪽 접촉 비트 {args_cli.kw_contact_fix} · 층 {args_cli.kw_contact_layer} · 관측 조립 감싼 러너 {globals().get('_CFX_NWRAP', 0)} 개 · 감싼 조립 호출 {globals().get('_CFX_CALLS', [0])[0]} 회(물리·채점·env 관측 불변)")
""" + A3)

A4 = '            _TALL["q"].append(robot.data.joint_pos[:_na].half().cpu()); _TALL["tq"].append(robot.data.applied_torque[:_na].half().cpu())   # 실측 관절각·적용 토크(추종 오차·포화 통계)\n'
src = rep(src, A4, A4 + """            if args_cli.trace_qd:
                _TALL.setdefault("qd", []).append(robot.data.joint_vel[:_na].half().cpu())
                _TALL.setdefault("ptg", []).append(robot.data.joint_pos_target[:_na].half().cpu())      # 1011: PD 가 실제로 받는 목표(지연·슬루·LPF 뒤) — 요구 토크 계산용
                # 1011 보고용: 좌·우 발 수직 접지력[N], 전신 무게중심의 (앞, 왼) 위치[m] — 두 발 중점 기준(골반 진행 방향 좌표)
                _TALL.setdefault("fz", []).append(_cs.data.net_forces_w[:_na][:, [_fl, _fr], 2].half().cpu())
                if "_MASS_B" not in globals():
                    globals()["_MASS_B"] = robot.root_physx_view.get_masses().to(dev)
                _comw = (robot.data.body_com_pos_w[:_na, :, :2] * _MASS_B[:_na, :, None]).sum(1) / _MASS_B[:_na].sum(1, keepdim=True)
                _midw = robot.data.body_pos_w[:_na, _FT, :2].mean(1)
                _dcm = _comw - _midw
                _TALL.setdefault("com", []).append(torch.stack([(_dcm * _dir_vec("f")[:_na, :2]).sum(-1), (_dcm * _dir_vec("l")[:_na, :2]).sum(-1)], -1).half().cpu())
""")

A5 = "        ct = (po[:, _FC1 - 2:_FC1] > 0.5).float().cpu().numpy().astype(_npk.float32)\n"
src = rep(src, A5, A5 + """        if args_cli.kw_contact_fix:      # 1010 점검 P7: 접촉 관측 고정(진단 대조군) — 러너·감독기 입력만, 물리·채점 불변
            _cs, _cv = args_cli.kw_contact_fix.split(":"); _ccols = {"L": [0], "R": [1], "B": [0, 1]}[_cs.strip().upper()]; assert _cv in ("0", "1")
            _ct_true = ct.copy(); _ct_fix = ct.copy(); _ct_fix[:, _ccols] = float(_cv)
            globals()["_CFX_NOW"] = (_ct_true, _ct_fix)
            if "_CFX_NWRAP" not in globals():
                globals()["_CFX_CALLS"] = [0]
                def _cfx_wrap(_obj, _lay, _e):
                    _o = _obj.assemble_obs
                    def _w(*a_, **k_):
                        _tr, _fx = globals()["_CFX_NOW"]; _use = (_fx if args_cli.kw_contact_layer in ("all", _lay) else _tr)[_e]
                        globals()["_CFX_CALLS"][0] += 1
                        if "contact2" in k_:
                            k_["contact2"] = _use
                        else:
                            assert len(a_) == 6, len(a_); a_ = tuple(a_[:5]) + (_use,)
                        return _o(*a_, **k_)
                    _obj.assemble_obs = _w
                _nw = 0
                for _e2, _R2 in enumerate(KWR):
                    _kc = _R2.ke.__dict__.get("core", _R2.ke); _wc = _R2.w3r.__dict__.get("core", _R2.w3r)
                    assert _kc is not _wc
                    _cfx_wrap(_kc, "kepol", _e2); _cfx_wrap(_wc, "w3pol", _e2); _nw += 2
                globals()["_CFX_NWRAP"] = _nw
            ct = _ct_fix if args_cli.kw_contact_layer in ("all", "sup") else _ct_true
""")
dst = os.path.join(R, "rl_walking/scripts/eval_transition_kwx_nbc.py")
open(dst, "w", encoding="utf-8").write(src); print("wrote", dst)
