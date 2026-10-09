#!/usr/bin/env python3
"""eval_ground_recovery.py 사본에 골반 관절 물리 고정(--lock_joints)을 넣어 eval_ground_recovery_lock.py 를 만든다(1009).
원본은 고치지 않는다. 앵커가 안 맞으면 assert. --lock_joints 가 비면 원본과 같은 동작(대조군 C0)."""
import os
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
src = open(os.path.join(R, "rl_walking/scripts/eval_ground_recovery.py"), encoding="utf-8").read()


def rep(s, a, b):
    assert s.count(a) == 1, (s.count(a), a)
    return s.replace(a, b)


A1 = 'parser.add_argument("--out_tag", type=str, default="rec")\n'
src = rep(src, A1, A1 +
          'parser.add_argument("--lock_joints", type=str, default="", help="1009: 물리 고정 관절 이름 \',\' 구분 — 위치 한계를 정책 기본자세 ± lock_band_deg 로 묶는다(모터 제거·기계 고정 모사). 정책 행동은 12 차원 그대로")\n'
          'parser.add_argument("--lock_band_deg", type=float, default=0.05, help="1009: 고정 한계 반폭 [deg]")\n'
          'parser.add_argument("--lock_obs_zero", action="store_true", default=False, help="1010: --lock_joints 관절의 정책 관측 열(joint_pos·joint_vel·joint_vel_ema, 이력 전 프레임)을 정책 입력 직전에 0 으로 — 실물 stage8 의 기구 고정 합성 피드백(위치 영점·속도 0) 계약. 기본 꺼짐 = 종전과 동일")\n'
          'parser.add_argument("--link_urdf", type=str, default="", help="1009: \'URDF경로:링크,링크\' — 이 URDF 의 inertial(질량·질량 중심·관성 텐서)로 해당 링크를 실행 중 덮어쓴다(부품 교체 모사). startup 질량 DR 뒤에 적용")\n'
          'parser.add_argument("--mass_delta", type=str, default="", help="1009: 링크 질량 변화 \'링크:kg,...\' (모터 제거 모사) — 질량 중심 유지, 관성은 질량 비율로 축소. startup 질량 DR 뒤에 적용")\n')

A2 = "N = args_cli.num_envs\n\npolicy = None; resume_path"
B2 = '''N = args_cli.num_envs
# ── 1009 링크 질량 변화(모터 제거) ──
_MD = []; _MD_INFO = ""
if args_cli.link_urdf:
    import xml.etree.ElementTree as _ET
    _up, _ul = args_cli.link_urdf.rsplit(":", 1); _bn_m = list(robot.body_names); _pv = robot.root_physx_view
    _m0 = _pv.get_masses().clone(); _i0 = _pv.get_inertias().clone(); _c0 = _pv.get_coms().clone()
    _m1 = _m0.clone(); _i1 = _i0.clone(); _c1 = _c0.clone(); _eid = torch.arange(N, device=_m0.device)
    _ur = {l.get("name"): l.find("inertial") for l in _ET.parse(_up).getroot().findall("link")}
    _parts = []; _LU_CHK = []
    for _nm in _ul.split(","):
        _b = _bn_m.index(_nm); _in = _ur[_nm]; _o = _in.find("origin"); _I = _in.find("inertia")
        assert all(abs(float(v)) < 1e-12 for v in (_o.get("rpy") or "0 0 0").split()), "inertial rpy ≠ 0 은 지원 안 함"
        _mass = float(_in.find("mass").get("value")); _xyz = [float(v) for v in _o.get("xyz").split()]
        _g = lambda k: float(_I.get(k))
        _T = torch.tensor([[_g("ixx"), _g("ixy"), _g("ixz")], [_g("ixy"), _g("iyy"), _g("iyz")], [_g("ixz"), _g("iyz"), _g("izz")]], dtype=_i0.dtype)
        _ev0 = torch.linalg.eigvalsh(_i0[0, _b].reshape(3, 3).double().cpu())
        _parts.append(f"{_nm} 종전: 질량 {float(_m0[0, _b]):.5f} · com {[round(float(v), 5) for v in _c0[0, _b]]} · 관성 고유값 {[f'{float(v):.4e}' for v in _ev0]} · 관성 9 {[f'{float(v):.3e}' for v in _i0[0, _b]]}")
        # 규약(자산 읽음값으로 확인): get/set_inertias 9 성분 = 링크 축 기준 전체 텐서(종전 읽음값이 원본 URDF ixx..izz 와 같음). com 쿼터니언은 PhysX 가 주축으로 따로 관리
        _m1[:, _b] = _mass; _c1[:, _b, :3] = torch.tensor(_xyz, dtype=_c0.dtype); _i1[:, _b] = _T.reshape(9)
        _LU_CHK.append((_nm, _b, _mass, _xyz, torch.linalg.eigvalsh(_T.double())))
    _pv.set_masses(_m1, _eid); _pv.set_coms(_c1, _eid); _pv.set_inertias(_i1, _eid)
    _m2 = _pv.get_masses(); _i2 = _pv.get_inertias(); _c2 = _pv.get_coms()
    for _nm, _b, _mass, _xyz, _evT in _LU_CHK:
        _ev2 = torch.linalg.eigvalsh(_i2[0, _b].reshape(3, 3).double().cpu())
        _okm = float((_m2[:, _b] - _mass).abs().max()) < 1e-6; _okc = float((_c2[:, _b, :3].cpu() - torch.tensor(_xyz)).abs().max()) < 1e-6
        _oki = float(((_ev2 - _evT).abs() / _evT).max()) < 1e-3 and float((_i2[0, _b].cpu() - _i1[0, _b].cpu()).abs().max()) < 1e-8
        _parts.append(f"{_nm} 적용 뒤 읽음: 질량 {float(_m2[0, _b]):.5f}(목표 {_mass}) {'일치' if _okm else '불일치!'} · com {[round(float(v), 5) for v in _c2[0, _b]]} {'일치' if _okc else '불일치!'} · 관성 9 {[f'{float(v):.3e}' for v in _i2[0, _b]]} · 고유값 {[f'{float(v):.4e}' for v in _ev2]} vs URDF {[f'{float(v):.4e}' for v in _evT]} {'일치' if _oki else '불일치!'}")
        _MD.append((_nm, _b, _mass - float(_m0[0, _b])))
        if not (_okm and _okc and _oki):
            _parts.append("적용 확인 실패 — 무효!")
    _MD_M0 = _m0.clone(); _MD_M1 = _m2.clone()
    _MD_INFO = f"링크 교체({os.path.basename(_up)}) | " + " || ".join(_parts) + f" | 전체 질량 env 평균 {float(_m0.sum(-1).mean()):.4f}→{float(_m2.sum(-1).mean()):.4f} kg (범위 {float(_m2.sum(-1).min()):.3f}–{float(_m2.sum(-1).max()):.3f})"
    print("[link_urdf] " + _MD_INFO, flush=True)
if args_cli.mass_delta:
    _bn_m = list(robot.body_names); _pv = robot.root_physx_view
    _m0 = _pv.get_masses().clone(); _i0 = _pv.get_inertias().clone(); _c0 = _pv.get_coms().clone()
    _m1 = _m0.clone(); _i1 = _i0.clone(); _eid = torch.arange(N, device=_m0.device)
    for _tok in args_cli.mass_delta.split(","):
        _nm, _dv = _tok.split(":"); _dv = float(_dv); assert _nm in _bn_m, f"--mass_delta 미지의 링크 {_nm} (있는 것 {_bn_m})"
        _b = _bn_m.index(_nm); _new = _m0[:, _b] + _dv
        if float(_new.min()) <= 0.0:
            raise SystemExit(f"[mass_delta] {_nm}: 질량 {float(_m0[:, _b].min()):.4f} kg 에서 {_dv:+.3f} → 0 이하. 실행하지 않음")
        _i1[:, _b] = _i0[:, _b] * (_new / _m0[:, _b]).unsqueeze(-1); _m1[:, _b] = _new; _MD.append((_nm, _b, _dv))
    _pv.set_masses(_m1, _eid); _pv.set_inertias(_i1, _eid)
    _m2 = _pv.get_masses(); _i2 = _pv.get_inertias(); _c2 = _pv.get_coms()
    _MD_M0 = _m0.clone(); _MD_M1 = _m2.clone()
    _MD_INFO = " · ".join(f"{_nm} {float(_m0[:, _b].mean()):.4f}→{float(_m2[:, _b].mean()):.4f} kg(env 범위 {float(_m2[:, _b].min()):.4f}–{float(_m2[:, _b].max()):.4f}, 관성 대각합 비 {float((_i2[:, _b, [0, 4, 8]].sum(-1) / _i0[:, _b, [0, 4, 8]].sum(-1)).mean()):.4f})" for _nm, _b, _dv in _MD)
    _MD_INFO += f" | 전체 질량 env 평균 {float(_m0.sum(-1).mean()):.4f}→{float(_m2.sum(-1).mean()):.4f} kg (범위 {float(_m2.sum(-1).min()):.3f}–{float(_m2.sum(-1).max()):.3f}) | 질량 중심 변화 최대 {float((_c2[..., :3] - _c0[..., :3]).abs().max()):.2e} m"
    print("[mass_delta] " + _MD_INFO, flush=True)
    _evm = uenv.event_manager
    print("[mass_delta] 이벤트 목록(모드별): " + str({_md: list(_evm.active_terms[_md]) for _md in _evm.active_terms}), flush=True)
# ── 1009 관절 물리 고정 ──
_LOCK_IDS = []; _LOCK_NAMES = []; _LOCK_TGT = None; _LOCK_SPAWN = None
lock_dev_max = torch.zeros(N, device=dev); lock_dev_all = 0.0
if args_cli.lock_joints:
    _jn_l = list(robot.data.joint_names)
    _LOCK_NAMES = [x.strip() for x in args_cli.lock_joints.split(",") if x.strip()]
    for _nm in _LOCK_NAMES:
        assert _nm in _jn_l, f"--lock_joints 미지의 관절 {_nm} (있는 것 {_jn_l})"
    _LOCK_IDS = [_jn_l.index(_nm) for _nm in _LOCK_NAMES]
    _LOCK_TGT = robot.data.default_joint_pos[:, _LOCK_IDS].clone()
    _LOCK_SPAWN = torch.rad2deg((robot.data.joint_pos[:, _LOCK_IDS] - _LOCK_TGT).abs()).amax(0).cpu().tolist()
    _hb = math.radians(float(args_cli.lock_band_deg))
    _lim_old = robot.data.joint_pos_limits[0, _LOCK_IDS].clone()
    robot.write_joint_position_limit_to_sim(torch.stack([_LOCK_TGT - _hb, _LOCK_TGT + _hb], dim=-1), joint_ids=_LOCK_IDS)
    robot.write_joint_state_to_sim(_LOCK_TGT, torch.zeros_like(_LOCK_TGT), joint_ids=_LOCK_IDS)
    print(f"[lock] 고정 {_LOCK_NAMES} ids {_LOCK_IDS} · 목표(env0) {torch.rad2deg(_LOCK_TGT[0]).cpu().tolist()}° ± {args_cli.lock_band_deg}° · 종전 한계(env0) {torch.rad2deg(_lim_old).cpu().tolist()}° · 리셋 직후 어긋남 최대 {_LOCK_SPAWN}° → 목표로 놓음", flush=True)

policy = None; resume_path'''
src = rep(src, A2, B2)

A3 = "    term = uenv.termination_manager.terminated\n    pitch, roll = _rp()\n"
B3 = A3 + '''    if i >= STEPS - 100 or (STEPS_PRE - 25 <= i < STEPS_PRE):      # 1009 정착 자세: 마지막 2 s(인계 뒤)·홀드 마지막 0.5 s 평균
        if "_SP" not in globals():
            globals()["_SP"] = {k: dict(p=torch.zeros(N, device=dev), r=torch.zeros(N, device=dev), q=torch.zeros(N, J, device=dev), h=torch.zeros(N, device=dev), n=0) for k in ("hold", "end")}
        _sp = _SP["end" if i >= STEPS - 100 else "hold"]
        _sp["p"] += pitch; _sp["r"] += roll; _sp["q"] += torch.rad2deg(robot.data.joint_pos); _sp["h"] += robot.data.root_pos_w[:, 2] - uenv.scene.env_origins[:, 2]; _sp["n"] += 1
    if _LOCK_IDS:      # 고정 확인: 아직 살아 있는 env 의 고정 관절 변위(종료 틱은 자동 리셋 뒤 값이라 제외)
        _ldv = torch.rad2deg((robot.data.joint_pos[:, _LOCK_IDS] - _LOCK_TGT).abs()).amax(-1)
        _lok = alive & ~term
        lock_dev_max = torch.where(_lok, torch.maximum(lock_dev_max, _ldv), lock_dev_max)
    if policy2 is not None and i >= STEPS_PRE:      # 1010: 낙상(종료) 틱에 어느 정책이 행동을 냈는가(0 = 서기, 1 = 회복) · 그때까지 외란 뒤 회복 모드 틱 수
        _lzf = term & alive & (_lz_mode_fall < 0)
        _lz_mode_fall = torch.where(_lzf, sw_mode.float(), _lz_mode_fall)
        _lz_rec_ticks_fall = torch.where(_lzf, sw_ticks - sw_ticks_pre_dist, _lz_rec_ticks_fall)
'''
src = rep(src, A3, B3)

A4 = 'P("시간축 (인계 기준 s | 생존 env 평균 피치 | 생존 비율 | 외란 힘 N | 양발접촉 비율):")\n'
B4 = '''if "_SP" in globals():
    _gS = pre_pass & alive
    for _k, _lb in (("hold", "홀드 마지막 0.5 s"), ("end", "인계 뒤 마지막 2 s")):
        _sp = _SP[_k]; _n = max(_sp["n"], 1); _qm = (_sp["q"] / _n)[_gS].mean(0); _qd = torch.rad2deg(robot.data.default_joint_pos)[_gS].mean(0)
        P(f"[정착 자세] {_lb} (사전검사 통과·끝까지 생존 env {int(_gS.sum())} 평균): 피치 {float((_sp['p'] / _n)[_gS].mean()):+.3f}° (env 표준편차 {float((_sp['p'] / _n)[_gS].std()):.3f}) · 롤 {float((_sp['r'] / _n)[_gS].mean()):+.3f}° (표준편차 {float((_sp['r'] / _n)[_gS].std()):.3f}) · 골반 높이 {float((_sp['h'] / _n)[_gS].mean()):.4f} m · 관절각[°] "
          + " ".join(f"{jn[_j]}={float(_qm[_j]):+.2f}(기본 {float(_qd[_j]):+.2f})" for _j in range(J)))
if _MD:
    _mE = robot.root_physx_view.get_masses()
    _keep = float((_mE - _MD_M1).abs().max())
    P(f"[질량] {_MD_INFO}")
    P(f"[질량] 실행 끝 재확인: 설정값 대비 링크 질량 최대 차 {_keep:.2e} kg · 전체 질량 env 평균 {float(_mE.sum(-1).mean()):.4f} kg → {'유지(덮어쓰기 없음)' if _keep < 1e-5 else '바뀜 — 무효!'}")
    P("[질량] 이벤트(모드별): " + str(dict((_md, list(uenv.event_manager.active_terms[_md])) for _md in uenv.event_manager.active_terms)))
else:
    P(f"[질량] 변화 없음 · 전체 질량 env 평균 {float(robot.root_physx_view.get_masses().sum(-1).mean()):.4f} kg")
if _LOCK_IDS:
    _lm = float(lock_dev_max.max()); _lmp = float(lock_dev_max[pre_pass].max()) if pre_pass.any() else float("nan")
    P(f"[고정] 관절 {','.join(_LOCK_NAMES)} · 한계 = 기본자세 ± {args_cli.lock_band_deg}° · 전체 구간(사전검사+인계 뒤, 생존 틱) 최대 변위: 전 env {_lm:.4f}° · 사전 검사 통과 env {_lmp:.4f}° · 판정(<0.3°) {'유효' if _lm < 0.3 else '무효!'} · 리셋 직후 어긋남 {_LOCK_SPAWN}°")
else:
    P("[고정] 없음(대조군)")
P(f"[고정 관측] --lock_obs_zero {'켬: 열 ' + str(len(_LZ_COLS)) + ' 개를 정책 입력 직전 0 으로 (0 으로 하기 전 값의 |최대| ' + format(_LZ_SEEN, '.4f') + ')' if _LZ_COLS else '끔(정책이 시뮬 값을 봄)'}")
if _LOCK_IDS:
    _gx = pre_pass & (post_n > 0)
    _lxc = torch.cat([_v[_m & pre_pass] for _v, _m in _lx_cmd]) if _lx_cmd else torch.zeros(0, device=dev)
    P(f"[굳은 관절 제외] 토크 포화(사전검사 통과 env 평균, ≥95 %): 굳은 관절 뺀 어느 관절이든 {100 * float((_lx_sat[_gx] / post_n[_gx]).mean()) if _gx.any() else float('nan'):.2f} % 틱 · 전 관절 {100 * float((sat[_gx] / post_n[_gx]).mean()) if _gx.any() else float('nan'):.2f} % 틱"
      + (f" · 굳은 관절 명령각 |명령−고정각| 첫 외란 뒤 2 s(생존 틱 {int(_lxc.numel())}): 중앙 {float(_lxc.median()):.2f}° · 99 % {float(_lxc.quantile(0.99)):.2f}° · 평균 {float(_lxc.mean()):.2f}°" if _lxc.numel() else " · 굳은 관절 명령각: 외란 없음"))
if policy2 is not None:
    _lzF = pre_pass & fell & (_lz_mode_fall >= 0); _lzR = _lzF & (_lz_mode_fall > 0.5); _lzS = _lzF & (_lz_mode_fall < 0.5)
    _lzS0 = _lzS & (_lz_rec_ticks_fall <= 0); _lzS1 = _lzS & (_lz_rec_ticks_fall > 0)
    P(f"[낙상 시 모드] 사전검사 통과 env 의 인계 뒤 낙상 {int((pre_pass & fell).sum())} (모드 기록 {int(_lzF.sum())}): 회복 모드 중 {int(_lzR.sum())} · 서기 모드 중 {int(_lzS.sum())} "
      f"(외란 뒤 회복 모드가 한 번도 안 켜짐 {int(_lzS0.sum())} · 켜졌다가 서기로 돌아온 뒤 {int(_lzS1.sum())}) · 회복 모드 중 낙상 env 의 외란 뒤 회복 모드 시간 중앙 {float((_lz_rec_ticks_fall[_lzR] / 50.0).median()) if _lzR.any() else float('nan'):.2f} s")
    with open(os.path.join(DATA, "log_picture", f"ground_recovery_{args_cli.out_tag}_modefall.csv"), "w") as _lzfh:
        _lzfh.write("env,pre_pass,fell,mode_at_fall,rec_s_before_fall\\n")
        for _e in range(N):
            _lzfh.write(f"{_e},{int(pre_pass[_e])},{int(fell[_e])},{int(_lz_mode_fall[_e])},{float(_lz_rec_ticks_fall[_e]) / 50.0:.2f}\\n")
''' + A4
src = rep(src, A4, B4)

A5 = "obs = env.get_observations()\n"
B5 = A5 + """_LZ_COLS = []; _lz_mode_fall = torch.full((N,), -1.0, device=dev); _lz_rec_ticks_fall = torch.zeros(N, device=dev)
if args_cli.lock_obs_zero:      # 1010: 기구 고정 관절 관측 열 = 0 (실물 합성 피드백 계약)
    assert _LOCK_IDS, "--lock_obs_zero 는 --lock_joints 와 함께"
    _lzo = 0; _lzJ = len(robot.data.joint_names)
    for _lzn, _lzd in zip(uenv.observation_manager.active_terms["policy"], uenv.observation_manager.group_obs_term_dim["policy"]):
        _lzs = 1
        for _v in (_lzd if isinstance(_lzd, (tuple, list)) else (_lzd,)): _lzs *= int(_v)
        if _lzn in ("joint_pos", "joint_vel", "joint_vel_ema"):
            assert _lzs % _lzJ == 0, (_lzn, _lzs)
            _LZ_COLS += [_lzo + _f * _lzJ + _j for _f in range(_lzs // _lzJ) for _j in _LOCK_IDS]
        _lzo += _lzs
    _LZ_PRE = float(obs["policy"][:, _LZ_COLS].abs().max())
    print(f"[lock_obs_zero] 고정 관절 관측 열 {len(_LZ_COLS)} 개(joint_pos·joint_vel·joint_vel_ema × 이력) 를 정책 입력 직전에 0 으로 · 0 으로 하기 전 첫 관측의 |값| 최대 {_LZ_PRE:.4f}", flush=True)
_LZ_SEEN = 0.0
_lx_sat = torch.zeros(N, device=dev); _lx_cmd = []; _LX_OTH = [_j for _j in range(len(robot.data.joint_names)) if _j not in _LOCK_IDS]      # 1010: 굳은 관절을 뺀 토크 포화·굳은 관절 명령각
"""
src = rep(src, A5, B5)

A6 = "        a = torch.zeros(N, J, device=dev) if (policy is None or i < STEPS_PRE) else policy(obs)\n"
B6 = """        if _LZ_COLS:
            _LZ_SEEN = max(_LZ_SEEN, float(obs["policy"][:, _LZ_COLS].abs().max())); obs["policy"][:, _LZ_COLS] = 0.0
""" + A6
src = rep(src, A6, B6)

A7 = "        tq = robot.data.applied_torque.abs(); sat[a_] += (tq[a_] >= 0.95 * _eff[a_]).any(1).float()\n"
B7 = A7 + """        if _LOCK_IDS:      # 1010: 굳은 관절의 헛토크(한계와 내부 상쇄)를 뺀 포화 · 밀기 뒤 2 s 의 굳은 관절 명령각
            _lx_sat[a_] += (tq[a_][:, _LX_OTH] >= 0.95 * _eff[a_][:, _LX_OTH]).any(1).float()
            if dists and dists[0].t0 <= t_rel < dists[0].t0 + 2.0:
                _lx_cmd.append((torch.rad2deg((cmd[:, _LOCK_IDS] - _LOCK_TGT).abs()).amax(1).clone(), a_.clone()))
"""
src = rep(src, A7, B7)

dst = os.path.join(R, "rl_walking/scripts/eval_ground_recovery_lock.py")
open(dst, "w", encoding="utf-8").write(src)
print("wrote", dst)
