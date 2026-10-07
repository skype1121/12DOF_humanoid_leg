"""eval_ground_recovery.py → eval_transition.py (1007 claude2): W3→KE 전환 평가 로직을 정확한 앵커 치환으로 삽입. 앵커가 정확히 1번이 아니면 실패."""
import sys
SRC = "rl_walking/scripts/eval_ground_recovery.py"; DST = "rl_walking/scripts/eval_transition.py"
src = open(SRC, encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, f"앵커 {n}회: {old[:90]!r}"
    src = src.replace(old, new)

# (1) 머리말
rep('"""지면 절차 평가 (0920 밤, 승윤님 절차)', '"""[1007 claude2 — eval_ground_recovery.py 사본 + W3→KE 전환 로직(reports/gen_w3/TRANSITION_W3KE.md). 원본은 수정하지 않음]\n지면 절차 평가 (0920 밤, 승윤님 절차)')

# (2) 인자
rep('parser.add_argument("--video", action="store_true"',
'''parser.add_argument("--ke", type=str, default="", help="1007 전환 평가: '<run>:<model_N.pt>' KE 주(서기) 정책 G21a — 주 --load_run/--checkpoint 는 W3 정책, --switch_to 는 KE 회복 정책(G24Ra)")
parser.add_argument("--ke_mode", type=str, default="none", choices=("none", "restart", "reset", "direct", "nostop"),
                    help="전환 팔: none = W3 STAND 유지(대조) / restart = 정착 뒤 명령 동결 --hold_s → KE 새 러너(이력 백필·last_action 0·EMA 재시작)+체인 리셋(블렌딩) / "
                         "reset = 정착 즉시 KE 러너 리셋만(체인 연속) / direct = 정착 즉시 KE, 상태 이어받기(리셋 없음) / nostop = stop 없이 t_stop 에 ge 강제 0 + reset")
parser.add_argument("--w3_cmd", type=str, default="0.06,0,0", help="W3 걷기 명령 vx,vy,wz")
parser.add_argument("--w3_on", type=float, default=2.0, help="ge 켬 시각 [s, 인계 기준]")
parser.add_argument("--w3_stop", type=float, default=6.0, help="stop 요청 기준 시각 [s]")
parser.add_argument("--stop_phases", type=int, default=8, help="env mod K 로 요청 시각을 한 주기 안 K 위상에 분산")
parser.add_argument("--stop_period", type=float, default=0.909, help="분산 폭 [s] (1.1 Hz 한 주기)")
parser.add_argument("--w3_walk_only", action="store_true", help="stop 요청 없이 계속 걷기(Q4 대조)")
parser.add_argument("--on_phases", type=int, default=1, help="ge 켬 시각을 env mod K 로 한 주기 안 K 위상에 분산(외란 위상 분산용, 1 = 종전)")
parser.add_argument("--w3_lead", type=str, default="0.02,0.08", help="w3_gait_update lead_vy_cmin,lead_wz_cmin (W3S3d 학습 설정)")
parser.add_argument("--settle", type=str, default="quiet", choices=("quiet", "fixed"), help="정착 판정: quiet = 양발(관측)∧감지|ω|<--sq_w 가 --sq_hold s 유지(상한 --settle_max) / fixed = ge 0 뒤 --settle_s")
parser.add_argument("--sq_w", type=float, default=0.15); parser.add_argument("--sq_hold", type=float, default=0.5)
parser.add_argument("--settle_max", type=float, default=3.0); parser.add_argument("--settle_s", type=float, default=1.0)
parser.add_argument("--hold_s", type=float, default=3.0, help="restart: 명령 동결(스택 재기동 가정) [s]")
parser.add_argument("--video", action="store_true"''')

# (3) C2MODE 블록 → W3 설치(전환 모드)
i0 = src.index("C2MODE = bool(args_cli.c2)"); i1 = src.index('if getattr(env_cfg.events, "disturbance", None) is not None:')
src = src[:i0] + '''C2MODE = False; TRMODE = True
if args_cli.c2 or args_cli.hold or args_cli.no_chain:
    raise SystemExit("eval_transition: --c2/--hold/--no_chain 미지원(전환 평가 전용)")
from isaaclab.managers import EventTermCfg as _ETC  # noqa: E402
from rl_walking import w3 as _w3m  # noqa: E402
_evc = env_cfg.events
for _k in ("w3_cmd", "w3_gait", "w3_track", "gait_phase"):
    if getattr(_evc, _k, None) is not None:
        raise SystemExit(f"전환 평가: env 에 이미 {_k} 이벤트가 있음 — 지면 평가 태스크(G11a)를 쓸 것")
_LVY, _LWZ = (float(x) for x in args_cli.w3_lead.split(","))
_W3CMD = [float(x) for x in args_cli.w3_cmd.split(",")]
os.environ["W3_GAIT_ENABLE"] = "0"
_evc.w3_cmd = _ETC(func=_w3m.install_w3_cmd, mode="startup", params={"p_step": 0.0, "stage": 0})

def _tr_gait_update(env, env_ids, f_step: float = 1.1, ramp_s: float = 0.2):
    """w3_gait_update 래퍼: 평가기가 틱마다 둔 _tr_want(ge 요청)·_tr_cmd(명령) 적용 → W3 래치(반주기 경계 끄기) → nostop 강제 끔."""
    want = getattr(env, "_tr_want", None)
    if want is not None:
        env._ge_cmd = want.clone()
        env.command_manager.get_term("base_velocity").vel_command_b[:] = env._tr_cmd
    _w3m.w3_gait_update(env, env_ids, f_step=f_step, ramp_s=ramp_s, lead_vy_cmin=_LVY, lead_wz_cmin=_LWZ)
    fo = getattr(env, "_tr_force_off", None)
    if fo is not None and bool(fo.any()):
        env._ge[fo] = False; env._gu[fo] = 0.0; env._gf[fo] = 0.0; env._ge_obs[fo] = 0.0
_evc.w3_gait = _ETC(func=_tr_gait_update, mode="interval", interval_range_s=(0.0, 0.0), params={"f_step": 1.1})
_evc.w3_track = _ETC(func=_w3m.w3_step_tracker, mode="interval", interval_range_s=(0.0, 0.0), params={})
_cmdc = env_cfg.commands.base_velocity
_cmdc.heading_command = False; _cmdc.resampling_time_range = (1.0e6, 1.0e6); _cmdc.rel_standing_envs = 0.0
_cmdc.ranges.lin_vel_x = (0.0, 0.0); _cmdc.ranges.lin_vel_y = (0.0, 0.0); _cmdc.ranges.ang_vel_z = (0.0, 0.0)
print(f"[전환] W3 설치(stage 0, lead {_LVY},{_LWZ}) · 명령 {_W3CMD} · ge 켬 {args_cli.w3_on}s · stop {args_cli.w3_stop}s + {args_cli.stop_phases} 위상 · 정착 {args_cli.settle} · 팔 {args_cli.ke_mode} · hold {args_cli.hold_s}s", flush=True)
''' + src[i1:]

# (4) KE 주 정책 로드
rep('''    print(f"[switch] 주 정책 {resume_path} ↔ 회복 정책 {_p2} · 켜기 {_SW_ON} · 끄기 {_SW_OFF}", flush=True)
''', '''    print(f"[switch] 주 정책 {resume_path} ↔ 회복 정책 {_p2} · 켜기 {_SW_ON} · 끄기 {_SW_OFF}", flush=True)
policy_ke = None
if args_cli.ke:
    _rk, _ck = args_cli.ke.split(":")
    _pk = get_checkpoint_path(os.path.join(DATA, "logs", "rsl_rl", agent_cfg.experiment_name), _rk, _ck)
    runner_ke = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner_ke.load(_pk); policy_ke = runner_ke.get_inference_policy(device=dev)
    print(f"[전환] KE 서기 정책 {_pk}", flush=True)
KEMODE = args_cli.ke_mode != "none"
if KEMODE and (policy_ke is None or policy2 is None):
    raise SystemExit("--ke_mode restart|reset|direct|nostop 는 --ke(G21a) 와 --switch_to(G24Ra) 둘 다 필요")
''')

# (5) 전환 상태·관측 배치
rep('''_DEF = robot.data.default_joint_pos.clone()      # 정책 기본자세(= 액션 오프셋) — 시뮬 홀드 명령각
''', '''# ── 1007 전환 상태 ──
_tn = list(uenv.observation_manager.active_terms["policy"]); _td = uenv.observation_manager.group_obs_term_dim["policy"]
_LAYT = {}; _oo_ = 0
for _n_, _d_ in zip(_tn, _td):
    _sz_ = 1
    for _v_ in (_d_ if isinstance(_d_, (tuple, list)) else (_d_,)): _sz_ *= int(_v_)
    _LAYT[_n_] = (_oo_, _oo_ + _sz_); _oo_ += _sz_
_HB = uenv.observation_manager._group_obs_term_history_buffer["policy"]
_HLEN = {n: int(_HB[n].max_length) for n in _LAYT if n in _HB}
_GC0, _GC1 = _LAYT["gait_clock"]; _GCF = 1.1
_FC0, _FC1 = _LAYT["feet_contact"]; _GY0, _GY1 = _LAYT["base_ang_vel"]
print(f"[전환] 관측 배치 {_LAYT} · 이력 {_HLEN}", flush=True)
_eidx = torch.arange(N, device=dev)
tr_tstop = float(args_cli.w3_stop) + (_eidx % max(1, args_cli.stop_phases)).float() / max(1, args_cli.stop_phases) * float(args_cli.stop_period)
tr_want = torch.zeros(N, dtype=torch.bool, device=dev); tr_cmdv = torch.zeros(N, 3, device=dev)
_nanv = lambda: torch.full((N,), float("nan"), device=dev)
tr_geoff = _nanv(); tr_settle = _nanv(); tr_sw = _nanv(); tr_u_stop = _nanv(); tr_u_sw = _nanv(); tr_req = _nanv()
tr_settle_to = torch.zeros(N, dtype=torch.bool, device=dev); settle_run = torch.zeros(N, device=dev)
ke_act = torch.zeros(N, dtype=torch.bool, device=dev); hold_mask = torch.zeros(N, dtype=torch.bool, device=dev)
hold_until = torch.full((N,), -1.0, device=dev); sw_now = torch.zeros(N, dtype=torch.bool, device=dev)
uenv._tr_force_off = torch.zeros(N, dtype=torch.bool, device=dev); ge_prev = torch.zeros(N, dtype=torch.bool, device=dev)
walk_lifts = torch.zeros(N, device=dev)
fell_walk = torch.zeros(N, dtype=torch.bool, device=dev); fell_win = fell_walk.clone(); fell_sw2 = fell_walk.clone(); fell_sw5 = fell_walk.clone(); t_fall_sw = _nanv()
pk_p2 = torch.zeros(N, device=dev); pk_r2 = pk_p2.clone(); pk_w2 = pk_p2.clone(); pk_pw = pk_p2.clone(); pk_rw = pk_p2.clone(); pk_ww = pk_p2.clone()
da_sw = torch.zeros(N, device=dev); dcmd_sw = torch.zeros(N, device=dev); cmd_jump0 = _nanv()
ke_first_on = _nanv(); ke_on_pre = torch.zeros(N, device=dev); nrm_sw_hold = torch.zeros(N, device=dev); nrm_sw_t = _nanv()
_TRW, _TRN = 25, 100
_RK = {"a": 0, "cmd": 0, "q": 0, "pitch": 1, "roll": 1, "w": 1, "v": 1, "both": 1, "ge": 1, "mode": 1, "alive": 1}
_RING = {}; TRS = {}
_HOLD = [hold_mask]; _FRZ = [torch.zeros_like(robot.data.default_joint_pos)]

def _tr_patch_clock(po, t_rel):
    """KE 시작 env 의 클록 관측 5 틀 = KE 시작 틱부터 1.1 Hz 회전(그 전 틀은 위상 0 — 러너 tick 0 백필과 같음)."""
    ids = ke_act
    ph = 2.0 * math.pi * _GCF * (t_rel - tr_sw[ids]).clamp(min=0.0)
    kk = (torch.arange(5, device=dev, dtype=torch.float32) - 4.0) * (2.0 * math.pi * _GCF * 0.02)
    phk = (ph.unsqueeze(1) + kk.unsqueeze(0)).clamp(min=0.0)
    po[ids, _GC0:_GC1] = torch.stack([torch.sin(phk), torch.cos(phk)], -1).reshape(-1, 10)

def _tr_runner_reset(ids, po):
    """KE 새 러너: 관측 이력 전 항 = 현재 프레임 백필(actions 는 0), EMA 상태(자이로·관절속도·obs_ema·act EMA) 현재값 재시작."""
    for n_, (a0, b0) in _LAYT.items():
        if n_ not in _HLEN:
            continue
        H = _HLEN[n_]; fr = (b0 - a0) // H
        newest = po[ids, b0 - fr:b0].clone()
        if n_ == "actions":
            newest.zero_()
        po[ids, a0:b0] = newest.repeat(1, H)
        buf = _HB[n_]._buffer
        if buf is not None:
            buf[:, ids] = newest.reshape(1, -1, *buf.shape[2:])
    for key_, val_ in (("_s8_gyro_ema", robot.data.root_ang_vel_b), ("_s8_jvel_ema", robot.data.joint_vel), ("_s8_enc_jvel_ema", robot.data.joint_vel)):
        st_ = getattr(uenv, key_, None)
        if st_ is not None and st_.shape[0] == N:
            st_[ids] = val_[ids]
    la_ = getattr(uenv, "_la_ema_st", None)
    if la_ is not None:
        la_["y"][ids] = 0.0
    for (a0, b0, _al) in _OBS_EMA:
        st_ = _OBS_EMA_ST.get(a0)
        if st_ is not None:
            st_[ids] = po[ids, a0:b0]
_DEF = robot.data.default_joint_pos.clone()      # 정책 기본자세(= 액션 오프셋) — 시뮬 홀드 명령각
''')

# (6) 훅: 동결(홀드) 출력
rep('''        out = _chain.step(a)
        if args_cli.cmd_deadband_deg > 0.0:''', '''        out = _chain.step(a)
        if bool(_HOLD[0].any()):                   # 1007 전환 restart: 스택 재기동 동안 stage8 홀드 = 마지막 명령 동결
            out = torch.where(_HOLD[0].unsqueeze(1), _FRZ[0], out)
        if args_cli.cmd_deadband_deg > 0.0:''')

# (7) 루프 머리: 전환 상태기
rep('''        _chain = DeployChain(_DEF, _land, qpos0=_land.clone(), stand=True, **_kw)
        _chain_on[0] = True
''', '''        _chain = DeployChain(_DEF, _land, qpos0=_land.clone(), stand=True, **_kw)
        _chain_on[0] = True
    sw_now = torch.zeros(N, dtype=torch.bool, device=dev)
    with torch.inference_mode():      # 관측·체인·EMA 텐서는 추론 텐서 — 제자리 갱신은 추론 모드 안에서만
        if i >= STEPS_PRE:      # ── 1007 전환 상태기(관측·접촉 비트·IMU 만 사용) ──
            _po0 = obs["policy"]; _ge_now = uenv._ge.clone(); _u_now = uenv._gu.clone()
            _w_obs = _po0[:, _GY1 - 3:_GY1][:, :2].norm(dim=-1); _both_obs = (_po0[:, _FC1 - 2:_FC1] > 0.5).all(dim=-1)
            _walk_on = t_rel >= (float(args_cli.w3_on) + (_eidx % max(1, args_cli.on_phases)).float() / max(1, args_cli.on_phases) * float(args_cli.stop_period))
            if args_cli.ke_mode == "nostop" or args_cli.w3_walk_only:
                _stop_req = torch.zeros(N, dtype=torch.bool, device=dev)
            else:
                _stop_req = t_rel >= tr_tstop
            tr_req = torch.where(_stop_req & torch.isnan(tr_req), torch.full_like(tr_req, t_rel), tr_req)
            tr_u_stop = torch.where(_stop_req & torch.isnan(tr_u_stop), _u_now, tr_u_stop)
            tr_want = _walk_on & ~_stop_req & ~ke_act
            tr_cmdv = torch.where(tr_want.unsqueeze(1), torch.tensor(_W3CMD, device=dev).unsqueeze(0).expand(N, 3), torch.zeros(N, 3, device=dev))
            uenv._tr_want = tr_want; uenv._tr_cmd = tr_cmdv
            _geoff_now = _stop_req & ~_ge_now & ge_prev & torch.isnan(tr_geoff)
            tr_geoff = torch.where(_geoff_now, torch.full_like(tr_geoff, t_rel), tr_geoff)
            _pend = ~torch.isnan(tr_geoff) & torch.isnan(tr_settle)
            if args_cli.settle == "quiet":
                settle_run = torch.where(_pend & _both_obs & (_w_obs < float(args_cli.sq_w)), settle_run + 0.02, torch.zeros_like(settle_run))
                _settle_now = _pend & ((settle_run >= float(args_cli.sq_hold) - 1e-6) | ((t_rel - tr_geoff) >= float(args_cli.settle_max) - 1e-6))
                tr_settle_to = tr_settle_to | (_settle_now & (settle_run < float(args_cli.sq_hold) - 1e-6))
            else:
                _settle_now = _pend & ((t_rel - tr_geoff) >= float(args_cli.settle_s) - 1e-6)
            tr_settle = torch.where(_settle_now, torch.full_like(tr_settle, t_rel), tr_settle)
            if args_cli.ke_mode == "restart":
                if bool(_settle_now.any()):
                    hold_mask[_settle_now] = True; hold_until[_settle_now] = t_rel + float(args_cli.hold_s)
                    _FRZ[0][_settle_now] = (_LAST_CMD[0] if _LAST_CMD[0] is not None else _DEF)[_settle_now]
                sw_now = hold_mask & (t_rel >= hold_until - 1e-6) & ~ke_act
            elif args_cli.ke_mode in ("reset", "direct"):
                sw_now = _settle_now & ~ke_act
            elif args_cli.ke_mode == "nostop":
                sw_now = (t_rel >= tr_tstop) & ~ke_act & alive
                tr_geoff = torch.where(sw_now, torch.full_like(tr_geoff, t_rel), tr_geoff); tr_settle = torch.where(sw_now, torch.full_like(tr_settle, t_rel), tr_settle)
                tr_req = torch.where(sw_now, torch.full_like(tr_req, t_rel), tr_req); tr_u_stop = torch.where(sw_now, _u_now, tr_u_stop)
            if bool(sw_now.any()):
                ke_act = ke_act | sw_now; tr_sw = torch.where(sw_now, torch.full_like(tr_sw, t_rel), tr_sw); tr_u_sw = torch.where(sw_now, _u_now, tr_u_sw)
                hold_mask[sw_now] = False; uenv._tr_want = tr_want & ~ke_act; uenv._tr_cmd = torch.where((tr_want & ~ke_act).unsqueeze(1), tr_cmdv, torch.zeros_like(tr_cmdv))
                if args_cli.ke_mode == "nostop":
                    uenv._tr_force_off[sw_now] = True
                if args_cli.ke_mode in ("restart", "reset", "nostop"):
                    _tr_runner_reset(sw_now, _po0)
                if args_cli.ke_mode == "restart" and _chain is not None:
                    _chain.land[sw_now] = _FRZ[0][sw_now]; _chain.reset(sw_now, robot.data.joint_pos)
                # 추적 창: 전환 전 25 틱(링 → 오래된 것부터)
                if _RING:
                    _ord = [(i - _TRW + j) % _TRW for j in range(_TRW)]
                    for k_ in _RING:
                        TRS[k_][sw_now, :_TRW] = _RING[k_][_ord][:, sw_now].transpose(0, 1)
            _HOLD[0] = hold_mask
            ge_prev = _ge_now          # 다음 틱의 '직전 ge'(env.step 뒤 값은 다음 루프 머리에서 읽음)
''')

# (8) 행동: KE 서기 정책 + 홀드 0 + 클록 패치
rep('''        a = torch.zeros(N, J, device=dev) if (policy is None or i < STEPS_PRE) else policy(obs)
''', '''        a = torch.zeros(N, J, device=dev) if (policy is None or i < STEPS_PRE) else policy(obs)
        if i >= STEPS_PRE and bool(ke_act.any()):          # 1007 전환: KE 클록 관측(KE 시작 틱부터 1.1 Hz) + KE 서기 정책
            _tr_patch_clock(obs["policy"], t_rel)
            if policy_ke is not None:
                a = torch.where(ke_act.unsqueeze(1), policy_ke(obs), a)
        if i >= STEPS_PRE and bool(hold_mask.any()):       # 재기동 동안 정책 없음(관측 actions 0)
            a = torch.where(hold_mask.unsqueeze(1), torch.zeros_like(a), a)
''')

# (9) 감지기: KE 시작 뒤에만 무장 + 상태 초기화 + 기준 추종
rep('''        if policy2 is not None and i >= STEPS_PRE:
            _pg = robot.data.projected_gravity_b
''', '''        if policy2 is not None and i >= STEPS_PRE:
            if bool(sw_now.any()):      # 1007 전환: KE 시작 env 감지기 상태 초기화(배포 = 감지기 새로 생성)
                sw_mode[sw_now] = False; sw_calm[sw_now] = 0.0; sw_t_on[sw_now] = 0.0; early_hold[sw_now] = 0.0
                if "eg_pt" in dir():
                    eg_pt[sw_now] = 0.0; eg_pw[sw_now] = 0.0
                if "refr_t" in dir():
                    refr_t[sw_now] = -1e9
            _pg = robot.data.projected_gravity_b
''')
rep('''                    if "_DF_G" not in dir() or i == STEPS_PRE:
                        _DF_G, _DF_W = _pg_det.clone(), _gy_det.clone()
''', '''                    if "_DF_G" not in dir() or i == STEPS_PRE:
                        _DF_G, _DF_W = _pg_det.clone(), _gy_det.clone()
                    if bool(sw_now.any()):
                        _DF_G[sw_now] = _pg_det[sw_now]; _DF_W[sw_now] = _gy_det[sw_now]
''')
rep('''                if i == STEPS_PRE or "sw_ref" not in dir():
                    sw_ref = _rp2.clone()
                if t_rel < float(args_cli.sw_ref_init_s):
                    sw_ref = _rp2.clone()
''', '''                if i == STEPS_PRE or "sw_ref" not in dir():
                    sw_ref = _rp2.clone()
                _follow = (~ke_act) | ((t_rel - tr_sw) < float(args_cli.sw_ref_init_s))   # 1007 전환: KE 시작 뒤 init_s 동안 기준 추종
                sw_ref = torch.where(_follow.unsqueeze(1), _rp2, sw_ref)
''')
rep('''            _on = (_tilt > _SW_ON[0]) | (_w > _SW_ON[1])
''', '''            _on = ((_tilt > _SW_ON[0]) | (_w > _SW_ON[1])) & ke_act      # 1007 전환: KE 시작 전(W3 구간) 감지기 비무장
''')
rep('''            _new_on = (~sw_mode) & _on
            swc_n += (_new_on & alive).float(); sw_mode = (sw_mode | _new_on) & ~_swoff
''', '''            _new_on = (~sw_mode) & _on
            swc_n += (_new_on & alive).float(); sw_mode = (sw_mode | _new_on) & ~_swoff
            ke_first_on = torch.where(_new_on & alive & torch.isnan(ke_first_on), t_rel - tr_sw, ke_first_on)
            ke_on_pre += (_new_on & alive & ((not dists) or (t_rel < min(d.t0 for d in dists)))).float()
''')

# (10) 틱 지표: 낙상 창·피크·정상 자세 복귀·걷기 이륙
rep('''    death_step[newly] = i; pitch_death[newly] = pitch[newly]; roll_death[newly] = roll[newly]
''', '''    death_step[newly] = i; pitch_death[newly] = pitch[newly]; roll_death[newly] = roll[newly]
    if i >= STEPS_PRE:      # ── 1007 전환 지표 ──
        _ks = t_rel - tr_sw
        fell_walk |= newly & torch.isnan(tr_geoff) & torch.isnan(tr_sw)
        fell_win |= newly & ~torch.isnan(tr_geoff) & (torch.isnan(tr_sw) | (_ks < 0))
        fell_sw2 |= newly & (_ks >= 0) & (_ks < 2.0); fell_sw5 |= newly & (_ks >= 0) & (_ks < 5.0)
        t_fall_sw = torch.where(newly & (_ks >= 0), _ks, t_fall_sw)
        _ok2 = a_ & ~term & (_ks >= 0) & (_ks < 2.0)
        pk_p2 = torch.where(_ok2, torch.maximum(pk_p2, pitch.abs()), pk_p2); pk_r2 = torch.where(_ok2, torch.maximum(pk_r2, roll.abs()), pk_r2); pk_w2 = torch.where(_ok2, torch.maximum(pk_w2, w_xy), pk_w2)
        _okw = a_ & ~term & ~torch.isnan(tr_geoff) & (torch.isnan(tr_sw) | (_ks < 0))
        pk_pw = torch.where(_okw, torch.maximum(pk_pw, pitch.abs()), pk_pw); pk_rw = torch.where(_okw, torch.maximum(pk_rw, roll.abs()), pk_rw); pk_ww = torch.where(_okw, torch.maximum(pk_ww, w_xy), pk_ww)
        _dqn_ = torch.rad2deg((robot.data.joint_pos - robot.data.default_joint_pos).abs()).mean(-1)
        _nrm_ = a_ & ~term & (pitch.abs() < 3.0) & (roll.abs() < 3.0) & (w_xy < 0.2) & both & (_dqn_ < 5.0)
        nrm_sw_hold = torch.where(_nrm_ & (_ks >= 0), nrm_sw_hold + 0.02, torch.zeros_like(nrm_sw_hold))
        nrm_sw_t = torch.where((nrm_sw_hold >= 0.5 - 1e-6) & torch.isnan(nrm_sw_t) & (_ks >= 0), _ks, nrm_sw_t)
''')
rep('''    lift_total += lift
''', '''    lift_total += lift
    walk_lifts += lift * ge_prev.float()
''')

# (11) 루프 끝: 불연속·링·추적
rep('''    if prev_cmd is not None: prev_dcmd = (cmd - prev_cmd).clone()
    prev_cmd = cmd.clone(); prev_a = a.clone(); prev_both = both.clone()
''', '''    if i >= STEPS_PRE:      # ── 1007 전환: 불연속(전환 −1~+10 틱)·틱 추적 ──
        _kk = torch.where(torch.isnan(tr_sw), torch.full_like(tr_sw, -999.0), (t_rel - tr_sw) * 50.0).round().long()
        if prev_a is not None and prev_cmd is not None:
            _da = (a - prev_a).abs().max(1).values; _dc = torch.rad2deg((cmd - prev_cmd).abs().max(1).values)
            _inw = (_kk >= -1) & (_kk <= 10) & a_
            da_sw = torch.where(_inw, torch.maximum(da_sw, _da), da_sw); dcmd_sw = torch.where(_inw, torch.maximum(dcmd_sw, _dc), dcmd_sw)
            cmd_jump0 = torch.where((_kk == 0) & a_, _dc, cmd_jump0)
        _cur = {"a": a, "cmd": cmd, "q": robot.data.joint_pos, "pitch": pitch, "roll": roll, "w": w_xy, "v": v_xy,
                "both": both.float(), "ge": uenv._ge.float(), "mode": (sw_mode.float() if policy2 is not None else torch.zeros(N, device=dev)), "alive": (a_ & ~term).float()}
        if not _RING:
            for k_, v_ in _cur.items():
                _RING[k_] = torch.zeros((_TRW, N) + tuple(v_.shape[1:]), device=dev)
                TRS[k_] = torch.full((N, _TRW + _TRN + 1) + tuple(v_.shape[1:]), float("nan"), device=dev)
        _sel = (_kk >= 0) & (_kk <= _TRN)
        if bool(_sel.any()):
            _ri = torch.nonzero(_sel).squeeze(1)
            for k_, v_ in _cur.items():
                TRS[k_][_ri, _TRW + _kk[_ri]] = v_[_ri].to(TRS[k_].dtype)
        for k_, v_ in _cur.items():
            _RING[k_][i % _TRW] = v_
    if prev_cmd is not None: prev_dcmd = (cmd - prev_cmd).clone()
    prev_cmd = cmd.clone(); prev_a = a.clone(); prev_both = both.clone()
''')

# (12) 요약 출력
rep('''P(f"env {N} · 사전 검사(홀드) {T_pre:.1f}s → 인계 → {T_after:.1f}s | DR''', '''if TRMODE:
    _m = pre_pass; _nm = int(_m.sum())
    def _q(x, mask, p=0.5):
        v = x[mask & ~torch.isnan(x)]
        return float(v.quantile(p)) if v.numel() else float("nan")
    _cnt = lambda b: int((b & _m).sum())
    _ge0 = ~torch.isnan(tr_geoff); _st = ~torch.isnan(tr_settle); _ke = ~torch.isnan(tr_sw)
    P(f"[전환 {args_cli.ke_mode}] 명령 {_W3CMD} ge 켬 {args_cli.w3_on}s · stop 요청 {args_cli.w3_stop}s+{args_cli.stop_phases}위상 · 정착 {args_cli.settle}(|ω|<{args_cli.sq_w} {args_cli.sq_hold}s 상한 {args_cli.settle_max}s) · hold {args_cli.hold_s}s"
      f" | 사전검사 통과 {_nm}: stop 전 이륙 평균 {float(walk_lifts[_m].mean()) if _nm else float('nan'):.1f} · ge 0 도달 {_cnt(_ge0)} (요청→ge0 중앙 {_q(tr_geoff - tr_req, _m):.2f}s)"
      f" · 정착 {_cnt(_st)} (ge0→정착 중앙 {_q(tr_settle - tr_geoff, _m):.2f}s p90 {_q(tr_settle - tr_geoff, _m, 0.9):.2f}, 상한 초과 {_cnt(tr_settle_to)}) · KE 시작 {_cnt(_ke)} (t_sw 중앙 {_q(tr_sw, _m):.2f}s)")
    P(f"  낙상(통과 env): 걷기 중 {_cnt(fell_walk)} · ge0~KE 창 {_cnt(fell_win)} · KE 뒤 2 s {_cnt(fell_sw2)} · KE 뒤 5 s {_cnt(fell_sw5)} · 전체 인계 뒤 {int((fell_post & _m).sum())}")
    _sv2 = _m & _ke & ~fell_sw2
    P(f"  과도(KE 뒤 2 s, 생존 중앙/p95): |피치| {_q(pk_p2, _sv2):.1f}/{_q(pk_p2, _sv2, 0.95):.1f}° · |롤| {_q(pk_r2, _sv2):.1f}/{_q(pk_r2, _sv2, 0.95):.1f}° · |ω| {_q(pk_w2, _sv2):.2f}/{_q(pk_w2, _sv2, 0.95):.2f}"
      f" | 창(ge0~KE): |피치| {_q(pk_pw, _m & _ge0):.1f}/{_q(pk_pw, _m & _ge0, 0.95):.1f}° · |ω| {_q(pk_ww, _m & _ge0):.2f}/{_q(pk_ww, _m & _ge0, 0.95):.2f}")
    P(f"  불연속(전환 −1~+10 틱, 중앙/최대): |Δaction| {_q(da_sw, _sv2):.3f}/{float(da_sw[_sv2].max()) if _sv2.any() else float('nan'):.3f} · |Δ목표| {_q(dcmd_sw, _sv2):.2f}/{float(dcmd_sw[_sv2].max()) if _sv2.any() else float('nan'):.2f}°/틱"
      f" · 첫 KE 틱 목표 점프 {_q(cmd_jump0, _sv2):.2f}/{float(cmd_jump0[_sv2 & ~torch.isnan(cmd_jump0)].max()) if (_sv2 & ~torch.isnan(cmd_jump0)).any() else float('nan'):.2f}°")
    P(f"  KE 감지기: 시작 뒤 외란 전 켜짐 env {100*float((ke_on_pre[_m & _ke] > 0).float().mean()) if (_m & _ke).any() else float('nan'):.1f} % · 첫 켜짐(KE 기준) 중앙 {_q(ke_first_on, _m):.2f}s"
      f" · 정상 자세 복귀(KE 기준) 중앙 {_q(nrm_sw_t, _m):.2f}s p90 {_q(nrm_sw_t, _m, 0.9):.2f}s · 5 s 안 {100*float(((nrm_sw_t <= 5.0) & _m & _ke).sum())/max(1, int((_m & _ke).sum())):.1f} %")
P(f"env {N} · 사전 검사(홀드) {T_pre:.1f}s → 인계 → {T_after:.1f}s | DR''')

# (13) CSV
rep('''+ (["c2_req_n", "c2_timeout_n", "c2_cancel_n", "c2_latch_n", "c2_first_req_timeout"] if C2MODE else []))''',
    '''+ (["c2_req_n", "c2_timeout_n", "c2_cancel_n", "c2_latch_n", "c2_first_req_timeout"] if C2MODE else []) + TRCOLS)''')
rep('''        if C2MODE:
            row += [float(c2_req_n[e]), float(c2_to_n[e]), float(c2_cancel_n[e]), float(c2_latch_n[e]), int(c2_cue_timeout[e])]
        wcsv.writerow(row)''', '''        if C2MODE:
            row += [float(c2_req_n[e]), float(c2_to_n[e]), float(c2_cancel_n[e]), float(c2_latch_n[e]), int(c2_cue_timeout[e])]
        row += [_fx(tr_req[e]), _fx(tr_geoff[e]), _fx(tr_settle[e]), int(tr_settle_to[e]), _fx(tr_sw[e]), _fx(tr_u_stop[e], 3), _fx(tr_u_sw[e], 3), float(walk_lifts[e]),
                int(fell_walk[e]), int(fell_win[e]), int(fell_sw2[e]), int(fell_sw5[e]), _fx(t_fall_sw[e]), _fx(pk_p2[e]), _fx(pk_r2[e]), _fx(pk_w2[e], 3), _fx(pk_pw[e]), _fx(pk_rw[e]), _fx(pk_ww[e], 3),
                _fx(da_sw[e], 4), _fx(dcmd_sw[e], 3), _fx(cmd_jump0[e], 3), _fx(ke_first_on[e]), float(ke_on_pre[e]), _fx(nrm_sw_t[e])]
        wcsv.writerow(row)''')
rep('''import csv  # noqa: E402
''', '''import csv  # noqa: E402
TRCOLS = ["tr_req", "tr_geoff", "tr_settle", "tr_settle_to", "tr_sw", "tr_u_stop", "tr_u_sw", "walk_lifts", "fell_walk", "fell_win", "fell_sw2", "fell_sw5", "t_fall_sw",
          "pk_p2", "pk_r2", "pk_w2", "pk_pw", "pk_rw", "pk_ww", "da_sw", "dcmd_sw", "cmd_jump0", "ke_first_on", "ke_on_pre", "nrm_sw_t"]
def _fx(v, nd=2):
    v = float(v)
    return "" if math.isnan(v) else round(v, nd)
''')

# (14) 추적 npz
rep('''if C2MODE and _C2TR is not None:''', '''if TRMODE and TRS:
    import numpy as _npt
    _npt.savez_compressed(os.path.join(DATA, "log_picture", f"trans_{args_cli.out_tag}.npz"), k0=_TRW, dt=0.02, joint_names=_npt.array(jn),
                          **{k_: v_.cpu().numpy() for k_, v_ in TRS.items()},
                          tr_req=tr_req.cpu().numpy(), tr_geoff=tr_geoff.cpu().numpy(), tr_settle=tr_settle.cpu().numpy(), tr_sw=tr_sw.cpu().numpy(),
                          tr_u_stop=tr_u_stop.cpu().numpy(), tr_u_sw=tr_u_sw.cpu().numpy(), pre_pass=pre_pass.cpu().numpy(), fell=fell.cpu().numpy(),
                          t_fall_post=t_fall_post.cpu().numpy(), ke_mode=args_cli.ke_mode, dist=args_cli.dist, args=" ".join(sys.argv[1:]))
    print(f"[trans] log_picture/trans_{args_cli.out_tag}.npz", flush=True)
if C2MODE and _C2TR is not None:''')

open(DST, "w", encoding="utf-8").write(src)
print("written", DST, len(src.splitlines()), "lines")
