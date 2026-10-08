"""9 단계 패치(1007): 러너 루프 평가기(make_eval_transition_kew3.py 적용 뒤)에 브리지의 헤딩 고리를 넣는다 — HEADING_SESSION_1007.md HD1–HD3.
- 고리 코드는 배포본 그대로: 브리지 `BridgeCore._w3_cmd_step`(w3_deploy.W3HeadingLoop 호출·회전 명령 시 해제·고장 시 걷기 정지)을 최소 대역 객체로 직접 부른다.
- 요 측정 = 참 쿼터니언에 브리지 `yaw_from_quat_wxyz` 를 적용한 값 + (선택) 표류 주입(--kw_yaw_drift °/min, env 짝수 +·홀수 −).
- --kw_hl ''(끔, 기본 = 종전과 같음) | 'on'(브리지 기본 설정) | JSON(parse_w3_heading_loop).  --kw_hl_e0 'a,b,...' = 래치 직후 ψ* 에 더할 초기 오차[°], env i 는 목록의 i % len 번째(HD2).
- 고리 고장이 나면 그 걷기 구간은 재시도하지 않는다(대본의 다음 걷기 사건에서 다시 켠다). 기록: hlerr(°)·wzc(rad/s)·hl_nfault·hl_e0."""
import os
P = os.environ.get("TR_DST", "rl_walking/scripts/eval_transition_kwh.py")
src = open(P, encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    n = src.count(old); assert n == count, f"앵커 {n}회: {old[:90]!r}"
    src = src.replace(old, new)

rep('''parser.add_argument("--kw_requiet", type=str, default="",''', '''parser.add_argument("--kw_hl", type=str, default="", help="1007 HD: 헤딩 고리 — '' 끔 | 'on' 브리지 기본 | JSON")
parser.add_argument("--kw_hl_e0", type=str, default="", help="HD2: 래치 직후 ψ* 초기 오차[°] 목록(env i → i % len)")
parser.add_argument("--kw_yaw_drift", type=float, default=0.0, help="HD3: 고리에 주는 요 측정에 더할 표류[°/min](env 짝수 +, 홀수 −)")
parser.add_argument("--kw_requiet", type=str, default="",''')
# 1007 X7c: 정착 상한(초) 인자 — 아주 크게 주면(정착 시간도 함께) KE 로 넘기지 않고 W3 STAND 로 머문다(걷기 사이를 W3 서기로 잇는 대조 팔)
rep('''parser.add_argument("--kw_hl", type=str, default="",''', '''parser.add_argument("--kw_settle_max", type=float, default=3.0, help="W3 → KE 정착 상한 [s](기본 3.0 = 종전). 1e9 = 넘기지 않음")
parser.add_argument("--kw_hl", type=str, default="",''')
rep('''handoff=args_cli.kw_handoff, settle_quiet_s=float(args_cli.kw_settle))''', '''handoff=args_cli.kw_handoff, settle_quiet_s=float(args_cli.kw_settle), settle_max_s=float(args_cli.kw_settle_max))''')
rep('''    kw_cur = [(0.0, 0.0, 0.0)] * N; kw_seq = [1] * N;''', '''    _KW_HLCFG = (_W3D.parse_w3_heading_loop(None if args_cli.kw_hl.strip().lower() == "on" else args_cli.kw_hl) if args_cli.kw_hl else None)

    class _KwHLShim:      # BridgeCore._w3_cmd_step 이 읽고 쓰는 속성만 가진 대역(같은 이름) — 고리·고장 처리 코드는 브리지 것을 그대로 실행
        def __init__(self, runner):
            self.runner = runner; self.w3mon = None; self.w3hf_ban = None; self.w3hl = _W3D.W3HeadingLoop(_KW_HLCFG); self.w3hl_on = True; self.w3hl_events = []; self.cmd = (0.0, 0.0, 0.0)
    KWH = [_KwHLShim(KWR[e_]) for e_ in range(N)] if _KW_HLCFG is not None else None
    _KW_E0 = [float(x) for x in args_cli.kw_hl_e0.split(",")] if args_cli.kw_hl_e0 else [0.0]
    kw_hl_nfault = [0] * N; kw_hlerr_tl = []; kw_wzc_tl = []
    if KWH is not None:
        print(f"[runner_loop] 헤딩 고리 켬(브리지 _w3_cmd_step) {_KW_HLCFG} · 초기 오차 목록 {_KW_E0}° · 요 표류 주입 {args_cli.kw_yaw_drift}°/min", flush=True)
    kw_cur = [(0.0, 0.0, 0.0)] * N; kw_seq = [1] * N;''')
rep('''        po_np = po.cpu().numpy(); kw_tick[0] += 1''', '''        po_np = po.cpu().numpy(); kw_tick[0] += 1
        _qw_np = robot.data.root_quat_w.cpu().numpy().astype(_npk.float64) if KWH is not None else None
        _hlerr = _npk.full(N, _npk.nan, _npk.float32); _wzc = _npk.zeros(N, _npk.float32); _hl_fault_now = _npk.zeros(N, bool)''')
rep('''"heading_loop": False, "req_id": "kw", "seq": kw_seq[e]}''', '''"heading_loop": bool(KWH is not None), "req_id": "kw", "seq": kw_seq[e]}''', count=2)
rep('''            R.step(gy[e], pg[e], kw_cur[e], qp[e], jv[e], ct[e])
''', '''            _cmd_e = kw_cur[e]
            if KWH is not None:
                _H = KWH[e]
                _psi = float(_BR.yaw_from_quat_wxyz(*_qw_np[e])) + math.radians(float(args_cli.kw_yaw_drift)) / 60.0 * now * (1.0 if e % 2 == 0 else -1.0)
                _nev = len(_H.w3hl_events); _lat0 = _H.w3hl.latched
                _cmd_e = _BR.BridgeCore._w3_cmd_step(_H, (math.cos(_psi / 2.0), 0.0, 0.0, math.sin(_psi / 2.0)), float(kw_cur[e][0]), float(kw_cur[e][1]), float(kw_cur[e][2]))
                if _H.w3hl.latched and not _lat0 and _KW_E0[e % len(_KW_E0)] != 0.0:      # HD2: 래치 직후 ψ* 에 초기 오차(다음 틱부터 반영)
                    _H.w3hl.tgt = _W3D.wrap_pi(_H.w3hl.tgt + math.radians(_KW_E0[e % len(_KW_E0)]))
                if len(_H.w3hl_events) > _nev:      # 고리 고장 → 브리지가 러너 want 0·명령 0 으로 걷기를 멈춤. 이 구간은 재시도하지 않는다
                    kw_cur[e] = (0.0, 0.0, 0.0); kw_events[e].append((round(now, 2), "hl_fault")); kw_hl_nfault[e] += 1; _hl_fault_now[e] = True
                if _H.w3hl.latched and _H.w3hl.err is not None:
                    _hlerr[e] = math.degrees(_H.w3hl.err)
                _wzc[e] = float(_cmd_e[2])
            R.step(gy[e], pg[e], _cmd_e, qp[e], jv[e], ct[e])
''')
rep('''        kind_t = torch.as_tensor(kind, device=dev); ge_t = torch.as_tensor(ge_top, device=dev);''', '''        if KWH is not None:
            kw_hlerr_tl.append(_hlerr.astype(_npk.float16)); kw_wzc_tl.append(_wzc.astype(_npk.float16))
            if _hl_fault_now.any():
                kw_des = kw_des & ~torch.as_tensor(_hl_fault_now, device=dev)
        kind_t = torch.as_tensor(kind, device=dev); ge_t = torch.as_tensor(ge_top, device=dev);''')
rep('''                          pg=_npk.stack(kw_pg_tl, 0)''', '''                          hl=str(args_cli.kw_hl), hl_e0=_npk.asarray([_KW_E0[e_ % len(_KW_E0)] for e_ in range(N)], _npk.float32), hl_nfault=_npk.asarray(kw_hl_nfault, _npk.int16), yaw_drift=float(args_cli.kw_yaw_drift),
                          hlerr=_npk.stack(kw_hlerr_tl, 0) if kw_hlerr_tl else _npk.zeros((0, N), _npk.float16), wzc=_npk.stack(kw_wzc_tl, 0) if kw_wzc_tl else _npk.zeros((0, N), _npk.float16),
                          pg=_npk.stack(kw_pg_tl, 0)''')
# 1007 X7d: 자세 되돌리기 개입(시뮬 전용 진단) — 지정 시각에 로봇의 물리 상태(몸통 위치·자세·관절각, 속도 0)와 출력 체인의 목표 기억을 스냅샷 시각(--kw_restore_snap) 상태로 되돌린다.
#   러너(이력·필터·직전 행동)·감지기·감독기는 건드리지 않는다. 출력 체인(슬루·저역 필터 기억)을 함께 되돌리는 이유: 그대로 두면 PD 가 곧바로 이전 자세의 목표로 끌어당겨 개입이 무효가 된다.
rep('''parser.add_argument("--kw_hl", type=str, default="",''', '''parser.add_argument("--kw_restore_at", type=str, default="", help="X7d: 자세 되돌리기 시각 목록 't1,t2,...'(인계 기준 s). '' = 끔")
parser.add_argument("--kw_restore_snap", type=float, default=1.5, help="X7d: 되돌릴 기준 상태를 찍는 시각 [s]")
parser.add_argument("--kw_hl", type=str, default="",''')
rep('''        po = obs["policy"]
        if i == STEPS_PRE:
            for R in KWR:
                R.reset()
''', '''        po = obs["policy"]
        if i == STEPS_PRE:
            for R in KWR:
                R.reset()
        if _KW_RST:      # X7d 자세 되돌리기(시뮬 전용)
            if _KW_SNAP[0] is None and t_rel >= float(args_cli.kw_restore_snap) - 1e-9:
                _KW_SNAP[0] = (torch.cat([robot.data.root_pos_w, robot.data.root_quat_w], -1).clone(), robot.data.joint_pos.clone(),
                               (_LAST_CMD[0].clone() if _LAST_CMD[0] is not None else None))
            for k_, tr_ in enumerate(_KW_RST):
                if (not _KW_RST_DONE[k_]) and _KW_SNAP[0] is not None and t_rel >= tr_ - 1e-9:
                    _KW_RST_DONE[k_] = True
                    ids_ = torch.nonzero(alive, as_tuple=False).flatten()
                    if len(ids_):
                        robot.write_root_pose_to_sim(_KW_SNAP[0][0][ids_], ids_); robot.write_root_velocity_to_sim(torch.zeros(len(ids_), 6, device=dev), ids_)
                        robot.write_joint_state_to_sim(_KW_SNAP[0][1][ids_], torch.zeros_like(_KW_SNAP[0][1][ids_]), env_ids=ids_)
                        if _chain is not None and _KW_SNAP[0][2] is not None:
                            _chain.prev[ids_] = _KW_SNAP[0][2][ids_]; _chain.filt[ids_] = _KW_SNAP[0][2][ids_]
                    kw_rst_log.append((round(float(t_rel), 2), int(len(ids_))))
                    print(f"[runner_loop] X7d 자세 되돌리기 t {float(t_rel):.2f}s · env {int(len(ids_))} 개", flush=True)
''')
rep('''    kw_cur = [(0.0, 0.0, 0.0)] * N; kw_seq = [1] * N;''', '''    _KW_RST = [float(x) for x in args_cli.kw_restore_at.split(",")] if args_cli.kw_restore_at else []
    _KW_RST_DONE = [False] * len(_KW_RST); _KW_SNAP = [None]; kw_rst_log = []
    kw_cur = [(0.0, 0.0, 0.0)] * N; kw_seq = [1] * N;''')
rep('''                          hl=str(args_cli.kw_hl),''', '''                          restore=json.dumps(kw_rst_log), hl=str(args_cli.kw_hl),''')
# 1007 X7e 준비: 개체별 무작위화 값 덤프(기록 전용) — 인계 시점에 env 별 영점 편향(관절·IMU)·관절 게인/마찰/관성·질량을 log_picture/drdump_<tag>.npz 로 저장.
rep('''        po = obs["policy"]
        if i == STEPS_PRE:
            for R in KWR:
                R.reset()
''', '''        po = obs["policy"]
        if i == STEPS_PRE:
            for R in KWR:
                R.reset()
            try:
                _dd = dict(joint_names=_npk.array(list(robot.data.joint_names)))
                for _k, _v in (("calib_joint_bias", getattr(uenv, "_calib_joint_bias", None)), ("calib_imu_tilt", getattr(uenv, "_calib_imu_tilt", None)), ("calib_imu_roll", getattr(uenv, "_calib_imu_roll", None)),
                               ("imu_lag", getattr(uenv, "_imu_lag", None)), ("stiffness", getattr(robot.data, "joint_stiffness", None)), ("damping", getattr(robot.data, "joint_damping", None)),
                               ("friction", getattr(robot.data, "joint_friction_coeff", None)), ("armature", getattr(robot.data, "joint_armature", None)),
                               ("default_joint_pos", robot.data.default_joint_pos), ("nominal_default_joint_pos", getattr(uenv, "_nominal_default_joint_pos", None)),
                               ("joint_pos0", robot.data.joint_pos), ("root_quat0", robot.data.root_quat_w)):
                    if _v is not None:
                        _dd[_k] = _v.detach().cpu().numpy()
                try:
                    _dd["masses"] = robot.root_physx_view.get_masses().cpu().numpy(); _dd["coms"] = robot.root_physx_view.get_coms().cpu().numpy()
                except Exception as _e_:
                    print(f"[runner_loop] drdump: 질량 조회 실패({_e_})", flush=True)
                _npk.savez_compressed(os.path.join(DATA, "log_picture", f"drdump_{args_cli.out_tag}.npz"), **_dd)
                print(f"[runner_loop] 개체별 무작위화 값 덤프: drdump_{args_cli.out_tag}.npz · 키 {sorted(_dd)}", flush=True)
            except Exception as _e_:
                print(f"[runner_loop] drdump 실패(기록 전용 — 평가는 계속): {_e_}", flush=True)
''')
# 1007 S3E 평가: 합성 러너의 W3 쪽 정책 파일만 다른 내보내기 폴더로 바꾼다(브리지 키·계약·설정은 그대로) — 새 W3 체크포인트를 배포 코드 폐루프로 평가.
rep('''parser.add_argument("--kw_hl", type=str, default="",''', '''parser.add_argument("--kw_w3_export", type=str, default="", help="S3E: W3 하위 키의 exported 폴더 경로 덮어쓰기('' = 등록된 W3S3d-15800)")
parser.add_argument("--kw_hl", type=str, default="",''')
rep('''    KWR = [_KWR(_BR._make_runner(_KW_KEYS[0]), _BR._make_runner(_KW_KEYS[1]),''', '''    if args_cli.kw_w3_export:
        _kwx_ = os.path.abspath(args_cli.kw_w3_export)
        if not (os.path.exists(os.path.join(_kwx_, "policy.pt")) or os.path.exists(os.path.join(_kwx_, "policy.onnx"))):
            raise SystemExit(f"--kw_w3_export: {_kwx_} 에 policy.pt/onnx 없음")
        print(f"[runner_loop] W3 하위 키 {_KW_KEYS[0]} 의 정책 폴더 덮어쓰기: {_BR.CHECKPOINTS[_KW_KEYS[0]]} → {_kwx_} · SOURCE_CHECKPOINT {open(os.path.join(_kwx_, 'SOURCE_CHECKPOINT')).read().strip()[:120] if os.path.exists(os.path.join(_kwx_, 'SOURCE_CHECKPOINT')) else '(없음)'}", flush=True)
        _BR.CHECKPOINTS[_KW_KEYS[0]] = _kwx_
    KWR = [_KWR(_BR._make_runner(_KW_KEYS[0]), _BR._make_runner(_KW_KEYS[1]),''')
# 1007 X13: 토크 한계 민감도 — 인계 시점에 env 그룹별로 관절 토크 한계를 배율로 낮춘다(env i → 목록의 i % len 번째). mdp_stage4.resample_effort_scale 과 같은 방식(PhysX 한계 + 액추에이터 모델 한계 둘 다).
rep('''parser.add_argument("--kw_hl", type=str, default="",''', '''parser.add_argument("--kw_effort_scales", type=str, default="", help="X13: 토크 한계 배율 목록 '1.0,0.8,0.7,0.6'(env i → i % len). '' = 끔")
parser.add_argument("--kw_hl", type=str, default="",''')
rep('''            try:
                _dd = dict(joint_names=_npk.array(list(robot.data.joint_names)))''', '''            if args_cli.kw_effort_scales:
                _es_ = [float(x) for x in args_cli.kw_effort_scales.split(",")]
                _sc_ = torch.tensor([_es_[e_ % len(_es_)] for e_ in range(N)], device=dev, dtype=torch.float32).unsqueeze(1)
                _nom_ = robot.data.joint_effort_limits.clone(); _new_ = _nom_ * _sc_
                robot.write_joint_effort_limit_to_sim(_new_)
                for _act_ in robot.actuators.values():
                    if hasattr(_act_, "effort_limit") and torch.is_tensor(_act_.effort_limit) and _act_.effort_limit.dim() == 2:
                        _act_.effort_limit[:] = _new_[:, _act_.joint_indices]
                kw_effort_scale[:] = _sc_.squeeze(1).cpu().numpy()
                print(f"[runner_loop] X13 토크 한계 배율 {_es_} 적용(env i → i % {len(_es_)}) · 명목 한계 예 {[round(float(x), 1) for x in _nom_[0]]}", flush=True)
            try:
                _dd = dict(joint_names=_npk.array(list(robot.data.joint_names)))''')
rep('''    _KW_RST = [float(x) for x in args_cli.kw_restore_at.split(",")] if args_cli.kw_restore_at else []''', '''    kw_effort_scale = _npk.ones(N, _npk.float32)
    _KW_RST = [float(x) for x in args_cli.kw_restore_at.split(",")] if args_cli.kw_restore_at else []''')
rep('''                          restore=json.dumps(kw_rst_log), hl=str(args_cli.kw_hl),''', '''                          effort_scale=kw_effort_scale, restore=json.dumps(kw_rst_log), hl=str(args_cli.kw_hl),''')
# 1007 X14: 콘솔 걷기 버튼(시간 제한 걷기)의 시뮬 평가 — 대본 사건 'move@t:이름'(w3_deploy.W3_MOVES) → step 켜기에 dur_s 를 실어 감독기에 보내고,
#   기한 끄기는 브리지 노드와 같은 순서로 감독기의 burst_arm/burst_due 가 한다(콘솔 타이머 없음).
rep('''            if kind not in ("walk", "stop"):''', '''            if kind == "move":
                _mc = _W3D.w3_move_cmd(arg.strip(), allow_disabled=True); _KW_DURS[round(float(tt), 4)] = float(_mc["dur_s"])
                ev.append((float(tt), True, [_mc["vx"], _mc["vy"], _mc["wz"]])); continue
            if kind not in ("walk", "stop"):''')
rep('''    def _kw_parse(s):''', '''    _KW_DURS = {}                                     # 1007 X14: 사건 시각 → 기한 [s](move 사건만)
    kw_dur = _npk.full(N, _npk.nan, _npk.float32)     # env 별 이번 걷기 요청의 기한(NaN = 기한 없음)

    def _kw_parse(s):''')
rep('''                kw_des = torch.where(m, torch.full_like(kw_des, bool(w)), kw_des)''', '''                kw_des = torch.where(m, torch.full_like(kw_des, bool(w)), kw_des)
                kw_dur[m.cpu().numpy()] = _KW_DURS.get(round(float(tt), 4), _npk.nan) if w else _npk.nan''')
rep('''"heading_loop": bool(KWH is not None), "req_id": "kw", "seq": kw_seq[e]}''', '''"heading_loop": bool(KWH is not None), "req_id": "kw", "seq": kw_seq[e], **({"dur_s": float(kw_dur[e])} if kw_dur[e] == kw_dur[e] else {})}''', count=2)
rep('''            S.observe(now, ct[e], gy[e], pg[e])''', '''            S.observe(now, ct[e], gy[e], pg[e])
            if S.burst_t_off is not None:                 # 1007 X14: 기한 도달 → 브리지 _w3_burst_end 와 같은 실행(명령 0·gait 0), 이 걷기 요청은 끝
                if S.burst_due(now, bool(clk.want)) is not None:
                    kw_cur[e] = (0.0, 0.0, 0.0); R.set_gait_enable(False); kw_events[e].append((round(now, 2), "burst_end")); _burst_end_now[e] = True
                    des[e] = False                         # 이 걷기 요청은 끝 — 같은 틱에 다시 켜기 요청을 내지 않는다(콘솔은 기한 끝에 요청을 보내지 않음)''')
rep('''                        kw_cur[e] = tuple(dec["cmd"]); R.set_gait_enable(True); en_now[e] = True''', '''                        kw_cur[e] = tuple(dec["cmd"]); R.set_gait_enable(True); en_now[e] = True
                        S.burst_arm(now, d["req_id"], dec.get("dur_s"))''')
rep('''        _hlerr = _npk.full(N, _npk.nan, _npk.float32); _wzc = _npk.zeros(N, _npk.float32); _hl_fault_now = _npk.zeros(N, bool)''', '''        _hlerr = _npk.full(N, _npk.nan, _npk.float32); _wzc = _npk.zeros(N, _npk.float32); _hl_fault_now = _npk.zeros(N, bool)
        _burst_end_now = _npk.zeros(N, bool)''')
rep('''        if KWH is not None:
            kw_hlerr_tl.append(_hlerr.astype(_npk.float16)); kw_wzc_tl.append(_wzc.astype(_npk.float16))''', '''        if _burst_end_now.any():
            kw_des = kw_des & ~torch.as_tensor(_burst_end_now, device=dev)
        if KWH is not None:
            kw_hlerr_tl.append(_hlerr.astype(_npk.float16)); kw_wzc_tl.append(_wzc.astype(_npk.float16))''')
# 1007 X16: 합성 러너의 KE 회복 정책 파일만 다른 내보내기 폴더로 바꾼다(서기 정책·감지기·키·계약은 그대로) — 새 회복 체크포인트를 배포 코드 폐루프로 평가.
rep("""parser.add_argument("--kw_hl", type=str, default="",""", """parser.add_argument("--kw_rec_export", type=str, default="", help="X16: KE 회복 정책의 exported 폴더 경로 덮어쓰기('' = 등록된 G24Ra-21793)")
parser.add_argument("--kw_hl", type=str, default="",""")
rep("""    KWR = [_KWR(_BR._make_runner(_KW_KEYS[0]), _BR._make_runner(_KW_KEYS[1]),""", """    if args_cli.kw_rec_export:
        _krx_ = os.path.abspath(args_cli.kw_rec_export)
        if not (os.path.exists(os.path.join(_krx_, "policy.pt")) or os.path.exists(os.path.join(_krx_, "policy.onnx"))):
            raise SystemExit(f"--kw_rec_export: {_krx_} 에 policy.pt/onnx 없음")
        _st_, _old_ = _BR.C5_PAIRS[_KW_KEYS[1]]
        print(f"[runner_loop] KE 키 {_KW_KEYS[1]} 의 회복 정책 폴더 덮어쓰기: {_old_} → {_krx_} · SOURCE_CHECKPOINT {open(os.path.join(_krx_, 'SOURCE_CHECKPOINT')).read().strip()[:120] if os.path.exists(os.path.join(_krx_, 'SOURCE_CHECKPOINT')) else '(없음)'}", flush=True)
        _BR.C5_PAIRS[_KW_KEYS[1]] = (_st_, _krx_)
    KWR = [_KWR(_BR._make_runner(_KW_KEYS[0]), _BR._make_runner(_KW_KEYS[1]),""")
# 1008 X17: 실물 stage8 의 전 관절 소프트 한계 클램프를 시뮬에 넣는다 — 학습·종전 평가는 발목 앞뒤만 자른다(계약 감사 CONTRACT_AUDIT.md #5).
#   인계 시점에 config/joint_limits_12dof.json 의 soft 한계를 전 관절의 soft_joint_pos_limits 에 쓰고, 액션 항의 클램프 마스크를 전 관절로 넓힌다.
#   (액션 항 SlewJointPositionAction 의 clamp_to_soft_limits 경로 그대로 — 슬루·LPF·블렌딩 뒤 최종 목표각을 자른다 = stage8 이 받은 목표를 자르는 것과 같은 자리.)
rep("""parser.add_argument("--kw_hl", type=str, default="",""", """parser.add_argument("--kw_soft_clamp_all", action="store_true", help="X17: 실물 stage8 처럼 전 관절 목표각을 config soft 한계로 자른다(기본 = 학습과 같이 발목 앞뒤만)")
parser.add_argument("--kw_soft_clamp_json", type=str, default="", help="X17b: --kw_soft_clamp_all 이 읽을 한계 표(기본 = config/joint_limits_12dof.json). 넓힌 soft 한계 제안의 대조 평가용")
parser.add_argument("--kw_hl", type=str, default="",""")
rep("""            if args_cli.kw_effort_scales:
                _es_ = [float(x) for x in args_cli.kw_effort_scales.split(",")]""", """            if args_cli.kw_soft_clamp_all:
                import json as _jsc
                _lim_ = _jsc.load(open(args_cli.kw_soft_clamp_json or os.path.join(REPO, "config", "joint_limits_12dof.json"), encoding="utf-8"))["limits_deg"]
                _nm_ = list(robot.data.joint_names); assert all(n in _lim_ for n in _nm_), [n for n in _nm_ if n not in _lim_]
                _sl_ = torch.tensor([[math.radians(_lim_[n]["soft_min"]), math.radians(_lim_[n]["soft_max"])] for n in _nm_], device=dev, dtype=torch.float32)
                _term_ = uenv.action_manager.get_term("joint_pos")
                assert bool(getattr(_term_, "_clamp_soft", False)), "액션 항의 clamp_to_soft_limits 가 꺼져 있다 — 이 옵션은 그 경로를 쓴다"
                _before_ = int(getattr(uenv, "_soft_clamp_mask", torch.zeros(len(_nm_), dtype=torch.bool)).sum())
                robot.data.soft_joint_pos_limits[:, :, :] = _sl_.unsqueeze(0)
                uenv._soft_clamp_mask = torch.ones(len(_nm_), dtype=torch.bool, device=dev)
                print(f"[runner_loop] X17 전 관절 소프트 한계 클램프 켬(마스크 {_before_} → {len(_nm_)} 관절) · 한계[deg] " + " ".join(f"{n.replace('_joint', '')}[{_lim_[n]['soft_min']:g},{_lim_[n]['soft_max']:g}]" for n in _nm_), flush=True)
            if args_cli.kw_effort_scales:
                _es_ = [float(x) for x in args_cli.kw_effort_scales.split(",")]""")
# 1008 X14b: 버튼 표에 없는 값의 시간 제한 걷기 — 대본 사건 'mv@t:vx,vy,wz,dur_s'(move@ 와 같은 경로: 감독기의 dur_s 검증·burst_arm/burst_due 그대로).
rep("""            if kind == "move":""", """            if kind == "mv":
                _v4 = [float(x) for x in arg.split(",")]; assert len(_v4) == 4, arg
                _KW_DURS[round(float(tt), 4)] = float(_v4[3]); ev.append((float(tt), True, _v4[:3])); continue
            if kind == "move":""")
# 1008 X18: 부분지지(줄이 체중 일부를 받침)의 이상화 — 링크마다 자기 무게의 일정 비율을 위로 준다(= 중력 감소, 회전력 없음. --land_s 와 같은 방식).
#   인계 뒤 1 s 에 걸쳐 0 → 비율로 올리고 끝까지 유지. 줄의 진자 복원력·탄성·느슨해짐은 없다(가정). 외란(--dist)·--osc_inj 와 함께 쓰지 않는다(외력 버퍼를 덮어쓴다).
rep("""parser.add_argument("--kw_soft_clamp_all", action="store_true",""", """parser.add_argument("--kw_unload", type=float, default=0.0, help="X18: 인계 뒤 링크마다 자기 무게의 이 비율(0–0.9)을 위로 받친다(부분지지의 이상화, 1 s 램프). --dist 와 함께 쓰지 않는다")
parser.add_argument("--kw_soft_clamp_all", action="store_true",""")
rep("""    if args_cli.osc_inj:      # 1007 §13b""", """    if float(args_cli.kw_unload) > 0.0 and i >= STEPS_PRE:
        assert 0.0 < float(args_cli.kw_unload) <= 0.9 and not dists and not args_cli.osc_inj, "--kw_unload 는 0–0.9, --dist·--osc_inj 없이"
        if "_UNL_M" not in globals():
            globals()["_UNL_M"] = robot.root_physx_view.get_masses().to(dev)          # (N, B)
            print(f"[kw_unload] 부분지지 {100 * float(args_cli.kw_unload):.0f} % (인계 뒤 1 s 램프) · 전체 질량 {float(_UNL_M[0].sum()):.2f} kg → 받치는 힘 {float(_UNL_M[0].sum()) * 9.81 * float(args_cli.kw_unload):.1f} N", flush=True)
        _ufr = float(args_cli.kw_unload) * min(1.0, max(0.0, float(t_rel)) / 1.0)
        _uB = _UNL_M.shape[1]
        _ufw = torch.zeros(N, _uB, 3, device=dev); _ufw[..., 2] = _UNL_M * 9.81 * _ufr
        _ufb = quat_apply_inverse(robot.data.body_quat_w.reshape(-1, 4), _ufw.reshape(-1, 3)).reshape(N, _uB, 3)
        robot.set_external_force_and_torque(_ufb, torch.zeros_like(_ufb))
    if args_cli.osc_inj:      # 1007 §13b""")
# 1008 X19: 접촉 관측 교란(정책·감독기·감지기가 보는 접촉 비트만 — 물리 접촉·채점 불변). --contact_drop(누락·고착 0) 뒤에 한 겹 더 씌운다.
#   'delay:N' = 비트를 N 정책 틱 더 늦게 · 'flip:p' = 틱마다 각 발 비트를 확률 p 로 뒤집음(채터·오판, 고정 난수) · 't0:x' = 인계 뒤 x s 부터만 적용(기본 0).
rep("""parser.add_argument("--kw_soft_clamp_all", action="store_true",""", """parser.add_argument("--contact_pert", type=str, default="", help="X19: 접촉 관측 교란 'delay:N;flip:p;t0:x'(관측 비트만, 물리·채점 불변). '' = 끔")
parser.add_argument("--kw_soft_clamp_all", action="store_true",""")
rep("""    print(f"[contact_drop] 정책 접촉 비트 강제 0: {args_cli.contact_drop}", flush=True)
""", """    print(f"[contact_drop] 정책 접촉 비트 강제 0: {args_cli.contact_drop}", flush=True)
if args_cli.contact_pert:
    _CP = {"delay": 0, "flip": 0.0, "t0": 0.0}
    for _it in args_cli.contact_pert.split(";"):
        _k, _v = _it.split(":"); assert _k.strip() in _CP, _k; _CP[_k.strip()] = float(_v)
    _CP["delay"] = int(_CP["delay"]); assert 0 <= _CP["delay"] <= 10 and 0.0 <= _CP["flip"] <= 0.5
    _fcp = getattr(env_cfg.observations.policy, "feet_contact", None)
    if _fcp is None:
        raise SystemExit("--contact_pert: 정책 관측에 feet_contact 항 없음")
    _fcp_orig = _fcp.func; _CPS = {"buf": None, "gen": None, "n": 0, "nflip": 0}

    import inspect as _insp_cp
    _fcp_names = set(_insp_cp.signature(_fcp_orig).parameters) - {"env"}

    def _fc_pert(env, sensor_cfg=None, threshold: float = 5.0, delay_steps: int = 0, threshold_range=None,
                 hold_ticks=None, extra_delay=None, tilt_off_deg=None):      # 매니저가 인자 이름을 검사하므로 --contact_drop 포장과 같은 꼴로 받는다
        _kw = dict(sensor_cfg=sensor_cfg, threshold=threshold, delay_steps=delay_steps, threshold_range=threshold_range,
                   hold_ticks=hold_ticks, extra_delay=extra_delay, tilt_off_deg=tilt_off_deg)
        out = _fcp_orig(env, **{k: v for k, v in _kw.items() if k in _fcp_names and v is not None}).clone()
        if _CPS["gen"] is None:
            _CPS["gen"] = torch.Generator(device="cpu"); _CPS["gen"].manual_seed(int(args_cli.seed) * 1000 + 19)
            _CPS["buf"] = [out.clone() for _ in range(_CP["delay"] + 1)]
        _CPS["buf"].append(out.clone()); _CPS["buf"].pop(0)
        t_rel = env.episode_length_buf.to(out.dtype) * float(env.step_dt) - float(args_cli.precheck_s)
        act = (t_rel >= _CP["t0"]).unsqueeze(-1)
        y = _CPS["buf"][0] if _CP["delay"] > 0 else out
        if _CP["flip"] > 0.0:
            fl = (torch.rand(tuple(out.shape), generator=_CPS["gen"]) < _CP["flip"]).to(out.device)
            _CPS["nflip"] += int((fl & act).sum()); y = torch.where(fl, 1.0 - (y > 0.5).to(out.dtype), y)
        _CPS["n"] += 1
        return torch.where(act, y, out)
    _fcp.func = _fc_pert
    print(f"[contact_pert] 접촉 관측 교란 {_CP} (관측 비트만 — 물리·채점 불변, --contact_drop 뒤에 적용)", flush=True)
""")
# 1008 X19b: 감독기 문턱 덮어쓰기 — 브리지의 --w3-safety 와 같은 파서(w3_deploy.parse_w3_safety, 범위 검증 그대로). '' = 브리지 기본값.
rep("""parser.add_argument("--kw_soft_clamp_all", action="store_true",""", """parser.add_argument("--kw_safety", type=str, default="", help="X19b: W3 감독기 문턱 JSON(브리지 --w3-safety 와 같은 형식·범위 검증). '' = 기본값")
parser.add_argument("--kw_soft_clamp_all", action="store_true",""")
rep("""    KWS = [_W3D.W3Supervisor(_BR.W3_SAFETY, cmd_limits=_BR.W3_CMD_LIMITS[_KW_KEYS[0]]) for _ in range(N)]""", """    _KW_SAFETY = _W3D.parse_w3_safety(args_cli.kw_safety) if args_cli.kw_safety else _BR.W3_SAFETY
    if args_cli.kw_safety:
        print(f"[runner_loop] X19b 감독기 문턱 덮어쓰기: {args_cli.kw_safety} → {_KW_SAFETY}", flush=True)
    KWS = [_W3D.W3Supervisor(_KW_SAFETY, cmd_limits=_BR.W3_CMD_LIMITS[_KW_KEYS[0]]) for _ in range(N)]""")
rep("""감독기 {_BR.W3_SAFETY}""", """감독기 {_KW_SAFETY}""")
# 1008 AIR1: 공중 시작 — 지지비 1.0 으로 띄워 놓고(스폰 높이 +clear) 인계한 뒤 hold s 부터 rel s 에 걸쳐 지지를 0 으로 내린다(env 의 hang_support 힘 모델 그대로).
#   인계 시각은 종전대로 --precheck_s. precheck_s ≤ hold 이면 "공중에서 정책 시작", precheck_s > hold 이면 "홀드로 착지한 뒤 정책 시작"(대조).
rep("""parser.add_argument("--kw_safety", type=str, default="",""", """parser.add_argument("--air_start", type=str, default="", help="AIR1: 'clear:lo,hi;hold:T;rel:T' — 지지비 1.0·스폰 높이 +clear[m] 로 시작, 에피소드 시각 hold[s] 부터 rel[s] 에 걸쳐 지지 0. '' = 종전(지면 시작)")
parser.add_argument("--kw_safety", type=str, default="",""")
rep("""assert float(_hp["r0_range"][1]) <= 1e-6, f"지면 절차 평가는 지지 0 태스크에서만 (r0_range={_hp['r0_range']})\"""", """_AIR = None
if args_cli.air_start:
    _AIR = {"clear": (0.05, 0.15), "hold": 5.0, "rel": 4.0, "tp": (0.1, 0.3)}
    for _it in args_cli.air_start.split(";"):
        _k, _, _v = _it.strip().partition(":")
        if _k == "clear":
            _a, _, _b = _v.partition(","); _AIR["clear"] = (float(_a), float(_b or _a))
        elif _k in ("hold", "rel"):
            _AIR[_k] = float(_v)
        elif _k == "tp":
            _a, _, _b = _v.partition(","); _AIR["tp"] = (float(_a), float(_b or _a))
        elif _k:
            raise SystemExit(f"--air_start: 모르는 항목 {_k!r}")
    _hp["r0_range"] = (1.0, 1.0); _hp["hold_range"] = (_AIR["hold"], _AIR["hold"]); _hp["release_range"] = (_AIR["rel"], _AIR["rel"])
    # 지지 = 힘제어 케이블(장력 r(t)·m·g, 방향 후크→고정 앵커; 후크는 base 위 0.25–0.40 m) + 앞뒤 2점 현수의 피치 구속(간격 tp m — 실측 없음, 가정 0.1–0.3).
    # base 원점에 수직힘만 주는 종전 힘 모델은 매달린 몸이 CoM 편심만큼 20° 넘게 기운다(연습 실행에서 확인) — 실물 지그와 다르다.
    _hp["cable"] = True; _hp["cable_force_mode"] = True; _hp["two_point_d_range"] = tuple(_AIR["tp"])
    _AIR["pre_air"] = (float(args_cli.precheck_s) + float(args_cli.level_s)) <= _AIR["hold"] + 1e-9      # 인계가 방출 시작보다 먼저 = 공중에서 정책 시작
    print(f"[air_start] 지지비 1.0·스폰 +{_AIR['clear']} m · 방출 {_AIR['hold']} s 부터 {_AIR['rel']} s · 인계 {'공중' if _AIR['pre_air'] else '착지 뒤(홀드로 착지)'} · 힘제어 케이블 + 2점 현수 간격 {_AIR['tp']} m (가정 모형 — 실물 줄·지그의 실측 아님)", flush=True)
else:
    assert float(_hp["r0_range"][1]) <= 1e-6, f"지면 절차 평가는 지지 0 태스크에서만 (r0_range={_hp['r0_range']})\"""")
rep("""_pr = env_cfg.events.reset_base.params["pose_range"]
""", """_pr = env_cfg.events.reset_base.params["pose_range"]
if _AIR is not None:
    _pr["z"] = tuple(_AIR["clear"])
""")
rep("""pre_pass = (~fell_pre) & pre_both & (pre_pmax <= 10.0)""", """if _AIR is not None and _AIR["pre_air"]:
    pre_both = torch.ones_like(pre_both)      # AIR1: 공중 인계 — 사전 검사에서 접촉을 요구하지 않는다(기울기·각속도·속도 기준은 그대로)
pre_pass = (~fell_pre) & pre_both & (pre_pmax <= 10.0)""")
# 1008 AIR1: 접지 확인(관측 접촉 비트 양발 0.3 s 연속 — 브리지 HangGuard CONTACT_CONFIRM_S 와 같은 정의, 한 번 확인되면 래치) 전에는 KE 감지기 정지(C5Runner.det_hold — 배포 모듈의 같은 스위치).
rep("""parser.add_argument("--air_start", type=str, default="",""", """parser.add_argument("--air_det_hold", action="store_true", help="AIR1: 접지 확인(양발 접촉 0.3 s) 전에는 KE 회복 감지기 정지(서기 정책만) — C5Runner.det_hold")
parser.add_argument("--air_contact_one", action="store_true", help="AIR1c 진단: --air_det_hold 와 함께 — 접지 확인 전에는 러너에 주는 접촉 비트를 (1, 1) 로 고정(정책 관측만, 물리·채점·접지 확인 판정 불변)")
parser.add_argument("--air_start", type=str, default="",""")
rep("""            R.step(gy[e], pg[e], _cmd_e, qp[e], jv[e], ct[e])
""", """            if args_cli.air_det_hold:
                if "_AIR_DH" not in globals():
                    globals()["_AIR_DH"] = {"run": [0] * N, "acq": [False] * N}
                if not _AIR_DH["acq"][e]:
                    _AIR_DH["run"][e] = _AIR_DH["run"][e] + 1 if (float(ct[e][0]) > 0.5 and float(ct[e][1]) > 0.5) else 0
                    _AIR_DH["acq"][e] = _AIR_DH["run"][e] >= 15
                R.ke.det_hold = not _AIR_DH["acq"][e]
            R.step(gy[e], pg[e], _cmd_e, qp[e], jv[e], (_npk.ones(2, dtype=_npk.float32) if (args_cli.air_contact_one and args_cli.air_det_hold and not _AIR_DH["acq"][e]) else ct[e]))
""")
open(P + ".tmp", "w", encoding="utf-8").write(src); os.replace(P + ".tmp", P); print("kwhl patch applied", len(src.splitlines()))
