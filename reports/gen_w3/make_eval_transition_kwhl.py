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
                _mc = _W3D.w3_move_cmd(arg.strip()); _KW_DURS[round(float(tt), 4)] = float(_mc["dur_s"])
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
open(P + ".tmp", "w", encoding="utf-8").write(src); os.replace(P + ".tmp", P); print("kwhl patch applied", len(src.splitlines()))
