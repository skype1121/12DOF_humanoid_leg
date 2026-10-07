"""6 단계 패치(1007): eval_transition(사본)에 --runner_loop — **실제 배포 코드**(브리지 _make_runner 로 만든 W3S3d 러너 + KE C5Runner 를 묶은
rl_walking/deploy/kew3_runner.KeW3Runner, 켜기 판정은 w3_deploy.W3Supervisor)를 env 마다 하나씩 두고 시뮬 폐루프에 넣는다.
러너 입력 = 정책 관측의 최신 프레임(자이로·중력·관절각 + 러너 기본자세·접촉 비트) + 관측 필터 전 엔코더 속도. 러너 출력 = 이번 틱 실행한 원시 행동 →
env 액션(지연·슬루·LPF·온셋·백래시는 다른 팔과 같은 env/체인 경로). 전환(KE ↔ W3)·회복 모드·정착 대기는 전부 러너가 한다(평가기 상태기는 건너뜀).
명령 대본 --kw_script 'walk@2:0.06,0,0;stop@6;walk@12:0,0,0.15;stop@18'(시각 = 인계 기준 s, walk 는 on_phases·stop 은 stop_phases 로 env 별 분산).
'' = walk@w3_on:w3_cmd; stop@w3_stop. 산출 log_picture/kwloop_<tag>.npz(틱별 행동 주체·want·ge, 사건 목록). 기본 꺼짐 = 종전과 같음."""
import os
P = os.environ.get("TR_DST", "rl_walking/scripts/eval_transition_dev.py")
src = open(P, encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    n = src.count(old); assert n == count, f"앵커 {n}회: {old[:90]!r}"
    src = src.replace(old, new)

rep('''parser.add_argument("--pre_w", type=float, default=0.25); parser.add_argument("--pre_w_s", type=float, default=1.0)''',
'''parser.add_argument("--pre_w", type=float, default=0.25); parser.add_argument("--pre_w_s", type=float, default=1.0)
parser.add_argument("--runner_loop", action="store_true", help="1007: 합성 러너(배포 코드 KeW3Runner + W3Supervisor)를 env 마다 폐루프로 — 전환·감지는 러너가 한다")
parser.add_argument("--kw_script", type=str, default="", help="명령 대본 'walk@t:vx,vy,wz;stop@t;...' (인계 기준 s). '' = walk@w3_on:w3_cmd; stop@w3_stop")
parser.add_argument("--kw_handoff", type=str, default="carry", choices=("carry", "fresh"))
parser.add_argument("--kw_settle", type=float, default=0.5, help="W3 → KE 정착 대기(조용한 양발 서기) [s]. 0 = 래치 다음 틱")
parser.add_argument("--kw_requiet", type=str, default="", help="1007 X6b: 배포 감지기 갇힘 해소 옵션 'T,w'(SwitchDetector.requiet) — '' = 끔(현행 KE)")''')

rep('''for i in range(STEPS):
    t_rel = (i - STEPS_PRE) / 50.0                          # 인계 기준 시각''',
'''KWL = bool(args_cli.runner_loop)
if KWL:
    import importlib.util as _ilu
    import numpy as _npk
    _spec = _ilu.spec_from_file_location("rl_bridge_node", os.path.join(REPO, "jetson", "scripts", "rl_bridge_node.py"))
    _BR = _ilu.module_from_spec(_spec); sys.modules["rl_bridge_node"] = _BR; _spec.loader.exec_module(_BR)
    from rl_walking.deploy.kew3_runner import KeW3Runner as _KWR, MODE_W3 as _KW_MW3
    from rl_walking.deploy import w3_deploy as _W3D
    _BR.enable_experimental_w3()
    _KW_KEYS = ("stand_w3s3d_15800", "stand_c5kr_g21a_g24ra")
    KWR = [_KWR(_BR._make_runner(_KW_KEYS[0]), _BR._make_runner(_KW_KEYS[1]), handoff=args_cli.kw_handoff, settle_quiet_s=float(args_cli.kw_settle)) for _ in range(N)]
    KWS = [_W3D.W3Supervisor(_BR.W3_SAFETY, cmd_limits=_BR.W3_CMD_LIMITS[_KW_KEYS[0]]) for _ in range(N)]
    if args_cli.kw_requiet:                               # 배포 감지기(c5_switch.SwitchDetector)의 갇힘 해소 옵션 — 생성자 인자와 같은 검증을 거쳐 설정
        _rq_ = tuple(float(x) for x in args_cli.kw_requiet.split(","))
        assert len(_rq_) == 2 and _rq_[0] > 0.0 and _rq_[1] > 0.0, f"--kw_requiet 'T,w' (받음 {args_cli.kw_requiet!r})"
        for R in KWR:
            R.ke.det.requiet = _rq_; R.ke.det.reset()
        print(f"[runner_loop] 배포 감지기 requiet = {_rq_}", flush=True)
    _KW_POSE = _npk.asarray(_BR.CKPT_DEFAULT_POSE[_KW_KEYS[1]], dtype=_npk.float32)
    assert list(_BR.JOINT_ORDER) == list(robot.data.joint_names), f"관절 순서 불일치: {robot.data.joint_names}"
    assert _npk.allclose(_KW_POSE, _npk.asarray(_BR.CKPT_DEFAULT_POSE[_KW_KEYS[0]], dtype=_npk.float32)), "두 키의 기본자세가 다름"
    # env 의 default_joint_pos 는 env 별 편향 DR(영점 오차 모형)이 더해진 값 — 정책 프레임 기준은 명목 자세(_nominal_default_joint_pos)
    _kw_nom = getattr(uenv, "_nominal_default_joint_pos", None)
    _kw_nom = _kw_nom if _kw_nom is not None else robot.data.default_joint_pos[0]
    _kw_dp = float((_kw_nom.cpu() - torch.as_tensor(_KW_POSE)).abs().max())
    assert _kw_dp < 1e-3, f"env 명목 기본자세 ≠ 러너 기본자세(최대 차 {_kw_dp:.4f} rad)"
    _KW_PG1 = _LAYT["projected_gravity"][1]; _KW_JP1 = _LAYT["joint_pos"][1]; _KW_JV1 = _LAYT["joint_vel"][1]
    assert KWR[0].obs_dim == obs["policy"].shape[1], f"러너 관측 {KWR[0].obs_dim} ≠ env {obs['policy'].shape[1]}"
    _KW_RAWJV = [None]
    policy2 = None; policy_ke = None                      # 시뮬 쪽 KE·회복 정책·감지기는 쓰지 않는다(러너 안의 것이 돈다)
    sw_mode = torch.zeros(N, dtype=torch.bool, device=dev)
    kw_des = torch.zeros(N, dtype=torch.bool, device=dev); kw_cmd = torch.zeros(N, 3, device=dev)
    kw_cur = [(0.0, 0.0, 0.0)] * N; kw_seq = [1] * N; kw_events = [[] for _ in range(N)]; kw_rej = [None] * N; kw_req_t = [None] * N
    kw_kind_tl = []; kw_want_tl = []; kw_ge_tl = []; kw_t_tl = []; kw_lastq = [0] * N; kw_yaw_tl = []; kw_xy_tl = []

    def _kw_parse(s):
        ev = []
        for tok in s.split(";"):
            tok = tok.strip()
            if not tok:
                continue
            head, _, arg = tok.partition(":"); kind, _, tt = head.partition("@")
            if kind not in ("walk", "stop"):
                raise SystemExit(f"--kw_script: 모르는 사건 {kind!r}")
            ev.append((float(tt), kind == "walk", [float(x) for x in arg.split(",")] if arg else [0.0, 0.0, 0.0]))
        return sorted(ev, key=lambda x: x[0])
    _KW_SCRIPT = _kw_parse(args_cli.kw_script) if args_cli.kw_script else [(float(args_cli.w3_on), True, list(_W3CMD)), (float(args_cli.w3_stop), False, [0.0, 0.0, 0.0])]
    _kw_off_on = (_eidx % max(1, args_cli.on_phases)).float() / max(1, args_cli.on_phases) * float(args_cli.stop_period)
    _kw_off_st = (_eidx % max(1, args_cli.stop_phases)).float() / max(1, args_cli.stop_phases) * float(args_cli.stop_period)
    _KW_EVT = [(tt + (_kw_off_on if w else _kw_off_st)) for (tt, w, c) in _KW_SCRIPT]
    _KW_APPLIED = [torch.zeros(N, dtype=torch.bool, device=dev) for _ in _KW_SCRIPT]
    print(f"[runner_loop] 합성 러너 {N} 개(브리지 _make_runner {_KW_KEYS}, handoff {args_cli.kw_handoff}, 정착 {args_cli.kw_settle}s) · 감독기 {_BR.W3_SAFETY} · 대본 {_KW_SCRIPT}", flush=True)

    def _kw_tick(obs, i, t_rel):
        """합성 러너 폐루프 한 틱 — 평가기 지표가 읽는 상태 변수(tr_*·ke_act·sw_mode·ge_prev)를 러너 상태로 채운다."""
        global tr_want, tr_cmdv, tr_req, tr_u_stop, tr_geoff, tr_settle, tr_settle_to, tr_sw, tr_u_sw, ke_t0, ke_act, w3_act, sw_now, sw_mode, ge_prev
        global tr_enable, tr_first_to, tr_first_foot, kw_des, kw_cmd
        po = obs["policy"]
        if i == STEPS_PRE:
            for R in KWR:
                R.reset()
        for k, (tt, w, c) in enumerate(_KW_SCRIPT):      # 대본 사건 적용(env 별 시각)
            m = (~_KW_APPLIED[k]) & (t_rel >= _KW_EVT[k] - 1e-9)
            if bool(m.any()):
                _KW_APPLIED[k] |= m
                kw_des = torch.where(m, torch.full_like(kw_des, bool(w)), kw_des)
                kw_cmd = torch.where(m.unsqueeze(1), torch.tensor(c, device=dev, dtype=torch.float32).unsqueeze(0).expand(N, 3), kw_cmd)
                if not w:
                    tr_req = torch.where(m & torch.isnan(tr_req), torch.full_like(tr_req, t_rel), tr_req)
        gy = po[:, _GY1 - 3:_GY1].cpu().numpy().astype(_npk.float32); pg = po[:, _KW_PG1 - 3:_KW_PG1].cpu().numpy().astype(_npk.float32)
        ct = (po[:, _FC1 - 2:_FC1] > 0.5).float().cpu().numpy().astype(_npk.float32)
        qp = (po[:, _KW_JP1 - 12:_KW_JP1].cpu().numpy() + _KW_POSE[None]).astype(_npk.float32)
        jv = (_KW_RAWJV[0] if _KW_RAWJV[0] is not None else po[:, _KW_JV1 - 12:_KW_JV1]).cpu().numpy().astype(_npk.float32)
        al = alive.cpu().numpy(); des = kw_des.cpu().numpy(); cm = kw_cmd.cpu().numpy(); now = float(t_rel)
        bd = t_rel >= (float(args_cli.blend_s) if args_cli.blend_s is not None else 1.5) - 1e-9
        act = _npk.zeros((N, J), _npk.float32); kind = _npk.full(N, -1, _npk.int8)
        want_v = _npk.zeros(N, bool); ge_top = _npk.zeros(N, bool); u_top = _npk.zeros(N, _npk.float32); first_ke = _npk.zeros(N, bool); en_now = _npk.zeros(N, bool)
        for e in range(N):
            if not al[e]:
                continue
            R = KWR[e]; S = KWS[e]; clk = R._w3_clock
            ge_top[e] = bool(clk.ge); u_top[e] = float(clk.u)
            first_ke[e] = R.last_event in ("to_ke", "force_ke")          # 직전 틱 끝에 KE 로 넘어감 → 이번 틱이 KE 첫 틱
            S.observe(now, ct[e], gy[e], pg[e])
            if des[e] and not clk.want:                                   # 켜기 요청(거부되면 다음 틱 새 seq 로 재시도 = 운영자 재시도)
                if kw_req_t[e] is None:
                    kw_req_t[e] = now; kw_events[e].append((round(now, 2), "enable_req"))
                why = R.enable_block_reason()
                if why is None:
                    d = {"cmd": "step", "enable": True, "vx": float(cm[e, 0]), "vy": float(cm[e, 1]), "wz": float(cm[e, 2]), "heading_loop": False, "req_id": "kw", "seq": kw_seq[e]}
                    kw_seq[e] += 1
                    dec = S.handle(d, now, active=True, warmed=True, blend_done=bool(bd), ge_now=bool(clk.ge), want_now=bool(clk.want), cmd_now=kw_cur[e])
                    if dec["accepted"] and dec["action"] == "set_want":
                        kw_cur[e] = tuple(dec["cmd"]); R.set_gait_enable(True); en_now[e] = True
                        kw_events[e].append((round(now, 2), "enable")); kw_req_t[e] = None
                    else:
                        why = dec["why"]
                if why is not None and kw_rej[e] is None:
                    kw_rej[e] = (round(now, 2), str(why)[:80])
            elif des[e] and clk.want and tuple(float(x) for x in cm[e]) != tuple(kw_cur[e]):     # 걷는 중 명령 변경
                d = {"cmd": "step", "enable": True, "vx": float(cm[e, 0]), "vy": float(cm[e, 1]), "wz": float(cm[e, 2]), "heading_loop": False, "req_id": "kw", "seq": kw_seq[e]}
                kw_seq[e] += 1
                dec = S.handle(d, now, active=True, warmed=True, blend_done=bool(bd), ge_now=bool(clk.ge), want_now=True, cmd_now=kw_cur[e])
                if dec["accepted"] and dec["action"] == "set_cmd":
                    kw_cur[e] = tuple(dec["cmd"]); kw_events[e].append((round(now, 2), "set_cmd"))
            elif (not des[e]) and clk.want:                               # 끄기(step enable false — 정지 래치 없음)
                d = {"cmd": "step", "enable": False, "req_id": "kw", "seq": kw_seq[e]}; kw_seq[e] += 1
                dec = S.handle(d, now, active=True, warmed=True, blend_done=bool(bd), ge_now=bool(clk.ge), want_now=True, cmd_now=kw_cur[e])
                if dec["action"] == "set_want":
                    kw_cur[e] = (0.0, 0.0, 0.0); R.set_gait_enable(False); kw_events[e].append((round(now, 2), "stop_req")); kw_req_t[e] = None
            R.step(gy[e], pg[e], kw_cur[e], qp[e], jv[e], ct[e])
            act[e] = R.last_step_action; kind[e] = R.last_step_kind; want_v[e] = bool(clk.want)
            if R.last_event:
                kw_events[e].append((round(now, 2), R.last_event))
            if kind[e] != 2 and R.ke.last_switch:
                kw_events[e].append((round(now, 2), "rec_" + R.ke.last_switch))
            _nq = int(getattr(R.ke.det, "n_requiet", 0))                  # 감지기 재기준(갇힘 해소) 발동 — 감지기는 KE 인계마다 새로 시작하므로 증가분만 사건으로
            if _nq > kw_lastq[e]:
                kw_events[e].append((round(now, 2), "requiet"))
            kw_lastq[e] = _nq
        kind_t = torch.as_tensor(kind, device=dev); ge_t = torch.as_tensor(ge_top, device=dev); fk = torch.as_tensor(first_ke, device=dev); en_t = torch.as_tensor(en_now, device=dev)
        ke_act = (kind_t == 0) | (kind_t == 1); w3_act = kind_t == 2; sw_mode = kind_t == 1
        tr_want = torch.as_tensor(want_v, device=dev); tr_cmdv = torch.where(tr_want.unsqueeze(1), torch.as_tensor(_npk.asarray(kw_cur, _npk.float32), device=dev), torch.zeros(N, 3, device=dev))
        uenv._tr_want = tr_want & w3_act; uenv._tr_cmd = torch.where((tr_want & w3_act).unsqueeze(1), tr_cmdv, torch.zeros_like(tr_cmdv))   # env 쪽 W3 추적기(지표용)를 러너 클록과 같이
        tr_enable = torch.where(en_t & torch.isnan(tr_enable), torch.full_like(tr_enable, t_rel), tr_enable)
        tr_u_stop = torch.where(~torch.isnan(tr_req) & torch.isnan(tr_u_stop), torch.as_tensor(u_top, device=dev), tr_u_stop)
        _geoff_now = ~torch.isnan(tr_req) & ~ge_t & ge_prev & torch.isnan(tr_geoff)
        tr_geoff = torch.where(_geoff_now, torch.full_like(tr_geoff, t_rel), tr_geoff)
        sw_now = fk & torch.isnan(tr_sw) & ~torch.isnan(tr_geoff)             # 첫 W3 → KE 전환(평가기 지표 기준 전환 시각)
        if bool(sw_now.any()):
            tr_sw = torch.where(sw_now, torch.full_like(tr_sw, t_rel), tr_sw); tr_settle = torch.where(sw_now, tr_sw, tr_settle); ke_t0 = torch.where(sw_now, tr_sw, ke_t0)
            tr_u_sw = torch.where(sw_now, torch.as_tensor(u_top, device=dev), tr_u_sw)
            tr_settle_to = tr_settle_to | (sw_now & ((tr_sw - tr_geoff) >= 3.0 - 0.03))
            if _RING:
                _ord = [(i - _TRW + j) % _TRW for j in range(_TRW)]
                for k_ in _RING:
                    TRS[k_][sw_now, :_TRW] = _RING[k_][_ord][:, sw_now].transpose(0, 1)
        _s3 = _w3m._st(uenv); _to = _s3["ev_takeoff"] > 0.5
        _first = ~torch.isnan(tr_enable) & torch.isnan(tr_first_to) & _to.any(1) & alive
        tr_first_to = torch.where(_first, t_rel - tr_enable, tr_first_to)
        tr_first_foot = torch.where(_first, torch.where(_to[:, 0], torch.zeros_like(tr_first_foot), torch.ones_like(tr_first_foot)), tr_first_foot)
        ge_prev = ge_t
        kw_kind_tl.append(kind.copy()); kw_want_tl.append(want_v.copy()); kw_ge_tl.append(ge_top.copy()); kw_t_tl.append(now)
        _qw = robot.data.root_quat_w                               # 기록 전용: 참 요(쿼터니언 wxyz)·골반 수평 위치(env 원점 기준) — 구간별 방향 변화·이동 집계용
        kw_yaw_tl.append(torch.atan2(2 * (_qw[:, 0] * _qw[:, 3] + _qw[:, 1] * _qw[:, 2]), 1 - 2 * (_qw[:, 2] ** 2 + _qw[:, 3] ** 2)).half().cpu().numpy())
        kw_xy_tl.append((robot.data.root_pos_w[:, :2] - uenv.scene.env_origins[:, :2]).half().cpu().numpy())
        return torch.as_tensor(act, device=dev)

for i in range(STEPS):
    t_rel = (i - STEPS_PRE) / 50.0                          # 인계 기준 시각''')

rep('''        if i >= STEPS_PRE and KE2W3:      # ── ke2w3 상태기''', '''        if i >= STEPS_PRE and KE2W3 and not KWL:      # ── ke2w3 상태기''')
rep('''        elif i >= STEPS_PRE:      # ── 1007 전환 상태기(관측·접촉 비트·IMU 만 사용) ──''', '''        elif i >= STEPS_PRE and not KWL:      # ── 1007 전환 상태기(관측·접촉 비트·IMU 만 사용) ──''')
rep('''        a = torch.zeros(N, J, device=dev) if (policy is None or i < STEPS_PRE) else policy(obs)''',
    '''        a = torch.zeros(N, J, device=dev) if (policy is None or i < STEPS_PRE) else (_kw_tick(obs, i, t_rel) if KWL else policy(obs))''')
rep('''        obs, _, dones, _ = env.step(a)
        if _OBS_EMA:''', '''        obs, _, dones, _ = env.step(a)
        if KWL:
            _KW_RAWJV[0] = obs["policy"][:, _KW_JV1 - 12:_KW_JV1].clone()      # 관측 필터(EMA) 전 엔코더 속도 최신 프레임 = 러너 qvel 입력
        if _OBS_EMA:''')
rep('''_TALL["mode"].append((sw_mode[:_na] if policy2 is not None else torch.zeros(_na, dtype=torch.bool, device=dev)).cpu())''',
    '''_TALL["mode"].append((sw_mode[:_na] if (policy2 is not None or KWL) else torch.zeros(_na, dtype=torch.bool, device=dev)).cpu())''')
rep('''"mode": (sw_mode.float() if policy2 is not None else torch.zeros(N, device=dev))''', '''"mode": (sw_mode.float() if (policy2 is not None or KWL) else torch.zeros(N, device=dev))''')
rep('''if C2MODE and _C2TR is not None:''', '''if KWL:
    import collections as _colk
    _kk = _npk.stack(kw_kind_tl, 0) if kw_kind_tl else _npk.zeros((0, N), _npk.int8)
    _npk.savez_compressed(os.path.join(DATA, "log_picture", f"kwloop_{args_cli.out_tag}.npz"), t=_npk.asarray(kw_t_tl, _npk.float32), kind=_kk,
                          want=_npk.stack(kw_want_tl, 0) if kw_want_tl else _npk.zeros((0, N), bool), ge=_npk.stack(kw_ge_tl, 0) if kw_ge_tl else _npk.zeros((0, N), bool),
                          pre_pass=pre_pass.cpu().numpy(), fell=fell.cpu().numpy(), t_fall_post=t_fall_post.cpu().numpy(), events=json.dumps(kw_events), rej=json.dumps(kw_rej),
                          script=json.dumps(_KW_SCRIPT), ev_t=torch.stack(_KW_EVT).cpu().numpy(), n_to_w3=_npk.asarray([R.n_to_w3 for R in KWR]), n_to_ke=_npk.asarray([R.n_to_ke for R in KWR]),
                          handoff=args_cli.kw_handoff, settle=float(args_cli.kw_settle), dist=args_cli.dist, args=" ".join(sys.argv[1:]),
                          requiet=args_cli.kw_requiet, yaw=_npk.stack(kw_yaw_tl, 0) if kw_yaw_tl else _npk.zeros((0, N), _npk.float16),
                          xy=_npk.stack(kw_xy_tl, 0) if kw_xy_tl else _npk.zeros((0, N, 2), _npk.float16))
    _pp_ = pre_pass.cpu().numpy()
    _rj_ = _colk.Counter((r[1][:40] if r is not None else "거부 없음") for r, p in zip(kw_rej, _pp_) if p)
    print(f"[runner_loop] log_picture/kwloop_{args_cli.out_tag}.npz · 통과 env {int(_pp_.sum())} · KE→W3 전환 합 {int(sum(R.n_to_w3 for R, p in zip(KWR, _pp_) if p))} · W3→KE 전환 합 {int(sum(R.n_to_ke for R, p in zip(KWR, _pp_) if p))}"
          f" · 행동 주체 틱 비율 KE 서기 {100*float((_kk[:, _pp_] == 0).mean()):.1f} % / KE 회복 {100*float((_kk[:, _pp_] == 1).mean()):.1f} % / W3 {100*float((_kk[:, _pp_] == 2).mean()):.1f} % · 첫 켜기 거부 사유 {dict(_rj_)}", flush=True)
if C2MODE and _C2TR is not None:''')
open(P + ".tmp", "w", encoding="utf-8").write(src); os.replace(P + ".tmp", P); print("runner_loop patch applied", len(src.splitlines()))
