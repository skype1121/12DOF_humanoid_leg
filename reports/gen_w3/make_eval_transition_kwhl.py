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
open(P + ".tmp", "w", encoding="utf-8").write(src); os.replace(P + ".tmp", P); print("kwhl patch applied", len(src.splitlines()))
