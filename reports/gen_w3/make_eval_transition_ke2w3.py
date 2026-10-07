"""2 단계 패치: eval_transition.py 에 --dir ke2w3 (KE 서기 → W3 전환 → 선행조건 → 켜기 → 걷기 → 정지) 추가. 기본 --dir w3ke 경로는 ke_t0 분리 외 불변."""
import sys
import os
P = os.environ.get("TR_DST", "rl_walking/scripts/eval_transition.py")
src = open(P, encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    n = src.count(old); assert n == count, f"앵커 {n}회: {old[:90]!r}"
    src = src.replace(old, new)

# 인자
rep('''parser.add_argument("--hold_s", type=float, default=3.0, help="restart: 명령 동결(스택 재기동 가정) [s]")''',
'''parser.add_argument("--hold_s", type=float, default=3.0, help="restart: 명령 동결(스택 재기동 가정) [s]")
parser.add_argument("--dir", type=str, default="w3ke", choices=("w3ke", "ke2w3"), help="w3ke = W3 걷기→stop→정착→KE / ke2w3 = KE 서기→W3 STAND→선행조건→켜기→걷기→stop")
parser.add_argument("--ke2w3_at", type=float, default=4.0, help="ke2w3: KE→W3 전환 요청 시각 [s, 인계 기준]")
parser.add_argument("--walk_s", type=float, default=5.0, help="ke2w3: 켜기 뒤 걷기 시간 [s]")
parser.add_argument("--pre_both_s", type=float, default=1.0); parser.add_argument("--pre_tilt", type=float, default=7.0)
parser.add_argument("--pre_w", type=float, default=0.25); parser.add_argument("--pre_w_s", type=float, default=1.0)''')

# 상태 텐서 추가
rep('''walk_lifts = torch.zeros(N, device=dev)
''', '''walk_lifts = torch.zeros(N, device=dev)
ke_t0 = _nanv()                      # KE 클록 원점(KE 러너 tick 0): w3ke = tr_sw, ke2w3 = 인계 0
KE2W3 = args_cli.dir == "ke2w3"
w3_act = torch.zeros(N, dtype=torch.bool, device=dev); tr_enable = _nanv(); pre_both_run = torch.zeros(N, device=dev); pre_w_run = torch.zeros(N, device=dev)
tr_first_to = _nanv(); tr_first_foot = torch.full((N,), -1.0, device=dev); fell_en5 = torch.zeros(N, dtype=torch.bool, device=dev); tr_stop2 = _nanv()
''')
rep('''    ids = ke_act
    ph = 2.0 * math.pi * _GCF * (t_rel - tr_sw[ids]).clamp(min=0.0)''', '''    ids = ke_act
    ph = 2.0 * math.pi * _GCF * (t_rel - ke_t0[ids]).clamp(min=0.0)''')

# 상태기: w3ke 분기 유지 + ke2w3 분기
rep('''        if i >= STEPS_PRE:      # ── 1007 전환 상태기(관측·접촉 비트·IMU 만 사용) ──
            _po0 = obs["policy"]; _ge_now = uenv._ge.clone(); _u_now = uenv._gu.clone()
            _w_obs = _po0[:, _GY1 - 3:_GY1][:, :2].norm(dim=-1); _both_obs = (_po0[:, _FC1 - 2:_FC1] > 0.5).all(dim=-1)
''', '''        if i >= STEPS_PRE and KE2W3:      # ── ke2w3 상태기: KE 서기 → W3 전환 → 선행조건 → 켜기 → 걷기 → stop ──
            _po0 = obs["policy"]; _ge_now = uenv._ge.clone(); _u_now = uenv._gu.clone()
            _w_obs = _po0[:, _GY1 - 3:_GY1][:, :2].norm(dim=-1); _both_obs = (_po0[:, _FC1 - 2:_FC1] > 0.5).all(dim=-1)
            _pg_obs = _po0[:, _LAYT["projected_gravity"][1] - 3:_LAYT["projected_gravity"][1]]
            _tilt_obs = torch.rad2deg(torch.acos((-_pg_obs[:, 2] / _pg_obs.norm(dim=-1).clamp(min=1e-6)).clamp(-1.0, 1.0)))
            if i == STEPS_PRE:
                ke_act[:] = args_cli.ke_mode != "w3only"; ke_t0[:] = 0.0
            _req = (t_rel >= float(args_cli.ke2w3_at)) & ke_act & ~w3_act & (hold_until < 0)
            if args_cli.ke_mode == "restart":
                if bool(_req.any()):
                    hold_mask[_req] = True; hold_until[_req] = t_rel + float(args_cli.hold_s); _FRZ[0][_req] = (_LAST_CMD[0] if _LAST_CMD[0] is not None else _DEF)[_req]
                    tr_req = torch.where(_req, torch.full_like(tr_req, t_rel), tr_req)
                sw_now = hold_mask & (t_rel >= hold_until - 1e-6) & ke_act
            elif args_cli.ke_mode in ("reset", "direct"):
                sw_now = _req; tr_req = torch.where(_req, torch.full_like(tr_req, t_rel), tr_req)
            else:
                sw_now = torch.zeros(N, dtype=torch.bool, device=dev)
            if bool(sw_now.any()):           # W3 시작(= 이 평가의 '전환 시각' tr_sw)
                ke_act = ke_act & ~sw_now; w3_act = w3_act | sw_now; hold_mask[sw_now] = False
                tr_sw = torch.where(sw_now, torch.full_like(tr_sw, t_rel), tr_sw); tr_geoff = tr_sw.clone(); tr_settle = tr_sw.clone()
                if args_cli.ke_mode in ("restart", "reset"):
                    _tr_runner_reset(sw_now, _po0)
                if args_cli.ke_mode == "direct":           # 상태 이어받기: 클록 이력만 (0,0)(W3 STAND 계약) — env 버퍼는 이미 W3 클록(ge 0 → (0,0))
                    _po0[sw_now, _GC0:_GC1] = 0.0
                if args_cli.ke_mode == "restart" and _chain is not None:
                    _chain.land[sw_now] = _FRZ[0][sw_now]; _chain.reset(sw_now, robot.data.joint_pos)
                if _RING:
                    _ord = [(i - _TRW + j) % _TRW for j in range(_TRW)]
                    for k_ in _RING:
                        TRS[k_][sw_now, :_TRW] = _RING[k_][_ord][:, sw_now].transpose(0, 1)
            # 켜기 선행조건(배포 W3Supervisor 정의): 양발(관측) pre_both_s ∧ 기울기 ≤ pre_tilt ∧ |ω|(관측) < pre_w 가 pre_w_s
            _pend_en = w3_act & torch.isnan(tr_enable)
            pre_both_run = torch.where(w3_act & _both_obs, pre_both_run + 0.02, torch.zeros_like(pre_both_run))
            pre_w_run = torch.where(w3_act & (_w_obs < float(args_cli.pre_w)), pre_w_run + 0.02, torch.zeros_like(pre_w_run))
            _en_now = _pend_en & (pre_both_run >= float(args_cli.pre_both_s) - 1e-6) & (pre_w_run >= float(args_cli.pre_w_s) - 1e-6) & (_tilt_obs <= float(args_cli.pre_tilt))
            tr_enable = torch.where(_en_now, torch.full_like(tr_enable, t_rel), tr_enable)
            _walk = w3_act & ~torch.isnan(tr_enable) & ((t_rel - tr_enable) < float(args_cli.walk_s))
            _stop2 = w3_act & ~torch.isnan(tr_enable) & ((t_rel - tr_enable) >= float(args_cli.walk_s))
            tr_stop2 = torch.where(_stop2 & torch.isnan(tr_stop2), torch.full_like(tr_stop2, t_rel), tr_stop2)
            tr_want = _walk
            tr_cmdv = torch.where(tr_want.unsqueeze(1), torch.tensor(_W3CMD, device=dev).unsqueeze(0).expand(N, 3), torch.zeros(N, 3, device=dev))
            uenv._tr_want = tr_want; uenv._tr_cmd = tr_cmdv
            _s3 = _w3m._st(uenv); _to = _s3["ev_takeoff"] > 0.5
            _first = ~torch.isnan(tr_enable) & torch.isnan(tr_first_to) & _to.any(1) & alive
            tr_first_to = torch.where(_first, t_rel - tr_enable, tr_first_to)
            tr_first_foot = torch.where(_first, torch.where(_to[:, 0], torch.zeros_like(tr_first_foot), torch.ones_like(tr_first_foot)), tr_first_foot)
            _HOLD[0] = hold_mask
            ge_prev = _ge_now
        elif i >= STEPS_PRE:      # ── 1007 전환 상태기(관측·접촉 비트·IMU 만 사용) ──
            _po0 = obs["policy"]; _ge_now = uenv._ge.clone(); _u_now = uenv._gu.clone()
            _w_obs = _po0[:, _GY1 - 3:_GY1][:, :2].norm(dim=-1); _both_obs = (_po0[:, _FC1 - 2:_FC1] > 0.5).all(dim=-1)
''')
rep('''            if bool(sw_now.any()):
                ke_act = ke_act | sw_now; tr_sw = torch.where(sw_now, torch.full_like(tr_sw, t_rel), tr_sw); tr_u_sw = torch.where(sw_now, _u_now, tr_u_sw)''',
'''            if bool(sw_now.any()):
                ke_act = ke_act | sw_now; tr_sw = torch.where(sw_now, torch.full_like(tr_sw, t_rel), tr_sw); tr_u_sw = torch.where(sw_now, _u_now, tr_u_sw); ke_t0 = torch.where(sw_now, tr_sw, ke_t0)''')

# 감지기: ke2w3 에서 W3 전환 시 회복 모드 끔
rep('''            if bool(sw_now.any()):      # 1007 전환: KE 시작 env 감지기 상태 초기화(배포 = 감지기 새로 생성)
                sw_mode[sw_now] = False;''', '''            if bool(sw_now.any()) or (KE2W3 and bool((~ke_act & sw_mode).any())):      # 1007 전환: KE 시작 env 감지기 초기화 / ke2w3 W3 전환 뒤 회복 모드 끔
                sw_mode = sw_mode & ke_act
                sw_mode[sw_now] = False;''')

# ke2w3 낙상: 켜기 뒤 5 s
rep('''        t_fall_sw = torch.where(newly & (_ks >= 0), _ks, t_fall_sw)
''', '''        t_fall_sw = torch.where(newly & (_ks >= 0), _ks, t_fall_sw)
        if KE2W3:
            fell_en5 |= newly & ~torch.isnan(tr_enable) & ((t_rel - tr_enable) >= 0) & ((t_rel - tr_enable) < 5.0)
''')

# 요약·CSV
rep('''    P(f"  KE 감지기: 시작 뒤 외란 전 켜짐 env''', '''    if KE2W3:
        _en = ~torch.isnan(tr_enable)
        P(f"  ke2w3: W3 시작 {_cnt(_ke)} · 선행조건 성립 {_cnt(_en)} (W3 시작→성립 중앙 {_q(tr_enable - tr_sw, _m):.2f}s p90 {_q(tr_enable - tr_sw, _m, 0.9):.2f}, 3 s 안 {100*float((((tr_enable - tr_sw) <= 3.0) & _m & _en).sum())/max(1, int((_m & _ke).sum())):.1f} %)"
          f" · 켜기→첫 이륙 중앙 {_q(tr_first_to, _m):.2f}s (1 s 안 {100*float(((tr_first_to <= 1.0) & _m).sum())/max(1, int((_m & _en).sum())):.1f} %, 왼발 {100*float(((tr_first_foot == 0) & _m).sum())/max(1, int((_m & ~torch.isnan(tr_first_to)).sum())):.1f} %)"
          f" · 켜기 뒤 5 s 낙상 {_cnt(fell_en5)} · stop 요청 도달 {_cnt(~torch.isnan(tr_stop2))}")
    P(f"  KE 감지기: 시작 뒤 외란 전 켜짐 env''')
rep('''TRCOLS = ["tr_req",''', '''TRCOLS = ["tr_enable", "tr_first_to", "tr_first_foot", "fell_en5", "tr_stop2", "tr_req",''')
rep('''        row += [_fx(tr_req[e]), _fx(tr_geoff[e]),''', '''        row += [_fx(tr_enable[e]), _fx(tr_first_to[e]), int(tr_first_foot[e]), int(fell_en5[e]), _fx(tr_stop2[e]), _fx(tr_req[e]), _fx(tr_geoff[e]),''')
rep('''                          t_fall_post=t_fall_post.cpu().numpy(), ke_mode=args_cli.ke_mode,''', '''                          t_fall_post=t_fall_post.cpu().numpy(), ke_mode=args_cli.ke_mode, dir=args_cli.dir, tr_enable=tr_enable.cpu().numpy(),''')
rep('''KEMODE = args_cli.ke_mode != "none"
''', '''KEMODE = args_cli.ke_mode != "none"
if args_cli.dir == "ke2w3" and (policy_ke is None or policy2 is None):
    raise SystemExit("--dir ke2w3 는 모든 팔에서 --ke(G21a) 와 --switch_to(G24Ra) 필요")
''')
rep('''                _follow = (~ke_act) | ((t_rel - tr_sw) < float(args_cli.sw_ref_init_s))   # 1007 전환: KE 시작 뒤 init_s 동안 기준 추종''',
    '''                _follow = (~ke_act) | ((t_rel - ke_t0) < float(args_cli.sw_ref_init_s))   # 1007 전환: KE 시작(ke_t0) 뒤 init_s 동안 기준 추종(ke2w3 = 인계 0 s)''')
open(P + ".tmp", "w", encoding="utf-8").write(src); os.replace(P + ".tmp", P); print("ke2w3 patch applied", len(src.splitlines()))
