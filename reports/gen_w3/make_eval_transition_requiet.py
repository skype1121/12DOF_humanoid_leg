"""5 단계 패치(1007): eval_transition(사본)에 --sw_requiet 'T,w' — 회복 모드 갇힘 해소 변형(TRANSITION_W3KE.md §5.6g X5).
회복 모드 중 (감지 |ω_xy| < w ∧ 양발 접촉)이 T s 이어지면 기준 자세 := 현재 감지 자세, 모드 끔. 기본 ''(끔) = 종전과 같음."""
import os
P = os.environ.get("TR_DST", "rl_walking/scripts/eval_transition_dev.py")
src = open(P, encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    n = src.count(old); assert n == count, f"앵커 {n}회: {old[:90]!r}"
    src = src.replace(old, new)

rep('''parser.add_argument("--pre_w", type=float, default=0.25); parser.add_argument("--pre_w_s", type=float, default=1.0)''',
'''parser.add_argument("--pre_w", type=float, default=0.25); parser.add_argument("--pre_w_s", type=float, default=1.0)
parser.add_argument("--sw_requiet", type=str, default="", help="1007 X5: 'T,w' — 회복 모드 중 감지 |ω_xy| < w ∧ 양발 접촉이 T s 이어지면 기준 자세를 현재 자세로 다시 잡고 모드 끔(갇힘 해소). '' = 끔")''')
rep('''            _new_on = (~sw_mode) & _on
            swc_n += (_new_on & alive).float(); sw_mode = (sw_mode | _new_on) & ~_swoff''',
'''            if args_cli.sw_requiet:      # 1007 X5: 갇힘 해소 — 모드 중 조용한 양발 서기 T s → 기준 := 현재 자세, 모드 끔
                _rqT, _rqW = [float(x) for x in args_cli.sw_requiet.split(",")]
                if "rq_run" not in dir():
                    rq_run = torch.zeros(N, device=dev); rq_n = torch.zeros(N, device=dev)
                    print(f"[sw_requiet] 회복 모드 중 |ω| < {_rqW} ∧ 양발 {_rqT} s → 기준 재설정·모드 끔", flush=True)
                rq_run = torch.where(sw_mode & (_w < _rqW) & _cf[:, 0] & _cf[:, 1], rq_run + 0.02, torch.zeros_like(rq_run))
                _rq_off = sw_mode & (rq_run >= _rqT - 1e-6) & ~_swoff
                if bool(_rq_off.any()) and float(args_cli.sw_ref_tau) > 0.0:
                    sw_ref = torch.where(_rq_off.unsqueeze(1), _rp2, sw_ref)
                _swoff = _swoff | _rq_off; rq_n += (_rq_off & alive).float()
            _new_on = (~sw_mode) & _on
            swc_n += (_new_on & alive).float(); sw_mode = (sw_mode | _new_on) & ~_swoff''')
rep('''if args_cli.trace_all > 0 and "_TALL" in dir():''', '''if args_cli.sw_requiet and "rq_n" in dir():
    print(f"[sw_requiet] 재기준 발동: env {int((rq_n > 0).sum())} / {N} · 총 {int(rq_n.sum())} 회", flush=True)
if args_cli.trace_all > 0 and "_TALL" in dir():''')
open(P + ".tmp", "w", encoding="utf-8").write(src); os.replace(P + ".tmp", P); print("sw_requiet patch applied", len(src.splitlines()))
