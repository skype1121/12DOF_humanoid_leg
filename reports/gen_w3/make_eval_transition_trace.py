"""3 단계 패치(1007): eval_transition 에 --trace_all N — 앞 N env 의 인계 뒤 전 구간 틱 추적(정책 관측 자이로·중력·접촉, 참 피치·롤·|ω|, ge·위상 u·생존·KE 활성·회복 모드·외란 힘).
기록 전용(기본 0 = 종전과 같음). 산출 log_picture/transall_<tag>.npz. 걷는 중 외란 감지기 오프라인 분석용(TRANSITION_W3KE.md Q6)."""
import os
P = os.environ.get("TR_DST", "rl_walking/scripts/eval_transition.py")
src = open(P, encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    n = src.count(old); assert n == count, f"앵커 {n}회: {old[:90]!r}"
    src = src.replace(old, new)

rep('''parser.add_argument("--pre_w", type=float, default=0.25); parser.add_argument("--pre_w_s", type=float, default=1.0)''',
'''parser.add_argument("--pre_w", type=float, default=0.25); parser.add_argument("--pre_w_s", type=float, default=1.0)
parser.add_argument("--trace_all", type=int, default=0, help="앞 N env 의 인계 뒤 전 구간 틱 추적 → log_picture/transall_<tag>.npz (기록 전용)")''')
rep('''        _cur = {"a": a, "cmd": cmd, "q": robot.data.joint_pos, "pitch": pitch, "roll": roll, "w": w_xy, "v": v_xy,''',
'''        if args_cli.trace_all > 0:
            _na = min(int(args_cli.trace_all), N)
            if "_TALL" not in dir():
                _TALL = {k_: [] for k_ in ("t", "gyro", "grav", "cont", "pitch", "roll", "w", "vxy", "ge", "u", "alive", "ke", "mode", "hold", "F", "ffwd", "flat", "h", "ref", "rp2")}
                _PG0, _PG1 = _LAYT["projected_gravity"]
            _pot = obs["policy"]
            _TALL["t"].append(t_rel); _TALL["F"].append(f_now)
            _TALL["gyro"].append(_pot[:_na, _GY1 - 3:_GY1].float().cpu()); _TALL["grav"].append(_pot[:_na, _PG1 - 3:_PG1].float().cpu())
            _TALL["cont"].append((_pot[:_na, _FC1 - 2:_FC1] > 0.5).cpu())
            _TALL["pitch"].append(pitch[:_na].half().cpu()); _TALL["roll"].append(roll[:_na].half().cpu()); _TALL["w"].append(w_xy[:_na].half().cpu())
            _TALL["vxy"].append(robot.data.root_lin_vel_b[:_na, :2].half().cpu())
            _TALL["ge"].append(uenv._ge[:_na].cpu()); _TALL["u"].append(uenv._gu[:_na].half().cpu()); _TALL["alive"].append((a_ & ~term)[:_na].cpu())
            _TALL["ke"].append(ke_act[:_na].cpu()); _TALL["hold"].append(hold_mask[:_na].cpu())
            _TALL["mode"].append((sw_mode[:_na] if policy2 is not None else torch.zeros(_na, dtype=torch.bool, device=dev)).cpu())
            # 1007 X2e: 자세·감지기 기준 기록(기록 전용) — 발-골반 앞축·좌우축 거리[m], 골반 높이, 감지기 기준(롤·피치°)·감지 자세(롤·피치°)
            _fpT = robot.data.body_pos_w[:_na, _FT, :2] - robot.data.root_pos_w[:_na, None, :2]
            _TALL["ffwd"].append((_fpT * _dir_vec("f")[:_na, None, :2]).sum(-1).half().cpu()); _TALL["flat"].append((_fpT * _dir_vec("l")[:_na, None, :2]).sum(-1).half().cpu())
            _TALL["h"].append(robot.data.root_pos_w[:_na, 2].half().cpu())
            _nan2 = torch.full((_na, 2), float("nan"))
            _TALL["ref"].append(sw_ref[:_na].float().cpu() if "sw_ref" in dir() else _nan2); _TALL["rp2"].append(_rp2[:_na].float().cpu() if "_rp2" in dir() else _nan2)
        _cur = {"a": a, "cmd": cmd, "q": robot.data.joint_pos, "pitch": pitch, "roll": roll, "w": w_xy, "v": v_xy,''')
rep('''if C2MODE and _C2TR is not None:''', '''if args_cli.trace_all > 0 and "_TALL" in dir():
    import numpy as _npa
    _npa.savez_compressed(os.path.join(DATA, "log_picture", f"transall_{args_cli.out_tag}.npz"),
                          t=_npa.array(_TALL["t"], dtype=_npa.float32), F=_npa.array(_TALL["F"], dtype=_npa.float32),
                          **{k_: torch.stack(v_, 0).numpy() for k_, v_ in _TALL.items() if k_ not in ("t", "F")},
                          pre_pass=pre_pass[:min(int(args_cli.trace_all), N)].cpu().numpy(), fell=fell[:min(int(args_cli.trace_all), N)].cpu().numpy(),
                          t_fall_post=t_fall_post[:min(int(args_cli.trace_all), N)].cpu().numpy(), tr_sw=tr_sw[:min(int(args_cli.trace_all), N)].cpu().numpy(),
                          dist=args_cli.dist, ke_mode=args_cli.ke_mode, args=" ".join(sys.argv[1:]))
    print(f"[trace_all] log_picture/transall_{args_cli.out_tag}.npz", flush=True)
if C2MODE and _C2TR is not None:''')
open(P + ".tmp", "w", encoding="utf-8").write(src); os.replace(P + ".tmp", P); print("trace_all patch applied", len(src.splitlines()))
