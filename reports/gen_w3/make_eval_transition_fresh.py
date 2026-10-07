"""4 단계 패치(1007): eval_transition --w3_fresh — ke2w3 에서 W3 로 넘어가는 틱에 gait 를 바로 켜는 경우(선행조건 0)를 **배포 러너의 fresh 시작**(W3GaitClock tick 0·1:
ge = want 즉시, 위상 u0 고정, 관측 게이트 1.0 즉시 — 학습 에피소드 시작과 같은 수열)으로 모사한다. 기본 꺼짐(= 종전: 에피소드 중 0→1 래치, 게이트 0.2 s 램프).
또 --ke_tick1: w3ke 에서 래치 '다음 틱'에 KE(배포 합성 러너 계약 — W3 가 래치 틱을 처리) 를 위한 고정 정착 지연은 --settle fixed --settle_s 0.02 로 이미 표현 가능하므로 옵션 추가 없음."""
import os
P = os.environ.get("TR_DST", "rl_walking/scripts/eval_transition.py")
src = open(P, encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    n = src.count(old); assert n == count, f"앵커 {n}회: {old[:90]!r}"
    src = src.replace(old, new)

rep('''parser.add_argument("--trace_all", type=int, default=0,''', '''parser.add_argument("--w3_fresh", action="store_true", help="ke2w3 즉시 켜기를 배포 러너 fresh 시작(게이트 즉시 1·위상 2 틱 고정)으로 모사(기본 = 에피소드 중 0→1 램프)")
parser.add_argument("--trace_all", type=int, default=0,''')
# 래퍼: fresh 강제 틱
rep('''    fo = getattr(env, "_tr_force_off", None)
    if fo is not None and bool(fo.any()):
        env._ge[fo] = False; env._gu[fo] = 0.0; env._gf[fo] = 0.0; env._ge_obs[fo] = 0.0
''', '''    fo = getattr(env, "_tr_force_off", None)
    if fo is not None and bool(fo.any()):
        env._ge[fo] = False; env._gu[fo] = 0.0; env._gf[fo] = 0.0; env._ge_obs[fo] = 0.0
    fr = getattr(env, "_tr_fresh", None)          # 1007 --w3_fresh: 배포 W3GaitClock 의 fresh 틱(tick ≤ 1) 모사 — ge = want, u = u0, 게이트 즉시
    if fr is not None and bool((fr > 0).any()):
        m = fr > 0
        env._ge[m] = env._ge_cmd[m]
        env._gu[m] = _tr_u0(env)[m]
        env._gf[m] = torch.where(env._ge[m], torch.full_like(env._gf[m], f_step), torch.zeros_like(env._gf[m]))
        env._ge_obs[m] = env._ge[m].float()
        env._tr_fresh[m] -= 1

def _tr_u0(env):
    """리드 발 규칙 위상(w3_gait_update 의 u0 와 같은 식)."""
    cw = env.command_manager.get_command("base_velocity")
    u0 = torch.zeros(env.num_envs, device=env.device)
    if _LVY > 0.0:
        u0 = torch.where(cw[:, 1] < -_LVY, torch.full_like(u0, 0.5), u0)
    if _LWZ > 0.0:
        u0 = torch.where((cw[:, 1].abs() < 0.02) & (cw[:, 2] < -_LWZ), torch.full_like(u0, 0.5), u0)
    return u0
''')
# ke2w3 전환 틱: 즉시 켜기 + fresh 모사
rep('''                if args_cli.ke_mode in ("restart", "reset"):
                    _tr_runner_reset(sw_now, _po0)
                if args_cli.ke_mode == "direct":           # 상태 이어받기: 클록 이력만 (0,0)(W3 STAND 계약) — env 버퍼는 이미 W3 클록(ge 0 → (0,0))''',
'''                _imm = args_cli.w3_fresh and float(args_cli.pre_both_s) <= 0.0 and float(args_cli.pre_w_s) <= 0.0 and args_cli.ke_mode in ("restart", "reset")
                if _imm:                                   # 배포 fresh 시작(tick 0): 이 틱 관측의 클록 = [sin 2πu0, cos 2πu0]·1 (이력 전부 백필), env 상태도 같은 값
                    uenv._tr_cmd = torch.tensor(_W3CMD, device=dev).unsqueeze(0).expand(N, 3).clone(); uenv._tr_want = sw_now.clone()
                    uenv.command_manager.get_term("base_velocity").vel_command_b[sw_now] = uenv._tr_cmd[sw_now]
                    _u0f = _tr_u0(uenv)
                    _po0[sw_now, _GC1 - 2] = torch.sin(2.0 * math.pi * _u0f[sw_now]); _po0[sw_now, _GC1 - 1] = torch.cos(2.0 * math.pi * _u0f[sw_now])
                    _vc0, _vc1 = _LAYT["velocity_commands"]
                    _po0[sw_now, _vc1 - 3:_vc1] = uenv._tr_cmd[sw_now]      # 이 틱 명령 관측(배포: 켜기 틱에 명령도 같이 들어감)
                    uenv._ge[sw_now] = True; uenv._ge_cmd[sw_now] = True; uenv._gu[sw_now] = _u0f[sw_now]; uenv._gf[sw_now] = 1.1; uenv._ge_obs[sw_now] = 1.0
                    if getattr(uenv, "_tr_fresh", None) is None:
                        uenv._tr_fresh = torch.zeros(N, dtype=torch.long, device=dev)
                    uenv._tr_fresh[sw_now] = 1                              # 다음 틱(tick 1)도 fresh
                if args_cli.ke_mode in ("restart", "reset"):
                    _tr_runner_reset(sw_now, _po0)
                if args_cli.ke_mode == "direct":           # 상태 이어받기: 클록 이력만 (0,0)(W3 STAND 계약) — env 버퍼는 이미 W3 클록(ge 0 → (0,0))''')
open(P + ".tmp", "w", encoding="utf-8").write(src); os.replace(P + ".tmp", P); print("w3_fresh patch applied", len(src.splitlines()))
