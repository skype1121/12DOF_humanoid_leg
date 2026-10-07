"""7 단계 패치(1007): eval_transition(사본)에 --slew_joint 'ankle_r:2.0,ankle_f:3.0' — 배포 체인의 슬루를 관절 이름 부분 일치로 따로 준다(나머지는 --slew_deg 또는 기본 4°/틱).
배경: 실물 sensor_stale 정지 6 건 중 4 건이 회복 모드에서 발목 롤 목표를 1 s 안에 16–26° 뒤집은 직후 발목 롤 모터(6·12) 피드백 동결(reports/real_1007/README.md).
발목 명령 속도를 제한했을 때 회복 성능 비용을 재기 위한 옵션(TRANSITION_W3KE.md X10). 기본 '' = 종전과 같음. --trace_all 에 체인 목표(cmd) 기록도 더한다(기록 전용)."""
import os
P = os.environ.get("TR_DST", "rl_walking/scripts/eval_transition_dev.py")
src = open(P, encoding="utf-8").read()
def rep(old, new, count=1):
    global src
    n = src.count(old); assert n == count, f"앵커 {n}회: {old[:90]!r}"
    src = src.replace(old, new)

rep('''parser.add_argument("--pre_w", type=float, default=0.25); parser.add_argument("--pre_w_s", type=float, default=1.0)''',
'''parser.add_argument("--pre_w", type=float, default=0.25); parser.add_argument("--pre_w_s", type=float, default=1.0)
parser.add_argument("--slew_joint", type=str, default="", help="1007 X10: 관절 이름 부분 일치별 슬루 [°/틱] 'ankle_r:2.0,ankle_f:3.0' — 나머지는 체인 기본. '' = 끔")''')
rep('''        _chain = DeployChain(_DEF, _land, qpos0=_land.clone(), stand=True, **_kw)''',
'''        _chain = DeployChain(_DEF, _land, qpos0=_land.clone(), stand=True, **_kw)
        if args_cli.slew_joint:      # 1007 X10: 관절별 슬루(이름 부분 일치) — 체인의 스칼라 슬루를 (12,) 텐서로 바꾼다
            _sj = torch.full((len(jn),), float(_chain.slew), device=dev)
            for _tok in args_cli.slew_joint.split(","):
                _k, _v = _tok.split(":"); _hit = [ci for ci, nm in enumerate(jn) if _k.strip() in nm]
                if not _hit:
                    raise SystemExit(f"--slew_joint: '{_k}' 와 맞는 관절 없음 {jn}")
                _sj[_hit] = math.radians(float(_v))
            _chain.slew = _sj
            print(f"[slew_joint] 관절별 슬루 [°/틱]: {dict(zip(jn, [round(math.degrees(float(x)), 2) for x in _sj]))}", flush=True)''')
# trace_all: 체인 목표 기록
rep('''"ge", "u", "alive", "ke", "mode", "hold", "F", "ffwd", "flat", "h", "ref", "rp2")}''', '''"ge", "u", "alive", "ke", "mode", "hold", "F", "ffwd", "flat", "h", "ref", "rp2", "cmd")}''')
rep('''            _TALL["h"].append(robot.data.root_pos_w[:_na, 2].half().cpu())''', '''            _TALL["h"].append(robot.data.root_pos_w[:_na, 2].half().cpu())
            _TALL["cmd"].append(cmd[:_na].half().cpu())      # 1007 X10: 체인이 낸 관절 목표[rad] — 발목 명령 스윙·속도 통계용''')
open(P + ".tmp", "w", encoding="utf-8").write(src); os.replace(P + ".tmp", P); print("slew_joint patch applied", len(src.splitlines()))
