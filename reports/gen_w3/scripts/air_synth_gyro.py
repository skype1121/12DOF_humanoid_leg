# 합성 대본(관절 = 직전 목표·관절속도 0·중력 고정·무접촉)에서 자이로 펄스 크기별로 서기 정책의 원시 행동이 유계인가 — 체크포인트 비교.
import sys, math, os
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); sys.path.insert(0, "/home/ryu/humanoid_leg_test1/jetson/scripts")
import numpy as np
import rl_bridge_node as B
D = "/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_stage4"
CAND = {"G21a-11300(지면 서기 기준)": "stand_crouch_g21a", "G21h-11500": D + "/2026-10-08_13-16-02_g21h_s1_from_11300/exported_11500", "G21h-12299(후보)": D + "/2026-10-08_13-16-02_g21h_s1_from_11300/exported_12299"}
def quat(p):
    p = math.radians(p); return (math.cos(p / 2), 0.0, math.sin(p / 2), 0.0)
def mk(name, src):
    if not os.path.isdir(src): return B._make_runner(src), src
    k = "stand_dbg_" + name.split("(")[0].replace("-", "_")
    B.CHECKPOINTS[k] = src; B.CKPT_DEFAULT_POSE[k] = list(B.CKPT_DEFAULT_POSE["stand_crouch_g21a"]); B.CKPT_GAIT_FREQ[k] = B.CKPT_GAIT_FREQ["stand_crouch_g21a"]; B.CKPT_REQUIRE_ENV_YAML.add(k)
    return B._make_runner(k), k
pose = np.asarray(B.CKPT_DEFAULT_POSE["stand_crouch_g21a"], dtype=np.float32)
print("| 정책 | 자이로 펄스(0.6 s) | 접촉 | 원시 행동 |a| 최대: 펄스 중 · 펄스 뒤 2 s · 뒤 4 s | 목표 이탈 최대 [°] | 판정 |"); print("|---|---|---|---|---|---|")
for name, src in CAND.items():
    for axis, amp in (("y", 0.0), ("y", 0.3), ("y", 0.6), ("y", 0.9), ("x", 0.6), ("x", 0.9)):
        for cont in ([0.0, 0.0], [1.0, 1.0]):
            R, key = mk(name, src); R.reset()
            q = pose.copy(); amax = [0.0, 0.0, 0.0]; dev = 0.0; bad = None
            for n in range(100 + 30 + 200):
                g = [0.0, 0.0, 0.0]
                if 100 <= n < 130: g["xyz".index(axis)] = amp
                out = R.step(tuple(g), B.correct_gravity(B.projected_gravity(quat(1.0))), np.zeros(3, np.float32), q, np.zeros(12, np.float32), cont)
                a = float(np.abs(R.last_action).max()) if np.isfinite(R.last_action).all() else float("inf")
                seg = 0 if n < 130 else (1 if n < 230 else 2); amax[seg] = max(amax[seg], a)
                if not np.isfinite(out).all(): bad = n; break
                dev = max(dev, float(np.degrees(np.abs(out - pose)).max())); q = np.asarray(out, np.float32)
            v = "비유한(틱 %d)" % bad if bad is not None else ("폭주" if amax[2] > 20 else ("커짐" if amax[2] > 3 * max(amax[0], 1.0) else "유계"))
            print(f"| {name} | {axis} {amp} rad/s | {'양발' if cont[0] else '없음'} | {amax[0]:.2f} · {amax[1]:.3g} · {amax[2]:.3g} | {dev:.0f} | {v} |", flush=True)
