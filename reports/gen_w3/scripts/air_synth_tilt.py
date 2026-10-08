# 합성 대본(관절 = 직전 목표·관절속도 0·무접촉·자이로 0)에서 **고정 기울기**(피치·롤)에 대한 서기 정책의 원시 행동 — 체크포인트 비교. 10 s.
import sys, math, os
sys.path.insert(0, "/home/ryu/humanoid_leg_test1"); sys.path.insert(0, "/home/ryu/humanoid_leg_test1/jetson/scripts")
import numpy as np
import rl_bridge_node as B
D = "/home/ryu/humanoid_leg_test1/logs/rsl_rl/biped12_stage4"
CAND = {"G21a-11300(지면 서기 기준)": "stand_crouch_g21a", "G21h-11500": D + "/2026-10-08_13-16-02_g21h_s1_from_11300/exported_11500", "G21h-12299(후보)": D + "/2026-10-08_13-16-02_g21h_s1_from_11300/exported_12299"}
def quat_pr(pitch, roll):
    p, r = math.radians(pitch) / 2, math.radians(roll) / 2
    return (math.cos(p) * math.cos(r), math.cos(p) * math.sin(r), math.sin(p) * math.cos(r), -math.sin(p) * math.sin(r))
def mk(name, src):
    if not os.path.isdir(src): return B._make_runner(src)
    k = "stand_dbg_" + name.split("(")[0].replace("-", "_")
    B.CHECKPOINTS[k] = src; B.CKPT_DEFAULT_POSE[k] = list(B.CKPT_DEFAULT_POSE["stand_crouch_g21a"]); B.CKPT_GAIT_FREQ[k] = B.CKPT_GAIT_FREQ["stand_crouch_g21a"]; B.CKPT_REQUIRE_ENV_YAML.add(k)
    return B._make_runner(k)
pose = np.asarray(B.CKPT_DEFAULT_POSE["stand_crouch_g21a"], dtype=np.float32)
print("| 정책 | 고정 기울기(피치, 롤) | |a| 최대: 0–2 s · 2–5 s · 5–10 s | 목표 이탈 최대 [°]: 2 s · 5 s · 10 s | 판정 |"); print("|---|---|---|---|---|")
for name, src in CAND.items():
    for pitch, roll in ((0, 0), (2, 0), (4, 0), (6, 0), (8, 0), (-2, 0), (-4, 0), (-6, 0), (-8, 0), (0, 4), (0, 8), (0, -8)):
        R = mk(name, src); R.reset(); q = pose.copy(); amax = [0.0, 0.0, 0.0]; dev = [0.0, 0.0, 0.0]; bad = None
        for n in range(500):
            out = R.step((0.0, 0.0, 0.0), B.correct_gravity(B.projected_gravity(quat_pr(pitch, roll))), np.zeros(3, np.float32), q, np.zeros(12, np.float32), [0.0, 0.0])
            if not np.isfinite(out).all(): bad = n; break
            seg = 0 if n < 100 else (1 if n < 250 else 2)
            amax[seg] = max(amax[seg], float(np.abs(R.last_action).max())); dev[seg] = max(dev[seg], float(np.degrees(np.abs(out - pose)).max())); q = np.asarray(out, np.float32)
        v = f"비유한(틱 {bad})" if bad is not None else ("폭주" if amax[2] > 20 else ("커짐" if amax[2] > 6 else "유계"))
        print(f"| {name} | {pitch:+d}°, {roll:+d}° | {amax[0]:.2f} · {amax[1]:.3g} · {amax[2]:.3g} | {dev[0]:.0f} · {max(dev[:2]):.0f} · {max(dev):.0f} | {v} |", flush=True)
