"""Real→Sim 재생 (0912): 실물 블랙박스(50 Hz)의 관절각 + IMU 자세를 Isaac Sim GUI 의 로봇에 그대로 넣어 실물 움직임을 시뮬에서 따라 보이게 한다.
- 관절: qpos_rad (JOINT_ORDER = 시뮬 DOF 순서, 확인됨) 을 텔레포트.
- 베이스: IMU 쿼터니언(base 프레임)에서 요(yaw)만 제거해 카메라를 향하게, 높이는 낮은 쪽 발목이 지면 위 ANKLE_H 가 되도록 매 프레임 맞춤.
- 물리 계산 없음(동역학 재현 아님) — 기록 재생 시각화. 접촉 비트는 발 위 작은 구(초록=접촉, 빨강=비접촉)로 표시.
사용(Isaac Sim 스크립트 콘솔/MCP execute_script):
    exec(open(PATH).read()); start_replay(BB_PATH, speed=1.0, t_start=0.0)
"""
import json, math, numpy as np

ANKLE_H = 0.08          # 발목 롤 관절 원점의 지면 위 높이 가정 [m] (발 두께·패드 포함 근사)
_STATE = {"frames": [], "i": 0, "sub": 0, "cb": None, "markers": {}}


def _load_bb(path, t_start=0.0):
    fr = []
    for l in open(path, encoding="utf-8"):
        try: d = json.loads(l)
        except ValueError: continue
        if d.get("live") and "qpos_rad" in d and "quat_wxyz" in d:
            fr.append(d)
    t0 = fr[0]["timestamp"] if fr else 0.0
    return [f for f in fr if f["timestamp"] - t0 >= t_start], t0


def _quat_wxyz_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _rpy_to_quat_wxyz(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dtype=np.float32)


def start_replay(bb_path, speed=1.0, t_start=0.0, physics_hz=200.0, record_hz=50.0):
    import omni.usd, omni.timeline
    from pxr import UsdGeom, Gf
    from isaacsim.core.api import World
    world = World.instance(); stage = omni.usd.get_context().get_stage()
    robot = world.scene.get_object("biped12")
    frames, t0 = _load_bb(bb_path, t_start)
    _STATE.update(frames=frames, i=0, sub=0, t0=t0)
    per = max(1, int(round(physics_hz / (record_hz * speed))))
    # 접촉 마커 구
    for side, path, col in (("L", "/World/markL", (0.1, 0.9, 0.1)), ("R", "/World/markR", (0.9, 0.1, 0.1))):
        if not stage.GetPrimAtPath(path):
            s = UsdGeom.Sphere.Define(stage, path); s.CreateRadiusAttr(0.03); s.CreateDisplayColorAttr([Gf.Vec3f(*col)])
        _STATE["markers"][side] = path
    if _STATE.get("sub_handle") is not None:
        _STATE["sub_handle"] = None
    import time as _time
    _STATE["wall0"] = _time.time(); _STATE["speed"] = speed; _STATE["rec_t0"] = frames[0]["timestamp"] if frames else 0.0

    def cb(_e):
        st = _STATE
        if not st["frames"]: return
        # 벽시계 기준 프레임 인덱스 (렌더 프레임률과 무관)
        tt = (_time.time() - st["wall0"]) * st["speed"]
        j = min(len(st["frames"]) - 1, int(tt * record_hz))
        if j <= st["i"] - 1 and st["i"] > 0: return
        st["i"] = j + 1
        f = st["frames"][j]
        q = np.array(f["qpos_rad"], dtype=np.float32)
        robot.set_joint_positions(q); robot.set_joint_velocities(np.zeros_like(q))
        r, p, _y = _quat_wxyz_to_rpy(f["quat_wxyz"])
        quat = _rpy_to_quat_wxyz(r, p, 0.0)
        # 높이: 고정(서기 자세에서 발목 원점이 base 아래 0.59 m → base z = 0.59 + ANKLE_H). 링크 xform 은 같은 콜백 안에서 갱신되지 않아 동적 보정은 발산함(0912).
        robot.set_world_pose(position=np.array([0.0, 0.0, 0.59 + ANKLE_H], dtype=np.float32), orientation=quat)
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32)); robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        # 마커: 접촉 비트에 따라 색·높이
        c = f.get("contact") or [0, 0]
        for k, (side, nm) in enumerate((("L", "left_ankle_r_joint"), ("R", "right_ankle_r_joint"))):
            prim = stage.GetPrimAtPath("/World/biped12/" + nm); t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
            m = stage.GetPrimAtPath(st["markers"][side]); xf = UsdGeom.Xformable(m); xf.ClearXformOpOrder(); xf.AddTranslateOp().Set(Gf.Vec3d(t[0], t[1], t[2] + 0.12))
            UsdGeom.Sphere(m).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.9, 0.1) if c[k] else Gf.Vec3f(0.9, 0.1, 0.1)])
        if st["i"] % 250 == 0:
            print(f"[real2sim] t={f['timestamp'] - st['t0']:.1f}s roll={f['rp_deg'][0]:+.1f} pitch={f['rp_deg'][1]:+.1f} contact={c} hang={f.get('hang_state','')[:8]}")

    import omni.kit.app
    _STATE["sub_handle"] = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(cb, name="real2sim")
    _STATE["cb"] = cb
    omni.timeline.get_timeline_interface().play()
    return {"frames": len(frames), "duration_s": (frames[-1]["timestamp"] - frames[0]["timestamp"]) if frames else 0, "per_step": per}


def stop_replay():
    _STATE["sub_handle"] = None; _STATE["cb"] = None
