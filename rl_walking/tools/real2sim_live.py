"""Real→Sim 라이브 동기화 (0912): 젯슨 콘솔 /api/state 를 백그라운드 스레드로 폴링해 stage8 관절 실제각(모터 프레임 deg) + IMU 쿼터니언을
Isaac Sim 로봇에 매 렌더 프레임 텔레포트한다. 관절 변환은 하드웨어맵 규약 joint_rad = radians(DIR × (motor_deg − SZ)).
토크·명령 없음(읽기만). 요는 시작 시점 기준 상대 추종, 높이는 발 최저점이 지면이 되도록 수렴 보정. 접촉 비트 마커(초록/빨강).
사용: exec(open(PATH).read()); start_live(host="172.16.35.89", hwmap="/home/ryu/humanoid_leg_test1/config/robot_12dof_hardware_map.json")
IMU 장착 편향(시각화 전용): 로봇이 기준 자세(지그·양발 수평 접지)일 때 zero_imu_here() 로 원시 쿼터니언 roll/pitch 를 캡처해 뺀다.
  ⚠ 부호: 쿼터니언 roll = −(콘솔 lat), pitch = +(콘솔 fwd). 0912 에 콘솔 부호로 set_imu_bias(1.2, −1.0) 를 넣어 롤 편향이 상쇄가 아니라
  2 배(−2.45°) 가 됐고 "발 높이 13 mm 차·기울어진 시뮬" 오진(USD 자산 비대칭 의심) 을 낳았다 — base 프레임 기준 링크 원점은 좌우 완전 대칭(y_sum·dz 0.0 mm).
"""
import json, math, threading, time, urllib.request, numpy as np

ANKLE_H = 0.08
_L = {"latest": None, "thread": None, "run": False, "sub": None, "n": 0, "err": None}
JO = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint", "left_hip_r_joint", "right_hip_r_joint",
      "left_knee_joint", "right_knee_joint", "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]


def _rpy(q):
    w, x, y, z = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _quat(r, p, y=0.0):
    cr, sr, cp, sp, cy, sy = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2), math.cos(y / 2), math.sin(y / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], dtype=np.float32)


def _poll(host, period):
    url = f"http://{host}:8800/api/state"
    while _L["run"]:
        try:
            d = json.load(urllib.request.urlopen(url, timeout=1.5)); _L["latest"] = d; _L["n"] += 1; _L["err"] = None
        except Exception as e:
            _L["err"] = str(e)[:80]
        time.sleep(period)


def start_live(host="172.16.35.89", hwmap="/home/ryu/humanoid_leg_test1/config/robot_12dof_hardware_map.json", period=0.05):
    import omni.usd, omni.timeline, omni.kit.app
    from pxr import UsdGeom, Gf
    from isaacsim.core.api import World
    from isaacsim.core.utils.types import ArticulationAction
    world = World.instance(); stage = omni.usd.get_context().get_stage(); robot = world.scene.get_object("biped12")
    m = json.load(open(hwmap)); jm = m.get("joints", m)
    DIR = {k: float(v["direction"]) for k, v in jm.items()}; SZ = {k: float(v["stand_zero_deg"]) for k, v in jm.items()}
    for side, path, col in (("L", "/World/markL", (0.1, 0.9, 0.1)), ("R", "/World/markR", (0.9, 0.1, 0.1))):
        if not stage.GetPrimAtPath(path):
            s = UsdGeom.Sphere.Define(stage, path); s.CreateRadiusAttr(0.03); s.CreateDisplayColorAttr([Gf.Vec3f(*col)])
    stop_live()
    _L["run"] = True; _L["thread"] = threading.Thread(target=_poll, args=(host, period), daemon=True); _L["thread"].start()
    last = {"q": None, "quat": None, "stamp": None}

    def cb(_e):
        d = _L["latest"]
        if not d: return
        s8 = d.get("stage8") or {}; js = s8.get("joints") or {}; imu = d.get("imu") or {}
        q = None
        if len(js) == 12 and all(isinstance(js.get(n, {}).get("latest_actual_deg"), (int, float)) for n in JO):
            q = np.array([math.radians(DIR[n] * (js[n]["latest_actual_deg"] - SZ[n])) for n in JO], dtype=np.float32)
        quat = None
        qq = imu.get("quat")
        if isinstance(qq, list) and len(qq) == 4:
            r, p, yw = _rpy(qq)
            if _L.get("yaw0") is None: _L["yaw0"] = yw          # 시작 시점 요를 0 으로(카메라 정면), 이후 회전은 그대로 추종
            r -= math.radians(_L.get("roll_bias_deg", 0.0)); p -= math.radians(_L.get("pitch_bias_deg", 0.0))   # IMU 장착 편향 보정(시각화용; 0912 지그 양발 접지에서 lat +1.3° 측정)
            quat = _quat(r, p, yw - _L["yaw0"])
        if q is not None and (last["q"] is None or np.any(np.abs(q - last["q"]) > 1e-4)):
            # 텔레포트 + 드라이브 목표도 같은 각으로 (0912 버그: 드라이브 목표가 이전 정적 자세(T4)에 남아 프레임 사이에 관절을 되끌어 발 높이·기울기 왜곡)
            robot.set_joint_positions(q); robot.set_joint_velocities(np.zeros_like(q)); robot.apply_action(ArticulationAction(joint_positions=q)); last["q"] = q
        if quat is not None:
            # 발이 바닥에 닿도록: 이전 프레임에 관측된 발 메시 최저점(min z)이 0 이 되게 base 높이를 비례 보정(한 프레임 지연, 수렴형)
            try:
                bb = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
                mins = []
                for nm in ("left_ankle_r_joint", "right_ankle_r_joint"):
                    pr = stage.GetPrimAtPath("/World/biped12/" + nm)
                    if pr: mins.append(bb.ComputeWorldBound(pr).ComputeAlignedRange().GetMin()[2])
                if mins:
                    err = 0.0 - min(mins)
                    _L["z"] = float(np.clip(_L.get("z", 0.59 + ANKLE_H) + 0.5 * err, 0.3, 1.2))
            except Exception as e:
                _L["z_err"] = str(e)[:80]
            robot.set_world_pose(position=np.array([0.0, 0.0, _L.get("z", 0.59 + ANKLE_H)], dtype=np.float32), orientation=quat)
        robot.set_linear_velocity(np.zeros(3, dtype=np.float32)); robot.set_angular_velocity(np.zeros(3, dtype=np.float32))
        c = (d.get("foot") or {}).get("contact") or [False, False]
        for k, (side, nm) in enumerate((("L", "left_ankle_r_joint"), ("R", "right_ankle_r_joint"))):
            prim = stage.GetPrimAtPath("/World/biped12/" + nm); t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
            mk = stage.GetPrimAtPath("/World/mark" + side); xf = UsdGeom.Xformable(mk); xf.ClearXformOpOrder(); xf.AddTranslateOp().Set(Gf.Vec3d(t[0], t[1], t[2] + 0.12))
            UsdGeom.Sphere(mk).GetDisplayColorAttr().Set([Gf.Vec3f(0.1, 0.9, 0.1) if c[k] else Gf.Vec3f(0.9, 0.1, 0.1)])

    _L["sub"] = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(cb, name="real2sim_live")
    omni.timeline.get_timeline_interface().play()
    return {"host": host, "period": period}


def set_imu_bias(roll_deg=0.0, pitch_deg=0.0):
    """쿼터니언 규약(roll = −lat, pitch = +fwd)의 편향 [deg]. 콘솔 값으로 넣으려면 set_imu_bias_console()."""
    _L["roll_bias_deg"] = float(roll_deg); _L["pitch_bias_deg"] = float(pitch_deg)


def set_imu_bias_console(lat_deg=0.0, fwd_deg=0.0):
    """콘솔 규약(lat + = 왼쪽, fwd + = 앞) 으로 편향 지정."""
    set_imu_bias(-float(lat_deg), float(fwd_deg))


def zero_imu_here():
    """지금 폴링된 IMU 자세를 기준(수평)으로 캡처. 로봇이 실제로 수평 접지된 상태에서만 호출."""
    d = _L["latest"] or {}; qq = (d.get("imu") or {}).get("quat")
    if not (isinstance(qq, list) and len(qq) == 4): return None
    r, p, _ = _rpy(qq); set_imu_bias(math.degrees(r), math.degrees(p))
    return {"roll_bias_deg": round(math.degrees(r), 2), "pitch_bias_deg": round(math.degrees(p), 2)}


def rezero_yaw():
    _L["yaw0"] = None


def status():
    d = _L["latest"] or {}; s8 = d.get("stage8") or {}; js = s8.get("joints") or {}
    ages = [js[n].get("feedback_age_sec") for n in JO if n in js]
    return {"polls": _L["n"], "err": _L["err"], "fresh": d.get("fresh"), "joints": len(js), "feedback_age_max": max([a for a in ages if isinstance(a, (int, float))], default=None),
            "imu": {k: (d.get("imu") or {}).get(k) for k in ("fwd", "lat")}, "contact": (d.get("foot") or {}).get("contact")}


def stop_live():
    _L["run"] = False; _L["sub"] = None
