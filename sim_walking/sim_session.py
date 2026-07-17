"""Isaac Sim GUI 세션용 12DOF 보행 개발 헬퍼.

MCP execute_script(cwd=repo)에서:
    import importlib, sim_walking.sim_session as S
    importlib.reload(S)
    s = S.Session.attach()

시뮬 전용. 실물 모터/CAN에는 아무것도 보내지 않는다.
관절 접근은 전부 '이름 키' 기반 (12DOF 하드웨어맵과 동일한 이름).
"""
import json
import os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROBOT_PATH = "/World/newURDF12DOF"
PELVIS_PATH = ROBOT_PATH + "/pelvis"
FOOT_LINKS = {"left": ROBOT_PATH + "/left_ankle_r_joint",
              "right": ROBOT_PATH + "/right_ankle_r_joint"}

# 해부학적 동작 -> (관절, 부호) 매핑. joint_direction 실측 기준.
# 값: 왼다리 부호, 오른다리 부호  (같은 해부학 동작을 낼 때)
ANAT_SIGN = {
    "hip_flexion":   {"joint": "hip_f_joint",   "left": +1, "right": -1},
    "hip_abduction": {"joint": "hip_a_joint",   "left": -1, "right": +1},
    "hip_ext_rot":   {"joint": "hip_r_joint",   "left": +1, "right": -1},
    "knee_flexion":  {"joint": "knee_joint",    "left": -1, "right": +1},
    "dorsiflexion":  {"joint": "ankle_f_joint", "left": +1, "right": -1},
    "eversion":      {"joint": "ankle_r_joint", "left": +1, "right": +1},
}


def anat(side, action, amount_rad):
    """(side, 해부학동작, 크기[rad]) -> (관절이름, 부호부여값)"""
    a = ANAT_SIGN[action]
    return f"{side}_{a['joint']}", a[side] * amount_rad


class Session:
    _inst = None

    def __init__(self, art, pelvis, feet, app):
        self.art = art
        self.pelvis = pelvis
        self.feet = feet
        self.app = app
        self.names = list(art.dof_names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.nd = art.num_dof
        self._last_target = np.zeros(self.nd)

    # ---------- 초기화 ----------
    @classmethod
    def attach(cls, fresh_physics=False, kp=None, kd=None):
        import omni.timeline
        import omni.kit.app
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation, RigidPrim

        tl = omni.timeline.get_timeline_interface()
        app = omni.kit.app.get_app()
        if fresh_physics:
            tl.stop()
            for _ in range(8):
                app.update()
            try:
                World.clear_instance()
            except Exception:
                pass
        if not tl.is_playing():
            tl.play()
            for _ in range(12):
                app.update()
        w = World.instance() or World()
        w.initialize_physics()
        _ = w.physics_sim_view  # lazy-create tensor view

        art = SingleArticulation(ROBOT_PATH, name="sw_art")
        art.initialize()
        pelvis = RigidPrim(PELVIS_PATH, name="sw_pelvis")
        try:
            pelvis.initialize()
        except Exception:
            pass
        feet = {}
        for side, path in FOOT_LINKS.items():
            rp = RigidPrim(path, name=f"sw_foot_{side}")
            try:
                rp.initialize()
            except Exception:
                pass
            feet[side] = rp

        s = cls(art, pelvis, feet, app)
        cfg = json.load(open(os.path.join(REPO, "config", "sim_dynamics.json")))
        g = cfg["sim_gains"]["default"]
        s.set_gains(kp if kp is not None else g["kp"],
                    kd if kd is not None else g["kd"])
        s.set_torque_limit(cfg["torque_limit_nm"])
        cls._inst = s
        return s

    def set_gains(self, kp, kd):
        self.art.get_articulation_controller().set_gains(
            kps=np.full(self.nd, float(kp)), kds=np.full(self.nd, float(kd)))

    def set_torque_limit(self, tmax):
        try:
            self.art._articulation_view.set_max_efforts(
                np.full((1, self.nd), float(tmax)))
        except Exception:
            pass  # USD maxForce가 이미 적용됨

    # ---------- 스텝/명령 ----------
    def step(self, n=1):
        for _ in range(n):
            self.app.update()

    def cmd(self, named=None, absolute=None):
        """named: {관절이름: rad} (미지정 관절은 이전 목표 유지).
        absolute: 길이 nd 배열로 전체 목표 교체."""
        from isaacsim.core.utils.types import ArticulationAction
        if absolute is not None:
            t = np.asarray(absolute, float).copy()
        else:
            t = self._last_target.copy()
            for k, v in (named or {}).items():
                t[self.idx[k]] = float(v)
        self._last_target = t
        self.art.apply_action(ArticulationAction(joint_positions=t))

    def zero(self, settle=60):
        self.cmd(absolute=np.zeros(self.nd))
        self.step(settle)

    # ---------- 상태 ----------
    def pelvis_pose(self):
        """(pos[3], roll, pitch, yaw[deg]) — 월드 기준. 로봇 정면=-Y.
        주의: pelvis 링크는 임포트 시 X축 +90° 회전돼 있어 roll 기준값이 90."""
        p, q = self.pelvis.get_world_poses()
        q = np.asarray(q[0], float)
        wq, x, y, z = q
        roll = np.degrees(np.arctan2(2 * (wq * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (wq * y - z * x), -1, 1)))
        yaw = np.degrees(np.arctan2(2 * (wq * z + x * y), 1 - 2 * (y * y + z * z)))
        return np.asarray(p[0], float), roll, pitch, yaw

    def lean(self):
        """직립 대비 기울기: (전후 lean_ap[+뒤로], 좌우 lean_lat)[deg]"""
        _, roll, pitch, _ = self.pelvis_pose()
        return (90.0 - roll), pitch

    def joints_deg(self):
        return {n: float(v) for n, v in
                zip(self.names, np.degrees(self.art.get_joint_positions()))}

    def joint_vel_max(self):
        return float(np.max(np.abs(self.art.get_joint_velocities())))

    def foot_state(self):
        out = {}
        for side, rp in self.feet.items():
            p, _ = rp.get_world_poses()
            v = rp.get_linear_velocities()
            out[side] = {"pos": np.asarray(p[0], float),
                         "vel": np.asarray(v[0], float)}
        return out

    def telemetry(self):
        pos, roll, pitch, yaw = self.pelvis_pose()
        ap, lat = self.lean()
        fs = self.foot_state()
        return {
            "pelvis_z": float(pos[2]), "pelvis_x": float(pos[0]),
            "pelvis_y": float(pos[1]),
            "lean_ap": float(ap), "lean_lat": float(lat), "yaw": float(yaw),
            "foot_z": {s: float(v["pos"][2]) for s, v in fs.items()},
            "jvel_max": self.joint_vel_max(),
        }

    def fallen(self):
        pos, _, _, _ = self.pelvis_pose()
        ap, lat = self.lean()
        return pos[2] < 0.45 or abs(ap) > 30 or abs(lat) > 30

    # ---------- 카메라/캡처 (세션 레이어 전용 — 파일 오염 없음) ----------
    def make_cam(self, path, eye, target, focal=24.0):
        import omni.usd
        from pxr import Usd, UsdGeom, Gf
        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        old = stage.GetEditTarget()
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
        try:
            eye = np.array(eye, float)
            target = np.array(target, float)
            up = np.array([0, 0, 1.0])
            f = target - eye
            f /= np.linalg.norm(f)
            r = np.cross(f, up)
            r /= np.linalg.norm(r)
            u = np.cross(r, f)
            R = np.column_stack([r, u, -f])
            t = np.trace(R)
            if t > 0:
                sq = np.sqrt(t + 1) * 2
                qw = 0.25 * sq
                qx = (R[2, 1] - R[1, 2]) / sq
                qy = (R[0, 2] - R[2, 0]) / sq
                qz = (R[1, 0] - R[0, 1]) / sq
            else:
                i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
                if i == 0:
                    sq = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                    qw = (R[2, 1] - R[1, 2]) / sq
                    qx = 0.25 * sq
                    qy = (R[0, 1] + R[1, 0]) / sq
                    qz = (R[0, 2] + R[2, 0]) / sq
                elif i == 1:
                    sq = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
                    qw = (R[0, 2] - R[2, 0]) / sq
                    qx = (R[0, 1] + R[1, 0]) / sq
                    qy = 0.25 * sq
                    qz = (R[1, 2] + R[2, 1]) / sq
                else:
                    sq = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
                    qw = (R[1, 0] - R[0, 1]) / sq
                    qx = (R[0, 2] + R[2, 0]) / sq
                    qy = (R[1, 2] + R[2, 1]) / sq
                    qz = 0.25 * sq
            qn = np.array([qw, qx, qy, qz])
            qn /= np.linalg.norm(qn)
            cam = UsdGeom.Camera.Define(stage, path)
            xf = UsdGeom.Xformable(cam.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*eye.tolist()))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
                Gf.Quatf(float(qn[0]), Gf.Vec3f(float(qn[1]), float(qn[2]), float(qn[3]))))
            cam.GetFocalLengthAttr().Set(focal)
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100000))
        finally:
            stage.SetEditTarget(old)
        return path

    def capture(self, out_path, cam_path=None):
        import omni.kit.viewport.utility as vpu
        vp = vpu.get_active_viewport()
        if cam_path:
            vp.camera_path = cam_path
            self.step(8)
        vpu.capture_viewport_to_file(vp, out_path)
        self.step(22)
        return os.path.exists(out_path)
