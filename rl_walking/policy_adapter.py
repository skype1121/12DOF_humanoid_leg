"""RL 정책 → 기존 walk_scene(sim_walking Session) 재생 어댑터.

용도: Isaac Sim GUI의 walk_scene.usd 안에서 학습된 정책을 눈으로 확인.
      (학습은 Isaac Lab 환경에서 했지만, 데모·비교는 기존 씬에서)

프레임 변환이 전부다:
- 학습 base 프레임: +X 전방, +Y 좌, +Z 상 (URDF에 주입한 base 링크)
- walk_scene pelvis 프레임: SolidWorks 원본 — +Y 상, +Z 전방, +X 좌
  (임포트 시 X+90° 회전으로 세워져 있음, 전방 = 월드 -Y)
- 관계: v_base = R @ v_pelvis,  R = Rz(90°)·Rx(90°)
  (R의 열 = pelvis 축의 base 좌표: X_p→(0,1,0), Y_p→(0,0,1), Z_p→(1,0,0))

관측 조립 (학습 obs 순서 그대로, 45차원):
  [base_ang_vel(3), projected_gravity(3), commands(3),
   joint_pos-default(12), joint_vel(12), last_action(12)]
관절 순서는 학습 환경 실측 순서(JOINT_ORDER) — walk_scene의 dof 순서와
다를 수 있으므로 반드시 이름으로 매핑한다.

액션 → 목표각: target = default + 0.25 * action, 슬루 ±4°/정책틱(50Hz).
시뮬 60fps에서 50Hz 정책틱은 6프레임당 5회 갱신으로 재현 (실물 노드와 동일 레이트).

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.

사용 (Isaac Sim Script Editor 또는 execute_script):
    import sys; sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
    from rl_walking.policy_adapter import RLWalkPolicy
    from sim_walking.sim_session import Session
    s = Session.attach(fresh_physics=True)
    pol = RLWalkPolicy(cmd=(0.5, 0.0, 0.0))   # 전진 0.5 m/s
    from sim_walking import runner as RN
    RN.run_gait(s, pol, duration_s=20.0, tag="rl_demo", balance=False)
    # balance=False 필수! RL 정책이 자체 밸런스 — FSM용 피드백과 중첩 금지
"""
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 학습 환경 관절 순서 (log_picture/03_스폰검증_리포트.txt 실측)
JOINT_ORDER = [
    "left_hip_f_joint", "right_hip_f_joint",
    "left_hip_a_joint", "right_hip_a_joint",
    "left_hip_r_joint", "right_hip_r_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_f_joint", "right_ankle_f_joint",
    "left_ankle_r_joint", "right_ankle_r_joint",
]

# 학습 기본자세 (biped12_cfg.BIPED12_DEFAULT_JOINT_POS와 동일)
DEFAULT_POS = {
    "left_hip_f_joint": 0.0, "left_hip_a_joint": 0.0, "left_hip_r_joint": 0.0,
    "left_knee_joint": -0.10471975511965977,
    "left_ankle_f_joint": 0.0244, "left_ankle_r_joint": 0.0,
    "right_hip_f_joint": 0.0, "right_hip_a_joint": 0.0, "right_hip_r_joint": 0.0,
    "right_knee_joint": 0.10471975511965977,
    "right_ankle_f_joint": -0.0244, "right_ankle_r_joint": 0.0,
}

ACTION_SCALE = 0.25
SLEW_RAD = np.deg2rad(4.0)          # 4°/정책틱 (실물 노드와 동일)
POLICY_DT = 0.02                    # 50Hz

# pelvis(원본 프레임) → 학습 base 프레임 회전
_R_PB = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def _quat_to_rot(q):
    """(w,x,y,z) 쿼터니언 → 회전행렬 (world<-body)."""
    w, x, y, z = [float(v) for v in q]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class RLWalkPolicy:
    """policy_interface 규약 준수 + Session 직접 참조로 IMU급 관측 확보.

    runner.run_gait(s, self, balance=False)로 구동:
      - targets(t): 60fps 호출 — 내부 50Hz 클록으로 정책 갱신, 사이엔 유지
      - reset(): 에피소드 초기화
    """

    def __init__(self, checkpoint=None, cmd=(0.5, 0.0, 0.0), session=None):
        import torch  # Isaac Sim 파이썬에 torch 존재
        self._torch = torch
        ckpt = checkpoint or self._latest_export()
        self.net = torch.jit.load(ckpt, map_location="cpu")
        self.net.eval()
        self.cmd = np.array(cmd, dtype=np.float32)
        self.s = session
        self.last_action = np.zeros(12, dtype=np.float32)
        self._target = None          # 슬루 기준점 (JOINT_ORDER 순)
        self._next_tick = 0.0
        self._default = np.array([DEFAULT_POS[n] for n in JOINT_ORDER], dtype=np.float32)

    @staticmethod
    def _latest_export():
        import glob
        cands = sorted(glob.glob(os.path.join(
            REPO, "logs", "rsl_rl", "biped12_flat", "*", "exported", "policy.pt")))
        if not cands:
            raise FileNotFoundError("exported/policy.pt 없음 — eval_policy.py 먼저 실행")
        return cands[-1]

    def attach(self, session):
        self.s = session

    def reset(self):
        self.last_action[:] = 0.0
        self._target = None
        self._next_tick = 0.0

    # ---------- 관측 ----------
    def _observe(self):
        s = self.s
        # pelvis 자세/각속도 (world)
        pos, quat = s.pelvis.get_world_poses()
        vels = s.pelvis.get_velocities()
        # RigidPrim API 방어: [N,6] 또는 (lin[N,3], ang[N,3]) 둘 다 처리
        if isinstance(vels, (tuple, list)):
            ang_w = np.asarray(vels[1][0], dtype=float)
        else:
            ang_w = np.asarray(vels[0][3:6], dtype=float)
        Rwp = _quat_to_rot(quat[0])               # world <- pelvis
        # world → pelvis → 학습 base
        g_p = Rwp.T @ np.array([0.0, 0.0, -1.0])
        g_b = _R_PB @ g_p
        w_p = Rwp.T @ ang_w
        w_b = _R_PB @ w_p
        # 관절 (이름 매핑 — walk_scene dof 순서와 무관하게, sim_session의 idx 사용)
        jp = s.art.get_joint_positions()
        jv = s.art.get_joint_velocities()
        jp = np.asarray(jp[0] if getattr(jp, "ndim", 1) > 1 else jp, dtype=float)
        jv = np.asarray(jv[0] if getattr(jv, "ndim", 1) > 1 else jv, dtype=float)
        q = np.array([jp[s.idx[n]] for n in JOINT_ORDER], dtype=np.float32)
        qd = np.array([jv[s.idx[n]] for n in JOINT_ORDER], dtype=np.float32)
        obs = np.concatenate([
            w_b.astype(np.float32),
            g_b.astype(np.float32),
            self.cmd,
            (q - self._default),
            qd,
            self.last_action,
        ])
        return obs, q

    # ---------- policy_interface ----------
    def targets(self, t):
        if self.s is None:
            raise RuntimeError("attach(session) 필요")
        if self._target is None:
            _, q = self._observe()
            self._target = q.copy()
        if t >= self._next_tick:                  # 50Hz 정책틱
            self._next_tick = t + POLICY_DT
            obs, _ = self._observe()
            with self._torch.no_grad():
                a = self.net(self._torch.from_numpy(obs).unsqueeze(0))[0].numpy()
            self.last_action = a.astype(np.float32)
            raw = self._default + ACTION_SCALE * self.last_action
            delta = np.clip(raw - self._target, -SLEW_RAD, SLEW_RAD)
            self._target = self._target + delta
        return {n: float(v) for n, v in zip(JOINT_ORDER, self._target)}

    def lat_ref(self, t):
        return 0.0
