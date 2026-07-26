"""Stage4(245차원) 정책 → walk_scene 재생 어댑터 — sim2sim 검증용.

policy_adapter.RLWalkPolicy(45차원판)의 Stage4판. 차이점:
- 관측 245 조립(이력 5프레임·게이트클록·last_action·접촉 2ch)은 실물 이식 경로와
  동일 코드인 rl_walking/deploy/policy_runner_stage4.PolicyRunnerStage4에 전부 위임.
  어댑터는 (1) walk_scene 원신호를 러너 입력 규약으로 변환(pelvis→base 프레임 회전,
  관절 이름 매핑), (2) 발 접촉 2ch 조회, (3) 50Hz 틱 스케줄만 담당한다.

접촉 2ch (feet_contact) — PhysX 접촉 리포트(net contact force) 방식 채택:
  isaacsim.core.prims.RigidPrim(track_contact_forces=True)로 발 링크당
  RigidContactView를 만들어 순접촉력을 조회, |F| > 5N → 1.0.
  선택 근거: 학습 관측(mdp_stage4.feet_contact_binary = ContactSensor
  net_forces_w 노름 > 5N)·실물 RA30P(법선힘 > 5N)와 '같은 물리량·같은 문턱'이라
  정의 갭이 없다. 발높이+수직속도 휴리스틱은 문턱 튜닝 의존(발 링크 원점 높이는
  자산·지형마다 다름)·스윙 저공비행 오검출·경계 채터가 있어 접촉 뷰 생성이
  실패할 때의 폴백으로만 사용한다 (mode 속성으로 어느 쪽이 쓰였는지 보고).
  주의: 접촉 리포트 API는 물리 파싱 전에 프리밋에 붙어야 하므로 FootContacts는
  Session.attach(fresh_physics=True) '이전', 타임라인 정지 상태에서 생성할 것.

50Hz 틱: 60fps 프레임에서 누적 스케줄(next += 0.02)로 6프레임당 5틱 = 평균 정확히
  50Hz (틱 간격 33/17/17/17/17ms 지터, 드리프트 0). 구 어댑터의 next = t+0.02
  방식은 실효 30Hz가 되는데, Stage4는 클록 φ=2π·1.4·tick·0.02가 시간을 틱으로
  재구성하므로 30Hz면 위상이 시뮬시간 대비 0.6배로 흘러 부적합 — 누적식 필수.

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
import os

import numpy as np

from sim_walking.sim_session import FOOT_LINKS
from rl_walking.policy_adapter import _R_PB, _quat_to_rot
from rl_walking.deploy.policy_runner_stage4 import (
    CONTACT_FORCE_THRESHOLD_N, JOINT_ORDER, POLICY_DT, PolicyRunnerStage4,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Stage4 R4 최종 채택 (2026-07-26_14-53-03/model_12994, 스탠스 22.6cm) — sim2sim 기본 대상
DEFAULT_V2_EXPORT = os.path.join(
    REPO, "logs", "rsl_rl", "biped12_stage4", "2026-07-26_14-53-03", "exported")


class FootContacts:
    """발 접촉 2ch [왼발, 오른발] 이진 — PhysX net contact force 우선, 휴리스틱 폴백.

    생성 시점 규약: 타임라인 '정지' 상태에서 (Session.attach 전에) 생성해야
    접촉 리포트 API가 물리 파싱에 포함된다. 생성 → Session.attach →
    zero(settle) 직립 후 initialize(session) 순서로 쓸 것.
    initialize가 직립 양발하중 상태에서 |F|>5N 자가검증을 하고, 실패하면
    mode='heuristic'(발높이+수직속도)으로 자동 전환한다.
    """

    def __init__(self, phys_dt=1.0 / 60.0):
        import omni.usd
        from pxr import Usd
        from isaacsim.core.prims import RigidPrim

        self.phys_dt = float(phys_dt)
        self.mode = "physx"
        self.fail_reason = None
        self._z0 = None                     # 휴리스틱 기준 발높이 {side: z}
        self.views = {}
        stage = omni.usd.get_context().get_stage()
        old = stage.GetEditTarget()
        # 세션 레이어에 API를 기록 — walk_scene.usd 루트 레이어 오염 방지
        stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
        try:
            for side, path in FOOT_LINKS.items():
                self.views[side] = RigidPrim(
                    path, name=f"s4_contact_{side}",
                    track_contact_forces=True,
                    prepare_contact_sensors=True,
                    reset_xform_properties=False,
                )
        except Exception as e:  # noqa: BLE001 — 폴백 경로로 계속
            self.mode = "heuristic"
            self.fail_reason = f"뷰 생성 실패: {type(e).__name__}: {e}"
        finally:
            stage.SetEditTarget(old)

    def initialize(self, session):
        """직립(양발 접지) 상태에서 호출 — 뷰 초기화 + 자가검증 + 휴리스틱 기준 채집."""
        st = session.foot_state()
        self._z0 = {side: float(v["pos"][2]) for side, v in st.items()}
        if self.mode != "physx":
            return
        try:
            for v in self.views.values():
                v.initialize()
            f = self.forces()
            norms = {s: float(np.linalg.norm(v)) for s, v in f.items()}
            if not all(np.isfinite(list(norms.values()))):
                raise RuntimeError(f"접촉력 비유한: {norms}")
            # 직립 양발하중이면 양쪽 다 문턱을 넘어야 함 (자중 ~10kg → 발당 ~49N)
            if not all(n > CONTACT_FORCE_THRESHOLD_N for n in norms.values()):
                raise RuntimeError(f"직립인데 접촉력 문턱 미달: {norms}")
            self.standing_forces_n = norms
        except Exception as e:  # noqa: BLE001
            self.mode = "heuristic"
            self.fail_reason = f"초기화/자가검증 실패: {type(e).__name__}: {e}"

    def forces(self):
        """{side: F[3] [N]} — PhysX 순접촉력 (임펄스/물리dt)."""
        return {side: np.asarray(
                    v.get_net_contact_forces(dt=self.phys_dt), dtype=float).reshape(3)
                for side, v in self.views.items()}

    def binary(self, session):
        """[왼발, 오른발] float32 이진 — PolicyRunnerStage4.contact2 규약."""
        if self.mode == "physx":
            f = self.forces()
            return np.float32([
                1.0 if np.linalg.norm(f["left"]) > CONTACT_FORCE_THRESHOLD_N else 0.0,
                1.0 if np.linalg.norm(f["right"]) > CONTACT_FORCE_THRESHOLD_N else 0.0,
            ])
        # 휴리스틱 폴백: 직립 기준높이 대비 +8mm 미만 '그리고' 수직속도 작음 → 접지
        st = session.foot_state()
        out = []
        for side in ("left", "right"):
            z = float(st[side]["pos"][2])
            vz = float(st[side]["vel"][2])
            out.append(1.0 if (z < self._z0[side] + 0.008 and abs(vz) < 0.15) else 0.0)
        return np.float32(out)


def read_signals(session):
    """walk_scene 원신호 → 러너 입력 (gyro_b, grav_b, qpos, qvel) — 전부 학습 base 프레임/JOINT_ORDER.

    프레임 변환은 policy_adapter._observe와 동일: world → pelvis → base(_R_PB).
    """
    s = session
    _, quat = s.pelvis.get_world_poses()
    vels = s.pelvis.get_velocities()
    if isinstance(vels, (tuple, list)):
        ang_w = np.asarray(vels[1][0], dtype=float)
    else:
        ang_w = np.asarray(vels[0][3:6], dtype=float)
    Rwp = _quat_to_rot(quat[0])
    g_b = _R_PB @ (Rwp.T @ np.array([0.0, 0.0, -1.0]))
    w_b = _R_PB @ (Rwp.T @ ang_w)
    jp = s.art.get_joint_positions()
    jv = s.art.get_joint_velocities()
    jp = np.asarray(jp[0] if getattr(jp, "ndim", 1) > 1 else jp, dtype=float)
    jv = np.asarray(jv[0] if getattr(jv, "ndim", 1) > 1 else jv, dtype=float)
    q = np.array([jp[s.idx[n]] for n in JOINT_ORDER], dtype=np.float32)
    qd = np.array([jv[s.idx[n]] for n in JOINT_ORDER], dtype=np.float32)
    return w_b.astype(np.float32), g_b.astype(np.float32), q, qd


class RLWalkPolicyStage4:
    """policy_interface 규약(targets/reset/lat_ref) — PolicyRunnerStage4 구동.

    runner.run_gait(s, self, balance=False)로 구동. balance=False 필수
    (RL 정책 자체 밸런스 — FSM용 피드백과 중첩 금지).
    """

    def __init__(self, contacts: FootContacts, checkpoint=None,
                 cmd=(0.5, 0.0, 0.0), session=None, backend="auto"):
        self.runner = PolicyRunnerStage4(
            model_path=(checkpoint or DEFAULT_V2_EXPORT), backend=backend)
        self.contacts = contacts
        self.cmd = np.array(cmd, dtype=np.float32)
        self.s = session
        self._next_tick = 0.0
        self._targets = None                # {관절명: rad} — 틱 사이 유지
        # 진단 통계 (sim2sim 리포트용)
        self.n_ticks = 0
        self.contact_sum = np.zeros(2)
        self._prev_c = np.float32([1.0, 1.0])
        self._air_ticks = [0, 0]
        self.steps_contact = [0, 0]         # 접촉 에지 기반 걸음수 [L, R] (공중≥2틱)

    def attach(self, session):
        self.s = session

    def reset(self):
        self.runner.reset()
        self._next_tick = 0.0
        self._targets = None
        self.n_ticks = 0
        self.contact_sum[:] = 0.0
        self._prev_c = np.float32([1.0, 1.0])
        self._air_ticks = [0, 0]
        self.steps_contact = [0, 0]

    # ---------- policy_interface ----------
    def targets(self, t):
        if self.s is None:
            raise RuntimeError("attach(session) 필요")
        if t + 1e-9 >= self._next_tick:     # 누적 스케줄 → 평균 정확히 50Hz
            self._next_tick += POLICY_DT
            gyro, grav, qpos, qvel = read_signals(self.s)
            c2 = self.contacts.binary(self.s)
            emitted = self.runner.step(gyro, grav, self.cmd, qpos, qvel, c2)
            self._targets = {n: float(v) for n, v in zip(JOINT_ORDER, emitted)}
            # 통계: 접촉 듀티 + 에지 기반 걸음수 (터치다운 시 공중 2틱=0.04s 이상만)
            self.n_ticks += 1
            self.contact_sum += c2
            for i in range(2):
                if c2[i] == 0.0:
                    self._air_ticks[i] += 1
                else:
                    if self._prev_c[i] == 0.0 and self._air_ticks[i] >= 2:
                        self.steps_contact[i] += 1
                    self._air_ticks[i] = 0
            self._prev_c = c2
        if self._targets is None:           # 방어 — 첫 호출 전 접근
            _, _, qpos, _ = read_signals(self.s)
            self._targets = {n: float(v) for n, v in zip(JOINT_ORDER, qpos)}
        return self._targets

    def lat_ref(self, t):
        return 0.0

    def stats(self):
        n = max(self.n_ticks, 1)
        return {
            "policy_ticks": self.n_ticks,
            "contact_duty_L": round(float(self.contact_sum[0]) / n, 3),
            "contact_duty_R": round(float(self.contact_sum[1]) / n, 3),
            "steps_contact_L": self.steps_contact[0],
            "steps_contact_R": self.steps_contact[1],
            "contact_mode": self.contacts.mode,
            "contact_fail_reason": self.contacts.fail_reason,
            "backend": self.runner.backend,
        }
