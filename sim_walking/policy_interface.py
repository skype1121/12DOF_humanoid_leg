"""정책 인터페이스 — 향후 RL 정책으로 교체 가능한 경계 정의.

runner.run_gait()는 정책 객체를 덕 타이핑으로 사용한다:

  필수:  targets(t: float) -> {관절이름: 목표각[rad]}   (12관절 전부)
  선택:  lat_ref(t) -> float      의도된 골반 롤[deg] (측방 피드백 기준값)
  선택:  observe(obs: dict)       매 관측 주기에 호출됨 (폐루프 정책용)
  선택:  reset()                  에피소드 시작 시

obs 딕셔너리 스키마 (runner가 제공, sim_session.Session 기반):
  t            float  시각 [s]
  pelvis_z/x/y float  골반 월드 위치 [m] (전진 = -y)
  lean_ap      float  전후 기울기 [deg] (+뒤로)
  lean_lat     float  좌우 기울기 [deg] (+왼쪽 +X)
  yaw          float  방향 [deg]
  foot_z       {left,right} 발 링크 z [m] (바닥 접지 ≈ 0.050)
  joint_pos    {관절이름: rad}
  joint_vel    {관절이름: rad/s}

규약:
  - 관절 이름은 config/robot_12dof_hardware_map.json 과 동일한 12개.
  - 부호 의미는 joint_direction(실측) 기준. 해부학 변환은 sim_session.ANAT_SIGN.
  - targets는 60Hz로 호출된다. 리밋은 config/sim_joint_limits.json.

기존 구현체:
  - gait_static.StaticGait  : 개루프 정적 4상 보행 (검증된 기본 데모)
  - gait_fsm.GaitFSM        : (구) 주기 로킹 — 참고용
RL 정책은 observe()에서 obs를 저장하고 targets()에서 행동을 내면 된다.
"""


def build_obs(session, t):
    """Session -> obs dict (스키마 문서와 일치)."""
    import numpy as np
    te = session.telemetry()
    jp = session.art.get_joint_positions()
    jv = session.art.get_joint_velocities()
    return {
        "t": t,
        "pelvis_z": te["pelvis_z"], "pelvis_x": te["pelvis_x"],
        "pelvis_y": te["pelvis_y"],
        "lean_ap": te["lean_ap"], "lean_lat": te["lean_lat"], "yaw": te["yaw"],
        "foot_z": te["foot_z"],
        "joint_pos": {n: float(v) for n, v in zip(session.names, jp)},
        "joint_vel": {n: float(v) for n, v in zip(session.names, jv)},
    }
