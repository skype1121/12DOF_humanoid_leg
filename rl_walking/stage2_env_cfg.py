"""2·3단계 커리큘럼 — 걷기 발현(런1) 후 강건화·다재화·이식정합.

2단계 (발현→강건화 순서 원칙):
- 명령 확장: 전진 0.2~0.8 + 측방 ±0.25 + 요 ±1.0 + 서기 10% (BHL biped 최종 범위 계열)
- 외란 강화: ±0.5 m/s, 4~8s 간격 (kd5 약축 보강)
- 마찰 확대: 0.2~1.25 (Berkeley Humanoid 최종 범위)
- 걸음새 다듬기: action_rate/joint_vel 페널티 상향 (진동 억제 — 저 kd 이식 대비)

3단계 (페이로드 + sim2real 갭 최소화 — 승윤님 요청 2026-07-19 새벽):
- 실물 질량 정합: URDF 10.49kg(12URDF0725) vs 실물 ~12kg → 전 링크 1.09~1.19 스케일 (관성 재계산)
- 페이로드 0~5kg: base 질량 + CoM 오프셋 (배터리 위/앞/뒤 부착 브래키팅)
- 액션 지연 0~1틱(0~20ms) env별 랜덤 — 실측 루프 지연 정합
- 관절 마찰 확대 (0~0.1)

사용: --resume --load_run <직전 런> 으로 워밍스타트 (관측/액션 차원 불변).
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .env_cfg import Biped12FlatEnvCfg


@configclass
class Biped12FlatStage2EnvCfg(Biped12FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # 명령 다양화 (서기 혼합 — stand_still 보상이 이제 실제로 작동)
        self.commands.base_velocity.rel_standing_envs = 0.10
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        # 외란 강화
        self.events.push_robot.interval_range_s = (4.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5)
        }
        # 마찰 확대
        self.events.physics_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.physics_material.params["dynamic_friction_range"] = (0.15, 1.0)
        # 부드러움 (실물 저 kd 진동 대비)
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.joint_vel_l2.weight = -1.0e-3
        # ── 걸음새 가드레일 (2단계 평가에서 케이던스 5.1/s 종종걸음 퇴행 적발 →
        #    "뚜벅뚜벅" 기준 강제. 승윤님 품질 기준 2026-07-19 05:2x) ──
        # 단일지지 지속 보상 유지 (절반 강등이 퇴행 원인이었음 — 원상 복구)
        self.rewards.feet_air_time.weight = 1.0
        # 케이던스 조절기: 에어타임 0.30s 미만 스텝은 음수, 초과는 양수
        # (base mdp.feet_air_time은 (last_air_time - threshold)를 착지 순간 지급)
        from isaaclab.managers import RewardTermCfg as _RewTerm
        from isaaclab.managers import SceneEntityCfg as _SceneCfg
        self.rewards.step_time_regulator = _RewTerm(
            func=mdp.feet_air_time,
            weight=2.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": _SceneCfg("contact_forces", body_names=".*_ankle_r_joint"),
                "threshold": 0.30,
            },
        )


@configclass
class Biped12FlatStage3EnvCfg(Biped12FlatStage2EnvCfg):
    """3단계: 페이로드(배터리 0~5kg 위/앞/뒤) + 실물 질량 정합 + 지연 랜덤화."""

    def __post_init__(self):
        super().__post_init__()
        # 후진 보행 추가 (음성 명령 "뒤로 가" 대비 — 승윤님 최종 목표)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.4, 0.8)
        # 실물 질량 정합: 실측 ~12kg → 전 링크 1.09~1.19 스케일 (11.4~12.5kg 브래키팅)
        # (12URDF0725 자산 자중 10.49kg 기준. 구 9.94kg 자산에서는 1.15~1.26이었음)
        self.events.link_mass_calib = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (1.09, 1.19),
                "operation": "scale",  # recompute_inertia 기본 True
            },
        )
        # 페이로드: 배터리 0~5kg (질량은 base에 가산, CoM은 위/앞/뒤 부착 범위)
        self.events.add_base_mass.params["mass_distribution_params"] = (0.0, 5.0)
        self.events.base_com.params["com_range"] = {
            "x": (-0.08, 0.08),   # 앞/뒤 부착
            "y": (-0.03, 0.03),
            "z": (0.0, 0.15),     # 위 부착 (배터리는 아래로 안 달림)
        }
        # 액션 지연 0~1틱 (실측 루프 지연 ~20ms 정합)
        self.actions.joint_pos.max_delay_steps = 1
        # 관절 마찰 확대
        self.events.joint_friction.params["friction_distribution_params"] = (0.0, 0.1)


@configclass
class Biped12FlatStage3PolishEnvCfg(Biped12FlatStage3EnvCfg):
    """걸음새 폴리시 — 신URDF 미세조정 후 라이브뷰 관찰 결함 교정 (2026-07-25 승윤님):
    ① 발 간격 좁아짐(모아걷기) ② 잔걸음·발 붙는 시간 ③ 뒤뚱거림(롤 요동)."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as _RewTerm
        from isaaclab.managers import SceneEntityCfg as _SceneCfg

        from . import mdp_rewards as custom

        # ① 양발 횡간격 0.18m 미만 벌점 (기본 스탠스 0.287m — 뚜렷한 모임만 잡음)
        self.rewards.feet_gap = _RewTerm(
            func=custom.feet_too_close,
            weight=-5.0,
            params={
                "min_gap": 0.18,
                "asset_cfg": _SceneCfg("robot", body_names=[".*_ankle_r_joint"]),
            },
        )
        # ② 잔걸음 교정: 스텝시간 기준 0.30→0.34s, 비중 2.0→3.0 (케이던스 ~2.9 목표)
        self.rewards.step_time_regulator.params["threshold"] = 0.34
        self.rewards.step_time_regulator.weight = 3.0
        # ③ 뒤뚱(롤·피치 각속도) 감쇠 — 1라운드 −0.2는 종종걸음 유인 확인, 절반 완화
        self.rewards.ang_vel_xy_l2.weight = -0.1
        # ④ 전진 순응 회복 (3라운드) — 1라운드에서 일부 env가 느림/제자리 경향.
        #    2라운드의 에어타임 2배(0.45s)는 "제자리 들썩임 보상 파밍" 평형을 만들어
        #    실패 (전진 0.38m, 발목롤 5.46 위반) → 원복하고 추종 보상 자체를 강화.
        #    (feet_air_time은 1라운드 값 1.0/0.4 유지 = base 그대로)
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        # ⑤ 방폭 — 신자산 평지 물리폭발 크래시 3회 (iter 9786/12047/12941).
        #    캡핑만으론 부족(관측 폭발 경로) → 계단과 동일하게 솔버 강화 동반
        from . import mdp_rewards as capped
        self.rewards.dof_acc_l2.func = capped.joint_acc_l2_capped
        self.rewards.joint_vel_l2.func = capped.joint_vel_l2_capped
        self.rewards.ang_vel_xy_l2.func = capped.ang_vel_xy_l2_capped
        self.rewards.lin_vel_z_l2.func = capped.lin_vel_z_l2_capped
        self.rewards.feet_contact_forces.func = capped.contact_forces_capped
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 12
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 2
