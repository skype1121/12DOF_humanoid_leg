"""Hanging-to-Stand 학습 환경 — V18(정정 세대) 위에 보조지지 스케줄만 얹는다 (0908 저녁).

커리큘럼 (승윤님 지시 0908):
  S1  발이 이미 바닥에 닿아 있고 지지 50~70% → 유지 0.5~2s → 2~8s 에 걸쳐 0 → 5s 이상 서기.
  S2  발–지면 간격 2~6cm, 지지 90~100% 에서 시작 → 접지 → 하중 이전 → 서기.
  S3  실제 매달림에 가까운 간격 5~15cm, 지지 100%, 유지·방출 시각 넓게.
  DR(질량·CoM·마찰·게인·지연·캘리브·센서)은 V18 대역 그대로 상속 (S4 확대는 별도 클래스).
관측 245 계약 불변 → v18b(2026-09-05_19-37-39/model_120998) 웜스타트 가능.
학습:  ~/IsaacLab/isaaclab.sh -p rl_walking/scripts/train.py --task Biped12-HangStand-S1-v0 --headless \
         --resume --load_run 2026-09-05_19-37-39 --checkpoint model_120998.pt --max_iterations 3000
평가:  rl_walking/scripts/eval_hang2stand.py (지지 0 이후 5s 생존율 등)
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp_hang
from .stage4_env_cfg import _FEET_SENSOR_CFG
from .stage4_v2_env_cfg import Biped12Stage4V2StandV18EnvCfg

_BASE = SceneEntityCfg("robot", body_names="base")


@configclass
class Biped12HangStandS1EnvCfg(Biped12Stage4V2StandV18EnvCfg):
    """S1: 접지 상태 + 지지 50~70% → 0 → 자유 기립."""

    r0_range = (0.5, 0.7)
    hold_range = (0.5, 2.0)
    release_range = (2.0, 8.0)
    clearance_z = (0.0, 0.0)          # 스폰 높이 추가분 [m]

    def __post_init__(self):
        super().__post_init__()
        # 에피소드: 유지 ≤2s + 방출 ≤8s + 자유 기립 ≥5s
        self.episode_length_s = 16.0
        # 스폰 높이 (접지 간격)
        self.events.reset_base.params["pose_range"]["z"] = tuple(self.clearance_z)
        # 보조지지 — reset 에서 스케줄 추출·첫 힘, interval(0,0) 로 매 스텝 갱신
        self.events.hang_support_reset = EventTerm(
            func=mdp_hang.hang_support_reset, mode="reset",
            params={"r0_range": tuple(self.r0_range), "hold_range": tuple(self.hold_range),
                    "release_range": tuple(self.release_range),
                    "rope_len_range": (1.0, 2.0), "bias_frac": 0.01, "point_xy": 0.03,
                    "asset_cfg": _BASE})
        self.events.hang_support_step = EventTerm(
            func=mdp_hang.hang_support_step, mode="interval", interval_range_s=(0.0, 0.0),
            params={"asset_cfg": _BASE, "sensor_cfg": _FEET_SENSOR_CFG, "threshold": 5.0})
        # 푸시는 방출 후에만 의미 — 간격 8~12s 는 대개 방출 뒤에 맞는다(그대로).
        # 보상: 양발 접촉 보상(gait_phase stand 분기, 2.0)을 지지비 게이트판으로 교체
        self.rewards.gait_phase = None
        self.rewards.feet_contact_gated = RewTerm(
            func=mdp_hang.feet_contact_gated, weight=2.0,
            params={"sensor_cfg": _FEET_SENSOR_CFG, "threshold": 5.0})
        # 지지 0 이후 생존 — 매 스텝 +1.0 (16s 에피소드에서 최대 ~700)
        self.rewards.survive_after_release = RewTerm(func=mdp_hang.survive_after_release, weight=1.0)
        # 매달림 중 발버둥 억제
        self.rewards.hang_quiet = RewTerm(func=mdp_hang.hang_quiet, weight=-0.002, params={"r_min": 0.5})


@configclass
class Biped12HangStandS2EnvCfg(Biped12HangStandS1EnvCfg):
    """S2: 작은 간격에서 접지 → 하중 이전 → 서기."""

    r0_range = (0.9, 1.0)
    hold_range = (1.0, 3.0)
    release_range = (3.0, 8.0)
    clearance_z = (0.02, 0.06)


@configclass
class Biped12HangStandS3EnvCfg(Biped12HangStandS1EnvCfg):
    """S3: 실제 매달림에 가까운 간격·타이밍."""

    r0_range = (1.0, 1.0)
    hold_range = (1.0, 4.0)
    release_range = (3.0, 10.0)
    clearance_z = (0.05, 0.15)

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0


# ─────────────────────── 축소 DR 판 (0908 저녁 2차) ───────────────────────
# S1(전체 DR) +1000it 중간 평가 51.6% (v18b 제로샷 59.4%) · 에피소드 길이 511→525 정체 = 학습 안 됨.
# V18 전체 DR 을 처음부터 물린 것이 지시("처음부터 랜덤화 범위를 과도하게 넓히지 말고 curriculum 으로 확대") 위반.
# 0906 정착 실험: 강성 0.2~0.4× 몸체는 어떤 정책으로도 못 서는 표본 → 학습 신호 오염. 축소 DR 로 S1r~S3r, 전체 DR 은 S4.
def _reduce_dr(cfg):
    ev = cfg.events
    ev.actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.1)
    ev.actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)
    if getattr(ev, "base_com", None):
        # 실물 추정: 영점자세 CoM 이 시뮬 명목보다 ≥26mm 뒤 (0908) → 후방 대역 유지, 전방·극단 제거
        ev.base_com.params["com_range"] = {"x": (-0.06, 0.0), "y": (-0.01, 0.01), "z": (0.0, 0.10)}
    if getattr(ev, "add_base_mass", None):
        _op = ev.add_base_mass.params.get("operation", "add")
        ev.add_base_mass.params["mass_distribution_params"] = ((0.95, 1.05) if _op == "scale" else (0.0, 1.0))
    if getattr(ev, "calib_bias", None):
        p = ev.calib_bias.params
        for k, half in (("joint_bias_rad", 0.026), ("imu_tilt_rad", 0.035), ("ankle_bias_rad", 0.035),
                        ("knee_bias_rad", 0.026), ("imu_roll_rad", 0.035)):
            if k in p and p[k] is not None:
                p[k] = (-half, half)
    if getattr(ev, "physics_material", None):
        ev.physics_material.params["static_friction_range"] = (0.6, 1.0)
        ev.physics_material.params["dynamic_friction_range"] = (0.5, 0.8)
    for name in ("stance_asym", "effort_scale"):
        if getattr(ev, name, None) is not None:
            setattr(ev, name, None)
    if getattr(ev, "joint_friction", None):
        ev.joint_friction.params["friction_distribution_params"] = (0.0, 0.05)
    if getattr(ev, "push_robot", None):
        ev.push_robot.params["velocity_range"] = {"x": (-0.15, 0.15), "y": (-0.15, 0.15)}


@configclass
class Biped12HangStandS1rEnvCfg(Biped12HangStandS1EnvCfg):
    """S1 + 축소 DR."""

    def __post_init__(self):
        super().__post_init__()
        _reduce_dr(self)


@configclass
class Biped12HangStandS2rEnvCfg(Biped12HangStandS2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _reduce_dr(self)


@configclass
class Biped12HangStandS3rEnvCfg(Biped12HangStandS3EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _reduce_dr(self)


# ─────────────────────── S4m: 혼합 스케줄 + 실측 대역 DR (0908 저녁 3차) ───────────────────────
# 체인2 결과: 단계별 정책은 앞 단계를 잊는다(S3r 정책이 S2r 조건 84.4→71.9%). 실물 절차는 A(70% 핸드오프)~F(실제 행잉)를
# 한 정책으로 밟아야 하므로 마지막 단계는 지지·간격·타이밍을 전 범위로 섞는다.
# DR 은 V18 극단(강성 0.2·CoM +0.06)이 아니라 실측 대역으로 확대: 실물 관절은 안 휜다(0908 k_ctrl≈kp) → 강성 0.7~1.1,
# 영점자세 CoM ≥26mm 후방 → x −0.08~0, boot_pose 잔차 ≤3.9°·IMU 롤 +2.6~3.1° → 캘리브 ±3°/IMU ±4°/롤 ±3.5°, 루프지연 4~8틱 → 3~8.
# 전체 DR(S3 cfg)은 학습 목표가 아니라 스트레스 평가로만 쓴다(S1 전체DR 정체 재현 위험·비실측 극단).
def _real_band_dr(cfg):
    ev = cfg.events
    ev.actuator_gains.params["stiffness_distribution_params"] = (0.7, 1.1)
    ev.actuator_gains.params["damping_distribution_params"] = (0.7, 1.3)
    if getattr(ev, "base_com", None):
        ev.base_com.params["com_range"] = {"x": (-0.08, 0.0), "y": (-0.015, 0.015), "z": (0.0, 0.12)}
    if getattr(ev, "add_base_mass", None):
        _op = ev.add_base_mass.params.get("operation", "add")
        ev.add_base_mass.params["mass_distribution_params"] = ((0.9, 1.1) if _op == "scale" else (0.0, 2.0))
    if getattr(ev, "calib_bias", None):
        p = ev.calib_bias.params
        for k, half in (("joint_bias_rad", 0.052), ("imu_tilt_rad", 0.070), ("ankle_bias_rad", 0.070),
                        ("knee_bias_rad", 0.052), ("imu_roll_rad", 0.061)):
            if k in p and p[k] is not None:
                p[k] = (-half, half)
    if getattr(ev, "physics_material", None):
        ev.physics_material.params["static_friction_range"] = (0.5, 1.1)
        ev.physics_material.params["dynamic_friction_range"] = (0.4, 0.9)
    if getattr(ev, "joint_friction", None):
        ev.joint_friction.params["friction_distribution_params"] = (0.0, 0.1)
    if getattr(ev, "push_robot", None):
        ev.push_robot.params["velocity_range"] = {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}
    cfg.actions.joint_pos.min_delay_steps = 3
    cfg.actions.joint_pos.max_delay_steps = 8


@configclass
class Biped12HangStandS4mEnvCfg(Biped12HangStandS1EnvCfg):
    """S4m: 혼합 스케줄(A~F 전 범위) + 실측 대역 DR."""

    r0_range = (0.5, 1.0)
    hold_range = (0.5, 4.0)
    release_range = (2.0, 10.0)
    clearance_z = (0.0, 0.15)

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        _real_band_dr(self)


@configclass
class Biped12HangStandS5sEnvCfg(Biped12HangStandS4mEnvCfg):
    """S5s: S4m + 안정 자세(좌우 대칭·발 배치 밴드·힙 편차 강화) — 0908 21:00 승윤님 요청."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pose_symmetry = RewTerm(func=mdp_hang.pose_symmetry, weight=-3.0)
        self.rewards.stance_shape = RewTerm(
            func=mdp_hang.stance_shape, weight=-5.0,
            params={"gap_range": (0.22, 0.32), "max_stagger": 0.04,
                    "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_r_joint"])})
        self.rewards.joint_deviation_hip.weight = -0.5


@configclass
class Biped12HangStandS5kEnvCfg(Biped12HangStandS5sEnvCfg):
    """S5k: S5s(대칭·발 배치) + **무릎 아주 약간(3~8°) 굽힘** + 골반 높이 유지 — 0908 22:05 승윤님 "아주 약간 무릎을 접은 안정 자세"."""

    def __post_init__(self):
        super().__post_init__()
        # 기존 밴드 8~16°(0.14~0.28rad) 는 웅크림을 만든다 → 3~8° 로 낮춤. 접촉 중인 다리만 적용(함수 내부 게이트).
        self.rewards.knee_stance_flexion.params["band_rad"] = (0.05, 0.14)
        self.rewards.knee_stance_flexion.weight = 4.0
        # 직립 유지: 골반 높이 목표 0.645m (무릎 0° 홀드 0.65 에서 약간 아래). 계단 계보에서 0 이던 가중치를 되살린다.
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.base_height_l2.params["target_height"] = 0.645


@configclass
class Biped12HangStandS5uEnvCfg(Biped12HangStandS5kEnvCfg):
    """S5u: S5k + **몸통 피치 밴드 ±5°** — 몸통·머리 수평(승윤님 0908 22:35). 다리 전체 앞기울임으로 CoM 을 맞추게 유도."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.torso_pitch_band = RewTerm(func=mdp_hang.torso_pitch_band, weight=-8.0, params={"band_rad": 0.087})


@configclass
class Biped12HangStandS5rEnvCfg(Biped12HangStandS5uEnvCfg):
    """S5r: S5u + **실물 현가 재현** — 지지 중 프레임 수평 유지 토크(두 줄 앞뒤, 오늘 착지 피치 −0.5°), 작용점 오프셋 ±1cm.
    0908 22:40 승윤님 "실물은 저 정도로 흔들리며 착지하지 않는다"."""

    def __post_init__(self):
        super().__post_init__()
        p = self.events.hang_support_reset.params
        p["level_gain_range"] = (50.0, 300.0)
        p["level_damp_range"] = (2.0, 10.0)
        p["point_xy"] = 0.01


@configclass
class Biped12HangStandS5vEnvCfg(Biped12HangStandS5rEnvCfg):
    """S5v: S5r + **힙 회전(hip_r) 편차 벌점** — S5u 가 몸통 피치 밴드를 힙 회전 ±30°(관절 한계, V10 힙롤 편법 전례)로
    우회했다. 0908 23:35."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import RewardTermCfg as _R
        import isaaclab_tasks.manager_based.locomotion.velocity.mdp as _mdp
        self.rewards.hip_r_deviation = _R(
            func=_mdp.joint_deviation_l1, weight=-3.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_r_joint"])})


@configclass
class Biped12HangStandS5wEnvCfg(Biped12HangStandS5vEnvCfg):
    """S5w: S5v + **몸통 수평 강제**(피치 밴드 ±3°, 가중 −25 = 3배) + 발 간격 어깨너비 중심(24~32cm).
    0909 00:05 승윤님 "고관절은 머리를 수평, 보폭은 어깨너비, 무릎은 약간 굽힌 안정 자세" — 기구·토크상 가능(실측 범위), 보상만 약했다."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.torso_pitch_band.weight = -25.0
        self.rewards.torso_pitch_band.params["band_rad"] = 0.052
        self.rewards.stance_shape.params["gap_range"] = (0.24, 0.32)


@configclass
class Biped12HangStandS6EnvCfg(Biped12HangStandS5wEnvCfg):
    """S6: S5w + 실물 잔여 위험 2축 DR — (1) 발목롤(AK45-36, 역구동 불가) 관절 마찰 0.1~0.6, (2) 접촉 임계 2~20N env 별
    (실물 패드 감도 제각각·캡 이탈). 0909 12:10 승윤님 "80% 까지 올릴 수 있나" — 시뮬에서 담을 수 있는 두 축."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import EventTermCfg as _E
        import isaaclab_tasks.manager_based.locomotion.velocity.mdp as _mdp
        # 전역 joint_friction(startup, 0~0.1) 뒤에 발목롤만 덮어쓴다 (startup 항은 정의 순서대로 적용)
        self.events.ankle_r_friction = _E(
            func=_mdp.randomize_joint_parameters, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_r_joint"]),
                    "friction_distribution_params": (0.1, 0.6), "operation": "abs"})
        # 접촉 임계 DR — 관측 함수의 threshold_range (기본 None 이면 종전 5N 고정)
        self.observations.policy.feet_contact.params["threshold_range"] = (2.0, 20.0)


@configclass
class Biped12HangStandS7EnvCfg(Biped12HangStandS6EnvCfg):
    """S7: S6 + **착지 후 정지** — 양발 접촉 0.5s 뒤 몸통 각속도²·수평속도²·관절속도² 벌점(−1.5), 발 들기 벌점(−2).
    0909 13:05 승윤님 "착지 후 흔들흔들은 용납되지 않는다 — 안정적인 스탠바이 자세로 서 있어야"."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.still_after_contact = RewTerm(
            func=mdp_hang.still_after_contact, weight=-1.5,
            params={"min_s": 0.5, "w_ang": 1.0, "w_lin": 1.0, "w_jvel": 0.002})
        self.rewards.feet_lift_after_contact = RewTerm(
            func=mdp_hang.feet_lift_after_contact, weight=-2.0,
            params={"min_s": 0.5, "sensor_cfg": _FEET_SENSOR_CFG, "threshold": 5.0})
        self.rewards.action_rate_l2.weight = -0.4


@configclass
class Biped12HangStandS7bEnvCfg(Biped12HangStandS7EnvCfg):
    """S7b: 정지 벌점을 **줄이 거의 풀린 뒤(r ≤ 0.3)** 로 게이트 + 가중 완화(−1.0/−1.5) — S7 이 방출 중에도 정지를 요구해
    완주율이 0.6→0.09 로 급락한 경우의 대안(하중 이전 중엔 자세 조정이 정당하다)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.still_after_contact.weight = -1.0
        self.rewards.still_after_contact.params["r_max"] = 0.3
        self.rewards.feet_lift_after_contact.weight = -1.5
        self.rewards.feet_lift_after_contact.params["r_max"] = 0.3


@configclass
class Biped12HangStandS7cEnvCfg(Biped12HangStandS6EnvCfg):
    """S7c: v4(S6) + 정지 벌점을 **줄이 거의 풀린 뒤(r ≤ 0.3)** 에만, 대신 강하게(각속도·수평속도·관절속도 −3, 발 들기 −4).
    S7(접촉 0.5s 뒤부터 전 구간)은 정지도 −25% 에 그치고 실제 행잉 93→74% 로 후퇴 — 하중 이전 중 자세 조정을 벌한 탓."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.still_after_contact = RewTerm(
            func=mdp_hang.still_after_contact, weight=-3.0,
            params={"min_s": 0.3, "w_ang": 1.0, "w_lin": 1.0, "w_jvel": 0.005, "r_max": 0.3})
        self.rewards.feet_lift_after_contact = RewTerm(
            func=mdp_hang.feet_lift_after_contact, weight=-4.0,
            params={"min_s": 0.3, "sensor_cfg": _FEET_SENSOR_CFG, "threshold": 5.0, "r_max": 0.3})
        self.rewards.action_rate_l2.weight = -0.4


@configclass
class Biped12HangStandS7dEnvCfg(Biped12HangStandS6EnvCfg):
    """S7d: v4(S6) + 정지 벌점을 **지지 0 도달 1.0s 뒤(정착 후 스탠바이)** 에만, 강하게(−4, 관절속도 가중 0.01) + 발 들기 −4.
    S7(접촉 후 전 구간)·S7c(r≤0.3)는 전이 구간을 벌해 실제 행잉 93→74/79 로 후퇴. 스탠바이 구간만 벌하면 전이는 자유."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.still_after_contact = RewTerm(
            func=mdp_hang.still_after_contact, weight=-4.0,
            params={"min_s": 0.3, "w_ang": 2.0, "w_lin": 2.0, "w_jvel": 0.01, "r_max": 0.0, "after_release_s": 1.0})
        self.rewards.feet_lift_after_contact = RewTerm(
            func=mdp_hang.feet_lift_after_contact, weight=-4.0,
            params={"min_s": 0.3, "sensor_cfg": _FEET_SENSOR_CFG, "threshold": 5.0, "r_max": 0.0, "after_release_s": 1.0})
        self.rewards.action_rate_l2.weight = -0.4


# ═══════════════ S8 (0909 저녁): raw+EMA 관측 · 발 하중 합 · settled 정숙 (승윤님 지시) ═══════════════
from isaaclab.managers import ObservationTermCfg as _ObsTerm


def _add_s8_obs(cfg, gyro_noise=0.2, jvel_noise=1.5, alpha=0.1, foot_load=False):
    """policy 그룹 **끝에** raw+EMA 채널(노이즈는 함수 안에서 넣고 EMA; cfg noise=None)[·발 하중 합] 추가. 이력 5, 평탄화.
    레이아웃: 245 + 5×(3+12) = 320 (S8a) / +5 = 325 (S8b) — symmetry.py policy_s8a/b, expand_checkpoint.py 와 동기.
    α=0.1 (τ≈10틱=200ms, 백색 노이즈 std ×0.23) — 이력 5틀(100ms) 창 밖의 정보를 주기 위해 창보다 느리게 잡는다."""
    P = cfg.observations.policy
    P.gyro_ema = _ObsTerm(func=mdp_hang.gyro_ema, params={"alpha": alpha, "noise": gyro_noise}, clip=(-20.0, 20.0),
                          history_length=5, flatten_history_dim=True)
    P.joint_vel_ema = _ObsTerm(func=mdp_hang.joint_vel_ema, params={"alpha": alpha, "noise": jvel_noise}, clip=(-60.0, 60.0),
                               history_length=5, flatten_history_dim=True)
    if foot_load:
        P.foot_load_total = _ObsTerm(func=mdp_hang.foot_load_total, params={"sensor_cfg": _FEET_SENSOR_CFG},
                                     history_length=5, flatten_history_dim=True)


def _scale_obs_noise(cfg, scale):
    """학습 관측 노이즈 대역 스케일 (자이로·중력·관절위치·관절속도). 실물 정지 std 의 10~20배인 대역을 좁힐 때."""
    P = cfg.observations.policy
    for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
        t = getattr(P, name)
        if t.noise is not None:
            t.noise.n_min = float(t.noise.n_min) * scale
            t.noise.n_max = float(t.noise.n_max) * scale


@configclass
class Biped12HangStandS8aEnvCfg(Biped12HangStandS6EnvCfg):
    """S8a: v4(S6) 환경 + raw+EMA(α0.1) 자이로/관절속도 관측 (보상·노이즈 불변). 승윤님 1순위 기대 구조."""

    def __post_init__(self):
        super().__post_init__()
        _add_s8_obs(self, alpha=0.1, foot_load=False)


@configclass
class Biped12HangStandS8aS1rEnvCfg(Biped12HangStandS1rEnvCfg):
    """S8a 정책의 실물유사(S1r) 평가 조건 — S1r 그대로 + S8a 관측."""

    def __post_init__(self):
        super().__post_init__()
        _add_s8_obs(self, alpha=0.1, foot_load=False)


@configclass
class Biped12HangStandS8aS3rEnvCfg(Biped12HangStandS3rEnvCfg):
    """S8a 정책의 실제 행잉(S3r) 평가 조건 — S3r 그대로 + S8a 관측."""

    def __post_init__(self):
        super().__post_init__()
        _add_s8_obs(self, alpha=0.1, foot_load=False)


@configclass
class Biped12HangStandS8bEnvCfg(Biped12HangStandS6EnvCfg):
    """S8b: S8a + 발 하중 합 관측(정규화, 게인·오프셋·지연 DR). 실물은 패드 합만 신뢰 가능(패드별 감도 다름) — 변형 arm."""

    def __post_init__(self):
        super().__post_init__()
        _add_s8_obs(self, alpha=0.1, foot_load=True)


@configclass
class Biped12HangStandS8nEnvCfg(Biped12HangStandS6EnvCfg):
    """S8n: v4 관측 그대로, 학습 노이즈 대역 50% (자이로 ±0.1·중력 ±0.025·관절위치 ±0.005·관절속도 ±0.75 — 실물 std 의 2~10배 유지)."""

    def __post_init__(self):
        super().__post_init__()
        _scale_obs_noise(self, 0.5)


def _add_settle_bonus(cfg, weight=1.0, use_jvel=True, use_acc=True):
    """settled-only 정숙 보너스(양, 0~1) — 최대 +weight×0.02×정착시간. S7 계열의 강한 벌점(착지 직후부터·r_max 게이트)과 달리
    지지 0·양발·기울기<5°·골반>0.6m 가 0.5s 유지된 뒤 1s 램프로만 켜진다(특권 게이트는 보상에만, 관측엔 없음)."""
    # σ 는 '현재 수준 ≈ e⁻¹ 지점' 으로 잡아 기울기를 살린다(S8a 정상상태: ω 0.10~0.12, v 0.023~0.028, q̇ 0.06, 액션 2차차분 RMS ≈ √3×0.088 ≈ 0.15).
    # s_acc 0.05 면 exp(−9)≈0 으로 포화해 학습 신호가 죽는다.
    cfg.rewards.settle_quiet_bonus = RewTerm(
        func=mdp_hang.settle_quiet_bonus, weight=weight,
        params={"use_jvel": use_jvel, "use_acc": use_acc, "sensor_cfg": _FEET_SENSOR_CFG,
                "s_ang": 0.08, "s_lin": 0.02, "s_jvel": 0.06, "s_acc": 0.15})


@configclass
class Biped12HangStandS8anEnvCfg(Biped12HangStandS8aEnvCfg):
    """S8an: raw+EMA 관측 + 학습 노이즈 50% (구조·대역 결합)."""

    def __post_init__(self):
        super().__post_init__()
        _scale_obs_noise(self, 0.5)


@configclass
class Biped12HangStandS8acEnvCfg(Biped12HangStandS8aEnvCfg):
    """S8ac: S8a + settled-only 정숙 보너스(ω·v·q̇·ä 4항, +1.0)."""

    def __post_init__(self):
        super().__post_init__()
        _add_settle_bonus(self, 1.0, True, True)


@configclass
class Biped12HangStandS8acLEnvCfg(Biped12HangStandS8aEnvCfg):
    """S8acL: S8a + 정숙 보너스 lite(ω·v 만) — 보너스 항 ablation."""

    def __post_init__(self):
        super().__post_init__()
        _add_settle_bonus(self, 1.0, False, False)


@configclass
class Biped12HangStandS8ncEnvCfg(Biped12HangStandS8nEnvCfg):
    """S8nc: S8n(노이즈 50%) + 정숙 보너스 4항."""

    def __post_init__(self):
        super().__post_init__()
        _add_settle_bonus(self, 1.0, True, True)


@configclass
class Biped12HangStandS8ancEnvCfg(Biped12HangStandS8anEnvCfg):
    """S8anc: raw+EMA + 노이즈 50% + 정숙 보너스 4항."""

    def __post_init__(self):
        super().__post_init__()
        _add_settle_bonus(self, 1.0, True, True)


@configclass
class Biped12HangStandS8cEnvCfg(Biped12HangStandS6EnvCfg):
    """S8c(대조): v4 관측·노이즈 그대로 + 정숙 보너스 4항 — 보상만의 효과."""

    def __post_init__(self):
        super().__post_init__()
        _add_settle_bonus(self, 1.0, True, True)


@configclass
class Biped12HangStandS8ac3EnvCfg(Biped12HangStandS8acEnvCfg):
    """S8ac3: S8ac + 에피소드 20→30 s + 보너스 +2.0.
    진단(0909 19:07): 학습 에피소드 평균 9 s, survive_after_release 커버리지 8%(≈1.6 s/에피소드) → 정착 게이트(0.5 s 유지+1 s 램프) 뒤 시간이
    ≈0.1 s 뿐이라 보너스가 리턴의 0.002%(게이트 완화 후에도 0.0005). 정착 후 정숙을 가르치려면 정착 구간이 학습에 존재해야 한다 —
    지지 스케줄은 그대로 두고 에피소드만 10 s 늘려 방출 후 서 있는 시간을 확보한다. 평가는 20 s 태스크(S8a/S8aS1r/S8aS3r)로 동일 조건 비교."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0
        self.rewards.settle_quiet_bonus.weight = 2.0


@configclass
class Biped12HangStandS8a3EnvCfg(Biped12HangStandS8aEnvCfg):
    """S8a3(대조): S8a + 에피소드 30 s, 보너스 없음 — S8ac4 의 '에피소드 연장' 효과 분리용."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 30.0


@configclass
class Biped12HangStandS8ac4EnvCfg(Biped12HangStandS8ac3EnvCfg):
    """S8ac4: S8ac3 + rational 형 보너스, σ = 현재 수준의 2배(ω 0.2·v 0.06·q̇ 0.12·ä 0.3).
    진단(0909 19:20): 배포 체인 없이 잰 S8a 정상상태(ω 0.123·v 0.032·q̇ 0.074·ä 0.156)는 체인 있을 때와 같음 → 학습환경에서 보너스가 0.015 로
    작았던 것은 LPF 차이가 아니라 exp 형이 DR 전 범위(CoM −0.08·푸시 3~8 s)의 운동 수준에서 포화한 탓. rational 은 0.5~4배 범위에서 기울기 유지."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.settle_quiet_bonus.params.update(
            {"rational": True, "s_ang": 0.2, "s_lin": 0.06, "s_jvel": 0.12, "s_acc": 0.3})


@configclass
class Biped12HangStandS8ac5EnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac5: S8ac4 + torso_pitch_band 가중치 −25 → −60.
    S8ac4 는 정착 피치가 −4.9° 까지 기울었다(보너스 0.028/step 이 밴드 벌점 0.017/step 을 이김 — 후방 CoM DR 에서 앞으로 기울면 CoP 가 중앙에 와 더 조용하다).
    승윤님 자세 기준(−2~−3°) 을 지키려면 밴드 벌점이 보너스보다 커야 한다: 4.9° 에서 60×0.0335×0.02 = 0.040/step > 0.028."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.torso_pitch_band.weight = -60.0


@configclass
class Biped12HangStandS8ac6EnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac6: 밴드 가중치 −40 (S8ac4 −25: 피치 −4.9°·실제행잉 94.1 / S8ac5 −60: 피치 −2.4°·실제행잉 91.0 사이의 절충 탐색)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.torso_pitch_band.weight = -40.0


@configclass
class Biped12HangStandS9aEnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S9a(v6 후보): S8ac4 + (1) 횡 CoM DR 확대 ±0.03m (실물 좌우 편심·발 롤), (2) 발목롤 마찰 좌우 비대칭 확대 0.05~0.8,
    (3) 발 하중 좌우 균형 벌점(정착 게이트). 0910 실물 A1: 착지 후 하중이 오른발로 쏠려(32/58N) 왼발 떠 넘어짐 — 관측 편향으론 재현 안 됨(물리·하중 원인).
    settle_quiet_bonus·raw+EMA 관측·정숙성(실물서 우수 확인)은 그대로 유지."""

    def __post_init__(self):
        super().__post_init__()
        ev = self.events
        if getattr(ev, "base_com", None):
            cr = dict(ev.base_com.params["com_range"]); cr["y"] = (-0.03, 0.03); ev.base_com.params["com_range"] = cr
        if getattr(ev, "ankle_r_friction", None):
            ev.ankle_r_friction.params["friction_distribution_params"] = (0.05, 0.8)
        self.rewards.foot_load_balance = RewTerm(
            func=mdp_hang.foot_load_balance, weight=-2.0,
            params={"sensor_cfg": _FEET_SENSOR_CFG})
