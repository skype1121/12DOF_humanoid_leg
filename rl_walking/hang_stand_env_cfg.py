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
import math
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
class Biped12HangStandS8ac4dEnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac4d (0911): S8ac4 + 액션 지연 DR **3~10틱**(S8ac4 는 3~8). 실측 정책루프 지연 중앙 6.3틱·p95 8.9틱
    (reports/real_prep_0911/realdata_model_id.md §5)이 S8ac4 대역 상단이라, 시뮬에서 실측 분포(6~9틱)만으로 DR 생존이
    −11~−13 pp(sim_eval_v5.md 조건 H). 물리·보상·관측 불변, 지연 대역만 변경(단일 변수). v5 웜스타트 조건부 후보.
    승격 기준·예산·종료 조건: promotion_criteria.md 부록 v5d (학습 전 고정)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.min_delay_steps = 3
        self.actions.joint_pos.max_delay_steps = 10


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


@configclass
class Biped12HangStandS9bEnvCfg(Biped12HangStandS9aEnvCfg):
    """S9b(v6b): S9a 의 하중균형 보상을 정착 게이트→양발접촉 게이트로(줄 풀리는 하중이전 구간 커버, v6a 버그 수정)·가중치 −4.
    0910: v6a 는 하중균형 보상이 정착 후에만 켜져 실제 한발빠짐 구간(줄 걸린 이전)에 신호가 없었다."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.foot_load_balance.weight = -4.0
        self.rewards.foot_load_balance.params["gate"] = "contact"


def _extend_back_com(cfg, x_back):
    ev = cfg.events
    if getattr(ev, "base_com", None):
        cr = dict(ev.base_com.params["com_range"]); cr["x"] = (x_back, 0.0); ev.base_com.params["com_range"] = cr


@configclass
class Biped12HangStandS10aEnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S10a(v6c): S8ac4 + 뒤 CoM 능동균형. (1) com_x −0.08→−0.12(실물 구간), (2) torso_pitch_band 비대칭
    (앞 12°까지 무벌·뒤 3° 엄격 — CoM 뒤일 때 앞으로 기울여 발 위로). S8ac4 정숙·raw+EMA·settle 유지.
    근거(0910): 홀드 0/256(정적불안정=능동균형 필수)·v5 com sweep −0.04→−0.10 = 91.8→43.4% 급락·실물 CoM ≥26mm 뒤."""

    def __post_init__(self):
        super().__post_init__()
        _extend_back_com(self, -0.10)                                  # v6c2: −0.12 는 희석 → −0.10(실물 구간 커버)
        self.rewards.torso_pitch_band.params["fwd_band_rad"] = 0.21   # 앞(음수) 12°까지 무벌 (부호 수정판)
        # 뒤 밴드는 기존 band_rad(S5w=0.052=3°) 유지. 앞으로 기울여 서는 해법 허용.


@configclass
class Biped12HangStandCableS3rEnvCfg(Biped12HangStandS8aS3rEnvCfg):
    """케이블 지지 모델(0911) — S8aS3r(실제 행잉·raw+EMA 320관측) 에 길이제약 케이블 적용. v5(320) 재현 게이트 전용(학습 아님).
    이 환경에서 v5 가 실물 궤적(방출 중 발력 저하 + p1축 회전 + 정책 준정지)을 재현하는지 확인 후에만 케이블 기반 학습."""

    def __post_init__(self):
        super().__post_init__()
        self.events.hang_support_reset.params["cable"] = True
        # 앵커 앞뒤 오프셋 범위를 넓혀(±0.08) 실물 후크 기하 불확실성 포함
        self.events.hang_support_reset.params["anchor_x_range"] = (-0.08, 0.08)


@configclass
class Biped12HangStandCableFS3rEnvCfg(Biped12HangStandCableS3rEnvCfg):
    """힘제어 케이블(0912): 장력 = r(t)·m·g, 방향 = 후크→고정 앵커. 로프 강성이 미측정이라 길이 모델(k 지배) 대신
    줄 모멘트(후크 비연직, 가설 H2b) 검정용. 학습 아님."""

    def __post_init__(self):
        super().__post_init__()
        self.events.hang_support_reset.params["cable_force_mode"] = True


@configclass
class Biped12HangStandS8ac4qEnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac4q(0912): S8ac4 + 준정적 롤 토크 램프 DR — 실물 0912 5회에서 확인된 '부분지지 아래 각속도 1~5 °/s 로 천천히 기울어 접촉을
    잃는' 이탈(시뮬 재현: r0 0.5, 0.5 Nm/s 램프 → 22 s 접촉 OFF·|ω| 4~5 °/s)을 학습 분포에 넣어 기울기 각에 대한 보정을 요구한다.
    물리·보상·관측 불변(단일 변수). 수평 유지 토크는 종전처럼 OFF(0912 확인: 지금까지 미적용이었음)."""

    def __post_init__(self):
        super().__post_init__()
        p = self.events.hang_support_reset.params
        p["level_gain_range"] = (0.0, 0.0); p["level_damp_range"] = (0.0, 0.0)
        p["ramp_rate_range"] = (0.05, 0.6); p["ramp_cap_range"] = (3.0, 15.0); p["ramp_start_range"] = (1.0, 8.0); p["ramp_sign_random"] = True

def _apply_real_bands(cfg):
    """0913 실측 대역(S8ac4r): 0912 실물 5회에서 '정책이 느린 기울기(≤1 °/s·≤5°)에 반응하지 않음'(이득 ≈0.1 °/°)의 원인 가설 =
    학습 관측 잡음·편향 대역이 실측의 20~300배라 그 신호가 학습 분포에서 잡음과 구분되지 않았고, 시뮬은 강체 kp150 이라 수동 강성이
    정책의 무반응을 가려 왔다(실물 구조 강성 43.5~107 Nm/rad → k_eff ≈ 34~62 = kp 의 0.23~0.41 배).
    변경(전부 실측→모델식별, 보상·관측 계약·액션 체인 불변):
      - 관측 잡음: 자이로 ±0.2→±0.05 rad/s(실측 std 0.006~0.010), 중력벡터 ±0.05→±0.02(≈1.1°), 관절속도 ±1.5→±0.3(실측 std 0.005),
        관절위치 ±0.01 유지; S8a EMA 항의 잡음 파라미터도 동일 축소.
      - IMU 장착 편향 DR: 롤 ±3.5°→±1.5°, 피치 ±4°→±1.5° — 실측 편향(롤 +1.3°, 피치 −0.6°)을 브리지에서 상수 보정하는 것을 전제(잔여 ±0.5°).
      - 액추에이터 강성 DR: (0.7,1.1)→(0.3,1.1) — 실물 k_eff/kp 0.23~0.41 의 대부분을 포함(0.23~0.30 은 분포 밖). 감쇠 (0.7,1.3) 유지. 관절속도 대역(±0.3)은 실측 std(0.005)의 ~60배로 다른 채널보다 넓다.
    """
    P = cfg.observations.policy
    for name, half in (("base_ang_vel", 0.05), ("projected_gravity", 0.02), ("joint_vel", 0.3)):
        t = getattr(P, name, None)
        if t is not None and t.noise is not None:
            t.noise.n_min = -half; t.noise.n_max = half
    for name, val in (("gyro_ema", 0.05), ("joint_vel_ema", 0.3)):
        t = getattr(P, name, None)
        if t is not None and isinstance(t.params, dict) and "noise" in t.params:
            t.params["noise"] = val
    ev = cfg.events
    if getattr(ev, "calib_bias", None):
        p = ev.calib_bias.params
        for k, half in (("imu_tilt_rad", 0.026), ("imu_roll_rad", 0.026)):
            if k in p and p[k] is not None:
                p[k] = (-half, half)
    if getattr(ev, "actuator_gains", None):
        ev.actuator_gains.params["stiffness_distribution_params"] = (0.3, 1.1)


@configclass
class Biped12HangStandS8ac4rEnvCfg(Biped12HangStandS8ac4qEnvCfg):
    """S8ac4r(0913): S8ac4q(준정적 롤 램프 DR) + 실측 대역(_apply_real_bands). 램프 cap 은 2~12 Nm(50 % 지지의 접촉 한계 ≈17 Nm 안, 강성 0.3 배에서도 학습 가능)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_real_bands(self)
        self.events.hang_support_reset.params["ramp_cap_range"] = (2.0, 12.0)


@configclass
class Biped12HangStandS8ac4rdEnvCfg(Biped12HangStandS8ac4rEnvCfg):
    """S8ac4rd(0913 12시, 승윤님 지시 '느린 기울기도 감지하고 서 있어야'): S8ac4r(램프 DR + 실측 잡음/IMU 편향 대역 + 강성 0.3) 에
    액션 지연 DR 을 실측 정책루프 지연(중앙 6.3틱, p95 8.9)에 맞춰 4~11 틱으로 확장. v6 의 강점(램프 15 Nm 홀드 자립 61~63/64·정적 모멘트·연성)은
    유지하고 약점(지연 6~10틱 −11~−30 pp)을 학습 분포로 메우는 단일 변수 실험. v6(2026-09-13_00-40-49 model_141486) 웜스타트 1500 it.
    수평 유지 토크는 S8ac4q 에서 명시 OFF(상속)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.min_delay_steps = 4
        self.actions.joint_pos.max_delay_steps = 11


@configclass
class Biped12HangStandS8ac4rdpEnvCfg(Biped12HangStandS8ac4rdEnvCfg):
    """S8ac4rdp(0913 13시): S8ac4rd(v6 + 지연 4~11틱) + 기계 유격 DR(내재형 DeadzoneImplicit, 피치 0~3°·횡 0~1.5° env 별).
    v6d 가 지연 6~9/8~10틱을 82/70 % 로 메웠지만 유격 P1(2°/1°, 느린 방출)이 92 → 48 % 로 빠짐 → 유격을 학습 분포에 넣어 둘 다 잡는 시도.
    v6d(2026-09-13_12-33-22_v6d model_142985) 웜스타트 1500 it. 단일 변수 = 유격 DR."""

    def __post_init__(self):
        super().__post_init__()
        _apply_joint_play(self, pitch_deg=3.0, lat_deg=1.5, rng=(0.0, 1.0))


@configclass
class Biped12HangStandS8ac4rdAEnvCfg(Biped12HangStandS8ac4rdEnvCfg):
    """S8ac4rdA(0913 20시, 승윤님 지시 '찔끔찔끔 말고 더 적극적으로 균형'): S8ac4rd(v6 + 실측 지연 4~11틱)에서 학습 목표를
    정숙 → 생존 쪽으로 옮긴다. action_rate 벌점 절반(−0.2→−0.1), settle_quiet_bonus 절반(2.0→1.0), 낙상 비용 3배(termination −10→−30),
    방출 후 생존 보상 2배(1→2), 횡 CoM DR y ±5 cm(실물 0913 착지 후 오른쪽 롤의 기구 편심 후보 폭). 실물 5회(0913 19~20시)에서
    v5·v5+보정기·v6 전부 착지/방출 직후 오른쪽으로 넘어간 뒤의 후보 — 원인 재현 스윕(sig_sweep)과 병행. v6d 웜스타트."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.settle_quiet_bonus.weight = 1.0
        self.rewards.termination_penalty.weight = -30.0
        self.rewards.survive_after_release.weight = 2.0
        cr = self.events.base_com.params["com_range"]
        cr["y"] = (-0.05, 0.05)


def _level_off(cfg):
    """0913 감사: S8ac4 직계 후보는 hang_support_reset 의 수평 유지 토크(lvl_k 50~300)가 0912 수정(set+add) 이후 학습에 **실제로 걸린다**.
    v5(학습 당시 미적용)·v6(S8ac4q 는 명시 OFF)과 조건을 맞추기 위해 끈다. v8·v9(0913 03~05시 학습)는 이 토크가 켜진 채 학습됨 → 결과에 교란 표기."""
    p = cfg.events.hang_support_reset.params
    p["level_gain_range"] = (0.0, 0.0); p["level_damp_range"] = (0.0, 0.0)


@configclass
class Biped12HangStandS8ac4rnEnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac4rn(0913, 대조): S8ac4 + 실측 대역만(램프 DR 없음) — 능동 보정이 대역 정합만으로 생기는지 램프 DR 이 필요한지 분리."""

    def __post_init__(self):
        super().__post_init__()
        _level_off(self)
        _apply_real_bands(self)

def _apply_joint_play(cfg, pitch_deg=3.0, lat_deg=1.5, rng=(0.0, 1.0)):
    """0913 기계 유격 DR: 액추에이터를 DeadzonePD(명시적)로 교체. 유격 반폭 = 기준(피치 관절 pitch_deg, 횡 관절 lat_deg) × U(rng) env 별.
    실측(0912 재배치): 피치 관절 1.5~3.7°, 횡 ≤0.9° → 기준 피치 3.0°(실측 상한보다 약간 작음)·횡 1.5°(실측보다 큼, 여유), 0 까지 포함(종전 강체와 연속). 그룹(ak70/ankle_roll)별로 스케일이 독립 추출된다."""
    import math as _m
    from rl_walking.actuators_play import DeadzoneImplicitActuatorCfg
    acts = cfg.scene.robot.actuators
    for k, a in list(acts.items()):
        exprs = a.joint_names_expr if isinstance(a.joint_names_expr, (list, tuple)) else [a.joint_names_expr]
        import re as _re
        pm = {}
        for jt in ("hip_f", "hip_a", "hip_r", "knee", "ankle_f", "ankle_r"):
            if any(_re.fullmatch(ex, f"left_{jt}_joint") for ex in exprs):
                pm[f".*_{jt}_joint"] = _m.radians(lat_deg if jt in ("hip_a", "ankle_r") else pitch_deg)
        acts[k] = DeadzoneImplicitActuatorCfg(joint_names_expr=a.joint_names_expr,
                                              effort_limit_sim=getattr(a, "effort_limit_sim", None), velocity_limit_sim=getattr(a, "velocity_limit_sim", None),
                                              stiffness=a.stiffness, damping=a.damping, armature=getattr(a, "armature", None), friction=getattr(a, "friction", None),
                                              play_rad=pm, play_rad_range=tuple(rng))


@configclass
class Biped12HangStandS8ac4rpEnvCfg(Biped12HangStandS8ac4rEnvCfg):
    """S8ac4rp(0913): S8ac4r + 기계 유격 DR(피치 0~3°, 횡 0~1.5°, env 별) — 0913 시뮬에서 유격 1.5° 만으로 v5 가 방출 후 6 % 생존(스모크)한 것에 대응."""

    def __post_init__(self):
        super().__post_init__()
        _apply_joint_play(self, pitch_deg=3.0, lat_deg=1.5, rng=(0.0, 1.0))

def _apply_real_noise_only(cfg):
    """실측 잡음 대역만(자이로 ±0.05·중력 ±0.02·관절속도 ±0.3, EMA 항 동일) — IMU 편향 DR·강성 DR 은 종전(S8ac4) 유지."""
    P = cfg.observations.policy
    for name, half in (("base_ang_vel", 0.05), ("projected_gravity", 0.02), ("joint_vel", 0.3)):
        t = getattr(P, name, None)
        if t is not None and t.noise is not None:
            t.noise.n_min = -half; t.noise.n_max = half
    for name, val in (("gyro_ema", 0.05), ("joint_vel_ema", 0.3)):
        t = getattr(P, name, None)
        if t is not None and isinstance(t.params, dict) and "noise" in t.params:
            t.params["noise"] = val


@configclass
class Biped12HangStandS8ac4rqEnvCfg(Biped12HangStandS8ac4qEnvCfg):
    """S8ac4rq(0913, 대안 후보): S8ac4q(램프 DR, cap 2~12) + 실측 잡음 대역만. 강성 DR(0.7~1.1)·IMU 편향 DR(롤 ±3.5°·피치 ±4°) 종전 유지 —
    v6(S8ac4r) 가 A1(50 % 지지 램프, 강성 ×0.3) 에서 v5 보다 약하게 나온 경우의 분리 실험(강성 DR 확장이 원인인지)."""

    def __post_init__(self):
        super().__post_init__()
        _apply_real_noise_only(self)
        self.events.hang_support_reset.params["ramp_cap_range"] = (2.0, 12.0)


@configclass
class Biped12HangStandS8ac4r5EnvCfg(Biped12HangStandS8ac4rEnvCfg):
    """S8ac4r5(0913, 대안 후보): S8ac4r 와 같되 강성 DR 하한 0.5 (0.3 → 0.5)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.5, 1.1)

@configclass
class Biped12HangStandS8ac4sEnvCfg(Biped12HangStandS8ac4qEnvCfg):
    """S8ac4s(0913, 대안 후보): S8ac4q(램프 DR, cap 2~12) + 강성 DR (0.3,1.1) 만 — 관측 잡음·IMU 편향 대역은 종전(S8ac4) 유지.
    v6(S8ac4r) 의 강점(정적 모멘트·연성 관절)이 램프+강성 DR 에서 오고, 약점(지연·잡음·접촉 비트 민감)이 잡음 대역 축소에서 온다는 가설의 분리용."""

    def __post_init__(self):
        super().__post_init__()
        self.events.hang_support_reset.params["ramp_cap_range"] = (2.0, 12.0)
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.3, 1.1)

@configclass
class Biped12HangStandS8ac4kEnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac4k(0913, 최종 후보 v8): S8ac4 + 액추에이터 강성 DR (0.7,1.1)→(0.3,1.1) **만**. 램프 DR 없음(v5q 프로브: 램프 DR 단독은
    C1 −8.6·C2 −11 pp 손실에 R6/B1 이득 없음), 잡음·IMU 편향 대역 종전. 단일 변수 = 강성 DR(실물 k_eff/kp 0.23~0.41 실측 포함)."""

    def __post_init__(self):
        super().__post_init__()
        _level_off(self)
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.3, 1.1)

def _apply_gravity_bands(cfg):
    """IMU 채널만 실측 대역(0913 §3.1: 느린 롤은 IMU 에만 보였다 → 중력벡터를 더 믿게): 중력벡터 잡음 ±0.05→±0.02, IMU 롤/피치 편향 DR ±3.5/4°→±1.5°.
    자이로·관절속도 대역은 종전 유지(v6 의 지연 민감 상승이 그쪽 대역 축소의 대가라는 가설). 브리지 imu_mount_bias_deg 보정 전제."""
    P = cfg.observations.policy
    t = getattr(P, "projected_gravity", None)
    if t is not None and t.noise is not None:
        t.noise.n_min = -0.02; t.noise.n_max = 0.02
    ev = cfg.events
    if getattr(ev, "calib_bias", None):
        p = ev.calib_bias.params
        for k in ("imu_tilt_rad", "imu_roll_rad"):
            if k in p and p[k] is not None:
                p[k] = (-0.026, 0.026)


@configclass
class Biped12HangStandS8ac4gEnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac4g(0913 후보): S8ac4 + IMU 채널 실측 대역만(중력 잡음·IMU 편향 DR). 단일 변수 = IMU 채널."""

    def __post_init__(self):
        super().__post_init__()
        _level_off(self)
        _apply_gravity_bands(self)


@configclass
class Biped12HangStandS8ac4kgEnvCfg(Biped12HangStandS8ac4kEnvCfg):
    """S8ac4kg(0913 후보): S8ac4k(강성 DR 0.3) + IMU 채널 실측 대역. v8 결과에 따라 선택."""

    def __post_init__(self):
        super().__post_init__()
        _level_off(self)
        _apply_gravity_bands(self)

@configclass
class Biped12HangStandS8ac4k5EnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac4k5(0913 마지막 후보 v10): S8ac4 + 강성 DR (0.7,1.1)→(0.5,1.1). v8(0.3 하한)은 정적 모멘트 100 % 를 얻었지만 스텝 폭주로 C1 54.7 —
    하한을 0.5 로 완화해 이득 일부(R6 ≥ 60 %)와 분포 내 성공률(C1 ≥ 92) 양립을 시험. 단일 변수 = 강성 DR 하한."""

    def __post_init__(self):
        super().__post_init__()
        _level_off(self)
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.5, 1.1)

@configclass
class Biped12HangStandS8ac4k4EnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac4k4(0913 자동 후속): S8ac4 + 강성 DR (0.4,1.1)."""

    def __post_init__(self):
        super().__post_init__()
        _level_off(self)
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.4, 1.1)


@configclass
class Biped12HangStandS8ac4zEnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac4z(0913 08시, 대조군 v5y): S8ac4 + 수평 유지 토크 OFF **만**. v5x(S8ac4 그대로 웜스타트 1500 it)가 C1 85.2·C2 74.6·both0 61.7 로
    v10(86.7/74.2/56.6)과 같았음 → 변형들의 C1/C2 손실은 '이어 학습' 자체의 산물. 단, v5x 는 수평 토크 ON 으로 학습(평가 기본 OFF 와 불일치)이라
    그 교란까지 뺀 대조군. 체크포인트 3개(마지막·−100·−200) C1/R6 로 체크포인트 간 변동도 같이 잰다."""

    def __post_init__(self):
        super().__post_init__()
        _level_off(self)


@configclass
class Biped12HangStandS8ac4k6EnvCfg(Biped12HangStandS8ac4EnvCfg):
    """S8ac4k6(0913 자동 후속): S8ac4 + 강성 DR (0.6,1.1)."""

    def __post_init__(self):
        super().__post_init__()
        _level_off(self)
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.6, 1.1)


# ─────────────────────── 0918 S9c: 크라우치 기본자세 + 물리 편향 DR ───────────────────────
#: 손 튜닝 수평 기준 자세 계열(0917: v3 힙 ±15·무릎 ∓30·발목앞뒤 ±20.5; v4 힙 ±26). 시뮬 URDF 는 힙 9.5° 에서 몸통 수평,
#: 실물은 15~26° 에서 수평 → nominal 힙 15°(중간), 물리 편향 DR 힙 ±12° 로 3~27° 를 덮는다.
CROUCH_DEFAULT_RAD = {
    "left_hip_f_joint": 0.2618, "right_hip_f_joint": -0.2618,      # ±15°
    "left_hip_a_joint": 0.0, "right_hip_a_joint": 0.0,
    "left_hip_r_joint": 0.0, "right_hip_r_joint": 0.0,
    "left_knee_joint": -0.5236, "right_knee_joint": 0.5236,        # ∓30°
    "left_ankle_f_joint": 0.3578, "right_ankle_f_joint": -0.3578,  # ±20.5°
    "left_ankle_r_joint": 0.0, "right_ankle_r_joint": 0.0,
}


@configclass
class Biped12HangStandS9cEnvCfg(Biped12HangStandS8ac4rEnvCfg):
    """S9c (0918 밤, 승윤님 방향): **크라우치 기본자세(v3) + 물리 기본자세 편향 DR + 착지 기울기 + 지면 연성**.
    배경: 정책 기본자세(무릎 6°·발목 1.4°)는 URDF 에서도 몸통이 4.6° 뒤로 누움 → 실물(−17.5° 홀드)에서 매번 뒤로 낙상.
    손 튜닝 v3/v4 크라우치는 실물 무보조 자립(0914~17). 같은 관절각에서 시뮬 몸통 +5° vs 실물 −1~+9°(착지마다 다름) →
    (1) 기본자세를 크라우치로, (2) 실물 영점/사슬 오프셋을 env 별 물리 기본자세 편향(힙 ±12°·무릎 ±5°·발목 ±8°)으로,
    (3) 스폰 피치 ±8°, (4) 지면 연성 k 2e3(0914 실물 패드·구조 연성 에뮬레이션에서 v5 87%·v6 100% 생존 조건).
    배포 계약: 이 정책의 DEFAULT_POSE = CROUCH_DEFAULT_RAD (러너·브리지 체크포인트별 기본자세)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = dict(CROUCH_DEFAULT_RAD)
        if getattr(self.rewards, "knee_stance_flexion", None) is not None:
            self.rewards.knee_stance_flexion.params["band_rad"] = (0.40, 0.65)     # 23~37° (크라우치)
        if getattr(self.rewards, "base_height_l2", None) is not None:
            self.rewards.base_height_l2.params["target_height"] = 0.58
        self.events.default_pose_bias = EventTerm(
            func=mdp_hang.resample_default_pose_bias, mode="reset",
            params={"hip_f_rad": (-0.21, 0.21), "knee_rad": (-0.087, 0.087), "ankle_f_rad": (-0.14, 0.14),
                    "reset_noise_rad": 0.05, "action_term": "joint_pos", "asset_cfg": _BASE})
        pr = self.events.reset_base.params["pose_range"]
        pr["pitch"] = (-0.14, 0.14)
        self.scene.terrain.physics_material.compliant_contact_stiffness = 2000.0
        self.scene.terrain.physics_material.compliant_contact_damping = 40.0


@configclass
class Biped12HangStandS9cHoldEnvCfg(Biped12HangStandS9cEnvCfg):
    """S9c 대조군 설정(평가용, 학습 안 함): 같은 조건에서 --hold(고정 자세 kp150)가 서는지."""


# ─────────────────────── 0918 저녁 S9d: 기본자세 = 실물 v4 명령 + 식별된 무릎 직렬 처짐 편향 ───────────────────────
#: 손 튜닝 수평 기준 v4(0917, 실물 무보조 자립·밀기 검증): 힙 ±26·무릎 ∓30·발목앞뒤 ±20.5 (명령값).
CROUCH_V4_DEFAULT_RAD = {
    "left_hip_f_joint": 0.4538, "right_hip_f_joint": -0.4538,      # ±26°
    "left_hip_a_joint": 0.0, "right_hip_a_joint": 0.0,
    "left_hip_r_joint": 0.0, "right_hip_r_joint": 0.0,
    "left_knee_joint": -0.5236, "right_knee_joint": 0.5236,        # ∓30°
    "left_ankle_f_joint": 0.3578, "right_ankle_f_joint": -0.3578,  # ±20.5°
    "left_ankle_r_joint": 0.0, "right_ankle_r_joint": 0.0,
}


@configclass
class Biped12HangStandS9dEnvCfg(Biped12HangStandS9cEnvCfg):
    """S9d (0918 저녁): **기본자세 = 실물 v4 명령 + 엔코더 뒤 무릎 직렬 처짐을 공유 물리 편향으로**.
    배경(0918 실기): v1c 정책은 실물에서 힙을 ~7° 로 펴(시뮬 정상상태 힙 9°) 13° 후방 스윙·발목 포화 발진 → 리프트가드.
    실물 v4 서기(엔코더 29.9/−34.9/19.1°, IMU −0.3°)는 URDF 강체로는 몸통 +14°. 균일 연성(k≈65)은 CoM 을 뒤꿈치 밖(−8.5 cm)으로
    보내 서기 불가 → 기각. **무릎 단독 직렬 연성 k≈40 Nm/rad**: 피치 +0.3°·CoM 발목축 뒤 5 cm(뒤 여유 1.5 cm = '뒤로 밀면 약함'
    실측과 부합, 0913 실측 k_struct 43.5~107 대역 안). 1점 식별 가설 — 시뮬 등가 = 무릎 공유 물리 편향 −8~−17°(중심 −13°).
    독립 편향(영점 오차)은 힙 ±8°·무릎 ±3°·발목 ±6° 로 축소. 무릎 밴드·골반고 목표는 편향 자세(FK: base z 0.563 m)에 맞춤.
    배포 계약: DEFAULT_POSE = CROUCH_V4_DEFAULT_RAD (실물 명령 프레임 = v4 그대로)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.init_state.joint_pos = dict(CROUCH_V4_DEFAULT_RAD)
        self.events.default_pose_bias.params.update({
            "hip_f_rad": (-0.14, 0.14), "knee_rad": (-0.052, 0.052), "ankle_f_rad": (-0.105, 0.105),
            "shared_knee_rad": (-0.30, -0.14),      # 왼무릎 기준 −17~−8° (오른쪽 부호 반전)
        })
        if getattr(self.rewards, "knee_stance_flexion", None) is not None:
            self.rewards.knee_stance_flexion.params["band_rad"] = (0.45, 0.95)     # 26~54° (물리 무릎 = 30+8~17 + PD 처짐)
        if getattr(self.rewards, "base_height_l2", None) is not None:
            self.rewards.base_height_l2.params["target_height"] = 0.565


@configclass
class Biped12HangStandS9dHoldEnvCfg(Biped12HangStandS9dEnvCfg):
    """S9d 대조군(평가용): 같은 플랜트에서 --hold(v4 명령 kp150 고정)가 서는지 = 식별 플랜트의 정합 검사."""


# ─────────────────────── 0919 밤 G1: 지면 출발(스탠바이 v5) 탐색용 세대 ───────────────────────
#: 승윤님 확정 절차(0918): 캘리브 → v5 기준자세 → 착지 → 자동수평 → 정책 인수. 정책은 공중·착지를 배우지 않는다.
#: 0919 실측(폰 링크각, 왼·오른): 무릎 = 하중 탄성 처짐 +10~12.5°(대역 무관), 발목 = 전동부 유격·마찰 대역 ~8°(뒤끝 −8 ↔ 앞끝 0, 엔코더 1.5°만 변함),
#: 힙 −1~−4. 시뮬 후보 시험(hold_study 0919): 무릎 상수 편향 −11 + 발목 kp×0.25 + 발목 관절마찰 2.0 → 낙상 없이 env 별 피치 분산 유지.
#: ⚠ 이것은 '히스테리시스 모델'이 아니라 강성·마찰을 바꾼 후보 시험이다. 모터 kp 를 낮추는 것과 엔코더 뒤 구조가 휘는 것(직렬 요소)은 다르다(코덱스 지적).
#: ⚠ 탐색용: 이 플랜트는 '실측 각도 차이를 넣은 가설 모델'이며 관절별로 다르게 버티는 특성을 재현하지 않는다(승윤님·코덱스 조건). 등록·배포·기본 정책 변경 금지.
CROUCH_V5_DEFAULT_RAD = {
    "left_hip_f_joint": 0.5282, "right_hip_f_joint": -0.5282,      # ±30.26°
    "left_hip_a_joint": 0.0, "right_hip_a_joint": 0.0,
    "left_hip_r_joint": 0.0, "right_hip_r_joint": 0.0,
    "left_knee_joint": -0.6720, "right_knee_joint": 0.6720,        # ∓38.5°
    "left_ankle_f_joint": 0.4887, "right_ankle_f_joint": -0.4887,  # ±28.0°
    "left_ankle_r_joint": 0.0, "right_ankle_r_joint": 0.0,
}


@configclass
class Biped12HangStandG1EnvCfg(Biped12HangStandS9cEnvCfg):
    """G1 (0919 밤, 탐색용): **지면 출발** — 스탠바이 v5(힙 ±30.26·무릎 ∓38.5·발목 ±28) 명령 자세로 지면 스폰, 줄 지지 0.
    플랜트(가설 모델·후보): 무릎 공유 편향 −8~−14°(하중 탄성 처짐 실측 +10~12.5), 발목 공유 편향 −5~0, 발목 kp×0.5~1.0 + 발목 관절마찰 0.5~2(강성·마찰 후보, 직렬 요소 아님; 0.2~0.5·1~3 은 명목 홀드 100 % 낙상이라 완화),
    힙 독립 ±4°, 무릎·발목 독립 ±2°. 지면 강체(S9c 연성 k2000 은 홀드를 악화시켜 제거). 자세 편차 벌점(정책 프레임 힙·무릎·발목)으로 램프 폭주 억제.
    판정(사전 문서화): 실측 근사 플랜트(무릎 −11·발목 −4·kp×0.25·마찰 2)에서 같은 초기조건의 고정 v5 홀드와 비교 — 생존·흔들림·자세 이탈·관절 한계 클램프·목 루프 안정.
    배포 계약: DEFAULT_POSE = CROUCH_V5_DEFAULT_RAD (관측 320 raw+EMA 불변). 이 결과로 모델 정합·실물 성능을 확정하지 않는다."""

    r0_range = (0.0, 0.0)          # 지지 없음
    hold_range = (0.1, 0.1)
    release_range = (0.1, 0.2)
    clearance_z = (0.0, 0.02)

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import EventTermCfg as _E
        import isaaclab_tasks.manager_based.locomotion.velocity.mdp as _mdp
        self.scene.robot.init_state.joint_pos = dict(CROUCH_V5_DEFAULT_RAD)
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.62)             # 편향 자세(무릎 ~49°) 발바닥 ≈ 지면 (hold_study: base−발 ≈0.56 + 발 원점 0.058)
        self.episode_length_s = 20.0
        # 지지 스케줄: 즉시 0 (hang_support 항은 유지 — 보상 게이트가 지지비 상태를 읽음)
        self.events.hang_support_reset.params.update({"r0_range": (0.0, 0.0), "hold_range": (0.1, 0.1), "release_range": (0.1, 0.2)})
        self.events.reset_base.params["pose_range"]["z"] = (0.0, 0.01)
        self.events.reset_base.params["pose_range"]["pitch"] = (-0.05, 0.05)   # 착지 편차 ±3°
        # 플랜트 편향(가설 모델)
        self.events.default_pose_bias.params.update({
            "hip_f_rad": (-0.07, 0.07), "knee_rad": (-0.035, 0.035), "ankle_f_rad": (-0.035, 0.035),
            "shared_knee_rad": (-0.244, -0.14),     # −14 ~ −8°
            "shared_ankle_f_rad": (-0.087, 0.0),    # −5 ~ 0° (대역 중심 −4 를 포함하되 강체 발목과도 겹치게)
        })
        # 발목앞뒤 저강성 + 관절마찰 (전동부 유격·마찰 대역 근사) — 전역 actuator_gains/joint_friction(startup) 뒤에 덮어씀
        _ank = SceneEntityCfg("robot", joint_names=[".*_ankle_f_joint"])
        self.events.ankle_f_gains = _E(func=_mdp.randomize_actuator_gains, mode="startup",
                                       params={"asset_cfg": _ank, "stiffness_distribution_params": (0.5, 1.0), "operation": "scale"})   # 0.2~0.5 는 G1Hold 명목 100 % 낙상 → 완화
        self.events.ankle_f_friction = _E(func=_mdp.randomize_joint_parameters, mode="startup",
                                          params={"asset_cfg": _ank, "friction_distribution_params": (0.5, 2.0), "operation": "abs"})
        # 지면: 강체 (S9c 의 연성 접촉 제거)
        self.scene.terrain.physics_material.compliant_contact_stiffness = 0.0
        self.scene.terrain.physics_material.compliant_contact_damping = 0.0
        # 보상: 무릎 밴드·골반고를 v5+처짐 물리 자세에 맞춤, 정책 프레임 자세 편차 벌점
        if getattr(self.rewards, "knee_stance_flexion", None) is not None:
            self.rewards.knee_stance_flexion.params["band_rad"] = (0.60, 1.05)   # 34~60° (물리 무릎 ≈ 38.5+11+처짐)
        if getattr(self.rewards, "base_height_l2", None) is not None:
            self.rewards.base_height_l2.params["target_height"] = 0.55
        self.rewards.leg_pose_deviation = RewTerm(
            func=_mdp.joint_deviation_l1, weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_f_joint", ".*_knee_joint", ".*_ankle_f_joint"])})


@configclass
class Biped12HangStandG1HoldEnvCfg(Biped12HangStandG1EnvCfg):
    """G1 대조군(평가용): 같은 플랜트·초기조건에서 --hold(v5 명령 kp150 고정)."""


@configclass
class Biped12HangStandG2EnvCfg(Biped12HangStandG1EnvCfg):
    """G2 (0920 새벽, 탐색용): G1 + **짧은 지지 커리큘럼**. G1(지지 0, v1c 웜스타트)은 6000회에도 보상 8~31·에피소드 1~3.6 s 로 서기를 못 배움
    (명목 홀드 100 % vs 정책 0 %). S9c 는 매달림 지지 75 % → 방출 커리큘럼 덕에 학습됐음. G2 는 접지 상태에서 지지 20~50 % 를 0.5~1 s 유지 뒤 1~3 s 안에 방출 —
    실물 절차(착지 완료 뒤 인수)와 같은 '지면 위 출발'이면서 초기 탐색을 돕는다. 정책은 방출 뒤 ≥15 s 자립을 배운다. 플랜트는 G1 과 같다(가설 모델)."""

    r0_range = (0.2, 0.5)
    hold_range = (0.5, 1.0)
    release_range = (1.0, 3.0)

    def __post_init__(self):
        super().__post_init__()
        self.events.hang_support_reset.params.update({"r0_range": (0.2, 0.5), "hold_range": (0.5, 1.0), "release_range": (1.0, 3.0)})


@configclass
class Biped12HangStandG2HoldEnvCfg(Biped12HangStandG2EnvCfg):
    """G2 대조군(평가용)."""


def _swap_ankle_f_deadzone_coulomb(cfg, play_deg=4.0, play_range=(0.75, 1.25), coulomb_nm=2.0, coulomb_range=(0.75, 1.5)):
    """발목앞뒤만 DeadzoneCoulombImplicit(유격+쿨롱 마찰)로 교체 — 0919 실측 발목 대역 ~8°(반폭 4)·hold_study 0920 후보(쿨롱 2 Nm) 기준."""
    import re as _re
    from isaaclab.actuators import ImplicitActuatorCfg as _Impl
    from rl_walking.actuators_play import DeadzoneCoulombImplicitActuatorCfg
    acts = cfg.scene.robot.actuators
    base = acts["ak70"]
    exprs = base.joint_names_expr if isinstance(base.joint_names_expr, (list, tuple)) else [base.joint_names_expr]
    keep = [ex for ex in exprs if not _re.fullmatch(ex, "left_ankle_f_joint")]
    acts["ak70"] = _Impl(joint_names_expr=keep, effort_limit_sim=base.effort_limit_sim, velocity_limit_sim=base.velocity_limit_sim,
                         stiffness=base.stiffness, damping=base.damping, armature=base.armature)
    acts["ankle_f_play"] = DeadzoneCoulombImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_f_joint"], effort_limit_sim=base.effort_limit_sim, velocity_limit_sim=base.velocity_limit_sim,
        stiffness=base.stiffness, damping=base.damping, armature=base.armature,
        play_rad=math.radians(play_deg), play_rad_range=tuple(play_range), coulomb_nm=coulomb_nm, coulomb_range=tuple(coulomb_range))


@configclass
class Biped12HangStandG3EnvCfg(Biped12HangStandG2EnvCfg):
    """G3 (0920 새벽, 탐색용): G2 + **발목앞뒤 유격 4°(×0.75~1.25) + 쿨롱 마찰 2 Nm(×0.75~1.5)** (DeadzoneCoulombImplicit), 무릎 공유 편향 −9~−15, 힙 공유 −4~0,
    발목 상수 편향 0(유격이 대역을 만듦). 근거: hold_study 0920 '무릎 −12.6 + 힙 −2 + 유격 4 + 쿨롱 2' = 낙상 0 %·env 별 정지점 ±3.5°·PD 오차 힙/무릎 일치·60 N 밀기 0 % 낙상.
    ⚠ 여전히 '실측 각도 차이 + 유격/마찰 후보'로 만든 가설 플랜트. 관측은 물리 관절각(엔코더 모델은 별도 항 준비 중)."""

    def __post_init__(self):
        super().__post_init__()
        _swap_ankle_f_deadzone_coulomb(self, play_deg=4.0, play_range=(0.75, 1.25), coulomb_nm=2.0, coulomb_range=(0.75, 1.5))
        self.events.default_pose_bias.params.update({
            "shared_knee_rad": (-0.262, -0.157),   # −15 ~ −9°
            "shared_hip_f_rad": (-0.07, 0.0),      # −4 ~ 0°
            "shared_ankle_f_rad": None,
            "ankle_f_rad": (-0.02, 0.02),
        })
        self.events.ankle_f_gains = None          # G1/G2 의 발목 kp 배율 후보 제거(유격 모델로 대체)
        self.events.ankle_f_friction = None


@configclass
class Biped12HangStandG3HoldEnvCfg(Biped12HangStandG3EnvCfg):
    """G3 대조군(평가용)."""


@configclass
class Biped12HangStandG4EnvCfg(Biped12HangStandG3EnvCfg):
    """G4 (0920 낮, 탐색용): G3 플랜트 + **엔코더측 관측**(유격 관절의 유격 안 이동을 정책이 못 봄 = 실물 로터 엔코더 계약) + 자세 편차 벌점 ×3 + 지지 힘 더 짧게(0.3~0.6 s 유지, 0.5~1.5 s 방출).
    근거: G2/G3 정책이 힙 18~21°·발목 30°(한계)로 v5 에서 이탈해 서고, 명목 낙상 대부분이 지지 힘 잔존 구간에서 발생(MORNING_0920 §4)."""

    r0_range = (0.0, 0.5)          # 0 포함: 지지 0 스폰(실물 절차)이 학습 분포 안에 있어야 함 — G2/G3 는 지지 0 스폰에서 0/256 (커리큘럼 의존)
    hold_range = (0.3, 0.6)
    release_range = (0.5, 1.5)

    def __post_init__(self):
        super().__post_init__()
        self.events.hang_support_reset.params.update({"r0_range": (0.0, 0.5), "hold_range": (0.3, 0.6), "release_range": (0.5, 1.5)})
        P = self.observations.policy
        P.joint_pos.func = mdp_hang.encoder_joint_pos_rel
        P.joint_pos.params = {}
        if getattr(P, "joint_vel", None) is not None:
            P.joint_vel.func = mdp_hang.encoder_joint_vel
            P.joint_vel.params = {}
        if getattr(P, "joint_vel_ema", None) is not None:
            _prm = dict(P.joint_vel_ema.params or {})
            P.joint_vel_ema.func = mdp_hang.encoder_joint_vel_ema
            P.joint_vel_ema.params = {k: v for k, v in _prm.items() if k in ("alpha", "noise")}
        self.rewards.leg_pose_deviation.weight = -1.5


@configclass
class Biped12HangStandG4HoldEnvCfg(Biped12HangStandG4EnvCfg):
    """G4 대조군(평가용)."""


@configclass
class Biped12HangStandG5EnvCfg(Biped12HangStandG4EnvCfg):
    """G5 (0920 낮, 탐색용): G4 와 같되 **지지 0 만**(무개입 자립 상태에서 정책 시작 = 승윤님 확정 절차) — 커리큘럼 없는 대조군. 스크래치."""

    r0_range = (0.0, 0.0)
    hold_range = (0.1, 0.1)
    release_range = (0.1, 0.2)

    def __post_init__(self):
        super().__post_init__()
        self.events.hang_support_reset.params.update({"r0_range": (0.0, 0.0), "hold_range": (0.1, 0.1), "release_range": (0.1, 0.2)})


@configclass
class Biped12HangStandG5HoldEnvCfg(Biped12HangStandG5EnvCfg):
    """G5 대조군(평가용)."""


@configclass
class Biped12HangStandG6EnvCfg(Biped12HangStandG5EnvCfg):
    """G6 (0920 오후, 탐색용): G5(지지 0·엔코더 관측·유격 발목) + **실측 대역으로 좁힌 DR** + 홀드 사전(러너 init_noise_std 0.25).
    승윤님 지적(0920): 정답 동작(홀드)을 아는데 6000회 실패면 정합을 강제로 올리자. 상속된 광대역 DR 이 측정과 충돌하던 항목:
    actuator kp 0.3~1.1(실측 정적 kp150 부합) → 0.85~1.1 · base_com x −8~0 cm(뒤 CoM 가정) → ±2 cm · add_mass 0~2 kg → 0~0.5 ·
    effort_scale 0.6~1.0 → 제거 · calib_bias 관절 ±3°/발목 ±4°(관측·명령측 영점) → ±1° (물리 편향은 default_pose_bias 가 담당) ·
    reset_robot_joints ±5.7° → ±1.1° · 스폰 롤 ±4° → ±2°. 푸시(0.25 m/s)·마찰·지연 3~8틱은 유지(실물에 있는 불확실성)."""

    def __post_init__(self):
        super().__post_init__()
        ev = self.events
        ev.actuator_gains.params["stiffness_distribution_params"] = (0.85, 1.1)
        ev.actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)
        if getattr(ev, "base_com", None) is not None:
            ev.base_com.params["com_range"] = {"x": (-0.02, 0.02), "y": (-0.01, 0.01), "z": (0.0, 0.05)}
        if getattr(ev, "add_base_mass", None) is not None:
            ev.add_base_mass.params["mass_distribution_params"] = (0.0, 0.5)
        if getattr(ev, "effort_scale", None) is not None:
            ev.effort_scale = None
        if getattr(ev, "calib_bias", None) is not None:
            for k in ("joint_bias_rad", "ankle_bias_rad", "knee_bias_rad"):
                if k in ev.calib_bias.params and ev.calib_bias.params[k] is not None:
                    ev.calib_bias.params[k] = (-0.0175, 0.0175)
        if getattr(ev, "reset_robot_joints", None):
            ev.reset_robot_joints.params["position_range"] = (-0.02, 0.02)
        ev.reset_base.params["pose_range"]["roll"] = (-0.035, 0.035)


@configclass
class Biped12HangStandG6HoldEnvCfg(Biped12HangStandG6EnvCfg):
    """G6 대조군(평가용)."""


@configclass
class Biped12HangStandG7EnvCfg(Biped12HangStandG6EnvCfg):
    """G7 (0921, 승윤님 지시 0920 밤): **지속 외력·정책 실행 중 느린 균형 상실·앞뒤 회복 스텝** 학습 — G6 플랜트·DR 그대로.
    (1) 발목 한계 = 배포 config(config/joint_limits_12dof.json: soft 왼 −22~+39 / 오른 −39~+18.5, hard 40·45) — USD 구값 ±30 덮어씀,
        정책 명령은 soft 로 클램프(stage8 경계클램프 정합) [측정: 배포 config 가 검증 범위]
    (2) 외란 스케줄 [가정]: 에피소드당 1~3회, 지속 램프(3~20 N, 1~4 s 상승·1~3 s 유지) 60 % / 순간 충격(5~30 N, 0.2~0.4 s) 40 %,
        앞뒤 70 % · 좌우 30 %(×0.6). 학습 푸시(0.25 m/s)는 유지.
    (3) 온셋 인계: 홀드 0.5~1.5 s → 스무스스텝 블렌딩 1.0~2.0 s (지면 절차 = stage8 홀드 → robotctl handoff), env LPF β 0.2~0.4(브리지 0.27).
    (4) 자세·발접촉·정숙 계열 항은 외란 창(외란 중 + 해제 뒤 3 s) **밖**에서만 → 필요한 스텝을 벌점으로 막지 않음.
        action-rate 는 정지 구간 ×1.5(채터 억제), 창 안 ×0.3.
    (5) 회복 보너스: 외란 해제 뒤 6 s 안 0.5 s 연속 '조용히 서기'(양발·|ω|<0.15·|v|<0.05·q̇<0.3) 1회 = 성공 정의(단순 생존 아님).
    평가는 eval_ground_recovery(방향별 독립 실행, 같은 정착 상태). 웜스타트 G6 model_4500."""

    def __post_init__(self):
        super().__post_init__()
        ev = self.events; rw = self.rewards
        ev.joint_limits_config = EventTerm(func=mdp_hang.set_joint_limits_from_config, mode="startup",
                                           params={"config_path": "/home/ryu/humanoid_leg_test1/config/joint_limits_12dof.json"})
        ev.disturbance = EventTerm(func=mdp_hang.disturbance_reset, mode="reset",
                                   params={"n_range": (1, 3), "t0_range": (2.0, 10.0), "gap_range": (3.0, 7.0), "ramp_prob": 0.6,
                                           "F_ramp_range": (3.0, 20.0), "rise_range": (1.0, 4.0), "hold_range": (1.0, 3.0),
                                           "F_imp_range": (5.0, 30.0), "dur_imp_range": (0.2, 0.4), "lateral_prob": 0.3, "lateral_scale": 0.6,
                                           "p_active": 0.9})
        a = self.actions.joint_pos
        a.onset_hold_range = (0.0, 0.3); a.onset_blend_range = (1.0, 1.5); a.lpf_beta_range = (0.2, 0.4); a.clamp_to_soft_limits = True   # 홀드 거의 0: 시뮬 DR 홀드는 2~3 s 안에 넘어지는 인공물(RECOVERY_EVAL §4) — 홀드 구간을 두면 그 인공물을 학습에 굽는다(스모크: 에피소드 2.4 s). 블렌딩(홀드→정책 크로스페이드)은 유지
        # 외란 창 밖에서만 적용할 항 (스텝 허용)
        for name in ("feet_contact_gated", "stance_shape", "torso_pitch_band", "pose_symmetry", "leg_pose_deviation",
                     "feet_slide", "foot_yaw_slip", "feet_gap", "march_anchor"):
            term = getattr(rw, name, None)
            if term is None:
                continue
            setattr(rw, name, RewTerm(func=mdp_hang.rest_gated, weight=term.weight,
                                      params={"func": term.func, "params": dict(term.params or {}), "recover_s": 3.0}))
        ar = rw.action_rate_l2
        rw.action_rate_l2 = RewTerm(func=mdp_hang.rest_gated, weight=ar.weight * 1.5,
                                    params={"func": ar.func, "params": dict(ar.params or {}), "recover_s": 3.0})
        rw.action_rate_window = RewTerm(func=mdp_hang.window_gated, weight=ar.weight * 0.3,
                                        params={"func": ar.func, "params": dict(ar.params or {}), "recover_s": 3.0})
        rw.recovery_bonus = RewTerm(func=mdp_hang.recovery_bonus, weight=150.0,     # ×dt(0.02) → 회복 1회 = +3.0
                                    params={"recover_s": 6.0, "quiet_s": 0.5})
        self.episode_length_s = 20.0


@configclass
class Biped12HangStandG7HoldEnvCfg(Biped12HangStandG7EnvCfg):
    """G7 대조군(평가용)."""


@configclass
class Biped12HangStandG8EnvCfg(Biped12HangStandG6EnvCfg):
    """G8 (0921, 승윤님 지시): G7 의 두 결함을 고치고 **기존 뒤쪽 회복 스텝 유지 → 앞쪽·지속 외력 회복 → 회복 후 정숙** 순으로 잡는다.

    G7 대비 바뀐 것:
    1. **회복 보너스 미발화 수정** — 경성 문턱(|ω|<0.15·|v|<0.05·q̇<0.3, 0.5 s 연속)은 학습 내내 0 이었다.
       → `recovery_quality`(등급형 rational, 게이트 열린 동안 매 스텝 0~1) + `recovery_event`(균형 상실 → 실제 회복 1 회성).
    2. **3 초 시간 게이트 → 상태 기반 게이트** — 외란 해제 3 s 뒤 벌점이 돌아와 4~6 s 째 회복 스텝을 벌줬고, 외란 이벤트 없이
       서서히 기울 때는 창이 안 열렸다. → `recovery_gate_step`: 외란 힘 or (기울기>5°·|ω|>0.35·|v|>0.12·한 발 비접촉) 이면 열림,
       정상 상태 0.3 s 유지되면 닫힘, 연속 8 s 안전장치. 자세·접촉·정숙 항은 **닫힌 동안에만** 적용(state_gated).
    3. action-rate 배율·env LPF 는 **분리 실험 변수**(G8a~G8d) — G7 에서 스텝이 사라진 원인 후보.

    보상 크기 주의: Isaac Lab 은 weight×값×dt(0.02). quality 5.0 → 게이트 열림 15 %·품질 0.5 면 에피소드 ~0.4(=survive 급),
    event 30.0 → 회복 1 회당 0.6. 게이트 상태는 직전 스텝 것(보상이 interval 이벤트보다 먼저 계산됨 — 20 ms 지연, 전 항 동일)."""

    AR_REST_SCALE = 1.0          # 정지 구간 action-rate 배율 (G7 = 1.5)
    AR_OPEN_SCALE = 0.3          # 회복 창 action-rate 배율
    LPF_RANGE = None             # env LPF β 대역 (G7 = (0.2, 0.4)), None = 없음
    W_QUALITY = 5.0
    W_EVENT = 30.0

    def __post_init__(self):
        super().__post_init__()
        ev = self.events; rw = self.rewards
        ev.joint_limits_config = EventTerm(func=mdp_hang.set_joint_limits_from_config, mode="startup",
                                           params={"config_path": "/home/ryu/humanoid_leg_test1/config/joint_limits_12dof.json"})
        ev.disturbance = EventTerm(func=mdp_hang.disturbance_reset, mode="reset",
                                   params={"n_range": (1, 3), "t0_range": (2.0, 10.0), "gap_range": (3.0, 7.0), "ramp_prob": 0.6,
                                           "F_ramp_range": (3.0, 20.0), "rise_range": (1.0, 4.0), "hold_range": (1.0, 3.0),
                                           "F_imp_range": (5.0, 30.0), "dur_imp_range": (0.2, 0.4), "lateral_prob": 0.3, "lateral_scale": 0.6,
                                           "p_active": 0.9})
        # 상태 기반 게이트는 지지력 적용(hang_support_step) 뒤에 매 스텝 갱신 — dst_F_now 최신값을 쓴다
        ev.recovery_gate = EventTerm(func=mdp_hang.recovery_gate_step, mode="interval", interval_range_s=(0.0, 0.0),
                                     params={"tilt_ok": 6.5, "w_ok": 1.5, "v_ok": 0.5, "settle_s": 0.3, "max_open_s": 4.0, "lift_tilt": 3.0})
        a = self.actions.joint_pos
        a.onset_hold_range = (0.0, 0.3); a.onset_blend_range = (1.0, 1.5); a.clamp_to_soft_limits = True
        a.lpf_beta_range = tuple(self.LPF_RANGE) if self.LPF_RANGE is not None else (1.0, 1.0)
        # 자세·접촉·정숙 계열 = 게이트가 닫힌(안정) 동안에만 → 필요한 스텝은 시간과 무관하게 허용
        for name in ("feet_contact_gated", "stance_shape", "torso_pitch_band", "pose_symmetry", "leg_pose_deviation",
                     "feet_slide", "foot_yaw_slip", "feet_gap", "march_anchor", "settle_quiet_bonus"):
            term = getattr(rw, name, None)
            if term is None:
                continue
            setattr(rw, name, RewTerm(func=mdp_hang.state_gated, weight=term.weight,
                                      params={"func": term.func, "params": dict(term.params or {})}))
        ar = self.rewards.action_rate_l2      # G6 상속분(래핑 전 원본)
        base_w = ar.weight
        rw.action_rate_l2 = RewTerm(func=mdp_hang.state_gated, weight=base_w * self.AR_REST_SCALE,
                                    params={"func": ar.func, "params": dict(ar.params or {})})
        rw.action_rate_window = RewTerm(func=mdp_hang.open_gated, weight=base_w * self.AR_OPEN_SCALE,
                                        params={"func": ar.func, "params": dict(ar.params or {})})
        rw.recovery_quality = RewTerm(func=mdp_hang.recovery_quality, weight=self.W_QUALITY,
                                      params={"s_ang": 0.30, "s_lin": 0.10, "s_jvel": 0.50, "tilt_ok": 6.5})
        rw.recovery_event = RewTerm(func=mdp_hang.recovery_event, weight=self.W_EVENT)
        self.episode_length_s = 20.0


@configclass
class Biped12HangStandG8aEnvCfg(Biped12HangStandG8EnvCfg):
    """분리 실험 A = G7 손잡이 그대로(action-rate ×1.5 + LPF 0.2~0.4)."""
    AR_REST_SCALE = 1.5
    LPF_RANGE = (0.2, 0.4)


@configclass
class Biped12HangStandG8bEnvCfg(Biped12HangStandG8EnvCfg):
    """분리 실험 B = action-rate 배율만 제거(LPF 유지)."""
    AR_REST_SCALE = 1.0
    LPF_RANGE = (0.2, 0.4)


@configclass
class Biped12HangStandG8cEnvCfg(Biped12HangStandG8EnvCfg):
    """분리 실험 C = LPF 만 제거(action-rate ×1.5 유지)."""
    AR_REST_SCALE = 1.5
    LPF_RANGE = None


@configclass
class Biped12HangStandG8dEnvCfg(Biped12HangStandG8EnvCfg):
    """분리 실험 D = 둘 다 제거(= G6 손잡이)."""
    AR_REST_SCALE = 1.0
    LPF_RANGE = None


@configclass
class Biped12HangStandG8HoldEnvCfg(Biped12HangStandG8EnvCfg):
    """G8 대조군(평가용)."""


@configclass
class Biped12HangStandG9EnvCfg(Biped12HangStandG8EnvCfg):
    """G9 (0921 밤, 분리 실험): **G8 에서 `knee_stance_flexion` 만 회복 게이트에 추가**. 그 외 전부 G8 과 동일.

    근거(§10.3): G8 에서 게이트되지 않는 보상 중 `knee_stance_flexion` 이 +1.27 로 전체 최대 양의 항이며 무릎을
    34~60° 크라우치 밴드에 묶는다. 앞 스텝은 스윙 무릎을 밴드 밖으로 굽혀야 해 충돌하고, 뒤 스텝은 크라우치를
    유지한 채 발을 끌 수 있어 충돌이 적다(§10.1: 뒤쪽 보폭 −14 cm 를 밴드 안에서 달성).
    → 균형을 잃은 동안에만 밴드 강제를 푼다. 정지 구간에서는 그대로 유지(크라우치 자세 계약 불변).
    골반 종료 완화·스텝 단계 보상 추가는 이번에 넣지 않는다(승윤님 보류 지시)."""

    def __post_init__(self):
        super().__post_init__()
        t = getattr(self.rewards, "knee_stance_flexion", None)
        assert t is not None and t.func is not mdp_hang.state_gated, "G8 에서 이미 게이트된 항이면 단일 변경이 아니다"
        self.rewards.knee_stance_flexion = RewTerm(func=mdp_hang.state_gated, weight=t.weight,
                                                   params={"func": t.func, "params": dict(t.params or {})})


@configclass
class Biped12HangStandG9HoldEnvCfg(Biped12HangStandG9EnvCfg):
    """G9 대조군(평가용)."""


@configclass
class Biped12HangStandG10EnvCfg(Biped12HangStandG9EnvCfg):
    """G10 (0922): **G9 + 명령 기준 관절 한계 벌점** 단일 변경. 그 외 전부 G9 와 동일.

    근거(reports/real_0922): 실물 2회차에서 발목 **명령**이 56.7° (소프트 한계 39°)까지 갔다. stage8 클램프가 잘라
    관절은 38° 에 박히고 정책은 한계 밖을 계속 요청하며 감겼다 → 접지 상실 → 낙상. 시뮬 지연 스윕에서 같은 서명이
    지연 9~11틱(실측 p95 8.9)에서 재현된다. 기존 `dof_pos_limits` 는 실측 관절각 기준이라 이 요청을 벌하지 못한다."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.cmd_limit_excess = RewTerm(func=mdp_hang.action_limit_excess, weight=-5.0)


@configclass
class Biped12HangStandG10HoldEnvCfg(Biped12HangStandG10EnvCfg):
    """G10 대조군(평가용)."""


# ─────────────── 0928 G11: 플랜트 정합 (발목 정지 마찰 + 기본자세 편향 절반) ───────────────
_G11_POSE_BIAS_HALF = {
    "hip_f_rad": (-0.035, 0.035), "knee_rad": (-0.0175, 0.0175), "ankle_f_rad": (-0.01, 0.01),
    "shared_knee_rad": (-0.236, -0.183), "shared_hip_f_rad": (-0.052, -0.018), "reset_noise_rad": 0.025,
}


@configclass
class Biped12HangStandG11aEnvCfg(Biped12HangStandG9EnvCfg):
    """G11a (0928): **G9 + 플랜트 정합만**. 보상 불변(물리 변경의 효과를 따로 보기 위한 분리 세대).

    근거(reports/real_0928/WHY_SIX_GENERATIONS_FAILED.md): 정책 없이 가만히 서 있기가 G10 학습 분포에서 31.8 % 만
    성공했다(실물은 같은 자세로 101 s 자립). 원인 두 가지를 시험으로 분리:
      1) 발목 유격 안 **정지 마찰 부재** — 쿨롱이 tanh 형이라 속도 0 에서 0 → 발목이 스톱에 기대야 섬.
         → DeadzoneCoulomb 에 stiction(고정점 모델), τs = 쿨롱×2 = 3~6 Nm(쿨롱 DR ×0.75~1.5 를 따름). **가정값.**
      2) 기본자세 편향의 조합 폭 — 폭을 절반으로.
    홀드 3 s: 61/192 → 150/192(이 환경). 기점은 G9(v5); G10 의 명령 한계 벌점은 지연 강건성 저하 의심이 미해결이라 뺀다."""

    def __post_init__(self):
        super().__post_init__()
        from rl_walking.actuators_play import DeadzoneCoulombImplicitActuatorCfg as _DZC
        n = 0
        for _k, _a in self.scene.robot.actuators.items():
            if isinstance(_a, _DZC):
                _a.stiction = True; _a.stiction_ratio = 2.0; n += 1
        assert n == 1, f"발목 DeadzoneCoulomb 액추에이터가 1개여야 한다 (찾음 {n})"
        self.events.default_pose_bias.params.update(dict(_G11_POSE_BIAS_HALF))


@configclass
class Biped12HangStandG11aHoldEnvCfg(Biped12HangStandG11aEnvCfg):
    """G11a 대조군(평가용)."""


@configclass
class Biped12HangStandG11bEnvCfg(Biped12HangStandG11aEnvCfg):
    """G11b (0928): **G11a + 정착 보너스를 회복 게이트에서 해제** 단일 변경.

    G8 부터 settle_quiet_bonus 가 state_gated(회복 게이트가 닫힌 동안만)로 묶였다. 외란 뒤 가라앉아야 할 바로 그
    구간에 정착 유인이 꺼져 있었다. 또 G10 학습 에피소드는 평균 10.5 s 인데 첫 외란이 평균 6 s 에 와서 '외란 뒤
    다시 조용해지는' 경험이 드물었다. 보너스 자체가 settled 램프(양발 접촉·기울기 8° 이내 0.5 s 유지)를 곱하므로
    스텝 중에는 자연히 0 이다 — 게이트를 풀어도 회복 스텝을 벌하지 않는다."""

    def __post_init__(self):
        super().__post_init__()
        t = getattr(self.rewards, "settle_quiet_bonus", None)
        assert t is not None, "settle_quiet_bonus 가 없다"
        assert t.func is mdp_hang.state_gated, "G8 이후라면 게이트돼 있어야 한다 — 단일 변경 전제 확인"
        self.rewards.settle_quiet_bonus = RewTerm(func=t.params["func"], weight=t.weight, params=dict(t.params["params"]))


@configclass
class Biped12HangStandG11bHoldEnvCfg(Biped12HangStandG11bEnvCfg):
    """G11b 대조군(평가용)."""


@configclass
class Biped12HangStandG11cEnvCfg(Biped12HangStandG11aEnvCfg):
    """G11c (0929): **G11a + 회복 품질 보상 가중 5 → 10** 단일 변경. G11a-8699 에서 이어서 학습.

    근거(reports/gen_g11/VERDICT.md): G11a 는 정지 도달 75→120·명령 부호반전 0.32→0.06 으로 크게 개선됐으나
    뒤 18 N 회복이 v5 44 % → 34~37 % 로 약해져 사전 기준 D(≥39 %)에서 탈락. 가라앉는 법을 배우며 뒤 밀기에
    내딛는 반응이 줄었다. recovery_quality 는 회복 게이트가 열린 동안 안정 상태에 가까울수록 주는 등급형 보상 —
    정지 구간에는 작용하지 않으므로 정착을 해치지 않고 회복만 강화한다."""
    W_QUALITY = 10.0


@configclass
class Biped12HangStandG11cHoldEnvCfg(Biped12HangStandG11cEnvCfg):
    """G11c 대조군(평가용)."""


@configclass
class Biped12HangStandG12EnvCfg(Biped12HangStandG11aEnvCfg):
    """G12 (0929): **G11a + 액션 지연 3~8 → 9~14 틱** 단일 변경. G11a-8699 에서 이어서 학습.

    근거: 실물 g11a(0929 14:02)는 정책 출력·발목·몸통이 같은 2 Hz 로 휘청이다 넘어졌다. 명령→관절 반응 지연 실측
    16틱(320 ms, 0922·0929 동일). 시뮬에서 액션 지연 6틱 → 반응 ~11틱, 11틱 → ~16틱 이므로 실물 ≈ 시뮬 지연 11틱.
    G11a 는 3~8틱으로 학습하고 6틱에서 정착을 판정했다 — 실물 루프 지연이 학습 범위 밖이었다.
    기각한 대안(재생·시뮬 대조로 확인): 좌우 비대칭 · 접촉 깜빡임 · 관측 잡음 · 관절 위치 갱신 누락 37 %."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.min_delay_steps = 9
        self.actions.joint_pos.max_delay_steps = 14


@configclass
class Biped12HangStandG12HoldEnvCfg(Biped12HangStandG12EnvCfg):
    """G12 대조군(평가용)."""



@configclass
class Biped12HangStandG13EnvCfg(Biped12HangStandG12EnvCfg):
    """G13 (0930): **G12 + 정책 관측 끝에 last_action 16틀 이력(192차원) 추가** 단일 변경. 320 → 512.
    G12-9500 을 expand_checkpoint 로 0 열 확장해 이어서 학습(확장 직후 출력 = G12 와 동일).

    근거(reports/gen_g13/OSC_REPRO_PLAN.md·OSC_REPRO_RESULT.md): 실물 0929 G12 는 보정 방향은 맞지만(피치와 역위상)
    명령→관절 반응이 0.34 s(17틱) 늦어, 흔들림이 커지다 넘어졌다. 시뮬도 지연이 길수록 작은 밀기 뒤 성장·낙상이
    늘었다(뒤 8 N: 지연 6 → 17틱 1.9 → 19.0 %). 홀드는 흔들림을 유지하지 않았다 → 정책 루프 문제.
    기존 관측의 액션 이력은 5틀(0.1 s)뿐이라 이미 보냈지만 아직 반영 안 된 보정(9~14틱)을 정책이 모른다 →
    같은 보정을 더 넣는 과보정 가설. 대조군: 같은 G12 환경·같은 반복수로 G12-9500 을 그대로 이어 학습."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.envs import mdp as _mdp
        P = self.observations.policy
        assert getattr(P, "action_hist16", None) is None
        P.action_hist16 = _ObsTerm(func=_mdp.last_action, clip=(-10.0, 10.0), history_length=16, flatten_history_dim=True)


@configclass
class Biped12HangStandG13HoldEnvCfg(Biped12HangStandG13EnvCfg):
    """G13 대조군(평가용)."""


# ─────────────── 0930 G14: 회복 이벤트 보너스 제거 (2×2: 관측 이력 × 보너스) ───────────────
def _drop_recovery_event(cfg):
    t = getattr(cfg.rewards, "recovery_event", None)
    assert t is not None and t.weight > 0, "recovery_event 가 있어야 단일 변경"
    cfg.rewards.recovery_event = None


@configclass
class Biped12HangStandG14aEnvCfg(Biped12HangStandG12EnvCfg):
    """G14a (0930): **G12 + recovery_event(균형 상실→회복 1회당 보너스) 제거** 단일 변경. 대조군 = G12c(같은 seed·반복수).

    근거(reports/gen_g13/G13_VERDICT.md): G12 환경에서 이어 학습하면 관측 변경과 무관하게 무외란 지속 흔들림이
    29 → 88~152 로 늘고 감쇠가 92.6 → 75~84 % 로 떨어졌다. 같은 기간 recovery_event 보상이 +40~60 % 늘었다.
    가설: 흔들리며 균형을 잃었다 되찾기를 반복하면 이 보너스를 반복 수령 → 조용히 서기보다 흔들림이 이득."""

    def __post_init__(self):
        super().__post_init__()
        _drop_recovery_event(self)


@configclass
class Biped12HangStandG14aHoldEnvCfg(Biped12HangStandG14aEnvCfg):
    """G14a 대조군(평가용)."""


@configclass
class Biped12HangStandG14bEnvCfg(Biped12HangStandG13EnvCfg):
    """G14b (0930): **G13(액션 16틀 관측) + recovery_event 제거**. 대조군 = G13(같은 seed·반복수). G14a 와 함께 2×2."""

    def __post_init__(self):
        super().__post_init__()
        _drop_recovery_event(self)


@configclass
class Biped12HangStandG14bHoldEnvCfg(Biped12HangStandG14bEnvCfg):
    """G14b 대조군(평가용)."""


# ─────────────── 0930 G15: stage8 지연 수정 뒤 — 보상·지연 단일 변경 (모두 G12-9500 에서 2000 반복, 대조군 = G12c) ───────────────
# 사전 기준: reports/gen_g15/G15_CRITERIA.md (학습 전 작성). 평가 = 새 실물 루프 상당 지연(1·3틱) + 스트레스(11틱·18 N).

@configclass
class Biped12HangStandG15qEnvCfg(Biped12HangStandG12EnvCfg):
    """G15q: **정숙 보너스 settled 조건의 골반고 문턱 0.60 → 0.52 m** 단일 변경.
    근거: CROUCH_V5 정착 골반고 0.594–0.602 m 라 문턱 0.60 에 걸려 settle_quiet_bonus 기여가 +0.001 (사실상 미발동, 구 스탠스용 문턱).
    0.52 = 정착 높이 −8 cm, 낙상 종료(0.45)보다 7 cm 위."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.settle_quiet_bonus
        assert t.func is mdp_hang.state_gated and t.params["func"] is mdp_hang.settle_quiet_bonus
        assert "min_height" not in t.params["params"], "기본값(0.60)을 쓰고 있어야 단일 변경"
        t.params["params"]["min_height"] = 0.52


@configclass
class Biped12HangStandG15uEnvCfg(Biped12HangStandG12EnvCfg):
    """G15u: **상태 게이트 해제** 단일 변경 — state_gated 로 감싼 항(균형 상실 중 꺼지던 자세·접촉·정숙·action-rate)을 상시 적용,
    게이트 열림 전용 약한 action-rate(action_rate_window)는 이중 계산이라 제거. 회복 품질·이벤트 보상은 그대로.
    근거: 오픈소스 4종 모두 상태 게이트 없음(reports/oss_review), 우리는 흔들려 게이트가 열리면 벌점이 꺼져 흔들림이 싸다.
    위험: G8 이 게이트를 둔 이유(회복 스텝을 벌하지 않기) — 18 N 뒤 밀기 생존을 비퇴보 지표로 본다."""

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards; n = 0
        for name in list(vars(rw).keys()):
            t = getattr(rw, name, None)
            if t is not None and getattr(t, "func", None) is mdp_hang.state_gated:
                setattr(rw, name, RewTerm(func=t.params["func"], weight=t.weight, params=dict(t.params.get("params") or {})))
                n += 1
        assert n >= 10, f"게이트된 항이 10개 이상이어야 한다 (찾음 {n})"
        assert getattr(rw, "action_rate_window", None) is not None
        rw.action_rate_window = None


@configclass
class Biped12HangStandG15wEnvCfg(Biped12HangStandG12EnvCfg):
    """G15w: **상시 롤·피치 각속도 exp 커널(가중 +1.0, std 0.5 rad/s) 추가** 단일 변경. 기존 ang_vel_xy_l2(−0.1)는 그대로."""

    def __post_init__(self):
        super().__post_init__()
        assert getattr(self.rewards, "ang_vel_xy_exp", None) is None
        self.rewards.ang_vel_xy_exp = RewTerm(func=mdp_hang.ang_vel_xy_exp, weight=1.0, params={"std": 0.5})


@configclass
class Biped12HangStandG15dEnvCfg(Biped12HangStandG12EnvCfg):
    """G15d: **액션 지연 9–14 → 0–14 틱** 단일 변경. stage8 적체 수정(1f67f48) 뒤 실물 루프 ≈ 시뮬 지연 0–1틱(추정, 실물 재측정 전).
    G12(9–14 학습)가 지연 1·3틱에서 오히려 가장 안정했으므로 긴 지연은 유지하고 짧은 쪽만 넓힌다."""

    def __post_init__(self):
        super().__post_init__()
        assert self.actions.joint_pos.min_delay_steps == 9 and self.actions.joint_pos.max_delay_steps == 14
        self.actions.joint_pos.min_delay_steps = 0


@configclass
class Biped12HangStandG15kEnvCfg(Biped12HangStandG12EnvCfg):
    """G15k: **발목앞뒤 PD kp 150 → 40, kd 5 → 2** 단일 변경 (유격·마찰 모델 그대로). 평가도 이 환경(같은 게인)으로 한다.
    근거(reports/oss_review): 참고 휴머노이드 전부 발목 kp 가 무릎의 1/4–1/10 (unitree G1 40/2, HV G1 20/0.2). 우리는 전 관절 150.
    가설: 백래시 ±4° + 강한 발목 PD + 지연 = 흔들림을 유지하는 한계 사이클 조건. ⚠ 실물 적용은 하드웨어맵 발목 게인 변경(모터 5·11)이
    필요하므로 시뮬 통과 뒤에도 승윤님 결정 사항. 균형 토크 권한이 줄어드는 대가(5° 기울기 중력 토크 ~3.8 Nm/발목 → 오차 ~5°)."""

    def __post_init__(self):
        super().__post_init__()
        a = self.scene.robot.actuators["ankle_f_play"]
        assert float(a.stiffness) == 150.0 and float(a.damping) == 5.0, (a.stiffness, a.damping)
        a.stiffness = 40.0
        a.damping = 2.0


# ─────────────── 1001 G16: 엔코더 속도 관측 결함 수정(기반) ───────────────
def _use_encoder_obs_v2(cfg):
    """정책 관측의 엔코더 관절각·속도·속도 EMA 를 *_v2 로 — 같은 스텝 이중 호출에서 joint_vel_ema 가 0 을 받던 결함 제거.
    실물 러너는 진짜 관절속도의 EMA 를 넣으므로(policy_runner_stage4) 이것이 배포 계약과 같은 의미다."""
    P = cfg.observations.policy
    for nm, fn in (("joint_pos", mdp_hang.encoder_joint_pos_rel_v2), ("joint_vel", mdp_hang.encoder_joint_vel_v2),
                   ("joint_vel_ema", mdp_hang.encoder_joint_vel_ema_v2)):
        t = getattr(P, nm, None)
        assert t is not None, nm
        t.func = fn
        t.params = {k: v for k, v in dict(t.params or {}).items() if k in ("alpha", "noise")}


@configclass
class Biped12HangStandG16oEnvCfg(Biped12HangStandG12EnvCfg):
    """G16o (1001): **G12 + 엔코더 속도 관측 결함 수정(v2)** 단일 변경. G12-9500 에서 이어 학습. 대조군 = G12c(같은 반복, 구관측).

    결함: G4(0920) 이후 모든 세대에서 mdp_hang 의 구버전 encoder_joint_vel* 가 개선판을 모듈 뒤쪽에서 덮어써,
    joint_vel_ema(정책 관측 60차원)가 항상 속도 0 + 잡음을 받았다. 실물 러너는 진짜 속도의 EMA 를 넣어, 흔들릴 때 그 채널이
    학습 분포의 6–17배(0929 실물 95 % 0.51·최대 1.34 rad/s vs 학습 95 % 0.078). reports/gen_g16/ 참조."""

    def __post_init__(self):
        super().__post_init__()
        _use_encoder_obs_v2(self)


@configclass
class Biped12HangStandG16aEnvCfg(Biped12HangStandG16oEnvCfg):
    """G16a: G16o + **recovery_quality 의 양발 접촉 조건 제거** 단일 변경 (가중·게이트·나머지 불변)."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.recovery_quality
        assert t.func is mdp_hang.recovery_quality and "require_both" not in (t.params or {})
        t.params["require_both"] = False


@configclass
class Biped12HangStandG16iEnvCfg(Biped12HangStandG16oEnvCfg):
    """G16i: G16o + **순간 충격 세기 F_imp_range (5,30) → (10,45) N** 단일 변경 — 스텝이 필요한 충격 비율 상향(van Marum Single Contact++)."""

    def __post_init__(self):
        super().__post_init__()
        p = self.events.disturbance.params
        assert tuple(p["F_imp_range"]) == (5.0, 30.0), p["F_imp_range"]
        p["F_imp_range"] = (10.0, 45.0)


@configclass
class Biped12HangStandG16dEnvCfg(Biped12HangStandG16oEnvCfg):
    """G16d: G16o + **액션 지연 DR 9–14 → 0–6 틱** 단일 변경 — stage8 적체 수정 뒤 짧은 루프(추정 1–3틱)용."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        assert a.min_delay_steps == 9 and a.max_delay_steps == 14
        a.min_delay_steps = 0
        a.max_delay_steps = 6


# ─────────────────────── 1001 G17: 실물 실측 정합 세대 (reports/gen_g17/) ───────────────────────
def _g17_sensor_contract(cfg, gyro_max_lag=3, hold_ticks=5, extra_delay=1, tilt_off_deg=(6.0, 20.0)):
    """정책 관측을 실물 센서 계약에 맞춘다 (관측 차원·순서 불변 — 러너·브리지 그대로):
    자이로(base_ang_vel·gyro_ema 입력) env 별 0..gyro_max_lag 틱 지연 / 발 접촉 = 법선력 & 발 평평(θ_off 2–6°) · 10 Hz 표본 유지 · 1틱 전송."""
    P = cfg.observations.policy
    P.base_ang_vel.func = mdp_hang.base_ang_vel_lagged
    P.base_ang_vel.params = {"max_lag": int(gyro_max_lag)}
    ge = P.gyro_ema
    ge.func = mdp_hang.gyro_ema_lagged
    ge.params = {"alpha": ge.params.get("alpha", 0.1), "noise": ge.params.get("noise", 0.05), "max_lag": int(gyro_max_lag)}
    fc = P.feet_contact
    keep = {k: v for k, v in dict(fc.params).items() if k in ("sensor_cfg", "threshold", "threshold_range")}
    keep.update({"hold_ticks": int(hold_ticks), "extra_delay": int(extra_delay), "tilt_off_deg": tuple(tilt_off_deg)})
    fc.func = mdp_hang.feet_contact_real
    fc.params = keep


@configclass
class Biped12HangStandG17oEnvCfg(Biped12HangStandG16oEnvCfg):
    """G17o (1001 밤) — 실측 정합 기반: G16o(관측 v2) + **액션 지연 DR 9–14 → 0–3틱**(1001 실측 명령 경로 p95 3 ms·응답 4틱 = 시뮬 지연 0–1)
    + **실물 센서 계약**(_g17_sensor_contract: 자이로 지연 0–3틱 가정·접촉 10 Hz 유지·발 기울면 접촉 꺼짐).
    보상·외란·게이트 불변. 이후 G17 변형의 공통 대조군."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        assert a.min_delay_steps == 9 and a.max_delay_steps == 14
        a.min_delay_steps = 0
        a.max_delay_steps = 3
        _g17_sensor_contract(self)


@configclass
class Biped12HangStandG17rEnvCfg(Biped12HangStandG17oEnvCfg):
    """G17r: G17o + **명령 변화 벌점 ×3**(action_rate_l2·action_rate_window 둘 다, 게이트 불변) 단일 변경.
    근거: G16d(짧은 지연 학습) 부호반전 ~100배 채터, 1001 실물 3 Hz 는 자이로·관절속도 고이득 되먹임 → 빠른 반응 억제(0802 march_v3 action_rate 6배 선례)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight *= 3.0
        self.rewards.action_rate_window.weight *= 3.0


@configclass
class Biped12HangStandG17pEnvCfg(Biped12HangStandG17oEnvCfg):
    """G17p: G17o + **빠른 지속 외력**: 램프 상승 1–4 s → 0.3–4 s, 램프 세기 3–20 → 3–25 N 단일 변경(외란 분포만).
    근거: 실물 실패 = 0.5 s 안팎으로 '꾸욱' 누르는 지속력(1001 3회차), G12 는 느린 앞 10 N(0.5 s 상승) 낙상 48 %."""

    def __post_init__(self):
        super().__post_init__()
        p = self.events.disturbance.params
        assert tuple(p["rise_range"]) == (1.0, 4.0) and tuple(p["F_ramp_range"]) == (3.0, 20.0)
        p["rise_range"] = (0.3, 4.0)
        p["F_ramp_range"] = (3.0, 25.0)


@configclass
class Biped12HangStandG17rpEnvCfg(Biped12HangStandG17pEnvCfg):
    """G17rp: G17p + 명령 변화 벌점 ×3 (G17r 과 G17p 의 결합 — 두 단일 변경 효과가 각각 확인될 때 해석)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2.weight *= 3.0
        self.rewards.action_rate_window.weight *= 3.0


@configclass
class Biped12HangStandG18oEnvCfg(Biped12HangStandG17oEnvCfg):
    """G18o (1002) — G18 공통 대조군: G17o + **접촉 관측 v2**(feet_contact_real_v2: 접촉 임계를 에피소드 시작에만 추출 —
    v1 은 delay 0 경로에서 _contact_reset_ids 미소비로 임계가 매 틱 재추출, OSS 감사 3). 그 외 G17o 와 동일.
    G18 변형은 모두 이것 위의 단일 변경이며 같은 출발점(G12-9500)·같은 반복·같은 seed 로 짝 비교한다."""

    def __post_init__(self):
        super().__post_init__()
        fc = self.observations.policy.feet_contact
        assert fc.func is mdp_hang.feet_contact_real
        fc.func = mdp_hang.feet_contact_real_v2


@configclass
class Biped12HangStandG18dEnvCfg(Biped12HangStandG18oEnvCfg):
    """G18d: G18o + **액션 지연 DR 0–3 → 1–3틱** 단일 변경 (OSS 분석 3: 실측 d≈1, 지연 0 은 실물에 없음;
    짧은 지연 학습 = 정지 떨림(G16d ~100배·G17o 6–8배) 가설)."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        assert a.min_delay_steps == 0 and a.max_delay_steps == 3
        a.min_delay_steps = 1


@configclass
class Biped12HangStandG18qEnvCfg(Biped12HangStandG18oEnvCfg):
    """G18q: G18o + **회복 품질 보상의 접촉 조건을 외란 창(외력 시작 ~ 종료 + 1.5 s) 안에서만 '한 발 이상'** 단일 변경
    (OSS 분석 2: 게이트 열림 중 유일한 양(+) 보상이 q *= both 라 스윙 = 구조적 손해; Ferigo 2021·van Marum 2024).
    창 밖은 양발 조건 그대로 → G16a 의 무외란 발 들기(게이트 자기 개방) 경로 차단."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.recovery_quality
        assert t.func is mdp_hang.recovery_quality
        t.params = dict(t.params or {}); t.params["step_window_s"] = 1.5


@configclass
class Biped12HangStandG18fEnvCfg(Biped12HangStandG18oEnvCfg):
    """G18f: G18o + **배포 LPF 를 학습에 복원(β 0.2–0.4) + 관측 '직전 액션' = 그 LPF 를 거친 명령(원시 액션 EMA, 같은 β)**.
    두 변경은 한 계약(관측이 실제 관절 명령과 일치)이라 묶는다. 근거: 1002 폐루프 대조 — 25 Hz 떨림 = last_action 자기루프(H-loop 지지),
    G18j 처럼 이 경로 이득을 깎으면 충격 회복(b18)을 잃는다 → 정보는 두고 루프만 필터를 거치게 한다.
    배포 계약 변경: 러너가 actions 관측에 EMA(β0.27)를 넣어야 함(평가기 --last_action_ema 0.27). G8 이후 학습엔 LPF 없음(배포만 β0.27)."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        assert tuple(a.lpf_beta_range) == (1.0, 1.0)
        a.lpf_beta_range = (0.2, 0.4)
        P = self.observations.policy
        P.actions.func = mdp_hang.last_action_lpf
        P.actions.params = {"action_name": "joint_pos"}


@configclass
class Biped12HangStandG19vEnvCfg(Biped12HangStandG18fEnvCfg):
    """G19v: G18f + **관절속도 관측 = 유한차분 속도의 EMA(β0.25)** 단일 변경 (배포 계약 변경: 러너 qvel 관측에 같은 EMA).
    근거: 1002 부록 D — G18f 잔여 떨림 경로 = 관절속도 되먹임(EMA 0.1 로 늦추면 부호반전 −98 %, b18 +5 %p 만)."""

    def __post_init__(self):
        super().__post_init__()
        jv = self.observations.policy.joint_vel
        assert jv.func is mdp_hang.encoder_joint_vel_v2
        p = dict(jv.params or {}); p["beta"] = 0.25
        jv.func = mdp_hang.encoder_joint_vel_lpf; jv.params = p


@configclass
class Biped12HangStandG19nEnvCfg(Biped12HangStandG18fEnvCfg):
    """G19n: G18f + **관절속도 관측 잡음 ±0.3 → ±1.5 rad/s** 단일 변경 (배포 계약 불변; legged_gym 기본값 수준, OSS 분석 3).
    가설: 잡음이 관절속도 고주파 되먹임 이득을 억제(정규화)해 잔여 떨림 감소."""

    def __post_init__(self):
        super().__post_init__()
        n = self.observations.policy.joint_vel.noise
        assert abs(n.n_max - 0.3) < 1e-9
        n.n_min, n.n_max = -1.5, 1.5


@configclass
class Biped12HangStandG20dEnvCfg(Biped12HangStandG19vEnvCfg):
    """G20d: G19v + **액션 지연 DR 0–3 → 1–5틱** 단일 변경. 근거: 1002 부록 E — 느린 흔들림은 단일 관측 경로가 아니라 지연 의존
    (G19v-10700 d0 <3.9 · d1 8.0 · d3 16.1 %) → 실측 d≈1·평가 가드 d3 을 학습 분포 안쪽에 두어 위상 여유를 확보(가설)."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        assert a.min_delay_steps == 0 and a.max_delay_steps == 3
        a.min_delay_steps, a.max_delay_steps = 1, 5


@configclass
class Biped12HangStandG20sEnvCfg(Biped12HangStandG19vEnvCfg):
    """G20s: G19v + **안정 상태(회복 게이트 닫힘)에서만 몸통 롤·피치 각속도 벌점 −1.0 추가** 단일 변경(기존 −0.1 상시 항은 그대로).
    근거: 부록 E — 흔들림 = 폐루프 감쇠 부족. 게이트가 열린(외란·회복) 동안은 꺼져 회복 스텝을 벌하지 않는다(G15 정숙 보상의 회복 손실 교훈)."""

    def __post_init__(self):
        super().__post_init__()
        from .mdp_rewards import ang_vel_xy_l2_capped
        self.rewards.quiet_ang_vel = RewTerm(func=mdp_hang.state_gated, weight=-1.0,
                                             params={"func": ang_vel_xy_l2_capped, "params": {}})


# =====================================================================================================
# G21 (1002, 승윤님 설계안) — 기준 정책 G19v-10700 에서 이어 학습. 각 클래스는 직전 단계 대비 단일 변경.
# =====================================================================================================
_G21_MG = 17.6 * 9.81


@configclass
class Biped12HangStandG21a1EnvCfg(Biped12HangStandG19vEnvCfg):
    """G21a1: feet_impact 문턱 180 N(12 kg 시절) → 1.5 mg ≈ 259 N (17.6 kg)."""

    def __post_init__(self):
        super().__post_init__()
        p = dict(self.rewards.feet_impact.params or {}); p["max_force_n"] = 1.5 * _G21_MG
        self.rewards.feet_impact.params = p


@configclass
class Biped12HangStandG21a2EnvCfg(Biped12HangStandG19vEnvCfg):
    """G21a2: 낙상 종료 벌점 실효 −0.2 → −1.5 (is_terminated 0/1 × weight × dt 0.02 → weight −75; Isaac Lab 2.3.2 에 dt 상쇄 함수 없음)."""

    def __post_init__(self):
        super().__post_init__()
        assert self.rewards.termination_penalty.weight == -10.0
        self.rewards.termination_penalty.weight = -75.0


@configclass
class Biped12HangStandG21a3EnvCfg(Biped12HangStandG19vEnvCfg):
    """G21a3: settle_quiet_bonus 의 골반고 이진 문턱(>0.60) → 연속 exp(−((h−0.60)/0.015)²) (settle_quiet_bonus_v2, 나머지 동일)."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.settle_quiet_bonus
        assert t.func is mdp_hang.state_gated and t.params["func"] is mdp_hang.settle_quiet_bonus
        inner = dict(t.params["params"] or {})
        t.params = {"func": mdp_hang.settle_quiet_bonus_v2, "params": inner}


def _g21a_fix(cfg):
    p = dict(cfg.rewards.feet_impact.params or {}); p["max_force_n"] = 1.5 * _G21_MG; cfg.rewards.feet_impact.params = p
    cfg.rewards.termination_penalty.weight = -75.0
    t = cfg.rewards.settle_quiet_bonus
    t.params = {"func": mdp_hang.settle_quiet_bonus_v2, "params": dict(t.params["params"] or {})}


@configclass
class Biped12HangStandG21aEnvCfg(Biped12HangStandG19vEnvCfg):
    """G21a: a1 + a2 + a3 결합(보상 결함 교정 묶음) — G21b 이후의 바탕."""

    def __post_init__(self):
        super().__post_init__()
        _g21a_fix(self)


@configclass
class Biped12HangStandG21bEnvCfg(Biped12HangStandG21aEnvCfg):
    """G21b: 무장 래치 회복 게이트(STAND/RECOVERY/SETTLE) + recovery_quality_v2 + 착지·하중·정착 이벤트 + 정착 보상 (한 묶음 — 설계안: 분리 불가).
    B(STAND) = 종전 state_gated 항 그대로(+ 무장 안 된 한 발 들기 −2), settle_quiet_bonus 는 SETTLE 의 settle_quality 로 대체.
    C(RECOVERY) = quality_v2 +5 · 착지 +5 · 하중 수용 +10 · 정착 성공 +20 · 회복 시간 −0.1 · action_rate −0.06 · 시간초과 실효 −1.5.
    D(SETTLE) = settle_quality +1 · action_rate −0.30 · feet_slide −0.15 · foot_yaw_slip −0.5."""

    def __post_init__(self):
        super().__post_init__()
        ev = self.events
        g = ev.recovery_gate
        assert g.func is mdp_hang.recovery_gate_step
        g.func = mdp_hang.recovery_gate_v2; g.params = {}
        pr = ev.push_robot
        pr.func = mdp_hang.push_with_arm; pr.params = {"velocity_range": dict(pr.params["velocity_range"]), "arm_s": 2.5}
        rw = self.rewards
        fs, fy, ar = rw.feet_slide, rw.foot_yaw_slip, rw.action_rate_l2
        rw.settle_quiet_bonus = None
        rw.recovery_event = None
        rw.recovery_quality = RewTerm(func=mdp_hang.recovery_quality_v2, weight=5.0)
        rw.unarmed_single_support = RewTerm(func=mdp_hang.unarmed_single_support, weight=-2.0)
        rw.valid_touchdown = RewTerm(func=mdp_hang.g2_event, weight=5.0, params={"name": "ev_td"})
        rw.load_accept = RewTerm(func=mdp_hang.g2_event, weight=10.0, params={"name": "ev_load"})
        rw.settle_success = RewTerm(func=mdp_hang.g2_event, weight=20.0, params={"name": "ev_settle"})
        rw.recovery_timeout = RewTerm(func=mdp_hang.g2_event, weight=-75.0, params={"name": "ev_timeout"})
        rw.recovery_time = RewTerm(func=mdp_hang.g2_recovery_time, weight=-0.10)
        arw = rw.action_rate_window
        rw.action_rate_window = RewTerm(func=mdp_hang.g2_phase_gated, weight=arw.weight,
                                        params={"func": arw.params["func"], "params": dict(arw.params.get("params") or {}), "phases": [1]})
        rw.settle_quality = RewTerm(func=mdp_hang.settle_quality, weight=1.0)
        rw.settle_action_rate = RewTerm(func=mdp_hang.g2_phase_gated, weight=-0.30,
                                        params={"func": ar.params["func"], "params": dict(ar.params.get("params") or {}), "phases": [2]})
        rw.settle_feet_slide = RewTerm(func=mdp_hang.g2_phase_gated, weight=-0.15,
                                       params={"func": fs.params["func"], "params": dict(fs.params.get("params") or {}), "phases": [2]})
        rw.settle_foot_yaw = RewTerm(func=mdp_hang.g2_phase_gated, weight=-0.5,
                                     params={"func": fy.params["func"], "params": dict(fy.params.get("params") or {}), "phases": [2]})


@configclass
class Biped12HangStandG21c1EnvCfg(Biped12HangStandG21bEnvCfg):
    """G21c1: G21b + 관절속도 관측 env 별 0–2 틱 지연 DR (EMA β0.25 뒤)."""

    def __post_init__(self):
        super().__post_init__()
        jv = self.observations.policy.joint_vel
        assert jv.func is mdp_hang.encoder_joint_vel_lpf
        p = dict(jv.params or {}); p["max_lag"] = 2
        jv.func = mdp_hang.encoder_joint_vel_lpf_lagged; jv.params = p


@configclass
class Biped12HangStandG21c2EnvCfg(Biped12HangStandG21c1EnvCfg):
    """G21c2: G21c1 + 관절각 관측 env 별 0–1 틱 지연 DR."""

    def __post_init__(self):
        super().__post_init__()
        jp = self.observations.policy.joint_pos
        assert jp.func is mdp_hang.encoder_joint_pos_rel_v2
        p = dict(jp.params or {}); p["max_lag"] = 1
        jp.func = mdp_hang.encoder_joint_pos_rel_v2_lagged; jp.params = p


@configclass
class Biped12HangStandG21dEnvCfg(Biped12HangStandG21c2EnvCfg):
    """G21d: G21c2 + STAND·SETTLE 에서만 액션 2차 차분 벌점 −0.02 (회복 중 빠른 동작은 그대로)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_accel_quiet = RewTerm(func=mdp_hang.g2_phase_gated, weight=-0.02,
                                                  params={"func": mdp_hang.action_accel_l2, "params": {}, "phases": [0, 2]})


@configclass
class Biped12HangStandG22aEnvCfg(Biped12HangStandG21aEnvCfg):
    """G22a (1003): G21a + **액션 지연 DR 0–3 → 0–5 틱** 단일 변경. 근거: G21 정식 판정 전부가 지연 3틱 흔들림(5.3–6.5 %)으로 C0 탈락 —
    평가 가드 d3 이 학습 분포 경계(0–3)였다. 실측 d≈1 은 그대로 안쪽, d3 을 분포 안쪽으로."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        assert a.min_delay_steps == 0 and a.max_delay_steps == 3
        a.max_delay_steps = 5


@configclass
class Biped12HangStandG22cEnvCfg(Biped12HangStandG21c2EnvCfg):
    """G22c (1003): G21c2(회복 최상) + 액션 지연 DR 0–3 → 0–5 틱 단일 변경."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        assert a.min_delay_steps == 0 and a.max_delay_steps == 3
        a.max_delay_steps = 5


@configclass
class Biped12HangStandG22wEnvCfg(Biped12HangStandG21aEnvCfg):
    """G22w (1003, 외부 검토 권고): G21a + **액션 지연 범위 0–3 유지, 표본 분포만 가중** 0:10 % · 1:30 % · 2:25 % · 3:35 % 단일 변경.
    실측 d≈1 을 많이 보이면서 약점(지연 3틱 흔들림)을 의도적으로 과표본. 지연은 에피소드마다 1회 추출·고정(종전과 같음).
    gyro 지연·보상·게이트는 G21a 그대로(원인 분리)."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        assert a.min_delay_steps == 0 and a.max_delay_steps == 3
        a.delay_probs = (0.10, 0.30, 0.25, 0.35)


# =====================================================================================================
# G22R · W1 (1003, 승윤님 설계 지시서) — 부모 G21a-11300 에서 독립 두 분기. plant·관측·액션·DR·PPO 동결, 과제만 변경.
# 기준 보상 = G21a env.yaml(지시서: yaml 우선). 변경점은 각 클래스 docstring 에 전부 적는다.
# =====================================================================================================
from isaaclab.managers import CurriculumTermCfg as _CurrTerm


@configclass
class _G22RCurriculumCfg:
    force_stage = _CurrTerm(func=mdp_hang.g22r_force_curriculum)


@configclass
class _W1CurriculumCfg:
    command_stage = _CurrTerm(func=mdp_hang.w1_command_curriculum)


@configclass
class Biped12HangStandG22REnvCfg(Biped12HangStandG21aEnvCfg):
    """G22R — Bidirectional Recovery (G21a + 회복 과제만).
    외란: disturbance_reset_g22r — 종전 횟수·시각·램프/충격 모양 유지, 에피소드 범주(무외란/앞/뒤)·힘 범위만 단계 R1–R4(성적 승격),
          좌우 외란 없음, 순간 충격 힘 = 램프 범위 ×1.5. 속도 밀기(push_robot ±0.25 m/s, 3–6 s)는 G21a 그대로 + 무장 래치.
    게이트: recovery_gate_g22r(무장 래치·회복 방향 래치·valid 이륙(≥60 ms·≥1 cm)·착지(≥2 cm)·하중(0.25 mg·100 ms)·정착 300 ms).
    보상: G21a 전부 유지하되 G21a 회복 항(recovery_quality +5·recovery_event +30 — 구 게이트용)은 지시서 회복 항으로 대체,
          action_rate_window 는 RECOVERY 에서만(−0.06). 추가: vel_quality +1.5·ang_quality +1.0·step_direction +0.5·
          touchdown +5·load +10·settle +25(이벤트 ×dt → 실효 0.1/0.2/0.5)·recovery_time −0.10·unarmed_takeoff −10·swing_timeout −10."""

    def __post_init__(self):
        super().__post_init__()
        ev, rw = self.events, self.rewards
        d = ev.disturbance
        assert d.func is mdp_hang.disturbance_reset
        d.func = mdp_hang.disturbance_reset_g22r; d.params = dict(d.params, branch="g22r")
        pr = ev.push_robot
        pr.func = mdp_hang.push_with_arm_g22; pr.params = {"velocity_range": dict(pr.params["velocity_range"]), "arm_s": 2.5, "branch": "g22r"}
        g = ev.recovery_gate
        assert g.func is mdp_hang.recovery_gate_step
        g.func = mdp_hang.recovery_gate_g22r; g.params = {}
        rw.recovery_quality = None
        rw.recovery_event = None
        arw = rw.action_rate_window
        rw.action_rate_window = RewTerm(func=mdp_hang.g2_phase_gated, weight=-0.06,
                                        params={"func": arw.params["func"], "params": dict(arw.params.get("params") or {}), "phases": [1]})
        rw.recovery_vel_quality = RewTerm(func=mdp_hang.g22r_vel_quality, weight=1.5)
        rw.recovery_ang_quality = RewTerm(func=mdp_hang.g22r_ang_quality, weight=1.0)
        rw.recovery_step_direction = RewTerm(func=mdp_hang.g22r_step_direction, weight=0.5)
        rw.valid_touchdown_event = RewTerm(func=mdp_hang.g22r_event, weight=5.0, params={"name": "ev_td"})
        rw.load_accept_event = RewTerm(func=mdp_hang.g22r_event, weight=10.0, params={"name": "ev_load"})
        rw.settle_success_event = RewTerm(func=mdp_hang.g22r_event, weight=25.0, params={"name": "ev_settle"})
        rw.recovery_time = RewTerm(func=mdp_hang.g2_recovery_time, weight=-0.10)
        rw.unarmed_takeoff_event = RewTerm(func=mdp_hang.g22r_event, weight=-10.0, params={"name": "ev_unarmed"})
        rw.swing_timeout_event = RewTerm(func=mdp_hang.g22r_event, weight=-10.0, params={"name": "ev_swto"})
        self.curriculum = _G22RCurriculumCfg()


# W1: 걷기 명령 중 끄는 G21a 서기 항(지시서 14 + 같은 성격의 정숙·회복 항) / 걷기 중 지시서 값으로 바꾸는 항
_W1_STAND_ONLY = ("stance_shape", "pose_symmetry", "march_anchor", "feet_contact_gated", "leg_pose_deviation",
                  "settle_quiet_bonus", "recovery_quality", "recovery_event", "action_rate_window")
_W1_SWAP = {"track_lin_vel_xy_exp": 3.0, "track_ang_vel_z_exp": 1.5, "flat_orientation_l2": -4.0, "base_height_l2": -1.5,
            "lin_vel_z_l2": -0.3, "ang_vel_xy_l2": -0.15, "feet_slide": -0.20, "foot_yaw_slip": -0.5, "action_rate_l2": -0.15}


@configclass
class Biped12HangStandW1EnvCfg(Biped12HangStandG21aEnvCfg):
    """W1 — Commanded Walking (G21a + 저속 전진 명령 과제만).
    명령: vx 만(vy=0, 요 0 — heading 유지 제어는 G21a 그대로), 2–4 s 리샘플, 단계 S0–S3 서기 비율·vx 범위(성적 승격, w1_command_curriculum).
    게이트: |vx_cmd| < 0.03 = 서기 → G21a 보상 그대로 / ≥ 0.03 = 걷기 → 서기 전용 항 OFF(_W1_STAND_ONLY), _W1_SWAP 항은 지시서 걷기 값,
          보행 항 추가(교대 +1.0·스텝 진행 이벤트 +5·발 들림 1.5–4 cm +0.2·양발 지지 과다 −0.2·공중 과다 이벤트 −5).
          토크·가속·한계·충격·비정상 접촉·종료 등 상시 항은 G21a 그대로. 회복 게이트는 G21a 그대로(회복 항은 서기에서만).
    외란: S0–S2 없음(속도 밀기 포함), S3 에서 12.5 % env 에 앞/뒤 5–8 N (disturbance_reset_g22r branch w1). G22R 회복 보상 복사 안 함."""

    def __post_init__(self):
        super().__post_init__()
        ev, rw = self.events, self.rewards
        c = self.commands.base_velocity
        c.resampling_time_range = (2.0, 4.0)
        c.ranges.lin_vel_x = (0.03, 0.08); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
        c.rel_standing_envs = 0.60
        if hasattr(c, "rel_spin_envs"):
            c.rel_spin_envs = 0.0; c.rel_backward_envs = 0.0
        d = ev.disturbance
        assert d.func is mdp_hang.disturbance_reset
        d.func = mdp_hang.disturbance_reset_g22r; d.params = dict(d.params, branch="w1")
        pr = ev.push_robot
        pr.func = mdp_hang.push_with_arm_g22; pr.params = {"velocity_range": dict(pr.params["velocity_range"]), "arm_s": 2.5, "branch": "w1"}
        ev.w1_gait = EventTerm(func=mdp_hang.w1_gait_step, mode="interval", interval_range_s=(0.0, 0.0), params={})
        for name in _W1_STAND_ONLY + tuple(_W1_SWAP):
            t = getattr(rw, name)
            setattr(rw, name, RewTerm(func=mdp_hang.w1_stand_gated, weight=t.weight,
                                      params={"func": t.func, "params": dict(t.params or {})}))
            if name in _W1_SWAP:
                # 걷기 판은 구 회복 게이트(state_gated) 를 벗긴 원 함수 — 걷는 동안 한 발 지지로 게이트가 열려 꺼지는 것 방지
                f, p = (t.params["func"], dict(t.params.get("params") or {})) if t.func is mdp_hang.state_gated else (t.func, dict(t.params or {}))
                setattr(rw, name + "_walk", RewTerm(func=mdp_hang.w1_walk_gated, weight=_W1_SWAP[name], params={"func": f, "params": p}))
        rw.gait_alternation = RewTerm(func=mdp_hang.w1_gait_alternation, weight=1.0)
        rw.step_progress_event = RewTerm(func=mdp_hang.w1_value, weight=5.0, params={"name": "ev_step"})
        rw.swing_clearance = RewTerm(func=mdp_hang.w1_value, weight=0.20, params={"name": "clear"})
        rw.excessive_double_support = RewTerm(func=mdp_hang.w1_value, weight=-0.20, params={"name": "ds_pen"})
        rw.excessive_air_time_event = RewTerm(func=mdp_hang.w1_value, weight=-5.0, params={"name": "ev_air"})
        self.curriculum = _W1CurriculumCfg()


@configclass
class Biped12HangStandW1bEnvCfg(Biped12HangStandW1EnvCfg):
    """W1b (1003 밤): W1 + **걷기 추종 보상 std 0.5 → 0.1 단일 변경**(걷기 게이트 track_lin_vel_xy_exp_walk 만; 서기 항·요 추종 그대로).
    근거(계산): std 0.5 에서 0.03/0.05/0.08 m/s 명령은 정지만으로 추종 보상 99.6/99.0/97.5 % → W1 착지 0. std 0.1 이면 91.4/77.9/52.7 %."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.track_lin_vel_xy_exp_walk
        inner = dict(t.params["params"]); assert abs(inner["std"] - 0.5) < 1e-9
        inner["std"] = 0.1; t.params = {"func": t.params["func"], "params": inner}


@configclass
class Biped12HangStandG22R2EnvCfg(Biped12HangStandG22REnvCfg):
    """G22R2 (1003 밤): G22R + **외란 힘 범위만 상향** 단일 변경 — R1 앞 10–12/뒤 10–15 · R2 12–14/12–18 · R3 13–15/15–18 · R4 15/18 N (비율 동일).
    근거: G22R 은 R1(앞 5–8 N, 발목 전략으로 충분)에서 앞 스텝을 한 번도 탐색하지 않음(sf15 이륙 0, 관절 한계·토크 포화 0 % — 기구 제약 증거 없음)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.disturbance.params = dict(self.events.disturbance.params, stage_set="r2")


@configclass
class Biped12HangStandStepB0EnvCfg(Biped12HangStandG21aEnvCfg):
    """STEP-B0 (1003 밤, 승윤님): 급속 하중전환 → 짧은 스윙 → 착지 primitive 부트스트랩. 부모 G21a-11300, plant·관측·PPO 동결.
    모드: vx_cmd 0.05(스텝 모드 신호, 추종 보상 아님) / 0(서기 — G21a 보상 그대로), 서기 50 %, 4–6 s 리샘플. 외란·속도 밀기 없음.
    스텝 모드: W1 과 같은 서기 전용 항 OFF + 추종 항 OFF, feet_slide −0.15·foot_yaw_slip −0.5·action_rate −0.1(구 게이트 벗긴 원 함수),
    sb0_step 일정(1.1 Hz 클록 위상) 보상: load_transfer +2.0 · swing_unload +0.5 · swing_clearance +1.0 · swing_progress +0.5 ·
    touchdown +5 · load_accept +10 · settle +15(이벤트 1틱 ×dt) · premature_swing −2.0. s_com 목표 없음."""

    def __post_init__(self):
        super().__post_init__()
        ev, rw = self.events, self.rewards
        c = self.commands.base_velocity
        c.resampling_time_range = (4.0, 6.0); c.ranges.lin_vel_x = (0.05, 0.05); c.ranges.lin_vel_y = (0.0, 0.0); c.ranges.ang_vel_z = (0.0, 0.0)
        c.rel_standing_envs = 0.5
        if hasattr(c, "rel_spin_envs"):
            c.rel_spin_envs = 0.0; c.rel_backward_envs = 0.0
        d = ev.disturbance
        d.func = mdp_hang.disturbance_reset_g22r; d.params = dict(d.params, branch="w1")     # W1 S0 = 외란 없음
        pr = ev.push_robot
        pr.func = mdp_hang.push_with_arm_g22; pr.params = {"velocity_range": dict(pr.params["velocity_range"]), "arm_s": 2.5, "branch": "w1"}
        ev.sb0 = EventTerm(func=mdp_hang.sb0_step, mode="interval", interval_range_s=(0.0, 0.0), params={})
        swap = {"feet_slide": -0.15, "foot_yaw_slip": -0.5, "action_rate_l2": -0.10}
        for name in _W1_STAND_ONLY + ("track_lin_vel_xy_exp", "track_ang_vel_z_exp") + tuple(swap):
            t = getattr(rw, name)
            setattr(rw, name, RewTerm(func=mdp_hang.w1_stand_gated, weight=t.weight, params={"func": t.func, "params": dict(t.params or {})}))
            if name in swap:
                f, p = (t.params["func"], dict(t.params.get("params") or {})) if t.func is mdp_hang.state_gated else (t.func, dict(t.params or {}))
                setattr(rw, name + "_step", RewTerm(func=mdp_hang.w1_walk_gated, weight=swap[name], params={"func": f, "params": p}))
        for name, wgt, key in (("load_transfer", 2.0, "r_transfer"), ("swing_unload", 0.5, "r_unload"), ("swing_clearance", 1.0, "r_clear"),
                               ("swing_progress", 0.5, "r_prog"), ("touchdown_event", 5.0, "ev_td"), ("load_accept_event", 10.0, "ev_load"),
                               ("settle_event", 15.0, "ev_settle"), ("premature_swing", -2.0, "r_prem")):
            setattr(rw, name, RewTerm(func=mdp_hang.sb0_value, weight=wgt, params={"name": key}))


@configclass
class Biped12HangStandStepB1EnvCfg(Biped12HangStandStepB0EnvCfg):
    """STEP-B1 (1004): STEP-B0 + **하중 보상 형태만** 교체 — 곱형 ramp(지지)×ramp(스윙) → 선형 (지지 Fz − 스윙 Fz)/mg clamp [0, 0.65], 가중치 +2.0,
    전환 구간에서만(같은 창). 나머지 구조 동일(단일 변수). 근거: STEP-B0 load_transfer 1500 반복 평탄(최대치 ~0.4 %) — 대칭 근처 기울기 ≈ 0."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.load_transfer = RewTerm(func=mdp_hang.sb0_value, weight=2.0, params={"name": "r_asym"})


@configclass
class _StepICurriculumCfg:
    imit_anneal = _CurrTerm(func=mdp_hang.sbi_anneal_curriculum)


@configclass
class Biped12HangStandStepI0EnvCfg(Biped12HangStandStepB1EnvCfg):
    """STEP-I0 (1004, NEXT 2): STEP-B1 + **시연 primitive 모방 보상** — joint_pose_reference +1.5 · foot_position_reference +2.0 ·
    contact_phase_reference +2.0 (스텝 모드만, 감쇠 배율 곱), load_asymmetry(선형) 가중 2.0 → 1.0. 감쇠 I0 1.0 → 0.75 → 0.5 → 0.25 → 0.05
    (스텝 모드 예정 반주기 블록의 하중 수용 비율 ≥ 0.5 로 승격). 기준 = reports/gen_step/ref_step_primitive.npz."""

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards
        self.events.sbi = EventTerm(func=mdp_hang.sbi_step, mode="interval", interval_range_s=(0.0, 0.0),
                                    params={"ref_path": "/home/ryu/humanoid_leg_test1/reports/gen_step/ref_step_primitive.npz"})
        rw.load_transfer = RewTerm(func=mdp_hang.sb0_value, weight=1.0, params={"name": "r_asym"})
        rw.joint_pose_reference = RewTerm(func=mdp_hang.sb0_value, weight=1.5, params={"name": "r_qref"})
        rw.foot_position_reference = RewTerm(func=mdp_hang.sb0_value, weight=2.0, params={"name": "r_foot"})
        rw.contact_phase_reference = RewTerm(func=mdp_hang.sb0_value, weight=2.0, params={"name": "r_cref"})
        self.curriculum = _StepICurriculumCfg()


@configclass
class _StepACurriculumCfg:
    assist_anneal = _CurrTerm(func=mdp_hang.step_assist_curriculum)


@configclass
class Biped12HangStandStepA0EnvCfg(Biped12HangStandStepI0EnvCfg):
    """STEP-A0 (1004, NEXT 3): STEP-I0 + **보조 탐색** — 스텝 모드 관절 목표 += α·(기준 관절 궤적 − 반주기 시작 값), α 2.0 → 1.5 → 1.0 → 0.5 → 0.25 → 0
    (블록 하중 수용 ≥ 0.5 로 감쇠; 1004 개루프: α 2 = 이륙 31 %·하중 수용 12.8 %). 모방 보상 배율은 1.0 고정(감쇠는 α 가 0 된 뒤 별도 세대). 합격은 α = 0 에서의 스텝 성공."""

    def __post_init__(self):
        super().__post_init__()
        self.events.step_assist = EventTerm(func=mdp_hang.install_step_assist, mode="startup",
                                            params={"ref_path": "/home/ryu/humanoid_leg_test1/reports/gen_step/ref_step_primitive.npz"})
        self.curriculum = _StepACurriculumCfg()


@configclass
class Biped12HangStandStepA0bEnvCfg(Biped12HangStandStepA0EnvCfg):
    """STEP-A0b (1004): STEP-A0 + **착지 이벤트 변위 조건 = 방향 무관 수평 2–10 cm** 단일 변경(sb0_step td_any_dir).
    근거: A0 학습 중 α=2 에서 착지·하중 수용 이벤트 0 — 이벤트가 '앞 3–10 cm' 만 인정, 보조 잔차는 앞뒤 관절을 고정해 제자리/옆 스텝만 만듦
    (개루프 α=2 eval_step 의 방향 무관 착지 14.5 %). 보조 스텝이 착지·하중·정착 보상과 α 감쇠 조건에 닿게 한다."""

    def __post_init__(self):
        super().__post_init__()
        self.events.sb0.params = dict(self.events.sb0.params, td_any_dir=True)


@configclass
class Biped12HangStandStepA0cEnvCfg(Biped12HangStandStepA0bEnvCfg):
    """STEP-A0c (1004): STEP-A0b + **선형 하중 비대칭 보상(r_asym) 제거** 단일 변경.
    근거: A0·A0b 모두 α=2 보조 아래 '발 붙인 채 하중만 빼기'로 수렴(A0 α=2 평가: ready 87 %·unload 89 %·이륙 0 %). 그 행동에 직접 값을 주는 항은
    r_asym 하나(접촉 무관, 스윙 0.175 mg 포화 — OSS 라운드 5). 제거하면 하중 전환은 보조·착지/하중/정착 이벤트로만 보상된다."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.load_transfer = None


@configclass
class Biped12HangStandStepA0dEnvCfg(Biped12HangStandStepA0cEnvCfg):
    """STEP-A0d (1004): STEP-A0c + **스텝 모드 env 균형 하네스**(골반 롤·피치 복원 토크 k 60 Nm/rad·c 5, 일정 강도) 단일 변경.
    근거(OSS 라운드 5 손익 계산, 가정 포함): 살아 있는 가치 V≈16 ≫ 스텝 이득 ≈1 → 시도당 낙상 5–6 % 이상이면 '안 드는' 쪽이 이득. 하네스로 낙상 위험을 낮춘다."""

    def __post_init__(self):
        super().__post_init__()
        self.events.step_harness = EventTerm(func=mdp_hang.step_balance_harness, mode="interval", interval_range_s=(0.0, 0.0), params={})


@configclass
class Biped12HangStandStepA0eEnvCfg(Biped12HangStandStepA0cEnvCfg):
    """STEP-A0e (1004): STEP-A0c + **스텝 모드 env 부분 체중 지지**(골반 수직 위 힘 0.2 mg × α/2 — 보조 α 와 함께 0 까지 감쇠) 단일 변경.
    개루프(G21a, α=2): 지지 0 → 이륙 30 %·낙상 7.8 %, 0.2 mg → 이륙 72 %·낙상 4.7 %, 0.4 mg → 이륙 92 %·낙상 49 %.
    근거: 자세 하네스(A0d)는 개루프에서 낙상과 함께 이륙도 줄임(k60 이륙 30→13 %, k200 3.8 %) — 몸통 운동이 이륙에 쓰인다. 수직 지지는 발 하중만 줄인다."""

    def __post_init__(self):
        super().__post_init__()
        self.events.step_lift = EventTerm(func=mdp_hang.step_lift_support, mode="interval", interval_range_s=(0.0, 0.0), params={})


@configclass
class Biped12HangStandStepA0fEnvCfg(Biped12HangStandStepA0cEnvCfg):
    """STEP-A0f (1004): STEP-A0c + **knee_stance_flexion(+4, 접지 발마다 무릎 굽힘 보상)을 스텝 모드에서 OFF** 단일 변경(서기 모드 그대로).
    근거(diag_reward_terms, G21a·α=2): 한 발 공중 틱 − 양발 접지 틱 보상 차의 최대 항 = knee_stance_flexion −0.025/틱(전체 0.16/틱의 ~16 %),
    스텝 이벤트 합 +0.0004 이하 — 사실상 '양발 접지' 보상이 이륙을 벌한다. A0·A0b·A0c 모두 보조 아래 이륙 억제로 수렴한 구조적 원인 후보."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.knee_stance_flexion
        self.rewards.knee_stance_flexion = RewTerm(func=mdp_hang.w1_stand_gated, weight=t.weight, params={"func": t.func, "params": dict(t.params or {})})


@configclass
class Biped12HangStandStepA0gEnvCfg(Biped12HangStandStepA0cEnvCfg):
    """STEP-A0g (1004): STEP-A0c + **보조 기준 위상 6틱 앞당김** 단일 변경.
    근거(diag_phase_events, G21a·α=2): 실제 이륙이 위상 0.4–0.6(보상 스윙 창 0.16–0.5 뒤, 접촉 기준=1 인 착지 창)에 몰림 — 관절 추종 지연 ~6틱.
    앞당김 6틱 개루프: 이륙 0.1–0.3·착지 0.2–0.4 로 창 안. 보조로 생긴 공중이 스윙 보상을 받고 접촉 위상 벌을 피하게 한다."""

    def __post_init__(self):
        super().__post_init__()
        self.events.step_assist.params = dict(self.events.step_assist.params, lead_ticks=6)


@configclass
class Biped12HangStandStepA0hEnvCfg(Biped12HangStandStepA0gEnvCfg):
    """STEP-A0h (1004): STEP-A0g + **기준 동작 = 스윙 굽힘 펄스 합성판**(ref_step_primitive_flex.npz: 스윙 8틱 사인 펄스 힙 +12°·무릎 −25°·발목 +13°) 단일 변경.
    보조 α 는 기준 진폭에 상대적이라 시작 α 를 1.0 으로(감쇠 1.0 → 0.75 → 0.5 → 0.25 → 0). 모방 보상도 같은 기준.
    근거(개루프 G21a·앞당김 6): 원 기준 α=2 → 진짜 스텝 12 %·발 높이 0.09 cm(끌기); 굽힘판 α=1 → 이륙 75 %·착지 62 %·하중 수용 53 %·공중 0.16 s·낙상 27 %/8 s."""

    def __post_init__(self):
        super().__post_init__()
        ref = "/home/ryu/humanoid_leg_test1/reports/gen_step/ref_step_primitive_flex.npz"
        self.events.step_assist.params = dict(self.events.step_assist.params, ref_path=ref, alpha0=1.0)
        self.events.sbi.params = dict(self.events.sbi.params, ref_path=ref)
        self.curriculum.assist_anneal.params = {"alphas": (1.0, 0.75, 0.5, 0.25, 0.0)}


@configclass
class Biped12HangStandStepA0iEnvCfg(Biped12HangStandStepA0hEnvCfg):
    """STEP-A0i (1004): STEP-A0h + **knee_stance_flexion 스텝 모드 OFF** 단일 변경(A0f 와 같은 처치를, 굽힘 기준·위상 정렬이 갖춰진 A0h 위에서).
    근거: A0h 학습 블록 착지 16 %(반복 10) → 1 %(반복 400) 로 정책이 보상에 닿는 보조 스텝까지 억제(같은 실패 5회째).
    diag_reward_terms: 틱당 총보상 0.16 중 knee_stance_flexion +0.064 가 최대 양(+) 항(서기 보상) — 스텝 이벤트(0.1–0.3/스텝)보다 '가만히 서 있기' 가치를 키운다."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.knee_stance_flexion
        self.rewards.knee_stance_flexion = RewTerm(func=mdp_hang.w1_stand_gated, weight=t.weight, params={"func": t.func, "params": dict(t.params or {})})


@configclass
class Biped12HangStandStepA0jEnvCfg(Biped12HangStandStepA0iEnvCfg):
    """STEP-A0j (1004): STEP-A0i + **놓친 스텝 벌점**(스텝 모드, 반주기에 유효 착지 없음 → 이벤트 실효 −1.0, 가중 −50) 단일 변경.
    근거(diag_reward_terms 반환 비교, G21a): A0h 구성 8 s 반환 스텝 35.9 vs 서기 61.7, A0i 구성 25.0 vs 30.1 — 스텝 모드에서 안 디뎌도 손해가 없다
    (서 있어도 contact/pose/foot 기준 보상·survive 를 거의 다 받음). 반주기당 가치 차 ~0.3 + 낙상 위험을 덮도록 −1.0."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.missed_step_event = RewTerm(func=mdp_hang.sb0_value, weight=-50.0, params={"name": "ev_miss"})


@configclass
class Biped12HangStandStepA0icEnvCfg(Biped12HangStandStepA0iEnvCfg):
    """STEP-A0i-c (1004): A0i 이어 학습용 — **α 감쇠 조건만** 블록 하중 수용 ≥ 0.50 → ≥ 0.25 (환경·보상 동일).
    근거: A0i 는 억제 없이 착지 0.17→0.30·하중 0.12→0.26 으로 상승 중이었으나 0.50 문턱(학습 확률 정책 블록)에 못 미쳐 600 반복 규칙으로 중단."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.assist_anneal.params = dict(self.curriculum.assist_anneal.params, min_full=0.25)


@configclass
class _StepAEnvCurriculumCfg:
    assist_anneal = _CurrTerm(func=mdp_hang.step_assist_curriculum_env)


@configclass
class Biped12HangStandStepA0kEnvCfg(Biped12HangStandStepA0iEnvCfg):
    """STEP-A0k (1004, 대기 후보): A0i + **env 별 α ~ U[0, α_max]·감시 env 20 % α=0·되돌림 있는 상대 감쇠**(α_max 1.0 → −0.25 씩) 단일 구조 변경(감쇠 방식).
    근거: OSS 라운드 2·7 — 전역 계단 감쇠는 특정 α 과적합·영구 정지 위험(A0i-c 문턱 0.25 는 α=1 블록 0.26 의 96 %), Yu 2018 롤아웃 내 보조 혼합·상대 진행 조건."""

    def __post_init__(self):
        super().__post_init__()
        self.events.assist_alpha_env = EventTerm(func=mdp_hang.sample_assist_alpha, mode="reset", params={})
        self.curriculum = _StepAEnvCurriculumCfg()


@configclass
class Biped12HangStandStepA0lEnvCfg(Biped12HangStandStepA0kEnvCfg):
    """STEP-A0l (1004, 대기): STEP-A0k + **놓친 스텝 벌점 실효 −0.4/반주기**(가중 −20) 단일 변경.
    근거(OSS 7·8): 스텝 모드에서 서 있어도 contact_phase_ref 일치 0.826 — 스텝 전부 빼먹어도 손해 0.007/틱(양 보상 합의 9 %). 크기는 서기 반주기 양 합(~1.8)보다 충분히 작게."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.missed_step_event = RewTerm(func=mdp_hang.sb0_value, weight=-20.0, params={"name": "ev_miss"})


@configclass
class Biped12HangStandStepA0mEnvCfg(Biped12HangStandStepA0icEnvCfg):
    """STEP-A0m (1004): STEP-A0i-c(전역 α 계단, 문턱 0.25) + **놓친 스텝 벌점 실효 −0.4/반주기** 단일 변경. A0i-11900 출발.
    근거: A0k(env 별 α 혼합)는 보조 스텝까지 붕괴 → 보조가 성립하는 전역 α 위에서 '안 디디기' 비용(OSS 7: 서기 접촉 일치 0.826)만 추가."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.missed_step_event = RewTerm(func=mdp_hang.sb0_value, weight=-20.0, params={"name": "ev_miss"})


@configclass
class _StepARelCurriculumCfg:
    assist_anneal = _CurrTerm(func=mdp_hang.step_assist_curriculum_rel)


@configclass
class Biped12HangStandStepA0nEnvCfg(Biped12HangStandStepA0iEnvCfg):
    """STEP-A0n (1004, 대기): A0i + **전역 α 미세 감쇠(0.1 단위)·상대 유지 기준(70 %)·되돌림(40 %)** — 감쇠 방식 1개 변경. A0i-11900 출발.
    근거: A0i-c 의 −0.25 계단은 블록 하중 수용을 0.25 → 0.11 로 반 이하로 떨어뜨림."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum = _StepARelCurriculumCfg()


@configclass
class Biped12HangStandW2aEnvCfg(Biped12HangStandStepA0mEnvCfg):
    """W2-1 (1004): STEP primitive(A0m) 위 저속 전진 보행. 보조 끔(α 0 고정, 감쇠 커리큘럼 제거).
    변경: ① 스텝 모드 vx ~ U[0.05, 0.15](서기 50 %, 4–6 s 리샘플) ② 걷기 추종 보상 ON(track_lin_vel_xy_exp, std 0.1, 가중 3.0, 스텝 모드)
    ③ 앞 착지 목표 Δx = 0.91·vx(2·v·T_h; 0.05 → 4.6 cm ≈ 종전 5 cm, 0.15 → 13.7 cm). 상한 0.15 = 슬루 4°/틱 계산(OSS 9).
    나머지(클록 일정·스텝 이벤트·놓친 스텝 벌점·서기 모드 G21a)는 A0m 그대로."""

    def __post_init__(self):
        super().__post_init__()
        self.events.step_assist.params = dict(self.events.step_assist.params, alpha0=0.0)
        self.curriculum = None
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (0.05, 0.15)
        self.events.sbi.params = dict(self.events.sbi.params, vx_foot_gain=0.91)
        t = self.rewards.track_lin_vel_xy_exp
        inner_f = t.params["func"]; inner_p = dict(t.params["params"] or {}); inner_p["std"] = 0.1
        self.rewards.track_lin_vel_walk = RewTerm(func=mdp_hang.w1_walk_gated, weight=3.0, params={"func": inner_f, "params": inner_p})


@configclass
class Biped12HangStandG23RaEnvCfg(Biped12HangStandStepA0mEnvCfg):
    """G23R-a (1004): STEP primitive(A0m) + **G22R 외란(앞/뒤 범주, 힘 R3 고정: 앞 10–12·뒤 12–18 N, 무외란 범주 속도 밀기 없음)** 단일 변경. 보조 끔(α 0).
    질문: 스텝 primitive 를 가진 정책이 기존 보상(서기 모드 G21a 회복 항) 아래에서 앞 밀기에 스텝을 쓰게 되는가. 계약(관측·명령 의미) 불변."""

    def __post_init__(self):
        super().__post_init__()
        self.events.step_assist.params = dict(self.events.step_assist.params, alpha0=0.0)
        d = self.events.disturbance; d.params = dict(d.params, branch="g22r", stage_override=2)   # R3 고정(앞 10–12·뒤 12–18 N) — G22R 커리큘럼 통계는 이 env 게이트로 안 모임
        pr = self.events.push_robot; pr.params = dict(pr.params, branch="g22r")
        self.curriculum = None


@configclass
class Biped12HangStandW2bEnvCfg(Biped12HangStandW2aEnvCfg):
    """W2-2 (1004): W2-1 + **스텝 모드 요 추종 보상**(track_ang_vel_z_exp, 가중 1.5 — W1 지시서 값) 단일 변경.
    근거: W2-1 요 표류 13–18°/8 s — StepB0 계보는 track_ang_vel_z 를 서기 전용으로 게이트(걷기 판 없음)."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.track_ang_vel_z_exp
        self.rewards.track_ang_vel_walk = RewTerm(func=mdp_hang.w1_walk_gated, weight=1.5,
                                                  params={"func": t.params["func"], "params": dict(t.params["params"] or {})})


@configclass
class _G23RbCurriculumCfg:
    hint_anneal = _CurrTerm(func=mdp_hang.recovery_hint_curriculum)


@configclass
class Biped12HangStandG23RbEnvCfg(Biped12HangStandG23RaEnvCfg):
    """G23R-b (1004): G23R-a + **학습 전용 밀림 → 스텝 모드 힌트**(서기 env 외력 & 기울기>4° 또는 |v|>0.15 → 1 s 동안 vx 0.05) 와
    힌트 확률 감쇠 1.0 → 0.75 → 0.5 → 0.25 → 0 (외란 에피소드 낙상 비악화 ×1.1 조건). 최종 판정은 힌트 0(배포 계약 불변).
    근거: G23R-a 서기 밀림에서 primitive 미사용(sf15 스텝 0 %), 스텝은 명령+클록 조건에 묶여 있음. 지시서 G22R3 위상 힌트 감쇠 설계."""

    def __post_init__(self):
        super().__post_init__()
        self.events.recovery_hint = EventTerm(func=mdp_hang.recovery_step_hint, mode="interval", interval_range_s=(0.0, 0.0), params={})
        self.curriculum = _G23RbCurriculumCfg()


@configclass
class Biped12HangStandW2cEnvCfg(Biped12HangStandW2bEnvCfg):
    """W2-3 (1004): W2-2 + **스텝 모드 요 추종 std 0.5 → 0.1** 단일 변경.
    근거: diag_yaw_drift — 표류 11.3°/7 s(한 발 지지 중 7.9°, 지지발 요 회전 합 14.4°, 방향 무작위). 표류율 ~0.03 rad/s 는 std 0.5 에서 보상 99.6 %(기울기 ≈ 0);
    평가 HeadingHold(kp 1.0) 명령에도 표류 불변 → 정책이 작은 요 명령을 따르지 않음. std 0.1 이면 0.03 rad/s 에서 91 %."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.track_ang_vel_walk
        p = dict(t.params["params"]); p["std"] = 0.1
        t.params = {"func": t.params["func"], "params": p}


@configclass
class Biped12HangStandW2dEnvCfg(Biped12HangStandW2cEnvCfg):
    """W2-4 (1004, 대기): W2-3 + **G21a 원래 외란 복원**(disturbance_reset: 1–3 회·램프 3–20 N/충격 5–30 N·좌우 30 %, push_robot ±0.25 m/s 3–6 s) 단일 변경.
    근거(Case E): STEP·W2 계열은 외란 전부 OFF 로 학습 — A0m sb10 낙상 27.7 %(G21a 9.8), W2-2 sb10 26.8·b18 28.6·sf10 47.3 %(G21a 9.8·4.5·22.3).
    스텝 모드 env 에도 같은 외란(실물 보행 중 외란 대비). 보상·명령·클록 그대로."""

    def __post_init__(self):
        super().__post_init__()
        d = self.events.disturbance
        d.func = mdp_hang.disturbance_reset
        p = dict(d.params); [p.pop(k, None) for k in ("branch", "w1_p_active", "w1_F", "imp_scale", "stage_set", "stage_override")]
        d.params = p
        pr = self.events.push_robot
        from isaaclab.envs.mdp.events import push_by_setting_velocity
        pr.func = push_by_setting_velocity; pr.params = {"velocity_range": dict(pr.params["velocity_range"])}


@configclass
class Biped12HangStandW2eEnvCfg(Biped12HangStandW2dEnvCfg):
    """W2-5 (1004): W2-4 + **hip_r_deviation(−3.0, 힙 회전 관절 편차 벌점) 스텝 모드 OFF**(서기 모드 유지) 단일 변경.
    근거: 요 표류 3회(W2-1·2·3: 13–18°/8 s, std 0.1 요 추종에도 불변) — 교정에 필요한 힙 회전 관절 사용이 −3.0 으로 막힘(가설)."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.hip_r_deviation
        self.rewards.hip_r_deviation = RewTerm(func=mdp_hang.w1_stand_gated, weight=t.weight, params={"func": t.func, "params": dict(t.params or {})})


@configclass
class Biped12HangStandG24RaEnvCfg(Biped12HangStandW2eEnvCfg):
    """G24R-a (1004): W2-5 + **외란 분포 = G22R R3 앞 편향 고정**(무외란 30 %·앞 50 %·뒤 20 %, 앞 10–12 N·뒤 12–18 N, 충격 ×1.5, 무외란 범주 속도 밀기 없음) 단일 변경.
    근거: W2-4 에서 앞 회복 스텝이 자발 출현(sf15 스텝 86.7 %)했으나 하중 수용 62 %·낙상 74 % — 앞 밀기 연습량을 늘린다(좌우 외란 제외 → sl8 감시)."""

    def __post_init__(self):
        super().__post_init__()
        d = self.events.disturbance
        d.func = mdp_hang.disturbance_reset_g22r; d.params = dict(d.params, branch="g22r", stage_override=2)
        pr = self.events.push_robot
        pr.func = mdp_hang.push_with_arm_g22; pr.params = {"velocity_range": dict(pr.params["velocity_range"]), "arm_s": 2.5, "branch": "g22r"}


@configclass
class Biped12HangStandG24RaR4EnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-a-R4 (1006, RH2): G24R-a + **외란 단계 R3 → R4 고정**(앞 램프 12–15 N, 충격 ×1.5 = 18–22.5 N × 0.2–0.4 s ≈ 3.6–9 N·s, 뒤 18 N) 단일 변경.
    근거(검토자): 평가 sf15(15 N)·fi40/fs80(8 N·s)이 R3 학습 분포(앞 10–12 N, 3–7.2 N·s) 밖 — 이어학습 표류가 '분포 밖 영역 미구속'(H2) 탓인지 가른다."""

    def __post_init__(self):
        super().__post_init__()
        d = self.events.disturbance
        assert d.params.get("stage_override") == 2
        d.params = dict(d.params, stage_override=3)


@configclass
class Biped12HangStandG24RaD5EnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-a-D5 (1006): G24R-a + **행동 지연 0–3 → 1–5 틱** 단일 변경(이어 학습, 대조 = 같은 경로 G24R-a 0–3 이어 학습).
    근거: A-DEC(지연 1→4 sf15 낙상 14→41 %)·DSW(착지 앞·뒤 지연 모두 해, fi40 은 착지 뒤 지연 1→4 15→39 %) — 회복 정책은 0–3 만 봤다."""

    def __post_init__(self):
        super().__post_init__()
        a = self.actions.joint_pos
        assert a.min_delay_steps == 0 and a.max_delay_steps == 3
        a.min_delay_steps, a.max_delay_steps = 1, 5


@configclass
class Biped12HangStandW2fEnvCfg(Biped12HangStandW2eEnvCfg):
    """W2-6 (1004): W2-5 + **스텝 모드 요 각속도 명령 ±0.3 rad/s 무작위**(heading 제어 끔, 직접 표본) 단일 변경 — 회전 명령 추종 학습.
    근거: 요 표류(13–19°/8 s)가 보상 변경 4회에 불변, 평가 HeadingHold 명령에도 불응 — 학습 중 요 명령이 항상 0 이었다. 배포 브리지 HeadingHold(kp 1.0·±0.6)가 쓸 수 있게."""

    def __post_init__(self):
        super().__post_init__()
        c = self.commands.base_velocity
        c.ranges.ang_vel_z = (-0.3, 0.3); c.heading_command = False
        if hasattr(c, "rel_heading_envs"):
            c.rel_heading_envs = 0.0


@configclass
class Biped12HangStandG24RbEnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-b (1004): G24R-a + **조용한 서기 연습량 증가**(서기 명령 비율 0.5 → 0.7, 외란 무외란 범주 0.3 → 0.5; 앞:뒤 비 유지) 단일 개념 변경(Case E 처방).
    근거: G24R-a 앞 회복 대폭 개선(sf15 낙상 32 %) 대신 서기 흔들림 nd 7.1·d3nd 11.6 % — 피치 1–1.5 Hz 느린 흔들림, 클록 고정으로 7.1→5.4 % 만."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.7
        self.events.disturbance.params = dict(self.events.disturbance.params, p_none_override=0.5)


@configclass
class Biped12HangStandW2gEnvCfg(Biped12HangStandW2fEnvCfg):
    """W2-7 (1004): W2-6 + **스텝 모드 요 추종 std 0.1 → 0.5** 단일 변경(회전 명령 ±0.3 유지).
    근거(OSS 11, 계산): std 0.1 에서 회전 안 하는 정책의 오차 0.3 → 보상 1e-4·기울기 ≈ 0 — W2-6 track_ang_vel 0.09 평탄과 일치. 공개 구현 std 0.45–0.5."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.track_ang_vel_walk
        p = dict(t.params["params"]); p["std"] = 0.5
        t.params = {"func": t.params["func"], "params": p}


@configclass
class Biped12HangStandG24RcEnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-c (1004): G24R-a + **G22R 회복 구조** 단일 개념 변경 — 회복 게이트 recovery_gate_g22r(무장 래치·회복 방향·valid 이륙/착지/하중·SETTLE),
    구 회복 항(recovery_quality·recovery_event, 서기 게이트) 제거, action_rate_window RECOVERY 만(−0.06), 추가: vel_quality +1.5·ang_quality +1.0·
    step_direction +0.5·touchdown +5·load_accept +10·settle +25·recovery_time −0.10·unarmed_takeoff −10·swing_timeout −10.
    근거: G24R-a 의 남은 실패 = 앞 15 N 하중/착지 72 %·정착 1.36 s·낙상 32 % — 이 구조가 겨냥하는 항목. G22R 당시엔 스텝 능력이 없어 무효였음."""

    def __post_init__(self):
        super().__post_init__()
        ev, rw = self.events, self.rewards
        ev.recovery_gate.func = mdp_hang.recovery_gate_g22r; ev.recovery_gate.params = {}
        rw.recovery_quality = None
        rw.recovery_event = None
        arw = rw.action_rate_window
        inner = arw.params["params"]["func"] if arw.func is mdp_hang.w1_stand_gated else arw.params["func"]
        rw.action_rate_window = RewTerm(func=mdp_hang.g2_phase_gated, weight=-0.06, params={"func": inner, "params": {}, "phases": [1]})
        rw.recovery_vel_quality = RewTerm(func=mdp_hang.g22r_vel_quality, weight=1.5)
        rw.recovery_ang_quality = RewTerm(func=mdp_hang.g22r_ang_quality, weight=1.0)
        rw.recovery_step_direction = RewTerm(func=mdp_hang.g22r_step_direction, weight=0.5)
        rw.valid_touchdown_event = RewTerm(func=mdp_hang.g22r_event, weight=5.0, params={"name": "ev_td"})
        rw.load_accept_rec_event = RewTerm(func=mdp_hang.g22r_event, weight=10.0, params={"name": "ev_load"})
        rw.settle_success_event = RewTerm(func=mdp_hang.g22r_event, weight=25.0, params={"name": "ev_settle"})
        rw.recovery_time = RewTerm(func=mdp_hang.g2_recovery_time, weight=-0.10)
        rw.unarmed_takeoff_event = RewTerm(func=mdp_hang.g22r_event, weight=-10.0, params={"name": "ev_unarmed"})
        rw.swing_timeout_event = RewTerm(func=mdp_hang.g22r_event, weight=-10.0, params={"name": "ev_swto"})


@configclass
class Biped12HangStandW2hEnvCfg(Biped12HangStandW2gEnvCfg):
    """W2-8 (1004): W2-7 + **스텝 모드 모방 보상 3종 끔**(joint_pose_reference·foot_position_reference·contact_phase_reference) 단일 개념 변경.
    근거(구조 재진단, 요 실패 5회째): W2 계보는 A0m 부터 모방 배율 1.0 그대로 — 기준 동작은 직진 스텝(힙 회전 가중 1.0·σq 0.10 rad,
    스윙발 상대 궤적 σp 3 cm 횡 0) 이라 회전 보행 자세를 벌한다. W2-7 상수 요 명령 진단: ±0.3 rad/s 에 실제 회전 추종비 −0.03~0.24."""

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards
        rw.joint_pose_reference = None; rw.foot_position_reference = None; rw.contact_phase_reference = None


@configclass
class _W2iCurriculumCfg:
    turn_anneal = _CurrTerm(func=mdp_hang.turn_assist_curriculum)


@configclass
class Biped12HangStandW2iEnvCfg(Biped12HangStandW2hEnvCfg):
    """W2-9 (1004): W2-8 + **회전 부트스트랩 보조**(스윙 힙 회전 += α·0.455·wz, 지지 반주기 복귀, 6틱 앞당김, α 2.0→0 감쇠 — 블록 회전 추종비 ≥ 0.5) 단일 개념 변경.
    근거: 요 명령 전달 정상(cmd_probe) & 보상 수정 5회·모방 끔에도 회전 0 → 탐색 실패(STEP primitive 와 같은 양상, 같은 처방)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.turn_assist = EventTerm(func=mdp_hang.install_turn_assist, mode="startup", params={})
        self.curriculum = _W2iCurriculumCfg()


@configclass
class Biped12HangStandW2jEnvCfg(Biped12HangStandW2iEnvCfg):
    """W2-10 (1004): W2-9 + **요 추종 보상을 반주기 이동평균 ω_z 로 교체, std 0.25**(가중 1.5, 걷기 모드 게이트 동일) — 요 목표 1 묶음.
    std 선택(학습 전 오프라인 diag): 회전 이득 std 0.5 +0.0046 · std 0.25 +0.0088/틱(요 미끄럼+힙 편차 벌점 −0.0089 와 대등). std 변경은 W2-7 이후 2 회째(가중 조정 상한).
    근거(diag_turn_terms, W2-8 정책 + 보조 α 0/2): 회전(추종비 0.62)해도 순간 ω_z 추종 보상 이득 0(0.0181→0.0179) — 스텝형 회전은 순간 각속도가 출렁.
    반면 스텝 누락·지지발 요 미끄럼·힙 편차·이벤트 항이 회전을 벌함(합 −0.031/틱) → W2-9 학습 중 정책이 보조를 상쇄(추종비 0.49→0.32)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_ang_vel_walk = RewTerm(func=mdp_hang.track_yaw_rate_halfcycle, weight=1.5, params={"std": 0.25})


@configclass
class Biped12HangStandG24RdEnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-d (1004, C5 회복 정책 후보): G24R-a + **외란 단계 R3 → R4 고정**(무외란 25 %·앞 55 %·뒤 20 %, 앞 12–15 N·뒤 18 N, 충격 ×1.5) 단일 변경.
    근거(C5 기본 판정): 서기는 C5 에서 G21a 가 맡고 인계 불일치 근거 없음 → 남은 미달 sf12 12.0·sf15 31.8 % 는 회복 정책 성능. G24R-a 학습 앞 힘 10–12 N 은 평가 sf15 밖."""

    def __post_init__(self):
        super().__post_init__()
        self.events.disturbance.params = dict(self.events.disturbance.params, stage_override=3)


def _strip_rec_gate(term):
    """보상 항 래퍼 체인에서 회복 게이트 층(state_gated·rest_gated)만 제거(모드 게이트 w1_stand/walk_gated 등은 유지)."""
    f, p = term.func, dict(term.params or {})
    if f in (mdp_hang.state_gated, mdp_hang.rest_gated):
        inner = RewTerm(func=p["func"], weight=term.weight, params=dict(p.get("params") or {}))
        return _strip_rec_gate(inner)
    if "func" in p and isinstance(p.get("params", None), (dict, type(None))) and callable(p["func"]):
        inner = _strip_rec_gate(RewTerm(func=p["func"], weight=term.weight, params=dict(p.get("params") or {})))
        p["func"] = inner.func; p["params"] = dict(inner.params or {})
        return RewTerm(func=f, weight=term.weight, params=p)
    return term


@configclass
class Biped12HangStandG24ReEnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-e (1004, C5 회복 정책 후보): G24R-a + **feet_gap(발 간격 벌점)을 회복 게이트 열림 중에도 적용**(회복 게이트 층만 제거) 단일 변경.
    근거: C5 앞 밀기 낙상 대부분 앞+옆 대각(|roll|>|pitch| 63/89, fd2_c5b), OSS 13: 발 간격 항이 회복 중 꺼짐(state/rest_gated)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_gap = _strip_rec_gate(self.rewards.feet_gap)


@configclass
class Biped12HangStandG24RfEnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-f (1004, 병렬 분기): G24R-a + **학습 외란에 옆 램프·충격 섞기**(외란 env 의 30 %, 크기 ×0.6 — G21a 원설정 값) 단일 변경.
    근거: disturbance_reset_g22r 가 lateral_prob 0 고정(OSS 13), C5 앞 밀기 낙상의 롤 지배."""

    def __post_init__(self):
        super().__post_init__()
        self.events.disturbance.params = dict(self.events.disturbance.params, lateral_override=0.3)


@configclass
class Biped12HangStandG24RgEnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-g (1004 밤, C5 회복 정책 후보): G24R-a + **공통 힙 앞뒤 편향 DR 신전 꼬리 확장 −3~−1° → −4.5~−1°** 단일 변경.
    근거: DX3(인과 개입) 숨은 힙 신전 +1.5° 가 C5 앞 밀기 낙상 2–3배(sf12 10→27 %), 자동수평 트림으로 상쇄 안 됨(DX5·DX5f); OSS 16 순위 1."""

    def __post_init__(self):
        super().__post_init__()
        self.events.default_pose_bias.params["shared_hip_f_rad"] = (-0.0785, -0.018)


@configclass
class Biped12HangStandG24RhEnvCfg(Biped12HangStandG24RaEnvCfg):
    """G24R-h (1004 밤, 병렬 분기): G24R-a + **joint_deviation_hip(힙 외전·회전 편차 −0.5) 을 회복 게이트 열림 중 끔(state_gated)** 단일 변경.
    근거: C5 앞 밀기 낙상 = 롤 진동에서 회복 실패(dx2: 롤>8° 는 생존에도 흔함), OSS 15 순위 1 — 롤 회복 수단(힙 외전) 해방."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.joint_deviation_hip
        self.rewards.joint_deviation_hip = RewTerm(func=mdp_hang.state_gated, weight=t.weight, params={"func": t.func, "params": dict(t.params or {})})


@configclass
class _G25CurriculumCfg:
    imp_level = _CurrTerm(func=mdp_hang.imp_level_curriculum)


@configclass
class Biped12HangStandG25EnvCfg(Biped12HangStandG24RaEnvCfg):
    """G25 (1004 밤, C5 회복 정책 후보): G24R-a + **앞/뒤 충격 단계 커리큘럼**(충격량 4–8 → 14–18 N·s, 지속 0.1–0.3 s,
    현재 단계 50 %·숙달 30 %·원 분포 20 %, 성공 ≥ 85 % 승격·< 60 % 3 블록 강등) 단일 개념 변경.
    근거: C5 앞 충격 envelope 경계 6–8 N·s(학습 분포 끝), 실물 '콱 밀기' 낙상(1004)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.disturbance.params = dict(self.events.disturbance.params, imp_curr=True)
        self.curriculum = _G25CurriculumCfg()


@configclass
class Biped12HangStandG24RaSafeEnvCfg(Biped12HangStandG24RaEnvCfg):
    """T1 대조군(1004 밤): G24R-a 환경 그대로 — 안전 PPO(고정 lr 1e-4·엔트로피 0)로 이어 학습 표류만 측정."""


@configclass
class Biped12HangStandG25DEnvCfg(Biped12HangStandG24RaEnvCfg):
    """T3 / G25D (1005 새벽, C5 회복 정책 후보): G24R-a + **행동 지연 DR 0–3 → 0–5 틱** 단일 변경.
    근거: 지연 민감도(sf12 1/3/4/6 틱 5.5/15/21/43 %), 실물 루프 ≈ 시뮬 2–3.4 틱(상관·위상 추정), G16d(지연 0–6 학습 → 18 N 낙상 54 → 9–24 %, 당시 기각 사유 '서기 흔들림'은 C5 에서 G21a 몫), OSS 20 순위 1."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.max_delay_steps = 5


@configclass
class Biped12HangStandW2kEnvCfg(Biped12HangStandW2iEnvCfg):
    """WT1 (1005, 보행 트랙): W2-8 + **배포 측 요 제어를 학습 루프에**: HeadingHold(kp 1.0, 명령 wz 적분 목표) → 명령 wz → 회전 보조 훅 α 1.0 고정(커리큘럼 없음).
    근거(YAWA, 평가 전용): W2-8 + HeadingHold + 보조 α1 은 요 표류 14–18° → 5–9°(−57 %)·낙상 +1.3 %p, 옆 표류 +5 cm. 정책이 보조 잔차에 적응하면 낙상·옆 표류 회복 기대.
    배포 계약 변경 후보(보조 훅 = 브리지 잔차, 승인 대상). 대조 = W2h 같은 반복 이어 학습(WT0)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.turn_assist = EventTerm(func=mdp_hang.install_turn_assist, mode="startup", params={"alpha0": 1.0})
        self.events.heading_hold = EventTerm(func=mdp_hang.heading_hold_train, mode="interval", interval_range_s=(0.0, 0.0), params={"kp": 1.0})
        self.curriculum = None


@configclass
class Biped12HangStandW2hSafeEnvCfg(Biped12HangStandW2hEnvCfg):
    """WT0 대조(1005, 보행 트랙): W2-8 환경 그대로 — 안전 PPO 이어 학습 표류만 측정."""


@configclass
class Biped12HangStandG24RcueEnvCfg(Biped12HangStandG24RaEnvCfg):
    """R1 (1005, 회복 트랙): G24R-a + **배포 cue 규칙을 학습에**(서기 env 외력 중 기울기>3°·|v|>0.15·|ω|>0.5 & 전진 중 → 1 s 동안 vx 0.03, 확률 1 고정).
    근거(ACT2): 평가 전용 방향 조건 cue(bamd4)가 지연4 sf15 57→33 %. 배포와 같은 신호로 학습해 정책이 cue 를 쓰는 법을 익히게."""

    def __post_init__(self):
        super().__post_init__()
        self.events.recovery_hint = EventTerm(func=mdp_hang.recovery_step_hint, mode="interval", interval_range_s=(0.0, 0.0),
                                              params={"vx_hint": 0.03, "win_s": 1.0, "tilt_deg": 3.0, "v_min": 0.15, "w_min": 0.5, "fwd_only": True})
        # 1005 R1′: cue(힌트) 중 sb0 유효 착지 창 상한 10 → 35 cm (진단: 회복 생존 첫 걸음 22–28 cm, 종전 창이면 '놓친 스텝' 벌점)
        self.events.sb0.params = dict(self.events.sb0.params, disp_hi_rec=0.35)


@configclass
class Biped12HangStandG24RcapEnvCfg(Biped12HangStandG24RcueEnvCfg):
    """R2 (1005): R1 + **착지 캡처 여유 이벤트 보상**(가중 10 → 실효 ±0.2/착지; (발 − ξ)·v̂ / 5 cm).
    근거(ACT3 진단): 전방 밀기 낙상의 80–94 % 가 첫 착지 캡처 여유 < 0 — 걸음이 짧다(낙상 14 cm vs 생존 22–25 cm, 착지 시각 동일). 기존 swing_progress 는 8 cm 포화."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.capture_margin_event = RewTerm(func=mdp_hang.sb0_value, weight=10.0, params={"name": "ev_cap"})
        self.events.sb0.params = dict(self.events.sb0.params, cap_on=True)
        # 1005 정정: G24R-a 의 착지 이벤트는 sb0_step(스텝 모드) 출처 — 게이트(recovery_gate_step)에는 착지 계산이 없어 처음 R2 는 보상 0 이었다.


@configclass
class Biped12HangStandW2vEnvCfg(Biped12HangStandW2hEnvCfg):
    """WV1 (1005, 보행 트랙): W2-8 + **sb0 보폭 기준을 명령 속도에 비례**(진행 포화·유효 착지 창 = |vx|/(2·1.1 Hz) 기준, 종전 고정 5 cm·3–10 cm) 단일 구조 변경.
    근거(WVEL): W2-8 실제 전진속도 명령 0.05/0.10/0.15/0.20 → 0.089/0.094/0.096/0.097 — 속도 추종 0. 고정 보폭 기준이 한 가지 걸음만 정의한다(가설)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.sb0.params = dict(self.events.sb0.params, stride_scale=True)


@configclass
class Biped12HangStandW2v2EnvCfg(Biped12HangStandW2hEnvCfg):
    """WV2 (1005, 보행 트랙): W2-8 + **sb0 보폭 상한만 명령 비례로 올림** — prog_sat = max(5 cm, |vx|/2.2), 착지 창 상한 = max(10 cm, 1.8·|vx|/2.2). 하한·이벤트·클록 불변.
    근거(라운드 24): 속도 고정은 보폭 상한(10 cm 착지 창·5 cm 진행 포화) + 반주기 1 걸음 이벤트 구조; WV1 은 창을 작은 명령 쪽으로도 줄여 붕괴."""

    def __post_init__(self):
        super().__post_init__()
        self.events.sb0.params = dict(self.events.sb0.params, stride_scale="up")


@configclass
class Biped12HangStandW2wEnvCfg(Biped12HangStandW2hEnvCfg):
    """W2w (1005, 보행 트랙): W2-8 + **sb0 유효 착지 창 상한 10 → 35 cm (항상)** 단일 구조 변경. 진행 포화·놓친 스텝·클록 불변.
    근거(WPUSH·WVEL): 걷기 정책이 보폭을 못 바꿈 — 속도 추종 0, 앞 램프 8 N 에도 25–50 % 낙상. 10 cm 넘는 걸음 = 착지 이벤트 상실 + 놓친 스텝 벌점."""

    def __post_init__(self):
        super().__post_init__()
        self.events.sb0.params = dict(self.events.sb0.params, disp_hi=0.35)


@configclass
class Biped12HangStandW2sEnvCfg(Biped12HangStandW2hEnvCfg):
    """WV3 (1005, 보행 트랙): W2-8 + **걷기 속도 추종 std 0.1 → 0.03**(가중 3 불변) 단일 변경 — 속도 추종 형태 첫 조정.
    근거: 실제 속도 명령 무관 0.09 m/s, 보폭 상한은 비구속(보폭 ~4 cm < 창 10 cm·포화 5 cm, 검토 13회차). std 0.1 에서 오차 0.05 m/s 도 보상 78 % → 속도 구분 기울기 약함."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.track_lin_vel_walk
        p = dict(t.params["params"]); p["std"] = 0.03
        t.params = {"func": t.params["func"], "params": p}


_SB0_WALK_TERMS = ("missed_step_event", "premature_swing", "swing_progress", "swing_clearance", "swing_unload",
                   "touchdown_event", "settle_event", "load_accept_event", "load_transfer")


@configclass
class Biped12HangStandW2oEnvCfg(Biped12HangStandW2hEnvCfg):
    """W2o (1005, 보행 트랙): W2-8 + **걷기 모드 보상 세트를 sb0 사건형 → OSS 표준형으로 교체** — sb0 사건·스윙 항(착지·하중·정착·놓친 스텝·조기 스윙·스윙 3종) 제거,
    이족 공중시간 보상(feet_air_time_positive_biped 식, 문턱 0.4 s, 가중 1.0, 걷기 모드 게이트) 추가. 속도 추종(std 0.1, 가중 3)·서기 모드 보상·관측·클록 불변.
    근거(라운드 25): 반주기당 사건 보상(+5/+10/+15/−20)이 추종 이득(0.018/틱)보다 커서 고정 보폭 국소해 — WV1–3·W2w 400 반복 이어 학습 전부 실패. OSS(humanoid-gym·unitree)는 고정 주기 클록으로도 사건 보상 없이 속도 추종."""

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards
        for n in _SB0_WALK_TERMS:
            if hasattr(rw, n):
                setattr(rw, n, None)
        rw.feet_air_time_walk = RewTerm(func=mdp_hang.feet_air_time_biped_walk, weight=1.0, params={"threshold": 0.4})



@configclass
class Biped12HangStandW2qEnvCfg(Biped12HangStandW2hEnvCfg):
    """W2q (1005, 보행 트랙): W2-8 + **명령 연동 cadence** — 보행 위상을 f(vx) = clamp(0.7 + 4·|vx|, 0.8, 1.6) Hz 로 적분(걷기 모드, 서기 1.1 Hz).
    관측 gait_clock·sb0 일정·보조 훅이 같은 위상을 공유. 사건 보상(sb0) 유지 → 고정 보폭이라도 cadence 로 속도가 변한다(보폭 4 cm 가정 0.07–0.12 m/s).
    근거: WV1–3·W2o — 보상만으로는 고정 보폭 국소해(사건 보상) 또는 정지(사건 보상 제거). **배포 계약 변경**(브리지 클록 위상 적분)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.gait_phase = EventTerm(func=mdp_hang.gait_phase_update, mode="interval", interval_range_s=(0.0, 0.0), params={})


@configclass
class Biped12HangStandW2qeEnvCfg(Biped12HangStandW2qEnvCfg):
    """W2q 평가용: + 회전 보조 훅(TURN_ALPHA 로 α, 커리큘럼 없음) — W2i 와 같은 배포 조건."""

    def __post_init__(self):
        super().__post_init__()
        self.events.turn_assist = EventTerm(func=mdp_hang.install_turn_assist, mode="startup", params={})
        self.curriculum = None



@configclass
class Biped12HangStandW2omEnvCfg(Biped12HangStandW2qEnvCfg):
    """W2om (1006, 보행 구조 challenger): W2q + **전방향 걷기 모드 게이트**(_walk_mode: |vx|·|vy| ≥ 0.03 또는 |wz| ≥ 0.1) +
    sb0 변위 축 = 명령 속도 방향(회전 전용은 방향 무관 ≥ 0.5 cm) + cadence 유효 속도 max(‖v‖, 0.25|wz|) + 명령 범주 혼합(앞 35·뒤 15·옆 20·회전 20·앞+회전 10 %).
    보상 항·가중은 W2q 그대로(게이트만 바뀜). 근거: 전방향 기준선 — 뒤·옆·회전 명령은 |vx| 게이트로 '서기 모드' 채점(서기 보상 + 스텝 사건 0) → 걸음 0;
    WOM(보상 교체 이어 학습)·WOMS(처음부터) 모두 걸음 미획득. **배포 계약 변경**: 브리지 cadence·모드 식도 같이 바뀌어야 함."""

    def __post_init__(self):
        super().__post_init__()
        self.events.omni = EventTerm(func=mdp_hang.install_omni_cmd, mode="startup", params={"wz_min": 0.1})


@configclass
class Biped12HangStandW2omAEnvCfg(Biped12HangStandW2omEnvCfg):
    """W2omA (1006): W2om + **뒤·옆·회전 범주에만 스텝 보조**(굽힘 펄스 기준, α 1.0 → 0.75 → 0.5 → 0.25 → 0, 그 범주 하중 수용 ≥ 0.5 로 감쇠) 단일 변경.
    근거: W2om +510 — 게이트·방향 사건을 열어도 새 범주에서 걸음 시작 자체가 없음(cadence 0.00–0.01) = 탐색 문제. 앞 걸음도 처음엔 이 보조(A0h)로 부트스트랩했다."""

    def __post_init__(self):
        super().__post_init__()
        self.events.omni_assist = EventTerm(func=mdp_hang.omni_assist_update, mode="interval", interval_range_s=(0.0, 0.0), params={})


@configclass
class Biped12HangStandW2omLEnvCfg(Biped12HangStandW2omAEnvCfg):
    """W2omL (1006, 옆걸음 기술): W2omA + **옆 방향 보조**(hip_a 잔차 ky 2 rad/(m/s), DIRCAL 채택 부호 B = raw (+,+)) + 굽힘 펄스, 둘 다 옆 범주만·같은 α 감쇠
    + **명령 범주 = 앞 60 %·옆 40 %**(풀 수 없는 뒤·회전 제외 — W2om 대조의 전방향 붕괴 방지).
    근거: DIRCAL — 옆은 기준 궤적으로 방향이 생김(+0.08→+0.058, −0.08→−0.096), 회전 12–14 %·뒤 0 은 기각."""

    def __post_init__(self):
        super().__post_init__()
        self.events.omni.params = {"wz_min": 0.1, "mix": (("fwd", 0.6), ("back", 0.0), ("lat", 0.4), ("turn", 0.0), ("fwd_turn", 0.0))}
        self.events.dir_assist = EventTerm(func=mdp_hang.install_dir_assist, mode="startup", params={"ky": -2.0})
        self.events.omni_assist.params = {"cats": (2,), "dir_too": True}


@configclass
class Biped12HangStandW2omMEnvCfg(Biped12HangStandW2omAEnvCfg):
    """W2omM (1006, 제자리 회전 기술): W2omA + ① 회전 범주 제자리 착지 사건(td_any_dir 하한 2 → 0.5 cm, omni_inplace) ② 명령 = 앞 50·제자리 회전 50 %
    ③ 회전 피드포워드 α 1.5 상시(배포 제어기 일부, omni 게이트로 제자리 회전에도) ④ 굽힘 보조는 회전 범주만(α 감쇠).
    근거: SRTURN — 회전은 앞 걸음에 묶임(vx 0.03 걸음 없음); W2omA α1 진단에서 회전 범주 걸음 0.45–0.49 Hz 는 나왔으나 사건 하한 2 cm 가 제자리 걸음을 보상하지 않았다."""

    def __post_init__(self):
        super().__post_init__()
        self.events.omni.params = {"wz_min": 0.1, "mix": (("fwd", 0.5), ("back", 0.0), ("lat", 0.0), ("turn", 0.5), ("fwd_turn", 0.0))}
        self.events.omni_assist.params = {"cats": (3,)}
        self.events.sb0.params = dict(self.events.sb0.params, omni_inplace=True)
        self.events.turn_assist = EventTerm(func=mdp_hang.install_turn_assist, mode="startup", params={"alpha0": 1.5})


@configclass
class Biped12HangStandW2omPEnvCfg(Biped12HangStandW2omAEnvCfg):
    """W2omP (1006, 제자리 걸음(march) 1단계 — cadence 만): W2omA + ① 회전 범주 제자리 착지 사건(omni_inplace, ≥ 0.5 cm) ② 명령 = 앞 50·제자리(회전 범주, wz 는 걷기 모드 신호) 50 %
    ③ 굽힘 보조는 제자리 범주만(α 감쇠) — **피드포워드 없음**(W2omL·W2omM 의 걸음 억제 원인 후보 제외). 초기화는 STEP 원형 A0m 14299(앞 편향 약함).
    근거: A0MT — 걸음 촉발 = vx 명령 관측(제자리 명령에 cadence 0); W2omA(사건 ≥ 2 cm)는 0.17 평탄, W2omM(피드포워드 추가)는 억제."""

    def __post_init__(self):
        super().__post_init__()
        self.events.omni.params = {"wz_min": 0.1, "mix": (("fwd", 0.5), ("back", 0.0), ("lat", 0.0), ("turn", 0.5), ("fwd_turn", 0.0))}
        self.events.omni_assist.params = {"cats": (3,)}
        self.events.sb0.params = dict(self.events.sb0.params, omni_inplace=True)


@configclass
class Biped12HangStandW2omP2EnvCfg(Biped12HangStandW2omPEnvCfg):
    """W2omP2 (1006): W2omP 이어학습 — 보조 α 단계를 이어받음(start_stage 2 = α 0.5; 프로세스 재시작 시 커리큘럼 초기화 함정 방지) 외 동일."""

    def __post_init__(self):
        super().__post_init__()
        self.events.omni_assist.params = {"cats": (3,), "start_stage": 2}


@configclass
class Biped12HangStandW2omP3EnvCfg(Biped12HangStandW2omPEnvCfg):
    """W2omP3 (1006): W2omP + **보조 감쇠 규칙 = 상대 미세 감쇠**(A0m 이 α 0 에 도달한 규칙: 0.1 단계, 그 α 첫 블록의 70 % 유지 시 감쇠, 40 % 미만 되돌림) 단일 변경, α 0.5 에서 시작.
    근거: W2omP·W2omP2·W2omP-s2 모두 α 0.5 에서 하중 수용 0.37–0.43 평탄 — 절대 문턱 0.5 가 내재화를 막는다(가설)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.omni_assist.params = {"cats": (3,), "rel": True, "rel_alpha0": 0.5}


@configclass
class Biped12HangStandW2omIEnvCfg(Biped12HangStandW2omPEnvCfg):
    """W2omI (1006, 제자리 걸음 — 정책이 직접 동작을 만들게): W2omP 에서 **굽힘 보조 제거(α 0 고정)** + **제자리 범주에만 모방 보상 3종**
    (joint_pose_reference +1.5 · foot_position_reference +2.0(vx 0 → 제자리 목표) · contact_phase_reference +2.0, STEP-I0 가중) 단일 개념 변경.
    근거: W2omP3 — 보조 하 하중 수용이 α 에 비례(보조 산물, 정책 미내재화). 모방 보상은 정책 자신의 관절 궤적을 채점한다(A0m 학습 때 사용)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.omni_assist = None
        rw = self.rewards
        for name, wgt, key in (("joint_pose_reference", 1.5, "r_qref"), ("foot_position_reference", 2.0, "r_foot"), ("contact_phase_reference", 2.0, "r_cref")):
            setattr(rw, name, RewTerm(func=mdp_hang.omni_cat_gated, weight=wgt, params={"func": mdp_hang.sb0_value, "params": {"name": key}, "cats": (3,)}))


@configclass
class Biped12HangStandW2omR1EnvCfg(Biped12HangStandW2omP3EnvCfg):
    """W2omR1 (1006, 진단자 원장 가설): W2omP3 + **track_lin_vel_walk 를 병진 명령(‖v_xy‖ ≥ 0.03)에만** 단일 변경.
    근거: 제자리 범주(vx 0)에서 선속도 추종 +3·std 0.1 은 서기에 ≈ +2.9/s, 걸음 흔들림에 벌 — 서기 대비 걸음 이득 +30–45 %(A0m 스텝 모드 +130 %)."""

    def __post_init__(self):
        super().__post_init__()
        t = self.rewards.track_lin_vel_walk
        self.rewards.track_lin_vel_walk = RewTerm(func=mdp_hang.w1_translate_gated, weight=t.weight, params=dict(t.params))


@configclass
class Biped12HangStandW2omR2EnvCfg(Biped12HangStandW2omR1EnvCfg):
    """W2omR2 (1006): W2omR1 + **제자리 범주 외란 끔**(지속·충격 힘 + 속도 밀기) — A0m 스텝 모드는 외란 없음."""

    def __post_init__(self):
        super().__post_init__()
        self.events.dist_off = EventTerm(func=mdp_hang.dist_off_cats, mode="interval", interval_range_s=(0.0, 0.0), params={"cats": (3,)})
        pr = self.events.push_robot
        self.events.push_robot = EventTerm(func=mdp_hang.push_filtered, mode=pr.mode, interval_range_s=pr.interval_range_s, params={"velocity_range": dict(pr.params["velocity_range"])})


@configclass
class Biped12HangStandW2omFEnvCfg(Biped12HangStandW2omR2EnvCfg):
    """W2omF (1006, 관측 걷기 플래그): W2omR2 + **서기 모드 gait 클록 정지(위상 0 고정)** 단일 변경 — 클록이 도는 것 = 걷기 모드.
    근거(진단자): 클록은 서기에서도 돌아 걷기 모드 구분 신호가 명령 벡터뿐; R1·R2 원장 수정은 α 0.2 하중 유지를 4배 개선했으나 내재화 실패. **배포 계약 변경**."""

    def __post_init__(self):
        super().__post_init__()
        self.events.gait_phase.params = dict(self.events.gait_phase.params, freeze_stand=True)


@configclass
class Biped12HangStandW2omeEnvCfg(Biped12HangStandW2omEnvCfg):
    """W2om 평가용: + 회전 보조 훅(TURN_ALPHA, 0 이면 무동작) — W2qe 와 같은 평가 조건."""

    def __post_init__(self):
        super().__post_init__()
        self.events.omni = EventTerm(func=mdp_hang.set_omni, mode="startup", params={"wz_min": 0.1})   # 평가: 게이트만(명령 혼합 없음 — --cmd_set 보존)
        self.events.turn_assist = EventTerm(func=mdp_hang.install_turn_assist, mode="startup", params={})
        self.curriculum = None




@configclass
class Biped12HangStandW2omFeEnvCfg(Biped12HangStandW2omeEnvCfg):
    """W2omF 평가용: W2ome + 서기 클록 정지(같은 관측 계약)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.gait_phase.params = dict(self.events.gait_phase.params, freeze_stand=True)

@configclass
class Biped12HangStandW2omdEnvCfg(Biped12HangStandW2omeEnvCfg):
    """W2om 방향 보조 진단 평가: + install_dir_assist(기본 0 — DIR_KX/KY/KZ env var 로 크기·부호 실측), STEP_ASSIST_ALPHA 로 굽힘 펄스 병용."""

    def __post_init__(self):
        super().__post_init__()
        self.events.dir_assist = EventTerm(func=mdp_hang.install_dir_assist, mode="startup", params={})

@configclass
class Biped12HangStandW2qSEnvCfg(Biped12HangStandW2qEnvCfg):
    """WSTEER (1006, 요 조향 기술 — 보행 검토자 제안): W2q + 구조 묶음
    ① 회전 피드포워드 α 1.0 상시(install_turn_assist, 감쇠 없음 — 배포 제어기 일부) ② joint_deviation_hip 기준점 = 기본자세 + 피드포워드 잔차
    ③ 걷기 요 추종 = 반주기 평균(track_yaw_rate_halfcycle, std 0.25, 가중 1.5) ④ 명령: vx U(0.05,0.15), wz 50 % ±U(0.15,0.35)·50 % 0, vy 0.
    근거 STEER(정책 요 무시, 피드포워드 인과 28–59 %) + W2-10 decision(회전 이득 ≈ 정자세 기준 손해). 대조 C = W2q 같은 반복 이어학습."""

    def __post_init__(self):
        super().__post_init__()
        self.events.turn_assist = EventTerm(func=mdp_hang.install_turn_assist, mode="startup", params={"alpha0": 1.0})
        self.events.steer_cmd = EventTerm(func=mdp_hang.install_steer_cmd, mode="startup", params={})
        self.curriculum = None
        t = self.rewards.joint_deviation_hip
        self.rewards.joint_deviation_hip = RewTerm(func=mdp_hang.joint_deviation_l1_ff, weight=t.weight, params=dict(t.params))
        self.rewards.track_ang_vel_walk = RewTerm(func=mdp_hang.track_yaw_rate_halfcycle, weight=1.5, params={"std": 0.25})


@configclass
class Biped12HangStandW2q2EnvCfg(Biped12HangStandW2qEnvCfg):
    """W2q2 (1005, 보행 트랙): W2q + **sb0 보폭 기준 = 스윙당 발 이동 |vx|/f(vx)**(0.05→5.6, 0.10→9.1, 0.15→11.5, 0.20→13.3 cm): 진행 포화 = 기준, 착지 창 상한 = max(10 cm, 1.4×기준) — 보폭 기준 한 개념.
    근거(W2q 300 반복 단계 분해): 명령→클록→실제 cadence 는 전달(0.80→1.33 Hz)되나 보폭이 cadence 에 반비례로 줄어 실제 vx 0.061→0.084 고정. 진행 포화 5 cm 고정이 걸음당 보폭 신호를 주지 않는다."""

    def __post_init__(self):
        super().__post_init__()
        self.events.sb0.params = dict(self.events.sb0.params, stride_scale="cad")


@configclass
class Biped12HangStandW2q2eEnvCfg(Biped12HangStandW2q2EnvCfg):
    """W2q2 평가용 + 회전 보조 훅(TURN_ALPHA)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.turn_assist = EventTerm(func=mdp_hang.install_turn_assist, mode="startup", params={})
        self.curriculum = None



@configclass
class Biped12HangStandW2rEnvCfg(Biped12HangStandW2qEnvCfg):
    """WR (1005, 보행 트랙, 마감 전 마지막 구조 대안): W2q + **Raibert형 몸 기준 착지 목표**(walk-these-ways 형) — 진행 보상(swing_progress, 이륙 기준 5 cm 포화)을 착지 사건 보상으로 교체:
    착지 스윙발의 골반 기준 앞축 위치 x_l 이 x_t = v_cmd/(4f) + 0.24·(v − v_cmd) 에 가까울수록 exp(−((x_l−x_t)/3 cm)²), 가중 5(착지 사건과 같은 크기).
    근거(W2q·W2q2 단계 분해): 명령→cadence 는 전달, 스윙당 이동이 cadence 에 반비례(보폭 단계 끊김). 이륙 기준 보폭 기준(W2q2)은 무효 → 몸 기준·속도 피드백 착지 목표."""

    def __post_init__(self):
        super().__post_init__()
        self.events.sb0.params = dict(self.events.sb0.params, raibert=True)
        self.rewards.swing_progress = None
        self.rewards.raibert_land = RewTerm(func=mdp_hang.sb0_value, weight=5.0, params={"name": "ev_rai"})


@configclass
class Biped12HangStandW2reEnvCfg(Biped12HangStandW2rEnvCfg):
    """WR 평가용 + 회전 보조 훅(TURN_ALPHA)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.turn_assist = EventTerm(func=mdp_hang.install_turn_assist, mode="startup", params={})
        self.curriculum = None



@configclass
class Biped12HangStandW2gsEnvCfg(Biped12HangStandW2qEnvCfg):
    """W2gs (1005 밤, 보행 challenger): W2q(명령 연동 cadence) + **걷기 모드 sb0 사건·스윙 보상 9종 → 클록 접촉 일정 보상**(gait_schedule_walk, 가중 2).
    근거: 사건 보상의 고정 착지 창(2–10 cm)·놓친 스텝 벌점이 고정 보폭 국소해(W2q·W2q2·WR), 사건 제거만 하면 정지(W2o). 일정은 걸음을 강제하되 보폭은 자유."""

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards
        for n in _SB0_WALK_TERMS:
            if hasattr(rw, n):
                setattr(rw, n, None)
        rw.gait_schedule = RewTerm(func=mdp_hang.gait_schedule_walk, weight=2.0, params={})


@configclass
class Biped12HangStandW2gseEnvCfg(Biped12HangStandW2gsEnvCfg):
    """W2gs 평가용 + 회전 보조 훅(TURN_ALPHA)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.turn_assist = EventTerm(func=mdp_hang.install_turn_assist, mode="startup", params={})
        self.curriculum = None



@configclass
class Biped12HangStandWOMEnvCfg(Biped12HangStandW2qEnvCfg):
    """WOM (1005 밤, 보행 구조 challenger): W2q 의 DR·관측·cadence 클록은 그대로, **보상 전체를 OSS 형 전방향 세트로 교체** + **명령 노름 게이트**.
    명령: vx −0.15~0.2, vy ±0.1, yaw ±0.5 rad/s, 서기 10 %(서기 자체는 supervisor G21a 담당), 4–6 s 리샘플.
    과제: 선속도 추종(요 프레임, std 0.1)×2, 요 추종(std 0.25)×1.5, 공중시간(문턱 0.3)×1, 양발 지지 클록 일정×1. 정규화: 기울기·z 속도·롤피치 각속도·행동 변화·토크·관절 가속·관절 한계·힙 편차·발 미끄럼·종료.
    근거(서브에이전트 1·5 + 데이터): sb0 계보는 |vx| 게이트라 제자리 회전 = 서기 모드로 스텝이 벌받고, 사건 보상이 보폭을 묶는다(WV1–3·W2q2·WR)."""

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards
        keep = {"termination_penalty"}
        for n in [k for k in vars(rw) if not k.startswith("_")]:
            if n not in keep and getattr(rw, n) is not None:
                setattr(rw, n, None)
        from isaaclab.envs import mdp as _imdp
        rw.track_lin = RewTerm(func=mdp_hang.track_lin_vel_xy_yaw_exp, weight=2.0, params={"std": 0.1})
        rw.track_yaw = RewTerm(func=mdp_hang.track_yaw_rate_exp, weight=1.5, params={"std": 0.25})
        rw.air_time = RewTerm(func=mdp_hang.feet_air_time_moving, weight=1.0, params={"threshold": 0.3})
        rw.schedule = RewTerm(func=mdp_hang.gait_schedule_ds, weight=1.0, params={})
        rw.flat_orient = RewTerm(func=_imdp.flat_orientation_l2, weight=-2.0)
        rw.lin_vel_z = RewTerm(func=_imdp.lin_vel_z_l2, weight=-0.5)
        rw.ang_vel_xy = RewTerm(func=_imdp.ang_vel_xy_l2, weight=-0.05)
        rw.act_rate = RewTerm(func=_imdp.action_rate_l2, weight=-0.2)
        rw.torques = RewTerm(func=_imdp.joint_torques_l2, weight=-1.0e-5)
        rw.acc = RewTerm(func=_imdp.joint_acc_l2, weight=-2.5e-7)
        rw.pos_lim = RewTerm(func=_imdp.joint_pos_limits, weight=-1.0)
        rw.hip_dev = RewTerm(func=_imdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_a_joint", ".*_hip_r_joint"])})
        rw.ankle_r_dev = RewTerm(func=_imdp.joint_deviation_l1, weight=-0.3, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_r_joint"])})
        c = self.commands.base_velocity
        c.ranges.lin_vel_x = (-0.15, 0.2); c.ranges.lin_vel_y = (-0.1, 0.1); c.ranges.ang_vel_z = (-0.5, 0.5)
        c.heading_command = False; c.rel_standing_envs = 0.1
        if hasattr(c, "rel_heading_envs"):
            c.rel_heading_envs = 0.0
        self.curriculum = None


class Biped12HangStandWOMXEnvCfg(Biped12HangStandWOMEnvCfg):
    """WOMX (1006, WOMS KILL 뒤 구조 수정 1개): 스케줄 항을 부호형(l_ok + r_ok − 1)으로 — 서기 0.6 → 0.2.
    WOMS 1000 반복: 전 명령 제자리 서기(cadence 0, 공중시간 0.0006) = 서기가 스케줄 보상 60 % 를 받는 국소 최적."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.schedule.params = {"signed": True}


# ================================================================ W3 (1006) — 명령 조건 보행 세대, Stage 0 = 제자리 교대 스텝 엔진 ================================================================
# 설계: reports/gen_w3/W3_DESIGN.md · 코어: rl_walking/w3.py. G21a 서기 정책 환경에서 출발(A0m 성공 조건), 외란 없음(스킬 획득 → 강건성 순서).
from rl_walking import w3 as _w3  # noqa: E402

_W3_REF = "/home/ryu/humanoid_leg_test1/reports/gen_step/ref_step_primitive_flex.npz"
# 무게 이동(골반 +1 m 왼쪽) 관절 잔차 [rad/m] — URDF FK/IK(기하 에이전트, 크라우치 자세): hip_a 양쪽 +0.96°/cm, hip_r +0.40°/cm, ankle_r 왼 +1.04°/오 −1.04°/cm
_W3_SHIFT = {"left_hip_a_joint": 1.676, "right_hip_a_joint": 1.676, "left_hip_r_joint": 0.698, "right_hip_r_joint": 0.698,
             "left_ankle_r_joint": 1.815, "right_ankle_r_joint": -1.815}


@configclass
class Biped12HangStandW3BaseEnvCfg(Biped12HangStandG21aEnvCfg):
    """W3 공통: gait_enable(STAND 30 % / STEP0 70 %)·위상(1.1 Hz)·스텝 추적기·STEP 보상. 기존(G21a) 보상은 STAND(ge 0)에만.
    STEP 보상 예산(초당): 생존 +0.5 · 주기 접촉 −1.5×(스윙 하중/mg + 지지발 속도) · 스윙 높이 +2 · 유효 스텝 +15/사건 · 착지 목표 +10/사건
    · 놓친 반주기 −10/사건 · 예정 밖 유효 스텝 −5 · 표류 −1·|v| · 기울기 −2 · ω_xy −0.05 · 행동 변화 −0.05. 종료 −75(그대로)."""

    p_step: float = 0.7

    def __post_init__(self):
        super().__post_init__()
        ev = self.events
        for k in ("disturbance", "push_robot", "recovery_gate"):
            if getattr(ev, k, None) is not None:
                setattr(ev, k, None)
        ev.w3_cmd = EventTerm(func=_w3.install_w3_cmd, mode="startup", params={"p_step": self.p_step, "stage": 0})
        ev.w3_gait = EventTerm(func=_w3.w3_gait_update, mode="interval", interval_range_s=(0.0, 0.0), params={"f_step": 1.1})
        ev.w3_ref = EventTerm(func=_w3.install_w3_ref, mode="startup", params={"ref_path": _W3_REF, "ref_scale": 0.0})
        ev.w3_track = EventTerm(func=_w3.w3_step_tracker, mode="interval", interval_range_s=(0.0, 0.0), params={})
        rw = self.rewards
        for n in [k for k in vars(rw) if not k.startswith("_")]:
            t = getattr(rw, n)
            if t is None or n == "termination_penalty":
                continue
            setattr(rw, n, RewTerm(func=_w3.w3_stand_gated, weight=t.weight, params={"func": t.func, "params": dict(t.params or {})}))
        from isaaclab.envs import mdp as _imdp
        rw.w3_alive = RewTerm(func=_w3.w3_alive_step, weight=0.5)
        rw.w3_periodic = RewTerm(func=_w3.w3_periodic_contact, weight=-1.5)
        rw.w3_height = RewTerm(func=_w3.w3_swing_height, weight=3.0)   # 1006 단위시험 2차 뒤 2→3(밀집 기울기 강화, 학습 전)
        rw.w3_valid = RewTerm(func=_w3.w3_value, weight=15.0, params={"name": "ev_valid"})
        rw.w3_place = RewTerm(func=_w3.w3_place_reward, weight=10.0)
        rw.w3_miss = RewTerm(func=_w3.w3_value, weight=-10.0, params={"name": "ev_miss"})
        rw.w3_wrong = RewTerm(func=_w3.w3_value, weight=-5.0, params={"name": "ev_wrong"})
        rw.w3_drift = RewTerm(func=_w3.w3_drift, weight=-1.0)
        rw.w3_orient = RewTerm(func=_w3.w3_step_gated, weight=-2.0, params={"func": _imdp.flat_orientation_l2, "params": {}})
        rw.w3_angvel = RewTerm(func=_w3.w3_step_gated, weight=-0.05, params={"func": _imdp.ang_vel_xy_l2, "params": {}})
        rw.w3_arate = RewTerm(func=_w3.w3_step_gated, weight=-0.05, params={"func": _imdp.action_rate_l2, "params": {}})
        self.curriculum = None


@configclass
class _W3ACurriculumCfg:
    w3_assist = _CurrTerm(func=_w3.w3_assist_curriculum)


@configclass
class Biped12HangStandW3AEnvCfg(Biped12HangStandW3BaseEnvCfg):
    """W3-A: A0m 레시피 — 굽힘 보조 scaffold(α 1 → 0, 절대 문턱 유효 스텝률 ≥ 0.25) + 모방(스윙발 굽힘 관절·창 가운데) +1.5. 참조 0."""

    def __post_init__(self):
        super().__post_init__()
        self.events.w3_ref.params = {"ref_path": _W3_REF, "ref_scale": 0.0, "assist_alpha0": 1.0}
        self.rewards.w3_imit = RewTerm(func=_w3.w3_imitation, weight=1.5, params={"ref_path": _W3_REF})
        self.curriculum = _W3ACurriculumCfg()


@configclass
class Biped12HangStandW3CEnvCfg(Biped12HangStandW3BaseEnvCfg):
    """W3-C: 참조 잔차 상시(제어기 일부) — 굽힘 펄스 + 관상면 무게 이동 ±4 cm(y = −0.04·cos 2πu, IK 계수). 보조 0, 모방 없음."""

    def __post_init__(self):
        super().__post_init__()
        self.events.w3_ref.params = {"ref_path": _W3_REF, "ref_scale": 1.0, "shift_amp": 0.04, "shift_lead": 0.25, "shift_joints": dict(_W3_SHIFT)}


@configclass
class Biped12HangStandW3UEnvCfg(Biped12HangStandW3BaseEnvCfg):
    """W3-U: OSS형 — 모드 플래그 없음(항상 STEP, 클록 상시), 참조·보조·모방 없음, 보상만. STAND 는 G21a 로 supervisor 전환."""

    p_step: float = 1.0


@configclass
class _W3CICurriculumCfg:
    w3_ref = _CurrTerm(func=_w3.w3_ref_curriculum)


@configclass
class Biped12HangStandW3CIEnvCfg(Biped12HangStandW3CEnvCfg):
    """W3-CI (1006): W3-C 이어학습 + ① 참조 내재화 — 참조 배율 1 → 0.75 → 0.5 → 0.25 → 0(A0m 절대 문턱: 유효 스텝률 ≥ 0.6 회복 뒤, 체류 ≥ 60) + 모방(스윙발 굽힘·창 가운데, 참조 = 굽힘 펄스)
    ② 요 표류 벌점 −0.5·|ω_z − wz_cmd|(빠진 항 보완). 목표: 참조 0 평가에서도 제자리 교대(사용자 Stage 0 기준)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.w3_imit = RewTerm(func=_w3.w3_imitation, weight=1.5, params={"ref_path": _W3_REF})
        self.rewards.w3_yaw = RewTerm(func=_w3.w3_yaw_drift, weight=-0.5)
        self.curriculum = _W3CICurriculumCfg()


@configclass
class Biped12HangStandW3A2EnvCfg(Biped12HangStandW3AEnvCfg):
    """W3-A2 (1006): W3-A 400 이어학습 — 보조 감쇠 이어받기(start_stage 3 = α 0.25 → 0, 절대 문턱) + α 0 장기 체류(A0m 조건) + 요 표류 벌점 −0.5(빠진 항)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.w3_yaw = RewTerm(func=_w3.w3_yaw_drift, weight=-0.5)
        self.curriculum.w3_assist.params = {"start_stage": 3}


@configclass
class Biped12HangStandW3A3EnvCfg(Biped12HangStandW3A2EnvCfg):
    """W3-A3 (1006, W3-A2 12200 이어): 표류 0.060 m/s(뒤)·요 65° 미해결 — 순간값 벌점은 걸음 흔들림이 지배.
    w3_yaw −0.5(순간 |ω_z|) → w3_yaw_net −4(한 주기 순 요 속도), w3_drift −1(EMA) → w3_drift_net −3(한 주기 이동평균). 나머지·보조 0 동일."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.w3_yaw = RewTerm(func=_w3.w3_yaw_net, weight=-4.0)
        self.rewards.w3_drift = RewTerm(func=_w3.w3_drift_net, weight=-3.0)
        self.curriculum.w3_assist.params = {"start_stage": 4}


@configclass
class Biped12HangStandW3S1EnvCfg(Biped12HangStandW3A2EnvCfg):
    """W3 Stage 1 (1006): 같은 스텝 엔진 + 앞 / 뒤 명령 — 명령 = STAND 30 % / STEP(제자리 43 %·앞 28.5 %·뒤 28.5 %, |vx| 0.03–0.10).
    착지 목표 x = vx/(4f)(w3_step_tracker), 표류 = |v − v_cmd|, 추종 w3_track +2(명령 ≠ 0, σ 0.05). 보조 0 고정(start_stage 4), 모방·요 벌점 유지."""

    def __post_init__(self):
        super().__post_init__()
        self.events.w3_cmd.params = {"p_step": 0.7, "stage": 1}
        self.rewards.w3_track = RewTerm(func=_w3.w3_track, weight=2.0)
        self.curriculum.w3_assist.params = {"start_stage": 4}


@configclass
class Biped12HangStandW3S1bEnvCfg(Biped12HangStandW3S1EnvCfg):
    """W3 Stage 1 v2 (1006, 단위시험 2차 + 적대검토): 서기가 0 아닌 명령에서 공짜 점수(−0.28 → −0.17)를 받던 w3_track 을
    서기 대비 개선분(w3_track_rel)으로, 진행량은 스윙발 이륙→착지 변위(w3_stride, ΔX* = vx/f, σ 0.03, 사건당 +10) — 몸 기울임·지지발 미끄럼 해킹 차단.
    뒤 범위 축소(−0.03 – −0.06: 실물 CoM 뒤 여유 작음)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.w3_cmd.params = {"p_step": 0.7, "stage": 1, "vx_back_range": (0.03, 0.06)}
        self.rewards.w3_track = RewTerm(func=_w3.w3_track_rel, weight=3.0)
        self.rewards.w3_stride = RewTerm(func=_w3.w3_stride, weight=10.0)


@configclass
class Biped12HangStandW3S1cEnvCfg(Biped12HangStandW3S1bEnvCfg):
    """W3 Stage 1 c (1006): W3S1b(앞 획득 +0.067 @0.06, 뒤 명령 기울기 정상 −3.6 cm 보폭 차) 는 명령 0 영점이 앞 +0.021 로 치우쳐 뒤가 −0.014 에 그침.
    영점 표류는 Stage 0 결함 — W3-A3 의 순 표류·순 요 벌점(w3_drift_net −3, w3_yaw_net −4)이 해소(−0.003). 두 검증된 변경의 합: S1b 보상 + A3 벌점, W3-A3 끝에서 시작."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.w3_yaw = RewTerm(func=_w3.w3_yaw_net, weight=-4.0)
        self.rewards.w3_drift = RewTerm(func=_w3.w3_drift_net, weight=-3.0)


# ================================================================ c2 (1006) — closure 2 구조 경로: W3 스텝 엔진 + 포획점 착지 + 상태 조건 정지 ================================================================
# 설계·반사실 단위시험 계획: reports/gen_w3/C2_DESIGN.md · 코어: rl_walking/w3_c2.py. 기본 꺼짐(이 클래스들을 고를 때만 적용).
from rl_walking import w3_c2 as _c2  # noqa: E402

_C2_DIST = {"n_range": (1, 2), "t0_range": (3.0, 9.0), "gap_range": (4.0, 7.0), "ramp_prob": 0.6,
            "F_ramp_range": (3.0, 20.0), "rise_range": (1.0, 4.0), "hold_range": (1.0, 3.0),
            "F_imp_range": (5.0, 30.0), "dur_imp_range": (0.2, 0.4), "lateral_prob": 0.0, "lateral_scale": 0.6, "p_active": 1.0,
            "branch": "g22r", "stage_override": 2, "p_none_override": 0.5}


@configclass
class Biped12HangStandW3C2EnvCfg(Biped12HangStandW3S1cEnvCfg):
    """c2 (1006, closure 2): W3S1c 끝에서 이어 학습(관측 actor 320·critic 335 불변 → --resume 그대로).
    - 외란 에피소드 50 %(g22r R3 고정 = G24Ra/KE 학습 분포: 앞 10–12 N·뒤 12–18 N 램프, 충격 ×1.5, 좌우 없음; 첫 외란 3–9 s — 스폰 과도 2 s 회피).
      외란 에피소드는 STAND(ge 0, 명령 0)로 시작하고 supervisor(감지기 → ge 1 → 상태 조건 정지 요청 → 반주기 래치 → POST)만 ge 를 바꾼다.
    - 무외란 에피소드(50 %)는 W3S1c 와 같은 명령·보상(Stage 1 리허설).
    - 보상 변경 2개: w3_place → c2_place(회복 착지만 포획 목표, 그 밖 같은 값), c2_settle(정착 1회/외란, 가중 50 × dt = +1.0).
      초과 스텝 벌점 없음. 기존 STAND 보상(G21a, ge 0 게이트)이 POST 의 자세 복귀를 맡는다.
    - 속도 밀기(push_robot) 없음(W3S1c 그대로 꺼짐) — 평가 fi40/fs80 충격은 g22r 충격(3–7 N·s)이 부분 대표."""

    C2_STOP_V = -1.0        # 비주석 클래스 상수: > 0 이면 정지 요청에 |v| 조건(c2v)
    C2_EXTRA = ""           # 'k=v,...' C2Params 추가 덮어쓰기(c2v 변형용)
    W_SETTLE = 50.0

    def __post_init__(self):
        super().__post_init__()
        ev = self.events; rw = self.rewards
        ev.disturbance = EventTerm(func=mdp_hang.disturbance_reset_g22r, mode="reset", params=dict(_C2_DIST))
        ev.w3_gait.func = _c2.c2_gait_update                     # 같은 자리(정의 순서) — w3_track 앞
        _c2p = {"stop_v": float(self.C2_STOP_V)}
        for _kv in filter(None, str(self.C2_EXTRA).split(",")):
            _k, _v = _kv.split("="); _c2p[_k.strip()] = float(_v)
        ev.w3_gait.params = {"f_step": 1.1, "c2": _c2p}
        ev.c2_track = EventTerm(func=_c2.c2_tracker, mode="interval", interval_range_s=(0.0, 0.0), params={})   # w3_track 뒤(정의 순서)
        rw.w3_place = RewTerm(func=_c2.c2_place_reward, weight=10.0, params={"sig": 0.04, "sig_cap": 0.05})
        rw.c2_settle = RewTerm(func=_c2.c2_value, weight=float(self.W_SETTLE), params={"name": "ev_settle"})
        self.curriculum.c2_log = _CurrTerm(func=_c2.c2_log)


@configclass
class Biped12HangStandW3C2vEnvCfg(Biped12HangStandW3C2EnvCfg):
    """c2v 변형(1006 C2CAL 반영): 정지 요청 = 한 보행 주기(0.909 s) 이동평균 속도 |v̄_T| < 0.07 m/s(제자리 걸음 p90 0.058 × 1.2;
    회복 중 p90 0.25–0.34) & IMU 창 max|ω| < 2.0·기울기 < 12.5°(제자리 걸음 p90 규칙) + 감지기 상대 기울기 τ 10 s(무외란 서기 오탐 22 → 0.9 %).
    학습 속도 = 참 몸통 속도 + 잡음 σ 0.03. 배포엔 다리 운동학 속도 추정기(주기 평균)가 선행 조건 — 현장 확인 필요."""

    C2_STOP_V = 0.07
    C2_EXTRA = "v_box_s=0.909,stop_w=2.0,stop_tilt=12.5,tilt_ref_tau=10,cancel_calm=1,kv_bias=0.03,kv_lf=0.02"   # cancel_calm: 취소 결함 수정 · kv_bias/kv_lf: 레드팀 a(상관 잡음, 가정)

    def __post_init__(self):
        super().__post_init__()
        self.events.c2_track.params = dict(self.events.c2_track.params or {}, cap_half_width=0.18)   # 레드팀 c: 포획 목표 y 폭 = 실측 발 간격 반폭(오차형에만 영향)


@configclass
class Biped12HangStandW3C2vmEnvCfg(Biped12HangStandW3C2vEnvCfg):
    """c2vm (1006, C2UNIT 기준 2 재설계): c2v + 회복 착지 점수 = 이동 방향 포획 여유 clip((발 − ξ)·v̂ / 5 cm, ±1)(|v| ≤ 0.10 이면 0).
    근거: 오차형 exp(−e²/σ²)는 제자리 걸음 착지에도 0.69 를 줘 방향 스텝(B6)을 구별 못 함(c2v S1c B6 0.66 < B2 0.69). 비회복 착지·무외란은 W3 와 비트 동일."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.w3_place.params = dict(self.rewards.w3_place.params, cap_mode="margin")


@configclass
class Biped12HangStandW3C2vdEnvCfg(Biped12HangStandW3C2vEnvCfg):
    """c2vd (1006, 문헌 권고: Atalante 2203.01148·Digit 2404.19173·Cassie 2401.16889·Kangaroo 2411.01000): c2v + 회복 착지 점수 = 차분형
    clip((Φ_전 − Φ_후)/5 cm, 0, 1), Φ = max(0, dist(ξ_eq, 지지영역) − 1 cm), ξ_eq = ξ + F/(mω0²)(지속 외력 평형 보정, 특권). 이미 포획된 착지 = 0."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.w3_place.params = dict(self.rewards.w3_place.params, cap_mode="diff")


@configclass
class Biped12HangStandW3C2k2EnvCfg(Biped12HangStandW3C2vdEnvCfg):
    """c2k2 (1006 밤, 단순화): W3C2vd 와 같되 정지 요청 = **회복 유효 착지 2 번째**(stop_td_k 2 — 상태 조건·속도 추정기 불필요, 시간 초과 4 s 유지).
    근거: 정지 시점 스윕(seed 1·2) — 고정 2 보 낙상이 상태 정지·오라클과 같은 급(충격 B3 24 vs 28 %, 램프 74 vs 78 %). 배포 = IMU 감지기(상대 기울기) + 접촉 0/1 착지 수만."""

    C2_EXTRA = "v_box_s=0.909,stop_w=2.0,stop_tilt=12.5,tilt_ref_tau=10,cancel_calm=1,kv_bias=0.03,kv_lf=0.02,stop_td_k=2"


@configclass
class Biped12HangStandW3C2k1EnvCfg(Biped12HangStandW3C2k2EnvCfg):
    """c2k1 (1007, 정지 시점 확정 대조): W3C2k2 와 같되 첫 회복 착지에서 정지 요청(stop_td_k 1)."""

    C2_EXTRA = "v_box_s=0.909,stop_w=2.0,stop_tilt=12.5,tilt_ref_tau=10,cancel_calm=1,kv_bias=0.03,kv_lf=0.02,stop_td_k=1"


_C2_KEEP_ON_REC = ("knee_stance_flexion", "feet_contact_gated", "track_lin_vel_xy_exp", "track_ang_vel_z_exp", "survive_after_release")   # C2_DESIGN §14 최소 유지 집합


@configclass
class Biped12HangStandW3C2k2hEnvCfg(Biped12HangStandW3C2k2EnvCfg):
    """c2k2h (1007, GO 후보 학습 설정): W3C2k2(차분형 v2 포획 · 고정 2 보 정지 · 상대 기울기 감지) + **회복 중 서기 보상 유지 집합**
    (knee_stance_flexion·feet_contact_gated·track_lin_vel_xy_exp·track_ang_vel_z_exp·survive_after_release — C2UNIT 6 seed 최소 집합, C2_DESIGN §14).
    무외란 에피소드(범주 0)는 W3 와 같은 게이팅(ge 0 만) → Stage 0/1 리허설 불변. settle_quiet_bonus 는 제외(상태 타이머 항, 기여 +0.04)."""

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards
        for n in _C2_KEEP_ON_REC:
            t = getattr(rw, n)
            assert t.func is _w3.w3_stand_gated, n
            setattr(rw, n, RewTerm(func=_c2.c2_stand_or_rec_gated, weight=t.weight, params=dict(t.params)))


@configclass
class Biped12HangStandW3C2iEnvCfg(Biped12HangStandW3C2EnvCfg):
    """c2i 변형: critic 에 ICP−CoM(요 프레임 2차원, 이력 5) 추가 — critic 입력 335 → 345, **--init_expand 필요**(resume 불가). actor 불변."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.critic.c2_icp = _ObsTerm(func=_c2.c2_icp_obs, history_length=5, flatten_history_dim=True)


@configclass
class Biped12HangStandW3S1dEnvCfg(Biped12HangStandW3S1cEnvCfg):
    """W3 Stage 1 d (1006): S1c 와 단 하나만 다름 — 앞 명령 범위를 뒤와 대칭(0.03–0.06). 가설: 명령 분포 비대칭(앞 0.03–0.10 vs 뒤 0.03–0.06)이
    명령 0 영점 앞 편향(S1c 13300 +0.024 → 13500 +0.036, A3 시작 −0.006)을 만든다. 같은 시작점(W3-A3 끝)·같은 seed 로 S1c 와 짝 비교."""

    def __post_init__(self):
        super().__post_init__()
        self.events.w3_cmd.params = {"p_step": 0.7, "stage": 1, "vx_range": (0.03, 0.06), "vx_back_range": (0.03, 0.06)}


# ================================================================ W3 Stage 2 (1006) — 옆걸음(vy). 기본 꺼짐(이 클래스를 고를 때만 적용) ================================================================
# 설계·반사실 단위시험·사전 통과 기준: reports/gen_w3/S2_DESIGN.md. CPU 단위시험: rl_walking/scripts/unit_w3_s2.py.
@configclass
class _W3S2CurriculumCfg:
    w3_assist = _CurrTerm(func=_w3.w3_assist_curriculum, params={"start_stage": 4})
    w3_vy = _CurrTerm(func=_w3.w3_vy_curriculum)


@configclass
class Biped12HangStandW3S2EnvCfg(Biped12HangStandW3S1cEnvCfg):
    """W3 Stage 2 (1006): W3S1c 끝에서 이어 학습(관측 actor 320·critic 335 불변 → --resume 그대로; 명령 채널 vy 는 이미 관측에 있음).
    - 명령: STAND 30 % / STEP(제자리 0.3·앞뒤 0.3(Stage 1 범위 리허설)·옆 0.4(±vy)), |vy| 0.03–0.04 → 0.06 → 0.08(w3_vy_curriculum:
      부호 정답 ≥ 90 %·폭 위반 < 2 %·체류 ≥ 100 반복).
    - 위상: vy < 0 이면 STEP 시작 위상 0.5(오른발 리드), vy > 0 은 0(왼발 리드) — 가는 쪽 발이 먼저(배포 러너 위상 생성기도 같은 규칙 필요 = 배포 계약 변경).
    - 보상 변경 2개: w3_stride → w3_stride2(옆 명령만 2D ΔP* = (vx/f, vy/f) + 폭 게이트; vy 0 은 Stage 1 과 비트 동일), w3_width −10/사건
      (착지 순간 좌우 발 간격 W < 2·W_HALF − 0.05 또는 교차). W_HALF 0.144 = STAND 정책 실측 간격 28.7 cm 의 절반(night_1005 LOG) → 문턱 0.238 m.
    - 그 밖(추종 w3_track_rel 2D·순 표류 −3·순 요 −4·착지 목표·유효 스텝·종료)은 W3S1c 그대로."""

    W_HALF = 0.144          # 비주석 클래스 상수(설정 필드 아님)

    def __post_init__(self):
        super().__post_init__()
        ev = self.events; rw = self.rewards
        ev.w3_cmd.params = {"p_step": 0.7, "stage": 2, "vx_back_range": (0.03, 0.06), "mix2": (0.3, 0.3, 0.4), "vy_range": (0.03, 0.04)}
        ev.w3_gait.params = {"f_step": 1.1, "lead_vy_cmin": 0.02}
        ev.w3_lat = EventTerm(func=_w3.w3_lat_tracker, mode="interval", interval_range_s=(0.0, 0.0),
                              params={"nominal_half_width": float(self.W_HALF), "width_margin": 0.05})   # w3_track 뒤(정의 순서)
        rw.w3_stride = RewTerm(func=_w3.w3_stride2, weight=10.0)
        rw.w3_width = RewTerm(func=_w3.w3_width_pen, weight=-10.0)
        self.curriculum = _W3S2CurriculumCfg()


@configclass
class Biped12HangStandW3S1eEnvCfg(Biped12HangStandW3S1cEnvCfg):
    """W3 Stage 1 e (1006, 레드팀 결함 1·4 수리): 착지 목표 폭이 ±0.10(발 간격 20 cm)인데 실제 걷기 간격 36 cm → 착지 보상 exp(−4) ≈ 0.02(가중 10 항이 사실상 죽음,
    명령 0 발 x 를 잡을 신호 부재 = 영점 편향 후보). 수리 = 목표 반폭 0.17(실측 걷기 0.18·서기 0.144 사이) + 착지·보폭 보상을 유효 스텝 사건에만 지급(수리로 살아나는
    두드리기 유인 차단). 그 밖 보상·명령 분포는 W3S1c 와 동일. 짝 비교: 같은 시작(S1c seed 2 끝)·같은 seed 로 W3S1c 이어학습(S1c2x)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.w3_track.params = {"nominal_half_width": 0.17}
        self.rewards.w3_place = RewTerm(func=_w3.w3_place_valid, weight=10.0)
        self.rewards.w3_stride = RewTerm(func=_w3.w3_stride_valid, weight=10.0)


# ================================================================ W3 Stage 3 (1006) — 제자리 회전(wz). 기본 꺼짐(이 클래스들을 고를 때만 적용) ================================================================
# 설계·반사실 단위시험·사전 통과 기준: reports/gen_w3/S3_DESIGN.md. CPU 단위시험: rl_walking/scripts/unit_w3_s3.py.
# 측정 사실(1006 요 표류 후속): W3 태스크 command term 은 heading_command=True·ang_vel_z (0,0) → 직접 넣은 wz 가 매 틱 0 으로 덮여 정책 wz 응답 이득 ≈ 0.
# Stage 3 설정에서만 heading_command 를 끈다(기존 태스크 불변).
@configclass
class _W3S3CurriculumCfg:
    w3_assist = _CurrTerm(func=_w3.w3_assist_curriculum, params={"start_stage": 4})
    w3_wz = _CurrTerm(func=_w3.w3_wz_curriculum)


@configclass
class _W3S3s2CurriculumCfg:
    w3_assist = _CurrTerm(func=_w3.w3_assist_curriculum, params={"start_stage": 4})
    w3_vy = _CurrTerm(func=_w3.w3_vy_curriculum)
    w3_wz = _CurrTerm(func=_w3.w3_wz_curriculum)


def _w3s3_apply(cfg, mix3: tuple, p_hloop: float = 0.0):
    """Stage 3 공통 변경(한 묶음): ① command term heading 끔 ② 명령 stage 3(회전 범주 ±wz, |wz| 0.1–0.2 → 커리큘럼 0.4)
    ③ wz < 0 오른발 리드 ④ 회전 추적기(w3_track 뒤) ⑤ 보상 2개: 발 요 착지(w3_turn_stride +10/사건) · 발 연결선 순 요 추종 개선분(w3_ytrack +3).
    p_hloop > 0 이면 외부 헤딩 P 고리(w3_heading_loop, kp 0.5 — 배포·eval_w3 --heading_loop 와 같은 식)를 STEP·비회전 명령의 그 비율에 건다."""
    c = cfg.commands.base_velocity
    c.heading_command = False
    if hasattr(c, "rel_heading_envs"):
        c.rel_heading_envs = 0.0
    ev = cfg.events; rw = cfg.rewards
    p = dict(ev.w3_cmd.params)
    p.update({"stage": 3, "mix3": tuple(mix3), "wz_range": (0.1, 0.2), "p_hloop": float(p_hloop)})
    ev.w3_cmd.params = p
    g = dict(ev.w3_gait.params); g["lead_wz_cmin"] = 0.08
    ev.w3_gait.params = g
    ev.w3_turn = EventTerm(func=_w3.w3_turn_tracker, mode="interval", interval_range_s=(0.0, 0.0), params={"cmin": 0.08})   # w3_track 뒤(정의 순서)
    if p_hloop > 0:
        ev.w3_hloop = EventTerm(func=_w3.w3_heading_loop, mode="interval", interval_range_s=(0.0, 0.0), params={"kp": 0.5, "wz_max": 0.4})
    rw.w3_turn = RewTerm(func=_w3.w3_turn_stride, weight=10.0, params={"sig": 0.0873, "cmin": 0.08, "k_opp": 0.5})
    rw.w3_ytrack = RewTerm(func=_w3.w3_yaw_track_rel, weight=3.0, params={"sig": 0.2, "cmin": 0.08, "src": "feet"})


@configclass
class Biped12HangStandW3S3EnvCfg(Biped12HangStandW3S1cEnvCfg):
    """W3 Stage 3 (1006): W3S1c 끝에서 이어 학습(관측 actor 320·critic 335 불변 → --resume 그대로; wz 는 velocity_commands 3 번째 채널에 이미 있음).
    - 명령: STAND 30 % / STEP(제자리 0.3·앞뒤 0.3(Stage 1 리허설)·회전 0.4(±wz)), |wz| 0.1–0.2 → 0.3 → 0.4(w3_wz_curriculum:
      발 요 부호 정답 ≥ 90 %·요 이득 ≥ 0.5·체류 ≥ 100 반복). heading_command 끔(이 설정만).
    - 보상 추가 2개(명령 wz 0 이면 정확히 0 → Stage 0/1 경로 총 보상 비트 동일): w3_turn_stride +10/사건(Δψ* = wz/f, σ 5°, 제자리 기준선 차감 + 반대 투영 벌점 0.5),
      w3_ytrack +3(발 연결선 한 주기 순 요 속도, σ 0.2, 개선분형). 순 요 벌점 w3_yaw_net −4(이미 명령 wz 대비)는 그대로.
    - 위상: wz < −0.08(옆 명령 없음)이면 오른발 리드(배포 러너 위상 생성기 계약 변경 — 실물 전 반영 필요)."""

    def __post_init__(self):
        super().__post_init__()
        _w3s3_apply(self, (0.3, 0.3, 0.0, 0.4))
        self.curriculum = _W3S3CurriculumCfg()


@configclass
class Biped12HangStandW3S3hEnvCfg(Biped12HangStandW3S1cEnvCfg):
    """W3S3 + 외부 헤딩 고리 학습(선택): STEP·비회전 명령 env 의 50 % 에서 wz = clip(kp·wrap(ψ* − ψ), ±0.4), kp 0.5, ψ* = 걷기 시작 요.
    배포(정책 밖 헤딩 P 고리)와 같은 형태를 학습 분포에 넣는다. W3S3 과의 단일 변경 짝 비교용."""

    def __post_init__(self):
        super().__post_init__()
        _w3s3_apply(self, (0.3, 0.3, 0.0, 0.4), p_hloop=0.5)
        self.curriculum = _W3S3CurriculumCfg()


@configclass
class Biped12HangStandW3S3s2EnvCfg(Biped12HangStandW3S2EnvCfg):
    """W3 Stage 3 (S2 상속판, S2 통과 시 사용): W3S2 끝에서 이어 학습. 명령 STEP 중 제자리 0.25·앞뒤 0.2·옆 0.25(S2 리허설)·회전 0.3.
    vy 커리큘럼은 이어서 돈다(start_stage 는 S2 결과에 맞춰 덮어쓸 것). 그 밖은 W3S3 과 같다."""

    def __post_init__(self):
        super().__post_init__()
        _w3s3_apply(self, (0.25, 0.2, 0.25, 0.3))
        self.curriculum = _W3S3s2CurriculumCfg()


@configclass
class Biped12HangStandW3S2dEnvCfg(Biped12HangStandW3S2EnvCfg):
    """W3S2 d (1006 밤, 병목 3 delay robustness 단일 변경): 학습 액션 지연 0–3 → 1–5 틱(실물 실효 2–4 틱 추정, 0930 수정 뒤 미측정).
    W3-A3 섭동: 지연 3 채점 낙상 1.7–3.3 %, 지연 4(학습 밖) 4.6 %. 그 외 W3S2 와 동일. W3S2 끝에서 이어 학습, 짝 = 같은 체크포인트의 지연 2/3/4/5 평가."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.min_delay_steps = 1
        self.actions.joint_pos.max_delay_steps = 5


# ================================================================ W3 Stage 4 (1006) — 혼합 명령(vx·vy·wz 동시, 곡선 보행). 기본 꺼짐(이 클래스를 고를 때만 적용) ================================================================
# 설계·반사실 단위시험·사전 통과 기준: reports/gen_w3/S4_DESIGN.md. CPU 단위시험: rl_walking/scripts/unit_w3_s4.py.
@configclass
class _W3S4CurriculumCfg:
    w3_assist = _CurrTerm(func=_w3.w3_assist_curriculum, params={"start_stage": 4})
    w3_vy = _CurrTerm(func=_w3.w3_vy_curriculum)
    w3_wz = _CurrTerm(func=_w3.w3_wz_curriculum)
    w3_mix = _CurrTerm(func=_w3.w3_mix_curriculum)


@configclass
class Biped12HangStandW3S4EnvCfg(Biped12HangStandW3S3s2EnvCfg):
    """W3 Stage 4 (1006): W3S3s2 끝에서 이어 학습(관측 actor 320·critic 335 불변 → --resume 그대로; vx·vy·wz 는 이미 velocity_commands 채널).
    - 명령: STAND 30 % / STEP(제자리 0.2·앞뒤 0.15·옆 0.15·회전 0.2(단일 축 리허설, Stage 3 식)·혼합 0.3). 혼합 안 조합 = 앞뒤+회전 0.4·앞뒤+옆 0.2·
      옆+회전 0.15·셋 다 0.25, 축별 부호 독립. 크기 = lo + s·U·(hi − lo)(단일 축 현재 범위 기준), s 0 → 0.5 → 1(w3_mix_curriculum:
      활성 축 부호 정답 ≥ 90 %·폭 위반 < 2 %·체류 ≥ 100 반복).
    - 리드 발: 코드 변경 없음(W3S3 규칙 그대로) — |vy| ≥ 0.02 면 vy 부호가 정함(옆 우선), 옆 없고 wz < −0.08 이면 오른발. 충돌 해소 근거 S4_DESIGN §2.
    - 보상 교체 3개(회전 + 병진 명령에서만 값이 달라짐 — 그 밖 기존 함수 값 그대로): w3_stride → w3_stride4(곡선 착지 목표),
      w3_track → w3_track_rel4, w3_drift → w3_drift_net4(창 동안 실제 요 경로로 명령 적분). 발 요 착지(w3_turn)·순 요(w3_yaw_net)·발 연결선 추종(w3_ytrack) 불변.
    - vy·wz 단일 축 커리큘럼 시작 단계는 S3s2 결과(도달 상한)에 맞춰 덮어쓸 것(커리큘럼 상태는 체크포인트에 저장되지 않음 — 기본 = 최대 단계 가정)."""

    VY_STAGE = 2            # 비주석 클래스 상수(설정 필드 아님): w3_vy_curriculum start_stage(0.08)
    WZ_STAGE = 2            # w3_wz_curriculum start_stage(0.4)
    W_HALF4 = 0.144         # 곡선 지레 항 발 횡 오프셋(W3S2 W_HALF 와 같은 값)

    def __post_init__(self):
        super().__post_init__()
        ev = self.events; rw = self.rewards
        p = dict(ev.w3_cmd.params)
        p.update({"stage": 4, "mix4": (0.2, 0.15, 0.15, 0.2, 0.3), "mix_pat": (0.4, 0.2, 0.15, 0.25), "mix_scale": 0.0})
        ev.w3_cmd.params = p
        ev.w3_mix = EventTerm(func=_w3.w3_mix_tracker, mode="interval", interval_range_s=(0.0, 0.0),
                              params={"cmin": 0.02, "wcmin": 0.08, "nominal_half_width": float(self.W_HALF4)})   # w3_turn 뒤(정의 순서)
        rw.w3_stride = RewTerm(func=_w3.w3_stride4, weight=10.0, params={"nominal_half_width": float(self.W_HALF4)})
        rw.w3_track = RewTerm(func=_w3.w3_track_rel4, weight=3.0)
        rw.w3_drift = RewTerm(func=_w3.w3_drift_net4, weight=-3.0)
        self.curriculum = _W3S4CurriculumCfg()
        self.curriculum.w3_vy.params = {"start_stage": int(self.VY_STAGE)}
        self.curriculum.w3_wz.params = {"start_stage": int(self.WZ_STAGE)}


@configclass
class Biped12HangStandW3S3dEnvCfg(Biped12HangStandW3S3s2EnvCfg):
    """W3S3 d (1007 새벽): W3S3s2(회전, 헤딩 고리 120 s 요 1.7°) 보상 + W3S2d 의 학습 지연 1–5 틱(지연 4–5 낙상 38→26·80→35).
    두 검증된 변경의 합, 시작 = W3S2d 끝. 회전은 +100–400 반복에서 충분(후반 이득 과대·낙상 증가) → 500 반복, 이른 체크포인트 선택."""

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.min_delay_steps = 1
        self.actions.joint_pos.max_delay_steps = 5


@configclass
class Biped12HangStandW3S3dREnvCfg(Biped12HangStandW3S3dEnvCfg):
    """W3S3d R (1007 claude2, reports/gen_w3/ROBUST_DESIGN.md): W3S3d 와 보상·관측·커리큘럼·학습 지연 1–5 동일, **DR 대역만 확장**(ENVELOPE §3.5 민감 축).
    IMU 피치 편향 ±1.5 → ±4°, 롤 ±1.5 → ±3°, 관절 영점 ±1 → ±3°(관절별 독립), CoM x ±2 → ±4 cm, 발목 유격 배수 0.75–1.25 → 0.3–2.0, kp 배수 0.85–1.1 → 0.7–1.3.
    시작 = W3S3d-15800 이어 학습. 평가·승격 기준은 설계 문서 §3(학습 전 고정)."""

    def __post_init__(self):
        super().__post_init__()
        cb = self.events.calib_bias.params
        cb["imu_tilt_rad"] = (-0.070, 0.070); cb["imu_roll_rad"] = (-0.052, 0.052); cb["joint_bias_rad"] = (-0.052, 0.052)
        cr = dict(self.events.base_com.params["com_range"]); cr["x"] = (-0.04, 0.04); self.events.base_com.params["com_range"] = cr
        for _a in self.scene.robot.actuators.values():
            if getattr(_a, "play_rad_range", None) is not None:
                _a.play_rad_range = (0.3, 2.0)
        self.events.actuator_gains.params["stiffness_distribution_params"] = (0.7, 1.3)


@configclass
class Biped12HangStandW3S4dEnvCfg(Biped12HangStandW3S4EnvCfg):
    """W3S4 d (1007 새벽): Stage 4 혼합(vx·vy·wz) + 학습 지연 1–5 틱, 시작 = W3S3d-15800(헤딩 고리 120 s 2.2°, 회전 이득 0.84–0.91).
    S3d 학습 중 vy·wz 커리큘럼은 상한 0.04·0.2 에서 진급하지 않았음(로그 지표 nan — 별도 점검 과제) — 평가상 vy 0.05–0.08·wz 0.3 추종하므로 시작 단계 1(vy 0.06·wz 0.3)."""

    VY_STAGE = 1
    WZ_STAGE = 1

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos.min_delay_steps = 1
        self.actions.joint_pos.max_delay_steps = 5


# ================================================================ c2 를 이후 W3 단계 위에(1007) ================================================================
def _c2_apply(cfg, c2_extra: str, keep_on_rec: tuple = (), cap_mode: str = "diff", w_settle: float = 50.0):
    """c2 변경 한 묶음을 임의의 W3 설정(cfg)에 적용(1007 — W3S3d 등 이후 단계 위에 얹기 위해). W3C2* 클래스와 같은 내용:
    외란 g22r R3 50 % · supervisor(c2_gait_update, 기존 w3_gait 파라미터의 리드 발 규칙 유지) · c2_tracker · 착지 보상 c2_place(cap_mode) · 정착 c2_settle ·
    c2_log · 회복 중 서기 보상 유지(keep_on_rec)."""
    ev = cfg.events; rw = cfg.rewards
    ev.disturbance = EventTerm(func=mdp_hang.disturbance_reset_g22r, mode="reset", params=dict(_C2_DIST))
    c2p = {}
    for kv in filter(None, str(c2_extra).split(",")):
        k, v = kv.split("="); c2p[k.strip()] = float(v)
    g = {k: v for k, v in dict(ev.w3_gait.params).items() if k in ("f_step", "ramp_s", "lead_vy_cmin", "lead_wz_cmin")}
    g["c2"] = c2p
    ev.w3_gait.func = _c2.c2_gait_update
    ev.w3_gait.params = g
    ev.c2_track = EventTerm(func=_c2.c2_tracker, mode="interval", interval_range_s=(0.0, 0.0), params={"cap_half_width": 0.18})
    rw.w3_place = RewTerm(func=_c2.c2_place_reward, weight=10.0, params={"sig": 0.04, "sig_cap": 0.05, "cap_mode": cap_mode})
    rw.c2_settle = RewTerm(func=_c2.c2_value, weight=float(w_settle), params={"name": "ev_settle"})
    cfg.curriculum.c2_log = _CurrTerm(func=_c2.c2_log)
    for n in keep_on_rec:
        t = getattr(rw, n)
        assert t.func is _w3.w3_stand_gated, n
        setattr(rw, n, RewTerm(func=_c2.c2_stand_or_rec_gated, weight=t.weight, params=dict(t.params)))


@configclass
class Biped12HangStandW3S3eEnvCfg(Biped12HangStandW3S3dEnvCfg):
    """W3S3 e (1007 claude2, reports/gen_w3/S3E_DESIGN.md): W3S3d 와 보상·관측·DR·커리큘럼·학습 지연 1–5 동일, **명령 리샘플 간격만 8–12 s → 4–6 s**.
    근거(전환 §5.11e·f): 20 s 에피소드에 명령 구간이 2 개뿐이라 '걷고 선 뒤 다시 걷기'(자기 정지 자세에서의 재출발)가 학습 분포에 없었다.
    배포 코드 폐루프의 왕복 대본에서 첫 걷기 낙상 0, 둘째 이후 1–3 %·걷기 한 번의 요 변화가 왕복마다 증가. 구간을 3–5 개로 늘려 재출발을 겪게 한다.
    시작 = W3S3d-15800 이어 학습. 평가·승격 기준은 설계 문서(학습 전 고정)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.resampling_time_range = (4.0, 6.0)


@configclass
class Biped12HangStandW3S3fEnvCfg(Biped12HangStandW3S3dEnvCfg):
    """W3S3 f (1007 claude2, reports/gen_w3/TORQUE_BUDGET_1007.md §7): W3S3d 와 보상·관측·커리큘럼·명령·학습 지연 동일, **토크 한계 무작위화만 추가** —
    리셋마다 전 관절 토크 한계 0.7–1.0 배(mdp_stage4.resample_effort_scale: PhysX 한계 + 액추에이터 모델 한계). 관측·행동 계약 불변.
    근거(X13): W3S3d-15800 은 한계 0.8 배에서 걷기 낙상 14.2 %, 0.7 배에서 98 % — 피크 토크에 기댄 걸음. 정책이 한계를 관측하지 못하므로
    낮은 한계에서도 성립하는 걸음(지지발 쪽으로 더 옮기는 걸음)을 배우게 하려는 단일 변경. 시작 = W3S3d-15800 이어 학습. 기준은 문서(학습 전 고정)."""

    def __post_init__(self):
        super().__post_init__()
        from . import mdp_stage4 as _m4
        self.events.effort_scale = EventTerm(func=_m4.resample_effort_scale, mode="reset", params={"scale_range": (0.7, 1.0)})


@configclass
class Biped12HangStandW3S3gEnvCfg(Biped12HangStandW3S3eEnvCfg):
    """W3S3 g (1007 claude2, reports/gen_w3/TRANSITION_W3KE.md §5.19·S3G 사전 등록): W3S3e(명령 리샘플 4–6 s = 재출발 포함)와 전부 같고 **보상 항 하나만 추가** —
    `w3_lean`: 걷는 중 한 주기 평균 옆 기울기 |ḡ_y| 선형 벌점(가중 −8, 1° 쏠림 ≈ −0.14/틱 — 요 벌점 −4·|ω| 의 2°/s 와 같은 크기).
    근거(배포 코드 폐루프 왕복 기록): 연속 걷기에서는 요 속도가 커지지 않고(4 s 구간 표준편차 4–6°), 멈췄다 다시 걸을 때마다 커진다(6 → 21°).
    걷는 중 평균 롤과 요 속도는 매 걷기 r 0.67–0.93·기울기 ≈ 2 (°/s)/° — 한쪽으로 쏠린 걸음이 그대로 유지되는 무조절 모드. 관측·행동 계약 불변.
    대조 = W3S3e 같은 반복 수(리샘플은 같고 이 항만 다름). 시작 = W3S3d-15800 이어 학습."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.w3_lean = RewTerm(func=_w3.w3_lean_net, weight=-8.0)


@configclass
class Biped12HangStandW3S3dC2EnvCfg(Biped12HangStandW3S3dEnvCfg):
    """W3S3d-c2 (1007, c2 GO 학습 설정 — 실행은 승인 뒤): W3S3d(회전·옆·앞뒤·지연 1–5) 끝에서 이어 학습 + c2 묶음(W3C2k2h 와 같은 내용):
    고정 2 보 정지(stop_td_k 2) · 상대 기울기 감지(τ 10 s) · 차분형 v2 포획 착지 · 회복 중 서기 보상 유지 5 항. 관측 actor 320·critic 335 불변 → --resume."""

    def __post_init__(self):
        super().__post_init__()
        _c2_apply(self, "v_box_s=0.909,stop_w=2.0,stop_tilt=12.5,tilt_ref_tau=10,cancel_calm=1,kv_bias=0.03,kv_lf=0.02,stop_td_k=2",
                  keep_on_rec=_C2_KEEP_ON_REC, cap_mode="diff")



# ================================================================ W3D (1007) — 걷는 중 외란 → 회복 스텝 → 정상 WALK/STAND 복귀. 기본 꺼짐(이 클래스를 고를 때만) ================================================================
# 설계·반사실 단위시험·사전 통과 기준: reports/gen_w3/WD_DESIGN.md · 코어 rl_walking/w3_dist.py · CPU 단위시험 rl_walking/scripts/unit_w3d.py.
from rl_walking import w3_dist as _w3d  # noqa: E402


@configclass
class Biped12HangStandW3DEnvCfg(Biped12HangStandW3S3dEnvCfg):
    """W3D (1007): W3S3d 상속(관측 actor 320·critic 335 불변 → W3S3d-15800 에서 --resume, 학습 지연 1–5 유지, 명령 분포·기존 보상 그대로).
    - 외란(ev.disturbance 자리 — 평가기가 지우면 같이 꺼짐): 에피소드 50 % 에 1–2 회, t0 U(3, 15) s(보행 위상 균일 분산), 충격 70 %(0.2–0.4 s)·짧은 램프 30 %,
      방향 = 발생 순간 몸통 앞 .35·뒤 .25·왼 .2·오른 .2, **걷는 env 만**(발생 직전 STAND 면 취소). 충격 크기 단계 6–10 → 14 → 18 → 22 N
      (w3d_curriculum: 낙상 ≤ 25 %·≥ 1500 에피소드·체류 ≥ 100 반복). 외란 에피소드의 30 % 는 외란 끝 + 0–1 s 에 정지 명령(에피소드 끝까지) → STAND 복귀.
    - 보상 추가(전부 가산, 기존 걷기 항 유지; 가중 단위는 기존 W3 항과 같음 — place 10 = 착지 사건당 10·dt):
      회복 착지 포획 차분형 v2 +W_CAP, 착지·보폭 손해 보전(place·stride 와 같은 가중), WALK 복귀 1 회 +W_RET, STAND 복귀 1 회 +W_SRET.
      외란 없는 env 에서 새 항 = 0."""

    WD_P = 0.5              # 비주석 클래스 상수(설정 필드 아님): 외란 에피소드 비율(0 = W3S3d 와 학습 전체 비트 동일)
    W_CAP = 10.0
    W_RET = 20.0
    W_SRET = 20.0
    WD_LEVEL = 0            # 충격 크기 시작 단계
    CUR2 = False            # True: vy 커리큘럼 판정을 주기 평균 가로 이득(w3_cur2)으로·wz 부호 문턱 0.85 — 기본 False(종전 판정 그대로). WD_DESIGN.md §6

    def __post_init__(self):
        super().__post_init__()
        ev = self.events; rw = self.rewards
        ev.disturbance = EventTerm(func=_w3d.w3d_dist_reset, mode="reset", params={"p_dist": float(self.WD_P)})
        ev.w3_gait.func = _w3d.w3d_gait_update                      # 같은 자리: 정지 시나리오 명령 덮기 → W3 엔진(대상 없으면 w3_gait_update 그대로)
        ev.w3d_step = EventTerm(func=_w3d.w3d_dist_step, mode="interval", interval_range_s=(0.0, 0.0), params={})   # w3_track·w3_lat·w3_turn 뒤(정의 순서)
        ev.w3d_track = EventTerm(func=_w3d.w3d_tracker, mode="interval", interval_range_s=(0.0, 0.0), params={})
        rw.w3d_cap = RewTerm(func=_w3d.w3d_cap, weight=float(self.W_CAP))
        rw.w3d_rplace = RewTerm(func=_w3d.w3d_relief_place, weight=float(rw.w3_place.weight), params={"sig": 0.04})
        rw.w3d_rstride = RewTerm(func=_w3d.w3d_relief_stride, weight=float(rw.w3_stride.weight))
        rw.w3d_ret = RewTerm(func=_w3d.w3d_value, weight=float(self.W_RET), params={"name": "ev_ret"})
        rw.w3d_sret = RewTerm(func=_w3d.w3d_value, weight=float(self.W_SRET), params={"name": "ev_sret"})
        self.curriculum.w3d = _CurrTerm(func=_w3d.w3d_curriculum, params={"start_level": int(self.WD_LEVEL)})
        if self.CUR2:
            from rl_walking import w3_cur2 as _wc2
            ev.w3_latgain = EventTerm(func=_wc2.w3_lat_gain_tracker, mode="interval", interval_range_s=(0.0, 0.0), params={})
            self.curriculum.w3_vy = _CurrTerm(func=_wc2.w3_vy_curriculum2, params=dict(self.curriculum.w3_vy.params or {}))
            self.curriculum.w3_wz.params = dict(self.curriculum.w3_wz.params or {}, min_sign=0.85)


@configclass
class Biped12HangStandW3D0EnvCfg(Biped12HangStandW3DEnvCfg):
    """W3D0: 외란 비율 0 — 비트 동일 대조(새 항·이벤트는 있으나 값 0, 난수 소비 0)."""

    WD_P = 0.0


@configclass
class Biped12HangStandW3S4dDEnvCfg(Biped12HangStandW3S4dEnvCfg):
    """W3S4dD (1007): W3S4d(혼합 vx·vy·wz, 지연 1–5) 위에 W3D 가산 묶음(외란·정지 시나리오·포획 착지·손해 보전·복귀 1 회). W3S4d-16299 에서 --resume.
    W3D 와 다른 점 1 개: 보폭 보전이 w3_stride4(곡선 착지 목표) 기준(w3d_relief_stride4). CUR2 없음(단일 변경). 외란 없는 env 에서 새 항 = 0."""

    WD_P = 0.5
    W_CAP = 10.0
    W_RET = 20.0
    W_SRET = 20.0
    WD_LEVEL = 0

    def __post_init__(self):
        super().__post_init__()
        ev = self.events; rw = self.rewards
        assert rw.w3_stride.func is _w3.w3_stride4 and ev.w3_gait.func is _w3.w3_gait_update
        ev.disturbance = EventTerm(func=_w3d.w3d_dist_reset, mode="reset", params={"p_dist": float(self.WD_P)})
        ev.w3_gait.func = _w3d.w3d_gait_update
        ev.w3d_step = EventTerm(func=_w3d.w3d_dist_step, mode="interval", interval_range_s=(0.0, 0.0), params={})   # w3_mix 뒤(정의 순서)
        ev.w3d_track = EventTerm(func=_w3d.w3d_tracker, mode="interval", interval_range_s=(0.0, 0.0), params={})
        rw.w3d_cap = RewTerm(func=_w3d.w3d_cap, weight=float(self.W_CAP))
        rw.w3d_rplace = RewTerm(func=_w3d.w3d_relief_place, weight=float(rw.w3_place.weight), params={"sig": 0.04})
        rw.w3d_rstride = RewTerm(func=_w3d.w3d_relief_stride4, weight=float(rw.w3_stride.weight),
                                 params={"nominal_half_width": float(rw.w3_stride.params.get("nominal_half_width", 0.144))})
        rw.w3d_ret = RewTerm(func=_w3d.w3d_value, weight=float(self.W_RET), params={"name": "ev_ret"})
        rw.w3d_sret = RewTerm(func=_w3d.w3d_value, weight=float(self.W_SRET), params={"name": "ev_sret"})
        self.curriculum.w3d = _CurrTerm(func=_w3d.w3d_curriculum, params={"start_level": int(self.WD_LEVEL)})


@configclass
class Biped12HangStandW3S4dD0EnvCfg(Biped12HangStandW3S4dDEnvCfg):
    """W3S4dD0: 외란 비율 0 — W3S4d 와 비트 동일 대조."""

    WD_P = 0.0


# ================================================================ W3S3t* (1007) — 걷기 토크 예산: 계약(관측·행동·클록·기본자세) 불변 단일 변경 후보. 기본 꺼짐(이 클래스들을 고를 때만) ================================================================
# 사전 등록 초안·근거 수치: reports/gen_w3/TORQUE_W3T_DESIGN.md · 분석 도구 reports/gen_w3/torque_phase.py · CPU 단위시험 rl_walking/scripts/unit_w3t.py.
# 배경(측정, TORQUE_BUDGET_1007.md §1·§6 + 위 도구): W3S3d-15800 걷기는 지지 힙 외전을 반주기의 21 % 동안 25 N·m 에 붙이고(스윙발 이륙 구간 v .1–.4),
# 지지 무릎을 65° 굽힌 채 평균 18.6 N·m 로 버틴다(그 자세의 한발 지지 정적 요구 23.5 N·m). 한계 0.8 배에서는 지지 무릎이 반주기의 61 % 동안 한계에 붙는다.
# 학습 보상에서 토크 항(torque_saturation −2·dof_torques_l2)은 w3_stand_gated 로 감싸져 걷는 동안 0 이다(env.yaml 확인) → 걷는 동안 토크가 공짜.
# 토크 한계 무작위화(W3S3f, TORQUE_BUDGET_1007.md §7)는 이 묶음이 아니다 — 같은 것을 다시 만들지 않는다.
@configclass
class Biped12HangStandW3S3tEnvCfg(Biped12HangStandW3S3dEnvCfg):
    """W3S3 t (1007, TORQUE_W3T_DESIGN.md §T2): W3S3d 와 관측·행동·명령·DR·커리큘럼·학습 지연 1–5·기존 보상 전부 같고 **보상 항 하나만 추가** —
    `w3_torque_sat` = 걷는 동안(ge 1)의 토크 포화 벌점: w3_step_gated(mdp_stage4.torque_saturation, thresh 0.6) × W_TSAT.
    함수·문턱은 서기에 이미 걸려 있는 항(가중 −2, w3_stand_gated)과 같다: 관절 평균 clamp(|τ_적용|/τ_한계 − 0.6, 0)²(15 N·m 초과분만).
    가중 근거(학습 전 계산): W3S3d-15800 걷기에서 이 값 평균 0.0076(명령별 0.0067–0.0088, 힙 외전 48 %·무릎 48 %) → 가중 −40 = 걷기 초당 −0.30
    (걷기 순 보상 +1.28/s 의 24 %, 기존 걷기 벌점 w3_arate −0.46·w3_yaw −0.39 와 같은 급). 서기 가중 −2 를 그대로 쓰면 −0.015/s 로 사실상 0.
    문턱 0.6 근거: 한발 지지 비율이 같은 한 힙 외전 보폭 평균(≈ 8 N·m)은 각운동량 균형으로 고정이라 0.33(연속 정격) 문턱은 어떤 걸음도 벌점 0 이 될 수 없다.
    0.6(15 N·m)은 벌점 0 인 해가 물리적으로 있다(힙 외전을 접지 내내 고르게 ≈ 12.4 N·m, 지지 무릎 ≤ 43° 굽힘). X13 의 0.7 배(17.5 N·m)보다 낮은 목표.
    시작 = W3S3d-15800 이어 학습(--resume). 기준은 문서(학습 전 고정)."""

    W_TSAT = -40.0          # 비주석 클래스 상수(설정 필드 아님 — env.yaml 에는 기록됨): w3_torque_sat 가중
    TSAT_THRESH = 0.6       # |τ|/τ_한계 문턱(서기 항과 같은 값)

    def __post_init__(self):
        super().__post_init__()
        from . import mdp_stage4 as _m4
        assert getattr(self.rewards, "w3_torque_sat", None) is None
        self.rewards.w3_torque_sat = RewTerm(func=_w3.w3_step_gated, weight=float(self.W_TSAT),
                                             params={"func": _m4.torque_saturation, "params": {"thresh": float(self.TSAT_THRESH)}})


@configclass
class Biped12HangStandW3S3tHEnvCfg(Biped12HangStandW3S3tEnvCfg):
    """W3S3 tH: W3S3t 와 같은 항, **가중만 2 배(−80 = 걷기 초당 −0.61, 걷기 순 보상의 47 %)** — 용량–반응의 둘째 팔(사전 등록 문서 §T2 의 조건에서만 돌린다).
    가만히 서는 STEP(순 −0.51/s)이 걷기보다 나아지는 가중은 약 −235 라 −80 도 그 아래다."""

    W_TSAT = -80.0


@configclass
class Biped12HangStandW3S3tDEnvCfg(Biped12HangStandW3S3dEnvCfg):
    """W3S3 tD (1007, TORQUE_W3T_DESIGN.md §T3): W3S3d 와 관측·행동·명령·DR·커리큘럼·학습 지연·보상 항·가중 전부 같고 **예정 스윙 창만** 좁힌다 —
    반주기 위상 v 의 (0.12, 0.88) → SWING_WIN (0.25, 0.75): 예정 한발 지지 76 % → 50 %(0.345 → 0.227 s), 예정 양발 지지 24 → 50 %.
    바뀌는 것은 두 보상 항의 `win` 인자뿐이다: w3_periodic(스윙 창 안 스윙발 하중 벌점)·w3_height(스윙 창 안 발 높이 5 cm 반사인 추종).
    걸음 주파수 1.1 Hz·클록 관측·리드 발 규칙·유효 스텝 판정(공중 ≥ 0.12 s·정점 ≥ 3 cm)·모방 창 (0.25, 0.75)은 그대로 → 배포 러너·합성 러너 변경 없음.
    물리 근거(reports/gen_w3/torque_phase.py bounds (C)): 한발 지지 0.76 → 0.50 이면 힙 외전 보폭 평균 8.7 → 5.7 N·m, 정적 요구의 절반 토크로 버틸 때 필요한
    몸통 롤 흔들림 8.5 → 3.7°(시간의 제곱), 무릎이 몸무게를 받치는 데 필요한 최소 한계 18.9 → 15.7 N·m(25 의 0.76 → 0.63 배 — X13 의 0.7 배가 물리적으로 가능해짐).
    알려진 위험(사전 명시, 추정): 스윙 0.227 s 안에 5 cm 를 들려면 무릎 명령 속도가 슬루 한계(4°/틱)에 닿을 수 있다 → 유효 스텝률·스윙 높이 회귀를 본다.
    시작 = W3S3d-15800 이어 학습(--resume). 기준은 문서(학습 전 고정)."""

    SWING_WIN = (0.25, 0.75)   # 비주석 클래스 상수: 반주기 위상 v 의 예정 스윙 창(w3_imit 창과 같은 값)

    def __post_init__(self):
        super().__post_init__()
        rw = self.rewards
        assert rw.w3_periodic.func is _w3.w3_periodic_contact and rw.w3_height.func is _w3.w3_swing_height
        _win = (float(self.SWING_WIN[0]), float(self.SWING_WIN[1]))
        rw.w3_periodic.params = dict(rw.w3_periodic.params or {}, win=_win)
        rw.w3_height.params = dict(rw.w3_height.params or {}, win=_win)


@configclass
class Biped12HangStandW3S3tKEnvCfg(Biped12HangStandW3S3dEnvCfg):
    """W3S3 tK (1007, TORQUE_W3T_DESIGN.md §T4): W3S3d 와 관측·행동·명령·DR·커리큘럼·학습 지연 1–5·기존 보상 전부 같고 **보상 항 하나만 추가** —
    `w3_knee_over` = 걷는 동안(ge 1) 접지한 다리의 무릎 굽힘이 KNEE_MAX_RAD(45°)를 넘는 초과분[rad](접지 다리 평균, 선형) × W_KNEE.
    기본자세(DEFAULT_POSE — KE 서기와 공유하는 배포 계약)는 그대로 두고, 걷는 중 지지 무릎만 행동 오프셋으로 펴게 하는 후보다(서기 ge 0 에서는 정확히 0).
    근거(X12 추적·정역학 모델, torque_phase.py): 걷기 지지 무릎 굽힘 중앙 64°[p10 61, p90 69] — KE 서기 55° 보다 깊다(지지 쪽 몸통 롤 11.8° 로 지지 힙이 약 3 cm 낮아지는 기하와 같은 방향; 원인 단정 아님),
    그 자세의 한발 지지 무릎 정적 요구 21–23.5 N·m(한계의 85–94 %). 45° 면 15.6 N·m, 골반 +4 cm. 한계 0.8 배에서 지지 무릎이 지지 반주기의 61 % 동안 한계에 붙는다.
    가중 근거: 현 걸음에서 이 값 평균 0.338 rad(19.4°, 명령별 0.313–0.360) → 가중 −1 = 걷기 초당 −0.34(걷기 순 보상 +1.28/s 의 26 % — W3S3t 의 −0.30 과 같은 급).
    걷기 보상에 지지 무릎을 묶는 항은 없다(knee_stance_flexion·base_height·leg_pose_deviation 은 서기 전용). w3_imit 는 스윙 다리만 채점해 현 걸음 시간표에서는 겹치지 않는다
    (스윙발이 모방 창 v .25–.75 안에서 착지하는 걸음이 되면 착지 다리의 참조 55–57° 와 12° 어긋난다 — W3S3tD 와 묶을 때 주의).
    알려진 부작용 후보(사전 명시): 힙 굴곡이 줄어 힙 외전 분담이 는다(준정적 환산 23.0 → 25.0 N·m), 자세가 높아져 균형 여유·착지 충격 흡수가 바뀐다.
    시작 = W3S3d-15800 이어 학습(--resume). 기준은 문서(학습 전 고정)."""

    W_KNEE = -1.0           # 비주석 클래스 상수: w3_knee_over 가중(초과 1 rad·초당)
    KNEE_MAX_RAD = 0.785    # 45°

    def __post_init__(self):
        super().__post_init__()
        assert getattr(self.rewards, "w3_knee_over", None) is None
        self.rewards.w3_knee_over = RewTerm(func=_w3.w3_stance_knee_over, weight=float(self.W_KNEE), params={"max_flex_rad": float(self.KNEE_MAX_RAD)})
