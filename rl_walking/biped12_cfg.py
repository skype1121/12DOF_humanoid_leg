"""biped12 로봇 ArticulationCfg — 실물 이식 제약을 시뮬에 내장.

핵심 결정 (근거: config/sim_dynamics.json sim2real_gain_experiment_2026_07_18):
- kd=5.0  : MIT 프로토콜 kd 상한. kd25로 학습하면 실물 전이 실패를 반복하므로
            처음부터 이 동역학에서 학습한다.
- kp=150.0: kd5 조합 실측에서 기립·트림 검증된 값 (kp60+ 기립, kp150 트림 직립 0.3°).
            실물 프로토콜 상한 500 이내라 그대로 이식 가능.
- tau=25Nm: AK70-10 실물 토크 한계 (URDF effort=0이므로 여기서 강제).
- armature=0.01: AK70-10 기어드 모터 반사 로터관성 추정치(미실측 — 공학적 추정).
            저 kd에서 물리 댐핑을 제공. DR로 0.5~1.5배 랜덤화해 오차 흡수.

관절 기본자세 = 검증된 스탠딩 중립(gait_static._neutral_targets):
  무릎 6° 굽힘(부호는 ANAT_SIGN), 발목 dorsi 트림 0.0244rad, 나머지 0.
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# 변환된 USD (biped12_base.urdf → convert_urdf.py --merge-joints 산출)
BIPED12_USD_PATH = "/home/ryu/humanoid_leg_test1/rl_walking/assets/usd/biped12.usd"

# 스탠딩 중립 자세 (rad) — sim_walking 검증값과 동일
BIPED12_DEFAULT_JOINT_POS = {
    "left_hip_f_joint": 0.0,
    "left_hip_a_joint": 0.0,
    "left_hip_r_joint": 0.0,
    "left_knee_joint": -0.10471975511965977,   # 무릎 6° 굽힘 (left은 -가 굽힘)
    "left_ankle_f_joint": 0.0244,              # dorsi 트림 (직립 밸런스 실측)
    "left_ankle_r_joint": 0.0,
    "right_hip_f_joint": 0.0,
    "right_hip_a_joint": 0.0,
    "right_hip_r_joint": 0.0,
    "right_knee_joint": 0.10471975511965977,   # right는 +가 굽힘
    "right_ankle_f_joint": -0.0244,
    "right_ankle_r_joint": 0.0,
}

BIPED12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BIPED12_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # hip_a 양다리 동시 내전 시 ±9.4°에서 다리끼리 충돌 —
            # 자기충돌 OFF로 학습하면 실물에서 다리가 부딪히는 정책이 나온다.
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # base 프레임 = Z-up, +X 전방 (URDF에 base 링크 주입으로 확보).
        # 직립 pelvis_z 실측 0.649 → 살짝 위에서 낙하 안착.
        pos=(0.0, 0.0, 0.66),
        joint_pos=dict(BIPED12_DEFAULT_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # AK70-10 ×10 (힙 6 + 무릎 2 + 발목피치 2): 감속 10:1
        # 피크 24.8Nm, MIT 속도범위 ±50rad/s (정격 32rad/s@48V)
        "ak70": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_f_joint"],
            effort_limit_sim=25.0,
            velocity_limit_sim=30.0,
            stiffness=150.0,            # kp — kd5 조합 실측 검증값 (프로토콜 Kp 0~500)
            damping=5.0,                # kd — MIT 프로토콜 상한 (이식 병목을 학습에 내장)
            armature=0.003,             # 반사관성 추정 (로터관성 미공개 — DR로 오차 흡수)
        ),
        # AK45-36 ×2 (발목롤): 감속 36:1 — 피크 24Nm이지만 속도가 1/8!
        # 정격 4.19rad/s, 무부하 5.45rad/s, MIT 범위 ±6rad/s (cubemars.com/product/AK45-36.html)
        # 반사관성 = 로터 181.9g·cm² × 36² = 0.0236 kg·m² (공식 스펙 기반)
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_r_joint"],
            effort_limit_sim=24.0,
            velocity_limit_sim=5.0,     # 이 관절로 빠른 측방 보상은 물리적으로 불가능
            stiffness=150.0,
            damping=5.0,
            armature=0.0236,
        ),
    },
)
