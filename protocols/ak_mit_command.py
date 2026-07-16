import math
from dataclasses import dataclass

from protocols.ak_mit_decoder import AK70_10_LIMITS


KP_DEFAULT = 30.0
KD_DEFAULT = 2.0
V_DES_DEFAULT = 0.0
TAU_FF_DEFAULT = 0.0

ID4_MOTOR_ID = 4
ID4_STANDARD_JOINT = "knee"

STANDARD_JOINT_TO_MOTOR_ID = {
    "hip_pitch": 1,
    "hip_roll": 2,
    "hip_yaw": 3,
    "knee": 4,
    "ankle_pitch": 5,
    "ankle_roll": 6,
}

JOINT_GAINS = {
    "hip_pitch": {"kp": 30.0, "kd": 2.0},
    "hip_roll": {"kp": 25.0, "kd": 2.0},
    "hip_yaw": {"kp": 20.0, "kd": 1.5},
    "knee": {"kp": 30.0, "kd": 2.0},
    "ankle_pitch": {"kp": 20.0, "kd": 1.5},
    "ankle_roll": {"kp": 15.0, "kd": 1.0},
}


@dataclass(frozen=True)
class AKMitCommandPreview:
    motor_id: int
    joint: str
    p_des_rad: float
    v_des_rad_s: float
    kp: float
    kd: float
    tau_ff: float
    frame_bytes: list
    frame_hex: str


def float_to_uint(value, value_min, value_max, bits):
    if bits <= 0:
        raise ValueError("bits must be positive")

    value = min(max(float(value), float(value_min)), float(value_max))
    span = float(value_max) - float(value_min)
    max_uint = (1 << int(bits)) - 1
    return int((value - float(value_min)) * max_uint / span)


def pack_ak_mit_command(
    p_des_rad,
    v_des_rad_s=V_DES_DEFAULT,
    kp=KP_DEFAULT,
    kd=KD_DEFAULT,
    tau_ff=TAU_FF_DEFAULT,
    limits=AK70_10_LIMITS,
):
    p_int = float_to_uint(p_des_rad, limits.position_min_rad, limits.position_max_rad, 16)
    v_int = float_to_uint(
        v_des_rad_s,
        limits.velocity_min_rad_s,
        limits.velocity_max_rad_s,
        12,
    )
    kp_int = float_to_uint(kp, limits.kp_min, limits.kp_max, 12)
    kd_int = float_to_uint(kd, limits.kd_min, limits.kd_max, 12)
    tau_int = float_to_uint(tau_ff, limits.torque_min, limits.torque_max, 12)

    return [
        (p_int >> 8) & 0xFF,
        p_int & 0xFF,
        (v_int >> 4) & 0xFF,
        ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F),
        kp_int & 0xFF,
        (kd_int >> 4) & 0xFF,
        ((kd_int & 0x0F) << 4) | ((tau_int >> 8) & 0x0F),
        tau_int & 0xFF,
    ]


def build_id4_knee_mit_preview(
    target_deg,
    kp=KP_DEFAULT,
    kd=KD_DEFAULT,
    v_des_rad_s=V_DES_DEFAULT,
    tau_ff=TAU_FF_DEFAULT,
):
    target_rad = math.radians(float(target_deg))
    frame_bytes = pack_ak_mit_command(
        target_rad,
        v_des_rad_s=v_des_rad_s,
        kp=kp,
        kd=kd,
        tau_ff=tau_ff,
    )
    frame_hex = " ".join(f"{value:02X}" for value in frame_bytes)
    return AKMitCommandPreview(
        motor_id=ID4_MOTOR_ID,
        joint=ID4_STANDARD_JOINT,
        p_des_rad=target_rad,
        v_des_rad_s=float(v_des_rad_s),
        kp=float(kp),
        kd=float(kd),
        tau_ff=float(tau_ff),
        frame_bytes=frame_bytes,
        frame_hex=frame_hex,
    )


def build_joint_mit_preview(
    joint,
    target_deg,
    motor_id=None,
    kp=None,
    kd=None,
    v_des_rad_s=V_DES_DEFAULT,
    tau_ff=TAU_FF_DEFAULT,
):
    if joint not in STANDARD_JOINT_TO_MOTOR_ID:
        raise ValueError("joint not allowed")

    if motor_id is None:
        motor_id = STANDARD_JOINT_TO_MOTOR_ID[joint]
    gains = JOINT_GAINS.get(joint, {})
    if kp is None:
        kp = gains.get("kp", KP_DEFAULT)
    if kd is None:
        kd = gains.get("kd", KD_DEFAULT)

    target_rad = math.radians(float(target_deg))
    frame_bytes = pack_ak_mit_command(
        target_rad,
        v_des_rad_s=v_des_rad_s,
        kp=kp,
        kd=kd,
        tau_ff=tau_ff,
    )
    frame_hex = " ".join(f"{value:02X}" for value in frame_bytes)
    return AKMitCommandPreview(
        motor_id=int(motor_id),
        joint=joint,
        p_des_rad=target_rad,
        v_des_rad_s=float(v_des_rad_s),
        kp=float(kp),
        kd=float(kd),
        tau_ff=float(tau_ff),
        frame_bytes=frame_bytes,
        frame_hex=frame_hex,
    )
