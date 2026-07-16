import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AKMitLimits:
    position_min_rad: float
    position_max_rad: float
    velocity_min_rad_s: float
    velocity_max_rad_s: float
    torque_min: float
    torque_max: float
    kp_min: float = 0.0
    kp_max: float = 500.0
    kd_min: float = 0.0
    kd_max: float = 5.0


@dataclass(frozen=True)
class AKMitFeedback:
    motor_id: int
    position_raw: int
    velocity_raw: int
    torque_raw: int
    position_rad: float
    position_deg: float
    velocity_rad_s: float
    torque_est: float
    temperature_raw: int
    error_raw: int
    error_ok: bool


AK70_10_LIMITS = AKMitLimits(
    position_min_rad=-12.5,
    position_max_rad=12.5,
    velocity_min_rad_s=-50.0,
    velocity_max_rad_s=50.0,
    torque_min=-25.0,
    torque_max=25.0,
)

# TODO: Add AK45-36 limits only after the exact model/range is confirmed.


def uint_to_float(x, x_min, x_max, bits):
    if not isinstance(bits, int) or bits <= 0:
        raise ValueError("bits must be positive int")
    if not isinstance(x, int) or isinstance(x, bool):
        raise ValueError("x must be int")
    max_uint = (1 << bits) - 1
    if x < 0 or x > max_uint:
        raise ValueError("x out of range for bit width")
    return x * (float(x_max) - float(x_min)) / max_uint + float(x_min)


def decode_ak_mit_feedback(data_bytes, limits=AK70_10_LIMITS):
    data = _normalize_data_bytes(data_bytes)

    motor_id = data[0]
    position_raw = (data[1] << 8) | data[2]
    velocity_raw = (data[3] << 4) | (data[4] >> 4)
    torque_raw = ((data[4] & 0x0F) << 8) | data[5]
    temperature_raw = data[6]
    error_raw = data[7]

    position_rad = uint_to_float(
        position_raw,
        limits.position_min_rad,
        limits.position_max_rad,
        16,
    )
    velocity_rad_s = uint_to_float(
        velocity_raw,
        limits.velocity_min_rad_s,
        limits.velocity_max_rad_s,
        12,
    )
    torque_est = uint_to_float(
        torque_raw,
        limits.torque_min,
        limits.torque_max,
        12,
    )

    return AKMitFeedback(
        motor_id=motor_id,
        position_raw=position_raw,
        velocity_raw=velocity_raw,
        torque_raw=torque_raw,
        position_rad=position_rad,
        position_deg=math.degrees(position_rad),
        velocity_rad_s=velocity_rad_s,
        torque_est=torque_est,
        temperature_raw=temperature_raw,
        error_raw=error_raw,
        error_ok=error_raw == 0,
    )


def decode_ak_mit_feedback_hex(hex_string, limits=AK70_10_LIMITS):
    if not isinstance(hex_string, str):
        raise ValueError("hex_string must be string")
    parts = hex_string.replace(",", " ").split()
    try:
        data = [int(part, 16) for part in parts]
    except ValueError as error:
        raise ValueError("hex_string contains non-hex byte") from error
    return decode_ak_mit_feedback(data, limits=limits)


def _normalize_data_bytes(data_bytes):
    if isinstance(data_bytes, bytes):
        data = list(data_bytes)
    elif isinstance(data_bytes, list):
        data = list(data_bytes)
    else:
        raise ValueError("data_bytes must be bytes or list[int]")

    if len(data) != 8:
        raise ValueError("feedback frame must contain 8 bytes")

    for value in data:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("all data bytes must be int")
        if value < 0 or value > 255:
            raise ValueError("data byte out of range")

    return data
