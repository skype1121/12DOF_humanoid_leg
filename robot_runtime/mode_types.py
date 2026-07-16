from enum import Enum


class RuntimeMode(str, Enum):
    SIM = "SIM"
    REAL = "REAL"
    SIM_TO_REAL = "SIM_TO_REAL"
    REAL_TO_SIM = "REAL_TO_SIM"


VALID_RUNTIME_MODES = tuple(mode.value for mode in RuntimeMode)
SIM_COMMAND_MODES = (RuntimeMode.SIM.value, RuntimeMode.SIM_TO_REAL.value)
REAL_COMMAND_MODES = (RuntimeMode.REAL.value, RuntimeMode.SIM_TO_REAL.value)


def normalize_mode(mode):
    value = mode.value if isinstance(mode, RuntimeMode) else str(mode)
    if value not in VALID_RUNTIME_MODES:
        raise ValueError(f"invalid runtime mode: {mode!r}")
    return value
