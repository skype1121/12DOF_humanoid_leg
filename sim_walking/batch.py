"""파라미터 비교 배치: 매 구성마다 fresh 리셋 후 러너 실행, 요약 반환."""
import shutil


def run_config(overrides: dict, tag: str, duration_s: float = 12.0):
    """fresh 물리 리셋 → zero → FSM 실행 → 핵심 지표 반환."""
    from . import sim_session as SS
    from . import gait_fsm as GF
    from . import runner as RN
    s = SS.Session.attach(fresh_physics=True)
    s.zero(settle=100)
    fsm = GF.GaitFSM(GF.GaitParams(**overrides))
    m = RN.run_gait(s, fsm, duration_s=duration_s, tag=tag)
    return {k: m[k] for k in ("tag", "fell_at", "forward_progress_m",
                              "lateral_drift_m", "max_abs_lean_ap",
                              "max_abs_lean_lat", "final_yaw")}


V2_BASE = dict(warmup_cycles=1.0, rock_deg=8.0, rock_ankle_frac=0.6,
               period=1.6, step_hip_deg=5.0, hip_lift_deg=6.0,
               knee_lift_deg=28.0, swing_dur=0.20, lean_fwd_deg=0.0,
               push_deg=0.0, clear_deg=6.0)
