"""결정론적 보행 궤적 생성기 (순수 파이썬 — Isaac 의존 없음, 오프라인 테스트 가능).

원리 (준정적+측방 로킹 하이브리드):
  - 위상 φ ∈ [0,1): 한 사이클 = 왼스텝 + 오른스텝.
  - 측방 로킹: ankle_r + hip_a 사인파로 CoM을 좌우로 흔들어
    스윙할 다리의 하중을 뺀다. rock>0 = 왼쪽으로 기울기(왼발 지지).
  - 스윙: 하중이 빠진 다리를 (무릎 굽힘 bump + 힙 굴곡 -step→+step)으로
    들어 앞으로 옮긴다. 오른스윙은 왼쪽으로 기운 φ=0.25 부근,
    왼스윙은 오른쪽으로 기운 φ=0.75 부근.
  - 스탠스: 힙 굴곡 +step→-step 선형 후퇴 = 몸을 앞으로 민다.
  - 발목_f: 직립 트림 + 힙 각도 보상으로 발바닥 수평 유지.

출력은 {관절이름: 목표각[rad]} — 부호는 joint_direction 실측 기준
(ANAT_SIGN 매핑으로 좌우 반전 자동 처리).
"""
import math
from dataclasses import dataclass, field

from .sim_session import ANAT_SIGN


def _ease(u):
    """0→1 코사인 이징."""
    u = min(1.0, max(0.0, u))
    return 0.5 * (1.0 - math.cos(math.pi * u))


def _bump(u):
    """0→1→0 부드러운 bump."""
    u = min(1.0, max(0.0, u))
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * u))


@dataclass
class GaitParams:
    period: float = 1.6          # 한 사이클(두 스텝) 시간 [s]
    rock_deg: float = 9.0        # 측방 로킹 진폭 [deg]
    rock_ankle_frac: float = 0.6 # 로킹 중 ankle_r 담당 비율(나머지 hip_a)
    step_hip_deg: float = 7.0    # 힙 굴곡/신전 스윙 진폭 [deg]
    knee_lift_deg: float = 28.0  # 스윙 무릎 굽힘 [deg]
    hip_lift_deg: float = 10.0   # 스윙 중 힙 추가 굴곡(발 들기 보조) [deg]
    stance_knee_deg: float = 6.0 # 상시 무릎 굽힘(충격흡수/클리어런스) [deg]
    dorsi_trim_rad: float = 0.0244  # 직립 트림 (config/sim_dynamics.json)
    swing_dur: float = 0.22      # 스윙 구간 길이 (위상 비율, 한 다리당)
    ankle_hip_comp: float = 1.0  # 발바닥 수평화: ankle_f -= comp*hip_f
    lean_fwd_deg: float = 0.0    # 상시 전방 기울기 바이어스(전진성 보조)
    warmup_cycles: float = 1.0   # 로킹만 하고 스텝 안 하는 워밍업 사이클 수
    push_deg: float = 8.0        # 스탠스 후기 푸시오프(까치발) [deg]
    push_window: float = 0.25    # 푸시오프 구간(스탠스 후반 비율)
    clear_deg: float = 6.0       # 스윙 중 추가 도르시(클리어런스) [deg]


@dataclass
class GaitFSM:
    p: GaitParams = field(default_factory=GaitParams)

    def _phase(self, t):
        return (t / self.p.period) % 1.0, t / self.p.period

    def _swing_u(self, phi, center):
        """스윙 정규화 진행도 u∈[0,1] (구간 밖이면 None)."""
        half = self.p.swing_dur / 2.0
        d = (phi - center) % 1.0
        if d > 0.5:
            d -= 1.0
        if -half <= d <= half:
            return (d + half) / self.p.swing_dur
        return None

    def _stance_u(self, phi, center):
        """스탠스 정규화 진행도 u∈[0,1] (착지=0, 이지=1)."""
        half = self.p.swing_dur / 2.0
        start = (center + half) % 1.0          # 스탠스 시작(착지)
        dur = 1.0 - self.p.swing_dur           # 스탠스 길이
        d = (phi - start) % 1.0
        return min(1.0, max(0.0, d / dur))

    def _stance_hip(self, phi, center):
        """스탠스 구간 힙 굴곡 [deg]: 착지(+step)→이지(-step) 선형."""
        u = self._stance_u(phi, center)
        return self.p.step_hip_deg * (1.0 - 2.0 * u)

    def targets(self, t):
        """시각 t[s] -> {관절이름: rad}"""
        p = self.p
        phi, cycles = self._phase(t)
        stepping = cycles >= p.warmup_cycles

        d2r = math.pi / 180.0
        out = {}

        # ---- 측방 로킹 (rock>0 = 왼쪽 기울기 = 왼발 지지) ----
        rock = p.rock_deg * math.sin(2.0 * math.pi * phi)
        ankle_rock = rock * p.rock_ankle_frac
        hip_rock = rock * (1.0 - p.rock_ankle_frac)
        # 왼쪽으로 기울기 = 양발 '왼쪽' 방향 회전.
        # eversion(+)은 발바닥 바깥기울임: 왼발 eversion+오른발 inversion = 몸 왼쪽 기움.
        # ANAT_SIGN eversion: left +, right + (양쪽 + = eversion).
        # 몸을 왼쪽으로 기울이려면: 왼발 inversion(-), 오른발 eversion(+)? -> 실측 검증되므로
        # 우선 좌우 반대 부호로 걸고 시뮬에서 부호 확인/보정한다.
        out["left_ankle_r_joint"] = -ankle_rock * d2r * ANAT_SIGN["eversion"]["left"]
        out["right_ankle_r_joint"] = +ankle_rock * d2r * ANAT_SIGN["eversion"]["right"]
        # hip_a: 왼쪽 기울기 = 왼힙 adduction + 오른힙 abduction 방향 조합(골반 평행이동)
        out["left_hip_a_joint"] = -hip_rock * d2r * ANAT_SIGN["hip_abduction"]["left"]
        out["right_hip_a_joint"] = +hip_rock * d2r * ANAT_SIGN["hip_abduction"]["right"]

        # ---- 시상면: 힙/무릎/발목 ----
        for side, sw_center in (("right", 0.25), ("left", 0.75)):
            sgn_hip = ANAT_SIGN["hip_flexion"][side]
            sgn_knee = ANAT_SIGN["knee_flexion"][side]
            sgn_dorsi = ANAT_SIGN["dorsiflexion"][side]

            knee_deg = p.stance_knee_deg
            ankle_extra_deg = 0.0    # +dorsi / -plantar (해부학 기준)
            if stepping:
                # 첫 스텝 사이클 동안 진폭 램프(불연속 방지)
                ramp = _ease(min(1.0, cycles - p.warmup_cycles))
                u = self._swing_u(phi, sw_center)
                if u is not None:      # 스윙
                    hip_deg = ramp * (-p.step_hip_deg + 2.0 * p.step_hip_deg * _ease(u)) \
                              + ramp * p.hip_lift_deg * _bump(u)
                    knee_deg = p.stance_knee_deg + ramp * p.knee_lift_deg * _bump(u)
                    # 클리어런스 + 푸시오프 잔량 블렌드(토우오프 연속성)
                    ankle_extra_deg = ramp * (p.clear_deg * _bump(u)
                        - p.push_deg * (1.0 - _ease(min(1.0, u / 0.25))))
                else:                  # 스탠스
                    hip_deg = ramp * self._stance_hip(phi, sw_center)
                    u_st = self._stance_u(phi, sw_center)
                    if u_st > 1.0 - p.push_window:  # 푸시오프(까치발)
                        u_p = (u_st - (1.0 - p.push_window)) / p.push_window
                        ankle_extra_deg = -ramp * p.push_deg * _ease(u_p)
            else:
                hip_deg = 0.0
            hip_deg += p.lean_fwd_deg

            hip_rad = hip_deg * d2r
            out[f"{side}_hip_f_joint"] = sgn_hip * hip_rad
            out[f"{side}_knee_joint"] = sgn_knee * knee_deg * d2r
            # 발목_f: 직립 트림 + 스윙 클리어런스 / 스탠스 푸시오프
            ankle_rad = p.dorsi_trim_rad + ankle_extra_deg * d2r
            out[f"{side}_ankle_f_joint"] = sgn_dorsi * ankle_rad
            # hip_r 미사용(0)
            out[f"{side}_hip_r_joint"] = 0.0

        return out


def smoke_test():
    """오프라인 검증: 궤적 연속성/리밋 준수 대략 확인."""
    import json, os
    fsm = GaitFSM(GaitParams())
    limits = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "sim_joint_limits.json")))["limits_deg"]
    r2d = 180.0 / math.pi
    prev = None
    max_step = {}
    viol = []
    for i in range(0, 1000):
        t = i * 0.01
        tg = fsm.targets(t)
        if prev:
            for k, v in tg.items():
                dv = abs(v - prev[k]) * r2d
                max_step[k] = max(max_step.get(k, 0.0), dv)
        for k, v in tg.items():
            lo, hi = limits[k]["lower"], limits[k]["upper"]
            if not (lo - 1e-6 <= v * r2d <= hi + 1e-6):
                viol.append((round(t, 2), k, round(v * r2d, 2), lo, hi))
        prev = tg
    print("max per-10ms jump (deg):",
          {k: round(v, 2) for k, v in sorted(max_step.items(), key=lambda x: -x[1])[:4]})
    print("limit violations:", len(viol), viol[:5])
    return len(viol) == 0


if __name__ == "__main__":
    ok = smoke_test()
    print("SMOKE:", "PASS" if ok else "FAIL")
