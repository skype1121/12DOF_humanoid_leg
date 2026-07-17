"""정적 4상 보행 상태기계 (순수 파이썬 — Isaac 의존 없음).

상태 사이클:  LEAN_L → STEP_R → LEAN_R → STEP_L → (반복)
  - LEAN_*: 양발 지지에서 체중을 지지발 쪽으로 천천히 이동·유지.
    체중이동 = hip_a 평행사변형(골반 평행이동) + ankle_r 전신 롤(기울기).
    이 로봇은 hip_a만으론(내전한계 13.9°) CoM이 한 발 위에 못 가므로
    롤 성분이 필수다.
  - STEP_*: 하중이 빠진 발을 실제로 들어 앞으로 옮긴다(스냅 아님).
    지지발 힙은 신전 방향으로 후퇴 = 몸을 앞으로 민다.
  - 진동 로킹이 없어 측방 공진을 원천 회피(전도 3회의 원인이었음).

runner의 측방 피드백과 협조: lat_ref(t)가 의도된 롤 각을 제공하고
피드백은 (측정 lean_lat − lat_ref)만 교정한다.
"""
import math
from dataclasses import dataclass, field

from .sim_session import ANAT_SIGN


def _ease(u):
    u = min(1.0, max(0.0, u))
    return 0.5 * (1.0 - math.cos(math.pi * u))


def _bump(u):
    u = min(1.0, max(0.0, u))
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * u))


@dataclass
class StaticGaitParams:
    lean_hip_rad: float = 0.22    # hip_a 평행사변형 진폭 [rad] (내전한계 13.9°=0.243 이내)
    lean_roll_deg: float = 6.0    # ankle_r 전신 롤 [deg]
    lean_dur: float = 1.2         # 체중이동 시간 [s]
    step_dur: float = 0.8         # 스윙 시간 [s]
    step_hip_deg: float = 8.0     # 힙 스윙 진폭 [deg] (스트라이드 결정)
    knee_lift_deg: float = 30.0   # 스윙 무릎 굽힘 [deg]
    hip_lift_deg: float = 8.0     # 스윙 힙 추가 굴곡 [deg]
    clear_deg: float = 8.0        # 스윙 발목 도르시 클리어런스 [deg]
    stance_knee_deg: float = 6.0  # 상시 무릎 굽힘 [deg]
    dorsi_trim_rad: float = 0.0244


class StaticGait:
    """상태: 0=LEAN_L, 1=STEP_R, 2=LEAN_R, 3=STEP_L

    num_steps 지정 시 해당 스텝 수를 마친 뒤 stop_dur 동안
    직립(중앙 체중, 힙 중립)으로 부드럽게 복귀·유지한다."""

    def __init__(self, p: StaticGaitParams = None,
                 num_steps: int = None, stop_dur: float = 1.5):
        self.p = p or StaticGaitParams()
        self.num_steps = num_steps
        self.stop_dur = stop_dur

    def _t_stop(self):
        """num_steps번째 스텝이 끝나는 시각 (STEP 상태 종료 시점)."""
        if not self.num_steps:
            return None
        p = self.p
        cyc = 2.0 * (p.lean_dur + p.step_dur)
        n_full = (self.num_steps - 1) // 2
        if self.num_steps % 2 == 1:      # 홀수 = 오른스텝 종료
            return n_full * cyc + p.lean_dur + p.step_dur
        return n_full * cyc + cyc        # 짝수 = 왼스텝 종료

    # ---------- 위상 ----------
    def _locate(self, t):
        p = self.p
        cyc = 2.0 * (p.lean_dur + p.step_dur)
        n = int(t // cyc)
        r = t - n * cyc
        bounds = [p.lean_dur, p.step_dur, p.lean_dur, p.step_dur]
        for si, d in enumerate(bounds):
            if r < d or si == 3:
                return n, si, min(1.0, r / d)
            r -= d
        return n, 3, 1.0

    # ---------- 측방 의도 롤 (runner 피드백 기준) ----------
    def _lean_dir(self, n, si, u):
        """현재 의도 lean 값 [-1..+1] (+1=왼쪽 완전 기울임)."""
        if si == 0:   # LEAN_L: (첫 사이클은 0에서) -1 → +1
            frm = -1.0 if n > 0 else 0.0
            return frm + (1.0 - frm) * _ease(u)
        if si == 1:
            return 1.0
        if si == 2:   # LEAN_R: +1 → -1
            return 1.0 - 2.0 * _ease(u)
        return -1.0

    def lat_ref(self, t):
        """의도된 pelvis 롤 각 [deg] (+왼쪽) — runner 측방 피드백 기준값."""
        ts = self._t_stop()
        if ts is not None and t >= ts:
            u = min(1.0, (t - ts) / self.stop_dur)
            n, si, uu = self._locate(ts - 1e-6)
            return self.p.lean_roll_deg * self._lean_dir(n, si, uu) * (1.0 - _ease(u))
        n, si, u = self._locate(t)
        return self.p.lean_roll_deg * self._lean_dir(n, si, u)

    # ---------- 시상면 힙 스케줄 ----------
    def _hip_deg(self, side, n, si, u):
        """상태별 힙 굴곡각 [deg] (해부학: +앞).

        v2: 지지다리 후퇴(몸 전진 견인)를 전부 '양발지지(LEAN)' 구간에 배치.
        한발지지 중 견인은 반작용 피치 스파이크(lean_ap ~18° → 장거리 전도)를
        유발했음(실측). 스윙 중 지지다리는 유지(hold)만 한다.
        오른다리: S1 스윙(-S→+S), S2(LEAN_R) +S→-S 견인, S3/S0 유지(-S).
        왼다리:  S3 스윙(-S→+S), S0(LEAN_L) +S→-S 견인, S1/S2 유지(-S).
        첫 사이클(n=0)은 0에서 시작해 첫 LEAN에서 견인."""
        S = self.p.step_hip_deg
        e = _ease(u)
        if side == "right":
            if si == 0:
                return (-S) if n > 0 else 0.0
            if si == 1:
                frm = -S if n > 0 else 0.0
                return frm + (S - frm) * e
            if si == 2:
                return S - 2.0 * S * e      # 양발지지 견인 +S→-S
            return -S                        # 유지
        else:
            if si == 0:
                if n > 0:
                    return S - 2.0 * S * e   # 양발지지 견인 +S→-S
                return -S * e                # 첫 사이클: 0→-S (양발지지)
            if si == 1:
                return -S                    # 유지 (첫 사이클 포함)
            if si == 2:
                return -S
            return -S + 2.0 * S * e

    # ---------- 타깃 ----------
    def targets(self, t):
        ts = self._t_stop()
        if ts is not None and t >= ts:
            # 정지 시퀀스: 롤/무릎/발목만 중립 복귀, hip_f는 착지 자세 유지
            # (벌린 발 그대로 힙을 0으로 강제하면 몸이 뒤로 끌려 전도 — 실측)
            u = min(1.0, (t - ts) / self.stop_dur)
            w = 1.0 - _ease(u)
            base = self._targets_raw(ts - 1e-6)
            neutral = self._neutral_targets()
            out = {k: neutral[k] + (base[k] - neutral[k]) * w for k in base}
            out["left_hip_f_joint"] = base["left_hip_f_joint"]
            out["right_hip_f_joint"] = base["right_hip_f_joint"]
            return out
        return self._targets_raw(t)

    def _neutral_targets(self):
        """직립 유지 자세 (트림 + 상시 무릎 굽힘)."""
        p = self.p
        d2r = math.pi / 180.0
        out = {}
        for side in ("left", "right"):
            out[f"{side}_hip_f_joint"] = 0.0
            out[f"{side}_hip_a_joint"] = 0.0
            out[f"{side}_hip_r_joint"] = 0.0
            out[f"{side}_knee_joint"] = ANAT_SIGN["knee_flexion"][side] \
                * p.stance_knee_deg * d2r
            out[f"{side}_ankle_f_joint"] = ANAT_SIGN["dorsiflexion"][side] \
                * p.dorsi_trim_rad
            out[f"{side}_ankle_r_joint"] = 0.0
        return out

    def _targets_raw(self, t):
        p = self.p
        d2r = math.pi / 180.0
        n, si, u = self._locate(t)
        out = {}

        # --- 측방: hip_a 평행사변형 + ankle_r 롤 ---
        ld = self._lean_dir(n, si, u)          # +1 = 왼쪽
        par = p.lean_hip_rad * ld              # raw +: 골반 왼쪽 이동(검증됨)
        roll = p.lean_roll_deg * d2r * ld      # 롤: 왼기울임 = L발 inversion, R발 eversion
        out["left_hip_a_joint"] = par
        out["right_hip_a_joint"] = par
        aL = -roll * ANAT_SIGN["eversion"]["left"]    # 왼발: inversion 방향
        aR = +roll * ANAT_SIGN["eversion"]["right"]   # 오른발: eversion 방향
        # 스윙 중인 발은 착지 대비 수평으로
        swing_side = "right" if si == 1 else ("left" if si == 3 else None)
        if swing_side == "left":
            aL *= (1.0 - _bump(u))
        elif swing_side == "right":
            aR *= (1.0 - _bump(u))
        out["left_ankle_r_joint"] = aL
        out["right_ankle_r_joint"] = aR

        # --- 시상면 ---
        for side in ("left", "right"):
            sgn_hip = ANAT_SIGN["hip_flexion"][side]
            sgn_knee = ANAT_SIGN["knee_flexion"][side]
            sgn_dorsi = ANAT_SIGN["dorsiflexion"][side]
            hip = self._hip_deg(side, n, si, u)
            knee = p.stance_knee_deg
            ankle_extra = 0.0
            if side == swing_side:
                hip += p.hip_lift_deg * _bump(u)
                knee += p.knee_lift_deg * _bump(u)
                ankle_extra = p.clear_deg * _bump(u)
            out[f"{side}_hip_f_joint"] = sgn_hip * hip * d2r
            out[f"{side}_knee_joint"] = sgn_knee * knee * d2r
            out[f"{side}_ankle_f_joint"] = sgn_dorsi * (p.dorsi_trim_rad
                                                        + ankle_extra * d2r)
            out[f"{side}_hip_r_joint"] = 0.0
        return out


def smoke_test():
    import json, os
    g = StaticGait()
    limits = json.load(open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "sim_joint_limits.json")))["limits_deg"]
    r2d = 180.0 / math.pi
    prev = None
    max_jump = {}
    viol = []
    for i in range(1600):
        t = i * 0.01
        tg = g.targets(t)
        if prev:
            for k, v in tg.items():
                max_jump[k] = max(max_jump.get(k, 0.0),
                                  abs(v - prev[k]) * r2d)
        for k, v in tg.items():
            lo, hi = limits[k]["lower"], limits[k]["upper"]
            if not (lo - 1e-6 <= v * r2d <= hi + 1e-6):
                viol.append((round(t, 2), k, round(v * r2d, 2)))
        prev = tg
    print("max jump/10ms:", {k: round(v, 2) for k, v in
                             sorted(max_jump.items(), key=lambda x: -x[1])[:4]})
    print("violations:", len(viol), viol[:4])
    return len(viol) == 0


if __name__ == "__main__":
    print("SMOKE:", "PASS" if smoke_test() else "FAIL")
