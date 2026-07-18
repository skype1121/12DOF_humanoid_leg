"""밸런스 피드백 (runner 블로킹 루프와 live_controller 콜백이 공유).

- 시상면: lean_ap → 양발목 대칭 도르시 보정 (P+I). 감도 실측 ~-168 deg/rad.
- 관상면: (lean_lat - lat_ref) → 양힙 hip_a 평행사변형 보정 (P+D).
- 요: yaw → 양힙 hip_r 보정 (P, 부호 실측 캘리브레이션: 음수).
기본 게인 = config/sim_walk_params.json 동결값.
"""
import numpy as np

from .sim_session import ANAT_SIGN

FPS = 60.0


class BalanceFeedback:
    def __init__(self, s,
                 bal_kp=0.010, bal_ki=0.03, bal_clamp=0.10,
                 lat_kp=0.012, lat_kd=0.010, lat_clamp=0.15,
                 yaw_kp=-0.008, yaw_clamp=0.12,
                 sample_every=2):
        self.s = s
        self.bal_kp, self.bal_ki, self.bal_clamp = bal_kp, bal_ki, bal_clamp
        self.lat_kp, self.lat_kd, self.lat_clamp = lat_kp, lat_kd, lat_clamp
        self.yaw_kp, self.yaw_clamp = yaw_kp, yaw_clamp
        self.sample_every = sample_every
        self.sgnL = ANAT_SIGN["dorsiflexion"]["left"]
        self.sgnR = ANAT_SIGN["dorsiflexion"]["right"]
        self.reset()

    def reset(self):
        self.frame = 0
        self.bal_i = 0.0
        self.lean_ap = 0.0
        self.lean_lat = 0.0
        self.lat_prev = None
        self.lat_rate = 0.0
        self.yaw = 0.0

    def apply(self, tg, lat_ref=0.0):
        """관절 목표 dict를 제자리 보정 후 반환 (매 프레임 호출)."""
        if self.frame % self.sample_every == 0:
            self.lean_ap, self.lean_lat = self.s.lean()
            dt = self.sample_every / FPS
            self.bal_i += self.bal_ki * self.lean_ap * dt
            self.bal_i = float(np.clip(self.bal_i, -self.bal_clamp, self.bal_clamp))
            if self.lat_prev is not None:
                self.lat_rate = (self.lean_lat - self.lat_prev) / dt
            self.lat_prev = self.lean_lat
            if self.yaw_kp:
                _, _, _, self.yaw = self.s.pelvis_pose()
        self.frame += 1

        corr = float(np.clip(self.bal_kp * self.lean_ap + self.bal_i,
                             -self.bal_clamp, self.bal_clamp))
        tg["left_ankle_f_joint"] = tg.get("left_ankle_f_joint", 0.0) + self.sgnL * corr
        tg["right_ankle_f_joint"] = tg.get("right_ankle_f_joint", 0.0) + self.sgnR * corr

        lcorr = -(self.lat_kp * (self.lean_lat - lat_ref) + self.lat_kd * self.lat_rate)
        lcorr = float(np.clip(lcorr, -self.lat_clamp, self.lat_clamp))
        tg["left_hip_a_joint"] = tg.get("left_hip_a_joint", 0.0) + lcorr
        tg["right_hip_a_joint"] = tg.get("right_hip_a_joint", 0.0) + lcorr

        if self.yaw_kp:
            ycorr = float(np.clip(self.yaw_kp * self.yaw, -self.yaw_clamp, self.yaw_clamp))
            tg["left_hip_r_joint"] = tg.get("left_hip_r_joint", 0.0) + ycorr
            tg["right_hip_r_joint"] = tg.get("right_hip_r_joint", 0.0) + ycorr
        return tg
