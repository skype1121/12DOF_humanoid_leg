"""leg_odometry 단위시험(CPU, Isaac 불필요): python3 rl_walking/tests_leg_odometry.py
1 URDF 상수 일치  2 합성 등속(지지발 고정·몸통 피치 진동·요 회전) 속도 복원  3 지지 없음 = 무효·창 제외
4 착지 첫 틱 제외  5 SettleJudge 최소 시간·유지·유효 비율  6 ContactImuJudge 양발 지속 요구
"""
import os, sys
import numpy as np
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_walking.deploy.leg_odometry import (LEG_CHAIN, LegOdometry, SettleJudge, ContactImuJudge, fk_sole, _rpy_batch)  # noqa: E402

URDF = os.path.join(os.path.dirname(__file__), "assets", "biped12_final3.urdf")
JN = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint", "left_hip_r_joint", "right_hip_r_joint",
      "left_knee_joint", "right_knee_joint", "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]


def t1_urdf():
    r = ET.parse(URDF).getroot()
    J = {j.get("name"): j for j in r.findall("joint")}
    for side in ("left", "right"):
        for name, xyz, ax in LEG_CHAIN[side]:
            j = J[name]
            assert np.allclose([float(v) for v in j.find("origin").get("xyz").split()], xyz, atol=1e-9), name
            assert np.allclose([float(v) for v in j.find("origin").get("rpy").split()], 0.0), name
            assert np.allclose([float(v) for v in j.find("axis").get("xyz").split()], ax), name
    b = J["base_to_pelvis"].find("origin")
    assert np.allclose([float(v) for v in b.get("rpy").split()], [np.pi / 2, 0, np.pi / 2]) and np.allclose([float(v) for v in b.get("xyz").split()], 0)
    print("1 URDF 상수 일치 OK")


def _ik_numeric(target_b, side, q0):
    """발바닥점을 몸통 프레임 target 에 두는 관절각(힙 f·a·r, 무릎, 발목 f 만 사용 — 수치 IK, 시험용)."""
    q = q0.copy()
    for _ in range(50):
        p, _ = fk_sole(q[None], side)
        e = target_b - p[0]
        if np.linalg.norm(e) < 1e-9:
            break
        Jm = np.zeros((3, 6))
        for k in range(5):
            dq = np.zeros(6); dq[k] = 1e-6
            Jm[:, k] = (fk_sole((q + dq)[None], side)[0][0] - p[0]) / 1e-6
        q = q + np.linalg.lstsq(Jm, e, rcond=None)[0]
    return q


def t2_const_velocity():
    """세계 고정 왼발, 몸통이 v=(0.12,−0.05) m/s 로 이동 + 피치 진동 + 요 회전. 이상 신호에서 v_bar 복원."""
    dt = 0.02; T = 48; v = np.array([0.12, -0.05])
    odo = LegOdometry(JN, n=1)
    q0 = np.array([0.2, 0, 0, -0.4, 0.2, 0.0]) * np.array([-1, 1, 1, -1, -1, 1])
    p_foot_w = None; yaw = 0.3; out = None
    for t in range(T):
        pitch = 0.05 * np.sin(2 * np.pi * 1.1 * t * dt); dpitch = 0.05 * 2 * np.pi * 1.1 * np.cos(2 * np.pi * 1.1 * t * dt)
        wz = 0.2; yaw_t = yaw + wz * t * dt
        R = _rpy_batch(np.zeros(1), np.array([pitch]), np.array([yaw_t]))[0]
        base = np.array([v[0] * t * dt, v[1] * t * dt, 0.55])
        if p_foot_w is None:
            p_foot_w = base + R @ fk_sole(q0[None], "left")[0][0]
        target_b = R.T @ (p_foot_w - base)
        qL = _ik_numeric(target_b, "left", q0); q0 = qL
        enc = np.zeros(12)
        for k, (nm, _, _) in enumerate(LEG_CHAIN["left"]):
            enc[JN.index(nm)] = qL[k]
        # 몸통 각속도(몸통 프레임): ω_w = (0, dpitch 를 요 회전된 y 축, wz) → R^T ω_w 근사 대신 정확히: R = Rz Ry, ω_w = ż + Rz ŷ θ̇
        Rz = _rpy_batch(np.zeros(1), np.zeros(1), np.array([yaw_t]))[0]
        w_w = np.array([0, 0, wz]) + Rz @ np.array([0, dpitch, 0])
        gyro = R.T @ w_w
        grav = R.T @ np.array([0, 0, -1.0])
        out = odo.step(enc[None], gyro[None], grav[None], np.array([[1, 0]]))
        assert np.linalg.norm(fk_sole(qL[None], "left")[0][0] - target_b) < 1e-4, ("IK 미수렴(시험 결함)", t, fk_sole(qL[None], "left")[0][0] - target_b)
    # 추정기 프레임은 요 0 에서 시작(참 요 0.3) → 참 속도를 −0.3 회전해 비교
    c, s = np.cos(-yaw), np.sin(-yaw); vt = np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])
    err = np.linalg.norm(out["v_bar"][0] - vt)
    assert err < 0.01, (out["v_bar"], vt, err)
    print(f"2 합성 등속 복원 OK (오차 {err*1000:.1f} mm/s)")


def t3_no_support():
    odo = LegOdometry(JN, n=2)
    enc = np.zeros((2, 12)); g = np.tile([0, 0, -1.0], (2, 1)); w = np.zeros((2, 3))
    for _ in range(3):
        o = odo.step(enc, w, g, np.array([[0, 0], [1, 1]]))
    assert (~o["valid"][0]) and o["valid"][1] and o["frac"][0] == 0, o
    print("3 지지 없음 = 무효 OK")


def t4_touchdown_first_tick():
    odo = LegOdometry(JN, n=1)
    g = np.array([[0, 0, -1.0]]); w = np.zeros((1, 3))
    odo.step(np.zeros((1, 12)), w, g, np.array([[0, 0]]))
    enc = np.zeros((1, 12)); enc[0, JN.index("left_knee_joint")] = 0.5      # 스윙 중 큰 이동
    o = odo.step(enc, w, g, np.array([[1, 0]]))                                # 착지 첫 틱
    assert not o["valid"][0], "착지 첫 틱은 무효여야 함"
    o = odo.step(enc, w, g, np.array([[1, 0]]))
    assert o["valid"][0] and np.allclose(o["v_inst"], 0), o
    print("4 착지 첫 틱 제외 OK")


def t5_settle_judge():
    j = SettleJudge(n=1, v_thr=0.07, hold_s=0.2, min_s=0.9, min_frac=0.5)
    slow = {"speed_bar": np.array([0.01]), "frac": np.array([1.0])}
    fired = [bool(j.step(slow, np.array([True]))[0]) for _ in range(60)]
    k = fired.index(True)
    assert abs((k + 1) * 0.02 - 0.9) < 1e-9, k                 # 최소 0.9 s(유지 0.2 는 그 안에 이미 충족)
    j2 = SettleJudge(n=1)
    lowfrac = {"speed_bar": np.array([0.0]), "frac": np.array([0.3])}
    assert not any(bool(j2.step(lowfrac, np.array([True]))[0]) for _ in range(100))
    j3 = SettleJudge(n=1)
    fast = {"speed_bar": np.array([0.2]), "frac": np.array([1.0])}
    for _ in range(60): j3.step(fast, np.array([True]))
    out = [bool(j3.step(slow, np.array([True]))[0]) for _ in range(12)]
    assert out.index(True) == 9, out                          # 0.2 s = 10 틱째
    assert not j3.step(slow, np.array([False]))[0]
    print("5 SettleJudge OK")


def t6_contact_imu_judge():
    j = ContactImuJudge(n=1, ds_s=0.5, w_thr=0.3, min_s=0.9)
    g = np.zeros((1, 3))
    alt = [bool(j.step(g, np.array([[t % 20 < 10, t % 20 >= 10]]), np.array([True]))[0]) for t in range(200)]
    assert not any(alt), "교대 걸음에서는 정착 판정이 나오면 안 됨"
    out = [bool(j.step(g, np.array([[1, 1]]), np.array([True]))[0]) for _ in range(40)]
    assert out.index(True) == 24, out                         # 양발 0.5 s = 25 틱째
    print("6 ContactImuJudge OK")


if __name__ == "__main__":
    t1_urdf(); t2_const_velocity(); t3_no_support(); t4_touchdown_first_tick(); t5_settle_judge(); t6_contact_imu_judge()
    print("전부 통과")
