"""IMU 장착 회전 보정 도구 — 2자세 중력 기반으로 R_base_imu를 산출.

배경: iAHRS가 몸통에 회전 장착되어 있어 수직 매달림 상태에서 IMU 프레임
투영중력이 (−1.00, +0.03, −0.00) — 즉 IMU +X축이 위를 향한다. RL 정책은
베이스 프레임(+X 전방, +Y 좌측, +Z 상방; 매달림 중력 = (0,0,−1)) 기준
투영중력·자이로를 소비하므로 상수 장착 회전 R_base_imu(v_base = R @ v_imu)를
구해 어댑터(imu_adapter_node.py)에서 적용해야 한다.

원리 (2자세 중력):
  A(hang): 수직 매달림 — 중력의 IMU 표현 g1 = R_base_imuᵀ @ (0,0,−1)
           → base_z_in_imu = −g1  (베이스 +Z축의 IMU 좌표)
  B(tilt): 베이스 +X(전방)를 아래로 10~30° 기울여 유지 — 베이스 프레임
           down벡터가 (sinθ, 0, −cosθ)로 이동하므로
           g2 = sinθ·base_x_in_imu − cosθ·base_z_in_imu
           → base_x_in_imu = normalize(g2의 base_z 직교 성분)  (sinθ>0라 부호 유지)
           base_y = base_z × base_x, 이후 x = y × z로 재직교화.
  R_base_imu = 행(row)이 [base_x, base_y, base_z]인 행렬. 검증: RRᵀ≈I, det=+1,
  R @ g1 ≈ (0,0,−1).

입력: /humanoid/imu/data (sensor_msgs/Imu, ~100Hz, RAW — 어댑터 미경유)
  orientation은 iAHRS RPY로 만든 world←imu 쿼터니언(필드 x,y,z,w).
  각 샘플의 중력 방향 = R_wiᵀ @ (0,0,−1) = −(R_wi 3행). |quat|−1 > 0.05면 기각.
  캡처 중 자이로 노름 ≥ 0.05 rad/s면 '정지 아님' 경고.

구조: 계산은 순수 함수(compute_mount_rotation 등)로 분리, rclpy는 캡처
함수 안에서만 lazy import → --selftest는 ROS 소싱 없는 로컬에서 그대로 돈다.

사용 (Jetson, ROS2 Humble 소싱 후):
  1) 로봇을 수직으로 매달고 정지:
     python3 imu_mount_calib.py --phase hang --sec 10
  2) 매달린 채 베이스 +X(전방)를 아래로 10~30° 기울여 유지:
     python3 imu_mount_calib.py --phase tilt --sec 10
     → /home/mama/rl_calib/imu_mount_calib.json 생성
  3) 다시 수직 매달림에서 확인 (베이스 투영중력 ≈ (0,0,−1)이어야 함):
     python3 imu_mount_calib.py --phase check
  ROS 없이 수학 검증: python3 imu_mount_calib.py --selftest
"""
import argparse
import json
import math
import os
import time

IN_TOPIC = "/humanoid/imu/data"
QOS_DEPTH = 50
DEFAULT_STATE = "/home/mama/rl_calib/imu_mount_calib_state.json"
DEFAULT_OUT = "/home/mama/rl_calib/imu_mount_calib.json"
QUAT_NORM_TOL = 0.05    # |‖q‖−1| 이 값 초과 샘플은 기각
GYRO_STILL_MAX = 0.05   # rad/s — 캡처 중 이 이상이면 '정지 아님' 경고
TILT_MIN_DEG = 5.0      # g1↔g2 사잇각 허용 범위 (미만/초과 시 중단)
TILT_MAX_DEG = 45.0


# ---------- 3벡터/3x3 순수 수학 (numpy 비의존) ----------

def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_norm(a):
    return math.sqrt(v_dot(a, a))


def v_normalize(a):
    n = v_norm(a)
    if n < 1e-12:
        raise ValueError("영벡터는 정규화 불가")
    return [a[0] / n, a[1] / n, a[2] / n]


def v_cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def mat_vec(r, v):
    return [v_dot(r[0], v), v_dot(r[1], v), v_dot(r[2], v)]


def mat_t(r):
    return [[r[j][i] for j in range(3)] for i in range(3)]


def mat_det(r):
    return (r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]))


def validate_rotation(r, eps=1e-6):
    """RRᵀ≈I 및 det=+1 검증 — 어긋나면 ValueError (한국어 사유 포함)."""
    for i in range(3):
        for j in range(3):
            rrt = v_dot(r[i], r[j])
            expect = 1.0 if i == j else 0.0
            if abs(rrt - expect) > eps:
                raise ValueError(
                    f"회전행렬 직교성 위반: (RRᵀ)[{i}][{j}]={rrt:+.9f} (허용 {eps})")
    d = mat_det(r)
    if abs(d - 1.0) > eps:
        raise ValueError(f"회전행렬 행렬식 이상: det={d:+.9f} (+1 필요 — 반사 의심)")


def gravity_from_quat_xyzw(x, y, z, w):
    """world←imu 쿼터니언(메시지 필드 순서 x,y,z,w) → IMU 프레임 중력 방향.

    g_imu = R_wiᵀ @ (0,0,−1) = −(R_wi의 3행). 단위 쿼터니언 전제
    (호출측에서 노름 검사 후 정규화해 넘길 것).
    """
    return [-(2.0 * (x * z - w * y)),
            -(2.0 * (y * z + w * x)),
            -(1.0 - 2.0 * (x * x + y * y))]


# ---------- 핵심: 2자세 → R_base_imu ----------

def compute_mount_rotation(g1, g2):
    """(g1: 매달림 중력, g2: +X 앞기울임 중력, 둘 다 IMU 프레임) → (R, tilt_deg).

    부호 유도: g2 = sinθ·base_x − cosθ·base_z 이므로 base_z 직교 성분이
    +sinθ·base_x (θ∈(0°,90°)에서 sinθ>0) → 그대로 정규화하면 base_x.
    (g2−g1의 직교 성분을 써도 동일 — g1 = −base_z라 base_z 성분만 다름.)
    사잇각이 [5°,45°] 밖이면 ValueError로 중단.
    """
    g1u = v_normalize(g1)
    g2u = v_normalize(g2)
    cos_t = max(-1.0, min(1.0, v_dot(g1u, g2u)))
    tilt_deg = math.degrees(math.acos(cos_t))
    if not (TILT_MIN_DEG <= tilt_deg <= TILT_MAX_DEG):
        raise ValueError(
            f"기울임 각도 {tilt_deg:.2f}° — 허용 범위 [{TILT_MIN_DEG:.0f}°,"
            f"{TILT_MAX_DEG:.0f}°] 밖. 베이스 +X(전방)를 아래로 10~30° 기울여"
            " 다시 캡처하세요.")
    base_z = [-c for c in g1u]
    proj = v_dot(g2u, base_z)                      # = −cosθ
    fwd = [g2u[i] - proj * base_z[i] for i in range(3)]  # = sinθ·base_x
    base_x = v_normalize(fwd)
    base_y = v_cross(base_z, base_x)               # 우수계: y = z × x
    base_x = v_cross(base_y, base_z)               # 재직교화: x = y × z
    r = [base_x, base_y, base_z]
    validate_rotation(r)
    return r, tilt_deg


def format_matrix(r):
    """3x3을 사람이 읽는 문자열로 (행 = base 축의 IMU 좌표)."""
    rows = []
    for name, row in zip(("base_x", "base_y", "base_z"), r):
        rows.append(f"  {name}_in_imu = [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")
    return "\n".join(rows)


# ---------- ROS 캡처 (rclpy lazy import) ----------

def capture_gravity(sec, label):
    """RAW IMU를 sec초 구독 → (평균 중력 단위벡터, 샘플수). ROS 필요.

    샘플별: |‖q‖−1| > 0.05 기각, 자이로 노름 추적. 종료 후 정지 위반/샘플
    산포 경고. 샘플 0개면 RuntimeError.
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import Imu

    samples = []          # 단위 중력벡터들
    gyro_norms = []
    rejected = [0]

    rclpy.init()
    node = Node("imu_mount_calib")

    def cb(msg):
        q = msg.orientation
        n = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
        if abs(n - 1.0) > QUAT_NORM_TOL:
            rejected[0] += 1
            return
        g = gravity_from_quat_xyzw(q.x / n, q.y / n, q.z / n, q.w / n)
        samples.append(v_normalize(g))
        av = msg.angular_velocity
        gyro_norms.append(math.sqrt(av.x * av.x + av.y * av.y + av.z * av.z))

    node.create_subscription(Imu, IN_TOPIC, cb, QoSProfile(depth=QOS_DEPTH))
    print(f"[{label}] {IN_TOPIC} 캡처 시작 — {sec:.0f}초간 자세 유지…")
    deadline = time.monotonic() + sec
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if not samples:
        raise RuntimeError(
            f"[{label}] 샘플 0개 — {IN_TOPIC} 수신 여부(humanoid_imu 노드 실행"
            f"/ROS_DOMAIN_ID)를 확인하세요. (기각 {rejected[0]}개)")
    if rejected[0]:
        print(f"[경고] 쿼터니언 노름 이상으로 {rejected[0]}개 샘플 기각")

    g_mean = v_normalize([sum(s[i] for s in samples) for i in range(3)])
    max_gyro = max(gyro_norms)
    if max_gyro >= GYRO_STILL_MAX:
        print(f"[경고] 캡처 중 자이로 최대 {max_gyro:.3f} rad/s ≥ "
              f"{GYRO_STILL_MAX} — 정지 상태가 아니었을 수 있음. 재캡처 권장.")
    # 샘플 산포: 평균벡터 대비 최대 사잇각
    max_dev = max(math.degrees(math.acos(max(-1.0, min(1.0, v_dot(s, g_mean)))))
                  for s in samples)
    print(f"[{label}] 샘플 {len(samples)}개, 자이로 max {max_gyro:.4f} rad/s, "
          f"산포 max {max_dev:.2f}°")
    if max_dev > 2.0:
        print(f"[경고] 중력 방향 산포 {max_dev:.2f}° > 2° — 자세가 흔들렸을 수"
              " 있음. 재캡처 권장.")
    print(f"[{label}] 평균 중력(IMU 프레임) = "
          f"({g_mean[0]:+.4f}, {g_mean[1]:+.4f}, {g_mean[2]:+.4f})")
    return g_mean, len(samples)


# ---------- 페이즈 ----------

def phase_hang(sec, state_path, date_str):
    """Phase A: 수직 매달림 중력 g1 캡처 → 상태 JSON 저장."""
    g1, n = capture_gravity(sec, "hang")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump({"g_hang_imu": g1, "n_samples": n, "date": date_str}, f, indent=2)
    print(f"[hang] 저장 완료: {state_path}")
    print("[hang] 다음 단계: 베이스 +X(전방)를 아래로 10~30° 기울여 유지한 뒤")
    print("        python3 imu_mount_calib.py --phase tilt --sec 10")
    return 0


def phase_tilt(sec, state_path, out_path, date_str):
    """Phase B: 앞기울임 중력 g2 캡처 + g1 로드 → R_base_imu 산출·저장."""
    if not os.path.isfile(state_path):
        print(f"[오류] 상태 파일 없음: {state_path} — 먼저 --phase hang을 실행하세요.")
        return 1
    with open(state_path) as f:
        state = json.load(f)
    g1 = [float(v) for v in state["g_hang_imu"]]

    g2, n = capture_gravity(sec, "tilt")
    try:
        r, tilt_deg = compute_mount_rotation(g1, g2)
    except ValueError as e:
        print(f"[중단] {e}")
        return 1

    # 새니티: R @ g1 ≈ (0,0,−1)
    g1_base = mat_vec(r, v_normalize(g1))
    err = v_norm([g1_base[0], g1_base[1], g1_base[2] + 1.0])
    print("R_base_imu (행 = base 축의 IMU 좌표):")
    print(format_matrix(r))
    print(f"기울임 각도: {tilt_deg:.2f}°  (허용 [{TILT_MIN_DEG:.0f}°,{TILT_MAX_DEG:.0f}°])")
    print(f"새니티 R@g1 = ({g1_base[0]:+.5f}, {g1_base[1]:+.5f}, "
          f"{g1_base[2]:+.5f})  — (0,0,−1) 대비 오차 {err:.2e}")
    if err > 1e-6:
        # g1은 base_z 구성에 그대로 쓰이므로 수학적으로 0이어야 정상
        print("[경고] R@g1 오차가 비정상적으로 큼 — 계산 과정 점검 필요")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "R_base_imu": r,
            "method": "two_pose_gravity_hang_tilt",
            "date": date_str,
            "g_hang_imu": v_normalize(g1),
            "g_tilt_imu": g2,
            "tilt_deg": tilt_deg,
            "n_samples_tilt": n,
        }, f, indent=2)
    print(f"[tilt] 보정 저장 완료: {out_path}")
    print("[tilt] 어댑터(imu_adapter_node.py) 재시작 시 자동 적용됩니다.")
    print("       확인: 수직 매달림에서 python3 imu_mount_calib.py --phase check")
    return 0


def phase_check(sec, out_path):
    """보정 로드 + 재캡처 → 베이스 프레임 투영중력 출력 (매달림≈(0,0,−1))."""
    if not os.path.isfile(out_path):
        print(f"[오류] 보정 파일 없음: {out_path} — hang→tilt를 먼저 완료하세요.")
        return 1
    with open(out_path) as f:
        calib = json.load(f)
    r = [[float(v) for v in row] for row in calib["R_base_imu"]]
    try:
        validate_rotation(r)
    except ValueError as e:
        print(f"[오류] 보정 파일의 R_base_imu 불량: {e}")
        return 1

    g_imu, _n = capture_gravity(sec, "check")
    g_base = mat_vec(r, g_imu)
    err_deg = math.degrees(math.acos(max(-1.0, min(1.0, -g_base[2]))))
    print(f"[check] 베이스 프레임 투영중력 = "
          f"({g_base[0]:+.4f}, {g_base[1]:+.4f}, {g_base[2]:+.4f})")
    print(f"[check] (0,0,−1) 대비 사잇각 {err_deg:.2f}° "
          f"(수직 매달림이라면 ≈0° 기대)")
    return 0


# ---------- 셀프테스트 (ROS 불필요) ----------

def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0], [0, c, -s], [0, s, c]]


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s], [0, 1, 0], [-s, 0, c]]


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def selftest():
    """ROS 없이 합성 데이터로 phase-B 수학 검증 — 실패 시 AssertionError 즉사."""
    # 예제 1: 실기와 같은 'IMU +X가 위' 장착 —
    #   base_x=(0,0,−1), base_y=(0,1,0), base_z=(1,0,0)  (IMU 좌표, det=+1)
    r_xup = [[0.0, 0.0, -1.0],
             [0.0, 1.0, 0.0],
             [1.0, 0.0, 0.0]]
    # 예제 2: 임의(고정 시드 대신 고정 각도 합성) 회전 — 난수 불필요
    r_rand = _mat_mul(_rot_z(0.31), _mat_mul(_rot_y(-0.62), _rot_x(1.05)))

    theta = math.radians(20.0)
    down_tilt_base = [math.sin(theta), 0.0, -math.cos(theta)]  # +X 아래로 20°

    for name, r_true in (("x-up", r_xup), ("random-ish", r_rand)):
        validate_rotation(r_true, eps=1e-12)
        rt = mat_t(r_true)
        g1 = mat_vec(rt, [0.0, 0.0, -1.0])       # 매달림 중력 (IMU)
        g2 = mat_vec(rt, down_tilt_base)          # 앞기울임 중력 (IMU)
        r_rec, tilt_deg = compute_mount_rotation(g1, g2)
        max_err = max(abs(r_rec[i][j] - r_true[i][j])
                      for i in range(3) for j in range(3))
        assert max_err < 1e-9, f"{name}: R 복원 오차 {max_err:.3e}"
        assert abs(tilt_deg - 20.0) < 1e-9, f"{name}: tilt {tilt_deg}"
        g1_base = mat_vec(r_rec, v_normalize(g1))
        assert (abs(g1_base[0]) < 1e-9 and abs(g1_base[1]) < 1e-9
                and abs(g1_base[2] + 1.0) < 1e-9), f"{name}: R@g1={g1_base}"
        print(f"[selftest] {name}: R 복원 max err {max_err:.2e}, "
              f"tilt {tilt_deg:.6f}° OK")

    # 예제 1 매달림 중력이 실측 (−1, +0.03, −0)과 부합하는지 (IMU +X가 위)
    rt = mat_t(r_xup)
    g1 = mat_vec(rt, [0.0, 0.0, -1.0])
    assert abs(g1[0] + 1.0) < 1e-12 and abs(g1[1]) < 1e-12 and abs(g1[2]) < 1e-12, g1
    print("[selftest] x-up 매달림 중력 = (−1,0,0) — 실측 (−1.00,+0.03,−0.00) 부합")

    # 기울임 각도 범위 밖 기각: 2° (과소), 60° (과대)
    for bad_deg in (2.0, 60.0):
        th = math.radians(bad_deg)
        g2_bad = mat_vec(rt, [math.sin(th), 0.0, -math.cos(th)])
        try:
            compute_mount_rotation(g1, g2_bad)
        except ValueError:
            print(f"[selftest] tilt {bad_deg:.0f}° 기각 OK")
        else:
            raise AssertionError(f"tilt {bad_deg}°가 기각되지 않음")

    print("[selftest] imu_mount_calib: 2자세 R 복원 + 범위 기각 ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="IMU 장착 회전(R_base_imu) 2자세 중력 보정 도구")
    ap.add_argument("--phase", choices=("hang", "tilt", "check"),
                    help="hang=수직 매달림 캡처, tilt=앞기울임 캡처+산출, "
                         "check=보정 적용 확인")
    ap.add_argument("--sec", type=float, default=None,
                    help="캡처 시간(초). 기본 hang/tilt=10, check=5")
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help=f"hang 결과(g1) 상태 JSON 경로 (기본 {DEFAULT_STATE})")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"최종 보정 JSON 경로 (기본 {DEFAULT_OUT})")
    ap.add_argument("--date", default=None,
                    help="결과 JSON date 필드 (기본: 현재 시각)")
    ap.add_argument("--selftest", action="store_true",
                    help="ROS 없이 합성 데이터로 수학 검증하고 종료")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if not a.phase:
        ap.error("--phase 또는 --selftest 중 하나가 필요합니다")

    date_str = a.date or time.strftime("%Y-%m-%d %H:%M:%S")
    sec = a.sec if a.sec is not None else (5.0 if a.phase == "check" else 10.0)

    if a.phase == "hang":
        return phase_hang(sec, a.state, date_str)
    if a.phase == "tilt":
        return phase_tilt(sec, a.state, a.out, date_str)
    return phase_check(sec, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
