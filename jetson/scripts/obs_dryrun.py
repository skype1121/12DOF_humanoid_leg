"""관측 드라이런 — 매달린 로봇에서 센서→관측 경로 전수 점검 (읽기 전용, 2026-07-27).

목적: 정책(onnxruntime)을 돌리기 전에, 정책이 소비할 관측의 원재료 전부를
0토크로 검증한다 — 신규 direction/stand_zero 변환 통합 ①과 드라이런 ②를
정책 없이 한 번에.
  - 12관절: MIT 진입 후 전부-0 명령(kp=kd=tau=0) 폴링만 — 토크 출력 0이라
    모터는 절대 안 움직임 (zero_torque_read.py 승인 패턴). 모터각→관절각 변환 검증.
  - IMU(/humanoid/imu): quat → 몸통 프레임 중력 투영벡터 g_b = R(q)^T·(0,0,−1)
  - 발힘(/humanoid/foot_force): 접지 판정 (>5N). /humanoid/pressure/data 는
    채널 가시성용 수신 카운트만 (참고 지표).
  - 수신율/신선도 회계 → 최종 JSON 리포트 + PASS/FAIL 블록

확정 변환 규약 (2026-07-27 실측 캘리브, robot_12dof_hardware_map.json):
  입력  motor→joint: joint_rad = radians( DIR × (motor_deg − stand_zero_deg) )
  출력  joint→motor: motor_target_deg = stand_zero_deg + DIR × degrees(policy_rad)
  앵커: 좌무릎(모터4, DIR=−1, SZ=14.22)에 정책 −6°(−0.10472rad)
        → 14.22 + (−1)(−6.0) = 20.22
        (calib_data/sim_default_stand_targets_20260727.json "4": 20.22 와 일치)

주의:
  - AK45(모터 6·12): 위치 스케일 1.0 확정(폰 각도기 실측) — 위치 보정 없음.
    속도 디코드는 과대판독이라 이 도구는 어떤 모터의 feedback.velocity_rad_s 도
    소비하지 않는다 (qvel 필요 없음 — 정책 라이브 시엔 위치 유한차분만 사용).
  - 게인/위치 명령 절대 송신 안 함 — 제로게인 프레임 + ENTER/EXIT 만.
  - direction/stand_zero_deg 없는 구맵(스테일)이면 즉시 하드 실패
    (젯슨 사본이 실제로 구맵이었던 이력 — 동기화 확인 가드).
  - 어떤 경로로 끝나도 finally에서 12모터 MIT 해제. Ctrl+C 시 부분 리포트 저장.

사용 (젯슨):
  python3 obs_dryrun.py [--duration 45] [--channel can1] [--out r.json] [--no-ros]
로컬 자가검증 (CAN·ROS·로봇 불필요):
  python3 obs_dryrun.py --selftest
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, "/home/mama/rl_calib")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
RL_CALIB_DIR = "/home/mama/rl_calib"
#: 탐색 순서 = rl_bridge_node 와 동일(리포 우선, 젯슨 배포 폴백) — 드라이런이
#: 검증한 상수와 브리지가 실제 로드할 상수가 같은 파일이어야 한다. 두 사본이
#: 공존하면 load_hw_map 에서 DIR/SZ 일치까지 강제 검증.
HW_MAP_PATHS = [
    os.path.join(REPO, "config", "robot_12dof_hardware_map.json"),
    os.path.join(RL_CALIB_DIR, "robot_12dof_hardware_map.json"),
]
LIMITS_PATHS = [
    os.path.join(RL_CALIB_DIR, "joint_limits_12dof.json"),
    os.path.join(REPO, "config", "joint_limits_12dof.json"),
]

ENTER_MIT = bytes([0xFF] * 7 + [0xFC])
EXIT_MIT = bytes([0xFF] * 7 + [0xFD])
IDS = list(range(1, 13))
AK45_IDS = (6, 12)          # 속도 디코드 과대판독 — velocity 소비 금지 (이 도구는 전 모터 미사용)

CYCLE_S = 0.2               # ~5 사이클/s
POLL_TIMEOUT_S = 0.01       # 모터당 응답 대기 (12×10ms 최악에도 사이클 예산 내)
BUS_ABORT_CYCLES = 20       # 12모터 전멸 사이클 연속 한계 → 루프 중단 (버스오프 등)
CONTACT_N = 5.0             # 접지 판정 임계 (rl_bridge CONTACT_FORCE_THRESHOLD_N 동일)
QUAT_NORM_TOL = 0.05        # |q|−1 허용 — 초과 샘플은 소비 금지(센서 이상 신호)
GRAVITY_TOL = 0.35          # |g_b−(0,0,−1)| — 매달림·대략 수직 기준 (정보용 게이트)

IMU_TOPIC = "/humanoid/imu"
FOOT_TOPIC = "/humanoid/foot_force"
PRESSURE_TOPIC = "/humanoid/pressure/data"


# ---------------------------------------------------------------- 설정 로드
def load_first(paths):
    """존재하는 첫 경로의 JSON 로드 → (data, path). 전부 없으면 (None, None)."""
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), p
    return None, None


def parse_hw_map(data, path):
    """joints → {motor_id: (joint_name, DIR, SZ)}. 구맵(스테일)이면 하드 실패."""
    joints = data.get("joints") or {}
    conv = {}
    for jname, cfg in joints.items():
        if not isinstance(cfg, dict) or "motor_id" not in cfg:
            continue
        mid = int(cfg["motor_id"])
        if "direction" not in cfg or "stand_zero_deg" not in cfg:
            raise SystemExit(
                f"[FAIL] 하드웨어 맵 스테일: '{jname}' 에 direction/stand_zero_deg 없음\n"
                f"       경로: {path}\n"
                f"       젯슨 사본이 2026-07-27 캘리브 이전 구맵일 수 있음 — "
                f"동기화 후 재실행하세요")
        conv[mid] = (jname, int(cfg["direction"]), float(cfg["stand_zero_deg"]))
    missing = [m for m in IDS if m not in conv]
    if missing:
        raise SystemExit(
            f"[FAIL] 하드웨어 맵에 모터 {missing} 항목 없음 — 12모터 전부 필요 ({path})")
    return conv


def load_hw_map():
    existing = [p for p in HW_MAP_PATHS if os.path.exists(p)]
    if not existing:
        raise SystemExit("[FAIL] robot_12dof_hardware_map.json 찾지 못함:\n  "
                         + "\n  ".join(HW_MAP_PATHS))
    parsed = []
    for p in existing:
        with open(p) as f:
            parsed.append((parse_hw_map(json.load(f), p), p))
    # 사본 공존 시 DIR/SZ 일치 강제 (rl_bridge_node 와 동일 정책): 드라이런
    # PASS 가 다른 사본으로 뜨는 브리지를 보증하는 척하는 사고 방지
    conv0, path0 = parsed[0]
    for conv, p in parsed[1:]:
        diff = [m for m in IDS
                if conv[m][1] != conv0[m][1]
                or abs(conv[m][2] - conv0[m][2]) > 0.01]
        if diff:
            raise SystemExit(
                f"[FAIL] 하드웨어맵 사본 불일치 (모터 {diff}): direction/stand_zero_deg 가\n"
                f"  {path0}\n  {p}\n  에서 서로 다름 — 구맵을 동기화 후 재실행하세요")
    return conv0, path0


def load_soft_limits():
    """URDF 프레임 soft 리밋 {joint_name: (min,max)} — 없으면 (None,None): 검사 스킵."""
    data, path = load_first(LIMITS_PATHS)
    if data is None:
        return None, None
    lim = {j: (float(v["soft_min"]), float(v["soft_max"]))
           for j, v in data["limits_deg"].items()}
    return lim, path


# ---------------------------------------------------------------- 변환·수학
def motor_to_joint_deg(motor_deg, direction, stand_zero_deg):
    """입력변환: 모터 인코더각(deg) → URDF 관절각(deg)."""
    return direction * (motor_deg - stand_zero_deg)


def motor_target_deg(policy_rad, direction, stand_zero_deg):
    """출력변환: 정책 목표(rad, URDF 관절 프레임) → 모터 목표각(deg)."""
    return stand_zero_deg + direction * math.degrees(policy_rad)


def quat_to_gravity(quat_wxyz):
    """quat_wxyz(world←body) → 몸통 프레임 중력 단위벡터 g_b = R(q)^T·(0,0,−1).

    scipy 없이 수동 구현(numpy만). 정규화 후 사용하되 |q|가 1에서
    QUAT_NORM_TOL 넘게 벗어나면 None (센서 이상 — 소비 금지).
    부호 근거는 --selftest 케이스 참조 (rl_bridge_node.tick() 과 동일식).
    """
    q = np.asarray(quat_wxyz, dtype=float).reshape(-1)
    if q.shape != (4,) or not np.isfinite(q).all():
        return None
    n = float(np.linalg.norm(q))
    if abs(n - 1.0) > QUAT_NORM_TOL:
        return None
    w, x, y, z = (q / n).tolist()
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return r.T @ np.array([0.0, 0.0, -1.0])


# ---------------------------------------------------------------- 누적 통계
def _agg():
    return {"min": None, "max": None, "sum": 0.0, "n": 0}


def _acc(a, v):
    a["n"] += 1
    a["sum"] += v
    a["min"] = v if a["min"] is None else min(a["min"], v)
    a["max"] = v if a["max"] is None else max(a["max"], v)


def _agg_out(a, last=None):
    return {"mean": (a["sum"] / a["n"]) if a["n"] else None,
            "min": a["min"], "max": a["max"], "last": last}


class TopicStat:
    """토픽 도착 시각 회계 — 카운트·최대 공백·나이."""

    def __init__(self):
        self.n = 0
        self.t_last = None
        self.max_gap = 0.0

    def hit(self):
        t = time.time()
        if self.t_last is not None:
            self.max_gap = max(self.max_gap, t - self.t_last)
        self.t_last = t
        self.n += 1

    def age(self):
        return None if self.t_last is None else time.time() - self.t_last


class SensorState:
    """ROS 콜백 누적 상태 (rclpy 미설치 환경에서도 클래스 자체는 임포트 가능)."""

    def __init__(self):
        self.imu = TopicStat()
        self.foot = TopicStat()
        self.press = TopicStat()
        self.imu_parse_err = 0
        self.foot_parse_err = 0
        self.quat_bad = 0                 # 노름 이탈/형식 불량으로 거부된 quat 수
        self.grav_sum = np.zeros(3)
        self.grav_n = 0
        self.gyro_sq = 0.0
        self.gyro_n = 0
        self.last_grav = None
        self.last_gyro_norm = None
        self.last_foot = None
        self.fl = _agg()
        self.fr = _agg()
        self.contact_n = [0, 0]
        self.foot_n = 0

    def on_imu(self, text):
        self.imu.hit()
        try:
            d = json.loads(text)
            gyro = [float(v) for v in d["gyro_rad_s"]]
            quat = [float(v) for v in d["quat_wxyz"]]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.imu_parse_err += 1
            return
        g = quat_to_gravity(quat)
        if g is None:
            self.quat_bad += 1
        else:
            self.grav_sum += g
            self.grav_n += 1
            self.last_grav = tuple(float(v) for v in g)
        if len(gyro) == 3:
            gn = math.sqrt(gyro[0] ** 2 + gyro[1] ** 2 + gyro[2] ** 2)
            self.gyro_sq += gn * gn
            self.gyro_n += 1
            self.last_gyro_norm = gn

    def on_foot(self, text):
        self.foot.hit()
        try:
            d = json.loads(text)
            ln, rn = float(d["left_n"]), float(d["right_n"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.foot_parse_err += 1
            return
        self.last_foot = (ln, rn)
        self.foot_n += 1
        _acc(self.fl, ln)
        _acc(self.fr, rn)
        # 접지 판정 = [left_n>5.0, right_n>5.0]
        self.contact_n[0] += 1 if ln > CONTACT_N else 0
        self.contact_n[1] += 1 if rn > CONTACT_N else 0


def _fmt_age(a):
    return "없음" if a is None else f"{a:.2f}s"


# ---------------------------------------------------------------- 드라이런 본체
def run_dryrun(a):
    conv, hw_path = load_hw_map()
    limits, limits_path = load_soft_limits()
    if limits is None:
        print("[주의] joint_limits_12dof.json 없음 — soft 리밋 검사는 스킵됩니다",
              flush=True)
    out_path = a.out or time.strftime("/tmp/obs_dryrun_%Y%m%d_%H%M%S.json")

    # CAN 지연 임포트 — 셀프테스트가 로컬(라이브러리 없음)에서도 돌기 위함
    import can  # noqa: E402
    from ak_mit_command import pack_ak_mit_command  # noqa: E402
    from ak_mit_decoder import decode_ak_mit_feedback  # noqa: E402

    # ROS 지연 임포트 (--no-ros 면 완전 미사용)
    state = SensorState()
    ros = None
    if not a.no_ros:
        try:
            import rclpy  # noqa: E402
            from std_msgs.msg import String  # noqa: E402
        except ImportError as e:
            raise SystemExit(
                f"[FAIL] rclpy 임포트 실패 ({e}) — ROS 환경 source 후 재실행하거나 "
                f"--no-ros 로 CAN 전용 실행")
        rclpy.init()
        node = rclpy.create_node("obs_dryrun")
        node.create_subscription(String, IMU_TOPIC,
                                 lambda m: state.on_imu(m.data), 50)
        node.create_subscription(String, FOOT_TOPIC,
                                 lambda m: state.on_foot(m.data), 50)
        # 압력 원시 채널: 수신 카운트만 (타입이 String 아니면 0으로 보임 — 참고 지표)
        node.create_subscription(String, PRESSURE_TOPIC,
                                 lambda m: state.press.hit(), 50)
        ros = (rclpy, node)

    def ros_spin():
        """CAN 루프 사이 인터리브 — 단일 스레드, executor 불필요."""
        if ros is not None:
            ros[0].spin_once(ros[1], timeout_sec=0)

    bus = can.Bus(channel=a.channel, interface="socketcan")
    zero_frame = pack_ak_mit_command(0.0, 0.0, 0.0, 0.0, 0.0)  # 유일한 명령 프레임

    def tx(mid, data):
        bus.send(can.Message(arbitration_id=mid, data=data,
                             is_extended_id=False))

    def poll_once(mid, timeout=POLL_TIMEOUT_S):
        """0토크 명령 1회 송신 → 해당 모터 응답 디코드 (없으면 None).
        velocity_rad_s 는 절대 소비하지 않음 (AK45 과대판독 — 전 모터 미사용)."""
        tx(mid, zero_frame)
        t_e = time.time() + timeout
        while time.time() < t_e:
            msg = bus.recv(timeout=max(0.0, t_e - time.time()))
            if msg is None:
                return None
            if msg.is_error_frame or len(msg.data) < 6:
                continue
            if msg.data[0] == mid:
                return decode_ak_mit_feedback(bytes(msg.data))
        return None

    mstats = {m: {"miss": 0, "send_fail": 0, "seen": None,
                  "last_motor": None, "last_joint": None,
                  "motor": _agg(), "joint": _agg()} for m in IDS}
    enter_fail = []
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))

    cycles = 0
    consec_dead = 0
    abort = None
    t0 = time.time()

    def build_report():
        dur = max(1e-9, time.time() - t0)
        joints_rep = {}
        for m in IDS:
            st = mstats[m]
            jname, d, sz = conv[m]
            joints_rep[str(m)] = {
                "joint": jname, "direction": d, "stand_zero_deg": sz,
                "n": st["motor"]["n"], "miss": st["miss"],
                "send_fail": st["send_fail"],
                "seen_ratio": (st["motor"]["n"] / cycles) if cycles else 0.0,
                "last_seen_age_s": (None if st["seen"] is None
                                    else time.time() - st["seen"]),
                "motor_deg": _agg_out(st["motor"], st["last_motor"]),
                "joint_deg": _agg_out(st["joint"], st["last_joint"]),
            }
        can_ok = cycles > 0 and not enter_fail and all(
            mstats[m]["motor"]["n"] >= 0.9 * cycles for m in IDS)
        ros_on = ros is not None
        imu_rate = state.imu.n / dur if ros_on else None
        foot_rate = state.foot.n / dur if ros_on else None
        grav_mean = ((state.grav_sum / state.grav_n).tolist()
                     if state.grav_n else None)
        grav_err = (float(np.linalg.norm(
            np.array(grav_mean) - np.array([0.0, 0.0, -1.0])))
            if grav_mean is not None else None)
        gyro_rms = (math.sqrt(state.gyro_sq / state.gyro_n)
                    if state.gyro_n else None)
        soft_ok, soft_viol = None, {}
        if limits is not None:
            soft_ok = True
            for m in IDS:
                jname = conv[m][0]
                jd = mstats[m]["joint"]
                if jd["n"] == 0 or jname not in limits:
                    continue
                lo, hi = limits[jname]
                if jd["min"] < lo or jd["max"] > hi:
                    soft_ok = False
                    soft_viol[jname] = {"obs_min": jd["min"],
                                        "obs_max": jd["max"],
                                        "soft": [lo, hi]}
        checks = {
            "can_12of12": can_ok,
            "imu_rate_ge_50hz": (imu_rate >= 50.0) if ros_on else None,
            "imu_quat_norm_ok": ((state.imu.n > 0 and state.quat_bad == 0
                                  and state.imu_parse_err == 0)
                                 if ros_on else None),
            "foot_rate_ge_5hz": (foot_rate >= 5.0) if ros_on else None,
            "joints_within_soft_limits": soft_ok,
            "gravity_down_ok": ((grav_err < GRAVITY_TOL)
                                if grav_err is not None
                                else (False if ros_on else None)),
        }
        # gravity_down_ok 는 IMU 축정렬 점검용 정보 게이트 — 종합판정에서 제외
        gate = [v for k, v in checks.items()
                if k != "gravity_down_ok" and v is not None]
        overall = bool(gate) and all(gate) and abort is None
        return {
            "tool": "obs_dryrun",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": a.channel,
            "mode": "can_only" if a.no_ros else "full",
            "duration_req_s": a.duration,
            "duration_actual_s": dur,
            "cycles": cycles,
            "partial": bool(stop) or abort is not None,
            "abort": abort,
            "enter_fail_motors": enter_fail,
            "hardware_map": hw_path,
            "joint_limits": limits_path,
            "joints": joints_rep,
            "topics": {
                IMU_TOPIC: {"count": state.imu.n, "rate_hz": imu_rate,
                            "max_gap_s": state.imu.max_gap,
                            "last_age_s": state.imu.age(),
                            "parse_err": state.imu_parse_err},
                FOOT_TOPIC: {"count": state.foot.n, "rate_hz": foot_rate,
                             "max_gap_s": state.foot.max_gap,
                             "last_age_s": state.foot.age(),
                             "parse_err": state.foot_parse_err},
                PRESSURE_TOPIC: {"count": state.press.n,
                                 "max_gap_s": state.press.max_gap,
                                 "last_age_s": state.press.age(),
                                 "note": "채널 가시성 참고용 — 판정 미포함"},
            } if ros_on else {},
            "imu": {"gravity_mean": grav_mean,
                    "gravity_mean_norm_err": grav_err,
                    "gravity_samples": state.grav_n,
                    "quat_rejected": state.quat_bad,
                    "gyro_rms_rad_s": gyro_rms},
            "foot": {"left_n": _agg_out(state.fl),
                     "right_n": _agg_out(state.fr),
                     "contact_ratio": ([state.contact_n[0] / state.foot_n,
                                        state.contact_n[1] / state.foot_n]
                                       if state.foot_n else None),
                     "contact_threshold_n": CONTACT_N},
            "checks": checks,
            "soft_limit_violations": soft_viol,
            "gravity_note": ("정보용 — 매달림에서 로봇이 대략 수직이면 참. "
                             "FAIL은 IMU 축정렬 점검 신호일 뿐 종합판정 미포함"),
            "overall_pass": overall,
        }

    def save_report(rep):
        tmp = out_path + ".part"
        with open(tmp, "w") as f:
            json.dump(rep, f, indent=1, ensure_ascii=False)
        os.replace(tmp, out_path)

    print(f"[obs_dryrun] {a.duration:.0f}s 관측 드라이런 시작 — 채널 {a.channel}, "
          f"모드 {'CAN전용' if a.no_ros else 'CAN+ROS'}, 리포트 {out_path}", flush=True)
    print(f"  하드웨어 맵: {hw_path}", flush=True)

    t_show = t0
    t_save = t0
    try:
        for m in IDS:
            try:
                tx(m, ENTER_MIT)
            except Exception:
                enter_fail.append(m)
            time.sleep(0.005)
        if enter_fail:
            print(f"[주의] MIT 진입 송신 실패 모터: {enter_fail}", flush=True)
        time.sleep(0.05)

        t_end = t0 + a.duration
        got = 0
        while not stop and time.time() < t_end:
            t_cyc = time.time()
            got = 0
            for m in IDS:
                jname, d, sz = conv[m]
                st = mstats[m]
                try:
                    fb = poll_once(m)
                except Exception:   # 송신 예외(버스오프 등) — 결측 기록 후 다음 모터
                    st["miss"] += 1
                    st["send_fail"] += 1
                    ros_spin()
                    continue
                if fb is None:
                    st["miss"] += 1
                else:
                    got += 1
                    md = fb.position_deg          # velocity 는 의도적으로 미사용
                    jd = motor_to_joint_deg(md, d, sz)
                    st["last_motor"] = md
                    st["last_joint"] = jd
                    st["seen"] = time.time()
                    _acc(st["motor"], md)
                    _acc(st["joint"], jd)
                ros_spin()
            cycles += 1
            consec_dead = consec_dead + 1 if got == 0 else 0
            if consec_dead >= BUS_ABORT_CYCLES:
                abort = f"버스 이상 — {BUS_ABORT_CYCLES}사이클 연속 12모터 무응답"
                print(f"[ABORT] {abort}", flush=True)
                break

            now = time.time()
            if now - t_show >= 1.0:
                t_show = now
                parts = [f"[{now - t0:5.1f}s]"]
                seen = [(m, mstats[m]["last_joint"]) for m in IDS
                        if mstats[m]["last_joint"] is not None]
                if seen:
                    wm, wj = max(seen, key=lambda t: abs(t[1]))
                    parts.append(f"관절 max |{conv[wm][0]} {wj:+.2f}°| "
                                 f"(수신 {got}/12)")
                else:
                    parts.append(f"관절 수신 없음 ({got}/12)")
                if ros is not None:
                    g = state.last_grav
                    parts.append("g=({:+.2f},{:+.2f},{:+.2f})".format(*g)
                                 if g is not None else "g=미수신")
                    parts.append(f"|gyro|={state.last_gyro_norm:.3f}"
                                 if state.last_gyro_norm is not None
                                 else "gyro=미수신")
                    ft = state.last_foot
                    if ft is not None:
                        c = (("L" if ft[0] > CONTACT_N else "-")
                             + ("R" if ft[1] > CONTACT_N else "-"))
                        parts.append(f"발 L{ft[0]:.1f}/R{ft[1]:.1f}N 접지[{c}]")
                    else:
                        parts.append("발힘=미수신")
                    parts.append(f"나이 imu {_fmt_age(state.imu.age())} "
                                 f"발 {_fmt_age(state.foot.age())}")
                miss = {m: mstats[m]["miss"] for m in IDS if mstats[m]["miss"]}
                if miss:
                    parts.append(f"누적miss={miss}")
                print(" | ".join(parts), flush=True)

            if now - t_save >= 5.0:     # 크래시 대비 주기 자동저장
                t_save = now
                try:
                    save_report(build_report())
                except Exception:
                    pass

            while time.time() - t_cyc < CYCLE_S and not stop:
                ros_spin()
                time.sleep(0.005)
    except Exception as e:            # 예기치 못한 예외도 리포트는 남긴다
        abort = f"예외 {type(e).__name__}: {e}"
        print(f"[ABORT] {abort}", flush=True)
    finally:
        # 어떤 경로로 끝나도 12모터 MIT 해제 (모터별 try/except)
        for m in IDS:
            try:
                tx(m, EXIT_MIT)
            except Exception:
                pass
        try:
            bus.shutdown()
        except Exception:
            pass
        if ros is not None:
            try:
                ros[1].destroy_node()
                ros[0].shutdown()
            except Exception:
                pass

    rep = build_report()
    try:
        save_report(rep)
    except Exception as e:
        print(f"[주의] 리포트 저장 실패: {e}", flush=True)

    print("\n===== 관측 드라이런 판정 =====", flush=True)
    for k, v in rep["checks"].items():
        mark = "SKIP" if v is None else ("PASS" if v else "FAIL")
        note = " (정보용 — 종합판정 미포함)" if k == "gravity_down_ok" else ""
        print(f"  {k:28s} {mark}{note}", flush=True)
    if rep["soft_limit_violations"]:
        print(f"  soft 리밋 위반: {list(rep['soft_limit_violations'])}", flush=True)
    if rep["partial"]:
        print(f"  ※ 부분 기록 (중단: {abort or 'Ctrl+C'})", flush=True)
    print(f"종합: {'PASS' if rep['overall_pass'] else 'FAIL'} — 리포트 {out_path}",
          flush=True)
    return 0 if rep["overall_pass"] else 2


# ---------------------------------------------------------------- 셀프테스트
def run_selftest():
    """로컬 자가검증 — CAN·ROS·로봇 불필요. 수학·변환·가드만 검사."""
    # 1) quat→중력 투영 (부호를 케이스별로 손계산 근거와 함께 고정)
    cases = [
        # 항등(직립): 몸통=월드 → 중력 그대로 (0,0,−1)
        ("항등(직립)", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
        # +90° 롤(x축, w=cos45,x=sin45): 몸통 y축이 월드 +z를 향함
        # → 월드 −z(중력) = 몸통 −y 방향
        ("롤 +90°", (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0),
         (0.0, -1.0, 0.0)),
        # +90° 피치(y축): 몸통 x축이 월드 −z를 향함 → 중력 = 몸통 +x
        ("피치 +90°", (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0),
         (1.0, 0.0, 0.0)),
        # 180° 롤(뒤집힘): 중력 = 몸통 +z
        ("롤 180°", (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ]
    for name, q, exp in cases:
        g = quat_to_gravity(q)
        assert g is not None, f"{name}: 유효 quat 인데 거부됨"
        assert np.allclose(g, exp, atol=1e-9), f"{name}: {g} != {exp}"
        assert abs(float(np.linalg.norm(g)) - 1.0) < 1e-9, f"{name}: 비단위벡터"
    assert quat_to_gravity((2.0, 0.0, 0.0, 0.0)) is None, "|q|=2 미거부"
    assert quat_to_gravity((0.0, 0.0, 0.0, 0.0)) is None, "|q|=0 미거부"
    g = quat_to_gravity((1.04, 0.0, 0.0, 0.0))   # 허용오차 내 → 정규화 후 사용
    assert g is not None and np.allclose(g, (0, 0, -1), atol=1e-9), "정규화 실패"
    print("[PASS] quat→중력 4케이스 + 노름 거부(|q|=2,0)/정규화(|q|=1.04)")

    # 2) 출력변환 앵커 — 실제 하드웨어 맵 로드 (폴백 경로 검증 겸용)
    conv, hw_path = load_hw_map()
    dirs = [conv[m][1] for m in IDS]
    assert dirs == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1], \
        f"direction 실측값(모터12만 예외 +1)과 불일치: {dirs}"
    anchors = [  # (모터, 정책각 deg, 기대 모터목표 deg)
        (4, -6.0, 20.22),    # 좌무릎: 14.22 + (−1)(−6.0)
        (10, +6.0, 15.87),   # 우무릎: 21.87 + (−1)(+6.0)
        (5, +1.4, 35.68),    # 좌발목F: 34.28 + (+1)(+1.4)
        (11, -1.4, 4.21),    # 우발목F: 5.61 + (+1)(−1.4)
    ]
    for mid, pol_deg, exp in anchors:
        _, d, sz = conv[mid]
        got = motor_target_deg(math.radians(pol_deg), d, sz)
        assert abs(got - exp) < 1e-6, f"모터{mid}: {got:.6f} != {exp}"
    print(f"[PASS] 출력변환 앵커 4종 (맵: {hw_path})")

    # 기존 산출물과 교차검증 (파일이 있을 때만 — 젯슨/리포지토리 공통)
    ref = os.path.join(os.path.dirname(_SCRIPT_DIR), "calib_data",
                       "sim_default_stand_targets_20260727.json")
    if os.path.exists(ref):
        with open(ref) as f:
            refd = json.load(f)
        for mid, pol_deg, _ in anchors:
            _, d, sz = conv[mid]
            got = motor_target_deg(math.radians(pol_deg), d, sz)
            assert abs(got - float(refd[str(mid)])) < 0.005, \
                f"모터{mid}: {got:.4f} != 산출물 {refd[str(mid)]}"
        print("[PASS] sim_default_stand_targets_20260727.json 교차검증 (4앵커)")
    else:
        print("[SKIP] sim_default_stand_targets 산출물 없음 — 교차검증 생략")

    # 3) 입력변환 + 12관절 왕복 일치
    _, d4, sz4 = conv[4]
    jr = math.radians(motor_to_joint_deg(20.22, d4, sz4))
    assert abs(jr - math.radians(-6.0)) < 1e-9, f"입력변환: {jr} != −0.10472rad"
    for m in IDS:
        _, d, sz = conv[m]
        for md in (-33.3, 0.0, 17.71, sz):
            back = motor_target_deg(
                math.radians(motor_to_joint_deg(md, d, sz)), d, sz)
            assert abs(back - md) < 1e-9, f"모터{m} 왕복 불일치 {md}→{back}"
    print("[PASS] 입력변환(모터4 20.22°→−0.10472rad) + 12관절 왕복 일치")

    # 4) 스테일 맵 하드 실패 가드
    stale = {"joints": {"left_knee_joint": {"motor_id": 4, "kp": 30.0}}}
    try:
        parse_hw_map(stale, "<셀프테스트>")
        raise AssertionError("스테일 맵인데 통과 — 가드 불량")
    except SystemExit:
        pass
    print("[PASS] 스테일 맵(direction/stand_zero_deg 결측) 하드 실패 가드")
    print("[PASS] 셀프테스트 전체 통과 — 젯슨 배포 가능")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="관측 드라이런 — 0토크 폴링 + IMU/발힘으로 관측 경로 검증 (읽기 전용)")
    ap.add_argument("--duration", type=float, default=45.0, help="측정 시간(초)")
    ap.add_argument("--channel", default="can1")
    ap.add_argument("--out", default="",
                    help="JSON 리포트 경로 (기본: /tmp/obs_dryrun_<시각>.json)")
    ap.add_argument("--no-ros", action="store_true",
                    help="CAN 전용 모드 (IMU/발힘 없이 디버깅)")
    ap.add_argument("--selftest", action="store_true",
                    help="로컬 수학·변환 자가검증 (CAN·ROS·로봇 불필요)")
    a = ap.parse_args()
    if a.selftest:
        return run_selftest()
    return run_dryrun(a)


if __name__ == "__main__":
    raise SystemExit(main())
