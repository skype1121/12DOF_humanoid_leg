#!/usr/bin/env python3
"""stage8 지연 옵션 전후 벤치 — 순수 파이썬 모의 CAN 버스 + 실제 ROS2 (sudo·vcan·실물 버스 불필요). 1006.

무엇을 재나 (브리지 틱 = 20 ms):
  ① 명령 → CAN 송신 지연 : 브리지 에뮬레이터가 목표를 발행한 시각 → 모의 모터가 그 목표 프레임을 받은 시각
  ② 관측 나이            : 브리지 틱 시각 − 그 틱이 보는 관절각(값)을 모터가 처음 응답한 시각
  ②b 표본 나이           : 브리지 틱 시각 − 그 틱이 쓰는 가장 최근 피드백 표본의 응답 시각
  ③ 루프 틱              : 틱 k 에 낸 목표가 틱 m 의 관측에 처음 보일 때 m − k
                           (이상 모터 = 받은 목표로 즉시 이동. 모터 물리 지연은 포함하지 않는다 —
                            순수 소프트웨어·전송 지연. 시뮬 '지연 0'도 m − k = 1 이 내재)

구성 (한 번에 변형 하나):
  - stage8 프로세스: 실제 Stage8TwelveAxisMitControlNode(rclpy) — CAN 버스만 python-can 'virtual'
    인터페이스로 교체(_get_or_open_can_bus 패치), 같은 프로세스의 모의 모터 12개 스레드가 응답.
    모의 버스 시간 모델: 프레임당 130 µs(1 Mbps·8바이트 표준 프레임) 직렬 점유 + 모터 응답 지연 100 µs
    (가정값 — 실측 아님). 핸드오프 상태 파일은 존재하지 않는 경로로 돌린다(실 파일 소비 방지).
  - 브리지 에뮬레이터 프로세스: 50 Hz 타이머, 최신 status 의 관절각 관측, 계산 시간 --compute-ms 모사,
    SET_JOINT_TARGETS 발행 (0.1° 계단 톱니 0–2.9°, 슬루 4°/틱 아래).
  - 격리 ROS 도메인(기본 87, 0 금지). 실물 채널(can0/can1)·젯슨·모터와 무관.

사용:
  python3 jetson/bench/latency_mock/latency_bench.py --variants base,fas,soc,fas_soc,fas_soc_100 \\
      --duration 20 --out <디렉터리>
  변형: base(옵션 없음=종전) · fas(--feedback-after-send) · soc(--send-on-command) ·
        fas_soc(둘 다) · fas_soc_100(둘 다 + --control-hz 100) · hz100(--control-hz 100 만)
"""
import argparse
import heapq
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VARIANT_ARGS = {
    "base": [],
    "fas": ["--feedback-after-send"],
    "soc": ["--send-on-command"],
    "fas_soc": ["--feedback-after-send", "--send-on-command"],
    "fas_soc_100": ["--feedback-after-send", "--send-on-command", "--control-hz", "100"],
    "hz100": ["--control-hz", "100"],
}
VCHAN = "latbench_virtual"
FRAME_WIRE_S = 130e-6
MOTOR_TURNAROUND_S = 100e-6
SPECIAL = {bytes([0xFF] * 7 + [x]) for x in (0xFC, 0xFD, 0xFE)}
STEP_DEG = 0.1
SAW_N = 30


# ----------------------------------------------------------------------------- stage8 측
def run_stage8(variant_args, motor_log_path, duration):
    import can
    sys.path.insert(0, str(REPO))
    from jetson.scripts import stage8_12axis_mit_control_node as s8
    from protocols.ak_mit_decoder import AK70_10_LIMITS

    s8.TAKEOVER_STATE_FILE = "/nonexistent/latbench_no_handoff.json"
    args, _ = s8.build_arg_parser().parse_known_args(variant_args)
    s8.apply_latency_args(args)
    s8.CAN_CHANNEL = "vcan_bench_unused"

    def open_virtual(self, can_module):
        if self.can_bus is None:
            self.can_bus = can_module.Bus(interface="virtual", channel=VCHAN, receive_own_messages=False)
        return self.can_bus
    s8.Stage8TwelveAxisMitControlCore._get_or_open_can_bus = open_virtual

    # 계측 전용: stage8 이 처리한 피드백 프레임의 '모터 응답 시각'(virtual 버스 송신 timestamp)을 status 에 덧붙인다.
    orig_handle = s8.Stage8TwelveAxisMitControlCore._handle_can_feedback_message
    orig_status = s8.Stage8TwelveAxisMitControlCore.get_status

    def handle_with_ts(self, message):
        ok = orig_handle(self, message)
        if ok:
            self.__dict__.setdefault("_bench_reply_ts", {})[int(message.arbitration_id)] = float(message.timestamp)
        return ok

    def status_with_ts(self):
        st = orig_status(self)
        st["bench_reply_ts"] = {str(k): v for k, v in self.__dict__.get("_bench_reply_ts", {}).items()}
        return st
    s8.Stage8TwelveAxisMitControlCore._handle_can_feedback_message = handle_with_ts
    s8.Stage8TwelveAxisMitControlCore.get_status = status_with_ts

    lim = AK70_10_LIMITS
    span = lim.position_max_rad - lim.position_min_rad
    abs_lim = s8.get_absolute_limits_deg()
    pos_raw = {}
    for joint in s8.JOINT_NAMES_12DOF:
        mid = s8.CAN_ID_BY_JOINT[joint]
        jl = abs_lim[joint]
        joint_mid = 0.5 * (jl["soft_min"] + jl["soft_max"])
        motor_deg = s8.STAND_ZERO_DEG_BY_JOINT[joint] + s8.DIRECTION_BY_JOINT[joint] * joint_mid
        pos_raw[mid] = int(round((math.radians(motor_deg) - lim.position_min_rad) / span * 65535))

    motor_bus = can.Bus(interface="virtual", channel=VCHAN, receive_own_messages=False)
    log = []           # (kind, t, mid, raw)  kind: rx=명령 수신, tx=응답 송신
    stop = threading.Event()
    heap = []
    heap_lock = threading.Condition()
    bus_free = [0.0]

    def receiver():
        while not stop.is_set():
            m = motor_bus.recv(timeout=0.05)
            if m is None:
                continue
            mid = int(m.arbitration_id)
            d = bytes(m.data)
            if mid not in pos_raw or len(d) != 8:
                continue
            now = time.monotonic()
            # 버스 직렬 점유: 명령 프레임 전송 완료 → 모터 처리 → 응답 프레임 전송 완료
            start = max(now, bus_free[0])
            cmd_done = start + FRAME_WIRE_S
            reply_done = cmd_done + MOTOR_TURNAROUND_S + FRAME_WIRE_S
            bus_free[0] = reply_done
            if d not in SPECIAL:
                pos_raw[mid] = (d[0] << 8) | d[1]
                log.append(("rx", time.time() + (cmd_done - now), mid, pos_raw[mid]))
            with heap_lock:
                heapq.heappush(heap, (reply_done, mid, pos_raw[mid]))
                heap_lock.notify()

    def replier():
        while not stop.is_set():
            with heap_lock:
                while not heap and not stop.is_set():
                    heap_lock.wait(0.05)
                if stop.is_set():
                    return
                due, mid, p = heap[0]
                wait = due - time.monotonic()
                if wait > 0:
                    heap_lock.wait(min(wait, 0.002))
                    continue
                heapq.heappop(heap)
            motor_bus.send(can.Message(arbitration_id=mid, is_extended_id=False,
                                       data=[mid, p >> 8, p & 0xFF, 0x7F, 0xF7, 0xFF, 30, 0]))
            log.append(("tx", time.time(), mid, p))

    threads = [threading.Thread(target=receiver, daemon=True), threading.Thread(target=replier, daemon=True)]
    for t in threads:
        t.start()

    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node = s8.Stage8TwelveAxisMitControlNode(rclpy, String)
    t_end = time.time() + duration
    try:
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(node.node, timeout_sec=0.05)
    finally:
        stop.set()
        with heap_lock:
            heap_lock.notify_all()
        st = node.core.get_status()
        json.dump({"log": log, "send_counts": st["send_counts"], "fb_after_send": st["fb_after_send"],
                   "latency_options": st["latency_options"]}, open(motor_log_path, "w"))
        node.destroy()
        rclpy.shutdown()
        motor_bus.shutdown()


# ----------------------------------------------------------------------------- 브리지 에뮬레이터
def run_bridge(out_path, stream_s, compute_ms, phase_drift_ms):
    import rclpy
    from std_msgs.msg import String
    sys.path.insert(0, str(REPO))
    from robot_runtime.dof12_mapping import JOINT_NAMES_12DOF
    CMD = "/humanoid/stage8_12axis_mit_command"
    ST = "/humanoid/stage8_12axis_mit_status"
    J0 = JOINT_NAMES_12DOF[0]
    rclpy.init()
    node = rclpy.create_node("latbench_bridge")
    pub = node.create_publisher(String, CMD, 10)
    state = {"st": None, "st_t": None, "armed": False}

    def on_st(m):
        try:
            d = json.loads(m.data)
        except Exception:
            return
        state["st"] = d
        state["st_t"] = time.time()
        state["armed"] = bool(d.get("armed"))
    node.create_subscription(String, ST, on_st, 10)

    def send(obj):
        pub.publish(String(data=json.dumps(obj)))

    def spin_for(s):
        t = time.time() + s
        while time.time() < t:
            rclpy.spin_once(node, timeout_sec=0.002)

    spin_for(3.0)
    send({"command": "SET_BASELINE_FROM_CURRENT_ALL"})
    spin_for(1.0)
    send({"command": "ARM_ALL"})
    spin_for(1.0)
    ticks = []
    k = [0]

    def tick():
        t = time.time()
        d = state["st"] or {}
        info = (d.get("joints") or {}).get(J0) or {}
        base = info.get("baseline_deg")
        act = info.get("latest_actual_deg")
        obs_rel = None if base is None or act is None else float(act) - float(base)
        if compute_ms > 0:   # 정책·체인 계산 모사 (busy-wait)
            t_c = time.perf_counter() + compute_ms / 1000.0
            while time.perf_counter() < t_c:
                pass
        v = STEP_DEG * (k[0] % SAW_N)
        t_pub = time.time()
        send({"command": "SET_JOINT_TARGETS", "targets": {j: v for j in JOINT_NAMES_12DOF},
              "seq": k[0] + 1, "sent_ts": t_pub, "src": "latbench"})
        reply_ts = (d.get("bench_reply_ts") or {}).get("1")
        ticks.append({"k": k[0], "t": t, "t_pub": t_pub, "v": v, "obs_rel": obs_rel, "base": base, "reply_ts": reply_ts,
                      "st_t": state["st_t"], "armed": state["armed"]})
        k[0] += 1

    # 위상 쓸기: 같은 머신의 두 20 ms 타이머는 위상이 기동 순간에 고정된다(세션마다 0–20 ms 중 하나).
    # 모든 위상을 표본하려고 브리지 주기를 +phase_drift_ms 만큼 어긋나게 둔다(20 s 에 약 2.5 회 한 바퀴).
    timer = node.create_timer(0.02 + phase_drift_ms / 1000.0, tick)
    spin_for(stream_s)
    node.destroy_timer(timer)
    send({"command": "STOP_ALL"})
    spin_for(0.3)
    json.dump({"ticks": ticks, "joint": J0}, open(out_path, "w"))
    node.destroy_node()
    rclpy.shutdown()


# ----------------------------------------------------------------------------- 분석
def analyze(drv_path, mot_path, motor_id=1):
    import numpy as np
    drv = json.load(open(drv_path))
    mot = json.load(open(mot_path))
    ticks = [t for t in drv["ticks"] if t["armed"] and t["base"] is not None]
    ticks = ticks[25:]          # 처음 0.5 s 제외
    base = ticks[0]["base"]
    rx = [(t, raw) for kind, t, mid, raw in mot["log"] if kind == "rx" and mid == motor_id]
    tx = [(t, raw) for kind, t, mid, raw in mot["log"] if kind == "tx" and mid == motor_id]
    span = 25.0

    def raw_to_rel(raw):
        return math.degrees(raw * span / 65535 - 12.5) - base

    rx_t = np.array([t for t, _ in rx])
    rx_v = np.array([raw_to_rel(r) for _, r in rx])
    tx_t = np.array([t for t, _ in tx])
    tx_v = np.array([raw_to_rel(r) for _, r in tx])
    tol = 0.035
    cmd2can, loop, age = [], [], []
    by_k = {t["k"]: t for t in ticks}
    for t in ticks:
        # ① 명령 → CAN: 발행 뒤 처음 그 값을 받은 시각
        i = np.searchsorted(rx_t, t["t_pub"])
        hit = np.nonzero(np.abs(rx_v[i:i + 400] - t["v"]) < tol)[0]
        if hit.size:
            cmd2can.append((rx_t[i + hit[0]] - t["t_pub"]) * 1000.0)
        # ③ 루프 틱: 이 값이 관측에 처음 보이는 틱
        for m in range(t["k"] + 1, t["k"] + 12):
            tm = by_k.get(m)
            if tm is None or tm["obs_rel"] is None:
                continue
            if abs(tm["obs_rel"] - t["v"]) < tol:
                loop.append(m - t["k"])
                break
        # ② 관측 나이: 이 틱 관측값을 모터가 응답한 가장 최근 시각(≤ 틱 시각)
        if t["obs_rel"] is not None:
            j = np.searchsorted(tx_t, t["t"]) - 1
            lo = max(0, j - 400)
            cand = np.nonzero(np.abs(tx_v[lo:j + 1] - t["obs_rel"]) < tol)[0]
            if cand.size:
                # 같은 값이 연속 응답된 구간의 '첫' 응답 = 그 값이 처음 측정된 시각
                c = lo + cand[-1]
                while c - 1 >= lo and abs(tx_v[c - 1] - t["obs_rel"]) < tol:
                    c -= 1
                age.append((t["t"] - tx_t[c]) * 1000.0)

    def stats(a):
        a = np.asarray(a, float)
        if a.size == 0:
            return None
        return {"n": int(a.size), "mean": round(float(a.mean()), 2), "p50": round(float(np.median(a)), 2),
                "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}
    # ②b 표본 나이: 관측에 쓰인 '가장 최근 피드백 표본'을 모터가 응답한 시각부터 틱까지.
    #     이상 모터(값 불변)에서는 ②가 같은 값 재송신을 못 가리므로, 실제 모터처럼 상태가 연속 변할 때의
    #     신선도는 이 값이 대변한다(100 Hz 재송신 효과는 여기서만 보인다).
    sample_age = [(t["t"] - t["reply_ts"]) * 1000.0 for t in ticks if t.get("reply_ts") is not None]
    tick_dt = np.diff([t["t"] for t in ticks]) * 1000.0
    # 모터 명령 공백(안전): 스트리밍 구간 + 브리지 정지(STOP_ALL) 뒤 2 s 까지 모터 1 이 받은 명령 프레임 간격
    t_lo, t_hi = ticks[0]["t"], ticks[-1]["t"] + 2.0
    rx_win = rx_t[(rx_t >= t_lo) & (rx_t <= t_hi)]
    rx_gap = np.diff(rx_win) * 1000.0 if rx_win.size > 1 else np.array([])
    lp = np.asarray(loop)
    return {"cmd_to_can_ms": stats(cmd2can), "obs_age_ms": stats(age), "sample_age_ms": stats(sample_age),
            "loop_ticks": {"n": int(lp.size), "mean": round(float(lp.mean()), 3) if lp.size else None,
                           "hist": {int(v): int((lp == v).sum()) for v in sorted(set(lp.tolist()))}},
            "bridge_tick_dt_ms": stats(tick_dt), "motor_rx_gap_ms": stats(rx_gap), "send_counts": mot["send_counts"],
            "fb_after_send": mot["fb_after_send"], "latency_options": mot["latency_options"],
            "motor_rx_rate_hz": round(len(rx) / (rx_t[-1] - rx_t[0]), 1) if len(rx) > 1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="base,fas,soc,fas_soc,fas_soc_100")
    ap.add_argument("--duration", type=float, default=20.0, help="스트리밍 시간 s")
    ap.add_argument("--compute-ms", type=float, default=1.0, help="브리지 틱 계산 시간 모사 ms")
    ap.add_argument("--phase-drift-ms", type=float, default=0.05,
                    help="브리지 주기 어긋남 ms (위상 쓸기). 0 = 기동 위상 고정(실물 세션 1회와 같음)")
    ap.add_argument("--domain", default="87")
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--file", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.domain in ("", "0"):
        sys.exit("격리 ROS 도메인만 (0 금지)")
    if args.role == "stage8":
        return run_stage8(VARIANT_ARGS[args.variant], args.file, args.duration + 8.0)
    if args.role == "bridge":
        return run_bridge(args.file, args.duration, args.compute_ms, args.phase_drift_ms)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ROS_DOMAIN_ID=str(args.domain))
    results = {}
    for v in args.variants.split(","):
        mot = out / f"motor_{v}.json"
        drv = out / f"bridge_{v}.json"
        common = [sys.executable, __file__, "--out", str(out), "--duration", str(args.duration),
                  "--compute-ms", str(args.compute_ms), "--phase-drift-ms", str(args.phase_drift_ms), "--domain", str(args.domain), "--variant", v]
        p8 = subprocess.Popen(common + ["--role", "stage8", "--file", str(mot)], env=env,
                              stdout=open(out / f"s8_{v}.log", "w"), stderr=subprocess.STDOUT)
        time.sleep(1.0)
        pb = subprocess.run(common + ["--role", "bridge", "--file", str(drv)], env=env,
                            stdout=open(out / f"br_{v}.log", "w"), stderr=subprocess.STDOUT)
        try:
            p8.wait(timeout=30)          # stage8 은 스스로 duration+8 s 뒤 종료
        except subprocess.TimeoutExpired:
            p8.terminate()               # 이 스크립트가 띄운 자식 PID 만 종료
            p8.wait(timeout=10)
        if pb.returncode != 0 or not mot.exists() or not drv.exists():
            results[v] = {"error": f"실행 실패 rc={pb.returncode} (로그 {out})"}
            continue
        results[v] = analyze(drv, mot)
        r = results[v]
        print(f"[{v}] 명령→CAN p50 {r['cmd_to_can_ms']['p50']} ms (평균 {r['cmd_to_can_ms']['mean']}, p95 "
              f"{r['cmd_to_can_ms']['p95']}) | 관측 나이 p50 {r['obs_age_ms']['p50']} ms (평균 "
              f"{r['obs_age_ms']['mean']}) | 표본 나이 평균 {r['sample_age_ms']['mean']} ms | 루프 틱 평균 {r['loop_ticks']['mean']} {r['loop_ticks']['hist']} | "
              f"송신 {r['send_counts']} | 모터 수신 {r['motor_rx_rate_hz']} Hz · 명령 공백 최대 "
              f"{r['motor_rx_gap_ms']['max']} ms", flush=True)
    json.dump(results, open(out / "summary.json", "w"), ensure_ascii=False, indent=1)
    print(f"요약: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
