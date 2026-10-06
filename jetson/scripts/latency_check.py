#!/usr/bin/env python3
"""stage8 지연 옵션 단계 적용 판정기 (1006) — **읽기 전용**: status 토픽 구독만, 명령·CAN 송신 없음.

/humanoid/stage8_12axis_mit_status (std_msgs/String JSON)를 N 초 구독해 다음을 집계하고
아래 CRITERIA(측정 전에 고정한 기준)로 PASS / FAIL / INCONCLUSIVE 를 낸다.

  ① 루프 타이밍·지터 : status.tick_timing(최근 250틱 링버퍼)을 겹치지 않게 표본 → 최악 p99·최대, 대역 밖 비율
  ② 송신 주기        : send_counts(timer·command·watchdog·hold) 증분 / 초 vs latency_options.control_hz
  ③ missed response : can_rx_count 증분 / (송신 틱 × 관절 수)  — 모든 모드
                       fb_after_send complete/ticks (송신 후 같은 틱 응답 완결률) — fas 모드만
  ④ stale response  : 관절별 feedback_stale 표본 수, 관절별 feedback_age 최대·p99
  ⑤ CAN error       : can_send_timeouts 증분, can_decode_error 표본 수, can_status 비정상 표본 수,
                       can_last_error 값, 새 errors
  ⑥ 실제 로드 옵션  : latency_options 가 --expect 와 같은지(재기동 누락 탐지), 측정 중 재기동 여부

판정이 '측정 불가'(송신 없음·표본 부족)면 INCONCLUSIVE — PASS 로 치지 않는다.

사용 (젯슨, ROS 환경):
  source /opt/ros/humble/setup.bash
  python3 jetson/scripts/latency_check.py --expect off --duration 60     # 1) 기준(옵션 없음)
  python3 jetson/scripts/latency_check.py --expect fas --duration 60     # 2) fas 적용 후
  결과 JSON: 기본 ~/logs/latency_check/latcheck_<expect>_<시각>.json + last_<expect>.json (--json 으로 경로 지정)
  종료 코드: 0 PASS · 1 FAIL · 2 INCONCLUSIVE · 3 실행 오류
  python3 jetson/scripts/latency_check.py --selftest                     # 판정 로직 자가 점검(ROS 불필요)
"""
import argparse
import json
import math
import os
import shutil
import socket
import sys
import time

STATUS_TOPIC = "/humanoid/stage8_12axis_mit_status"
SEND_SOURCES = ("timer", "command", "watchdog", "hold")
NORMAL_CAN_STATUS = {"OK", "BUS_OPEN", "BUS_OPEN_READ_ONLY", "LINK_UP", "LINK_UP_READ_ONLY",
                     "BRINGUP_OK", "BRINGUP_OK_READ_ONLY"}
EXPECT_OPTIONS = {"off": (False, False), "fas": (True, False), "fas,soc": (True, True)}

# --------------------------------------------------------------------------------------------
# 사전 판정 기준 (1006, 측정 전에 고정 — 결과를 보고 바꾸지 않는다. 바꾸면 버전·근거를 남긴다)
# --------------------------------------------------------------------------------------------
CRITERIA_VERSION = "lat_1006_v1"
CRITERIA = {
    "tick_p99_ms_max": 25.0,            # 틱 간격 p99 (50 Hz = 20 ms) — 창별 최악값
    "tick_max_ms_max": 45.0,            # 틱 간격 최대 (2 주기 + 여유) — 창별 최악값
    "tick_out_of_band_frac_max": 0.02,  # 20 ms ±5 ms 밖 틱 비율 (전체 가중 평균)
    "send_rate_rel_tol": 0.10,          # 송신 틱/초 가 control_hz 의 ±10 % 안
    "rx_per_send_min": 0.99,            # 응답 프레임 / (송신 틱 × 관절 수)
    "fas_complete_frac_min": 0.99,      # fas: 송신 후 같은 틱 안 응답 완결률
    "stale_samples_max": 0,             # 어느 관절이든 feedback_stale=True 인 status 표본 수
    "feedback_age_max_s": 0.10,         # 관절별 피드백 나이 최대 (5 틱)
    "can_send_timeouts_delta_max": 0,
    "can_decode_error_samples_max": 0,
    "can_status_abnormal_samples_max": 0,
    "new_errors_max": 0,                # 측정 중 새로 생긴 stage8 errors 항목 수
    "status_gap_max_s": 0.10,           # status 수신 공백 최대 (콘솔·브리지가 보는 신선도)
    "min_duration_s": 20.0,             # 이보다 짧으면 INCONCLUSIVE
    "min_send_ticks": 500,              # 송신 틱이 이보다 적으면 INCONCLUSIVE (홀드/armed 스트리밍 아님)
    "min_tick_windows": 2,              # 겹치지 않는 tick_timing 창 수 최소
}


def _q(xs, f):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(f * (len(s) - 1)))]


def _sends_total(st):
    sc = st.get("send_counts") or {}
    return sum(int(sc.get(k, 0) or 0) for k in SEND_SOURCES)


def _active_joints(st):
    """판정 대상 관절 — armed 면 베이스라인 관절, 인수 홀드면 홀드 관절 수만큼(이름 미노출 → 전 관절), 그 외 전 관절."""
    joints = st.get("joints") or {}
    if st.get("armed"):
        act = [j for j, v in joints.items() if v.get("baseline_deg") is not None]
        return act or list(joints)
    return list(joints)


def _frames_per_send(st):
    if st.get("armed"):
        n = st.get("baseline_count")
        if n:
            return int(n)
    tk = st.get("takeover") or {}
    if tk.get("holding") and tk.get("hold_joint_count"):
        return int(tk["hold_joint_count"])
    return int(st.get("joint_count") or 12)


def summarize(samples, expect=None):
    """samples = [(수신 monotonic 초, status dict)] — 시간순. 판정용 요약 dict 를 만든다(순수 함수)."""
    out = {"n_status": len(samples)}
    if len(samples) < 2:
        out["insufficient"] = "status 표본 2개 미만"
        return out
    t0, s0 = samples[0]
    t1, s1 = samples[-1]
    dur = t1 - t0
    out["duration_s"] = round(dur, 3)
    gaps = [b[0] - a[0] for a, b in zip(samples, samples[1:])]
    out["status_rate_hz"] = round((len(samples) - 1) / dur, 2) if dur > 0 else None
    out["status_gap_max_s"] = round(max(gaps), 4)

    # 재기동·옵션 변경 탐지
    restarts = 0
    for (_, a), (_, b) in zip(samples, samples[1:]):
        if (_sends_total(b) < _sends_total(a) or int(b.get("can_rx_count") or 0) < int(a.get("can_rx_count") or 0)
                or (b.get("latency_options") or {}) != (a.get("latency_options") or {})):
            restarts += 1
    out["restarts_or_option_changes"] = restarts
    lo = s1.get("latency_options") or {}
    out["latency_options"] = lo
    out["expect"] = expect
    if expect is not None:
        fas, soc = EXPECT_OPTIONS[expect]
        out["options_match"] = (bool(lo.get("feedback_after_send")) == fas and bool(lo.get("send_on_command")) == soc
                                and abs(float(lo.get("control_hz") or 0) - 50.0) < 1e-6)
    control_hz = float(lo.get("control_hz") or 50.0)
    out["control_hz"] = control_hz

    # 상태(측정 조건)
    tk = s1.get("takeover") or {}
    out["state"] = ("armed" if s1.get("armed") else "takeover_hold" if tk.get("holding")
                    else "waiting_takeover" if tk.get("waiting") else "idle")
    out["armed_frac"] = round(sum(1 for _, s in samples if s.get("armed")) / len(samples), 4)

    # ② 송신 주기
    sc0, sc1 = s0.get("send_counts") or {}, s1.get("send_counts") or {}
    d_src = {k: int(sc1.get(k, 0) or 0) - int(sc0.get(k, 0) or 0) for k in SEND_SOURCES}
    d_send = sum(d_src.values())
    out["send_ticks"] = d_send
    out["send_ticks_by_source"] = d_src
    out["send_rate_hz"] = round(d_send / dur, 3) if dur > 0 else None

    # ③ missed response
    d_rx = int(s1.get("can_rx_count") or 0) - int(s0.get("can_rx_count") or 0)
    fps = _frames_per_send(s1)
    out["frames_per_send"] = fps
    out["rx_frames"] = d_rx
    out["rx_per_send"] = round(d_rx / (d_send * fps), 5) if d_send > 0 else None
    fb0, fb1 = s0.get("fb_after_send") or {}, s1.get("fb_after_send") or {}
    d_fb_ticks = int(fb1.get("ticks", 0) or 0) - int(fb0.get("ticks", 0) or 0)
    d_fb_comp = int(fb1.get("complete", 0) or 0) - int(fb0.get("complete", 0) or 0)
    d_fb_rep = int(fb1.get("replies", 0) or 0) - int(fb0.get("replies", 0) or 0)
    out["fas_ticks"] = d_fb_ticks
    out["fas_complete_frac"] = round(d_fb_comp / d_fb_ticks, 5) if d_fb_ticks > 0 else None
    out["fas_replies_per_tick_frac"] = round(d_fb_rep / (d_fb_ticks * fps), 5) if d_fb_ticks > 0 else None
    out["fas_max_wait_ms"] = fb1.get("max_wait_ms")

    # ① 틱 타이밍 — 링버퍼(250틱) 창이 겹치지 않게 표본
    win_ticks = 250
    span_s = win_ticks / control_hz
    wins = []
    last_t = None
    for t, s in samples:
        tt = s.get("tick_timing") or {}
        if int(tt.get("n") or 0) < win_ticks:
            continue
        if last_t is None or t - last_t >= span_s:
            if t - t0 >= span_s * 0.98 or last_t is not None:   # 첫 창도 측정 구간 안의 틱이어야 함
                wins.append(tt)
                last_t = t
    if not wins:   # 측정 시간이 짧으면 마지막 스냅샷이라도
        tt = s1.get("tick_timing") or {}
        if int(tt.get("n") or 0) > 0:
            wins.append(tt)
    out["tick_windows"] = len(wins)
    if wins:
        ntot = sum(int(w["n"]) for w in wins)
        out["tick_mean_ms"] = round(sum(w["mean_ms"] * w["n"] for w in wins) / ntot, 3)
        out["tick_p50_ms_worst"] = max(w["p50_ms"] for w in wins)
        out["tick_p99_ms_worst"] = max(w["p99_ms"] for w in wins)
        out["tick_max_ms_worst"] = max(w["max_ms"] for w in wins)
        out["tick_out_of_band_frac"] = round(sum(w["out_of_band_frac"] * w["n"] for w in wins) / ntot, 5)
        out["tick_out_of_band_frac_worst"] = max(w["out_of_band_frac"] for w in wins)

    # ④ stale
    stale_samples = 0
    stale_joints = {}
    ages = {}
    for _, s in samples:
        js = s.get("joints") or {}
        any_stale = bool(s.get("feedback_stale"))
        for j in _active_joints(s):
            v = js.get(j) or {}
            if v.get("feedback_stale"):
                any_stale = True
                stale_joints[j] = stale_joints.get(j, 0) + 1
            a = v.get("feedback_age_sec")
            if a is not None:
                ages.setdefault(j, []).append(float(a))
        stale_samples += 1 if any_stale else 0
    out["stale_samples"] = stale_samples
    out["stale_by_joint"] = stale_joints
    allages = [a for v in ages.values() for a in v]
    out["feedback_age_max_s"] = round(max(allages), 4) if allages else None
    out["feedback_age_p99_s"] = round(_q(allages, 0.99), 4) if allages else None
    out["feedback_age_by_joint"] = {j: {"max": round(max(v), 4), "p99": round(_q(v, 0.99), 4)} for j, v in sorted(ages.items())}

    # ⑤ CAN error
    out["can_send_timeouts_delta"] = int(s1.get("can_send_timeouts") or 0) - int(s0.get("can_send_timeouts") or 0)
    dec = [s.get("can_decode_error") for _, s in samples if s.get("can_decode_error")]
    out["can_decode_error_samples"] = len(dec)
    out["can_decode_error_values"] = sorted(set(map(str, dec)))[:10]
    abn = [s.get("can_status") for _, s in samples if s.get("can_status") not in NORMAL_CAN_STATUS]
    out["can_status_abnormal_samples"] = len(abn)
    out["can_status_values"] = sorted(set(str(s.get("can_status")) for _, s in samples))
    out["can_last_error_values"] = sorted(set(str(s.get("can_last_error")) for _, s in samples if s.get("can_last_error")))[:10]
    out["can_rx_ignored_delta"] = int(s1.get("can_rx_ignored_count") or 0) - int(s0.get("can_rx_ignored_count") or 0)
    e0 = set(s0.get("errors") or [])
    new_err = sorted({e for _, s in samples for e in (s.get("errors") or []) if e not in e0})
    out["new_errors"] = new_err
    w0 = set(s0.get("warnings") or [])
    out["new_warnings"] = sorted({w for _, s in samples for w in (s.get("warnings") or []) if w not in w0})
    out["cmd_timing_last"] = s1.get("cmd_timing")
    return out


def judge(summary, criteria=CRITERIA):
    """요약 → (판정, 항목 목록). 항목 = (이름, 값, 기준 문자열, 'PASS'|'FAIL'|'N/A')."""
    c = criteria
    rows = []
    incon = []
    if summary.get("insufficient"):
        return "INCONCLUSIVE", [("표본", summary.get("n_status"), "≥ 2", "FAIL")], [summary["insufficient"]]

    def chk(name, val, ok, crit):
        rows.append((name, val, crit, "N/A" if val is None else ("PASS" if ok else "FAIL")))

    if summary["duration_s"] < c["min_duration_s"]:
        incon.append(f"측정 시간 {summary['duration_s']} s < {c['min_duration_s']} s")
    if summary["send_ticks"] < c["min_send_ticks"]:
        incon.append(f"송신 틱 {summary['send_ticks']} < {c['min_send_ticks']} (상태 {summary['state']} — 인수 홀드 또는 armed 스트리밍 중에 재야 함)")
    if summary.get("tick_windows", 0) < c["min_tick_windows"]:
        incon.append(f"tick_timing 창 {summary.get('tick_windows', 0)} < {c['min_tick_windows']}")

    if summary.get("expect") is not None:
        chk("실제 로드 옵션 = 기대", f"{summary['expect']} vs fas={summary['latency_options'].get('feedback_after_send')}"
            f",soc={summary['latency_options'].get('send_on_command')},hz={summary['latency_options'].get('control_hz')}",
            summary.get("options_match"), "일치 (50 Hz)")
    chk("측정 중 재기동/옵션 변경", summary["restarts_or_option_changes"], summary["restarts_or_option_changes"] == 0, "= 0")
    chk("틱 간격 p99 (최악 창) ms", summary.get("tick_p99_ms_worst"),
        summary.get("tick_p99_ms_worst") is not None and summary["tick_p99_ms_worst"] <= c["tick_p99_ms_max"], f"≤ {c['tick_p99_ms_max']}")
    chk("틱 간격 최대 (최악 창) ms", summary.get("tick_max_ms_worst"),
        summary.get("tick_max_ms_worst") is not None and summary["tick_max_ms_worst"] <= c["tick_max_ms_max"], f"≤ {c['tick_max_ms_max']}")
    chk("틱 대역 밖 비율 (±5 ms)", summary.get("tick_out_of_band_frac"),
        summary.get("tick_out_of_band_frac") is not None and summary["tick_out_of_band_frac"] <= c["tick_out_of_band_frac_max"],
        f"≤ {c['tick_out_of_band_frac_max']}")
    rate, hz = summary.get("send_rate_hz"), summary["control_hz"]
    chk("송신 틱/초", rate, rate is not None and abs(rate / hz - 1.0) <= c["send_rate_rel_tol"],
        f"{hz:g} ± {c['send_rate_rel_tol'] * 100:g} %")
    chk("응답/송신 (missed response)", summary.get("rx_per_send"),
        summary.get("rx_per_send") is not None and summary["rx_per_send"] >= c["rx_per_send_min"], f"≥ {c['rx_per_send_min']}")
    fas_on = bool(summary["latency_options"].get("feedback_after_send"))
    if fas_on:
        chk("fas 같은 틱 응답 완결률", summary.get("fas_complete_frac"),
            summary.get("fas_complete_frac") is not None and summary["fas_complete_frac"] >= c["fas_complete_frac_min"],
            f"≥ {c['fas_complete_frac_min']}")
    else:
        rows.append(("fas 같은 틱 응답 완결률", None, "fas 꺼짐", "N/A"))
    chk("stale 표본 수", summary["stale_samples"], summary["stale_samples"] <= c["stale_samples_max"], f"≤ {c['stale_samples_max']}")
    chk("관절 피드백 나이 최대 s", summary.get("feedback_age_max_s"),
        summary.get("feedback_age_max_s") is not None and summary["feedback_age_max_s"] <= c["feedback_age_max_s"],
        f"≤ {c['feedback_age_max_s']}")
    chk("CAN 송신 타임아웃 증분", summary["can_send_timeouts_delta"],
        summary["can_send_timeouts_delta"] <= c["can_send_timeouts_delta_max"], f"≤ {c['can_send_timeouts_delta_max']}")
    chk("CAN 디코드 오류 표본", summary["can_decode_error_samples"],
        summary["can_decode_error_samples"] <= c["can_decode_error_samples_max"], f"≤ {c['can_decode_error_samples_max']}")
    chk("CAN 상태 비정상 표본", summary["can_status_abnormal_samples"],
        summary["can_status_abnormal_samples"] <= c["can_status_abnormal_samples_max"], f"≤ {c['can_status_abnormal_samples_max']}")
    chk("새 stage8 errors", len(summary["new_errors"]), len(summary["new_errors"]) <= c["new_errors_max"], f"≤ {c['new_errors_max']}")
    chk("status 수신 공백 최대 s", summary["status_gap_max_s"], summary["status_gap_max_s"] <= c["status_gap_max_s"],
        f"≤ {c['status_gap_max_s']}")

    if any(r[3] == "FAIL" for r in rows):
        return "FAIL", rows, incon
    if incon or any(r[3] == "N/A" and r[2] != "fas 꺼짐" for r in rows):
        return "INCONCLUSIVE", rows, incon
    return "PASS", rows, incon


def format_report(verdict, rows, incon, summary):
    lines = [f"== stage8 지연 판정 ({CRITERIA_VERSION}) — 기대 옵션 {summary.get('expect')} · 상태 {summary.get('state')} · "
             f"{summary.get('duration_s')} s · status {summary.get('n_status')}개"]
    for name, val, crit, res in rows:
        lines.append(f"  [{res:4}] {name:28} {val!s:>24}   기준 {crit}")
    if summary.get("send_ticks_by_source"):
        lines.append(f"  참고: 송신 경로별 {summary['send_ticks_by_source']} · 응답 {summary.get('rx_frames')} 프레임 · "
                     f"fas 응답/틱 {summary.get('fas_replies_per_tick_frac')} · fas 최대 대기 {summary.get('fas_max_wait_ms')} ms")
        lines.append(f"  참고: 틱 평균 {summary.get('tick_mean_ms')} ms · p50(최악) {summary.get('tick_p50_ms_worst')} · "
                     f"창 {summary.get('tick_windows')} · 피드백 나이 p99 {summary.get('feedback_age_p99_s')} s · "
                     f"CAN 상태 {summary.get('can_status_values')} · last_error {summary.get('can_last_error_values')} · "
                     f"무시 프레임 증분 {summary.get('can_rx_ignored_delta')}")
        if summary.get("new_errors"):
            lines.append(f"  새 errors: {summary['new_errors']}")
        if summary.get("new_warnings"):
            lines.append(f"  새 warnings: {summary['new_warnings']}")
    for m in incon:
        lines.append(f"  판정 불가 사유: {m}")
    lines.append(f"판정: {verdict}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# ROS 구독 (읽기 전용)
# --------------------------------------------------------------------------------------------
def collect(duration, topic=STATUS_TOPIC, wait_first=10.0):
    import rclpy
    from std_msgs.msg import String
    rclpy.init()
    node = rclpy.create_node(f"latency_check_{os.getpid()}")
    samples = []

    def on_status(msg):
        try:
            samples.append((time.monotonic(), json.loads(msg.data)))
        except Exception:
            pass
    node.create_subscription(String, topic, on_status, 50)   # 구독만 — 퍼블리셔 없음
    try:
        t_wait = time.monotonic() + wait_first
        while not samples and time.monotonic() < t_wait:
            rclpy.spin_once(node, timeout_sec=0.05)
        if samples:
            t_end = samples[0][0] + duration
            while time.monotonic() < t_end:
                rclpy.spin_once(node, timeout_sec=0.01)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return samples


def save_json(path, payload, expect):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    last = os.path.join(os.path.dirname(os.path.abspath(path)), f"last_{expect.replace(',', '_')}.json")
    if os.path.abspath(path) != last:
        shutil.copyfile(path, last)
    return last


# --------------------------------------------------------------------------------------------
# selftest (ROS 불필요) — 판정 로직 자가 점검. 배포 스크립트 게이트에서 호출.
# --------------------------------------------------------------------------------------------
def synth_samples(duration=30.0, hz=50.0, fas=False, soc=False, status_hz=100.0, stale_at=None, timeouts_at=None,
                  missed_every=0, tick_p99=20.6, tick_max=22.0, oob=0.0, armed=True, restart_at=None, n_joints=12):
    """합성 status 열 — 단위 테스트·selftest 용."""
    joints = [f"j{i}" for i in range(n_joints)]
    out = []
    n = int(duration * status_hz)
    rx = 0
    for i in range(n):
        t = i / status_hz
        ticks = int(t * hz)
        if restart_at is not None and t >= restart_at:
            ticks = int((t - restart_at) * hz)
        missed = (ticks // missed_every) if missed_every else 0
        rx = ticks * n_joints - missed
        st = {
            "armed": armed, "baseline_count": n_joints if armed else 0, "joint_count": n_joints,
            "takeover": {"waiting": False, "holding": not armed, "hold_joint_count": 0 if armed else n_joints},
            "latency_options": {"feedback_after_send": fas, "send_on_command": soc, "control_hz": hz, "feedback_wait_ms": 4.0},
            "send_counts": {"timer": ticks if armed else 0, "command": 0, "watchdog": 0, "hold": 0 if armed else ticks},
            "fb_after_send": {"ticks": ticks if fas else 0, "complete": (ticks - missed) if fas else 0,
                              "replies": (ticks * n_joints - missed) if fas else 0, "max_wait_ms": 3.2 if fas else 0.0},
            "tick_timing": {"n": min(250, ticks), "mean_ms": 20.0, "p50_ms": 20.0, "p99_ms": tick_p99, "max_ms": tick_max,
                            "out_of_band_frac": oob},
            "can_rx_count": rx, "can_send_timeouts": 1 if (timeouts_at is not None and t >= timeouts_at) else 0,
            "can_decode_error": None, "can_status": "OK", "can_last_error": None, "can_rx_ignored_count": 0,
            "feedback_stale": False, "errors": [], "warnings": [],
            "joints": {j: {"baseline_deg": 0.0 if armed else None, "feedback_age_sec": 0.012,
                           "feedback_stale": bool(stale_at is not None and abs(t - stale_at) < 0.05)} for j in joints},
        }
        out.append((t, st))
    return out


def selftest():
    ok = True

    def expect(name, cond):
        nonlocal ok
        print(f"  {'OK ' if cond else 'BAD'} {name}")
        ok = ok and cond
    v, _, _ = judge(summarize(synth_samples(), "off"))
    expect("정상 base → PASS", v == "PASS")
    v, _, _ = judge(summarize(synth_samples(fas=True), "fas"))
    expect("정상 fas → PASS", v == "PASS")
    v, _, _ = judge(summarize(synth_samples(fas=False), "fas"))
    expect("옵션 불일치(재기동 누락) → FAIL", v == "FAIL")
    v, _, _ = judge(summarize(synth_samples(fas=True, missed_every=20), "fas"))
    expect("fas 응답 누락 5 % → FAIL", v == "FAIL")
    v, _, _ = judge(summarize(synth_samples(stale_at=10.0), "off"))
    expect("stale 1회 → FAIL", v == "FAIL")
    v, _, _ = judge(summarize(synth_samples(timeouts_at=12.0), "off"))
    expect("CAN 송신 타임아웃 증분 → FAIL", v == "FAIL")
    v, _, _ = judge(summarize(synth_samples(tick_p99=31.0), "off"))
    expect("틱 p99 31 ms → FAIL", v == "FAIL")
    v, _, _ = judge(summarize(synth_samples(oob=0.05), "off"))
    expect("대역 밖 5 % → FAIL", v == "FAIL")
    v, _, _ = judge(summarize(synth_samples(restart_at=15.0), "off"))
    expect("측정 중 재기동 → FAIL", v == "FAIL")
    v, _, _ = judge(summarize(synth_samples(duration=8.0), "off"))
    expect("8 s 측정 → INCONCLUSIVE", v == "INCONCLUSIVE")
    v, _, _ = judge(summarize(synth_samples(armed=False), "off"))
    expect("인수 홀드(hold 송신) 정상 → PASS", v == "PASS")
    v, _, _ = judge(summarize([], "off"))
    expect("표본 없음 → INCONCLUSIVE", v == "INCONCLUSIVE")
    print(f"latency_check selftest: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage8 지연 옵션 판정기 (읽기 전용 — status 구독만)")
    ap.add_argument("--expect", choices=sorted(EXPECT_OPTIONS), help="이번 측정에서 stage8 에 로드돼 있어야 할 옵션")
    ap.add_argument("--duration", type=float, default=60.0, help="구독 시간 s (기본 60)")
    ap.add_argument("--topic", default=STATUS_TOPIC)
    ap.add_argument("--json", default=None, help="결과 JSON 경로 (기본 ~/logs/latency_check/latcheck_<expect>_<시각>.json)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.expect:
        ap.error("--expect off|fas|fas,soc 가 필요하다 (실제 로드 옵션 대조)")
    try:
        samples = collect(a.duration, a.topic)
    except Exception as error:   # noqa: BLE001
        print(f"구독 실패: {error}", file=sys.stderr)
        return 3
    summary = summarize(samples, a.expect)
    verdict, rows, incon = judge(summary)
    print(format_report(verdict, rows, incon, summary), flush=True)
    if not a.no_save:
        path = a.json or os.path.expanduser(
            f"~/logs/latency_check/latcheck_{a.expect.replace(',', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.json")
        payload = {"tool": "latency_check", "criteria_version": CRITERIA_VERSION, "criteria": CRITERIA,
                   "verdict": verdict, "inconclusive_reasons": incon,
                   "rows": [{"name": r[0], "value": r[1], "criterion": r[2], "result": r[3]} for r in rows],
                   "summary": summary, "host": socket.gethostname(), "wall_time": time.time(),
                   "requested_duration_s": a.duration}
        last = save_json(path, payload, a.expect)
        print(f"결과: {path} (사본 {last})")
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
