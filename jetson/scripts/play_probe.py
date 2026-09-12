#!/usr/bin/env python3
"""무동력 관절 유격(백래시) 실측 보조 — 콘솔 `/api/state` 읽기 전용 (0913).

**왜 이 측정인가 (MEASURED 근거).**
- 0912 실기 5회: 몸통 롤이 5~8° 변하는 동안 **관절 엔코더로 설명되는 롤은 0.9~2.0°**
  뿐이었다 (`reports/real_prep_0913/analysis_0912_trials_v2.md` §3.1). 잔차 4~6° 는
  발이 기울었거나(패드 모서리·고무), 로터 엔코더와 링크 사이가 움직인 것(유격·구조
  컴플라이언스)이다. 어느 쪽인지는 **실측으로만** 갈린다.
- 같은 문서 §4.8(시뮬): 피치 관절 유격 2° 에서 v5 방출 후 생존 84~92 %, 3° 에서
  8~23 %. 오늘까지 시험한 실측 기반 요소 중 v5 를 크게 떨어뜨린 유일한 항목이다.
- 0912 밤 지그 재배치 뒤 로터 엔코더와 바닥 진실이 2~3° 어긋난 것이 관측됐다(§3.1·§4.8).
  그런데 **유격을 관절별로 재는 절차가 우리에게 없다.** 이 도구는 그 한 가지만 한다.

**무엇을 재는가 (정의).**
    유격[°] = 링크 각도 폭 − 엔코더 각도 폭
        링크 각도 폭 : 손으로 링크를 한쪽 끝까지 밀었다 반대로 미는 동안 폰 경사계로
                       읽은 각도의 폭 (양 끝 두 값의 차, 사람이 입력)
        엔코더 각도 폭: 같은 구간에서 `stage8.joints[<joint>].latest_actual_deg`
                       의 최대 − 최소 [deg, 모터 프레임]
    로터 엔코더는 감속기·구조 뒤쪽의 놀음을 못 보므로, 그 차이가 관절측 유격이다.
    부호 규약(direction·stand_zero)은 폭(절대값)에 영향을 주지 않으므로 변환하지 않는다.

**무엇을 재지 않는가 / 한계 (측정 사실과 해석의 구분).**
- 유격과 구조 컴플라이언스를 분리하지 못한다. 세게 밀수록 처짐이 섞인다 → **살짝** 민다.
- 폰 경사계는 링크의 *절대* 각도다. 부모 링크·로봇·지그가 같이 움직이면 그만큼 과대
  평가된다 → 다른 관절의 동반 이동과 IMU fwd/lat 변화를 같이 기록해 플래그로 남긴다.
- 하중 하 처짐(§3.1 잔차의 다른 후보)과 발 기울어짐은 이 도구의 측정 대상이 아니다.
- 여기서 나온 수치는 **실측**이고, 시뮬 생존율 표(§4.8)는 **시뮬 값**이다. 둘을 곱해
  실물 성공 확률로 쓰지 않는다.

**안전 (양보 불가).**
- 콘솔의 상태 엔드포인트를 **GET 으로만** 읽는다. 명령 엔드포인트는 호출하지 않고
  ROS·CAN 도 건드리지 않는다. 모터 명령이 나갈 코드 경로가 없다(--selftest 가 검사).
- 무동력 조건: stage8 DISARMED + 브리지 정책 정지. 둘 중 하나라도 켜져 있으면 측정을
  시작하지 않고, 캡처 중에 켜지면 그 측정을 **무효로 기록**한다(값을 조용히 버리지 않는다).
- stage8 의 읽기전용 피드백 폴링은 **한 번도 ARM/인수홀드 송신을 하지 않은 기동**에서만
  돈다(노드의 `_ever_streamed` 래치). ARM 이력이 있으면 엔코더 값이 얼어 있으므로 이
  도구는 '엔코더 갱신 없음' 으로 무효 처리한다. 재기동은 콘솔 절차(운영자)로 한다.
- 12관절 라운드로빈 요청 주기 0.02 s → **관절당 피드백 갱신 ≈0.24 s**. 그래서 나이
  한계 기본값은 0.5 s 다(브리지의 0.2 s 게이트를 완화한 것이 아니라, 읽기전용 폴링의
  실제 갱신 주기에 맞춘 이 도구 전용 계측 기준).

**절차 (관절 1개당).**
  1) 폰 경사계를 측정 링크(그 관절 **바로 아래** 링크)에 평평히 올린다. 부모 링크는 손으로 고정.
  2) 링크를 한쪽 끝까지 살짝 밀어 붙인 채 Enter → 홀드 표본 수집 → 폰 각도 입력
  3) 반대쪽 끝까지 살짝 밀어 붙인 채 Enter → 폰 각도 입력
  4) 유격 = 링크 폭 − 엔코더 폭 이 바로 출력되고 JSON 에 저장된다(관절별·반복별).

**사용.**
  # 젯슨 콘솔(8800)이 떠 있는 상태에서 — 젯슨 접속·실행은 운영자
  python3 play_probe.py --host 172.16.35.89 --out ~/logs/play_0913.json
  python3 play_probe.py --host 172.16.35.89 --joint left_knee_joint --repeat 2
  python3 play_probe.py --host 172.16.35.89 --monitor 20      # 측정 없이 상태만 확인
  python3 play_probe.py --report ~/logs/play_0913.json        # 표만 다시 출력(네트워크 없음)
  python3 play_probe.py --selftest                            # 계산 로직 검증(로컬)
"""
import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
import unicodedata
import urllib.request

# ─────────────────────────────── 계약·상수 ───────────────────────────────

#: 관절명·순서는 배포 계약(rl_walking/deploy/policy_runner.py JOINT_ORDER, 좌우 인터리브)과
#: 같게 둔다. 이 도구는 순서에 의존하지 않지만 다른 산출물과 나란히 읽기 위해 맞춘다.
JOINT_NAMES = [
    "left_hip_f_joint", "right_hip_f_joint",
    "left_hip_a_joint", "right_hip_a_joint",
    "left_hip_r_joint", "right_hip_r_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_f_joint", "right_ankle_f_joint",
    "left_ankle_r_joint", "right_ankle_r_joint",
]

#: 그룹 — 시뮬 유격 배터리(§4.8·§4.10)가 피치/횡을 나눠 쟀으므로 같은 축으로 집계한다.
#: hip_r 은 힙 **회전(요)** 이다(analysis §6 정정: '힙 롤' 아님).
PITCH_JOINTS = ("left_hip_f_joint", "right_hip_f_joint", "left_knee_joint",
                "right_knee_joint", "left_ankle_f_joint", "right_ankle_f_joint")
LATERAL_JOINTS = ("left_hip_a_joint", "right_hip_a_joint",
                  "left_ankle_r_joint", "right_ankle_r_joint")
YAW_JOINTS = ("left_hip_r_joint", "right_hip_r_joint")
GROUPS = (("피치", PITCH_JOINTS), ("횡(롤)", LATERAL_JOINTS), ("요", YAW_JOINTS))

DEFAULT_HOST = "172.16.35.89"
DEFAULT_PORT = 8800
POLL_HZ = 5.0
MAX_AGE_SEC = 0.5          # 읽기전용 라운드로빈 0.24 s + 지터 (위 안전 주석 참조)
MAX_AGE_CAP_SEC = 1.0      # 이 이상으로는 느슨하게 못 준다(계측 품질 하한)
HOLD_SEC = 1.2
HTTP_TIMEOUT_SEC = 1.5
STATE_PATH = "/api/state"

#: 측정 품질 임계 — 전부 '이 측정을 믿을 수 있는가' 를 가르는 계측 기준이다.
#: (실기 가드가 아니다. 실기 가드·임계는 이 도구가 읽지도 바꾸지도 않는다.)
THRESHOLDS = {
    "min_samples": 10,        # 5 Hz × 2 s
    "valid_fraction": 0.8,    # 신선한 표본 비율
    "min_changes": 3,         # 엔코더 값이 실제로 갱신된 횟수
    "min_enc_span": 0.05,     # [deg] 이보다 작으면 '엔코더가 안 움직임'
    "min_hold_samples": 3,
    "neg_tol": 0.20,          # [deg] 음수 유격 허용 오차
    "crosstalk": 1.0,         # [deg] 다른 관절 동반 이동
    "imu": 1.0,               # [deg] 몸통 기울기 변화
    "tau": 0.5,               # [Nm] 토크 추정 — 무동력 확인용(추정치)
    "mismatch": 0.5,          # [deg] 전구간 폭 vs 홀드 기준 폭 차이
}

#: 시뮬 참조값(§4.8) — 실물 성공 확률이 아니다. 표 아래에 근거와 함께만 쓴다.
SIM_REFERENCE = ("시뮬 참조(analysis §4.8, v5 명목 64 env): 피치 유격 1° 100 % · "
                 "2° 84~92 % · 3° 8~23 % (방출 후 생존). 실물 성공 확률 아님.")


# ─────────────────────── 순수 함수 (selftest 대상) ───────────────────────


def finite(value):
    """유한한 수면 float, 아니면 None (bool 은 수로 보지 않는다)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def span(values):
    return (max(values) - min(values)) if values else None


def median(values):
    return statistics.median(values) if values else None


def parse_state(doc, t, max_age=MAX_AGE_SEC):
    """콘솔 `/api/state` 문서 → 표본 1개.

    신선하지 않거나 무효인 관절은 **값으로 채우지 않고** 사유와 함께 dropped 에 남긴다
    (센서 무응답·오래된 값을 정상 입력처럼 쓰지 않는다).
    """
    doc = doc if isinstance(doc, dict) else {}
    s8 = doc.get("stage8") if isinstance(doc.get("stage8"), dict) else {}
    raw = s8.get("joints") if isinstance(s8.get("joints"), dict) else {}
    joints, dropped, ages, taus = {}, {}, [], {}
    for name, j in raw.items():
        if not isinstance(j, dict):
            dropped[name] = "형식"
            continue
        tau = finite(j.get("feedback_torque_est"))
        if tau is not None:
            taus[name] = tau
        value = finite(j.get("latest_actual_deg"))
        age = finite(j.get("feedback_age_sec"))
        if j.get("feedback_valid") is not True:
            dropped[name] = "무효"
        elif j.get("feedback_stale") is True:
            dropped[name] = "stale"
        elif value is None or age is None:
            dropped[name] = "값없음"
        elif age > max_age:
            dropped[name] = "age%.2f" % age
        else:
            joints[name] = value
            ages.append(age)
    imu = doc.get("imu") if isinstance(doc.get("imu"), dict) else {}
    bridge = doc.get("bridge") if isinstance(doc.get("bridge"), dict) else {}
    return {
        "t": round(float(t), 3),
        "joints": joints,
        "dropped": dropped,
        "age_max": round(max(ages), 3) if ages else None,
        "tau": taus,
        "armed": s8.get("armed") is True,
        "bridge_live": bridge.get("live") is True,
        "fwd": finite(imu.get("fwd")),
        "lat": finite(imu.get("lat")),
        "imu_age": finite(imu.get("age")),
    }


def in_window(samples, window):
    if not window:
        return []
    t0, t1 = window
    return [s for s in samples if t0 <= s["t"] <= t1]


def measure(joint, samples, hold_a, hold_b, link_a_deg, link_b_deg,
            max_age=MAX_AGE_SEC, th=None, note=None, stamp=None):
    """캡처 표본 + 양 끝 홀드 구간 + 폰 각도 2개 → 측정 기록 1건.

    headline 은 과제 정의 그대로 `유격 = 링크 폭 − 엔코더(최대−최소) 폭`,
    교차검증용으로 `홀드 중앙값 기준 폭`도 같이 남긴다(오버슈트 검출).
    """
    th = dict(THRESHOLDS if th is None else th)
    used = [s for s in samples if joint in s["joints"]]
    values = [s["joints"][joint] for s in used]
    enc_span = span(values)
    changes = sum(1 for a, b in zip(values, values[1:]) if abs(b - a) > 1e-6)

    a_values = [s["joints"][joint] for s in in_window(used, hold_a)]
    b_values = [s["joints"][joint] for s in in_window(used, hold_b)]
    enc_a = median(a_values) if len(a_values) >= th["min_hold_samples"] else None
    enc_b = median(b_values) if len(b_values) >= th["min_hold_samples"] else None
    enc_hold_span = abs(enc_b - enc_a) if (enc_a is not None and enc_b is not None) else None

    link_span = abs(float(link_b_deg) - float(link_a_deg))
    play = (link_span - enc_span) if enc_span is not None else None
    play_hold = (link_span - enc_hold_span) if enc_hold_span is not None else None

    cross = {}
    for sample in used:
        for name, value in sample["joints"].items():
            if name != joint:
                cross.setdefault(name, []).append(value)
    cross_spans = {name: span(vals) for name, vals in cross.items() if span(vals) is not None}
    if cross_spans:
        cross_name = max(cross_spans, key=lambda n: cross_spans[n])
        cross_span = cross_spans[cross_name]
    else:
        cross_name, cross_span = None, None

    fwd_span = span([s["fwd"] for s in samples if s["fwd"] is not None])
    lat_span = span([s["lat"] for s in samples if s["lat"] is not None])
    tau_all = [abs(v) for s in samples for v in s["tau"].values()]
    tau_joint = [abs(s["tau"][joint]) for s in samples if joint in s["tau"]]
    ages = [s["age_max"] for s in samples if s["age_max"] is not None]

    armed_seen = any(s["armed"] for s in samples)
    live_seen = any(s["bridge_live"] for s in samples)

    flags = []

    def flag(code, level, msg):
        flags.append({"code": code, "level": level, "msg": msg})

    if armed_seen:
        flag("armed", "invalid", "stage8 ARMED — 무동력 조건 위반(토크 인가 상태)")
    if live_seen:
        flag("policy", "invalid", "브리지 정책 live — 관절 목표가 스스로 움직인다")
    n_all, n_used = len(samples), len(used)
    if n_used < th["min_samples"]:
        flag("samples", "invalid", "유효 표본 %d 개 (≥%d 필요)" % (n_used, th["min_samples"]))
    elif n_used < th["valid_fraction"] * n_all:
        flag("stale", "invalid", "신선한 표본 %d/%d — 피드백 결손" % (n_used, n_all))
    if changes < th["min_changes"]:
        flag("frozen", "invalid",
             "엔코더 갱신 %d 회 — 읽기전용 폴링 정지(ARM 이력 래치) 의심, stage8 재기동 후 재측정" % changes)
    if enc_span is None or enc_span < th["min_enc_span"]:
        flag("enc_flat", "invalid", "엔코더 변화 없음 — 관절 선택·측정 축·피드백 확인")
    if play is not None and play < -th["neg_tol"]:
        flag("negative", "invalid", "엔코더 폭 > 링크 폭 — 폰 각도 오독·부모 링크 이동 의심")
    if cross_span is not None and cross_span > th["crosstalk"]:
        flag("crosstalk", "warn",
             "%s 가 %.2f° 동반 이동 — 링크 각도(절대) 과대 가능" % (cross_name, cross_span))
    tilt_span = max([v for v in (fwd_span, lat_span) if v is not None], default=None)
    if tilt_span is not None and tilt_span > th["imu"]:
        flag("imu", "warn", "몸통 기울기 %.2f° 변화 — 로봇/지그가 같이 움직였다" % tilt_span)
    if tau_all and max(tau_all) > th["tau"]:
        flag("torque", "warn", "토크 추정 최대 %.2f Nm — 무동력 아님 의심(추정치)" % max(tau_all))
    if enc_hold_span is None:
        flag("hold", "warn", "홀드 구간 표본 부족 — 홀드 기준 유격 없음")
    elif play is not None and play_hold is not None and abs(play - play_hold) > th["mismatch"]:
        flag("mismatch", "warn",
             "전구간 %.2f° vs 홀드 %.2f° 차이 — 밀 때 오버슈트/되튐" % (play, play_hold))

    valid = not any(f["level"] == "invalid" for f in flags)

    def r2(value):
        return round(value, 2) if value is not None else None

    return {
        "stamp": stamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "joint": joint,
        "link_a_deg": round(float(link_a_deg), 2),
        "link_b_deg": round(float(link_b_deg), 2),
        "link_span_deg": r2(link_span),
        "enc_a_deg": r2(enc_a),
        "enc_b_deg": r2(enc_b),
        "enc_span_deg": r2(enc_span),
        "enc_hold_span_deg": r2(enc_hold_span),
        "play_deg": r2(play),
        "play_hold_deg": r2(play_hold),
        "n_samples": n_all,
        "n_used": n_used,
        "n_changes": changes,
        "age_max_sec": r2(max(ages)) if ages else None,
        "max_age_sec": round(float(max_age), 3),
        "cross_joint": cross_name,
        "cross_span_deg": r2(cross_span),
        "imu_fwd_span_deg": r2(fwd_span),
        "imu_lat_span_deg": r2(lat_span),
        "tau_abs_max_nm": r2(max(tau_all)) if tau_all else None,
        "tau_joint_abs_max_nm": r2(max(tau_joint)) if tau_joint else None,
        "armed_seen": armed_seen,
        "policy_live_seen": live_seen,
        "valid": valid,
        "flags": flags,
        "note": note,
    }


def summarize(records):
    """기록들 → 관절별·그룹별 집계. **유효 기록만** 중앙값에 들어간다."""
    per = {}
    for rec in records:
        if rec.get("valid") and rec.get("play_deg") is not None:
            per.setdefault(rec["joint"], []).append(float(rec["play_deg"]))
    joints = {}
    for name, values in per.items():
        joints[name] = {
            "n": len(values),
            "median_deg": round(statistics.median(values), 2),
            "min_deg": round(min(values), 2),
            "max_deg": round(max(values), 2),
        }
    groups = {}
    for label, names in GROUPS:
        values = [v for name in names for v in per.get(name, [])]
        groups[label] = {
            "n_joint": sum(1 for name in names if per.get(name)),
            "n": len(values),
            "median_deg": round(statistics.median(values), 2) if values else None,
            "max_deg": round(max(values), 2) if values else None,
        }
    warn = sum(1 for r in records
               if r.get("valid") and any(f["level"] == "warn" for f in r.get("flags", [])))
    return {
        "n_records": len(records),
        "n_valid": sum(1 for r in records if r.get("valid")),
        "n_invalid": sum(1 for r in records if not r.get("valid")),
        "n_warn": warn,
        "joints": joints,
        "groups": groups,
        "sim_reference": SIM_REFERENCE,
    }


def _width(text):
    """터미널 표시 폭(한글·기호는 2칸)."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text, width):
    return text + " " * max(0, width - _width(text))


def _cell(value, width=8, fmt="%.2f"):
    text = fmt % value if isinstance(value, (int, float)) else "—"
    return " " * max(0, width - _width(text)) + text


def render_table(doc):
    """JSON 문서 → 사람이 읽는 표(문자열)."""
    records = doc.get("records", [])
    summary = summarize(records)
    out = []
    out.append("무동력 관절 유격 실측 — %s (host %s, 기록 %d 건)"
               % (doc.get("created", "?"), doc.get("host", "?"), len(records)))
    if doc.get("note"):
        out.append("메모: %s" % doc["note"])
    out.append("")
    out.append("[측정 기록]  유격 = 링크폭 − 엔코더폭 (엔코더폭 = 최대−최소)")
    out.append("  #  " + _pad("관절", 22) + "  링크폭  엔코폭    유격  홀드유격  판정   플래그")
    for i, rec in enumerate(records, 1):
        codes = ",".join(f["code"] for f in rec.get("flags", [])) or "-"
        out.append("%3d  %s%s%s%s%s  %s  %s" % (
            i, _pad(rec.get("joint", "?")[:20], 22),
            _cell(rec.get("link_span_deg")), _cell(rec.get("enc_span_deg")),
            _cell(rec.get("play_deg")), _cell(rec.get("play_hold_deg"), 10),
            _pad("OK" if rec.get("valid") else "무효", 5), codes))
    out.append("")
    out.append("[관절별] 유효 기록만 집계")
    out.append("  " + _pad("관절", 22) + "   n   중앙[°]   최소[°]   최대[°]")
    for name in JOINT_NAMES:
        agg = summary["joints"].get(name)
        if not agg:
            continue
        out.append("  %s%4d%s%s%s" % (_pad(name, 22), agg["n"], _cell(agg["median_deg"], 10),
                                         _cell(agg["min_deg"], 10), _cell(agg["max_deg"], 10)))
    extra = [n for n in summary["joints"] if n not in JOINT_NAMES]
    for name in extra:
        agg = summary["joints"][name]
        out.append("  %s%4d%s%s%s  (계약 밖 관절명)"
                   % (_pad(name[:20], 22), agg["n"], _cell(agg["median_deg"], 10),
                      _cell(agg["min_deg"], 10), _cell(agg["max_deg"], 10)))
    out.append("")
    out.append("[축 그룹]")
    out.append("  " + _pad("그룹", 10) + "관절수    n   중앙[°]   최대[°]")
    for label, _names in GROUPS:
        agg = summary["groups"][label]
        out.append("  %s%6d%5d%s%s" % (_pad(label, 10), agg["n_joint"], agg["n"],
                                          _cell(agg["median_deg"], 10), _cell(agg["max_deg"], 10)))
    out.append("")
    out.append("유효 %d · 무효 %d · 경고 포함 %d" %
               (summary["n_valid"], summary["n_invalid"], summary["n_warn"]))
    out.append(SIM_REFERENCE)
    out.append("주의: 유격과 구조 처짐은 이 측정으로 분리되지 않는다(살짝 밀 것). "
               "링크 각도는 절대각이라 부모 링크·지그가 움직이면 과대평가된다.")
    return "\n".join(out)


def new_doc(host, note=None):
    return {
        "tool": "play_probe.py",
        "format": 1,
        "unit": "deg",
        "host": host,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "definition": "play_deg = link_span_deg - enc_span_deg (enc = latest_actual_deg max-min)",
        "records": [],
    }


def load_doc(path):
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict) or doc.get("tool") != "play_probe.py":
        raise ValueError("%s 는 play_probe 결과 파일이 아니다" % path)
    doc.setdefault("records", [])
    return doc


def save_doc(path, doc):
    """요약을 갱신해 원자적으로 저장(측정 1건마다 호출 — 중간에 죽어도 남는다)."""
    doc["summary"] = summarize(doc.get("records", []))
    doc["saved"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    os.replace(tmp, path)
    return path


# ─────────────────────────── 읽기 전용 폴링 ───────────────────────────


def state_url(host, port=DEFAULT_PORT):
    return "http://%s:%d%s" % (host, int(port), STATE_PATH)


def fetch_state(url, timeout=HTTP_TIMEOUT_SEC):
    """콘솔 상태를 **GET 으로만** 읽는다. 본문(data)을 싣지 않으므로 쓰기가 불가능하다."""
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(4_000_000)
    doc = json.loads(body.decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("상태 형식 오류(객체 아님)")
    return doc


class Poller(object):
    """5 Hz 읽기 전용 폴러 — 표본을 시간축으로 모아 둔다."""

    def __init__(self, url, hz=POLL_HZ, max_age=MAX_AGE_SEC, timeout=HTTP_TIMEOUT_SEC):
        self.url, self.hz, self.max_age, self.timeout = url, float(hz), float(max_age), timeout
        self.samples = []
        self.lock = threading.Lock()
        self.thread = None
        self.run = False
        self.n_ok = 0
        self.n_err = 0
        self.err = None

    def _loop(self):
        period = 1.0 / self.hz
        while self.run:
            started = time.time()
            try:
                sample = parse_state(fetch_state(self.url, self.timeout),
                                     time.monotonic(), self.max_age)
                with self.lock:
                    self.samples.append(sample)
                    self.n_ok += 1
                    self.err = None
            except Exception as exc:                      # 네트워크·형식 오류는 기록하고 계속
                with self.lock:
                    self.n_err += 1
                    self.err = ("%s: %s" % (type(exc).__name__, exc))[:120]
            rest = period - (time.time() - started)
            if rest > 0:
                time.sleep(rest)

    def start(self):
        self.run = True
        self.thread = threading.Thread(target=self._loop, name="play_probe_poll", daemon=True)
        self.thread.start()

    def stop(self):
        self.run = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def latest(self):
        with self.lock:
            return self.samples[-1] if self.samples else None

    def window(self, t0, t1):
        with self.lock:
            return [s for s in self.samples if t0 <= s["t"] <= t1]

    def health(self):
        with self.lock:
            return {"ok": self.n_ok, "err": self.n_err, "last_err": self.err}


# ───────────────────────────── 대화형 절차 ─────────────────────────────


def ask(prompt):
    """한 줄 입력. 중단(q/Ctrl-C/EOF)이면 None."""
    try:
        text = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("")
        return None
    return None if text.lower() in ("q", "quit", "exit") else text


def ask_float(prompt):
    while True:
        text = ask(prompt)
        if text is None:
            return None
        try:
            return float(text)
        except ValueError:
            print("  숫자로 입력하세요(중단 q).")


def status_text(sample, health):
    if sample is None:
        return "상태 없음 (폴링 ok %d / err %d: %s)" % (health["ok"], health["err"], health["last_err"])
    return ("ARMED %-3s 정책 %-3s 신선관절 %2d/12 최대나이 %s s  fwd %s lat %s  (ok %d/err %d)"
            % ("예" if sample["armed"] else "아니오",
               "live" if sample["bridge_live"] else "정지",
               len(sample["joints"]),
               "%.2f" % sample["age_max"] if sample["age_max"] is not None else "—",
               "%+.1f" % sample["fwd"] if sample["fwd"] is not None else "—",
               "%+.1f" % sample["lat"] if sample["lat"] is not None else "—",
               health["ok"], health["err"]))


def preflight(sample, joint):
    """측정 시작 전 거부 사유 목록(비어 있으면 진행 가능)."""
    problems = []
    if sample is None:
        return ["콘솔 상태를 못 읽었다 — 호스트·포트·콘솔 기동 확인"]
    if sample["armed"]:
        problems.append("stage8 ARMED — 토크가 걸린 채로는 유격을 못 잰다(DISARM 후)")
    if sample["bridge_live"]:
        problems.append("정책 live — 정지 후 측정")
    if joint not in sample["joints"]:
        problems.append("%s 피드백이 신선하지 않다(%s) — stage8 읽기전용 폴링 상태 확인"
                        % (joint, sample["dropped"].get(joint, "없음")))
    return problems


def hold_capture(poller, seconds):
    """지금부터 seconds 동안의 표본 구간 (t0, t1) 을 반환."""
    t0 = time.monotonic()
    time.sleep(seconds)
    return (t0, time.monotonic())


def run_measurement(poller, joint, args):
    """관절 1개 측정 1회. 중단이면 None."""
    sample = poller.latest()
    problems = preflight(sample, joint)
    if problems:
        for text in problems:
            print("  [거부] %s" % text)
        return "skip"
    print("  폰 경사계를 %s 아래 링크에 올리고, 부모 링크는 손으로 고정하세요." % joint)
    if ask("  ① 한쪽 끝까지 **살짝** 밀어 붙인 채 Enter (중단 q): ") is None:
        return None
    t_start = time.monotonic()
    window_a = hold_capture(poller, args.hold)
    link_a = ask_float("  ① 그 자세의 폰 경사계 각도 [deg]: ")
    if link_a is None:
        return None
    if ask("  ② 반대쪽 끝까지 살짝 밀어 붙인 채 Enter: ") is None:
        return None
    window_b = hold_capture(poller, args.hold)
    link_b = ask_float("  ② 그 자세의 폰 경사계 각도 [deg]: ")
    if link_b is None:
        return None
    t_end = time.monotonic()
    samples = poller.window(t_start, t_end)
    record = measure(joint, samples, window_a, window_b, link_a, link_b,
                     max_age=poller.max_age, note=args.note)
    return record


def print_record(record):
    print("  링크 폭 %.2f°  엔코더 폭 %s°  →  **유격 %s°**  (홀드 기준 %s°)"
          % (record["link_span_deg"],
             "%.2f" % record["enc_span_deg"] if record["enc_span_deg"] is not None else "—",
             "%.2f" % record["play_deg"] if record["play_deg"] is not None else "—",
             "%.2f" % record["play_hold_deg"] if record["play_hold_deg"] is not None else "—"))
    print("  표본 %d(유효 %d, 갱신 %d회)  동반이동 %s %s°  기울기 fwd %s / lat %s"
          % (record["n_samples"], record["n_used"], record["n_changes"],
             record["cross_joint"] or "—",
             "%.2f" % record["cross_span_deg"] if record["cross_span_deg"] is not None else "—",
             "%.2f" % record["imu_fwd_span_deg"] if record["imu_fwd_span_deg"] is not None else "—",
             "%.2f" % record["imu_lat_span_deg"] if record["imu_lat_span_deg"] is not None else "—"))
    for flag in record["flags"]:
        print("  [%s] %s" % ("무효" if flag["level"] == "invalid" else "경고", flag["msg"]))
    if not record["valid"]:
        print("  → 이 측정은 **무효**로 기록된다(값은 남기되 집계에서 뺀다).")


def monitor(poller, seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        print("  " + status_text(poller.latest(), poller.health()))
        time.sleep(1.0)
    return 0


def interactive(args):
    url = state_url(args.host, args.port)
    poller = Poller(url, args.hz, args.max_age)
    poller.start()
    try:
        print("콘솔 읽기 전용 폴링 %s (%.1f Hz, 나이 한계 %.2f s)" % (url, args.hz, args.max_age))
        deadline = time.time() + 8.0
        while poller.latest() is None and time.time() < deadline:
            time.sleep(0.2)
        if poller.latest() is None:
            print("상태 수신 실패: %s" % poller.health()["last_err"])
            return 2
        print("  " + status_text(poller.latest(), poller.health()))
        if args.monitor > 0:
            return monitor(poller, args.monitor)

        path = args.out or os.path.expanduser(
            time.strftime("~/logs/play_probe_%Y%m%d_%H%M.json"))
        path = os.path.expanduser(path)
        doc = load_doc(path) if os.path.exists(path) else new_doc(args.host, args.note)
        print("기록 파일: %s (기존 %d 건)" % (path, len(doc["records"])))

        queue = []
        for joint in (args.joint or []):
            queue.extend([joint] * max(1, args.repeat))
        while True:
            if queue:
                joint = queue.pop(0)
            else:
                text = ask("\n측정할 관절명 (목록 l, 종료 q): ")
                if text is None:
                    break
                if text in ("l", "list"):
                    for name in JOINT_NAMES:
                        print("  " + name)
                    continue
                joint = text
            if joint not in JOINT_NAMES:
                print("  [거부] 계약에 없는 관절명: %s (목록은 l)" % joint)
                continue
            print("\n== %s ==" % joint)
            print("  " + status_text(poller.latest(), poller.health()))
            record = run_measurement(poller, joint, args)
            if record is None:
                break
            if record == "skip":
                continue
            doc["records"].append(record)
            save_doc(path, doc)
            print_record(record)
        print("\n" + render_table(doc))
        save_doc(path, doc)
        print("\n저장: %s" % path)
        return 0
    finally:
        poller.stop()


# ──────────────────────────────── selftest ────────────────────────────────


def _fake_doc(joint_deg, armed=False, live=False, fwd=0.0, lat=0.0, age=0.05,
              stale=(), tau=0.0, missing=()):
    """콘솔 `/api/state` 모양의 가짜 상태 1건."""
    joints = {}
    for name in JOINT_NAMES:
        if name in missing:
            continue
        joints[name] = {
            "allowed": True,
            "baseline_deg": 0.0,
            "latest_actual_deg": float(joint_deg.get(name, 0.0)),
            "feedback_age_sec": age,
            "feedback_stale": name in stale,
            "feedback_valid": True,
            "feedback_torque_est": tau,
        }
    return {"stage8": {"armed": armed, "joints": joints, "feedback_ok": True},
            "bridge": {"live": live, "armed": False},
            "imu": {"fwd": fwd, "lat": lat, "tilt": 0.0, "age": 0.1},
            "foot": {"contact": [True, True]}}


def selftest():
    """가짜 상태 시퀀스로 계산·판정 로직을 검증한다(네트워크·로봇 없음)."""
    import tempfile
    from unittest import mock

    fails = []

    def check(name, condition, detail=""):
        if not condition:
            fails.append("%s %s" % (name, detail))

    dt = 0.2
    joint = "left_knee_joint"

    def build(values, name=None, **kw):
        """값 리스트 → 표본 리스트 (표본 i 의 시각 = i × dt 초).

        name: 그 값이 들어갈 관절(기본 left_knee_joint)
        others: {표본 index: {다른 관절명: deg}}, per_sample: {index: _fake_doc 인자}
        """
        others = kw.pop("others", {})
        per_sample = kw.pop("per_sample", {})
        max_age = kw.pop("max_age", MAX_AGE_SEC)
        out = []
        for i, value in enumerate(values):
            state = dict(others.get(i, {}))
            state[name or joint] = value
            kwargs = dict(kw)
            kwargs.update(per_sample.get(i, {}))
            out.append(parse_state(_fake_doc(state, **kwargs), i * dt, max_age))
        return out

    #: 밀기(4) + A홀드(6) + 되밀기(5) + B홀드(6) = 21 표본, 4.0 s
    push_a = [0.0, 1.5, 3.0, 4.2]
    hold_a = [4.60, 4.61, 4.60, 4.59, 4.60, 4.60]
    push_b = [3.0, 1.0, -1.0, -3.0, -4.4]
    hold_b = [-4.80, -4.79, -4.80, -4.81, -4.80, -4.80]
    values = push_a + hold_a + push_b + hold_b
    win_a = (4 * dt, 9 * dt)
    win_b = (15 * dt, 20 * dt)

    # 1) 정상 측정 — 엔코더 폭 max−min, 유격 = 링크 폭 − 엔코더 폭
    rec = measure(joint, build(values), win_a, win_b, 12.30, -0.20)
    check("1 엔코더폭", abs(rec["enc_span_deg"] - 9.42) < 1e-6, rec["enc_span_deg"])
    check("1 링크폭", abs(rec["link_span_deg"] - 12.50) < 1e-6, rec["link_span_deg"])
    check("1 유격", abs(rec["play_deg"] - 3.08) < 1e-6, rec["play_deg"])
    check("1 홀드기준", abs(rec["play_hold_deg"] - 3.10) < 1e-6, rec["play_hold_deg"])
    check("1 유효", rec["valid"] is True and not rec["flags"], rec["flags"])
    check("1 표본수", rec["n_samples"] == 21 and rec["n_used"] == 21, rec["n_used"])

    # 2) stale 표본은 폭에 섞이지 않는다(값 99° 를 넣어도 무시) — 소수면 유효
    stale_vals = list(values)
    per_sample = {}
    for i in (6, 7, 12):
        stale_vals[i] = 99.0
        per_sample[i] = {"stale": (joint,)}
    rec2 = measure(joint, build(stale_vals, per_sample=per_sample), win_a, win_b, 12.30, -0.20)
    check("2 stale 제외", abs(rec2["enc_span_deg"] - 9.42) < 1e-6, rec2["enc_span_deg"])
    check("2 표본", rec2["n_used"] == 18 and rec2["n_samples"] == 21, rec2["n_used"])
    check("2 유효", rec2["valid"] is True, rec2["flags"])

    # 3) stale 이 과반이면 무효
    many = {i: {"stale": (joint,)} for i in range(0, 21, 2)}
    rec3 = measure(joint, build(values, per_sample=many), win_a, win_b, 12.30, -0.20)
    check("3 결손 무효", rec3["valid"] is False and
          any(f["code"] == "stale" for f in rec3["flags"]), rec3["flags"])

    # 4) 나이 한계 초과는 값으로 쓰지 않는다
    old = parse_state(_fake_doc({joint: 5.0}, age=0.9), 0.0, 0.5)
    check("4 나이 제외", joint not in old["joints"] and old["dropped"][joint].startswith("age"),
          old["dropped"].get(joint))
    fresh = parse_state(_fake_doc({joint: 5.0}, age=0.24), 0.0, 0.5)
    check("4 나이 통과", fresh["joints"].get(joint) == 5.0, fresh["dropped"])

    # 5) ARMED / 정책 live 는 무효 (무동력 조건 위반)
    rec5 = measure(joint, build(values, armed=True), win_a, win_b, 12.30, -0.20)
    check("5 ARMED 무효", rec5["valid"] is False and
          any(f["code"] == "armed" for f in rec5["flags"]), rec5["flags"])
    rec5b = measure(joint, build(values, live=True), win_a, win_b, 12.30, -0.20)
    check("5 정책 무효", rec5b["valid"] is False and
          any(f["code"] == "policy" for f in rec5b["flags"]), rec5b["flags"])

    # 6) 엔코더가 얼어 있으면(ARM 이력 래치) 무효
    rec6 = measure(joint, build([3.3] * 21), win_a, win_b, 12.30, -0.20)
    check("6 동결 무효", rec6["valid"] is False and
          {"frozen", "enc_flat"} <= {f["code"] for f in rec6["flags"]}, rec6["flags"])

    # 7) 음수 유격(엔코더 폭 > 링크 폭)은 무효
    rec7 = measure(joint, build(values), win_a, win_b, 4.00, -4.00)
    check("7 음수 무효", rec7["valid"] is False and
          any(f["code"] == "negative" for f in rec7["flags"]), rec7["play_deg"])

    # 8) 다른 관절 동반 이동 → 경고(유효는 유지)
    others = {i: {"left_hip_f_joint": 0.0 if i < 10 else 2.5} for i in range(21)}
    rec8 = measure(joint, build(values, others=others), win_a, win_b, 12.30, -0.20)
    check("8 동반이동 경고", rec8["valid"] is True and
          any(f["code"] == "crosstalk" for f in rec8["flags"]), rec8["flags"])
    check("8 동반이동 값", rec8["cross_joint"] == "left_hip_f_joint" and
          abs(rec8["cross_span_deg"] - 2.5) < 1e-6, rec8["cross_span_deg"])

    # 9) 몸통 기울기 변화 → 경고
    tilt = {i: {"lat": 0.0 if i < 10 else 2.5} for i in range(21)}
    rec9 = measure(joint, build(values, per_sample=tilt), win_a, win_b, 12.30, -0.20)
    check("9 기울기 경고", rec9["valid"] is True and
          any(f["code"] == "imu" for f in rec9["flags"]), rec9["flags"])
    check("9 기울기 값", abs(rec9["imu_lat_span_deg"] - 2.5) < 1e-6, rec9["imu_lat_span_deg"])

    # 10) 토크 추정이 크면 경고(무동력 아님 의심)
    rec10 = measure(joint, build(values, tau=1.2), win_a, win_b, 12.30, -0.20)
    check("10 토크 경고", any(f["code"] == "torque" for f in rec10["flags"]), rec10["flags"])

    # 11) 홀드 구간 표본 부족 → 홀드 기준 없음 + 경고
    rec11 = measure(joint, build(values), (100.0, 101.0), (102.0, 103.0), 12.30, -0.20)
    check("11 홀드 없음", rec11["play_hold_deg"] is None and
          any(f["code"] == "hold" for f in rec11["flags"]), rec11["flags"])

    # 12) 밀 때 오버슈트 → 전구간/홀드 불일치 경고
    over = [0.0, 2.0, 6.00, 5.0] + hold_a + [3.0, 0.0, -3.0, -6.40, -5.0] + hold_b
    rec12 = measure(joint, build(over), win_a, win_b, 12.30, -0.20)
    check("12 오버슈트", any(f["code"] == "mismatch" for f in rec12["flags"]), rec12["flags"])
    check("12 두 기준", abs(rec12["enc_span_deg"] - 12.40) < 1e-6 and
          abs(rec12["enc_hold_span_deg"] - 9.40) < 1e-6,
          (rec12["enc_span_deg"], rec12["enc_hold_span_deg"]))

    # 13) 관절 항목 자체가 없거나 IMU 가 없어도 죽지 않는다
    none_imu = dict(_fake_doc({joint: 1.0}, missing=(joint,)))
    none_imu["imu"] = None
    parsed = parse_state(none_imu, 0.0)
    check("13 결측 처리", joint not in parsed["joints"] and parsed["fwd"] is None, parsed)
    rec13 = measure(joint, [parsed], None, None, 10.0, 0.0)
    check("13 결측 무효", rec13["valid"] is False, rec13["flags"])

    # 14) 집계 — 유효 기록만 중앙값, 무효는 제외
    records = [
        measure(joint, build(values), win_a, win_b, 12.30, -0.20),
        measure(joint, build(values), win_a, win_b, 12.10, -0.20),
        measure("left_hip_a_joint", build(values, name="left_hip_a_joint"),
                win_a, win_b, 10.10, 0.10),
        measure(joint, build(values, armed=True), win_a, win_b, 30.00, -0.20),
    ]
    summary = summarize(records)
    check("14 무효 제외", summary["n_valid"] == 3 and summary["n_invalid"] == 1, summary)
    check("14 관절 중앙", abs(summary["joints"][joint]["median_deg"] - 2.98) < 1e-6,
          summary["joints"][joint])
    check("14 피치 그룹", summary["groups"]["피치"]["n"] == 2, summary["groups"])
    check("14 횡 그룹", summary["groups"]["횡(롤)"]["n"] == 1, summary["groups"])

    # 15) 표 렌더 + JSON 왕복(이어쓰기)
    with tempfile.TemporaryDirectory(prefix="play-probe-") as directory:
        path = os.path.join(directory, "play.json")
        doc = new_doc("selftest-host", "지그, 무동력")
        doc["records"].extend(records)
        save_doc(path, doc)
        again = load_doc(path)
        check("15 왕복", len(again["records"]) == 4 and again["summary"]["n_valid"] == 3,
              again.get("summary"))
        again["records"].append(measure(joint, build(values), win_a, win_b, 12.30, -0.20))
        save_doc(path, again)
        third = load_doc(path)
        check("15 이어쓰기", len(third["records"]) == 5, len(third["records"]))
        table = render_table(third)
        check("15 표", joint in table and "피치" in table and "유격" in table, table[:80])

    # 16) 계약: 상태 읽기는 GET·본문 없음, 명령 엔드포인트를 쓰지 않는다
    captured = {}

    class _Resp(object):
        def read(self, _n=None):
            return json.dumps(_fake_doc({joint: 1.0})).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    def _fake_urlopen(request, timeout=None):
        captured["method"] = request.get_method()
        captured["data"] = request.data
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _Resp()

    with mock.patch.object(urllib.request, "urlopen", _fake_urlopen):
        doc16 = fetch_state(state_url("10.0.0.1"), 1.0)
    check("16 GET 전용", captured.get("method") == "GET" and captured.get("data") is None,
          captured)
    check("16 상태 경로", captured.get("url", "").endswith(STATE_PATH), captured.get("url"))
    check("16 파싱", "stage8" in doc16, list(doc16))

    source = open(os.path.abspath(__file__), "r", encoding="utf-8").read()
    forbidden = ["/api/" + "action",       # 콘솔 명령 엔드포인트
                 "PO" + "ST",              # 쓰기 메서드
                 "create_" + "publisher",  # ROS 발행
                 "rcl" + "py",             # ROS 진입
                 "can" + "_bus",           # CAN 송신
                 "pk" + "ill"]             # 자기매칭 종료 금지(CLAUDE.md)
    hit = [token for token in forbidden if token in source]
    check("16 쓰기 경로 없음", not hit, hit)
    check("16 폴링 기본", POLL_HZ == 5.0 and MAX_AGE_SEC <= MAX_AGE_CAP_SEC,
          (POLL_HZ, MAX_AGE_SEC))

    # 17) 관절명 계약 — 배포 계약 12개와 같은 이름·순서
    check("17 관절 계약", len(JOINT_NAMES) == 12 and
          sorted(JOINT_NAMES) == sorted(list(PITCH_JOINTS) + list(LATERAL_JOINTS) + list(YAW_JOINTS)),
          JOINT_NAMES)

    # 18) preflight — ARMED/정책/피드백 없음을 각각 거부
    armed_sample = parse_state(_fake_doc({joint: 1.0}, armed=True), 0.0)
    check("18 ARMED 거부", any("ARMED" in p for p in preflight(armed_sample, joint)),
          preflight(armed_sample, joint))
    live_sample = parse_state(_fake_doc({joint: 1.0}, live=True), 0.0)
    check("18 정책 거부", any("정책" in p for p in preflight(live_sample, joint)),
          preflight(live_sample, joint))
    ok_sample = parse_state(_fake_doc({joint: 1.0}), 0.0)
    check("18 정상 통과", preflight(ok_sample, joint) == [], preflight(ok_sample, joint))
    check("18 상태없음 거부", len(preflight(None, joint)) == 1, preflight(None, joint))

    for text in fails:
        print("[FAIL] %s" % text)
    print("play_probe selftest: " + ("ALL PASS" if not fails else "%d 건 실패" % len(fails)))
    return 0 if not fails else 1


def main():
    parser = argparse.ArgumentParser(
        description="무동력 관절 유격 실측 — 콘솔 /api/state 를 GET 으로만 읽는다")
    parser.add_argument("--host", default=DEFAULT_HOST, help="콘솔 호스트 (기본 %(default)s)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--hz", type=float, default=POLL_HZ, help="폴링 [Hz] (기본 5)")
    parser.add_argument("--max-age", dest="max_age", type=float, default=MAX_AGE_SEC,
                        help="관절 피드백 나이 한계 [s] (기본 %(default)s, 읽기전용 갱신 0.24 s)")
    parser.add_argument("--hold", type=float, default=HOLD_SEC, help="양 끝 홀드 표본 구간 [s]")
    parser.add_argument("--joint", action="append", help="측정할 관절(여러 번 지정 가능)")
    parser.add_argument("--repeat", type=int, default=1, help="관절당 반복 횟수")
    parser.add_argument("--out", help="결과 JSON (기본 ~/logs/play_probe_<stamp>.json, 있으면 이어쓰기)")
    parser.add_argument("--note", help="기록에 남길 메모(지그 상태 등)")
    parser.add_argument("--report", help="저장된 JSON 의 표만 출력(네트워크 없음)")
    parser.add_argument("--monitor", type=float, default=0.0, help="측정 없이 상태만 N 초 표시")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    if args.report:
        print(render_table(load_doc(os.path.expanduser(args.report))))
        return 0
    if not 0.5 <= args.hz <= 20.0:
        parser.error("--hz 는 0.5~20 Hz")
    if not 0.05 <= args.max_age <= MAX_AGE_CAP_SEC:
        parser.error("--max-age 는 0.05~%.2f s (계측 품질 하한)" % MAX_AGE_CAP_SEC)
    if args.hold < 0.4:
        parser.error("--hold 는 0.4 s 이상 (홀드 표본 %d 개 필요)" % THRESHOLDS["min_hold_samples"])
    try:
        return interactive(args)
    except KeyboardInterrupt:                 # 측정 기록은 건별로 이미 저장돼 있다
        print("\n중단 — 지금까지의 기록은 저장돼 있다.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
