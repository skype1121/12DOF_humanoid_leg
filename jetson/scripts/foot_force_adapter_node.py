"""발바닥 힘 어댑터 — 압력센서 JSON을 브리지 계약 foot_force JSON으로 변환.

입력: /humanoid/pressure/data (std_msgs/String, 10Hz, 기존 압력 노드 JSON):
  {"timestamp", "source", "values":[8 ints 0-1023], "left":[4], "right":[4],
   "raw_ads1115":[8], "voltages":[8]}
출력: /humanoid/foot_force (std_msgs/String, 입력과 동일 레이트):
  {"timestamp", "left_n", "right_n", "left_raw_sum", "right_raw_sum",
   "left_contact", "right_contact", "left_pads_on", "right_pads_on",
   "calibrated": true, "dead_mask": [[4 bool],[4 bool]], "dead_note": str}
  dead_mask/dead_note 는 **계측 전용 진단**이다 (0912 추가). 접촉 판정·임계·
  pad bias 동작은 이 진단과 무관하게 그대로다 — 소비측은 무시해도 된다.

변환: left_n = NEWTON_PER_COUNT × max(0, sum(left 카운트) − LEFT_BIAS_COUNTS),
우측 동일. "calibrated": true = 실측 보정 완료 (2026-08-02 자립 스탠딩 보정).

강건성: JSON 파싱/형식 오류 메시지는 스킵하고 카운트 — 1·21·41…번째마다만
경고 로그 (스팸 방지). 변환은 순수 함수로 분리, rclpy는 main()에서만 lazy
import → --selftest는 ROS 소싱 없이 로컬에서 돈다.

사용: python3 foot_force_adapter_node.py             (Jetson, ROS2 Humble 소싱 후)
      python3 foot_force_adapter_node.py --selftest  (ROS 불필요 — 변환 검증)
      ros2 run 없이 파라미터: --ros-args -p newton_per_count:=0.08
"""
import argparse
import copy
import json
import math

IN_TOPIC = "/humanoid/pressure/data"
OUT_TOPIC = "/humanoid/foot_force"
QOS_DEPTH = 10  # 입력이 10Hz라 1초 버퍼면 충분
ERR_LOG_EVERY = 20

#: ★ 실측 보정 (2026-08-02 카운트 앵커, 2026-09-06 체중 재보정) ★
#: 절차: 무부하(매달림 0727) 카운트 = L328/R62 → 바이어스.
#:       자립 스탠딩(줄·손 무접촉, 0802) 20표본 평균 = L3356/R3559 카운트
#:       → 순카운트 (3356−328)+(3559−62) = 6525 = 체중 1.0 의 앵커.
#:       체중 = 17.62kg×9.81 = 172.9N (저울 실측 0904) → npc = 172.9/6525 = 0.02650.
#:       검산: L 80.2N + R 92.7N = 172.9N (좌우 46/54% — 당시 IMU 좌우 1.1°와 일치).
#: ⚠ 0802~0905 는 "12.0kg 추정 = 117.7N" 으로 npc 0.01804 를 썼다 — 보고 N 이 참값의 0.68배.
#:    그 시절 블랙박스(foot_n)는 ×1.469 (analyze_fall 이 날짜로 자동 보정). 백분율(발력/체중)은
#:    같은 카운트 앵커를 나누므로 구·신 모두 유효 — 4차 "90% 자립" 은 그대로 90% 다.
#: 구 플레이스홀더 0.05는 2.8배 과대 — 접촉 임계 5N 판정이 통째로 틀어졌었음.
#: 접촉 임계 5N(브리지 CONTACT_FORCE_THRESHOLD_N) = 시뮬 feet_contact threshold 5.0N 과 같은 참값 축
#:    (구 스케일에선 실효 7.35N 이었음).
#: ROS 파라미터 newton_per_count/left_bias_counts/right_bias_counts 로 override 가능.
NEWTON_PER_COUNT = 0.02650
LEFT_BIAS_COUNTS = 328.0
RIGHT_BIAS_COUNTS = 62.0
#: 0910 승윤님: 접지 판정은 '4개 패드 중 하나라도 눌리면 접지' (합산 5N 규칙은 발이 모서리로 닿을 때 왼발을 너무 일찍 비접촉으로 판정 →
#: 정책이 "왼발 떴다" 입력을 받고 반응, A1 3회 공통). 패드별 (count − pad_bias) > PAD_CONTACT_COUNTS 면 그 패드 눌림. 200 counts ≈ 5.3 N (0912 까지 80).
#: 0912: 왼발 index 2 패드 상시 눌림(고장, 429~728 카운트 표류) → 젯슨 sensors_up 에서 `-p left_pad_bias:="[0.0,0.0,5000.0,0.0]"` 로 판정 제외(패드 교체 전까지).
PAD_CONTACT_COUNTS = 200.0   # 0912 승윤님 결정: 80 → 200 카운트(≈5.3 N; 패드가 매우 민감해 80 은 스침도 눌림). 학습 접촉 임계 DR 2~20 N 안.

#: ── 죽은 채널 진단 (0912 추가, 계측 전용) ──────────────────────────────────
#: 0912 관측(측정): 지그로 양발 수평 접지 상태에서 왼 index 2·3, 오른 index 3 이
#: 0 카운트, 같은 발의 다른 채널은 200~900. 이 관측을 코드가 자동으로 표시만 한다.
#: 판정 규칙: 같은 발의 '다른' 채널 보정합 ≥ pad_contact_counts (= 그 발은 접지 중)
#:   AND 이 채널 보정값 ≤ pad_contact_counts (= 눌림 신호 없음)
#:   AND DEAD_WINDOW_SEC 동안 연속으로 0 또는 변화 없음  →  dead '후보'.
#: ⚠ 후보는 가설이다. 패드·배선·커넥터·ADC 중 무엇인지는 현장 점검으로 확정한다.
#: ⚠ 이 진단은 left_contact/right_contact/left_pads_on/right_pads_on/임계/bias 계산에
#:   일절 관여하지 않는다 (접촉 판정은 pressure_to_foot_force 단독, 아래 함수는 별도 경로).
DEAD_WINDOW_SEC = 5.0        # N초 연속이면 후보 (ROS 파라미터 dead_window_s 로 override)
DEAD_CHANGE_COUNTS = 2.0     # |값 − 창 기준값| 이 이하면 '변화 없음' (ADC 잡음 폭보다 작게)
PAD_EXCLUDE_BIAS = 2000.0    # pad bias 가 이 이상 = 판정 제외 채널(운영 관례 5000) → dead 판정에서도 제외
DEAD_MAX_GAP_SEC = 1.0       # 입력 10Hz 기준, 샘플 간극이 이보다 크면 연속성 불인정 → 창만 재시작


def new_dead_state():
    """죽은 채널 진단용 초기 상태 dict (발×채널 = 2×4)."""
    return {
        "ref": [[None] * 4, [None] * 4],      # 창 기준 카운트 (변화 감시용)
        "since": [[None] * 4, [None] * 4],    # 현재 창의 시작 timestamp
        "dead": [[False] * 4, [False] * 4],   # dead 후보 래치
        "last_ts": None,
        "note": "",
    }


def update_dead_channels(state, left, right, ts,
                         pad_contact_counts=PAD_CONTACT_COUNTS,
                         left_pad_bias=None, right_pad_bias=None,
                         dead_window_s=DEAD_WINDOW_SEC,
                         change_counts=DEAD_CHANGE_COUNTS,
                         exclude_bias=PAD_EXCLUDE_BIAS,
                         max_gap_s=DEAD_MAX_GAP_SEC):
    """상태 dict + 이번 샘플 → **새** 상태 dict (순수: 입력 state 를 변경하지 않는다).

    입력 state: new_dead_state() 형식. ts 는 센서 timestamp(초) — 벽시계를 읽지
    않으므로 로그 재생으로도 같은 결과가 나온다.
    반환 state["dead"] = [[4 bool],[4 bool]] (왼/오), state["note"] = 사람이 읽는 요약.

    래치 규칙: 후보로 표시된 채널은 '값이 실제로 변할 때'(|Δ| > change_counts) 해제된다.
    발이 떠서 창이 멈추는 것만으로는 해제하지 않는다 — 접지 사이클마다 표시가
    깜빡여 로그를 도배하는 것을 막기 위함.
    간극/시계 역행(restart)은 창만 다시 시작하고 래치는 유지한다 (연속성만 불인정).
    """
    feet = (list(left), list(right))
    biases = (list(left_pad_bias) if left_pad_bias else [0.0] * 4,
              list(right_pad_bias) if right_pad_bias else [0.0] * 4)
    if any(len(x) != 4 for x in feet + biases):
        raise ValueError("dead 진단 입력 길이 오류 (발·bias 각 4 필요)")
    ts = float(ts)
    window = float(dead_window_s)
    last = state.get("last_ts")
    restart = last is None or ts < float(last) or (ts - float(last)) > float(max_gap_s)
    out = {"ref": [[None] * 4, [None] * 4], "since": [[None] * 4, [None] * 4],
           "dead": [[False] * 4, [False] * 4], "last_ts": ts, "note": ""}
    excluded = []
    for f in (0, 1):
        vals = [float(v) for v in feet[f]]
        bias = [float(b) for b in biases[f]]
        corr = [vals[i] - bias[i] for i in range(4)]
        for i in range(4):
            if bias[i] >= float(exclude_bias):
                # 접촉 판정에서 뺀 채널(예: 0912 왼 index 2 상시 예압) — dead 판정도 하지 않는다
                excluded.append("%s%d" % ("LR"[f], i))
                out["ref"][f][i] = vals[i]
                continue
            others = sum(max(0.0, corr[j]) for j in range(4)
                         if j != i and bias[j] < float(exclude_bias))
            loaded = others >= float(pad_contact_counts)   # 같은 발의 다른 채널이 접지 중
            silent = corr[i] <= float(pad_contact_counts)  # 이 채널은 눌림 신호 없음
            prev_ref = state["ref"][f][i]
            prev_since = state["since"][f][i]
            prev_dead = bool(state["dead"][f][i])
            changed = prev_ref is not None and abs(vals[i] - float(prev_ref)) > float(change_counts)
            if changed:                       # 값이 움직였다 = 살아있다 → 래치 해제·창 재시작
                out["ref"][f][i] = vals[i]
                out["since"][f][i] = ts if (loaded and silent) else None
                continue
            out["ref"][f][i] = vals[i] if prev_ref is None else float(prev_ref)
            out["dead"][f][i] = prev_dead
            if not (loaded and silent):
                continue                      # 창 정지(since=None), 래치는 유지
            since = ts if (restart or prev_since is None) else float(prev_since)
            out["since"][f][i] = since
            if ts - since >= window:
                out["dead"][f][i] = True
    names = dead_channel_names(out["dead"])
    parts = []
    if names:
        parts.append("dead 후보 %s (같은 발 다른 패드 접지 중 %.1fs 무변화)"
                     % (",".join(names), window))
    if excluded:
        parts.append("판정제외(bias) %s" % ",".join(excluded))
    out["note"] = (" / ".join(parts) + " — 계측용, 접촉 판정 불변") if parts else ""
    return out


def dead_channel_names(mask):
    """[[4 bool],[4 bool]] → ["L2","R3"…] (좌→우, index 순)."""
    return ["%s%d" % ("LR"[f], i) for f in (0, 1) for i in range(4) if mask[f][i]]


def dead_transition_msg(prev_state, new_state):
    """dead 마스크가 **바뀐 샘플에서만** 경고 문자열, 아니면 None (로그 1회 규칙)."""
    prev = dead_channel_names(prev_state["dead"]) if prev_state else []
    new = dead_channel_names(new_state["dead"])
    started = [n for n in new if n not in prev]
    cleared = [n for n in prev if n not in new]
    if not started and not cleared:
        return None
    bits = []
    if started:
        bits.append("dead 후보 발생 %s" % ",".join(started))
    if cleared:
        bits.append("회복 %s" % ",".join(cleared))
    return ("[패드 진단] %s — 계측용(접촉 판정·임계 불변), 현장 점검 필요"
            % " / ".join(bits))


def attach_dead_diag(out, state, raw,
                     pad_contact_counts=PAD_CONTACT_COUNTS,
                     left_pad_bias=None, right_pad_bias=None,
                     dead_window_s=DEAD_WINDOW_SEC,
                     change_counts=DEAD_CHANGE_COUNTS):
    """out 에 dead_mask/dead_note 를 붙이고 (새 상태, 경고문|None, 예외|None) 반환.

    **진단은 절대 발행 경로를 막지 않는다** — 실패해도 예외를 올리지 않고
    dead_mask=전부 False + dead_note="diag 오류: …" 로 표시만 한다
    (조용히 정상값처럼 위장하지 않기 위해 note 에 오류를 남긴다).
    경고문은 dead 마스크가 바뀐 샘플에서만 not-None (로그 1회 규칙).
    """
    try:
        nxt = update_dead_channels(state, raw["left"], raw["right"], out["timestamp"],
                                   pad_contact_counts, left_pad_bias, right_pad_bias,
                                   dead_window_s, change_counts)
    except Exception as e:   # noqa: BLE001 — 진단 실패가 접촉 판정을 죽이면 안 됨
        out["dead_mask"] = [[False] * 4, [False] * 4]
        out["dead_note"] = "diag 오류: %s: %s" % (type(e).__name__, e)
        return state, None, e
    out["dead_mask"] = [list(r) for r in nxt["dead"]]
    out["dead_note"] = nxt["note"]
    return nxt, dead_transition_msg(state, nxt), None


def pressure_to_foot_force(raw, newton_per_count,
                           left_bias=LEFT_BIAS_COUNTS,
                           right_bias=RIGHT_BIAS_COUNTS,
                           pad_contact_counts=PAD_CONTACT_COUNTS,
                           left_pad_bias=None, right_pad_bias=None):
    """압력 JSON dict → foot_force dict (순수 함수, ROS 비의존).

    필수: timestamp(숫자), left/right(각 4개 숫자 리스트).
    형식이 어긋나면 KeyError/ValueError/TypeError — 호출측이 잡아서 스킵.
    바이어스 차감 후 음수는 0으로 바닥 처리 (무부하 노이즈가 음수 힘으로
    새어나가 접촉 판정을 흔들지 않게).
    """
    left = raw["left"]
    right = raw["right"]
    if len(left) != 4 or len(right) != 4:
        raise ValueError(f"left/right 길이 오류: {len(left)}/{len(right)} (4 필요)")
    left_sum = sum(int(v) for v in left)
    right_sum = sum(int(v) for v in right)
    lpb = list(left_pad_bias) if left_pad_bias else [0.0] * 4
    rpb = list(right_pad_bias) if right_pad_bias else [0.0] * 4
    if len(lpb) != 4 or len(rpb) != 4:
        raise ValueError("pad bias 길이 오류 (4 필요)")
    l_on = [int(int(v) - float(b) > pad_contact_counts) for v, b in zip(left, lpb)]
    r_on = [int(int(v) - float(b) > pad_contact_counts) for v, b in zip(right, rpb)]
    return {
        "timestamp": float(raw["timestamp"]),
        "left_n": newton_per_count * max(0.0, left_sum - left_bias),
        "right_n": newton_per_count * max(0.0, right_sum - right_bias),
        "left_raw_sum": left_sum,
        "right_raw_sum": right_sum,
        # 0910: 패드 단위 접지 — 하나라도 눌리면 True (브리지 contact2 가 이 필드를 우선 사용)
        "left_contact": bool(any(l_on)),
        "right_contact": bool(any(r_on)),
        "left_pads_on": int(sum(l_on)),
        "right_pads_on": int(sum(r_on)),
        "calibrated": True,
    }


def selftest():
    """ROS 없이 순수 변환 검증 — 실패 시 AssertionError로 즉사."""
    # 1) 정상 샘플: 합산과 뉴턴 스케일링
    sample = json.dumps({
        "timestamp": 123.456, "source": "ads1115",
        "values": [100, 200, 300, 400, 10, 20, 30, 40],
        "left": [100, 200, 300, 400], "right": [10, 20, 30, 40],
        "raw_ads1115": [0] * 8, "voltages": [0.0] * 8,
    })
    d = pressure_to_foot_force(json.loads(sample), NEWTON_PER_COUNT)
    assert d["left_raw_sum"] == 1000, d["left_raw_sum"]
    assert d["right_raw_sum"] == 100, d["right_raw_sum"]
    assert math.isclose(d["left_n"],
                        NEWTON_PER_COUNT * (1000 - LEFT_BIAS_COUNTS)), d["left_n"]
    assert math.isclose(d["right_n"],
                        NEWTON_PER_COUNT * (100 - RIGHT_BIAS_COUNTS)), d["right_n"]
    assert d["calibrated"] is True
    assert math.isclose(d["timestamp"], 123.456)

    # 1b) 체중 앵커: 0802 자립 스탠딩 카운트(L3356/R3559) → 172.9N (저울 17.62kg) ±0.3N
    st = pressure_to_foot_force(
        {"timestamp": 0.0, "left": [3356, 0, 0, 0], "right": [3559, 0, 0, 0]}, NEWTON_PER_COUNT)
    tot = st["left_n"] + st["right_n"]
    assert abs(tot - 172.9) < 0.3, f"체중 앵커 어긋남: {tot:.2f}N ≠ 172.9N (npc={NEWTON_PER_COUNT})"

    # 2) 계수 override 반영 확인 (바이어스 0으로 스케일만 검증)
    d2 = pressure_to_foot_force(json.loads(sample), 0.1, 0.0, 0.0)
    assert math.isclose(d2["left_n"], 100.0), d2["left_n"]

    # 2b) 바이어스 바닥 처리: 무부하 수준(바이어스 이하) 카운트 → 정확히 0N
    low = {"timestamp": 1.0, "left": [80, 80, 80, 80], "right": [10, 10, 10, 10]}
    d3 = pressure_to_foot_force(low, NEWTON_PER_COUNT)
    assert d3["left_n"] == 0.0 and d3["right_n"] == 0.0, (d3["left_n"], d3["right_n"])

    # 3) JSON 직렬화 왕복 + 계약 필수 키
    back = json.loads(json.dumps(d))
    for k in ("timestamp", "left_n", "right_n",
              "left_raw_sum", "right_raw_sum", "calibrated"):
        assert k in back, f"필수 키 누락: {k}"

    # 4) 손상 입력은 반드시 예외로 튕겨야 함 (노드가 잡아서 스킵)
    bads = [
        {"timestamp": 1.0, "right": [1, 2, 3, 4]},                    # left 누락
        {"timestamp": 1.0, "left": [1, 2, 3], "right": [1, 2, 3, 4]},  # 길이 3
        {"timestamp": 1.0, "left": [1, 2, "x", 4], "right": [1, 2, 3, 4]},  # 비숫자
        {"left": [1, 2, 3, 4], "right": [1, 2, 3, 4]},                # timestamp 누락
    ]
    for bad in bads:
        try:
            pressure_to_foot_force(bad, NEWTON_PER_COUNT)
        except (KeyError, ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"손상 입력이 통과됨: {bad}")

    # 4) 패드 단위 접지(0910): 한 패드만 눌려도 True, 예압 패드는 pad_bias 로 상쇄, 전부 무부하면 False
    o = pressure_to_foot_force({"timestamp": 1.0, "left": [0, 0, 700, 0], "right": [0, 0, 0, 0]}, 0.0265, 700.0, 0.0,
                               80.0, [0, 0, 700, 0], None)
    assert o["left_contact"] is False and o["right_contact"] is False and o["left_pads_on"] == 0, o
    o = pressure_to_foot_force({"timestamp": 1.0, "left": [0, 0, 700, 150], "right": [0, 90, 0, 0]}, 0.0265, 700.0, 0.0,
                               80.0, [0, 0, 700, 0], None)
    assert o["left_contact"] is True and o["left_pads_on"] == 1 and o["right_contact"] is True and o["right_pads_on"] == 1, o
    o = pressure_to_foot_force({"timestamp": 1.0, "left": [0, 0, 760, 0], "right": [0, 0, 0, 0]}, 0.0265, 700.0, 0.0,
                               80.0, [0, 0, 700, 0], None)
    assert o["left_contact"] is False, "예압 패드 +60 카운트 요동은 접지 아님"
    # 5) 0912: 기본 임계 200 — 150 카운트 스침은 눌림 아님, 250 은 눌림; 고장 패드(왼 index 2)는 bias 5000 으로 제외
    o = pressure_to_foot_force({"timestamp": 1.0, "left": [0, 150, 600, 0], "right": [0, 0, 250, 0]}, 0.0265, 0.0, 0.0,
                               PAD_CONTACT_COUNTS, [0, 0, 5000, 0], None)
    assert PAD_CONTACT_COUNTS == 200.0 and o["left_contact"] is False and o["left_pads_on"] == 0 and o["right_contact"] is True and o["right_pads_on"] == 1, o

    # ─────────── 6) 죽은 채널 진단 (0912 추가, 계측 전용 — 접촉 판정 불변) ───────────
    def feed(left, right, secs, st=None, t0=0.0, dt=0.1, collect=None, **kw):
        """dt 간격으로 secs 동안 샘플 주입 → 마지막 상태. left/right 는 리스트 또는 k→리스트 함수."""
        st = st if st is not None else new_dead_state()
        for k in range(int(round(secs / dt)) + 1):
            nxt = update_dead_channels(
                st, left(k) if callable(left) else left,
                right(k) if callable(right) else right, t0 + k * dt, **kw)
            if collect is not None:
                m = dead_transition_msg(st, nxt)
                if m:
                    collect.append((round(t0 + k * dt, 3), m))
            st = nxt
        return st

    NORMAL = [600, 600, 600, 600]
    # 6a) 정상: 네 채널 모두 접지 + 카운트가 움직임 → 10초 지나도 dead 없음
    st = feed(lambda k: [600 + 5 * (k % 2), 700 - 5 * (k % 2), 400 + 3 * (k % 2), 900 - 4 * (k % 2)],
              lambda k: [500 + 5 * (k % 2), 500, 500 + 5 * (k % 2), 500], 10.0)
    assert st["dead"] == [[False] * 4, [False] * 4], st
    assert st["note"] == "", st["note"]
    # 정상이지만 값이 '고정'인 채널: 눌림(600 > 임계)이면 후보 아님 (silent 조건)
    st = feed(NORMAL, NORMAL, 10.0)
    assert st["dead"] == [[False] * 4, [False] * 4], st

    # 6b) 죽은 채널: 왼 index 3 만 0, 같은 발 다른 패드는 600 → 5초 '이후'에만 후보
    dead_left = [600, 600, 600, 0]
    st4 = feed(dead_left, NORMAL, 4.9)
    assert st4["dead"] == [[False] * 4, [False] * 4], ("5초 전에 후보로 찍힘", st4["dead"])
    st5 = feed(dead_left, NORMAL, 5.0)
    assert st5["dead"] == [[False, False, False, True], [False] * 4], st5["dead"]
    assert dead_channel_names(st5["dead"]) == ["L3"], st5["note"]
    assert "L3" in st5["note"] and "계측용" in st5["note"], st5["note"]
    # 0912 관측 그대로(왼 2·3, 오른 3 이 0): 세 채널이 후보로 표시되어야 한다
    st_obs = feed([600, 900, 0, 0], [200 + 1, 700, 500, 0], 6.0)
    assert dead_channel_names(st_obs["dead"]) == ["L2", "L3", "R3"], st_obs["note"]
    # 래치: 발이 떠서 전 채널 0 이 되어도 '값이 변한' 채널만 해제된다 (L3 은 계속 0 → 유지)
    st_air = feed([0, 0, 0, 0], NORMAL, 3.0, st=copy.deepcopy(st5), t0=5.1)
    assert st_air["dead"][0][3] is True, "창 정지만으로 래치가 풀렸다"
    # 그 채널이 실제로 반응하면(600) 해제
    st_back = feed(NORMAL, NORMAL, 0.5, st=copy.deepcopy(st_air), t0=8.2)
    assert st_back["dead"] == [[False] * 4, [False] * 4], st_back["dead"]

    # 6c) 전 채널 0 = 단순 무접촉 → 20초가 지나도 dead 아님 (다른 채널 접지 증거 없음)
    st0 = feed([0, 0, 0, 0], [0, 0, 0, 0], 20.0)
    assert st0["dead"] == [[False] * 4, [False] * 4], st0["dead"]
    assert st0["note"] == "", st0["note"]
    # 한쪽 발만 접지: 뜬 발(오른)은 후보 아님, 접지한 발의 0 채널만 후보
    st1 = feed(dead_left, [0, 0, 0, 0], 6.0)
    assert dead_channel_names(st1["dead"]) == ["L3"], st1["note"]

    # 6d) bias 로 판정 제외한 채널(운영 관례 5000)은 dead 판정에서도 제외
    stx = feed([600, 600, 0, 0], NORMAL, 6.0, left_pad_bias=[0.0, 0.0, 5000.0, 0.0])
    assert dead_channel_names(stx["dead"]) == ["L3"], ("제외 채널이 후보로 찍힘", stx["note"])
    assert "판정제외(bias) L2" in stx["note"], stx["note"]
    # 제외 채널의 카운트는 '다른 채널 접지 증거'에도 넣지 않는다 (상시 예압이 접지를 위장하지 않게)
    sty = feed([0, 0, 900, 0], [0, 0, 0, 0], 6.0, left_pad_bias=[0.0, 0.0, 0.0, 0.0])
    assert dead_channel_names(sty["dead"]) == ["L0", "L1", "L3"], sty["note"]
    stz = feed([0, 0, 900, 0], [0, 0, 0, 0], 6.0, left_pad_bias=[0.0, 0.0, 5000.0, 0.0])
    assert stz["dead"] == [[False] * 4, [False] * 4], ("제외 채널이 접지 증거로 쓰임", stz["note"])

    # 6e) '0 아님 + 변화 없음' 도 후보 / 같은 대역에서 요동치면 후보 아님 (대조군)
    st_flat = feed([600, 600, 600, 150], NORMAL, 6.0)
    assert dead_channel_names(st_flat["dead"]) == ["L3"], st_flat["note"]
    st_wob = feed(lambda k: [600, 600, 600, 100 + 90 * (k % 2)], NORMAL, 6.0)
    assert st_wob["dead"] == [[False] * 4, [False] * 4], ("요동 채널이 후보로 찍힘", st_wob["dead"])

    # 6f) 로그 경고는 상태 전이 시 1회만
    msgs = []
    st_l = feed(dead_left, NORMAL, 8.0, collect=msgs)
    assert len(msgs) == 1 and msgs[0][0] == 5.0, msgs
    assert "dead 후보 발생 L3" in msgs[0][1], msgs[0][1]
    st_l = feed(NORMAL, NORMAL, 1.0, st=st_l, t0=8.1, collect=msgs)
    assert len(msgs) == 2 and "회복 L3" in msgs[1][1], msgs
    assert dead_transition_msg(st_l, st_l) is None

    # 6g) 샘플 간극·시계 역행은 창만 재시작 (연속 N초를 위조하지 않음)
    st_g = feed(dead_left, NORMAL, 3.0)
    st_g = feed(dead_left, NORMAL, 4.0, st=st_g, t0=6.0)   # 3.0 → 6.0 = 3초 간극
    assert st_g["dead"] == [[False] * 4, [False] * 4], ("간극이 연속으로 집계됨", st_g["dead"])
    st_g = feed(dead_left, NORMAL, 1.1, st=st_g, t0=10.1)
    assert st_g["dead"][0][3] is True, st_g["dead"]
    st_b = feed(dead_left, NORMAL, 4.9)
    st_b = feed(dead_left, NORMAL, 4.0, st=st_b, t0=0.0)   # 시계 역행
    assert st_b["dead"] == [[False] * 4, [False] * 4], ("시계 역행이 연속으로 집계됨", st_b["dead"])

    # 6h) 순수 함수: 입력 state 를 변경하지 않는다 (같은 입력 → 같은 출력)
    base = feed(dead_left, NORMAL, 4.9)
    snap = copy.deepcopy(base)
    n1 = update_dead_channels(base, dead_left, NORMAL, 5.0)
    assert base == snap, "입력 state 가 변경됨 (순수성 위반)"
    n2 = update_dead_channels(base, dead_left, NORMAL, 5.0)
    assert n1 == n2, "같은 입력에 다른 출력"
    assert n1["dead"][0][3] is True

    # 6i) 진단은 접촉 판정에 영향을 주지 않는다 + 출력 JSON 계약(기존 키 유지 + 신규 2키)
    pre = pressure_to_foot_force({"timestamp": 9.0, "left": dead_left, "right": NORMAL},
                                 NEWTON_PER_COUNT, 0.0, 0.0, PAD_CONTACT_COUNTS,
                                 [0, 0, 0, 0], None)
    out = dict(pre)
    out["dead_mask"] = [list(r) for r in n1["dead"]]
    out["dead_note"] = n1["note"]
    assert pressure_to_foot_force({"timestamp": 9.0, "left": dead_left, "right": NORMAL},
                                  NEWTON_PER_COUNT, 0.0, 0.0, PAD_CONTACT_COUNTS,
                                  [0, 0, 0, 0], None) == pre, "접촉 변환이 진단에 의해 달라짐"
    assert out["left_contact"] is True and out["left_pads_on"] == 3, out
    back = json.loads(json.dumps(out))
    for k in ("timestamp", "left_n", "right_n", "left_raw_sum", "right_raw_sum",
              "left_contact", "right_contact", "left_pads_on", "right_pads_on",
              "calibrated", "dead_mask", "dead_note"):
        assert k in back, f"필수 키 누락: {k}"
    assert back["dead_mask"] == [[False, False, False, True], [False] * 4], back["dead_mask"]
    assert all(isinstance(v, bool) for row in back["dead_mask"] for v in row)
    assert isinstance(back["dead_note"], str)
    # 손상 입력은 진단 함수도 예외로 튕긴다 (노드가 잡아 진단만 중단)
    for bad in ([600, 600, 600], [600, 600, 600, 600, 600]):
        try:
            update_dead_channels(new_dead_state(), bad, NORMAL, 0.0)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(f"손상 입력이 진단을 통과함: {bad}")

    # 6j) 노드가 쓰는 경로(attach_dead_diag): 키 부착·전이 1회·실패해도 예외 없음
    raw_ok = {"timestamp": 0.0, "left": dead_left, "right": NORMAL}
    s, warn, err = new_dead_state(), None, None
    seen = []
    for k in range(61):                      # 0.0 ~ 6.0 s
        raw_ok["timestamp"] = round(k * 0.1, 3)
        o = pressure_to_foot_force(raw_ok, NEWTON_PER_COUNT)
        s, warn, err = attach_dead_diag(o, s, raw_ok)
        assert err is None and o["dead_mask"] is not None
        if warn:
            seen.append((raw_ok["timestamp"], warn))
    assert len(seen) == 1 and seen[0][0] == 5.0 and "L3" in seen[0][1], seen
    assert o["dead_mask"] == [[False, False, False, True], [False] * 4], o["dead_mask"]
    assert json.loads(json.dumps(o))["dead_note"] == o["dead_note"]
    bad_o = {"timestamp": 7.0}
    s2, warn2, err2 = attach_dead_diag(bad_o, s, {"left": [1, 2, 3], "right": NORMAL})
    assert err2 is not None and warn2 is None and s2 is s, (err2, warn2)
    assert bad_o["dead_mask"] == [[False] * 4, [False] * 4] and bad_o["dead_note"].startswith("diag 오류"), bad_o

    print("[selftest] foot_force_adapter: 합산·뉴턴 스케일·손상입력 거부·패드 단위 접지(하나라도 눌리면 True, 예압 상쇄)·임계 200·고장 패드 제외"
          " + 죽은 채널 진단(정상/죽음/전채널0=무접촉/bias 제외/무변화 대조/전이 1회/간극·역행/순수성/접촉 판정 불변) ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="압력 JSON → foot_force JSON(std_msgs/String) 어댑터")
    ap.add_argument("--selftest", action="store_true",
                    help="ROS 없이 변환 로직만 검증하고 종료")
    a, _ = ap.parse_known_args()  # --ros-args 통과 허용
    if a.selftest:
        return selftest()

    # rclpy는 여기서만 import — selftest 경로가 ROS 미소싱 환경에서도 돌게 함
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from std_msgs.msg import String

    rclpy.init()
    node = Node("foot_force_adapter")
    node.declare_parameter("left_npc_scale", 1.0)   # 0911: per-foot 스케일(왼발 저평가 보정 — 균형스탠스 표본으로 정함, 기본 1.0=무변화)
    node.declare_parameter("right_npc_scale", 1.0)
    node.declare_parameter("newton_per_count", NEWTON_PER_COUNT)
    node.declare_parameter("left_bias_counts", LEFT_BIAS_COUNTS)
    node.declare_parameter("right_bias_counts", RIGHT_BIAS_COUNTS)
    node.declare_parameter("pad_contact_counts", PAD_CONTACT_COUNTS)
    node.declare_parameter("left_pad_bias", [0.0, 0.0, 0.0, 0.0])    # 패드별 무부하 카운트(예압 패드 타어) — 0910 왼[2] 700
    node.declare_parameter("right_pad_bias", [0.0, 0.0, 0.0, 0.0])
    # 죽은 채널 진단(계측 전용) — 접촉 판정·임계·bias 동작에는 관여하지 않는다
    node.declare_parameter("dead_window_s", DEAD_WINDOW_SEC)
    node.declare_parameter("dead_change_counts", DEAD_CHANGE_COUNTS)
    qos = QoSProfile(depth=QOS_DEPTH)
    pub = node.create_publisher(String, OUT_TOPIC, qos)
    stat = {"ok": 0, "err": 0, "diag_err": 0}
    dstate = {"s": new_dead_state()}

    def cb(msg):
        # 파라미터를 매 콜백 재조회 (10Hz라 저렴) — 보정 작업 중
        # `ros2 param set … newton_per_count X` 가 즉시 반영되게 함
        npc = float(node.get_parameter("newton_per_count").value)
        lsc = float(node.get_parameter("left_npc_scale").value); rsc = float(node.get_parameter("right_npc_scale").value)
        lb = float(node.get_parameter("left_bias_counts").value)
        rb = float(node.get_parameter("right_bias_counts").value)
        pcc = float(node.get_parameter("pad_contact_counts").value)
        lpb = [float(v) for v in node.get_parameter("left_pad_bias").value]
        rpb = [float(v) for v in node.get_parameter("right_pad_bias").value]
        try:
            raw = json.loads(msg.data)
            out = pressure_to_foot_force(raw, npc, lb, rb, pcc, lpb, rpb)
            out["left_n"] *= lsc; out["right_n"] *= rsc   # per-foot 스케일 (접촉 이진은 패드 카운트 기반이라 불변)
        except Exception as e:
            stat["err"] += 1
            if stat["err"] % ERR_LOG_EVERY == 1:  # 1·21·41…번째만 로그
                node.get_logger().warning(
                    f"압력 JSON 파싱 실패 누적 {stat['err']}건 "
                    f"(최근: {type(e).__name__}: {e}) — 스킵")
            return
        # 죽은 채널 진단(계측 전용): out 에 dead_mask/dead_note 만 덧붙인다.
        # 접촉 판정은 위에서 이미 끝났고 여기서 바뀌지 않는다. 진단 실패는 발행을 막지 않는다.
        try:    # 진단 파라미터 조회 실패도 발행을 막지 않게 (기본값으로 진행)
            dws = float(node.get_parameter("dead_window_s").value)
            dcc = float(node.get_parameter("dead_change_counts").value)
        except Exception:
            dws, dcc = DEAD_WINDOW_SEC, DEAD_CHANGE_COUNTS
        dstate["s"], warn, diag_e = attach_dead_diag(
            out, dstate["s"], raw, pcc, lpb, rpb, dws, dcc)
        if diag_e is not None:
            stat["diag_err"] += 1
            if stat["diag_err"] % ERR_LOG_EVERY == 1:
                node.get_logger().warning(
                    f"죽은 채널 진단 실패 누적 {stat['diag_err']}건 "
                    f"(최근: {type(diag_e).__name__}: {diag_e}) — 진단만 중단, 접촉 판정은 정상")
        elif warn:
            node.get_logger().warning(warn)   # 상태 전이 시 1회만
        stat["ok"] += 1
        pub.publish(String(data=json.dumps(out)))

    node.create_subscription(String, IN_TOPIC, cb, qos)
    node.get_logger().info(
        f"{IN_TOPIC} → {OUT_TOPIC} 변환 시작 (newton_per_count="
        f"{float(node.get_parameter('newton_per_count').value)}, 바이어스 "
        f"L{float(node.get_parameter('left_bias_counts').value):.0f}/"
        f"R{float(node.get_parameter('right_bias_counts').value):.0f} — "
        f"실측 보정 2026-08-02)")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
