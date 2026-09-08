#!/usr/bin/env python3
"""발바닥 패드 8점 압력 → 압력중심(CoP) = 정적 서기에서 무게중심의 지면 투영 (0907).

실물 무게중심 전후 위치를 '역산 가정(+16mm)' 이 아니라 측정으로 확정한다. 패드 좌표는 URDF(final3) 의
발목롤 링크 기준: 앞 패드 z=+70.6mm, 뒤 패드 z=−50.9mm, 좌우 x=±25mm (`rl_walking/assets/biped12_final3.urdf`).

사용 (젯슨, ROS2 소싱):
  python3 cop_from_pads.py --identify                 # 패드를 하나씩 눌러 left[i]/right[i] 인덱스 ↔ 위치 확인 (실시간 막대)
  python3 cop_from_pads.py --live --seconds 10 --tare ~/logs/pads_hang.jsonl   # 서 있는 10초 기록·CoP 출력 (무부하 기록으로 바이어스 차감)
  python3 cop_from_pads.py --live --seconds 5 --save ~/logs/pads_hang.jsonl --no-report   # 매달린 상태 무부하 기록만
사용 (아무 PC):
  python3 cop_from_pads.py --file pads_stand.jsonl --tare pads_hang.jsonl [--layout lf,rf,lb,rb] [--sim_ref_mm -12.3]
  python3 cop_from_pads.py --selftest

--layout: left[0..3]/right[0..3] 이 어느 패드인지 (lf=왼앞 rf=오른앞 lb=왼뒤 rb=오른뒤; 발 기준 좌우). 기본은 URDF 링크 순서 추정 —
  **--identify 로 확인한 순서를 반드시 넣을 것** (틀리면 앞뒤가 뒤집힌다).
출력: 발별·전체 CoP 전후[mm, +앞 = 발목롤 축 앞], 좌우[mm], 앞뒤 하중 비율. --sim_ref_mm 을 주면 시뮬 명목 CoP 와의 차이
(= 실물 무게중심이 모델보다 얼마나 앞/뒤에 있는가)를 함께 낸다. 시뮬 기준값: eval_drift_dr --nominal --settle_s 2 --steps 5 --cop.
"""
import argparse
import json
import math
import sys
import time

#: URDF final3 패드 좌표 [m], 발목롤 링크 기준 (z=전후 +앞, x=좌우) — 양발 동일
PAD_XYZ = {
    "lf": (+0.025125, +0.070601), "rf": (-0.024875, +0.070601),   # (lateral x, fore-aft z)
    "lb": (+0.025125, -0.050933), "rb": (-0.024875, -0.050933),
}
#: 0908 --identify 실측 확정 (매달림, 8패드 개별 가압 130초 · 겹침 0 · 양발 동일 배선):
#: left[0]=앞바깥 left[1]=뒤바깥 left[2]=앞안쪽 left[3]=뒤안쪽,
#: right[0]=앞안쪽 right[1]=뒤안쪽 right[2]=앞바깥 right[3]=뒤바깥.
#: 로봇 좌우로 옮기면 양발 모두 인덱스 순서 lf,lb,rf,rb 다.
#: **종전 기본값 "lf,rf,lb,rb" 는 1·2번이 뒤바뀐 추정치였다** — 그대로 쓰면 뒤 패드를
#: 앞으로 계산해 CoP 전후가 오염된다 (이 도구가 경고하던 바로 그 사고).
DEFAULT_LAYOUT = "lf,lb,rf,rb"
TOPIC = "/humanoid/pressure/data"


def parse_layout(s):
    names = [t.strip() for t in s.split(",")]
    assert sorted(names) == sorted(PAD_XYZ), f"--layout 은 lf,rf,lb,rb 의 순열이어야 함: {s}"
    return names


def cop_of(vals, layout):
    """4패드 카운트(바이어스 차감 후) → (전후 mm, 좌우 mm, 앞 비율, 합). 합이 0 이면 None."""
    tot = sum(max(0.0, v) for v in vals)
    if tot <= 0.0:
        return None
    z = sum(max(0.0, v) * PAD_XYZ[n][1] for v, n in zip(vals, layout)) / tot
    x = sum(max(0.0, v) * PAD_XYZ[n][0] for v, n in zip(vals, layout)) / tot
    front = sum(max(0.0, v) for v, n in zip(vals, layout) if n in ("lf", "rf")) / tot
    return z * 1000.0, x * 1000.0, front, tot


def summarize(msgs, tare, layout):
    """msgs: [{"left":[4],"right":[4],...}], tare: 같은 형식의 무부하 평균 dict 또는 None."""
    L = [[float(v) for v in m["left"]] for m in msgs]
    R = [[float(v) for v in m["right"]] for m in msgs]
    n = len(L)
    if n == 0:
        raise SystemExit("표본 없음")
    tl = tare["left"] if tare else [0.0] * 4
    tr = tare["right"] if tare else [0.0] * 4
    ml = [sum(L[i][k] for i in range(n)) / n - tl[k] for k in range(4)]
    mr = [sum(R[i][k] for i in range(n)) / n - tr[k] for k in range(4)]
    cl, cr = cop_of(ml, layout), cop_of(mr, layout)
    out = {"n": n, "left_counts": ml, "right_counts": mr, "left": cl, "right": cr}
    wl, wr = (cl[3] if cl else 0.0), (cr[3] if cr else 0.0)
    if wl + wr > 0.0:
        zl, zr = (cl[0] if cl else 0.0), (cr[0] if cr else 0.0)
        fl, fr = (cl[2] if cl else 0.0), (cr[2] if cr else 0.0)
        out["total_z_mm"] = (zl * wl + zr * wr) / (wl + wr)
        out["front_share"] = (fl * wl + fr * wr) / (wl + wr)
        out["left_share"] = wl / (wl + wr)
    # 시계열 표준편차(흔들림) — 전후 CoP 의 틱별 산포
    zs = []
    for i in range(n):
        a = cop_of([L[i][k] - tl[k] for k in range(4)], layout)
        b = cop_of([R[i][k] - tr[k] for k in range(4)], layout)
        if a and b:
            zs.append((a[0] * a[3] + b[0] * b[3]) / (a[3] + b[3]))
    if len(zs) > 1:
        mu = sum(zs) / len(zs)
        out["total_z_std_mm"] = math.sqrt(sum((z - mu) ** 2 for z in zs) / (len(zs) - 1))
    return out


def report(s, layout, sim_ref_mm=None):
    print(f"패드 레이아웃 left/right[0..3] = {layout} (틀리면 앞뒤가 뒤집힘 — --identify 로 확인)")
    print(f"표본 {s['n']}개 | 바이어스 차감 후 카운트 L={[round(v) for v in s['left_counts']]} R={[round(v) for v in s['right_counts']]}")
    for side in ("left", "right"):
        c = s[side]
        if c is None:
            print(f"  {side:5s}: 하중 없음"); continue
        print(f"  {side:5s}: CoP 전후 {c[0]:+6.1f}mm (발목롤 축 기준, +앞) · 좌우 {c[1]:+5.1f}mm · 앞 패드 하중 {c[2]*100:4.0f}% · 합 {c[3]:.0f}카운트")
    if "total_z_mm" in s:
        print(f"  전체 : CoP 전후 {s['total_z_mm']:+6.1f}mm · 앞 하중 {s['front_share']*100:.0f}% · 왼발 하중 {s['left_share']*100:.0f}%"
              + (f" · 틱별 산포 σ {s['total_z_std_mm']:.1f}mm" if "total_z_std_mm" in s else ""))
        if sim_ref_mm is not None:
            d = s["total_z_mm"] - sim_ref_mm
            print(f"  시뮬 명목 CoP {sim_ref_mm:+.1f}mm 대비 실물 {d:+.1f}mm → 실물 무게중심이 모델보다 {abs(d):.0f}mm {'앞' if d > 0 else '뒤'}"
                  f" (판정 가정 com+16 과 비교: {'일치 방향' if d > 0 else '반대 방향'})")


def load_jsonl(path):
    msgs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if isinstance(o, dict) and "left" in o and "right" in o and len(o["left"]) == 4 and len(o["right"]) == 4:
                msgs.append(o)
    return msgs


def tare_from(path):
    msgs = load_jsonl(path)
    if not msgs:
        raise SystemExit(f"무부하 기록에 표본 없음: {path}")
    n = len(msgs)
    return {"left": [sum(float(m["left"][k]) for m in msgs) / n for k in range(4)],
            "right": [sum(float(m["right"][k]) for m in msgs) / n for k in range(4)], "n": n}


def selftest():
    # 수학 검증용 고정 레이아웃 — **배포 기본값과 일부러 분리한다.** 예전엔 이 자리에
    # DEFAULT_LAYOUT 을 썼는데, 그러면 기본값을 실측으로 고치는 순간 기대값이 같이
    # 움직여 테스트가 아무것도 지키지 못한다 (0908 실제로 그렇게 깨졌다).
    layout = parse_layout("lf,rf,lb,rb")
    # 앞 패드 2개에 300, 뒤 패드 2개에 100 → CoP z = (600·70.601 − 200·50.933)/800 = +40.22mm, 앞 하중 75%
    m = {"left": [300, 300, 100, 100], "right": [300, 300, 100, 100]}
    s = summarize([m] * 5, None, layout)
    assert abs(s["total_z_mm"] - 40.22) < 0.05, s["total_z_mm"]
    assert abs(s["front_share"] - 0.75) < 1e-9
    # 바이어스 차감: 무부하 50 을 빼면 (250,250,50,50) → z = (500·70.601 − 100·50.933)/600 = +50.35
    t = {"left": [50] * 4, "right": [50] * 4}
    s2 = summarize([m] * 3, t, layout)
    assert abs(s2["total_z_mm"] - 50.35) < 0.05, s2["total_z_mm"]
    # 레이아웃 순열: lb,rb,lf,rf 로 읽으면 (300,300 이 뒤 패드) → z = (200·70.601 − 600·50.933)/800 = −20.55mm (앞뒤 뒤집힘)
    s3 = summarize([m] * 3, None, parse_layout("lb,rb,lf,rf"))
    assert abs(s3["total_z_mm"] + 20.55) < 0.05, s3["total_z_mm"]
    # 좌우: 왼발만 하중 → left_share 1.0
    s4 = summarize([{"left": [100] * 4, "right": [0] * 4}] * 2, None, layout)
    assert abs(s4["left_share"] - 1.0) < 1e-9 and s4["right"] is None
    # 배포 기본값은 0908 --identify 실측값이어야 한다 (추정치로 되돌아가는 것 차단)
    assert DEFAULT_LAYOUT == "lf,lb,rf,rb", DEFAULT_LAYOUT
    print("[selftest] cop_from_pads: CoP 전후·앞 비율·바이어스 차감·레이아웃 순열·"
          "좌우 하중·배포 기본값 — ALL PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description="패드 8점 압력 → 압력중심(CoP)")
    ap.add_argument("--file", help="압력 메시지 JSONL (한 줄 = 메시지 JSON)")
    ap.add_argument("--live", action="store_true", help=f"ROS2 {TOPIC} 를 --seconds 동안 구독")
    ap.add_argument("--identify", action="store_true", help="실시간 패드별 막대 — 패드를 눌러 인덱스 확인")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--save", help="live 로 받은 원본 메시지를 JSONL 로 저장 (기본: ~/logs/pads_<stamp>.jsonl)")
    ap.add_argument("--no-report", action="store_true", help="기록만 하고 CoP 계산 생략")
    ap.add_argument("--tare", help="무부하(매달림) 기록 JSONL — 패드별 바이어스 차감")
    ap.add_argument("--layout", default=DEFAULT_LAYOUT)
    ap.add_argument("--sim_ref_mm", type=float, default=None, help="시뮬 명목 CoP 전후[mm] (eval_drift_dr --cop 출력)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    layout = parse_layout(a.layout)
    tare = tare_from(a.tare) if a.tare else None
    if a.file:
        msgs = load_jsonl(a.file)
        report(summarize(msgs, tare, layout), layout, a.sim_ref_mm)
        return 0
    if not (a.live or a.identify):
        ap.error("--file 또는 --live/--identify 필요")
    import os
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    rclpy.init()
    node = Node("cop_from_pads")
    got = []
    def cb(msg):
        try:
            o = json.loads(msg.data)
            if len(o.get("left", [])) == 4 and len(o.get("right", [])) == 4:
                got.append(o)
        except (ValueError, AttributeError):
            pass
    node.create_subscription(String, TOPIC, cb, 10)
    t0 = time.time()
    if a.identify:
        print("패드를 하나씩 누르세요 — 값이 오르는 인덱스가 그 패드. Ctrl-C 로 종료.")
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                if got:
                    o = got[-1]
                    bar = lambda v: "█" * int(min(40, max(0.0, float(v)) / 60.0))
                    line = " | ".join(f"L{k}:{float(o['left'][k]):5.0f} {bar(o['left'][k]):<8}" for k in range(4)) + " || " \
                        + " | ".join(f"R{k}:{float(o['right'][k]):5.0f} {bar(o['right'][k]):<8}" for k in range(4))
                    print("\r" + line[:200], end="", flush=True)
        except KeyboardInterrupt:
            print()
        finally:
            node.destroy_node(); rclpy.shutdown()
        return 0
    while rclpy.ok() and time.time() - t0 < a.seconds:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node(); rclpy.shutdown()
    save = a.save or os.path.expanduser(f"~/logs/pads_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    os.makedirs(os.path.dirname(save), exist_ok=True)
    with open(save, "w", encoding="utf-8") as f:
        for o in got:
            f.write(json.dumps(o) + "\n")
    print(f"{len(got)}개 메시지 저장: {save}")
    if not a.no_report:
        report(summarize(got, tare, layout), layout, a.sim_ref_mm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
