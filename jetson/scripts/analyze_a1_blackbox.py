#!/usr/bin/env python3
"""A1 실물 블랙박스(rl_bridge_*.jsonl, 0910 기록 필드 포함) 요약 — 8개 질문용 지표 + 실패 순서.
사용: python3 jetson/scripts/analyze_a1_blackbox.py <jsonl> [--label 이름]
구간: 공중 = armed & 양발 무접촉 & 발력<5N · 고원 = armed & 양발 접촉 & 발력 40~70% · 정지 직전 = 마지막 stop_all 전 10 s."""
import argparse, json, math, statistics as st, time


def hms(t): return time.strftime("%H:%M:%S", time.localtime(t))


def rms(x): return math.sqrt(sum(v * v for v in x) / len(x)) if x else float("nan")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--label", default=""); a = ap.parse_args()
    rows, ev = [], []
    for l in open(a.path):
        try: d = json.loads(l)
        except ValueError: continue
        (ev if "event" in d else rows).append(d)
    rows = [r for r in rows if "qpos_rad" in r and r.get("armed")]
    if not rows: print("armed 틱 없음"); return
    # 0910 20:20 정정: 러너 JOINT_ORDER 는 좌우 교대(interleaved). 이전 판(0~5 왼/6~11 오른)은 오독 → '비대칭 자세' 결론 폐기
    J = ["Lhf", "Rhf", "Lha", "Rha", "Lhr", "Rhr", "Lkn", "Rkn", "Laf", "Raf", "Lar", "Rar"]
    I = {n: i for i, n in enumerate(J)}
    W = 172.9
    def pct(r): fn = r["foot_n"]; return 100 * (fn[0] + fn[1]) / W
    air = [r for r in rows if r["contact"] == [0.0, 0.0] and (r["foot_n"][0] + r["foot_n"][1]) < 5]
    plat = [r for r in rows if r["contact"] == [1.0, 1.0] and 40 <= pct(r) <= 70]
    stops = [e for e in ev if e.get("event") == "stop_all"]
    print(f"# {a.label or a.path}  armed {len(rows)}틱 {rows[-1]['timestamp']-rows[0]['timestamp']:.0f}s  이벤트 {[(hms(e['timestamp']), e['event'], (e.get('reason') or e.get('why') or '')[:30]) for e in ev]}")
    def qmean(R, i): return st.mean(math.degrees(r["qpos_rad"][i]) for r in R)
    if air:
        print(f"공중({len(air)}틱 {len(air)/50:.0f}s): 힙f L/R {qmean(air,I['Lhf']):+.1f}/{qmean(air,I['Rhf']):+.1f} · 힙a {qmean(air,I['Lha']):+.1f}/{qmean(air,I['Rha']):+.1f} · 힙r {qmean(air,I['Lhr']):+.1f}/{qmean(air,I['Rhr']):+.1f} · 무릎 {qmean(air,I['Lkn']):+.1f}/{qmean(air,I['Rkn']):+.1f} · 발목f {qmean(air,I['Laf']):+.1f}/{qmean(air,I['Raf']):+.1f} · 발목r {qmean(air,I['Lar']):+.1f}/{qmean(air,I['Rar']):+.1f} · 롤 {st.mean(r['rp_deg'][0] for r in air):+.2f}° 피치 {st.mean(r['rp_deg'][1] for r in air):+.2f}°")
    if plat:
        g = [r["gyro_rad_s"] for r in plat]; ra = [r["raw_action"] for r in plat]
        d1 = [math.sqrt(sum((x - y) ** 2 for x, y in zip(ra[i], ra[i - 1])) / 12) for i in range(1, len(ra))]
        print(f"고원 40~70%({len(plat)}틱): 피치 {st.mean(r['rp_deg'][1] for r in plat):+.2f}° 롤 {st.mean(r['rp_deg'][0] for r in plat):+.2f}° · L/R 발력 {st.mean(r['foot_n'][0] for r in plat):.0f}/{st.mean(r['foot_n'][1] for r in plat):.0f} N · gyro std {st.pstdev([x[0] for x in g]):.4f}/{st.pstdev([x[1] for x in g]):.4f} · action-rate {rms(d1):.4f}")
        print(f"   고원 관절각: 힙f L/R {qmean(plat,I['Lhf']):+.1f}/{qmean(plat,I['Rhf']):+.1f} · 힙a {qmean(plat,I['Lha']):+.1f}/{qmean(plat,I['Rha']):+.1f} · 힙r {qmean(plat,I['Lhr']):+.1f}/{qmean(plat,I['Rhr']):+.1f} · 무릎 {qmean(plat,I['Lkn']):+.1f}/{qmean(plat,I['Rkn']):+.1f} · 발목f {qmean(plat,I['Laf']):+.1f}/{qmean(plat,I['Raf']):+.1f} · 발목r {qmean(plat,I['Lar']):+.1f}/{qmean(plat,I['Rar']):+.1f}")
        tq = [abs(t) for r in plat for t in r["tau_est_nm"] if t is not None]; print(f"   τ 최대 {max(tq):.2f} Nm · 추종오차 최대 {max(max(abs(math.degrees(q)-t) for q,t in zip(r['qpos_rad'], (r['targets_deg_joint'] if isinstance(r['targets_deg_joint'],list) else list(r['targets_deg_joint'].values())))) for r in plat):.2f}°")
    # 실패 순서: 마지막 stop 전 12 s, 1 s 간격
    if stops:
        t_end = stops[-1]["timestamp"]; seg = [r for r in rows if t_end - 12 <= r["timestamp"] <= t_end]
        print("정지 전 12 s (1 s 간격): 시각 발력% L/R 접촉 롤 피치 | 힙fL 힙fR 무릎L 무릎R 발목rL 발목rR | gyroZ")
        last = 0
        for r in seg:
            if r["timestamp"] - last < 1.0: continue
            last = r["timestamp"]; q = [math.degrees(v) for v in r["qpos_rad"]]
            print(f"   {hms(r['timestamp'])} {pct(r):5.1f} {r['foot_n'][0]:3.0f}/{r['foot_n'][1]:3.0f} {r['contact']} {r['rp_deg'][0]:+5.2f} {r['rp_deg'][1]:+5.2f} | {q[I['Lhf']]:+5.1f} {q[I['Rhf']]:+5.1f} {q[I['Lkn']]:+5.1f} {q[I['Rkn']]:+5.1f} {q[I['Lar']]:+5.1f} {q[I['Rar']]:+5.1f} | {r['gyro_rad_s'][2]:+.3f}")
        # 어느 발이 먼저 빠졌나
        first = next((r for r in seg if r["contact"] != [1.0, 1.0] and any(pct(x) > 30 for x in seg[:seg.index(r)])), None)
        if first: print(f"   → 먼저 빠진 발: {'왼발' if first['contact'][0]==0.0 else '오른발'} ({hms(first['timestamp'])}), 그때 롤 {first['rp_deg'][0]:+.2f}° 피치 {first['rp_deg'][1]:+.2f}°")


if __name__ == "__main__":
    main()
