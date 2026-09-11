#!/usr/bin/env python3
"""전이 중 앞뒤 CoP·피치·발력 궤적 — 실패 방향(앞/뒤)·최대 발력·CoM 벽 여부 판정 (2026-09-11).
사용: python3 analyze_cop_transfer.py <pads.jsonl> <rl_bridge.jsonl> [--from HH:MM:SS --to HH:MM:SS]
패드 앞뒤 좌표(0908 확정): 앞 +70.6 / 뒤 −50.9mm (발목축 기준). layout lf,lb,rf,rb (각 발 [앞,뒤,앞,뒤])."""
import json, sys, time, argparse
F, B = 70.6, -50.9
XF = [F, B, F, B]  # 각 발 4패드 앞뒤


def cop(vals):
    tot = sum(max(0.0, v) for v in vals)
    return (sum(max(0.0, v) * x for v, x in zip(vals, XF)) / tot) if tot > 0 else None


def hms(t): return time.strftime("%H:%M:%S", time.localtime(t))


def load_pads(p):
    rows = []
    for l in open(p):
        try: d = json.loads(l)
        except ValueError: continue
        m = d.get("msg", d)
        if isinstance(m, str):
            try: m = json.loads(m)
            except ValueError: continue
        if isinstance(m, dict) and "left" in m and "timestamp" in m: rows.append(m)
    return rows


def load_bb(p):
    rows = []
    for l in open(p):
        try: d = json.loads(l)
        except ValueError: continue
        if "rp_deg" in d and "foot_n" in d: rows.append(d)
    return rows


def nearest(rows, t):
    return min(rows, key=lambda r: abs(r["timestamp"] - t)) if rows else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pads"); ap.add_argument("bb", nargs="?")
    ap.add_argument("--from", dest="t0", default=""); ap.add_argument("--to", dest="t1", default=""); ap.add_argument("--every", type=float, default=1.0)
    a = ap.parse_args()
    pads = load_pads(a.pads); bb = load_bb(a.bb) if a.bb else []
    print(f"패드 {len(pads)}행 {hms(pads[0]['timestamp'])}~{hms(pads[-1]['timestamp'])}, 블랙박스 {len(bb)}행")
    print("시각      전체CoP앞뒤  왼CoP  오른CoP  발력N(패드합)  |  블랙박스: 발력% 피치° 롤° 접촉")
    last = 0
    for r in pads:
        t = r["timestamp"]; s = hms(t)
        if a.t0 and s < a.t0: continue
        if a.t1 and s > a.t1: continue
        if t - last < a.every: continue
        last = t
        L = [float(v) for v in r["left"]]; R = [float(v) for v in r["right"]]
        ac = cop(L + R); lc = cop(L); rc = cop(R); tot = sum(max(0, v) for v in L + R)
        b = nearest(bb, t) if bb else None
        fmt = lambda v: " -- " if v is None else f"{v:+5.1f}"
        bs = ""
        if b and abs(b["timestamp"] - t) < 0.3:
            fn = b["foot_n"]; rp = b["rp_deg"] or [None, None]
            bs = f"| {100*(fn[0]+fn[1])/172.9:5.1f}% 피치{rp[1] if rp[1] is None else round(rp[1],1)} 롤{rp[0] if rp[0] is None else round(rp[0],1)} {b['contact']}"
        print(f"{s}  {fmt(ac)}      {fmt(lc)} {fmt(rc)}   {int(tot):5d}   {bs}")
    # 요약: 최대 지속 발력에서 CoP
    if pads:
        hi = max(pads, key=lambda r: sum(max(0, float(v)) for v in r["left"] + r["right"]))
        L = [float(v) for v in hi["left"]]; R = [float(v) for v in hi["right"]]
        print(f"\n최대 하중 순간 {hms(hi['timestamp'])}: 패드합 {int(sum(max(0,v) for v in L+R))}, 전체 CoP 앞뒤 {cop(L+R):+.1f}mm (지지면 뒤 −50.9 ~ 앞 +70.6)")
        print("판정 가이드: CoP 가 +70 부근(앞끝)이면 앞으로 쏠림/뒤꿈치 들림, −51 부근(뒤끝)이면 뒤로.")


if __name__ == "__main__":
    main()
