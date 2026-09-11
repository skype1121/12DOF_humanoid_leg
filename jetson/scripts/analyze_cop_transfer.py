#!/usr/bin/env python3
"""전이 중 접촉 분포·앞뒤 접촉중심 진단 (2026-09-11 정정판).

⚠ 진단 전용. 압력패드는 접지 판단용이며 정밀 하중계가 아니다 (CLAUDE.md 규칙 6).
  - 여기서 나오는 '앞뒤 접촉중심(mm)'·'패드합'·'발력%'은 전부 **미보정 상대 지표**다.
    체중 지지율·정밀 CoP·좌우 실제 하중·전신 CoM 으로 해석하지 마라.
  - 뒤쪽 패드(각 발 index 3)는 사망(항상 0)이라 뒤쪽이 과소대표되어 접촉중심이
    구조적으로 앞으로 치우친다. 절대값의 앞/뒤 부호만으로 낙상 방향을 단정하지 마라.
  - 이 출력으로 실패 원인·자립 성공을 확정하지 마라. 방향·성공은 IMU 기울기/각속도,
    관절 상태, 운영자·영상 기록으로 판정한다.

사용: python3 analyze_cop_transfer.py <pads.jsonl> [<rl_bridge.jsonl>] \
        [--from HH:MM:SS --to HH:MM:SS] [--every SEC] [--selftest]

패드 앞뒤 좌표(0908 --identify): 앞 +70.6 / 뒤 −50.9mm (발목축 기준).
패드 순서(각 발 4개, 앞뒤 패턴은 양발 공통 [앞,뒤,앞,뒤]):
  좌: [0]앞바깥 [1]뒤바깥 [2]앞안쪽 [3]뒤안쪽(사망)
  우: [0]앞안쪽 [1]뒤안쪽 [2]앞바깥 [3]뒤바깥(사망)
"""
import json, sys, time, argparse
F, B = 70.6, -50.9
XF4 = [F, B, F, B]   # 한 발 4패드 앞뒤 좌표


def cop(vals):
    """앞뒤 접촉중심(mm, 진단용). vals 길이에 맞춰 좌표 타일링 — 4(한 발)/8(양발).

    ★정정(2026-09-11): 종전엔 좌표가 [F,B,F,B] 4개뿐이라 cop(L+R)의 8개 값을
    zip 이 4개로 잘라 **오른발을 통째로 무시**했다(전체 CoP == 왼발 CoP). 8값에는
    양발 8좌표를 쓴다. 앞뒤 패턴이 양발 공통이라 XF4 타일링으로 정확히 복원된다.
    """
    n = len(vals)
    if n % 4 != 0:
        raise ValueError(f"cop: 값 개수는 4의 배수여야 함(패드 4/발), 받은 값 {n}")
    coords = XF4 * (n // 4)
    tot = sum(max(0.0, v) for v in vals)
    return (sum(max(0.0, v) * x for v, x in zip(vals, coords)) / tot) if tot > 0 else None


def contact_split(L, R):
    """좌우 접촉 분포 진단 — **하중비 아님**. 눌린 패드 수와 원값합의 좌우 상대치.

    미보정이므로 '왼발이 N% 더 실렸다' 로 해석 금지. 좌우 중 어느 쪽에 접촉이
    더 분포하는지의 정성 지표로만 쓴다(측면 접촉 이탈 진단 보조)."""
    ln = sum(1 for v in L if v > 0); rn = sum(1 for v in R if v > 0)
    ls = sum(max(0.0, v) for v in L); rs = sum(max(0.0, v) for v in R)
    return ln, rn, ls, rs


def hms(t): return time.strftime("%H:%M:%S", time.localtime(t))


def load_pads(p):
    """압력 원값 로그(pads_inc/pads_cop/pads.jsonl). 각 행 dict 에 left[4]·right[4]·timestamp."""
    rows = []
    for l in open(p):
        try: d = json.loads(l)
        except ValueError: continue
        m = d.get("msg", d)
        if isinstance(m, str):
            try: m = json.loads(m)
            except ValueError: continue
        if isinstance(m, dict) and "left" in m and "right" in m and "timestamp" in m:
            rows.append(m)
    return rows


def load_bb(p):
    rows = []
    for l in open(p):
        try: d = json.loads(l)
        except ValueError: continue
        if "rp_deg" in d and "foot_n" in d and d.get("live"):
            rows.append(d)
    return rows


def nearest(rows, t):
    return min(rows, key=lambda r: abs(r["timestamp"] - t)) if rows else None


def selftest():
    ok = True
    def chk(cond, msg):
        nonlocal ok
        print(("  PASS " if cond else "  FAIL ") + msg); ok = ok and cond
    # 한 발: 앞 패드만 → 앞 좌표
    chk(abs(cop([100, 0, 0, 0]) - F) < 1e-9, "한 발 앞패드만 → +F")
    chk(abs(cop([0, 100, 0, 0]) - B) < 1e-9, "한 발 뒤패드만 → -B")
    # ★회귀: 8값 CoP 가 오른발을 반영하는가 (종전 버그: 오른발 무시)
    left_only = [100, 0, 0, 0]; right_heavy = [0, 0, 0, 0, 0, 100, 0, 0]  # 오른발 뒤
    c8 = cop(left_only + [0, 100, 0, 0])   # 왼앞100 + 오른뒤100 → 중앙 (F+B)/2
    chk(abs(c8 - (F + B) / 2) < 1e-9, "8값: 왼앞+오른뒤 → 앞뒤 평균 (오른발 반영)")
    # 오른발 뒤패드(index5)를 실어 앞뒤가 갈리게 — 종전 truncation 버그면 오른발 무시로 둘이 같아진다
    chk(cop([100, 0, 0, 0]) != cop([100, 0, 0, 0, 0, 100, 0, 0]),
        "8값 CoP != 왼발-only CoP (truncation 버그 부재)")
    chk(cop(right_heavy) is not None and abs(cop(right_heavy) - B) < 1e-9,
        "8값: 오른발 뒤패드만 → -B (왼발 0 무시 아님)")
    # 4의 배수 아님 → 예외
    try: cop([1, 2, 3]); chk(False, "3값 예외")
    except ValueError: chk(True, "4의 배수 아닌 입력 → ValueError")
    ln, rn, ls, rs = contact_split([10, 0, 5, 0], [0, 0, 0, 0])
    chk(ln == 2 and rn == 0 and ls == 15 and rs == 0, "contact_split 카운트/합")
    print("[selftest] " + ("ALL PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pads", nargs="?"); ap.add_argument("bb", nargs="?")
    ap.add_argument("--from", dest="t0", default=""); ap.add_argument("--to", dest="t1", default="")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.pads:
        ap.error("pads.jsonl 필요 (또는 --selftest)")
    pads = load_pads(a.pads); bb = load_bb(a.bb) if a.bb else []
    if not pads:
        print("패드 행 0 — 형식 확인"); return
    print("⚠ 미보정 진단 — 아래 mm/합/%%는 하중·정밀CoP 아님. 방향·성공 판정 금지 (CLAUDE.md).")
    print(f"패드 {len(pads)}행 {hms(pads[0]['timestamp'])}~{hms(pads[-1]['timestamp'])}, 블랙박스 {len(bb)}행")
    print("시각      앞뒤접촉중심  왼   오른  패드합  좌우패드수·합(하중아님) | BB: (미보정)fn합 피치° 롤° 접촉")
    last = 0
    for r in pads:
        t = r["timestamp"]; s = hms(t)
        if a.t0 and s < a.t0: continue
        if a.t1 and s > a.t1: continue
        if t - last < a.every: continue
        last = t
        L = [float(v) for v in r["left"]]; R = [float(v) for v in r["right"]]
        ac = cop(L + R); lc = cop(L); rc = cop(R); tot = sum(max(0, v) for v in L + R)
        ln, rn, ls, rs = contact_split(L, R)
        b = nearest(bb, t) if bb else None
        fmt = lambda v: " --  " if v is None else f"{v:+6.1f}"
        bs = ""
        if b and abs(b["timestamp"] - t) < 0.3:
            fn = b.get("foot_n") or [0, 0]; rp = b.get("rp_deg") or [None, None]
            p_ = "--" if rp[1] is None else round(rp[1], 1)
            r_ = "--" if rp[0] is None else round(rp[0], 1)
            bs = f"| fn합{fn[0] + fn[1]:6.0f} 피치{p_} 롤{r_} {b.get('contact')}"
        print(f"{s} {fmt(ac)}  {fmt(lc)} {fmt(rc)} {int(tot):5d}  L{ln}/{int(ls):4d} R{rn}/{int(rs):4d}  {bs}")
    # 요약 — 미보정, 진단만
    hi = max(pads, key=lambda r: sum(max(0, float(v)) for v in r["left"] + r["right"]))
    L = [float(v) for v in hi["left"]]; R = [float(v) for v in hi["right"]]
    ln, rn, ls, rs = contact_split(L, R)
    print(f"\n[진단·미보정] 최대 패드합 순간 {hms(hi['timestamp'])}: 패드합 {int(sum(max(0,v) for v in L+R))}, "
          f"앞뒤 접촉중심 {cop(L+R):+.1f}mm, 좌우 접촉분포 L{ln}패드/합{int(ls)} vs R{rn}패드/합{int(rs)}")
    print("주의: 접촉중심 mm 는 뒤패드 사망으로 앞으로 편향된 미보정 지표다. 절대 부호로 낙상 방향을 단정하지 말고,")
    print("      IMU 기울기/각속도·관절·영상과 함께 볼 것. 좌우 접촉분포는 하중비가 아니라 접촉 존재의 상대 분포다.")


if __name__ == "__main__":
    main()
