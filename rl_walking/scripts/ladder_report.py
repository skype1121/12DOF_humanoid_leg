"""검증 사다리 집계 — 조합별 json 을 모아 표와 승격 판정을 낸다.

세대 승격 조건 두 가지를 여기서 강제한다:
  ① 랜덤화 **ON** 에서 전 단계 게이트 통과 (OFF 성적으로 승격 금지)
  ② 이전 세대 대비 **부드러움·에너지 회귀 금지** (성공률이 좋아도 탈락)
"""
import json
import os
import sys

REPO = "/home/ryu/humanoid_leg_test1"
GATE_NAME = {0: "강체 홀드 낙하 (모델정합)", 1: "강체 홀드 서기 (자립)",
             2: "정책 서기", 3: "정책 서기 + 외란", 4: "보행"}


def load(tag):
    p = os.path.join(REPO, "demo_output", f"ladder_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else []


def main(tags, base_tag=None):
    base = {(r["stage"], r["dr"]): r for r in load(base_tag)} if base_tag else {}
    out = []

    def P(s=""):
        print(s)
        out.append(s)

    for tag in tags:
        rows = load(tag)
        if not rows:
            P(f"\n[{tag}] 결과 없음")
            continue
        P("")
        P("=" * 96)
        P(f"검증 사다리 — {tag}" + (f"   (회귀 기준: {base_tag})" if base_tag else ""))
        P("=" * 96)
        P(f"{'단계':<22}{'DR':<5}{'낙상':>9}{'추세°/s':>10}{'E_smth':>10}"
          f"{'에너지':>10}{'명령진폭':>10}{'포화%':>8}{'판정':>10}")
        P("-" * 96)
        for r in rows:
            ok, reg = r.get("pass_gate", False), ""
            b = base.get((r["stage"], r["dr"]))
            if b:
                if r["e_smth"] > b["e_smth"] * 1.15:
                    ok, reg = False, " 부드러움회귀"
                elif r["energy"] > b["energy"] * 1.15:
                    ok, reg = False, " 에너지회귀"
            r["pass_final"] = ok
            P(f"{GATE_NAME[r['stage']]:<22}{'ON' if r['dr'] else 'OFF':<5}"
              f"{r['fall']:>4}/{r['n']:<4}{r['trend_dps']:>10.2f}{r['e_smth']:>10.4f}"
              f"{r['energy']:>10.2f}{r['cmd_amp_deg']:>10.2f}"
              f"{r['sat_ratio']*100:>7.1f}%{('통과' if ok else '탈락'):>10}{reg}")
        P("-" * 96)
        on = [r for r in rows if r["dr"]]
        off = [r for r in rows if not r["dr"]]
        if on and all(r["pass_final"] for r in on):
            P("■ 승격 가능 — 랜덤화 ON 전 단계 통과")
        else:
            bad = [GATE_NAME[r["stage"]] for r in on if not r["pass_final"]]
            P(f"■ **승격 불가** — ON 탈락: {', '.join(bad) if bad else '표본없음'}")
        gap = [GATE_NAME[r["stage"]] for r in off
               if r["pass_final"] and not next(
                   (o["pass_final"] for o in on if o["stage"] == r["stage"]), True)]
        if gap:
            P(f"  ⚠ OFF 에서만 통과한 단계: {', '.join(gap)} — **이 성적으로 승격 금지**")
        low = [r["stage"] for r in on if not r["pass_final"]]
        if low:
            P(f"  ↳ 가장 낮은 탈락 단계 = {min(low)}단계. 그 위 단계의 실패는 "
              "이 단계 탓일 수 있으므로 **먼저 여기를 고친다.**")
    p = os.path.join(REPO, "log_picture", "검증사다리_집계.txt")
    with open(p, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\n리포트: {p}")


main(sys.argv[1].split(","), sys.argv[2] if len(sys.argv) > 2 else None)
