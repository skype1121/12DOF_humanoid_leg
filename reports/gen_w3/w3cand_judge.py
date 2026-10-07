"""W3 후보 2 단계 판정 보조(S3E_DESIGN.md §4 E4·E6) — 단일 걷기 회귀(eval_w3 JSON)와 걷는 중 외란(정책 직접 평가 CSV)을 기준선 W3S3d-15800 과 나란히.
사용: python3 reports/gen_w3/w3cand_judge.py --name s3g --ck 16299
E4: 명령별 낙상 +2 %p 이하 ∧ 속도·회전 추종 이득이 기준선의 0.85 배 이상. E6: 방향·크기별 낙상 +5 %p 이하.
E3(왕복)·E5(전환)는 kw_scen_summ.py / kw_summ.py 를 쓴다(이 스크립트가 명령줄을 출력)."""
import argparse, csv, json, os
import numpy as np
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
E4 = [("c0", "제자리", None), ("f06", "앞 0.06", ("fwd_mps", 0.06)), ("d4f06", "앞 0.06 지연 4", ("fwd_mps", 0.06)), ("l05", "옆 0.05", ("lat_mps", 0.05)),
      ("tl01", "좌회전 0.1", ("yaw_rate_dps", 5.73)), ("tr01", "우회전 0.1", ("yaw_rate_dps", -5.73)), ("tl02", "좌회전 0.2", ("yaw_rate_dps", 11.46)), ("tr02", "우회전 0.2", ("yaw_rate_dps", -11.46))]
E6 = ["nd", "b12", "b18", "f12", "f18"]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def falls(tag):
    p = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
    if not os.path.exists(p):
        return None
    R = [r for r in csv.DictReader(open(p)) if r["pre_pass"] == "1"]
    return sum(r["fell"] == "1" for r in R), len(R), {int(r["env"]): r["fell"] == "1" for r in R}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--ck", required=True); ap.add_argument("--base", default="s3d_15800")
    a = ap.parse_args(); ok4 = True; ok6 = True; miss = []
    print(f"### E4 단일 걷기 회귀 — {a.name}-{a.ck} 대 기준선 {a.base} (eval_w3, 256 env, 30 s)")
    print("| 명령 | 낙상 기준선 → 후보 | 추종 값 기준선 → 후보 (명령) | 이득 비 | 판정 |"); print("|---|---|---|---|---|")
    for key, lab, tr in E4:
        b = jl(os.path.join(LP, f"w3eval_{a.base}_{key}.json")); c = jl(os.path.join(LP, f"w3eval_{a.name}_{a.ck}_{key}.json"))
        if b is None or c is None:
            miss.append(key); print(f"| {lab} | (자료 없음: {'기준선' if b is None else ''} {'후보' if c is None else ''}) | | | |"); continue
        df = 100 * (c["fall_rate"] - b["fall_rate"]); good = df <= 2.0 + 1e-9; gr = ""; tv = ""
        if tr is not None:
            k, cmd = tr; gb = b[k] / cmd; gc = c[k] / cmd; ratio = gc / gb if abs(gb) > 1e-9 else float("nan"); good = good and (ratio >= 0.85)
            tv = f"{b[k]:+.3f} → {c[k]:+.3f} ({cmd:+.2f})"; gr = f"{ratio:.2f}"
        ok4 = ok4 and good
        print(f"| {lab} | {100 * b['fall_rate']:.1f} → {100 * c['fall_rate']:.1f} % ({df:+.1f} %p) | {tv} | {gr} | {'충족' if good else '미달'} |")
    print(f"- **E4**: {'충족' if ok4 and not miss else ('자료 부족 ' + ','.join(miss) if miss else '미달')}")
    print(f"### E6 걷는 중 외란 — 정책 직접 평가(128 env × seed 1–2), 사전 검사 통과 env 낙상")
    print("| 조건 | 기준선 | 후보 | 차 | env 짝(후보만 / 기준선만) | 판정 |"); print("|---|---|---|---|---|---|")
    for cnd in E6:
        fb = fc = nb = nc = 0; on_c = on_b = 0; got = True
        for sd in (1, 2):
            B = falls(f"tr_walk_q6{cnd}_d2_s{sd}"); C = falls(f"tr_{a.name}{a.ck}_walk_q6{cnd}_d2_s{sd}")
            if B is None or C is None:
                got = False; break
            fb += B[0]; nb += B[1]; fc += C[0]; nc += C[1]; common = set(B[2]) & set(C[2]); on_c += sum(C[2][e] and not B[2][e] for e in common); on_b += sum(B[2][e] and not C[2][e] for e in common)
        if not got:
            miss.append("q6" + cnd); print(f"| {cnd} | (자료 없음) | | | | |"); continue
        d = 100 * (fc / max(nc, 1) - fb / max(nb, 1)); good = d <= 5.0 + 1e-9; ok6 = ok6 and good
        print(f"| {cnd} | {fb} / {nb} ({100 * fb / max(nb, 1):.1f} %) | {fc} / {nc} ({100 * fc / max(nc, 1):.1f} %) | {d:+.1f} %p | {on_c} / {on_b} | {'충족' if good else '미달'} |")
    print(f"- **E6**: {'충족' if ok6 and not any(m.startswith('q6') for m in miss) else ('자료 부족' if any(m.startswith('q6') for m in miss) else '미달')}")
    print("### E3·E5 명령줄")
    print(f"python3 reports/gen_w3/kw_scen_summ.py --tags tr_{a.name}{a.ck}_x7_d2_s1,tr_{a.name}{a.ck}_x7_d2_s2 --name '{a.name}-{a.ck} X7'")
    print(f"python3 reports/gen_w3/kw_summ.py --arm tr_{a.name}{a.ck}_kew3q --ref tr_kew3q --conds nd,b18,sf15 --seeds 1,2,3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
