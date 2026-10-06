"""C2UNIT 판정(CPU) — c2_landscape.py 의 B1–B6 결과를 모아 사전 기준 5개(reports/gen_w3/C2_DESIGN.md §4)를 판정.
사용: python3 c2_unit_judge.py <태그접두> <seed...>
해석 고정(실행 전 기록, 1006): 기준 2 의 '가중 적용 뒤 사건당 ≥ +0.15' = 착지 1회당 raw × 가중(10) 의 차(보상 지형 표의 '기여' 단위, 초당 값) → raw 차 ≥ 0.015.
  더 엄격한 해석(에피소드 반환 기여 = raw × 10 × dt = 0.2·raw, 차 ≥ 0.15 → raw 차 ≥ 0.75)도 함께 출력한다.
짝 비교: 팔마다 따로 돈 시뮬이지만 같은 seed·같은 env 번호의 첫 외란 표본은 같다(첫 리셋 RNG 동일) → env 번호로 짝, 차의 평균 ± 표준오차."""
import sys, os
import numpy as np
LP = os.environ.get("C2_LP", "/home/ryu/humanoid_leg_test1/log_picture")
pfx = sys.argv[1]; seeds = sys.argv[2:] or ["1"]
ARMS = ("B1", "B2", "B3", "B4", "B5", "B6")


def load(arm):
    parts = []
    for sd in seeds:
        p = f"{LP}/c2unit_{pfx}_{arm}_s{sd}.npz"
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True); parts.append({k: z[k] for k in z.files})
    if not parts:
        return None
    keys = [k for k in parts[0] if parts[0][k].ndim == 1 and parts[0][k].shape[0] > 1]
    return {k: np.concatenate([q[k] for q in parts]) for k in keys}


R = {a: load(a) for a in ARMS}
miss = [a for a in ARMS if R[a] is None]
L = [f"C2UNIT 판정 · {pfx} · seed {' '.join(seeds)}" + (f" · 없는 팔 {miss}" if miss else "")]
ok_all = {}
base = next(R[a] for a in ARMS if R[a] is not None)
cat = base["cat"]; valid = np.ones_like(cat, dtype=bool)
for a in ARMS:
    if R[a] is not None:
        valid &= R[a]["fell_pre"] < 0.5                     # 창 전(스폰 과도) 낙상 env 는 모든 팔에서 제외(짝 유지)
dist = (cat > 0.5) & valid; nd = (cat < 0.5) & valid
L.append(f"env: 외란 {int(dist.sum())}(앞 {int(((cat == 1) & valid).sum())}·뒤 {int(((cat == 2) & valid).sum())}) · 무외란 {int(nd.sum())} · 창 전 낙상 제외 {int((~valid).sum())}")
r1 = lambda X: X["other_w"] + X["cap_w"] + X["settle_w"]
r1b = lambda X: X["other_w"] + X["cap_w"]
r0 = lambda X: X["other_w"] + X["nom_w"]
for a in ARMS:
    X = R[a]
    if X is None:
        continue
    m = dist
    capr = X["cap_raw_rec"][m].sum() / max(X["n_rec_td"][m].sum(), 1); nomr = X["nom_raw_rec"][m].sum() / max(X["n_rec_td"][m].sum(), 1)
    L.append(f"  {a}: 외란 창 낙상 {100*X['fell_w'][m].mean():.1f} % · R0 {r0(X)[m].mean():+.3f} · R1 {r1(X)[m].mean():+.3f} · R1a {(X['other_w']+X['nom_w']+X['settle_w'])[m].mean():+.3f} · R1b {r1b(X)[m].mean():+.3f}"
             f" · 켜짐 {X['trig_w'][m].mean():.2f} · 요청 {X['req_w'][m].mean():.2f}(초과 {X['timeout_w'][m].mean():.2f}) · 래치 {X['latch_w'][m].mean():.2f} · 정착 {X['settle_ev_w'][m].mean():.2f}"
             f" · 회복 착지 {X['n_rec_td'][m].mean():.2f}/env 포획 raw {capr:.3f} 명목 raw {nomr:.3f} · 요청 뒤 이륙 {np.nanmean(X['to_after_req'][m]):.2f} · 표류 중앙 {np.nanmedian(X['drift_w'][m]):.1f} cm")


def pdiff(A, B, f, m):
    d = f(R[A])[m] - f(R[B])[m]
    return d.mean(), d.std(ddof=1) / np.sqrt(max(len(d), 2))


# 1. 불변
c1 = True; det = []
for a in ARMS:
    X = R[a]
    if X is None:
        continue
    e1 = np.abs(X["cap_all"] - X["nom_all"])[nd].max() if nd.any() else 0.0
    e2 = X["settle_all"][nd].max() if nd.any() else 0.0
    e3 = X["trig_all"][nd].max() if nd.any() else 0.0
    det.append(f"{a}: |포획−명목| 최대 {e1:.2e} · 정착 {e2:.2e} · 켜짐 {e3:.0f}")
    c1 &= (e1 <= 1e-6) and (e2 == 0) and (e3 == 0)
if R["B1"] is not None:
    X = R["B1"]; e4 = np.abs(X["cap_all"] - X["nom_all"]).max(); e5 = X["settle_all"].max(); e6 = X["trig_all"].max()
    det.append(f"B1 전 env: |포획−명목| {e4:.2e} · 정착 {e5:.2e} · 켜짐 {e6:.0f}"); c1 &= (e4 <= 1e-6) and (e5 == 0) and (e6 == 0)
ok_all[1] = c1
L.append(f"[기준 1 불변] {'PASS' if c1 else 'FAIL'} — 무외란 env: " + " | ".join(det))
# 2. 포획 배치(생존 env 회복 착지당 포획 raw)
if all(R[a] is not None for a in ("B2", "B3", "B6")):
    def capraw(a):
        X = R[a]; m = dist & (X["fell_w"] < 0.5)
        return X["cap_raw_rec"][m].sum() / max(X["n_rec_td"][m].sum(), 1), X["nom_raw_rec"][m].sum() / max(X["n_rec_td"][m].sum(), 1), int(X["n_rec_td"][m].sum())
    c6, n6, k6 = capraw("B6"); c3, n3, k3 = capraw("B3"); c2_, n2, k2 = capraw("B2")
    order = c6 > c3 >= c2_; d = c6 - c2_
    ok2 = order and (10.0 * d >= 0.15)
    ok_all[2] = ok2
    L.append(f"[기준 2 포획 배치] {'PASS' if ok2 else 'FAIL'} — 회복 착지당 포획 raw B6 {c6:.3f}(n {k6}) · B3 {c3:.3f}(n {k3}) · B2 {c2_:.3f}(n {k2}) · 순서 B6>B3≥B2 {'예' if order else '아니오'}"
             f" · B6−B2 raw {d:+.3f} → ×가중 {10*d:+.3f}(≥0.15 필요) · 엄격 해석 ×가중×dt {0.2*d:+.3f}(≥0.15: {'예' if 0.2*d >= 0.15 else '아니오'})")
    L.append(f"    진단(R0 명목 착지 raw): B6 {n6:.3f} · B3 {n3:.3f} · B2 {n2:.3f} — 기대 B6 ≤ B2(현행 보상이 포획 배치를 벌함): {'예' if n6 <= n2 else '아니오'}")
# 2m·2d (1006 재설계, 실행 전 등록; 문헌 권고 반영 — 기준 2 계열은 **충격 외란 env(kind0 = 2)** 에서만 판정, 램프는 보고만).
#   점수 = 회복 착지당 평균(외란 env 전체 착지 — 생존 선택 편향 회피), 통과 = B6 − B2 > 0 이고 env 부트스트랩(2000) 95 % 하한 > 0.
#   2m = 이동 방향 포획 여유형(mg_raw_rec), 2d = 차분형 Φ 감소(df_raw_rec, 램프 외력은 ξ_eq 로 보정).
imp = dist & (base["kind0"] > 1.5); ramp = dist & (base["kind0"] > 0.5) & (base["kind0"] < 1.5)
rng = np.random.default_rng(0)
def crit2(field, tag, label):
    if not (all(R[a] is not None for a in ("B2", "B6")) and field in R["B2"]):
        return
    def ratio(a, m):
        X = R[a]; return X[field][m].sum() / max(X["n_rec_td"][m].sum(), 1)
    def boot(a, m):
        X = R[a]; idx = np.where(m)[0]; out = np.empty(2000)
        for k in range(2000):
            j = rng.choice(idx, idx.size); out[k] = X[field][j].sum() / max(X["n_rec_td"][j].sum(), 1)
        return out
    res = {}
    for mlab, m in (("충격", imp), ("램프", ramp)):
        if m.sum() < 5:
            continue
        d = boot("B6", m) - boot("B2", m); lo, hi = np.percentile(d, [2.5, 97.5])
        s6, s2 = ratio("B6", m), ratio("B2", m); s3_ = ratio("B3", m) if R["B3"] is not None and field in R["B3"] else float("nan")
        res[mlab] = (s6, s3_, s2, lo, hi, int(m.sum()), int(R["B6"]["n_rec_td"][m].sum()), int(R["B2"]["n_rec_td"][m].sum()))
    if "충격" in res:
        s6, s3_, s2, lo, hi, n, k6, k2 = res["충격"]
        ok = bool((s6 - s2 > 0) and (lo > 0)); ok_all[tag] = ok
        L.append(f"[기준 {tag} {label}, 충격 env {n}] {'PASS' if ok else 'FAIL'} — 회복 착지당 B6 {s6:+.3f} · B3 {s3_:+.3f} · B2(제자리) {s2:+.3f} · B6−B2 {s6-s2:+.3f} (95 % {lo:+.3f}~{hi:+.3f}) · 착지 B6 {k6} B2 {k2}")
    if "램프" in res:
        s6, s3_, s2, lo, hi, n, k6, k2 = res["램프"]
        L.append(f"    (보고) 램프 env {n}: B6 {s6:+.3f} · B3 {s3_:+.3f} · B2 {s2:+.3f} · B6−B2 95 % {lo:+.3f}~{hi:+.3f}")
crit2("cap_raw_rec", "2i", "오차형(종전 점수)")
crit2("mg_raw_rec", "2m", "포획 여유형")
crit2("df_raw_rec", "2d", "차분형 Φ")
# 3. 멈추는 것이 이긴다
if all(R[a] is not None for a in ("B2", "B3", "B5")):
    m = dist
    v3, v5, v2 = r1(R["B3"])[m].mean(), r1(R["B5"])[m].mean(), r1(R["B2"])[m].mean()
    d32, se32 = pdiff("B3", "B2", r1, m); d32b, se32b = pdiff("B3", "B2", r1b, m)
    ok3 = (v3 > v5 > v2) and d32 >= 1.0
    ok_all[3] = ok3
    L.append(f"[기준 3 정지] {'PASS' if ok3 else 'FAIL'} — R1 회복 1건(8 s) 평균 B3 {v3:+.3f} · B5 {v5:+.3f} · B2 {v2:+.3f} · B3−B2 짝 {d32:+.3f} ± {se32:.3f}(≥ +1.0 필요)"
             f" · R1b(정착 없음) B3−B2 {d32b:+.3f} ± {se32b:.3f}")
# 4. 파밍
c4 = True; det4 = []
for a in ARMS:
    X = R[a]
    if X is None:
        continue
    t0 = X["t0"]; nd_in = 1 + ((X["kind1"] > 0) & (X["t01"] < t0 + 8.0)).astype(int)
    over = int((X["settle_ev_w"][dist] > nd_in[dist]).sum())
    det4.append(f"{a} 정착>외란 수 env {over}"); c4 &= over == 0
ok_all[4] = c4
L.append(f"[기준 4 파밍] {'PASS' if c4 else 'FAIL'} — " + " · ".join(det4) + " (무외란 켜짐 0·B1 c2 항 0 은 기준 1 에서)")
# 5. 이른 정지
if all(R[a] is not None for a in ("B3", "B4")):
    d43, se43 = pdiff("B4", "B3", r1, dist)
    ok5 = d43 <= 0.0
    ok_all[5] = ok5
    L.append(f"[기준 5 이른 정지] {'PASS' if ok5 else 'FAIL'} — R1 B4−B3 짝 {d43:+.3f} ± {se43:.3f}(≤ 0 필요) · 외란 창 낙상 B4 {100*R['B4']['fell_w'][dist].mean():.1f} % vs B3 {100*R['B3']['fell_w'][dist].mean():.1f} %")
ok_all = {k: bool(v) for k, v in ok_all.items()}
for _fld, _sfx in (("capm_w", "m"), ("capd_w", "d")):
  if _fld in base:      # 여유형/차분형 보상으로 기준 3·5 재판정(같은 궤적)
      r1m = lambda X, f=_fld: X["other_w"] + X[f] + X["settle_w"]
      if all(R[a] is not None for a in ("B2", "B3", "B4", "B5")):
          v3, v5, v2 = r1m(R["B3"])[dist].mean(), r1m(R["B5"])[dist].mean(), r1m(R["B2"])[dist].mean()
          d32, se32 = pdiff("B3", "B2", r1m, dist); d43, se43 = pdiff("B4", "B3", r1m, dist)
          ok_all["3" + _sfx] = bool((v3 > v5 > v2) and d32 >= 1.0); ok_all["5" + _sfx] = bool(d43 <= 0.0)
          L.append(f"[기준 3·5 ({_fld} R1)] 3 {'PASS' if ok_all['3' + _sfx] else 'FAIL'}: B3 {v3:+.3f} · B5 {v5:+.3f} · B2 {v2:+.3f} · B3−B2 {d32:+.3f} ± {se32:.3f} | "
                   f"5 {'PASS' if ok_all['5' + _sfx] else 'FAIL'}: B4−B3 {d43:+.3f} ± {se43:.3f}")
          L.append(f"{_fld} 설정 학습 시작 조건(1·2{_sfx}·3{_sfx}·4·5{_sfx}): " + ("충족" if all(ok_all.get(k) for k in (1, "2" + _sfx, "3" + _sfx, 4, "5" + _sfx)) else "미충족"))
kill = (2 in ok_all and not ok_all[2]) and (3 in ok_all and not ok_all[3])
L.append(f"종합: 통과 {[k for k, v in ok_all.items() if v]} · 실패 {[k for k, v in ok_all.items() if not v]} → " +
         ("Kill(기준 2·3 모두 실패: c2 보상 설계 폐기)" if kill else ("C2 학습 시작 조건 충족" if all(ok_all.get(k) for k in ((1, "2m", 3, 4, 5) if "2m" in ok_all else (1, 2, 3, 4, 5))) else "학습 시작 조건 미충족(실패 기준 수정 뒤 재시험)")))
txt = "\n".join(L); print(txt)
open(f"{LP}/c2unit_judge_{pfx}.txt", "w").write(txt + "\n")
