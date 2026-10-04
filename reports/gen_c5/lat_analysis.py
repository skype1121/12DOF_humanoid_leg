"""C5 앞 밀기 낙상 좌우 분해 (1004): 낙상 방향(직전 틱 피치/롤), 발 횡간격(시작·양발 지지 최소), 최대 롤, 최대 골반 횡속도, 한발 지지 시간."""
import csv, collections, statistics as st, sys
def f(x):
    try: return float(x)
    except: return None
for tag in sys.argv[1:]:
    for c in ("sf12", "sf15"):
        R = [r for sd in (1, 2, 3) for r in csv.DictReader(open(f"/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_{tag}_{c}_s{sd}_envs.csv")) if r["pre_pass"] == "1"]
        F = [r for r in R if r["fell"] == "1"]; A = [r for r in R if r["fell"] != "1"]
        cnt = collections.Counter()
        for r in F:
            p, rl = f(r["fall_pitch"]), f(r["fall_roll"])
            cnt["앞" if p > 0 and abs(p) >= abs(rl) else "뒤" if p < 0 and abs(p) >= abs(rl) else "좌우"] += 1
        def m(G, k):
            v = [f(r[k]) for r in G if f(r[k]) is not None]; return f"{st.median(v):.2f}" if v else "—"
        print(f"{tag} {c}: n {len(R)} 낙상 {len(F)} {dict(cnt)}")
        for nm, G in (("낙상", F), ("생존", A)):
            print(f"   {nm}: 시작 횡간격 {m(G,'sep0_cm')} cm · 양발지지 최소 횡간격 {m(G,'sep_min_ds_cm')} cm · 최대 롤 {m(G,'roll_max_post')}° · 최대 횡속도 {m(G,'vlat_max')} m/s · 한발 지지 {m(G,'single_support_s')} s · 이륙 {m(G,'takeoffs')}")
