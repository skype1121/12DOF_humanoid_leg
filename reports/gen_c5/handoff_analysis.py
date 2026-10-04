"""C5 handoff 분석 (1004): A 전환 순간 액션·관측 튐, B 전환 순간 상태 분포(G21a 가 버틴 상태 vs G24R-a 자체 상태).
사용: python3 handoff_analysis.py <c5 태그> <monitor 태그>   (log_picture/ground_recovery_<태그>_handoff.npz)"""
import sys, json, numpy as np
R = "/home/ryu/humanoid_leg_test1/log_picture/ground_recovery_{}_handoff.npz"
def load(t):
    z = np.load(R.format(t), allow_pickle=True); d = {k: z[k] for k in z.files}
    d["layout"] = json.loads(str(d["obs_layout"])) if str(d["obs_layout"]) else {}
    ok = ~np.isnan(d["t_sw"]) & d["pre_pass"].astype(bool); d["ok"] = ok; return d
c5, mon = load(sys.argv[1]), load(sys.argv[2]); H = (c5["a_exec"].shape[1] - 1) // 2
print(f"표본: C5 전환 env {c5['ok'].sum()} · 모니터 감지 env {mon['ok'].sum()} · 첫 켜짐 시각 중앙 C5 {np.nanmedian(c5['t_sw'][c5['ok']]):.2f} / 모니터 {np.nanmedian(mon['t_sw'][mon['ok']]):.2f} s")
def jump(d, k, i0, i1):
    x = d[k][d["ok"]]; return np.degrees(np.abs(x[:, i1] - x[:, i0]).max(-1)) if x.ndim == 3 else np.abs(x[:, i1] - x[:, i0])
print("\n[A] 액션 튐 (실행 액션, 관절 최대 |Δ| deg, 정책 출력 단위 rad×57.3 — 체인 전)")
for nm, d in (("C5", c5), ("모니터(G24R-a 단독)", mon)):
    pre = jump(d, "a_exec", H - 2, H - 1); sw = jump(d, "a_exec", H - 1, H); post = jump(d, "a_exec", H, H + 1)
    print(f"  {nm}: 전환 전 틱 {np.median(pre):.2f} (p90 {np.percentile(pre,90):.2f}) · 전환 틱 {np.median(sw):.2f} (p90 {np.percentile(sw,90):.2f}) · 다음 틱 {np.median(post):.2f}")
x1 = c5["a1"][c5["ok"], H]; x2 = c5["a2"][c5["ok"], H]
print(f"  C5 전환 틱 두 정책 출력 차 |a2−a1| 관절 최대: 중앙 {np.median(np.degrees(np.abs(x2-x1).max(-1))):.2f}° p90 {np.percentile(np.degrees(np.abs(x2-x1).max(-1)),90):.2f}°")
print("\n[A] 관측 튐 (항별 |Δobs| 평균, 전환 틱 vs 전환 전 틱 — 공유 env 관측이라 정책 교체 자체로는 불연속 없어야 함)")
for n, (a, b) in c5["layout"].items():
    o = c5["obs"][c5["ok"]][:, :, a:b]; pre = np.abs(o[:, H-1] - o[:, H-2]).mean(); sw = np.abs(o[:, H] - o[:, H-1]).mean(); po = np.abs(o[:, H+1] - o[:, H]).mean()
    om = mon["obs"][mon["ok"]][:, :, a:b]; msw = np.abs(om[:, H+1] - om[:, H]).mean()
    print(f"  {n:24s} 전 {pre:.4f} · 전환 {sw:.4f} · 다음 {po:.4f} | 모니터 다음 {msw:.4f}")
print("\n[B] 전환(감지) 순간 상태: C5(G21a 가 버틴 상태) vs 모니터(G24R-a 자체)")
def st(d, m):
    s = {k: d[k][m][:, H] for k in ("tilt", "w", "vxy")}
    s["q_dev"] = np.degrees(np.abs(d["q"][m][:, H]).max(-1)); s["qd_rms"] = np.sqrt((d["qd"][m][:, H] ** 2).mean(-1))
    fz = d["fz"][m][:, H]; s["fz_min/max"] = fz.min(-1) / np.maximum(fz.max(-1), 1e-3); s["Fz합 N"] = fz.sum(-1)
    s["피치레이트"] = d["gyro"][m][:, H, 1]
    return s
for k in ("tilt", "w", "vxy", "q_dev", "qd_rms", "fz_min/max", "Fz합 N", "피치레이트"):
    a = st(c5, c5["ok"])[k]; b = st(mon, mon["ok"])[k]
    print(f"  {k:12s} C5 중앙 {np.median(a):7.3f} [p10 {np.percentile(a,10):.3f}, p90 {np.percentile(a,90):.3f}] · 모니터 {np.median(b):7.3f} [{np.percentile(b,10):.3f}, {np.percentile(b,90):.3f}]")
print("\n[B] C5 안에서 낙상 vs 생존 (감지 순간)")
for nm, m in (("낙상", c5["ok"] & c5["fell"].astype(bool)), ("생존", c5["ok"] & ~c5["fell"].astype(bool))):
    s = st(c5, m); print(f"  {nm} n={m.sum()}: " + " · ".join(f"{k} {np.median(v):.3f}" for k, v in s.items()))
print("\n[B] 관절별 감지 순간 q 편차 평균(deg) C5 − 모니터")
c = np.degrees(c5["q"][c5["ok"]][:, H]).mean(0); mm = np.degrees(mon["q"][mon["ok"]][:, H]).mean(0)
print("  " + " ".join(f"{x:+.1f}" for x in (c - mm)))
