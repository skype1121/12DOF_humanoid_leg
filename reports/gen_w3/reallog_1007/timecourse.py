"""활성화 전후 시간 경과(0.2 s 구간 요약): 기준 대비 [Δ롤, Δ피치], 기울기 편차, |ω_xy|, 접촉 비트, calm 타이머, foot_raw_sum(진단 원값)."""
import sys, pickle, os
import numpy as np
sys.path.insert(0, ".")
import c5lib as L
fn, t0, t1, step = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
D = pickle.load(open(os.path.join(L.S, fn + ".pkl"), "rb")); R = L.replay(D, "rv3")
t = D["t"]; k = int(round(step / 0.02))
i0 = int(np.searchsorted(t, t0)); i1 = int(np.searchsorted(t, t1))
print(f"# {fn}  t[s]  mode | Δ롤 Δ피치(자세−기준)° | 기울기편차° 구간(최소–최대) | |ω_xy| 구간최대 | 접촉 L R(구간 평균) | calm[s] | raw_sum L R(중앙) | ref[롤,피치]")
for i in range(i0, i1, k):
    s = slice(i, min(i + k, i1))
    d = D["rp"][s] - R["ref"][s]
    print(f"{t[i]:7.2f}  {int(D['mode'][s].max())}{'*' if D['mode'][s].min()!=D['mode'][s].max() else ' '} | {d[:,0].mean():+6.2f} {d[:,1].mean():+6.2f} | {R['tilt'][s].min():5.2f}–{R['tilt'][s].max():5.2f} | {D['wxy'][s].max():5.2f} | "
          f"{D['c'][s][:,0].mean():.1f} {D['c'][s][:,1].mean():.1f} | {R['calm'][s].max():4.2f} | {np.nanmedian(D['raw_sum'][s][:,0]):6.0f} {np.nanmedian(D['raw_sum'][s][:,1]):6.0f} | {R['ref'][s][-1][0]:+.2f} {R['ref'][s][-1][1]:+.2f}")
