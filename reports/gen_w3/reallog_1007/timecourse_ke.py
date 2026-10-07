import sys, pickle, os
import numpy as np
sys.path.insert(0, ".")
import c5lib as L
fn, t0, t1, step = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
D = pickle.load(open(os.path.join(L.S, fn + ".pkl"), "rb")); R = L.replay(D, "ke"); t = D["t"]; k = int(round(step / 0.02))
i0 = int(np.searchsorted(t, t0)); i1 = int(np.searchsorted(t, t1))
print(f"# {fn} [KE 열린 고리 재생]  t  재생모드 | Δ롤 Δ피치(필터 자세−기준) | 기울기편차 최소–최대 | 원 |ω_xy| 최대 · 감지 |ω| 최대 | 접촉 L R | calm · hold | pk_t pk_w | ref")
for i in range(i0, i1, k):
    s = slice(i, min(i + k, i1)); d = R["rpf"][s] - R["ref"][s]
    print(f"{t[i]:7.2f}  {int(R['mode'][s].max())}{'*' if R['mode'][s].min()!=R['mode'][s].max() else ' '} | {d[:,0].mean():+6.2f} {d[:,1].mean():+6.2f} | {R['tilt'][s].min():5.2f}–{R['tilt'][s].max():5.2f} | {D['wxy'][s].max():5.2f} {R['w'][s].max():5.2f} | "
          f"{D['c'][s][:,0].mean():.1f} {D['c'][s][:,1].mean():.1f} | {R['calm'][s].max():4.2f} {R['hold'][s].max():4.2f} | {R['pk_t'][s].max():4.1f} {R['pk_w'][s].max():4.2f} | {R['ref'][s][-1][0]:+.2f} {R['ref'][s][-1][1]:+.2f}")
