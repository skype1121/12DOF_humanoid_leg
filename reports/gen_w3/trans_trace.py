"""전환 틱 추적(log_picture/trans_<tag>.npz) 프로파일: 전환 k = −5..+30 틱의 |Δ목표|(관절 최대, °/틱)·|Δaction|·|피치|·|롤|·|ω| env 중앙/p95 — 팔별 비교.
사용: python3 reports/gen_w3/trans_trace.py tr_restart_nd_d2_s1 tr_reset_nd_d2_s1 ...   (태그 여러 개 = 같은 표에 팔별 행)
"""
import os, sys
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def prof(tag):
    z = np.load(os.path.join(REPO, "log_picture", f"trans_{tag}.npz"), allow_pickle=True)
    k0 = int(z["k0"]); pp = z["pre_pass"]; ok = pp & ~np.isnan(z["tr_sw"])
    cmd = z["cmd"][ok]; a = z["a"][ok]; pitch = z["pitch"][ok]; roll = z["roll"][ok]; w = z["w"][ok]; alive = z["alive"][ok]
    dc = np.degrees(np.abs(np.diff(cmd, axis=1)).max(-1))       # (n, T-1): k 번째 = cmd[k+1]-cmd[k]
    da = np.abs(np.diff(a, axis=1)).max(-1)
    out = {}
    for name, arr, off in (("dcmd", dc, 1), ("da", da, 1), ("pitch", np.abs(pitch), 0), ("roll", np.abs(roll), 0), ("w", w, 0)):
        rows = []
        for k in range(-5, 31):
            j = k0 + k - off + (0 if off == 0 else 0)
            col = arr[:, k0 + k - (1 if off else 0)] if off else arr[:, k0 + k]
            m = alive[:, k0 + k] > 0.5
            v = col[m & ~np.isnan(col)]
            rows.append((k, float(np.median(v)) if v.size else float("nan"), float(np.percentile(v, 95)) if v.size else float("nan")))
        out[name] = rows
    return int(ok.sum()), out


def main():
    tags = sys.argv[1:]
    res = {t: prof(t) for t in tags}
    for name, unit in (("dcmd", "°/틱"), ("da", "action"), ("pitch", "°"), ("roll", "°"), ("w", "rad/s")):
        print(f"\n## {name} [{unit}] — env 중앙 (p95), k = 전환 뒤 틱(0 = KE/W3 첫 틱)")
        ks = [-5, -2, -1, 0, 1, 2, 3, 5, 8, 10, 15, 20, 30]
        print("| 팔(n) | " + " | ".join(f"k{k:+d}" for k in ks) + " |")
        print("|---|" + "---|" * len(ks))
        for t in tags:
            n, o = res[t]; d = {k: (m, p) for k, m, p in o[name]}
            print(f"| {t}({n}) | " + " | ".join(f"{d[k][0]:.2f}({d[k][1]:.2f})" for k in ks) + " |")


if __name__ == "__main__":
    main()
