"""접촉 npz 채터 제거 케이던스/에어타임 — R1·R2 동일 산식 공정 비교용.

규칙: 공중 구간이 MIN_AIR(기본 0.1s=5프레임@50Hz) 미만이면 채터로 보고 스탠스에
병합. 병합 후 첫접촉 수 → 케이던스, 병합 후 공중 구간 평균 → 에어타임.
"""
import sys

import numpy as np

DT = 0.02


def dedup_foot(c, min_air_frames):
    """c: (T,) 0/1 접촉. 반환: (첫접촉 수, [에어타임들])"""
    c = c.astype(bool).copy()
    # 짧은 공중 구간 병합
    t = 0
    T = len(c)
    while t < T:
        if not c[t]:
            s = t
            while t < T and not c[t]:
                t += 1
            # [s, t) 공중 — 경계가 접촉으로 둘러싸이고 짧으면 채터
            if s > 0 and t < T and (t - s) < min_air_frames:
                c[s:t] = True
        else:
            t += 1
    # 첫접촉(공중→접촉 에지) + 직전 공중 길이
    n = 0
    airs = []
    t = 1
    while t < T:
        if c[t] and not c[t - 1]:
            n += 1
            s = t - 1
            while s >= 0 and not c[s]:
                s -= 1
            airs.append((t - 1 - s) * DT)
        t += 1
    return n, airs


def analyze(path, min_air_s=0.1):
    d = np.load(path, allow_pickle=True)
    C = d["contact"]
    if C.ndim == 2:  # (T, 2) — env0 단독 (평지 트레이스)
        C = C[:, None, :]
    T, N, _ = C.shape
    dur = T * DT
    mf = int(round(min_air_s / DT))
    nL = nR = 0
    airsL, airsR = [], []
    for e in range(N):
        n, a = dedup_foot(C[:, e, 0], mf)
        nL += n
        airsL += a
        n, a = dedup_foot(C[:, e, 1], mf)
        nR += n
        airsR += a
    cad = (nL + nR) / N / dur
    mAL = float(np.mean(airsL)) if airsL else 0.0
    mAR = float(np.mean(airsR)) if airsR else 0.0
    ratio = max(mAL, 1e-9) / max(mAR, 1e-9)
    sym = max(ratio, 1.0 / ratio)
    name = path.split("/")[-1]
    print(f"{name:32s} 케이던스 {cad:.2f} | 에어 좌 {mAL:.3f} (n={nL}) / "
          f"우 {mAR:.3f} (n={nR}) — 비 {ratio:.2f} (대칭비 {sym:.2f})")
    return cad, mAL, mAR


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
