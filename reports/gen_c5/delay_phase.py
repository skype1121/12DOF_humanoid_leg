"""순수 지연 추정(검토 9회차): 명령→실측 전달함수 위상의 3–9 Hz 기울기 → 지연[s]. 실물 블랙박스·시뮬 덤프에 같은 방법.
교차스펙트럼 Welch(창 128 틱, 50 % 겹침), 결맞음 ≥ 0.5 인 bin 만, 위상 unwrap 후 가중 선형회귀."""
import json, sys, glob, numpy as np
JO = ["left_hip_f_joint","right_hip_f_joint","left_hip_a_joint","right_hip_a_joint","left_hip_r_joint","right_hip_r_joint","left_knee_joint","right_knee_joint","left_ankle_f_joint","right_ankle_f_joint","left_ankle_r_joint","right_ankle_r_joint"]
FS = 50.0
def csd(x, y, nper=128):
    step = nper // 2; w = np.hanning(nper); S = []; Pxx = []; Pyy = []
    for s in range(0, len(x) - nper + 1, step):
        X = np.fft.rfft((x[s:s+nper] - x[s:s+nper].mean()) * w); Y = np.fft.rfft((y[s:s+nper] - y[s:s+nper].mean()) * w)
        S.append(np.conj(X) * Y); Pxx.append(np.abs(X) ** 2); Pyy.append(np.abs(Y) ** 2)
    return np.fft.rfftfreq(nper, 1 / FS), np.mean(S, 0), np.mean(Pxx, 0), np.mean(Pyy, 0)
def delay(cmd, q, lo=3.0, hi=9.0):
    f, Sxy, Pxx, Pyy = csd(np.asarray(cmd, float), np.asarray(q, float))
    coh = np.abs(Sxy) ** 2 / (Pxx * Pyy + 1e-12); ph = np.unwrap(np.angle(Sxy))
    m = (f >= lo) & (f <= hi) & (coh >= 0.5)
    if m.sum() < 3: return None
    A = np.vstack([f[m], np.ones(m.sum())]).T; W = coh[m]
    k, b = np.linalg.lstsq(A * W[:, None], ph[m] * W, rcond=None)[0]
    return -k / (2 * np.pi)            # s
def real(path):
    R = [json.loads(l) for l in open(path) if l.strip()]; R = [r for r in R if r.get("live") and r.get("targets_deg_joint") and r.get("qpos_rad")]
    tg = np.array([[r["targets_deg_joint"][j] for j in JO] for r in R]); q = np.degrees(np.array([r["qpos_rad"] for r in R]))
    return [delay(tg[:, k], q[:, k]) for k in range(12)]
def sim(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        R = [json.loads(l) for l in open(f) if l.strip()]; R = [r for r in R if r.get("phase") == "policy" and r.get("cmd_deg")]
        if len(R) < 256: continue
        tg = np.array([r["cmd_deg"] for r in R]); q = np.array([r["q_deg"] for r in R])
        out += [delay(tg[:, k], q[:, k]) for k in range(tg.shape[1])]
    return out
def summ(lab, v):
    v = [x for x in v if x is not None]
    print(f"{lab}: 순수 지연 중앙 {1000*np.median(v):.1f} ms = {50*np.median(v):.2f} 틱 (IQR {1000*np.percentile(v,25):.1f}–{1000*np.percentile(v,75):.1f} ms, n {len(v)})")
for p in sys.argv[1:]:
    if p.endswith(".jsonl") and "rl_bridge" in p: summ(p.split("/")[-1], real(p))
    else: summ(p, sim(p))
