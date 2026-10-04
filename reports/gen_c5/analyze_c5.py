"""C5 실물 진단 블랙박스 분석 (1004) — 전환 매끄러움 판정(낙상률 아님).
사용: python3 reports/gen_c5/analyze_c5.py <rl_bridge_*.jsonl> [--t0 <s> --t1 <s>]
출력:
  - 모드 타임라인(켜짐/꺼짐 시각, 회복 체류 길이)
  - 전환 순간 목표 점프: 전환 틱 원시 액션 |Δ| 최대(관절), 같은 세션 평소 틱 분포 대비
  - 전환 틱 두 정책 출력 차 |raw_rec − raw_stand| 최대
  - 채터: 1 s 안에 켜짐/꺼짐 반복 횟수, 고착: 밀기 없이 회복 > 5 s
  - 3 Hz 재발 감시: 전환 전후 2 s 피치 각속도 2.5–4 Hz 대역 RMS
즉시 중단 기준(REAL_GATE_C5.md)과 같은 정의를 쓴다.
"""
import json
import math
import sys

import numpy as np


def load(path, t0=None, t1=None):
    R = [json.loads(l) for l in open(path) if l.strip()]
    R = [r for r in R if r.get("live") and r.get("armed") and r.get("c5") is not None]
    if not R:
        raise SystemExit("C5 기록 없음(정책이 stand_c5_g21a_g24ra 가 아니거나 정책 구간 없음)")
    T0 = R[0]["timestamp"]
    for r in R:
        r["t"] = r["timestamp"] - T0
    if t0 is not None:
        R = [r for r in R if r["t"] >= t0]
    if t1 is not None:
        R = [r for r in R if r["t"] < t1]
    return R


def band_rms(x, fs=50.0, lo=2.5, hi=4.0):
    x = np.asarray(x, float) - np.mean(x)
    if len(x) < 32:
        return float("nan")
    X = np.fft.rfft(x * np.hanning(len(x)))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    m = (f >= lo) & (f <= hi)
    return float(np.sqrt(np.sum(np.abs(X[m]) ** 2)) / len(x))


def main():
    args = sys.argv[1:]
    path = args[0]
    t0 = float(args[args.index("--t0") + 1]) if "--t0" in args else None
    t1 = float(args[args.index("--t1") + 1]) if "--t1" in args else None
    R = load(path, t0, t1)
    mode = np.array([r["c5"]["mode"] for r in R]); t = np.array([r["t"] for r in R])
    if all(r.get("targets_deg_joint") for r in R):       # 실물: 체인(슬루·LPF·블렌딩) 뒤 실제 명령 목표각 [deg] — 판정 기준
        from_tg = True
        tg = np.array([[r["targets_deg_joint"][j] for j in sorted(r["targets_deg_joint"])] for r in R], float)
        d_raw = np.abs(np.diff(tg, axis=0)).max(1)
    else:                                                 # 합성 기록: 정책 원시 출력(체인 전) ×0.25
        from_tg = False
        raw = np.array([r["raw_action"] for r in R], float)
        d_raw = np.degrees(np.abs(np.diff(raw, axis=0)).max(1) * 0.25)
    print("목표 점프 기준:", "실제 명령 목표각(체인 후)" if from_tg else "정책 원시 출력×0.25(체인 전 — 합성 기록)")
    ons = [i for i in range(1, len(mode)) if mode[i] == 1 and mode[i - 1] == 0]
    offs = [i for i in range(1, len(mode)) if mode[i] == 0 and mode[i - 1] == 1]
    print(f"기록 {len(R)} 틱 · {t[-1]:.1f} s · 회복 체류 {100 * mode.mean():.1f} % · 켜짐 {len(ons)} · 꺼짐 {len(offs)}")
    normal = d_raw[[i - 1 for i in range(1, len(mode)) if mode[i] == mode[i - 1]]]
    print(f"평소 틱 목표 변화(deg): 중앙 {np.median(normal):.2f} · p99 {np.percentile(normal, 99):.2f}")
    chatter = 0; stuck = []
    for k, i in enumerate(ons):
        j = next((o for o in offs if o > i), None)
        dur = (t[j] - t[i]) if j is not None else (t[-1] - t[i])
        jump = d_raw[i - 1] if i - 1 < len(d_raw) else float("nan")
        c5 = R[i]["c5"]
        dd = (np.degrees(np.abs(np.array(c5["raw_rec"]) - np.array(c5["raw_stand"])).max() * 0.25)
              if c5.get("raw_rec") and c5.get("raw_stand") else float("nan"))
        q = [r["rp_deg"][1] for r in R[max(0, i - 100):i + 100] if r.get("rp_deg")]
        print(f"  켜짐 t={t[i]:7.2f} s · 회복 {dur:5.2f} s · 전환 틱 목표 점프 {jump:.2f}° (평소 p99 대비 ×{jump / max(np.percentile(normal, 99), 1e-6):.1f}) · "
              f"두 정책 차 {dd:.2f}° · 전후 2 s 피치 3 Hz RMS {band_rms(q):.4f} · 기준 자세 {c5.get('ref_deg')}")
        if j is not None and k + 1 < len(ons) and t[ons[k + 1]] - t[j] < 1.0:
            chatter += 1
        if dur > 5.0:
            stuck.append(round(float(t[i]), 1))
    for i in offs:
        print(f"  꺼짐 t={t[i]:7.2f} s · 복귀 틱 목표 점프 {d_raw[i - 1]:.2f}°")
    print(f"채터(꺼진 뒤 1 s 안 재켜짐) {chatter} 회 · 5 s 넘는 회복 체류 {stuck} (밀기 기록과 대조 — 밀기 없이면 '고착' 중단 기준)")


if __name__ == "__main__":
    main()
