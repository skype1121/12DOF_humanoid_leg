"""1007 대비책 C(실효 지연) — 실물 블랙박스에 감시기와 같은 지연 추정(w3_contingency.lag_xcorr)을 돌려 분포를 낸다(읽기 전용, 계약 검증 아님·측정).

입력: reports/real_100[1-4]/rl_bridge_*.jsonl (0930 stage8 적체 수정 뒤 세션). 틱 기록의 targets_deg_joint(LPF·보정 뒤 관절 프레임 송출 목표,
블렌딩 전 — 블렌딩 끝난 뒤 = 실제 송출) 와 qpos_rad(엔코더 → 관절 프레임). 블렌딩 중(blend_alpha < 1)·비활성 틱은 창에서 뺀다.
창 2 s(100 틱)·10 틱 간격·관절 = 힙 피치·무릎 4 개(LAT_JOINTS)·들뜸 문턱·상관 문턱은 감시기 기본값 그대로.
사용: python3 reports/gen_w3/cont_lat_real.py [jsonl ...]  → 표(파일별 추정 분포)·JSON(log_picture/cont_lat_real.json)
"""
import glob
import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from rl_walking.deploy import w3_contingency as C  # noqa: E402
from rl_walking.deploy.policy_runner import JOINT_ORDER  # noqa: E402

W = 100


def load(path):
    T, Q, TS = [], [], []
    with open(path) as f:
        for ln in f:
            if '"targets_deg_joint"' not in ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            tj, qp = d.get("targets_deg_joint"), d.get("qpos_rad")
            if not tj or not qp or (d.get("blend_alpha") is not None and d["blend_alpha"] < 1.0):
                T.append(None); Q.append(None); TS.append(d.get("timestamp")); continue
            T.append([math.radians(tj[j]) for j in C.LAT_JOINTS])
            Q.append([qp[JOINT_ORDER.index(j)] for j in C.LAT_JOINTS])
            TS.append(d.get("timestamp"))
    return T, Q, TS


def run(path, cfg=C.W3MonitorConfig()):
    T, Q, TS = load(path)
    ests, corrs, n_win, n_none = [], [], 0, 0
    i = 0
    while i + W <= len(T):
        seg_t, seg_q = T[i:i + W], Q[i:i + W]
        ts = TS[i:i + W]
        ok = all(x is not None for x in seg_t) and all(ts[k] is not None for k in (0, -1)) and (ts[-1] - ts[0]) < W * 0.02 * 1.3
        if ok:
            n_win += 1
            L, c, _ = C.lag_xcorr(np.array(seg_t), np.array(seg_q), cfg.lat_maxlag, cfg.lat_min_std, cfg.lat_min_corr, cfg.lat_min_joints)
            if L is None:
                n_none += 1
            else:
                ests.append(L); corrs.append(c)
        i += cfg.lat_every
    return {"file": os.path.relpath(path, REPO), "ticks": len(T), "windows": n_win, "no_est": n_none,
            "est_n": len(ests), "est_p10_50_90": [float(x) for x in np.percentile(ests, [10, 50, 90])] if ests else None,
            "est_hist": {int(k): int(v) for k, v in zip(*np.unique(ests, return_counts=True))} if ests else {},
            "corr_p50": float(np.median(corrs)) if corrs else None}


def main():
    ps = sys.argv[1:] or sorted(glob.glob(os.path.join(REPO, "reports", "real_100[1-4]", "rl_bridge_*.jsonl")))
    out = [run(p) for p in ps if os.path.getsize(p) > 0]
    print("| 파일 | 틱 | 창 | 추정 없음(들뜸·상관 부족) | 추정 p10/50/90 [틱] | 분포 | 상관 중앙 |")
    print("|---|---|---|---|---|---|---|")
    for r in out:
        print(f"| {r['file']} | {r['ticks']} | {r['windows']} | {r['no_est']} | {r['est_p10_50_90']} | {r['est_hist']} | {r['corr_p50']} |")
    allh = {}
    for r in out:
        for k, v in r["est_hist"].items():
            allh[k] = allh.get(k, 0) + v
    print("합계 분포:", dict(sorted(allh.items())))
    json.dump({"files": out, "total_hist": allh}, open(os.path.join(REPO, "log_picture", "cont_lat_real.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
