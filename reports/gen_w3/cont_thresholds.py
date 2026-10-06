"""1007 대비책 — 사전 등록 규칙(CONTINGENCY §P)대로 감시 문턱을 시뮬 추적에서 계산해 --monitor JSON 으로 낸다(판단 없이 규칙만).

규칙: walk_tilt_deg = E2 끔 팔(지연 2 의 4 명령) 생존 env 저역 편차 최대 × 1.3 (0.5° 올림) · walk_w = 저역 RMS 최대 × 1.3 (0.1 올림)
      lat_max_ticks = E1 지연 6 틱 추적의 교차상관 추정 중앙값. 상한(20°·4 rad/s) 초과면 그 항목 0(끔).
사용: python3 reports/gen_w3/cont_thresholds.py  → stdout = JSON 한 줄(+ log_picture/cont_thresholds.json 에 근거 포함)
"""
import json
import math
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "reports", "gen_w3"))
import cont_trace_ana as A  # noqa: E402

OFF = ["cE2_off_fwd_d2", "cE2_off_zero_d2", "cE2_off_lat_d2", "cE2_off_turn_d2"]


def main():
    rs = [A.ana(t) for t in OFF]
    dev_max = max(r["ok_dev_lp_p50_99_999_max"][3] for r in rs)
    w_max = max(r["ok_wrms_p50_99_999_max"][3] for r in rs)
    tilt = math.ceil(dev_max * 1.3 * 2) / 2
    wthr = math.ceil(w_max * 1.3 * 10) / 10
    d6 = A.ana("cE1_d6")
    lat = d6["lag_sent_p10_50_90"][1] if d6["lag_sent_p10_50_90"] else 0.0
    cfg = {"walk_tilt_deg": tilt if tilt <= 20 else 0.0, "walk_w": wthr if wthr <= 4 else 0.0, "lat_max_ticks": float(lat)}
    json.dump({"cfg": cfg, "basis": {"dev_lp_max": dev_max, "wrms_max": w_max, "d6_lag": d6["lag_sent_p10_50_90"], "off": rs, "d6": d6}},
              open(os.path.join(REPO, "log_picture", "cont_thresholds.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
