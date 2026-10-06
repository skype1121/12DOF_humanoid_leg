"""1007 대비책 보정 — eval_w3 --trace_n 산출(log_picture/w3trace_<tag>.npz)을 감시기 코드 그대로(W3WalkMonitor, note 모드) 재생해
판정 입력의 시뮬 분포와 '켰다면 생겼을' 사건을 잰다(오프라인, 시뮬 측정 — 폐루프 효과는 eval_w3 --monitor 짝 A/B 로 따로).

(1) 실효 지연: 감시기와 같은 lag_xcorr(창 2 s·10 틱 간격·LAT_JOINTS·기본 문턱)를 '송출 모사'(지연 앞 목표 → 러너 슬루 → LPF β0.27) 대
    관절 위치에 — 실물 블랙박스(cont_lat_real.py)와 같은 정의. 참고: '적용 목표'(지연·슬루·LPF 뒤) 대 관절 = 순수 PD 추종 지연.
(2) 걷는 중 감시 신호(감시기 내부값): 저역 기울기 편차·|ω_xy| 저역 RMS 의 분위(생존 env, 시작 3 s 제외), 낙상 env 는 낙상 전 마지막 값들과
    '낙상 몇 초 전에 문턱을 넘었나'(문턱 = 현재 기본값).
(3) note 사건 수(종류별): 정상 걷기 오트립 후보.
사용: python3 reports/gen_w3/cont_trace_ana.py tag1 tag2 ...
"""
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from rl_walking.deploy import w3_contingency as C  # noqa: E402


def ana(tag, t_skip=3.0, cfg=None):
    cfg = cfg or C.W3MonitorConfig(mode="note")
    z = np.load(os.path.join(REPO, "log_picture", f"w3trace_{tag}.npz"), allow_pickle=True)
    dt = float(z["dt"]); names = list(z["joint_names"])
    li = [names.index(j) for j in C.LAT_JOINTS]
    T, NE = z["q"].shape[0], z["q"].shape[1]
    tf = z["t_fall"]
    k0 = int(round(t_skip / dt))
    L_sent, L_app = [], []
    dev_ok, w_ok, ev_ok, ev_fall, lead_tilt, lead_w, pre_dev, pre_w = [], [], {}, {}, [], [], [], []
    for e in range(NE):
        fell = bool(np.isfinite(tf[e]))
        kend = T if not fell else min(T, int(round(tf[e] / dt)) - 1)
        walk = (z["ge"][:, e] > 0.5) | (z["want"][:, e] > 0.5)
        # (1) 지연
        for k in range(k0, kend - 100 - (50 if fell else 0), cfg.lat_every):
            if not walk[k:k + 100].all():
                continue
            a, _, _ = C.lag_xcorr(z["sent"][k:k + 100, e][:, li], z["q"][k:k + 100, e][:, li], cfg.lat_maxlag, cfg.lat_min_std,
                                  cfg.lat_min_corr, cfg.lat_min_joints)
            b, _, _ = C.lag_xcorr(z["app"][k:k + 100, e][:, li], z["q"][k:k + 100, e][:, li], cfg.lat_maxlag, cfg.lat_min_std,
                                  cfg.lat_min_corr, cfg.lat_min_joints)
            if a is not None:
                L_sent.append(a)
            if b is not None:
                L_app.append(b)
        # (2)(3) 감시기 재생
        m = C.W3WalkMonitor(cfg, names)
        devs, ws, t_tilt, t_w = [], [], None, None
        for k in range(kend):
            m.tick(walking=bool(walk[k]), contact2=z["cc"][k, e], gyro=z["gyro"][k, e], grav=z["grav"][k, e], yaw=float(z["yaw"][k, e]))
            if k >= k0 and m.walking and m.last_dev is not None:
                devs.append(m.last_dev); ws.append(m.last_wrms)
                if t_tilt is None and m.last_dev > cfg.walk_tilt_deg:
                    t_tilt = k * dt
                if t_w is None and m.last_wrms is not None and m.last_wrms > cfg.walk_w:
                    t_w = k * dt
        tgt = ev_fall if fell else ev_ok
        for ev in m.events:
            if ev["t"] >= t_skip:
                tgt[ev["kind"]] = tgt.get(ev["kind"], 0) + 1
        if fell:
            if devs:
                pre_dev.append(float(np.max(devs[-25:]))); pre_w.append(float(np.max(ws[-25:])))
            lead_tilt.append(None if t_tilt is None else round(float(tf[e] - t_tilt), 2))
            lead_w.append(None if t_w is None else round(float(tf[e] - t_w), 2))
        else:
            dev_ok += devs; w_ok += ws

    def q(x, ps=(50, 99, 99.9)):
        return [round(float(v), 3) for v in np.percentile(x, ps)] + [round(float(np.max(x)), 3)] if len(x) else None
    hist = lambda L: {int(k): int(v) for k, v in zip(*np.unique(L, return_counts=True))} if L else {}
    return {"tag": tag, "delay": int(z["delay"]), "cmd": z["cmd"].tolist(), "n_env": NE, "falls": int(np.isfinite(tf).sum()),
            "lag_sent_p10_50_90": [float(v) for v in np.percentile(L_sent, [10, 50, 90])] if L_sent else None, "lag_sent_hist": hist(L_sent),
            "lag_app_p50": float(np.median(L_app)) if L_app else None,
            "ok_dev_lp_p50_99_999_max": q(dev_ok), "ok_wrms_p50_99_999_max": q([w for w in w_ok if w is not None]),
            "events_ok_envs": ev_ok, "events_fall_envs": ev_fall,
            "fall_pre_dev_max": pre_dev, "fall_pre_wrms_max": pre_w, "fall_lead_s_tilt": lead_tilt, "fall_lead_s_w": lead_w}


def main():
    res = [ana(t) for t in sys.argv[1:]]
    for r in res:
        print(json.dumps(r, ensure_ascii=False))
    json.dump(res, open(os.path.join(REPO, "log_picture", "cont_trace_ana.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
