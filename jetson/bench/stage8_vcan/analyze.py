import json, sys, numpy as np
d = sys.argv[1]
for v in sys.argv[2:]:
    try:
        drv = json.load(open(f"{d}/drv_{v}.json")); can = json.load(open(f"{d}/can_{v}.json"))
    except FileNotFoundError:
        print(v, "없음"); continue
    pub = np.array(drv["pub"]); st = drv["st"]
    t0, t1 = pub[0, 0], pub[-1, 0]
    seq = np.array([(t, s) for t, s, *_ in st if s is not None and t0 + 1 <= t <= t1 - 1], float)
    rate = (seq[-1, 1] - seq[0, 1]) / (seq[-1, 0] - seq[0, 0])
    st_t = np.array([t for t, *_ in st if t0 + 1 <= t <= t1 - 1]); st_rate = len(st_t) / (st_t[-1] - st_t[0])
    c = np.array([(t, p) for t, mid, p in can if mid == 1 and t0 <= t <= t1], float)
    # CAN 명령 위치(모터 1)와 발행 목표의 지연: 발행 시각 격자에 CAN 위치를 보간 → 상호상관
    g = np.arange(t0 + 1, t1 - 1, 0.005)
    a = np.interp(g, pub[:, 0], pub[:, 1]); b = np.interp(g, c[:, 0], c[:, 1])
    a -= a.mean(); b -= b.mean()
    best = max(range(0, 200), key=lambda L: np.corrcoef(a[:len(a) - L], b[L:])[0, 1])   # 양의 상관만 (반주기 역상 오선택 방지)
    r = np.corrcoef(a[:len(a) - best], b[best:])[0, 1]
    can_rate = len(c) / (c[-1, 0] - c[0, 0])
    ct = drv.get("cmd_timing") or []
    ct_mid = [c for t, c in ct if t0 + 5 <= t <= t1 - 1]
    ct_s = (f" | stage8 기록: 지연 p50 {ct_mid[-1]['p50_age_ms_1s']} ms, 최대(1s) {max(c['max_age_ms_1s'] or 0 for c in ct_mid)} ms, "
            f"누락 {ct_mid[-1]['seq_gaps']}, 늦음 {ct_mid[-1]['stale']}") if ct_mid else ""
    print(f"{v}: 명령 처리 {rate:5.1f}/s (발행 50) | 상태 수신 {st_rate:5.1f}/s | CAN 명령 {can_rate:5.1f}/s | 발행→CAN 지연 {best*5:4d} ms (r={r:+.3f})" + ct_s)
