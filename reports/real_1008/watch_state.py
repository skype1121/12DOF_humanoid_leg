#!/usr/bin/env python3
"""미러된 콘솔 상태(state.jsonl)를 따라가며 바뀐 것만 한 줄씩 찍는다(읽기 전용, 로컬 파일만).
사용: watch_state.py [최대 초=175] [뒤에서부터 읽을 행 수=15]"""
import json, time, os, sys
LIM = float(sys.argv[1]) if len(sys.argv) > 1 else 175.0
BACK = int(sys.argv[2]) if len(sys.argv) > 2 else 15
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "state.jsonl")
f = open(P, "rb")
sz = os.path.getsize(P)
f.seek(max(0, sz - 12000 * BACK)); f.readline()
rows = f.read().split(b"\n")
buf = rows[-1]; rows = rows[:-1][-BACK:]
def row(d):
    b = d["bridge"]; k = b.get("kew3") or {}; w = (d["walk"].get("bridge") or {})
    return dict(t=time.strftime("%T", time.localtime(d["_t"])), live=b.get("live"), mode=k.get("mode"), ticks=k.get("ticks"),
                c="".join(str(int(x)) for x in d["foot"]["contact"]), fwd=d["imu"]["fwd"], lat=d["imu"]["lat"], wxy=w.get("w_xy"),
                dev=k.get("air_dev"), hold=k.get("air_hold"), a2k=k.get("n_air_to_ke"), ge=w.get("ge"), std=w.get("standing"),
                hs=b.get("hang_state"), fault=b.get("fault"), phase=d["trial"].get("phase"), tf=d["trial"].get("fault"), blk=(k.get("block") or "")[:10])
t0 = time.time(); last = None; lastp = 0; ke_t = None
def handle(ln):
    global last, lastp, ke_t
    try: d = json.loads(ln.decode("utf-8", "replace"))
    except Exception: return None
    r = row(d); key = (r["live"], r["mode"], r["c"], r["hold"], r["a2k"], r["hs"], r["fault"], r["phase"], r["tf"], r["ge"], r["std"])
    if key != last or time.time() - lastp > 6:
        print(" ".join(f"{k}={v}" for k, v in r.items()), flush=True); last = key; lastp = time.time()
    if r["mode"] not in ("air", None) and ke_t is None: ke_t = time.time()
    if r["hold"]: return "공중 폭주 차단 걸림"
    if r["live"] is False and d["_t"] > t0: return "브리지 내려감"
    if ke_t and time.time() - ke_t > 25: return "KE 인계 뒤 25 s 관찰 끝"
    return None
for ln in rows:
    handle(ln)
while time.time() - t0 < LIM:
    chunk = f.read()
    if not chunk:
        time.sleep(0.5); continue
    buf += chunk
    *lines, buf = buf.split(b"\n")
    for ln in lines:
        why = handle(ln)
        if why:
            print("--", why); sys.exit(0)
print("-- 시간 끝")
