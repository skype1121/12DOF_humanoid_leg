#!/usr/bin/env python3
"""미러된 콘솔 상태(raw/state.jsonl)를 따라가다가 정책 실행 한 번이 끝나면 종료한다(로컬 파일만 읽는다).
끝 = 브리지가 살아서 돌다가 고장·정지·내려감, 또는 공중 폭주 차단. 미러가 40 s 넘게 멈춰도 종료(코드 3)."""
import json, os, sys, time
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
P, HB = os.path.join(D, "state.jsonl"), os.path.join(D, ".heartbeat")
f = open(P, "rb"); f.seek(0, 2)
seen_live = False; end_t = None; why = None; buf = b""; last = {}
while True:
    chunk = f.read()
    if not chunk:
        if time.time() - os.path.getmtime(HB) > 90:
            print("미러 멈춤(90 s 넘게 갱신 없음)"); sys.exit(3)
        # 파일은 스택이 내려간 뒤에만 가져온다(미러가 제어 중에는 ssh 를 쓰지 않는다) — 가져온 뒤, 또는 4 분이 지나면 알린다
        SY = os.path.join(D, "sync.log")
        synced = end_t and os.path.isfile(SY) and os.path.getmtime(SY) > end_t + 3
        if end_t and (synced or time.time() - end_t > 240):
            print(f"실행 끝: {why} | 시도 {last.get('n')} | 마지막 kew3 {last.get('k')} | 파일 {'가져옴' if synced else '아직(스택이 살아 있음)'}"); sys.exit(0)
        time.sleep(1); continue
    buf += chunk
    *lines, buf = buf.split(b"\n")
    for ln in lines:
        try: d = json.loads(ln.decode("utf-8", "replace"))
        except Exception: continue
        b = d.get("bridge") or {}; k = b.get("kew3") or {}; tr = d.get("trial") or {}
        live = bool(b.get("live")); fault = b.get("fault"); tf = tr.get("fault")
        if live and not fault and not tf:
            seen_live = True; last = {"n": tr.get("number"), "k": {x: k.get(x) for x in ("mode", "ticks", "n_air_to_ke", "air_hold", "air_dev")}}
        if seen_live and end_t is None:
            if k.get("air_hold"): why = "공중 폭주 차단"
            elif fault or tf: why = f"고장 {fault or tf}"
            elif not live: why = "브리지 내려감"
            if why: end_t = time.time()
