"""실물 브리지 블랙박스 목록(읽기 전용). 파일별: 크기·줄 수·틱/사건 수·시각·길이·체크포인트·c5 필드 유무."""
import glob, json, os, sys, hashlib, datetime, collections
from multiprocessing import Pool

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.dirname(os.path.abspath(__file__))

def scan(path):
    r = dict(path=path, size=os.path.getsize(path), n_lines=0, n_tick=0, n_event=0, bad=0)
    ck = collections.Counter(); ev = collections.Counter(); hs = collections.Counter(); gm = collections.Counter()
    c5n = 0; c5keys = set(); mode1 = 0; live_armed = 0; t0 = None; t1 = None; last_stop = None; n_on = None
    te0 = None; te1 = None; live_any = 0
    with open(path, "rb") as f:
        for ln in f:
            if not ln.strip():
                continue
            r["n_lines"] += 1
            try:
                d = json.loads(ln)
            except ValueError:
                r["bad"] += 1; continue
            ts = d.get("timestamp")
            if ts is not None:
                te0 = ts if te0 is None else te0; te1 = ts
            if "event" in d:
                r["n_event"] += 1; ev[str(d["event"])] += 1; continue
            if "qpos_rad" not in d:
                r["n_event"] += 1; ev["(기타:" + ",".join(sorted(d.keys())[:3]) + ")"] += 1; continue
            r["n_tick"] += 1
            if t0 is None:
                t0 = ts
            t1 = ts
            ck[str(d.get("checkpoint"))] += 1
            hs[str(d.get("hang_state"))] += 1
            gm[str(d.get("guard_mode", "(필드 없음)"))] += 1
            if d.get("live"):
                live_any += 1
            if d.get("live") and d.get("armed"):
                live_armed += 1
            c5 = d.get("c5")
            if c5 is not None:
                c5n += 1; c5keys |= set(c5.keys()); mode1 += int(c5.get("mode") or 0); n_on = c5.get("n_on")
            if d.get("stop_reason"):
                last_stop = d.get("stop_reason")
    r.update(ck=dict(ck), ev=dict(ev), hs=dict(hs), gm=dict(gm), c5n=c5n, c5keys=sorted(c5keys), mode1=mode1, n_on=n_on,
             live_armed=live_armed, live_any=live_any, t0=t0, t1=t1, te0=te0, te1=te1, last_stop=last_stop)
    return r

def md5head(path, n=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:10]

if __name__ == "__main__":
    pats = [REPO + "/reports/real_*/**/rl_bridge_*.jsonl", REPO + "/.scratch/**/rl_bridge_*.jsonl",
            "/home/ryu/logs/**/rl_bridge_*.jsonl", "/home/ryu/archive/jetson_logs_0929/rl_bridge_*.jsonl"]
    files = []
    for p in pats:
        files += glob.glob(p, recursive=True)
    files = sorted(set(files))
    # 중복 제거: (파일명 앞 31 자 = rl_bridge_YYYYMMDD_HHMMSS, 크기, 앞 1MB md5)
    seen = {}; uniq = []; dup = collections.defaultdict(list)
    for p in files:
        key = (os.path.basename(p)[:25], os.path.getsize(p), md5head(p) if os.path.getsize(p) else "0")
        if key in seen:
            dup[seen[key]].append(p)
        else:
            seen[key] = p; uniq.append(p)
    with Pool(12) as pool:
        res = pool.map(scan, uniq)
    for r in res:
        r["dups"] = dup.get(r["path"], [])
    json.dump(res, open(os.path.join(OUT, "inventory.json"), "w"), ensure_ascii=False, indent=1)
    res.sort(key=lambda r: os.path.basename(r["path"]))
    for r in res:
        f = lambda t: "—" if t is None else datetime.datetime.fromtimestamp(t).strftime("%m-%d %H:%M:%S")
        dur = (r["t1"] - r["t0"]) if r["t0"] is not None else 0.0
        print(f"{os.path.basename(r['path']):<46} {r['size']/1e6:7.1f}MB 줄{r['n_lines']:>6} 틱{r['n_tick']:>6} 사건{r['n_event']:>3} "
              f"{f(r['t0'])}–{f(r['t1'])[6:]} {dur:6.1f}s live&armed {r['live_armed']:>6} ck={r['ck']} c5틱={r['c5n']} mode1={r['mode1']} "
              f"c5keys={r['c5keys']} gm={r['gm']} stop={r['last_stop']} ev={r['ev']} dup={len(r['dups'])} [{os.path.dirname(r['path']).replace(REPO, '.')}]")
