#!/usr/bin/env python3
"""젯슨 세션 기록·브리지 블랙박스·콘솔 상태를 '시작 시각부터'만 로컬로 받는다.
**제어 중에는 젯슨에 ssh 로 들어가지 않는다**(1008 22:02 변경): 5 초마다 콘솔 HTTP(/api/state) 한 번만 읽고(웹 화면이 늘 하는 요청과 같다),
콘솔이 띄운 프로세스가 하나도 없을 때에만 ssh 로 파일을 가져온다(find · tail — 읽기뿐, 제어·ROS CLI·쓰기 없음).
 까닭: 21:57 시도 7 에서 IMU 노드·콘솔이 0.2–0.3 s 멈춰 센서 신선도 정지가 났고, 2 초마다 ssh 로 읽던 종전 미러가 관여했는지 가려지지 않았다.
사용: mirror2.py <ssh control socket 경로> <시작 epoch>
 - 시작 때 이미 있던 파일은 시작 시각 뒤의 줄만 받는다(jsonl 은 ts/timestamp 로 거른다, 그 밖의 옛 파일은 받지 않는다).
 - 시작 뒤에 생긴 파일은 처음부터 받는다.
 - 받은 위치는 raw/.pos.json 에 적어 두어 다시 띄워도 이어 받는다."""
import json, os, subprocess, sys, time
CM_FULL, START = sys.argv[1], float(sys.argv[2])
HOST = "mama@172.16.35.89"
HERE = os.path.dirname(os.path.abspath(__file__)); DST = os.path.join(HERE, "raw"); os.makedirs(DST, exist_ok=True)
os.chdir(os.path.dirname(CM_FULL)); CM = "./" + os.path.basename(CM_FULL)       # 유닉스 소켓 경로 길이 제한 → 상대 경로
SSH = ["ssh", "-o", f"ControlPath={CM}", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", HOST]
POS_F = os.path.join(DST, ".pos.json"); ERR = os.path.join(DST, "mirror.err")
SINCE = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(START))
DAY = time.strftime("%Y-%m-%d 00:00:00", time.localtime(START))
RATE = 200_000          # 되짚어 받을 양 추정(바이트/초) — 타임라인 실측 약 112 kB/s


def err(msg):
    with open(ERR, "a") as f:
        f.write(time.strftime("%T ") + msg + "\n")


def ensure_master():
    if subprocess.run(SSH[:-1] + ["-O", "check", HOST], capture_output=True).returncode == 0:
        return True
    r = subprocess.run(["sshpass", "-p", "***", "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ControlMaster=yes", "-o", f"ControlPath={CM}",
                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3", "-o", "ConnectTimeout=5", "-N", "-f", HOST], capture_output=True)
    if r.returncode != 0:
        err("접속 실패 " + r.stderr.decode("utf-8", "replace")[-200:]); return False
    return True


def rsh(cmd, timeout=20):
    r = subprocess.run(SSH + [cmd], capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"rc {r.returncode}: {r.stderr.decode('utf-8', 'replace')[-200:]}")
    return r.stdout


def listing(since):
    out = rsh("cd ~/logs && find . -maxdepth 1 -name 'rl_bridge_*.jsonl' -newermt '%s' -printf '%%s %%P\\n'; "
              "find sessions -mindepth 2 -maxdepth 2 -path 'sessions/session_*' -type f -newermt '%s' -printf '%%s %%p\\n'" % (since, since))
    res = {}
    for ln in out.decode("utf-8", "replace").splitlines():
        sz, _, path = ln.partition(" ")
        if sz.isdigit() and path and ".." not in path:
            res[path] = int(sz)
    return res


def row_ts(line):
    try:
        d = json.loads(line.decode("utf-8", "replace"))
    except Exception:
        return None
    v = d.get("ts", d.get("timestamp"))
    return float(v) if isinstance(v, (int, float)) else None


def http_state():
    import urllib.request
    with urllib.request.urlopen("http://172.16.35.89:8800/api/state", timeout=3) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def sync(pos, first):
    """ssh 로 한 번 가져온다. 호출 쪽이 '콘솔이 띄운 프로세스 없음'을 확인한 뒤에만 부른다."""
    now = time.time()
    if first:                                   # 시작 때 있던 파일: jsonl 은 시작 시각 뒤 줄만(조금 되짚어 읽고 거른다), 나머지는 건너뛴다
        for path, sz in listing(DAY).items():
            if path.endswith(".jsonl"):
                back = int((now - START + 10) * RATE)
                pos[path] = {"pos": max(0, sz - back), "filter": True, "align": sz - back > 0}
            else:
                pos[path] = {"pos": sz, "filter": False, "align": False}
        with open(os.path.join(DST, "START.txt"), "w") as f:
            f.write(f"기록 시작 {SINCE} (epoch {START}) — 이 시각 전의 줄·파일은 받지 않았다. 젯슨 원본은 ~/logs 에 그대로 있다.\n")
            for path in sorted(pos):
                f.write(f"  시작 때 있던 파일 {path}: 읽기 시작 위치 {pos[path]['pos']}\n")
    got = 0
    for path, sz in listing(SINCE).items():
        st = pos.setdefault(path, {"pos": 0, "filter": False, "align": False})      # 시작 뒤에 생긴 파일은 처음부터
        while sz > st["pos"]:
            n = min(sz - st["pos"], 32_000_000)
            data = rsh("tail -c +%d ~/logs/'%s' | head -c %d" % (st["pos"] + 1, path, n), timeout=120)
            if not data:
                break
            st["pos"] += len(data); got += len(data)
            if st.get("align"):                     # 줄 중간에서 읽기 시작했을 수 있다 — 첫 줄바꿈까지 버린다
                i = data.find(b"\n")
                if i < 0:
                    continue
                data = data[i + 1:]; st["align"] = False
            local = os.path.join(DST, path); os.makedirs(os.path.dirname(local), exist_ok=True)
            if st.get("filter"):                    # 시작 시각 전 줄을 버린다(시각이 시작을 넘는 첫 줄부터는 거르지 않는다)
                keep = []; carry = st.pop("carry", "").encode("latin1")
                lines = (carry + data).split(b"\n"); tail = lines.pop()
                for ln in lines:
                    if st.get("filter"):
                        t = row_ts(ln)
                        if t is None or t < START:
                            continue
                        st["filter"] = False
                    keep.append(ln + b"\n")
                if st.get("filter"):
                    st["carry"] = tail.decode("latin1")
                    data = b"".join(keep)
                else:
                    data = b"".join(keep) + tail
            with open(local, "ab") as f:
                f.write(data)
    tmp = POS_F + ".tmp"
    json.dump(pos, open(tmp, "w")); os.replace(tmp, POS_F)
    return got


def main():
    pos = {}          # 경로 → {"pos": 젯슨 파일에서 읽은 위치, "filter": 시작 시각 전 줄을 버리는 중인지}
    if os.path.isfile(POS_F):
        pos = json.load(open(POS_F))
    first = not pos
    last_sync = 0.0; was_busy = True
    while True:
        try:
            d = http_state()
            busy = sorted(k for k, v in (d.get("procs") or {}).items() if isinstance(v, dict) and v.get("alive"))
            for p in d.get("procs", {}).values():
                if isinstance(p, dict) and "lines" in p:
                    p["lines"] = p["lines"][-3:]
            d["_t"] = round(time.time(), 2); d["_mirror"] = "제어 중 ssh 없음" if busy else "유휴"
            with open(os.path.join(DST, "state.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n")
            # 콘솔이 띄운 프로세스가 하나도 없을 때만 가져온다: 막 내려간 직후 한 번, 그 뒤로는 60 초마다
            if not busy and (was_busy or time.time() - last_sync > 60):
                if ensure_master():
                    got = sync(pos, first); first = False; last_sync = time.time()
                    with open(os.path.join(DST, "sync.log"), "a") as f:
                        f.write(time.strftime("%T ") + f"가져옴 {got} 바이트\n")
            was_busy = bool(busy)
            with open(os.path.join(DST, ".heartbeat"), "w") as f:
                f.write(str(int(time.time())))
        except Exception as e:  # noqa: BLE001
            err(f"{type(e).__name__}: {e}")
        time.sleep(5)


main()
