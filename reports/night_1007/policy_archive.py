"""폐기 후보(문서에 기각·무효 기록이 있는 run)를 logs/rsl_rl/_archive_1007/ 로 옮긴다 — 지우지 않는다(같은 디스크 안 이동, 되돌릴 수 있음).
사용: python3 reports/night_1007/policy_archive.py [--apply] [--undo <기록 json>]
대상 = policy_inventory.json 의 cls '폐기 후보' ∧ sub '문서 근거' 중: 이미 분리 폴더(_void_·_archive_) 아래가 아니고, 어떤 세션 scratchpad 스크립트도 이름으로 가리키지 않고,
실행 중 프로세스 인자에도 없는 것. 대상 run 을 가리키는 심볼릭 링크는 먼저 지운다(링크 목록을 기록). 기록: reports/night_1007/policy_archive_1008.json."""
import argparse, glob, json, os, subprocess, sys, time
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INV = os.path.join(REPO, "reports", "night_1007", "policy_inventory.json")
REC = os.path.join(REPO, "reports", "night_1007", "policy_archive_1008.json")
DST = "logs/rsl_rl/_archive_1007"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true"); ap.add_argument("--undo", default="")
    a = ap.parse_args(); os.chdir(REPO)
    if a.undo:
        rec = json.load(open(a.undo)); n = 0
        for m in rec["moved"]:
            if os.path.isdir(m["to"]) and not os.path.exists(m["from"]):
                os.makedirs(os.path.dirname(m["from"]), exist_ok=True); os.rename(m["to"], m["from"]); n += 1
        for l in rec["links_removed"]:
            if not os.path.lexists(l["link"]):
                os.symlink(l["target"], l["link"])
        print(f"되돌림: run {n} 개 · 링크 {len(rec['links_removed'])} 개"); return 0
    d = json.load(open(INV)); runs = [r for r in d["runs"] if r["cls"] == "폐기 후보" and r.get("sub") == "문서 근거"]
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    scripts = ""
    for root in glob.glob("/tmp/claude-1000/-home-ryu-humanoid-leg-test1/*/scratchpad"):
        for f in glob.glob(root + "/**/*.sh", recursive=True):
            try:
                if os.path.getsize(f) < 2_000_000:
                    scripts += open(f, errors="ignore").read() + "\n"
            except OSError:
                pass
    todo = []; skip = []
    for r in runs:
        p = r["path"]; nm = r["name"]
        if "/_void_" in "/" + p or "/_archive_" in "/" + p:
            skip.append((nm, "이미 분리 폴더 아래")); continue
        if not os.path.isdir(p) or os.path.islink(p):
            skip.append((nm, "실제 폴더 아님")); continue
        if nm in ps:
            skip.append((nm, "실행 중 프로세스가 가리킴")); continue
        if nm in scripts:
            skip.append((nm, "scratchpad 대기열 스크립트가 가리킴")); continue
        if r.get("children") or r.get("hard_refs"):
            skip.append((nm, f"자식 {len(r.get('children') or [])}·직접 참조 {len(r.get('hard_refs') or [])}")); continue
        todo.append(r)
    # 대상 run 을 가리키는 링크
    targets = {os.path.realpath(r["path"]): r for r in todo}; links = []
    for exp in glob.glob("logs/rsl_rl/*"):
        if not os.path.isdir(exp) or os.path.basename(exp).startswith("_"):
            continue
        for e in os.listdir(exp):
            lp = os.path.join(exp, e)
            if os.path.islink(lp) and os.path.realpath(lp) in targets:
                links.append({"link": lp, "target": os.readlink(lp)})
    size = sum(r.get("size_bytes", 0) for r in todo)
    print(f"대상 {len(todo)} run · {size / 2**30:.2f} GiB · 지울 링크 {len(links)} · 건너뜀 {len(skip)}")
    import collections
    print("건너뜀 사유:", dict(collections.Counter(w.split(' ')[0] if w.startswith('자식') else w for _, w in skip)))
    if not a.apply:
        for r in todo[:5]:
            print("  예:", r["path"], "→", os.path.join(DST, os.path.basename(r["exp"]), r["name"]))
        print("(--apply 로 실행)"); return 0
    rec = {"when": time.strftime("%Y-%m-%dT%H:%M:%S"), "dst": DST, "moved": [], "links_removed": links, "skipped": [{"name": n, "why": w} for n, w in skip]}
    for l in links:
        os.unlink(l["link"])
    for r in todo:
        to = os.path.join(DST, os.path.basename(r["exp"]), r["name"]); os.makedirs(os.path.dirname(to), exist_ok=True)
        assert not os.path.exists(to), to
        os.rename(r["path"], to)
        rec["moved"].append({"from": r["path"], "to": to, "size_bytes": r.get("size_bytes", 0), "n_ckpt": r.get("n_ckpt"), "reason": (r.get("reasons") or [""])[0] if isinstance(r.get("reasons"), list) else str(r.get("reasons"))[:200]})
        json.dump(rec, open(REC + ".tmp", "w"), ensure_ascii=False, indent=1); os.replace(REC + ".tmp", REC)
    broken = [p for p in glob.glob("logs/rsl_rl/*/*") if os.path.islink(p) and not os.path.exists(p)]
    print(f"옮김 {len(rec['moved'])} · 링크 제거 {len(links)} · 끊어진 링크 {len(broken)} · 기록 {os.path.relpath(REC, REPO)}")
    for b in broken[:10]:
        print("  끊어진 링크:", b, "→", os.readlink(b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
