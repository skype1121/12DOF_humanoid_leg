"""회복 supervisor 격자 요약(1007): log_picture/w3eval_rg_<정책>_d<지연>_<모드>_s<seed>.json → 짝 표(seed 1+2 합). python3 rl_walking/scripts/summarize_rsup.py"""
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LP = os.path.join(REPO, "log_picture")


def J(t):
    p = os.path.join(LP, f"w3eval_{t}.json")
    return json.load(open(p)) if os.path.exists(p) else None


out = []
for pol, name in (("s4", "W3S4d-16299"), ("s3", "W3S3d-15800")):
    for d in (2, 4):
        runs = {m: [J(f"rg_{pol}_d{d}_{m}_s{s}") for s in (1, 2)] for m in ("off", "a", "b")}
        ref = next((r for r in runs["off"] if r), None)
        if ref is None:
            continue
        out += [f"### {name} · 지연 {d} (seed 1+2 합, 낙상/n · 정상 복귀 · 첫 회복 착지 힘 방향 변위 중앙 cm)", "",
                "| 그룹 | off | (a) 정지 래치 지연 | (b) 포획점 오프셋 |", "|---|---|---|---|"]
        for gi, g0 in enumerate(ref["dist"]["groups"]):
            row = [f"`{g0['label']}`"]
            for m in ("off", "a", "b"):
                fa = n = rc = 0; dd = []
                for r in runs[m]:
                    if r is None:
                        continue
                    g = r["dist"]["groups"][gi]; fa += g["falls"]; n += g["n"]; rc += g["recovered"]
                    dd.append(g.get("step1_disp_dir_cm_p10_50_90", [float("nan")] * 3)[1])
                row.append(f"{fa}/{n} ({100 * fa / max(n, 1):.0f} %) · {rc} · " + "/".join(f"{x:+.1f}" for x in dd))
            out.append("| " + " | ".join(row) + " |")
        fr = []
        for m in ("off", "a", "b"):
            ft = sum(r["rsup"]["false_trig"] for r in runs[m] if r and "rsup" in r); qt = sum(r["rsup"]["quiet_env_s"] for r in runs[m] if r and "rsup" in r)
            det = [(g["det_1s"], g["n"]) for r in runs[m] if r and "rsup" in r for g in r["rsup"]["groups"]]
            fr.append(f"{m}: 오발동 {ft:.0f} 회 / {qt / 60:.0f} env·분 = {ft / max(qt / 60, 1e-9):.3f} 회/env·분")
        detb = {}
        for r in runs["b"]:
            if r and "rsup" in r:
                for g in r["rsup"]["groups"]:
                    a_ = detb.setdefault(g["label"], [0, 0]); a_[0] += g["det_1s"]; a_[1] += g["n"]
        out += ["", "- " + " · ".join(fr), "- (b) 외란 뒤 1 s 안 감지: " + " · ".join(f"`{k}` {v[0]}/{v[1]}" for k, v in detb.items()), ""]
print("\n".join(out))
