#!/usr/bin/env python3
"""1010 점검 요약(NEWBODY_CHECKS.md): 실행 태그별 낙상·못 켠 걷기·전 구간 완료·낙상 위치·회복 모드 켜짐·스텝 지표, 그리고 기준 실행과의 같은 env 짝 비교.
입력: log_picture/{ground_recovery_<tag>_envs.csv, ground_recovery_<tag>.txt, kwloop_<tag>.npz, transall_<tag>.npz}. 유효 여부는 newbody/chk/out(또는 tr/out)에 복사됐는지로 본다.
사용: chk_summ.py [--ref <기준 tag>] <tag> ...   /   --push "t1,t2" 를 주면 밀기 뒤 4 s 안 낙상을 밀기별로 센다."""
import argparse
import csv
import json
import os
import re
from math import sqrt
import numpy as np

H = os.path.dirname(os.path.abspath(__file__)); LP = os.path.join(H, "..", "..", "..", "log_picture")


def wil(k, n, z=1.96):
    if n == 0:
        return "—"
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{k}/{n} = {100 * p:.1f} % ({100 * max(0, c - h):.1f}–{100 * (c + h):.1f})"


def valid(tag):
    return any(os.path.isfile(os.path.join(H, d, "out", f"ground_recovery_{tag}.txt")) for d in ("chk", "tr"))


def load(tag):
    pz = os.path.join(LP, f"kwloop_{tag}.npz"); pc = os.path.join(LP, f"ground_recovery_{tag}_envs.csv"); pt = os.path.join(LP, f"ground_recovery_{tag}.txt")
    if not (os.path.isfile(pz) and os.path.isfile(pc) and valid(tag)):
        return None
    z = np.load(pz, allow_pickle=True); txt = open(pt, errors="ignore").read()
    ev = json.loads(str(z["events"])); script = json.loads(str(z["script"])); evt = z["ev_t"]; t = z["t"].astype(float); kind = z["kind"]
    pp = z["pre_pass"].astype(bool); fell = z["fell"].astype(bool); tf = z["t_fall_post"].astype(float)
    N = len(pp); D = dict(tag=tag, N=N, pp=pp, fell=fell & pp, tf=tf, t=t, kind=kind, script=script, evt=evt, ev=ev, txt=txt)
    m = re.search(r"최대 변위: 전 env ([\d.]+)°", txt); D["lockdev"] = float(m.group(1)) if m else float("nan")
    m = re.search(r"전체 질량 env 평균 (?:[\d.]+→)?([\d.]+) kg", txt); D["mass"] = float(m.group(1)) if m else float("nan")
    walks = [i for i, (tt, w, c) in enumerate(script) if w and tt < t[-1]]
    never = 0; done = 0; surv = 0; en_dly = []; s2k = []; cap = 0; fall_pos = {"걷기 켠 뒤 2 s 안": 0, "걷기 중(2 s 뒤)": 0, "정지 전환": 0, "KE 서기": 0}
    rec_env = 0; rec_ticks = 0; ke_ticks = 0; rec_fall = 0
    for e in np.nonzero(pp)[0]:
        E = ev[e]; dead = bool(fell[e]); t_end = tf[e] if dead and tf[e] >= 0 else t[-1] + 0.02
        en = [tt for tt, nm in E if nm == "to_w3"]; tk = [tt for tt, nm in E if nm == "to_ke"]; st = [tt for tt, nm in E if nm == "stop_req"]
        ok_all = True
        for j, k in enumerate(walks):
            treq = float(evt[k][e])
            if treq >= t_end:
                ok_all = False; continue
            nxt = float(evt[walks[j + 1]][e]) if j + 1 < len(walks) else 1e9
            acc = [x for x in en if treq - 1e-6 <= x < nxt]
            if acc:
                en_dly.append(acc[0] - treq)
                tstop = float(evt[k + 1][e]) if k + 1 < len(script) else 1e9
                back = [x for x in tk if x >= tstop - 1e-6 and x < nxt]
                if back:
                    s2k.append(back[0] - tstop); cap += int(back[0] - tstop >= 2.95)
                elif tstop < t_end:
                    ok_all = False
            else:
                ok_all = False
                if not dead or t_end > min(nxt, treq + 4.0):
                    never += 1
        if not dead:
            surv += 1; done += int(ok_all)
        else:      # 낙상 위치
            kd = kind[min(int(np.searchsorted(t, t_end)) - 1, len(t) - 1), e]; last_en = max([x for x in en if x <= t_end], default=None); last_st = max([x for x in st if x <= t_end], default=None)
            if kd == 2:
                if last_st is not None and (last_en is None or last_st > last_en):
                    fall_pos["정지 전환"] += 1
                elif last_en is not None and t_end - last_en <= 2.0:
                    fall_pos["걷기 켠 뒤 2 s 안"] += 1
                else:
                    fall_pos["걷기 중(2 s 뒤)"] += 1
            else:
                fall_pos["KE 서기"] += 1
        ie = int(np.searchsorted(t, t_end)); kk = kind[:ie, e]
        r = int((kk == 1).sum()); rec_ticks += r; ke_ticks += int(((kk == 0) | (kk == 1)).sum()); rec_env += int(r > 0); rec_fall += int(r > 0 and dead)
    D.update(n=int(pp.sum()), k=int((fell & pp).sum()), never=never, surv=surv, done=done, en_p95=(np.percentile(en_dly, 95) if en_dly else float("nan")),
             s2k_p95=(np.percentile(s2k, 95) if s2k else float("nan")), cap=cap, fall_pos=fall_pos, rec_env=rec_env, rec_tick_pct=100 * rec_ticks / max(1, ke_ticks), nwalk=len(walks), nacc=len(en_dly))
    m = re.search(r"\[굳은 관절 제외\].*?든 ([\d.]+) % 틱", txt); D["sat"] = float(m.group(1)) if m else float("nan")
    return D


def steps(tag):
    """걷기 구간 스텝 지표(추적 env, 실제 env 접촉 관측 기준): 발별 이륙/s, 한발 지지 비율, 앞 0.06 구간 몸통 프레임 앞 속도."""
    pa = os.path.join(LP, f"transall_{tag}.npz"); pk = os.path.join(LP, f"kwloop_{tag}.npz")
    if not os.path.isfile(pa):
        return None
    a = np.load(pa, allow_pickle=True); k = np.load(pk, allow_pickle=True); na = a["cont"].shape[1]
    kind = k["kind"][:, :na]; cont = a["cont"]; alive = a["alive"]; ge = a["ge"]; vx = a["vxy"][..., 0].astype(float); pp = k["pre_pass"][:na].astype(bool)
    w3 = (kind == 2) & alive & ge & pp[None, :]
    if w3.sum() == 0:
        return dict(lift=float("nan"), single=float("nan"), vx=float("nan"), walk_s=0.0)
    lift = ((cont[:-1] & ~cont[1:]) & w3[1:, :, None]).sum() / (w3.sum() * 0.02) / 2.0
    single = float(((cont.sum(-1) == 1) & w3).sum() / w3.sum())
    script = json.loads(str(k["script"])); evt = k["ev_t"][:, :na]; t = a["t"].astype(float); fw = np.zeros_like(w3)
    for i, (tt, w, c) in enumerate(script):
        if w and abs(c[0]) > 1e-6 and i + 1 < len(script):
            fw |= (t[:, None] >= evt[i][None, :] + 2.0) & (t[:, None] < evt[i + 1][None, :])
    fw &= w3
    return dict(lift=float(lift), single=single, vx=float(vx[fw].mean()) if fw.any() else float("nan"), walk_s=float(w3.sum() * 0.02 / max(1, pp.sum())))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", default=""); ap.add_argument("--push", default=""); ap.add_argument("--steps", action="store_true"); ap.add_argument("tags", nargs="+")
    a = ap.parse_args(); ref = load(a.ref) if a.ref else None
    hdr = "| 실행 | 유효(변위°·질량 kg) | 낙상 (95 % Wilson) | 끝내 못 켠 걷기 | 생존 env 전 구간 완료 | 켜기 지연 p95 · 정지→인계 p95(상한 도달) | 낙상 위치(켠 뒤 2 s 안 / 걷기 중 / 정지 전환 / KE 서기) | 회복 모드 켜진 env · KE 틱 중 회복 | 포화 틱 % |"
    if ref:
        hdr += f" 짝: 기준만 낙상 / 이쪽만 낙상 |"
    if a.steps:
        hdr += " 이륙/s(발당) · 한발 지지 % · 앞 0.06 속도 |"
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    for tag in ([a.ref] if a.ref else []) + a.tags:
        D = load(tag)
        if D is None:
            print(f"| {tag} | **무효/없음** |" + " |" * (hdr.count("|") - 3)); continue
        fp = D["fall_pos"]
        row = (f"| {tag} | {D['lockdev']:.3f} · {D['mass']:.2f} | {wil(D['k'], D['n'])} | {D['never']} | {D['done']}/{D['surv']} | {D['en_p95']:.2f} s · {D['s2k_p95']:.2f} s ({D['cap']}) | "
               f"{fp['걷기 켠 뒤 2 s 안']} / {fp['걷기 중(2 s 뒤)']} / {fp['정지 전환']} / {fp['KE 서기']} | {D['rec_env']}/{D['n']} · {D['rec_tick_pct']:.2f} % | {D['sat']:.1f} |")
        if ref:
            if D["N"] == ref["N"]:
                c = D["pp"] & ref["pp"]; row += f" {int((ref['fell'] & ~D['fell'] & c).sum())} / {int((D['fell'] & ~ref['fell'] & c).sum())} (공통 {int(c.sum())}) |"
            else:
                row += " — |"
        if a.steps:
            s = steps(tag); row += f" {s['lift']:.2f} · {100 * s['single']:.0f} · {s['vx']:+.4f} (걷기 {s['walk_s']:.1f} s/env) |" if s else " — |"
        print(row)
        if a.push:
            out = []
            for tp in [float(x) for x in a.push.split(",")]:
                at = D["pp"] & ~(D["fell"] & (D["tf"] >= 0) & (D["tf"] < tp)); f4 = at & D["fell"] & (D["tf"] >= tp) & (D["tf"] < tp + 4.0)
                out.append(f"밀기 {tp:g} s: 그때 서 있던 {int(at.sum())} 중 4 s 안 낙상 {wil(int(f4.sum()), int(at.sum()))}")
            print("|  ↳ " + " · ".join(out) + " |")


if __name__ == "__main__":
    main()
