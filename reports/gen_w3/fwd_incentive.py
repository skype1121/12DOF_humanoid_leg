"""전진 유인 진단 — 추적 궤적에서 w3_place·w3_stride·w3_track(서기 대비 개선분)·새 w3_stride_lin 의 값을 명령별로 다시 계산한다(CPU·numpy, Isaac 불필요).

사용: python3 reports/gen_w3/fwd_incentive.py [--tag x12_w3load_cmds_d2_s21] [--lp DIR] [--noise 0.056]
입력: log_picture/transall_<TAG>.npz(q·roll·pitch·ffwd·flat·ge·alive) + kwloop_<TAG>.npz(xy·yaw·script). 대본 구간은 torque_phase.SEG 와 같다.

계산(보상 코드 rl_walking/w3.py 와 같은 식, 착지 사건은 운동학으로 다시 찾음 — 스윙발이 8 mm 넘게 높았다가 내려온 틱, 공중 ≥ 0.12 s·정점 ≥ 3 cm):
  place   = exp(−‖(rx, ry) − (vx/(4f), ±반폭 + vy/(4f))‖²/0.04²)        (rx·ry = 착지 발의 골반 기준 요 프레임 위치; 반폭 0.10 = 현 설정, 0.17 = W3S1e 수리값)
  stride  = exp(−((dx − vx/f)/0.03)²) − exp(−((vx/f)/0.03)²)             (dx = 스윙발 이륙→착지 월드 변위의 앞 성분; |vx| ≥ 0.02 일 때만)
  track   = exp(−(e/0.05)²) − exp(−(|c|/0.05)²)                          (e = 한 주기(45 틱) 이동평균 속도와 명령의 차; |c| ≥ 0.02 일 때만)
  lin     = clip(Δp·ΔP*/‖ΔP*‖², −1, 1)                                   (W3S3tF 의 w3_stride_lin; ΔP* = (vx/f, vy/f))
--noise: env 별 보폭 오프셋(표준편차 m)을 dx 에 더했을 때 stride·lin 의 기대값 — 학습 조건의 속도 오차(한 주기 평균 0.062 m/s ÷ 1.1 Hz = 0.056 m)를 흉내 낸 민감도 계산.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torque_phase as TP  # noqa: E402

F = 1.1
DT = TP.DT
SEGC = [("제자리", 3, 8, (0.0, 0.0, 0.0)), ("앞 0.03", 9, 14, (0.03, 0.0, 0.0)), ("앞 0.06", 15, 20, (0.06, 0.0, 0.0)), ("앞 0.10", 21, 26, (0.10, 0.0, 0.0)),
        ("뒤 0.04", 32, 37, (-0.04, 0.0, 0.0)), ("뒤 0.06", 38, 43, (-0.06, 0.0, 0.0)), ("옆 0.035", 49, 54, (0.0, 0.035, 0.0)), ("회전 0.15", 60, 65, (0.0, 0.0, 0.15))]


def landings(tag, lp, model):
    """명령 구간별 착지 사건 목록과 한 주기 평균 속도. 반환 {이름: dict}."""
    z = np.load(os.path.join(lp, f"transall_{tag}.npz"), allow_pickle=True); k = np.load(os.path.join(lp, f"kwloop_{tag}.npz"), allow_pickle=True)
    script = json.loads(str(k["script"]))
    for (nm, a, b, c), row in zip(SEGC, [r for r in script if r[1]]):
        assert tuple(round(x, 4) for x in row[2]) == tuple(round(x, 4) for x in c), f"대본과 구간 명령 불일치: {nm} {row}"
    pre = z["pre_pass"].astype(bool); K = TP.K
    xy = k["xy"].astype(np.float64); yaw = k["yaw"].astype(np.float64)
    ffwd = z["ffwd"].astype(np.float64); flat = z["flat"].astype(np.float64)
    roll = z["roll"].astype(np.float64); pitch = z["pitch"].astype(np.float64); Tn, N = roll.shape
    fl = model.foot_low(z["q"].astype(np.float64).reshape(-1, 12), TP.gdir_from_rp(roll.reshape(-1), pitch.reshape(-1))).reshape(Tn, N, 2)
    hgt = np.stack([fl[..., 0] - fl[..., 1], fl[..., 1] - fl[..., 0]], -1); air = hgt > 0.008
    fh = np.stack([np.cos(yaw), np.sin(yaw)], -1); lh = np.stack([-np.sin(yaw), np.cos(yaw)], -1)
    pw = xy[:, :, None, :] + ffwd[..., None] * fh[:, :, None, :] + flat[..., None] * lh[:, :, None, :]
    Kc = int(round((1.0 / F) / DT)); out = {}
    for nm, a, b, cmd in SEGC:
        ok = pre & z["alive"][K(b) - 1] & z["ge"][K(a)]
        ev = []                                                                   # (dx, dy, rx, ry, 발)
        for e in np.where(ok)[0]:
            for i in (0, 1):
                ar = air[K(a) - 30:K(b), e, i]; lift = None; apex = 0.0
                for j in range(1, len(ar)):
                    tt = K(a) - 30 + j
                    if ar[j] and not ar[j - 1]:
                        lift = tt; apex = 0.0
                    if ar[j] and lift is not None:
                        apex = max(apex, hgt[tt, e, i])
                    if (not ar[j]) and ar[j - 1] and lift is not None:
                        if tt >= K(a) and (tt - lift) * DT >= 0.12 and apex >= 0.03:
                            d = pw[tt, e, i] - pw[lift, e, i]; c_, s_ = np.cos(yaw[tt, e]), np.sin(yaw[tt, e])
                            ev.append((c_ * d[0] + s_ * d[1], -s_ * d[0] + c_ * d[1], ffwd[tt, e, i], flat[tt, e, i], i))
                        lift = None
        vb = (xy[K(a) + Kc:K(b)] - xy[K(a):K(b) - Kc]) / (Kc * DT); yw = yaw[K(a) + Kc:K(b)]
        vf = (vb[..., 0] * np.cos(yw) + vb[..., 1] * np.sin(yw))[:, ok]; vl = (-vb[..., 0] * np.sin(yw) + vb[..., 1] * np.cos(yw))[:, ok]
        out[nm] = dict(cmd=cmd, ev=np.array(ev, float).reshape(-1, 5), dur=(b - a) * int(ok.sum()), vf=vf, vl=vl, n_env=int(ok.sum()))
    return out


def stride_val(dx, tx, sig=0.03):
    return np.exp(-((dx - tx) / sig) ** 2) - np.exp(-(tx / sig) ** 2)


def lin_val(dx, dy, tx, ty):
    return np.clip((dx * tx + dy * ty) / max(tx * tx + ty * ty, 1e-8), -1.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp", default=os.path.join(TP.REPO, "log_picture")); ap.add_argument("--tag", default="x12_w3load_cmds_d2_s21")
    ap.add_argument("--noise", type=float, default=0.056, help="env 별 보폭 오프셋 표준편차[m] — 민감도 계산")
    a = ap.parse_args()
    model = TP.Model(os.path.join(TP.REPO, "rl_walking", "assets", "biped12_final3.urdf"))
    D = landings(a.tag, a.lp, model)
    print(f"# fwd_incentive.py — {a.tag} (배포 코드 폐루프·지연 2·무외란: 학습 조건이 아니다) · 가중 place 10·stride 10·track 3·lin 10, 사건 항은 가중 × dt 0.02")
    print("\n## 1. 명령별 값")
    print("| 명령 | 유효 착지 /s | 착지 옆 위치 \\|ry\\| 중앙 m · 앞 위치 rx m | place: 반폭 0.10 → 초당 · 반폭 0.17 → 초당 | 스윙발 변위 dx 중앙 [p10, p90] m · 목표 vx/f | stride 평균 · 양수 % → 초당 | 한 주기 평균 속도 앞 · 옆 m/s · 오차 e 중앙 | track 평균 · 음수 % → 초당 | lin 평균 → 초당 |"); print("|---|---|---|---|---|---|---|---|---|")
    for nm, d in D.items():
        cmd = d["cmd"]; ev = d["ev"]; rate = len(ev) / d["dur"]; tx, ty = cmd[0] / F, cmd[1] / F
        pl = {}
        for hw in (0.10, 0.17):
            tyy = np.where(ev[:, 4] == 0, hw, -hw) + cmd[1] / (4 * F)
            pl[hw] = np.exp(-((ev[:, 2] - cmd[0] / (4 * F)) ** 2 + (ev[:, 3] - tyy) ** 2) / 0.04 ** 2).mean()
        st = stride_val(ev[:, 0], tx) if abs(cmd[0]) >= 0.02 else np.zeros(len(ev))
        ln = lin_val(ev[:, 0], ev[:, 1], tx, ty) if (abs(cmd[0]) >= 0.02 or abs(cmd[1]) >= 0.02) else np.zeros(len(ev))
        cn = float(np.hypot(cmd[0], cmd[1])); e_ = np.hypot(d["vf"] - cmd[0], d["vl"] - cmd[1])
        tr = (np.exp(-(e_ / 0.05) ** 2) - np.exp(-(cn / 0.05) ** 2)) if cn >= 0.02 else np.zeros_like(e_)
        ev_w = 10 * DT * rate
        print(f"| {nm} | {rate:.2f} | {np.median(np.abs(ev[:, 3])):.3f} · {np.median(ev[:, 2]):+.3f} | {pl[0.10]:.3f} → {ev_w * pl[0.10]:+.3f} · {pl[0.17]:.3f} → {ev_w * pl[0.17]:+.3f} | "
              f"{np.median(ev[:, 0]):+.3f} [{np.percentile(ev[:, 0], 10):+.3f}, {np.percentile(ev[:, 0], 90):+.3f}] · {tx:+.3f} | {st.mean():+.2f} · {100 * (st > 0).mean():.0f} → {ev_w * st.mean():+.3f} | "
              f"{np.median(d['vf']):+.3f} · {np.median(d['vl']):+.3f} · {np.median(e_):.3f} | {tr.mean():+.2f} · {100 * (tr < 0).mean():.0f} → {3 * tr.mean():+.2f} | {ln.mean():+.2f} → {ev_w * ln.mean():+.3f} |")
    print(f"\n## 2. env 별 보폭 오프셋(표준편차 {a.noise:.3f} m)을 더했을 때의 기대값 — 전진을 절반으로 줄이면 잃는 보상(초당, 그 명령 동안)")
    print("| 명령 | stride(가우시안 σ 0.03): 온전 → 절반 = 손실 | 같은 것, 오프셋 있을 때 | lin(선형): 온전 → 절반 = 손실 | 같은 것, 오프셋 있을 때 |"); print("|---|---|---|---|---|")
    rng = np.random.default_rng(0); eps = rng.normal(0.0, a.noise, 200000)
    for nm, d in D.items():
        cmd = d["cmd"]
        if abs(cmd[0]) < 0.02:
            continue
        tx = cmd[0] / F; w = 10 * DT * 2.2
        s_full, s_half = stride_val(tx, tx), stride_val(0.5 * tx, tx)
        s_full_n, s_half_n = stride_val(tx + eps, tx).mean(), stride_val(0.5 * tx + eps, tx).mean()
        l_full, l_half = lin_val(tx, 0, tx, 0), lin_val(0.5 * tx, 0, tx, 0)
        l_full_n, l_half_n = lin_val(tx + eps, 0, tx, 0).mean(), lin_val(0.5 * tx + eps, 0, tx, 0).mean()
        print(f"| {nm} | {w * s_full:+.3f} → {w * s_half:+.3f} = {w * (s_full - s_half):.3f} | {w * s_full_n:+.3f} → {w * s_half_n:+.3f} = {w * (s_full_n - s_half_n):.3f} | "
              f"{w * l_full:+.3f} → {w * l_half:+.3f} = {w * (l_full - l_half):.3f} | {w * l_full_n:+.3f} → {w * l_half_n:+.3f} = {w * (l_full_n - l_half_n):.3f} |")
    print("- 오프셋 = 학습 조건에서 env 마다 다른 속도 치우침(관측에 선속도가 없어 정책이 고치지 못함)의 흉내. 학습 로그의 w3_drift −0.185/s ÷ 가중 3 = 한 주기 평균 속도 오차 0.062 m/s → 보폭으로 0.056 m.")


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    main()
