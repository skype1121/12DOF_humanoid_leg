#!/usr/bin/env python3
"""스윙 발 이륙 직전의 지지쪽 힙 외전(hip_a)·발목 롤(ankle_r) 토크/각/CoP 동시 분석 — 기존 시뮬 기록(transall_*.npz)만 읽는다.

질문: 8번(right_hip_a) 요구 토크 집중이 발목 롤 활용 부족 때문인가.
읽기 전용. 사용: python3 ankle_roll_liftoff.py  (표준 출력 = 마크다운)

기록 정의(생성기 rl_walking/scripts/eval_transition_kwx_nbc.py 2325–2384 행에서 확인):
  fz[...,0/1]  = 접촉 센서 net_forces_w 의 월드 z, 열 = [left_ankle_r 링크, right_ankle_r 링크]
  cont         = **정책이 본** 접촉 비트(10 Hz 표본 유지·지연·발 평평 조건) → 이륙 검출에 쓰지 않는다. 이륙 = fz 가 5 N 아래로.
  com[...,0/1] = 전신 무게중심의 (앞, 왼) 위치 [m], 원점 = 두 발 링크 원점의 중점, 골반 진행 방향 좌표
  flat[...,0/1]= 발 링크 원점(= 발목 롤 축)의 왼쪽(+) 위치 [m], 원점 = 루트(골반), 열 = [왼발, 오른발]
  tq = applied_torque(한계로 자른 뒤), ptg = joint_pos_target, q·qd = 관절 상태
  요구 토크(자르기 전) = kp_env·(ptg − q) − kd_env·qd  (kp·kd 는 drdump 의 env 별 값 — 명목 150/5, 무작위화 128–165 / 4–6)

부호(URDF 축에서 유도, 자료로 검산 — 아래 '검산' 절):
  골반 프레임 x = 왼, y = 위, z = 앞. hip_a 축 = 양쪽 −z, ankle_r 축 = 오른 −z / 왼 +z.
  '안쪽(+)' = 몸 정중선 쪽. 지지 다리(질량 무시) 모멘트 평형:
     T_hip = T_ank + fz·dx + ΔH·Fx          (T_hip, T_ank: 안쪽+ 규약으로 맞춘 관절 토크, dx = 발목축의 힙축 대비 안쪽 위치,
                                              ΔH = 힙축–발목축 높이차, Fx = 그 발이 받는 지면 횡력(안쪽+))
     T_ank = fz·c + h_a·Fx                  (c = 발목축 기준 CoP 의 안쪽 위치, h_a = 발목 롤 축 높이 0.0536 m)
  오른쪽: T_hip = +tq[R_hip_a], T_ank = +tq[R_ankle_r], dx = flat_R + 0.1435
  왼쪽  : T_hip = −tq[L_hip_a], T_ank = +tq[L_ankle_r], dx = −(flat_L − 0.1435)
"""
import os
import sys
import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
LP = os.path.join(R, "log_picture")
DT = 0.02
PRE, POST = 15, 10                 # −0.30 s … +0.20 s
W0, W1 = -5, 0                     # '이륙 직전 0.1 s' = 틱 −5 … −1
HIP_X = 0.1435                     # hip_a 축의 골반 중심 대비 횡 위치 [m] (URDF 0.071 + 0.0725)
H_ANK = 0.0536                     # 발목 롤 축 → 패드 바닥 [m]
PAD_C, PAD_E = 0.025, 0.038        # 패드 중심 / 패드 바깥 가장자리의 발목축 기준 횡 거리 [m] (패드 반지름 13 mm)
LIM_HIP, LIM_ANK = 25.0, 24.0      # 시뮬 effort_limit_sim (biped12_cfg.py 74–89 행)
G = 9.81
FZ_OFF = 5.0
JN = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint", "left_hip_r_joint", "right_hip_r_joint",
      "left_knee_joint", "right_knee_joint", "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]
POL = [("NBj-s2 (1010 실물에서 모터 8 정지)", "r_nbj2_e10_s1"), ("NBjD-s1", "x_nbjd1_e10_s1"),
       ("NBjE-s1 (현재 젯슨)", "y_nbje1_e10_s1"), ("NBjH-s2", "x_nbjh2_e10_s1")]


def load(tag):
    a = np.load(f"{LP}/transall_{tag}.npz", allow_pickle=True)
    d = np.load(f"{LP}/drdump_{tag}.npz", allow_pickle=True)
    assert list(d["joint_names"]) == JN, "관절 순서 불일치"
    f = lambda k: a[k].astype(np.float32)   # noqa: E731
    D = {k: f(k) for k in ("q", "qd", "ptg", "tq", "fz", "com", "flat", "roll", "h")}
    D["ge"] = a["ge"]; D["alive"] = a["alive"]; D["pp"] = a["pre_pass"]
    D["kp"] = d["stiffness"].astype(np.float32); D["kd"] = d["damping"].astype(np.float32)
    D["dem"] = D["kp"][None] * (D["ptg"] - D["q"]) - D["kd"][None] * D["qd"]
    D["mass"] = float(d["masses"].sum(1).mean())
    return D


def side_series(D, st):
    """지지쪽 st('R' 또는 'L')의 안쪽(+) 규약 시계열."""
    if st == "R":
        jh, ja, fi, sh = 3, 11, 1, +1.0
        dx = D["flat"][..., 1] + HIP_X * np.cos(np.radians(D["roll"]))
        com_med = D["com"][..., 1] - (D["flat"][..., 1] - D["flat"].mean(-1))          # 무게중심의 지지 발목축 대비 안쪽 위치
    else:
        jh, ja, fi, sh = 2, 10, 0, -1.0
        dx = -(D["flat"][..., 0] - HIP_X * np.cos(np.radians(D["roll"])))
        com_med = -(D["com"][..., 1] - (D["flat"][..., 0] - D["flat"].mean(-1)))
    fz = D["fz"][..., fi]; ok = fz > 20.0
    S = {"hip_dem": sh * D["dem"][..., jh], "hip_tq": sh * D["tq"][..., jh], "hip_q": np.degrees(sh * D["q"][..., jh]),
         "ank_dem": D["dem"][..., ja], "ank_tq": D["tq"][..., ja], "ank_q": np.degrees(D["q"][..., ja]),
         "fz": fz, "fz_sw": D["fz"][..., 1 - fi], "dx": dx, "com_med": com_med}
    dH = np.clip(D["h"] - H_ANK, 0.3, 0.7)
    with np.errstate(divide="ignore", invalid="ignore"):
        S["cop_raw"] = np.where(ok, S["ank_tq"] / fz, np.nan)                          # 횡력 무시 추정
        S["Fx"] = (S["hip_tq"] - S["ank_tq"] - fz * dx) / dH                           # 다리 질량 무시 잔차로 본 횡력(안쪽+)
        S["cop"] = np.where(ok, (S["ank_tq"] - H_ANK * S["Fx"]) / fz, np.nan)          # 횡력 보정 추정
    S["t_fzdx"] = fz * dx; S["t_Fx"] = dH * S["Fx"]
    return S, fi


def events(D, st):
    """스윙 발(st 의 반대쪽) 이륙 틱: fz_sw 가 3 틱 이상 > 5 N 이다가 3 틱 이상 ≤ 5 N. 창 전체가 걷기 게이트 켬·생존·유효 env."""
    S, fi = side_series(D, st)
    on = S["fz_sw"] > FZ_OFF; T, N = on.shape; ev = []
    for e in np.where(D["pp"])[0]:
        o = on[:, e]
        idx = np.where(o[2:-3] & o[1:-4] & o[:-5] & ~o[3:-2] & ~o[4:-1] & ~o[5:])[0] + 3
        for i in idx:
            if i - PRE < 0 or i + POST >= T: continue
            sl = slice(i - PRE, i + POST + 1)
            if D["ge"][sl, e].all() and D["alive"][sl, e].all():
                ev.append((e, i))
    return S, ev


def stack(S, ev, k):
    return np.array([S[k][i - PRE:i + POST + 1, e] for e, i in ev], dtype=np.float32)


def pct(x, axis=0):
    return np.nanpercentile(x, [50, 5, 95], axis=axis)


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


def analyze(name, tag, verbose):
    D = load(tag); out = {"name": name, "tag": tag}
    for st in ("R", "L"):
        S, ev = events(D, st)
        W = {k: stack(S, ev, k) for k in ("hip_dem", "hip_tq", "hip_q", "ank_dem", "ank_tq", "ank_q", "fz", "fz_sw", "cop_raw", "cop", "Fx", "t_fzdx", "t_Fx", "dx", "com_med")}
        env = np.array([e for e, _ in ev])
        r = np.arange(len(ev)); res = {"n": len(ev), "nenv": len(set(env.tolist())), "W": W, "env": env}
        for key, (a0, a1) in (("pk", (W0, W1)), ("pk2", (W0, 6))):                     # pk: 틱 −5 … −1(이륙 직전) / pk2: 틱 −5 … +5(솟음 꼭대기 포함)
            w = slice(PRE + a0, PRE + a1)
            ipk = np.argmax(W["hip_dem"][:, w], 1)                                     # 사건별: 창 안 힙 요구 최대 틱에서의 값
            d = {k: W[k][:, w][r, ipk] for k in W}; d["tick"] = ipk + a0
            # CoP 여유: 힙 토크(+)를 줄이는 방향 = CoP 를 바깥(−)으로. 바깥 여유 = c + edge, 안쪽 여유 = edge − c
            d["room_c"] = d["cop"] + PAD_C; d["room_e"] = d["cop"] + PAD_E
            d["gain_c"] = d["fz"] * np.clip(d["room_c"], 0, None); d["gain_e"] = d["fz"] * np.clip(d["room_e"], 0, None)
            res[key] = d
        out[st] = res
        if verbose:
            print(f"\n#### {name} — 지지 = {'오른' if st == 'R' else '왼'}발 (스윙발 이륙 사건 {len(ev)} 개, env {res['nenv']} 개)\n")
            print("이륙 기준 시각별 중앙값 [p5, p95]. 토크·CoP 는 안쪽(+) 규약(오른쪽은 기록 부호 그대로).\n")
            ks = [PRE + k for k in (-15, -10, -7, -5, -4, -3, -2, -1, 0, 2, 5, 10)]
            print("| 항목 | " + " | ".join(f"{(k - PRE) * DT:+.2f}" for k in ks) + " |"); print("|---|" + "---|" * len(ks))
            rows = [("힙 외전 요구 N·m", "hip_dem", 1, "{:.0f}"), ("힙 외전 실제 N·m", "hip_tq", 1, "{:.0f}"), ("힙 외전 각 °", "hip_q", 1, "{:+.1f}"),
                    ("발목 롤 요구 N·m", "ank_dem", 1, "{:+.1f}"), ("발목 롤 실제 N·m", "ank_tq", 1, "{:+.1f}"), ("발목 롤 각 °", "ank_q", 1, "{:+.1f}"),
                    ("지지발 fz N", "fz", 1, "{:.0f}"), ("스윙발 fz N", "fz_sw", 1, "{:.0f}"),
                    ("CoP 안쪽+ mm(횡력 무시)", "cop_raw", 1000, "{:+.0f}"), ("CoP 안쪽+ mm(횡력 보정)", "cop", 1000, "{:+.0f}"),
                    ("지면 횡력 Fx 안쪽+ N(잔차)", "Fx", 1, "{:+.0f}"), ("fz·dx N·m", "t_fzdx", 1, "{:+.1f}"), ("ΔH·Fx N·m", "t_Fx", 1, "{:+.1f}"),
                    ("무게중심 안쪽+ mm(지지 발목축 기준)", "com_med", 1000, "{:+.0f}")]
            for lab, k, sc, fm in rows:
                p = pct(W[k] * sc)
                print(f"| {lab} | " + " | ".join(f"{fm.format(p[0][i])} [{fm.format(p[1][i])}, {fm.format(p[2][i])}]" for i in ks) + " |")
    return D, out


def summary_row(o, st, key="pk"):
    r = o[st]; pk = r[key]; m = lambda x, s=1: np.nanmedian(x) * s   # noqa: E731
    q = lambda x, p, s=1: np.nanpercentile(x, p) * s                   # noqa: E731
    return (f"| {o['name']} | {st} | {r['n']} | {m(pk['tick']) * DT:+.2f} | {m(pk['hip_dem']):.0f} / {q(pk['hip_dem'], 95):.0f} | {m(pk['hip_tq']):.0f} | {m(pk['hip_q']):+.1f} | "
            f"{m(pk['ank_dem']):+.1f} | {m(pk['ank_tq']):+.1f} [{q(pk['ank_tq'], 5):+.1f}, {q(pk['ank_tq'], 95):+.1f}] | {100 * q(np.abs(pk['ank_tq']), 95) / LIM_ANK:.0f} | {m(pk['ank_q']):+.1f} | {m(pk['fz']):.0f} / {m(pk['fz_sw']):.0f} | "
            f"{m(pk['cop'], 1e3):+.0f} [{q(pk['cop'], 5, 1e3):+.0f}, {q(pk['cop'], 95, 1e3):+.0f}] | {100 * np.nanmean(pk['cop'] > PAD_C):.0f} / {100 * np.nanmean(pk['cop'] < -PAD_C):.0f} | "
            f"{(PAD_E - m(pk['cop'])) * 1e3:.0f} ({100 * (PAD_E - m(pk['cop'])) / (2 * PAD_E):.0f} %) | {m(pk['room_e'], 1e3):.0f} ({100 * m(pk['room_e']) / (2 * PAD_E):.0f} %) | "
            f"{m(pk['gain_c']):.1f} / {m(pk['gain_e']):.1f} | {100 * m(pk['gain_e'] / np.clip(pk['hip_dem'], 1e-6, None)):.0f} |")


def decomp_row(o, st, key="pk"):
    pk = o[st][key]; m = np.nanmedian
    return (f"| {o['name']} | {st} | {m(pk['hip_tq']):.1f} | {m(pk['ank_tq']):+.1f} | {m(pk['t_fzdx']):+.1f} | {m(pk['t_Fx']):+.1f} | {m(pk['Fx']):+.0f} | {m(pk['dx']) * 1e3:+.0f} | "
            f"{100 * m(pk['ank_tq'] / np.clip(pk['hip_tq'], 1e-6, None)):.0f} | {100 * m(pk['t_Fx'] / np.clip(pk['hip_tq'], 1e-6, None)):.0f} |")


def cross(o, st):
    r = o[st]; pk = r["pk"]; y = pk["hip_dem"]; env = r["env"]; lines = []
    for lab, x in (("발목 롤 실제 토크(안쪽+)", pk["ank_tq"]), ("CoP 안쪽+ (보정)", pk["cop"]), ("지면 횡력 Fx(잔차)", pk["Fx"]), ("fz·dx", pk["t_fzdx"]),
                   ("무게중심 안쪽 거리", pk["com_med"]), ("지지발 fz", pk["fz"])):
        rho, n = spearman(x, y)
        # env 안 비교(게인·마찰 등 env 고정 무작위화 제거): env 평균을 뺀 뒤 상관
        xd = x.copy(); yd = y.copy()
        for e in np.unique(env):
            mk = env == e; xd[mk] -= np.nanmean(x[mk]); yd[mk] -= np.nanmean(y[mk])
        rho2, _ = spearman(xd, yd)
        ok = np.isfinite(x); t = np.nanpercentile(x[ok], [33.3, 66.7])
        med = [np.median(y[ok & (x <= t[0])]), np.median(y[ok & (x > t[0]) & (x <= t[1])]), np.median(y[ok & (x > t[1])])]
        lines.append(f"| {o['name']} | {st} | {lab} | {rho:+.2f} | {rho2:+.2f} | {med[0]:.0f} / {med[1]:.0f} / {med[2]:.0f} | {n} |")
    return lines


def checks(D, name):
    """부호·식 검산: (1) 자른 요구 = 기록 토크, (2) 한발 지지에서 CoP 추정 ≈ ZMP(무게중심 − (z/g)·가속), (3) 한발 지지에서 잔차 횡력 ≈ M·a."""
    print(f"\n검산 ({name}):")
    for j, lim in ((3, LIM_HIP), (2, LIM_HIP), (11, LIM_ANK), (10, LIM_ANK)):
        m = D["alive"] & D["pp"][None]
        err = np.abs(np.clip(D["dem"][..., j], -lim, lim) - D["tq"][..., j])[m]
        print(f"- {JN[j]}: |clip(요구, ±{lim:.0f}) − 기록 토크| 중앙 {np.median(err):.2f} · p95 {np.percentile(err, 95):.2f} N·m, 기록 토크 최대 절댓값 {np.abs(D['tq'][..., j][m]).max():.1f}")
    for st in ("R", "L"):
        S, fi = side_series(D, st)
        ss = (S["fz_sw"] < 1.0) & (S["fz"] > 100) & D["ge"] & D["alive"] & D["pp"][None]
        ss[:2] = False; ss[-2:] = False
        ss &= np.roll(ss, 1, 0) & np.roll(ss, -1, 0)
        acc = (np.roll(S["com_med"], -1, 0) - 2 * S["com_med"] + np.roll(S["com_med"], 1, 0)) / DT ** 2      # float16 기록이라 잡음 큼 → 중앙값으로만 본다
        zc = np.clip(D["h"] - 0.03, 0.3, 0.8)
        zmp = S["com_med"] - zc / G * acc
        a, b, c = S["cop"][ss], S["cop_raw"][ss], S["com_med"][ss]
        print(f"- 지지 {st} 한발 지지 틱 {int(ss.sum())}: CoP 보정 중앙 {1e3 * np.nanmedian(a):+.1f} mm · CoP 횡력무시 {1e3 * np.nanmedian(b):+.1f} mm · 무게중심 안쪽 {1e3 * np.median(c):+.1f} mm · "
              f"ZMP(무게중심−z/g·a) 중앙 {1e3 * np.median(zmp[ss]):+.1f} mm · 상관(CoP 보정, 무게중심) {np.corrcoef(a[np.isfinite(a)], c[np.isfinite(a)])[0, 1]:+.2f} · "
              f"잔차 횡력 Fx 중앙 {np.median(S['Fx'][ss]):+.1f} N vs M·a 중앙 {D['mass'] * np.median(acc[ss]):+.1f} N · 힙 실제 토크 중앙 {np.median(S['hip_tq'][ss]):+.1f} · 발목 롤 실제 {np.median(S['ank_tq'][ss]):+.1f} N·m")


def static_calc():
    import xml.etree.ElementTree as ET
    root = ET.parse(os.path.join(R, "rl_walking/assets/biped12_final3_hiprpla.urdf")).getroot()
    mass = {l.get("name"): float(l.find("inertial/mass").get("value")) for l in root.findall("link") if l.find("inertial") is not None}
    M = sum(mass.values())
    leg = lambda s: sum(v for k, v in mass.items() if k.startswith(s) and not k.endswith("hip_f_joint"))   # noqa: E731
    mR = leg("right"); M_up = M - mR
    # 기준 자세(외전 0)에서 오른 힙 축 위쪽 질량(골반·센서·양쪽 hip_f 링크·왼다리)의 횡 위치: 골반 중심 0, 왼다리 ≈ 왼 힙 축 +0.1435, hip_f 링크 ±0.06
    x_up = (leg("left") * HIP_X + mass["left_hip_f_joint"] * 0.060 - mass["right_hip_f_joint"] * 0.060) / M_up       # 골반 중심 대비 왼(+)
    d_up = x_up + HIP_X                                                                                              # 오른 힙 축 대비
    print("\n### 정적 계산 (URDF 질량·기하, 기준 자세 근사)\n")
    print(f"- 전체 {M:.2f} kg · 한쪽 다리(hip_a 링크 아래) {mR:.2f} kg · 오른 힙 외전 축 위쪽 질량 M_up {M_up:.2f} kg, 그 무게중심은 오른 힙 축에서 안쪽 {1e3 * d_up:.0f} mm(골반 수평·외전 0 근사).")
    print(f"- 왼발이 완전히 떴는데 골반이 전혀 안 넘어온 경우(최악)의 정적 힙 외전 토크 = M_up·g·d = {M_up * G * d_up:.1f} N·m.")
    print(f"- 무게중심이 지지 발목축 바로 위(지지 다리 수직, CoP = 발목축)일 때 = 0 에 가깝다. CoP 가 발목축에서 c 만큼 안쪽이면 힙 = 발목 = M·g·c: 패드 중심 {PAD_C * 1e3:.0f} mm → {M * G * PAD_C:.1f} N·m, 패드 가장자리 {PAD_E * 1e3:.0f} mm → {M * G * PAD_E:.1f} N·m.")
    print(f"- 발목 롤이 CoP 이동만으로 바꿀 수 있는 힙 토크의 상한 = fz·(발 폭) : 전 체중 {M * G:.0f} N × 패드 중심 간 {2 * PAD_C * 1e3:.0f} mm = {M * G * 2 * PAD_C:.1f} N·m, 패드 바깥 가장자리 간 {2 * PAD_E * 1e3:.0f} mm = {M * G * 2 * PAD_E:.1f} N·m (한쪽 끝에서 반대쪽 끝까지 옮길 때의 최대 폭).")
    return M, M_up, d_up


if __name__ == "__main__":
    print("# 이륙 직전 힙 외전·발목 롤·CoP 동시 분석 (시뮬 기록, 256 env 합성 러너 왕복 45 s)\n")
    print(f"자료: log_picture/transall_<tag>.npz · 이륙 = 스윙발 fz 가 {FZ_OFF:.0f} N 아래로(3 틱 유지) · 창 −0.30 … +0.20 s 전체가 걷기 게이트 켬·생존·유효 env.")
    print(f"'이륙 직전 0.1 s' 표의 값 = 틱 −5 … −1 중 **힙 외전 요구가 최대인 틱**에서 읽은 사건별 값의 중앙값.\n")
    OUT = []; DS = {}
    print("## 1. 파형\n")
    for name, tag in POL:
        if not os.path.exists(f"{LP}/transall_{tag}.npz"):
            print(f"({tag} 없음)"); continue
        D, o = analyze(name, tag, verbose=tag in ("y_nbje1_e10_s1", "r_nbj2_e10_s1")); OUT.append(o); DS[tag] = (name, D)
    HDR = ("| 정책 | 지지 | 사건 | 읽은 시각 s | 힙 요구 중앙 / p95 | 힙 실제 | 힙 각 ° | 발목 롤 요구 | 발목 롤 실제 [p5, p95] | 발목 롤 abs p95 의 모터 한계(24) 대비 % | 발목 롤 각 ° | fz 지지 / 스윙 N | CoP 안쪽+ mm [p5, p95] | "
           "CoP 가 패드 중심 밖인 사건 % (안쪽 / 바깥) | 안쪽 가장자리까지 mm (발 폭 76 대비) | 바깥 가장자리까지 mm (발 폭 대비) | CoP 를 바깥 끝까지 옮기면 줄어드는 힙 토크 N·m (패드 중심 / 가장자리) | 힙 요구 대비 % (가장자리) |\n" + "|---|" * 1 + "---|" * 17)
    DH = "| 정책 | 지지 | 힙 실제 N·m | 발목 롤 N·m | fz·dx N·m | ΔH·Fx N·m | Fx N | dx mm | 발목 롤 몫 % | 횡력 몫 % |\n|---|---|---|---|---|---|---|---|---|---|"
    for key, ttl in (("pk", "2. 이륙 직전 0.1 s (틱 −5 … −1 중 힙 요구 최대 틱)"), ("pk2", "2b. 솟음 꼭대기 (틱 −5 … +5 중 힙 요구 최대 틱 — 이륙 뒤 0.1 s 까지 포함)")):
        print(f"\n## {ttl} — 정책·좌우 비교\n"); print(HDR)
        for o in OUT:
            for st in ("R", "L"): print(summary_row(o, st, key))
        print("\n힙 외전 실제 토크의 분해 (같은 틱, 중앙값) — T_hip = T_ank + fz·dx + ΔH·Fx\n"); print(DH)
        for o in OUT:
            for st in ("R", "L"): print(decomp_row(o, st, key))
    print("\n## 3. 사건 간 횡단 비교 (상관 — 인과 아님)\n")
    print("y = 사건별 힙 외전 요구(틱 −5 … −1 최대). x = 같은 틱의 값. 3 분위 = x 가 작은/중간/큰 사건의 힙 요구 중앙값.\n")
    print("| 정책 | 지지 | x | 스피어만 ρ | env 안 ρ | 힙 요구 중앙 (x 하/중/상) | n |"); print("|---|---|---|---|---|---|---|")
    for o in OUT:
        if o["tag"] in ("y_nbje1_e10_s1", "r_nbj2_e10_s1"):
            for st in ("R", "L"):
                for ln in cross(o, st): print(ln)
    static_calc()
    print("\n## 검산\n")
    for tag in ("y_nbje1_e10_s1", "r_nbj2_e10_s1"):
        if tag in DS: checks(DS[tag][1], DS[tag][0])
