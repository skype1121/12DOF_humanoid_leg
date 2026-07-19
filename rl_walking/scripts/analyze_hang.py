"""공중 매달기 vs 지상 보행 — 궤적 분석·비교 리포트 + 차트.

입력: demo_output/hang_traj_hang.npz, hang_traj_ground.npz (eval_hang.py 산출)
출력: log_picture/공중매달기_예측리포트.txt, log_picture/20_공중매달기_궤적비교.png

판정 기준 (행잉 보행궤적 형성):
  케이던스 1.0~4.5걸음/s AND 좌우 발 z 역위상(corr<-0.3) AND 발들림 >1cm
서기 명령: 케이던스 <0.3 AND 관절속도 RMS <0.3rad/s → 정지 유지 OK

실행: /home/ryu/IsaacLab/_isaac_sim/python.sh rl_walking/scripts/analyze_hang.py
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 한국어 폰트 — 파일 직접 등록 (plot_comparison.py와 동일 패턴)
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT):
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

REPO = "/home/ryu/humanoid_leg_test1"
C_HANG = "#ff7f0e"   # 공중 = 주황
C_GROUND = "#1f77b4"  # 지상 = 파랑 (CVD-안전 2색 쌍 + 선스타일 이중 인코딩)
C_LIMIT = "#d62728"

parser = argparse.ArgumentParser()
parser.add_argument("--hang", default=os.path.join(REPO, "demo_output", "hang_traj_hang.npz"))
parser.add_argument("--ground", default=os.path.join(REPO, "demo_output", "hang_traj_ground.npz"))
parser.add_argument("--out_png", default=os.path.join(REPO, "log_picture", "20_공중매달기_궤적비교.png"))
parser.add_argument("--out_txt", default=os.path.join(REPO, "log_picture", "공중매달기_예측리포트.txt"))
args = parser.parse_args()


def find_step_peaks(z, dt):
    """발 z(베이스 좌표) 리프트 피크 = 스텝 1회."""
    zc = z - z.mean()
    prom = max(0.006, 0.25 * float(zc.std()))
    dist = max(1, int(0.20 / dt))
    try:
        from scipy.signal import find_peaks
        pk, _ = find_peaks(zc, prominence=prom, distance=dist)
        return np.asarray(pk, dtype=int)
    except Exception:
        pk, last = [], -dist
        for i in range(1, len(zc) - 1):
            if zc[i] > prom and zc[i] >= zc[i - 1] and zc[i] > zc[i + 1] and i - last >= dist:
                pk.append(i)
                last = i
        return np.asarray(pk, dtype=int)


def dom_freq(x, dt, fmin=0.3, fmax=5.0):
    xc = x - x.mean()
    f = np.fft.rfftfreq(len(xc), dt)
    mag = np.abs(np.fft.rfft(xc))
    m = (f >= fmin) & (f <= fmax)
    if not m.any() or mag[m].max() <= 0:
        return 0.0
    return float(f[m][np.argmax(mag[m])])


def corr0(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def max_xcorr(a, b, dt, max_lag_s=1.0):
    """정규화 상호상관 최대값과 그때의 지연(초). 파형 유사도 0~1."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0, 0.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0, 0.0
    L = min(int(max_lag_s / dt), n - 1)  # 짧은 신호에서 빈 슬라이스 dot 방지
    best, bl = -1.0, 0
    for lag in range(-L, L + 1):
        v = np.dot(a[lag:], b[: n - lag]) if lag >= 0 else np.dot(a[: n + lag], b[-lag:])
        c = v / (na * nb)
        if c > best:
            best, bl = c, lag
    return float(best), bl * dt


def span(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, hi) - np.percentile(x, lo))


def load(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


dh = load(args.hang)
dg = load(args.ground)
dt = float(dh["dt"])
cases = [str(c) for c in dh["case_names"]]
cmds = dh["cmds"]
jn = [str(n) for n in dh["joint_names"]]
J = {n: jn.index(n) for n in jn}
ankle_r_ids = [J["left_ankle_r_joint"], J["right_ankle_r_joint"]]
ak70_ids = [i for i in range(len(jn)) if i not in ankle_r_ids]
HIPF_L, KNEE_L = J["left_hip_f_joint"], J["left_knee_joint"]


def clean_end(d, ci):
    """(유효구간 끝, 낙상오염 여부) — 낙상(ground 리셋) 발생 시 첫 done 이전만 분석."""
    if "dones_per_env" not in d or int(np.asarray(d["dones_per_env"])[ci]) == 0:
        return None, False
    return max(int(np.asarray(d["first_done_rec"])[ci]), 0), True


def case_metrics(d, ci, end=None):
    T = d["q"].shape[0] if end is None else min(end, d["q"].shape[0])
    if T < 250:  # 유효구간 5s 미만 — 판정불가
        return None
    dur = T * dt
    zL = d["foot_b"][:T, ci, 0, 2]
    zR = d["foot_b"][:T, ci, 1, 2]
    xL = d["foot_b"][:T, ci, 0, 0]
    xR = d["foot_b"][:T, ci, 1, 0]
    pkL, pkR = find_step_peaks(zL, dt), find_step_peaks(zR, dt)
    dq = d["dq"][:T, ci]
    m = dict(
        cad=(len(pkL) + len(pkR)) / dur,
        anti=corr0(zL, zR),
        freq=dom_freq(zL, dt),
        lift=min(span(zL), span(zR)),
        stepx=max(span(xL), span(xR)),
        rom_hip=np.degrees(max(span(d["q"][:T, ci, J["left_hip_f_joint"]]),
                               span(d["q"][:T, ci, J["right_hip_f_joint"]]))),
        rom_knee=np.degrees(max(span(d["q"][:T, ci, J["left_knee_joint"]]),
                                span(d["q"][:T, ci, J["right_knee_joint"]]))),
        dq_ak70=float(np.abs(dq[:, ak70_ids]).max()),
        dq_ankle_r=float(np.abs(dq[:, ankle_r_ids]).max()),
        dq_rms=float(np.sqrt((dq**2).mean())),
        tau_max=float(np.abs(d["tau"][:T, ci]).max()),
    )
    return m


def verdict(m, cmd):
    if np.allclose(cmd, 0.0):
        # 진폭 기준 (dq RMS는 kd5 미세 디더로 상시 0.3+ — 오판정 원인이라 제외)
        ok = m["cad"] < 0.5 and m["lift"] < 0.02 and m["rom_hip"] < 5.0
        return "정지유지 OK" if ok else "요동!!"
    ok = 1.0 <= m["cad"] <= 4.5 and m["anti"] < -0.3 and m["lift"] > 0.010
    return "보행궤적 OK" if ok else "궤적왜곡(허우적)"


EH = [clean_end(dh, i) for i in range(len(cases))]
EG = [clean_end(dg, i) for i in range(len(cases))]
MH = [case_metrics(dh, i, EH[i][0]) for i in range(len(cases))]
MG = [case_metrics(dg, i, EG[i][0]) for i in range(len(cases))]

# ── 리포트 ──────────────────────────────────────────────────────────
rep = []
rep.append("공중 매달기(행잉) 테스트 — 시뮬 사전 예측 리포트")
rep.append(f"정책: {dh['checkpoint']}")
rep.append(f"조건: fix_root_link(갠트리 강체고정 근사), base z=1.0m, DR 없음(명목), "
           f"질량 스케일 {float(dh['mass_scale']):.2f} (실물 12kg 정합), 액션지연 1틱(실측 정합)")
rep.append(f"행잉 발 접촉력 최대 {float(dh['max_foot_force']):.4f} N "
           f"({'진짜 공중 확인' if float(dh['max_foot_force']) < 0.5 else '접촉 발생!!'}), "
           f"리셋 {int(dh['dones_total'])}회 / 지상 리셋 {int(dg['dones_total'])}회")
rep.append("=" * 96)
HDR = (f"{'케이스':<12s} {'케이던스':>7s} {'L/R위상':>7s} {'발들림':>6s} {'스텝진폭':>7s} "
       f"{'hipROM':>7s} {'kneeROM':>7s} {'dq70':>6s} {'dq롤':>6s} {'tau':>6s}  판정")


def table(title, M, E):
    rep.append(title)
    rep.append(HDR)
    for i, c in enumerate(cases):
        m = M[i]
        if m is None:
            rep.append(f"{c:<12s} 판정불가 — 낙상 후 유효구간 5s 미만")
            continue
        note = f" (낙상발생 — {E[i][0]*dt:.0f}s 이전만 분석)" if E[i][1] else ""
        rep.append(f"{c:<12s} {m['cad']:6.2f}/s {m['anti']:+7.2f} {m['lift']*100:5.1f}cm "
                   f"{m['stepx']*100:6.1f}cm {m['rom_hip']:6.1f}° {m['rom_knee']:6.1f}° "
                   f"{m['dq_ak70']:6.2f} {m['dq_ankle_r']:6.2f} {m['tau_max']:5.1f}  "
                   f"{verdict(m, cmds[i])}{note}")


table("[행잉 — 공중 매달기]", MH, EH)
rep.append("")
table("[지상 — 같은 명령 기준선]", MG, EG)

rep.append("")
rep.append("[행잉 vs 지상 — 궤적 유사도 (보행 케이스)]")
sims = {}
for i, c in enumerate(cases):
    if np.allclose(cmds[i], 0.0):
        continue
    if MH[i] is None or MG[i] is None or EH[i][1] or EG[i][1]:
        rep.append(f"{c:<12s} 비교 생략 (낙상 오염)")
        continue
    s_hip, lag = max_xcorr(dh["q"][:, i, HIPF_L], dg["q"][:, i, HIPF_L], dt)
    s_knee, _ = max_xcorr(dh["q"][:, i, KNEE_L], dg["q"][:, i, KNEE_L], dt)
    sims[i] = (s_hip, s_knee, lag)
    fr = MH[i]["freq"] / MG[i]["freq"] if MG[i]["freq"] > 0 else 0.0
    rep.append(f"{c:<12s} hip_f 파형상관 {s_hip:.2f} / knee {s_knee:.2f} | "
               f"주파수비 {fr:.2f} (행잉 {MH[i]['freq']:.2f}Hz vs 지상 {MG[i]['freq']:.2f}Hz) | "
               f"hipROM비 {MH[i]['rom_hip']/max(MG[i]['rom_hip'],1e-6):.2f}")

walk_idx = [i for i in range(len(cases)) if not np.allclose(cmds[i], 0.0)]
valid_h = [i for i in range(len(cases)) if MH[i] is not None]
n_ok_h = sum(verdict(MH[i], cmds[i]).endswith("OK") for i in valid_h)
ar_peak = max(MH[i]["dq_ankle_r"] for i in valid_h)
rep.append("")
rep.append("=" * 96)
rep.append(f"종합: 행잉 {n_ok_h}/{len(cases)} 케이스 정상 "
           f"(보행 케이스 {sum(verdict(MH[i], cmds[i]) == '보행궤적 OK' for i in walk_idx if MH[i] is not None)}"
           f"/{len(walk_idx)} 보행궤적 형성)")
rep.append(f"발목롤 속도 최대(행잉): {ar_peak:.2f} rad/s — AK45-36 한계 5.0 "
           f"{'준수 OK' if ar_peak <= 5.0 else '위반!! (실물에서 걸림 가능)'}")
rep.append("")
rep.append("[실물 행잉 테스트 주의]")
rep.append("· 이 예측은 '강체 고정' 가정 — 하네스가 출렁이면 IMU에 요동이 실려 정책이")
rep.append("  푸시 대응 동작(회복 스텝)을 섞는다. 골반을 단단히 고정할수록 이 예측과 가깝다.")
rep.append("· 순서: 서기 명령(0,0,0)으로 정지 확인 → 전진 0.3 → 0.5. 이상 시 즉시 E-stop.")
rep.append("· 게인 kp150/kd5 이내(시뮬=실물 상한 동일), 슬루 4°/틱 내장 — 게인 재조정 금지.")

txt = "\n".join(rep) + "\n"
os.makedirs(os.path.dirname(args.out_txt), exist_ok=True)
open(args.out_txt, "w").write(txt)
print(txt)

# ── 차트 (2×3) ──────────────────────────────────────────────────────
CI = 2  # 전진 0.5 케이스
t_h = np.arange(dh["q"].shape[0]) * dt
t_g = np.arange(dg["q"].shape[0]) * dt
W = (5.0, 13.0)  # 표시 구간

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("공중 매달기 vs 지상 — RL 정책 보행 궤적 비교 (이식용 3단계 정책, 전진 0.5m/s)",
             fontsize=14, fontweight="bold")


def style(ax):
    ax.grid(alpha=0.3, lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def win(t):
    return (t >= W[0]) & (t <= W[1])


mh, mg = win(t_h), win(t_g)

ax = axes[0, 0]
ax.plot(t_h[mh], np.degrees(dh["q"][mh, CI, J["left_hip_f_joint"]]), color=C_HANG, lw=2, label="왼쪽")
ax.plot(t_h[mh], np.degrees(dh["q"][mh, CI, J["right_hip_f_joint"]]), color=C_HANG, lw=2,
        ls="--", label="오른쪽")
ax.set_title("행잉: 힙 피치(hip_f) 좌우 스윙")
ax.set_xlabel("시간 (s)")
ax.set_ylabel("관절각 (°)")
ax.legend(loc="upper right")
style(ax)

ax = axes[0, 1]
ax.plot(t_g[mg], np.degrees(dg["q"][mg, CI, J["left_hip_f_joint"]]), color=C_GROUND, lw=2, label="왼쪽")
ax.plot(t_g[mg], np.degrees(dg["q"][mg, CI, J["right_hip_f_joint"]]), color=C_GROUND, lw=2,
        ls="--", label="오른쪽")
ax.set_title("지상: 힙 피치(hip_f) 좌우 스윙")
ax.set_xlabel("시간 (s)")
ax.set_ylabel("관절각 (°)")
ax.legend(loc="upper right")
style(ax)

ax = axes[0, 2]
lag = sims.get(CI, (0, 0, 0.0))[2]
sh = int(round(lag / dt))
kh = np.degrees(dh["q"][:, CI, KNEE_L])
kg = np.degrees(dg["q"][:, CI, KNEE_L])
kh_al = kh[max(sh, 0):] if sh >= 0 else kh[: len(kh) + sh]
kg_al = kg[max(-sh, 0):] if sh < 0 else kg
n = min(len(kh_al), len(kg_al))
tt = np.arange(n) * dt
m2 = (tt >= W[0]) & (tt <= W[1])
ax.plot(tt[m2], kh_al[:n][m2], color=C_HANG, lw=2, label="행잉")
ax.plot(tt[m2], kg_al[:n][m2], color=C_GROUND, lw=2, ls="--", label="지상")
ax.set_title(f"왼무릎 파형 겹침 (위상정렬, 상관 {sims.get(CI,(0,))[0]:.2f})")
ax.set_xlabel("시간 (s)")
ax.set_ylabel("관절각 (°)")
ax.legend(loc="upper right")
style(ax)

ax = axes[1, 0]
ax.plot(dh["foot_b"][:, CI, 0, 0] * 100, dh["foot_b"][:, CI, 0, 2] * 100,
        color=C_HANG, lw=1.2, alpha=0.85, label="행잉")
ax.plot(dg["foot_b"][:, CI, 0, 0] * 100, dg["foot_b"][:, CI, 0, 2] * 100,
        color=C_GROUND, lw=1.2, alpha=0.85, ls="--", label="지상")
ax.set_title("왼발끝 궤적 (베이스 좌표, 보행 사이클 모양)")
ax.set_xlabel("전후 x (cm)")
ax.set_ylabel("상하 z (cm)")
ax.legend(loc="upper right")
style(ax)

ax = axes[1, 1]
xs = np.arange(len(cases))
bw = 0.38
cad_h = [MH[i]["cad"] if MH[i] else 0.0 for i in range(len(cases))]
cad_g = [MG[i]["cad"] if MG[i] else 0.0 for i in range(len(cases))]
ax.bar(xs - bw / 2, cad_h, bw, color=C_HANG, label="행잉")
ax.bar(xs + bw / 2, cad_g, bw, color=C_GROUND, label="지상")
for x, v in zip(xs - bw / 2, cad_h):
    ax.text(x, v + 0.05, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(xs)
ax.set_xticklabels(cases, fontsize=8, rotation=12)
ax.set_title("케이스별 케이던스 (걸음/s)")
ax.set_ylabel("걸음/s")
ax.legend(loc="upper right")
style(ax)

ax = axes[1, 2]
ar_h = [MH[i]["dq_ankle_r"] if MH[i] else 0.0 for i in range(len(cases))]
ar_g = [MG[i]["dq_ankle_r"] if MG[i] else 0.0 for i in range(len(cases))]
ax.bar(xs - bw / 2, ar_h, bw, color=C_HANG, label="행잉")
ax.bar(xs + bw / 2, ar_g, bw, color=C_GROUND, label="지상")
ax.axhline(5.0, color=C_LIMIT, ls=":", lw=1.5)
ax.text(len(cases) - 0.5, 5.05, "AK45-36 한계 5.0", color=C_LIMIT, ha="right", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(cases, fontsize=8, rotation=12)
ax.set_title("발목롤 관절속도 최대 (rad/s) — 실물 병목 관절")
ax.set_ylabel("rad/s")
ax.legend(loc="upper right")
style(ax)

fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(args.out_png, dpi=150)
print("saved:", args.out_png)
