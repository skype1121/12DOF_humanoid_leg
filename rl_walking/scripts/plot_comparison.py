"""FSM vs RL 비교 차트 — 발표용 핵심 증거.

비교 대상 (전부 같은 로봇, 같은 시뮬):
- FSM @ kd25 (구 검증 게인): 걷지만 실물 이식 불가 (kd 상한 5)
- FSM @ kp150/kd5 (실물 게인): 4.6s 전도, 0보 — 게인 정합 실험의 실패 기록
- RL @ kp150/kd5 (실물 게인): 학습으로 해결 — 무낙상 0.5m/s

실행: /home/ryu/IsaacLab/_isaac_sim/python.sh rl_walking/scripts/plot_comparison.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 한국어 폰트 — 파일 직접 등록 (matplotlib 캐시가 시스템 폰트를 못 볼 수 있음)
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT):
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

REPO = "/home/ryu/humanoid_leg_test1"
OUT = os.path.join(REPO, "log_picture", "10_FSM_vs_RL_비교.png")

# 데이터 로드
rl = np.load(os.path.join(REPO, "demo_output", "rl_eval_traj_run1.npz"), allow_pickle=True)
root = rl["root"]            # [N,3] (학습 프레임: +X 전방)
pg = rl["pg"]                # [N,3]
contact = rl["contact"]      # [N,2] (L,R)
t_rl = np.arange(len(root)) / 50.0

fsm25 = json.load(open(os.path.join(REPO, "demo_output", "telemetry", "telem_cycle_walk2.json")))
fsm5 = json.load(open(os.path.join(REPO, "demo_output", "telemetry", "telem_kp150kd5_w4.json")))


def fsm_series(d):
    rows = d["rows"]
    t = np.array([r["t"] for r in rows])
    fwd = -(np.array([r["pelvis_y"] for r in rows]) - rows[0]["pelvis_y"])  # 전방=-Y
    lean = np.hypot(np.array([r["lean_ap"] for r in rows]),
                    np.array([r["lean_lat"] for r in rows]))
    return t, fwd, lean, d["metrics"].get("fell_at")


t25, f25, l25, fell25 = fsm_series(fsm25)
t5, f5, l5, fell5 = fsm_series(fsm5)
fwd_rl = root[:, 0] - root[0, 0]
lean_rl = np.degrees(np.arccos(np.clip(-pg[:, 2], -1, 1)))

fig = plt.figure(figsize=(14, 9))
fig.suptitle("4상 FSM vs RL 정책 — 같은 로봇, 같은 시뮬레이터", fontsize=15, fontweight="bold")

# A: 전진 거리
ax = plt.subplot(2, 2, 1)
ax.plot(t_rl, fwd_rl, lw=2, color="tab:blue",
        label="RL @ kp150/kd5 (실물 게인) — 0.5m/s 명령")
ax.plot(t25, f25, lw=1.6, color="tab:green", label="FSM @ kd25 (이식 불가 게인)")
ax.plot(t5, f5, lw=1.6, color="tab:red", label="FSM @ kp150/kd5 (실물 게인)")
if fell5:
    ax.scatter([t5[-1]], [f5[-1]], marker="x", s=120, color="tab:red", zorder=5)
    ax.annotate(f"전도 ({fell5}s, 0보)", (t5[-1], f5[-1]),
                textcoords="offset points", xytext=(8, -12), color="tab:red")
ax.set_xlabel("시간 [s]")
ax.set_ylabel("전진 거리 [m]")
ax.set_title("전진 거리 — 실물 게인(kd5)에서 FSM은 전도, RL은 주행")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# B: 기울기
ax = plt.subplot(2, 2, 2)
ax.plot(t_rl, lean_rl, lw=1.2, color="tab:blue", label="RL @ kd5")
ax.plot(t25, l25, lw=1.2, color="tab:green", label="FSM @ kd25")
ax.plot(t5, l5, lw=1.2, color="tab:red", label="FSM @ kd5")
ax.axhline(30, ls="--", c="gray", lw=0.8)
ax.text(0.2, 30.6, "낙상 판정 30°", fontsize=8, color="gray")
ax.set_xlabel("시간 [s]")
ax.set_ylabel("몸통 기울기 [deg]")
ax.set_title("직립도 — RL 평균 2.3° (FSM 피크 15~18°)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# C: RL 게이트 다이어그램 (5~10s 구간)
ax = plt.subplot(2, 2, 3)
w0, w1 = 250, 500  # 5~10s
seg = contact[w0:w1]
tt = t_rl[w0:w1]
for k, (lab, col) in enumerate([("왼발", "tab:orange"), ("오른발", "tab:purple")]):
    on = seg[:, k].astype(bool)
    ax.fill_between(tt, k + 0.1, k + 0.9, where=on, color=col, alpha=0.75, label=lab + " 접지")
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(["왼발", "오른발"])
ax.set_xlabel("시간 [s]")
ax.set_title("RL 게이트 다이어그램 — 규칙적 좌우 교대 (케이던스 2.0걸음/s = 사람 영역)")
ax.grid(alpha=0.3, axis="x")

# D: 요약 막대
ax = plt.subplot(2, 2, 4)
cats = ["보행 속도\n[m/s]", "직립도(평균)\n[deg]", "무낙상 지속\n[s, 측정상한]"]
fsm25_v = [0.043, 8.0, 60.0]   # 검증 최장 60s 무낙상, 0.68m/16s, lean 수~15
fsm5_v = [0.007, 12.0, 4.6]
rl_v = [0.44, 2.25, 30.0]
x = np.arange(len(cats))
w = 0.26
ax.bar(x - w, fsm25_v, w, color="tab:green", label="FSM @ kd25")
ax.bar(x, fsm5_v, w, color="tab:red", label="FSM @ kd5(실물)")
ax.bar(x + w, rl_v, w, color="tab:blue", label="RL @ kd5(실물)")
for xi, v in zip(x - w, fsm25_v):
    ax.text(xi, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
for xi, v in zip(x, fsm5_v):
    ax.text(xi, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
for xi, v in zip(x + w, rl_v):
    ax.text(xi, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=9)
ax.set_yscale("log")
ax.set_title("요약 (로그 스케일) — RL만 실물 게인에서 보행 성립")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(OUT, dpi=110)
print("saved:", OUT)
