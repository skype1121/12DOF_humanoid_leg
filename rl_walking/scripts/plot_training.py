"""학습 곡선 플롯 — 텐서보드 이벤트 → 4패널 PNG (발표·증거자료용).

실행: /home/ryu/IsaacLab/_isaac_sim/python.sh rl_walking/scripts/plot_training.py \
        [--run 2026-07-19_04-18-49] [--out log_picture/xx.png] [--title "..."]
(Kit 불필요 — tensorboard + matplotlib만 사용)
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 한국어 폰트 — 파일 직접 등록
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT):
    font_manager.fontManager.addfont(_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

REPO = "/home/ryu/humanoid_leg_test1"
LOGROOT = os.path.join(REPO, "logs", "rsl_rl", "biped12_flat")

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=None, help="런 폴더명 (기본: 최신)")
ap.add_argument("--out", default=None)
ap.add_argument("--title", default="")
args = ap.parse_args()

run = args.run or sorted(os.listdir(LOGROOT))[-1]
run_dir = os.path.join(LOGROOT, run)
ev_file = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))[0]
out = args.out or os.path.join(REPO, "log_picture", f"학습곡선_{run}.png")

acc = EventAccumulator(ev_file, size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags()["scalars"]


def series(tag):
    if tag not in tags:
        return [], []
    ev = acc.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]


fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(args.title or f"biped12 RL 학습 곡선 — {run}", fontsize=13)

ax = axes[0][0]
s, v = series("Train/mean_reward")
ax.plot(s, v, lw=1.2)
ax.set_title("Mean reward")
ax.grid(alpha=0.3)

ax = axes[0][1]
s, v = series("Train/mean_episode_length")
ax.plot(s, v, lw=1.2, color="tab:green")
ax.axhline(1000, ls="--", c="gray", lw=0.8)
ax.set_title("Mean episode length (max 1000 = 20s 무낙상)")
ax.grid(alpha=0.3)

ax = axes[1][0]
for tag, lab in [
    ("Episode_Reward/track_lin_vel_xy_exp", "선속도 추종"),
    ("Episode_Reward/track_ang_vel_z_exp", "각속도 추종"),
    ("Episode_Reward/feet_air_time", "에어타임(걸음)"),
]:
    s, v = series(tag)
    ax.plot(s, v, lw=1.1, label=lab)
ax.set_title("과제 보상 (스텝당)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1][1]
for tag, lab in [
    ("Episode_Reward/flat_orientation_l2", "직립도 페널티"),
    ("Episode_Reward/action_rate_l2", "액션레이트"),
    ("Episode_Reward/feet_slide", "발 미끄럼"),
    ("Episode_Reward/undesired_contacts", "이상 접촉"),
]:
    s, v = series(tag)
    ax.plot(s, v, lw=1.1, label=lab)
ax.set_title("걸음새 페널티 (0에 가까울수록 좋음)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

for row in axes:
    for a in row:
        a.set_xlabel("iteration")

plt.tight_layout()
plt.savefig(out, dpi=110)
print("saved:", out)
# 최종 수치 요약
for tag in ("Train/mean_reward", "Train/mean_episode_length",
            "Episode_Reward/track_lin_vel_xy_exp", "Episode_Reward/feet_air_time"):
    s, v = series(tag)
    if v:
        print(f"{tag}: last={v[-1]:.4f}")
