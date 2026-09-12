#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0912 실물 시험 블랙박스 → 시험별 재구성 그림 (PNG).

입력: rl_bridge_<stamp>.jsonl (50 Hz). 사용 키
  timestamp / live / rp_deg=[roll(+왼), pitch(+앞)] / contact=[L,R] / foot_raw_sum=[L,R]
  qpos_rad(12, JOINT_ORDER 좌우 교대) / targets_deg_joint(dict) / gyro_rad_s / hang_state / blend_alpha

시험 1장당 4패널:
  (1) roll / pitch [deg]
  (2) contact L/R 비트 + foot_raw_sum L/R (우축, **접촉 여부 판정용** — 하중·CoP 로 해석 금지)
  (3) 롤 관절(left/right hip_a, ankle_r) 목표 vs 실제 [deg]
  (4) |gyro| [deg/s]
수직선: 첫 접촉 상실 시각(안정창 ≥2 s 뒤 첫 비-[1,1]), 롤 −10° 도달 시각.
추가로 5회 요약 1장: roll 궤적 겹치기(접촉 상실 시각 t=0 정렬).

판정 규약은 jetson/scripts/analyze_correction_chain.py 와 동일하게 맞췄다
(t0 = 첫 live 틱, contact_first_off 안정창 2 s, 관절 deg = qpos_rad × 180/π).

사용:
  python3 plot_trial_recon.py <bb1.jsonl> [bb2.jsonl ...] [--out DIR]
  python3 plot_trial_recon.py --selftest
"""
import argparse
import json
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

# ── 계약 상수 (깨지 말 것) ────────────────────────────────────────────────
JOINT_ORDER = [
    "left_hip_f_joint", "right_hip_f_joint",
    "left_hip_a_joint", "right_hip_a_joint",
    "left_hip_r_joint", "right_hip_r_joint",
    "left_knee_joint", "right_knee_joint",
    "left_ankle_f_joint", "right_ankle_f_joint",
    "left_ankle_r_joint", "right_ankle_r_joint",
]
R2D = 57.29577951308232
ROLL_JOINTS = ["left_hip_a_joint", "right_hip_a_joint",
               "left_ankle_r_joint", "right_ankle_r_joint"]
STABLE_MIN_S = 2.0      # analyze_correction_chain.contact_first_off 와 동일
ROLL_MARK_DEG = -10.0   # 롤 −10° (− = 오른쪽 기울기)

# ── 색 (dataviz 검증 팔레트, light; all-pairs PASS) ───────────────────────
C_BLUE, C_ORANGE, C_AQUA, C_VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SERIES_HUES = [C_BLUE, C_ORANGE, C_AQUA, C_VIOLET]
C_MUTED = "#6f6e69"
C_INK, C_INK2 = "#0b0b0b", "#52514e"
C_EVENT_CONTACT = "#e34948"   # 첫 접촉 상실 (status red)
C_EVENT_ROLL = "#4a3aa7"      # 롤 −10°
C_GRID = "#dcdbd5"

# ── 라벨 (한글 폰트 없으면 영문) ──────────────────────────────────────────
KO = {
    "roll": "roll (+ = 왼쪽) [deg]", "pitch": "pitch (+ = 앞) [deg]",
    "p1y": "몸통 자세 [deg]",
    "cL": "contact L (비트)", "cR": "contact R (비트)",
    "rL": "foot_raw_sum L (우축)", "rR": "foot_raw_sum R (우축)",
    "p2y": "접촉 비트 (0/1)",
    "p2y2": "foot_raw_sum [raw] — 접촉 여부 판정용 (하중·CoP 아님)",
    "p3y": "롤 관절각 [deg]  (실선 = 실제, 파선 = 목표)",
    "p4y": "각속도 크기 [deg/s]",
    "wxy": "|w| roll·pitch 축 (gx, gy)", "w3": "|w| 3축",
    "xl": "t [s]  (첫 live 틱 기준)",
    "vc": "첫 접촉 상실", "vr": "롤 −10° 도달", "vs": "정지 이벤트",
    "band": "접촉 상실 → 롤 −10°",
    "no_roll": "롤 −10° 미도달",
    "no_off": "접촉 상실 없음",
    "note": "패드 카운트(foot_raw_sum)는 접촉 여부 판정용이다. 하중 비율·CoP·체중 지지율로 해석하지 않는다.",
    "sum_title": "0912 실물 5회 — 롤 궤적 겹치기 (t = 0 : 첫 접촉 상실)",
    "sum_x": "t − t(첫 접촉 상실) [s]",
    "sum_y": "roll (+ = 왼쪽) [deg]",
    "tbl": ["시험", "구간 [s]", "첫 접촉상실 [s]", "롤 −10° [s]", "Δ [s]",
            "상실 전 10 s 롤 [deg/s]", "최소 롤 [deg]", "정지"],
    "ctx": "비교 대상 밖 (롤 −10° 미도달)",
}
EN = {
    "roll": "roll (+ = left) [deg]", "pitch": "pitch (+ = fwd) [deg]",
    "p1y": "trunk attitude [deg]",
    "cL": "contact L (bit)", "cR": "contact R (bit)",
    "rL": "foot_raw_sum L (right axis)", "rR": "foot_raw_sum R (right axis)",
    "p2y": "contact bit (0/1)",
    "p2y2": "foot_raw_sum [raw] - contact detection ONLY (not load / not CoP)",
    "p3y": "roll-joint angle [deg]  (solid = actual, dashed = target)",
    "p4y": "angular rate magnitude [deg/s]",
    "wxy": "|w| roll/pitch axes (gx, gy)", "w3": "|w| 3-axis",
    "xl": "t [s]  (from first live tick)",
    "vc": "first contact loss", "vr": "roll reaches -10 deg", "vs": "stop event",
    "band": "contact loss -> roll -10 deg",
    "no_roll": "roll -10 deg not reached",
    "no_off": "no contact loss",
    "note": "foot_raw_sum (pad counts) is for contact detection ONLY - never read as load share / CoP / body-weight support.",
    "sum_title": "Real 0912, 5 trials - roll traces overlaid (t = 0 : first contact loss)",
    "sum_x": "t - t(first contact loss) [s]",
    "sum_y": "roll (+ = left) [deg]",
    "tbl": ["trial", "span [s]", "contact loss [s]", "roll -10 [s]", "d [s]",
            "roll rate, 10 s pre-loss [deg/s]", "min roll [deg]", "stop"],
    "ctx": "context only (roll -10 deg not reached)",
}


def setup_font():
    """한글 글리프가 있는 폰트를 찾으면 rcParams 에 걸고 True, 없으면 False."""
    from matplotlib.ft2font import FT2Font
    prefer = ["Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Nanum Gothic",
              "Malgun Gothic", "AppleGothic", "UnDotum", "Baekmuk Gulim",
              "Source Han Sans KR", "Pretendard", "D2Coding"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name not in have:
            continue
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=name),
                                         fallback_to_default=False)
            cmap = FT2Font(path).get_charmap()
            if 0xAC00 in cmap and 0xD55C in cmap:   # '가', '한'
                plt.rcParams["font.family"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return True
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return False


def style():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": C_GRID, "axes.labelcolor": C_INK2, "text.color": C_INK,
        "xtick.color": C_INK2, "ytick.color": C_INK2,
        "grid.color": C_GRID, "grid.linewidth": 0.6,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.6, "figure.dpi": 140,
    })


# ── 로딩·판정 ────────────────────────────────────────────────────────────
def load_bb(path):
    """(ticks, events). ticks = live 정책 틱(rp_deg 보유), events = {'event': ...} 행."""
    ticks, events = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("event"):
                events.append(d)
            elif d.get("live") and "rp_deg" in d:
                ticks.append(d)
    return ticks, events


def contact_first_off(ticks, min_stable_s=STABLE_MIN_S):
    """양발 접촉이 min_stable_s 이상 이어진 뒤 처음으로 [1,1] 이 깨진 틱 인덱스.
    반환 (i_off, (i_stable0, i_stable1)). analyze_correction_chain.py 와 같은 규약."""
    run0 = None
    for i, t in enumerate(ticks):
        c = tuple(t.get("contact") or (0, 0))
        if c == (1.0, 1.0):
            if run0 is None:
                run0 = i
            if (t["timestamp"] - ticks[run0]["timestamp"]) >= min_stable_s:
                for j in range(i + 1, len(ticks)):
                    if tuple(ticks[j].get("contact") or (0, 0)) != (1.0, 1.0):
                        return j, (run0, j - 1)
                return None, (run0, len(ticks) - 1)
        else:
            run0 = None
    return None, (None, None)


def roll_cross_index(ticks, i_from=0, thr=ROLL_MARK_DEG):
    """i_from 이후 roll 이 thr 에 도달한 첫 인덱스 (thr<0 이면 이하, thr>0 이면 이상)."""
    for j in range(max(0, i_from), len(ticks)):
        r = ticks[j]["rp_deg"][0]
        if (r <= thr) if thr < 0 else (r >= thr):
            return j
    return None


def joint_deg_actual(tick, name):
    """관절 실제각 [deg] = qpos_rad[JOINT_ORDER.index(name)] × 180/π."""
    return tick["qpos_rad"][JOINT_ORDER.index(name)] * R2D


def joint_deg_target(tick, name):
    tg = tick.get("targets_deg_joint") or {}
    v = tg.get(name)
    return float("nan") if v is None else float(v)


def stamp_of(path):
    m = re.search(r"rl_bridge_(\d{8}_\d{6})", os.path.basename(path))
    if m:
        return m.group(1)
    return os.path.splitext(os.path.basename(path))[0]


def summarize(path):
    """한 시험을 그림·표에 필요한 형태로 정리."""
    ticks, events = load_bb(path)
    if not ticks:
        raise ValueError(f"live 정책 틱 없음: {path}")
    t0 = ticks[0]["timestamp"]
    t = [x["timestamp"] - t0 for x in ticks]
    i_off, stable = contact_first_off(ticks)
    i_roll = roll_cross_index(ticks, i_off if i_off is not None else 0)
    roll = [x["rp_deg"][0] for x in ticks]
    pitch = [x["rp_deg"][1] for x in ticks]
    # 상실 전 10 s 롤 속도
    rate = None
    if i_off is not None:
        t_a = t[i_off] - 10.0
        k = next((i for i in range(i_off, -1, -1) if t[i] <= t_a), 0)
        dt = t[i_off] - t[k]
        if dt > 1.0:
            rate = (roll[i_off] - roll[k]) / dt
    stops = [e for e in events if e.get("event") in ("stop_all", "lift_guard")]
    return {
        "path": path, "stamp": stamp_of(path), "ticks": ticks, "events": events,
        "t0": t0, "t": t, "roll": roll, "pitch": pitch,
        "i_off": i_off, "i_roll": i_roll, "stable": stable,
        "t_off": None if i_off is None else t[i_off],
        "t_roll": None if i_roll is None else t[i_roll],
        "rate": rate, "stops": stops,
        "dur": t[-1], "checkpoint": ticks[0].get("checkpoint", "?"),
    }


# ── 그리기 ───────────────────────────────────────────────────────────────
def _event_lines(ax, S, L, label=False):
    if S["t_off"] is not None:
        ax.axvline(S["t_off"], color=C_EVENT_CONTACT, lw=1.6, zorder=1,
                   label=L["vc"] if label else None)
    if S["t_roll"] is not None:
        ax.axvline(S["t_roll"], color=C_EVENT_ROLL, lw=1.6, ls=(0, (5, 3)), zorder=1,
                   label=L["vr"] if label else None)
    if S["t_off"] is not None and S["t_roll"] is not None:
        ax.axvspan(S["t_off"], S["t_roll"], color=C_MUTED, alpha=0.07, lw=0, zorder=0,
                   label=L["band"] if label else None)
    for e in S["stops"]:
        ax.axvline(e["timestamp"] - S["t0"], color=C_MUTED, lw=1.0, ls=":", zorder=1,
                   label=L["vs"] if label else None)
        label = False   # 정지 이벤트는 첫 개만 범례에


def plot_trial(S, out_dir, L):
    t = S["t"]
    ticks = S["ticks"]
    fig, axes = plt.subplots(4, 1, figsize=(13.0, 11.4), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0, 1.15, 0.85],
                                          "hspace": 0.16})
    ax1, ax2, ax3, ax4 = axes

    # (1) roll / pitch
    ax1.plot(t, S["roll"], color=C_BLUE, label=L["roll"], zorder=3)
    ax1.plot(t, S["pitch"], color=C_ORANGE, label=L["pitch"], zorder=3)
    ax1.axhline(0.0, color=C_GRID, lw=0.8)
    ax1.axhline(ROLL_MARK_DEG, color=C_EVENT_ROLL, lw=0.8, ls=(0, (2, 3)), alpha=0.7)
    ax1.set_ylabel(L["p1y"])
    _event_lines(ax1, S, L, label=True)
    ax1.legend(loc="lower left", ncol=3)

    # (2) contact bits + foot_raw_sum (우축; 접촉 여부용)
    cL = [float((x.get("contact") or [0, 0])[0]) for x in ticks]
    cR = [float((x.get("contact") or [0, 0])[1]) for x in ticks]
    ax2.step(t, [v + 0.02 for v in cL], where="post", color=C_BLUE, lw=2.0,
             label=L["cL"], zorder=3)
    ax2.step(t, [v - 0.02 for v in cR], where="post", color=C_ORANGE, lw=2.0,
             label=L["cR"], zorder=3)
    ax2.set_ylim(-0.22, 1.22)
    ax2.set_yticks([0, 1])
    ax2.set_ylabel(L["p2y"])
    axr = ax2.twinx()
    axr.grid(False)
    axr.spines["right"].set_visible(True)
    axr.spines["right"].set_color(C_GRID)
    rL = [float((x.get("foot_raw_sum") or [0, 0])[0]) for x in ticks]
    rR = [float((x.get("foot_raw_sum") or [0, 0])[1]) for x in ticks]
    axr.plot(t, rL, color=C_BLUE, lw=1.0, ls=(0, (4, 2)), alpha=0.85, label=L["rL"], zorder=2)
    axr.plot(t, rR, color=C_ORANGE, lw=1.0, ls=(0, (4, 2)), alpha=0.85, label=L["rR"], zorder=2)
    axr.set_ylabel(L["p2y2"], color=C_INK2, fontsize=8)
    axr.tick_params(axis="y", labelsize=8, colors=C_INK2)
    _event_lines(ax2, S, L)
    h = ax2.get_legend_handles_labels()
    hr = axr.get_legend_handles_labels()
    ax2.legend(h[0] + hr[0], h[1] + hr[1], loc="center left", ncol=4)

    # (3) 롤 관절 목표 vs 실제
    for name, col in zip(ROLL_JOINTS, SERIES_HUES):
        act = [joint_deg_actual(x, name) for x in ticks]
        tgt = [joint_deg_target(x, name) for x in ticks]
        short = name.replace("_joint", "")
        ax3.plot(t, act, color=col, lw=1.6, label=f"{short} act", zorder=3)
        ax3.plot(t, tgt, color=col, lw=1.1, ls=(0, (4, 2)), alpha=0.9,
                 label=f"{short} tgt", zorder=3)
    ax3.axhline(0.0, color=C_GRID, lw=0.8)
    ax3.set_ylabel(L["p3y"])
    _event_lines(ax3, S, L)
    ax3.legend(loc="upper left", ncol=4)

    # (4) |gyro|
    wxy = [math.hypot(x["gyro_rad_s"][0], x["gyro_rad_s"][1]) * R2D for x in ticks]
    w3 = [math.sqrt(sum(g * g for g in x["gyro_rad_s"])) * R2D for x in ticks]
    ax4.plot(t, w3, color=C_MUTED, lw=1.0, alpha=0.8, label=L["w3"], zorder=2)
    ax4.plot(t, wxy, color=C_BLUE, lw=1.6, label=L["wxy"], zorder=3)
    ax4.set_ylabel(L["p4y"])
    ax4.set_xlabel(L["xl"])
    _event_lines(ax4, S, L)
    ax4.legend(loc="upper left", ncol=2)

    for ax in axes:
        ax.set_xlim(t[0], t[-1])

    head = (f"{S['stamp']}  |  checkpoint {S['checkpoint']}  |  live {S['dur']:.1f} s, "
            f"{len(ticks)} ticks")
    sub = []
    sub.append(f"{L['vc']}: " + (f"{S['t_off']:.2f} s" if S["t_off"] is not None else L["no_off"]))
    sub.append(f"{L['vr']}: " + (f"{S['t_roll']:.2f} s" if S["t_roll"] is not None else L["no_roll"]))
    if S["t_off"] is not None and S["t_roll"] is not None:
        sub.append(f"d = {S['t_roll'] - S['t_off']:.2f} s")
    if S["rate"] is not None:
        sub.append(f"{L['tbl'][5]}: {S['rate']:+.2f}")
    if S["stops"]:
        sub.append(f"{L['vs']}: " + ", ".join(
            f"{e['event']}@{e['timestamp'] - S['t0']:.1f}s" for e in S["stops"]))
    fig.suptitle(head, x=0.012, ha="left", y=0.995, fontsize=12, color=C_INK)
    fig.text(0.012, 0.972, "   ·   ".join(sub), ha="left", fontsize=9, color=C_INK2)
    fig.text(0.012, 0.010, L["note"], ha="left", fontsize=8, color=C_INK2)
    fig.subplots_adjust(left=0.072, right=0.905, top=0.948, bottom=0.062, hspace=0.16)

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"trial_{S['stamp']}.png")
    fig.savefig(png)
    plt.close(fig)
    return png


def plot_summary(sums, out_dir, L, name="summary_5trials.png"):
    """roll 궤적 겹치기 — 첫 접촉 상실 시각을 t=0 으로 정렬."""
    fig, (ax, axt) = plt.subplots(2, 1, figsize=(13.0, 8.4),
                                  gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.28})
    # 색 배정: 롤 −10° 를 만든 시험이 비교 대상 → 고정 순서 hue, 그 밖은 muted(순환 금지)
    ev = [S for S in sums if S["i_roll"] is not None]
    colors, styles = {}, {}
    for i, S in enumerate(ev):
        if i < len(SERIES_HUES):
            colors[S["stamp"]] = SERIES_HUES[i]
            styles[S["stamp"]] = "-"
        else:
            colors[S["stamp"]] = C_MUTED
            styles[S["stamp"]] = (0, (1 + i, 2))
    dash = [(0, (5, 2)), (0, (1, 2)), (0, (7, 2, 1, 2))]
    j = 0
    for S in sums:
        if S["stamp"] in colors:
            continue
        colors[S["stamp"]] = C_MUTED
        styles[S["stamp"]] = dash[j % len(dash)]
        j += 1

    for S in sums:
        t_ref = S["t_off"] if S["t_off"] is not None else 0.0
        x = [v - t_ref for v in S["t"]]
        lab = S["stamp"][-6:]
        if S["i_roll"] is None:
            lab += f"  ({L['ctx']})"
        ax.plot(x, S["roll"], color=colors[S["stamp"]], ls=styles[S["stamp"]],
                lw=1.8 if S["i_roll"] is not None else 1.2,
                alpha=1.0 if S["i_roll"] is not None else 0.75, label=lab, zorder=3)
        if S["t_roll"] is not None:
            ax.plot([S["t_roll"] - t_ref], [ROLL_MARK_DEG], marker="o", ms=6,
                    color=colors[S["stamp"]], mec="#fcfcfb", mew=1.2, zorder=4)
    ax.axvline(0.0, color=C_EVENT_CONTACT, lw=1.6, zorder=2, label=L["vc"] + " (t = 0)")
    ax.axhline(ROLL_MARK_DEG, color=C_EVENT_ROLL, lw=1.2, ls=(0, (5, 3)), zorder=2,
               label=L["vr"])
    ax.axhline(0.0, color=C_GRID, lw=0.8, zorder=1)
    ax.set_xlabel(L["sum_x"])
    ax.set_ylabel(L["sum_y"])
    ax.set_title(L["sum_title"], loc="left", fontsize=12, color=C_INK, pad=8)
    ax.legend(loc="lower left", ncol=3)

    axt.axis("off")
    rows = []
    for S in sums:
        f = lambda v, n=2: "-" if v is None else f"{v:.{n}f}"
        rows.append([
            S["stamp"][-6:],
            f"0 ~ {S['dur']:.1f}",
            f(S["t_off"]),
            f(S["t_roll"]) if S["t_roll"] is not None else L["no_roll"],
            "-" if (S["t_off"] is None or S["t_roll"] is None) else f"{S['t_roll'] - S['t_off']:.2f}",
            "-" if S["rate"] is None else f"{S['rate']:+.2f}",
            f"{min(S['roll']):.1f}",
            ", ".join(e["event"] for e in S["stops"]) or "-",
        ])
    tbl = axt.table(cellText=rows, colLabels=L["tbl"], loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(C_GRID)
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_text_props(color=C_INK2)
            cell.set_facecolor("#f2f1ec")
        elif c == 0:
            st = sums[r - 1]["stamp"]
            cell.set_text_props(color=colors[st])
    fig.text(0.012, 0.012, L["note"], ha="left", fontsize=8, color=C_INK2)
    fig.subplots_adjust(left=0.068, right=0.985, top=0.935, bottom=0.055, hspace=0.30)

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, name)
    fig.savefig(png)
    plt.close(fig)
    return png


# ── 셀프테스트 ───────────────────────────────────────────────────────────
def _synth(path, n=600, hz=50.0, off_i=250, roll_from=-0.2, roll_slope=-2.0,
           never_roll=False, t0=1000.0):
    """합성 블랙박스: off_i 틱에서 오른발 접촉 0, 그 뒤 roll 이 선형으로 내려감."""
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(n):
            ts = t0 + i / hz
            if i < off_i:
                roll = roll_from
                c, raw = [1.0, 1.0], [900, 900]
            else:
                s = (i - off_i) / hz
                roll = roll_from + (0.0 if never_roll else roll_slope * s)
                c, raw = [1.0, 0.0], [900, 0]
            row = {
                "timestamp": ts, "live": True, "checkpoint": "selftest",
                "rp_deg": [roll, -1.0], "contact": c, "foot_raw_sum": raw,
                "qpos_rad": [k * 0.001 for k in range(12)],
                "targets_deg_joint": {nm: 0.05 * (k + 1) for k, nm in enumerate(JOINT_ORDER)},
                "gyro_rad_s": [0.01, 0.02, 0.005],
                "hang_state": "CONTACT_ACQUIRED", "blend_alpha": 1.0,
                "tau_est_nm": [0.0] * 12,
            }
            fh.write(json.dumps(row) + "\n")
        fh.write(json.dumps({"timestamp": t0 + (n - 1) / hz, "event": "stop_all",
                             "reason": "selftest"}) + "\n")


def selftest():
    import tempfile
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + msg)
        ok = ok and bool(cond)

    print("# plot_trial_recon --selftest")
    # 0) 계약
    chk(len(JOINT_ORDER) == 12, "JOINT_ORDER 12 개")
    chk(all(JOINT_ORDER[i].startswith("left_") and JOINT_ORDER[i + 1].startswith("right_")
            and JOINT_ORDER[i][5:] == JOINT_ORDER[i + 1][6:] for i in range(0, 12, 2)),
        "JOINT_ORDER 좌우 교대 계약 (lhf,rhf,lha,rha,...)")
    chk([JOINT_ORDER.index(n) for n in ROLL_JOINTS] == [2, 3, 10, 11],
        "롤 관절 인덱스 = hip_a 2/3, ankle_r 10/11")

    d = tempfile.mkdtemp(prefix="plot_trial_recon_")
    a = os.path.join(d, "rl_bridge_20260101_000001.jsonl")
    b = os.path.join(d, "rl_bridge_20260101_000002.jsonl")
    _synth(a, off_i=250, roll_from=-0.2, roll_slope=-2.0)       # 5.0 s 에 상실, 이후 −10° 도달
    _synth(b, off_i=150, never_roll=True)                       # 상실은 있으나 롤 −10° 없음

    Sa, Sb = summarize(a), summarize(b)
    # 1) 판정 시각
    chk(abs(Sa["t_off"] - 5.0) < 0.021, f"첫 접촉 상실 = 5.00 s (측정 {Sa['t_off']:.3f})")
    exp = 5.0 + (10.0 - 0.2) / 2.0                              # roll = −0.2 − 2·s → −10
    chk(Sa["t_roll"] is not None and abs(Sa["t_roll"] - exp) < 0.05,
        f"롤 −10° 도달 = {exp:.2f} s (측정 {Sa['t_roll']})")
    chk(Sb["t_roll"] is None, "롤 −10° 미도달 시험은 None (크래시 없음)")
    chk(abs(Sb["t_off"] - 3.0) < 0.021, f"두 번째 시험 접촉 상실 = 3.00 s (측정 {Sb['t_off']:.3f})")
    # 2) 안정창 2 s 규약: 상실 직전 2 s 이상 [1,1]
    i0, i1 = Sa["stable"]
    chk(i0 == 0 and i1 == 249, f"안정창 인덱스 (0, 249) (측정 {Sa['stable']})")
    # 3) 관절 deg 변환·매핑
    tk = Sa["ticks"][0]
    chk(abs(joint_deg_actual(tk, "left_hip_a_joint") - 0.002 * R2D) < 1e-9,
        "left_hip_a 실제각 = qpos_rad[2]×R2D")
    chk(abs(joint_deg_actual(tk, "right_ankle_r_joint") - 0.011 * R2D) < 1e-9,
        "right_ankle_r 실제각 = qpos_rad[11]×R2D")
    chk(abs(joint_deg_target(tk, "left_hip_a_joint") - 0.15) < 1e-9,
        "left_hip_a 목표각 = targets_deg_joint[name]")
    # 4) 이벤트
    chk(len(Sa["stops"]) == 1 and Sa["stops"][0]["event"] == "stop_all", "정지 이벤트 수집")
    # 5) 산출물 (산출물로 판정)
    L = KO if setup_font() else EN
    style()
    outd = os.path.join(d, "figs")
    pngs = [plot_trial(Sa, outd, L), plot_trial(Sb, outd, L)]
    pngs.append(plot_summary([Sa, Sb], outd, L, name="summary_selftest.png"))
    for p in pngs:
        sz = os.path.getsize(p) if os.path.exists(p) else 0
        chk(sz > 20000, f"PNG 생성 {os.path.basename(p)} ({sz} B)")
        with open(p, "rb") as fh:
            chk(fh.read(8) == b"\x89PNG\r\n\x1a\n", f"PNG 매직 {os.path.basename(p)}")
    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="실물 블랙박스 → 시험별 재구성 그림")
    ap.add_argument("blackbox", nargs="*", help="rl_bridge_<stamp>.jsonl 경로들")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap.add_argument("--out", default=os.path.join(root, "reports", "real_20260912", "figs"),
                    help="PNG 출력 디렉토리")
    ap.add_argument("--summary-name", default="summary_5trials.png")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.blackbox:
        ap.error("블랙박스 경로가 없다 (또는 --selftest)")
    L = KO if setup_font() else EN
    style()
    sums = []
    for p in sorted(a.blackbox):
        S = summarize(p)
        sums.append(S)
        png = plot_trial(S, a.out, L)
        f_off = "-" if S["t_off"] is None else "%.2f s" % S["t_off"]
        f_rol = "-" if S["t_roll"] is None else "%.2f s" % S["t_roll"]
        print("%s: live %.1f s, %d ticks | 첫 접촉 상실 %s | 롤 -10도 %s -> %s"
              % (S["stamp"], S["dur"], len(S["ticks"]), f_off, f_rol, png))
    if len(sums) >= 2:
        print("summary -> " + plot_summary(sums, a.out, L, name=a.summary_name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
