"""학습 단계 3클립 + 보상 곡선 → 1920x1080 영상 프레임 합성 → mp4.

사용: python3 compose_video.py --clips d0,d1,d2 --iters 0,40,4500 --tb g6_tb.json --out out.mp4 [--preview t.png --at 5.0]
곡선 = 실제 학습 기록(Train/mean_reward), 클립 = 해당 반복 시점 체크포인트를 같은 학습 환경에서 재생한 것.
"""
import argparse
import glob
import json
import math
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from style import AMBER, BG, CARD, CARD_LINE, CORAL, DIM, GREEN, GRID, SUB, TXT, font, tw

ap = argparse.ArgumentParser()
ap.add_argument("--clips", required=True)
ap.add_argument("--iters", required=True)
ap.add_argument("--tb", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--loops", type=int, default=4)
ap.add_argument("--max_iter", type=int, default=4500)
ap.add_argument("--states", default="넘어짐,비틀거림,섬")
ap.add_argument("--notes", default="2초를 버티지 못함,버티다가 가끔 쓰러짐,무작위 오차·밀림에도 서 있음")
ap.add_argument("--preview", default="")
ap.add_argument("--at", type=float, default=6.0)
a = ap.parse_args()

W, H = 1920, 1080
COLS = [CORAL, AMBER, GREEN]
ITERS = [int(v) for v in a.iters.split(",")]
STATES = a.states.split(",")
NOTES = a.notes.split(",")
SEC_PER_ITER = 4096 * 24 * 0.02          # 병렬 4096대 x 24스텝 x 0.02초 = 로봇 경험 1966초


def exp_txt(it):
    s = it * SEC_PER_ITER
    if s <= 0:
        return "로봇 경험 0"
    if s < 3 * 86400:
        return f"로봇 경험 약 {s / 3600:.0f}시간"
    return f"로봇 경험 약 {s / 86400:.0f}일"


# ---------------------------------------------------------------- 클립
clips = []
for dname in a.clips.split(","):
    fr = sorted(glob.glob(os.path.join(dname, "f*.png")))
    assert fr, dname
    clips.append(fr)
NF = min(len(c) for c in clips)
PW, PH = 600, 400
PX = [40 + i * (PW + 20) for i in range(3)]
PY = 196
CARD_Y0, CARD_Y1 = 132, PY + PH + 8

# ---------------------------------------------------------------- 곡선 데이터
tb = json.load(open(a.tb))["data"]
raw = np.array(tb["Train/mean_reward"])[: a.max_iter + 1, 1]
n = len(raw)
sm = np.empty(n)
for i in range(n):
    w = int(min(40, max(0, i // 8)))
    sm[i] = raw[max(0, i - w): i + w + 1].mean()

CX0, CX1, CY0, CY1 = 118, 1872, 716, 1004      # 그림 영역
YMAX = 160.0


def xpix(it):
    return CX0 + (CX1 - CX0) * math.sqrt(max(0.0, it) / a.max_iter)


def ypix(v):
    return CY1 - (CY1 - CY0) * (max(0.0, v) / YMAX)


S = 2


def P(v):
    return int(round(v * S))


# ---------------------------------------------------------------- 정적 배경(제목·카드·축·옅은 곡선)
bg = Image.new("RGB", (W * S, H * S), BG)
d = ImageDraw.Draw(bg)


def text(x, y, s, f, fill, anchor="la", dr=None):
    (dr or d).text((P(x), P(y)), s, font=f, fill=fill, anchor=anchor)


text(40, 46, "강화학습으로 ‘서는 법’을 배우는 과정", font(P(44), "Bold"), TXT, anchor="lm")
text(W - 40, 48, "Isaac Lab 시뮬레이션  ·  12관절 하체 로봇  ·  PPO", font(P(22), "Regular"), DIM, anchor="rm")
text(40, 102, "같은 신경망 제어기를 학습 도중 세 시점에서 꺼내, 학습에 쓴 것과 같은 시뮬레이션 환경(무작위 오차·밀림 포함)에 넣은 모습", font(P(22), "Regular"), SUB, anchor="lm")

f_hd = font(P(28), "Bold")
f_hs = font(P(21), "Regular")
f_nb = font(P(21), "Bold")
for i in range(3):
    x0 = PX[i]
    d.rounded_rectangle((P(x0 - 6), P(CARD_Y0), P(x0 + PW + 6), P(CARD_Y1)), radius=P(14), fill=CARD, outline=CARD_LINE, width=P(1))
    cx, cy = x0 + 22, CARD_Y0 + 32
    d.ellipse((P(cx - 16), P(cy - 16), P(cx + 16), P(cy + 16)), fill=COLS[i])
    text(cx, cy - 1, str(i + 1), f_nb, BG, anchor="mm")
    text(x0 + 50, cy - 1, f"학습 {ITERS[i]:,}회", f_hd, COLS[i], anchor="lm")
    text(x0 + PW - 8, cy + 1, exp_txt(ITERS[i]), f_hs, SUB, anchor="rm")

# 곡선 영역
text(40, 652, "목적함수(보상) 값", font(P(27), "Bold"), TXT, anchor="lm")
tx = 40 + tw(d, "목적함수(보상) 값", font(P(27), "Bold")) / S + 14
text(tx, 654, "— 학습이 진행될수록 커진다", font(P(22), "Regular"), SUB, anchor="lm")
text(W - 40, 654, "로봇 4,096대 동시 학습  ·  4,500회 = 로봇 경험 약 102일  ·  실제 계산 약 2.5시간 (GPU 1대)", font(P(21), "Regular"), SUB, anchor="rm")
f_tk = font(P(18), "Regular")
for v in (0, 40, 80, 120, 160):
    y = ypix(v)
    d.line((P(CX0), P(y), P(CX1), P(y)), fill=GRID, width=P(1))
    text(CX0 - 12, y, str(v), f_tk, DIM, anchor="rm")
for it in (0, 100, 500, 1000, 2000, 3000, 4500):
    x = xpix(it)
    d.line((P(x), P(CY1), P(x), P(CY1 + 6)), fill=DIM, width=P(1))
    text(x, CY1 + 12, f"{it:,}", f_tk, DIM, anchor="ma")
text((CX0 + CX1) / 2, CY1 + 40, "학습 반복 횟수  (초반이 잘 보이도록 가로축은 제곱근 눈금)", font(P(19), "Regular"), DIM, anchor="ma")
text(CX0 - 78, (CY0 + CY1) / 2 - 14, "보상", f_tk, DIM, anchor="lm")
text(CX0 - 78, (CY0 + CY1) / 2 + 12, "합계", f_tk, DIM, anchor="lm")
# 옅은 원곡선(항상 보임)
pts_raw = [(P(xpix(i)), P(ypix(raw[i]))) for i in range(n)]
d.line(pts_raw, fill=(52, 62, 78), width=P(1.5), joint="curve")
bg_small = bg.resize((W, H), Image.LANCZOS)

# 밝은 곡선 레이어(와이프로 드러냄) + 아래 채움
curve = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
dc = ImageDraw.Draw(curve)
pts = [(P(xpix(i)), P(ypix(sm[i]))) for i in range(n)]
dc.polygon(pts + [(P(CX1), P(CY1)), (P(CX0), P(CY1))], fill=(96, 165, 250, 34))
dc.line(pts, fill=(134, 190, 255, 255), width=P(3.5), joint="curve")
curve_small = curve.resize((W, H), Image.LANCZOS)

# 표식(단계별)
f_mk = font(P(20), "Bold")
markers = []
for i, it in enumerate(ITERS):
    mx, my = xpix(it), ypix(sm[min(it, n - 1)])
    lay = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    dm = ImageDraw.Draw(lay)
    dm.ellipse((P(mx - 15), P(my - 15), P(mx + 15), P(my + 15)), fill=COLS[i], outline=BG, width=P(3))
    text(mx, my - 1, str(i + 1), f_mk, BG, anchor="mm", dr=dm)
    lab = f"{it:,}회 · 보상 {sm[min(it, n - 1)]:.0f}"
    lw = tw(dm, lab, f_mk) / S
    if i == 2:
        lx, ly, anc = mx - 24, my + 34, "rm"
    elif i == 0:
        lx, ly, anc = mx + 4, my - 36, "lm"
    else:
        lx, ly, anc = mx + 24, my + 4, "lm"
    text(lx, ly, lab, f_mk, COLS[i], anchor=anc, dr=dm)
    markers.append((mx, my, lay.resize((W, H), Image.LANCZOS)))

# 상태 라벨(영상 위 왼쪽 아래)
f_st = font(P(34), "Bold")
f_nt = font(P(20), "Medium")
labels = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
dl = ImageDraw.Draw(labels)
for i in range(3):
    x0, y0 = PX[i], PY
    sw = tw(dl, STATES[i], f_st) / S
    nw = tw(dl, NOTES[i], f_nt) / S
    bw = 18 + sw + 14 + nw + 18
    dl.rounded_rectangle((P(x0 + 12), P(y0 + 12), P(x0 + 12 + bw), P(y0 + 62)), radius=P(12), fill=(10, 13, 18, 205))
    text(x0 + 30, y0 + 36, STATES[i], f_st, COLS[i], anchor="lm", dr=dl)
    text(x0 + 30 + sw + 14, y0 + 38, NOTES[i], f_nt, TXT, anchor="lm", dr=dl)
labels_small = labels.resize((W, H), Image.LANCZOS)

# 영상 모서리 마스크
pmask = Image.new("L", (PW * S, PH * S), 0)
ImageDraw.Draw(pmask).rounded_rectangle((0, 0, PW * S - 1, PH * S - 1), radius=P(10), fill=255)
pmask = pmask.resize((PW, PH), Image.LANCZOS)

T_DRAW = 3.2      # 곡선 그려지는 시간 [s]


def ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def frame(k):
    t = k / a.fps
    im = bg_small.copy()
    # 영상
    fade = min(1.0, (k + 1) / 8.0, (NF - k) / 8.0)
    for i in range(3):
        fr = Image.open(clips[i][k]).convert("RGB").resize((PW, PH), Image.LANCZOS)
        if fade < 1.0:
            fr = Image.blend(Image.new("RGB", (PW, PH), CARD), fr, 0.25 + 0.75 * fade)
        im.paste(fr, (PX[i], PY), pmask)
    im.paste(labels_small, (0, 0), labels_small)
    # 곡선 와이프
    u = ease(t / T_DRAW)
    xr = int(CX0 + (CX1 - CX0) * u) + 2
    box = (0, CY0 - 30, xr, CY1 + 2)
    crop = curve_small.crop(box)
    im.paste(crop, (box[0], box[1]), crop)
    for mx, my, lay in markers:
        if mx <= xr:
            # 표식 등장 뒤 은은한 맥동 고리
            ph = (t * 0.6) % 1.0
            r = 17 + 16 * ph
            al = int(150 * (1 - ph))
            ov = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
            do = ImageDraw.Draw(ov)
            do.ellipse((48 - r, 48 - r, 48 + r, 48 + r), outline=(255, 255, 255, al), width=2)
            im.paste(ov, (int(mx) - 48, int(my) - 48), ov)
            im.paste(lay, (0, 0), lay)
    return im


if a.preview:
    frame(int(a.at * a.fps) % NF).save(a.preview)
    print("preview", a.preview)
    raise SystemExit

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
       "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out]
pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
cache = []
for k in range(NF):
    b = frame(k).tobytes()
    cache.append(b)
    pr.stdin.write(b)
for _ in range(a.loops - 1):
    for b in cache:
        pr.stdin.write(b)
pr.stdin.close()
rc = pr.wait()
print("ffmpeg rc", rc, "frames/loop", NF, "loops", a.loops, "dur", NF * a.loops / a.fps, "s", a.out)
