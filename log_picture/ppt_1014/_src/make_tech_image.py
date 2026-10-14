"""제어 기술력 — 숫자로 본 결과. 1920x1080 전체화면용 이미지.

수치 출처: reports/real_1012/RESULT.md(서기 289 s·롤/피치 표준편차), reports/real_1001/REAL_1001.md(68 s·지연 320→80 ms),
reports/real_1008(공중 시작 92.7 s·공중 다리 이동 ≤ 4.0°), reports/real_1009/lock_eval(관절 고정 시뮬 — 질량 정정판 1/619),
reports/real_0918(무릎 처짐·발목 유격 실측), G6 학습 기록(4,096대·4,500회·8,837 s).
"""
import sys

from PIL import Image, ImageDraw

from style import AMBER, BG, BLUE, CARD, CARD_LINE, CORAL, DIM, GREEN, SUB, TXT, VIOLET, font, tw

OUT = sys.argv[1]
W, H = 1920, 1080
S = 2

im = Image.new("RGB", (W * S, H * S), BG)
d = ImageDraw.Draw(im)


def P(v):
    return int(round(v * S))


def text(x, y, s, f, fill, anchor="la"):
    d.text((P(x), P(y)), s, font=f, fill=fill, anchor=anchor)


def width(s, f):
    return tw(d, s, f) / S


def chip(x, y, s, col, h=36):
    f = font(P(19), "Medium")
    w = width(s, f) + 30
    d.rounded_rectangle((P(x), P(y), P(x + w), P(y + h)), radius=P(h / 2), outline=col, width=P(1.5))
    text(x + w / 2, y + h / 2 - 1, s, f, col, anchor="mm")
    return w


def card(x0, y0, w, h):
    d.rounded_rectangle((P(x0), P(y0), P(x0 + w), P(y0 + h)), radius=P(16), fill=CARD, outline=CARD_LINE, width=P(1))


def dim(c, k=0.42):
    return tuple(int(CARD[i] + (c[i] - CARD[i]) * k) for i in range(3))


MX = 56
# ---- 제목
text(MX, 40, "제어 기술력 — 숫자로 본 결과", font(P(50), "Bold"), TXT)
text(MX, 112, "시뮬레이션에서만 학습한 신경망 제어기를 이족 로봇 실물에 올렸다. 실물 데이터로는 학습하지 않았다.",
     font(P(25), "Regular"), SUB)

TOP, BOT, GAP = 168, 1012, 22
# ================= 왼쪽 큰 카드 =================
LW = 820
x0, y0 = MX, TOP
card(x0, y0, LW, BOT - TOP)
px = x0 + 36
chip(px, y0 + 26, "실물  ·  2026-10-12", GREEN)
text(px, y0 + 84, "유격 있는 기구 위에서, 혼자 서 있는 동안의 몸통 흔들림", font(P(27), "Medium"), TXT)
fbig = font(P(150), "Bold")
text(px - 6, y0 + 212, "±0.07°", fbig, GREEN, anchor="lm")
bw = width("±0.07°", fbig)
text(px + bw + 22, y0 + 190, "피치 (앞뒤)", font(P(24), "Medium"), SUB, anchor="lm")
text(px + bw + 22, y0 + 228, "롤 (좌우) ±0.03°", font(P(24), "Medium"), SUB, anchor="lm")
text(px, y0 + 300, "무보조 서기 4분 49초(289초), 흔들림은 마지막 10초 구간의 값", font(P(23), "Regular"), SUB)

d.line((P(px), P(y0 + 352), P(x0 + LW - 36), P(y0 + 352)), fill=CARD_LINE, width=P(1))
text(px, y0 + 372, "이 로봇의 기구 상태", font(P(22), "Medium"), DIM)
stats = [("12°", "무릎 하중 처짐", "9/19 실측 10~12.5°"), ("8°", "발목 유격", "엔코더는 1.5°만 봄"), ("2개", "플라스틱으로 고정한 관절", "10/9 모터 고장")]
cw = (LW - 72) / 3
for i, (num, c1, c2) in enumerate(stats):
    sx = px + i * cw
    fn = font(P(56), "Bold")
    text(sx, y0 + 436, num, fn, AMBER, anchor="lm")
    text(sx, y0 + 486, c1, font(P(20), "Medium"), TXT, anchor="lm")
    text(sx, y0 + 514, c2, font(P(18), "Regular"), SUB, anchor="lm")
    assert width(c1, font(P(20), "Medium")) < cw - 10 and width(c2, font(P(18), "Regular")) < cw - 10, c1

d.line((P(px), P(y0 + 548), P(x0 + LW - 36), P(y0 + 548)), fill=CARD_LINE, width=P(1))
text(px, y0 + 566, "무보조 자립 기록", font(P(22), "Medium"), DIM)
bars = [("10/1", 68, "68초"), ("10/8", 92.7, "92.7초"), ("10/10", 180, "약 3분"), ("10/12", 289, "4분 49초")]
bx0, bx1 = px + 8, x0 + LW - 44
base_y, max_h = y0 + 790, 150
slot = (bx1 - bx0) / len(bars)
bwid = 118
for i, (date, v, lab) in enumerate(bars):
    cx = bx0 + slot * (i + 0.5)
    hgt = max_h * v / 289
    col = GREEN if i == len(bars) - 1 else dim(GREEN, 0.50)
    d.rounded_rectangle((P(cx - bwid / 2), P(base_y - hgt), P(cx + bwid / 2), P(base_y)), radius=P(6), fill=col)
    text(cx, base_y - hgt - 20, lab, font(P(25), "Bold"), TXT if i == len(bars) - 1 else SUB, anchor="mm")
    text(cx, base_y + 24, date, font(P(21), "Regular"), SUB, anchor="mm")
d.line((P(bx0), P(base_y), P(bx1), P(base_y)), fill=CARD_LINE, width=P(1.5))

# ================= 오른쪽 2x2 =================
RX = MX + LW + GAP
RW = (W - MX - RX - GAP) / 2
RH = (BOT - TOP - GAP) / 2
cards = [
    ("시뮬 학습", BLUE, "약 1,000", "배", "실시간보다 빠른 학습",
     ["로봇 4,096대를 동시에 시뮬레이션", "2.5시간에 102일치 경험", "GPU 1장, 신경망 512·256·128"]),
    ("시뮬 + 실물  ·  10/9~10/12", VIOLET, "12 → 10", "관절", "모터가 고장 나도 섰다",
     ["관절 2개를 플라스틱으로 고정", "기존 제어기 시뮬 619회 중 낙상 1회", "실물: 고정 뒤 약 3분, 4분 49초 자립"]),
    ("실물  ·  10/8", GREEN, "92.7", "초", "공중에서 켜서 착지 후 자립",
     ["매달린 상태에서 제어 시작", "공중 다리 움직임 4.0° 이내", "시뮬레이션과 같은 수준"]),
    ("실물  ·  9/30~10/1", AMBER, "320 → 80", "ms", "명령 응답 지연 4분의 1",
     ["원인: 설정 파일을 초당 약 740번 재파싱", "명령 처리 초당 31개 → 50개", "수정 다음 날 68초 무보조 자립"]),
]
BIGSZ = 82
while any(width(c[2], font(P(BIGSZ), "Bold")) + 12 + width(c[3], font(P(32), "Bold")) >= RW - 60 for c in cards):
    BIGSZ -= 2
for i, (tag, col, big, unit, head, lines) in enumerate(cards):
    cx0 = RX + (i % 2) * (RW + GAP)
    cy0 = TOP + (i // 2) * (RH + GAP)
    card(cx0, cy0, RW, RH)
    qx = cx0 + 30
    chip(qx, cy0 + 24, tag, col, h=34)
    fb = font(P(BIGSZ), "Bold")
    text(qx - 3, cy0 + 128, big, fb, col, anchor="lm")
    text(qx + width(big, fb) + 12, cy0 + 144, unit, font(P(32), "Bold"), col, anchor="lm")
    text(qx, cy0 + 196, head, font(P(29), "Bold"), TXT)
    d.line((P(qx), P(cy0 + 250), P(cx0 + RW - 30), P(cy0 + 250)), fill=CARD_LINE, width=P(1))
    for j, ln in enumerate(lines):
        yy = cy0 + 284 + j * 38
        d.ellipse((P(qx + 2), P(yy - 4), P(qx + 10), P(yy + 4)), fill=col)
        text(qx + 24, yy - 1, ln, font(P(22), "Regular"), SUB, anchor="lm")

# ---- 바닥글
foot = ("실물 수치는 로봇 기록(콘솔·블랙박스) 기준   ·   시뮬 수치는 실물 성공률이 아님   ·   "
        "실물에서 관절 고정 제어와 학습 제어를 같은 조건으로 비교한 실험은 없음")
text(MX, 1046, foot, font(P(20), "Regular"), DIM, anchor="lm")

# 폭 점검(카드 밖으로 나가는 글자 없게)
for tag, col, big, unit, head, lines in cards:
    assert width(head, font(P(29), "Bold")) < RW - 60, head
    for ln in lines:
        assert 24 + width(ln, font(P(22), "Regular")) < RW - 56, ln
assert width(foot, font(P(20), "Regular")) < W - 2 * MX, "foot"

im = im.resize((W, H), Image.LANCZOS)
im.save(OUT)
print("saved", OUT, im.size)
