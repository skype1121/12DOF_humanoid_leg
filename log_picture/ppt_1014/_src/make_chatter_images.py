"""'떨림을 누르면 회복이 죽는다' — 문제 한 장 + 해결 한 장. 1920x1080 전체화면용.

수치 출처:
  reports/gen_g17/MORNING_1002.md (직전 명령 입력을 늦춘 폐루프 대조: 부호반전 0.392 → 0.002, 뒤 18 N 충격 낙상 18 → 70 %)
  reports/gen_g15/G15_VERDICT.md (정숙 보상 3종: 뒤 18 N 낙상 45 → 74–96 %)
  reports/gen_g17/G17_VERDICT.md (벌점 ×3: 부호반전 0.18–0.39 → 0.001–0.006, 느린 흔들림, 24/24 탈락)
  reports/gen_g18/G18_VERDICT.md (G18o → G18f 두 seed: 부호반전 0.23–0.39 → 0.023–0.055, 뒤 18 N 17–30 → 7–27 %)
  reports/gen_g18/g21_verdict_out.txt (같은 선별 평가: 대조 g21o 흔들림 2.7–8.0 %·뒤 18 N 8.0–11.6 % → g21a 0–0.9 %·3.6–4.5 %)
  reports/gen_g18/G21_VERDICT.md (G21a-11300 정식: 흔들림 2.2 %, 낙상 19.1/10.2/7.4 vs G12 44.6/84.9/63.4, 지연 3틱 흔들림 5.6 > 2.9)
  reports/gen_c5 (서기 + 회복 전환: 앞 10 N 2.5 %·뒤 18 N 1.4 %·오탐 1.4–1.8 %·앞 15 N 31.8 %)
  reports/real_1003/REAL_1003.md (실물: 1.46 s 발산 → 약 0.2 s 정착, 2–4.5 Hz RMS 0.01 → 0.002 rad/s)
"""
import math
import sys

from PIL import Image, ImageDraw

from style import AMBER, BG, BLUE, CARD, CARD_LINE, CORAL, DIM, GREEN, SUB, TXT, VIOLET, font, tw

OUT_A, OUT_B = sys.argv[1], sys.argv[2]
W, H = 1920, 1080
S = 2
MX = 56


class Canvas:
    def __init__(self):
        self.im = Image.new("RGB", (W * S, H * S), BG)
        self.d = ImageDraw.Draw(self.im)

    def P(self, v):
        return int(round(v * S))

    def text(self, x, y, s, f, fill, anchor="la"):
        self.d.text((self.P(x), self.P(y)), s, font=f, fill=fill, anchor=anchor)

    def width(self, s, f):
        return tw(self.d, s, f) / S

    def F(self, size, weight="Regular"):
        return font(self.P(size), weight)

    def card(self, x0, y0, w, h, fill=CARD, outline=CARD_LINE):
        self.d.rounded_rectangle((self.P(x0), self.P(y0), self.P(x0 + w), self.P(y0 + h)), radius=self.P(16), fill=fill,
                                 outline=outline, width=self.P(1))

    def box(self, x0, y0, w, h, col, label, sub=None):
        self.d.rounded_rectangle((self.P(x0), self.P(y0), self.P(x0 + w), self.P(y0 + h)), radius=self.P(12),
                                 fill=mix(col, 0.16), outline=col, width=self.P(2))
        if sub:
            self.text(x0 + w / 2, y0 + h / 2 - 15, label, self.F(25, "Bold"), TXT, anchor="mm")
            self.text(x0 + w / 2, y0 + h / 2 + 17, sub, self.F(18), SUB, anchor="mm")
        else:
            self.text(x0 + w / 2, y0 + h / 2 - 1, label, self.F(25, "Bold"), TXT, anchor="mm")

    def line(self, pts, col, wd=3):
        self.d.line([(self.P(x), self.P(y)) for x, y in pts], fill=col, width=self.P(wd), joint="curve")

    def arrow(self, pts, col, wd=3, head=13):
        self.line(pts, col, wd)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        a = math.atan2(y1 - y0, x1 - x0)
        p1 = (x1 + head * 0.25 * math.cos(a), y1 + head * 0.25 * math.sin(a))
        p2 = (x1 - head * math.cos(a - 0.45), y1 - head * math.sin(a - 0.45))
        p3 = (x1 - head * math.cos(a + 0.45), y1 - head * math.sin(a + 0.45))
        self.d.polygon([(self.P(x), self.P(y)) for x, y in (p1, p2, p3)], fill=col)

    def chip(self, x, y, s, col, h=34, size=19):
        f = self.F(size, "Medium")
        w = self.width(s, f) + 28
        self.d.rounded_rectangle((self.P(x), self.P(y), self.P(x + w), self.P(y + h)), radius=self.P(h / 2), outline=col,
                                 width=self.P(1.5))
        self.text(x + w / 2, y + h / 2 - 1, s, f, col, anchor="mm")
        return w

    def badge(self, cx, cy, n, col, r=19):
        self.d.ellipse((self.P(cx - r), self.P(cy - r), self.P(cx + r), self.P(cy + r)), fill=col)
        self.text(cx, cy - 1, str(n), self.F(23, "Bold"), BG, anchor="mm")

    def wrap(self, s, f, maxw):
        out, cur = [], ""
        for word in s.split(" "):
            t = (cur + " " + word).strip()
            if self.width(t, f) <= maxw:
                cur = t
            else:
                out.append(cur)
                cur = word
        if cur:
            out.append(cur)
        return out

    def para(self, x, y, s, f, fill, maxw, lh):
        for i, ln in enumerate(self.wrap(s, f, maxw)):
            self.text(x, y + i * lh, ln, f, fill)
        return y + len(self.wrap(s, f, maxw)) * lh

    def metric(self, x, y, w, label, before, after, vb, va, vmax, good):
        """라벨 / 전 → 후 (후 = 좋아졌으면 초록, 나빠졌으면 산호색) / 가는 막대 두 줄."""
        col = GREEN if good else CORAL
        self.text(x, y, label, self.F(19), SUB)
        fb, fa = self.F(27, "Bold"), self.F(36, "Bold")
        yy = y + 54
        self.text(x, yy, before, fb, SUB, anchor="ls")
        xx = x + self.width(before, fb) + 10
        self.text(xx, yy, "→", fb, DIM, anchor="ls")
        xx += self.width("→", fb) + 10
        self.text(xx, yy, after, fa, col, anchor="ls")
        by = y + 70
        for k, (v, c) in enumerate(((vb, mix(SUB, 0.55)), (va, col))):
            ww = max(4, w * min(v / vmax, 1.0))
            self.d.rounded_rectangle((self.P(x), self.P(by + k * 13), self.P(x + ww), self.P(by + k * 13 + 8)), radius=self.P(4), fill=c)
        return y + 104

    def save(self, path):
        im = self.im.resize((W, H), Image.LANCZOS)
        im.save(path)
        print("saved", path, im.size)


def mix(c, k):
    return tuple(int(CARD[i] + (c[i] - CARD[i]) * k) for i in range(3))


# =====================================================================================================
# A. 문제
# =====================================================================================================
c = Canvas()
c.text(MX, 40, "문제 — 떨림을 누르면 회복이 죽는다", c.F(50, "Bold"), TXT)
c.text(MX, 112, "서 있을 때의 명령 떨림과 밀렸을 때의 회복이, 제어기 안에서 같은 되먹임 고리를 쓰고 있었다", c.F(25), SUB)
_cs = "시뮬레이션 대조 실험"
c.chip(W - MX - (c.width(_cs, c.F(20, "Medium")) + 28), 46, _cs, BLUE, h=38, size=20)

TOP, BOT = 168, 916
# ---- 왼쪽: 고리 그림
LW = 800
c.card(MX, TOP, LW, BOT - TOP)
lx, ly = MX, TOP
c.text(lx + 36, ly + 28, "제어 고리", c.F(22, "Medium"), DIM)
# 상자 세 개
bx_c = (lx + 60, ly + 300, 250, 120)      # 제어기
bx_f = (lx + 400, ly + 320, 130, 80)      # 필터
bx_r = (lx + 600, ly + 300, 150, 120)     # 로봇
c.box(*bx_c, BLUE, "신경망 제어기", "50 Hz")
c.box(*bx_f, VIOLET, "필터")
c.box(*bx_r, GREEN, "로봇")
ym = bx_c[1] + bx_c[3] / 2
# 제어기 → 필터 → 로봇
c.arrow([(bx_c[0] + bx_c[2] + 4, ym), (bx_f[0] - 6, ym)], SUB)
c.arrow([(bx_f[0] + bx_f[2] + 4, ym), (bx_r[0] - 6, ym)], SUB)
c.text((bx_c[0] + bx_c[2] + bx_f[0]) / 2, ym + 26, "관절 명령", c.F(18), SUB, anchor="mm")
# 바깥 고리: 로봇 → 센서 → 제어기
yb = bx_c[1] + bx_c[3] + 110
xr = bx_r[0] + bx_r[2] / 2
xc = bx_c[0] + bx_c[2] / 2
c.arrow([(xr, bx_r[1] + bx_r[3] + 4), (xr, yb), (xc, yb), (xc, bx_c[1] + bx_c[3] + 8)], SUB)
c.text((xr + xc) / 2, yb + 26, "센서: 몸통 기울기 · 관절 각도와 속도", c.F(19), SUB, anchor="mm")
# 안쪽 고리(문제의 경로): 명령(필터 전) → 제어기 입력
xs = bx_c[0] + bx_c[2] + 62
yt = bx_c[1] - 120
c.arrow([(xs, ym - 6), (xs, yt), (xc, yt), (xc, bx_c[1] - 8)], CORAL, wd=4.5, head=16)
c.d.ellipse((c.P(xs - 7), c.P(ym - 7), c.P(xs + 7), c.P(ym + 7)), fill=CORAL)
c.text((xs + xc) / 2, yt - 26, "방금 낸 명령을 다시 입력으로 본다", c.F(22, "Bold"), CORAL, anchor="mm")
# 설명
ty = ly + 590
for i, (col, s1, s2) in enumerate([
        (CORAL, "이 안쪽 고리가 초당 25번 떨리는 명령을 만든다", "가만히 서 있을 때도 모터가 계속 미세하게 흔들림"),
        (CORAL, "그런데 같은 고리가 밀렸을 때의 빠른 회복에도 쓰인다", "고리를 약하게 하면 조용해지지만 밀렸을 때 못 버팀")]):
    yy = ty + i * 74
    c.d.ellipse((c.P(lx + 40), c.P(yy + 8), c.P(lx + 54), c.P(yy + 22)), fill=col)
    c.text(lx + 70, yy, s1, c.F(23, "Bold"), TXT)
    c.text(lx + 70, yy + 34, s2, c.F(20), SUB)

# ---- 오른쪽: 증거 세 칸
RX = MX + LW + 22
RW = W - MX - RX
GAP = 18
EH = (BOT - TOP - 2 * GAP) / 3
ev = [
    ("고리를 늦췄더니", "방금 낸 명령 입력에 저역 필터를 걸어 평가",
     [("떨림 (명령 방향 뒤집힘, 주기당 횟수)", "0.392", "0.002", 0.392, 0.002, 0.4, True),
      ("뒤에서 18 N 충격 → 낙상", "18 %", "70 %", 18, 70, 100, False)], None),
    ("‘조용히 서라’ 보상을 넣었더니", "정숙 보상 세 가지를 각각 넣어 학습",
     [("뒤에서 18 N 충격 → 낙상", "45 %", "74~96 %", 45, 85, 100, False)],
     ("세 가지 모두 같은 방향", "조용함을 보상하면 회복 스텝이 사라진다")),
    ("명령 변화 벌점을 3배로 했더니", "떨림을 직접 벌주는 가장 흔한 방법",
     [("떨림 (명령 방향 뒤집힘, 주기당 횟수)", "0.18~0.39", "0.001~0.006", 0.29, 0.004, 0.4, True)],
     ("대신 느린 몸통 흔들림", "두 방식 후보 24개 전부 기준 미달")),
]
for i, (title, sub, metrics, note) in enumerate(ev):
    y0 = TOP + i * (EH + GAP)
    c.card(RX, y0, RW, EH)
    c.badge(RX + 46, y0 + 46, i + 1, AMBER)
    c.text(RX + 78, y0 + 28, title, c.F(27, "Bold"), TXT)
    c.text(RX + 78, y0 + 68, sub, c.F(19), SUB)
    mw = 400
    for j, m in enumerate(metrics):
        c.metric(RX + 78 + j * (mw + 40), y0 + 112, mw - 60, *m)
    if note:
        nx = RX + 78 + mw + 40
        c.text(nx, y0 + 128, note[0], c.F(27, "Bold"), CORAL)
        c.text(nx, y0 + 170, note[1], c.F(20), SUB)

# ---- 아래 띠
c.card(MX, BOT + 22, W - 2 * MX, 1012 - BOT - 22, fill=mix(AMBER, 0.10), outline=mix(AMBER, 0.45))
c.text(W / 2, (BOT + 22 + 1012) / 2 - 1, "조용함 ↔ 민첩함 :  제어공학의 ‘대역폭 대 잡음’ 절충이 학습 제어기 안에서도 그대로 나타난다",
       c.F(26, "Medium"), TXT, anchor="mm")
c.text(MX, 1046, "수치는 모두 시뮬레이션 평가(로봇 약 110~340대, 무작위 오차 포함) · 실물 성공률이 아님", c.F(20), DIM, anchor="lm")
c.save(OUT_A)

# =====================================================================================================
# B. 해결
# =====================================================================================================
c = Canvas()
c.text(MX, 40, "해결 — 고리를 끊지 않고, 옮기고 나눴다", c.F(50, "Bold"), TXT)
c.text(MX, 112, "떨림 경로를 없애면 회복도 사라진다. 그래서 경로는 살려 둔 채 세 가지를 바꿨다", c.F(25), SUB)

TOP, MID = 168, 640
GAP = 22
CW3 = (W - 2 * MX - 2 * GAP) / 3
steps = [
    (BLUE, "고리를 필터 뒤로 옮겼다",
     "제어기가 ‘자기가 낸 명령’이 아니라 ‘필터를 거쳐 실제로 나간 명령’을 보게 했다. 관절 속도 입력에도 같은 방식의 필터를 걸었다.",
     [("떨림 (명령 방향 뒤집힘, 주기당 횟수)", "0.23~0.39", "0.023~0.055", 0.31, 0.039, 0.4, True),
      ("뒤 18 N 충격 낙상", "17~30 %", "7~27 %", 23.5, 17, 100, True)],
     "시뮬 · 학습 시드 2개에서 재현"),
    (VIOLET, "죽어 있던 보상을 고쳤다",
     "‘자리 잡고 조용히 서면 가점’이 높이 문턱(0.60 m) 때문에 한 번도 켜지지 않고 있었다(서는 높이 0.597 m). 문턱을 연속 함수로 바꿨다.",
     [("느린 몸통 흔들림이 이어진 비율", "2.7~8.0 %", "0~0.9 %", 5.35, 0.45, 10, True),
      ("뒤 18 N 충격 낙상", "8.0~11.6 %", "3.6~4.5 %", 9.8, 4.05, 100, True)],
     "시뮬 · 정식 평가(약 340대)에서는 흔들림 2.2 %"),
    (GREEN, "서기와 회복을 나눴다",
     "조용한 서기 제어기와 회복 전담 제어기를 따로 두었다. 기울기 4° 또는 각속도 0.5 rad/s 를 넘으면 회복으로 넘기고, 1초간 잠잠하면 서기로 돌아온다.",
     [("앞 10 N 지속 밀기 낙상", "19.1 %", "2.5 %", 19.1, 2.5, 100, True),
      ("뒤 18 N 충격 낙상", "7.4 %", "1.4 %", 7.4, 1.4, 100, True)],
     "시뮬 · 밀지 않았는데 회복이 켜진 비율 1.4~1.8 %"),
]
for i, (col, title, body, metrics, note) in enumerate(steps):
    x0 = MX + i * (CW3 + GAP)
    c.card(x0, TOP, CW3, MID - TOP)
    c.badge(x0 + 46, TOP + 48, i + 1, col)
    c.text(x0 + 78, TOP + 28, title, c.F(29, "Bold"), col)
    yy = c.para(x0 + 32, TOP + 88, body, c.F(20), SUB, CW3 - 64, 30)
    assert yy <= TOP + 184, (title, yy - TOP)
    c.line([(x0 + 32, TOP + 196), (x0 + CW3 - 32, TOP + 196)], CARD_LINE, 1)
    for j, m in enumerate(metrics):
        c.metric(x0 + 32, TOP + 212 + j * 100, CW3 - 64 - 120, *m)
    c.text(x0 + 32, MID - 40, note, c.F(18), DIM)

# ---- 종합 띠
BY0, BY1 = MID + 16, MID + 16 + 58
c.card(MX, BY0, W - 2 * MX, BY1 - BY0, fill=mix(GREEN, 0.09), outline=mix(GREEN, 0.40))
cy = (BY0 + BY1) / 2 - 1
xx = MX + 28
for s_, f_, col_ in [("종합 (시뮬, 이전 실물 제어기 → 최종 구성)", c.F(21, "Medium"), SUB), ("     뒤 18 N 충격 낙상  ", c.F(21), TXT),
                     ("63.4 % → 1.4 %", c.F(25, "Bold"), GREEN), ("     앞 10 N 지속 밀기 낙상  ", c.F(21), TXT), ("44.6 % → 2.5 %", c.F(25, "Bold"), GREEN),
                     ("     정지 떨림은 같은 수준  ", c.F(21), TXT), ("0.002 → 0.002~0.003", c.F(25, "Bold"), GREEN)]:
    c.text(xx, cy, s_, f_, col_, anchor="lm")
    xx += c.width(s_, f_)
assert xx < W - MX - 20, xx

# ---- 아래: 실물 / 남은 한계
BT, BB = BY1 + 16, 1012
RW2 = 560
LW2 = W - 2 * MX - GAP - RW2
c.card(MX, BT, LW2, BB - BT)
c.chip(MX + 32, BT + 22, "실물에서 확인", GREEN)
c.text(MX + 196, BT + 39, "같은 로봇을 약하게 밀었을 때 (기울기 변화 1.0~1.3°, 각 1회 시험)", c.F(21), SUB, anchor="lm")
hx = MX + 32
fbig = c.F(60, "Bold")
c.text(hx, BT + 80, "10/2  이전 제어기", c.F(20, "Medium"), SUB)
c.text(hx, BT + 172, "1.46초", fbig, CORAL, anchor="ls")
c.text(hx + c.width("1.46초", fbig) + 12, BT + 172, "만에 발산", c.F(26, "Bold"), CORAL, anchor="ls")
c.text(hx, BT + 196, "3.2 Hz 로 흔들림이 커지며 15° 넘게 기욺", c.F(19), SUB)
hx2 = hx + 420
c.arrow([(hx2 - 54, BT + 148), (hx2 - 20, BT + 148)], DIM, wd=4, head=15)
c.text(hx2, BT + 80, "10/3  새 서기 제어기", c.F(20, "Medium"), SUB)
c.text(hx2, BT + 172, "0.2초", fbig, GREEN, anchor="ls")
c.text(hx2 + c.width("0.2초", fbig) + 12, BT + 172, "만에 정착", c.F(26, "Bold"), GREEN, anchor="ls")
c.text(hx2, BT + 196, "발 들림 없이 원래 자세로", c.F(19), SUB)
# 오른쪽 세 줄
fx = hx2 + 380
c.line([(fx - 30, BT + 78), (fx - 30, BB - 24)], CARD_LINE, 1)
facts = [("0.01 → 0.002", "rad/s", "가만히 설 때 진동 (10/2 → 10/3)"),
         ("0회", "", "서기 106초 동안 회복 오작동 (10/8)"),
         ("±0.07°", "", "4분 49초 서기의 몸통 흔들림 (10/12)")]
for i, (num, unit, cap) in enumerate(facts):
    fy = BT + 100 + i * 68
    fn = c.F(28, "Bold")
    c.text(fx, fy, num, fn, GREEN, anchor="ls")
    nx = fx + c.width(num, fn) + 6
    if unit:
        c.text(nx, fy, unit, c.F(18, "Medium"), GREEN, anchor="ls")
    c.text(fx, fy + 8, cap, c.F(17), SUB)
    assert fx + c.width(cap, c.F(17)) < MX + LW2 - 16, cap
assert BT + 100 + 2 * 68 + 32 < BB, "facts overflow"

x1 = MX + LW2 + GAP
c.card(x1, BT, RW2, BB - BT)
c.chip(x1 + 32, BT + 22, "남은 한계", AMBER)
lims = [("강한 앞 밀기", "시뮬 앞 15 N 지속 낙상 31.8 %, 실물도 앞 스텝을 못 냄"),
        ("긴 지연에서의 흔들림", "지연 3주기 조건 5.6 % (사전 기준 2.9 %) 로 미달"),
        ("실물 비교", "제어기마다 1회, 다른 날 시험 (같은 날 대조 없음)")]
yy = BT + 76
for t, b in lims:
    c.d.ellipse((c.P(x1 + 34), c.P(yy + 9), c.P(x1 + 44), c.P(yy + 19)), fill=AMBER)
    c.text(x1 + 58, yy, t, c.F(21, "Bold"), TXT)
    c.text(x1 + 58, yy + 31, b, c.F(18), SUB)
    assert x1 + 58 + c.width(b, c.F(18)) < x1 + RW2 - 16, b
    yy += 64
assert yy <= BB, yy - BT
c.text(MX, 1046, "시뮬 수치는 실물 성공률이 아님  ·  각 단계의 앞 숫자는 그 변경만 뺀 대조군 (①·②는 대조 학습, ③은 서기 제어기 단독)", c.F(20), DIM, anchor="lm")
c.save(OUT_B)
