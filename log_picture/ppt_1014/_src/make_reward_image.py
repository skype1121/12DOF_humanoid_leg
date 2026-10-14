"""보상(목적함수) 항목 목록 — 1920x1080 전체화면용 이미지.

항목·가중치 출처: logs/rsl_rl/biped12_stage4/2026-09-20_15-06-59_g6/params/env.yaml (실제 학습 설정, 32개 항목).
"""
import sys

from PIL import Image, ImageDraw

from style import AMBER, BG, BLUE, CARD, CARD_LINE, CORAL, DIM, GREEN, SUB, TXT, VIOLET, font, mono, tw

OUT = sys.argv[1]
W, H = 1920, 1080
S = 2                                    # 2배로 그리고 축소(계단 현상 방지)

GROUPS = [
    ("해야 할 일", "잘하면 가점", GREEN, [
        ("무릎을 살짝 굽힌 자세로 지지", "knee_stance_flexion", "+4.0"),
        ("양발을 바닥에 붙이고 있기", "feet_contact_gated", "+2.0"),
        ("제자리 유지 (수평 속도 명령 0 추종)", "track_lin_vel_xy_exp", "+2.0"),
        ("제자리 유지 (회전 속도 명령 0 추종)", "track_ang_vel_z_exp", "+2.0"),
        ("자리 잡은 뒤 흔들림 없이 정지", "settle_quiet_bonus", "+2.0"),
        ("지지 없이 서 있는 매 순간", "survive_after_release", "+1.0"),
    ]),
    ("바른 자세", "벗어나면 감점", BLUE, [
        ("몸통 앞뒤 기울기 ±3° 초과", "torso_pitch_band", "−25.0"),
        ("몸통이 수평에서 기울어짐", "flat_orientation_l2", "−8.0"),
        ("발 간격·앞뒤 어긋남 (0.24~0.32 m 밖)", "stance_shape", "−5.0"),
        ("두 발이 너무 가까움 (0.22 m 미만)", "feet_gap", "−5.0"),
        ("좌우 비대칭 자세", "pose_symmetry", "−3.0"),
        ("골반 높이 목표 0.55 m 에서 벗어남", "base_height_l2", "−2.0"),
        ("제자리에서 밀려남", "march_anchor", "−1.0"),
        ("기준 관절 자세에서 벗어남 (4개 항목)", "joint_deviation ×4", "−0.2 ~ −3.0"),
    ]),
    ("부드럽게, 적은 힘으로", "거칠수록 감점", VIOLET, [
        ("명령이 급하게 바뀜 (떨림)", "action_rate_l2", "−0.2"),
        ("몸통 위아래 출렁임", "lin_vel_z_l2", "−0.2"),
        ("몸통 기울기 흔들림 (각속도)", "ang_vel_xy_l2", "−0.1"),
        ("매달린 동안 발버둥", "hang_quiet", "−0.002"),
        ("관절 속도", "joint_vel_l2", "−0.0005"),
        ("관절 토크 (에너지)", "dof_torques_l2", "−0.0001"),
        ("관절 가속도", "dof_acc_l2", "−2.5e-7"),
    ]),
    ("하드웨어 보호·안전", "위험할수록 감점", AMBER, [
        ("넘어짐", "termination_penalty", "−10.0"),
        ("관절 가동 한계에 접근", "dof_pos_limits", "−5.0"),
        ("모터 토크가 한계의 60 % 초과", "torque_saturation", "−2.0"),
        ("발 이외 부위가 바닥에 닿음", "undesired_contacts", "−1.0"),
        ("디딘 발이 비틀리며 미끄러짐", "foot_yaw_slip", "−0.5"),
        ("발 미끄러짐", "feet_slide", "−0.1"),
        ("착지 충격", "feet_impact", "−0.01"),
        ("과도한 접지력 (150 N 초과)", "feet_contact_forces", "−0.002"),
    ]),
]
n_terms = sum(len(g[3]) for g in GROUPS) + 3      # 'joint_deviation ×4' 한 줄 = 4개
assert n_terms == 32, n_terms

im = Image.new("RGB", (W * S, H * S), BG)
d = ImageDraw.Draw(im)


def P(v):
    return int(round(v * S))


def text(x, y, s, f, fill, anchor="la"):
    d.text((P(x), P(y)), s, font=f, fill=fill, anchor=anchor)


MX = 56
# ---- 제목
text(MX, 40, "목적함수(보상) — ‘잘 서 있다’를 숫자로 정의하기", font(P(50), "Bold"), TXT)
sub = "제어 주기 0.02초마다 아래 32개 항목을 계산해 가중합을 만들고, 그 합이 커지도록 신경망 제어기를 학습한다"
text(MX, 112, sub, font(P(25), "Regular"), SUB)
# 수식 칩 (오른쪽 위)
chip = "J = Σ 가중치 × 항목값"
fch = font(P(26), "Medium")
cw = tw(d, chip, fch) / S
cx1 = W - MX
d.rounded_rectangle((P(cx1 - cw - 44), P(40), P(cx1), P(92)), radius=P(26), outline=CARD_LINE, width=P(1.5), fill=CARD)
text(cx1 - 22, 66, chip, fch, TXT, anchor="rm")

# ---- 카드 2x2
TOP, BOT, GAP = 168, 1012, 22
CW = (W - 2 * MX - GAP) / 2
CH = (BOT - TOP - GAP) / 2
HEAD = 62
ROWS = 8
RH = (CH - HEAD - 18) / ROWS
f_head = font(P(29), "Bold")
f_hint = font(P(21), "Regular")
f_desc = font(P(24), "Medium")
f_code = mono(P(16))
f_w = font(P(26), "Bold")
f_num = font(P(22), "Bold")

for gi, (title, hint, col, rows) in enumerate(GROUPS):
    x0 = MX + (gi % 2) * (CW + GAP)
    y0 = TOP + (gi // 2) * (CH + GAP)
    d.rounded_rectangle((P(x0), P(y0), P(x0 + CW), P(y0 + CH)), radius=P(16), fill=CARD, outline=CARD_LINE, width=P(1))
    # 머리: 번호 원 + 제목 + 힌트
    cxn, cyn = x0 + 40, y0 + HEAD / 2 + 4
    d.ellipse((P(cxn - 17), P(cyn - 17), P(cxn + 17), P(cyn + 17)), fill=col)
    text(cxn, cyn - 1, str(gi + 1), f_num, BG, anchor="mm")
    text(x0 + 70, cyn - 1, title, f_head, col, anchor="lm")
    tx = x0 + 70 + tw(d, title, f_head) / S + 16
    text(tx, cyn + 1, hint, f_hint, SUB, anchor="lm")
    nitem = sum(4 if "×4" in r[1] else 1 for r in rows)
    text(x0 + CW - 26, cyn + 1, f"{nitem}개 항목", f_hint, DIM, anchor="rm")
    d.line((P(x0 + 22), P(y0 + HEAD + 6), P(x0 + CW - 22), P(y0 + HEAD + 6)), fill=CARD_LINE, width=P(1))
    for ri, (desc, code, w) in enumerate(rows):
        cy = y0 + HEAD + 14 + RH * (ri + 0.5)
        text(x0 + 28, cy - 1, desc, f_desc, TXT, anchor="lm")
        wcol = GREEN if w.startswith("+") else CORAL
        text(x0 + CW - 28, cy - 1, w, f_w, wcol, anchor="rm")
        text(x0 + CW - 200, cy, code, f_code, DIM, anchor="rm")

# ---- 바닥글
foot = "가중치: + 가점 / − 감점   ·   항목마다 단위가 달라 가중치 크기끼리 직접 비교할 수 없음   ·   출처: 실제 학습 설정 (서기 정책 G6 세대, 2026-09-20)"
text(MX, 1046, foot, font(P(20), "Regular"), DIM, anchor="lm")

im = im.resize((W, H), Image.LANCZOS)
im.save(OUT)
print("saved", OUT, im.size)
