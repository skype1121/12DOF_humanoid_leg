"""1002 측면 영상 마커 추적 (60 fps): 노란 커넥터 보드 3군(허벅지 아래·무릎·발목) 중심 → 영상 평면 정강이 각·허벅지-무릎 각.
영상 시각 = 블랙박스 + 8 s (승윤님: 영상 8 s 에 정책 시작). 출력 CSV: t_bb, 발목·무릎·허벅지 중심(px), 정강이각(deg, 영상 평면, 아래쪽 끝이 화면 왼쪽 + ).
원근·카메라 기울기 보정 없음 → 절대값이 아니라 변화량(조용한 서기 기준)만 쓴다."""
import subprocess, sys, numpy as np, cv2, csv
V = sys.argv[1]; T0, T1 = float(sys.argv[2]), float(sys.argv[3]); OUT = sys.argv[4]
W, H = 540, 960
cmd = ['ffmpeg', '-v', 'error', '-ss', str(T0), '-t', str(T1 - T0), '-i', V, '-vf', f'scale={W}:{H}', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
rows = []; k = 0; fps = 60000 / 1001
prev = None
while True:
    buf = p.stdout.read(W * H * 3)
    if len(buf) < W * H * 3: break
    im = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, (18, 110, 140), (35, 255, 255)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    pts = [(cen[i][0], cen[i][1], st[i, 4]) for i in range(1, n) if st[i, 4] >= 60 and cen[i][1] > 0.45 * H]
    pts.sort(key=lambda q: q[1])
    # y 간격 > 30 px(반축소) 로 군집
    cl = []
    for q in pts:
        if cl and q[1] - cl[-1][-1][1] < 30: cl[-1].append(q)
        else: cl.append([q])
    cent = [(sum(a * x for x, y, a in c) / sum(a for *_, a in c), sum(a * y for x, y, a in c) / sum(a for *_, a in c), sum(a for *_, a in c)) for c in cl]
    cent = [c for c in cent if c[2] >= 150]
    t_bb = T0 + k / fps - 8.0; k += 1
    if len(cent) >= 3:
        th, kn, an = cent[-3], cent[-2], cent[-1]
        shank = np.degrees(np.arctan2(kn[0] - an[0], an[1] - kn[1]))   # 아래 끝(발목)이 무릎보다 화면 왼쪽이면 +
        thk = np.degrees(np.arctan2(th[0] - kn[0], kn[1] - th[1]))
        rows.append([round(t_bb, 4), *[round(v, 1) for v in (th[0], th[1], kn[0], kn[1], an[0], an[1])], round(shank, 2), round(thk, 2)])
    else:
        rows.append([round(t_bb, 4)] + [''] * 8)
with open(OUT, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['t_bb', 'th_x', 'th_y', 'kn_x', 'kn_y', 'an_x', 'an_y', 'shank_deg', 'thigh_knee_deg']); w.writerows(rows)
ok = sum(1 for r in rows if r[1] != '')
print(f'프레임 {len(rows)} · 3군 검출 {ok} ({100*ok/max(len(rows),1):.0f} %)')
