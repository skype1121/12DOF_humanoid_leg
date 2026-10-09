#!/usr/bin/env python3
"""URDF(영자세) 를 정면·측면으로 그리고 1009 변경 부위(hip_r 링크)를 색으로 표시한 한 장 그림."""
import struct, sys, numpy as np, xml.etree.ElementTree as ET
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib import font_manager as fm

URDF = "rl_walking/assets/biped12_final3_hiprpla.urdf"
OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/real_1009/model_change_1009.png"
for f in fm.findSystemFonts():
    if "NotoSansCJK" in f and "Regular" in f:
        fm.fontManager.addfont(f); plt.rcParams["font.family"] = fm.FontProperties(fname=f).get_name(); break

def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr], [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])
def T(o):
    M = np.eye(4)
    if o is not None:
        M[:3, :3] = rpy(*map(float, (o.get("rpy") or "0 0 0").split())); M[:3, 3] = list(map(float, (o.get("xyz") or "0 0 0").split()))
    return M
def stl(f):
    d = open(f, "rb").read(); n = struct.unpack("<I", d[80:84])[0]
    a = np.frombuffer(d[84:84+n*50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3, 3).astype(float)

root = ET.parse(URDF).getroot()
links = {l.get("name"): l for l in root.findall("link")}
world = {}
children = {}
for j in root.findall("joint"):
    children.setdefault(j.find("parent").get("link"), []).append(j)
allc = {j.find("child").get("link") for j in root.findall("joint")}
base = [n for n in links if n not in allc][0]
def walk(n, M):
    world[n] = M
    for j in children.get(n, []):
        walk(j.find("child").get("link"), M @ T(j.find("origin")))   # 영자세(관절각 0)
walk(base, np.eye(4))

CH = ("left_hip_r_joint", "right_hip_r_joint")
HA = {j.find("parent").get("link"): float(j.find("origin").get("xyz").split()[1]) for j in root.findall("joint") if j.get("name") in CH}
tris, cols = [], []
for n, l in links.items():
    for v in l.findall("visual"):
        m = v.find("geometry").find("mesh")
        if m is None: continue
        M = world[n] @ T(v.find("origin"))
        Lm = T(v.find("origin")); tl = stl(m.get("filename")); tl = tl @ Lm[:3, :3].T + Lm[:3, 3]   # 링크 프레임
        t = tl @ world[n][:3, :3].T + world[n][:3, 3]
        yc = tl[:, :, 1].mean(1)                     # 링크 프레임 y(관절 0° 에서 상하, 아래가 −)
        c = np.zeros(len(t), int)
        if n in CH:
            c[yc > -0.012] = 1                       # hip_r 링크의 윗부분 = 회전 모터 몸체
        elif n in HA:
            c[yc < HA[n] + 0.045] = 1                # hip_a 링크의 아랫부분 = 회전 모터를 품은 원통 하우징
        elif n == "right_hip_f_joint":
            c[:] = 2
        tris.append(t); cols.append(c)
tris = np.concatenate(tris); cols = np.concatenate(cols)
ext = tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0)
up = int(np.argmax(ext))                      # 가장 긴 축 = 상하
ax_lr, ax_fb = 1, 0                          # 이 URDF: 앞 = +x(발이 발목 앞으로 더 긴 쪽), 왼쪽 = +y
print("extent", ext.round(3), "up axis", up, "lr", ax_lr, "fb", ax_fb)

nrm = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0]); nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True)+1e-12)
BASE = {0: np.array([0.72, 0.74, 0.78]), 1: np.array([0.95, 0.45, 0.10]), 2: np.array([0.20, 0.55, 0.90])}
def draw(ax, h, depth, sign, title):
    d = sign*tris[:, :, depth].mean(1); o = np.argsort(d)
    light = np.zeros(3); light[depth] = sign*0.8; light[up] = 0.5; light[h] = 0.33; light /= np.linalg.norm(light)
    sh = 0.35+0.65*np.clip(np.abs(nrm @ light), 0, 1)
    fc = np.array([BASE[int(c)] for c in cols])*sh[:, None]
    P = np.stack([tris[:, :, h], tris[:, :, up]], -1)
    ax.add_collection(PolyCollection(P[o], facecolors=fc[o], edgecolors="none", antialiaseds=False))
    ax.set_aspect("equal"); ax.autoscale(); ax.axis("off"); ax.set_title(title, fontsize=15, pad=4)
    return P

fig, axs = plt.subplots(1, 2, figsize=(15, 10.5), gridspec_kw={"width_ratios": [2.3, 1]})
fig.patch.set_facecolor("white")
draw(axs[0], ax_lr, ax_fb, 1, "정면 (그림 왼쪽 = 로봇 오른쪽)")
draw(axs[1], ax_fb, ax_lr, -1, "오른쪽 측면 (앞 →)")
def cen(name, h):
    M = world[name]; return M[h, 3], M[up, 3]
a = axs[0]
xmin, xmax = a.get_xlim(); ymin, ymax = a.get_ylim()
for name, side, txt in (("left_hip_r_joint", 1, "왼 골반 회전 모터(3번) + 그 하우징\n→ PLA 통짜 출력물 211 g, 0° 고정\n무릎 쪽 하우징은 그대로"),
                        ("right_hip_r_joint", -1, "오른 골반 회전 모터(9번) + 그 하우징\n→ PLA 통짜 출력물 211 g, 0° 고정\n무릎 쪽 하우징은 그대로")):
    x, y = cen(name, ax_lr); s = np.sign(x) if x != 0 else side
    a.annotate(txt, xy=(x, y+0.005), xytext=(x+s*0.24, y-0.10), fontsize=12, ha="left" if s > 0 else "right", va="center",
               arrowprops=dict(arrowstyle="-|>", color="#c2410c", lw=1.8), bbox=dict(boxstyle="round,pad=0.4", fc="#fff3e6", ec="#c2410c"))
x, y = cen("right_hip_f_joint", ax_lr); s = np.sign(x) if x != 0 else -1
a.annotate("오른 골반 굽힘 (hip_f), 모터 7\n고장 → 3번 모터를 옮겨 ID 7\n질량 변화 없음, 영점 재캡처 필요", xy=(x, y), xytext=(x+s*0.24, y+0.16), fontsize=12,
           ha="left" if s > 0 else "right", va="center", arrowprops=dict(arrowstyle="-|>", color="#1d4ed8", lw=1.8),
           bbox=dict(boxstyle="round,pad=0.4", fc="#e8f1ff", ec="#1d4ed8"))
a.set_xlim(xmin-0.62, xmax+0.62)
fig.suptitle("12 자유도 모델 — 1009 변경 부위", fontsize=20, y=0.975)
fig.text(0.5, 0.045, "주황 = PLA 통짜로 바뀌는 부분(회전 모터 + 그 하우징)   파랑 = 모터만 교체   회색 = 변경 없음\n"
         "URDF 에서는 주황이 두 링크에 걸쳐 있다(하우징 = hip_a 링크 아랫부분, 모터 = hip_r 링크 윗부분) — 질량 값은 기존 하우징 무게 확인 뒤 확정   ·   관절각 0° 자세",
         ha="center", fontsize=12.5, linespacing=1.6)
fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.10, wspace=0.0)
fig.savefig(OUT, dpi=110); print("saved", OUT, len(tris), "tris")
