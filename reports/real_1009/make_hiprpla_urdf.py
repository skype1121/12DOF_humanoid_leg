#!/usr/bin/env python3
"""변형 URDF 생성(1009 정정판): hip_r 모터(실측 611 g)와 그 원통 하우징(실측 288 g)을 PLA 통짜 출력물(211.23 g)로 대체.

URDF 에서 하우징은 hip_a 링크의 아랫부분, 모터는 hip_r 링크의 윗부분이다. 그래서 두 링크를 고친다:
  hip_a 링크: − 하우징 288 g + 출력물 211.23 g   (출력물은 hip_a 쪽에 붙은 것으로 둔다 — 관절이 굳으므로 어느 쪽이든 같다)
  hip_r 링크: − 모터 611 g
빠지는/더해지는 물체는 메시의 해당 부분 경계 상자에서 얻은 원통으로 근사한다(가정):
  모터 = 속이 찬 원통, 하우징 = 속 빈 원통(안지름 = 모터 지름), 출력물 = 하우징 외형의 속이 찬 원통.
"""
import re, struct, sys, numpy as np, xml.etree.ElementTree as ET
SRC = "rl_walking/assets/biped12_final3.urdf"; DST = "rl_walking/assets/biped12_final3_hiprpla.urdf"
M_MOTOR, M_HOUSING, M_PRINT = 0.611, 0.288, 0.21123
def stl(f):
    d = open(f, "rb").read(); n = struct.unpack("<I", d[80:84])[0]
    a = np.frombuffer(d[84:84+n*50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3, 3).astype(float)
def cyl(m, r_out, h, r_in=0.0):          # 축 = y, 중심 기준 관성(대각)
    iy = 0.5*m*(r_out**2+r_in**2); ix = m*(3*(r_out**2+r_in**2)+h**2)/12
    return np.diag([ix, iy, ix])
def shift(I, m, d):                        # 질량 중심 기준 I → d 만큼 떨어진 점 기준
    return I + m*(d@d*np.eye(3)-np.outer(d, d))
def combine(parts):                        # [(부호, m, com, Icom)] → m, com, Icom
    m = sum(s*p for s, p, _, _ in parts); c = sum(s*p*np.asarray(q) for s, p, q, _ in parts)/m
    I = sum(s*shift(Ic, p, np.asarray(q)-c) for s, p, q, Ic in parts)
    return m, c, I
root = ET.parse(SRC).getroot(); L = {l.get("name"): l for l in root.findall("link")}
def inert(n):
    i = L[n].find("inertial"); a = i.find("inertia").attrib
    I = np.array([[a["ixx"], a["ixy"], a["ixz"]], [a["ixy"], a["iyy"], a["iyz"]], [a["ixz"], a["iyz"], a["izz"]]], float)
    return float(i.find("mass").get("value")), np.array(i.find("origin").get("xyz").split(), float), I
def mesh(n):
    v = L[n].find("visual"); o = np.array((v.find("origin").get("xyz") or "0 0 0").split(), float)
    return stl(v.find("geometry").find("mesh").get("filename")) + o
s = open(SRC).read(); rep = {}
for side in ("left", "right"):
    ha, hr = f"{side}_hip_a_joint", f"{side}_hip_r_joint"
    yj = [float(j.find("origin").get("xyz").split()[1]) for j in root.findall("joint") if j.get("name") == hr][0]
    ta = mesh(ha); ta = ta[ta[:, :, 1].mean(1) < yj+0.045].reshape(-1, 3)      # 하우징 부분(hip_a 프레임)
    tr = mesh(hr); tr = tr[tr[:, :, 1].mean(1) > -0.012].reshape(-1, 3)        # 모터 부분(hip_r 프레임)
    hc = (ta.min(0)+ta.max(0))/2; hh = np.ptp(ta[:, 1]); hro = np.percentile(np.hypot(ta[:, 0]-hc[0], ta[:, 2]-hc[2]), 90)
    mc = (tr.min(0)+tr.max(0))/2; mh = np.ptp(tr[:, 1]); mr = np.percentile(np.hypot(tr[:, 0]-mc[0], tr[:, 2]-mc[2]), 98)
    print(f"{side}: 하우징 중심 {hc.round(4)} 높이 {hh*1000:.0f} mm 바깥 r {hro*1000:.0f} mm | 모터 중심 {mc.round(4)} 높이 {mh*1000:.0f} mm r {mr*1000:.0f} mm")
    m, c, I = inert(ha)
    rep[ha] = combine([(1, m, c, I), (-1, M_HOUSING, hc, cyl(M_HOUSING, hro, hh, min(mr, hro*0.95))), (1, M_PRINT, hc, cyl(M_PRINT, hro, hh))])
    m, c, I = inert(hr)
    rep[hr] = combine([(1, m, c, I), (-1, M_MOTOR, mc, cyl(M_MOTOR, mr, mh))])
for n, (m, c, I) in rep.items():
    ev = np.linalg.eigvalsh(I); m0 = inert(n)[0]
    ok = m > 0 and ev.min() > 0 and ev[0]+ev[1] >= ev[2]*0.999
    print(f"{n}: {m0:.4f} → {m:.4f} kg  com {c.round(5)}  주관성 {ev}  {'OK' if ok else '물리적으로 불가!'}")
    assert ok, n
    k = re.search(r'(<link name="%s">\s*<inertial>)(.*?)(</inertial>)' % n, s, re.S)
    body = ('\n      <origin xyz="%.9g %.9g %.9g" rpy="0 0 0" />\n      <mass value="%.9g" />\n      '
            '<inertia ixx="%.9e" ixy="%.9e" ixz="%.9e" iyy="%.9e" iyz="%.9e" izz="%.9e" />\n    ') % (*c, m, I[0, 0], I[0, 1], I[0, 2], I[1, 1], I[1, 2], I[2, 2])
    s = s[:k.start(2)]+body+s[k.end(2):]
hdr = ('<!-- 변형본(1009 정정판): hip_r 모터(실측 611 g)와 그 원통 하우징(실측 288 g)을 PLA 통짜 출력물(211.23 g)로 대체.\n'
       '     원본 biped12_final3.urdf 에서 좌·우 hip_a 링크(− 하우징 + 출력물)와 hip_r 링크(− 모터)의 inertial 만 바꿨다.\n'
       '     빠지는·더해지는 물체는 원통 근사(가정). 생성: reports/real_1009/make_hiprpla_urdf.py. 관절 정의·메시는 원본과 동일. -->\n')
i = s.index("<robot"); open(DST, "w").write(s[:i]+hdr+s[i:])
tot = lambda p: sum(float(l.find("inertial").find("mass").get("value")) for l in ET.parse(p).getroot().findall("link") if l.find("inertial") is not None)
print("전체 %.4f → %.4f kg" % (tot(SRC), tot(DST)))
