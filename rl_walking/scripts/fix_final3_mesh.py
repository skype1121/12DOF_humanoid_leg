"""final3_urdf(2026-09-03 재익스포트)의 right_hip_r 메시 프레임 수복.

final1 에서 right_hip_r 은 관절·관성·메시가 전부 무릎 좌표계였다(fix_final1_frames.py).
final3 에서 SW 트리를 고쳐 관절 원점과 관성 원점은 힙 좌표계로 왔지만,
**STL 메시만 옛 무릎 좌표계 그대로** 익스포트됐다:
  좌 hip_r STL y[-208, +50]  vs  우 hip_r STL y[-50, +208]  (x 도 +19.85mm 어긋남)
→ 우측 허벅지 메시가 힙 위로 솟아 hip_f 와 13kN 관입 (프로브 실측).

수복: visual/collision origin 을 O2=(+0.01985, -0.158275, 0) 로 (무릎→힙 프레임 병진).
관성은 이미 힙 프레임(좌우 미러 일치)이므로 건드리지 않는다.
검증: 이동 후 메시 바운드가 좌측의 x-미러와 1mm 이내로 일치해야 함.

실행: python3 rl_walking/scripts/fix_final3_mesh.py
출력: final3_urdf/urdf/final3_urdf_fixed.urdf
"""
import os
import struct
import xml.etree.ElementTree as ET

REPO = "/home/ryu/humanoid_leg_test1"
SRC = os.path.join(REPO, "final3_urdf", "urdf", "final3_urdf.urdf")
OUT = os.path.join(REPO, "final3_urdf", "urdf", "final3_urdf_fixed.urdf")
MESH = os.path.join(REPO, "final3_urdf", "meshes")
O2 = (0.01985, -0.158275, 0.0)


def stl_bounds(fn):
    d = open(fn, "rb").read()
    n = struct.unpack("<I", d[80:84])[0]
    vs, off = [], 84
    for _ in range(n):
        rec = struct.unpack("<12fH", d[off:off + 50]); off += 50
        vs += [rec[3:6], rec[6:9], rec[9:12]]
    xs, ys, zs = zip(*vs)
    return [(min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))]


tree = ET.parse(SRC)
root = tree.getroot()
link = next(l for l in root.findall("link") if l.get("name") == "right_hip_r_link")
rep = []
for tag in ("visual", "collision"):
    o = link.find(tag).find("origin")
    before = o.get("xyz")
    assert before.split() == ["0", "0", "0"], f"예상 밖 {tag} origin {before} — 이미 수복됐거나 다른 익스포트"
    o.set("xyz", " ".join(f"{v:.10g}" for v in O2))
    rep.append(f"right_hip_r_link/{tag} origin {before} → {o.get('xyz')}")

# 검증: 이동 후 바운드 == 좌측 x-미러
bl = stl_bounds(os.path.join(MESH, "left_hip_r_link.STL"))
br = stl_bounds(os.path.join(MESH, "right_hip_r_link.STL"))
br_shift = [(lo + d, hi + d) for (lo, hi), d in zip(br, O2)]
mirror_l = [(-bl[0][1], -bl[0][0]), bl[1], bl[2]]
for ax, a, b in zip("xyz", br_shift, mirror_l):
    err = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    rep.append(f"  검증 {ax}: 우(이동후)[{a[0]*1000:+.1f},{a[1]*1000:+.1f}] vs 좌미러[{b[0]*1000:+.1f},{b[1]*1000:+.1f}] 차 {err*1000:.2f}mm")
    assert err < 0.002, f"미러 검증 실패 {ax}: {err*1000:.2f}mm"

ET.indent(tree, space="  ")
tree.write(OUT, encoding="utf-8", xml_declaration=True)
rep.append(f"출력: {OUT}")
print("\n".join(rep))
