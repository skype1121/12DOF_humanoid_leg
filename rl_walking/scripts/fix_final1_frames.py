"""final1_urdf(2026-09-03 SW 익스포트)의 right_hip_r 프레임 붕괴 수복.

증상: 익스포터 트리에서 right_hip_r 관절에 무릎 자리 좌표계·무릎 축이 잡혀
  - right_hip_r_joint origin (0.01985,-0.28232,-0.02035) = 힙→무릎 총거리, axis (1,0,0)
  - right_knee_joint origin (0,0,0)
  → 우측 힙 요(yaw) DOF 소실, 무릎 피치 중복.

수복 (구 biped12_base.urdf 값과 좌측 미러로 확정):
  - right_hip_r_joint origin → (0,-0.12405,-0.02), axis → (0,1,0)
  - right_knee_joint  origin → (0.01985,-0.15827,-0.00035)   [합 불변 검증]
  - right_hip_r_link 의 visual/collision/inertial 원점을 +O2 평행이동
    (프레임이 무릎→힙으로 올라가므로; 회전 없음 → 관성텐서 불변)
  검증: 이동 후 inertial CoM 이 좌측과 x부호만 반전으로 일치해야 함.

부수: D455 질량 0 → 0.103 kg(실물 근사, 박스 관성) — CoM 전방 기여 복원.

실행: python3 rl_walking/scripts/fix_final1_frames.py
출력: final1_urdf/urdf/final1_urdf_fixed.urdf
"""
import os
import xml.etree.ElementTree as ET

REPO = "/home/ryu/humanoid_leg_test1"
SRC = os.path.join(REPO, "urdf", "final1_urdf", "urdf", "final1_urdf.urdf")
OUT = os.path.join(REPO, "urdf", "final1_urdf", "urdf", "final1_urdf_fixed.urdf")

O_OLD = (0.01985, -0.28232, -0.02035)   # 붕괴 상태의 hip_a→hip_r
O1 = (0.0, -0.12405, -0.02)             # 수복: hip_a→hip_r (구 자산과 동일)
O2 = (0.01985, -0.15827, -0.00035)      # 수복: hip_r→knee  (구 자산과 동일)
assert all(abs(a - (b + c)) < 1e-12 for a, b, c in zip(O_OLD, O1, O2)), "합 불변 위반"

tree = ET.parse(SRC)
root = tree.getroot()
joints = {j.get("name"): j for j in root.findall("joint")}
links = {l.get("name"): l for l in root.findall("link")}
rep = []


def set_origin(el, xyz):
    o = el.find("origin")
    o.set("xyz", " ".join(f"{v:.10g}" for v in xyz))


def get_origin(el):
    return tuple(float(v) for v in el.find("origin").get("xyz").split())


# 1) 관절 프레임 수복
j_hip = joints["right_hip_r_joint"]
got = get_origin(j_hip)
assert all(abs(a - b) < 1e-9 for a, b in zip(got, O_OLD)), f"예상 밖 소스 원점 {got} — 이미 수복됐거나 다른 익스포트"
set_origin(j_hip, O1)
j_hip.find("axis").set("xyz", "0 1 0")
rep.append(f"right_hip_r_joint: origin {got} → {O1}, axis (1,0,0) → (0,1,0)")

j_knee = joints["right_knee_joint"]
assert get_origin(j_knee) == (0.0, 0.0, 0.0), "예상 밖 무릎 원점"
set_origin(j_knee, O2)
rep.append(f"right_knee_joint: origin (0,0,0) → {O2}")

# 2) right_hip_r_link 기하 재표현 (+O2 평행이동)
link = links["right_hip_r_link"]
for tag in ("visual", "collision", "inertial"):
    el = link.find(tag)
    old = get_origin(el)
    new = tuple(a + b for a, b in zip(old, O2))
    set_origin(el, new)
    rep.append(f"right_hip_r_link/{tag}: {tuple(round(v,6) for v in old)} → {tuple(round(v,6) for v in new)}")

# 검증: 좌측과 미러 (x부호 반전, y·z 근사 일치)
com_r = get_origin(link.find("inertial"))
com_l = get_origin(links["left_hip_r_link"].find("inertial"))
assert abs(com_r[0] + com_l[0]) < 1e-3, f"미러 실패 x: {com_r[0]} vs {com_l[0]}"
assert abs(com_r[1] - com_l[1]) < 1e-3, f"미러 실패 y: {com_r[1]} vs {com_l[1]}"
rep.append(f"미러 검증 OK: 우 CoM {tuple(round(v,6) for v in com_r)} ↔ 좌 {tuple(round(v,6) for v in com_l)}")

# 3) D455 질량 근사 주입 (0 → 0.103 kg, 124×26×29mm 박스 관성)
d455 = links["D455"].find("inertial")
if float(d455.find("mass").get("value")) == 0.0:
    m, a, b, c = 0.103, 0.124, 0.026, 0.029
    d455.find("mass").set("value", str(m))
    inertia = d455.find("inertia")
    inertia.set("ixx", f"{m/12*(b*b+c*c):.6g}")
    inertia.set("iyy", f"{m/12*(a*a+c*c):.6g}")
    inertia.set("izz", f"{m/12*(a*a+b*b):.6g}")
    rep.append(f"D455: 질량 0 → {m} kg (실물 근사; 박스 관성)")

ET.indent(tree, space="  ")
tree.write(OUT, encoding="utf-8", xml_declaration=True)
rep.append(f"출력: {OUT}")
print("\n".join(rep))
