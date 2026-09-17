"""URDF 기구학·질량으로 자세별 기하 분석: 발 평평 기준 몸통 피치, 전신 CoM vs 발목축/발끝/뒤꿈치, 높이."""
import math, sys, xml.etree.ElementTree as ET, numpy as np
sys.path.insert(0, "/home/ryu/humanoid_leg_test1/rl_walking/scripts")
from com_report import vec, rot_rpy, rot_axis
POSES = {
  "zero":    {},
  "default": {"left_knee_joint": -6.0, "right_knee_joint": 6.0, "left_ankle_f_joint": 1.4, "right_ankle_f_joint": -1.4},
  "v1_0914": {"left_hip_f_joint": 16.5, "right_hip_f_joint": -16.5, "left_knee_joint": -30, "right_knee_joint": 30, "left_ankle_f_joint": 19, "right_ankle_f_joint": -19},
  "v2_0914": {"left_hip_f_joint": 20.5, "right_hip_f_joint": -20.5, "left_knee_joint": -30, "right_knee_joint": 30, "left_ankle_f_joint": 19, "right_ankle_f_joint": -19},
  "v3_0917": {"left_hip_f_joint": 15.0, "right_hip_f_joint": -15.0, "left_knee_joint": -30, "right_knee_joint": 30, "left_ankle_f_joint": 20.5, "right_ankle_f_joint": -20.5},
}
def load(urdf):
    root = ET.parse(urdf).getroot(); links = {}
    for l in root.findall("link"):
        i = l.find("inertial")
        if i is None: links[l.get("name")] = (0.0, np.zeros(3)); continue
        o = i.find("origin"); links[l.get("name")] = (float(i.find("mass").get("value")), vec(o.get("xyz") if o is not None else None))
    joints = [{"name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
               "xyz": vec((j.find("origin").get("xyz") if j.find("origin") is not None else None)), "rpy": vec((j.find("origin").get("rpy") if j.find("origin") is not None else None)),
               "axis": vec(j.find("axis").get("xyz") if j.find("axis") is not None else None, "0 0 1")} for j in root.findall("joint")]
    return links, joints
def fk(links, joints, pose_deg):
    children = {}
    for j in joints: children.setdefault(j["parent"], []).append(j)
    roots = set(links) - {j["child"] for j in joints}
    world = {r: (np.eye(3), np.zeros(3)) for r in roots}; stack = list(roots)
    while stack:
        par = stack.pop(); rp, tp = world[par]
        for j in children.get(par, []):
            r = rp @ rot_rpy(j["rpy"]); t = tp + rp @ j["xyz"]
            if j["type"] in ("revolute", "continuous"): r = r @ rot_axis(j["axis"], math.radians(pose_deg.get(j["name"], 0.0)))
            world[j["child"]] = (r, t); stack.append(j["child"])
    return world
REAL_FWD = {"v1_0914": -2.0, "v2_0914": +1.5, "v3_0917": -0.4}   # 실물 IMU fwd(+앞) 실측
def analyze(urdf, tag):
    links, joints = load(urdf)
    print(f"\n=== {tag}: {urdf.split('/')[-1]}")
    print("  자세      URDF몸통피치(발평평)  실물IMU   차이  | CoM(발목축 기준, 발평평)  여유 앞/뒤 | 실물추정(차이를 발목에서 흡수) CoM  여유 앞/뒤 | CoM높이")
    for pname, pose in POSES.items():
        w = fk(links, joints, pose)
        m_tot = 0.0; com = np.zeros(3)
        for n, (m, c) in links.items():
            if m <= 0: continue
            r, t = w[n]; com += m * (t + r @ c); m_tot += m
        com /= m_tot
        foot = [j["child"] for j in joints if "ankle_r_joint" in j["name"]]
        rf, tf = w[foot[0]]
        up = rf[:, 1]                                     # 발 프레임 y축 = 영자세에서 base +Z(위)
        phi = math.atan2(-up[0], up[2])                   # 발바닥을 수평으로 만드는 Y축 회전각(+ = 몸통 앞으로 숙임)
        R = rot_axis(np.array([0, 1, 0]), phi)
        assert abs((R @ up)[0]) < 1e-6
        comf = R @ com; ankle = np.mean([R @ w[j["child"]][1] for j in joints if "ankle_f" in j["name"]], axis=0); footp = np.mean([R @ w[f][1] for f in foot], axis=0)
        dx = (comf[0] - ankle[0]) * 100; pitch = math.degrees(phi)
        real = REAL_FWD.get(pname)
        if real is not None:
            # 실물은 발 평평·몸통 real 피치. URDF 와의 차이 d 를 발목에서 흡수(발 위 전체가 d 만큼 뒤로 기움)한 경우의 CoM
            d = math.radians(real - pitch)
            Rd = rot_axis(np.array([0, 1, 0]), d)
            com_r = ankle + Rd @ (comf - ankle); dxr = (com_r[0] - ankle[0]) * 100
            rtxt = f"{real:+5.1f}° {real-pitch:+5.1f}° | 실물추정 CoM {'앞' if dxr>0 else '뒤'} {abs(dxr):4.1f} cm  여유 {8.5-dxr:4.1f}/{6.5+dxr:4.1f}"
        else:
            rtxt = "   —      —  |"
        print(f"  {pname:8s} {pitch:+5.1f}° {rtxt} | URDF CoM {'앞' if dx>0 else '뒤'} {abs(dx):4.1f} cm 여유 {8.5-dx:4.1f}/{6.5+dx:4.1f} | 높이 {(comf[2]-footp[2])*100:4.1f} cm | 질량 {m_tot:.2f}")
for u, t in (("/home/ryu/humanoid_leg_test1/rl_walking/assets/biped12_final3.urdf", "학습 자산 final3 (17.62 kg)"),):
    analyze(u, t)
