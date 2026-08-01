"""URDF 시상면 질량 분석 — 스탠드 자세 무게중심 vs 발목 지지축 (2026-08-02).

배경: 실물 첫 자립 기립(0801)에서 자연 평형이 '뒤로 5.7° (발목 트림 2.5°)'로
실측됨 = 실물 CoM이 발목 축보다 2~4cm 뒤. 학습 URDF는 CoM이 발목 위 0.25cm로
사실상 균형 — 이 갭(CAD에 없는 젯슨·보드·배선 ≈1.51kg, 몸통 뒤·위 장착)이
심투리얼 균형 갭의 주범. 이 도구는 payload 캘리브레이션 루프의 판정계:
make_base_urdf.py --payload-* 로 생성 → com_report.py 로 CoM 확인 → Isaac 평형
재현(뒤 5.7°@트림2.5)으로 최종 정합.

좌표 규약 (base 프레임): +X=앞(발끝 여유 9.5 vs 뒤꿈치 8.0cm로 검증), +Z=위.
Isaac 월드(전방=-Y, 좌=+X)와는 Z축 -90° 회전 관계 — joint-direction 검증과 무모순.

실행: python3 rl_walking/scripts/com_report.py [--urdf 경로] [--pose stand|zero]
  stand = 정책 기본자세 (무릎 ±6°, 발목F ±1.4° — 부호는 좌우 반대 규약)
"""
import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_URDF = "/home/ryu/humanoid_leg_test1/rl_walking/assets/biped12_base.urdf"


def vec(s, default="0 0 0"):
    return np.array([float(x) for x in (s or default).split()])


def rot_rpy(rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0],
                   [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)],
                   [0, 1, 0],
                   [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0],
                   [math.sin(y), math.cos(y), 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def rot_axis(axis, th):
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


def _load_policy_pose():
    """정책 기본자세를 소스에서 직접 로드 (URDF 관절 프레임 rad).

    실패 시 하드코딩 폴백 — 검증값(2026-08-02, policy_runner.DEFAULT_POSE_RAD):
    left_knee −6°/right_knee +6° (URDF 부호는 raw 모터와 달리 left +=폄),
    left_ankle_f +1.4°/right −1.4°. 손부호로 무릎을 반대로 굽힌 사고 재발 방지.
    """
    try:
        import sys
        sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
        from rl_walking.deploy.policy_runner import (DEFAULT_POSE_RAD,
                                                     JOINT_ORDER)
        return {j: float(r) for j, r in zip(JOINT_ORDER, DEFAULT_POSE_RAD)}
    except Exception:
        pose = {}
        for side, s in (("left", 1.0), ("right", -1.0)):
            pose[f"{side}_knee_joint"] = math.radians(-6.0 * s)
            pose[f"{side}_ankle_f_joint"] = math.radians(1.4 * s)
        return pose


POLICY_POSE = _load_policy_pose()


def stand_angle_rad(joint_name):
    return POLICY_POSE.get(joint_name, 0.0)


def analyze(urdf_path, pose="stand"):
    root = ET.parse(urdf_path).getroot()
    links = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            links[link.get("name")] = (0.0, np.zeros(3))
            continue
        origin = inertial.find("origin")
        links[link.get("name")] = (
            float(inertial.find("mass").get("value")),
            vec(origin.get("xyz") if origin is not None else None))

    joints = []
    for j in root.findall("joint"):
        origin = j.find("origin")
        axis = j.find("axis")
        joints.append({
            "name": j.get("name"), "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": vec(origin.get("xyz") if origin is not None else None),
            "rpy": vec(origin.get("rpy") if origin is not None else None),
            "axis": vec(axis.get("xyz") if axis is not None else None, "0 0 1"),
        })

    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    roots = set(links) - {j["child"] for j in joints}
    world = {r: (np.eye(3), np.zeros(3)) for r in roots}
    stack = list(roots)
    while stack:
        parent = stack.pop()
        rp, tp = world[parent]
        for j in children.get(parent, []):
            r = rp @ rot_rpy(j["rpy"])
            t = tp + rp @ j["xyz"]
            if j["type"] in ("revolute", "continuous"):
                ang = stand_angle_rad(j["name"]) if pose == "stand" else 0.0
                r = r @ rot_axis(j["axis"], ang)
            world[j["child"]] = (r, t)
            stack.append(j["child"])

    total_m, com = 0.0, np.zeros(3)
    rows = []
    for name, (m, c) in links.items():
        if m <= 0:
            continue
        r, t = world[name]
        p = t + r @ c
        total_m += m
        com += m * p
        rows.append((m, name, p))
    com /= total_m

    ankle_x = np.mean([world[j["child"]][1][0] for j in joints
                       if "ankle_f" in j["name"]])
    return total_m, com, ankle_x, sorted(rows, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--pose", default="stand", choices=("stand", "zero"))
    a = ap.parse_args()
    total_m, com, ankle_x, rows = analyze(a.urdf, a.pose)
    print(f"URDF: {a.urdf} (자세: {a.pose})")
    print(f"총질량 {total_m:.3f} kg | CoM base=({com[0]:+.4f}, {com[1]:+.4f}, "
          f"{com[2]:+.4f}) m")
    offset_cm = (com[0] - ankle_x) * 100
    tag = "앞" if offset_cm > 0 else "뒤"
    print(f"발목 지지축 x={ankle_x:+.4f} → CoM은 발목 대비 {tag} "
          f"{abs(offset_cm):.2f} cm")
    print(f"(실물 실측 목표: 뒤 2~4cm — 0801 자연평형 뒤 5.7°@트림2.5 역산)")
    print("\n링크별 (질량 내림차순):")
    for m, name, p in rows:
        print(f"  {m:6.3f} kg  {name:22s} x={p[0]:+.4f}  z={p[2]:+.4f}")


if __name__ == "__main__":
    main()
