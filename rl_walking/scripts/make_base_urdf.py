"""SolidWorks 익스포트 URDF → RL 파이프라인용 biped12_base.urdf 생성.

이식가이드 §4-1의 1단계. 변환 내용:
  1. robot name → "biped12"
  2. 링크/관절명 *_Link → *_joint (구 자산 관례 — env 정규식·접촉센서가 이 이름 기준)
     (0725 익스포트의 left_hip_r "관절"명이 left_hip_r_Link로 잘못 나온 버그도 이걸로 해소)
  3. 메시 경로 package://... → 소스 패키지 절대경로
  4. revolute+제로리밋 → continuous·리밋태그 제거 (검증된 임포트 경로 재현;
     실측 리밋은 postprocess_usd.py가 USD에 주입)
  5. 관절 축을 검증 규약(ANAT_SIGN·sim_joint_limits.json 부호 기준)과 대조,
     반전돼 있으면 복원 + 리포트 (0725: left_hip_f가 +1,0,0으로 뒤집혀 나옴)
  6. base 링크 + 고정조인트(rpy π/2,0,π/2: SolidWorks Y-up → Isaac Z-up) 주입
  7. (옵션) --payload-mass/--payload-xyz: CAD에 없는 실물 탑재물(젯슨·보드·배선
     ≈1.51kg = 실물 12.0 − URDF 10.49)을 base 프레임 좌표(+X=앞, +Z=위)에 고정
     링크로 주입. 배터리 3.3kg은 CAD 골반에 이미 포함(골반 CoM 뒤 3.5cm 실증) —
     중복 주입 금지. payload 사용 시 --out을 별도 파일로 지정할 것 (기본 자산
     biped12_base.urdf는 현행 정책의 학습 재현성 보존을 위해 불변 유지).

실행: python3 rl_walking/scripts/make_base_urdf.py [--src <pkg>/urdf/xxx.urdf]
     [--out <출력.urdf>] [--payload-mass 1.51 --payload-xyz "-0.15 0 0.10"]
"""
import argparse
import os
import xml.etree.ElementTree as ET

REPO = "/home/ryu/humanoid_leg_test1"
DEFAULT_SRC = os.path.join(REPO, "12URDF0725", "urdf", "12URDF.urdf")
OUT = os.path.join(REPO, "rl_walking", "assets", "biped12_base.urdf")

# 검증된 부호 규약 (joint_direction 실측 + log_picture/05 무중력 프로브 합격 자산 기준)
EXPECTED_AXES = {
    "left_hip_f_joint": (-1, 0, 0), "right_hip_f_joint": (1, 0, 0),
    "left_hip_a_joint": (0, 0, -1), "right_hip_a_joint": (0, 0, -1),
    "left_hip_r_joint": (0, 1, 0), "right_hip_r_joint": (0, 1, 0),
    "left_knee_joint": (-1, 0, 0), "right_knee_joint": (1, 0, 0),
    "left_ankle_f_joint": (-1, 0, 0), "right_ankle_f_joint": (1, 0, 0),
    "left_ankle_r_joint": (0, 0, 1), "right_ankle_r_joint": (0, 0, -1),
}

parser = argparse.ArgumentParser()
parser.add_argument("--src", default=DEFAULT_SRC)
parser.add_argument("--out", default=OUT)
parser.add_argument("--payload-mass", type=float, default=None,
                    help="미모델 탑재물 질량 kg (예: 1.51). 없으면 주입 안 함")
parser.add_argument("--payload-xyz", default="-0.15 0 0.10",
                    help="payload CoM의 base 프레임 좌표 'x y z' m (+X=앞, +Z=위)")
args = parser.parse_args()

if args.payload_mass is not None and os.path.abspath(args.out) == OUT:
    raise SystemExit("[중단] payload 주입본을 기본 자산(biped12_base.urdf)에 덮어쓰기 "
                     "금지 — 현행 정책 학습 재현성 보존. --out 을 별도 파일로 지정")

mesh_dir = os.path.abspath(os.path.join(os.path.dirname(args.src), "..", "meshes"))
assert os.path.isdir(mesh_dir), f"메시 폴더 없음: {mesh_dir}"

tree = ET.parse(args.src)
root = tree.getroot()
report = [f"소스: {args.src}"]

root.set("name", "biped12")


def to_joint_name(n):
    return n[:-5] + "_joint" if n.endswith("_Link") else n


# 2) 이름 통일 (링크명 + 관절명 + parent/child 참조)
for el in root.iter():
    if el.tag in ("link", "joint") and el.get("name", "").endswith("_Link"):
        report.append(f"이름 변경: {el.tag} {el.get('name')} → {to_joint_name(el.get('name'))}")
        el.set("name", to_joint_name(el.get("name")))
    if el.tag in ("parent", "child"):
        el.set("link", to_joint_name(el.get("link")))

# 3) 메시 절대경로
for mesh in root.iter("mesh"):
    fn = os.path.basename(mesh.get("filename"))
    path = os.path.join(mesh_dir, fn)
    assert os.path.isfile(path), f"메시 없음: {path}"
    mesh.set("filename", path)

# 4) continuous 변환 + 리밋 제거, 5) 축 규약 검사·복원
for joint in root.findall("joint"):
    name = joint.get("name")
    if joint.get("type") == "revolute":
        joint.set("type", "continuous")
        for lim in joint.findall("limit"):
            joint.remove(lim)
    axis_el = joint.find("axis")
    if axis_el is None or name not in EXPECTED_AXES:
        continue
    axis = tuple(int(float(v)) for v in axis_el.get("xyz").split())
    exp = EXPECTED_AXES[name]
    if axis != exp:
        axis_el.set("xyz", f"{exp[0]} {exp[1]} {exp[2]}")
        report.append(f"★ 축 복원: {name} {axis} → {exp} (규약 반전 익스포트 교정)")

# 6) base 링크 주입 (pelvis 앞에)
base_link = ET.fromstring(
    '<link name="base"><inertial><origin xyz="0 0 0" rpy="0 0 0"/>'
    '<mass value="0.001"/><inertia ixx="1e-06" ixy="0" ixz="0" iyy="1e-06" iyz="0" izz="1e-06"/>'
    "</inertial></link>"
)
base_joint = ET.fromstring(
    '<joint name="base_to_pelvis" type="fixed">'
    '<origin xyz="0 0 0" rpy="1.5707963267948966 0 1.5707963267948966"/>'
    '<parent link="base"/><child link="pelvis"/></joint>'
)
root.insert(0, base_joint)
root.insert(0, base_link)

# 7) (옵션) 미모델 탑재물 주입 — base 프레임(+X=앞) 고정 링크
if args.payload_mass is not None:
    m = args.payload_mass
    assert 0.0 < m < 5.0, f"payload 질량 비정상: {m}"
    px, py, pz = (float(v) for v in args.payload_xyz.split())
    # 관성: 0.20×0.15×0.10 m 박스 근사 (전장 박스 스케일 — 자세동역학엔 CoM이 지배적)
    a, b, c = 0.20, 0.15, 0.10
    ixx, iyy, izz = (m / 12 * (b**2 + c**2), m / 12 * (a**2 + c**2),
                     m / 12 * (a**2 + b**2))
    payload_link = ET.fromstring(
        f'<link name="payload"><inertial><origin xyz="0 0 0" rpy="0 0 0"/>'
        f'<mass value="{m}"/><inertia ixx="{ixx:.6g}" ixy="0" ixz="0" '
        f'iyy="{iyy:.6g}" iyz="0" izz="{izz:.6g}"/></inertial></link>')
    payload_joint = ET.fromstring(
        f'<joint name="base_to_payload" type="fixed">'
        f'<origin xyz="{px} {py} {pz}" rpy="0 0 0"/>'
        f'<parent link="base"/><child link="payload"/></joint>')
    root.insert(2, payload_joint)
    root.insert(2, payload_link)
    report.append(f"★ payload 주입: {m:.2f} kg @ base ({px:+.3f}, {py:+.3f}, "
                  f"{pz:+.3f}) — 실물 미모델 전장(젯슨·보드·배선) 근사")

# 검증: 관절 12 + base_to_pelvis, 기대 관절명 전부 존재
jnames = {j.get("name") for j in root.findall("joint")}
missing = set(EXPECTED_AXES) - jnames
assert not missing, f"누락 관절: {missing}"
report.append(f"관절 {len(jnames) - 1} + base_to_pelvis, 링크 {len(root.findall('link'))}")

ET.indent(tree, space="  ")
tree.write(args.out, encoding="utf-8", xml_declaration=True)
report.append(f"출력: {args.out}")
print("\n".join(report))
