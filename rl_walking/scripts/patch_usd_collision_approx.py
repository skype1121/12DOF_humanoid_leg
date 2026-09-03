"""base(=골반 병합체) 콜리전을 convexHull → convexDecomposition 으로 교체.
신형 골반 어셈블리의 콘벡스 헐이 힙 영역(오목부)을 삼켜 self-collision ON 에서
비인접 링크와 상호관입 → base_contact 1N 즉시 종료. 실물 형상은 오목하므로
분해 근사가 맞다. 인스터블 구조라 configuration 레이어를 직접 연다."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd_dir", required=True)
parser.add_argument("--targets", default="pelvis",
                    help="쉼표구분 — 콜리전 경로에 이 문자열이 포함된 프림만 패치")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os
from pxr import Usd, UsdPhysics

targets = args.targets.split(",")
patched = []
for layer_name in ("biped12_base.usd", "biped12_physics.usd", "biped12_robot.usd"):
    path = os.path.join(args.usd_dir, "configuration", layer_name)
    if not os.path.exists(path):
        continue
    stage = Usd.Stage.Open(path)
    changed = 0
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if ("/colliders/" not in p and "/collisions/" not in p) or not any(t in p for t in targets):
            continue
        if prim.GetTypeName() != "Mesh":
            continue
        api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        before = api.GetApproximationAttr().Get()
        api.CreateApproximationAttr("convexDecomposition")
        changed += 1
        patched.append(f"{layer_name}: {p} ({before} → convexDecomposition)")
    if changed:
        stage.Save()
    print(f"{layer_name}: {changed}개 패치")
report = "\n".join([f"패치 {len(patched)}건"] + patched)
# 검증: 다시 열어 실제 값 확인
verify = []
for layer_name in ("biped12_base.usd", "biped12_physics.usd", "biped12_robot.usd"):
    path = os.path.join(args.usd_dir, "configuration", layer_name)
    if not os.path.exists(path):
        continue
    st2 = Usd.Stage.Open(path)
    for prim in st2.Traverse():
        p = str(prim.GetPath())
        if ("/colliders/" in p or "/collisions/" in p) and prim.GetTypeName() == "Mesh":
            a = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            verify.append(f"{layer_name} {p} approx={a}")
report += "\n== 전 콜리전 메시 approximation ==\n" + "\n".join(verify)
open("/home/ryu/humanoid_leg_test1/log_picture/02f3_decomp_패치검증.txt", "w").write(report + "\n")
print(report)
app.close()
