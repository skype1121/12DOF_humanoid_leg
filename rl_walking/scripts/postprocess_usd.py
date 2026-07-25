"""변환된 biped12.usd 검증 + 관절 리밋 주입.

URDF 관절이 전부 continuous(리밋 없음)라서 임포터가 리밋을 안 넣는다.
실측 리밋(config/sim_joint_limits.json, 자기충돌 스윕 기반)을 USD에 굽는다.
UsdPhysics.RevoluteJoint 의 limit 단위는 도(deg).

실행: ./isaaclab.sh -p rl_walking/scripts/postprocess_usd.py --headless
(pxr 로드를 위해 Kit 앱을 헤드리스로 부팅)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import os
import sys

from pxr import Usd, UsdPhysics

REPO = "/home/ryu/humanoid_leg_test1"
USD = os.path.join(REPO, "rl_walking", "assets", "usd", "biped12.usd")
LIMITS = os.path.join(REPO, "config", "sim_joint_limits.json")
# Kit 헤드리스에서 stdout이 유실되는 경우가 있어 결과를 파일에 직접 기록
OUT = os.path.join(REPO, "log_picture", "02_usd변환검증_관절리밋주입.txt")

rep = []


def p(line=""):
    rep.append(str(line))


stage = Usd.Stage.Open(USD)
if stage is None:
    open(OUT, "w").write(f"USD 열기 실패: {USD}\n")
    sys.exit(1)

limits = json.load(open(LIMITS))["limits_deg"]

p("biped12.usd 변환 검증 + 실측 관절리밋 주입 (자동 생성)")
p("=" * 70)
p("1) 프림 트리 (depth<=2)")
for prim in stage.Traverse():
    depth = str(prim.GetPath()).count("/")
    if depth <= 2:
        p(f"   {prim.GetPath()}  [{prim.GetTypeName()}]")

p("=" * 70)
p("2) 리지드 바디 + 질량")
total_mass = 0.0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MassAPI) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        mass_api = UsdPhysics.MassAPI(prim)
        m = mass_api.GetMassAttr().Get()
        if m:
            total_mass += m
        p(f"   {prim.GetName():28s} mass={m}")
p(f"   >>> 총질량 = {total_mass:.4f} kg (기대 10.490 + base 0.001 — 12URDF0725)")

p("=" * 70)
p("3) 관절: 리밋 주입 전 상태 → 주입 (단위 deg)")
injected, joint_names = [], []
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        j = UsdPhysics.RevoluteJoint(prim)
        name = prim.GetName()
        joint_names.append(name)
        lo_before = j.GetLowerLimitAttr().Get()
        hi_before = j.GetUpperLimitAttr().Get()
        if name in limits:
            lo, hi = limits[name]["lower"], limits[name]["upper"]
            j.CreateLowerLimitAttr(float(lo))
            j.CreateUpperLimitAttr(float(hi))
            injected.append(name)
            p(f"   {name:24s} axis={j.GetAxisAttr().Get()}  "
              f"[{lo_before}, {hi_before}] -> [{lo}, {hi}]")
        else:
            p(f"   {name:24s} !! limits_deg에 없음 — 미주입")

p("=" * 70)
p(f"4) 결과: 관절 {len(joint_names)}개, 리밋 주입 {len(injected)}개")
missing = [k for k in limits if k not in joint_names]
if missing:
    p(f"   !! USD에 없는 리밋 키: {missing}")
if len(injected) == 12 and not missing:
    stage.Save()  # dirty 레이어 전부 저장
    p("   저장 완료 OK")
else:
    p("   !! 12개 미주입 — 저장 안 함 (검토 필요)")

open(OUT, "w").write("\n".join(rep) + "\n")
print("\n".join(rep), flush=True)

simulation_app.close()
