"""base↔hip_a 충돌 페어 필터링 — 신형 골반 메시(±122mm)와 hip_a 헐의 근사 겹침(28mm)이
self-collision ON 에서 스폰 즉시 수천 N 관입을 만들어 base_contact 종료를 유발.
실물은 설계 공차로 비접촉이고 실간섭은 자기충돌 스윕 기반 관절리밋이 이미 차단.
루트 레이어(biped12.usd)에 over 로 저작 — 컴포지션 최강."""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--usd", required=True)
parser.add_argument("--pairs", default="base:left_hip_a_joint,base:right_hip_a_joint",
                    help="쉼표구분 bodyA:bodyB")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import Usd, Sdf, UsdPhysics

stage = Usd.Stage.Open(args.usd)
report = []
for pair in args.pairs.split(","):
    a, b = pair.split(":")
    pa, pb = f"/biped12/{a}", f"/biped12/{b}"
    prim = stage.OverridePrim(pa)
    api = UsdPhysics.FilteredPairsAPI.Apply(prim)
    rel = api.CreateFilteredPairsRel()
    rel.AddTarget(Sdf.Path(pb))
    report.append(f"필터 저작: {pa} ↔ {pb}")
stage.Save()

# 검증 재오픈
st2 = Usd.Stage.Open(args.usd)
for pair in args.pairs.split(","):
    a, _ = pair.split(":")
    prim = st2.GetPrimAtPath(f"/biped12/{a}")
    api = UsdPhysics.FilteredPairsAPI(prim)
    tg = api.GetFilteredPairsRel().GetTargets() if prim.HasAPI(UsdPhysics.FilteredPairsAPI) else []
    report.append(f"검증 /biped12/{a}: API={prim.HasAPI(UsdPhysics.FilteredPairsAPI)} targets={[str(t) for t in tg]}")

open("/home/ryu/humanoid_leg_test1/log_picture/02f6_충돌필터저작.txt", "w").write("\n".join(report) + "\n")
print("\n".join(report))
app.close()
