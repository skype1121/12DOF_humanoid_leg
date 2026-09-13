#!/usr/bin/env python3
"""발목롤(left/right_ankle_r) 관절 프레임 부호 정정 제안 도구 (2026-09-14).

근거(문서 reports/real_prep_0913/promotion_criteria_v7.md '좌표계 재검토'):
  - URDF(final3) 운동학·Isaac 검증: 관절 + = 외번(발바닥이 바깥을 향함 = 안쪽 가장자리 아래), 양쪽 동일.
  - 실물 무토크 손 시험(0914 01:2x): 안쪽 가장자리 아래 = 관절 프레임 − (왼 −14.1°, 오른 −16.5°; joint = direction×(motor−zero)).
  → 실물 관절 프레임이 시뮬과 양쪽 모두 반전. 정정 = 두 관절의 `direction` 부호 반전(stand_zero_deg 는 모터 프레임이라 불변).

이 도구는 **기본적으로 아무것도 쓰지 않는다**(diff 만 출력). `--apply <맵경로>` 는 백업을 만든 뒤 두 관절의 direction 만 뒤집는다.
젯슨 반영은 승윤님 승인 후: 두 사본(~/humanoid_repo/config, ~/rl_calib) 모두 같은 값 → stage8·브리지 재기동(구맵 인수 거부) → 손 시험 재확인
(안쪽 가장자리 아래 = 관절 + 로 읽혀야 함) → 공중 정책 시작 시 매달림 발바닥이 안쪽 가장자리 아래(외번)인지 눈으로 확인.
"""
import argparse, json, os, shutil, sys, time
J = ("left_ankle_r_joint", "right_ankle_r_joint")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map", nargs="?", default="config/robot_12dof_hardware_map.json")
    ap.add_argument("--apply", action="store_true", help="백업 후 direction 반전 기록 (승인 뒤에만)")
    a = ap.parse_args()
    m = json.load(open(a.map, encoding="utf-8"))
    cur = {j: m["joints"][j]["direction"] for j in J}
    print(f"맵 {a.map}: 현재 direction {cur} → 제안 {{'left_ankle_r_joint': {-cur[J[0]]}, 'right_ankle_r_joint': {-cur[J[1]]}}}")
    print("  (stand_zero_deg·sign·motor_id 불변; 관절 한계는 관절 프레임 대칭 여부를 stage8 기동 로그로 확인)")
    if not a.apply:
        print("  --apply 없음: 아무것도 쓰지 않음"); return 0
    bak = f"{a.map}.bak_ankle_r_dir_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(a.map, bak)
    for j in J:
        m["joints"][j]["direction"] = -int(m["joints"][j]["direction"])
    m.setdefault("_provenance", {}).setdefault("direction_history", []).append(
        {"at": time.strftime("%Y%m%d_%H%M%S"), "changed": [{"joint": j, "from": cur[j], "to": -cur[j]} for j in J],
         "why": "0914 발목롤 관절 프레임 반전(URDF 외번=+ vs 실물 손 시험 안쪽아래=−)", "backup": bak})
    json.dump(m, open(a.map, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  기록 완료, 백업 {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
