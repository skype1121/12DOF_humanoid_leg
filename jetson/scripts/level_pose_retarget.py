#!/usr/bin/env python3
"""영점 재캡처 뒤 '수평 기준 자세'(관절 프레임 트림)를 기존 물리 자세 그대로 보존하는 좌표 변환 (0918 밤, 승윤님 조건 2).

관절각 = DIR × (모터각 − SZ). 모터각(물리 자세)을 고정하면
    trim_new = trim_old + DIR × (SZ_old − SZ_new)
기본은 표만 출력(쓰기 없음). --write 는 level_pose.json 을 백업 뒤 갱신 — 지지 상태에서 확인 뒤에만.

  python3 jetson/scripts/level_pose_retarget.py --old <구 하드웨어맵.json> --new <새 하드웨어맵.json> \
      --level ~/rl_calib/level_pose.json [--write]
"""
import argparse
import json
import os
import shutil
import time

ORDER = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint",
         "left_hip_r_joint", "right_hip_r_joint", "left_knee_joint", "right_knee_joint",
         "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]


def joints_of(m):
    return m.get("joints") or m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    jo, jn = joints_of(json.load(open(a.old))), joints_of(json.load(open(a.new)))
    lv = json.load(open(a.level))
    trims = lv["trims_joint_frame_deg"]
    print(f"{'관절':<20}{'DIR':>4}{'SZ_old':>9}{'SZ_new':>9}{'ΔSZ':>8}{'trim_old':>10}{'trim_new':>10}{'모터각(불변)':>12}  비고")
    new_trims = {}
    warn = []
    for j in ORDER:
        d_o, d_n = int(jo[j]["direction"]), int(jn[j]["direction"])
        so, sn = float(jo[j]["stand_zero_deg"]), float(jn[j]["stand_zero_deg"])
        t_old = float(trims.get(j, 0.0))
        note = ""
        if d_o != d_n:
            note = "⚠ direction 이 바뀜 — 변환 불가, 수동 검토"; warn.append(j)
        motor = so + d_o * t_old
        t_new = t_old + d_n * (so - sn)
        new_trims[j] = round(t_new, 2)
        if abs(so - sn) > 2.0 and not note:
            note = f"앵커 {so-sn:+.1f}° 이동 → 트림 {t_new-t_old:+.1f}°"
        print(f"{j:<20}{d_n:>4}{so:>9.2f}{sn:>9.2f}{so-sn:>+8.2f}{t_old:>+10.2f}{t_new:>+10.2f}{motor:>12.2f}  {note}")
    # 좌우 대칭 점검(힙앞뒤·무릎·발목앞뒤는 반대 부호가 정상)
    for l, r in (("left_hip_f_joint", "right_hip_f_joint"), ("left_knee_joint", "right_knee_joint"), ("left_ankle_f_joint", "right_ankle_f_joint")):
        asym = new_trims[l] + new_trims[r]
        print(f"  대칭 점검 {l[5:]:<14} L {new_trims[l]:+.2f} / R {new_trims[r]:+.2f} → 합 {asym:+.2f}° (0 이면 대칭)")
    if warn:
        print("⚠ direction 변경 관절:", warn, "— 쓰기 거부")
        return 2
    if a.write:
        bak = a.level + ".bak_retarget_" + time.strftime("%m%d_%H%M")
        shutil.copy2(a.level, bak)
        hist = lv.get("history", [])
        hist.insert(0, {"version": "pre_retarget_" + time.strftime("%m%d_%H%M"), "trims": trims, "hold_kd": lv.get("hold_kd"), "note": lv.get("note", ""),
                        "old_map_md5": os.path.basename(a.old), "new_map": os.path.basename(a.new)})
        lv["trims_joint_frame_deg"] = new_trims
        lv["note"] = (lv.get("note", "") + f" | {time.strftime('%m%d %H:%M')} 영점 재캡처 좌표변환(물리 자세 보존): {os.path.basename(a.old)} → {os.path.basename(a.new)}")
        lv["history"] = hist
        json.dump(lv, open(a.level, "w"), ensure_ascii=False, indent=2)
        print(f"✓ 갱신: {a.level} (백업 {bak})")
    else:
        print("(쓰기 없음 — --write 로 적용, 지지 상태에서 확인 뒤)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
