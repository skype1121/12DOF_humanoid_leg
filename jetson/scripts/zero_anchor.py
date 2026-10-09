#!/usr/bin/env python3
"""영점 재앵커 도구 — 현재 자세를 읽어 stand_zero_deg 를 보정/재캡처한다.

이 저장소에서 가장 위험한 쓰기다. stand_zero_deg 가 틀어지면
`joint_deg = direction × (motor_deg − stand_zero_deg)` 전체가 틀어지고,
그 뒤로는 **boot_pose_check 조차 틀린 기준에 대해 통과**해 버린다.
그래서 두 가지 모드를 엄격히 구분한다.

  multiple  (권장)  프레임 시프트만 보정. AK 엔코더는 로터 절대식이라
                    전원 사이클/브라운아웃에서 창 크기(AK70=36°, AK45=10°)의
                    정수배로 어긋난다. 원래 앵커는 그대로 두고 k×window 만
                    더한다. 자세 오차가 섞여 들어가지 않는다.
  recapture (위험)  현재 읽기를 그대로 새 앵커로 삼는다. 로봇이 **진짜 기준
                    자세**에 있을 때만 유효하다.

**핵심 안전장치**: 프레임 시프트(k≠0)가 하나라도 있으면 recapture 를 거부한다.
시프트된 상태에서 재캡처하면 어긋난 프레임이 영구히 앵커로 박혀 복구가
불가능해진다 (배수 보정 > 재캡처 — 프로젝트 확립 원칙).

사용:
    python3 zero_anchor.py --preview                  # 읽기만, JSON+표
    python3 zero_anchor.py --apply multiple           # 배수 보정
    python3 zero_anchor.py --apply recapture --in-reference-pose
    python3 zero_anchor.py --selftest
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

CALIB = "/home/mama/rl_calib"
sys.path.insert(0, CALIB)

HW_MAP = os.path.join(CALIB, "robot_12dof_hardware_map.json")
HW_MAP_REPO = "/home/mama/humanoid_repo/config/robot_12dof_hardware_map.json"
BACKUP_DIR = os.path.join(CALIB, "hw_map_backups")
#: 재앵커 중에 이 프로세스들이 살아 있으면 안 된다 — 제로토크 폴링이
#: stage8 의 MIT 스트리밍과 같은 버스에서 충돌한다.
CONFLICT = ("stage8_12axis_mit_control_node.py", "rl_bridge_node.py",
            "hang_stand.py")


def conflicting_processes():
    """버스를 쓰고 있는 프로세스 탐지 (탐지만 — 종료는 사람이 한다)."""
    found = []
    for name in CONFLICT:
        try:
            out = subprocess.run(["pgrep", "-f", name], capture_output=True,
                                 text=True, timeout=10).stdout.split()
        except (OSError, subprocess.SubprocessError):
            continue
        for pid in out:
            if pid.strip() and int(pid) != os.getpid():
                found.append((name, int(pid)))
    return found


def read_pose(n=15):
    """12모터 제로토크 폴링 — boot_pose_check 의 검증된 경로를 그대로 쓴다.

    기구 고정(하드웨어맵 "mech_fixed": true) 모터는 폴링하지 않고 결과에도 넣지 않는다 —
    판정·쓰기 대상이 아니므로 그 관절의 stand_zero_deg 는 어떤 모드로도 바뀌지 않는다."""
    import can
    import boot_pose_check as bpc
    hw = bpc.load_hw()
    # 이 파일은 저장소 사본으로도 돌지만 boot_pose_check 는 ~/rl_calib 사본을 쓴다 — 그쪽이 구판(고정 선언
    # 미지원)이면 종전대로 12모터 전부 폴링한다. 그 경우 선언된 고정 모터는 '무응답'(쓰기 차단)으로 드러난다.
    fixed = bpc.load_fixed_ids() if hasattr(bpc, "load_fixed_ids") else set()
    if fixed:
        print(f"[기구 고정] 모터 {sorted(fixed)} 기구 고정 — 제외 (판정·영점 쓰기 대상 아님)")
    bus = can.interface.Bus(channel="can1", interface="socketcan")
    out = {}
    try:
        for mid in [m for m in range(1, 13) if m not in fixed]:
            try:
                bus.send(can.Message(arbitration_id=mid, data=bpc.ENTER_MIT,
                                     is_extended_id=False))
            except Exception:
                pass
            time.sleep(0.01)
            st = bpc.poll_stats(bus, mid, n=n)
            st["stand_zero"] = hw.get(mid, {}).get("stand_zero")
            st["joint"] = hw.get(mid, {}).get("joint")
            st["window"] = bpc.WINDOW[mid]
            st["tol"] = bpc.TOL[mid]
            out[mid] = st
    finally:
        try:
            bus.shutdown()
        except Exception:
            pass
    return out


def classify(st):
    """모터 하나의 상태 판정 → (분류, k, 잔차)."""
    if st["med"] is None:
        return "무응답", None, None
    if st["err"]:
        return "폴트", None, None
    if st["spread"] > 2.0:
        return "불안정", None, None
    if st["stand_zero"] is None:
        return "기준없음", None, None
    delta = st["med"] - st["stand_zero"]
    k = round(delta / st["window"])
    resid = delta - k * st["window"]
    if abs(resid) > st["tol"]:
        return "자세불일치", k, resid
    return ("일치" if k == 0 else "프레임시프트"), k, resid


def analyze(pose):
    rows = []
    for mid in sorted(pose):
        st = pose[mid]
        kind, k, resid = classify(st)
        rows.append({
            "motor": mid, "joint": st["joint"], "kind": kind,
            "reading": (round(st["med"], 2) if st["med"] is not None else None),
            "stand_zero": st["stand_zero"], "k": k,
            "residual": (round(resid, 2) if resid is not None else None),
            "n_ok": st["n_ok"], "spread": round(st["spread"], 2),
            "window": st["window"], "tol": st.get("tol"),
            # 배수 보정 후의 새 앵커 / 재캡처 시의 새 앵커
            "new_multiple": (round(st["stand_zero"] + k * st["window"], 2)
                             if (k not in (None, 0) and st["stand_zero"] is not None)
                             else st["stand_zero"]),
            "new_recapture": (round(st["med"], 2) if st["med"] is not None else None),
        })
    bad = [r for r in rows if r["kind"] in ("무응답", "폴트", "불안정", "기준없음")]
    shifted = [r for r in rows if r["kind"] == "프레임시프트"]
    posture = [r for r in rows if r["kind"] == "자세불일치"]
    return {
        "rows": rows, "n_bad": len(bad), "n_shift": len(shifted),
        "n_posture": len(posture),
        "can_multiple": not bad and bool(shifted),
        # 시프트가 남아 있으면 재캡처 금지 — 어긋난 프레임이 영구 박힘
        "can_recapture": not bad and not shifted,
        "blockers": ([f"{r['kind']} 모터 {r['motor']}" for r in bad]
                     + ([f"프레임시프트 {len(shifted)}개 — 배수 보정 먼저"]
                        if shifted else [])),
    }


def render(rep):
    L = []
    L.append(f"{'모터':<5}{'관절':<20}{'판정':<10}{'읽기':>9}{'현앵커':>9}"
             f"{'k':>4}{'잔차':>8}{'응답':>7}")
    L.append("-" * 74)
    for r in rep["rows"]:
        rd = f"{r['reading']:+.2f}" if r["reading"] is not None else "—"
        sz = f"{r['stand_zero']:+.2f}" if r["stand_zero"] is not None else "—"
        k = str(r["k"]) if r["k"] is not None else "—"
        rs = f"{r['residual']:+.2f}" if r["residual"] is not None else "—"
        L.append(f"{r['motor']:<5}{(r['joint'] or '?'):<20}{r['kind']:<10}"
                 f"{rd:>9}{sz:>9}{k:>4}{rs:>8}{r['n_ok']:>5}/15")
    L.append("-" * 74)
    L.append(f"프레임시프트 {rep['n_shift']} · 자세불일치 {rep['n_posture']} · "
             f"불량 {rep['n_bad']}")
    if rep["can_multiple"]:
        L.append("→ 배수 보정 가능 (권장): 원래 앵커 유지, k×창만 보정")
        for r in rep["rows"]:
            if r["kind"] == "프레임시프트":
                L.append(f"   모터 {r['motor']}: {r['stand_zero']:+.2f} → "
                         f"{r['new_multiple']:+.2f}  (k={r['k']}×{r['window']:.0f}°)")
    if rep["can_recapture"]:
        L.append("→ 재캡처 가능: 지금 자세가 '진짜 기준 자세'일 때만 의미 있음")
    for b in rep["blockers"]:
        L.append(f"✗ 차단: {b}")
    return "\n".join(L)


def apply_map(rep, mode):
    """백업 → 원자적 쓰기 → 젯슨 2곳 동기 → md5 확인."""
    key = "new_multiple" if mode == "multiple" else "new_recapture"
    with open(HW_MAP) as f:
        hw = json.load(f)
    joints = hw.get("joints") or {}
    changed = []
    for r in rep["rows"]:
        j = r["joint"]
        if j not in joints or r[key] is None:
            continue
        old = joints[j].get("stand_zero_deg")
        new = r[key]
        if old is None or abs(float(old) - float(new)) < 1e-9:
            continue
        joints[j]["stand_zero_deg"] = float(new)
        changed.append((j, old, new))
    if not changed:
        return False, "변경할 항목이 없습니다 (이미 일치)", []

    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(BACKUP_DIR, f"hw_map_{stamp}.json")
    shutil.copy2(HW_MAP, backup)

    prov = hw.setdefault("_provenance", {})
    prov.setdefault("zero_anchor_history", []).append({
        "at": stamp, "mode": mode,
        "changed": [{"joint": j, "from": o, "to": n} for j, o, n in changed],
        "backup": backup,
    })
    tmp = HW_MAP + ".tmp"
    with open(tmp, "w") as f:
        json.dump(hw, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, HW_MAP)
    msgs = [f"백업: {backup}", f"{len(changed)}개 관절 갱신"]
    try:
        shutil.copy2(HW_MAP, HW_MAP_REPO)
        msgs.append("젯슨 repo 사본 동기")
    except OSError as exc:
        msgs.append(f"repo 사본 동기 실패: {exc}")
    return True, " / ".join(msgs), changed


def md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def selftest():
    """판정·차단 로직 검사 — CAN 없이 순수 함수만."""
    fails = []
    W, T = 36.0, 8.0

    def st(med, sz, spread=0.1, err=None, w=W, tol=T):
        return {"med": med, "stand_zero": sz, "spread": spread, "err": err,
                "window": w, "tol": tol, "n_ok": 15, "joint": "j"}

    cases = [
        (st(32.0, 31.99), "일치"),
        (st(67.99, 31.99), "프레임시프트"),      # +36
        (st(-4.01, 31.99), "프레임시프트"),      # -36
        (st(45.0, 31.99), "자세불일치"),         # 13° — 창의 절반 미만, k=0, 잔차 초과
        (st(None, 31.99), "무응답"),
        (st(32.0, 31.99, err=7), "폴트"),
        (st(32.0, 31.99, spread=9.0), "불안정"),
    ]
    for s, want in cases:
        got, _, _ = classify(s)
        if got != want:
            fails.append(f"분류 오류: {s['med']} vs {s['stand_zero']} → {got} (기대 {want})")

    # 시프트가 있으면 재캡처는 반드시 차단되어야 한다 (핵심 안전장치)
    pose = {1: st(67.99, 31.99), 2: st(32.0, 31.99)}
    for m in (1, 2):
        pose[m]["joint"] = f"j{m}"
    rep = analyze(pose)
    if rep["can_recapture"]:
        fails.append("프레임시프트가 있는데 재캡처가 허용됨 — 치명적")
    if not rep["can_multiple"]:
        fails.append("시프트가 있는데 배수 보정이 불가로 나옴")
    # 보정값이 원래 앵커 + k×창 인지
    r1 = [r for r in rep["rows"] if r["motor"] == 1][0]
    if abs(r1["new_multiple"] - (31.99 + 36.0)) > 1e-6:
        fails.append(f"배수 보정값 오류: {r1['new_multiple']}")

    # 전부 일치면 재캡처 허용, 배수 보정은 할 것 없음
    pose2 = {1: st(32.0, 31.99), 2: st(31.9, 31.99)}
    for m in (1, 2):
        pose2[m]["joint"] = f"j{m}"
    rep2 = analyze(pose2)
    if not rep2["can_recapture"] or rep2["can_multiple"]:
        fails.append("전부 일치 상태의 허용 판정 오류")

    # 불량이 하나라도 있으면 둘 다 차단
    pose3 = {1: st(None, 31.99), 2: st(32.0, 31.99)}
    for m in (1, 2):
        pose3[m]["joint"] = f"j{m}"
    rep3 = analyze(pose3)
    if rep3["can_recapture"] or rep3["can_multiple"]:
        fails.append("불량 모터가 있는데 쓰기가 허용됨")

    for f in fails:
        print(f"[FAIL] {f}")
    print("selftest: " + ("ALL PASS" if not fails else f"{len(fails)}건 실패"))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser(description="영점 재앵커")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--apply", choices=("multiple", "recapture"))
    ap.add_argument("--in-reference-pose", action="store_true",
                    help="recapture 시 필수 — 지금 자세가 진짜 기준 자세임을 선언")
    ap.add_argument("--json", action="store_true", help="기계 판독용 JSON도 출력")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.preview and not a.apply:
        ap.error("--preview 또는 --apply 중 하나가 필요합니다")

    busy = conflicting_processes()
    if busy:
        print("✗ CAN 버스를 쓰는 프로세스가 살아 있습니다 — 제로토크 폴링과 충돌합니다.")
        for n, p in busy:
            print(f"   {n} (pid {p})")
        print("   스택을 먼저 내리고(스택 종료) 다시 실행하세요.")
        return 2

    pose = read_pose()
    rep = analyze(pose)
    print(render(rep))
    if a.json:
        print("JSON " + json.dumps(rep, ensure_ascii=False))
    if not a.apply:
        return 0 if rep["n_bad"] == 0 else 1

    if a.apply == "multiple" and not rep["can_multiple"]:
        print("✗ 배수 보정 불가 — 위 차단 사유를 해소하세요.")
        return 2
    if a.apply == "recapture":
        if not rep["can_recapture"]:
            print("✗ 재캡처 불가 — 프레임 시프트가 남아 있으면 어긋난 프레임이 "
                  "영구히 앵커로 박힙니다. 배수 보정을 먼저 하세요.")
            return 2
        if not a.in_reference_pose:
            print("✗ 재캡처에는 --in-reference-pose 가 필요합니다 "
                  "(지금 자세가 진짜 기준 자세임을 선언).")
            return 2

    ok, msg, changed = apply_map(rep, a.apply)
    print(("✓ " if ok else "· ") + msg)
    for j, o, n in changed:
        print(f"   {j}: {o:+.2f} → {n:+.2f}")
    if ok:
        print(f"   md5: {md5(HW_MAP)}")
        print("\n⚠ 반드시 할 것")
        print("  1) stage8·브리지 재기동 — 구맵을 물고 있으면 인수를 거부하고")
        print("     브레이크 모드로 방치되는 붕괴 사고가 있었습니다.")
        print("  2) boot_pose_check 재실행으로 12/12 확인.")
        print("  3) 메인 PC 사본 동기화:")
        print("     scp mama@<젯슨IP>:~/rl_calib/robot_12dof_hardware_map.json"
              " config/   (젯슨은 DHCP — 0907 기준 172.16.35.89, 콘솔 접속 주소와 동일)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
