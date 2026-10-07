#!/usr/bin/env python3
"""계약 감사(contract_audit.py)의 민감도 시험 — 배포 코드 사본에 결함을 하나씩 심어 감사가 그 항목을 '불일치'로 잡는지 본다.

  python3 reports/night_1007/contract_audit_mutants.py            # 전부(약 6 분, CPU)
  python3 reports/night_1007/contract_audit_mutants.py --only lpf_beta_045,slew_3deg
  python3 reports/night_1007/contract_audit_mutants.py --json 결과.json  # 변이별 기대 항목·적중을 JSON 으로(contract_audit.py --run-mutants 가 읽는다)

방법: 저장소의 배포 파일(rl_walking/deploy·robot_runtime·config·브리지)을 임시 폴더에 복사 → 문자열 한 곳을 바꿈 →
`contract_audit.py --deploy-tree <사본> --json` 실행 → 변이 없는 사본(대조군)의 판정과 비교.
  대조군: 변이 없는 사본의 판정 = 저장소를 직접 감사한 판정(사본을 만드는 것 자체가 결과를 바꾸지 않는다).
  통과 조건: 변이마다 '기대 항목'이 대조군에서는 불일치가 아니었고 변이에서는 불일치(또는 감사 중단)여야 하고,
            회귀 관문(contract_audit.py --allow-known)의 종료 코드가 대조군 0 → 변이 1 이어야 한다.
저장소 파일은 건드리지 않는다. GPU·ROS·젯슨 없음.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
AUDIT = os.path.join(HERE, "contract_audit.py")
BR = "jetson/scripts/rl_bridge_node.py"
PR = "rl_walking/deploy/policy_runner.py"
PR4 = "rl_walking/deploy/policy_runner_stage4.py"
DC = "rl_walking/deploy/deploy_chain.py"
W3 = "rl_walking/deploy/w3_deploy.py"
KW = "rl_walking/deploy/kew3_runner.py"

#: (이름, 파일, 바꿀 문자열, 새 문자열, 기대 횟수, 불일치로 바뀌어야 하는 항목 이름의 일부(하나라도 맞으면 됨), 설명)
MUTANTS = [
    ("act_obs_beta_030", BR, "OBS_ACT_EMA_BETA_DEPLOY = 0.27", "OBS_ACT_EMA_BETA_DEPLOY = 0.30", 1, ["actions(직전 행동)"], "관측 직전 행동 EMA β 0.27 → 0.30"),
    ("jvel_obs_beta_05", BR, 'jv = env_contract.get("obs_joint_vel_beta")', "jv = 0.5", 1, ["joint_vel = 엔코더 유한차분 속도의 EMA"], "관측 관절속도 EMA β 0.25 → 0.5"),
    ("lpf_beta_045", DC, "STAND_LPF_BETA = 0.27", "STAND_LPF_BETA = 0.45", 1, ["액션 LPF β"], "액션 LPF β 0.27 → 0.45(학습 대역 0.2–0.4 밖)"),
    ("blend_2s", DC, "BLEND_S = 1.5 ", "BLEND_S = 2.0 ", 1, ["온셋 블렌딩(인계)"], "온셋 블렌딩 1.5 → 2.0 s(학습 대역 1.0–1.5 밖)"),
    ("gait_freq_14", BR, '"stand_crouch_g21a": 1.1,', '"stand_crouch_g21a": 1.4,', 1, ["gait_clock(서기 정책)"], "G21a 클록 1.1 → 1.4 Hz"),
    ("clock_one_tick_ahead", PR4, "else gait_clock_stage4(self._tick, self.gait_freq))", "else gait_clock_stage4(self._tick + 1, self.gait_freq))", 1,
     ["gait_clock(서기 정책)"], "클록 위상 한 틱 앞섬"),
    ("default_pose_hip", BR, "left_hip_f_joint=0.5282, right_hip_f_joint=-0.5282,", "left_hip_f_joint=0.5182, right_hip_f_joint=-0.5182,", 1,
     ["joint_pos = 엔코더각 − 기본자세", "오프셋 = 기본자세"], "기본자세 힙 0.5282 → 0.5182 rad(0.57°)"),
    ("history_reversed", PR4, "return self._buf.reshape(-1)      # (T·D,) — 프레임0(最古)부터", "return self._buf[::-1].reshape(-1)", 1,
     ["이력 배열 순서"], "이력 배열을 최신 먼저로"),
    ("term_order_swap", PR4, 'STAGE4_TERMS_S8A = STAGE4_TERMS + (("gyro_ema", 3), ("joint_vel_ema", 12))',
     'STAGE4_TERMS_S8A = STAGE4_TERMS + (("joint_vel_ema", 12), ("gyro_ema", 3))', 1, ["항 목록·순서"], "관측 끝 두 항 순서 바꿈"),
    ("contact_lr_swap", PR4, '"feet_contact": self._contact2(contact2),', '"feet_contact": self._contact2(contact2)[::-1].copy(),', 1,
     ["feet_contact 순서"], "접촉 좌우 뒤바꿈"),
    ("ema_alpha_02", PR4, "EMA_ALPHA_S8A = 0.1 ", "EMA_ALPHA_S8A = 0.2 ", 1, ["gyro_ema 계수", "joint_vel_ema 계수"], "S8a EMA α 0.1 → 0.2"),
    ("prev_action_emitted", PR4, "        self._prev_emitted = emitted\n        return emitted.copy()\n\n\n# ---",
     "        self._prev_emitted = emitted\n        self.last_action = ((emitted - self.default_pose) / ACTION_SCALE).astype(np.float32)\n        return emitted.copy()\n\n\n# ---", 1,
     ["actions(직전 행동)"], "직전 행동을 슬루 뒤 목표(행동 단위)로 되먹임"),
    ("scale_020", PR, "ACTION_SCALE = 0.25 ", "ACTION_SCALE = 0.20 ", 1, ["스케일"], "행동 스케일 0.25 → 0.20"),
    ("slew_3deg", PR, "SLEW_RAD = 0.06981317007977318 ", "SLEW_RAD = 0.05235987755982988 ", 1, ["슬루 상한"], "슬루 4 → 3°/틱"),
    ("obs_clip_added", PR4, '            "base_ang_vel": vec(gyro_rad_s, 3, "gyro_rad_s"),', '            "base_ang_vel": np.clip(vec(gyro_rad_s, 3, "gyro_rad_s"), -20.0, 20.0),', 1,
     [], "(반대 방향 대조) 자이로 관측에 학습과 같은 ±20 클립을 넣음 — '클립' 줄의 통과 값이 20 으로 바뀌어야 한다"),
    ("det_on_tilt_35", BR, "C5_DETECTOR_KW = dict(on_tilt_deg=3.0,", "C5_DETECTOR_KW = dict(on_tilt_deg=3.5,", 1, ["켜기 기울기 편차"], "감지기 켜기 문턱 3.0 → 3.5°"),
    ("det_early_hold_05", BR, 'early=(0.6, 6.0, 0.4, 0.7)', 'early=(0.6, 6.0, 0.4, 0.5)', 1, ["조기 복귀 (최소 s"], "krh 조기 복귀 유지 0.7 → 0.5 s"),
    ("det_in_filt_05", BR, "in_filt=0.33", "in_filt=0.5", 2, ["감지 입력 EMA α"], "감지 입력 EMA 0.33 → 0.5"),
    ("cue_vx_005", BR, "cue_vx=0.03", "cue_vx=0.05", 2, ["회복 cue"], "회복 cue vx 0.03 → 0.05(평가와 달라짐 — 반대로 학습 범위 안으로는 들어온다)"),
    ("w3_fstep_12", W3, "    f_step: float = 1.1 ", "    f_step: float = 1.2 ", 1, ["스텝 주파수 f_step"], "W3 스텝 주파수 1.1 → 1.2 Hz"),
    ("w3_ramp_04", W3, "    ramp_s: float = 0.2 ", "    ramp_s: float = 0.4 ", 1, ["관측 게이트 램프"], "W3 게이트 램프 0.2 → 0.4 s"),
    ("w3_lead_wz", W3, "W3S3D_CONFIG = W3Config(lead_vy_cmin=0.02, lead_wz_cmin=0.08)", "W3S3D_CONFIG = W3Config(lead_vy_cmin=0.02, lead_wz_cmin=0.25)", 1,
     ["리드 발 규칙"], "리드 발 wz 문턱 0.08 → 0.25(−0.15 회전이 왼발로 시작)"),
    ("w3_vy_limit", W3, "    vy: tuple = (0.03, 0.04)", "    vy: tuple = (0.03, 0.08)", 1, ["명령 관측 범위(걷기)"], "옆걸음 상한 0.04 → 0.08(학습 밖)"),
    ("w3_enable_tilt", W3, "    enable_tilt_deg: float = 7.0 ", "    enable_tilt_deg: float = 9.0 ", 1, ["걷기 켜기 선행조건"], "켜기 기울기 상한 7 → 9°"),
    ("kew3_settle_02", BR, 'handoff="carry", settle_quiet_s=0.5)}', 'handoff="carry", settle_quiet_s=0.2)}', 1, ["W3 → KE 인계 규칙"], "합성 정착 0.5 → 0.2 s"),
    ("kew3_settle_w", KW, 'settle_quiet_s: float = 0.0, settle_w: float = 0.15,', 'settle_quiet_s: float = 0.0, settle_w: float = 0.6,', 1,
     ["W3 → KE 인계 규칙"], "합성 정착 |ω| 문턱 0.15 → 0.6(시끄러워도 인계)"),
]


def copy_tree(dst):
    for rel in ("rl_walking/deploy", "robot_runtime", "config"):
        shutil.copytree(os.path.join(REPO, rel), os.path.join(dst, rel), ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(os.path.join(REPO, "rl_walking/__init__.py"), os.path.join(dst, "rl_walking/__init__.py"))
    os.makedirs(os.path.join(dst, "jetson/scripts"))
    shutil.copyfile(os.path.join(REPO, BR), os.path.join(dst, BR))


def run_audit(tree, out_json):
    # --allow-known: 문서에 적힌 불일치만 있으면 0 — 회귀 관문이 변이를 실제로 막는지(종료 코드 1)도 함께 본다
    cmd = [sys.executable, AUDIT, "--json", out_json, "--allow-known"] + (["--deploy-tree", tree] if tree else [])
    r = subprocess.run(cmd, capture_output=True)
    rows = None
    if os.path.isfile(out_json):
        with open(out_json, encoding="utf-8") as f:
            rows = json.load(f)
    return r.returncode, rows, r.stderr.decode(errors="replace")[-600:]


def verdicts(rows):
    return {(k, r["cat"], r["item"]): r["verdict"] for k, rs in rows.items() for r in rs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="쉼표로 변이 이름")
    ap.add_argument("--json", default=None, help="결과(대조군·변이별 기대 항목·적중·바뀐 줄 수)를 JSON 으로 저장")
    a = ap.parse_args()
    res = {"control": None, "mutants": []}
    only = {x for x in a.only.split(",") if x}
    tmp = tempfile.mkdtemp(prefix="ca_mut_")
    fails = 0
    try:
        rc0, direct, e0 = run_audit(None, os.path.join(tmp, "direct.json"))
        base_tree = os.path.join(tmp, "base")
        os.makedirs(base_tree)
        copy_tree(base_tree)
        rc1, base, e1 = run_audit(base_tree, os.path.join(tmp, "base.json"))
        if direct is None or base is None:
            print("[중단] 대조군 감사 실패\n" + e0 + e1)
            return 2
        vd, vb = verdicts(direct), verdicts(base)
        same = (vd == vb) and rc0 == 0 and rc1 == 0          # 대조군: 판정이 같고, 회귀 관문은 둘 다 통과(0)
        print(f"[대조군] 저장소 직접 감사 = 변이 없는 사본 감사: {'같음' if same else '다름'} (행 {len(vd)}·{len(vb)}, 종료 코드 {rc0}·{rc1})")
        res["control"] = {"same": same, "rows_direct": len(vd), "rows_copy": len(vb), "rc_direct": rc0, "rc_copy": rc1}
        if not same:
            fails += 1
            for k in sorted(set(vd) | set(vb)):
                if vd.get(k) != vb.get(k):
                    print("   ", k, vd.get(k), "→", vb.get(k))
        text_b = {(k, r["cat"], r["item"]): r["head"] for k, rs in base.items() for r in rs}
        for name, rel, old, new, cnt, expect, desc in MUTANTS:
            if only and name not in only:
                continue
            tree = os.path.join(tmp, name)
            os.makedirs(tree)
            copy_tree(tree)
            p = os.path.join(tree, rel)
            with open(p, encoding="utf-8") as f:
                src = f.read()
            if src.count(old) != cnt:
                print(f"[FAIL] {name}: 바꿀 문자열이 {src.count(old)} 번 나옴(기대 {cnt}) — 변이를 심지 못함")
                res["mutants"].append({"name": name, "desc": desc, "ok": False, "expect": expect, "hit": [], "n_flipped": 0, "note": "변이를 심지 못함(배포 소스가 바뀜)"})
                fails += 1
                continue
            with open(p, "w", encoding="utf-8") as f:
                f.write(src.replace(old, new))
            rc, rows, err = run_audit(tree, os.path.join(tmp, name + ".json"))
            shutil.rmtree(tree, ignore_errors=True)
            if rows is None:
                print(f"[PASS] {name}: {desc} → 감사 중단(종료 코드 {rc}) — 결함이 러너 생성·기동 검사에서 막힘: {err.strip().splitlines()[-1][:160] if err.strip() else ''}")
                res["mutants"].append({"name": name, "desc": desc, "ok": True, "expect": expect, "hit": [], "n_flipped": 0, "rc": rc, "note": f"감사 중단(종료 코드 {rc}) — 기동 검사에서 막힘"})
                continue
            vm = verdicts(rows)
            flipped = sorted({k[2] for k in vm if vm[k] == "불일치" and vb.get(k) != "불일치"})
            lost = sorted({k[2] for k in vb if vb[k] == "불일치" and vm.get(k) not in ("불일치", None)})
            lost_to = {it: "·".join(sorted({vm[k] for k in vb if k[2] == it and vb[k] == "불일치" and vm.get(k) not in ("불일치", None)})) for it in lost}
            if not expect:          # 반대 방향 대조: 기존 불일치가 사라지거나 본문 값이 바뀌어야 한다
                text_m = {(k, r["cat"], r["item"]): r["head"] for k, rs in rows.items() for r in rs}
                changed = sorted({k[2] for k in text_m if "클립" in k[2] and text_m[k] != text_b.get(k)})
                ok = bool(changed) and rc == 0               # 고친 방향이라 새 불일치가 없어야 한다(관문 통과)
                print(f"[{'PASS' if ok else 'FAIL'}] {name}: {desc} → 값이 바뀐 줄 {changed} · 불일치가 사라진 줄 {lost} · 관문 종료 코드 {rc}")
                res["mutants"].append({"name": name, "desc": desc, "ok": ok, "expect": ["'클립' 줄의 배포 값이 바뀜(판정은 그대로 불일치 — 6 항 중 1 항만 고친 것)"],
                                       "hit": changed, "n_flipped": len(flipped), "rc": rc, "lost_to": lost_to, "note": "반대 방향 대조"})
            else:
                hit = [e for e in expect if any(e in it for it in flipped)]
                ok = bool(hit) and rc == 1                   # 기대 항목이 잡히고, 회귀 관문(--allow-known)이 막아야 한다
                print(f"[{'PASS' if ok else 'FAIL'}] {name}: {desc} → 새로 불일치가 된 항목 {len(flipped)}: {flipped[:9]} · 관문 종료 코드 {rc}" + (f" · 기지 줄의 변화 {lost_to}" if lost else ""))
                res["mutants"].append({"name": name, "desc": desc, "ok": ok, "expect": expect, "hit": hit, "n_flipped": len(flipped), "flipped": flipped, "rc": rc, "lost_to": lost_to, "note": ""})
            fails += 0 if ok else 1
        print("mutants:", "ALL PASS" if fails == 0 else f"{fails} FAILED")
        res["fails"] = fails
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=1)
        return 0 if fails == 0 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
