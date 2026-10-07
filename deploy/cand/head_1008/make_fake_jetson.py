#!/usr/bin/env python3
"""가짜 젯슨 홈 만들기 — 젯슨 인벤토리(md5)대로 저장소 사본을 로컬에 재구성한다. **로컬 시험 전용, 젯슨 접속 없음.**

  python3 deploy/cand/head_1008/make_fake_jetson.py --out <새 폴더> [--policy-root <logs/ 가 있는 폴더>] [--hwmap <맵 파일>]

결과(<out> = 젯슨의 /home/mama 자리):
  <out>/humanoid_repo/   인벤토리의 파일. 내용은 md5 가 같은 것을 ① 워크트리 같은 경로 ② deploy/cand/c5krh_1007(1007 패치본)
                         ③ 정책 폴더(--policy-root) ④ 그 경로의 git 이력(구판) 순으로 찾아 복사한다.
                         못 찾은 코드·설정은 자리표시 파일(md5 다름)로 두고, 못 찾은 구판 정책(logs/)은 만들지 않는다
                         (자리표시를 두면 브리지 selftest 가 그것을 정책으로 읽으려다 실패한다).
  <out>/rl_calib/        자리표시 파일들 — 배포 스크립트가 이 폴더를 건드리지 않는지 확인하는 용도.
  <out>/logs, <out>/run  빈 폴더.
  <out>/.fake/           배포 스크립트의 가짜 모드가 읽는 프로세스 목록·콘솔 상태·서비스 정보(기본 = 제어 프로세스 없음·콘솔 유휴).
  <out>/.fake/REBUILD.txt 재구성 결과(파일별 출처, 자리표시 목록).

하드웨어맵: 젯슨의 맵(영점 정본)은 저장소에 없다 → --hwmap 으로 준 파일(기본: 워크트리의 config/robot_12dof_hardware_map.json)을
형식만 맞는 대역으로 둔다(md5 는 젯슨과 다르다 — 배포가 이 파일을 바꾸지 않는지 확인하는 표지 역할도 한다).
git 은 읽기만 쓴다(log·rev-parse·cat-file, cwd = 이 저장소)."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
HWMAP_REL = "config/robot_12dof_hardware_map.json"
CALIB_PLACEHOLDERS = ["robot_12dof_hardware_map.json", "level_pose.json", "imu_mount_calib.json", "last_pose_journal.json", "can_channel",
                      "joint_limits_12dof.json", "boot_pose_check.py", "hang_stand.py", "release_all.py", "stage8_cmd.py", "can_up.sh",
                      "sensors_up.sh", "stack_down.sh", "ping_all.py", "imu_adapter_node.py", "foot_force_adapter_node.py",
                      "ak_mit_command.py", "ak_mit_decoder.py"]


def md5_bytes(data):
    return hashlib.md5(data).hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def find_in_history(path, want):
    """그 경로를 바꾼 커밋들을 새것부터 훑어 md5 가 want 인 판의 내용과 (커밋, 날짜)를 돌려준다. 없으면 (None, None)."""
    try:
        log = git("log", "--format=%H %cs", "--", path).decode().splitlines()
    except subprocess.CalledProcessError:
        return None, None
    seen = set()
    for line in log:
        commit, date = line.split()[:2]
        try:
            blob = git("rev-parse", f"{commit}:{path}").decode().strip()
        except subprocess.CalledProcessError:
            continue
        if blob in seen:
            continue
        seen.add(blob)
        data = git("cat-file", "blob", blob)
        if md5_bytes(data) == want:
            return data, f"{commit[:8]} {date}"
    return None, None


def read_inventory(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            m, p = line.split(None, 1)
            rows.append((p[2:] if p.startswith("./") else p, m))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="만들 폴더(없어야 한다 — 기존 폴더를 지우지 않는다)")
    ap.add_argument("--inventory", default=os.path.join(HERE, "jetson_inventory_before_md5.txt"))
    ap.add_argument("--policy-root", default=REPO, help="logs/rsl_rl/... 이 있는 폴더(기본: 저장소 루트)")
    ap.add_argument("--hwmap", default=os.path.join(REPO, HWMAP_REL), help="가짜 젯슨 하드웨어맵으로 둘 파일(형식 대역)")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        print(f"이미 있음: {out} — 지우지 않는다. 새 경로를 주거나 직접 지운 뒤 다시 실행", file=sys.stderr)
        return 2
    repo_out = os.path.join(out, "humanoid_repo")
    os.makedirs(repo_out)
    report, placeholders, omitted = [], [], []
    cand = os.path.join(REPO, "deploy", "cand", "c5krh_1007")
    for rel, want in read_inventory(a.inventory):
        dst = os.path.join(repo_out, rel)
        if not rel.startswith("logs/"):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        src_note = None
        if rel == HWMAP_REL:
            shutil.copyfile(a.hwmap, dst)
            src_note = f"대역({os.path.relpath(a.hwmap, REPO)} — 젯슨 맵 {want[:8]} 은 저장소에 없음)"
            placeholders.append(rel)
        else:
            wt = os.path.join(REPO, rel)
            pol = os.path.join(a.policy_root, rel)
            cnd = os.path.join(cand, os.path.basename(rel))
            if os.path.isfile(wt) and md5_file(wt) == want:
                os.makedirs(os.path.dirname(dst), exist_ok=True)      # 1008: logs/ 가 있는 메인 체크아웃에서 돌리면 정책 파일이 이 가지로 온다(폴더가 아직 없음)
                shutil.copyfile(wt, dst)
                src_note = "워크트리(HEAD 와 같음)"
            elif rel.startswith("jetson/scripts/") and os.path.isfile(cnd) and md5_file(cnd) == want:
                shutil.copyfile(cnd, dst)
                src_note = "deploy/cand/c5krh_1007(1007 패치본)"
            elif rel.startswith("logs/") and os.path.isfile(pol) and md5_file(pol) == want:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(pol, dst)
                src_note = "정책 폴더(로컬과 같음)"
            else:
                data, where = (None, None) if rel.startswith("logs/") else find_in_history(rel, want)
                if data is not None:
                    with open(dst, "wb") as f:
                        f.write(data)
                    src_note = f"git 이력 {where}"
                elif rel.startswith("logs/"):
                    # 로컬에 없는 구판 정책: 자리표시 파일을 두면 브리지 selftest 가 그것을 정책으로 읽으려다 실패한다 → 만들지 않는다
                    src_note = "생략(로컬에 같은 md5 없음 — 가짜 트리에 만들지 않음)"
                    omitted.append(rel)
                else:
                    with open(dst, "w", encoding="utf-8") as f:
                        f.write(f"FAKE-PLACEHOLDER {rel} jetson_md5={want}\n")
                    src_note = "자리표시(내용 없음)"
                    placeholders.append(rel)
        report.append(f"{want}  {rel}  ← {src_note}")
    calib = os.path.join(out, "rl_calib")
    os.makedirs(calib)
    for name in CALIB_PLACEHOLDERS:
        with open(os.path.join(calib, name), "w", encoding="utf-8") as f:
            f.write(f"FAKE rl_calib/{name} — 배포 스크립트는 이 파일을 건드리면 안 된다\n")
    for d in ("logs", "run", ".fake"):
        os.makedirs(os.path.join(out, d))
    fake = os.path.join(out, ".fake")
    with open(os.path.join(fake, "ps.txt"), "w", encoding="utf-8") as f:      # 가짜 `ps -eo pid,args` — 콘솔 서비스만
        f.write("    PID COMMAND\n    916 python3 /home/mama/humanoid_repo/jetson/scripts/operator_console.py --port 8800\n")
    state = {"session": {"active": False}, "bridge": {"live": False, "armed": False}, "procs": {}, "trial": {"phase": "준비"}}
    with open(os.path.join(fake, "state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False)
    with open(os.path.join(fake, "systemd.txt"), "w", encoding="utf-8") as f:   # KillMode·활성 상태·MainPID
        f.write("KillMode=process\nActiveState=active\nMainPID=916\n")
    with open(os.path.join(fake, "REBUILD.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    n_ph, n_om = len(placeholders), len(omitted)
    print(f"가짜 젯슨 홈: {out}")
    print(f"  humanoid_repo: 인벤토리 {len(report)} 파일 중 md5 재현 {len(report) - n_ph - n_om} · 자리표시/대역 {n_ph} {placeholders} · "
          f"생략 {n_om}(로컬에 없는 구판 정책)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
