#!/usr/bin/env python3
"""HEAD 동기화 배포(head_1008)의 매니페스트·계획서·차이 목록을 **한 명령으로** 다시 만든다. 젯슨에 접속하지 않는다.

  python3 deploy/make_head_manifest.py                      # 생성(덮어쓰기)
  python3 deploy/make_head_manifest.py --check              # 만들지 않고 지금 파일들이 최신인지 확인(다르면 종료 코드 1)
  python3 deploy/make_head_manifest.py --policy-root <logs/ 가 있는 폴더>   # 워크트리처럼 logs/ 가 없는 곳에서

콘솔·브리지 등 실행 파일이 바뀌면(다른 작업 병합 뒤) 이 명령을 다시 돌리고, 이어서 로컬 왕복 시험(deploy/test_head_1008_local.sh)을 돌린다.

만드는 것
  deploy/head_1008.md5                         배포 뒤 젯슨 저장소 사본이 가져야 할 md5(전송 + 확인만). 젯슨에서 `md5sum -c` 가능.
  deploy/head_1008.plan                        배포 스크립트가 읽는 계획서(탭 구분): 경로·동작(send|verify)·묶음·원본 위치·새 md5·크기·젯슨 현재 md5
  deploy/head_1008.VERSION                     기준 커밋·인벤토리·매니페스트 md5(젯슨에도 둔다)
  deploy/cand/head_1008/ref_bridge/*.ref       selftest_w3_bridge_node 의 기준 커밋 브리지 원문(젯슨에는 git 이 없어 파일로 싣는다)
  deploy/cand/head_1008/DIFF_LIST.md           차이 목록(전송·확인만·제외·젯슨 전용·import 사슬·정책 파일) — 사람이 읽는 표

입력
  deploy/cand/head_1008/jetson_inventory_before_md5.txt   젯슨 `~/humanoid_repo` 의 실제 md5(주 작업자가 읽기 전용으로 뽑은 것). '젯슨 현재 md5'·'기대 구 md5' 의 원본.
  HEAD 워크트리의 실행 파일, 정책 파일(logs/rsl_rl/…), git 이력(젯슨 구판의 출처 확인·기준 브리지 원문).

파일 집합(묶음)
  session  세션 진입점(콘솔·브리지·stage8)의 import 사슬 — 실제로 ast 로 따라간다 + 콘솔 UI
  tool     운영 도구·selftest 진입점과 그 사슬(robotctl·w3_cmd·latency_check·selftest_w3_bridge_node·posture_probe·zero_* …)
  test     오프라인 단위 시험 — TEST_GLOBS 무늬(tests/test_stage8_*·test_console_*·test_operator_console·…)에 맞는 추적 파일은 자동
  sync     그 밖의 jetson/scripts·rl_walking/deploy·robot_runtime·protocols(저장소 사본을 HEAD 와 맞춘다) — SYNC_GLOBS, 추적 파일은 자동
  policy   지정한 키가 읽는 정책 파일(브리지의 CHECKPOINTS·C5_PAIRS·KEW3_SPECS 에서 계산) + --extra-policy-run 으로 준 폴더
  ref      기준 브리지 원문 + 실행기(selftest_w3_node_refs.py)
  record   매니페스트·VERSION·운영 명령 문서
  parity   reports/night_1007/jetson_parity.py + jetson/parity_ref.npz (둘 다 있을 때만 — 젯슨 게이트가 --check 로 돌린다)
새 파일이 자동으로 들어가는 경우: 진입점이 import 하는 모듈(사슬 추적), 위 글로브에 맞는 **git 추적** 파일. 손으로 넣어야 하는 경우:
  새 진입점(ENTRY_*), 글로브 밖 폴더의 파일, 새 정책 키(POLICY_KEYS 또는 --keys / --extra-policy-run), 허용 폴더 밖 경로(forbidden_reason 도 함께).
어느 커밋에서 돌려도 같은 규칙으로 다시 만든다(입력 = 그 시점 워크트리 + 인벤토리 파일 + 정책 폴더). 추적되지 않은 새 파일은 글로브에 안 잡힌다 — 커밋 뒤에 돌릴 것.
**절대 넣지 않는 것**(코드로 강제 — forbidden_reason): config/ 전체(하드웨어맵·영점·IMU 장착 보정·지연 옵션), jetson/calib_data/,
  ~/rl_calib, level_pose·journal·calib 류 파일. 걸리면 생성을 중단한다. 배포 스크립트도 같은 규칙을 따로 검사한다."""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import importlib.util
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NAME = "head_1008"
CAND = f"deploy/cand/{NAME}"
OUT_MD5 = f"deploy/{NAME}.md5"
OUT_PLAN = f"deploy/{NAME}.plan"
OUT_VERSION = f"deploy/{NAME}.VERSION"
OUT_DIFF = f"{CAND}/DIFF_LIST.md"
REF_DIR = f"{CAND}/ref_bridge"
INVENTORY = f"{CAND}/jetson_inventory_before_md5.txt"

#: 세션 진입점 — 콘솔이 저장소 사본(SCRIPTS)에서 띄우는 것과 콘솔 자신
ENTRY_SESSION = ["jetson/scripts/operator_console.py", "jetson/scripts/rl_bridge_node.py",
                 "jetson/scripts/stage8_12axis_mit_control_node.py"]
DATA_SESSION = ["jetson/scripts/console_ui.html"]
#: 운영 도구·selftest 진입점(콘솔이 SCRIPTS 에서 띄우는 보조 도구 포함)
ENTRY_TOOL = ["jetson/scripts/robotctl.py", "jetson/scripts/w3_cmd.py", "jetson/scripts/latency_check.py",
              "jetson/scripts/selftest_w3_bridge_node.py", "jetson/scripts/posture_probe.py", "jetson/scripts/zero_kinematic_check.py",
              "jetson/scripts/zero_anchor.py", "jetson/scripts/analyze_fall.py",
              "rl_walking/deploy/kew3_runner.py", "rl_walking/deploy/c5_switch.py", "rl_walking/deploy/policy_runner_stage4.py",
              "rl_walking/deploy/w3_contingency.py"]
#: 오프라인 단위 시험 — 글로브(추적 파일만). 새 시험 파일이 이 무늬에 맞으면 자동으로 들어간다(예: tests/test_console_walk*.py).
#: 젯슨 게이트가 실제로 돌리는 것은 배포 스크립트의 selftest 목록에 적힌 것뿐이다 — 여기는 "파일을 싣는다"까지.
TEST_GLOBS = ["tests/test_stage8_*.py", "tests/test_latency_check.py", "tests/test_dof12_mapping_cache.py",
              "tests/test_operator_console.py", "tests/test_console_*.py", "tests/test_startup_simplify_bridge.py", "tests/test_robotctl.py"]
#: 배포 뒤 젯슨에서 돌릴 러너 동치 확인(로컬 torch 기준 ↔ 젯슨 onnxruntime) — 허용 폴더 밖이라 이 두 경로만 명시로 허용한다. 있으면 싣는다.
PARITY_FILES = ["reports/night_1007/jetson_parity.py", "reports/night_1007/jetson/parity_ref.npz"]
SYNC_GLOBS = ["jetson/scripts/*.py", "jetson/scripts/*.sh", "jetson/scripts/*.html", "rl_walking/__init__.py", "rl_walking/deploy/*.py",
              "robot_runtime/*.py", "protocols/*.py"]
REF_FILES = [f"{CAND}/selftest_w3_node_refs.py"]
RECORD_FILES = [f"{CAND}/OPERATOR_COMMANDS.md"]
#: 정책 파일을 실어야 하는 키(사용자 지시의 세션 키). 여기에 브리지 STAND_DEFAULT(selftest 필수)와 콘솔 선택지 전부가 '확인' 대상으로 더해진다.
#: 1008: 둘째 합성 키(첫 합성 키 + KE 감지기 갇힘 해소)를 뒤에 더했다 — 읽는 정책 파일은 첫 합성 키와 같은 세 run 이라 묶음에 새 파일은 늘지 않는다
#: (콘솔 선택지에 올라가므로 '확인' 대상 키 목록·젯슨 selftest 인자와 맞춘다).
#: 1008 오후: 공중·착지 서기 정책(G21h) 단독 키와 그것을 쓰는 셋째 합성 키 — G21h 정책 폴더(exported + params/env.yaml)가 새로 실린다.
POLICY_KEYS = ["stand_crouch_g21a", "stand_c5kr_g21a_g24ra", "stand_c5krh_g21a_g24ra", "stand_w3s3d_15800",
               "stand_kew3_g21a_g24ra_w3s3d15800", "stand_kew3q_g21a_g24ra_w3s3d15800",
               "stand_crouch_g21h", "stand_kew3qa_g21a_g24ra_w3s3d15800",
               "stand_w3s3dnbj2_16299", "stand_kew3qa_g21a_g24ra_w3nbj2_16299"]      # 1010 새 몸 종종걸음(배포 계약 폴더 deploy_16299)
#: 진입점별 import 사슬 표에만 쓰는 목록(젯슨에서 ~/rl_calib 사본으로 도는 것 — 저장소 사본 기준으로 사슬을 적는다)
ENTRY_CALIB = ["jetson/scripts/hang_stand.py", "jetson/scripts/boot_pose_check.py", "jetson/scripts/release_all.py",
               "jetson/scripts/stage8_cmd.py", "jetson/scripts/ping_all.py", "jetson/scripts/imu_adapter_node.py",
               "jetson/scripts/foot_force_adapter_node.py"]

ALLOWED_PREFIXES = ("jetson/scripts/", "rl_walking/", "robot_runtime/", "protocols/", "tests/", "logs/rsl_rl/", "deploy/")
#: 코드·문서·기록 확장자만 보낸다. **.json·.yaml(정책 params/env.yaml 제외)·확장자 없는 파일은 전부 금지** — 하드웨어맵·영점·보정·
#: 수평 자세·저널·CAN 채널 설정이 전부 그런 파일이다.
ALLOWED_EXT = (".py", ".sh", ".html", ".md", ".ref", ".md5", ".VERSION")
FORBIDDEN_NAME_RE = re.compile(r"(hardware_map|level_pose|imu_mount_calib|pose_journal|stand_snapshot|stand_zero|foot_force_calib|"
                               r"latency_opts|rl_calib)", re.I)
POLICY_FILE_RE = re.compile(r"^logs/rsl_rl/[A-Za-z0-9_]+/[A-Za-z0-9_.-]+/(exported/(policy\.pt|policy\.onnx|SOURCE_CHECKPOINT)|params/env\.yaml)$")
SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def forbidden_reason(path):
    """배포 금지 사유(문자열) 또는 None. 배포 스크립트의 forbidden_path() 와 같은 규칙(두 곳에서 따로 검사한다)."""
    if not SAFE_PATH_RE.match(path) or path.startswith("/") or ".." in path.split("/") or path.startswith("~"):
        return "경로 형식(절대 경로·..·허용 밖 문자)"
    if path in PARITY_FILES:
        return None
    if path.startswith("config/"):
        return "config/ — 하드웨어맵·영점·보정·지연 옵션 파일(젯슨이 정본)"
    if path.startswith("jetson/calib_data/"):
        return "jetson/calib_data/ — 보정 자료"
    if not path.startswith(ALLOWED_PREFIXES):
        return "허용 목록 밖 폴더"
    if path.startswith("logs/"):
        return None if POLICY_FILE_RE.match(path) else "logs/ 아래는 정책 exported·params/env.yaml 만"
    if not path.endswith(ALLOWED_EXT):
        return "허용 확장자 밖(.json·.yaml·확장자 없는 파일은 보내지 않는다 — 맵·영점·보정 파일 차단)"
    if FORBIDDEN_NAME_RE.search(path) and not path.endswith((".py", ".sh")):
        return "하드웨어맵·영점·보정 류 이름"
    return None


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args, check=True):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=check).stdout


# ------------------------------------------------------------------ import 사슬(정적)
def _resolve(mod, level, cur_file):
    cands = []
    if level > 0:
        base = os.path.dirname(cur_file)
        for _ in range(level - 1):
            base = os.path.dirname(base)
        cands.append(base)
    else:
        cands = [os.path.dirname(cur_file), REPO]
    parts = mod.split(".") if mod else []
    for c in cands:
        p, ok, chain = c, True, []
        for i, part in enumerate(parts):
            d = os.path.join(p, part)
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "__init__.py")):
                chain.append(os.path.join(d, "__init__.py"))
                p = d
            elif os.path.isfile(d + ".py") and i == len(parts) - 1:
                chain.append(d + ".py")
            elif os.path.isdir(d) and i < len(parts) - 1:
                p = d
            else:
                ok = False
                break
        if ok and (chain or not parts):
            if not parts and level > 0 and os.path.isfile(os.path.join(c, "__init__.py")):
                chain.append(os.path.join(c, "__init__.py"))
            return chain
    return []


def import_closure(entry):
    """진입점의 저장소 안 import 사슬(자기 포함, 저장소 상대 경로)과 저장소에서 못 푼 최상위 모듈 이름 집합."""
    seen, unresolved = {}, set()
    stack = [os.path.join(REPO, entry)]
    std = set(sys.stdlib_module_names)
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen[f] = True
        try:
            tree = ast.parse(open(f, encoding="utf-8").read(), f)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                items = [(a.name, 0, []) for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                items = [(node.module or "", node.level, [a.name for a in node.names])]
            else:
                continue
            for mod, level, names in items:
                chain = _resolve(mod, level, f)
                if not chain:
                    top = mod.split(".")[0] if mod else ""
                    if top and top not in std and top != "__future__":
                        unresolved.add(top)
                    continue
                stack += [c for c in chain if c not in seen]
                last = chain[-1]
                if os.path.basename(last) == "__init__.py":
                    for n in names:
                        sub = os.path.join(os.path.dirname(last), n + ".py")
                        subpkg = os.path.join(os.path.dirname(last), n, "__init__.py")
                        if os.path.isfile(sub) and sub not in seen:
                            stack.append(sub)
                        elif os.path.isfile(subpkg) and subpkg not in seen:
                            stack.append(subpkg)
    return sorted(os.path.relpath(p, REPO) for p in seen), unresolved


# ------------------------------------------------------------------ 입력
def read_inventory(path):
    inv = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                m, p = line.split(None, 1)
                inv[p[2:] if p.startswith("./") else p] = m
    return inv


def history_origin(path, want):
    """젯슨 md5 가 그 경로 git 이력의 어느 판인지 → "커밋8 날짜"(없으면 None)."""
    log = git("log", "--format=%H %cs", "--", path, check=False).decode().splitlines()
    seen = set()
    for line in log:
        commit, date = line.split()[:2]
        blob = git("rev-parse", f"{commit}:{path}", check=False).decode().strip()
        if not blob or blob in seen:
            continue
        seen.add(blob)
        if hashlib.md5(git("cat-file", "blob", blob)).hexdigest() == want:
            return f"{commit[:8]} {date}"
    return None


def load_bridge():
    """HEAD 브리지 모듈을 읽어 정책 경로 레지스트리를 얻는다(실험 키 등록 포함). ROS·CAN·GPU 를 쓰지 않는다(모듈 import 만)."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    path = os.path.join(REPO, "jetson", "scripts", "rl_bridge_node.py")
    spec = importlib.util.spec_from_file_location("rl_bridge_node_manifest", path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.argv
    sys.argv = ["rl_bridge_node.py"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old
    mod.enable_experimental_w3()
    mod.enable_experimental_kew3()
    return mod


def console_choices():
    src = open(os.path.join(REPO, "jetson/scripts/operator_console.py"), encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CKPT_CHOICES" for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise RuntimeError("operator_console.py 에서 CKPT_CHOICES 를 못 찾음")


def console_launch_table():
    """콘솔이 어느 폴더의 어떤 스크립트를 띄우는가 — os.path.join(CALIB|SCRIPTS, "이름") 을 원문에서 뽑는다."""
    src = open(os.path.join(REPO, "jetson/scripts/operator_console.py"), encoding="utf-8").read()
    calib = sorted(set(re.findall(r'os\.path\.join\(CALIB,\s*"([^"]+)"\)', src)))
    scripts = sorted(set(re.findall(r'os\.path\.join\(SCRIPTS,\s*"([^"]+)"\)', src)))
    return calib, scripts


def policy_files(b, key):
    """키가 읽는 정책 파일 [(저장소 상대 경로, 필수 여부, 설명)] — 브리지 레지스트리에서 계산."""
    if key in b.KEW3_CKPTS:
        sp = b.KEW3_CKPTS[key]
        return policy_files(b, sp.w3) + policy_files(b, sp.ke)
    dirs = [b.CHECKPOINTS[key]]
    if key in b.C5_PAIRS:
        stand, rec = b.C5_PAIRS[key]
        dirs = [b.CHECKPOINTS[stand], rec]
    out = []
    need_env = key in b.CKPT_REQUIRE_ENV_YAML or key in b.C5_PAIRS
    for d in dirs:
        rel = os.path.relpath(d, b.REPO)
        run = os.path.dirname(rel)
        out.append((f"{rel}/policy.onnx", True, "추론(젯슨 백엔드 = onnxruntime)"))
        out.append((f"{rel}/policy.pt", False, "torch 백엔드용(젯슨에는 torch 없음 — 종전 관례대로 함께 둔다)"))
        out.append((f"{rel}/SOURCE_CHECKPOINT", False, "출처 표시(실행 중 읽지 않음)"))
        out.append((f"{run}/params/env.yaml", need_env, "학습 계약(관측 필터·기본자세·W3 명령 범위) — 브리지가 기동·selftest 때 읽는다"))
    return out


def ref_revs():
    src = open(os.path.join(REPO, "jetson/scripts/selftest_w3_bridge_node.py"), encoding="utf-8").read()
    revs = re.findall(r'add_argument\("--ref[a-z0-9-]*",\s*default="([0-9a-f]{7,40})"', src)
    if not revs:
        raise RuntimeError("selftest_w3_bridge_node.py 에서 --ref* 기본 커밋을 못 찾음")
    return revs


# ------------------------------------------------------------------ 본체
def build(args):
    inv = read_inventory(os.path.join(REPO, args.inventory))
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    entries = {}          # path → dict
    closures = {}

    def add(path, group, root="repo", note=""):
        if path in entries:
            return
        why = forbidden_reason(path)
        if why:
            raise SystemExit(f"[중단] 배포 금지 파일이 목록에 들어옴: {path} — {why}")
        entries[path] = {"path": path, "group": group, "root": root, "note": note}

    for e in ENTRY_SESSION + ENTRY_TOOL + ENTRY_CALIB:
        closures[e] = import_closure(e)
    if "session" in groups:
        for e in ENTRY_SESSION:
            for p in closures[e][0]:
                add(p, "session")
        for p in DATA_SESSION:
            add(p, "session")
    if "tool" in groups:
        for e in ENTRY_TOOL:
            for p in closures[e][0]:
                add(p, "tool")
    tracked = set(git("ls-files").decode().splitlines())
    if "test" in groups:
        for pat in TEST_GLOBS:
            for full in sorted(glob.glob(os.path.join(REPO, pat))):
                p = os.path.relpath(full, REPO)
                if p in tracked:
                    add(p, "test")
    if "sync" in groups:
        for pat in SYNC_GLOBS:
            for full in sorted(glob.glob(os.path.join(REPO, pat))):
                p = os.path.relpath(full, REPO)
                if p in tracked:
                    add(p, "sync")
    b = load_bridge()
    keys_send = list(POLICY_KEYS if args.keys is None else [k for k in args.keys.split(",") if k])
    keys_verify = [b.STAND_DEFAULT] + console_choices()
    policy_table = {}
    if "policy" in groups:
        for key in keys_send + [k for k in keys_verify if k not in keys_send]:
            strict = key in keys_send            # 실으라고 지정한 키는 엄격, 콘솔 선택지·기본 키(확인용)는 로컬에 없으면 건너뛴다
            if key not in b.CHECKPOINTS:
                if strict:
                    raise SystemExit(f"[중단] 브리지에 없는 정책 키: {key}")
                print(f"  (확인 생략) 브리지 레지스트리에 없는 콘솔 선택지: {key}")
                continue
            rows = []
            missing = [rel for rel, required, what in policy_files(b, key) if required and not os.path.isfile(os.path.join(args.policy_root, rel))]
            if missing and not strict:
                print(f"  (확인 생략) {key}: 로컬에 정책 파일 없음 {missing[0]}")
                continue
            for rel, required, what in policy_files(b, key):
                full = os.path.join(args.policy_root, rel)
                if not os.path.isfile(full):
                    if required:
                        raise SystemExit(f"[중단] {key}: 필수 정책 파일 없음 {full} (--policy-root 확인)")
                    continue
                run = rel.split("/exported/")[0].split("/params/")[0]
                if not required and rel not in inv and any(p.startswith(run + "/") for p in inv):
                    continue      # 젯슨에 이미 있는 run 에는 선택 파일(출처 표시 등)을 새로 얹지 않는다 — 기존 정책 폴더는 건드리지 않는다
                rows.append((rel, required, what))
                add(rel, "policy", root="policy", note=key if key in keys_send else f"{key}(확인)")
            policy_table[key] = rows
        # --extra-policy-run: 브리지 키와 무관하게 정책 폴더 하나를 더 싣는다(밤사이 새 후보 — 키 등록 코드는 따로 병합돼 있어야 그 키로 돈다)
        for run in [r.strip().rstrip("/") for r in (args.extra_policy_run or "").split(",") if r.strip()]:
            rows = []
            for sub, required in (("exported/policy.onnx", True), ("exported/policy.pt", False), ("exported/SOURCE_CHECKPOINT", False), ("params/env.yaml", True)):
                rel = f"{run}/{sub}"
                if not os.path.isfile(os.path.join(args.policy_root, rel)):
                    if required:
                        raise SystemExit(f"[중단] --extra-policy-run {run}: 필수 파일 없음 {rel} (--policy-root 확인)")
                    continue
                rows.append((rel, required, "추가 정책 폴더"))
                add(rel, "policy", root="policy", note=f"추가 폴더 {os.path.basename(run)}")
            policy_table[f"(추가 폴더) {os.path.basename(run)}"] = rows
            keys_send.append(f"(추가 폴더) {os.path.basename(run)}")
    revs = ref_revs()
    if "ref" in groups:
        os.makedirs(os.path.join(REPO, REF_DIR), exist_ok=True)
        want = set()
        for rev in revs:
            data = git("show", f"{rev}:jetson/scripts/rl_bridge_node.py")
            rel = f"{REF_DIR}/rl_bridge_node.{rev}.ref"
            want.add(rel)
            full = os.path.join(REPO, rel)
            if not (os.path.isfile(full) and open(full, "rb").read() == data):
                if args.check:
                    raise SystemExit(f"[check] 기준 브리지 원문이 없거나 다름: {rel}")
                with open(full, "wb") as f:
                    f.write(data)
            add(rel, "ref", note=f"기준 커밋 {rev} 의 jetson/scripts/rl_bridge_node.py 원문")
        for full in glob.glob(os.path.join(REPO, REF_DIR, "*.ref")):
            if os.path.relpath(full, REPO) not in want:
                if args.check:
                    raise SystemExit(f"[check] 쓰이지 않는 기준 원문이 남아 있음: {os.path.relpath(full, REPO)}")
                os.remove(full)
        for p in REF_FILES:
            add(p, "ref")
    if "record" in groups:
        for p in RECORD_FILES:
            if os.path.isfile(os.path.join(REPO, p)):
                add(p, "record")
    if "parity" in groups:
        have = [p for p in PARITY_FILES if os.path.isfile(os.path.join(REPO, p))]
        if len(have) == len(PARITY_FILES):          # 도구와 기준 npz 가 둘 다 있을 때만(하나만 가면 젯슨 게이트가 돌릴 수 없다)
            for p in have:
                add(p, "parity", note="러너 동치 확인(로컬 기준 ↔ 젯슨 onnxruntime)")
        elif have:
            print(f"  (parity 생략) {PARITY_FILES} 중 {have} 만 있다 — 둘 다 있어야 싣는다")

    # md5·크기·젯슨 상태
    for e in entries.values():
        full = os.path.join(args.policy_root if e["root"] == "policy" else REPO, e["path"])
        if not os.path.isfile(full):
            raise SystemExit(f"[중단] 원본 없음: {full}")
        e["md5"], e["size"] = md5_file(full), os.path.getsize(full)
        e["jetson"] = inv.get(e["path"], "-")
        e["action"] = "verify" if e["jetson"] == e["md5"] else "send"
        if e["action"] == "send" and e["jetson"] != "-" and e["root"] == "repo":
            org = history_origin(e["path"], e["jetson"])
            if org is None and os.path.isfile(os.path.join(REPO, "deploy/cand/c5krh_1007", os.path.basename(e["path"]))) \
                    and md5_file(os.path.join(REPO, "deploy/cand/c5krh_1007", os.path.basename(e["path"]))) == e["jetson"]:
                org = "8af2f6e5 + 1007 krh 패치(deploy/cand/c5krh_1007)"
            if org is None:
                raise SystemExit(f"[중단] {e['path']}: 젯슨 판({e['jetson'][:8]})이 git 이력에 없다 — 젯슨 로컬 수정본일 수 있어 덮어쓰지 않는다."
                                 " 확인 뒤 인벤토리를 갱신하거나 이 파일을 묶음에서 빼라")
            e["note"] = (e["note"] + " " if e["note"] else "") + f"젯슨 구판 = {org}"
    return entries, closures, policy_table, revs, inv, b, keys_send, keys_verify


def render_md5(entries):
    return "".join(f"{e['md5']}  {e['path']}\n" for e in sorted(entries.values(), key=lambda x: x["path"]))


def render_version(entries, md5_text):
    # 이 파일이 자기 자신이 들어가는 커밋의 해시를 담을 수는 없다 → "실행 코드(session·tool·test·sync)를 마지막으로 바꾼 커밋"을 적는다.
    # deploy/ 아래 산출물만 커밋해도 값이 변하지 않아 --check 가 안정적이다.
    paths = sorted(e["path"] for e in entries.values() if e["group"] in ("session", "tool", "test", "sync"))
    src = git("log", "-1", "--format=%H %cI", "--", *paths).decode().strip() or "unknown"
    dirty = [ln[3:] for ln in git("status", "--porcelain", "--", *paths).decode().splitlines() if ln.strip()]
    inv_md5 = md5_file(os.path.join(REPO, INVENTORY))
    n_send = sum(1 for e in entries.values() if e["action"] == "send")
    lines = [f"package: {NAME}",
             f"source_commit: {src}",
             "source_uncommitted: " + (", ".join(dirty) if dirty else "none"),
             f"manifest: {OUT_MD5} md5 {hashlib.md5(md5_text.encode()).hexdigest()}",
             f"files: {len(entries) + 2} (send {n_send + 2} incl. this file and the manifest, verify {len(entries) - n_send})",
             f"jetson_inventory: {INVENTORY} md5 {inv_md5}",
             "note: 파일 md5(매니페스트)가 정본이다. source_commit = 실행 코드(콘솔·브리지·stage8·러너·도구·시험)를 마지막으로 바꾼 커밋,",
             "      source_uncommitted = 생성 시점에 그 커밋과 달랐던 실행 코드 파일(none 이어야 커밋과 일치)."]
    return "\n".join(lines) + "\n"


def render_plan(entries, version_text, md5_text, keys_send):
    order = {"session": 0, "tool": 1, "test": 2, "sync": 3, "policy": 4, "ref": 5, "record": 6, "parity": 7}
    rows = sorted(entries.values(), key=lambda e: (order[e["group"]], e["path"]))
    out = [f"# {NAME} 배포 계획서 — deploy/make_head_manifest.py 가 만든다(손으로 고치지 말 것).",
           "# 열: 경로\t동작(send=젯슨과 다르거나 없음 → 백업 뒤 복사 | verify=젯슨과 같아야 함, 복사 안 함)\t묶음\t원본 위치(repo|policy)\t새 md5\t크기\t젯슨 현재 md5(- = 없음)\t비고",
           f"# keys={','.join(keys_send)}"]
    for e in rows:
        out.append("\t".join([e["path"], e["action"], e["group"], e["root"], e["md5"], str(e["size"]), e["jetson"], e["note"] or "-"]))
    for path, text in ((OUT_MD5, md5_text), (OUT_VERSION, version_text)):
        out.append("\t".join([path, "send", "record", "repo", hashlib.md5(text.encode()).hexdigest(), str(len(text.encode())), "-",
                              "매니페스트(젯슨에서 md5sum -c)" if path == OUT_MD5 else "버전 기록"]))
    return "\n".join(out) + "\n"


def render_diff(entries, closures, policy_table, revs, inv, b, keys_send, keys_verify, version_text):
    E = sorted(entries.values(), key=lambda e: e["path"])
    send_old = [e for e in E if e["action"] == "send" and e["jetson"] != "-"]
    send_new = [e for e in E if e["action"] == "send" and e["jetson"] == "-"]
    verify = [e for e in E if e["action"] == "verify"]
    used_by = {}
    for ent, (files, _un) in closures.items():
        for f in files:
            used_by.setdefault(f, []).append(os.path.basename(ent))
    L = [f"# 차이 목록 — 젯슨 현재 판 ↔ HEAD ({NAME})", "",
         "> `python3 deploy/make_head_manifest.py` 가 만든 파일이다(손으로 고치지 말 것). 젯슨 현재 md5 는",
         f"> `{INVENTORY}`(주 작업자가 젯슨에서 읽기 전용으로 뽑은 목록 173 파일)에서 왔다.", "",
         "```", version_text.rstrip("\n"), "```", "",
         "## 요약", "",
         "| 구분 | 파일 수 | 크기 합 |", "|---|---|---|",
         f"| 전송 — 변경(젯슨에 구판이 있음) | {len(send_old)} | {sum(e['size'] for e in send_old) / 1024:.0f} KB |",
         f"| 전송 — 신규(젯슨에 없음) | {len(send_new)} (+ 매니페스트·VERSION 2) | {sum(e['size'] for e in send_new) / 1024:.0f} KB |",
         f"| 확인만(젯슨과 같음 — 복사하지 않음) | {len(verify)} | {sum(e['size'] for e in verify) / 1024:.0f} KB |",
         "| 삭제 | 0 | — |", "",
         "묶음: session = 콘솔·브리지·stage8 의 import 사슬 · tool = 운영 도구·selftest · test = 오프라인 단위 시험 · sync = 저장소 사본 맞춤 · "
         "policy = 정책 파일 · ref = selftest 기준 원문 · record = 기록·문서", "",
         "## 1. 전송 — 변경 (젯슨의 구판을 백업한 뒤 덮어쓴다)", "",
         "| 경로 | 묶음 | 젯슨 md5 | 새 md5 | 젯슨 판의 출처 | 쓰는 진입점 |", "|---|---|---|---|---|---|"]
    for e in send_old:
        org = e["note"].split("젯슨 구판 = ")[-1] if "젯슨 구판" in e["note"] else "-"
        L.append(f"| `{e['path']}` | {e['group']} | `{e['jetson'][:8]}` | `{e['md5'][:8]}` | {org} | {', '.join(sorted(set(used_by.get(e['path'], [])))[:6]) or '-'} |")
    L += ["", "## 2. 전송 — 신규 (젯슨에 없는 파일, 롤백 때 지운다)", "",
          "| 경로 | 묶음 | 새 md5 | 크기 | 비고 |", "|---|---|---|---|---|"]
    for e in send_new:
        L.append(f"| `{e['path']}` | {e['group']} | `{e['md5'][:8]}` | {e['size']} | {e['note'] or ', '.join(sorted(set(used_by.get(e['path'], [])))[:4]) or '-'} |")
    L += ["", "## 3. 확인만 (젯슨 md5 = HEAD md5 — 배포 때 다시 읽어 같지 않으면 중단)", "",
          "| 경로 | 묶음 | md5 |", "|---|---|---|"]
    for e in verify:
        L.append(f"| `{e['path']}` | {e['group']} | `{e['md5'][:8]}` |")
    # 4. 계획에 없는 차이
    tracked = set(git("ls-files").decode().splitlines())
    L += ["", "## 4. 전송하지 않는 차이 (젯슨 인벤토리에 있고 HEAD 와 다르거나 HEAD 에 없는데 계획에 넣지 않은 것)", "",
          "| 경로 | 젯슨 md5 | HEAD | 사유 |", "|---|---|---|---|"]
    for p, m in sorted(inv.items()):
        if p in entries:
            continue
        if p in tracked:
            h = md5_file(os.path.join(REPO, p))
            if h == m:
                continue
            why = forbidden_reason(p) or ("학습용 코드(젯슨에서 실행하지 않음 — isaaclab 필요)" if p.startswith("rl_walking/") else "실행 사슬 밖")
            L.append(f"| `{p}` | `{m[:8]}` | `{h[:8]}` | {why} |")
        elif p.startswith("logs/"):
            continue
        else:
            L.append(f"| `{p}` | `{m[:8]}` | 없음 | 젯슨 전용(HEAD 에 추적 파일 없음) |")
    pol_other = sorted(p for p in inv if p.startswith("logs/") and p not in entries)
    L += ["", f"젯슨에만 있거나 이번 배포와 무관한 정책 파일 {len(pol_other)} 개(구 세대 exported·env.yaml)는 건드리지 않는다.", "",
          "HEAD 의 `jetson/` 아래에 있지만 계획에 없는 것: `jetson/calib_data/`(보정 자료 — 금지), `jetson/bench/`(개발 PC 벤치), "
          "`jetson/systemd/`(서비스 정의 — 이번 배포는 서비스를 바꾸지 않는다), `jetson/__init__.py`·`README.md`·`deploy_*.sh`(구 배포 도구).", "",
          "## 5. 진입점별 import 사슬 (저장소 사본 기준, ast 로 추적 — 함수 안 지연 import 포함)", ""]
    for title, ents in (("콘솔이 저장소 사본에서 띄우는 것 + 콘솔", ENTRY_SESSION), ("운영 도구·selftest", ENTRY_TOOL),
                        ("젯슨에서 `~/rl_calib` 사본으로 도는 것(저장소 사본의 사슬 — 이번 배포 대상 아님)", ENTRY_CALIB)):
        L += [f"### {title}", "", "| 진입점 | 저장소 안 의존 | 저장소 밖 모듈(제3자·젯슨 전용) |", "|---|---|---|"]
        for ent in ents:
            files, un = closures[ent]
            deps = [f for f in files if f != ent]
            L.append(f"| `{ent}` | {', '.join('`' + d + '`' for d in deps) or '없음'} | {', '.join(sorted(un)) or '없음'} |")
        L.append("")
    calib, scripts = console_launch_table()
    L += ["## 6. 콘솔이 띄우는 스크립트의 위치", "",
          f"- 저장소 사본(`~/humanoid_repo/jetson/scripts`)에서: {', '.join('`' + s + '`' for s in scripts)} → 이번 배포로 바뀐다(위 표).",
          f"- `~/rl_calib` 에서: {', '.join('`' + s + '`' for s in calib)} → **배포 대상이 아니다**(젯슨 정본). 저장소에 같은 이름의 사본이 "
          "`jetson/scripts/` 에 있는지는 아래 표. `~/rl_calib` 의 실제 md5 는 인벤토리에 없다 — 배포 스크립트 dry-run 이 읽어 저장소 사본과 나란히 보여 준다(읽기만).", "",
          "| `~/rl_calib` 의 파일 | 저장소 사본 | HEAD md5 | 젯슨 저장소 사본(인벤토리) |", "|---|---|---|---|"]
    calib_all = sorted(set(calib) | {"stack_down.sh", "ping_all.py", "imu_adapter_node.py", "foot_force_adapter_node.py", "zero_torque_read.py",
                                     "ak_mit_command.py", "ak_mit_decoder.py", "robot_12dof_hardware_map.json", "imu_mount_calib.json",
                                     "last_pose_journal.json", "can_channel", "joint_limits_12dof.json"})
    for name in calib_all:
        cand = [f"jetson/scripts/{name}", f"protocols/{name}", f"config/{name}"]
        rp = next((c for c in cand if c in tracked), None)
        if rp is None:
            stem = name.rsplit(".", 1)[0]
            dated = sorted(t for t in tracked if t.startswith("jetson/calib_data/" + stem + "_"))
            extra = (" (날짜 붙은 옛 사본 " + ", ".join("`" + d + "`" for d in dated) + " 은 있으나 실행본과의 일치는 미확인)") if dated else ""
            L.append(f"| `{name}` | 없음 — **젯슨 정본·저장소 사본 없음**{extra} | - | - |")
        else:
            L.append(f"| `{name}` | `{rp}`{' (배포 금지 — 젯슨 정본)' if forbidden_reason(rp) else ''} | `{md5_file(os.path.join(REPO, rp))[:8]}` | "
                     f"{'`' + inv[rp][:8] + '`' if rp in inv else '없음'} |")
    L += ["", "## 7. 정책 파일", "",
          "브리지가 실제로 읽는 것은 `exported/policy.onnx`(onnxruntime 이 있으면 우선) 또는 `policy.pt`, 그리고 `params/env.yaml`(관측 필터·기본자세·"
          "W3 명령 범위 계약)이다. `SOURCE_CHECKPOINT` 는 출처 표시일 뿐 읽지 않는다. `params/agent.yaml`·`model_*.pt` 는 싣지 않는다.", "",
          "| 파일 | 크기 | md5 | 젯슨(인벤토리) | 동작 | 쓰는 키 |", "|---|---|---|---|---|---|"]
    users = {}
    for key in keys_send + [k for k in keys_verify if k not in keys_send]:
        for rel, required, what in policy_table.get(key, []):
            users.setdefault(rel, []).append(key if key in keys_send else f"{key}(확인)")
    for rel in sorted(users, key=lambda r: (entries[r]["action"] != "send", r)):
        e = entries[rel]
        jet = "**없음 — 새로 간다**" if e["jetson"] == "-" else ("같음(이미 있음)" if e["jetson"] == e["md5"] else f"다름 `{e['jetson'][:8]}`")
        L.append(f"| `{rel}` | {e['size']} | `{e['md5']}` | {jet} | {e['action']} | {', '.join(users[rel])} |")
    L += ["", "키가 가리키는 폴더(브리지 레지스트리):", ""]
    for key in keys_send:
        if key.startswith("(추가 폴더)"):
            L.append(f"- `{key}` = 브리지 키와 무관하게 실은 정책 폴더(`--extra-policy-run`) — 그 키로 돌리려면 키 등록 코드가 함께 있어야 한다")
            continue
        if key in b.KEW3_CKPTS:
            sp = b.KEW3_CKPTS[key]
            L.append(f"- `{key}` = 걷기 `{sp.w3}` + 서기·회복 `{sp.ke}` (전환 {sp.handoff}, 정착 {sp.settle_quiet_s:g} s) — 자체 파일 없음, 하위 키의 파일을 쓴다")
        elif key in b.C5_PAIRS:
            L.append(f"- `{key}` = 서기 `{b.C5_PAIRS[key][0]}` + 회복 `{os.path.relpath(b.C5_PAIRS[key][1], b.REPO)}`")
        else:
            L.append(f"- `{key}` = `{os.path.relpath(b.CHECKPOINTS[key], b.REPO)}`")
    L += ["", "## 8. selftest 기준 원문", "",
          f"`jetson/scripts/selftest_w3_bridge_node.py` 의 비트 비교 기준 커밋 {', '.join('`' + r + '`' for r in revs)} 의 브리지 원문을 "
          f"`{REF_DIR}/` 에 둔다(젯슨에는 git 이 없어 `git show` 가 실패한다 — 실행기 `{CAND}/selftest_w3_node_refs.py` 가 파일에서 읽는다).", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="쓰지 않고 현재 산출물이 최신인지 확인")
    ap.add_argument("--inventory", default=INVENTORY)
    ap.add_argument("--policy-root", default=os.environ.get("HEAD1008_POLICY_ROOT", REPO), help="logs/rsl_rl/… 이 있는 폴더(기본: 저장소 루트)")
    ap.add_argument("--groups", default="session,tool,test,sync,policy,ref,record,parity", help="넣을 묶음(쉼표). 기본 전부")
    ap.add_argument("--keys", default=None, help="정책 파일을 실을 키(쉼표). 기본: " + ",".join(POLICY_KEYS))
    ap.add_argument("--extra-policy-run", default=None,
                    help="키와 무관하게 더 실을 정책 폴더(쉼표, 저장소 상대 — 예 logs/rsl_rl/biped12_stepb0/<run>). exported/policy.onnx·params/env.yaml 필수")
    args = ap.parse_args()
    args.policy_root = os.path.abspath(args.policy_root)
    entries, closures, policy_table, revs, inv, b, keys_send, keys_verify = build(args)
    md5_text = render_md5(entries)
    version_text = render_version(entries, md5_text)
    plan_text = render_plan(entries, version_text, md5_text, keys_send)
    diff_text = render_diff(entries, closures, policy_table, revs, inv, b, keys_send, keys_verify, version_text)
    outs = {OUT_MD5: md5_text, OUT_VERSION: version_text, OUT_PLAN: plan_text, OUT_DIFF: diff_text}
    stale = []
    for rel, text in outs.items():
        full = os.path.join(REPO, rel)
        cur = open(full, encoding="utf-8").read() if os.path.isfile(full) else None
        if cur != text:
            stale.append(rel)
            if not args.check:
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(text)
    n_send = sum(1 for e in entries.values() if e["action"] == "send")
    by_group = {}
    for e in entries.values():
        by_group.setdefault(e["group"], [0, 0])[0 if e["action"] == "send" else 1] += 1
    print(f"{NAME}: 파일 {len(entries)} (+ 매니페스트·VERSION) — 전송 {n_send} · 확인만 {len(entries) - n_send} · "
          + " · ".join(f"{g} {v[0]}/{v[1]}" for g, v in sorted(by_group.items())) + " (묶음 전송/확인)")
    print(f"  전송 크기 합 {sum(e['size'] for e in entries.values() if e['action'] == 'send') / 1e6:.2f} MB · 기준 원문 {len(revs)} 개({', '.join(revs)})")
    if args.check:
        if stale:
            print(f"[check] 최신이 아님: {', '.join(stale)} — python3 deploy/make_head_manifest.py 로 다시 만들 것")
            return 1
        print("[check] 산출물이 현재 트리와 일치")
        return 0
    print("  " + ("갱신: " + ", ".join(stale) if stale else "변경 없음(이미 최신)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
