#!/usr/bin/env python3
"""시뮬(학습) ↔ 배포(브리지·러너) 계약 감사 (2026-10-07 밤).

  python3 reports/night_1007/contract_audit.py            # 정책별 계약 표 출력, 불일치가 있으면 종료 코드 1
  python3 reports/night_1007/contract_audit.py --md 표.md  # 같은 표를 파일로도 저장
  python3 reports/night_1007/contract_audit.py --allow-known   # 문서에 적힌(꼬리표 '기지'·'신규') 불일치만 있으면 종료 코드 0 (회귀 관문용)
  python3 reports/night_1007/contract_audit.py --report reports/night_1007/CONTRACT_AUDIT.md --run-existing --run-mutants
                                                            # 보고서 전체를 다시 생성(기존 selftest 11 개 재실행 약 6 분 + 민감도 시험 약 6 분)

무엇을 하나 — 정책별로 **학습 쪽 계약**과 **배포 쪽 계약**을 항목별로 뽑아 나란히 놓고 일치/불일치/확인 불가를 판정한다.
  학습 쪽 : 각 run 의 params/env.yaml·agent.yaml(PyYAML), 체크포인트(model_N.pt) 가중치, 학습 로그(tfevents)의 커리큘럼 도달값,
            시뮬 소스(rl_walking/mdp_hang.py·mdp_stage4.py·w3.py·actions.py·onset.py)의 함수를 **소스 그대로**(AST 추출) torch CPU 로 실행
            — 기준은 각 run 의 **학습 시작 시점 커밋**(run 폴더 시각 직전의 브랜치 커밋)이고, 그 뒤 바뀐 계약 함수는 따로 줄로 보고한다,
            시뮬 롤아웃 기록(log_picture/w3parity_rec_*.npz — 있으면)의 관절 순서·이력 배열 순서, 시뮬 평가 스크립트의 인자(safe_tr.sh·imuh_q.sh).
  배포 쪽 : 브리지 모듈을 CPU 에서 import 해 `_make_runner(키)` 로 **실제 러너 객체**를 만들고 합성 입력을 넣어 조립된 관측 벡터·목표각을 읽는다
            (속성값을 읽는 데서 그치지 않고 입출력에서 β·α·주파수·스케일·슬루·기본자세를 역산한다). 노드(_tick)는 가짜 rclpy 로 main() 을 그대로 돌려
            LPF·블렌딩·유한차분 속도·발행 목표를 본다. HEAD 와 **젯슨 판**(커밋 e7710c63 + deploy/cand/c5krh_1007 의 브리지)을 따로 올려 같은 표로 비교한다.
            실물 블랙박스(reports/real_1003·real_1007 의 rl_bridge_*.jsonl — 젯슨이 실제로 낸 관측·행동·목표)를 두 판에 재생해 대조한다.

하지 않는 것 — 젯슨 접속·ROS·CAN·GPU·Isaac 실행 없음(CUDA_VISIBLE_DEVICES="" 강제, rclpy 는 가짜 모듈, 블랙박스는 임시 폴더). 기존 파일 수정 없음.
            Isaac Lab 소스 트리(/home/ryu/IsaacLab)는 읽기만 한다(관측 매니저 파일 한 개와 git describe·status).
**계약 검증(열린 고리)은 폐루프 성능 증명이 아니다**(CLAUDE.md 6). '젯슨 판' = 저장소가 기록한 배포본이며 젯슨의 현재 파일을 읽은 것이 아니다.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # GPU 금지 — torch import 전에
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import ast  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

sys.dont_write_bytecode = True                   # 감사 대상 트리에 __pycache__ 를 남기지 않는다

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
#: 감사할 배포 소스 트리(기본 = 저장소 HEAD 작업 트리). --deploy-tree 로 바꿀 수 있다(변이 시험용 — contract_audit_mutants.py).
DEPLOY_TREE = REPO

#: 감사 대상(브리지 키). 실험 키는 워커가 enable_experimental_w3()·enable_experimental_kew3() 로 등록해 만든다.
POLICY_KEYS = ["stand_crouch_g21a", "stand_c5kr_g21a_g24ra", "stand_c5krh_g21a_g24ra", "stand_w3s3d_15800",
               "stand_kew3_g21a_g24ra_w3s3d15800"]
#: 젯슨에 올라가 있는 판(지시서): 커밋 e7710c63 의 러너·c5_switch + deploy/cand/c5krh_1007 의 브리지(= e7710c63 브리지 + c5krh_1007.patch).
JET_COMMIT = "e7710c63"
JET_BRIDGE_REL = "deploy/cand/c5krh_1007/rl_bridge_node.py"
JET_PATCH_REL = "deploy/cand/c5krh_1007/c5krh_1007.patch"
JET_KEYS = ["stand_crouch_g21a", "stand_c5kr_g21a_g24ra", "stand_c5krh_g21a_g24ra"]
REAL_LOG_REL = "reports/real_1007/rl_bridge_20261007_200441.jsonl"       # 젯슨 krh 첫 실물 실행 블랙박스(읽기 전용 사본) — 현 젯슨 판이 낸 기록
#: 재생 대조에 쓰는 실물 블랙박스(키가 감사 대상인 것만 표에 들어간다). 1003 = G21a 306 s 세션(그때의 젯슨 판이 낸 기록 — 계약이 그대로면 재생이 맞아야 한다)
REAL_LOGS = [REAL_LOG_REL, "reports/real_1003/rl_bridge_20261003_174906.jsonl"]
#: 실물 로그에서 로봇이 지면에 서 있던 구간의 끝 [s](없으면 전 구간) — 그 뒤(줄 회수·공중)의 명령은 한계 초과 집계에서 따로 센다. (끝 시각, 근거)
REAL_LOG_GROUND = {"reports/real_1003/rl_bridge_20261003_174906.jsonl": (245.0, "reports/real_1003/REAL_1003.md '245 s 이후 줄 회수 중 양발 무접촉'")}
SIM_REC_REL = "log_picture/w3parity_rec_w3s3d_15800_d3.npz"             # 시뮬 롤아웃 기록(1007 parity 감사, GPU 1 회 산출물)
EVAL_SH_KE = "reports/gen_w3/scripts/imuh_q.sh"                         # KE(kr = imu 팔)·krh(imuH 팔) 시뮬 평가 인자
EVAL_SH_TR = "reports/gen_w3/scripts/safe_tr.sh"                        # 전환(합성) 시뮬 평가 인자
#: 평가 **산출물** 머리의 '인자:' 줄(실제로 돈 인자 — 있으면 스크립트보다 먼저 쓴다. log_picture 는 주 체크아웃에만 있을 수 있다)
EVAL_RES_KE = {"kr": "log_picture/ground_recovery_sab_imu_nd_s7.txt", "krh": "log_picture/ground_recovery_sab_imuH_nd_s7.txt"}
EVAL_RES_TR = "log_picture/ground_recovery_tr_kew3_nd_d2_s1.txt"        # X6: 배포 코드 합성 러너를 시뮬 폐루프에 넣은 평가
EVAL_RES_TR_DIRECT = "log_picture/ground_recovery_tr_direct_nd_d2_s1.txt"   # 평가기 direct 팔(상태 이어받기)

OK, BAD, UNK = "일치", "불일치", "확인 불가"
TOL_OBS = 2e-5           # 관측 재구성 허용(float32 연산 순서)
TOL_ACT = 2e-5           # 목표각 재구성 허용 [rad]


# =====================================================================================================================
# 공용 도구
# =====================================================================================================================
def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a, repo=REPO, binary=False):
    r = subprocess.run(["git", "-C", repo, *a], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr.decode(errors='replace').strip()[:300]}")
    return r.stdout if binary else r.stdout.decode()


def find_logs(repo=REPO):
    """정책·학습 산출물 루트(logs/). 워크트리에는 없을 수 있어 주 체크아웃(git 공용 디렉터리의 부모)도 본다."""
    cands = [os.path.join(repo, "logs")]
    try:
        common = git("rev-parse", "--git-common-dir").strip()
        common = common if os.path.isabs(common) else os.path.join(repo, common)
        cands.append(os.path.join(os.path.dirname(os.path.realpath(common)), "logs"))
    except Exception:  # noqa: BLE001
        pass
    for c in cands:
        if os.path.isdir(os.path.join(c, "rsl_rl")):
            return os.path.realpath(c)
    return None


def find_file(rel, repo=REPO):
    """저장소 상대 경로 → 실제 파일(워크트리에 없으면 주 체크아웃). 없으면 None."""
    p = os.path.join(repo, rel)
    if os.path.isfile(p):
        return p
    try:
        common = git("rev-parse", "--git-common-dir").strip()
        common = common if os.path.isabs(common) else os.path.join(repo, common)
        q = os.path.join(os.path.dirname(os.path.realpath(common)), rel)
        if os.path.isfile(q):
            return q
    except Exception:  # noqa: BLE001
        pass
    return None


def line_of(rel, pattern, repo=REPO):
    """rel 파일에서 정규식이 처음 맞는 줄 번호 → 'rel:N'. 못 찾으면 rel."""
    try:
        with open(os.path.join(repo, rel), encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                if re.search(pattern, ln):
                    return f"{rel}:{i}"
    except OSError:
        pass
    return rel


def yaml_line(path, keys):
    """env.yaml 에서 중첩 키(keys = ['observations', 'policy', ...])가 있는 줄 → '<run 이름>/params/env.yaml:N (키 경로)'. 못 찾으면 키 경로만."""
    name = "/".join(path.split(os.sep)[-3:])
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return ".".join(keys)
    i = 0
    for depth, k in enumerate(keys):
        want = " " * (2 * depth) + str(k) + ":"
        while i < len(lines) and not (lines[i].startswith(want) and (len(lines[i]) == len(want) or lines[i][len(want)] == " ")):
            i += 1
        if i >= len(lines):
            return f"{name} ({'.'.join(keys)})"
        i += 1
    return f"{name}:{i} ({'.'.join(keys)})"


def fnum(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, (bool, str)):
        return str(x)
    if isinstance(x, (list, tuple)):
        return "(" + ", ".join(fnum(v, nd) for v in x) + ")"
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


# ---------- 최소 protobuf 판독(ONNX initializer·tfevents 스칼라) — onnx·tensorboard 패키지 없이 ----------
def _varint(b, i):
    r = 0
    s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        if not (c & 0x80):
            return r, i
        s += 7


def pb_fields(b):
    i, n, out = 0, len(b), []
    while i < n:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            ln, i = _varint(b, i)
            v = b[i:i + ln]
            i += ln
        elif wt == 5:
            v = b[i:i + 4]
            i += 4
        elif wt == 1:
            v = b[i:i + 8]
            i += 8
        else:
            raise ValueError(f"protobuf wire type {wt}")
        out.append((fn, wt, v))
    return out


def onnx_graph(path):
    """policy.onnx → {"ops": [(op, attrs)], "inits": {이름: ndarray}, "inputs": [...], "outputs": [...]}."""
    import numpy as np
    with open(path, "rb") as f:
        b = f.read()
    graph = next(v for fn, wt, v in pb_fields(b) if fn == 7 and wt == 2)
    ops, inits, inputs, outputs = [], {}, [], []
    for fn, wt, v in pb_fields(graph):
        if fn == 1:
            op, attrs = "", {}
            for f2, w2, x in pb_fields(v):
                if f2 == 4:
                    op = x.decode()
                elif f2 == 5:
                    an = af = ai = None
                    for f3, w3, y in pb_fields(x):
                        if f3 == 1:
                            an = y.decode()
                        elif f3 == 2:
                            af = struct.unpack("<f", y)[0]
                        elif f3 == 3:
                            ai = y
                    attrs[an] = af if af is not None else ai
            ops.append((op, attrs))
        elif fn == 5:
            dims, dtype, name, raw, floats = [], None, "", None, []
            for f2, w2, x in pb_fields(v):
                if f2 == 1:
                    if w2 == 0:
                        dims.append(x)
                    else:
                        j = 0
                        while j < len(x):
                            d, j = _varint(x, j)
                            dims.append(d)
                elif f2 == 2:
                    dtype = x
                elif f2 == 8:
                    name = x.decode()
                elif f2 == 9:
                    raw = x
                elif f2 == 4:
                    floats.extend(struct.unpack(f"<{len(x) // 4}f", x) if w2 == 2 else struct.unpack("<f", x))
            if dtype == 1:
                arr = np.frombuffer(raw, dtype="<f4") if raw is not None else np.asarray(floats, dtype=np.float32)
                inits[name] = arr.reshape(dims) if dims else arr
        elif fn == 11:
            inputs.append(next(x.decode() for f2, _, x in pb_fields(v) if f2 == 1))
        elif fn == 12:
            outputs.append(next(x.decode() for f2, _, x in pb_fields(v) if f2 == 1))
    return {"ops": ops, "inits": inits, "inputs": inputs, "outputs": outputs}


def tfevents_scalars(run_dir, want=lambda tag: True):
    """run 의 events.out.tfevents.* → {tag: [(step, value)]}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))):
        with open(p, "rb") as f:
            data = f.read()
        i, n = 0, len(data)
        while i + 12 <= n:
            (ln,) = struct.unpack("<Q", data[i:i + 8])
            i += 12
            rec = data[i:i + ln]
            i += ln + 4
            step, summ = 0, None
            try:
                fl = pb_fields(rec)
            except Exception:  # noqa: BLE001 — 잘린 마지막 기록
                break
            for fn, wt, v in fl:
                if fn == 2 and wt == 0:
                    step = v
                elif fn == 5 and wt == 2:
                    summ = v
            if summ is None:
                continue
            for fn, wt, v in pb_fields(summ):
                if fn != 1:
                    continue
                tag = val = None
                for f2, w2, x in pb_fields(v):
                    if f2 == 1 and w2 == 2:
                        tag = x.decode(errors="replace")
                    elif f2 == 2 and w2 == 5:
                        val = struct.unpack("<f", x)[0]
                if tag is not None and val is not None and want(tag):
                    out.setdefault(tag, []).append((step, val))
    return out


# ---------- env.yaml (python 태그: tuple·slice 만 쓰인다 — 임의 객체 생성 없이 읽는다) ----------
def load_yaml(path):
    import yaml

    class L(yaml.SafeLoader):
        pass

    L.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, node: tuple(ld.construct_sequence(node, deep=True)))
    L.add_constructor("tag:yaml.org,2002:python/object/apply:builtins.slice",
                      lambda ld, node: ("slice",) + tuple(ld.construct_sequence(node, deep=True)))
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=L)


def short(func):
    return str(func or "").split(":")[-1].strip()


#: 관측 함수 → 프레임 차원(학습 코드의 반환 모양 — 합×이력이 체크포인트 actor 입력 차원과 맞는지로 검산한다)
OBS_DIM_BY_FUNC = {"base_ang_vel": 3, "base_ang_vel_lagged": 3, "projected_gravity": 3, "projected_gravity_lagged": 3,
                   "projected_gravity_biased": 3, "generated_commands": 3, "joint_pos_rel": 12, "encoder_joint_pos_rel": 12,
                   "encoder_joint_pos_rel_v2": 12, "joint_vel_rel": 12, "encoder_joint_vel": 12, "encoder_joint_vel_v2": 12,
                   "encoder_joint_vel_lpf": 12, "last_action": 12, "last_action_lpf": 12, "gait_clock": 2, "feet_contact_binary": 2,
                   "feet_contact_real": 2, "feet_contact_real_v2": 2, "gyro_ema": 3, "gyro_ema_lagged": 3, "joint_vel_ema": 12,
                   "encoder_joint_vel_ema": 12, "encoder_joint_vel_ema_v2": 12}


class Train:
    """한 run 의 학습 계약(env.yaml·agent.yaml·체크포인트·학습 로그)."""

    def __init__(self, run_dir):
        self.run = run_dir
        self.name = os.path.basename(run_dir.rstrip("/"))
        self.tag = self.name.split("_", 2)[-1] if self.name.count("_") >= 2 else self.name
        self.env_path = os.path.join(run_dir, "params", "env.yaml")
        self.ok = os.path.isfile(self.env_path)
        self.env = load_yaml(self.env_path) if self.ok else {}
        ap = os.path.join(run_dir, "params", "agent.yaml")
        self.agent = load_yaml(ap) if os.path.isfile(ap) else {}
        pol = (self.env.get("observations") or {}).get("policy") or {}
        self.obs_group = {k: v for k, v in pol.items() if not isinstance(v, dict)}
        self.terms = [(k, v) for k, v in pol.items() if isinstance(v, dict) and "func" in v]
        self.act = ((self.env.get("actions") or {}).get("joint_pos")) or {}
        self.ev = self.env.get("events") or {}
        self.cmd = ((self.env.get("commands") or {}).get("base_velocity")) or {}
        self.src_ckpt = None
        p = os.path.join(run_dir, "exported", "SOURCE_CHECKPOINT")
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                self.src_ckpt = f.read().strip()
        # 학습 시작 시각(폴더 이름) → 그 시각 직전의 브랜치 커밋. rsl_rl 은 커밋 해시를 남기지 않고 git status·diff 만 남긴다.
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", self.name)
        self.start = f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}" if m else None
        self.commit = None
        if self.start:
            try:
                self.commit = git("rev-list", "-1", f"--before={self.start}", "HEAD").strip() or None
            except RuntimeError:
                self.commit = None
        self.git_state = None
        gd = os.path.join(run_dir, "git")
        if os.path.isdir(gd):
            for fn in sorted(os.listdir(gd)):
                if not fn.endswith(".diff") or fn == "IsaacLab.diff":
                    continue
                try:
                    with open(os.path.join(gd, fn), encoding="utf-8", errors="replace") as f:
                        txt = f.read()
                except OSError:
                    continue
                st_, _, df = txt.partition("--- git diff ---")
                mb = re.search(r"^On branch (\S+)", st_, re.M)
                unt = []
                if "Untracked files:" in st_:
                    for ln in st_.split("Untracked files:", 1)[1].splitlines():
                        if ln.startswith("\t"):
                            unt.append(ln.strip())
                self.git_state = {"file": fn, "branch": mb.group(1) if mb else None, "dirty": re.findall(r"^diff --git a/(\S+) b/", df, re.M), "untracked": unt}
                break
        self.isaac_state = None                          # 학습 때의 Isaac Lab 판(rsl_rl 기록)
        ip = os.path.join(gd, "IsaacLab.diff")
        if os.path.isfile(ip):
            try:
                with open(ip, encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                st_, _, df = txt.partition("--- git diff ---")
                mh_ = re.search(r"^(HEAD detached at \S+|On branch \S+)", st_, re.M)
                self.isaac_state = {"head": mh_.group(1) if mh_ else None, "dirty": re.findall(r"^diff --git a/(\S+) b/", df, re.M)}
            except OSError:
                pass

    # ---- 관측 ----
    def layout(self):
        """[(항, 차원|None, 이력)]"""
        return [(n, OBS_DIM_BY_FUNC.get(short(t.get("func"))), int(t.get("history_length") or self.obs_group.get("history_length") or 1))
                for n, t in self.terms]

    def term(self, name):
        return dict(self.terms).get(name) or {}

    def obs_sig(self):
        return json.dumps(self.terms, sort_keys=True, default=str)

    def default_pose(self, order):
        jp = (((self.env.get("scene") or {}).get("robot") or {}).get("init_state") or {}).get("joint_pos") or {}
        return [jp.get(j) for j in order]

    def dt(self):
        return float((self.env.get("sim") or {}).get("dt", float("nan"))) * int(self.env.get("decimation", 0) or 0)

    def w3(self):
        g = (self.ev.get("w3_gait") or {}).get("params") if self.ev.get("w3_gait") else None
        c = (self.ev.get("w3_cmd") or {}).get("params") if self.ev.get("w3_cmd") else None
        return g, c

    def actuators(self):
        a = ((self.env.get("scene") or {}).get("robot") or {}).get("actuators") or {}
        return {k: (v.get("stiffness"), v.get("damping"), v.get("effort_limit_sim")) for k, v in a.items()}

    def usd(self):
        return ((((self.env.get("scene") or {}).get("robot") or {}).get("spawn") or {}).get("usd_path"))


# ---------- 시뮬 소스 사실(AST) ----------
_SRC_CACHE = {}


def _read_src(rel, commit=None):
    """저장소 파일 본문 — commit 이 있으면 그 커밋의 것(git show), 없으면 작업 트리."""
    k = (rel, commit)
    if k not in _SRC_CACHE:
        if commit:
            _SRC_CACHE[k] = git("show", f"{commit}:{rel}")
        else:
            with open(os.path.join(REPO, rel), encoding="utf-8") as f:
                _SRC_CACHE[k] = f.read()
    return _SRC_CACHE[k]


def _func_src(rel, names, commit=None):
    src = _read_src(rel, commit)
    tree = ast.parse(src)
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef,)) and node.name in names and node.name not in out:
            out[node.name] = (ast.get_source_segment(src, node), node)
    return out


def _defaults(node):
    """FunctionDef → {인자: 기본값(리터럴)}"""
    a = node.args
    pos = a.args
    out = {}
    for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        try:
            out[arg.arg] = ast.literal_eval(d)
        except Exception:  # noqa: BLE001
            out[arg.arg] = None
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            try:
                out[arg.arg] = ast.literal_eval(d)
            except Exception:  # noqa: BLE001
                out[arg.arg] = None
    return out


def sim_source_facts(commit=None):
    """시뮬 코드에서 계약에 걸리는 규칙을 소스 문자열로 확인(코드가 바뀌면 이 감사가 알아채게). commit = 그 커밋의 소스(없으면 작업 트리)."""
    F = {}
    try:
        mh = _func_src("rl_walking/mdp_hang.py", ["_ema_state", "last_action_lpf", "encoder_joint_vel_lpf", "encoder_joint_vel_v2",
                                                  "set_joint_limits_from_config", "feet_contact_real_v2", "base_ang_vel_lagged"], commit)
        F["ema_fresh_le1"] = "episode_length_buf <= 1" in mh["_ema_state"][0]
        F["la_fresh_eq0"] = "episode_length_buf == 0" in mh["last_action_lpf"][0]
        F["la_uses_term_beta"] = '"_beta"' in mh["last_action_lpf"][0] and "action_manager.action" in mh["last_action_lpf"][0]
        F["jv_fresh_eq0"] = "episode_length_buf == 0" in mh["encoder_joint_vel_lpf"][0]
        F["jv_fd_dt"] = "(full - prev) / dt" in mh["encoder_joint_vel_v2"][0]
        F["soft_clamp_joints_default"] = _defaults(mh["set_joint_limits_from_config"][1]).get("joints")
        F["contact_v2_flat"] = "tilt < st[\"thr\"]" in mh["feet_contact_real_v2"][0]
    except Exception as e:  # noqa: BLE001
        F["err_mdp_hang"] = repr(e)
    try:
        w3 = _func_src("rl_walking/w3.py", ["w3_gait_update", "install_w3_cmd", "w3_heading_loop"], commit)
        F["w3_gait_defaults"] = _defaults(w3["w3_gait_update"][1])
        F["w3_cmd_defaults"] = _defaults(w3["install_w3_cmd"][1])
        F["w3_hl_defaults"] = _defaults(w3["w3_heading_loop"][1])
        m = re.search(r"cw\[:, 1\]\.abs\(\) < ([0-9.]+)", w3["w3_gait_update"][0])
        F["w3_lead_lat_eps"] = float(m.group(1)) if m else None
        F["w3_fresh_le1"] = "episode_length_buf <= 1" in w3["w3_gait_update"][0]
    except Exception as e:  # noqa: BLE001
        F["err_w3"] = repr(e)
    try:
        src = _read_src("rl_walking/actions.py", commit)
        body = src[src.index("def process_actions"):src.index("@configclass")]
        marks = [("지연", "torch.gather(self._buf"), ("슬루", "delta, -self._max_delta"), ("LPF", "self._lpf_state = b *"),
                 ("온셋 블렌딩", "onset_alpha("), ("백래시", "self._bl_out = torch.where"), ("소프트 클램프", "clamped = torch.maximum")]
        pos = [(body.find(pat), nm) for nm, pat in marks]
        F["action_order"] = [nm for p, nm in sorted(pos) if p >= 0]
        F["action_order_complete"] = all(p >= 0 for p, _ in pos)
        F["slew_init_from_joint_pos"] = "self._prev_target[ids] = cur" in body and "self._asset.data.joint_pos[ids]" in body
        F["lpf_init_from_joint_pos"] = "self._lpf_state[ids] = cur" in body
        F["onset_hold_tgt_default"] = "hold_tgt = self._asset.data.default_joint_pos" in body
    except Exception as e:  # noqa: BLE001
        F["err_actions"] = repr(e)
    # Isaac Lab 관측 매니저(저장소 밖 — 있으면 읽기만): 잡음 → 클립 → 스케일 → 이력 버퍼 순서
    F["isaac_obs_order"] = None
    for root in (os.environ.get("ISAACLAB_PATH"), "/home/ryu/IsaacLab"):
        p = os.path.join(root or "", "source/isaaclab/isaaclab/managers/observation_manager.py")
        if root and os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    src = f.read()
                i_n, i_c, i_s, i_h = (src.find(x) for x in ("term_cfg.noise.func(obs", "obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])",
                                                             "obs.mul_(term_cfg.scale)", "circular_buffer.append(obs)"))
                F["isaac_obs_order"] = bool(0 <= i_n < i_c < i_s < i_h)
                F["isaac_obs_line"] = f"{p}:{src[:i_c].count(chr(10)) + 1}" if i_c >= 0 else p
                try:                                         # 읽기 전용: 지금 깔린 판과 그 파일의 변경 여부
                    F["isaac_now"] = git("describe", "--tags", "--always", repo=root).strip()
                    F["isaac_obs_clean"] = git("status", "--porcelain", "--untracked-files=no", "--", "source/isaaclab/isaaclab/managers/observation_manager.py", repo=root).strip() == ""
                except RuntimeError:
                    F["isaac_now"] = None
            except OSError:
                pass
            break
    return F


#: 계약에 관여하는 이벤트 함수(관측 항 함수·행동 항 클래스는 env.yaml 에서 자동으로 모은다)
CONTRACT_EVENT_FUNCS = ("set_joint_limits_from_config", "w3_gait_update", "install_w3_cmd", "w3_heading_loop")
#: 학습 뒤 바뀐 계약 함수 가운데 차이를 읽어 확인한 것 — (파일, 이름, 학습 시점 md5, 현재 md5) → (요약, 종전 동작이 유지되는 조건).
#: 함수가 다시 바뀌면 md5 가 달라져 '미검토'로 떨어진다.
REVIEWED_SRC_CHANGES = {
    ("rl_walking/actions.py", "SlewJointPositionAction", "2a16dd6b", "f8a28618"):
        ("지연 표본에 가중 옵션(delay_probs) 추가 — 비어 있으면 종전 균등 표본", "delay"),
    ("rl_walking/mdp_stage4.py", "_gait_phase", "980caf46", "67c53277"):
        ("위상 누적기(env._gu)가 있을 때의 분기와 진단용 환경변수(FREEZE_GAIT_CLOCK) 추가 — 없으면 종전 시간식 2π·f·t", "phase"),
    ("rl_walking/mdp_stage4.py", "gait_clock", "b77b3875", "bdbd5f2a"):
        ("관측 게이트(env._ge_obs)가 있을 때의 곱과 진단용 환경변수(W3_CLOCK_DIAG) 추가 — 없으면 종전 [sin φ, cos φ]", "phase"),
}


def _top_defs(src):
    """{이름: (소스 조각, 노드)} — 최상위 함수·클래스·대입."""
    out = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    lines = src.split("\n")

    def seg(n):                                  # ast.get_source_segment 와 같은 본문(열 오프셋은 UTF-8 바이트) — 줄 목록은 한 번만 만든다
        ls = lines[n.lineno - 1:n.end_lineno]
        if not ls:
            return ""
        ls[-1] = ls[-1].encode("utf-8")[:n.end_col_offset].decode("utf-8", "ignore")
        ls[0] = ls[0].encode("utf-8")[n.col_offset:].decode("utf-8", "ignore")
        return "\n".join(ls)

    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = (seg(n), n)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = (seg(n), n)
    return out


def contract_source_check(tr, facts_head):
    """학습 시작 시점 커밋의 시뮬 소스와 현재 소스 비교.
    대상 = env.yaml 의 관측(policy) 항 함수·행동 항 클래스·한계/위상 이벤트 함수 + 온셋 식, 그리고 그것들이 이름으로 가리키는 같은 파일들의 최상위 정의(재귀)."""
    import difflib
    out = {"commit": tr.commit, "start": tr.start, "git_state": tr.git_state}
    if not tr.commit:
        out["error"] = "학습 시점 커밋을 찾지 못함"
        return out
    out["commit_date"] = git("log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M:%S", tr.commit).strip()
    refs = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, str) and re.fullmatch(r"rl_walking(\.\w+)+:\w+", o):
            refs.add(o)

    walk((tr.env.get("observations") or {}).get("policy"))
    walk(tr.env.get("actions"))
    for v in tr.ev.values():
        if isinstance(v, dict) and short(v.get("func")) in CONTRACT_EVENT_FUNCS:
            refs.add(str(v.get("func")))
    refs.add("rl_walking.onset:onset_alpha")
    todo = [(r.split(":")[0].replace(".", "/") + ".py", r.split(":")[1]) for r in sorted(refs)]
    files = sorted({f for f, _ in todo})
    hd, od = {}, {}
    for f in files:
        try:
            hd[f] = _top_defs(_read_src(f))
        except OSError:
            hd[f] = {}
        try:
            od[f] = _top_defs(_read_src(f, tr.commit))
        except RuntimeError:
            od[f] = None                                  # 그 커밋에 파일이 없다
    owners = {}
    for f, d in hd.items():
        for n in d:
            owners.setdefault(n, []).append(f)
    seen, absent = set(), []
    while todo:
        f, n = todo.pop()
        if (f, n) in seen:
            continue
        seen.add((f, n))
        if n not in hd.get(f, {}):
            absent.append(f"{f}:{n}")
            continue
        for x in ast.walk(hd[f][n][1]):
            nm = x.id if isinstance(x, ast.Name) else x.attr if isinstance(x, ast.Attribute) else None
            for f2 in owners.get(nm, []) if nm else []:
                if (f2, nm) not in seen:
                    todo.append((f2, nm))
    same, changed, missing = 0, [], []
    md5 = lambda t: hashlib.md5(t.encode()).hexdigest()[:8]  # noqa: E731
    for f, n in sorted(seen):
        if n not in hd.get(f, {}):
            continue
        new = hd[f][n][0]
        if od.get(f) is None or n not in od[f]:
            missing.append(f"{f}:{n}")
            continue
        old = od[f][n][0]
        if old == new:
            same += 1
            continue
        dl = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0))
        add = sum(1 for x in dl if x.startswith("+") and not x.startswith("+++"))
        rem = sum(1 for x in dl if x.startswith("-") and not x.startswith("---"))
        rv = REVIEWED_SRC_CHANGES.get((f, n, md5(old), md5(new)))
        changed.append({"file": f, "name": n, "add": add, "rem": rem, "old": md5(old), "new": md5(new), "review": rv[0] if rv else None, "need": rv[1] if rv else None})
    ev_funcs = {short(v.get("func")) for v in tr.ev.values() if isinstance(v, dict)}
    conds = {"delay": not tr.act.get("delay_probs"), "phase": not (ev_funcs & {"gait_phase_update", "w3_gait_update"})}
    out.update(n=same + len(changed) + len(missing), same=same, changed=changed, missing=missing, absent_head=absent, conds=conds,
               cond_fail=sorted({c["need"] for c in changed if c["need"] and not conds.get(c["need"], False)}))
    # 감사가 읽는 규칙(sim_source_facts)이 두 판에서 같은가
    try:
        f_old = sim_source_facts(tr.commit)
        keys = [k for k in facts_head if not k.startswith(("isaac_", "err_"))]
        if tr.w3()[0] is None:
            keys = [k for k in keys if not k.startswith("w3_")]
        out["facts_diff"] = [k for k in keys if f_old.get(k) != facts_head.get(k)] + [k for k in f_old if k.startswith("err_") and not (k == "err_w3" and tr.w3()[0] is None)]
        out["facts_n"] = len(keys)
    except Exception as e:  # noqa: BLE001
        out["facts_diff"] = [repr(e)]
    gs = tr.git_state or {}
    out["dirty_rl"] = [f for f in gs.get("dirty", []) if f.startswith("rl_walking/") and f.endswith(".py")]
    out["untracked_rl"] = [f for f in gs.get("untracked", []) if f.startswith("rl_walking/") and (f.endswith(".py") or f.endswith("/"))]
    return out


def _extract_funcs(rel, names, commit=None):
    tree = ast.parse(_read_src(rel, commit))
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in body}
    if missing:
        raise RuntimeError(f"{rel}: 함수 없음 {sorted(missing)}")
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def sim_exec_filters(tr, betas, n_ticks=600, commit=None):
    """시뮬 관측 필터·클록 함수를 **소스 그대로** torch CPU 로 실행하고 HEAD 러너(스텁 정책)와 같은 신호열로 대조.
    tr = 학습 계약(env.yaml 의 β·α·주파수), betas = 배포 러너가 실제로 쓰는 (act β, jvel β). 반환 {항: (초기 구간 최대 |Δ|, 정상 구간 최대 |Δ|)}."""
    import numpy as np
    import torch
    torch.set_num_threads(1)
    if DEPLOY_TREE not in sys.path:
        sys.path.insert(0, DEPLOY_TREE)
    from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4
    ns = {"torch": torch, "math": math, "os": os}
    exec(_extract_funcs("rl_walking/mdp_hang.py", ["_ema_state", "last_action_lpf", "encoder_joint_vel_lpf", "gyro_ema_lagged",
                                                   "encoder_joint_vel_ema_v2"], commit), ns)
    ns["encoder_joint_vel_v2"] = lambda env, asset_cfg=None, action_term="joint_pos", play_rad=None: env._x_jv.clone()
    ns["base_ang_vel_lagged"] = lambda env, max_lag=3, asset_cfg=None: env._x_gy.clone()
    ns2 = {"torch": torch, "math": math, "os": os, "GAIT_FREQ_HZ": 1.4}
    exec(_extract_funcs("rl_walking/mdp_stage4.py", ["_gait_phase", "gait_clock"], commit), ns2)
    p_jv = tr.term("joint_vel").get("params") or {}
    p_ge = tr.term("gyro_ema").get("params") or {}
    p_je = tr.term("joint_vel_ema").get("params") or {}
    p_ck = tr.term("gait_clock").get("params") or {}
    act_beta, jv_beta = betas
    rng = np.random.default_rng(7)
    gy = (rng.standard_normal((n_ticks, 3)) * 0.3).astype(np.float32)
    jv = (rng.standard_normal((n_ticks, 12)) * 0.8).astype(np.float32)
    raws = (rng.standard_normal((n_ticks, 12)) * 0.7).astype(np.float32)
    reset_at = 300                                   # 중간 리셋(에피소드 재시작 = 러너 reset)
    res = {}
    for fresh_ticks in (1, 2):
        it = {"k": 0}
        r = PolicyRunnerStage4(model_path=lambda obs: raws[it["k"]], obs_dim=320, gait_freq=float(p_ck.get("freq", 1.4)),
                               act_obs_ema_beta=act_beta, jvel_obs_ema_beta=jv_beta,
                               **({"ema_fresh_ticks": 2} if fresh_ticks == 2 else {}))
        r.reset()
        env = types.SimpleNamespace(num_envs=1, device="cpu", step_dt=float(tr.dt()), common_step_counter=0,
                                    episode_length_buf=torch.zeros(1, dtype=torch.long))
        term = types.SimpleNamespace(_beta=torch.tensor([float(act_beta if act_beta is not None else 1.0)]))
        env.action_manager = types.SimpleNamespace(action=torch.zeros(1, 12), get_term=lambda n: term)
        offs, o = {}, 0
        for name, dim in r.terms:
            L = r._hist[name]._buf.shape[0]
            offs[name] = (o + dim * (L - 1), o + dim * L)        # 최신 프레임
            o += dim * L
        err = {k: [0.0, 0.0] for k in ("joint_vel", "actions", "gait_clock", "gyro_ema", "joint_vel_ema")}
        prev_raw = np.zeros(12, np.float32)
        ep = 0
        for k in range(n_ticks):
            if k == reset_at:
                r.reset()
                env.episode_length_buf[:] = 0
                prev_raw = np.zeros(12, np.float32)
                ep = k
            x_jv = jv[k].copy()
            if k == ep:
                x_jv[:] = 0.0                                    # 에피소드 첫 틱 속도 0(시뮬 encoder_joint_vel_v2 fresh, 브리지 첫 틱 qvel 0)
            env._x_jv = torch.as_tensor(x_jv).unsqueeze(0)
            env._x_gy = torch.as_tensor(gy[k]).unsqueeze(0)
            env.action_manager.action = torch.as_tensor(prev_raw).unsqueeze(0)
            sim = {
                "joint_vel": ns["encoder_joint_vel_lpf"](env, beta=float(p_jv.get("beta", 0.25)))[0].numpy(),
                "actions": ns["last_action_lpf"](env, "joint_pos")[0].numpy(),
                "gait_clock": ns2["gait_clock"](env, float(p_ck.get("freq", 1.4)))[0].numpy(),
                "gyro_ema": ns["gyro_ema_lagged"](env, alpha=float(p_ge.get("alpha", 0.25)), noise=0.0, max_lag=0)[0].numpy(),
                "joint_vel_ema": ns["encoder_joint_vel_ema_v2"](env, alpha=float(p_je.get("alpha", 0.25)), noise=0.0)[0].numpy(),
            }
            it["k"] = k
            r.step(gy[k], np.float32([0.0, 0.0, -1.0]), np.zeros(3, np.float32), r.default_pose.copy(), x_jv, np.float32([1.0, 1.0]))
            for name, v in sim.items():
                a, b = offs[name]
                d = float(np.abs(r.last_obs[a:b] - v).max())
                slot = 0 if (k - ep) < 220 else 1          # 0.9^220 ≈ 1e-10 — 시작 과도가 사라진 뒤를 '정상 구간'으로
                err[name][slot] = max(err[name][slot], d)
            prev_raw = raws[k].copy()
            env.episode_length_buf += 1
            env.common_step_counter += 1
        res[fresh_ticks] = err
    return res


def sim_exec_onset(blend_s, commit=None):
    """학습 onset_alpha(소스 실행) vs 배포 blend_alpha — 같은 hold 0·blend_s 에서 최대 |Δ|."""
    import torch
    if DEPLOY_TREE not in sys.path:
        sys.path.insert(0, DEPLOY_TREE)
    from rl_walking.deploy.deploy_chain import blend_alpha
    ns = {"torch": torch}
    exec(_extract_funcs("rl_walking/onset.py", ["onset_alpha"], commit), ns)
    worst = 0.0
    for i in range(0, 121):
        t = i * 0.02
        a = float(ns["onset_alpha"](torch.tensor([t]), torch.tensor([0.0]), torch.tensor([float(blend_s)]))[0])
        worst = max(worst, abs(a - blend_alpha(t, float(blend_s))))
    return worst


def parse_eval_args(rel, var):
    """평가 셸 스크립트의 VAR="..." 문자열 → {--flag: 값 또는 True}. 연관 배열 [키]="..." 는 var='ARM:키'."""
    p = find_file(rel)
    if p is None:
        return None
    with open(p, encoding="utf-8") as f:
        src = f.read()
    if ":" in var:
        arr, key = var.split(":")
        m = re.search(r"\[" + re.escape(key) + r"\]=\"([^\"]*)\"", src[src.index(f"declare -A {arr}"):])
    else:
        m = re.search(r"^" + re.escape(var) + r"=\"([^\"]*)\"", src, re.M)
    if not m:
        return None
    return _toks_to_dict(m.group(1).split())


def _toks_to_dict(toks):
    out, i = {}, 0
    while i < len(toks):
        if toks[i].startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[toks[i]] = toks[i + 1]
                i += 2
            else:
                out[toks[i]] = True
                i += 1
        else:
            i += 1
    return out


def parse_result_header(rel):
    """평가 산출물(ground_recovery_<tag>.txt) 머리 → (실제 인자 dict, 정착 판정 (|ω| 문턱, 유지 s, 상한 s) 또는 None). 파일이 없으면 (None, None)."""
    p = find_file(rel)
    if p is None:
        return None, None
    args, settle = None, None
    with open(p, encoding="utf-8", errors="replace") as f:
        for i, ln in enumerate(f):
            if i > 14:
                break
            if ln.startswith("인자:"):
                args = _toks_to_dict(ln[len("인자:"):].split())
            m = re.search(r"정착 quiet\(\|ω\|<([0-9.]+) ([0-9.]+)s 상한 ([0-9.]+)s\)", ln)
            if m:
                settle = tuple(float(x) for x in m.groups())
    return args, settle


def argparse_defaults(rel, flags):
    """평가기 소스의 parser.add_argument("--x", ..., default=…) 기본값(리터럴)."""
    p = os.path.join(REPO, rel)
    out = {}
    try:
        with open(p, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except OSError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args:
            try:
                name = ast.literal_eval(node.args[0])
            except Exception:  # noqa: BLE001
                continue
            if name in flags:
                for kw in node.keywords:
                    if kw.arg == "default":
                        try:
                            out[name] = ast.literal_eval(kw.value)
                        except Exception:  # noqa: BLE001
                            out[name] = None
    return out


# =====================================================================================================================
# 워커 — 한 소스 트리(HEAD 또는 젯슨 판) 안에서 브리지를 올려 실제 객체로 자료를 낸다 (별도 프로세스)
# =====================================================================================================================
def _w_load_bridge(tree):
    import importlib.util
    path = os.path.join(tree, "jetson", "scripts", "rl_bridge_node.py")
    spec = importlib.util.spec_from_file_location("ca_bridge_node", path)
    B = importlib.util.module_from_spec(spec)
    sys.modules["ca_bridge_node"] = B
    spec.loader.exec_module(B)
    import rl_walking
    here = os.path.realpath(rl_walking.__file__)
    assert here.startswith(os.path.realpath(tree) + os.sep), f"rl_walking 이 다른 트리에서 로드됨: {here}"
    return B


def _w_relocate(B, logs):
    src = os.path.join(B.REPO, "logs") + os.sep
    dst = logs.rstrip(os.sep) + os.sep
    for k, v in list(B.CHECKPOINTS.items()):
        if isinstance(v, str) and v.startswith(src) and not os.path.isdir(v):
            B.CHECKPOINTS[k] = dst + v[len(src):]
    for k, (s, p) in list(getattr(B, "C5_PAIRS", {}).items()):
        if p.startswith(src) and not os.path.isdir(p):
            B.C5_PAIRS[k] = (s, dst + p[len(src):])


def _w_rel(B, logs, p):
    for pre in (os.path.join(B.REPO, "logs") + os.sep, logs.rstrip(os.sep) + os.sep):
        if isinstance(p, str) and p.startswith(pre):
            return p[len(pre):]
    return p


def _w_core(R):
    return R.__dict__.get("core", R)


def _w_inputs(n, default, seed=1007, with_pulses=True):
    """결정적 합성 입력열(두 트리에 같은 값). 기울기 펄스 2 회(회복 감지기 켜짐·cue·조기 복귀·불응기 경로를 지난다)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    f32 = np.float32
    gyro = (rng.standard_normal((n, 3)) * 0.04).astype(f32)
    pitch = np.radians(0.6 * np.sin(np.arange(n) * 0.013) + 0.2 * rng.standard_normal(n))
    roll = np.radians(-1.5 + 0.4 * np.sin(np.arange(n) * 0.021) + 0.2 * rng.standard_normal(n))
    contact = np.ones((n, 2), f32)
    if with_pulses:
        for t0, dur, dp, dr, wy, wx in ((210, 14, 7.5, 0.0, 0.9, 0.0), (430, 12, 0.0, -7.0, 0.0, -0.9), (470, 10, 6.5, 0.0, 0.8, 0.0)):
            pitch[t0:t0 + dur] += np.radians(dp)
            roll[t0:t0 + dur] += np.radians(dr)
            gyro[t0:t0 + 3, 1] += f32(wy)
            gyro[t0:t0 + 3, 0] += f32(wx)
        contact[224:231, 0] = 0.0                     # 한발 구간(조기 복귀의 양발 조건)
        contact[330:333, 1] = 0.0
    tx, ty = np.tan(pitch), np.tan(roll)
    c = 1.0 / np.sqrt(1.0 + tx * tx + ty * ty)
    grav = np.stack([c * tx, c * ty, -c], 1).astype(f32)          # pitch = atan2(g_x, −g_z), roll = atan2(g_y, −g_z)
    qpos = (np.asarray(default, f32)[None] + (rng.standard_normal((n, 12)) * 0.04).astype(f32)).astype(f32)
    qvel = (rng.standard_normal((n, 12)) * 0.6).astype(f32)
    qvel[0] = 0.0
    return dict(gyro=gyro, grav=grav, qpos=qpos, qvel=qvel, contact=contact, cmd=np.zeros((n, 3), f32))


def _w_runner_attrs(B, core):
    PR4 = sys.modules["rl_walking.deploy.policy_runner_stage4"]
    PR = sys.modules["rl_walking.deploy.policy_runner"]
    return {"obs_dim": int(core.obs_dim), "terms": [[n, int(d), int(core._hist[n]._buf.shape[0])] for n, d in core.terms],
            "default_pose": [float(v) for v in core.default_pose], "act_beta": core.act_obs_ema_beta, "jv_beta": core.jvel_obs_ema_beta,
            "jvel_ema_mode": core.jvel_ema_mode, "gait_freq": float(core.gait_freq), "backend": core.backend,
            "ema_fresh_ticks": int(getattr(core, "ema_fresh_ticks", 1)), "has_ema_fresh_opt": hasattr(core, "ema_fresh_ticks"),
            "w3": core.__dict__.get("w3") is not None, "hturn2": core.__dict__.get("hturn2") is not None,
            "ACTION_SCALE": float(PR.ACTION_SCALE), "SLEW_RAD": float(PR.SLEW_RAD), "RATE_HZ": float(PR.RATE_HZ),
            "POLICY_DT": float(PR4.POLICY_DT), "HISTORY_LEN": int(PR4.HISTORY_LEN), "EMA_ALPHA_S8A": float(PR4.EMA_ALPHA_S8A),
            "limits_loaded": core._limits_lo is not None}


def _w_det_attrs(det):
    g = lambda n, d=None: getattr(det, n, d)  # noqa: E731
    return {"on_tilt_deg": g("on_tilt"), "on_w": g("on_w"), "off_tilt_deg": g("off_tilt"), "off_w": g("off_w"), "calm_s": g("calm_s"),
            "dwell_s": g("dwell_s"), "dt": g("dt"), "ref_tau_s": g("ref_tau_s"), "ref_cap_deg": g("ref_cap"), "ref_init_s": g("ref_init_s"),
            "off_contact": g("off_contact"), "early": (None if g("early") is None else list(g("early"))),
            "refrac": (None if g("refrac") is None else list(g("refrac"))), "in_filt": g("in_filt"),
            "early_gate": (None if g("early_gate") is None else list(g("early_gate"))), "settle": g("settle") is not None,
            "imu_f32": bool(g("imu_f32", False)), "requiet": g("requiet"), "has_imu_f32_opt": hasattr(det, "imu_f32")}


def _w_probe_seq(B, key, R, A, tag):
    """합성 입력열을 러너에 넣어 관측·원시 행동·목표각을 기록(검산은 부모 프로세스)."""
    import numpy as np
    core = _w_core(R)
    is_c5 = "core" in R.__dict__
    w3 = core.__dict__.get("_w3_clock") is not None
    n = 600
    S = _w_inputs(n, core.default_pose)
    want = np.zeros(n, bool)
    if w3:      # 걷기 일정: 서기 → 앞 → 옆(−, 위상 유지) → 정지 → 옆(−) 시작(오른발 리드) → 정지 → 회전(−) 시작(오른발 리드) → 정지 → 제자리
        S = _w_inputs(n, core.default_pose, with_pulses=False)
        sched = [(50, 200, (0.06, 0.0, 0.0)), (200, 300, (0.0, -0.035, 0.0)), (360, 470, (0.0, -0.035, 0.0)), (510, 550, (0.0, 0.0, -0.15)),
                 (570, 600, (0.0, 0.0, 0.0))]
        for a, b, c in sched:
            want[a:b] = True
            S["cmd"][a:b] = np.float32(c)
    obs = np.zeros((n, core.obs_dim), np.float32)
    raw = np.zeros((n, 12), np.float32)
    out = np.zeros((n, 12), np.float32)
    mode = np.zeros(n, np.int8)
    clk = np.zeros((n, 3), np.float32)
    lead = []
    R.reset()
    for k in range(n):
        if w3:
            R.set_gait_enable(bool(want[k]))
        o = R.step(S["gyro"][k], S["grav"][k], S["cmd"][k], S["qpos"][k], S["qvel"][k], S["contact"][k])
        obs[k] = core.last_obs
        raw[k] = core.last_action
        out[k] = o
        if is_c5:
            mode[k] = int(R.last_mode)
        if w3:
            c = core._w3_clock
            clk[k] = (float(c.u), float(c.ge), float(c.ge_obs))
            lead.append(getattr(c, "last_lead", None))
    for nm, v in S.items():
        A[f"{tag}/in_{nm}"] = v
    A[f"{tag}/obs"], A[f"{tag}/raw"], A[f"{tag}/out"], A[f"{tag}/mode"], A[f"{tag}/want"], A[f"{tag}/clk"] = obs, raw, out, mode, want, clk
    info = {"n": n, "is_c5": is_c5, "w3": w3}
    if is_c5:
        d = R.det
        info.update(n_on=int(d.n_on), n_early=int(getattr(d, "n_early", 0)), n_cue=int(getattr(R, "n_cue", 0)),
                    mode_ticks=int(mode.sum()))
    if w3:
        info["lead"] = [x for i, x in enumerate(lead) if i in (51, 361, 511, 571)]
    # 클립 프로브: 범위 밖 큰 값 한 틱
    R.reset()
    big_g = np.float32([100.0, -100.0, 50.0])
    R.step(big_g, np.float32([0, 0, -1]), np.zeros(3, np.float32), core.default_pose + np.float32(50.0), np.full(12, 200.0, np.float32),
           np.float32([1, 1]))
    A[f"{tag}/clip_obs"] = np.array(core.last_obs, np.float32)
    # 슬루 기준 프로브: 첫 틱 qpos 를 기본자세에서 1 rad 띄움 → 첫 출력은 qpos 에서 슬루 한 번
    R.reset()
    q0 = (core.default_pose + np.float32(1.0)).astype(np.float32)
    o0 = R.step(np.zeros(3, np.float32), np.float32([0, 0, -1]), np.zeros(3, np.float32), q0, np.zeros(12, np.float32), np.float32([1, 1]))
    A[f"{tag}/slew0"] = (np.asarray(o0, np.float32) - q0)
    R.reset()
    return info


def _w_probe_kew3(B, key, R, A, tag):
    """합성 러너 전환 규칙을 실제 객체로 계측(스텁 아님 — 실모델). 입력은 조용한 양발 서기 + 필요한 구간만 자이로를 키운다."""
    import numpy as np
    ck, cw = _w_core(R.ke), _w_core(R.w3r)
    pose = np.asarray(ck.default_pose, np.float32)
    f32 = np.float32
    ev = {"handoff": R.handoff, "settle_quiet_s": float(R.settle_quiet_s), "settle_w": float(R.settle_w), "settle_max_s": float(R.settle_max_s),
          "dt": float(R.dt)}
    rng = np.random.default_rng(3)

    def step(cmd=(0.0, 0.0, 0.0), w=0.0, contact=(1.0, 1.0)):
        g = f32([0.01 * rng.standard_normal(), w, 0.0])
        q = (pose + (rng.standard_normal(12) * 0.01).astype(f32)).astype(f32)
        return R.step(g, f32([0.0, 0.0, -1.0]), f32(cmd), q, (rng.standard_normal(12) * 0.05).astype(f32), f32(contact))

    def cmd_frame(core):      # 관측 velocity_commands 최신 프레임
        o = 0
        for name, dim in core.terms:
            L = core._hist[name]._buf.shape[0]
            if name == "velocity_commands":
                return [float(v) for v in core.last_obs[o + dim * (L - 1): o + dim * L]]
            o += dim * L

    def clk_block(core):
        o = 0
        for name, dim in core.terms:
            L = core._hist[name]._buf.shape[0]
            if name == "gait_clock":
                return np.array(core.last_obs[o:o + dim * L], np.float32).reshape(L, dim)
            o += dim * L

    R.reset()
    for _ in range(60):
        last = step(cmd=(0.06, 0.0, 0.0))                     # KE 모드: 외부 명령이 들어와도 KE 관측 명령은 0 이어야
    ev["ke_mode_cmd_obs"] = cmd_frame(ck)
    ev["ke_mode_start"] = R.mode
    ke_ema = {k: v.copy() for k, v in ck._ema.items()}
    ke_tick = int(ck._tick)
    R.set_gait_enable(True)
    o = step(cmd=(0.06, 0.0, 0.0))
    ev["to_w3_same_tick"] = (R.mode == "w3" and R.last_event == "to_w3")
    ev["w3_first_tick_ge"] = bool(R._w3_clock.ge)
    ev["w3_first_tick_clock_hist_zero"] = bool(float(np.abs(clk_block(cw)).max()) == 0.0)
    ev["to_w3_jump_deg"] = float(np.degrees(np.abs(np.asarray(o) - np.asarray(last)).max()))
    ev["w3_tick_after_carry"] = int(cw._tick)
    ev["ke_tick_before"] = ke_tick
    ev["carry_ema_keys"] = sorted(ke_ema)
    step(cmd=(0.06, 0.0, 0.0))
    ev["w3_second_tick_ge"] = bool(R._w3_clock.ge)
    ev["w3_second_tick_ge_obs"] = float(R._w3_clock.ge_obs)
    for _ in range(120):
        last = step(cmd=(0.06, 0.0, 0.0))
    ev["w3_cmd_obs_walking"] = cmd_frame(cw)
    R.set_gait_enable(False)
    n_latch = 0
    while bool(R._w3_clock.ge) and n_latch < 60:
        last = step()
        n_latch += 1
    ev["latch_ticks"] = n_latch
    n_noisy = 0
    for _ in range(30):                                       # 시끄러운 동안(|ω_xy| 0.5 > 0.15)은 넘기지 않는다
        last = step(w=0.5)
        n_noisy += int(R.mode == "w3")
    ev["noisy_stay_w3"] = (n_noisy == 30)
    n_q = 0
    while R.mode == "w3" and n_q < 80:
        last = step()
        n_q += 1
    ev["quiet_ticks_to_ke"] = n_q
    ev["ke_tick_after_carry"] = int(ck._tick)
    ev["ke_clock_hist_after_carry"] = [[float(v) for v in row] for row in ck._hist["gait_clock"]._buf]
    ev["ke_det_n_on_after"] = int(R.ke.det.n_on)
    o = step()
    ev["to_ke_jump_deg"] = float(np.degrees(np.abs(np.asarray(o) - np.asarray(last)).max()))
    # 상한: 래치 뒤 계속 시끄러우면 settle_max_s 에 인계
    R.reset()
    step()
    R.set_gait_enable(True)
    for _ in range(40):
        step(cmd=(0.06, 0.0, 0.0))
    R.set_gait_enable(False)
    k = 0
    while bool(R._w3_clock.ge) and k < 60:
        step()
        k += 1
    n_cap = 0
    while R.mode == "w3" and n_cap < 400:
        step(w=0.5)
        n_cap += 1
    ev["cap_ticks_to_ke"] = n_cap
    # 한 발 접촉이면 조용해도 넘기지 않는다
    R.reset()
    step()
    R.set_gait_enable(True)
    for _ in range(40):
        step(cmd=(0.06, 0.0, 0.0))
    R.set_gait_enable(False)
    k = 0
    while bool(R._w3_clock.ge) and k < 60:
        step()
        k += 1
    n1 = 0
    for _ in range(40):
        step(contact=(1.0, 0.0))
        n1 += int(R.mode == "w3")
    ev["single_support_stays_w3"] = (n1 == 40)
    # 회복 모드 켜짐 중 켜기 차단
    R.reset()
    for _ in range(100):
        step()
    for _ in range(6):
        step(w=2.0)
    ev["block_reason_in_recovery"] = R.enable_block_reason()
    R.reset()
    return ev


class _FakeClock:
    def __init__(self, t0=2_000_000_000.0):
        self.t = t0

    def time(self):
        return self.t

    def __getattr__(self, name):
        return getattr(time, name)


def _w_node_session(B, key, logs, tmp_home, A, tag, walk=False):
    """rl_bridge_node.main() 을 가짜 rclpy 로 그대로 실행(--live, 발행은 기록만). 플랜트 = 발행 목표를 1 차 지연으로 따라가는 가정 모형."""
    import numpy as np
    clock = _FakeClock()
    B.time = clock
    pubs = {}
    rec = {"t": [], "raw": [], "tgt_joint": [], "tgt_motor": [], "rel": [], "alpha": [], "q_fb": [], "obs": [], "mode": [], "w3": [],
           "active": []}
    info = {"errors": [], "acks": []}

    class _Msg:
        def __init__(self, data=""):
            self.data = data

    class _Pub:
        def __init__(self, topic):
            self.topic, self.msgs = topic, []

        def publish(self, m):
            self.msgs.append(m.data)

    class _Log:
        def info(self, s):
            pass

        def debug(self, s):
            pass

        def warn(self, s):
            pass

        warning = warn

        def error(self, s):
            info["errors"].append(str(s)[:200])

    class _Node:
        def __init__(self, name):
            self._subs, self._log = {}, _Log()

        def create_subscription(self, typ, topic, cb, depth):
            self._subs[topic] = cb

        def create_publisher(self, typ, topic, depth):
            pubs[topic] = _Pub(topic)
            return pubs[topic]

        def create_timer(self, period, cb):
            return None

        def get_logger(self):
            return self._log

        def destroy_node(self):
            pass

    def driver(node):
        JO = list(B.JOINT_ORDER)
        fm = node.frames
        pose = np.asarray(B.ckpt_default_pose(key), np.float64)
        q = pose + np.radians([1.0, -1.2, 0.3, -0.2, 0.1, 0.0, -0.8, 0.9, 0.6, -0.5, 0.2, -0.1])       # 스탠바이 자세 = 정책 기본자세에서 ~1° 어긋남
        base = {j: fm.joint_rad_to_motor_deg(j, float(q[i])) for i, j in enumerate(JO)}
        st = {"rel": {j: 0.0 for j in JO}, "k": 0, "pitch": 0.0, "wy": 0.02, "contact": (True, True), "hold": {}, "yaw": 0.0}
        info["dir"] = [float(fm.dir_by_joint[j]) for j in JO]
        info["sz"] = [float(fm.sz_by_joint[j]) for j in JO]
        info["base"] = [float(base[j]) for j in JO]
        info["guard_mode"] = getattr(node, "guard_mode", None)
        info["imu_bias"] = [float(v) for v in B.IMU_MOUNT_BIAS_DEG]

        def feed():
            now = clock.t
            st["k"] += 1
            js = {}
            fb = np.zeros(12)
            for i, j in enumerate(JO):
                m = fm.joint_rad_to_motor_deg(j, float(q[i]))
                if j in st["hold"] and st["hold"][j][0] > 0:       # 라운드로빈 폴링 흉내: 피드백 값이 N 틱 머문다
                    st["hold"][j][0] -= 1
                    m = st["hold"][j][1]
                else:
                    st["hold"].pop(j, None)
                fb[i] = m
                js[j] = {"baseline_deg": base[j], "latest_actual_deg": m, "feedback_age_sec": 0.0, "commanded_relative_deg": st["rel"][j]}
            node._on_status(_Msg(json.dumps({"armed": True, "joints": js})))
            p, y = st["pitch"], st["yaw"]
            cy, sy, cp, sp = math.cos(y / 2), math.sin(y / 2), math.cos(p / 2), math.sin(p / 2)
            node._on_imu(_Msg(json.dumps({"gyro_rad_s": [0.03 * math.sin(0.1 * st["k"]), st["wy"], 0.0],
                                          "quat_wxyz": [cy * cp, -sy * sp, cy * sp, sy * cp], "timestamp": now})))
            cl, cr = st["contact"]
            node._on_foot(_Msg(json.dumps({"left_n": 40.0 * cl, "right_n": 40.0 * cr, "left_contact": cl, "right_contact": cr,
                                           "timestamp": now})))
            return fb

        def tick(n=1):
            for _ in range(n):
                fb = feed()
                nc = len(pubs[B.COMMAND_TOPIC].msgs)
                nd = len(pubs[B.BRIDGE_DEBUG_TOPIC].msgs)
                node._tick()
                new_c = [json.loads(m) for m in pubs[B.COMMAND_TOPIC].msgs[nc:]]
                new_d = [json.loads(m) for m in pubs[B.BRIDGE_DEBUG_TOPIC].msgs[nd:]]
                tg = next((m["targets"] for m in new_c if m.get("command") == "SET_JOINT_TARGETS"), None)
                dbg = next((m for m in new_d if "targets_deg_joint" in m and m.get("raw_action") is not None), None)     # 합성 러너 인계 틱은 obs 가 null — 정책 틱으로 센다
                rec["t"].append(clock.t)
                rec["q_fb"].append([fm.motor_deg_to_joint_rad(j, float(fb[i])) for i, j in enumerate(JO)])
                rec["active"].append(bool(node.core.active))
                if dbg is not None and tg is not None:
                    r = node.core.runner
                    rec["raw"].append([float(v) for v in r.last_action])
                    rec["obs"].append(np.array(r.last_obs, np.float32) if r.last_obs is not None else np.full(320, np.nan, np.float32))
                    rec["tgt_joint"].append([float(dbg["targets_deg_joint"][j]) for j in JO])
                    rec["tgt_motor"].append([float(dbg["targets_deg_motor"][j]) for j in JO])
                    rec["rel"].append([float(tg[j]) for j in JO])
                    rec["alpha"].append(float(dbg["blend_alpha"]))
                    rec["mode"].append(int(getattr(r, "last_mode", 0)) if hasattr(r, "last_mode") or "core" in r.__dict__ else 0)
                    w = dbg.get("w3")
                    rec["w3"].append([float(w["u"]), float(w["ge"]), float(w["ge_obs"]), float(w["want"])] if w else [0.0, 0.0, 0.0, 0.0])
                    for i, j in enumerate(JO):
                        st["rel"][j] = float(tg[j])
                        q[i] += 0.35 * (fm.motor_deg_to_joint_rad(j, base[j] + float(tg[j])) - q[i])
                else:
                    for kk in ("raw", "tgt_joint", "tgt_motor", "rel"):
                        rec[kk].append([float("nan")] * 12)
                    rec["obs"].append(np.full(320, np.nan, np.float32))
                    rec["alpha"].append(float("nan"))
                    rec["mode"].append(-1)
                    rec["w3"].append([float("nan")] * 4)
                clock.t += B.TICK_SEC

        def cmd(d):
            node._on_cmd(_Msg(json.dumps(d)))
            ap = pubs.get(getattr(B, "BRIDGE_ACK_TOPIC", "_none_"))
            if ap is not None and ap.msgs:
                a = json.loads(ap.msgs[-1])
                info["acks"].append({"cmd": a.get("cmd"), "accepted": a.get("accepted"), "why": str(a.get("why"))[:120]})

        tick(3)
        cmd({"cmd": "stand"})
        tick(170)                                              # 인계(블렌딩 1.5 s) + 감지기 기준 자세 창
        JOh = JO[6]
        st["hold"][JOh] = [3, fm.joint_rad_to_motor_deg(JOh, float(q[6]))]     # 한 관절 피드백이 3 틱 머문 뒤 갱신(유한차분 '머문 틱' 식 계측)
        tick(8)
        if not walk:
            st["pitch"] = math.radians(6.5)
            st["wy"] = 0.7
            tick(1)
            st["wy"] = 0.02
            tick(11)
            st["pitch"] = 0.0
            tick(150)
        else:
            cmd({"cmd": "step", "enable": True, "req_id": "a", "seq": 1, "vx": 0.06})        # heading_loop 기본값(켜짐)
            tick(60)
            for i in range(50):                                # 요가 천천히 돈다(IMU) → 헤딩 고리가 wz 를 넣는지
                st["yaw"] = math.radians(0.08 * (i + 1))
                tick(1)
            info["cmd_obs_hl_on"] = [float(v) for v in np.asarray(node.core.runner.last_obs)[30 + 12:30 + 15]]
            info["core_cmd"] = [float(v) for v in node.core.cmd]
            cmd({"cmd": "step", "enable": False, "req_id": "a", "seq": 2})
            tick(120)
            cmd({"cmd": "step", "enable": True, "req_id": "a", "seq": 3, "vx": 0.06, "heading_loop": False})
            tick(40)
            st["yaw"] = math.radians(6.0)
            tick(10)
            info["cmd_obs_hl_off"] = [float(v) for v in np.asarray(node.core.runner.last_obs)[30 + 12:30 + 15]]
            cmd({"cmd": "step", "enable": False, "req_id": "a", "seq": 4})
            tick(100)
        if getattr(node.core, "kew3", False):
            info["kew3"] = node.core.kew3_snapshot()
        cmd({"cmd": "estop"})
        tick(2)

    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    rclpy.spin = driver
    node_mod = types.ModuleType("rclpy.node")
    node_mod.Node = _Node
    rclpy.node = node_mod
    std = types.ModuleType("std_msgs")
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.String = _Msg
    std.msg = std_msg
    sys.modules.update({"rclpy": rclpy, "rclpy.node": node_mod, "std_msgs": std, "std_msgs.msg": std_msg})
    argv = ["--live", "--force-frame", "--checkpoint", key]
    if key in getattr(sys.modules.get("rl_walking.deploy.w3_deploy"), "W3_KEYS", ()) or key in getattr(B, "KEW3_KEYS", ()):
        argv = ["--experimental-w3"] + argv
    if key in getattr(B, "KEW3_KEYS", ()):
        argv = ["--experimental-kew3"] + argv
    old_argv, old_home = sys.argv, os.environ.get("HOME")
    sys.argv = ["rl_bridge_node.py"] + argv
    os.environ["HOME"] = tmp_home                   # 블랙박스(~/logs)는 임시 폴더로
    try:
        rc = B.main()
    finally:
        sys.argv = old_argv
        if old_home is not None:
            os.environ["HOME"] = old_home
        B.time = time
        _w_relocate(B, logs)
    info["rc"] = rc
    info["argv"] = argv
    for k, v in rec.items():
        A[f"{tag}/{k}"] = np.asarray(v, np.float32 if k == "obs" else np.float64)
    return info


def _w_replay(B, logs, path):
    """실물 블랙박스(젯슨이 낸 관측·행동·목표)를 이 트리의 BridgeCore 에 재생. 직전 원시 행동만 기록값으로 맞춘다(교사 강제)."""
    import numpy as np
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "event" not in d and d.get("obs") is not None:
                rows.append(d)
    if not rows:
        return {"n": 0}
    key = rows[0]["checkpoint"]
    if key not in B.CHECKPOINTS:
        return {"n": len(rows), "key": key, "registered": False}
    saved = list(B.IMU_MOUNT_BIAS_DEG)
    B.IMU_MOUNT_BIAS_DEG[:] = [float(v) for v in rows[0].get("imu_mount_bias_deg", [0.0, 0.0])]
    try:
        core = B.BridgeCore(key)
        core.reset()
        core.heading_hold = False
        core.cmd = (0.0, 0.0, 0.0)
        R = core.runner
        c = _w_core(R)
        offs, o = {}, 0
        for nme, dim in c.terms:
            L = c._hist[nme]._buf.shape[0]
            offs[nme] = (o, o + dim * L)
            o += dim * L
        e_obs = {nme: 0.0 for nme in offs}
        e_raw = e_tgt = e_net_s = e_net_r = e_ref = 0.0
        mode_mis = 0
        filt = None
        prev_raw = None
        is_c5 = "core" in R.__dict__
        for d in rows:
            if prev_raw is not None:
                c.last_action = np.asarray(prev_raw, np.float32)
            out = core.tick(d["gyro_rad_s"], d["quat_wxyz"], d["qpos_rad"], d["qvel_rad_s"], d["contact"])
            ob = np.asarray(c.last_obs, np.float64)
            lo = np.asarray(d["obs"], np.float64)
            for nme, (a, b) in offs.items():
                e_obs[nme] = max(e_obs[nme], float(np.abs(ob[a:b] - lo[a:b]).max()))
            e_raw = max(e_raw, float(np.abs(np.asarray(c.last_action, np.float64) - np.asarray(d["raw_action"])).max()))
            if is_c5 and d.get("c5"):
                c5 = d["c5"]
                lo32 = np.asarray(d["obs"], np.float32)
                e_net_s = max(e_net_s, float(np.abs(np.asarray(c._net(lo32), np.float64) - np.asarray(c5["raw_stand"])).max()))
                if not c5.get("cue"):
                    e_net_r = max(e_net_r, float(np.abs(np.asarray(R._rec_net(lo32), np.float64) - np.asarray(c5["raw_rec"])).max()))
                mode_mis += int(int(R.last_mode) != int(c5["mode"]))
                if c5.get("ref_deg") is not None and R.det.ref is not None:
                    e_ref = max(e_ref, float(np.abs(np.asarray(R.det.ref, np.float64) - np.asarray(c5["ref_deg"])).max()))
            filt = B.lpf_step(filt, [float(v) for v in out])
            tj = np.array([d["targets_deg_joint"][j] for j in B.JOINT_ORDER])
            e_tgt = max(e_tgt, float(np.abs(np.degrees(np.asarray(filt)) - tj).max()))
            prev_raw = d["raw_action"]
        res = {"n": len(rows), "key": key, "registered": True, "obs_err": e_obs, "raw_err": e_raw, "tgt_deg_err": e_tgt,
               "net_stand_err": e_net_s, "net_rec_err": e_net_r, "mode_mismatch": mode_mis, "ref_deg_err": e_ref,
               "backend_here": c.backend, "is_c5": is_c5}
        if is_c5:
            res.update(n_on=int(R.det.n_on), n_early=int(getattr(R.det, "n_early", 0)),
                       log_n_on=int(rows[-1]["c5"]["n_on"]), log_n_early=int(rows[-1]["c5"].get("n_early", 0)))
        return res
    finally:
        B.IMU_MOUNT_BIAS_DEG[:] = saved


def worker_main(a):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    tree = os.path.abspath(a.tree)
    B = _w_load_bridge(tree)
    _w_relocate(B, a.logs)
    W3 = sys.modules.get("rl_walking.deploy.w3_deploy")
    for fn in ("enable_experimental_w3", "enable_experimental_kew3"):       # 등록 함수가 경로를 다시 쓰므로 그 뒤에 옮긴다(main() 안 호출 포함)
        if hasattr(B, fn):
            orig = getattr(B, fn)
            setattr(B, fn, (lambda o: (lambda: (o(), _w_relocate(B, a.logs))[0]))(orig))
    reg_err = {}
    if hasattr(B, "enable_experimental_w3"):
        B.enable_experimental_w3()
    if hasattr(B, "enable_experimental_kew3"):
        try:
            B.enable_experimental_kew3()
        except Exception as e:  # noqa: BLE001 — 합성 키 등록 거부(하위 키 계약 불일치 등)는 그 키만 '미등록'으로 두고 나머지를 계속 감사한다
            reg_err["kew3"] = f"{type(e).__name__}: {e}"
    PR4 = sys.modules["rl_walking.deploy.policy_runner_stage4"]
    DC = sys.modules["rl_walking.deploy.deploy_chain"]
    g = lambda n, d=None: getattr(B, n, d)  # noqa: E731
    keys = a.keys.split(",")
    reg = {"JOINT_ORDER": list(B.JOINT_ORDER), "RATE_HZ": B.RATE_HZ, "TICK_SEC": B.TICK_SEC, "BLEND_S": B.BLEND_S,
           "STAND_LPF_BETA": B.STAND_LPF_BETA, "OBS_ACT_EMA_BETA_DEPLOY": g("OBS_ACT_EMA_BETA_DEPLOY"), "POLICY_KP": B.POLICY_KP,
           "POLICY_KD": B.POLICY_KD, "SLEW_RAD": B.SLEW_RAD, "ACTION_SCALE": DC.ACTION_SCALE, "STAND_DEFAULT": g("STAND_DEFAULT"),
           "CONTACT_FORCE_THRESHOLD_N": g("CONTACT_FORCE_THRESHOLD_N"), "SENSOR_STALE_SEC": g("SENSOR_STALE_SEC"),
           "FOOT_STALE_SEC": g("FOOT_STALE_SEC"), "LIFT_STOP_S": g("LIFT_STOP_S"), "CKPT_GAIT_FREQ_DEFAULT": g("CKPT_GAIT_FREQ_DEFAULT"),
           "C5_DETECTOR_KW": g("C5_DETECTOR_KW"), "RUNNER_EMA_FRESH": g("RUNNER_EMA_FRESH"), "C5_IMU_F32": g("C5_IMU_F32"),
           "C5_SETTLE": g("C5_SETTLE"), "HISTORY_LEN": PR4.HISTORY_LEN, "EMA_ALPHA_S8A": PR4.EMA_ALPHA_S8A,
           "GAIT_FREQ_HZ": PR4.GAIT_FREQ_HZ, "POLICY_DT": PR4.POLICY_DT, "keys": {}, "reg_err": reg_err}
    try:
        fm = B.load_frame_map()
        reg["map"] = {"path": os.path.relpath(fm.path, tree) if str(fm.path).startswith(tree) else str(fm.path),
                      "gain_mismatches": [list(x) for x in fm.gain_mismatches()],
                      "roll_comp_enabled": bool((fm.roll_comp or {}).get("enabled", False)) if isinstance(fm.roll_comp, dict) else bool(fm.roll_comp),
                      "pitch_comp_enabled": bool((fm.pitch_comp or {}).get("enabled", False)) if isinstance(fm.pitch_comp, dict) else bool(fm.pitch_comp),
                      "guard_mode": getattr(fm, "guard_mode", None), "imu_mount_bias_deg": [float(v) for v in B.IMU_MOUNT_BIAS_DEG],
                      "dir": [float(fm.dir_by_joint[j]) for j in B.JOINT_ORDER]}
    except Exception as e:  # noqa: BLE001
        reg["map"] = {"error": repr(e)}
    A = {}
    out = {"tree": tree, "registry": reg, "probe": {}, "node": {}, "replay": {}, "torch": torch.__version__}
    # 중력 투영식·장착 편향 보정(순수 함수) — 부모가 독립 식과 대조
    rng = np.random.default_rng(11)
    qs = rng.standard_normal((400, 4))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    A["grav/quat"] = qs
    A["grav/pg"] = np.array([B.projected_gravity(tuple(q)) for q in qs])
    A["grav/yaw"] = np.array([PR4.yaw_from_quat_wxyz(*q) for q in qs])
    rp = np.radians(rng.uniform(-12, 12, (200, 2)))
    gin = np.stack([np.tan(rp[:, 1]), np.tan(rp[:, 0]), -np.ones(200)], 1)
    gin /= np.linalg.norm(gin, axis=1, keepdims=True)
    A["grav/cg_in"] = gin
    A["grav/cg_out"] = np.array([B.correct_gravity(tuple(v), [1.3, -0.6]) for v in gin])
    A["clock/long"] = np.array([PR4.gait_clock_stage4(t, 1.1) for t in (0, 1, 1000, 30000, 180000)], np.float32)
    tmp_home = tempfile.mkdtemp(prefix="ca_home_")

    def ensure_registered():      # main() 은 --experimental-kew3 없이 불리면 합성 키 등록을 되돌린다(같은 프로세스 재호출 규칙) → 키마다 다시 확인
        if hasattr(B, "enable_experimental_w3") and not B.EXPERIMENTAL_W3:
            B.enable_experimental_w3()
        if hasattr(B, "enable_experimental_kew3") and not B.EXPERIMENTAL_KEW3 and "kew3" not in reg_err:
            B.enable_experimental_kew3()
    try:
        for key in keys:
            ensure_registered()
            K = {"registered": key in B.CHECKPOINTS}
            reg["keys"][key] = K
            if not K["registered"]:
                continue
            K["ckpt_rel"] = _w_rel(B, a.logs, B.CHECKPOINTS[key])
            K["default_pose"] = [float(v) for v in B.ckpt_default_pose(key)]
            K["gait_freq"] = B.CKPT_GAIT_FREQ.get(key, B.CKPT_GAIT_FREQ_DEFAULT)
            K["jvel_ema_mode_reg"] = B.CKPT_JVEL_EMA_MODE.get(key, "v2")
            K["require_env_yaml"] = key in B.CKPT_REQUIRE_ENV_YAML
            K["lpf_applies"] = bool(key.startswith("stand"))
            K["c5_pair"] = None
            if key in getattr(B, "C5_PAIRS", {}):
                s, p = B.C5_PAIRS[key]
                K["c5_pair"] = [s, _w_rel(B, a.logs, p)]
                K["c5_extra"] = getattr(B, "C5_EXTRA", {}).get(key, {})
            if key in getattr(B, "KEW3_CKPTS", {}):
                sp = B.KEW3_CKPTS[key]
                K["kew3"] = {"w3": sp.w3, "ke": sp.ke, "handoff": sp.handoff, "settle_quiet_s": sp.settle_quiet_s}
            if W3 is not None and key in getattr(B, "W3_CKPTS", {}):
                import dataclasses
                K["w3cfg"] = dataclasses.asdict(B.W3_CKPTS[key])
                K["w3_run_rel"] = K["ckpt_rel"]
                if key in B.W3_CMD_LIMITS:
                    K["w3_limits"] = dataclasses.asdict(B.W3_CMD_LIMITS[key])
                K["w3_safety"] = dataclasses.asdict(B.W3_SAFETY)
                K["w3_hl"] = None if B.W3_HL_CFG is None else dataclasses.asdict(B.W3_HL_CFG)
                K["w3_moves"] = {k: dict(v) for k, v in getattr(W3, "W3_MOVES", {}).items()}
                K["w3_cmd_limit_ranges"] = getattr(W3, "W3_CMD_LIMIT_RANGES", None)
            P = {}
            out["probe"][key] = P
            try:
                R = B._make_runner(key)
                P["runner"] = type(R).__name__
                if P["runner"] == "KeW3Runner":
                    P["attrs_ke"] = _w_runner_attrs(B, _w_core(R.ke))
                    P["attrs_w3"] = _w_runner_attrs(B, _w_core(R.w3r))
                    P["det"] = _w_det_attrs(R.ke.det)
                    P["cue"] = {"cue_vx": R.ke.cue_vx, "cue_s": R.ke.cue_s, "cue_dir_imu": R.ke.cue_dir_imu}
                    P["kew3"] = _w_probe_kew3(B, key, R, A, f"probe/{key}")
                else:
                    P["attrs"] = _w_runner_attrs(B, _w_core(R))
                    if P["runner"] == "C5Runner":
                        P["det"] = _w_det_attrs(R.det)
                        P["cue"] = {"cue_vx": R.cue_vx, "cue_s": R.cue_s, "cue_dir_imu": R.cue_dir_imu}
                        P["rec_backend"] = R.rec_backend
                    P["seq"] = _w_probe_seq(B, key, R, A, f"probe/{key}")
            except Exception as e:  # noqa: BLE001
                import traceback
                P["error"] = f"{type(e).__name__}: {e}"
                P["trace"] = traceback.format_exc()[-1500:]
            if not a.no_node:
                try:
                    walk = bool(W3 is not None and key in getattr(B, "W3_CKPTS", {}))
                    out["node"][key] = _w_node_session(B, key, a.logs, tmp_home, A, f"node/{key}", walk=walk)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    out["node"][key] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
        out["replay"] = {}
        for rp_ in [x for x in (a.replay or "").split(",") if x]:
            if not os.path.isfile(rp_):
                continue
            try:
                out["replay"][os.path.basename(rp_)] = _w_replay(B, a.logs, rp_)
            except Exception as e:  # noqa: BLE001
                import traceback
                out["replay"][os.path.basename(rp_)] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    finally:
        shutil.rmtree(tmp_home, ignore_errors=True)
    np.savez(a.out_npz, **A)
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=lambda o: (o.tolist() if hasattr(o, "tolist") else str(o)))
    return 0


# =====================================================================================================================
# 부모 — 젯슨 판 트리 구성, 워커 실행, 검산, 표
# =====================================================================================================================
def build_jetson_tree(dst):
    """dst 에 젯슨 판 소스 트리(e7710c63 의 rl_walking/deploy·robot_runtime·config + 후보 브리지)를 만든다. 반환 = 근거 dict."""
    ev = {}
    paths = ["rl_walking/__init__.py", "rl_walking/deploy", "robot_runtime", "config"]
    data = git("archive", JET_COMMIT, *paths, binary=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(dst)
    os.makedirs(os.path.join(dst, "jetson", "scripts"), exist_ok=True)
    cand = os.path.join(REPO, JET_BRIDGE_REL)
    shutil.copyfile(cand, os.path.join(dst, "jetson", "scripts", "rl_bridge_node.py"))
    ev["bridge_md5"] = md5_of(cand)
    ev["runner_md5"] = md5_of(os.path.join(dst, "rl_walking/deploy/policy_runner_stage4.py"))
    ev["c5_md5"] = md5_of(os.path.join(dst, "rl_walking/deploy/c5_switch.py"))
    ev["base_bridge_md5"] = hashlib.md5(git("show", f"{JET_COMMIT}:jetson/scripts/rl_bridge_node.py", binary=True)).hexdigest()
    # e7710c63 브리지 + 패치(브리지 부분) = 후보 파일인가
    ev["patch_reproduces"] = None
    try:
        with open(os.path.join(REPO, JET_PATCH_REL), encoding="utf-8") as f:
            ptxt = f.read()
        heads = [m.start() for m in re.finditer(r"^--- ", ptxt, re.M)]
        part = ptxt[heads[0]:heads[1]] if len(heads) > 1 else ptxt
        with tempfile.TemporaryDirectory(prefix="ca_patch_") as td:
            src = os.path.join(td, "b.py")
            with open(src, "wb") as f:
                f.write(git("show", f"{JET_COMMIT}:jetson/scripts/rl_bridge_node.py", binary=True))
            pf = os.path.join(td, "p.patch")
            with open(pf, "w", encoding="utf-8") as f:
                f.write(part)
            o = os.path.join(td, "out.py")
            r = subprocess.run(["patch", "-s", "-o", o, src, pf], capture_output=True)
            if r.returncode == 0 and os.path.isfile(o):
                ev["patch_reproduces"] = (md5_of(o) == ev["bridge_md5"])
    except Exception as e:  # noqa: BLE001
        ev["patch_error"] = repr(e)
    # 배포 매니페스트(저장소 기록)와의 md5 대조
    man = {}
    for rel in ("deploy/c5krh_1007.md5", "deploy/c5kr_1005.md5", "deploy/g21a_1003.md5"):
        p = os.path.join(REPO, rel)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    sp = ln.split()
                    if len(sp) == 2 and len(sp[0]) == 32:
                        man.setdefault(sp[1], {})[rel] = sp[0]
    ev["manifest"] = man
    return ev


def run_worker(tree, logs, keys, out_dir, tag, replay=None, no_node=False):
    oj, on = os.path.join(out_dir, f"{tag}.json"), os.path.join(out_dir, f"{tag}.npz")
    env = dict(os.environ)
    env["PYTHONPATH"] = ""                 # ROS(rclpy)·저장소 루트가 섞이지 않게 — 워커는 가짜 rclpy 만 쓴다
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), "--_worker", "--tree", tree, "--logs", logs, "--keys", ",".join(keys),
           "--out-json", oj, "--out-npz", on]
    if replay:
        cmd += ["--replay", ",".join(replay) if isinstance(replay, (list, tuple)) else replay]
    if no_node:
        cmd += ["--no-node"]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=tree), oj, on


# ---------- 검산(부모, numpy) ----------
def layout_offsets(layout):
    offs, o = {}, 0
    for name, dim, hist in layout:
        offs[name] = (o, dim, hist)
        o += dim * hist
    return offs, o


def ref_obs(layout, P, S, raw, clock_frames=None, sim_fresh=True):
    """학습 계약대로 다시 조립한 관측열(N, D). P = default·beta_act·beta_jv·alpha·freq·dt. sim_fresh: S8a EMA 재시작 = 틱 ≤ 1(시뮬) / 틱 0 만(배포 기본)."""
    import numpy as np
    f32 = np.float32
    N = raw.shape[0]
    fr = {"base_ang_vel": S["gyro"], "projected_gravity": S["grav"], "velocity_commands": S["cmd"],
          "joint_pos": (S["qpos"] - np.asarray(P["default"], f32)[None]).astype(f32), "feet_contact": S["contact"]}
    qv = S["qvel"]
    if P["beta_jv"] is None:
        fr["joint_vel"] = qv
    else:
        y = np.zeros_like(qv)
        y[0] = qv[0]
        b = f32(P["beta_jv"])
        for n in range(1, N):
            y[n] = y[n - 1] + b * (qv[n] - y[n - 1])
        fr["joint_vel"] = y
    ap = np.vstack([np.zeros((1, 12), f32), raw[:-1]])
    if P["beta_act"] is None:
        fr["actions"] = ap
    else:
        y = np.zeros_like(ap)
        y[0] = ap[0]
        b = f32(P["beta_act"])
        for n in range(1, N):
            y[n] = y[n - 1] + b * (ap[n] - y[n - 1])
        fr["actions"] = y
    if clock_frames is not None:
        fr["gait_clock"] = clock_frames
    else:
        t = np.arange(N, dtype=f32) * f32(P["dt"])
        ph = f32(2.0 * math.pi * P["freq"]) * t
        fr["gait_clock"] = np.stack([np.sin(ph), np.cos(ph)], 1).astype(f32)
    a = f32(P["alpha"])
    for name, x in (("gyro_ema", S["gyro"]), ("joint_vel_ema", qv)):
        e = np.zeros_like(x)
        st = None
        for n in range(N):
            if st is None or (sim_fresh and n <= 1):
                st = x[n].copy()
                if not sim_fresh:
                    e[n] = st
                    continue
            st = ((f32(1.0) - a) * st + a * x[n]).astype(f32)
            e[n] = st
        fr[name] = e
    cols = []
    for name, dim, hist in layout:
        f = fr[name]
        idx = np.clip(np.arange(N)[:, None] - (hist - 1) + np.arange(hist)[None, :], 0, None)      # 오래된 것 먼저 + 첫 틱 백필
        cols.append(f[idx].reshape(N, hist * dim))
    return np.concatenate(cols, 1).astype(f32)


def analyze_probe(Z, key, layout, P, w3=False):
    """워커 기록 → 계측값·재구성 오차."""
    import numpy as np
    f32 = np.float32
    g = lambda n: Z[f"probe/{key}/{n}"]  # noqa: E731
    S = {k: g(f"in_{k}") for k in ("gyro", "grav", "cmd", "qpos", "qvel", "contact")}
    obs, raw, out = g("obs"), g("raw"), g("out")
    N = obs.shape[0]
    offs, D = layout_offsets(layout)
    M = {"obs_dim_ok": obs.shape[1] == D}
    if not M["obs_dim_ok"]:
        return M
    clk = None
    if w3:
        c = g("clk")
        ph = f32(2.0 * math.pi) * c[:, 0]
        clk = (np.stack([np.sin(ph), np.cos(ph)], 1) * c[:, 2:3]).astype(f32)
    ref_sim = ref_obs(layout, P, S, raw, clk, sim_fresh=True)
    ref_dep = ref_obs(layout, P, S, raw, clk, sim_fresh=False)
    M["err"], M["err_early"], M["err_late"], M["err_deprule"] = {}, {}, {}, {}
    for name, (o, dim, hist) in offs.items():
        sl = slice(o, o + dim * hist)
        d = np.abs(obs[:, sl] - ref_sim[:, sl])
        M["err"][name] = float(d.max())
        M["err_early"][name] = float(d[:100].max())
        M["err_late"][name] = float(d[200:].max())
        M["err_deprule"][name] = float(np.abs(obs[:, sl] - ref_dep[:, sl]).max())
    new = lambda name: obs[:, offs[name][0] + offs[name][1] * (offs[name][2] - 1): offs[name][0] + offs[name][1] * offs[name][2]]  # noqa: E731
    old = lambda name: obs[:, offs[name][0]: offs[name][0] + offs[name][1]]  # noqa: E731
    # 이력 순서: 최신 칸 = 이번 틱 입력, 맨 앞 칸 = (이력−1) 틱 전 입력, 첫 틱 = 전 칸 같은 값
    h = offs["base_ang_vel"][2]
    M["hist_newest_last"] = bool(np.array_equal(new("base_ang_vel"), S["gyro"]) and np.array_equal(old("base_ang_vel")[h - 1:], S["gyro"][:N - h + 1]))
    M["hist_backfill"] = bool(all(np.array_equal(obs[0, o:o + dim * hist].reshape(hist, dim), np.tile(obs[0, o:o + dim], (hist, 1)))
                                  for o, dim, hist in offs.values()))
    M["scale_passthrough"] = bool(np.array_equal(new("base_ang_vel"), S["gyro"]) and np.array_equal(new("projected_gravity"), S["grav"])
                                  and np.array_equal(new("feet_contact"), S["contact"]))
    # 기본자세(관측): joint_pos 최신 = qpos − default
    M["default_from_obs"] = (S["qpos"] - new("joint_pos")).astype(np.float64).mean(0)
    # 필터 계수 역산
    def fit(y, x):          # y[n] − y[n−1] = k (x[n] − y[n−1])
        dy = (y[1:] - y[:-1]).astype(np.float64).ravel()
        dx = (x[1:] - y[:-1]).astype(np.float64).ravel()
        return float((dy * dx).sum() / max((dx * dx).sum(), 1e-30))
    ap = np.vstack([np.zeros((1, 12), f32), raw[:-1]])
    M["beta_act"] = fit(new("actions"), ap)
    M["beta_jv"] = fit(new("joint_vel"), S["qvel"])
    M["alpha_gyro"] = fit(new("gyro_ema")[2:], S["gyro"][2:])
    M["alpha_jv"] = fit(new("joint_vel_ema")[2:], S["qvel"][2:])
    # 직전 행동의 정의: 원시 출력 재귀 잔차 vs '슬루 뒤 목표를 행동 단위로 되돌린 값'을 넣었을 때 잔차(변이)
    ya = new("actions").astype(np.float64)
    b = M["beta_act"]
    M["prev_action_resid_raw"] = float(np.abs(ya[1:] - (ya[:-1] + b * (ap[1:] - ya[:-1]))).max())
    emitted_as_act = (np.vstack([np.zeros((1, 12)), (out[:-1] - np.asarray(P["default"])[None]) / P["scale"]]))
    M["prev_action_resid_emitted"] = float(np.abs(ya[1:] - (ya[:-1] + b * (emitted_as_act[1:] - ya[:-1]))).max())
    # 클록 주파수(시간식 클록): 위상 기울기
    if not w3:
        c = new("gait_clock").astype(np.float64)
        ph = np.unwrap(np.arctan2(c[:, 0], c[:, 1]))
        M["clock_freq"] = float(np.polyfit(np.arange(N) * P["dt"], ph, 1)[0] / (2.0 * math.pi))
        M["clock_t0"] = [float(c[0, 0]), float(c[0, 1])]
    else:
        c = g("clk").astype(np.float64)
        want = g("want")
        walking = (c[:, 1] > 0.5)
        du = np.diff(c[:, 0])
        ok = walking[1:] & walking[:-1] & (du > 0)
        M["w3_freq"] = float(np.median(du[ok]) / P["dt"]) if ok.any() else float("nan")
        on_idx = [i for i in range(1, N) if want[i] and not want[i - 1]]
        off_idx = [i for i in range(1, N) if (not want[i]) and want[i - 1]]
        M["w3_on_delay"] = [int(next(j for j in range(i, N) if c[j, 1] > 0.5) - i) for i in on_idx]
        M["w3_ramp_ticks"] = [int(next(j for j in range(i, N) if c[j, 2] >= 1.0 - 1e-6) - i + 1) for i in on_idx if any(c[j, 2] >= 1.0 - 1e-6 for j in range(i, N))]
        offs_u = []
        for i in off_idx:
            j = next((j for j in range(i, N) if c[j, 1] < 0.5), None)
            if j is not None:
                un = c[j - 1, 0] + P["f_step"] * P["dt"]
                offs_u.append((int(j - i), float(c[j - 1, 0]), bool((c[j - 1, 0] < 0.5 <= un) or un >= 1.0)))
        M["w3_off"] = offs_u
        M["w3_start_u"] = [float(c[i, 0]) for i in on_idx]
        M["w3_stand_clock_zero"] = bool(np.abs(new("gait_clock")[:40]).max() == 0.0)
        M["w3_cmd_obs_eq_input"] = bool(np.array_equal(new("velocity_commands"), S["cmd"]))
    # 행동: 스케일·기본자세(포화 안 된 틱에서 out = d + s·raw), 슬루
    d_out = np.abs(np.diff(out.astype(np.float64), axis=0))
    M["slew_max"] = float(max(d_out.max(), float(np.abs(out[0].astype(np.float64) - S["qpos"][0]).max())))
    M["slew0"] = float(np.abs(Z[f"probe/{key}/slew0"]).max())
    M["slew0_all_equal"] = bool(np.allclose(np.abs(Z[f"probe/{key}/slew0"]), M["slew0"], atol=1e-6))
    unsat = np.vstack([np.zeros((1, 12), bool), d_out < (P["slew"] * 0.98)])
    sc, df = [], []
    for j in range(12):
        m = unsat[:, j]
        if m.sum() >= 20 and float(np.ptp(raw[m, j])) > 1e-3:
            A_ = np.stack([raw[m, j].astype(np.float64), np.ones(int(m.sum()))], 1)
            s_, d_ = np.linalg.lstsq(A_, out[m, j].astype(np.float64), rcond=None)[0]
            sc.append(float(s_))
            df.append(float(d_))
        else:
            sc.append(float("nan"))
            df.append(float("nan"))
    M["scale_fit"], M["default_fit"], M["unsat_frac"] = sc, df, float(unsat.mean())
    # 목표각 재구성(학습 식: 기본자세 + 스케일·원시 → 슬루, 기준 = 첫 틱 qpos)
    prev = S["qpos"][0].astype(f32).copy()
    dfl = np.asarray(P["default"], f32)
    worst = 0.0
    for n in range(N):
        tgt = dfl + f32(P["scale"]) * raw[n]
        prev = (prev + np.clip(tgt - prev, -f32(P["slew"]), f32(P["slew"]))).astype(f32)
        worst = max(worst, float(np.abs(prev - out[n]).max()))
    M["act_err"] = worst
    # 클립: 큰 값이 그대로 들어가는가
    co = Z[f"probe/{key}/clip_obs"]
    nw = lambda name: co[offs[name][0] + offs[name][1] * (offs[name][2] - 1): offs[name][0] + offs[name][1] * offs[name][2]]  # noqa: E731
    M["clip_probe"] = {"base_ang_vel": float(np.abs(nw("base_ang_vel")).max()), "joint_vel": float(np.abs(nw("joint_vel")).max()),
                       "joint_pos": float(np.abs(nw("joint_pos")).max()), "gyro_ema": float(np.abs(nw("gyro_ema")).max()),
                       "joint_vel_ema": float(np.abs(nw("joint_vel_ema")).max())}
    M["raw_absmax"] = float(np.abs(raw).max())
    return M


def analyze_node(Z, key, info, layout, P, lpf_beta, blend_s):
    """노드 세션 기록 → LPF β·블렌딩·유한차분 식·처리 순서 재구성 오차."""
    import numpy as np
    f32 = np.float32
    g = lambda n: Z[f"node/{key}/{n}"]  # noqa: E731
    raw, tj, tm, rel, al, qfb, t, obs = g("raw"), g("tgt_joint"), g("tgt_motor"), g("rel"), g("alpha"), g("q_fb"), g("t"), g("obs")
    ok = ~np.isnan(raw[:, 0])
    idx = np.where(ok)[0]
    M = {"n_policy_ticks": int(ok.sum())}
    if ok.sum() < 50:
        return M
    # 연속 구간(첫 stand 이후 estop 전까지 끊김 없는 정책 틱)
    i0 = int(idx[0])
    seg = [i0]
    for i in idx[1:]:
        if i == seg[-1] + 1:
            seg.append(int(i))
        else:
            break
    seg = np.array(seg)
    M["n_seg"] = int(len(seg))
    offs, D = layout_offsets(layout)
    # 러너 슬루 → LPF 재구성
    dfl = np.asarray(P["default"], f32)
    prev = qfb[i0].astype(f32)
    em = np.zeros((len(seg), 12))
    for n, i in enumerate(seg):
        tgt = dfl + f32(P["scale"]) * raw[i].astype(f32)
        prev = (prev + np.clip(tgt - prev, -f32(P["slew"]), f32(P["slew"]))).astype(f32)
        em[n] = prev
    y = np.radians(tj[seg])
    dy = (y[1:] - y[:-1]).ravel()
    dx = (em[1:] - y[:-1]).ravel()
    M["lpf_beta_fit"] = float((dy * dx).sum() / max((dx * dx).sum(), 1e-30))
    M["lpf_first_is_slew_out"] = float(np.abs(y[0] - em[0]).max())
    yy = em[0].copy()
    worst = 0.0
    for n in range(len(seg)):
        if n > 0:
            yy = yy + lpf_beta * (em[n] - yy)
        worst = max(worst, float(np.abs(yy - y[n]).max()))
    M["chain_err_rad"] = worst
    # 블렌딩: α(t) = smoothstep((t − t0)/blend_s), 발행 = 시작점·(1−α) + 모터(LPF 출력)·α
    el = t[seg] - t[i0]
    a_ref = np.clip(el / blend_s, 0.0, 1.0)
    a_ref = a_ref * a_ref * (3.0 - 2.0 * a_ref)
    M["alpha_err"] = float(np.abs(a_ref - al[seg]).max())
    M["alpha_first_one_s"] = float(el[np.argmax(al[seg] >= 1.0)]) if (al[seg] >= 1.0).any() else float("nan")
    dirv, sz, base = np.asarray(info["dir"]), np.asarray(info["sz"]), np.asarray(info["base"])
    motor = sz[None] + dirv[None] * tj[seg]
    frm = tm[i0]                                                    # α = 0 인 첫 틱의 발행 = 시작점
    M["blend_from_is_stage8_cmd"] = float(np.abs(frm - base).max())      # 가짜 stage8 스트리밍 명령 = baseline + rel 0
    recon = frm[None] * (1.0 - a_ref[:, None]) + motor * a_ref[:, None]
    M["blend_err_deg"] = float(np.abs(recon - tm[seg]).max())
    M["pub_rel_err_deg"] = float(np.abs((tm[seg] - base[None]) - rel[seg]).max())
    # 유한차분 속도: 브리지 식(머문 틱으로 나눔) vs 단순 차분 → 관측 joint_vel(β EMA) 최신 프레임과 대조
    q = qfb.astype(np.float64)
    hold = np.ones(12)
    qv_b = np.zeros((len(seg), 12))
    qv_s = np.zeros((len(seg), 12))
    for n, i in enumerate(seg):
        if n == 0:
            continue
        for j in range(12):
            a_, b_ = q[i, j], q[i - 1, j]
            if a_ == b_:
                hold[j] = min(hold[j] + 1, 12)
                qv_b[n, j] = 0.0
            else:
                qv_b[n, j] = (a_ - b_) / (P["dt"] * hold[j])
                hold[j] = 1
            qv_s[n, j] = (a_ - b_) / P["dt"]
    jo, jd, jh = offs["joint_vel"]
    jv_obs = obs[seg][:, jo + jd * (jh - 1): jo + jd * jh].astype(np.float64)

    def ema(x, b):
        y_ = np.zeros_like(x)
        y_[0] = x[0]
        for n in range(1, len(x)):
            y_[n] = y_[n - 1] + b * (x[n] - y_[n - 1])
        return y_
    bj = P["beta_jv"] if P["beta_jv"] is not None else 1.0
    M["jv_err_bridge_law"] = float(np.nanmax(np.abs(ema(qv_b, bj) - jv_obs)))
    M["jv_err_simple_fd"] = float(np.nanmax(np.abs(ema(qv_s, bj) - jv_obs)))
    M["fd_laws_differ_ticks"] = int((np.abs(qv_b - qv_s).max(1) > 1e-9).sum())
    co, cd, ch = offs["velocity_commands"]
    M["cmd_obs_absmax"] = float(np.nanmax(np.abs(obs[seg][:, co:co + cd * ch])))
    go, gd, gh = offs["gait_clock"]
    ck = obs[seg][:, go + gd * (gh - 1): go + gd * gh].astype(np.float64)
    if not P.get("w3"):
        ph = np.unwrap(np.arctan2(ck[:, 0], ck[:, 1]))
        M["clock_freq"] = float(np.polyfit(el, ph, 1)[0] / (2.0 * math.pi))
    M["mode_ticks"] = int((g("mode")[seg] == 1).sum())
    M["slew_pub_max_deg"] = float(np.abs(np.diff(tm[seg], axis=0)).max())
    return M


def independent_gravity(quat):
    """R(q)ᵀ·(0,0,−1) — 회전행렬을 직접 만들어 계산(브리지 식과 다른 경로)."""
    import numpy as np
    w, x, y, z = quat
    Rm = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return Rm.T @ np.array([0.0, 0.0, -1.0])


def real_log_stats(path, ground_until=None):
    """실물 블랙박스 자체에서 읽는 사실: 틱 주기, 블렌딩 α(t), 유한차분 속도 식, 반복 표본 비율, 관측·행동 최대값."""
    import numpy as np
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if "event" not in d and d.get("obs") is not None:
                rows.append(d)
    if len(rows) < 100:
        return None
    t = np.array([d["timestamp"] for d in rows])
    dt = np.diff(t)
    S = {"n": len(rows), "key": rows[0]["checkpoint"], "dur_s": float(t[-1] - t[0]), "dt_mean_ms": float(dt.mean() * 1e3),
         "dt_p01_ms": float(np.percentile(dt, 1) * 1e3), "dt_p99_ms": float(np.percentile(dt, 99) * 1e3), "dt_max_ms": float(dt.max() * 1e3),
         "obs_filter": rows[0].get("obs_filter"), "jvel_ema_mode": rows[0].get("jvel_ema_mode"), "imu_bias": rows[0].get("imu_mount_bias_deg"),
         "guard_mode": rows[0].get("guard_mode"), "cmd_absmax": float(max(abs(v) for d in rows for v in d["cmd"]))}
    al = np.array([d["blend_alpha"] for d in rows])
    el = t - t[0]
    best = None
    for bs in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5):
        a = np.clip(el / bs, 0, 1)
        a = a * a * (3 - 2 * a)
        e = float(np.abs(a - al).max())
        if best is None or e < best[1]:
            best = (bs, e)
    S["blend_s_best"], S["blend_alpha_err"] = best
    q = np.array([d["qpos_rad"] for d in rows])
    qv = np.array([d["qvel_rad_s"] for d in rows])
    hold = np.ones(12)
    eb = es = 0.0
    rep = 0
    differ = 0
    for n in range(1, len(rows)):
        for j in range(12):
            a_, b_ = q[n, j], q[n - 1, j]
            if a_ == b_:
                hold[j] = min(hold[j] + 1, 12)
                vb = 0.0
                rep += 1
            else:
                vb = (a_ - b_) / (0.02 * hold[j])
                if hold[j] > 1:
                    differ += 1
                hold[j] = 1
            eb = max(eb, abs(vb - qv[n, j]))
            es = max(es, abs((a_ - b_) / 0.02 - qv[n, j]))
    S["qvel_err_bridge_law"], S["qvel_err_simple_fd"] = float(eb), float(es)
    S["repeat_frac"] = rep / (12.0 * (len(rows) - 1))
    S["hold_gt1_updates"] = int(differ)
    # 반복 표본을 정지 관절(양자화)과 움직이는 관절(갱신 누락)로 나눈다 — 5 틱 창 평균 속도 기준
    w = 5
    spd = np.abs(q[w:] - q[:-w]) / (w * 0.02)
    same = (q[1:] == q[:-1])[w - 1:]
    mv, st_ = spd > 0.2, spd < 0.02
    S["repeat_moving"] = float(same[mv].mean()) if mv.any() else None
    S["repeat_still"] = float(same[st_].mean()) if st_.any() else None
    dq = np.abs(np.diff(q, axis=0))
    S["quantum_rad"] = float(dq[dq > 0].min()) if (dq > 0).any() else None
    # 발 원카운트 합이 바뀌는 간격(틱) = 어댑터 갱신 주기의 대용. 같은 값이 길게 이어진 구간은 누락인지 같은 값 연속인지 구분할 수 없다.
    fs = [json.dumps(d.get("foot_raw_sum")) for d in rows]
    if any(x != "null" for x in fs):
        runs, c = [], 1                                   # (길이, 그 구간의 값)
        for i in range(1, len(fs)):
            if fs[i] == fs[i - 1]:
                c += 1
            else:
                runs.append((c, fs[i - 1]))
                c = 1
        inner = runs[1:] if len(runs) > 2 else runs      # 처음(잘린 구간)과 마지막(미완)은 뺀다
        # 원값이 전부 0 인 구간(발이 뜬 상태 — 값이 그대로인 것이 정상)은 뺀다
        inner = [r for r, v in inner if any(x for x in (json.loads(v) or []) if x)]
        if inner:
            vals, cnts = np.unique(np.array(inner), return_counts=True)
            S["foot_hold_mode"] = int(vals[np.argmax(cnts)])
            S["foot_hold_mode_n"], S["foot_hold_n"] = int(cnts.max()), int(len(inner))
            S["foot_hold_longer"] = int(sum(1 for r in inner if r > S["foot_hold_mode"]))
            S["foot_hold_max"] = int(max(inner))
    # IMU 표본이 직전 틱과 같은 틱(자이로·쿼터니언 둘 다) — 브리지가 그 틱에 새 IMU 메시지를 못 받은 경우
    S["imu_repeat_frac"] = float(np.mean([(rows[i]["gyro_rad_s"] == rows[i - 1]["gyro_rad_s"]) and (rows[i]["quat_wxyz"] == rows[i - 1]["quat_wxyz"]) for i in range(1, len(rows))]))
    # S8a EMA 시작 규칙 차이의 실제 크기: 틱 1 에서 배포 0.9·x0 + 0.1·x1, 시뮬 x1 → 차 = 0.9·|x1 − x0|
    gy = np.array([d["gyro_rad_s"] for d in rows[:2]])
    S["ema_start_gyro"] = float(0.9 * np.abs(gy[1] - gy[0]).max())
    S["ema_start_jv"] = float(0.9 * np.abs(qv[1] - qv[0]).max())
    ob = np.array([d["obs"] for d in rows])
    ra = np.array([d["raw_action"] for d in rows])
    S["obs_absmax"] = {"base_ang_vel": float(np.abs(ob[:, 0:15]).max()), "joint_pos": float(np.abs(ob[:, 45:105]).max()),
                       "joint_vel": float(np.abs(ob[:, 105:165]).max()), "actions": float(np.abs(ob[:, 165:225]).max()),
                       "gyro_ema": float(np.abs(ob[:, 245:260]).max()), "joint_vel_ema": float(np.abs(ob[:, 260:320]).max())}
    S["raw_absmax"] = float(np.abs(ra).max())
    S["contact_values"] = sorted({float(v) for d in rows for v in d["contact"]})
    # stage8 가 잰 '브리지 발행 → 처리' 지연과 status 나이(명령·피드백 경로의 한 구간 — 모터 응답까지의 총 지연은 아니다)
    s8 = [d["s8_cmd"] for d in rows if isinstance(d.get("s8_cmd"), dict) and d["s8_cmd"].get("p50_age_ms_1s") is not None]
    if s8:
        la = np.array([x["last_age_ms"] for x in s8 if x.get("last_age_ms") is not None])
        S["s8_n"] = len(s8)
        S["s8_age_med_ms"], S["s8_age_p99_ms"], S["s8_age_max_ms"] = float(np.median(la)), float(np.percentile(la, 99)), float(la.max())
        S["s8_stale"] = int(max(x.get("stale", 0) for x in s8)) + int(max(x.get("seq_gaps", 0) for x in s8))
    sa = np.array([d["status_age_s"] for d in rows if d.get("status_age_s") is not None])
    if sa.size:
        S["status_age_med_ms"], S["status_age_max_ms"] = float(np.median(sa) * 1e3), float(sa.max() * 1e3)
    # 목표각이 config soft 한계를 넘은 틱(관절별) · stage8 클램프 누계 · 거부 사유
    try:
        with open(os.path.join(REPO, "config", "joint_limits_12dof.json"), encoding="utf-8") as f:
            lim = json.load(f)
        lim = lim.get("limits_deg", lim)
        JOk = list(rows[0]["targets_deg_joint"].keys())
        tj = np.array([[d["targets_deg_joint"][j] for j in JOk] for d in rows])
        ex, tt = {}, []
        for i, j in enumerate(JOk):
            e = np.maximum(lim[j]["soft_min"] - tj[:, i], tj[:, i] - lim[j]["soft_max"])
            if (e > 0).any():
                ex[j] = (int((e > 0).sum()), float(e.max()))
                tt += [k for k in np.where(e > 0)[0]]
                S.setdefault("exceed_t_by_joint", {})[j] = (round(float(t[np.where(e > 0)[0][0]] - t[0]), 2), round(float(t[np.where(e > 0)[0][-1]] - t[0]), 2))
        S["exceed"] = ex
        if tt:
            S["exceed_t"] = (float(t[min(tt)] - t[0]), float(t[max(tt)] - t[0]))
            S["exceed_modes"] = sorted({int(rows[k]["c5"]["mode"]) for k in tt if rows[k].get("c5")})
        # 관절·방향(경계)별: 전 구간과 지면 구간(ground_until 이전)
        tt_rel = t - t[0]
        gmask = np.ones(len(rows), bool) if ground_until is None else (tt_rel < float(ground_until))
        sides = []
        for i, j in enumerate(JOk):
            for nm_, e in (("soft_min", lim[j]["soft_min"] - tj[:, i]), ("soft_max", tj[:, i] - lim[j]["soft_max"])):
                m_ = e > 0
                if not m_.any():
                    continue
                mg = m_ & gmask
                sides.append({"joint": j, "side": nm_, "bound": float(lim[j][nm_]), "n": int(m_.sum()), "max": float(e[m_].max()), "n_ground": int(mg.sum()),
                              "max_ground": float(e[mg].max()) if mg.any() else None,
                              "t_ground": (round(float(tt_rel[mg][0]), 2), round(float(tt_rel[mg][-1]), 2)) if mg.any() else None})
        S["exceed_sides"], S["ground_until"] = sides, ground_until
        # 좌우 soft 범위(|min|·|max| 쌍 — 부호 규약과 무관한 비교)가 다른 관절
        asym_all = {}
        for base in ("hip_f", "hip_a", "hip_r", "knee", "ankle_f", "ankle_r"):
            L_, R_ = lim.get(f"left_{base}_joint"), lim.get(f"right_{base}_joint")
            if L_ and R_:
                a_, b_ = sorted([abs(L_["soft_min"]), abs(L_["soft_max"])]), sorted([abs(R_["soft_min"]), abs(R_["soft_max"])])
                d_ = [(x, y) for x, y in zip(a_, b_) if abs(x - y) > 1e-9]
                if d_:
                    asym_all[base] = d_
        S["soft_asym_all"] = asym_all
        cc = rows[-1].get("soft_clamp_counts") or []
        S["clamp_by_joint"] = {j: int(c) for j, c in zip(JOk, cc) if c}
        # 브리지가 읽는 stage8 누계는 한 틱 늦다(틱 k 의 명령은 틱 k+1 의 상태에 반영) → 마지막 틱 몫을 뺀 초과 틱 수와 같아야 한다
        ex_wo_last = {}
        for i, j in enumerate(JOk):
            e = np.maximum(lim[j]["soft_min"] - tj[:-1, i], tj[:-1, i] - lim[j]["soft_max"])
            if (e > 0).any():
                ex_wo_last[j] = int((e > 0).sum())
        S["clamp_matches_exceed"] = (ex_wo_last == S["clamp_by_joint"])
        S["clamp_total"] = rows[-1].get("soft_clamp_total")
        S["reject_reasons"] = sorted({str(d.get("last_reject_reason")) for d in rows} - {"None"})
        S["soft_asym"] = {p: (lim[f"left_{p}_joint"]["soft_max"], -lim[f"right_{p}_joint"]["soft_min"]) for p in ("hip_f", "ankle_f")}
    except Exception:  # noqa: BLE001
        pass
    c5 = [d.get("c5") for d in rows if d.get("c5")]
    if c5:
        S["mode_ticks"] = int(sum(int(c["mode"]) for c in c5))
        S["n_on"] = int(c5[-1]["n_on"])
    return S


# ---------- 표 ----------
class Rows:
    def __init__(self):
        self.rows = []

    def add(self, cat, item, train, head, jet, verdict, note="", cover="", known="", ev=""):
        self.rows.append(dict(cat=cat, item=item, train=str(train), head=str(head), jet=str(jet), verdict=verdict, note=note,
                              cover=cover, known=known, ev=ev))

    def count(self):
        c = {OK: 0, BAD: 0, UNK: 0}
        for r in self.rows:
            c[r["verdict"]] += 1
        return c


def verdict_eq(a, b, tol=0.0):
    try:
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            a, b = list(a), list(b)
            return OK if len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b)) else BAD
        return OK if abs(float(a) - float(b)) <= tol else BAD
    except (TypeError, ValueError):
        return OK if a == b else BAD


def same_or(vals, fmt=fnum):
    """여러 run 의 값이 같으면 하나로, 다르면 'a / b' 로."""
    keys = list(vals)
    if all(vals[k] == vals[keys[0]] for k in keys):
        return fmt(vals[keys[0]])
    return " / ".join(f"{k}: {fmt(vals[k])}" for k in keys)


def jet_cell(Hv, Jv, fmt=fnum, tol=0.0):
    """젯슨 판 값 칸: 값 + HEAD 와의 일치 표시."""
    if Jv is None:
        return "—"
    try:
        same = verdict_eq(Hv, Jv, tol) == OK
    except Exception:  # noqa: BLE001
        same = (Hv == Jv)
    return f"{fmt(Jv)}" + ("" if same else " ⚠HEAD 와 다름")


def build_rows(key, ctx):
    """한 정책 키의 계약 표."""
    import numpy as np
    H, J = ctx["H"], ctx["J"]
    regH = H["registry"]
    K = regH["keys"].get(key, {})
    R = Rows()
    if not K.get("registered"):
        err = (regH.get("reg_err") or {}).get("kew3") if "kew3" in key else None
        R.add("등록", "브리지 키", "—", "미등록" + (f" — 등록 거부: {err}" if err else ""), "—", BAD if err else UNK,
              "브리지가 하위 키 계약 불일치로 합성 키 등록을 거부했다" if err else "배포 트리의 브리지에 키가 없다")
        return R
    PH = H["probe"].get(key, {})
    kj = (J or {}).get("registry", {}).get("keys", {}).get(key, {}) if J else {}
    in_jet = bool(kj.get("registered"))
    PJ = (J or {}).get("probe", {}).get(key, {}) if in_jet else {}
    JO = regH["JOINT_ORDER"]
    T = ctx["trains"]
    # 구성 요소(정책 파일) — 역할: stand / rec / walk
    comps = []                       # (역할, Train, 하위 키)
    if K.get("kew3"):
        ke, w3k = K["kew3"]["ke"], K["kew3"]["w3"]
        kke = regH["keys"][ke]
        comps = [("서기", T[os.path.dirname(kke["ckpt_rel"])], ke), ("회복", T[os.path.dirname(kke["c5_pair"][1])], ke),
                 ("걷기", T[os.path.dirname(regH["keys"][w3k]["ckpt_rel"])], w3k)]
    elif K.get("c5_pair"):
        comps = [("서기", T[os.path.dirname(K["ckpt_rel"])], key), ("회복", T[os.path.dirname(K["c5_pair"][1])], key)]
    elif K.get("w3cfg"):
        comps = [("걷기", T[os.path.dirname(K["ckpt_rel"])], key)]
    else:
        comps = [("서기", T[os.path.dirname(K["ckpt_rel"])], key)]
    names = "·".join(f"{r} {t.tag}" for r, t, _ in comps)
    is_kew3 = bool(K.get("kew3"))
    is_c5 = bool(K.get("c5_pair")) or is_kew3
    is_w3 = bool(K.get("w3cfg"))
    att = PH.get("attrs") or PH.get("attrs_ke") or {}
    attJ = PJ.get("attrs") or {}
    jmark = "—(젯슨 미배포 키)" if not in_jet else None
    F = ctx["facts"]
    net = ctx["net"]

    def J_(v, fmt=fnum, ref=None, tol=0.0):
        if not in_jet:
            return jmark
        return jet_cell(ref, v, fmt, tol) if ref is not None else fmt(v)

    if "error" in PH:
        R.add("프로브", "러너 생성·프로브", "—", "오류: " + PH["error"], "—", UNK, PH.get("trace", "")[-300:])
        return R

    # ------------------------------------------------------------------ 0. 정책 파일(네트워크)
    for role, tr, _ in comps:
        nv = net.get(tr.run, {})
        pt_ok, onnx_ok = nv.get("pt_equal"), nv.get("onnx_equal")
        v = OK if (pt_ok and onnx_ok) else (UNK if (pt_ok is None or onnx_ok is None) else BAD)
        man = nv.get("manifest_md5") or {}
        jtxt = jmark or ("; ".join(f"{k} {'=' if ok else '≠'} 매니페스트" for k, ok in man.items()) if man else "매니페스트에 없음")
        if in_jet and man and not all(man.values()):
            v = BAD
        R.add("0 정책 파일", f"{role} 정책 가중치 = 학습 체크포인트", f"{tr.name}/{nv.get('ckpt', '?')} actor {nv.get('n_tensors', '?')} 텐서",
              f"policy.pt {nv.get('pt_match', '?')} 비트 동일 · policy.onnx {nv.get('onnx_match', '?')} 비트 동일(그래프 {nv.get('onnx_ops', '?')})",
              jtxt, v, f"SOURCE_CHECKPOINT = {tr.src_ckpt}; 순전파 pt vs 체크포인트 MLP 최대 |Δ| {fnum(nv.get('fwd_err'), 2)}(출력 최대 |값| {fnum(nv.get('fwd_mag'), 3)} — float32 연산 순서 차)",
              cover="없음(W3S3d 만 selftest_w3s3d_onnx — onnx vs pt 출력)")
        ag = tr.agent.get("policy", {})
        nrm = ag.get("actor_obs_normalization")
        R.add("0 정책 파일", f"{role} 관측 정규화·행동 클립", f"actor_obs_normalization {nrm} · clip_actions {tr.agent.get('clip_actions')} · actions.clip {tr.act.get('clip')}",
              "정규화 없음(내보낸 그래프에 정규화 파라미터 0) · 러너 행동 클립 없음", J_("같은 파일"),
              OK if (nrm in (False, None) and tr.agent.get("clip_actions") is None and tr.act.get("clip") is None and nv.get("pt_extra_params", 0) == 0) else BAD,
              cover="없음")
    # 학습 시점의 시뮬 소스(계약 함수)가 현재 소스와 같은가 — '학습 값' 칸의 소스 근거가 유효한지
    for role, tr, _ in comps:
        sc = (ctx.get("srcchk") or {}).get(tr.run) or {}
        if not sc:
            continue
        if sc.get("error"):
            R.add("0 정책 파일", f"{role} 학습 시점의 시뮬 소스(계약 함수)", f"run 시작 {tr.start}", "비교 실패: " + str(sc["error"])[:200], J_("(학습 쪽 항목)"), UNK,
                  "학습 값 칸의 소스 근거는 현재 저장소 소스다 — 학습 때와 같은지 확인하지 못했다", cover="없음")
            continue
        ch = sc.get("changed") or []
        unrev = [c for c in ch if not c["review"]]
        gs = sc.get("git_state") or {}
        bad_src = bool(sc.get("dirty_rl") or sc.get("untracked_rl") or unrev or sc.get("missing") or sc.get("facts_diff") or sc.get("cond_fail") or not gs)
        R.add("0 정책 파일", f"{role} 학습 시점의 시뮬 소스(계약 함수)",
              f"커밋 {sc['commit'][:8]}({sc.get('commit_date')}) = run 시작 {tr.start} 직전의 브랜치 커밋 · run 이 남긴 작업 트리 기록({gs.get('file', '없음')}, 브랜치 {gs.get('branch')}): "
              f"바뀐 추적 파일 {len(gs.get('dirty', []))} 개 중 rl_walking/*.py {len(sc.get('dirty_rl') or [])} 개 · 미추적 rl_walking {len(sc.get('untracked_rl') or [])} 개",
              f"현재 소스와 비교: 관측 항 함수·행동 항 클래스·한계/위상 이벤트·온셋 식과 그 보조 정의 {sc.get('n')} 개 중 {sc.get('same')} 개 같음"
              + (" · 바뀐 것: " + "; ".join(f"{c['name']}(+{c['add']}/−{c['rem']} 줄 — {c['review'] or '미검토'})" for c in ch) if ch else "")
              + (f" · 학습 시점에 없던 것 {sc['missing']}" if sc.get("missing") else "")
              + f" · 감사가 읽는 규칙 {sc.get('facts_n')} 개는 두 판에서 " + ("같음" if not sc.get("facts_diff") else f"다름 {sc['facts_diff']}"),
              J_("(학습 쪽 항목)"), UNK if bad_src else OK,
              "필터·클록·온셋의 '시뮬 소스 실행 대조'는 이 커밋의 소스로 했다. 커밋은 run 폴더 이름의 시각으로 추정한 것이다(rsl_rl 은 git status·diff 만 남기고 커밋 해시는 남기지 않는다)."
              + (f" 바뀐 함수가 종전 동작을 유지하는 조건: 지연 가중 없음 {sc['conds'].get('delay')} · 위상 누적기 이벤트(gait_phase_update·w3_gait_update) 없음 {sc['conds'].get('phase')}"
                 + (f" — **조건 미충족 {sc['cond_fail']}**" if sc.get("cond_fail") else "") if ch else ""),
              cover="없음")

    # 평가에 쓴 체크포인트 이름 = 내보낸 것
    ev_ck = ctx["eval_ckpt"]
    for role, tr, _ in comps:
        want = ev_ck.get(tr.name)
        if want:
            R.add("0 정책 파일", f"{role} 시뮬 평가 체크포인트 = 배포 파일", f"평가 인자 {want}", f"SOURCE_CHECKPOINT {tr.src_ckpt}", J_("같은 파일"),
                  OK if want == tr.src_ckpt else BAD, cover="없음")

    # ------------------------------------------------------------------ 1. 관측
    tr0 = comps[0][1]
    lay0 = tr0.layout()
    same_obs = all(t.obs_sig() == tr0.obs_sig() for _, t, _ in comps)
    if len(comps) > 1:
        R.add("1 관측", "구성 정책들의 학습 관측 설정이 서로 같은가", names, "하나의 관측 조립을 공유(C5Runner 코어 / 합성 러너 이어받기)", J_("같은 구조"),
              OK if same_obs else BAD, "env.yaml observations.policy 전체(함수·인자·잡음·클립·이력) 비교", cover="브리지 주석(수동 확인)")
    dims_ok = all(d is not None for _, d, _ in lay0)
    tot = sum((d or 0) * h for _, d, h in lay0)
    net_in = net.get(tr0.run, {}).get("in_dim")
    terms_dep = [(n, d, h) for n, d, h in att.get("terms", [])]
    R.add("1 관측", "항 목록·순서·프레임 차원", " ".join(f"{n}({d})" for n, d, _ in lay0), " ".join(f"{n}({d})" for n, d, _ in terms_dep),
          J_(" ".join(f"{n}({d})" for n, d, _ in attJ.get("terms", [])) if in_jet else None),
          OK if dims_ok and [(n, d) for n, d, _ in lay0] == [(n, d) for n, d, _ in terms_dep] else BAD,
          f"학습 합 {tot} = 체크포인트 actor 입력 {net_in} = 러너 obs_dim {att.get('obs_dim')}"
          + (f"; 시뮬 기록 obs_terms 순서 {'일치' if ctx['simrec'].get('terms_match') else '불일치'}" if ctx["simrec"].get("ok") else ""),
          cover="selftest_w3_rollout_parity [P1](W3S3d) · 러너 --selftest 골든(245)")
    R.add("1 관측", "이력 길이(항별 프레임 수)", same_or({t.tag: sorted({h for _, _, h in t.layout()}) for _, t, _ in comps}, str),
          str(sorted({h for _, _, h in terms_dep})), J_(str(sorted({h for _, _, h in attJ.get("terms", [])})) if in_jet else None),
          OK if [h for _, _, h in lay0] == [h for _, _, h in terms_dep] else BAD, cover="[P1]")
    seq = ctx["meas"].get(key) or {}
    seq_ke = seq
    if is_kew3:                         # 합성 키의 러너 프로브는 하위 키 것을 쓴다
        seq = ctx["meas"].get(K["kew3"]["ke"]) or {}
        seq_ke = seq
    sr = ctx["simrec"]
    R.add("1 관측", "이력 배열 순서·리셋 첫 틱", "오래된 것 먼저, 첫 틱 전 칸 백필" + (f"(시뮬 기록 실측: 순서 {sr.get('oldest_first')}·백필 {sr.get('backfill')})" if sr.get("ok") else "(Isaac CircularBuffer — 시뮬 기록 없음)"),
          f"최신 = 마지막 칸 {seq.get('hist_newest_last')} · 첫 틱 백필 {seq.get('hist_backfill')}(프로브)",
          J_("프로브 비트 동일" if ctx["bit"].get(key, {}).get("obs") else "프로브 다름" if in_jet else None),
          OK if (seq.get("hist_newest_last") and seq.get("hist_backfill") and (not sr.get("ok") or (sr.get("oldest_first") and sr.get("backfill")))) else BAD,
          cover="[P1]·러너 골든")
    R.add("1 관측", "스케일", "없음(전 항 scale: null)" if all(v.get("scale") is None for _, t, _ in comps for _, v in t.terms) else "항별 스케일 있음",
          f"스케일 없음(프로브: 관측 = 입력 {seq.get('scale_passthrough')})", J_("프로브 비트 동일" if ctx["bit"].get(key, {}).get("obs") else None),
          OK if seq.get("scale_passthrough") and all(v.get("scale") is None for _, v in tr0.terms) else BAD, cover="[P1]")
    clips = {n: v.get("clip") for n, v in tr0.terms if v.get("clip") is not None}
    cp = seq.get("clip_probe", {})
    clipped = [n for n, c in clips.items() if n in cp and cp[n] <= float(c[1]) + 1e-6]
    rl = ctx.get("reallog") or {}
    rls = [v for v in (ctx.get("reallogs") or {}).values() if v]
    mx = {}
    for v in rls:
        for n_, x_ in (v.get("obs_absmax") or {}).items():
            mx[n_] = max(mx.get(n_, 0.0), x_)
    raw_mx = max((v.get("raw_absmax", 0.0) for v in rls), default=None)
    rl_desc = "·".join(f"{os.path.basename(os.path.dirname(rel)).replace('real_', '')} {v['key'].replace('stand_', '')} {v['dur_s']:.0f} s" for rel, v in (ctx.get("reallogs") or {}).items() if v)
    R.add("1 관측", "클립", "; ".join(f"{n} ±{fnum(c[1])}" for n, c in clips.items()) or "없음",
          "클립 없음(프로브: " + ", ".join(f"{n} {fnum(v)}" for n, v in cp.items()) + " 그대로 통과)",
          J_("같음(프로브 비트 동일)" if ctx["bit"].get(key, {}).get("obs") else None),
          OK if (not clips or len(clipped) == len([n for n in clips if n in cp])) else BAD,
          f"학습과 같은 쪽은 시뮬(평가)뿐이다. 값이 범위 안이면 두 쪽이 같다. 실물 로그({rl_desc}) 최대: " + ", ".join(f"{n} {fnum(v, 3)}" for n, v in mx.items())
          + f", 원시 행동 {fnum(raw_mx, 3)} — "
          + ("전부 범위 안이라 이 기록들에서는 차이가 0 이다" if not [n for n, c in clips.items() if mx.get(n, 0.0) > float(c[1])]
             else f"**범위를 넘은 항 {[n for n, c in clips.items() if mx.get(n, 0.0) > float(c[1])]} — 이 기록에서 이미 값이 달랐다**")
          + ". 범위 밖(낙상·폭주·센서 튐)에서는 시뮬만 자른다"
          + ": 시뮬 X2 폭주(원시 행동 2 000 까지)에서 시뮬 관측은 ±10 에 묶였고(TRANSITION_W3KE §5.6) 배포는 묶이지 않는다 → 폭주 양상이 시뮬과 달라질 수 있다(추측)" if mx else "값이 범위 안이면 동일",
          cover="없음", known="신규(이 감사 — 정상 범위에서는 값이 같다)",
          ev=yaml_line(tr0.env_path, ["observations", "policy", "base_ang_vel", "clip"]) + " 외 5 항"
          + (f" · Isaac Lab {F.get('isaac_obs_line')}(잡음 → 클립 → 스케일 → 이력 순서 {'확인' if F.get('isaac_obs_order') else '확인 실패'}"
             + "; 판: 학습 기록 " + same_or({t.tag: ((t.isaac_state or {}).get("head"), len((t.isaac_state or {}).get("dirty") or [])) for _, t, _ in comps},
                                         lambda v: f"'{v[0]}'·바뀐 파일 {v[1]} 개")
             + f" / 현재 {F.get('isaac_now')}·이 파일 변경 {'없음' if F.get('isaac_obs_clean') else '있음 또는 미확인'})" if F.get("isaac_obs_order") is not None else "")
          + " ↔ " + line_of("rl_walking/deploy/policy_runner_stage4.py", r"^        frames = \{$") + "(조립부에 클립 없음)")
    t = tr0.term("base_ang_vel")
    R.add("1 관측", "base_ang_vel(자이로) 처리", f"{short(t.get('func'))}(몸통 각속도 rad/s, 지연 0–{(t.get('params') or {}).get('max_lag', 0)} 틱 DR) + 잡음",
          "IMU gyro_rad_s 그대로(프로브 |Δ| " + fnum(seq.get("err", {}).get("base_ang_vel"), 2) + ")",
          J_("비트 동일" if ctx["bit"].get(key, {}).get("obs") else None), OK if seq.get("err", {}).get("base_ang_vel", 1) <= TOL_OBS else BAD,
          cover="[P1]")
    R.add("1 관측", "IMU 축 정렬(몸통 프레임 +x 앞·+z 위)", "시뮬 몸통(base) 프레임",
          "IMU 어댑터가 낸 gyro·quat 을 그대로 사용 — 축 부호는 현장 확인 기록(0912–0914)에 기댄다", J_("같음"), UNK, "오프라인으로 알 수 없다(현장 확인 사항)", cover="없음(PARITY #2)")
    ge = ctx["grav"]
    cb = (tr0.ev.get("calib_bias") or {}).get("params", {})
    R.add("1 관측", "projected_gravity 정의", "쿼터니언으로 (0,0,−1) 을 몸통 프레임에(단위벡터) + 장착 편향 DR 피치 " + fnum(cb.get("imu_tilt_rad")) + "·롤 " + fnum(cb.get("imu_roll_rad")) + " rad",
          f"projected_gravity(q): 독립 회전행렬 식과 최대 |Δ| {fnum(ge.get('H_pg'), 2)} · 맵 편향 보정 {regH['map'].get('imu_mount_bias_deg')}° 식 오차 {fnum(ge.get('H_cg'), 2)}°",
          J_(f"|Δ| {fnum(ge.get('J_pg'), 2)} · 보정 {(J or {}).get('registry', {}).get('map', {}).get('imu_mount_bias_deg')}°" if in_jet else None),
          OK if ge.get("H_pg", 1) < 1e-9 and ge.get("H_cg", 1) < 1e-6 and (not in_jet or ge.get("J_pg", 1) < 1e-9) else BAD,
          "보정 뒤 남는 실제 장착 오차가 학습 대역(±1.5°) 안인지는 현장 확인 사항", cover="[P5]·브리지 selftest 'IMU 장착 편향'")
    # 명령 관측
    for role, tr, sub in comps:
        cr = tr.cmd.get("ranges", {})
        g3, c3 = tr.w3()
        if role == "걷기":
            lim = regH["keys"][sub].get("w3_limits") or {}
            cd = F.get("w3_cmd_defaults", {})
            tr_rng = {"vx 앞": tuple(c3.get("vx_range", cd.get("vx_range"))), "vx 뒤": tuple(c3.get("vx_back_range") or c3.get("vx_range", cd.get("vx_range"))),
                      "|vy|": tuple(c3.get("vy_range", cd.get("vy_range"))), "|wz|": tuple(c3.get("wz_range", cd.get("wz_range")))}
            cur = ctx["curric"].get(tr.run, {})
            dep_rng = {"vx 앞": tuple(lim.get("vx_fwd", ())), "vx 뒤": tuple(lim.get("vx_back", ())), "|vy|": tuple(lim.get("vy", ())), "|wz|": tuple(lim.get("wz", ()))}
            reach = (f"; 학습 로그 도달값 vy_max {fnum(cur.get('vy_max'))}·wz_max {fnum(cur.get('wz_max'))}·보조 α {fnum(cur.get('alpha'))}" if cur else "; 학습 로그 없음")
            okr = all(verdict_eq(tr_rng[k], dep_rng[k], 1e-9) == OK for k in tr_rng)
            if cur:
                okr = okr and abs(cur.get("vy_max", -1) - dep_rng["|vy|"][1]) < 1e-6 and abs(cur.get("wz_max", -1) - dep_rng["|wz|"][1]) < 1e-6
            R.add("1 관측", "명령 관측 범위(걷기)", f"stage {c3.get('stage')} 단일 축: " + ", ".join(f"{k} {fnum(v)}" for k, v in tr_rng.items()) + reach,
                  "W3CmdLimits " + ", ".join(f"{k} {fnum(v)}" for k, v in dep_rng.items()) + " · 범위 밖·두 축 동시 거부(validate_w3_cmd)",
                  jmark or "—", OK if okr else BAD, "걷기 버튼 표(W3_MOVES) " + ("전부 범위 안" if ctx["moves_ok"] else "범위 밖 항목 있음"),
                  cover="브리지 selftest 'W3S3d 명령 채널' ①②")
            hl = regH["keys"][sub].get("w3_hl")
            nd = (H["node"].get(key) or {})
            hl_obs = nd.get("cmd_obs_hl_on")
            R.add("1 관측", "명령 관측: 헤딩 고리(외부 P 고리)", f"학습에 없음(p_hloop {c3.get('p_hloop')}, heading_command {tr.cmd.get('heading_command')}) — 걷기 명령은 한 축만",
                  (f"기본 켜짐(K {hl['kp']}·±{hl['wz_max']} rad/s): 앞·옆 명령에 wz 를 **동시에** 넣는다" if hl else "꺼짐")
                  + (f" — 노드 세션 측정 관측 명령 ({', '.join(fnum(v, 3) for v in hl_obs)})" if hl_obs else ""),
                  jmark or "—", BAD if hl else OK,
                  "학습과 같은 쪽이 없다(학습 분포 밖: 두 축 동시·|wz| < 0.1). 시뮬 평가(eval_w3 --heading_loop 0.5 --hl_wzmax 0.4)는 같은 식으로 있음 — PARITY_AUDIT_1007 §4.5(오차 15° 문턱이 정상 걸음에서 30–41 % env 를 멈춘다, 결정 D7)"
                  + ("; 합성 러너 + 고리의 폐루프 평가는 없음(X6·X7 은 고리 끔 — KEW3_DESIGN §8.3)" if is_kew3 else ""),
                  cover="PARITY #17·#18(문서)·selftest_w3_equiv [F](식)", known="기지(PARITY #17–18, D7)",
                  ev=yaml_line(tr.env_path, ["events", "w3_cmd", "params", "p_hloop"]) + " ↔ " + line_of("rl_walking/deploy/w3_deploy.py", r"^W3_HL_DEFAULT = ") + "·" + line_of("jetson/scripts/rl_bridge_node.py", r"def _w3_cmd_step"))
            rs = tr.cmd.get("resampling_time_range")
            R.add("1 관측", "명령 변경 간격", f"리샘플 {fnum(rs)} s(STEP→STEP 위상 유지)", f"최소 체류 {lim.get('min_dwell_s')} s(위상 유지)", jmark or "—",
                  BAD if lim.get("min_dwell_s", 99) < float(rs[0]) else OK, "학습 범위는 8 s 이상 — 배포는 그보다 잦은 변경을 허용한다. 시뮬 1 s 순환 평가에서 낙상 증가 신호 없음(PARITY §4.6)",
                  cover="PARITY #19", known="기지(PARITY #19)",
                  ev=yaml_line(tr.env_path, ["commands", "base_velocity", "resampling_time_range"]) + " ↔ " + line_of("rl_walking/deploy/w3_deploy.py", r"min_dwell_s: float = "))
        else:
            st = float(tr.cmd.get("rel_standing_envs", 0.0))
            vx = tuple(cr.get("lin_vel_x", (0, 0)))
            only0 = (st >= 1.0) or (vx == (0.0, 0.0) and tuple(cr.get("lin_vel_y", (0, 0))) == (0.0, 0.0))
            tr_txt = "(0,0,0) 고정" if only0 else f"서기 {st:.0%}: (0,0,0) / 그 밖 vx U{fnum(vx)}, vy {fnum(cr.get('lin_vel_y'))}, wz {fnum(cr.get('ang_vel_z'))}"
            if role == "회복":
                cue = PH.get("cue") or {}
                ev = ctx["eval_ke"].get("kr") or {}
                dep = f"회복 정책 입력만 cue vx {cue.get('cue_vx')} 를 {cue.get('cue_s')} s(켜짐 틱 필터 ω_y>0), 그 밖 0"
                inr = (cue.get("cue_vx") is None) or (vx[0] - 1e-9 <= float(cue["cue_vx"]) <= vx[1] + 1e-9) or float(cue["cue_vx"]) == 0.0
                same_eval = (str(ev.get("--sw_rec_vx")) == str(cue.get("cue_vx")) or abs(float(ev.get("--sw_rec_vx", "nan")) - float(cue.get("cue_vx") or 0)) < 1e-9) if ev else None
                R.add("1 관측", "명령 관측(회복 정책)", tr_txt, dep, J_((PJ.get("cue") or {}).get("cue_vx"), ref=cue.get("cue_vx")) if in_jet else jmark,
                      OK if inr else BAD, f"시뮬 평가와는 같고 학습 범위와는 다르다: cue 값이 학습 명령 범위 하한보다 작다(보상 모드 문턱 walk_min 0.03 과 같음). 시뮬 평가 인자 --sw_rec_vx {ev.get('--sw_rec_vx') if ev else '?'} 와는 "
                      + ("같다" if same_eval else "다르다" if same_eval is False else "대조 불가") + " — 의도된 선택(reports/gen_c5/act1/PLAN.md 'am: 최소 진폭')",
                      cover="브리지 selftest 'C5 kr 경로'(값 전달)", known="기지(의도·시뮬 평가 동일)",
                      ev=yaml_line(tr.env_path, ["commands", "base_velocity", "ranges", "lin_vel_x"]) + " ↔ " + line_of("jetson/scripts/rl_bridge_node.py", r"^C5_EXTRA = "))
            else:
                nd = (H["node"].get(key) or {})
                nm = ctx["nodemeas"].get(key) or {}
                dep = "stand 가족: 명령 (0,0,0)·heading_hold 강제 끔" + (f"(노드 세션 관측 명령 최대 |값| {fnum(nm.get('cmd_obs_absmax'), 2)})" if nm and not (is_w3 or is_kew3) else "")
                if is_kew3:
                    dep = f"KE 모드: 외부 명령과 무관하게 0(합성 러너 프로브 관측 명령 {PH.get('kew3', {}).get('ke_mode_cmd_obs')})"
                okc = only0 and ((nm.get("cmd_obs_absmax", 0.0) == 0.0) if (nm and not (is_w3 or is_kew3)) else True)
                if is_kew3:
                    okc = only0 and all(abs(v) == 0.0 for v in PH.get("kew3", {}).get("ke_mode_cmd_obs", [1]))
                R.add("1 관측", "명령 관측(서기 정책)", tr_txt, dep, J_("같음(노드 발행 비트 동일)" if ctx["bit"].get(key, {}).get("node") else "확인 못 함") if in_jet else jmark,
                      OK if okc else BAD, cover="브리지 selftest W3('명령 관측 0')·없음(G21a)")
    # joint_pos
    dp_tr = {t.tag: t.default_pose(JO) for _, t, _ in comps}
    dp0 = dp_tr[tr0.tag]
    dp_dep = att.get("default_pose", [])
    dp_obs = seq.get("default_from_obs")
    e1 = max(abs(a - b) for a, b in zip(dp0, dp_dep)) if dp_dep and None not in dp0 else float("nan")
    e2 = float(np.abs(np.asarray(dp_obs) - np.asarray(dp0)).max()) if dp_obs is not None and None not in dp0 else float("nan")
    same_dp = all(v == dp0 for v in dp_tr.values())
    R.add("1 관측", "joint_pos = 엔코더각 − 기본자세", f"{short(tr0.term('joint_pos').get('func'))}; 기본자세(init_state, deg) " + " ".join(f"{math.degrees(v):+.2f}" for v in dp0)
          + ("" if same_dp else " ⚠구성 정책 간 다름"),
          f"러너 default_pose 와 최대 |Δ| {fnum(e1, 2)} rad · 관측에서 역산한 기본자세와 {fnum(e2, 2)} rad",
          J_(f"최대 |Δ| {fnum(max(abs(a - b) for a, b in zip(dp0, attJ.get('default_pose', dp0))), 2)} rad" if in_jet else None),
          OK if same_dp and e1 < 1e-6 and e2 < 2e-6 else BAD, "학습 env 별 물리 기본자세 편향(default_pose_bias)은 정책 프레임에서 0 으로 보인다 — 배포 기준은 명목값",
          cover="브리지 selftest '학습 계약'(G21a·C5 키)·'W3 실험 플래그'(W3)")
    jp_f = short(tr0.term("joint_pos").get("func"))
    R.add("1 관측", "엔코더 관측 방식", f"{jp_f}·{short(tr0.term('joint_vel').get('func'))}: 로터(모터)측 각 — 유격 안 링크 이동·직렬 처짐은 관측에 안 보임",
          "모터 엔코더 피드백(로터측)을 관절 프레임으로 변환(DIR × (모터 deg − 영점))해 그대로 사용", J_("같음"),
          OK if jp_f.endswith("_v2") else BAD, "하드웨어 사실(모터 내장 엔코더 = 로터측)에 근거한 일치 — 영점·방향값의 옳고 그름은 현장 확인 사항(boot_pose_check)",
          cover="브리지 selftest '왕복 항등'·'수치 앵커'(변환식)")
    # joint_vel
    jt = tr0.term("joint_vel")
    bj_tr = (jt.get("params") or {}).get("beta")
    sx = ctx["simexec"].get(tr0.run) or {}
    R.add("1 관측", "joint_vel = 엔코더 유한차분 속도의 EMA", f"{short(jt.get('func'))} β {bj_tr}(에피소드 첫 틱 = 원값)",
          f"프로브 역산 β {fnum(seq.get('beta_jv'), 5)} · 시뮬 소스 실행 대조 최대 |Δ| {fnum(max(sx.get(1, {}).get('joint_vel', [float('nan')])), 2)}",
          J_(attJ.get("jv_beta"), ref=att.get("jv_beta")), OK if (bj_tr is not None and abs(seq.get("beta_jv", 9) - bj_tr) < 1e-4 and max(sx.get(1, {}).get("joint_vel", [1])) < TOL_OBS) else BAD,
          cover="브리지 selftest '관측 필터 계약'(값)·[P1]")
    nm0 = ctx["nodemeas"].get(key) or {}
    R.add("1 관측", "관절속도 원천(유한차분 식)", "Δθ_enc / dt (틱마다, 리셋 틱 0)" + ("" if F.get("jv_fd_dt") else " ⚠소스 확인 실패"),
          f"브리지 _tick: Δq / (dt × 머문 틱), 같은 값 반복 틱은 0(노드 세션 재구성 오차 {fnum(nm0.get('jv_err_bridge_law'), 2)}; 단순 차분으로 재구성하면 {fnum(nm0.get('jv_err_simple_fd'), 2)})",
          J_("같은 식(노드 발행 비트 동일)" if ctx["bit"].get(key, {}).get("node") else "확인 못 함") if in_jet else jmark,
          (BAD if (nm0.get("jv_err_bridge_law", 9) < 1e-4 and nm0.get("jv_err_simple_fd", 0) > 1e-3)      # 브리지 식으로만 재구성된다 = 학습 식과 다르다
           else OK if nm0.get("jv_err_simple_fd", 9) < 1e-4                                              # 단순 차분으로 재구성된다 = 학습 식
           else UNK),                                                                                    # 어느 식으로도 재구성되지 않는다(다른 결함) → 판정 보류
          "피드백 값이 N 틱 머문 뒤 바뀔 때만 다르다: 브리지는 0·…·0·Δq/(N·dt), 학습 식에 같은 신호를 넣으면 0·…·0·Δq/dt. 시뮬 엔코더는 매 틱 갱신이라 머무는 표본이 없다(양자화 미모형)."
          + (f" 실물 1007 로그: 직전과 같은 값 — 정지 관절 {rl.get('repeat_still', 0):.0%}(엔코더 양자 {fnum(rl.get('quantum_rad'), 2)} rad), 움직이는 관절(> 0.2 rad/s) {rl.get('repeat_moving', 0):.1%}; "
             f"기록 qvel 은 브리지 식과 {fnum(rl.get('qvel_err_bridge_law'), 2)}, 단순 차분과 최대 {fnum(rl.get('qvel_err_simple_fd'), 2)} rad/s 차." if rl else "")
          + " 시뮬 폐루프 영향 신호 없음(PARITY §4.3)", cover="PARITY #6·[P3]", known="기지(PARITY #6)",
          ev=line_of("rl_walking/mdp_hang.py", r"env\._enc_vel = \(full - prev\) / dt") + " ↔ " + line_of("jetson/scripts/rl_bridge_node.py", r"qvel\.append\(\(a - b\) / \(TICK_SEC \* self\.qpos_hold_ticks"))
    # actions (직전 행동)
    at = tr0.term("actions")
    lr = tr0.act.get("lpf_beta_range")
    b_act = seq.get("beta_act")
    lpf_b = regH["STAND_LPF_BETA"]
    in_rng = lr is not None and float(lr[0]) - 1e-9 <= lpf_b <= float(lr[1]) + 1e-9
    R.add("1 관측", "actions(직전 행동) 정의·필터", f"{short(at.get('func'))}: 정책 원시 출력(스케일·지연·슬루·LPF 전)의 EMA, β = 그 env 의 액션 LPF β ∈ U{fnum(lr)}"
          + ("" if F.get("la_uses_term_beta") else " ⚠소스 확인 실패"),
          f"프로브 역산 β {fnum(b_act, 5)}(= 브리지 액션 LPF β {lpf_b}); 원시 출력 재귀 잔차 {fnum(seq.get('prev_action_resid_raw'), 2)} vs 슬루 뒤 값을 넣었다면 {fnum(seq.get('prev_action_resid_emitted'), 2)}"
          f"; 시뮬 소스 실행 대조 {fnum(max(sx.get(1, {}).get('actions', [float('nan')])), 2)}",
          J_(attJ.get("act_beta"), ref=att.get("act_beta")),
          OK if (b_act is not None and abs(b_act - lpf_b) < 1e-4 and in_rng and seq.get("prev_action_resid_raw", 1) < 1e-5 and max(sx.get(1, {}).get("actions", [1])) < TOL_OBS) else BAD,
          "C5/합성: 관측의 직전 행동 = 실제로 실행된 정책의 원시 출력(시뮬 평가 --switch_to 와 같은 규약)", cover="브리지 selftest '관측 필터 계약'·[P1]")
    # gait clock (시간식) — 서기·회복 정책
    for role, tr, sub in comps:
        if role == "걷기":
            continue
        ck = tr.term("gait_clock")
        f_tr = (ck.get("params") or {}).get("freq")
        f_dep = seq_ke.get("clock_freq")
        e_ck = max(sx.get(1, {}).get("gait_clock", [float("nan")]))
        lg = ctx.get("clock_long")
        R.add("1 관측", f"gait_clock({role} 정책)", f"{short(ck.get('func'))}: [sin φ, cos φ], φ = 2π·{f_tr}·(틱·dt), 리셋 = 위상 0",
              f"프로브 위상 기울기 {fnum(f_dep, 6)} Hz · 첫 틱 {fnum(seq_ke.get('clock_t0'))} · 시뮬 소스 실행 대조 {fnum(e_ck, 2)}"
              + (f" · 1 시간 뒤 float32 위상 오차 {fnum(lg, 2)}" if lg is not None else ""),
              J_(kj.get("gait_freq"), ref=K.get("gait_freq")), OK if (f_tr is not None and f_dep is not None and abs(f_dep - f_tr) < 1e-3 and e_ck < 1e-4) else BAD,
              "합성 러너: W3 → KE 인계 때 KE 클록을 위상 0 에서 다시 시작(시뮬 direct 팔의 클록 패치와 같음)" if is_kew3 else "",
              cover="러너 --selftest(식)·실물 로그 재생(이 감사)")
        break
    # feet contact
    fc = tr0.term("feet_contact")
    fp = fc.get("params") or {}
    bn = ((fp.get("sensor_cfg") or {}).get("body_names")) or []
    R.add("1 관측", "feet_contact 순서·값", f"[{', '.join(bn)}] (preserve_order {(fp.get('sensor_cfg') or {}).get('preserve_order')}) 0/1",
          f"contact2 [왼, 오] 0/1 그대로(프로브 |Δ| {fnum(seq.get('err', {}).get('feet_contact'), 2)}); 0/1 이외 값은 러너가 거부",
          J_("비트 동일" if ctx["bit"].get(key, {}).get("obs") else None),
          OK if (len(bn) == 2 and bn[0].startswith("left") and bn[1].startswith("right") and seq.get("err", {}).get("feet_contact", 1) == 0.0) else BAD,
          cover="[P1]·[P6] 좌우 뒤바꿈 변이")
    R.add("1 관측", "접촉 0/1 의 물리 정의", f"{short(fc.get('func'))}: 발 법선력 > U{fnum(fp.get('threshold_range'))} N ∧ 발바닥 기울기 < U{fnum(fp.get('tilt_off_deg'))}° → {fp.get('hold_ticks')} 틱마다 갱신 + {fp.get('extra_delay')} 틱 지연",
          "어댑터: 패드 4 개 중 하나라도 (카운트 − 바이어스) > 200 → 1 (미보정 카운트, 갱신 100 ms). 브리지는 비트를 그대로 쓴다",
          J_("같음"), UNK, "학습 쪽 조건은 실물 패드의 가정 모형이다(평평 조건 θ_off 는 가정값). 패드 비트와 실제 접지 시각의 대조는 현장 확인 필요. 시뮬 폐루프에서는 조건 차이에 둔감(PARITY §4.2)",
          cover="PARITY #9(문서)")
    # S8a EMA
    for nm_, src in (("gyro_ema", "alpha_gyro"), ("joint_vel_ema", "alpha_jv")):
        tt = tr0.term(nm_)
        a_tr = (tt.get("params") or {}).get("alpha")
        late = seq.get("err_late", {}).get(nm_)
        e_sx = sx.get(1, {}).get(nm_, [float("nan"), float("nan")])
        e_sx2 = sx.get(2, {}).get(nm_, [float("nan"), float("nan")])
        R.add("1 관측", f"{nm_} 계수", f"{short(tt.get('func'))} α {a_tr}" + ("(원시 속도에서 — v2)" if nm_ == "joint_vel_ema" else ""),
              f"프로브 역산 α {fnum(seq.get(src), 5)}" + (f" · 모드 {att.get('jvel_ema_mode')}" if nm_ == "joint_vel_ema" else "") + f" · 정상 구간 재구성 오차 {fnum(late, 2)}",
              J_(kj.get("jvel_ema_mode_reg") if nm_ == "joint_vel_ema" else "프로브 비트 동일" if ctx["bit"].get(key, {}).get("obs") else None),
              OK if (a_tr is not None and abs(seq.get(src, 9) - a_tr) < 1e-4 and late is not None and late < TOL_OBS
                     and (nm_ != "joint_vel_ema" or (att.get("jvel_ema_mode") == "v2" and short(tt.get("func")).endswith("_v2")))) else BAD,
              cover="[P1]·브리지 selftest '학습 계약'(EMA 계약 v2)")
        R.add("1 관측", f"{nm_} 시작 규칙", "에피소드 틱 0·1 모두 현재값으로 재시작(_ema_state: episode_length_buf ≤ 1)" + ("" if F.get("ema_fresh_le1") else " ⚠소스 확인 실패"),
              f"틱 0 에서만 재시작(ema_fresh_ticks {att.get('ema_fresh_ticks')}) — 시뮬 소스 실행 대조 시작 구간 최대 |Δ| {fnum(e_sx[0], 3)}, 정상 구간 {fnum(e_sx[1], 2)}"
              + (f"; 옵션 ema_fresh_ticks 2 면 {fnum(e_sx2[0], 2)}" if att.get("has_ema_fresh_opt") else ""),
              J_("틱 0 만(옵션 없음)" if in_jet and not attJ.get("has_ema_fresh_opt") else "틱 0 만") if in_jet else jmark,
              BAD if (e_sx[0] > 1e-4 and e_sx[1] < TOL_OBS) else (OK if e_sx[0] <= 1e-4 else BAD),
              "시뮬 쪽이 학습 계약이다. 정책 시작 뒤 0.9ⁿ 로 사라지는 과도 — 틱 1 의 차 = 0.9·|x₁ − x₀|(왼쪽 수치는 합성 신호 기준). 실물 세션 시작에서의 크기: "
              + "; ".join(f"{os.path.basename(os.path.dirname(rel))} {fnum(v.get('ema_start_gyro' if nm_ == 'gyro_ema' else 'ema_start_jv'), 2)} rad/s" for rel, v in (ctx.get("reallogs") or {}).items() if v)
              + ". 합성 러너 carry 는 ema_fresh_ticks 1 을 요구한다(2 면 기동 거부)",
              cover="PARITY #10·#11·[P1]", known="기지(PARITY #10–11, 옵션 --runner-ema-fresh 2 기본 꺼짐)",
              ev=line_of("rl_walking/mdp_hang.py", r"fresh = env\.episode_length_buf <= 1") + " ↔ " + line_of("rl_walking/deploy/policy_runner_stage4.py", r"if self\.ema_fresh_ticks == 1:"))
    # 전체 재구성(다른 항)
    worst = {n: v for n, v in seq.get("err_deprule", {}).items() if not (is_w3 and not is_kew3 and n == "gait_clock")}
    wn = max(worst, key=worst.get) if worst else None
    R.add("1 관측", "관측 벡터 전체 재구성(600 틱)", "학습 계약으로 다시 조립한 값(항 순서·이력·필터·기본자세)",
          f"실제 러너 관측과 최대 |Δ| {fnum(worst.get(wn) if wn else None, 2)}({wn}); S8a EMA 시작 규칙만 배포식으로 둔 재구성",
          J_("HEAD 와 관측·목표 비트 동일" if (ctx["bit"].get(key, {}).get("obs") and ctx["bit"].get(key, {}).get("out")) else "HEAD 와 다름") if in_jet else jmark,
          OK if worst and max(worst.values()) < TOL_OBS and (not in_jet or (ctx["bit"].get(key, {}).get("obs") and ctx["bit"].get(key, {}).get("out"))) else (UNK if not worst else BAD),
          "러너 프로브: 감지기 켜짐 " + str((PH.get("seq") or {}).get("n_on")) + " 회·회복 모드 " + str((PH.get("seq") or {}).get("mode_ticks")) + " 틱 포함" if is_c5 and not is_kew3 else "",
          cover="[P1]–[P2](W3S3d, 시뮬 기록 원천)")

    # ------------------------------------------------------------------ 2. 행동
    srj = ctx["simrec"].get("joint_names")
    usd_same = all(t.usd() == tr0.usd() for _, t, _ in comps) and (ctx["simrec"].get("usd") in (None, tr0.usd()))
    R.add("2 행동", "관절 순서(12)", ("시뮬 기록 joint_names: " + " ".join(n.replace("_joint", "") for n in srj)) if srj else "env.yaml 에 순서 없음(joint_names ['.*'])",
          "JOINT_ORDER: " + " ".join(n.replace("_joint", "") for n in JO), J_("같음" if (J or {}).get("registry", {}).get("JOINT_ORDER") == JO else "다름") if in_jet else jmark,
          (OK if srj == JO and usd_same else BAD) if srj else UNK,
          "시뮬 기록은 W3S3d 태스크에서 남긴 것 — 세 run 의 로봇 자산(usd_path)이 같아 순서도 같다" if srj and usd_same else ("자산이 다르다" if srj else "시뮬 기록(log_picture/w3parity_rec_*.npz) 없음"),
          cover="sim_validate_runner_stage4.py(Isaac 필요, 단언)·[P1]")
    out_dim = {t.tag: net.get(t.run, {}).get("out_dim") for _, t, _ in comps}
    R.add("2 행동", "차원·행동 클립", same_or(out_dim, lambda v: f"actor 출력 {v}") + f" · actions.clip {tr0.act.get('clip')} · clip_actions {tr0.agent.get('clip_actions')}",
          f"러너 출력 12 · 클립 없음(프로브 원시 행동 최대 |값| {fnum(seq.get('raw_absmax'), 3)} 가 그대로 목표식에 들어간다 — 재구성 오차는 아래)", J_("같음"),
          OK if all(v == 12 for v in out_dim.values()) and tr0.act.get("clip") is None and tr0.agent.get("clip_actions") is None else BAD, cover="러너 --selftest")
    sc_tr = {t.tag: t.act.get("scale") for _, t, _ in comps}
    scf = [v for v in seq.get("scale_fit", []) if v == v]
    R.add("2 행동", "스케일", same_or(sc_tr), f"프로브 역산 {fnum(float(np.mean(scf)) if scf else None, 6)}(관절 {len(scf)}/12, 포화 안 된 틱 {seq.get('unsat_frac', 0):.0%})",
          J_(attJ.get("ACTION_SCALE"), ref=att.get("ACTION_SCALE")),
          OK if scf and all(abs(v - float(tr0.act.get("scale"))) < 1e-4 for v in scf) and len(set(sc_tr.values())) == 1 else BAD, cover="브리지 selftest '학습 계약'")
    dff = seq.get("default_fit", [])
    e3 = max((abs(a - b) for a, b in zip(dff, dp0) if a == a), default=float("nan"))
    R.add("2 행동", "오프셋 = 기본자세(use_default_offset)", f"use_default_offset {tr0.act.get('use_default_offset')} · 명목 기본자세(위 joint_pos 줄)",
          f"프로브 역산 오프셋과 학습 기본자세 최대 |Δ| {fnum(e3, 2)} rad", J_("목표 비트 동일" if ctx["bit"].get(key, {}).get("out") else None),
          OK if tr0.act.get("use_default_offset") is True and e3 < 1e-4 else BAD, cover="브리지 selftest '학습 계약'")
    sl_tr = {t.tag: t.act.get("max_delta_per_step") for _, t, _ in comps}
    R.add("2 행동", "슬루 상한(틱당)", same_or(sl_tr, lambda v: f"{math.degrees(float(v)):.4f}°"), f"프로브 최대 틱 변화 {math.degrees(seq.get('slew_max', float('nan'))):.4f}° · 1 rad 떨어진 첫 틱 이동 {math.degrees(seq.get('slew0', float('nan'))):.4f}°",
          J_(attJ.get("SLEW_RAD"), ref=att.get("SLEW_RAD"), fmt=lambda v: f"{math.degrees(float(v)):.4f}°"),
          OK if abs(seq.get("slew0", 9) - float(tr0.act.get("max_delta_per_step"))) < 1e-6 and seq.get("slew_max", 9) <= float(tr0.act.get("max_delta_per_step")) + 1e-6 and len(set(sl_tr.values())) == 1 else BAD,
          cover="브리지 selftest '학습 계약'·'정책 500틱'")
    R.add("2 행동", "슬루 기준 초기화", "리셋 틱의 현재 관절각(_needs_init → joint_pos)" + ("" if F.get("slew_init_from_joint_pos") else " ⚠소스 확인 실패"),
          f"첫 step 의 qpos(프로브: 전 관절이 qpos 에서 정확히 슬루 한 번 {seq.get('slew0_all_equal')})", J_("목표 비트 동일" if ctx["bit"].get(key, {}).get("out") else None),
          OK if seq.get("slew0_all_equal") and F.get("slew_init_from_joint_pos") else BAD, cover="러너 --selftest")
    R.add("2 행동", "목표각 재구성(600 틱)", "기본자세 + 스케일·원시 → 슬루(학습 식)", f"실제 러너 출력과 최대 |Δ| {fnum(seq.get('act_err'), 2)} rad · 타깃 클램프 없음",
          J_("HEAD 와 비트 동일" if ctx["bit"].get(key, {}).get("out") else "HEAD 와 다름") if in_jet else jmark, OK if seq.get("act_err", 1) < TOL_ACT else BAD, cover="[P1]·[P4]")
    order = F.get("action_order", [])
    nmk = ctx["nodemeas"].get(key) or {}
    R.add("2 행동", "처리 순서", " → ".join(order) + ("" if F.get("action_order_complete") else " ⚠소스 확인 일부 실패"),
          f"러너 슬루 → 브리지 LPF → (보정기 꺼짐) → 온셋 블렌딩 → 전송 → stage8 슬루·클램프; 노드 세션 재구성 오차 {fnum(nmk.get('chain_err_rad'), 2)} rad·블렌딩 {fnum(nmk.get('blend_err_deg'), 2)}°",
          J_("노드 발행 비트 동일" if ctx["bit"].get(key, {}).get("node") else "확인 못 함") if in_jet else jmark,
          OK if (order[:4] == ["지연", "슬루", "LPF", "온셋 블렌딩"] and nmk.get("chain_err_rad", 1) < 1e-5 and nmk.get("blend_err_deg", 1) < 1e-6) else (UNK if not nmk else BAD),
          "지연의 자리만 다르다(학습: 슬루 앞 / 실물: 체인 뒤 전송 지연) — 시불변 연산과는 교환되고, 온셋 블렌딩 시계만 지연만큼 어긋난다",
          cover="deploy_chain --selftest·[P4]")
    lr_all = {t.tag: t.act.get("lpf_beta_range") for _, t, _ in comps}
    R.add("2 행동", "액션 LPF β", "env 별 " + same_or(lr_all, lambda v: f"U{fnum(v)}"), f"STAND_LPF_BETA {lpf_b} · 노드 세션 역산 {fnum(nmk.get('lpf_beta_fit'), 5)}(키가 stand 로 시작 → 적용 {K.get('lpf_applies')})",
          J_((J or {}).get("registry", {}).get("STAND_LPF_BETA"), ref=lpf_b),
          OK if all(v is not None and float(v[0]) - 1e-9 <= lpf_b <= float(v[1]) + 1e-9 for v in lr_all.values()) and K.get("lpf_applies") and abs(nmk.get("lpf_beta_fit", lpf_b) - lpf_b) < 1e-4 else BAD,
          cover="브리지 selftest 'stand LPF'(주파수 응답)")
    R.add("2 행동", "LPF 초기값", "리셋 틱의 현재 관절각" + ("" if F.get("lpf_init_from_joint_pos") else " ⚠소스 확인 실패"),
          f"첫 슬루 출력(노드 세션: 첫 발행 − 첫 슬루 출력 = {fnum(nmk.get('lpf_first_is_slew_out'), 2)} rad)", J_("같음"),
          BAD if nmk.get("lpf_first_is_slew_out", 1) < 1e-9 else UNK,
          f"시뮬 쪽이 학습 계약이다. 첫 틱 차이 ≤ 슬루 × (1−β) = {math.degrees(float(tr0.act.get('max_delta_per_step', 0)) * (1 - lpf_b)):.2f}°, (1−β)ⁿ 로 사라진다. 그 구간은 온셋 블렌딩 α ≈ 0 이라 발행 목표에 거의 안 나온다",
          cover="PARITY #13·[P4]", known="기지(PARITY #13)",
          ev=line_of("rl_walking/actions.py", r"self\._lpf_state\[ids\] = cur") + " ↔ " + line_of("rl_walking/deploy/deploy_chain.py", r"^def lpf_step") + "·" + line_of("jetson/scripts/rl_bridge_node.py", r"self\.tgt_filt = lpf_step\(self\.tgt_filt, targets_rad\)"))
    ob_all = {t.tag: (t.act.get("onset_hold_range"), t.act.get("onset_blend_range")) for _, t, _ in comps}
    bs = regH["BLEND_S"]
    okb = all(v[0] is not None and float(v[0][0]) <= 0.0 <= float(v[0][1]) and float(v[1][0]) - 1e-9 <= bs <= float(v[1][1]) + 1e-9 for v in ob_all.values())
    R.add("2 행동", "온셋 블렌딩(인계)", same_or(ob_all, lambda v: f"유지 U{fnum(v[0])} s + 스무스스텝 U{fnum(v[1])} s"),
          f"유지 0 s + 스무스스텝 BLEND_S {bs} s — onset_alpha(소스 실행) vs blend_alpha 최대 |Δ| {fnum(ctx.get('onset_err'), 2)}; 노드 세션 α(t) 오차 {fnum(nmk.get('alpha_err'), 2)}(기록 3 자리 반올림)",
          J_((J or {}).get("registry", {}).get("BLEND_S"), ref=bs), OK if okb and (ctx.get("onset_err") or 1) < 1e-6 and nmk.get("alpha_err", 1) < 6e-4 else BAD,
          "학습 대역의 경계값(유지 0·블렌딩 상한)", cover="브리지 selftest '온셋 블렌딩'·deploy_chain --selftest")
    R.add("2 행동", "블렌딩 시작점", "env 기본자세(편향 DR 포함) 홀드 목표" + ("" if F.get("onset_hold_tgt_default") else " ⚠소스 확인 실패"),
          f"인계 순간 stage8 이 스트리밍하던 명령(노드 세션: 첫 발행 − 가짜 stage8 명령 = {fnum(nmk.get('blend_from_is_stage8_cmd'), 2)}°)", J_("같음"), UNK,
          "실물 스탠바이 자세와 정책 기본자세의 차가 학습 편향 대역 안인지는 현장 자세에 달렸다(착지 자동 인수 경로만 8° 게이트 TD_POSE_TOL_DEG)", cover="없음")
    dl = {t.tag: (t.act.get("min_delay_steps"), t.act.get("max_delay_steps")) for _, t, _ in comps}
    lo, hi = max(v[0] for v in dl.values()), min(v[1] for v in dl.values())
    sc_j = F.get("soft_clamp_joints_default")
    jl = (tr0.ev.get("joint_limits_config") or {}).get("params", {})
    s8f = ctx.get("stage8") or {}
    clamp_re = jl.get("joints", sc_j) or ""
    ctx["clamp_regex"] = clamp_re

    def _ex_txt(rel_, v_):
        nm_ = os.path.basename(os.path.dirname(rel_)).replace("real_", "")
        gi = REAL_LOG_GROUND.get(rel_)
        sd = v_.get("exceed_sides") or []
        gr = [x for x in sd if x["n_ground"]]
        txt = f"{nm_} {v_['key'].replace('stand_', '')}" + (f"(지면 구간 = {gi[0]:.0f} s 이전)" if gi else "(전 구간 지면)") + ": " + (
            ", ".join(f"{x['joint'].replace('_joint', '')} {'<' if x['side'] == 'soft_min' else '>'} {x['bound']:g}° {x['n_ground']} 틱({x['t_ground'][0]}–{x['t_ground'][1]} s, 최대 +{x['max_ground']:.1f}°)"
                      for x in gr) or "초과 없음")
        if gi:
            txt += f"; 그 뒤(줄 회수·공중) {sum(x['n'] - x['n_ground'] for x in sd)} 틱은 따로"
        return txt + f" · stage8 클램프 누계 {sum((v_.get('clamp_by_joint') or {}).values())}" + ("(= 전 구간 초과 틱 수; 브리지가 읽는 누계는 한 틱 늦다)" if v_.get("clamp_matches_exceed") else "(초과 틱 수와 다름 — 확인 필요)")

    ex_logs = [(rel_, v_) for rel_, v_ in (ctx.get("reallogs") or {}).items() if v_ and v_.get("exceed_sides")]
    not_trained = sorted({x["joint"] for _, v_ in ex_logs for x in v_["exceed_sides"] if x["n_ground"] and not re.fullmatch(clamp_re, x["joint"])})
    nt_txt = "; ".join(f"{os.path.basename(os.path.dirname(rel_)).replace('real_', '')} {x['joint'].replace('_joint', '')} {x['n_ground']} 틱(+{x['max_ground']:.1f}°)"
                       for rel_, v_ in ex_logs for x in v_["exceed_sides"] if x["n_ground"] and x["joint"] in not_trained)
    ctx["clamp_not_trained"] = nt_txt
    asym_all = rl.get("soft_asym_all") or {}
    asym_txt = " · ".join(f"{k} " + ", ".join(f"{x:g}°/{y:g}°" for x, y in v) for k, v in asym_all.items())
    ctx["soft_asym_txt"] = asym_txt
    ld = ctx.get("limits_date")
    R.add("2 행동", "소프트 한계 클램프(자르는 관절)", f"clamp_to_soft_limits {tr0.act.get('clamp_to_soft_limits')}: config soft 를 '{clamp_re}' 관절만(그 밖은 물리 한계까지 통과)"
          + (f" · 한계 파일 마지막 변경 {ld[0]}" + ("(모든 run 학습 시작 전 — 학습과 실물이 같은 표를 읽었다)" if ld[2] else " ⚠학습 시작 뒤 바뀜 — 학습 때 값과 다를 수 있다") if ld else ""),
          "stage8 노드: **전 관절**을 config soft 경계로 자른다(러너·브리지는 자르지 않음)" + ("" if s8f.get("clamp_all") else " ⚠소스 확인 실패"), J_("같음(stage8 — e7710c63 판도 같은 규칙)" if s8f.get("jet_same") else "stage8 판 미확인"),
          BAD if clamp_re != ".*" else OK,
          "학습과 같은 쪽이 없다(실물이 더 많은 관절을 자른다). 실물 로그에서 목표가 soft 를 넘은 틱 — " + " / ".join(_ex_txt(rel_, v_) for rel_, v_ in ex_logs) + ". "
          + (f"그중 **학습에서 자르지 않는 관절: {nt_txt}**. " if nt_txt else "")
          + (f"soft 범위가 좌우로 다른 관절(왼/오른): {asym_txt}. " if asym_txt else "")
          + "이 클램프가 1007 두 번째 회복의 결과에 영향을 줬는지는 모른다 — 대조 실험이 없고, 같은 회복에서 35.4 s 무렵 발목 롤 드라이버 두 개가 재시작해 35.60 s 에 스택이 정지했다(reports/real_1007/README.md). "
          + ("KE 회복·전환 시뮬 평가기(eval_ground_recovery·eval_transition*)에는 stage8 식 전 관절 클램프가 없다 — " if ctx.get("ke_eval_has_s8_clamp") is False else "")
          + "W3S3d 걷기만 대조가 있다: 발목 롤 목표의 12–17 % 틱이 잘리고 폐루프 낙상 차 신호 없음(PARITY §4.4). "
          + "종전 문서에는 '클램프 누계 증가'(REAL_1003·PARITY §4.4)와 '좌우 비대칭·stand 세대 기준 재검토 권고'(0905 B7)로 적혀 있다 — 이 줄이 더한 것은 관절·방향별 실측과 KE 평가의 빈칸이다"
          + (f"(0905 이후 한계 파일 변경 {len(ctx.get('limits_since_0905') or [])} 건: {'; '.join(ctx.get('limits_since_0905') or []) or '없음'})" if ld else ""),
          cover="PARITY #14(문서, 발목 롤만)·0905 감사 B7", known="기지(PARITY #14·§4.4, 0905 B7, REAL_1003)",
          ev=line_of("rl_walking/mdp_hang.py", r"^def set_joint_limits_from_config") + "(joints 기본값)·" + line_of("rl_walking/actions.py", r"if self\._clamp_soft:") + " ↔ "
          + line_of("jetson/scripts/stage8_12axis_mit_control_node.py", r"clamped = min\(max\(joint_abs_deg") + " · config/joint_limits_12dof.json · " + " · ".join(rel_ for rel_, _ in ex_logs))
    mx_by = {os.path.basename(os.path.dirname(rel_)).replace("real_", ""): max(x["max"] for x in v_["exceed_sides"]) for rel_, v_ in ex_logs}
    mx_ex = max(mx_by.values(), default=None)
    rej_all = sorted({r_ for _, v_ in ex_logs for r_ in (v_.get("reject_reasons") or [])})
    R.add("2 행동", "한계 초과 명령의 거부", "없음(클램프만 — 목표가 아무리 넘어도 경계값으로 계속 구동)",
          "stage8: 경계를 25° 넘게 넘으면 그 관절 명령을 거부(limit_runaway)" + ("" if s8f.get("reject_25") else " ⚠소스 확인 실패"), J_("같음(stage8)" if s8f.get("jet_same") else "stage8 판 미확인"), BAD,
          "학습과 같은 쪽이 없다. "
          + ("실물 로그의 최대 초과(전 구간): " + ", ".join(f"{k} +{v:.1f}°" for k, v in mx_by.items()) + f" — 거부 문턱까지 {25.0 - mx_ex:.1f}° 남았다(거부 기록 {rej_all or '없음'}). " if mx_ex is not None else "")
          + "넘는 순간 실물은 그 관절이 직전 명령에 머물고 시뮬은 경계에서 계속 민다(발목 명령이 soft 39° 를 넘어 56.7° 까지 감긴 0922 실물 낙상과 같은 계열 — rl_bridge_node.py 의 stand_crouch_v6 주석)",
          cover="0905 감사 B7(문서)", known="기지(0905 감사 B7)",
          ev=line_of("rl_walking/actions.py", r"if self\._clamp_soft:") + " ↔ " + line_of("jetson/scripts/stage8_12axis_mit_control_node.py", r"if excess > 25\.0:"))
    R.add("2 행동", "노드(stage8) 슬루·주기", f"슬루는 한 번({math.degrees(float(tr0.act.get('max_delta_per_step', 0))):.1f}°/틱)",
          f"러너 슬루 뒤 stage8 이 한 번 더: {s8f.get('step')}°/틱 @ {s8f.get('period')} s — 한도 안 입력에는 항등. 노드 세션 발행 틱 변화 최대 {nmk.get('slew_pub_max_deg', float('nan')):.6f}°",
          J_(f"{JET_COMMIT} 판 stage8 상수 같음(젯슨에 어느 판이 있는지는 미확인)" if s8f.get("jet_same") else "stage8 판 미확인"),
          OK if (s8f.get("step") == 4.0 and s8f.get("period") == 0.02 and nmk.get("slew_pub_max_deg", 99) <= 4.0 + 1e-4) else (UNK if not nmk else BAD),
          "전송 지터로 한 노드 틱에 명령이 두 번 오거나 빠지면 노드 슬루가 실제로 걸릴 수 있다(실물 1007: 발행→처리 지연 최대 "
          + (f"{rl.get('s8_age_max_ms'):.1f} ms, 누락 {rl.get('s8_stale')})" if rl.get("s8_n") else "기록 없음)"), cover="policy_runner 문서(이중 슬루 항등)·없음(시험)")
    ac = {t.tag: sorted({(v[0], v[1]) for v in t.actuators().values()}) for _, t, _ in comps}
    gm = regH["map"].get("gain_mismatches")
    R.add("2 행동", "PD 게인(kp/kd)", same_or(ac, str) + f" · DR ×{fnum((tr0.ev.get('actuator_gains') or {}).get('params', {}).get('stiffness_distribution_params'))}/×{fnum((tr0.ev.get('actuator_gains') or {}).get('params', {}).get('damping_distribution_params'))}",
          f"POLICY_KP {regH['POLICY_KP']}/KD {regH['POLICY_KD']} · 맵 불일치 관절 {len(gm) if gm is not None else '?'}(--live 는 불일치면 기동 거부)",
          J_(f"KP {(J or {}).get('registry', {}).get('POLICY_KP')}/KD {(J or {}).get('registry', {}).get('POLICY_KD')}" + (" · 젯슨 맵 사본(1005) 게인 같음" if ctx.get("jetmap_same") else " · 젯슨 맵 사본 없음" if ctx.get("jetmap_same") is None else " · 젯슨 맵 사본 게인 다름")) if in_jet else jmark,
          OK if all(v == [(150.0, 5.0)] for v in ac.values()) and regH["POLICY_KP"] == 150.0 and regH["POLICY_KD"] == 5.0 and gm == [] else BAD,
          cover="브리지 selftest '게인 정책'")
    R.add("2 행동", "학습에 없는 가산 보정(roll/pitch comp)", "없음", f"맵 roll_comp {regH['map'].get('roll_comp_enabled')} · pitch_comp {regH['map'].get('pitch_comp_enabled')}",
          J_("젯슨 맵 사본(1005)도 꺼짐" if ctx.get("jetmap_comp_off") else "젯슨 맵 사본 없음" if ctx.get("jetmap_comp_off") is None else "젯슨 맵 사본에서 켜짐") if in_jet else jmark,
          OK if not regH["map"].get("roll_comp_enabled") and not regH["map"].get("pitch_comp_enabled") else BAD, cover="브리지 selftest '롤 보정기'(꺼짐 = 비트 동일)")

    # ------------------------------------------------------------------ 3. 시간
    dts = {t.tag: t.dt() for _, t, _ in comps}
    R.add("3 시간", "제어 주기", same_or(dts, lambda v: f"{v * 1e3:.1f} ms") + f"(sim.dt × decimation {tr0.env.get('decimation')})",
          f"TICK_SEC {regH['TICK_SEC'] * 1e3:.1f} ms · 러너 POLICY_DT {att.get('POLICY_DT', 0) * 1e3:.1f} ms" + (f" · 감지기 dt {PH.get('det', {}).get('dt', 0) * 1e3:.1f} ms" if is_c5 else "")
          + (f" · W3Config.dt {(regH['keys'][comps[-1][2]].get('w3cfg') or {}).get('dt', 0) * 1e3:.1f} ms" if (is_w3 or is_kew3) else "")
          + (f" · 실물 1007 로그 틱 간격 평균 {rl.get('dt_mean_ms'):.2f} ms(1–99 % {rl.get('dt_p01_ms'):.1f}–{rl.get('dt_p99_ms'):.1f}, 최대 {rl.get('dt_max_ms'):.1f})" if rl else ""),
          J_((J or {}).get("registry", {}).get("TICK_SEC"), ref=regH["TICK_SEC"], fmt=lambda v: f"{float(v) * 1e3:.1f} ms"),
          OK if all(abs(v - regH["TICK_SEC"]) < 1e-9 for v in dts.values()) and abs(att.get("POLICY_DT", 0) - regH["TICK_SEC"]) < 1e-12 else BAD,
          "클록·필터·감지기 타이머는 틱 수로 시간을 센다 — 실물 틱 간격이 20 ms 에서 벗어나면 그만큼 어긋난다", cover="브리지 selftest '학습 계약'(50 Hz)")
    R.add("3 시간", "행동 지연", same_or(dl, lambda v: f"U{{{v[0]}..{v[1]}}} 틱") + (f" → 공통 구간 {lo}..{hi} 틱" if len(set(dl.values())) > 1 else ""),
          "인위 지연 없음 — 실물 루프 지연이 그 몫(총 지연 미측정)"
          + (f". 실물 1007 로그의 구간 실측: 브리지 발행 → stage8 처리 중앙 {rl.get('s8_age_med_ms'):.1f} ms(99 % {rl.get('s8_age_p99_ms'):.1f}, 최대 {rl.get('s8_age_max_ms'):.1f}) · status 나이 중앙 {rl.get('status_age_med_ms'):.0f} ms(최대 {rl.get('status_age_max_ms'):.0f})" if rl.get("s8_n") else ""),
          J_("같음(위 실측이 젯슨 세션 값)"), UNK,
          "명령 전송·피드백 수신 구간은 한 틱(20 ms) 안으로 측정됐지만 모터 응답까지의 총 지연은 이 기록으로 알 수 없다 → 현장 측정 필요(REAL_TEST1 M1). "
          + (f"학습 하한이 {lo} 틱인 정책(걷기)이 있다 — 걷기 시뮬 평가는 지연 1–7 틱만 했고 0 틱 평가는 없다(PARITY §4.1: ≥ 5 틱에서 낙상 3–7 배)" if lo >= 1 else "")
          + ("; 합성 러너는 한 하드웨어 지연이 두 정책의 학습 범위에 모두 들어야 한다" if len(set(dl.values())) > 1 else ""),
          cover="없음(PARITY #15)")
    cbp = (tr0.ev.get("calib_bias") or {}).get("params", {})
    fcp = tr0.term("feet_contact").get("params") or {}
    fh = [(os.path.basename(os.path.dirname(rel)), v) for rel, v in (ctx.get("reallogs") or {}).items() if v and v.get("foot_hold_mode") is not None]
    if fh:
        R.add("3 시간", "접촉 갱신 주기", f"{fcp.get('hold_ticks')} 틱마다 갱신(feet_contact hold_ticks) — 그 사이는 직전 비트 유지",
              "브리지는 틱마다 최신 수신 비트를 읽는다. 실물 로그에서 발 원카운트 합이 바뀌는 간격: "
              + "; ".join(f"{nm} 최빈 {v['foot_hold_mode']} 틱({v['foot_hold_mode_n']}/{v['foot_hold_n']} 구간, 더 긴 구간 {v['foot_hold_longer']} 개·최대 {v['foot_hold_max']} 틱)" for nm, v in fh),
              J_("같은 실측(젯슨 세션 기록)"), OK if all(v["foot_hold_mode"] == fcp.get("hold_ticks") for _, v in fh) else BAD,
              "어댑터 갱신 주기(100 ms = 5 틱)가 학습의 유지 틱과 같다. 원값이 0 으로 이어진 구간(발이 뜬 상태)은 셈에서 뺐다. "
              "하중이 실린 채 더 길게 이어진 구간은 메시지 한 번 누락(나이 200 ms)으로 보이나 같은 값이 연속으로 나온 것일 수도 있다 — 이 기록으로는 구분하지 못한다",
              cover="없음")
    imu_rep = "; ".join(f"{os.path.basename(os.path.dirname(rel))} {v['imu_repeat_frac']:.2%}" for rel, v in (ctx.get("reallogs") or {}).items() if v and v.get("imu_repeat_frac") is not None)
    R.add("3 시간", "센서 관측 지연(IMU·접촉)", f"자이로 0–{(tr0.term('base_ang_vel').get('params') or {}).get('max_lag')} 틱 · 자세(중력) {fnum(cbp.get('imu_lag_steps'))} 틱 · 접촉 {fcp.get('hold_ticks')} 틱 유지 + {fcp.get('extra_delay')} 틱",
          f"브리지는 최신 수신값을 그대로 쓴다(신선도 한계 IMU·관절 {regH.get('SENSOR_STALE_SEC')} s, 발 {regH.get('FOOT_STALE_SEC')} s). 센서 안쪽·전송 지연은 미측정"
          + (f". 브리지에서 본 것: IMU 표본이 직전 틱과 같은 틱 {imu_rep}(거의 매 틱 새 표본)" if imu_rep else ""), J_("같음"), UNK,
          f"현장 측정 필요. 발 접촉은 {regH.get('FOOT_STALE_SEC')} s 묵은 값까지 정상 입력으로 쓴다 — 학습의 최대 지연({(fcp.get('hold_ticks') or 0) + (fcp.get('extra_delay') or 0)} 틱 = "
          f"{((fcp.get('hold_ticks') or 0) + (fcp.get('extra_delay') or 0)) * regH['TICK_SEC']:.2f} s)보다 길다(발 메시지가 2 번 넘게 빠진 경우)", cover="없음")

    # ------------------------------------------------------------------ 4. 위상/클록 (W3)
    if is_w3 or is_kew3:
        role, tr, sub = comps[-1]
        g3, c3 = tr.w3()
        wc = regH["keys"][sub].get("w3cfg") or {}
        gd = F.get("w3_gait_defaults", {})
        sq = ctx["meas"].get(sub) or {}
        ramp_tr = g3.get("ramp_s", gd.get("ramp_s"))
        R.add("4 위상", "스텝 주파수 f_step", g3.get("f_step", gd.get("f_step")), f"W3Config {wc.get('f_step')} · 프로브(걷는 중 위상 증가/틱) {fnum(sq.get('w3_freq'), 5)} Hz", jmark or "—",
              OK if abs(float(g3.get("f_step", gd.get("f_step"))) - float(wc.get("f_step", -1))) < 1e-12 and abs(sq.get("w3_freq", 0) - float(wc.get("f_step", -1))) < 1e-3 else BAD,
              cover="브리지 selftest 'W3S3d 명령 채널' ①·selftest_w3_equiv [B]")
        R.add("4 위상", "관측 게이트 램프", f"ramp_s {ramp_tr}(env.yaml 에 없으면 w3_gait_update 기본값)", f"W3Config {wc.get('ramp_s')} · 프로브: 켜진 뒤 게이트 1 까지 {sq.get('w3_ramp_ticks')} 틱",
              jmark or "—", OK if abs(float(ramp_tr) - float(wc.get("ramp_s", -1))) < 1e-12 and all(abs(n * wc.get("dt", 0.02) - float(ramp_tr)) < 1e-9 for n in sq.get("w3_ramp_ticks", [0])) else BAD,
              cover="selftest_w3_equiv [B]")
        offl = sq.get("w3_off", [])
        R.add("4 위상", "gait_enable 래치", "켜기: 다음 갱신에 즉시(위상 0 또는 리드 규칙) / 끄기: 반주기 경계에서만 / 에피소드 틱 ≤ 1 은 즉시 반영" + ("" if F.get("w3_fresh_le1") else " ⚠소스 확인 실패"),
              f"프로브: 켜기 지연 {sq.get('w3_on_delay')} 틱 · 끄기 {[(n, round(u, 3), b) for n, u, b in offl]}(대기 틱, 직전 위상, 경계 통과) · 서기 클록 (0,0) {sq.get('w3_stand_clock_zero')}",
              jmark or "—", OK if offl and all(b for _, _, b in offl) and all(d == 0 for d in sq.get("w3_on_delay", [1])) and sq.get("w3_stand_clock_zero") else BAD,
              cover="selftest_w3_equiv [B](시뮬 소스 실행, 켜짐 길이 1–50 전수)·[P1]")
        lead_tr = (g3.get("lead_vy_cmin", gd.get("lead_vy_cmin")), g3.get("lead_wz_cmin", gd.get("lead_wz_cmin")), F.get("w3_lead_lat_eps"))
        lead_dp = (wc.get("lead_vy_cmin"), wc.get("lead_wz_cmin"), wc.get("lead_lat_eps"))
        R.add("4 위상", "리드 발 규칙 문턱(vy, wz, 옆 없음 판정)", fnum(lead_tr), f"W3Config {fnum(lead_dp)} · 프로브 시작 위상 {[round(u, 2) for u in sq.get('w3_start_u', [])]}(앞, 옆−, 회전−, 제자리) · 리드 {(H['probe'].get(sub, {}).get('seq') or {}).get('lead')}",
              jmark or "—", OK if verdict_eq(lead_tr, lead_dp, 1e-12) == OK and [round(u, 2) for u in sq.get("w3_start_u", [])] == [0.0, 0.5, 0.5, 0.0] else BAD,
              cover="selftest_w3_equiv [E]·[P1](옆 − 시작)")
        cur = ctx["curric"].get(tr.run, {})
        wr = (tr.ev.get("w3_ref") or {}).get("params", {})
        R.add("4 위상", "행동 보조·참조 잔차", f"ref_scale {wr.get('ref_scale')} · 보조 α 학습 로그 {fnum(cur.get('alpha')) if cur else '기록 없음'}(커리큘럼 start_stage {((tr.env.get('curriculum') or {}).get('w3_assist') or {}).get('params', {}).get('start_stage')})",
              "배포에 잔차 없음(러너 w3 경로는 클록 관측만 바꾼다)", jmark or "—",
              OK if (wr.get("ref_scale") in (0, 0.0) and cur and cur.get("alpha") == 0.0) else (UNK if not cur else BAD), cover="PARITY #22(기록 최대 0)")

    # ------------------------------------------------------------------ 5. 감독기/감지기
    if is_c5:
        det = PH.get("det") or {}
        detJ = PJ.get("det") or {}
        ksub = K["kew3"]["ke"] if is_kew3 else key
        arm = "krh" if "krh" in ksub else "kr"
        ev = ctx["eval_ke"].get(arm) or {}
        if is_kew3:
            ev = ctx["eval_tr"] or ev

        def sp(flag, n=None):
            v = ev.get(flag)
            if v is None or v is True:
                return v
            xs = [float(x) for x in str(v).split(",")]
            return xs if n is None else xs[n]
        src = ctx.get("eval_src", {}).get("tr" if is_kew3 else arm, "평가 인자 없음")
        R.add("5 감지기", "평가 설정의 출처", src, "브리지 C5_DETECTOR_KW·C5_EXTRA → 실제 감지기 객체에서 읽은 값", J_("같은 방식") if in_jet else jmark, OK if ev else UNK,
              "아래 줄들의 '학습 값' 칸 = 이 시뮬 평가에서 실제로 쓴 값" if ev else "평가 산출물·스크립트를 찾지 못했다", cover="없음")
        pairs = [("켜기 기울기 편차 [°]", sp("--sw_on", 0), det.get("on_tilt_deg"), detJ.get("on_tilt_deg")),
                 ("켜기 |ω_xy| [rad/s]", sp("--sw_on", 1), det.get("on_w"), detJ.get("on_w")),
                 ("끄기 기울기 [°]", sp("--sw_off", 0), det.get("off_tilt_deg"), detJ.get("off_tilt_deg")),
                 ("끄기 |ω_xy|", sp("--sw_off", 1), det.get("off_w"), detJ.get("off_w")),
                 ("끄기 평온 지속 [s]", sp("--sw_off", 2), det.get("calm_s"), detJ.get("calm_s")),
                 ("켜진 뒤 최소 [s]", sp("--sw_off", 3), det.get("dwell_s"), detJ.get("dwell_s")),
                 ("기준 자세 τ [s]", sp("--sw_ref_tau", 0), det.get("ref_tau_s"), detJ.get("ref_tau_s")),
                 ("기준 추종 창 [s]", sp("--sw_ref_init_s", 0), det.get("ref_init_s"), detJ.get("ref_init_s")),
                 ("조기 복귀 (최소 s, 기울기, |ω|, 유지 s)", sp("--sw_early_imu"), det.get("early"), detJ.get("early")),
                 ("불응기 (s, 기울기, |ω|)", sp("--sw_refrac"), det.get("refrac"), detJ.get("refrac")),
                 ("조기 복귀 자격 (최대 기울기, 최대 |ω|)", sp("--early_gate"), det.get("early_gate"), detJ.get("early_gate")),
                 ("감지 입력 EMA α", sp("--det_filt", 0), det.get("in_filt"), detJ.get("in_filt"))]
        for nm_, e_, d_, j_ in pairs:
            R.add("5 감지기", nm_, fnum(e_) if e_ is not None else "평가 인자 없음", fnum(d_), J_(j_, ref=d_) if in_jet else jmark,
                  UNK if e_ is None else verdict_eq(e_, d_, 1e-9), cover="브리지 selftest 'C5 kr 경로'·'C5 krh 키'(값 전달)·selftest_c5_imu_f32 [B](식)")
        R.add("5 감지기", "끄기 접촉 요구", f"{ev.get('--sw_off_contact')}", det.get("off_contact"), J_(detJ.get("off_contact"), ref=det.get("off_contact"), fmt=str) if in_jet else jmark,
              OK if ev.get("--sw_off_contact") == det.get("off_contact") else (UNK if not ev else BAD), cover="selftest_c5_imu_f32 [B]")
        R.add("5 감지기", "기준 자세 상한·갱신 조건", "±8°, 서기·|ω| < 0.1·양발일 때만 갱신(평가기 소스 상수)" + ("" if ctx.get("eval_ref_consts") else " ⚠소스 확인 실패"),
              f"ref_cap {det.get('ref_cap_deg')}°, |ω| < 0.1·양발(c5_switch 소스)", J_(detJ.get("ref_cap_deg"), ref=det.get("ref_cap_deg")) if in_jet else jmark,
              OK if det.get("ref_cap_deg") == 8.0 and ctx.get("eval_ref_consts") else BAD, cover="selftest_c5_imu_f32 [B]")
        cue = PH.get("cue") or {}
        cueJ = PJ.get("cue") or {}
        R.add("5 감지기", "회복 cue (vx, 지속 s, 방향 = IMU ω_y)", f"({ev.get('--sw_rec_vx')}, {ev.get('--sw_rec_vx_s')}, {bool(ev.get('--cue_dir_imu'))})",
              f"({cue.get('cue_vx')}, {cue.get('cue_s')}, {cue.get('cue_dir_imu')})", J_(f"({cueJ.get('cue_vx')}, {cueJ.get('cue_s')}, {cueJ.get('cue_dir_imu')})") if in_jet else jmark,
              (OK if (abs(float(ev.get("--sw_rec_vx")) - float(cue.get("cue_vx"))) < 1e-9 and abs(float(ev.get("--sw_rec_vx_s")) - float(cue.get("cue_s"))) < 1e-9
                      and bool(ev.get("--cue_dir_imu")) == bool(cue.get("cue_dir_imu")) and (not in_jet or cueJ == cue)) else BAD) if ev else UNK,
              cover="selftest_odic_equiv [A]")
        R.add("5 감지기", "감지 입력 원천", "정책 관측 최신 프레임(지연·잡음 자이로·중력, 접촉 0/1) — --det_from_obs " + str(bool(ev.get("--det_from_obs"))),
              "정책이 받는 값과 같은 중력(장착 편향 보정 뒤)·자이로·접촉 비트", J_("같음"), OK if ev.get("--det_from_obs") else (UNK if not ev else BAD), cover="없음(구조)")
        R.add("5 감지기", "타이머 정밀도", "float32 누적(시뮬 텐서)", f"float64 누적(imu_f32 {det.get('imu_f32')})", J_("float64(옵션 없음)" if not detJ.get("has_imu_f32_opt") else "float64") if in_jet else jmark,
              BAD if not det.get("imu_f32") else OK, "평가 쪽(float32)이 기준이다. 꺼짐 시각이 일부 1 틱 이르다(91 건 중 44 건). 짝 비교 낙상 차 없음(PARITY §4.7). 실물 1007 재생에서는 회복 모드 불일치 0 틱", cover="selftest_c5_imu_f32 [B]", known="기지(PARITY #24, 옵션 --c5-imu-f32 기본 꺼짐)",
              ev=line_of("rl_walking/scripts/eval_ground_recovery.py", r"sw_calm = torch\.where\(sw_mode & _quiet") + " ↔ " + line_of("rl_walking/deploy/c5_switch.py", r"self\.calm = self\.calm \+ self\.dt if"))
        ev_f = (ev.get("--last_action_ema"), ev.get("--obs_ema"), ev.get("--enc_obs"), ev.get("--contact_ver") if ev.get("--contact_real") else None)
        dep_f = (att.get("act_beta"), f"joint_vel:{att.get('jv_beta')}", att.get("jvel_ema_mode"), "v2" if short(tr0.term("feet_contact").get("func")).endswith("_v2") else "v1")
        try:
            okf = (abs(float(ev_f[0]) - float(dep_f[0])) < 1e-9 and str(ev_f[1]).split(":")[0] == "joint_vel"
                   and abs(float(str(ev_f[1]).split(":")[1]) - float(att.get("jv_beta"))) < 1e-9 and ev_f[2] == dep_f[2] and ev_f[3] == dep_f[3])
        except (TypeError, ValueError, IndexError):
            okf = None
        R.add("5 감지기", "평가 하네스의 관측 덮어쓰기(직전 행동 EMA, 관절속도 EMA, 엔코더 판, 접촉 판)", fnum(ev_f) if ev else "평가 인자 없음",
              f"배포 러너 ({dep_f[0]}, {dep_f[1]}, {dep_f[2]}) · 학습 접촉 함수 {dep_f[3]}", J_("같음"), (OK if okf else BAD) if okf is not None else UNK,
              "평가는 다른 태스크(G11a) 위에 인자로 G21a 관측 계약을 덮어쓴 것 — 덮어쓴 값이 배포·학습과 같은지", cover="없음")
        gd0 = ctx.get("eval_gyro_delay")
        R.add("5 감지기", "평가 하네스의 자이로 지연", f"학습 0–{(tr0.term('base_ang_vel').get('params') or {}).get('max_lag')} 틱 DR / KE·전환 평가는 태스크 {', '.join(ctx.get('eval_task') or ['?'])} + 관측 덮어쓰기(--gyro_delay {gd0} — 기본값, 평가 인자에 없음)",
              "실물 IMU 경로 지연 미측정", J_("같음"), UNK, "감지기·회복 평가 수치는 자이로 지연 0 조건에서 나왔다 — 실물 지연이 크면 평가가 덮지 않는 영역",
              cover="없음")
    if is_w3 or is_kew3:
        sub = comps[-1][2]
        sf = regH["keys"][sub].get("w3_safety") or {}
        pd = ctx.get("eval_pre") or {}
        e_ = (pd.get("--pre_both_s"), pd.get("--pre_tilt"), pd.get("--pre_w"), pd.get("--pre_w_s"))
        d_ = (sf.get("enable_contact_s"), sf.get("enable_tilt_deg"), sf.get("enable_quiet_w"), sf.get("enable_quiet_s"))
        R.add("5 감독기", "걷기 켜기 선행조건(양발 s, 기울기 °, |ω| , 정착 s)", f"{fnum(e_)}(eval_transition 기본값) — 학습에는 없음(에피소드 70 % 가 곧바로 STEP)", fnum(d_), jmark or "—",
              verdict_eq(e_, d_, 1e-9) if None not in e_ else UNK, "배포 전용 안전 조건(학습보다 엄격)", cover="브리지 selftest 'W3 세션 감독기'·시뮬 X6")
        lr_ = regH["keys"][sub].get("w3_cmd_limit_ranges") or {}
        lim = regH["keys"][sub].get("w3_limits") or {}
        R.add("5 감독기", "명령 상한 덮어쓰기(--w3-cmd-limits)", f"학습 |vy| ≤ {fnum(lim.get('vy', [0, 0])[1])}, |wz| ≤ {fnum(lim.get('wz', [0, 0])[1])}",
              f"기본 = 학습 범위. 기동 인자로 vy_max ≤ {fnum((lr_.get('vy_max') or [0, 0])[1])}, wz_max ≤ {fnum((lr_.get('wz_max') or [0, 0])[1])} 까지 열 수 있다", jmark or "—", OK,
              "기본값은 일치. 인자를 주면 학습 범위 밖(확장 평가 근거만 있음)", cover="브리지 selftest 'W3S3d 명령 채널' ②")

    # ------------------------------------------------------------------ 6. 합성 러너
    if is_kew3:
        kw = PH.get("kew3") or {}
        a1, a2 = PH.get("attrs_ke") or {}, PH.get("attrs_w3") or {}
        same_sub = all(a1.get(k) == a2.get(k) for k in ("obs_dim", "terms", "default_pose", "act_beta", "jv_beta", "jvel_ema_mode", "ema_fresh_ticks"))
        R.add("6 합성", "두 하위 러너 계약 호환", "세 run 의 관측 설정 " + ("동일" if same_obs else "다름") + " · 기본자세 " + ("동일" if same_dp else "다름"),
              f"KE 코어 = W3 코어: 관측 레이아웃·기본자세·필터 β·EMA 모드·ema_fresh {same_sub}", "—(젯슨 미배포 키)", OK if same_sub and same_obs and same_dp else BAD,
              cover="kew3_runner --selftest [6]·브리지 selftest 'KE↔W3 합성'")
        et = ctx["eval_tr_defaults"]
        es = ctx.get("eval_tr_settle")                    # 평가 산출물의 정착 판정 줄 (|ω| 문턱, 유지 s, 상한 s) — 없으면 평가기 기본값
        ev_set = (es[1], es[0], es[2]) if es else (et.get("--sq_hold"), et.get("--sq_w"), et.get("--settle_max"))
        kd_ = ctx.get("eval_kw_defaults") or {}
        etr = ctx.get("eval_tr") or {}
        x6 = bool(etr.get("--runner_loop"))
        ev_mode = etr.get("--kw_handoff", kd_.get("--kw_handoff")) if x6 else etr.get("--ke_mode")
        ev_ks = float(etr.get("--kw_settle", kd_.get("--kw_settle", "nan"))) if x6 else None
        R.add("6 합성", "W3 → KE 인계 규칙(정착 s, |ω_xy| 문턱, 상한 s, 방식)",
              f"평가기 정착 판정 {fnum(ev_set)}" + ("(산출물 머리)" if es else "(평가기 기본값)") + (f" · X6(배포 코드 폐루프) --kw_handoff {ev_mode} --kw_settle {fnum(ev_ks)}" if x6 else f" · --ke_mode {ev_mode}"),
              f"({kw.get('settle_quiet_s')}, {kw.get('settle_w')}, {kw.get('settle_max_s')}, {kw.get('handoff')}) · 프로브: 조용한 양발 {kw.get('quiet_ticks_to_ke')} 틱 뒤 인계, 시끄러우면 {kw.get('cap_ticks_to_ke')} 틱(상한), 한발이면 대기 {kw.get('single_support_stays_w3')}",
              "—(젯슨 미배포 키)",
              OK if (verdict_eq(ev_set, (kw.get("settle_quiet_s"), kw.get("settle_w"), kw.get("settle_max_s")), 1e-9) == OK
                     and kw.get("handoff") == "carry" and (ev_mode in ("carry", "direct")) and (ev_ks is None or abs(ev_ks - kw.get("settle_quiet_s", -1)) < 1e-9)
                     and abs(kw.get("quiet_ticks_to_ke", 0) * kw.get("dt", 0.02) - kw.get("settle_quiet_s", -1)) < 0.021
                     and abs(kw.get("cap_ticks_to_ke", 0) * kw.get("dt", 0.02) - kw.get("settle_max_s", -1)) < 0.021 and kw.get("single_support_stays_w3")) else BAD,
              "정착 신호 = 정책 관측과 같은 원시 자이로 |ω_xy|·접촉 비트(평가기: 관측 최신 프레임). X6 은 브리지 _make_runner + KeW3Runner 를 그대로 돌린 평가라 같은 코드다. 근거 문서 TRANSITION_W3KE §5.6k·§5.10c",
              cover="kew3_runner --selftest [8]·시뮬 X6(배포 코드 폐루프)")
        R.add("6 합성", "KE → W3 인계", "켜기 수락 틱에 W3(첫 틱 STAND, 다음 틱 gait 0→1, 게이트 0.2 s 램프) — 평가 direct 팔",
              f"프로브: 같은 틱 전환 {kw.get('to_w3_same_tick')} · 첫 틱 ge {kw.get('w3_first_tick_ge')}·클록 이력 0 {kw.get('w3_first_tick_clock_hist_zero')} · 둘째 틱 ge {kw.get('w3_second_tick_ge')}(게이트 {fnum(kw.get('w3_second_tick_ge_obs'), 2)}) · 전환 틱 목표 변화 {fnum(kw.get('to_w3_jump_deg'), 3)}°(슬루 상한 이내 — 슬루 기준을 이어받는다. 합성 입력이라 크기 자체는 뜻이 없다)",
              "—(젯슨 미배포 키)",
              OK if (kw.get("to_w3_same_tick") and not kw.get("w3_first_tick_ge") and kw.get("w3_first_tick_clock_hist_zero") and kw.get("w3_second_tick_ge")
                     and kw.get("to_w3_jump_deg", 99) <= 4.0 + 1e-3) else BAD, cover="kew3_runner --selftest [7]·시뮬 X1/X6")
        ckh = kw.get("ke_clock_hist_after_carry") or []
        R.add("6 합성", "인계 뒤 KE 상태", "이력·필터·직전 행동 이어받기, 클록만 위상 0 에서(평가 direct 팔의 클록 패치), 감지기 새로",
              f"프로브: KE 틱 {kw.get('ke_tick_after_carry')} · 클록 이력 전부 (0,1) {all(abs(r[0]) < 1e-9 and abs(r[1] - 1) < 1e-9 for r in ckh) if ckh else None} · 감지기 켜짐 누계 {kw.get('ke_det_n_on_after')} · 첫 틱 목표 변화 {fnum(kw.get('to_ke_jump_deg'), 3)}°(슬루 상한 이내)",
              "—(젯슨 미배포 키)", OK if (kw.get("ke_tick_after_carry") == 0 and ckh and all(abs(r[0]) < 1e-9 and abs(r[1] - 1) < 1e-9 for r in ckh)
                                    and kw.get("ke_det_n_on_after") == 0 and kw.get("to_ke_jump_deg", 99) <= 4.0 + 1e-3) else BAD,
              cover="kew3_runner --selftest [7]")
        R.add("6 합성", "회복 모드 중 걷기 켜기", "평가 X6: 감독기 선행조건 + KE 회복 모드면 거부", f"프로브: 회복 모드에서 사유 '{kw.get('block_reason_in_recovery')}'", "—(젯슨 미배포 키)",
              OK if kw.get("block_reason_in_recovery") else BAD, cover="kew3_runner --selftest [4]")
        R.add("6 합성", "서기 쪽 KE 변형", "전환 평가: 조기 복귀 유지 " + fnum((ctx["eval_tr"] or {}).get("--sw_early_imu", "?").split(",")[-1] if ctx["eval_tr"] else None) + " s",
              f"KEW3_SPECS ke = {K['kew3']['ke']}(유지 {(PH.get('det') or {}).get('early', [None] * 4)[3]} s)", "젯슨 콘솔에는 krh(0.7 s)가 따로 있다",
              OK if ctx["eval_tr"] and abs(float(ctx["eval_tr"]["--sw_early_imu"].split(",")[-1]) - float((PH.get("det") or {}).get("early", [0, 0, 0, -1])[3])) < 1e-9 else UNK,
              cover="없음")

    if is_kew3:
        for sk, nm_ in ((K["kew3"]["ke"], "서기·회복(KE) 하위 키"), (K["kew3"]["w3"], "걷기(W3) 하위 키")):
            ms = ctx["meas"].get(sk) or {}
            wv = {n: v for n, v in (ms.get("err_deprule") or {}).items()}
            R.add("6 합성", f"{nm_} 러너 프로브", f"학습 계약 재구성({sk} 표와 같은 검사)",
                  f"관측 최대 |Δ| {fnum(max(wv.values()) if wv else None, 2)} · 목표각 {fnum(ms.get('act_err'), 2)} rad", "—(젯슨 미배포 키)",
                  OK if wv and max(wv.values()) < TOL_OBS and ms.get("act_err", 1) < TOL_ACT else BAD, "하위 러너는 브리지 _make_runner(하위 키) 그대로 — 항목별 판정은 그 키의 표", cover="브리지 selftest 'KE↔W3 합성'")
    if not in_jet:
        nd = H["node"].get(key) or {}
        nmh = ctx["nodemeas"].get(key) or {}
        acks = nd.get("acks", [])
        kws = nd.get("kew3") or {}
        okn = (nd.get("rc") == 0 and nmh.get("chain_err_rad", 1) < 1e-5 and nmh.get("blend_err_deg", 1) < 1e-6 and nmh.get("pub_rel_err_deg", 1) < 1e-9
               and all(a_.get("accepted") for a_ in acks if a_.get("cmd") in ("stand", "step")))
        R.add("6 합성" if is_kew3 else "2 행동", "노드 세션(가짜 ROS 로 main() 그대로, HEAD)", "학습 식 재구성: 슬루 → LPF β → 스무스스텝 블렌딩 → 모터 프레임",
              f"정책 틱 {nmh.get('n_policy_ticks')}(연속 {nmh.get('n_seg')}) · 체인 {fnum(nmh.get('chain_err_rad'), 2)} rad · 블렌딩 {fnum(nmh.get('blend_err_deg'), 2)}° · 발행 = 목표 − 기준선 {fnum(nmh.get('pub_rel_err_deg'), 2)}°"
              f" · 명령 응답 {sum(1 for a_ in acks if a_.get('accepted'))}/{len(acks)} 수락" + (f" · 전환 KE→W3 {kws.get('n_to_w3')} 회·W3→KE {kws.get('n_to_ke')} 회" if kws else ""),
              "—(젯슨 미배포 키)", OK if okn else (UNK if not nmh or "error" in nd else BAD),
              ("오류: " + str(nd.get("error"))) if "error" in nd else "서기 → 걷기(앞 0.06, 헤딩 고리 켬) → 정지 → 걷기(고리 끔) → 정지. 플랜트는 가정 모형 — 계약·순서 검증이지 폐루프 성능이 아니다",
              cover="selftest_w3_bridge_node(스텁 정책)")

    # ------------------------------------------------------------------ 7. 젯슨 판
    if in_jet:
        jb = ctx["jet_build"]
        bit = ctx["bit"].get(key, {})
        man = jb.get("manifest", {})
        mb = man.get("rl_bridge_node.py", {}).get("deploy/c5krh_1007.md5")
        mr = next(iter(man.get("rl_walking/deploy/policy_runner_stage4.py", {}).values()), None)
        mc = man.get("rl_walking/deploy/c5_switch.py", {}).get("deploy/c5kr_1005.md5")
        okf = (mb == jb.get("bridge_md5")) and (mr == jb.get("runner_md5")) and (mc == jb.get("c5_md5")) and jb.get("patch_reproduces") is not False
        R.add("7 젯슨 판", "재구성한 파일 = 저장소가 기록한 배포본", "—", "(HEAD 는 젯슨 미배포 변경 다수)",
              f"브리지 md5 {jb.get('bridge_md5', '')[:8]} {'=' if mb == jb.get('bridge_md5') else '≠'} c5krh_1007.md5 · 러너 {jb.get('runner_md5', '')[:8]} {'=' if mr == jb.get('runner_md5') else '≠'} 매니페스트 · c5_switch {jb.get('c5_md5', '')[:8]} {'=' if mc == jb.get('c5_md5') else '≠'} · {JET_COMMIT} + 패치 = 후보 {jb.get('patch_reproduces')}",
              OK if okf else BAD, "젯슨의 현재 파일을 읽은 것은 아니다(접속 금지) — 배포 기록 기준", cover="deploy_c5krh_1007.sh(md5 확인 절차)")
        dif = ctx["reg_diff"].get(key, [])
        R.add("7 젯슨 판", "레지스트리·상수 값 HEAD 와 같은가", "—", f"비교 {ctx['reg_n']} 항목", "전부 같음" if not dif else "다름: " + "; ".join(dif), OK if not dif else BAD,
              "경로·감지기 설정·cue·기본자세·클록·EMA 모드·env.yaml 요구·블렌딩·LPF·슬루·게인·틱", cover="브리지 selftest 'C5 krh 키'(kr 대비)")
        R.add("7 젯슨 판", "러너 프로브(600 틱) HEAD 와 비트 비교", "—", "기준", f"관측 {bit.get('obs')} · 원시 행동 {bit.get('raw')} · 목표각 {bit.get('out')} · 회복 모드열 {bit.get('mode')}",
              OK if bit.get("obs") and bit.get("raw") and bit.get("out") and bit.get("mode") else BAD, cover="selftest_odic_equiv [A]·[C](기준 커밋 비트 비교, 다른 기준)")
        R.add("7 젯슨 판", "노드 세션(가짜 ROS, main() 그대로) 발행 HEAD 와 비트 비교", "—",
              f"정책 틱 {ctx['nodemeas'].get(key, {}).get('n_policy_ticks')}" + (f"(회복 모드 {ctx['nodemeas'].get(key, {}).get('mode_ticks')} 틱)" if is_c5 else ""),
              f"발행 목표 {bit.get('node')} · 관측 {bit.get('node_obs')}",
              OK if bit.get("node") and bit.get("node_obs") else (UNK if bit.get("node") is None else BAD),
              "인계·블렌딩·LPF·유한차분 속도(머문 틱 포함)·기울기 펄스(회복 전환) 구간", cover="selftest_w3_bridge_node(기준 커밋 비교, 다른 기준)")
        for rel_ in [r_ for r_ in REAL_LOGS if (H.get("replay") or {}).get(os.path.basename(r_), {}).get("key") == key]:
            rp, rpj = (H.get("replay") or {}).get(os.path.basename(rel_)) or {}, ((J or {}).get("replay") or {}).get(os.path.basename(rel_)) or {}
            rls_ = (ctx.get("reallogs") or {}).get(rel_) or {}
            if not rp.get("registered"):
                continue
            def rtxt(r):
                if not r or r.get("error") or not r.get("registered"):
                    return "재생 실패: " + str((r or {}).get("error", "키 없음"))
                return (f"관측 최대 |Δ| {fnum(max(r['obs_err'].values()), 2)} · 원시 행동 {fnum(r['raw_err'], 2)} · LPF 뒤 목표 {fnum(r['tgt_deg_err'], 2)}°"
                        + (f" · 회복 모드 불일치 {r['mode_mismatch']}/{r['n']} 틱 · 기준 자세 {fnum(r.get('ref_deg_err'), 2)}° · 켜짐 {r.get('n_on')}/{r.get('log_n_on')}" if r.get("is_c5") else f" · {r['n']} 틱"))
            okr = all(r and r.get("registered") and max(r["obs_err"].values()) < 2e-4 and r["raw_err"] < 2e-3 and r["tgt_deg_err"] < 0.02 and r["mode_mismatch"] == 0
                      for r in (rp, rpj))
            cur_ed = (rel_ == REAL_LOG_REL)
            R.add("7 젯슨 판", f"실물 블랙박스 재생 — {os.path.basename(os.path.dirname(rel_))}(젯슨이 실제로 낸 값과 대조)",
                  f"기록 {rp.get('n')} 틱({rls_.get('dur_s', 0):.1f} s), 기록 반올림 1e-4" + ("" if cur_ed else " — 그때의 젯슨 판(현 판 이전)이 낸 기록"), rtxt(rp), rtxt(rpj),
                  OK if okr else BAD,
                  "젯슨 추론(백엔드는 기록에 없다 — 문서상 onnxruntime) 대 PC(torch): " + (f"기록 관측을 그대로 넣은 서기 정책 출력 최대 |Δ| {fnum(rp.get('net_stand_err'), 2)}, 회복 정책 {fnum(rp.get('net_rec_err'), 2)}(관측 반올림 한도). " if rp.get("is_c5") else "")
                  + "관측·클록·필터·직전 행동·슬루·LPF 가 같은 입력에서 같은 값을 낸다는 계약 검증이며 폐루프 성능 증명이 아니다",
                  cover="reports/gen_c5/replay_kr*.py(이전 로그, 감지기만)")
            R.add("7 젯슨 판", f"실물 기록의 배포 체인 값 — {os.path.basename(os.path.dirname(rel_))}", f"학습 온셋 블렌딩 상한 {fnum((tr0.act.get('onset_blend_range') or [0, 0])[1])} s · 관측 필터 β ({lpf_b}, {bj_tr})",
                  "—", f"블렌딩 α(t) 가장 잘 맞는 길이 {rls_.get('blend_s_best')} s(오차 {fnum(rls_.get('blend_alpha_err'), 2)}) · 기록 obs_filter {rls_.get('obs_filter')} · jvel_ema {rls_.get('jvel_ema_mode')} · 장착 편향 {rls_.get('imu_bias')}° · 명령 최대 |값| {rls_.get('cmd_absmax')}"
                  f" · 틱 간격 평균 {rls_.get('dt_mean_ms', 0):.2f} ms(최대 {rls_.get('dt_max_ms', 0):.1f})",
                  OK if rls_ and rls_.get("blend_s_best") == regH["BLEND_S"] and rls_.get("obs_filter") == [att.get("act_beta"), att.get("jv_beta")] and rls_.get("cmd_absmax") == 0.0 else BAD,
                  cover="없음")
        opts = []
        if att.get("has_ema_fresh_opt") and not attJ.get("has_ema_fresh_opt"):
            opts.append("--runner-ema-fresh")
        if is_c5 and (PH.get("det") or {}).get("has_imu_f32_opt") and not (PJ.get("det") or {}).get("has_imu_f32_opt"):
            opts.append("--c5-imu-f32·--c5-settle(odic)·requiet")
        R.add("7 젯슨 판", "HEAD 에만 있는 선택 기능", "—", ", ".join(opts) or "없음", "없음(기본 꺼짐 기능이라 기본 동작은 위 비트 비교대로 같다)", OK,
              f"후보 브리지 → HEAD 브리지: 바뀌거나 빠진 줄 {ctx.get('bridge_removed')} · 더해진 줄 {ctx.get('bridge_added')}", cover="없음")
    return R


def render(key, R, md=True):
    out = [f"### {key}", "", "| 분류 | 항목 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 판정 | 비고 | 기존 검증 |", "|---|---|---|---|---|---|---|---|"]
    esc = lambda s: str(s).replace("|", "¦").replace("\n", " ")  # noqa: E731
    for r in R.rows:
        v = r["verdict"] + (f" [{r['known']}]" if r["known"] and r["verdict"] == BAD else "")
        out.append("| " + " | ".join(esc(x) for x in (r["cat"], r["item"], r["train"], r["head"], r["jet"], v, r["note"], r["cover"])) + " |")
    c = R.count()
    out += ["", f"집계: 일치 {c[OK]} · 불일치 {c[BAD]} · 확인 불가 {c[UNK]}", ""]
    return "\n".join(out)


def render_matrix(all_rows):
    """항목 × 정책 판정표(값 없이 판정만): ○ 일치 · ✕ 불일치 · ? 확인 불가 · – 해당 없음."""
    sym = {OK: "○", BAD: "✕", UNK: "?"}
    order, seen = [], set()
    for key in POLICY_KEYS:
        for r in all_rows[key].rows:
            k = (r["cat"], r["item"])
            if k not in seen:
                seen.add(k)
                order.append(k)
    order.sort(key=lambda k: k[0])                      # 분류 번호순(같은 분류 안에서는 처음 나온 순서 유지 — 안정 정렬)
    short_ = [k.replace("stand_", "").replace("_g21a_g24ra", "").replace("crouch_", "") for k in POLICY_KEYS]
    out = ["| 분류 | 항목 | " + " | ".join(short_) + " | 기존 검증(없음 = 이 감사가 처음) |", "|---|---|" + "---|" * (len(POLICY_KEYS) + 1)]
    for cat, item in order:
        cells, cover = [], ""
        for key in POLICY_KEYS:
            r = next((x for x in all_rows[key].rows if x["cat"] == cat and x["item"] == item), None)
            cells.append("–" if r is None else sym[r["verdict"]])
            cover = cover or (r["cover"] if r else "")
        out.append(f"| {cat} | {item.replace('|', '¦')} | " + " | ".join(cells) + f" | {(cover or '없음').replace('|', '¦')} |")
    return "\n".join(out)


def render_compact(all_rows):
    """보고서용 정책별 표(5 칸). 기준 정책 표와 완전히 같은 줄은 생략하고 개수만 적는다."""
    refs = {POLICY_KEYS[0]: [], POLICY_KEYS[1]: [POLICY_KEYS[0]], POLICY_KEYS[2]: [POLICY_KEYS[1]], POLICY_KEYS[3]: [POLICY_KEYS[0]],
            POLICY_KEYS[4]: [POLICY_KEYS[1], POLICY_KEYS[3]]}
    esc = lambda x: str(x).replace("|", "¦").replace("\n", " ")  # noqa: E731
    out = []
    for key in POLICY_KEYS:
        R = all_rows[key]
        no_jet = all(r["jet"].startswith("—") for r in R.rows)
        ident = lambda r: (r["cat"], r["item"], r["train"], r["head"], r["verdict"]) + (() if no_jet else (r["jet"],))  # noqa: E731
        base = set()
        for rk in refs[key]:
            nj = all(r["jet"].startswith("—") for r in all_rows[rk].rows)
            base |= {(r["cat"], r["item"], r["train"], r["head"], r["verdict"]) + (() if (no_jet or nj) else (r["jet"],)) for r in all_rows[rk].rows}
        rows = [r for r in R.rows if ident(r) not in base]
        c = R.count()
        out += [f"### {key}", "", f"일치 {c[OK]} · 불일치 {c[BAD]} · 확인 불가 {c[UNK]}"
                + (f" — 아래는 {'·'.join(refs[key])} 표와 다른 {len(rows)} 줄(같은 {len(R.rows) - len(rows)} 줄 생략" + (", 젯슨 판 칸은 전부 '젯슨 미배포 키'" if no_jet else "") + ")" if refs[key] else ""), "",
                "| 분류 | 항목 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 판정 |", "|---|---|---|---|---|---|"]
        for r in rows:
            v = r["verdict"] + (f" [{r['known'].split('(')[0]}]" if r["known"] and r["verdict"] == BAD else "")
            out.append("| " + " | ".join(esc(x) for x in (r["cat"], r["item"], r["train"], r["head"], "—" if no_jet else r["jet"], v)) + " |")
        out.append("")
    return "\n".join(out)


#: 기존 계약 관련 검증(--run-existing 으로 재실행). (이름, 인자, 무엇을 고정하나)
EXISTING = [
    ("러너 --selftest", ["rl_walking/deploy/policy_runner_stage4.py", "--selftest"], "245 골든 조립(항 블록·이력 오래된 것 먼저·첫 틱 백필)·클록 식·슬루·직전 행동 = 원시"),
    ("deploy_chain --selftest", ["rl_walking/deploy/deploy_chain.py", "--selftest"], "러너 슬루 → LPF → 블렌딩(벡터화 = 스칼라 참조)·순서 감수성"),
    ("c5_switch --selftest", ["rl_walking/deploy/c5_switch.py", "--selftest"], "감지기 히스테리시스·조기 복귀·불응기·cue·odic·갇힘 해소(배포 구현 내부 일관성)"),
    ("kew3_runner --selftest", ["rl_walking/deploy/kew3_runner.py", "--selftest"], "합성 러너 전환 틱·이어받기·정착 대기·계약 불일치 거부(스텁 정책)"),
    ("브리지 --selftest", ["jetson/scripts/rl_bridge_node.py", "--selftest"], "30 항목: '학습 계약'(기본자세·스케일·슬루·50 Hz·default_offset, 서기 run)·'관측 필터 계약'·C5·W3·합성 등록·정책 500 틱"),
    ("브리지 --selftest(합성 키 기동 인자)", ["jetson/scripts/rl_bridge_node.py", "--selftest", "--experimental-w3", "--experimental-kew3", "--checkpoint",
                                   "stand_kew3_g21a_g24ra_w3s3d15800"], "같은 30 항목 + 합성·하위 키를 필수로"),
    ("selftest_w3_equiv", ["rl_walking/deploy/selftest_w3_equiv.py"], "W3 클록·래치·리드 발·헤딩 고리 = 시뮬 소스 실행([B]·[E]·[F]), 플래그 꺼짐 비트 동일([A]·[C])"),
    ("selftest_w3_rollout_parity", ["rl_walking/deploy/selftest_w3_rollout_parity.py"], "W3S3d 시뮬 롤아웃 기록 원천 → 러너·브리지 코어 관측 320·행동·체인([P1]–[P6])"),
    ("selftest_odic_equiv", ["rl_walking/deploy/selftest_odic_equiv.py"], "감지기 기본(imu) 경로 기준 커밋 비트 동일([A]·[C])·odic = 시뮬 평가 소스 실행([B])"),
    ("selftest_c5_imu_f32 --ref 870b17a3~1", ["rl_walking/deploy/selftest_c5_imu_f32.py", "--ref", "870b17a3~1"], "KE imu 조기 복귀: 시뮬 평가 소스 실행 vs 배포(float32 옵션 0 불일치·기본 float64 는 −1 틱)"),
    ("selftest_w3_bridge_node", ["jetson/scripts/selftest_w3_bridge_node.py"], "가짜 ROS 로 노드 main(): W3·합성 세션 명령·감독기·지연 동결, 플래그 꺼짐 발행 = 기준 커밋(스텁 정책)"),
]

BLIND_SPOTS = [
    "관측 클립 — 어떤 기존 검증도 다루지 않는다(이 감사가 처음: 1 절의 '클립').",
    "서기·회복(G21a·G24Ra) 관측 320 의 시뮬 대조 — 롤아웃 parity([P1]–[P4])는 W3S3d 만이다. 시간식 클록(1.1 Hz)·C5 공유 조립 경로는 시뮬 원천 대조가 없었다 → 이 감사: 세 run 관측 설정 동일성 + 시뮬 소스 실행 + 학습 계약 재구성 + 실물 재생.",
    "회복 정책(G24Ra) run 의 env.yaml 자동 검사 — 브리지 '학습 계약'은 서기 run(G21a)만 읽는다(C5 키의 경로 = 서기 정책). 종전에는 주석의 수동 확인뿐.",
    "G21a·G24Ra 의 policy.onnx(젯슨 추론 경로) — onnx 대 pt 시험은 W3S3d 만 있고 Isaac 파이썬이 필요하다 → 이 감사: 세 run 모두 체크포인트 = pt = onnx 가중치 비트 비교(패키지 불필요) + 실물 기록의 원시 행동 재생.",
    "감지기 설정값 ↔ 시뮬 평가 인자의 자동 대조 — 종전에는 브리지 주석으로만 대응시켰다 → 이 감사: 평가 산출물 머리의 '인자:' 줄과 실제 감지기 객체를 대조.",
    "젯슨 판 ↔ HEAD 직접 비교 — 기존 비트 비교들은 기준 커밋이 제각각(e8489de4·b58a39cb·443ef820·8995a833 …)이고 젯슨 판을 기준으로 삼은 것이 없다.",
    "실물 블랙박스로 관측·행동 전체를 재생하는 상시 도구 — 감지기 재생(reports/gen_c5/replay_kr*.py)만 있었다.",
    "노드(_tick) 수준의 LPF·블렌딩·유한차분을 실모델로 재구성하는 시험 — 스텁 정책의 기준 커밋 비트 비교만 있었다.",
    "stage8 식 전 관절 soft 클램프·25° 거부가 들어간 **KE 회복** 시뮬 평가 — W3 걷기만 있다(PARITY §4.4). 실물 1007 에서는 회복 중 힙이 잘렸다(1 절의 '소프트 한계 클램프').",
    "W3S3d 의 지연 0 틱 평가 — 학습 1–5, 평가 1–7 틱. 실물 명령 전송 구간은 한 틱 안으로 측정됐다(2 절의 '행동 지연').",
    "KE·전환 평가의 자이로 지연 — 0 틱 조건뿐(2 절).",
    "합성 러너 + 헤딩 고리의 폐루프 평가 — 없음(KEW3_DESIGN §8.3 에 이미 적힘).",
]

LIMITS = [
    "**젯슨의 현재 파일·프로세스를 읽지 않았다**(접속 금지). '젯슨 판'은 저장소의 배포 기록(커밋 e7710c63, 후보 브리지, md5 매니페스트)으로 재구성한 것이다. 실물 블랙박스 재생이 그 재구성과 맞는 것은 1003(G21a)·1007(krh) 두 세션에 한한다 — 저장소에 c5kr 키의 실물 기록은 없다"
    "(1004·1005 의 기록 6 개는 앞선 키 stand_c5_g21a_g24ra 의 것이고 이 감사의 대상 키가 아니어서 재생하지 않았다).",
    "젯슨의 **stage8 노드 판**은 확인 범위 밖이다(지시서는 브리지·러너만 지정). HEAD 와 커밋 e7710c63 의 stage8 소스에서 슬루·주기·클램프·거부 규칙이 같음을 읽었고, "
    "실물 1007 기록의 클램프 누계가 그 규칙(soft 초과 틱마다 한 번)과 맞는 것만 확인했다.",
    "민감도 시험은 **배포 쪽** 결함만 심는다. 학습 쪽 값을 잘못 읽는 종류의 감사 결함은 '일치' 줄이 실제 러너 값과 맞아떨어지는 것으로만 간접 확인된다.",
    "열린 고리 계약 검증이다. 같은 입력 → 같은 관측·행동임을 보였을 뿐, 실물 센서의 분포(잡음·지연·패드 오판)가 학습 분포 안인지와 폐루프 성능은 말하지 못한다.",
    "학습 쪽 관절 순서·이력 배열 순서의 근거는 W3S3d 태스크에서 남긴 시뮬 기록(log_picture/w3parity_rec_*.npz) 하나다. G21a·G24Ra 는 같은 로봇 자산(usd_path)·같은 관측 설정이라는 점으로 잇는다 — 그 두 태스크에서 직접 남긴 시뮬 기록은 없다.",
    "노드 세션의 플랜트는 '발행 목표를 1 차 지연으로 따라가는' 가정 모형이고 접촉은 항상 양발이다. LPF·블렌딩·유한차분·발행 식의 재구성에만 쓴다.",
    "학습 쪽 값 가운데 env.yaml 에 없는 것(필터 함수의 시작 규칙, 행동 체인 순서, 소프트 클램프 관절 기본값, w3_gait_update 의 ramp_s·리드 발 상수)은 소스에서 읽는다. "
    "기준은 **학습 시작 시점 커밋**이다 — 단 rsl_rl 은 커밋 해시를 남기지 않아 run 폴더 이름의 시각 직전 커밋으로 추정했다(run 이 남긴 git status·diff 에 rl_walking 변경이 없음은 확인). "
    "그 커밋 이후 바뀐 계약 함수는 '학습 시점의 시뮬 소스' 줄에 적었고, 차이를 읽어 확인한 것만 통과시킨다(함수가 다시 바뀌면 '미검토'로 떨어진다).",
    "실물 블랙박스는 관측 4 자리·쿼터니언 5 자리로 반올림돼 있다. 재생 허용오차(관측 2e-4·원시 행동 2e-3)보다 작은 계약 차이는 이 재생으로 보이지 않는다(그런 차이는 프로브·비트 비교가 맡는다).",
    "젯슨 추론 백엔드(onnxruntime / torch)는 블랙박스에 기록되지 않는다. 가중치 비트 비교와 원시 행동 재생으로 어느 쪽이든 같은 출력임을 보였다.",
    "평가 산출물·시뮬 기록·실물 블랙박스는 저장소의 `log_picture/`·`reports/real_*` 파일에 기댄다(워크트리에 없으면 주 체크아웃에서 읽는다). 파일이 없으면 해당 줄은 '확인 불가'로 떨어진다.",
]


def run_existing():
    """기존 selftest 재실행(CPU). 저장소 logs/ 가 있어야 한다."""
    td = tempfile.mkdtemp(prefix="ca_existing_")      # 자식들이 남기는 임시 파일은 여기에 — 끝나면 지운다
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="", TMPDIR=td)
    out = []
    try:
        for name, argv, what in EXISTING:
            t0 = time.time()
            r = subprocess.run([sys.executable] + argv, cwd=REPO, env=env, capture_output=True)
            txt = r.stdout.decode(errors="replace")
            last = ""
            for ln in txt.splitlines():
                if re.search(r"ALL PASS|FAILED|selftest:|C5 selftest", ln):
                    last = ln.strip()
            out.append((name, r.returncode, time.time() - t0, last[:160], what, txt.count("[PASS]"), txt.count("[FAIL]")))
    finally:
        shutil.rmtree(td, ignore_errors=True)
    return out


def run_mutants():
    """민감도 시험 실행 — (종료 코드, 결과 dict). 결과 JSON 은 임시 파일로 받는다."""
    fd, pj = tempfile.mkstemp(prefix="ca_mut_", suffix=".json")
    os.close(fd)
    try:
        r = subprocess.run([sys.executable, os.path.join(HERE, "contract_audit_mutants.py"), "--json", pj], cwd=REPO, capture_output=True)
        try:
            with open(pj, encoding="utf-8") as f:
                res = json.load(f)
        except (OSError, ValueError):
            res = {"control": None, "mutants": [], "error": (r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace"))[-800:]}
        return r.returncode, res
    finally:
        try:
            os.remove(pj)
        except OSError:
            pass


def render_report(ctx, all_rows, summary, logs, existing=None, mutants=None, elapsed=0.0):
    """보고서(CONTRACT_AUDIT.md) 전체를 자료에서 만든다 — 숫자를 손으로 옮기지 않는다."""
    H, J = ctx["H"], ctx["J"]
    esc = lambda x: str(x).replace("|", "¦").replace("\n", " ")  # noqa: E731
    sk = lambda k: k.replace("stand_", "")  # noqa: E731
    head = git("rev-parse", "--short", "HEAD").strip()
    bad, unk = {}, {}
    for key in POLICY_KEYS:
        for r in all_rows[key].rows:
            tgt = bad if r["verdict"] == BAD else unk if r["verdict"] == UNK else None
            if tgt is not None:
                g = tgt.setdefault((r["cat"], r["item"]), {"r": r, "keys": [], "rows": {}})
                g["keys"].append(key)
                g["rows"][key] = r

    def merged(v, fld, numeric_ok=False):
        """키마다 값이 다르면 병기한다. 한 값이 나머지의 접두이면(뒤에 덧붙인 꼴) 가장 긴 것 하나만."""
        vals = {}
        for k in v["keys"]:
            vals.setdefault(v["rows"][k][fld], []).append(sk(k))
        if len(vals) == 1 or numeric_ok:
            return v["r"][fld]
        longest = max(vals, key=len)
        if all(longest.startswith(x.rstrip()) for x in vals):
            return longest
        return " / ".join(f"〔{', '.join(ks)}〕 {val}" for val, ks in vals.items())

    def find_row(key, cat_prefix, item_prefix):
        return next((r for r in all_rows[key].rows if r["cat"].startswith(cat_prefix) and r["item"].startswith(item_prefix)), None)

    n_new = sum(1 for v in bad.values() if v["r"]["known"].startswith("신규"))
    n_known = sum(1 for v in bad.values() if v["r"]["known"].startswith("기지"))
    n_untag = len(bad) - n_new - n_known
    bit = ctx.get("bit") or {}
    jet_bit_ok = bool(bit) and all(all(bool(v) for v in b.values() if v is not None) for b in bit.values())
    rps = [(nm, r) for nm, r in (H.get("replay") or {}).items() if isinstance(r, dict) and r.get("registered")]
    L = ["# 시뮬 ↔ 젯슨 배포 계약 감사 (2026-10-07 밤)", "",
         f"> 자동 생성: `python3 reports/night_1007/contract_audit.py --report <파일> --run-existing --run-mutants` · 기준 커밋 {head} · {time.strftime('%Y-%m-%d %H:%M')}. 숫자는 전부 그 실행의 산출이다(손으로 옮긴 값 없음).", "",
         "지시: \"시뮬과 Jetson 배포 코드의 observation/action/filter/history/previous-action/phase/supervisor 설정이 동일한지도 확인하세요.\"", "",
         "- 대상 브리지 키: " + " · ".join(f"`{k}`" for k in POLICY_KEYS),
         "- 학습 쪽 = 각 run 의 `params/env.yaml`·`agent.yaml`·체크포인트·학습 로그 + 학습 시점 커밋의 시뮬 소스(읽기·실행). "
         "감지기·감독기·합성 러너처럼 **학습에 없는 층**은 그 층을 검증한 시뮬 평가의 값(평가 산출물 머리에 찍힌 인자)을 '학습 값' 칸에 적었다.",
         "- 배포 쪽 = 브리지 `_make_runner(키)` 로 만든 **실제 러너 객체**의 입출력 + 가짜 ROS 로 돌린 노드 `main()` + 젯슨이 남긴 실물 블랙박스의 재생.",
         f"- HEAD 와 젯슨 판(커밋 {JET_COMMIT} 의 러너·c5_switch + `{JET_BRIDGE_REL}`)을 따로 올려 같은 표로 비교했다.",
         "- 젯슨 접속·ROS·CAN·GPU·Isaac 실행 없음. 기존 코드 수정 없음. **계약 검증(열린 고리)이며 폐루프 성능·실물 성공의 증명이 아니다**(CLAUDE.md 6).", "",
         "## 0. 결론 먼저", ""]
    L.append(f"- 불일치는 중복을 빼면 **{len(bad)} 줄**이다: 종전 문서에 있던 것 {n_known}, 이 감사가 새로 찾은 것 {n_new}" + (f", **어느 문서에도 없는 것 {n_untag}(회귀 신호)**" if n_untag else "")
             + f". 확인 불가는 {len(unk)} 줄 — 현장 측정이 있어야 하는 것과 평가 조건이다.")
    for (cat, item), v in bad.items():
        if v["r"]["known"].startswith("신규"):
            L.append(f"- **새로 찾은 불일치 — {item}**({', '.join(sk(k) for k in v['keys'])}): 학습 `{v['r']['train']}` ↔ 배포 {v['r']['head'].split('(')[0].strip()}. 정상 범위에서는 두 쪽 값이 같고, 범위 밖(낙상·폭주·센서 튐)에서만 갈린다 — 1 절.")
    if ctx.get("clamp_not_trained"):
        L.append("- **종전 문서에 '클램프 누계 증가'로만 적혀 있던 소프트 한계 클램프를 관절별로 풀었다**: 학습은 발목 앞뒤만 자르는데 실물 stage8 은 전 관절을 자른다. "
                 f"지면에 서 있던 구간에서 학습이 자르지 않는 관절이 잘린 틱 — {ctx['clamp_not_trained']}. "
                 + (f"soft 범위는 좌우로 다르다(왼/오른: {ctx['soft_asym_txt']}). " if ctx.get("soft_asym_txt") else "")
                 + "KE 회복 시뮬 평가에는 이 클램프가 없다(5 절 빈칸 9) — 영향은 대조 평가가 없어 모른다.")
    L.append("- 관측 320 의 항 순서·이력·필터 계수·직전 행동 정의·클록, 행동 스케일·기본자세·슬루·LPF·블렌딩, 감지기 설정, 합성 러너 인계 규칙은 학습(또는 시뮬 평가)과 일치한다 — 속성값을 읽은 것이 아니라 러너 입출력에서 역산했다(3 절).")
    if J is not None:
        rd = {k: v for k, v in (ctx.get("reg_diff") or {}).items() if v}
        L.append(f"- **젯슨 판은 {', '.join(sk(k) for k in JET_KEYS)} 에서 HEAD 와 " + ("관측·원시 행동·목표각·노드 발행이 비트 동일하다" if jet_bit_ok else "비트가 다르다 — 부록의 '7 젯슨 판' 줄 참조")
                 + "**. 레지스트리·상수 " + ("차이 없음" if not rd else "차이: " + str(rd)[:300]) + ".")
    else:
        L.append("- **젯슨 판을 재구성하지 못했다** — " + str(ctx["jet_build"].get("error") or ctx["jet_build"].get("worker_error", ""))[:300])
    for nm, r in rps:
        rj = ((J or {}).get("replay") or {}).get(nm) or {}
        L.append(f"- 실물 블랙박스 재생 `{nm}`({sk(r.get('key', ''))}, {r.get('n')} 틱 = {r.get('n', 0) * 0.02:.0f} s): HEAD 관측 최대 |Δ| {fnum(max(r['obs_err'].values()), 2)} · 원시 행동 {fnum(r['raw_err'], 2)} · LPF 뒤 목표 {fnum(r['tgt_deg_err'], 2)}°"
                 + (f" · 회복 모드 불일치 {r['mode_mismatch']} 틱" if r.get("is_c5") else "")
                 + (f" / 젯슨 판 {fnum(max(rj['obs_err'].values()), 2)} · {fnum(rj['raw_err'], 2)} · {fnum(rj['tgt_deg_err'], 2)}°" if rj.get("registered") else "") + " (기록 반올림 1e-4 한도).")
    L += ["", "| 브리지 키 | 일치 | 불일치 | 그중 종전 문서에 있던 것 | 그중 이번에 찾은 것 | 꼬리표 없음 | 확인 불가 |", "|---|---|---|---|---|---|---|"]
    for key, c in summary:
        b = [r for r in all_rows[key].rows if r["verdict"] == BAD]
        kn, nw = sum(1 for r in b if r["known"].startswith("기지")), sum(1 for r in b if r["known"].startswith("신규"))
        L.append(f"| `{key}` | {c[OK]} | {c[BAD]} | {kn} | {nw} | {c[BAD] - kn - nw} | {c[UNK]} |")
    L += ["", "건수는 표의 줄 수다(한 원인이 여러 줄·여러 키에 걸린다). 스크립트 종료 코드는 불일치가 있으면 1 이다. `--allow-known` 은 꼬리표가 붙은(이 문서에 적힌) 불일치만 있을 때 0 — 새 불일치가 생기면 다시 1.", "",
          "완료 구분: **오프라인 검증 완료** = 이 문서의 일치·불일치 판정 전부 / **미검증** = 5 절의 남은 빈칸(평가가 없는 조건) / **현장 확인 필요** = 2 절.", "",
          f"## 1. 불일치 ({len(bad)} 줄)", "",
          "비고 칸은 어느 쪽이 학습과 같은지로 시작한다. 측정이 아닌 것은 (추측)으로 적었다. 조치는 하지 않았다(감사 범위).", "",
          "| # | 항목 | 해당 키 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 비고·영향 | 근거(파일:줄) | 구분 |", "|---|---|---|---|---|---|---|---|---|"]
    for i, ((cat, item), v) in enumerate(bad.items(), 1):
        r = v["r"]
        jt = next((x["jet"] for k in v["keys"] for x in all_rows[k].rows if x["cat"] == cat and x["item"] == item and x["jet"] and not x["jet"].startswith("—")), "—")
        L.append(f"| {i} | **{esc(item)}** | {', '.join(sk(k) for k in v['keys'])} | {esc(merged(v, 'train'))} | {esc(merged(v, 'head', numeric_ok=True))} | {esc(jt)} | {esc(merged(v, 'note'))} | {esc(r['ev'])} | {esc(r['known'] or '꼬리표 없음')} |")
    L += ["", "결정이 필요한 것(이 감사는 고치지 않았다):", "",
          "- 관측 클립: 러너 조립부에 학습과 같은 클립을 넣으면 일치한다. 정상 범위에서는 출력이 바뀌지 않는다(6 절 민감도 시험의 반대 방향 대조 `obs_clip_added`). 젯슨 배포 파일이 바뀌므로 승인 사항이다.",
          "- 소프트 한계: 0905 감사 B7 이 \"soft 표를 stand 세대 기준으로 재검토\"를 권고했다. 그 뒤 한계 파일(config/joint_limits_12dof.json)의 변경은 "
          + (f"{len(ctx.get('limits_since_0905') or [])} 건({'; '.join(ctx.get('limits_since_0905') or []) or '없음'})" if ctx.get("limits_since_0905") is not None else "확인하지 못했다")
          + " — 힙·무릎을 재검토했는지는 기록에서 확인하지 못했다. "
          "선택지 ① KE 회복 시뮬 평가에 stage8 과 같은 전 관절 클램프·25° 거부를 넣어 영향부터 재기 ② 실물 기구 한계를 확인한 뒤 soft 표 재검토(안전 한계이므로 성공률을 높이려는 목적으로 넓히지 않는다 — CLAUDE.md).", "",
          f"## 2. 확인 불가 ({len(unk)} 줄) — 왜 못 했고 무엇이 필요한가", "",
          "| 항목 | 해당 키 | 학습 값 | 배포 값 | 이유·필요한 것 |", "|---|---|---|---|---|"]
    for (cat, item), v in unk.items():
        r = v["r"]
        L.append(f"| **{esc(item)}** | {', '.join(sk(k) for k in v['keys'])} | {esc(merged(v, 'train'))} | {esc(merged(v, 'head', numeric_ok=True))} | {esc(merged(v, 'note'))} |")
    L += ["", "## 3. 일치 — 핵심 근거(측정값)", "",
          "판정표(4 절)의 ○ 가운데 지시서의 일곱 낱말(observation · action · filter · history · previous-action · phase · supervisor)과 젯슨 판에 바로 답하는 줄만 뽑았다. 값은 실제 객체의 입출력에서 잰 것이다. 전체는 부록 A.", "",
          "| 무엇 | 키 | 학습 값 | 배포 값(HEAD) | 배포 값(젯슨 판) | 판정 |", "|---|---|---|---|---|---|"]
    G, KR, KRH, W, KW = POLICY_KEYS
    PICK = [("observation", G, "1", "항 목록·순서·프레임 차원"), ("observation", G, "1", "관측 벡터 전체 재구성"),
            ("observation", KRH, "1", "구성 정책들의 학습 관측 설정이 서로 같은가"), ("observation", G, "1", "스케일"),
            ("observation", G, "1", "projected_gravity 정의"), ("observation", G, "1", "joint_pos = 엔코더각"), ("observation", G, "1", "feet_contact 순서·값"),
            ("observation", G, "1", "명령 관측(서기 정책)"),
            ("history", G, "1", "이력 길이"), ("history", G, "1", "이력 배열 순서·리셋 첫 틱"),
            ("filter", G, "1", "joint_vel = 엔코더 유한차분 속도의 EMA"), ("filter", G, "1", "gyro_ema 계수"),
            ("filter", G, "1", "joint_vel_ema 계수"), ("filter", G, "2", "액션 LPF β"),
            ("previous-action", G, "1", "actions(직전 행동) 정의·필터"),
            ("action", G, "2", "관절 순서(12)"), ("action", G, "2", "차원·행동 클립"), ("action", G, "2", "스케일"), ("action", G, "2", "오프셋 = 기본자세"),
            ("action", G, "2", "슬루 상한"), ("action", G, "2", "슬루 기준 초기화"), ("action", G, "2", "목표각 재구성"), ("action", G, "2", "처리 순서"),
            ("action", G, "2", "온셋 블렌딩(인계)"), ("action", G, "2", "PD 게인"), ("시간", G, "3", "제어 주기"), ("시간", G, "3", "접촉 갱신 주기"),
            ("phase", G, "1", "gait_clock(서기 정책)"), ("phase", W, "4", "스텝 주파수 f_step"), ("phase", W, "4", "관측 게이트 램프"),
            ("phase", W, "4", "gait_enable 래치"), ("phase", W, "4", "리드 발 규칙 문턱"),
            ("supervisor", W, "1", "명령 관측 범위(걷기)"), ("supervisor", W, "5", "걷기 켜기 선행조건"),
            ("supervisor", KR, "5", "평가 설정의 출처"), ("supervisor", KR, "5", "켜기 기울기 편차"), ("supervisor", KR, "5", "켜기 |ω_xy|"),
            ("supervisor", KR, "5", "끄기 기울기"), ("supervisor", KR, "5", "끄기 평온 지속"), ("supervisor", KR, "5", "기준 자세 τ"),
            ("supervisor", KR, "5", "조기 복귀 (최소 s"), ("supervisor", KRH, "5", "조기 복귀 (최소 s"), ("supervisor", KRH, "5", "불응기"),
            ("supervisor", KRH, "5", "감지 입력 EMA"), ("supervisor", KRH, "5", "회복 cue"),
            ("합성", KW, "6", "두 하위 러너 계약 호환"), ("합성", KW, "6", "W3 → KE 인계 규칙"), ("합성", KW, "6", "KE → W3 인계"),
            ("합성", KW, "6", "인계 뒤 KE 상태"), ("합성", KW, "6", "회복 모드 중 걷기 켜기"), ("합성", KW, "6", "노드 세션"),
            ("정책 파일", G, "0", "서기 정책 가중치"), ("정책 파일", KRH, "0", "회복 정책 가중치"), ("정책 파일", W, "0", "걷기 정책 가중치"),
            ("학습 쪽 기준", G, "0", "서기 학습 시점의 시뮬 소스"), ("학습 쪽 기준", KRH, "0", "회복 학습 시점의 시뮬 소스"), ("학습 쪽 기준", W, "0", "걷기 학습 시점의 시뮬 소스"),
            ("젯슨 판", KRH, "7", "재구성한 파일 = 저장소가 기록한 배포본"), ("젯슨 판", KRH, "7", "레지스트리·상수 값"),
            ("젯슨 판", KRH, "7", "러너 프로브(600 틱) HEAD 와 비트 비교"), ("젯슨 판", KRH, "7", "노드 세션(가짜 ROS, main() 그대로) 발행"),
            ("젯슨 판", G, "7", "실물 블랙박스 재생"), ("젯슨 판", KRH, "7", "실물 블랙박스 재생"),
            ("젯슨 판", KRH, "7", "실물 기록의 배포 체인 값"), ("젯슨 판", KRH, "7", "HEAD 에만 있는 선택 기능")]
    for what, key, catp, pref in PICK:
        r = find_row(key, catp, pref)
        if r is None:
            L.append(f"| {what} | {sk(key)} | (항목 '{esc(pref)}' 없음 — 표가 바뀌었다) | — | — | {UNK} |")
        else:
            L.append(f"| {what} | {sk(key)} | **{esc(r['item'])}**: {esc(r['train'])} | {esc(r['head'])} | {esc(r['jet'])} | {r['verdict']} |")
    L += ["", "## 4. 항목 × 정책 판정표와 기존 검증 대응", "",
          "○ 일치 · ✕ 불일치 · ? 확인 불가 · – 해당 없음. 마지막 칸 = 그 항목을 덮는 기존 검증('없음' = 이 감사가 처음). [P1]–[P6] = selftest_w3_rollout_parity, PARITY #n = reports/gen_w3/PARITY_AUDIT_1007.md 표의 번호.", "",
          render_matrix(all_rows), "",
          "## 5. 기존 검증 — 재실행 결과와 빈칸", ""]
    if existing:
        L += ["| 기존 검증 | 무엇을 고정하나 | 종료 코드 | [PASS] / [FAIL] 줄 수(그 표기를 쓰는 검증만) | 마지막 판정 줄 | 시간 |", "|---|---|---|---|---|---|"]
        for name, rc, sec, last, what, n_pass, n_fail in existing:
            L.append(f"| {esc(name)} | {esc(what)} | {rc} | {f'{n_pass} / {n_fail}' if (n_pass or n_fail) else '—'} | {esc(last)} | {sec:.0f} s |")
        L += ["", "재실행 결과: " + ("**전부 종료 코드 0**." if all(e[1] == 0 for e in existing) else "**실패한 검증이 있다 — 위 표 참조**.")]
    else:
        L += ["(이 실행에서는 재실행하지 않았다 — `--run-existing`.)", "", "| 기존 검증 | 무엇을 고정하나 |", "|---|---|"]
        L += [f"| {esc(name)} | {esc(what)} |" for name, _a, what in EXISTING]
    L += ["", "문서·평가 형태의 기존 검증(여기서 재실행하지 않음): `reports/gen_w3/PARITY_AUDIT_1007.md`(W3S3d·KE 26 항목 + 폐루프 영향 평가), `reports/audits/deploy_contract_0905_summary.md`(stand_v18 세대 37 건), "
          "`rl_walking/deploy/selftest_w3s3d_onnx.py`(W3S3d onnx 대 pt — Isaac 파이썬 필요), `rl_walking/deploy/sim_validate_runner_stage4.py`(Isaac 필요 — 관절 순서 단언), "
          "TRANSITION_W3KE §5.10c X6(배포 코드 합성 러너를 시뮬 폐루프에 넣은 평가).", "",
          "**기존 검증의 빈칸** — 1–8 은 이 감사가 메웠고 9–12 는 남아 있다(시뮬 평가가 있어야 한다):", ""]
    L += [f"{i}. {t}" for i, t in enumerate(BLIND_SPOTS, 1)]
    L += ["", "## 6. 감사 방법과 감사 자체의 민감도", "",
          "| 단계 | 무엇으로 | 덮는 것 |", "|---|---|---|",
          "| 학습 값 추출 | env.yaml·agent.yaml(PyYAML), 체크포인트 가중치, tfevents 커리큘럼 스칼라, 시뮬 소스 AST | 관측 항·이력·클립·필터 인자, 행동 스케일·슬루·지연·LPF·온셋 대역, 명령 범위·커리큘럼 도달값, W3 위상 인자 |",
          "| 학습 시점 소스 확인 | run 폴더 시각 → 그때의 브랜치 커밋, run 이 남긴 git status·diff, 계약 함수(관측 항 함수·행동 항 클래스·한계/위상 이벤트와 보조 정의)의 소스를 현재와 비교 | '학습 값' 칸의 소스 근거가 학습 때와 같은지 |",
          "| 시뮬 소스 실행 | `_ema_state`·`last_action_lpf`·`encoder_joint_vel_lpf`·`gait_clock`·`onset_alpha` 를 **학습 시점 커밋의 소스 그대로** torch CPU 로 | 필터 시작 규칙·계수, 클록 식, 블렌딩 식 |",
          "| 러너 프로브 | `_make_runner(키)` 실모델에 합성 입력 600 틱(기울기 펄스 3 회·한발 구간) + 클립·슬루 기준 프로브 | 관측 320 재구성, β·α·주파수·스케일·기본자세·슬루 역산, 직전 행동 정의, 클립 유무, W3 래치·리드 발, 합성 인계 규칙 |",
          "| 노드 세션 | 가짜 rclpy 로 `rl_bridge_node.main()` 그대로(--live, 발행은 기록만, HOME 은 임시 폴더) | 유한차분 속도·LPF·블렌딩·모터 프레임·발행, W3 명령·헤딩 고리, 합성 인계 |",
          f"| 젯슨 판 | `git archive {JET_COMMIT}` + 후보 브리지로 임시 트리, 같은 워커 | HEAD 대 젯슨 판 비트 비교, 레지스트리·상수 |",
          "| 실물 재생 | 젯슨 블랙박스의 입력을 BridgeCore 에 넣고 기록된 관측·원시 행동·목표·회복 모드와 대조 | 젯슨이 실제로 낸 값 = 재구성한 코드의 값 |", ""]
    if mutants:
        rc, mres = mutants
        ms = mres.get("mutants") or []
        npass, nfail = sum(1 for m in ms if m["ok"]), sum(1 for m in ms if not m["ok"])
        ctl = mres.get("control") or {}
        L += [f"**민감도 시험**(`contract_audit_mutants.py`): 배포 코드 사본에 결함을 하나씩 심어 감사가 그 항목을 불일치로 바꾸는지 본다 — **변이 {len(ms)} 종 중 {npass} 통과 / {nfail} 실패**(종료 코드 {rc}).", "",
              "- 대조군(변이 없는 사본의 판정 = 저장소 직접 감사의 판정이어야 한다 — 다르면 사본 과정이 판정을 바꾼 것): "
              + (f"**{'같음' if ctl.get('same') else '다름'}**(행 {ctl.get('rows_direct')}·{ctl.get('rows_copy')}, `--allow-known` 종료 코드 {ctl.get('rc_direct')}·{ctl.get('rc_copy')})" if ctl else "결과 없음 — " + esc(mres.get("error", ""))[:300]),
              "- 통과 조건: 변이마다 ① '잡혀야 하는 항목'이 대조군에서는 불일치가 아니었고 변이에서는 불일치이고 ② 회귀 관문(`--allow-known`)의 종료 코드가 0(대조군) → 1(변이)로 바뀌어야 한다. "
              "'새 불일치 줄 수'는 같은 결함이 다른 줄(전체 재구성·비트 비교·실물 재생)에서도 잡힌 것까지 센 수다.",
              "- '기지 줄의 변화'는 종전부터 불일치이던 줄이 그 변이에서 다른 판정으로 바뀐 것이다 — 결함 때문에 그 줄의 재구성이 성립하지 않으면 '확인 불가'가 된다"
              "(`cue_vx_005` 만 값이 학습 범위 안으로 들어가 '일치'가 된다).", "",
              "| 결과 | 변이 | 심은 결함 | 잡혀야 하는 항목 → 실제로 잡힌 항목 | 새 불일치 줄 수 | 관문 종료 코드 | 기지 줄의 변화 |", "|---|---|---|---|---|---|---|"]
        for m in ms:
            lt = m.get("lost_to") or {}
            L.append(f"| {'통과' if m['ok'] else '**실패**'} | `{m['name']}` | {esc(m['desc'])} | {esc(' / '.join(m.get('expect') or []))} → {esc(' / '.join(m.get('hit') or []) or '없음')}"
                     + (f" ({esc(m['note'])})" if m.get("note") else "") + f" | {m.get('n_flipped', 0)} | {m.get('rc', '—')} | "
                     + (esc("; ".join(f"{k} → {v}" for k, v in lt.items())) if lt else "—") + " |")
        L += ["", "이 시험이 보이는 것은 '이 26 가지 결함은 잡는다'까지다 — 목록에 없는 종류의 결함(예: 현장 센서 축, stage8 노드 안쪽)은 7 절의 한계에 적었다.", ""]
    else:
        L += ["민감도 시험(`contract_audit_mutants.py`)은 이 실행에서 돌리지 않았다 — `--run-mutants`.", ""]
    L += ["## 7. 한계", ""] + [f"- {t}" for t in LIMITS]
    L += ["", "## 부록 A. 정책별 표", "",
          "칸: 분류 · 항목 · 학습 값 · 배포 값(HEAD) · 배포 값(젯슨 판) · 판정. 기준 정책 표와 완전히 같은 줄은 생략했다. 비고·기존 검증 칸까지 있는 전체 표는 스크립트를 인자 없이 실행하면 나온다.", "",
          render_compact(all_rows), "",
          f"(감사 실행 {elapsed:.0f} s — 기존 검증·민감도 시험 시간 포함 · 학습 산출물 {logs} · torch {H.get('torch')})", ""]
    return "\n".join(L)



def main():
    ap = argparse.ArgumentParser(description="시뮬(학습) ↔ 배포 계약 감사")
    ap.add_argument("--md", default=None, help="표를 이 파일로도 저장")
    ap.add_argument("--report", default=None, help="보고서(요약·불일치·확인 불가·판정표·기존 검증·한계·정책별 표) 전체를 이 파일로 생성")
    ap.add_argument("--run-existing", action="store_true", help="--report 에 넣을 기존 selftest 11 개 재실행(약 6 분)")
    ap.add_argument("--run-mutants", action="store_true", help="--report 에 넣을 민감도 시험 실행(contract_audit_mutants.py, 약 3 분)")
    ap.add_argument("--md-compact", default=None, help="보고서용 축약판(항목 × 정책 판정표 + 기준 정책과 다른 줄만)을 이 파일로 저장")
    ap.add_argument("--json", default=None, help="행 전체를 JSON 으로 저장")
    ap.add_argument("--allow-known", action="store_true", help="꼬리표('기지'·'신규' = 문서에 적힌 것)가 붙은 불일치만 있으면 종료 코드 0 — 회귀 관문용")
    ap.add_argument("--keep", action="store_true", help="임시 폴더(젯슨 판 트리·워커 산출)를 지우지 않음")
    ap.add_argument("--logs", default=None, help="정책·학습 산출물 루트(기본: 저장소 logs/ → 주 체크아웃 logs/)")
    ap.add_argument("--deploy-tree", default=None, help="감사할 배포 소스 트리(기본 = 이 저장소). 변이 시험용")
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--tree", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--keys", default="", help=argparse.SUPPRESS)
    ap.add_argument("--out-json", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-npz", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--replay", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--no-node", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._worker:
        return worker_main(a)
    global DEPLOY_TREE
    if a.deploy_tree:
        DEPLOY_TREE = os.path.abspath(a.deploy_tree)

    import numpy as np
    t0 = time.time()
    logs = os.path.realpath(a.logs) if a.logs else find_logs()
    if logs is None:
        print("[중단] logs/rsl_rl 을 찾지 못함 — --logs 로 지정", file=sys.stderr)
        return 2
    real_log = find_file(REAL_LOG_REL)
    real_logs = [(rel, find_file(rel)) for rel in REAL_LOGS]
    real_logs = [(rel, p_) for rel, p_ in real_logs if p_]
    tmp = tempfile.mkdtemp(prefix="contract_audit_")
    ctx = {}
    try:
        # ---- 젯슨 판 트리 + 워커 2 개(병렬) ----
        jet_tree = os.path.join(tmp, "jetson_tree")
        os.makedirs(jet_tree)
        try:
            ctx["jet_build"] = build_jetson_tree(jet_tree)
            jet_ok = True
        except Exception as e:  # noqa: BLE001
            ctx["jet_build"] = {"error": repr(e)}
            jet_ok = False
        pH, jH, nH = run_worker(DEPLOY_TREE, logs, POLICY_KEYS, tmp, "head", replay=[p_ for _, p_ in real_logs])
        pJ = None
        if jet_ok:
            pJ, jJ, nJ = run_worker(jet_tree, logs, JET_KEYS, tmp, "jet", replay=[p_ for _, p_ in real_logs])
        oH, eH = pH.communicate()
        if pH.returncode != 0 or not os.path.isfile(jH):
            print("[중단] HEAD 워커 실패\n" + eH.decode(errors="replace")[-3000:], file=sys.stderr)
            return 2
        with open(jH, encoding="utf-8") as f:
            H = json.load(f)
        ZH = np.load(nH, allow_pickle=False)
        J, ZJ = None, None
        if pJ is not None:
            oJ, eJ = pJ.communicate()
            if pJ.returncode == 0 and os.path.isfile(jJ):
                with open(jJ, encoding="utf-8") as f:
                    J = json.load(f)
                ZJ = np.load(nJ, allow_pickle=False)
            else:
                ctx["jet_build"]["worker_error"] = eJ.decode(errors="replace")[-2000:]
        ctx.update(H=H, J=J, ZH=ZH, ZJ=ZJ)
        regH = H["registry"]

        # ---- 학습 계약: run 경로는 브리지 레지스트리에서 역추적(exported/ 의 부모) ----
        run_rels = set()
        for k, K in regH["keys"].items():
            if not K.get("registered"):
                continue
            run_rels.add(os.path.dirname(K["ckpt_rel"]))
            if K.get("c5_pair"):
                run_rels.add(os.path.dirname(K["c5_pair"][1]))
        trains = {rr: Train(os.path.join(logs, rr)) for rr in sorted(run_rels)}
        ctx["trains"] = trains
        ctx["facts"] = sim_source_facts()

        # ---- 네트워크: 체크포인트 = policy.pt = policy.onnx ----
        import torch
        torch.set_num_threads(1)
        man = (ctx["jet_build"].get("manifest") or {})
        net = {}
        for rr, tr in trains.items():
            nv = {"ckpt": tr.src_ckpt}
            try:
                ck = torch.load(os.path.join(tr.run, tr.src_ckpt), map_location="cpu", weights_only=True)["model_state_dict"]
                actor = {k: v for k, v in ck.items() if k.startswith("actor.")}
                pt = torch.jit.load(os.path.join(tr.run, "exported", "policy.pt"), map_location="cpu").eval()
                pp = dict(pt.named_parameters())
                nv["n_tensors"] = len(actor)
                nv["pt_match"] = f"{sum(1 for k, v in actor.items() if k in pp and torch.equal(v, pp[k]))}/{len(actor)}"
                nv["pt_equal"] = all(k in pp and torch.equal(v, pp[k]) for k, v in actor.items())
                nv["pt_extra_params"] = len([k for k in pp if k not in actor]) + len(list(pt.named_buffers()))
                nv["in_dim"] = int(actor["actor.0.weight"].shape[1])
                nv["out_dim"] = int(actor[f"actor.{max(int(k.split('.')[1]) for k in actor)}.weight"].shape[0])
                og = onnx_graph(os.path.join(tr.run, "exported", "policy.onnx"))
                nv["onnx_match"] = f"{sum(1 for k, v in actor.items() if k in og['inits'] and np.array_equal(og['inits'][k], v.numpy()))}/{len(actor)}"
                nv["onnx_equal"] = all(k in og["inits"] and np.array_equal(og["inits"][k], v.numpy()) for k, v in actor.items()) and len(og["inits"]) == len(actor)
                nv["onnx_ops"] = "–".join(op for op, _ in og["ops"])
                x = torch.as_tensor(np.random.default_rng(0).standard_normal((32, nv["in_dim"])).astype(np.float32) * 0.5)
                with torch.no_grad():
                    h = x
                    ks = sorted({int(k.split(".")[1]) for k in actor})
                    for i, li in enumerate(ks):
                        h = h @ actor[f"actor.{li}.weight"].T + actor[f"actor.{li}.bias"]
                        if i < len(ks) - 1:
                            h = torch.nn.functional.elu(h)
                    nv["fwd_err"] = float((pt(x) - h).abs().max())
                    nv["fwd_mag"] = float(h.abs().max())
                act_name = (tr.agent.get("policy") or {}).get("activation")
                nv["onnx_act_ok"] = (act_name == "elu") and all(op in ("Gemm", "Elu") for op, _ in og["ops"]) and all(abs((at.get("alpha", 1.0) or 1.0) - 1.0) < 1e-9 for op, at in og["ops"] if op == "Elu")
                if not nv["onnx_act_ok"]:
                    nv["onnx_equal"] = False
            except Exception as e:  # noqa: BLE001
                nv["error"] = repr(e)
                nv.setdefault("pt_equal", None)
                nv.setdefault("onnx_equal", None)
            mm = {}
            for fn in ("exported/policy.pt", "exported/policy.onnx", "params/env.yaml"):
                rel = os.path.join("logs", rr, fn)
                if rel in man:
                    want = next(iter(man[rel].values()))
                    p = os.path.join(tr.run, fn)
                    mm[fn.split("/")[-1]] = os.path.isfile(p) and md5_of(p) == want
            nv["manifest_md5"] = mm
            net[tr.run] = nv
        ctx["net"] = net

        # ---- 학습 로그 커리큘럼 도달값(W3) ----
        cur = {}
        for rr, tr in trains.items():
            if (tr.env.get("curriculum") or {}):
                try:
                    sc = tfevents_scalars(tr.run, lambda t: t.startswith("Curriculum/"))
                    it_dep = int(re.sub(r"\D", "", tr.src_ckpt or "0") or 0)
                    pick = lambda tag: (max((v for s, v in sc.get(tag, []) if s <= it_dep), default=None))  # noqa: E731
                    cur[tr.run] = {"vy_max": pick("Curriculum/w3_vy/vy_max"), "wz_max": pick("Curriculum/w3_wz/wz_max"),
                                   "alpha": pick("Curriculum/w3_assist/alpha"), "upto": it_dep}
                    if all(v is None for k, v in cur[tr.run].items() if k != "upto"):
                        cur.pop(tr.run)
                except Exception:  # noqa: BLE001
                    pass
        ctx["curric"] = cur

        # ---- 시뮬 기록(관절 순서·이력 순서) ----
        sr = {"ok": False}
        p = find_file(SIM_REC_REL)
        if p:
            try:
                z = np.load(p, allow_pickle=False)
                meta = json.loads(str(z["meta"]))
                sr.update(ok=True, joint_names=list(meta.get("joint_names") or []), obs_terms=meta.get("obs_terms"), obs_dims=meta.get("obs_dims"),
                          task=meta.get("task"))
                ob, gy, elb = z["obs"], z["gyro"], z["elb"]
                ok_old = True
                n_chk = 0
                for t_ in range(6, min(ob.shape[0], 400)):
                    for e in range(ob.shape[1]):
                        if elb[t_, e] >= 5:
                            blk = ob[t_, e, 0:15].reshape(5, 3)
                            ok_old &= bool(np.array_equal(blk[4], gy[t_, e]) and np.array_equal(blk[0], gy[t_ - 4, e]) and np.array_equal(blk[3], gy[t_ - 1, e]))
                            n_chk += 1
                sr["oldest_first"] = bool(ok_old and n_chk > 0)
                e0 = [(t_, e) for t_ in range(ob.shape[0]) for e in range(ob.shape[1]) if elb[t_, e] == 0][:6]
                sr["backfill"] = bool(e0) and all(np.array_equal(ob[t_, e, 0:15].reshape(5, 3), np.tile(ob[t_, e, 0:3], (5, 1))) for t_, e in e0)
                w3run = next((tr for tr in trains.values() if tr.w3()[0] is not None), None)
                if w3run is not None:
                    sr["terms_match"] = [n for n, _, _ in w3run.layout()] == sr["obs_terms"] and [d * h for _, d, h in w3run.layout()] == sr["obs_dims"]
                    sr["usd"] = w3run.usd()
            except Exception as e:  # noqa: BLE001
                sr = {"ok": False, "error": repr(e)}
        ctx["simrec"] = sr

        # ---- 평가 인자 ----
        base = parse_eval_args(EVAL_SH_KE, "BASE") or {}
        ctx["eval_ke"], ctx["eval_src"] = {}, {}
        for arm, nm in (("imu", "kr"), ("imuH", "krh")):
            ra, _ = parse_result_header(EVAL_RES_KE[nm])                 # 실제로 돈 인자(산출물) 우선
            if ra:
                ctx["eval_ke"][nm], ctx["eval_src"][nm] = ra, EVAL_RES_KE[nm] + " 의 '인자:' 줄"
                continue
            x = parse_eval_args(EVAL_SH_KE, f"ARM:{arm}")
            if base and x:
                ctx["eval_ke"][nm], ctx["eval_src"][nm] = dict(base, **x), EVAL_SH_KE + "(스크립트 — 산출물 없음)"
        ra, rs = parse_result_header(EVAL_RES_TR)
        rd, rds = parse_result_header(EVAL_RES_TR_DIRECT)
        tr_base = parse_eval_args(EVAL_SH_TR, "BASE") or {}
        tr_ke = parse_eval_args(EVAL_SH_TR, "KE")
        if ra:
            ctx["eval_tr"], ctx["eval_src"]["tr"] = ra, EVAL_RES_TR + " 의 '인자:' 줄"
        else:
            ctx["eval_tr"] = dict(tr_base, **tr_ke) if tr_ke else None
            ctx["eval_src"]["tr"] = EVAL_SH_TR + "(스크립트 — 산출물 없음)"
        ctx["eval_tr_settle"] = rds or rs                                # 평가기 정착 판정 줄(|ω| 문턱, 유지, 상한)
        ctx["eval_tr_direct"] = rd
        ctx["eval_kw_defaults"] = argparse_defaults("rl_walking/scripts/eval_transition_kw.py", {"--kw_handoff", "--kw_settle"})
        ctx["eval_tr_defaults"] = argparse_defaults("rl_walking/scripts/eval_transition.py", {"--sq_hold", "--sq_w", "--settle_max", "--settle"})
        ctx["eval_pre"] = argparse_defaults("rl_walking/scripts/eval_transition.py", {"--pre_both_s", "--pre_tilt", "--pre_w", "--pre_w_s"})
        gd = argparse_defaults("rl_walking/scripts/eval_ground_recovery.py", {"--gyro_delay"})
        used = [x for x in list(ctx["eval_ke"].values()) + [ctx["eval_tr"]] if x]
        ctx["eval_gyro_delay"] = (gd.get("--gyro_delay") if not any("--gyro_delay" in x for x in used)
                                  else sorted({str(x.get("--gyro_delay")) for x in used if "--gyro_delay" in x}))
        ctx["eval_task"] = sorted({str(x.get("--task")) for x in used if x.get("--task")})
        try:
            with open(os.path.join(REPO, "rl_walking/scripts/eval_ground_recovery.py"), encoding="utf-8") as f:
                es = f.read()
            ctx["eval_ref_consts"] = (".clamp(-8.0, 8.0)" in es) and ("(_w < 0.1) & _cf[:, 0] & _cf[:, 1]" in es)
        except OSError:
            ctx["eval_ref_consts"] = False
        ev_ck = {}
        for args_ in used:
            if args_.get("--load_run") and args_.get("--checkpoint"):
                ev_ck[args_["--load_run"]] = args_["--checkpoint"]
            for fl in ("--switch_to", "--ke"):
                if isinstance(args_.get(fl), str) and ":" in args_[fl]:
                    r_, c_ = args_[fl].split(":")
                    ev_ck[r_] = c_
        ctx["eval_ckpt"] = ev_ck

        # ---- 시뮬 소스 실행 대조(관측 필터·클록·온셋) ----
        # 기준 소스 = 각 run 의 **학습 시작 시점 커밋**(없으면 작업 트리). 같은 소스·같은 관측 설정이면 한 번만 실행한다.
        ctx["simexec"], ctx["srcchk"] = {}, {}
        pa0 = {}
        for rr, tr in trains.items():
            k0 = next((k for k, K in regH["keys"].items() if K.get("registered") and os.path.dirname(K["ckpt_rel"]) == rr), None)
            pa0 = pa0 or (H["probe"].get(k0) or {}).get("attrs") or (H["probe"].get(k0) or {}).get("attrs_ke") or {}
        memo = {}
        for rr, tr in trains.items():
            try:
                ctx["srcchk"][tr.run] = contract_source_check(tr, ctx["facts"])
            except Exception as e:  # noqa: BLE001
                ctx["srcchk"][tr.run] = {"error": repr(e), "commit": tr.commit}
            try:
                srcsig = hashlib.md5((_extract_funcs("rl_walking/mdp_hang.py", ["_ema_state", "last_action_lpf", "encoder_joint_vel_lpf", "gyro_ema_lagged", "encoder_joint_vel_ema_v2"], tr.commit)
                                      + _extract_funcs("rl_walking/mdp_stage4.py", ["_gait_phase", "gait_clock"], tr.commit)).encode()).hexdigest()
                mk = (srcsig, tr.obs_sig(), tr.dt())
                if mk not in memo:
                    memo[mk] = {int(k): v for k, v in sim_exec_filters(tr, (pa0.get("act_beta"), pa0.get("jv_beta")), commit=tr.commit).items()}
                ctx["simexec"][tr.run] = memo[mk]
            except Exception as e:  # noqa: BLE001
                ctx["simexec"][tr.run] = {"error": repr(e)}
        ctx["simexec_n"] = len(memo)
        try:
            ctx["onset_err"] = max(sim_exec_onset(regH["BLEND_S"], commit=c_) for c_ in sorted({tr.commit for tr in trains.values()}, key=str))
        except Exception:  # noqa: BLE001
            ctx["onset_err"] = None

        # ---- 중력식 ----
        ge = {}
        for nm, Z in (("H", ZH), ("J", ZJ)):
            if Z is None:
                continue
            q, pg = Z["grav/quat"], Z["grav/pg"]
            ge[f"{nm}_pg"] = float(max(np.abs(independent_gravity(q[i]) - pg[i]).max() for i in range(len(q))))
            gi, go = Z["grav/cg_in"], Z["grav/cg_out"]
            r_in, p_in = np.degrees(np.arctan2(gi[:, 1], -gi[:, 2])), np.degrees(np.arctan2(gi[:, 0], -gi[:, 2]))
            r_o, p_o = np.degrees(np.arctan2(go[:, 1], -go[:, 2])), np.degrees(np.arctan2(go[:, 0], -go[:, 2]))
            ge[f"{nm}_cg"] = float(max(np.abs(r_o - (r_in - 1.3)).max(), np.abs(p_o - (p_in + 0.6)).max()))
        ctx["grav"] = ge
        ck = ZH["clock/long"].astype(np.float64)
        exact = np.array([[math.sin(2 * math.pi * 1.1 * t * 0.02), math.cos(2 * math.pi * 1.1 * t * 0.02)] for t in (0, 1, 1000, 30000, 180000)])
        ctx["clock_long"] = float(np.abs(ck - exact).max())

        # ---- 실물 로그 자체 통계 ----
        ctx["reallog"] = real_log_stats(real_log) if real_log else None
        ctx["reallogs"] = {}
        for rel, p_ in real_logs:
            try:
                ctx["reallogs"][rel] = ctx["reallog"] if p_ == real_log else real_log_stats(p_, ground_until=(REAL_LOG_GROUND.get(rel) or (None,))[0])
            except Exception:  # noqa: BLE001
                pass

        # ---- 프로브·노드 계측 ----
        meas, nodemeas, bit = {}, {}, {}
        for key in POLICY_KEYS:
            K = regH["keys"].get(key, {})
            P = H["probe"].get(key, {})
            if not K.get("registered") or "seq" not in P:
                continue
            tr = trains[os.path.dirname(K["ckpt_rel"])]
            w3 = bool(K.get("w3cfg"))
            g3 = (tr.w3()[0] or {}) if w3 else {}
            Pm = {"default": tr.default_pose(regH["JOINT_ORDER"]), "beta_act": regH["STAND_LPF_BETA"] if short(tr.term("actions").get("func")) == "last_action_lpf" else None,
                  "beta_jv": (tr.term("joint_vel").get("params") or {}).get("beta"), "alpha": (tr.term("gyro_ema").get("params") or {}).get("alpha", 0.1),
                  "freq": (tr.term("gait_clock").get("params") or {}).get("freq", 1.4), "dt": tr.dt(), "scale": float(tr.act.get("scale")),
                  "slew": float(tr.act.get("max_delta_per_step")), "f_step": float(g3.get("f_step", 1.1)), "w3": w3}
            try:
                meas[key] = analyze_probe(ZH, key, tr.layout(), Pm, w3=w3)
            except Exception as e:  # noqa: BLE001
                meas[key] = {"error": repr(e)}
            ctx.setdefault("Pm", {})[key] = Pm
        for key in POLICY_KEYS:
            K = regH["keys"].get(key, {})
            nd = H["node"].get(key) or {}
            if not K.get("registered") or "error" in nd or f"node/{key}/raw" not in ZH.files:
                if "error" in nd:
                    nodemeas[key] = {"error": nd["error"]}
                continue
            tr = trains[os.path.dirname(K["ckpt_rel"])]
            Pm = ctx.get("Pm", {}).get(key) or ctx["Pm"].get(K.get("kew3", {}).get("ke", ""), None)
            if Pm is None:
                continue
            try:
                nodemeas[key] = analyze_node(ZH, key, nd, tr.layout(), dict(Pm, w3=bool(K.get("w3cfg"))), regH["STAND_LPF_BETA"], regH["BLEND_S"])
            except Exception as e:  # noqa: BLE001
                nodemeas[key] = {"error": repr(e)}
        ctx["meas"], ctx["nodemeas"] = meas, nodemeas
        if ZJ is not None:
            for key in JET_KEYS:
                b = {}
                for nm in ("obs", "raw", "out", "mode"):
                    a_, b_ = f"probe/{key}/{nm}", f"probe/{key}/{nm}"
                    b[nm] = bool(a_ in ZH.files and b_ in ZJ.files and np.array_equal(ZH[a_], ZJ[b_]))
                a_ = f"node/{key}/rel"
                if a_ in ZH.files and a_ in ZJ.files:
                    b["node"] = bool(np.array_equal(ZH[a_], ZJ[a_], equal_nan=True) and np.array_equal(ZH[f"node/{key}/tgt_motor"], ZJ[f"node/{key}/tgt_motor"], equal_nan=True))
                    b["node_obs"] = bool(np.array_equal(ZH[f"node/{key}/obs"], ZJ[f"node/{key}/obs"], equal_nan=True))
                else:
                    b["node"] = b["node_obs"] = None
                bit[key] = b
        ctx["bit"] = bit

        # ---- 레지스트리 HEAD vs 젯슨 판 ----
        reg_diff, reg_n = {}, 0
        if J is not None:
            regJ = J["registry"]
            top = ["JOINT_ORDER", "RATE_HZ", "TICK_SEC", "BLEND_S", "STAND_LPF_BETA", "OBS_ACT_EMA_BETA_DEPLOY", "POLICY_KP", "POLICY_KD", "SLEW_RAD",
                   "ACTION_SCALE", "STAND_DEFAULT", "CONTACT_FORCE_THRESHOLD_N", "SENSOR_STALE_SEC", "FOOT_STALE_SEC", "LIFT_STOP_S",
                   "CKPT_GAIT_FREQ_DEFAULT", "C5_DETECTOR_KW", "HISTORY_LEN", "EMA_ALPHA_S8A", "GAIT_FREQ_HZ", "POLICY_DT"]
            per = ["ckpt_rel", "default_pose", "gait_freq", "jvel_ema_mode_reg", "require_env_yaml", "lpf_applies", "c5_pair", "c5_extra"]
            reg_n = len(top) + len(per)
            for key in JET_KEYS:
                d = [t_ for t_ in top if json.dumps(regH.get(t_), sort_keys=True) != json.dumps(regJ.get(t_), sort_keys=True)]
                kh, kj_ = regH["keys"].get(key, {}), regJ["keys"].get(key, {})
                d += [p_ for p_ in per if json.dumps(kh.get(p_), sort_keys=True) != json.dumps(kj_.get(p_), sort_keys=True)]
                ph, pj = H["probe"].get(key, {}), J["probe"].get(key, {})
                for blk in ("det", "cue"):
                    if blk in ph or blk in pj:
                        ks = [k for k in (ph.get(blk) or {}) if not k.startswith("has_")]
                        d += [f"{blk}.{k}" for k in ks if json.dumps((ph.get(blk) or {}).get(k)) != json.dumps((pj.get(blk) or {}).get(k))]
                mh, mj = regH.get("map", {}), regJ.get("map", {})
                d += [f"map.{k}" for k in ("gain_mismatches", "roll_comp_enabled", "pitch_comp_enabled", "imu_mount_bias_deg", "dir") if mh.get(k) != mj.get(k)]
                reg_diff[key] = d
        ctx["reg_diff"], ctx["reg_n"] = reg_diff, reg_n

        # ---- 젯슨 맵 사본(있으면) ----
        ctx["jetmap_same"] = ctx["jetmap_comp_off"] = None
        jm = find_file("deploy/hwmap_jetson_1005.json")
        if jm:
            try:
                with open(jm, encoding="utf-8") as f:
                    mj = json.load(f)
                ctx["jetmap_same"] = all(float(v.get("kp", -1)) == 150.0 and float(v.get("kd", -1)) == 5.0 for v in mj["joints"].values())
                ctx["jetmap_comp_off"] = not (mj.get("roll_comp") or {}).get("enabled", False) and not (mj.get("pitch_comp") or {}).get("enabled", False)
            except Exception:  # noqa: BLE001
                pass

        # ---- stage8 노드 상수(소스) — HEAD 와 젯슨 커밋 판 ----
        def s8_consts(src):
            m1 = re.search(r"^MAX_TARGET_STEP_DEG_PER_TICK = ([0-9.]+)", src, re.M)
            m2 = re.search(r"^CONTROL_PERIOD_SEC = ([0-9.]+)", src, re.M)
            return {"step": float(m1.group(1)) if m1 else None, "period": float(m2.group(1)) if m2 else None,
                    "reject_25": "if excess > 25.0:" in src and "limit_runaway" in src,
                    "clamp_all": "limit_verdict = check_absolute_deg(joint, joint_abs_deg)" in src and "clamped = min(max(joint_abs_deg, lim[\"soft_min\"]), lim[\"soft_max\"])" in src}
        try:
            with open(os.path.join(REPO, "jetson/scripts/stage8_12axis_mit_control_node.py"), encoding="utf-8") as f:
                s8h = s8_consts(f.read())
            try:
                s8j = s8_consts(git("show", f"{JET_COMMIT}:jetson/scripts/stage8_12axis_mit_control_node.py"))
                s8h["jet_same"] = (s8j == {k: v for k, v in s8h.items() if k != "jet_same"})
            except Exception:  # noqa: BLE001
                s8h["jet_same"] = None
            ctx["stage8"] = s8h
        except OSError:
            ctx["stage8"] = {}

        # ---- 한계 파일(config/joint_limits_12dof.json)의 마지막 변경이 모든 run 학습 시작 전인가 ----
        try:
            d_lim = git("log", "-1", "--format=%ad", "--date=short", "--", "config/joint_limits_12dof.json").strip()
            d_runs = sorted(m.group(0) for m in (re.match(r"\d{4}-\d{2}-\d{2}", os.path.basename(tr.run)) for tr in trains.values()) if m)
            ctx["limits_date"] = (d_lim, d_runs[0] if d_runs else None, bool(d_lim and d_runs and d_lim < d_runs[0]))
            # 0905 감사(B7: soft 표 재검토 권고) 뒤의 한계 파일 변경 이력
            ctx["limits_since_0905"] = [ln.strip()[:64] for ln in git("log", "--since=2026-09-05", "--format=%ad %s", "--date=short", "--", "config/joint_limits_12dof.json").splitlines() if ln.strip()]
        except Exception:  # noqa: BLE001
            ctx["limits_date"] = None

        # ---- KE 회복·전환 평가기에 stage8 식 전 관절 클램프가 있는가(소스) ----
        try:
            has = False
            for rel_ in ("rl_walking/scripts/eval_ground_recovery.py", "rl_walking/scripts/eval_transition.py", "rl_walking/scripts/eval_transition_kw.py"):
                with open(os.path.join(REPO, rel_), encoding="utf-8") as f:
                    has = has or ("s8_clamp" in f.read())
            ctx["ke_eval_has_s8_clamp"] = has
        except OSError:
            ctx["ke_eval_has_s8_clamp"] = None

        # ---- 걷기 버튼 표가 범위 안인가 ----
        mv_ok = True
        for key, K in regH["keys"].items():
            lim = K.get("w3_limits")
            if not lim or not K.get("w3_moves"):
                continue
            for nm, m in K["w3_moves"].items():
                nz = [k for k in ("vx", "vy", "wz") if m[k] != 0.0]
                if len(nz) > 1:
                    mv_ok = False
                r_ = {"vx": lim["vx_fwd"] if m["vx"] > 0 else lim["vx_back"], "vy": lim["vy"], "wz": lim["wz"]}
                for k in nz:
                    if not (r_[k][0] - 1e-9 <= abs(m[k]) <= r_[k][1] + 1e-9):
                        mv_ok = False
        ctx["moves_ok"] = mv_ok

        # ---- 후보 브리지 대 HEAD 브리지 줄 차이 ----
        try:
            r = subprocess.run(["diff", os.path.join(REPO, JET_BRIDGE_REL), os.path.join(REPO, "jetson/scripts/rl_bridge_node.py")], capture_output=True)
            dl = r.stdout.decode(errors="replace").splitlines()
            ctx["bridge_removed"], ctx["bridge_added"] = sum(1 for ln in dl if ln.startswith("<")), sum(1 for ln in dl if ln.startswith(">"))
        except Exception:  # noqa: BLE001
            ctx["bridge_removed"] = ctx["bridge_added"] = "?"

        # ---- 표 ----
        head = git("rev-parse", "--short", "HEAD").strip()
        lines = [f"# 계약 감사 표 (자동 생성 — reports/night_1007/contract_audit.py, HEAD {head}, {time.strftime('%Y-%m-%d %H:%M')})", "",
                 f"- 학습 산출물 루트: {logs}" + ("" if DEPLOY_TREE == REPO else f" · **배포 트리 = {DEPLOY_TREE}**(--deploy-tree, 변이 시험)"), f"- 젯슨 판: 커밋 {JET_COMMIT} 의 러너·c5_switch + {JET_BRIDGE_REL}" + ("" if J is not None else " — **구성 실패**: " + str(ctx['jet_build'].get('error') or ctx['jet_build'].get('worker_error', ''))[:300]),
                 f"- 실물 블랙박스: {REAL_LOG_REL if real_log else '없음'} · 시뮬 기록: {SIM_REC_REL if sr.get('ok') else '없음'}",
                 f"- 판정: {OK} / {BAD} / {UNK}. 불일치 꼬리표: '기지(…)' = 종전 문서에 있던 것, '신규(…)' = 이 감사가 찾아 CONTRACT_AUDIT.md 에 적은 것, 꼬리표 없음 = 문서에 없는 새 불일치(회귀 신호). "
                 f"허용오차 관측 {TOL_OBS:g}·목표각 {TOL_ACT:g} rad.", ""]
        all_rows, summary = {}, []
        for key in POLICY_KEYS:
            try:
                R = build_rows(key, ctx)
            except Exception as e:  # noqa: BLE001
                import traceback
                R = Rows()
                R.add("오류", "표 작성 실패", "—", f"{type(e).__name__}: {e}", "—", UNK, traceback.format_exc()[-600:])
            all_rows[key] = R
            lines.append(render(key, R))
            c = R.count()
            summary.append((key, c))
        lines += ["## 정책별 집계", "", "| 브리지 키 | 일치 | 불일치 | 그중 기지(종전 문서) | 그중 신규(이 감사) | 꼬리표 없음 | 확인 불가 |", "|---|---|---|---|---|---|---|"]
        n_bad = n_new = 0
        for key, c in summary:
            bad = [r for r in all_rows[key].rows if r["verdict"] == BAD]
            kn = sum(1 for r in bad if r["known"].startswith("기지"))
            nw = sum(1 for r in bad if r["known"].startswith("신규"))
            n_bad += c[BAD]
            n_new += c[BAD] - kn - nw                  # 꼬리표 없는 불일치 = 문서에 없는 것(회귀 신호)
            lines.append(f"| {key} | {c[OK]} | {c[BAD]} | {kn} | {nw} | {c[BAD] - kn - nw} | {c[UNK]} |")
        lines += ["", "## 불일치 목록(중복 제거)", ""]
        seen = {}
        for key in POLICY_KEYS:
            for r in all_rows[key].rows:
                if r["verdict"] == BAD:
                    seen.setdefault((r["cat"], r["item"]), {"r": r, "keys": []})["keys"].append(key)
        for (cat, item), v in seen.items():
            r = v["r"]
            lines.append(f"- **[{cat}] {item}** ({', '.join(k.replace('stand_', '') for k in v['keys'])}) — 학습: {r['train']} / 배포: {r['head']}"
                         + (f" / 젯슨 판: {r['jet']}" if r["jet"] not in ("—", "") and not r["jet"].startswith("—") else "") + f" — {r['note']}" + (f" 〔{r['known']}〕" if r["known"] else "")
                         + (f" · 근거: {r['ev']}" if r.get("ev") else ""))
        lines += ["", "## 확인 불가 목록(중복 제거)", ""]
        seen = {}
        for key in POLICY_KEYS:
            for r in all_rows[key].rows:
                if r["verdict"] == UNK:
                    seen.setdefault((r["cat"], r["item"]), {"r": r, "keys": []})["keys"].append(key)
        for (cat, item), v in seen.items():
            r = v["r"]
            lines.append(f"- **[{cat}] {item}** ({', '.join(k.replace('stand_', '') for k in v['keys'])}) — 학습: {r['train']} / 배포: {r['head']} — {r['note']}")
        lines += ["", f"소요 {time.time() - t0:.0f} s · torch {H.get('torch')} · 워커 2 개(HEAD·젯슨 판)", ""]
        text = "\n".join(lines)
        print(text)
        if a.md:
            with open(a.md, "w", encoding="utf-8") as f:
                f.write(text)
        if a.report:
            ex_res = run_existing() if a.run_existing else None
            mu_res = run_mutants() if a.run_mutants else None
            with open(a.report, "w", encoding="utf-8") as f:
                f.write(render_report(ctx, all_rows, summary, logs, existing=ex_res, mutants=mu_res, elapsed=time.time() - t0))
        if a.md_compact:
            with open(a.md_compact, "w", encoding="utf-8") as f:
                f.write("## 항목 × 정책 판정표 (○ 일치 · ✕ 불일치 · ? 확인 불가 · – 해당 없음)\n\n" + render_matrix(all_rows) + "\n\n## 정책별 표\n\n" + render_compact(all_rows))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as f:
                json.dump({k: R.rows for k, R in all_rows.items()}, f, ensure_ascii=False, indent=1)
        if n_bad == 0:
            return 0
        if a.allow_known and n_new == 0:
            return 0
        return 1
    finally:
        if a.keep:
            print(f"[임시 폴더 보존] {tmp}", file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
