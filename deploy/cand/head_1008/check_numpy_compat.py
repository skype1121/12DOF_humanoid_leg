#!/usr/bin/env python3
"""젯슨 numpy(1.21.5) 호환 정적 점검 — 배포할 .py 가 numpy 1.21.5 에 없는 이름·키워드 인자를 쓰는지 본다(실행 없음, ast 만).

  python3 deploy/cand/head_1008/check_numpy_compat.py <파일.py> [...]        # 종료 코드 0 = 위반 없음, 1 = 위반 있음
  python3 deploy/cand/head_1008/check_numpy_compat.py --plan deploy/head_1008.plan   # 계획서의 .py 전부

기준: numpy_1.21.5_api.json — numpy 1.21.5 의 공개 이름(numpy·linalg·random·fft·testing·lib·ma 등)과 주요 함수의 인자 이름.
잡는 것: np.<1.21.5 에 없는 이름>(예: np.concat·np.atan2·np.trapezoid — numpy 2 에서 생긴 별칭), `from numpy import <없는 이름>`,
        np.quantile(..., method=) 처럼 1.21.5 서명에 없는 키워드 인자, 파이썬 3.10 문법 오류(compile).
못 잡는 것(정적 한계 — 젯슨 모사 환경의 selftest 로 따로 확인): 배열 메서드·속성(.mT 등), 값 기반 형 승격 차이
        (numpy 1.x 는 np.float32 스칼라와 파이썬 float 의 연산이 float64, numpy 2 는 float32), 기본값·반올림 동작 차이."""
import argparse
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SUBMODS = ("linalg", "random", "fft", "testing", "lib", "ma", "char", "polynomial", "core", "typing")


def load_api(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def attr_chain(node):
    """Attribute 사슬 → [루트 이름, 속성1, 속성2, ...] (루트가 Name 이 아니면 None)."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + parts[::-1]
    return None


def check_file(path, api):
    mods, sigs = api["modules"], api["signatures"]
    out = []
    try:
        src = open(path, encoding="utf-8").read()
        tree = ast.parse(src, path)
        compile(src, path, "exec")
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: 문법 오류 — {e.msg}"]
    alias = {}          # 로컬 이름 → numpy 모듈 경로("numpy", "numpy.linalg" …)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "numpy" or a.name.startswith("numpy."):
                    alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else "numpy"
        elif isinstance(node, ast.ImportFrom) and node.module and (node.module == "numpy" or node.module.startswith("numpy.")):
            known = mods.get(node.module)
            for a in node.names:
                if known is not None and a.name != "*" and a.name not in known:
                    out.append(f"{path}:{node.lineno}: from {node.module} import {a.name} — numpy {api['numpy_version']} 에 없음")
                sub = f"{node.module}.{a.name}"
                if sub in mods:
                    alias[a.asname or a.name] = sub
    if not alias:
        return out
    seen = set()
    for node in ast.walk(tree):
        chain = attr_chain(node) if isinstance(node, ast.Attribute) else None
        if chain and chain[0] in alias:
            mod, rest = alias[chain[0]], chain[1:]
            while rest and f"{mod}.{rest[0]}" in mods and rest[0] in SUBMODS + ("stride_tricks",):
                mod, rest = f"{mod}.{rest[0]}", rest[1:]
            if rest and mod in mods and not rest[0].startswith("__") and rest[0] not in mods[mod]:
                key = (node.lineno, mod, rest[0])
                if key not in seen:
                    seen.add(key)
                    out.append(f"{path}:{node.lineno}: {mod}.{rest[0]} — numpy {api['numpy_version']} 에 없음")
        if isinstance(node, ast.Call):
            ch = attr_chain(node.func) if isinstance(node.func, ast.Attribute) else None
            if ch and ch[0] in alias and alias[ch[0]] == "numpy" and len(ch) == 2:
                sig = sigs.get("numpy." + ch[1])
                if sig is not None and "kwargs" not in sig and "kwds" not in sig:
                    for kw in node.keywords:
                        if kw.arg is not None and kw.arg not in sig:
                            out.append(f"{path}:{node.lineno}: numpy.{ch[1]}({kw.arg}=…) — numpy {api['numpy_version']} 서명에 없는 인자")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--plan", help="deploy/head_1008.plan — 탭 구분(경로·동작·묶음·원본 위치…). 원본 위치가 repo 인 .py 를 본다")
    ap.add_argument("--api", default=os.path.join(HERE, "numpy_1.21.5_api.json"))
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.dirname(HERE))), help="--plan 의 상대 경로 기준(저장소 루트)")
    a = ap.parse_args()
    files = list(a.files)
    if a.plan:
        with open(a.plan, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                cols = line.rstrip("\n").split("\t")
                if cols[0].endswith(".py") and cols[1] in ("send", "verify") and cols[3] == "repo":
                    files.append(os.path.join(a.root, cols[0]))
    if not files:
        ap.print_usage()
        return 2
    api = load_api(a.api)
    bad = []
    for p in files:
        bad += check_file(p, api)
    for b in bad:
        print(b)
    print(f"numpy {api['numpy_version']} 정적 점검: 파일 {len(files)} 개 · 위반 {len(bad)} 건")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
