#!/usr/bin/env python3
"""정책·체크포인트 전수 목록 — **읽기 전용**(logs/·log_picture/ 는 읽기만, 쓰는 파일은 아래 두 개뿐).

  python3 reports/night_1007/policy_inventory.py            # 익스포트·params·핵심 체크포인트 md5 만(몇 초)
  python3 reports/night_1007/policy_inventory.py --md5 all  # 모든 model_N.pt md5(처음 한 번 약 20 GB 읽기 ≈ 1 분. 이전 JSON 의 값을 크기·mtime 이 같으면 재사용)
  python3 reports/night_1007/policy_inventory.py --check    # 분류표 근거 문구가 문서에 실제로 있는지만 검사(출력 파일 안 씀)

산출물: reports/night_1007/POLICY_INVENTORY.md · reports/night_1007/policy_inventory.json  (--out-dir 로 다른 곳에 쓸 수 있다)

하는 일
  1. logs/rsl_rl/** 의 run 디렉터리를 심볼릭 링크를 따라가지 않고 한 번씩만 훑는다(링크는 별칭으로 기록).
  2. 참조 수집: 브리지 레지스트리(ast 로 소스만 읽음 — import 안 함)·콘솔 선택지·스크립트·selftest·tests·매니페스트·문서·
     log_picture 평가 산출물 머리말·웜스타트 계보(params/agent.yaml).
  3. 분류 초안: 자동 규칙(배포 = KEW3_SPECS 가 가리키는 run · 기준선 = 레지스트리/현행 코드가 읽는 run · 계보) + 아래 CURATED 표(문서 판정).
     CURATED 의 근거 문구는 실행할 때마다 문서에서 다시 찾아 줄 번호를 붙인다. 못 찾으면 그 run 은 불명으로 내려간다.
     '평가 중'으로 적어 둔 run 은 문서에 판정 문구가 새로 생겼는지도 찾아 경고한다(분류표가 낡았다는 신호).
  4. 확인: exported*/policy.pt 의 가중치가 어느 체크포인트와 같은지(SOURCE_CHECKPOINT 글자가 아니라 가중치로), policy.onnx 안에 같은 가중치가 있는지,
     deploy/*.md5 매니페스트와 md5 가 같은지, 배포 후보 세 정책의 env.yaml 계약이 같은지.
  5. 젯슨 요구 파일 목록(코드가 읽는 파일만) + reports/night_1007/jetson/*md5*.txt(누군가 받아 둔 목록)가 있으면 그것과 대조. **젯슨에 접속하지 않는다.**
  6. 명명 제안, 분리 계획(옮기면 깨지는 참조 목록).

분류표를 고치는 법: 새 판정이 나오면 CURATED 에 `_c(이름 정규식, 분류, 역할, 사유, [(문서, 문서에 그대로 있는 문구)], {핵심 체크포인트: 이유})` 한 줄을 넣는다.
금지: 이 스크립트는 아무것도 지우거나 옮기지 않는다. GPU·Isaac·ROS·젯슨을 쓰지 않는다(torch 는 CPU 로 가중치·차원만 읽는다, --no-torch 로 끔).
"""
from __future__ import annotations

import argparse
import ast
import collections
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # torch 가 GPU 를 건드리지 않게(차원 읽기만)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LOGROOT = os.path.join(REPO, "logs", "rsl_rl")
OUT_JSON = os.path.join(HERE, "policy_inventory.json")
OUT_MD = os.path.join(HERE, "POLICY_INVENTORY.md")
MODEL_RE = re.compile(r"^model_(\d+)\.pt$")
TS_RE = re.compile(r"(20\d\d-\d\d-\d\d_\d\d-\d\d-\d\d)([A-Za-z0-9_]*)")

C_DEPLOY, C_EVAL, C_BASE, C_REC, C_DISCARD, C_UNK = "배포 후보", "평가 중 후보", "기준선·검증 의존", "기록 보존", "폐기 후보", "불명"
CLASS_ORDER = [C_DEPLOY, C_EVAL, C_BASE, C_REC, C_DISCARD, C_UNK]
KEEP = {C_DEPLOY, C_EVAL, C_BASE, C_REC}

#: 합성 구조에서 기대하는 (역할 → 체크포인트). 레지스트리에서 뽑은 run 의 exported/SOURCE_CHECKPOINT 와 다르면 '불일치'로 표시한다.
EXPECT_DEPLOY = {"서기(STAND)": 11300, "회복(RECOVERY)": 21793, "걷기(WALK)": 15800}
#: 학습 태스크(코드·문서로 확인한 것만). 그 밖은 run 이름과 rl_walking/__init__.py 등록 이름의 일치로 '추정'.
TASK_VERIFIED = {
    "2026-10-02_21-23-08_g21a": ("Biped12-HangStand-G21a-v0", "reports/gen_g18/runner_equiv_g21a.txt 1 행(이 태스크와 이 run 의 exported 를 짝지어 러너 동치를 봄) · rl_walking/hang_stand_env_cfg.py Biped12HangStandG21aEnvCfg"),
    "2026-10-04_10-08-41_g24ra_seed1_from_w2e20294": ("Biped12-HangStand-G24Ra-v0", "reports/gen_c5/g24rcont/hypothesis.md('변경: 없음(Biped12-HangStand-G24Ra-v0')"),
    "2026-10-07_00-11-12_w3S3d_from_s2d": ("Biped12-HangStand-W3S3d-v0", "rl_walking/deploy/w3_deploy.py W3S3D_TASK · reports/gen_w3/REPRO.md"),
    "2026-10-07_21-51-10_w3S3e_from_s3d": ("Biped12-HangStand-W3S3e-v0", "reports/gen_w3/scripts/w3s3e.sh · reports/gen_w3/S3E_DESIGN.md §2"),
    "2026-10-07_22-25-32_w3S3f_from_15800": ("Biped12-HangStand-W3S3f-v0", "reports/gen_w3/TORQUE_BUDGET_1007.md §7"),
    "2026-10-07_22-52-26_w3S3g_from_15800": ("Biped12-HangStand-W3S3g-v0", "reports/gen_w3/TRANSITION_W3KE.md §5.19"),
    "2026-10-07_05-33-59_w3S3d_seed2": ("Biped12-HangStand-W3S3d-v0", "reports/gen_w3/REPRO.md 방법"),
    "2026-10-07_06-43-54_w3S3d_seed3": ("Biped12-HangStand-W3S3d-v0", "reports/gen_w3/REPRO.md 방법"),
    "2026-10-07_10-15-20_w3S3d_seed4": ("Biped12-HangStand-W3S3d-v0", "reports/gen_w3/REPRO.md 방법"),
    "2026-10-07_10-28-16_w3S3d_seed5": ("Biped12-HangStand-W3S3d-v0", "reports/gen_w3/REPRO.md 방법"),
    "2026-10-04_17-30-51_g24rcont_seed1_from_g24ra21793": ("Biped12-HangStand-G24Ra-v0", "reports/gen_c5/g24rcont/hypothesis.md"),
}
#: '현행' 스크립트 위치 — 여기서 체크포인트를 직접 읽으면 기준선·검증 의존. 그 밖(reports/gen_g*, reports/real_* …)은 과거 분석 스크립트로 보고 기록 보존.
CURRENT_DIRS = ("jetson/", "rl_walking/", "tests/", "deploy/", "scripts/", "reports/regress/", "reports/gen_w3/")
ACTIVE_HOURS = 24.0      # 분류표에 없고 최근 이 시간 안에 파일이 바뀐 run = 평가 중 후보(자동)

LOG = "reports/night_1005/LOG.md"
G6R = "reports/gen_g6/RECOVERY_EVAL_0920.md"


def _c(pat, cls, role, why, ev=(), keys=None, note=None):
    return dict(pat=re.compile(pat), cls=cls, role=role, why=why, ev=list(ev), keys=dict(keys or {}), note=note)


# ---------------------------------------------------------------------------------------------------------------------
# 분류표(문서 판정). pat = run 디렉터리 이름(타임스탬프 포함)에 fullmatch. 위에서부터 첫 일치.
# cls 는 '문서만으로 본' 분류다 — 자동 규칙(레지스트리·selftest·tests·현행 스크립트·계보)이 더 강하면 그쪽이 이긴다.
# ev = [(문서 경로, 문서 안에 그대로 있는 문구)] — 실행할 때마다 다시 찾아 줄 번호를 붙인다.
# 역할 말뜻: 기각 = 후보(처치)로 평가했고 사전 기준 탈락 · 무효 = 스모크·벤치·사고·중단 · 분리 실험 = 후보가 아닌 진단/절제 실험(끝남)
#           같은 실험 대조 = 그 실험의 처치가 전부 기각된 대조군 · 대조군 = 여러 판정 문서가 수치로 인용 · 선정 근거 = 남기는 정책을 고른 근거 수치
# ---------------------------------------------------------------------------------------------------------------------
TS = r"20\d\d-\d\d-\d\d_\d\d-\d\d-\d\d"
CURATED = [
    # ── 0719·0725 평지 세대(45 차원, 10.49 kg 자산) — RL보행_명령어모음.txt §6 이 역할을 적어 둔 것만
    _c(r"2026-07-19_04-21-02", C_REC, "문서 인용(구세대)", "런1 — 걸음새(케이던스 2.0/s), 이식 가이드가 exported 를 가리킴",
       [("RL보행_명령어모음.txt", "2026-07-19_04-21-02   런1"), ("docs/RL보행_이식가이드.md", "2026-07-19_04-21-02/exported/policy.onnx")], {2999: "log_picture/2026-07/정책평가_run1.txt"}),
    _c(r"2026-07-19_05-32-42", C_REC, "문서 인용(구세대)", "3단계 '이식용 최종'(구 자산)", [("RL보행_명령어모음.txt", "2026-07-19_05-32-42   3단계")], {8997: "평가 산출물"}),
    _c(r"2026-07-19_07-15-42", C_REC, "문서 인용(구세대)", "계단 최종(구 자산)",
       [("RL보행_명령어모음.txt", "2026-07-19_07-15-42   계단 최종"), ("docs/RL보행_최종보고서.md", "최종 계단 체크포인트: `logs/rsl_rl/biped12_flat/2026-07-19_07-15-42/`")], {13995: "평가 산출물"}),
    _c(r"2026-07-25_21-07-04", C_REC, "문서 인용(구세대)", "신 URDF 미세조정(model_9999)", [("RL보행_명령어모음.txt", "2026-07-25_21-07-04   신URDF(12URDF0725) 미세조정")], {9999: "문서"}),
    _c(r"2026-07-25_21-59-19", C_REC, "문서 인용(구세대)", "'신자산 이식용 최종'(model_12300)", [("RL보행_명령어모음.txt", "2026-07-25_21-59-19   ★신자산 이식용 최종")], {12300: "문서"}),
    _c(r"2026-07-25_22-04-32", C_REC, "문서 인용(구세대)", "폴리시 계속(model_13499) — 대안", [("RL보행_명령어모음.txt", "2026-07-25_22-04-32   폴리시 계속")], {13499: "문서"}),
    _c(r"2026-07-25_(22-36-10|23-00-29)", C_DISCARD, "기각", "대칭 교정 시도 2회 — 문서가 '실패 기록'으로 적음", [("RL보행_명령어모음.txt", "대칭 교정 시도 2회 — 실패 기록")]),
    _c(r"2026-07-25_23-44-35", C_REC, "문서 인용(구세대)", "계단 신자산 재정합(model_14994)", [("RL보행_명령어모음.txt", "2026-07-25_23-44-35   ★계단 신자산 재정합")], {14994: "문서"}),
    # ── stand_v18b (0905) — 자동(레지스트리)으로 잡힌다. 문서 근거만 덧붙임
    _c(r"2026-09-05_19-37-39", C_REC, "이전 데뷔 후보", "stand_v18b '데뷔 1순위'(0905~0906) — 체크포인트 원본은 없고 exported 만 남아 있음",
       [("models/README.md", "| **stand_v18b** | 09-05_19-37-39 / model_120998 |")]),
    # ── G2~G10 (0920~0922)
    _c(TS + r"_g2", C_REC, "문서 인용", "0920 재판정 표의 비교 행(커리큘럼 의존 결론 철회의 근거)", [("reports/gen_g6/SUMMARY.md", "| G2 model_5500 | 명목 |")], {5500: "SUMMARY §2"}),
    _c(TS + r"_g3", C_REC, "문서 인용", "0920 재판정 표의 비교 행", [("reports/gen_g6/SUMMARY.md", "| G3 model_5400 | 명목 / DR(상속 광대역) |")], {5400: "SUMMARY §2", 5600: "SUMMARY §2(피커)"}),
    _c(TS + r"_g5", C_DISCARD, "기각", "명목 0/256 '진짜 실패'", [("reports/gen_g6/SUMMARY.md", "| G5 model_5999 | 명목 | 0/256 (진짜 실패")]),
    _c(TS + r"_g6", C_REC, "이전 기준선", "G6-4500 = 0920 재평가 기준선(이후 G9 의 웜스타트 원본)", [("reports/gen_g6/SUMMARY.md", "| G6 model_4500 | 명목 | 256/256 | 256/256 | 256/256 |")], {4500: "기준선·G9 부모", 4900: "피커"}),
    _c(TS + r"_g7", C_DISCARD, "기각", "G7 은 승격 대상이 아니다", [(G6R, "**판정**: G7 은 승격 대상이 아니다")]),
    _c(TS + r"_g8[abcd]", C_DISCARD, "분리 실험", "action-rate 배율 × env LPF 분리 실험(각 800 반복) — 결론이 문서에 있고 후보가 아님", [(G6R, "### 8.4 분리 실험 (action-rate 배율 × env LPF, 각 800 반복 웜스타트")]),
    _c(TS + r"_g8", C_DISCARD, "기각", "G8 체크포인트 선택 — 적격 후보 없음(익스포트 사본은 deploy/cand/g8_8499 에 따로 있음)", [(G6R, "### 8.8 G8 체크포인트 선택 결과 — **이번에는 승격 없음(적격 후보 없음)**")]),
    _c(TS + r"_g9", C_REC, "이전 실물 정책", "stand_crouch_v5(G9-5700) — 0922 실물 정책", [(G6R, "### 12.1 결과 — **처음으로 고정 기준(§8.6)을 통과한 후보: G9 model_5700**")], {5700: "stand_crouch_v5", 8499: "deploy/cand"}),
    _c(TS + r"_g10", C_REC, "이전 실물 정책", "stand_crouch_v6(G10-6600) — 0922 실물 정책", [("reports/real_0922/G10_VERDICT.md", "# G10 체크포인트 판정 결과")], {6600: "stand_crouch_v6"}),
    # ── G11~G12
    _c(TS + r"_g11a", C_REC, "이전 비교 시험 후보", "G11a-8699 — 실물 비교 시험 후보(승격 아님), G12 의 부모", [("reports/gen_g11/VERDICT.md", "| **G11a 8699** |")], {8699: "stand_crouch_g11a"}),
    _c(TS + r"_g11b", C_DISCARD, "기각", "G11 판정 — 적격 후보 없음, G11b 는 G11a 대비 일관된 이득 없음", [("reports/gen_g11/VERDICT.md", "**적격 후보 없음 → 승격 없음.**"), ("reports/gen_g11/VERDICT.md", "G11b(정착 보너스 게이트 해제)는 G11a 대비 일관된 이득이 없다")]),
    _c(TS + r"_g11c", C_DISCARD, "기각", "G11c 판정 — 승격 없음(정착↔뒤 회복 교환)", [("reports/gen_g11/VERDICT.md", "## G11c 판정 (2026-09-29 13:3x) — 승격 없음")]),
    _c(TS + r"_g12", C_REC, "기준선", "G12-9500 = G13~G18 판정의 배포 기준선", [("reports/gen_g11/G12_VERDICT.md", "| **G12-9500** |")], {9500: "stand_crouch_g12", 10000: "G12 판정", 10698: "G12 판정"}),
    _c(TS + r"_g12c", C_REC, "대조군", "이어 학습 대조(G12-9500 에서 변경 없이 2000 반복) — G13·G14·G15·G16·G17 판정 문서가 수치로 인용",
       [("reports/gen_g13/G13_VERDICT.md", "G12c run 2026-09-30_12-05-46_g12c"), ("reports/gen_g13/G14_VERDICT.md", "| G12c-10000 / 10500 / 11000 / 11499 |"),
        ("reports/gen_g15/G15_VERDICT.md", "G12c-10500 / 11499 (대조: 이어 학습만)"), ("reports/gen_g16/G16_VERDICT.md", "G12c(legacy 학습) vs G16o(v2 학습)"),
        ("reports/gen_g17/G17_VERDICT.md", "## 이어 학습 대조 G12c")], {10000: "G13", 10500: "G13·G15·jac_blocks", 10800: "jac_blocks", 11000: "G13", 11100: "jac_blocks", 11499: "G13·G15·jac_blocks", 10100: "G17", 10400: "G17", 10700: "G17"}),
    # ── G13~G16
    _c(TS + r"_g13init", C_DISCARD, "기각", "G13 의 시작 체크포인트(관측 차원을 넓힌 G12-9500 사본) — G13 과 같은 판정", [("reports/gen_g13/G13_VERDICT.md", "# G13 판정 (2026-09-30 15:2x) — 통과 후보 없음, G12-9500 유지")]),
    _c(TS + r"_g13", C_DISCARD, "기각", "G13 판정 — 통과 후보 없음", [("reports/gen_g13/G13_VERDICT.md", "# G13 판정 (2026-09-30 15:2x) — 통과 후보 없음, G12-9500 유지")]),
    _c(TS + r"_g14[ab]", C_DISCARD, "기각", "G14 판정 — 가설 미지지, 통과 후보 없음", [("reports/gen_g13/G14_VERDICT.md", "# G14 판정 (2026-09-30 19:2x) — 가설 미지지, 통과 후보 없음 → G12-9500 유지")]),
    _c(TS + r"_g15[wqudk]", C_DISCARD, "기각", "G15 판정 — 통과 후보 없음", [("reports/gen_g15/G15_VERDICT.md", "# G15 판정 (2026-10-01 01:2x) — 통과 후보 없음, G12-9500 유지")]),
    _c(TS + r"_g16a43", C_DISCARD, "기각", "G16a seed 43 — 학습 붕괴 재현, 10719 반복에서 중단", [("reports/gen_g16/G16_VERDICT.md", "G16a-s43 은 10719 반복에서 명시 PID 로 중단")]),
    _c(TS + r"_g16o", C_DISCARD, "기각", "G16 판정 — H-o(관측 v2 학습 비퇴보) 기각", [("reports/gen_g16/G16_VERDICT.md", "- **최종: 통과 후보 없음 → 아침 실물 정책 G12-9500 유지**")],
       {10500: "reports/gen_g16/jac_blocks.py", 10800: "reports/gen_g16/jac_blocks.py", 11100: "reports/gen_g16/jac_blocks.py", 11499: "reports/gen_g16/jac_blocks.py"}),
    _c(TS + r"_g16(o43|a|d)", C_DISCARD, "기각", "G16 판정 — H-o·H-a·H-d 기각, 통과 후보 없음", [("reports/gen_g16/G16_VERDICT.md", "- **최종: 통과 후보 없음 → 아침 실물 정책 G12-9500 유지**")]),
    # ── G17~G22
    _c(TS + r"_g17[or]_void", C_DISCARD, "무효", "G17 1차 학습 폐기(접촉 관측 계약 불일치로 중단)", [("reports/gen_g17/G17_CRITERIA.md", "**G17 1차 학습 폐기**")]),
    _c(TS + r"_g17(o|r|p|o43|rp|r43)", C_DISCARD, "기각", "G17 판정 — 선별 C0 통과 0/24", [("reports/gen_g17/G17_VERDICT.md", "선별 C0 통과 체크포인트 **0 / 24**")]),
    _c(TS + r"_g18(o|j|d|q|f|f43)", C_DISCARD, "기각", "G18 판정 — 배포 자격(C0) 통과 0(G18f 는 '가장 유망'이었으나 통과 못 함)", [("reports/gen_g18/G18_VERDICT.md", "- 배포 자격(C0) 통과 체크포인트 **0** → 실물 정책은 계속 G12-9500.")]),
    _c(TS + r"_g19v", C_REC, "선행 단계", "G19v-10700 = G21 설계의 기준(G21a 의 웜스타트 원본)", [("reports/gen_g18/G21_VERDICT.md", "기준 G19v-10700")], {10700: "G21a 부모", 11000: "G19 판정 최근접"}),
    _c(TS + r"_g19n", C_DISCARD, "기각", "G19n — 회복 상실한 정숙 정책, 승격 금지", [("reports/gen_g18/G19_DESIGN_CRITERIA.md", "회복 상실한 정숙 정책, **승격 금지**")]),
    _c(TS + r"_g20[ds]", C_DISCARD, "기각", "G20d·G20s — 둘 다 정식 C0 탈락", [("reports/gen_g18/G20_DESIGN_CRITERIA.md", "- **둘 다 정식 C0 탈락 → 배포 자격 없음.**")]),
    _c(TS + r"_g21a_void_explosion", C_DISCARD, "무효", "G21a 1차 — 물리 폭발 전염으로 무효(재실행이 배포본)", [("reports/gen_g18/G21_DESIGN_CRITERIA.md", "**G21a 1차 무효**")]),
    _c(TS + r"_g21a", C_REC, "최종 STAND 후보", "G21a-11300 = 최종 STAND 후보", [("reports/FINAL_REPORT_1005.md", "## 1. 최종 STAND 후보"), ("reports/gen_g18/G21_VERDICT.md", "| **G21a-11300 (재실행)** |")], {11300: "배포", 11600: "G21 선별"}),
    _c(TS + r"_g21(o|b|a1|a2|a3|c1|c2|a3s43|d)", C_DISCARD, "기각", "G21 판정 — 배포 자격 통과 0(G21a 만 진단 실물 시험 후보로 남음)", [("reports/gen_g18/G21_VERDICT.md", "- **배포 자격 통과 0**")]),
    _c(TS + r"_g22c_void_stopped", C_DISCARD, "무효", "G22c — 권고 순서에 따라 중단(무효)", [("reports/gen_g18/G22_DESIGN_CRITERIA.md", "권고 순서에 따라 중단(무효, `_void_stopped`)")]),
    _c(TS + r"_g22(a|w|ws43)", C_DISCARD, "기각", "G22 판정 — H-22 미지지, 정식 C0 탈락", [("reports/gen_g18/G22_DESIGN_CRITERIA.md", "- **H-22 미지지**(G22a·G22w 모두)"), ("reports/gen_g18/G22_DESIGN_CRITERIA.md", "**C0 탈락**(G21a-11300 정식 2.2 / 5.6 보다 나쁨)")]),
    # ── G22R·W1 (1003)
    _c(r"2026-10-03_18-40-32_g22r_.*", C_DISCARD, "무효", "G22R 첫 기동분(같은 이름으로 19:01 에 다시 시작) — _void_1003_bench 로 옮겨져 있음", []),
    _c(TS + r"_g22r2_.*", C_REC, "참조 데이터 출처", "G22R2-13099 롤아웃이 스텝 기준 데이터(ref_steps_g22r2_s1.npz)·이륙 직전 진단의 출처 — 판정 자체는 탈락",
       [("reports/gen_g22r/verdict_g22r2_13099_s1.md", "판정(사전 기준): 탈락"), ("reports/gen_step/CORRECTION_1004_forward_steps.md", "G22R2-13099"), ("reports/gen_g22r/STEPB0_CRITERIA.md", "근거: diag_pretakeoff3_g22r2.md")], {13099: "참조 데이터", 12500: "판정"}),
    _c(TS + r"_g22r_.*", C_DISCARD, "기각", "G22R·W1 판정 — 통과 0", [("reports/gen_g22r/VERDICT_1003.md", "**통과 0 → G21a-11300 유지**")]),
    _c(TS + r"_w1_.*", C_DISCARD, "기각", "W1 — 착지 0(걷지 않음)", [("reports/gen_g22r/VERDICT_1003.md", "**W1 착지 0 — 걷지 않음**")]),
    _c(TS + r"_w1b_.*", C_DISCARD, "기각", "W1b — 스텝 0", [("reports/gen_step/w2a/hypothesis.md", "W1/W1b 는 스텝 0.")]),
    # ── STEP 부트스트랩(1003 밤~1004)
    _c(TS + r"_stepb0_.*", C_DISCARD, "기각", "STEP-B0 — 탈락: 스텝 0", [("reports/gen_g22r/STEPB0_CRITERIA.md", "## 결과 (1004 00:22, 선별 seed 1) — **탈락: 스텝 0**")]),
    _c(TS + r"_stepb1_.*", C_DISCARD, "기각", "STEP-B1 — FAIL(fail-fast)", [("reports/gen_step/stepb1/decision.md", "# STEP-B1 decision — **FAIL (fail-fast, 600 반복에서 중단)**")]),
    _c(TS + r"_stepi0_.*", C_DISCARD, "기각", "STEP-I0 — FAIL(fail-fast)", [("reports/gen_step/stepi0/decision.md", "# STEP-I0 decision — **FAIL (fail-fast, 400 반복에서 중단)**")]),
    _c(TS + r"_stepa0_.*", C_DISCARD, "기각", "STEP-A0 — FAIL(fail-fast 600)", [("reports/gen_step/stepa0/decision.md", "# STEP-A0 decision — **FAIL (fail-fast 600)")]),
    _c(TS + r"_stepa0b_.*", C_DISCARD, "기각", "STEP-A0b — A0·A0b·A0c 3회 같은 실패", [("reports/gen_step/stepa0c/decision.md", "A0·A0b·A0c 3회 같은 실패")]),
    _c(TS + r"_stepa0c_.*", C_DISCARD, "기각", "STEP-A0c — FAIL", [("reports/gen_step/stepa0c/decision.md", "# STEP-A0c decision — **FAIL (fail-fast 400)**")]),
    _c(TS + r"_stepa0e_.*", C_DISCARD, "기각", "STEP-A0e — FAIL", [("reports/gen_step/stepa0e/decision.md", "# STEP-A0e decision — **FAIL (fail-fast 400)**")]),
    _c(TS + r"_stepa0f_.*", C_DISCARD, "무효", "A0f — 시작 직후 중단, 무효", [("reports/gen_step/stepa0f/decision.md", "A0f: 시작 직후(약 2 분) 중단")]),
    _c(TS + r"_stepa0g_.*", C_DISCARD, "기각", "STEP-A0g — FAIL", [("reports/gen_step/stepa0g/decision.md", "# STEP-A0g decision — **FAIL (fail-fast 400)**")]),
    _c(TS + r"_stepa0h_.*", C_DISCARD, "기각", "STEP-A0h — FAIL", [("reports/gen_step/stepa0h/decision.md", "# STEP-A0h decision — **FAIL (fail-fast 400)**")]),
    _c(TS + r"_stepa0j_.*", C_DISCARD, "무효", "A0j — 시작 직후 중단·보류", [("reports/gen_step/stepa0i/decision.md", "A0j(놓친 스텝 벌점, G21a 재시작)는 시작 직후 중단·보류")]),
    _c(TS + r"_stepa0ic_.*", C_DISCARD, "기각", "STEP-A0i-c — FAIL(진행 정체)", [("reports/gen_step/stepa0ic/decision.md", "# STEP-A0i-c decision — **FAIL(진행 정체")]),
    _c(TS + r"_stepa0i_.*", C_DISCARD, "기각", "STEP-A0i — 사전 규칙 FAIL(단, 첫 양성 신호 — A0m 의 부모)", [("reports/gen_step/stepa0i/decision.md", "# STEP-A0i decision — **FAIL")], {11900: "A0m 부모"}),
    _c(TS + r"_stepa0k_.*", C_DISCARD, "기각", "STEP-A0k — FAIL", [("reports/gen_step/stepa0k/decision.md", "# STEP-A0k decision — **FAIL(")]),
    _c(TS + r"_stepa0m_.*", C_REC, "선행 단계", "STEP-A0m — 스텝 primitive PASS(서기 보존 FAIL, 배포 후보 아님) — W2·G24R·W2om 계열의 출발점",
       [("reports/gen_step/stepa0m/decision.md", "STEP primitive PASS / 서기 보존 FAIL (Case E)")], {14299: "W2a·G23R·W2omP 부모", 13100: "decision"}),
    # ── W2 보행(1004~1005)
    _c(TS + r"_w2a_.*", C_DISCARD, "기각", "W2-1 — 부분 실패", [("reports/gen_step/w2a/decision.md", "# W2-1 decision — **부분 실패**")], {15798: "W2b 부모"}),
    _c(TS + r"_w2b_.*", C_DISCARD, "기각", "W2-2 — FAIL", [("reports/gen_step/w2b/decision.md", "# W2-2 decision — **FAIL**")], {16797: "W2c 부모"}),
    _c(TS + r"_w2c_.*", C_DISCARD, "기각", "W2-3 — FAIL", [("reports/gen_step/w2c/decision.md", "# W2-3 decision — **FAIL**")], {17796: "W2d 부모"}),
    _c(TS + r"_w2d_.*", C_DISCARD, "기각", "W2-4 — 사전 기준 FAIL(앞 회복 스텝 첫 출현)", [("reports/gen_step/w2d/decision.md", "# W2-4 decision — **사전 기준 FAIL / 앞 회복 스텝 첫 출현**")], {19295: "W2e 부모"}),
    _c(TS + r"_w2e_.*", C_DISCARD, "기각", "W2-5 — FAIL", [("reports/gen_step/w2e/decision.md", "# W2-5 decision — **FAIL**")], {20294: "G24Ra·W2f 부모"}),
    _c(TS + r"_w2f_.*", C_DISCARD, "기각", "W2-6 — FAIL", [("reports/gen_step/w2f/decision.md", "# W2-6 decision — **FAIL**")], {21793: "W2g 부모"}),
    _c(TS + r"_w2g_.*", C_DISCARD, "기각", "W2-7 — FAIL", [("reports/gen_step/w2g/decision.md", "# W2-7 decision — **FAIL**")], {23292: "W2h 부모"}),
    _c(TS + r"_w2h_.*", C_REC, "이전 최종 WALK 후보", "W2-8 24791 = 1005 사이클의 최종 WALK 후보(이후 W3 로 대체) — W2q·WT·WV·W2o 의 부모",
       [("reports/gen_walk/WALK_FINAL_1005.md", "## 최종 보행 후보 = W2-8"), ("reports/FINAL_REPORT_1005.md", "## 2. 최종 WALK 후보")], {24791: "후보", 24600: "인접 재현", 24700: "인접 재현"}),
    _c(TS + r"_w2i_.*", C_DISCARD, "기각", "W2-9 — 조기 중단(판정 불가)", [("reports/gen_step/w2i/decision.md", "# W2-9 decision — **조기 중단")]),
    _c(TS + r"_w2j_.*", C_DISCARD, "기각", "W2-10 — FAIL, 요(회전) 갈래 일시 정지", [("reports/gen_step/w2j/decision.md", "# W2-10 decision — **FAIL**")]),
    _c(TS + r"_w2w_.*", C_DISCARD, "기각", "W2w — 비구속 변수(폐기 목록)", [("reports/FINAL_REPORT_1005.md", "W2w: 비구속 변수")]),
    _c(TS + r"_w2o_.*", C_DISCARD, "기각", "W2o — 사건 보상 제거, 1000 에서 보행 붕괴 → 중단", [("reports/gen_walk/W2o_RESULT.md", "# W2o 결과 (1005) — 사건 보상 제거")]),
    _c(TS + r"_w2q2_.*", C_DISCARD, "기각", "W2q2 — 중단", [("reports/gen_walk/W2q2_RESULT_WR_PLAN.md", "## W2q2 — 중단")]),
    _c(TS + r"_w2qc_.*", C_REC, "대조군", "w2qc = W2q 이어 학습 대조(W2q2·WR 판정과 보행 최종 표가 인용)",
       [("reports/gen_walk/W2q2_RESULT_WR_PLAN.md", "| 대조 w2qc 26089 |"), ("reports/gen_walk/WALK_FINAL_1005.md", "| 대조 w2qc 25900(같은 반복) |")], {25900: "최종 표", 26089: "W2q2 판정"}),
    _c(TS + r"_w2qC_.*", C_DISCARD, "무효", "대조 C(W2q 이어학습) — 학습 붕괴, 대조 무효", [(LOG, "## 대조 C(W2q 이어학습) — 학습 붕괴, 대조 무효")]),
    _c(TS + r"_w2q_.*", C_REC, "이전 후보(HTURN2)", "W2q 25590 = HTURN2(앞 보행 + 회전 피드포워드) 실험 키의 정책 — 1006 '방향 전환 기술' 후보",
       [(LOG, "## HTURN2 결과 — **사전 규칙 통과"), ("reports/gen_walk/WALK_FINAL_1005.md", "| W2q 25590 (cadence 연동) |")], {25590: "HTURN2_MODEL", 25100: "W2q 계획", 25300: "W2q 계획"}),
    _c(TS + r"_wr_.*", C_DISCARD, "기각", "WR — 같은 반복 대조와 구별되지 않음", [("reports/gen_walk/WALK_FINAL_1005.md", "보폭 구조 대안 2개(W2q2·WR)는 같은 반복 대조와 구별되지 않았다")]),
    _c(TS + r"_w2gs_.*", C_DISCARD, "기각", "W2gs — 중단", [(LOG, "## W2gs 결과 — 중단(현재 best 보다 못함)")]),
    _c(TS + r"_woms_.*", C_DISCARD, "기각", "WOMS — KILL", [(LOG, "- **WOMS 결정: KILL**")]),
    _c(TS + r"_womx_.*", C_DISCARD, "기각", "WOMX — KILL", [(LOG, "## WOMX 1000 — KILL (사전 기준)")]),
    _c(TS + r"_wom_.*", C_DISCARD, "기각", "WOM — 중단", [(LOG, "## WOM +500 — 중단")]),
    _c(TS + r"_wt1_.*", C_DISCARD, "기각", "WT1 — 중단(학습 루프 내 요 제어 이득 없음)", [("reports/gen_walk/wt1/RESULT.md", "WT1 중단: 학습 루프 내 요 제어는 이득 없음")]),
    _c(TS + r"_wt0_.*", C_REC, "대조군", "WT0 = W2hSafe 대조 — WT·WV 두 결과 문서가 대조 수치로 인용", [("reports/gen_walk/wt1/RESULT.md", "| WT0 (W2hSafe 대조) |"), ("reports/gen_walk/wv1/RESULT.md", "WT0(대조)")], {25190: "WT·WV 대조"}),
    _c(TS + r"_wv1_.*", C_DISCARD, "기각", "WV1 — 중단", [("reports/gen_walk/wv1/RESULT.md", "# WV 결과 (1005) — WV1 중단")]),
    _c(TS + r"_wv2_.*", C_DISCARD, "기각", "WV2 — 중단", [("reports/gen_walk/wv1/RESULT.md", "## WV2 (상한만 확장) — 중단")]),
    _c(TS + r"_wv3_.*", C_DISCARD, "기각", "WV3 — 중단", [("reports/gen_walk/wv1/RESULT.md", "## WV3 (속도 추종 std 0.1 → 0.03) — 중단")]),
    _c(TS + r"_wsteer2_.*", C_DISCARD, "기각", "WSTEER2 — KILL", [(LOG, "## WSTEER2 선택 평가 — KILL")]),
    _c(TS + r"_wsteer_.*", C_DISCARD, "기각", "WSTEER — KILL", [(LOG, "## WSTEER +1000(26589) — KILL")]),
    _c(TS + r"_cstab_.*", C_DISCARD, "분리 실험", "CSTAB — 학습 틀 안정성 진단(결론 '확률적 사건'), 후보 아님", [(LOG, "## CSTAB 결과 — '확률적 사건'(사전 규칙)")]),
    _c(TS + r"_w2omA_.*", C_DISCARD, "기각", "W2omA — KILL", [(LOG, "## W2omA +300 — KILL")]),
    _c(TS + r"_w2omL_.*", C_DISCARD, "기각", "W2omL — KILL", [(LOG, "## W2omL +300 — KILL")]),
    _c(TS + r"_w2omM_.*", C_DISCARD, "기각", "W2omM — KILL", [(LOG, "## W2omM +300 — KILL")]),
    _c(TS + r"_w2omPs2_.*", C_DISCARD, "기각", "W2omP seed 2 — 양성 신호만 재현, W2omP 계보는 KILL 로 끝남", [(LOG, "## W2omP-s2 +300 — 양성 신호 재현"), (LOG, "## W2omP 끝(14898, α_flex 0 평가) — KILL")]),
    _c(TS + r"_w2omP2_.*", C_DISCARD, "기각", "W2omP2 — KILL", [(LOG, "## W2omP2 +500 — KILL")]),
    _c(TS + r"_w2omP3_.*", C_DISCARD, "기각", "W2omP3 — KILL", [(LOG, "## W2omP3 — KILL")]),
    _c(TS + r"_w2omP_.*", C_DISCARD, "기각", "W2omP — KILL", [(LOG, "## W2omP 끝(14898, α_flex 0 평가) — KILL")]),
    _c(TS + r"_w2omI_.*", C_DISCARD, "기각", "W2omI — KILL", [(LOG, "## W2omI +300 — KILL")]),
    _c(TS + r"_w2omR[12]_.*", C_DISCARD, "기각", "W2omR1·R2 — 둘 다 KILL", [(LOG, "## W2omR1·R2 결과 — 둘 다 KILL")]),
    _c(TS + r"_w2omF_.*", C_DISCARD, "기각", "W2omF — KILL(학습 전체 붕괴)", [(LOG, "## W2omF 결과 — KILL(학습 전체 붕괴)")]),
    _c(TS + r"_w2om_.*", C_DISCARD, "기각", "W2om — 사전 기준 KILL", [(LOG, "## W2om +510 (model_26100) — 사전 기준 KILL")]),
    # ── 회복 G23R·G24R·토너먼트(1004~1006)
    _c(TS + r"_g23ra_.*", C_DISCARD, "기각", "G23R-a — FAIL", [("reports/gen_step/g23ra/decision.md", "# G23R-a decision — **FAIL**")]),
    _c(TS + r"_g23rb_.*", C_DISCARD, "기각", "G23R-b — FAIL(진행 정체로 중단)", [("reports/gen_step/g23rb/decision.md", "# G23R-b decision — **FAIL")]),
    _c(TS + r"_g24ra_.*", C_REC, "최종 RECOVERY 정책", "G24R-a-21793 = C5/KE 의 회복 정책(단독 사전 기준은 FAIL, 전환 구조에서 채택)",
       [("reports/gen_c5/G3_SELECT_1004.md", "- **회복 체크포인트 = G24R-a-21793 확정**"), ("reports/FINAL_REPORT_1005.md", "## 3. 최종 RECOVERY 후보")],
       {21793: "배포", 21700: "KE 회귀(CKPT 21700)", 21400: "G4 우연 배제", 21500: "G4", 21600: "G4"}),
    _c(TS + r"_g24rb_.*", C_DISCARD, "기각", "G24R-b — FAIL", [("reports/gen_step/g24rb/decision.md", "# G24R-b decision — **FAIL**")]),
    _c(TS + r"_g24rc_.*", C_DISCARD, "기각", "G24R-c — 사전 기준 FAIL, C5 안에서도 차이 없음", [("reports/gen_step/g24rc/decision.md", "# G24R-c decision — **사전 기준 FAIL**"), ("reports/gen_c5/c5c/decision.md", "# C5-c decision — **차이 없음**")]),
    _c(TS + r"_g24rd_.*", C_DISCARD, "기각", "G24R-d — FAIL(앞 회복 유의 악화)", [("reports/gen_c5/g24rd/decision.md", "# G24R-d decision — **FAIL")]),
    _c(TS + r"_g24re_.*", C_DISCARD, "기각", "G24R-e — FAIL", [("reports/gen_c5/g24re/decision.md", "# G24R-e decision — **FAIL**")]),
    _c(TS + r"_g24rf_.*", C_DISCARD, "무효", "G24R-f — 중단(옆 외란이 앞/뒤 사건을 대체하는 교란)", [("reports/gen_c5/g24rcont/hypothesis.md", "- G24R-f 는 중단(")]),
    _c(TS + r"_g24rg_.*", C_DISCARD, "기각", "G24R-g — fail-fast 중단", [("reports/gen_c5/g24rg/decision.md", "# G24R-g decision — **fail-fast 중단(+907)**")]),
    _c(TS + r"_g24rh_.*", C_DISCARD, "기각", "G24R-h — 잠정 FAIL", [("reports/gen_c5/g24rh/decision.md", "# G24R-h decision — **잠정 FAIL**")]),
    _c(TS + r"_g24rcontlr_.*", C_REC, "선정 근거", "G24R-a-cont-lr — 회복 체크포인트 선정 게이트(G3)의 비교 대상 4 체크포인트",
       [("reports/gen_c5/G3_SELECT_1004.md", "| cont-lr 23000 |"), ("reports/gen_c5/G3_SELECT_1004.md", "- **회복 체크포인트 = G24R-a-21793 확정**")], {23000: "G3", 23100: "G3", 23200: "G3", 23292: "G3"}),
    _c(TS + r"_g24rcont_.*", C_REC, "대조군", "G24R-a-cont — 이어 학습 대조(G24R-e·g·h 판정과 G3 선정 문서가 인용)",
       [("reports/gen_c5/g24rcont/hypothesis.md", "이어 학습 대조군"), ("reports/gen_c5/g24re/decision.md", "이어 학습 대조(cont)"), ("reports/gen_c5/G3_SELECT_1004.md", "cont 23292")], {22200: "e·g·h 대조", 22700: "e·g·h 대조", 23292: "e·h 대조·G3"}),
    _c(TS + r"_t1b_.*", C_DISCARD, "같은 실험 대조", "T1b — 토너먼트 1b 의 대조(처치 T2b = G25 계열 중단)", [("reports/gen_c5/tourney1/RESULT.md", "## 토너먼트 1b (연장, 수정 판정) — **G25 충격 커리큘럼 중단(kill)**")]),
    _c(TS + r"_t2b?_.*", C_DISCARD, "기각", "T2·T2b(G25 충격 커리큘럼) — 중단", [("reports/gen_c5/tourney1/RESULT.md", "## 토너먼트 1b (연장, 수정 판정) — **G25 충격 커리큘럼 중단(kill)**")]),
    _c(TS + r"_t3_.*", C_DISCARD, "기각", "T3(G25D 지연 DR 0–5) — 중단", [("reports/gen_c5/tourney2/RESULT.md", "# 토너먼트 2 결과 — **T3 G25D(지연 DR 0–5) 중단**")]),
    _c(TS + r"_t1_.*", C_REC, "대조군", "T1(안전 PPO 무변경 이어 학습) — 토너먼트 1·2·R′ 세 결과 문서의 같은 반복 대조",
       [("reports/gen_c5/tourney1/RESULT.md", "| T1 22000/22200/22292 |"), ("reports/gen_c5/tourney2/RESULT.md", "같은 반복 대조 T1"), ("reports/gen_c5/rcap/RESULT.md", "R0 대조(T1)")], {22000: "대조", 22200: "대조", 22292: "대조"}),
    _c(TS + r"_r2_.*", C_DISCARD, "기각", "R′ 토너먼트 r1n(run 이름은 r2) — 전부 중단", [("reports/gen_c5/rcap/NOTE_1005.md", "'r1n'으로 재명명해 평가에 사용한다"), ("reports/gen_c5/rcap/RESULT.md", "# R′ 토너먼트 결과 (1005) — 전부 중단")]),
    _c(TS + r"_r(1p|2p)_.*", C_DISCARD, "기각", "R′ 토너먼트 — 전부 중단", [("reports/gen_c5/rcap/RESULT.md", "# R′ 토너먼트 결과 (1005) — 전부 중단")]),
    _c(TS + r"_rd[35]_.*", C_DISCARD, "기각", "RD(rd5·대조 rd3) — 사전 성공 미달, RDS KILL 로 계보 종료", [(LOG, "## RD 결과 — 사전 성공 미달, kill 아님 → 결정: KE(G24R-a 21793) 유지"), (LOG, "## RDS 결과 — KILL(선별 통과 0) → RD 계보 종료")]),
    _c(TS + r"_rh2_.*", C_DISCARD, "기각", "RH2 — KILL", [(LOG, "## RH2 결과 — KILL")]),
    _c(TS + r"_pa1_.*", C_DISCARD, "기각", "PA1 — KILL", [(LOG, "## PA1 22300 — KILL")]),
    # ── W3 (1006~1007)
    _c(TS + r"_w3C_.*", C_DISCARD, "기각", "W3-C — Stage 0 토너먼트에서 A 가 1위(C 는 참조 0 에서 소멸)", [(LOG, "- 토너먼트 판정: **A가 보조 0 자립 기준으로 1위**")]),
    _c(TS + r"_w3CI_.*", C_DISCARD, "분리 실험", "W3-CI — 엔진 2 번째 경로 확인으로 종료(추가 학습 없음)", [(LOG, "- 정리: W3-CI 추가 학습 없음(엔진 2 번째 경로 확인으로 종료)")]),
    _c(TS + r"_w3U_.*", C_DISCARD, "기각", "W3-U — KILL", [(LOG, "## W3-U 400 — KILL(사전: 낙상 > 30 %)")]),
    _c(TS + r"_w3A_.*", C_REC, "선행 단계", "W3-A 400 — 보조 0·참조 0 제자리 교대걸음 자립(W3 계보의 뿌리, 실험 키 stand_w3a_11699)", [(LOG, "## W3-A 400 결과 — **보조 0·참조 0에서 제자리 교대걸음 자립**")], {11699: "W3_MODEL·W3-A2 부모"}),
    _c(TS + r"_w3A2_.*", C_REC, "선행 단계", "W3-A2 — W3 계보(W3-A3 의 부모)", [(LOG, "## W3-A2 끝(12698) 평가 — 낙상 분리 지표 첫 적용")], {12698: "W3-A3·S1b 부모", 12200: "중간 평가"}),
    _c(TS + r"_w3A3_.*", C_REC, "선행 단계", "W3-A3 — W3 계보(W3S1c 의 부모)", [(LOG, "## W3-A3 · W3S1b 끝 결과 + SPAWNCK")], {13197: "W3S1c 부모"}),
    _c(TS + r"_w3S1b_.*", C_DISCARD, "기각", "W3S1b — 뒤 추종 FAIL(W3S1c 로 대체)", [(LOG, "- W3S1b 판정(사전): (b) 앞 +0.067")]),
    _c(TS + r"_w3S1c2x_.*", C_DISCARD, "같은 실험 대조", "S1c2x — W3S1e 의 이어학습 대조(처치 미채택)", [(LOG, "## W3S1e(착지 보상 수리) vs S1c2x(이어학습 대조) — 미채택")]),
    _c(TS + r"_w3S1c2_.*", C_REC, "선정 근거", "W3S1c 학습 seed 2 — closure 1 재현성 판정의 수치", [(LOG, "## closure 1 재현성: W3S1c seed 2 — 앞은 재현, 영점·뒤는 미재현"), (LOG, "| S1c seed 2 (14196) |")], {14196: "재현성"}),
    _c(TS + r"_w3S1c3_.*", C_REC, "선정 근거", "W3S1c 학습 seed 3 — 'Stage 1 레시피는 seed 민감' 판정의 수치", [(LOG, "- W3S1c seed 3 끝(14196, 21:10)")], {14196: "재현성"}),
    _c(TS + r"_w3S1c_.*", C_REC, "선행 단계", "W3S1c-14196 — Stage 1 앞/뒤 통과(W3S2 의 부모)", [(LOG, "## ✅ W3S1c 끝(14196): Stage 1 앞/뒤 사전 기준 전부 통과")], {14196: "W3S2 부모·stopest_run.sh"}),
    _c(TS + r"_w3S1d2_.*", C_DISCARD, "기각", "W3S1d seed 2(대칭 범위) — 명령 범위 비대칭 가설 기각", [(LOG, "- 명령 범위 비대칭 가설 **기각**")]),
    _c(TS + r"_w3S1e_.*", C_DISCARD, "기각", "W3S1e — 미채택", [(LOG, "## W3S1e(착지 보상 수리) vs S1c2x(이어학습 대조) — 미채택")]),
    _c(TS + r"_w3S2d_.*", C_REC, "선행 단계", "W3S2d — 지연 DR 1–5(W3S3d 의 부모)", [(LOG, "## W3S3s2 끝 + 확장 평가 / W3S2d 지연 짝 비교")], {15694: "W3S3d 부모"}),
    _c(TS + r"_w3S2_.*", C_REC, "선행 단계", "W3S2-15195 — 옆걸음 획득(W3S2d·W3S3s2 의 부모)", [(LOG, "## W3S2 끝(15195) 진급 판정")], {15195: "W3S2d 부모"}),
    _c(TS + r"_w3S3s2_.*", C_REC, "선행 단계", "W3S3s2 — 제자리 회전 획득(W3S3d 의 보상 정의 출처, 이후 W3S3d 로 합침)",
       [(LOG, "## ✅ W3S3s2 +105(15300): 제자리 회전 획득"), (LOG, "**다음(단일 경로로 합침)**: W3S3d = W3S3s2 보상 + 학습 지연 1–5")], {15300: "회전 획득", 16194: "확장 평가"}),
    _c(TS + r"_w3S3d_seed[2-5]", C_REC, "선정 근거", "W3S3d 학습 seed 2–5 재현 — '15800 은 운 좋은 점, 레시피 기대 낙상 2–3 %' 판단의 수치(평가 원자료는 scratchpad repro/)",
       [("reports/gen_w3/REPRO.md", "## 결과 2 — 학습 seed 재현")], {15800: "REPRO +106", 15900: "REPRO +206", 15993: "REPRO +299"}),
    _c(TS + r"_w3S3dC2_.*", C_DISCARD, "기각", "c2(W3S3dC2) — KILL", [("reports/gen_w3/C2_DESIGN.md", "— **KILL(사전 규칙)**"), ("claude2.md", "기각·보관: c2 W3S3dC2")]),
    _c(TS + r"_w3S3dR_.*", C_DISCARD, "기각", "W3S3dR — 불합격, 기준선 15800 유지", [("reports/gen_w3/ROBUST_DESIGN.md", "- **판정: 불합격(①②③④). 기준선 15800 유지.**")]),
    _c(TS + r"_w3S3d_from_s2d", C_REC, "최종 WALK/TURN 후보", "W3S3d-15800 = 실물 WALK/TURN 1차 후보",
       [(LOG, "## ✅ W3S3d-15800 후보 확장 평가"), ("claude2.md", "| WALK/TURN 1차 | **W3S3d-15800**")], {15800: "배포", 15700: "REPRO 이웃", 15900: "REPRO 이웃", 16000: "REPRO 이웃", 16100: "REPRO 이웃", 16193: "REPRO 이웃·c2 시작"}),
    _c(TS + r"_w3S3e_.*", C_BASE, "대조군(진행 중인 실험)", "W3S3e — 후보로서는 '통과 후보 없음'(1007 22:52). 다만 진행 중인 W3S3g 의 사전 등록 대조가 이 run 의 같은 체크포인트(16100·16299)다 — W3S3g 판정이 끝나면 폐기 후보",
       [("reports/gen_w3/S3E_DESIGN.md", "**판정: 통과 후보 없음 → W3S3d-15800 유지."), ("reports/gen_w3/TRANSITION_W3KE.md", "**대조 = W3S3e 의 같은 체크포인트**")], {16100: "W3S3g 대조", 16299: "W3S3g 대조"}),
    _c(TS + r"_w3S3f_.*", C_EVAL, "학습·평가 중", "W3S3f — 토크 한계 무작위화(0.7–1.0) 단일 변경, 후보 체크포인트 16100·16299(사전 등록만 있고 결과 없음)",
       [("reports/gen_w3/TORQUE_BUDGET_1007.md", "## 7. W3S3f 사전 등록 — 토크 한계 무작위화 학습")], {16100: "평가 후보", 16299: "평가 후보"}, note="W3S3f"),
    _c(TS + r"_w3S3g_.*", C_EVAL, "학습·평가 중", "W3S3g — W3S3e + 쏠림 벌점(w3_lean) 단일 변경, 후보 체크포인트 16100·16299(사전 등록만 있고 결과 없음)",
       [("reports/gen_w3/TRANSITION_W3KE.md", "**W3S3g 사전 등록(학습 전)**")], {16100: "평가 후보", 16299: "평가 후보"}, note="W3S3g"),
    _c(TS + r"_w3S4dDb?_.*", C_DISCARD, "기각", "W3S4dD run1·run2 — 통과 후보 없음", [("reports/gen_w3/WD_DESIGN.md", "- **판정: 통과 후보 없음 → W3S4d-16299 유지.**"), ("claude2.md", "기각·보관: c2 W3S3dC2")]),
    _c(TS + r"_w3S4d2_.*", C_REC, "2차 후보", "W3S4d-16299 = 혼합 명령 2차 후보", [("claude2.md", "| 혼합 2차 | W3S4d-16299"), (LOG, "## W3S4d 끝(16299): 혼합 명령 성립")], {16299: "2차 후보", 16000: "REPRO", 16100: "REPRO", 16200: "REPRO"}),
    _c(TS + r"_w3S4d_.*", C_REC, "선행 단계", "W3S4d 첫 실행 — 디스크 가득 참으로 15934 에서 끊김, 15900 에서 w3S4d2 로 이어 학습", [(LOG, "## 운영 사고: 디스크 가득 참 → W3S4d 학습 중단·재개")], {15900: "W3S4d2 부모"}),
    # ── 문서 근거가 '가설·계획'뿐이라 불명으로 두는 것(이유를 적어 둠)
    _c(TS + r"_g24rglr_.*", C_UNK, "판정 기록 없음", "가설 문서만 있고 결과·중단 기록이 없다(체크포인트 2 개에서 멈춤)", [("reports/gen_c5/g24rglr/hypothesis.md", "# G24R-g-lr 가설")]),
    _c(TS + r"_g4", C_UNK, "판정 기록 없음", "문서에 '진행 중'으로만 남아 있다(6000 중 3000 반복에서 멈춤, 평가 산출물 0)", [("docs/PLANT_ID_SOLUTION_0920.md", "| G4 | G3 플랜트 + 엔코더측 관측 + 자세 벌점 ×3 |")]),
]
#: 이름만으로 무효로 보는 것(문서 근거가 따로 없는 스모크·벤치·점검 실행). 근거 = run 이름 + agent.yaml max_iterations + 평가 산출물 수(자동으로 붙임).
NAME_VOID = re.compile(r"(^|_)(smoke\d*|bench|g\d+smoke\d*|g\d+chk\d*|g\d+dbg\d*)(_|$)", re.I)
BENCH_EV = ("reports/gen_g22r/DESIGN_CRITERIA.md", "벤치(1003, RTX 5080 16 GB, 40/30 반복)")


# ---------------------------------------------------------------------------------------------------------------------
def rel(p):
    return os.path.relpath(p, REPO)


def md5_file(path, bufsize=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(bufsize), b""):
            h.update(c)
    return h.hexdigest()


def human(n):
    for u, d in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= d:
            return f"{n / d:.2f} {u}" if u == "GiB" else f"{n / d:.1f} {u}"
    return f"{n} B"


def read_text(path, limit=None):
    try:
        with open(path, "rb") as f:
            b = f.read(limit) if limit else f.read()
        return b.decode("utf-8", "replace")
    except OSError:
        return ""


class Md5Cache:
    """이전 JSON 의 md5 를 (경로, 크기, mtime_ns) 가 같을 때만 재사용."""

    def __init__(self, prev):
        self.m = {}
        self.hit = self.miss = 0
        for k, v in (prev or {}).get("md5_cache", {}).items():
            self.m[k] = v

    def get(self, path):
        st = os.stat(path)
        r = rel(path)
        e = self.m.get(r)
        if e and e[0] == st.st_size and e[1] == st.st_mtime_ns:
            self.hit += 1
            return e[2]
        self.miss += 1
        d = md5_file(path)
        self.m[r] = [st.st_size, st.st_mtime_ns, d]
        return d

    def peek(self, path):
        try:
            st = os.stat(path)
        except OSError:
            return None
        e = self.m.get(rel(path))
        return e[2] if e and e[0] == st.st_size and e[1] == st.st_mtime_ns else None


# ---------------------------------------------------------------------------------------------------------------------
# 1. run 훑기
def is_run_dir(p):
    try:
        names = os.listdir(p)
    except OSError:
        return False
    if "params" in names:
        return True
    return any(MODEL_RE.match(n) or n.startswith("events.out.tfevents") or n.startswith("exported") for n in names)


def scan_runs(root=LOGROOT):
    runs, aliases, strays = [], [], []

    def walk(d, depth):
        for e in sorted(os.scandir(d), key=lambda e: e.name):
            if e.is_symlink():
                tgt = os.path.realpath(e.path)
                aliases.append(dict(link=rel(e.path), points_to=rel(tgt) if tgt.startswith(REPO) else tgt, raw=os.readlink(e.path),
                                    exists=os.path.exists(tgt), is_dir=os.path.isdir(tgt)))
            elif e.is_dir(follow_symlinks=False):
                if is_run_dir(e.path):
                    runs.append(e.path)
                elif depth < 5:
                    walk(e.path, depth + 1)
            else:
                strays.append(rel(e.path))
    if os.path.isdir(root):
        walk(root, 0)
    return runs, aliases, strays


def run_facts(p, md5c, md5_all=False):
    name = os.path.basename(p)
    r = dict(path=rel(p), exp=rel(os.path.dirname(p)), name=name)
    m = TS_RE.match(name)
    r["ts"] = m.group(1) if m else None
    r["label"] = name[20:] if m and len(name) > 19 else ""
    ck, links, other, size, latest, tfe = {}, [], [], 0, 0.0, 0
    for e in sorted(os.scandir(p), key=lambda e: e.name):
        mm = MODEL_RE.match(e.name)
        if e.is_symlink():
            (links if mm else other).append(dict(name=e.name, points_to=os.readlink(e.path)))
            continue
        if mm and e.is_file():
            st = e.stat()
            ck[int(mm.group(1))] = dict(size=st.st_size, mtime=st.st_mtime)
        elif e.name.startswith("events.out.tfevents"):
            tfe += 1
        elif e.name not in ("params", "git") and not e.name.startswith("exported"):
            other.append(e.name)
    for dp, dn, fn in os.walk(p):
        for f in fn:
            fp = os.path.join(dp, f)
            if not os.path.islink(fp):
                st = os.lstat(fp)
                size += st.st_size
                latest = max(latest, st.st_mtime)
    r["size_bytes"] = size
    r["latest_mtime"] = latest
    r["latest"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(latest)) if latest else None
    nums = sorted(ck)
    r["ckpts"] = nums
    r["n_ckpt"] = len(nums)
    r["ckpt_min"], r["ckpt_max"] = (nums[0], nums[-1]) if nums else (None, None)
    r["ckpt_bytes"] = sum(v["size"] for v in ck.values())
    r["ckpt_sizes"] = {str(k): v["size"] for k, v in ck.items()}
    r["ckpt_links"] = links
    r["tfevents"] = tfe
    r["other_entries"] = other
    if nums:
        r["train_span"] = [time.strftime("%Y-%m-%d %H:%M", time.localtime(ck[nums[0]]["mtime"])), time.strftime("%Y-%m-%d %H:%M", time.localtime(ck[nums[-1]]["mtime"]))]
    # exported*
    ex = []
    for n in sorted(os.listdir(p)):
        d = os.path.join(p, n)
        if n.startswith("exported") and os.path.isdir(d) and not os.path.islink(d):
            files = {}
            for f in sorted(os.listdir(d)):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    files[f] = dict(size=os.path.getsize(fp), md5=md5c.get(fp))
            src = read_text(os.path.join(d, "SOURCE_CHECKPOINT")).strip() or None
            sm = MODEL_RE.match(src) if src else None
            ex.append(dict(dir=n, files=files, source_checkpoint=src, source_ckpt=int(sm.group(1)) if sm else None))
    r["exported"] = ex
    # params
    par = {}
    for f in ("env.yaml", "agent.yaml"):
        fp = os.path.join(p, "params", f)
        if os.path.isfile(fp):
            par[f] = dict(size=os.path.getsize(fp), md5=md5c.get(fp))
    r["params"] = par
    r["ckpt_md5"] = {}
    if md5_all:
        for k in nums:
            r["ckpt_md5"][str(k)] = md5c.get(os.path.join(p, f"model_{k}.pt"))
    else:
        for k in nums:
            v = md5c.peek(os.path.join(p, f"model_{k}.pt"))
            if v:
                r["ckpt_md5"][str(k)] = v
    return r


# ---------------------------------------------------------------------------------------------------------------------
# 2. params 요약
def _yaml_loader():
    import yaml
    base = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

    class L(base):
        pass

    def any_tag(loader, suffix, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)
    L.add_multi_constructor("tag:yaml.org,2002:python/", any_tag)
    L.add_multi_constructor("!", any_tag)
    return yaml, L


_POSES = {"CROUCH_V5": (0.5282, -0.6720, 0.4887), "CROUCH_V4": (0.4538, -0.5236, 0.3578), "CROUCH_V1": (0.2618, -0.5236, 0.3578)}


def _short(func):
    return str(func).split(":")[-1] if func else None


def summarize_agent(path):
    try:
        yaml, L = _yaml_loader()
        a = yaml.load(read_text(path), Loader=L) or {}
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e)[:120])
    pol, alg = a.get("policy") or {}, a.get("algorithm") or {}
    return dict(seed=a.get("seed"), max_iterations=a.get("max_iterations"), experiment_name=a.get("experiment_name"), run_name=a.get("run_name"),
                resume=a.get("resume"), load_run=a.get("load_run"), load_checkpoint=a.get("load_checkpoint"), save_interval=a.get("save_interval"),
                runner=a.get("class_name"), algo=alg.get("class_name"), actor=pol.get("actor_hidden_dims"), init_noise_std=pol.get("init_noise_std"),
                lr=alg.get("learning_rate"), mirror=bool((alg.get("symmetry_cfg") or {}).get("use_mirror_loss")))


def summarize_env(path):
    try:
        yaml, L = _yaml_loader()
        d = yaml.load(read_text(path), Loader=L) or {}
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e)[:120])
    out = {}
    sim = d.get("sim") or {}
    dt, dec = sim.get("dt"), d.get("decimation")
    out["control"] = dict(sim_dt=dt, decimation=dec, hz=round(1.0 / (dt * dec), 3) if dt and dec else None)
    sc = d.get("scene") or {}
    out["num_envs"] = sc.get("num_envs")
    out["episode_length_s"] = d.get("episode_length_s")
    rb = sc.get("robot") or {}
    usd = ((rb.get("spawn") or {}).get("usd_path") or "")
    out["usd"] = usd.replace(REPO + "/", "")
    jp = ((rb.get("init_state") or {}).get("joint_pos") or {})
    key = (jp.get("left_hip_f_joint"), jp.get("left_knee_joint"), jp.get("left_ankle_f_joint"))
    label = None
    if all(isinstance(v, (int, float)) for v in key):
        for n, ref in _POSES.items():
            if all(abs(a - b) < 1e-4 for a, b in zip(key, ref)):
                label = n
        out["default_pose"] = dict(left_hip_f=key[0], left_knee=key[1], left_ankle_f=key[2], label=label or "기타")
    else:
        out["default_pose"] = dict(label="관절명 사전 아님" if jp else "없음", raw=str(jp)[:80])
    act = ((d.get("actions") or {}).get("joint_pos") or {})
    mdl = act.get("max_delta_per_step")
    out["action"] = dict(cls=_short(act.get("class_type")), scale=act.get("scale"), use_default_offset=act.get("use_default_offset"),
                         slew_deg_per_tick=round(mdl * 57.29578, 3) if isinstance(mdl, (int, float)) else None,
                         delay_steps=[act.get("min_delay_steps"), act.get("max_delay_steps")] if "max_delay_steps" in act else None,
                         lpf_beta_range=act.get("lpf_beta_range"), backlash_rad_range=act.get("backlash_rad_range"))
    pol = ((d.get("observations") or {}).get("policy") or {})
    terms = []
    for k, v in pol.items():
        if isinstance(v, dict) and "func" in v:
            pr = {a: b for a, b in (v.get("params") or {}).items() if not isinstance(b, (dict, list))}
            terms.append(dict(name=k, func=_short(v.get("func")), hist=v.get("history_length"), params=pr))
    out["obs_policy"] = terms
    tf = {t["name"]: t for t in terms}
    out["obs_filter"] = dict(actions=(tf.get("actions") or {}).get("func"), joint_vel=(tf.get("joint_vel") or {}).get("func"),
                             joint_vel_beta=((tf.get("joint_vel") or {}).get("params") or {}).get("beta"),
                             joint_vel_ema=(tf.get("joint_vel_ema") or {}).get("func"), feet_contact=(tf.get("feet_contact") or {}).get("func"),
                             gait_clock_freq=((tf.get("gait_clock") or {}).get("params") or {}).get("freq"))
    cmd = ((d.get("commands") or {}).get("base_velocity") or {})
    rg = cmd.get("ranges") or {}
    out["command"] = dict(cls=_short(cmd.get("class_type")), resample_s=cmd.get("resampling_time_range"), heading_command=cmd.get("heading_command"),
                          vx=rg.get("lin_vel_x"), vy=rg.get("lin_vel_y"), wz=rg.get("ang_vel_z"))
    ev = d.get("events") or {}
    out["events"] = [k for k, v in ev.items() if v is not None]
    rw = d.get("rewards") or {}
    live = sorted((k, v.get("weight")) for k, v in rw.items() if isinstance(v, dict) and v.get("weight") not in (0, 0.0, None))
    out["n_rewards"] = len(live)
    out["rewards_fp"] = hashlib.md5(json.dumps(live, sort_keys=True).encode()).hexdigest()[:8]
    out["terminations"] = [k for k, v in (d.get("terminations") or {}).items() if v is not None]
    out["curriculum"] = [k for k, v in (d.get("curriculum") or {}).items() if v is not None]
    # 배포 계약 지문: 러너가 맞춰야 하는 것만(관측 항 순서·함수·이력·인자, 액션 스케일·슬루·오프셋, 기본자세, 제어 주기). DR·잡음·보상은 뺌.
    sig = dict(control=out["control"], pose=[round(float(jp[j]), 4) for j in sorted(jp)] if jp and all(isinstance(x, (int, float)) for x in jp.values()) else str(jp),
               action=[out["action"]["cls"], out["action"]["scale"], out["action"]["use_default_offset"], out["action"]["slew_deg_per_tick"]],
               obs=[[t["name"], t["func"], t["hist"], sorted(t["params"].items())] for t in terms])
    out["contract_fp"] = hashlib.md5(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return out


def ckpt_dims(path):
    """체크포인트의 actor/critic 입력 차원·action 차원(CPU). 실패하면 None."""
    try:
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=False)
        it = sd.get("iter")
        sd = sd.get("model_state_dict", sd)
        a = sd.get("actor.0.weight")
        c = sd.get("critic.0.weight")
        s = sd.get("log_std") if "log_std" in sd else sd.get("std")
        return dict(obs=int(a.shape[1]) if a is not None else None, critic_obs=int(c.shape[1]) if c is not None else None,
                    act=int(s.shape[0]) if s is not None else None, iter=int(it) if it is not None else None)
    except Exception as e:  # noqa: BLE001
        return dict(error=repr(e)[:100])


class SigCache:
    """체크포인트 actor 가중치 지문(8 개 텐서 바이트의 md5) — (경로, 크기, mtime_ns) 가 같으면 이전 JSON 값을 재사용."""

    def __init__(self, prev):
        self.m = dict((prev or {}).get("sig_cache", {}))
        self.hit = self.miss = 0

    def ckpt(self, path):
        st = os.stat(path)
        e = self.m.get(rel(path))
        if e and e[0] == st.st_size and e[1] == st.st_mtime_ns:
            self.hit += 1
            return e[2]
        self.miss += 1
        import torch
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd)
        h = hashlib.md5()
        for k in sorted(k for k in sd if k.startswith("actor.")):
            h.update(k.encode())
            h.update(sd[k].detach().cpu().numpy().tobytes())
        self.m[rel(path)] = [st.st_size, st.st_mtime_ns, h.hexdigest()]
        return h.hexdigest()


def jit_actor(path):
    """TorchScript policy.pt → (가중치 지문, [(이름, 바이트)…]). 지문 정의는 SigCache.ckpt 와 같다."""
    import torch
    sd = torch.jit.load(path, map_location="cpu").state_dict()
    h = hashlib.md5()
    parts = []
    for k in sorted(k for k in sd if k.startswith("actor.")):
        b = sd[k].detach().cpu().numpy().tobytes()
        h.update(k.encode())
        h.update(b)
        parts.append((k, b))
    return h.hexdigest(), parts


def export_provenance(run, sigc, order_hint=()):
    """exported*/policy.pt 가 이 run 의 어느 체크포인트에서 나왔는지 **가중치로** 확인하고, policy.onnx 안에 같은 가중치 바이트가 있는지 본다.
    (가중치가 같다는 것까지만 본다 — 그래프 구조·연산 동치는 보지 않는다.)"""
    p = os.path.join(REPO, run["path"])
    for e in run["exported"]:
        pt, ox = os.path.join(p, e["dir"], "policy.pt"), os.path.join(p, e["dir"], "policy.onnx")
        pv = dict(checked=False)
        e["provenance"] = pv
        if not os.path.isfile(pt):
            pv["note"] = "policy.pt 없음"
            continue
        try:
            sig, parts = jit_actor(pt)
        except Exception as ex:  # noqa: BLE001
            pv["note"] = "policy.pt 읽기 실패: " + repr(ex)[:80]
            continue
        pv["checked"] = True
        pv["actor_sig"] = sig
        if os.path.isfile(ox):
            ob = open(ox, "rb").read()
            pv["onnx_has_same_weights"] = all(ob.find(b) != -1 for _, b in parts)
        cands = []
        if e["source_ckpt"] is not None:
            cands.append(e["source_ckpt"])
        cands += [c for c in order_hint if c not in cands]
        cands += [c for c in sorted(run["ckpts"], reverse=True) if c not in cands]
        pv["match_ckpt"] = None
        n = 0
        for c in cands:
            fp = os.path.join(p, f"model_{c}.pt")
            if not os.path.isfile(fp) or os.path.islink(fp):
                continue
            n += 1
            try:
                if sigc.ckpt(fp) == sig:
                    pv["match_ckpt"] = c
                    break
            except Exception:  # noqa: BLE001
                continue
        pv["n_compared"] = n
        pv["source_agrees"] = (pv["match_ckpt"] == e["source_ckpt"]) if e["source_ckpt"] is not None else None


# ---------------------------------------------------------------------------------------------------------------------
# 3. 레지스트리(소스만 읽음)
def _module_strs(tree):
    """모듈 최상위의 이름 = 문자열 상수 대입."""
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            out[n.targets[0].id] = n.value.value
    return out


def _last_str(node):
    s = [x.value for x in ast.walk(node) if isinstance(x, ast.Constant) and isinstance(x.value, str)]
    return s[-1] if s else None


def _find_assign(tree, name):
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return n.value
    return None


def parse_registry():
    reg = dict(checkpoints={}, c5_pairs={}, w3_specs={}, hturn2={}, kew3_specs={}, console_choices=[], console_default=None, stand_default=None,
               require_env_yaml=[], jvel_ema_mode={}, errors=[])
    try:
        bt = ast.parse(read_text(os.path.join(REPO, "jetson/scripts/rl_bridge_node.py")))
        v = _find_assign(bt, "CHECKPOINTS")
        for k, val in zip(v.keys, v.values):
            reg["checkpoints"][k.value] = _last_str(val)
        v = _find_assign(bt, "C5_PAIRS")
        for k, val in zip(v.keys, v.values):
            reg["c5_pairs"][k.value] = dict(stand_key=val.elts[0].value, rec_path=_last_str(val.elts[1]))
        reg["stand_default"] = _module_strs(bt).get("STAND_DEFAULT")
        v = _find_assign(bt, "CKPT_REQUIRE_ENV_YAML")
        reg["require_env_yaml"] = sorted(x.value for x in v.elts)
        v = _find_assign(bt, "CKPT_JVEL_EMA_MODE")
        reg["jvel_ema_mode"] = {k.value: x.value for k, x in zip(v.keys, v.values)}
        bstr = _module_strs(bt)
        wt = ast.parse(read_text(os.path.join(REPO, "rl_walking/deploy/w3_deploy.py")))
        wstr = _module_strs(wt)
        v = _find_assign(wt, "W3_SPECS")
        for k, val in zip(v.keys, v.values):
            key = wstr.get(k.id) if isinstance(k, ast.Name) else k.value
            el = [wstr.get(x.id) if isinstance(x, ast.Name) else None for x in val.elts]
            reg["w3_specs"][key] = dict(run=el[0], model=el[1], task=el[2])
        ht = _module_strs(ast.parse(read_text(os.path.join(REPO, "rl_walking/deploy/hturn2.py"))))
        reg["hturn2"] = {ht.get("HTURN2_CKPT_KEY"): dict(run=ht.get("HTURN2_RUN_REL"), model=ht.get("HTURN2_MODEL"))}
        v = _find_assign(bt, "KEW3_SPECS")
        for k, val in zip(v.keys, v.values):
            key = bstr.get(k.id) if isinstance(k, ast.Name) else k.value
            kw = {}
            for a in val.keywords:
                if isinstance(a.value, ast.Constant):
                    kw[a.arg] = a.value.value
                elif isinstance(a.value, ast.Attribute):
                    kw[a.arg] = wstr.get(a.value.attr)
            reg["kew3_specs"][key] = kw
        ct = ast.parse(read_text(os.path.join(REPO, "jetson/scripts/operator_console.py")))
        v = _find_assign(ct, "CKPT_CHOICES")
        reg["console_choices"] = [x.value for x in v.elts]
        reg["console_default"] = reg["console_choices"][0] if reg["console_choices"] else None
    except Exception as e:  # noqa: BLE001
        reg["errors"].append(repr(e)[:200])
    return reg


def run_of_path(p):
    """'logs/rsl_rl/<exp>/<run>[/exported]' → run 디렉터리 이름."""
    if not p:
        return None
    q = p.rstrip("/")
    if os.path.basename(q).startswith("exported"):
        q = os.path.dirname(q)
    return os.path.basename(q)


# ---------------------------------------------------------------------------------------------------------------------
# 4. 참조 스캔
SKIP_DIRS = {".git", ".claude", "logs", "log_picture", "__pycache__", "outputs", "demo_output", "isaacsim_scene", "urdf", "node_modules", "old"}
TEXT_EXT = {".py", ".sh", ".md", ".txt", ".json", ".md5", ".yaml", ".yml", ".csv", ".tsv", ".log", ".html", ".service"}
HARD_LINE = re.compile(r"--load_run|--switch_to|--ke\b|--w3\b|--resume|torch\.load|jit\.load|load_run\s*=|MODEL_DIR|exported|params/env\.yaml|model_\d+\.pt|:model_|_RUN\s*=|RUN_REL|actor\(|PN=|RA=|G12=|G21=|[A-Z0-9_]+\s*=\s*\(?[\"']20\d\d-|runs\s*=\s*\{")
LOADER_FILE = re.compile(r"torch\.load|jit\.load|isaaclab\.sh|PolicyRunner|_load_backend|onnx|run_regress|get_checkpoint_path|md5sum|scp |os\.path\.isfile|os\.path\.isdir")


def kind_of(r):
    b = os.path.basename(r)
    ext = os.path.splitext(r)[1]
    if r in ("jetson/scripts/rl_bridge_node.py", "jetson/scripts/operator_console.py") or (r.startswith("rl_walking/deploy/") and ext == ".py" and not b.startswith("selftest")):
        return "registry"
    if b.startswith("selftest") and ext == ".py":
        return "selftest"
    if r.startswith("tests/") and ext == ".py":
        return "test"
    if (r.startswith("deploy/") and ext == ".md5") or r.startswith("models/") or r.startswith("deploy/cand/"):
        return "manifest"
    if ext == ".md" or (ext == ".txt" and "/" not in r):
        return "doc"
    if ext in (".sh", ".py"):
        return "script"
    return "result"


def _doc_lines_py(text):
    """.py 의 docstring·주석 줄 번호(여기 나온 run 이름은 사용 예 — 실행 의존이 아님)."""
    soft = set()
    try:
        tree = ast.parse(text)
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
                f = n.body[0]
                if isinstance(f, ast.Expr) and isinstance(f.value, ast.Constant) and isinstance(f.value.value, str):
                    soft.update(range(f.lineno, (f.end_lineno or f.lineno) + 1))
    except SyntaxError:
        pass
    for i, l in enumerate(text.split("\n"), 1):
        if l.lstrip().startswith("#"):
            soft.add(i)
    return soft


def scan_refs(run_names):
    ts_map = collections.defaultdict(list)
    for b in run_names:
        ts_map[b[:19]].append(b)
    refs = collections.defaultdict(list)
    dangling = collections.defaultdict(list)
    nfiles = 0
    for dp, dn, fn in os.walk(REPO):
        dn[:] = [x for x in dn if x not in SKIP_DIRS]
        for f in fn:
            ext = os.path.splitext(f)[1]
            if ext not in TEXT_EXT:
                continue
            p = os.path.join(dp, f)
            r = rel(p)
            if r.startswith("reports/night_1007/policy_inventory") or r == "reports/night_1007/POLICY_INVENTORY.md":
                continue                                  # 자기 산출물은 참조로 세지 않는다
            try:
                if os.path.getsize(p) > 8 * 2**20:
                    continue
            except OSError:
                continue
            txt = read_text(p)
            if "202" not in txt:
                continue
            nfiles += 1
            k = kind_of(r)
            soft_lines = set()
            if ext == ".py":
                soft_lines = _doc_lines_py(txt)
            loader = bool(LOADER_FILE.search(txt))
            for i, line in enumerate(txt.split("\n"), 1):
                if "202" not in line:
                    continue
                for m in TS_RE.finditer(line):
                    ts, full = m.group(1), m.group(1) + m.group(2)
                    cands = ts_map.get(ts)
                    cks = [int(x) for x in re.findall(r"model_(\d+)", line)]
                    if not cands:
                        if k != "result":
                            dangling[full if len(full) > 19 else ts].append(dict(kind=k, file=r, line=i, ckpts=cks[:4]))
                        continue
                    hit = [b for b in cands if full.startswith(b)] or ([b for b in cands if b.startswith(full)] if len(full) > 19 else [])
                    if not hit:
                        hit = cands if len(cands) == 1 else []
                    targets = [max(hit, key=len)] if hit else cands       # 타임스탬프만 같고 이름이 갈리면 둘 다(모호)
                    comment = (i in soft_lines) or (ext == ".sh" and line.lstrip().startswith("#"))
                    hard = (k in ("registry", "selftest", "test", "script")) and not comment and bool(HARD_LINE.search(line)) and (loader or k != "script")
                    tail = [int(x) for x in re.findall(r"model_(\d+)", line[m.end():m.end() + 200])]
                    for b in targets:
                        refs[b].append(dict(kind=k, file=r, line=i, hard=hard, comment=comment, ckpts=(tail[:1] if tail else cks[:1]), ambiguous=len(targets) > 1,
                                            current=r.startswith(CURRENT_DIRS), text=line.strip()[:160]))
    return refs, dangling, nfiles


EV_PAIR = re.compile(r"(20\d\d-\d\d-\d\d_\d\d-\d\d-\d\d[A-Za-z0-9_]*)(?:\s+--checkpoint\s+|[:/])model_(\d+)\.pt")
EV_TASK = re.compile(r"--task\s+(\S+)")


def scan_eval_outputs(run_names):
    """log_picture/**/*.{txt,json,md} 머리말(6 KB)에서 (run, 체크포인트) 사용 횟수."""
    lp = os.path.join(REPO, "log_picture")
    pairs = collections.Counter()
    last, example, tasks, unknown = {}, {}, collections.defaultdict(collections.Counter), collections.Counter()
    n = 0
    for dp, dn, fn in os.walk(lp):
        for f in fn:
            if os.path.splitext(f)[1] not in (".txt", ".json", ".md"):
                continue
            p = os.path.join(dp, f)
            head = read_text(p, 6000)
            if "202" not in head:
                continue
            n += 1
            seen = set((m.group(1), int(m.group(2))) for m in EV_PAIR.finditer(head))
            if not seen:
                continue
            try:
                mt = os.path.getmtime(p)
            except OSError:
                mt = 0
            tk = EV_TASK.search(head)
            invalid = "_invalid" in rel(p) or "무효" in rel(p)
            for b, c in seen:
                if b in run_names:
                    if invalid:
                        continue
                    pairs[(b, c)] += 1
                    last[(b, c)] = max(last.get((b, c), 0), mt)
                    example.setdefault((b, c), rel(p))
                    if tk:
                        tasks[b][tk.group(1)] += 1
                else:
                    unknown[b] += 1
    return pairs, last, example, tasks, unknown, n


# ---------------------------------------------------------------------------------------------------------------------
def find_needle(doc, needle, cache={}):
    """문서에서 문구를 찾아 (줄 번호, 그 줄) — 없으면 (None, None)."""
    if doc not in cache:
        cache[doc] = read_text(os.path.join(REPO, doc)).split("\n")
    for i, l in enumerate(cache[doc], 1):
        if needle in l:
            return i, l.strip()[:200]
    return None, None


STALE_RE = re.compile(r"KILL|불합격|통과 후보 없음|미채택|기각|탈락|\*\*판정")


def stale_hints(tag, docs):
    """'평가 중'으로 적어 둔 run 에 대해, 문서에 판정 문구가 새로 생겼는지 찾는다(분류표가 낡았다는 신호). tag = 문서가 쓰는 이름(예: W3S3f)."""
    hits = []
    if not tag:
        return hits
    for doc in docs:
        lines = read_text(os.path.join(REPO, doc)).split("\n")
        own = tag.lower().replace("w3", "") in os.path.basename(doc).lower()       # 그 run 전용 문서(S3E_DESIGN.md 같은)
        for i, l in enumerate(lines, 1):
            if "사전" in l and "결과" not in l:
                continue
            head = l.startswith("#")
            if tag in l and ((head and re.search(r"결과|판정", l)) or STALE_RE.search(l)):
                hits.append(dict(doc=doc, line=i, text=l.strip()[:140]))
            elif own and (l.startswith("**판정") or (head and "판정" in l)):
                hits.append(dict(doc=doc, line=i, text=l.strip()[:140]))
    return hits[:4]


def stage_names():
    """rl_walking/__init__.py 에 등록된 HangStand 단계 이름(태스크 id = Biped12-HangStand-<단계>-v0)."""
    txt = read_text(os.path.join(REPO, "rl_walking/__init__.py"))
    names = set()
    for m in re.finditer(r"for\s+_stage(?:,\s*_agent)?\s+in\s+\((.*?)\):", txt, re.S):
        for s in re.findall(r"\"([A-Za-z0-9]+)\"", m.group(1)):
            if not s.startswith("Biped12"):
                names.add(s)
    ids = set(re.findall(r"id=\"(Biped12-[A-Za-z0-9-]+)\"", txt))
    return names, ids


def git_info():
    def g(*a):
        try:
            return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    return dict(branch=g("rev-parse", "--abbrev-ref", "HEAD"), head=g("rev-parse", "--short", "HEAD"), head_date=g("log", "-1", "--format=%ci"))


# ---------------------------------------------------------------------------------------------------------------------
# 5. 조립·분류
FRESH_S = 180.0      # 이 시간 안에 바뀐 파일은 쓰는 중일 수 있어 md5·차원 읽기를 건너뛴다


def build(md5_mode="key", use_torch=True, prev=None, verbose=True):
    t0 = time.time()
    now = time.time()
    md5c = Md5Cache(prev)
    sigc = SigCache(prev)

    def say(*a):
        if verbose:
            print(f"[{time.time() - t0:6.1f}s]", *a, file=sys.stderr, flush=True)

    run_dirs, aliases, strays = scan_runs()
    say(f"run 디렉터리 {len(run_dirs)} · 심볼릭 링크 {len(aliases)}")
    runs = []
    for p in run_dirs:
        r = run_facts(p, md5c, md5_all=False)
        ap, ep = os.path.join(p, "params", "agent.yaml"), os.path.join(p, "params", "env.yaml")
        r["agent"] = summarize_agent(ap) if os.path.isfile(ap) else None
        r["env"] = summarize_env(ep) if os.path.isfile(ep) else None
        runs.append(r)
    by_name = {r["name"]: r for r in runs}
    dup = [n for n, c in collections.Counter(r["name"] for r in runs).items() if c > 1]
    say(f"params 요약 끝 · 이름 중복 {dup}")

    # 별칭(디렉터리 링크) → run
    by_path = {r["path"]: r for r in runs}
    for r in runs:
        r["aliases"] = []
    exp_aliases = []
    for a in aliases:
        if a["link"].count("/") >= 4 and MODEL_RE.match(os.path.basename(a["link"])):
            continue                                       # run 안의 model_N.pt 링크는 run_facts 가 기록
        t = by_path.get(a["points_to"])
        if t is not None:
            t["aliases"].append(a["link"])
        else:
            exp_aliases.append(a)

    # 계보
    for r in runs:
        r["parent"], r["parent_ckpt"], r["children"] = None, None, []
    for r in runs:
        ag = r.get("agent") or {}
        if ag.get("resume") and ag.get("load_run") in by_name:
            r["parent"] = ag["load_run"]
            m = MODEL_RE.match(str(ag.get("load_checkpoint") or ""))
            r["parent_ckpt"] = int(m.group(1)) if m else None
            by_name[ag["load_run"]]["children"].append(dict(run=r["name"], ckpt=r["parent_ckpt"]))
        elif ag.get("resume") and ag.get("load_run") and ag.get("load_run") != ".*":
            r["parent_missing"] = f"{ag.get('load_run')}/{ag.get('load_checkpoint')}"
    # g13init 처럼 params 없는 시작 체크포인트 디렉터리: 같은 반복 번호 하나만 든 run 을 부모로 지정한 자식이 있으면 계보로 잡힌다(위에서 처리됨)

    # 레지스트리
    reg = parse_registry()
    for r in runs:
        r["registry"] = []
    missing_keys = []
    for key, path in reg["checkpoints"].items():
        rn = run_of_path(path)
        ex = os.path.isdir(os.path.join(REPO, path))
        if rn in by_name:
            by_name[rn]["registry"].append(dict(table="CHECKPOINTS", key=key, path=path, exported_exists=ex))
        else:
            missing_keys.append(dict(table="CHECKPOINTS", key=key, path=path))
    for key, v in reg["c5_pairs"].items():
        rn = run_of_path(v["rec_path"])
        if rn in by_name:
            by_name[rn]["registry"].append(dict(table="C5_PAIRS(회복)", key=key, path=v["rec_path"], exported_exists=os.path.isdir(os.path.join(REPO, v["rec_path"]))))
        else:
            missing_keys.append(dict(table="C5_PAIRS", key=key, path=v["rec_path"]))
    for key, v in list(reg["w3_specs"].items()) + list(reg["hturn2"].items()):
        rn = run_of_path(v.get("run"))
        tab = "W3_SPECS(실험 플래그)" if key in reg["w3_specs"] else "HTURN2(실험 플래그)"
        if rn in by_name:
            by_name[rn]["registry"].append(dict(table=tab, key=key, path=v["run"] + "/exported", model=v.get("model"),
                                                exported_exists=os.path.isdir(os.path.join(REPO, v["run"], "exported"))))
        else:
            missing_keys.append(dict(table=tab, key=key, path=v.get("run")))

    def key_runs(key):
        """키 → 그 키가 읽는 run 이름들."""
        out = []
        if key in reg["kew3_specs"]:
            sp = reg["kew3_specs"][key]
            return key_runs(sp.get("w3")) + key_runs(sp.get("ke"))
        if key in reg["c5_pairs"]:
            out.append(run_of_path(reg["c5_pairs"][key]["rec_path"]))
            out += key_runs(reg["c5_pairs"][key]["stand_key"]) if reg["c5_pairs"][key]["stand_key"] != key else []
        if key in reg["checkpoints"]:
            out.append(run_of_path(reg["checkpoints"][key]))
        if key in reg["w3_specs"]:
            out.append(run_of_path(reg["w3_specs"][key]["run"]))
        return [x for x in dict.fromkeys(out) if x]

    for r in runs:
        r["console_choices"] = []
    for ch in reg["console_choices"]:
        for rn in key_runs(ch):
            if rn in by_name:
                by_name[rn]["console_choices"].append(ch)

    # 배포 후보 = KEW3_SPECS(합성 구조)가 가리키는 세 run
    deploy = collections.OrderedDict()
    deploy_note = None
    for ckey, sp in reg["kew3_specs"].items():
        w3k, kek = sp.get("w3"), sp.get("ke")
        if w3k in reg["w3_specs"] and kek in reg["c5_pairs"]:
            deploy["서기(STAND)"] = dict(run=run_of_path(reg["checkpoints"].get(reg["c5_pairs"][kek]["stand_key"])), key=reg["c5_pairs"][kek]["stand_key"], via=f"KEW3_SPECS[{ckey}].ke={kek} → C5_PAIRS[..][0]")
            deploy["회복(RECOVERY)"] = dict(run=run_of_path(reg["c5_pairs"][kek]["rec_path"]), key=f"(C5_PAIRS[{kek}] 의 회복 경로 — 단독 키 없음)", via=f"KEW3_SPECS[{ckey}].ke={kek} → C5_PAIRS[..][1]")
            deploy["걷기(WALK)"] = dict(run=run_of_path(reg["w3_specs"][w3k]["run"]), key=w3k, via=f"KEW3_SPECS[{ckey}].w3", model=reg["w3_specs"][w3k]["model"])
            deploy_note = f"합성 키 {ckey} (handoff={sp.get('handoff')}, settle_quiet_s={sp.get('settle_quiet_s')})"
    if len(deploy) != 3:
        deploy_note = "⚠ 레지스트리에서 합성 구조를 해석하지 못해 고정 이름을 썼다"
        deploy = collections.OrderedDict([("서기(STAND)", dict(run="2026-10-02_21-23-08_g21a", key="stand_crouch_g21a", via="고정")),
                                          ("회복(RECOVERY)", dict(run="2026-10-04_10-08-41_g24ra_seed1_from_w2e20294", key="-", via="고정")),
                                          ("걷기(WALK)", dict(run="2026-10-07_00-11-12_w3S3d_from_s2d", key="stand_w3s3d_15800", via="고정"))])
    deploy_runs = {v["run"]: role for role, v in deploy.items()}

    # 참조
    names = set(by_name)
    refs, dangling, nfiles = scan_refs(names)
    say(f"참조 스캔 {nfiles} 파일 · 참조된 run {len(refs)} · 로컬에 없는 이름 {len(dangling)}")
    pairs, last, example, eval_tasks, ev_unknown, n_ev = scan_eval_outputs(names)
    say(f"평가 산출물 머리말 {n_ev} 파일 · (run, 체크포인트) 쌍 {len(pairs)}")
    ev_by_run = collections.defaultdict(dict)
    for (b, c), n in pairs.items():
        ev_by_run[b][c] = dict(n=n, last=time.strftime("%Y-%m-%d %H:%M", time.localtime(last[(b, c)])), example=example[(b, c)])

    stages, task_ids = stage_names()
    models_meta = {}
    for dp in ("models", "models/legacy"):
        d = os.path.join(REPO, dp)
        for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            mp = os.path.join(d, n, "meta.json")
            if os.path.isfile(mp):
                try:
                    models_meta[n] = dict(json.load(open(mp, encoding="utf-8")), dir=f"{dp}/{n}")
                except Exception:  # noqa: BLE001
                    pass
    meta_by_run = collections.defaultdict(list)
    for n, m in models_meta.items():
        meta_by_run[m.get("run")].append(m)

    # ── run 별 정리
    for r in runs:
        nm = r["name"]
        rr = refs.get(nm, [])
        kinds = collections.Counter(x["kind"] for x in rr)
        r["ref_counts"] = dict(kinds)
        r["hard_refs"] = [x for x in rr if x["hard"]]
        r["doc_files"] = sorted(set(x["file"] for x in rr if x["kind"] == "doc"))
        r["refs_sample"] = [dict(kind=x["kind"], file=x["file"], line=x["line"], hard=x["hard"], ckpts=x["ckpts"]) for x in rr if x["kind"] != "result"][:40]
        r["evaluated"] = {str(c): v for c, v in sorted(ev_by_run.get(nm, {}).items())}
        r["n_eval_files"] = sum(v["n"] for v in ev_by_run.get(nm, {}).values())
        r["eval_tasks"] = dict(eval_tasks.get(nm, {}))
        r["age_h"] = round((now - r["latest_mtime"]) / 3600.0, 2) if r["latest_mtime"] else None
        # 태스크
        if nm in TASK_VERIFIED:
            r["task"], r["task_src"] = TASK_VERIFIED[nm][0], "확인: " + TASK_VERIFIED[nm][1]
        elif meta_by_run.get(nm):
            r["task"], r["task_src"] = meta_by_run[nm][0].get("task"), f"확인: {meta_by_run[nm][0]['dir']}/meta.json"
        else:
            lab = re.split(r"_(?:seed\d+|from|c[RS]\d|s\d(?:_|$))", r["label"])[0]
            hit = [s for s in stages if s.lower() == lab.lower()]
            r["task"], r["task_src"] = (f"Biped12-HangStand-{hit[0]}-v0", "추정: run 이름 = 등록 단계 이름") if hit else (None, "미상(env.yaml 에 태스크 id 가 없다)")
        # 분류표
        cur = next((c for c in CURATED if c["pat"].fullmatch(nm)), None)
        ev = []
        ok = True
        if cur:
            for doc, nd in cur["ev"]:
                ln, txt = find_needle(doc, nd)
                ev.append(dict(doc=doc, line=ln, needle=nd, found=ln is not None))
                ok = ok and ln is not None
        r["curated"] = dict(cls=cur["cls"], role=cur["role"], why=cur["why"], evidence=ev, evidence_ok=ok, keys={str(k): v for k, v in cur["keys"].items()}) if cur else None
        if cur and cur["cls"] == C_EVAL:
            r["curated"]["stale_hints"] = stale_hints(cur.get("note"), sorted(set(d for d, _ in cur["ev"])))

    # ── 1차 분류
    for r in runs:
        nm, cur = r["name"], r["curated"]
        reasons = []
        cls = None
        sub = None
        void_name = bool(NAME_VOID.search(r["label"])) or "/_void" in ("/" + r["path"])
        cur_hard = [x for x in r["hard_refs"] if x["current"] and x["kind"] in ("registry", "selftest", "test", "script")]
        old_hard = [x for x in r["hard_refs"] if not x["current"] and x["kind"] == "script"]
        if nm in deploy_runs:
            cls = C_DEPLOY
            reasons.append(f"합성 구조의 {deploy_runs[nm]} 정책 — {deploy[deploy_runs[nm]]['via']}")
        elif cur and cur["cls"] == C_EVAL:
            cls = C_EVAL
            reasons.append(cur["why"])
            for h in cur.get("stale_hints") or []:
                reasons.append(f"⚠ 문서에 판정 문구가 생긴 것으로 보인다(`{h['doc']}`:{h['line']} '{h['text'][:60]}') — 분류표(CURATED)를 고쳐야 한다")
        elif cur is None and not void_name and r["age_h"] is not None and r["age_h"] < ACTIVE_HOURS:
            cls = C_EVAL
            reasons.append(f"분류표에 없고 최근 {ACTIVE_HOURS:.0f} 시간 안에 생김(자동) — 학습·평가 중일 수 있어 읽기만 했다")
        elif r["registry"] or cur_hard:
            cls = C_BASE
            if r["registry"]:
                ks = ", ".join(sorted(set(x["key"] for x in r["registry"])))
                reasons.append(f"브리지 레지스트리 키 {ks}" + ("" if all(x["exported_exists"] for x in r["registry"]) else " (exported 없음 — 키만 등록)"))
            if r["console_choices"]:
                reasons.append("콘솔 선택지 " + ", ".join(r["console_choices"]))
            fs = sorted(set(x["file"] for x in cur_hard if not (r["registry"] and x["file"] == "jetson/scripts/rl_bridge_node.py")))
            if fs:
                reasons.append("현행 코드가 직접 읽음: " + ", ".join(fs[:6]) + (f" 외 {len(fs) - 6}" if len(fs) > 6 else ""))
        elif cur and cur["cls"] == C_BASE and cur["evidence_ok"]:
            cls = C_BASE
            reasons.append(f"{cur['role']} — {cur['why']}")
        elif cur and cur["cls"] == C_REC:
            cls = C_REC
            reasons.append(f"{cur['role']} — {cur['why']}")
        elif old_hard:
            cls = C_REC
            reasons.append("과거 분석 스크립트가 체크포인트를 직접 읽음: " + ", ".join(sorted(set(x["file"] for x in old_hard))[:4]))
        elif cur and cur["cls"] == C_DISCARD:
            if cur["evidence"] and cur["evidence_ok"]:
                cls, sub = C_DISCARD, "문서 근거"
            elif not cur["evidence"]:
                cls, sub = C_DISCARD, "이름·설정 근거"
            else:
                cls = C_UNK
                reasons.append("⚠ 분류표의 근거 문구를 문서에서 찾지 못함 — 문서가 바뀌었는지 확인 필요")
            reasons.append(f"{cur['role']} — {cur['why']}")
        elif cur and cur["cls"] == C_UNK:
            cls = C_UNK
            reasons.append(f"{cur['role']} — {cur['why']}")
        elif void_name:
            cls = C_DISCARD
            it = (r.get("agent") or {}).get("max_iterations")
            is_bench = "bench" in r["label"].lower()
            ln, _ = find_needle(*BENCH_EV)
            if is_bench and ln:
                sub = "문서 근거"
                r["curated"] = dict(cls=C_DISCARD, role="무효", why="1003 학습 속도 벤치(단독 대 동시 실행)", evidence=[dict(doc=BENCH_EV[0], line=ln, needle=BENCH_EV[1], found=True)], evidence_ok=True, keys={})
                reasons.append("무효 — 1003 학습 속도 벤치(40/30 반복), 결론(4096 env 순차 학습)이 문서에 있음")
            else:
                sub = "이름·설정 근거"
                reasons.append(f"무효 — run 이름이 스모크·점검·디버그(agent.yaml max_iterations {it}, 체크포인트 {r['n_ckpt']} 개, 평가 산출물 {r['n_eval_files']} 건). 문서 기록은 없다")
        else:
            cls = C_UNK
            reasons.append("근거를 찾지 못함(문서·코드·스크립트에 이 run 의 판정이나 사용처가 없다)")
        # 분류표가 '기각'인데 더 강한 자동 규칙으로 남는 경우 — 문서 판정을 메모로 남긴다
        if cur and cls in (C_DEPLOY, C_BASE) and cur["cls"] in (C_DISCARD, C_REC) and not any(cur["why"] in x for x in reasons):
            reasons.append(f"문서 기록: {cur['role']} — {cur['why']}")
        r["cls"], r["sub"], r["reasons"] = cls, sub, reasons
        r["cls_base"] = cls

    # ── 계보 승격: 남기는 run 의 조상은 (폐기 후보·불명이어도) 기록 보존으로
    changed = True
    while changed:
        changed = False
        for r in runs:
            if r["cls"] in KEEP and r["parent"]:
                p = by_name[r["parent"]]
                if p["cls"] not in KEEP:
                    was = p["cls"]
                    p["cls"], p["sub"] = C_REC, None
                    p["reasons"].insert(0, f"계보 — {r['name']} 의 웜스타트 원본(model_{r['parent_ckpt']}.pt) — 문서 판정만 보면 {was}")
                    p["lineage_upgrade"] = True
                    changed = True

    # ── 핵심 체크포인트
    for r in runs:
        nm = r["name"]
        key = collections.defaultdict(list)
        link = {}
        for l in r["ckpt_links"]:
            a, b = MODEL_RE.match(l["name"]), MODEL_RE.match(os.path.basename(l["points_to"]))
            if a and b:
                link[int(a.group(1))] = int(b.group(1))

        def add(c, why):
            if c is None:
                return
            c = int(c)
            if c in link:
                why = f"{why} (model_{c}.pt 는 model_{link[c]}.pt 의 링크)"
                c = link[c]
            if why not in key[c]:
                key[c].append(why)
        for e in r["exported"]:
            add(e["source_ckpt"], f"{e['dir']}/SOURCE_CHECKPOINT")
        for x in r["registry"]:
            m = MODEL_RE.match(str(x.get("model") or ""))
            if m:
                add(int(m.group(1)), f"{x['table']} {x['key']}")
        for m in meta_by_run.get(nm, []):
            mm = MODEL_RE.match(str(m.get("checkpoint") or ""))
            if mm:
                add(int(mm.group(1)), f"{m['dir']}/meta.json")
        if r["curated"]:
            for c, why in r["curated"]["keys"].items():
                add(int(c), "분류표: " + why)
        for ch in r["children"]:
            if by_name[ch["run"]]["cls"] in KEEP:
                add(ch["ckpt"], f"웜스타트 원본 ← {ch['run']}")
        for x in r["hard_refs"]:
            for c in x["ckpts"]:
                add(c, f"{x['file']}:{x['line']}")
        for x in refs.get(nm, []):
            if x["kind"] == "manifest":
                for c in x["ckpts"]:
                    add(c, f"{x['file']}:{x['line']}")
        if r["cls"] in KEEP:
            for c, v in r["evaluated"].items():
                add(int(c), f"평가 산출물 {v['n']} 건")
        for a_, b_ in link.items():
            if r["cls"] in KEEP and b_ not in key:
                key[b_].append(f"model_{a_}.pt 링크의 대상")
        if r["cls"] == C_EVAL:
            for c in r["ckpts"]:
                add(c, "평가 중 — 전부 남김")
        elif r["cls"] in KEEP and r["age_h"] is not None and r["age_h"] < ACTIVE_HOURS:
            for c in r["ckpts"]:
                add(c, f"최근 {ACTIVE_HOURS:.0f} 시간 안에 바뀐 run — 솎기에서 뺌")
        have = set(r["ckpts"])
        r["key_ckpts"] = {str(c): v for c, v in sorted(key.items()) if c in have}
        r["key_missing"] = {} if r["cls"] == C_EVAL else {str(c): v for c, v in sorted(key.items()) if c not in have}   # 참조는 있는데 파일이 없는 체크포인트(학습 중인 run 은 제외)
        keep = set(int(c) for c in r["key_ckpts"]) if r["cls"] in KEEP else set()
        r["thin_ckpts"] = [c for c in r["ckpts"] if c not in keep] if r["cls"] in KEEP else []
        r["thin_bytes"] = sum(int(r["ckpt_sizes"][str(c)]) for c in r["thin_ckpts"])

    # ── md5(핵심 또는 전부) · 차원
    say("md5·차원 읽기 시작")
    for r in runs:
        p = os.path.join(REPO, r["path"])
        todo = r["ckpts"] if md5_mode == "all" else [int(c) for c in r["key_ckpts"]]
        for c in todo:
            fp = os.path.join(p, f"model_{c}.pt")
            try:
                if now - os.path.getmtime(fp) < FRESH_S:
                    r.setdefault("md5_skipped_fresh", []).append(c)
                    continue
                r["ckpt_md5"][str(c)] = md5c.get(fp)
            except OSError:
                pass
        r["dims"] = None
        if use_torch and r["ckpts"]:
            pick = int(next(iter(r["key_ckpts"]), r["ckpts"][-1])) if r["key_ckpts"] else r["ckpts"][-1]
            fp = os.path.join(p, f"model_{pick}.pt")
            if now - os.path.getmtime(fp) >= FRESH_S:
                r["dims"] = dict(ckpt_dims(fp), ckpt=pick)
    say(f"md5 캐시 재사용 {md5c.hit} · 새로 계산 {md5c.miss}")
    if use_torch:
        for r in runs:
            if r["exported"] and (r["age_h"] is None or r["age_h"] * 3600 > FRESH_S):
                hint = sorted((int(c) for c in r["evaluated"]), key=lambda c: -r["evaluated"][str(c)]["n"]) + [int(c) for c in r["key_ckpts"]]
                try:
                    export_provenance(r, sigc, hint)
                except Exception as ex:  # noqa: BLE001
                    r["provenance_error"] = repr(ex)[:120]
        say(f"익스포트 가중치 대조 끝 · 체크포인트 지문 재사용 {sigc.hit} · 새로 읽음 {sigc.miss}")

    # ── 로컬에 없는 run 을 가리키는 참조(이미 끊어진 것)
    dang = []
    for nm, v in sorted(dangling.items()):
        kc = collections.Counter(x["kind"] for x in v)
        code = [x for x in v if x["kind"] in ("registry", "selftest", "test", "script")]
        dang.append(dict(name=nm, counts=dict(kc), code=[f"{x['file']}:{x['line']}" for x in code][:8], eval_outputs=ev_unknown.get(nm, 0)))

    out = dict(runs=runs, aliases=aliases, exp_aliases=exp_aliases, strays=strays, registry=reg, missing_keys=missing_keys, deploy=deploy, deploy_note=deploy_note,
               dangling=dang, models_meta=models_meta, n_ref_files=nfiles, n_eval_files=n_ev, md5c=md5c, sigc=sigc, use_torch=use_torch, stages=sorted(stages), by_name=by_name, refs=refs)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 6. logs/rsl_rl 밖의 정책 사본
def weights_vs_ckpt(inv, policy_pt, run_name, ckpt_file):
    """사본 policy.pt 의 가중치가 <run>/<ckpt_file> 과 같은가 — True/False, 확인할 수 없으면 None."""
    if not inv.get("use_torch"):
        return None
    r = inv["by_name"].get(run_name)
    if not r or not os.path.isfile(policy_pt) or not ckpt_file:
        return None
    fp = os.path.join(REPO, r["path"], ckpt_file)
    if not os.path.isfile(fp):
        return None
    try:
        return jit_actor(policy_pt)[0] == inv["sigc"].ckpt(fp)
    except Exception:  # noqa: BLE001
        return None


def outside_copies(inv):
    md5c = inv["md5c"]
    exp_md5 = {}
    for r in inv["runs"]:
        for e in r["exported"]:
            for f, v in e["files"].items():
                if f.startswith("policy."):
                    exp_md5.setdefault(v["md5"], []).append(f"{r['path']}/{e['dir']}/{f}")
    out = dict(models=[], deploy_cand=[], worktrees=[])
    for n, m in sorted(inv["models_meta"].items()):
        d = os.path.join(REPO, m["dir"])
        files, ok = {}, True
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if os.path.isfile(fp) and f != "meta.json":
                h = md5c.get(fp)
                want = (m.get("files") or {}).get(f)
                files[f] = dict(size=os.path.getsize(fp), md5=h, meta_md5=want, match_meta=(h == want) if want else None, same_as=exp_md5.get(h, [])[:2])
                ok = ok and (want is None or h == want)
        run = m.get("run")
        wm = weights_vs_ckpt(inv, os.path.join(d, "policy.pt"), run, m.get("checkpoint"))
        ck_md5_now = None
        mm = MODEL_RE.match(str(m.get("checkpoint") or ""))
        if run in inv["by_name"] and mm:
            ck_md5_now = inv["by_name"][run]["ckpt_md5"].get(mm.group(1))
        out["models"].append(dict(name=n, dir=m["dir"], run=run, run_local=run in inv["by_name"], checkpoint=m.get("checkpoint"), checkpoint_md5=m.get("checkpoint_md5"),
                                  checkpoint_md5_now=ck_md5_now, task=m.get("task"), legacy=m.get("legacy"), files=files, files_match_meta=ok, weights_match_ckpt=wm))
    cd = os.path.join(REPO, "deploy", "cand")
    for n in sorted(os.listdir(cd)) if os.path.isdir(cd) else []:
        d = os.path.join(cd, n)
        files = {}
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                h = md5c.get(fp)
                files[f] = dict(size=os.path.getsize(fp), md5=h, same_as=exp_md5.get(h, [])[:2])
        wm, src = None, None
        mm = re.fullmatch(r"([a-z0-9]+)_(\d+)", n)
        if mm:
            for r in inv["runs"]:
                if gen_token(r) == mm.group(1) and int(mm.group(2)) in r["ckpts"]:
                    src = f"{r['path']}/model_{mm.group(2)}.pt"
                    wm = weights_vs_ckpt(inv, os.path.join(d, "policy.pt"), r["name"], f"model_{mm.group(2)}.pt")
        out["deploy_cand"].append(dict(name=n, dir=f"deploy/cand/{n}", files=files, claimed_source=src, weights_match_ckpt=wm))
    wd = os.path.join(REPO, ".claude", "worktrees")
    for n in sorted(os.listdir(wd)) if os.path.isdir(wd) else []:
        lg = os.path.join(wd, n, "logs")
        if os.path.islink(lg) or not os.path.isdir(lg):
            continue
        items = []
        for dp, dn, fn in os.walk(lg):
            for f in fn:
                if MODEL_RE.match(f) or f in ("policy.pt", "policy.onnx"):
                    fp = os.path.join(dp, f)
                    relp = os.path.relpath(fp, os.path.join(wd, n))
                    main = os.path.join(REPO, relp)
                    same = None
                    if os.path.isfile(main):
                        same = os.path.getsize(main) == os.path.getsize(fp) and md5c.get(main) == md5_file(fp)
                    items.append(dict(file=relp, size=os.path.getsize(fp), in_main=os.path.isfile(main), identical=same))
        if items:
            out["worktrees"].append(dict(worktree=f".claude/worktrees/{n}", n=len(items), bytes=sum(x["size"] for x in items),
                                         all_identical_to_main=all(x["identical"] for x in items), only_here=[x["file"] for x in items if not x["in_main"]], items=items[:40]))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 7. 젯슨 요구 파일(코드가 읽는 파일만 — 접속하지 않는다)
def jetson_snapshot(path=None):
    """누군가 받아 둔 젯슨 쪽 md5 목록(`md5sum` 출력, 경로는 저장소 상대)을 읽는다. 이 스크립트는 젯슨에 접속하지 않는다.
    기본: reports/night_1007/jetson/*md5*.txt 가운데 가장 새 파일. 없으면 None."""
    import glob
    cands = [path] if path else sorted(glob.glob(os.path.join(HERE, "jetson", "*md5*.txt")), key=os.path.getmtime)
    cands = [c for c in cands if c and os.path.isfile(c)]
    if not cands:
        return None
    f = cands[-1]
    m = {}
    for line in read_text(f).split("\n"):
        mm = re.match(r"^([0-9a-f]{32})\s+\*?(\S+)$", line.strip())
        if mm:
            m[mm.group(2).lstrip("./")] = mm.group(1)
    return dict(file=rel(f), taken=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f))), md5=m)


def snapshot_compare(inv, snap):
    """스냅숏의 logs/rsl_rl 줄 전부를 지금 PC 파일과 대조."""
    if not snap:
        return None
    rows = []
    for path, h in sorted(snap["md5"].items()):
        if not path.startswith("logs/rsl_rl/"):
            continue
        fp = os.path.join(REPO, path)
        if os.path.isfile(fp):
            got = inv["md5c"].get(fp)
            rows.append(dict(path=path, jetson=h, pc=got, state="같음" if got == h else "다름"))
        else:
            rows.append(dict(path=path, jetson=h, pc=None, state="PC 에 없음"))
    runs = collections.OrderedDict()
    for r in rows:
        runs.setdefault("/".join(r["path"].split("/")[:4]), []).append(r)
    return dict(file=snap["file"], taken=snap["taken"], n=len(rows), counts=dict(collections.Counter(r["state"] for r in rows)), rows=rows,
                by_run={k: sorted(set(x["state"] for x in v)) for k, v in runs.items()})


def jetson_files(inv, snap=None):
    reg, by_name = inv["registry"], inv["by_name"]

    def finfo(relp):
        p = os.path.join(REPO, relp)
        js = None
        if snap:
            js = snap["md5"].get(relp)
        if not os.path.isfile(p):
            return dict(path=relp, exists=False, jetson_md5=js, jetson=("기록 없음" if snap is None else ("있음(PC 에 없어 대조 못 함)" if js else "없음")))
        h = inv["md5c"].get(p)
        return dict(path=relp, exists=True, size=os.path.getsize(p), md5=h, jetson_md5=js,
                    jetson=("기록 없음" if snap is None else ("같음" if js == h else ("**다름**" if js else "없음"))))

    groups = collections.OrderedDict()      # exported 디렉터리 → 요구하는 쪽 목록

    def want(expdir, tier, label, env_needed=True):
        if not expdir:
            return
        g = groups.setdefault(expdir.rstrip("/"), dict(tiers=[], labels=[], env_needed=False))
        if tier not in g["tiers"]:
            g["tiers"].append(tier)
        if label not in g["labels"]:
            g["labels"].append(label)
        g["env_needed"] = g["env_needed"] or env_needed

    for role, v in inv["deploy"].items():
        r = by_name.get(v["run"])
        if r:
            want(r["path"] + "/exported", "합성 구조", role, env_needed=(role != "회복(RECOVERY)"))
    sd = reg.get("stand_default")
    want(reg["checkpoints"].get(sd), "selftest 필수", f"STAND_DEFAULT={sd}: --selftest '정책 500틱'·'학습 계약'이 파일을 필수로 요구")
    for k in ["stand_crouch_g12", "stand_crouch_g21a"] + sorted(reg.get("jvel_ema_mode", {})):
        want(reg["checkpoints"].get(k), "selftest 필수", f"{k}: '학습 계약' selftest 가 env.yaml 을 필수로 요구")
    for ch in reg["console_choices"]:
        want(reg["checkpoints"].get(ch), "콘솔 선택지", f"콘솔 {ch}" + ("(기본값)" if ch == reg.get("console_default") else ""))
        if ch in reg["c5_pairs"]:
            want(reg["c5_pairs"][ch]["rec_path"], "콘솔 선택지", f"콘솔 {ch} 의 회복 정책", env_needed=False)
    out = []
    for expdir, g in groups.items():
        rd = os.path.dirname(expdir)
        files = [dict(finfo(f"{expdir}/policy.onnx"), need="필수 — 젯슨 실행 파일(onnxruntime 경로)"),
                 dict(finfo(f"{expdir}/policy.pt"), need="선택 — PC torch 경로. 젯슨 selftest 일부 분기는 존재 여부만 본다"),
                 dict(finfo(f"{rd}/params/env.yaml"), need=("필수 — 관측 필터·기본자세·(W3) 명령 계약을 여기서 읽는다" if g["env_needed"] else "선택 — 코드가 읽지 않는다(회복 정책은 서기 정책 계약을 쓴다). 배포 스크립트는 기록용으로 함께 보냄")),
                 dict(finfo(f"{expdir}/SOURCE_CHECKPOINT"), need="선택 — 코드가 읽지 않는다(출처 기록)")]
        out.append(dict(exported=expdir, run=os.path.basename(rd), tiers=g["tiers"], labels=g["labels"], files=files))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 8. 매니페스트 대조 · 계약 대조 · 이름 제안
def manifest_check(inv):
    """deploy/*.md5 의 logs/rsl_rl 줄을 지금 파일과 대조(코드 줄은 개수만 — 매니페스트는 작성 시점의 코드 스냅숏이다)."""
    out = []
    d = os.path.join(REPO, "deploy")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if not f.endswith(".md5"):
            continue
        pol, code_same, code_diff, code_missing = [], 0, 0, 0
        for line in read_text(os.path.join(d, f)).split("\n"):
            m = re.match(r"^([0-9a-f]{32})\s+\*?(\S+)$", line.strip())
            if not m:
                continue
            want, path = m.group(1), m.group(2).lstrip("./")
            fp = os.path.join(REPO, path)
            if path.startswith("logs/rsl_rl/"):
                if os.path.isfile(fp):
                    got = inv["md5c"].get(fp)
                    pol.append(dict(path=path, state="일치" if got == want else "불일치", want=want, got=got))
                else:
                    pol.append(dict(path=path, state="파일 없음", want=want, got=None))
            elif os.path.isfile(fp):
                if md5_file(fp) == want:
                    code_same += 1
                else:
                    code_diff += 1
            else:
                code_missing += 1
        out.append(dict(manifest=f"deploy/{f}", policy=pol, code_same=code_same, code_diff=code_diff, code_missing=code_missing))
    return out


def contract_compare(inv):
    """배포 후보 세 정책의 env.yaml 계약 대조(러너가 맞춰야 하는 항목)."""
    rows, envs = [], collections.OrderedDict()
    for role, v in inv["deploy"].items():
        r = inv["by_name"].get(v["run"])
        if r and r.get("env") and "error" not in r["env"]:
            envs[role] = r
    if len(envs) < 2:
        return dict(rows=[], note="env.yaml 을 읽지 못해 대조하지 못함")

    def row(label, fn, must_equal=True):
        vals = collections.OrderedDict((role, fn(r)) for role, r in envs.items())
        same = len(set(json.dumps(x, sort_keys=True, default=str) for x in vals.values())) == 1
        rows.append(dict(item=label, values=vals, same=same, must_equal=must_equal))
    row("제어 주기 [Hz] (sim dt × decimation)", lambda r: [r["env"]["control"]["hz"], r["env"]["control"]["sim_dt"], r["env"]["control"]["decimation"]])
    row("기본자세(힙·무릎·발목 앞뒤, 왼쪽 rad)", lambda r: [r["env"]["default_pose"].get("label"), r["env"]["default_pose"].get("left_hip_f"), r["env"]["default_pose"].get("left_knee"), r["env"]["default_pose"].get("left_ankle_f")])
    row("액션(클래스·스케일·기본자세 오프셋·슬루 °/틱)", lambda r: [r["env"]["action"]["cls"], r["env"]["action"]["scale"], r["env"]["action"]["use_default_offset"], r["env"]["action"]["slew_deg_per_tick"]])
    row("관측 항(이름:함수×이력) 순서", lambda r: [f"{t['name']}:{t['func']}×{t['hist']}" for t in r["env"]["obs_policy"]])
    row("관측 필터(actions·joint_vel β·joint_vel_ema·접촉)", lambda r: [r["env"]["obs_filter"][k] for k in ("actions", "joint_vel", "joint_vel_beta", "joint_vel_ema", "feet_contact")])
    row("관측 클록 주파수 인자", lambda r: r["env"]["obs_filter"]["gait_clock_freq"])
    row("배포 계약 지문(위 항목의 해시)", lambda r: r["env"]["contract_fp"])
    row("체크포인트 차원 obs / critic / act", lambda r: [(r.get("dims") or {}).get(k) for k in ("obs", "critic_obs", "act")])
    row("자산(USD)", lambda r: r["env"]["usd"])
    row("학습 액션 지연 DR [틱]", lambda r: r["env"]["action"]["delay_steps"], must_equal=False)
    row("학습 LPF β 범위", lambda r: r["env"]["action"]["lpf_beta_range"], must_equal=False)
    row("명령 범위 vx / vy / wz · 리샘플 [s] · heading_command", lambda r: [r["env"]["command"][k] for k in ("vx", "vy", "wz", "resample_s", "heading_command")], must_equal=False)
    row("이벤트 수 / 보상 항 수(가중 ≠ 0) / 보상 지문", lambda r: [len(r["env"]["events"]), r["env"]["n_rewards"], r["env"]["rewards_fp"]], must_equal=False)
    return dict(rows=rows, note=None)


ROLE_TOKEN = {"서기(STAND)": "stand", "회복(RECOVERY)": "recover", "걷기(WALK)": "walk"}


def gen_token(r):
    lab = re.split(r"_(?:seed\d+|from|c[RS]\d|s\d(?:_|$))", r["label"])[0] or r["name"]
    return re.sub(r"[^a-z0-9]", "", lab.lower())


def naming(inv):
    out = []
    for role, v in inv["deploy"].items():
        r = inv["by_name"].get(v["run"])
        if not r:
            continue
        ex = next((e for e in r["exported"] if e["dir"] == "exported"), None)
        ck = ex["source_ckpt"] if ex else None
        pt = ((ex or {}).get("files") or {}).get("policy.pt", {}).get("md5")
        ox = ((ex or {}).get("files") or {}).get("policy.onnx", {}).get("md5")
        day = (r["ts"] or "")[2:10].replace("-", "")
        alias = f"{ROLE_TOKEN[role]}_{gen_token(r)}_{ck}"
        out.append(dict(role=role, alias=alias, version=f"{alias}@{day}-{(ox or '?' * 8)[:8]}", bridge_key=v["key"], run=r["path"], ckpt=ck, policy_pt_md5=pt, policy_onnx_md5=ox,
                        ckpt_md5=r["ckpt_md5"].get(str(ck)), env_md5=(r["params"].get("env.yaml") or {}).get("md5"), task=r.get("task"), train_date=r["ts"]))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 9. 출력
def to_json(inv, extra):
    runs = []
    for r in inv["runs"]:
        d = {k: v for k, v in r.items() if k not in ("hard_refs", "latest_mtime", "cls_base", "ckpt_sizes")}
        d["hard_refs"] = [dict(kind=x["kind"], file=x["file"], line=x["line"], current=x["current"], ckpts=x["ckpts"]) for x in r["hard_refs"]][:40]
        d["ckpt_size_each"] = sorted(set(r["ckpt_sizes"].values()))
        runs.append(d)
    fields = {
        "runs[].path": "run 디렉터리(저장소 상대, 실제 경로)", "runs[].aliases": "이 run 을 가리키는 심볼릭 링크", "runs[].cls": "분류(classes 중 하나)", "runs[].sub": "폐기 후보의 근거 종류(문서 근거 | 이름·설정 근거)",
        "runs[].reasons": "분류 사유(첫 줄이 결정 사유)", "runs[].curated": "분류표(문서 판정)에서 온 역할·사유·근거(doc, line, needle, found)", "runs[].ckpts": "model_N.pt 의 N 목록(링크 제외)",
        "runs[].ckpt_md5": "체크포인트 md5(--md5 all 이면 전부)", "runs[].key_ckpts": "참조가 걸린 체크포인트 → 이유 목록", "runs[].thin_ckpts": "남기는 run 안에서 참조가 없는 체크포인트(솎기 후보)",
        "runs[].exported[]": "exported*/ 의 파일(md5·크기)·SOURCE_CHECKPOINT·provenance(가중치로 찾은 원본 체크포인트, onnx 에 같은 가중치가 있는지)",
        "runs[].registry": "이 run 을 가리키는 브리지 레지스트리 키", "runs[].hard_refs": "실행 코드 줄이 직접 읽는 곳", "runs[].evaluated": "log_picture 평가 산출물 수(체크포인트별)",
        "runs[].parent/parent_ckpt/children": "웜스타트 계보(agent.yaml)", "runs[].env/agent/dims": "params 요약·체크포인트 차원", "runs[].task/task_src": "학습 태스크와 그 출처(확인|추정|미상)",
        "md5_cache/sig_cache": "다음 실행이 재사용하는 캐시(경로 → [크기, mtime_ns, 값])",
    }
    return dict(generated=extra["generated"], command=extra["command"], repo=REPO, git=extra["git"], disk=extra["disk"], md5_mode=extra["md5_mode"], fields=fields,
                classes=CLASS_ORDER, summary=extra["summary"], deploy=inv["deploy"], deploy_note=inv["deploy_note"], naming=extra["naming"],
                contract_compare=extra["contract"], manifests=extra["manifests"], jetson_files=extra["jetson"], jetson_snapshot=extra["jetson_snapshot"],
                registry=inv["registry"], registry_missing_runs=inv["missing_keys"], dangling_refs=inv["dangling"],
                aliases=inv["aliases"], experiment_aliases=inv["exp_aliases"], strays=inv["strays"], outside_logs=extra["outside"],
                runs=runs, md5_cache=inv["md5c"].m, sig_cache=inv["sigc"].m)


def summarize(inv):
    runs = inv["runs"]
    by = collections.OrderedDict()
    for c in CLASS_ORDER:
        rs = [r for r in runs if r["cls"] == c]
        by[c] = dict(runs=len(rs), ckpts=sum(r["n_ckpt"] for r in rs), bytes=sum(r["size_bytes"] for r in rs),
                     key_ckpts=sum(len(r["key_ckpts"]) for r in rs), thin_ckpts=sum(len(r["thin_ckpts"]) for r in rs), thin_bytes=sum(r["thin_bytes"] for r in rs))
    sub = collections.OrderedDict()
    for s in ("문서 근거", "이름·설정 근거"):
        rs = [r for r in runs if r["cls"] == C_DISCARD and r["sub"] == s]
        sub[s] = dict(runs=len(rs), ckpts=sum(r["n_ckpt"] for r in rs), bytes=sum(r["size_bytes"] for r in rs))
    tot = dict(runs=len(runs), ckpts=sum(r["n_ckpt"] for r in runs), bytes=sum(r["size_bytes"] for r in runs), ckpt_bytes=sum(r["ckpt_bytes"] for r in runs))
    return dict(by_class=by, discard_sub=sub, total=tot, thin_ckpts=sum(v["thin_ckpts"] for v in by.values()), thin_bytes=sum(v["thin_bytes"] for v in by.values()))


def short_path(p):
    return p.replace("logs/rsl_rl/", "", 1)


def ev_str(cur, n=2):
    if not cur or not cur.get("evidence"):
        return ""
    return " · ".join(f"`{e['doc']}`:{e['line']}" if e["found"] else f"`{e['doc']}`(문구 못 찾음)" for e in cur["evidence"][:n]) + (f" 외 {len(cur['evidence']) - n}" if len(cur["evidence"]) > n else "")


def ck_range(r):
    if not r["n_ckpt"]:
        return "0"
    return f"{r['n_ckpt']} ({r['ckpt_min']}–{r['ckpt_max']})"


def write_md(inv, extra, path):
    S = extra["summary"]
    runs = inv["runs"]
    by_name = inv["by_name"]
    reg = inv["registry"]
    L = []
    w = L.append
    w("# 정책·체크포인트 전수 목록과 분류 초안 (1007 밤)")
    w("")
    w(f"> 자동 생성: {extra['generated']} · 브랜치 `{extra['git']['branch']}` @ `{extra['git']['head']}` · 명령:")
    w(f"> `{extra['command']}`")
    w("> 이 문서는 **초안**이다. 최종 후보 선별은 평가 결과로 주 작업자가 정한다. 이 작업에서는 **아무것도 지우거나 옮기지 않았다**(읽기만).")
    w("> 젯슨·ROS·CAN·GPU·Isaac 은 쓰지 않았다. 젯슨에 관한 내용은 저장소 코드·문서와, 같은 폴더에 이미 받아져 있던 md5 목록(스냅숏)을 읽은 것이고 접속해서 확인한 것이 아니다.")
    w("> 기계가 읽는 같은 내용: `reports/night_1007/policy_inventory.json` · 다시 만들기: `python3 reports/night_1007/policy_inventory.py [--md5 all]`")
    w("")
    # ── 0
    w("## 0. 한눈에")
    w("")
    w(f"- `logs/rsl_rl/` 의 run 디렉터리 **{S['total']['runs']} 개**, 체크포인트(model_N.pt) **{S['total']['ckpts']} 개**, 합계 **{human(S['total']['bytes'])}**(체크포인트만 {human(S['total']['ckpt_bytes'])}). run 을 가리키는 심볼릭 링크 {len([a for a in inv['aliases'] if a['is_dir'] and a not in inv['exp_aliases']])} 개와 실험 폴더 링크 {len(inv['exp_aliases'])} 개는 별칭으로만 세고, 실제 경로를 한 번씩만 셌다.")
    w(f"- 디스크: 전체 {human(extra['disk']['total'])} 중 여유 {human(extra['disk']['free'])}({extra['disk']['free_pct']:.0f} %) — 측정 시각 {extra['generated']}.")
    w("")
    w("| 분류 | run | 체크포인트 | 용량 | 그중 핵심 체크포인트 | 솎을 수 있는 체크포인트(남기는 run 안) |")
    w("|---|---:|---:|---:|---:|---:|")
    for c in CLASS_ORDER:
        v = S["by_class"][c]
        w(f"| {c} | {v['runs']} | {v['ckpts']} | {human(v['bytes'])} | {v['key_ckpts'] if c in KEEP else '—'} | {str(v['thin_ckpts']) + ' 개 · ' + human(v['thin_bytes']) if c in KEEP and v['thin_ckpts'] else '—'} |")
    w(f"| **합계** | **{S['total']['runs']}** | **{S['total']['ckpts']}** | **{human(S['total']['bytes'])}** | | {S['thin_ckpts']} 개 · {human(S['thin_bytes'])} |")
    w("")
    ds = S["discard_sub"]
    w(f"- 폐기 후보의 근거별 내역: **문서에 기각·무효 기록이 있는 것 {ds['문서 근거']['runs']} 개({human(ds['문서 근거']['bytes'])})**, 문서 기록 없이 run 이름·설정만으로 무효로 본 것(스모크·점검·디버그·재기동 전 실행) {ds['이름·설정 근거']['runs']} 개({human(ds['이름·설정 근거']['bytes'])}). 뒤쪽은 지시의 정의('문서에 기각 기록')를 엄격히 적용하면 불명이다 — §5.2 에 따로 모았다.")
    w(f"- '솎을 수 있는 체크포인트'는 남기는 run 안에서 **어떤 참조(익스포트 원본·레지스트리·계보·스크립트·문서·평가 산출물)도 걸리지 않은** 중간 저장본이다. run 을 남기더라도 이것만 따로 치울 수 있다. 최근 {ACTIVE_HOURS:.0f} 시간 안에 바뀐 run 은 솎기 대상에서 뺐다.")
    w("")
    w("눈여겨볼 것:")
    w("")
    n_src = n_w = 0
    for role, v in inv["deploy"].items():
        r0 = by_name.get(v["run"])
        ex0 = next((e for e in (r0 or {}).get("exported", []) if e["dir"] == "exported"), None)
        if ex0 and ex0["source_ckpt"] == EXPECT_DEPLOY.get(role):
            n_src += 1
            if (ex0.get("provenance") or {}).get("match_ckpt") == ex0["source_ckpt"]:
                n_w += 1
    man_state = collections.defaultdict(set)
    for m in extra["manifests"]:
        for x in m["policy"]:
            man_state[x["path"]].add(x["state"])
    n_man = 0
    for role, v in inv["deploy"].items():
        r0 = by_name.get(v["run"])
        if r0 and all(man_state.get(f"{r0['path']}/exported/{f}") == {"일치"} for f in ("policy.pt", "policy.onnx")):
            n_man += 1
    cbad = [row["item"] for row in extra["contract"]["rows"] if row["must_equal"] and not row["same"]]
    w(f"- 배포 후보 세 정책: 코드(레지스트리)가 가리키는 run 의 `SOURCE_CHECKPOINT` 가 기대 체크포인트와 같은 것 **{n_src}/3**, "
      + (f"익스포트 가중치가 그 체크포인트와 같은 것 **{n_w}/3**, " if inv.get("use_torch") else "(가중치 대조는 --no-torch 라 하지 않았다) ")
      + f"`exported/policy.pt`·`policy.onnx` 의 md5 가 배포 매니페스트(`deploy/*.md5`)와 같은 것 **{n_man}/3** (§1). "
      + ("세 정책의 러너 계약 항목은 전부 같다(§1 계약 대조)." if extra["contract"]["rows"] and not cbad else "⚠ 러너 계약 대조에서 다른 항목: " + (", ".join(cbad) or "대조 못 함") + "."))
    reg_names0 = set(m["run"] for m in extra["outside"]["models"])
    n_unreg = len([1 for v in inv["deploy"].values() if v["run"] not in reg_names0])
    w(f"- 세 정책 중 {n_unreg} 개가 등록부 `models/` 에 없다. 지금은 git 이 무시하는 `logs/` 와 `deploy/*.md5` 에만 기록이 있다(§8.1·§9).")
    bad0 = []
    meta0 = {}
    for m in extra["outside"]["models"]:
        mm0 = MODEL_RE.match(str(m.get("checkpoint") or ""))
        if mm0:
            meta0[m["run"]] = (m["name"], int(mm0.group(1)))
    for r0 in runs:
        for e0 in r0["exported"]:
            mc0 = (e0.get("provenance") or {}).get("match_ckpt")
            if e0["dir"] == "exported" and mc0 is not None and r0["name"] in meta0 and meta0[r0["name"]][1] != mc0:
                bad0.append(f"`{short_path(r0['path'])}/exported` 는 model_{mc0} 의 익스포트인데 등록부 `{meta0[r0['name']][0]}` 는 model_{meta0[r0['name']][1]}")
    if bad0:
        w("- **익스포트와 등록부가 어긋난 곳이 있다**: " + "; ".join(bad0) + " (§7.4).")
    w(f"- 브리지 `CHECKPOINTS` {len(reg['checkpoints'])} 키 중 {len([m for m in inv['missing_keys'] if m['table'] == 'CHECKPOINTS'])} 키는 가리키는 run 이 로컬에 없다(§7.2).")
    js0 = extra.get("jetson_snapshot")
    if js0:
        parts = []
        for g in extra["jetson"]:
            if g["tiers"][0] == "합성 구조":
                req = [f for f in g["files"] if f["need"].startswith("필수")]
                st = "젯슨에 있고 PC 와 같다" if all(f["jetson"] == "같음" for f in req) else ("**젯슨에 없다**" if all(f["jetson"] == "없음" for f in req) else "일부만 맞는다(§3.4)")
                parts.append(f"{g['labels'][0]} — {st}")
        w(f"- 젯슨 스냅숏({js0['taken']} 에 받아 둔 목록, 이 작업에서 접속하지 않음) 기준 필수 파일: " + " · ".join(parts) + " (§3.4).")
    stale = [(r0, h0) for r0 in runs for h0 in ((r0.get("curated") or {}).get("stale_hints") or [])]
    if stale:
        w("- ⚠ '평가 중'으로 적어 둔 run 에 판정 문구가 생긴 것으로 보인다 — 분류표를 고쳐야 한다: " + "; ".join(f"`{short_path(r0['path'])}` ← `{h0['doc']}`:{h0['line']}" for r0, h0 in stale[:4]))
    w("")
    w("분류 규칙(위에서부터 첫 일치):")
    w("")
    w("1. **배포 후보** — 브리지 `KEW3_SPECS`(합성 구조)가 가리키는 세 run. 이름을 박아 둔 것이 아니라 코드에서 뽑는다.")
    w("2. **평가 중 후보** — 사전 등록 문서만 있고 결과가 아직 없는 run, 또는 분류표에 없고 최근 24 시간 안에 생긴 run(자동). 읽기만 했다.")
    w("3. **기준선·검증 의존** — 브리지 레지스트리 키가 가리키거나, selftest·tests·현행 스크립트(`jetson/`·`rl_walking/`·`tests/`·`deploy/`·`reports/regress/`·`reports/gen_w3/`)의 **실행 코드 줄**이 직접 읽는 run. 주석·docstring 의 사용 예는 세지 않는다.")
    w("4. **기록 보존** — (a) 남기는 run 의 웜스타트 조상(계보), (b) 여러 판정 문서가 수치로 인용하는 대조군, (c) 남기는 정책을 고른 근거 수치(선정 게이트·재현 seed), (d) 이전 최종·2차 후보와 선행 단계, (e) 과거 분석 스크립트가 체크포인트를 직접 읽는 것.")
    w("5. **폐기 후보** — 후보(처치)로 평가했고 문서에 기각·중단·무효 기록이 있으며 위 1–4 에 걸리지 않는 run. 근거 문서와 줄 번호를 붙였다. 같은 실험의 처치가 전부 기각된 대조군도 여기에 둔다.")
    w("6. **불명** — 판정 기록도 사용처도 찾지 못한 run.")
    w("")
    w("분류표(문서 판정)는 스크립트 안 `CURATED` 에 있고, 실행할 때마다 근거 문구를 문서에서 다시 찾아 줄 번호를 붙인다. 문구가 사라지면 그 run 은 불명으로 내려간다.")
    w("")
    # ── 1 배포 후보
    w("## 1. 배포 후보 — 합성 구조의 세 정책")
    w("")
    w(f"출처: `jetson/scripts/rl_bridge_node.py` {inv['deploy_note']}.")
    w("")
    w("| 역할 | 브리지 키 | run(실제 경로) | 체크포인트 | 학습 태스크 | 학습 시각(체크포인트 첫–끝 저장) | 근거 문서:줄 |")
    w("|---|---|---|---|---|---|---|")
    for role, v in inv["deploy"].items():
        r = by_name.get(v["run"])
        if not r:
            w(f"| {role} | {v['key']} | ⚠ run 없음: {v['run']} | | | | |")
            continue
        ex = next((e for e in r["exported"] if e["dir"] == "exported"), None)
        ck = ex["source_ckpt"] if ex else None
        exp = EXPECT_DEPLOY.get(role)
        flag = "" if ck == exp else f" ⚠ 기대 {exp}"
        w(f"| {role} | `{v['key']}` | `{r['path']}` | model_{ck}.pt{flag} | `{r.get('task')}` ({r.get('task_src', '').split(':')[0]}) | {' – '.join(r.get('train_span') or [])} | {ev_str(r['curated'], 3)} |")
    w("")
    w("| 역할 | 파일 | 크기 [B] | md5 |")
    w("|---|---|---:|---|")
    for n in extra["naming"]:
        r = by_name[os.path.basename(n["run"])]
        ex = next((e for e in r["exported"] if e["dir"] == "exported"), None)
        rows = []
        if ex:
            for f in ("policy.onnx", "policy.pt", "SOURCE_CHECKPOINT"):
                if f in ex["files"]:
                    rows.append((f"exported/{f}", ex["files"][f]["size"], ex["files"][f]["md5"]))
        for f in ("env.yaml", "agent.yaml"):
            if f in r["params"]:
                rows.append((f"params/{f}", r["params"][f]["size"], r["params"][f]["md5"]))
        if n["ckpt"] is not None and str(n["ckpt"]) in r["ckpt_md5"]:
            rows.append((f"model_{n['ckpt']}.pt (학습 체크포인트 원본)", int(r["ckpt_sizes"][str(n["ckpt"])]), r["ckpt_md5"][str(n["ckpt"])]))
        for i, (f, sz, h) in enumerate(rows):
            w(f"| {n['role'] if i == 0 else ''} | `{f}` | {sz} | `{h}` |")
    w("")
    w("점검 결과(오프라인):")
    w("")
    for role, v in inv["deploy"].items():
        r = by_name.get(v["run"])
        if not r:
            continue
        ex = next((e for e in r["exported"] if e["dir"] == "exported"), None)
        ck = ex["source_ckpt"] if ex else None
        exp = EXPECT_DEPLOY.get(role)
        al = ", ".join(f"`{a}`" for a in r["aliases"]) or "없음"
        pv = (ex or {}).get("provenance") or {}
        if pv.get("checked"):
            wt = (f"`exported/policy.pt` 의 가중치(actor 8 텐서)가 `model_{pv.get('match_ckpt')}.pt` 와 **바이트까지 같다**" if pv.get("match_ckpt") is not None else "**⚠ `exported/policy.pt` 의 가중치와 같은 체크포인트를 이 run 에서 찾지 못했다**")
            wt += (" · `policy.onnx` 안에도 같은 가중치 바이트가 들어 있다" if pv.get("onnx_has_same_weights") else " · **⚠ `policy.onnx` 에서 같은 가중치 바이트를 찾지 못했다**")
        else:
            wt = "가중치 대조는 하지 않았다(" + (pv.get("note") or "--no-torch") + ")"
        w(f"- {role}: `exported/SOURCE_CHECKPOINT` = `{(ex or {}).get('source_checkpoint')}` → 기대값 model_{exp}.pt 와 {'**일치**' if ck == exp else '**불일치 ⚠**'}. {wt}. 별칭(심볼릭 링크) {al}. 평가 산출물 {r['n_eval_files']} 건(그중 model_{ck}: {(r['evaluated'].get(str(ck)) or {}).get('n', 0)} 건).")
    w("")
    w("- 가중치 대조가 보는 것은 '같은 숫자가 들어 있다'까지다. onnx 그래프의 연산 구조가 TorchScript 와 같은 출력을 내는지는 이 검사로 알 수 없다(W3S3d 는 `rl_walking/deploy/selftest_w3s3d_onnx.py` 가 따로 본다).")
    w("")
    w("매니페스트 대조(`deploy/*.md5` 의 정책 쪽 줄을 지금 파일과 비교):")
    w("")
    w("| 매니페스트 | 정책·env 줄 | 코드 줄(지금 작업 트리와 같음 / 다름 / 없음) |")
    w("|---|---|---|")
    for m in extra["manifests"]:
        st = collections.Counter(x["state"] for x in m["policy"])
        pol = " · ".join(f"{k} {v}" for k, v in st.items()) or "정책 줄 없음"
        w(f"| `{m['manifest']}` | {pol} | {m['code_same']} / {m['code_diff']} / {m['code_missing']} |")
    w("")
    w("- 정책 쪽 '파일 없음'은 0905 이전 세대처럼 run 디렉터리가 이미 로컬에 없는 경우다(§7.2). 코드 줄이 '다름'인 것은 매니페스트가 작성 시점의 코드 스냅숏이기 때문이다 — 지금 코드로 배포하려면 매니페스트를 새로 만들어야 한다.")
    w("")
    cc = extra["contract"]
    if cc["rows"]:
        roles = list(cc["rows"][0]["values"].keys())
        w("세 정책의 학습 env.yaml 계약 대조(러너가 한 계약으로 세 정책을 돌릴 수 있는가):")
        w("")
        w("| 항목 | " + " | ".join(roles) + " | 같음 |")
        w("|---|" + "---|" * len(roles) + "---|")
        for row in cc["rows"]:
            cells = []
            for rl in roles:
                v = row["values"][rl]
                t = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
                cells.append((t[:150] + "…") if len(t) > 150 else t)
            if row["item"].startswith("관측 항") and row["same"]:
                cells = [cells[0]] + ["(같음)"] * (len(roles) - 1)
            mark = "예" if row["same"] else ("**아니오 ⚠**" if row["must_equal"] else "아니오(학습 분포 차이 — 계약 아님)")
            w(f"| {row['item']} | " + " | ".join(cells) + f" | {mark} |")
        w("")
        bad = [row["item"] for row in cc["rows"] if row["must_equal"] and not row["same"]]
        w("- 맞아야 하는 항목이 전부 같다." if not bad else f"- ⚠ 맞아야 하는데 다른 항목: {', '.join(bad)} — 아래 메모 참조.")
        w("")
    lin_missing = []
    for role, v in inv["deploy"].items():
        cur = by_name.get(v["run"])
        seen = set()
        while cur and cur["name"] not in seen and cur["parent"]:
            seen.add(cur["name"])
            par = by_name.get(cur["parent"])
            if par is None or cur["parent_ckpt"] not in par["ckpts"]:
                lin_missing.append(f"{cur['parent']}/model_{cur['parent_ckpt']}.pt")
            cur = par
    w("계보(웜스타트 체인, `params/agent.yaml` 의 load_run·load_checkpoint — "
      + ("부모 체크포인트 파일이 전부 로컬에 있다" if not lin_missing else "⚠ 로컬에 없는 부모 체크포인트: " + ", ".join(sorted(set(lin_missing)))) + "):")
    w("")
    for role, v in inv["deploy"].items():
        r = by_name.get(v["run"])
        chain, cur, seen = [], r, set()
        while cur and cur["name"] not in seen:
            seen.add(cur["name"])
            chain.append(cur)
            cur = by_name.get(cur["parent"]) if cur["parent"] else None
        txt = " ← ".join(f"{gen_token(c) or c['name']}" + (f"(model_{chain[i - 1]['parent_ckpt']})" if i > 0 else "") for i, c in enumerate(chain))
        w(f"- {role}: {txt}" + ("(처음부터 학습)" if chain and not chain[-1]["parent"] else ""))
    w("")
    # ── 2 평가 중
    w("## 2. 평가 중 후보 (읽기만 했다)")
    w("")
    w("| run | 체크포인트 | exported* | 근거 | 마지막 변경 |")
    w("|---|---|---|---|---|")
    for r in runs:
        if r["cls"] == C_EVAL:
            exs = ", ".join(f"{e['dir']}(← {e['source_checkpoint']})" for e in r["exported"]) or "없음"
            w(f"| `{r['path']}` | {ck_range(r)} | {exs} | {r['reasons'][0]} {ev_str(r['curated'])} | {r['latest']} |")
    w("")
    w("- 학습 중인 run 은 체크포인트가 더 생긴다. 최근 3 분 안에 바뀐 파일은 md5 를 건너뛰었다(쓰는 중일 수 있음).")
    w("")
    # ── 3 젯슨
    w("## 3. 젯슨 쪽 정책 파일 요구 목록")
    w("")
    w("코드에서 확인한 것: 브리지는 정책마다 `<run>/exported/` 에서 `policy.onnx` 를 먼저 찾고(onnxruntime 이 있으면), 없으면 `policy.pt` 를 torch 로 읽는다(`rl_walking/deploy/policy_runner.py` `_load_backend`). "
      "계약(관측 필터·기본자세·W3 명령 범위)은 `<run>/params/env.yaml` 에서 줄 단위로 읽는다(`_ckpt_env_yaml`·`_parse_env_contract`·`_w3_env_contract`). "
      "`SOURCE_CHECKPOINT`·`agent.yaml`·`model_N.pt` 는 읽지 않는다. 회복 정책(C5 의 두 번째 정책)은 `policy.*` 만 읽고 계약은 서기 정책 것을 쓴다(`C5Runner`).")
    w("")
    w("현장 확인 필요: 젯슨에 어떤 백엔드가 깔려 있는지는 이 작업에서 확인하지 않았다. 주 작업자 기록(`reports/night_1007/LOG.md` 22:20 상태판)에는 'torch 없음, onnxruntime 1.15.1 → 젯슨 백엔드는 onnx' 로 적혀 있다 — 그렇다면 젯슨에서 실제로 도는 파일은 `policy.onnx` 다.")
    w("")
    for tier in ("합성 구조", "selftest 필수", "콘솔 선택지"):
        gs = [g for g in extra["jetson"] if g["tiers"][0] == tier]
        if not gs:
            continue
        w(f"### 3.{['합성 구조', 'selftest 필수', '콘솔 선택지'].index(tier) + 1} {tier}" + {"합성 구조": " — 세 정책(이것이 최소 집합)", "selftest 필수": " — 브리지 `--selftest` 가 PASS 하려면 함께 있어야 하는 것", "콘솔 선택지": " — 콘솔 드롭다운에서 고를 수 있는 나머지"}[tier])
        w("")
        w("| 요구하는 쪽 | 파일(저장소 상대 경로 = 젯슨 `~/humanoid_repo` 상대 경로) | 필요 | 크기 [B] | md5(PC) | 젯슨 스냅숏 |")
        w("|---|---|---|---:|---|---|")
        for g in gs:
            for i, f in enumerate(g["files"]):
                lab = "<br>".join(g["labels"]) if i == 0 else ""
                if f["exists"]:
                    w(f"| {lab} | `{f['path']}` | {f['need']} | {f['size']} | `{f['md5']}` | {f['jetson']} |")
                else:
                    w(f"| {lab} | `{f['path']}` | {f['need']} | — | **PC 에 없음** | {f['jetson']} |")
        w("")
    js = extra["jetson_snapshot"]
    if js:
        w(f"### 3.4 젯슨 스냅숏과의 대조 — `{js['file']}` ({js['taken']} 에 받아 둔 목록)")
        w("")
        w("이 파일은 이 작업이 만든 것이 아니다(같은 폴더에 있던 기록을 읽기만 했다). **젯슨에는 접속하지 않았고, 스냅숏 뒤에 젯슨이 바뀌었는지는 모른다.**")
        w("")
        need = [f for g in extra["jetson"] if g["tiers"][0] == "합성 구조" for f in g["files"]]
        miss = [f for f in need if f["jetson"] == "없음" and f["need"].startswith("필수")]
        diff = [f for f in need if "다름" in f["jetson"]]
        w(f"- 합성 구조에 필요한 파일 {len(need)} 개 중 스냅숏과 **같음 {len([f for f in need if f['jetson'] == '같음'])}**, 다름 {len(diff)}, 스냅숏에 **없음 {len([f for f in need if f['jetson'] == '없음'])}**.")
        for g in extra["jetson"]:
            if g["tiers"][0] == "합성 구조":
                st = collections.Counter(f["jetson"] for f in g["files"])
                w(f"  - {g['labels'][0]} `{short_path(os.path.dirname(g['exported']))}`: " + " · ".join(f"{k} {v}" for k, v in st.items()))
        if miss:
            w("- **젯슨에 없는 필수 파일**(스냅숏 기준): " + ", ".join(f"`{f['path']}`" for f in miss))
        c = js["counts"]
        w(f"- 스냅숏의 `logs/rsl_rl/` 파일 {js['n']} 개 전체: PC 와 같음 {c.get('같음', 0)} · **다름 {c.get('다름', 0)}** · PC 에 없음 {c.get('PC 에 없음', 0)}(PC 에서는 그 run 디렉터리가 이미 없다).")
        for r in js["rows"]:
            if r["state"] == "다름":
                w(f"  - **다름**: `{r['path']}` — 젯슨 `{r['jetson']}` / PC `{r['pc']}`")
        only = [k for k, v in js["by_run"].items() if v == ["PC 에 없음"]]
        if only:
            w(f"  - PC 에 run 이 없는 것 {len(only)} 개: " + ", ".join(f"`{short_path(k)}`" for k in only))
        w("")
    tot = sum(f["size"] for g in extra["jetson"] if g["tiers"][0] == "합성 구조" for f in g["files"] if f["exists"])
    w(f"- 합성 구조 최소 집합의 크기: {human(tot)}(세 정책의 onnx·pt·env.yaml·SOURCE_CHECKPOINT 전부).")
    w("- 합성 키는 `--experimental-w3 --experimental-kew3` 를 함께 줄 때만 등록된다(기본 꺼짐). 이 목록은 정책 쪽 파일만이다 — 코드 파일(브리지·러너·`w3_deploy.py`·`kew3_runner.py`·`c5_switch.py` 등)은 배포 패키지 담당의 몫이다.")
    w("- `policy_runner_stage4.py --selftest` 의 실모델 시험은 기본 경로가 없으면 `logs/rsl_rl/biped12_stage4/*/exported/policy.pt` 가운데 **이름순 마지막**을 쓴다(젯슨에서는 `policy.pt` 가 하나도 없으면 실패). 무엇이 마지막인지는 그 폴더의 디렉터리·링크 구성에 달려 있다.")
    w("")
    # ── 4 전체 표
    w("## 4. 분류 전체 표")
    w("")
    w("열 설명 — 체크포인트: 개수(최소–최대) · exported: 디렉터리(← SOURCE_CHECKPOINT) · 평가: log_picture 결과 파일 머리말에 이 run 이 나온 횟수 · 핵심: 참조가 걸린 체크포인트 · 솎기: 남기는 run 안에서 참조가 없는 체크포인트 수.")
    w("")
    for c in CLASS_ORDER:
        rs = [r for r in runs if r["cls"] == c]
        if not rs:
            continue
        w(f"### 4.{CLASS_ORDER.index(c) + 1} {c} — {len(rs)} 개 · {human(sum(r['size_bytes'] for r in rs))}")
        w("")
        if c == C_DISCARD:
            w("| run | 체크포인트 | 용량 | 평가 | 역할 — 사유 | 근거(문서:줄) |")
            w("|---|---|---:|---:|---|---|")
            for r in sorted(rs, key=lambda r: -r["size_bytes"]):
                cur = r["curated"]
                w(f"| `{short_path(r['path'])}` | {ck_range(r)} | {human(r['size_bytes'])} | {r['n_eval_files']} | {r['reasons'][0]} | {ev_str(cur, 3) or '(문서 근거 없음 — 이름·설정)'} |")
        else:
            w("| run | 날짜 | 체크포인트 | 용량 | exported | 태스크 | 평가 | 핵심 체크포인트 | 솎기 | 사유 | 근거(문서:줄) |")
            w("|---|---|---|---:|---|---|---:|---|---:|---|---|")
            for r in rs:
                exs = "<br>".join(f"{e['dir']}(← {e['source_checkpoint'] or '표시 없음'})" for e in r["exported"]) or "—"
                keys = ", ".join(r["key_ckpts"]) or "—"
                task = (r.get("task") or "미상").replace("Biped12-", "")
                if r.get("task_src", "").startswith("추정"):
                    task += "(추정)"
                w(f"| `{short_path(r['path'])}` | {(r['ts'] or '')[:10]} | {ck_range(r)} | {human(r['size_bytes'])} | {exs} | {task} | {r['n_eval_files']} | {keys} | {len(r['thin_ckpts']) or '—'} | {' / '.join(r['reasons'][:3])} | {ev_str(r['curated'])} |")
        w("")
    # ── 5 폐기 후보 근거
    w("## 5. 폐기 후보 — 근거와 용량 순위")
    w("")
    dis = sorted([r for r in runs if r["cls"] == C_DISCARD], key=lambda r: -r["size_bytes"])
    w("### 5.1 용량 순 상위 20 (문서 근거 있는 것)")
    w("")
    w("| # | run | 용량 | 체크포인트 | 판정 | 근거 문서:줄 |")
    w("|---:|---|---:|---|---|---|")
    for i, r in enumerate([r for r in dis if r["sub"] == "문서 근거"][:20], 1):
        w(f"| {i} | `{short_path(r['path'])}` | {human(r['size_bytes'])} | {ck_range(r)} | {r['curated']['why']} | {ev_str(r['curated'], 3)} |")
    w("")
    w("### 5.2 문서 기록 없이 이름·설정만으로 무효로 본 것")
    w("")
    w("지시의 정의대로라면 불명이다. run 이름(smoke·chk·dbg·`_void_*` 폴더)과 반복 수, 평가 산출물이 없다는 사실만 근거다.")
    w("")
    w("| run | 용량 | 체크포인트 | max_iterations | 평가 산출물 | 메모 |")
    w("|---|---:|---|---:|---:|---|")
    for r in [r for r in dis if r["sub"] == "이름·설정 근거"]:
        w(f"| `{short_path(r['path'])}` | {human(r['size_bytes'])} | {ck_range(r)} | {(r.get('agent') or {}).get('max_iterations')} | {r['n_eval_files']} | {r['reasons'][0][:120]} |")
    w("")
    w("### 5.3 근거 문서별 묶음")
    w("")
    bydoc = collections.defaultdict(list)
    for r in dis:
        for e in (r["curated"] or {}).get("evidence", [])[:1]:
            bydoc[e["doc"]].append(r)
    w("| 근거 문서 | run 수 | 용량 | run(짧은 이름) |")
    w("|---|---:|---:|---|")
    for doc, rs in sorted(bydoc.items(), key=lambda kv: -sum(r["size_bytes"] for r in kv[1])):
        w(f"| `{doc}` | {len(rs)} | {human(sum(r['size_bytes'] for r in rs))} | {', '.join((r['label'] and gen_token(r)) or r['name'] for r in rs)} |")
    w("")
    w("- `logs/rsl_rl/_void_1003_bench/` 는 이미 누군가(1003~1004) 무효 실행을 따로 모아 둔 폴더다. 그 안의 run 은 실험 폴더 밖에 있어서 `--load_run` 으로 잡히지 않는다 — 사실상 이미 격리된 상태다.")
    w("")
    # ── 6 불명
    w("## 6. 불명")
    w("")
    unk = [r for r in runs if r["cls"] == C_UNK]
    w("| run | 용량 | 체크포인트 | exported | 왜 불명인가 | 참고 |")
    w("|---|---:|---|---|---|---|")
    for r in unk:
        exs = ", ".join(f"{e['dir']}" for e in r["exported"]) or "—"
        hint = []
        if r["parent"]:
            hint.append(f"부모 {r['parent']}/model_{r['parent_ckpt']}")
        if r["children"]:
            hint.append("자식 " + ", ".join(c["run"] for c in r["children"]))
        if r["doc_files"]:
            hint.append("문서 언급 " + ", ".join(r["doc_files"][:2]))
        w(f"| `{short_path(r['path'])}` | {human(r['size_bytes'])} | {ck_range(r)} | {exs} | {r['reasons'][0]} {ev_str(r['curated'])} | {' · '.join(hint) or '—'} |")
    w("")
    # ── 7 참조 관계
    w("## 7. 참조 관계")
    w("")
    w("### 7.1 브리지 레지스트리와 콘솔 선택지")
    w("")
    w(f"- 콘솔 선택지(`operator_console.py` `CKPT_CHOICES`): {', '.join('`' + c + '`' for c in reg['console_choices'])} — 기본값 `{reg['console_default']}`.")
    w(f"- `STAND_DEFAULT` = `{reg['stand_default']}` · env.yaml 필수 키(`CKPT_REQUIRE_ENV_YAML`): {', '.join('`' + c + '`' for c in reg['require_env_yaml'])} · `CKPT_JVEL_EMA_MODE`: {reg['jvel_ema_mode']}.")
    w("")
    w("| 표 | 키 | 가리키는 run | exported | 콘솔 | 로컬 |")
    w("|---|---|---|---|---|---|")
    rows = []
    for key, p in reg["checkpoints"].items():
        rows.append(("CHECKPOINTS", key, p))
    for key, v in reg["c5_pairs"].items():
        rows.append(("C5_PAIRS(회복)", key, v["rec_path"]))
    for key, v in reg["w3_specs"].items():
        rows.append((f"W3_SPECS · {v['model']}", key, v["run"] + "/exported"))
    for key, v in reg["hturn2"].items():
        rows.append((f"HTURN2 · {v['model']}", key, v["run"] + "/exported"))
    for tab, key, p in rows:
        rn = run_of_path(p)
        here = rn in by_name
        exok = os.path.isdir(os.path.join(REPO, p))
        if not here and tab == "CHECKPOINTS":
            continue
        w(f"| {tab} | `{key}` | `{short_path(os.path.dirname(p.rstrip('/')))}` | {'있음' if exok else '**없음**'} | {'예' if key in reg['console_choices'] else ''} | {'있음' if here else '**run 없음**'} |")
    for key, sp in reg["kew3_specs"].items():
        w(f"| KEW3_SPECS(합성) | `{key}` | 걷기 = `{sp.get('w3')}` · 서기+회복 = `{sp.get('ke')}` | — | | |")
    w("")
    w("### 7.2 이미 끊어져 있는 참조(로컬에 run 이 없는 것)")
    w("")
    mk = [m for m in inv["missing_keys"]]
    w(f"- `CHECKPOINTS` 의 {len(reg['checkpoints'])} 키 중 **{len([m for m in mk if m['table'] == 'CHECKPOINTS'])} 키**는 run 디렉터리가 로컬 `logs/rsl_rl/` 에 없다: " + ", ".join(f"`{m['key']}`" for m in mk if m["table"] == "CHECKPOINTS") + ".")
    w("  이 키들은 브리지 selftest 에서 '미배포'로 SKIP 된다(파일이 없는 등록 항목은 FAIL 이 아니라 SKIP). 정책 사본은 `models/<이름>/` 에 일부 남아 있다(§8).")
    code_d = [d for d in inv["dangling"] if d["code"]]
    w(f"- 코드·스크립트가 이름으로 가리키는데 로컬에 없는 run: {len(code_d)} 개(문서만 가리키는 것까지 합치면 {len(inv['dangling'])} 개). 코드 쪽:")
    for d in sorted(code_d, key=lambda d: d["name"]):
        w(f"  - `{d['name']}` ← {', '.join('`' + c + '`' for c in d['code'][:4])}" + (f" 외 {len(d['code']) - 4}" if len(d["code"]) > 4 else "") + (f" · 평가 산출물 {d['eval_outputs']} 건" if d["eval_outputs"] else ""))
    km = [(r, r["key_missing"]) for r in runs if r["key_missing"]]
    if km:
        w("- run 은 있는데 **참조된 체크포인트 파일이 없는 것**:")
        for r, k in km:
            w(f"  - `{short_path(r['path'])}`: " + "; ".join(f"model_{c}.pt ← {v[0]}" for c, v in list(k.items())[:6]))
    w("")
    w("### 7.3 심볼릭 링크(별칭)")
    w("")
    dl = [a for a in inv["aliases"] if a["is_dir"] and a not in inv["exp_aliases"]]
    fl = [(r, l) for r in runs for l in r["ckpt_links"]]
    w(f"- run 디렉터리 링크 {len(dl)} 개: 대부분 `biped12_stage4/<run>` → `../biped12_stepb0/<run>` (G11a 지면 평가 태스크가 `biped12_stage4` 에서 run 을 찾기 때문 — `claude2.md` §8). 부모 run 링크(`biped12_{{w1,g22r,stepb0}}/…_g21a` → `../biped12_stage4/…_g21a`)도 있다.")
    for a in inv["exp_aliases"]:
        w(f"- 실험 폴더 링크: `{a['link']}` → `{a['raw']}` — **자기 자신의 형제를 가리키는 고리**다(`biped12_stage4/biped12_stepb0/…` 로 같은 run 이 한 번 더 보인다). 재귀로 훑는 도구(`find -L`, `du -L`, glob `**`)는 중복 집계한다.")
    w(f"- run 안의 파일 링크 {len(fl)} 개: `model_11600.pt → model_11599.pt`, `model_11000.pt → model_10999.pt` 꼴(마지막 저장본을 평가 격자의 반복 번호로 부르기 위한 것). 링크 대상을 치우면 그 번호를 쓰는 평가가 깨진다. 핵심 체크포인트 계산에서는 링크를 대상 파일로 바꿔 셌다.")
    broken = [a for a in inv["aliases"] if not a["exists"]]
    w(f"- 깨진 링크: {len(broken)} 개" + ("" if not broken else " — " + ", ".join(f"`{a['link']}`" for a in broken)))
    w("")
    w("### 7.4 `exported*/` 가 어느 체크포인트에서 나왔는가(가중치 대조)")
    w("")
    w("`SOURCE_CHECKPOINT` 파일은 1003 이후 run 에만 있다. 없는 run 은 `policy.pt` 의 가중치를 그 run 의 체크포인트들과 맞대어 찾았다. '등록부'는 `models/*/meta.json` 이 적어 둔 체크포인트다.")
    w("")
    w("| run | 디렉터리 | SOURCE_CHECKPOINT | 가중치가 같은 체크포인트 | 등록부가 말하는 체크포인트 | onnx 에 같은 가중치 | 판정 |")
    w("|---|---|---|---|---|---|---|")
    meta_ck = {}
    for m in extra["outside"]["models"]:
        mm = MODEL_RE.match(str(m.get("checkpoint") or ""))
        if mm:
            meta_ck.setdefault(m["run"], []).append((m["name"], int(mm.group(1))))
    prov_bad = []
    for r in runs:
        for e in r["exported"]:
            pv = e.get("provenance") or {}
            claims = meta_ck.get(r["name"], []) if e["dir"] == "exported" else []
            mc = pv.get("match_ckpt")
            if not pv.get("checked"):
                verdict = "대조 안 함(" + (pv.get("note") or "최근 변경·--no-torch") + ")"
            elif mc is None:
                verdict = "일치하는 체크포인트 없음" + ("(체크포인트 원본이 로컬에 없다)" if not r["ckpts"] else " ⚠")
            elif e["source_ckpt"] is not None and mc != e["source_ckpt"]:
                verdict = "**⚠ SOURCE_CHECKPOINT 와 다름**"
            elif claims and any(c != mc for _, c in claims):
                verdict = "**⚠ 등록부와 다름**"
            else:
                verdict = "일치"
            if "⚠" in verdict:
                prov_bad.append((r, e, mc, claims))
            w(f"| `{short_path(r['path'])}` | {e['dir']} | {e['source_checkpoint'] or '—'} | {('model_' + str(mc) + '.pt') if mc is not None else '—'} | {', '.join(f'{n}: model_{c}.pt' for n, c in claims) or '—'} | {'예' if pv.get('onnx_has_same_weights') else ('—' if not pv.get('checked') else '**아니오 ⚠**')} | {verdict} |")
    w("")
    for r, e, mc, claims in prov_bad:
        keys = sorted(set(x["key"] for x in r["registry"]))
        mf = sorted(set(m["manifest"] for m in extra["manifests"] for x in m["policy"] if x["state"] == "불일치" and x["path"].startswith(f"{r['path']}/{e['dir']}/")))
        w(f"- ⚠ `{r['path']}/{e['dir']}` 는 **model_{mc}.pt 의 익스포트**다. 등록부({', '.join(n for n, _ in claims)})는 model_{claims[0][1] if claims else '?'}.pt 의 익스포트를 담고 있고(가중치로 확인, §8.1)"
          + (f", {', '.join('`' + x + '`' for x in mf)} 에 적힌 md5 도 지금 이 디렉터리의 파일과 다르다" if mf else "") + ". "
          + (f"브리지 키 {', '.join('`' + k + '`' for k in keys)} 는 이 디렉터리를 읽으므로, **PC 에서 그 키로 도는 정책은 등록부·매니페스트가 말하는 정책과 다르다.**" if keys else ""))
        js2 = extra.get("jetson_snapshot")
        if js2:
            jr = [x for x in js2["rows"] if x["path"].startswith(f"{r['path']}/{e['dir']}/policy.")]
            if jr:
                reg_md5 = set(f["md5"] for m in extra["outside"]["models"] if m["run"] == r["name"] for f in m["files"].values())
                txt = f"  - 젯슨 스냅숏({js2['taken']})의 같은 경로: " + ", ".join(f"`{os.path.basename(x['path'])}` PC 와 {x['state']}" + ("(= 등록부 사본과 같은 파일)" if x["jetson"] in reg_md5 else "") for x in jr)
                if all(x["jetson"] in reg_md5 for x in jr):
                    txt += " — 스냅숏 기준으로 젯슨에는 등록부와 같은 파일이 있고, PC 의 `exported/` 만 다르다."
                w(txt)
    if not prov_bad:
        w("- 어긋난 것 없음.")
    w("")
    # ── 8 밖 사본
    o = extra["outside"]
    w("## 8. `logs/rsl_rl/` 밖에 있는 정책 사본")
    w("")
    w("### 8.1 `models/` (등록부 — `rl_walking/scripts/register_model.py` 가 만든다, git 추적)")
    w("")
    w("| 이름 | run / 체크포인트 | 태스크 | 파일 md5 = meta.json | run 이 로컬에 | 사본 가중치 = 그 체크포인트 | 지금 run 의 exported 와 같은 파일 |")
    w("|---|---|---|---|---|---|---|")
    for m in o["models"]:
        same = "예" if any(f["same_as"] for f in m["files"].values()) else ("—" if not m["run_local"] else "**아니오 ⚠**(§7.4)")
        wm = {True: "예", False: "**아니오 ⚠**", None: "확인 불가(체크포인트 없음)" if m["run_local"] else "—"}[m["weights_match_ckpt"]]
        w(f"| `{m['dir']}` | {m['run']} / {m['checkpoint']} | {(m['task'] or '').replace('Biped12-', '')} | {'일치' if m['files_match_meta'] else '**불일치 ⚠**'} | {'있음' if m['run_local'] else '없음'} | {wm} | {same} |")
    w("")
    reg_names = set(m["run"] for m in o["models"])
    miss = [f"{role} `{by_name[v['run']]['path']}`" for role, v in inv["deploy"].items() if v["run"] in by_name and v["run"] not in reg_names]
    w("- **배포 후보 가운데 `models/` 에 등록되지 않은 것: " + (", ".join(miss) if miss else "없음") + ".** 등록부의 마지막 항목은 `stand_crouch_g12`(0929)다. 그 뒤 세대는 git 이 무시하는 `logs/` 에만 있고, 체크섬은 `deploy/*.md5` 에 흩어져 있다.")
    w("")
    w("### 8.2 `deploy/cand/` (익스포트 보관 사본)")
    w("")
    w("| 폴더 | 파일 | md5 | 이름이 말하는 체크포인트와 가중치가 같은가 | 같은 내용의 exported |")
    w("|---|---|---|---|---|")
    for c in o["deploy_cand"]:
        wm = {True: "예", False: "**아니오 ⚠**", None: "—"}[c.get("weights_match_ckpt")]
        for f, v in c["files"].items():
            if f.startswith("policy."):
                w(f"| `{c['dir']}` | {f} | `{v['md5']}` | {wm if f == 'policy.pt' else ''} | {', '.join('`' + short_path(x) + '`' for x in v['same_as']) or '없음(그 run 의 `exported/` 는 지금 다른 체크포인트의 익스포트다)'} |")
    w("")
    w("### 8.3 워크트리 안의 체크포인트")
    w("")
    if o["worktrees"]:
        for t in o["worktrees"]:
            w(f"- `{t['worktree']}/logs` 는 링크가 아니라 **실제 폴더**다: 체크포인트·정책 파일 {t['n']} 개({human(t['bytes'])}). 본 트리와 내용이 "
              + ("**전부 같다**(md5) — 본 트리에 사본이 있다." if t["all_identical_to_main"] else "**다르거나 본 트리에 없는 것이 있다 ⚠**: " + ", ".join(t["only_here"][:5])))
    else:
        w("- 워크트리의 `logs` 는 전부 본 트리로 가는 링크이거나 없다(따로 든 체크포인트 없음).")
    w("")

    # ── 9 명명 제안
    w("## 9. 명명 제안(초안) — 기존 브리지 키는 그대로 두고 별칭·매니페스트 수준에서")
    w("")
    w("지금 이름의 문제(사실): ① 걷기 정책의 키가 `stand_w3s3d_15800` 로 `stand_` 로 시작한다(서기 가족 경로를 타게 하려는 접두사라 바꾸면 동작이 바뀐다). "
      "② 회복 정책은 단독 키가 없다(`C5_PAIRS` 값 안의 경로로만 존재). ③ `stand_crouch_g21a` 는 체크포인트 번호가 이름에 없다. "
      "④ 합성 키(`stand_c5_*`·`stand_c5kr_*`·`stand_c5krh_*`·`stand_kew3_*`)는 같은 정책 파일에 감지기·전환 설정만 다른 이름이다 — 정책 버전과 구성 버전이 한 이름에 섞여 있다.")
    w("")
    w("제안 규칙:")
    w("")
    w("- 정책 별칭 = `<역할>_<세대>_<체크포인트>` — 역할은 `stand`·`recover`·`walk`, 세대는 run 이름의 태그를 소문자로, 체크포인트는 `SOURCE_CHECKPOINT` 의 반복 번호.")
    w("- 버전 꼬리표 = `@<학습 시작일 YYMMDD>-<policy.onnx md5 앞 8 자>` — 젯슨에서 실제로 도는 파일(onnx) 기준. 같은 체크포인트를 다시 내보내 파일이 달라지면 꼬리표가 달라진다.")
    w("- 구성(합성) 이름 = `<구성>__<정책 별칭들>` — 구성은 `ke`(서기 + 회복 + 감지기), `kew3`(ke + 걷기). 감지기·전환 값은 구성 매니페스트에 적고 이름에는 넣지 않는다.")
    w("- 브리지 키·콘솔 선택지·블랙박스의 `checkpoint` 문자열은 **바꾸지 않는다**. 별칭은 매니페스트(`models/<별칭>/meta.json`, `deploy/<구성>.md5`)와 문서에서만 쓴다.")
    w("")
    w("| 역할 | 제안 별칭 @ 버전 | 기존 브리지 키 | run | 체크포인트 | policy.onnx md5 | policy.pt md5 | model_N.pt md5 |")
    w("|---|---|---|---|---|---|---|---|")
    for n in extra["naming"]:
        w(f"| {n['role']} | `{n['version']}` | `{n['bridge_key']}` | `{short_path(n['run'])}` | {n['ckpt']} | `{n['policy_onnx_md5']}` | `{n['policy_pt_md5']}` | `{n['ckpt_md5']}` |")
    w("")
    al = {n["role"]: n["alias"] for n in extra["naming"]}
    if len(al) == 3:
        ke = f"ke__{al['서기(STAND)']}__{al['회복(RECOVERY)']}"
        w("| 구성 | 제안 이름 | 대응하는 기존 키 | 달라지는 것 |")
        w("|---|---|---|---|")
        for key in reg["c5_pairs"]:
            w(f"| 서기 + 회복 | `{ke}` | `{key}` | 감지기·조기 복귀 설정(`C5_DETECTOR_KW`·`C5_EXTRA`) — 정책 파일은 같다 |")
        for key in reg["kew3_specs"]:
            w(f"| 서기 + 회복 + 걷기 | `kew3__{al['서기(STAND)']}__{al['회복(RECOVERY)']}__{al['걷기(WALK)']}` | `{key}` | `KEW3_SPECS` 의 전환 규칙(handoff·정착 시간) |")
        w("")
    w("등록 방법 제안(실행하지 않았다 — `models/` 는 git 추적 폴더라 주 작업자 승인 뒤):")
    w("")
    w("```bash")
    for n in extra["naming"]:
        r = by_name[os.path.basename(n["run"])]
        par = f"{gen_token(by_name[r['parent']])} model_{r['parent_ckpt']}" if r["parent"] else "scratch"
        w(f"python3 rl_walking/scripts/register_model.py {n['alias']} --experiment {os.path.basename(r['exp'])} --run {r['name']} --ckpt model_{n['ckpt']}.pt \\")
        w(f"    --task {n['task']} --usd {(r.get('env') or {}).get('usd', '')} --warm \"{par}\" --summary \"{n['role']} · 합성 구조 1007\"")
    w("```")
    w("")
    w("- 이 도구는 `exported/policy.*` 를 그대로 복사한다. 복사 전에 `exported/SOURCE_CHECKPOINT` 가 `--ckpt` 와 같은지 봐야 한다(§1 점검이 그 확인이다 — 평가 체인 스크립트가 `exported/` 를 지우고 다른 체크포인트로 다시 만드는 경우가 있다: `reports/gen_w3/scripts/s3e_eval.sh`).")
    w("- 관찰: 브리지·러너 코드에는 로드한 정책 파일의 해시를 계산하는 부분이 없다(`hashlib`·`md5` 없음). 실행 기록에 남는 정책 식별자는 키 이름뿐이라, 파일이 바뀌어도 기록으로는 구분되지 않는다. 체크섬 대조는 지금 배포 스크립트 단계에서만 이뤄진다.")
    w("")
    # ── 10 분리 계획
    sp = extra["separation"]
    w("## 10. 폐기 후보 분리 계획(초안 — 실행하지 않았다)")
    w("")
    w("### 10.1 옮기면 깨지는 것")
    w("")
    w(f"폐기 후보 {sp['n_discard']} 개({human(sp['discard_bytes'])})를 `logs/rsl_rl/<실험>/` 밖으로 옮긴다고 할 때:")
    w("")
    w(f"1. **심볼릭 링크 {len(sp['links'])} 개가 끊어진다**(전부 `biped12_stage4/<run>` → `../biped12_stepb0|g22r/<run>`). run 보다 링크를 먼저 지워야 한다:")
    for a in sp["links"][:60]:
        w(f"   - `{a}`")
    w(f"2. **스크립트가 이름으로 가리키는 곳 {len(sp['script_refs'])} 줄**(실행 코드 줄만. 주석·docstring 제외):")
    for x in sp["script_refs"][:40]:
        w(f"   - `{x['file']}`:{x['line']} → `{x['run']}`" + (" (체크포인트를 직접 읽음)" if x["hard"] else " (이름만 — 결과 파일 머리말 대조용)"))
    if not sp["script_refs"]:
        w("   - 없음(run 디렉터리 이름을 그대로 적은 실행 줄은 없다)")
    gd = [g for g in sp["glob_refs"] if g["runs"] is None or g["discard"]]
    w(f"   - 이름 꼬리로 run 을 찾는 줄(`ls -d logs/rsl_rl/<실험>/*_<꼬리> | tail -1`) {len(sp['glob_refs'])} 곳 중 폐기 후보에 걸리거나 꼬리가 변수인 것:")
    for g in gd:
        w(f"     - `{g['file']}`:{g['line']} `{g['pattern']}` → " + ("꼬리가 변수라 실행 때 정해진다" if g["runs"] is None else ", ".join(f"`{n}`" for n in g["discard"])))
    w(f"3. **문서가 경로·이름을 적어 둔 곳**: 문서 {sp['n_doc_files']} 개의 {sp['n_doc_lines']} 줄(판정 기록 — 고칠 필요는 없지만 경로가 죽은 링크가 된다). 많은 순: " + ", ".join(f"`{d}`({n})" for d, n in sp["doc_top"][:8]))
    w(f"4. **평가 산출물 {sp['n_eval_files']} 건**의 머리말이 이 run 들의 체크포인트 경로를 가리킨다(`log_picture/` — 결과 파일은 남지만 같은 평가를 다시 돌릴 수 없게 된다).")
    w(f"5. **웜스타트 기록**: 폐기 후보끼리의 부모–자식 {sp['n_child_links']} 쌍은 `params/agent.yaml` 의 `load_run` 이 죽은 이름이 된다(재현용 기록일 뿐 실행에는 영향 없음). 남기는 run 의 부모는 계보 규칙으로 이미 기록 보존에 들어가 있다.")
    w("6. **이름순 마지막에 기대는 코드**:")
    w(f"   - `isaaclab_tasks.utils.get_checkpoint_path(log_root, run_regex, ckpt_regex)` 는 실험 폴더의 디렉터리(링크 포함) 가운데 이름이 정규식에 **앞에서부터 맞는**(`re.match`) 것을 이름순으로 세워 **마지막**을 고른다. `--load_run` 을 안 주면(`.*`) 폴더의 마지막 항목이 뽑힌다 — 지금 `biped12_stepb0` 의 마지막은 `{sp['last_stepb0']}`, `biped12_stage4` 의 마지막은 `{sp['last_stage4']}`"
      + ("(고리 링크 — 안에 model_*.pt 가 없어서 `.*` 로 부르면 '체크포인트 없음' 오류로 끝난다)." if sp["last_stage4"] == "biped12_stepb0" else "."))
    w("   - 같은 함수라서 `--load_run <이름>` 도 앞부분 일치다: 옮긴 뒤 같은 접두사의 다른 run 이 남아 있으면 그쪽이 잡힌다(타임스탬프가 붙어 있어 지금은 겹치는 이름이 없다).")
    w(f"   - `rl_walking/deploy/policy_runner_stage4.py --selftest` 실모델 시험의 대체 경로 = `biped12_stage4/*/exported/policy.pt` 의 이름순 마지막 — 지금 PC 에서는 `{sp['selftest_fallback']}`.")
    w("   - 평가 체인 스크립트의 `ls -d logs/rsl_rl/biped12_stepb0/*_<이름> | tail -1`(`reports/gen_w3/scripts/w3s3e.sh`·`w3s3dr.sh`·`w3train.sh`) — 같은 꼬리 이름의 run 을 옮기면 다른 run 이 잡히거나 비게 된다.")
    w("7. **평가 태스크가 실험 폴더를 정한다**: `--load_run`·`--switch_to`·`--ke` 는 `logs/rsl_rl/<태스크의 experiment_name>/<run>` 에서만 찾는다(`eval_ground_recovery.py`·`eval_transition*.py`). 실험 폴더 밖으로 나간 run 은 이름을 알아도 잡히지 않는다 — 분리의 목적이기도 하고, 되돌릴 때는 원래 폴더로 돌려놓아야 한다는 뜻이기도 하다.")
    if sp.get("scratch_dirs"):
        sr = [x for x in sp["scratch_refs"] if x["cls"] in (C_DISCARD, C_UNK)]
        fresh = [x for x in sr if x["newest_age_h"] < ACTIVE_HOURS]
        w(f"8. **저장소 밖 대기열 스크립트**(세션 scratchpad {len(sp['scratch_dirs'])} 곳의 .sh/.py, 이번 실행에서 함께 봄): 폐기 후보·불명 run 을 가리키는 스크립트가 {len(sr)} 개 run 에 대해 있고, "
          + (f"그중 **최근 {ACTIVE_HOURS:.0f} 시간 안에 고쳐진 스크립트가 가리키는 것 {len(fresh)} 개** — 도는 중인 대기열일 수 있으니 옮기기 전에 확인: " + ", ".join(f"`{x['run']}` ← `{x['newest_file']}`({x['newest_age_h']} h 전)" for x in fresh) if fresh
             else f"전부 {min((x['newest_age_h'] for x in sr), default=0):.0f} 시간 넘게 손대지 않은 것이다(그 run 을 평가하던 일회성 스크립트). 가장 최근: " + ", ".join(f"`{x['run']}` ← `{x['newest_file']}`({x['newest_age_h']:.0f} h 전)" for x in sorted(sr, key=lambda x: x["newest_age_h"])[:3])))
    w("")
    w("### 10.2 제안 순서")
    w("")
    ev_runs = [r for r in runs if r["cls"] == C_EVAL]
    w("0. 조건: 학습·평가가 하나도 돌지 않을 때" + (f"(이 목록을 만든 시각에는 평가 중 후보 {len(ev_runs)} 개 — {', '.join(gen_token(r) for r in ev_runs)} — 가 있고 평가 대기열이 돌고 있었다)" if ev_runs else "") + ". 평가 중 후보와 그 부모·대조 run 은 건드리지 않는다.")
    w("1. `python3 reports/night_1007/policy_inventory.py --md5 all` 로 전 체크포인트 md5 를 남긴다(이 JSON 이 옮기기 전 기록이 된다).")
    w("2. 주 작업자가 §11 의 결정 사항을 정하고, JSON 에서 대상 목록을 뽑는다(`runs[].cls == \"폐기 후보\"`, 필요하면 `sub` 로 좁힘).")
    w("3. 옮길 곳: 같은 파일시스템의 `logs/rsl_rl/_archive_1007/<원래 실험 폴더>/<run>` — `_void_1003_bench/` 와 같은 구조다. 같은 디스크 안 이동이라 용량은 줄지 않는다. 용량을 줄이려면 그 폴더를 외부 저장소로 내보낸 뒤 지우는 단계가 따로 필요하고, 그것은 사용자 결정이다.")
    w("4. 순서: ① 대상 run 을 가리키는 링크 제거(10.1-1) → ② run 디렉터리 이동 → ③ `find logs/rsl_rl -xtype l` 로 끊어진 링크 0 확인 → ④ 회귀 확인: `python3 rl_walking/deploy/policy_runner_stage4.py --selftest` · `python3 jetson/scripts/rl_bridge_node.py --selftest` · `python3 jetson/scripts/operator_console.py --selftest` · `pytest tests/test_bridge_ema_contract.py` · `reports/regress/run_regress.sh --mode quick` → ⑤ 이 스크립트를 다시 돌려 분류·끊어진 참조가 예상대로인지 본다.")
    w(f"5. (선택) 솎기: 남기는 run 안의 비핵심 체크포인트 {S['thin_ckpts']} 개({human(S['thin_bytes'])})는 JSON 의 `runs[].thin_ckpts` 에 있다. `model_N.pt` 링크의 대상(`ckpt_links`)과 `key_ckpts` 는 제외돼 있다.")
    w("6. 하지 않는 것: `exported*/`·`params/`·`git/`·tfevents 는 run 과 함께 움직이고 따로 떼지 않는다. `log_picture/` 의 결과 파일은 손대지 않는다.")
    w("")
    w("### 10.3 범위 밖이지만 같이 볼 것")
    w("")
    w(f"- `deploy/` 의 옛 배포 묶음 `stand_7차_*.tar.gz` {sp['n_tar']} 개({human(sp['tar_bytes'])}) — 안에 당시 정책 파일이 들어 있다(0907~0929 세대). 이 목록에서는 열어 보지 않았다.")
    w("- `.claude/worktrees/` — 에이전트 작업 사본. `logs` 는 대부분 본 트리 링크지만 사본 하나가 수 GB 다(추적되는 `log_picture/`·`reports/` 때문). 디스크 압박의 주원인은 체크포인트(20 GiB)가 아니라 이쪽일 수 있다 — 이 작업 중에도 여유 공간이 0 까지 내려갔다 돌아왔다. 정리는 `claude2.md` 의 절차(병합 확인 뒤 `git worktree remove`)를 따르는 주 작업자 몫이다.")
    w("")
    # ── 11 결정
    w("## 11. 주 작업자가 정해야 할 것")
    w("")
    n_ctrl = len([r for r in runs if r["cls"] == C_REC and r["curated"] and r["curated"]["role"] in ("대조군", "선정 근거")])
    b_ctrl = sum(r["size_bytes"] for r in runs if r["cls"] == C_REC and r["curated"] and r["curated"]["role"] in ("대조군", "선정 근거"))
    n_lin = len([r for r in runs if r.get("lineage_upgrade")])
    b_lin = sum(r["size_bytes"] for r in runs if r.get("lineage_upgrade"))
    exp_only = [f"`{x['key']}`" for r in runs for x in r["registry"] if not x["exported_exists"]]
    w(f"1. **폐기 후보의 범위** — 문서 근거 {ds['문서 근거']['runs']} 개({human(ds['문서 근거']['bytes'])})만인가, 이름·설정 근거 {ds['이름·설정 근거']['runs']} 개({human(ds['이름·설정 근거']['bytes'])})도 포함인가.")
    w(f"2. **대조군·선정 근거 run 을 남길 것인가** — 여러 판정 문서가 수치로 인용하는 대조군과 선정 근거 {n_ctrl} 개({human(b_ctrl)})를 기록 보존에 뒀다. 문서의 수치는 run 이 없어도 남지만 같은 평가를 다시 돌릴 수는 없게 된다.")
    w(f"3. **계보 조상을 어디까지 남길 것인가** — 문서 판정은 기각이지만 남기는 run 의 웜스타트 원본이라 올린 것이 {n_lin} 개({human(b_lin)})다. 부모 체크포인트 하나만 남기고 나머지를 솎으면 대부분 줄어든다(§0 의 솎기 수치에 포함).")
    w(f"4. **솎기를 할 것인가** — {S['thin_ckpts']} 개 · {human(S['thin_bytes'])}. 폐기 후보 전체({human(S['by_class'][C_DISCARD]['bytes'])})와 합치면 `logs/rsl_rl/` 의 {100.0 * (S['thin_bytes'] + S['by_class'][C_DISCARD]['bytes']) / max(1, S['total']['bytes']):.0f} % 다.")
    w(f"5. **불명 {S['by_class'][C_UNK]['runs']} 개** — 0719·0725 평지 세대 중 문서에 역할이 안 적힌 것, G4, G24R-g-lr. 판정 기록을 남기고 폐기 후보로 내릴지, 그대로 둘지.")
    w("6. **배포 후보의 등록·명명** — §9 의 별칭과 `models/` 등록을 할지. 세 정책 모두 지금은 `logs/`(git 무시)에만 있다.")
    w(f"7. **실험 키의 처리** — exported 가 없는 채 등록만 된 키: {', '.join(exp_only) or '없음'}. 키를 남기면 그 run 은 기준선·검증 의존, 키를 내리면 기록 보존(계보·이전 후보)으로 바뀐다.")
    w(f"8. **끊어진 레지스트리 키 {len([m for m in mk if m['table'] == 'CHECKPOINTS'])} 개** — 0918 이전 세대 키(`walk`·`march*`·`stand_v*`·`stand_hang_*`·`stand_crouch_v1*`)는 run 이 로컬에 없다. 레지스트리에서 내릴지는 코드 변경이라 이 작업 범위 밖이다.")
    ev_now = ", ".join(gen_token(r) for r in runs if r["cls"] == C_EVAL) or "없음"
    w(f"9. **평가 중 후보({ev_now}) 판정 뒤 재분류** — 통과하면 걷기 정책이 바뀌므로 §1·§3·§9 가 전부 달라진다. 스크립트는 `KEW3_SPECS`·`W3_SPECS` 를 읽으므로 코드가 바뀐 뒤 다시 돌리면 따라온다. 탈락하면 분류표(`CURATED`)의 그 줄을 근거 문구와 함께 고쳐야 한다(문서에 판정 문구가 생기면 스크립트가 §0 에 경고를 띄운다).")
    w("10. **`biped12_stage4/biped12_stepb0` 고리 링크** — 쓰는 곳을 코드에서 찾지 못했다(§7.3). 남길지 지울지.")
    w("")
    # ── 12 한계
    w("## 12. 이 목록의 한계")
    w("")
    w("- **학습 태스크 id 는 run 안에 저장되지 않는다**(`params/env.yaml` 에 없다). 표의 태스크는 코드·문서·`models/*/meta.json` 으로 확인한 것과, run 이름이 `rl_walking/__init__.py` 의 등록 단계 이름과 같아서 추정한 것을 구분해 적었다.")
    w("- 문서 인용은 run 디렉터리 이름이 문서에 그대로 나올 때만 자동으로 잡힌다. 'G15w'·'W2-8' 같은 약칭으로 쓴 판정은 분류표에 사람이 옮긴 것이고(근거 문구는 매번 다시 확인), 약칭으로만 인용된 다른 문서는 세지 못한다.")
    w("- 평가 산출물 수는 `log_picture/**/*.{txt,json,md}` 머리말 6 KB 에서 run·체크포인트를 찾은 것이다. 무효 폴더(`_invalid_*`)는 뺐다. scratchpad 에만 원자료가 있는 평가(예: `REPRO.md` 의 학습 seed 재현)는 0 으로 나온다 — 그런 run 은 분류표에서 따로 처리했다.")
    w("- '실행 코드 줄이 직접 읽는다'는 판정은 줄 단위 규칙이다(주석·docstring 제외, `--load_run`·`torch.load`·경로 상수 등). 변수에 담아 넘기는 스크립트는 놓칠 수 있다 — `reports/gen_w3/scripts/*.sh` 는 손으로 한 번 더 읽어 확인했다.")
    w(f"- md5: {'모든 체크포인트' if extra['md5_mode'] == 'all' else '익스포트·params·핵심 체크포인트만'} 계산했다. 최근 3 분 안에 바뀐 파일은 건너뛰었다.")
    w("- 젯슨의 실제 파일 상태는 확인하지 않았다(접속 금지). §3 은 '이 파일들이 있어야 한다'는 요구 목록이지 '젯슨에 있다'는 확인이 아니다.")
    w("- 분류는 **문서에 적힌 판정을 옮긴 것**이고 정책 성능을 다시 잰 것이 아니다. 시뮬 판정을 실물 성공 가능성으로 읽지 않는다.")
    w("")
    return L


def scratch_refs(inv, dirs):
    """세션 scratchpad 같은 저장소 밖 폴더의 .sh/.py 가 가리키는 run(깊이 3 까지). 실행 대기열이 폐기 후보를 읽는지 보려는 것."""
    names = set(inv["by_name"])
    ts_map = collections.defaultdict(list)
    for n in names:
        ts_map[n[:19]].append(n)
    now = time.time()
    hit = collections.defaultdict(list)
    for root in dirs or []:
        root = os.path.abspath(root)
        for dp, dn, fn in os.walk(root):
            if dp[len(root):].count(os.sep) >= 3:
                dn[:] = []
            for f in fn:
                if os.path.splitext(f)[1] not in (".sh", ".py"):
                    continue
                fp = os.path.join(dp, f)
                try:
                    if os.path.getsize(fp) > 2**20:
                        continue
                    age = (now - os.path.getmtime(fp)) / 3600.0
                except OSError:
                    continue
                txt = read_text(fp)
                if "run 디렉터리 이름(타임스탬프 포함)에 fullmatch" in txt or "logs/rsl_rl 사실 수집(읽기 전용)" in txt:
                    continue                      # 이 목록을 만들던 도구 자신
                for m in set((m.group(1), m.group(1) + m.group(2)) for m in TS_RE.finditer(txt)):
                    for b in ts_map.get(m[0], []):
                        if m[1].startswith(b) or (len(m[1]) > 19 and b.startswith(m[1])):
                            hit[b].append((round(age, 1), os.path.join(os.path.basename(os.path.dirname(root)), os.path.relpath(fp, root))))
    out = []
    for b, v in sorted(hit.items()):
        v = sorted(set(v))
        out.append(dict(run=b, cls=inv["by_name"][b]["cls"], n_files=len(v), newest_age_h=v[0][0], newest_file=v[0][1]))
    return out


GLOB_RE = re.compile(r"logs/rsl_rl/([A-Za-z0-9_]+)/\*_?([A-Za-z0-9$_{}]+)")


def glob_refs(inv):
    """스크립트의 `logs/rsl_rl/<실험>/*_<꼬리>` 꼴 — run 을 이름 꼬리로 찾는 줄."""
    out = []
    for base in ("reports", "rl_walking", "scripts", "jetson", "deploy", "tests"):
        for dp, dn, fn in os.walk(os.path.join(REPO, base)):
            dn[:] = [x for x in dn if x not in SKIP_DIRS]
            for f in fn:
                if os.path.splitext(f)[1] not in (".sh", ".py"):
                    continue
                r = rel(os.path.join(dp, f))
                if r.startswith("reports/night_1007/policy_inventory"):
                    continue
                for i, line in enumerate(read_text(os.path.join(dp, f)).split("\n"), 1):
                    if "logs/rsl_rl/" not in line or line.lstrip().startswith("#"):
                        continue
                    for m in GLOB_RE.finditer(line):
                        exp, tail = m.group(1), m.group(2)
                        if "$" in tail:
                            hit = None
                        else:
                            hit = [x["name"] for x in inv["runs"] if x["exp"].endswith("/" + exp) and x["name"].endswith("_" + tail)]
                        e = dict(file=r, line=i, pattern=f"{exp}/*_{tail}", runs=hit)
                        if e not in out:
                            out.append(e)
    return out


def separation(inv):
    runs, by_name = inv["runs"], inv["by_name"]
    dis = [r for r in runs if r["cls"] == C_DISCARD]
    dn = set(r["name"] for r in dis)
    links = sorted(a for r in dis for a in r["aliases"])
    srefs, doc_lines, docs = [], 0, collections.Counter()
    for r in dis:
        for x in inv["refs"].get(r["name"], []):
            if x["kind"] in ("script", "selftest", "test", "registry") and not x["comment"]:
                srefs.append(dict(file=x["file"], line=x["line"], run=r["name"], hard=x["hard"]))
            elif x["kind"] == "doc":
                doc_lines += 1
                docs[x["file"]] += 1
    seen, uniq = set(), []
    for x in srefs:
        k = (x["file"], x["line"], x["run"])
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    import glob
    def last_run(exp):
        d = os.path.join(LOGROOT, exp)
        xs = sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))) if os.path.isdir(d) else []
        return xs[-1] if xs else None
    fb = sorted(glob.glob(os.path.join(LOGROOT, "biped12_stage4", "*", "exported", "policy.pt")))
    gl = glob_refs(inv)
    for g in gl:
        g["discard"] = [n for n in (g["runs"] or []) if n in dn]
    tars = glob.glob(os.path.join(REPO, "deploy", "stand_7차_*.tar.gz"))
    return dict(n_discard=len(dis), discard_bytes=sum(r["size_bytes"] for r in dis), links=links, script_refs=sorted(uniq, key=lambda x: (x["file"], x["line"])),
                n_doc_files=len(docs), n_doc_lines=doc_lines, doc_top=docs.most_common(10), n_eval_files=sum(r["n_eval_files"] for r in dis),
                n_child_links=sum(1 for r in dis if r["parent"] in dn), last_stepb0=last_run("biped12_stepb0"), last_stage4=last_run("biped12_stage4"),
                selftest_fallback=rel(os.path.dirname(fb[-1])) if fb else None, n_tar=len(tars), tar_bytes=sum(os.path.getsize(t) for t in tars), glob_refs=gl)


def main():
    ap = argparse.ArgumentParser(description="정책·체크포인트 전수 목록(읽기 전용)")
    ap.add_argument("--md5", choices=("key", "all"), default="key", help="key = 익스포트·params·핵심 체크포인트만(기본) · all = 모든 model_N.pt")
    ap.add_argument("--no-torch", action="store_true", help="체크포인트 차원 읽기를 끈다")
    ap.add_argument("--check", action="store_true", help="분류표 근거 문구 검사만 하고 끝낸다(파일을 쓰지 않는다)")
    ap.add_argument("--out-dir", default=HERE, help="산출물 폴더(기본: 이 스크립트 폴더)")
    ap.add_argument("--jetson-md5", default=None, help="젯슨에서 받아 둔 md5sum 목록 파일(기본: reports/night_1007/jetson/*md5*.txt 중 최신). 이 스크립트는 젯슨에 접속하지 않는다")
    ap.add_argument("--scratch", action="append", default=[], help="저장소 밖 스크립트 폴더(세션 scratchpad 등). 그 안의 .sh/.py 가 가리키는 run 을 함께 본다. 여러 번 줄 수 있다")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.check:
        bad = 0
        for c in CURATED:
            for doc, nd in c["ev"]:
                ln, _ = find_needle(doc, nd)
                if ln is None:
                    bad += 1
                    print(f"[없음] {c['pat'].pattern} ← {doc} : {nd}")
        print(f"분류표 {len(CURATED)} 규칙 · 근거 문구 {sum(len(c['ev']) for c in CURATED)} 개 중 못 찾은 것 {bad} 개")
        return 1 if bad else 0
    out_json, out_md = os.path.join(a.out_dir, "policy_inventory.json"), os.path.join(a.out_dir, "POLICY_INVENTORY.md")
    prev = None
    if os.path.isfile(out_json):
        try:
            prev = json.load(open(out_json, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = None
    inv = build(md5_mode=a.md5, use_torch=not a.no_torch, prev=prev, verbose=not a.quiet)
    import shutil
    du = shutil.disk_usage(REPO)
    extra = dict(generated=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 command="python3 reports/night_1007/policy_inventory.py" + (" --md5 all" if a.md5 == "all" else "") + "".join(f" --scratch {x}" for x in a.scratch),
                 git=git_info(), disk=dict(total=du.total, free=du.free, free_pct=100.0 * du.free / du.total), md5_mode=a.md5)
    extra["summary"] = summarize(inv)
    extra["naming"] = naming(inv)
    extra["contract"] = contract_compare(inv)
    extra["manifests"] = manifest_check(inv)
    snap = jetson_snapshot(a.jetson_md5)
    extra["jetson"] = jetson_files(inv, snap)
    extra["jetson_snapshot"] = snapshot_compare(inv, snap)
    extra["outside"] = outside_copies(inv)
    extra["separation"] = separation(inv)
    extra["separation"]["scratch_dirs"] = [os.path.abspath(x) for x in a.scratch]
    extra["separation"]["scratch_refs"] = scratch_refs(inv, a.scratch)
    lines = write_md(inv, extra, out_md)
    js = to_json(inv, extra)
    js["separation"] = extra["separation"]
    for path, text in ((out_md, "\n".join(lines) + "\n"), (out_json, json.dumps(js, ensure_ascii=False, indent=1, default=str) + "\n")):
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    S = extra["summary"]
    print(f"run {S['total']['runs']} · 체크포인트 {S['total']['ckpts']} · {human(S['total']['bytes'])}")
    for c in CLASS_ORDER:
        v = S["by_class"][c]
        print(f"  {c:12s} run {v['runs']:4d} · 체크포인트 {v['ckpts']:5d} · {human(v['bytes']):>10s}" + (f" · 솎기 {v['thin_ckpts']} 개 {human(v['thin_bytes'])}" if v["thin_ckpts"] else ""))
    print(f"쓴 파일: {rel(out_md)} · {rel(out_json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
