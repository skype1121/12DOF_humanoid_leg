#!/usr/bin/env python3
"""새 몸 걷기 키의 정책을 바꾼다(1011): 지금 등록된 새 몸 걷기 키(W3NBJ_*)의 이름·정책 폴더·체크포인트·태스크를 새 정책으로 **치환**한다.
키를 하나 더 붙이지 않는다 — 콘솔 메뉴·브리지 합성 키 표·selftest 의 서수 단언이 '새 몸 키 하나'를 전제하기 때문(승윤님 '정책 깔끔하게 한 개').
하는 일: ① <run>_deploy<N> 형제 폴더(exported + params/env.yaml — 매니페스트 금지 규칙에 맞는 자리)를 <run>/deploy_<N> 에서 복사
        ② 저장소의 옛 키 문자열 두 개와 W3NBJ_RUN_REL·W3NBJ_MODEL·W3NBJ_TASK 를 치환.
사용: register_walk_key.py <run 이름> <N> <태스크 id> <새 태그>     예) … 2026-10-11_02-02-43_w3s3dnbjd_s1_from_16299 17798 Biped12-HangStand-W3S3dNBjD-v0 nbjd1_17798
     --dry 를 붙이면 바뀔 곳만 센다."""
import os, re, shutil, subprocess, sys
R = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
dry = "--dry" in sys.argv; a = [x for x in sys.argv[1:] if x != "--dry"]
run, n, task, tag = a
L = os.path.join(R, "logs/rsl_rl/biped12_stepb0")
src = open(os.path.join(R, "rl_walking/deploy/w3_deploy.py"), encoding="utf-8").read()
old_w3 = re.search(r'^W3NBJ_CKPT_KEY = "([A-Za-z0-9_]+)"', src, re.M).group(1)
old_rel = re.search(r'^W3NBJ_RUN_REL = "([^"]+)"', src, re.M).group(1)
old_model = re.search(r'^W3NBJ_MODEL = "([^"]+)"', src, re.M).group(1)
old_task = re.search(r'^W3NBJ_TASK = "([^"]+)"', src, re.M).group(1)
bsrc = open(os.path.join(R, "jetson/scripts/rl_bridge_node.py"), encoding="utf-8").read()
old_kw = re.search(r'^KEW3NBJ_CKPT_KEY = "([A-Za-z0-9_]+)"', bsrc, re.M).group(1)
new_w3 = f"stand_w3s3d{tag}"; new_kw = f"stand_kew3qa_g21a_g24ra_w3{tag}"
dep_src = os.path.join(L, run, f"deploy_{n}"); new_rel = f"logs/rsl_rl/biped12_stepb0/{run.rsplit('_from_', 1)[0]}_deploy{n}"
assert os.path.exists(os.path.join(dep_src, "exported", "policy.onnx")) and os.path.exists(os.path.join(dep_src, "params", "env.yaml")), dep_src
print(f"W3 키  {old_w3} → {new_w3}\n합성 키 {old_kw} → {new_kw}\n폴더   {old_rel} → {new_rel}\n모델   {old_model} → model_{n}.pt · 태스크 {old_task} → {task}")
files = subprocess.check_output(["git", "grep", "-l", "-E", f"{old_w3}|{old_kw}|{re.escape(old_rel)}", "--", "jetson", "rl_walking/deploy", "tests", "deploy/make_head_manifest.py",
                                 "reports/night_1007/jetson_parity.py"], cwd=R, text=True).split()
tot = 0
for f in files:
    p = os.path.join(R, f); s = open(p, encoding="utf-8").read(); s0 = s
    s = s.replace(old_kw, new_kw).replace(old_w3, new_w3).replace(old_rel, new_rel)
    if f.endswith("w3_deploy.py"):
        s = s.replace(f'W3NBJ_MODEL = "{old_model}"', f'W3NBJ_MODEL = "model_{n}.pt"').replace(f'W3NBJ_TASK = "{old_task}"', f'W3NBJ_TASK = "{task}"')
    c = sum(1 for x, y in zip(s0.split("\n"), s.split("\n")) if x != y); tot += c
    print(f"  {f}: {c} 줄")
    if not dry and s != s0:
        open(p, "w", encoding="utf-8").write(s)
if not dry:
    dst = os.path.join(R, new_rel)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst); shutil.copytree(os.path.join(dep_src, "exported"), os.path.join(dst, "exported")); shutil.copytree(os.path.join(dep_src, "params"), os.path.join(dst, "params"))
    print("복사:", dst, sorted(os.listdir(os.path.join(dst, "exported"))))
print(f"{'바뀔' if dry else '바꾼'} 줄 합 {tot}")
