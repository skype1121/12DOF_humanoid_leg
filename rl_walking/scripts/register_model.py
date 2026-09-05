"""학습 모델 레지스트리 등록 — models/<name>/ 에 policy.pt·policy.onnx·env.yaml·agent.yaml 복사 + meta.json(md5).

logs/ 는 gitignore 라 체크포인트·익스포트가 저장소에 없다. 세대 데뷔 후보의 재현 단위(정책 + 파라미터)를
버전관리하기 위해 세대당 ~2.3MB 만 복사한다. 실행:
  python3 rl_walking/scripts/register_model.py stand_v17 --run 2026-09-04_21-48-32 --ckpt model_126997.pt \
      --task Biped12-Velocity-Stage4V2-Stand-V17-v0 --usd rl_walking/assets/usd_final3/biped12.usd \
      --warm "v16 model_120998" --summary "V6고정 32.8/10.9/7.8% · 명목 0/64 · 발요차 2.7°" [--legacy]
"""
import argparse
import hashlib
import json
import os
import shutil

REPO = "/home/ryu/humanoid_leg_test1"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name"); ap.add_argument("--run", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", default=""); ap.add_argument("--usd", default=""); ap.add_argument("--warm", default="")
    ap.add_argument("--summary", default=""); ap.add_argument("--legacy", action="store_true")
    ap.add_argument("--experiment", default="biped12_stage4")
    a = ap.parse_args()
    src = os.path.join(REPO, "logs", "rsl_rl", a.experiment, a.run)
    dst = os.path.join(REPO, "models", "legacy" if a.legacy else "", a.name).replace("//", "/")
    os.makedirs(dst, exist_ok=True)
    copied = {}
    for rel in ("exported/policy.pt", "exported/policy.onnx", "params/env.yaml", "params/agent.yaml"):
        s = os.path.join(src, rel)
        if os.path.isfile(s):
            d = os.path.join(dst, os.path.basename(rel)); shutil.copy2(s, d); copied[os.path.basename(rel)] = md5(d)
    ck = os.path.join(src, a.ckpt)
    meta = {"name": a.name, "run": a.run, "checkpoint": a.ckpt, "checkpoint_md5": md5(ck) if os.path.isfile(ck) else None,
            "task": a.task, "usd": a.usd, "warm_start": a.warm, "summary": a.summary, "legacy": a.legacy,
            "files": copied, "source_dir": os.path.relpath(src, REPO)}
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), ensure_ascii=False, indent=2)
    print(dst, json.dumps(copied, indent=1))


if __name__ == "__main__":
    main()
