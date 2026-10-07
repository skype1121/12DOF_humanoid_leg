#!/usr/bin/env python3
"""젯슨 배포 코드 대 로컬(시뮬 폐루프로 검증한 코드) 러너 동치 확인 — 같은 입력열 → 같은 목표각·모드인지.

배경: 로컬 평가(배포 코드 폐루프)는 torch(policy.pt) 백엔드로 돌았고, 젯슨에는 torch 가 없어 onnxruntime(policy.onnx)으로 돈다.
이 스크립트는 브리지의 `_make_runner(키)` 로 만든 **실제 배포 러너**에 같은 입력열(자이로·투영 중력·명령·관절각·관절속도·접촉 + 걷기 켜기/끄기 사건)을 넣어
틱마다 낸 목표각(rad, 12)·원시 행동·모드를 비교한다. 관측 조립·이력·필터·직전 행동·클록 위상·감지기·합성 전환·추론 백엔드가 전부 이 경로에 있다.
입력열은 합성(개루프)이다 — 계약·수치 동치 확인이지 폐루프 성능 증명이 아니다(CLAUDE.md 6).

사용:
  로컬:  CUDA_VISIBLE_DEVICES="" python3 reports/night_1007/jetson_parity.py --gen /tmp/parity_ref.npz [--keys k1,k2]
  젯슨:  python3 reports/night_1007/jetson_parity.py --check parity_ref.npz [--tol 5e-4]
종료 코드: 0 전 키 통과 · 1 불일치 · 2 실행 불가(키·파일 없음)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "jetson", "scripts"))
T = 600
# 1008: 둘째 합성 키(첫 합성 키 + KE 감지기 갇힘 해소)를 **맨 뒤에** 더했다 — 앞의 5 키·순서·입력열은 그대로(키마다 독립 입력열·독립 러너라 기존 키의 기준값은 바뀌지 않는다).
# --check 는 기준 파일에 들어 있는 키만 본다: 5 키 기준 파일이면 5 키만 대조하고, 새 키를 대조하려면 6 키로 --gen 한 기준 파일이 필요하다.
KEYS = ["stand_crouch_g21a", "stand_c5kr_g21a_g24ra", "stand_c5krh_g21a_g24ra", "stand_w3s3d_15800", "stand_kew3_g21a_g24ra_w3s3d15800",
        "stand_kew3q_g21a_g24ra_w3s3d15800"]
WALK_EVENTS = [(100, True, (0.06, 0.0, 0.0)), (300, False, (0.0, 0.0, 0.0)), (420, True, (0.0, 0.0, 0.15)), (520, False, (0.0, 0.0, 0.0))]


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_bridge():
    spec = importlib.util.spec_from_file_location("_br_parity", os.path.join(REPO, "jetson", "scripts", "rl_bridge_node.py"))
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    return b


def make_inputs(pose, seed=7):
    """합성 입력열(결정적): 기본자세 근방의 관절 흔들림 + 두 번의 기울기 펄스(회복 모드 유발) + 펄스 중 한발 들림."""
    rng = np.random.default_rng(seed)
    k = np.arange(T)
    pitch = np.radians(1.0 + 0.3 * np.sin(2 * np.pi * k / 180.0)); roll = np.radians(0.4 * np.sin(2 * np.pi * k / 230.0))
    for (k0, amp, dur) in ((200, 8.0, 30), (360, -7.0, 25)):
        w = (k >= k0) & (k < k0 + dur); pitch = pitch + np.radians(amp) * np.where(w, np.sin(np.pi * (k - k0) / dur), 0.0)
    grav = np.stack([np.sin(pitch), -np.sin(roll) * np.cos(pitch), -np.cos(roll) * np.cos(pitch)], 1).astype(np.float32)
    gyro = (0.02 * rng.standard_normal((T, 3))).astype(np.float32); gyro[:, 1] += np.gradient(pitch) / 0.02; gyro[:, 0] += np.gradient(roll) / 0.02
    q = (np.asarray(pose, np.float32)[None] + 0.02 * np.sin(2 * np.pi * k[:, None] / (90.0 + 7.0 * np.arange(12)[None])) + 0.002 * rng.standard_normal((T, 12))).astype(np.float32)
    qv = (np.gradient(q, axis=0) / 0.02).astype(np.float32)
    ct = np.ones((T, 2), np.float32); ct[205:215, 0] = 0.0; ct[365:372, 1] = 0.0
    return gyro.astype(np.float32), grav, q, qv, ct


def kew3_keys(b):
    """브리지의 합성 키 전부(1008: KEW3_KEYS — 키가 하나뿐이던 판은 KEW3_CKPT_KEY, 합성 러너가 없는 판은 빈 튜플)."""
    ks = getattr(b, "KEW3_KEYS", None)
    if ks is None:
        one = getattr(b, "KEW3_CKPT_KEY", None)
        ks = () if one is None else (one,)
    return tuple(ks)


def run_key(b, key, inp=None):
    if key in getattr(b._W3, "W3_KEYS", ()) or key in kew3_keys(b):
        if not b.EXPERIMENTAL_W3:
            b.enable_experimental_w3()
    if key in kew3_keys(b) and not getattr(b, "EXPERIMENTAL_KEW3", False):
        b.enable_experimental_kew3()
    r = b._make_runner(key)
    pose = b.CKPT_DEFAULT_POSE.get(key, None)
    if pose is None:
        pose = getattr(r, "default_pose", None)
    if inp is None:
        inp = make_inputs(pose)
    gyro, grav, q, qv, ct = inp
    walking = hasattr(r, "set_gait_enable") and (key in getattr(b._W3, "W3_KEYS", ()) or key in kew3_keys(b))
    r.reset()
    out = np.zeros((T, 12), np.float64); raw = np.zeros((T, 12), np.float32); mode = np.zeros(T, np.int16); cmd = (0.0, 0.0, 0.0); ev = {k0: (w, c) for k0, w, c in WALK_EVENTS}
    for k in range(T):
        if walking and k in ev:
            w, c = ev[k]
            if w and getattr(r, "enable_block_reason", lambda: None)() is not None:
                pass                                   # 합성 러너: KE 회복 모드 중이면 켜지 않음(브리지와 같은 게이트) — 양쪽에서 같은 판단이어야 함
            else:
                r.set_gait_enable(bool(w)); cmd = tuple(c)
        o = r.step(gyro[k], grav[k], cmd, q[k], qv[k], ct[k])
        out[k] = np.asarray(o, np.float64).reshape(-1)[:12]
        la = getattr(r, "last_step_action", None)
        if la is None:
            la = getattr(getattr(r, "core", r), "last_action", None)
        raw[k] = np.asarray(la, np.float32).reshape(-1)[:12] if la is not None else 0.0
        m = getattr(r, "last_step_kind", None)
        if m is None:
            m = getattr(r, "last_mode", 0)
        clk = getattr(r, "_w3_clock", None)
        mode[k] = int(m) + (10 if (clk is not None and bool(clk.ge)) else 0)
    backend = str(getattr(r, "backend", "?"))
    return dict(inp=inp, out=out, raw=raw, mode=mode, backend=backend, pose=np.asarray(pose, np.float32))


def files_of(b, key):
    """키가 읽는 정책 파일 md5(있는 것만)."""
    res = {}
    keys = [key]
    if key in kew3_keys(b):
        sp = b.KEW3_SPECS[key]; keys = [sp.w3, sp.ke]
    for kk in keys:
        dirs = []
        if kk in b.C5_PAIRS:
            stand, rec = b.C5_PAIRS[kk]; dirs += [b.CHECKPOINTS[stand], rec]
        elif kk in b.CHECKPOINTS:
            dirs.append(b.CHECKPOINTS[kk])
        for d in dirs:
            for fn in ("policy.onnx", "policy.pt"):
                p = os.path.join(d, fn)
                if os.path.isfile(p):
                    res[os.path.relpath(p, REPO)] = _md5(p)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gen"); ap.add_argument("--check"); ap.add_argument("--keys", default=",".join(KEYS)); ap.add_argument("--tol", type=float, default=5e-4)
    a = ap.parse_args()
    b = load_bridge(); keys = [k for k in a.keys.split(",") if k]
    if a.gen:
        D = {}
        for key in keys:
            R = run_key(b, key); D[key] = R
            print(f"[gen] {key}: 백엔드 {R['backend']} · 목표각 범위 {np.degrees(R['out'].min()):+.1f}~{np.degrees(R['out'].max()):+.1f}° · 모드 값 {sorted(set(R['mode'].tolist()))} · 파일 {files_of(b, key)}")
        np.savez_compressed(a.gen, keys=np.array(keys), **{f"{k}__{n}": v for k, R in D.items() for n, v in (("gyro", R["inp"][0]), ("grav", R["inp"][1]), ("q", R["inp"][2]), ("qv", R["inp"][3]), ("ct", R["inp"][4]), ("out", R["out"]), ("raw", R["raw"]), ("mode", R["mode"]), ("backend", np.array(R["backend"])))},
                            **{f"{k}__files": np.array(repr(files_of(b, k))) for k in keys})
        print(f"[gen] 기준 저장: {a.gen}"); return 0
    if not a.check or not os.path.isfile(a.check):
        print("기준 파일 없음(--check)"); return 2
    z = np.load(a.check, allow_pickle=True); bad = 0
    for key in [str(k) for k in z["keys"]]:
        if key not in keys:
            continue
        inp = tuple(z[f"{key}__{n}"] for n in ("gyro", "grav", "q", "qv", "ct"))
        try:
            R = run_key(b, key, inp)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {key}: 실행 불가 {type(e).__name__}: {e}"); bad += 1; continue
        ref_out, ref_raw, ref_mode = z[f"{key}__out"], z[f"{key}__raw"], z[f"{key}__mode"]
        d_out = float(np.abs(R["out"] - ref_out).max()); d_raw = float(np.abs(R["raw"] - ref_raw).max()); nm = int((R["mode"] != ref_mode).sum())
        first = int(np.argmax(np.abs(R["out"] - ref_out).max(1) > a.tol)) if d_out > a.tol else -1
        ok = d_out <= a.tol and nm == 0
        fl_ref = eval(str(z[f"{key}__files"])); fl = files_of(b, key)   # noqa: S307 — 우리가 쓴 dict repr
        onnx_same = all(fl.get(p) == m for p, m in fl_ref.items() if p.endswith(".onnx") and p in fl)
        print(f"[{'PASS' if ok else 'FAIL'}] {key}: 백엔드 기준 {str(z[f'{key}__backend'])} / 여기 {R['backend']} · 목표각 최대 차 {d_out:.2e} rad ({math.degrees(d_out):.4f}°) · 원시 행동 최대 차 {d_raw:.2e} · 모드 불일치 틱 {nm}"
              + (f" · 첫 초과 틱 {first}" if first >= 0 else "") + f" · onnx md5 {'같음' if onnx_same else '다름/없음'}")
        bad += (not ok)
    print("jetson parity: " + ("ALL PASS" if bad == 0 else f"{bad} FAILED")); return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
