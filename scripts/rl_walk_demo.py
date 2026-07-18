#!/usr/bin/env python3
"""RL 정책 보행 데모 — walk_scene.usd 에서 학습된 정책 재생 (원커맨드).

walk_demo.py의 RL판. Isaac Sim(GUI)+MCP 확장 실행 중일 때:
    python3 scripts/rl_walk_demo.py                 # 20초 전진 0.5m/s
    python3 scripts/rl_walk_demo.py --cmd-x 0.4 --duration 30
    python3 scripts/rl_walk_demo.py --video         # 프레임 캡처 + mp4

주의:
- 정책은 Isaac Lab(200Hz 물리, kp150/kd5, armature 포함)에서 학습됨.
  walk_scene은 60Hz 물리라 완전 동일 조건은 아님 — 게인은 kp150/kd5로
  강제 정합하지만, 이 재생 자체가 사실상 sim-to-sim 전이 테스트다.
- 시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
import argparse
import glob
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST, PORT = "localhost", int(os.environ.get("ISAAC_MCP_PORT", 8766))
SCENE = os.path.join(REPO, "isaacsim_scene", "walk_scene.usd")


def send(sock, command_type, params=None, timeout=600.0):
    payload = json.dumps({"type": command_type, "params": params or {}})
    sock.sendall(payload.encode())
    sock.settimeout(timeout)
    chunks = []
    while True:
        chunk = sock.recv(1 << 16)
        if not chunk:
            break
        chunks.append(chunk)
        try:
            return json.loads(b"".join(chunks).decode())
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Isaac 응답 수신 실패")


def remote_code(duration, cmd_x, video, frames_dir, checkpoint):
    cap = (f"capture_every=15, capture_dir={frames_dir!r}," if video else "")
    ck = "None" if checkpoint is None else repr(checkpoint)
    return f"""
import sys, importlib, shutil, json
for pkg in ("sim_walking", "rl_walking"):
    shutil.rmtree({REPO!r}+"/"+pkg+"/__pycache__", ignore_errors=True)
for k in [k for k in sys.modules if k.startswith(("sim_walking","rl_walking"))]:
    del sys.modules[k]
importlib.invalidate_caches()
sys.path.insert(0, {REPO!r})
import sim_walking.sim_session as SS, sim_walking.runner as RN
from rl_walking.policy_adapter import RLWalkPolicy
# 게인을 학습 조건(kp150/kd5)으로 정합 — 이게 없으면 다른 플랜트에서 재생하는 것
s = SS.Session.attach(fresh_physics=True, kp=150.0, kd=5.0)
s.zero(settle=60)
s.make_cam("/World/SessCamRL", (2.3, 1.1, 0.85), (0.0, -0.35, 0.30))
import omni.kit.viewport.utility as vpu
vpu.get_active_viewport().camera_path = "/World/SessCamRL"
pol = RLWalkPolicy(checkpoint={ck}, cmd=({cmd_x}, 0.0, 0.0))
pol.attach(s); pol.reset()

class TrackCam:
    # 정책 호출에 편승해 카메라가 골반을 추적 (전진 시 화면 이탈 방지)
    def __init__(self, s, pol, every=0.15):
        self.s, self.pol, self.every, self._next = s, pol, every, 0.0
    def targets(self, t):
        if t >= self._next:
            self._next = t + self.every
            p, _ = self.s.pelvis.get_world_poses()
            px, py = float(p[0][0]), float(p[0][1])
            self.s.make_cam("/World/SessCamRL", (px + 2.0, py - 1.5, 0.95),
                            (px, py, 0.35))
        return self.pol.targets(t)
    def lat_ref(self, t):
        return self.pol.lat_ref(t)
    def reset(self):
        self.pol.reset()

m = RN.run_gait(s, TrackCam(s, pol), duration_s={duration}, tag="rl_demo", balance=False, {cap})
print("METRICS_JSON:" + json.dumps(m))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--cmd-x", type=float, default=0.5, help="전진 속도 명령 [m/s]")
    ap.add_argument("--checkpoint", default=None, help="exported/policy.pt 경로 (기본: 최신)")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "demo_output"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "rl_frames")
    if args.video:
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.makedirs(frames_dir, exist_ok=True)

    print(f"[1/4] Isaac Sim 접속 {HOST}:{PORT} ...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
    except OSError as e:
        print("접속 실패:", e)
        print("Isaac Sim을 확장과 함께 실행하세요 (보행데모_명령어모음.txt 0번)")
        return 1

    info = send(sock, "scene.get_info")
    print("[2/4] 씬 확인")
    if "walk_scene.usd" not in json.dumps(info):
        print("      walk_scene.usd 로 전환 중...")
        send(sock, "simulation.execute_script", {"code":
            f"import omni.usd; omni.usd.get_context().open_stage({SCENE!r})"})
        time.sleep(3)

    print(f"[3/4] RL 보행 재생 ({args.duration:.0f}s, cmd_x={args.cmd_x}) ...")
    t0 = time.time()
    resp = send(sock, "simulation.execute_script",
                {"code": remote_code(args.duration, args.cmd_x, args.video,
                                     frames_dir, args.checkpoint)})
    raw = json.dumps(resp)
    mm = re.search(r"METRICS_JSON:(\{.*?\})(?:\\n|\")", raw)
    if not mm:
        print("지표 파싱 실패. 원시 응답:", raw[:2000])
        return 1
    metrics = json.loads(mm.group(1).replace('\\"', '"'))
    print(f"      완료 ({time.time()-t0:.0f}s 소요)")

    print("[4/4] 결과")
    for k in ("steps_total", "forward_progress_m", "lateral_drift_m",
              "final_yaw", "fell_at", "max_abs_lean_ap", "duration_run"):
        print(f"   {k:20s} = {metrics.get(k)}")

    with open(os.path.join(args.out, "rl_last_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, ensure_ascii=False)

    if args.video and shutil.which("ffmpeg"):
        pngs = sorted(glob.glob(os.path.join(frames_dir, "rl_demo_*.png")))
        if pngs:
            mp4 = os.path.join(args.out, "rl_walk_demo.mp4")
            subprocess.run(["ffmpeg", "-y", "-framerate", "8", "-pattern_type",
                            "glob", "-i", os.path.join(frames_dir, "rl_demo_*.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-vf", "scale=1280:-2", mp4],
                           capture_output=True)
            print(f"   video: {mp4} ({len(pngs)} frames)")

    ok = (metrics.get("fell_at") is None
          and metrics.get("forward_progress_m", 0) >= 0.5 * args.cmd_x * args.duration * 0.5)
    print()
    print("=== 판정:", "PASS (전진, 무낙상)" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
