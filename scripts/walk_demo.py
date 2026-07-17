#!/usr/bin/env python3
"""12DOF 하체 보행 데모 — 원커맨드 실행기.

사용법:
    python3 scripts/walk_demo.py                # 기본 16초 데모 (판정 출력)
    python3 scripts/walk_demo.py --duration 24  # 더 길게
    python3 scripts/walk_demo.py --video        # 프레임 캡처 + mp4 생성

전제: Isaac Sim 5.1(GUI)이 isaac.sim.mcp_extension과 함께 실행 중
      (localhost:8766). 씬은 자동으로 walk_scene.usd 로 전환된다.

시뮬 전용 데모다 — 실물 모터/CAN/Jetson에는 아무것도 보내지 않는다.
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


def send(sock, command_type, params=None, timeout=300.0):
    """확장 TCP 프로토콜: JSON 요청 → JSON 응답(완성될 때까지 수신)."""
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


def remote_code(duration, video, frames_dir, num_steps=None, speed=1.0):
    gait = json.load(open(os.path.join(REPO, "config", "sim_walk_params.json")))
    gp = dict(gait["gait"])
    if speed != 1.0:  # 속도: 위상 시간 스케일 (안정성은 1.0에서 검증됨)
        gp["lean_dur"] /= speed
        gp["step_dur"] /= speed
    cap = (f"capture_every=15, capture_dir={frames_dir!r}," if video else "")
    ns = "None" if num_steps is None else str(int(num_steps))
    return f"""
import sys, importlib, shutil, json
shutil.rmtree({REPO!r}+"/sim_walking/__pycache__", ignore_errors=True)
for k in [k for k in sys.modules if k.startswith("sim_walking")]: del sys.modules[k]
importlib.invalidate_caches()
sys.path.insert(0, {REPO!r})
import sim_walking.sim_session as SS, sim_walking.gait_static as GS, sim_walking.runner as RN
s = SS.Session.attach(fresh_physics=True)
s.zero(settle=100)
s.make_cam("/World/SessCamWalk", (2.3, 1.1, 0.85), (0.0, -0.35, 0.30))
import omni.kit.viewport.utility as vpu
vpu.get_active_viewport().camera_path = "/World/SessCamWalk"
g = GS.StaticGait(GS.StaticGaitParams(**{json.dumps(gp)}), num_steps={ns})
dur = {duration} if g._t_stop() is None else g._t_stop() + g.stop_dur + 3.0
fb = {json.dumps(gait["feedback"])}
m = RN.run_gait(s, g, duration_s=dur, tag="demo", {cap}
                bal_kp=fb["bal_kp"], bal_ki=fb["bal_ki"], bal_clamp=fb["bal_clamp"],
                lat_kp=fb["lat_kp"], lat_kd=fb["lat_kd"], lat_clamp=fb["lat_clamp"],
                yaw_kp=fb["yaw_kp"], yaw_clamp=fb["yaw_clamp"])
print("METRICS_JSON:" + json.dumps(m))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=16.0,
                    help="시뮬 시간[s] (--steps 지정 시 무시)")
    ap.add_argument("--steps", type=int, default=None,
                    help="이 스텝 수만큼 걷고 우아하게 정지")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="보행 속도 배율 (1.0에서 검증됨; 1.2 정도까지 시도 가능)")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "demo_output"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "frames")
    if args.video:
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.makedirs(frames_dir, exist_ok=True)

    print(f"[1/4] Isaac Sim 접속 {HOST}:{PORT} ...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
    except OSError as e:
        print("접속 실패:", e)
        print("Isaac Sim을 확장과 함께 실행하세요:")
        print("  ISAACSIM_ROOT=/home/ryu/isaacsim/isaac-sim-5.1.0 "
              "/home/ryu/isaacsim-mcp-server/scripts/run_isaac_sim.sh")
        return 1

    info = send(sock, "scene.get_info")
    stage = json.dumps(info)
    print("[2/4] 현재 스테이지 확인")
    if "walk_scene.usd" not in stage:
        print("      walk_scene.usd 로 전환 중...")
        send(sock, "simulation.execute_script", {"code":
            f"import omni.usd; omni.usd.get_context().open_stage({SCENE!r})"})
        time.sleep(3)

    print(f"[3/4] 보행 데모 실행 ({args.duration:.0f}s 시뮬 — 실제 수십 초 소요)...")
    t0 = time.time()
    resp = send(sock, "simulation.execute_script",
                {"code": remote_code(args.duration, args.video, frames_dir,
                                     num_steps=args.steps, speed=args.speed)})
    raw = json.dumps(resp)
    mm = re.search(r"METRICS_JSON:(\{.*?\})(?:\\n|\")", raw)
    if not mm:
        print("지표 파싱 실패. 원시 응답:", raw[:2000])
        return 1
    metrics = json.loads(mm.group(1).replace('\\"', '"'))
    print(f"      완료 ({time.time()-t0:.0f}s 소요)")

    with open(os.path.join(args.out, "last_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, ensure_ascii=False)

    print("[4/4] 결과")
    for k in ("steps_total", "steps_left", "steps_right", "forward_progress_m",
              "lateral_drift_m", "final_yaw", "fell_at", "max_abs_lean_ap",
              "duration_run"):
        print(f"   {k:20s} = {metrics.get(k)}")

    if args.video and shutil.which("ffmpeg"):
        pngs = sorted(glob.glob(os.path.join(frames_dir, "demo_*.png")))
        if pngs:
            mp4 = os.path.join(args.out, "walk_demo.mp4")
            subprocess.run(["ffmpeg", "-y", "-framerate", "8", "-pattern_type",
                            "glob", "-i", os.path.join(frames_dir, "demo_*.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-vf", "scale=1280:-2", mp4],
                           capture_output=True)
            print(f"   video: {mp4} ({len(pngs)} frames)")

    ok = (metrics.get("fell_at") is None
          and metrics.get("steps_total", 0) >= 3
          and metrics.get("forward_progress_m", 0) >= 0.20)
    print()
    print("=== 판정:", "PASS ✅ (3보 이상 전진, 무낙상)" if ok else "FAIL ❌", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
