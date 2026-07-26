#!/usr/bin/env python3
"""Stage4(245차원) 정책 sim2sim 재생 — walk_scene.usd 60Hz (원커맨드).

rl_walk_demo.py(45차원판)의 Stage4판. Isaac Sim(GUI)+MCP 확장 실행 중일 때:
    python3 scripts/rl_walk_demo_stage4.py                  # 2s 서기 + 20s 전진 0.5m/s
    python3 scripts/rl_walk_demo_stage4.py --checkpoint <exported 경로>
    python3 scripts/rl_walk_demo_stage4.py --video

관측 조립은 실물 이식 경로와 동일한 PolicyRunnerStage4(이력·클록 내장)를 쓰고,
접촉 2ch은 PhysX net contact force(>5N)로 조회한다 — rl_walking/policy_adapter_stage4.py.

주의 (sim2sim 플랜트 차이 — 이 재생 자체가 전이 테스트):
- 학습: Isaac Lab 200Hz 물리·armature 포함·자산 10.49kg / walk_scene: 60Hz·무armature·
  구자산 9.94kg. 게인만 kp150/kd5로 강제 정합.
- 정책틱은 누적 스케줄로 평균 50Hz (6프레임당 5틱, 지터 ±13ms).
시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
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


def send(sock, command_type, params=None, timeout=900.0):
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


def remote_code(duration, stand_s, cmd_x, video, frames_dir, checkpoint, tag,
                heading_hold=False, hh_kp=1.0):
    cap = (f"capture_every=15, capture_dir={frames_dir!r}," if video else "")
    ck = "None" if checkpoint is None else repr(checkpoint)
    return f"""
import sys, importlib, shutil, json
for pkg in ("sim_walking", "rl_walking"):
    shutil.rmtree({REPO!r}+"/"+pkg+"/__pycache__", ignore_errors=True)
    shutil.rmtree({REPO!r}+"/"+pkg+"/deploy/__pycache__", ignore_errors=True)
for k in [k for k in sys.modules if k.startswith(("sim_walking","rl_walking"))]:
    del sys.modules[k]
importlib.invalidate_caches()
sys.path.insert(0, {REPO!r})

# 접촉 리포트 API는 물리 파싱 전에 붙어야 함 — 타임라인 정지 후 뷰 생성 → attach
import omni.timeline, omni.kit.app
tl = omni.timeline.get_timeline_interface(); tl.stop()
app = omni.kit.app.get_app()
for _ in range(8): app.update()

import sim_walking.sim_session as SS, sim_walking.runner as RN
from rl_walking.policy_adapter_stage4 import FootContacts, RLWalkPolicyStage4
from rl_walking.deploy.policy_runner_stage4 import HeadingHold, yaw_from_quat_wxyz

# 물리 dt 실측 (walk_scene 기본 60Hz — 접촉 임펄스→힘 환산에 사용)
import omni.usd
from pxr import UsdPhysics, PhysxSchema
stage = omni.usd.get_context().get_stage()
tsps = 60.0
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.Scene):
        attr = prim.GetAttribute("physxScene:timeStepsPerSecond")
        if attr and attr.Get():
            tsps = float(attr.Get())
        break

contacts = FootContacts(phys_dt=1.0/tsps)
s = SS.Session.attach(fresh_physics=True, kp=150.0, kd=5.0)
s.zero(settle=60)
contacts.initialize(s)   # 직립 양발하중에서 자가검증 (실패 시 휴리스틱 폴백)

s.make_cam("/World/SessCamS4", (2.3, 1.1, 0.85), (0.0, -0.35, 0.30))
import omni.kit.viewport.utility as vpu
vpu.get_active_viewport().camera_path = "/World/SessCamS4"

pol = RLWalkPolicyStage4(contacts, checkpoint={ck}, cmd=(0.0, 0.0, 0.0))
pol.attach(s); pol.reset()

STAND_S = {stand_s}
CMD_X = {cmd_x}
HH_ON = {heading_hold}

class TrackCam:
    # 정책 호출에 편승: 명령 스케줄(서기→전진) + 카메라 골반 추적
    def __init__(self, s, pol, every=0.15):
        self.s, self.pol, self.every, self._next = s, pol, every, 0.0
        self.hh, self.hh_ref_set = HeadingHold(kp={hh_kp}), False
    def targets(self, t):
        import numpy as _np
        if t < STAND_S:
            self.pol.cmd = _np.float32([0.0, 0.0, 0.0])
        else:
            wz = 0.0
            if HH_ON:
                _, q = self.s.pelvis.get_world_poses()
                yaw = yaw_from_quat_wxyz(float(q[0][0]), float(q[0][1]),
                                         float(q[0][2]), float(q[0][3]))
                if not self.hh_ref_set:
                    # 보행 시작 순간의 헤딩을 기준으로 고정
                    self.hh.set_ref(yaw); self.hh_ref_set = True
                wz = self.hh.update(yaw)
            self.pol.cmd = _np.float32([CMD_X, 0.0, wz])
        if t >= self._next:
            self._next = t + self.every
            p, _ = self.s.pelvis.get_world_poses()
            px, py = float(p[0][0]), float(p[0][1])
            self.s.make_cam("/World/SessCamS4", (px + 2.0, py - 1.5, 0.95),
                            (px, py, 0.35))
        return self.pol.targets(t)
    def lat_ref(self, t):
        return self.pol.lat_ref(t)
    def reset(self):
        self.pol.reset()

m = RN.run_gait(s, TrackCam(s, pol), duration_s=STAND_S + {duration},
                tag={tag!r}, balance=False, {cap})
m.update(pol.stats())
m["phys_hz"] = tsps
m["heading_hold"] = HH_ON
m["standing_forces_n"] = getattr(contacts, "standing_forces_n", None)
print("METRICS_JSON:" + json.dumps(m))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0, help="전진 구간 길이 [s]")
    ap.add_argument("--stand", type=float, default=2.0, help="선행 서기 구간 [s]")
    ap.add_argument("--cmd-x", type=float, default=0.5, help="전진 속도 명령 [m/s]")
    ap.add_argument("--checkpoint", default=None,
                    help="exported/ 경로 (기본: Stage4 v2 평지 2026-07-26_04-07-10)")
    ap.add_argument("--tag", default="rl_s4v2")
    ap.add_argument("--heading-hold", action="store_true",
                    help="상위 헤딩 폐루프 켜기 (IMU 요→wz 명령, 드리프트 상쇄)")
    ap.add_argument("--hh-kp", type=float, default=1.0, help="헤딩 폐루프 P 이득")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=os.path.join(REPO, "demo_output"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, f"{args.tag}_frames")
    if args.video:
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.makedirs(frames_dir, exist_ok=True)

    print(f"[1/4] Isaac Sim 접속 {HOST}:{PORT} ...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((HOST, PORT))
    except OSError as e:
        print("접속 실패:", e)
        print("Isaac Sim을 확장과 함께 실행하세요 (RL보행_명령어모음.txt 0번)")
        return 1

    info = send(sock, "scene.get_info")
    print("[2/4] 씬 확인")
    if "walk_scene.usd" not in json.dumps(info):
        print("      walk_scene.usd 로 전환 중...")
        send(sock, "simulation.execute_script", {"code":
            f"import omni.usd; omni.usd.get_context().open_stage({SCENE!r})"})
        time.sleep(3)

    print(f"[3/4] Stage4 재생 (서기 {args.stand:.0f}s + 전진 {args.duration:.0f}s, "
          f"cmd_x={args.cmd_x}) ...")
    t0 = time.time()
    resp = send(sock, "simulation.execute_script",
                {"code": remote_code(args.duration, args.stand, args.cmd_x,
                                     args.video, frames_dir, args.checkpoint,
                                     args.tag, args.heading_hold, args.hh_kp)})
    raw = json.dumps(resp)
    mm = re.search(r"METRICS_JSON:(\{.*?\})(?:\\n|\")", raw)
    if not mm:
        print("지표 파싱 실패. 원시 응답:", raw[:3000])
        return 1
    metrics = json.loads(mm.group(1).replace('\\"', '"'))
    print(f"      완료 ({time.time()-t0:.0f}s 소요)")

    print("[4/4] 결과")
    for k in ("steps_total", "steps_contact_L", "steps_contact_R",
              "forward_progress_m", "lateral_drift_m", "final_yaw", "fell_at",
              "max_abs_lean_ap", "duration_run", "policy_ticks",
              "contact_duty_L", "contact_duty_R", "contact_mode", "backend",
              "phys_hz", "heading_hold"):
        print(f"   {k:20s} = {metrics.get(k)}")

    with open(os.path.join(args.out, f"{args.tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1, ensure_ascii=False)

    if args.video and shutil.which("ffmpeg"):
        pngs = sorted(glob.glob(os.path.join(frames_dir, f"{args.tag}_*.png")))
        if pngs:
            mp4 = os.path.join(args.out, f"{args.tag}_walk.mp4")
            subprocess.run(["ffmpeg", "-y", "-framerate", "8", "-pattern_type",
                            "glob", "-i", os.path.join(frames_dir, f"{args.tag}_*.png"),
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
