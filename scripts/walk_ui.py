#!/usr/bin/env python3
"""걷기 전용 UI (Main UI와 별개) — Isaac Sim 보행 데모 리모컨.

사용법:  python3 scripts/walk_ui.py
전제:    Isaac Sim(GUI) + isaac.sim.mcp_extension 실행 중 (localhost:8766)

기능: 스탠딩 / 연속 전진 / N보 걷고 서기 / 걷다가 서기 /
      좌우앞뒤 외란(세기 조절) / 넘어짐 리셋 / 실시간 상태 표시
시뮬 전용 — 실물 모터/CAN/Jetson에는 아무것도 보내지 않는다.
"""
import json
import os
import queue
import socket
import threading
import tkinter as tk
from tkinter import ttk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST, PORT = "localhost", int(os.environ.get("ISAAC_MCP_PORT", 8766))

# 월드 좌표 규약 (joint_direction 실측): 정면=-Y, 왼쪽=+X
DIRS = {"앞": (0, -1), "뒤": (0, +1), "왼쪽": (+1, 0), "오른쪽": (-1, 0)}


class IsaacLink:
    """확장 TCP로 live_controller 명령 전송 (스레드 안전)."""

    def __init__(self):
        self.sock = None
        self.lock = threading.Lock()

    def _connect(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(60.0)
        s.connect((HOST, PORT))
        self.sock = s

    def _send_raw(self, command_type, params):
        payload = json.dumps({"type": command_type, "params": params}).encode()
        self.sock.sendall(payload)
        chunks = []
        while True:
            chunk = self.sock.recv(1 << 16)
            if not chunk:
                raise ConnectionError("connection closed")
            chunks.append(chunk)
            try:
                return json.loads(b"".join(chunks).decode())
            except json.JSONDecodeError:
                continue

    def lc(self, cmd: dict):
        """live_controller.command(cmd) 실행 → dict."""
        code = (
            "import sys, json\n"
            f"sys.path.insert(0, {REPO!r})\n"
            "from sim_walking import live_controller as LC\n"
            f"print('RESP:' + json.dumps(LC.command({cmd!r})))\n"
        )
        with self.lock:
            for attempt in (1, 2):
                try:
                    if self.sock is None:
                        self._connect()
                    resp = self._send_raw("simulation.execute_script",
                                          {"code": code})
                    break
                except (OSError, ConnectionError) as e:
                    self.sock = None
                    if attempt == 2:
                        return {"ok": False, "error": f"연결 실패: {e}"}
        raw = json.dumps(resp)
        i = raw.find("RESP:")
        if i < 0:
            return {"ok": False, "error": f"응답 파싱 실패: {raw[:200]}"}
        j = raw.find("\\n", i)
        frag = raw[i + 5: j if j > 0 else None]
        try:
            return json.loads(frag.replace('\\"', '"'))
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"JSON 오류: {e}: {frag[:200]}"}


class WalkUI:
    POLL_MS = 700

    def __init__(self, root):
        self.root = root
        self.link = IsaacLink()
        self.q = queue.Queue()
        self.busy = False
        root.title("12DOF 보행 데모 리모컨 (시뮬 전용)")
        root.geometry("560x640")
        self._build()
        self._poll_queue()
        self._poll_status()

    # ---------- 레이아웃 ----------
    def _build(self):
        pad = dict(padx=6, pady=4)

        st = ttk.LabelFrame(self.root, text="상태")
        st.pack(fill="x", **pad)
        self.mode_var = tk.StringVar(value="― 미연결 ―")
        self.mode_lbl = tk.Label(st, textvariable=self.mode_var,
                                 font=("", 22, "bold"))
        self.mode_lbl.pack(pady=2)
        self.stat_var = tk.StringVar(value="스탠딩 시작을 눌러 초기화하세요")
        tk.Label(st, textvariable=self.stat_var, font=("", 11)).pack(pady=2)

        mo = ttk.LabelFrame(self.root, text="동작")
        mo.pack(fill="x", **pad)
        row1 = tk.Frame(mo); row1.pack(fill="x", pady=3)
        self.btn_stand = tk.Button(row1, text="● 스탠딩 시작/유지", height=2,
                                   command=lambda: self.run({"cmd": "stand"}))
        self.btn_stand.pack(side="left", expand=True, fill="x", padx=4)
        self.btn_walk = tk.Button(row1, text="▶ 계속 걷기", height=2,
                                  command=self._walk_cont)
        self.btn_walk.pack(side="left", expand=True, fill="x", padx=4)
        self.btn_stopw = tk.Button(row1, text="✋ 걷다가 서기", height=2,
                                   command=lambda: self.run({"cmd": "stop_walk"}))
        self.btn_stopw.pack(side="left", expand=True, fill="x", padx=4)

        row2 = tk.Frame(mo); row2.pack(fill="x", pady=3)
        tk.Label(row2, text="스텝 수:").pack(side="left", padx=(8, 2))
        self.steps_var = tk.IntVar(value=4)
        tk.Spinbox(row2, from_=1, to=30, width=4,
                   textvariable=self.steps_var).pack(side="left")
        tk.Button(row2, text="→ N보 걷고 서기", height=1,
                  command=self._walk_n).pack(side="left", expand=True,
                                             fill="x", padx=6)
        tk.Label(row2, text="속도:").pack(side="left", padx=(8, 2))
        self.speed_var = tk.DoubleVar(value=1.0)
        tk.Scale(row2, from_=0.8, to=1.1, resolution=0.05, orient="horizontal",
                 variable=self.speed_var, length=110).pack(side="left", padx=4)

        pu = ttk.LabelFrame(self.root, text="외란 (골반에 0.15s 힘)")
        pu.pack(fill="x", **pad)
        prow = tk.Frame(pu); prow.pack(pady=3)
        tk.Label(prow, text="세기 [N]:").grid(row=1, column=0, padx=6)
        self.force_var = tk.IntVar(value=15)
        tk.Scale(prow, from_=5, to=40, orient="horizontal",
                 variable=self.force_var, length=140).grid(row=1, column=1)
        tk.Button(prow, text="↑ 앞", width=8,
                  command=lambda: self._push("앞")).grid(row=0, column=3, padx=3)
        tk.Button(prow, text="↓ 뒤", width=8,
                  command=lambda: self._push("뒤")).grid(row=2, column=3, padx=3)
        tk.Button(prow, text="← 왼쪽", width=8,
                  command=lambda: self._push("왼쪽")).grid(row=1, column=2, padx=3)
        tk.Button(prow, text="→ 오른쪽", width=8,
                  command=lambda: self._push("오른쪽")).grid(row=1, column=4, padx=3)

        misc = tk.Frame(self.root); misc.pack(fill="x", **pad)
        tk.Button(misc, text="♻ 리셋 (넘어졌을 때)", fg="#b91c1c",
                  command=lambda: self.run({"cmd": "reset"})).pack(
            side="left", expand=True, fill="x", padx=4)

        lg = ttk.LabelFrame(self.root, text="로그")
        lg.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(lg, height=8, font=("monospace", 9))
        self.log.pack(fill="both", expand=True)

    # ---------- 동작 ----------
    def _walk_cont(self):
        self.run({"cmd": "walk", "speed": float(self.speed_var.get())})

    def _walk_n(self):
        self.run({"cmd": "walk", "steps": int(self.steps_var.get()),
                  "speed": float(self.speed_var.get())})

    def _push(self, name):
        dx, dy = DIRS[name]
        f = int(self.force_var.get())
        self.run({"cmd": "push", "fx": dx * f, "fy": dy * f, "dur": 0.15},
                 note=f"외란 {name} {f}N")

    def run(self, cmd, note=None):
        """명령을 워커 스레드로 전송 (UI 프리즈 방지)."""
        label = note or cmd.get("cmd")
        self._log(f"> {label}")
        threading.Thread(target=lambda: self.q.put((label, self.link.lc(cmd))),
                         daemon=True).start()

    def _poll_queue(self):
        """워커 스레드 결과를 메인 스레드에서만 처리 (tk는 스레드 불안전)."""
        try:
            while True:
                label, resp = self.q.get_nowait()
                if label == "__status__":
                    self._show_status(resp)
                    continue
                if resp.get("ok"):
                    show = {k: v for k, v in resp.items() if k != "ok"}
                    self._log(f"  ✓ {label}: {json.dumps(show, ensure_ascii=False)}")
                else:
                    self._log(f"  ✗ {label}: {resp.get('error')}")
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _poll_status(self):
        threading.Thread(
            target=lambda: self.q.put(("__status__",
                                       self.link.lc({"cmd": "status"}))),
            daemon=True).start()
        self.root.after(self.POLL_MS, self._poll_status)

    def _show_status(self, st):
        if not st.get("ok"):
            self.mode_var.set("― 미연결 ―")
            self.mode_lbl.config(fg="gray")
            return
        mode = st.get("mode", "?")
        colors = {"STAND": "#16a34a", "WALK": "#2563eb", "FALLEN": "#dc2626"}
        icon = {"STAND": "● 스탠딩", "WALK": "▶ 보행 중",
                "FALLEN": "⚠ 넘어짐! 리셋 필요"}
        self.mode_var.set(icon.get(mode, mode))
        self.mode_lbl.config(fg=colors.get(mode, "black"))
        self.stat_var.set(
            f"전진 {st.get('forward_m', 0):+.2f} m   "
            f"기울기 전후 {st.get('lean_ap', 0):+.1f}° / 좌우 {st.get('lean_lat', 0):+.1f}°   "
            f"방향 {st.get('yaw', 0):+.1f}°   골반 {st.get('pelvis_z', 0):.3f} m")

    def _log(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")


if __name__ == "__main__":
    root = tk.Tk()
    WalkUI(root)
    root.mainloop()
