"""Isaac 상주 걷기 컨트롤러 — UI/TCP 단문 명령용 (논블로킹).

runner.run_gait(블로킹 루프)와 달리, Kit 업데이트 콜백에 붙어 매 프레임
관절 목표를 갱신한다. 명령은 즉시 반환되므로 걷는 중에도 외란 주입·정지
요청이 가능하다.

사용 (execute_script에서):
    import sys; sys.path.insert(0, "/home/ryu/humanoid_leg_test1")
    from sim_walking import live_controller as LC
    LC.command({"cmd": "stand"})          # 초기화+스탠딩 (밸런스 유지)
    LC.command({"cmd": "walk"})           # 연속 걷기
    LC.command({"cmd": "walk", "steps": 4})  # 4보 걷고 발모아 서기→스탠딩
    LC.command({"cmd": "stop_walk"})      # 걷는 중 우아한 정지
    LC.command({"cmd": "push", "fx": 20, "fy": 0, "dur": 0.15})
    LC.command({"cmd": "status"})
    LC.command({"cmd": "reset"})          # 넘어졌을 때: 씬 리셋+재초기화
    LC.command({"cmd": "shutdown"})

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
import json
import os
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 60.0

_ctrl = None


class LiveWalkController:
    def __init__(self):
        from .sim_session import Session
        from .feedback import BalanceFeedback
        cfg = json.load(open(os.path.join(REPO, "config", "sim_walk_params.json")))
        self.gait_cfg = cfg["gait"]
        self.fb_cfg = cfg["feedback"]
        self.s = Session.attach(fresh_physics=True)
        self.fb = BalanceFeedback(self.s, **self.fb_cfg)
        self.mode = "STAND"          # STAND | WALK | HOLD | FALLEN
        self._hold_targets = None    # HOLD(즉시정지) 시점 목표 동결
        self.gait = None
        self.t = 0.0                 # WALK 모드 보행 시각
        self.push_left = 0           # 남은 외란 프레임
        self.push_vec = np.zeros(3)
        self.err_count = 0
        self.last_err = ""
        self.y0 = self._pelvis_y()
        self.steps_planned = None
        self._neutral = self._make_neutral()
        self._sub = None
        self._subscribe()

    # ---------- 내부 ----------
    def _make_neutral(self):
        from .gait_static import StaticGait, StaticGaitParams
        return StaticGait(StaticGaitParams(**self.gait_cfg))._neutral_targets()

    def _pelvis_y(self):
        p, _ = self.s.pelvis.get_world_poses()
        return float(p[0][1])

    def _subscribe(self):
        import omni.kit.app
        stream = omni.kit.app.get_app().get_update_event_stream()
        self._sub = stream.create_subscription_to_pop(
            self._on_update, name="sim_walking.live_controller")

    def _unsubscribe(self):
        if self._sub:
            self._sub.unsubscribe()
            self._sub = None

    def _on_update(self, e):
        try:
            if self.mode == "FALLEN":
                return
            if self.mode == "HOLD":
                # 즉시정지: 목표 동결 (실물 STOP_ALL 철학 — 피드백도 미적용)
                tg = dict(self._hold_targets)
            elif self.mode == "WALK" and self.gait is not None:
                self.t += 1.0 / FPS
                tg = self.gait.targets(self.t)
                ref = self.gait.lat_ref(self.t)
                ts = self.gait._t_stop()
                if ts is not None and self.t > ts + self.gait.stop_dur + 0.3:
                    self.mode = "STAND"     # 정지 완료 → 스탠딩 유지로 전환
                    self.gait = None
                tg = self.fb.apply(tg, lat_ref=ref)
            else:
                tg = self.fb.apply(dict(self._neutral), lat_ref=0.0)
            if self.push_left > 0:
                self.push_left -= 1
                try:
                    self.s.pelvis.apply_forces(
                        np.array([self.push_vec], dtype=float), is_global=True)
                except Exception:
                    pass
            self.s.cmd(named=tg)
            if self.s.fallen():
                self.mode = "FALLEN"
        except Exception as ex:  # 콜백 예외 스팸 방지
            self.err_count += 1
            self.last_err = f"{type(ex).__name__}: {ex}"
            if self.err_count > 30:
                self._unsubscribe()

    # ---------- 명령 ----------
    def stand(self):
        if self.mode == "WALK" and self.gait is not None:
            return self.stop_walk()
        self.mode = "STAND"
        return {"ok": True, "mode": self.mode}

    def walk(self, steps=None, speed=1.0):
        from .gait_static import StaticGait, StaticGaitParams
        if self.mode == "FALLEN":
            return {"ok": False, "error": "FALLEN — reset 필요"}
        gp = dict(self.gait_cfg)
        if speed and speed != 1.0:
            speed = max(0.5, min(1.1, float(speed)))  # 검증 범위 클램프
            gp["lean_dur"] /= speed
            gp["step_dur"] /= speed
        self.gait = StaticGait(StaticGaitParams(**gp),
                               num_steps=int(steps) if steps else None)
        self.t = 0.0
        self.steps_planned = steps
        self.mode = "WALK"
        return {"ok": True, "mode": self.mode, "steps": steps, "speed": speed}

    def stop_walk(self):
        if self.mode != "WALK" or self.gait is None:
            return {"ok": True, "mode": self.mode, "note": "걷는 중 아님"}
        ts = self.gait.request_stop(self.t)
        return {"ok": True, "mode": self.mode, "stop_at_t": round(ts, 2),
                "in": round(ts - self.t, 2)}

    def halt(self):
        """즉시 정지: 현재 관절 목표를 동결(HOLD). 복귀는 stand/walk/reset."""
        if self.mode == "FALLEN":
            return {"ok": False, "error": "FALLEN — reset 필요"}
        self._hold_targets = {n: float(v) for n, v in
                              zip(self.s.names, self.s._last_target)}
        self.gait = None
        self.mode = "HOLD"
        return {"ok": True, "mode": self.mode, "note": "목표 동결(피드백 미적용)"}

    def push(self, fx=0.0, fy=0.0, fz=0.0, dur=0.15):
        fx = float(np.clip(fx, -60, 60))
        fy = float(np.clip(fy, -60, 60))
        self.push_vec = np.array([fx, fy, float(np.clip(fz, -30, 30))])
        self.push_left = max(1, int(float(dur) * FPS))
        return {"ok": True, "force": [fx, fy, fz], "frames": self.push_left}

    def _alive(self):
        """물리 뷰와 업데이트 구독이 둘 다 살아있는지 확인.

        타임라인 Stop/씬 재오픈으로 Physics Simulation View가 사라지면
        get_joint_positions()가 None을 돌려주고, _on_update가 매 프레임
        실패해 err_count>30에서 스스로 unsubscribe 한다(영구 정지).
        """
        if self._sub is None:
            return False
        try:
            return self.s.art.get_joint_positions() is not None
        except Exception:
            return False

    def status(self):
        recovered = False
        if not self._alive():
            # 죽은 컨트롤러를 그대로 두면 UI가 영영 응답을 못 받는다.
            # Isaac 안에 캐시된 _ctrl은 UI를 재시작해도 되살아나지 않으므로
            # 여기서 한 번 자동으로 재attach 한다.
            try:
                self.reset()
                recovered = True
            except Exception as ex:
                return {"ok": False, "mode": "DEAD",
                        "error": f"{type(ex).__name__}: {ex}",
                        "hint": "walk_scene.usd 가 열려 있는지 확인하고 "
                                "Isaac Sim에서 ▶Play 를 누르세요."}
        te = self.s.telemetry()
        ap, lat = te["lean_ap"], te["lean_lat"]
        d = {
            "ok": True, "mode": self.mode, "walk_t": round(self.t, 2),
            "forward_m": round(self.y0 - te["pelvis_y"], 3),
            "lean_ap": round(ap, 1), "lean_lat": round(lat, 1),
            "yaw": round(te["yaw"], 1), "pelvis_z": round(te["pelvis_z"], 3),
            "err_count": self.err_count,
        }
        if self.last_err:
            d["last_err"] = self.last_err
        if self.mode == "WALK" and self.gait is not None:
            ts = self.gait._t_stop()
            d["stopping"] = ts is not None
        if recovered:
            d["recovered"] = True
        return d

    def reset(self):
        """씬 정지→재생으로 로봇을 초기 자세로 복원 후 재초기화."""
        from .sim_session import Session
        from .feedback import BalanceFeedback
        self._unsubscribe()
        self.s = Session.attach(fresh_physics=True)
        self.fb = BalanceFeedback(self.s, **self.fb_cfg)
        self.mode = "STAND"
        self.gait = None
        self.t = 0.0
        self.push_left = 0
        self.err_count = 0
        self.last_err = ""
        self.y0 = self._pelvis_y()
        self._subscribe()
        return {"ok": True, "mode": self.mode, "note": "scene reset"}

    def shutdown(self):
        self._unsubscribe()
        return {"ok": True, "note": "controller detached"}


def get():
    global _ctrl
    if _ctrl is None:
        _ctrl = LiveWalkController()
    return _ctrl


def command(d):
    """UI가 execute_script로 호출하는 단일 진입점. dict 반환."""
    global _ctrl
    try:
        cmd = d.get("cmd", "status")
        if cmd == "shutdown":
            if _ctrl is not None:
                r = _ctrl.shutdown()
                _ctrl = None
                return r
            return {"ok": True, "note": "not running"}
        c = get()
        if cmd == "stand":
            return c.stand()
        if cmd == "walk":
            return c.walk(steps=d.get("steps"), speed=d.get("speed", 1.0))
        if cmd == "stop_walk":
            return c.stop_walk()
        if cmd == "halt":
            return c.halt()
        if cmd == "push":
            return c.push(fx=d.get("fx", 0.0), fy=d.get("fy", 0.0),
                          fz=d.get("fz", 0.0), dur=d.get("dur", 0.15))
        if cmd == "status":
            return c.status()
        if cmd == "reset":
            return c.reset()
        return {"ok": False, "error": f"unknown cmd: {cmd}"}
    except Exception as ex:
        import traceback
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}",
                "trace": traceback.format_exc()[-500:]}
