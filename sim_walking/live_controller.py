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
        self._rl_hh = None
        self._rl_hh_ref_set = False
        self._rl_cmd = (0.0, 0.0, 0.0)
        self._rl_ckpt = ""
        self._rl_yaw_fn = None
        self._rl_stand_until = 0.0
        self._rl_stop_at = None
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
        # 물리 스텝 이벤트 우선 (2026-07-26): 렌더 업데이트 이벤트는 물리와
        # 1:1이 보장되지 않아 RL 정책 클록이 물리 시간과 어긋남 — 7~9s 후
        # 리듬 desync 낙상 실측. 물리 스텝 구독 = run_gait의 s.step(1)과 동일
        # 하게 1스텝:1명령 보장. 실패 시 종전 업데이트 이벤트 폴백.
        try:
            from omni.physx import get_physx_interface
            self._sub = get_physx_interface().subscribe_physics_step_events(
                self._on_physics_step)
            self._sub_kind = "physics"
        except Exception:
            import omni.kit.app
            stream = omni.kit.app.get_app().get_update_event_stream()
            self._sub = stream.create_subscription_to_pop(
                self._on_update, name="sim_walking.live_controller")
            self._sub_kind = "update"

    def _on_physics_step(self, _dt):
        self._on_update(None)

    def _unsubscribe(self):
        if self._sub:
            try:
                self._sub.unsubscribe()
            except AttributeError:
                pass  # physics 구독 객체는 참조 해제로 소멸
            self._sub = None

    def _on_update(self, e):
        try:
            if self.mode == "FALLEN":
                return
            if self.mode == "HOLD":
                # 즉시정지: 목표 동결 (실물 STOP_ALL 철학 — 피드백도 미적용)
                tg = dict(self._hold_targets)
            elif self.mode == "RL" and self.gait is not None and \
                    self._rl_stop_at is not None and self.t >= self._rl_stop_at:
                # 소프트 정지 완료: 정책이 명령 0으로 1.5s 자체 정지한 뒤 인계
                # (급전환 시 회전 보행 중 낙상 실측 — 2026-07-26)
                self.gait = None
                self._rl_stop_at = None
                self.mode = "STAND"
                tg = self.fb.apply(dict(self._neutral), lat_ref=0.0)
            elif self.mode == "RL" and self.gait is not None:
                # RL 모드: 밸런스 피드백 미적용 (정책 자체 밸런스 — 중첩 금지),
                # 절대각 소프트 리밋 클램프 (조작 실수·정책 이상 방어)
                import numpy as _np
                self.t += 1.0 / FPS
                if self.t < self._rl_stand_until:
                    # 서기 워밍업 (데모 동일 — 명령 0으로 정책 자체 안정화)
                    self.gait.cmd = _np.float32([0.0, 0.0, 0.0])
                elif self._rl_hh is not None:
                    _, q = self.s.pelvis.get_world_poses()
                    yaw = self._rl_yaw_fn(
                        float(q[0][0]), float(q[0][1]),
                        float(q[0][2]), float(q[0][3]))
                    if not self._rl_hh_ref_set:
                        # 보행 시작 순간의 헤딩을 기준으로 고정
                        self._rl_hh.set_ref(yaw)
                        self._rl_hh_ref_set = True
                    wz = self._rl_hh.update(yaw)
                    self.gait.cmd = _np.float32(
                        [self._rl_cmd[0], self._rl_cmd[1], wz])
                else:
                    self.gait.cmd = _np.float32(list(self._rl_cmd))
                tg = self.gait.targets(self.t)
                tg = self._clamp_limits(tg)
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

    # ---------- RL 보행 ----------
    #: 체크포인트 별칭 → exported 디렉토리 (2026-07-26 확정).
    #: walk = R4(12994) 이식 1순위 — 물리스텝 구독 수정 후 sim2sim 15s 무낙상
    #: 재검증 통과 (렌더 이벤트 desync가 전이 실패로 위장했던 누명 해소).
    RL_CHECKPOINTS = {
        "walk": "logs/rsl_rl/biped12_stage4/2026-07-26_14-53-03/exported",
        "walk_r3": "logs/rsl_rl/biped12_stage4/2026-07-26_13-09-16/exported",
        "march": "logs/rsl_rl/biped12_stage4/2026-07-26_21-41-30/exported",
        "rough": "logs/rsl_rl/biped12_stage4/2026-07-26_13-35-55/exported",
    }

    def _clamp_limits(self, tg):
        """절대각 소프트 리밋 클램프 (joint_limits_12dof.json) — RL 모드 방어선."""
        try:
            from robot_runtime.joint_limits import clamp_targets_rad
            return clamp_targets_rad(tg, mode="soft")
        except Exception:
            return tg

    def rl_walk(self, checkpoint="walk", cmd_x=0.0, cmd_y=0.0, wz=0.0,
                heading_hold=True):
        """RL 정책 보행 시작. checkpoint: walk|march|rough 또는 exported 경로.

        march 정책은 '조용히 서기' 불가 — 정지하려면 rl_stop(일반 서기 복귀).
        시뮬 v1은 HeadingHold만 지원 (PositionHold는 실물 브리지 전용 —
        walk_scene 좌표 변환 검증 전). march는 느린 표류가 정상.
        """
        if self.mode == "FALLEN":
            return {"ok": False, "error": "FALLEN — reset 필요"}
        from rl_walking.policy_adapter_stage4 import (
            FootContacts, RLWalkPolicyStage4)
        from rl_walking.deploy.policy_runner_stage4 import (
            HeadingHold, yaw_from_quat_wxyz)
        ck = self.RL_CHECKPOINTS.get(str(checkpoint), str(checkpoint))
        if not os.path.isabs(ck):
            ck = os.path.join(REPO, ck)
        if not os.path.isdir(ck):
            return {"ok": False, "error": f"checkpoint 없음: {ck}"}
        # 자세 정돈 후 핸드오프 (rl_walk_demo_stage4와 동일 절차 — 스폰/FSM
        # 과도 상태에서 정책 인계 시 낙상: 2026-07-26 시뮬 검증 실측)
        self.s.zero(settle=60)
        contacts = FootContacts(phys_dt=1.0 / FPS)
        contacts.initialize(self.s)
        pol = RLWalkPolicyStage4(
            contacts, checkpoint=ck, cmd=(0.0, 0.0, 0.0))
        pol.attach(self.s)
        pol.reset()
        self.gait = pol
        self.t = 0.0
        self._rl_stand_until = 2.0   # 서기 2s 후 요청 명령 적용 (데모 동일)
        self._rl_stop_at = None
        self._rl_cmd = (float(cmd_x), float(cmd_y), float(wz))
        self._rl_ckpt = str(checkpoint)
        # 헤딩 폐루프: 명시 wz 명령이 없을 때만 (회전 명령과 병용 금지)
        use_hh = bool(heading_hold) and abs(float(wz)) < 1e-6
        self._rl_hh = HeadingHold(kp=1.0, wz_limit=0.6) if use_hh else None
        self._rl_hh_ref_set = False
        self._rl_yaw_fn = yaw_from_quat_wxyz
        self.mode = "RL"
        return {"ok": True, "mode": self.mode, "checkpoint": self._rl_ckpt,
                "cmd": list(self._rl_cmd), "heading_hold": use_hh}

    def rl_cmd(self, cmd_x=None, cmd_y=None, wz=None):
        """RL 보행 중 속도 명령 변경 (지정 안 한 성분은 유지)."""
        if self.mode != "RL" or self.gait is None:
            return {"ok": False, "error": "RL 모드 아님"}
        import numpy as _np
        cx, cy, cw = self._rl_cmd
        if cmd_x is not None:
            cx = float(cmd_x)
        if cmd_y is not None:
            cy = float(cmd_y)
        if wz is not None:
            cw = float(wz)
            if abs(cw) >= 1e-6:
                self._rl_hh = None    # 명시 회전 → 헤딩홀드 해제
        self._rl_cmd = (cx, cy, cw)
        if self._rl_hh is None:
            self.gait.cmd = _np.float32([cx, cy, cw])
        return {"ok": True, "cmd": [cx, cy, cw],
                "heading_hold": self._rl_hh is not None}

    def rl_stop(self):
        """RL 소프트 정지: 명령 0으로 1.5s 정책 자체 정지 → STAND 인계."""
        if self.mode != "RL" or self.gait is None:
            self.gait = None
            self.mode = "STAND"
            return {"ok": True, "mode": self.mode}
        self._rl_cmd = (0.0, 0.0, 0.0)
        self._rl_hh = None
        self._rl_stop_at = self.t + 1.5
        return {"ok": True, "mode": "RL_STOPPING",
                "note": "명령 0 정착 1.5s 후 STAND 인계"}

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
        if self.mode == "RL":
            d["rl_checkpoint"] = self._rl_ckpt
            d["rl_cmd"] = list(self._rl_cmd)
            d["heading_hold"] = self._rl_hh is not None
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
        if cmd == "rl_walk":
            return c.rl_walk(checkpoint=d.get("checkpoint", "walk"),
                             cmd_x=d.get("cmd_x", 0.0),
                             cmd_y=d.get("cmd_y", 0.0),
                             wz=d.get("wz", 0.0),
                             heading_hold=d.get("heading_hold", True))
        if cmd == "rl_cmd":
            return c.rl_cmd(cmd_x=d.get("cmd_x"), cmd_y=d.get("cmd_y"),
                            wz=d.get("wz"))
        if cmd == "rl_stop":
            return c.rl_stop()
        return {"ok": False, "error": f"unknown cmd: {cmd}"}
    except Exception as ex:
        import traceback
        return {"ok": False, "error": f"{type(ex).__name__}: {ex}",
                "trace": traceback.format_exc()[-500:]}
