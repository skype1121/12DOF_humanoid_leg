"""KE ↔ W3 합성 러너 (2026-10-07, claude2) — "ge 0 인 동안은 KE, 걷는 동안만 W3". 설계: reports/gen_w3/KEW3_DESIGN.md.

**실험 모듈 · 브리지 미연결 · 젯슨 미배포 · 실물 미검증.** 근거 = 시뮬 전환 평가(reports/gen_w3/TRANSITION_W3KE.md §5·§1c).

    KE(G21a 서기 + G24Ra 회복 + 감지기, C5Runner)  ⇄  W3(PolicyRunnerStage4 w3=…, 걷기·회전)
       step 켜기(want True)  : **그 틱에** W3 로.
       stop(want False)      : W3 가 반주기 경계에서 ge 0 으로 래치한 뒤(settle_quiet_s > 0 이면 W3 STAND 로 정착을 기다린 뒤) KE.
    전환 방식 handoff:
      "carry"(기본) — 관측 이력·관측 필터(EMA)·last_action·슬루 기준을 이어받는다(클록 이력만 받는 쪽 계약 값). 감지기는 새로 시작.
                      시뮬 direct 팔과 같다. 근거: 러너를 새로 시작하면 joint_vel_ema(α 0.1)가 순간 관절속도로 초기화돼 KE 정책이 본 적 없는
                      입력이 된다 — 걷기 직후 reset 전환은 행동 폭주로 낙상 13 %(TRANSITION_W3KE.md §5.6, 절제 대조).
      "fresh"       — 받는 러너 reset()(이력 첫 틱 백필·last_action 0·EMA 재시작) + 슬루 기준만 이어받기. 시뮬 reset 팔. 조용한 서기에서만 안전.
    어느 방식이든 슬루 기준(직전 방출 목표)은 이어받는다: reset 된 러너는 첫 틱에 실측 qpos 를 슬루 기준으로 쓰는데, 실물은 명령–엔코더
    추종 오차가 수 도라 그대로 두면 전환 틱에 목표가 끌려간다(시뮬은 체인이 연속이었다).

브리지가 읽는 러너 인터페이스(step·reset·set_gait_enable·_w3_clock·last_obs·last_action·blackbox·needs_knee_tau …)를 그대로 낸다.
노드의 LPF·가드·워밍업·블렌딩 상태는 건드리지 않는다(전환은 러너 안의 일).

검증: python3 rl_walking/deploy/kew3_runner.py --selftest   (numpy 만, 스텁 정책)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

MODE_KE, MODE_W3 = "ke", "w3"


def _core(r):
    """C5Runner 는 관측·슬루 상태를 .core(PolicyRunnerStage4)에 둔다 — 단일 러너는 자기 자신."""
    return r.__dict__.get("core", r) if hasattr(r, "__dict__") else r


class KeW3Runner:
    """W3 세션의 '서기 정책'을 KE 로 바꾼 합성 러너."""

    def __init__(self, w3_runner, ke_runner, *, handoff: str = "carry", settle_quiet_s: float = 0.0, settle_w: float = 0.15,
                 settle_max_s: float = 3.0, dt: float = 0.02):
        """handoff: "carry" | "fresh". settle_quiet_s: 0 = 래치 다음 틱에 KE / > 0 = 래치 뒤 W3 STAND 로 (양발 접촉 ∧ |ω_xy| < settle_w)가
        이 시간 이어지면 KE(상한 settle_max_s — 시뮬 전환 평가의 정착 판정과 같은 정의: 0.5 s·0.15 rad/s·상한 3 s)."""
        if handoff not in ("carry", "fresh"):
            raise ValueError(f"handoff 는 'carry' | 'fresh' (받음 {handoff!r})")
        self.handoff = handoff
        self.settle_quiet_s, self.settle_w, self.settle_max_s, self.dt = float(settle_quiet_s), float(settle_w), float(settle_max_s), float(dt)
        self._settle_run = 0.0; self._since_latch = None; self._w3_arm_want = False
        cw, ck = _core(w3_runner), _core(ke_runner)
        if getattr(cw, "_w3_clock", None) is None:
            raise ValueError("w3_runner 가 W3 러너가 아님(_w3_clock 없음)")
        if getattr(ck, "_w3_clock", None) is not None:
            raise ValueError("ke_runner 가 W3 러너임 — KE(C5Runner) 를 넘길 것")
        if cw.obs_dim != ck.obs_dim or list(cw.terms) != list(ck.terms):
            raise ValueError(f"관측 레이아웃 다름: {cw.obs_dim} vs {ck.obs_dim}")
        if not np.array_equal(np.asarray(cw.default_pose), np.asarray(ck.default_pose)):
            raise ValueError("기본자세 다름 — 관측 joint_pos·행동 오프셋 기준이 바뀐다")
        if (cw.act_obs_ema_beta, cw.jvel_obs_ema_beta) != (ck.act_obs_ema_beta, ck.jvel_obs_ema_beta):
            raise ValueError("관측 필터 β 다름")
        if getattr(cw, "jvel_ema_mode", None) != getattr(ck, "jvel_ema_mode", None):
            raise ValueError("jvel_ema_mode 다름")
        # carry 는 KE 코어의 틱을 0 으로 되돌린다(클록 위상 0 — 시뮬 direct 팔의 클록 패치와 같음). ema_fresh_ticks 2(parity 옵션)는 틱 < 2 에서
        # gyro_ema·joint_vel_ema 를 순간값으로 재시작하므로, 그 조합은 '이어받기'가 조용히 '재시작'이 된다(X2 폭주 경로) → 기동 거부.
        if handoff == "carry" and (int(getattr(cw, "ema_fresh_ticks", 1)) != 1 or int(getattr(ck, "ema_fresh_ticks", 1)) != 1):
            raise ValueError("handoff 'carry' 는 ema_fresh_ticks 1 에서만 — 2 는 전환 틱에 저역 필터가 순간값으로 재시작된다(TRANSITION_W3KE.md §5.6)")
        self.w3r, self.ke = w3_runner, ke_runner
        self.mode = MODE_KE
        self._want = False
        self._force_ke = False
        self.n_to_w3 = 0
        self.n_to_ke = 0
        self.last_event = ""          # 이번 틱 전환 사건: "" | "to_w3" | "to_ke" | "force_ke"
        self.ticks_in_mode = 0

    # ---------- 브리지가 읽는 속성 ----------
    @property
    def _active(self):
        return self.ke if self.mode == MODE_KE else self.w3r

    def __getattr__(self, name):
        """last_obs·last_action·_ema·last_jvel_ema·obs_dim·backend·act_obs_ema_beta … 는 지금 도는 러너에 위임."""
        d = self.__dict__
        if "w3r" not in d or "ke" not in d:
            raise AttributeError(name)
        return getattr(d["ke"] if d.get("mode", MODE_KE) == MODE_KE else d["w3r"], name)

    @property
    def _w3_clock(self):
        """W3 클록은 항상 W3 러너의 것 — 감독기·헤딩 고리·블랙박스가 ge·u·want 를 읽는다(KE 모드에서는 ge 0)."""
        return _core(self.w3r)._w3_clock

    @property
    def w3(self):
        return _core(self.w3r).w3

    @property
    def det(self):
        return self.ke.det

    @property
    def last_mode(self):
        """KE 회복 모드(0/1) — W3 모드에서는 0."""
        return int(self.ke.last_mode) if self.mode == MODE_KE else 0

    @property
    def needs_knee_tau(self):
        return bool(getattr(self.ke, "needs_knee_tau", False))

    def enable_block_reason(self):
        """켜기 추가 선행조건: KE 회복 모드가 켜져 있으면 거부(사유 문자열), 아니면 None. 감독기 판정 위에 얹는다."""
        if self.mode == MODE_KE and int(self.ke.last_mode) == 1:
            return "KE 회복 모드 켜짐 — 서기 복귀 뒤에 켤 것"
        return None

    def blackbox(self) -> dict:
        bb = dict(self.ke.blackbox()) if hasattr(self.ke, "blackbox") else {}
        bb["kew3"] = {"mode": self.mode, "event": self.last_event, "n_to_w3": int(self.n_to_w3), "n_to_ke": int(self.n_to_ke),
                      "ticks": int(self.ticks_in_mode), "want": bool(self._want)}
        return bb

    # ---------- 상태 ----------
    def reset(self) -> None:
        """세션 시작(stand) — 두 러너 초기화, KE 로 시작."""
        self.w3r.reset()
        self.ke.reset()
        self.mode = MODE_KE
        self._want = False
        self._force_ke = False
        self.last_event = ""
        self.ticks_in_mode = 0
        self._settle_run = 0.0; self._since_latch = None; self._w3_arm_want = False

    @staticmethod
    def _carry(src, dst, clock_frame, tick: int) -> None:
        """src → dst(둘 다 PolicyRunnerStage4 코어): 이력·EMA·last_action·슬루 기준 복사, 클록 이력은 dst 계약 값, 틱 지정."""
        for name, h in src._hist.items():
            d = dst._hist[name]
            d._buf = h._buf.copy()
            d._pushes = h._pushes
        dst._hist["gait_clock"]._buf[:] = np.asarray(clock_frame, dtype=np.float32)
        dst._ema = {k: v.copy() for k, v in src._ema.items()}
        dst.last_action = src.last_action.copy()
        dst.last_jvel_ema = None if src.last_jvel_ema is None else src.last_jvel_ema.copy()
        dst._prev_emitted = None if src._prev_emitted is None else np.array(src._prev_emitted, dtype=np.float32, copy=True)
        dst._tick = int(tick)

    def set_gait_enable(self, want: bool) -> None:
        """명령이 원하는 gait_enable. 켜기 = 다음 step 에서 W3 로 전환(같은 틱에 ge 켜짐), 끄기 = W3 래치가 반주기 경계에서 내린다."""
        self._want = bool(want)
        if self.mode == MODE_W3:
            self.w3r.set_gait_enable(self._want)
        else:
            self._w3_clock.set_want(self._want)        # 감독기 스냅샷 일관(KE 모드의 W3 러너는 돌지 않는다)

    def force_ke(self) -> None:
        """(실험·기본 미사용) 래치를 기다리지 않고 다음 step 부터 KE — 걷는 중 외란 인계(Q6) 자리. 감지기가 검증되기 전에는 호출하지 않는다."""
        self._want = False
        if self.mode == MODE_W3:
            self._force_ke = True

    def _to_w3(self) -> None:
        ck, cw = _core(self.ke), _core(self.w3r)
        prev = ck._prev_emitted
        self.w3r.reset()
        if self.handoff == "carry":
            # 이력 이어받기 + 클록 이력 (0,0)(W3 STAND 계약). 틱 ≥ 2 = fresh 아님 → 첫 W3 틱은 STAND(클록 0), 다음 틱에 0→1 래치(게이트 0.2 s 램프).
            self._carry(ck, cw, (0.0, 0.0), max(2, int(ck._tick)))
            self._w3_arm_want = True
        else:
            cw._prev_emitted = None if prev is None else np.array(prev, dtype=np.float32, copy=True)
            self.w3r.set_gait_enable(True)                  # fresh(tick 0): ge·게이트 즉시
        self._settle_run = 0.0; self._since_latch = None
        self.mode = MODE_W3
        self.n_to_w3 += 1
        self.ticks_in_mode = 0
        self.last_event = "to_w3"

    def _to_ke(self, event: str = "to_ke") -> None:
        ck, cw = _core(self.ke), _core(self.w3r)
        prev = cw._prev_emitted
        self.ke.reset()                                 # 감지기 새로(기준 자세 ref_init_s 추종), 코어 초기화
        if self.handoff == "carry":
            self._carry(cw, ck, (0.0, 1.0), 0)          # KE 클록 = 위상 0 부터(이력도 위상 0 프레임) — 시뮬 direct 팔의 클록 패치와 같다
        else:
            ck._prev_emitted = None if prev is None else np.array(prev, dtype=np.float32, copy=True)
        self.w3r.reset()                                # 클록 ge 0·want 0
        self._settle_run = 0.0; self._since_latch = None; self._w3_arm_want = False
        self.mode = MODE_KE
        self._force_ke = False
        self.n_to_ke += 1
        self.ticks_in_mode = 0
        self.last_event = event

    # ---------- 스텝 ----------
    def step(self, gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s, contact2, tau_knee_nm=None) -> np.ndarray:
        self.last_event = ""
        if self.mode == MODE_W3 and self._force_ke:
            self._to_ke("force_ke")
        if self.mode == MODE_KE and self._want:
            self._to_w3()
        if self.mode == MODE_W3:
            out = self.w3r.step(gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s, contact2)
            self.ticks_in_mode += 1
            if self._w3_arm_want:                       # carry: 첫 W3 틱(STAND) 뒤에 켠다 — 시뮬 direct 즉시 켜기와 같은 수열
                self._w3_arm_want = False
                if self._want:
                    self.w3r.set_gait_enable(True)
                return out
            if (not self._want) and (not bool(self._w3_clock.ge)):
                if self.settle_quiet_s <= 0.0:
                    self._to_ke()                       # 래치 꺼짐 — 이 틱까지 W3, 다음 틱부터 KE
                    self.last_event = "to_ke"
                else:                                   # W3 STAND 로 정착 대기(양발 ∧ |ω_xy| < settle_w 가 settle_quiet_s, 상한 settle_max_s)
                    g = np.asarray(gyro_rad_s, dtype=np.float64).reshape(-1); c = np.asarray(contact2, dtype=np.float64).reshape(-1)
                    quiet = bool(c[0] > 0.5 and c[1] > 0.5 and float(np.hypot(g[0], g[1])) < self.settle_w)
                    self._since_latch = 0.0 if self._since_latch is None else self._since_latch + self.dt
                    self._settle_run = self._settle_run + self.dt if quiet else 0.0
                    if self._settle_run >= self.settle_quiet_s - 1e-9 or self._since_latch >= self.settle_max_s - 1e-9:
                        self._to_ke()
                        self.last_event = "to_ke"
            else:
                self._settle_run = 0.0; self._since_latch = None
            return out
        zero = np.zeros(3, dtype=np.float32)            # KE 는 명령 (0,0,0) 로만 학습(회복 cue 는 러너 안에서)
        if getattr(self.ke, "needs_knee_tau", False) or tau_knee_nm is not None:
            out = self.ke.step(gyro_rad_s, gravity_unit, zero, qpos_rad, qvel_rad_s, contact2, tau_knee_nm=tau_knee_nm)
        else:
            out = self.ke.step(gyro_rad_s, gravity_unit, zero, qpos_rad, qvel_rad_s, contact2)
        self.ticks_in_mode += 1
        return out


# ---------------------------------------------------------------------------
# 자가검증 (numpy, 스텁 정책)
# ---------------------------------------------------------------------------
def _mk(seed=0):
    from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4
    from rl_walking.deploy.c5_switch import C5Runner, SwitchDetector
    from rl_walking.deploy import w3_deploy as W3
    rng = np.random.default_rng(seed)
    Ww = rng.normal(0, 0.05, (12, 320)).astype(np.float32); Ws = rng.normal(0, 0.05, (12, 320)).astype(np.float32); Wr = rng.normal(0, 0.05, (12, 320)).astype(np.float32)
    f = lambda W: (lambda o: np.tanh(W @ np.asarray(o, np.float32)).astype(np.float32))
    pose = (rng.normal(0, 0.2, 12)).astype(np.float32)
    kw = dict(obs_dim=320, gait_freq=1.1, default_pose_rad=pose, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
    det_kw = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5, off_contact="any",
                  early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33, early_gate=(5.0, 1.0))
    mk_w3 = lambda: PolicyRunnerStage4(model_path=f(Ww), w3=W3.W3S3D_CONFIG, **kw)
    mk_ke = lambda **extra: C5Runner(f(Ws), f(Wr), detector=SwitchDetector(**det_kw), cue_vx=0.03, cue_s=1.0, cue_dir_imu=True, **kw, **extra)
    return mk_w3, mk_ke, pose, kw


def _inp(k, pose, w=0.0, tilt=0.0, q_off=0.0):
    rng = np.random.default_rng(1000 + k)
    th = np.radians(tilt)
    return dict(gyro_rad_s=np.array([0.01 * np.sin(k), w, 0.0], np.float32), gravity_unit=np.array([np.sin(th), 0.0, -np.cos(th)], np.float32),
                qpos_rad=(pose + q_off + rng.normal(0, 0.01, 12)).astype(np.float32), qvel_rad_s=rng.normal(0, 0.05, 12).astype(np.float32),
                contact2=np.array([1.0, 1.0], np.float32))


def selftest() -> int:
    from rl_walking.deploy.policy_runner import SLEW_RAD
    mk_w3, mk_ke, pose, kw = _mk()
    cmd0 = np.zeros(3, np.float32); cmdf = np.array([0.06, 0.0, 0.0], np.float32)
    R = KeW3Runner(mk_w3(), mk_ke(), handoff="fresh"); R.reset()
    # [1] KE 모드 = 단독 KE 러너와 비트 동일
    ref_ke = mk_ke(); ref_ke.reset(); last = None
    for k in range(60):
        a = R.step(cmd=cmd0, **_inp(k, pose)); b = ref_ke.step(cmd=cmd0, **_inp(k, pose))
        assert R.mode == MODE_KE and np.array_equal(a, b), f"KE 모드 불일치 틱 {k}"
        last = a
    assert not R._w3_clock.ge and R.enable_block_reason() is None and R.blackbox()["kew3"]["mode"] == "ke"
    # [2] 켜기: 같은 틱에 W3, fresh 클록(ge 즉시·게이트 1), 이력 백필, 슬루 기준 이어받기(실측 qpos 가 멀어도 목표는 연속)
    R.set_gait_enable(True)
    ref_w3 = mk_w3(); ref_w3.reset(); ref_w3._prev_emitted = np.array(last, np.float32, copy=True); ref_w3.set_gait_enable(True)
    far = 0.5                                                # 실측 qpos 를 직전 목표에서 0.5 rad(28.6°) 떨어뜨림 — 슬루 기준이 qpos 면 목표가 크게 튄다
    a = R.step(cmd=cmdf, **_inp(60, pose, q_off=far)); b = ref_w3.step(cmd=cmdf, **_inp(60, pose, q_off=far))
    assert R.mode == MODE_W3 and R.last_event == "to_w3" and np.array_equal(a, b), "켜기 틱 W3 불일치"
    assert bool(R._w3_clock.ge) and float(R._w3_clock.ge_obs) == 1.0, "fresh 시작이면 ge·게이트 즉시"
    assert float(np.abs(a - last).max()) <= SLEW_RAD + 1e-6, f"전환 틱 목표 점프 {np.degrees(np.abs(a - last).max()):.2f}° > 슬루"
    lo = R.last_obs; gc = lo[225:235].reshape(5, 2); ac = lo[165:225]
    assert np.allclose(gc, gc[-1]) and abs(float(gc[-1, 1]) - 1.0) < 1e-6 and float(np.abs(ac).max()) == 0.0, "W3 fresh 관측(클록 백필·last_action 0)"
    for k in range(61, 140):
        a = R.step(cmd=cmdf, **_inp(k, pose)); b = ref_w3.step(cmd=cmdf, **_inp(k, pose))
        assert R.mode == MODE_W3 and np.array_equal(a, b), f"W3 모드 불일치 틱 {k}"
    last_w3 = a
    # [3] 끄기: 반주기 경계 래치까지 W3, 래치 틱 다음부터 KE(새 러너 상태·슬루 기준 이어받기)
    R.set_gait_enable(False); ref_w3.set_gait_enable(False); k = 140; n_wait = 0
    while True:
        a = R.step(cmd=cmd0, **_inp(k, pose)); b = ref_w3.step(cmd=cmd0, **_inp(k, pose)); k += 1; n_wait += 1
        assert np.array_equal(a, b), "끄기 대기 중 W3 불일치"
        if R.last_event == "to_ke":
            break
        assert n_wait <= 25, "반주기(0.455 s = 23 틱) 안에 래치가 안 내려감"
    assert R.mode == MODE_KE and not R._w3_clock.ge and not R._w3_clock.want and R.n_to_ke == 1
    last_w3 = a
    ref_ke2 = mk_ke(); ref_ke2.reset(); ref_ke2.core._prev_emitted = np.array(last_w3, np.float32, copy=True)
    a = R.step(cmd=cmdf, **_inp(k, pose, q_off=far)); b = ref_ke2.step(cmd=cmd0, **_inp(k, pose, q_off=far))    # KE 모드는 명령을 0 으로 넣는다
    assert np.array_equal(a, b) and float(np.abs(a - last_w3).max()) <= SLEW_RAD + 1e-6, "KE 첫 틱 불일치 또는 목표 점프"
    assert R.det.n_on == 0 and R.last_mode == 0 and float(np.abs(R.last_obs[165:225]).max()) == 0.0, "KE fresh(감지기·last_action)"
    # [4] KE 회복 모드 중 켜기 차단 사유
    for j in range(5):
        R.step(cmd=cmd0, **_inp(k + 1 + j, pose, w=2.0))
    assert R.last_mode == 1 and R.enable_block_reason() is not None, "회복 모드 켜짐 사유"
    # [5] force_ke: 걷는 중 즉시 인계
    R2 = KeW3Runner(mk_w3(), mk_ke(), handoff="fresh"); R2.reset(); R2.step(cmd=cmd0, **_inp(0, pose)); R2.set_gait_enable(True)
    for j in range(1, 8):
        R2.step(cmd=cmdf, **_inp(j, pose))
    assert R2.mode == MODE_W3 and bool(R2._w3_clock.ge)
    R2.force_ke(); R2.step(cmd=cmdf, **_inp(8, pose))
    assert R2.mode == MODE_KE and R2.last_event == "force_ke" and not R2._w3_clock.ge
    # [6] 계약 불일치 거부
    from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4
    from rl_walking.deploy import w3_deploy as W3
    stub = lambda o: np.zeros(12, np.float32)
    for bad_kw, why in ((dict(kw, default_pose_rad=pose + 0.01), "기본자세"), (dict(kw, act_obs_ema_beta=None), "관측 필터")):
        try:
            KeW3Runner(PolicyRunnerStage4(model_path=stub, w3=W3.W3S3D_CONFIG, **bad_kw), mk_ke()); raise AssertionError(f"{why} 불일치 허용")
        except ValueError:
            pass
    try:
        KeW3Runner(PolicyRunnerStage4(model_path=stub, **kw), mk_ke()); raise AssertionError("비 W3 러너 허용")
    except ValueError:
        pass
    # [7] carry: 이력·EMA·last_action 이어받기(클록 이력만 받는 쪽 값), 첫 W3 틱 STAND → 다음 틱 0→1 램프, 래치 뒤 KE 는 위상 0 클록·감지기 새로
    C = KeW3Runner(mk_w3(), mk_ke(), handoff="carry"); C.reset()
    for k in range(40):
        last = C.step(cmd=cmd0, **_inp(k, pose))
    ke_obs = C.last_obs.copy(); ke_ema = {kk: v.copy() for kk, v in _core(C.ke)._ema.items()}; ke_la = C.last_action.copy()
    C.set_gait_enable(True)
    a = C.step(cmd=cmdf, **_inp(40, pose, q_off=far))
    assert C.mode == MODE_W3 and C.last_event == "to_w3" and not bool(C._w3_clock.ge), "carry: 첫 W3 틱은 STAND"
    assert float(np.abs(a - last).max()) <= SLEW_RAD + 1e-6, "carry 전환 틱 목표 점프"
    o = C.last_obs
    for (a0, b0, fr) in ((0, 15, 3), (15, 30, 3), (45, 105, 12), (105, 165, 12), (165, 225, 12), (245, 260, 3), (260, 320, 12)):
        assert np.array_equal(o[a0:b0 - fr], ke_obs[a0 + fr:b0]), f"carry: 이력 이어받기 실패 {a0}:{b0}"       # 앞 4 프레임 = KE 의 뒤 4 프레임
    assert float(np.abs(o[225:235]).max()) == 0.0, "carry: W3 클록 이력 (0,0)"
    assert np.allclose(o[213:225], ke_ema["act_obs"] + np.float32(0.27) * (ke_la - ke_ema["act_obs"]), atol=1e-6), "carry: last_action·관측 EMA 연속"
    C.step(cmd=cmdf, **_inp(41, pose))
    assert bool(C._w3_clock.ge) and abs(float(C._w3_clock.ge_obs) - 0.1) < 1e-6, "carry: 둘째 틱에 0→1 래치(게이트 램프 시작)"
    for k in range(42, 100):
        C.step(cmd=cmdf, **_inp(k, pose))
    C.set_gait_enable(False); k = 100; w3_obs = None
    while C.last_event != "to_ke":
        w3_obs = None; a = C.step(cmd=cmd0, **_inp(k, pose)); k += 1
        if C.mode == MODE_KE:
            break
        assert k < 130
    w3_last = a; w3_tail = _core(C.ke)._hist["joint_vel_ema"]._buf.copy(); ema_before = {kk: v.copy() for kk, v in _core(C.ke)._ema.items()}
    assert _core(C.ke)._tick == 0 and np.allclose(_core(C.ke)._hist["gait_clock"]._buf, np.array([0.0, 1.0], np.float32)), "carry: KE 클록 이력 = 위상 0"
    a = C.step(cmd=cmd0, **_inp(k, pose, q_off=far))
    assert C.mode == MODE_KE and float(np.abs(a - w3_last).max()) <= SLEW_RAD + 1e-6, "carry: KE 첫 틱 목표 점프"
    jv = C.last_obs[260:320].reshape(5, 12)
    assert np.array_equal(jv[:4], w3_tail[1:]), "carry: joint_vel_ema 이력 이어받기(재시작 없음)"
    assert np.allclose(jv[4], 0.9 * ema_before["joint_vel_ema"] + 0.1 * _inp(k, pose, q_off=far)["qvel_rad_s"], atol=1e-5), "carry: joint_vel_ema 가 순간값으로 재시작하지 않음"
    assert C.det.n_on == 0, "carry: 감지기는 새로"
    # [8] 정착 대기: 래치 뒤 조용한 양발 서기가 settle_quiet_s 이어져야 KE(시끄러우면 상한까지 W3 STAND)
    S = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5, settle_w=0.15, settle_max_s=3.0); S.reset()
    S.step(cmd=cmd0, **_inp(0, pose)); S.set_gait_enable(True)
    for k in range(1, 30):
        S.step(cmd=cmdf, **_inp(k, pose))
    S.set_gait_enable(False); k = 30
    while bool(S._w3_clock.ge):
        S.step(cmd=cmd0, **_inp(k, pose)); k += 1
    n_noisy = 0
    for j in range(40):                                  # |ω| 큰 동안은 넘기지 않는다
        S.step(cmd=cmd0, **_inp(k, pose, w=0.5)); k += 1; n_noisy += 1
        assert S.mode == MODE_W3, "정착 전 KE 인계"
    n_q = 0
    while S.mode == MODE_W3:
        S.step(cmd=cmd0, **_inp(k, pose)); k += 1; n_q += 1
        assert n_q <= 30
    assert 24 <= n_q <= 26, f"정착 0.5 s(25 틱) 뒤 인계여야 함: {n_q}"
    # [9] carry + ema_fresh_ticks 2 는 기동 거부(전환 틱 필터 재시작 경로 차단), fresh 는 허용
    try:
        KeW3Runner(mk_w3(), mk_ke(ema_fresh_ticks=2), handoff="carry")
        raise AssertionError("carry + ema_fresh 2 를 받아들임")
    except ValueError:
        pass
    KeW3Runner(mk_w3(), mk_ke(ema_fresh_ticks=2), handoff="fresh")
    print(f"[selftest] kew3_runner: KE 모드 = 단독 KE(60 틱 비트 동일) · 켜기 틱 W3 fresh(ge·게이트 즉시, 이력 백필) = 단독 W3(80 틱) · "
          f"슬루 기준 이어받기(qpos 28.6° 떨어져도 전환 틱 |Δ목표| ≤ {np.degrees(SLEW_RAD):.0f}°) · 끄기 = 반주기 래치({n_wait} 틱) 다음 틱 KE fresh · "
          f"회복 모드 켜기 차단 · force_ke · 계약 불일치 거부 · carry(이력·EMA·last_action 연속, 클록만 교체, 감지기 새로) · 정착 대기({n_q} 틱) · carry + ema_fresh 2 거부 — ALL PASS")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(__doc__)
