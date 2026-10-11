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

1008 AIR(공중·착지 구간 — 선택, air_runner 를 줄 때만. 안 주면 모든 동작·기록이 종전과 같다): 승윤님 결정 "정책을 공중에서부터 구동한다".
    AIR(매달림 시작으로 배운 서기 정책 G21h, PolicyRunnerStage4)  →  KE  ⇄  W3
       start_air()           : 세션 시작(reset 뒤)에 부르면 AIR 로 시작한다. 안 부르면 종전처럼 KE 로 시작(= 공중 러너가 없는 합성 러너와 같은 동작).
       AIR → KE              : 러너가 스스로 넘긴다 — 양발 접촉 연속 air_contact_s ∧ 조용한 양발(|ω_xy| < air_w) 연속 air_quiet_s,
                               또는 양발 접촉 연속 air_max_s(시끄러워도 — 그때는 회복이 있는 KE 가 낫다). 한 번 넘기면 AIR 로 돌아오지 않는다.
                               이어받기는 이력·필터·last_action·슬루 기준·**클록 틱과 클록 이력까지 그대로**(둘 다 같은 계약의 서기 정책). 감지기는 새로.
       AIR 동안              : 걷기 켜기 거부(enable_block_reason), 회복 감지기 없음(공중 러너는 단일 서기 정책).
    왜 이 구조인가: G21h 를 서기 정책 전체로 바꾸면 공중 시작은 서지만 지면 회복이 기준 미달이었다(reports/gen_w3/AIR_START_1008.md §9).
    공중·착지에만 쓰고 넘기면 착지 뒤의 서기·회복·걷기는 평가가 끝난 종전 구성 그대로다.

브리지가 읽는 러너 인터페이스(step·reset·set_gait_enable·_w3_clock·last_obs·last_action·blackbox·needs_knee_tau …)를 그대로 낸다.
노드의 LPF·가드·워밍업·블렌딩 상태는 건드리지 않는다(전환은 러너 안의 일).

검증: python3 rl_walking/deploy/kew3_runner.py --selftest   (numpy 만, 스텁 정책)
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

MODE_KE, MODE_W3, MODE_AIR = "ke", "w3", "air"
_ACTION_SCALE = 0.25      # policy_runner.ACTION_SCALE 와 같은 값(자가검증 [12] 가 대조) — 전환 완화의 '실행한 행동' 환산에만 쓴다


def _core(r):
    """C5Runner 는 관측·슬루 상태를 .core(PolicyRunnerStage4)에 둔다 — 단일 러너는 자기 자신."""
    return r.__dict__.get("core", r) if hasattr(r, "__dict__") else r


class KeW3Runner:
    """W3 세션의 '서기 정책'을 KE 로 바꾼 합성 러너."""

    def __init__(self, w3_runner, ke_runner, *, handoff: str = "carry", settle_quiet_s: float = 0.0, settle_w: float = 0.15,
                 settle_max_s: float = 3.0, dt: float = 0.02, air_runner=None, air_contact_s: float = 1.0, air_quiet_s: float = 0.5,
                 air_w: float = 0.15, air_max_s: float = 3.0, air_guard_deg=30.0, air_guard_s: float = 0.5, air_guard_rate_deg: float = 1.0,
                 blend_s: float = 0.0, blend_stand: str = "hold"):
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
        # 1008 AIR(선택): 공중·착지 구간의 서기 러너. 단일 PolicyRunnerStage4 만 받는다(감지기·W3 클록 없는 것) — 계약은 KE 코어와 같아야 한다.
        self.air = air_runner
        self.air_contact_s, self.air_quiet_s, self.air_w, self.air_max_s = float(air_contact_s), float(air_quiet_s), float(air_w), float(air_max_s)
        self._air_both_run = 0.0; self._air_quiet_run = 0.0
        self.n_air_to_ke = 0
        # 1008 공중 폭주 차단(AIR 모드 전용 안전 장치 — AIR_START_1008.md §14.5). 양발 무접촉이 air_guard_s 넘게 이어진 동안(= 매달려 있음) 공중 정책이 낸
        # **목표가 기본자세에서 air_guard_deg 넘게 벗어나면**(어느 관절이든; 행동·목표가 비유한이면 즉시) 정책을 끊고 래치한다 — 이후 이 세션은 목표를
        # 틱당 air_guard_rate_deg 씩 **기본자세로 되돌린 뒤 그 자세를 유지**한다(KE 인계·걷기 없음, reset 으로만 풀린다). 접촉 중·무접촉 air_guard_s 전에는 안 걸린다.
        # 왜: 서기 정책들의 last_action 자기 고리는 분포 밖 입력(큰 피치 각속도·공중의 고정 피치)에서 발산할 수 있고, 그때 목표는 슬루 한도(200°/s)로 관절 한계까지 간다
        # (합성 대본에서 G21h-12299·지면 기준 G21a 모두). 원시 행동 |a| 가 아니라 목표 이탈각으로 잰다 — |a| 1 = 14.3° 라 |a| 문턱은 너무 느슨했다(6 = 86°).
        # 평소 공중의 목표 이탈은 10° 안팎(시뮬 — air_dev_max 로 기록). air_guard_deg None = 끔.
        self.air_guard_deg = None if air_guard_deg is None else float(air_guard_deg)
        self.air_guard_s, self.air_guard_rate_deg = float(air_guard_s), float(air_guard_rate_deg)
        if self.air_guard_deg is not None and not (self.air_guard_deg > 0.0 and self.air_guard_s >= 0.0 and self.air_guard_rate_deg > 0.0):
            raise ValueError("air_guard_deg > 0, air_guard_s ≥ 0, air_guard_rate_deg > 0 여야 함")
        self.air_hold = False             # 폭주 차단 래치(기본자세로 되돌려 유지 중)
        self._air_hold_out = None
        self._air_none_run = 0.0
        self.n_air_hold = 0               # 래치 횟수(프로세스 수명 누계)
        self.air_dev_max = 0.0            # 이 세션에서 매달린 동안(무접촉 air_guard_s 뒤) 목표가 기본자세에서 벗어난 최대각 [deg] — 기록용(문턱 여유 확인)
        self.air_act_max = 0.0            # 같은 동안의 원시 행동 |a| 최대 — 기록용
        self.air_act_max_all = 0.0        # 이 세션의 AIR 모드 전체(착지 ~ 인계 포함)에서 본 |a| 최대 — 기록용
        if air_runner is not None:
            ca = _core(air_runner)
            if ca is not air_runner or hasattr(air_runner, "det"):
                raise ValueError("air_runner 는 단일 서기 러너(PolicyRunnerStage4)여야 함 — C5Runner(감지기)·합성 러너는 안 됨")
            if getattr(ca, "_w3_clock", None) is not None:
                raise ValueError("air_runner 가 W3 러너임 — 서기 정책 러너를 넘길 것")
            if ca.obs_dim != ck.obs_dim or list(ca.terms) != list(ck.terms):
                raise ValueError(f"공중 러너 관측 레이아웃 다름: {ca.obs_dim} vs {ck.obs_dim}")
            if not np.array_equal(np.asarray(ca.default_pose), np.asarray(ck.default_pose)):
                raise ValueError("공중 러너 기본자세 다름 — 관측 joint_pos·행동 오프셋 기준이 바뀐다")
            if (ca.act_obs_ema_beta, ca.jvel_obs_ema_beta) != (ck.act_obs_ema_beta, ck.jvel_obs_ema_beta):
                raise ValueError("공중 러너 관측 필터 β 다름")
            if getattr(ca, "jvel_ema_mode", None) != getattr(ck, "jvel_ema_mode", None):
                raise ValueError("공중 러너 jvel_ema_mode 다름")
            if float(ca.gait_freq) != float(ck.gait_freq):
                raise ValueError(f"공중 러너 클록 주파수 다름: {ca.gait_freq} vs {ck.gait_freq} — 클록 틱을 이어받을 수 없다")
            if int(getattr(ca, "ema_fresh_ticks", 1)) != 1 or int(getattr(ck, "ema_fresh_ticks", 1)) != 1:
                raise ValueError("공중 러너 인계는 ema_fresh_ticks 1 에서만(이어받기)")
            if not (self.air_contact_s > 0.0 and self.air_quiet_s > 0.0 and self.air_w > 0.0 and self.air_max_s >= self.air_contact_s):
                raise ValueError("air_contact_s·air_quiet_s·air_w > 0, air_max_s ≥ air_contact_s 여야 함")
        self.w3r, self.ke = w3_runner, ke_runner
        # 1012 서기 → 걷기 전환 완화(선택, 기본 끔 — blend_s 0 이면 아래 상태를 쓰는 분기에 들어가지 않아 모든 출력·기록이 종전과 같다).
        # 걷기로 넘어간 뒤 blend_s 초 동안 **내보내는 목표** = (1 − α)·서기 목표 + α·걷기 러너 목표, α = smoothstep(경과/blend_s).
        #   blend_stand "hold" = 전환 직전에 내보낸 서기 목표를 고정 / "live" = 섞는 동안 서기 러너(KE)도 같은 입력으로 계속 돌려 그 틱의 서기 목표를 쓴다
        #   (KE 가 회복 모드로 들어가면 그 뒤로는 마지막 서기 목표를 고정 — 회복 정책 출력과 걷기 출력을 섞지 않는다).
        # 걷기 러너 안의 상태(이력·last_action·슬루 기준)는 건드리지 않는다 — 걷기 정책은 자기 원시 행동을 last_action 으로 본다(적용된 목표와 다름: blend_dev_deg 로 기록).
        # 걸음 시계는 멈추지 않는다(정책이 배운 적 없는 입력을 만들지 않는다). 끄기 요청·정착·KE 복귀 경로는 종전 그대로.
        if blend_stand not in ("hold", "live"):
            raise ValueError(f"blend_stand 는 'hold' | 'live' (받음 {blend_stand!r})")
        if not float(blend_s) >= 0.0:
            raise ValueError("blend_s ≥ 0 이어야 함")
        self.blend_s, self.blend_stand = float(blend_s), str(blend_stand)
        self._bl_stand = None             # 섞는 중인 서기 목표(None = 섞지 않음)
        self._bl_live = False
        self.blend_alpha = 1.0            # 이번 틱 α(섞지 않으면 1)
        self.blend_dev_deg = 0.0          # 이번 틱 |내보낸 목표 − 걷기 러너 목표| 최대 [deg]
        self.n_blend = 0
        self.last_raw_action = None       # 이번 틱 행동 주체의 원시 행동(섞기와 무관) — 기록용
        self.mode = MODE_KE
        self._want = False
        self._force_ke = False
        self.n_to_w3 = 0
        self.n_to_ke = 0
        self.last_event = ""          # 이번 틱 전환 사건: "" | "to_w3" | "to_ke" | "force_ke"
        self.ticks_in_mode = 0
        self.last_step_action = None  # 이번 틱에 실제로 실행한 원시 행동(전환이 같은 틱에 일어나도 '행동한 러너'의 값) — 시뮬 폐루프·기록용
        self.last_step_kind = 0       # 이번 틱 행동 주체: 0 KE 서기 · 1 KE 회복 · 2 W3 · 3 AIR(공중·착지 서기 — air_runner 가 있을 때만)

    # ---------- 브리지가 읽는 속성 ----------
    @property
    def _active(self):
        if self.mode == MODE_AIR:
            return self.air
        return self.ke if self.mode == MODE_KE else self.w3r

    def __getattr__(self, name):
        """last_obs·last_action·_ema·last_jvel_ema·obs_dim·backend·act_obs_ema_beta … 는 지금 도는 러너에 위임."""
        d = self.__dict__
        if "w3r" not in d or "ke" not in d:
            raise AttributeError(name)
        m = d.get("mode", MODE_KE)
        if m == MODE_AIR and d.get("air") is not None:
            return getattr(d["air"], name)
        return getattr(d["ke"] if m == MODE_KE else d["w3r"], name)

    @property
    def has_air(self) -> bool:
        """공중·착지 러너가 있는가(이 합성 러너가 공중 정책 시작을 받을 수 있는가)."""
        return self.air is not None

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
        if self.mode == MODE_AIR and self.air_hold:
            return "공중 폭주 차단(기본자세 유지 중) — 정지한 뒤 처음부터 다시 시작할 것"
        if self.mode == MODE_AIR:
            return "공중·착지 정책 구간 — 착지가 가라앉아 KE 로 넘어간 뒤에 켤 것"
        if self.mode == MODE_KE and int(self.ke.last_mode) == 1:
            return "KE 회복 모드 켜짐 — 서기 복귀 뒤에 켤 것"
        return None

    def blackbox(self) -> dict:
        bb = dict(self.ke.blackbox()) if hasattr(self.ke, "blackbox") else {}
        bb["kew3"] = {"mode": self.mode, "event": self.last_event, "n_to_w3": int(self.n_to_w3), "n_to_ke": int(self.n_to_ke),
                      "ticks": int(self.ticks_in_mode), "want": bool(self._want)}
        if self.blend_s > 0.0:                          # 전환 완화를 켠 키에서만 키 추가(끈 키의 스키마는 불변)
            bb["kew3"]["blend_a"] = round(float(self.blend_alpha), 3); bb["kew3"]["blend_dev"] = round(float(self.blend_dev_deg), 2)
        if self.air is not None:                        # 공중 러너가 있는 키에서만 키 추가(없는 키의 스키마는 불변)
            bb["kew3"]["n_air_to_ke"] = int(self.n_air_to_ke)
            bb["kew3"]["air_hold"] = bool(self.air_hold); bb["kew3"]["n_air_hold"] = int(self.n_air_hold)
            bb["kew3"]["air_dev"] = round(float(min(self.air_dev_max, 1e4)), 1)
        return bb

    # ---------- 상태 ----------
    def reset(self) -> None:
        """세션 시작(stand) — 두 러너 초기화, KE 로 시작."""
        self.w3r.reset()
        self.ke.reset()
        if self.air is not None:
            self.air.reset()
        self.mode = MODE_KE
        self._want = False
        self._force_ke = False
        self.last_event = ""
        self.ticks_in_mode = 0
        self._settle_run = 0.0; self._since_latch = None; self._w3_arm_want = False
        self._air_both_run = 0.0; self._air_quiet_run = 0.0
        self.air_hold = False; self._air_hold_out = None; self._air_none_run = 0.0
        self.air_dev_max = 0.0; self.air_act_max = 0.0; self.air_act_max_all = 0.0
        self._bl_stand = None; self._bl_live = False; self.blend_alpha = 1.0; self.blend_dev_deg = 0.0

    def start_air(self) -> None:
        """1008: 이 세션을 AIR(공중·착지 서기 정책)로 시작한다 — **reset() 바로 뒤, 첫 step 전에** 부른다(공중 정책 시작 = 브리지 hang_arm).
        부르지 않으면 종전처럼 KE 로 시작한다. 공중 러너가 없거나 이미 스텝이 돈 뒤면 거부."""
        if self.air is None:
            raise RuntimeError("공중·착지 러너가 없는 합성 러너 — start_air 불가")
        if self.ticks_in_mode != 0 or self.mode != MODE_KE or self._want:
            raise RuntimeError("start_air 는 reset() 바로 뒤(첫 step 전)에만")
        self.air.reset()
        self.mode = MODE_AIR
        self._air_both_run = 0.0; self._air_quiet_run = 0.0
        self.air_hold = False; self._air_hold_out = None; self._air_none_run = 0.0
        self.air_dev_max = 0.0; self.air_act_max = 0.0; self.air_act_max_all = 0.0
        self.last_event = "start_air"

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

    @staticmethod
    def _carry_same(src, dst) -> None:
        """src → dst(둘 다 같은 계약의 서기 정책 코어): 이력 전부(클록 이력 포함)·EMA·last_action·슬루 기준·클록 틱을 그대로 복사 —
        받는 정책은 주는 정책이 보던 관측 이력을 끊김 없이 이어서 본다(AIR → KE)."""
        for name, h in src._hist.items():
            d = dst._hist[name]
            d._buf = h._buf.copy()
            d._pushes = h._pushes
        dst._ema = {k: v.copy() for k, v in src._ema.items()}
        dst.last_action = src.last_action.copy()
        dst.last_jvel_ema = None if src.last_jvel_ema is None else src.last_jvel_ema.copy()
        dst._prev_emitted = None if src._prev_emitted is None else np.array(src._prev_emitted, dtype=np.float32, copy=True)
        dst._tick = int(src._tick)

    def _air_to_ke(self) -> None:
        ca, ck = _core(self.air), _core(self.ke)
        self.ke.reset()                                 # 감지기 새로(기준 자세 ref_init_s 추종), 코어 초기화 → 아래에서 공중 러너의 상태를 이어받는다
        self._carry_same(ca, ck)
        self._air_both_run = 0.0; self._air_quiet_run = 0.0
        self.mode = MODE_KE
        self.n_air_to_ke += 1
        self.ticks_in_mode = 0
        self.last_event = "air_to_ke"

    def set_gait_enable(self, want: bool) -> None:
        """명령이 원하는 gait_enable. 켜기 = 다음 step 에서 W3 로 전환(같은 틱에 ge 켜짐), 끄기 = W3 래치가 반주기 경계에서 내린다."""
        if self.mode == MODE_AIR:                       # 공중·착지 구간: 켜기는 받지 않는다(남겨 두면 KE 로 넘어간 틱에 걷기가 켜진다)
            self._want = False
            self._w3_clock.set_want(False)
            return
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
        if self.blend_s > 0.0 and prev is not None:     # 전환 완화: 전환 직전에 내보낸 서기 목표에서 출발
            self._bl_stand = np.array(prev, dtype=np.float32, copy=True); self._bl_live = (self.blend_stand == "live"); self.n_blend += 1
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
        self._bl_stand = None; self._bl_live = False; self.blend_alpha = 1.0; self.blend_dev_deg = 0.0
        self.mode = MODE_KE
        self._force_ke = False
        self.n_to_ke += 1
        self.ticks_in_mode = 0
        self.last_event = event

    def _blend_out(self, out, gyro_rad_s, gravity_unit, qpos_rad, qvel_rad_s, contact2, tau_knee_nm):
        """걷기 러너가 낸 목표 out 을 서기 목표와 섞어 돌려준다(전환 완화 중일 때만 불린다). 걷기 러너의 상태는 바꾸지 않는다."""
        s = self.ticks_in_mode * self.dt / self.blend_s
        if s >= 1.0 - 1e-9:                             # 끝 — 이 틱부터 걷기 러너 목표 그대로(비트 동일)
            self._bl_stand = None; self._bl_live = False; self.blend_alpha = 1.0; self.blend_dev_deg = 0.0
            return out
        if self._bl_live:                               # 실시간 서기 목표: KE 를 같은 입력으로 한 틱 돌린다(명령 0)
            zero = np.zeros(3, dtype=np.float32)
            if getattr(self.ke, "needs_knee_tau", False) or tau_knee_nm is not None:
                st = self.ke.step(gyro_rad_s, gravity_unit, zero, qpos_rad, qvel_rad_s, contact2, tau_knee_nm=tau_knee_nm)
            else:
                st = self.ke.step(gyro_rad_s, gravity_unit, zero, qpos_rad, qvel_rad_s, contact2)
            if int(self.ke.last_mode) == 1:             # KE 가 회복 모드로 — 회복 출력은 섞지 않는다: 마지막 서기 목표로 고정
                self._bl_live = False
            else:
                self._bl_stand = np.array(st, dtype=np.float32, copy=True)
        a = float(3.0 * s * s - 2.0 * s * s * s)
        o = np.asarray(out, dtype=np.float32)
        mix = ((1.0 - a) * self._bl_stand + a * o).astype(np.float32)
        self.blend_alpha = a
        self.blend_dev_deg = float(np.degrees(np.abs(mix - o).max()))
        # 시뮬 폐루프·기록용 '실행한 행동' = 내보낸 목표와 같은 값이 되도록 환산(원시 행동은 last_raw_action 에 남는다)
        cw = _core(self.w3r)
        self.last_step_action = ((mix - np.asarray(cw.default_pose, dtype=np.float32)) / float(_ACTION_SCALE)).astype(np.float32)
        return mix

    # ---------- 스텝 ----------
    def step(self, gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s, contact2, tau_knee_nm=None) -> np.ndarray:
        self.last_event = ""
        if self.mode == MODE_AIR:                       # 1008 공중·착지 구간: 공중 러너가 행동, 착지가 가라앉으면 이 틱까지 AIR·다음 틱부터 KE
            _pose = np.asarray(_core(self.air).default_pose, dtype=np.float32)
            if self.air_hold:                           # 폭주 차단 래치: 기본자세로 천천히 되돌린 뒤 유지(정책·인계 없음) — reset 으로만 풀린다
                self.ticks_in_mode += 1
                self.last_step_action = np.zeros(12, dtype=np.float32); self.last_step_kind = 3
                _r = float(np.radians(self.air_guard_rate_deg))
                self._air_hold_out = (self._air_hold_out + np.clip(_pose - self._air_hold_out, -_r, _r)).astype(np.float32)
                return self._air_hold_out.copy()
            _prev = _core(self.air)._prev_emitted
            _prev = None if _prev is None else np.array(_prev, dtype=np.float32, copy=True)
            out = self.air.step(gyro_rad_s, gravity_unit, np.zeros(3, dtype=np.float32), qpos_rad, qvel_rad_s, contact2)
            self.ticks_in_mode += 1
            self.last_step_action = np.array(_core(self.air).last_action, dtype=np.float32, copy=True); self.last_step_kind = 3
            g = np.asarray(gyro_rad_s, dtype=np.float64).reshape(-1); c = np.asarray(contact2, dtype=np.float64).reshape(-1)
            _fin = bool(np.isfinite(self.last_step_action).all()) and bool(np.isfinite(np.asarray(out)).all())
            _am = float(np.abs(self.last_step_action).max()) if _fin else float("inf")
            _dev = float(np.degrees(np.abs(np.asarray(out, dtype=np.float32) - _pose).max())) if _fin else float("inf")
            self._air_none_run = self._air_none_run + self.dt if (c[0] <= 0.5 and c[1] <= 0.5) else 0.0
            if _fin:
                self.air_act_max_all = max(self.air_act_max_all, _am)
            if self._air_none_run >= self.air_guard_s - 1e-9:      # 매달려 있는 동안
                if _fin:
                    self.air_act_max = max(self.air_act_max, _am); self.air_dev_max = max(self.air_dev_max, _dev)
                if self.air_guard_deg is not None and _prev is not None and ((not _fin) or _dev > self.air_guard_deg):
                    self.air_hold = True; self._air_hold_out = _prev      # 직전 방출값에서 출발(이번 틱의 목표는 내보내지 않는다) — 이후 기본자세로 되돌린다
                    self.n_air_hold += 1
                    self.last_event = "air_hold"
                    self.last_step_action = np.zeros(12, dtype=np.float32)
                    _r = float(np.radians(self.air_guard_rate_deg))
                    self._air_hold_out = (self._air_hold_out + np.clip(_pose - self._air_hold_out, -_r, _r)).astype(np.float32)
                    return self._air_hold_out.copy()
            both = bool(c[0] > 0.5 and c[1] > 0.5)
            self._air_both_run = self._air_both_run + self.dt if both else 0.0
            self._air_quiet_run = self._air_quiet_run + self.dt if (both and float(np.hypot(g[0], g[1])) < self.air_w) else 0.0
            if ((self._air_both_run >= self.air_contact_s - 1e-9 and self._air_quiet_run >= self.air_quiet_s - 1e-9)
                    or self._air_both_run >= self.air_max_s - 1e-9):
                self._air_to_ke()
            return out
        if self.mode == MODE_W3 and self._force_ke:
            self._to_ke("force_ke")
        if self.mode == MODE_KE and self._want:
            self._to_w3()
        if self.mode == MODE_W3:
            out = self.w3r.step(gyro_rad_s, gravity_unit, cmd, qpos_rad, qvel_rad_s, contact2)
            self.ticks_in_mode += 1
            self.last_step_action = np.array(_core(self.w3r).last_action, dtype=np.float32, copy=True); self.last_step_kind = 2
            if self._bl_stand is not None:              # 1012 전환 완화(켠 키·전환 뒤 blend_s 동안만)
                self.last_raw_action = self.last_step_action
                out = self._blend_out(out, gyro_rad_s, gravity_unit, qpos_rad, qvel_rad_s, contact2, tau_knee_nm)
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
        self.last_step_action = np.array(_core(self.ke).last_action, dtype=np.float32, copy=True); self.last_step_kind = int(self.ke.last_mode)
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
    Wa = rng.normal(0, 0.05, (12, 320)).astype(np.float32)      # 1008 공중·착지 서기 정책 스텁(위 셋의 난수열 뒤에 뽑는다 — 종전 스텁 값 불변)
    f = lambda W: (lambda o: np.tanh(W @ np.asarray(o, np.float32)).astype(np.float32))
    pose = (rng.normal(0, 0.2, 12)).astype(np.float32)
    kw = dict(obs_dim=320, gait_freq=1.1, default_pose_rad=pose, act_obs_ema_beta=0.27, jvel_obs_ema_beta=0.25)
    det_kw = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5, off_contact="any",
                  early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33, early_gate=(5.0, 1.0))
    mk_w3 = lambda: PolicyRunnerStage4(model_path=f(Ww), w3=W3.W3S3D_CONFIG, **kw)
    mk_ke = lambda **extra: C5Runner(f(Ws), f(Wr), detector=SwitchDetector(**det_kw), cue_vx=0.03, cue_s=1.0, cue_dir_imu=True, **kw, **extra)
    _mk.air = lambda **extra: PolicyRunnerStage4(model_path=f(Wa), **dict(kw, **extra))     # selftest [10] 용(반환값 꼴은 종전 그대로)
    _mk.air_with = lambda net, **extra: PolicyRunnerStage4(model_path=net, **dict(kw, **extra))     # selftest [11] 용(임의 스텁 정책)
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
    # [8b] last_step_action/kind: 전환 틱에도 '행동한 러너'의 원시 행동
    T = KeW3Runner(mk_w3(), mk_ke(), handoff="fresh"); T.reset()
    T.step(cmd=cmd0, **_inp(0, pose)); assert T.last_step_kind == 0 and np.array_equal(T.last_step_action, _core(T.ke).last_action)
    T.set_gait_enable(True); T.step(cmd=cmdf, **_inp(1, pose)); assert T.last_step_kind == 2 and np.array_equal(T.last_step_action, _core(T.w3r).last_action)
    T.set_gait_enable(False); k = 2
    while T.mode == MODE_W3:
        T.step(cmd=cmd0, **_inp(k, pose)); k += 1
        assert k < 40
    assert T.last_step_kind == 2 and T.last_event == "to_ke" and float(np.abs(T.last_step_action).max()) > 0.0 and float(np.abs(_core(T.ke).last_action).max()) == 0.0, "전환 틱 행동 = W3 의 것(KE fresh 는 0)"
    # [9] carry + ema_fresh_ticks 2 는 기동 거부(전환 틱 필터 재시작 경로 차단), fresh 는 허용
    try:
        KeW3Runner(mk_w3(), mk_ke(ema_fresh_ticks=2), handoff="carry")
        raise AssertionError("carry + ema_fresh 2 를 받아들임")
    except ValueError:
        pass
    KeW3Runner(mk_w3(), mk_ke(ema_fresh_ticks=2), handoff="fresh")
    # [10] 1008 AIR(공중·착지 구간): 공중 러너가 있어도 start_air 를 안 부르면 종전과 비트 동일 · AIR = 단독 공중 러너와 비트 동일 ·
    #      착지 정착 뒤 한 번만 KE 로(이력·클록 틱까지 이어받기) · 시끄러우면 상한에서 · 한 발 접촉으로는 안 넘김 · 걷기 켜기 거부 · 계약 불일치 거부
    mk_air = _mk.air
    def inp_c(k, c, **kw_):
        d = _inp(k, pose, **kw_); d["contact2"] = np.array(c, np.float32); return d
    old_bb_keys = set(KeW3Runner(mk_w3(), mk_ke(), handoff="carry").blackbox()["kew3"])
    A0 = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5, air_runner=mk_air()); A0.reset()
    B0 = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5); B0.reset()
    assert A0.has_air and not B0.has_air and A0.mode == MODE_KE
    for k in range(80):                                   # 지면 시작(start_air 없음) = 공중 러너 없는 합성 러너와 같은 출력
        if k == 30:
            A0.set_gait_enable(True); B0.set_gait_enable(True)
        if k == 55:
            A0.set_gait_enable(False); B0.set_gait_enable(False)
        cm = cmdf if 30 <= k < 55 else cmd0
        a = A0.step(cmd=cm, **_inp(k, pose)); b = B0.step(cmd=cm, **_inp(k, pose))
        assert np.array_equal(a, b) and A0.mode == B0.mode and A0.last_event == B0.last_event, f"공중 러너가 있는데 지면 시작이 달라짐(틱 {k})"
    assert set(B0.blackbox()["kew3"]) == old_bb_keys and set(A0.blackbox()["kew3"]) == old_bb_keys | {"n_air_to_ke", "air_hold", "n_air_hold", "air_dev"}, "블랙박스 스키마"
    try:
        B0.reset(); B0.start_air(); raise AssertionError("공중 러너 없이 start_air 허용")
    except RuntimeError:
        pass
    try:
        A0.start_air(); raise AssertionError("스텝이 돈 뒤 start_air 허용")
    except RuntimeError:
        pass
    A = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5, air_runner=mk_air()); A.reset(); A.start_air()
    ref_air = mk_air(); ref_air.reset()
    assert A.mode == MODE_AIR and A.last_event == "start_air" and A.enable_block_reason() is not None and A.last_mode == 0
    for k in range(100):                                  # 공중(접촉 0,0): 큰 각속도·기울기에도 계속 AIR, 출력 = 단독 공중 러너
        dd = inp_c(k, (0.0, 0.0), w=(1.2 if 40 <= k < 60 else 0.0), tilt=(6.0 if 40 <= k < 60 else 0.0))
        a = A.step(cmd=cmdf, **dd); b = ref_air.step(cmd=cmd0, **dd)
        assert A.mode == MODE_AIR and np.array_equal(a, b) and A.last_step_kind == 3, f"AIR 불일치 틱 {k}"
    A.set_gait_enable(True)                               # AIR 중 걷기 켜기는 버린다
    assert A.mode == MODE_AIR and A._want is False and not A._w3_clock.want and A.blackbox()["kew3"]["mode"] == "air"
    k = 100
    for j in range(60):                                   # 한 발만 접촉: 넘기지 않는다
        dd = inp_c(k, (1.0, 0.0)); A.step(cmd=cmd0, **dd); ref_air.step(cmd=cmd0, **dd); k += 1
    assert A.mode == MODE_AIR and A.n_air_to_ke == 0
    n_c = 0
    while A.mode == MODE_AIR:                             # 조용한 양발 접촉: 1.0 s(50 틱)째에 넘긴다(조용함 0.5 s 는 그 안에 찬다)
        dd = inp_c(k, (1.0, 1.0)); a = A.step(cmd=cmd0, **dd); b = ref_air.step(cmd=cmd0, **dd); k += 1; n_c += 1
        assert np.array_equal(a, b), "인계 틱까지 AIR 출력"
        assert n_c <= 60
    assert n_c == 50 and A.last_event == "air_to_ke" and A.n_air_to_ke == 1 and A.last_step_kind == 3, f"양발 접촉 1.0 s 에 인계여야 함: {n_c}"
    ca, ck = ref_air, _core(A.ke)
    assert ck._tick == ca._tick and all(np.array_equal(ck._hist[n_]._buf, ca._hist[n_]._buf) for n_ in ca._hist), "AIR → KE: 이력·클록 틱 이어받기"
    assert np.array_equal(ck.last_action, ca.last_action) and np.array_equal(ck._prev_emitted, ca._prev_emitted) and A.det.n_on == 0
    air_last = a
    a = A.step(cmd=cmd0, **inp_c(k, (1.0, 1.0), q_off=far)); k += 1
    assert A.mode == MODE_KE and A.last_step_kind == 0 and float(np.abs(a - air_last).max()) <= SLEW_RAD + 1e-6, "KE 첫 틱 목표 점프"
    assert A.enable_block_reason() is None
    for j in range(80):                                   # 다시 들려도(접촉 0,0) AIR 로 돌아가지 않는다
        A.step(cmd=cmd0, **inp_c(k, (0.0, 0.0))); k += 1
    assert A.mode == MODE_KE and A.n_air_to_ke == 1
    N = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", air_runner=mk_air()); N.reset(); N.start_air(); n_n = 0
    while N.mode == MODE_AIR:                             # 양발 접촉이지만 계속 흔들림(|ω| 0.5): 상한 3.0 s(150 틱)에 넘긴다
        N.step(cmd=cmd0, **inp_c(n_n, (1.0, 1.0), w=0.5)); n_n += 1
        assert n_n <= 160
    assert n_n == 150, f"시끄러운 양발 접촉은 상한 3.0 s 에 인계: {n_n}"
    assert A.n_air_hold == 0 and N.n_air_hold == 0 and not A.air_hold and 0.0 < A.air_act_max <= A.air_act_max_all <= 1.0 and 0.0 < A.air_dev_max <= 14.4, "스텁(|a| ≤ 1 = 14.3°)에서 폭주 차단이 걸림"
    # [11] 1008 공중 폭주 차단: 매달린 동안(무접촉 0.5 s 뒤) 목표가 기본자세에서 30° 넘게 벗어나면 정책을 끊고 래치 → 틱당 1° 씩 기본자세로 되돌려 유지.
    #      인계·걷기 없음, reset 으로만 풀림 · 무접촉 0.5 s 전·접촉 중에는 안 걸림 · 비유한이면 즉시 · 끄면(None) 단독 러너와 같은 출력 · 값 검증
    gain = {"g": 0.3}
    def mk_big():                                         # |a| 최대 = 20·g 인 스텁 공중 정책(g 를 틱마다 바꿔 폭주를 흉내 낸다) — 목표 이탈 = 20·g·14.3°(슬루 4°/틱으로 따라감)
        base = _core(mk_air())._net
        def net(o):
            y = np.asarray(base(o), np.float32); return (20.0 * gain["g"] * y / max(float(np.abs(y).max()), 1e-9)).astype(np.float32)
        return _mk.air_with(net)
    def run(G, contacts, gains):
        gain["g"] = gains[0]; G.reset(); G.start_air(); outs = []
        for k_, (c_, g_) in enumerate(zip(contacts, gains)):
            gain["g"] = g_; outs.append(G.step(cmd=cmd0, **inp_c(k_, c_)))
        return outs
    NC, BC = (0.0, 0.0), (1.0, 1.0)
    dev = lambda o_: float(np.degrees(np.abs(np.asarray(o_) - pose).max()))
    G = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5, air_runner=mk_big())
    o = run(G, [NC] * 120, [0.05] * 40 + [0.5] * 80)     # 40 틱째부터 |a| = 10(목표 143°) → 목표가 4°/틱으로 벌어지다 30° 를 넘는 틱에 래치
    assert G.air_hold and G.n_air_hold == 1 and G.mode == MODE_AIR and "폭주" in G.enable_block_reason(), "매달린 채 목표가 30° 넘게 벗어나는데 래치 없음"
    k_l = next(i for i in range(41, 120) if dev(o[i]) < dev(o[i - 1]) - 1e-6)
    assert 22.0 <= max(dev(x) for x in o) <= 30.0 + 1e-3 and 42 <= k_l <= 46, f"래치 전 최대 이탈 {max(dev(x) for x in o):.1f}° · 래치 틱 {k_l}"
    assert all(dev(o[i]) <= dev(o[i - 1]) + 1e-6 and float(np.degrees(np.abs(o[i] - o[i - 1]).max())) <= 1.0 + 1e-4 for i in range(k_l, 120)), "래치 뒤 되돌림이 틱당 1° 를 넘거나 다시 벌어짐"
    assert dev(o[-1]) < 1e-3 and G.blackbox()["kew3"]["air_hold"] is True and G.blackbox()["kew3"]["n_air_hold"] == 1 and 30.0 < G.blackbox()["kew3"]["air_dev"] <= 34.1, "되돌린 뒤 기본자세 유지(air_dev 는 래치를 건 틱의 값까지 기록)"
    for k_ in range(120, 120 + 200):                      # 래치 뒤: 양발이 닿아 4 s 가 지나도 KE 로 넘기지 않고 기본자세, 걷기 켜기도 버린다
        G.set_gait_enable(True); a = G.step(cmd=cmdf, **inp_c(k_, BC))
        assert dev(a) < 1e-3 and G.mode == MODE_AIR and G.n_air_to_ke == 0 and not G._want, "래치 뒤 인계·걷기"
    G.reset(); assert not G.air_hold and G.mode == MODE_KE and G.air_dev_max == 0.0
    o = run(G, [NC] * 20 + [BC] * 60, [0.5] * 80)         # 무접촉 0.4 s 뒤 접촉: 매달림 조건(0.5 s)이 차기 전이라 안 걸리고 정상 인계(착지 과도 보호)
    assert not G.air_hold and G.n_air_hold == 1 and G.mode == MODE_KE and G.n_air_to_ke == 1, "접촉 중 큰 목표에 래치"
    o = run(G, [NC] * 80, [0.05] * 40 + [0.1] * 40)       # |a| = 2(목표 28.6°) → 30° 아래라 안 걸림
    assert not G.air_hold and 27.0 < G.air_dev_max < 30.0, f"30° 아래에서 래치 또는 기록 이상({G.air_dev_max:.1f}°)"
    o = run(G, [NC] * 60, [0.05] * 40 + [float("nan")] * 20)         # 비유한 → 그 틱에 즉시(비유한 목표를 내보내지 않는다)
    assert G.air_hold and all(np.isfinite(x).all() for x in o) and dev(o[40]) <= dev(o[39]) + 1e-6, "비유한 행동이 목표로 나감"
    Goff = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5, air_runner=mk_big(), air_guard_deg=None)
    ref_b = mk_big()
    gain["g"] = 0.5; Goff.reset(); Goff.start_air(); ref_b.reset()
    for k_ in range(60):
        dd = inp_c(k_, NC); assert np.array_equal(Goff.step(cmd=cmd0, **dd), ref_b.step(cmd=cmd0, **dd)) and not Goff.air_hold, "끈 차단이 출력을 바꿈"
    for bad in (dict(air_guard_deg=0.0), dict(air_guard_s=-1.0), dict(air_guard_rate_deg=0.0)):
        try:
            KeW3Runner(mk_w3(), mk_ke(), handoff="carry", air_runner=mk_air(), **bad); raise AssertionError(f"잘못된 차단 값 수락 {bad}")
        except ValueError:
            pass
    print("[selftest] kew3_runner 공중 폭주 차단: 매달린 채(무접촉 0.5 s 뒤) 목표가 기본자세에서 30° 넘게 벗어나면 래치(최대 이탈 ≤ 30°) → 틱당 1° 씩 기본자세로 되돌려 유지"
          "(양발이 닿아 4 s 가 지나도 인계·걷기 없음, reset 으로 풀림) · 무접촉 0.4 s 뒤 접촉이면 안 걸리고 정상 인계 · 28.6° 는 안 걸림 · 비유한은 그 틱에 · 끄면 단독 러너와 비트 동일 · 잘못된 값 거부 · 스텁 무발동")
    print(f"[selftest] kew3_runner AIR: start_air 없으면 종전과 비트 동일(80 틱·걷기 포함)·블랙박스 스키마 불변 · AIR = 단독 공중 러너(100 틱, 큰 흔들림에도 유지) · "
          f"걷기 켜기 거부 · 한 발 접촉 미인계 · 조용한 양발 {n_c} 틱에 KE(이력·클록 틱 이어받기·목표 점프 ≤ 슬루) · 재들림에도 AIR 로 안 돌아감 · 시끄러우면 {n_n} 틱 상한 · 계약 불일치 5 종 거부")
    print(f"[selftest] kew3_runner: KE 모드 = 단독 KE(60 틱 비트 동일) · 켜기 틱 W3 fresh(ge·게이트 즉시, 이력 백필) = 단독 W3(80 틱) · "
          f"슬루 기준 이어받기(qpos 28.6° 떨어져도 전환 틱 |Δ목표| ≤ {np.degrees(SLEW_RAD):.0f}°) · 끄기 = 반주기 래치({n_wait} 틱) 다음 틱 KE fresh · "
          f"회복 모드 켜기 차단 · force_ke · 계약 불일치 거부 · carry(이력·EMA·last_action 연속, 클록만 교체, 감지기 새로) · 정착 대기({n_q} 틱) · carry + ema_fresh 2 거부 — ALL PASS")
    return 0


def selftest_blend() -> int:
    """[12] 1012 서기 → 걷기 전환 완화: 끄면 종전과 비트 동일 · hold/live 식 · 연속성 · 끝난 뒤 걷기 목표 그대로 · 섞는 중 끄기 → KE 복귀."""
    from rl_walking.deploy.policy_runner import ACTION_SCALE, SLEW_RAD
    assert float(ACTION_SCALE) == float(_ACTION_SCALE), "행동 배율 상수 불일치"
    mk_w3, mk_ke, pose, kw = _mk()
    cmd0 = np.zeros(3, np.float32)
    seq = [False] * 40 + [True] * 150 + [False] * 120 + [True] * 60       # 서기 → 걷기 → 끄기 → 다시 걷기

    def run(R, n=None, on_step=None):
        R.reset(); outs = []; bbs = []
        for k, w in enumerate(seq[:n]):
            R.set_gait_enable(w)
            o = R.step(cmd=cmd0, **_inp(k, pose))
            outs.append(np.array(o, np.float32, copy=True)); bbs.append(R.blackbox())
            if on_step:
                on_step(k, R, o)
        return np.stack(outs), bbs

    # (a) 끈 러너 = 인자를 주지 않은 러너(비트 동일·블랙박스 스키마 동일)
    o0, b0 = run(KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5))
    o1, b1 = run(KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5, blend_s=0.0, blend_stand="live"))
    assert np.array_equal(o0, o1) and [sorted(x["kew3"]) for x in b0] == [sorted(x["kew3"]) for x in b1], "blend_s 0 이 종전과 다름"
    assert "blend_a" not in b0[0]["kew3"]
    n_on = 40                                                              # 켜기 틱(이 틱에 W3 로)
    for T in (0.25, 0.5, 1.0, 2.0):
        K = int(np.ceil(T / 0.02 - 1e-9))                                 # 섞기가 끝나는 걷기 틱(경과 ≥ T)
        # (b) hold: 내보낸 목표 = (1 − α)·전환 직전 서기 목표 + α·(끈 쌍둥이의 걷기 목표), 끝난 뒤 쌍둥이와 비트 동일
        Rh = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5, blend_s=T, blend_stand="hold")
        acts = []
        oh, bh = run(Rh, on_step=lambda k, R, o: acts.append((np.array(R.last_step_action, np.float32), int(R.last_step_kind))))
        hold = o0[n_on - 1]
        for i in range(1, K):                                             # ticks_in_mode = i (첫 걷기 틱이 1)
            s_ = i * 0.02 / T; a_ = 3 * s_ * s_ - 2 * s_ ** 3; k = n_on + i - 1
            want = ((1.0 - a_) * hold + a_ * o0[k]).astype(np.float32)
            assert np.allclose(oh[k], want, atol=1e-6), f"hold 식 불일치 T={T} i={i}"
            assert abs(bh[k]["kew3"]["blend_a"] - round(a_, 3)) < 1e-9
            assert np.allclose(pose + ACTION_SCALE * acts[k][0], oh[k], atol=1e-5) and acts[k][1] == 2, "실행한 행동 환산이 내보낸 목표와 다름"
        assert np.array_equal(oh[n_on + K - 1:190], o0[n_on + K - 1:190]), f"섞기 끝난 뒤가 종전 걷기와 다름 T={T}"
        d1 = float(np.degrees(np.abs(oh[n_on] - oh[n_on - 1]).max())); d0 = float(np.degrees(np.abs(o0[n_on] - o0[n_on - 1]).max()))
        assert d1 <= d0 + 1e-4, f"전환 틱 목표 변화가 줄지 않음 {d1} > {d0}"
        step_max = float(np.degrees(np.abs(np.diff(oh[n_on - 1:n_on + K + 5], axis=0)).max()))
        assert step_max <= np.degrees(SLEW_RAD) + 1e-3, f"섞는 동안 한 틱 변화 {step_max}° 가 슬루 한도를 넘음"
        assert np.array_equal(oh[:n_on], o0[:n_on]), "켜기 전 서기 구간이 바뀜"
        # (c) live: 서기 목표 = 계속 도는 단독 KE 의 그 틱 목표
        ref = mk_ke(); ref.reset(); ko = [np.array(ref.step(cmd=cmd0, **_inp(k, pose)), np.float32) for k in range(n_on + K + 2)]
        Rl = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5, blend_s=T, blend_stand="live")
        ol, _ = run(Rl, n=n_on + K + 2)
        for i in range(1, K):
            s_ = i * 0.02 / T; a_ = 3 * s_ * s_ - 2 * s_ ** 3; k = n_on + i - 1
            assert np.allclose(ol[k], ((1.0 - a_) * ko[k] + a_ * o0[k]).astype(np.float32), atol=1e-6), f"live 식 불일치 T={T} i={i}"
        assert np.array_equal(ol[n_on + K - 1:], o0[n_on + K - 1:n_on + K + 2])
    # (d) 섞는 중 끄기: 종전 경로(반주기 래치 → 정착 → KE)로 돌아오고 섞기 상태가 지워진다
    Rc = KeW3Runner(mk_w3(), mk_ke(), handoff="carry", settle_quiet_s=0.5, blend_s=2.0, blend_stand="hold"); Rc.reset()
    back = None
    for k in range(400):
        Rc.set_gait_enable(40 <= k < 55)                                  # 0.3 s 만 켰다 끔(섞기 15 %)
        Rc.step(cmd=cmd0, **_inp(k, pose))
        if k > 55 and Rc.mode == MODE_KE and back is None:
            back = k
    assert back is not None and Rc._bl_stand is None and Rc.blend_alpha == 1.0 and Rc.n_to_ke == 1, "섞는 중 끄기 뒤 KE 로 돌아오지 않음"
    print(f"[selftest] kew3_runner 전환 완화: 끄면 종전과 비트 동일(370 틱·블랙박스 스키마) · hold/live 식(T 0.25·0.5·1·2 s) · 끝난 뒤 걷기 목표 비트 동일 · "
          f"한 틱 변화 ≤ 슬루 한도 · 실행한 행동 환산 = 내보낸 목표 · 섞는 중 끄기 → {back - 55} 틱 뒤 KE 복귀: PASS")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest() or selftest_blend())
    print(__doc__)
