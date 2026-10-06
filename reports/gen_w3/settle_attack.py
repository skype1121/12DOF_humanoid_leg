"""STOPEST §13a (1007): KE 회복 종료 판정기 공격 — 합성 입력 생성기 + 판정기 단독 CPU 재현(배포 구현 c5_switch.SwitchDetector 그대로).

생성기(운동학 일관): 1003 조용한 서기 qpos 에서 출발. 몸통 피치 θ(t)(발목축 둘레 회전, 발 고정 — 발목 앞뒤 관절이 θ 를 흡수)와
골반 수평 병진 x(t)(다리 강체 기울기 — 힙·발목 앞뒤 관절 반대 부호)를 만든 뒤, 같은 운동에서
  자이로(몸통) = R_wbᵀ ω_w (+ 잡음 0.01) · 중력(몸통) = R_wbᵀ(0,0,−1) · 엔코더 12(JOINT_ORDER) · 접촉 0/1 · 무릎 τ(관절 프레임)
를 만든다. 무릎 처짐(엔코더 = 링크 + (τ − τ0)/k, k 42 — 시뮬 §8a 주입과 같은 모형)과 τ 변동(피치 비례)을 넣는다.
참값: 골반 수평 속도(왼 발바닥점 고정 → 몸통 위치 미분), 움직임 끝 시각.

판정기 = stand_c5kr KE 감지기(rl_bridge_node C5_DETECTOR_KW + C5_EXTRA det) 위의 조기 복귀 변형:
  imu(현행) · odic · odicw · odicwN(불응기 제거) · odicwH(유지 0.6 s) · xV/xDS/xMIN/xK/xTAU(odicw 에서 조건 하나 제거).
지표(시나리오별): 끄기 사건(경로 early/calm)·**조기 해제**(끈 뒤 1 s 안 원 |ω_xy| ≥ 1.0 또는 참 속도 ≥ 0.08 m/s)·재켜짐·
**흔들림 노출**(첫 켜짐 뒤 서기 모드인데 감지 |ω_xy| > 0.5 인 시간)·마지막 끄기의 참 정착 대비 지연·끝까지 못 끔.
열린 고리(판정기 출력이 운동을 바꾸지 않음) — 폐루프 효과는 시뮬 A/B 몫. 사용: python3 reports/gen_w3/settle_attack.py
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
from rl_walking.deploy import c5_switch as C5  # noqa: E402
from rl_walking.deploy.leg_odometry import fk_sole  # noqa: E402
from rl_walking.deploy.policy_runner import JOINT_ORDER  # noqa: E402

_spec = importlib.util.spec_from_file_location("_bridge_for_attack", os.path.join(REPO, "jetson", "scripts", "rl_bridge_node.py"))
B = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(B)
DET_KW = dict(B.C5_DETECTOR_KW, **B.C5_EXTRA["stand_c5kr_g21a_g24ra"]["det"])

DT = 0.02
# 1003 G21a 조용한 서기(210.5 s) 관절각 [deg] — reports/real_1003/rl_bridge_20261003_174906.jsonl 에서 읽은 값(재현성 위해 상수)
Q0 = np.radians([28.0, -27.7, 0.2, 0.5, 0.7, -0.2, -38.9, 38.3, 26.6, -26.2, -1.4, -0.8])
TAU0 = np.array([3.9, -3.4])          # 무릎 서기 토크(관절 프레임, §10 재생 측정 중앙 근처)
K_TRUE = 42.0
IDX = {s: [JOINT_ORDER.index(f"{s}_{j}_joint") for j in ("hip_f", "hip_a", "hip_r", "knee", "ankle_f", "ankle_r")] for s in ("left", "right")}
IAF = {s: JOINT_ORDER.index(f"{s}_ankle_f_joint") for s in ("left", "right")}
IHF = {s: JOINT_ORDER.index(f"{s}_hip_f_joint") for s in ("left", "right")}
IK = [JOINT_ORDER.index("left_knee_joint"), JOINT_ORDER.index("right_knee_joint")]


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _sole(q, s):
    p, R = fk_sole(q[IDX[s]][None], s)
    return p[0], R[0]


def _calib():
    """부호: (1) 피치 θ 를 발목 앞뒤가 흡수(발 자세 불변), (2) 병진 α 를 힙·발목이 반대로(몸통 자세·발 자세 불변). 수치 탐색."""
    n0 = {s: _sole(Q0, s)[1][:, 1] for s in ("left", "right")}
    phi0 = math.atan2(-n0["left"][0], n0["left"][2])          # 몸통 기본 피치(발 법선이 월드 z 가 되게)
    sig = {}
    for s in ("left", "right"):
        R0 = Ry(phi0) @ _sole(Q0, s)[1]
        best = None
        for sg in (1, -1):
            q = Q0.copy(); q[IAF[s]] += sg * 0.05
            e = np.abs(Ry(phi0 + 0.05) @ _sole(q, s)[1] - R0).max()
            best = (e, sg) if best is None or e < best[0] else best
        sig[s] = best[1]
    tr = {}
    for s in ("left", "right"):
        R0 = _sole(Q0, s)[1]; p0 = _sole(Q0, s)[0]
        best = None
        for sh in (1, -1):
            for sa in (1, -1):
                q = Q0.copy(); q[IHF[s]] += sh * 0.05; q[IAF[s]] += sa * 0.05
                p, R = _sole(q, s)
                e = np.abs(R - R0).max() + abs(p[1] - p0[1]) + abs(p[2] - p0[2]) * 0.0
                if (p[0] - p0[0]) < 0 and (best is None or e < best[0]):      # α > 0 → 발이 몸통 뒤로 = 골반이 앞으로
                    best = (e, sh, sa)
        tr[s] = best[1:]
    return phi0, sig, tr


PHI_FK, SIG, TR = _calib()
# 몸통 월드 피치는 실측 서기값(1003 210.5 s 보정 중력 피치 1.9°)을 쓴다 — FK 상 발 평평 자세(15.7°)와 다른 것은 실물 무릎 처짐·발목 유격(0919)
# 때문이며, 감지기 상대 기울기 기준(±8° 상한) 때문에 절대값이 중요하다. 발 고정은 회전 합성이라 φ0 와 무관하게 성립(생성기 점검 오른 발 < 1 mm).
PHI0 = math.radians(1.9)
VX_GAIN = 1.0


def envelope(t, t0, t1, ramp=0.15):
    a = np.clip((t - t0) / ramp, 0, 1) * np.clip((t1 - t) / ramp, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * a)


class Scenario:
    """시간 배열 위의 피치 각속도 θ̇(t)·병진 속도 ẋ(t)·접촉·τ 결측 마스크. 운동학은 make() 가 적분해 만든다."""

    def __init__(self, name, T, seed=0):
        self.name, self.T = name, T
        self.t = np.arange(int(round(T / DT))) * DT
        n = len(self.t)
        self.th = np.zeros(n); self.vx = np.zeros(n); self.wx_ratio = 0.2
        self.contact = np.ones((n, 2)); self.tau_ok = np.ones(n, bool); self.gyro_bias = np.zeros(3)
        self.rng = np.random.default_rng(seed)

    def burst(self, t0, dur, f, peak):
        """θ = (peak/2πf) · env · sin(2πf(t − t0)) — 버스트 뒤 자세가 제자리로 돌아온다(순변위 0). θ̇ 피크 ≈ peak."""
        e = envelope(self.t, t0, t0 + dur)
        self.th += peak / (2 * np.pi * f) * e * np.sin(2 * np.pi * f * (self.t - t0))
        return self

    def drift(self, t0, dur, v):
        e = envelope(self.t, t0, t0 + dur, ramp=0.3)
        self.vx += v * e
        return self

    def make(self):
        n = len(self.t)
        th = self.th; self.wy = np.gradient(th, DT); al = np.zeros(n)
        L = {}
        q = np.tile(Q0, (n, 1))
        # 병진: 다리 강체 기울기 α — 골반 속도 ≈ L_leg · α̇ 를 1 차로 맞춘 뒤 실제 속도는 FK 로 다시 잰다
        Lleg = float(np.linalg.norm(_sole(Q0, "left")[0] - np.array([0.0, 0.0, 0.0])))
        al = np.cumsum(self.vx) * DT / max(Lleg, 1e-6) * VX_GAIN
        for s in ("left", "right"):
            q[:, IAF[s]] += SIG[s] * th
            sh, sa = TR[s]
            q[:, IHF[s]] += sh * al; q[:, IAF[s]] += sa * al
        R = np.array([Ry(PHI0 + a) for a in th])
        pL = np.array([_sole(q[k], "left")[0] for k in range(n)])
        pb = -np.einsum("nij,nj->ni", R, pL)          # 왼 발바닥점 월드 원점 고정 → 몸통 위치
        vtrue = np.zeros(n); vtrue[1:] = np.linalg.norm(np.diff(pb[:, :2], axis=0), axis=1) / DT
        ww = np.stack([self.wx_ratio * self.wy, self.wy, np.zeros(n)], 1)        # 월드 각속도(롤 성분 = 0.2 배 동위상 — 1003 측정)
        gyro = np.einsum("nji,nj->ni", R, ww) + self.gyro_bias + self.rng.normal(0, 0.01, (n, 3))
        grav = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, -1.0]))
        tau = TAU0[None, :] * (1.0 + 6.7 * th[:, None]) + self.rng.normal(0, 0.1, (n, 2))     # 피치 0.12 rad → ±0.8 배(실측 회복 중 폭 5–8 N·m 급)
        enc = q.copy(); enc[:, IK] += (tau - TAU0[None, :]) / K_TRUE                           # 엔코더가 보는 처짐(§8a 주입 모형)
        moving = (np.abs(self.wy) > 0.05) | (np.abs(self.vx) > 0.01)
        self.t_end = float(self.t[np.where(moving)[0][-1]]) + DT if moving.any() else 0.0
        self.data = dict(gyro=gyro, grav=grav, enc=enc, tau=tau, vtrue=vtrue, wraw=np.linalg.norm(ww[:, :2], axis=1))
        return self


class NoTauBlock(C5.OdicSettle):
    """xTAU: τ 결측 차단 제거 — τ 없음이어도 판정 가능(마지막 유효 τ 유지는 그대로)."""

    def update(self, *a, **k):
        super().update(*a, **k)
        self.tau_ok = True


def make_detector(arm):
    kw = dict(DET_KW)
    if arm == "imu":
        return C5.SwitchDetector(**kw)
    if arm == "imuT":      # §13e 대조: 현행 imu 조기 복귀를 늦게만(켜진 뒤 최소 0.6 → 1.1 s)
        kw["early"] = (1.1, 6.0, 0.4, 0.2)
        return C5.SwitchDetector(**kw)
    st = dict(C5.ODICW_DEFAULT)
    cls = C5.OdicSettle; post = {}
    if arm == "odic":
        st["w_thr"] = 0.0
    elif arm == "odicwN":
        kw["refrac"] = None
    elif arm == "odicwH":
        kw["early"] = (0.6, 6.0, 0.4, 0.6)
    elif arm == "xV":
        post["v_thr"] = 1e9
    elif arm == "xDS":
        st["ds_s"] = 0.0
    elif arm == "xMIN":
        st["min_s"] = 0.0
    elif arm == "xK":
        st["k_hat"] = 0.0
    elif arm == "xTAU":
        cls = NoTauBlock
    elif arm != "odicw":
        raise ValueError(arm)
    s = cls(dt=DT, **st)
    for k_, v_ in post.items():
        setattr(s, k_, v_)
    return C5.SwitchDetector(**kw, settle=s)


def run(sc, arm):
    det = make_detector(arm); D = sc.data; n = len(sc.t)
    mode = np.zeros(n, int); wf = np.zeros(n); ev = []; ne = 0
    for k in range(n):
        tk = tuple(float(x) for x in D["tau"][k]) if sc.tau_ok[k] else None
        if arm == "imu":
            m = det.update(D["grav"][k], D["gyro"][k], sc.contact[k])
        else:
            m = det.update(D["grav"][k], D["gyro"][k], sc.contact[k], qpos_rad=D["enc"][k], tau_knee_nm=tk)
        wf[k] = float(np.linalg.norm(np.asarray(det.last_w)[:2])) if hasattr(det, "last_w") else 0.0
        if k and mode[k - 1] == 1 and m == 0:
            ev.append((k, "early" if det.n_early > ne else "calm"))
        ne = det.n_early; mode[k] = m
    return mode, wf, ev


def metrics(sc, mode, wf, ev):
    D = sc.data; t = sc.t; n = len(t); W = int(1.0 / DT)
    prem = 0
    for (k, why) in ev:
        nxt = slice(k, min(n, k + W))
        if D["wraw"][nxt].max() >= 1.0 or D["vtrue"][nxt].max() >= 0.08:
            prem += 1
    on_k = np.where(mode == 1)[0]
    expo = 0.0
    if len(on_k):
        after = np.arange(n) >= on_k[0]
        expo = float(((mode == 0) & after & (wf > 0.5)).sum() * DT)
    n_on = int(((mode[1:] == 1) & (mode[:-1] == 0)).sum() + (mode[0] == 1))
    stuck = bool(mode[-1] == 1)
    last_off = t[ev[-1][0]] if ev and not stuck else None
    lat = None if last_off is None else last_off - sc.t_end
    return dict(n_on=n_on, n_off=len(ev), prem=prem, expo=expo, stuck=stuck, lat=lat,
                paths="".join("e" if w == "early" else "c" for _, w in ev), rec_s=float(mode.sum() * DT))


ARMS = ["imu", "imuT", "odic", "odicw", "odicwN", "odicwH", "xV", "xDS", "xMIN", "xK", "xTAU"]


def scenarios():
    S = []
    # A 1003형: 3.3 Hz·피크 2.5 rad/s·버스트 1.5 s × 3, 틈 g, 끝 6 s 정숙(calm 경로 여유)
    for g in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5):
        sc = Scenario(f"A 틈{g:.1f}s", 3.0 + 3 * 1.5 + 2 * g + 6.0, seed=1)
        t0 = 3.0
        for _ in range(3):
            sc.burst(t0, 1.5, 3.3, 2.5); t0 += 1.5 + g
        S.append(sc)
    # B 주기·진폭 × 틈 0.6
    for f in (2.0, 3.3, 5.0):
        for pk in (0.8, 1.5, 3.0):
            sc = Scenario(f"B {f:.1f}Hz·{pk:.1f}", 3.0 + 3 * 1.5 + 2 * 0.6 + 6.0, seed=2)
            t0 = 3.0
            for _ in range(3):
                sc.burst(t0, 1.5, f, pk); t0 += 2.1
            S.append(sc)
    # C 경계 정착: 강한 버스트(켜짐 유발) 뒤 지속 저진폭 진동 — 감지 필터 |ω| 정상 피크 = 0.4 × 0.9 / 1.1 (아래 _scale_for_wf 로 맞춤)
    for r in (0.9, 1.1):
        sc = Scenario(f"C 경계 ×{r:.1f}", 3.0 + 1.5 + 10.0 + 6.0, seed=3)
        sc.burst(3.0, 1.5, 3.3, 2.5)
        sc._cont = (4.4, 10.0, 3.3, _scale_for_wf(3.3, 0.4 * r))
        sc.burst(*sc._cont)
        S.append(sc)
    # D 긴 체류: 버스트 뒤 필터 피크 0.45 지속 20 s
    sc = Scenario("D 긴 체류 0.45", 3.0 + 1.5 + 20.0 + 6.0, seed=4)
    sc.burst(3.0, 1.5, 3.3, 2.5); sc.burst(4.4, 20.0, 3.3, _scale_for_wf(3.3, 0.45)); S.append(sc)
    # E 거짓 정착: 버스트(켜짐) 도중 시작한 회전 없는 병진 v 가 버스트 뒤 3 s 더 지속(3.2–7.5 s)
    for v in (0.08, 0.10, 0.15):
        sc = Scenario(f"E 병진 {v:.2f}", 3.0 + 1.5 + 3.0 + 6.0, seed=5)
        sc.burst(3.0, 1.5, 3.3, 2.5); sc.drift(3.2, 4.3, v); S.append(sc)
    # F 늦은 정착: 버스트 뒤 참 정숙 + 신호 결함
    for kind in ("깨끗", "왼패드 깜빡 10%", "τ 결측 10%", "자이로 편향 0.03", "왼패드 0 고착", "τ 결측 연속 2 s"):
        sc = Scenario(f"F {kind}", 3.0 + 1.5 + 8.0, seed=6)
        sc.burst(3.0, 1.5, 3.3, 2.5)
        m = sc.t >= 4.5
        if kind == "왼패드 깜빡 10%":
            sc.contact[m, 0] = (sc.rng.random(m.sum()) > 0.1).astype(float)
        elif kind == "τ 결측 10%":
            sc.tau_ok[m] = sc.rng.random(m.sum()) > 0.1
        elif kind == "자이로 편향 0.03":
            sc.gyro_bias = np.array([0.03, 0.03, 0.0])
        elif kind == "왼패드 0 고착":
            sc.contact[m, 0] = 0.0
        elif kind == "τ 결측 연속 2 s":
            sc.tau_ok[(sc.t >= 4.5) & (sc.t < 6.5)] = False
        S.append(sc)
    return [s.make() for s in S]


def _vx_gain():
    """병진 1 차 근사(L_leg) 보정: 0.1 m/s 명령의 정상 구간 참 속도가 0.1 이 되게 하는 배율."""
    sc = Scenario("cal", 6.0).drift(1.0, 4.0, 0.1).make()
    m = (sc.t > 2.0) & (sc.t < 4.5)
    return 0.1 / float(sc.data["vtrue"][m].mean())


VX_GAIN = _vx_gain()


def _scale_for_wf(f, target):
    """감지 EMA(α 0.33, 50 Hz) 뒤 |ω_xy| 정상 피크가 target 이 되는 원 피크 ω_y (롤 0.2 배 포함). 수치로 계산."""
    a = 0.33; t = np.arange(0, 6, DT); x = np.sin(2 * np.pi * f * t)
    y = np.zeros_like(x)
    for k in range(1, len(x)):
        y[k] = y[k - 1] + a * (x[k] - y[k - 1])
    pk = np.abs(y[len(y) // 2:]).max() * math.hypot(1.0, 0.2)
    return target / pk


def main():
    out = []
    P = lambda s="": (out.append(s), print(s, flush=True))   # noqa: E731
    P("# §13a 합성 공격 — KE 회복 종료 판정기 단독(배포 c5_switch, 열린 고리) — 진단, 폐루프 성능 아님")
    P(f"생성기 보정: 몸통 월드 피치 {math.degrees(PHI0):.1f}°(FK 발 평평 기준 {math.degrees(PHI_FK):.1f}°) · 발목 흡수 부호 {SIG} · 병진(힙, 발목) 부호 {TR}")
    SC = scenarios()
    # 생성기 자기 점검: 오른 발바닥점 월드 고정 오차(왼 고정 가정이 오른에도 성립하는지)
    P("칸: 켜짐 수/끄기 경로(e 조기·c calm) · 조기 해제 수(끈 뒤 1 s 원|ω|≥1 또는 참속도≥0.08) · 흔들림 노출 s(서기 모드 & 감지|ω|>0.5) · "
      "마지막 끄기 지연(참 움직임 끝 대비 s, ∞ = 끝까지 못 끔) · 회복 모드 총 s")
    R = {}
    for sc in SC:
        P(f"\n## {sc.name} — 길이 {sc.T:.1f} s · 움직임 끝 {sc.t_end:.2f} s · 원|ω| 최대 {sc.data['wraw'].max():.2f} · 참속도 최대 {sc.data['vtrue'].max():.3f}")
        for arm in ARMS:
            mode, wf, ev = run(sc, arm)
            m = metrics(sc, mode, wf, ev); R[(sc.name, arm)] = m
            lat = "∞" if m["stuck"] else ("—" if m["lat"] is None else f"{m['lat']:+.2f}")
            P(f"  {arm:>7}: 켜짐 {m['n_on']} [{m['paths'] or '-'}] · 조기 해제 {m['prem']} · 노출 {m['expo']:.2f} s · 지연 {lat} · 회복 {m['rec_s']:.1f} s")
    # 요약표
    P("\n## 요약 — 시나리오군별 합계(조기 해제 / 노출 s / 못 끔 수 / 지연 중앙)")
    for grp in ("A", "B", "C", "D", "E", "F"):
        names = [sc.name for sc in SC if sc.name.startswith(grp)]
        row = []
        for arm in ARMS:
            ms = [R[(nm, arm)] for nm in names]
            lats = [m["lat"] for m in ms if m["lat"] is not None]
            row.append(f"{arm} {sum(m['prem'] for m in ms)}/{sum(m['expo'] for m in ms):.1f}/{sum(m['stuck'] for m in ms)}/{(np.median(lats) if lats else float('nan')):+.2f}")
        P(f"  {grp}: " + " · ".join(row))
    od = os.path.join(REPO, "reports", "gen_w3", "stopest_results"); os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "settle_attack.txt"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(out) + "\n")
    return R


if __name__ == "__main__":
    main()
