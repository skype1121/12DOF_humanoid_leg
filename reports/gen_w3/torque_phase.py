"""걷기 토크의 위상별 분해 + 정적 하한 + 포화 벌점 항의 값 분포 (CPU·numpy, Isaac 불필요) — TORQUE_W3T_DESIGN.md 의 수치 출처.

사용:
  python3 reports/gen_w3/torque_phase.py                      # X12(명령별 걷기) 전 절
  python3 reports/gen_w3/torque_phase.py --sections phase --seg 15,20
  python3 reports/gen_w3/torque_phase.py --sections scale     # X13(토크 한계 배율별) 위상 비교
  python3 reports/gen_w3/torque_phase.py --lp /다른/log_picture
  python3 reports/gen_w3/torque_phase.py --judge s3t  --prefix tr_s3t16299_x13     # TORQUE_BUDGET_1007.md §10 의 T1·T2·T3 수치와 충족/미달(W3S3t 팔)
  python3 reports/gen_w3/torque_phase.py --judge s3tk --prefix tr_s3tk16299_x13    # 같은 표, W3S3tK 팔(T1 = 지지 무릎 각·무릎 RMS, T3 에 힙 외전 포화 시간)
  python3 reports/gen_w3/torque_phase.py --judge s3t  --prefix tr_s3dctl16193_x13  # 같은 반복 대조(변경 없음)도 같은 명령 — 기준선은 --base(기본 tr_x13)
    (X13 실행 이름 = <prefix>_d2_s<seed>: transall_·kwloop_ npz 와 ground_recovery_…_envs.csv 를 읽는다. 판정 기준의 정본은 TORQUE_BUDGET_1007.md §10.)

입력: log_picture/transall_<TAG>.npz (eval_transition_kwx.py --trace_all: t·tq(적용 토크)·q·cmd·roll·pitch·u(위상)·ge·alive·h·flat …),
      X13 은 kwloop_<TAG>.npz 의 effort_scale(env 별 한계 배율)도 읽는다. 자산 = rl_walking/assets/biped12_final3.urdf.

절:
  model  URDF 정역학 모델 점검(질량·KE 서기 무릎 토크 대조)
  phase  반주기 위상 v = frac(2u) 별 지지/스윙 다리 관절 토크·포화·정적 요구·몸통 롤·골반 높이
  decomp 평균제곱 토크 = 주기 프로파일 몫 + 위상 내 분산 몫, 지지/스윙 몫, 추종 오차(kp 항)
  steps  스텝(반주기)별 집계와 상관: 지지 힙 외전 토크 대 몸통 롤·지지발 옆 거리·무릎 굽힘
  cmds   명령(대본 구간)별 요약
  sat    torque_saturation 항(관절 평균 clamp(|τ|/한계 − thresh, 0)²)의 걷기 중 값 분포와 가중별 초당 크기
  knee   W3S3tK 의 항(걷는 중 접지 다리 무릎 굽힘의 문턱 초과분[rad], 접지 다리 평균)의 현 걷기 값 분포 — 접지는 운동학 판정
  scale  X13: 한계 배율 1.0 / 0.8 / 0.7 에서 같은 정책의 위상 프로파일
  bounds 모델만: 지지 힙 외전 정적 모멘트의 구성·골반 롤/스윙 다리/힙 굴곡 의존, 무릎 굽힘별 정적 토크, 걸음 시간표별 하한

규약: 관절 순서 LhF,RhF,LhA,RhA,LhR,RhR,Lkn,Rkn,LaF,RaF,LaR,RaR. roll·pitch = 평가기 규약(+ = 왼/앞 기울임, 참 중력 기준).
정적 유지 토크(중력만): 지지 다리 관절 = 관절 위쪽 전부를 받침, 스윙 다리 관절 = 관절 아래 링크를 듦. 가속·충격·마찰은 포함하지 않는다.
지지/스윙 구분은 **예정 위상**(u < 0.5 = 왼발 스윙)과 운동학(순기구학 발 높이 차)으로 한다 — 추적의 cont 는 정책 관측용 접촉(실물 계약: 10 Hz 유지·평평 조건)이라
물리 접촉과 최대 0.1 s 어긋난다. 시뮬 결과이며 실물 토크가 아니다(TORQUE_BUDGET_1007.md 머리말의 눈금 차 주의).
"""
import argparse
import copy
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
G = 9.81
DT = 0.02
JN = ["left_hip_f_joint", "right_hip_f_joint", "left_hip_a_joint", "right_hip_a_joint", "left_hip_r_joint", "right_hip_r_joint",
      "left_knee_joint", "right_knee_joint", "left_ankle_f_joint", "right_ankle_f_joint", "left_ankle_r_joint", "right_ankle_r_joint"]
SHORT = ["LhF", "RhF", "LhA", "RhA", "LhR", "RhR", "Lkn", "Rkn", "LaF", "RaF", "LaR", "RaR"]
LEG = {"L": [0, 2, 4, 6, 8, 10], "R": [1, 3, 5, 7, 9, 11]}                    # 다리별 색인(hF, hA, hR, kn, aF, aR)
NM = ["hF", "hA", "hR", "kn", "aF", "aR"]
#: 다리별 부호 — 곱하면 '지지(받치는) 방향'이 + 가 된다(한발 지지 자료의 평균 부호로 정함: 왼 hA −·kn +, 오른 hA +·kn −).
SGN = {"L": np.array([+1, -1, -1, +1, +1, +1.0]), "R": np.array([-1, +1, +1, -1, -1, +1.0])}
LIM = np.array([25.0] * 10 + [24.0] * 2)                                       # biped12_cfg: AK70 25 · 발목 롤 AK45 24 N·m
SEG = [("제자리", 3, 8), ("앞 0.03", 9, 14), ("앞 0.06", 15, 20), ("앞 0.10", 21, 26), ("뒤 0.04", 32, 37), ("뒤 0.06", 38, 43),
       ("옆 0.035", 49, 54), ("회전 0.15", 60, 65)]                            # load_by_cmd.py 와 같은 대본 구간(명령 적용 뒤 1 s 제외)
STAND = [(29.5, 31.0), (46.5, 48.0), (57.8, 59.0)]
K = lambda x: int(round(x / DT))                                                # noqa: E731


# ================================================================ URDF 정역학 모델 ================================================================
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])


def _axis_rot(axis, ang):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s = np.sin(ang)[:, None, None]; c = np.cos(ang)[:, None, None]
    return np.eye(3)[None] + s * Kx[None] + (1 - c) * (Kx @ Kx)[None]


class Model:
    """base 프레임 FK·중력 유지 토크(배치). 관절 이름 = child 링크 이름(이 URDF 규약)."""

    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.mass, self.com, self.inertia = {}, {}, {}
        for L in root.findall("link"):
            n = L.get("name"); I = L.find("inertial")
            self.mass[n] = 0.0; self.com[n] = np.zeros(3); self.inertia[n] = np.zeros((3, 3))
            if I is None:
                continue
            o = I.find("origin"); e = I.find("inertia")
            self.mass[n] = float(I.find("mass").get("value"))
            self.com[n] = np.array([float(x) for x in (o.get("xyz") if o is not None else "0 0 0").split()])
            if e is not None:
                xx, xy, xz, yy, yz, zz = (float(e.get(k)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
                self.inertia[n] = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        js = []
        for J in root.findall("joint"):
            o = J.find("origin"); ax = J.find("axis")
            js.append(dict(name=J.get("name"), type=J.get("type"), parent=J.find("parent").get("link"), child=J.find("child").get("link"),
                           xyz=np.array([float(x) for x in o.get("xyz").split()]), R0=_rpy(*[float(x) for x in o.get("rpy").split()]),
                           axis=np.array([float(x) for x in ax.get("xyz").split()]) if ax is not None else np.zeros(3)))
        children = {j["child"] for j in js}
        self.root = [n for n in self.mass if n not in children][0]
        order, done = [], {self.root}
        while js:
            for j in list(js):
                if j["parent"] in done:
                    order.append(j); done.add(j["child"]); js.remove(j)
        self.joints = order
        self.kids = {}
        for j in self.joints:
            self.kids.setdefault(j["parent"], []).append(j["child"])
        self.links = list(self.mass)
        self.total_mass = sum(self.mass.values())

    def subtree(self, link):
        out, st = [], [link]
        while st:
            n = st.pop(); out.append(n); st += self.kids.get(n, [])
        return out

    def fk(self, q):
        """q (M,12)[rad] → 링크 회전 R[n](M,3,3)·원점 p[n](M,3), 관절 축 a[name](M,3)·원점 pj[name](M,3)."""
        q = np.asarray(q, float); M = q.shape[0]
        R = {self.root: np.repeat(np.eye(3)[None], M, 0)}; p = {self.root: np.zeros((M, 3))}
        a, pj = {}, {}
        for j in self.joints:
            Rp, pp = R[j["parent"]], p[j["parent"]]
            pc = pp + Rp @ j["xyz"]; Rc = Rp @ j["R0"]
            if j["type"] in ("continuous", "revolute") and j["name"] in JN:
                Rc = Rc @ _axis_rot(j["axis"], q[:, JN.index(j["name"])])
                a[j["name"]] = Rc @ (j["axis"] / np.linalg.norm(j["axis"])); pj[j["name"]] = pc
            R[j["child"]] = Rc; p[j["child"]] = pc
        return R, p, a, pj

    def coms(self, R, p):
        return {n: p[n] + R[n] @ self.com[n] for n in self.links}

    def hold_torque(self, q, gdir, stance):
        """한발 지지(stance = 'L'|'R')의 중력 유지 토크 τ (M,12)[N·m]. gdir (M,3) = base 프레임 중력 단위벡터.
        지지 다리: τ_j = +a_j·Σ_{k∉subtree(j)} (c_k − p_j)×m_k g · 스윙 다리: τ_j = −a_j·Σ_{k∈subtree(j)} (c_k − p_j)×m_k g."""
        R, p, a, pj = self.fk(q); c = self.coms(R, p)
        g = G * np.asarray(gdir, float)
        tau = np.zeros((q.shape[0], 12))
        for ji, name in enumerate(JN):
            sub = set(self.subtree(name))
            is_st = (name.startswith("left") and stance == "L") or (name.startswith("right") and stance == "R")
            acc = np.zeros((q.shape[0], 3))
            for n in self.links:
                if self.mass[n] <= 0 or (n in sub) == is_st:
                    continue
                acc += np.cross(c[n] - pj[name], self.mass[n] * g)
            t = (a[name] * acc).sum(1)
            tau[:, ji] = t if is_st else -t
        return tau

    def foot_low(self, q, gdir):
        """발별 가장 낮은 패드 높이(중력 반대 방향, base 원점 기준) (M,2) — 왼, 오른."""
        R, p, a, pj = self.fk(q); up = -np.asarray(gdir, float)
        out = []
        for side in ("left", "right"):
            out.append(np.min(np.stack([(p[n] * up).sum(1) for n in self.links if n.startswith(side + "foot_")], 0), 0))
        return np.stack(out, 1)

    def halved(self, keep):
        """양발 대칭 서기 근사용: 반대쪽 다리 질량 0·골반 절반(다리마다 '골반 절반 + 자기 다리'를 받친다고 봄 — 시상면 관절에만 유효)."""
        m2 = copy.deepcopy(self); other = "right" if keep == "L" else "left"
        for n in m2.links:
            if n.startswith(other):
                m2.mass[n] = 0.0
        for n in ("pelvis", "imu"):
            if n in m2.mass:
                m2.mass[n] *= 0.5
        return m2


def gdir_from_rp(roll_deg, pitch_deg):
    """평가기 규약(roll = atan2(pg_y, −pg_z)·pitch = atan2(pg_x, −pg_z)) → base 프레임 중력 단위벡터."""
    v = np.stack([np.tan(np.radians(pitch_deg)), np.tan(np.radians(roll_deg)), -np.ones_like(np.asarray(roll_deg, float))], -1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ================================================================ 자료 ================================================================
def load_seg(lp, tag, a, b, scale=None, want="walk"):
    """구간 [a,b) s 를 잘라 (T,N,…) 사전으로. scale 을 주면 kwloop 의 effort_scale 이 그 값인 env 만. want: 'walk'(ge) / 'stand'(KE 서기)."""
    z = np.load(os.path.join(lp, f"transall_{tag}.npz"), allow_pickle=True)
    n = z["tq"].shape[1]; m = z["pre_pass"].astype(bool)[:n]
    if scale is not None:
        kw = np.load(os.path.join(lp, f"kwloop_{tag}.npz"), allow_pickle=True)
        m = m & (np.abs(kw["effort_scale"].astype(float)[:n] - scale) < 1e-6)
    sl = slice(K(a), K(b))
    d = {k: z[k][sl][:, m].astype(np.float64) for k in ("q", "tq", "cmd", "roll", "pitch", "u", "h", "flat")}
    al = z["alive"][sl] & (z["ge"][sl] if want == "walk" else (z["ke"][sl] & ~z["ge"][sl]))
    d["al"] = al[:, m]
    return d


def cat(ds):
    """같은 구간 길이의 조각을 env 축으로 잇기(X13 seed 합치기)."""
    return {k: np.concatenate([d[k] for d in ds], 1) for k in ds[0]}


def samples(ds, keys=("q", "tq", "roll", "pitch")):
    """길이가 다른 구간 조각들을 유효 표본만 평평하게 이어 붙이기."""
    return {k: np.concatenate([d[k][d["al"]] for d in ds], 0) for k in keys}


def fold(d, model):
    """예정 지지/스윙 다리로 접기(좌우 거울). 반환 사전: v·al·st/sw(토크, 지지 방향 +)·st_th/sw_th(정적 모델)·오차·롤·높이·옆거리·스윙발 높이 차."""
    q, tq, cmd, roll, pitch, u, h, flat, al = (d[k] for k in ("q", "tq", "cmd", "roll", "pitch", "u", "h", "flat", "al"))
    T, N = roll.shape
    g = gdir_from_rp(roll.reshape(-1), pitch.reshape(-1)); qf = q.reshape(-1, 12)
    fl = model.foot_low(qf, g).reshape(T, N, 2)
    thL = model.hold_torque(qf, g, "L").reshape(T, N, 12); thR = model.hold_torque(qf, g, "R").reshape(T, N, 12)
    half = (u >= 0.5).astype(int); v = np.remainder(2.0 * u, 1.0)
    droll = np.gradient(np.radians(roll), DT, axis=0); ddroll = np.gradient(droll, DT, axis=0)
    ddh = np.gradient(np.gradient(h, DT, axis=0), DT, axis=0)
    dq = np.gradient(q, DT, axis=0); dcmd = np.gradient(cmd, DT, axis=0)
    o = {k: np.zeros((T, N, 6)) for k in ("st", "sw", "st_th", "sw_th", "st_err", "st_dq", "st_dcmd")}
    for hv, s_, w_ in ((0, "R", "L"), (1, "L", "R")):
        mk = half == hv; th = thR if s_ == "R" else thL
        o["st"][mk] = (tq[..., LEG[s_]] * SGN[s_])[mk]; o["sw"][mk] = (tq[..., LEG[w_]] * SGN[w_])[mk]
        o["st_th"][mk] = (th[..., LEG[s_]] * SGN[s_])[mk]; o["sw_th"][mk] = (th[..., LEG[w_]] * SGN[w_])[mk]
        o["st_err"][mk] = ((cmd - q)[..., LEG[s_]] * SGN[s_])[mk]; o["st_dq"][mk] = (dq[..., LEG[s_]] * SGN[s_])[mk]
        o["st_dcmd"][mk] = (dcmd[..., LEG[s_]] * SGN[s_])[mk]
    o.update(v=v, half=half, al=al, h=h, ddh=ddh,
             sroll=np.where(half == 1, roll, -roll), sdroll=np.where(half == 1, droll, -droll), sddroll=np.where(half == 1, ddroll, -ddroll),
             slat=np.where(half == 1, flat[..., 0], -flat[..., 1]), swlat=np.where(half == 1, -flat[..., 1], flat[..., 0]),
             dzf=np.where(half == 1, fl[..., 1] - fl[..., 0], fl[..., 0] - fl[..., 1]),
             kn_st=np.degrees(np.where(half == 1, -q[..., 6], q[..., 7])), kn_sw=np.degrees(np.where(half == 1, q[..., 7], -q[..., 6])))
    return o


# ================================================================ 절 ================================================================
def sec_model(model, lp, tag):
    print("\n## [model] URDF 정역학 모델 점검")
    print(f"- 전체 질량 {model.total_mass:.3f} kg · 골반 {model.mass['pelvis']:.3f} kg · 다리 한 쪽 {sum(model.mass[n] for n in model.subtree('left_hip_f_joint')):.3f} kg")
    R, p, a, pj = model.fk(np.zeros((1, 12)))
    print(f"- 힙 외전 관절 좌우 간격 {pj['left_hip_a_joint'][0][1] - pj['right_hip_a_joint'][0][1]:.4f} m · 골반 CoM(base) {np.round(model.coms(R, p)['pelvis'][0], 4)}")
    d = samples([load_seg(lp, tag, a_, b_, want="stand") for a_, b_ in STAND])
    q = d["q"]; tq = d["tq"]; g = gdir_from_rp(d["roll"], d["pitch"])
    print(f"- KE 서기(양발) 표본 {len(q)}: 평균 무릎 굽힘 왼 {-np.degrees(q[:, 6].mean()):.1f}° 오른 {np.degrees(q[:, 7].mean()):.1f}°")
    for side in ("L", "R"):
        th = model.halved(side).hold_torque(q, g, side); kn = LEG[side][3]; hf = LEG[side][0]
        print(f"  {side} 무릎: 모델(골반 절반) {th[:, kn].mean():+.2f} 대 시뮬 {tq[:, kn].mean():+.2f} N·m · 힙 앞뒤: 모델 {th[:, hf].mean():+.2f} 대 시뮬 {tq[:, hf].mean():+.2f}")
    print("  (양발 지지의 좌우면 관절(힙 외전·발목 롤)은 닫힌 사슬이라 이 근사가 맞지 않는다 — 한발 지지에서만 쓴다.)")


def _phase_table(f, lim, nb=10):
    al, v = f["al"], f["v"]
    e = np.linspace(0, 1, nb + 1)
    print("| v | 스윙발 높이 mm · 한발 % | 지지 힙 외전: 평균 · p90 · 포화 % · 정적 요구 | 지지 무릎: 평균 · p90 · 포화 % · 정적 요구 | 지지 힙 회전: 평균 · 정적 | 지지 발목 롤: 평균 · 정적 | 스윙 힙 외전 · 스윙 무릎 | 지지쪽 롤 ° · 롤 속도 rad/s · 롤 가속 | 골반 높이 mm | 지지 무릎 굽힘 ° |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i in range(nb):
        mk = al & (v >= e[i]) & (v < e[i + 1])
        if mk.sum() < 30:
            continue
        hA = f["st"][..., 1][mk]; kn = f["st"][..., 3][mk]
        print(f"| {e[i]:.1f}–{e[i + 1]:.1f} | {1e3 * np.median(f['dzf'][mk]):.0f} · {100 * (f['dzf'][mk] > 0.008).mean():.0f} | "
              f"{hA.mean():.1f} · {np.percentile(np.abs(hA), 90):.1f} · {100 * (np.abs(hA) >= lim - 0.5).mean():.0f} · {f['st_th'][..., 1][mk].mean():.1f} | "
              f"{kn.mean():.1f} · {np.percentile(np.abs(kn), 90):.1f} · {100 * (np.abs(kn) >= lim - 0.5).mean():.0f} · {f['st_th'][..., 3][mk].mean():.1f} | "
              f"{f['st'][..., 2][mk].mean():.1f} · {f['st_th'][..., 2][mk].mean():.1f} | {f['st'][..., 5][mk].mean():.1f} · {f['st_th'][..., 5][mk].mean():.1f} | "
              f"{f['sw'][..., 1][mk].mean():.1f} · {f['sw'][..., 3][mk].mean():.1f} | {f['sroll'][mk].mean():.1f} · {f['sdroll'][mk].mean():+.2f} · {f['sddroll'][mk].mean():+.1f} | "
              f"{1e3 * f['h'][mk].mean():.0f} | {f['kn_st'][mk].mean():.1f} |")


def _summary(f, lim):
    A = f["al"]; st, sw = f["st"], f["sw"]
    r = lambda x: float(np.sqrt((x ** 2).mean()))                              # noqa: E731
    out = {}
    for j, nm in ((1, "hA"), (3, "kn"), (2, "hR")):
        s, w = st[..., j][A], sw[..., j][A]
        out[nm] = dict(st_mean=s.mean(), st_rms=r(s), sw_mean=w.mean(), sw_rms=r(w), mean=0.5 * (s.mean() + w.mean()), rms=float(np.sqrt(0.5 * ((s ** 2).mean() + (w ** 2).mean()))),
                       sat=100 * 0.5 * ((np.abs(s) >= lim - 0.5).mean() + (np.abs(w) >= lim - 0.5).mean()), st_sat=100 * (np.abs(s) >= lim - 0.5).mean(),
                       th=f["st_th"][..., j][A & (f["dzf"] > 0.008)].mean())
    out["ss"] = 100 * (f["dzf"][A] > 0.008).mean(); out["apex"] = 1e3 * np.percentile(f["dzf"][A], 95)
    nb = 23; vb = np.minimum((f["v"] * nb).astype(int), nb - 1)
    prof = np.array([f["sroll"][A & (vb == i)].mean() if (A & (vb == i)).any() else np.nan for i in range(nb)])
    out["roll_pk"] = float(np.nanmax(prof)); out["kn_st"] = f["kn_st"][A & (f["dzf"] > 0.008)].mean(); out["h_rng"] = 1e3 * float(np.ptp([f["h"][A & (vb == i)].mean() for i in range(nb) if (A & (vb == i)).any()]))
    return out


def sec_phase(model, lp, tag, a, b):
    f = fold(load_seg(lp, tag, a, b), model)
    print(f"\n## [phase] {tag} · {a:g}–{b:g} s · 걷기 표본 {int(f['al'].sum())} (env {int(f['al'].any(0).sum())}) — 반주기 위상 v 별(토크 N·m, 지지 방향 +)")
    _phase_table(f, 25.0)
    s = _summary(f, 25.0)
    print(f"- 한발 지지(운동학: 스윙발이 8 mm 넘게 높음) {s['ss']:.1f} % · 스윙발 높이 p95 {s['apex']:.0f} mm · 지지쪽 몸통 롤 최대(위상 평균) {s['roll_pk']:.1f}° · 골반 높이 변동(위상 평균) {s['h_rng']:.0f} mm · 한발 지지 중 지지 무릎 {s['kn_st']:.1f}°")
    for nm, lab in (("hA", "힙 외전"), ("kn", "무릎"), ("hR", "힙 회전")):
        x = s[nm]
        print(f"- {lab}: 지지 반주기 평균 {x['st_mean']:.1f}·RMS {x['st_rms']:.1f}(포화 {x['st_sat']:.1f} %) · 스윙 반주기 평균 {x['sw_mean']:.1f}·RMS {x['sw_rms']:.1f} → 한 관절 보폭 평균 {x['mean']:.1f}·RMS {x['rms']:.1f} · 한발 지지 정적 요구 평균 {x['th']:.1f}")
    return f


def sec_decomp(f):
    al = f["al"]; nb = 23; vb = np.minimum((f["v"] * nb).astype(int), nb - 1)
    print("\n## [decomp] 평균제곱 토크 분해 — E[τ²] = 주기 프로파일 Σμ(v)² + 위상 내 분산 Σσ²(v) (지지·스윙 반주기 합친 한 관절)")
    print("| 관절 | RMS | 주기 프로파일 몫 % (그 RMS) | 위상 내 분산 몫 % (그 RMS) | 평균제곱에서 지지 반주기 몫 % |"); print("|---|---|---|---|---|")
    for j, nm in enumerate(NM):
        mu2 = s2 = n = 0.0; st2 = (f["st"][..., j][al] ** 2).mean(); sw2 = (f["sw"][..., j][al] ** 2).mean()
        for part in (f["st"], f["sw"]):
            for i in range(nb):
                x = part[..., j][al & (vb == i)]
                if len(x):
                    mu2 += len(x) * x.mean() ** 2; s2 += len(x) * x.var(); n += len(x)
        mu2 /= n; s2 /= n
        print(f"| {nm} | {np.sqrt(mu2 + s2):.1f} | {100 * mu2 / (mu2 + s2):.1f} ({np.sqrt(mu2):.1f}) | {100 * s2 / (mu2 + s2):.1f} ({np.sqrt(s2):.1f}) | {100 * st2 / (st2 + sw2):.1f} |")
    print("\n지지 다리 추종 오차(명령 − 실측, 지지 방향 +)와 PD 항 — kp 150·kd 5 (적용 토크는 한계에서 잘림)")
    print("| v | 힙 외전: 오차 ° · kp·오차 · −kd·속도 · 적용 τ · 명령 속도 p90 rad/s | 무릎: 오차 ° · kp·오차 · −kd·속도 · 적용 τ · 명령 속도 p90 |"); print("|---|---|---|")
    e = np.linspace(0, 1, 11)
    for i in range(10):
        mk = al & (f["v"] >= e[i]) & (f["v"] < e[i + 1]); cells = []
        for j in (1, 3):
            er = f["st_err"][..., j][mk]; dv = f["st_dq"][..., j][mk]
            cells.append(f"{np.degrees(er.mean()):.1f} · {150 * er.mean():.1f} · {-5 * dv.mean():+.1f} · {f['st'][..., j][mk].mean():.1f} · {np.percentile(np.abs(f['st_dcmd'][..., j][mk]), 90):.2f}")
        print(f"| {e[i]:.1f}–{e[i + 1]:.1f} | {cells[0]} | {cells[1]} |")


def sec_steps(f):
    al, v, half = f["al"], f["v"], f["half"]; T, N = v.shape; rows = []
    for e in range(N):
        cuts = np.where(np.diff(half[:, e]) != 0)[0] + 1
        for s in np.split(np.arange(T), cuts):
            if len(s) < 18 or len(s) > 27 or not al[s, e].all():
                continue
            vv = v[s, e]; mid = (vv >= 0.1) & (vv < 0.9); hA = f["st"][s, e, 1]; kn = f["st"][s, e, 3]
            rows.append([hA[mid].mean(), np.sqrt((hA ** 2).mean()), (hA >= 24.5).mean(), f["sroll"][s, e].max(), f["slat"][s, e][mid].mean(), f["swlat"][s, e].max(),
                         kn[mid].mean(), np.sqrt((kn ** 2).mean()), (kn >= 24.5).mean(), f["kn_st"][s, e][mid].mean(), f["h"][s, e].max() - f["h"][s, e].min(), f["dzf"][s, e].max()])
    R = np.array(rows)
    cols = ["지지 힙 외전 평균(v .1–.9)", "지지 힙 외전 RMS", "힙 외전 포화 틱 비율", "지지쪽 몸통 롤 최대 °", "지지발–골반 옆 거리 m", "스윙발–골반 옆 거리 최대 m",
            "지지 무릎 평균(v .1–.9)", "지지 무릎 RMS", "무릎 포화 틱 비율", "지지 무릎 굽힘 °", "골반 높이 변동 m", "스윙발 높이 최대 m"]
    print(f"\n## [steps] 스텝(반주기) {len(R)} 개 집계 — 중앙 [p10, p90]")
    for k, c in enumerate(cols):
        print(f"- {c}: {np.median(R[:, k]):.3f} [{np.percentile(R[:, k], 10):.3f}, {np.percentile(R[:, k], 90):.3f}]")
    cc = lambda i, j: float(np.corrcoef(R[:, i], R[:, j])[0, 1])               # noqa: E731
    print("- 스텝 사이 상관(Pearson r): 지지 힙 외전 평균 대 — 몸통 롤 최대 {:+.2f} · 지지발 옆 거리 {:+.2f} · 스윙발 옆 거리 최대 {:+.2f} · 스윙발 높이 {:+.2f}".format(cc(0, 3), cc(0, 4), cc(0, 5), cc(0, 11)))
    print("  지지 무릎 평균 대 — 지지 무릎 굽힘 {:+.2f} · 몸통 롤 최대 {:+.2f} · 골반 높이 변동 {:+.2f}".format(cc(6, 9), cc(6, 3), cc(6, 10)))
    for k, lab, tgt in ((3, "지지쪽 몸통 롤 최대 °", (0, 1, 2)), (4, "지지발–골반 옆 거리 m", (0, 1, 2)), (9, "지지 무릎 굽힘 °", (6, 7, 8))):
        qs = np.percentile(R[:, k], [0, 33.3, 66.7, 100]); cells = []
        for i in range(3):
            mk = (R[:, k] >= qs[i]) & (R[:, k] <= qs[i + 1])
            cells.append(f"{qs[i]:.3g}–{qs[i + 1]:.3g}: 평균 {R[mk, tgt[0]].mean():.1f}·RMS {R[mk, tgt[1]].mean():.1f}·포화 {100 * R[mk, tgt[2]].mean():.0f} %")
        print(f"- {lab} 삼분위별 {'힙 외전' if tgt[0] == 0 else '무릎'}: " + " | ".join(cells))
    sl = np.polyfit(R[:, 3], R[:, 0], 1)
    print(f"- 직선 맞춤: 지지 힙 외전 평균 = {sl[1]:.1f} {sl[0]:+.2f}·(몸통 롤 최대 °) N·m")


def sec_cmds(model, lp, tag):
    print(f"\n## [cmds] {tag} — 명령별 요약(걷기 틱)")
    print("| 명령 | 표본 | 한발 지지 % · 스윙발 p95 mm · 몸통 롤 최대 ° · 지지 무릎 ° | 힙 외전: 지지 평균·RMS·포화 % → 보폭 평균·RMS | 무릎: 지지 평균·RMS·포화 % → 보폭 평균·RMS | 한발 지지 정적 요구: 힙 외전 · 무릎 |"); print("|---|---|---|---|---|---|")
    for nm, a, b in SEG:
        f = fold(load_seg(lp, tag, a, b), model); s = _summary(f, 25.0); hA, kn = s["hA"], s["kn"]
        print(f"| {nm} | {int(f['al'].sum())} | {s['ss']:.0f} · {s['apex']:.0f} · {s['roll_pk']:.1f} · {s['kn_st']:.1f} | {hA['st_mean']:.1f}·{hA['st_rms']:.1f}·{hA['st_sat']:.0f} → {hA['mean']:.1f}·{hA['rms']:.1f} | "
              f"{kn['st_mean']:.1f}·{kn['st_rms']:.1f}·{kn['st_sat']:.0f} → {kn['mean']:.1f}·{kn['rms']:.1f} | {hA['th']:.1f} · {kn['th']:.1f} |")


def sat_term(tq, lim, thresh):
    """mdp_stage4.torque_saturation 과 같은 식: 관절 평균 clamp(|τ|/(한계 + 1e-6) − thresh, 0)²."""
    return (np.clip(np.abs(tq) / (lim + 1e-6) - thresh, 0.0, None) ** 2).mean(-1)


def sec_sat(lp, tag, x13_tags):
    print(f"\n## [sat] torque_saturation 값(가중 전)의 걷기 중 분포 — {tag}(한계 1.0 배)")
    ds = {nm: load_seg(lp, tag, a, b) for nm, a, b in SEG}
    walk = np.concatenate([d["tq"][d["al"]] for d in ds.values()])
    stand = samples([load_seg(lp, tag, a, b, want="stand") for a, b in STAND], keys=("tq",))["tq"]
    print("| thresh (N·m) | 걷기 전체: 평균 · 중앙 · p90 · p99 | 명령별 평균(최소–최대) | 관절 기여: 힙 외전 L+R · 무릎 L+R · 그 밖 | KE 서기 평균 |"); print("|---|---|---|---|---|")
    for th in (0.6, 0.5, 0.4, 0.33):
        v = sat_term(walk, LIM, th); per = [sat_term(d["tq"][d["al"]], LIM, th).mean() for d in ds.values()]
        pj = (np.clip(np.abs(walk) / (LIM + 1e-6) - th, 0, None) ** 2).mean(0); tot = pj.sum()
        print(f"| {th:.2f} ({th * 25:.1f}) | {v.mean():.4f} · {np.median(v):.4f} · {np.percentile(v, 90):.4f} · {np.percentile(v, 99):.4f} | {min(per):.4f}–{max(per):.4f} | "
              f"{100 * pj[[2, 3]].sum() / tot:.0f} % · {100 * pj[[6, 7]].sum() / tot:.0f} % · {100 * (tot - pj[[2, 3, 6, 7]].sum()) / tot:.0f} % | {sat_term(stand, LIM, th).mean():.5f} |")
    v6 = sat_term(walk, LIM, 0.6).mean()
    print(f"- thresh 0.6 기준 가중별 걷기 초당 크기(= 가중 × 평균 {v6:.4f}): " + " · ".join(f"−{w:g} → {-w * v6:.3f}" for w in (2, 10, 20, 40, 60, 80, 120)))
    print("\n관절별 |τ| [N·m] (걷기 전체)")
    print("| 관절 | RMS | 평균 | p50 | p90 | p99 | ≥ 15 % | ≥ 17.5 % | ≥ 20 % | ≥ 한계−0.5 % |"); print("|---|---|---|---|---|---|---|---|---|---|")
    A = np.abs(walk)
    for j in range(12):
        x = A[:, j]
        print(f"| {SHORT[j]} | {np.sqrt((x ** 2).mean()):.1f} | {x.mean():.1f} | {np.median(x):.1f} | {np.percentile(x, 90):.1f} | {np.percentile(x, 99):.1f} | "
              f"{100 * (x >= 15).mean():.1f} | {100 * (x >= 17.5).mean():.1f} | {100 * (x >= 20).mean():.1f} | {100 * (x >= LIM[j] - 0.5).mean():.1f} |")
    have = [t for t in x13_tags if os.path.exists(os.path.join(lp, f"transall_{t}.npz")) and os.path.exists(os.path.join(lp, f"kwloop_{t}.npz"))]
    if have:
        print("\nX13(한계 배율별, 제자리 3–8 s·앞 0.06 9–16 s, 낙상 전 틱) — thresh 0.6")
        print("| 배율 | 표본 | 그 env 한계 기준 평균 | 명목 한계(25/24) 기준 평균 |"); print("|---|---|---|---|")
        for g in (1.0, 0.8, 0.7, 0.6):
            tq = np.concatenate([d["tq"][d["al"]] for t in have for d in (load_seg(lp, t, 3, 8, scale=g), load_seg(lp, t, 9, 16, scale=g))])
            if len(tq) >= 100:
                print(f"| {g:g} | {len(tq)} | {sat_term(tq, LIM * g, 0.6).mean():.4f} | {sat_term(tq, LIM, 0.6).mean():.4f} |")


def sec_knee(model, lp, tag):
    print(f"\n## [knee] {tag} — w3_stance_knee_over 값(가중 전, rad): 접지 다리 무릎 굽힘의 문턱 초과분(상한 0.6), 접지 다리 평균. 접지 = 예정 지지발 + (예정 스윙발이 8 mm 이내)")
    print("| 명령 | 한발 지지 중 지지 무릎 ° 중앙 [p10, p90] · 접지 중 스윙 다리 무릎 ° · 공중 스윙 다리 무릎 ° | 문턱 40°: 평균 | 문턱 45°: 평균 [p10, p90] | 문턱 50°: 평균 |"); print("|---|---|---|---|---|")
    acc = {40.0: [], 45.0: [], 50.0: []}
    for nm, a, b in SEG:
        f = fold(load_seg(lp, tag, a, b), model); al = f["al"]
        st = np.radians(f["kn_st"]); sw = np.radians(f["kn_sw"]); swc = (f["dzf"] <= 0.008).astype(float); ss = al & (swc < 0.5)
        v = {k0: ((np.clip(st - np.radians(k0), 0, 0.6) + np.clip(sw - np.radians(k0), 0, 0.6) * swc) / (1.0 + swc))[al] for k0 in acc}
        for k0 in acc:
            acc[k0].append(v[k0].mean())
        print(f"| {nm} | {np.median(f['kn_st'][ss]):.1f} [{np.percentile(f['kn_st'][ss], 10):.1f}, {np.percentile(f['kn_st'][ss], 90):.1f}] · {np.median(f['kn_sw'][al & (swc > 0.5)]):.1f} · {np.median(f['kn_sw'][ss]):.1f} | "
              f"{v[40.0].mean():.3f} | {v[45.0].mean():.3f} [{np.percentile(v[45.0], 10):.3f}, {np.percentile(v[45.0], 90):.3f}] | {v[50.0].mean():.3f} |")
    for k0 in acc:
        m_ = float(np.mean(acc[k0]))
        print(f"- 문턱 {k0:.0f}°: 명령 평균 {m_:.3f} rad({np.degrees(m_):.1f}°) → 가중 −1 = 걷기 초당 {-m_:.3f} · −2 = {-2 * m_:.3f}")


def sec_scale(model, lp, x13_tags, a, b):
    have = [t for t in x13_tags if os.path.exists(os.path.join(lp, f"transall_{t}.npz")) and os.path.exists(os.path.join(lp, f"kwloop_{t}.npz"))]
    if not have:
        print("\n## [scale] X13 추적 없음 —", x13_tags); return
    for g in (1.0, 0.8, 0.7):
        d = cat([load_seg(lp, t, a, b, scale=g) for t in have])
        print(f"\n## [scale] X13 배율 {g:g}(한계 {25 * g:.1f} N·m) · {a:g}–{b:g} s · env {d['al'].shape[1]} · 걷기 표본 {int(d['al'].sum())} (낙상한 env 는 낙상 전 틱만)")
        if d["al"].sum() < 500:
            print("- 표본 부족"); continue
        f = fold(d, model); _phase_table(f, 25.0 * g); s = _summary(f, 25.0 * g)
        print(f"- 한발 지지 {s['ss']:.1f} % · 스윙발 p95 {s['apex']:.0f} mm · 몸통 롤 최대 {s['roll_pk']:.1f}° · 한발 지지 중 지지 무릎 {s['kn_st']:.1f}°")
        for nm, lab in (("hA", "힙 외전"), ("kn", "무릎")):
            x = s[nm]
            print(f"- {lab}: 지지 반주기 평균 {x['st_mean']:.1f}·RMS {x['st_rms']:.1f}(포화 {x['st_sat']:.1f} %) → 보폭 평균 {x['mean']:.1f}·RMS {x['rms']:.1f} · 한발 지지 정적 요구 {x['th']:.1f}")


def sec_bounds(model, lp, tag):
    from scipy.optimize import least_squares
    d = load_seg(lp, tag, 3, 8); u = d["u"]; v = np.remainder(2 * u, 1.0)
    mk = d["al"] & (u >= 0.5) & (v >= 0.45) & (v < 0.65)                      # 왼 지지 한발 중간
    q0 = np.median(d["q"][mk], 0); r0 = float(np.median(d["roll"][mk])); p0 = float(np.median(d["pitch"][mk]))
    g0 = gdir_from_rp(np.array([r0]), np.array([p0]))
    iLhF, iLhA, iLhR, iLkn, iLaF, iRhF, iRhA, iRkn, iRaF = (JN.index(n) for n in ("left_hip_f_joint", "left_hip_a_joint", "left_hip_r_joint", "left_knee_joint", "left_ankle_f_joint",
                                                                                  "right_hip_f_joint", "right_hip_a_joint", "right_knee_joint", "right_ankle_f_joint"))
    th0 = model.hold_torque(q0[None], g0, "L")[0]
    print(f"\n## [bounds] 정적 하한(모델만) — 기준 자세 = {tag} 제자리 걸음 왼 지지 한발 중간(v .45–.65)의 중앙값: 몸통 롤 {r0:.1f}°(지지 쪽)·피치 {p0:.1f}°")
    print("- 관절각 중앙 [deg]: " + " ".join(f"{SHORT[j]} {np.degrees(q0[j]):.1f}" for j in range(12)))
    print(f"- 그 자세의 정적 유지 토크(지지 방향 +): 힙 외전 {-th0[iLhA]:.1f} · 힙 회전 {-th0[iLhR]:.1f} · 무릎 {th0[iLkn]:.1f} · 발목 롤 {th0[JN.index('left_ankle_r_joint')]:.1f} N·m"
          f" (같은 틱의 시뮬 적용 토크 중앙: 힙 외전 {-np.median(d['tq'][mk][:, iLhA]):.1f} · 무릎 {np.median(d['tq'][mk][:, iLkn]):.1f} · 발목 롤 {np.median(d['tq'][mk][:, 10]):.1f})")
    # (A0) 구성
    R, p, a, pj = model.fk(q0[None]); c = model.coms(R, p); ax = a["left_hip_a_joint"][0]; pv = pj["left_hip_a_joint"][0]
    grp = {"골반(+IMU)": ["pelvis", "imu", "base"], "지지 다리 힙 앞뒤 링크": ["left_hip_f_joint"], "스윙 다리 힙 앞뒤 링크": ["right_hip_f_joint"], "스윙 다리 나머지": model.subtree("right_hip_a_joint")}
    vals = {k: (-sum(float(ax @ np.cross(c[n][0] - pv, model.mass[n] * G * g0[0])) for n in ns), sum(model.mass[n] for n in ns)) for k, ns in grp.items()}
    tot = sum(x[0] for x in vals.values()); mt = sum(x[1] for x in vals.values())
    print("- (A0) 지지 힙 외전축 둘레 정적 모멘트의 구성: " + " · ".join(f"{k} {m_:.2f} kg → {val:.1f} N·m({100 * val / tot:.0f} %)" for k, (val, m_) in vals.items())
          + f" · 합 {tot:.1f} N·m(받치는 질량 {mt:.2f} kg, 등가 팔 길이 {tot / (G * mt):.3f} m)")
    sub = set(model.subtree("left_hip_a_joint")); Iup = 0.0
    for n in model.links:
        if n in sub or model.mass[n] <= 0:
            continue
        r = c[n][0] - pv
        Iup += float(ax @ (R[n][0] @ model.inertia[n] @ R[n][0].T) @ ax) + model.mass[n] * float(r @ r - (ax @ r) ** 2)
    print(f"- 위쪽 덩어리(스윙 다리 강체 부착)의 그 축 둘레 관성 I ≈ {Iup:.2f} kg·m² → 힙 토크가 정적 요구의 절반일 때 롤 각가속 {0.5 * tot / Iup:.1f} rad/s²")
    print("\n(A) 골반 롤(지지 쪽 +)·스윙 다리 위치에 따른 지지 다리 정적 토크 [힙 외전 · 힙 회전] N·m — 옆 흔들림(발목·지지 다리 기울기)은 이 값에 들어가지 않는다")
    print(f"| 골반 롤 ° | 스윙 다리 = 자료 중앙(힙 외전 {np.degrees(q0[iRhA]):.1f}°) | 스윙 다리 연직 | 스윙 다리 안쪽 한계(−13.2°) |"); print("|---|---|---|---|")
    for phi in (0.0, 5.0, 10.0, 15.0, 20.0):
        gd = gdir_from_rp(np.array([phi]), np.array([p0])); cells = []
        for mode in ("data", "vertical", "adduct"):
            q = q0.copy()
            if mode == "vertical":
                q[iRhA] = np.radians(phi)
            elif mode == "adduct":
                q[iRhA] = np.radians(-13.2)
            t = model.hold_torque(q[None], gd, "L")[0]; cells.append(f"{-t[iLhA]:.1f} · {-t[iLhR]:.1f}")
        print(f"| {phi:.0f} | " + " | ".join(cells) + " |")

    def posture(kap, theta, com_fwd=0.0):
        gd = gdir_from_rp(np.array([0.0]), np.array([theta])); up = -gd[0]; fwd = np.array([1.0, 0, 0]) - up * up[0]; fwd /= np.linalg.norm(fwd)

        def build(x):
            q = np.zeros(12); q[iLhF] = x[0]; q[iLkn] = -np.radians(kap); q[iLaF] = x[1]; q[iRhF] = -x[0]; q[iRkn] = np.radians(kap); q[iRaF] = -x[1]
            return q

        def res(x):
            R_, p_, a_, pj_ = model.fk(build(x)[None]); c_ = model.coms(R_, p_)
            fr = 0.5 * (p_["leftfoot_leftfront_RA30P"][0] + p_["leftfoot_rightfront_RA30P"][0]); bk = 0.5 * (p_["leftfoot_leftback_RA30P"][0] + p_["leftfoot_rightback_RA30P"][0])
            com = sum(model.mass[n] * c_[n][0] for n in model.links) / model.total_mass
            return [float((fr - bk) @ up) * 20.0, float((com - 0.5 * (fr + bk)) @ fwd) - com_fwd]
        s = least_squares(res, x0=[np.radians(0.5 * kap)] * 2); q = build(s.x)
        R_, p_, a_, pj_ = model.fk(q[None])
        return q, gd, -float(p_["leftfoot_leftfront_RA30P"][0] @ up)
    print("\n(B) 지지 무릎 굽힘별 정적 토크(몸통 피치 5°·발 평평·전신 CoM 이 발 중앙 위; 스윙 다리는 같은 자세로 옆에 든 근사) + 그 힙 굴곡에서의 좌우면 분담")
    print("| 지지 무릎 ° | 힙 굴곡 ° | 골반 높이 m | 한발 지지 무릎 N·m (CoM 뒤 2 cm · 중앙 · 앞 2 cm) | 양발(절반씩) 무릎 | 그 힙 굴곡에서 한발 지지 힙 외전 · 힙 회전 |"); print("|---|---|---|---|---|---|")
    for kap in (20, 30, 40, 45, 50, 55, 60, 65, 70):
        q, gd, hgt = posture(kap, 5.0)
        t1 = [model.hold_torque(posture(kap, 5.0, cf)[0][None], gd, "L")[0][iLkn] for cf in (-0.02, 0.0, 0.02)]
        t2 = model.halved("L").hold_torque(q[None], gd, "L")[0][iLkn]
        qq = q0.copy(); qq[iLhF] = q[iLhF]; tt = model.hold_torque(qq[None], g0, "L")[0]
        print(f"| {kap} | {np.degrees(q[iLhF]):.1f} | {hgt:.3f} | {t1[0]:.1f} · {t1[1]:.1f} · {t1[2]:.1f} | {t2:.1f} | {-tt[iLhA]:.1f} · {-tt[iLhR]:.1f} |")
    Mg = tot; kn_ss = th0[iLkn]
    print(f"\n(C) 걸음 시간표별 하한 — 힙 외전 정적 모멘트 M = {Mg:.1f} N·m, 한발 지지 무릎 정적 요구 τk = {kn_ss:.1f} N·m(기준 자세), I = {Iup:.2f} kg·m², 걸음 1.1 Hz(반주기 0.4545 s)")
    print("| 한발 지지 비율(반주기 중) | 한발 지지 시간 s | 힙 외전 보폭 평균 = M·d (각운동량 균형) | 힙 외전 RMS: 준정적 M·√d ~ 접지 내내 고르게 M·d/√(1−d) | 힙 외전 피크: 준정적 M ~ 고르게 M·d/(1−d) | 힙 토크 M/2 로 버틸 때 필요한 롤 흔들림(탄도, 양끝 대칭) ° | 무릎이 몸무게를 받치려면 필요한 한계 τk/(1+양발 비율) (25 대비) |"); print("|---|---|---|---|---|---|---|")
    for s in (0.83, 0.76, 0.70, 0.60, 0.50, 0.43, 0.35):
        dd = 0.5 * s; Tss = s * 0.4545; alpha = 0.5 * Mg / Iup
        print(f"| {s:.2f} | {Tss:.3f} | {Mg * dd:.1f} | {Mg * np.sqrt(dd):.1f} ~ {Mg * dd / np.sqrt(1 - dd):.1f} | {Mg:.1f} ~ {Mg * dd / (1 - dd):.1f} | {np.degrees(alpha * Tss ** 2 / 8):.1f} | {kn_ss / (2 - s):.1f} ({kn_ss / (2 - s) / 25:.2f}) |")
    print("  d = 한 다리의 한발 지지 시간 / 보폭 주기. 무릎 식: 주기 평균 수직 지지 = 몸무게 ⇒ τ_lim·(한발 비율 + 2·양발 비율) ≥ τk (무릎이 내내 한계에 붙어 있을 때의 최소 — 제어 여유 0).")


# ================================================================ 판정 보조(TORQUE_BUDGET_1007.md §10) ================================================================
def _wilson(k, n, z=1.96):
    if n <= 0:
        return 0.0, 1.0
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def x13_metrics(model, lp, prefix, seeds, scales=(1.0, 0.8, 0.7, 0.6)):
    """X13 한 실행 묶음(<prefix>_d2_s<seed>)의 배율별 수치. 낙상·전진·RMS·포화 시간은 x13_summ.py 와 같은 정의(구간 2–8·8–16 s, 전진 = 8→16 s 변위를 8 s 요로 투영,
    RMS·포화 = 9–16 s·16 s 까지 생존한 추적 env·seed 평균). 포화 벌점 값·지지 무릎 각·스윙발 높이·한발 지지 비율·몸통 롤은 걷기 틱(제자리 3–8 s + 앞 9–16 s, 낙상 전)."""
    import csv
    out = {g: dict(n=[0, 0], f=[0, 0], fwd=[], rms=[], sat=[], tq=[], kn=[], dz=[], v=[], sroll=[], ss=[]) for g in scales}
    missing = []
    for sd in seeds:
        tag = f"{prefix}_d2_s{sd}"
        pk = os.path.join(lp, f"kwloop_{tag}.npz"); pc = os.path.join(lp, f"ground_recovery_{tag}_envs.csv"); pt = os.path.join(lp, f"transall_{tag}.npz")
        if not (os.path.exists(pk) and os.path.exists(pc) and os.path.exists(pt)):
            missing.append(tag); continue
        z = np.load(pk, allow_pickle=True); pre = z["pre_pass"].astype(bool); sc = z["effort_scale"].astype(float); t = z["t"].astype(float)
        xy = z["xy"].astype(float); yaw = z["yaw"].astype(float)
        tf = np.full(pre.shape[0], np.inf)
        for r in csv.DictReader(open(pc)):
            if r["fell"] == "1" and r["t_fall_rel"] not in ("", "nan"):
                tf[int(r["env"])] = float(r["t_fall_rel"])
        Kt = lambda x: min(int(np.searchsorted(t, x - 1e-6)), len(t) - 1)      # noqa: E731
        zt = np.load(pt, allow_pickle=True); tqa = np.abs(zt["tq"].astype(np.float32)); n128 = tqa.shape[1]
        for g in scales:
            m = pre & (np.abs(sc - g) < 1e-6); A = out[g]
            for i, (t0, t1) in enumerate(((2.0, 8.0), (8.0, 16.0))):
                al = m & (tf >= t0); A["n"][i] += int(al.sum()); A["f"][i] += int((al & (tf < t1)).sum())
            ok = m & (tf >= 16.0); d = xy[Kt(16.0)] - xy[Kt(8.0)]; c0, s0 = np.cos(yaw[Kt(8.0)]), np.sin(yaw[Kt(8.0)])
            A["fwd"] += list((c0 * d[:, 0] + s0 * d[:, 1])[ok])
            mm = m[:n128] & (tf[:n128] >= 16.0)
            if mm.any():
                X = tqa[Kt(9.0):Kt(16.0)][:, mm]
                A["rms"].append(np.sqrt((X[..., [2, 3, 6, 7]] ** 2).mean((0, 1)))); A["sat"].append((X[..., [2, 3, 6, 7]] >= 25.0 * g - 0.5).mean((0, 1)))
            for a_, b_ in ((3.0, 8.0), (9.0, 16.0)):
                d_ = load_seg(lp, tag, a_, b_, scale=g)
                if d_["al"].sum() < 50:
                    continue
                f = fold(d_, model); al_ = f["al"]
                A["tq"].append(d_["tq"][al_]); A["dz"].append(f["dzf"][al_]); A["v"].append(f["v"][al_]); A["sroll"].append(f["sroll"][al_])
                A["kn"].append(f["kn_st"][al_ & (f["dzf"] > 0.008)]); A["ss"].append((f["dzf"][al_] > 0.008))
    res = {}
    for g in scales:
        A = out[g]; r = dict(n_walk=A["n"][0], f_walk=A["f"][0] + A["f"][1], n_trace=0)
        r["fall"] = 100.0 * r["f_walk"] / max(r["n_walk"], 1); r["fall_ci"] = tuple(100 * x for x in _wilson(r["f_walk"], max(r["n_walk"], 1)))
        r["fwd"] = float(np.median(A["fwd"])) if A["fwd"] else float("nan")
        r["rms"] = np.mean(A["rms"], 0) if A["rms"] else np.full(4, np.nan); r["sat"] = 100 * np.mean(A["sat"], 0) if A["sat"] else np.full(4, np.nan)
        if A["tq"]:
            tq = np.concatenate(A["tq"]); dz = np.concatenate(A["dz"]); v = np.concatenate(A["v"]); sr = np.concatenate(A["sroll"]); kn = np.concatenate(A["kn"])
            r["n_trace"] = len(tq); r["tsat"] = float(sat_term(tq, LIM, 0.6).mean()); r["tsat_env"] = float(sat_term(tq, LIM * g, 0.6).mean())
            r["kn_st"] = float(kn.mean()) if len(kn) else float("nan"); r["apex"] = 1e3 * float(np.percentile(dz, 95)); r["ss"] = 100 * float(np.concatenate(A["ss"]).mean())
            vb = np.minimum((v * 23).astype(int), 22); r["roll_pk"] = float(np.nanmax([sr[vb == i].mean() if (vb == i).any() else np.nan for i in range(23)]))
        else:
            r.update(tsat=float("nan"), tsat_env=float("nan"), kn_st=float("nan"), apex=float("nan"), ss=float("nan"), roll_pk=float("nan"))
        res[g] = r
    return res, missing


def judge(model, lp, kind, prefix, base, seeds):
    """§10 의 T1(기전)·T2(낮은 한계)·T3(해치지 않음) 수치와 충족/미달. T4(왕복)는 이 도구가 계산하지 않는다. 기준 문턱의 정본은 TORQUE_BUDGET_1007.md §10."""
    C, miss = x13_metrics(model, lp, prefix, seeds)
    B, miss_b = x13_metrics(model, lp, base, seeds)
    print(f"# 판정 보조(TORQUE_BUDGET_1007.md §10) — 팔 {kind} · 후보 {prefix} · 기준선 {base} · seed {','.join(str(x) for x in seeds)} · lp {lp}")
    if miss or miss_b:
        print("- 없는 실행:", ", ".join(miss + miss_b), "(transall_·kwloop_ npz 와 ground_recovery_…_envs.csv 셋 다 있어야 한다)")
        if miss:
            return 2
    c, b = C[1.0], B[1.0]; ok = {}
    yn = lambda x: "**충족**" if x else "미달"                                 # noqa: E731
    fl = lambda r: f"{r['f_walk']} / {r['n_walk']} ({r['fall']:.1f} %, {r['fall_ci'][0]:.1f}–{r['fall_ci'][1]:.1f})"   # noqa: E731
    print("| # | 항목(한계 배율) | 기준 | 기준선 | 후보 | 판정 |"); print("|---|---|---|---|---|---|")
    if kind == "s3t":
        ok["T1"] = c["tsat"] <= 0.0035
        print(f"| T1 기전 | torque_saturation 평균(문턱 0.6·명목 한계, 1.0 배 걷기 {c['n_trace']} 틱) | ≤ 0.0035 | {b['tsat']:.4f} | {c['tsat']:.4f} | {yn(ok['T1'])} |")
    else:
        kr_c = float(np.mean(c["rms"][2:])); kr_b = float(np.mean(b["rms"][2:])); o1, o2 = c["kn_st"] <= 52.0, kr_c <= 11.0; ok["T1"] = o1 and o2
        print(f"| T1 기전 | 한발 지지 중 지지 무릎 굽힘(1.0 배 걷기 틱, 운동학 접지) | ≤ 52° | {b['kn_st']:.1f}° | {c['kn_st']:.1f}° | {yn(o1)} |")
        print(f"| T1 기전 | 무릎 RMS(앞 0.06 9–16 s, L·R 평균; L/R {c['rms'][2]:.1f}/{c['rms'][3]:.1f}) | ≤ 11.0 N·m | {kr_b:.1f} | {kr_c:.1f} | {yn(o2)} |")
    o8, o7 = C[0.8]["fall"] <= 5.0, C[0.7]["fall"] <= 30.0; ok["T2"] = o8 and o7
    print(f"| T2 낮은 한계 | 걷기 구간(2–16 s) 낙상, 0.8 배 | ≤ 5 % | {fl(B[0.8])} | {fl(C[0.8])} | {yn(o8)} |")
    print(f"| T2 낮은 한계 | 걷기 구간 낙상, 0.7 배 | ≤ 30 % | {fl(B[0.7])} | {fl(C[0.7])} | {yn(o7)} |")
    oa, ob, oc = c["fall"] <= 2.0, c["fwd"] >= 0.53, c["apex"] >= 45.0; ok["T3"] = oa and ob and oc
    print(f"| T3 해치지 않음 | 걷기 구간 낙상, 1.0 배 | ≤ 2 % | {fl(b)} | {fl(c)} | {yn(oa)} |")
    print(f"| T3 해치지 않음 | 8 s 전진 중앙(앞 0.06, 8→16 s), 1.0 배 | ≥ 0.53 m | {b['fwd']:+.2f} | {c['fwd']:+.2f} | {yn(ob)} |")
    print(f"| T3 해치지 않음 | 스윙발 높이 p95(1.0 배 걷기 틱) | ≥ 45 mm | {b['apex']:.0f} | {c['apex']:.0f} | {yn(oc)} |")
    if kind == "s3tk":
        hs_c = float(np.mean(c["sat"][:2])); hs_b = float(np.mean(b["sat"][:2])); od = hs_c <= hs_b + 3.0; ok["T3"] = ok["T3"] and od
        print(f"| T3 해치지 않음 | 힙 외전 포화 시간(9–16 s, L·R 평균), 1.0 배 | ≤ 기준선 + 3 %p ({hs_b + 3.0:.1f} %) | {hs_b:.1f} % | {hs_c:.1f} % | {yn(od)} |")
    print(f"| 참고 | 8 s 전진이 명령 거리의 0.75–1.5 배(0.36–0.72 m) | (판정 아님) | {'안' if 0.36 <= b['fwd'] <= 0.72 else '밖'} | {'안' if 0.36 <= c['fwd'] <= 0.72 else '밖'} | — |")
    print("| 참고 | RMS 힙 외전 L/R · 무릎 L/R [N·m], 1.0 배 | (판정 아님) | " + f"{b['rms'][0]:.1f}/{b['rms'][1]:.1f} · {b['rms'][2]:.1f}/{b['rms'][3]:.1f} | {c['rms'][0]:.1f}/{c['rms'][1]:.1f} · {c['rms'][2]:.1f}/{c['rms'][3]:.1f} | — |")
    print("| 참고 | 한계 포화 시간 힙 외전 L/R · 무릎 L/R [%], 1.0 배 | (판정 아님) | " + f"{b['sat'][0]:.0f}/{b['sat'][1]:.0f} · {b['sat'][2]:.0f}/{b['sat'][3]:.0f} | {c['sat'][0]:.0f}/{c['sat'][1]:.0f} · {c['sat'][2]:.0f}/{c['sat'][3]:.0f} | — |")
    print(f"| 참고 | 한발 지지 비율 % · 몸통 롤 최대 ° · 지지 무릎 ° · torque_saturation, 1.0 배 | (판정 아님) | {b['ss']:.0f} · {b['roll_pk']:.1f} · {b['kn_st']:.1f} · {b['tsat']:.4f} | {c['ss']:.0f} · {c['roll_pk']:.1f} · {c['kn_st']:.1f} · {c['tsat']:.4f} | — |")
    for g in (0.8, 0.7):
        print(f"| 참고 | {g:g} 배: 지지 무릎 ° · 한발 지지 % · 스윙발 p95 mm · 무릎 포화 L/R % · 8 s 전진 m | (판정 아님) | {B[g]['kn_st']:.1f} · {B[g]['ss']:.0f} · {B[g]['apex']:.0f} · {B[g]['sat'][2]:.0f}/{B[g]['sat'][3]:.0f} · {B[g]['fwd']:+.2f} | "
              f"{C[g]['kn_st']:.1f} · {C[g]['ss']:.0f} · {C[g]['apex']:.0f} · {C[g]['sat'][2]:.0f}/{C[g]['sat'][3]:.0f} · {C[g]['fwd']:+.2f} | — |")
    print("- 종합: " + " · ".join(f"{k} {yn(v_)}" for k, v_ in ok.items()) + " · T4(왕복)는 X7c C0 요약으로 따로 본다.")
    print("- 낙상·전진·RMS·포화 시간 정의 = x13_summ.py(전 env 256·seed 합). 포화 벌점 값·무릎 각·스윙발 높이·한발 지지 = 추적 env(앞 128 개 중 그 배율)·걷기 틱(제자리 3–8 s + 앞 9–16 s, 낙상 전).")
    return 0


# ================================================================ main ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lp", default=os.path.join(REPO, "log_picture"))
    ap.add_argument("--tag", default="x12_w3load_cmds_d2_s21")
    ap.add_argument("--x13", default="tr_x13_d2_s1,tr_x13_d2_s2")
    ap.add_argument("--seg", default="3,8", help="phase·decomp·steps·scale 절의 구간 [s] 'a,b'(기본 = 제자리 걸음)")
    ap.add_argument("--sections", default="model,phase,decomp,steps,cmds,sat,knee,scale,bounds")
    ap.add_argument("--urdf", default=os.path.join(REPO, "rl_walking", "assets", "biped12_final3.urdf"))
    ap.add_argument("--judge", default="", choices=("", "s3t", "s3tk"), help="TORQUE_BUDGET_1007.md §10 의 T1·T2·T3 수치와 충족/미달만 찍는다(팔 이름)")
    ap.add_argument("--prefix", default="", help="--judge 의 후보 X13 실행 접두(예: tr_s3t16299_x13 → <prefix>_d2_s<seed>)")
    ap.add_argument("--base", default="tr_x13", help="--judge 의 기준선 X13 실행 접두")
    ap.add_argument("--seeds", default="1,2")
    a = ap.parse_args()
    if a.judge:
        if not a.prefix:
            ap.error("--judge 에는 --prefix 가 필요하다")
        return judge(Model(a.urdf), a.lp, a.judge, a.prefix, a.base, [int(x) for x in a.seeds.split(",")])
    sec = a.sections.split(","); s0, s1 = [float(x) for x in a.seg.split(",")]; x13 = [t for t in a.x13.split(",") if t]
    model = Model(a.urdf)
    print(f"# torque_phase.py — tag {a.tag} · lp {a.lp} · 구간 {s0:g}–{s1:g} s · 절 {a.sections}")
    f = None
    if "model" in sec:
        sec_model(model, a.lp, a.tag)
    if "phase" in sec:
        f = sec_phase(model, a.lp, a.tag, s0, s1)
    if "decomp" in sec or "steps" in sec:
        f = f if f is not None else fold(load_seg(a.lp, a.tag, s0, s1), model)
        if "decomp" in sec:
            sec_decomp(f)
        if "steps" in sec:
            sec_steps(f)
    if "cmds" in sec:
        sec_cmds(model, a.lp, a.tag)
    if "sat" in sec:
        sec_sat(a.lp, a.tag, x13)
    if "knee" in sec:
        sec_knee(model, a.lp, a.tag)
    if "scale" in sec:
        sec_scale(model, a.lp, x13, s0, s1)
    if "bounds" in sec:
        sec_bounds(model, a.lp, a.tag)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    sys.exit(main())
