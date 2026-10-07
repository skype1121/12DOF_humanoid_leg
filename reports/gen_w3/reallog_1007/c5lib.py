"""실물 블랙박스 C5 회복 모드 분석 공용 함수(읽기 전용 분석 — 로봇·젯슨 미접촉).
- load(): 블랙박스 틱·사건 분리, 감지기 입력(장착 편향 보정 중력·자이로·접촉 0/1) 복원 — 브리지 BridgeCore.tick 과 같은 식.
- replay(): 감지기를 기록 입력에 열린 고리로 돌리고 내부량(기준 대비 기울기·|ω_xy|·기준·타이머)을 틱별로 남긴다.
"""
import importlib.util, json, math, os, sys
import numpy as np

REPO = "/home/ryu/humanoid_leg_test1"
S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "rl_walking", "deploy"))   # 옛 c5_switch(단독 import 경로)가 policy_runner_stage4 를 찾도록
sys.path.insert(0, REPO)


def _mod(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


RV3MOD = _mod("c5_switch_rv3", os.path.join(S, "c5_switch_rv3_f8e4025d.py"))      # 1004 젯슨 배포본(md5 2c8943b8 = deploy/c5_1004.md5)
from rl_walking.deploy import c5_switch as NOW  # noqa: E402  현재 작업 트리(KE 로컬·1005 배포본과 imu 경로 비트 동일 주장 — selftest_odic_equiv [A])

# 브리지 설정 사본(코드에서 읽은 값 — jetson/scripts/rl_bridge_node.py)
RV3_KW = dict(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5)   # 1004 배포 브리지 C5_DETECTOR_KW(끄기 = 양발)
RV4_KW = dict(RV3_KW, off_contact="any")                                                                                       # 현재 C5_DETECTOR_KW
KE_KW = dict(RV4_KW, early=(0.6, 6.0, 0.4, 0.2), refrac=(1.5, 6.0, 1.0), in_filt=0.33, early_gate=(5.0, 1.0))                 # + C5_EXTRA[stand_c5kr_g21a_g24ra]["det"]


def projected_gravity(q):
    w, x, y, z = q
    return (-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y)))


def correct_gravity(g, b):
    b = [float(b[0]), float(b[1])] if b is not None else [0.0, 0.0]
    if abs(b[0]) < 1e-12 and abs(b[1]) < 1e-12:
        return g
    gx, gy, gz = (float(v) for v in g)
    if -gz < 0.3:
        return g
    roll = math.atan2(gy, -gz) - math.radians(float(b[0])); pitch = math.atan2(gx, -gz) - math.radians(float(b[1]))
    tx, ty = math.tan(pitch), math.tan(roll); c = 1.0 / math.sqrt(1.0 + tx * tx + ty * ty)
    return (c * tx, c * ty, -c)


def rp_deg(g):
    g = np.asarray(g, float)
    return np.degrees(np.array([math.atan2(g[1], -g[2]), math.atan2(g[0], -g[2])]))     # [롤, 피치]


def load(path):
    T = []; EV = []
    with open(path, "rb") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if "event" in d or "qpos_rad" not in d:
                EV.append(d); continue
            T.append(d)
    if not T:
        return None
    n = len(T)
    ts = np.array([r["timestamp"] for r in T], float)
    D = dict(path=path, name=os.path.basename(path), n=n, ts=ts, t=ts - ts[0], ev=EV, ck=sorted({str(r.get("checkpoint")) for r in T}),
             quat=np.array([r["quat_wxyz"] for r in T], float), gy=np.array([r["gyro_rad_s"] for r in T], float),
             c=np.array([r["contact"] for r in T], float), q=np.array([r["qpos_rad"] for r in T], float),
             foot_n=np.array([r.get("foot_n") or [np.nan, np.nan] for r in T], float),
             raw_sum=np.array([[np.nan if v is None else v for v in (r.get("foot_raw_sum") or [None, None])] for r in T], float),
             hs=np.array([str(r.get("hang_state")) for r in T]), blend=np.array([r.get("blend_alpha", np.nan) for r in T], float),
             rp_raw=np.array([r.get("rp_deg") or [np.nan, np.nan] for r in T], float),
             bias=np.array([r.get("imu_mount_bias_deg") or [0.0, 0.0] for r in T], float),
             live=np.array([bool(r.get("live")) for r in T]), armed=np.array([bool(r.get("armed")) for r in T]),
             age=np.array([np.nan if r.get("status_age_s") is None else r["status_age_s"] for r in T], float),
             stop=[r.get("stop_reason") for r in T], gm=sorted({str(r.get("guard_mode")) for r in T}),
             tgt=np.array([[r["targets_deg_joint"][j] for j in JO] if isinstance(r.get("targets_deg_joint"), dict) else [np.nan] * 12 for r in T], float),
             tau=np.array([[np.nan if v is None else v for v in (r.get("tau_est_nm") or [None] * 12)] for r in T], float))
    has = np.array([r.get("c5") is not None for r in T])
    D["has_c5"] = has
    if has.any():
        D["mode"] = np.array([int((r.get("c5") or {}).get("mode") or 0) for r in T])
        D["sw"] = np.array([str((r.get("c5") or {}).get("switch") or "") for r in T])
        D["n_on"] = np.array([int((r.get("c5") or {}).get("n_on") or 0) for r in T])
        D["ref_log"] = np.array([((r.get("c5") or {}).get("ref_deg") or [np.nan, np.nan]) for r in T], float)
        D["c5keys"] = sorted(set().union(*[set((r.get("c5") or {}).keys()) for r in T]))
        D["raw_s"] = np.array([((r.get("c5") or {}).get("raw_stand") or [np.nan] * 12) for r in T], float)
        D["raw_r"] = np.array([((r.get("c5") or {}).get("raw_rec") or [np.nan] * 12) for r in T], float)
    D["g"] = np.array([correct_gravity(projected_gravity(qq), b) for qq, b in zip(D["quat"], D["bias"])], float)
    D["rp"] = np.array([rp_deg(g) for g in D["g"]])            # 감지기가 보는(편향 보정) [롤, 피치] — 필터 전
    D["wxy"] = np.linalg.norm(D["gy"][:, :2], axis=1)           # 원 자이로 |ω_xy|
    return D


from rl_walking.deploy.policy_runner import JOINT_ORDER as JO  # noqa: E402


def replay(D, kind, i0=0, i1=None):
    """kind: 'rv3'(1004 배포본 모듈) | 'rv3now'(현재 모듈·끄기 양발) | 'rv4' | 'ke'. 반환 dict(틱별 배열)."""
    i1 = D["n"] if i1 is None else i1
    if kind == "rv3":
        det = RV3MOD.SwitchDetector(**RV3_KW)
    elif kind == "rv3now":
        det = NOW.SwitchDetector(**dict(RV3_KW, off_contact="both"))
    elif kind == "rv4":
        det = NOW.SwitchDetector(**RV4_KW)
    elif kind == "ke":
        det = NOW.SwitchDetector(**KE_KW)
    else:
        raise ValueError(kind)
    n = i1 - i0
    R = dict(mode=np.zeros(n, int), tilt=np.zeros(n), w=np.zeros(n), ref=np.zeros((n, 2)), calm=np.zeros(n), t_on=np.zeros(n),
             n_on=np.zeros(n, int), n_early=np.zeros(n, int), hold=np.zeros(n), pk_t=np.zeros(n), pk_w=np.zeros(n), rpf=np.zeros((n, 2)),
             off_why=[""] * n)
    prev = 0; ne = 0
    for k in range(n):
        i = i0 + k
        g, w3, c = D["g"][i], D["gy"][i], D["c"][i]
        m = det.update(g, w3, c)
        if getattr(det, "in_filt", 0.0) > 0.0:
            gf = det._fg / max(float(np.linalg.norm(det._fg)), 1e-6); wf = det._fw
        else:
            gf, wf = g, w3
        rpf = rp_deg(gf)
        R["mode"][k] = m; R["rpf"][k] = rpf; R["ref"][k] = det.ref
        R["tilt"][k] = float(np.linalg.norm(rpf - det.ref)); R["w"][k] = float(np.linalg.norm(np.asarray(wf, float)[:2]))
        R["calm"][k] = det.calm; R["t_on"][k] = det.t_on; R["n_on"][k] = det.n_on
        R["n_early"][k] = getattr(det, "n_early", 0); R["hold"][k] = getattr(det, "early_hold", 0.0)
        R["pk_t"][k] = getattr(det, "pk_tilt", 0.0); R["pk_w"][k] = getattr(det, "pk_w", 0.0)
        if prev == 1 and m == 0:
            R["off_why"][k] = "early" if R["n_early"][k] > ne else "calm"
        prev = m; ne = R["n_early"][k]
    return R


def runs(mask):
    """연속 True 구간 [(시작, 끝(미포함))]."""
    m = np.asarray(mask, bool); out = []; s = None
    for i, v in enumerate(m):
        if v and s is None:
            s = i
        if (not v) and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(m)))
    return out


def activations(mode):
    return runs(np.asarray(mode) == 1)
