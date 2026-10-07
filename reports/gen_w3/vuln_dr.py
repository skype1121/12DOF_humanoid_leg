"""반복 낙상 개체와 개체별 무작위화 값의 관계(탐색 — 가설 생성용, 다중 비교 주의).
입력: log_picture/drdump_tr_drdump_d2_s{1,2}.npz(인계 시점 env 별 값, 128 env) · 왕복 실행들의 env CSV(같은 seed·env 수 = 같은 개체).
개체 분류: 왕복 실행(X7·X7b)에서 2 회 이상 낙상 = 반복 낙상(V), 0 회 = 무낙상(O). 지표: AUC(V 대 O)와 순열 검정 p(양측, 5000 회).
사용: python3 reports/gen_w3/vuln_dr.py"""
import csv, os
import numpy as np
LP = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "log_picture")
# 서로 다른 궤적만(1007 정정: X7·X7 현행 감지기·S_orig·HD1 끔은 nd 에서 같은 궤적이라 하나만 센다). HD1 켬은 고리 때문에 궤적이 다르다.
RUNS = {1: ["tr_kwscq_nd_d2_s1", "tr_x7b_ff_d2_s1", "tr_x7b_bf_d2_s1", "tr_hd1on_d2_s1"], 2: ["tr_kwscq_nd_d2_s2", "tr_x7b_ff_d2_s2", "tr_x7b_bf_d2_s2", "tr_hd1on_d2_s2"]}
PAIRS = [("hip_f", -1), ("hip_a", -1), ("hip_r", +1), ("knee", -1), ("ankle_f", -1), ("ankle_r", +1)]   # 둘째 값: 좌우가 거울 부호(−1)면 '좌 + 우'가 비대칭, 같은 부호(+1)면 '좌 − 우'가 비대칭 — 아래에서 둘 다 계산해 보고


def auc(v, o):
    v = np.asarray(v, float); o = np.asarray(o, float); a = np.concatenate([v, o]); r = a.argsort().argsort().astype(float) + 1
    # 동률 평균 순위
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True); s = np.zeros(len(cnt)); np.add.at(s, inv, r); r = (s / cnt)[inv]
    return (r[:len(v)].sum() - len(v) * (len(v) + 1) / 2) / (len(v) * len(o))


def main():
    rng = np.random.default_rng(0); F = {}; lab = []
    for s, tags in RUNS.items():
        p = os.path.join(LP, f"drdump_tr_drdump_d2_s{s}.npz")
        if not os.path.exists(p):
            print("덤프 없음:", p); continue
        z = np.load(p, allow_pickle=True); jn = [str(x) for x in z["joint_names"]]; J = {n: i for i, n in enumerate(jn)}
        cnt = {}; nrun = 0
        for tag in tags:
            q = os.path.join(LP, f"ground_recovery_{tag}_envs.csv")
            if not os.path.exists(q):
                continue
            nrun += 1
            for r in csv.DictReader(open(q)):
                if r["pre_pass"] == "1":
                    cnt[int(r["env"])] = cnt.get(int(r["env"]), 0) + (r["fell"] == "1")
        envs = sorted(cnt); lab += [cnt[e] for e in envs]
        def add(name, arr):
            F.setdefault(name, []).extend(np.asarray(arr, float)[envs].tolist())
        cb = np.degrees(z["calib_joint_bias"]) if "calib_joint_bias" in z.files else None
        if cb is not None:
            for n, i in J.items():
                add(f"영점 편향 {n.replace('_joint','')} [°]", cb[:, i])
            for nm, _ in PAIRS:
                l, r = cb[:, J[f"left_{nm}_joint"]], cb[:, J[f"right_{nm}_joint"]]
                add(f"영점 편향 {nm} 좌+우 [°]", l + r); add(f"영점 편향 {nm} 좌−우 [°]", l - r); add(f"영점 편향 {nm} |좌+우|", np.abs(l + r)); add(f"영점 편향 {nm} |좌−우|", np.abs(l - r))
        for k, nm in (("calib_imu_roll", "IMU 롤 편향 [°]"), ("calib_imu_tilt", "IMU 피치 편향 [°]")):
            if k in z.files:
                add(nm, np.degrees(z[k])); add("|" + nm + "|", np.abs(np.degrees(z[k])))
        if "imu_lag" in z.files:
            add("IMU 지연 [틱]", z["imu_lag"])
        for k, nm in (("stiffness", "kp"), ("damping", "kd"), ("friction", "관절 마찰"), ("armature", "관절 관성")):
            if k in z.files and z[k].ndim == 2:
                A = z[k]
                for n, i in J.items():
                    add(f"{nm} {n.replace('_joint','')}", A[:, i])
                for pn, _ in PAIRS:
                    l, r = A[:, J[f"left_{pn}_joint"]], A[:, J[f"right_{pn}_joint"]]
                    add(f"{nm} {pn} 좌/우 비", l / np.maximum(r, 1e-9)); add(f"{nm} {pn} |log 좌/우|", np.abs(np.log(np.maximum(l, 1e-9) / np.maximum(r, 1e-9))))
        if "masses" in z.files:
            M = z["masses"]; add("총 질량 [kg]", M.sum(1))
            for b in range(M.shape[1]):
                if np.ptp(M[:, b]) > 1e-6:
                    add(f"질량 body{b}", M[:, b])
        if "coms" in z.files:
            C = z["coms"]
            for ax, an in enumerate("xyz"):
                if np.ptp(C[:, 0, ax]) > 1e-6:
                    add(f"몸통 CoM {an} [m]", C[:, 0, ax])
        if "default_joint_pos" in z.files and "nominal_default_joint_pos" in z.files:
            D = np.degrees(z["default_joint_pos"] - z["nominal_default_joint_pos"])
            for pn, _ in PAIRS:
                l, r = D[:, J[f"left_{pn}_joint"]], D[:, J[f"right_{pn}_joint"]]
                if np.ptp(l) > 1e-6 or np.ptp(r) > 1e-6:
                    add(f"기본자세 편향 {pn} 좌+우 [°]", l + r); add(f"기본자세 편향 {pn} |좌+우|", np.abs(l + r))
    lab = np.array(lab); V = lab >= 2; O = lab == 0
    print(f"개체 {len(lab)} · 반복 낙상(≥ 2 회) {int(V.sum())} · 1 회 {int((lab == 1).sum())} · 무낙상 {int(O.sum())} · 특징 {len(F)} 개")
    res = []
    for nm, v in F.items():
        v = np.array(v, float)
        if len(v) != len(lab) or not np.isfinite(v).all() or np.ptp(v) < 1e-12:
            continue
        a = auc(v[V], v[O]); x = np.concatenate([v[V], v[O]]); nv = int(V.sum()); cntp = 0
        for _ in range(3000):
            rng.shuffle(x); cntp += abs(auc(x[:nv], x[nv:]) - 0.5) >= abs(a - 0.5) - 1e-12
        res.append((abs(a - 0.5), nm, a, (cntp + 1) / 3001, float(np.median(v[V])), float(np.median(v[O])), float(np.percentile(v[O], 10)), float(np.percentile(v[O], 90))))
    res.sort(reverse=True)
    print("| 특징 | AUC(반복 낙상 대 무낙상) | 순열 p | 반복 낙상 중앙 | 무낙상 중앙 (p10–p90) |"); print("|---|---|---|---|---|")
    for r in res[:18]:
        print(f"| {r[1]} | {r[2]:.2f} | {r[3]:.4f} | {r[4]:+.3f} | {r[5]:+.3f} ({r[6]:+.3f} – {r[7]:+.3f}) |")
    print(f"- 주의: 특징 {len(res)} 개를 한꺼번에 본 탐색이다. p < {0.05/ max(len(res),1):.4f}(본페로니) 아래만 강한 후보로 읽고, 인과는 그 축만 끈 실행으로 확인한다.")


if __name__ == "__main__":
    main()
