"""1005 전환의 물리적 영향 진단(고정 scorecard 와 별도): 무외란 서기(nd) 덤프에서 모드 전환 틱마다
실행 액션 점프·체인 뒤 관절 명령 점프·토크 변화·전환 뒤 1 s ω/기울기 증가·발 하중 상실(이륙) 을 재고,
같은 env 들의 평상 틱(전환 ±1 s 밖) 분포와 비교한다. 사용: python3 switch_physical.py <태그 접두> [seed...]"""
import sys, glob, json, numpy as np
tag = sys.argv[1]; seeds = sys.argv[2:] or ["1", "2"]
W = 50   # 1 s
tr = {"on": [], "off": []}; base = []
for sd in seeds:
    for fp in sorted(glob.glob(f"/home/ryu/humanoid_leg_test1/log_picture/recbb_{tag}_nd_s{sd}_env*.jsonl")):
        L = [json.loads(l) for l in open(fp)]
        L = [x for x in L if "par_mode" in x and "raw_deg" in x and "cmd_deg" in x]
        if len(L) < 3 * W: continue
        m = np.array([x["par_mode"] for x in L]); raw = np.array([x["raw_deg"] for x in L]); cmd = np.array([x["cmd_deg"] for x in L])
        tau = np.array([x["tau"] for x in L]); w = np.array([x["w_xy"] for x in L]); tl = np.hypot([x["pitch"] for x in L], [x["roll"] for x in L])
        fz = np.array([[x["fz_L"], x["fz_R"]] for x in L])
        draw = np.r_[0, np.abs(np.diff(raw, axis=0)).max(1)]; dcmd = np.r_[0, np.abs(np.diff(cmd, axis=0)).max(1)]; dtau = np.r_[0, np.abs(np.diff(tau, axis=0)).max(1)]
        ks = [k for k in range(1, len(m)) if m[k] != m[k - 1]]
        near = np.zeros(len(m), bool)
        for k in ks: near[max(0, k - W):k + W] = True
        for k in range(W, len(m) - W):
            if not near[k]:
                base.append((draw[k:k + 3].max(), dcmd[k:k + 5].max(), dtau[k:k + 5].max(), w[k:k + W].mean() - w[k - W:k].mean(), tl[k:k + W].max() - tl[k - W:k].max(), int(((fz[k:k + W] < 5).any(0) & ~(fz[k - W:k] < 5).any(0)).any())))
        for k in ks:
            if k < W or k > len(m) - W: continue
            kind = "on" if m[k] == 1 else "off"
            tr[kind].append((draw[k:k + 3].max(), dcmd[k:k + 5].max(), dtau[k:k + 5].max(), w[k:k + W].mean() - w[k - W:k].mean(),
                             tl[k:k + W].max() - tl[k - W:k].max(), int(((fz[k:k + W] < 5).any(0) & ~(fz[k - W:k] < 5).any(0)).any())))
names = ["실행 액션 점프°", "관절 명령 점프°(5틱)", "토크 변화 Nm(5틱)", "ω 평균 증가(1 s)", "기울기 최대 증가°(1 s)", "새 발 하중 상실(1 s)"]
def summ(A, label):
    A = np.array(A)
    if len(A) == 0: print(f"  {label}: 없음"); return
    print(f"  {label} (n {len(A)}): " + " · ".join(f"{n} 중앙 {np.median(A[:, i]):.3f} p90 {np.percentile(A[:, i], 90):.3f}" if i < 5 else f"{n} {100 * A[:, i].mean():.0f} %" for i, n in enumerate(names)))
print(f"== {tag} nd, seed {seeds}")
summ(base, "평상 틱(전환과 같은 창 길이)")
summ(tr["on"], "회복 켜짐 전환")
summ(tr["off"], "회복 꺼짐 전환")
