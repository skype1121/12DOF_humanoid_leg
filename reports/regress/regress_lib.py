"""고정 회귀 검사(1007) — 실행 계획(plan)·실행 유효성 검사(check)·매니페스트(manifest).

run_regress.sh 가 부른다. 학습 없음, 젯슨·실물 없음. GPU 실행은 run_regress.sh 가 한다(이 파일은 계획·판정만).

사용(직접 부를 일은 거의 없음):
  python3 regress_lib.py plan  <run_dir> <설정 JSON>        → <run_dir>/jobs.tsv, manifest.json
  python3 regress_lib.py check <run_dir> <job_id> <log> <시작 epoch>  → exit 0 유효 / 1 무효(이유 출력)
  python3 regress_lib.py finish <run_dir>                    → manifest 에 종료 시각·상태 집계 기록
"""
import csv, hashlib, json, math, os, re, subprocess, sys, time

DATA = os.environ.get("REGRESS_DATA_REPO", "/home/ryu/humanoid_leg_test1")   # 체크포인트(logs/)·평가 산출물(log_picture/)
WT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))      # 평가 코드는 이 도구가 있는 체크아웃의 것
LP = os.path.join(DATA, "log_picture")
DT = 0.02      # eval_w3 정책 주기(50 Hz) — steps_done 검사용

# ── 고정 항목 정의 ───────────────────────────────────────────────────────────
# W3 명령 항목: (항목, 명령 vx,vy,wz, 추가 인자) — 제자리·앞·뒤·옆±·회전±·혼합
W3_CMDS = [("z", "0,0,0", ""), ("f", "0.06,0,0", ""), ("b", "-0.04,0,0", ""), ("l", "0,0.05,0", ""), ("r", "0,-0.05,0", ""),
           ("tl", "0,0,0.2", "--turn_metrics"), ("tr", "0,0,-0.2", "--turn_metrics"), ("mix", "0.05,0.04,0.2", "--turn_metrics")]
# 걷기 중 외란(앞 0.06 걷기 기준, 위상 분산 0.91 s = 1.1 Hz 한 주기, 램프 끔). 힘 0 그룹 = 같은 시각 대조군
W3_DIST = ("push:f:6:0:0.3|push:f:6:12:0.3|push:b:6:18:0.3|cmd=0,0.05,0;push:l:6:0:0.2|cmd=0,0.05,0;push:l:6:30:0.2")
W3_DIST_LABELS = {"push:f:6:0:0.3": "f0", "push:f:6:12:0.3": "f12", "push:b:6:18:0.3": "b18",
                  "cmd=0,0.05,0;push:l:6:0:0.2": "l0", "cmd=0,0.05,0;push:l:6:30:0.2": "l30"}
GR_DIST = {"nd": "", "sf15": "--dist ramp:f:2:15:0.5:1.0", "fi40": "--dist push:f:2:40:0.2", "b18": "--dist push:b:2:18:0.3"}
# 서기·KE 공통 계약(kabs.sh·settle_ab_run.sh 와 같은 인자 — 같은 seed 면 기준선 sab_* 와 RNG 짝이 맞는다)
GR_BASE = ("--headless --task Biped12-HangStand-G11a-v0 --num_envs 128 --precheck_s 3 --after_s 8 --enc_obs v2 --contact_real "
           "--contact_ver v2 --delay_steps 2,2")
STAND_CONTRACT = "--last_action_ema 0.27 --obs_ema joint_vel:0.25"
KE_SWITCH = ("--sw_on 3.0,0.5 --sw_off 1.5,0.3,1.0,0.4 --sw_ref_tau 10 --sw_ref_init_s 1.5 --switch_to {rec} --sw_off_contact any "
             "--sw_rec_vx 0.03 --sw_rec_vx_s 1 --cue_dir_imu --sw_early_imu 0.6,6,0.4,0.2 --sw_refrac 1.5,6,1.0 --early_gate 5,1.0 "
             "--det_from_obs --det_filt 0.33")
KE_ARMS = {"imu": "", "odicw": "--settle_judge odo --settle_v 0.07 --settle_knee_inject 42 --settle_knee_comp 42 --settle_w 0.4"}

MODES = {   # 빠른 모드 = env 축소·짧은 채점·seed 1 · 전체 모드 = 기준선과 같은 env 수·seed 확장
    "quick": dict(w3_envs=128, w3_sec=15, hl_envs=96, stop=(8, 14), dist_envs=160, dist_sec=14, w3_seeds=[1], dist_seeds=[1],
                  stand_seeds=[1], ke_seeds=[1], ke_arms=["odicw"], d4hl=False),
    "full": dict(w3_envs=256, w3_sec=30, hl_envs=256, stop=(12, 20), dist_envs=320, dist_sec=16, w3_seeds=[1, 2], dist_seeds=[1, 2],
                 stand_seeds=[1, 2, 3], ke_seeds=[1, 2, 3, 4, 5, 6], ke_arms=["imu", "odicw"], d4hl=True),
}


def sh(cmd, cwd=WT):
    try:
        return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=120).stdout.strip()
    except Exception as e:   # noqa: BLE001
        return f"ERR {e}"


def md5(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_ckpt(run, ck):
    """logs/rsl_rl/<실험>/<run>/<ck> 를 찾는다(실험 이름은 과제 설정 몫이라 검색)."""
    root = os.path.join(DATA, "logs", "rsl_rl")
    for exp in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        p = os.path.join(root, exp, run, ck)
        if os.path.exists(p):
            return p
    return None


def plan(run_dir, cfg):
    M = MODES[cfg["mode"]]; rid = cfg["rid"]; only = set(cfg["only"].split(","))
    jobs = []   # (job_id, kind, tag, timeout_s, 인자 문자열, 메타)

    def tg(x):
        return f"rg_{rid}_{x}"
    w3run, w3ck = cfg["w3"].split(":"); stand_run, stand_ck = cfg["stand"].split(":")
    W3B = f"--headless --task {cfg['w3_task']} --load_run {w3run} --checkpoint {w3ck} --warmup 2 {cfg.get('w3_extra', '')}".strip()
    if "w3" in only:
        for sd in M["w3_seeds"]:
            for nm, cm, ex in W3_CMDS:
                jobs.append((f"W-{nm}-s{sd}", "w3", tg(f"w_{nm}_s{sd}"), 900,
                             f"{W3B} --num_envs {M['w3_envs']} --seconds {M['w3_sec']} --delay 2 --seed {sd} --cmd={cm} {ex}",
                             dict(item=f"W-{nm}", seed=sd, cmd=cm, delay=2, num_envs=M["w3_envs"], seconds=M["w3_sec"])))
        jobs.append(("W-hl-s1", "w3", tg("w_hl_s1"), 1800,
                     f"{W3B} --num_envs {M['hl_envs']} --long --ramp_off --delay 2 --seed 1 --cmd=0,0,0 --heading_loop 0.5 --turn_metrics",
                     dict(item="W-hl", seed=1, cmd="0,0,0", delay=2, num_envs=M["hl_envs"], seconds=120.0)))
        for sd in M["w3_seeds"]:
            jobs.append((f"W-d4-s{sd}", "w3", tg(f"w_d4_s{sd}"), 900,
                         f"{W3B} --num_envs {M['w3_envs']} --seconds {M['w3_sec']} --delay 4 --seed {sd} --cmd=0.06,0,0",
                         dict(item="W-d4", seed=sd, cmd="0.06,0,0", delay=4, num_envs=M["w3_envs"], seconds=M["w3_sec"])))
        if M["d4hl"]:
            jobs.append(("W-d4hl-s1", "w3", tg("w_d4hl_s1"), 900,
                         f"{W3B} --num_envs {M['w3_envs']} --seconds {M['w3_sec']} --delay 4 --seed 1 --cmd=0,0,0 --heading_loop 0.5 --turn_metrics",
                         dict(item="W-d4hl", seed=1, cmd="0,0,0", delay=4, num_envs=M["w3_envs"], seconds=M["w3_sec"])))
        sa, ss = M["stop"]
        jobs.append(("W-stop-s1", "w3", tg("w_stop_s1"), 900,
                     f"{W3B} --num_envs {M['w3_envs']} --seconds {ss} --delay 2 --seed 1 --cmd=0.06,0,0 --stop_at {sa}",
                     dict(item="W-stop", seed=1, cmd="0.06,0,0", delay=2, num_envs=M["w3_envs"], seconds=ss)))
    if "st" in only:   # W3 STAND 모드(ge 0, gait_enable 0) — 서기·시작(착지) 과도 낙상·정지 떨림
        for sd in M["w3_seeds"]:
            jobs.append((f"W-st-s{sd}", "w3", tg(f"w_st_s{sd}"), 900,
                         f"{W3B} --num_envs {M['w3_envs']} --seconds {M['w3_sec']} --delay 2 --seed {sd} --cmd=0,0,0 --gait_enable 0",
                         dict(item="W-st", seed=sd, cmd="0,0,0", delay=2, num_envs=M["w3_envs"], seconds=M["w3_sec"], gait=0)))
    if "dist" in only:
        for sd in M["dist_seeds"]:
            jobs.append((f"W-dist-s{sd}", "w3", tg(f"w_dist_s{sd}"), 1200,
                         f"{W3B} --num_envs {M['dist_envs']} --seconds {M['dist_sec']} --delay 2 --seed {sd} --cmd=0.06,0,0 --ramp_off "
                         f"--dist_jitter 0.91 --dist \"{W3_DIST}\"",
                         dict(item="W-dist", seed=sd, cmd="0.06,0,0", delay=2, num_envs=M["dist_envs"], seconds=M["dist_sec"])))
    if "stand" in only:
        for sd in M["stand_seeds"]:
            for c in ("nd", "b18"):
                jobs.append((f"S-{c}-s{sd}", "gr", tg(f"s_{c}_s{sd}"), 900,
                             f"{GR_BASE} --load_run {stand_run} --checkpoint {stand_ck} {STAND_CONTRACT} --seed {sd} {GR_DIST[c]} {cfg.get('stand_extra', '')}",
                             dict(item=f"S-{c}", seed=sd, cond=c, num_envs=128)))
    if "ke" in only:
        rec = cfg["rec"]
        for sd in M["ke_seeds"]:
            for c in ("nd", "sf15", "fi40", "b18"):
                for arm in (cfg.get("ke_arms") or M["ke_arms"]):
                    jobs.append((f"K-{arm}-{c}-s{sd}", "gr", tg(f"k_{arm}_{c}_s{sd}"), 900,
                                 f"{GR_BASE} --load_run {stand_run} --checkpoint {stand_ck} {STAND_CONTRACT} {KE_SWITCH.format(rec=rec)} "
                                 f"--seed {sd} {KE_ARMS[arm]} {GR_DIST[c]} {cfg.get('ke_extra', '')}",
                                 dict(item=f"K-{arm}-{c}", seed=sd, cond=c, arm=arm, num_envs=128)))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "jobs.tsv"), "w") as fh:
        for j in jobs:
            fh.write("\t".join([j[0], j[1], j[2], str(j[3]), re.sub(r"\s+", " ", j[4]).strip()]) + "\n")
    # 매니페스트: 코드 해시·체크포인트 md5·인자
    ckpts = {"w3": find_ckpt(w3run, w3ck), "stand": find_ckpt(stand_run, stand_ck)}
    rr, rc = cfg["rec"].split(":"); ckpts["rec"] = find_ckpt(rr, rc)
    man = dict(rid=rid, cfg=cfg, mode_params=M, started=time.strftime("%Y-%m-%d %H:%M:%S"),
               code=dict(worktree=WT, head=sh("git rev-parse HEAD"), branch=sh("git rev-parse --abbrev-ref HEAD"),
                         dirty=sh("git status --porcelain -- rl_walking reports/regress | head -50").splitlines(),
                         diff_md5=hashlib.md5(sh("git diff HEAD -- rl_walking").encode()).hexdigest()),
               scripts={p: md5(os.path.join(WT, p)) for p in ("rl_walking/scripts/eval_w3.py", "rl_walking/scripts/eval_ground_recovery.py",
                                                              "reports/regress/THRESHOLDS.md", "reports/regress/summarize.py",
                                                              "reports/regress/regress_lib.py", "reports/regress/run_regress.sh")},
               checkpoints={k: dict(path=v, md5=md5(v)) for k, v in ckpts.items()},
               gpu=sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"),
               data_repo=DATA, jobs={j[0]: dict(kind=j[1], tag=j[2], args=re.sub(r"\s+", " ", j[4]).strip(), **j[5]) for j in jobs})
    missing = [k for k, v in ckpts.items() if v is None and (k != "w3" or only & {"w3", "dist"}) and (k != "rec" or "ke" in only)]
    json.dump(man, open(os.path.join(run_dir, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    if missing:
        print(f"체크포인트 없음: {missing}", file=sys.stderr); sys.exit(2)
    print(f"계획: 작업 {len(jobs)} 개 → {run_dir}/jobs.tsv")


# ── 유효성 검사 ──────────────────────────────────────────────────────────────
BAD = re.compile(r"PxgCudaDeviceMemoryAllocator failed|CUDA error|CUBLAS_STATUS_ALLOC|out of memory|Unable to allocate|Failed to allocate|"
                 r"PhysX error|fail to launch|Create Stream|Failed to fetch")


def check(run_dir, job_id, log, t0):
    man = json.load(open(os.path.join(run_dir, "manifest.json"))); J = man["jobs"][job_id]; t0 = float(t0)
    why = []
    txt = open(log, errors="replace").read() if os.path.exists(log) else ""
    m = BAD.search(txt)
    if m:
        why.append(f"로그에 메모리/PhysX 실패 '{m.group(0)}'")
    if J["kind"] == "w3":
        jp = os.path.join(LP, f"w3eval_{J['tag']}.json")
        if not os.path.exists(jp):
            why.append("산출물 없음(w3eval json)")
        elif os.path.getmtime(jp) < t0:
            why.append("산출물이 이번 실행 전 파일")
        else:
            try:
                d = json.load(open(jp))
            except Exception as e:   # noqa: BLE001
                why.append(f"json 읽기 실패 {e}"); d = None
            if d:
                exp_ck = J["args"].split("--checkpoint ")[1].split()[0]; exp_run = J["args"].split("--load_run ")[1].split()[0]
                if not (d.get("checkpoint", "").endswith(exp_ck) and exp_run in d.get("checkpoint", "")):
                    why.append(f"체크포인트 불일치 {d.get('checkpoint')}")
                for k, v in (("seed", J["seed"]), ("delay", J["delay"]), ("num_envs", J["num_envs"])):
                    if d.get(k) != v:
                        why.append(f"{k} 불일치 {d.get(k)} ≠ {v}")
                if [round(x, 4) for x in d.get("cmd", [])] != [round(float(x), 4) for x in J["cmd"].split(",")]:
                    why.append(f"명령 불일치 {d.get('cmd')}")
                T = d.get("warmup_s", 2.0) + (d.get("seconds") or 0)
                if abs(d.get("steps_done", 0) - round(T / DT)) > 2:
                    why.append(f"완주 안 함 steps {d.get('steps_done')} ≠ {round(T / DT)}")
                core = [d.get(k) for k in ("alternation_rate", "fwd_mps", "lat_mps", "pitch_rms_deg", "action_rms")]
                if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in core):
                    why.append("핵심 지표 NaN(물리 발산 의심)")
                elif ((d.get("valid_steps") or 0) == 0 and d.get("gait_enable", 1) == 1) or (d.get("pitch_rms_deg") or 0) < 1e-6 or (d.get("action_rms") or 0) < 1e-6:
                    why.append("물리 정지 의심(유효 스텝 0 또는 피치·행동 RMS 0)")
                if d.get("fall_rate", 0) >= 0.999:
                    why.append("전 env 낙상(물리 폭발·자산 오류 의심 — 판정 대신 무효 처리)")
    else:
        tp = os.path.join(LP, f"ground_recovery_{J['tag']}.txt"); cp = os.path.join(LP, f"ground_recovery_{J['tag']}_envs.csv")
        if not (os.path.exists(tp) and os.path.exists(cp)):
            why.append("산출물 없음(txt/envs.csv)")
        elif os.path.getmtime(cp) < t0 or os.path.getmtime(tp) < t0:
            why.append("산출물이 이번 실행 전 파일")
        else:
            T = open(tp, errors="replace").read()
            arg = next((l for l in T.splitlines() if l.startswith("인자:")), "")
            if f"--seed {J['seed']} " not in arg + " " or f"--out_tag {J['tag']}" not in arg:
                why.append("인자 줄의 seed/out_tag 불일치")
            if "HangStand-G11a-v0 |" not in T:
                why.append("과제 표기 없음")
            if not re.search(r"사전 검사\(무개입 자립 [0-9.]*s\) 통과: [1-9]", T):
                why.append("사전 검사 통과 0(접촉 0 — OOM 등 조용한 실패 의심)")
            rows = list(csv.DictReader(open(cp)))
            if len(rows) != J["num_envs"]:
                why.append(f"CSV 행 {len(rows)} ≠ env {J['num_envs']}")
            # 물리 정지: 시간축 표의 생존 env 평균 피치가 전 구간 같은 값이면 정지
            tl = re.findall(r"^\s+(-?[0-9.]+)\s+([+-][0-9.]+|nan)\s+([0-9.]+)\s+(\d+)\s+([0-9.]+)\s*$", T, re.M)
            pv = [float(x[1]) for x in tl if x[1] != "nan"]
            if len(pv) < 5:
                why.append("시간축 표 없음/짧음(완주 안 함 의심)")
            elif max(pv) - min(pv) < 1e-3:
                why.append("물리 정지 의심(시간축 피치 불변)")
    if why:
        print("; ".join(why)); sys.exit(1)
    print("유효"); sys.exit(0)


def finish(run_dir):
    p = os.path.join(run_dir, "manifest.json"); man = json.load(open(p))
    st = {}
    sp = os.path.join(run_dir, "status.tsv")
    if os.path.exists(sp):
        for l in open(sp):
            x = l.rstrip("\n").split("\t")
            if len(x) >= 3:
                st[x[0]] = dict(status=x[1], tries=x[2], note=x[3] if len(x) > 3 else "", t_s=x[4] if len(x) > 4 else "")
    man["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); man["status"] = st
    json.dump(man, open(p, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "plan":
        plan(sys.argv[2], json.loads(sys.argv[3]))
    elif c == "check":
        check(*sys.argv[2:6])
    elif c == "finish":
        finish(sys.argv[2])
