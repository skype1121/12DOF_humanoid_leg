"""1005 계약 대조: 시뮬 eval 덤프(recbb_<tag>_env*.jsonl 의 par_* = 감지기 입력·출력)를 배포 모듈 c5_switch 에 재생해
모드·cue 시퀀스를 틱 단위로 비교한다. 로그 재생은 계약 검증이지 폐루프 성능 증명이 아니다(CLAUDE.md).
사용: python3 replay_kr.py <tag>"""
import sys, json, glob, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../rl_walking/deploy"))
import numpy as np
import c5_switch as C

tag = sys.argv[1]
REFRAC = tuple(float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1.0,6,1.0").split(","))
INF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0      # krF: 감지 입력 EMA α
files = sorted(glob.glob(f"/home/ryu/humanoid_leg_test1/log_picture/recbb_{tag}_env*.jsonl"))
tot = mism_m = mism_c = 0; n_on = n_cue = 0
for fp in files:
    det = C.SwitchDetector(on_tilt_deg=3.0, on_w=0.5, off_tilt_deg=1.5, off_w=0.3, calm_s=1.0, dwell_s=0.4, ref_tau_s=10.0, ref_init_s=1.5,
                           off_contact="any", early=(0.6, 6.0, 0.4, 0.2), refrac=REFRAC, in_filt=INF)
    run = C.C5Runner.__new__(C.C5Runner); run.det = det; run.cue_vx = 0.03; run.cue_s = 1.0; run.cue_dir_imu = True
    run.cue_active = False; run.cue_t = 0.0; run.n_cue = 0
    prev = 0; first = None
    for line in open(fp):
        r = json.loads(line)
        if "par_mode" not in r:
            continue
        m = det.update(r["par_g"], r["par_w"], r["par_c"])
        cue = run.cue_update(m, prev, r["par_w"])
        prev = m; tot += 1
        if m != r["par_mode"]:
            mism_m += 1
            if first is None: first = (os.path.basename(fp), r["t_rel"], "mode", m, r["par_mode"])
        if int(cue) != r["par_cue"]:
            mism_c += 1
            if first is None: first = (os.path.basename(fp), r["t_rel"], "cue", int(cue), r["par_cue"])
    n_on += det.n_on; n_cue += run.n_cue
    if first: print("첫 불일치:", first)
print(f"{tag}: env {len(files)} · 틱 {tot} · 모드 불일치 {mism_m} · cue 불일치 {mism_c} · 재생 켜짐 {n_on} · cue {n_cue}")
