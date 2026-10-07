"""계약 검증: 1004 배포본 감지기(RV3)를 기록 입력에 재생 → 기록된 c5.mode·n_on·ref_deg 와 틱 단위 대조."""
import sys, pickle, os
import numpy as np
sys.path.insert(0, ".")
import c5lib as L
R1004 = L.REPO + "/reports/real_1004/"
FILES = ["rl_bridge_20261004_221209.jsonl", "rl_bridge_20261004_221920.jsonl", "rl_bridge_20261004_223349.jsonl"]
for fn in FILES:
    D = L.load(R1004 + fn)
    pickle.dump(D, open(os.path.join(L.S, fn + ".pkl"), "wb"))
    dts = np.diff(D["ts"])
    print(f"\n== {fn} ck={D['ck']} 틱 {D['n']} · {D['t'][-1]:.2f} s · c5 필드 틱 {int(D['has_c5'].sum())} · c5 키 {D['c5keys']} · guard_mode {D['gm']}")
    print(f"   틱 간격: 중앙 {np.median(dts)*1e3:.2f} ms · p1 {np.percentile(dts,1)*1e3:.1f} · p99 {np.percentile(dts,99)*1e3:.1f} · 최대 {dts.max()*1e3:.1f} ms · >40 ms {int((dts>0.04).sum())} 회 · live&armed 전 틱 {bool((D['live']&D['armed']).all())}")
    print(f"   imu_mount_bias_deg 고유값 {np.unique(D['bias'],axis=0).tolist()} · rp_deg(원시)−bias 와 재구성 rp 최대차 {np.nanmax(np.abs((D['rp_raw']-D['bias'])-D['rp'])):.4f}°")
    for kind in ("rv3", "rv3now"):
        R = L.replay(D, kind)
        mm = int((R["mode"] != D["mode"]).sum()); nn = int((R["n_on"] != D["n_on"]).sum())
        dref = np.abs(R["ref"] - D["ref_log"]); 
        bad = np.where(R["mode"] != D["mode"])[0]
        print(f"   재생 {kind:6}: 모드 불일치 {mm}/{D['n']} 틱 · n_on 불일치 {nn} 틱 · |ref 재생 − 기록| 최대 {np.nanmax(dref):.4f}° (기록은 소수 3 자리) · 불일치 틱 예 {bad[:8].tolist()} t={np.round(D['t'][bad[:8]],2).tolist()}")
    # switch 필드와 mode 전이 일치
    on_i = [i for i in range(1, D["n"]) if D["mode"][i] == 1 and D["mode"][i-1] == 0]; off_i = [i for i in range(1, D["n"]) if D["mode"][i] == 0 and D["mode"][i-1] == 1]
    sw_on = np.where(D["sw"] == "on")[0].tolist(); sw_off = np.where(D["sw"] == "off")[0].tolist()
    print(f"   기록 switch 'on' 틱 {sw_on} = 모드 0→1 틱 {on_i} : {sw_on == on_i} · 'off' 틱 {sw_off} = 1→0 틱 {off_i} : {sw_off == off_i} · 마지막 n_on {int(D['n_on'][-1])}")
    print(f"   사건 기록: {[(round(e.get('timestamp',0)-D['ts'][0],2), e.get('event'), (e.get('state') or e.get('why') or e.get('reason') or '')) for e in D['ev']][:30]}")
    print(f"   마지막 틱 t={D['t'][-1]:.2f} s mode={int(D['mode'][-1])} stop_reason={D['stop'][-1]} hang_state={D['hs'][-1]}")
