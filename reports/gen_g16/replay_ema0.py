"""1001 러너 ema0 모드의 실물 로그 재생 검증 + 레드팀 반사실 재현 (개루프 — 폐루프 성능 증명이 아님).

입력: 0929 21:01 브리지 블랙박스(G12-9500, 0–2 s 무보조 뒤 손 보조). 기록된 관측의 최신 틀에서 러너 입력을 되살려
(1) v2 러너가 기록 관측 320 을 다시 만드는지(하네스 검증), (2) ema0 러너가 [260:320] 만 0 이고 나머지는 같은지,
(3) 같은 관측에서 EMA 경로가 만든 목표각 차이(Δ = π(v2) − π(ema0), 액션 × 0.25 rad → °)와 관절속도의 관계를 구간별로 잰다.
직전 액션은 기록값을 강제(개루프)해 관측 이력을 기록과 같게 유지한다."""
import json, math, os, sys
import numpy as np
REPO = '/home/ryu/humanoid_leg_test1'
sys.path.insert(0, REPO)
from rl_walking.deploy.policy_runner_stage4 import PolicyRunnerStage4, JOINT_ORDER  # noqa: E402

LOG = f'{REPO}/reports/real_0929/g12/rl_bridge_20260929_210129.jsonl'
EXP = f'{REPO}/logs/rsl_rl/biped12_stage4/2026-09-29_14-32-50_g12/exported'
POSE = [0.5282, -0.5282, 0.0, 0.0, 0.0, 0.0, -0.6720, 0.6720, 0.4887, -0.4887, 0.0, 0.0]


def main():
    recs = [json.loads(l) for l in open(LOG)]
    recs = [r for r in recs if r.get('obs') is not None]
    import importlib.util
    spec = importlib.util.spec_from_file_location('tbe', f'{REPO}/tests/test_bridge_ema_contract.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    pose = m.M.CKPT_DEFAULT_POSE['stand_crouch_g12']
    rv = PolicyRunnerStage4(EXP, gait_freq=1.1, default_pose_rad=pose)
    rz = PolicyRunnerStage4(EXP, gait_freq=1.1, default_pose_rad=pose, jvel_ema_mode='ema0')
    rv.reset(); rz.reset()
    t0 = recs[0]['timestamp']
    err_obs = err_raw = 0.0; dz = []; tt = []; qv = []; ema_rms = []
    prev_raw = np.zeros(12, np.float32)
    for r in recs:
        o = np.asarray(r['obs'], np.float32)
        gyro, grav, cmd = o[12:15], o[27:30], o[42:45]
        qpos = o[93:105] + np.asarray(pose, np.float32); qvel = o[153:165]; con = o[243:245]
        for rr in (rv, rz):
            rr.last_action = prev_raw.copy()
        rv.step(gyro, grav, cmd, qpos, qvel, con)
        rz.step(gyro, grav, cmd, qpos, qvel, con)
        ov, oz = rv.last_obs, rz.last_obs
        err_obs = max(err_obs, float(np.abs(ov - o).max()))
        assert np.array_equal(ov[:260], oz[:260]) and not np.any(oz[260:320])
        err_raw = max(err_raw, float(np.abs(rv.last_action - np.asarray(r['raw_action'], np.float32)).max()))
        dz.append(np.degrees(0.25 * (rv.last_action - rz.last_action)))
        tt.append(r['timestamp'] - t0); qv.append(qvel.copy()); ema_rms.append(float(np.sqrt(np.mean(rv.last_jvel_ema ** 2))))
        prev_raw = np.asarray(r['raw_action'], np.float32)
    dz = np.asarray(dz); tt = np.asarray(tt); qv = np.asarray(qv); ema_rms = np.asarray(ema_rms)
    print(f'틱 {len(recs)} · v2 재조립 관측 최대 오차 {err_obs:.2e} (JSON 4자리 반올림) · 기록 raw 재현 최대 오차 {err_raw:.3f}')
    print('ema0: [0:260] 비트 동일·[260:320] 전부 0 — 전 틱 확인')
    groups = {'힙앞뒤': [0, 1], '힙벌림': [2, 3], '힙회전': [4, 5], '무릎': [6, 7], '발목앞뒤': [8, 9], '발목롤': [10, 11]}
    print(f"\n{'구간':12s} {'EMA RMS p95':>11s} " + ' '.join(f'{g:>7s}' for g in groups) + '   (EMA 경로 Δ목표 RMS °, 슬루·LPF 전)')
    for a, b, name in ((0, 2, '0–2 s 무보조'), (2, 10, '2–10 s'), (10, 18, '10–18 s'), (19, 50, '19–50 s'), (50, 1e9, '50 s–')):
        k = (tt >= a) & (tt < b)
        if not k.any(): continue
        rms = {g: float(np.sqrt(np.mean(dz[k][:, idx] ** 2))) for g, idx in groups.items()}
        print(f"{name:12s} {np.percentile(ema_rms[k], 95):11.3f} " + ' '.join(f'{rms[g]:7.2f}' for g in groups))
    # 위상: Δ 가 관절속도와 같은 부호(반감쇠)인가 반대(감쇠)인가 — 관절별 상관, 19–50 s
    k = (tt >= 19) & (tt < 50)
    print('\n19–50 s EMA 경로 Δ목표 vs 관절속도 상관 (음수 = 속도 반대로 밀어 감쇠 쪽, 양수 = 반감쇠 쪽):')
    print('  ' + ' '.join(f'{JOINT_ORDER[j].replace("_joint", "")}:{np.corrcoef(dz[k][:, j], qv[k][:, j])[0, 1]:+.2f}' for j in range(12)))


if __name__ == '__main__':
    main()
