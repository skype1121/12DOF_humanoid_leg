"""1001 아침 실물 시험 분석 — 브리지 블랙박스 1개 → 지연·신선도·흔들림·EMA 요약 + 시뮬 지연 조회표(G12-9500 v2).

사용: python3 reports/gen_g16/morning_analyze.py ~/logs/rl_bridge_YYYYMMDD_HHMMSS.jsonl [시작s 끝s]
측정 사실(기록값)·계산(지연 추정)·해석을 구분해 출력한다. 자립 성공 판정은 하지 않는다(줄 상태·영상·운영자 확인 필요 — CLAUDE.md).

지연 매핑(G16_CRITERIA §6, 사전 고정): d*_cmd = ceil(p95(브리지 발행→stage8 처리 ms) + 20 ms(stage8 송신 주기 상한)) / 20 → {1,3,6} 중 가장 가까운 값.
보조 추정 d*_resp = (목표→엔코더 상관 최대 지연 틱) − 5 (0930 분해: 시뮬 반응 ≈ 액션 지연 + ~5틱). 둘이 2틱 넘게 어긋나면 둘 다 평가.
"""
import csv, json, math, os, sys
import numpy as np

JOINTS = ['left_hip_f_joint', 'right_hip_f_joint', 'left_hip_a_joint', 'right_hip_a_joint', 'left_hip_r_joint', 'right_hip_r_joint',
          'left_knee_joint', 'right_knee_joint', 'left_ankle_f_joint', 'right_ankle_f_joint', 'left_ankle_r_joint', 'right_ankle_r_joint']
LP = '/home/ryu/humanoid_leg_test1/log_picture'
DT = 0.02


def load(path, a=None, b=None):
    rows = [json.loads(l) for l in open(path)]
    t = [r for r in rows if 'event' not in r and r.get('qpos_rad') is not None]
    ev = [r for r in rows if 'event' in r]
    t0 = t[0]['timestamp']
    for r in t:
        r['_t'] = r['timestamp'] - t0
    if a is not None:
        t = [r for r in t if a <= r['_t'] < b]
    return t, ev, t0


def xlag(x, y, maxl=40):
    """y 가 x 보다 몇 틱 늦나(미분 신호 상관 최대)."""
    x = np.diff(x); y = np.diff(y)
    x = x - x.mean(); y = y - y.mean()
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None, 0.0
    best = (0, -2.0)
    for L in range(0, maxl + 1):
        c = np.corrcoef(x[:len(x) - L], y[L:])[0, 1]
        if c > best[1]:
            best = (L, c)
    return best


def sim_row(d):
    """G12-9500 v2 관측 시뮬 결과(eval seed 1–3 합) — 무외란 낙상·흔들림 지속·감쇠, 뒤 8 N 성장·낙상."""
    def stat(tag):
        n = gf = sus = damp = fall = 0
        for sd in (1, 2, 3):
            p = f'{LP}/ground_recovery_osc15_g12v2_9500_d{d}_{tag}_s{sd}_envs.csv'
            if not os.path.exists(p):
                return None
            for r in csv.DictReader(open(p)):
                if r['pre_pass'] != '1':
                    continue
                n += 1; pp = [float(r[f'p2p_{k}']) if r[f'p2p_{k}'] else math.nan for k in range(14)]
                if r['fell'] == '1':
                    fall += 1; gf += float(r['t_fall_rel'] or 0) > 3; continue
                if max(pp[6:14]) >= 4 and max(pp[6:14]) >= 2 * max(pp[1:4]): gf += 1
                if max(pp[10:14]) <= 2: damp += 1
                else: sus += 1
        return n, 100 * fall / n, 100 * sus / n, 100 * damp / n, 100 * gf / n
    nd, b8 = stat('nd'), stat('b8')
    if nd is None:
        return f'd{d}: 시뮬 결과 없음'
    s = f'd{d}: 무외란 낙상 {nd[1]:.1f} % · 흔들림 지속 {nd[2]:.1f} % · 감쇠 {nd[3]:.1f} %'
    if b8:
        s += f' | 뒤 8 N 성장·낙상 {b8[4]:.1f} %'
    return s + f' (n={nd[0]})'


def main():
    path = sys.argv[1]
    a = float(sys.argv[2]) if len(sys.argv) > 3 else None
    b = float(sys.argv[3]) if len(sys.argv) > 3 else None
    t, ev, t0 = load(path, a, b)
    live = [r for r in t if r.get('live')]
    ts = np.array([r['_t'] for r in t])
    print(f'== {os.path.basename(path)} — 틱 {len(t)} ({ts[-1] - ts[0]:.1f} s), live {len(live)}')
    ck = sorted({r.get('checkpoint') for r in t if r.get('checkpoint')})
    modes = sorted({r.get('jvel_ema_mode', '(기록 없음=v2 이전판)') for r in t if r.get('obs') is not None})
    print(f'   정책: {ck} · jvel_ema 모드: {modes}')
    evc = {}
    for e in ev:
        evc[e['event']] = evc.get(e['event'], 0) + 1
    print(f'   이벤트: {evc}')

    print('\n[1] 명령 경로 (측정: stage8 이 잰 브리지 발행 → 처리)')
    s8 = [r['s8_cmd'] for r in t if isinstance(r.get('s8_cmd'), dict)]
    dcmd = None
    if not s8:
        print('   s8_cmd 없음 (구 stage8 또는 미수신)')
    else:
        age = np.array([x['last_age_ms'] for x in s8 if x.get('last_age_ms') is not None], float)
        cnt = np.array([x['count_1s'] for x in s8 if x.get('count_1s') is not None], float)
        gaps = max((x.get('seq_gaps') or 0) for x in s8); stale = sum(1 for x in s8 if x.get('stale'))
        p95 = float(np.percentile(age, 95))
        print(f'   나이 p50 {np.percentile(age, 50):.1f} · p95 {p95:.1f} · 최대 {age.max():.1f} ms | 초당 처리 중앙 {np.median(cnt):.0f} (최소 {cnt.min():.0f})'
              f' | 순번 누락 누적 {gaps} · stale 틱 {stale}')
        raw = math.ceil((p95 + 20.0) / 20.0)
        dcmd = min((1, 3, 6), key=lambda g: (abs(g - raw), g))
        print(f'   → (계산) ceil((p95 + 20 ms)/20) = {raw}틱 → 사전 규칙 d*_cmd = {dcmd}')

    print('\n[2] 목표 → 엔코더 응답 지연 (계산: 미분 상관 최대, live 구간)')
    L = live if len(live) > 200 else t
    q = np.degrees(np.array([r['qpos_rad'] for r in L], float))
    tg = np.array([[r['targets_deg_joint'][j] for j in JOINTS] for r in L], float)
    lags = []
    for name, idx in (('힙앞뒤', (0, 1)), ('무릎', (6, 7)), ('발목앞뒤', (8, 9)), ('발목롤', (10, 11)), ('힙벌림', (2, 3))):
        out = []
        for j in idx:
            lg, c = xlag(tg[:, j], q[:, j])
            if lg is not None and c > 0.3:
                out.append((lg, c)); lags.append(lg)
        print(f'   {name:5s}: ' + (' / '.join(f'{lg}틱 (r={c:.2f})' for lg, c in out) if out else '상관 부족(명령 변화 작음)'))
    dresp = None
    if lags:
        med = float(np.median(lags)); dresp = max(0, int(round(med - 5)))
        print(f'   → 중앙 {med:.1f}틱 ({med * 20:.0f} ms) → (계산) d*_resp = {dresp}   [0929 기준값: 16–17틱]')
    if dcmd is not None and dresp is not None and abs(dcmd - dresp) > 2:
        print(f'   ⚠ d*_cmd {dcmd} 과 d*_resp {dresp} 가 2틱 넘게 다름 — 둘 다 평가 (사전 규칙)')

    print('\n[3] 관절 상태 신선도 (측정: 직전 틱과 같은 qpos 비율)')
    qr = np.array([r['qpos_rad'] for r in L], float)
    same = (np.diff(qr, axis=0) == 0)
    mov = np.abs(np.diff(tg, axis=0)) > 0.2
    allr = same.mean(0); movr = np.array([same[mov[:, j], j].mean() if mov[:, j].sum() > 20 else np.nan for j in range(12)])
    print(f'   전체 평균 {100 * allr.mean():.0f} % (관절 {100 * allr.min():.0f}–{100 * allr.max():.0f} %) · 명령이 움직이는 틱만 {100 * np.nanmean(movr):.0f} %'
          '   [0929: 전체 55–85 %, 평균 72 %]')

    print('\n[4] 흔들림 (측정: 피치 1 s 창 최대−최소, 자이로 y 주파수, 양발 접촉)')
    th = np.array([r['rp_deg'][1] if r.get('rp_deg') else np.nan for r in t], float)
    gy = np.array([r['gyro_rad_s'][1] for r in t], float)
    ct = np.array([r['contact'] for r in t], float)
    print(f"   {'구간':>9s} {'p2p°':>6s} {'주파Hz':>6s} {'양발%':>6s} {'EMA_RMS':>8s}")
    def _ema(r):   # 새 브리지 = jvel_ema_rms, 옛 기록 = 관측 joint_vel_ema 블록의 최신 틀(obs[308:320]; v2 동작이던 시기)
        if r.get('jvel_ema_rms') is not None:
            return r['jvel_ema_rms']
        o = r.get('obs')
        return float(np.sqrt(np.mean(np.square(o[308:320])))) if o is not None and len(o) >= 320 else np.nan
    ema = np.array([_ema(r) for r in t], float)
    W = 5.0
    for s0 in np.arange(ts[0], ts[-1], W):
        m = (ts >= s0) & (ts < s0 + W)
        if m.sum() < 50:
            continue
        p2p = np.nanmax([np.nanmax(th[m & (ts >= s0 + k) & (ts < s0 + k + 1)]) - np.nanmin(th[m & (ts >= s0 + k) & (ts < s0 + k + 1)])
                         for k in range(int(W)) if (m & (ts >= s0 + k) & (ts < s0 + k + 1)).sum() > 10])
        x = gy[m] - gy[m].mean(); F = np.abs(np.fft.rfft(x * np.hanning(len(x)))); fr = np.fft.rfftfreq(len(x), DT); k = np.argmax(F[1:]) + 1
        both = (ct[m].min(1) > 0.5).mean()
        e = np.nanpercentile(ema[m], 95) if np.isfinite(ema[m]).any() else np.nan
        print(f'   {s0:4.0f}-{s0 + W:3.0f}s {p2p:6.1f} {fr[k]:6.2f} {100 * both:6.0f} {e:8.3f}')
    print('   (EMA_RMS = 실제 관절속도 EMA RMS p95 — 학습 legacy 분포 p95 0.078. ema0 모드에서도 계산만 되고 정책에는 0.)')

    print('\n[5] 시뮬 기대치 (G12-9500, v2 관측, eval seed 1–3) — 실측 지연에 해당하는 행')
    for d in sorted({x for x in (dcmd, dresp) if x is not None} | {0, 1, 2, 3}):
        print('   ' + sim_row(d))
    print('\n해석 주의: 시뮬 수치는 실물 성공 확률이 아니다(플랜트 괴리: 유격·무릎 처짐·샘플홀드). 실측 서명과의 일치 여부로만 쓴다.')


if __name__ == '__main__':
    main()
