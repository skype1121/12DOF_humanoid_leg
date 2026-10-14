import json
def L(t): return json.load(open(f'log_picture/w3eval_{t}.json'))
a, b = L('rs_id2_s4_d2'), L('rs_off_s4_d2')
ga = [(g['falls'], g['recovered'], g['n']) for g in a['dist']['groups']]
gb = [(g['falls'], g['recovered'], g['n']) for g in b['dist']['groups']]
print('bit-identity', ga == gb, a['fall_score'], b['fall_score'], a['fwd_mps'] == b['fwd_mps'], a['valid_steps'] == b['valid_steps'])
for t in ('rs_id2_s4_d2', 'rs_cal_s4_d4', 'rs_cal_s3_d2', 'rs_cal_s3_d4'):
    r = L(t)['rsup']
    print(t, {k: round(v, 2) for k, v in r['dev0_q'].items()},
          {k[5:]: (round(v['mean_dpitch'], 2), round(v['mean_droll'], 2), round(v['p50_mag'], 2)) for k, v in r.get('dev_push_by_group', {}).items()})
