"""학습 로그(tfevents)의 보상 항별 크기 → 걷기(STEP) 초당 환산 — tensorboard 없이 읽는다(TFRecord + protobuf 최소 파서, CPU).

사용: python3 reports/gen_w3/reward_scale.py [RUN_DIR] [--it 15750,15850]
  기본 RUN_DIR = logs/rsl_rl/biped12_stepb0/2026-10-07_00-11-12_w3S3d_from_s2d (W3S3d, 채택 체크포인트 15800 부근 창).

Isaac Lab 의 Episode_Reward/<항> = 에피소드 합(가중·dt 곱) / max_episode_length_s = 에피소드 20 s 기준 초당 평균(서기 30 %·걷기 70 % env 가 섞임).
걷기 초당 = 그 값 ÷ 걷기 시간 비율, 걷기 시간 비율 = Episode_Reward/w3_alive ÷ 0.5(w3_alive 는 ge 1 일 때 가중 0.5 로 1 을 준다).
점검: 항 합 × 20 s 가 Train/mean_reward 와 맞는지 함께 찍는다. TORQUE_W3T_DESIGN.md 의 '걷기 순 보상 +1.28/s' 출처.
"""
import argparse
import glob
import os
import struct

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _varint(b, i):
    r = 0; s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s
        if not (c & 0x80):
            return r, i
        s += 7


def _fields(b):
    i = 0; n = len(b)
    while i < n:
        key, i = _varint(b, i); f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v = b[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = _varint(b, i); v = b[i:i + ln]; i += ln
        elif wt == 5:
            v = b[i:i + 4]; i += 4
        else:
            raise ValueError(f"wire type {wt}")
        yield f, wt, v


def read_scalars(path):
    """{tag: [(step, value), ...]} — Event(step=2, summary=5) → Summary.value(1) → tag(1)·simple_value(2)."""
    out = {}
    data = open(path, "rb").read(); i = 0; n = len(data)
    while i + 12 <= n:
        ln = struct.unpack("<Q", data[i:i + 8])[0]; i += 12
        rec = data[i:i + ln]; i += ln + 4
        step = 0; summ = None
        for f, wt, v in _fields(rec):
            if f == 2 and wt == 0:
                step = v
            elif f == 5 and wt == 2:
                summ = v
        if summ is None:
            continue
        for f, wt, v in _fields(summ):
            if f != 1 or wt != 2:
                continue
            tag = val = None
            for f2, w2, v2 in _fields(v):
                if f2 == 1 and w2 == 2:
                    tag = v2.decode("utf-8", "replace")
                elif f2 == 2 and w2 == 5:
                    val = struct.unpack("<f", v2)[0]
            if tag is not None and val is not None:
                out.setdefault(tag, []).append((step, val))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=os.path.join(REPO, "logs", "rsl_rl", "biped12_stepb0", "2026-10-07_00-11-12_w3S3d_from_s2d"))
    ap.add_argument("--it", default="15750,15850")
    a = ap.parse_args(); lo, hi = [int(x) for x in a.it.split(",")]
    files = sorted(glob.glob(os.path.join(a.run, "events.out.tfevents.*")))
    if not files:
        raise SystemExit(f"tfevents 없음: {a.run}")
    d = {}
    for f in files:
        for k, v in read_scalars(f).items():
            d.setdefault(k, []).extend(v)

    def avg(tag):
        xs = [v for s, v in d.get(tag, []) if lo <= s <= hi]
        return sum(xs) / len(xs) if xs else float("nan")
    rew = {t.split("/", 1)[1]: avg(t) for t in d if t.startswith("Episode_Reward/")}
    ge = rew["w3_alive"] / 0.5
    print(f"# {os.path.basename(a.run)} · 반복 {lo}–{hi} · 평균 에피소드 길이 {avg('Train/mean_episode_length'):.0f} 틱 · 평균 보상 {avg('Train/mean_reward'):.1f} · 걷기(ge 1) 시간 비율 {ge:.3f}")
    print(f"종료 비율: time_out {avg('Episode_Termination/time_out'):.3f} · bad_orientation {avg('Episode_Termination/bad_orientation'):.3f} · base_too_low {avg('Episode_Termination/base_too_low'):.3f}")
    step = sorted([k for k in rew if k.startswith("w3_")], key=lambda k: -abs(rew[k]))
    stand = sorted([k for k in rew if not k.startswith("w3_") and k != "termination_penalty"], key=lambda k: -abs(rew[k]))
    print("\n| 걷기 항 | 에피소드 초당 | 걷기 초당 |"); print("|---|---|---|")
    for k in step:
        print(f"| {k} | {rew[k]:+.4f} | {rew[k] / ge:+.3f} |")
    tot = sum(rew[k] for k in step); pos = sum(rew[k] for k in step if rew[k] > 0); neg = sum(rew[k] for k in step if rew[k] < 0)
    print(f"| 합 | {tot:+.4f} | {tot / ge:+.3f} (양수 {pos / ge:+.3f} · 음수 {neg / ge:+.3f}) |")
    tot2 = sum(rew[k] for k in stand); allr = tot + tot2 + rew.get("termination_penalty", 0.0)
    print(f"\n서기 전용 항 합(에피소드 초당) {tot2:+.4f} · 종료 벌점 {rew.get('termination_penalty', 0.0):+.4f} · 전체 {allr:+.4f} → ×20 s = {20 * allr:.1f} (로그 평균 보상 {avg('Train/mean_reward'):.1f} 와 대조)")
    print("서기 전용 항 상위: " + " · ".join(f"{k} {rew[k]:+.4f}" for k in stand[:10]))
    print("토크·에너지 계열(전부 서기 전용 = w3_stand_gated): " + " · ".join(f"{k} {rew[k]:+.5f}" for k in ("torque_saturation", "dof_torques_l2", "dof_acc_l2", "joint_vel_l2", "action_rate_l2") if k in rew))
    print("커리큘럼: " + " · ".join(f"{t.split('/', 1)[1]} {avg(t):.3f}" for t in sorted(d) if t.startswith("Curriculum/")))


if __name__ == "__main__":
    main()
