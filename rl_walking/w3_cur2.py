"""W3 vy·wz 커리큘럼 진급 판정 대안(1007, 선택 — 기본 꺼짐). 기존 w3_vy_curriculum·w3_wz_curriculum·추적기는 손대지 않는다(비트 동일).

점검 결과(reports/gen_w3/WD_DESIGN.md §6): W3S3d 학습 로그의 Curriculum/w3_vy·w3_wz 지표 nan 은 **첫 블록 전(학습 시작 ~6 반복)의 초기값**뿐이다.
진급 판정 입력은 학습 env 에서 매 틱 계산된다(블록 출력 98 회·117 회, 마지막 반복 lat_sign_ok 0.72·turn_sign_ok 0.87·yaw_gain 0.85).
진급이 안 된 원인 = **착지 사건별 부호 정답률 문턱 0.90 이 학습 분포에서 도달 불가**: vy 블록 0.69–0.76(98 블록 전부), wz 0.52–0.89(117 블록 전부 < 0.90).
같은 정책의 평가(고정 명령·지연 2·워밍업 뒤)는 vy ±0.05 97 %·±0.03 87–89 %, wz 98–100 % — 학습 블록 통계는 저속 상한(|vy| 0.03–0.04 → 착지 Δy 3–4 cm
≈ 걸음 흔들림 수준), 명령 전환·STEP 시작 직후 착지, 지연 1–5·DR 이 섞인 값(어느 것이 지배적인지는 미검증 — 가설).

대안(이 모듈): 사건 부호 대신 **한 보행 주기 평균 속도 이득**(가로 v̄_y·sign(vy)/|vy|, wz 쪽은 기존 yaw_gain 그대로)으로 판정하고, 명령 전환·STEP 시작 뒤
skip_s 동안의 표본은 뺀다. w3_wz 는 코드 변경 없이 params={"min_sign": 0.85} 로도 같은 효과(현재 0.86–0.89, 이득 0.85 ≥ 0.5).
"""
from __future__ import annotations

import torch

from . import mdp_hang as H
from . import w3 as W3


def _g(env):
    g = getattr(env, "_w3cur2", None)
    if g is None or g["age"].shape[0] != env.num_envs:
        N, dev = env.num_envs, env.device
        g = {"age": torch.zeros(N, device=dev), "prev_cmd": torch.zeros(N, 3, device=dev), "num": 0.0, "den": 0.0, "n": 0.0}
        env._w3cur2 = g
    return g


def w3_lat_gain_tracker(env, env_ids, cmin: float = 0.02, skip_s: float = 0.91):
    """mode="interval"(0 s): STEP·옆 명령 env 의 한 주기 이동평균 속도(W3._v_box, 요 프레임 y) 이득 누적. 명령이 바뀌거나 ge 가 꺼졌다 켜지면 skip_s 동안 제외."""
    g = _g(env)
    cmd = env.command_manager.get_command("base_velocity")
    ge = W3._ge(env)
    chg = (cmd - g["prev_cmd"]).abs().sum(1) > 1e-6
    fresh = env.episode_length_buf <= 1
    g["age"] = torch.where(chg | fresh | ~ge, torch.zeros_like(g["age"]), g["age"] + env.step_dt)
    g["prev_cmd"] = cmd.clone()
    m = ge & (cmd[:, 1].abs() >= cmin) & (g["age"] >= skip_s)
    if bool(m.any()):
        yaw = W3._yaw_of(env.scene[H._base_cfg().name].data.root_quat_w)
        vb = W3._v_box(env)
        vy_b = -vb[:, 0] * torch.sin(yaw) + vb[:, 1] * torch.cos(yaw)
        g["num"] += float((vy_b * torch.sign(cmd[:, 1]))[m].sum()); g["den"] += float(cmd[:, 1].abs()[m].sum()); g["n"] += float(m.sum())


def w3_vy_curriculum2(env, env_ids, vmax: tuple = (0.04, 0.06, 0.08), vmin: float = 0.03, min_gain: float = 0.6, max_bad: float = 0.02,
                      min_iters: int = 100, min_ticks: float = 50000.0, steps_per_iter: int = 24, start_stage: int = 0):
    """|vy| 상한 커리큘럼(w3_vy_curriculum 과 같은 단계·폭 위반 조건) — 판정 = 블록 평균 가로 속도 이득 ≥ min_gain(w3_lat_gain_tracker 필요).
    착지 사건 부호율은 로그로만 남긴다."""
    g = _g(env); L = W3._lat(env)
    c = getattr(env, "_w3_vycur2", None)
    if c is None:
        c = {"stage": int(start_stage), "since": 0, "b": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), "gain": float("nan"), "sign": float("nan"), "bad": float("nan")}
        env._w3_vycur2 = c
        env._w3_vy_range = (vmin, vmax[c["stage"]])
    cur = (g["num"], g["den"], g["n"], float(L["n_lat_ok"].sum()), float(L["n_lat_td"].sum()), float(L["n_bad"].sum()), float(L["n_td"].sum()))
    d = [a - b for a, b in zip(cur, c["b"])]
    it = int(env.common_step_counter) // steps_per_iter
    if d[2] >= min_ticks:
        c["gain"] = d[0] / max(d[1], 1e-9); c["sign"] = d[3] / max(d[4], 1.0); c["bad"] = d[5] / max(d[6], 1.0)
        adv = c["gain"] >= min_gain and c["bad"] < max_bad and it - c["since"] >= min_iters and c["stage"] < len(vmax) - 1
        print(f"[W3 vy 블록2] 반복 {it} |vy|≤{vmax[c['stage']]}: 가로 이득 {c['gain']:.3f} · 사건 부호 {c['sign']:.3f} · 폭 위반 {c['bad']:.3f}"
              + (" → 상한 확대" if adv else ""), flush=True)
        if adv:
            c["stage"] += 1; c["since"] = it; env._w3_vy_range = (vmin, vmax[c["stage"]])
        c["b"] = cur
    return {"vy_max": float(env._w3_vy_range[1]), "lat_gain": c["gain"], "lat_sign_ok": c["sign"], "width_bad": c["bad"]}
