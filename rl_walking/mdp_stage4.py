"""Stage4 전용 mdp 함수 — 게이트 클록 관측 + 접촉 관측 + 주기 보상.

설계 근거:
- Siekmann et al. 2021 "Sim-to-Real Learning of All Common Bipedal Gaits via
  Periodic Reward Composition": 위상 클록으로 스탠스/스윙 구간을 정의하고
  구간-접촉 일치에 보상 — 좌우 π 오프셋이 대칭 걸음을 '정의 자체'로 강제.
  (Stage3 대칭 벌점 2회 실패 교훈: 수렴 후 패치가 아니라 초기부터 위상으로 형성)
- 접촉 관측은 실물 RA30P 발바닥 힘센서 대응 — 이진값(>5N)이라 sim2real 갭 최소.

시뮬 전용 — 실물 모터/CAN에는 아무것도 보내지 않는다.
"""
from __future__ import annotations

import math

import torch

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg

# 게이트 주파수 [Hz] — 케이던스 2.8보/s (= 1.4Hz × 양발). Stage3 실측 뚜벅걸음 대역.
GAIT_FREQ_HZ = 1.4


def _gait_phase(env, freq: float) -> torch.Tensor:
    """에피소드 경과시간 기반 게이트 위상 φ = 2π·f·t. (N,) [rad].

    episode_length_buf는 리셋 시 0 — 모든 env가 φ=0(왼발 스탠스 시작)에서 출발.
    관측(gait_clock)과 보상(gait_phase_reward)이 반드시 이 함수를 공유할 것
    (위상 정의가 어긋나면 클록이 무의미해진다).
    """
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    return 2.0 * math.pi * freq * t


def gait_clock(env, freq: float = GAIT_FREQ_HZ) -> torch.Tensor:
    """게이트 클록 관측 [sin φ, cos φ]. (N, 2).

    실물 이식 시 배포 노드가 동일 공식(2π·1.4·t)으로 재현 가능 — 외부 입력 불필요.
    """
    phase = _gait_phase(env, freq)
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)


def feet_contact_binary(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 5.0,
) -> torch.Tensor:
    """발 접촉 이진 관측 (N, 2) — [왼발, 오른발] 순서, 법선힘 노름 > threshold[N].

    실물 RA30P 발바닥 힘센서 대응. sensor_cfg.body_ids는 SceneEntityCfg가
    '이름으로' 해석한 인덱스 (preserve_order=True 필수 — 접촉센서 바디 순서는
    USD DFS 순회라 좌/우가 이름 순서와 다를 수 있는 함정, log_picture/03 참조).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 2, 3)
    return (torch.norm(forces, dim=-1) > threshold).to(torch.float32)


def gait_phase_reward(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    freq: float = GAIT_FREQ_HZ,
    force_threshold: float = 5.0,
    always_walk: bool = False,
) -> torch.Tensor:
    """Siekmann 2021 간이판 주기 보상 — 위상-접촉 일치도. 값 0~1.

    왼발 스탠스 = sin(φ) ≥ 0 구간, 오른발 스탠스 = sin(φ) < 0 (π 오프셋).
    reward = Σ_발 [ I(자기 스탠스구간)·I(접촉) + I(스윙구간)·I(비접촉) ] / 2
           = ( I(왼스탠스)==I(왼접촉) + I(오른스탠스)==I(오른접촉) ) / 2.
    서기 명령(|cmd_xy| + |wz| < 0.1)은 클록을 무시하고 양발 접촉 보상으로 대체
    — 서기 env가 제자리 발구름을 하지 않게.

    게이트에 |wz| 포함 (Stage4 v2 적대 리뷰 수정): 선속도 노름만 보면
    제자리회전 env(vx=vy=0, wz≠0)가 '서기'로 오분류되어 양발 접지 보상(2.0)이
    스텝 턴을 벌하고 접지 비틀기(요-슬립 게이밍)를 정확히 조장한다.
    서기 env는 명령 3성분 전부 0이라 이 변경에 영향 없음.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 2, 3) [L, R]
    contact = torch.norm(forces, dim=-1) > force_threshold  # (N, 2) bool

    sin_phi = torch.sin(_gait_phase(env, freq))
    left_stance = sin_phi >= 0.0  # (N,) bool — 오른발 스탠스는 그 여집합 (π 오프셋)

    walk_reward = (
        (left_stance == contact[:, 0]).to(torch.float32)
        + ((~left_stance) == contact[:, 1]).to(torch.float32)
    ) * 0.5

    stand_reward = (contact[:, 0] & contact[:, 1]).to(torch.float32)

    if always_walk:
        # march(제자리 걷기)판: 명령 0에서도 클록 리듬 스텝 — 서기 보상 없음.
        # 이 플래그를 쓰는 정책은 '조용히 서기'가 불가능해지므로 배포 노드에서
        # 정지 상태는 반드시 일반 체크포인트로 전환할 것.
        return walk_reward

    cmd_vec = env.command_manager.get_command(command_name)
    cmd = torch.norm(cmd_vec[:, :2], dim=1) + torch.abs(cmd_vec[:, 2])
    return torch.where(cmd > 0.1, walk_reward, stand_reward)


def stand_still_full_command(
    env,
    command_name: str = "base_velocity",
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """서기(전 명령 ≈ 0) env만 기본자세 이탈 벌점 — SE(2) 3성분 전부 게이트.

    isaaclab_tasks의 stand_still_joint_deviation_l1은 선속도 노름만 보고
    게이트해서 제자리회전 env(vx=vy=0, wz≠0)까지 '서기'로 오분류 —
    스텝 턴에 필요한 관절 이탈을 벌점하는 충돌 (Stage4 v2 적대 리뷰 수정).
    v2 cfg가 stand_still.func를 이 함수로 교체한다 (v1 계열은 원본 유지).
    """
    cmd = env.command_manager.get_command(command_name)
    cmd_mag = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
    return mdp.joint_deviation_l1(env, asset_cfg) * (cmd_mag < command_threshold)


def march_position_hold(
    env,
    command_name: str = "base_velocity",
    dead_zone: float = 0.3,
    cap: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """명령 0(march) env의 스폰 원점 XY 이탈 벌점 — 제자리 드리프트 앵커.

    march 1라운드 실측: 리듬·대칭 완성에도 30s에 평균 2.0m 표류 — 명령 0
    속도추종은 영점 부근 기울기가 없어 위치 누적오차를 못 잡는다. 원점 거리
    데드존(리셋 산포 ±0.3m 허용) 밖 선형 벌점으로 위치를 앵커. cap은 방폭
    관례(물리폭발 시 원거리 순간치 폭주 차단). 이동 명령 env는 벌점 0.
    """
    asset = env.scene[asset_cfg.name]
    dist = torch.norm(
        asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    pen = (dist - dead_zone).clamp(min=0.0, max=cap)
    cmd = env.command_manager.get_command(command_name)
    zero_cmd = (torch.norm(cmd[:, :2], dim=1) + cmd[:, 2].abs()) < 0.1
    return torch.where(zero_cmd, pen, torch.zeros_like(pen))


def foot_yaw_slip(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 5.0,
    cap: float = 20.0,
) -> torch.Tensor:
    """접지 요-슬립 벌점 (Stage4 v2) — 접촉 중인 발의 월드 z축 |각속도| 합.

    제자리회전 명령을 '접지한 발바닥 비틀기(트위스트 슬립)'로 게이밍하지 않고
    발을 들어 딛는 스텝 턴으로 돌도록 유도. cmd 게이팅 없음 — 전진 보행 중에도
    접지발의 요 회전은 물리적으로 슬립이므로 일괄 벌점 (양수 반환, weight 음수).

    구현: 접촉(법선힘 노름 > threshold[N]) 마스크 × |body_ang_vel_w[..., 2]| 합,
    cap으로 클램프. 캡은 방폭 캡핑 5종과 같은 규약 (mdp_rewards 참조) — 실측
    정상값 최대 3.1(4096env 스모크) 대비 ~6배 여유라 학습 신호 불변, 계단
    모서리 물리폭발 시 각속도 스파이크(수백 rad/s)의 보상 오염만 차단.
    weight -0.5 × cap 20 = 스텝당 최대 -10 (다른 캡 항들과 동일 대역).
    sensor_cfg(접촉센서)와 asset_cfg(로봇 바디)는 반드시 같은 발 순서
    [왼발, 오른발] + preserve_order=True로 페어를 맞출 것 — 접촉센서 바디
    나열은 USD DFS 순회라 이름 순서와 다를 수 있는 함정 (log_picture/03).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 발, 3)
    in_contact = (torch.norm(forces, dim=-1) > threshold).to(torch.float32)  # (N, 발)
    asset = env.scene[asset_cfg.name]
    foot_yaw_vel = torch.abs(asset.data.body_ang_vel_w[:, asset_cfg.body_ids, 2])  # (N, 발)
    return (foot_yaw_vel * in_contact).sum(dim=1).clamp_max(cap)


def knee_stance_flexion(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    band_rad: tuple = (0.14, 0.35),
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """지지발 무릎 굴곡 밴드 보상 (0803 실물 1차 지시 반영). 값 0~1.

    실물 관찰: 뻣뻣한 직립 스탠스는 '아슬아슬'(충격 흡수 불능) + march_v3는
    무릎을 과신전 스톱에 눌러 기대는 니락 전략까지 학습(스탠스 3.4% 틱).
    처방: 접촉 중인 다리의 무릎 굴곡을 8~20°(0.14~0.35rad) 밴드로 유도 —
    "약간 굽혀 안정적으로 눌러주는" 스탠스. 굴곡 부호는 좌우 반대
    (왼무릎 −=굴곡, 오른무릎 +=굴곡)이므로 부호 보정 후 공통 밴드 적용.
    밴드 밖은 0.15rad당 1씩 선형 감쇠 (과신전=음수 굴곡도 자연히 벌점 방향).
    asset_cfg.joint_ids = [left_knee, right_knee] (preserve_order=True).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (N, 2, 3) [L, R]
    in_contact = (torch.norm(forces, dim=-1) > force_threshold).to(torch.float32)
    asset = env.scene[asset_cfg.name]
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, 2) [L, R]
    flex = torch.stack([-knee[:, 0], knee[:, 1]], dim=1)  # 굴곡 + 로 통일
    lo, hi = band_rad
    dist = torch.clamp(lo - flex, min=0.0) + torch.clamp(flex - hi, min=0.0)
    per_leg = torch.clamp(1.0 - dist / 0.15, min=0.0)  # 밴드 안=1, 밖 선형 감쇠
    # 접촉 중인 다리만 평가 (스윙 다리는 자유) — 접촉 없으면 중립 0.5
    n_contact = in_contact.sum(dim=1)
    reward = (per_leg * in_contact).sum(dim=1) / n_contact.clamp(min=1.0)
    return torch.where(n_contact > 0, reward, torch.full_like(reward, 0.5))


def feet_impact_force(
    env,
    sensor_cfg: SceneEntityCfg,
    max_force_n: float = 180.0,
    cap: float = 400.0,
) -> torch.Tensor:
    """착지 충격 벌점 (0803 실물 1차 '쿵' 반영) — 초과 수직 접촉력 합 [N].

    실물 관찰: march_v3 착지가 "쿵하고 너무 쌔게" — 시뮬보다 단단한 실바닥에서
    충격이 요·병진 교란으로 전이(첫 스텝 요 비틀림 낙하의 후보 기여).
    체중 118N(12kg) 기준 1.5배(180N) 초과분을 선형 벌점. cap은 방폭 관례.
    weight 예: -0.01 → 300N 착지 시 -1.2/틱 (부드러운 착지로 케이던스 이내
    해결 가능한 대역).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    fz = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # (N, 발) 수직
    excess = (fz - max_force_n).clamp(min=0.0)
    return excess.sum(dim=1).clamp_max(cap)


# ---------------------------------------------------------------------------
# 실물 관측 현실화 (DRv6, 2026-08-03 stand_v3 발진 2건 블랙박스·버스 전수 분석)
# ---------------------------------------------------------------------------

_ANKLE_F_IDS_CACHE: dict = {}


def _ankle_f_ids(env, asset_name: str):
    """asset 관절 배열에서 ankle_f 인덱스 (env 인스턴스별 1회 계산)."""
    key = (id(env), asset_name)
    if key not in _ANKLE_F_IDS_CACHE:
        names = env.scene[asset_name].joint_names
        _ANKLE_F_IDS_CACHE[key] = [
            i for i, n in enumerate(names) if "ankle_f" in n]
    return _ANKLE_F_IDS_CACHE[key]


def joint_pos_rel_staircase(
    env,
    quant_step_rad: float = 0.012,
    ankle_dither_rad: float = 0.0175,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절각 관측의 실물 계단·디더 재현 — joint_pos_rel 대체.

    실측 근거(0803 블랙박스): 로터 엔코더 관측은 연속이 아니라
    freeze-then-jump — 정지 시 틱의 48~71%가 홀드, 갱신 시 스텝 p90 0.8~2°
    (양자화 0.7°≈0.012rad로 근사). 발목F는 감속기 백래시+프린트 컴플라이언스로
    ±1°(0.0175rad) 저주파 디더가 상시 관측됨 (힙의 10배). 시뮬의 매끈한
    실관절각만 본 정책은 이 분포에서 잔여 오차를 실신호로 오인해 과반응
    (5~9Hz 정책 주도 발진의 관측측 기여) — 정책이 이를 무시하도록 학습 분포에
    주입한다. 형상·순서 불변(symmetry 미러 계약 유지).
    """
    q = mdp.joint_pos_rel(env, asset_cfg)
    q = torch.round(q / quant_step_rad) * quant_step_rad
    ids = _ankle_f_ids(env, asset_cfg.name)
    if ids and ankle_dither_rad > 0.0:
        dither = (torch.rand(q.shape[0], len(ids), device=q.device)
                  * 2.0 - 1.0) * ankle_dither_rad
        q[:, ids] = q[:, ids] + dither
    return q


def joint_vel_rel_dropout(
    env,
    zero_prob: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절속도 관측의 실물 절단 재현 — joint_vel_rel 대체.

    실측 근거(0803): 배포 qvel은 홀드된 관측의 유한차분이라 정지 시 틱의
    48~71%가 정확히 0 (감쇠 신호 누더기). iid 마스킹은 홀드 런의 시간 상관을
    못 담는 1차 근사지만, 정책이 속도 관측 의존(D 게인)을 낮추게 하는 데는
    유효. 실물 이식에서 감쇠는 관측 아닌 액션 형성으로 확보해야 함을 학습.
    """
    v = mdp.joint_vel_rel(env, asset_cfg)
    mask = (torch.rand_like(v) >= zero_prob).to(v.dtype)
    return v * mask


def feet_swing_clearance(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_m: float = 0.045,
    sole_offset_m: float = 0.0525,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """스윙 발 들기 높이 보상 (0808 실물 관찰 '바닥을 긁는다' 반영). 값 0~1.

    실물 관찰: march 보행에서 발이 바닥을 긁는 느낌이 강하다.
    코드를 보니 원인이 명확했다 — **발 높이를 보상하는 항이 어디에도 없었다.**
    보행은 feet_air_time(공중 시간)과 게이트 클록으로만 만들어지므로, 정책에게는
    '떠 있기'만 충족하고 최소한으로 들어 스치듯 지나가는 것이 보상상 최적이다.

    실측(march_v5b, 32env×15s, cmd 0.3): 스윙 사이클 1221회의 최고점 중앙값
    23.3mm, **p10 7.5mm, 최소 1.1mm, 20mm 미만이 41.6%**. 실물 바닥의 미세
    요철이나 발이 완전히 수평이 아닌 순간에 닿는다.

    처방: 스윙 중(비접촉)인 발의 발바닥 높이를 target_m 까지 선형 보상하고
    거기서 포화시킨다(과도한 들기는 보상하지 않는다 — 에너지·안정성 손해).
    sole_offset_m = 발목 롤축→발바닥 52.5mm (STL 실측 0808).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.norm(forces, dim=-1) > force_threshold        # (N, 발)
    asset = env.scene[asset_cfg.name]
    h = (asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
         - env.scene.env_origins[:, 2].unsqueeze(1) - sole_offset_m)
    r = (h / target_m).clamp(0.0, 1.0)
    r = torch.where(in_contact, torch.zeros_like(r), r)
    return r.mean(dim=1)


def joint_vel_over_limit(
    env,
    asset_cfg: SceneEntityCfg,
    limit_rad_s: float = 5.0,
    cap: float = 100.0,
) -> torch.Tensor:
    """지정 관절의 속도 한계 초과분 벌점 [rad/s] (0809 발목롤 위반 대응).

    AK45-36(발목 롤, 모터 6·12)은 감속비 1/8로 최대 5.0rad/s 다. 발 들기를
    키우자 스윙 중 발목롤이 빨라져 5.26~5.68rad/s 로 한계를 넘었다 —
    실물에서는 추종 실패이자 과열·손상 경로다. joint_vel_l2(전 관절 균등,
    가중 −5e-4)로는 특정 관절의 한계 초과를 못 잡는다: 초과분만 선택적으로,
    한계 이내는 무벌점으로 때린다.
    cap 은 방폭 관례(폭발 시 무한대 보상 방지).
    """
    asset = env.scene[asset_cfg.name]
    v = asset.data.joint_vel[:, asset_cfg.joint_ids].abs()
    return (v - limit_rad_s).clamp(min=0.0).sum(dim=1).clamp_max(cap)


# ---------------------------------------------------------------------------
# 캘리브레이션 상수 편향 랜덤화 (0809 — 실기 보류 사유를 학습으로 해소)
# ---------------------------------------------------------------------------
#: 실측: stand_v8 은 영점 오차 3°에서 낙상 43.8%(28/64), 최대 기울임 32.1°.
#: 원인은 관측 노이즈가 전부 Unoise(매 스텝 제로평균)뿐이고 **에피소드 내내
#: 유지되는 상수 편향**을 학습한 적이 없기 때문. 제로평균 노이즈는 걸러내는
#: 법을 가르치지 치우친 기준에 적응하는 법을 가르치지 않는다.
#: 실물 영점은 매달림 육안 정렬로 잡았고 기구학 검증 이력이 없다 —
#: boot_pose_check 허용오차가 AK70에서 8°라 3° 오차는 게이트를 통과한다.


def resample_calib_bias(
    env,
    env_ids,
    joint_bias_rad: tuple = (-0.052, 0.052),   # ±3°
    imu_tilt_rad: tuple = (-0.052, 0.052),     # ±3°
):
    """에피소드마다 관측 상수 편향을 새로 뽑는다 (mode="reset" 이벤트).

    관절 편향은 관절마다 독립 — 실물 영점 오차도 관절마다 다르다.
    IMU 편향은 몸통 하나에 대한 장착 기울기이므로 env당 스칼라.
    """
    n_j = env.scene["robot"].data.joint_pos.shape[1]
    if not hasattr(env, "_calib_joint_bias"):
        env._calib_joint_bias = torch.zeros(env.num_envs, n_j, device=env.device)
        env._calib_imu_tilt = torch.zeros(env.num_envs, device=env.device)
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    env._calib_joint_bias[ids] = torch.empty(
        len(ids), n_j, device=env.device).uniform_(*joint_bias_rad)
    env._calib_imu_tilt[ids] = torch.empty(
        len(ids), device=env.device).uniform_(*imu_tilt_rad)


def joint_pos_rel_biased(
    env,
    quant_step_rad: float = 0.012,
    ankle_dither_rad: float = 0.0175,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """관절각 관측 + 에피소드 상수 편향 (영점 오차 모사).

    Stand 계열은 joint_pos_rel_staircase(실물 계단·디더 재현)를 쓰므로 그것을
    **감싼다** — 파라미터 계약을 유지하고 관측 현실화를 잃지 않기 위해서다.
    """
    q = joint_pos_rel_staircase(env, quant_step_rad, ankle_dither_rad, asset_cfg)
    b = getattr(env, "_calib_joint_bias", None)
    return q if b is None else q + b[:, : q.shape[1]]


def projected_gravity_biased(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """중력벡터 관측 + 에피소드 상수 피치 회전 (IMU 장착 오차 모사)."""
    g = mdp.projected_gravity(env, asset_cfg)
    th = getattr(env, "_calib_imu_tilt", None)
    if th is None:
        return g
    c, s = torch.cos(th), torch.sin(th)
    out = g.clone()
    gx, gz = g[:, 0], g[:, 2]
    out[:, 0] = gx * c - gz * s
    out[:, 2] = gx * s + gz * c
    return out
