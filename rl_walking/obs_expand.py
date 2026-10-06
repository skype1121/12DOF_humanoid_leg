"""관측 확장 웜스타트(--init_expand) — 기존 체크포인트의 첫 층 입력 열을 항 이름 기준으로 새 관측 레이아웃에 옮긴다.

배경(1006, W3-B): 새 후보는 정책 관측에 항(예: 착지 목표 4차원/프레임)을 추가해 입력 차원이 커진다. 그래도 서기 정책
(G21a-11300, actor 320 / critic 335) 에서 출발해야 하므로, 체크포인트 run 의 params/env.yaml(관측 그룹 정의)과
새 env 의 ObservationManager(활성 항·차원)를 이름으로 대조해 첫 층 가중을 만든다.

레이아웃 규약(Isaac Lab ObservationManager, 코드 확인):
  - 그룹 안 = 항 정의 순서대로 연결(concatenate_terms=True).
  - 항별 이력: CircularBuffer(N, H, D).reshape(N, -1) → **항 단위**로 H 프레임이 연속(오래된→최신), 그 다음 항.
    (프레임별 64차원 묶음이 아님. G21a 정책 320 = Σ항 H·D, last_action 블록 = 165:225.)
  - 그룹의 history_length 가 None 이 아니면 모든 항의 이력을 덮어쓴다.
  - 네트워크 입력 = rsl_rl obs_groups[set] 의 그룹을 순서대로 연결(critic = [policy | critic]).

규칙: 기존 항 열 = 그대로 복사, 새 항 열 = 0(→ 새 항 값과 무관하게 원 정책과 같은 출력), 항 순서가 바뀌면 이름으로 재배치.
기존에 있었는데 새 env 에 없는 항, 같은 이름인데 차원이 바뀐 항(이력 길이 변경 포함)은 오류(조용한 손실 금지).
나머지 층·log_std 는 그대로. 관측 정규화기가 켜져 있으면 새 열은 mean 0·var 1·std 1, 기존 열은 복사.
optimizer 상태는 새로 시작(호출 쪽에서 optimizer 를 load 하지 않음).

이 모듈은 torch·yaml 만 쓴다(Isaac 불필요) — `python obs_expand.py --selftest [--ckpt ...]` 로 CPU 단위 시험.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import yaml

FIRST_LAYER = {"actor": "actor.0.weight", "critic": "critic.0.weight"}
NORMALIZER = {"actor": "actor_obs_normalizer", "critic": "critic_obs_normalizer"}
NET_SET = {"actor": "policy", "critic": "critic"}     # rsl_rl obs_groups 키


@dataclass
class Term:
    key: str        # "그룹/항"
    hist: int       # 이력 프레임 수(0/None → 1)
    frame_dim: int | None   # 프레임당 차원(구 레이아웃은 새 env 에서 채움)

    @property
    def dim(self) -> int:
        return self.hist * self.frame_dim


# ---------------------------------------------------------------------------------------------------------
# env.yaml(구 체크포인트 run) 파싱
# ---------------------------------------------------------------------------------------------------------
class _AnyTagLoader(yaml.SafeLoader):
    pass


def _construct_any(loader, tag_suffix, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_AnyTagLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_any)
_AnyTagLoader.add_multi_constructor("!", _construct_any)


def _hist(h) -> int:
    return int(h) if h else 1


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=_AnyTagLoader)


def yaml_group_terms(env_yaml: dict, group: str) -> list[tuple[str, int]]:
    """env.yaml 의 관측 그룹 → [(항 이름, 이력)] (정의 순서, None 항 제외)."""
    obs = env_yaml["observations"]
    if group not in obs or obs[group] is None:
        raise KeyError(f"구 env.yaml 에 관측 그룹 '{group}' 없음")
    g = obs[group]
    if not g.get("concatenate_terms", True):
        raise ValueError(f"그룹 '{group}' concatenate_terms=False — 지원 안 함")
    g_hist = g.get("history_length")
    out = []
    for name, cfg in g.items():
        if not isinstance(cfg, dict) or "func" not in cfg:
            continue        # 그룹 설정 키(concatenate_terms 등) 또는 비활성(None) 항
        h = g_hist if g_hist is not None else cfg.get("history_length")
        out.append((name, _hist(h)))
    return out


# ---------------------------------------------------------------------------------------------------------
# 레이아웃 대조
# ---------------------------------------------------------------------------------------------------------
def net_layout(group_terms: dict[str, list[Term]], obs_groups: dict[str, list[str]], net: str) -> list[Term]:
    return [t for g in obs_groups[NET_SET[net]] for t in group_terms[g]]


def fill_old_frame_dims(old: list[Term], new: list[Term], old_total: int, net: str) -> None:
    """구 항의 프레임 차원을 새 env 의 같은 이름 항에서 가져오고, 합을 체크포인트 열 수와 맞춘다."""
    new_by = {t.key: t for t in new}
    unknown = []
    for t in old:
        if t.key in new_by:
            t.frame_dim = new_by[t.key].frame_dim
        else:
            unknown.append(t)
    known = sum(t.dim for t in old if t.frame_dim is not None)
    if len(unknown) == 1 and (old_total - known) % unknown[0].hist == 0 and old_total > known:
        unknown[0].frame_dim = (old_total - known) // unknown[0].hist     # 진단 표용(이 항은 아래에서 오류)
    elif unknown:
        for t in unknown:
            t.frame_dim = 0
    s = sum(t.dim for t in old)
    if s != old_total:
        raise ValueError(f"[{net}] 구 레이아웃 합 {s} != 체크포인트 입력 {old_total} — 같은 이름 항의 차원/이력이 바뀌었거나 "
                         f"env.yaml 이 체크포인트와 맞지 않음. 구 항: {[(t.key, t.hist, t.frame_dim) for t in old]}")


def build_index_map(old: list[Term], new: list[Term], net: str):
    """새 입력 열 → 구 입력 열(-1 = 새 열). 반환 (src_idx LongTensor[D_new], 표 행 목록)."""
    new_keys = {t.key for t in new}
    removed = [t.key for t in old if t.key not in new_keys]
    if removed:
        raise ValueError(f"[{net}] 새 env 에 없는 구 항 {removed} — 열을 버리면 원 정책과 동치가 깨짐(지원 안 함)")
    old_off, o = {}, 0
    for t in old:
        old_off[t.key] = (o, t)
        o += t.dim
    src, rows, n = [], [], 0
    for t in new:
        if t.key in old_off:
            oo, ot = old_off[t.key]
            if (ot.hist, ot.frame_dim) != (t.hist, t.frame_dim):
                raise ValueError(f"[{net}] 항 '{t.key}' 형상 변경 {ot.hist}x{ot.frame_dim} → {t.hist}x{t.frame_dim}")
            src += list(range(oo, oo + t.dim))
            rows.append((net, t.key, n, n + t.dim, t.hist, t.frame_dim, f"{oo}:{oo + t.dim}"))
        else:
            src += [-1] * t.dim
            rows.append((net, t.key, n, n + t.dim, t.hist, t.frame_dim, "NEW(0)"))
        n += t.dim
    return torch.tensor(src, dtype=torch.long), rows


def format_table(rows) -> str:
    lines = [f"{'net':6s} {'항(그룹/이름)':32s} {'새 열':>11s} {'H x D':>7s}  구 열"]
    for net, key, a, b, h, d, s in rows:
        lines.append(f"{net:6s} {key:32s} {f'{a}:{b}':>11s} {f'{h}x{d}':>7s}  {s}")
    return "\n".join(lines)


def expand_state_dict(sd: dict, maps: dict[str, torch.Tensor]) -> dict:
    """첫 층 가중(·정규화기)을 새 입력 순서로 재배치. 새 열 가중 = 0."""
    out = dict(sd)
    for net, src in maps.items():
        w = sd[FIRST_LAYER[net]]
        if int(src.max()) >= w.shape[1]:
            raise ValueError(f"[{net}] 매핑 범위 초과 {int(src.max())} >= {w.shape[1]}")
        keep = src >= 0
        nw = torch.zeros(w.shape[0], src.numel(), dtype=w.dtype)
        nw[:, keep] = w[:, src[keep]]
        out[FIRST_LAYER[net]] = nw
        pre = NORMALIZER[net]
        for buf, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            k = f"{pre}.{buf}"
            if k in sd:
                b = sd[k]
                nb = torch.full((b.shape[0], src.numel()), fill, dtype=b.dtype)
                nb[:, keep] = b[:, src[keep]]
                out[k] = nb
    return out


def plan(old_group_terms: dict[str, list[tuple[str, int]]], old_obs_groups: dict[str, list[str]],
         new_group_terms: dict[str, list[Term]], new_obs_groups: dict[str, list[str]], sd: dict):
    """구/새 레이아웃 → (maps{net: src_idx}, rows). old_group_terms 는 yaml 파싱 결과(프레임 차원 미정)."""
    maps, rows = {}, []
    for net in ("actor", "critic"):
        old_gt = {g: [Term(f"{g}/{n}", h, None) for n, h in old_group_terms[g]] for g in old_obs_groups[NET_SET[net]]}
        old = net_layout(old_gt, old_obs_groups, net)
        new = net_layout(new_group_terms, new_obs_groups, net)
        fill_old_frame_dims(old, new, sd[FIRST_LAYER[net]].shape[1], net)
        m, r = build_index_map(old, new, net)
        maps[net], rows = m, rows + r
    return maps, rows


# ---------------------------------------------------------------------------------------------------------
# train.py 연결
# ---------------------------------------------------------------------------------------------------------
def resolve_ckpt(spec: str, log_root: str) -> str:
    if os.path.isfile(spec):
        return os.path.abspath(spec)
    run, ck = spec.rsplit(":", 1)
    if os.path.isdir(run):
        p = os.path.join(run, ck)
    else:
        from isaaclab_tasks.utils import get_checkpoint_path
        p = get_checkpoint_path(log_root, run, ck)
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return os.path.abspath(p)


def new_group_terms_from_env(env) -> dict[str, list[Term]]:
    om = env.unwrapped.observation_manager
    out = {}
    for g, names in om.active_terms.items():
        cfgs = om._group_obs_term_cfgs[g]
        dims = om.group_obs_term_dim[g]
        terms = []
        for n, c, d in zip(names, cfgs, dims):
            if len(d) != 1:
                raise ValueError(f"항 '{g}/{n}' 평탄화 안 된 형상 {d} — 지원 안 함")
            h = _hist(c.history_length)
            if d[0] % h:
                raise ValueError(f"항 '{g}/{n}' 차원 {d[0]} 이 이력 {h} 로 나누어떨어지지 않음")
            terms.append(Term(f"{g}/{n}", h, d[0] // h))
        out[g] = terms
    return out


def init_expand_runner(runner, env, spec: str, log_root: str, log_dir: str) -> dict:
    """runner.alg.policy 에 확장 가중을 로드. optimizer 는 손대지 않음(=새로 시작). 반환 = 정보 dict."""
    ckpt = resolve_ckpt(spec, log_root)
    run_dir = os.path.dirname(ckpt)
    old_env = load_yaml(os.path.join(run_dir, "params", "env.yaml"))
    old_agent = load_yaml(os.path.join(run_dir, "params", "agent.yaml"))
    old_obs_groups = old_agent.get("obs_groups") or {"policy": ["policy"], "critic": ["policy", "critic"]}
    new_obs_groups = runner.cfg["obs_groups"]
    need = sorted({g for s in old_obs_groups.values() for g in s})
    old_group_terms = {g: yaml_group_terms(old_env, g) for g in need}
    new_gt = new_group_terms_from_env(env)

    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd_old = d["model_state_dict"]
    maps, rows = plan(old_group_terms, old_obs_groups, new_gt, new_obs_groups, sd_old)
    table = format_table(rows)
    policy = runner.alg.policy
    for net in ("actor", "critic"):
        want = getattr(policy, net)[0].weight.shape[1]
        if want != maps[net].numel():
            raise ValueError(f"[{net}] 새 네트워크 입력 {want} != 매핑 {maps[net].numel()}")
    sd_new = expand_state_dict(sd_old, maps)
    policy.load_state_dict({k: v.to(next(policy.parameters()).device) for k, v in sd_new.items()}, strict=True)
    runner.current_learning_iteration = int(d.get("iter", 0))

    # 실제 네트워크로 동치 확인: 새 열 0 인 무작위 입력 → 확장 정책 vs 구 첫 층으로 바꾼 사본
    import copy
    errs = {}
    for net in ("actor", "critic"):
        seq = getattr(policy, net)
        old_seq = copy.deepcopy(seq).cpu()
        w = sd_old[FIRST_LAYER[net]]
        old_seq[0] = torch.nn.Linear(w.shape[1], w.shape[0])
        old_seq[0].weight.data.copy_(w)
        old_seq[0].bias.data.copy_(sd_old[FIRST_LAYER[net].replace("weight", "bias")])
        new_seq = copy.deepcopy(seq).cpu()
        old_seq, new_seq = old_seq.double(), new_seq.double()       # 합 순서 차이(열 재배치)에 의한 float32 오차 배제
        x = torch.randn(64, maps[net].numel(), dtype=torch.float64)
        x[:, maps[net] < 0] = 0.0
        x_old = torch.zeros(64, w.shape[1], dtype=torch.float64)
        x_old[:, maps[net][maps[net] >= 0]] = x[:, maps[net] >= 0]
        with torch.no_grad():
            errs[net] = float((old_seq(x_old) - new_seq(x)).abs().max())
    norm_on = any(k.endswith("_obs_normalizer._mean") for k in sd_new)
    info = {"ckpt": ckpt, "iter": runner.current_learning_iteration, "errs": errs, "table": table,
            "dims": {n: (sd_old[FIRST_LAYER[n]].shape[1], maps[n].numel()) for n in maps}, "normalizer": norm_on}
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "init_expand_map.txt"), "w") as f:
        f.write(f"src={ckpt}\niter={info['iter']}\ndims={info['dims']}\nequiv_max_abs={errs}\n\n{table}\n")
    print(f"[init_expand] {ckpt} (iter {info['iter']}) dims {info['dims']} optimizer=새로 시작", flush=True)
    print(table, flush=True)
    print(f"[init_expand] 새 열 0 동치 확인(실 네트워크 float64, 정규화기 제외) max|Δ| = {errs}", flush=True)
    if max(errs.values()) > 1e-9:
        raise RuntimeError(f"[init_expand] 동치 확인 실패 {errs}")
    return info


# ---------------------------------------------------------------------------------------------------------
# 단위 시험 (CPU)
# ---------------------------------------------------------------------------------------------------------
_G21A_FRAME_DIMS = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3, "joint_pos": 12, "joint_vel": 12,
                    "actions": 12, "gait_clock": 2, "feet_contact": 2, "gyro_ema": 3, "joint_vel_ema": 12,
                    "base_lin_vel": 3}


def _make_ac(sd: dict, obs_dims: dict[str, int]):
    """rsl_rl ActorCritic(가능하면) 를 sd 로 만든다. 없으면 같은 구조의 Sequential 로 대체."""
    try:
        from tensordict import TensorDict
        from rsl_rl.modules import ActorCritic
        obs = TensorDict({g: torch.zeros(1, d) for g, d in obs_dims.items()}, batch_size=[1])
        ac = ActorCritic(obs, {"policy": ["policy"], "critic": ["policy", "critic"]}, 12,
                         actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                         activation="elu", init_noise_std=0.25, noise_std_type="log")
        ac.load_state_dict(sd, strict=True)
        ac.eval()
        return (lambda o: ac.act_inference(TensorDict(o, batch_size=[next(iter(o.values())).shape[0]])),
                lambda o: ac.evaluate(TensorDict(o, batch_size=[next(iter(o.values())).shape[0]])), "rsl_rl ActorCritic")
    except ImportError:
        def mlp(prefix, din, dout):
            dims = [din, 512, 256, 128, dout]
            layers = []
            for i in range(4):
                layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
                if i < 3:
                    layers.append(torch.nn.ELU())
            s = torch.nn.Sequential(*layers)
            s.load_state_dict({k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + ".")})
            return s
        a = mlp("actor", sd["actor.0.weight"].shape[1], 12)
        c = mlp("critic", sd["critic.0.weight"].shape[1], 1)
        return (lambda o: a(o["policy"]), lambda o: c(torch.cat([o["policy"], o["critic"]], -1)), "Sequential 대체")


def _assemble(group_terms: dict[str, list[Term]], vals: dict[str, torch.Tensor], B: int) -> dict[str, torch.Tensor]:
    """항 이름 → 값으로 그룹 텐서 조립(시험용 독립 경로: 매핑표를 쓰지 않음)."""
    return {g: torch.cat([vals[t.key] if t.key in vals else torch.zeros(B, t.dim) for t in ts], -1)
            for g, ts in group_terms.items()}


def selftest(ckpt: str | None, env_yaml: str | None) -> None:
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)      # 열 재배치로 합 순서가 바뀌므로 float32 반올림 차를 배제하고 비교
    if ckpt and os.path.isfile(ckpt):
        sd_old = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        old_env = load_yaml(env_yaml or os.path.join(os.path.dirname(ckpt), "params", "env.yaml"))
        sd_old = {k: v.double() for k, v in sd_old.items()}
        src = f"실 체크포인트 {ckpt}"
    else:   # 합성 G21a 형상
        sd_old = {}
        for net, din, dout in (("actor", 320, 12), ("critic", 335, 1)):
            dims = [din, 512, 256, 128, dout]
            for i, li in enumerate((0, 2, 4, 6)):
                sd_old[f"{net}.{li}.weight"] = torch.randn(dims[i + 1], dims[i]) * 0.1
                sd_old[f"{net}.{li}.bias"] = torch.randn(dims[i + 1]) * 0.1
        sd_old["log_std"] = torch.zeros(12)
        pol = [(n, 5) for n in ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel",
                                "actions", "gait_clock", "feet_contact", "gyro_ema", "joint_vel_ema")]
        old_env = {"observations": {"policy": {"concatenate_terms": True, "history_length": None,
                                               **{n: {"func": "x", "history_length": h} for n, h in pol}},
                                    "critic": {"concatenate_terms": True, "history_length": None,
                                               "base_lin_vel": {"func": "x", "history_length": 5}}}}
        src = "합성 가중(G21a 형상)"
    og = {"policy": ["policy"], "critic": ["policy", "critic"]}
    old_gt_names = {g: yaml_group_terms(old_env, g) for g in ("policy", "critic")}
    print(f"[selftest] 원본 = {src}")
    print(f"[selftest] 구 policy 항 순서(env.yaml) = {[n for n, _ in old_gt_names['policy']]}")

    def T(g, n, h=5, d=None):
        return Term(f"{g}/{n}", h, d if d is not None else _G21A_FRAME_DIMS[n])

    old_terms = {g: [T(g, n, h) for n, h in old_gt_names[g]] for g in og["critic"]}
    scenarios = {
        # A: 정책 그룹 끝에 착지 목표 4차원×5 프레임 추가 (W3-B 기본 형태)
        "A 끝에 추가(step_target 4x5)": {
            "policy": old_terms["policy"] + [Term("policy/step_target", 5, 4)],
            "critic": old_terms["critic"]},
        # B: 중간 삽입 + 항 순서 교환 + critic privileged 새 항
        "B 중간 삽입·순서 교환·critic 새 항": {
            "policy": (old_terms["policy"][:3] + [Term("policy/step_target", 1, 4)] + [old_terms["policy"][4],
                       old_terms["policy"][3]] + old_terms["policy"][5:]),
            "critic": [Term("critic/foot_target_err", 1, 6)] + old_terms["critic"]},
    }
    act_o, crit_o, kind = _make_ac(sd_old, {"policy": sd_old["actor.0.weight"].shape[1],
                                            "critic": sd_old["critic.0.weight"].shape[1] - sd_old["actor.0.weight"].shape[1]})
    print(f"[selftest] 네트워크 = {kind}")
    B = 256
    for name, new_gt in scenarios.items():
        maps, rows = plan(old_gt_names, og, new_gt, og, sd_old)
        sd_new = expand_state_dict(sd_old, maps)
        dims = {g: sum(t.dim for t in ts) for g, ts in new_gt.items()}
        act_n, crit_n, _ = _make_ac(sd_new, dims)
        # 항별 값(이름으로) — 구/새 레이아웃을 각각 독립 조립
        vals = {t.key: torch.randn(B, t.dim) for ts in old_terms.values() for t in ts}
        with torch.no_grad():
            o_old = _assemble(old_terms, vals, B)
            o_new0 = _assemble(new_gt, vals, B)                       # 새 항 = 0
            ea = float((act_o(o_old) - act_n(o_new0)).abs().max())
            ec = float((crit_o(o_old) - crit_n(o_new0)).abs().max())
            vals_nz = dict(vals)
            for ts in new_gt.values():
                for t in ts:
                    if t.key not in vals_nz:
                        vals_nz[t.key] = torch.randn(B, t.dim) * 3
            o_newnz = _assemble(new_gt, vals_nz, B)
            ea_nz = float((act_o(o_old) - act_n(o_newnz)).abs().max())  # 새 열 가중 0 → 이것도 0
        # 새 열에 기울기가 흐르는지(학습 가능) — 첫 층 가중 grad 의 새 열 부분이 0 이 아님
        w = sd_new["actor.0.weight"].clone().requires_grad_(True)
        x = o_newnz["policy"]
        (torch.nn.functional.elu(x @ w.T).sum()).backward()
        g_new = float(w.grad[:, maps["actor"] < 0].abs().sum())
        print(f"\n=== 시나리오 {name}: actor {sd_old['actor.0.weight'].shape[1]}→{maps['actor'].numel()}, "
              f"critic {sd_old['critic.0.weight'].shape[1]}→{maps['critic'].numel()}")
        print(format_table(rows))
        print(f"  새 항=0  : actor max|Δ| = {ea:.3e}, critic max|Δ| = {ec:.3e}")
        print(f"  새 항≠0  : actor max|Δ| = {ea_nz:.3e} (새 열 가중 0 이므로 0 이어야 함)")
        print(f"  새 열 grad 합 = {g_new:.3e} (>0 이어야 학습 가능)")
        assert ea < 1e-9 and ec < 1e-9 and ea_nz < 1e-9 and g_new > 0, name
    # 정규화기 확장
    sd_n = dict(sd_old)
    D = sd_old["actor.0.weight"].shape[1]
    sd_n["actor_obs_normalizer._mean"] = torch.randn(1, D)
    sd_n["actor_obs_normalizer._var"] = torch.rand(1, D) + 0.5
    sd_n["actor_obs_normalizer._std"] = sd_n["actor_obs_normalizer._var"].sqrt()
    maps, _ = plan(old_gt_names, og, scenarios["B 중간 삽입·순서 교환·critic 새 항"], og, sd_old)
    e = expand_state_dict(sd_n, maps)
    k = maps["actor"] >= 0
    assert torch.equal(e["actor_obs_normalizer._mean"][:, k], sd_n["actor_obs_normalizer._mean"][:, maps["actor"][k]])
    assert torch.all(e["actor_obs_normalizer._mean"][:, ~k] == 0) and torch.all(e["actor_obs_normalizer._std"][:, ~k] == 1)
    # 오류 경로: 구 항 삭제·형상 변경
    for bad, why in ((
            {"policy": old_terms["policy"][1:], "critic": old_terms["critic"]}, "항 삭제"), (
            {"policy": [Term("policy/base_ang_vel", 3, 3)] + old_terms["policy"][1:], "critic": old_terms["critic"]},
            "이력 변경")):
        try:
            plan(old_gt_names, og, bad, og, sd_old)
            raise AssertionError(f"{why} 가 통과됨")
        except ValueError as ex:
            print(f"  [오류 경로 OK] {why}: {str(ex)[:90]}")
    print("\n[selftest] obs_expand: 끝 추가·중간 삽입·순서 교환·critic 새 항·정규화기·오류 경로 — PASS")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", default=None, help="실 체크포인트(.pt). run/params/env.yaml 을 함께 읽음")
    ap.add_argument("--env_yaml", default=None)
    a = ap.parse_args()
    selftest(a.ckpt, a.env_yaml)
