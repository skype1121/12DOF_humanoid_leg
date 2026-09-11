# 배포 경로 정합성 & 롤백 점검 (item 7/8)

- 작성 2026-09-11 22:55 KST · 저장소 `/home/ryu/humanoid_leg_test1` · 브랜치 `hang2stand-prep-0911` · HEAD `258229a`
- 실행한 것: 로컬 파일 읽기·md5·tar 목록(추출 없음)·CPU 전용 selftest 2종·CPU torch 가중치 대조
- **실행하지 않은 것: ssh·젯슨 접촉·Isaac/GPU·기존 파일 수정.** 젯슨 상태는 전부 UNKNOWN(로컬 산출물에서 추론만)
- 태그 규칙: **MEASURED**(원자료에서 직접) / **COMPUTED**(계산식 명시) / **HYPOTHESIS**(반증 안 된 추정)
- 압력패드 유래 수치(`foot_n`·`pct`·`raw_sum`·CoP)는 전부 **비교정 진단값(uncalibrated diagnostic)** — 하중으로 읽지 않는다. 접촉 비트만 판정용.
- ⚠ 감사 도중 저장소가 움직였다: 세션 시작 스냅샷은 HEAD `1aaedb1` + 더티 트리(`deploy_package.sh`·`operator_console.py` 수정, 콘솔 3파일 untracked)였고, 22:51 에 `8489315`·`258229a` 가 커밋되며 전부 트래킹됐다. 아래는 **`258229a` 기준**.

---

## 1. `models/stand_hang_v5/` — 전 파일 md5 + meta.json

**MEASURED** (`md5sum models/stand_hang_v5/*`, 5개 파일 전부):

| 파일 | md5 | 크기 | mtime |
|---|---|---|---|
| `agent.yaml` | `9586eabc80506496e0a8163771e95768` | 1206 | 2026-09-09 19:15:49 |
| `env.yaml` | `2e8aa287cb79844b124ecc3b52f3e386` | 48688 | 2026-09-09 19:15:49 |
| `meta.json` | `9c85034564d51ef66f07937e242f7236` | 1079 | 2026-09-09 20:54:54 |
| `policy.onnx` | `fd6ba6a2da8f9b4331e6a66d03357668` | 1321751 | 2026-09-09 20:21:57 |
| `policy.pt` | `2a20b7b1cd4ec3da4428edb9a859ce0b` | 1332626 | 2026-09-09 20:21:57 |

`meta.json` 핵심 필드 (**MEASURED**):

```
name            stand_hang_v5
run             2026-09-09_19-15-42
checkpoint      model_139987.pt
checkpoint_md5  c38ff6a14d75055f78a0ba8a60be3ec8
task            Biped12-HangStand-S8ac4-v0
usd             rl_walking/assets/usd_final3/biped12.usd
warm_start      S8a(raw+EMA 320차원, v4 확장 웜스타트 138488) ← stand_hang_v4(S6 136989)
legacy          false
source_dir      logs/rsl_rl/biped12_stage4/2026-09-09_19-15-42
files.policy.pt   2a20b7b1cd4ec3da4428edb9a859ce0b   ← 실제 파일과 일치
files.policy.onnx fd6ba6a2da8f9b4331e6a66d03357668   ← 실제 파일과 일치
files.env.yaml    2e8aa287cb79844b124ecc3b52f3e386   ← 실제 파일과 일치
files.agent.yaml  9586eabc80506496e0a8163771e95768   ← 실제 파일과 일치
```

`meta.json` 의 `files` 4개 md5 는 **현재 파일과 전부 일치**(MEASURED). `meta.json` 자신은 해시 대상이 아니다(자기참조 불가) — 무결성 앵커는 `files` 블록.

**보조 확인(MEASURED)** `models/stand_hang_v5p/`(S8ac6, run `2026-09-09_20-50-01`) 도 자기 `meta.json` 과 일치. v5/v5p 는 `agent.yaml` md5 가 동일(같은 PPO 설정), 나머지는 전부 다름 — 혼동 위험 낮음.

---

## 2. 브리지 · 콘솔의 stand_hang_v5 등록

### 2.1 `jetson/scripts/rl_bridge_node.py`

**MEASURED** — `CHECKPOINTS` 딕트(112행~)에 있는 항목은 **경로 하나뿐**:

```python
"stand_hang_v5": os.path.join(
    REPO, "logs/rsl_rl/biped12_stage4/2026-09-09_19-15-42/exported"),
```

부속 등록:
- `CKPT_GAIT_FREQ["stand_hang_v5"] = 1.1` (미등록 시 기본 1.4 — 위상 어긋남 = 보행 파괴)
- `STAND_DEFAULT = "stand_v18b"` — **v5 가 아니다**
- 기대 파일: `_selftest_policy` 가 `policy.pt` **또는** `policy.onnx` 중 하나만 있으면 통과(둘 다 없으면 SKIP, `required` 면 FAIL)
- 학습 계약: `_selftest_contract` 가 `<exported 의 부모>/params/env.yaml` 을 파싱해 DEFAULT_POSE 12관절·`scale` 0.25·`max_delta_per_step`(슬루 4°/틱)·`decimation×dt`=50Hz·`use_default_offset=True` 를 러너 상수와 대조

**브리지에는 md5 가 기록돼 있지 않다** (MEASURED — `CHECKPOINTS`·주석 어디에도 해시 없음). 로드 경로(`_load_backend`, `rl_walking/deploy/policy_runner.py:127`)도 해시를 검증하지 않는다: 디렉토리를 받으면 **`policy.onnx` 를 먼저** 시도하고 실패해야 `policy.pt` 로 폴백한다. 즉 젯슨에 유효한 TorchScript/ONNX 가 들어 있기만 하면 내용이 무엇이든 조용히 실행된다. md5 게이트는 브리지 밖(`deploy/MANIFEST.md5` + `deploy_to_jetson.sh` [2/7]·[3c/7])에만 있다.

**로컬 실측 (MEASURED)** — `python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_hang_v5` (CPU, CUDA 비활성):
```
selftest: ALL PASS
[PASS] 학습 계약: ... stand_v18b(지연 3~6틱), stand_hang_v5(지연 3~8틱)
[PASS] 정책 500틱: ×19 ... (50Hz 예산 20ms, 백엔드 torch)
   stand_hang_v5 0.13ms [슬루 최대 4.00°/틱, 소프트리밋 초과 +28.1°(관절 1, Stage8 클램프 영역)]
```
- 백엔드가 **torch** 인 것은 이 PC 에 `onnxruntime` 이 없기 때문(MEASURED: `importlib.util.find_spec('onnxruntime')` = None). 젯슨에 onnxruntime 이 있으면 같은 코드가 **`policy.onnx` 를 쓴다** → 로컬 selftest 가 검증한 바이트 파일과 실물이 실행할 파일이 다를 수 있다(§5-8).
- "관절 1" 은 0-기반 인덱스(`int(e.argmax())`, JOINT_ORDER 순) = **`right_hip_f`** (COMPUTED). 물리 없는 목 센서 500틱 표류라 코드 주석대로 정보값이며 FAIL 조건이 아니다.

### 2.2 `jetson/scripts/operator_console.py`

**MEASURED**:
```python
CKPT_CHOICES = ["stand_hang_v5", "stand_v18b", "stand_hang_v4", "stand_hang_v3", "stand_hang_v2",
                "stand_hang_v1", "stand_v17", "stand_v15", "stand_v18", "stand_v16",
                "stand_v9", "stand_v8", "stand_v7", "march_v5b"]
CKPT_DEFAULT = CKPT_CHOICES[0]      # = "stand_hang_v5"
```
- 브리지 기동 인자: `rl_bridge_node.py ... --live --checkpoint <ckpt>` (791행), 셀프테스트 버튼도 `--checkpoint <ckpt or CKPT_DEFAULT>` (815행)
- `console selftest: ALL PASS` (MEASURED, 로컬 실행) — `set(CKPT_CHOICES) ⊆ 브리지 CHECKPOINTS` 를 소스 파싱으로 검증

**교차표 (MEASURED)** — Y/- 는 각 원본에 등록돼 있는지:

| gen | 브리지 CHECKPOINTS | 콘솔 드롭다운 | `models/` 레지스트리 | `GENS` 기본값 | 최신 패키지(0910_1526) |
|---|---|---|---|---|---|
| stand_hang_v5 | Y | Y(기본) | Y | – | Y |
| stand_hang_v5p | Y | – | Y | – | – |
| stand_hang_v4 | Y | Y | Y | – | Y |
| stand_hang_v3 / v2 / v1 | Y | Y | Y | – | – |
| stand_v18b | Y | Y | Y | Y | Y |
| stand_v18 / v17 / v16 / v15 | Y | Y | Y | Y | – |
| stand_v14 | Y | – | Y | – | – |
| **stand_v9 / v8 / v7 / march_v5b** | Y | **Y** | **없음** | – | – |
| walk / march / drv2 / march_drv2 / march_v3 / stand_v3 / v5 / v6 | Y | – | 없음 | – | – |

---

## 3. `jetson/deploy_package.sh` — 정확히 무엇을 싣는가

최근 Codex 수정본(현재 `258229a`; 세션 시작 시점엔 미커밋 상태였다). **수정하지 않았다.**

### 3.1 정책 자산 (`GENS` 루프)
`GENS=${GENS:-"stand_v15 stand_v16 stand_v17 stand_v18 stand_v18b"}` — 각 `g` 에 대해 `models/$g/meta.json` 이 있어야 하고(없으면 건너뜀), `run` 을 읽어:

| 원본 | 패키지 내 경로 |
|---|---|
| `models/$g/policy.pt` | `logs/rsl_rl/biped12_stage4/<run>/exported/policy.pt` |
| `models/$g/policy.onnx` (있으면) | `logs/rsl_rl/biped12_stage4/<run>/exported/policy.onnx` |
| `models/$g/env.yaml` (있으면) | `logs/rsl_rl/biped12_stage4/<run>/params/env.yaml` |

- **`models/$g/agent.yaml` 과 `meta.json` 은 싣지 않는다**(MEASURED) → 젯슨에는 세대의 **등록 근거(md5·run·ckpt)가 존재하지 않는다**. 젯슨에서 "이 정책이 무엇인가"를 자력으로 답할 수 없다.
- 패키지의 `exported/policy.pt` 출처는 **레지스트리(`models/`)이지 로컬 `logs/.../exported/` 가 아니다**(MEASURED). 둘이 갈라져 있어도 패키지는 레지스트리를 따른다(§5-5).

### 3.2 등록 누락 검출
```bash
grep -q "\"$g\"" jetson/scripts/rl_bridge_node.py || echo "  ⚠ $g 가 ... CHECKPOINTS 에 없음 — 등록 필요"
```
**경고만 출력하고 계속 진행한다**(MEASURED — `exit` 없음). 반대 방향(브리지·콘솔에는 있는데 패키지에 없는 세대)은 검사하지 않는다.

### 3.3 고정 파일 목록 (18개)
`jetson/scripts/`: `rl_bridge_node.py`, `analyze_fall.py`, `posture_probe.py`, `operator_console.py`, **`console_state.py`, `console_demo.py`, `console_ui.html`**(Codex 추가), `stiffness_log.py`, `foot_force_adapter_node.py`, `release_all.py`, `zero_kinematic_check.py`, `cop_from_pads.py`, `boot_pose_check.py`
`docs/`: `stand_v18_7차_실행카드.md`, `hang_to_stand_실물절차.md`, `A1_stand_hang_v5_실행카드.md`, **`operator_console_renewal.md`**(Codex 추가)
`jetson/calib_data/sim_default_stand_targets_20260727.json` (selftest 수치 앵커)

디렉토리: `rl_walking/deploy/*.py` (7개), `rl_walking/__init__.py`, `robot_runtime/*.py` (7개), `config/joint_limits_12dof.json`

**일부러 제외**: `config/robot_12dof_hardware_map.json` — 젯슨의 맵(실물 영점)이 유일한 원본(주석에 명시).

Codex 의 diff 는 **한 줄 추가**뿐이다(MEASURED, `git show 8489315 -- jetson/deploy_package.sh` 상당):
```
+         jetson/scripts/console_state.py jetson/scripts/console_demo.py jetson/scripts/console_ui.html docs/operator_console_renewal.md \
```
이 추가는 **필수**다: 새 콘솔은 임포트 시점에 `PAGE = Path(__file__).with_name("console_ui.html").read_text(...)` (888행)과 `from console_state import ...` (41/43행)을 실행하므로, 세 파일 없이 `operator_console.py` 만 갱신하면 systemd 서비스가 **임포트 단계에서 죽는다**(COMPUTED — 모듈 최상위 실행).

### 3.4 md5 / selftest 단계
- `deploy_package.sh` 안: `MANIFEST.md5` 생성이 전부. `( cd "$STAGE" && find . -type f ! -path ./deploy/MANIFEST.md5 -print0 | sort -z | xargs -0 md5sum > deploy/MANIFEST.md5 )` → 스테이지 사본을 `deploy/MANIFEST.md5` 로 복사 + tar 안에도 포함. **selftest 는 하나도 돌리지 않는다.**
- selftest 는 전부 `jetson/deploy_to_jetson.sh` 에 있다(MEASURED):
  - `[2/7]` `LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5` (종료코드 판정 — 한국어 로케일 grep 금지 주석 있음)
  - `[3/7]` 브리지 selftest `--checkpoint stand_v18b`
  - `[3b/7]` 브리지 selftest `--checkpoint stand_hang_v5` + `policy_runner_stage4.py --selftest`
  - `[3c/7]` `md5sum logs/.../2026-09-09_19-15-42/exported/policy.pt` 를 `models/stand_hang_v5/meta.json` 의 값과 대조 — **`policy.pt` 만. `policy.onnx` 는 이 단계에서 대조하지 않는다**(MANIFEST 단계에서는 포함됨)
  - `[4/7]` `foot_force_adapter_node.py --selftest`, `release_all.py --selftest`
  - `[5/7]`·`[5b/7]` `~/rl_calib/release_all.py`·`boot_pose_check.py` 를 `.bak_<mmdd_HHMM>` 백업 후 덮어씀
  - `[6/7]` `sensors_up.sh` 가 띄우는 어댑터 사본의 `NEWTON_PER_COUNT`(0.02650) 확인·필요 시 갱신 — ⚠ 이 상수는 **비교정 진단값의 스케일**이지 하중 교정이 아니다
  - `[7/7]` `systemctl restart operator-console`
- 셸 안전성(MEASURED, 별도 실증): `set -euo pipefail` 아래의 `[ -f ... ] && cp ...` 는 AND-리스트 비말단 실패라 `set -e` 가 발동하지 않는다(`bash -c 'set -euo pipefail; [ -f /nonexistent ] && echo x; echo SURVIVED'` → `SURVIVED`, exit 0). 즉 onnx 없는 세대도 정상 통과한다.

---

## 4. 학습 런 ↔ 레지스트리: `policy.pt` 는 변환 익스포트다 (바이트 동일 아님)

**MEASURED**
- `logs/rsl_rl/biped12_stage4/2026-09-09_19-15-42/model_139987.pt` **존재** (8,019,081 B, mtime 2026-09-09 20:13:06), md5 `c38ff6a14d75055f78a0ba8a60be3ec8` = `meta.json.checkpoint_md5` **일치**
- `models/stand_hang_v5/policy.pt` = `2a20b7b1…` (1,332,626 B, mtime 20:21:57) → **바이트 동일 아님** (다른 파일·다른 크기)
- 같은 런의 `exported/policy.pt`·`policy.onnx`·`params/env.yaml`·`params/agent.yaml` md5 는 `models/stand_hang_v5/` 의 동명 파일과 **4/4 완전 일치**

### 왜 다른가 — 코드 근거
1. `rl_walking/scripts/play.py:177-179` — 체크포인트를 `runner.load(resume_path)` 로 올린 뒤
   `export_policy_as_jit(policy_nn, normalizer=..., path=<resume_path 의 디렉토리>/exported, filename="policy.pt")` 와 `export_policy_as_onnx(...)` 를 호출한다. 즉 `exported/policy.pt` 는 **actor 만 TorchScript 로 트레이스한 추론 전용 산출물**이고, `model_*.pt` 는 actor+critic+optimizer 를 담은 학습 체크포인트다.
2. `rl_walking/scripts/register_model.py:37-44` — `exported/policy.pt`, `exported/policy.onnx`, `params/env.yaml`, `params/agent.yaml` 을 `models/<name>/` 으로 **복사**하고, `checkpoint_md5` 에는 `--ckpt` 로 지정된 **원본 체크포인트의 md5** 를 기록한다.

### 동일 가중치임을 실증 (MEASURED, CPU torch)
```
model_139987.pt : iter = 139987, model_state_dict 의 actor.* 8개 텐서
models/stand_hang_v5/policy.pt (torch.jit.load) : state_dict 키 8개 동일
  actor.0.weight shape = torch.Size([512, 320])
  8/8 텐서 torch.equal == True,  MISMATCH: []
models/stand_hang_v5/policy.onnx : actor.0.weight(655,360 B)·actor.6.weight 의
  float32 raw 블록이 체크포인트 텐서 바이트와 **완전 일치**(부분문자열 검색)
```
→ **`policy.pt`/`policy.onnx` 는 `model_139987.pt` 에서 나온 변환 익스포트이며 actor 가중치는 비트 단위로 동일하다.** 관측 차원 **320 확정**(MEASURED: `[512, 320]`).

**COMPUTED 교차검산** — ONNX 크기로도 320 이 나온다. hidden `[512,256,128]`·출력 12(`agent.yaml`)일 때 파라미터 바이트 = `4×(obs×512 + 512 + 512×256+256 + 256×128+128 + 128×12+12)` = `2048×obs + 665,136`. obs=320 → 1,320,496 B (실파일 1,321,751 B, 그래프 메타 1,255 B 잔차). obs=245 → 1,166,896 B (잔차 154,855 B, 불가능).

**정규화기 확인 (MEASURED)** — JIT 모듈은 `forward(x) = actor(normalizer(x))` 이고 자식은 `['actor','normalizer']` 지만 state_dict 키는 actor.* 8개뿐 → `normalizer` 는 파라미터 없는 Identity. `agent.yaml` 의 `actor_obs_normalization: false`·`empirical_normalization: null` 과 정합. **숨은 정규화 없음.**

**provenance 사슬 (MEASURED)** — `agent.yaml`: `load_run: 2026-09-09_17-12-13`, `load_checkpoint: model_138488.pt`, `resume: true` = S8a 웜스타트. `meta.json.warm_start` 와 일치.

### 남은 약한 고리 (HYPOTHESIS 로만 닫힌 부분)
`register_model.py` 는 `exported/` 가 **정말 `--ckpt` 로 지정한 체크포인트에서 나왔는지 검증하지 않는다**(MEASURED — 복사만 함). 이번 v5 는 위 가중치 대조로 **사후 확정**했지만, 절차 자체에는 그 게이트가 없다. `play.py` 는 `--checkpoint` 미지정 시 `get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)` 로 **최신 런/최신 ckpt** 를 고른다고 알려져 있으나, 그 기본값은 저장소 밖(`isaaclab` `RslRlOnPolicyRunnerCfg`)에 있어 **본 감사에서 확인하지 않았다(UNKNOWN — 규칙상 저장소 밖 미열람)**. `rl_walking/agents.py` 의 `Biped12Stage4PPORunnerCfg` 는 `load_run`/`load_checkpoint` 를 설정하지 않는다(MEASURED).

---

## 5. 불일치 · 누락 · UNKNOWN 목록

레지스트리 ↔ 브리지 ↔ 콘솔 ↔ 패키지 ↔ 학습런 ↔ `eval_hang2stand.py` 6자 대조.

### 5-1. **[높음] 패키지 기본 `GENS` 가 hang 세대를 전혀 포함하지 않는다** (MEASURED)
`GENS` 기본값 = `stand_v15 stand_v16 stand_v17 stand_v18 stand_v18b`. `GENS=` 를 빼고 `bash jetson/deploy_package.sh` 를 돌리면 **stand_hang_v1~v5p 가 하나도 안 들어간 새 tar.gz** 가 만들어지고, 그게 `deploy_to_jetson.sh` 의 기본 선택(`ls -t | head -1`)이 된다. 콘솔 기본 정책은 `stand_hang_v5` 이므로 그 패키지로는 데뷔 정책 파일이 젯슨에 없다.
올바른 명령은 `docs/A1_stand_hang_v5_실행카드.md` 1절에만 적혀 있다: `GENS="stand_v18b stand_hang_v4 stand_hang_v5" bash jetson/deploy_package.sh`.
**완화(MEASURED)**: `deploy_to_jetson.sh` `[3b/7]`·`[3c/7]` 가 v5 selftest·md5 로 **중단**시키므로 조용히 잘못 배포되지는 않는다. 다만 `[2/7]` MD5 는 새 MANIFEST 로 통과하므로 방어는 3b/3c 두 단계에만 의존한다.

### 5-2. **[높음] 마지막 패키지(0910_1526)가 HEAD 보다 낡았다 — 인계 워밍업 수정이 안 실려 있다** (MEASURED)
`deploy/MANIFEST.md5`(2026-09-10 15:26, 커밋 `9c934e9`) vs 현재 워킹트리 비교 — 드리프트 5건:

| 패키지 대상 파일 | MANIFEST md5 | 현재 md5 | 마지막 커밋 |
|---|---|---|---|
| `jetson/scripts/rl_bridge_node.py` | `423706aa` | `57ce5fb1` | `2a88452` 09-11 21:31 (stand 직후 `sensor_stale` tick-1 오정지 수정 = **인계 워밍업**) |
| `jetson/scripts/operator_console.py` | `fda3157a` | `b2d3e26c` | `8489315` 09-11 22:51 (Codex 리뉴얼) |
| `jetson/scripts/foot_force_adapter_node.py` | `071a0d57` | `b715d63d` | `1a18b8f` 09-11 15:10 (per-foot npc 스케일) |
| `rl_walking/__init__.py` | `67f67d55` | `38f3f2d0` | `ededc14` 09-11 03:14 (케이블 지지 모델) |
| `docs/A1_stand_hang_v5_실행카드.md` | `8e61b810` | `5a35627c` | `fc431f9` 09-10 20:51 |

+ 패키지 목록에 새로 추가된 4파일(`console_state.py`·`console_demo.py`·`console_ui.html`·`operator_console_renewal.md`)은 MANIFEST 에 아예 없다.
**HYPOTHESIS**: 젯슨이 0910_1526 패키지 상태라면 **tick-1 오정지 수정이 없는 브리지**로 A1 을 돌게 된다. 젯슨 실제 상태는 UNKNOWN.

### 5-3. **[높음] 콘솔 기본 정책이 젯슨에 실제로 복사됐는지 MANIFEST 로 증명되지 않는다** (MEASURED + HYPOTHESIS)
커밋 `8a2d7f0`(09-10 15:33) 이 `CKPT_CHOICES` 첫 항목을 `stand_v18b` → `stand_hang_v5` 로 바꿨다. 메시지는 "젯슨 사본 갱신·**MANIFEST 항목 갱신**·콘솔 재기동". 그러나 **MEASURED: 그 커밋은 `operator_console.py` 한 파일만 바꿨고(`git show --stat`), `deploy/MANIFEST.md5` 는 건드리지 않았다.** MANIFEST 의 콘솔 md5 `fda3157a` 는 **그 직전 커밋 `56cb903` 의 콘솔**(기본값 `stand_v18b`)이고, 0910_1431·0910_1526 두 tar 안의 콘솔도 동일하게 `fda3157a` 다.
→ 젯슨에 기본값 v5 콘솔이 **수동 scp** 로만 올라갔다면, 젯슨에서 `md5sum -c deploy/MANIFEST.md5` 는 `./jetson/scripts/operator_console.py` 줄에서 **FAILED** 가 나고, 같은 패키지의 재배포는 `[2/7]` 에서 중단된다. (젯슨 미확인 → HYPOTHESIS.)

### 5-4. **[높음] 콘솔 드롭다운 4개는 패키지로 만들 수 없는 세대다** (MEASURED)
`stand_v9`, `stand_v8`, `stand_v7`, `march_v5b` 는 브리지 `CHECKPOINTS` 에는 있으나 **`models/<gen>/` 레지스트리가 없다** → `deploy_package.sh` 가 구조적으로 실을 수 없고(meta.json 없으면 건너뜀), MANIFEST 로 md5 검증도 불가능하다. 젯슨에 남아 있다면 과거 배포의 잔존물이며, 선택하면 **어떤 가중치인지 확인할 수단이 없는 정책**이 실물에 올라간다. 콘솔 selftest 는 "브리지에 이름이 있는지"만 보므로 이 경로를 막지 못한다.

### 5-5. **[중간] 로컬 `logs/.../exported/policy.pt` 와 레지스트리가 갈라져 있다 (stand_v18b)** (MEASURED)
`logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported/policy.pt` = `1ddaef70…` (mtime 09-08 20:17) ≠ `models/stand_v18b/policy.pt` = `fac90eb5…` = MANIFEST 값. 같은 디렉토리의 `policy.onnx` 는 **일치**.
가중치 대조(MEASURED, torch): 두 TorchScript 의 state_dict 8/8 텐서 `torch.equal == True` → **기능적으로 동일, 재익스포트로 컨테이너 바이트만 달라진 것**. 영향:
- 패키지는 `models/` 를 쓰므로 **패키지 내용은 영향 없다**.
- 그러나 브리지의 `CHECKPOINTS` 는 `logs/.../exported` 를 가리키므로, **로컬 selftest·로컬 판정은 젯슨이 실행할 바이트와 다른 파일을 검증했다**(이번 경우 가중치 동일이라 무해). md5 하나로는 "재익스포트"와 "다른 정책"을 구분할 수 없으므로 규약이 필요하다.

### 5-6. **[중간] `eval_hang2stand.py` 는 익스포트 산출물을 한 번도 실행하지 않는다** (MEASURED)
`rl_walking/scripts/eval_hang2stand.py:196-202`:
```python
resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(resume_path); policy = runner.get_inference_policy(device=dev)
```
→ 세대 판정은 **원 체크포인트 `model_*.pt`** 를 rsl_rl 러너로 돌린 결과다. 실물은 **`exported/policy.{onnx,pt}`** 를 `PolicyRunnerStage4` 로 돌린다. 두 경로는 관측 조립 구현부터 다르다(env observation manager vs `policy_runner_stage4.py` 의 320 조립+EMA). 공유되는 것은 `rl_walking/deploy/deploy_chain.py`(슬루→LPF→블렌딩)뿐이다.
- 320 조립의 **자기 골든 테스트**는 있다(`policy_runner_stage4.py --selftest`, "S8a 320 조립").
- **env ↔ 러너 교차검증** 스크립트 `rl_walking/deploy/sim_validate_runner_stage4.py` 는 존재하지만 245·`Biped12-Velocity-Stage4-Play-v0`·`--gait-freq 1.4` 기본이며, **320/HangStand 조합으로 돌린 기록을 찾지 못했다**(`reports/`·`docs/` 전수 grep, 언급 0건). → **UNKNOWN**: 320 관측 조립의 배포측 구현이 학습 env 와 1:1 인지 실증된 적 없음.

### 5-7. **[중간] `--checkpoint` 의 의미가 두 도구에서 정반대다** (MEASURED)
- 브리지/콘솔: `--checkpoint stand_hang_v5` = **레지스트리 키**
- `eval_hang2stand.py`/`play.py`: `--checkpoint model_139987.pt` = **체크포인트 파일명**, 런은 `--load_run 2026-09-09_19-15-42`
같은 플래그 이름에 다른 의미 → 카드/로그를 옮겨 쓸 때 오조작 경로. `eval_hang2stand.py` 에서 `--load_run` 을 빠뜨리면 조용히 다른 런이 평가된다(기본값은 저장소 밖 — **UNKNOWN**, §4). `logs/rsl_rl/biped12_stage4/` 에는 82개 런이 있고 최신은 `2026-09-11_03-06-04` 로 v5 가 아니다(MEASURED).

### 5-8. **[중간] 실물이 실제로 실행할 백엔드가 확인되지 않았다** (MEASURED + UNKNOWN)
`_load_backend(..., "auto")` 는 **onnx 우선**. 로컬은 onnxruntime 부재로 torch 로 돌았다. 젯슨의 onnxruntime 유무는 **UNKNOWN** → 젯슨이 `policy.onnx` 를 쓴다면 로컬 selftest 가 검증한 것은 다른 파일이다. `deploy_to_jetson.sh` `[3c/7]` 의 md5 대조 대상도 `policy.pt` 뿐이다(MANIFEST 단계에서는 onnx 도 포함되므로 무결성 자체는 덮인다).

### 5-9. **[중간] 콘솔이 부르지만 패키지가 절대 싣지 않는 스크립트 2종** (MEASURED)
`operator_console.py` 는 `~/humanoid_repo/jetson/scripts/` 에서 다음을 실행한다: `rl_bridge_node.py`, `analyze_fall.py`, `posture_probe.py`, `zero_kinematic_check.py`, **`zero_anchor.py`**, **`stage8_12axis_mit_control_node.py`**.
뒤의 두 개는 `deploy_package.sh` 파일 목록에 **없다** → 배포로 갱신되지도, MANIFEST 로 검증되지도 않는다. 하필 `zero_anchor.py` 는 CLAUDE.md 가 "가장 위험한 쓰기"로 규정한 영점 재앵커 도구이고, `stage8_12axis_mit_control_node.py` 는 CAN 으로 모터를 구동하는 노드다. (구 콘솔도 동일하게 불렀다 — **신규 결함 아님**, 기존 사각지대.)
`~/rl_calib/` 쪽 호출(`can_up.sh`·`sensors_up.sh`·`stage8_cmd.py`·`hang_stand.py`)은 설계상 젯슨 로컬 자산이며 `release_all.py`·`boot_pose_check.py` 만 `deploy_to_jetson.sh` 가 갱신한다.

### 5-10. **[중간] 콘솔 리뉴얼이 selftest 가드 3개를 떨어뜨렸다** (MEASURED — `1aaedb1` vs `258229a` 대조)
구판 `selftest()` 에 있었고 신판에 없는 검사:
- `if CKPT_DEFAULT != "stand_hang_v5": fails.append(...)` — **승윤님 확정 데뷔 정책 고정핀**
- `⑤ 경로 실재 확인` — `~/rl_calib`, `jetson/scripts` 디렉토리 존재(= 젯슨 배치 확인)
- `⑥ 발력 백분율 산식` 회귀 검사
신판이 유지/추가한 것: `set(CKPT_CHOICES) ⊆ 브리지 CHECKPOINTS`, 드롭다운 렌더, TrialState/estop/payload 검사. "기본값 = 첫 항목"은 `CKPT_DEFAULT = CKPT_CHOICES[0]` 로 구조적 보장이 되어 별도 검사가 불필요해졌다.

### 5-11. **[중간] 문서 ↔ 코드: 콘솔 기본값 기술이 어긋난다** (MEASURED)
- `docs/A1_stand_hang_v5_실행카드.md` §4: "정책 드롭다운에 `stand_hang_v5` 가 보이고 **기본값은 `stand_v18b` 그대로**. 드롭다운을 stand_hang_v5 로 바꾼다."
- `jetson/deploy_to_jetson.sh:57`: "드롭다운 기본값 **stand_v18b** 는 재기동 후 반영"
- 실제 코드: `CKPT_DEFAULT = "stand_hang_v5"`
카드대로 "기본값이 v18b 인지" 눈으로 확인하면 **불일치로 보여 오판 유발**. 반대로 카드를 믿고 넘기면 기본값이 이미 v5 라는 사실을 놓친다.

### 5-12. **[낮음~중간] 판정 1순위 지표의 위치가 코드와 카드에서 갈렸다** (MEASURED)
- 구 콘솔: `BODY_WEIGHT_N = 172.9  # 실물 17.62kg(저울 0904) — 발력 총합 %의 분모.`
- 신 콘솔: `BODY_WEIGHT_N = 172.9  # 구 로그 필드 호환용. 화면·진행·성공 판정에 사용하지 않는다.` (`pct` 계산 자체는 510행에 남아 있고 텔레메트리로 노출됨)
- A1 카드 §5 10·11 단계 합격선은 여전히 "콘솔 발력 ≥95%(≥164 N)"
CLAUDE.md 는 "판정 1순위는 발력 총합 %" 라고 못박고 있어, **카드·CLAUDE.md 와 신 콘솔의 설계 의도가 서로 다르다**. 별도로: 본 감사 규칙상 `foot_n`·`pct`·`raw_sum` 은 **비교정 진단값**이며 하중이 아니다 — 어느 쪽을 택하든 "N" 단위 합격선은 교정 근거가 필요하다. 승윤님 결정 사항.

### 5-13. **[해소됨] 배포 대상 파일이 git 밖에 있던 상태** (MEASURED)
세션 시작 시 `console_state.py`·`console_demo.py`·`console_ui.html`·`operator_console_renewal.md` 가 **untracked** 였다. 새 콘솔은 임포트 시 이들을 읽으므로, 그 상태의 패키지는 **git 으로 재현·롤백이 불가능**했다. 22:51 커밋 `8489315` 로 전부 트래킹됨 → **해소**. (기록 목적으로 남긴다: 패키지는 워킹트리에서 tar 를 만들므로 untracked 파일도 배포된다.)

### 5-14. UNKNOWN 목록 (이 감사로는 닫을 수 없음)
1. 젯슨 `~/humanoid_repo` 의 실제 파일 md5·패키지 세대 — ssh 금지
2. 젯슨의 `onnxruntime` 유무 → 실제 추론 백엔드
3. 젯슨에 남아 있는 레거시 정책(`stand_v7/v8/v9/march_v5b`) 파일의 내용·출처
4. `stand_hang_v5` 가 젯슨에 실제로 존재하는지, 콘솔 기본값 v5 판이 올라가 있는지
5. `play.py`/`eval_hang2stand.py` 의 `load_run`/`load_checkpoint` **기본값**(저장소 밖 `isaaclab` 정의 — 규칙상 미열람)
6. 320 관측 조립의 env ↔ 배포러너 교차검증 실행 기록(찾지 못함 ≠ 없음)
7. `~/rl_calib/` 사본들(`hang_stand.py`·`stage8_cmd.py`·`sensors_up.sh`·`can_up.sh`)의 버전

---

## 6. 롤백 절차 (초안)

**전제**: 가진 것은 `deploy_package.sh`(+`deploy_to_jetson.sh`)와 git 뿐. 젯슨에서 수행할 모든 단계는 **Jetson action (not performed)** 으로 표시한다. 본 감사에서는 **한 단계도 실행하지 않았다.**

### 6.0 롤백 전에 반드시 아는 것
- **git 으로 되돌릴 수 없는 것**: `logs/` 는 `.gitignore` 대상이라 **정책 바이너리는 git 에 없다**. 복원 원본은 (a) `models/<gen>/policy.{pt,onnx}` (git 추적됨) 와 (b) `deploy/*.tar.gz`(gitignore, 로컬 디스크에만) 뿐이다.
- **tar 풀기는 덧씌우기다. 지우지 않는다.** 옛 패키지를 풀어도 나중에 추가된 파일(`console_ui.html`, `stand_hang_v5` 정책 등)은 젯슨에 남는다. 구 콘솔은 그 파일들을 임포트하지 않으므로 무해하고, 구 MANIFEST 는 그 파일들을 나열하지 않으므로 `md5sum -c` 도 통과한다.
- **하드웨어맵은 절대 되돌리지 않는다.** `config/robot_12dof_hardware_map.json` 은 패키지에 없다. 젯슨의 맵이 유일 원본. PC 에서 맵을 밀어 넣지 말 것. (맵을 건드렸다면 **stage8 재기동 필수** — 구맵을 물면 인수 거부 후 브레이크 모드 방치.)
- **`~/rl_calib/` 사본은 tar 가 되돌리지 않는다.** `deploy_to_jetson.sh` `[5/7]`·`[5b/7]` 이 `release_all.py`·`boot_pose_check.py` 를 `.bak_<mmdd_HHMM>` 로 백업해 두므로, 롤백 시 그 백업에서 손으로 되돌려야 한다.
- **로컬 보유 패키지 (MEASURED)**: `deploy/stand_7차_20260910_1526.tar.gz`(현행, v18b+hang_v4+hang_v5), `…_20260910_1431.tar.gz`(직전, 동일 3세대 — 차이는 `rl_walking/deploy/policy_runner_stage4.py` 1파일), `…_20260909_1301.tar.gz`(v15~v18b + hang_v1~v4, 9세대), 그 외 `deploy/old/` 에 0905~0907 판 다수.

### 6.1 안전 선행 조건 (실기 중이라면)
1. **Jetson action (not performed)** — 시험 중이면 먼저 회수: **① 줄 재인장(사람) → ② 스택 종료(`bash ~/rl_calib/stack_down.sh`) → ③ 그 다음에만 `python3 ~/rl_calib/release_all.py`.** 순서 엄수. 굽힌 관절 즉시 릴리즈 금지(회생 과전압으로 SMPS 사망 전례).
2. **Jetson action (not performed)** — 콘솔이 띄운 프로세스는 **콘솔로 내린다**(「스택 종료」·「기록 종료」). 콘솔 밖에서 죽여야 하면 **명시 PID 로만**:
   ```
   ps -o pid=,lstart=,cmd= -C python3 | grep -F rl_bridge_node.py     # PID 눈으로 확인
   kill -TERM <PID>        # 10초 후에도 살아 있으면 kill -KILL <PID>
   ```
   **`pkill`·`pgrep` 금지** — 자기매칭 무한대기 사고 이력.

### 6.2 롤백 대상 결정 (PC, 젯슨 무접촉)
3. 되돌릴 지점을 정한다. 기준은 **패키지 파일**이지 커밋이 아니다.
   ```
   ls -lt deploy/stand_7차_*.tar.gz deploy/old/stand_7차_*.tar.gz
   tar tzf <후보>.tar.gz | grep exported/policy.pt        # 어떤 세대가 들었는지
   tar xzOf <후보>.tar.gz ./deploy/MANIFEST.md5 | head -20 # 그 패키지의 기준 md5
   ```
4. **그 패키지의 콘솔/브리지가 실제로 무엇인지 확인**한다(커밋 메시지를 믿지 않는다 — §5-3 전례):
   ```
   tar xzOf <후보>.tar.gz ./jetson/scripts/operator_console.py | md5sum
   tar xzOf <후보>.tar.gz ./jetson/scripts/operator_console.py | grep -n 'CKPT_CHOICES\|CKPT_DEFAULT'
   tar xzOf <후보>.tar.gz ./jetson/scripts/rl_bridge_node.py   | md5sum
   ```
   md5 가 어느 커밋과도 안 맞으면 **그 패키지는 워킹트리 상태로 만들어진 것**이다 — git 으로 재현 불가, tar 가 유일 원본이므로 **그 tar 를 지우지 말 것**.

### 6.3 (a) 패키지 tar 가 남아 있는 경우 — 권장 경로
5. **Jetson action (not performed)** — 옛 패키지를 그대로 재배포:
   ```
   bash jetson/deploy_to_jetson.sh --host <IP> --pkg deploy/old/stand_7차_<옛stamp>.tar.gz --no-restart
   ```
   `--no-restart` 로 먼저 파일·검증만 끝낸다. `[2/7]` MD5, `[3/7]`·`[3b/7]` selftest, `[3c/7]` v5 md5 가 전부 통과해야 한다. 어느 단계든 "— 중단" 이면 진행 금지.
   ⚠ `[3b/7]`·`[3c/7]` 은 `stand_hang_v5` 를 요구한다. **v5 가 없던 시절 패키지(0909_1202 이전)로 롤백하면 이 단계에서 반드시 실패한다** — 그 경우 `deploy_to_jetson.sh` 를 쓰지 말고 6.4 의 수동 경로를 쓴다.
6. **Jetson action (not performed)** — `~/rl_calib/` 사본 되돌리기(자동 아님):
   ```
   ls -lt ~/rl_calib/release_all.py.bak_* ~/rl_calib/boot_pose_check.py.bak_*
   cp ~/rl_calib/release_all.py.bak_<해당> ~/rl_calib/release_all.py
   cp ~/rl_calib/boot_pose_check.py.bak_<해당> ~/rl_calib/boot_pose_check.py
   python3 ~/rl_calib/release_all.py --selftest | tail -1
   python3 -m py_compile ~/rl_calib/boot_pose_check.py
   ```
7. **Jetson action (not performed)** — 콘솔 재기동 + 확인:
   ```
   sudo systemctl restart operator-console && sleep 2 && systemctl is-active operator-console
   ```
   브라우저 새로고침 → 드롭다운 첫 항목(=기본값)이 **롤백 대상 판의 값**인지 눈으로 확인.

### 6.4 (b) tar 가 없고 git 만 있는 경우 — 재조립
8. (PC) 되돌릴 커밋으로 **별도 워크트리**를 만든다(현재 작업 트리를 오염시키지 않는다):
   ```
   git worktree add /tmp/rollback <되돌릴커밋>
   cd /tmp/rollback
   GENS="<그 시점의 세대 목록>" bash jetson/deploy_package.sh
   ```
   `GENS` 는 **기본값에 의존하지 말고 명시**한다(§5-1). 그 시점 카드(`docs/A1_*.md` 등)에서 실제 값을 확인할 것.
   ⚠ 정책 바이너리는 `models/<gen>/` 에서 온다. 그 커밋의 `models/` 에 세대가 없으면(예: `stand_v7/v8/v9/march_v5b`) **재조립으로 복원할 수 없다**(§5-4).
9. 재조립본이 원래 패키지와 같은지 대조(가능한 경우):
   ```
   diff <(tar xzOf /tmp/rollback/deploy/stand_7차_*.tar.gz ./deploy/MANIFEST.md5) \
        <(tar xzOf deploy/old/stand_7차_<원본>.tar.gz ./deploy/MANIFEST.md5)
   ```
   차이가 나면 **차이 나는 파일마다 이유를 적고**, 재익스포트로 인한 컨테이너 차이(§5-5)인지 실제 내용 차이인지 가중치로 확인한다.
10. 이후는 6.3 의 5~7 단계와 동일. **Jetson action (not performed)**.

### 6.5 롤백 후 필수 확인 (Jetson action (not performed))
11. `cd ~/humanoid_repo && LC_ALL=C md5sum -c --quiet deploy/MANIFEST.md5; echo "exit=$?"` — **종료코드로 판정**(한국어 로케일에서 `OK` 대신 `성공`이 찍혀 문자열 grep 은 금지).
12. `python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint <롤백 후 사용할 ckpt>` → `selftest: ALL PASS`, 「학습 계약」 줄에 그 ckpt 가 보여야 한다.
13. `python3 rl_walking/deploy/policy_runner_stage4.py --selftest` → ALL PASS.
14. `python3 jetson/scripts/operator_console.py --selftest` → `console selftest: ALL PASS`.
15. 콘솔 웹에서 드롭다운 기본값과 「브리지 live」 로그의 `checkpoint=` 가 **의도한 값**인지 확인. 브리지 debug 의 `obs` 길이가 기대 차원(v5 = **320**)인지 확인.
16. **CLAUDE.md 5번 규칙**: 로그에 Traceback 이 없다고 성공이 아니다. **산출물**(MANIFEST 종료코드 0, `ALL PASS` 줄, 블랙박스 파일 생성)로 판정한다.
17. 영점/하드웨어맵을 건드린 적이 있다면: **배수 보정 우선**(`zero_anchor.py`) → 맵 갱신 → **stage8 재기동** → `boot_pose_check` 12/12. 롤백은 영점을 되돌리지 않는다.

### 6.6 롤백 절차 자체의 취약점 (알고 들어갈 것)
- `deploy_to_jetson.sh` 는 `--pkg` 생략 시 **`ls -t` 로 최신 tar** 를 고른다 → 롤백 중에는 **항상 `--pkg` 를 명시**한다.
- `deploy_to_jetson.sh` 는 `sshpass` 로 고정 암호를 쓰고 `StrictHostKeyChecking=no` 다. 롤백 중 호스트가 바뀌었다면(젯슨 DHCP) 엉뚱한 장비에 쓸 위험이 있다 — `--host` 를 명시하고 `$SSH hostname` 출력을 눈으로 확인한다.
- 패키지는 **삭제를 표현할 수 없다**. "이 파일을 젯슨에서 없애기"는 tar 로 불가능하며 별도 수동 삭제가 필요하다(그 자체가 위험 작업 — 승윤님 결정 사항).
- `models/` 에 없는 세대(§5-4)는 어떤 롤백으로도 재현되지 않는다. 콘솔 드롭다운에서 그것들을 빼는 것이 근본 대책이다.

---

### 6.1 주의: 로컬 `logs/…/exported/policy.pt` ≠ 레지스트리인 세대가 있다 (MEASURED, 0911 23:5x)
- 워킹트리에서 `md5sum -c deploy/MANIFEST.md5` 를 돌리면 `logs/rsl_rl/biped12_stage4/2026-09-05_19-37-39/exported/policy.pt`(stand_v18b) 가 FAILED 로 나온다: 로컬 파일 md5 `1ddaef70…`(mtime 0908 20:17, 그 run 을 play.py 로 다시 돌려 재익스포트된 것) ≠ 레지스트리 `models/stand_v18b/policy.pt` `fac90eb5…`(meta.json, model_120998).
- 패키지는 **레지스트리 사본**을 브리지 CHECKPOINTS 경로로 복사하므로(`deploy_package.sh` 설계) 패키지 자체는 옳고 자기 일관(풀린 트리 MD5_OK). 젯슨에 풀면 레지스트리 사본이 그 경로에 놓인다. 로컬에서 v18b 를 브리지로 실행할 일이 있으면 `models/stand_v18b/policy.pt` 를 기준으로 보라.
- stand_hang_v5 는 로컬 logs 사본 = 레지스트리(`2a20b7b1…`) 로 일치한다. 나머지 FAILED 2건(`docs/A1_…실행카드.md`, `docs/operator_console_renewal.md`) 은 2337 패키지 뒤의 문서 갱신이었고 최종 패키지 `stand_7차_20260912_0526.tar.gz` 재생성으로 해소(워킹트리 대조 시 남는 FAILED 는 v18b 로컬 재익스포트 1건뿐).

## 7. 권고 (승윤님 결정 대기, 본 감사에서는 아무것도 바꾸지 않음)

1. `deploy_package.sh` 의 `GENS` 기본값을 현행 배포 세트(`stand_v18b stand_hang_v4 stand_hang_v5`)로 맞추거나, 기본값 사용 시 **명시 실패**시킨다 (§5-1).
2. HEAD 기준으로 패키지를 **재조립**한다 — 인계 워밍업 수정(`2a88452`)·콘솔 리뉴얼·어댑터 per-foot 스케일이 전부 미배포다 (§5-2).
3. 패키지에 `models/<gen>/meta.json` 을 함께 실어 젯슨이 **자력으로 provenance 를 답할 수 있게** 한다 (§3.1).
4. `deploy_to_jetson.sh` `[3c/7]` 의 md5 대조를 `policy.onnx` 까지 확장하고, 실제 선택된 백엔드를 로그로 남긴다 (§5-8).
5. 콘솔 `CKPT_CHOICES` 에서 `models/` 레지스트리가 없는 4개(`stand_v7/v8/v9/march_v5b`)를 제거한다 (§5-4).
6. 콘솔 selftest 에 "기본 정책 = 승윤님 확정 데뷔 후보" 고정핀을 되살린다 (§5-10).
7. A1 카드 §4 와 `deploy_to_jetson.sh:57` 의 "기본값 stand_v18b" 기술을 코드와 맞춘다 (§5-11).
8. `register_model.py` 에 "`exported/` 가 `--ckpt` 에서 나왔는지" 검증(가중치 대조)을 코드로 넣는다 — 이번 v5 는 사후 확인으로 통과했지만 절차엔 게이트가 없다 (§4).
9. 320 관측 조립의 **env ↔ 배포러너 교차검증**을 HangStand 태스크로 한 번 돌려 산출물로 남긴다 (§5-6). 이것이 남은 계약 리스크 중 유일하게 시뮬만으로 닫을 수 있는 항목이다.

---

## 부록 A. 본 감사에서 실행한 명령 (전부 읽기 전용 / CPU)
```
md5sum models/stand_hang_v5/* ; cat models/stand_hang_v5/meta.json
md5sum logs/rsl_rl/biped12_stage4/2026-09-09_19-15-42/{model_139987.pt,exported/*,params/*}
tar tzf deploy/stand_7차_*.tar.gz | grep exported/policy.pt        # 추출 없음
tar xzOf deploy/stand_7차_*.tar.gz ./deploy/MANIFEST.md5           # stdout 비교만
git show <rev>:<path> | md5sum                                     # 커밋별 대조
CUDA_VISIBLE_DEVICES= python3 jetson/scripts/rl_bridge_node.py --selftest --checkpoint stand_hang_v5
CUDA_VISIBLE_DEVICES= python3 jetson/scripts/operator_console.py --selftest
CUDA_VISIBLE_DEVICES= python3 -c "torch.jit.load / torch.load 로 actor 가중치 8/8 대조"
```
`operator_console.py --selftest` 는 `SESSION_DIR` 을 임시 디렉토리로 patch 한 뒤 돌므로 저장소에 파일을 남기지 않는다(MEASURED — 코드 확인 + 실행 후 `git status` 무변화).

## 부록 B. 용어
- **레지스트리** = `models/<gen>/` (git 추적, policy.pt·onnx·env.yaml·agent.yaml·meta.json)
- **학습 런** = `logs/rsl_rl/biped12_stage4/<run>/` (gitignore; `model_*.pt` 원 체크포인트 + `exported/` 익스포트 + `params/`)
- **패키지** = `deploy/stand_7차_<stamp>.tar.gz` (gitignore) + 그 안팎의 `deploy/MANIFEST.md5`(git 추적)
- **익스포트** = `play.py` 의 `export_policy_as_jit`/`_as_onnx` 산출물. actor 전용, 학습 체크포인트와 **바이트 동일하지 않음**
