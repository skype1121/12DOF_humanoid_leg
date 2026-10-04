"""rsl_rl PPO 러너 설정 — G1 flat 레시피 기반, 비대칭 액터-크리틱."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class Biped12FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "biped12_flat"
    # 비대칭: 크리틱만 특권 관측(base_lin_vel 등 critic 그룹)을 추가로 받는다
    obs_groups = {"policy": ["policy"], "critic": ["policy", "critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # log-std: 물리 폭발 배치가 그래디언트를 흔들어도 std>0 보장
        # (scalar형은 stage2에서 음수 진입 → normal() 크래시 실측)
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Biped12Stage4PPORunnerCfg(Biped12FlatPPORunnerCfg):
    """Stage4 — 기존 러너 복제 + 미러 대칭 손실 (Yu 2018 계열, rsl_rl 네이티브).

    use_data_augmentation=False / use_mirror_loss=True: 배치 자체는 원본만 학습하고,
    미러 관측에 대한 액터 평균이 미러 액션과 일치하도록 MSE 손실만 추가 (coeff 0.5).
    미러 함수(관측 245/크리틱 260 레이아웃)는 rl_walking/symmetry.py 참조.
    """

    max_iterations = 3000
    experiment_name = "biped12_stage4"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=0.5,
            # 문자열 경로 — rsl_rl ppo.py가 string_to_callable로 해석 (lazy import)
            data_augmentation_func="rl_walking.symmetry:mirror_biped12",
        ),
    )


@configclass
class Biped12Stage4PPOHoldPriorRunnerCfg(Biped12Stage4PPORunnerCfg):
    """0920 G6: '정답 동작 = 홀드(액션 0)' 에서 출발 — 초기 행동 노이즈 std 1.0 → 0.25 (액션 스케일 0.25 rad × 1.0 = ±14° 무작위 명령이
    kp150 에서 첫 스텝부터 서기를 무너뜨림). 나머지 PPO·대칭 손실 동일."""

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.policy.init_noise_std = 0.25


@configclass
class Biped12G22RPPORunnerCfg(Biped12Stage4PPOHoldPriorRunnerCfg):
    """1003 G22R: G21a 와 같은 PPO(동결), 실험 디렉터리만 분리."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "biped12_g22r"


@configclass
class Biped12W1PPORunnerCfg(Biped12Stage4PPOHoldPriorRunnerCfg):
    """1003 W1: G21a 와 같은 PPO(동결), 실험 디렉터리만 분리."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "biped12_w1"


@configclass
class Biped12StepB0PPORunnerCfg(Biped12Stage4PPOHoldPriorRunnerCfg):
    """1003 STEP-B0: G21a 와 같은 PPO(동결), 실험 디렉터리만 분리."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "biped12_stepb0"


@configclass
class Biped12StepB0SafePPORunnerCfg(Biped12StepB0PPORunnerCfg):
    """1004 밤: 이어 학습 표류 방지(독립 검토 7회차) — 고정 학습률 1e-4·엔트로피 0. 같은 실험 디렉터리."""

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
