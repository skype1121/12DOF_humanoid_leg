"""rsl_rl PPO 러너 설정 — G1 flat 레시피 기반, 비대칭 액터-크리틱."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


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
