from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    exp_name: str
    gym_id: str
    BC_learning_rate: float
    BC_weight_decay: float
    PPO_learning_rate: float
    PPO_weight_decay: float
    seed: int
    exact_seed: bool
    total_timesteps: int
    torch_deterministic: bool
    cuda: bool
    prod_mode: bool
    capture_video: bool
    wandb_project_name: str
    wandb_entity: Optional[str]
    create_run_id_file: bool
    evaluation_model_paths: Optional[list[str]]
    names: Optional[list[str]]
    render: bool
    render_all: bool
    n_minibatch: int
    num_bot_envs: int
    num_league_exploiters: int
    num_main_exploiters: int
    num_main_agents: int
    new_hist_rewards: int
    attack_reward_weight: float
    less_draw: float
    num_steps: int
    gamma: float
    gae_lambda: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    clip_coef: float
    update_epochs: int
    kle_stop: bool
    kle_rollback: bool
    target_kl: float
    kl_coeff: float
    norm_adv: bool
    anneal_lr: bool
    clip_vloss: bool
    model_path: str
    initial_BC: bool
    BC_finetuning: bool
    ppo: bool
    evaluate: bool
    selfplay: bool
    pfsp: bool
    league_training: bool
    train_on_old_mains: bool
    selfplay_save_interval: int
    checkpoint_frequency: int
    resume: bool
    epochs: int
    warmup_epochs: int
    newdata: bool
    deterministic: bool
    nurwins: bool
    num_eval_episodes: int
    num_parallel_eval_envs: int
    num_parallel_selfplay_eval_games: int
    dbg_no_main_agent_ppo_update: bool


    
