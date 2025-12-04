import os
from typing import Any, List, Optional, Sequence, Tuple

import jpype
import time
import numpy as np
import torch
from gym_microrts.envs.microrts_vec_env import MicroRTSGridModeVecEnv
from jpype.types import JArray, JInt
import agent_model
from microrts_space_transform import MicroRTSSpaceTransform


def bot_evaluate_agent(
    args,
    evaluation_opponents: Optional[Sequence[Tuple[str, Any]]],
    device: torch.device,
    get_scalar_features,
    reward_weight: np.ndarray,
    vecstats_monitor_cls,
) -> None:
    opponents = evaluation_opponents
    checkpoint_path = _resolve_checkpoint_path(args.model_path)
    global_step = 0
    start_time = time.time()
    

    target_episodes = args.num_eval_episodes
    mapsize = 16 * 16
    position_indices = (
        torch.arange(mapsize, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .repeat(args.num_envs, 1)
        .unsqueeze(2)
    )

    aggregate_stats = {"win": 0, "draw": 0, "loss": 0}
    aggregate_episode_rewards: List[float] = []
    opponent_table_rows: List[Tuple] = []

    if args.render_all:
        from ppo import Rendering

    for opponent_name, opponent_ai in opponents:
        eval_env = _make_eval_env(opponent_ai, args, reward_weight, vecstats_monitor_cls)

        agent = agent_model.Agent(eval_env.action_plane_space.nvec, device).to(device)
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        agent.eval()

        try:
            obs_np, _, res = eval_env.reset()
            obs = torch.as_tensor(obs_np, device=device)
            z_features = torch.zeros((args.num_envs, 8), dtype=torch.long, device=device)
            attack_weight = 0.05
            winloss_weight = 10.0

            local_stats = {"win": 0, "draw": 0, "loss": 0}
            local_episode_rewards: List[float] = []
            completed = 0

            with torch.inference_mode():
                while completed < target_episodes:
                    if args.render:
                        if args.render_all:
                            _render_eval_env(eval_env, args, Rendering)
                        else:
                            eval_env.render("human")

                    for env_index in range(args.num_envs):
                        z_features[env_index] = agent.z_encoder(obs[env_index].view(-1))

                    scalar_features = get_scalar_features(obs.cpu(), res, args.num_envs).to(device)
                    actions, _, _, invalid_masks = agent.get_action(obs, scalar_features, z_features, envs=eval_env)

                    real_action = torch.cat([position_indices, actions], dim=2).cpu().numpy()
                    valid_mask = invalid_masks[:, :, 0].bool().cpu().numpy()
                    valid_actions = real_action[valid_mask]
                    valid_counts = invalid_masks[:, :, 0].sum(1).long().cpu().numpy()

                    java_valid_actions = _build_java_actions(valid_actions, valid_counts)

                    next_obs_np, _, _, _, _, _, infos, res = eval_env.step(java_valid_actions)
                    next_obs_np = eval_env._from_microrts_obs(next_obs_np)
                    obs = torch.as_tensor(next_obs_np, device=device)

                    global_step += args.num_envs

                    for info in infos:
                        stats_entry = info.get("microrts_stats")
                        if not stats_entry:
                            continue

                        reward = stats_entry.get("RAIWinLossRewardFunction", 0)
                        if reward > 0:
                            local_stats["win"] += 1
                        elif reward < 0:
                            local_stats["loss"] += 1
                        else:
                            local_stats["draw"] += 1

                        if "episode" in info:
                            winloss_weight = winloss_weight * (-0.00013 * info["episode"]["l"] + 1.16)
                            local_episode_rewards.append(
                                info["microrts_stats"]["RAIWinLossRewardFunction"] * winloss_weight
                                + info["microrts_stats"]["AttackRewardFunction"] * attack_weight
                            )

                        completed += 1
                        if completed >= target_episodes:
                            break
                        else:
                            if completed % (target_episodes//10) == 0:
                                print()
                                print(f"Evaluation vs {opponent_name}: {completed}/{target_episodes} games completed, {local_stats}")
                                
        finally:
            _force_close_java_windows()

        _log_local_results(
            opponent_name,
            local_stats,
            local_episode_rewards,
            aggregate_stats,
            aggregate_episode_rewards,
            global_step,
            start_time
        )
        opponent_table_rows.append(
            _build_table_row(opponent_name, local_stats, local_episode_rewards)
        )

    _dispose_big_render_window(eval_env)

    return aggregate_stats, aggregate_episode_rewards, opponent_table_rows


def _resolve_checkpoint_path(model_path: str) -> str:
    if model_path.endswith(".pt"):
        checkpoint_path = model_path
    else:
        checkpoint_path = f"models/{model_path}/agent.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    return checkpoint_path


def _make_eval_env(opponent_ai, args, reward_weight, vecstats_monitor_cls):
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=args.num_envs,
        max_steps=2000,
        render_theme=1,
        ai2s=[opponent_ai for _ in range(args.num_envs)],
        map_paths=["maps/16x16/basesWorkers16x16A.xml"],
        reward_weight=reward_weight,
    )
    env = MicroRTSSpaceTransform(env)
    return vecstats_monitor_cls(env, args.gamma)


def _dispose_big_render_window(env) -> None:
    from ppo import Rendering

    Rendering._destroy_tk_window(permanent=True)
    Rendering._viewer_disabled = True

def _force_close_java_windows() -> None:
    if not jpype.isJVMStarted():
        return
    
    cls = jpype.JClass("java.awt.Window")
    getter = getattr(cls, "getWindows")
    windows = list(getter())
    for window in windows:
        window.dispose()

def _render_eval_env(eval_env, args, rendering):
    if not args.render:
        return
    if args.render_all:
        rendering.render_all_envs(eval_env)
    else:
        eval_env.render("human")


def _build_java_actions(valid_actions: np.ndarray, valid_counts: np.ndarray):
    java_valid_actions: List = []
    valid_index = 0
    for count in valid_counts:
        java_env_action = []
        for _ in range(count):
            java_env_action.append(JArray(JInt)(valid_actions[valid_index]))
            valid_index += 1
        java_valid_actions.append(JArray(JArray(JInt))(java_env_action))
    return JArray(JArray(JArray(JInt)))(java_valid_actions)


def _log_local_results(
    opponent_name: str,
    local_stats: dict,
    local_episode_rewards: List[float],
    aggregate_stats: dict,
    aggregate_episode_rewards: List[float],
    global_step: int,
    start_time: int
):
    total_games = sum(local_stats.values())
    avg_reward = float(np.mean(local_episode_rewards)) if local_episode_rewards else 0.0
    win_rate = local_stats["win"] / total_games if total_games else 0.0
    draw_rate = local_stats["draw"] / total_games if total_games else 0.0
    loss_rate = local_stats["loss"] / total_games if total_games else 0.0

    aggregate_episode_rewards.extend(local_episode_rewards)
    for key in aggregate_stats:
        aggregate_stats[key] += local_stats[key]

    print(
        f"Evaluation vs {opponent_name} over {total_games} games | "
        f"win: {local_stats['win']} ({win_rate:.2%}), "
        f"draw: {local_stats['draw']} ({draw_rate:.2%}), "
        f"loss: {local_stats['loss']} ({loss_rate:.2%}), "
        f"avg reward: {avg_reward:.3f}"
    )

    sps = int(global_step / (time.time() - start_time))
    print("SPS:", sps)


def _build_table_row(opponent_name: str, local_stats: dict, local_episode_rewards: List[float]):
    total_games = sum(local_stats.values())
    avg_reward = float(np.mean(local_episode_rewards)) if local_episode_rewards else 0.0
    win_rate = local_stats["win"] / total_games if total_games else 0.0
    draw_rate = local_stats["draw"] / total_games if total_games else 0.0
    loss_rate = local_stats["loss"] / total_games if total_games else 0.0
    return (
        opponent_name,
        total_games,
        local_stats["win"],
        local_stats["draw"],
        local_stats["loss"],
        win_rate,
        draw_rate,
        loss_rate,
        avg_reward,
    )