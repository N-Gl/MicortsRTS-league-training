import os
from typing import Any, List, Optional, Sequence, Tuple

import jpype
import time
import numpy as np
import torch
from gym_microrts.envs.microrts_vec_env import MicroRTSGridModeVecEnv
from jpype.types import JArray, JInt
from league import MainPlayer, Payoff
from microrts_space_transform import MicroRTSSpaceTransform
import agent_model
import selfplay_only
from log_aggregate_result_table import Logger



# TODO: mache ein neues Parameter in args für die anzahl an environments, args.num_selfplay_envs, args.num_bot_envs sollte unten in der Methode berechnet werden und dann auch benutzt

def evaluate_agent(
    args,
    default_opponent_paths: Sequence[Tuple[str, Any, str, Optional[Any]]],
    device: torch.device,
    get_scalar_features,
    reward_weight: np.ndarray,
    vecstats_monitor_cls
):
    opponents = default_opponent_paths

    checkpoint_path = _resolve_checkpoint_path(args.model_path)
    global_step = 0
    start_time = time.time()

    target_episodes = args.num_eval_episodes
    mapsize = 16 * 16
    position_indices = (
        torch.arange(mapsize, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .repeat(args.num_parallel_selfplay_eval_games, 1)
        .unsqueeze(2)
    )

    aggregate_stats = {"win": 0, "draw": 0, "loss": 0}
    aggregate_episode_rewards: List[float] = []
    opponent_table_rows: List[Tuple] = []
    active_league_agents: List[MainPlayer] = []

    if args.render_all:
        from ppo import Rendering

    for idx, (opponent_name, opponent_ai, opponent_path,  league_agent) in enumerate(opponents):
        eval_env = _make_selfplay_eval_env(args, reward_weight, vecstats_monitor_cls)
        active_league_agents = []

        if not opponent_ai == agent_model.Agent:
            opponent_ai = agent_model.Bot_Agent(
                        eval_env, 
                        range(args.Bot_as_player_1, args.num_parallel_selfplay_eval_games//2, 2), 
                        opponent_ai, 
                        device=device,
                        player_id=args.Bot_as_player_1
                    ).to(device)
        else:
            opponent_ai = opponent_ai(eval_env.action_plane_space.nvec, device).to(device)
            opponent_ai.load_state_dict(torch.load(opponent_path, map_location=device, weights_only=True))
            opponent_ai.eval()

        if league_agent is None:
            league_agent = MainPlayer(opponent_ai, Payoff(), args)

        agent = agent_model.Agent(eval_env.action_plane_space.nvec, device).to(device)
        agent.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        agent.eval()

        main_league_agent = MainPlayer(agent, Payoff(), args)

        if args.Bot_as_player_1:
            for _ in range(args.num_parallel_selfplay_eval_games//2):
                active_league_agents.append(main_league_agent)
                active_league_agents.append(league_agent)
        else:
            for _ in range(args.num_parallel_selfplay_eval_games//2):
                active_league_agents.append(league_agent)
                active_league_agents.append(main_league_agent)

        try:
            obs_np, _, res = eval_env.reset()
            obs = torch.as_tensor(obs_np, device=device)
            selfplay_only.adjust_obs_selfplay(args, obs, True)
            z_features = torch.zeros((args.num_parallel_selfplay_eval_games, 8), dtype=torch.long, device=device)
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

                    for env_index in range(args.num_parallel_selfplay_eval_games):
                        z_features[env_index] = agent.z_encoder(obs[env_index].view(-1)) # TODO: selfplay_get_z_encoded_features

                    scalar_features = get_scalar_features(obs, res, args.num_parallel_selfplay_eval_games).to(device)
                    actions, _, _, invalid_masks = agent.selfplay_get_action(
                        obs, scalar_features, z_features, 
                        num_selfplay_envs=args.num_parallel_selfplay_eval_games, num_envs=args.num_parallel_selfplay_eval_games, 
                        envs=eval_env, active_league_agents=active_league_agents
                        )

                    real_action = torch.cat([position_indices, actions], dim=2).cpu().numpy()
                    valid_mask = invalid_masks[:, :, 0].bool().cpu().numpy()
                    valid_actions = real_action[valid_mask]
                    valid_counts = invalid_masks[:, :, 0].sum(1).long().cpu().numpy()

                    selfplay_only.adjust_action_selfplay(args, valid_actions, valid_counts)

                    java_valid_actions = _build_java_actions(valid_actions, valid_counts)

                    next_obs_np, _, _, _, _, ds, infos, res = eval_env.step(java_valid_actions)
                    next_obs_np = eval_env._from_microrts_obs(next_obs_np)
                    obs = torch.as_tensor(next_obs_np, device=device)
                    selfplay_only.adjust_obs_selfplay(args, obs, False)

                    global_step += args.num_parallel_selfplay_eval_games

                    if np.any(['episode' in info.keys() for info in infos]):

                        where_done = np.where(ds)

                        for done_idx in where_done[0]:
                            if done_idx % 2 == 1:
                                    continue
                                    
                            info = infos[done_idx]

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

        Logger.log_local_results(
            opponent_name,
            local_stats,
            local_episode_rewards,
            aggregate_stats,
            aggregate_episode_rewards,
            global_step,
            start_time
        )
        opponent_table_rows.append(
            Logger.build_table_row(opponent_name, local_stats, local_episode_rewards)
        )
        if idx + 1 < len(opponents):
            print(f"next opponent: {opponents[idx + 1][0]}")

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

def _make_selfplay_eval_env(args, reward_weight, vecstats_monitor_cls):
    if args.num_parallel_selfplay_eval_games % 2 != 0:
        raise ValueError(
            f"num_selfplay_envs must be even for selfplay evaluation (got {args.num_parallel_selfplay_eval_games}). "
            "Each selfplay match consumes two environments, so please provide an even number."
        )
    
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=args.num_parallel_selfplay_eval_games,
        num_bot_envs=0,
        max_steps=2000,
        always_player_1=True,
        bot_envs_alternate_player=False,
        render_theme=1,
        ai2s=[],
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

