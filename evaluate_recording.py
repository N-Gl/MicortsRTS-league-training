from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import time
import torch

import agent_model
from evaluate import (
    _build_java_actions,
    _build_table_row,
    _dispose_big_render_window,
    _force_close_java_windows,
    _log_endgame_unit_counts,
    _log_local_results,
    _make_eval_env,
    _render_eval_env,
    _resolve_checkpoint_path,
    _sanitize_metric_component,
)
from selfplay_evaluate_recording import _SelfplayEvaluationRecorder, _sanitize_path_component


def bot_evaluate_agent(
    args,
    evaluation_opponents: Optional[Sequence[Tuple[str, Any]]],
    device: torch.device,
    get_scalar_features,
    reward_weight: np.ndarray,
    vecstats_monitor_cls,
    writer=None,
    evaluated_agent_name: Optional[str] = None,
) -> None:
    opponents = evaluation_opponents
    checkpoint_path = _resolve_checkpoint_path(args.model_path)
    global_step = 0
    start_time = time.time()

    aggregate_stats = {"win": 0, "draw": 0, "loss": 0}
    aggregate_episode_rewards: List[float] = []
    opponent_table_rows: List[Tuple] = []
    agent_metric_name = evaluated_agent_name or Path(checkpoint_path).stem

    if args.render_all:
        from ppo import Rendering

    for opponent_index, (opponent_name, opponent_ai) in enumerate(opponents):
        opponent_metric_name = _sanitize_metric_component(opponent_name)
        eval_env = _make_eval_env(opponent_ai, args, reward_weight, vecstats_monitor_cls)
        recorder = _make_bot_eval_recorder(args, checkpoint_path, opponent_name, opponent_index)
        mapsize = 16 * 16
        position_indices = (
            torch.arange(mapsize, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .repeat(args.num_envs, 1)
            .unsqueeze(2)
        )

        agent = agent_model.build_agent(
            eval_env.action_plane_space.nvec,
            device,
            unit_exploiters=getattr(args, "unit_exploiters", False),
        )
        agent.set_weights(checkpoint_path)
        agent.eval()

        unit_bonus_distr = torch.zeros(4, device=agent.device)

        try:
            obs_np, _, res = eval_env.reset()
            if recorder is not None:
                recorder.capture(eval_env)
            obs = torch.as_tensor(obs_np, device=device)
            z_features = torch.zeros((args.num_envs, 8), dtype=torch.long, device=device)
            attack_weight = 0.05
            winloss_weight = 10.0

            local_stats = {"win": 0, "draw": 0, "loss": 0}
            local_episode_rewards: List[float] = []
            completed = 0
            target_episodes = args.num_eval_episodes

            round_completed = 0
            env_done_in_round = [False for _ in range(args.num_envs)]
            round_target = min(args.num_envs, target_episodes - completed)

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
                    actions, _, _, invalid_masks = agent.get_action(
                        obs,
                        scalar_features,
                        z_features,
                        envs=eval_env,
                        unit_bonus_distr=unit_bonus_distr,
                    )

                    real_action = torch.cat([position_indices, actions], dim=2).cpu().numpy()
                    valid_mask = invalid_masks[:, :, 0].bool().cpu().numpy()
                    valid_actions = real_action[valid_mask]
                    valid_counts = invalid_masks[:, :, 0].sum(1).long().cpu().numpy()

                    java_valid_actions = _build_java_actions(valid_actions, valid_counts)

                    next_obs_np, _, _, _, _, _, infos, res = eval_env.step(java_valid_actions)
                    if recorder is not None:
                        recorder.capture(eval_env)
                    next_obs_np = eval_env._from_microrts_obs(next_obs_np)
                    obs = torch.as_tensor(next_obs_np, device=device)

                    global_step += args.num_envs

                    for env_index, info in enumerate(infos):
                        if env_done_in_round[env_index]:
                            continue

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
                            if writer is not None:
                                _log_endgame_unit_counts(
                                    writer=writer,
                                    agent_name=agent_metric_name,
                                    scalar_features_step=scalar_features,
                                    done_idx=env_index,
                                    game_index=completed,
                                    game_type=f"bot_game_{opponent_metric_name}",
                                )
                            winloss_weight = winloss_weight * (-0.00013 * info["episode"]["l"] + 1.16)
                            local_episode_rewards.append(
                                info["microrts_stats"]["RAIWinLossRewardFunction"] * winloss_weight
                                + info["microrts_stats"]["AttackRewardFunction"] * attack_weight
                            )

                        env_done_in_round[env_index] = True
                        round_completed += 1
                        completed += 1
                        if completed >= target_episodes:
                            break

                    if completed >= target_episodes:
                        break

                    if round_completed >= round_target:
                        print(f"Evaluation vs {opponent_name}: {completed}/{target_episodes} games completed, {local_stats}")
                        env_done_in_round = [False for _ in range(args.num_envs)]
                        round_completed = 0
                        round_target = min(args.num_envs, target_episodes - completed)

        finally:
            if recorder is not None:
                recorder.close()
            _force_close_java_windows()

        _log_local_results(
            opponent_name,
            local_stats,
            local_episode_rewards,
            aggregate_stats,
            aggregate_episode_rewards,
            global_step,
            start_time,
        )
        opponent_table_rows.append(
            _build_table_row(opponent_name, local_stats, local_episode_rewards)
        )

    _dispose_big_render_window(eval_env)

    return aggregate_stats, aggregate_episode_rewards, opponent_table_rows


def _make_bot_eval_recorder(args, checkpoint_path: str, opponent_name: str, opponent_index: int):
    if not getattr(args, "capture_video", False):
        return None

    exp_name = _sanitize_path_component(getattr(args, "exp_name", "") or "evaluation")
    model_name = _sanitize_path_component(Path(checkpoint_path).stem or "agent")
    opponent_label = _sanitize_path_component(opponent_name or f"opponent_{opponent_index:02d}")
    output_dir = Path("videos") / exp_name / "evaluation" / model_name / "bots"
    file_stem = f"{opponent_index:02d}_{opponent_label}"

    try:
        from ppo import Rendering, Image as PilImage
    except Exception as exc:
        return _SelfplayEvaluationRecorder(
            output_dir=output_dir,
            file_stem=file_stem,
            rendering=None,
            pil_image=None,
            client_index=getattr(args, "capture_video_env_index", 0),
            init_error=f"Could not import recorder helpers from ppo.py: {exc}",
        )

    return _SelfplayEvaluationRecorder(
        output_dir=output_dir,
        file_stem=file_stem,
        rendering=Rendering,
        pil_image=PilImage,
        client_index=getattr(args, "capture_video_env_index", 0),
    )
