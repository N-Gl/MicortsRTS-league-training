import os
import re
from typing import Any, List, Optional, Sequence, Tuple
from pathlib import Path

import jpype
import time
import numpy as np
import torch
from gym_microrts.envs.microrts_vec_env import MicroRTSGridModeVecEnv
from jpype.types import JArray, JInt
from league import MainPlayer, Payoff
from microrts_space_transform import MicroRTSSpaceTransform
import agent_model
import selfplay_league
import selfplay_only
from evaluate import _log_endgame_unit_counts, _sanitize_metric_component
from log_aggregate_result_table import Logger


# TODO: mache ein neues Parameter in args für die anzahl an environments, args.num_selfplay_envs, args.num_bot_envs sollte unten in der Methode berechnet werden und dann auch benutzt

def evaluate_agent(
    args,
    default_opponent_paths: Sequence[Tuple[str, Any, str, Optional[Any]]],
    device: torch.device,
    get_scalar_features,
    reward_weight: np.ndarray,
    vecstats_monitor_cls,
    writer=None,
    evaluated_agent_name: Optional[str] = None,
):
    opponents = default_opponent_paths

    checkpoint_path = _resolve_checkpoint_path(args.model_path)
    global_step = 0
    start_time = time.time()
    agent_metric_name = evaluated_agent_name or Path(checkpoint_path).stem

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

    for idx, (opponent_name, opponent_ai, opponent_path,  league_agent, opp_unit_exploiter) in enumerate(opponents):
        opponent_metric_name = _sanitize_metric_component(opponent_name)
        eval_env = _make_selfplay_eval_env(args, reward_weight, vecstats_monitor_cls)
        active_league_agents = []
        recorder = _make_selfplay_eval_recorder(args, checkpoint_path, opponent_name, idx)

        if not opponent_ai == agent_model.Agent:
            opponent_ai = agent_model.Bot_Agent(
                        eval_env, 
                        range(args.Bot_as_player_1, args.num_parallel_selfplay_eval_games//2, 2), 
                        opponent_ai, 
                        device=device,
                        player_id=args.Bot_as_player_1
                    ).to(device)
        else:
            opponent_ai = agent_model.build_agent(
                eval_env.action_plane_space.nvec,
                device,
                unit_exploiters=opp_unit_exploiter,
            )
            opponent_ai.set_weights(opponent_path)
            opponent_ai.eval()

        if league_agent is None:
            league_agent = MainPlayer(opponent_ai, Payoff(), args)

        agent = agent_model.build_agent(
            eval_env.action_plane_space.nvec,
            device,
            unit_exploiters=getattr(args, "unit_exploiters", False),
        )
        agent.set_weights(checkpoint_path)
        agent.eval()

        main_league_agent = MainPlayer(agent, Payoff(), args)
        main_league_agent.unit_bonus_distr = torch.zeros(4, device=agent.device)

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
            if recorder is not None:
                recorder.capture(eval_env)
            obs = torch.as_tensor(obs_np, device=device)
            selfplay_league.adjust_obs_selfplay(args, obs, True)
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
                    unit_bonus_distr = None
                    if args.unit_exploiters:
                        unit_bonus_distr = torch.zeros(
                            (args.num_parallel_selfplay_eval_games, 4),
                            dtype=torch.float,
                            device=device,
                        )
                    actions, logprob, entropy, invalid_masks = agent.selfplay_get_action(
                        obs, scalar_features, z_features, 
                        num_selfplay_envs=args.num_parallel_selfplay_eval_games, num_envs=args.num_parallel_selfplay_eval_games, 
                        envs=eval_env, active_league_agents=active_league_agents, dbg_deterministic_actions=args.dbg_deterministic_actions, unit_bonus_distr=unit_bonus_distr
                        )

                    real_action = torch.cat([position_indices, actions], dim=2).cpu().numpy()
                    valid_mask = invalid_masks[:, :, 0].bool().cpu().numpy()
                    valid_actions = real_action[valid_mask]
                    valid_counts = invalid_masks[:, :, 0].sum(1).long().cpu().numpy()

                    # TODO: debug nachher löschen:
                    dbg_valid_actions = valid_actions.copy()

                    selfplay_league.adjust_action_selfplay(args, valid_actions, valid_counts)

                    java_valid_actions = _build_java_actions(valid_actions, valid_counts)

                    next_obs_np, _, _, _, _, ds, infos, res = eval_env.step(java_valid_actions)
                    if recorder is not None:
                        recorder.capture(eval_env)
                    next_obs_np = eval_env._from_microrts_obs(next_obs_np)
                    obs = torch.as_tensor(next_obs_np, device=device)
                    selfplay_league.adjust_obs_selfplay(args, obs, False)

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
                                if writer is not None:
                                    _log_endgame_unit_counts(
                                        writer=writer,
                                        agent_name=agent_metric_name,
                                        scalar_features_step=scalar_features,
                                        done_idx=done_idx,
                                        game_index=completed,
                                        game_type=f"selfplay_eval_{opponent_metric_name}",
                                    )
                                winloss_weight = winloss_weight * (-0.00013 * info["episode"]["l"] + 1.16)
                                local_episode_rewards.append(
                                    info["microrts_stats"]["RAIWinLossRewardFunction"] * winloss_weight
                                    + info["microrts_stats"]["AttackRewardFunction"] * attack_weight
                                )

                            completed += 1
                                
                            if completed >= target_episodes:
                                    break
                            else:
                                if target_episodes >= 10 and completed % (target_episodes//10) == 0:
                                    print()
                                    print(f"Evaluation vs {opponent_name}: {completed}/{target_episodes} games completed, {local_stats}")
                                

        finally:
            if recorder is not None:
                recorder.close()
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


def _make_selfplay_eval_recorder(args, checkpoint_path: str, opponent_name: str, opponent_index: int):
    if not getattr(args, "capture_video", False):
        return None

    exp_name = _sanitize_path_component(getattr(args, "exp_name", "") or "evaluation")
    model_name = _sanitize_path_component(Path(checkpoint_path).stem or "agent")
    opponent_label = _sanitize_path_component(opponent_name or f"opponent_{opponent_index:02d}")
    output_dir = Path("videos") / exp_name / "evaluation" / model_name / "selfplay"
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


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "item"


class _SelfplayEvaluationRecorder:
    def __init__(
        self,
        output_dir: Path,
        file_stem: str,
        rendering,
        pil_image,
        client_index: int = 0,
        init_error: Optional[str] = None,
    ) -> None:
        self.output_dir = output_dir
        self.file_stem = file_stem
        self.rendering = rendering
        self.pil_image = pil_image
        self.init_error = init_error
        fallback_index = os.environ.get("MICRORTS_EVAL_VIDEO_ENV_INDEX", "0")
        self.client_index = max(0, int(client_index if client_index is not None else fallback_index))
        self.capture_attempts = 0
        self.sample_stride = 1
        self.max_frames = 120
        self.max_image_size = (1280, 1280)
        self.frames: List[Any] = []
        self.last_reason = init_error or ""

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture(self, env) -> None:
        self.capture_attempts += 1

        if self.rendering is None:
            self.last_reason = self.init_error or "Recorder helpers unavailable"
            return

        if self.capture_attempts != 1 and self.capture_attempts % self.sample_stride != 0:
            return

        frame = self._capture_frame(env)
        if frame is None:
            return

        self.frames.append(frame.copy())
        while len(self.frames) > self.max_frames:
            self.frames = self.frames[::2]
            self.sample_stride *= 2

    def close(self) -> None:
        if not self.frames:
            self._write_status_file()
            return

        if self._save_animation(".webp", "WEBP", quality=80, method=6):
            return
        if self._save_animation(".gif", "GIF"):
            return

        frame_dir = self.output_dir / f"{self.file_stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(self.frames):
            frame.save(frame_dir / f"{index:04d}.png")

    def _capture_frame(self, env):
        try:
            vec_client = self.rendering._locate_vec_client(env)
        except Exception as exc:
            self.last_reason = f"Could not locate vec_client: {exc}"
            return None

        if vec_client is None:
            self.last_reason = "Could not locate vec_client"
            return None

        try:
            clients = self.rendering._get_clients(vec_client)
        except Exception as exc:
            self.last_reason = f"Could not collect clients: {exc}"
            return None

        if not clients:
            self.last_reason = "No render clients found"
            return None

        client = clients[min(self.client_index, len(clients) - 1)]
        try:
            frame_bytes = client.render(True)
        except Exception as exc:
            self.last_reason = f"client.render(True) failed: {exc}"
            return None
        if frame_bytes is None:
            self.last_reason = "client.render(True) returned no frame"
            return None

        frame = self.rendering._bytes_to_image(frame_bytes)
        if frame is None:
            self.last_reason = "Could not decode rendered frame"
            return None

        return self._resize_frame(frame)

    def _resize_frame(self, image):
        if image.width <= self.max_image_size[0] and image.height <= self.max_image_size[1]:
            return image

        resized = image.copy()
        resampling = getattr(self.pil_image, "Resampling", None)
        lanczos = getattr(resampling, "LANCZOS", getattr(self.pil_image, "LANCZOS", None))
        if lanczos is not None:
            resized.thumbnail(self.max_image_size, resample=lanczos)
        else:
            resized.thumbnail(self.max_image_size)
        return resized

    def _save_animation(self, suffix: str, format_name: str, **save_kwargs) -> bool:
        output_path = self.output_dir / f"{self.file_stem}{suffix}"
        first_frame = self.frames[0]
        remaining_frames = self.frames[1:]
        try:
            first_frame.save(
                output_path,
                format=format_name,
                save_all=True,
                append_images=remaining_frames,
                duration=120,
                loop=0,
                **save_kwargs,
            )
            return True
        except Exception as exc:
            self.last_reason = f"{format_name} save failed: {exc}"
            return False

    def _write_status_file(self) -> None:
        status_path = self.output_dir / f"{self.file_stem}_capture.txt"
        status_path.write_text(
            "\n".join(
                [
                    f"capture_attempts={self.capture_attempts}",
                    f"saved_frames={len(self.frames)}",
                    f"sample_stride={self.sample_stride}",
                    f"client_index={self.client_index}",
                    f"last_reason={self.last_reason or 'No frames captured'}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
