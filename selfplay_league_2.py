from collections import deque
import copy
import os
import sys
import time
import random
from typing import Callable
import signal
import threading
import numpy as np
import torch
from jpype.types import JArray, JInt
from stable_baselines3.common.vec_env import VecVideoRecorder
from VecstatsMonitor import VecstatsMonitor

from microrts_space_transform import MicroRTSSpaceTransform
from gym_microrts.envs.microrts_vec_env import  MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai

from agent_model import Agent
import ppo_update
import league

class Selfplay_agent:
    def __init__(self, agent):
        self.agent = agent


def adjust_obs_selfplay(args, next_obs, is_new_env: bool = False):
    if is_new_env:
        # flippe jede zweite selfplay Umgebung (Spieler 1 -> Spieler 0)
        # da keine Unit eine Richtung bekommen hat müssen die Richtungen nicht angepasst werden
        if args.num_selfplay_envs > 1:
            if 2 < args.num_selfplay_envs:
                tmp = next_obs[1:args.num_selfplay_envs:2].flip(1, 2).contiguous().clone()
                next_obs[1:args.num_selfplay_envs:2] = tmp
            else:
                tmp = next_obs[1].flip(0, 1).contiguous().clone()
                next_obs[1] = tmp
            return

    if args.num_selfplay_envs > 1:
        # jede zweite selfplay Umgebung:
        if 2 < args.num_selfplay_envs:
            tmp = next_obs[1:args.num_selfplay_envs:2].flip(1, 2).contiguous().clone()
            # flip Observations (Spieler 1 -> Spieler 0)
            next_obs[1:args.num_selfplay_envs:2] = tmp

            # switch players in the observation (player 1 -> player 0) 
            # next_obs[1:args.num_selfplay_envs:2, :, :, 4:6:-1] = next_obs[1:args.num_selfplay_envs:2, :, :, 6:4] muss man nicht machen (sind schon gedreht), wenn doch --> auch wenn is_new_env=True, im else-Teil
            # next_obs[1:args.num_selfplay_envs:2, :, :, 59:66] = tmp[:, :, :, 66:73]
            # next_obs[1:args.num_selfplay_envs:2, :, :, 66:73] = tmp[:, :, :, 59:66]

            # rottate directions 180°
            next_obs[1:args.num_selfplay_envs:2, :, :, 22:26] = next_obs[1:args.num_selfplay_envs:2, :, :, 22:26].roll(shifts=2, dims=3)
            next_obs[1:args.num_selfplay_envs:2, :, :, 27:31] = next_obs[1:args.num_selfplay_envs:2, :, :, 27:31].roll(shifts=2, dims=3)
            next_obs[1:args.num_selfplay_envs:2, :, :, 32:36] = next_obs[1:args.num_selfplay_envs:2, :, :, 32:36].roll(shifts=2, dims=3)
            next_obs[1:args.num_selfplay_envs:2, :, :, 37:41] = next_obs[1:args.num_selfplay_envs:2, :, :, 37:41].roll(shifts=2, dims=3)
            # for i in range(0, 4):
            #     next_obs[1:args.num_selfplay_envs:2, :, :, 22 + 5 * i : 26 + 5 * i] = (
            #         next_obs[1:args.num_selfplay_envs:2, :, :, 22 + 5 * i : 26 + 5 * i].roll(shifts=2, dims=3)
            #     )
            next_obs[1:args.num_selfplay_envs:2, :, :, 50:54] = next_obs[1:args.num_selfplay_envs:2, :, :, 50:54].roll(shifts=2, dims=3)
        else:
            tmp = next_obs[1].flip(0, 1).contiguous().clone()
            next_obs[1] = tmp

            # switch players in the observation (player 1 -> player 0)
            # next_obs[1, :, :, 4] = tmp[:, :, 5]
            # next_obs[1, :, :, 5] = tmp[:, :, 4]
            # next_obs[1, :, :, 59:66] = tmp[:, :, 66:73]
            # next_obs[1, :, :, 66:73] = tmp[:, :, 59:66]

            # rottate directions 180° auch alle Richtungen, die nicht benutzt werden, werden geändert (benutze torch.roll(next_obs[...], shifts=2, dims=...))
            permutation = [21, 24, 25, 22, 23, 26, 29, 30, 27, 28, 31, 34, 35, 32, 33, 36, 39, 40, 37, 38]
            for i, p in enumerate(permutation):
                next_obs[1, :, :, i + 21] = tmp[:, :, p]
            permutation = [49, 52, 53, 50, 51]
            for i, p in enumerate(permutation):
                next_obs[1, :, :, i + 49] = tmp[:, :, p]


def adjust_action_selfplay(args, valid_actions: np.ndarray, valid_actions_counts: np.ndarray):
    if args.num_selfplay_envs > 1:
        # Position anpassen
        index = 0
        for j, i in enumerate(valid_actions_counts):
            if j % 2 == 1 and j < args.num_selfplay_envs:
                valid_actions[index:index + i, 0] = np.abs(valid_actions[index:index + i, 0] - 255)
                valid_actions[index:index + i, 2:6] = (valid_actions[index:index + i, 2:6] + 2) % 4
                valid_actions[index:index + i, 7] = np.abs(valid_actions[index:index + i, 7] - 48)
            index += i

            # real_action[i, :, 0] = torch.tensor(range(255, -1, -1)).to(device)
                # TO DO (selfplay): wird die Arrayposition der Spielpositionen vorausgesetzt? (muss es aufsteigend sortiert sein?) (wenn nicht --> unten entfernen)
            # real_action[1:args.num_selfplay_envs:2] = real_action[1:args.num_selfplay_envs:2].flip(1)

                # Richtungen anpassen (move direction, harvest direction, return direction, produce direction)
            # real_action[1:args.num_selfplay_envs:2, :, 2:6] = (real_action[1:args.num_selfplay_envs:2, :, 2:6] + 2) % 4
                # relative attack position anpassen (nur für a_r = 7)
            #real_action[1:args.num_selfplay_envs:2, :, 7] = torch.abs(real_action[1:args.num_selfplay_envs:2, :, 7] - 48)


def _resolve_checkpoint_path(model_path: str) -> str:
    if model_path.endswith(".pt"):
        checkpoint_path = model_path
    else:
        checkpoint_path = f"models/{model_path}/agent.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    return checkpoint_path

def load_agent_from_checkpoint(model_path: str, device: torch.device, agent) -> Agent:
    exploiter_ckpt_path = _resolve_checkpoint_path(model_path)
    
    agent.agent.load_state_dict(torch.load(exploiter_ckpt_path, map_location=device, weights_only=True))
    agent._initial_weights = {k: v.detach().clone() for k, v in agent.agent.state_dict().items()}


def render_all_envs(env_transform):
    try:
        if env_transform is None:
            return
        if hasattr(env_transform, "interface") and hasattr(env_transform.interface, "vec_client"):
            vec_client = env_transform.interface.vec_client
            if hasattr(vec_client, "selfPlayClients") and len(vec_client.selfPlayClients) > 0:
                for client in vec_client.selfPlayClients:
                    try:
                        client.render(False)
                    except Exception:
                        pass
            if hasattr(vec_client, "clients") and len(vec_client.clients) > 0:
                for client in vec_client.clients:
                    try:
                        client.render(False)
                    except Exception:
                        pass
            return
    except Exception:
        pass

    try:
        env_transform.render()
    except Exception:
        pass

# TODO (debugging): debugging function
def break_on_stdout(trigger="Issuing a non legal action", include_stderr: bool = True):
    """Pipe stdout (and optionally stderr) through a watcher and drop into pdb when trigger text appears."""
    trigger_bytes = trigger.encode()
    orig_stdout_fd = os.dup(sys.stdout.fileno())
    orig_stderr_fd = os.dup(sys.stderr.fileno()) if include_stderr else None
    read_fd, write_fd = os.pipe()

    def _sigusr1(_sig, _frame):
        breakpoint()

    signal.signal(signal.SIGUSR1, _sigusr1)

    def _reader():
        buf = b""
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            os.write(orig_stdout_fd, chunk)
            buf = (buf + chunk)[-8192:]
            if trigger_bytes in buf:
                os.kill(os.getpid(), signal.SIGUSR1)

    threading.Thread(target=_reader, daemon=True).start()
    sys.stdout.flush()
    if include_stderr:
        sys.stderr.flush()
    os.dup2(write_fd, sys.stdout.fileno())
    if include_stderr:
        os.dup2(write_fd, sys.stderr.fileno())

    def cleanup():
        try:
            sys.stdout.flush()
            if include_stderr:
                sys.stderr.flush()
            os.dup2(orig_stdout_fd, sys.stdout.fileno())
            if include_stderr and orig_stderr_fd is not None:
                os.dup2(orig_stderr_fd, sys.stderr.fileno())
        finally:
            for fd in (orig_stdout_fd, orig_stderr_fd if include_stderr else None, read_fd, write_fd):
                if fd is None:
                    continue
                try:
                    os.close(fd)
                except OSError:
                    pass

    return cleanup



class LeagueTrainer:
    def __init__(
        self,
        agent,
        supervised_agent,
        other_historicals,
        envs,
        sp_envs,
        args,
        writer,
        device: torch.device,
        experiment_name: str,
        get_scalar_features: Callable
    ):
        self.agent = agent
        self.supervised_agent = supervised_agent
        self.other_historicals = other_historicals
        self.envs = envs
        self.sp_envs = sp_envs
        self.args = args
        self.writer = writer
        self.device = device
        self.experiment_name = experiment_name
        self.get_scalar_features = get_scalar_features
        self.active_league_agents = []
        self.league_agent = Selfplay_agent(agent)
        self.league_supervised_agent = Selfplay_agent(supervised_agent)
        self.indices = torch.tensor(range(args.num_selfplay_envs, args.num_envs), dtype=torch.long, device=device)
        self.indices = torch.cat(
            (torch.tensor(range(0, args.num_selfplay_envs, 2), dtype=torch.long, device=device), self.indices)
        )
        self.hist_reward: int = 0

        self.indices_per_exploiter = {}
        self.b_indices_per_exploiter = {}
        self.main_indices = slice(0, 0, 1)
        self.b_main_indices = slice(0, 0, 1)
        self.main_indices_count = 0

        

    def _add_unit_bonus_to_score(
        self,
        score_tensor: torch.Tensor,
        own_unit_counts: torch.Tensor,
        opp_unit_counts: torch.Tensor,
        unit_bonus_weights: torch.Tensor,
    ) -> torch.Tensor:
        unit_bonus = (own_unit_counts - opp_unit_counts) * unit_bonus_weights
        return score_tensor + unit_bonus.sum(dim=1)

    def _unit_bonus_max(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [
                self.args.unit_bonus_max_worker,
                self.args.unit_bonus_max_light,
                self.args.unit_bonus_max_heavy,
                self.args.unit_bonus_max_ranged,
            ],
            device=device,
            dtype=torch.float,
        )

    def _sample_unit_bonus_distr(self, shape, device: torch.device) -> torch.Tensor:
        return torch.rand(shape, device=device) * self._unit_bonus_max(device)


    def train(self):
        args = self.args
        num_done_botgames = 0
        num_done_selfplaygames = 0
        last_logged_selfplay_games = 0
        agent: Agent = self.agent
        envs = self.envs
        sp_envs = self.sp_envs
        writer = self.writer
        device = self.device
        supervised_agent = self.supervised_agent or Agent(agent.action_plane_nvec, agent.device, initial_weights=agent.state_dict(), unit_exploiters=args.unit_exploiters)
        last_bot_env_change = 0

        if args.render:
            if args.render_all:
                from ppo import Rendering

        if args.num_envs == 0:
            raise ValueError("league training requires at least one environment")
        
        league_instance, self.active_league_agents = league.initialize_league(args, device, agent, other_initial_agents=self.other_historicals)

        if not args.cur_main_exploiter_path is None:
            for ag, _ in agent.get_unique_agents(self.active_league_agents, output_league_agents=True).items():
                if isinstance(ag, (league.MainExploiter, league.LeagueExploiter)):
                    load_agent_from_checkpoint(args.cur_main_exploiter_path, device, ag)

        if not args.cur_main_path is None:
            for ag, _ in agent.get_unique_agents(self.active_league_agents, output_league_agents=True).items():
                if isinstance(ag, (league.MainPlayer)):
                    load_agent_from_checkpoint(args.cur_main_path, device, ag)


        if self.args.Unit_reward_per_exploiter:
            assert args.unit_exploiters, "Unit_reward_per_exploiter requires unit_exploiters to be true"
            self.unit_bonus_distr = torch.zeros((args.num_envs, 4), device=device)
            for ag, indices in agent.get_unique_agents(self.active_league_agents, output_league_agents=True).items():
                bonus = ag.unit_bonus_distr
                if bonus is None:
                    bonus = torch.zeros(4, device=device)
                elif bonus.device != device:
                    bonus = bonus
                self.unit_bonus_distr[indices] = bonus

        elif args.unit_exploiters: # initialize unit bonus distr even, if Unit_reward_per_exploiter is true
            self.unit_bonus_distr = self._sample_unit_bonus_distr((args.num_envs, 4), device)
            for ag, indices in agent.get_unique_agents(self.active_league_agents, output_league_agents=True).items():
                if isinstance(ag, (league.MainPlayer)) or (isinstance(ag, league.Historical) and isinstance(ag.parent, league.MainPlayer)):
                    self.unit_bonus_distr[indices] = torch.zeros((len(indices), 4), device=device)
        else:
            self.unit_bonus_distr = None
            
                    
        


        # updates indices for main / exploiter agents in self.active_league_agents
        self._refresh_main_indices(args)
        self._refresh_exploiter_indices(args)


        optimizer = torch.optim.Adam(agent.parameters(), lr=args.PPO_learning_rate, eps=1e-5)
        if args.anneal_lr:
            lr_fn = lambda frac: frac * args.PPO_learning_rate  # noqa: E731
            exploiter_lr_fn = lambda frac: frac * args.exploiter_PPO_learning_rate  # noqa: E731
        else:
            lr_fn = None
            exploiter_lr_fn = None

        if args.dbg_non_legal_action:
            cleanup_break = break_on_stdout("Issuing a non legal action")

        mapsize = 16 * 16
        action_space_shape = (mapsize, envs.action_plane_space.shape[0])
        invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum() + 1)

        sp_inds = slice(0, args.num_selfplay_envs)
        bot_inds = slice(args.num_selfplay_envs, args.num_envs)

        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)

        rewards_attack = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_winloss = torch.zeros((args.num_steps, args.num_envs)).to(device)
        delta_rewards_score = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        args.global_step = 0
        start_time = time.time()

        next_obs_np, _, bot_res = envs.reset()
        bot_next_obs = torch.Tensor(next_obs_np).to(device)

        if args.num_selfplay_envs > 0:
            next_obs_np, _, sp_res = sp_envs.reset()
            sp_next_obs = torch.Tensor(next_obs_np).to(device)
            adjust_obs_selfplay(args, sp_next_obs, is_new_env=True)
        else:
            sp_res = []
            sp_next_obs = torch.zeros((0,) + envs.single_observation_space.shape, device=device)

        next_done = torch.zeros(args.num_envs).to(device)
        scalar_features = torch.zeros((args.num_steps, args.num_envs, 11)).to(device)
        z_features = torch.zeros((args.num_steps, args.num_envs, 8), dtype=torch.long).to(device)
        last_sp_scorerew = torch.zeros(args.num_selfplay_envs, device=device)
        last_bot_scorerew = torch.zeros(args.num_bot_envs, device=device)
        delta_score_sums = torch.zeros(args.num_envs, device=device)


        num_updates = args.total_timesteps // args.batch_size

        bot_position_indices = (
            torch.arange(mapsize, device=device, dtype=torch.int64).unsqueeze(0).repeat(args.num_bot_envs, 1).unsqueeze(2)
        )
        sp_position_indices = (
            torch.arange(mapsize, device=device, dtype=torch.int64).unsqueeze(0).repeat(args.num_selfplay_envs, 1).unsqueeze(2)
        )

        print("League PPO training started")

        unique_agents = agent.get_unique_agents(self.active_league_agents)
        
        for update in range(1, num_updates + 1):
            for agent_type, agent_idx in unique_agents:
                agent_args = copy.deepcopy(args)
                agent_args.num_envs = len(agent_idx)
                if isinstance(agent_type, league.MainPlayer):
                    agent_args.num_main_envs = len(agent_idx)
                    agent_args.num_main_exploiters = 0
                    agent_args.num_league_exploiters = 0
                    agent_args.num_bot_envs = args.num_bot_envs

                    # TODO: implement change_envs function
                    agent_type_indices, main_indices_count, main_indices, b_main_indices, indices_per_exploiter, b_indices_per_exploiter = self.change_envs(len(agent_idx), action_space_shape, invalid_action_shape, sp_inds, bot_inds, obs, actions, logprobs, invalid_action_masks, rewards_attack, rewards_winloss, delta_rewards_score, dones, values, bot_res, bot_next_obs, sp_res, sp_next_obs, next_done, scalar_features, z_features, last_sp_scorerew, last_bot_scorerew, bot_position_indices, sp_position_indices, self.indices)

                elif isinstance(agent_type, league.MainExploiter):
                    agent_args.num_main_envs = 0
                    agent_args.num_main_exploiters = len(agent_idx)
                    agent_args.num_league_exploiters = 0
                    agent_args.num_bot_envs = args.num_bot_envs_per_main_exploiter
                    agent_type_indices, main_indices_count, main_indices, b_main_indices, indices_per_exploiter, b_indices_per_exploiter = self.change_envs(len(agent_idx), action_space_shape, invalid_action_shape, sp_inds, bot_inds, obs, actions, logprobs, invalid_action_masks, rewards_attack, rewards_winloss, delta_rewards_score, dones, values, bot_res, bot_next_obs, sp_res, sp_next_obs, next_done, scalar_features, z_features, last_sp_scorerew, last_bot_scorerew, bot_position_indices, sp_position_indices, self.indices)


                elif isinstance(agent_type, league.LeagueExploiter):
                    agent_args.num_main_envs = 0
                    agent_args.num_main_exploiters = 0
                    agent_args.num_league_exploiters = len(agent_idx)
                    agent_args.num_bot_envs = 0
                    agent_type_indices, main_indices_count, main_indices, b_main_indices, indices_per_exploiter, b_indices_per_exploiter = self.change_envs(len(agent_idx), action_space_shape, invalid_action_shape, sp_inds, bot_inds, obs, actions, logprobs, invalid_action_masks, rewards_attack, rewards_winloss, delta_rewards_score, dones, values, bot_res, bot_next_obs, sp_res, sp_next_obs, next_done, scalar_features, z_features, last_sp_scorerew, last_bot_scorerew, bot_position_indices, sp_position_indices, self.indices)

                


                active_league_agent_update_args = {
                    "update": update,
                    "args": agent_args,
                    "num_done_botgames": num_done_botgames,
                    "num_done_selfplaygames": num_done_selfplaygames,
                    "last_logged_selfplay_games": last_logged_selfplay_games,
                    "last_bot_env_change": last_bot_env_change,
                    "delta_score_sums": delta_score_sums,
                    "agent": agent,
                    "envs": envs,
                    "sp_envs": sp_envs,
                    "writer": writer,
                    "device": device,
                    "supervised_agent": supervised_agent,
                    "league_instance": league_instance,
                    "optimizer": optimizer,
                    "lr_fn": lr_fn,
                    "exploiter_lr_fn": exploiter_lr_fn,
                    "action_space_shape": action_space_shape,
                    "invalid_action_shap": invalid_action_shape,
                    "sp_inds": sp_inds,
                    "bot_inds": bot_inds,
                    "obs": obs,
                    "actions": actions,
                    "logprobs": logprobs,
                    "invalid_action_mask": invalid_action_masks,
                    "rewards_attack": rewards_attack,
                    "rewards_winloss": rewards_winloss,
                    "delta_rewards_score": delta_rewards_score,
                    "dones": dones,
                    "values": values,
                    "start_time": start_time,
                    "bot_res": bot_res,
                    "bot_next_obs": bot_next_obs,
                    "sp_res": sp_res,
                    "sp_next_obs": sp_next_obs,
                    "next_done": next_done,
                    "scalar_features": scalar_features,
                    "z_features": z_features,
                    "last_sp_scorerew": last_sp_scorerew,
                    "last_bot_scorerew": last_bot_scorerew,
                    "num_updates": num_updates,
                    "bot_position_indice": bot_position_indices,
                    "sp_position_indices": sp_position_indices,
                    "active_league_agents": self.active_league_agents[agent_idx],
                    "unit_bonus_distr": self.unit_bonus_distr[agent_idx],
                    "hist_reward": self.hist_reward[agent_idx],
                    "indices": agent_type_indices,
                    "main_indices_count": main_indices_count,
                    "main_indices": main_indices,
                    "b_main_indices": b_main_indices,
                    "indices_per_exploiter": indices_per_exploiter,
                    "b_indices_per_exploiter": b_indices_per_exploiter,
                    "experiment_name": self.experiment_name,
                }

                self.unit_bonus_distr[agent_idx] = active_league_agent_update_args["unit_bonus_distr"]

                active_league_agent = self.active_league_agents[agent_idx]
                self.update(Rendering, active_league_agent_update_args)

        if args.dbg_non_legal_action and cleanup_break:
            cleanup_break()

    def update(self, Rendering, active_league_agent_update_args):
        update = active_league_agent_update_args["update"]
        args = active_league_agent_update_args["args"]
        num_done_botgames = active_league_agent_update_args["num_done_botgames"]
        num_done_selfplaygames = active_league_agent_update_args["num_done_selfplaygames"]
        agent = active_league_agent_update_args["agent"]
        envs = active_league_agent_update_args["envs"]
        sp_envs = active_league_agent_update_args["sp_envs"]
        writer = active_league_agent_update_args["writer"]
        device = active_league_agent_update_args["device"]
        supervised_agent = active_league_agent_update_args["supervised_agent"]
        league_instance = active_league_agent_update_args["league_instance"]
        optimizer = active_league_agent_update_args["optimizer"]
        lr_fn = active_league_agent_update_args["lr_fn"]
        exploiter_lr_fn = active_league_agent_update_args["exploiter_lr_fn"]
        action_space_shape = active_league_agent_update_args["action_space_shape"]
        invalid_action_shape = active_league_agent_update_args["invalid_action_shap"]
        sp_inds = active_league_agent_update_args["sp_inds"]
        bot_inds = active_league_agent_update_args["bot_inds"]
        obs = active_league_agent_update_args["obs"]
        actions = active_league_agent_update_args["actions"]
        logprobs = active_league_agent_update_args["logprobs"]
        invalid_action_masks = active_league_agent_update_args["invalid_action_mask"]
        rewards_attack = active_league_agent_update_args["rewards_attack"]
        rewards_winloss = active_league_agent_update_args["rewards_winloss"]
        delta_rewards_score = active_league_agent_update_args["delta_rewards_score"]
        dones = active_league_agent_update_args["dones"]
        values = active_league_agent_update_args["values"]
        start_time = active_league_agent_update_args["start_time"]
        bot_res = active_league_agent_update_args["bot_res"]
        bot_next_obs = active_league_agent_update_args["bot_next_obs"]
        sp_res = active_league_agent_update_args["sp_res"]
        sp_next_obs = active_league_agent_update_args["sp_next_obs"]
        next_done = active_league_agent_update_args["next_done"]
        scalar_features = active_league_agent_update_args["scalar_features"]
        z_features = active_league_agent_update_args["z_features"]
        last_sp_scorerew = active_league_agent_update_args["last_sp_scorerew"]
        last_bot_scorerew = active_league_agent_update_args["last_bot_scorerew"]
        num_updates = active_league_agent_update_args["num_updates"]
        bot_position_indices = active_league_agent_update_args["bot_position_indice"]
        sp_position_indices = active_league_agent_update_args["sp_position_indices"]
        last_logged_selfplay_games = active_league_agent_update_args["last_logged_selfplay_games"]
        last_bot_env_change = active_league_agent_update_args["last_bot_env_change"]
        delta_score_sums = active_league_agent_update_args["delta_score_sums"]
        active_league_agents = active_league_agent_update_args["active_league_agents"]
        unit_bonus_distr = active_league_agent_update_args["unit_bonus_distr"]
        hist_reward = active_league_agent_update_args["hist_reward"]
        indices = active_league_agent_update_args["indices"]
        main_indices_count = active_league_agent_update_args["main_indices_count"]
        main_indices = active_league_agent_update_args["main_indices"]
        b_main_indices = active_league_agent_update_args["b_main_indices"]
        indices_per_exploiter = active_league_agent_update_args["indices_per_exploiter"]
        b_indices_per_exploiter = active_league_agent_update_args["b_indices_per_exploiter"]
        experiment_name = active_league_agent_update_args["experiment_name"]

        skip_update_count = 0
        should_log_every_20_updates = (update % 20 == 0)
        if args.dbg_seed:
            self._seed_for_update(update, args.seed)

        if lr_fn is not None:
            main_frac = 1.0 - (update - 1.0) / num_updates
            if main_frac < 0.0:
                main_frac = 0.0
            lrnow = lr_fn(main_frac)
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            if args.render:
                if args.render_all:
                    # only workes for 1 at a time
                    # Rendering.render_all_envs(envs)
                    if sp_envs is not None:
                        Rendering.render_all_envs(sp_envs)
                    elif envs is not None:
                        render_all_envs(envs)
                else:
                    if envs is not None:
                        envs.render("human")
                    if sp_envs is not None:
                        sp_envs.render("human")
                    
            args.global_step += (args.num_selfplay_envs // 2) + args.num_bot_envs
            obs[step, bot_inds] = bot_next_obs
            obs[step, sp_inds] = sp_next_obs
            next_obs = obs[step]
            res = sp_res + bot_res
            scalar_features[step] = self.get_scalar_features(next_obs, res, args.num_envs).to(device)
            dones[step] = next_done

            with torch.no_grad():
                # unique_agents = agent.get_unique_agents(active_league_agents, selfplay_only=True)
                unique_agents = agent.get_unique_agents(active_league_agents)
                sp_only_unique_agents = agent.get_unique_agents(active_league_agents[:args.num_selfplay_envs])

                z_features[step] = agent.selfplay_get_z_encoded_features(
                    args=args,
                    device=device,
                    z_features=z_features,
                    next_obs=next_obs,
                    step=step,
                    unique_agents=unique_agents
                )

                # debugging
                # for i in range(args.num_envs):
                #     with torch.no_grad():
                #         # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                #         old_zFeatures[step][i] = agent.z_encoder(obs[step][i].view(-1))
                # assert(torch.all(old_zFeatures == zFeatures))

                # critic(forward(...))
                # # values[step] = agent.get_value(obs[step, self.indices], scalar_features[step, self.indices], z_features[step, self.indices]).flatten()
                # values[step] = agent.get_value(obs[step], scalar_features[step], z_features[step]).flatten()
                values[step] = agent.selfplay_and_Bot_get_value(
                    obs[step],
                    scalar_features[step],
                    z_features[step],
                    num_selfplay_envs=args.num_selfplay_envs,
                    num_envs=args.num_envs,
                    unique_agents=unique_agents,
                    only_player_0=True,
                    unit_bonus_distr=unit_bonus_distr
                ).flatten()

                # debug:
                # a = (Variables)
                # import pickle, os
                # with open(f"dump_var.pkl", "wb") as f:
                #     pickle.dump(a , f)

                # import pickle, glob
                # files = sorted(glob.glob("dump_*.pkl"))
                # c = pickle.load(open(files[0], "rb"))
                # arr = []
                # for a, b in zip(c, (Variables)):
                #     if isinstance(a == b, bool):
                #         arr.append((a == b))
                #     else:
                #         arr.append(torch.all(a == b).item())
                

                # self.check_values(scalar_features, z_features, values, agent, step, obs=obs[step], flatten=True)

                # gesamplete action (aus Verteilung der Logits) (24, 256, 7),
                # actor(forward(...)), invalid_action_masks
                # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                actions[step, bot_inds], logprobs[step, bot_inds], _, invalid_action_masks[step, bot_inds] = agent.get_action(
                    obs[step, bot_inds],
                    scalar_features[step, args.num_selfplay_envs:],
                    z_features[step, args.num_selfplay_envs:],
                    envs=envs,
                    unit_bonus_distr=unit_bonus_distr[args.num_selfplay_envs:] if unit_bonus_distr is not None else None
                )

                if args.num_selfplay_envs > 0:
                    actions[step, sp_inds], logprobs[step, sp_inds], _, invalid_action_masks[step, sp_inds] = agent.selfplay_get_action(
                        obs[step, sp_inds],
                        scalar_features[step, :args.num_selfplay_envs],
                        z_features[step, :args.num_selfplay_envs],
                        num_selfplay_envs=args.num_selfplay_envs,
                        num_envs=args.num_selfplay_envs,
                        envs=sp_envs,
                        active_league_agents=active_league_agents,
                        unique_agents=sp_only_unique_agents,
                        dbg_deterministic_actions=args.dbg_deterministic_actions,
                        unit_bonus_distr=unit_bonus_distr[:args.num_selfplay_envs] if unit_bonus_distr is not None else None
                    )

            # Die Grid-Position zu jedem Action hinzugefügt (24, 256, 8)
            bot_real_action = torch.cat([bot_position_indices, actions[step, bot_inds]], dim=2).cpu().numpy()
            sp_real_action = torch.cat([sp_position_indices, actions[step, sp_inds]], dim=2).cpu().numpy()
            # print("real_action shape:", real_action.shape)
            # print("Grid-Position:", [real_action[0][i][0].item() for i in
            # range(10)]) # -> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


            # =============
            # invalid_action_masks angewandt
            # =============


            # Debug Beispiel
            # valid_actions = np.array([np.array([34.0, 0.0, 1.0, 3.0, 1.0, 2.0, 3.0, 21.0]),
            #                            np.array([238.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            #                            np.array([34.0, 0.0, 2.0, 0.0, 0.0, 2.0, 3.0, 24.0])])
            # valid_actions_counts = [1, 1, 1]
            bot_valid_actions = bot_real_action[invalid_action_masks[step, bot_inds][:, :, 0].bool().cpu().numpy()]
            bot_valid_counts = invalid_action_masks[step, bot_inds][:, :, 0].sum(1).long().cpu().numpy()
            sp_valid_actions = sp_real_action[invalid_action_masks[step, sp_inds][:, :, 0].bool().cpu().numpy()]
            sp_valid_counts = invalid_action_masks[step, sp_inds][:, :, 0].sum(1).long().cpu().numpy()

            # adjust actions for selfplay environments (player 1 -> player 0)
            # TODO (optimize): nur die Indizes anpassen, die man anpassen muss (bei type move nicht harvest, return, produce, attack anpassen)
            adjust_action_selfplay(args, sp_valid_actions, sp_valid_counts)

            '''
            valid_actions:
            [[Pos, Type, move direction, harvest direction, return (recource) direction, produce direction, produce type, relative attack position],
             [Spiel0 (Spieler1)],
             [Spiel1 (Spieler0)]]
            Pos: 0-255 (16*16) links oben nach rechts unten (obenecke = 0)
            Type: 0: NOP, 1: Move, 2: Harvest, 3: Return, 4: Produce (Produce direction + Produce type), 5: Attack (wenn z.B.: move direction = 1, aber Type = 2 --> move direction wird ignoriert)
            direction: 0: North, 1: East, 2: South, 3: West
            produce type: 0: (light), 1: (Ranged), 2: (Baracks / Heavy), 3: (Worker) (je nach Unit unterschiedlich)
            relative attack position: 0-255 (16*16) links oben nach rechts unten (obenecke = 0) wo angegriffen wird
            '''

            bot_java_valid_actions = []
            bot_valid_index = 0
            for count in bot_valid_counts:
                java_env_action = []
                for _ in range(count):
                    java_env_action.append(JArray(JInt)(bot_valid_actions[bot_valid_index]))
                    bot_valid_index += 1
                bot_java_valid_actions.append(JArray(JArray(JInt))(java_env_action))
            bot_java_valid_actions = JArray(JArray(JArray(JInt)))(bot_java_valid_actions)

            sp_java_valid_actions = []
            sp_valid_index = 0
            for count in sp_valid_counts:
                java_env_action = []
                for _ in range(count):
                    java_env_action.append(JArray(JInt)(sp_valid_actions[sp_valid_index]))
                    sp_valid_index += 1
                sp_java_valid_actions.append(JArray(JArray(JInt))(java_env_action))
            sp_java_valid_actions = JArray(JArray(JArray(JInt)))(sp_java_valid_actions)
            # java_valid_actions.shape: (Envs, num_valid_actions_in_Env, valid_action (8)) (py_arr = np.array(java_valid_actions))
            # np_valid_actions = np.array(
            # [[np.array(list(inner), dtype=np.int32) for inner in outer]
            #  for outer in java_valid_actions],
            # dtype=object
            # )
            # =============

            # =============
            # Schritt in der Umgebung mit der in get_action gesampleten Action
            # =============

            bot_next_obs, _, bot_attackrew, bot_winlossrew, bot_scorerew, bot_ds, bot_infos, bot_res = envs.step(bot_java_valid_actions)
            bot_next_obs = torch.Tensor(envs._from_microrts_obs(bot_next_obs)).to(device) # next_obs zu Tensor mit shape (24, 16, 16, 73) (von (24, X))
            if args.num_selfplay_envs > 0:
                sp_next_obs, _, sp_attackrew, sp_winlossrew, sp_scorerew, sp_ds, sp_infos, sp_res = sp_envs.step(sp_java_valid_actions)
                sp_next_obs = torch.Tensor(sp_envs._from_microrts_obs(sp_next_obs)).to(device)
                
                adjust_obs_selfplay(args, sp_next_obs)
            else:
                sp_attackrew = np.array([], dtype=np.float32)
                sp_winlossrew = np.array([], dtype=np.float32)
                sp_scorerew = np.array([], dtype=np.float32)
                sp_ds = np.array([], dtype=np.bool_)
                sp_infos = []
                sp_res = []

            if args.dbg_exploiter_update:
                for i in range(0, args.num_selfplay_envs, 2):
                    if not torch.all(sp_next_obs[0] == sp_next_obs[i]):
                        breakpoint()

            '''winloss = min(0.01, 6.72222222e-9 * args.global_step)
            densereward = max(0, 0.8 + (-4.44444444e-9 * args.global_step))
            if args.global_step < 100000000:
                scorew = 0.19 + 1.754e-8 * args.global_step
            else:
                scorew = 0.5 - 1.33e-8 * args.global_step'''


            # densereward = 0
            winloss = 10
            attack = args.attack_reward_weight
            # =============

            # update rewards
            # rewards_dense[step] = torch.Tensor(denserew* densereward).to(device)

            ### Debugging (scorerews always == 0)
            # Breakpoint if values change
            # if not np.array_equal(scorerews, _last_scorerews):
            #     breakpoint()
            # _last_scorerews = np.copy(scorerews)


            attack_tensor = torch.as_tensor(np.concatenate([sp_attackrew, bot_attackrew]), device=device, dtype=torch.float)
            if args.dyn_attack_reward > 0:
                # done_tensor = torch.as_tensor(ds, device=device, dtype=torch.bool)
                # draw_mask = (winloss_tensor == 0) & done_tensor
                sc = scalar_features[step]
                # own_recources = sc[:, 0]
                # opp_recources = sc[:, 1]
                own_light = sc[:, 4]
                own_heavy = sc[:, 5]
                own_ranged = sc[:, 6]
                opp_light = sc[:, 8]
                opp_heavy = sc[:, 9]
                opp_ranged = sc[:, 10]
                # strength_ratio = (own_heavy + 0.5 * (own_light + own_ranged) + own_recources * 0.3) / torch.clip(opp_heavy + 0.5 * (opp_light + opp_ranged) + opp_recources * 0.3, min=0.00001)
                strength_ratio = (
                    (own_heavy + 0.5 * (own_light + own_ranged))
                    / torch.clip(opp_heavy + 0.5 * (opp_light + opp_ranged), min=0.00001)
                ) ** 1.5
                # less_draw_scaled = torch.clip(args.dyn_attack_reward * strength_ratio, max=0.1)
                # rewards_winloss[step] = winloss_tensor * winloss - less_draw_scaled * draw_mask.float()
                attack_scaled = torch.clip(args.dyn_attack_reward * strength_ratio, max=1.5, min=0.5)
                rewards_attack[step] = attack_tensor + attack * attack_scaled * (attack_tensor > 0).float()
            else:
                rewards_attack[step] = attack_tensor * attack

            rewards_winloss[step] = torch.Tensor(np.concatenate([sp_winlossrew, bot_winlossrew])).to(device)
            sp_score_tensor = torch.as_tensor(sp_scorerew, device=device, dtype=torch.float)
            bot_score_tensor = torch.as_tensor(bot_scorerew, device=device, dtype=torch.float)
            if args.unit_exploiters:
                sc = scalar_features[step]
                sp_sc = sc[:args.num_selfplay_envs]
                bot_sc = sc[args.num_selfplay_envs:]
                # rewards for opponent are wrong but not used
                sp_score_tensor = self._add_unit_bonus_to_score(
                    sp_score_tensor,
                    sp_sc[:, 3:7],
                    sp_sc[:, 7:11],
                    unit_bonus_distr[:args.num_selfplay_envs],
                )
                bot_score_tensor = self._add_unit_bonus_to_score(
                    bot_score_tensor,
                    bot_sc[:, 3:7],
                    bot_sc[:, 7:11],
                    unit_bonus_distr[args.num_selfplay_envs:],
                )


            sp_score_delta = sp_score_tensor - last_sp_scorerew
            bot_score_delta = bot_score_tensor - last_bot_scorerew
            score_delta = torch.tanh(1.5 * args.rewardscore * torch.cat([sp_score_delta, bot_score_delta]))
            sp_done_tensor = torch.as_tensor(sp_ds, device=device, dtype=torch.bool)
            bot_done_tensor = torch.as_tensor(bot_ds, device=device, dtype=torch.bool)
            last_sp_scorerew = torch.where(sp_done_tensor, torch.zeros_like(sp_score_tensor), sp_score_tensor) # if done: 0 else: current score
            last_bot_scorerew = torch.where(bot_done_tensor, torch.zeros_like(bot_score_tensor), bot_score_tensor)
            delta_rewards_score[step] = score_delta
            delta_score_sums += score_delta
            next_done = torch.Tensor(np.concatenate([sp_ds, bot_ds])).to(device)

            # =============
            # Logging PPO training
            # =============
            infos =  sp_infos + bot_infos
            if np.any(['episode' in info.keys() for info in infos]):
                if not hasattr(writer, "recent_bot_winloss"):
                            writer.recent_bot_winloss = deque([0.0] * 10, maxlen=200)
                if not hasattr(writer, "recent_selfplay_winloss"):
                            writer.recent_selfplay_winloss = deque([0.0] * 10, maxlen=200)

                where_done = torch.where(next_done)
                done_mask = next_done.bool()
                for done_idx in where_done[0]:
                    delta_score_sum = delta_score_sums[done_idx].item()
                    infos[done_idx]["delta_score_sum"] = delta_score_sum
                    infos[done_idx]["delta_score_sum_weighted"] = delta_score_sum
                    done_agent = active_league_agents[done_idx]

                    # dyn_winloss = winloss
                    game_length = infos[done_idx]["episode"]["l"]
                    # dyn_winloss = winloss * (-0.00013 * game_length + 1.16)  # ca. 0.9 bei 2000 und 1.1 bei 500
                    if done_idx > args.num_selfplay_envs - 1 or done_idx % 2 == 0:
                        done_agent.agent.steps = done_agent.agent.get_steps() + infos[done_idx]["episode"]["l"]

                        if isinstance(done_agent, league.MainPlayer):
                            # game_length = infos[done_idx]["episode"]["l"]
                            # dyn_winloss = winloss * (-0.00013 * game_length + 1.16)  # ca. 0.9 bei 2000 und 1.1 bei 500
                            league.log_general_main_results(writer, args.global_step, infos, winloss, game_length, attack, done_idx, hist_reward, done_agent)
                            
                    if done_idx > args.num_selfplay_envs - 1:
                        league.log_bot_game_results(args, writer, infos, attack, done_idx, winloss, num_done_botgames, done_agent)
                        num_done_botgames += 1
                        last_bot_env_change += 1

                    elif done_idx % 2 == 0:
                        # update League match results
                        active_league_agents[done_idx + 1], last_logged_selfplay_games, old_opp = league_instance.handle_game_end(
                            args,
                            agent,
                            writer,
                            active_league_agents,
                            infos,
                            attack,
                            done_idx,
                            done_agent,
                            winloss,
                            hist_reward,
                            num_done_selfplaygames,
                            indices_per_exploiter,
                            last_logged_selfplay_games
                        )
                        num_done_selfplaygames += 1

                        if args.save_gpu_memory:
                            league.offload_historical_to_cpu(old_opp, active_agents=active_league_agents)

                # get new unit bonus distribution for next Game
                self.get_new_unit_bonus_distr(where_done[0], device)

                delta_score_sums = torch.where(done_mask, torch.zeros_like(delta_score_sums), delta_score_sums)
                    
            # =============
        # =========================


        

        

        
        # =========================
        # PPO update
        # =========================
        
        # unique_agents = agent.get_unique_agents(active_league_agents, selfplay_only=True)
        unique_agents = agent.get_unique_agents(active_league_agents)

        with torch.no_grad():
            next_scalar_features = self.get_scalar_features(next_obs, res, args.num_envs).to(device)
            next_z_features = agent.selfplay_get_z_encoded_features(
                args, device, z_features, next_obs, args.num_steps, unique_agents
            )
            

            # next_value = agent.get_value(next_obs, next_scalar_features, next_z_features).reshape(1, -1)
            next_value = agent.selfplay_and_Bot_get_value(
                next_obs,
                next_scalar_features,
                next_z_features,
                num_selfplay_envs=args.num_selfplay_envs,
                num_envs=args.num_envs,
                unique_agents=unique_agents,
                only_player_0=True,
                unit_bonus_distr=unit_bonus_distr
            ).reshape(1, -1)

            # self.check_values(
            #     scalar_features, z_features, next_value, 
            #     agent, step, 
            #     next_scalar_features=next_scalar_features, 
            #     next_z_features=next_z_features, 
            #     next_obs=next_obs, 
            #     flatten=False
            #     )
            
            rewards_winloss = rewards_winloss * winloss

            # dont calculate GAE for Player 1 Environments
            b_next_value = next_value[:, indices]
            b_values = values[:, indices]
            b_rewards_attack = rewards_attack[:, indices]
            b_rewards_winloss = rewards_winloss[:, indices]
            b_delta_rewards_score = delta_rewards_score[:, indices]
            b_dones = dones[:, indices]
            b_next_done = next_done[indices]

            # (returns, advantages werden für exploiters weitergegeben, deshalb muss man sie hier auch berechnen oder unten anpassen)
            # oder 2 Variablen jeweils speichern. Hier kann man auch nur die obs, ... zusammenstellen, die exploiters brauchen (spart Speicher)
            # Debug helper: skip the entire PPO update phase (no GAE, no grads, no loss logging)
            b_advantages, b_returns = ppo_update.gae(args, device, b_next_value, b_values, b_rewards_attack, b_rewards_winloss, b_delta_rewards_score, b_dones, b_next_done)



        # flatten the batch
        # args.num_steps, args.num_envs Dimensionen vereinigen  (shape (steps*envs, 11))
        # (ScFeatures für jeden Step, Environment sortiert Step, dann nach Environments)
        # b_Sc = scalar_features[:, self.indices].reshape(-1, scalar_features.shape[-1])
        # args.num_steps, args.num_envs Dimensionen vereinigen  (shape (steps*envs, 8))
        # (zFeatures für jeden Step, Environment sortiert Step, dann nach Environments)
        # b_z = z_features[:, self.indices].reshape(-1, z_features.shape[-1])
        # dasselbe mit obs                                      (shape (steps*envs, 16, 16, 73)
        # b_obs = obs[:, self.indices].reshape((-1,) + envs.single_observation_space.shape)
        # dasselbe mit actions                                  (shape (steps*envs, 256, 7))
        # b_actions = actions[:, self.indices].reshape((-1,) + action_space_shape)
        # dasselbe mit logprobs, advantages, returns, values    (shape (steps*envs,))
        # b_logprobs = logprobs[:, self.indices].reshape(-1)
        # b_values = values[:, self.indices].reshape(-1)
        # b_values = b_values.reshape(-1)
        # b_advantages = advantages[:, self.indices].reshape(-1)
        # b_advantages = b_advantages.reshape(-1)
        # b_returns = returns[:, self.indices].reshape(-1)
        # b_returns = b_returns.reshape(-1)
        # dasselbe mit invalid_action_masks                     (shape (steps*envs, 256, 79))
        # b_invalid_action_masks = invalid_action_masks[:, self.indices].reshape((-1,) + invalid_action_shape)
        
        

        # inds: indices from the batch
        main_batch_size = int(main_indices_count * args.num_steps)
        main_minibatch_size = int(main_batch_size // args.n_minibatch) # new (BA Parameter) (minibatch size = 3072 (=(num_envs*num_steps)/ n_minibatch = (24*512)/4))

        
        
        main_agent_batch = {
            "agent": agent,
            "optimizer": optimizer,
            "obs": obs[:, main_indices].reshape((-1,) + envs.single_observation_space.shape),
            "sc": scalar_features[:, main_indices].reshape(-1, scalar_features.shape[-1]),
            "z": z_features[:, main_indices].reshape(-1, z_features.shape[-1]),
            "actions": actions[:, main_indices].reshape((-1,) + action_space_shape),
            "logprobs": logprobs[:, main_indices].reshape(-1),
            "advantages": b_advantages[:, b_main_indices].reshape(-1),
            "returns": b_returns[:, b_main_indices].reshape(-1),
            "values": values[:, main_indices].reshape(-1),
            "masks": invalid_action_masks[:, main_indices].reshape((-1,) + invalid_action_shape),
            "skip_policy_update": args.dbg_no_main_agent_ppo_update
        }
        if unit_bonus_distr is not None:
            main_unit_bonus = unit_bonus_distr[main_indices]
            main_unit_bonus = main_unit_bonus.unsqueeze(0).expand(args.num_steps, -1, -1).reshape(-1, 4)
            main_agent_batch["unit_bonus_distr"] = main_unit_bonus
        
        if args.dbg_deterministic_actions:
            print("\nactions are deterministic (dbg_deterministic_actions) (for debugging purposes only - to get deterministic behaviour between different runs)\n")

        if args.dbg_exploiter_update:
            if not args.dbg_deterministic_actions:
                print("\nuse deterministic actions for main agent PPO update for dbg_deterministic_actions\n")

            if not args.sp:
                print("\nuse args.sp otherwise the observations will diverge because of old Historicals\n")
            self.dbg_prep(main_batch_size)

        pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss, grad_norm = ppo_update.update(
            args,
            envs,
            main_agent_batch,
            device,
            supervised_agent,
            update,
            main_batch_size,
            main_minibatch_size
            )
        if args.dbg_no_main_agent_ppo_update and pg_stop_iter is None:
            pg_stop_iter = -2

        ppo_update.log(
            args,
            writer,
            optimizer,
            args.global_step,
            start_time,
            update,
            pg_stop_iter,
            pg_loss,
            entropy_loss,
            kl_loss,
            approx_kl,
            v_loss,
            loss,
            log_SPS=False,
            grad_norm=grad_norm,
            advantages=main_agent_batch["advantages"],
            values=main_agent_batch["values"],
            returns=main_agent_batch["returns"],
            delta_rewards_score=b_delta_rewards_score[:, b_main_indices]
        )

        # bot_exploiters = np.where(
        #     [isinstance(ag, (league.MainExploiter, league.LeagueExploiter)) for ag in self.active_league_agents[args.num_selfplay_envs:]]
        # )[0] + args.num_selfplay_envs
        # selfplay_exploiters = np.where([isinstance(ag, (league.MainExploiter, league.LeagueExploiter)) for ag in self.active_league_agents[0:args.num_selfplay_envs:2]])[0]
        # exploiter_indices = np.concatenate((selfplay_exploiters * 2, bot_exploiters))
        # b_exploiter_indices = np.concatenate((selfplay_exploiters, bot_exploiters - (args.num_selfplay_envs // 2)))

        if len(indices_per_exploiter) > 0:
            env_shape = (sp_envs or envs).single_observation_space.shape
    
            # update every exploiter individually
            for exploiter, exploiter_idx in indices_per_exploiter.items():
                if exploiter.recent_reset:
                    exploiter.recent_reset = False
                    skip_update_count += 1
                    continue

                if args.dbg_seed:
                    self._seed_for_update(update, args.seed)
                b_exploiter_idx = b_indices_per_exploiter[exploiter]

                if exploiter.optimizer is None:
                    exploiter.optimizer = torch.optim.Adam(exploiter.agent.parameters(), lr=args.exploiter_PPO_learning_rate, eps=1e-5)
                    print(f"Created optimizer for exploiter {exploiter}")
                    exploiter.last_reset_update = update

                exploiter_agent_batch = {
                        "player": exploiter,
                        "agent": exploiter.agent,
                        "optimizer": exploiter.optimizer,
                        "obs": obs[:, exploiter_idx].reshape((-1,) + env_shape),
                        "sc": scalar_features[:, exploiter_idx].reshape(-1, scalar_features.shape[-1]),
                        "z": z_features[:, exploiter_idx].reshape(-1, z_features.shape[-1]),
                        "actions": actions[:, exploiter_idx].reshape((-1,) + action_space_shape),
                        "logprobs": logprobs[:, exploiter_idx].reshape(-1),
                        "advantages": b_advantages[:, b_exploiter_idx].reshape(-1),
                        "returns": b_returns[:, b_exploiter_idx].reshape(-1),
                        "values": values[:, exploiter_idx].reshape(-1),
                        "masks": invalid_action_masks[:, exploiter_idx].reshape((-1,) + invalid_action_shape),
                        "gamma": args.exploiter_gamma,
                        "gae_lambda": args.exploiter_gae_lambda,
                        "ent_coef": args.exploiter_ent_coef,
                        "vf_coef": args.exploiter_vf_coef,
                        "max_grad_norm": args.exploiter_max_grad_norm,
                        "clip_coef": args.exploiter_clip_coef,
                        "update_epochs": args.exploiter_update_epochs,
                        "kle_stop": args.exploiter_kle_stop,
                        "kle_rollback": args.exploiter_kle_rollback,
                        "target_kl": args.exploiter_target_kl,
                        "kl_coeff": args.exploiter_kl_coeff,
                        "norm_adv": args.exploiter_norm_adv,
                        "anneal_lr": args.exploiter_anneal_lr,
                        "clip_vloss": args.exploiter_clip_vloss
                    }
                if unit_bonus_distr is not None:
                    exploiter_unit_bonus = unit_bonus_distr[exploiter_idx]
                    exploiter_unit_bonus = exploiter_unit_bonus.unsqueeze(0).expand(args.num_steps, -1, -1).reshape(-1, 4)
                    exploiter_agent_batch["unit_bonus_distr"] = exploiter_unit_bonus
                    

                if exploiter_lr_fn is not None:
                    reset_update = getattr(exploiter, "last_reset_update", None)
                    if reset_update is None:
                        reset_update = update
                        exploiter.last_reset_update = reset_update
                    exploiter_frac = 1.0 - (update - reset_update) / num_updates
                    if exploiter_frac < 0.0:
                        exploiter_frac = 0.0
                    exploiter_lrnow = exploiter_lr_fn(exploiter_frac)
                else:
                    exploiter_lrnow = args.exploiter_PPO_learning_rate

                exploiter_agent_batch["optimizer"].param_groups[0]["lr"] = exploiter_lrnow

                exploiter_batch_size = exploiter_agent_batch["obs"].shape[0]
                exploiter_minibatch_size = max(exploiter_batch_size // max(args.n_minibatch, 1), 1)

                if args.dbg_exploiter_update:
                    self.dbg_post_first_update(exploiter_agent_batch, main_agent_batch, pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss, exploiter_batch_size)

                update_envs = sp_envs if sp_envs is not None else envs
                pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss, grad_norm = ppo_update.update(
                    args,
                    update_envs,
                    exploiter_agent_batch,
                    device,
                    supervised_agent,
                    update,
                    exploiter_batch_size,
                    exploiter_minibatch_size
                    )
                
                

                # TODO (debugging): debugging löschen
                # if not torch.all(exploiter_agent_batch["obs"] == main_agent_batch["obs"]):
                #     print("Exploiter obs different from main agent obs")
                # if not torch.all(exploiter_agent_batch["sc"] == main_agent_batch["sc"]):
                #     print("Exploiter sc different from main agent sc")
                # if not torch.all(exploiter_agent_batch["z"] == main_agent_batch["z"]):
                #     print("Exploiter z different from main agent z")


                if args.dbg_exploiter_update:
                    self.dbg_post_updates(pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss, optimizer)
                
                league.log_exploiter_ppo_update(
                    args,
                    writer,
                    exploiter_agent_batch,
                    indices_per_exploiter,
                    pg_stop_iter,
                    pg_loss,
                    entropy_loss,
                    kl_loss,
                    approx_kl,
                    v_loss,
                    loss,
                    experiment_name,
                    update,
                    grad_norm=grad_norm,
                    advantages=exploiter_agent_batch["advantages"],
                    delta_rewards_score=b_delta_rewards_score[:, b_exploiter_idx]
                )

            
            # TODO (optimize): wenn ich ppo_update.update benutze, dann soll get_action das immer noch combiniert funktionieren (sonst ist es langsam) (benutze _train_exploiters aus league_training.py?)
            # oder league.train_exploiters entfernen
            # pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss = league.train_exploiters(
            #     args,
            #     envs,
            #     agent_batches,
            #     writer,
            #     args.global_step,
            #     update,
            #     agent,
            #     experiment_name,
            #     exploiter_indices
            # )
            writer.add_scalar("debug/exploiter_skip_updates_count", skip_update_count, args.global_step)
            if skip_update_count:
                print(f"Skipped exploiter updates this rollout: {skip_update_count}")

        if not args.dbg_no_main_agent_ppo_update:
            if args.prod_mode and update % args.checkpoint_frequency == 0:
                print("Saving model checkpoint...")
                if (update < 500 and not args.early_updates):
                    if (update % (args.checkpoint_frequency * 5) == 0):
                        league.save_league_model(save_agent=agent, experiment_name=experiment_name, dir_name="Main_agent_backups", file_name=f"agent_update_{update}")
                else:
                    league.save_league_model(save_agent=agent, experiment_name=experiment_name, dir_name="Main_agent_backups", file_name=f"agent_update_{update}")

        if should_log_every_20_updates:
            writer.add_scalar("charts/sps", int(args.global_step / (time.time() - start_time)), args.global_step)
            print("SPS:", int(args.global_step / (time.time() - start_time)))

        cur_winrate = np.mean(np.add(writer.recent_bot_winloss, 1) / 2) if hasattr(writer, "recent_bot_winloss") else 0.0
        if cur_winrate < args.min_bot_winrate:
            intended_bot_envs = args.max_num_bot_envs
        elif cur_winrate > args.max_bot_winrate:
            intended_bot_envs = args.min_num_bot_envs
        elif args.max_num_bot_envs - args.min_num_bot_envs == 0:
            intended_bot_envs = args.num_bot_envs
        else:
            intended_bot_envs = np.floor(args.max_num_bot_envs - (cur_winrate - args.min_bot_winrate) / ((args.max_bot_winrate - args.min_bot_winrate) / (args.max_num_bot_envs - args.min_num_bot_envs)))

        # remove or add an Bot environment depending on the number of played games in relation to selfplay games
        # if args.dyn_num_bot_envs and last_bot_env_change >= 12 and args.num_bot_envs > args.min_num_bot_envs and (num_done_selfplaygames * args.bot_removing_done_training_ratio <= num_done_botgames or np.mean(np.add(writer.recent_bot_winloss, 1) / 2) > args.min_bot_winrate):
        if args.dyn_num_bot_envs and last_bot_env_change >= 12 and args.num_bot_envs > args.min_num_bot_envs and intended_bot_envs < args.num_bot_envs:
            print("\nRemoving a Bot Environment")

            envs.close()
            envs = self.get_new_bot_envs(args, args.num_bot_envs - 1)
            last_bot_env_change = 0

            agent.remove_last_bot_env()

            obs = obs[:, :args.num_envs]
            actions = actions[:, :args.num_envs]
            logprobs = logprobs[:, :args.num_envs]
            invalid_action_masks = invalid_action_masks[:, :args.num_envs]

            sp_inds = slice(0, args.num_selfplay_envs)
            bot_inds = slice(args.num_selfplay_envs, args.num_envs)

            obs[:, bot_inds].zero_()
            actions[:, bot_inds].zero_()
            logprobs[:, bot_inds].zero_()
            invalid_action_masks[:, bot_inds].zero_()

            if args.unit_exploiters:
                # Do not zero out the others botenvs in unit_bonus_distr, as they are arnt done and are not reinitialized after this
                unit_bonus_distr = unit_bonus_distr[:args.num_envs]



            rewards_attack = rewards_attack[:, :args.num_envs]
            rewards_attack[:, args.num_selfplay_envs:].zero_()
            rewards_winloss = rewards_winloss[:, :args.num_envs]
            rewards_winloss[:, args.num_selfplay_envs:].zero_()
            delta_rewards_score = delta_rewards_score[:, :args.num_envs]
            delta_rewards_score[:, args.num_selfplay_envs:].zero_()
            delta_score_sums = delta_score_sums[:args.num_envs]
            delta_score_sums[args.num_selfplay_envs:].zero_()
            # TODO (optimize): muss man die wirklich resetten?
            dones = dones[:, :args.num_envs]
            dones[:, args.num_selfplay_envs:].zero_()
            values = values[:, :args.num_envs]
            values[:, args.num_selfplay_envs:].zero_()

            next_obs_np, _, bot_res = envs.reset()
            bot_next_obs = torch.Tensor(next_obs_np).to(device)
            last_bot_scorerew = torch.zeros(args.num_bot_envs, device=device)

            next_done = next_done[:args.num_envs]
            next_done[args.num_selfplay_envs:].zero_()

            scalar_features = scalar_features[:, :args.num_envs]
            scalar_features[:, args.num_selfplay_envs:].zero_()
            z_features = z_features[:, :args.num_envs]
            z_features[:, args.num_selfplay_envs:].zero_()

            bot_position_indices = bot_position_indices[:args.num_bot_envs]




            

            print("New number of Bot Environments:", args.num_bot_envs)
            print("")

        # elif args.dyn_num_bot_envs and last_bot_env_change >= 12 and args.num_bot_envs < args.max_num_bot_envs and num_done_selfplaygames * args.bot_adding_done_training_ratio > num_done_botgames:
        elif args.dyn_num_bot_envs and last_bot_env_change >= 12 and args.num_bot_envs < args.max_num_bot_envs and intended_bot_envs > args.num_bot_envs:
            print("\nAdding an Bot Environment")

            envs.close()
            envs = self.get_new_bot_envs(args, args.num_bot_envs + 1)
            last_bot_env_change = 0

            agent.add_bot_env()

            if args.unit_exploiters:
                # Do not zero out the others botenvs in unit_bonus_distr, as they are arnt done and are not reinitialized after this
                unit_bonus_distr = torch.cat((unit_bonus_distr, unit_bonus_distr[-1:].clone()))
                self.unit_bonus_distr = unit_bonus_distr
                self.get_new_unit_bonus_distr(torch.tensor([args.num_envs - 1]), device)
                unit_bonus_distr = self.unit_bonus_distr

            # Rebuild rollout buffers to avoid temporary peak allocations from torch.cat on large CUDA tensors.
            sp_next_done = next_done[:args.num_selfplay_envs].clone()
            sp_delta_score_sums = delta_score_sums[:args.num_selfplay_envs].clone()

            obs_dtype = obs.dtype
            actions_dtype = actions.dtype
            logprobs_dtype = logprobs.dtype
            invalid_action_masks_dtype = invalid_action_masks.dtype
            rewards_attack_dtype = rewards_attack.dtype
            rewards_winloss_dtype = rewards_winloss.dtype
            delta_rewards_score_dtype = delta_rewards_score.dtype
            dones_dtype = dones.dtype
            values_dtype = values.dtype
            scalar_features_dtype = scalar_features.dtype
            z_features_dtype = z_features.dtype

            del obs, actions, logprobs, invalid_action_masks
            del rewards_attack, rewards_winloss, delta_rewards_score, dones, values
            del scalar_features, z_features
            if device.type == "cuda":
                torch.cuda.empty_cache()

            obs = torch.zeros(
                (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
                device=device,
                dtype=obs_dtype,
            )
            actions = torch.zeros(
                (args.num_steps, args.num_envs) + action_space_shape,
                device=device,
                dtype=actions_dtype,
            )
            logprobs = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=logprobs_dtype)
            invalid_action_masks = torch.zeros(
                (args.num_steps, args.num_envs) + invalid_action_shape,
                device=device,
                dtype=invalid_action_masks_dtype,
            )

            rewards_attack = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=rewards_attack_dtype)
            rewards_winloss = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=rewards_winloss_dtype)
            delta_rewards_score = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=delta_rewards_score_dtype)
            dones = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=dones_dtype)
            values = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=values_dtype)
            scalar_features = torch.zeros((args.num_steps, args.num_envs, 11), device=device, dtype=scalar_features_dtype)
            z_features = torch.zeros((args.num_steps, args.num_envs, 8), device=device, dtype=z_features_dtype)
            delta_score_sums = torch.zeros((args.num_envs), device=device, dtype=sp_delta_score_sums.dtype)
            delta_score_sums[:args.num_selfplay_envs] = sp_delta_score_sums

            sp_inds = slice(0, args.num_selfplay_envs)
            bot_inds = slice(args.num_selfplay_envs, args.num_envs)
            obs[:, bot_inds].zero_()
            actions[:, bot_inds].zero_()
            logprobs[:, bot_inds].zero_()
            invalid_action_masks[:, bot_inds].zero_()


            next_obs_np, _, bot_res = envs.reset()
            bot_next_obs = torch.Tensor(next_obs_np).to(device)
            last_bot_scorerew = torch.zeros(args.num_bot_envs, device=device)

            next_done = torch.zeros((args.num_envs), device=device, dtype=sp_next_done.dtype)
            next_done[:args.num_selfplay_envs] = sp_next_done

            bot_position_indices = torch.cat((bot_position_indices, bot_position_indices[:1].clone()))

            print("New number of Bot Environments:", args.num_bot_envs)
            print("")

        if should_log_every_20_updates:
            writer.add_scalar("charts/num_parallel_Bot_Games", args.num_bot_envs, args.global_step)


    def get_new_unit_bonus_distr(self, indices, device: torch.device) -> torch.Tensor:
        if self.args.Unit_reward_per_exploiter:
            for i in indices:
                bonus = self.active_league_agents[i].unit_bonus_distr
                if bonus is None:
                    bonus = torch.zeros(4, device=device)
                elif bonus.device != device:
                    bonus = bonus.to(device)
                self.unit_bonus_distr[i] = bonus
        elif self.args.unit_exploiters:
            for i in indices:
                if not isinstance(self.active_league_agents[i], league.MainPlayer) and not (isinstance(self.active_league_agents[i], league.Historical) and isinstance(self.active_league_agents[i].parent, league.MainPlayer)):
                    self.unit_bonus_distr[i] = self._sample_unit_bonus_distr((4,), device)
                else:
                    self.unit_bonus_distr[i] = torch.zeros(4, device=device)

    @staticmethod
    def _as_compact_slice(indices: np.ndarray):
        arr = np.asarray(indices, dtype=np.int64)
        if arr.size == 0:
            return slice(0, 0, 1)
        if arr.size == 1:
            start = int(arr[0])
            return slice(start, start + 1, 1)
        diffs = np.diff(arr)
        if np.all(diffs == 1):
            return slice(int(arr[0]), int(arr[-1]) + 1, 1)
        return arr


    def _refresh_main_indices(self, args):
        if args.training_on_bot_envs:
            main_indices = np.where([isinstance(ag, league.MainPlayer) for ag in self.active_league_agents[args.num_selfplay_envs:]])[0] + args.num_selfplay_envs
            b_main_indices = main_indices - (args.num_selfplay_envs // 2)
        else:
            main_indices = np.array([], dtype=np.int64)
            b_main_indices = np.array([], dtype=np.int64)

        if args.train_on_old_mains:  # TODO: Dosnt work, because Player 1 can change in an rollout. (is that a problem?)
            selfplay_mains = np.where((isinstance(self.active_league_agents, league.MainPlayer)))[0]
            main_indices = np.concatenate((selfplay_mains, main_indices), axis=0)
            b_main_indices = np.concatenate((b_main_indices, selfplay_mains // 2), axis=0)
        else:
            selfplay_mains = np.where([isinstance(ag, league.MainPlayer) for ag in self.active_league_agents[0:args.num_selfplay_envs:2]])[0]
            main_indices = np.concatenate((selfplay_mains * 2, main_indices))
            b_main_indices = np.concatenate((selfplay_mains, b_main_indices))

        self.main_indices_count = int(main_indices.size)
        self.main_indices = self._as_compact_slice(main_indices)
        self.b_main_indices = self._as_compact_slice(b_main_indices)

    def _refresh_exploiter_indices(self, args):
        indices_per_exploiter = {}
        b_indices_per_exploiter = {}
        for idx, p in enumerate(self.active_league_agents):
            if isinstance(p, (league.MainExploiter, league.LeagueExploiter)):
                indices_per_exploiter.setdefault(p, []).append(idx)
                b_indices_per_exploiter.setdefault(p, []).append(
                    idx - (args.num_selfplay_envs // 2) if idx >= args.num_selfplay_envs else idx // 2
                )

        self.indices_per_exploiter = {
            exploiter: self._as_compact_slice(exploiter_indices)
            for exploiter, exploiter_indices in indices_per_exploiter.items()
        }
        self.b_indices_per_exploiter = {
            exploiter: self._as_compact_slice(exploiter_indices)
            for exploiter, exploiter_indices in b_indices_per_exploiter.items()
        }

    # TODO (optimize): in obs, ... die envs entfernen, die man nicht braucht (spart Rechenzeit)
    def get_new_bot_envs(self, args, num_bots):

        args.num_bot_envs = num_bots
        args.num_envs = args.num_selfplay_envs + args.num_bot_envs

        if len(self.active_league_agents) < args.num_envs:
            # assert isinstance(self.active_league_agents[0], league.MainPlayer) "self.active_league_agents[0] must be an MainPlayer"
            self.active_league_agents.append(self.active_league_agents[0])
            self.indices = torch.cat((self.indices, torch.tensor([args.num_envs - 1], device=self.device)))
        else:
            self.active_league_agents = self.active_league_agents[:args.num_envs]
            self.indices = self.indices[:-1]

        self._refresh_main_indices(args)
        self._refresh_exploiter_indices(args)


        opponents = [microrts_ai.coacAI for _ in range((args.num_bot_envs+1)//2)] + [microrts_ai.mayari for _ in range((args.num_bot_envs)//2)]
        reward_weight = np.array([1.0, 1.0, 1.0, 0.2, 1.0, 4.0, 5.25, 6.0, 0])
        
        envs = MicroRTSGridModeVecEnv(
            num_selfplay_envs=0,
            num_bot_envs=num_bots,
            max_steps=2000, # new (BA Parameter) (max episode length of 2000)
            always_player_1=True,
            bot_envs_alternate_player=False,
            render_theme=1,
            ai2s=opponents, # new (BA Parameter) (Targeted training during PPO training) 16 CoacAI and 8 Mayari environments
            # ai2s=[microrts_ai.coacAI for _ in range(3)] + 
            # [microrts_ai.mayari for _ in range(4)] + 
            # [microrts_ai.mixedBot for _ in range(4)] + 
            # [microrts_ai.izanagi for _ in range(3)] +
            # [microrts_ai.droplet for _ in range(4)] +
            # [microrts_ai.tiamat for _ in range(3)] +
            # [microrts_ai.workerRushAI for _ in range(3)],
            map_paths=["maps/16x16/basesWorkers16x16A.xml"], # new (BA Parameter) (All evaluations were conducted on the basesWorkers16x16A map)
            reward_weight=reward_weight,
        )
        envsT = MicroRTSSpaceTransform(envs)
        # print(envsT.__class__.mro())
        # print(hasattr(envsT, "step_async"))
        # print(envsT.step_async.__qualname__)
        # print(envsT.step_wait.__qualname__)

        envsT = VecstatsMonitor(envsT, args.gamma)
        if args.capture_video:
            envs = VecVideoRecorder(envs, f'videos/{args.exp_name}',
                                    record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)

        return envsT

    # TODO (debugging): Debugging (nachher entfernen)
    def assert_supervised_grads_zero(self, supervised_agent):
        max_abs = 0.0
        names = []
        for name, p in supervised_agent.named_parameters():
            if p.grad is None:
                continue
            m = p.grad.detach().abs().max().item()
            if m > max_abs:
                max_abs = m
                names = [name]
                print(f"\n\n[supervised] max|grad|={max_abs} in {names}!!!\n\n")

    # TODO (debugging): Debugging (nachher entfernen)
    def check_values(self, scalar_features, z_features, values, agent, step, next_scalar_features=None, next_z_features=None, obs=None, next_obs=None, flatten = False):
        return
        if flatten:
            if not torch.allclose(values[step, ::2], agent.get_value(obs[step], scalar_features[step], z_features[step]).flatten()[::2], rtol=1e-3, atol=1e-5):
                print(f"\n\nValue mismatch at (flatten) step: {step}, distance: {values[step, ::2] - agent.get_value(obs[step], scalar_features[step], z_features[step]).flatten()[::2]}\n\n")

        else:
            if not torch.allclose(values[0, ::2], agent.get_value(next_obs, next_scalar_features, next_z_features).reshape(1, -1)[0, ::2], rtol=1e-3, atol=1e-5):
                print(f"\n\nValue mismatch at (reshape) step: {step}, distance: {values[0, ::2] - agent.get_value(next_obs, scalar_features[-1], z_features[-1]).reshape(1, -1)[0, ::2]}\n\n")


    def _seed_for_update(self, offset: int, base_seed) -> None:
        update_seed = int(base_seed) + int(offset)
        random.seed(update_seed)
        np.random.seed(update_seed)
        torch.manual_seed(update_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(update_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def dbg_prep(self, main_batch_size):
        import random
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        import copy
        self.db_agents = copy.deepcopy(self.active_league_agents)
        self.db_sd = [copy.deepcopy(self.active_league_agents[i].agent.state_dict()) for i in range(len(self.active_league_agents))]
        
        inds = np.arange(main_batch_size)
        np.random.shuffle(inds)
        print(f"main_batch_size: {main_batch_size}, main_inds: {inds[:10]}")
    
    def dbg_post_first_update(self, exploiter_agent_batch, main_agent_batch, pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss, exploiter_batch_size):

        import random
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        inds = np.arange(exploiter_batch_size)
        np.random.shuffle(inds)
        print(f"exploiter_batch_size: {exploiter_batch_size}, exploiter_inds: {inds[:10]}")

        if not torch.all(exploiter_agent_batch["obs"] == main_agent_batch["obs"]):
            print("Exploiter obs not same as main agent obs")
            breakpoint()
        if not torch.all(exploiter_agent_batch["sc"] == main_agent_batch["sc"]):
            print("Exploiter sc not same as main agent sc")
            breakpoint()
        if not torch.all(exploiter_agent_batch["z"] == main_agent_batch["z"]):
            print("Exploiter z not same as main agent z")
            breakpoint()
        if not torch.all(exploiter_agent_batch["actions"] == main_agent_batch["actions"]):
            print("Exploiter actions not same as main agent actions")
            breakpoint()
        if not torch.all(exploiter_agent_batch["logprobs"] == main_agent_batch["logprobs"]):
            print("Exploiter logprobs not same as main agent logprobs")
            breakpoint()
        if not torch.all(exploiter_agent_batch["advantages"] == main_agent_batch["advantages"]):
            print("Exploiter advantages not same as main agent advantages")
            breakpoint()
        if not torch.all(exploiter_agent_batch["returns"] == main_agent_batch["returns"]):
            print("Exploiter returns not same as main agent returns")
            breakpoint()
        if not torch.all(exploiter_agent_batch["values"] == main_agent_batch["values"]):
            print("Exploiter values not same as main agent values")
            breakpoint()
        if not torch.all(exploiter_agent_batch["masks"] == main_agent_batch["masks"]):
            print("Exploiter masks not same as main agent masks")
            breakpoint()
        print("\ninputs are equal:")
        print(torch.all(torch.tensor([
                    torch.all(exploiter_agent_batch["obs"] == main_agent_batch["obs"]),
                    torch.all(exploiter_agent_batch["sc"] == main_agent_batch["sc"]),
                    torch.all(exploiter_agent_batch["z"] == main_agent_batch["z"]),
                    torch.all(exploiter_agent_batch["actions"] == main_agent_batch["actions"]),
                    torch.all(exploiter_agent_batch["logprobs"] == main_agent_batch["logprobs"]),
                    torch.all(exploiter_agent_batch["advantages"] == main_agent_batch["advantages"]),
                    torch.all(exploiter_agent_batch["returns"] == main_agent_batch["returns"]),
                    torch.all(exploiter_agent_batch["values"] == main_agent_batch["values"]),
                    torch.all(exploiter_agent_batch["masks"] == main_agent_batch["masks"])
                    ])).item())

        self.db_pg_stop_iter, self.db_pg_loss, self.db_entropy_loss, self.db_kl_loss, self.db_approx_kl, self.db_v_loss, self.db_loss = pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss

    def dbg_post_updates(self, pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss, optimizer):
        print("\noptimizers are equal:")
        print("  lr:")
        print(f"  {torch.all(torch.tensor([optimizer.param_groups[0]['lr'] == self.active_league_agents[8].optimizer.param_groups[0]['lr']])).item()}")
        print("  params:")
        print(f"  {torch.all(torch.tensor([torch.all((optimizer.param_groups[0]['params'][i].data == self.active_league_agents[8].optimizer.param_groups[0]['params'][i].data)) for i in range(len(optimizer.param_groups[0]['params']))])).item()}")

        print("\noutputs are equal:")
        print(torch.all(torch.tensor([
                    self.db_pg_stop_iter == pg_stop_iter,
                    self.db_pg_loss == pg_loss,
                    self.db_entropy_loss == entropy_loss,
                    self.db_kl_loss == kl_loss,
                    self.db_approx_kl == approx_kl,
                    self.db_v_loss == v_loss,
                    self.db_loss == loss
                    ])).item())

        # for k in self.active_league_agents[0].agent.state_dict().keys():
        #    print(torch.all((self.active_league_agents[0].agent.state_dict()[k] == self.db_sd[k])))
        print("\nMain changed:")
        print(not torch.all(torch.tensor([torch.all((self.active_league_agents[0].agent.state_dict()[k] == self.db_sd[0][k])) for k in self.active_league_agents[0].agent.state_dict().keys()])).item())
        
        print(f"\n{self.active_league_agents[8]} changed:")
        print(not torch.all(torch.tensor([torch.all((self.active_league_agents[8].agent.state_dict()[k] == self.db_sd[8][k])) for k in self.active_league_agents[0].agent.state_dict().keys()])).item())

        # for k in self.active_league_agents[0].agent.state_dict().keys():
        #     print(torch.all((self.active_league_agents[0].agent.state_dict()[k] ==   self.active_league_agents[8].agent.state_dict()[k])))
        print(f"\n{self.active_league_agents[0]} and {self.active_league_agents[8]} are equal:")
        equal = torch.all(torch.tensor([torch.all((self.active_league_agents[0].agent.state_dict()[k] == self.active_league_agents[8].agent.state_dict()[k])) for k in self.active_league_agents[0].agent.state_dict().keys()])).item()
        print(equal)
        if not equal:
            breakpoint()
        return
