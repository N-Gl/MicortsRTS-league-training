from collections import deque
import os
import sys
import time
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
            for i in range(0, 4):
                next_obs[1:args.num_selfplay_envs:2, :, :, 22 + 5 * i : 26 + 5 * i] = (
                    next_obs[1:args.num_selfplay_envs:2, :, :, 22 + 5 * i : 26 + 5 * i].roll(shifts=2, dims=3)
                )
            next_obs[1:args.num_selfplay_envs:2, :, :, 50:54] = next_obs[1:args.num_selfplay_envs:2, :, :, 50:54].roll(
                shifts=2, dims=3
            )
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

# TODO: benutze die Methode aus league.py (ist dasselbe?)
def save_league_model(save_agent, experiment_name: str, dir_name: str, file_name: str):
    os.makedirs(f"league_models/{experiment_name}/{dir_name}", exist_ok=True)
    if isinstance(save_agent, Agent):
        torch.save(save_agent.state_dict(), f"league_models/{experiment_name}/{dir_name}/{file_name}.pt")
    else:
        torch.save(save_agent.agent.state_dict(), f"league_models/{experiment_name}/{dir_name}/{file_name}.pt")


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

# TODO: debugging function
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

class SelfPlayTrainer:
    def __init__(
        self,
        agent,
        supervised_agent,
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
        self.envs = envs
        self.sp_envs = sp_envs
        self.args = args
        self.writer = writer
        self.device = device
        self.experiment_name = experiment_name
        self.get_scalar_features = get_scalar_features
        self.checkpoint_frequency = args.checkpoint_frequency
        self.active_league_agents = []
        self.league_agent = Selfplay_agent(agent)
        # TODO: initialisiere die League Agents, wie in league_training.py
        self.league_supervised_agent = Selfplay_agent(supervised_agent)
        for _ in range(args.num_main_agents):
            self.active_league_agents.append(self.league_agent)
            self.active_league_agents.append(self.league_supervised_agent)
        for _ in range(args.num_bot_envs):
            self.active_league_agents.append(self.league_agent)
        # TODO: richtig initialisiert?
        self.indices = torch.tensor(range(args.num_selfplay_envs, args.num_envs), dtype=torch.long, device=device)
        self.indices = torch.cat(
            (torch.tensor(range(0, args.num_selfplay_envs, 2), dtype=torch.long, device=device), self.indices)
        )

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
        supervised_agent = self.supervised_agent or Agent(agent.action_plane_nvec, agent.device, initial_weights=agent.state_dict())
        last_bot_env_change = 0

        if args.render:
            if args.render_all:
                from ppo import Rendering

        if args.num_selfplay_envs == 0:
            raise ValueError("league training requires num_selfplay_envs > 0")
        if args.num_main_agents == 0:
            raise ValueError("league training requires at least one main agent")
        
        league_instance, active_league_agents, agent_type = league.initialize_league(args, device, agent)

        optimizer = torch.optim.Adam(agent.parameters(), lr=args.PPO_learning_rate, eps=1e-5)
        if args.anneal_lr:
            lr_fn = lambda frac: frac * args.PPO_learning_rate  # noqa: E731
        else:
            lr_fn = None

        
        cleanup_break = break_on_stdout("Issuing a non legal action")

        mapsize = 16 * 16
        action_space_shape = (mapsize, envs.action_plane_space.shape[0])
        invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum() + 1)

        bot_obs = torch.zeros((args.num_steps, args.num_bot_envs) + envs.single_observation_space.shape).to(device)
        bot_actions = torch.zeros((args.num_steps, args.num_bot_envs) + action_space_shape).to(device)
        bot_logprobs = torch.zeros((args.num_steps, args.num_bot_envs)).to(device)
        bot_invalid_action_masks = torch.zeros((args.num_steps, args.num_bot_envs) + invalid_action_shape).to(device)

        rewards_attack = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_winloss = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_score = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        sp_obs = torch.zeros((args.num_steps, args.num_selfplay_envs) + envs.single_observation_space.shape).to(device)
        sp_actions = torch.zeros((args.num_steps, args.num_selfplay_envs) + action_space_shape).to(device)
        sp_logprobs = torch.zeros((args.num_steps, args.num_selfplay_envs)).to(device)
        sp_invalid_action_masks = torch.zeros((args.num_steps, args.num_selfplay_envs) + invalid_action_shape).to(device)

        global_step = 0
        start_time = time.time()

        next_obs_np, _, bot_res = envs.reset()
        bot_next_obs = torch.Tensor(next_obs_np).to(device)

        next_obs_np, _, sp_res = sp_envs.reset()
        sp_next_obs = torch.Tensor(next_obs_np).to(device)
        adjust_obs_selfplay(args, sp_next_obs, is_new_env=True)

        next_done = torch.zeros(args.num_envs).to(device)
        scalar_features = torch.zeros((args.num_steps, args.num_envs, 11)).to(device)
        z_features = torch.zeros((args.num_steps, args.num_envs, 8), dtype=torch.long).to(device)

        num_updates = args.total_timesteps // args.batch_size
        bot_position_indices = (
            torch.arange(mapsize, device=device, dtype=torch.int64).unsqueeze(0).repeat(args.num_bot_envs, 1).unsqueeze(2)
        )
        sp_position_indices = (
            torch.arange(mapsize, device=device, dtype=torch.int64).unsqueeze(0).repeat(args.num_selfplay_envs, 1).unsqueeze(2)
        )

        

        print("League PPO training started")
        

        for update in range(1, num_updates + 1):

            if lr_fn is not None:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = lr_fn(frac)

            for step in range(args.num_steps):
                if args.render:
                    if args.render_all:
                        # TODO: funktioniert nicht richtig
                        Rendering.render_all_envs(envs)
                        Rendering.render_all_envs(sp_envs)
                    else:
                        envs.render("human")
                        sp_envs.render("human")
                        
                global_step += (args.num_selfplay_envs // 2) + args.num_bot_envs
                # global_step += (args.num_main_agents // 2) + args.num_bot_envs
                bot_obs[step] = bot_next_obs
                sp_obs[step] = sp_next_obs
                next_obs = torch.cat([sp_next_obs, bot_next_obs], dim=0)
                res = sp_res + bot_res
                scalar_features[step] = self.get_scalar_features(next_obs, res, args.num_envs).to(device)
                dones[step] = next_done

                with torch.no_grad():
                    # unique_agents = agent.get_unique_agents(self.active_league_agents, selfplay_only=True)
                    unique_agents = agent.get_unique_agents(self.active_league_agents)
                    sp_only_unique_agents = agent.get_unique_agents(self.active_league_agents[:args.num_selfplay_envs])

                    z_features[step] = agent.selfplay_get_z_encoded_features(
                        args=args,
                        device=device,
                        z_features=z_features,
                        next_obs=bot_next_obs,
                        sp_next_obs=sp_next_obs,
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
                        torch.cat([sp_obs[step], bot_obs[step]], dim=0),
                        scalar_features[step],
                        z_features[step],
                        num_selfplay_envs=args.num_selfplay_envs,
                        num_envs=args.num_envs,
                        unique_agents=unique_agents,
                        only_player_0=True
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
                    

                    self.check_values(scalar_features, z_features, values, agent, step, obs=torch.cat([sp_obs[step], bot_obs[step]], dim=0), flatten=True)

                    # gesamplete action (aus Verteilung der Logits) (24, 256, 7),
                    # actor(forward(...)), invalid_action_masks
                    # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                    bot_actions[step], bot_logprobs[step], _, bot_invalid_action_masks[step] = agent.get_action(
                        bot_obs[step],
                        scalar_features[step, args.num_selfplay_envs:],
                        z_features[step, args.num_selfplay_envs:],
                        envs=envs
                    )
###############################################################################################################################################
                    sp_actions[step], sp_logprobs[step], _, sp_invalid_action_masks[step] = agent.selfplay_get_action(
                        sp_obs[step],
                        scalar_features[step, :args.num_selfplay_envs],
                        z_features[step, :args.num_selfplay_envs],
                        num_selfplay_envs=args.num_selfplay_envs,
                        num_envs=args.num_selfplay_envs,
                        envs=sp_envs,
                        active_league_agents=self.active_league_agents,
                        unique_agents=sp_only_unique_agents
                    )

                # Die Grid-Position zu jedem Action hinzugefügt (24, 256, 8)
                bot_real_action = torch.cat([bot_position_indices, bot_actions[step]], dim=2).cpu().numpy()
                sp_real_action = torch.cat([sp_position_indices, sp_actions[step]], dim=2).cpu().numpy()
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
                bot_valid_actions = bot_real_action[bot_invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
                bot_valid_counts = bot_invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()
                sp_valid_actions = sp_real_action[sp_invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
                sp_valid_counts = sp_invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()

                # Anpassungen für Spieler 1 nach (Spieler 1 -> Spieler 0)
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
                sp_next_obs, _, sp_attackrew, sp_winlossrew, sp_scorerew, sp_ds, sp_infos, sp_res = sp_envs.step(sp_java_valid_actions)
                sp_next_obs = torch.Tensor(sp_envs._from_microrts_obs(sp_next_obs)).to(device)
                
                adjust_obs_selfplay(args, sp_next_obs)

                '''winloss = min(0.01, 6.72222222e-9 * global_step)
                densereward = max(0, 0.8 + (-4.44444444e-9 * global_step))

                if global_step < 100000000:
                    scorew = 0.19 + 1.754e-8 * global_step
                else:
                    scorew = 0.5 - 1.33e-8 * global_step'''


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

                rewards_winloss[step] = torch.Tensor(np.concatenate([sp_winlossrew, bot_winlossrew]) * winloss).to(device)
                rewards_score[step] = torch.Tensor(np.concatenate([sp_scorerew, bot_scorerew])).to(device)
                next_done = torch.Tensor(np.concatenate([sp_ds, bot_ds])).to(device)

                # =============
                # Logging PPO training
                # =============
                infos =  sp_infos + bot_infos
                for info in infos:

                    if 'episode' in info.keys():
                        game_length = info['episode']['l']
                        winloss = winloss * (-0.00013 * game_length + 1.16)
                        writer.add_scalar("charts/old_episode_reward", info['episode']['r'], global_step)
                        writer.add_scalar("charts/Game_length", game_length,global_step)
                        writer.add_scalar("charts/Episode_reward", info['microrts_stats']['RAIWinLossRewardFunction'] * winloss + info['microrts_stats']['AttackRewardFunction'] * attack, global_step)
                        writer.add_scalar("charts/AttackReward", info['microrts_stats']['AttackRewardFunction'] * attack, global_step)
                        writer.add_scalar("charts/WinLossRewardFunction", info['microrts_stats']['RAIWinLossRewardFunction']* winloss, global_step)

                        where_done = torch.where(next_done)

                        break
                if np.any(['episode' in info.keys() for info in infos]):
                    if not hasattr(writer, "recent_bot_winloss"):
                                writer.recent_bot_winloss = deque([0.0] * 50, maxlen=200)
                    if not hasattr(writer, "recent_selfplay_winloss"):
                                writer.recent_selfplay_winloss = deque([0.0] * 50, maxlen=200)

                    for done_idx in where_done[0]:
                        if done_idx > args.num_selfplay_envs - 1:
                            print(f"Game {int(args.num_selfplay_envs/2 + int(done_idx - (args.num_selfplay_envs - 1)))} ended, result: {infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction']}")
                            writer.recent_bot_winloss.append(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'])

                            selfplay_winrate = np.mean(np.clip(writer.recent_selfplay_winloss, 0, 1))
                            selfplay_withdraw = np.mean(np.add(writer.recent_selfplay_winloss, 1) / 2)
                            bot_winrate = np.mean(np.clip(writer.recent_bot_winloss, 0, 1))
                            with_draw = np.mean(np.add(writer.recent_bot_winloss, 1) / 2)

                            winloss_values = np.array(np.clip(writer.recent_bot_winloss, 0, 1))
                            writer.add_scalar("progress/num_bot_games", num_done_botgames, global_step)
                            writer.add_scalar(f"winrates/bot_Winrate_with_Draw_0", bot_winrate, num_done_botgames)
                            writer.add_scalar(f"winrates/bot_Winrate_std", np.std(winloss_values), num_done_botgames)
                            writer.add_scalar(f"winrates/bot_Winrate_with_draw_0.5", with_draw, num_done_botgames)
                            print(f"global_step={global_step}, episode_reward={(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * winloss + infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack):.3f}")
                            print(f"bot_winrate_{len(writer.recent_bot_winloss)}={bot_winrate:.3f}, bot_winrate_with_draw_0.5_{len(writer.recent_bot_winloss)}={with_draw:.3f}")
                            print(f"match in Botgame {int(done_idx - (args.num_selfplay_envs - 1))} \n")
                            num_done_botgames += 1
                            last_bot_env_change += 1

                        elif done_idx % 2 == 0:
                            print(f"Game {int(done_idx/2)} ended, result: {infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction']}")
                            writer.recent_selfplay_winloss.append(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'])

                            selfplay_withdraw = np.mean(np.add(writer.recent_selfplay_winloss, 1) / 2)
                            selfplay_winrate = np.mean(np.clip(writer.recent_selfplay_winloss, 0, 1))
                            bot_winrate = np.mean(np.clip(writer.recent_bot_winloss, 0, 1))
                            with_draw = np.mean(np.add(writer.recent_bot_winloss, 1) / 2)

                            winloss_values = np.array(writer.recent_selfplay_winloss)
                            writer.add_scalar("progress/num_selfplay_games", num_done_selfplaygames, global_step)
                            writer.add_scalar(f"winrates/selfplay_Winrate_with_draw", selfplay_withdraw, num_done_selfplaygames)
                            writer.add_scalar(f"winrates/selfplay_Winrate_no_draw", selfplay_winrate, num_done_selfplaygames)
                            writer.add_scalar(f"winrates/selfplay_Winrate_no_draw_std", np.std(winloss_values), num_done_selfplaygames)
                            print(f"global_step={global_step}, episode_reward={(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * winloss + infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack):.3f}")
                            print(f"selfplay_winrate_no_draw_{len(writer.recent_selfplay_winloss)}={selfplay_winrate:.3f}, selfplay_winrate_with_draw_0.5_{len(writer.recent_selfplay_winloss)}={selfplay_withdraw:.3f}\n")
                            num_done_selfplaygames += 1
                            last_bot_env_change += 1
                        
                # =============
            # =========================


            

            

            # Debug helper: skip the entire PPO update phase (no GAE, no grads, no loss logging)
            if args.dbg_no_main_agent_ppo_update:
                continue
        # =========================
        # PPO update
        # =========================
            
            # unique_agents = agent.get_unique_agents(self.active_league_agents, selfplay_only=True)
            unique_agents = agent.get_unique_agents(self.active_league_agents)

            obs = torch.cat([sp_obs, bot_obs], dim=1)
            actions = torch.cat([sp_actions, bot_actions], dim=1)
            logprobs = torch.cat([sp_logprobs, bot_logprobs], dim=1)
            invalid_action_masks = torch.cat([sp_invalid_action_masks, bot_invalid_action_masks], dim=1)


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
                    only_player_0=True
                ).reshape(1, -1)

                self.check_values(
                    scalar_features, z_features, next_value, 
                    agent, step, 
                    next_scalar_features=next_scalar_features, 
                    next_z_features=next_z_features, 
                    next_obs=next_obs, 
                    flatten=False
                    )

                # dont update supervised_agent
                b_next_value = next_value[:, self.indices]
                b_values = values[:, self.indices]
                b_rewards_attack = rewards_attack[:, self.indices]
                b_rewards_winloss = rewards_winloss[:, self.indices]
                b_rewards_score = rewards_score[:, self.indices]
                b_dones = dones[:, self.indices]
                b_next_done = next_done[self.indices]


                b_advantages, b_returns = ppo_update.gae(args, device, b_next_value, b_values, b_rewards_attack, b_rewards_winloss, b_rewards_score, b_dones, b_next_done)


            # flatten the batch
            # args.num_steps, args.num_envs Dimensionen vereinigen  (shape (steps*envs, 11))
            # (ScFeatures für jeden Step, Environment sortiert Step, dann nach Environments)
            b_Sc = scalar_features[:, self.indices].reshape(-1, scalar_features.shape[-1])
            # args.num_steps, args.num_envs Dimensionen vereinigen  (shape (steps*envs, 8))
            # (zFeatures für jeden Step, Environment sortiert Step, dann nach Environments)
            b_z = z_features[:, self.indices].reshape(-1, z_features.shape[-1])
            # dasselbe mit obs                                      (shape (steps*envs, 16, 16, 73)
            b_obs = obs[:, self.indices].reshape((-1,) + envs.single_observation_space.shape)
            # dasselbe mit actions                                  (shape (steps*envs, 256, 7))
            b_actions = actions[:, self.indices].reshape((-1,) + action_space_shape)
            # dasselbe mit logprobs, advantages, returns, values    (shape (steps*envs,))
            b_logprobs = logprobs[:, self.indices].reshape(-1)
            # b_values = values[:, self.indices].reshape(-1)
            b_values = b_values.reshape(-1)
            # b_advantages = advantages[:, self.indices].reshape(-1)
            b_advantages = b_advantages.reshape(-1)
            # b_returns = returns[:, self.indices].reshape(-1)
            b_returns = b_returns.reshape(-1)
            # dasselbe mit invalid_action_masks                     (shape (steps*envs, 256, 79))
            b_invalid_action_masks = invalid_action_masks[:, self.indices].reshape((-1,) + invalid_action_shape)
            

            new_batch_size = int(len(self.indices) * args.num_steps)
            minibatch_size = int(new_batch_size // args.n_minibatch) # new (BA Parameter) (minibatch size = 3072 (=(num_envs*num_steps)/ n_minibatch = (24*512)/4))

            
            pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss = ppo_update.update(args, agent, envs, device, supervised_agent, optimizer, update, b_values, b_advantages, b_returns, b_Sc, b_z, b_obs, b_actions, b_logprobs, b_invalid_action_masks, new_batch_size, minibatch_size)

            ppo_update.log(args, writer, optimizer, global_step, start_time, update, pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss)

            if args.prod_mode and update % self.checkpoint_frequency == 0:
                if (update < 500 and not args.early_updates):
                    if (update % (self.checkpoint_frequency * 5) == 0):
                        save_league_model(save_agent=agent, experiment_name=self.experiment_name, dir_name="Main_agent_backups", file_name=f"agent_update_{update}")
                else:
                    save_league_model(save_agent=agent, experiment_name=self.experiment_name, dir_name="Main_agent_backups", file_name=f"agent_update_{update}")

            # TODO: Verbessere, wann neue Bots geladen werden
            # remove or add an Bot environment depending on the number of played games in relation to selfplay games
            if  last_bot_env_change >= 50 and args.num_bot_envs > 0 and num_done_selfplaygames * args.bot_removing_done_training_ratio <= num_done_botgames:
                print("\nRemoving a Bot Environment")

                envs.close()
                envs = self.get_new_bot_envs(args, args.num_bot_envs - 1)
                last_bot_env_change = 0

                agent.remove_last_bot_env()

                bot_obs = bot_obs.zero_()[:, :args.num_bot_envs]
                bot_actions = bot_actions.zero_()[:, :args.num_bot_envs]
                bot_logprobs = bot_logprobs.zero_()[:, :args.num_bot_envs]
                bot_invalid_action_masks = bot_invalid_action_masks.zero_()[:, :args.num_bot_envs]

                rewards_attack = rewards_attack[:, :args.num_envs]
                rewards_attack[:, args.num_selfplay_envs:].zero_()
                rewards_winloss = rewards_winloss[:, :args.num_envs]
                rewards_winloss[:, args.num_selfplay_envs:].zero_()
                rewards_score = rewards_score[:, :args.num_envs]
                rewards_score[:, args.num_selfplay_envs:].zero_()
                # TODO (optimize): muss man die wirklich resetten?
                dones = dones[:, :args.num_envs]
                dones[:, args.num_selfplay_envs:].zero_()
                values = values[:, :args.num_envs]
                values[:, args.num_selfplay_envs:].zero_()

                next_obs_np, _, bot_res = envs.reset()
                bot_next_obs = torch.Tensor(next_obs_np).to(device)

                next_done = next_done[:args.num_envs]
                next_done[args.num_selfplay_envs:].zero_()

                scalar_features = scalar_features[:, :args.num_envs]
                scalar_features[:, args.num_selfplay_envs:].zero_()
                z_features = z_features[:, :args.num_envs]
                z_features[:, args.num_selfplay_envs:].zero_()

                bot_position_indices = bot_position_indices[:args.num_bot_envs]

                print("New number of Bot Environments:", args.num_bot_envs)
                print("")

            elif last_bot_env_change >= 50 and args.num_bot_envs < args.max_num_bot_envs and num_done_selfplaygames * args.bot_adding_done_training_ratio > num_done_botgames:
                print("\nAdding an Bot Environment")

                envs.close()
                envs = self.get_new_bot_envs(args, args.num_bot_envs + 1)
                last_bot_env_change = 0

                agent.add_bot_env()

                bot_obs = torch.zeros((args.num_steps, args.num_bot_envs) + envs.single_observation_space.shape, device=device)
                bot_actions = torch.zeros((args.num_steps, args.num_bot_envs) + action_space_shape, device=device)
                bot_logprobs = torch.zeros((args.num_steps, args.num_bot_envs), device=device)
                bot_invalid_action_masks = torch.zeros((args.num_steps, args.num_bot_envs) + invalid_action_shape, device=device)

                num_added_envs = args.num_envs - rewards_attack.shape[1]

                rewards_attack = torch.cat(
                    (rewards_attack, torch.zeros((args.num_steps, num_added_envs), device=device, dtype=rewards_attack.dtype)), dim=1
                )
                rewards_winloss = torch.cat(
                    (rewards_winloss, torch.zeros((args.num_steps, num_added_envs), device=device, dtype=rewards_winloss.dtype)), dim=1
                )
                rewards_score = torch.cat(
                    (rewards_score, torch.zeros((args.num_steps, num_added_envs), device=device, dtype=rewards_score.dtype)), dim=1
                )
                dones = torch.cat((dones, torch.zeros((args.num_steps, num_added_envs), device=device, dtype=dones.dtype)), dim=1)
                values = torch.cat((values, torch.zeros((args.num_steps, num_added_envs), device=device, dtype=values.dtype)), dim=1)
                rewards_attack[:, args.num_selfplay_envs:].zero_()
                rewards_winloss[:, args.num_selfplay_envs:].zero_()
                rewards_score[:, args.num_selfplay_envs:].zero_()
                # TODO (optimize): muss man die wirklich resetten?
                dones[:, args.num_selfplay_envs:].zero_()
                values[:, args.num_selfplay_envs:].zero_()


                next_obs_np, _, bot_res = envs.reset()
                bot_next_obs = torch.Tensor(next_obs_np).to(device)

                
                next_done = torch.cat((next_done, torch.zeros((num_added_envs), device=device, dtype=next_done.dtype)))
                next_done[args.num_selfplay_envs:].zero_()

                # scalar_features = torch.zeros((args.num_steps, args.num_envs, 11), device=device)
                scalar_features = torch.cat((scalar_features, torch.zeros((args.num_steps, num_added_envs, 11), device=device, dtype=scalar_features.dtype)), dim=1)
                scalar_features[:, args.num_selfplay_envs:].zero_()
                # z_features = torch.zeros((args.num_steps, args.num_envs, 8), dtype=torch.long, device=device)
                z_features = torch.cat((z_features, torch.zeros((args.num_steps, num_added_envs, 8), device=device, dtype=z_features.dtype)), dim=1)
                z_features[:, args.num_selfplay_envs:].zero_()

                bot_position_indices = torch.cat((bot_position_indices, bot_position_indices[:1].clone()))

                print("New number of Bot Environments:", args.num_bot_envs)
                print("")

            writer.add_scalar("charts/num_parallel_Bot_Games", args.num_bot_envs, global_step)

        if cleanup_break:
            cleanup_break()


    # TODO (optimize): in obs, ... die envs entfernen, die man nicht braucht (spart Rechenzeit)
    def get_new_bot_envs(self, args, num_bots):

        args.num_bot_envs = num_bots
        args.num_envs = args.num_selfplay_envs + args.num_bot_envs

        # bring active_league_agents Länge in Einklang mit neuer Env-Anzahl
        if len(self.active_league_agents) < args.num_envs:
            self.active_league_agents.append(self.league_agent)
            self.indices = torch.cat((self.indices, torch.tensor([args.num_envs - 1], device=self.device)))
        else:
            self.active_league_agents = self.active_league_agents[:args.num_envs]
            self.indices = self.indices[:-1]


        


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

    # TODO: Debugging (nachher entfernen)
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

    # TODO: Debugging (nachher entfernen)
    def check_values(self, scalar_features, z_features, values, agent, step, next_scalar_features=None, next_z_features=None, obs=None, next_obs=None, flatten = False):
        return
        if flatten:
            if not torch.allclose(values[step, ::2], agent.get_value(obs[step], scalar_features[step], z_features[step]).flatten()[::2], rtol=1e-3, atol=1e-5):
                print(f"\n\nValue mismatch at (flatten) step: {step}, distance: {values[step, ::2] - agent.get_value(obs[step], scalar_features[step], z_features[step]).flatten()[::2]}\n\n")

        else:
            if not torch.allclose(values[0, ::2], agent.get_value(next_obs, next_scalar_features, next_z_features).reshape(1, -1)[0, ::2], rtol=1e-3, atol=1e-5):
                print(f"\n\nValue mismatch at (reshape) step: {step}, distance: {values[0, ::2] - agent.get_value(next_obs, scalar_features[-1], z_features[-1]).reshape(1, -1)[0, ::2]}\n\n")
