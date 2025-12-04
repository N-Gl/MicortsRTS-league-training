from collections import deque
import os
import time
from typing import Callable

import numpy as np
import torch
from jpype.types import JArray, JInt

from agent_model import Agent



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
                # TODO: aus 25 wurde 26 funktioniert jetzt auch in League?
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

class SelfPlayTrainer:
    def __init__(
        self,
        agent,
        supervised_agent,
        envs,
        args,
        writer,
        device: torch.device,
        experiment_name: str,
        get_scalar_features: Callable,
        checkpoint_frequency: int = 10,
    ):
        self.agent = agent
        self.supervised_agent = supervised_agent
        self.envs = envs
        self.args = args
        self.writer = writer
        self.device = device
        self.experiment_name = experiment_name
        self.get_scalar_features = get_scalar_features
        self.checkpoint_frequency = checkpoint_frequency
        self.active_league_agents = []
        self.league_agent = Selfplay_agent(agent)
        self.league_supervised_agent = Selfplay_agent(supervised_agent)
        for _ in range(args.num_main_agents):
            self.active_league_agents.append(self.league_agent)
            self.active_league_agents.append(self.league_supervised_agent)
        for _ in range(args.num_bot_envs):
            self.active_league_agents.append(self.league_agent)
        self.indices = torch.tensor(range(args.num_selfplay_envs, args.num_envs), dtype=torch.int).to(device)
        self.indices = torch.cat((torch.tensor(range(0, args.num_selfplay_envs, 2)).to(device), self.indices))

    def train(self):
        args = self.args
        num_done_botgames = 0
        num_done_selfplaygames = 0
        agent: Agent = self.agent
        envs = self.envs
        writer = self.writer
        device = self.device
        # supervised_agent = self.supervised_agent or Agent(agent.action_plane_nvec, agent.device, initial_weights=agent.state_dict())
        supervised_agent = self.supervised_agent or Agent(agent.action_plane_nvec, agent.device, initial_weights=agent.state_dict())

        if args.render:
            if args.render_all:
                from ppo import Rendering

        optimizer = torch.optim.Adam(agent.parameters(), lr=args.PPO_learning_rate, eps=1e-5)
        if args.anneal_lr:
            lr_fn = lambda frac: frac * args.PPO_learning_rate  # noqa: E731
        else:
            lr_fn = None

        mapsize = 16 * 16
        action_space_shape = (mapsize, envs.action_plane_space.shape[0])
        invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum() + 1)

        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_attack = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_winloss = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_score = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)

        global_step = 0
        start_time = time.time()
        next_obs_np, _, res = envs.reset()

        next_obs = torch.Tensor(next_obs_np).to(device)

        adjust_obs_selfplay(args, next_obs, is_new_env=True)
        next_done = torch.zeros(args.num_envs).to(device)

        scalar_features = torch.zeros((args.num_steps, args.num_envs, 11)).to(device)
        z_features = torch.zeros((args.num_steps, args.num_envs, 8), dtype=torch.long).to(device)

        num_updates = args.total_timesteps // args.batch_size
        position_indices = (
            torch.arange(mapsize, device=device, dtype=torch.int64).unsqueeze(0).repeat(args.num_envs, 1).unsqueeze(2)
        )

        

        print("PPO training started")
        

        for update in range(1, num_updates + 1):

            if lr_fn is not None:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = lr_fn(frac)

            for step in range(args.num_steps):
                if args.render:
                    if args.render_all:
                        Rendering.render_all_envs(envs)
                    else:
                        envs.render("human")
                        
                with torch.no_grad():
                    # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                    z_features[step] = agent.z_encoder(next_obs.view(args.num_envs, -1))

                    # debugging
                    # for i in range(args.num_envs):
                    #     with torch.no_grad():
                    #         # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                    #         old_zFeatures[step][i] = agent.z_encoder(obs[step][i].view(-1))
                    # assert(torch.all(old_zFeatures == zFeatures))

                # TODO: global_step += (args.num_main_agents) + args.num_bot_envs
                global_step += (args.num_main_agents // 2) + args.num_bot_envs
                obs[step] = next_obs
                scalar_features[step] = self.get_scalar_features(next_obs, res, args.num_envs).to(device)
                dones[step] = next_done

                with torch.no_grad():

                    # critic(forward(...))
                    # # values[step] = agent.get_value(obs[step, self.indices], scalar_features[step, self.indices], z_features[step, self.indices]).flatten()
                    # values[step] = agent.get_value(obs[step], scalar_features[step], z_features[step]).flatten()
                    unique_agents = agent.get_unique_agents(self.active_league_agents)

                    values[step] = agent.selfplay_get_value(obs[step],
                                                            scalar_features[step],
                                                            z_features[step],
                                                            num_selfplay_envs=args.num_selfplay_envs,
                                                            num_envs=args.num_envs,
                                                            unique_agents=unique_agents,
                                                            only_player_0=True).flatten()
                    

                    self.check_values(scalar_features, z_features, values, agent, step, obs=obs, flatten=True)

                    # gesamplete action (aus Verteilung der Logits) (24, 256, 7),
                    # actor(forward(...)), invalid_action_masks
                    # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                    action, logprob, _, invalid_action_masks[step] = agent.selfplay_get_action(
                        obs[step],
                        scalar_features[step],
                        z_features[step],
                        num_selfplay_envs=args.num_selfplay_envs,
                        num_envs=args.num_envs,
                        envs=envs,
                        active_league_agents=self.active_league_agents
                    )

                # (Shape: (step, num_envs, 256 (16 * 16), action) (step, 24, 256, 7))
                actions[step] = action
                # print("actions shape per step:", actions[step].shape)
                logprobs[step] = logprob

                # Die Grid-Position zu jedem Action hinzugefügt (24, 256, 8)
                real_action = torch.cat([position_indices, action], dim=2).cpu().numpy()
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
                valid_actions = real_action[invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
                valid_counts = invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()

                # Anpassungen für Spieler 1 nach (Spieler 1 -> Spieler 0)
                adjust_action_selfplay(args, valid_actions, valid_counts)

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

                java_valid_actions = []
                valid_index = 0
                for count in valid_counts:
                    java_env_action = []
                    for _ in range(count):
                        java_env_action.append(JArray(JInt)(valid_actions[valid_index]))
                        valid_index += 1
                    java_valid_actions.append(JArray(JArray(JInt))(java_env_action))
                java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)
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

                next_obs, _, attackrew, winlossrew, scorerew, ds, infos, res = envs.step(java_valid_actions)
                next_obs = torch.Tensor(envs._from_microrts_obs(next_obs)).to(device) # next_obs zu Tensor mit shape (24, 16, 16, 73) (von (24, X))
                
                adjust_obs_selfplay(args, next_obs)

                '''winloss = min(0.01, 6.72222222e-9 * global_step)
                densereward = max(0, 0.8 + (-4.44444444e-9 * global_step))

                if global_step < 100000000:
                    scorew = 0.19 + 1.754e-8 * global_step
                else:
                    scorew = 0.5 - 1.33e-8 * global_step'''


                # densereward = 0
                winloss = 10
                scorew = 0.2
                attack = args.attack_reward_weight
                # =============

                # update rewards
                # rewards_dense[step] = torch.Tensor(denserew* densereward).to(device)

                ### Debugging (scorerews always == 0)
                # Breakpoint if values change
                # if not np.array_equal(scorerews, _last_scorerews):
                #     breakpoint()
                # _last_scorerews = np.copy(scorerews)

                rewards_attack[step] = torch.Tensor(attackrew * attack).to(device)
                rewards_winloss[step] = torch.Tensor(winlossrew * winloss).to(device) # TODO: würde hier +1 gegen viele Draws helfen?
                rewards_score[step] = torch.Tensor(scorerew * scorew).to(device)
                next_done = torch.Tensor(ds).to(device)

                # =============
                # Logging PPO training
                # =============
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
                                writer.recent_bot_winloss = deque(maxlen=200)
                    if not hasattr(writer, "recent_selfplay_winloss"):
                                writer.recent_selfplay_winloss = deque(maxlen=200)

                    for done_idx in where_done[0]:
                        if done_idx > args.num_selfplay_envs - 1:
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

                        break
                # =============
            # =========================


        # =========================
        # PPO update
        # =========================

            with torch.no_grad():
                # next_value = agent.get_value(next_obs, scalar_features[-1], z_features[-1]).reshape(1, -1)
                unique_agents = agent.get_unique_agents(self.active_league_agents)

                next_value = agent.selfplay_get_value(
                    next_obs,
                    scalar_features[-1],
                    z_features[-1],
                    num_selfplay_envs=args.num_selfplay_envs,
                    num_envs=args.num_envs,
                    unique_agents=unique_agents,
                    only_player_0=True
                ).reshape(1, -1)

                self.check_values(scalar_features, z_features, next_value, agent, step, next_obs=next_obs, flatten=False)

                # dont update supervised_agent
                b_next_value = next_value[:, self.indices]
                b_values = values[:, self.indices]
                b_rewards_attack = rewards_attack[:, self.indices]
                b_rewards_winloss = rewards_winloss[:, self.indices]
                b_dones = dones[:, self.indices]
                b_next_done = next_done[self.indices]

                # =============
                # GAE
                # =============

                b_advantages = torch.zeros_like(b_rewards_winloss).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        # zuerst nonterminal, nextvalues aus dem Schritt im Setup
                        # benutzen (anders repräsentiert)
                        nextnonterminal = 1.0 - b_next_done
                        nextvalues = b_next_value
                    else:
                        # für jede Umgebung: 1 -> nicht done in step t+1, 0 -> done in
                        # step t+1
                        nextnonterminal = 1.0 - b_dones[t + 1]
                        nextvalues = b_values[t + 1]
                    
                    # TD-Error = R_(t+1) + γ * V(S_(t+1)) - V(S_t) per environment
                    # V(S_t): Value of the state reached in the rollout after Action in step t-1
                    # nextvalues: Critic-approximated values per environment in step t
                    # for the next step in the rollout (if not terminated)
                    # rewards_dense[t] + + rewards_dense[t] +rewards_score[t]
                    delta = b_rewards_winloss[t] + b_rewards_attack[t] + args.gamma * nextvalues * nextnonterminal - b_values[t]
                    # A_t="TD-Error" + γ * λ * A_(t-1)
                    b_advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                b_returns = b_advantages + b_values
                # =============


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
            # b_advantages = advantages[:, self.indices].reshape(-1)
            b_advantages = b_advantages.reshape(-1)
            # b_returns = returns[:, self.indices].reshape(-1)
            b_returns = b_returns.reshape(-1)
            # b_values = values[:, self.indices].reshape(-1)
            b_values = b_values.reshape(-1)
            # dasselbe mit invalid_action_masks                     (shape (steps*envs, 256, 79))
            b_invalid_action_masks = invalid_action_masks[:, self.indices].reshape((-1,) + invalid_action_shape)
            

            new_batch_size = int(len(self.indices) * args.num_steps)
            minibatch_size = int(new_batch_size // args.n_minibatch) # new (BA Parameter) (minibatch size = 3072 (=(num_envs*num_steps)/ n_minibatch = (24*512)/4))

            # Optimizing policy and value network with minibatch updates
            # --num_minibatches, --update-epochs
            # minibatches_size = int(args.batch_size // args.num_minibatches)
            # inds: self.indices from the batch
            inds = np.arange(new_batch_size)

            for _ in range(args.update_epochs):
                np.random.shuffle(inds)

                for start in range(0, new_batch_size, minibatch_size):
                    end = start + minibatch_size
                    minibatch_ind = inds[start:end]
                    mb_advantages = b_advantages[minibatch_ind]

                    if args.norm_adv:
                        # normalize the advantages
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # forward pass: get network output for the minibatch
                    # We also provide actions here
                    new_values = agent.get_value(b_obs[minibatch_ind], b_Sc[minibatch_ind], b_z[minibatch_ind]).view(-1)

                    # get_action nur für logprobs und entropy, um ratio zu berechnen (um zu vergleichen, wie wahrscheinlich die Action mit dem neuen θ im Vergleich zu dem alten θ_old ist)
                    _, newlogproba, entropy, _ = agent.get_action(
                        b_obs[minibatch_ind],
                        b_Sc[minibatch_ind],
                        b_z[minibatch_ind],
                        b_actions.long()[minibatch_ind],
                        b_invalid_action_masks[minibatch_ind],
                        envs,
                    )
                    ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                    # for logging
                    approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

                    # Policy loss L^CLIP(θ) = E ̂_t ["min" (r_t (θ)*Â_t,"clip" (r_t (θ),1-ϵ,1+ϵ)*Â_t )]
                    # --clip-coef
                    # pg_loss = -L^CLIP(θ) (opposite)
                    # it is the same (but negative), because in loss1 and 2 there is a minus sign and advantages are calculated differently
                    # geht gegen 0, wenn es keine Verbesserung mehr gibt
                    # gibt ein Wert für die Verbesserung der Policy in der aktuellen Iteration an
                    # < 0  ⇒ Surrogate im Mittel verbessert (guter Update) (Policy verbessert sich)
                    # ≈ 0  ⇒ kaum/keine (geclippte) Verbesserung
                    # > 0  ⇒ Surrogate im Mittel schlechter (schlechter Update)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()


                    # Value loss Clipping
                    # --clip_vloss
                    # MSE(approximierte Values, returns) with or without clip()
                    if args.clip_vloss:
                        v_loss_unclipped = (new_values - b_returns[minibatch_ind]) ** 2
                        v_clipped = b_values[minibatch_ind] + torch.clamp(
                            new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

                    # KL Divergence Loss
                    with torch.no_grad():
                        # get_action nur für logprobs, um KL Divergenz zu berechnen
                        _, sl_logprobs, _, _ = supervised_agent.get_action(
                            b_obs[minibatch_ind],
                            b_Sc[minibatch_ind],
                            b_z[minibatch_ind],
                            b_actions.long()[minibatch_ind],
                            b_invalid_action_masks[minibatch_ind],
                            envs,
                        )
                    kl_loss = args.kl_coeff * torch.nn.functional.kl_div(
                        newlogproba, sl_logprobs, log_target=True, reduction="batchmean"
                    )

                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    self.assert_supervised_grads_zero(supervised_agent)
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("charts/update", update, global_step)
            writer.add_scalar("losses/value_loss", args.vf_coef * v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/kl_loss", kl_loss.item(), global_step)
            writer.add_scalar("losses/total_loss", loss.item(), global_step)
            writer.add_scalar("losses/entropy_loss", args.ent_coef * entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
            # if args.kle_stop or args.kle_rollback:
            #     writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
            writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))

            if args.prod_mode and update % self.checkpoint_frequency == 0:
                if (update < 500):
                    if (update % (self.checkpoint_frequency * 5) == 0):
                        save_league_model(save_agent=agent, experiment_name=self.experiment_name, dir_name="Main_agent_backups", file_name=f"agent_update_{update}")
                else:
                    save_league_model(save_agent=agent, experiment_name=self.experiment_name, dir_name="Main_agent_backups", file_name=f"agent_update_{update}")



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
    def check_values(self, scalar_features, z_features, values, agent, step, obs=None, next_obs=None, flatten = False):
        return
        if flatten:
            if not torch.allclose(values[step, ::2], agent.get_value(obs[step], scalar_features[step], z_features[step]).flatten()[::2], rtol=1e-3, atol=1e-5):
                print("\n\nValue mismatch at (flatten) step\n\n", step)

        else:
            if not torch.allclose(values[0, ::2], agent.get_value(next_obs, scalar_features[-1], z_features[-1]).reshape(1, -1)[0, ::2], rtol=1e-3, atol=1e-5):
                print("\n\nValue mismatch at (reshape) step\n\n", step)