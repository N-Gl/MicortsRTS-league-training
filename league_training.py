from __future__ import annotations

import collections
import os
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, List, Sequence, Tuple
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from jpype.types import JArray, JInt

from agent_model import Agent
from selfplay_only import adjust_action_selfplay, adjust_obs_selfplay, render_all_envs
from log_aggregate_result_table import Logger


class SelfplayAgentType(IntEnum):
    CUR_MAIN = 0    # used, when the current main agent plays against a bot (not for non-selfplaying envs)
    OLD_MAIN = 1
    MAIN_EXPLOITER = 2
    LEAGUE_EXPLOITER = 3


def save_league_model(save_agent, experiment_name=None, dir_name=None, file_name=None):
    os.makedirs(f"league_models/{experiment_name}/{dir_name}", exist_ok=True)
    if isinstance(save_agent, Agent):
        torch.save(save_agent.state_dict(), f"league_models/{experiment_name}/{dir_name}/{file_name}.pt")
    else:
        torch.save(save_agent.agent.state_dict(), f"league_models/{experiment_name}/{dir_name}/{file_name}.pt")


# aus Pseudocode von "Grandmaster level in StarCraft II using multi-agent reinforcement learning"
def remove_monotonic_suffix(win_rates, players):
    # if not win_rates:
    if win_rates is None:
        return win_rates, players

    for i in range(len(win_rates) - 1, 0, -1):
        if win_rates[i - 1] < win_rates[i]:
            return win_rates[: i + 1], players[: i + 1]

    return np.array([]), []


def pfsp(win_rates, weighting="linear", enabled=True):
    weightings = {
        "variance": lambda x: x * (1 - x),
        "linear": lambda x: 1 - x,
        "linear_capped": lambda x: np.minimum(0.5, 1 - x),
        "squared": lambda x: (1 - x) ** 2,
    }
    fn = weightings[weighting]
    if enabled:
        probs = fn(np.asarray(win_rates))
    else:
        probs = np.ones_like(win_rates)
    norm = probs.sum()
    if norm < 1e-10:
        return np.ones_like(win_rates) / len(win_rates)
    return probs / norm


class Payoff:
    def __init__(self):
        # jeder Spieler (auch Historical (inaktive Spieler))
        self._players = []
        self._no_decay_games = collections.defaultdict(lambda: 0)
        self._no_decay_wins = collections.defaultdict(lambda: 0)
        self._no_decay_draws = collections.defaultdict(lambda: 0)
        self._no_decay_losses = collections.defaultdict(lambda: 0)
        self._games = collections.defaultdict(lambda: 0)
        self._wins = collections.defaultdict(lambda: 0)
        self._draws = collections.defaultdict(lambda: 0)
        self._losses = collections.defaultdict(lambda: 0)
        self._games = collections.defaultdict(lambda: 0)
        self._decay = 0.99

    def _win_rate(self, _home, _away):
        if self._games[_home, _away] == 0:
          return 0.5
    
        return (self._wins[_home, _away] +
                0.5 * self._draws[_home, _away]) / self._games[_home, _away]

    def __getitem__(self, match):
        home, away = match
    
        if isinstance(home, Player):
          home = [home]
        if isinstance(away, Player):
          away = [away]
    
        win_rates = np.array([[self._win_rate(h, a) for a in away] for h in home])
        if win_rates.shape[0] == 1 or win_rates.shape[1] == 1:
          win_rates = win_rates.reshape(-1)
    
        return win_rates

    def update(self, home, away, result):
        for stats in (self._games, self._wins, self._draws, self._losses):
          stats[home, away] *= self._decay
          stats[away, home] *= self._decay


        
        self._games[home, away] += 1
        self._no_decay_games[home, away] += 1
        # win:
        if result == 1:
          self._wins[home, away] += 1
          self._no_decay_wins[home, away] += 1
        # Draw:
        elif result == 0:
          self._draws[home, away] += 1
          self._no_decay_draws[home, away] += 1
        # loss:
        else:
            self._losses[home, away] += 1
            self._no_decay_losses[home, away] += 1


        if not away is home:
            self._games[away, home] += 1
            self._no_decay_games[away, home] += 1
            # win:
            if result == 1:
                self._losses[away, home] += 1
                self._no_decay_losses[away, home] += 1
            # Draw:
            elif result == 0:
                self._draws[away, home] += 1
                self._no_decay_draws[away, home] += 1
            # loss:
            else:
                self._wins[away, home] += 1
                self._no_decay_wins[away, home] += 1

    def add_player(self, player):
        self._players.append(player)

    @property
    def players(self):
      return self._players


class Player:

    def get_match(self):
        pass

    def ready_to_checkpoint(self):
        return False

    def _create_checkpoint(self):
        name = getattr(self, 'name', self.__class__.__name__)
        print(f"Creating Historical Checkpoint of {name}")
        return Historical(self, self.payoff, args=self.args, payoff_idx=len(self.payoff.players))
    
    def checkpoint(self):
        raise NotImplementedError
    
    @property
    def payoff(self):
        return self._payoff


class MainPlayer(Player):
    def __init__(
        self,
        agent: torch.nn.Module,
        payoff: Payoff,
        args
    ):
        self.agent = agent
        self._payoff = payoff
        self.args = args

    def _pfsp_branch(self):
        '''sucht einen neuen gegner für selfplay mit pfsp verteilung'''
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="squared", enabled=self.args.pfsp)), True

    def _selfplay_branch(self, opponent):
        '''sucht einen neuen gegner für selfplay, wenn der gegner zu stark ist (winrate gegen ihn < 0.3). Es wird
        ein checkpoint aus der vergangenheit als gegner gewählt mit pfsp Verteilung'''
        # Play self-play match
        if self._payoff[self, opponent] > 0.3:
            return opponent, False

        # If opponent is too strong, look for a checkpoint
        # as curriculum
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="variance", enabled=self.args.pfsp)), True

    def _verification_branch(self, opponent):
        '''sucht einen neuen exploiter gegner für selfplay, wenn min einer der Exploiter self oft self schlägt (winrate gegen ihn < 0.3) (jetzt: 0,35). -> Es wird checkpoint eines Exploiters aus der vergangenheit als gegner gewählt mit pfsp Verteilung.
        Sonst wird ein checkpoint eines historischen gegners gewählt, wenn mainagent gegen min einer der Gegner eine kleinere winrate als 0.7 hat (Sonst None).'''
        # Check exploitation
        exploiters = set([
            player for player in self._payoff.players
            if isinstance(player, MainExploiter)
        ])
        exp_historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent in exploiters
        ]
        win_rates = self._payoff[self, exp_historical]
        if len(win_rates) and win_rates.min() < 0.35:
            return np.random.choice(
                exp_historical, p=pfsp(win_rates, weighting="squared", enabled=self.args.pfsp)), True
        
        # Check forgetting
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]
        win_rates, historical = remove_monotonic_suffix(win_rates, historical)
        if len(win_rates) and win_rates.min() < 0.7:
            return np.random.choice(
                historical, p=pfsp(win_rates, weighting="squared", enabled=self.args.pfsp)), True

        return None

    def get_match(self):
        '''Decides which opponent to play against.
        Main agents are trained with a proportion of 35% SP, 50% PFSP
        against all past players in the league, and an additional 15% of PFSP
        matches against forgotten main players the agent can no longer beat
        and past main exploiters.
        If there are no forgotten players or strong exploiters, the 15% is used for self-play instead.'''
        # TODO (league training): es wird zu oft gegen einfache Gegner gespielt (vorallem, wenn es viele Gegner gibt) 
        # und lange Spiele mit Draws werden so stark bewertet, dass ein klarer win zu einem draw wird.
        coin_toss = np.random.random()

        # Make sure you can beat the League
        # 50% of the time, play PFSP against all historical players (jetzt: 0,7)
        if coin_toss < 0.7:
            return self._pfsp_branch()

        # 35% of the time, play self-play TODO (training): implementiere mehr main Agenten oder passe die Wahrscheinlichkeiten an (jetzt: 0,15)
        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        opponent = np.random.choice(main_agents)

        # Verify if there are some rare players we omitted
        # 15% of PFSP matches against forgotten main players the agent can no longer beat and past main exploiters
        if coin_toss > 1 - 0.15:
            request = self._verification_branch(opponent)
            if request is not None:
                return request

        return self._selfplay_branch(opponent)

    def ready_to_checkpoint(self):
        '''Decides whether the agent is ready to create a new checkpoint. 
        (wenn (min winrate gegen alle historischen gegner > 0.7 und steps_passed >= args.selfplay_save_interval) 
        oder mehr als args.selfplay_save_interval steps vergangen sind)'''
        # weil nur eine Instanz von dem agent für Mainagent ex, ist checkpoint_step in agent gespeichert
        steps_passed = self.agent.get_steps() - self.agent.checkpoint_step
        if steps_passed < (self.args.selfplay_save_interval // 20) * self.args.num_main_agents: # TODO (league training): * args.num_main_agents entfernen, wenn mehrere main agents genutzt werden
          return False

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > self.args.selfplay_save_interval * self.args.num_main_agents # TODO (league training): * args.num_main_agents entfernen, wenn mehrere main agents genutzt werden


    def checkpoint(self):
        '''Creates a new checkpoint of the agent.'''
        self.agent.checkpoint_step = self.agent.get_steps()
        return self._create_checkpoint()


class MainExploiter(Player):
    def __init__(
        self,
        initial_agent: torch.nn.Module,
        payoff: Payoff,
        args,
    ):
        self.args = args
        self.agent = Agent(action_plane_nvec=initial_agent.action_plane_nvec, device=initial_agent.device, initial_weights=initial_agent.state_dict()).to(initial_agent.device)
        self._initial_weights = initial_agent.state_dict()
        self._payoff = payoff
        self._checkpoint_step = 0

    def get_match(self):
        '''wählt einen zufälligen main agenten als gegner, wenn die winrate gegen diesen gegner > 0.1 ist.
        Sonst wird ein historischer checkpoint dieses Gegners gewählt mit pfsp verteilung.'''
        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        opponent = np.random.choice(main_agents)

        if self._payoff[self, opponent] > 0.1:
            return opponent, True

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]
        win_rates = self._payoff[self, historical]

        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="variance", enabled=self.args.pfsp)), True

    def checkpoint(self):
        '''Resets the agent to its initial weights and creates a new checkpoint.'''
        checkpoint = self._create_checkpoint()
        self.agent.set_weights(self._initial_weights)
        self._checkpoint_step = self.agent.get_steps()
        return checkpoint  # TODO: vorher: return self._create_checkpoint(): resetett man die gewichte vor dem checkpoint, sodass der checkpoint immer die gleichen gewichte hat?
    
    def ready_to_checkpoint(self):
        '''Decides whether the agent is ready to create a new checkpoint. wie bei MainPlayer'''
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < self.args.selfplay_save_interval // 20:
            return False

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > self.args.selfplay_save_interval


class LeagueExploiter(Player):
    def __init__(
        self,
        initial_agent: torch.nn.Module,
        payoff: Payoff,
        args,
    ):
        self.args = args
        self.agent = Agent(action_plane_nvec=initial_agent.action_plane_nvec, device=initial_agent.device, initial_weights=initial_agent.state_dict()).to(initial_agent.device)
        self._initial_weights = initial_agent.state_dict()
        self._payoff = payoff
        self._checkpoint_step = 0

    def get_match(self):
        '''wählt einen gegner aus allen historischen gegnern mit pfsp verteilung.'''
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="linear_capped", enabled=self.args.pfsp)), True
    
    def checkpoint(self):
        '''Resets agent zu den initialen gewichten mit 25% chance und erstellt einen neuen checkpoint.'''
        checkpoint = self._create_checkpoint()
        if np.random.random() < 0.25:
            self.agent.set_weights(self._initial_weights)
        self._checkpoint_step = self.agent.get_steps()
        return checkpoint  # TODO: vorher: return self._create_checkpoint(): resetett man die gewichte vor dem checkpoint, sodass der checkpoint immer die gleichen gewichte hat?
    
    def ready_to_checkpoint(self):
        '''Decides whether the agent is ready to create a new checkpoint. wie bei MainPlayer'''
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < self.args.selfplay_save_interval // 20:
            return False
        
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > self.args.selfplay_save_interval
    

class Historical(Player):
    def __init__(
        self,
        parent: Player,
        payoff: Payoff,
        args,
        payoff_idx=None
    ):
        self.agent = Agent(action_plane_nvec=parent.agent.action_plane_nvec, device=parent.agent.device, initial_weights=parent.agent.state_dict()).to(parent.agent.device)
        self._payoff = payoff
        self._parent = parent

        parent_name = getattr(self._parent, 'name', self._parent.__class__.__name__)
        hist_name = getattr(self, 'name', self.__class__.__name__) + f"_{payoff_idx}" if payoff_idx is not None else ""
        save_league_model(save_agent=self, experiment_name=args.exp_name, dir_name=parent_name, file_name=hist_name)

    @property
    def parent(self):
        return self._parent

    def get_match(self):
        raise ValueError("Historical players should not request matches")

    def ready_to_checkpoint(self):
        return False


class League:
    def __init__(
        self,
        initial_agent: torch.nn.Module,
        args,
    ):
        # am Anfang legt man fest, wie viele Environments für das Training von jedem Agententyp genutzt werden (denen gibt man mit .match einen Gegner)
        self._payoff = Payoff()
        self.args = args
        

        # nur aktive Spieler (nicht Historical)
        self._learning_agents =  []
        # for _ in range(args.num_main_agents):
        #   main_agent = MainPlayer(initial_agent, self._payoff, args=args)
        #   self._learning_agents.append(main_agent)

        # only 1 Mainagent:
        main_agent = MainPlayer(initial_agent, self._payoff, args=args)
        for _ in range(args.num_main_agents):
            self._learning_agents.append(main_agent)

        # (TODO (League training): müssen die main_agents ihre Gewichte unterschiedlich updaten können, um besser gegen andere main_agents zu trainieren?
        # gerade ex nur eine Instanz als main_agent (wenn ein main_agent ein update macht, dann updaten alle main_agents ihre Gewichte gleich) 
        # (initial_agents statt initial_agent?))
        self._payoff.add_player(main_agent.checkpoint())
        self._payoff.add_player(main_agent)

        for _ in range(args.num_main_exploiters):
          main_exploiter = MainExploiter(initial_agent, self._payoff, args=args)
          self._learning_agents.append(
              main_exploiter)
          self._payoff.add_player(main_exploiter)
        for _ in range(args.num_league_exploiters):
          league_exploiter = LeagueExploiter(initial_agent, self._payoff, args=args)
          self._learning_agents.append(
              league_exploiter)
          self._payoff.add_player(league_exploiter)

    def update(self, home, away, result):
        return self._payoff.update(home, away, result)

    def get_player(self, idx):
        return self._learning_agents[idx]

    def add_player(self, player):
        self._payoff.add_player(player)

    @property
    def payoff(self):
        return self._payoff

    @property
    def learning_agents(self):
        return self._learning_agents


@dataclass
class LeagueTrainer:
    agent: Agent
    supervised_agent: Agent
    envs: Any
    args: Any
    writer: Any
    device: torch.device
    experiment_name: str
    get_scalar_features: Callable
    action_plane_nvec: Sequence[int]
    checkpoint_frequency: int = 0
    hist_reward = 0
    num_done_selfplaygames = 0
    last_logged_selfplay_games = 0
    num_done_botgames = 0


    def __post_init__(self):
        self.checkpoint_frequency = self.args.checkpoint_frequency


    def _init_agent_type(self):
        agent_type = []
        if self.args.num_selfplay_envs > 0:
            for i in range(self.args.num_main_agents):
                agent_type.append(SelfplayAgentType.CUR_MAIN)
                agent_type.append(-1)

            for i in range(self.args.num_main_exploiters):
                agent_type.append(SelfplayAgentType.MAIN_EXPLOITER) 
                agent_type.append(-1)

            for i in range(self.args.num_league_exploiters):
                agent_type.append(SelfplayAgentType.LEAGUE_EXPLOITER) 
                agent_type.append(-1)
        else:
            agent_type = None

        if self.args.num_selfplay_envs > 0:
            agent_type = torch.tensor(agent_type, dtype=torch.long).to(self.device)
            assert self.args.num_selfplay_envs == len(agent_type), "Number of selfplay envs must be equal to the number of agent types (each agent plays against itself)"
        else:
            assert agent_type is None, "If no selfplay envs are used, agent_type must be None"
        return agent_type

    def _initialize_league(self):

        agent_type = self._init_agent_type()

        # TODO (League training): (don't fill non selfplaying envs) Fokus auf die agenten gibt, gegen nur cur main und alte main: (--FSP)

        league = League(initial_agent=self.agent, args=self.args)
        
        # initiale Environments mit den jeweiligen Gegnern gefüllt
        active_league_agents = []
        assert len(league._learning_agents) == self.args.num_selfplay_envs//2, "Number of learning agents must be half of the number of selfplay envs"
        for idx, player0 in enumerate(league._learning_agents):
            opp = player0.get_match()[0]
            active_league_agents.append(player0)
            active_league_agents.append(opp)

            if isinstance(opp, MainPlayer):
                agent_type[idx * 2 + 1] = SelfplayAgentType.CUR_MAIN
            elif isinstance(opp, LeagueExploiter) or isinstance(opp._parent, LeagueExploiter):
                agent_type[idx * 2 + 1] = SelfplayAgentType.LEAGUE_EXPLOITER
            elif isinstance(opp, MainExploiter) or isinstance(opp._parent, MainExploiter):
                agent_type[idx * 2 + 1] = SelfplayAgentType.MAIN_EXPLOITER
            elif isinstance(opp._parent, MainPlayer):
                agent_type[idx * 2 + 1] = SelfplayAgentType.OLD_MAIN
            else:
                raise ValueError("Unknown agent type")
            
            # wenn active_league_agents[0], active_league_agents[2] Main Agents sind: active_league_agents[0].agent is active_league_agents[2].agent == True

        
        #( TODO: soll ich das machen? fill remaining environments (bot envs) with the main agent reference (sonst muss man selfplay_get_action, selfplay_get_value anpassen))
        while len(active_league_agents) < self.args.num_envs:
            active_league_agents.append(league.learning_agents[0])
        
        return league, active_league_agents, agent_type

    def _on_checkpoint(self):
        self.hist_reward += self.args.new_hist_rewards





    



    def train(self):
        print("League PPO training Setup")
        args = self.args
        device = self.device
        envs = self.envs
        agent: Agent = self.agent
        writer = self.writer
        supervised_agent: Agent = self.supervised_agent or agent

        if args.render:
            if args.render_all:
                from ppo import Rendering

        if args.num_selfplay_envs == 0:
            raise ValueError("league training requires num_selfplay_envs > 0")
        if args.num_main_agents == 0:
            raise ValueError("league training requires at least one main agent")


        league, active_league_agents, agent_type = self._initialize_league()
        

        optimizer = optim.Adam(agent.parameters(), lr=args.PPO_learning_rate, eps=1e-5)
        if args.anneal_lr:
            lr_fn = lambda frac: frac * args.PPO_learning_rate  # noqa: E731
        else:
            lr_fn = None

        mapsize = 16 * 16
        action_space_shape = (mapsize, envs.action_plane_space.shape[0])
        invalid_action_shape = (mapsize, envs.action_plane_space.nvec.sum() + 1)



        obs = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) +
                              action_space_shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # rewards_dense = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_attack = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_winloss = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards_score = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        invalid_action_masks = torch.zeros(
            (args.num_steps, args.num_envs) + invalid_action_shape).to(device)
        

        scalar_features = torch.zeros((args.num_steps, args.num_envs, 11)).to(device)
        z_features = torch.zeros((args.num_steps, args.num_envs, 8), dtype=torch.long).to(device)

        global_step = 0
        start_time = time.time()

        next_obs_np, _, res = envs.reset()
        next_obs = torch.Tensor(next_obs_np).to(device)
        adjust_obs_selfplay(args, next_obs, is_new_env=True)

        next_done = torch.zeros(args.num_envs).to(device)
        next_scalar = self.get_scalar_features(next_obs, res, args.num_envs)

        num_updates = args.total_timesteps // max(args.batch_size, 1)

        position_indices = (
            torch.arange(mapsize, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .repeat(args.num_envs, 1)
            .unsqueeze(2)
        )

        print("League PPO training started")

        for update in range(1, num_updates + 1):

            if lr_fn is not None:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = lr_fn(frac)
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(args.num_steps):
                if args.render:
                    if args.render_all:
                        Rendering.render_all_envs(envs)
                        # render_all_envs(envs)
                    else:
                        envs.render("human")

                global_step += (args.num_selfplay_envs // 2) + args.num_bot_envs
                obs[step] = next_obs
                scalar_features[step] = next_scalar

                # TODO: sollte man das direkt im gleichen step machen? (wie es vorher war (jetzt) wird der nächste schritt als done markiert)
                dones[step] = next_done

                with torch.no_grad():
                    # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                    unique_agents = agent.get_unique_agents(active_league_agents)
                    z_features[step] = agent.selfplay_get_z_encoded_features(args, device, z_features, next_obs, step, unique_agents)

                    # debugging
                    # for i in range(args.num_envs):
                    #     with torch.no_grad():
                    #         # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                    #         old_zFeatures[step][i] = agent.z_encoder(obs[step][i].view(-1))
                    # assert(torch.all(old_zFeatures == zFeatures))



                    # TODO: noch ein mal prüfen?
                    # TODO: nach dem ersten update kommen nicht mehr die gleichen Werte, wie in selfplay_only.py mit selfplay_get_value raus. 
                    # (es werden weiterhin gleiche eingaben gegeben) (Ich habe schon getestet, dass Envs nicht auf sc der anderen Envs zugreifen
                    # fehler im update der gewichte?)
                    values[step] = agent.selfplay_get_value(next_obs, scalar_features[step], z_features[step], 
                                                            num_selfplay_envs=args.num_selfplay_envs, num_envs=args.num_envs, 
                                                            unique_agents=unique_agents,
                                                            only_player_0=True).flatten() # critic(forward(...))
                    
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


                    # gesamplete action (aus Verteilung der Logits) (24, 256, 7),
                    # actor(forward(...)), invalid_action_masks
                    # obs sind zuerst alles 0en, dannach jeweils Spieler 1 zu Spieler 0 geändert
                    action, logproba, _, invalid_action_masks[step] = agent.selfplay_get_action(
                        next_obs,
                        scalar_features[step],
                        z_features[step],
                        args.num_selfplay_envs,
                        args.num_envs,
                        envs=envs,
                        unique_agents=unique_agents
                    )

                # (Shape: (step, num_envs, 256 (16 * 16), action) (step, 24, 256, 7))
                actions[step] = action
                # print("actions shape per step:", actions[step].shape)

                logprobs[step] = logproba

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

                next_obs_np, _, attackrew, winlossrew, scorerew, ds, infos, res = envs.step(java_valid_actions)
                next_obs = torch.Tensor(envs._from_microrts_obs(next_obs_np)).to(device) # next_obs zu Tensor mit shape (24, 16, 16, 73) (von (24, X))
                next_scalar = self.get_scalar_features(next_obs, res, args.num_envs)

                adjust_obs_selfplay(args, next_obs)

                '''winloss = min(0.01, 6.72222222e-9 * global_step)
                densereward = max(0, 0.8 + (-4.44444444e-9 * global_step))

                if global_step < 100000000:
                    scorew = 0.19 + 1.754e-8 * global_step
                else:
                    scorew = 0.5 - 1.33e-8 * global_step'''

                # densereward = 0
                winloss_weight = 10
                scorew = 0.2
                # TODO (training): anpassen
                attack_weight = args.attack_reward_weight
                # =============

                # update rewards
                # rewards_dense[step] = torch.Tensor(denserew* densereward).to(device)



                ### Debugging (scorerews always == 0)
                # Breakpoint if values change
                # if not np.array_equal(scorerews, _last_scorerews):
                #     breakpoint()
                # _last_scorerews = np.copy(scorerews)

                rewards_attack[step] = torch.Tensor(attackrew * attack_weight).to(device)
                rewards_winloss[step] = torch.Tensor(winlossrew * winloss_weight).to(device)
                rewards_score[step] = torch.Tensor(scorerew * scorew).to(device)
                next_done = torch.Tensor(ds).to(device)

                # =============
                # Logging PPO training
                # =============

                if any("episode" in info for info in infos):
                    if not hasattr(writer, "recent_bot_winloss"):
                        writer.recent_bot_winloss = deque(maxlen=200)
                    if not hasattr(writer, "recent_selfplay_winloss"):
                        writer.recent_selfplay_winloss = deque(maxlen=200)
                    
                    where_done = torch.where(next_done.bool())

                    for done_idx in where_done[0].cpu().numpy():
                        done_agent = active_league_agents[done_idx]
                        if done_idx > args.num_selfplay_envs - 1 or done_idx % 2 == 0:
                            done_agent.agent.steps = done_agent.agent.get_steps() + infos[done_idx]["episode"]["l"]

                            if isinstance(done_agent, MainPlayer):
                                dyn_winloss = self._log_general_main_results(writer, global_step, infos, winloss_weight, attack_weight, done_idx)

                        if done_idx > args.num_selfplay_envs - 1:
                            self._log_bot_game_results(args, writer, global_step, infos, attack_weight, done_idx, dyn_winloss)

                        elif done_idx % 2 == 0:
                            # update League match results
                            league.update(active_league_agents[done_idx], active_league_agents[done_idx + 1], infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'])
                            
                            print(f"Game {int(done_idx/2)} ended: {getattr(done_agent, 'name', done_agent.__class__.__name__)} vs {getattr(active_league_agents[done_idx + 1], 'name', active_league_agents[done_idx + 1].__class__.__name__)}, result: {infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction']}")
                            self._log_selfplay_results(args, agent, writer, global_step, infos, done_idx, done_agent, dyn_winloss, attack_weight)

                            if done_agent.ready_to_checkpoint():
                                league.add_player(done_agent.checkpoint())
                                self._on_checkpoint()

                            # neuen gegner in diesem environment auswählen
                            opp = active_league_agents[done_idx + 1] = done_agent.get_match()[0]
                            if isinstance(active_league_agents[1], MainPlayer):
                                agent_type[done_idx + 1] = SelfplayAgentType.CUR_MAIN
                            elif isinstance(active_league_agents[1], LeagueExploiter) or isinstance(active_league_agents[1]._parent, LeagueExploiter):
                                agent_type[done_idx + 1] = SelfplayAgentType.LEAGUE_EXPLOITER
                            elif isinstance(active_league_agents[1], MainExploiter) or isinstance(active_league_agents[1]._parent, MainExploiter):
                                agent_type[done_idx + 1] = SelfplayAgentType.MAIN_EXPLOITER
                            elif isinstance(active_league_agents[1]._parent, MainPlayer):
                                agent_type[done_idx + 1] = SelfplayAgentType.OLD_MAIN
                            else:
                                raise ValueError("Unknown agent type")

                            if isinstance(opp, Historical):
                                opp_name = getattr(opp, "name", opp.__class__.__name__) + "_" + getattr(opp._parent, "name", opp._parent.__class__.__name__)
                            else:
                                opp_name = getattr(opp, "name", opp.__class__.__name__)

                            
                            print(f"New Match in Game {int(done_idx/2)}: {getattr(done_agent, 'name', done_agent.__class__.__name__)} vs {opp_name}")

                # =============
            # =========================



        # =========================
        # PPO update
        # =========================
            with torch.no_grad():
                unique_agents = agent.get_unique_agents(active_league_agents)
                next_z_features = agent.selfplay_get_z_encoded_features(args, device, z_features, next_obs, args.num_steps, unique_agents)

                last_value = agent.selfplay_get_value(
                    next_obs,
                    next_scalar,
                    next_z_features,
                    num_selfplay_envs=args.num_selfplay_envs,
                    num_envs=args.num_envs,
                    unique_agents=unique_agents,
                    only_player_0=True
                ).reshape(1, -1)

                # =============
                # GAE
                # =============
                # TODO (optimize): ich brauche keine Values von den Environments, die nicht in indices sind (oben im Code dasselbe nur für main_agents) 
                # (returns, advantages werden für exploiters weitergegeben, deshalb muss man sie hier auch berechnen oder unten anpassen)
                # oder 2 Variablen jeweils speichern. Hier kann man auch nur die obs, ... zusammenstellen, die exploiters brauchen (spart Speicher)
                advantages = torch.zeros_like(rewards_winloss).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        # zuerst nonterminal, nextvalues aus dem Schritt im Setup
                        # benutzen (anders repräsentiert)
                        nextnonterminal = 1.0 - next_done
                        nextvalues = last_value
                    else:
                        # für jede Umgebung: 1 -> nicht done in step t+1, 0 -> done in
                        # step t+1
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]


                    # TD-Error = R_(t+1) + γ * V(S_(t+1)) - V(S_t) per environment
                    # V(S_t): Value of the state reached in the rollout after Action in step t-1
                    # nextvalues: Critic-approximated values per environment in step t
                    # for the next step in the rollout (if not terminated)
                    # rewards_dense[t] + + rewards_dense[t] +rewards_score[t]
                    # TODO (training): adjust reward for getting new Historical checkpoints (breaks value_loss, total_loss)
                    # delta = args.hist_reward + rewards_winloss[t] + rewards_attack[t] + \
                    #     args.gamma * nextvalues * nextnonterminal - values[t]
                    delta = rewards_winloss[t] + rewards_attack[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    # A_t="TD-Error" + γ * λ * A_(t-1)
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
                # =============

            # flatten the batch
            # (League training): entferne alle Environments, die keine main agenten sind (Exploiter)
            # TODO (training): oder einfach wie in BC von den Exploitern auch trainieren (vielleicht schlechter, da die Exploiter nicht optimal spielen und ein Bias in die Richtung entsteht (nur mit alten main Agenten, weil es gibt so oder so den ratio, der nach
            # Ähnlichkeit zu den eigenen logprobs guckt?))

            # debugging
            #     last = 1
            #     for i in range(0, args.num_selfplay_envs, 2):
            #         cur = active_league_agents[i].agent is agent
            #         assert last >= cur, f"main agents must be in the beginning of the envs (env {i} is main: {active_league_agents[i]}, last was: {active_league_agents[i-2]})"
            #         last = cur


            # TODO (optimize): ich brauche keine obs, mask, actions, logprobs etc. von den player 1 Environments (kann Speicher sparen)

            # dont update Player 1 (TODO: Player 1 can change in an rollout. (is that a problem?))
            # TODO: auch auf Player 1 trainieren (Player 1 darf in einem Rollout sich nicht ändern) (man müsste oben auch die Values für Player 1 berechnen (gerade immer 0))
            indices = torch.tensor(range(args.num_selfplay_envs, args.num_envs), dtype=torch.int).to(device)
            if args.train_on_old_mains: # TODO: Dosnt work, because Player 1 can change in an rollout. (is that a problem?)
                indices = torch.cat((torch.where((agent_type == SelfplayAgentType.CUR_MAIN) | (agent_type == SelfplayAgentType.OLD_MAIN))[0], indices))
            else:
                indices = torch.cat((torch.where(agent_type[0:args.num_selfplay_envs:2] == SelfplayAgentType.CUR_MAIN)[0] * 2, indices))
            
            
            # args.num_steps, args.num_envs Dimensionen vereinigen  (shape (steps*envs, 8))
            # (scalar_features für jeden Step, Environment sortiert Step, dann nach Environments)
            b_z = z_features[:, indices].reshape(-1, z_features.shape[-1])
            # args.num_steps, args.num_envs Dimensionen vereinigen  (shape (steps*envs, 11))
            # (scalar_features für jeden Step, Environment sortiert Step, dann nach Environments)
            b_Sc = scalar_features[:, indices].reshape(-1, scalar_features.shape[-1])
            # dasselbe mit obs                                      (shape (steps*envs, 16, 16, 73))
            b_obs = obs[:, indices].reshape((-1,) + envs.single_observation_space.shape)
            # dasselbe mit actions                                  (shape (steps*envs, 256, 7))
            b_actions = actions[:, indices].reshape((-1,) + action_space_shape)
            # dasselbe mit logprobs, advantages, returns, values    (shape (steps*envs,))
            b_logprobs = logprobs[:, indices].reshape(-1)
            b_advantages = advantages[:, indices].reshape(-1)
            b_returns = returns[:, indices].reshape(-1)
            b_values = values[:, indices].reshape(-1)
            # dasselbe mit invalid_action_masks                     (shape (steps*envs, 256, 79))
            b_invalid_action_masks = invalid_action_masks[:, indices].reshape((-1,) + invalid_action_shape)

            # Optimizing policy and value network with minibatch updates
            # --num_minibatches, --update-epochs
            # minibatches_size = int(args.batch_size // args.num_minibatches)
            # inds: indices from the batch
            new_batch_size = int(len(indices) * args.num_steps)
            minibatch_size = int(new_batch_size // args.n_minibatch) # new (BA Parameter) (minibatch size = 3072 (=(num_envs*num_steps)/ n_minibatch = (24*512)/4))
            # new (BA Parameter) minibatch_size
            inds = np.arange(new_batch_size)

            # Go (update_epochs times) through all mini-batches
            '''
            für jeden Minibatch im Batch berechne Â_t (Advantage Schätzer (R_t^((λ))-V_ϕ^(π_old)) (hier eher (V_ϕ (s_t )-R_t^((λ)) und dann - genommen in pg_loss))) (normalisiert),
            Wahrscheinlichkeit Action a in State s zu bekommen mit neuem θ / Wahrscheinlichkeit Action a in State s zu bekommen mit altem θ_old
            pgLoss (gegenteil von L_clip) ausrechnen, kombinieren mit Entropie Bonus, KL Divergenz Loss und Value Loss mit Updates minimieren
            '''

            epoch_indices = range(args.update_epochs)
            if args.dbg_no_main_agent_ppo_update:
                print("\nDebug: skipping PPO update for main agent\n")
                epoch_indices = []
            for _ in epoch_indices:
                np.random.shuffle(inds)

                for start in range(0, new_batch_size, minibatch_size):
                    end = start + minibatch_size
                    minibatch_ind = inds[start:end]
                    mb_advantages = b_advantages[minibatch_ind]

                    # normalize the advantages
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # forward pass: get network output for the minibatch
                    # We also provide actions here
                    # TODO (league training): muss man hier nicht mehr mit den unique_agents machen? nein, weil nur main agenten im batch sind
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
                    kl_loss = args.kl_coeff * F.kl_div(
                        newlogproba, sl_logprobs, log_target=True, reduction="batchmean"
                    )

                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

        # =========================

            # =========================
            # Logging PPO update
            # =========================

            writer.add_scalar("main_charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("progress/update", update, global_step)

            if not args.dbg_no_main_agent_ppo_update:
                writer.add_scalar("losses/value_loss", args.vf_coef * v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/kl_loss", kl_loss.item(), global_step)
                writer.add_scalar("losses/total_loss", loss.item(), global_step)
                writer.add_scalar("losses/entropy_loss", args.ent_coef * entropy_loss.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)

                if args.prod_mode and update % self.checkpoint_frequency == 0:
                    print("Saving model checkpoint...")
                    save_league_model(save_agent=agent, experiment_name=self.experiment_name, dir_name="current_main_agent", file_name=f"agent")

                    if update < 500:
                        if update % (self.checkpoint_frequency * 5) == 0:
                            save_league_model(save_agent=agent, experiment_name=self.experiment_name, dir_name="Main_agent_backups", file_name=f"agent_update_{update}")

                    else:
                        save_league_model(save_agent=agent, experiment_name=self.experiment_name, dir_name="Main_agent_backups", file_name=f"agent_update_{update}")

                if args.kle_stop or args.kle_rollback:
                    print("uncomment kle_stop and kle_rollback logging if needed")
                #     writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
                # writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
            

            self._train_exploiters(
                league,
                active_league_agents,
                advantages,
                returns,
                values,
                obs,
                scalar_features,
                z_features,
                actions,
                logprobs,
                invalid_action_masks,
                action_space_shape,
                invalid_action_shape,
                lrnow,
                envs,
                writer,
                global_step,
                update
            )

            
            writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
        

        print("Training finished., total steps:", global_step)

    def _train_exploiters(
        self,
        league: League,
        active_league_agents: List[Player],
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        obs: torch.Tensor,
        scalar_features: torch.Tensor,
        z_features: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        invalid_action_masks: torch.Tensor,
        action_space_shape,
        invalid_action_shape,
        lrnow,
        envs,
        writer,
        global_step,
        update
    ):
        
        args = self.args
        env_shape = envs.single_observation_space.shape
        

        # dont update Player 1
        exploiter_indices = []
        for idx in range(0, args.num_selfplay_envs, 2):
            if isinstance(active_league_agents[idx], (LeagueExploiter, MainExploiter)):
                exploiter_indices.append(idx)

        if not exploiter_indices:
            return

        agent_batches = []
        for idx, exploiter_idx in enumerate(exploiter_indices):
            player = active_league_agents[exploiter_idx]
            # TODO (optimize): optimizer außerhalb des training-loops erstellen
            agent_batches.append(
                {
                    "player": player,
                    "optimizer": optim.Adam(player.agent.parameters(), lr=args.PPO_learning_rate, eps=1e-5),
                    "obs": obs[:, exploiter_idx].reshape((-1,) + env_shape),
                    "sc": scalar_features[:, exploiter_idx].reshape(-1, scalar_features.shape[-1]),
                    "z": z_features[:, exploiter_idx].reshape(-1, z_features.shape[-1]),
                    "actions": actions[:, exploiter_idx].reshape((-1,) + action_space_shape),
                    "logprobs": logprobs[:, exploiter_idx].reshape(-1),
                    "advantages": advantages[:, exploiter_idx].reshape(-1),
                    "returns": returns[:, exploiter_idx].reshape(-1),
                    "values": values[:, exploiter_idx].reshape(-1),
                    "masks": invalid_action_masks[:, exploiter_idx].reshape((-1,) + invalid_action_shape)
                }
            )
            agent_batches[idx]["optimizer"].param_groups[0]["lr"] = lrnow
            

        minibatch_size = max(args.num_steps // max(args.n_minibatch, 1), 1)
        inds = np.arange(args.num_steps)

        for _ in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.num_steps, minibatch_size):
                end = start + minibatch_size
                minibatch_ind = inds[start:end]

                batch_obs = []
                batch_sc = []
                batch_z = []
                batch_actions = []
                batch_masks = []
                chunk_sizes = []
                logits_chunks = []
                value_chunks = []

                for entry in agent_batches:
                    mb_obs = entry["obs"][minibatch_ind]
                    mb_sc = entry["sc"][minibatch_ind]
                    mb_z = entry["z"][minibatch_ind]
                    mb_actions = entry["actions"][minibatch_ind]
                    mb_masks = entry["masks"][minibatch_ind]

                    batch_obs.append(mb_obs)
                    batch_sc.append(mb_sc)
                    batch_z.append(mb_z)
                    batch_actions.append(mb_actions)
                    batch_masks.append(mb_masks)
                    chunk_sizes.append(mb_obs.shape[0])

                    current_agent = entry["player"].agent
                    logits_chunks.append(current_agent.actor(current_agent.forward(mb_obs, mb_sc, mb_z)))
                    value_chunks.append(current_agent.get_value(mb_obs, mb_sc, mb_z).view(-1))

                combined_obs = torch.cat(batch_obs, dim=0)
                combined_sc = torch.cat(batch_sc, dim=0)
                combined_z = torch.cat(batch_z, dim=0)
                combined_actions = torch.cat(batch_actions, dim=0)
                combined_masks = torch.cat(batch_masks, dim=0)
                combined_logits = torch.cat(logits_chunks, dim=0)

                _, combined_logprobs, combined_entropy, _ = self.agent.get_action(
                    combined_obs,
                    combined_sc,
                    combined_z,
                    combined_actions.long(),
                    combined_masks,
                    envs,
                    logits=combined_logits
                )

                logprob_splits = torch.split(combined_logprobs, chunk_sizes, dim=0)
                entropy_splits = torch.split(combined_entropy, chunk_sizes, dim=0)

                # free up memory
                del batch_obs, batch_sc, batch_z, batch_actions, batch_masks
                del combined_obs, combined_sc, combined_z, combined_actions, combined_masks
                del logits_chunks, combined_logits, chunk_sizes
                for entry in agent_batches:
                    entry["optimizer"].zero_grad() #(set_to_none=True)?

                total_loss = None
                for entry, new_values, newlogproba, entropy_slice in zip(
                    agent_batches, value_chunks, logprob_splits, entropy_splits
                ):
                    mb_advantages = entry["advantages"][minibatch_ind]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    ratio = (newlogproba - entry["logprobs"][minibatch_ind]).exp()

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy_slice.mean()

                    if args.clip_vloss:
                        v_loss_unclipped = (new_values - entry["returns"][minibatch_ind]) ** 2
                        v_clipped = entry["values"][minibatch_ind] + torch.clamp(
                            new_values - entry["values"][minibatch_ind], -args.clip_coef, args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - entry["returns"][minibatch_ind]) ** 2
                        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        v_loss = 0.5 * ((new_values - entry["returns"][minibatch_ind]) ** 2)

                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                    total_loss = loss if total_loss is None else total_loss + loss

                del logprob_splits, entropy_splits

                if total_loss is not None:
                    total_loss.backward()
                    for entry in agent_batches:
                        torch.nn.utils.clip_grad_norm_(entry["player"].agent.parameters(), args.max_grad_norm)
                        entry["optimizer"].step()
                    del total_loss

                del value_chunks
                # TODO: sollte ich wirklich nach jedem minibatch den GPU cache leeren?
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()

        for i, entry in enumerate(agent_batches):

            # TODO: updaten sich die Gewichte (Veränderung = True)
            # import copy
            # old_w = {k: v.detach().clone() for k, v in self.agent.state_dict().items()}
            # is_changing = {k: not torch.equal(v, old_w[k]) for k, v in self.agent.state_dict().items()}

            
            name = getattr(entry["player"], "name", entry["player"].__class__.__name__) + "_in_env_" + str(exploiter_indices[i])

            writer.add_scalar(f"{name}/learning_rate", entry["optimizer"].param_groups[0]["lr"], global_step)
            writer.add_scalar(f"{name}/value_loss", args.vf_coef * v_loss.item(), global_step)
            writer.add_scalar(f"{name}/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar(f"{name}/total_loss", loss.item(), global_step)
            writer.add_scalar(f"{name}/entropy_loss", args.ent_coef * entropy_loss.item(), global_step)

            if args.prod_mode and update % self.checkpoint_frequency == 0:
                    print("Saving model checkpoint...")
                    save_league_model(save_agent=entry["player"].agent, experiment_name=self.experiment_name, dir_name=f"current_{getattr(entry['player'], 'name', entry['player'].__class__.__name__)}", file_name=f"{name}")

                    if update < 500:
                        if update % (self.checkpoint_frequency * 5) == 0:
                            save_league_model(save_agent=entry["player"].agent, experiment_name=self.experiment_name, dir_name=getattr(entry["player"], "name", entry["player"].__class__.__name__), file_name=f"{name}_update_{update}")
                            
                    else:
                        save_league_model(save_agent=entry["player"].agent, experiment_name=self.experiment_name, dir_name=getattr(entry["player"], "name", entry["player"].__class__.__name__), file_name=f"{name}_update_{update}")

    def _log_general_main_results(self, writer, global_step, infos, winloss_weight, attack_weight, done_idx):
        game_length = infos[done_idx]["episode"]["l"]

        dyn_winloss = winloss_weight * (-0.00013 * game_length + 1.16)  # ca. 0.9 bei 2000 und 1.1 bei 500 TODO (training): für die ersten 3 millionen steps nur, wenn man gewinnt == 0?
        writer.add_scalar("main_charts/old_episode_reward", infos[done_idx]['episode']['r'], global_step)
        writer.add_scalar("main_charts/Game_length", game_length, global_step)
        writer.add_scalar("main_charts/Episode_reward_with_hist_reward", self.hist_reward + infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss + 
                                              infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight, global_step)
        writer.add_scalar("main_charts/Episode_reward", infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss + 
                                              infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight, global_step)
        writer.add_scalar("main_charts/AttackReward", infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight, global_step)
        writer.add_scalar("main_charts/WinLossRewardFunction", infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss, global_step)
        return dyn_winloss
        
    def _log_bot_game_results(self, args, writer, global_step, infos, attack_weight, done_idx, dyn_winloss):
        writer.recent_bot_winloss.append(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'])
        bot_winrate = np.mean(np.clip(writer.recent_bot_winloss, 0, 1))
        with_draw = np.mean(np.add(writer.recent_bot_winloss, 1) / 2)
        winloss_values = np.array(np.clip(writer.recent_bot_winloss, 0, 1))
        writer.add_scalar(f"main_winrates/bot_Winrate_with_Draw_0", bot_winrate, self.num_done_botgames)
        writer.add_scalar(f"main_winrates/bot_Winrate_std", np.std(winloss_values), self.num_done_botgames)
        writer.add_scalar(f"main_winrates/bot_Winrate_with_draw_0.5", with_draw, self.num_done_botgames)
        writer.add_scalar("progress/num_bot_games", self.num_done_botgames, self.num_done_botgames)
        print(f"global_step={global_step}, episode_reward={(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss + infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight):.3f}, bot_winrate_{len(writer.recent_bot_winloss)}={bot_winrate:.3f}, bot_winrate_with_draw_0.5_{len(writer.recent_bot_winloss)}={with_draw:.3f}")
        print(f"match in Botgame {int(done_idx - (args.num_selfplay_envs - 1))}, result: {infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction']}")
        self.num_done_botgames += 1


    def _log_selfplay_results(self, args, agent, writer, global_step, infos, done_idx, done_agent, dyn_winloss, attack_weight):
        selfplay_winrate = None
        selfplay_with_draw = None
        if isinstance(done_agent, MainPlayer):
            writer.recent_selfplay_winloss.append(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'])
            selfplay_winrate = np.mean(np.clip(writer.recent_selfplay_winloss, 0, 1))
            selfplay_with_draw = np.mean(np.add(writer.recent_selfplay_winloss, 1) / 2)
            winloss_values = np.array(writer.recent_selfplay_winloss)
            writer.add_scalar(f"main_winrates/selfplay_Winrate_no_draw", selfplay_winrate, self.num_done_selfplaygames)
            writer.add_scalar(f"main_winrates/selfplay_Winrate_with_draw", selfplay_with_draw, self.num_done_selfplaygames)
            writer.add_scalar(f"main_winrates/selfplay_Winrate_no_draw_std", np.std(winloss_values), self.num_done_selfplaygames)
            writer.add_scalar("progress/num_selfplay_games", self.num_done_selfplaygames, self.num_done_selfplaygames)
        self.num_done_selfplaygames += 1
                            
        # TODO (league training): auch andere Agents loggen? (wäre pro exploiter pro Gegner eine Zeile in der Tabelle)
        if (self.num_done_selfplaygames < 10 or self.last_logged_selfplay_games + 25 <= self.num_done_selfplaygames) and isinstance(done_agent, MainPlayer):
            self.last_logged_selfplay_games = self.num_done_selfplaygames
            win_rates = []
            opp_names = []
            game_count = []
            opponent_table_rows: List[Tuple] = []
            count_league_players = 0
            for i, p1 in enumerate(done_agent.payoff.players):
                if not isinstance(p1, LeagueExploiter): # league exploiters will be turned to historicals bevore playing against main agents
                    if isinstance(p1, Historical):
                        name = getattr(p1, "name", p1.__class__.__name__) + "_" + getattr(p1._parent, "name", p1._parent.__class__.__name__)+ "_" + str(i)
                    else:
                        name = getattr(p1, "name", p1.__class__.__name__) + "_" + str(i)
                                        
                    opp_names.append(name)
                    game_count.append(done_agent._payoff._games[done_agent, p1])

                    win_rates.append(done_agent.payoff._win_rate(done_agent, p1))

                                        
                    wins = done_agent.payoff._wins[done_agent, p1]
                    losses = done_agent.payoff._losses[done_agent, p1]
                    draws = done_agent.payoff._draws[done_agent, p1]

                    only_win_rate = wins / game_count[i-count_league_players] if game_count[i-count_league_players] > 0 else 0
                    draw_rate = draws / game_count[i-count_league_players] if game_count[i-count_league_players] > 0 else 0
                    loss_rate = losses / game_count[i-count_league_players] if game_count[i-count_league_players] > 0 else 0
                                        
                    opponent_table_rows.append((name,
                                                                    done_agent.payoff._no_decay_games[done_agent, p1],
                                                                    done_agent.payoff._no_decay_wins[done_agent, p1],
                                                                    done_agent.payoff._no_decay_draws[done_agent, p1],
                                                                    done_agent.payoff._no_decay_losses[done_agent, p1],
                                                                    only_win_rate,
                                                                    draw_rate,
                                                                    loss_rate
                                                                    ))
                else:
                    count_league_players += 1
                                    
            if done_agent.agent is agent:
                self_name = getattr(done_agent, "name", done_agent.__class__.__name__)
            else:
                self_name = getattr(done_agent, "name", done_agent.__class__.__name__) + "_in_env_" + str(done_idx)

            Logger.log_wandb_summary(
                                    args,
                                    opponent_table_rows,
                                    no_reward=True,
                                    step=global_step,
                                    table_name="league/games_summary",
                                    with_name=self_name
                                    )

            self_name = getattr(done_agent, "name", done_agent.__class__.__name__)
            for opp, games, r in zip(opp_names, game_count, win_rates):
                if games > 0:
                    writer.add_scalar(f"winrate_per_opponent/{self_name}_vs_{opp}", r, games)


        # print(f"{self_name} Win rates against all opponents: {list(zip(opp_names, rates))}")
        if selfplay_winrate is not None:
            print(f"global_step={global_step}, episode_reward={(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss + infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight):.3f}, selfplay_winrate_no_draw_{len(writer.recent_selfplay_winloss)}={selfplay_winrate:.3f}, selfplay_winrate_with_draw_0.5_{len(writer.recent_selfplay_winloss)}={selfplay_with_draw:.3f}")




# =========================
