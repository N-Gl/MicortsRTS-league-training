import collections
import numpy as np
import torch
import os
from enum import IntEnum
from typing import List, Tuple

from agent_model import Agent
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

def _init_agent_type(args, device):
    agent_type = []
    if args.num_selfplay_envs > 0:
        for i in range(args.num_main_envs):
            agent_type.append(SelfplayAgentType.CUR_MAIN)
            agent_type.append(-1)
        for i in range(args.num_main_exploiters):
            agent_type.append(SelfplayAgentType.MAIN_EXPLOITER) 
            agent_type.append(-1)
        for i in range(args.num_league_exploiters):
            agent_type.append(SelfplayAgentType.LEAGUE_EXPLOITER) 
            agent_type.append(-1)
    else:
        agent_type = None
    if args.num_selfplay_envs > 0:
        agent_type = torch.tensor(agent_type, dtype=torch.long).to(device)
        assert args.num_selfplay_envs == len(agent_type), "Number of selfplay envs must be equal to the number of agent types (each agent plays against itself)"
    else:
        assert agent_type is None, "If no selfplay envs are used, agent_type must be None"
    return agent_type

# TODO: rufe die Methode richtig auf
def _on_checkpoint(hist_reward, args):
    hist_reward += args.new_hist_rewards
     

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
        if self.args.sp:
            return self._payoff.players[0], True
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
        (wenn (min winrate gegen alle historischen gegner > 0.7 und steps_passed >= args.selfplay_ready_save_interval) 
        oder mehr als args.selfplay_save_interval steps vergangen sind)'''
        # weil nur eine Instanz von dem agent für Mainagent ex, ist checkpoint_step in agent gespeichert
        steps_passed = self.agent.get_steps() - self.agent.checkpoint_step
        if steps_passed < (self.args.selfplay_ready_save_interval) * self.args.num_main_envs: # TODO (league training): * args.num_main_envs entfernen, wenn mehrere main agents genutzt werden
          return False

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > self.args.selfplay_save_interval // (self.args.num_selfplay_envs // 2 + self.args.num_bot_envs) * self.args.num_main_envs # TODO (league training): * args.num_main_envs entfernen, wenn mehrere main agents genutzt werden


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
        optimizer=None
    ):
        self.args = args
        self.agent = Agent(action_plane_nvec=initial_agent.action_plane_nvec, device=initial_agent.device, initial_weights=initial_agent.state_dict()).to(initial_agent.device)
        self._initial_weights = {k: v.detach().clone() for k, v in initial_agent.state_dict().items()}
        # self._initial_weights = initial_agent.state_dict() und copy later -> to reset the exploiter back to the main agent weights after each checkpoint.
        self._payoff = payoff
        self._checkpoint_step = 0
        self.num_resets_checkpoints = 0
        self.optimizer = optimizer

    def get_match(self):
        '''wählt einen zufälligen main agenten als gegner, wenn die winrate gegen diesen gegner > 0.1 ist.
        Sonst wird ein historischer checkpoint dieses Gegners gewählt mit pfsp verteilung.'''
        
        main_agents = [
            player for player in self._payoff.players
            if isinstance(player, MainPlayer)
        ]
        opponent = np.random.choice(main_agents)

        if self._payoff[self, opponent] > 0.1 and not self.args.sp:
            return opponent, True

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical) and player.parent == opponent
        ]

        if self.args.sp:
            opp = historical[0]
            if not isinstance(opp.parent, MainPlayer):
                print("Warning: In selfplay mode, the expoiter is playing against a historical of a different agent than main player.")
            return opp, True
        
        win_rates = self._payoff[self, historical]

        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="variance", enabled=self.args.pfsp)), True

    def checkpoint(self):
        '''Resets the agent to its initial weights and creates a new checkpoint.'''
        checkpoint = self._create_checkpoint()

        self.agent.set_weights(self._initial_weights)
        self.optimizer = None

        self._checkpoint_step = self.agent.get_steps()
        return checkpoint  # TODO: vorher: return self._create_checkpoint(): resetett man die gewichte vor dem checkpoint, sodass der checkpoint immer die gleichen gewichte hat?
    
    def ready_to_checkpoint(self):
        '''Decides whether the agent is ready to create a new checkpoint. wie bei MainPlayer'''
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < self.args.selfplay_ready_save_interval:
            return False

        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > self.args.selfplay_save_interval // (self.args.num_selfplay_envs // 2 + self.args.num_bot_envs)  * self.args.num_envs_per_main_exploiters


class LeagueExploiter(Player):
    def __init__(
        self,
        initial_agent: torch.nn.Module,
        payoff: Payoff,
        args,
        optimizer=None
    ):
        self.args = args
        self.agent = Agent(action_plane_nvec=initial_agent.action_plane_nvec, device=initial_agent.device, initial_weights=initial_agent.state_dict()).to(initial_agent.device)
        self._initial_weights = {k: v.detach().clone() for k, v in initial_agent.state_dict().items()}
        self._payoff = payoff
        self._checkpoint_step = 0
        self.num_resets_checkpoints = 0
        self.optimizer = optimizer

    def get_match(self):
        '''wählt einen gegner aus allen historischen gegnern mit pfsp verteilung.'''

        
        
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]

        if self.args.sp:
            opp = historical[0]
            if not isinstance(opp.parent, MainPlayer):
                print("Warning: In selfplay mode, the expoiter is playing against a historical of a different agent than main player.")
            return opp, True

        win_rates = self._payoff[self, historical]
        return np.random.choice(
            historical, p=pfsp(win_rates, weighting="linear_capped", enabled=self.args.pfsp)), True
    
    def checkpoint(self):
        '''Resets agent zu den initialen gewichten mit 25% chance und erstellt einen neuen checkpoint.'''
        checkpoint = self._create_checkpoint()

        if np.random.random() < 0.25:
            self.agent.set_weights(self._initial_weights)
            self.optimizer = None

        self._checkpoint_step = self.agent.get_steps()
        return checkpoint  # TODO: vorher: return self._create_checkpoint(): resetett man die gewichte vor dem checkpoint, sodass der checkpoint immer die gleichen gewichte hat?
    
    def ready_to_checkpoint(self):
        '''Decides whether the agent is ready to create a new checkpoint. wie bei MainPlayer'''
        steps_passed = self.agent.get_steps() - self._checkpoint_step
        if steps_passed < self.args.selfplay_ready_save_interval:
            return False
        
        historical = [
            player for player in self._payoff.players
            if isinstance(player, Historical)
        ]
        win_rates = self._payoff[self, historical]
        return win_rates.min() > 0.7 or steps_passed > self.args.selfplay_save_interval // (self.args.num_selfplay_envs // 2 + self.args.num_bot_envs) * self.args.num_envs_per_league_exploiters
    

class Historical(Player):
    def __init__(
        self,
        parent: Player,
        payoff: Payoff,
        args,
        payoff_idx=None
    ):
        self.agent = Agent(action_plane_nvec=parent.agent.action_plane_nvec, device=parent.agent.device, initial_weights=parent.agent.state_dict()).to(parent.agent.device)
        if args.save_gpu_memory:
            offload_historical_to_cpu(self)
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


def _move_player_to_device(player, device: torch.device):
    """Ensure a player's agent resides on the target device and clear cached tensors."""
    agent = getattr(player, "agent", None)
    if agent is None or agent.device == device:
        return

    agent.to(device)
    agent.device = device
    agent.sp_logits = None
    agent._values = None


def offload_historical_to_cpu(player, active_agents=None):
    """Push inactive historical checkpoints back to CPU to prevent GPU memory from growing."""
    if not isinstance(player, Historical):
        return

    if active_agents is not None:
        for p in active_agents:
            if p is player:
                return

    _move_player_to_device(player, torch.device("cpu"))


class League:
    def __init__(
        self,
        initial_main_agent: torch.nn.Module,
        other_initial_agents: List[torch.nn.Module],
        args,
    ):
        # am Anfang legt man fest, wie viele Environments für das Training von jedem Agententyp genutzt werden (denen gibt man mit .match einen Gegner)
        self._payoff = Payoff()
        self.args = args
        

        # nur aktive Spieler (nicht Historical)
        self._learning_agents =  []
        # for _ in range(args.num_main_envs):
        #   main_agent = MainPlayer(initial_agent, self._payoff, args=args)
        #   self._learning_agents.append(main_agent)

        for i in range(len(other_initial_agents)):
            main_agent_historical = MainPlayer(other_initial_agents[i], self._payoff, args=args)
            self._payoff.add_player(main_agent_historical.checkpoint())

        # only 1 Mainagent:
        main_agent = MainPlayer(initial_main_agent, self._payoff, args=args)
        for _ in range(args.num_main_envs):
            self._learning_agents.append(main_agent)

        # (TODO (League training): müssen die main_agents ihre Gewichte unterschiedlich updaten können, um besser gegen andere main_agents zu trainieren?
        # gerade ex nur eine Instanz als main_agent (wenn ein main_agent ein update macht, dann updaten alle main_agents ihre Gewichte gleich) 
        # (initial_agents statt initial_agent?))
        self._payoff.add_player(main_agent.checkpoint())
        self._payoff.add_player(main_agent)

        for _ in range(args.num_main_exploiters):
            main_exploiter = MainExploiter(initial_main_agent, self._payoff, args=args)
            for _ in range(args.num_envs_per_main_exploiters):
                self._learning_agents.append(
                    main_exploiter)
            self._payoff.add_player(main_exploiter)
        for _ in range(args.num_league_exploiters):
            league_exploiter = LeagueExploiter(initial_main_agent, self._payoff, args=args)
            for _ in range(args.num_envs_per_league_exploiters):
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
    


    def _log_selfplay_results(self, args, agent, writer, global_step, infos, done_idx, done_agent, dyn_winloss, attack_weight, num_done_selfplaygames, last_logged_selfplay_games, indices_per_exploiter):
        if isinstance(done_agent, MainPlayer):
            writer.recent_selfplay_winloss.append(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'])

            selfplay_with_draw = np.mean(np.add(writer.recent_selfplay_winloss, 1) / 2)
            selfplay_winrate = np.mean(np.clip(writer.recent_selfplay_winloss, 0, 1))

            winloss_values = np.array(writer.recent_selfplay_winloss)
            writer.add_scalar("progress/num_selfplay_games", num_done_selfplaygames, global_step)
            writer.add_scalar(f"main_winrates/selfplay_Winrate_with_draw", selfplay_with_draw, num_done_selfplaygames)
            writer.add_scalar(f"main_winrates/selfplay_Winrate_no_draw", selfplay_winrate, num_done_selfplaygames)
            writer.add_scalar(f"main_winrates/selfplay_Winrate_no_draw_std", np.std(winloss_values), num_done_selfplaygames)

        # TODO (league training): auch andere Agents loggen? (wäre pro exploiter pro Gegner eine Zeile in der Tabelle)
        if (num_done_selfplaygames < 10 or last_logged_selfplay_games + 25 <= num_done_selfplaygames) and ((args.log_exploiter_tables) or isinstance(done_agent, MainPlayer)):

            if done_agent.agent is agent:
                self_name = getattr(done_agent, "name", done_agent.__class__.__name__)
            else:
                self_name = getattr(done_agent, "name", done_agent.__class__.__name__) + "_in_env_" + str(indices_per_exploiter[done_agent])

            last_logged_selfplay_games = num_done_selfplaygames
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

            Logger.log_wandb_summary(
                                    args,
                                    opponent_table_rows,
                                    no_reward=True,
                                    step=global_step,
                                    table_name=f"league/{self_name}_summary",
                                    with_name=self_name
                                    )
            
            if args.log_exploiter_winrates or isinstance(done_agent, MainPlayer):
                for opp, games, r in zip(opp_names, game_count, win_rates):
                    if games > 0:
                        writer.add_scalar(f"winrate_per_opponent/{self_name}_vs_{opp}", r, games)
        # print(f"{self_name} Win rates against all opponents: {list(zip(opp_names, rates))}")
        print(f"global_step={global_step}, episode_reward={(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss + infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight):.3f}")
        if isinstance(done_agent, MainPlayer):
            print(f"selfplay_winrate_no_draw_{len(writer.recent_selfplay_winloss)}={selfplay_winrate:.3f}, selfplay_winrate_with_draw_0.5_{len(writer.recent_selfplay_winloss)}={selfplay_with_draw:.3f}\n")
        return last_logged_selfplay_games

    # TODO: rufe die Methode richtig auf
    def handle_game_end(self, args, agent, writer, active_league_agents, global_step, infos, attack_weight, done_idx, done_agent, dyn_winloss, hist_reward, num_done_selfplaygames, indices_per_exploiter, last_logged_selfplay_games):
        old_opp = active_league_agents[done_idx + 1]
        self.update(active_league_agents[done_idx], active_league_agents[done_idx + 1], infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'])

        print(f"Game {int(done_idx/2)} ended: {getattr(done_agent, 'name', done_agent.__class__.__name__)} vs {getattr(active_league_agents[done_idx + 1], 'name', active_league_agents[done_idx + 1].__class__.__name__)}, result: {infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction']}")
        last_logged_selfplay_games = self._log_selfplay_results(
            args,
            agent,
            writer,
            global_step,
            infos,
            done_idx,
            done_agent,
            dyn_winloss,
            attack_weight,
            num_done_selfplaygames,
            last_logged_selfplay_games,
            indices_per_exploiter
        )
        if done_agent.ready_to_checkpoint():
            self.add_player(done_agent.checkpoint())

            if isinstance(done_agent, MainExploiter) or isinstance(done_agent, LeagueExploiter):
                exploiter_name = getattr(done_agent, "name", done_agent.__class__.__name__) + "_in_env_" + str(indices_per_exploiter[done_agent])
                done_agent.num_resets_checkpoints += 1
                print(exploiter_name + f" created its {done_agent.num_resets_checkpoints}th new Historical checkpoint and reset its weights.")
                writer.add_scalar(f"{exploiter_name}/num_resets_checkpoints", done_agent.num_resets_checkpoints, global_step)

            _on_checkpoint(hist_reward=hist_reward, args=args)
        # neuen gegner in diesem environment auswählen
        opp = done_agent.get_match()[0]
        if args.save_gpu_memory:
            _move_player_to_device(opp, agent.device)

        if isinstance(opp, Historical):
            opp_name = getattr(opp, "name", opp.__class__.__name__) + "_" + getattr(opp._parent, "name", opp._parent.__class__.__name__)
        else:
            opp_name = getattr(opp, "name", opp.__class__.__name__)

        print(f"New Match in Game {int(done_idx/2)}: {getattr(done_agent, 'name', done_agent.__class__.__name__)} vs {opp_name}\n")

        return opp, last_logged_selfplay_games, old_opp


def initialize_league(args, device, agent, other_initial_agents=[]):
    # TODO (League training): (don't fill non selfplaying envs) Fokus auf die agenten gibt, gegen nur cur main und alte main: (--FSP)
    league_instance = League(initial_main_agent=agent, other_initial_agents=other_initial_agents, args=args)

    # initiale Environments mit den jeweiligen Gegnern gefüllt
    active_league_agents = []
    assert len(league_instance.learning_agents) == args.num_selfplay_envs // 2, "Number of learning agents must be half of the number of selfplay envs"
    for player0 in league_instance.learning_agents:
        opp = player0.get_match()[0]
        active_league_agents.append(player0)
        active_league_agents.append(opp)

        # wenn active_league_agents[0], active_league_agents[2] Main Agents sind: active_league_agents[0].agent is active_league_agents[2].agent == True

    unique_agents = agent.get_unique_agents(active_league_agents, output_league_agents=True)
    for m_exp in unique_agents.keys():
            if isinstance(m_exp, MainExploiter):
                for _ in range(args.num_bot_envs_per_main_exploiter):
                    active_league_agents.append(m_exp)

    # ( TODO: soll ich das machen? fill remaining environments (bot envs) with the main agent reference (sonst muss man selfplay_get_action, selfplay_get_value anpassen))
    while len(active_league_agents) < args.num_envs:
        active_league_agents.append(league_instance.learning_agents[0])

    return league_instance, active_league_agents

def log_general_main_results(writer, global_step, infos, dyn_winloss, game_length, attack_weight, done_idx, hist_reward):
    writer.add_scalar("main_charts/old_episode_reward", infos[done_idx]['episode']['r'], global_step)
    writer.add_scalar("main_charts/Game_length", game_length, global_step)
    writer.add_scalar("main_charts/Episode_reward_with_hist_reward", hist_reward + infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss + 
                                          infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight, global_step)
    writer.add_scalar("main_charts/Episode_reward", infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss + 
                                          infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight, global_step)
    writer.add_scalar("main_charts/AttackReward", infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight, global_step)
    writer.add_scalar("main_charts/WinLossRewardFunction", infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss, global_step)

def log_bot_game_results(args, writer, global_step, infos, attack_weight, done_idx, dyn_winloss, num_done_botgames):
    print(f"Game {int(args.num_selfplay_envs/2 + int(done_idx - (args.num_selfplay_envs - 1)))} ended, result: {infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction']}")
    writer.recent_bot_winloss.append(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'])

    bot_winrate = np.mean(np.clip(writer.recent_bot_winloss, 0, 1))
    with_draw = np.mean(np.add(writer.recent_bot_winloss, 1) / 2)

    winloss_values = np.array(np.clip(writer.recent_bot_winloss, 0, 1))
    writer.add_scalar("progress/num_bot_games", num_done_botgames, global_step)
    writer.add_scalar(f"main_winrates/bot_Winrate_with_Draw_0", bot_winrate, num_done_botgames)
    writer.add_scalar(f"main_winrates/bot_Winrate_std", np.std(winloss_values), num_done_botgames)
    writer.add_scalar(f"main_winrates/bot_Winrate_with_draw_0.5", with_draw, num_done_botgames)
    print(f"global_step={global_step}, episode_reward={(infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction'] * dyn_winloss + infos[done_idx]['microrts_stats']['AttackRewardFunction'] * attack_weight):.3f}, bot_winrate_{len(writer.recent_bot_winloss)}={bot_winrate:.3f}, bot_winrate_with_draw_0.5_{len(writer.recent_bot_winloss)}={with_draw:.3f}")
    print(f"bot_winrate_{len(writer.recent_bot_winloss)}={bot_winrate:.3f}, bot_winrate_with_draw_0.5_{len(writer.recent_bot_winloss)}={with_draw:.3f}")
    print(f"match in Botgame {int(done_idx - (args.num_selfplay_envs - 1))}, result: {infos[done_idx]['microrts_stats']['RAIWinLossRewardFunction']}\n")

def log_exploiter_ppo_update(args, writer, exploiter_agent_batch, exploiter_indices, pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss, global_step, experiment_name, update, grad_norm=None):
    player = exploiter_agent_batch["player"]
    name = getattr(player, "name", exploiter_agent_batch["player"].__class__.__name__) + "_in_env_" + str(exploiter_indices[player])

    writer.add_scalar(f"{name}/learning_rate", exploiter_agent_batch["optimizer"].param_groups[0]["lr"], global_step)
    writer.add_scalar(f"{name}/value_loss", args.vf_coef * v_loss.item(), global_step)
    writer.add_scalar(f"{name}/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar(f"{name}/kl_loss", kl_loss.item(), global_step)
    writer.add_scalar(f"{name}/total_loss", loss.item(), global_step)
    writer.add_scalar(f"{name}/entropy_loss", args.ent_coef * entropy_loss.item(), global_step)
    writer.add_scalar(f"{name}/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar(f"{name}/grad_norm_before_clipping", grad_norm, global_step)

    if args.kle_stop or args.kle_rollback:
        writer.add_scalar(f"{name}/pg_stop_iter", pg_stop_iter, global_step)

    if args.prod_mode and update % args.checkpoint_frequency == 0:
            print("Saving model checkpoint...")
            save_league_model(save_agent=exploiter_agent_batch["player"].agent, experiment_name=experiment_name, dir_name=f"current_{getattr(exploiter_agent_batch['player'], 'name', exploiter_agent_batch['player'].__class__.__name__)}", file_name=f"{name}")

            if update < 500:
                if update % (args.checkpoint_frequency * 5) == 0:
                    save_league_model(save_agent=exploiter_agent_batch["player"].agent, experiment_name=experiment_name, dir_name=getattr(exploiter_agent_batch["player"], "name", exploiter_agent_batch["player"].__class__.__name__), file_name=f"{name}_update_{update}")
                    
            else:
                save_league_model(save_agent=exploiter_agent_batch["player"].agent, experiment_name=experiment_name, dir_name=getattr(exploiter_agent_batch["player"], "name", exploiter_agent_batch["player"].__class__.__name__), file_name=f"{name}_update_{update}")


def train_exploiters(
        args,
        envs,
        agent_batches,
        writer,
        global_step,
        update,
        agent: Agent,
        experiment_name: str,
        exploiter_indices
    ):
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

                # TODO: wenn ich ppo_update.update benutze, dann soll das immer noch combiniert funktionieren (sonst ist es langsam)
                combined_obs = torch.cat(batch_obs, dim=0)
                combined_sc = torch.cat(batch_sc, dim=0)
                combined_z = torch.cat(batch_z, dim=0)
                combined_actions = torch.cat(batch_actions, dim=0)
                combined_masks = torch.cat(batch_masks, dim=0)
                combined_logits = torch.cat(logits_chunks, dim=0)

                _, combined_logprobs, combined_entropy, _ = agent.get_action(
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

                    loss = pg_loss - args.exploiter_ent_coef * entropy_loss + args.vf_coef * v_loss
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

            
            name = getattr(entry["player"], "name", entry["player"].__class__.__name__) + "_in_env_" + str(exploiter_indices[i].item())

            writer.add_scalar(f"{name}/learning_rate", entry["optimizer"].param_groups[0]["lr"], global_step)
            writer.add_scalar(f"{name}/value_loss", args.vf_coef * v_loss.item(), global_step)
            writer.add_scalar(f"{name}/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar(f"{name}/total_loss", loss.item(), global_step)
            writer.add_scalar(f"{name}/entropy_loss", args.exploiter_ent_coef * entropy_loss.item(), global_step)

            if args.prod_mode and update % args.checkpoint_frequency == 0:
                    print("Saving model checkpoint...")
                    save_league_model(save_agent=entry["player"].agent, experiment_name=experiment_name, dir_name=f"current_{getattr(entry['player'], 'name', entry['player'].__class__.__name__)}", file_name=f"{name}")

                    if update < 500:
                        if update % (args.checkpoint_frequency * 5) == 0:
                            save_league_model(save_agent=entry["player"].agent, experiment_name=experiment_name, dir_name=getattr(entry["player"], "name", entry["player"].__class__.__name__), file_name=f"{name}_update_{update}")
                            
                    else:
                        save_league_model(save_agent=entry["player"].agent, experiment_name=experiment_name, dir_name=getattr(entry["player"], "name", entry["player"].__class__.__name__), file_name=f"{name}_update_{update}")
