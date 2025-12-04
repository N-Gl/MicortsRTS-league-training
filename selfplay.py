import atexit
import copy
import time
import os
from collections import deque
import numpy as np
# from stable_baselines3.common.vec_env import VecEnvWrapper
import torch
import torch.nn as nn
import jpype
from jpype.imports import registerDomain
from jpype.types import JArray


from agent_interface import AgentInterface

class Selfplay:
    def __init__(self, num_selfplay_envs, main_agent: AgentInterface,
                 mapPath="maps/16x16/basesWorkers16x16.xml", microrts_path="gym_microrts/microrts",
                 rewardFunctionInterfaceArr: list = None, max_main_agents=100, device='cuda'):
        
        # ===============================
        # Import Javaclasses with JPype
        # ===============================

        
        if not jpype.isJVMStarted():
            registerDomain("ai")
            registerDomain("ts", alias="tests")

            jars = ['RAISocketAI.jar',
                    'microrts.jar',
                    #
                    "Coac.jar",
                    "Droplet.jar",
                    "GRojoA3N.jar",
                    "Izanagi.jar",
                    "MixedBot.jar",
                    "TiamatBot.jar",
                    "UMSBot.jar",
                    "mayariBot.jar",]
            
            for jar in jars:
                jpype.addClassPath(os.path.join(microrts_path, jar))

            jpype.startJVM(convertStrings=False)
            atexit.register(jpype.shutdownJVM)

        self.RAIGridnetClient = jpype.JClass("tests.RAIGridnetClient")
        self.unitTypeTable = jpype.JClass("rts.units.UnitTypeTable")


        from ai.reward import (
            AttackRewardFunction,
            ProduceBuildingRewardFunction,
            ProduceHeavyUnitRewardFunction,
            ProduceLightUnitRewardFunction,
            ProduceRangedUnitRewardFunction,
            ProduceWorkerRewardFunction,
            RAIWinLossRewardFunction,
            ResourceGatherRewardFunction,
            RewardFunctionInterface,
            ScoreRewardFunction,
            )
        
        from ai.core import AI
        self.AI = AI
        # ===============================
        

        self.num_selfplay_envs = num_selfplay_envs
        self.main_agent = main_agent
            
        self.mapPath = mapPath
        self.microrts_path = microrts_path
        self.max_main_agents = max_main_agents
        self.device = device
        self.next_agent_index = 0  # To keep track of saved agent files

        self.rfs = rewardFunctionInterfaceArr
        if self.rfs is None:
            self.rfs = JArray(RewardFunctionInterface)(
                [
                    RAIWinLossRewardFunction(),
                    ResourceGatherRewardFunction(),
                    ProduceWorkerRewardFunction(),
                    ProduceBuildingRewardFunction(),
                    AttackRewardFunction(),
                    ProduceLightUnitRewardFunction(),
                    ProduceRangedUnitRewardFunction(),
                    ProduceHeavyUnitRewardFunction(),
                    ScoreRewardFunction(),
                    # CloserToEnemyBaseRewardFunction(),
                ]
            )

        self.tstart = time.time()
        self.prev_main_agents = deque()
        # Liste of Tupels (Location of Agent.pt, Score)
        self.environment = [None] * num_selfplay_envs
        # Liste of running Environments (Environment, Agent0, Agent1)
        # TODO: erwarte für die Agenten AgentInterface
        



    def add_main_agent(self, model: AgentInterface, score=-100):
        '''
        Add new Agent to List of previous main agents Self-Play.
        If the List exceeds max_main_agents, the worst agents are removed.
        TODO: Calculate Score of Agent differently (amount of Wins?)
        TODO: logging if prod_mode, when Method is called
        TODO: Calculate Score of Agent differently (amount of Wins?)
        '''

        # model_copy = model
        # self.prev_main_agents.append((model_copy, score))

        os.makedirs("saved_modules", exist_ok=True)
        save_path = f"saved_modules/selfplay_agent_{self.next_agent_index}.pt"
        torch.save(model.state_dict(), save_path)
        self.prev_main_agents.append((save_path, score))

        self.next_agent_index += 1

        if len(self.prev_main_agents) >= 1e6:
            self.next_agent_index = 0

        # Sort and Remove worst agent if exceeding max_main_agents
        self.prev_main_agents = deque(sorted(self.prev_main_agents, key=lambda x: x[1], reverse=True))
        if len(self.prev_main_agents) > self.max_main_agents:
            agent_path, _ = self.prev_main_agents[self.max_main_agents]
            if os.path.exists(agent_path):
                os.remove(agent_path)
            self.prev_main_agents = self.prev_main_agents[:self.max_main_agents]
        else:
            self.prev_main_agents = deque(sorted(self.prev_main_agents, key=lambda x: x[1], reverse=True))

        print(
            f"Added new main agent to self-play pool. Current pool size: {len(self.prev_main_agents)}"
        )

    def new_environment(self, i: int):
        '''
        Start self-play environments with random opponents from the pool and save it in Position i.
        If the opponent pool is empty, the current main agent will be saved in the pool. 
        TODO: fill in p
        new Agents are sampled from the pool of previous main agents with a probability of p.
        TODO: new Agents are sampled from the pool of main exploiter agents with a probability of p.
        TODO: new Agents are sampled from the pool of League exploiter agents with a probability of p.
        TODO: If no previous main agents are available, the current main agent will be added to the pool and used as opponent.
        '''

        # main agent == player 0
        agent1 = self.main_agent

        if len(self.prev_main_agents) == 0:
            self.add_main_agent(self.main_agent)

        agent2_path, _ = self.prev_main_agents[np.random.randint(len(self.prev_main_agents))]
        agent2 = copy.deepcopy(self.main_agent)
        agent2.load_state_dict(torch.load(agent2_path, map_location=torch.device(self.device)))
        agent2.eval()

        # temp:
        self.environment[i] = ("Bestes env", agent1, agent2)
        # self.environment[i] = (self.RAIGridnetClient(self.rfs, self.microrts_path, self.mapPath, 
        #                                            JArray(self.AI)([agent2(self.unitTypeTable)]), self.unitTypeTable, partialObs=False), agent1, agent2)
        # TODO: RAIGridnetClient gibt nicht 'sc' und 'z' an die forward Methode 
        # (andere environment schnittstelle finden)
        # https://github.com/sgoodfriend/rl-algo-impls/blob/main/rl_algo_impls/runner/selfplay_evaluate.py

    def selfplay_step(self):
        '''
        Step the self-play environment.
        If an environment is done or if the environment pool is empty, a new Environment will be created with a new opponent from the pool.
        '''
    
        if self.environment[0] is None:
            for i in range(self.num_selfplay_envs):
                self.new_environment(i)

        print(self.environment[0])
        
        return 


    ### debugging
    def debugging_print(self):
        print("SelfPlay Pool:")
        for i, (agent, score) in enumerate(self.prev_main_agents):
            print(f"Agent {i}: Score {score}")
        print("End of SelfPlay Pool")
    