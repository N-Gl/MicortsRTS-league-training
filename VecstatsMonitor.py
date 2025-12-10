import numpy as np
import time

from stable_baselines3.common.vec_env import VecEnvWrapper



class VecstatsMonitor(VecEnvWrapper):
        def __init__(self, venv, gamma=None):
            super().__init__(venv)
            self.eprets = None
            self.eplens = None
            self.epcount = 0
            self.tstart = time.time()
            self.gamma = gamma
            self.raw_rewards = None

        def reset(self):
            obs = self.venv.reset()
            n = self.num_envs
            self.eprets = np.zeros(n, dtype=float)
            self.eplens = np.zeros(n, dtype=int)
            self.raw_rewards = [[] for _ in range(n)]
            self.tstart = time.time()
            return obs

        def step_wait(self):
            obs, denserews,attackrews,winlossrews, scorerews , dones, infos,res = self.venv.step_wait()

            self.eprets += denserews +winlossrews +scorerews +attackrews
            self.eplens += 1

            for i, info in enumerate(infos):
                if 'raw_rewards' in info:
                    self.raw_rewards[i].append(info['raw_rewards'])

            newinfos = list(infos)

            for i, done in enumerate(dones):
                if done:
                    info = infos[i].copy()
                    ep_ret = float(self.eprets[i])
                    ep_len = int(self.eplens[i])
                    ep_time = round(time.time() - self.tstart, 6)
                    info['episode'] = {'r': ep_ret, 'l': ep_len, 't': ep_time}


                    self.epcount += 1

                    if self.raw_rewards[i]:
                        agg = np.sum(np.array(self.raw_rewards[i]), axis=0)
                        raw_names = [str(rf) for rf in self.rfs]
                        info['microrts_stats'] = dict(zip(raw_names, agg.tolist()))
                    else:
                        info['microrts_stats'] = {}

                    if winlossrews[i] == 0:
                        info['microrts_stats']['draw'] = True
                    else:
                        info['microrts_stats']['draw'] = False

                    self.eprets[i] = 0.0
                    self.eplens[i] = 0
                    self.raw_rewards[i] = []
                    newinfos[i] = info

            return obs, denserews,attackrews,winlossrews, scorerews, dones, newinfos,res

        def step(self, actions):
            self.venv.step_async(actions)
            return self.step_wait()