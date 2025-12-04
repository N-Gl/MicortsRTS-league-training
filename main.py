from typing import Any, Sequence, Tuple
import hydra
from hydra.core.config_store import ConfigStore
import agent_model
from conf.config import ExperimentConfig
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
#from gym_microrts.envs.vec_env2 import MicroRTSGridModeVecEnv
from gym_microrts.envs.microrts_vec_env import  MicroRTSGridModeVecEnv
from omegaconf import OmegaConf


from gym_microrts import microrts_ai
import time
import json
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
from microrts_space_transform import MicroRTSSpaceTransform
import atexit

from agent_model import build_agent

cstore = ConfigStore.instance()
cstore.store(name="experiment_config", node=ExperimentConfig)


def _resolve_checkpoint_path(model_path: str, exp_name=None, resume=True) -> str:
    if resume:
        if model_path.endswith(".pt"):
            if exp_name == None:
                raise ValueError("exp_name must be provided when model_path ends with '.pt'")
            return model_path
        else:
            checkpoint_path = f"models/{model_path}/agent.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    else:
        checkpoint_path = f"models/{model_path}/agent.pt"
    return checkpoint_path


@hydra.main(config_path="conf", config_name="config")
def main(cfg: ExperimentConfig):
    args = cfg.ExperimentConfig

    if args.seed == None:
        args.seed = int(time.time())

    if not args.model_path and not args.evaluation_model_paths:
        raise ValueError("Please provide a model path via config model_path")
    
    if args.league_training and args.selfplay:
        raise ValueError("league_training and selfplay are not possible with the same call")
    
    
    if args.evaluate:
        args.__dict__.setdefault('num_selfplay_envs', args.num_parallel_eval_envs)
    else:
        args.__dict__.setdefault('num_selfplay_envs', (args.num_main_agents + args.num_main_exploiters + args.num_league_exploiters) * 2)

    args.__dict__.setdefault('num_envs', args.num_selfplay_envs + args.num_bot_envs)
    # TODO: ist die args.batch_size korrekt?
    args.__dict__.setdefault('batch_size', int((args.num_selfplay_envs//2 + args.num_bot_envs) * args.num_steps))
    args.__dict__.setdefault('minibatch_size', int(args.batch_size // max(args.n_minibatch, 1)))

    print("Experiment config:")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2, sort_keys=True))
    print("change config in conf/config.yaml")


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
    
    if args.exp_name:
        experiment_name = f"{args.exp_name}"
    else:
        experiment_name = f"{args.model_path}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    wandb_log_fn = None

    RUN_ID_PATH = f"models/{experiment_name}/wandb_run_id.txt"
    if args.prod_mode:
        import  wandb
        if os.path.exists(RUN_ID_PATH):
            # Resume: read the previous run ID
            with open(RUN_ID_PATH, "r") as f:
                run_id = f.read().strip()
            resume_mode = "must"
        else:
            # First time: no file exists
            run_id = None
            resume_mode = "allow"
        run = wandb.init(
            project=args.wandb_project_name, entity=args.wandb_entity,
            # sync_tensorboard=True,
            config=vars(args), name=experiment_name, monitor_gym=True, resume=resume_mode,id=run_id, save_code=False,
            settings=wandb.Settings(console="off") # console wird nicht synchronisiert
            )

        if resume_mode == "allow":
            if args.create_run_id_file:
                os.makedirs(os.path.dirname(RUN_ID_PATH), exist_ok=True)
                with open(RUN_ID_PATH, "w") as f:
                    f.write(run.id)
        wandb.tensorboard.patch(save=False)
        writer = SummaryWriter(f"/tmp/{experiment_name}")
        writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
        wandb_log_fn = wandb.log


    def _close_loggers():
        try:
            writer.flush()
            writer.close()
        except Exception:
            pass
        if args.prod_mode:
            try:
                wandb.finish()
            except Exception:
                pass


    atexit.register(_close_loggers)

    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    if args.num_envs < 4:
        device = torch.device('cpu')
        print(f"Device: {device}, because num_envs < 4")
    else:
        print(device)

    if args.deterministic:
        args.seed = args.seed + 1 * args.initial_BC + 2 * args.BC_finetuning + 4 * args.ppo + 8 * args.evaluate + 16 * args.selfplay + 16 * args.league_training
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
    reward_weight = np.array([1.0, 1.0, 1.0, 0.2, 1.0, 4.0, 5.25, 6.0, 0])


    if args.initial_BC or args.BC_finetuning:
        args.num_bot_envs = 2
    if args.evaluate:
        args.num_bot_envs = args.num_parallel_eval_envs

    else:
        if args.num_bot_envs > 16:
            opponents = [microrts_ai.coacAI for _ in range(args.num_bot_envs-8)] + [microrts_ai.mayari for _ in range(8)]
        elif args.num_bot_envs > 8:
            opponents = [microrts_ai.coacAI for _ in range((args.num_bot_envs+1)//2+2)] + [microrts_ai.mayari for _ in range((args.num_bot_envs)//2-2)]
        else:
            opponents = [microrts_ai.coacAI for _ in range((args.num_bot_envs+1)//2)] + [microrts_ai.mayari for _ in range((args.num_bot_envs)//2)]

        print(f"opponents: \n{opponents}")
    
        envs = MicroRTSGridModeVecEnv(
            num_selfplay_envs=args.num_selfplay_envs,
            num_bot_envs=args.num_bot_envs,
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
            envs = VecVideoRecorder(envs, f'videos/{experiment_name}',
                                    record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)



    #def getScalarFeatures(obs, res, numenvs):
    #    ScFeatures = torch.zeros(numenvs, 11).to(device)
#
    #    for i in range(numenvs):
#
    #        res_plane = (obs[i, :, :, 1] * obs[i, :, :, 7])
    #        lightunit_plane = (obs[i, :, :, 11])
    #        heavyunit_plane = (obs[0, :, :, 12])
    #        rangedunit_plane = (obs[0, :, :, 13])
    #        total_res = res_plane.sum().item()
#
#
    #        worker_plane = obs[i, :, :, 10]
    #        building_plane = obs[i, :, :, 9]
    #        player0_plane = obs[i, :, :, 4]
    #        player1_plane = obs[i, :, :, 5]
#
    #        ScFeatures[i,0]  = res[i][0] #Player0 res
    #        ScFeatures[i, 1] =  res[i][1] #Player1 res
    #        ScFeatures[i, 2] =  total_res #vorhandene res
    #        ScFeatures[i, 3] = (worker_plane * player0_plane).sum().item()  # Player0 worker
    #        ScFeatures[i, 4] = (lightunit_plane * player0_plane).sum().item()  # Player0 light
    #        ScFeatures[i, 5] = (heavyunit_plane * player0_plane).sum().item()  # Player0 heavy
    #        ScFeatures[i, 6] = (rangedunit_plane * player0_plane).sum().item()  # Player0 ranged
    #        ScFeatures[i, 7] = (worker_plane * player1_plane).sum().item()  # Player1 worker
    #        ScFeatures[i, 8] = (lightunit_plane * player1_plane).sum().item()  # Player1 light
    #        ScFeatures[i, 9] = (heavyunit_plane * player1_plane).sum().item()  # Player1 heavy
    #        ScFeatures[i, 10] = (rangedunit_plane * player1_plane).sum().item()  # Player1 ranged
#
    #    #Time step in the game
    #    return ScFeatures


    def getScalarFeatures(obs, res, numenvs):
        # old_Sc = old_getScalarFeatures(obs, res, numenvs)
        num_envs = obs.shape[0]
        device = obs.device
        dtype = obs.dtype
        spatial_dims = (1, 2)

        res_tensor = torch.as_tensor(np.array(res), device=device, dtype=dtype)
        res_tensor = res_tensor.reshape(num_envs, -1)

        player0_res = res_tensor[:, 0]
        player1_res = res_tensor[:, 1]

        res_plane = obs[..., 1] * obs[..., 7]
        worker_plane = obs[..., 10]
        light_plane = obs[..., 11]
        heavy_plane = obs[..., 12]
        ranged_plane = obs[..., 13]
        player0_plane = obs[..., 4]
        player1_plane = obs[..., 5]

        total_res = res_plane.sum(dim=spatial_dims)
        player0_workers = (worker_plane * player0_plane).sum(dim=spatial_dims)
        player0_light = (light_plane * player0_plane).sum(dim=spatial_dims)
        player0_heavy = (heavy_plane * player0_plane).sum(dim=spatial_dims)
        player0_ranged = (ranged_plane * player0_plane).sum(dim=spatial_dims)
        player1_workers = (worker_plane * player1_plane).sum(dim=spatial_dims)
        player1_light = (light_plane * player1_plane).sum(dim=spatial_dims)
        player1_heavy = (heavy_plane * player1_plane).sum(dim=spatial_dims)
        player1_ranged = (ranged_plane * player1_plane).sum(dim=spatial_dims)
        
        sc_features = torch.stack(
            [
                player0_res,
                player1_res,
                total_res,
                player0_workers,
                player0_light,
                player0_heavy,
                player0_ranged,
                player1_workers,
                player1_light,
                player1_heavy,
                player1_ranged,
            ],
            dim=1,
        )
     
        # print(torch.all(old_Sc == sc_features))
        return sc_features


    if not args.evaluate:
        action_plane_nvec = envsT.action_plane_space.nvec

        agent = build_agent(action_plane_nvec, device)

        # new
        if args.BC_model_path:
            path_BCagent = _resolve_checkpoint_path(args.BC_model_path, args.exp_name, resume=args.resume)
        else:
            path_BCagent = _resolve_checkpoint_path(args.model_path, args.exp_name, resume=args.resume)
        supervised_agent = build_agent(action_plane_nvec, device)
        # end new

        start_epoch = 1
        if args.prod_mode and wandb.run.resumed:
            if run.summary.get('charts/BCepoch'):
                start_epoch = run.summary.get('charts/BCepoch') + 1
            else:
                start_epoch =  1

        if args.resume:
            ckpt_path = _resolve_checkpoint_path(args.model_path, args.exp_name, resume=args.resume)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
            agent.load_state_dict(torch.load(ckpt_path, map_location=device,weights_only=True))
            agent.train()
            print(f"resumed at epoch {start_epoch}")


            # new (moved out of if-Statement)
            # path_BCagent = f"models/BCagent.pt"
            # supervised_agent = build_agent(action_plane_nvec, device)
            # end new

            supervised_agent.load_state_dict(
                torch.load(
                    path_BCagent,
                    map_location=device,
                    weights_only=True))
            for param in supervised_agent.parameters():
                param.requires_grad = False
            supervised_agent.eval()


    if args.initial_BC:
        from bc import BehaviorCloning
        args.nurwins = False
        BehaviorCloning(
            agent=agent,
            args=args,
            writer=writer,
            device=device,
            experiment_name=experiment_name,
            get_scalar_features=getScalarFeatures,
            start_epoch=start_epoch,
            replay_dir="replays/initial",
            model_dir=f"models/{experiment_name}/initial_BC/{experiment_name}",
            wandb_log_fn=wandb_log_fn,
        ).run()

    if args.BC_finetuning:
        from bc import BehaviorCloning
        args.nurwins = True
        BehaviorCloning(
            agent=agent,
            args=args,
            writer=writer,
            device=device,
            experiment_name=experiment_name,
            get_scalar_features=getScalarFeatures,
            start_epoch=start_epoch,
            replay_dir="replays/finetuning",
            model_dir=f"models/{experiment_name}/BC_finetuning/{experiment_name}",
            wandb_log_fn=wandb_log_fn,
        ).run()

    if args.ppo:
        from ppo import PPOTrainer
        ppo_trainer = PPOTrainer(
            agent=agent,
            supervised_agent=supervised_agent,
            envs=envsT,
            args=args,
            writer=writer,
            device=device,
            experiment_name=experiment_name,
            get_scalar_features=getScalarFeatures,
            checkpoint_frequency=10,
        )
        ppo_trainer.train()

    if args.selfplay:
        from selfplay_only import SelfPlayTrainer

        selfplay_trainer = SelfPlayTrainer(
            agent=agent,
            supervised_agent=supervised_agent,
            envs=envsT,
            args=args,
            writer=writer,
            device=device,
            experiment_name=experiment_name,
            get_scalar_features=getScalarFeatures,
            checkpoint_frequency=10,
        )
        selfplay_trainer.train()



    if args.league_training:
        from league_training import LeagueTrainer

        league_trainer = LeagueTrainer(
            agent=agent,
            supervised_agent=supervised_agent,
            envs=envsT,
            args=args,
            writer=writer,
            device=device,
            experiment_name=experiment_name,
            get_scalar_features=getScalarFeatures,
            action_plane_nvec=action_plane_nvec,
            checkpoint_frequency=10,
        )
        league_trainer.train()


    if args.evaluate:
        # from evaluate import evaluate_agent


        default_opponent_paths = [
            # ["PPO_with_basis_Thesis_BCagent", agent_model.Agent, "saved_models/19_10_2025 (good_but_with_basis_Thesis_BCagent)/agent.pt", None],
            # ["finished_PPO_Basis_thesis", agent_model.Agent, "models/finished_PPO_Basis_Thesis/finished_PPO_Basis_thesis.pt", None]
        ]
        # 2nd element: uninitialized Agent class 
        # last element: League Agent (if None: main Agent)


        default_bot_opponents: Sequence[Tuple[str, Any]] = [
            ("coacAI", microrts_ai.coacAI),
            ("mayari", microrts_ai.mayari),
            ("passiveAI", microrts_ai.passiveAI),
            ("workerRushAI", microrts_ai.workerRushAI),
            ("lightRushAI", microrts_ai.lightRushAI),
            ("randomAI", microrts_ai.randomAI),
            ("randomBiasedAI", microrts_ai.randomBiasedAI),
            ("rojo", microrts_ai.rojo),
            ("mixedBot", microrts_ai.mixedBot),
            ("izanagi", microrts_ai.izanagi), 
            ("tiamat", microrts_ai.tiamat),
            ("droplet", microrts_ai.droplet),
            # ("guidedRojoA3N", microrts_ai.guidedRojoA3N),
            ("naiveMCTSAI", microrts_ai.naiveMCTSAI),
        ]

        if args.model_path:
            model_paths = [args.model_path]
        else:
            model_paths = args.evaluation_model_paths

        for name, eval_path in zip(args.names, model_paths):
            args.model_path = eval_path
            print(f"\nEvaluating model at path: {args.model_path}")

            bot_aggregate_stats, bot_aggregate_episode_rewards, bot_opponent_table_rows = {}, [], []
            aggregate_stats, aggregate_episode_rewards, opponent_table_rows = {}, [], []
            if len(default_bot_opponents) > 0:
                from evaluate import bot_evaluate_agent
                bot_aggregate_stats, bot_aggregate_episode_rewards, bot_opponent_table_rows = bot_evaluate_agent(
                    args=args,
                    evaluation_opponents=default_bot_opponents,
                    device=device,
                    get_scalar_features=getScalarFeatures,
                    reward_weight=reward_weight,
                    vecstats_monitor_cls=VecstatsMonitor,
                )

            if len(default_opponent_paths) > 0:
                from selfplay_evaluate import evaluate_agent
                args.num_parallel_selfplay_eval_games = args.num_parallel_selfplay_eval_games * 2
                aggregate_stats, aggregate_episode_rewards, opponent_table_rows = evaluate_agent(
                    args=args,
                    default_opponent_paths=default_opponent_paths,
                    device=device,
                    get_scalar_features=getScalarFeatures,
                    reward_weight=reward_weight,
                    vecstats_monitor_cls=VecstatsMonitor,
                )

        

            aggregate_stats.update(bot_aggregate_stats)
            aggregate_episode_rewards.extend(bot_aggregate_episode_rewards)
            opponent_table_rows.extend(bot_opponent_table_rows)
            from log_aggregate_result_table import Logger
            Logger.log_aggregate_results(aggregate_stats, aggregate_episode_rewards, writer)
            table=Logger.log_wandb_summary(args, opponent_table_rows, aggregate_stats, with_name=name, table=table if 'table' in locals() else None)
            print("evaluation finished, logged summary to wandb")


if __name__ == "__main__":
    main()
