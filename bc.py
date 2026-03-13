import gc
import io
import os
from typing import Callable, Sequence, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import zstandard as zstd
from gym_microrts import microrts_ai
from gym_microrts.envs.microrts_bot_vec_env import MicroRTSBotGridVecEnv
from microrts_space_transformbots import MicroRTSSpaceTransformbot
from selfplay_league import adjust_action_selfplay, adjust_obs_selfplay
from torch.utils.data import DataLoader, IterableDataset

_BC_ADJUST_ARGS = type("AdjustArgs", (), {"num_selfplay_envs": 2})()



def getScalarFeatures(obs, res, numenvs, device):
        ScFeatures = torch.zeros(numenvs, 11).to(device)

        for i in range(numenvs):

            res_plane = (obs[i, :, :, 1] * obs[i, :, :, 7])
            lightunit_plane = (obs[i, :, :, 11])
            heavyunit_plane = (obs[i, :, :, 12])
            rangedunit_plane = (obs[i, :, :, 13])
            total_res = res_plane.sum().item()


            worker_plane = obs[i, :, :, 10]
            building_plane = obs[i, :, :, 9]
            player0_plane = obs[i, :, :, 4]
            player1_plane = obs[i, :, :, 5]

            ScFeatures[i,0]  = res[i][0] #Player0 res
            ScFeatures[i, 1] =  res[i][1] #Player1 res
            ScFeatures[i, 2] =  total_res #vorhandene res
            ScFeatures[i, 3] = (worker_plane * player0_plane).sum().item()  # Player0 worker
            ScFeatures[i, 4] = (lightunit_plane * player0_plane).sum().item()  # Player0 light
            ScFeatures[i, 5] = (heavyunit_plane * player0_plane).sum().item()  # Player0 heavy
            ScFeatures[i, 6] = (rangedunit_plane * player0_plane).sum().item()  # Player0 ranged
            ScFeatures[i, 7] = (worker_plane * player1_plane).sum().item()  # Player1 worker
            ScFeatures[i, 8] = (lightunit_plane * player1_plane).sum().item()  # Player1 light
            ScFeatures[i, 9] = (heavyunit_plane * player1_plane).sum().item()  # Player1 heavy
            ScFeatures[i, 10] = (rangedunit_plane * player1_plane).sum().item()  # Player1 ranged

        #Time step in the game
        return ScFeatures


class ReplayDataset(IterableDataset):
    """Streams compressed replay files without loading everything into memory."""

    def __init__(self, data_files: Sequence[str]):
        self.data_files = data_files

    def __iter__(self):
        for path in self.data_files:
            try:
                with open(path, "rb") as f:
                    dctx = zstd.ZstdDecompressor()
                    with dctx.stream_reader(f) as reader:
                        buffer = io.BytesIO(reader.read())
                        data = torch.load(buffer, map_location="cpu", weights_only=True)

                for sample in zip(data["obs"], data["act"], data["sc"], data["z"]):
                    yield sample

                del data
                del buffer
                gc.collect()
            except Exception as exc:  # pragma: no cover - defensive, matches old behaviour
                print(f"[ReplayDataset] Error loading {os.path.basename(path)}: {exc}")


class BehaviorCloning:
    def __init__(
        self,
        agent: nn.Module,
        args,
        writer,
        device: torch.device,
        experiment_name: str,
        get_scalar_features: Callable[[torch.Tensor, Sequence, int], torch.Tensor],
        start_epoch: int = 1,
        replay_dir: str = "replays",
        model_dir: str = "models",
        wandb_log_fn: Optional[Callable[..., None]] = None,
    ):
        self.agent = agent
        self.args = args
        self.writer = writer
        self.device = device
        self.experiment_name = experiment_name
        self.get_scalar_features = get_scalar_features
        self.start_epoch = start_epoch
        self.replay_dir = replay_dir
        self.model_dir = model_dir
        self.wandb_log_fn = wandb_log_fn

    @staticmethod
    def _adjust_obs_to_player0(obs_batch, is_new_env: bool):
        obs_tensor = torch.as_tensor(obs_batch).clone()
        if obs_tensor.ndim != 4 or obs_tensor.shape[0] != 1:
            return obs_batch
        wrapped_obs = torch.zeros(
            (2, obs_tensor.shape[1], obs_tensor.shape[2], obs_tensor.shape[3]),
            dtype=obs_tensor.dtype,
            device=obs_tensor.device,
        )
        wrapped_obs[1] = obs_tensor[0]
        adjust_obs_selfplay(_BC_ADJUST_ARGS, wrapped_obs, is_new_env=is_new_env)
        return wrapped_obs[1:2].cpu().numpy()

    @staticmethod
    def _adjust_actions_to_player0(expert_actions):
        if len(expert_actions) == 0:
            return np.zeros((0, 8), dtype=np.int64)
        action_arr = np.asarray(expert_actions, dtype=np.int64)
        if action_arr.ndim != 2 or action_arr.shape[1] < 8:
            return action_arr
        action_counts = np.array([0, action_arr.shape[0]], dtype=np.int64)
        adjust_action_selfplay(_BC_ADJUST_ARGS, action_arr, action_counts)
        return action_arr

    def run(self):
        print("BC training Setup")

        os.makedirs(self.replay_dir, exist_ok=True)

        if self.args.newdata:
            self._collect_new_data()

        print("BC training started")
        replay_dir = os.path.abspath(self.replay_dir)
        replay_files = sorted(
            [
                os.path.join(replay_dir, file_name)
                for file_name in os.listdir(replay_dir)
                if file_name.endswith(".pt.zst")
            ]
        )

        if not replay_files:
            raise FileNotFoundError(
                f"No replay files found in {self.replay_dir}. "
                "Run with --newdata or provide existing datasets."
            )

        replay_dataset = ReplayDataset(replay_files)

        if self.args.anneal_lr:
            lr_fn = lambda frac: frac * 1e-3  # noqa: E731
        else:
            lr_fn = None

        self.agent.train()
        optimizer = optim.AdamW(
            self.agent.parameters(),
            lr=1e-3,
            eps=self.args.BC_learning_rate,
            # weight_decay=self.args.BC_weight_decay,
        )
        # new (BA Parameter) (initial learning rate of 1 × 10^−3)

        train_loader = DataLoader(
            replay_dataset,
            batch_size=2048, # new (BA Parameter) (batch size of 2048)
            num_workers=0,
            pin_memory=True,
            pin_memory_device="cuda",
        )

        warmup_epochs = self.args.warmup_epochs

        try:
            for epoch in range(self.start_epoch, self.args.epochs): # new (BA Parameter) 100 Epochen, BC-finetuning 70 Epochen (sicher: 80)
                if lr_fn is not None:
                    frac = 1.0 - (epoch - 1.0) / (self.args.epochs - 1)
                    lr_now = lr_fn(frac)
                    optimizer.param_groups[0]["lr"] = lr_now
                avg_train_loss, lr = self._train_epoch(train_loader, optimizer, warmup_epochs, epoch)
                self._log_epoch_metrics(epoch, avg_train_loss, lr)

                epoch_model_dir = f"{self.model_dir}_epoch_{epoch}"
                os.makedirs(epoch_model_dir, exist_ok=True)
                torch.save(self.agent.state_dict(), f"{epoch_model_dir}/agent.pt")

                print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}, learning rate: {lr:.6f}")
        except KeyboardInterrupt:
            print("BC training interrupted early by user, flushing metrics before exit.")
        finally:
            self._finalize_logging()

    def _collect_new_data(self):
        bc_expert_ai = getattr(self.args, "bc_expert_ai", None)
        bc_opponent_ai = getattr(self.args, "bc_opponent_ai", bc_expert_ai)
        bc_opponent_ais = getattr(self.args, "bc_opponent_ais", None)
        bc_num_runs = int(getattr(self.args, "bc_num_runs", 250))
        bc_expert_player = int(getattr(self.args, "bc_expert_player", 0))

        if bc_expert_player not in (0, 1):
            raise ValueError("bc_expert_player must be 0 or 1.")
        expert_reference_index = bc_expert_player

        if bc_expert_ai:
            opponent_ai_names = []
            if bc_opponent_ais:
                if isinstance(bc_opponent_ais, str):
                    opponent_ai_names = [bc_opponent_ais]
                else:
                    opponent_ai_names = [str(ai_name) for ai_name in bc_opponent_ais]
            elif bc_opponent_ai:
                opponent_ai_names = [bc_opponent_ai]

            if not opponent_ai_names:
                raise ValueError(
                    "Set bc_opponent_ai or bc_opponent_ais when bc_expert_ai is provided."
                )
            if bc_num_runs <= 0:
                raise ValueError("bc_num_runs must be > 0.")
            expert_ai_callable = self._resolve_ai_callable(bc_expert_ai)
            if expert_reference_index == 0:
                opponents = [
                    [expert_ai_callable, self._resolve_ai_callable(opponent_ai_name)]
                    for opponent_ai_name in opponent_ai_names
                ]
            else:
                opponents = [
                    [self._resolve_ai_callable(opponent_ai_name), expert_ai_callable]
                    for opponent_ai_name in opponent_ai_names
                ]
            num_runs = [bc_num_runs for _ in opponents]
            print(
                f"Using custom BC replay collection: {bc_expert_ai} vs {opponent_ai_names}, "
                f"{bc_num_runs} episodes per opponent, expert as player {bc_expert_player}."
            )
        else:
            opponents = [
            # new (BA Parameter) CoacAI gegen alle 14 Bots - guidedRojoA3N (wegen Crash) rausgenommen
            [microrts_ai.coacAI, microrts_ai.workerRushAI],
            [microrts_ai.coacAI, microrts_ai.passiveAI],
            [microrts_ai.coacAI, microrts_ai.lightRushAI],
            [microrts_ai.coacAI, microrts_ai.coacAI],
            [microrts_ai.coacAI, microrts_ai.mayari],
            [microrts_ai.coacAI, microrts_ai.randomAI],
            [microrts_ai.coacAI, microrts_ai.randomBiasedAI],
            [microrts_ai.coacAI, microrts_ai.rojo],
            [microrts_ai.coacAI, microrts_ai.mixedBot],
            [microrts_ai.coacAI, microrts_ai.izanagi],
            [microrts_ai.coacAI, microrts_ai.tiamat],
            [microrts_ai.coacAI, microrts_ai.droplet],
            [microrts_ai.coacAI, microrts_ai.naiveMCTSAI],

            # new (BA Parameter) mayari gegen alle 14 Bots - guidedRojoA3N (wegen Crash) rausgenommen
            [microrts_ai.mayari, microrts_ai.workerRushAI],
            [microrts_ai.mayari, microrts_ai.passiveAI],
            [microrts_ai.mayari, microrts_ai.lightRushAI],
            [microrts_ai.mayari, microrts_ai.coacAI],
            [microrts_ai.mayari, microrts_ai.mayari],
            [microrts_ai.mayari, microrts_ai.randomAI],
            [microrts_ai.mayari, microrts_ai.randomBiasedAI],
            [microrts_ai.mayari, microrts_ai.rojo],
            [microrts_ai.mayari, microrts_ai.mixedBot],
            [microrts_ai.mayari, microrts_ai.izanagi],
            [microrts_ai.mayari, microrts_ai.tiamat],
            [microrts_ai.mayari, microrts_ai.droplet],
            [microrts_ai.mayari, microrts_ai.naiveMCTSAI],
            ]

            if expert_reference_index == 1:
                opponents = [[ai_pair[1], ai_pair[0]] for ai_pair in opponents]

            num_runs = [ 200, 50, 200, 250, 250, 100, 300, 100, 250, 250, 250, 250, 250,
                200, 50, 200, 250, 250, 100, 300, 100, 250, 250, 250, 250, 250]

        expert_name_to_id = {
            "coacAI": 0,
            "mayari": 1,
            "workerRushAI": 2,
            "passiveAI": 3,
            "lightRushAI": 4,
            "randomAI": 5,
            "randomBiasedAI": 6,
            "rojo": 7,
            "mixedBot": 8,
            "mixedbot": 8,
            "izanagi": 9,
            "droplet": 10,
            "tiamat": 11,
            "naiveMCTSAI": 12,
        }

        for index, ai_pair in enumerate(opponents):
            print(f"next opponent: {ai_pair[0].__name__} vs. {ai_pair[1].__name__}")
            env = MicroRTSBotGridVecEnv(
                max_steps=2048,
                ais=ai_pair,
                map_paths=["maps/16x16/basesWorkers16x16A.xml"],
                reference_indexes=[expert_reference_index],
            )
            env_transform = MicroRTSSpaceTransformbot(env)

            obs_batch, _, res = env_transform.reset()
            if expert_reference_index == 1:
                obs_batch = self._adjust_obs_to_player0(obs_batch, is_new_env=True)

            obsten = torch.zeros((0, 16, 16, 73), dtype=torch.int32)
            actten = torch.zeros((0, 256, 7), dtype=torch.int8)
            scten = torch.zeros((0, 11), dtype=torch.int8)
            ztorch = torch.zeros((0, 1), dtype=torch.int8)

            for ep in range(num_runs[index]):
                dones = np.array([False])
                obs_arr = []
                act_arr = []

                while not dones.all():
                    if self.args.render:
                        env_transform.render()

                    obs_arr.append(obs_batch)
                    res_arr = np.asarray(res)
                    if res_arr.ndim >= 2 and res_arr.shape[0] > expert_reference_index:
                        selected_res = res_arr[[expert_reference_index]]
                    else:
                        selected_res = res
                    scten = torch.cat(
                        [scten, self.get_scalar_features(obs_batch, selected_res, 1)], dim=0
                    )

                    obs_batch, _, dones, action, res, reward = env_transform.step("")
                    if expert_reference_index == 1:
                        obs_batch = self._adjust_obs_to_player0(obs_batch, is_new_env=False)

                    arr = np.zeros((256, 7), dtype=np.int64)
                    expert_actions = action[expert_reference_index]
                    if expert_reference_index == 1:
                        expert_actions = self._adjust_actions_to_player0(expert_actions)
                    for j in range(len(expert_actions)):
                        arr[expert_actions[j][0]] = expert_actions[j][1:]
                    act_arr.append(arr)

                if self.args.nurwins and reward.item() != 1:
                    pass
                else:
                    expert_name = ai_pair[expert_reference_index].__name__
                    expert_id = expert_name_to_id.setdefault(
                        expert_name, max(expert_name_to_id.values(), default=-1) + 1
                    )
                    obsten = torch.cat((obsten, torch.tensor(np.array(obs_arr)).squeeze(1)), dim=0)
                    actten = torch.cat((actten, torch.tensor(np.array(act_arr))), dim=0)
                    ztorch = torch.cat(
                        (
                            ztorch,
                            torch.tensor(expert_id).repeat(len(obs_arr), 1),
                        ),
                        dim=0,
                    )

                if ((ep + 1) % 50 == 0) or ((ep + 1) == num_runs[index]):
                    replay_path = os.path.join(
                        self.replay_dir, f"replay_{index}_up_to_ep{ep + 1}.pt.zst"
                    )
                    print("Collecting Data ep: ", index, " ", ep)
                    with open(replay_path, "wb") as f:
                        compressor = zstd.ZstdCompressor(level=1).stream_writer(f)
                        buffer = io.BytesIO()
                        torch.save(
                            {
                                "obs": obsten,
                                "act": actten,
                                "sc": scten,
                                "z": ztorch,
                            }, buffer
                        )
                        compressor.write(buffer.getvalue())
                        compressor.flush(zstd.FLUSH_FRAME)

                    obsten = torch.zeros((0, 16, 16, 73), dtype=torch.int32)
                    actten = torch.zeros((0, 256, 7), dtype=torch.int32)
                    scten = torch.zeros((0, 11), dtype=torch.int8)
                    ztorch = torch.zeros((0, 1), dtype=torch.int8)

            env_transform.close()
            env.close()

    def _resolve_ai_callable(self, ai_name: str):
        ai_callable = getattr(microrts_ai, ai_name, None)
        if ai_callable is None:
            raise ValueError(
                f"Unknown AI '{ai_name}' for BC replay collection. "
                "Use names from gym_microrts.microrts_ai, e.g. workerRushAI."
            )
        return ai_callable

    def _train_epoch(self, loader, optimizer, warmup_epochs, epoch):
        train_loss_sum = 0.0
        train_count = 0
        alpha = max(0.0, 1.0 - (epoch - self.start_epoch) / warmup_epochs)

        for obs, expert_actions, sc, zt in loader:
            obs = obs.to(self.device)
            expert_actions = expert_actions.to(self.device)
            sc = sc.to(self.device)
            zt = zt.to(self.device)

            optimizer.zero_grad()

            z_embed = self.agent.z_embedding(zt)
            z_enc = self.agent.z_encoder(obs.view(obs.size(0), -1))
            z = alpha * z_embed.squeeze(1) + (1 - alpha) * z_enc
            unit_bonus_distr = None
            if getattr(self.agent, "unit_exploiters", False):
                unit_bonus_distr = torch.zeros((obs.size(0), 4), device=self.device)

            loss = self.agent.bc_loss_fn(obs, sc, expert_actions, z, unit_bonus_distr=unit_bonus_distr)

            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item() * obs.size(0)
            train_count += obs.size(0)

        return train_loss_sum / max(train_count, 1), optimizer.param_groups[0]['lr']

    def _log_epoch_metrics(self, epoch, train_loss, lr):
        if self.writer is not None:
            self.writer.add_scalar("charts/BCepoch", epoch)
            self.writer.add_scalar("charts/BCLossTrain", train_loss, epoch)
            self.writer.add_scalar("charts/BClearnrate", lr, epoch)
            self.writer.flush()

        if self.wandb_log_fn is not None:
            self.wandb_log_fn(
                {
                    "charts/BCepoch": epoch,
                    "charts/BCLossTrain": train_loss,
                    "charts/BClearnrate": lr,
                },
                step=epoch,
            )

    def _finalize_logging(self):
        if self.writer is not None:
            self.writer.flush()
