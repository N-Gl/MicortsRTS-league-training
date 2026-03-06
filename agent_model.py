from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(
        self,
        probs: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        validate_args: Optional[bool] = None,
        masks: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (logits.device if logits is not None else torch.device("cpu"))
        if masks is None or masks.numel() == 0:
            self.masks = None
            super().__init__(probs=probs, logits=logits, validate_args=validate_args)
            return

        self.masks = masks.type(torch.bool).to(self.device)
        masked_logits = torch.where(
            self.masks,
            logits,
            torch.tensor(-1e8, device=self.device),
        )
        super().__init__(probs=probs, logits=masked_logits, validate_args=validate_args)

    def entropy(self) -> torch.Tensor:
        if self.masks is None or self.masks.numel() == 0:
            return super().entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(
            self.masks,
            p_log_p,
            torch.tensor(0.0, device=self.device),
        )
        return -p_log_p.sum(-1)


class ZSampler(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)


class ScalarFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 32, num_layers: int = 2):
        super().__init__()
        layers = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        reduced_channels = max(1, channels // 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            layer_init(nn.Conv2d(channels, reduced_channels, kernel_size=1)),
            nn.GELU(),
            layer_init(nn.Conv2d(reduced_channels, channels, kernel_size=1)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.conv1(x))
        out = self.conv2(out)
        w = self.se(out)
        out = out * w
        return F.gelu(out + x)


class Agent(nn.Module):
    def __init__(
        self,
        action_plane_nvec: Sequence[int],
        device: torch.device,
        mapsize: int = 16 * 16,
        lstm_hidden: int = 384,
        lstm_layers: int = 3,
        initial_weights: Optional[Union[str, Dict[str, torch.Tensor]]] = None,
        logits: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        unit_exploiters: Optional[bool] = False # TODO übergebe den Wert aus args
    ):
        super().__init__()
        self.device = device
        self.mapsize = mapsize
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.sp_logits = logits
        self._values = values
        self.steps = 0
        self.checkpoint_step = 0
        self.action_plane_nvec = action_plane_nvec # used in case of copying Agents without envs
        nvec = np.asarray(action_plane_nvec)
        self.action_nvec_list = nvec.tolist()
        self.num_action_params = len(self.action_nvec_list)
        self.action_dim = int(nvec.sum())
        self.unit_exploiters = unit_exploiters
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(73, 64, kernel_size=3, stride=2, padding=1)),
            nn.GELU(),
            ResBlock(64),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.GELU(),
            ResBlock(64),
            layer_init(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)),
            nn.GELU(),
            ResBlock(64),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, 256)),
            nn.ReLU(),
        )
        self.z_embedding = nn.Embedding(num_embeddings=2, embedding_dim=8)
        self.z_encoder = ZSampler(obs_dim=self.mapsize * 73, z_dim=8)
        self.scalar_encoder = ScalarFeatureEncoder(11)
        # print(envsT.action_plane_space.nvec.sum())

        if self.unit_exploiters:
            self.actor = layer_init(
                nn.Linear(256 + 32 + 8 + 4, self.mapsize * self.action_dim),
                std=0.01,
            )
            self.critic = layer_init(nn.Linear(256 + 32 + 8 + 4, 1), std=1)
        else:
            self.actor = layer_init(
                nn.Linear(256 + 32 + 8, self.mapsize * self.action_dim),
                std=0.01,
            )
            self.critic = layer_init(nn.Linear(256 + 32 + 8, 1), std=1)
        if initial_weights is not None:
            self.set_weights(initial_weights)

    def forward(self, x: torch.Tensor, sc: torch.Tensor, z: torch.Tensor, unit_bonus_distr: torch.Tensor = None) -> torch.Tensor:
        sc_feat = self.scalar_encoder(sc)
        obs_feat = self.network(x.permute((0, 3, 1, 2)))
        z = z.view(z.size(0), -1)
        if self.unit_exploiters:
            if unit_bonus_distr is None:
                raise ValueError("unit_bonus_distr must be provided when unit_exploiters is True.") # TODO remove exception later
                unit_bonus_distr = torch.zeros((x.size(0), 4), device=self.device)
            
            unit_bonus_distr = unit_bonus_distr.view(unit_bonus_distr.size(0), -1)
            return torch.cat([obs_feat, sc_feat, z, unit_bonus_distr], dim=-1)
        return torch.cat([obs_feat, sc_feat, z], dim=-1)
    # {f"Current: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB | Max: {torch.cuda.max_memory_allocated(0)/1024**2:.2f} MB"}

    def set_weights(self, weights: Union[str, Dict[str, torch.Tensor]]) -> None:
        if isinstance(weights, str):
            weights = torch.load(weights, map_location=self.device, weights_only=True)
        if not isinstance(weights, dict):
            raise NotImplementedError("Only loading from dict or filepath is implemented.")
        model_state = self.state_dict()
        patched_weights = dict(weights)

        # Backward/forward compatibility for checkpoints trained with/without unit_exploiters.
        for key in ("actor.weight", "critic.weight"):
            if key not in patched_weights or key not in model_state:
                continue
            src = patched_weights[key]
            dst = model_state[key]
            if src.shape == dst.shape:
                continue
            if src.ndim != 2 or dst.ndim != 2 or src.shape[0] != dst.shape[0]:
                continue
            src_width = src.shape[1]
            dst_width = dst.shape[1]

            # Old checkpoint (without unit_bonus features) -> new model (with unit_bonus features):
            # append 4 columns initialized to 0 so the new features do not affect outputs initially.
            if self.unit_exploiters and src_width + 4 == dst_width:
                adapted = torch.zeros_like(dst)
                adapted[:, :src_width] = src.to(device=dst.device, dtype=dst.dtype)
                patched_weights[key] = adapted
                print(
                    f"Adjusted checkpoint tensor '{key}' from {tuple(src.shape)} to {tuple(dst.shape)} "
                    "by appending 4 zero-initialized feature columns."
                )
                continue

            # New checkpoint (with unit_bonus features) -> old model (without unit_bonus features).
            if src_width == dst_width + 4:
                adapted = src[:, :dst_width].to(device=dst.device, dtype=dst.dtype)
                patched_weights[key] = adapted
                print(
                    f"Adjusted checkpoint tensor '{key}' from {tuple(src.shape)} to {tuple(dst.shape)} "
                    "by dropping the last 4 feature columns."
                )

        self.load_state_dict(patched_weights, strict=True)

    def get_steps(self) -> int:
        """How many agent steps the agent has been trained for."""
        return self.steps

    def bc_loss_fn(self, obs: torch.Tensor, sc: torch.Tensor, expert_actions: torch.Tensor, z: torch.Tensor, unit_bonus_distr: torch.Tensor = None) -> torch.Tensor:
        B, H, W, _ = obs.shape
        features = self.forward(obs, sc, z, unit_bonus_distr)
        flat = self.actor(features)
        grid_logits = flat.view(-1, self.action_dim)
        split_logits = torch.split(grid_logits, self.action_nvec_list, dim=1)
        invalid_action_masks = torch.ones(
            (B * H * W, self.action_dim + 1),
            dtype=torch.bool,
            device=self.device,
        )
        split_invalid_action_masks = torch.split(
            invalid_action_masks[:, 1:],
            self.action_nvec_list,
            dim=1,
        )

        multi_categoricals = [
            CategoricalMasked(logits=logits, masks=masks, device=self.device)
            for logits, masks in zip(split_logits, split_invalid_action_masks)
        ]

        expert_actions = expert_actions.view(-1, expert_actions.shape[-1]).T
        logprob = torch.stack(
            [categorical.log_prob(action) for action, categorical in zip(expert_actions, multi_categoricals)]
        )
        bc_loss = -logprob.sum() / (B * H * W)

        kl = 0.0
        for cat in multi_categoricals:
            probs = cat.probs
            policy_dist = Categorical(logits=cat.logits)
            expert_dist = Categorical(probs=probs)
            kl += kl_divergence(expert_dist, policy_dist).sum()
        kl_loss = kl / (B * H * W)
        return bc_loss + 0.01 * kl_loss
    
    def get_unique_agents(self, active_league_agents, selfplay_only = False, output_league_agents = False) -> Dict:
        unique_agents = {}
        if selfplay_only:
            for idx, _ in enumerate(active_league_agents):
                unique_agents.setdefault(self, []).append(idx)
        elif output_league_agents:
            for idx, p in enumerate(active_league_agents):
                unique_agents.setdefault(p, []).append(idx)
        else:
            for idx, p in enumerate(active_league_agents):
                unique_agents.setdefault(p.agent, []).append(idx)
        return unique_agents

    def get_action(
        self,
        x: torch.Tensor,
        sc: torch.Tensor,
        z: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        invalid_action_masks: Optional[torch.Tensor] = None,
        envs=None,
        selfplay_envs: bool = False,
        num_selfplay_envs: int = 0,
        logits: Optional[torch.Tensor] = None,
        process_envs: Optional[Any] = None,
        dbg_deterministic_actions: bool = False,
        unit_bonus_distr: Optional[torch.Tensor] = None,
        return_entropy_per_param: bool = False,
    ):
        if process_envs is None:
            process_envs = range(envs.num_envs)
        if logits is None:
            logits = self.actor(self.forward(x, sc, z, unit_bonus_distr))
        grid_logits = logits.view(-1, self.action_dim)
        split_logits = torch.split(grid_logits, self.action_nvec_list, dim=1)
        # torch.where(x[29] != x[28])[0].shape[0] != 0 or torch.where(sc[29] != sc[28])[0].shape[0] != 0 or torch.where(z[29] != z[28])[0].shape[0] != 0
        # np.any([torch.where(z[i] != z[i+1])[0].shape[0] != 0 or torch.where(x[i] != x[i+1])[0].shape[0] != 0 or torch.where(sc[i] != sc[i+1])[0].shape[0] != 0 for i in range(0, num_selfplay_envs//2, 2)])

        if action is None:
            invalid_action_masks = torch.stack([torch.tensor(envs.debug_matrix_mask(i), dtype=torch.bool) for i in process_envs]).to(self.device)
            
            if selfplay_envs and num_selfplay_envs > 1:
                invalid_action_masks = invalid_action_masks.clone()
                upper = min(num_selfplay_envs, invalid_action_masks.shape[0])
                invalid_action_masks[1:upper:2] = invalid_action_masks[1:upper:2].flip(1, 2)
            # =====
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
        else:
            invalid_action_masks = invalid_action_masks.to(self.device)
            invalid_action_masks = invalid_action_masks.reshape(-1, invalid_action_masks.shape[-1])
            action = action.to(self.device)
            action = action.view(-1, action.shape[-1]).T

        split_invalid_action_masks = torch.split(
            invalid_action_masks[:, 1:],
            self.action_nvec_list,
            dim=1,
        )
        if action is None and selfplay_envs and num_selfplay_envs > 1 and envs is not None:
            self._adjust_selfplay_masks(split_invalid_action_masks, num_selfplay_envs, len(process_envs))
        multi_categoricals = [
            CategoricalMasked(logits=logits, masks=masks, device=self.device)
            for logits, masks in zip(split_logits, split_invalid_action_masks)
        ]

        if action is None:
            if dbg_deterministic_actions:
                action = torch.stack([categorical.logits.argmax(dim=-1) for categorical in multi_categoricals])
            else:
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])

        logprob = torch.stack(
            [categorical.log_prob(act) for act, categorical in zip(action, multi_categoricals)]
        )
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        num_predicted_parameters = self.num_action_params

        logprob = logprob.T.reshape(-1, self.mapsize, num_predicted_parameters)
        entropy = entropy.T.reshape(-1, self.mapsize, num_predicted_parameters)
        action = action.T.reshape(-1, self.mapsize, num_predicted_parameters)
        invalid_action_masks = invalid_action_masks.view(-1, self.mapsize, self.action_dim + 1)

        logprob_sum = logprob.sum(1).sum(1)
        entropy_sum = entropy.sum(1).sum(1)
        if return_entropy_per_param:
            return action, logprob_sum, entropy_sum, invalid_action_masks, entropy
        return action, logprob_sum, entropy_sum, invalid_action_masks
    
    
    def selfplay_get_z_encoded_features(self, args, device, z_features, next_obs, step, unique_agents, sp_next_obs=None):
        '''encode z-features per agent (main/supervised) so each policy uses its own encoder'''
        if sp_next_obs is not None:
            next_obs = torch.cat([sp_next_obs, next_obs], dim=0)
        flat_next_obs = next_obs.view(args.num_envs, -1)
        next_z_features = torch.empty((args.num_envs, z_features.shape[2]), device=device)
        for cur_agent, indices in unique_agents.items():
            cur_agent = getattr(cur_agent, "agent", cur_agent)
            if not indices:
                continue
            index_tensor = torch.as_tensor(indices, device=device)
            encoded = cur_agent.z_encoder(flat_next_obs[index_tensor].view(len(index_tensor), -1))
            next_z_features[index_tensor] = encoded
            
        return next_z_features

    def selfplay_get_action(
        self,
        x: torch.Tensor,
        sc: torch.Tensor,
        z: torch.Tensor,
        num_selfplay_envs: int,
        num_envs: int,
        action: Optional[torch.Tensor] = None,
        invalid_action_masks: Optional[torch.Tensor] = None,
        envs=None,
        active_league_agents = None,
        unique_agents: Optional[Dict] = None,
        dbg_deterministic_actions: bool = False,
        unit_bonus_distr: Optional[torch.Tensor] = None,
    ):
        '''
        returns action, logprob, entropy, invalid_action_masks for selfplay and bot envs combined.
        Also returns action, logprob, entropy, invalid_action_masks for not main Agents
        '''
        # active_league_agents: all activly trained Agents (bot and selfplay Agents)
        
        if self.sp_logits is None:
            # self.sp_logits = torch.empty((num_envs, self.mapsize * self.action_dim), device=self.device)
            self.sp_logits = torch.zeros((num_selfplay_envs, self.mapsize * self.action_dim), device=self.device)

        if unique_agents is None:
            unique_agents = self.get_unique_agents(active_league_agents, output_league_agents = True)

        bot_replacements = []
        # TODO (optimize): indices als Tensor statt Python-Liste
        for agent, indices in unique_agents.items():

            if agent.unit_bonus_distr is not None and unit_bonus_distr is not None:
                cur_unit_bonus_distr = unit_bonus_distr[indices]
            else:
                cur_unit_bonus_distr = None
            
            agent = getattr(agent, "agent", agent)

            if not indices:
                continue

            if isinstance(agent, Bot_Agent):
                bot_replacements.append((agent, indices))
                # placeholder logits; will be overridden by bot actions later
                self.sp_logits[indices] = 0.0
                continue
            
            if agent is not self:
                with torch.no_grad():
                    subset_logits = agent.actor(agent.forward(x[indices], sc[indices], z[indices], cur_unit_bonus_distr))
            else:
                subset_logits = agent.actor(agent.forward(x[indices], sc[indices], z[indices], cur_unit_bonus_distr))


            if agent is not self:
                # TODO (league training): sollte man wirklich alle non_main_agenten detatchen? (Wahrscheinlich schon) (oder sogar torch.no_grad()) (auch in selfplay_get_value?)
                subset_logits = subset_logits.detach()
            self.sp_logits[indices] = subset_logits# .to(self.device)


        action, logprob, entropy, invalid_action_masks = self.get_action(
            x,
            sc,
            z,
            action,
            invalid_action_masks,
            envs=envs,
            selfplay_envs=num_selfplay_envs > 0,
            num_selfplay_envs=num_selfplay_envs,
            logits=self.sp_logits[:num_selfplay_envs],
            dbg_deterministic_actions=dbg_deterministic_actions,
            unit_bonus_distr=unit_bonus_distr
        )

        for bot_agent, indices in bot_replacements:
            bot_actions, bot_logprob, bot_entropy, bot_invalid_masks = bot_agent.get_action(
                len(indices)
            )
            action[indices] = bot_actions
            logprob[indices] = bot_logprob
            entropy[indices] = bot_entropy
            invalid_action_masks[indices] = bot_invalid_masks




        return action, logprob, entropy, invalid_action_masks
    

    def _adjust_selfplay_masks(self, split_masks, num_selfplay_envs: int, total_envs: int) -> None:
        mapsize = self.mapsize
        limit = min(num_selfplay_envs, total_envs)
        direction_indices = [idx for idx in range(1, min(5, len(split_masks)))]
        attack_index = 6 if 6 < len(split_masks) else None
        for env_idx in range(1, limit, 2):
            start = env_idx * mapsize
            end = start + mapsize
            for idx in direction_indices:
                split_masks[idx][start:end] = torch.roll(split_masks[idx][start:end], shifts=2, dims=1)
            if attack_index is not None:
                split_masks[attack_index][start:end] = torch.flip(split_masks[attack_index][start:end], dims=[1])

    def get_value(self, x: torch.Tensor, sc: torch.Tensor, z: torch.Tensor, unit_bonus_distr: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.critic(self.forward(x, sc, z, unit_bonus_distr))
    
    def selfplay_and_Bot_get_value(self, x: torch.Tensor, sc: torch.Tensor, z: torch.Tensor, active_league_agents=None, num_selfplay_envs=0, num_envs=0, unique_agents=None, learning_indices=None, unit_bonus_distr: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        returns value for selfplay and bot envs combined.
        Also returns value for not main Agents
        only_player_0: only calculate for player 0 in selfplay envs not changing value for player 1 (initially 0) (to save computation time) Depricated: replaced with learning_indices
        '''
        if self._values is None:
            self._values = torch.zeros(num_envs, device=self.device)
            # debugging: torch.fill_(self._values, -1000000.0)

        if unique_agents is None:
            unique_agents = self.get_unique_agents(active_league_agents, output_league_agents = True)

        for agent, indices in unique_agents.items():
            agent = getattr(agent, "agent", agent)
            if not indices:
                continue
            if learning_indices is not None:
                indices = [idx for idx in indices if idx in learning_indices]
                if not indices:
                    continue
            if agent is not self:
                with torch.no_grad():
                    self._values[indices] = agent.get_value(x[indices], sc[indices], z[indices], unit_bonus_distr[indices] if unit_bonus_distr is not None else None).flatten()# .to(self.device).detach()
            else:
                self._values[indices] = agent.get_value(x[indices], sc[indices], z[indices], unit_bonus_distr[indices] if unit_bonus_distr is not None else None).flatten()# .to(self.device)

        return self._values#.clone()
    

    def remove_last_bot_env(self):
        if self._values is not None:
            self._values = self._values[:-1]

    def add_bot_env(self):
        if self._values is not None:
            self._values = torch.cat([self._values, torch.zeros(1, device=self.device)], dim=0)

        


def build_agent(
    action_plane_nvec: Sequence[int],
    device: torch.device,
    unit_exploiters: bool = False,
) -> Agent:
    """Factory-methode"""
    return Agent(
        action_plane_nvec=action_plane_nvec,
        device=device,
        unit_exploiters=unit_exploiters,
    ).to(device)



class Bot_Agent(nn.Module):
    """
    Lightweight agent wrapper that forwards actions from a scripted microrts bot
    while matching the Agent API expected by the selfplay helpers.
    """

    def __init__(self, envsT, env_ids, bot, device, mapsize: int = 16 * 16, player_id = 1):
        super().__init__()
        self.device = device
        self.envsT = envsT
        self.env_ids = env_ids
        self.bot = bot(self.envsT.interface.real_utt)
        self.mapsize = mapsize
        self.action_dim = int(np.asarray(self.envsT.action_plane_space.nvec).sum())
        self.player_id = player_id

    def forward(self, x: torch.Tensor, sc: torch.Tensor, z: torch.Tensor, unit_bonus_distr: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.tensor(x.shape[0], device=self.device)
    
    def actor(self, features: torch.Tensor) -> torch.Tensor:
        num_envs = features.item()
        return torch.zeros((num_envs, self.mapsize * self.action_dim), device=self.device)
    
    def get_action(self, num_envs: int):
        actions = torch.zeros(
            (num_envs, self.envsT.height * self.envsT.width, len(self.envsT.action_plane_space.nvec)),
            dtype=torch.int64,
            device=self.device,
        )
        player_id = self.player_id
        for env_id in self.env_ids:
            gs = self.envsT.interface.get_game_state(env_id)
            player_action = self.bot.getAction(player_id, gs)
            bot_pairs = list(player_action.getActions())
            self._pairs_to_actions_inplace(bot_pairs, actions, self.envsT, env_idx=env_id)

        mask = torch.ones(
            (num_envs, self.envsT.height * self.envsT.width, self.action_dim + 1),
            dtype=torch.bool,
            device=self.device,
        )
        return actions, 0, 0, mask


    def _pairs_to_actions_inplace(self, pairs, act: torch.Tensor, envsT, env_idx=0):
        h, w = envsT.height, envsT.width

        def attack_idx(unit, ua):
            dx = dy = 0
            if hasattr(ua, "getLocationX") and hasattr(ua, "getLocationY"):
                dx = int(ua.getLocationX()) - int(unit.getX())
                dy = int(ua.getLocationY()) - int(unit.getY())
            return max(0, min((dy + 3) * 7 + (dx + 3), 48))

        for p in pairs:
            unit = getattr(p, "m_a", None) or getattr(p, "first", None) or p.getA()
            ua   = getattr(p, "m_b", None) or getattr(p, "second", None) or p.getB()
            x, y = int(unit.getX()), int(unit.getY())
            pos  = y * w + x
            t = int(ua.getType())
            act[env_idx, pos, 0] = t
            if t == 1:
                act[env_idx, pos, 1] = int(ua.getDirection())
            elif t == 2:
                act[env_idx, pos, 2] = int(ua.getDirection())
            elif t == 3:
                act[env_idx, pos, 3] = int(ua.getDirection())
            elif t == 4:
                act[env_idx, pos, 4] = int(ua.getDirection())
                act[env_idx, pos, 5] = int(getattr(ua.getUnitType(), "ID", 0))
            elif t in (5, 6):
                act[env_idx, pos, 6] = attack_idx(unit, ua)


        

   
