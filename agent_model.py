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

    ):
        super().__init__()
        self.device = device
        self.mapsize = mapsize
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.logits = logits
        self._values = values
        self.steps = 0
        self.checkpoint_step = 0
        self.action_plane_nvec = action_plane_nvec # used in case of copying Agents without envs
        nvec = np.asarray(action_plane_nvec)
        self.action_nvec_list = nvec.tolist()
        self.num_action_params = len(self.action_nvec_list)
        self.action_dim = int(nvec.sum())

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
        self.actor = layer_init(
            nn.Linear(256 + 32 + 8, self.mapsize * self.action_dim),
            std=0.01,
        )
        self.critic = layer_init(nn.Linear(256 + 32 + 8, 1), std=1)
        if initial_weights is not None:
            self.set_weights(initial_weights)

    def forward(self, x: torch.Tensor, sc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        sc_feat = self.scalar_encoder(sc)
        obs_feat = self.network(x.permute((0, 3, 1, 2)))
        z = z.view(z.size(0), -1)
        return torch.cat([obs_feat, sc_feat, z], dim=-1)

    def set_weights(self, weights: Union[str, Dict[str, torch.Tensor]]) -> None:
        if isinstance(weights, dict):
            self.load_state_dict(weights)
        elif isinstance(weights, str):
            self.load_state_dict(torch.load(weights, map_location=self.device, weights_only=True))
        else:
            raise NotImplementedError("Only loading from dict or filepath is implemented.")

    def get_steps(self) -> int:
        """How many agent steps the agent has been trained for."""
        return self.steps

    def bc_loss_fn(self, obs: torch.Tensor, sc: torch.Tensor, expert_actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = obs.shape
        features = self.forward(obs, sc, z)
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
    
    def get_unique_agents(self, active_league_agents, selfplay_only = False) -> Dict:
        unique_agents = {}
        if selfplay_only:
            for idx, _ in enumerate(active_league_agents):
                unique_agents.setdefault(self, []).append(idx)
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
    ):
        if logits is None:
            logits = self.actor(self.forward(x, sc, z))
        grid_logits = logits.view(-1, self.action_dim)
        split_logits = torch.split(grid_logits, self.action_nvec_list, dim=1)
        # torch.where(x[29] != x[28])[0].shape[0] != 0 or torch.where(sc[29] != sc[28])[0].shape[0] != 0 or torch.where(z[29] != z[28])[0].shape[0] != 0
        # np.any([torch.where(z[i] != z[i+1])[0].shape[0] != 0 or torch.where(x[i] != x[i+1])[0].shape[0] != 0 or torch.where(sc[i] != sc[i+1])[0].shape[0] != 0 for i in range(0, num_selfplay_envs//2, 2)])

        if action is None:
            # if hasattr(envs, "get_action_mask"):
            #     mask_np = envs.get_action_mask()
            #     src_mask = getattr(envs, "source_unit_mask", None)
            #     if src_mask is None:
            #         src_mask = mask_np.any(axis=-1, keepdims=True)
            #     else:
            #         src_mask = src_mask.reshape(envs.num_envs, -1, 1)
# 
            #     if selfplay_envs and num_selfplay_envs > 1:
            #         upper = min(num_selfplay_envs, mask_np.shape[0])
            #         h = w = int(np.sqrt(mask_np.shape[1]))
            #         mask_p1 = mask_np[1:upper:2].reshape(-1, h, w, mask_np.shape[-1])
            #         mask_p1 = np.flip(mask_p1, axis=(1, 2))
            #         mask_np[1:upper:2] = mask_p1.reshape(-1, h * w, mask_np.shape[-1])
            #         src_p1 = src_mask[1:upper:2].reshape(-1, h, w, 1)
            #         src_p1 = np.flip(src_p1, axis=(1, 2))
            #         src_mask[1:upper:2] = src_p1.reshape(-1, h * w, 1)
# 
            #     mask_np = np.concatenate([src_mask, mask_np], axis=2)
            #     invalid_action_masks = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
            # else:
            #     raise AttributeError("Environment does not provide get_action_mask for action filtering.")
            # =====
            invalid_action_masks = torch.stack([torch.tensor(envs.debug_matrix_mask(i), dtype=torch.bool) for i in range(envs.num_envs)]).to(self.device)
            
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
            self._adjust_selfplay_masks(split_invalid_action_masks, num_selfplay_envs, envs.num_envs)
        multi_categoricals = [
            CategoricalMasked(logits=logits, masks=masks, device=self.device)
            for logits, masks in zip(split_logits, split_invalid_action_masks)
        ]

        if action is None:
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

        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks
    
    
    def selfplay_get_z_encoded_features(self, args, device, z_features, next_obs, step, unique_agents):
        '''encode z-features per agent (main/supervised) so each policy uses its own encoder'''
        flat_next_obs = next_obs.view(args.num_envs, -1)
        next_z_features = torch.empty((args.num_envs, z_features.shape[2]), device=device)
        for cur_agent, indices in unique_agents.items():
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
        unique_agents: Optional[Dict] = None
    ):
        '''
        returns action, logprob, entropy, invalid_action_masks for selfplay and bot envs combined.
        Also returns action, logprob, entropy, invalid_action_masks for not main Agents
        '''
        # active_league_agents: all activly trained Agents (bot and selfplay Agents)
        
        # logits = torch.zeros((num_envs, self.mapsize * self.action_dim), device=self.device)
        if self.logits is None:
            self.logits = torch.empty((num_envs, self.mapsize * self.action_dim), device=self.device)

        if unique_agents is None:
            unique_agents = self.get_unique_agents(active_league_agents)

        # TODO (optimize): indices als Tensor statt Python-Liste
        bot_replacements = []
        for agent, indices in unique_agents.items():
            if not indices:
                continue
            if isinstance(agent, Bot_Agent):
                bot_replacements.append((agent, indices))
                # placeholder logits; will be overridden by bot actions later
                self.logits[indices] = 0.0
                continue

            if agent is not self:
                with torch.no_grad():
                    subset_logits = agent.actor(agent.forward(x[indices], sc[indices], z[indices]))
            else:
                subset_logits = agent.actor(agent.forward(x[indices], sc[indices], z[indices]))

            if agent is not self:
                # TODO (league training): sollte man wirklich alle non_main_agenten detatchen? (Wahrscheinlich schon) (oder sogar torch.no_grad()) (auch in selfplay_get_value?)
                subset_logits = subset_logits.detach()
            self.logits[indices] = subset_logits# .to(self.device)

        action, logprob, entropy, invalid_action_masks = self.get_action(
            x,
            sc,
            z,
            action,
            invalid_action_masks,
            envs=envs,
            selfplay_envs=num_selfplay_envs > 0,
            num_selfplay_envs=num_selfplay_envs,
            logits=self.logits
        )
        for bot_agent, indices in bot_replacements:
            bot_actions, _, _, bot_masks = bot_agent.get_action(
                x,
                sc,
                z,
                envs=envs,
                selfplay_envs=num_selfplay_envs > 0,
                num_selfplay_envs=num_selfplay_envs,
            )
            action[indices] = bot_actions[indices]
            invalid_action_masks[indices] = bot_masks[indices]
            logprob[indices] = 0.0
            entropy[indices] = 0.0

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

    def get_value(self, x: torch.Tensor, sc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.critic(self.forward(x, sc, z))
    
    def selfplay_get_value(self, x: torch.Tensor, sc: torch.Tensor, z: torch.Tensor, active_league_agents=None, num_selfplay_envs=0, num_envs=0, unique_agents=None, only_player_0=False) -> torch.Tensor:
        '''
        returns value for selfplay and bot envs combined.
        Also returns value for not main Agents
        only_player_0: only calculate for player 0 in selfplay envs not changing value for player 1 (initially 0) (to save computation time)
        '''
        if self._values is None:
            self._values = torch.zeros(num_envs, device=self.device)
            # debugging: torch.fill_(self._values, -1000000.0)

        if unique_agents is None:
            unique_agents = self.get_unique_agents(active_league_agents)

        for agent, indices in unique_agents.items():
            if not indices:
                continue
            if only_player_0:
                indices = [i for i in indices if i % 2 == 0 or i >= num_selfplay_envs]
                if not indices:
                    continue
            if agent is not self:
                with torch.no_grad():
                    self._values[indices] = agent.get_value(x[indices], sc[indices], z[indices]).flatten()# .to(self.device).detach()
            else:
                self._values[indices] = agent.get_value(x[indices], sc[indices], z[indices]).flatten()# .to(self.device)

        return self._values#.clone()

        


def build_agent(action_plane_nvec: Sequence[int], device: torch.device) -> Agent:
    """Factory-methode"""
    return Agent(action_plane_nvec=action_plane_nvec, device=device).to(device)


class _ZeroZEncoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0] if obs is not None else 0
        return torch.zeros((batch, 8), dtype=torch.long, device=self.device)


class Bot_Agent(nn.Module):
    """
    Lightweight agent wrapper that forwards actions from a scripted microrts bot
    while matching the Agent API expected by the selfplay helpers.
    """

    def __init__(self, action_plane_nvec: Sequence[int], device: torch.device, bot: Any):
        super().__init__()
        self.device = device
        nvec = np.asarray(action_plane_nvec)
        self.action_nvec_list = nvec.tolist()
        self.action_dim = int(nvec.sum())
        self.num_action_params = len(self.action_nvec_list)
        self.mapsize = 16 * 16
        self.bot_factory = bot
        self.bot = None
        self.unit_type_lookup: Optional[Dict[str, int]] = None
        self.z_encoder = _ZeroZEncoder(device)

    def _ensure_bot(self, envs) -> None:
        if self.bot is not None:
            return
        interface = getattr(envs, "interface", envs)
        utt = getattr(interface, "real_utt", None)
        if utt is None:
            raise ValueError("Bot_Agent benötigt eine Umgebung mit real_utt, um den Bot zu initialisieren.")
        self.bot = self.bot_factory(utt) if callable(self.bot_factory) else self.bot_factory

    def _ensure_unit_type_lookup(self, interface) -> None:
        if self.unit_type_lookup is not None:
            return
        utt_dict = getattr(interface, "utt", None)
        lookup: Dict[str, int] = {}
        if utt_dict:
            for idx, unit_type in enumerate(utt_dict.get("unitTypes", [])):
                name = unit_type.get("name") or str(idx)
                lookup[name] = idx
                lookup[str(unit_type.get("ID", idx))] = idx
        self.unit_type_lookup = lookup

    def _unit_type_index(self, unit_type) -> int:
        if unit_type is None or self.unit_type_lookup is None:
            return 0
        for attr in ("getName", "name"):
            value = getattr(unit_type, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if isinstance(value, str) and value in self.unit_type_lookup:
                return int(self.unit_type_lookup[value])
        for attr in ("getID", "ID"):
            value = getattr(unit_type, attr, None)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    value = None
            if value is not None:
                return int(self.unit_type_lookup.get(str(value), value))
        return 0

    def _safe_direction(self, unit_action) -> int:
        direction = getattr(unit_action, "getDirection", None)
        if callable(direction):
            try:
                return int(direction())
            except Exception:
                return 0
        return 0

    def _attack_index(self, unit, unit_action) -> int:
        dx = dy = 0
        if hasattr(unit_action, "getLocationX") and hasattr(unit_action, "getLocationY"):
            try:
                dx = int(unit_action.getLocationX()) - int(unit.getX())
                dy = int(unit_action.getLocationY()) - int(unit.getY())
            except Exception:
                dx = dy = 0
        elif hasattr(unit_action, "getDirection"):
            try:
                direction = int(unit_action.getDirection())
                direction_map = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
                dx, dy = direction_map.get(direction, (0, 0))
            except Exception:
                dx = dy = 0
        attack_idx = (dy + 3) * 7 + (dx + 3)
        return int(np.clip(attack_idx, 0, 48))

    def _encode_unit_action(self, unit, unit_action, envs, env_idx) -> torch.Tensor:
        encoded = [0] * self.num_action_params
        try:
            action_type = int(unit_action.getType())
        except Exception:
            action_type = 0
        encoded[0] = action_type

        if action_type == 1:
            encoded[1] = self._safe_direction(unit_action)
        elif action_type == 2:
            encoded[2] = self._safe_direction(unit_action)
        elif action_type == 3:
            encoded[3] = self._safe_direction(unit_action)
        elif action_type == 4:
            encoded[4] = self._safe_direction(unit_action)
            unit_type = None
            if hasattr(unit_action, "getUnitType"):
                try:
                    unit_type = unit_action.getUnitType()
                except Exception:
                    unit_type = None
            encoded[5] = self._unit_type_index(unit_type)
        elif action_type in (5, 6):
            encoded[6] = self._attack_index(unit, unit_action)
        return torch.as_tensor(encoded, device=self.device)

    def _env_geometry(self, envs, env_idx: int):
        interface = getattr(envs, "interface", envs)
        env_h = interface.heights[env_idx]
        env_w = interface.widths[env_idx]
        pad_h = (envs.height - env_h) // 2 if hasattr(envs, "height") else 0
        pad_w = (envs.width - env_w) // 2 if hasattr(envs, "width") else 0
        return env_h, env_w, pad_h, pad_w

    def _split_pair(self, pair):
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            return pair[0], pair[1]
        unit = getattr(pair, "m_a", None) or getattr(pair, "first", None)
        action = getattr(pair, "m_b", None) or getattr(pair, "second", None)
        if unit is None and hasattr(pair, "getA"):
            try:
                unit = pair.getA()
            except Exception:
                unit = None
        if action is None and hasattr(pair, "getB"):
            try:
                action = pair.getB()
            except Exception:
                action = None
        return unit, action

    def _base_mask(self, envs) -> torch.Tensor:
        if hasattr(envs, "get_action_mask"):
            try:
                mask_np = envs.get_action_mask()
                source_mask = getattr(envs, "source_unit_mask", None)
                if source_mask is not None:
                    merged = np.concatenate(
                        [source_mask.reshape(envs.num_envs, -1, 1), mask_np], axis=2
                    )
                    return torch.as_tensor(merged, dtype=torch.bool, device=self.device)
            except Exception:
                pass
        mapsize = self.mapsize
        if hasattr(envs, "height") and hasattr(envs, "width"):
            mapsize = envs.height * envs.width
        return torch.zeros(
            (envs.num_envs, mapsize, self.action_dim + 1), dtype=torch.bool, device=self.device
        )

    def _actions_for_env(self, envs, env_idx: int, base_mask_env: torch.Tensor):
        interface = getattr(envs, "interface", envs)
        player = env_idx % 2 if env_idx < interface.num_selfplay_envs else interface.players[env_idx]
        try:
            gs = interface.get_game_state(env_idx)
            player_action = self.bot.getAction(player, gs)
            raw_pairs = list(player_action.getActions()) if player_action is not None else []
        except Exception:
            raw_pairs = []

        env_h, env_w, pad_h, pad_w = self._env_geometry(envs, env_idx)
        padded_w = envs.width if hasattr(envs, "width") else env_w
        mapsize = envs.height * envs.width if hasattr(envs, "height") else self.mapsize
        actions = torch.zeros(
            (mapsize, self.num_action_params), dtype=torch.long, device=self.device
        )
        mask = base_mask_env.clone()

        for pair in raw_pairs:
            unit, unit_action = self._split_pair(pair)
            if unit is None or unit_action is None:
                continue
            try:
                x, y = int(unit.getX()), int(unit.getY())
            except Exception:
                continue
            pos = (y + pad_h) * padded_w + (x + pad_w)
            if pos < 0 or pos >= mapsize:
                continue
            actions[pos] = self._encode_unit_action(unit, unit_action, envs, env_idx)
            mask[pos, 0] = True

        return actions, mask

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
    ):
        if envs is None:
            raise ValueError("Bot_Agent benötigt ein envs-Objekt, um Bot-Actions zu berechnen.")
        self._ensure_bot(envs)
        interface = getattr(envs, "interface", envs)
        self._ensure_unit_type_lookup(interface)
        if hasattr(envs, "height") and hasattr(envs, "width"):
            self.mapsize = envs.height * envs.width

        base_mask = self._base_mask(envs)
        actions = torch.zeros(
            (envs.num_envs, self.mapsize, self.num_action_params),
            dtype=torch.long,
            device=self.device,
        )
        invalid_masks = base_mask.clone()

        for env_idx in range(envs.num_envs):
            env_actions, env_mask = self._actions_for_env(envs, env_idx, base_mask[env_idx])
            actions[env_idx] = env_actions
            invalid_masks[env_idx] = env_mask

        logprob = torch.zeros(envs.num_envs, device=self.device)
        entropy = torch.zeros(envs.num_envs, device=self.device)
        return actions, logprob, entropy, invalid_masks

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
        active_league_agents=None,
        unique_agents: Optional[Dict] = None,
    ):
        return self.get_action(
            x,
            sc,
            z,
            action=action,
            invalid_action_masks=invalid_action_masks,
            envs=envs,
            selfplay_envs=num_selfplay_envs > 0,
            num_selfplay_envs=num_selfplay_envs,
        )

    def get_value(self, x: torch.Tensor, sc: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=self.device)

    def selfplay_get_value(
        self,
        x: torch.Tensor,
        sc: torch.Tensor,
        z: torch.Tensor,
        active_league_agents=None,
        num_selfplay_envs: int = 0,
        num_envs: int = 0,
        unique_agents=None,
        only_player_0: bool = False,
    ) -> torch.Tensor:
        return torch.zeros(x.shape[0], device=self.device)

    def selfplay_get_z_encoded_features(self, args, device, z_features, next_obs, step, unique_agents):
        return torch.zeros((args.num_envs, z_features.shape[2]), dtype=torch.long, device=self.device)
