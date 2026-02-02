import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import agent_model
import league
import ppo_update


def _make_dummy_args():
    return types.SimpleNamespace(Unit_reward_per_exploiter=False, unit_exploiters=False)


def _make_args(**kwargs):
    if "dbg_update_gaes" not in kwargs:
        kwargs["dbg_update_gaes"] = False
    return types.SimpleNamespace(**kwargs)


def old_gae(self, next_obs, next_scalar_features, next_z_features, b_rewards_winloss, b_rewards_attack, b_next_done, b_dones, b_values):
    dbg_last_value = self.agent.get_value(next_obs.to(self.device), next_scalar_features, next_z_features).reshape(1, -1)[:, self.indices]
    dbg_advantages = torch.zeros_like(b_rewards_winloss).to(self.device)
    dbg_lastgaelam = 0
    for t in reversed(range(self.args.num_steps)):
        if t == self.args.num_steps - 1:
            nextnonterminal = 1.0 - b_next_done
            nextvalues = dbg_last_value
        else:
            nextnonterminal = 1.0 - b_dones[t + 1]
            nextvalues = b_values[t + 1]
        delta = (
            b_rewards_winloss[t]
            + b_rewards_attack[t]
            + self.args.gamma * nextvalues * nextnonterminal
            - b_values[t]
        )
        dbg_advantages[t] = dbg_lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * dbg_lastgaelam
    dbg_returns = dbg_advantages  + b_values
    return dbg_returns, dbg_advantages


def old_ppo_update(self, next_obs, scalar_features, z_features, rewards_winloss, rewards_attack, next_done, dones, values, obs, actions, logprobs, invalid_action_masks, optimizer, action_space_shape, invalid_action_shape, step):
    import torch.nn as nn

    with torch.no_grad():
        last_value = self.agent.get_value(next_obs.to(self.device), scalar_features[step], z_features[step]).reshape(1, -1)
        advantages = torch.zeros_like(rewards_winloss).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.args.num_steps)):
            if t == self.args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = (
                rewards_winloss[t]
                + rewards_attack[t]
                + self.args.gamma * nextvalues * nextnonterminal
                - values[t]
            )
            advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    flat_z = z_features.reshape(-1, 8)
    flat_scalar = scalar_features.reshape(-1, 11)
    flat_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
    flat_actions = actions.reshape((-1,) + action_space_shape)
    flat_logprobs = logprobs.reshape(-1)
    flat_advantages = advantages.reshape(-1)
    flat_returns = returns.reshape(-1)
    flat_values = values.reshape(-1)
    flat_invalid_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)

    indices = np.arange(self.args.batch_size)
    for epoch_pi in range(self.args.update_epochs):
        np.random.shuffle(indices)

        for start in range(0, self.args.batch_size, self.args.minibatch_size):
            end = start + self.args.minibatch_size
            minibatch_ind = indices[start:end]
            mb_advantages = flat_advantages[minibatch_ind]

            if self.args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, new_logproba, entropy, _ = self.agent.get_action(
                flat_obs[minibatch_ind],
                flat_scalar[minibatch_ind],
                flat_z[minibatch_ind],
                flat_actions.long()[minibatch_ind],
                flat_invalid_masks[minibatch_ind],
                self.envs,
            )
            ratio = (new_logproba - flat_logprobs[minibatch_ind]).exp()

            approx_kl = (flat_logprobs[minibatch_ind] - new_logproba).mean()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            new_values = self.agent.get_value(
                flat_obs[minibatch_ind], flat_scalar[minibatch_ind], flat_z[minibatch_ind]
            ).view(-1)
            if self.args.clip_vloss:
                v_loss_unclipped = (new_values - flat_returns[minibatch_ind]) ** 2
                v_clipped = flat_values[minibatch_ind] + torch.clamp(
                    new_values - flat_values[minibatch_ind], -self.args.clip_coef, self.args.clip_coef
                )
                v_loss_clipped = (v_clipped - flat_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - flat_returns[minibatch_ind]) ** 2)

            if self.supervised_agent is not None:
                with torch.no_grad():
                    _, sl_logprobs, _, _ = self.supervised_agent.get_action(
                        flat_obs[minibatch_ind],
                        flat_scalar[minibatch_ind],
                        flat_z[minibatch_ind],
                        flat_actions.long()[minibatch_ind],
                        flat_invalid_masks[minibatch_ind],
                        self.envs,
                    )
                kl_div = F.kl_div(new_logproba, sl_logprobs, log_target=True, reduction="batchmean")
            else:
                kl_div = torch.tensor(0.0, device=self.device)
            kl_loss = self.args.kl_coeff * kl_div

            loss = pg_loss - self.args.ent_coef * entropy_loss + self.args.vf_coef * v_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
            optimizer.step()

            return approx_kl, pg_loss, v_loss, entropy_loss, kl_div, loss, optimizer


def _make_ready_to_checkpoint_args(**overrides):
    base = dict(
        selfplay_ready_save_interval=10,
        selfplay_save_interval=100,
        main_selfplay_save_interval=100,
        main_exploiter_selfplay_save_interval=100,
        league_exploiter_selfplay_save_interval=100,
        num_selfplay_envs=2,
        num_bot_envs=0,
        num_main_envs=1,
        num_envs_per_main_exploiters=1,
        num_envs_per_league_exploiters=1,
        main_winrate_threshold=0.7,
        main_exploiter_winrate_threshold=0.7,
        league_exploiter_winrate_threshold=0.7,
        save_gpu_memory=False,
        exp_name="test_exp",
        total_timesteps=30000000,
        global_step=1000,
        checkpoint_end_buffer_steps=5000,
        main_PFSP_prob=0.7,
        main_SP_prob=0.1,
        Unit_reward_per_exploiter=False,
        unit_exploiters=False,
        use_no_draw_winrates=True,
        main_exploiter_ready_use_historicals=True,
    )
    base.update(overrides)
    return _make_args(**base)


def _make_match_args(**overrides):
    base = dict(
        sp=False,
        pfsp=True,
        pfsp_min_prob_factor=0.0,
        main_PFSP_prob=0.7,
        main_SP_prob=0.1,
        main_winrate_threshold=0.7,
        main_exploiter_no_draw_winrate_threshold=0.7,
        main_exploiter_vs_main_winrate_threshold=0.5,
        main_exploiter_winrate_threshold=0.7,
        main_exploiter_ready_use_historicals=True,
        main_exploiter_max_historical_winrate_threshold=2,
        main_exploiter_main_prob_when_max_historical_winrate=0,
        main_exploiter_pfsp_weighting="variance",
        league_exploiter_pfsp_weighting="variance",
        save_gpu_memory=False,
        exp_name="test_exp",
        main_pfsp_weighting="focused_strong",
        Unit_reward_per_exploiter=False,
        unit_exploiters=False,
        use_no_draw_winrates=True,
    )
    base.update(overrides)
    return _make_args(**base)


def _pick_param(agent):
    for name, param in agent.named_parameters():
        return name, param
    raise AssertionError("Agent has no parameters to test.")


def _assert_exploiter_initial_weights_are_independent(exploiter_cls):
    device = torch.device("cpu")
    main_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    payoff = league.Payoff()
    exploiter = exploiter_cls(main_agent, payoff, args=_make_dummy_args())

    name, param = _pick_param(main_agent)
    before_initial = exploiter._initial_weights[name].clone()
    before_main = param.detach().clone()

    with torch.no_grad():
        param.add_(1.0)

    assert not torch.allclose(param, before_main), "Main agent parameter did not change."
    assert torch.allclose(
        exploiter._initial_weights[name], before_initial
    ), "Exploiter initial weights should be a snapshot, not a reference."


def _assert_exploiter_reset_clears_payoff_entries(exploiter_cls):
    args = _make_ready_to_checkpoint_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    base_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    other_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(base_agent, payoff, args=args)
    other_player = league.MainPlayer(other_agent, payoff, args=args, name="OtherMain")
    exploiter = exploiter_cls(base_agent, payoff, args=args)

    payoff.add_player(main_player)
    payoff.add_player(other_player)
    payoff.add_player(exploiter)

    payoff.update(exploiter, main_player, 1)
    payoff.update(main_player, other_player, 0)

    stats = (
        ("no_decay_games", payoff._no_decay_games),
        ("no_decay_wins", payoff._no_decay_wins),
        ("no_decay_draws", payoff._no_decay_draws),
        ("no_decay_losses", payoff._no_decay_losses),
        ("games", payoff._games),
        ("wins", payoff._wins),
        ("draws", payoff._draws),
        ("losses", payoff._losses),
    )

    before_with_exploiter = {}
    before_without_exploiter = {}
    for name, stat in stats:
        before_with_exploiter[name] = {key for key in stat.keys() if exploiter in key}
        before_without_exploiter[name] = {key for key in stat.keys() if exploiter not in key}

    assert (exploiter, main_player) in payoff._games
    assert (main_player, exploiter) in payoff._games
    assert (main_player, other_player) in payoff._games
    assert (other_player, main_player) in payoff._games
    assert (exploiter, main_player) in payoff._no_decay_games
    assert (main_player, exploiter) in payoff._no_decay_games

    exploiter.reset()

    for name, stat in stats:
        for key in before_with_exploiter[name]:
            assert key not in stat
        for key in before_without_exploiter[name]:
            assert key in stat


def test_main_exploiter_initial_weights_are_independent():
    _assert_exploiter_initial_weights_are_independent(league.MainExploiter)


def test_league_exploiter_initial_weights_are_independent():
    _assert_exploiter_initial_weights_are_independent(league.LeagueExploiter)


def test_main_exploiter_reset_clears_payoff_entries():
    _assert_exploiter_reset_clears_payoff_entries(league.MainExploiter)


def test_league_exploiter_reset_clears_payoff_entries():
    _assert_exploiter_reset_clears_payoff_entries(league.LeagueExploiter)


def test_old_gae_equals_new_gae():
    device = torch.device("cpu")
    torch.manual_seed(0)

    class DummyAgent:
        def __init__(self, next_value):
            self._next_value = next_value

        def get_value(self, *_args, **_kwargs):
            return self._next_value

    for _ in range(50):
        num_steps = 512
        num_envs = 20
        args = _make_args(num_steps=num_steps, gamma=0.99, gae_lambda=0.95)

        b_rewards_winloss = torch.randn(num_steps, num_envs, device=device)
        b_rewards_attack = torch.randn(num_steps, num_envs, device=device)
        b_rewards_score = torch.zeros(num_steps, num_envs, device=device)
        b_values = torch.randn(num_steps, num_envs, device=device)
        b_dones = torch.randint(0, 2, (num_steps, num_envs), device=device, dtype=torch.float)
        b_next_done = torch.randint(0, 2, (num_envs,), device=device, dtype=torch.float)

        b_next_value = torch.randn(num_envs, device=device)
        next_obs = torch.zeros((num_envs, 16, 16, 73), device=device)
        next_scalar_features = torch.zeros((num_envs, 11), device=device)
        next_z_features = torch.zeros((num_envs, 8), device=device)

        dummy = _make_dummy_args()
        dummy.agent = DummyAgent(b_next_value)
        dummy.device = device
        dummy.args = args
        dummy.indices = torch.arange(num_envs, device=device)

        dbg_returns, dbg_advantages = old_gae(
            dummy,
            next_obs,
            next_scalar_features,
            next_z_features,
            b_rewards_winloss,
            b_rewards_attack,
            b_next_done,
            b_dones,
            b_values,
        )

        b_advantages, b_returns = ppo_update.gae(
            args,
            device,
            b_next_value,
            b_values,
            b_rewards_attack,
            b_rewards_winloss,
            b_rewards_score,
            b_dones,
            b_next_done,
        )

        assert torch.equal(dbg_advantages, b_advantages), "Advantages do not match between old and new GAE implementations."
        assert torch.equal(dbg_returns, b_returns), "Returns do not match between old and new GAE implementations."


def test_old_ppo_update_returns_match_new_update():
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    class DummyEnv:
        def __init__(self, num_envs, obs_shape):
            self.num_envs = num_envs
            self.single_observation_space = types.SimpleNamespace(shape=obs_shape)

    class DummyAgent(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.value_head = torch.nn.Linear(input_dim, 1, bias=False)
            self.logit_head = torch.nn.Linear(input_dim, 1, bias=False)

        def get_value(self, x, sc, z, unit_bonus_distr=None):
            flat = torch.cat([x.reshape(x.shape[0], -1), sc, z], dim=1)
            return self.value_head(flat)

        def get_action(self, x, sc, z, action, invalid_action_masks, envs, **_kwargs):
            flat = torch.cat([x.reshape(x.shape[0], -1), sc, z], dim=1)
            logprob = self.logit_head(flat).squeeze(1)
            entropy = torch.zeros_like(logprob)
            return action, logprob, entropy, invalid_action_masks

    num_steps = 512
    num_envs = 20
    obs_shape = (16, 16, 73)
    sc_dim = 11
    z_dim = 8
    action_space_shape = (1, 2)
    invalid_action_shape = (1, 3)

    args = _make_args(
        num_steps=num_steps,
        gamma=1.0,
        gae_lambda=0.95,
        batch_size=num_steps * num_envs,
        update_epochs=4,
        minibatch_size=num_steps * num_envs,
        norm_adv=True,
        clip_coef=0.1,
        clip_vloss=True,
        ent_coef=0.005,
        vf_coef=0.5,
        kl_coeff=0.4,
        max_grad_norm=0.5,
        target_kl=0.01,
        kle_stop=True,
        kle_rollback=True,
        value_warmup_updates=-1,
    )

    dummy_envs = DummyEnv(num_envs=num_envs, obs_shape=obs_shape)

    obs_dim = int(np.prod(obs_shape))
    for i in range(100):
        obs = torch.randn(num_steps, num_envs, *obs_shape, device=device)
        scalar_features = torch.randn(num_steps, num_envs, sc_dim, device=device)
        z_features = torch.randn(num_steps, num_envs, z_dim, device=device)
        actions = torch.zeros(num_steps, num_envs, *action_space_shape, device=device)
        logprobs = torch.zeros(num_steps, num_envs, device=device)
        invalid_action_masks = torch.ones(num_steps, num_envs, *invalid_action_shape, device=device)
        rewards_winloss = torch.randn(num_steps, num_envs, device=device)
        rewards_attack = torch.randn(num_steps, num_envs, device=device)
        dones = torch.randint(0, 2, (num_steps, num_envs), device=device, dtype=torch.float)
        next_done = torch.randint(0, 2, (num_envs,), device=device, dtype=torch.float)
        values = torch.randn(num_steps, num_envs, device=device)
        next_obs = torch.randn(num_envs, *obs_shape, device=device)

        base_agent = DummyAgent(input_dim=obs_dim + sc_dim + z_dim).to(device)
        old_agent = DummyAgent(input_dim=obs_dim + sc_dim + z_dim).to(device)
        new_agent = DummyAgent(input_dim=obs_dim + sc_dim + z_dim).to(device)
        old_agent.load_state_dict(base_agent.state_dict())
        new_agent.load_state_dict(base_agent.state_dict())
        old_optimizer = torch.optim.Adam(old_agent.parameters(), lr=1e-3)
        new_optimizer = torch.optim.Adam(new_agent.parameters(), lr=1e-3)

        dummy = _make_dummy_args()
        dummy.agent = old_agent
        dummy.device = device
        dummy.args = args
        dummy.envs = dummy_envs
        dummy.supervised_agent = DummyAgent(input_dim=obs_dim + sc_dim + z_dim).to(device)
        dummy.supervised_agent.load_state_dict(base_agent.state_dict())

        with torch.no_grad():
            next_value = new_agent.get_value(next_obs, scalar_features[-1], z_features[-1]).reshape(-1)
        new_advantages, new_returns = ppo_update.gae(
            args,
            device,
            next_value,
            values,
            rewards_attack,
            rewards_winloss,
            torch.zeros_like(rewards_winloss),
            dones,
            next_done,
        )

        np.random.seed(i)
        old_approx_kl, old_pg_loss, old_v_loss, old_entropy_loss, old_kl_div, old_loss, _ = old_ppo_update(
            dummy,
            next_obs,
            scalar_features,
            z_features,
            rewards_winloss,
            rewards_attack,
            next_done,
            dones,
            values,
            obs,
            actions,
            logprobs,
            invalid_action_masks,
            old_optimizer,
            action_space_shape,
            invalid_action_shape,
            num_steps - 1,
        )

        agent_batch = {
            "agent": new_agent,
            "optimizer": new_optimizer,
            "obs": obs.reshape((-1,) + obs_shape),
            "sc": scalar_features.reshape(-1, sc_dim),
            "z": z_features.reshape(-1, z_dim),
            "actions": actions.reshape((-1,) + action_space_shape),
            "logprobs": logprobs.reshape(-1),
            "advantages": new_advantages.reshape(-1),
            "returns": new_returns.reshape(-1),
            "values": values.reshape(-1),
            "masks": invalid_action_masks.reshape((-1,) + invalid_action_shape),
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "clip_coef": args.clip_coef,
            "kl_coeff": args.kl_coeff,
            "max_grad_norm": args.max_grad_norm,
            "update_epochs": args.update_epochs,
            "value_warmup_updates": -1,
            "kle_stop": False,
            "kle_rollback": False,
            "norm_adv": args.norm_adv,
            "clip_vloss": args.clip_vloss,
        }

        np.random.seed(i)
        _, new_pg_loss, new_entropy_loss, new_kl_loss, new_approx_kl, new_v_loss, new_loss, _ = ppo_update.update(
            args,
            dummy_envs,
            agent_batch,
            device,
            dummy.supervised_agent,
            update=0,
            new_batch_size=args.batch_size,
            minibatch_size=args.batch_size,
        )

        assert torch.equal(old_pg_loss, new_pg_loss), "Policy loss does not match between old and new update."
        assert torch.equal(old_entropy_loss, new_entropy_loss), "Entropy loss does not match between old and new update."
        assert torch.equal(old_v_loss, new_v_loss), "Value loss does not match between old and new update."
        assert torch.equal(old_approx_kl, new_approx_kl), "Approx KL does not match between old and new update."
        assert torch.equal(old_kl_div * args.kl_coeff, new_kl_loss), "KL loss does not match between old and new update."
        assert torch.equal(old_loss, new_loss), "Total loss does not match between old and new update."

def test_main_exploiter_checkpoint_resets_training_state(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_ready_to_checkpoint_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    base_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    exploiter = league.MainExploiter(base_agent, payoff, args=args)
    payoff.add_player(exploiter)

    initial_weights = {k: v.detach().clone() for k, v in exploiter._initial_weights.items()}
    name, param = _pick_param(exploiter.agent)
    with torch.no_grad():
        param.add_(1.0)
    trained_weights = {k: v.detach().clone() for k, v in exploiter.agent.state_dict().items()}
    assert not torch.allclose(trained_weights[name], initial_weights[name])

    exploiter.optimizer = torch.optim.Adam(exploiter.agent.parameters(), lr=0.001)
    exploiter.agent.steps = 123

    checkpoint = exploiter.checkpoint()

    assert isinstance(checkpoint, league.Historical)
    assert checkpoint.parent is exploiter
    for key, tensor in trained_weights.items():
        assert torch.allclose(checkpoint.agent.state_dict()[key], tensor)
    for key, tensor in initial_weights.items():
        assert torch.allclose(exploiter.agent.state_dict()[key], tensor)
    assert exploiter.optimizer is None
    assert exploiter._checkpoint_step == exploiter.agent.get_steps()


def test_league_exploiter_checkpoint_resets_training_state(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(league.np.random, "random", lambda: 0.0)
    args = _make_ready_to_checkpoint_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    base_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    exploiter = league.LeagueExploiter(base_agent, payoff, args=args)
    payoff.add_player(exploiter)

    initial_weights = {k: v.detach().clone() for k, v in exploiter._initial_weights.items()}
    name, param = _pick_param(exploiter.agent)
    with torch.no_grad():
        param.add_(1.0)
    trained_weights = {k: v.detach().clone() for k, v in exploiter.agent.state_dict().items()}
    assert not torch.allclose(trained_weights[name], initial_weights[name])

    exploiter.optimizer = torch.optim.Adam(exploiter.agent.parameters(), lr=0.001)
    exploiter.agent.steps = 321

    checkpoint = exploiter.checkpoint()

    assert isinstance(checkpoint, league.Historical)
    assert checkpoint.parent is exploiter
    for key, tensor in trained_weights.items():
        assert torch.allclose(checkpoint.agent.state_dict()[key], tensor)
    for key, tensor in initial_weights.items():
        assert torch.allclose(exploiter.agent.state_dict()[key], tensor)
    assert exploiter.optimizer is None
    assert exploiter._checkpoint_step == exploiter.agent.get_steps()


def test_state_dict_clone_matches_initial_state():
    device = torch.device("cpu")
    initial_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    state = initial_agent.state_dict()
    cloned = {k: v.detach().clone() for k, v in state.items()}
    for key, tensor in state.items():
        assert torch.allclose(
            tensor, cloned[key]
        ), f"Cloned state for {key} does not match initial state_dict."


def test_remove_monotonic_suffix_truncates_on_increase():
    win_rates = np.array([0.9, 0.6, 0.65, 0.6])
    players = ["p0", "p1", "p2", "p3"]
    trimmed_rates, trimmed_players = league.remove_monotonic_suffix(win_rates, players)
    assert trimmed_rates.tolist() == [0.9, 0.6, 0.65]
    assert trimmed_players == ["p0", "p1", "p2"]


def test_player_ready_to_checkpoint_is_false():
    player = league.Player()
    assert not player.ready_to_checkpoint()


def test_historical_ready_to_checkpoint_is_false(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_ready_to_checkpoint_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    parent_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    parent = league.MainPlayer(parent_agent, payoff, args=args)
    historical = league.Historical(parent, payoff, args=args, historical_count=0)
    assert not historical.ready_to_checkpoint()


def test_main_player_ready_to_checkpoint_creates_historical_when_ready(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_ready_to_checkpoint_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(agent, payoff, args=args)
    payoff.add_player(main_player)

    historical = main_player.checkpoint()
    payoff.add_player(historical)
    for _ in range(20):
        payoff.update(main_player, historical, 1)

    agent.steps = args.selfplay_ready_save_interval * args.num_main_envs - 1
    assert not main_player.ready_to_checkpoint()
    agent.steps = args.selfplay_ready_save_interval * args.num_main_envs
    assert main_player.ready_to_checkpoint()
    checkpoint = main_player.checkpoint()
    assert isinstance(checkpoint, league.Historical)


def test_main_player_ready_to_checkpoint_uses_raw_save_interval(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_ready_to_checkpoint_args(
        main_selfplay_save_interval=100,
        num_selfplay_envs=4,
        num_main_envs=3,
    )
    payoff = league.Payoff()
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(agent, payoff, args=args)
    payoff.add_player(main_player)

    historical = main_player.checkpoint()
    payoff.add_player(historical)
    payoff.update(main_player, historical, 0)

    agent.steps = 120
    assert main_player.ready_to_checkpoint()


def test_main_exploiter_ready_to_checkpoint_creates_historical_when_ready(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_ready_to_checkpoint_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    base_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(base_agent, payoff, args=args)
    exploiter = league.MainExploiter(base_agent, payoff, args=args)
    payoff.add_player(main_player)
    payoff.add_player(exploiter)

    main_hist = main_player.checkpoint()
    payoff.add_player(main_hist)
    for _ in range(20):
        payoff.update(exploiter, main_hist, 1)
    exploiter.agent.steps = args.selfplay_ready_save_interval - 1
    assert not exploiter.ready_to_checkpoint()
    exploiter.agent.steps = args.selfplay_ready_save_interval
    assert exploiter.ready_to_checkpoint()
    checkpoint = exploiter.checkpoint()
    assert isinstance(checkpoint, league.Historical)


def test_main_exploiter_ready_to_checkpoint_uses_raw_save_interval(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_ready_to_checkpoint_args(
        main_exploiter_selfplay_save_interval=100,
        num_selfplay_envs=4,
        num_envs_per_main_exploiters=3,
    )
    payoff = league.Payoff()
    device = torch.device("cpu")
    base_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(base_agent, payoff, args=args)
    exploiter = league.MainExploiter(base_agent, payoff, args=args)
    payoff.add_player(main_player)
    payoff.add_player(exploiter)

    payoff.update(exploiter, main_player, 0)
    exploiter.agent.steps = 120
    assert exploiter.ready_to_checkpoint()


def test_league_exploiter_ready_to_checkpoint_creates_historical_when_ready(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_ready_to_checkpoint_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    base_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(base_agent, payoff, args=args)
    payoff.add_player(main_player)
    historical = main_player.checkpoint()
    payoff.add_player(historical)

    exploiter = league.LeagueExploiter(base_agent, payoff, args=args)
    payoff.add_player(exploiter)
    for _ in range(20):
        payoff.update(exploiter, historical, 1)
    exploiter.agent.steps = args.selfplay_ready_save_interval - 1
    assert not exploiter.ready_to_checkpoint()
    exploiter.agent.steps = args.selfplay_ready_save_interval
    assert exploiter.ready_to_checkpoint()
    checkpoint = exploiter.checkpoint()
    assert isinstance(checkpoint, league.Historical)


def test_league_exploiter_ready_to_checkpoint_uses_raw_save_interval(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_ready_to_checkpoint_args(
        league_exploiter_selfplay_save_interval=100,
        num_selfplay_envs=4,
        num_envs_per_league_exploiters=3,
    )
    payoff = league.Payoff()
    device = torch.device("cpu")
    base_agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(base_agent, payoff, args=args)
    payoff.add_player(main_player)
    historical = main_player.checkpoint()
    payoff.add_player(historical)

    exploiter = league.LeagueExploiter(base_agent, payoff, args=args)
    payoff.add_player(exploiter)
    payoff.update(exploiter, historical, 0)
    exploiter.agent.steps = 120
    assert exploiter.ready_to_checkpoint()


def test_main_player_get_match_avoids_active_exploiters(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_match_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(agent, payoff, args=args)
    payoff.add_player(main_player)

    main_exploiter = league.MainExploiter(agent, payoff, args=args, main_exp_idx=0)
    league_exploiter = league.LeagueExploiter(agent, payoff, args=args, league_exp_idx=0)
    payoff.add_player(main_exploiter)
    payoff.add_player(league_exploiter)

    historical = league.Historical(main_player, payoff, args=args, historical_count=0)
    payoff.add_player(historical)

    coin_tosses = iter([0.1, 0.6, 0.4, 0.75, 0.9])
    monkeypatch.setattr(league.np.random, "random", lambda: next(coin_tosses))

    opponent_one, _ = main_player.get_match()
    opponent_two, _ = main_player.get_match()
    opponent_three, _ = main_player.get_match()
    opponent_four, _ = main_player.get_match()
    opponent_five, _ = main_player.get_match()

    for opponent in (opponent_one, opponent_two, opponent_three, opponent_four, opponent_five):
        assert not isinstance(opponent, league.MainExploiter)
        assert not isinstance(opponent, league.LeagueExploiter)


def test_main_player_pfsp_prefers_winrate_point_two(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_match_args(pfsp=True, pfsp_min_prob_factor=0.0)
    payoff = league.Payoff()
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(agent, payoff, args=args)
    payoff.add_player(main_player)

    historicals = []
    for idx in range(4):
        historical = league.Historical(main_player, payoff, args=args, historical_count=idx)
        historicals.append(historical)
        payoff.add_player(historical)

    monkeypatch.setattr(
        payoff,
        "array_win_rate_no_draw",
        lambda home, away: np.array([0.2, 0.4, 0.8, 0.05]),
    )

    captured = {}

    def fake_choice(options, p=None):
        captured["p"] = p
        return options[0]

    monkeypatch.setattr(league.np.random, "choice", fake_choice)
    monkeypatch.setattr(league.np.random, "random", lambda: 0.1)

    opponent, _ = main_player.get_match()
    assert opponent is historicals[0]

    probs = captured["p"]
    assert probs[0] > probs[1]
    assert probs[0] > probs[2]
    assert probs[0] > probs[3]


def test_main_player_pfsp_sampling_prefers_winrate_point_two(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_match_args(pfsp=True, pfsp_min_prob_factor=0.0)
    payoff = league.Payoff()
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(agent, payoff, args=args)
    payoff.add_player(main_player)

    historicals = []
    for idx in range(4):
        historical = league.Historical(main_player, payoff, args=args, historical_count=idx)
        historicals.append(historical)
        payoff.add_player(historical)

    win_rates = np.array([0.2, 0.4, 0.8, 0.05])
    monkeypatch.setattr(payoff, "array_win_rate_no_draw", lambda home, away: win_rates)
    monkeypatch.setattr(league.np.random, "random", lambda: 0.1)

    rng = np.random.RandomState(0)

    def seeded_choice(options, p=None):
        return options[rng.choice(len(options), p=p)]

    monkeypatch.setattr(league.np.random, "choice", seeded_choice)

    counts = np.zeros(len(historicals), dtype=int)
    index_by_hist = {hist: idx for idx, hist in enumerate(historicals)}
    samples = 5000
    for _ in range(samples):
        opponent, _ = main_player.get_match()
        counts[index_by_hist[opponent]] += 1

    observed = counts / samples
    expected = league.pfsp(win_rates, weighting="focused_strong", enabled=True, min_prob_factor=0.0)
    tolerance = 5 * np.sqrt(expected * (1 - expected) / samples)

    assert observed[0] > observed[1]
    assert observed[0] > observed[2]
    assert observed[0] > observed[3]
    for idx, (obs, exp, tol) in enumerate(zip(observed, expected, tolerance)):
        assert abs(obs - exp) <= tol, f"index {idx} observed {obs} expected {exp} tolerance {tol}"


def test_main_player_selfplay_rate_about_ten_percent(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_match_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(agent, payoff, args=args)
    payoff.add_player(main_player)

    historical = league.Historical(main_player, payoff, args=args, historical_count=0)
    payoff.add_player(historical)

    monkeypatch.setattr(main_player, "_pfsp_branch", lambda: (historical, True))
    monkeypatch.setattr(main_player, "_verification_branch", lambda opponent: (historical, True))

    coin_tosses = iter([(i + 0.5) / 100 for i in range(100)])
    monkeypatch.setattr(league.np.random, "random", lambda: next(coin_tosses))

    matches = [main_player.get_match()[0] for _ in range(100)]
    selfplay_count = sum(isinstance(opp, league.MainPlayer) for opp in matches)

    assert selfplay_count == 10


def test_main_exploiter_get_match_returns_main_or_main_historical(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_match_args(
        main_exploiter_no_draw_winrate_threshold=0.7,
        main_exploiter_vs_main_winrate_threshold=0.9,
        main_exploiter_winrate_threshold=0.7,
        main_exploiter_ready_use_historicals=False,
    )
    payoff = league.Payoff()
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(agent, payoff, args=args)
    main_exploiter = league.MainExploiter(agent, payoff, args=args, main_exp_idx=0)
    league_exploiter = league.LeagueExploiter(agent, payoff, args=args, league_exp_idx=0)

    payoff.add_player(main_player)
    payoff.add_player(main_exploiter)
    payoff.add_player(league_exploiter)

    main_hist = league.Historical(main_player, payoff, args=args, historical_count=0)
    other_hist = league.Historical(league_exploiter, payoff, args=args, historical_count=0)
    payoff.add_player(main_hist)
    payoff.add_player(other_hist)

    payoff._games[main_exploiter, main_player] = 10

    def win_rates(home, away):
        if away is main_player:
            return 0.0
        if isinstance(away, list):
            return np.array([0.95 for _ in away])
        raise AssertionError("Unexpected opponent for win rate lookup.")

    monkeypatch.setattr(payoff, "array_win_rate_no_draw", win_rates)
    np.random.seed(0)

    matches = [main_exploiter.get_match()[0] for _ in range(100)]
    assert any(opp is main_player for opp in matches)
    for opp in matches:
        assert opp is main_player or (isinstance(opp, league.Historical) and opp.parent is main_player)


def test_main_exploiter_vs_main_winrate_threshold(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_match_args(
        main_exploiter_no_draw_winrate_threshold=0.9,
        main_exploiter_vs_main_winrate_threshold=0.5,
        main_exploiter_ready_use_historicals=False,
    )
    payoff = league.Payoff()
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(agent, payoff, args=args)
    main_exploiter = league.MainExploiter(agent, payoff, args=args, main_exp_idx=0)
    payoff.add_player(main_player)
    payoff.add_player(main_exploiter)
    main_hist = league.Historical(main_player, payoff, args=args, historical_count=0)
    payoff.add_player(main_hist)
    payoff._games[main_exploiter, main_player] = 10

    def win_rates(home, away):
        if away is main_player:
            return 0.0
        if isinstance(away, list):
            return np.array([0.6 for _ in away])
        raise AssertionError("Unexpected opponent for win rate lookup.")

    monkeypatch.setattr(payoff, "array_win_rate_no_draw", win_rates)
    monkeypatch.setattr(league.np.random, "random", lambda: 0.0)

    opponent, _ = main_exploiter.get_match()
    assert opponent is main_player

    main_exploiter.args.main_exploiter_vs_main_winrate_threshold = 0.6
    opponent, _ = main_exploiter.get_match()
    assert isinstance(opponent, league.Historical)
    assert opponent.parent is main_player


def test_league_exploiter_get_match_uses_historicals(monkeypatch):
    monkeypatch.setattr(league, "save_league_model", lambda *args, **kwargs: None)
    args = _make_match_args()
    payoff = league.Payoff()
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2], device=device)
    main_player = league.MainPlayer(agent, payoff, args=args)
    league_exploiter = league.LeagueExploiter(agent, payoff, args=args, league_exp_idx=0)
    payoff.add_player(main_player)
    payoff.add_player(league_exploiter)

    hist_one = league.Historical(main_player, payoff, args=args, historical_count=0)
    hist_two = league.Historical(league_exploiter, payoff, args=args, historical_count=1)
    payoff.add_player(hist_one)
    payoff.add_player(hist_two)

    captured = {}

    def fake_choice(options, p=None):
        captured["options"] = options
        captured["p"] = p
        return options[-1]

    monkeypatch.setattr(league.np.random, "choice", fake_choice)

    opponent, _ = league_exploiter.get_match()
    assert opponent is hist_two
    assert all(isinstance(player, league.Historical) for player in captured["options"])


def test_remove_monotonic_suffix_handles_none():
    players = ["p0"]
    trimmed_rates, trimmed_players = league.remove_monotonic_suffix(None, players)
    assert trimmed_rates is None
    assert trimmed_players == players


def test_pfsp_returns_uniform_when_all_zero_weight():
    win_rates = np.ones(4)
    probs = league.pfsp(win_rates, weighting="linear", enabled=True)
    assert np.allclose(probs, np.ones(4) / 4)


def test_payoff_update_tracks_symmetric_results():
    payoff = league.Payoff()
    p1, p2 = object(), object()
    payoff.update(p1, p2, 1)
    assert payoff._games[p1, p2] == 1
    assert payoff._wins[p1, p2] == 1
    assert payoff._games[p2, p1] == 1
    assert payoff._losses[p2, p1] == 1
    assert payoff._win_rate(p1, p2) == 1.0
    assert payoff._win_rate(p2, p1) == 0.0


def test_payoff_update_tracks_draw_results():
    payoff = league.Payoff()
    p1, p2 = object(), object()
    payoff.update(p1, p2, 0)
    assert payoff._games[p1, p2] == 1
    assert payoff._draws[p1, p2] == 1
    assert payoff._games[p2, p1] == 1
    assert payoff._draws[p2, p1] == 1
    assert payoff._win_rate(p1, p2) == 0.5
    assert payoff._win_rate(p2, p1) == 0.5


def test_adjust_action_selfplay_transforms_odd_envs():
    selfplay_league = pytest.importorskip("selfplay_league")
    args = _make_args(num_selfplay_envs=4)
    valid_actions_counts = np.array([2, 2, 2, 2], dtype=np.int64)
    valid_actions = np.zeros((8, 8), dtype=np.int64)

    valid_actions[0, 0] = 10
    valid_actions[1, 0] = 20
    valid_actions[0, 2:6] = [0, 1, 2, 3]
    valid_actions[1, 2:6] = [0, 1, 2, 3]
    valid_actions[0, 7] = 5
    valid_actions[2, 7] = 25
    valid_actions[1, 0] = 100

    valid_actions[3, 0] = 200
    valid_actions[2, 2:6] = [0, 1, 2, 3]
    valid_actions[3, 2:6] = [3, 2, 1, 0]
    valid_actions[3, 7] = 1

    valid_actions[4, 0] = 10
    valid_actions[5, 0] = 20
    valid_actions[4, 2:6] = [0, 1, 2, 3]
    valid_actions[5, 2:6] = [0, 1, 2, 3]
    valid_actions[4, 7] = 7
    valid_actions[5, 7] = 25
    valid_actions[5, 0] = 100

    valid_actions[7, 0] = 200
    valid_actions[6, 2:6] = [0, 1, 2, 3]
    valid_actions[7, 2:6] = [3, 2, 1, 0]
    valid_actions[6, 7] = 7
    valid_actions[7, 7] = 1
    selfplay_league.adjust_action_selfplay(args, valid_actions, valid_actions_counts)

    assert valid_actions[0, 0] == 10
    assert valid_actions[1, 0] == 100
    assert valid_actions[3, 0] == 55
    assert valid_actions[7, 0] == 55
    assert np.array_equal(valid_actions[0, 2:6], [0, 1, 2, 3])
    assert np.array_equal(valid_actions[1, 2:6], [0, 1, 2, 3])
    assert np.array_equal(valid_actions[2, 2:6], [2, 3, 0, 1])
    assert np.array_equal(valid_actions[3, 2:6], [1, 0, 3, 2])
    assert np.array_equal(valid_actions[5, 2:6], [0, 1, 2, 3])
    assert np.array_equal(valid_actions[7, 2:6], [1, 0, 3, 2])
    assert valid_actions[0, 7] == 5
    assert valid_actions[2, 7] == 23
    assert valid_actions[4, 7] == 7
    assert valid_actions[3, 7] == 47


def test_adjust_obs_selfplay():
    selfplay_league = pytest.importorskip("selfplay_league")
    args = _make_args(num_selfplay_envs=4)
    next_obs = torch.zeros((4, 3, 3, 73))
    
    expected = next_obs.clone()

    next_obs[:, 2, 1] = torch.tensor([0.1000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
        0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0627, 1.0000, 1.0000, 1.0000, 0.1250, 1.0000, 1.0000, 1.0000,
        0.0000, 0.0000, 0.0000, 0.1250, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000,
        0.0000])
    next_obs[:, 0, 2] = torch.tensor([0.4000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0588, 1.0000, 1.0000, 1.0000, 0.2812, 1.0000, 1.0000, 1.0000,
        1.0000, 0.0000, 0.0000, 0.2812, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000,
        0.0000])

    selfplay_league.adjust_obs_selfplay(args, next_obs)

    

    expected[1, 0, 1] = torch.tensor([0.1000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
        0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0627, 1.0000, 1.0000, 1.0000, 0.1250, 1.0000, 1.0000, 1.0000,
        0.0000, 0.0000, 0.0000, 0.1250, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000,
        0.0000])
    expected[1, 2, 0] = torch.tensor([0.4000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0588, 1.0000, 1.0000, 1.0000, 0.2812, 1.0000, 1.0000, 1.0000,
        1.0000, 0.0000, 0.0000, 0.2812, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000,
        0.0000])
    
    expected[3, 0, 1] = torch.tensor([0.1000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
        0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0627, 1.0000, 1.0000, 1.0000, 0.1250, 1.0000, 1.0000, 1.0000,
        0.0000, 0.0000, 0.0000, 0.1250, 1.0000, 1.0000, 1.0000, 0.0000, 0.0000,
        0.0000])
    expected[3, 2, 0] = torch.tensor([0.4000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        1.0000, 0.0588, 1.0000, 1.0000, 1.0000, 0.2812, 1.0000, 1.0000, 1.0000,
        1.0000, 0.0000, 0.0000, 0.2812, 1.0000, 1.0000, 1.0000, 1.0000, 0.0000,
        0.0000])

    assert torch.equal(next_obs[1, 0, 1], expected[1, 0, 1])
    assert torch.equal(next_obs[1, 2, 0], expected[1, 2, 0])
    assert torch.equal(next_obs[3, 0, 1], expected[3, 0, 1])
    assert torch.equal(next_obs[3, 2, 0], expected[3, 2, 0])

def test_adjust_obs_selfplay_adjusted_2_times():
    selfplay_league = pytest.importorskip("selfplay_league")
    args = _make_args(num_selfplay_envs=8)
    next_obs = torch.rand((8, 16, 16, 73))
    original = next_obs.clone()
    selfplay_league.adjust_obs_selfplay(args, next_obs)
    selfplay_league.adjust_obs_selfplay(args, next_obs)
    assert torch.equal(next_obs, original)

def test_gae_accumulates_rewards():
    args = _make_args(num_steps=3, gamma=1.0, gae_lambda=1.0)
    device = torch.device("cpu")
    rewards = torch.tensor([1.0, 2.0, 3.0], device=device)
    zeros = torch.zeros_like(rewards)
    b_advantages, b_returns = ppo_update.gae(
        args,
        device,
        b_next_value=torch.tensor(0.0, device=device),
        b_values=zeros,
        b_rewards_attack=zeros,
        b_rewards_winloss=rewards,
        b_rewards_score=zeros,
        b_dones=zeros,
        b_next_done=torch.tensor(0.0, device=device)
    )
    assert torch.allclose(b_advantages, torch.tensor([6.0, 5.0, 3.0], device=device))
    assert torch.allclose(b_returns, b_advantages)


def test_update_skip_policy_update_returns_none():
    args = _make_args(
        ent_coef=0.01,
        vf_coef=0.5,
        clip_coef=0.1,
        target_kl=0.03,
        kl_coeff=0.0,
        max_grad_norm=0.5,
        update_epochs=1,
        value_warmup_updates=0,
        kle_stop=False,
        kle_rollback=False,
        norm_adv=False,
        clip_vloss=False,
    )
    agent = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(agent.parameters(), lr=0.1)
    params_before = [p.detach().clone() for p in agent.parameters()]
    opt_state_before = optimizer.state_dict()
    dummy_tensor = torch.zeros(1)
    agent_batch = {
        "agent": agent,
        "optimizer": optimizer,
        "values": dummy_tensor,
        "advantages": dummy_tensor,
        "returns": dummy_tensor,
        "sc": dummy_tensor,
        "z": dummy_tensor,
        "obs": dummy_tensor,
        "actions": dummy_tensor,
        "logprobs": dummy_tensor,
        "masks": dummy_tensor,
        "skip_policy_update": True,
    }
    result = ppo_update.update(
        args,
        envs=None,
        agent_batch=agent_batch,
        device=torch.device("cpu"),
        supervised_agent=object(),
        update=1,
        new_batch_size=1,
        minibatch_size=1
    )
    assert result == (None, None, None, None, None, None, None, None)
    for before, after in zip(params_before, agent.parameters()):
        assert torch.equal(before, after)
    assert optimizer.state_dict() == opt_state_before


def test_categorical_masked_respects_masks():
    logits = torch.tensor([[0.0, 0.0]])
    masks = torch.tensor([[1, 0]], dtype=torch.bool)
    dist = agent_model.CategoricalMasked(logits=logits, masks=masks, device=torch.device("cpu"))
    probs = dist.probs
    assert probs[0, 0] > 0.999
    assert probs[0, 1] < 1e-6
    assert torch.allclose(dist.entropy(), torch.zeros_like(dist.entropy()))


def test_adjust_selfplay_masks_rolls_and_flips():
    device = torch.device("cpu")
    agent = agent_model.Agent(action_plane_nvec=[2, 2, 2, 2, 2, 2, 2], device=device, mapsize=2)
    total_envs = 2
    num_selfplay_envs = 2
    split_masks = [
        torch.zeros((total_envs * agent.mapsize, 4), dtype=torch.int64)
        for _ in range(7)
    ]
    for mask in split_masks:
        for row in range(mask.shape[0]):
            mask[row] = torch.tensor([0, 1, 2, 3])
    original = [mask.clone() for mask in split_masks]

    agent._adjust_selfplay_masks(split_masks, num_selfplay_envs=num_selfplay_envs, total_envs=total_envs)

    start = agent.mapsize
    end = start + agent.mapsize
    expected_roll = torch.tensor([[2, 3, 0, 1], [2, 3, 0, 1]])
    expected_flip = torch.tensor([[3, 2, 1, 0], [3, 2, 1, 0]])
    for idx in range(1, 5):
        assert torch.equal(split_masks[idx][start:end], expected_roll)
    assert torch.equal(split_masks[6][start:end], expected_flip)
    for idx in range(7):
        assert torch.equal(split_masks[idx][:start], original[idx][:start])


def _load_league_sp_xp_config():
    omegaconf = pytest.importorskip("omegaconf")
    base_cfg = omegaconf.OmegaConf.load("conf/default_config.yaml")
    override_cfg = omegaconf.OmegaConf.load("conf/league_sp_xp_conf_old_ppo_args.yaml")
    merged = omegaconf.OmegaConf.merge(base_cfg, override_cfg)
    return merged.ExperimentConfig


def test_league_sp_xp_config_defaults():
    cfg = _load_league_sp_xp_config()
    assert cfg.league_training is True
    assert cfg.selfplay is False
    assert cfg.sp is True
    expected_selfplay_envs = (
        cfg.num_main_envs
        + cfg.num_main_exploiters * cfg.num_envs_per_main_exploiters
        + cfg.num_league_exploiters * cfg.num_envs_per_league_exploiters
    ) * 2
    assert expected_selfplay_envs == 42


def test_resolve_checkpoint_path_uses_config_model_path():
    cfg = _load_league_sp_xp_config()
    main = pytest.importorskip("main")
    resolved = main._resolve_checkpoint_path(cfg.model_path, exp_name=cfg.exp_name, resume=True)
    assert resolved == cfg.model_path
