import types

import numpy as np
import pytest
import torch

import agent_model
import league
import ppo_update


def _make_dummy_args():
    return types.SimpleNamespace()


def _make_args(**kwargs):
    return types.SimpleNamespace(**kwargs)


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


def test_main_exploiter_initial_weights_are_independent():
    _assert_exploiter_initial_weights_are_independent(league.MainExploiter)


def test_league_exploiter_initial_weights_are_independent():
    _assert_exploiter_initial_weights_are_independent(league.LeagueExploiter)


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
