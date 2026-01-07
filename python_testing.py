import types

import torch

import agent_model
import league


def _make_dummy_args():
    return types.SimpleNamespace()


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
