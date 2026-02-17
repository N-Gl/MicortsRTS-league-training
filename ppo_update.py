import torch
import numpy as np
import time


def r2_score(y_pred, y_true):
    if y_true.numel() == 0:
        return None
    y_true = y_true.detach().float().view(-1)
    y_pred = y_pred.detach().float().view(-1)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    if ss_tot.item() == 0:
        return torch.tensor(0.0, device=y_true.device)
    return 1.0 - ss_res / ss_tot

def log(
    args,
    writer,
    optimizer,
    global_step,
    start_time,
    update,
    pg_stop_iter,
    pg_loss,
    entropy_loss,
    kl_loss,
    approx_kl,
    v_loss,
    loss,
    log_SPS=True,
    grad_norm=None,
    advantages=None,
    values=None,
    returns=None,
    delta_rewards_score=None,
):
    should_log_every_20_updates = (update % 20 == 0)

    writer.add_scalar("main_charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("progress/update", update, global_step)
    if loss is not None:
        writer.add_scalar("losses/value_loss", args.vf_coef * v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/kl_loss", kl_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", args.ent_coef * entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("main_charts/grad_norm_before_clipping", grad_norm, global_step)
        if getattr(args, "log_unweighted_losses", True):
            writer.add_scalar("losses/value_loss_raw", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy_loss_raw", entropy_loss.item(), global_step)

    if values is not None and returns is not None:
        r2 = r2_score(values, returns)
        if r2 is not None:
            writer.add_scalar("main_charts/r2_score", r2.item(), global_step)

    if advantages is not None and should_log_every_20_updates:
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.as_tensor(advantages)
        if advantages.numel() > 0:
            with torch.no_grad():
                adv = advantages.detach().float()
                writer.add_scalar("main_advantage/advantage_mean", adv.mean().item(), global_step)
                writer.add_scalar("main_advantage/advantage_std", adv.std(unbiased=False).item(), global_step)
                writer.add_scalar("main_advantage/advantage_min", adv.min().item(), global_step)
                writer.add_scalar("main_advantage/advantage_max", adv.max().item(), global_step)
                writer.add_scalar("main_advantage/advantage_pos_frac", (adv > 0).float().mean().item(), global_step)


    with torch.no_grad():
        drs = delta_rewards_score.detach().float() if delta_rewards_score is not None else torch.tensor(0.0)
        writer.add_scalar(f"main_delta_scores/mean", drs.mean().item(), args.global_step)
        writer.add_scalar(f"main_delta_scores/abs_mean", drs.abs().mean().item(), args.global_step)

    if (args.kle_stop or args.kle_rollback):
        if pg_stop_iter == -1:
            writer.add_scalar("debug/pg_stop_iter", args.update_epochs, global_step)
        elif args.kle_rollback:
            writer.add_scalar("debug/pg_stop_iter", 0, global_step)
        else:
            writer.add_scalar("debug/pg_stop_iter", pg_stop_iter, global_step)

    if log_SPS:
        writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))


def gae(args, device, b_next_value, b_values, b_rewards_attack, b_rewards_winloss, b_rewards_score, b_dones, b_next_done):
    b_advantages = torch.zeros_like(b_rewards_winloss).to(device)
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            # zuerst nonterminal, nextvalues aus dem Schritt im Setup
            # benutzen (anders repräsentiert)
            nextnonterminal = 1.0 - b_next_done
            nextvalues = b_next_value
        else:
            # für jede Umgebung: 1 -> nicht done in step t+1, 0 -> done in
            # step t+1
            nextnonterminal = 1.0 - b_dones[t + 1]
            nextvalues = b_values[t + 1]
                
        # TD-Error = R_(t+1) + γ * V(S_(t+1)) - V(S_t) per environment
        # V(S_t): Value of the state reached in the rollout after Action in step t-1
        # nextvalues: Critic-approximated values per environment in step t
        # for the next step in the rollout (if not terminated)
        # rewards_dense[t] + + rewards_dense[t] +rewards_score[t]
        # TODO (training): adjust reward for getting new Historical checkpoints (breaks value_loss, total_loss)
        # delta = args.hist_reward + rewards_winloss[t] + rewards_attack[t] + \
        #     args.gamma * nextvalues * nextnonterminal - values[t]
        delta = (
                    b_rewards_winloss[t]
                    + b_rewards_attack[t]
                    + b_rewards_score[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - b_values[t]
                )
        # A_t="TD-Error" + γ * λ * A_(t-1)
        
        b_advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    b_returns = b_advantages + b_values

    # TODO (debugging): remove later
    if args.dbg_update_gaes and b_advantages.mean().item() < 0:
        breakpoint()

    return b_advantages, b_returns
    
    
# TODO: Debugging (nachher entfernen)
def assert_supervised_grads_zero(supervised_agent):
    max_abs = 0.0
    names = []
    for name, p in supervised_agent.named_parameters():
        if p.grad is None:
            continue
        m = p.grad.detach().abs().max().item()
        if m > max_abs:
            max_abs = m
            names = [name]
            print(f"\n\n[supervised] max|grad|={max_abs} in {names}!!!\n\n")

    
def update(args, envs, agent_batch, device, supervised_agent, update, new_batch_size, minibatch_size):

    agent = agent_batch["agent"]
    optimizer = agent_batch["optimizer"]
    b_values, b_advantages, b_returns = agent_batch["values"], agent_batch["advantages"], agent_batch["returns"]
    b_Sc, b_z, b_obs = agent_batch["sc"], agent_batch["z"], agent_batch["obs"]
    b_actions, b_logprobs, b_invalid_action_masks = agent_batch["actions"], agent_batch["logprobs"], agent_batch["masks"]
    b_unit_bonus_distr = agent_batch.get("unit_bonus_distr")
    ent_coef = agent_batch.get("ent_coef", args.ent_coef)
    vf_coef = agent_batch.get("vf_coef", args.vf_coef)
    clip_coef = agent_batch.get("clip_coef", args.clip_coef)
    target_kl = agent_batch.get("target_kl", args.target_kl)
    kl_coeff = agent_batch.get("kl_coeff", args.kl_coeff)
    max_grad_norm = agent_batch.get("max_grad_norm", args.max_grad_norm)
    update_epochs = agent_batch.get("update_epochs", args.update_epochs)
    value_warmup_updates = agent_batch.get("value_warmup_updates", args.value_warmup_updates)
    kle_stop = agent_batch.get("kle_stop", args.kle_stop)
    kle_rollback = agent_batch.get("kle_rollback", args.kle_rollback)
    norm_adv = agent_batch.get("norm_adv", args.norm_adv)
    clip_vloss = agent_batch.get("clip_vloss", args.clip_vloss)
    skip_policy_update = agent_batch.get("skip_policy_update", False)# or args.dbg_no_main_agent_ppo_update
    agent_idx = agent_batch.get("agent_idx", None)
    # values_shape = agent_batch.get("values_shape", (-1,))
    # returns_shape = agent_batch.get("returns_shape", (-1,))
    # advantages_shape = agent_batch.get("advantages_shape", (-1,))
    
    if b_unit_bonus_distr is not None:
        b_unit_bonus_distr = b_unit_bonus_distr.to(device)
    elif getattr(agent, "unit_exploiters", False):
        b_unit_bonus_distr = torch.zeros((new_batch_size, 4), device=device)

    # Optimizing policy and value network with minibatch updates
    # --num_minibatches, --update-epochs
    # minibatches_size = int(args.batch_size // args.num_minibatches)
    # new (BA Parameter) minibatch_size
    inds = np.arange(new_batch_size)

    # Go (update_epochs times) through all mini-batches
    '''
    für jeden Minibatch im Batch berechne Â_t (Advantage Schätzer (R_t^((λ))-V_ϕ^(π_old)) (hier eher (V_ϕ (s_t )-R_t^((λ)) und dann - genommen in pg_loss))) (normalisiert),
    Wahrscheinlichkeit Action a in State s zu bekommen mit neuem θ / Wahrscheinlichkeit Action a in State s zu bekommen mit altem θ_old
    pgLoss (gegenteil von L_clip) ausrechnen, kombinieren mit Entropie Bonus, KL Divergenz Loss und Value Loss mit Updates minimieren
    '''

    value_only_phase = update <= value_warmup_updates
    # Optional rollback: snapshot params before policy epochs
    old_params = None
    if not value_only_phase and kle_rollback:
        # create a detached copy of parameters for rollback
        old_params = {k: v.detach().clone() for k, v in agent.state_dict().items()}
    pg_stop_iter = -1
    grad_norm = None

    epoch_indices = range(update_epochs)
    if skip_policy_update:
        print("\nDebug: skipping PPO update for main agent\n")
        # epoch_indices = []
        return None, None, None, None, None, None, None, None
    
    for epoch_pi in epoch_indices:
        np.random.shuffle(inds)
        for start in range(0, new_batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_ind = torch.as_tensor(inds[start:end], device=device)

            if agent_batch.get("full_tensores", False):

                if isinstance(agent_idx, slice):
                    agent_idx = torch.arange(b_obs.shape[1], device=device)[agent_idx]
                else:
                    agent_idx = torch.as_tensor(agent_idx, device=device, dtype=torch.long)

                k = agent_idx.numel()
                step_idx = torch.div(minibatch_ind, k, rounding_mode="floor")
                idx_in_agent_idx = torch.remainder(minibatch_ind, k)
                env_idx = agent_idx[idx_in_agent_idx]
                cur_idx = (step_idx, env_idx)

                mb_obs = b_obs[cur_idx]
                # old_mb_obs = b_obs[:, agent_idx].reshape(agent_batch["obs_shape"])[minibatch_ind]
                # print(torch.equal(mb_obs, old_mb_obs)) # True

                mb_sc = b_Sc[cur_idx]
                mb_z = b_z[cur_idx]
                mb_actions = b_actions[cur_idx]
                mb_masks = b_invalid_action_masks[cur_idx]
                # old_mb_masks = b_invalid_action_masks[:, agent_idx].reshape(agent_batch["masks_shape"])[minibatch_ind]
                # print(torch.equal(old_mb_masks, mb_masks)) # True
                mb_logprobs_old = b_logprobs[cur_idx]
                mb_values_old = b_values[cur_idx]

                b_env_idx = torch.where(env_idx < args.num_selfplay_envs, env_idx // 2, env_idx - (args.num_selfplay_envs // 2))
                mb_returns = b_returns[(step_idx, b_env_idx)]
                mb_advantages = b_advantages[(step_idx, b_env_idx)]
                
            else:
                mb_obs = b_obs[minibatch_ind]
                mb_sc = b_Sc[minibatch_ind]
                mb_z = b_z[minibatch_ind]
                mb_masks = b_invalid_action_masks[minibatch_ind]
                mb_logprobs_old = b_logprobs[minibatch_ind]
                mb_returns = b_returns[minibatch_ind]
                mb_values_old = b_values[minibatch_ind]
                mb_advantages = b_advantages[minibatch_ind]
                mb_actions = b_actions[minibatch_ind]
            # if mb_actions.dtype != torch.long:
            #     mb_actions = mb_actions.long()

            if norm_adv:
                # normalize the advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            mb_unit_bonus_distr = (
                b_unit_bonus_distr[minibatch_ind] if b_unit_bonus_distr is not None else None
            )

            # forward pass: get network output for the minibatch
            # We also provide actions here
            # (TODO (league training): muss man hier nicht mehr mit den unique_agents machen? nein, weil nur main agenten im batch sind)
            new_values = agent.get_value(
                mb_obs,
                mb_sc,
                mb_z,
                unit_bonus_distr=mb_unit_bonus_distr,
            ).view(-1)

            if value_only_phase:
                # Warmup: skip policy update, only train value head/backbone
                pg_loss = torch.zeros((), device=device)
                entropy_loss = torch.zeros((), device=device)
                kl_loss = torch.zeros((), device=device)
                approx_kl = torch.zeros((), device=device)
            else:
                # get_action nur für logprobs und entropy, um ratio zu berechnen (um zu vergleichen, wie wahrscheinlich die Action mit dem neuen θ im Vergleich zu dem alten θ_old ist)
                newlogproba, entropy = agent.get_action(
                        mb_obs,
                        mb_sc,
                        mb_z,
                        mb_actions,
                        mb_masks,
                        envs,
                        unit_bonus_distr=mb_unit_bonus_distr,
                    ) [1:3]
                ratio = (newlogproba - mb_logprobs_old).exp()

                # KL estimate for early stopping / rollback
                approx_kl = (mb_logprobs_old - newlogproba).mean()

                # Policy loss L^CLIP(θ) = E ̂_t ["min" (r_t (θ)*Â_t,"clip" (r_t (θ),1-ϵ,1+ϵ)*Â_t )]
                # --clip-coef
                # pg_loss = -L^CLIP(θ) (opposite)
                # it is the same (but negative), because in loss1 and 2 there is a minus sign and advantages are calculated differently
                # geht gegen 0, wenn es keine Verbesserung mehr gibt
                # gibt ein Wert für die Verbesserung der Policy in der aktuellen Iteration an
                # < 0  ⇒ Surrogate im Mittel verbessert (guter Update) (Policy verbessert sich)
                # ≈ 0  ⇒ kaum/keine (geclippte) Verbesserung
                # > 0  ⇒ Surrogate im Mittel schlechter (schlechter Update)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

                # TODO (debugging): remove later
                if args.dbg_update_gaes and pg_loss.item() > 0:
                    breakpoint()

            # Value loss Clipping
            # --clip_vloss
            # MSE(approximierte Values, returns) with or without clip()
            if clip_vloss:
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_clipped = mb_values_old + torch.clamp(
                        new_values - mb_values_old, -clip_coef, clip_coef
                    )
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - mb_returns) ** 2)

            if value_only_phase:
                pass
            else:
                # KL Divergence Loss
                with torch.no_grad():
                    # get_action nur für logprobs, um KL Divergenz zu berechnen
                   sl_logprobs = supervised_agent.get_action(
                            mb_obs,
                            mb_sc,
                            mb_z,
                            mb_actions,
                            mb_masks,
                            envs,
                            unit_bonus_distr=mb_unit_bonus_distr,
                        ) [1]
                kl_loss = kl_coeff * torch.nn.functional.kl_div(
                        newlogproba, sl_logprobs, log_target=True, reduction="batchmean"
                    )
            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss + kl_loss

            optimizer.zero_grad() # optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # TODO: nur für Debugging (nachher entfernen)
            if args.dbg_exploiter_update:
                assert_supervised_grads_zero(supervised_agent)
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            grad_norm = grad_norm.item()
            optimizer.step()

            # KL early stop / rollback
            if not value_only_phase and (kle_stop or kle_rollback):
                if approx_kl.item() > target_kl:
                    pg_stop_iter = epoch_pi
                    if kle_rollback and old_params is not None:
                        # revert to snapshot and exit epochs
                        agent.load_state_dict(old_params)
                    break
        if pg_stop_iter != -1:
            break
    return pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss, grad_norm
