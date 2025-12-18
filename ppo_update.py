import torch
import numpy as np
import time

def log(args, writer, optimizer, global_step, start_time, update, pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss, log_SPS=True):
    writer.add_scalar("main_charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("progress/update", update, global_step)
    if not args.dbg_no_main_agent_ppo_update:
        writer.add_scalar("losses/value_loss", args.vf_coef * v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/kl_loss", kl_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", args.ent_coef * entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)

    if args.kle_stop or args.kle_rollback:
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

    value_only_phase = update <= args.value_warmup_updates
    # Optional rollback: snapshot params before policy epochs
    old_params = None
    if not value_only_phase and args.kle_rollback:
        # create a detached copy of parameters for rollback
        old_params = {k: v.detach().clone() for k, v in agent.state_dict().items()}
    pg_stop_iter = -1

    epoch_indices = range(args.update_epochs)
    if args.dbg_no_main_agent_ppo_update:
        print("\nDebug: skipping PPO update for main agent\n")
        epoch_indices = []
    
    for epoch_pi in epoch_indices:
        np.random.shuffle(inds)
        for start in range(0, new_batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]

            if args.norm_adv:
                # normalize the advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # forward pass: get network output for the minibatch
            # We also provide actions here
            # (TODO (league training): muss man hier nicht mehr mit den unique_agents machen? nein, weil nur main agenten im batch sind)
            new_values = agent.get_value(b_obs[minibatch_ind], b_Sc[minibatch_ind], b_z[minibatch_ind]).view(-1)

            if value_only_phase:
                # Warmup: skip policy update, only train value head/backbone
                pg_loss = torch.zeros((), device=device)
                entropy_loss = torch.zeros((), device=device)
                kl_loss = torch.zeros((), device=device)
                approx_kl = torch.zeros((), device=device)
            else:
                # get_action nur für logprobs und entropy, um ratio zu berechnen (um zu vergleichen, wie wahrscheinlich die Action mit dem neuen θ im Vergleich zu dem alten θ_old ist)
                _, newlogproba, entropy, _ = agent.get_action(
                        b_obs[minibatch_ind],
                        b_Sc[minibatch_ind],
                        b_z[minibatch_ind],
                        b_actions.long()[minibatch_ind],
                        b_invalid_action_masks[minibatch_ind],
                        envs,
                    )
                ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

                # KL estimate for early stopping / rollback
                approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

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
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()

            # Value loss Clipping
            # --clip_vloss
            # MSE(approximierte Values, returns) with or without clip()
            if args.clip_vloss:
                v_loss_unclipped = (new_values - b_returns[minibatch_ind]) ** 2
                v_clipped = b_values[minibatch_ind] + torch.clamp(
                        new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef
                    )
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

            if value_only_phase:
                pass
            else:
                # KL Divergence Loss
                with torch.no_grad():
                    # get_action nur für logprobs, um KL Divergenz zu berechnen
                    _, sl_logprobs, _, _ = supervised_agent.get_action(
                            b_obs[minibatch_ind],
                            b_Sc[minibatch_ind],
                            b_z[minibatch_ind],
                            b_actions.long()[minibatch_ind],
                            b_invalid_action_masks[minibatch_ind],
                            envs,
                        )
                kl_loss = args.kl_coeff * torch.nn.functional.kl_div(
                        newlogproba, sl_logprobs, log_target=True, reduction="batchmean"
                    )
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss + kl_loss

            optimizer.zero_grad()
            loss.backward()
            # TODO: nur für Debugging (nachher entfernen)
            assert_supervised_grads_zero(supervised_agent)
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            # KL early stop / rollback
            if not value_only_phase and (args.kle_stop or args.kle_rollback):
                if approx_kl.item() > args.target_kl:
                    pg_stop_iter = epoch_pi
                    if args.kle_rollback and old_params is not None:
                        # revert to snapshot and exit epochs
                        agent.load_state_dict(old_params)
                    break
        if pg_stop_iter != -1:
            break
    return pg_stop_iter, pg_loss, entropy_loss, kl_loss, approx_kl, v_loss, loss