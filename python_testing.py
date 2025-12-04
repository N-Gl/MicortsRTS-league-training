from enum import IntEnum
import torch
import numpy as np




arr = []
for _ in range(10000):
    coin_toss = np.random.random()

    # Make sure you can beat the League
    # 50% of the time, play PFSP against all historical players (jetzt: 0,7)
    if coin_toss < 0.7:
      arr.append(1)
    else:
        # 35% of the time, play self-play TODO (training): implementiere mehr main Agenten oder passe die Wahrscheinlichkeiten an (jetzt: 0,15)

        opponent = np.random.choice(["a", "b", "c"])

        # Verify if there are some rare players we omitted
        # 15% of PFSP matches against forgotten main players the agent can no longer beat and past main exploiters
        if coin_toss > 1 - 0.15:
            arr.append(2)
        else:
            arr.append(3)

print(arr.count("1")/10000)


print(arr.count(1)/10000)

# next_obs = torch.rand(6, 2, 2, 5)
# 
# num_selfplay_envs = 4
# 
# # next_obs = torch.rand(2, 2, 2, 3)
# # 
# # num_selfplay_envs = 2
# if num_selfplay_envs > 1:
#     if 2 < num_selfplay_envs:
#         next_obs[1:num_selfplay_envs:2, :, :, 1:4] = next_obs[1:num_selfplay_envs:2, :, :, 4:1]
#     else:
#         next_obs[1:num_selfplay_envs:2, :, :, 1:4] = next_obs[1:num_selfplay_envs:2, :, :, 4:1]
# 
# 
# print(next_obs.shape)

# zFeatures = torch.rand(3, 2, 4)
# 
# b_z = zFeatures.reshape(-1, 8)
# print(zFeatures.shape)
# print(b_z.shape)
# print(zFeatures)
# print(b_z)



# x = ("halloha", 2)
# print(x[0][3:len(x[0])-2])



# dones = torch.zeros((10, 2))
# dones[1, 0] = 1.0
# dones[4, 1] = 1.0
# dones[7, 1] = 1.0
# dones[8, 1] = 1.0
# dones[8, 0] = 1.0
# 
# print(dones)
# nextnonterminal = 1.0 - dones
# print(nextnonterminal)






#    class SelfplayAgentType(IntEnum):
#        # TODO: füge aktuellen main Agent hinzu, da sie mit in selfplay envs eingemischt seien können, wenn sie gegen andere selfplaying agents spielen.
#        OLD_MAIN = 0
#        MAIN_EXPLOITER = 1
#        LEAGUE_EXPLOITER = 2
#    
#        def not_implemented(self, name):
#            raise NotImplementedError(f"{name} get_action wasn't found in get_action_calls")
#    
#        def get_agents_action(self, split_x, split_Sc, split_z, split_action, split_invalid_action_masks, split_env_Indices):
#            # TODO: fill in the get_action calls
#            get_action_calls = {
#                self.OLD_MAIN: self.not_implemented("OLD_MAIN"),
#                self.MAIN_EXPLOITER: self.not_implemented("MAIN_EXPLOITER"),
#                self.LEAGUE_EXPLOITER: self.not_implemented("LEAGUE_EXPLOITER")
#            }
#    
#    
#            # split_action = {SelfplayAgentType.OLD_MAIN: split_x[0],
#            #     SelfplayAgentType.LEAGUE_EXPLOITER: split_x[1],
#            #     SelfplayAgentType.MAIN_EXPLOITER: split_x[2],
#            #     SelfplayAgentType.OLD_MAIN: split_x[3],
#            #     SelfplayAgentType.MAIN_EXPLOITER: split_x[4]}
#            # split_logproba = {SelfplayAgentType.OLD_MAIN: split_Sc[0],
#            #     SelfplayAgentType.LEAGUE_EXPLOITER: split_Sc[1],
#            #     SelfplayAgentType.MAIN_EXPLOITER: split_Sc[2],
#            #     SelfplayAgentType.OLD_MAIN: split_Sc[3],
#            #     SelfplayAgentType.MAIN_EXPLOITER: split_Sc[4]}
#            # split_entropy = {SelfplayAgentType.OLD_MAIN: split_z[0],
#            #     SelfplayAgentType.LEAGUE_EXPLOITER: split_z[1],
#            #     SelfplayAgentType.MAIN_EXPLOITER: split_z[2],
#            #     SelfplayAgentType.OLD_MAIN: split_z[3],
#            #     SelfplayAgentType.MAIN_EXPLOITER: split_z[4]}
#            # split_invalid_action_masks = {SelfplayAgentType.OLD_MAIN: split_invalid_action_masks[0],
#            #     SelfplayAgentType.LEAGUE_EXPLOITER: split_invalid_action_masks[1],
#            #     SelfplayAgentType.MAIN_EXPLOITER: split_invalid_action_masks[2],
#            #     SelfplayAgentType.OLD_MAIN: split_invalid_action_masks[3],
#            #     SelfplayAgentType.MAIN_EXPLOITER: split_invalid_action_masks[4]}
#    
#    
#            return split_x[:2], split_Sc[:2], split_z[:2], split_invalid_action_masks[:2]
#    
#    
#    
#    
#    agent_type = torch.tensor([
#        SelfplayAgentType.OLD_MAIN,
#        SelfplayAgentType.LEAGUE_EXPLOITER,
#        SelfplayAgentType.MAIN_EXPLOITER,
#        SelfplayAgentType.OLD_MAIN,
#    ], dtype=torch.long)
#    
#    
#    x = torch.rand(8, 10, 3)
#    Sc = torch.rand(8, 10, 1)
#    z = torch.rand(8, 2,)
#    invalid_action_masks = torch.rand(8, 10, 5)
#    
#    
#    
#    action_old = None
#    
#    
#    # print(f"all shapes: x: {x.shape}, Sc: {Sc.shape}, z: {z.shape}, m: {invalid_action_masks.shape}")
#    
#    indices_by_type = {t: (agent_type == t).nonzero(as_tuple=True)[0] for t in SelfplayAgentType}
#    # Teile die Inputs nach Agententypen auf und speichere auch den Index der Envs, rufe get_action von den jeweiligen Agenten auf
#    split_x = {t: x[idx] for t, idx in indices_by_type.items()}
#    split_Sc = {t: Sc[idx] for t, idx in indices_by_type.items()}
#    split_z = {t: z[idx] for t, idx in indices_by_type.items()}
#    split_action_old = {t: action_old[idx] for t, idx in indices_by_type.items()} if action_old is not None else None
#    split_invalid_action_masks = {t: invalid_action_masks[idx] for t, idx in indices_by_type.items()}
#    split_env_Indices = {t: idx for t, idx in indices_by_type.items()}
#    
#    # print(f"all shapes: x: {len(split_x), split_x[0].shape}, Sc: {len(split_Sc), split_Sc[0].shape}, z: {len(split_z), split_z[0].shape}, m: {len(split_invalid_action_masks), split_invalid_action_masks[0].shape}, i: {len(split_env_Indices), split_env_Indices[0].shape}")
#    
#    # for every agenttype: 1 SubEnvView (in ENUM method) with the respective indices
#    split_action, split_logproba, split_entropy, split_invalid_action_masks = \
#        torch.rand(4, 10, 3), torch.rand(4, 10, 1), torch.rand(4, 2,), torch.rand(4, 10, 5)
#    
#    # Handle non Selfplay Agents (current main Agent)
#    not_main = len(agent_type)
#    main_actions, main_logproba, main_entropy, main_invalid_action_masks = \
#        torch.rand(4, 10, 3), torch.rand(4, 10, 1), torch.rand(4, 2,), torch.rand(4, 10, 5)
#    
#    # print(f"all shapes: split_action: {len(split_action), split_action[0].shape}, split_logproba: {len(split_logproba), split_logproba[0].shape}, split_entropyz: {len(split_entropy), split_entropy[0].shape}, split_invalid_action_masks: {len(split_invalid_action_masks), split_invalid_action_masks[0].shape}, i: {len(split_env_Indices), split_env_Indices[0].shape}")
#    # print(f"all shapes: main_actions: {main_actions.shape}, main_logproba: {main_logproba.shape}, main_entropy: {main_entropy.shape}, m: {main_invalid_action_masks.shape}")
#    # print("split_env_Indices: ", split_env_Indices)
#    
#    # concatinate and resort to original order
#    split_env_Indices = torch.cat([split_env_Indices[i] for i in range(len(split_env_Indices))], dim=0)
#    
#    action = torch.empty_like(split_action)
#    for i, idx in enumerate(split_env_Indices):
#        action[i] = split_action[idx]
#    action = torch.cat([action, main_actions], dim=0)
#    
#    logprob = torch.empty_like(split_logproba)
#    for i, idx in enumerate(split_env_Indices):
#        logprob[i] = split_logproba[idx]
#    logprob = torch.cat([logprob, main_logproba], dim=0)
#    
#    entropy = torch.empty_like(split_entropy)
#    for i, idx in enumerate(split_env_Indices):
#        entropy[i] = split_entropy[idx]
#    entropy = torch.cat([entropy, main_entropy], dim=0)
#    
#    invalid_action_masks = torch.empty_like(split_invalid_action_masks)
#    for i, idx in enumerate(split_env_Indices):
#        invalid_action_masks[i] = split_invalid_action_masks[idx]
#    invalid_action_masks = torch.cat([invalid_action_masks, main_invalid_action_masks], dim=0)
#    
#    arr = torch.tensor(range(100))
#    new_arr = torch.zeros(8)
#    for i, idx in enumerate(split_env_Indices):
#        new_arr[i] = arr[idx]
#    arr = torch.cat([new_arr, torch.tensor([6, 7, 8, 9])], dim=0)
# print(f"arr: {arr}")

# print(f"action: {action}")


# print(f"all shapes: main_actions: {action.shape}, main_logproba: {logprob.shape}, main_entropy: {entropy.shape}, m: {invalid_action_masks.shape}, i: {split_env_Indices.shape}")

# action = action.permute(split_env_Indices.tolist())
# logprob = logprob.permute(split_env_Indices.tolist())
# entropy = entropy.permute(split_env_Indices.tolist())
# invalid_action_masks = invalid_action_masks.permute(split_env_Indices.tolist())




# Deterministische Permutation mit einer Mapping-Liste.
# Default-Interpretation: perm[i] = Zielposition des Elements, das aktuell an Index i steht.
# Beispiel: perm = [2,0,1,...] verschiebt arr[0] nach pos 2, arr[1] nach pos 0, arr[2] nach pos 1 usw.


# Alternative Interpretation (falls perm statt Zielpositionen Quellindices enthält):
# # perm[i] = Quelle für Position i -> new_arr[i] = arr[perm[i]]
# new_arr = arr[perm]  # funktioniert, wenn perm is LongTensor mit Länge n und gültigen Indizes

# for t in SelfplayAgentType:
            # print(f"Agent type {t}:")
            # print(f"{agent_type}")

# print(split_x[0].shape, ":  ", x.shape)
# print(split_Sc[0].shape, ":  ", Sc.shape)
# print(split_z[0].shape, ":  ", z.shape)
# print(split_invalid_action_masks[0].shape, ":  ", invalid_action_masks.shape)
# print(split_env_Indices[0].shape)





# indices_by_type = {
#     0: [0, 3],
#     1: [1, 4],
#     2: [2]
# }
# 
# x = torch.rand(8, 10, 3)
# 
# split_x = {t: [x[i] for i in idx] for t, idx in indices_by_type.items()}
# 
# print(x[0] is split_x[0])
# print(x[0] is split_x[0][0])
# print(x[0] is split_x[0][0][0])


