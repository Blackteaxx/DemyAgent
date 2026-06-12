
# ------------------------------------------------------------
# File: hierarchical_dueling.py
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math


###############################################################################
# Basic Helpers
###############################################################################
def compute_entropy(probs):
    p_ = probs + 1e-9
    return - (p_ * p_.log()).sum()


###############################################################################
# ReplayBuffer
###############################################################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0

    def store(self, sub_trajs, ref_model_state):
        item = (sub_trajs, ref_model_state)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.idx] = item
        self.idx = (self.idx + 1) % self.capacity

    def get_all(self):
        return list(self.buffer)


###############################################################################
# get_distribution
###############################################################################
def get_distribution(pi_model, states_so_far, local_ex, gamma, print_stats_flag=False):

    assert len(states_so_far) > 0
    s = len(states_so_far)

    seq_tensor = torch.stack(states_so_far, dim=0).unsqueeze(0) # =>(1, s, H, W)
    out = pi_model(seq_tensor)  # =>(1, s, H, W)
    
    # last => shape(1, H, W)
    z_t = out[0, s-1, :, :]  # =>(H,W)


    ###########################################
    distances = []
    for i in range(local_ex.shape[0]):
        ex_img = local_ex[i]
        #
        dist_sq = (z_t - ex_img).pow(2).sum()
        distances.append(dist_sq)

    q_img = pi_model.query_param
    #
    dist_q = (z_t - q_img).pow(2).sum()
    distances.append(dist_q)

    ##
    dis_cat = torch.stack(distances, dim=0)  # =>(#ex+1)

    ##
    dis_cat = dis_cat - dis_cat.min()
    #
    sc_cat = torch.exp(- dis_cat / gamma)
    probs = sc_cat / sc_cat.sum()

    ###########################################

    eps = 1e-7
    probs = torch.clamp(probs, min=eps, max=1.0)
    probs = probs / probs.sum()

    ###
    return probs, z_t


###############################################################################
# generate_trajectory
###############################################################################
def generate_trajectory(
    ep,
    pi_model,
    s0,
    horizon,
    exemplars,
    query_sample_indices,
    eval_traj_reward_func,
    args
):
    ###
    gamma = args.gamma

    ###
    sub_trajectories = []
    trajectory = []
    step_info = []

    local_ex = exemplars.detach().clone()
    statesSoFar = [s0.detach().clone()]
    trajectory.append((statesSoFar[:], None))

    #
    chosenActionsSoFar = [s0.detach().clone()]

    #
    exem_index_list = list(range(local_ex.shape[0]))
    exemplar_seq_indices = []

    for t in range(horizon):
        # Generate distribution
        dist, _ = get_distribution(pi_model, statesSoFar, local_ex, gamma, print_stats_flag=True)

        ###
        epsilon_val = args.epsilon_explore_val

        ###
        a_idx  = torch.multinomial(dist, 1).item()
        
        #
        if random.random() <= epsilon_val:
            a_idx  = local_ex.shape[0]

        ##
        logp_chosen  = torch.log(dist[a_idx] + 1e-9)

        if a_idx == local_ex.shape[0]:
            if len(exemplar_seq_indices) > 0:
                step_info.append((logp_chosen, a_idx))
                #
                trajectory.append((None, None))

                r_sub = eval_traj_reward_func(exemplar_indices=exemplar_seq_indices, 
                                              eval_target_indices=query_sample_indices)
                print("Query: ", t, ", reward: ", r_sub)
                sub_trajectories.append((trajectory[:], r_sub, step_info[:]))
                #
                trajectory.pop()
                step_info.pop()

            if local_ex.shape[0] > 0:
                valid_probs = dist[: local_ex.shape[0]]
                valid_probs = valid_probs / valid_probs.sum()
                #
                ex_idx = torch.multinomial(valid_probs, 1).item()
                logp2 = torch.log(valid_probs[ex_idx] + 1e-9)
                #
                step_info.append((logp2, ex_idx))
                chosen_ex = local_ex[ex_idx]
                #
                statesSoFar.append(chosen_ex)
                chosenActionsSoFar.append(chosen_ex)
                #
                trajectory.append((None, None))  
                #
                local_ex = torch.cat([ local_ex[:ex_idx], local_ex[ex_idx+1:] ], dim=0)
                exem_index = exem_index_list.pop(ex_idx)
                exemplar_seq_indices.append(exem_index)
            else:
                break
        else:
            # normal
            step_info.append((logp_chosen, a_idx))
            chosen_ex = local_ex[a_idx]
            #
            statesSoFar.append(chosen_ex)
            chosenActionsSoFar.append(chosen_ex)
            #
            trajectory.append((None, None))
            #
            local_ex = torch.cat([ local_ex[:a_idx], local_ex[a_idx+1:] ], dim=0)

            exem_index = exem_index_list.pop(a_idx)
            exemplar_seq_indices.append(exem_index)

        if local_ex.shape[0] == 0:
            break

    ###
    trajectory.append((None, None))
    
    #
    r_final = eval_traj_reward_func(exemplar_indices=exemplar_seq_indices, 
                                    eval_target_indices=query_sample_indices)
    print("Final reward: ", r_final)
    
    #
    sub_trajectories.append((trajectory[:], r_final, step_info[:]))

    return sub_trajectories



###############################################################################
# generate_trajectory for TESTING
###############################################################################
def generate_trajectory_testing(
    pi_model,
    s0,
    horizon,
    exemplars,
    gamma,
    trunc_trajectory_flag=False
):

    #########################
    local_ex = exemplars.detach().clone()
    statesSoFar = [s0.detach().clone()]
    chosenActionsSoFar = [s0.detach().clone()]

    #
    exem_index_list = list(range(local_ex.shape[0]))
    exemplar_seq_indices = []

    #
    for _ in range(horizon):
        dist, _ = get_dist_testing(pi_model, statesSoFar, local_ex, gamma, 
                                   trunc_flag=trunc_trajectory_flag)
        
        a_idx = torch.argmax(dist).item()

        ###
        if a_idx == local_ex.shape[0] and trunc_trajectory_flag:
            if len(exemplar_seq_indices) > 0: 
                break

            a_idx = torch.argmax(dist[:-1]).item()
        
        ###
        chosen_ex = local_ex[a_idx]
        statesSoFar.append(chosen_ex)
        chosenActionsSoFar.append(chosen_ex)

        ###
        local_ex = torch.cat([ local_ex[:a_idx], local_ex[a_idx+1:] ], dim=0)

        exem_index = exem_index_list.pop(a_idx)
        exemplar_seq_indices.append(exem_index)

        if local_ex.shape[0] == 0:
            break

    return exemplar_seq_indices


###############################################################################

def get_dist_testing(pi_model, states_so_far, local_ex, gamma, trunc_flag=False):
    s = len(states_so_far)

    seq_tensor = torch.stack(states_so_far, dim=0).unsqueeze(0) # =>(1, s, H, W)
    out = pi_model(seq_tensor)  # =>(1, s, H, W)
    
    # Sequential Model generated embedding
    z_emb = out[0, s-1, :, :]  # => (H,W)

    ###########################################
    distances = []
    for i in range(local_ex.shape[0]):
        ex_img = local_ex[i]
        dist_sq = (z_emb - ex_img).pow(2).sum()
        distances.append(dist_sq)

    if trunc_flag:
        q_img = pi_model.query_param
        dist_q = (z_emb - q_img).pow(2).sum()
        distances.append(dist_q)

    #
    dis_cat = torch.stack(distances, dim=0)  # =>(#ex+1)

    ###
    dis_cat = dis_cat - dis_cat.min()
    sc_cat = torch.exp(- dis_cat / gamma)
    probs = sc_cat / sc_cat.sum()
    
    ###########################################

    eps = 1e-7
    probs = torch.clamp(probs, min=eps, max=1.0)
    probs = probs / probs.sum()

    return probs, z_emb


###############################################################################
# best_sub_trajectory
###############################################################################
def best_sub_trajectory(sub_trajs):
    best_i = 0
    best_val = float('-inf')
    for i, (traj_, rew_, st_inf) in enumerate(sub_trajs):
        if rew_ > best_val:
            best_val = rew_
            best_i = i
    
    return sub_trajs[best_i], best_val


###############################################################################
# worst_sub_trajectory
###############################################################################
def worst_sub_trajectory(sub_trajs):
    worst_i = 0
    worst_val = float('inf')
    for i, (traj_, rew_, st_inf) in enumerate(sub_trajs):
        if rew_ <= worst_val:
            worst_val = rew_
            worst_i = i
    
    return sub_trajs[worst_i], worst_val


###############################################################################
# re_run_subtraj
###############################################################################
def re_run_subtraj(pi_model, traj, step_info, exemplars, gamma, replay_training_flag=True, avg_entropy_step_flag=True):

    local_ex = exemplars.detach().clone()
    logp_sum = torch.tensor(0.0).to(pi_model.device)
    entropy_sum = torch.tensor(0.0).to(pi_model.device)

    step_len = len(step_info)

    init_states, _ = traj[0]
    states_list = [init_states[0].detach()]
    chosenActionsSoFar = [init_states[0].detach()]

    for i in range(step_len):
        (lp_stored, chosen_idx) = step_info[i]
        
        if replay_training_flag:
            dist, _ = get_distribution(pi_model, states_list, local_ex, gamma)
            logp_sum += torch.log(dist[chosen_idx] + 1e-9)
            
            entropy_sum += compute_entropy(dist)

            # ###
            if chosen_idx < local_ex.shape[0]:
                ###
                chosen_ex = local_ex[chosen_idx]
                states_list.append(chosen_ex)
                chosenActionsSoFar.append(chosen_ex)
                
                local_ex = torch.cat([ local_ex[:chosen_idx], local_ex[chosen_idx+1:] ], dim=0)
        else:
            logp_sum += lp_stored

    if avg_entropy_step_flag:
        entropy_sum = entropy_sum / step_len

    return logp_sum, entropy_sum


###############################################################################
# compute_intra_trajectory_loss
###############################################################################

def compute_intra_trajectory_loss(
    sub_trajs,
    pi_theta,
    pi_ref,
    exemplars,
    args,
    replay_training_flag=True,
    print_stats_flag=False
):

    beta = args.beta
    gamma = args.gamma
    entropy_loss_coefficient = args.entropy_loss_coefficient
    #
    avg_entropy_flag = True if entropy_loss_coefficient < 0 else False

    device = pi_theta.device

    total_loss = torch.tensor(0.0, device=device)
    total_entropy = torch.tensor(0.0, device=device)

    ###
    if len(sub_trajs) > 1:
        sc_list, log_prob_list = [], []

        for (traj_, rew_, st_info) in sub_trajs:
            logp_ref, _ = re_run_subtraj(
                pi_ref, traj_, st_info, exemplars, gamma,
                replay_training_flag=replay_training_flag,
                avg_entropy_step_flag=avg_entropy_flag
            )
            sc_ = (rew_ / beta) + logp_ref
            sc_list.append(sc_)

            logp_th, entropy = re_run_subtraj(
                pi_theta, traj_, st_info, exemplars, gamma,
                replay_training_flag=replay_training_flag,
                avg_entropy_step_flag=avg_entropy_flag
            )

            log_prob_list.append(logp_th)

            total_entropy += entropy

        num_trajs = len(sub_trajs)
        total_entropy = total_entropy / num_trajs

        sc_tensor = torch.stack(sc_list)
        th_tensor = torch.stack(log_prob_list)

        #    pi_star(i) = exp(sc_tensor[i]) / sum_j exp(sc_tensor[j])
        pi_star = F.softmax(sc_tensor, dim=0)

        #    log_p_theta(i) = log( exp(th_tensor[i]) / sum_j exp(th_tensor[j]) )
        log_p_theta = F.log_softmax(th_tensor, dim=0)

        ce_loss = - (pi_star * log_p_theta).sum()

        if entropy_loss_coefficient != 0:
            if print_stats_flag:
                print(
                    f"- [Intra]: CE loss: {ce_loss.item():.6f}. "
                    f"Entropy Loss: {(entropy_loss_coefficient * total_entropy).item():.6f} -"
                )
            ce_loss += entropy_loss_coefficient * total_entropy

        total_loss = ce_loss

        if torch.isnan(total_loss):
            print("=== [compute_intra_trajectory_loss] NaN DETECTED! Debug Info ===")
            print(f"sc_list: {sc_list}")
            print(f"log_prob_list: {log_prob_list}")
            print(f"sc_tensor: {sc_tensor}")
            print(f"th_tensor: {th_tensor}")
            print(f"beta: {beta}")
            print(f"entropy_loss_coefficient: {entropy_loss_coefficient}")
            print(f"total_entropy: {total_entropy}")
            print(f"Final total_loss: {total_loss}")

    return total_loss


###############################################################################
# compute_inter_trajectory_loss
###############################################################################

def compute_inter_trajectory_loss(
    T1_best,
    T2_best,
    pi_theta,
    pi_ref,
    exemplars,
    args,
    replay_training_flag=True,
    print_stats_flag=False
):

    beta = args.beta
    gamma = args.gamma
    entropy_loss_coefficient = args.entropy_loss_coefficient
    #
    avg_entropy_flag = True if entropy_loss_coefficient < 0 else False

    # Unpack T1
    (traj1, rew1, info1) = T1_best
    # Re-run T1 under reference model
    logp1_ref, _ = re_run_subtraj(
        pi_ref, traj1, info1, exemplars, gamma,
        replay_training_flag=replay_training_flag,
        avg_entropy_step_flag=avg_entropy_flag
    )
    sc1 = (rew1 / beta) + logp1_ref

    (traj2, rew2, info2) = T2_best
    logp2_ref, _ = re_run_subtraj(
        pi_ref, traj2, info2, exemplars, gamma,
        replay_training_flag=replay_training_flag,
        avg_entropy_step_flag=avg_entropy_flag
    )
    sc2 = (rew2 / beta) + logp2_ref

    ref_scores = torch.stack([sc1, sc2], dim=0)   # shape [2]
    pi_star = F.softmax(ref_scores, dim=0)        # shape [2]
    pi_star1, pi_star2 = pi_star[0], pi_star[1]

    denom_1 = ref_scores.sum()

    logp1_th, entropy_1 = re_run_subtraj(
        pi_theta, traj1, info1, exemplars, gamma,
        replay_training_flag=replay_training_flag,
        avg_entropy_step_flag=avg_entropy_flag
    )
    logp2_th, entropy_2 = re_run_subtraj(
        pi_theta, traj2, info2, exemplars, gamma,
        replay_training_flag=replay_training_flag,
        avg_entropy_step_flag=avg_entropy_flag
    )

    # shape [2] => [ logp1_th, logp2_th ]
    theta_scores = torch.stack([logp1_th, logp2_th], dim=0)
    exp_1_traj = theta_scores[0]   
    exp_2_traj = theta_scores[1]   

    # denom_2 for continuity in debug prints
    denom_2 = theta_scores.sum()

    # shape: [2], [ log_p_normed_1, log_p_normed_2 ]
    log_p_theta = F.log_softmax(theta_scores, dim=0)
    log_p_normed_1, log_p_normed_2 = log_p_theta[0], log_p_theta[1]

    # Cross-entropy => - sum_i pi_star(i) * log p_theta(i)
    ce_loss = - (pi_star1 * log_p_normed_1 + pi_star2 * log_p_normed_2)

    # Entropy regularization
    if entropy_loss_coefficient != 0:
        if print_stats_flag:
            print("- [Inter]: CE loss: {:.6f}. Entropy Loss: {:.6f} -".format(
                ce_loss.item(),
                (entropy_loss_coefficient * (entropy_1 + entropy_2) / 2).item()
            ))
        loss = ce_loss + (entropy_loss_coefficient * (entropy_1 + entropy_2) / 2)
    else:
        loss = ce_loss

    if torch.isnan(loss).any():
        print("=== [compute_inter_trajectory_loss] NaN DETECTED! Detailed Debug Info ===")
        print(f"rew1: {rew1}, rew2: {rew2}, beta: {beta}")
        print(f"logp1_ref: {logp1_ref}, logp2_ref: {logp2_ref}")
        print(f"sc1: {sc1}, sc2: {sc2}, denom_1: {denom_1}")
        print(f"pi_star1: {pi_star1}, pi_star2: {pi_star2}")
        print(f"logp1_th: {logp1_th}, logp2_th: {logp2_th}")
        print(f"entropy_1: {entropy_1}, entropy_2: {entropy_2}")
        print(f"exp_1_traj: {exp_1_traj}, exp_2_traj: {exp_2_traj}")
        print(f"denom_2: {denom_2}")
        print(f"log_p_normed_1: {log_p_normed_1}, log_p_normed_2: {log_p_normed_2}")
        print(f"entropy_loss_coefficient: {entropy_loss_coefficient}")
        print(f"Final computed CE portion: {-ce_loss.item()}")
        print(f"Final loss with entropy term: {loss}")

    return loss



def normalize_subtrajectories(target_trajectories: list, source_trajectories: list) -> list:
    rewards = []
    normalized_subtrajectories = []
    normalized_reward_elements = []
    raw_reward_elements = []

    if len(target_trajectories) > 0:
        for (_, reward, _) in source_trajectories:
            if isinstance(reward, torch.Tensor):
                rewards.append(float(reward.item()))
            else:
                rewards.append(float(reward))
        
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        mean_r = rewards_tensor.mean()
        std_r = rewards_tensor.std()
        epsilon = 1e-8
        std_r = std_r + epsilon
        
        for traj, reward, step_info in target_trajectories:
            raw_reward_elements.append(reward)
            if isinstance(reward, torch.Tensor):
                normalized_reward = (reward - mean_r) / std_r
            else:
                normalized_reward = (torch.tensor(reward, dtype=torch.float32) - mean_r) / std_r
            normalized_reward_elements.append(normalized_reward)
            normalized_subtrajectories.append((traj, normalized_reward, step_info))
    
    return normalized_subtrajectories, (raw_reward_elements, normalized_reward_elements)



