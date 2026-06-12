# ------------------------------------------------------------
# File: main.py
# ------------------------------------------------------------

import os
import sys
import datetime
import time
import traceback
import glob
import re
import pickle

# Get the directory of the current notebook or script
current_dir = os.getcwd()

# Append the "data" subdirectory to sys.path
data_path = os.path.join(current_dir, "data")
sys.path.append(data_path)
os.environ['HF_HOME'] = os.path.join(current_dir, "HF_CACHE")

import argparse
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
from collections import defaultdict, Counter

from model import SequenceViT, custom_vit_init
from phase2_train.utils.hierarchical_dueling import (
    ReplayBuffer,
    generate_trajectory,
    generate_trajectory_testing,
    compute_intra_trajectory_loss,
    compute_inter_trajectory_loss,
    best_sub_trajectory,
    worst_sub_trajectory,
    normalize_subtrajectories
)

from phase2_train.utils.data_loader import DataLoader

######################################################################

class Logger_class(object):
    def __init__(self, stdout, stderr, folder_str, dt_string, algo):
        self.terminal = stdout
        self.err_terminal = stderr
        self.log = open('{}/{}_log_{}_'.format(folder_str, algo, dt_string) + ".log", "w", buffering=1)
        print("date and time =", dt_string)

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        self.terminal.write(message)

    def flush(self):
        self.log.flush()
        self.terminal.flush()

    def write_error(self, message):
        self.log.write("[ERROR] " + message)
        self.log.flush()
        self.err_terminal.write(message)

######################################################################
######################################################################

def detach_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach()
    elif isinstance(obj, list):
        return [detach_tensors(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_tensors(item) for item in obj)
    elif isinstance(obj, dict):
        return {key: detach_tensors(value) for key, value in obj.items()}
    elif hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            setattr(obj, attr, detach_tensors(value))
    return obj

######################################################################
######################################################################

def register_logger(args):
    now = datetime.datetime.now()
    dt_string = now.strftime("%m-%d-%Y_%H-%M-%S") + '_{}_{}_{}_{}'.format(
        str(args.dataset), str(args.train_count),
        str(args.entropy_loss_coefficient),
        str(args.lr)
    )

    folder_str = './Running_logs/{}'.format(str(args.dataset))
    algo = 'Hierarchical_Dueling'
    os.makedirs(folder_str, exist_ok=True)

    logger = Logger_class(sys.stdout, sys.stderr, folder_str, dt_string, algo)
    sys.stdout = logger
    sys.stderr = logger

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__stderr__.write("KeyboardInterrupt detected. Exiting gracefully...\n")
            sys.exit(1)
        
        error_message = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        sys.stderr.write_error("\n==== Unhandled Exception ====\n" + error_message + "\n")

    sys.excepthook = exception_handler


######################################################################


def save_final_policy_model(args, pi_theta, final_test_score, test_results_data):
    model_dir = "./saved_pi_theta_models"
    os.makedirs(model_dir, exist_ok=True)

    pattern = os.path.join(
        model_dir,
        f"final_pi_theta_{args.dataset}_{args.seed}_{args.horizon}_*.pth"
    )
    existing = glob.glob(pattern)

    best_path = None
    best_score = -float("inf")
    for path in existing:
        fname = os.path.basename(path)
        name_match = re.match(
            rf"final_pi_theta_{re.escape(args.dataset)}_{args.seed}_{args.horizon}_(?P<score>[\d\.]+)\.pth$",
            fname
        )
        if name_match:
            score = float(name_match.group("score"))
            if score > best_score:
                best_score = score
                best_path = path

    should_save = (best_path is None) or (final_test_score > best_score)
    if should_save:
        if best_path:
            os.remove(best_path)
            print(f"Deleted previous best model: {best_path}")

        new_fname = f"final_pi_theta_{args.dataset}_{args.seed}_{args.horizon}_{final_test_score:.4f}.pth"
        save_path = os.path.join(model_dir, new_fname)
        torch.save(pi_theta.state_dict(), save_path)
        print(f"*** Final_model: New best model saved at {save_path} "
              f"with validation accuracy: {final_test_score:.4f} ***")
    else:
        print(f"No improvement over existing best ({best_score:.4f}), not saving.")

    results_filename = f"test_exemplar_choices_{args.dataset}_{args.seed}_{args.horizon}.pkl"
    results_save_path = os.path.join(model_dir, results_filename)
    try:
        with open(results_save_path, 'wb') as f:
            pickle.dump(test_results_data, f)
        print(f"--- Saved test exemplar choices to: {results_save_path} ---")
    except Exception as e:
        print(f"--- Error saving test results to {results_save_path}: {e} ---")


######################################################################


def update_reference(pi_theta, pi_ref, tau=0.01):
    for param, ref_param in zip(pi_theta.parameters(), pi_ref.parameters()):
        ref_param.data.mul_(1 - tau)
        ref_param.data.add_(tau * param.data)


def log_loss_transform(loss, L_th, max_scale=100.0, eps=1e-8):
    max_abs = L_th * max_scale
    loss_clipped = torch.clamp(loss, min=-max_abs, max=max_abs)
    transformed = torch.sign(loss_clipped) * L_th * torch.log1p(torch.abs(loss_clipped) / (L_th + eps))
    return transformed


def tanh_soft_clip(loss, threshold):
    return threshold * torch.tanh(loss / threshold)


##########

def gamma_scheduling_func(ep, init_gamma, final_gamma, warmup_rounds):
    slope = (init_gamma - final_gamma) / warmup_rounds
    return max(init_gamma - (ep * slope), final_gamma)


######################################################################

def get_args():
    parser = argparse.ArgumentParser()
    #
    parser.add_argument("--dataset", type=str, default="agnews")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model_name", type=str, default='Qwen/Qwen2.5-3B')
    #
    parser.add_argument("--train_count", type=int, default=100)
    parser.add_argument("--val_count", type=int, default=100)
    parser.add_argument("--test_count", type=int, default=400)
    #
    parser.add_argument("--embed_max_len", type=int, default=320)
    #
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--horizon", type=int, default=4)
    #
    parser.add_argument("--batch_query_training", type=int, default=5)
    #
    parser.add_argument("--beta", type=float, default=0.01)
    #
    parser.add_argument("--entropy_loss_coefficient", type=float, default=0)
    #
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--depth", type=int, default=4)
    #
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--dim_head", type=int, default=64)
    
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=32)
    ###
    parser.add_argument("--replay_capacity", type=int, default=50)
    parser.add_argument("--replay_train_samples_per_round", type=int, default=10)
    parser.add_argument("--replay_accumulation_steps", type=int, default=5)
    parser.add_argument("--replay_updates", type=int, default=1)
    #
    parser.add_argument('--active_stop_for_testing', action='store_true')
    parser.add_argument('--use_true_valid_and_testing', action='store_true')

    ###
    parser.add_argument("--query_action_loss_coef", type=float, default=1, help="Balance coefficient for query action loss.")

    ###
    parser.add_argument("--trajctories_per_init_state", type=int, default=3, help="Trajecotories queries for each initial state.")

    ##
    parser.add_argument("--te_intensity", type=float, default=0.2, help="Temporal embedding initialization intensity.")

    #####
    parser.add_argument("--epsilon_explore_val", type=float, default=0.7, help="Epsilon exploration probability - with prob. epsilon, query reward.")
    #
    parser.add_argument("--loss_clip_threshold", type=float, default=10, help="Loss clipping range.")
    parser.add_argument("--grad_clip_threshold", type=float, default=-1, help="Gradient clipping range.")
    #
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability.")

    #####
    parser.add_argument("--skip_threshold", type=float, default=1e-2, help="Skip round probability.")
    #
    parser.add_argument("--gamma", type=float, default=1e1)
    parser.add_argument("--final_gamma", type=float, default=1)
    parser.add_argument("--warmup_rounds", type=int, default=200)

    ###################################
    args = parser.parse_args()

    ### Manual Overwrite
    args.active_stop_for_testing = True
    args.use_true_valid_and_testing = True

    return args


######################################################################
######################################################################

def main():

    args = get_args()
    start_time = time.time()
    #
    register_logger(args=args)
    print("=" * 30)
    print(args)
    print("=" * 30)
    
    ################################################################################

    device = torch.device(args.device)

    data_loader = DataLoader(data_name=args.dataset, seed=args.seed, embed_max_len=args.embed_max_len, 
                             train_count=args.train_count, val_count=args.val_count, test_count=args.test_count,
                             device=device, model_name=args.model_name, embed_with_LLM_flag=False, multi_gpu_flag=True)

    pi_theta = SequenceViT(
        args=args,
        image_size=(data_loader.H, data_loader.W),
        patch_size=(args.patch_size, args.patch_size),
        seq_len=args.embed_max_len,
        horizon=args.horizon,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        init_query_emb=torch.mean(data_loader.exemplars_embs.detach(), dim=0),
        dim_head=args.dim_head,
        device=device,
        pos_emb_pattern='3d',
        te_intensity=args.te_intensity,
        causal_flag=True
    ).to(device)
    pi_theta.apply(custom_vit_init)
    pi_theta.train()

    optimizer = optim.AdamW(pi_theta.parameters(), lr=args.lr)

    exemplars = data_loader.exemplars_embs

    s0_list = data_loader.query_emb_list

    # build replay
    replay_buffer = ReplayBuffer(args.replay_capacity)
    pi_ref = copy.deepcopy(pi_theta)

    ###
    test_per_rounds = 50

    ###
    best_acc = 0
    model_dir = "./saved_pi_theta_models"
    os.makedirs(model_dir, exist_ok=True)
    model_filename = f"pi_theta_{args.dataset}_{args.horizon}.pth"
    save_model_path = os.path.join(model_dir, model_filename)

    ##########################################################################
    ################################ Training ################################

    episodes = args.episodes
    num_traj_per_init_state = args.trajctories_per_init_state
    init_gamma = args.gamma
    skipped_rounds = 0
    ep = 0
    ##
    for ep in range(episodes):
        
        ###
        T = args.horizon

        ###
        args.gamma = gamma_scheduling_func(ep=ep, init_gamma=init_gamma, final_gamma=args.final_gamma, warmup_rounds=args.warmup_rounds)

        ###
        if (ep + 1) % 1 == 0:
            update_reference(pi_theta, pi_ref, tau=0.01)
            pi_ref.eval()

        if args.batch_query_training > 1:
            q_idx = random.sample(range(len(s0_list)), k=args.batch_query_training)
            print("[q_idx]: ", q_idx)
            s0 = torch.mean(torch.stack([s0_list[idx] for idx in q_idx]), dim=0).detach().clone()
        else:
            q_idx = ep % len(s0_list)
            s0 = s0_list[q_idx].detach().clone()
            q_idx = [q_idx]
        
        trajectory_collection = []
        trajectory_union = []
        for _ in range(num_traj_per_init_state):
            this_T = generate_trajectory(
                ep = ep,
                pi_model = pi_theta,
                s0 = s0,
                horizon = T,
                exemplars = exemplars,
                query_sample_indices=q_idx,
                eval_traj_reward_func=data_loader.eval_exemplars,
                args=args
            )
            trajectory_collection.append(this_T)
            trajectory_union += this_T

        ####
        normalized_traj_collection = []
        normalized_traj_union = []
        for this_T in trajectory_collection:
            normalized_T, n_rewards_rollout = normalize_subtrajectories(target_trajectories=this_T, source_trajectories=trajectory_union)
            print(f"--- n_rewards_rollout: {n_rewards_rollout}")
            normalized_traj_collection.append(normalized_T)
            normalized_traj_union += normalized_T
        trajectory_collection = normalized_traj_collection
        trajectory_union = normalized_traj_union

        ############################################################################################
        fine_chosen_trajs = []
        ###
        all_stop_trajs = []
        for this_T in trajectory_collection:
            all_stop_trajs += [st for st in this_T if len(st[0]) < (args.horizon+2)]
        if len(all_stop_trajs) > 1:
            best_traj, best_val = best_sub_trajectory(all_stop_trajs)
            worst_traj, worst_val = worst_sub_trajectory(all_stop_trajs)
        else:
            best_traj, best_val = best_sub_trajectory(trajectory_union)
            worst_traj, worst_val = worst_sub_trajectory(trajectory_union)
        
        #####################################
        if best_val - worst_val < args.skip_threshold: 
            skipped_rounds += 1
            continue
        #####################################

        ####
        fine_chosen_trajs.append(best_traj)
        fine_chosen_trajs.append(worst_traj)
        
        ####
        normalized_fine_chosen_trajs, n_rewards = normalize_subtrajectories(target_trajectories=fine_chosen_trajs, 
                                                                            source_trajectories=fine_chosen_trajs)

        ### ----------------------------------------------------------- Use all sub-trajectoires
        L_intra_sum = compute_intra_trajectory_loss(
            trajectory_union, 
            pi_theta, 
            pi_ref,
            exemplars = exemplars,
            args=args,
            replay_training_flag=True,
            print_stats_flag=True
        )

        ####
        L_inter_margin = compute_intra_trajectory_loss(
            normalized_fine_chosen_trajs,
            pi_theta, 
            pi_ref,
            exemplars = exemplars,
            args=args,
            replay_training_flag=True,
            print_stats_flag=True
        )
        
        ############################################################################################
        if args.loss_clip_threshold > 0:
            L_total = args.query_action_loss_coef * log_loss_transform(L_inter_margin, args.loss_clip_threshold) + \
                                log_loss_transform(L_intra_sum, args.loss_clip_threshold)
        else:
            L_total = args.query_action_loss_coef * L_inter_margin + L_intra_sum

        #########
        optimizer.zero_grad()
        L_total.backward()
        

        ###
        if args.grad_clip_threshold > 0:
            torch.nn.utils.clip_grad_norm_(pi_theta.parameters(), max_norm=args.grad_clip_threshold)

        #
        optimizer.step()
        
        ###
        detached_trajectories_union = []
        detached_trajectory_collection = []
        for this_T in trajectory_collection:
            detached_T = detach_tensors(this_T)
            detached_trajectory_collection.append(detached_T)
            detached_trajectories_union += detached_T
        #
        replay_buffer.store(detached_trajectory_collection, None)

        ########################################################################################
        ################################ Replay Buffer Training ################################

        for r_t in range(args.replay_updates):
            items = replay_buffer.get_all()
            if not items:
                break
            #
            shuffled_indices = list(range(len(items)))
            random.shuffle(shuffled_indices)
            if args.replay_train_samples_per_round > 0:
                shuffled_indices = shuffled_indices[:args.replay_train_samples_per_round]

            ##### Mini-batch Updates
            optimizer.zero_grad() 

            for i, index in enumerate(shuffled_indices):
                #
                (replay_trajectory_collection, _) = items[index]
                replay_traj_collection_union = []
                for this_T in replay_trajectory_collection:
                    replay_traj_collection_union += this_T

                ### ----------------------------------------------------------- Use all sub-trajectoires
                replay_L_intra_sum = compute_intra_trajectory_loss(
                    replay_traj_collection_union,
                    pi_theta,
                    pi_ref,
                    exemplars=exemplars,
                    args=args,
                    replay_training_flag=True,
                    print_stats_flag=True
                )

                ##############################################################################################################
                fine_chosen_trajs_replay = []
                ###
                all_stop_trajs = []
                for this_T in replay_trajectory_collection:
                    all_stop_trajs += [st for st in this_T if len(st[0]) < (args.horizon+2)]
                if len(all_stop_trajs) > 1:
                    best_sub_traj, best_val = best_sub_trajectory(all_stop_trajs)
                    worst_sub_traj, worst_val = worst_sub_trajectory(all_stop_trajs)
                else:
                    best_sub_traj, best_val = best_sub_trajectory(replay_traj_collection_union)
                    worst_sub_traj, worst_val = worst_sub_trajectory(replay_traj_collection_union)
                ###
                fine_chosen_trajs_replay.append(best_sub_traj)
                fine_chosen_trajs_replay.append(worst_sub_traj)
                
                ####
                normalized_fine_chosen_trajs_replay, n_rewards_replay = normalize_subtrajectories(target_trajectories=fine_chosen_trajs_replay, 
                                                                                                  source_trajectories=fine_chosen_trajs_replay)                
                ###
                replay_inter_margin = compute_intra_trajectory_loss(
                    normalized_fine_chosen_trajs_replay,
                    pi_theta,
                    pi_ref,
                    exemplars=exemplars,
                    args=args,
                    replay_training_flag=True
                )

                ##############################################################################################################
                if args.loss_clip_threshold > 0:
                    clipped_replay_inter_margin = log_loss_transform(replay_inter_margin, args.loss_clip_threshold)
                    clipped_replay_L_intra_sum = log_loss_transform(replay_L_intra_sum, args.loss_clip_threshold)
                    L_total_replay = args.query_action_loss_coef * clipped_replay_inter_margin + clipped_replay_L_intra_sum
                else:
                    L_total_replay = args.query_action_loss_coef * replay_inter_margin + replay_L_intra_sum
                
                ##### Mini-batch Updates
                loss_item = L_total_replay / min(args.replay_accumulation_steps, len(shuffled_indices))  # average the loss over the mini-batch
                loss_item.backward()

                ##
                if (i + 1) % args.replay_accumulation_steps == 0 or (i + 1) == len(shuffled_indices):
                    total_norm = 0
                    for p in pi_theta.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)

                    ###
                    if args.grad_clip_threshold > 0:
                        torch.nn.utils.clip_grad_norm_(pi_theta.parameters(), max_norm=args.grad_clip_threshold)

                    optimizer.step()       # update parameters
                    optimizer.zero_grad()
                    #
                    print("Replay step.")
        
 
        ################################################################################################################
        #################################################### Validating ################################################
        if args.use_true_valid_and_testing:
            if (ep + 1) % test_per_rounds == 0 or ep in [0, episodes-1]:
                pi_theta.eval()
                correct_count = 0
                #
                validating_start_time = time.time()
                with torch.no_grad():
                    valid_rounds = len(data_loader.processor.val_dataset)
                    valid_emb_list = data_loader.query_emb_list
                    query_length_dist = defaultdict(lambda: 0)
                    exemplar_count = Counter()
                    
                    for test_round in tqdm(range(valid_rounds)):
                        v_idx = test_round % len(valid_emb_list)
                        s0 = valid_emb_list[v_idx].detach().clone()
                        v_idx = [v_idx]
                        
                        exemplar_indices = generate_trajectory_testing(
                            pi_model = pi_theta,
                            s0 = s0,
                            horizon = T,
                            exemplars = exemplars,
                            gamma = args.gamma,
                            trunc_trajectory_flag=args.active_stop_for_testing
                        )

                        query_length_dist[len(exemplar_indices)] += 1

                        correct_count += data_loader.valid_exemplars_with_training_data(exemplar_indices, valid_indices=v_idx)
                        exemplar_count.update(exemplar_indices)

                this_acc = correct_count / valid_rounds
                print("=== Episode: {}. Validation result: {} ===".format(ep, this_acc))
                print("=== Length: {} ===".format(query_length_dist))

                top_10_exemplars = exemplar_count.most_common(20)
                print("=== Top 10 Most Frequently Selected Exemplars ===")
                for idx, count in top_10_exemplars:
                    print(f"Exemplar Index: {idx}, Selection Count: {count}")

                ###
                if this_acc >= best_acc:
                    best_acc = this_acc
                    torch.save(pi_theta.state_dict(), save_model_path)
                    print(f"*** New best model saved at {save_model_path} with validation accuracy: {best_acc:.4f} ***")

                ###
                pi_theta.train()

                print("-- [Validation time this round]: {}".format(time.time() - validating_start_time))
                print("-- [Time elapsed]: {}".format(time.time() - start_time))
        
        else:
            if (ep + 1) % test_per_rounds == 0 or ep in [0, episodes-1]:
                pi_theta.eval()
                #
                testing_start_time = time.time()
                if (ep + 1) % test_per_rounds == 0 or ep in [0, episodes-1]:
                    test_rounds = len(data_loader.processor.test_dataset)
                    test_emb_list = data_loader.test_emb_list
                    correct_count = 0
                    query_length_dist = defaultdict(lambda: 0)
                    exemplar_count = Counter()
                    
                    for test_round in tqdm(range(test_rounds)):
                        t_idx = test_round % len(test_emb_list)
                        s0 = test_emb_list[t_idx]
                        t_idx = [t_idx]
                        
                        exemplar_indices = generate_trajectory_testing(
                            pi_model = pi_theta,
                            s0 = s0,
                            horizon = T,
                            exemplars = exemplars,
                            gamma = args.gamma,
                            trunc_trajectory_flag=args.active_stop_for_testing
                        )
                        query_length_dist[len(exemplar_indices)] += 1

                        correct_count += data_loader.test_exemplars_with_testing_data(exemplar_indices, test_indices=t_idx)
                        exemplar_count.update(exemplar_indices)

                    print("=== Episode: {}. Test result: {} ===".format(ep, correct_count / test_rounds))
                    print("=== Length: {} ===".format(query_length_dist))

                    top_10_exemplars = exemplar_count.most_common(20)
                    print("=== Top 10 Most Frequently Selected Exemplars ===")
                    for idx, count in top_10_exemplars:
                        print(f"Exemplar Index: {idx}, Selection Count: {count}")
            
                ###
                pi_theta.train()
                print("-- [Time elapsed]: {}".format(time.time() - start_time))
                print("-- [Testing time this round]: {}".format(time.time() - testing_start_time))
        
        pi_theta.train()

    ################################################################################################################
    ################################################ Final Validating ##############################################
    if args.use_true_valid_and_testing:
        pi_theta.eval()
        correct_count = 0
        #
        validating_start_time = time.time()
        with torch.no_grad():
            valid_rounds = len(data_loader.processor.val_dataset)
            valid_emb_list = data_loader.query_emb_list
            query_length_dist = defaultdict(lambda: 0)
            exemplar_count = Counter()
            
            for test_round in tqdm(range(valid_rounds)):
                v_idx = test_round % len(valid_emb_list)
                s0 = valid_emb_list[v_idx].detach().clone()
                v_idx = [v_idx]
                
                exemplar_indices = generate_trajectory_testing(
                    pi_model = pi_theta,
                    s0 = s0,
                    horizon = T,
                    exemplars = exemplars,
                    gamma = args.gamma,
                    trunc_trajectory_flag=args.active_stop_for_testing
                )
                query_length_dist[len(exemplar_indices)] += 1

                correct_count += data_loader.valid_exemplars_with_training_data(exemplar_indices, valid_indices=v_idx)
                exemplar_count.update(exemplar_indices)  # Add selected indices to the counter

        this_acc = correct_count / valid_rounds
        print("=== Episode: {}. Validation result: {} ===".format(ep, this_acc))
        print("=== Length: {} ===".format(query_length_dist))

        top_10_exemplars = exemplar_count.most_common(20)
        print("=== Top 10 Most Frequently Selected Exemplars ===")
        for idx, count in top_10_exemplars:
            print(f"Exemplar Index: {idx}, Selection Count: {count}")

        ###
        if this_acc >= best_acc:
            best_acc = this_acc
            torch.save(pi_theta.state_dict(), save_model_path)
            print(f"*** New best model saved at {save_model_path} with validation accuracy: {best_acc:.4f} ***")

        ###
        pi_theta.train()

        print("-- [Validation time this round]: {}".format(time.time() - validating_start_time))
        print("-- [Time elapsed]: {}".format(time.time() - start_time))
    
    else:
        pi_theta.eval()
        #
        testing_start_time = time.time()
        if (ep + 1) % test_per_rounds == 0 or ep in [0, episodes-1]:
            test_rounds = len(data_loader.processor.test_dataset)
            test_emb_list = data_loader.test_emb_list
            correct_count = 0
            query_length_dist = defaultdict(lambda: 0)
            exemplar_count = Counter()
            
            for test_round in tqdm(range(test_rounds)):
                t_idx = test_round % len(test_emb_list)
                s0 = test_emb_list[t_idx]
                t_idx = [t_idx]
                
                exemplar_indices = generate_trajectory_testing(
                    pi_model = pi_theta,
                    s0 = s0,
                    horizon = T,
                    exemplars = exemplars,
                    gamma = args.gamma,
                    trunc_trajectory_flag=args.active_stop_for_testing
                )
                #
                query_length_dist[len(exemplar_indices)] += 1

                correct_count += data_loader.test_exemplars_with_testing_data(exemplar_indices, test_indices=t_idx)
                exemplar_count.update(exemplar_indices)  # Add selected indices to the counter

            print("=== Episode: {}. Test result: {} ===".format(ep, correct_count / test_rounds))
            print("=== Length: {} ===".format(query_length_dist))

            top_10_exemplars = exemplar_count.most_common(20)
            print("=== Top 10 Most Frequently Selected Exemplars ===")
            for idx, count in top_10_exemplars:
                print(f"Exemplar Index: {idx}, Selection Count: {count}")
    
        ###
        pi_theta.train()
        print("-- [Time elapsed]: {}".format(time.time() - start_time))
        print("-- [Testing time this round]: {}".format(time.time() - testing_start_time))

    #########################################################################
    ################################ Final Testing ##########################
    if args.use_true_valid_and_testing:
        pi_theta.load_state_dict(torch.load(save_model_path, map_location=device))
        pi_theta.eval()
        #
        testing_start_time = time.time()
        test_results_data = {}

        with torch.no_grad():
            test_rounds = len(data_loader.processor.test_dataset)
            test_emb_list = data_loader.test_emb_list
            correct_count = 0
            query_length_dist = defaultdict(lambda: 0)
            exemplar_count = Counter()
            
            for test_round in tqdm(range(test_rounds)):
                t_idx_orig = test_round % len(data_loader.processor.test_dataset)
                s0 = test_emb_list[t_idx_orig]
                t_idx = [t_idx_orig]
                
                exemplar_indices = generate_trajectory_testing(
                    pi_model = pi_theta,
                    s0 = s0,
                    horizon = T,
                    exemplars = exemplars,
                    gamma = args.gamma,
                    trunc_trajectory_flag=args.active_stop_for_testing
                )
                query_length_dist[len(exemplar_indices)] += 1

                correct_count += data_loader.test_exemplars_with_testing_data(exemplar_indices, test_indices=t_idx)
                exemplar_count.update(exemplar_indices) 

                test_results_data[t_idx_orig] = exemplar_indices 

            #
            final_result = correct_count / test_rounds
            print("=== Episode: {}. Final Test result: {} ===".format(ep, final_result))
            print("=== Length: {} ===".format(query_length_dist))

            top_10_exemplars = exemplar_count.most_common(20)
            print("=== Top 10 Most Frequently Selected Exemplars ===")
            for idx, count in top_10_exemplars:
                print(f"Exemplar Index: {idx}, Selection Count: {count}")

            save_final_policy_model(args, pi_theta, final_test_score=final_result, 
                                    test_results_data=test_results_data)
            
        print("-- [Testing time]: {}".format(time.time() - testing_start_time))
    print("-- [Time elapsed]: {}".format(time.time() - start_time))


if __name__=="__main__":
    main()


