import os
import copy
import time
import numpy as np
from tqdm import tqdm
import torch
import sys

sys.path.append('../')
from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNCifar, CNNFemnist, CNNCifar100
from utils_dir import get_dataset, exp_details, check_gpu_pytorch, average_weights

def power_of_choice_selection(args, global_model, train_dataset, user_groups, num_users, fraction, d=10):
    """
    Standard Power-of-Choice client selection based on local loss values.
    
    Args:
        args: Arguments object
        global_model: Current global model
        train_dataset: Training dataset
        user_groups: Dictionary mapping client IDs to data indices
        num_users: Total number of clients
        fraction: Fraction of clients to select
        d: Number of candidates to sample for each selection (power-of-d)
    
    Returns:
        List of selected client indices
    """
    m = max(int(fraction * num_users), 1)  # Number of clients to select
    device = next(global_model.parameters()).device
    
    # Collect all candidates for m selections
    all_candidates = []
    for _ in range(m):
        candidates = np.random.choice(range(num_users), min(d, num_users), replace=False)
        all_candidates.extend(candidates)
    
    # Remove duplicates to avoid computing loss twice for same client
    unique_candidates = list(set(all_candidates))
    
    # Compute local losses for all unique candidates
    candidate_losses = []
    global_model.eval()
    
    for idx in unique_candidates:
        # Use LocalUpdate's inference method to compute loss efficiently
        local_model = LocalUpdate(args=args, dataset=train_dataset, 
                                idxs=user_groups[idx], logger=None)
        
        # Compute loss on this client's training data
        _, loss = local_model.inference(model=global_model, eval_type='train')
        
        # Normalize by number of batches for fair comparison
        avg_loss = loss / len(local_model.trainloader) if len(local_model.trainloader) > 0 else float('inf')
        candidate_losses.append((idx, avg_loss))
    
    # Sort by loss (descending - highest loss first)
    candidate_losses.sort(key=lambda x: x[1], reverse=True)
    
    # Select top m clients with highest losses
    selected_clients = [idx for idx, _ in candidate_losses[:m]]
    
    global_model.train()
    return selected_clients

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    args.d = getattr(args, 'd', 10)
    
    exp_details(args)
    device = check_gpu_pytorch()

    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # Build model
    if args.model == 'cnn':
        if args.dataset == 'cifar' or args.dataset == 'cifar10':
            args.num_classes = 10
            args.num_channels = 3
            global_model = CNNCifar(args)
            print(f"CIFAR-10: Model created with {global_model.fc2.out_features} output classes")
            
        elif args.dataset == 'cifar100':
            args.num_classes = 100
            args.num_channels = 3
            global_model = CNNCifar100(args)  # Use CNNCifar100 if available, else CNNCifar
            print(f"CIFAR-100: Model created with {global_model.fc2.out_features} output classes")
            
        elif args.dataset == 'femnist':
            args.num_classes = 62  # 10 digits + 26 uppercase + 26 lowercase
            args.num_channels = 1
            global_model = CNNFemnist(args)
            print(f"FEMNIST: Model created with {global_model.fc2.out_features} output classes")
            
        else:
            exit(f'Error: unsupported dataset {args.dataset}. Supported: cifar, cifar10, cifar100, femnist')
            
    else:
        exit('Error: only CNN model is supported. Use --model=cnn')
    
    global_model.to(device)
    global_model.train()
    
    train_loss, train_accuracy = [], []
    
    for epoch in tqdm(range(args.epochs)):
        
        local_weights, local_losses = [], []
        
        # Power-of-Choice client selection based on loss
        selected_clients = power_of_choice_selection(
            args, global_model, train_dataset, user_groups, 
            args.num_users, args.frac, args.d
        )
        
        for idx in selected_clients:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=None)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        # Use average_weights from utils_dir for consistency
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # Calculate training accuracy
        list_acc = []
        global_model.eval()
        for c in selected_clients:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[c], logger=None)
            # Use inference method for consistency
            acc, _ = local_model.inference(model=global_model, eval_type='train')
            list_acc.append(acc)
        global_model.train()
        
        train_accuracy.append(np.mean(list_acc))
        
        # Test accuracy
        test_acc, _ = test_inference(args, global_model, test_dataset)
        
        # CRITICAL: This exact format is required for auto_compare.py parsing
        print(f"Round {epoch+1}: Train Accuracy = {train_accuracy[-1]*100:.2f}%, Test Accuracy = {test_acc*100:.2f}%, Loss = {loss_avg:.4f}")
    
    # Final test accuracy - also parsed by auto_compare.py
    test_acc, _ = test_inference(args, global_model, test_dataset)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")