
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random
import torch

from tensorboardX import SummaryWriter
from datetime import datetime

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNFemnist, CNNCifar100
from utils_dir import (get_dataset, exp_details, average_weights, plot_data_distribution,
                      ComprehensiveAnalyzer, write_fedavg_comprehensive_analysis, check_gpu_pytorch, set_seed)

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
   
def initialize_cifar100_settings(args):
    """Initialize settings specifically for CIFAR-100 based on PUMB analysis"""
    if args.dataset == 'cifar100':
        # Force CIFAR-100 settings
        args.num_classes = 100
        args.num_channels = 3

        # Use Adam optimizer for better convergence (PUMB used this successfully)
        args.optimizer = 'adam'
        
        # # Add learning rate decay for better convergence
        # args.lr_decay = False
        # args.decay_rate = 0.98
        # args.min_lr = 0.0001
        
        print(f"CIFAR-100 Configuration (PUMB-optimized):")
        print(f"  - Classes: {args.num_classes}")
        print(f"  - Batch size: {args.local_bs}")
        print(f"  - Learning rate: {args.lr} with decay")
        print(f"  - Optimizer: {args.optimizer}")
        
    return args



if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    
    # Set random seed FIRST, before any other operations
    if hasattr(args, 'seed') and args.seed is not None:
        set_seed(args.seed)
    
    args = initialize_cifar100_settings(args)

    # Create save directories based on current working directory
    current_dir = os.getcwd()
    if 'Algorithms' in current_dir or 'algorithms' in current_dir:
        save_base = '../../save'
    else:
        save_base = '../save'
    
    # Create all necessary directories
    os.makedirs(f'{save_base}/objects', exist_ok=True)
    os.makedirs(f'{save_base}/images', exist_ok=True)
    os.makedirs(f'{save_base}/logs', exist_ok=True)

    # Check and set device
    device = check_gpu_pytorch()
    
    # load dataset and user groups FIRST (this sets args.num_classes correctly)
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
            global_model = CNNCifar100(args)
            print(f"CIFAR-100: Model created with {global_model.fc3.out_features} output classes")
            
        elif args.dataset == 'femnist':
            args.num_classes = 62
            args.num_channels = 1
            global_model = CNNFemnist(args)
            print(f"FEMNIST: Model created with {global_model.fc2.out_features} output classes")
            
        else:
            exit(f'Error: unsupported dataset {args.dataset}. Supported: cifar, cifar10, cifar100, femnist')
            
    else:
        exit('Error: only CNN model is supported. Use --model=cnn')

    # NOW show experiment details (after dataset loading and model creation)
    exp_details(args)

    plot_data_distribution(
        user_groups, train_dataset,
        save_path=f'{save_base}/images/data_distribution_{args.dataset}_iid[{args.iid}]_alpha[{getattr(args, "alpha", "NA")}].png',
        title="Client Data Distribution (IID={})".format(args.iid)
    )

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # Check data loading
    print(f"\nDataset info:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of user groups: {len(user_groups)}")
    if len(user_groups) > 0:
        first_user_data = user_groups[0]
        print(f"First user has {len(first_user_data)} samples")
        
    # copy weights
    global_weights = global_model.state_dict()

    # Initialize comprehensive analyzer for detailed metrics
    analyzer = ComprehensiveAnalyzer()

    if hasattr(args, 'optimizer') and args.optimizer.lower() == 'adam':
        print(f"Using Adam optimizer with lr={args.lr}")
    else:
        print(f"Using SGD optimizer with lr={args.lr}")

    # Training
    train_loss, train_accuracy = [], []
    test_accuracy_history = []  # Track test accuracy history
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        round_start_time = time.time()
        
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # Track selected clients and their data
        selected_clients = list(idxs_users)
        client_losses = {}
        data_sizes = {}

        # Apply learning rate decay if enabled
        current_lr = args.lr
        if hasattr(args, 'lr_decay') and args.lr_decay:
            decay_rate = getattr(args, 'decay_rate', 0.98)
            min_lr = getattr(args, 'min_lr', 0.0001)
            current_lr = max(min_lr, args.lr * (decay_rate ** epoch))
            
            # Update args.lr for LocalUpdate to use
            args.lr = current_lr
            
            if epoch % 20 == 0:
                print(f"Round {epoch+1}: Learning rate = {current_lr:.6f}")

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=logger)
            
            # Get loss before training
            loss_before, _ = local_model.inference(model=copy.deepcopy(global_model), eval_type='train')
            
            # Train the local model
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            
            # FIX: Get loss after training using updated model
            updated_model = copy.deepcopy(global_model)
            updated_model.load_state_dict(w)
            loss_after, _ = local_model.inference(model=updated_model, eval_type='train')
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            
            # Track client-specific data with corrected values
            client_losses[idx] = (loss_before, loss_after)
            data_sizes[idx] = len(user_groups[idx])

        # Update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # FIX: Calculate training accuracy on TRAINING data using updated global model
        list_acc, list_loss = [], []
        global_model.eval()
        
        for c in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[c], logger=logger)
            # FIX: Use eval_type='train' to evaluate on training data
            acc, loss = local_model.inference(model=global_model, eval_type='train')
            list_acc.append(acc)
            list_loss.append(loss)
            
        train_accuracy.append(sum(list_acc)/len(list_acc))
        
        # Calculate test accuracy every round
        test_acc_current, test_loss_current = test_inference(args, global_model, test_dataset)
        test_accuracy_history.append(test_acc_current)
        
        # print global training loss and accuracy after every round
        if (epoch+1) % print_every == 0:
            print(f'Round {epoch+1}: Train Accuracy = {100*train_accuracy[-1]:.2f}%, Test Accuracy = {100*test_acc_current:.2f}%, Loss = {loss_avg:.4f}')
        
        # Collect data for comprehensive analysis
        round_time = time.time() - round_start_time
        
        # Calculate aggregation weights (uniform for FedAvg)
        aggregation_weights = {client_id: 1.0/len(selected_clients) for client_id in selected_clients}
        
        # Calculate client reliabilities based on loss improvement
        client_reliabilities = {}
        for client_id in selected_clients:
            loss_before, loss_after = client_losses[client_id]
            loss_improvement = max(0, loss_before - loss_after)
            # Normalize by data size for reliability metric
            reliability = (loss_improvement + 1e-6) * data_sizes[client_id] / 1000.0
            client_reliabilities[client_id] = min(1.0, reliability)
        
        # Log data to analyzer
        analyzer.log_round_data(
            round_num=epoch + 1,
            train_acc=train_accuracy[-1],
            train_loss=loss_avg,
            test_acc=test_acc_current,
            selected_clients=selected_clients,
            aggregation_weights=aggregation_weights,
            client_reliabilities=client_reliabilities,
            client_qualities=None,
            memory_bank_size=None,
            avg_similarity=None,
            round_time=round_time
        )

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = f'{save_base}/objects/{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}].pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy, test_accuracy_history], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # Enhanced plotting with both training and test accuracy
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Accuracy comparison
    ax1.plot(range(len(train_accuracy)), [100*acc for acc in train_accuracy], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(range(len(test_accuracy_history)), [100*acc for acc in test_accuracy_history], 'r-', label='Test Accuracy', linewidth=2)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'FedAvg Performance - Final Test Accuracy: {test_acc*100:.2f}%')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training loss
    ax2.plot(range(len(train_loss)), train_loss, 'g-', label='Training Loss', linewidth=2)
    ax2.set_ylabel('Training Loss')
    ax2.set_xlabel('Communication Rounds')
    ax2.set_title('Training Loss vs Communication Rounds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f'{save_base}/images/fed_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}]_opt[{getattr(args, "optimizer", "NA")}]_lr[{getattr(args, "lr", "NA")}]_alpha[{getattr(args, "alpha", "NA")}]_{timestamp}.png'
 
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Generate comprehensive analysis report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fedavg_filename = f"{save_base}/logs/fedavg_comprehensive_analysis_{timestamp}.txt"
    write_fedavg_comprehensive_analysis(analyzer, args, test_acc, total_time, fedavg_filename, getattr(args, 'seed', None))

    print(f"\n✅ FedAvg comprehensive analysis saved to: {fedavg_filename}")

    # Print key metrics to console for immediate feedback
    convergence_metrics = analyzer.calculate_convergence_metrics()
    client_analysis = analyzer.analyze_client_selection_quality()
    
    print(f"\n🔍 FEDAVG RESULTS SUMMARY:")
    print(f"   Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Final Training Accuracy: {train_accuracy[-1]:.4f} ({train_accuracy[-1]*100:.2f}%)")
    print(f"   Convergence Speed: {convergence_metrics.get('convergence_round', 'N/A')} rounds")
    print(f"   Training Stability: {convergence_metrics.get('training_stability', 0):.6f}")
    print(f"   Unique Clients Selected: {client_analysis.get('total_unique_clients', 0)}")
    print(f"   Avg Participation Rate: {client_analysis.get('avg_participation_rate', 0):.4f}")
    print(f"   Total Runtime: {total_time:.2f} seconds")
    print(f"\n📁 All results saved in {save_base}/logs/ directory")
