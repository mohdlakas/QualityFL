#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for SCAFFOLD Federated Learning
"""

import os
import copy
import sys
import time
import numpy as np
import torch
import sys
from tqdm import tqdm

sys.path.append('../')
from options import args_parser
from update import test_inference
from models import CNNCifar, CNNMnist
from utils_dir import get_dataset, exp_details
from scaffold import (SCAFFOLDLocalUpdate, initialize_control_variates,
                     aggregate_model_updates, aggregate_control_updates,
                     update_global_model, update_control_variates)


def run_scaffold(args):
    """
    Main function to run SCAFFOLD federated learning
    """
    start_time = time.time()
    
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    # Load dataset and split users
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # Build model
    if args.model == 'cnn':
        if args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Set model to device
    global_model.to(device)
    global_model.train()
    
    # Initialize control variates
    print("Initializing SCAFFOLD control variates...")
    c_global = initialize_control_variates(global_model)
    c_local_dict = {}
    
    # Move control variates to device
    for name in c_global.keys():
        c_global[name] = c_global[name].to(device)
    
    # Initialize local control variates for all clients
    for idx in range(args.num_users):
        c_local_dict[idx] = copy.deepcopy(c_global)
    
    # Training metrics
    train_loss, train_accuracy = [], []
    test_accuracy = []
    print_every = 1
    
    print("Starting SCAFFOLD training...")
    for epoch in tqdm(range(args.epochs)):
        # Client sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # Collect updates
        delta_models = []
        delta_controls = []
        local_losses = []
        
        # Client updates
        for idx in idxs_users:
            # Create local update object
            local_model = SCAFFOLDLocalUpdate(args=args, dataset=train_dataset,
                                            idxs=user_groups[idx], logger=None)
            
            # Get local updates
            delta_model, delta_control, loss = local_model.update_weights_scaffold(
                model=copy.deepcopy(global_model),
                c_global=c_global,
                c_local=c_local_dict[idx],
                global_round=epoch
            )
            
            # Collect updates
            delta_models.append(delta_model)
            delta_controls.append(delta_control)
            local_losses.append(loss)
            
            # Update local control variate: c_i = c_i + Δc_i
            with torch.no_grad():
                for name in c_local_dict[idx].keys():
                    c_local_dict[idx][name].add_(delta_control[name])
        
        # Aggregate model updates
        aggregated_delta_model = aggregate_model_updates(delta_models)
        
        # Aggregate control updates
        aggregated_delta_control = aggregate_control_updates(delta_controls, args.num_users)
        
        with torch.no_grad():
            for name, param in global_model.named_parameters():
                param.data.add_(aggregated_delta_model[name], alpha=args.scaffold_stepsize)
        
        # Update global control: c = c + aggregated_delta_c
        update_control_variates(c_global, aggregated_delta_control)
        
        # Calculate average loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        # Calculate accuracy
        if (epoch + 1) % print_every == 0:
            # Test accuracy
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            test_accuracy.append(test_acc)
            
            print(f'\nGlobal Round: {epoch+1}')
            print(f'Average Local Training Loss: {loss_avg:.3f}')
            print(f'Test Accuracy: {100*test_acc:.2f}%')
            
            # Check control variate magnitudes for monitoring
            c_global_norm = 0
            for name, c_val in c_global.items():
                c_global_norm += torch.norm(c_val).item()**2
            c_global_norm = np.sqrt(c_global_norm)
            print(f'Global Control Variate Norm: {c_global_norm:.4f}')
    
    # Final test accuracy
    test_acc_final, test_loss_final = test_inference(args, global_model, test_dataset)
    
    print(f'\n\nResults after {args.epochs} global rounds:')
    print(f'Test Accuracy: {100*test_acc_final:.2f}%')
    print(f'Test Loss: {test_loss_final:.3f}')
    print(f'\nTotal Runtime: {time.time()-start_time:.0f}s')
    
    # Save results
    results = {
        'train_loss': train_loss,
        'test_accuracy': test_accuracy,
        'test_acc_final': test_acc_final,
        'runtime': time.time() - start_time
    }
    
    return global_model, results


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    
    exp_details(args)
    
    # Run SCAFFOLD
    model, results = run_scaffold(args)
    # ADD THIS PLOTTING CODE:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['train_loss'], 'b-', linewidth=2)
    plt.title('SCAFFOLD Training Loss')
    plt.xlabel('Global Rounds')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([acc * 100 for acc in results['test_accuracy']], 'r-', linewidth=2, marker='o')
    plt.title('SCAFFOLD Test Accuracy')
    plt.xlabel('Global Rounds')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'scaffold_results_{args.dataset}.png')
    plt.show()
    
    print(f"Final Results: {results['test_acc_final']*100:.2f}% accuracy")

    # Save model if needed
    if hasattr(args, 'save_model') and args.save_model:
        torch.save(model.state_dict(), f'./save/scaffold_{args.dataset}_{args.model}_epochs{args.epochs}.pth')