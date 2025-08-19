#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCAFFOLD Federated Learning Implementation
Fixed version that integrates with your existing codebase
"""

import os
import copy
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import sys

sys.path.append('../')
from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNCifar
from utils_dir import get_dataset, exp_details, check_gpu_pytorch, ComprehensiveAnalyzer, write_comprehensive_analysis


class SCAFFOLDLocalUpdate(LocalUpdate):
    """
    SCAFFOLD local update class that extends your existing LocalUpdate
    """
    def __init__(self, args, dataset, idxs, logger):
        super().__init__(args, dataset, idxs, logger)
        
    def update_weights_scaffold(self, model, global_round, c_global, c_local):
        """
        SCAFFOLD local update with control variates
        """
        # Set device from model if not already set
        if self.device is None:
            self.device = next(model.parameters()).device
            self.criterion = self.criterion.to(self.device)
        
        # Set model to train mode
        model.train()
        
        # Store initial model weights for delta calculation
        w_old = copy.deepcopy(model.state_dict())
        
        # Initialize optimizer
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                      momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                       weight_decay=1e-4)
        
        epoch_loss = []
        
        # Local training with SCAFFOLD correction
        for iter in range(self.args.local_ep):
            batch_loss = []
            
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                
                # SCAFFOLD: Apply control variate correction to gradients
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in c_global and name in c_local:
                            # Control variate correction: g_i - c_i + c
                            correction = c_global[name] - c_local[name]
                            param.grad.data += correction
                
                optimizer.step()
                
                # Track loss
                if not torch.isnan(loss) and not torch.isinf(loss):
                    batch_loss.append(loss.item())
                else:
                    print(f"WARNING: Invalid loss detected: {loss.item()}")
                    batch_loss.append(2.0)  # Fallback
                
            # Calculate epoch loss
            if batch_loss:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            else:
                epoch_loss.append(2.0)  # Fallback
        
        # Calculate final loss
        final_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 2.0
        
        # Get updated weights
        w_new = model.state_dict()
        
        # Calculate control variate update (c_i^+ = c_i - c + (y - x) / (K * η))
        # where y = w_new, x = w_old, K = local_ep, η = lr
        c_delta = {}
        for name in w_new.keys():
            if name in c_global and name in c_local:
                # SCAFFOLD control variate update formula
                weight_diff = w_old[name] - w_new[name]  # x - y (note the order)
                c_delta[name] = weight_diff / (self.args.local_ep * self.args.lr) - c_global[name]
            else:
                print(f"WARNING: Missing control variate for {name}")
                c_delta[name] = torch.zeros_like(w_new[name])
        
        return w_new, final_loss, c_delta


def initialize_control_variates(model):
    """
    Initialize control variates for SCAFFOLD
    Start with zeros as per the original SCAFFOLD paper
    """
    c = {}
    for name, param in model.named_parameters():
        c[name] = torch.zeros_like(param.data)
    return c


def scaffold_aggregate_weights(local_weights):
    """
    Aggregate local model weights (standard FedAvg aggregation)
    """
    if not local_weights:
        return {}
    
    # Average all local weights
    global_weights = copy.deepcopy(local_weights[0])
    
    for key in global_weights.keys():
        weight_tensors = [local_weights[i][key] for i in range(len(local_weights))]
        global_weights[key] = torch.stack(weight_tensors, 0).mean(0)
    
    return global_weights


def scaffold_aggregate_control_variates(c_deltas, num_users):
    """
    Aggregate control variate updates according to SCAFFOLD
    c^+ = c + (1/N) * Σ(c_i^+ - c_i)
    """
    if not c_deltas:
        return {}
    
    # Initialize aggregated control variate update
    c_global_delta = {}
    
    # Average all control variate deltas
    for name in c_deltas[0].keys():
        delta_tensors = [c_delta[name] for c_delta in c_deltas]
        # Average the deltas and scale by participation rate
        c_global_delta[name] = torch.stack(delta_tensors, 0).mean(0) * (len(c_deltas) / num_users)
    
    return c_global_delta


def run_scaffold_experiment(args):
    """
    Main SCAFFOLD federated learning experiment
    """
    start_time = time.time()
    
    # Set device
    device = check_gpu_pytorch()
    
    # Get dataset
    print("Loading dataset...")
    train_dataset, test_dataset, user_groups = get_dataset(args)
    
    # Initialize model
    print("Initializing model...")
    if args.model == 'cnn':
        if args.dataset == 'cifar100':
            args.num_classes = 100
        elif args.dataset in ['cifar', 'cifar10']:
            args.num_classes = 10
        else:
            raise ValueError(f'Unsupported dataset: {args.dataset}')
        
        global_model = CNNCifar(args)
    else:
        raise ValueError('Only CNN model is supported')
    
    global_model.to(device)
    global_model.train()
    
    # Initialize SCAFFOLD control variates
    print("Initializing SCAFFOLD control variates...")
    c_global = initialize_control_variates(global_model)
    c_local = {i: initialize_control_variates(global_model) for i in range(args.num_users)}
    
    # Move control variates to device
    for name in c_global.keys():
        c_global[name] = c_global[name].to(device)
        for i in range(args.num_users):
            c_local[i][name] = c_local[i][name].to(device)
    
    # Initialize tracking
    train_loss, train_accuracy = [], []
    analyzer = ComprehensiveAnalyzer()
    
    print("Starting SCAFFOLD training...")
    print(f"Total rounds: {args.epochs}")
    print(f"Clients per round: {max(int(args.frac * args.num_users), 1)}")
    
    # Training loop
    for epoch in tqdm(range(args.epochs), desc="SCAFFOLD Training"):
        local_weights, local_losses, c_deltas = [], [], []
        
        # Select clients for this round
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        round_start_time = time.time()
        
        # Train selected clients
        for idx in idxs_users:
            # Create local update instance
            local_model = SCAFFOLDLocalUpdate(
                args=args, 
                dataset=train_dataset,
                idxs=user_groups[idx], 
                logger=None
            )
            
            # Perform local update with SCAFFOLD
            w, loss, c_delta = local_model.update_weights_scaffold(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                c_global=c_global,
                c_local=c_local[idx]
            )
            
            # Store results
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            c_deltas.append(c_delta)
            
            # Update local control variate: c_i^+ = c_i + c_delta_i
            for name in c_local[idx].keys():
                c_local[idx][name] = c_local[idx][name] + c_delta[name]
        
        # Aggregate model weights (standard FedAvg)
        global_weights = scaffold_aggregate_weights(local_weights)
        
        # Aggregate control variates (SCAFFOLD specific)
        c_global_delta = scaffold_aggregate_control_variates(c_deltas, args.num_users)
        
        # Update global control variate: c^+ = c + c_global_delta
        for name in c_global.keys():
            c_global[name] = c_global[name] + c_global_delta[name]
        
        # Update global model
        global_model.load_state_dict(global_weights)
        
        # Calculate metrics
        loss_avg = sum(local_losses) / len(local_losses)
        if np.isnan(loss_avg) or np.isinf(loss_avg):
            print(f"WARNING: Invalid loss_avg at round {epoch+1}, using fallback")
            loss_avg = 2.0
        
        train_loss.append(loss_avg)
        
        # Calculate training accuracy on selected clients
        train_acc = calculate_training_accuracy(global_model, train_dataset, user_groups, 
                                              idxs_users, device)
        train_accuracy.append(train_acc)
        
        # Track round time
        round_time = time.time() - round_start_time
        
        # Update analyzer
        analyzer.track_training_accuracy(train_acc)
        analyzer.track_round_time(round_time)
        analyzer.track_client_selection(idxs_users)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            test_acc, _ = test_inference(args, global_model, test_dataset)
            analyzer.track_test_accuracy(test_acc)
            print(f"Round {epoch+1:3d}: Train Acc = {train_acc*100:5.2f}%, "
                  f"Test Acc = {test_acc*100:5.2f}%, Loss = {loss_avg:6.4f}, "
                  f"Time = {round_time:5.2f}s")
        else:
            print(f"Round {epoch+1:3d}: Train Acc = {train_acc*100:5.2f}%, "
                  f"Loss = {loss_avg:6.4f}, Time = {round_time:5.2f}s")
    
    # Final evaluation
    print("\nFinal evaluation...")
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    total_time = time.time() - start_time
    
    print(f"\nSCAFFOLD Training Complete!")
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Total Training Time: {total_time:.2f} seconds")
    
    # Generate comprehensive analysis
    report_filename = f"scaffold_analysis_{args.dataset}_{args.alpha if not args.iid else 'iid'}.txt"
    write_comprehensive_analysis(analyzer, args, test_acc, total_time, report_filename)
    print(f"Detailed analysis saved to: {report_filename}")
    
    return {
        'train_accuracy': train_accuracy,
        'train_loss': train_loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'total_time': total_time,
        'analyzer': analyzer
    }


def calculate_training_accuracy(model, dataset, user_groups, selected_clients, device):
    """
    Calculate training accuracy on selected clients' data
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for client_idx in selected_clients:
            # Get client's data indices
            client_indices = user_groups[client_idx]
            if isinstance(client_indices, (set, list)):
                client_indices = list(client_indices)
            
            # Create data loader for this client
            client_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset, client_indices),
                batch_size=64, shuffle=False
            )
            
            # Evaluate on this client's data
            for images, labels in client_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
    model.train()  # Set back to training mode
    return correct / total if total > 0 else 0.0


if __name__ == '__main__':
    # Parse arguments
    args = args_parser()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Print experiment details
    exp_details(args)
    
    # Run SCAFFOLD experiment
    results = run_scaffold_experiment(args)
    
    print("SCAFFOLD experiment completed successfully!")