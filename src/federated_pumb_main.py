import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
import matplotlib
import matplotlib.pyplot as plt
import logging
matplotlib.use('Agg')

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNCifar, CNNCifar100, CNNFemnist

from federated_PUMB import PUMBFederatedServer
from utils_dir import (get_dataset, exp_details, plot_data_distribution,
                      ComprehensiveAnalyzer, write_comprehensive_analysis, check_gpu_pytorch, set_seed)

from datetime import datetime

current_dir = os.getcwd()
if 'Algorithms' in current_dir or 'algorithms' in current_dir:
    save_base = '../../save'  # From src/Algorithms to project root
else:
    save_base = '../save'     # From src to project root

# Create all necessary directories
os.makedirs(f'{save_base}/objects', exist_ok=True)
os.makedirs(f'{save_base}/images', exist_ok=True)
os.makedirs(f'{save_base}/logs', exist_ok=True)

def initialize_cifar100_settings(args):
    """Initialize settings specifically for CIFAR-100"""
    if args.dataset == 'cifar100':
        # Force CIFAR-100 settings
        args.num_classes = 100
        args.num_channels = 3
        
        # Optimize hyperparameters for CIFAR-100
        if args.local_bs > 32:
            args.local_bs = min(32, args.local_bs)
            print(f"Reduced batch size to {args.local_bs} for CIFAR-100")
        
        if args.lr > 0.001:
            args.lr = 0.001
            print(f"Reduced learning rate to {args.lr} for CIFAR-100")
        
        # Use Adam optimizer for better convergence
        args.optimizer = 'adam'
        
        print(f"CIFAR-100 Configuration:")
        print(f"  - Classes: {args.num_classes}")
        print(f"  - Batch size: {args.local_bs}")
        print(f"  - Learning rate: {args.lr}")
        print(f"  - Optimizer: {args.optimizer}")
        
    return args


    
if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    
    # Set random seed FIRST, before any other operations
    if hasattr(args, 'seed') and args.seed is not None:
        set_seed(args.seed)
    
    args = initialize_cifar100_settings(args)

    device = check_gpu_pytorch()

    # Load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # Plot client data distribution (IID/non-IID)
    plot_data_distribution(
        user_groups, train_dataset,
        save_path=f'{save_base}/images/data_distribution_{args.dataset}_iid[{args.iid}]_alpha[{getattr(args, "alpha", "NA")}].png',
        title="Client Data Distribution (IID={})".format(args.iid)
    )
    
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
            global_model = CNNCifar100(args)  # Same architecture, different output size
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

    #optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr)
    if hasattr(args, 'optimizer') and args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr, weight_decay=1e-4)
        print(f"PUMB Server using Adam optimizer with lr={args.lr}")
    else:
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr)
        print(f"PUMB Server using SGD optimizer with lr={args.lr}")
    #loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.NLLLoss()
    # Initialize PUMB server
    server = PUMBFederatedServer(global_model, optimizer, loss_fn, args, embedding_dim=512)

    # NEW: Store quality metric info in args for dynamic exp_details
    quality_metric = server.quality_calc
    args.quality_metric_type = type(quality_metric).__name__
    args.quality_alpha = quality_metric.alpha
    args.quality_beta = quality_metric.beta
    args.quality_gamma = quality_metric.gamma
    
    # ADD: Store baseline quality if it exists
    if hasattr(quality_metric, 'baseline_quality'):
        args.quality_baseline = quality_metric.baseline_quality

    # Now call exp_details with enhanced info
    exp_details(args)

    # Initialize comprehensive analyzer for detailed metrics
    analyzer = ComprehensiveAnalyzer()
    
    # FIX: Initialize tracking lists for analyzer
    analyzer.similarity_scores = []
    analyzer.client_ranking_history = []
    analyzer.quality_scores_history = []
    
    train_loss, train_accuracy = [], []
    test_accuracy_history = []


    for epoch in tqdm(range(args.epochs)):
        round_start_time = time.time()  # Track round time
        
        # 🆕 TRACK ACTUAL VALUES DURING FIRST ROUND
        if epoch == 0:
            # Get actual values from the server components
            actual_exploration_ratio = server.client_selector.exploration_ratio
            actual_initial_rounds = server.client_selector.initial_rounds
            actual_quality_type = type(server.quality_calc).__name__
            
            # Track in analyzer
            analyzer.track_actual_config(
                pumb_exploration_ratio=actual_exploration_ratio,
                pumb_initial_rounds=actual_initial_rounds,
                quality_metric_type=actual_quality_type,
                embedding_dim=512,
                memory_bank_size_limit=getattr(server.memory_bank, 'max_size', 'Unlimited'),
                quality_alpha=server.quality_calc.alpha,
                quality_beta=server.quality_calc.beta,
                quality_gamma=server.quality_calc.gamma
            )
            
            print(f"🎯 CONFIRMED ACTUAL VALUES BEING USED:")
            print(f"   Exploration Ratio: {actual_exploration_ratio}")
            print(f"   Initial Rounds: {actual_initial_rounds}")
            print(f"   Quality Metric: {actual_quality_type}")
            print(f"   Quality Alpha (α): {server.quality_calc.alpha}")
            print(f"   Quality Beta (β): {server.quality_calc.beta}")
            print(f"   Quality Gamma (γ): {server.quality_calc.gamma}")
            print(f"   Memory Bank Max Size: {getattr(server.memory_bank, 'max_size', 'Unlimited')}")

        # Store previous model state BEFORE updates (important!)
        if epoch > 0:
            server.prev_model_state = server._get_model_state_copy()

        # Intelligent Client Selection
        m = max(int(args.frac * args.num_users), 1)
        available_clients = list(range(args.num_users))
        selected_clients = server.select_clients(available_clients, m)

        client_models = {}
        client_losses = {}
        data_sizes = {}
        #param_updates = {}
        training_losses = {}
        client_embeddings = {}  # NEW: Store pre-computed embeddings
        all_loss_improvements = []

        # # Collect all client data
        # for idx in selected_clients:
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                             idxs=user_groups[idx], logger=None)
            
        #     initial_state = copy.deepcopy(global_model.state_dict())
        #     loss_before = local_model.inference(model=global_model)[1]
            
        #     updated_weights, loss = local_model.update_weights(
        #         model=copy.deepcopy(global_model), global_round=epoch)

        #     temp_model = copy.deepcopy(global_model)
        #     temp_model.load_state_dict(updated_weights)
        #     loss_after = local_model.inference(model=temp_model)[1]

        #     param_update = {name: updated_weights[name] - initial_state[name]
        #                     for name in updated_weights}

        #     client_models[idx] = updated_weights
        #     client_losses[idx] = (loss_before, loss_after)
        #     data_sizes[idx] = len(user_groups[idx])
        #     param_updates[idx] = param_update

        #     loss_improvement = max(0, loss_before - loss_after)
        #     all_loss_improvements.append(loss_improvement)

        # Collect all client data with efficient approach
        for idx in selected_clients:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[idx], logger=None,
                                    embedding_gen=server.embedding_gen)  # Pass embedding generator
            
            # Get loss before training
            loss_before = local_model.inference(model=global_model)[1]
            
            # EFFICIENT: Single model copy for training
            model_copy = copy.deepcopy(global_model)
            update_result = local_model.update_weights_efficient(
                model=model_copy, global_round=epoch)

            # EFFICIENT: Use the already trained model for loss_after (no extra copy!)
            loss_after = local_model.inference(model=model_copy)[1]

            # Store results efficiently
            client_models[idx] = update_result['model_state']
            client_losses[idx] = (loss_before, loss_after)
            data_sizes[idx] = update_result['data_size']
            
            training_losses[idx] = update_result['avg_loss']  # This is the training loss from update_weights_efficient
            
            # NEW: Store pre-computed embedding or stats
            if update_result['embedding'] is not None:
                client_embeddings[idx] = update_result['embedding']
            elif update_result['update_stats'] is not None:
                # Generate embedding from stats server-side (fallback)
                client_embeddings[idx] = server.embedding_gen.generate_embedding_from_stats(
                    update_result['update_stats'])

            loss_improvement = max(0, loss_before - loss_after)
            all_loss_improvements.append(loss_improvement)


        # FIX: Track quality scores for analysis
        client_qualities = {}
        
        # Update memory bank with all client data
        # for idx in selected_clients:
        #     loss_before, loss_after = client_losses[idx]
            
        #     quality = server.quality_calc.calculate_quality(
        #         loss_before, loss_after, data_sizes, param_updates[idx],
        #         epoch, idx, all_loss_improvements
        #     )
            
        #     # FIX: Store quality scores for analysis
        #     client_qualities[idx] = quality
            
        #     embedding = server.embedding_gen.generate_embedding(param_updates[idx])
        #     server.memory_bank.add_update(idx, embedding, quality, epoch)

        # FIX: Update memory bank section - replace the memory bank update loop
        for idx in selected_clients:
            loss_before, loss_after = client_losses[idx]
            
            # Calculate parameter update for quality calculation
            if epoch > 0 and server.prev_model_state is not None:
                initial_state = server.prev_model_state
            else:
                initial_state = server._get_model_state_copy()
                
            param_update = {name: client_models[idx][name] - initial_state[name]
                           for name in client_models[idx]}
            
            quality = server.quality_calc.calculate_quality(
                loss_before, loss_after, data_sizes, param_update,
                epoch, idx, all_loss_improvements
            )
            
            # Store quality scores for analysis
            client_qualities[idx] = quality
            
            # NEW: Use pre-computed embedding if available, otherwise generate
            if idx in client_embeddings:
                embedding = client_embeddings[idx]
            else:
                embedding = server.embedding_gen.generate_embedding(param_update)
            
            # Use efficient memory bank update
            server.memory_bank.add_update_efficient(idx, embedding, quality, epoch)

        # Store global state BEFORE similarity computation
        current_state = server._get_model_state_copy()
        server.memory_bank.store_global_state(epoch, current_state)
        
        # Update memory bank round count properly - ONLY ONCE per round
        server.memory_bank.update_round_count()
        #print(f"Round {epoch}: Memory bank round_count updated to {server.memory_bank.round_count}")

        # Improved embedding diversity check
        # if epoch >= 5 and server.memory_bank.round_count > 5:
        #     all_embeddings = []
        #     for client_id in selected_clients:
        #         if client_id in server.memory_bank.client_embeddings:
        #             recent_emb = server.memory_bank.client_embeddings[client_id][-1]
        #             all_embeddings.append(recent_emb.flatten())
            
        #     if len(all_embeddings) > 1:
        #         all_embeddings = np.array(all_embeddings)
        #         pairwise_similarities = []
        #         for i in range(len(all_embeddings)):
        #             for j in range(i+1, len(all_embeddings)):
        #                 sim = np.dot(all_embeddings[i], all_embeddings[j]) / (
        #                     np.linalg.norm(all_embeddings[i]) * np.linalg.norm(all_embeddings[j]) + 1e-8
        #                 )
        #                 pairwise_similarities.append(sim)
                
        #         print(f"Round {epoch}: Embedding diversity - mean similarity: {np.mean(pairwise_similarities):.4f}, "
        #               f"std: {np.std(pairwise_similarities):.4f}")

        # Get aggregation weights
        aggregation_weights = server.client_selector.get_aggregation_weights(
            selected_clients, client_models, data_sizes,
            server.global_direction, server.embedding_gen,
            server.quality_calc, epoch
        )

        # FIX: Calculate similarity scores for analysis
        similarities = []
        client_reliabilities = {}
        
        for client_id in selected_clients:
            # Get reliability
            reliability = server.memory_bank.get_client_reliability(client_id)
            client_reliabilities[client_id] = reliability
            
            # # Calculate similarity if we have embeddings
            # if client_id in server.memory_bank.client_embeddings and len(server.memory_bank.client_embeddings[client_id]) > 0:
            #     current_embedding = server.embedding_gen.generate_embedding(param_updates[client_id])
            #     sim = server.memory_bank.compute_similarity(client_id, current_embedding)
            #     similarities.append(sim)

            # Calculate similarity if we have embeddings
            if client_id in server.memory_bank.client_embeddings and len(server.memory_bank.client_embeddings[client_id]) > 0:
                # Use pre-computed embedding if available
                if client_id in client_embeddings:
                    current_embedding = client_embeddings[client_id]
                else:
                    # Calculate parameter update for similarity
                    if epoch > 0:
                        initial_state = server.prev_model_state
                    else:
                        initial_state = server._get_model_state_copy()
                    param_update = {name: client_models[client_id][name] - initial_state[name]
                                   for name in client_models[client_id]}
                    current_embedding = server.embedding_gen.generate_embedding(param_update)
                
                sim = server.memory_bank.compute_similarity(client_id, current_embedding)
                similarities.append(sim)

        # FIX: Store average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        analyzer.similarity_scores.append(avg_similarity)

        # FIX: Track client rankings for stability analysis
        if client_reliabilities:
            rankings = sorted(client_reliabilities.items(), key=lambda x: x[1], reverse=True)
            analyzer.client_ranking_history.append(rankings)

        # FIX: Track quality scores
        analyzer.quality_scores_history.append(client_qualities)

        # Store aggregation weights for analysis
        #analyzer.track_aggregation_weights(epoch, aggregation_weights)
        
        # Debug output for weights
        #print(f"Round {epoch}: Weights = {aggregation_weights}")
        #print(f"Round {epoch}: Weight sum = {sum(aggregation_weights.values())}")
        #print(f"Round {epoch}: Weight std = {np.std(list(aggregation_weights.values()))}")

        # Update global model with weighted aggregation
        server.update_global_model(client_models, client_losses, data_sizes, aggregation_weights)
        
        # Update global direction AFTER model update
        updated_state = server._get_model_state_copy()
        if server.prev_model_state is not None:
            server.global_direction = {
                name: updated_state[name] - server.prev_model_state[name]
                for name in updated_state
            }
        server.prev_model_state = updated_state

        # Evaluation and logging
        #loss_avg = np.mean([loss_after for _, loss_after in client_losses.values()])
        loss_avg = np.mean(list(training_losses.values()))  # Use training loss for fair comparison

        train_loss.append(loss_avg)

        # ADD THIS: Print test accuracy every round for auto_compare parsing
        # test_acc_current, _ = test_inference(args, global_model, test_dataset)
        # print(f"Round {epoch+1}: Test Accuracy = {test_acc_current*100:.2f}%")
        
        # FIX: Calculate round time properly
        round_time = time.time() - round_start_time
        
        # FIX: Comprehensive tracking for analysis
        analyzer.track_round_time(round_time)
        analyzer.track_client_selection(selected_clients)
        analyzer.track_memory_bank_size(len(server.memory_bank.memories))
        
        # Track client reliability improvements
        for client_id in selected_clients:
            reliability = server.memory_bank.get_client_reliability(client_id)
            analyzer.track_client_reliability(client_id, reliability)

        # Compute training accuracy (FIX: Only evaluate selected clients for fair comparison)
        list_acc = []
        
        # STANDARDIZED EVALUATION METHOD (same as FedAvg)
        global_model.eval()
        for c in selected_clients:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                    idxs=user_groups[c], logger=None)
            # Use same evaluation method as FedAvg for fair comparison
            acc, loss = local_model.inference(model=global_model, eval_type='train')
            list_acc.append(acc)
        
        # ORIGINAL MANUAL EVALUATION METHOD (commented out for testing)
        # for c in selected_clients:  # Only selected clients
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                             idxs=user_groups[c], logger=None)
        #     
        #     # FIXED: Evaluate on training data instead of test data
        #     correct, total = 0, 0
        #     loss_sum = 0
        #     #criterion = torch.nn.CrossEntropyLoss().to(device)
        #     criterion = torch.nn.NLLLoss().to(device)

        #     with torch.no_grad():
        #         for images, labels in local_model.trainloader:  # ← Use trainloader!
        #             images, labels = images.to(device), labels.to(device)
        #             outputs = global_model(images)
        #             
        #             _, predicted = torch.max(outputs, 1)
        #             total += labels.size(0)
        #             correct += (predicted == labels).sum().item()
        #     
        #     acc = correct / total if total > 0 else 0
        #     list_acc.append(acc)



        train_accuracy.append(np.mean(list_acc))
        test_acc_current, _ = test_inference(args, global_model, test_dataset)
        test_accuracy_history.append(test_acc_current)
        print(f"Round {epoch+1}: Train Accuracy = {train_accuracy[-1]*100:.2f}%, Test Accuracy = {test_acc_current*100:.2f}%, Loss = {loss_avg:.4f}")

        # Track accuracy for analysis
        analyzer.track_training_accuracy(train_accuracy[-1])
        
        # FIX: Comprehensive round data logging
        test_acc_for_round = None
        # if epoch % print_every == 0 or epoch == args.epochs - 1:
        #     test_acc_for_round, _ = test_inference(args, global_model, test_dataset)
        #     test_accuracy_history.append(test_acc_for_round)
        #     analyzer.track_test_accuracy(test_acc_for_round)
        #     print(f"Round {epoch}: Test Accuracy = {test_acc_for_round:.4f}")

        # FIX: Log comprehensive round data
        analyzer.log_round_data(
            round_num=epoch,
            train_acc=train_accuracy[-1],
            train_loss=train_loss[-1],
            test_acc=test_acc_for_round,
            selected_clients=selected_clients,
            aggregation_weights=aggregation_weights,
            client_reliabilities=client_reliabilities,
            client_qualities=client_qualities,
            memory_bank_size=len(server.memory_bank.memories),
            avg_similarity=avg_similarity,
            round_time=round_time
        )

    # Final evaluation
    test_acc, _ = test_inference(args, global_model, test_dataset)
    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("Final Test Accuracy: {:.2f}%".format(100*test_acc))

    total_time = time.time() - start_time

    # Save train_loss, train_accuracy, and test_accuracy_history
    file_name = f'{save_base}/objects/PUMB_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_E[{args.local_ep}]_B[{args.local_bs}].pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy, test_accuracy_history], f)

    print('\n Total Run Time: {0:0.4f}'.format(total_time))

    # Enhanced plotting with both training and test accuracy (matching federated_main.py style)
    plt.figure(figsize=(12, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Accuracy comparison
    ax1.plot(range(len(train_accuracy)), [100*acc for acc in train_accuracy], 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(range(len(test_accuracy_history)), [100*acc for acc in test_accuracy_history], 'r-', label='Test Accuracy', linewidth=2)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(f'PUMB Performance - Final Test Accuracy: {test_acc*100:.2f}%')
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
    plot_filename = f'{save_base}/images/PUMB_{args.dataset}_{args.model}_{args.epochs}_C[{args.frac}]_iid[{args.iid}]_alpha[{getattr(args, "alpha", "NA")}]_E[{args.local_ep}]_B[{args.local_bs}]_explr[{getattr(args, "pumb_exploration_ratio", "NA")}]_lr[{getattr(args, "lr", "NA")}]_initR[{getattr(args, "pumb_initial_rounds", "NA")}]_loss_acc_{timestamp}.png'

    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Generate comprehensive analysis report
    comprehensive_filename = f"{save_base}/logs/comprehensive_analysis_{timestamp}.txt"
    write_comprehensive_analysis(analyzer, args, test_acc, total_time, comprehensive_filename, getattr(args, 'seed', None))

    print(f"\n✅ Comprehensive analysis saved to: {comprehensive_filename}")
    print(f"📊 Basic experiment summary saved to: experiment_summary_{timestamp}.txt")
    
    # Print key metrics to console for immediate feedback
    convergence_metrics = analyzer.calculate_convergence_metrics()
    client_analysis = analyzer.analyze_client_selection_quality()
    memory_analysis = analyzer.analyze_memory_bank_effectiveness()
    
    print(f"\n🔍 QUICK RESULTS SUMMARY:")
    print(f"   Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Convergence Speed: {convergence_metrics.get('convergence_round', 'N/A')} rounds")
    print(f"   Training Stability: {convergence_metrics.get('training_stability', 0):.6f}")
    print(f"   Unique Clients Selected: {client_analysis.get('total_unique_clients', 0)}")
    print(f"   Memory Bank Final Size: {memory_analysis.get('final_memory_size', 0)}")
    print(f"   Avg Client Reliability Improvement: {client_analysis.get('avg_reliability_improvement', 0):.4f}")
    print(f"\n📁 All results saved in ../save/logs/ directory")

    # After final evaluation and analysis
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
