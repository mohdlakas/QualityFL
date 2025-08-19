#!/usr/bin/env python
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob

def load_experiment_results():
    """Load all experiment results from save directory"""
    results = {'FedAvg': {}, 'PUMB': {}}
    alpha_values = [0.01, 0.1, 0.5, 1.0]
    
    save_dir = '../save/objects'
    
    print("🔍 Looking for experiment results...")
    
    for alpha in alpha_values:
        print(f"\n📊 Loading results for α={alpha}")
        
        # Look for FedAvg results (federated_main.py output)
        fedavg_patterns = [
            f'{save_dir}/fedavg_cifar_cnn_100_C[0.1]_iid[0]_E[5]_B[32].pkl',
            f'{save_dir}/FedAvg_cifar_cnn_100_C[0.1]_iid[0]_E[5]_B[32].pkl',
            f'{save_dir}/*fedavg*alpha*{alpha}*.pkl',
            f'{save_dir}/*FedAvg*alpha*{alpha}*.pkl'
        ]
        
        for pattern in fedavg_patterns:
            files = glob.glob(pattern)
            if files:
                try:
                    with open(files[0], 'rb') as f:
                        data = pickle.load(f)
                        if len(data) >= 2:
                            train_loss, train_accuracy = data[0], data[1]
                            results['FedAvg'][alpha] = {
                                'train_loss': train_loss,
                                'train_accuracy': train_accuracy,
                                'final_accuracy': train_accuracy[-1] if train_accuracy else 0
                            }
                            print(f"  ✅ FedAvg α={alpha}: {train_accuracy[-1]*100:.2f}% final accuracy")
                            break
                except Exception as e:
                    print(f"  ❌ Error loading FedAvg α={alpha}: {e}")
        
        # Look for PUMB results
        pumb_patterns = [
            f'{save_dir}/PUMB_cifar_cnn_100_C[0.1]_iid[0]_E[5]_B[32].pkl',
            f'{save_dir}/*PUMB*alpha*{alpha}*.pkl',
            f'{save_dir}/*pumb*alpha*{alpha}*.pkl'
        ]
        
        for pattern in pumb_patterns:
            files = glob.glob(pattern)
            if files:
                try:
                    with open(files[0], 'rb') as f:
                        data = pickle.load(f)
                        if len(data) >= 2:
                            train_loss, train_accuracy = data[0], data[1]
                            results['PUMB'][alpha] = {
                                'train_loss': train_loss,
                                'train_accuracy': train_accuracy,
                                'final_accuracy': train_accuracy[-1] if train_accuracy else 0
                            }
                            print(f"  ✅ PUMB α={alpha}: {train_accuracy[-1]*100:.2f}% final accuracy")
                            break
                except Exception as e:
                    print(f"  ❌ Error loading PUMB α={alpha}: {e}")
    
    return results

def create_comprehensive_comparison_plot(results):
    """Create comprehensive comparison plots"""
    alpha_values = [0.01, 0.1, 0.5, 1.0]
    alpha_labels = ["α=0.01\n(Extreme\nNon-IID)", "α=0.1\n(High\nNon-IID)", 
                   "α=0.5\n(Moderate\nNon-IID)", "α=1.0\n(Low\nNon-IID)"]
    
    # Set up colors and style
    colors = {'FedAvg': '#FF6B6B', 'PUMB': '#4ECDC4'}
    plt.style.use('default')
    
    # Create main figure
    fig = plt.figure(figsize=(20, 16))
    
    # ===============================
    # 1. Final Accuracy Comparison
    # ===============================
    plt.subplot(3, 3, 1)
    fedavg_final = []
    pumb_final = []
    
    for alpha in alpha_values:
        fedavg_acc = results.get('FedAvg', {}).get(alpha, {}).get('final_accuracy', 0)
        pumb_acc = results.get('PUMB', {}).get(alpha, {}).get('final_accuracy', 0)
        fedavg_final.append(fedavg_acc * 100)
        pumb_final.append(pumb_acc * 100)
    
    x = np.arange(len(alpha_values))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, fedavg_final, width, label='FedAvg', 
                   color=colors['FedAvg'], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + width/2, pumb_final, width, label='PUMB', 
                   color=colors['PUMB'], alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.xlabel('Non-IID Level (α)', fontsize=12, fontweight='bold')
    plt.ylabel('Final Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Final Accuracy Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    plt.xticks(x, alpha_labels, fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
    
    # ===============================
    # 2. PUMB Advantage Analysis
    # ===============================
    plt.subplot(3, 3, 2)
    pumb_advantages = []
    for i, alpha in enumerate(alpha_values):
        if fedavg_final[i] > 0:
            advantage = ((pumb_final[i] - fedavg_final[i]) / fedavg_final[i]) * 100
            pumb_advantages.append(advantage)
        else:
            pumb_advantages.append(0)
    
    bar_colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in pumb_advantages]
    bars = plt.bar(alpha_labels, pumb_advantages, color=bar_colors, alpha=0.7, 
                  edgecolor='black', linewidth=1)
    
    plt.xlabel('Non-IID Level (α)', fontsize=12, fontweight='bold')
    plt.ylabel('PUMB Advantage (%)', fontsize=12, fontweight='bold')
    plt.title('PUMB Performance Advantage\nover FedAvg', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{height:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=10, fontweight='bold')
    
    # ===============================
    # 3. Training Loss Curves (α=0.5)
    # ===============================
    plt.subplot(3, 3, 3)
    if 0.5 in results.get('FedAvg', {}) and 0.5 in results.get('PUMB', {}):
        fedavg_loss = results['FedAvg'][0.5]['train_loss']
        pumb_loss = results['PUMB'][0.5]['train_loss']
        
        plt.plot(fedavg_loss, label='FedAvg', color=colors['FedAvg'], linewidth=2.5)
        plt.plot(pumb_loss, label='PUMB', color=colors['PUMB'], linewidth=2.5)
        plt.xlabel('Communication Rounds', fontsize=12, fontweight='bold')
        plt.ylabel('Training Loss', fontsize=12, fontweight='bold')
        plt.title('Training Loss Curves\n(α=0.5 - Moderate Non-IID)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
    
    # ===============================
    # 4. Training Accuracy Curves (α=0.5)
    # ===============================
    plt.subplot(3, 3, 4)
    if 0.5 in results.get('FedAvg', {}) and 0.5 in results.get('PUMB', {}):
        fedavg_acc = results['FedAvg'][0.5]['train_accuracy']
        pumb_acc = results['PUMB'][0.5]['train_accuracy']
        
        plt.plot(fedavg_acc, label='FedAvg', color=colors['FedAvg'], linewidth=2.5)
        plt.plot(pumb_acc, label='PUMB', color=colors['PUMB'], linewidth=2.5)
        plt.xlabel('Communication Rounds', fontsize=12, fontweight='bold')
        plt.ylabel('Training Accuracy', fontsize=12, fontweight='bold')
        plt.title('Training Accuracy Curves\n(α=0.5 - Moderate Non-IID)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
    
    # ===============================
    # 5. All Alpha Loss Curves
    # ===============================
    plt.subplot(3, 3, 5)
    for method in ['FedAvg', 'PUMB']:
        if method in results:
            for i, alpha in enumerate(alpha_values):
                if alpha in results[method]:
                    loss = results[method][alpha]['train_loss']
                    alpha_val = 0.6 + i * 0.1  # Varying transparency
                    plt.plot(loss, label=f'{method} α={alpha}', 
                            color=colors[method], alpha=alpha_val, linewidth=1.5)
    
    plt.xlabel('Communication Rounds', fontsize=12, fontweight='bold')
    plt.ylabel('Training Loss', fontsize=12, fontweight='bold')
    plt.title('Training Loss: All α Values', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # ===============================
    # 6. All Alpha Accuracy Curves
    # ===============================
    plt.subplot(3, 3, 6)
    for method in ['FedAvg', 'PUMB']:
        if method in results:
            for i, alpha in enumerate(alpha_values):
                if alpha in results[method]:
                    accuracy = results[method][alpha]['train_accuracy']
                    alpha_val = 0.6 + i * 0.1  # Varying transparency
                    plt.plot(accuracy, label=f'{method} α={alpha}', 
                            color=colors[method], alpha=alpha_val, linewidth=1.5)
    
    plt.xlabel('Communication Rounds', fontsize=12, fontweight='bold')
    plt.ylabel('Training Accuracy', fontsize=12, fontweight='bold')
    plt.title('Training Accuracy: All α Values', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # ===============================
    # 7. Convergence Speed Analysis
    # ===============================
    plt.subplot(3, 3, 7)
    fedavg_convergence = []
    pumb_convergence = []
    
    for alpha in alpha_values:
        for method, conv_list in [('FedAvg', fedavg_convergence), ('PUMB', pumb_convergence)]:
            if method in results and alpha in results[method]:
                accuracy = results[method][alpha]['train_accuracy']
                final_acc = results[method][alpha]['final_accuracy']
                target_acc = 0.8 * final_acc
                
                convergence_round = len(accuracy)
                for i, acc in enumerate(accuracy):
                    if acc >= target_acc:
                        convergence_round = i
                        break
                conv_list.append(convergence_round)
            else:
                conv_list.append(100)
    
    x = np.arange(len(alpha_values))
    bars1 = plt.bar(x - width/2, fedavg_convergence, width, label='FedAvg', 
                   color=colors['FedAvg'], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + width/2, pumb_convergence, width, label='PUMB', 
                   color=colors['PUMB'], alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.xlabel('Non-IID Level (α)', fontsize=12, fontweight='bold')
    plt.ylabel('Rounds to 80% Final Accuracy', fontsize=12, fontweight='bold')
    plt.title('Convergence Speed\n(Lower is Better)', fontsize=14, fontweight='bold')
    plt.xticks(x, alpha_labels, fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # ===============================
    # 8. Performance Stability
    # ===============================
    plt.subplot(3, 3, 8)
    fedavg_stability = []
    pumb_stability = []
    
    for alpha in alpha_values:
        for method, stab_list in [('FedAvg', fedavg_stability), ('PUMB', pumb_stability)]:
            if method in results and alpha in results[method]:
                accuracy = results[method][alpha]['train_accuracy']
                if len(accuracy) >= 20:
                    # Calculate std of last 20% of training
                    last_portion = accuracy[-int(len(accuracy)*0.2):]
                    stability = np.std(last_portion)
                    stab_list.append(stability)
                else:
                    stab_list.append(0)
            else:
                stab_list.append(0)
    
    x = np.arange(len(alpha_values))
    bars1 = plt.bar(x - width/2, fedavg_stability, width, label='FedAvg', 
                   color=colors['FedAvg'], alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + width/2, pumb_stability, width, label='PUMB', 
                   color=colors['PUMB'], alpha=0.8, edgecolor='black', linewidth=1)
    
    plt.xlabel('Non-IID Level (α)', fontsize=12, fontweight='bold')
    plt.ylabel('Training Stability (Std)', fontsize=12, fontweight='bold')
    plt.title('Training Stability\n(Lower is Better)', fontsize=14, fontweight='bold')
    plt.xticks(x, alpha_labels, fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # ===============================
    # 9. Summary Statistics Table
    # ===============================
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['α', 'FedAvg\nAcc(%)', 'PUMB\nAcc(%)', 'Advantage\n(%)', 'Winner']
    
    for i, alpha in enumerate(alpha_values):
        winner = "PUMB" if pumb_final[i] > fedavg_final[i] else "FedAvg" if fedavg_final[i] > pumb_final[i] else "Tie"
        row = [
            f"{alpha}",
            f"{fedavg_final[i]:.1f}",
            f"{pumb_final[i]:.1f}", 
            f"{pumb_advantages[i]:+.1f}",
            winner
        ]
        table_data.append(row)
    
    # Add summary row
    avg_advantage = np.mean([x for x in pumb_advantages if x != 0])
    pumb_wins = sum(1 for x in pumb_advantages if x > 0)
    table_data.append(['', '', '', '', ''])
    table_data.append(['Summary', f'{np.mean(fedavg_final):.1f}', f'{np.mean(pumb_final):.1f}', 
                      f'{avg_advantage:+.1f}', f'{pumb_wins}/{len(alpha_values)}'])
    
    table = plt.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     colWidths=[0.15, 0.18, 0.18, 0.18, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color the summary row
    for i in range(len(headers)):
        table[(len(table_data), i)].set_facecolor('#E8E8E8')
    
    plt.title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"../save/images/comprehensive_alpha_comparison_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive comparison plot saved to: {save_path}")
    
    plt.show()
    
    return save_path, pumb_advantages

def print_detailed_summary(results):
    """Print detailed experiment summary"""
    print("\n" + "="*80)
    print("🏆 COMPREHENSIVE ALPHA COMPARISON SUMMARY")
    print("="*80)
    
    alpha_values = [0.01, 0.1, 0.5, 1.0]
    
    print(f"\n{'Alpha':<8} {'Level':<15} {'FedAvg':<12} {'PUMB':<12} {'Advantage':<12} {'Winner':<8}")
    print("-" * 75)
    
    total_advantage = 0
    valid_comparisons = 0
    pumb_wins = 0
    
    level_names = {0.01: "Extreme", 0.1: "High", 0.5: "Moderate", 1.0: "Low"}
    
    for alpha in alpha_values:
        fedavg_acc = results.get('FedAvg', {}).get(alpha, {}).get('final_accuracy', 0) * 100
        pumb_acc = results.get('PUMB', {}).get(alpha, {}).get('final_accuracy', 0) * 100
        
        if fedavg_acc > 0 and pumb_acc > 0:
            advantage = ((pumb_acc - fedavg_acc) / fedavg_acc) * 100
            total_advantage += advantage
            valid_comparisons += 1
            winner = "PUMB" if pumb_acc > fedavg_acc else "FedAvg"
            if winner == "PUMB":
                pumb_wins += 1
        else:
            advantage = 0
            winner = "N/A"
        
        level = level_names.get(alpha, "Unknown")
        print(f"{alpha:<8} {level:<15} {fedavg_acc:<12.2f} {pumb_acc:<12.2f} {advantage:<12.1f} {winner:<8}")
    
    print("-" * 75)
    if valid_comparisons > 0:
        avg_advantage = total_advantage / valid_comparisons
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Average PUMB Advantage: {avg_advantage:+.1f}%")
        print(f"   PUMB Wins: {pumb_wins}/{valid_comparisons} ({pumb_wins/valid_comparisons*100:.1f}%)")
        print(f"   Best PUMB Performance: α={alpha_values[np.argmax([results.get('PUMB', {}).get(a, {}).get('final_accuracy', 0) for a in alpha_values])]}")
        print(f"   Best FedAvg Performance: α={alpha_values[np.argmax([results.get('FedAvg', {}).get(a, {}).get('final_accuracy', 0) for a in alpha_values])]}")
    
    print(f"\n📋 KEY INSIGHTS:")
    print(f"   • Lower α (more non-IID) = more challenging federated learning")
    print(f"   • PUMB's intelligent client selection helps in non-IID scenarios")
    print(f"   • Memory bank becomes more valuable with heterogeneous data")

if __name__ == "__main__":
    print("📊 Alpha Comparison Analysis")
    print("="*50)
    
    # Load results
    results = load_experiment_results()
    
    # Check if we have any results
    total_experiments = sum(len(results[method]) for method in results)
    
    if total_experiments == 0:
        print("❌ No experiment results found!")
        print("Make sure you've run the experiments and the pickle files are in ../save/objects/")
        print("Expected files:")
        for alpha in [0.01, 0.1, 0.5, 1.0]:
            print(f"  - PUMB_cifar_cnn_100_C[0.1]_iid[0]_E[5]_B[32].pkl (for α={alpha})")
            print(f"  - fedavg_cifar_cnn_100_C[0.1]_iid[0]_E[5]_B[32].pkl (for α={alpha})")
    else:
        print(f"✅ Found {total_experiments} experiment results")
        
        # Create comprehensive plots
        plot_path, advantages = create_comprehensive_comparison_plot(results)
        
        # Print detailed summary
        print_detailed_summary(results)
        
        print(f"\n📁 Results saved to: {plot_path}")
        print("🎉 Analysis complete!")