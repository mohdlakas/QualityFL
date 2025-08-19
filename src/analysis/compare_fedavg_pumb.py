#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparison script to run both FedAvg and PUMB experiments with identical parameters
for fair comparison and statistical analysis.
"""

import subprocess
import sys
import json
import numpy as np
from datetime import datetime
import os
import glob
import re

def run_experiment(script_name, args, experiment_name):
    """Run a single experiment with specified script and arguments."""
    print(f"\n🚀 Running {experiment_name}...")
    
    # Construct command
    cmd = [sys.executable, script_name] + args
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {experiment_name} completed successfully")
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"❌ {experiment_name} failed:")
        print(f"   Error: {e}")
        print(f"   Stdout: {e.stdout}")
        print(f"   Stderr: {e.stderr}")
        return False, e.stdout, e.stderr

def parse_analysis_file(filepath, method_name):
    """Parse a comprehensive analysis file and extract key metrics."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        metrics = {'method': method_name}
        
        # Extract key performance metrics
        test_acc_match = re.search(r'Final Test Accuracy: ([\d.]+)', content)
        if test_acc_match:
            metrics['test_accuracy'] = float(test_acc_match.group(1))
            
        conv_match = re.search(r'Convergence Speed: (\d+) rounds', content)
        if conv_match:
            metrics['convergence_rounds'] = int(conv_match.group(1))
            
        stability_match = re.search(r'Training Stability: ([\d.e-]+)', content)
        if stability_match:
            metrics['training_stability'] = float(stability_match.group(1))
            
        runtime_match = re.search(r'Total Time Seconds: ([\d.]+)', content)
        if runtime_match:
            metrics['total_time'] = float(runtime_match.group(1))
            
        # Extract client selection metrics
        unique_clients_match = re.search(r'Total Unique Clients Selected: (\d+)', content)
        if unique_clients_match:
            metrics['unique_clients_selected'] = int(unique_clients_match.group(1))
            
        participation_rate_match = re.search(r'Average Participation Rate: ([\d.]+)', content)
        if participation_rate_match:
            metrics['avg_participation_rate'] = float(participation_rate_match.group(1))
            
        return metrics
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def get_latest_analysis_files():
    """Get the most recent analysis files for both methods."""
    log_dir = "../save/logs"
    
    # Find FedAvg files
    fedavg_pattern = os.path.join(log_dir, "fedavg_comprehensive_analysis_*.txt")
    fedavg_files = glob.glob(fedavg_pattern)
    
    # Find PUMB files  
    pumb_pattern = os.path.join(log_dir, "comprehensive_analysis_*.txt")
    pumb_files = [f for f in glob.glob(pumb_pattern) if "fedavg" not in f]
    
    # Get latest files
    latest_fedavg = max(fedavg_files) if fedavg_files else None
    latest_pumb = max(pumb_files) if pumb_files else None
    
    return latest_fedavg, latest_pumb

def generate_comparison_report(fedavg_metrics, pumb_metrics, output_file):
    """Generate a detailed comparison report between FedAvg and PUMB."""
    
    with open(output_file, 'w') as f:
        f.write("FEDAVG vs PUMB COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if not fedavg_metrics or not pumb_metrics:
            f.write("ERROR: Missing data for comparison\n")
            if not fedavg_metrics:
                f.write("- FedAvg metrics not found\n")
            if not pumb_metrics:
                f.write("- PUMB metrics not found\n")
            return
        
        # Performance comparison
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-" * 30 + "\n")
        
        metrics_to_compare = [
            ('test_accuracy', 'Test Accuracy', '.4f', 'higher_better'),
            ('convergence_rounds', 'Convergence Rounds', '.0f', 'lower_better'),
            ('training_stability', 'Training Stability', '.6f', 'lower_better'),
            ('total_time', 'Total Time (seconds)', '.2f', 'lower_better'),
            ('unique_clients_selected', 'Unique Clients Selected', '.0f', 'depends'),
            ('avg_participation_rate', 'Avg Participation Rate', '.4f', 'depends')
        ]
        
        f.write(f"{'Metric':<25} {'FedAvg':<15} {'PUMB':<15} {'Improvement':<15} {'Winner':<10}\n")
        f.write("-" * 85 + "\n")
        
        for metric_key, metric_name, format_str, better_direction in metrics_to_compare:
            fedavg_val = fedavg_metrics.get(metric_key)
            pumb_val = pumb_metrics.get(metric_key)
            
            if fedavg_val is not None and pumb_val is not None:
                # Calculate improvement
                if better_direction == 'higher_better':
                    improvement = ((pumb_val - fedavg_val) / fedavg_val) * 100
                    winner = "PUMB" if pumb_val > fedavg_val else "FedAvg"
                elif better_direction == 'lower_better':
                    improvement = ((fedavg_val - pumb_val) / fedavg_val) * 100
                    winner = "PUMB" if pumb_val < fedavg_val else "FedAvg"
                else:
                    improvement = ((pumb_val - fedavg_val) / fedavg_val) * 100
                    winner = "Depends"
                
                f.write(f"{metric_name:<25} ")
                f.write(f"{fedavg_val:{format_str}} ".ljust(15))
                f.write(f"{pumb_val:{format_str}} ".ljust(15))
                f.write(f"{improvement:+.2f}% ".ljust(15))
                f.write(f"{winner:<10}\n")
            else:
                f.write(f"{metric_name:<25} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<10}\n")
        
        f.write("\n")
        
        # Key findings
        f.write("KEY FINDINGS:\n")
        f.write("-" * 15 + "\n")
        
        if fedavg_metrics.get('test_accuracy') and pumb_metrics.get('test_accuracy'):
            acc_improvement = ((pumb_metrics['test_accuracy'] - fedavg_metrics['test_accuracy']) / 
                             fedavg_metrics['test_accuracy']) * 100
            f.write(f"• Test Accuracy: PUMB shows {acc_improvement:+.2f}% ")
            f.write("improvement over FedAvg\n" if acc_improvement > 0 else "vs FedAvg\n")
        
        if fedavg_metrics.get('convergence_rounds') and pumb_metrics.get('convergence_rounds'):
            conv_improvement = ((fedavg_metrics['convergence_rounds'] - pumb_metrics['convergence_rounds']) / 
                              fedavg_metrics['convergence_rounds']) * 100
            f.write(f"• Convergence Speed: PUMB converges {conv_improvement:+.1f}% ")
            f.write("faster than FedAvg\n" if conv_improvement > 0 else "vs FedAvg\n")
        
        if fedavg_metrics.get('training_stability') and pumb_metrics.get('training_stability'):
            stab_improvement = ((fedavg_metrics['training_stability'] - pumb_metrics['training_stability']) / 
                              fedavg_metrics['training_stability']) * 100
            f.write(f"• Training Stability: PUMB shows {stab_improvement:+.1f}% ")
            f.write("more stable training\n" if stab_improvement > 0 else "vs FedAvg\n")
        
        # Statistical significance note
        f.write("\nSTATISTICAL SIGNIFICANCE:\n")
        f.write("-" * 25 + "\n")
        f.write("This is a single-run comparison. For statistical significance:\n")
        f.write("1. Run multiple experiments with different seeds\n")
        f.write("2. Calculate mean ± std for each metric\n")
        f.write("3. Perform t-tests to establish significance (p < 0.05)\n")
        f.write("4. Report effect sizes (Cohen's d) for practical significance\n")

def run_comparison_experiment(config_name, base_args, seeds=[42, 123, 456]):
    """Run comparison experiments with multiple seeds."""
    print(f"\n📊 Running comparison experiment: {config_name}")
    print(f"Base arguments: {' '.join(base_args)}")
    
    results = {'config_name': config_name, 'fedavg': [], 'pumb': []}
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        
        # Add seed to arguments
        experiment_args = base_args + ["--seed", str(seed)]
        
        # Run FedAvg
        fedavg_success, _, _ = run_experiment("federated_main.py", experiment_args, f"FedAvg (seed={seed})")
        
        # Run PUMB with additional PUMB-specific parameters
        pumb_args = experiment_args + [
            "--pumb_exploration_ratio", "0.2",
            "--pumb_initial_rounds", "5"
        ]
        pumb_success, _, _ = run_experiment("federated_pumb_main.py", pumb_args, f"PUMB (seed={seed})")
        
        if fedavg_success and pumb_success:
            # Get the latest analysis files
            latest_fedavg, latest_pumb = get_latest_analysis_files()
            
            if latest_fedavg and latest_pumb:
                # Parse results
                fedavg_metrics = parse_analysis_file(latest_fedavg, "FedAvg")
                pumb_metrics = parse_analysis_file(latest_pumb, "PUMB")
                
                if fedavg_metrics and pumb_metrics:
                    fedavg_metrics['seed'] = seed
                    pumb_metrics['seed'] = seed
                    results['fedavg'].append(fedavg_metrics)
                    results['pumb'].append(pumb_metrics)
                    
                    # Generate single-run comparison
                    comparison_file = f"../save/logs/comparison_{config_name}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    generate_comparison_report(fedavg_metrics, pumb_metrics, comparison_file)
                    print(f"📄 Comparison report: {comparison_file}")
    
    return results

def main():
    """Main comparison function."""
    print("⚖️  FEDAVG vs PUMB COMPARISON SUITE")
    print("=" * 50)
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"../save/comparison_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define comparison configurations
    comparison_configs = [
        {
            "name": "CIFAR10_Custom_Config",
            "args": [
                "--dataset", "cifar",
                "--model", "cnn",
                "--epochs", "100",
                "--num_users", "100",
                "--frac", "0.2",
                "--local_ep", "5",
                "--local_bs", "32",
                "--optimizer", "adam",
                "--lr", "0.001",
                "--iid", "0",
                "--alpha", "0.5",
                "--verbose", "1",
                "--gpu", "1",
                "--gpu_id", "0"
            ]
        }
    ]
    
all_results = {}
local_ep_values = [3, 4, 5]  # Add as many local epoch values as you want

for config in comparison_configs:
    for local_ep in local_ep_values:
        args = config["args"].copy()
        # Find the index of '--local_ep' and update its value
        if '--local_ep' in args:
            ep_idx = args.index('--local_ep') + 1
            args[ep_idx] = str(local_ep)
        else:
            args += ['--local_ep', str(local_ep)]
        
        config_name = f"{config['name']}_localep{local_ep}"
        config_results = run_comparison_experiment(
            config_name,
            args,
            seeds=[42]  # or any fixed seed
        )
        all_results[config_name] = config_results
    
    # Generate aggregate comparison report
    aggregate_file = os.path.join(results_dir, "aggregate_comparison_report.txt")
    with open(aggregate_file, 'w') as f:
        f.write("FEDAVG vs PUMB - AGGREGATE COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configurations Tested: {len(comparison_configs)}\n\n")
        
        for config_name, config_results in all_results.items():
            f.write(f"CONFIGURATION: {config_name}\n")
            f.write("-" * 40 + "\n")
            
            fedavg_results = config_results['fedavg']
            pumb_results = config_results['pumb']
            
            f.write(f"Successful runs: FedAvg={len(fedavg_results)}, PUMB={len(pumb_results)}\n")
            
            if fedavg_results and pumb_results:
                # Calculate aggregate statistics
                fedavg_accs = [r['test_accuracy'] for r in fedavg_results if 'test_accuracy' in r]
                pumb_accs = [r['test_accuracy'] for r in pumb_results if 'test_accuracy' in r]
                
                if fedavg_accs and pumb_accs:
                    fedavg_mean = np.mean(fedavg_accs)
                    fedavg_std = np.std(fedavg_accs)
                    pumb_mean = np.mean(pumb_accs)
                    pumb_std = np.std(pumb_accs)
                    
                    f.write(f"Test Accuracy Results:\n")
                    f.write(f"  FedAvg: {fedavg_mean:.4f} ± {fedavg_std:.4f}\n")
                    f.write(f"  PUMB:   {pumb_mean:.4f} ± {pumb_std:.4f}\n")
                    
                    improvement = ((pumb_mean - fedavg_mean) / fedavg_mean) * 100
                    f.write(f"  PUMB Improvement: {improvement:+.2f}%\n")
            
            f.write("\n")
    
    # Save raw results
    json_file = os.path.join(results_dir, "raw_comparison_results.json")
    with open(json_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(all_results, f, indent=2, default=convert_numpy)
    
    print(f"\n🎯 Comparison completed!")
    print(f"📁 Results saved in: {results_dir}")
    print(f"📊 Aggregate report: {aggregate_file}")
    print(f"💾 Raw data: {json_file}")
    
    # Print summary
    print(f"\n📋 COMPARISON SUMMARY:")
    for config_name, config_results in all_results.items():
        fedavg_count = len(config_results['fedavg'])
        pumb_count = len(config_results['pumb'])
        print(f"   {config_name}: FedAvg={fedavg_count}, PUMB={pumb_count} successful runs")

if __name__ == "__main__":
    main()
