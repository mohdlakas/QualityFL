#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run multiple PUMB experiments with different seeds for statistical significance analysis.
This script automates the collection of data needed for research paper validation.
"""

import subprocess
import sys
import json
import numpy as np
from datetime import datetime
import os
from utils_dir import aggregate_multiple_runs

def run_single_experiment(seed, base_args):
    """Run a single experiment with a specific seed."""
    print(f"\n🚀 Running experiment with seed {seed}...")
    
    # Construct command
    cmd = [
        sys.executable, "federated_pumb_main.py",
        "--seed", str(seed),
        *base_args
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the experiment
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Experiment with seed {seed} completed successfully")
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment with seed {seed} failed:")
        print(f"   Error: {e}")
        print(f"   Stdout: {e.stdout}")
        print(f"   Stderr: {e.stderr}")
        return False, e.stdout, e.stderr

def parse_experiment_configs():
    """Parse different experimental configurations for comprehensive testing."""
    
    configs = [
        {
            "name": "CIFAR10_IID_baseline",
            "args": [
                "--dataset", "cifar",
                "--model", "cnn", 
                "--epochs", "20",
                "--num_users", "100",
                "--frac", "0.1",
                "--local_ep", "5",
                "--local_bs", "10",
                "--lr", "0.01",
                "--iid", "1",
                "--pumb_exploration_ratio", "0.4",
                "--pumb_initial_rounds", "3"
            ]
        },
        {
            "name": "CIFAR10_NonIID_challenging",
            "args": [
                "--dataset", "cifar",
                "--model", "cnn",
                "--epochs", "20", 
                "--num_users", "100",
                "--frac", "0.1",
                "--local_ep", "5",
                "--local_bs", "10",
                "--lr", "0.01",
                "--iid", "0",
                "--alpha", "0.5",
                "--pumb_exploration_ratio", "0.4",
                "--pumb_initial_rounds", "3"
            ]
        },
        {
            "name": "MNIST_NonIID_extreme",
            "args": [
                "--dataset", "mnist",
                "--model", "cnn",
                "--epochs", "15",
                "--num_users", "100", 
                "--frac", "0.1",
                "--local_ep", "5",
                "--local_bs", "10",
                "--lr", "0.01",
                "--iid", "0",
                "--alpha", "0.1",  # More extreme non-IID
                "--pumb_exploration_ratio", "0.4",
                "--pumb_initial_rounds", "3"
            ]
        }
    ]
    
    return configs

def main():
    """Main function to run multiple experiments."""
    print("🔬 PUMB Statistical Significance Testing Suite")
    print("=" * 60)
    
    # Configuration
    num_seeds = 5  # Run 5 experiments per configuration for statistical significance
    base_seeds = [42, 123, 456, 789, 999]  # Fixed seeds for reproducibility
    
    # Get experimental configurations
    configs = parse_experiment_configs()
    
    print(f"📋 Will run {len(configs)} configurations × {num_seeds} seeds = {len(configs) * num_seeds} total experiments")
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"../save/statistical_analysis_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    
    for config in configs:
        config_name = config["name"]
        config_args = config["args"]
        
        print(f"\n📊 Running configuration: {config_name}")
        print(f"   Arguments: {' '.join(config_args)}")
        
        config_results = {
            "config_name": config_name,
            "config_args": config_args,
            "seeds": [],
            "success_count": 0,
            "failed_seeds": []
        }
        
        for i, seed in enumerate(base_seeds[:num_seeds]):
            print(f"\n   Experiment {i+1}/{num_seeds} (seed={seed})")
            
            success, stdout, stderr = run_single_experiment(seed, config_args)
            
            if success:
                config_results["seeds"].append(seed)
                config_results["success_count"] += 1
            else:
                config_results["failed_seeds"].append(seed)
                
        all_results[config_name] = config_results
        
        print(f"\n✅ Configuration {config_name} completed:")
        print(f"   Successful runs: {config_results['success_count']}/{num_seeds}")
        if config_results["failed_seeds"]:
            print(f"   Failed seeds: {config_results['failed_seeds']}")
    
    # Save summary
    summary_file = os.path.join(results_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate statistical analysis report
    print(f"\n📈 Generating statistical significance analysis...")
    
    analysis_file = os.path.join(results_dir, "statistical_significance_report.txt")
    with open(analysis_file, 'w') as f:
        f.write("PUMB FEDERATED LEARNING - STATISTICAL SIGNIFICANCE ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Configurations Tested: {len(configs)}\n")
        f.write(f"Experiments per Configuration: {num_seeds}\n\n")
        
        for config_name, results in all_results.items():
            f.write(f"Configuration: {config_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Arguments: {' '.join(results['config_args'])}\n")
            f.write(f"Successful Runs: {results['success_count']}/{num_seeds}\n")
            f.write(f"Seeds Used: {results['seeds']}\n")
            if results["failed_seeds"]:
                f.write(f"Failed Seeds: {results['failed_seeds']}\n")
            f.write("\n")
            
        f.write("INSTRUCTIONS FOR STATISTICAL ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Check individual comprehensive_analysis_*.txt files in ../save/logs/\n")
        f.write("2. Extract final test accuracies for each configuration\n") 
        f.write("3. Calculate mean ± std for each configuration\n")
        f.write("4. Perform t-tests between PUMB and baseline methods\n")
        f.write("5. Report confidence intervals for key metrics\n")
        f.write("6. Use this data to validate PUMB's statistical significance\n")
    
    print(f"\n🎯 All experiments completed!")
    print(f"📁 Results saved in: {results_dir}")
    print(f"📊 Summary: {summary_file}")
    print(f"📈 Analysis: {analysis_file}")
    
    # Print quick summary
    print(f"\n📋 QUICK SUMMARY:")
    total_experiments = sum(r["success_count"] for r in all_results.values())
    total_possible = len(configs) * num_seeds
    print(f"   Total successful experiments: {total_experiments}/{total_possible}")
    
    for config_name, results in all_results.items():
        success_rate = results["success_count"] / num_seeds * 100
        print(f"   {config_name}: {results['success_count']}/{num_seeds} ({success_rate:.1f}%)")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Check comprehensive analysis files in ../save/logs/")
    print(f"   2. Aggregate results for statistical significance")
    print(f"   3. Compare PUMB vs baseline methods")
    print(f"   4. Generate plots and tables for your research paper")

if __name__ == "__main__":
    main()
