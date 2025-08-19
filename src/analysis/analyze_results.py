#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Results analysis script for PUMB experiments.
Extracts and aggregates results from multiple comprehensive analysis files.
"""

import os
import re
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import glob

def parse_comprehensive_analysis_file(filepath):
    """Parse a comprehensive analysis file and extract key metrics."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract metrics using regex patterns
        metrics = {}
        
        # Performance metrics
        test_acc_match = re.search(r'Final Test Accuracy: ([\d.]+)', content)
        if test_acc_match:
            metrics['test_accuracy'] = float(test_acc_match.group(1))
            
        conv_match = re.search(r'Convergence Speed: (\d+) rounds', content)
        if conv_match:
            metrics['convergence_rounds'] = int(conv_match.group(1))
            
        stability_match = re.search(r'Training Stability: ([\d.e-]+)', content)
        if stability_match:
            metrics['training_stability'] = float(stability_match.group(1))
            
        # Client selection metrics
        unique_clients_match = re.search(r'Total Unique Clients Selected: (\d+)', content)
        if unique_clients_match:
            metrics['unique_clients_selected'] = int(unique_clients_match.group(1))
            
        participation_rate_match = re.search(r'Average Participation Rate: ([\d.]+)', content)
        if participation_rate_match:
            metrics['avg_participation_rate'] = float(participation_rate_match.group(1))
            
        reliability_improvement_match = re.search(r'Average Reliability Improvement: ([\d.e-]+)', content)
        if reliability_improvement_match:
            metrics['avg_reliability_improvement'] = float(reliability_improvement_match.group(1))
            
        # Memory bank metrics
        memory_size_match = re.search(r'Final Memory Size: (\d+)', content)
        if memory_size_match:
            metrics['final_memory_size'] = int(memory_size_match.group(1))
            
        similarity_match = re.search(r'Average Similarity Score: ([\d.e-]+)', content)
        if similarity_match:
            metrics['avg_similarity_score'] = float(similarity_match.group(1))
            
        # Extract experiment configuration
        dataset_match = re.search(r'Dataset: (\w+)', content)
        if dataset_match:
            metrics['dataset'] = dataset_match.group(1)
            
        iid_match = re.search(r'Iid: (\w+)', content)
        if iid_match:
            metrics['iid'] = iid_match.group(1)
            
        alpha_match = re.search(r'Alpha: ([\d.]+|NA)', content)
        if alpha_match:
            metrics['alpha'] = alpha_match.group(1)
            
        seed_match = re.search(r'Experiment Seed: (\d+|None)', content)
        if seed_match:
            metrics['seed'] = seed_match.group(1)
            
        return metrics
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def group_results_by_configuration(results):
    """Group results by experimental configuration."""
    groups = {}
    
    for result in results:
        if result is None:
            continue
            
        # Create configuration key
        config_key = f"{result.get('dataset', 'unknown')}_{result.get('iid', 'unknown')}"
        if result.get('alpha') != 'NA':
            config_key += f"_alpha{result.get('alpha', 'unknown')}"
            
        if config_key not in groups:
            groups[config_key] = []
        groups[config_key].append(result)
    
    return groups

def calculate_statistics(values):
    """Calculate comprehensive statistics for a list of values."""
    if not values:
        return {}
        
    values = np.array(values)
    n = len(values)
    
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0
    sem = std / np.sqrt(n) if n > 1 else 0
    
    # 95% confidence interval
    if n > 1:
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    else:
        ci = (mean, mean)
    
    return {
        'count': n,
        'mean': mean,
        'std': std,
        'sem': sem,
        'min': np.min(values),
        'max': np.max(values),
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'values': values.tolist()
    }

def generate_comparison_report(grouped_results, output_file):
    """Generate a comprehensive comparison report."""
    
    with open(output_file, 'w') as f:
        f.write("PUMB FEDERATED LEARNING - RESULTS COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("SUMMARY TABLE:\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Configuration':<25} {'Count':<7} {'Test Acc':<12} {'Conv Speed':<12} {'Stability':<12}\n")
        f.write("-" * 70 + "\n")
        
        for config_name, results in grouped_results.items():
            test_accs = [r.get('test_accuracy') for r in results if r.get('test_accuracy') is not None]
            conv_rounds = [r.get('convergence_rounds') for r in results if r.get('convergence_rounds') is not None]
            stabilities = [r.get('training_stability') for r in results if r.get('training_stability') is not None]
            
            test_acc_stats = calculate_statistics(test_accs)
            conv_stats = calculate_statistics(conv_rounds)
            stab_stats = calculate_statistics(stabilities)
            
            f.write(f"{config_name:<25} {test_acc_stats.get('count', 0):<7} ")
            f.write(f"{test_acc_stats.get('mean', 0):.4f}±{test_acc_stats.get('std', 0):.4f:<12} ")
            f.write(f"{conv_stats.get('mean', 0):.1f}±{conv_stats.get('std', 0):.1f:<12} ")
            f.write(f"{stab_stats.get('mean', 0):.2e}±{stab_stats.get('std', 0):.2e:<12}\n")
        
        f.write("\n\n")
        
        # Detailed analysis for each configuration
        for config_name, results in grouped_results.items():
            f.write(f"DETAILED ANALYSIS: {config_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Number of experiments: {len(results)}\n\n")
            
            # Performance metrics
            metrics_to_analyze = [
                ('test_accuracy', 'Test Accuracy', '.4f'),
                ('convergence_rounds', 'Convergence Rounds', '.1f'),
                ('training_stability', 'Training Stability', '.6f'),
                ('unique_clients_selected', 'Unique Clients Selected', '.0f'),
                ('avg_participation_rate', 'Avg Participation Rate', '.4f'),
                ('avg_reliability_improvement', 'Avg Reliability Improvement', '.6f'),
                ('final_memory_size', 'Final Memory Size', '.0f'),
                ('avg_similarity_score', 'Avg Similarity Score', '.6f')
            ]
            
            for metric_key, metric_name, format_str in metrics_to_analyze:
                values = [r.get(metric_key) for r in results if r.get(metric_key) is not None]
                if values:
                    stats_dict = calculate_statistics(values)
                    f.write(f"{metric_name}:\n")
                    f.write(f"  Mean ± Std: {stats_dict['mean']:{format_str}} ± {stats_dict['std']:{format_str}}\n")
                    f.write(f"  95% CI: [{stats_dict['ci_lower']:{format_str}}, {stats_dict['ci_upper']:{format_str}}]\n")
                    f.write(f"  Range: [{stats_dict['min']:{format_str}}, {stats_dict['max']:{format_str}}]\n")
                    f.write(f"  Individual values: {[format(v, format_str) for v in values]}\n\n")
            
            # List all seeds used
            seeds = [r.get('seed') for r in results if r.get('seed') is not None]
            f.write(f"Seeds used: {seeds}\n\n")
            
        # Statistical significance notes
        f.write("STATISTICAL SIGNIFICANCE ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        f.write("To establish statistical significance:\n")
        f.write("1. Compare PUMB results with baseline FedAvg results\n")
        f.write("2. Perform two-sample t-tests between methods\n")
        f.write("3. Check for significance at p < 0.05 level\n")
        f.write("4. Report effect sizes (Cohen's d) for meaningful differences\n\n")
        
        # Add recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 20 + "\n")
        
        # Find best performing configuration
        best_config = None
        best_accuracy = 0
        for config_name, results in grouped_results.items():
            test_accs = [r.get('test_accuracy') for r in results if r.get('test_accuracy') is not None]
            if test_accs:
                mean_acc = np.mean(test_accs)
                if mean_acc > best_accuracy:
                    best_accuracy = mean_acc
                    best_config = config_name
        
        if best_config:
            f.write(f"Best performing configuration: {best_config} (Mean accuracy: {best_accuracy:.4f})\n")
        
        # Check for convergence speed
        fastest_config = None
        fastest_rounds = float('inf')
        for config_name, results in grouped_results.items():
            conv_rounds = [r.get('convergence_rounds') for r in results if r.get('convergence_rounds') is not None]
            if conv_rounds:
                mean_rounds = np.mean(conv_rounds)
                if mean_rounds < fastest_rounds:
                    fastest_rounds = mean_rounds
                    fastest_config = config_name
        
        if fastest_config:
            f.write(f"Fastest converging configuration: {fastest_config} (Mean rounds: {fastest_rounds:.1f})\n")

def create_comparison_plots(grouped_results, output_dir):
    """Create comparison plots for different configurations."""
    
    # Test accuracy comparison
    plt.figure(figsize=(12, 8))
    
    config_names = []
    mean_accuracies = []
    std_accuracies = []
    
    for config_name, results in grouped_results.items():
        test_accs = [r.get('test_accuracy') for r in results if r.get('test_accuracy') is not None]
        if test_accs:
            stats_dict = calculate_statistics(test_accs)
            config_names.append(config_name.replace('_', '\n'))
            mean_accuracies.append(stats_dict['mean'])
            std_accuracies.append(stats_dict['std'])
    
    x_pos = np.arange(len(config_names))
    plt.bar(x_pos, mean_accuracies, yerr=std_accuracies, capsize=5, alpha=0.7)
    plt.xlabel('Configuration')
    plt.ylabel('Test Accuracy')
    plt.title('PUMB Test Accuracy Comparison\n(Error bars show ±1 std)')
    plt.xticks(x_pos, config_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Convergence speed comparison
    plt.figure(figsize=(12, 8))
    
    mean_convergence = []
    std_convergence = []
    valid_configs = []
    
    for config_name, results in grouped_results.items():
        conv_rounds = [r.get('convergence_rounds') for r in results if r.get('convergence_rounds') is not None]
        if conv_rounds:
            stats_dict = calculate_statistics(conv_rounds)
            valid_configs.append(config_name.replace('_', '\n'))
            mean_convergence.append(stats_dict['mean'])
            std_convergence.append(stats_dict['std'])
    
    if valid_configs:
        x_pos = np.arange(len(valid_configs))
        plt.bar(x_pos, mean_convergence, yerr=std_convergence, capsize=5, alpha=0.7, color='orange')
        plt.xlabel('Configuration')
        plt.ylabel('Convergence Rounds')
        plt.title('PUMB Convergence Speed Comparison\n(Lower is better)')
        plt.xticks(x_pos, valid_configs, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'convergence_speed_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to analyze results."""
    print("📊 PUMB Results Analysis Tool")
    print("=" * 50)
    
    # Find all comprehensive analysis files
    log_dir = "../save/logs"
    pattern = os.path.join(log_dir, "comprehensive_analysis_*.txt")
    analysis_files = glob.glob(pattern)
    
    if not analysis_files:
        print(f"❌ No comprehensive analysis files found in {log_dir}")
        print("   Make sure you've run experiments with the updated PUMB code.")
        return
    
    print(f"📁 Found {len(analysis_files)} analysis files")
    
    # Parse all files
    results = []
    for filepath in analysis_files:
        print(f"   Parsing: {os.path.basename(filepath)}")
        parsed_result = parse_comprehensive_analysis_file(filepath)
        if parsed_result:
            results.append(parsed_result)
    
    print(f"✅ Successfully parsed {len(results)} files")
    
    if not results:
        print("❌ No valid results found!")
        return
    
    # Group by configuration
    grouped_results = group_results_by_configuration(results)
    print(f"📋 Found {len(grouped_results)} different configurations:")
    for config_name, config_results in grouped_results.items():
        print(f"   {config_name}: {len(config_results)} experiments")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"../save/analysis_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate comparison report
    report_file = os.path.join(output_dir, "comparison_report.txt")
    generate_comparison_report(grouped_results, report_file)
    print(f"📄 Comparison report saved: {report_file}")
    
    # Create plots
    create_comparison_plots(grouped_results, output_dir)
    print(f"📈 Comparison plots saved in: {output_dir}")
    
    # Save raw data as JSON for further analysis
    json_file = os.path.join(output_dir, "raw_results.json")
    with open(json_file, 'w') as f:
        json.dump(grouped_results, f, indent=2)
    print(f"💾 Raw results saved: {json_file}")
    
    print(f"\n🎯 Analysis completed!")
    print(f"📁 All results in: {output_dir}")
    
    # Print quick summary
    print(f"\n📋 QUICK SUMMARY:")
    for config_name, config_results in grouped_results.items():
        test_accs = [r.get('test_accuracy') for r in config_results if r.get('test_accuracy') is not None]
        if test_accs:
            mean_acc = np.mean(test_accs)
            std_acc = np.std(test_accs)
            print(f"   {config_name}: {mean_acc:.4f} ± {std_acc:.4f} ({len(test_accs)} runs)")

if __name__ == "__main__":
    main()
