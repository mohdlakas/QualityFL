#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test script to verify the comprehensive analysis is working.
Run this to test the analysis functionality with a short experiment.
"""

import subprocess
import sys
import os

def run_test_experiment():
    """Run a quick test experiment to verify analysis functionality."""
    print("🧪 Running test experiment to verify comprehensive analysis...")
    
    # Quick test parameters (very short experiment)
    test_args = [
        "--dataset", "mnist",
        "--model", "cnn",
        "--epochs", "3",  # Very short for testing
        "--num_users", "10",  # Few users for speed
        "--frac", "0.5",
        "--local_ep", "1",
        "--local_bs", "10",
        "--lr", "0.01",
        "--iid", "1",
        "--seed", "42"
    ]
    
    # Run the experiment
    cmd = [sys.executable, "federated_pumb_main.py"] + test_args
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Test experiment completed successfully!")
        print("\nOutput preview:")
        print(result.stdout[-500:])  # Last 500 characters
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Test experiment failed!")
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def check_generated_files():
    """Check if the comprehensive analysis files were generated."""
    log_dir = "../save/logs"
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory {log_dir} not found!")
        return False
    
    files = os.listdir(log_dir)
    comprehensive_files = [f for f in files if f.startswith("comprehensive_analysis_")]
    
    print(f"\n📁 Found {len(comprehensive_files)} comprehensive analysis file(s):")
    for f in comprehensive_files:
        print(f"   {f}")
    
    if comprehensive_files:
        # Show preview of latest file
        latest_file = max(comprehensive_files)
        filepath = os.path.join(log_dir, latest_file)
        
        print(f"\n📄 Preview of {latest_file}:")
        print("-" * 50)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[:20]:  # First 20 lines
                print(line.rstrip())
        print("...")
        print("-" * 50)
        
        return True
    else:
        print("❌ No comprehensive analysis files found!")
        return False

def main():
    """Main test function."""
    print("🔍 PUMB Comprehensive Analysis Test")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("federated_pumb_main.py"):
        print("❌ Please run this script from the src/ directory!")
        return
    
    # Create necessary directories
    os.makedirs("../save/logs", exist_ok=True)
    os.makedirs("../save/objects", exist_ok=True)
    os.makedirs("../save/images", exist_ok=True)
    
    # Run test experiment
    if run_test_experiment():
        # Check generated files
        if check_generated_files():
            print("\n✅ Test completed successfully!")
            print("\n💡 Next steps:")
            print("   1. Run longer experiments with: python federated_pumb_main.py [args]")
            print("   2. Run multiple seeds with: python run_multiple_experiments.py")
            print("   3. Analyze results with: python analyze_results.py")
            print("   4. Use the comprehensive analysis files for your research paper!")
        else:
            print("\n⚠️  Test experiment ran but analysis files not found!")
    else:
        print("\n❌ Test failed! Check the error messages above.")

if __name__ == "__main__":
    main()
