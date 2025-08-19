#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify FedAvg comprehensive analysis is working correctly.
"""

import subprocess
import sys
import os

def test_fedavg_analysis():
    """Test FedAvg comprehensive analysis with a quick experiment."""
    print("🧪 Testing FedAvg comprehensive analysis...")
    
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
    
    # Run FedAvg experiment
    cmd = [sys.executable, "federated_main.py"] + test_args
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ FedAvg test experiment completed successfully!")
        
        # Show key output lines
        output_lines = result.stdout.split('\n')
        key_lines = [line for line in output_lines if any(keyword in line.lower() for keyword in 
                    ['results summary', 'test accuracy', 'convergence', 'stability', 'comprehensive analysis'])]
        
        if key_lines:
            print("\n📊 Key output:")
            for line in key_lines[-10:]:  # Last 10 relevant lines
                print(f"   {line}")
        
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FedAvg test experiment failed!")
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def check_fedavg_analysis_files():
    """Check if FedAvg analysis files were generated."""
    log_dir = "../save/logs"
    
    if not os.path.exists(log_dir):
        print(f"❌ Log directory {log_dir} not found!")
        return False
    
    files = os.listdir(log_dir)
    fedavg_files = [f for f in files if f.startswith("fedavg_comprehensive_analysis_")]
    
    print(f"\n📁 Found {len(fedavg_files)} FedAvg comprehensive analysis file(s):")
    for f in fedavg_files:
        print(f"   {f}")
    
    if fedavg_files:
        # Show preview of latest file
        latest_file = max(fedavg_files)
        filepath = os.path.join(log_dir, latest_file)
        
        print(f"\n📄 Preview of {latest_file}:")
        print("-" * 50)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for line in lines[:25]:  # First 25 lines
                print(line.rstrip())
        print("...")
        print("-" * 50)
        
        return True
    else:
        print("❌ No FedAvg comprehensive analysis files found!")
        return False

def main():
    """Main test function."""
    print("🔍 FEDAVG COMPREHENSIVE ANALYSIS TEST")
    print("=" * 45)
    
    # Check if we're in the right directory
    if not os.path.exists("federated_main.py"):
        print("❌ Please run this script from the src/ directory!")
        return
    
    # Create necessary directories
    os.makedirs("../save/logs", exist_ok=True)
    os.makedirs("../save/objects", exist_ok=True)
    os.makedirs("../save/images", exist_ok=True)
    
    # Run test experiment
    if test_fedavg_analysis():
        # Check generated files
        if check_fedavg_analysis_files():
            print("\n✅ FedAvg analysis test completed successfully!")
            print("\n💡 Next steps:")
            print("   1. Run longer FedAvg experiments with: python federated_main.py [args]")
            print("   2. Run PUMB experiments with: python federated_pumb_main.py [args]")
            print("   3. Compare methods with: python compare_fedavg_pumb.py")
            print("   4. Use the analysis files for your research paper!")
            print("\n🔬 The comprehensive analysis now tracks:")
            print("   • Performance metrics (accuracy, convergence, stability)")
            print("   • Client selection patterns")
            print("   • Training dynamics")
            print("   • Statistical data for significance testing")
        else:
            print("\n⚠️  Test experiment ran but analysis files not found!")
    else:
        print("\n❌ Test failed! Check the error messages above.")

if __name__ == "__main__":
    main()
