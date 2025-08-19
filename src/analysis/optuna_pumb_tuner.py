import subprocess
import optuna
import os
import csv
import shutil
import glob
from datetime import datetime

# Setup directories
summary_dir = "../save/optuna_results"
plot_dir = os.path.join(summary_dir, "plots")
os.makedirs(summary_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
csv_path = os.path.join(summary_dir, "tuning_summary.csv")

# Write CSV header
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Trial", "frac", "local_ep", "exploration_ratio", "initial_rounds", "test_acc", "train_acc", "plot_file"])

def objective(trial):
    # Suggest hyperparameters
    frac = 0.5
    local_ep = trial.suggest_categorical('local_ep', [2, 3, 5, 7])
    pumb_ratio = 0.1
    pumb_rounds = 10

    # Build command
    command = [
        "python", "federated_pumb_main.py",
        "--dataset=cifar",
        "--model=cnn",
        "--epochs=50",
        "--num_users=100",
        f"--frac={frac}",
        f"--local_ep={local_ep}",
        "--local_bs=32",
        "--optimizer=adam",
        "--lr=0.0005",
        "--iid=0",
        "--alpha=0.5",
        "--verbose=0",
        "--seed=42",
        f"--pumb_exploration_ratio={pumb_ratio}",
        f"--pumb_initial_rounds={pumb_rounds}",
        "--gpu=1",
        "--gpu_id=0"
    ]

    # Run experiment
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    test_acc = 0.0
    train_acc = 0.0

    for line in result.stdout.splitlines():
        if "Final Test Accuracy" in line:
            try:
                test_acc = float(line.strip().split(":")[-1].replace("%", "").strip()) / 100.0
            except:
                pass
        if "Avg Train Accuracy" in line:
            try:
                train_acc = float(line.strip().split(":")[-1].replace("%", "").strip()) / 100.0
            except:
                pass

    # Locate the latest plot file
    latest_plot = ""
    try:
        plot_files = sorted(
            glob.glob("../save/images/PUMB_*_loss_acc_*.png"),
            key=os.path.getmtime,
            reverse=True
        )
        if plot_files:
            latest_plot = plot_files[0]
            new_plot_name = f"trial_{trial.number}_acc_{round(test_acc*100, 2)}.png"
            dest_path = os.path.join(plot_dir, new_plot_name)
            shutil.copyfile(latest_plot, dest_path)
    except Exception as e:
        print(f"Plot copy failed for trial {trial.number}: {e}")

    # Log to CSV
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trial.number, round(frac, 3), local_ep,
            round(pumb_ratio, 3), pumb_rounds,
            round(test_acc, 4), round(train_acc, 4),
            os.path.basename(latest_plot) if latest_plot else "N/A"
        ])

    return test_acc

if __name__ == "__main__":
    # Force testing each value exactly once
    search_space = {'local_ep': [2, 5]}
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space)
    )
    study.optimize(objective, n_trials=4)  # Will test each value once

    print("✅ Best hyperparameters found:")
    print(study.best_params)
    print("🏆 Best test accuracy:", round(study.best_value * 100, 2), "%")
    print(f"\n📄 Summary saved to: {csv_path}")
    print(f"🖼️ Plots saved in: {plot_dir}")
