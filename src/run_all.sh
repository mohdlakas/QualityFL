#!/bin/bash
python federated_main.py --dataset=cifar  --model=cnn --epochs=100 --num_users=100 --frac=0.2 --local_ep=5 --local_bs=32 --optimizer=adam --lr=0.0005 --iid=0 --alpha=0.05 --seed=42 --gpu=1 --gpu_id=0
python federated_main.py --dataset=cifar  --model=cnn --epochs=100 --num_users=100 --frac=0.2 --local_ep=5 --local_bs=32 --optimizer=adam --lr=0.0005 --iid=0 --alpha=0.1 --seed=42 --gpu=1 --gpu_id=0
python federated_main.py --dataset=cifar  --model=cnn --epochs=100 --num_users=100 --frac=0.2 --local_ep=5 --local_bs=32 --optimizer=adam --lr=0.0005 --iid=0 --alpha=0.5 --seed=42 --gpu=1 --gpu_id=0
python federated_main.py --dataset=cifar  --model=cnn --epochs=100 --num_users=100 --frac=0.2 --local_ep=5 --local_bs=32 --optimizer=adam --lr=0.0005 --iid=0 --alpha=1.0 --seed=42 --gpu=1 --gpu_id=0

python federated_pumb_main.py --dataset=cifar --model=cnn --epochs=100 --num_users=100 --frac=0.2 --local_ep=5 --local_bs=32 --optimizer=adam --lr=0.0005 --iid=0 --alpha=0.01 --seed=42 --pumb_exploration_ratio=0.5 --pumb_initial_rounds=10 --gpu=1 --gpu_id=0
python federated_pumb_main.py --dataset=cifar --model=cnn --epochs=100 --num_users=100 --frac=0.2 --local_ep=5 --local_bs=32 --optimizer=adam --lr=0.0005 --iid=0 --alpha=0.1 --seed=42 --pumb_exploration_ratio=0.5 --pumb_initial_rounds=10 --gpu=1 --gpu_id=0
python federated_pumb_main.py --dataset=cifar --model=cnn --epochs=100 --num_users=100 --frac=0.2 --local_ep=5 --local_bs=32 --optimizer=adam --lr=0.0005 --iid=0 --alpha=0.5 --seed=42 --pumb_exploration_ratio=0.5 --pumb_initial_rounds=10 --gpu=1 --gpu_id=0
python federated_pumb_main.py --dataset=cifar --model=cnn --epochs=100 --num_users=100 --frac=0.2 --local_ep=5 --local_bs=32 --optimizer=adam --lr=0.0005 --iid=0 --alpha=1.0 --seed=42 --pumb_exploration_ratio=0.5 --pumb_initial_rounds=10 --gpu=1 --gpu_id=0

python federated_pumb_main.py --dataset=cifar100 --model=cnn --epochs=100 --num_users=100 --frac=0.1 --iid=0 --alpha=0.3