#!/bin/bash
python scaffold_main.py --dataset=cifar --model=cnn --epochs=100 --num_users=100 --frac=0.1 --local_ep=3 --local_bs=32 --optimizer=adam --lr=0.001 --iid=0 --alpha=0.5 --seed=42 --gpu=1
python scaffold_main.py --dataset=cifar --model=cnn --epochs=100 --num_users=100 --frac=0.2 --local_ep=2 --local_bs=32 --optimizer=adam --lr=0.001 --iid=0 --alpha=0.5 --seed=42 --gpu=1
python scaffold_main.py --dataset=cifar --model=cnn --epochs=100 --num_users=100 --frac=0.1 --local_ep=5 --local_bs=32 --optimizer=adam --lr=0.001 --iid=0 --alpha=0.5 --seed=42 --gpu=1

