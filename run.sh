#!/usr/bin/env bash

# Image classification example
python src/run_main.py --model=cnn --dataset=mnist --alg=fedavg \
       --diric=1 --epochs=500 --optimizer=adsgd \
       --exp_name=image --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --frac=0.1 

# FedAdam example
python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedadam --global_lr=0.00001 \
       --diric=0.1 --epochs=2000 --optimizer=adsgd \
       --exp_name=fedadam --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --frac=0.1 

# FedProx example
python src/run_main.py --model=resnet18 --dataset=cifar --alg=fedprox \
       --diric=0.01 --epochs=2000 --optimizer=adsgd --fedprox_mu=0.01 \
       --exp_name=fedprox --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --frac=0.1 

# MOON example
python src/run_main.py --model=resnet18 --dataset=cifar --alg=moon \
       --diric=0.01 --epochs=2000 --optimizer=adsgd --moon_mu=1 --moon_temperature=0.5 \
       --exp_name=moon --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --frac=0.1 

# Text classification example
python src/run_main.py --is_text=true --model=bert --dataset=agnews --alg=fedavg \
       --diric=1 --epochs=100 --optimizer=adsgd  \
       --exp_name=text --local_bs=16 --local_ep=1 \
       --num_users=50 --loss=cross_entropy --frac=0.1 

# CIFAR100 classification with a Resnet-50 using 8 different client optimizers
python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
       --diric=0.1 --epochs=2000 --optimizer=adsgd \
       --exp_name=CIFAR100/adsgd_amplpt1 --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --lr_amplifier=0.1 \
       --frac=0.1 

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
       --diric=0.1 --epochs=2000 --optimizer=sgd \
       --exp_name=CIFAR100/sgd_lrpt1_decay --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --momentum=0 --lr=0.1 \
       --frac=0.1 --lr_decay 

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
       --diric=0.1 --epochs=2000 --optimizer=sgd \
       --exp_name=CIFAR100/sgdM_lrpt1_decay --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --momentum=0.9 --lr=0.1 \
       --frac=0.1 --lr_decay 

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
       --diric=0.1 --epochs=2000 --optimizer=sgd \
       --exp_name=CIFAR100/sgd_lrpt1 --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --momentum=0 --lr=0.1 \
       --frac=0.1 

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
       --diric=0.1 --epochs=2000 --optimizer=sgd \
       --exp_name=CIFAR100/sgdM_lrpt1 --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --momentum=0.9 --lr=0.1 \
       --frac=0.1 

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
       --diric=0.1 --epochs=2000 --optimizer=adam \
       --exp_name=CIFAR100/adam --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --lr=0.001 \
       --frac=0.1 

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
       --diric=0.1 --epochs=2000 --optimizer=adagrad \
       --exp_name=CIFAR100/adagrad --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy --lr=0.01 \
       --frac=0.1 

python src/run_main.py --model=resnet50 --dataset=cifar100 --alg=fedavg \
       --diric=0.1 --epochs=2000 --optimizer=sps \
       --exp_name=CIFAR100/sps --local_bs=64 --local_ep=1 \
       --num_users=100 --loss=cross_entropy \
       --frac=0.1 



