#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    
    # federated arguments
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of participating clients')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size")
    parser.add_argument('--alg', type=str, default='fedavg',
                        help="fedavg or fedprox or moon or fedadam")
    
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for (use 0 for equal splits)')
    parser.add_argument('--diric', type=float, default=0.1,
                        help='dirichlet parameter for non-iid setting')
    parser.add_argument('--item_per_user', type=int, default=500,
                        help='items per user (default: 500)')    
    
    # Client optimizer arguments
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")

    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay', action='store_true',
                        help="flag to decay LR or not after 50th and 75th rounds")
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='SGD momentum (default: 0.0)')

    # Delta-SGD arguments
    parser.add_argument('--lr_amplifier', type=float, default=0.02,
                        help='ADSGD amplifier (default: 0.02)')
    parser.add_argument('--lr_damping', type=float, default=1,
                        help='ADSGD damping (default: 1)')
    parser.add_argument('--lr_eps', type=float, default=1e-8,
                        help='ADSGD amplifier (default: 1e-8)')
    
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    parser.add_argument('--is_text', type=str, default=None,
                        help="flag for text classification or not")
    
    # FedProx and Moon arguments
    parser.add_argument('--fedprox_mu', type=float, default=1.0,
                        help='mu for fedprox')
    parser.add_argument('--moon_mu', type=float, default=1.0,
                        help='mu for moon')
    parser.add_argument('--moon_temperature', type=float, default=0.5,
                        help='temperature for moon')
    
    # FedAdam arguments
    parser.add_argument('--global_lr', type=float, default=0.001,
                        help='server learning rate for FedAdam') 

    # other arguments
    parser.add_argument('--loss', type=str, default='cross_entropy',
                        help="which loss function; cross_entropy")
    parser.add_argument('--exp_name', type=str, default='experiment', help='directory of experiment') 
    parser.add_argument('--verbose', type=int, default=1, help='verbose')

    args = parser.parse_args()
    return args
