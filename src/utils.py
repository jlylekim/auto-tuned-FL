#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
import os
import random
import copy
import torch
from torchvision import datasets, transforms
from sampling import dirichlet_sampler_idsize, dirichlet_sampler_nonidsize, dirichlet_sampler_idsize_text

from datasets import load_dataset #library for automatic dataload
from transformers import AutoTokenizer #transformers is hugging face's library

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '/data/cifar10'
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ]) 
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
            ])    
        
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train)
        
        train_dataset.name = 'CIFAR10'

        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test)

        # sample training data amongst users
        if args.unequal:
            # Chose uneuqal splits for every user
            print(f'trying diric with param {args.diric} with unequal sized local data') 
            user_groups = dirichlet_sampler_nonidsize(train_dataset, args.num_users, args.diric, args.item_per_user)
        elif args.diric is not None:
            print(f'trying diric with param {args.diric}') 
            user_groups = dirichlet_sampler_idsize(train_dataset, args.num_users, args.diric, args.item_per_user)
        else:
            raise NotImplementedError()

    elif args.dataset == 'cifar100':
        data_dir = '/data/cifar100'
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
        ])

        train_dataset = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train)
        
        train_dataset.name = 'CIFAR100'

        test_dataset = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test)
        
        # sample training data amongst users
        if args.unequal:
            # raise NotImplementedError()
            print(f'trying diric with param {args.diric} with unequal sized local data') 
            user_groups = dirichlet_sampler_nonidsize(train_dataset, args.num_users, args.diric, args.item_per_user)
        elif args.diric is not None:
            print(f'trying diric with param {args.diric}') 
            user_groups = dirichlet_sampler_idsize(train_dataset, args.num_users, args.diric, args.item_per_user)
        else:
            raise NotImplementedError()

    
    elif args.dataset == 'agnews':
        dataset = load_dataset("ag_news")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)


        train_dataset = tokenized_dataset['train']        
        train_dataset.name = 'agnews'

        test_dataset = tokenized_dataset['test']
        
        train_dataset.classes = list(set(train_dataset['label']))
        train_dataset.targets = train_dataset['label']

        if args.unequal:
            raise NotImplementedError()
        elif args.diric is not None:
            print(f'trying diric with param {args.diric}') 
            user_groups = dirichlet_sampler_idsize_text(train_dataset, args.num_users, args.diric, args.item_per_user)
        else:
            raise NotImplementedError()        

    elif args.dataset == 'dbpedia_14':
        dataset = load_dataset("dbpedia_14")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        def tokenize_function(examples):
            return tokenizer(examples["content"], padding="max_length", truncation=True)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)


        train_dataset = tokenized_dataset['train']        
        train_dataset.name = 'dbpedia_14'

        test_dataset = tokenized_dataset['test']

        test_dataset = test_dataset.select(np.arange(0, 70000, 10)) 
        
        train_dataset.classes = list(set(train_dataset['label']))
        train_dataset.targets = train_dataset['label']

        if args.unequal:
            raise NotImplementedError()
        elif args.diric is not None:
            print(f'trying diric with param {args.diric}') 
            user_groups = dirichlet_sampler_idsize_text(train_dataset, args.num_users, args.diric, args.item_per_user)
        else:
            raise NotImplementedError()

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '/data/mnist'
        elif args.dataset == 'fmnist':
            data_dir = '/data/fmnist'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        if args.dataset=='mnist':
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)
            train_dataset.name = 'MNIST'
            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                        transform=apply_transform)
        elif args.dataset=='fmnist':
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                        transform=apply_transform)
            train_dataset.name = 'FMNIST'
            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                        transform=apply_transform)

        # sample training data amongst users
        if args.unequal:
            raise NotImplementedError()
        elif args.diric is not None:
            print(f'trying diric with param {args.diric}') 
            user_groups = dirichlet_sampler_idsize(train_dataset, args.num_users, args.diric, args.item_per_user)
        else:
            raise NotImplementedError()

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    # Returns the average of the weights.
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def weighted_average_weights(w, weights):
    # Returns the weighted average of the weights.
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
            w_avg[key] += weights[i] * w[i][key]
        w_avg[key] = torch.div(w_avg[key], sum(weights))
    return w_avg

def get_weights(dict_users, sampled_users): 
    weights = []
    for i in sampled_users:
        local_data_length = len(dict_users[i])
        weights.append(local_data_length)
    return weights

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Loss     : {args.loss}')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning rate  : {args.lr}')
    print(f'    Momentum  : {args.momentum}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Algorithm  : {args.alg}')
    print(f'    Diric concentration param  : {args.diric}')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Number of users  : {args.num_users}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

# source: https://github.com/ymalitsky/adaptive_GD/blob/master/pytorch/utils.py
def seed_everything(seed=1029):
    '''
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True