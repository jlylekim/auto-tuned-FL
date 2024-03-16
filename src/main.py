import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import os
import json

import torch
from tensorboardX import SummaryWriter


from update import LocalUpdate, test_inference
from cnn import CNNMnist, CNNCifar
from resnet import ResNet18, ResNet50
from utils import get_dataset, average_weights, exp_details, seed_everything, get_weights, weighted_average_weights

# for text classification
from transformers import AutoModelForSequenceClassification

def run(args, seed):
    start_time = time.time()

    exp_dir = os.path.join('logs', args.exp_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    logger = SummaryWriter(exp_dir)
    exp_details(args)

    seed_everything(seed)

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print("Using device {0!s}".format(device)) 

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNMnist(args=args) 
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
    
    elif args.model == 'resnet18':
        # ref: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        if args.dataset == 'cifar' or args.dataset == 'mnist':
            global_model = ResNet18(num_classes=10)
        elif args.dataset == 'cifar100':
            global_model = ResNet18(num_classes=100)
        
    elif args.model == 'resnet50':
        if args.dataset == 'cifar' or args.dataset == 'mnist':
            global_model = ResNet50(num_classes=10)
        elif args.dataset == 'cifar100':
            global_model = ResNet50(num_classes=100)

    elif args.model == 'bert':
        if args.dataset == 'agnews':
            global_model  = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=4)
        elif args.dataset == 'dbpedia_14':
            global_model  = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=14)

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    ##############################################
    if args.alg == 'fedadam':
        print('Running FedAdam with global step size:')
        print(args.global_lr)
        global_optimizer = torch.optim.Adam(global_model.parameters(), lr=args.global_lr)
    ##############################################

    # Training
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    print_every = 10

    # prev_nets for MOON
    prev_local_nets = {}

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        if (epoch+1) % print_every == 0:
            print(f'\n | Global Training Round : {epoch+1} |\n')

        ##############################################
        if args.alg == 'fedadam':
            global_optimizer.zero_grad()
        ##############################################

        logger.add_scalar('lr', args.lr, epoch)

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        if epoch == 1:
            print("Number of users: %d, frac: %f, num_users: %d" %(m, args.frac, args.num_users))
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # uncomment below to use different number of data per client
        # weight_by_num_data = get_weights(user_groups, idxs_users)

        # option to divide learning rate by 10 after 50th and 75th epochs for sgd
        if args.optimizer == 'sgd':
            if args.lr_decay: 
                if (epoch-1) == int(0.5 * args.epochs):
                    print("decaying learning rate of SGD(M) at epoch: %d" %(epoch-1))
                    args.lr /= 10 
                if (epoch-1) == int(0.75 * args.epochs):
                    print("decaying again learning rate of SGD(M) at epoch: %d" %(epoch-1))
                    args.lr /= 10 
        
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            
            if args.alg == 'fedprox':
                w, loss = local_model.update_weights_prox(
                    model=copy.deepcopy(global_model), global_round=epoch, mu=args.fedprox_mu)
            elif args.alg == 'fedavg' or args.alg == 'fedadam':
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch)
            elif args.alg == 'moon':
                # check prev_local_nets (dictionary) to see if w_t^i exist
                if idx in prev_local_nets:
                    prev_local_net = prev_local_nets[idx]
                else: 
                    prev_local_nets[idx] = global_model.state_dict()
                    prev_local_net = prev_local_nets[idx]
                # if so, use it. If not, use w_t^i <- global_model
                w, loss, w_prev = local_model.update_weights_moon(
                    model=copy.deepcopy(global_model), 
                    prev_local_weight = prev_local_net, 
                    global_round=epoch, 
                    mu=args.moon_mu,
                    temperature=args.moon_temperature
                    )
                # update the dictionary
                prev_local_nets[idx] = w_prev 
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        if args.alg == 'fedadam':
            global_weights = average_weights(local_weights)
            global_model_grad = copy.deepcopy(global_model) #copy placeholder of global_model
            global_model_grad.load_state_dict(global_weights) #load average of local gradients to dummy
            for w_glob, w_grad in zip(global_model.parameters(), global_model_grad.parameters()):
                w_glob.grad = w_glob.data - w_grad.data
            global_optimizer.step()
        else: 
            global_weights = average_weights(local_weights)
            
            # uncomment below to use different number of data per client
            # global_weights = weighted_average_weights(local_weights, weight_by_num_data)
            global_model.load_state_dict(global_weights)
        
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        
        logger.add_scalar('train-loss', loss_avg, epoch)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        test_acc_epoch, test_loss_epoch = test_inference(args, global_model, test_dataset)

        test_accuracy.append(test_acc_epoch)
        test_loss.append(test_loss_epoch)
        logger.add_scalar('test-acc', test_acc_epoch, epoch)
        logger.add_scalar('test-loss', test_loss_epoch, epoch)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            print('Test Accuracy: {:.2f}% \n'.format(100*test_accuracy[-1]))

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_accuracy[-1]))

    # Saving the objects test_loss and test_accuracy:
    file_name_test = '{}/test_{}_{}_{}_{}_C[{}]_diric[{}]_E[{}]_B[{}]_{}_{}.pkl'.\
        format(exp_dir, args.alg, args.dataset, args.model, args.epochs, args.frac, args.diric,
               args.local_ep, args.local_bs, args.optimizer, seed)

    with open(file_name_test, 'wb') as f:
        pickle.dump([test_loss, test_accuracy], f)

    # Saving the objects train_loss and train_accuracy:
    file_name_train = '{}/train_{}_{}_{}_{}_C[{}]_diric[{}]_E[{}]_B[{}]_{}_{}.pkl'.\
        format(exp_dir, args.alg, args.dataset, args.model, args.epochs, args.frac, args.diric,
               args.local_ep, args.local_bs, args.optimizer, seed)

    with open(file_name_train, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    return None


