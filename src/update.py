#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import copy 
from adsgd import Adsgd 
from sps import Sps 

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs, is_text=None):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.is_text=is_text

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        ########################################################
        # below: only used for text classification
        if self.is_text == 'true':
            data = self.dataset[self.idxs[item]]
            return {'input_ids':torch.tensor(data['input_ids']), 
                    'attention_mask': torch.tensor(data['attention_mask'])}, torch.tensor(data['label'])
        ########################################################
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        # self.trainloader, self.validloader, self.testloader = self.train_val_test(
        #     dataset, list(idxs))
        self.trainloader, self.testloader = self.train_val_test(dataset, list(idxs), is_text=self.args.is_text)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # Default criterion set to cross entropy loss function
        if self.args.loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss().to(self.device) 


    def train_val_test(self, dataset, idxs, is_text=None):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train and test (90, 10)
        idxs_train = idxs[:int(0.9*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        # 
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train, is_text),
                                 batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                #  batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test, is_text),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        # return trainloader, validloader, testloader
        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        #################################################
        # model_prev was being used for debugging for FedAdam with Taha
        # model_prev = copy.deepcopy(model)
        #################################################
        model.train()
        epoch_loss = []

        # Set client optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'sps':
            optimizer = Sps(model.parameters(), adapt_flag='constant')
        elif self.args.optimizer == 'adsgd':
            prev_net = copy.deepcopy(model)
            prev_net.to(self.device) 
            prev_net.train() 

            optimizer = Adsgd(model.parameters(), 
                              amplifier=self.args.lr_amplifier, 
                              damping=self.args.lr_damping, 
                              weight_decay=0, 
                              eps=self.args.lr_eps)
            
            prev_optimizer = Adsgd(prev_net.parameters(), weight_decay=0)

        if self.args.optimizer == 'adsgd':
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    optimizer.zero_grad()
                    prev_optimizer.zero_grad()
                    ########################################################
                    # below: only used for text classification
                    if self.args.is_text == 'true':
                        batch = {k: v.to(self.device) for k, v in images.items()} #converts everything from cpu to cuda
                        labels = labels.to(self.device) # adding labels in the dictinary
                        log_probs_prev = prev_net(**batch)['logits']
                        log_probs = model(**batch)['logits'] # forward pass + computes loss function internally
                    ########################################################
                    else: 
                        images, labels = images.to(self.device), labels.to(self.device)
                        log_probs_prev = prev_net(images)
                        log_probs = model(images)

                    prev_loss = self.criterion(log_probs_prev, labels)
                    prev_loss.backward()

                    loss = self.criterion(log_probs, labels)
                    loss.backward()

                    optimizer.compute_dif_norms(prev_optimizer)
                    prev_net.load_state_dict(model.state_dict()) 

                    optimizer.step()
                    
                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                            global_round, iter, loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        elif self.args.optimizer == 'sps':
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    model.zero_grad()
                    ########################################################
                    # below: only used for text classification
                    if self.args.is_text == 'true':
                        batch = {k: v.to(self.device) for k, v in images.items()} #converts everything from cpu to cuda
                        labels = labels.to(self.device) # adding labels in the dictinary
                        log_probs = model(**batch)['logits'] # forward pass + computes loss function internally
                    ########################################################
                    else:
                        images, labels = images.to(self.device), labels.to(self.device)
                        log_probs = model(images)

                    loss = self.criterion(log_probs, labels)
                    loss.backward()
                    optimizer.step(loss=loss)

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                            global_round, iter, loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        else: 
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    model.zero_grad()
                    ########################################################
                    # below: only used for text classification
                    if self.args.is_text == 'true':
                        batch = {k: v.to(self.device) for k, v in images.items()} #converts everything from cpu to cuda
                        labels = labels.to(self.device) # adding labels in the dictinary
                        log_probs = model(**batch)['logits'] # forward pass + computes loss function internally
                    ########################################################
                    else:
                        images, labels = images.to(self.device), labels.to(self.device)
                        log_probs = model(images)

                    loss = self.criterion(log_probs, labels)

                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                            global_round, iter, loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        #################################################################
        # model_prev is a dummy whose weight (.data) is the gradient
        # from k=0 to K-1
        # for w, w_prev in zip(model.parameters(), model_prev.parameters()):
            # w_prev.data = w_prev.data - w.data
        #################################################################

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        #################################################################
        # return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model_prev.state_dict()
        #################################################################

    # modified from https://github.com/Xtra-Computing/NIID-Bench
    def update_weights_prox(self, model, global_round, mu):
        temp = copy.deepcopy(model)
        global_weight_collector = list(temp.to(self.device).parameters())
        
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'sps':
            optimizer = Sps(model.parameters(), adapt_flag='constant')
        elif self.args.optimizer == 'adsgd':
            prev_net = copy.deepcopy(model)
            prev_net.to(self.device) 
            prev_net.train() 

            optimizer = Adsgd(model.parameters(), 
                              amplifier=self.args.lr_amplifier, 
                              damping=self.args.lr_damping, 
                              weight_decay=0, 
                              eps=self.args.lr_eps)
            
            prev_optimizer = Adsgd(prev_net.parameters(), weight_decay=0)

        if self.args.optimizer == 'adsgd':
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    prev_optimizer.zero_grad()

                    log_probs_prev = prev_net(images)
                    prev_loss = self.criterion(log_probs_prev, labels)

                    # modification for fedprox
                    prev_fed_prox_reg = 0.0
                    for param_index, param in enumerate(prev_net.parameters()):
                        prev_fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                    prev_loss += prev_fed_prox_reg

                    prev_loss.backward()

                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)

                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                    loss += fed_prox_reg
                    
                    loss.backward()

                    optimizer.compute_dif_norms(prev_optimizer)
                    prev_net.load_state_dict(model.state_dict()) 

                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                            global_round, iter, loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        elif self.args.optimizer == 'sps':
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    log_probs = model(images)
                    loss = self.criterion(log_probs, labels)

                    # modification for fedprox
                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                    loss += fed_prox_reg

                    loss.backward()
                    optimizer.step(loss=loss)

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                            global_round, iter, loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        else: 
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    images, labels = images.to(self.device), labels.to(self.device)

                    model.zero_grad()
                    log_probs = model(images) 
                    loss = self.criterion(log_probs, labels)

                    # modification for fedprox
                    fed_prox_reg = 0.0
                    for param_index, param in enumerate(model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index]))**2)
                    loss += fed_prox_reg

                    loss.backward()
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                            global_round, iter, loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # modified from https://github.com/Xtra-Computing/NIID-Bench
    def update_weights_moon(self, model, prev_local_weight, global_round, mu, temperature):
        prev_glob_net = model.state_dict()
        global_net = copy.deepcopy(model) # copy model to use as global net

        prev_local_model = copy.deepcopy(model)
        prev_local_model.load_state_dict(prev_local_weight)

        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=self.args.momentum)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'adagrad':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'sps':
            optimizer = Sps(model.parameters(), adapt_flag='constant')
        elif self.args.optimizer == 'adsgd':
            prev_net = copy.deepcopy(model) 
            prev_net.to(self.device) 
            prev_net.train() 

            optimizer = Adsgd(model.parameters(), 
                              amplifier=self.args.lr_amplifier, 
                              damping=self.args.lr_damping, 
                              weight_decay=0, 
                              eps=self.args.lr_eps)
            
            prev_optimizer = Adsgd(prev_net.parameters(), weight_decay=0)

        # modification for MOON
        prev_local_model.to(self.device) 
        cos=torch.nn.CosineSimilarity(dim=-1)
        # modification for MOON

        if self.args.optimizer == 'adsgd':
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (x, target) in enumerate(self.trainloader):
                    x, target = x.to(self.device), target.to(self.device)

                    optimizer.zero_grad()

                    x.requires_grad = False
                    target.requires_grad = False 
                    target = target.long()
                    
                    pro1, out = model(x, two_output = True)
                    pro2, _ = global_net(x, two_output = True)

                    posi = cos(pro1, pro2)
                    logits = posi.reshape(-1,1)

                    # modification for moon: assuming only using 1 prev model
                    prev_local_model.to(self.device)
                    pro3, _ = prev_local_model(x, two_output = True)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    logits /= temperature
                    labels = torch.zeros(x.size(0)).cuda().long()

                    loss2 = mu * self.criterion(logits, labels)
                    loss1 = self.criterion(out, target)
                    loss = loss1 + loss2

                    ##############################
                    prev_optimizer.zero_grad()
                    pro1_prev, out_prev = prev_net(x, two_output = True)
                    posi_prev = cos(pro1_prev, pro2)
                    logits_prev = posi_prev.reshape(-1, 1)
                    nega_prev = cos(pro1_prev, pro3)
                    logits_prev = torch.cat((logits_prev, nega_prev.reshape(-1,1)), dim=1)
                    logits_prev /= temperature
                    loss2_prev = mu * self.criterion(logits_prev, labels)
                    loss1_prev = self.criterion(out_prev, target)
                    loss_prev = loss1_prev + loss2_prev
                    loss_prev.backward(retain_graph=True)
                    ##############################

                    loss.backward() 

                    optimizer.compute_dif_norms(prev_optimizer)
                    prev_net.load_state_dict(model.state_dict())

                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                            global_round, iter, loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        elif self.args.optimizer == 'sps':
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (x, target) in enumerate(self.trainloader):
                    x, target = x.to(self.device), target.to(self.device)

                    model.zero_grad()
                    x.requires_grad = False
                    target.requires_grad = False 
                    target = target.long()
                    
                    pro1, out = model(x, two_output = True)
                    pro2, _ = global_net(x, two_output = True)

                    posi = cos(pro1, pro2)
                    logits = posi.reshape(-1,1)

                    # modification for moon: assuming only using 1 prev model
                    prev_local_model.to(self.device)
                    pro3, _ = prev_local_model(x, two_output = True)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    logits /= temperature
                    labels = torch.zeros(x.size(0)).cuda().long() 

                    loss2 = mu * self.criterion(logits, labels)

                    loss1 = self.criterion(out, target)
                    loss = loss1 + loss2

                    loss.backward() 
                    optimizer.step(loss=loss)

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                            global_round, iter, loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        else: 
            for iter in range(self.args.local_ep):
                batch_loss = []
                for batch_idx, (x, target) in enumerate(self.trainloader):
                    x, target = x.to(self.device), target.to(self.device)

                    model.zero_grad()
                    x.requires_grad = False
                    target.requires_grad = False 
                    target = target.long()
                    
                    pro1, out = model(x, two_output = True)
                    pro2, _ = global_net(x, two_output = True)

                    posi = cos(pro1, pro2)
                    logits = posi.reshape(-1,1)

                    # modification for moon: assuming only using 1 prev model
                    prev_local_model.to(self.device)
                    pro3, _ = prev_local_model(x, two_output = True)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                    logits /= temperature
                    labels = torch.zeros(x.size(0)).cuda().long()

                    loss2 = mu * self.criterion(logits, labels)

                    loss1 = self.criterion(out, target)
                    loss = loss1 + loss2

                    loss.backward() 
                    optimizer.step()

                    if self.args.verbose and (batch_idx % 10 == 0):
                        print('| Global Round : {} | Local Epoch : {} | Loss: {:.6f}'.format(
                            global_round, iter, loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), prev_glob_net
    


    def inference(self, model):
        """ Returns the inference accuracy and loss.  
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            ########################################################
            # below: only used for text classification
            if self.args.is_text == 'true':
                batch = {k: v.to(self.device) for k, v in images.items()} #converts everything from cpu to cuda
                labels = labels.to(self.device) # adding labels in the dictinary
                outputs = model(**batch)['logits'] # forward pass + computes loss function internally
            ########################################################
            else:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
            # Inference
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    if args.loss == 'cross_entropy':
        criterion = nn.CrossEntropyLoss().to(device) 

    if args.is_text == 'true':
        testloader = DataLoader(DatasetSplit(test_dataset, np.arange(len(test_dataset)), args.is_text),
                                    batch_size=16, shuffle=False)
    else:
        testloader = DataLoader(test_dataset, batch_size=128,
                                shuffle=False)
    

    for batch_idx, (images, labels) in enumerate(testloader):
        if args.is_text == 'true':
            batch = {k: v.to(device) for k, v in images.items()} #converts everything from cpu to cuda
            labels = labels.to(device) # adding labels in the dictinary
            outputs = model(**batch)['logits'] # forward pass + computes loss function internally
        else:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
        # Inference
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()


        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
