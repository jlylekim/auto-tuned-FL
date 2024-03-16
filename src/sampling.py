#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# source: https://github.com/huerdong/FedVert-Experiments
def dirichlet_sampler_idsize(dataset, num_users, diric=100, items_per_user=500):

    dataset_size = len(dataset)
    num_classes = len(dataset.classes)
    
    # Assume data is initially even distributed
    prior_distribution = [1 for i in range(num_classes)]
    distributions = np.random.dirichlet(diric * np.array(prior_distribution), num_users).transpose()

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(dataset_size)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # for plotting
    # hsv = plt.get_cmap('tab10')  
    # color_num = num_classes
    # plot_colors = hsv(np.linspace(0, 1.0, color_num))
    # space = [0.0 for i in range(num_users)]
    # for i in range(num_classes):
    #     plt.barh(range(num_users), distributions[i], left=space, color=plot_colors[i])
    #     space += distributions[i]
    # # plt.savefig(f'./out/{dataset.name}_users{num_users}_dir{diric}_distribution_idsize.png')


    distributions = distributions.transpose()

    for i in range(num_users):
        images_distribution = np.round((items_per_user * distributions[i])).astype(int)
        images_distribution[-1] = max(items_per_user - sum(images_distribution[0:-1]), 0) # Maybe we'll get extras but it is fine
        for j in range(len(images_distribution)):
            idxs_set = idxs_labels[0, np.where(idxs_labels[1, :] == j)][0].tolist()
            # replace=True to ensure each user has enough number of classes
            dict_users[i] = np.concatenate((dict_users[i], np.random.choice(idxs_set, images_distribution[j], replace=True)), axis=0)
    return dict_users


def dirichlet_sampler_idsize_text(dataset, num_users, diric=100, items_per_user=500):

    dataset_size = len(dataset)
    num_classes = len(dataset.classes)
    
    # Assume data is initially even distributed
    prior_distribution = [1 for i in range(num_classes)]
    distributions = np.random.dirichlet(diric * np.array(prior_distribution), num_users).transpose()

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(dataset_size)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # for plotting
    # hsv = plt.get_cmap('tab10')  
    # color_num = num_classes
    # plot_colors = hsv(np.linspace(0, 1.0, color_num))
    # space = [0.0 for i in range(num_users)]
    # for i in range(num_classes):
    #     plt.barh(range(num_users), distributions[i], left=space, color=plot_colors[i])
    #     space += distributions[i]
    # plt.savefig(f'./out/{dataset.name}_users{num_users}_dir{diric}_distribution_idsize.png')

    distributions = distributions.transpose()

    for i in range(num_users):
        images_distribution = np.round((items_per_user * distributions[i])).astype(int)
        images_distribution[-1] = max(items_per_user - sum(images_distribution[0:-1]), 0) # Maybe we'll get extras but it is fine
        for j in range(len(images_distribution)):
            idxs_set = idxs_labels[0, np.where(idxs_labels[1, :] == j)][0].tolist()
            # replace=True to ensure each user has enough number of classes
            dict_users[i] = np.concatenate((dict_users[i], np.random.choice(idxs_set, images_distribution[j], replace=True)), axis=0)

    return dict_users


def dirichlet_sampler_nonidsize(dataset, num_users, diric=100, items_per_user=500):
    dataset_size = len(dataset)
    num_classes = len(dataset.classes)

    prior_distribution = [1 for _ in range(num_classes)]
    distributions = np.random.dirichlet(diric * np.array(prior_distribution), num_users).transpose()

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(dataset_size)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    hsv = plt.get_cmap('hsv')
    color_num = num_classes
    plot_colors = hsv(np.linspace(0, 1.0, color_num))
    space = [0.0 for _ in range(num_users)]
    for i in range(num_classes):
        plt.barh(range(num_users), distributions[i], left=space, color=plot_colors[i])
        space += distributions[i]
    plt.savefig(f'./{dataset.name}_users{num_users}_dir{diric}_distribution_idsize.png')

    distributions = distributions.transpose()

    items_per_user_list = []
    for _ in range(num_users):
        lower_bound = int(0.2 * items_per_user)
        upper_bound = items_per_user
        items = np.random.randint(lower_bound, upper_bound + 1)
        items_per_user_list.append(items)

    for i in range(num_users):
        images_distribution = np.round((items_per_user_list[i] * distributions[i])).astype(int)
        images_distribution[-1] = max(items_per_user_list[i] - sum(images_distribution[0:-1]), 0)
        for j in range(len(images_distribution)):
            idxs_set = idxs_labels[0, np.where(idxs_labels[1, :] == j)][0].tolist()
            dict_users[i] = np.concatenate((dict_users[i], np.random.choice(idxs_set, images_distribution[j], replace=True)), axis=0)

    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_iid(dataset_train, num)
