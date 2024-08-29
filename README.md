# Federated-Learning with Auto-Tuned Clients

Pytorch implmentation of the paper [Adaptive Federated Learning with Auto-Tuned Clients](https://openreview.net/forum?id=g0mlwqs8pi), modified from
[this repository](https://github.com/AshwinRJ/Federated-Learning-PyTorch).

__Authors: Junhyung Lyle Kim, Mohammad Taha Toghani, Cesar A. Uribe, and Anastasios Kyrillidis.__

## Requirments
Required packages can be found in `requirments.txt`.

## Data
* Datasets will be automatically downloaded from torchvision datasets.
* Experiments are run on MNIST, Fashion MNIST, CIFAR10, CIFAR100 datasets for image classification tasks, and on Agnews and Dbpedia datasets for text classification tasks.

## Usage
* To run an image classification experiment on CIFAR100 dataset using ResNet-18:
```
python src/run_main.py --model=resnet18 --dataset=cifar100 --alg=fedavg \
--diric=1 --epochs=2000 --optimizer=adsgd --exp_name=exp1 --local_bs=64 --local_ep=1 
```
* To run a text classification experiment on Agnews dataset using DistilBERT:
```
python src/run_main.py --is_text=true --model=bert --dataset=agnews --alg=fedavg \
--diric=1 --epochs=2000 --optimizer=adsgd --exp_name=exp1 --local_bs=64 --local_ep=1 
```
There are other parameters---please refer to the options section below.
For more examples, please refer to `run.sh`.

## Options
Some details of the options are provided below. Please also refer to ```options.py```.

* ```--alg:```  Default: 'fedavg'. Options: 'fedavg', 'moon', 'fedprox', 'fedadam'.
* ```--optimizer:```  Default: 'adsgd'. Options: 'adsgd', 'sgd', 'adam', 'adagrad', 'sps'.
* ```--dataset:```  Default: 'cifar'. Options: 'mnist', 'fmnist', 'cifar', 'cifar100', 'agnews', 'dbpedia_14'.
* ```--model:```    Default: 'cnn'. Options: 'cnn', 'resnet18', 'resnet50', 'bert'.
* ```--epochs:```   Number of rounds of training.
* ```--verbose:```  Detailed log outputs. Set to 0 to deactivate.
* ```--diric:```    Dirichlet concentration parameter for generating non-iid federated data. The default is 0.1.
* ```--num_users:```Number of users. The default is 100.
* ```--frac:```     Fraction of participating users. The default is 0.1.
* ```--local_ep:``` Number of local training epochs in each user. The default is 1.
* ```--local_bs:``` Batch size of local updates in each user. The default is 64.
