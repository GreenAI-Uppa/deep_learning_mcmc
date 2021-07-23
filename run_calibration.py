import time
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import torch
import numpy as np, math
from deep_learning_mcmc import nets, optimizers, stats
import argparse

parser = argparse.ArgumentParser(description='Train a model on cifar10 with either mcmc or stochastic gradient based approach')
parser.add_argument('--data_folder',
                    help='absolute path toward the data folder which contains the cifar10 dataset. Pytorch will download it if it does not exist',
                    required=True, type=str)
args = parser.parse_args()

training_data = datasets.CIFAR10(root=args.data_folder,
    train=True,
    download=True,
    transform=ToTensor())

test_data = datasets.CIFAR10(root=args.data_folder,
    train=False,
    download=True,
    transform=ToTensor())


#setting the batch size
batch_size = 50000

# getting the data
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=20)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=20)

loss_fn = nets.my_mse_loss

nb_filters = 64
for nn_size in [300,"full"]:
    # setting the optimizer
    for variance_prop in [0.0000000001,0.000000001]:
        st_prop = stats.Student(variance_prop)
        for variance_prior in [0.0000000001,0.00000001]:
            st_prior = stats.Student(variance_prior)
            for lambda_ in [1000000,100000]:
                optimizer = optimizers.MCMCOptimizer(st_prop, nn_size=nn_size,iter_mcmc=20000, lamb=lambda_, prior=st_prior)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                ####
                print('--Student analysis--')
                print('nn_size=',nn_size,'variance_prop=',variance_prop,'variance_prior=',variance_prior,'lambda_=',lambda_)
                start_epoch = time.time()
                model = nets.ConvNet(nb_filters)#particular convnet with one convolution and one fully connected layer
                model = model.to(device)
                acceptance_ratio_f, acceptance_ratio_l = optimizer.train_1_epoch(train_dataloader, model, loss_fn, optimizer)
                end_epoch = time.time() 
                ####
                print('training time',time.time() - start_epoch)
                loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
                print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n Acceptance ratios: {acceptance_ratio_f, acceptance_ratio_l}")
                loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
                print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")    


for nn_size in [300,"full"]:
    # setting the optimizer
    for variance_prop in [0.0000000001,0.000000001]:
        st_prop = stats.Student(variance_prop)
        for variance_prior in [0.0000000001,0.00000001]:
            st_prior = stats.HeavyTail(variance_prior,10)
            for lambda_ in [1000000,100000]:
                optimizer = optimizers.MCMCOptimizer(st_prop, nn_size=nn_size,iter_mcmc=20000, lamb=lambda_, prior=st_prior)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                ####
                print('--HeavyTail R=10 analysis--')
                print('nn_size=',nn_size,'variance_prop=',variance_prop,'variance_prior=',variance_prior,'lambda_=',lambda_)
                start_epoch = time.time()
                model = nets.ConvNet(nb_filters)#particular convnet with one convolution and one fully connected layer
                model = model.to(device)
                acceptance_ratio_f, acceptance_ratio_l = optimizer.train_1_epoch(train_dataloader, model, loss_fn, optimizer)
                end_epoch = time.time() 
                ####
                print('training time',time.time() - start_epoch)
                loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
                print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n Acceptance ratios: {acceptance_ratio_f, acceptance_ratio_l}")
                loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
                print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")    



for nn_size in [300,"full"]:
    # setting the optimizer
    for variance_prop in [0.0000000001,0.000000001]:
        st_prop = stats.Student(variance_prop)
        for variance_prior in [0.0000000001,0.00000001]:
            st_prior = stats.HeavyTail(variance_prior,100)
            for lambda_ in [1000000,100000]:
                optimizer = optimizers.MCMCOptimizer(st_prop, nn_size=nn_size,iter_mcmc=20000, lamb=lambda_, prior=st_prior)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                ####
                print('--HeavyTail R=100 analysis--')
                print('nn_size=',nn_size,'variance_prop=',variance_prop,'variance_prior=',variance_prior,'lambda_=',lambda_)
                start_epoch = time.time()
                model = nets.ConvNet(nb_filters)#particular convnet with one convolution and one fully connected layer
                model = model.to(device)
                acceptance_ratio_f, acceptance_ratio_l = optimizer.train_1_epoch(train_dataloader, model, loss_fn, optimizer)
                end_epoch = time.time() 
                ####
                print('training time',time.time() - start_epoch)
                loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
                print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n Acceptance ratios: {acceptance_ratio_f, acceptance_ratio_l}")
                loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
                print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")    
