from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import torch
import numpy as np, math
import nets, optimizers, stats
import argparse

parser = argparse.ArgumentParser(description='Train a model on cifar10 with either mcmc or stochastic gradient based approach')
parser.add_argument('--batch_size',
                    help='batch size',
                    default=50000, type=int)
parser.add_argument('--lr',
                    help='learning rate for the gradient optimization',
                    default=0.001, type=float)
parser.add_argument('--student_variance_prop',
                    help='Variance of the student law used to generate proposal within the mcmc',
                    default=0.001, type=float)
parser.add_argument('--student_variance_prior',
                    help='Variance of the student law used as a prior on the network parameters',
                    default=0.001, type=float)
parser.add_argument('--epochs',
                    help='number of epochs : pass all the data in the training loop (in the case of mcmc, each data is used iter_mcmc iterations)',
                    default=400, type=int)
parser.add_argument('--iter_mcmc',
                    help='number of iterations for the mcmc algorithm',
                    default=50, type=int)
parser.add_argument('--lambda', dest='lamb',
                    help='value for the lambda parameter which is a tradeoff between the data and the student regularisation prior',
                    default=10000, type=float)
parser.add_argument('--exp_name', dest='exp_name',
                    help='basename for the json file in which the accuracy and the loss will be recorded',
                    default='results', type=str)
parser.add_argument('--data_folder',
                    help='data which contain the cifar10 dataset. Pytorch will download it if it does not exist',
                    default="/home/paul/data/pytorch_cifar10", type=str)
parser.add_argument('--hidden_size',
                    help='number of neurones in the hidden layer',
                    default=100, type=int)
parser.add_argument('--use_gradient',
                    help='if passed, the program will used sgd optimization',
                    action="store_true")
parser.add_argument('--num_xp',
        help='number of times the experiments will be reproduced with different initialisation',
        default=10, type=int)

args = parser.parse_args()

training_data = datasets.CIFAR10(root=args.data_folder,
    train=True,
    download=True,
    transform=ToTensor())

test_data = datasets.CIFAR10(root=args.data_folder,
    train=False,
    download=True,
    transform=ToTensor())

batch_size = args.batch_size 
lr = args.lr #0.001 # learning rate for the gradient descent
student_variance_prop = args.student_variance_prop #0.00000000001 # variance for the student distribution
student_variance_prior = args.student_variance_prior #0.00000000001 # variance for the student distribution
st_prop = stats.Student(student_variance_prop, 0)
st_prior = stats.Student(student_variance_prior, 0)
loss_fn = nets.my_mse_loss
iter_mcmc = args.iter_mcmc #50
epochs = args.epochs #1000
lamb = args.lamb #1000000 
exp_name=args.exp_name
use_gradient = args.use_gradient
num_xp = args.num_xp

if use_gradient and batch_size > 1000:
    print("!!!!!!!!!!!!!!!!!!!!!!")
    print("WARNING, you are using SGD and the batch size is ", batch_size)
    print("This might be too high, consider the option --batch_size 64")
    print()
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

input_size = 3072 
hidden_size = 300
output_size = 10
for i in range(num_xp):
    model = nets.One_layer(input_size, output_size, act='relu')
    results = {}
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        if use_gradient:
            optimizers.train_1_epoch(train_dataloader, model, loss_fn, lr = lr)
        else:
            acceptance_ratio = optimizers.train_1_epoch(train_dataloader, model, loss_fn, proposal=st_prop, prior=st_prior, lamb=lamb, iter_mcmc=iter_mcmc)
        loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
        results[t] = {}
        if not use_gradient:
            print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n Acceptance ratio: {acceptance_ratio:>2f}\n")
            results[t]['accept_ratio'] = acceptance_ratio
        else:
            print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")

        results[t]['train'] = {'training loss' : loss, 'training accuracy' : accuracy }
        loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
        print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
        results[t]['test'] = {'test loss' : loss, 'testing accuracy' : accuracy }
        if use_gradient:
          json.dump(results, open(exp_name+'_'+str(lr)+'_'+str(i)+'.json','w'))
        else:
          json.dump(results, open(exp_name+'_'+str(lamb)+'_'+str(i)+'.json','w'))
