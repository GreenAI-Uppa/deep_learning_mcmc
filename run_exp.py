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
parser.add_argument('--batch_size',
                    help='batch size',
                    default=50000, type=int)
parser.add_argument('--lr',
                    help='learning rate for the gradient optimization',
                    default=0.001, type=float)
parser.add_argument('--student_variance_prior',
                    help='Variance of the student law used as a prior on the network parameters',
                    default=0.00000000001, type=float)
parser.add_argument('--student_variance_prop',
                    help='Variance of the student law used to generate proposal within the mcmc',
                    default=0.00000000001, type=float)
parser.add_argument('--epochs',
                    help='number of epochs : pass all the data in the training loop (in the case of mcmc, each data is used iter_mcmc iterations)',
                    default=64, type=int)
parser.add_argument('--iter_mcmc',
                    help='number of iterations for the mcmc algorithm',
                    default=50, type=int)
parser.add_argument('--lambda', dest='lamb',
                    help='value for the lambda parameter which is a tradeoff between the data and the student regularisation prior',
                    default=1000000, type=float)
parser.add_argument('--exp_name', dest='exp_name',
                    help='basename for the json file in which the accuracy and the loss will be recorded',
                    default='results', type=str)
parser.add_argument('--hidden_size',
                    help='size of the hidden layer in the mlp',
                    default=None, type=int)
parser.add_argument('--data_folder',
                    help='absolute path toward the data folder which contains the cifar10 dataset. Pytorch will download it if it does not exist',
                    required=True, type=str)
parser.add_argument('--optimizer',
                    help='"grad or mcmc"',
                    default="mcmc", type=str)
parser.add_argument('--max_data_points',
                    help='maximum number of data points used',
                    default=None, type=int)
parser.add_argument('--config_file',
                    help='json file containing various setups (learning rate, mcmc iterations, variance of the priors and the proposal, batch size,...)',
                    default=None, type=str)



args = parser.parse_args()
training_data = datasets.CIFAR10(root=args.data_folder,
    train=True,
    download=True,
    transform=ToTensor())

test_data = datasets.CIFAR10(root=args.data_folder,
    train=False,
    download=True,
    transform=ToTensor())

params = vars(args)
# overriding the parameters with the json file config if it exists
if params['config_file'] is not None:
    json_params = json.load(open(params['config_file']))
    for k,v in json_params.items():
        params[k] = v


print(params)
batch_size = params['batch_size']
if params['optimizer']=='grad' and params['batch_size'] > 1000:
    print("!!!!!!!!!!!!!!!!!!!!!!")
    print("WARNING, you are using SGD and the batch size is ", batch_size)
    print("This might be too high, consider the option --batch_size 64")
    print()
# getting the data
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=20)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=20)


# setting the model
input_size = training_data.data.shape[1] * training_data.data.shape[2] * training_data.data.shape[3]
output_size = len(training_data.classes)
if params['hidden_size'] is None:
    layer_sizes = [input_size, output_size]
else:
    layer_sizes = [input_size, params['hidden_size'], output_size]
model = nets.MLP(layer_sizes, act='relu')

# setting the optimizer
use_gradient = params['optimizer'] == 'grad'
if use_gradient:
    optimizer = optimizers.GradientOptimizer(lr=params['lr'])
else:
    st_prop = stats.Student(params['student_variance_prop'])
    st_prior = stats.Student(params['student_variance_prior'])
    optimizer = optimizers.MCMCOptimizer(st_prop, iter_mcmc=params['iter_mcmc'], lamb=params['lamb'], prior=st_prior)


exp_name = params['exp_name']
if use_gradient:
    exp_name = exp_name+'_'+str(params['lr'])
else:
    exp_name = exp_name+'_'+str(params['lamb'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Using {} device'.format(device))
model = model.to(device)

results = {}
epochs = params['epochs']
loss_fn = nets.my_mse_loss
start_all = time.time()
for t in range(epochs):
    start_epoch = time.time()
    print(f"Epoch {t+1} is running\n--------------------- duration = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)) +"----------")
    if use_gradient:
        optimizer.train_1_epoch(train_dataloader, model, loss_fn)
    else:
        acceptance_ratio = optimizer.train_1_epoch(train_dataloader, model, loss_fn, optimizer)
    results[t] = {}
    end_epoch = time.time() 
    results[t]['training time'] = time.time() - start_epoch
    loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
    if use_gradient:
        print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    else:
        print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n Acceptance ratio: {acceptance_ratio:>2f}")
    if not use_gradient:
        results[t]['accept_ratio'] = acceptance_ratio
    results[t]['train'] = {'training loss' : loss, 'training accuracy' : accuracy }
    loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    results[t]['test'] = {'test loss' : loss, 'testing accuracy' : accuracy }
    results[t]['training time'] = time.time() - end_epoch 
    json.dump(results, open(exp_name+'.json','w'))
    torch.save(model, exp_name+'.th')
