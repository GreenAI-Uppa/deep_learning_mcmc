import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import torch
import numpy as np, math
from deep_learning_mcmc import nets, optimizers, stats
import argparse
from torchvision.datasets import MNIST

parser = argparse.ArgumentParser(description='Train a model on cifar10 or mnist with either mcmc or stochastic gradient based approach')
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
parser.add_argument('--data_folder',
                    help='absolute path toward the data folder which contains the cifar10 dataset. Pytorch will download it if it does not exist',
                    required=True, type=str)
parser.add_argument('--sparse_threshold',
                    help='threshold to evaluate test error of a sparse version of the ConvNet',
                    default=0.01,type=float)
parser.add_argument('--optimizer',
                    help='"grad or mcmc"',
                    default="mcmc", type=str)
parser.add_argument('--max_data_points',
                    help='maximum number of data points used',
                    default=None, type=int)
parser.add_argument('--config_file',
                    help='json file containing various setups (learning rate, mcmc iterations, variance of the priors and the proposal, batch size,...)',
                    default=None, type=str)
parser.add_argument('--dataset',
                    help='dataset used for experience',
                    default='CIFAR10', type=str)

args = parser.parse_args()
params = vars(args)
# overriding the parameters with the json file config if it exists
if params['config_file'] is not None:
    json_params = json.load(open(params['config_file']))
    for k,v in json_params.items():
        params[k] = v

dataset = params['dataset']

if dataset == 'MNIST':
    print('MNIST DATASET STUDY')
    transform = transforms.Compose([transforms.ToTensor()])
    training_data = MNIST(root = '../dataMNIST', train=True, download=True, transform=transform)
    test_data = MNIST(root = '../dataMNIST', train=False, download=True, transform=transform)
else:
    training_data = datasets.CIFAR10(root=args.data_folder,
        train=True,
        download=True,
        transform=ToTensor())

    test_data = datasets.CIFAR10(root=args.data_folder,
        train=False,
        download=True,
        transform=ToTensor())

examples = enumerate(training_data)
batch_idx, (ex_train_data, example_targets) = next(examples)
examples = enumerate(test_data)
batch_idx, (ex_test_data, example_targets) = next(examples)

print('Size of inputs',ex_train_data.shape)
channels = ex_train_data.shape[0]
img_size = ex_train_data.shape[1]



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
#input_size = training_data.data.shape[1] * training_data.data.shape[2] * training_data.data.shape[3]
output_size = len(training_data.classes)

nb_filters = params['nb_filters']
model = nets.ConvNet(nb_filters,channels)#particular convnet with one convolution and one fully connected layer

# setting the optimizer
use_gradient = params['optimizer'] == 'grad'

if use_gradient:
    optimizer = optimizers.GradientOptimizer(lr=params['lr'])
else:
    nn_size = params['nn_size']
    st_prop = stats.Student(params['student_variance_prop'])
    #st_prior = stats.HeavyTail(params['student_variance_prop'],100)
    st_prior = stats.Student(params['student_variance_prior'])
    optimizer = optimizers.MCMCOptimizer(st_prop, nn_size=nn_size,iter_mcmc=params['iter_mcmc'], lamb=params['lamb'], prior=st_prior)


exp_name = params['exp_name']
'''
if use_gradient:
    exp_name = exp_name+'_'+str(params['lr'])
else:
    exp_name = exp_name+'_'+str(params['lamb'])
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
print('Using {} device'.format(device))
model = model.to(device)

results = {}
sparse_results = []
epochs = params['epochs']
sparse_threshold = params['sparse_threshold']
loss_fn = nets.my_mse_loss
start_all = time.time()
res = []
for exp_number in range(10):
    for t in range(epochs):
        start_epoch = time.time()
        print(f"Epoch {t+1} is running\n--------------------- duration = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)) +"----------")
        if use_gradient:
            optimizer.train_1_epoch(train_dataloader, model, loss_fn)
        else:
            acceptance_ratio_f, acceptance_ratio_l = optimizer.train_1_epoch(train_dataloader, model, loss_fn, optimizer)
        results[t] = {}
        end_epoch = time.time() 
        results[t]['training time'] = time.time() - start_epoch
        loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
        if use_gradient:
            print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
        else:
            print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n Acceptance ratios: {acceptance_ratio_f, acceptance_ratio_l}")
        if not use_gradient:
            results[t]['accept_ratio_filter'] = acceptance_ratio_f
            results[t]['accept_ratio_linear'] = acceptance_ratio_l
        results[t]['train'] = {'training loss' : loss, 'training accuracy' : accuracy }
        loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
        print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")    
        results[t]['test'] = {'test loss' : loss, 'testing accuracy' : accuracy}
        #sparse evaluation of the linear model
        for i in range(9):
            proba = 0.1+i*0.1
            loss_sparse, accuracy_sparse, kept = nets.evaluate_sparse(test_dataloader, model, loss_fn,proba)
            print(f"Sparse Test Error: \n Accuracy: {(100*accuracy_sparse):>0.1f}%, Avg loss: {loss_sparse:>8f}, Sparsity index: {kept:>8f} \n")
            if i == 0:
                results[t]['sparse test'] = [{'test loss sparse' : loss_sparse, 'testing accuracy sparse' : accuracy_sparse, 'l0 norm': kept }]
            else:
                results[t]['sparse test'].append({'test loss sparse' : loss_sparse, 'testing accuracy sparse' : accuracy_sparse, 'l0 norm': kept })
        results[t]['training time'] = time.time() - end_epoch
        torch.save(model, exp_name+'.th')
        json.dump(results, open(exp_name+'_'+str(exp_number)+'.json','w'))
