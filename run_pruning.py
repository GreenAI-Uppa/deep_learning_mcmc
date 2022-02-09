import time, os, datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import math
import json
import torch
import numpy as np, math
#import sys
#sys.path.insert()
from deep_learning_mcmc import nets, optimizers, stats, selector
import argparse


parser = argparse.ArgumentParser(description='Train a model on cifar10 with either mcmc or stochastic gradient based approach')
parser.add_argument('--data_folder',
                    help='absolute path toward the data folder which contains the cifar10 dataset. Pytorch will download it if it does not exist',
                    required=True, type=str)
parser.add_argument('--config_file',
                    help='json file containing various setups (learning rate, mcmc iterations, variance of the priors and the proposal, batch size,...)',
                    default=None, type=str)
parser.add_argument('--measure_power',
                    help='if set, will record the power draw. This requires the deep_learning_measure package.',
                    action='store_true')
parser.add_argument('--verbose',
                    help='if set, will print the details of each mcmc iteration.',
                    action='store_true')


args = parser.parse_args()
params = vars(args)
json_params = json.load(open(params['config_file']))
for k,v in json_params.items():
    params[k] = v

dataset = params['dataset']

if dataset == 'MNIST':
    print('MNIST DATASET')
    channels = 1
    transform = transforms.Compose([transforms.ToTensor()])
    training_data = MNIST(root = args.data_folder, train=True, download=True, transform=transform)
    test_data = MNIST(root = args.data_folder, train=False, download=True, transform=transform)
else:
    print('CIFAR10 DATASET')
    channels = 3
    training_data = datasets.CIFAR10(root=args.data_folder,
        train=True,
        download=True,
        transform=ToTensor())

    test_data = datasets.CIFAR10(root=args.data_folder,
        train=False,
        download=True,
        transform=ToTensor())

'''
args = parser.parse_args()
training_data = datasets.CIFAR10(root=args.data_folder,
    train=True,
    download=True,
    transform=ToTensor())

test_data = datasets.CIFAR10(root=args.data_folder,
    train=False,
    download=True,
    transform=ToTensor())
'''
examples = enumerate(training_data)
batch_idx, (ex_train_data, example_targets) = next(examples)
examples = enumerate(test_data)
batch_idx, (ex_test_data, example_targets) = next(examples)

print('Image input size',ex_train_data.shape)
img_size = ex_train_data.shape[1]

input_size = ex_train_data.shape[0] * ex_train_data.shape[1] * ex_train_data.shape[2]

print('Experience config --')
print(params)
# getting the batch size
batch_size = params['batch_size']

# getting the data
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=16)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=16)
# setting the model


print('Training size',training_data.data.shape)

output_size = len(training_data.classes)
if "nb_filters" not in params["architecture"]:
    layer_sizes = [input_size, output_size]
else:
    layer_sizes = [input_size, params["architecture"]['nb_filters'], output_size]

if "boolean_flags" in params["architecture"]:
    boolean_flags = [bool(b) for b in params['architecture']['boolean_flags']]
else:
    boolean_flags = [False for _ in layer_sizes[1:]]
if "activations" not in params["architecture"]:
    activations=None
else:
    activations = params["architecture"]["activations"]



use_gradient = params['optimizer']["name"] == 'grad'
# setting the optimizer
if params["optimizer"]["name"] == "grad":
    if 'pruning_proba' in params["optimizer"]:
        optimizer = optimizers.GradientOptimizer(lr=params["optimizer"]['lr'],pruning_proba=params["optimizer"]['pruning_proba'])
    else:
        optimizer = optimizers.GradientOptimizer(lr=params["optimizer"]['lr'])
elif params["optimizer"]["name"] == "binaryConnect":
    optimizer = optimizers.BinaryConnectOptimizer(lr=params["optimizer"]['lr'])
else:
    config = {'name': params['optimizer']['selector']['name'], 'layer_conf':[]}
    for layer_conf in params['optimizer']['selector']['layer_conf']:
        layer_distr = layer_conf['layer_distr']
        if 'get_idx_param' in layer_conf:
            get_idx = getattr(selector, layer_conf['get_idx'])(layer_conf['get_idx_param'])
        else:
            get_idx = getattr(selector, layer_conf['get_idx'])()
        config['layer_conf'].append({'layer_distr': layer_distr, 'get_idx': get_idx})
    selector =  selector.build_selector(config)
    samplers = stats.build_samplers(params["optimizer"]["samplers"])
    if 'pruning_proba' in params["optimizer"]:
        optimizer = optimizers.MCMCOptimizer(samplers, iter_mcmc=params["optimizer"]["iter_mcmc"], prior=samplers, selector=selector,pruning_proba=params["optimizer"]['pruning_proba'])
    else:
        optimizer = optimizers.MCMCOptimizer(samplers, iter_mcmc=params["optimizer"]["iter_mcmc"], prior=samplers, selector=selector)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
epochs = params['epochs']
loss_fn = torch.nn.CrossEntropyLoss()

exp_name = params['exp_name']

for k in range(10):
    print('Experience',k,'/ 9 is running')
    if "variance_init" in params:
        st_init = stats.Student(params['variance_init'])
        if 'pruning_proba' in params["optimizer"]:
            model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations, init_sparse=st_init,pruning_proba = params["optimizer"]['pruning_proba'])
        else:
            model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations, init_sparse=st_init)
    else:
        if 'pruning_proba' in params["optimizer"]:
            model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations,pruning_proba = params["optimizer"]['pruning_proba'])
        else:
            model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations)
    model = model.to(device)
    start_epoch = time.time()
    if use_gradient:
        result = optimizer.train_1_epoch(train_dataloader, model, loss_fn)
    else:
        result = optimizer.train_1_epoch(train_dataloader, model, loss_fn, verbose=params['verbose'])
    time_spent = time.time() - start_epoch
    print(time_spent)
    result['time'] = time_spent
    json.dump(result, open(exp_name+'_'+str(k)+'.json','w'))
    print(exp_name+'_'+str(k)+'.json generated')