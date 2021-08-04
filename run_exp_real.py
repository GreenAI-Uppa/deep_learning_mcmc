import time, os, datetime
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import torch
import numpy as np, math
#import sys
#sys.path.insert()
from deep_learning_mcmc import nets, optimizers, stats, utils
from power_consumption_measure import measure_utils, model_complexity
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

# getting the data
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=16)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=16)

# setting the model
input_size = training_data.data.shape[1] * training_data.data.shape[2] * training_data.data.shape[3]
output_size = len(training_data.classes)
if "hidden_size" not in params:
    layer_sizes = [input_size, output_size]
else:
    layer_sizes = [input_size, params['hidden_size'], output_size]

use_gradient = params['optimizer']["name"] == 'grad'
# setting the optimizer
if params["optimizer"]["name"] == "grad":
    optimizer = optimizers.GradientOptimizer(lr=params["optimizer"]['lr'])
else:
    selector =  utils.build_selector(params["optimizer"]["selector"], layer_sizes)
    sampler = stats.build_distr(params["optimizer"]["sampler"])
    prior = stats.build_distr(params["optimizer"]["prior"])
    optimizer = optimizers.MCMCOptimizer(sampler, iter_mcmc=params["optimizer"]["iter_mcmc"], lamb=params["optimizer"]["lamb"], prior=prior, selector=selector)
input_image_size = (batch_size, training_data.data.shape[1], training_data.data.shape[2], training_data.data.shape[3]) 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
epochs = params['epochs']
loss_fn = nets.my_mse_loss
num_simu = 10
results = []

model = nets.MLP(layer_sizes)
model = model.to(device)
exp_name = params['exp_name']
if use_gradient:
    exp_name = '_'.join((exp_name, str(params["optimizer"]['lr'])))
else:
    exp_name = '_'.join(( exp_name, str(params["optimizer"]['lamb'])))
if args.measure_power:
    outdir_power = exp_name+'_power'
    p, q = measure_utils.measure_yourself(outdir=outdir_power, period=2)
training_time = 0
eval_time = 0
start_all = time.time()
for t in range(epochs):
    start_epoch = time.time()
    print(f"Epoch {t+1} is running\n--------------------- duration = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)) +"----------")
    if use_gradient:
        optimizer.train_1_epoch(train_dataloader, model, loss_fn)
    else:
        acceptance_ratio = optimizer.train_1_epoch(train_dataloader, model, loss_fn, optimizer)
    result = {"epoch":t}
    end_epoch = time.time() 
    training_time += time.time() - start_epoch
    result['training_time'] = time.time() - start_epoch
    result['end_training_epoch'] = datetime.datetime.now().__str__() 
    loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
    print("time of training one epoch: ",result['training_time'])
    if use_gradient:
        print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    else:
        print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n") #Acceptance ratio: {acceptance_ratio:>2f}")
        print("Acceptance ratio",acceptance_ratio[0],"exploration",acceptance_ratio[1])
    if not use_gradient:
        result['accept_ratio'] = acceptance_ratio
    result['train_loss'] = loss
    result['train_accuracy'] = accuracy
    loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    result['test_loss'] = loss
    result['test_accuracy'] = accuracy
    json.dump(results, open(exp_name+'.json','w'))
    torch.save(model, exp_name+'.th')
    result['eval_time'] = time.time() - end_epoch
    eval_time += time.time() - end_epoch
    result['end_eval'] = datetime.datetime.now().__str__()
    results.append(result)
if args.power_measure:
    q.put(measure_utils.STOP_MESSAGE)
    print("wrapping stopped, computing final statistics")
    summary = model_complexity.get_summary(model, input_image_size)
    json.dump(summary, open(os.path.join(outdir_power,'model_summary.json'), 'w'))
