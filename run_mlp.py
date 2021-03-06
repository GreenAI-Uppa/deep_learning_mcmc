import time, os, datetime
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
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
                    help='if set, will print the loss for each mcmc iteration.',
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
json_params = json.load(open(params['config_file']))
for k,v in json_params.items():
    params[k] = v

# getting the batch size
batch_size = params['batch_size']

# getting the data
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=16)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=16)

# setting the model
input_size = training_data.data.shape[1] * training_data.data.shape[2] * training_data.data.shape[3]
print(training_data.data.shape)



channels = training_data.data.shape[3]
output_size = len(training_data.classes)
if "hidden_size" not in params["architecture"]:
    layer_sizes = [input_size, output_size]
else:
    layer_sizes = [input_size, params["architecture"]['hidden_size'], output_size]

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
    optimizer = optimizers.GradientOptimizer(lr=params["optimizer"]['lr'])
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
    optimizer = optimizers.MCMCOptimizer(samplers, iter_mcmc=params["optimizer"]["iter_mcmc"], prior=samplers, selector=selector)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
epochs = params['epochs']
#loss_fn = nets.my_mse_loss
loss_fn = torch.nn.CrossEntropyLoss()
num_simu = 10
results = []

model = nets.MLP(layer_sizes, binary_flags=boolean_flags, activations=activations)
exp_name = params['exp_name']
if params['measure_power']:
    from deep_learning_power_measure.power_measure import experiment, parsers
    input_image_size = (batch_size, training_data.data.shape[3], training_data.data.shape[1], training_data.data.shape[2])
    driver = parsers.JsonParser(os.path.join(os.getcwd(),'power_measure'))
    exp = experiment.Experiment(driver, model=model, input_size=input_image_size)
    p, q = exp.measure_yourself(period=2)
model = model.to(device)
training_time = 0
eval_time = 0
start_all = time.time()
for t in range(epochs):
    start_epoch = time.time()
    print(f"Epoch {t+1} is running\n--------------------- duration = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)) +"----------")
    if use_gradient:
        optimizer.train_1_epoch(train_dataloader, model, loss_fn)
    else:
        acceptance_ratio = optimizer.train_1_epoch(train_dataloader, model, loss_fn, verbose=params['verbose'])
    result = {"epoch":t}
    end_epoch = time.time() 
    training_time += time.time() - start_epoch
    result['training_time'] = time.time() - start_epoch
    result['end_training_epoch'] = datetime.datetime.now().__str__() 
    loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
    if use_gradient:
        print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    else:
        print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n") #Acceptance ratio: {acceptance_ratio:>2f}")
        print("Acceptance ratio",acceptance_ratio)
    if not use_gradient:
        result['accept_ratio'] = acceptance_ratio.to_dict()
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
if params['measure_power']:
    q.put(experiment.STOP_MESSAGE)
    print("power measuring stopped")
