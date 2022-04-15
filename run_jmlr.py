import time, os, datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import math
import json
import torch
import numpy as np, math
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


print('CIFAR10 DATASET preparation')
channels = 3
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

print('Image input size',ex_train_data.shape)

input_size = ex_train_data.shape[0] * ex_train_data.shape[1] * ex_train_data.shape[2]

if params['verbose']:
    print('Experience config --')
    for elt in params:
        print(elt,":",params[elt])

# getting the batch size
batch_size = params['batch_size']

# getting the data
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=0)

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


# setting the optimizer
if params['optimizer']["name"] == "mcmc":
    use_gradient = False
    #check sampler and binary layers
    if params['architecture']['boolean_flags']:
        for k,elt in enumerate(params['architecture']['boolean_flags']):
            if elt ==1 and params['optimizer']['samplers'][k]['sampler']['name'] == 'Student':
                print('ERROR: a binary layer has a Student sampler')
                quit()
            if elt ==0 and params['optimizer']['samplers'][k]['sampler']['name'] == 'BinarySampler':
                print('ERROR: a continuous layer has a binary sampler')
                quit()
else:
    use_gradient = True


if params["optimizer"]["name"] == "grad":
    optimizer = optimizers.GradientOptimizer(lr=params["optimizer"]['lr'])
elif params["optimizer"]["name"] == "binaryConnect":
    if "progressive_pruning" in params["optimizer"].keys():
        print('ERROR: progressive pruning not available for BinaryConnectOptimizer')
        quit()
    optimizer = optimizers.BinaryConnectOptimizer(lr=params["optimizer"]['lr'])
else:
    #MCMC optimization
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
    optimizer = optimizers.MCMCOptimizer(samplers, prior=samplers, selector=selector)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
loss_fn = torch.nn.CrossEntropyLoss()
result = {}

if "variance_init" in params:
    st_init = stats.Student(params['variance_init'])
    if params['optimizer']['progressive_pruning'] ==1:
        model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations, init_sparse=st_init,pruning_level_start = params["optimizer"]['pruning_level_start'])
    else:
        model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations, init_sparse=st_init)
else:
    if params['optimizer']['name'] == 'binaryConnect':
        model = nets.BinaryConnectConv(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations)
    else:
        model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations,pruning_level_start = params["optimizer"]['pruning_level_start'])

exp_name = params['exp_name']

if params['measure_power']:
    from deep_learning_power_measure.power_measure import experiment, parsers
    input_image_size = (batch_size, training_data.data.shape[3], training_data.data.shape[1], training_data.data.shape[2])
    driver = parsers.JsonParser(os.path.join(os.getcwd(),'power_measure'))
    exp = experiment.Experiment(driver,model=model,input_size=input_image_size)
    p, q = exp.measure_yourself(period=2)
model = model.to(device)
previous_w_updated = 0
start_xp = time.time()
print(f"Training is running\n---------------------")
if params['optimizer']["name"] == "binaryConnect":
    res = optimizer.train(train_dataloader, model, loss_fn, params['optimizer']["iters"], verbose=params['verbose'])
elif params['optimizer']["name"] == "grad":
    res = optimizer.train(train_dataloader, model, loss_fn, params['optimizer']["iters"],params['optimizer']["progressive_pruning"], params['optimizer']["pruning_level_start"], params['optimizer']["pruning_schedule"], path=args.data_folder, verbose=params['verbose'])
else:
    acceptance_ratio, res = optimizer.train(train_dataloader, model, loss_fn, params['optimizer']["iters"],params['optimizer']["progressive_pruning"], params['optimizer']["pruning_level_start"], params['optimizer']["pruning_schedule"], path=args.data_folder, verbose=params['verbose'])
end_xp = time.time()
result['training_details'] = res
result['training_time'] = time.time() - start_xp
#result['end_training_time'] = datetime.datetime.now().__str__()
loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
if use_gradient:
    print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
else:
    print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n") #Acceptance ratio: {acceptance_ratio:>2f}")
    print(acceptance_ratio)
    result['acceptance_ratio'] = acceptance_ratio.to_dict()
result['train_loss'] = loss
result['train_accuracy'] = accuracy
loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
result['test_loss'] = loss
result['test_accuracy'] = accuracy

if params['naive_pruning_evaluation'] == 1:
    for i in range(9):
        proba = 0.1+i*0.1
        loss_sparse, accuracy_sparse, kept = nets.evaluate_sparse(test_dataloader, model, loss_fn,proba,boolean_flags)
        if i == 0:
            result['naive_pruning_evaluation'] = [{'test loss sparse' : loss_sparse, 'testing accuracy sparse' : accuracy_sparse, 'pruning_level':1-kept}]
        else:
            result['naive_pruning_evaluation'].append({'test loss sparse' : loss_sparse, 'testing accuracy sparse' : accuracy_sparse, 'pruning_level':1-kept})

result['eval_time'] = time.time() - end_xp
print(result)
json.dump(result, open(exp_name+'.json','w'))

if params['measure_power']:
    q.put(experiment.STOP_MESSAGE)
    print("power measuring stopped")
    driver = parsers.JsonParser("power_measure")
    exp_result = experiment.ExpResults(driver)
    exp_result.print()

print(exp_name+'.json generated')