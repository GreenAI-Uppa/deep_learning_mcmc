import time, os, datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import math
import json
import torch
import numpy as np, math
from deep_learning_mcmc import nets, optimizers, stats, selector, evaluator
import argparse


def parse_optimizer_config(params):
    # setting the optimizer
    if params["optimizer"]["name"] == "grad":
        if 'pruning_proba' in params["optimizer"]:
            optimizer = optimizers.GradientOptimizer(lr=params["optimizer"]['lr'],pruning_proba=params["optimizer"]['pruning_proba'])
        else:
            optimizer = optimizers.GradientOptimizer(lr=params["optimizer"]['lr'])
    elif params["optimizer"]["name"] == "binaryConnect":
        optimizer = optimizers.BinaryConnectOptimizer(lr=params["optimizer"]['lr'])
    elif params["optimizer"]["name"] == "layer_wise":
        optimizer = optimizers.LayerWiseOptimizer(lr=params["optimizer"]['lr'])
    elif params["optimizer"]["name"] == "layer_wise_async":
        optimizer = optimizers.LayerWiseAsyncOptimizer(lr=params["optimizer"]['lr'])
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
    return optimizer


def parse_model_config(params, training_data, train_dataloader):
    """ setting the model"""
    n, w, h, channels = training_data.data.shape
    n_classes = 10 # for cifar10
    # building the model
    kernel_size = 5
    conv1 = nets.ConvNetAuxResult(32, 3, stride=1, kernel_size=kernel_size, input_size=w)
    out_conv1 = (w - kernel_size ) + 1 # (image_width - kernel_size + padding )/ stride + 1
    conv2 = nets.ConvNetAuxResult(64, 32, kernel_size=kernel_size, stride=1, input_size=conv1.outconv_size)
    fc1 = nets.FCAuxResult([conv2.input_layer_size, 120, n_classes], flatten=True)
    fc2 = nets.FCAuxResult([120, 84, n_classes])
    model = [conv1, conv2, fc1, fc2]

    # small check that it runs correctly
    X, y = next(iter(train_dataloader))
    outconv1, aux = conv1(X)
    print(outconv1.shape, conv1.outconv_size, conv1.input_layer_size, aux.shape)
    outconv2, aux = conv2(outconv1)
    print(outconv2.shape, conv2.outconv_size, conv2.input_layer_size, aux.shape)
    outfc1, aux = fc1(outconv2)
    print(outfc1.shape, aux.shape)
    outfc2, aux = fc2(outfc1)
    print(outfc2.shape, aux.shape)
    return model


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

channels = 3
training_data = datasets.CIFAR10(root=params['data_folder'],
    train=True,
    download=True,
    transform=ToTensor())

test_data = datasets.CIFAR10(root=params['data_folder'],
    train=False,
    download=True,
    transform=ToTensor())


print('Experience config --')
print(params)

# getting the batch size
batch_size = params['batch_size']
# getting the data
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=16)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=16)

optimizer = parse_optimizer_config(params)
model = parse_model_config(params, training_data, train_dataloader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


epochs = params['epochs']
loss_fn = torch.nn.CrossEntropyLoss()

results = {}
exp_name = params['exp_name']

if params['measure_power']:
    from deep_learning_power_measure.power_measure import experiment, parsers
    input_image_size = (batch_size, training_data.data.shape[3], training_data.data.shape[1], training_data.data.shape[2])
    driver = parsers.JsonParser(os.path.join(os.getcwd(),'power_measure'))
    exp = experiment.Experiment(driver,model=model,input_size=input_image_size)
    p, q = exp.measure_yourself(period=2)

model = [m.to(device) for m in model ]
training_time = 0
eval_time = 0
start_all = time.time()
previous_w_updated = 0
#start_epoch = time.time()
optimizer.train_async(train_dataloader,test_dataloader, model, loss_fn, epochs)

print(f"Testing\n--------------------- duration = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)) +"----------")
loss, accuracy = evaluator.evaluateLayerWise(test_dataloader, model, loss_fn)
for l in loss:
    acc = accuracy[l]
    los = loss[l]
    print(f"Training Error: Layer {l} \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {los:>8f} \n")
# result = {"epoch":t}
# end_epoch = time.time()
# training_time += time.time() - start_epoch
# result['training_time'] = time.time() - start_epoch
# result['end_training_epoch'] = datetime.datetime.now().__str__()

# result['training_loss'] = loss
# result['training_accuracy'] = accuracy

#result['testing_loss'] = loss
#result['testing_accuracy'] = accuracy

# if int(math.log(t+1,10)) == math.log(t+1,10):
#     torch.save(model, exp_name+str(t+1)+'.th')
# result['eval_time'] = time.time() - end_epoch
# eval_time += time.time() - end_epoch
# result['end_eval'] = datetime.datetime.now().__str__()
# results[t]=result
# json.dump(results, open(exp_name+'.json','w'))

if params['measure_power']:
    q.put(experiment.STOP_MESSAGE)
    print("power measuring stopped")
    driver = parsers.JsonParser("power_measure")
    exp_result = experiment.ExpResults(driver)
    exp_result.print()

# print(exp_name+'.json generated')
# print('Report is written at '+str(exp_name)+'.csv')
print("Total running duration = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)))
