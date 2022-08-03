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
    if 'pruning_level' in params["optimizer"]:
        optimizer = optimizers.GradientOptimizer(lr=params["optimizer"]['lr'],pruning_level=params["optimizer"]['pruning_level'])
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
    if 'pruning_level' in params["optimizer"]:
        optimizer = optimizers.MCMCOptimizer(samplers, iter_mcmc=params["optimizer"]["iter_mcmc"], prior=samplers, selector=selector,pruning_level=params["optimizer"]['pruning_level'])
    else:
        optimizer = optimizers.MCMCOptimizer(samplers, iter_mcmc=params["optimizer"]["iter_mcmc"], prior=samplers, selector=selector)

device = 'cpu' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
epochs = params['epochs']
loss_fn = torch.nn.CrossEntropyLoss()
results = {}

if "variance_init" in params:
    st_init = stats.Student(params['variance_init'])
    if 'pruning_level' in params["optimizer"]:
        model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations, init_sparse=st_init,pruning_level = params["optimizer"]['pruning_level'])
    else:
        model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations, init_sparse=st_init)
else:
    if 'pruning_level' in params["optimizer"]:
        model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations,pruning_level = params["optimizer"]['pruning_level'])
    else:
        model = nets.ConvNet(params['architecture']['nb_filters'], channels, binary_flags=boolean_flags,  activations=activations)


exp_name = params['exp_name']

if params['measure_power']:
    from deep_learning_power_measure.power_measure import experiment, parsers
    input_image_size = (batch_size, training_data.data.shape[3], training_data.data.shape[1], training_data.data.shape[2])
    driver = parsers.JsonParser(os.path.join(os.getcwd(),'power_measure'))
    exp = experiment.Experiment(driver,model=model,input_size=input_image_size)
    p, q = exp.measure_yourself(period=2)
model = model.to(device)
# convert torch model to keras model
from pytorch2keras import pytorch_to_keras
from torch.autograd import Variable
import larq_compute_engine as lce
import larq as lq

from torchinfo import summary
import tensorflow as tf
# we should specify shape of the input tensor
# from 50000, 3,32 32 to 64,3,11,11)
input_np = np.random.uniform(0, 1, (1, 3, 32, 32))
input_var = Variable(torch.FloatTensor(input_np))
k_model = pytorch_to_keras(model, input_var,[(3, None, None,)], verbose=True)  
#torch.save(model, 'model.pt')
#k_model.save('k_model.h5')
print(summary(model))
print(k_model.summary())
print(k_model.get_layer(index=1).get_weights())
print(k_model.get_layer(index=1).input_shape)
weights_initializer=k_model.get_layer(index=1).kernel_initializer
bias_initializer = k_model.get_layer(index=1).bias_initializer
config = k_model.get_layer(index=1).get_config()
print("################## config #####################")
print(config)
layerconv = lq.layers.QuantConv2D(64, kernel_size=config['kernel_size'],
                                kernel_quantizer="ste_sign",
                                kernel_constraint="weight_clip",
                                data_format='channels_first',
                                activation = 'linear',
                                kernel_initializer = tf.keras.initializers.zeros(),
                                bias_initializer=tf.keras.initializers.zeros(),
                                #use_bias=False,
                                strides=3,
                                input_shape=(3, None, None)
                                )

print(layerconv.weights)
#layerconv.set_weights(k_model.get_layer(index=1).get_weights())

model_larq = tf.keras.models.Sequential()
print(k_model.get_layer(index=0))
model_larq.add(k_model.get_layer(index=0))
#print(k_model.get_layer(index=0).output_shape)

model_larq.add(layerconv)
#print("Before update weights")
#print(model_larq.get_layer(index=0).weights)
layerconv.set_weights(k_model.get_layer(index=1).get_weights())
#print("After update weights")
#print(model_larq.get_layer(index=0).weights)
model_larq.add(k_model.get_layer(index=2))
model_larq.add(k_model.get_layer(index=3))
model_larq.add(k_model.get_layer(index=4))
#model_larq.save('larq_model.h5')
print("summary model")
print(model_larq.summary())
# pour modifer les poids du modèle tfile
#input_wt = tf.keras.layers.Input(shape=(1, 3, 200), dtype=tf.float32)
#input_data = tf.keras.layers.Input(shape=(64, 64, 3,), dtype=tf.float32)

with open("BinaryMcmc.tflite", "wb") as flatbuffer_file:
    flatbuffer_bytes = lce.convert_keras_model(model_larq)
    flatbuffer_file.write(flatbuffer_bytes)
#print("Interpreter")





training_time = 0
eval_time = 0
start_all = time.time()
previous_w_updated = 0
for t in range(epochs):
    '''if "pruning_level" in params["optimizer"] and params["optimizer"]["pruning_level"]>0:
        bin_mat = torch.abs(model.conv1.weight.data) > 0
        print(int(torch.sum(bin_mat)),'/',torch.flatten(bin_mat).shape[0],'kept values for layer 0')
        bin_mat = torch.abs(model.fc1.weight.data) > 0
        print(int(torch.sum(bin_mat)),'/',torch.flatten(bin_mat).shape[0],'kept values for layer 1')
    '''
    start_epoch = time.time()
    print(f"Epoch {t+1} is running\n--------------------- duration = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)) +"----------")
    if use_gradient:
        print("0K use gradient")
        optimizer.train_1_epoch(train_dataloader, model,k_model, flatbuffer_bytes,loss_fn)
    else:
        acceptance_ratio = optimizer.train_1_epoch(train_dataloader, model,k_model,flatbuffer_bytes, loss_fn, verbose=params['verbose'])
    result = {"epoch":t}
    end_epoch = time.time()
    training_time += time.time() - start_epoch
    result['training_time'] = time.time() - start_epoch
    result['end_training_epoch'] = datetime.datetime.now().__str__()
    loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
    if use_gradient:
        result['iterations'] = (t+1)*int(50000/batch_size)
        result['passforwards'] = (t+1)*50000
        result['backwards'] = (t+1)*50000
        result['weights_updated'] = (t+1)*int(50000/batch_size)*64266
        print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    else:
        result['iterations'] = (t+1)*params["optimizer"]["iter_mcmc"]*int(50000/batch_size)
        result['passforwards'] = (t+1)*params["optimizer"]["iter_mcmc"]*int(50000/batch_size)
        result['backwards'] = 0
        print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n") #Acceptance ratio: {acceptance_ratio:>2f}")
        print("Acceptance ratio",acceptance_ratio)
    if not use_gradient:
        result['accept_ratio'] = acceptance_ratio.to_dict()
        acc_0 = acceptance_ratio.to_dict()["layer_0"]
        acc_1 = acceptance_ratio.to_dict()["layer_1"]
        if 'get_idx_param' in params['optimizer']['selector']['layer_conf'][1]:
            result['weights_updated'] = previous_w_updated + int(50000/batch_size)*params["optimizer"]["iter_mcmc"]*(0.5*363+0.5*layer_conf['get_idx_param'])
            previous_w_updated = result['weights_updated']
    result['train_loss'] = loss
    result['train_accuracy'] = accuracy
    loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    result['test_loss'] = loss
    result['test_accuracy'] = accuracy
    for i in range(9):
        proba = 0.1+i*0.1
        loss_sparse, accuracy_sparse, kept = nets.evaluate_sparse(test_dataloader, model, loss_fn,proba,boolean_flags)
        if i == 0:
            result['sparse test'] = [{'test loss sparse' : loss_sparse, 'testing accuracy sparse' : accuracy_sparse, 'l0 norm': kept }]
        else:
            result['sparse test'].append({'test loss sparse' : loss_sparse, 'testing accuracy sparse' : accuracy_sparse, 'l0 norm': kept })
    for i in range(9):
        proba = 0.91+i*0.01
        loss_sparse, accuracy_sparse, kept = nets.evaluate_sparse(test_dataloader, model, loss_fn,proba,boolean_flags)
        result['sparse test'].append({'test loss sparse' : loss_sparse, 'testing accuracy sparse' : accuracy_sparse, 'l0 norm': kept })
    if int(math.log(t+1,10)) == math.log(t+1,10):
        torch.save(model, exp_name+str(t+1)+'.th')
    result['eval_time'] = time.time() - end_epoch
    eval_time += time.time() - end_epoch
    result['end_eval'] = datetime.datetime.now().__str__()
    results[t]=result
    json.dump(results, open(exp_name+'.json','w'))
if params['measure_power']:
    q.put(experiment.STOP_MESSAGE)
    print("power measuring stopped")
    driver = parsers.JsonParser("power_measure")
    exp_result = experiment.ExpResults(driver)
    exp_result.print()

print(exp_name+'.json generated')

print('Report is written at '+str(exp_name)+'.csv')
