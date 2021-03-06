import time, os
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import torch
import numpy as np, math
from deep_learning_mcmc import nets, optimizers, stats, selector 

training_data = datasets.CIFAR10(root="/home/pgay/cifar/",
    train=True,
    download=True,
    transform=ToTensor())

test_data = datasets.CIFAR10(root="/home/pgay/cifar/",
    train=False,
    download=True,
    transform=ToTensor())

batch_size = 50000
# getting the data
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=12)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=12)

# setting the model
input_size = training_data.data.shape[1] * training_data.data.shape[2] * training_data.data.shape[3]
output_size = len(training_data.classes)
layer_sizes = [input_size, 100, output_size]

model = nets.MLP(layer_sizes, binary_flags = [True, False], activations = ['Sigmoid', 'Softmax'])

#config = {'name':'MixedSelector', 'neighborhood_size': {'binary': 1, 'real':10}, 'layer_distr':[0.9, 0.1]}
config = {'name':'MixedSelector', 'layer_conf': [{'layer_distr' : 0.9 , 'get_idx' : selector.get_idces_uniform_linear(1)}, {'layer_distr' : 0.1 , 'get_idx' : selector.get_idces_uniform_linear(40)}]}
s= selector.build_selector(config)
print('generating layer index',s.get_layer_idx())
print('generating neighborhood', s.get_neighborhood(model))


student_variance_prop = 0.00000000001
st_prop = stats.build_distr({'name':'MixedSampler','variance': student_variance_prop })
# setting the optimizer
optimizer = optimizers.MCMCOptimizer(st_prop, iter_mcmc=2000, lamb=10000000, prior=st_prop, selector=s)

loss_fn = nets.my_mse_loss
#loss_fn = nets.nn.CrossEntropyLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
model = model.to(device)

results = {}
epochs = 100
start_all = time.time()
for t in range(epochs):
    start_epoch = time.time()
    print(f"Epoch {t+1} is running\n--------------------- duration = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)) +"----------")
    acceptance_ratio = optimizer.train_1_epoch(train_dataloader, model, loss_fn, verbose=True)
    loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
    print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f}")# \n Acceptance ratio: {acceptance_ratio:>2f}")
    loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
    print("acceptance_ratio", acceptance_ratio)
