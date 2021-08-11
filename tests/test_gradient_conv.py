import time, os
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json
import torch
import numpy as np, math
from deep_learning_mcmc import nets, optimizers, stats, selector 

training_data = datasets.CIFAR10(root="/home/paul/data/pytorch_cifar10/",
    train=True,
    download=True,
    transform=ToTensor())

test_data = datasets.CIFAR10(root="/home/paul/data/pytorch_cifar10/",
    train=False,
    download=True,
    transform=ToTensor())

batch_size = 64 
# getting the data
train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=12)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=12)

# setting the model
input_size = training_data.data.shape[1] * training_data.data.shape[2] * training_data.data.shape[3]
output_size = len(training_data.classes)
layer_sizes = [input_size, output_size]

channels = training_data.data.shape[3]
nb_filters = 64
model = nets.ConvNet(nb_filters,channels)

optimizer = optimizers.GradientOptimizer()

loss_fn = nets.my_mse_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
model = model.to(device)

results = {}
epochs = 100
start_all = time.time()
for t in range(epochs):
    start_epoch = time.time()
    print(f"Epoch {t+1} is running\n--------------------- duration = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)) +"----------")
    optimizer.train_1_epoch(train_dataloader, model, loss_fn)
    loss, accuracy = nets.evaluate(train_dataloader, model, loss_fn)
    print(f"Training Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f}")
    loss, accuracy = nets.evaluate(test_dataloader, model, loss_fn)
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
