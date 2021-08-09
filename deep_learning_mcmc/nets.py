from torch import nn
import torch
import numpy as np
from abc import ABC, abstractmethod

class MCMCLayers(ABC):
    @property
    @abstractmethod
    def is_binary(self):
        pass

class BinaryLinear(nn.Linear):
    is_binary = True
    def __init__(self, input, output):
        super(BinaryLinear, self).__init__(input, output)
        self.weight.data = np.sign(self.weight.data)
        self.bias.data = np.sign(self.bias.data)

    def update(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1]] *= -1
        self.bias.data[idces_b] *= -1

    def undo(self, neighborhood, proposal):
        self.update(neighborhood, proposal)

class Linear4MCMC(nn.Linear):
    is_binary = False
    def update(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1]] += proposal[:idces_w.shape[0]]
        if idces_b.shape[0] !=0 :
            self.bias.data[idces_b] += proposal[idces_w.shape[0]:]

    def undo(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1]] -= proposal[:idces_w.shape[0]]
        if idces_b.shape[0] !=0 :
            self.bias.data[idces_b] -= proposal[idces_w.shape[0]:]

class MLP(nn.Module):
    def __init__(self, sizes, activations='ReLU'):
        """
        builds a multi layer perceptron
        sizes : list of the size of the different layers
        activations : can be a string or a list of string. see torch.nn for possible values (ReLU, Softmax,...)
        """
        if len(sizes)< 2:
            raise Exception("sizes argument is" +  sizes.__str__() + ' . At least two elements are needed to have the input and output sizes')
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        input_size = sizes[0]
        output_size = sizes[-1]
        self.linears = nn.ModuleList()
        self.activations = []
        for i in range(1, len(sizes)):
            self.linears.append(nn.Linear(sizes[i-1], sizes[i]))
            if isinstance(activations, str):
                act = activations
            else:
                act = activations[i-1]
            activation = getattr(nn, act)()
            self.activations.append(activation)

    def forward(self, x):
        x = self.flatten(x)
        for linear, activation in zip(self.linears, self.activations):
            x = linear(x)
            if activation is not None:
                x = activation(x)
        return x


class MLP2(nn.Module):
    def __init__(self, sizes, binary_flags=None, activations=None):
        """
        builds a multi layer perceptron
        sizes : list of the size of the different layers
        activations : can be a string or a list of string. see torch.nn for possible values (ReLU, Softmax,...)
        """
        if len(sizes)< 2:
            raise Exception("sizes argument is" +  sizes.__str__() + ' . At least two elements are needed to have the input and output sizes')
        super(MLP2, self).__init__()
        self.flatten = nn.Flatten()
        input_size, output_size = sizes[0], sizes[-1]
        self.linears = nn.ModuleList()
        if activations is None:
            activations = ['ReLU' for i in range(1, len(sizes))]
        if binary_flags is None:
            binary_flags = [False for i in range(1, len(sizes)) ]
        self.activations = []
        for i in range(len(sizes)-1):
            if binary_flags[i]:
                linear = BinaryLinear(sizes[i], sizes[i+1])
            else:
                linear = Linear4MCMC(sizes[i], sizes[i+1])
            self.linears.append(linear)
            activation = getattr(nn, activations[i])()
            self.activations.append(activation)

    def forward(self, x):
        x = self.flatten(x)
        for linear, activation in zip(self.linears, self.activations):
            x = linear(x)
            if activation is not None:
                x = activation(x)
        return x

class BinaryNetwork(MLP):
    def __init__(self, sizes,  binary_flags, activations='ReLU'):
        """
        builds a multi layer perceptron
        sizes : list of the size of the different layers
        binary_flags : list of booleans indicating whether or not the corresponding layer has binary weights
        activations : can be a string or a list of string. see torch.nn for possible values (ReLU, Softmax,...)
        """
        if len(sizes)< 2:
            raise Exception("sizes argument is" +  sizes.__str__() + ' . At least two elements are needed to have the input and output sizes')
        super(BinaryNetwork, self).__init__(sizes, activations=activations)
        self.binary_flags = binary_flags
        for linear, binary_flag in zip(self.linears, self.binary_flags):
            if binary_flag:
                linear.weight.data = np.sign(linear.weight.data)
                linear.bias.data = np.sign(linear.bias.data)

mse_loss = nn.MSELoss()
def my_mse_loss(x,y):
    #mse_loss = nn.MSELoss()
    y = y.reshape((y.shape[0],1))
    y_onehot = torch.FloatTensor(x.shape[0], x.shape[1]).to(y.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return mse_loss(x, y_onehot)

def evaluate(dataloader, model, loss_fn):
    device = next(model.parameters()).device
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    #test_loss /= size
    correct /= size
    return test_loss, correct
