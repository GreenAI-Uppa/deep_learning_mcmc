from torch import nn
import torch
import numpy as np
from abc import ABC, abstractmethod
import torch.nn.functional as F


class Conv2d4MCMC(nn.Conv2d):
    is_binary = False
    def update(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1],idces_w[:,2],idces_w[:,3]] += proposal[:idces_w.shape[0]]
        if idces_b.shape[0] !=0 :
            self.bias.data[idces_b] += proposal[idces_w.shape[0]:]

    def undo(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1]] -= proposal[:idces_w.shape[0]]
        if idces_b.shape[0] !=0 :
            self.bias.data[idces_b] -= proposal[idces_w.shape[0]:]

class Linear4MCMC(nn.Linear):
    is_binary = False
    def update(self, neighborhood, proposal):
        self.selected_size = -1
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1]] += proposal[:idces_w.shape[0]]
        if idces_b.shape[0] !=0 :
            self.bias.data[idces_b] += proposal[idces_w.shape[0]:]

    def undo(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1]] -= proposal[:idces_w.shape[0]]
        if idces_b.shape[0] !=0 :
            self.bias.data[idces_b] -= proposal[idces_w.shape[0]:]

    def get_selected_size(self, idces):
        idces_w, idces_b = idces
        return idces_w.shape[0] + idces_b.shape[0]

    def getParamLine(self, idces):
        idces_w, idces_b = idces
        w = self.weight.data[idces_w[:,0],idces_w[:,1]]
        b = self.bias.data[idces_b]
        return torch.cat((w,b))

class BinaryLinear(Linear4MCMC):
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

class MLP(nn.Module):
    def __init__(self, sizes, binary_flags=None, activations=None):
        """
        builds a multi layer perceptron
        sizes : list of the size of the different layers
        activations : can be a string or a list of string. see torch.nn for possible values (ReLU, Softmax,...)
        """
        if len(sizes)< 2:
            raise Exception("sizes argument is" +  sizes.__str__() + ' . At least two elements are needed to have the input and output sizes')
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        input_size, output_size = sizes[0], sizes[-1]
        self.layers = nn.ModuleList()
        if binary_flags is None:
            binary_flags = [False for i in range(1, len(sizes)) ]
        self.activations = []
        for i in range(len(sizes)-1):
            if binary_flags[i]:
                linear = BinaryLinear(sizes[i], sizes[i+1])
            else:
                linear = Linear4MCMC(sizes[i], sizes[i+1])
            self.layers.append(linear)
            if activations is None:
                activation = nn.ReLU()
            else:
                activation = getattr(nn, activations[i])()
            self.activations.append(activation)

    def forward(self, x):
        x = self.flatten(x)
        for linear, activation in zip(self.layers, self.activations):
            x = linear(x)
            if activation is not None:
                x = activation(x)
        return x

class ConvNet(nn.Module):
    def __init__(self,nb_filters,channels, binary_flags=None, activations=None):
        super(ConvNet, self).__init__()
        self.nb_filters = nb_filters
        self.channels = channels
        self.layers = nn.ModuleList()
        if channels == 3:
            self.conv1 = nn.Conv2d(in_channels=channels, out_channels=nb_filters, kernel_size=11, stride=3, padding=0)
        else:
            self.conv1 = nn.Conv2d(in_channels=channels, out_channels=nb_filters, kernel_size=7, stride=3, padding=0)
        self.layers.append(self.conv1)
        self.fc1 = nn.Linear(self.nb_filters * 8 * 8, 10)
        self.layers.append(self.fc1)
        self.activations = []
        if activations is None:
            self.activations = [nn.ReLU() for i in self.layers ]
        else:
            self.activations = [ getattr(nn, activations[i])() for i in range(len(self.layers)) ]


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, self.nb_filters * 8 * 8)
        x = F.relu(self.fc1(x))
        return x

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
