from torch import nn
import torch
import numpy as np

class MLP(nn.Module):
    def __init__(self, sizes, activations='ReLU'):
        """
        builds a multi layer perceptron
        sizes : list of the size of the different layers
        act : activation function either "relu", "elu", or "soft" (softmax)
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
            x = activation(x)
        return x

class BinaryNetwork(MLP):
    def __init__(self, sizes, activations='ReLU'):
        """
        builds a multi layer perceptron
        sizes : list of the size of the different layers
        act : activation function either "relu", "elu", or "soft" (softmax)
        """
        if len(sizes)< 2:
            raise Exception("sizes argument is" +  sizes.__str__() + ' . At least two elements are needed to have the input and output sizes')
        super(BinaryNetwork, self).__init__(sizes, activations='ReLU')
        for linear in self.linears:
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
    test_loss /= size
    correct /= size
    return test_loss, correct
