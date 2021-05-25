from torch import nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, act='relu'):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3072, 10)
        if act == 'soft':
            print('using softmax activation')
            self.activation = nn.Softmax()
        elif act =='elu':
            print('using rely activation')
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
        self.soft_input = []

    def forward(self, x):
        x = self.flatten(x)
        self.soft_input = self.linear(x)
        logits = self.activation(self.soft_input)
        return logits
    def delta_loss(y):
        pass



loss = nn.MSELoss()
def my_mse_loss(x,y):
    mse_loss = nn.MSELoss() 
    y = y.reshape((y.shape[0],1))
    y_onehot = torch.FloatTensor(x.shape[0], x.shape[1])
    y_onehot.zero_() 
    y_onehot.scatter_(1, y, 1)
    return mse_loss(x, y_onehot)

def evaluate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    return test_loss, correct

"""
def evaluate(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            corr, test_l = eval_(X, y, model, loss_fn)
            test_loss += test_l
            correct += corr
    test_loss /= size
    correct /= size
    return test_loss, correct
"""

def eval_(X, y, model, loss_fn):
    with torch.no_grad():
        pred = model(X)
        test_loss = loss_fn(pred, y).item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
    return test_loss, correct
