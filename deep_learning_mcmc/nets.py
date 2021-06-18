from torch import nn
import torch
import numpy as np

class MLP(nn.Module):
    def __init__(self, sizes, act='relu'):
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
        for i in range(1, len(sizes)):
            self.linears.append(nn.Linear(sizes[i-1], sizes[i]))
        if act == 'soft':
            print('using softmax activation')
            self.activation = nn.Softmax()
        elif act =='elu':
            print('using rely activation')
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)
        return x

loss = nn.MSELoss()
def my_mse_loss(x,y):
    mse_loss = nn.MSELoss()
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


def evaluate_sparse(dataloader, model, loss_fn, threshold):
    """
    evaluate a sparse version of a linear model
    dataloader, model, and loss_fn : see evaluate function
    threshold : values of the threshold in the weights matrix associated to the first layer of the MLP (not apply to the bias term)
    Return loss, acccuracy and the percentage of values kept after threshold in the first layer
    """
    device = next(model.parameters()).device
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    bin_mat = torch.abs(model.linears[0].weight.data)>threshold
    kept = float(torch.sum(bin_mat)/float(torch.flatten(bin_mat).shape[0]))
    sparse_weights = model.linears[0].weight.data*(bin_mat)
    m = nn.ELU()            
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred_s = []
            for x in X:
                y_pred = torch.matmul(sparse_weights,torch.flatten(x))+model.linears[0].bias.data
                output = m(y_pred)
                pred_s.append(np.array(output))
            pred_s = torch.from_numpy(np.array(pred_s))
            test_loss += loss_fn(pred_s, y).item()
            correct += (pred_s.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    return test_loss, correct, kept


def eval_(X, y, model, loss_fn):
    with torch.no_grad():
        pred = model(X)
        test_loss = loss_fn(pred, y).item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
    return test_loss, correct

