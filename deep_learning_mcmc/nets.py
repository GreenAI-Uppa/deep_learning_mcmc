from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

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
            print('using elu activation')
            self.activation = nn.ELU()
        else:
            print('using Relu activation')
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        for linear in self.linears:
            x = linear(x)
            x = self.activation(x)
        return x


class ConvNet(nn.Module):
  def __init__(self,nb_filters,channels):
    super(ConvNet, self).__init__()
    self.nb_filters = nb_filters
    self.channels = channels
    if channels == 3:
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=nb_filters, kernel_size=11, stride=3, padding=0)
    else:
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=nb_filters, kernel_size=7, stride=3, padding=0)
    #self.pool = nn.MaxPool2d(2, 2)
    #self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(self.nb_filters * 8 * 8, 10)
    #self.fc2 = nn.Linear(120, 84)
    #self.fc3 = nn.Linear(84, 10)
  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = x.view(-1, self.nb_filters * 8 * 8)
    x = F.relu(self.fc1(x))
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


def evaluate_sparse(dataloader, model, loss_fn, proba):
    """
    evaluate a sparse version of a linear model
    dataloader, model, and loss_fn : see evaluate function
    threshold : values of the threshold in the weights matrix associated to the first layer of the MLP (not apply to the bias term)
    Return loss, acccuracy and the percentage of values kept after threshold in the first layer
    """
    device = next(model.parameters()).device
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    model_sparse = ConvNet(model.nb_filters,model.channels)
    model_sparse = model_sparse.to(device)
    #if quantile:
    q1 = torch.quantile(torch.flatten(torch.abs(model.conv1.weight.data)),proba, dim=0)
    print('Quantile 1',q1)
    bin_mat1 = torch.abs(model.conv1.weight.data) > q1
    q2 = torch.quantile(torch.flatten(torch.abs(model.fc1.weight.data)),proba, dim=0)
    bin_mat2 = torch.abs(model.fc1.weight.data)> q2
    print('Quantile 2',q2)
    '''
    bin_mat1 = torch.abs(model.conv1.weight.data)>threshold
    bin_mat2 = torch.abs(model.fc1.weight.data)>threshold
    '''
    model_sparse.conv1.weight.data = (bin_mat1)*model.conv1.weight.data
    model_sparse.fc1.weight.data = (bin_mat2)*model.fc1.weight.data
    print(torch.sum(bin_mat1)+torch.sum(bin_mat2),'parameters kept over',torch.flatten(bin_mat1).shape[0]+torch.flatten(bin_mat2).shape[0])
    #print('Model is set to Sparse Version')
    #model = model_sparse
    kept = float((torch.sum(bin_mat1)+torch.sum(bin_mat2))/(float(torch.flatten(bin_mat1).shape[0])+float(torch.flatten(bin_mat2).shape[0])))
    with torch.no_grad():
        if 'MLP' in str(type(model)):
            for X, y in dataloader:
                pred_s = []
                for x in X:
                    y_pred = torch.matmul(sparse_weights,torch.flatten(x))+model.linears[0].bias.data
                    output = m(y_pred)
                    pred_s.append(np.array(output))
                pred_s = torch.from_numpy(np.array(pred_s))
                test_loss += loss_fn(pred_s, y).item()
                correct += (pred_s.argmax(1) == y).type(torch.float).sum().item()
        else:
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)
                pred_s = model_sparse(X)
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

