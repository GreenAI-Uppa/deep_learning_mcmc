from torch import nn
import torch
import numpy as np
import copy
from abc import ABC, abstractmethod
import torch.nn.functional as F


class Conv2d4MCMC(nn.Conv2d):
    is_binary = False
    def update(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1],idces_w[:,2],idces_w[:,3]] += proposal[:idces_w.shape[0]]
        if idces_b.shape[0] !=0 :
            self.bias.data[idces_b] += proposal[idces_w.shape[0]:]

    def get_selected_size(self, idces):
        idces_w, idces_b = idces
        return idces_w.shape[0] + idces_b.shape[0]

    def undo(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1],idces_w[:,2],idces_w[:,3]] -= proposal[:idces_w.shape[0]]
        if idces_b.shape[0] !=0 :
            self.bias.data[idces_b] -= proposal[idces_w.shape[0]:]

    def getParamLine(self, idces):
        idces_w, idces_b = idces
        w = self.weight.data[idces_w[:,0],idces_w[:,1],idces_w[:,2],idces_w[:,3]]
        b = self.bias.data[idces_b]
        return torch.cat((w,b))

    def get_idx_flattened_1_filter(self, idx_filter):
        """
        return a num_params x 4 tensor which contains the indices of one filter coefficients
        """
        channels, k1, k2 = self.weight.data[0].shape
        idces_w = torch.ones(0,3)
        for i in range(channels):
            t1 = torch.ones((k1*k2,1)) * i
            tmp = torch.ones(0,2)
            for j in range(k1):
                t2 = torch.ones((k2,1)) * j
                t3 = torch.arange(0,k2).reshape(k2,1)
                tmp = torch.cat((tmp, torch.cat((t2,t3), dim=1) ))
            idces_w = torch.cat((idces_w, torch.cat((t1,tmp), dim=1) ))
        idces_w = torch.cat((torch.ones(k1*k2*channels,1)*idx_filter, idces_w), dim=1).long()
        return idces_w


class BinaryConv2d(Conv2d4MCMC):
    is_binary = True
    def __init__(self, *args, **kwargs):
        super(BinaryConv2d, self).__init__(*args, **kwargs)
        self.weight.data = torch.sign(self.weight.data)
        self.bias.data = torch.sign(self.bias.data)

    def update(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1],idces_w[:,2],idces_w[:,3]] *= proposal[:idces_w.shape[0]]
        if idces_b.shape[0] !=0 :
            self.bias.data[idces_b] *= proposal[idces_w.shape[0]:]

    def undo(self, neighborhood, proposal):
        self.update(neighborhood, proposal)


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
        self.weight.data = torch.sign(self.weight.data)
        self.bias.data = torch.sign(self.bias.data)

    def update(self, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        self.weight.data[idces_w[:,0],idces_w[:,1]] *= proposal[:idces_w.shape[0]]
        self.bias.data[idces_b] *= proposal[idces_w.shape[0]:]

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
        activations[-1] = None # because the loss function contains its own activation

    def forward(self, x):
        x = self.flatten(x)
        for linear, activation in zip(self.layers, self.activations):
            x = linear(x)
            if activation is not None:
                x = activation(x)
        return x

class ConvNet(nn.Module):
    '''
    A simple ConvNet constructor
    nb_filters = number of filters for the convolution layer
    channels = number of input channels (3 for CIFAR-10, 1 for MNIST)
    binary_flags = list of 2 booleans to binarize the conv layer or the fc layer (1 = binarization)
    activations = list of activations by layer
    init_sparse = boolean (1 = Student heavy tailed initialization)
    pruning_level = exact sparsity coefficient at init and for proposal epsilon or gradient steps
    '''
    def __init__(self,nb_filters,channels, binary_flags=None, activations=None, init_sparse=False, pruning_level=0):
        super(ConvNet, self).__init__()
        self.nb_filters = nb_filters
        self.channels = channels
        self.init_sparse = init_sparse
        self.layers = nn.ModuleList()
        self.pruning_level = pruning_level
        if binary_flags and binary_flags[0]:
            if channels == 3:
                self.conv1 = BinaryConv2d(in_channels=channels, out_channels=nb_filters, kernel_size=11, stride=3, padding=0)
            else:
                self.conv1 = BinaryConv2d(in_channels=channels, out_channels=nb_filters, kernel_size=7, stride=3, padding=0)
        else:
            if channels == 3:
                self.conv1 = Conv2d4MCMC(in_channels=channels, out_channels=nb_filters, kernel_size=11, stride=3, padding=0)
                if init_sparse:
                    print('INIT SPARSE for CIFAR10')
                    init_values = self.init_sparse.sample(n=nb_filters*channels*11*11)
                    self.conv1.weight.data = torch.tensor(init_values.astype('float32')).reshape((nb_filters,channels,11,11))
                    q1 = torch.quantile(torch.flatten(torch.abs(self.conv1.weight.data)),self.pruning_level, dim=0)
                    bin_mat = torch.abs(self.conv1.weight.data) > q1
                    self.conv1.weight.data = (bin_mat)*self.conv1.weight.data
            else:
                self.conv1 = Conv2d4MCMC(in_channels=channels, out_channels=nb_filters, kernel_size=7, stride=3, padding=0)
                if init_sparse:
                    print('INIT SPARSE for MNIST')
                    init_values = self.init_sparse.sample(n=nb_filters*7*7)
                    self.conv1.weight.data = torch.tensor(init_values.astype('float32')).reshape((nb_filters,channels,7,7))
                    q1 = torch.quantile(torch.flatten(torch.abs(self.conv1.weight.data)),self.pruning_level, dim=0)
                    bin_mat = torch.abs(self.conv1.weight.data) > q1
                    self.conv1.weight.data = (bin_mat)*self.conv1.weight.data
        self.layers.append(self.conv1)
        if binary_flags and binary_flags[1]:
            self.fc1 = BinaryLinear(self.nb_filters * 8 * 8, 10)
        else:
            self.fc1 = Linear4MCMC(self.nb_filters * 8 * 8, 10)
            if init_sparse:
                init_values_fc = self.init_sparse.sample(n=10*self.nb_filters * 8 * 8)
                self.fc1.weight.data = torch.tensor(init_values_fc.astype('float32')).reshape((10,self.nb_filters*8*8))
                q1 = torch.quantile(torch.flatten(torch.abs(self.fc1.weight.data)),self.pruning_level, dim=0)
                bin_mat = torch.abs(self.fc1.weight.data) > q1
                self.fc1.weight.data = (bin_mat)*self.fc1.weight.data
        self.layers.append(self.fc1)
        self.activations = []
        if activations is None:
            self.activations = [nn.ReLU() for i in self.layers ]
        else:
            self.activations = [ getattr(nn, activations[i])() for i in range(len(self.layers)) ]

    def forward(self, x):
        x = self.activations[0](self.conv1(x))
        x = x.view(-1, self.nb_filters * 8 * 8)
        x = self.fc1(x)
        return x

class BinaryConnectConv(ConvNet):
    def __init__(self,nb_filters,channels, binary_flags=[True, False], activations=None, init_sparse=False, pruning_level=0):

        # building a non binary convnet first
        super().__init__(nb_filters,channels, binary_flags=None, activations=activations, init_sparse=init_sparse, pruning_level=pruning_level)
        """copy the real layers and binarize the others accoding to the binary_flags"""
        if binary_flags is None:
            raise Exception("binary_flags is None, it doesn't make sense to use BinaryConnect without binary_layers. Please set, for instance, binary_flags=[True, False] ")
        # binarizing the required layers and saving a copy of the real weights
        self.layers_reals = []
        for i, layer in enumerate(self.layers):
             if binary_flags[i]:
                 self.layers_reals.append([copy.deepcopy(layer.weight.data), copy.deepcopy(layer.bias.data)])
                 layer.weight.data = np.sign(layer.weight.data)
                 layer.bias.data = np.sign(layer.bias.data)
             else:
                 self.layers_reals.append(None)

    def to(self, device):
        model =super().to(device)
        for layer_real in model.layers_reals:
            if layer_real is not None:
                layer_real[0] = layer_real[0].to(device)
                layer_real[1] = layer_real[1].to(device)
        return model


class AlexNet(nn.Module):
    '''
    AlexNet constructor
    nb_filters = list of filters for convolution layers
    channels = number of input channels (3 for CIFAR-10, 1 for MNIST)
    binary_flags = list of boolean to binarize layers (1 = binarization)
    activations = list of activations by layer
    init_sparse = boolean (1 = Student heavy tailed initialization)
    pruning_level = exact sparsity coefficient at init and for proposal epsilon or gradient steps
    '''
    def __init__(self,nb_filters,channels, kernel_sizes, strides, paddings, binary_flags=None, activations=None, init_sparse=False, pruning_level=0):
        super(AlexNet, self).__init__()
        self.nb_filters = nb_filters
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.init_sparse = init_sparse
        self.layers = nn.ModuleList()
        self.pruning_level = pruning_level
        #conv layers constructor
        in_channels = [channels]
        for k,binary_flag in enumerate(binary_flags[:5]):
            #loop over convolution layers
            if binary_flag:
                self.layers.append(BinaryConv2d(in_channels=in_channels[k], out_channels=nb_filters[k], kernel_size=kernel_sizes[k], stride=strides[k], padding=paddings[k]))
            else:
                self.layers.append(Conv2d4MCMC(in_channels=in_channels[k], out_channels=nb_filters[k], kernel_size=kernel_sizes[k], stride=strides[k], padding=paddings[k]))
                if init_sparse:
                    print('INIT SPARSE for layer',k)
                    init_values = self.init_sparse.sample(n=nb_filters[k]*in_channels[k]*kernel_sizes[k]*kernel_sizes[k])
                    self.layers[k].weight.data = torch.tensor(init_values.astype('float32')).reshape((nb_filters[k],in_channels[k],kernel_sizes[k],kernel_sizes[k]))
                    '''
                    exact sparsity impossible: quantile function do not "scale" to alexnet :)
                    q1 = torch.quantile(torch.flatten(torch.abs(self.layers[k].weight.data)),self.pruning_level, dim=0)
                    bin_mat = torch.abs(self.layers[k].weight.data) > q1
                    self.layers[k].weight.data = (bin_mat)*self.layers[k].weight.data
                    '''
            in_channels.append(nb_filters[k])
            print('init of layer',k+1,'OK')
        #fully connected layers contructor
        i_o_list = [(nb_filters[-1] * 6 * 6, 4096),(4096, 4096),(4096, 10)]
        for k,binary_flag in enumerate(binary_flags[5:]):
            if binary_flag:
                self.layers.append(BinaryLinear(i_o_list[k][0],i_o_list[k][1]))
            else:
                self.layers.append(Linear4MCMC(i_o_list[k][0],i_o_list[k][1]))
                if init_sparse:
                    print('INIT SPARSE for layer',k+5)
                    init_values_fc = self.init_sparse.sample(n=i_o_list[k][0]*i_o_list[k][1])
                    self.layers[k+5].weight.data = torch.tensor(init_values_fc.astype('float32')).reshape((i_o_list[k][1],i_o_list[k][0]))
                    '''
                    exact sparsity impossible: quantile function do not "scale" to alexnet :)
                    q1 = torch.quantile(torch.flatten(torch.abs(self.layers[k+5].weight.data)),self.pruning_level, dim=0)
                    bin_mat = torch.abs(self.layers[k+5].weight.data) > q1
                    self.layers[k+5].weight.data = (bin_mat)*self.layers[k+5].weight.data
                    '''
            print('init of layer',k+6,'OK')
        self.activations = []
        if activations is None:
            self.activations = [nn.ReLU(inplace=True) for i in self.layers]
        else:
            self.activations = [ getattr(nn, activations[i])() for i in range(len(self.layers)) ]
    def forward(self, x):
        for k,layer in enumerate(self.layers[:2]):
            x = self.activations[k](layer(x))
            x = nn.MaxPool2d(kernel_size = 2, stride = 2)(x)
        for k,layer in enumerate(self.layers[2:5]):
            x = self.activations[k+2](layer(x))
        x = nn.MaxPool2d(kernel_size = 3, stride = 2)(x)
        x = nn.AdaptiveAvgPool2d((6, 6))(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = nn.Dropout(p = 0.5)(x)
        x = x.view(-1, self.nb_filters[4] * 6 * 6)
        for k,layer in enumerate(self.layers[5:]):
            x = self.layers[k+5](x)
            if k != 2:
                x = self.activations[k+5](x)
            if k == 0:
                x = nn.Dropout(p = 0.5)(x)
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
    test_loss /= size/dataloader.batch_size
    correct /= size
    return test_loss, correct

def evaluate_sparse(dataloader, model, loss_fn, proba,boolean_flags,fc=True):
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
    #shrinkage of the convlayer
    if boolean_flags[0] != 1:
        #weights
        q1 = torch.quantile(torch.flatten(torch.abs(model.conv1.weight.data)),proba, dim=0)
        bin_mat1 = torch.abs(model.conv1.weight.data) > q1
        bin_mat1 = bin_mat1.to(device)
        model_sparse.conv1.weight.data = (bin_mat1)*model.conv1.weight.data
        #bias
        q1bias = torch.quantile(torch.flatten(torch.abs(model.conv1.bias.data)),proba, dim=0)
        bin_mat1bias = torch.abs(model.conv1.bias.data)> q1bias
        bin_mat1bias = bin_mat1bias.to(device)
        model_sparse.conv1.bias.data = (bin_mat1bias)*model.conv1.bias.data
    else:
        model_sparse.conv1.weight.data = model.conv1.weight.data
        model_sparse.conv1.bias.data = model.conv1.bias.data        
    if fc and boolean_flags[1]!= 1:
        #shrinkage of fc layer
        #weights
        q2 = torch.quantile(torch.flatten(torch.abs(model.fc1.weight.data)),proba,dim=0)
        bin_mat2 = torch.abs(model.fc1.weight.data) > q2
        bin_mat2 = bin_mat2.to(device)
        model_sparse.fc1.weight.data = (bin_mat2)*model.fc1.weight.data
        #bias
        q2bias = torch.quantile(torch.flatten(torch.abs(model.fc1.bias.data)),proba, dim=0)
        bin_mat2bias = torch.abs(model.fc1.bias.data) > q2bias
        bin_mat2bias = bin_mat2bias.to(device)
        model_sparse.fc1.bias.data = (bin_mat2bias)*model.fc1.bias.data
    else:
        model_sparse.fc1.weight.data = model.fc1.weight.data
        model_sparse.fc1.bias.data = model.fc1.bias.data
    if fc and boolean_flags[1] != 1:
        if boolean_flags[0] == 0:
            kept = float((torch.sum(bin_mat1)+torch.sum(bin_mat2))/(float(torch.flatten(bin_mat1).shape[0])+float(torch.flatten(bin_mat2).shape[0])))
        else:
            kept = float(torch.sum(bin_mat2)/float(torch.flatten(bin_mat2).shape[0]))
    else:
        if boolean_flags[0] == 0:
            kept = float(torch.sum(bin_mat1))/float(torch.flatten(bin_mat1).shape[0])
        else:
            kept = 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred_s = model_sparse(X)
            test_loss += loss_fn(pred_s, y).item()
            correct += (pred_s.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    return test_loss, correct, kept
