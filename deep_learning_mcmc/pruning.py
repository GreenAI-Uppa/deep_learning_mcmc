from torch import nn
import torch
import numpy as np
import copy
from abc import ABC, abstractmethod
from deep_learning_mcmc import nets




#######################
#Mozer pruning function
#######################

class MozerPruner():
    def forward_alpha(self,model,alpha, x):
        '''
        forward with continuous dropout at fc layer
        -model: initial model (ConvNet instance)
        -alpha: coefficient added to each output unit of the first layer
        -x: input
        '''
        x = model.activations[0](model.conv1(x))
        x = x.view(-1, model.nb_filters * 8 * 8)
        x = [torch.mul(elt,alpha) for elt in x]
        x = torch.stack(x)
        x = model.fc1(x)
        return x

    def forward_alpha_conv(self,model,alpha, x):
        '''
        forward with continuous dropout at conv layer
        -model: initial model (ConvNet instance)
        -alpha: coefficient added to each filter of the first layer
        -x: input
        '''
        x = model.activations[0](model.conv1(x))
        for idx_filter in range(model.nb_filters):
            x[:,idx_filter,:,:] *= alpha[idx_filter]
        x = x.view(-1, model.nb_filters * 8 * 8)
        x = model.fc1(x)
        return x

    def relevance(self,model,dataloader):
        '''
        compute relevance coefficients based on Mozer and Smolesky 1989
        -model: model analyzed
        -dataloader: dataset to compute the gradient of the loss at alpha=1
        '''
        autograd_tensor = torch.ones((model.nb_filters * 8 * 8), requires_grad=True) # size is 4096; input of hidden layer
        num_items_read = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        device = next(model.parameters()).device
        autograd_tensor = autograd_tensor.to(device)
        gg = []
        lengths = []
        for _, (X, y) in enumerate(dataloader):
            if 1000000 <= num_items_read:
                break
            X = X[:min(1000000 - num_items_read, X.shape[0])]
            y = y[:min(1000000 - num_items_read, X.shape[0])]
            num_items_read = min(1000000, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            pred = self.forward_alpha(model,autograd_tensor,X)
            loss = loss_fn(pred, y)
            gg.append(torch.autograd.grad(loss, autograd_tensor, retain_graph=True))
            lengths.append(X.shape[0])  # contains the batch size
        normalization = torch.tensor([elt/sum(lengths) for elt in lengths])
        # transform the list to a tensor. Each line is one loop iteration (ie one list element)
        tensor_gg = torch.tensor([list(gg[k][0]) for k in range(len(gg))]) # 40 x 4096
        result = [torch.sum(torch.mul(normalization,elt)) for elt in [tensor_gg[:,k] for k in range(tensor_gg.shape[1])]]
        return(-torch.tensor(result)) # tensor of size 4096

    def relevance_conv(self,model,dataloader):
        '''
        compute relevance coefficients based on Mozer and Smolesky 1989
        -model: model analyzed
        -dataloader: dataset to compute the gradient of the loss at alpha=1
        '''
        autograd_tensor = torch.ones((model.nb_filters), requires_grad=True)
        num_items_read = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        device = next(model.parameters()).device
        autograd_tensor = autograd_tensor.to(device)
        gg = []
        lengths = []
        for _, (X, y) in enumerate(dataloader):
            if 1000000 <= num_items_read:
                break
            X = X[:min(1000000 - num_items_read, X.shape[0])]
            y = y[:min(1000000 - num_items_read, X.shape[0])]
            num_items_read = min(1000000, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            pred = self.forward_alpha_conv(model,autograd_tensor,X)
            loss = loss_fn(pred, y)
            gg.append(torch.autograd.grad(loss, autograd_tensor, retain_graph=True))
            lengths.append(X.shape[0])
        normalization = torch.tensor([elt/sum(lengths) for elt in lengths])
        tensor_gg = torch.tensor([list(gg[k][0]) for k in range(len(gg))])
        result = [torch.sum(torch.mul(normalization,elt)) for elt in [tensor_gg[:,k] for k in range(tensor_gg.shape[1])]]
        return(-torch.tensor(result))


    def skeletonize(self,model,pruning_level,dataloader):
        '''
        skeletone the fc layer based on Mozer and Smolesky with the order statistic of the relevance coefficient of each hidden unit
        -model: model to prune
        -pruning_level: pourcentage of weights of the fc layer let to zero
        -dataloader: dataset to compute the relevance function above
        '''
        relevance_ = self.relevance(model,dataloader)
        size = int(model.fc1.weight.data.shape[1]*(1-pruning_level))
        keep_indices = torch.argsort(-relevance_)[:size]
        device = next(model.parameters()).device
        cpt = 0
        for index in set(range(model.fc1.weight.data.shape[1]))-set([int(elt) for elt in keep_indices]):
            cpt+=1
            #skeletone.fc1.weight.data[:,index] = torch.zeros(10)
            model.fc1.weight.data[:,index] = torch.zeros(10)
        loss_fn = torch.nn.CrossEntropyLoss()
        #print('test accuracy',nets.evaluate(dataloader,model,loss_fn)[1],'after skeletonization')
        return()

    def skeletonize_conv(self,model,pruning_level,dataloader):
        '''
        skeletone the fc layer based on Mozer and Smolesky with the order statistic of the relevance coefficient of each hidden unit
        -model: model to prune
        -pruning_level: pourcentage of weights of the fc layer let to zero
        -dataloader: dataset to compute the relevance function above
        '''
        relevance_ = self.relevance_conv(model,dataloader)
        size = int(model.nb_filters*(1-pruning_level))
        # selection of the end of the list, ie the ones with low relevance
        remove_indices = torch.argsort(-relevance_)[size:]
        device = next(model.parameters()).device
        cpt = 0
        # prune the selected filters, ie set them to 0
        for index in remove_indices:  #set(range(model.fc1.weight.data.shape[1]))-set([int(elt) for elt in keep_indices]):
            model.conv1.weight.data[index] = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        print('test accuracy',nets.evaluate(dataloader,model,loss_fn)[1],'after skeletonization')
        return()

#######################
#MCMC pruning Mozer inspired
#######################


class MCMCPruner():
    def skeletonize_conv(self,model,pruning_level,relevance):
        '''
        simple MCMC skeletonization of the convolutional layer learnt by the MCMC ietrations itself
        -model: model to prune
        -pruning_level: pourcentage of coefficients killed
        -relevance: dictionnary with keys=filter index and values=number of accepts in the mcmc optimizer (we keep the filters with biggest accepts)
        '''
        size = int(model.conv1.weight.data.shape[0]*(1-pruning_level))
        relevance_list = torch.tensor([relevance[k] for k in range(model.conv1.weight.data.shape[0])])
        keep_indices = torch.argsort(-relevance_list)[:size]
        device = next(model.parameters()).device
        for index in set(range(model.conv1.weight.data.shape[0]))-set([int(elt) for elt in keep_indices]):
            model.conv1.weight.data[index,:,:,:] = torch.zeros([3,11,11])
        return()

    def skeletonize(self,model,pruning_level,relevance):
        '''
        simple MCMC skeletonization of the convolutional layer learnt by the MCMC ietrations itself
        -model: model to prune
        -pruning_level: pourcentage of coefficients killed
        -relevance: dictionnary with keys=filter index and values=number of accepts in the mcmc optimizer (we keep the filters with biggest accepts)
        '''
        n_output, n_input = model.fc1.weight.data.shape
        size = int((n_output * n_input + n_output)*(1-pruning_level))
        relevance = torch.cat((relevance['weight'], relevance['bias']),dim=1)
        remove_indices = torch.argsort(torch.flatten(-relevance))[size:]
        remove_rows = torch.div(remove_indices, relevance.shape[1], rounding_mode='floor')
        remove_cols = remove_indices % relevance.shape[1]
        selected_bias = remove_cols == n_input
        model.fc1.weight.data[remove_rows[torch.logical_not(selected_bias)],remove_cols[torch.logical_not(selected_bias)]] = 0
        model.fc1.bias.data[remove_cols[selected_bias]] = 0
        return()

#######################
#End of Mozer pruning function
#######################


