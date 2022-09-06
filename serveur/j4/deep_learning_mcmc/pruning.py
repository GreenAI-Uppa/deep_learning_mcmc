from torch import nn
import torch
import numpy as np
import copy
from abc import ABC, abstractmethod
from deep_learning_mcmc import nets

#######################
#MCMC pruning class
#######################


class MCMCPruner():
    def skeletonize_conv(self,model,pruning_level,relevance):
        '''
        simple MCMC skeletonization of the convolutional layer learnt by the MCMC iterations itself
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

    def skeletonize_fc(self,model,pruning_level,relevance):
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
        model.fc1.bias.data[remove_rows[selected_bias]] = 0
        return()