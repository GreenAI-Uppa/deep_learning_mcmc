from abc import ABC, abstractmethod
import torch
import random
import sys
current_module = sys.modules[__name__]
from typing import Tuple


def get_idces_uniform_linear(neighborhood_size):
    """
    select neighborhood_size weights from a linear layer according to a uniform law
    """
    def get_idx(layer):
        n_output, n_input = layer.weight.data.shape
        f_size = n_output * n_input
        if neighborhood_size > f_size:
            raise Exception("neighborhood_size is "+str(neighborhood_size)+" but number of parameters in the filter is "+str(f_size)+"\n neighborhood_size should be lower than this number")
        a, b = torch.arange(n_output), torch.arange(n_input)
        idces_w = torch.cat((a.repeat(n_input).reshape(f_size,1),b.repeat_interleave(n_output).reshape(f_size,1)), dim=1)
        idces_of_idces = list(range(f_size))
        random.shuffle(idces_of_idces)
        idces_of_idces  = idces_of_idces[:neighborhood_size]
        idces_w = idces_w[idces_of_idces,:].long()
        idces_b = idces_w[idces_w[:,1]==n_input][:,0]
        idces_w = idces_w[idces_w[:,1]<n_input]
        return idces_w, idces_b
    return get_idx

def get_idces_linear_square(square_side):
    """
    select neighborhood_size weights from a linear layer according to a uniform law
    """
    def get_idx(layer):
        n_output, n_input = layer.weight.data.shape
        f_size = n_output * n_input
        if neighborhood_size > f_size:
            raise Exception("neighborhood_size is "+str(neighborhood_size)+" but number of parameters in the filter is "+str(f_size)+"\n neighborhood_size should be lower than this number")
        a, b = torch.arange(n_output), torch.arange(n_input)
        idces_w = torch.cat((a.repeat(n_input).reshape(f_size,1),b.repeat_interleave(n_output).reshape(f_size,1)), dim=1)
        idces_of_idces = list(range(f_size))
        random.shuffle(idces_of_idces)
        idces_of_idces  = idces_of_idces[:neighborhood_size]
        idces_w = idces_w[idces_of_idces,:].long()
        idces_b = idces_w[idces_w[:,1]==n_input][:,0]
        idces_w = idces_w[idces_w[:,1]<n_input]
        return idces_w, idces_b
    return get_idx

def get_idces_line_linear():
    """
    select one row of a fully connected layer
    """
    def get_idx(layer):
        n_output, n_input = layer.weight.data.shape
        idx_row = torch.randint(0, n_output, (1,))
        idces_w = torch.cat((torch.ones(n_input,1)*idx_row, torch.arange(0,n_input).reshape(n_input, 1) ),dim=1).long()
        idces_b = idx_row
        return idces_w, idces_b
    return get_idx

def get_idces_filter_conv():
    """
    select one filter and gives the whole filter as a selected neighborhood
    """
    def get_idx(layer):
        n_filter, channels, k1, k2 = layer.weight.data.shape
        idx_filter = torch.randint(0, n_filter, (1,))
        idces_w = layer.get_idx_flattened_1_filter(idx_filter)
        return idces_w, idx_filter
    return get_idx

# TODO number of nb_filters


def get_idces_uniform_conv(neighborhood_size):
    """
    select neighborhood_size weights from a conv layer according to a uniform law
    """
    def get_idx(layer):
        n_filter, channels, k1, k2 = layer.weight.data.shape
        idx_filter = torch.randint(0, n_filter, (1,))
        filter_idces = layer.get_idx_flattened_1_filter(idx_filter)
        f_size = filter_idces.shape[0]
        if neighborhood_size > f_size:
            raise Exception("neighborhood_size is "+str(neighborhood_size)+" but number of parameters in the filter is "+str(f_size)+"\n neighborhood_size should be lower than this number")
        idces_of_idces = list(range(f_size))
        random.shuffle(idces_of_idces)
        idces_of_idces  = idces_of_idces[:neighborhood_size]
        idces_w = filter_idces[idces_of_idces,:].long()
        return idces_w, idx_filter
    return get_idx

def build_selector(config):
    selector_name = config["name"]
    selector_class = getattr(current_module, selector_name)
    return selector_class(config)

class Selector():
    """
    select one layer and select a proposal according its config over this layer.
    """
    def __init__(self, config):
        number_of_layers = len(config['layer_conf'])
        if 'layer_distr' not in config['layer_conf'][0]:
            self.layer_distr = [1/number_of_layers for i in range(number_of_layers)]
        else:
            self.layer_distr = [ layer_config['layer_distr'] for layer_config in config['layer_conf'] ]
        self.layer_distr = torch.cumsum(torch.Tensor(self.layer_distr),0)
        self.config = config['layer_conf']

    def set_neighborhood_info(self, neighborhood, layer):
        """
        simply return the size of the neighborhood considered

        idces : indices of the neighborhood considered
        layer : pytorch object layer from the neighborhood has been extracted
        """
        layer_idx, idces = neighborhood
        self.neighborhood_info = layer_idx, layer.get_selected_size(idces)

    def get_layer_idx(self):
        """
        randomly selects a layer according to the distribution given by
        self.layer_distr
        """
        seed = torch.rand(1)
        layer_idx = 0
        for i, x in enumerate(self.layer_distr):
            if x > seed:
                break
            layer_idx += 1
        return layer_idx

    def get_neighborhood(self, model : torch.nn.Module):
        """
        select one layer of the model
        select a subset of parameters from the selected layer

        return
        layer_idx : the index of the selected layer
        idces : the idces of the weights for the selected layer
        """
        layer_idx = self.get_layer_idx()
        idces = self.config[layer_idx]['get_idx'](model.layers[layer_idx])
        return layer_idx, idces

    def getParamLine(self, neighborhood, model):
        """
        return the parameter value of the selected neighborhood as a line

        neighborhood : the idces of the neighborhood considered
        """
        layer_idx, idces = neighborhood
        self.set_neighborhood_info(neighborhood, model.layers[layer_idx])
        return model.layers[layer_idx].getParamLine(idces)

    def update(self, model, neighborhood, proposal):
        """
        update the model weight given a proposal.
        Since the proposal
        comes out as 1D vector, the information contained in the
         neighborhood parameter allows to select the weights which should
         be updated


        model : the model to be updated
        neighborhood : the required information to select the weights which
        should be updated by the proposal
        proposal : the value for the new weight as a 1D vector
        """
        layer_idx, idces = neighborhood
        model.layers[layer_idx].update(idces, proposal)

    def undo(self, model, neighborhood, proposal):
        """
        inverse function of update
        """
        layer_idx, idces = neighborhood
        model.layers[layer_idx].undo(idces, proposal)

    def get_proposal_as_string(self, neighborhood):
        """
        return the name of the layer related to the considered neighborhood
        This string can be used to keep track of the acceptance ratio
        of the different proposal

        neighborhood : contains the information needed to explain where (ie in
        which layer) the proposal has been applied.
        """
        layer_idx, idces = neighborhood
        return 'layer_'+str(layer_idx)
