from abc import ABC, abstractmethod
import torch
import sys
current_module = sys.modules[__name__]
from typing import Tuple


def get_idces_uniform_linear(neighborhood_size):
    """
    will select neighborhood_size weights from a linear layer according to a uniform law
    """
    def get_idx(layer):
        n_output, n_input = layer.weight.data.shape
        idces_w = torch.cat((torch.randint(0,n_output,(neighborhood_size,1)), torch.randint(0,n_input+1,(neighborhood_size,1))), dim=1)
        idces_b = idces_w[idces_w[:,1]==n_input][:,0]
        idces_w = idces_w[idces_w[:,1]<n_input]
        return idces_w, idces_b
    return get_idx

def get_idces_line_linear(layer):
    """
    will select one row of a linear layer
    """
    n_output, n_input = layer.weight.data.shape
    idx_row = torch.randint(0, n_output, (1,))
    idces_w = torch.cat((torch.ones(n_input,1)*idx_row, torch.arange(0,n_input).reshape(n_input, 1) ),dim=1).long()
    idces_b = idx_row
    return idces_w, idces_b


def get_idces_filter_conv(layer):
    """
    will select one row of a linear layer
    """
    n_filter, channels, k1, k2 = layer.weight.data.shape
    idx_filter = torch.randint(0, n_filter, (1,))
    n_params = channels * k1 * k2
    idces_w = layer.get_all_idx_flattened()
    return idces_w, idx_filter

def get_idces_uniform_conv(neighborhood_size):
    """
    will select neighborhood_size weights from a linear layer according to a uniform law
    """
    def get_idx(layer):
        """
        will select one row of a linear layer
        """
        n_filter, channels, k1, k2 = layer.weight.data.shape
        idx_filter = torch.randint(0, n_filter, (1,))
        idx_f = torch.ones((neighborhood_size,1)) * idx_filter
        idx_chan = torch.randint(0, channels, (neighborhood_size,1) )
        idx_k1 = torch.randint(0, k1, (neighborhood_size,1))
        idx_k2 = torch.randint(0, k2, (neighborhood_size,1))
        idces_w = torch.cat((idx_f, idx_chan, idx_k1, idx_k2), dim=1)
        return idces_w, idx_filter
    return get_idx

def build_selector(config):
    selector_name = config["name"]
    selector_class = getattr(current_module, selector_name)
    return selector_class(config)

class Selector(ABC):
    @abstractmethod
    def __init__(sizes, config):
        self.sizes = sizes

    @abstractmethod
    def get_neighborhood(model):
        pass

    @abstractmethod
    def getParamLine(neighborhood, model):
        pass

    @abstractmethod
    def update(model, neighborhood, proposal):
        pass

    @abstractmethod
    def update(model, neighborhood, proposal):
        pass

    @abstractmethod
    def undo(model, neighborhood, proposal):
        pass

    def get_proposal_as_string(self, neighborhood):
        return "layer_x"

class UniformSelector(Selector):
    """
    select one layer and pick some weights with a uniform law over this layer.
    """
    def __init__(self, config):
        number_of_layers = len(config['layer_conf'])
        if 'layer_distr' not in config['layer_conf'][0]:
            self.layer_distr = [1/number_of_layers for i in range(number_of_layers)]
        else:
            self.layer_distr = [ layer_config['layer_distr'] for layer_config in config['layer_conf'] ]
        self.layer_distr = torch.cumsum(torch.Tensor(self.layer_distr),0)
        self.config = config['layer_conf']

    def set_neighborhood_info(self, idces, layer):
        self.neighborhood_info = layer.get_selected_size(idces)

    def get_layer_idx(self):
        seed = torch.rand(1)
        layer_idx = 0
        for i, x in enumerate(self.layer_distr):
            if x > seed:
                break
            layer_idx += 1
        return layer_idx

    def get_neighborhood(self, model):
        layer_idx = self.get_layer_idx()
        idces = self.config[layer_idx]['get_idx'](model.layers[layer_idx])
        return layer_idx, idces

    def getParamLine(self, neighborhood, model):
        layer_idx, idces = neighborhood
        self.set_neighborhood_info(idces, model.layers[layer_idx])
        return model.layers[layer_idx].getParamLine(idces)

    def update(self, model, neighborhood, proposal):
        layer_idx, idces = neighborhood
        model.layers[layer_idx].update(idces, proposal)

    def undo(self, model, neighborhood, proposal):
        layer_idx, idces = neighborhood
        model.layers[layer_idx].undo(idces, proposal)

    def get_proposal_as_string(self, neighborhood):
        layer_idx, idces = neighborhood
        return 'layer_'+str(layer_idx)

class MixedSelector(UniformSelector):
    """
    select one layer and pick some weights with a uniform law over this layer.
    The neighborhood_info attribute return also the kind of sampler for binary or real weights, which should be used for the current neighborhood
    This Selector should be used with the MixedSampler.
    """
    def set_neighborhood_info(self, idces, layer):
        self.neighborhood_info = layer.is_binary, layer.get_selected_size(idces)


class OneHiddenSelector(Selector):
    """
    Selector which was used in the first experiment
    It has the particularity that the two layers are sampled within the same mcmc iteration

    It still there mostly for archive purposes and will be probably removed soon
    """
    def __init__(self, layer_sizes, config):
        self.sizes = layer_sizes

    def get_neighborhood(self):
        n_input, n_hidden, n_output = self.sizes
        idx_hidden = torch.randint(0, n_hidden, (1,))
        idx_output = -1
        if torch.randint(0,int(n_hidden/n_output), (1,)) == 0: # the bias of the second layer will be selected
            idx_output = torch.randint(0,n_output, (1,))
        idces_w1 = torch.cat((torch.ones(n_input,1)*idx_hidden, torch.arange(0,n_input).reshape(n_input, 1) ),dim=1).long()
        idces_b1 = idx_hidden
        idces_w2 = torch.cat((torch.arange(0,n_output).reshape(n_output, 1), torch.ones(n_output,1)*idx_hidden ), dim=1).long()
        if idx_output == -1:
            idces_b2 = torch.Tensor([])
        else:
            idces_b2 = torch.Tensor([idx_output]).long()
        self.set_neighborhood_size((idces_w1, idces_b1, idces_w2, idces_b2))
        return idces_w1, idces_b1, idces_w2, idces_b2

    def getParamLine(self, neighborhood, model):
        idces_w1, idces_b1, idces_w2, idces_b2 = neighborhood
        w1 = model.layers[0].weight.data[idces_w1[:,0],idces_w1[:,1]]
        b1 = model.layers[0].bias.data[idces_b1]
        w2 = model.layers[1].weight.data[idces_w2[:,0],idces_w2[:,1]]
        if idces_b2.shape[0] == 0:
            return torch.cat((w1, b1, w2))
        else:
            b2 = model.layers[0].bias.data[idces_b2]
            return torch.cat((w1, b1, w2, b2))

    def set_neighborhood_info(self, neighborhood):
        idces_w1, idces_b1, idces_w2, idces_b2 = neighborhood
        self.neighborhood_info = idces_w1.shape[0] + 1 + idces_w2.shape[0] + idces_b2.shape[0]

    def update(self, model, neighborhood, proposal):
        idces_w1, idces_b1, idces_w2, idces_b2 = neighborhood
        model.layers[0].update((idces_w1, idces_b1), proposal[:idces_w1.shape[0]+1])
        model.layers[1].update((idces_w2, idces_b2), proposal[idces_w1.shape[0]+1:])

    def undo(self, model, neighborhood, proposal):
        idces_w1, idces_b1, idces_w2, idces_b2 = neighborhood
        model.layers[0].undo((idces_w1, idces_b1), proposal[:idces_w1.shape[0]+1])
        model.layers[1].undo((idces_w2, idces_b2), proposal[idces_w1.shape[0]+1:])
