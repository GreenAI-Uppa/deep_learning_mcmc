from abc import ABC, abstractmethod
import torch
import sys
current_module = sys.modules[__name__]
from typing import Tuple

def build_selector(config, layer_sizes):
    selector_name = config["name"]
    selector_class = getattr(current_module, selector_name)
    if issubclass(selector_class, RandomSelector):
        neighborhood_size = config["neighborhood_size"]
        return selector_class(layer_sizes, neighborhood_size)
    else:
        return selector_class(layer_sizes)

class Selector(ABC):
    @abstractmethod
    def __init__(model):
        pass

    @property
    @abstractmethod
    def neighborhood_size(self):
        pass

    @abstractmethod
    def get_neighborhood():
        pass

    @abstractmethod
    def getParamLine(neighborhood, model):
        pass

    @abstractmethod
    def update(model, neighborhood, proposal):
        pass

    def get_proposal_as_string(self, neighborhood):
        return "accepted"


class RandomSelector(Selector):
    def __init__(self, neighborhood_size):
        self.neighborhood_size = neighborhood_size

class LinearSelector(Selector):
    neighborhood_size = None
    def __init__(self, sizes):
        self.neighborhood_size, self.n_rows = sizes
        self.neighborhood_size += 1

    def get_neighborhood(self):
        idx_row = torch.randint(0, self.n_rows, (1,))
        return idx_row

    def getParamLine(self, neighborhood, model):
        params_line = torch.cat((model.linears[0].weight.data[neighborhood][0],model.linears[0].bias.data[neighborhood]))
        return params_line

    def update(self, model, neighborhood, proposal):
        model.linears[0].weight.data[neighborhood] += proposal[:-1]
        model.linears[0].bias.data[neighborhood] += proposal[-1]

    def undo(self, model, neighborhood, proposal):
        model.linears[0].weight.data[neighborhood] -= proposal[:-1]
        model.linears[0].bias.data[neighborhood] -= proposal[-1]

class BinLinSelector(LinearSelector):
    def update(self, model, neighborhood, proposal):
        model.linears[0].weight.data[neighborhood] *= -1
        model.linears[0].bias.data[neighborhood] *= -1

    def undo(self, model, neighborhood, proposal):
        model.linears[0].weight.data[neighborhood] *= -1
        model.linears[0].bias.data[neighborhood] *= -1

class BinSelector(RandomSelector):
    neighborhood_size = None
    def __init__(self, sizes, neighborhood_size):
        super(BinSelector, self).__init__(neighborhood_size)
        self.n_inputs, self.n_outputs = sizes

    def get_neighborhood(self):
        idces_w = torch.cat((torch.randint(0,self.n_outputs,(self.neighborhood_size,1)), torch.randint(0,self.n_inputs+1,(self.neighborhood_size,1))), dim=1)
        idces_b = idces_w[idces_w[:,1]==self.n_inputs][:,0]
        idces_w = idces_w[idces_w[:,1]<self.n_inputs]
        return idces_w, idces_b

    def getParamLine(self, neighborhood, model):
        return None

    def update(self, model, neighborhood, proposal):
        idces_w, idces_b = neighborhood
        model.linears[0].weight.data[idces_w[:,0],idces_w[:,1]] *= -1
        model.linears[0].bias.data[idces_b] *= -1

    def undo(self, model, neighborhood, proposal):
        self.update(model, neighborhood, proposal)

class Bin1HSelector1(RandomSelector):
    neighborhood_size = None
    def __init__(self, sizes: Tuple[int, ...], neighborhood_size: int):
        """
        neighborhood_size : percentage over the number of parameters between 0 and 1.

        """
        self.neighborhood_size = neighborhood_size
        self.n_inputs, self.n_hidden, self.n_outputs = sizes

    def get_neighborhood(self):
        if torch.rand(1) > 0.1:
            layer_idx = 0
            idces_w = torch.cat((torch.randint(0,self.n_hidden,(self.neighborhood_size,1)), torch.randint(0,self.n_inputs+1,(self.neighborhood_size,1))), dim=1)
            idces_b = idces_w[idces_w[:,1]==self.n_inputs][:,1]
            idces_w = idces_w[idces_w[:,1]<self.n_inputs]
        else:
            layer_idx = 1
            idces_w = torch.cat((torch.randint(0,self.n_outputs,(self.neighborhood_size,1)), torch.randint(0,self.n_hidden+1,(self.neighborhood_size,1))), dim=1)
            idces_b = idces_w[idces_w[:,1]==self.n_hidden][:,1]
            idces_w = idces_w[idces_w[:,1]<self.n_hidden]
        return layer_idx, idces_w, idces_b

    def getParamLine(self, neighborhood, model):
        return None

    def get_proposal_as_string(self, neighborhood):
        layer_idx, idces_w, idces_b = neighborhood
        return 'layer_'+str(layer_idx)

    def update(self, model, neighborhood, proposal):
        layer_idx, idces_w, idces_b = neighborhood
        model.linears[layer_idx].weight.data[idces_w[:,0],idces_w[:,1]] *= -1
        model.linears[layer_idx].bias.data[idces_b] *= -1

    def undo(self, model, neighborhood, proposal):
        self.update(model, neighborhood, proposal)


class Bin1HSelector(RandomSelector):
    neighborhood_size = None
    def __init__(self, sizes: Tuple[int, ...], neighborhood_size: int):
        """
        neighborhood_size : percentage over the number of parameters between 0 and 1.

        """
        self.n_inputs, self.n_hidden, self.n_outputs = sizes
        self.neighborhood_size_w1 = int(self.neighborhood_size * (self.n_inputs+1) * self.n_hidden)
        self.neighborhood_size_w2 = int(self.neighborhood_size * (self.n_hidden+1) * self.n_outputs)

    def get_neighborhood(self):
        idces_w = torch.cat((torch.randint(0,self.n_inputs+1,(self.neighborhood_size_w1,1)), torch.randint(0,self.n_hidden,(self.neighborhood_size_w1,1))), dim=1)
        idces_b1 = idces_w[idces_w[:,0]==self.n_inputs][:,1]
        idces_w1 = idces_w[idces_w[:,0]<self.n_inputs]
        idces_w = torch.cat((torch.randint(0,self.n_hidden+1,(self.neighborhood_size_w2,1)), torch.randint(0,self.n_outputs,(self.neighborhood_size_w2,1))), dim=1)
        idces_b2 = idces_w[idces_w[:,0]==self.n_hidden][:,1]
        idces_w2 = idces_w[idces_w[:,0]<self.n_hidden]
        return idces_w1, idces_b1, idces_w2, idces_b2

    def getParamLine(self, neighborhood, model):
        return None

    def update(self, model, neighborhood, proposal):
        idces_w1, idces_b1, idces_w2, idces_b2 = neighborhood
        model.linears[0].weight.data[idces_w1[:,1],idces_w1[:,0]] *= -1
        model.linears[0].bias.data[idces_b1] *= -1
        model.linears[1].weight.data[idces_w2[:,1],idces_w2[:,0]] *= -1
        model.linears[1].bias.data[idces_b2] *= -1

    def undo(self, model, neighborhood, proposal):
        self.update(model, neighborhood, proposal)

class OneHiddenSelector(Selector):
    neighborhood_size = None
    def __init__(self, layer_sizes):
        self.n_input, self.n_hidden, self.n_output = layer_sizes

    def get_neighborhood(self):
        idx_hidden = idx_hidden = torch.randint(0, self.n_hidden, (1,))
        idx_output = -1
        if torch.randint(0,int(self.n_hidden/self.n_output), (1,)) == 0: # the bias of the second layer will be selected
            idx_output = torch.randint(0,self.n_output, (1,))
        if idx_output == -1:
            self.neighborhood_size = self.n_input + 1 + self.n_output
        else:
            self.neighborhood_size = self.n_input + 1 + self.n_output + 1
        return idx_hidden, idx_output

    def getParamLine(self, neighborhood, model):
        """
        given the idx of a hidden neuron, it selects the corresponding weights associated with it
        """
        idx_hidden, idx_output = neighborhood
        if idx_output == -1:
            params_line = torch.cat((model.linears[0].weight.data[idx_hidden][0], model.linears[0].bias.data[idx_hidden], model.linears[1].weight.data[:,idx_hidden][:,0]))
        else:
            params_line = torch.cat((model.linears[0].weight.data[idx_hidden][0],model.linears[0].bias.data[idx_hidden], model.linears[1].weight.data[:,idx_hidden][:,0], model.linears[1].bias.data[idx_output]))
        return params_line

    def update(self, model, neighborhood, proposal):
        idx_hidden, idx_output = neighborhood
        model.linears[0].weight.data[idx_hidden] += proposal[:model.linears[0].weight.data.shape[1]]
        model.linears[0].bias.data[idx_hidden] += proposal[model.linears[0].weight.data.shape[1]]
        model.linears[1].weight.data[:,idx_hidden] += proposal[model.linears[0].weight.data.shape[1]+1:model.linears[0].weight.data.shape[1]+1+self.n_output].reshape(self.n_output,1)
        if idx_output != -1:
            model.linears[1].bias.data[idx_output] += proposal[-1]

    def undo(self, model, neighborhood, proposal):
        idx_hidden, idx_output = neighborhood
        model.linears[0].weight.data[idx_hidden] -= proposal[:model.linears[0].weight.data.shape[1]]
        model.linears[0].bias.data[idx_hidden] -= proposal[model.linears[0].weight.data.shape[1]]
        model.linears[1].weight.data[:,idx_hidden] -= proposal[model.linears[0].weight.data.shape[1]+1:model.linears[0].weight.data.shape[1]+1+self.n_output].reshape(self.n_output,1)
        if idx_output != -1:
            model.linears[1].bias.data[idx_output] -= proposal[-1]
