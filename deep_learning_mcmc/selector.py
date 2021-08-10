from abc import ABC, abstractmethod
import torch
import sys
current_module = sys.modules[__name__]
from typing import Tuple

def build_selector(layer_sizes,config):
    selector_name = config["name"]
    selector_class = getattr(current_module, selector_name)
    return selector_class(layer_sizes, config)

class Selector(ABC):
    @abstractmethod
    def __init__(sizes, config):
        self.sizes = sizes

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
        return "layer_x"

class MixedSelector(Selector):
    def __init__(self, layer_sizes, config):
        self.sizes = layer_sizes
        self.set_neighborhood_size(config)
        number_of_layers = len(self.sizes) - 1
        if 'layer_distr' not in config:
            self.layer_distr = [1/number_of_layers for i in range(number_of_layers)]
        else:
            self.layer_distr = config['layer_distr']
        assert len(self.layer_distr) == number_of_layers
        self.layer_distr = torch.cumsum(torch.Tensor(self.layer_distr),0)

    def set_neighborhood_size(self, config):
        self.neighborhood_size = config['neighborhood_size']

    def get_layer_idx(self):
        seed = torch.rand(1)
        layer_idx = 0
        for i, x in enumerate(self.layer_distr):
            if x > seed:
                break
            layer_idx += 1
        return layer_idx

    def get_neighborhood(self):
        layer_idx = self.get_layer_idx()
        idces_w = torch.cat((torch.randint(0,self.sizes[layer_idx+1],(self.neighborhood_size,1)), torch.randint(0,self.sizes[layer_idx]+1,(self.neighborhood_size,1))), dim=1)
        idces_b = idces_w[idces_w[:,1]==self.sizes[layer_idx]][:,0]
        idces_w = idces_w[idces_w[:,1]<self.sizes[layer_idx]]
        # add some assertion to check that the proposal falls within the range of the layer indices.
        return layer_idx, idces_w, idces_b

    def getParamLine(self, neighborhood, model):
        layer_idx, idces_w, idces_b = neighborhood
        if model.linears[layer_idx].is_binary:
            return None
        else:
            w = model.linears[layer_idx].weight.data[idces_w[:,0],idces_w[:,1]]
            b = model.linears[layer_idx].bias.data[idces_b]
            return torch.cat((w,b))

    def update(self, model, neighborhood, proposal):
        layer_idx, idces_w, idces_b = neighborhood
        model.linears[layer_idx].update((idces_w, idces_b), proposal)

    def undo(self, model, neighborhood, proposal):
        layer_idx, idces_w, idces_b = neighborhood
        model.linears[layer_idx].undo((idces_w, idces_b), proposal)

    def get_proposal_as_string(self, neighborhood):
        layer_idx, idces_w, idces_b = neighborhood
        return 'layer_'+str(layer_idx)

class LinearSelector(MixedSelector):
    neighborhood_size = None
    def set_neighborhood_size(self, config):
        self.neighborhood_size = self.sizes[0]+1

    def get_neighborhood(self):
        layer_idx = self.get_layer_idx()
        idx_row = torch.randint(0, self.sizes[layer_idx+1], (1,))
        idces_w = torch.cat((torch.ones(self.sizes[layer_idx],1)*idx_row, torch.arange(0,self.sizes[layer_idx]).reshape(self.sizes[layer_idx], 1) ),dim=1).long()
        idces_b = idx_row
        return layer_idx, idces_w, idces_b

class OneHiddenSelector(Selector):
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
        w1 = model.linears[0].weight.data[idces_w1[:,0],idces_w1[:,1]]
        b1 = model.linears[0].bias.data[idces_b1]
        w2 = model.linears[1].weight.data[idces_w2[:,0],idces_w2[:,1]]
        if idces_b2.shape[0] == 0:
            return torch.cat((w1, b1, w2))
        else:
            b2 = model.linears[0].bias.data[idces_b2]
            return torch.cat((w1, b1, w2, b2))

    def set_neighborhood_size(self, neighborhood):
        idces_w1, idces_b1, idces_w2, idces_b2 = neighborhood
        self.neighborhood_size = idces_w1.shape[0] + 1 + idces_w2.shape[0] + idces_b2.shape[0]

    def update(self, model, neighborhood, proposal):
        idces_w1, idces_b1, idces_w2, idces_b2 = neighborhood
        model.linears[0].update((idces_w1, idces_b1), proposal[:idces_w1.shape[0]+1])
        model.linears[1].update((idces_w2, idces_b2), proposal[idces_w1.shape[0]+1:])

    def undo(self, model, neighborhood, proposal):
        idces_w1, idces_b1, idces_w2, idces_b2 = neighborhood
        model.linears[0].undo((idces_w1, idces_b1), proposal[:idces_w1.shape[0]+1])
        model.linears[1].undo((idces_w2, idces_b2), proposal[idces_w1.shape[0]+1:])


class BinLinSelector_odd(LinearSelector):
    def update(self, model, neighborhood, proposal):
        model.linears[0].weight.data[neighborhood] *= -1
        model.linears[0].bias.data[neighborhood] *= -1

    def undo(self, model, neighborhood, proposal):
        model.linears[0].weight.data[neighborhood] *= -1
        model.linears[0].bias.data[neighborhood] *= -1

class BinSelector_old(Selector):
    neighborhood_size = None
    def __init__(self, sizes, config):
        super(BinSelector, self).__init__(config)
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



class Bin1HSelector1_old(Selector):
    neighborhood_size = None
    def __init__(self, sizes: Tuple[int, ...], config: dict):
        """
        neighborhood_size : percentage over the number of parameters between 0 and 1.

        """
        super(BinSelector, self).__init__(config)
        self.n_inputs, self.n_hidden, self.n_outputs = sizes

    def get_neighborhood(self):
        if torch.rand(1) > 0.1:
            layer_idx = 0
            idces_w = torch.cat((torch.randint(0,self.n_hidden,(self.neighborhood_size,1)), torch.randint(0,self.n_inputs+1,(self.neighborhood_size,1))), dim=1)
            idces_b = idces_w[idces_w[:,1]==self.n_inputs][:,0]
            idces_w = idces_w[idces_w[:,1]<self.n_inputs]
        else:
            layer_idx = 1
            idces_w = torch.cat((torch.randint(0,self.n_outputs,(self.neighborhood_size,1)), torch.randint(0,self.n_hidden+1,(self.neighborhood_size,1))), dim=1)
            idces_b = idces_w[idces_w[:,1]==self.n_hidden][:,0]
            idces_w = idces_w[idces_w[:,1]<self.n_hidden]
        return layer_idx, idces_w, idces_b

    def getParamLine(self, neighborhood, model):
        return None

    def get_proposal_as_string(self, neighborhood):
        layer_idx, idces_w, idces_b = neighborhood
        return 'layer_'+str(layer_idx)

    def update(self, model, neighborhood, proposal):
        layer_idx, idces_w, idces_b = neighborhood
        if model.binary_flags[layer_idx]:
            model.linears[layer_idx].weight.data[idces_w[:,0],idces_w[:,1]] *= -1
            model.linears[layer_idx].bias.data[idces_b] *= -1
        else:
            model.linears[layer_idx].weight.data[idces_w[:,0],idces_w[:,1]] += proposal[:len(idces_w)]
            model.linears[layer_idx].bias.data[idces_b] += proposal[len(idces_w):]

    def undo(self, model, neighborhood, proposal):
        layer_idx, idces_w, idces_b = neighborhood
        if model.binary_flags[layer_idx]:
            model.linears[layer_idx].weight.data[idces_w[:,0],idces_w[:,1]] *= -1
            model.linears[layer_idx].bias.data[idces_b] *= -1
        else:
            model.linears[layer_idx].weight.data[idces_w[:,0],idces_w[:,1]] -= proposal[:len(idces_w)]
            model.linears[layer_idx].bias.data[idces_b] -= proposal[len(idces_w):]


class Bin1HSelector_old(Selector):
    neighborhood_size = None
    def __init__(self, sizes: Tuple[int, ...], config: dict):
        """
        neighborhood_size : percentage over the number of parameters between 0 and 1.

        """
        super(BinSelector, self).__init__(config)
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

class OneHiddenSelector_old(Selector):
    neighborhood_size = None
    def __init__(self, layer_sizes, config):
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
