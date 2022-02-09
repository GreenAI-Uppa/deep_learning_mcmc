"""optimizers contains gradient and mcmc optimizers and the main loop content
to train the network"""
import collections
from abc import ABC, abstractmethod
import torch
import numpy as np
from deep_learning_mcmc import nets
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import copy
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import json, random
from deep_learning_mcmc import pruning

class Optimizer(ABC):
    """Generic optimizer interface"""
    def __init__(self, data_points_max = 1000000000):
        """
        number of data points to used in the dataset
        """
        self.data_points_max = data_points_max

    def train_1_epoch(self, dataloader, model, loss_fn):
        """train the data for 1 epoch"""
        num_items_read = 0
        # attempting to guess the device on the model.
        device = next(model.parameters()).device
        for _, (X, y) in enumerate(dataloader):
            if self.data_points_max <= num_items_read:
                break
            X = X[:min(self.data_points_max - num_items_read, X.shape[0])]
            y = y[:min(self.data_points_max - num_items_read, X.shape[0])]
            num_items_read = min(self.data_points_max, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            self.train_1_batch(X, y, model, loss_fn)

    @abstractmethod
    def train_1_batch(self, X, y, model, loss_fn=torch.nn.CrossEntropyLoss()):
        """abtract method to train a batch of data"""

class GradientOptimizer(Optimizer):
    """plain vanillia Stochastic Gradient optimizer, no adaptative learning rate"""
    def __init__(self, data_points_max = 1000000000, lr=0.001, pruning_level=0):
        super().__init__(data_points_max = 1000000000)
        self.lr = lr
        self.pruning_level = pruning_level
        #self.current_batch = 0


    def train_1_epoch(self, dataloader, model, loss_fn,path):

        """train the data for 1 epoch"""
        num_items_read = 0
        # attempting to guess the device on the model.
        device = next(model.parameters()).device
        if self.pruning_level>0:
            test_data = datasets.CIFAR10(root=path,
            train=False,
            download=True,
            transform=ToTensor())
            pruning_dataloader = DataLoader(test_data, batch_size=256, num_workers=8)
        liste = []
        for elt in dataloader:
            liste.append(elt)
        current_pruning_level = self.pruning_level
        res = {}
        Pruner = pruning.MozerPruner()
        for i in range(200000):
            """
            if i <= self.current_batch:
                continue
            if self.current_batch + 1 < i:
                break
            print("passing")
            """
            k = random.choice(range(len(dataloader)))
            (X,y) = liste[k]
            if self.data_points_max <= num_items_read:
                break
            X = X[:min(self.data_points_max - num_items_read, X.shape[0])]
            y = y[:min(self.data_points_max - num_items_read, X.shape[0])]
            num_items_read = min(self.data_points_max, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            self.train_1_batch(X, y, model, loss_fn)
            if i>0 and self.pruning_level>0 and i%200 == 0:#skeletonize any 2000 gradient step
                acc_before = nets.evaluate(pruning_dataloader,model,loss_fn)
                l0_before = torch.nonzero(model.fc1.weight.data).shape[0]+torch.nonzero(model.conv1.weight.data).shape[0]
                #print('iteration',i,':','pruning level',current_pruning_level,'| Performances before skeletonization',acc_before,'l0 norm',l0_before)
                Pruner.skeletonize_conv(model,current_pruning_level,dataloader)
                Pruner.skeletonize(model,current_pruning_level,dataloader)
                acc_after = nets.evaluate(pruning_dataloader,model,loss_fn)
                l0_after = torch.nonzero(model.fc1.weight.data).shape[0]+torch.nonzero(model.conv1.weight.data).shape[0]
                if i%2000 == 0:
                    res[i]={'pruning_level':current_pruning_level,'acc_before':acc_before,'acc_after':acc_after,'l0_before':l0_before,'l0_after':l0_after}
                    print('iteration',i,':','pruning level',current_pruning_level)
                    print('perf before skeletonization',acc_before,'l0 norm',l0_before,'perf after',acc_after,'l0 norm',l0_after)
                    current_pruning_level += 0.01

        with open('ICML/64_gradient_newfreq.json','w') as outputfile:
            json.dump(res,outputfile)
        """
        if len(dataloader) - 1 <= i:
            i = 0
        self.current_batch = i
        print("current batch:", self.current_batch, i, len(dataloader))
        """

    def train_1_batch(self, X, y, model, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        SGD optimization
        """
        device = next(model.parameters()).device
        pred = model(X)
        los = loss_fn(pred, y)
        gg = torch.autograd.grad(los, model.parameters(), retain_graph=True)
        for i, layer in enumerate(model.layers): #[model.conv1, model.fc1]):
            layer.weight.data -=  gg[2*i] * self.lr
            layer.bias.data -=  gg[2*i+1] * self.lr

class BinaryConnectOptimizer(Optimizer):
    """plain vanillia Stochastic Gradient optimizer, no adaptative learning rate"""
    def __init__(self, data_points_max = 1000000000, lr=0.001, pruning_level=0):
        super().__init__(data_points_max = 1000000000)
        self.lr = lr
        self.pruning_level = pruning_level
        #self.current_batch = 0

    def train_1_epoch(self, dataloader, model, loss_fn):
        """train the data for 1 epoch"""
        num_items_read = 0
        # attempting to guess the device on the model.
        device = next(model.parameters()).device
        for i, (X, y) in enumerate(dataloader):
            """
            if i <= self.current_batch:
                continue
            if self.current_batch + 1 < i:
                break
            print("passing")
            """
            if self.data_points_max <= num_items_read:
                break
            X = X[:min(self.data_points_max - num_items_read, X.shape[0])]
            y = y[:min(self.data_points_max - num_items_read, X.shape[0])]
            num_items_read = min(self.data_points_max, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            self.train_1_batch(X, y, model, loss_fn)
        """
        if len(dataloader) - 1 <= i:
            i = 0
        self.current_batch = i
        print("current batch:", self.current_batch, i, len(dataloader))
        """

    def train_1_batch(self, X, y, model, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        SGD optimization
        """
        device = next(model.parameters()).device
        pred = model(X)
        los = loss_fn(pred, y)
        # getting the gradient for all the layers
        gg = torch.autograd.grad(los, model.parameters(), retain_graph=True)
        for i, real_layer in enumerate(model.layers_reals): #[model.conv1, model.fc1]):
            if real_layer is None:# no copy, so this layer is not binary
                model.layers[i].weight.data -=  gg[2*i] * self.lr
                model.layers[i].bias.data -=  gg[2*i+1] * self.lr
            else:
                # update the real copies of the weights
                real_layer[0] -=  gg[2*i] * self.lr
                real_layer[1] -=  gg[2*i+1] * self.lr
                # update the binary version
                model.layers[i].weight.data = torch.sign(real_layer[0])
                model.layers[i].bias.data = torch.sign(real_layer[1])
                # clip the real value of the weights
                torch.clip(real_layer[0],-1,1, out=real_layer[0])
                torch.clip(real_layer[1],-1,1, out=real_layer[1])



class MCMCOptimizer(Optimizer):
    def __init__(self, sampler, data_points_max = 1000000000, iter_mcmc=1, lamb=1000,  prior=None, selector=None, pruning_level=0):
        """
        variance_prop : zero centered univariate student law class to generate the proposals
        variance_prior : zero centered univariate student law class used as a prior on the parameter values
        lamb : ponderation between the data and the prior
        iter_mcmc : number of mcmc iterations
        """
        super().__init__(data_points_max = 1000000000)
        self.iter_mcmc = iter_mcmc
        self.lamb = lamb
        self.sampler = sampler
        self.pruning_level = pruning_level
        if prior is None:
            self.prior = self.sampler
        else:
            self.prior = prior
        self.selector = selector

    def train_1_epoch(self, dataloader, model, loss_fn, path, verbose=False):
        """
        train for 1 epoch and collect the acceptance ratio
        """
        num_items_read = 0
        acceptance_ratio = Acceptance_ratio()
        device = next(model.parameters()).device
        for _, (X, y) in enumerate(dataloader):
            if self.data_points_max <= num_items_read:
                break
            X = X[:min(self.data_points_max - num_items_read, X.shape[0])]
            y = y[:min(self.data_points_max - num_items_read, X.shape[0])]
            num_items_read = min(self.data_points_max, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            acceptance_ratio += self.train_1_batch(X, y, model, dataloader, loss_fn=torch.nn.CrossEntropyLoss(), path=path, verbose=verbose)
        return acceptance_ratio

    def train_1_batch(self, X, y, model, dataloader, loss_fn, path, verbose=False):
        """
        perform mcmc iterations with a neighborhood corresponding to one line of the parameters.

        the acceptance of the proposal depends on the following criterion
           exp(lamb * (loss_previous - loss_prop) ) * stud(params_prop) / stud(params_previous)

        inputs:
        X : input data
        y : input labels
        model : neural net we want to optimize
        loss_fn : loss function

        outputs:
        acceptance_ratio
        model : optimised model (modified by reference)
        """
        device = next(model.parameters()).device
        ar = Acceptance_ratio()
        pred = model(X)
        loss = loss_fn(pred,y).item()
        if self.pruning_level>0:
            Pruner = pruning.MCMCPruner()
            relevance_dict = {}
            for cle in range(model.conv1.weight.data.shape[0]):
                relevance_dict[cle] = 0
            relevance_dict_linear_layer_w = torch.zeros(model.fc1.weight.data.shape)
            relevance_dict_linear_layer_b = torch.zeros(model.fc1.bias.data.shape[0],1)
            relevance_dict_linear_layer = {'weight':relevance_dict_linear_layer_w, 'bias':relevance_dict_linear_layer_b}

            test_data = datasets.CIFAR10(root=path,
            train=False,
            download=True,
            transform=ToTensor())
            test_dataloader = DataLoader(test_data, batch_size=256, num_workers=16)
            loss_fn = torch.nn.CrossEntropyLoss()
        current_pruning_level = self.pruning_level
        res = {}
        for i in range(self.iter_mcmc):
            #if i>100 and i%200 == 0 and self.pruning_level>0:
            #    print('iteration',i,sorted([(cle,relevance_dict[cle]) for cle in range(model.conv1.weight.data.shape[0]) if relevance_dict[cle]>0],key=lambda tup: tup[1],reverse=True)[:15])
            if i>0 and self.pruning_level>0 and i%2000 == 0:#skeletonize iteration
                acc_before = nets.evaluate(test_dataloader,model,loss_fn)
                l0_before = torch.nonzero(model.conv1.weight.data).shape[0]+torch.nonzero(model.fc1.weight.data).shape[0]
                #print('iteration',i,':','pruning level',current_pruning_level,'| Performances before skeletonization',acc_before,'l0 norm',l0_before)
                Pruner.skeletonize_conv(model,current_pruning_level,relevance_dict)
                Pruner.skeletonize(model,current_pruning_level,relevance_dict_linear_layer)
                acc_after = nets.evaluate(test_dataloader,model,loss_fn)
                l0_after = torch.nonzero(model.conv1.weight.data).shape[0]+torch.nonzero(model.fc1.weight.data).shape[0]
                loss = loss_fn(model(X),y)#update loss as new init to metropolis hasting
                print(i,end='|')
                #print('Performances after skeletonization',acc_after,'l0 norm',l0_after)
                if i%2000 == 0:
                    res[i]={'pruning_level':current_pruning_level,'acc_before':acc_before,'acc_after':acc_after,'l0_before':l0_before,'l0_after':l0_after}
                    print(i,'pruning level',current_pruning_level,'---')
                    print('perf before skeletonization',acc_before,'l0 norm',l0_before,'perf after',acc_after,'l0 norm',l0_after)
                    current_pruning_level += 0.01
                #print('update pruning level to',current_pruning_level)
            # selecting a layer and weights at random
            layer_idx, idces = self.selector.get_neighborhood(model)
            neighborhood = layer_idx, idces
            params_line = self.selector.getParamLine(neighborhood, model)
            epsilon = self.sampler.sample(self.selector.neighborhood_info)
            if epsilon is not None:
                epsilon = torch.tensor(epsilon.astype('float32')).to(device)
            # getting the ratio of the students
            student_ratio = self.prior.get_ratio(epsilon, params_line, self.selector.neighborhood_info)

            # applying the changes to get the new value of the loss
            self.selector.update(model, neighborhood, epsilon)
            pred = model(X)
            loss_prop = loss_fn(pred, y)
            # computing the change in the loss
            lamb = self.sampler.get_lambda(self.selector.neighborhood_info)
            data_term = torch.exp(lamb * (loss -loss_prop))

            rho  = min(1, data_term * student_ratio)
            if verbose:
                print(i,'moove layer',layer_idx,'rho=',float(rho),'data term=',float(data_term),'ratio=',float(student_ratio),'| ','loss_prop',float(loss_prop),'loss gain',float(loss-loss_prop))
            key = self.selector.get_proposal_as_string(neighborhood)
            ar.incr_prop_count(key) # recording so that we can later compute the acceptance ratio
            if rho > torch.rand(1).to(device):
                # accepting, keeping the new value of the loss
                ar.incr_acc_count(key)
                loss = loss_prop
                decision = 'accepted'
                if layer_idx == 0 and self.pruning_level>0:
                    relevance_dict[int(idces[0][0][0])]+=1
                if layer_idx == 1 and self.pruning_level>0:
                    relevance_dict_linear_layer['weight'][idces[0][:,0],idces[0][:,1]] +=1
                    relevance_dict_linear_layer['bias'][idces[1]] +=1
            else:
                # not accepting, so undoing the change
                self.selector.undo(model, neighborhood, epsilon)
                decision = 'rejected'
                if layer_idx == 0 and self.pruning_level>0:
                    relevance_dict[int(idces[0][0][0])]-=1
                if layer_idx == 1 and self.pruning_level>0:
                    relevance_dict_linear_layer['weight'][idces[0][:,0],idces[0][:,1]] -=1
                    relevance_dict_linear_layer['bias'][idces[1]] -=1
            if verbose:
                print('moove',decision)
        with open('ICML/64_mcmc_rejectintheloop.json', 'w') as outfile:
            json.dump(res, outfile)
        return ar


class Acceptance_ratio():
    """
    wrapper around a counter to maintain multiple aspect ratios
    """
    def __init__(self):
        self.proposal_accepted = collections.Counter()
        self.proposal_count = collections.Counter()

    def incr_prop_count(self, key):
        self.proposal_count[key] += 1

    def incr_acc_count(self, key):
        self.proposal_accepted[key] += 1

    def to_dict(self):
        return dict([ (k,str(self.proposal_accepted[k]/v)) for (k, v) in self.proposal_count.items()])

    def __add__(self, acceptance_ratio):
        """
        redefining the + operator
        """
        result = Acceptance_ratio()
        for k, v in self.proposal_accepted.items():
            result.proposal_accepted[k] = v
        for k, v in acceptance_ratio.proposal_accepted.items():
            if k not in result.proposal_accepted:
                result.proposal_accepted[k] = 0
            result.proposal_accepted[k] += v
        for k, v in self.proposal_count.items():
            result.proposal_count[k] = v
        for k, v in acceptance_ratio.proposal_count.items():
            if k not in result.proposal_count:
                result.proposal_count[k] = 0
            result.proposal_count[k] += v
        return result

    def __str__(self):
        """
        redefining the output of the printing
        """
        result = ["acceptance ratios : "]
        for k, v in self.proposal_count.items():
            result.append(k + ": " + str(self.proposal_accepted[k]/v))
        return '\n'.join(result)
