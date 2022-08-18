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
import time
import deep_learning_mcmc.evaluator as evaluator

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


    def train_1_epoch(self, dataloader, model, loss_fn):

        """train the data for 1 epoch"""
        num_items_read = 0
        # attempting to guess the device on the model.
        device = next(model.parameters()).device
        if self.pruning_level>0:
            test_data = datasets.CIFAR10(root='../data',
            train=False,
            download=True,
            transform=ToTensor())
            pruning_dataloader = DataLoader(test_data, batch_size=64, num_workers=8)
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
            if i>0 and self.pruning_level>0 and i%200 == 0:#skeletonize any 50 gradient step
                skeletonization(model,self.pruning_level,pruning_dataloader)
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
        for i, layer in enumerate(model.layers):
            layer.weight.data -=  gg[2*i] * self.lr
            layer.bias.data -=  gg[2*i+1] * self.lr

class LayerWiseOptimizer(GradientOptimizer):
    """Layer wise training with gradient optimizer"""
    def __init__(self, buffer_max_size=10, data_points_max = 1000000000, lr=0.001, pruning_proba=0):
        super().__init__(data_points_max = data_points_max, lr=lr, pruning_proba=pruning_proba)
        self.buffer_max_size = buffer_max_size

    def scheduler(self, L, n):
      """
      L : number of model blocks (or number of machines)
      n : number of batches
      dummy scheduler : [-1,0,1,2,3,....L] * Nbatches
      -1 means that a new data is fed to the 0 machine
      so this won't be actually asynchronized
      """
      for n in range(n):
          for l in range(-1, L):
              yield l

    def train_1_epoch(self, dataloader, model, loss_fn):
        """train the data for 1 epoch"""
        buffers = dict([ (i,{ 'data': [], 'counters': [] }) for i in range(len(model))])
        iterDataloader = iter(dataloader)
        X, y = next(iterDataloader)
        #buffers[1]['data'].append((X,y))
        device = next(model[0].parameters()).device
        schedule = self.scheduler(len(model), len(dataloader)-1)
        r=0
        print("dataloader", len(dataloader))
        for i, l in enumerate(schedule):
            # print('iter',i,'layer',l)
            if i%100==0:
                print(i,'iterations')
            if l == -1:
                X, y = next(iterDataloader)
            else:
                X, y = self.read_buffer(buffers[l])
                X = X.to(device)
                y = y.to(device)
                X = self.train_1_batch(X, y, model[l], loss_fn)
            if l<len(model)-1:
                # send the output X of the previous layer to the next machine
                self.write_buffer(buffers[l+1], (X,y))
            #import pdb; pdb.set_trace()

    def train_1_batch(self, X, y, model, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        SGD optimization, and returns the output of the conv layer
        """
        device = next(model.parameters()).device
        X, pred = model(X)
        los = loss_fn(pred, y)
        gg = torch.autograd.grad(los, model.parameters(), retain_graph=True)
        for i, layer in enumerate(model.layers):
            layer.weight.data -=  gg[2*i] * self.lr
            layer.bias.data -=  gg[2*i+1] * self.lr
        return X

    def write_buffer(self, b, x):
        """
        b buffer : a list of batches and counters
        x : batch of data
        write data to the buffer
        Just append the data if the buffer is not full
        otherwise, replace the most used point
        if there are several of them, remove with a 'first in first out' scheme
        """
        if len(b['data']) < self.buffer_max_size:
            b['data'].append(x)
            b['counters'].append(0)
        else:
            # select the batch which have been used the most (max counter)
            # and among them the one which has been added the earliest to the
            # buffer (min index)
            cmax = max(b['counters'])
            idx = min([ i  for (i,c) in enumerate(b['counters']) if c==cmax ])
            # replace the selected old batch by the new batch
            b['counters'][idx] = 0
            b['data'][idx] = x
            #print('replacing',idx)

    def read_buffer(self, b):
        """
        get the least used minibatch
        if there are several of them, take the oldest one in the buffer,
        ie the one with the mininum index,
        to be coherent with the 'first in first out' writing
        """
        cmin = min(b['counters'])
        idx = min([ i  for (i,c) in enumerate(b['counters']) if c==cmin ])
        # recording that this batch has been used
        b['counters'][idx] += 1
        #print('reading ',idx)
        return b['data'][idx]

class LayerWiseAsyncOptimizer(GradientOptimizer):
    """Layer wise training with gradient optimizer"""
    def __init__(self, buffer_max_size=30, data_points_max = 1000000000, lr=0.001, pruning_proba=0):
        super().__init__(data_points_max = data_points_max, lr=lr, pruning_proba=pruning_proba)
        self.buffer_max_size = buffer_max_size

    def train_async(self, trainDataloader,testDataloader, model, loss_fn, nepochs):
        print("dataloader", len(trainDataloader))

        start_all = time.time()
        nepochs = nepochs-1
        n_cnn = len(model)
        layer_noise = 3
        noise = 0.1
        #Declarations
        to_train = True
        first_iter = True
        counter_first_iter = 0
        iteration_tracker = [0]*n_cnn
        epoch_tracker = [-1]*n_cnn
        epoch_finished = [False for _ in range(n_cnn)]
        n_layer = 0
        continue_to_train = [True] * n_cnn
        

        device = next(model[0].parameters()).device
        iterDataloader = iter(trainDataloader)
        num_batch = len(trainDataloader)
        print(num_batch)
        # scheduler pour selectionner le worker
        proba = torch.ones(n_cnn).float()
        proba[layer_noise]=proba[layer_noise]-noise
        proba = proba*1.0/proba.sum()
        schedule = torch.distributions.categorical.Categorical(probs=proba)

        #init buffers
        buffers = dict([ (i,{ 'data': [], 'counters': [] }) for i in range(len(model))])
        print(nepochs)
        while to_train: #start training
            if first_iter:
                n_layer = counter_first_iter//2
                if(counter_first_iter>2*(n_cnn-1)):
                    first_iter = False
                counter_first_iter += 1
            else:
                n_layer = schedule.sample().item()

            if epoch_finished[n_layer]:
                epoch_tracker[n_layer] = epoch_tracker[n_layer] + 1
                print("Epoch Tracker: ")
                print(*epoch_tracker)
                if n_layer==len(model)-1:
                    print(f"Training Epoch {epoch_tracker[n_layer]}\n----- = "+time.strftime("%H:%M:%S",time.gmtime(time.time() - start_all)) +"----------")
                    loss, accuracy = evaluator.evaluateLayerWise(trainDataloader, model, loss_fn)
                    for l in loss:
                        acc = accuracy[l]
                        los = loss[l]
                        print(f"Training Error: Layer {n_layer} \n Accuracy: {(100*acc):>0.1f}%, Avg loss: {los:>8f} \n")

            if all(e==False for e in continue_to_train):
                to_train = False
            
            if epoch_tracker[n_layer]>=nepochs:
                continue_to_train[n_layer] = False
                if n_layer!=layer_noise:
                    proba[n_layer] = 0
                    proba = proba*1.0/proba.sum()
                    schedule = torch.distributions.categorical.Categorical(probs=proba)
            epoch_finished[n_layer]= False

            if n_layer == 0:
                try:
                    X,y = next(iterDataloader)
                    X = X.to(device)
                    y = y.to(device)
                except StopIteration:
                    iterDataloader = iter(trainDataloader)
                    X,y = next(iterDataloader)
                    X = X.to(device)
                    y = y.to(device)
            else:
                sample = self.read_buffer(buffers[n_layer-1])
                if sample is not None:
                    X, y = sample
                    X = X.to(device)
                    y = y.to(device)
                else:
                    print(f"Experiencing buffer overuse in layer {n_layer}, not processing")
                    X = None
            iteration_tracker[n_layer] = (iteration_tracker[n_layer] + 1) % num_batch
            if iteration_tracker[n_layer] == 0:
                epoch_finished[n_layer]=True
            if X is not None:
                if continue_to_train[n_layer]:
                    X = self.train_1_batch(X, y, model[n_layer], loss_fn)
                    if (n_layer<n_cnn-1):
                        self.write_buffer(buffers[n_layer], (X,y))
                                            
    def train_1_batch(self, X, y, model, loss_fn=torch.nn.CrossEntropyLoss()):
        """
        SGD optimization, and returns the output of the conv layer
        """
        X, pred = model(X)
        los = loss_fn(pred, y)
        gg = torch.autograd.grad(los, model.parameters(), retain_graph=True)
        for i, layer in enumerate(model.layers):
            layer.weight.data -=  gg[2*i] * self.lr
            layer.bias.data -=  gg[2*i+1] * self.lr
        return X

    def write_buffer(self, b, x):
        """
        b buffer : a list of batches and counters
        x : batch of data
        write data to the buffer
        Just append the data if the buffer is not full
        otherwise, replace the most used point
        if there are several of them, remove with a 'first in first out' scheme
        """
        if len(b['data']) < self.buffer_max_size:
            b['data'].append(x)
            b['counters'].append(0)
        else:
            # select the batch which have been used the most (max counter)
            # and among them the one which has been added the earliest to the
            # buffer (min index)
            cmax = max(b['counters'])
            idx = min([ i  for (i,c) in enumerate(b['counters']) if c==cmax ])
            # replace the selected old batch by the new batch
            b['counters'][idx] = 0
            b['data'][idx] = x
            #print('replacing',idx)

    def read_buffer(self, b):
        """
        get the least used minibatch
        if there are several of them, take the oldest one in the buffer,
        ie the one with the mininum index,
        to be coherent with the 'first in first out' writing
        """
        cmin = min(b['counters'])
        idx = min([ i  for (i,c) in enumerate(b['counters']) if c==cmin ])
        # recording that this batch has been used
        b['counters'][idx] += 1
        #print('reading ',idx)
        return b['data'][idx]

#######################
#Mozer pruning function
#######################
def forward_alpha(model,alpha, x):
        x = model.activations[0](model.conv1(x))
        x = x.view(-1, model.nb_filters * 8 * 8)
        x = [torch.mul(elt,alpha) for elt in x]
        x = torch.stack(x)
        x = model.fc1(x)
        return x

def relevance(model,dataloader):
    autograd_tensor = torch.ones((model.nb_filters * 8 * 8), requires_grad=True)
    num_items_read = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    autograd_tensor = autograd_tensor.to(device)
    gg = []
    lengths = []
    '''    test_data = datasets.CIFAR10(root='../data',
        train=False,
        download=True,
        transform=ToTensor())
    dataloader = DataLoader(test_data, batch_size=256, num_workers=8)
    '''
    #print(device,'used for training')
    for _, (X, y) in enumerate(dataloader):
        if 1000000 <= num_items_read:
            break
        X = X[:min(1000000 - num_items_read, X.shape[0])]
        y = y[:min(1000000 - num_items_read, X.shape[0])]
        num_items_read = min(1000000, num_items_read + X.shape[0])
        X = X.to(device)
        y = y.to(device)
        pred = forward_alpha(model,autograd_tensor,X)
        loss = loss_fn(pred, y)
        gg.append(torch.autograd.grad(loss, autograd_tensor, retain_graph=True))
        lengths.append(X.shape[0])
    normalization = torch.tensor([elt/sum(lengths) for elt in lengths])
    tensor_gg = torch.tensor([list(gg[k][0]) for k in range(len(gg))])
    result = [torch.sum(torch.mul(normalization,elt)) for elt in [tensor_gg[:,k] for k in range(tensor_gg.shape[1])]]
    return(-torch.tensor(result))

def skeletonization(model,pruning_level,dataloader):
    relevance_ = relevance(model,dataloader)
    size = int(model.fc1.weight.data.shape[1]*(1-pruning_level))
    keep_indices = torch.argsort(-relevance_)[:size]
    device = next(model.parameters()).device
    cpt = 0
    for index in set(range(model.fc1.weight.data.shape[1]))-set([int(elt) for elt in keep_indices]):
        cpt+=1
        #skeletone.fc1.weight.data[:,index] = torch.zeros(10)
        model.fc1.weight.data[:,index] = torch.zeros(10)
    loss_fn = torch.nn.CrossEntropyLoss()
    print('test accuracy',nets.evaluate(dataloader,model,loss_fn)[1],'after skeletonization')
    return()

#######################
#End of Mozer pruning function
#######################


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
            if self.data_points_max <= num_items_read:
                break
            X = X[:min(self.data_points_max - num_items_read, X.shape[0])]
            y = y[:min(self.data_points_max - num_items_read, X.shape[0])]
            num_items_read = min(self.data_points_max, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            self.train_1_batch(X, y, model, loss_fn)

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

    def train_1_epoch(self, dataloader, model, loss_fn, verbose=False):
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
            acceptance_ratio += self.train_1_batch(X, y, model, dataloader, loss_fn=torch.nn.CrossEntropyLoss(), verbose=verbose)
        return acceptance_ratio

    def train_1_batch(self, X, y, model, dataloader, loss_fn, verbose=False):
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
            test_data = datasets.CIFAR10(root='../data',
            train=False,
            download=True,
            transform=ToTensor())
            pruning_dataloader = DataLoader(test_data, batch_size=64, num_workers=8)
        for i in range(self.iter_mcmc):
            if i>0 and self.pruning_level>0 and i%200 == 0:#skeletonize any 50 mcmc iterations
                skeletonization(model,self.pruning_level,pruning_dataloader)
            # selecting a layer and a  at random
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
            else:
                # not accepting, so undoing the change
                self.selector.undo(model, neighborhood, epsilon)
                decision = 'rejected'
            if verbose:
                print('moove',decision)
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


