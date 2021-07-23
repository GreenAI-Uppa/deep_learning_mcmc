from abc import ABC, abstractmethod
import torch



def train_1_epoch_small_nei(dataloader, model, loss_fn, **kwargs):
    """
    either gradient or mcmc are used, depending on the arguments in kwargs
    using a small neighborood each time
    """
    if 'iter_mcmc' in kwargs:
        acceptance_ratio = 0.
    for batch, (X, y) in enumerate(dataloader):
        if 'iter_mcmc' in kwargs:
            iter_mcmc, lamb, proposal, prior= kwargs['iter_mcmc'], kwargs['lamb'], kwargs['proposal'], kwargs['prior']
            acceptance_ratio += mcmc_small_nei(X, y, model, loss_fn, proposal, prior=prior, lamb=lamb, iter_mcmc=iter_mcmc)
        else:
            lr = kwargs['lr']
            gradient(X, y, model, loss_fn, lr=lr)
    if 'iter_mcmc' in kwargs:
        return acceptance_ratio / (batch+1)
    else:
        return 0

class Optimizer(ABC):
    def __init__(self, data_points_max = 1000000000):
        """
        number of data points to used in the dataset
        """
        self.data_points_max = data_points_max

    def train_1_epoch(self, dataloader, model, loss_fn):
        num_items_read = 0
        # attempting to guess the device on the model. 
        device = next(model.parameters()).device
        for batch, (X, y) in enumerate(dataloader):
            if self.data_points_max <= num_items_read:
                break
            X = X[:min(self.data_points_max - num_items_read, X.shape[0])]
            y = y[:min(self.data_points_max - num_items_read, X.shape[0])]
            num_items_read = min(self.data_points_max, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            self.train_1_batch(X, y, model, loss_fn)

    @abstractmethod
    def train_1_batch(self, X, y, model):
        pass

class GradientOptimizer(Optimizer):
    def __init__(self, data_points_max = 1000000000, lr=0.001):
        super(GradientOptimizer, self).__init__(data_points_max = 1000000000)
        self.lr = lr

    def train_1_batch(self, X, y, model, loss_fn):
        """
        SGD optimization
        inputs:
        lr : learning rate
        """
        pred = model(X)
        los = loss_fn(pred, y)
        gg = torch.autograd.grad(los, model.parameters(), retain_graph=True)
        for i, layer in enumerate([model.conv1,model.fc1]):
            layer.weight.data -=  gg[2*i] * self.lr
            layer.bias.data -=  gg[2*i+1] * self.lr

class MCMCOptimizer(Optimizer):
    def __init__(self, sampler, nn_size = 'full', data_points_max = 1000000000, iter_mcmc=1, lamb=1000,  prior=None):
        """
        variance_prop : zero centered univariate student law class to generate the proposals
        variance_prior : zero centered univariate student law class used as a prior on the parameter values
        lamb : ponderation between the data and the prior
        iter_mcmc : number of mcmc iterations
        """
        super(MCMCOptimizer, self).__init__(data_points_max = 1000000000)
        self.iter_mcmc = iter_mcmc
        self.lamb = lamb
        self.sampler = sampler
        self.nn_size = nn_size
        if prior is None:
            self.prior = self.sampler
        else:
            self.prior = prior

    def train_1_epoch(self, dataloader, model, loss_fn, optimizer, **kwargs):
        """
        either gradient or mcmc are used, depending on the arguments in kwargs
        """
        results = {}
        num_items_read = 0
        acceptance_ratio_f, acceptance_ratio_l = 0, 0
        device = next(model.parameters()).device
        for batch, (X, y) in enumerate(dataloader):
            if self.data_points_max <= num_items_read:
                break
            X = X[:min(self.data_points_max - num_items_read, X.shape[0])]
            y = y[:min(self.data_points_max - num_items_read, X.shape[0])]
            num_items_read = min(self.data_points_max, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            mcmc_onebatch = self.train_1_batch(X, y, model, loss_fn)
            acceptance_ratio_f += mcmc_onebatch[0]
            acceptance_ratio_l += mcmc_onebatch[1]
        return acceptance_ratio_f / (batch+1), acceptance_ratio_l / (batch+1)

    def train_1_batch(self, X, y, model, loss_fn):
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
        n_filters = model.conv1.weight.data.shape[0]
        filter_size = model.conv1.weight.data.shape[1]*model.conv1.weight.data.shape[2]*model.conv1.weight.data.shape[3]
        n_output = model.fc1.weight.data.shape[0]
        n_hidden = model.fc1.weight.data.shape[1]
        accepts_f, not_accepts_f = 0., 0. # to keep track of the acceptance ratio of filter's mooves
        accepts_l, not_accepts_l = 0., 0. # to keep track of the acceptance ratio of fully connected layer's moove 
        cpt_f = 0#to avoid division by zero for acceptance ratios
        device = next(model.parameters()).device
        model = model.to(device)
        pred = model(X)
        loss = loss_fn(pred,y).item()
        cpt = 0
        for i in range(self.iter_mcmc):
            # selecting a filter at random
            if torch.randn(1)>0.1:
                moove = 'filter'
            else:
                moove = 'linear'
            if moove == 'filter':
                cpt_f +=1
                idx_row = torch.randint(0, n_filters, (1,))
                # sampling a proposal for this filter
                epsilon = torch.tensor(self.sampler.sample(filter_size+1).astype('float32'))[:,0]
                epsilon = epsilon.to(device)
                u = torch.flatten(model.conv1.weight.data[idx_row])#filter, needs torch.size([]) for filter
                v = model.conv1.bias.data[idx_row].reshape(1)#bias, do not do anything
                params_line = torch.cat((u,v))
                params_line = params_line.to(device)
                # getting the ratio of the students
                student_ratio, params_tilde = self.prior.get_ratio(epsilon, params_line)
                # applying the changes to get the new value of the loss
                model.conv1.weight.data[idx_row] += epsilon[:-1].reshape(model.conv1.weight.data[idx_row].shape)
                model.conv1.bias.data[idx_row] += epsilon[-1]
                pred = model(X)
                loss_prop = loss_fn(pred, y)
                # computing the change in the loss
                data_term = torch.exp(self.lamb * (loss -loss_prop))
                rho  = min(1, data_term * student_ratio)
                if rho > torch.rand(1).to(device):
                    # accepting, keeping the new value of the loss
                    accepts_f += 1
                    #print('Accept filter','data term',data_term,'prior ratio',student_ratio)
                    loss = loss_prop
                else:
                    # not accepting, so undoing the change
                    not_accepts_f += 1
                    model.conv1.weight.data[idx_row] -= epsilon[:-1].reshape(model.conv1.weight.data[idx_row].shape)
                    model.conv1.bias.data[idx_row] -= epsilon[-1]
                    #print('Reject filter moove','data term',data_term,'prior ratio',student_ratio)
            if moove == 'linear':
                idx_col = torch.randint(0, n_output, (1,))
                if type(self.nn_size) == int:
                    start = torch.randint(0,n_hidden-self.nn_size,(1,))
                    epsilon = torch.tensor(self.sampler.sample(self.nn_size+1).astype('float32'))[:,0]
                    epsilon = epsilon.to(device)
                    params_line = torch.cat((model.fc1.weight.data[idx_col][0][start:(start+self.nn_size)],model.fc1.bias.data[idx_col]))
                    params_line = params_line.to(device)
                    model.fc1.weight.data[idx_col][0][start:(start+self.nn_size)] += epsilon[:-1]
                    model.fc1.bias.data[idx_col] += epsilon[-1]
                elif self.nn_size == "vertical":
                    epsilon = torch.tensor(self.sampler.sample(model.fc1.weight.data[idx_col].shape[1]+1).astype('float32'))[:,0]
                    epsilon = epsilon.to(device)
                    params_line = torch.cat((model.fc1.weight.data[idx_col][0],model.fc1.bias.data[idx_col]))
                    params_line = params_line.to(device)
                    model.fc1.weight.data[idx_col] += epsilon[:-1]
                    model.fc1.bias.data[idx_col] += epsilon[-1]
                elif 'square' in self.nn_size:
                    square_size = int(self.nn_size[6:])
                    epsilon = torch.tensor(self.sampler.sample(square_size**2+1).astype('float32'))[:,0]
                    epsilon = epsilon.to(device)
                    #if (idx_col <= (n_output-square_size)):
                    #params_line = torch.cat((model.fc1.weight.data[idx_col][0][start:(start+self.nn_size)],model.fc1.bias.data[idx_col]))
                student_ratio, params_tilde = self.prior.get_ratio(epsilon, params_line)
                pred = model(X)
                loss_prop = loss_fn(pred, y)
                #computing the change in the loss
                data_term = torch.exp(self.lamb * (loss -loss_prop))
                rho  = min(1, data_term * student_ratio)
                if rho > torch.rand(1).to(device):
                    # accepting, keeping the new value of the loss
                    accepts_l += 1
                    #print('Accept linear moove','data term',data_term,'prior ratio',student_ratio)
                    loss = loss_prop
                else:
                    # not accepting, so undoing the change
                    not_accepts_l += 1
                    if type(self.nn_size) == int:
                        model.fc1.weight.data[idx_col][0][start:(start+self.nn_size)] -= epsilon[:-1]
                    else:
                        model.fc1.weight.data[idx_col] -= epsilon[:-1]
                    model.fc1.bias.data[idx_col] -= epsilon[-1]
                    #print('Reject linear moove','data term',data_term,'prior ratio',student_ratio)
        if cpt_f > 0 :
            acceptance_ratio_f = accepts_f / (not_accepts_f + accepts_f)
        else:
            acceptance_ratio_f = 0
        if cpt_f != self.iter_mcmc:
            acceptance_ratio_l = accepts_l / (not_accepts_l + accepts_l)
        else:
            acceptance_ratio_l = 0
        return acceptance_ratio_f, acceptance_ratio_l

def mcmc_small_nei(X, y, model, loss_fn, proposal, prior=None, lamb=1000000, iter_mcmc=50):
    """
    perform mcmc iterations with a neighborhood corresponding to one line of the parameters.

    the acceptance of the proposal depends on the following criterion
       exp(lamb * (loss_previous - loss_prop) ) * stud(params_prop) / stud(params_previous)

    inputs:
    X : input data
    y : input labels
    model : neural net we want to optimize
    loss_fn : loss function
    st : zero centered univariate student law class used to generate the proposals and as a prior on the parameter values
    lamb : ponderation between the data and the prior
    iter_mcmc : number of mcmc iterations

    outputs:
    acceptance_ratio
    model : optimised model (modified by reference)
    """
    n_outputs, n_inputs = model.linear.weight.data.shape
    accepts, not_accepts = 0., 0. # to keep track of the acceptance ratop
    pred = model(X)
    loss_prev = loss_fn(pred,y).item()
    Xflat = model.flatten(X)
    if prior is None:
        prior = proposal
    # pre sampling the proposals

    num_param = n_inputs * n_outputs + n_outputs
    epsilon = torch.tensor(proposal.sample(num_param).astype('float32')).reshape(n_outputs, n_inputs+1)
    for i in range(iter_mcmc):
        # selecting a parameter  at random
        idx_row = torch.randint(0, n_outputs, (1,))
        idx_col = torch.randint(0, n_inputs+1, (1,))

        # selecting a proposal for this parameter
        eps = epsilon[idx_row, idx_col]
        pred_new = pred

        # getting the ratio of the students and the new value for the prediction. It depends whether it's a bias weight or a more general weight
        if idx_col == n_inputs: # changing the bias
            pred_new[:,idx_row] = pred[:,idx_row] + epsilon[idx_row, idx_col]
            student_ratio, _ = prior.get_ratio(epsilon[idx_row,idx_col], model.linear.bias.data[idx_row])
        else:
            pred_new[:,idx_row] = pred[:,idx_row] + Xflat[:,idx_col]*epsilon[idx_row, idx_col]
            student_ratio, _ = prior.get_ratio(epsilon[idx_row,idx_col], model.linear.weight.data[idx_row, idx_col])

        # getting the new value for the loss
        loss_new = loss_fn(pred_new, y)

        # computing the data term
        data_term = torch.exp(lamb * (loss_prev-loss_new))

        # acceptance ratio
        rho  = min(1, data_term * student_ratio)
        if rho > torch.rand(1):
            # accepting, keeping the new value of the loss
            if idx_col==n_inputs:
                model.linear.bias.data[idx_row] += epsilon[idx_row, idx_col]
            else:
                model.linear.weight.data[idx_row, idx_col] += epsilon[idx_row, idx_col]
            loss_prev = loss_new
            accepts += 1
        else:
            not_accepts += 1
    acceptance_ratio = accepts / (not_accepts + accepts)
    return acceptance_ratio
