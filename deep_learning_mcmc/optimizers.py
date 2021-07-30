from abc import ABC, abstractmethod
import torch

def train_1_epoch_small_nei(dataloader, model, loss_fn, **kwargs):
    """
    either gradient or mcmc are used, depending on the arguments in kwargs
    using a small neighborhood each time
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
        for i, layer in enumerate(model.linears):
            layer.weight.data -=  gg[2*i] * self.lr
            layer.bias.data -=  gg[2*i+1] * self.lr

class MCMCOptimizer(Optimizer):
    def __init__(self, sampler, data_points_max = 1000000000, iter_mcmc=1, lamb=1000,  prior=None, selector=None):
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
        if prior is None:
            self.prior = self.sampler
        else:
            self.prior = prior
        self.selector = selector

    def train_1_epoch(self, dataloader, model, loss_fn, optimizer, **kwargs):
        """
        either gradient or mcmc are used, depending on the arguments in kwargs
        """
        acceptance_ratio,  num_items_read, results = 0, 0, {}
        device = next(model.parameters()).device
        for batch, (X, y) in enumerate(dataloader):
            if self.data_points_max <= num_items_read:
                break
            X = X[:min(self.data_points_max - num_items_read, X.shape[0])]
            y = y[:min(self.data_points_max - num_items_read, X.shape[0])]
            num_items_read = min(self.data_points_max, num_items_read + X.shape[0])
            X = X.to(device)
            y = y.to(device)
            acceptance_ratio += self.train_1_batch(X, y, model, loss_fn)
        return acceptance_ratio / (batch+1)

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
        device = next(model.parameters()).device
        accepts, not_accepts = 0., 0. # to keep track of the acceptance ratop
        pred = model(X)
        loss = loss_fn(pred,y).item()
        for i in range(self.iter_mcmc):
            # selecting a line at random
            neighborhood = self.selector.get_neighborhood()
            epsilon = torch.tensor(self.sampler.sample(self.selector.neighborhood_size).astype('float32')).to(device)
            params_line = self.selector.getParamLine(neighborhood, model)
            # getting the ratio of the students
            student_ratio = self.prior.get_ratio(epsilon, params_line)

            # applying the changes to get the new value of the loss
            self.selector.update(model, neighborhood, epsilon)
            pred = model(X)
            loss_prop = loss_fn(pred, y)

            # computing the change in the loss
            data_term = torch.exp(self.lamb * (loss -loss_prop))

            rho  = min(1, data_term * student_ratio)
            if rho > torch.rand(1).to(device):
              # accepting, keeping the new value of the loss
              accepts += 1
              loss = loss_prop
            else:
              # not accepting, so undoing the change
              not_accepts += 1
              self.selector.undo(model, neighborhood, epsilon)
        acceptance_ratio = accepts / (not_accepts + accepts)
        return acceptance_ratio

class MCMCSmallNei(MCMCOptimizer):
    def train_1_batch(self, X, y, model, loss_fn):
        """
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
        device = next(model.parameters()).device
        n_outputs, n_inputs = model.linears[0].weight.data.shape
        accepts, not_accepts = 0., 0. # to keep track of the acceptance ratop
        pred = model(X)
        loss_prev = loss_fn(pred.double(),y).item()
        Xflat = model.flatten(X)
        # pre sampling the proposals
        num_param = n_inputs * n_outputs + n_outputs
        epsilon = torch.tensor(self.sampler.sample(num_param).astype('float32')).reshape(n_outputs, n_inputs+1).to(device)
        for i in range(self.iter_mcmc):
            # selecting a parameter  at random
            idx_row = torch.randint(0, n_outputs, (1,))
            idx_col = torch.randint(0, n_inputs+1, (1,))

            # selecting a proposal for this parameter
            eps = epsilon[idx_row, idx_col]
            pred_new = pred

            # getting the ratio of the students and the new value for the prediction. It depends whether it's a bias weight or a more general weight
            if idx_col == n_inputs: # changing the bias
                pred_new[:,idx_row] = pred[:,idx_row] + epsilon[idx_row, idx_col]
                student_ratio, _ = self.prior.get_ratio(epsilon[idx_row,idx_col], model.linears[0].bias.data[idx_row])
            else:
                pred_new[:,idx_row] = pred[:,idx_row] + Xflat[:,idx_col]*epsilon[idx_row, idx_col]
                student_ratio = self.prior.get_ratio(epsilon[idx_row,idx_col], model.linears[0].weight.data[idx_row, idx_col])
            # getting the new value for the loss
            loss_prop = loss_fn(pred_new.double(), y)

            # computing the data term
            data_term = torch.exp(self.lamb * (loss_prev-loss_prop))
            # acceptance ratio
            rho  = min(1, data_term * student_ratio)
            if rho > torch.rand(1).to(device):
                # accepting, keeping the new value of the loss
                if idx_col==n_inputs:
                    model.linears[0].bias.data[idx_row] += epsilon[idx_row, idx_col]
                else:
                    model.linears[0].weight.data[idx_row, idx_col] += epsilon[idx_row, idx_col]
                loss_prev = loss_prop
                pred = pred_new
                accepts += 1
            else:
                not_accepts += 1
        acceptance_ratio = accepts / (not_accepts + accepts)
        return acceptance_ratio

class MCMCSmallNeiAppr(MCMCOptimizer):
    def train_1_batch(self, X, y, model, loss_fn):
        """
        perform one approximate mcmc iteration

        We generate an epsilon proposal for each parameter, and compute the acceptance ratio for each of these proposals indepedantly.

        So we are dealing with a batch_size X n_outputs X num_parameters matrix. It is not safe to use large batch sizes.

        the acceptance of the proposal depends on the following criterion
           exp(lamb * (loss_previous - loss_prop) ) * prior(params_prop) / prior(params_previous)

        inputs:
        X : input data
        y : input labels
        model : neural net we want to optimize
        loss_fn : loss function
        proposal : zero centered univariate student law class used to generate the proposals
        prior : prior on the parameter values
        lamb : ponderation between the data and the prior
        iter_mcmc : number of mcmc iterations

        outputs:
        acceptance_ratio
        model : optimised model (modified by reference)
        """
        n_outputs, n_inputs = model.linears[0].weight.data.shape
        num_param = n_inputs * n_outputs + n_outputs # linear layer : inputs x outputs + bias weights
        batch_size = X.shape[0]

        # getting the current value of the loss
        pred = model(X)
        device = pred.get_device()
        if device == -1:
            device = 'cpu'

        loss_prev = loss_fn(pred.double(),y).item()

        # sampling the proposals
        epsilon = torch.tensor(self.sampler.sample(num_param).astype('float32')).reshape(n_outputs, n_inputs+1).to(device)
        eps_matrix = epsilon.reshape(1, n_outputs, n_inputs+1).repeat(batch_size, 1, 1) # minibatch_size x output_size x (input_size+1)

        # getting the predictions for these proposals
        X_mat = model.flatten(X).reshape(batch_size, 1, n_inputs) # minibatch_size x input_size
        X_mat = X_mat.repeat(1, n_outputs, 1) # minibatch_size x output_size x input_size

        # this corresponds will be added to the prediction
        # pred_i = pred_hat_i + eps_ij * xj if eps_ij is not a bias parameter
        # pred_i = pred_hat_i + eps_ij if eps_ij is a bias parameter
        eps_matrix[:,:,:-1] *= X_mat

        # now we need to reshape and build the matrix so that we can just perform adding to prediction matrix
        eps_matrix = model.flatten(eps_matrix) #.reshape(batch_size, 1, (n_outputs* (n_inputs+1))) # minibatch_size x (output_size x (input_size+1))
        eps_matrix_ = torch.zeros(batch_size, 10, (n_outputs* (n_inputs+1)))
        for i in range(n_outputs):
            eps_matrix_[:,i,i*(n_inputs+1):(i+1)*(n_inputs+1)] = eps_matrix[:,i*(n_inputs+1):(i+1)*(n_inputs+1)]
        # reshaping as well the prediction matrix.
        pred_mat = pred.reshape(batch_size, n_outputs, 1).repeat(1, 1, num_param) # minibatch x output_size x (output_size x (input_size+1))

        # adding
        pred_mat += eps_matrix_
        pred_mat[pred_mat<0] = 0 # performing relu

        # converting to one hot
        y = y.reshape((y.shape[0],1))
        y_mat = torch.FloatTensor(y.shape[0], n_outputs)
        y_mat.zero_()
        y_mat.scatter_(1, y, 1)

        # reshaping the ground truth so that is matches the prediction
        y_mat = y_mat.reshape(batch_size, n_outputs, 1).repeat(1,1,num_param)
        # computing the loss
        losses = ((pred_mat.double() - y_mat.double())**2).mean(dim=[0,1]) # 1 x num_params
        # the delta of the loss
        data_terms = torch.exp(self.lamb * (loss_prev - losses)).reshape(n_outputs, n_inputs+1) # output_size X (input_size+1)
        student_ratios = self.prior.get_ratio(epsilon, torch.cat((model.linear.weight.data,model.linear.bias.data.reshape(n_outputs,1)), dim=1), do_mul=False) # (output_size X (input_size+1))
        rho = data_terms * student_ratios
        accepted_changes = rho > torch.rand(n_outputs,n_inputs+1).to(device)
        model.linears[0].weight.data[accepted_changes[:,:-1]] = params_prop[:,:-1][accepted_changes[:,:-1]]
        model.linears[0].bias.data[accepted_changes[:,-1]] = params_prop[:,-1][accepted_changes[:,-1]]
        pred = model(X)
        loss_prev = loss_fn(pred,y).item()
        acceptance_ratio = float(accepted_changes.sum() / float(num_param))
        return acceptance_ratio
