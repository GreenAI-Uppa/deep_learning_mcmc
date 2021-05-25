import torch


def gradient(X, y, model, loss_fn, lr=0.001):
    """
    SGD optimization

    lr : learning rate
    """
    pred = model(X)
    los = loss_fn(pred, y)
    gg = torch.autograd.grad(los, model.parameters(), retain_graph=True)
    model.linear.weight.data = model.linear.weight.data - gg[0] * lr 
    model.linear.bias.data = model.linear.bias.data - gg[1] * lr


def gradient_line(X, y, model, loss_fn):
    """
    SGD optimization on one random subset of the parameters

    """
    n_output = model.linear.weight.data.shape[0]
    for i in range(n_output):
        pred = model(X)
        los = loss_fn(pred, y)
        gg = torch.autograd.grad(los, model.parameters(), retain_graph=True)
        idx_row = torch.randint(0, n_output, (1,))
        model.linear.weight.data[idx_row] -= gg[0][idx_row] * lr 
        model.linear.bias.data[idx_row] -= gg[1][idx_row] * lr


def train_1_epoch(dataloader, model, loss_fn, **kwargs): 
    """
    either gradient or mcmc are used, depending on the arguments in kwargs
    """
    if 'iter_mcmc' in kwargs:
        acceptance_ratio = 0.
    for batch, (X, y) in enumerate(dataloader):
        if 'iter_mcmc' in kwargs:
            iter_mcmc, lamb, st = kwargs['iter_mcmc'], kwargs['lamb'], kwargs['student']
            acceptance_ratio += mcmc(X, y, model, loss_fn, st, lamb=lamb, iter_mcmc=iter_mcmc)
        else:
            lr = kwargs['lr']
            gradient(X, y, model, loss_fn, lr=lr)
    if 'iter_mcmc' in kwargs:
        return acceptance_ratio / (batch+1)
    else:
        return 0

def mcmc(X, y, model, loss_fn, st, lamb=1000000, iter_mcmc=50):
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
    n_output = model.linear.weight.data.shape[0]
    accepts, not_accepts = 0., 0. # to keep track of the acceptance ratop
    pred = model(X)
    loss = loss_fn(pred,y).item()
    for i in range(iter_mcmc):
        # selecting a line at random
        idx_row = torch.randint(0, n_output, (1,))
        # sampling a proposal for this line
        epsilon = torch.tensor(st.sample(model.linear.weight.data[idx_row].shape[1]+1).astype('float32'))[:,0]
        params_line = torch.cat((model.linear.weight.data[idx_row][0],model.linear.bias.data[idx_row]))

        # getting the ratio of the students
        student_ratio, params_tilde = st.get_ratio(epsilon, params_line)

        # applying the changes to get the new value of the loss
        model.linear.weight.data[idx_row] += epsilon[:-1]
        model.linear.bias.data[idx_row] += epsilon[-1]
        pred = model(X)
        loss_prop = loss_fn(pred, y)

        # computing the change in the loss
        data_term = torch.exp(lamb * (loss -loss_prop))

        rho  = min(1, data_term * student_ratio)
        if rho > torch.rand(1):
          # accepting, keeping the new value of the loss 
          accepts += 1
          loss = loss_prop
        else:
          # not accepting, so undoing the change
          not_accepts += 1
          model.linear.weight.data[idx_row] -= epsilon[:-1]
          model.linear.bias.data[idx_row] -= epsilon[-1]
    acceptance_ratio = accepts / (not_accepts + accepts)
    return acceptance_ratio
