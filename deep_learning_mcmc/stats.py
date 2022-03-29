import numpy as np
import functools
from operator import mul
import math
import sys
current_module = sys.modules[__name__]

def build_samplers(config):
    """
    build a list of sampler and prior
    both are the same, only the usage change.
    """
    samplers = []
    lambdas = []
    for sampler_config in config:
        sampler = build_sampler(sampler_config['sampler'])
        prior = build_sampler(sampler_config['prior'])
        samplers.append({'sampler':sampler, 'prior': prior})
        lambdas.append(sampler_config['lamb'])
    return SamplerList(samplers, lambdas)

def build_sampler(config):
    """build 1 sampler or prior"""
    selector_name = config["name"]
    if selector_name == "Student" or selector_name == "StudentParallel":
        variance = config["variance"]
        return getattr(current_module, selector_name)(variance)
    elif selector_name=="BinarySampler":
        return getattr(current_module, selector_name)()
    else:
        raise RuntimeError("Found unknown sampler:" +selector_name+ " in the config. Please provide Student or BinarySampler")


class SamplerList(object):
    def __init__(self, samplers, lambdas):
        self.samplers = samplers
        self.lambdas = lambdas

    def get_lambda(self, neighborhood):
        """return the lambda value given the layer index"""
        layer_idx, neighborhood_size = neighborhood
        return self.lambdas[layer_idx]

    def sample(self, neighborhood):
        """
        return a 1d proposal for the specified layer index and neighborhood
        """
        layer_idx, neighborhood_size = neighborhood
        return self.samplers[layer_idx]['sampler'].sample(neighborhood_size)

    def get_ratio(self, epsilon, params, neighborhood_info):
        """
        compute the
        epsilon : delta drawn from the sampler
        params : parameter weight of the model
        neighborhood_info : information about the current network layer
        """
        layer_idx, _ = neighborhood_info
        return self.samplers[layer_idx]['prior'].get_ratio(epsilon, params)

class Student(object):
    """
    univariate student distribution
    """
    def __init__(self, s, m=0, df=1):
        """
        s : variance
        m : mean
        df : degree of freedom
        """
        self.m = m
        self.d = 1 # univariate
        self.S = np.eye(1) * s
        self.df = df
        # precomputing this term for later use when computing data likelihood
        self.gamma_factor = math.gamma(1. * (self.d+self.df)/2) / ( math.gamma(1.*self.df/2) * pow(self.df*math.pi,1.*self.d/2) * pow(np.linalg.det(self.S),1./2) )

    def sample(self, n=1):
        '''
        Output:
        Produce M samples of d-dimensional multivariate t distribution
        Input:
        n = # of samples to produce
        '''
        g = np.tile(np.random.gamma(self.df/2.,2./self.df,n),(self.d,1)).T
        Z = np.random.multivariate_normal(np.zeros(self.d),self.S,n)
        return (self.m + Z/np.sqrt(g)).reshape(n,)

    def t_distribution_fast(self, x):
        '''
        for use when computing the likelihood of multiples univariate variables
        '''
        assert self.df == 1
        Denom = 1 + x*x/self.S[0][0]
        d = 1. * self.gamma_factor / Denom
        return d

    def t_distribution(self, x):
        '''
        copied from https://stackoverflow.com/questions/29798795/multivariate-student-t-distribution-with-python
        Multivariate t-student density:
        output:
            the density of the given element
        input:
            x = parameter (d dimensional numpy array or scalar)
            mu = mean (d dimensional numpy array or scalar)
            Sigma = scale matrix (dxd numpy array)
            df = degrees of freedom
            d: dimension

        If x is a large vector, see the t_distribution_fast function
        '''
        Num = math.gamma(1. * (self.d+self.df)/2)
        Denom = ( math.gamma(1.*self.df/2) * pow(self.df*math.pi,1.*self.d/2) * pow(np.linalg.det(self.S),1./2) * pow(1 + (1./self.df)*np.dot(np.dot((x - self.m),np.linalg.inv(self.S)), (x - self.m)),1.* (self.d+self.df)/2))
        d = 1. * Num / Denom
        return d

    def get_ratio(self, epsilon, params, no_reduction=False):
        """
        compute the likelihood ratio of two variables
           student(params[i] + epsilon[i])
        Prod_i (   ------------------------      )
               student(params[i])
        """
        #apply the move to get theta tilde
        params_tilde = params + epsilon

        # get the likelihood of the theta
        den = self.t_distribution_fast(params)

        # get the likelihood of the theta tilde
        num = self.t_distribution_fast(params_tilde)

        ratio = num / den
        return functools.reduce(mul, ratio, 1)

class StudentParallel(Student):
    def sample(self, sizes):
        '''
        Output:
        Produce M samples of d-dimensional multivariate t distribution
        Input:
        n = # of samples to produce
        '''
        n = sizes[0] * sizes[1]
        g = np.tile(np.random.gamma(self.df/2.,2./self.df,n),(self.d,1)).T
        Z = np.random.multivariate_normal(np.zeros(self.d),self.S,n)
        return (self.m + Z/np.sqrt(g)).reshape(sizes[0],sizes[1])

    def get_ratio(self, epsilon, params):
        """
        compute the likelihood ratio of two variables
           student(params[i] + epsilon[i])
        Prod_i (   ------------------------      )
               student(params[i])
        """
        #apply the move to get theta tilde
        params_tilde = params + epsilon

        # get the likelihood of the theta
        den = self.t_distribution_fast(params)

        # get the likelihood of the theta tilde
        num = self.t_distribution_fast(params_tilde)

        ratio = num / den
        return ratio

class BinarySampler(object):
    def sample(self, n, p=0.5):
        """
        return vector of realisations of bernouli variables
        """
        return np.random.binomial(size=n, n=1, p = p) * 2 - 1

    def get_ratio(self, epsilon, params):
        """ no prior """
        return 1
