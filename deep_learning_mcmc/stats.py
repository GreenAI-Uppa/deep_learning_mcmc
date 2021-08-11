import numpy as np
import functools
from operator import mul
import math
import sys
current_module = sys.modules[__name__]


def build_distr(config):
    selector_name = config["name"]
    if selector_name == "Student":
        variance = config["variance"]
        return getattr(current_module, selector_name)(variance)
    elif selector_name=="BinarySampler":
        return getattr(current_module, selector_name)()
    else:
        variance = config["variance"]
        real_sampler = Student(variance)
        binary_sampler = BinarySampler()
        return MixedSampler(real_sampler, binary_sampler)

class MixedSampler(object):
    def __init__(self, real_sampler, binary_sampler):
        self.real_sampler = real_sampler
        self.binary_sampler = binary_sampler

    def sample(self, neighborhood):
        is_binary, neighborhood_size = neighborhood
        if is_binary:
            return self.binary_sampler.sample(neighborhood_size)
        else:
            return self.real_sampler.sample(neighborhood_size)

    def get_lambda(self, neighborhood):
        pass

    def get_ratio(self, epsilon, params, neighborhood_info):
        is_binary, _ = neighborhood_info
        if is_binary:
            return self.binary_sampler.get_ratio(epsilon, params, neighborhood_info)
        else:
            return self.real_sampler.get_ratio(epsilon, params, neighborhood_info)

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

    def get_ratio(self, epsilon, params, neighborhood_info):
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


class BinarySampler(object):
    def sample(self, n):
        return None
    def get_ratio(self, epsilon, params, neighborhood_info):
        return 1
