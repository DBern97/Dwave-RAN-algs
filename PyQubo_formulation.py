""" 
Methods that help build and/or create the PAPR/EVM problem(s) using mostly the PyQUBO library.

The reason for using PyQUBO instead of Sympy was PyQUBO's compile() and other helpful 
(sometimes also faster) methods that are tailered for binary samplers. 
"""


import numpy as np
import sympy as sp 
import pyqubo as pq

from sympy.physics.quantum import TensorProduct
from scipy.linalg import dft

import itertools 

from Radio import *


def Float_Approx(x, domain):
    """
    Takes array of binary variables and/or values 'x' and 
    outputs an approximation of a float that lives in 'domain'.
    
    Used to created the QUBO model AND to later convert bit-strings
    into floats. 
    """
    
    x = np.array(x)
    nq = len(x)
    
    low, high = domain
    d = - low
    
    # scale
    c = (high + d) / 2

    # Powers of the binary vars
    powers = np.linspace(0, -int(nq-1), nq, dtype=int)

    # array of coefficients i.e. 2**powers
    coeff = 2*np.ones(nq)
    coeff = np.power(coeff, powers)

    approx = c*np.dot(coeff, x) - d

    return approx


#might not have to be a class as they dont need to share any params here. 
# its only later when comiling them together that a class is useful.

# class PyQUBO_Helpers:
#     def __init__(self, num_bin_vars, nq_per_var):
        
#         self.num_bin = num_bin_vars
#         self.nq = nq_per_var

def Create_Float_Vec(length, nq, domains, var_name):
    """
    TODO: do nq per element bcs larger domains might
    need more qubits to achieve the same accuracy ?
    """

    qubits = pq.Array.create(str(var_name), shape=(length*nq, ), vartype='BINARY')

    # get the variable names in qubits.. should find better way
    empty = np.sum(qubits).compile()
    bin_vars = empty.variables

    xtilde = np.zeros(length, dtype='object')

    ## if domains contains just one domain:
    if len(domains.shape) == 1:
        for i in range(length):
            xtilde[i] = Float_Approx(qubits[i*nq : nq*(i+1),], domains)
    
    ## accomodate for varying domains per element
    elif len(domains.shape) > 1:
        for i in range(length):
            xtilde[i] = Float_Approx(qubits[i*nq : nq*(i+1),], domains[i])

    xtilde = pq.Array(xtilde)

    return xtilde, bin_vars

def L2(x):
    "Given a pyqubo array 'x', return its L2-norm."
    return x.dot(x)

def EVM(x, s, noise=True):
    """
    Given a symbolic vector 'x' (pyQubo array, e.g. Create_float_vec() ) 
    and message vector 's' (can be complex), return the L2-norm 
    corresponding to the EVM constraint.

    Uses H and n from Radio.py.
    """
    
    # dimensions of channel matrix H
    no_users = int(len(s))
    no_transmit = int(len(x)/2)

    s_im = s.imag
    s_re = s.real

    s = pq.Array(np.concatenate((s_re,s_im)))

    H = sp.Matrix(Channel_Rayleigh(no_users, no_transmit))

    # TODO: also adjust for only real xtilde
    T = sp.ones(2)
    T[0,1] = -1
    ID = sp.eye(2)
    T = T-ID

    H_re = sp.re(H)
    H_im = sp.im(H)

    # make H into block matrix to treat real and imaginary parts 'separately'
    H = np.array(TensorProduct(T, H_im)) + np.array(TensorProduct(ID, H_re))
    H = pq.Array(H)

    n = Gauss_noise(no_users)
    n = pq.Array(np.concatenate((n.real,n.imag)))
    if noise is False:
        n = np.zeros(no_users)
    else:
        None

    EVM_pen = H.matmul(x) + n - s
    EVM = EVM_pen.T.matmul(EVM_pen)

    return EVM

def Time_Dom(xtilde, only_real=False):
    """
    Given 'xtilde' (e.g. pyQubo array, but float arr should also work)
    understood as an array with its real and imaginary parts stacked
    as xtilde=[real, imag], return its IDFT.
    """

    if only_real is False:
        size = int(len(xtilde)/2)

        # important to scale so that F is unitary
        F = np.around(dft(size, scale='sqrtn'), decimals=4) #TODO: handle accuracy as a parameter when calling the functions
        F = F.T.conj()
        F = sp.Matrix(F)

        # Create re/im block matrix
        F_re = sp.re(F)
        F_im = sp.im(F)

        T = sp.ones(2)
        T[0,1] = -1
        ID = sp.eye(2)
        T = T - ID

        # rewriting F s.t. real and imaginary parts can be treated 'separately'
        F = sp.Matrix(TensorProduct(T, F_im)) + sp.Matrix(TensorProduct(ID, F_re))
        F = np.array(F, dtype=float)
        F = pq.Array(F)

    ## this was useful when testing
    else:
        size = int(len(xtilde))

        F = np.around(dft(size, scale='sqrtn'), decimals=4) 
        F = F.T.conj()
        F = F.real
        F = pq.Array(F)

    # pyqubo syntax
    ytilde = F.matmul(xtilde)

    return ytilde


def Max_Norm_LP(ytilde, mu, k_slack_vars, gamma):
    """
    Given 'ytilde' (e.g. pyQubo array but float array should also work)
    return Linear Program equivalent of max-norm of ytilde in a unconstrained
    form, i.e. where all constraints are added to the objective.

    mu           :  mu is the objective to be minimised. Expected as pyqubo array.
    k_slack_vars :  an array of slack variables k_n. k_n 'replace' inequalities.
    gamma        :  penalty scalar for the sum of constraints.
    """

    # write each constraint into a vector
    term = np.zeros(len(ytilde), dtype='object')

    for i in range(len(ytilde)):
        term[i] = (ytilde[i]**2 + k_slack_vars[i] - mu)**2

    term = pq.Array(term)

    # max-norm as obj + gamma*constraints
    max_norm = mu + gamma*sum(term)

    return max_norm

# def Compile(problem, qubo=True):

#     compiled = problem.compile()

#     if qubo is False:
#         compiled = compiled.to_ising(index_labels=True)
#     else:
#         compiled = compiled.to_qubo(index_labels=True)
    
#     return compiled

if __name__ == "__main__":
    # trial = PyQUBO_Helpers(2,4)
    approximation = float_approx([-4,4], 4, 'binary')
    print(approximation)