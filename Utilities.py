
"""
Contains helper functions to read, plot and analyse anneal data.
"""

import numpy as np
import networkx as nx
import sympy as sp 
import pyqubo as pq

from sympy.physics.quantum import TensorProduct
from scipy.linalg import dft

import itertools 

from Radio import *




def Compile(problem, qubo=True):

    compiled = problem.compile()

    if qubo is False:
        compiled = compiled.to_ising(index_labels=True)
    else:
        compiled = compiled.to_qubo(index_labels=True)
    
    return compiled

# TODO: for ising as well
def Dict_to_Mat(qubo_dict):
    keylist = list(qubo_dict.keys())

    num_vars = int(max(keylist)[0] + 1)
    qubo_mat = np.zeros((num_vars, num_vars))

    for key in keylist:
        qubo_mat[key[0]][key[1]] = qubo_dict[key]

    # qubo_mat = sp.Matrix(qubo_mat)
    return qubo_mat

def chain(embedding):
    chains = list(embedding[0].values())

    chain_lengths = []
    for chain in chains:
        chain_lengths.append(len(chain))
    return chains, chain_lengths

def quad_error(sample, variables):
    """
    Given a result sample and list of variables,
    check if quadratisation was respected. Checks if
    qk*ql == zkl.

    Returns a realtive error for given sample.
    """
    
    combinations = list(itertools.combinations(variables, 2))

    res = []
    for pair in combinations:
        res.append(sample[pair[0]]*sample[pair[1]] == sample[pair[0]+' * '+pair[1]])

    res = np.array(res)
    res = res.astype(int)
    
    error = np.count_nonzero(res==0) / len(res)
    
    return error

def nq_embedding(chains):
    return len(list(itertools.chain(*chains)))

import random

def create_s(length):
    real = np.array([random.randrange(-3,4,2) for p in range(0,length)])
    im = np.array([random.randrange(-3,4,2) for p in range(0,length)])

    return real + im*1J

def Embedder(problem):
    return 0
