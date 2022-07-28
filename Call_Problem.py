
import numpy as np
import sympy as sp 
import scipy as sc
import networkx as nx
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from sympy.physics.quantum import TensorProduct
import itertools 

from Radio import Channel_Rayleigh, Gauss_Noise
from PyQubo_formulation import *


class Problemv2:
    def __init__(self, problem_size, x_domain, nq_per_var, s_message):
        
        self.no_users, self.no_transmit = np.array(problem_size, dtype=int)
        self.nq = nq_per_var
        self.s = s_message

        self.Helpers = PyQUBO_Helpers(self.no_transmit*2, self.nq)
        self.xtilde = self.Helpers.Create_X(x_domain)

        self.L2  = self.Helpers.L2(self.xtilde)
        self.EVM = self.Helpers.EVM(self.xtilde, self.s)
        self.Max = self.Helpers.Max_Norm_LP
    
    def Get_Problem(self, problem, term_penalties):
        
        valid_problems = {'L2', 'EVM', 'MAX', 'PAPR', 'FULL'}

        if problem not in valid_problems:
            raise ValueError("problem_type must be one of: %r." % valid_problems)

        valid_fun = {'L2' :  self.L2, 
                    'EVM' :  self.EVM,
                    'MAX' :  self.Max(self.xtilde, term_penalties[0]),
                    'PAPR':  self.Max(self.xtilde, term_penalties[0]) - self.L2, 
                    'FULL':  self.Max(self.xtilde, term_penalties[0]) - self.L2 + term_penalties[1]*self.EVM
                    }
        
        problem = valid_fun[problem]
        
        return problem



############################### REWRITTING THE ABOVE BELOW, TO ACCOUNT FOR N = # OF OFDM SUBCARRIERS. ###############################



class Make_Problem:
    def __init__(self, MIMO_params, nq_per_var, message):
        
        self.K, self.M, self.N = MIMO_params

        self.nqx, self.nqk, self.nqmu = nq_per_var

        self.s = message ## augmented so containts [s_1^T, s_2^T, ..., s_K^T]
    
    ## User might want to change 'nq_per_var' without running the whole method again.
    @property
    def nq_specs(self):
        return self.nqx, self.nqk, self.nqmu
    
    @nq_specs.setter
    def nq_specs(self, nqx_val, nqk_val, nqmu_val):
        self.nqx, self.nqk, self.nqmu = nqx_val, nqk_val, nqmu_val
    
    def Get_Problem(self):

        return []



if __name__ == "__main__":
    s = np.array([1-1J, 3+1J], dtype=complex)
    trial = Problemv2([2,2], [-4,4], 4, s)
    test = trial.Get_Problem('MAX', [2,3,1])
    test_model = test.compile()
    print(test_model.to_qubo(index_label=True),'\n')
    print(len(test_model.variables), '\n')

    from Utilities import Dict_to_Mat
    qubomat = Dict_to_Mat(test_model.to_qubo(index_label=True)[0])
    graph = nx.from_numpy_array(qubomat)
    nx.draw(graph, node_size=4)
    plt.show()

    import dwave_networkx as dnx 
    G_pegasus = dnx.pegasus_graph(6)

    import minorminer as mm 
    embedding = mm.find_embedding(graph, G_pegasus, return_overlap=True)
    print(embedding, '\n')
    print(len(embedding))
    dnx.draw_pegasus_embedding(G_pegasus, embedding[0], node_size=4)
    plt.show()
    embed_arr = Dict_to_Mat(embedding)
    print(embed_arr)