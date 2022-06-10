""" 
Methods that help build and/or create the PAPR/EVM problem(s) using mostly the Sympy library.

Sympy gives greater flexibility but seems slower than PyQUBO. See PyQUBO_formulation.py for
almost equivalent methods.
"""


import numpy as np
import networkx as nx
import sympy as sp 
import pyqubo as pq

from sympy.physics.quantum import TensorProduct
from scipy.linalg import dft

import itertools 

from Radio import *



class Sympy_Helpers:
    def __init__(self):
        self.nq = 0

    ### Helper functions to get symbolic expressions into QUBO format:
    def Bin_Approx_Symbolic(self, name, domain, nq):
        """ 
        Create a binary (similar to radix-2) approximation of continuous 'name'
        variable in some 'domain' with 'nq' qubits.
        """

        low, high = domain

        # for now, to match the nq requirement
        nq = nq - 2

        # Largest power and the offset
        p_max = int(np.log2(high) - 1)
        sub = int(nq/2) - p_max

        # Powers of the binary vars
        powers = np.linspace(-int(nq/2), int(nq/2), nq+1, dtype=int)
        powers = powers - sub

        # array of coefficients i.e. 2**powers
        coeff = 2*np.ones(nq+1)
        coeff = np.power(coeff, powers)

        upper = np.sum(coeff)

        c = -(upper - low)

        # now coeff is an array of coeffs for a single element of x.
        coeff = np.append(coeff,c)

        # Create bin vars symbolically
        varname = 'q'+str(name)+'_'
        qname = sp.symbols('{}:{}'.format(varname,nq+2))
        qname = np.array(list(qname))

        # Create the symbolic sum
        elements = coeff*qname
        bin_approx = sp.Add(*[elements[j] for j in range(len(elements))])

        return bin_approx
    
    def Create_Xtilde_Sym(self, length, domain, nq):
        """
        Create a vector representing th eprecoded vector x, split into Re() and Im() parts.
        Each element is created as per Bin_Approx_Symbolic.

        Returns a sympy Matrix object.
        """

        a = sp.Matrix(sp.symbols('a:{}'.format(int(length))))
        b = sp.Matrix(sp.symbols('b:{}'.format(int(length))))

        for i in range(length):
            a[i] = self.Bin_Approx_Symbolic(str(a[i]), domain, nq)
            b[i] = self.Bin_Approx_Symbolic(str(b[i]), domain, nq)

        x = a.col_join(b)

        return x
    
    def Quad2Lin(self, x):
        """
        Given expression 'x' with QUBO variables, turns quadtraticc terms
        into linear.  
        """
        x = sp.Poly(x)
        variables = x.args[1:]

        for q in variables:
            x = x.subs(q**2,q)
        return x

    def Create_Qubo(self, expression):
        """
        Creates a QUBO matrix given an 'expression' that is at most quadratic.
        """
        expression = sp.Poly(expression)
        variables = expression.args[1:]

        Qubo = sp.zeros(len(variables))

        print(r'Creating a {}x{} upper-triangular QUBO matrix..'.format(len(variables), len(variables)))

        for i in range(len(variables)):
            Qubo[i,i] = expression.coeff_monomial(variables[i])
            for j in range(i+1,len(variables)):
                Qubo[i,j] = expression.coeff_monomial(variables[i]*variables[j])

        return Qubo

    """ TODO: find and replace all high oredr terms with the min number of auxiliaries and use pyQUBO library
    def Add_Aux(poly):

        # just in case not Poly object
        poly = sp.Poly(poly)
        variables = poly.args[1:]

        monomials = poly.monoms()
        monomials = np.array(monomials)

        # find the high order monoms:
        trouble = np.count_nonzero(monomials, axis=1)
        trouble = np.argwhere(trouble > 2)
        highord = monomials[trouble][:,0]

        print(highord.shape)
        totals = np.sum(highord, axis=0)
        print(totals)

        # look which occur most freqently
        var_occur = np.argwhere(highord == 1)
        print(var_occur)

        # only need to look at the 'second' index
        var_occur = var_occur[:,1]
        total = Counter(var_occur)



        return total
    """
    
    
    
    def Aux_ab(self, poly):

        variables = poly.args[1:]

        # group them together - this requires knowing len(x) and nq_x TODO: handle in a class
        # do nq_x*len(x) = 4*2 in this case so:
        qubits_a, qubits_b = variables[:8], variables[8:16]


        # define all possible pairs to be replaced:
        combo_a, combo_b = list(itertools.combinations(qubits_a, 2)), list(itertools.combinations(qubits_b, 2))


        # define the auxiliares for each pair in combo_a & combo_b:
        combo_indices = np.array(list(itertools.combinations(range(len(qubits_a)),2)))

        z_a = []
        z_b = []

        #TODO: fix the names - for example za_14 is qa0_1*qa1_0, but the name does not indicate that
        for i,j in combo_indices:
            z_a.append(sp.symbols('za_{}'.format(str(i)+str(j))))
            z_b.append(sp.symbols('zb_{}'.format(str(i)+str(j))))


        # replace all instances of qa_i*qa_j with za_ij and extra constraint // same for b's:
        M = sp.symbols('M') # penalty constant

        replacements_a = [(combo_a[i][0]*combo_a[i][1], z_a[i]) for i in range(len(z_a))]

        penalties_a = [M*(combo_a[i][0]*combo_a[i][1] - 2*combo_a[i][0]*z_a[i] - 2*combo_a[i][1]*z_a[i] + 3*z_a[i])
                       for i in range(len(z_a))]

        replacements_b = [(combo_b[i][0]*combo_b[i][1], z_b[i]) for i in range(len(z_b))]

        penalties_b = [M*(combo_b[i][0]*combo_b[i][1] - 2*combo_b[i][0]*z_b[i] - 2*combo_b[i][1]*z_b[i] + 3*z_b[i])
                       for i in range(len(z_b))]

        penalties = sp.Add(*[penalties_a[j]+penalties_b[j] for j in range(len(penalties_a))])


        # do the substitutions:
        for i in range(len(qubits_a)):
            poly = poly.subs(replacements_a)
            poly = poly.subs(replacements_b)


        # add penalties to main polynomial:
        new_poly = poly + penalties
    
        return new_poly

