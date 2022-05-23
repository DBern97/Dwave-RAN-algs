
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








class PyQUBO_Helpers:
    def __init__(self, num_bin_vars, nq_per_var):
        
        self.num_bin = num_bin_vars
        self.nq = nq_per_var
    
    def Bin_Approx(self, domain, nq):
        """ 
        Create a binary (similar to radix-2) approximation of continuous
        variable in some 'domain' with 'nq' qubits.
        """

        low, high = domain

        if low < 0:
            nq = nq - 2

            # Largest power and the offset
            p_max = np.floor(np.log2(high) - 1)
            sub = int(nq/2) - p_max

            # Powers of the binary vars
            powers = np.linspace(-int(nq/2), int(nq/2), nq+1, dtype=int)
            powers = powers - sub

            # array of coefficients i.e. 2**powers
            coeff = 2*np.ones(nq+1)
            coeff = np.power(coeff, powers)

            # upper = np.sum(coeff)
            c = low

            # now coeff is an array of coeffs for a single element of x.
            coeff = np.append(coeff,c)
        
        elif low == 0:
            p_max = np.floor(np.log2(high) - 1)

            # now, we just need powers of two from p_max - 1 
            coeff = np.zeros(nq)
            for i in range(nq):
                coeff[i] = 2**(p_max - i)
            
            coeff = np.array(coeff)
            # upper = np.sum(coeff)
            
        # if necessary to find a way for cases like domain=[2,8] (needs offset)
        # elif low > 0:
        # c = -(upper - low)

        # bin_app = pq.Array.create(str(name), shape=(nq,), vartype='BINARY')
        # bin_app = bin_app.dot(coeffs)
        
        return coeff

    def Create_X(self, domain):

        qubits = pq.Array.create('q', shape=(self.num_bin*self.nq, ), vartype='BINARY')

        xtilde = np.zeros(self.num_bin, dtype='object')

        for i in range(self.num_bin):
            xtilde[i] = qubits[i*self.nq : self.nq*(i+1),].dot(self.Bin_Approx(domain, self.nq))
        
        xtilde = pq.Array(xtilde)

        return xtilde
    
    def L2(self, xtilde):
        return xtilde.dot(xtilde)
    
    def EVM(self, x, s):
        """
        Given a symbolic vector 'x' (xtilde!) and message vector 's' (complex), return the
        L2-norm corresponding to the EVM constraint.

        Uses H and n from within the class.
        """
        # s = self.s
        no_users = int(len(s))
        no_transmit = int(len(x)/2)

        s_im = s.imag
        s_re = s.real

        s = pq.Array(np.concatenate((s_re,s_im)))

        H = sp.Matrix(Channel_Rayleigh(no_users, no_transmit))

        T = sp.ones(2)
        T[0,1] = -1
        ID = sp.eye(2)
        T = T-ID
    #     print(T)

        H_re = sp.re(H)
        H_im = sp.im(H)
    #     print(H_im)

    #     H = sp.Matrix(TensorProduct(T,H))
        H = np.array(TensorProduct(T, H_im)) + np.array(TensorProduct(ID, H_re))
    #     H = sp.Matrix(sp.BlockMatrix([H_re, -H_im, H_im, H_re]))
        H = pq.Array(H)

        n = Gauss_noise(no_users)
        n = pq.Array(np.concatenate((n.real,n.imag)))

        EVM_pen = H.matmul(x) + n - s
        EVM = EVM_pen.T.matmul(EVM_pen)

        return EVM

    def Max_Norm_LP(self, xtilde, gamma):
        
        size = int(len(xtilde)/2)

        F = np.around(dft(size, scale='sqrtn'), decimals=4) #TODO: handle accuracy as a parameter
        F = F.T.conj()
        F = sp.Matrix(F)

        # Create re/im block matrix
        F_re = sp.re(F)
        F_im = sp.im(F)

        T = sp.ones(2)
        T[0,1] = -1
        ID = sp.eye(2)
        T = T - ID

        F = sp.Matrix(TensorProduct(T, F_im)) + sp.Matrix(TensorProduct(ID, F_re))
        F = np.array(F, dtype=float)
        F = pq.Array(F)

        ytilde = F.matmul(xtilde)

        muarr = pq.Array.create('muq', shape=(self.nq, ), vartype='BINARY')
        muarr = muarr.dot(self.Bin_Approx([0,4], self.nq))

        karr = pq.Array.create('kq', shape=(self.nq, ), vartype='BINARY')
        karr = karr.dot(self.Bin_Approx([0,4], self.nq))

        term = np.zeros(len(ytilde), dtype=object)
        for i in range(len(ytilde)):
            term[i] = (ytilde[i]**2 - muarr + karr)**2

        term = pq.Array(term)
        max_norm = muarr + gamma*sum(term)

        return max_norm


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

def Embedder(problem):
    return 0

###### Radio/telco -related functions:
def Channel_Rayleigh(no_users, no_transmit, scale=None):
    """
    Returns a random, complex Rayleigh fading wireles channel 
    matrix of size (no_users, no_transmit). 
    
    'no_users = NK and no_transmit = NM', but N can really be set to 1.'
        
    NOTE: For testing, specify seed globally when calling this.
    """
    if scale is None:
        scale = 1
    else:
        None
    
    re_H = np.random.rayleigh(scale, (no_users, no_transmit))
    im_H = np.random.rayleigh(scale, (no_users, no_transmit))
    
    H = re_H + im_H*1J
    
    return H

def Gauss_noise(length, scale=None):
    """
    Returns a random, complex vector where elemnts are drawn from a Gaussian distribution. 
        
    NOTE: For testing, specify seed globally when calling this.
    """

    if scale is None:
        scale=0.1
    else:
        None

    noise_re = np.random.normal(loc = 0, scale = np.sqrt(scale/2), size = (length,))
    noise_im = np.random.normal(loc = 0, scale = np.sqrt(scale/2), size = (length,))

    noise = noise_re + noise_im * 1j

    return noise


if __name__ == "__main__":
    
    trial = PyQUBO_Helpers(4,4)
    x = trial.Create_X([-2,2])
    print(len(x), '\n')
    s = np.array([1-1J, 3+1J], dtype=complex)