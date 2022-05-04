""" 
ABOUT: 
Class to generate the PAPR or EVM (or both) problem into a QUBO form.

Auxiliary variables for quadratisation only added from theory. TODO: automise this.

Author: David Bern 
"""


import numpy as np
import sympy as sp 
import scipy as sc
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

from sympy.physics.quantum import TensorProduct
import itertools 




class PAPR_Problem:
    def __init__(self, problem_size):

        self.no_users, self.no_transmit = np.array(problem_size, dtype=int)

        # self.len_x = something
        self.len_s = int(2*self.no_users)
        print('Expecting s to be {}-long'.format(self.len_s))
    
### Radio/telco -related functions:

    def Channel_Rayleigh(self, scale=None):
        """Returns a random, complex Rayleigh fading wireles channel 
            matrix of size (no_users, no_transmit). 
            
            'no_users = NK and no_transmit = NM', but N can really be set to 1.
            
            NOTE: For testing, specify seed globally when calling this.
        """
        if scale is None:
            scale = 1
        else:
            None
        
        re_H = np.random.rayleigh(scale, (self.no_users, self.no_transmit))
        im_H = np.random.rayleigh(scale, (self.no_users, self.no_transmit))
        
        H = re_H + im_H*1J
        
        return H


def Gauss_noise(self, length, scale=None):
    
    if scale is None:
        scale=0.1
    else:
        None
    
    noise_re = np.random.normal(loc = 0, scale = np.sqrt(scale/2), size = (length,))
    noise_im = np.random.normal(loc = 0, scale = np.sqrt(scale/2), size = (length,))
    
    noise = noise_re + noise_im * 1j
    
    return noise


### Helper functions to get symbolic expressions into QUBO format:

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

""" TODO: find and replace all high oredr terms with the min number of auxiliaries
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



### Symbolically building the problem:

def Bin_Approx_Symbolic(self, name, domain, nq):
    """ Create a binary (similar to radix-2) approximation of continuous 'name'
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


def L2_Sym(x):
    """ The L2-norm of the give symbolic (Matrix object) 'x'. """
    return x.T*x


def EVM_Sym(self, x, s):
    """
    Given a symbolic vector 'x' (xtilde!) and message vector 's' (complex), return the
    L2-norm corresponding to the EVM constraint.

    Uses H and n from within the class.
    """

    no_users = int(len(s))
    no_transmit = int(len(x)/2)
   
    s_im = s.imag
    s_re = s.real
    
    s = sp.Matrix(np.concatenate((s_re,s_im)))
    
    H = sp.Matrix(self.Channel_Rayleigh(no_users, no_transmit))
    
    T = sp.ones(2)
    T[0,1] = -1
    ID = sp.eye(2)
    T = T-ID
#     print(T)
    
    H_re = sp.re(H)
    H_im = sp.im(H)
#     print(H_im)
    
#     H = sp.Matrix(TensorProduct(T,H))
    H = sp.Matrix(TensorProduct(T, H_im)) + sp.Matrix(TensorProduct(ID, H_re))
#     H = sp.Matrix(sp.BlockMatrix([H_re, -H_im, H_im, H_re]))
    
    n = self.Gauss_noise(no_users)
    n = sp.Matrix(np.concatenate((n.real,n.imag)))
    
    EVM_pen = H*x + n - s
    EVM = EVM_pen.T*EVM_pen
    
    return EVM


def Max_Norm_LP_Constraint_Sym(self, x, mu):
    """
    Given symbolic vector 'x' (xtilde!) and 'mu', return the constraints from minimising the max-norm 
    as a Linear Program (LP). Returns a long expression (sum), \mu and \gamma to be added later.
    """
    
    size = int(len(x)/2)
    
    F = np.around(sc.linalg.dft(size), decimals=4) #TODO: handle accuracy as a parameter
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
    
    # Time array y:
    y = F*x
    
    # put all constraints in a list (sum them later)
    constraint = list(sp.symbols('C:{}'.format(size)))
    print(constraint)
    
    # TODO: too slow to handle like this
    for i in range(len(constraint)):
        constraint[i] = sp.Pow(y[i],2) + sp.Pow(y[i+size],2) - mu
        constraint[i] = sp.expand(constraint[i])
        constraint[i] = self.Quad2Lin(constraint[i])
        
        constraint[i] = sp.Pow(constraint[i],2)
        constraint[i] = sp.expand(constraint[i])
        constraint[i] = self.Quad2Lin(constraint[i])
    
    all_constraints = sp.Add(*[constraint[j] for j in range(len(constraint))])
    max_norm_min = mu + all_constraints
    
    return max_norm_min


def Get_Full_Problem(self):
    problem = []
    return problem 

### Getters/ Doers - to get L2, EVM or so, call a get_...
def Get_L2_Qubo(self):

    return 