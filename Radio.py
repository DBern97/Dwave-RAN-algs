"""
A collection of functions helpful in modelling a MIMO system
"""

import numpy as np
import random



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

def Gauss_Noise(length, scale=None):
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



def Create_S(length, QAM_size):
    """
    Creates an 'length'-long array of QAM constellation symbols within
    'QAM_size'-QAM.
    """

    scope = np.log2(QAM_size)
    real = np.array([random.randrange(-(scope-1), scope, 2) for p in range(0,length)])
    im = np.array([random.randrange(-(scope-1), scope, 2) for p in range(0,length)])

    return real + im*1J



##TODO:
def Bit_to_QAM(bit_string, QAM_size):
    QAM_symbols = []

    mapping_table = {
                    (0,0,0,0) : -3-3j,
                    (0,0,0,1) : -3-1j,
                    (0,0,1,0) : -3+3j,
                    (0,0,1,1) : -3+1j,
                    (0,1,0,0) : -1-3j,
                    (0,1,0,1) : -1-1j,
                    (0,1,1,0) : -1+3j,
                    (0,1,1,1) : -1+1j,
                    (1,0,0,0) :  3-3j,
                    (1,0,0,1) :  3-1j,
                    (1,0,1,0) :  3+3j,
                    (1,0,1,1) :  3+1j,
                    (1,1,0,0) :  1-3j,
                    (1,1,0,1) :  1-1j,
                    (1,1,1,0) :  1+3j,
                    (1,1,1,1) :  1+1j
                    }

    return QAM_symbols