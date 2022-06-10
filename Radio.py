"""
A collection of functions helpful in modelling a MIMO system
"""

import numpy as np



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