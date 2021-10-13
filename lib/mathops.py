"""
Library functions to perform circular convolution operations.
"""

__author__ = "Ashwinkumar Ganesan, Sunil Gandhi, Hang Gao"
__email__ = "gashwin1@umbc.edu,sunilga1@umbc.edu,hanggao@umbc.edu"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Pytorch functions.
"""
def complex_multiplication(left, right):
    """
    Multiply two vectors in complex domain.
    """
    left_real, left_complex = left[..., 0], left[..., 1]
    right_real, right_complex = right[..., 0], right[..., 1]

    output_real = left_real * right_real - left_complex * right_complex
    output_complex = left_real * right_complex + left_complex * right_real
    return torch.stack([output_real, output_complex], dim=-1)

def complex_division(left, right):
    """
    Divide two vectors in complex domain.
    """
    left_real, left_complex = left[..., 0], left[..., 1]
    right_real, right_complex = right[..., 0], right[..., 1]

    output_real = torch.div((left_real * right_real + left_complex * right_complex),(right_real**2 + right_complex**2))
    output_complex = torch.div((left_complex * right_real - left_real * right_complex ),(right_real**2 + right_complex**2))
    return torch.stack([output_real, output_complex], dim=-1)

def circular_conv(a, b):
    """ Defines the circular convolution operation
    a: tensor of shape (batch, D)
    b: tensor of shape (batch, D)
    """
    left = torch.rfft(a, 1, onesided=False)
    right = torch.rfft(b, 1, onesided=False)
    output = complex_multiplication(left, right)
    output = torch.irfft(output, 1, signal_sizes=a.shape[-1:], onesided=False)
    return output

def get_appx_inv(a):
    """
    Compute approximate inverse of vector a.
    """
    return torch.roll(torch.flip(a, dims=[-1]), 1,-1)

def get_inv(a, typ=torch.DoubleTensor):
    """
    Compute exact inverse of vector a.
    """
    left = torch.rfft(a, 1, onesided=False)
    complex_1 = np.zeros(left.shape)
    complex_1[...,0] = 1
    op = complex_division(typ(complex_1),left)
    return torch.irfft(op,1,onesided=False)

def complexMagProj(x):
    """
    Normalize a vector x in complex domain.
    """
    c = torch.rfft(x, 1, onesided=False)
    c_ish=c/torch.norm(c, dim=-1,keepdim=True)
    output = torch.irfft(c_ish, 1, signal_sizes=x.shape[1:], onesided=False)
    return output

def normalize(x):
    return x/torch.norm(x)

"""
Numpy Functions.
"""
# Make them work with batch dimensions
def cc(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b))

def np_inv(a):
    return np.fft.irfft((1.0/np.fft.rfft(a)),n=a.shape[-1])

def np_appx_inv(a):
    #Faster implementation
    return np.roll(np.flip(a, axis=-1), 1,-1)

def npcomplexMagProj(x):
    """
    Normalize a vector x in complex domain.
    """
    c = np.fft.rfft(x)

    # Look at real and image as if they were real
    c_ish = np.vstack([c.real, c.imag])

    # Normalize magnitude of each complex/real pair
    c_ish=c_ish/np.linalg.norm(c_ish, axis=0)
    c_proj = c_ish[0,:] + 1j * c_ish[1,:]
    return np.fft.irfft(c_proj,n=x.shape[-1])

def nrm(a):
    return a / np.linalg.norm(a)
