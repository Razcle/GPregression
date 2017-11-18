""" Different Kernels for GP Regression """
import numpy as np


def rbf(x, y, sigma=1.0, eps=1.0):
    """ Evaluates the radial basis function kernel at points x1 and x2.
    K(x, y) = eps * exp( - sigma*|x - y|^2)"""
    dist = np.dot((x - y), (x - y)) * sigma
    return eps * np.exp(-sigma * dist)
