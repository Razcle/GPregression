""" Different Kernels for GP Regression """
import numpy as np


def rbf(x, y, sigma=1.0, eps=1.0):
    """ Evaluates the radial basis function kernel at points x1 and x2.
    K(x, y) = eps * exp( - sigma*|x - y|^2)"""
    dist = np.dot((x - y), (x - y)) * sigma
    return eps * np.exp(-sigma * dist)


def periodic(x, y, eta=1.0, tau=5.0):
    print('warning this  wont work for data of greater than 1 dimension!')
    dist = 2 * np.sin(np.pi*(x - y)/tau)**2
    dist = dist/eta**2
    return np.exp(-dist)


def polynomial(x, y, degree=6):
    return (1 + np.dot(x, y))**degree
