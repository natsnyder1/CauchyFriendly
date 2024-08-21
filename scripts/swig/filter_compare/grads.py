#!/usr/bin/env python3
import numpy as np

EP_GRAD = 5e-5 # the step taken in finding the gradient of function f at x: [f(x + 1.0*EP_GRAD) - f(x) ] / EP_GRAD

# returns Forward Difference Gradient of f, standard 1st order
def fd_gf(x, f):
    # numerical gradient 
    n = x.size
    zr = np.zeros(n)
    ep = EP_GRAD
    grad_f = np.zeros((n))
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        f2 = f(x + ep*ei)
        f1 = f(x)
        gfi = (f2 - f1) / ep
        grad_f[i] = gfi 
    return grad_f

# returns Forward Difference Double Gradient of f, the Hessian, standard 2nd order
def fd_dgf(x, f):
    # numerical double gradient
    n = x.size
    zr = np.zeros(n)
    ep = EP_GRAD
    dgrad_f = np.zeros((n,n))
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        for j in range(n):
            ej = zr.copy()
            ej[j] = 1.0 
            DGF = ( f(x + ep*ei + ep*ej) - f(x + ep*ej) - f(x + ep*ei) + f(x) ) / ep**2
            dgrad_f[i,j] = DGF
    return dgrad_f


# returns Central Difference Gradient of f, 2nd Order Expansion
def cd2_gf(x, f):
    # numerical gradient 
    n = x.size
    zr = np.zeros(n)
    ep = EP_GRAD
    grad_f = np.zeros((n))
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        f1 = f(x - ep*ei)
        f2 = f(x + ep*ei)
        gfi = (f2 - f1) / (2.0*ep)
        grad_f[i] = gfi 
    return grad_f 

# returns Central Difference Gradient of f, 4th Order Expansion
def cd4_gf(x, f):
    # numerical gradient 
    n = x.size
    zr = np.zeros(n)
    ep = EP_GRAD
    grad_f = np.zeros((n))
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        gfi = (-1.0 * f(x + 2.0*ep*ei) + 8.0*f(x + ep*ei) - 8.0 * f(x - ep*ei) + f(x - 2.0*ep*ei) ) / (12.0*ep)
        grad_f[i] = gfi 
    return grad_f


# returns Central Difference Double Gradient of f, the Hessian, 4th Order expansion
def cd4_dgf(x, f):
    # numerical double gradient
    n = x.size
    zr = np.zeros(n)
    ep = EP_GRAD
    dgrad_f = np.zeros((n,n))
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        for j in range(n):
            ej = zr.copy()
            ej[j] = 1.0 
            if( i == j):
                DGF = (-1.0*f(x + 2.0*ep*ei) + 16.0 * f(x + ep*ei) - 30.0 * f(x) + 16.0 * f(x - ep * ei) - f(x - 2.0*ep*ei) ) / (12.0 * ep**2)
            else:
                DGF = ( f(x + ep*ei + ep*ej) - f(x + ep*ei - ep*ej) - f(x - ep*ei + ep*ej) + f(x - ep*ei - ep*ej) ) / (4.0*ep**2)
            dgrad_f[i,j] = DGF
    return dgrad_f

# returns Central Difference Gradient of vector f, the matrix Jacobian, 4th Order expansion
def cd4_gvf(x, f):
    # numerical gradient 
    n = x.size
    m = f(x).size
    ep = EP_GRAD
    G = np.zeros((m,n))
    zr = np.zeros(n)
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        G[:,i] = (-1.0 * f(x + 2.0*ep*ei) + 8.0*f(x + ep*ei) - 8.0 * f(x - ep*ei) + f(x - 2.0*ep*ei) ) / (12.0*ep) 
    return G

# returns Central Difference Double Gradient of vector f, the sum of Hessians, 4th order expansion
def cd4_dgvf(x, f):
    # returns tensor of numerical double gradients
    n = x.size
    m = f(x).size
    zr = np.zeros(n)
    ep = EP_GRAD
    DGS = np.zeros((m,n,n))
    for i in range(n):
        ei = zr.copy()
        ei[i] = 1.0
        for j in range(n):
            ej = zr.copy()
            ej[j] = 1.0 
            if( i == j):
                DGF = (-1.0*f(x + 2.0*ep*ei) + 16.0 * f(x + ep*ei) - 30.0 * f(x) + 16.0 * f(x - ep * ei) - f(x - 2.0*ep*ei) ) / (12.0 * ep**2)
            else:
                DGF = ( f(x + ep*ei + ep*ej) - f(x + ep*ei - ep*ej) - f(x - ep*ei + ep*ej) + f(x - ep*ei - ep*ej) ) / (4.0*ep**2)
            DGS[:,i,j] = DGF
    return DGS



if __name__ == "__main__":
    foo_func = lambda x : 0.7 * x[0]**2 + 0.4*x[0]*x[1] + 0.3 * x[1]**2
    x = np.array([1.0, 2.0])
    GF = cd4_gf(x,foo_func)
    DGF = cd4_dgf(x,foo_func)
    print("Grad: ", GF, ",  Hessian: ", DGF)
