import numpy as np 
import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
enum_path = file_dir + "/../scripts/swig/cauchy/enumeration"
sys.path.append(enum_path)
import enu as cell_enum

ZERO_EPS = 1e-10

def make_child(A, p, b, t):
    #
    m, n = A.shape

    at = A[t]
    for i in range()
    return Ac, pc, bc, cmap 


I = np.eye(3)
ones3 = np.ones(3)
A = np.array([ I[1] - I[0], I[2] - I[0], I[2]-I[3] ])
p = np.array([.2,.3,.4])
b = np.array([1,2,3])
Phi = np.array([[1.4, -0.6, -1.0], 
                [-0.2,  1.0,  0.5],  
                [0.6, -0.6, -0.2]] )
B = cell_enum.inc_enu(A)
G = np.arange(1,B.shape[0]+1) + 1j * np.arange(1,B.shape[0]+1)
Gamma = np.array([[.1, 0.3, -0.2]]).T 
HPs_dynam = np.array([Gamma.T, Gamma.T @ Phi.T])
beta_dyn = 0.2
eta_term = 0.3
betas = beta_dyn * np.ones(HPs_dynam.shape[0])
etas = eta_term * ones3
Abar = np.vstack((A,HPs_dynam,I))
pbar = np.concatenate((p,))
