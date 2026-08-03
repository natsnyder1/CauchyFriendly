from inv_cpdf_fullscript import *
import numpy as np
import math
pi = math.pi
comb = math.comb
copysign = math.copysign
from collections import Counter
import copy
import random
import matplotlib
import matplotlib.pyplot as plt
from inv_cpdf_fileimport import *


# Tests 
def test_2d():
    # constants
    n_dim = 2 # number of dimensions
    m_i = 2 # number of hyperplanes 
    alpha_1 = 0.1
    alpha_2 = 0.2
    gamma = 0.3
    z1 = 1

    # term components
    # G1 = [
    #   [[1/(2*pi)*(1/(1j*z1+alpha_1+alpha_2+gamma)-1/(1j*z1-alpha_1+alpha_2+gamma))]],
    #   [[1/(2*pi)*(1/(1j*z1+alpha_1+alpha_2-gamma)-1/(1j*z1-alpha_1+alpha_2-gamma))]],
    #   [[1/(2*pi)*(1/(1j*z1+alpha_1-alpha_2+gamma)-1/(1j*z1-alpha_1-alpha_2+gamma))]],
    #   [[1/(2*pi)*(1/(1j*z1+alpha_1-alpha_2-gamma)-1/(1j*z1-alpha_1-alpha_2-gamma))]] 
    #   ]
    G1 = [
      [[1/(2*pi)*(1/(1j*z1+alpha_1+gamma+alpha_2)-1/(1j*z1-alpha_1+gamma+alpha_2))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1+gamma-alpha_2)-1/(1j*z1-alpha_1+gamma-alpha_2))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1-gamma+alpha_2)-1/(1j*z1-alpha_1-gamma+alpha_2))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1-gamma-alpha_2)-1/(1j*z1-alpha_1-gamma-alpha_2))]] 
      ]
    A1 = np.array([[-1,0], [-1, 1]])
    b1 = [z1,0]
    p1 = [gamma,alpha_2]
    Q1 = np.identity(2)

    # child 2
    G2 = [
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1+gamma)-1/(1j*z1-alpha_2+alpha_1+gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1-gamma)-1/(1j*z1-alpha_2+alpha_1-gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1+gamma)-1/(1j*z1-alpha_2-alpha_1+gamma))]],
        [[ 1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1-gamma)-1/(1j*z1-alpha_2-alpha_1-gamma))]]
        ]
    A2 = np.array([[1,-1], [0, -1]])
    b2 = [0,z1]
    p2 = [alpha_1,gamma]
    Q2 = np.identity(2)

    G3 = [
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1+alpha_2)-1/(1j*z1-gamma+alpha_1+alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1-alpha_2)-1/(1j*z1-gamma+alpha_1-alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma-alpha_1+alpha_2)-1/(1j*z1-gamma-alpha_1+alpha_2))]],
        [[ 1/(2*pi)*(1/(1j*z1+gamma-alpha_1-alpha_2)-1/(1j*z1-gamma-alpha_1-alpha_2))]]
            ]
    A3 = np.array([[1,0], [0, 1]])
    b3 = [0,0]
    p3 = [alpha_1,alpha_2]
    Q3 = np.identity(2)

    B = [0,1,2,3]

    # # define terms
    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B,enumeration_G=convertConstGToBundle(G1,2) )    
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B,enumeration_G=convertConstGToBundle(G2,2))
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q3,expnt_p = p3,enumeration_B=B,enumeration_G=convertConstGToBundle(G3,2))
    
    # define terms
    # term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B,enumeration_G=G1 )    
    # term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B,enumeration_G=G2)
    # term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q3,expnt_p = p3,enumeration_B=B,enumeration_G=G3)
    
    resultingint = findSymbolic_UCPDF([term1])

    resultingint = findSymbolic_UCPDF([term1,term2,term3])
    #resultingint2 = findSymbolicMarginal_UCPDF([term1,term2,term3],[0,1])
    #print(resultingint)
    fz = calcfz([term1,term2,term3])
    fz,x_hat,P = calc_fz_moment([term1,term2,term3])



    try1 = findSymbolic_UCPDF([term1])
    #try1b = findSymbolicMarginal_UCPDF([term1],[0,1])
    try2 = findSymbolic_UCPDF([term2])
    try3 = findSymbolic_UCPDF([term3])
    #print(f"first:{evaluateAtX(try1,[0.2,0.1],fz)}, second: {evaluateAtX(try2,[0.2,0.1],fz)}, third: {evaluateAtX(try3,[0.2,0.1],fz)} ")

    # print_UCPDF(resultingint)

    # print(evaluateAtX(resultingint,[0.2,0.1],fz))
    # plot2d_from_2d(try1,fz,orig_dim=0)
    plot2d_from_2d(try1,fz,orig_dim=2)
    plot2d_from_2d(try2,fz,orig_dim=2)
    plot2d_from_2d(try3,fz,orig_dim=2)
    plot2d_from_2d(resultingint,fz,orig_dim=2)
    # plot2d_from_2d(resultingint2,fz)


def test_2d_new():

    n_dim = 2
    m_i = 2

    z1 = 0.0338
    p0 = [0.1, 0.05]
    gamma = 0.2

    A1=np.array([ [-1,1], 
    [-1,0], ]) 
    p1=[ 0.05, 0.2 ]
    b1=[ 0.0338, 0 ]
    Enc_B1 = [ 0,  1] 
    G1 = [-0.620646 + 0.204233*1j, -1.73079 + 1.54525*1j]
    G1_manual =  [ 1/(2*pi) * (1/(1j*z1 + 0.1 + 0.05 + 0.2) - 1/(1j*z1 - 0.1 +0.05 +0.2)), 1/(2*pi) * (1/(1j*z1 + 0.1 + 0.05 - 0.2) - 1/(1j*z1 - 0.1 +0.05 -0.2))]

    A2=np.array([ [1,-1], 
    [0,-1], ]) 
    p2=[ 0.1, 0.2 ]
    b2=[ 0, 0.0338 ]
    Enc_B2 = [ 0,  1] 
    G2 = [-0.193845 + 0.0455222*1j, -1.30399 + 1.38654*1j]
    G2_manual =  [ 1/(2*pi) * (1/(1j*z1 + 0.05 + 0.1 + 0.2) - 1/(1j*z1 - 0.05 +0.1 +0.2)), 1/(2*pi) * (1/(1j*z1 + 0.05 + 0.1 - 0.2) - 1/(1j*z1 - 0.05 + 0.1 -0.2))]


    A3=np.array([ [1,0], 
    [0,1], ]) 
    p3=[ 0.1, 0.05 ]
    b3=[ 0, 0 ]
    Enc_B3 = [ 0,  1] 
    G3 = [2.92464 + 1.59077*1j, 1.81449 - 0.158711*1j]
    G3_manual=  [ 1/(2*pi) * (1/(1j*z1 + 0.2 + 0.1 + 0.05) - 1/(1j*z1 - 0.2 + 0.1 +0.05)), 1/(2*pi) * (1/(1j*z1 + 0.2 + 0.1 - 0.05 ) - 1/(1j*z1 - 0.2 + 0.1 - 0.05))]



    new_B1,new_G1 = expand_BG_tables(m_i,Enc_B1,G1)
    new_B2,new_G2 = expand_BG_tables(m_i,Enc_B2,G2)
    new_B3,new_G3 = expand_BG_tables(m_i,Enc_B3,G3)

    Q = np.identity(n_dim)

    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q,expnt_p = p1,enumeration_B=new_B1,enumeration_G=convertListGToTableG(new_G1,n_dim))  
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q,expnt_p = p2,enumeration_B=new_B2,enumeration_G=convertListGToTableG(new_G2,n_dim))  
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q,expnt_p = p3,enumeration_B=new_B3,enumeration_G=convertListGToTableG(new_G3,n_dim))  
    
    fz = calcfz([term1,term2,term3])
    print(fz)


def test_2d_diffH():
    n_dim = 2
    m_i = 2

    z1 = 0.0338
    p0 = [0.08, 0.1]
    gamma = 0.2
    H = [2, 1]
    
    a1 = p0[0]
    a2 = p0[1]
    h1 = H[0]
    h2 = H[1]


    scale = 4.324

    A1=np.array([ [-0.5,1], 
    [-0.5,0], ]) 
    p1=[ 0.1, 0.2 ]
    b1=[ 0.0169, 0 ]
    Enc_B1 = [ 0,  1] 
    G1 = [-1.06075 + 0.340072*1j, 3.80022 + 1.53439*1j]
    G1_manual = [(1/(z1*1j+a1*h1 + a2*h2 + gamma) - 1/(z1*1j - a1*h1 + a2*h2 + gamma)),(1/(z1*1j+a1*h1 + a2*h2 - gamma) - 1/(z1*1j - a1*h1 + a2*h2 -gamma)),(1/(z1*1j+a1*h1 - a2*h2 + gamma) - 1/(z1*1j - a1*h1 - a2*h2 + gamma)),(1/(z1*1j+a1*h1 - a2*h2 - gamma) - 1/(z1*1j - a1*h1 - a2*h2 - gamma))]

    A2=np.array([ [0.5,-1], 
    [0,-1], ]) 
    p2=[ 0.16, 0.2 ]
    b2=[ 0, 0.0338 ]
    Enc_B2 = [ 0,  1] 
    G2 = [-0.374611 + 0.0769602*1j, 4.48636 + 1.27128*1j]

    A3=np.array([ [0.5,0], 
    [0,1], ]) 
    p3=[ 0.16, 0.1 ]
    b3=[ 0, 0 ]
    Enc_B3 = [ 0,  1] 
    G3 = [-2.42561 + 1.61135*1j, 2.43536 - 0.263111*1j]


    new_B1,new_G1 = expand_BG_tables(m_i,Enc_B1,G1)
    new_B2,new_G2 = expand_BG_tables(m_i,Enc_B2,G2)
    new_B3,new_G3 = expand_BG_tables(m_i,Enc_B3,G3)

    Q = np.identity(n_dim)

    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q,expnt_p = p1,enumeration_B=new_B1,enumeration_G=convertListGToTableG(new_G1,n_dim))  
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q,expnt_p = p2,enumeration_B=new_B2,enumeration_G=convertListGToTableG(new_G2,n_dim))  
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q,expnt_p = p3,enumeration_B=new_B3,enumeration_G=convertListGToTableG(new_G3,n_dim))  
    
    fz = calcfz([term1,term2,term3])
    print(fz)

def test_4d():
    n_dim = 4 # number of dimensions
    m_i = 4 # number of hyperplanes

    """

    A1=np.array([ [0.479452,0,0.205479,0], 
    [0.731707,-1.21951,0.731707,-0], 
    [0.558659,-0.27933,0.111732,-0], 
    [0,0,0,-10], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p1=[ 0.292, 0.0656, 0.0895, 0.005, 0.027 ]
    b1=[ 0, 0, 0, 0 ]
    Enc_B1 = [ 9,  8,  11,  10,  13,  12,  15,  14,  1,  0,  3,  2,  4,  7,  6] 
    G1 = [0.130918 - 0.333187*1j, 0.351195 + 0.292181*1j, 0.273443 - 0.326724*1j, 0.198048 + 0.338787*1j, 0.163201 - 0.337845*1j, 0.312511 + 0.312487*1j, 0.312511 - 0.312487*1j, 0.163201 + 0.337845*1j, 0.608875 - 0.483871*1j, 0.781649 - 0.274526*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.706254 - 0.390984*1j]

    new sim result <
    Al=np.array([ [0.7,0,0.3,0], 
    [0.272727,-0.454545,0.272727,-0], 
    [0.588235,-0.294118,0.117647,-0], 
    [0,0,0,-1], 
    [0.1,0.3,-0.2,0.4], ]) 
    pl=[ 0.2, 0.176, 0.085, 0.05, 0.1 ]
    bl=[ 0, 0, 0, 0 ]
    Enc_Bl = [ 22,  9,  23,  8,  20,  11,  21,  10,  18,  13,  19,  12,  16,  15,  17,  14,  30,  1,  31,  0,  28,  3,  29,  2,  27,  4,  24,  7,  25,  6] 
    Gl = [0.130918 - 0.333187*1j, 0.351195 + 0.292181*1j, 0.273443 - 0.326724*1j, 0.198048 + 0.338787*1j, 0.163201 - 0.337845*1j, 0.312511 + 0.312487*1j, 0.312511 - 0.312487*1j, 0.163201 + 0.337845*1j, 0.163201 - 0.337845*1j, 0.312511 + 0.312487*1j, 0.312511 - 0.312487*1j, 0.163201 + 0.337845*1j, 0.198048 - 0.338787*1j, 0.273443 + 0.326724*1j, 0.351195 - 0.292181*1j, 0.130918 + 0.333187*1j, 0.608875 - 0.483871*1j, 0.781649 - 0.274526*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.706254 - 0.390984*1j, 0.829152 - 0.141498*1j, 0.829152 + 0.141498*1j, 0.706254 + 0.390984*1j, 0.809157 + 0.209592*1j, 0.659847 + 0.440739*1j, 0.809157 + 0.209592*1j, 0.659847 + 0.440739*1j]

    A=np.array([ [0.7,0,0.3,0], 
    [0.272727,-0.454545,0.272727,-0], 
    [0.588235,-0.294118,0.117647,-0], 
    [0,0,0,-1], 
    [0.1,0.3,-0.2,0.4], ]) 
    p=[ 0.2, 0.176, 0.085, 0.05, 0.1 ]
    b=[ 0, 0, 0, 0 ]
    Enc_B = [ 9,  8,  11,  10,  13,  12,  15,  14,  1,  0,  3,  2,  4,  7,  6] 
    G = [0.130918 - 0.333187*1j, 0.351195 + 0.292181*1j, 0.273443 - 0.326724*1j, 0.198048 + 0.338787*1j, 0.163201 - 0.337845*1j, 0.312511 + 0.312487*1j, 0.312511 - 0.312487*1j, 0.163201 + 0.337845*1j, 0.608875 - 0.483871*1j, 0.781649 - 0.274526*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.706254 - 0.390984*1j]
    >
    
    A2=np.array([ [0.612903,-0.645161,0.483871,-0], 
    [0.54755,-0.240154,0.12488,-0], 
    [0.729167,-0,0.3125,5.20833], 
    [0.479452,-0,0.205479,-0], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p2=[ 0.124, 0.1041, 0.0096, 0.146, 0.027 ]
    b2=[ -0.2541, 0, -0.1089, 0 ]
    Enc_B2 = [ 15,  13,  12,  11,  10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0] 
    G2 = [0.608875 - 0.483871*1j, 0.781649 - 0.274526*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.706254 - 0.390984*1j, 0.829152 - 0.141498*1j, 0.0163725 - 0.122822*1j, 0.189147 + 0.0865223*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0359991 - 0.133128*1j]

    A3= np.array([ [0.612903,-0.645161,0.483871,0], 
    [0.519836,-0.0683995,-0.0273598,-0], 
    [0.560748,-0.934579,0.560748,-2.33645], 
    [0.731707,-1.21951,0.731707,0], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p3=[ 0.62, 0.0731, 0.0214, 0.164, 0.027 ]
    b3=[ 0.4356, -0.726, 0.4356, 0 ]
    Enc_B3 = [ 2,  0,  1,  6,  7,  4,  5,  10,  11,  8,  9,  14,  15,  12,  13] 
    G3 = [0.0163725 - 0.122822*1j, 0.189147 + 0.0865223*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0359991 - 0.133128*1j, 0.158897 + 0.116359*1j, 0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j]

    A4=np.array([ [0.54755,-0.240154,0.12488,0], 
    [0.519836,-0.0683995,-0.0273598,0], 
    [0.529101,-0.26455,0.10582,-0.529101], 
    [0.558659,-0.27933,0.111732,0], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p4=[ 2.082, 0.2924, 0.0945, 0.895, 0.027 ]
    b4=[ 1.815, -0.9075, 0.363, 0 ]
    Enc_B4 = [ 15,  14,  13,  12,  11,  10,  9,  8,  7,  6,  5,  4,  2,  1,  0] 
    G4 = [0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j, 0.0333342 + 0.0331912*1j, 0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j]

    A5=np.array([ [0.729167,0,0.3125,5.20833], 
    [0.560748,-0.934579,0.560748,-2.33645], 
    [0.529101,-0.26455,0.10582,-0.529101], 
    [-0,-0,-0,-10], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 
    p5=[ 0.192, 0.0856, 0.0945, 0.05, 0.027 ]
    b5=[ 0, 0, 0, 1.815 ]
    Enc_B5 = [ 1,  0,  3,  2,  5,  4,  7,  6,  9,  8,  11,  10,  12,  15,  14] 
    G5 = [0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j, 0.0333342 + 0.0331912*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j, 0 + 0*1j]

    new_B1,new_G1 = expand_BG_tables(m_i,Enc_B1,G1)
    new_B2,new_G2 = expand_BG_tables(m_i,Enc_B2,G2)
    new_B3,new_G3 = expand_BG_tables(m_i,Enc_B3,G3)
    new_B4,new_G4 = expand_BG_tables(m_i,Enc_B4,G4)
    new_B5,new_G5 = expand_BG_tables(m_i,Enc_B5,G5)
    """
   
    A1=np.array([ [0.5,0,0,0], 
    [0,2,0,0], 
    [0,0,5,0], 
    [-0,-0,-0,-10], ]) 
    p1=[ 0.2, 0.04, 0.01, 0.01 ]
    b1=[ 0, 0, 0, 0 ]
    Enc_B1 = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15] 
    G1 = [0.130918 - 0.333187*1j, 0.351195 + 0.292181*1j, 0.273443 - 0.326724*1j, 0.198048 + 0.338787*1j, 0.163201 - 0.337845*1j, 0.312511 + 0.312487*1j, 0.312511 - 0.312487*1j, 0.163201 + 0.337845*1j, 0.163201 - 0.337845*1j, 0.312511 + 0.312487*1j, 0.312511 - 0.312487*1j, 0.163201 + 0.337845*1j, 0.198048 - 0.338787*1j, 0.273443 + 0.326724*1j, 0.351195 - 0.292181*1j, 0.130918 + 0.333187*1j]
    
    A2=np.array([ [-0.5,2,0,0], 
    [-0.5,0,5,0], 
    [-0.5,-0,-0,-10], 
    [-0.5,0,0,0], ]) 
    p2=[ 0.04, 0.01, 0.01, 0.1 ]
    b2=[ -0.1815, 0, 0, 0 ]
    Enc_B2 = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15] 
    G2 = [0.608875 - 0.483871*1j, 0.781649 - 0.274526*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.659847 - 0.440739*1j, 0.809157 - 0.209592*1j, 0.706254 - 0.390984*1j, 0.829152 - 0.141498*1j, 0.829152 + 0.141498*1j, 0.706254 + 0.390984*1j, 0.809157 + 0.209592*1j, 0.659847 + 0.440739*1j, 0.809157 + 0.209592*1j, 0.659847 + 0.440739*1j, 0.781649 + 0.274526*1j, 0.608875 + 0.483871*1j]

    A3=np.array([ [0.5,-2,0,0], 
    [0,-2,5,0], 
    [-0,-2,-0,-10], 
    [0,-2,0,0], ]) 
    p3=[ 0.2, 0.01, 0.01, 0.1 ]
    b3=[ 0, -0.726, 0, 0 ]
    Enc_B3 = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15] 
    G3 = [0.0163725 - 0.122822*1j, 0.189147 + 0.0865223*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0255381 - 0.128253*1j, 0.174848 + 0.102894*1j, 0.0359991 - 0.133128*1j, 0.158897 + 0.116359*1j, 0.158897 - 0.116359*1j, 0.0359991 + 0.133128*1j, 0.174848 - 0.102894*1j, 0.0255381 + 0.128253*1j, 0.174848 - 0.102894*1j, 0.0255381 + 0.128253*1j, 0.189147 - 0.0865223*1j, 0.0163725 + 0.122822*1j]

    A4=np.array([ [0.5,0,-5,0], 
    [0,2,-5,0], 
    [-0,-0,-5,-10], 
    [0,0,-5,0], ]) 
    p4=[ 0.2, 0.04, 0.01, 0.1 ]
    b4=[ 0, 0, -1.815, 0 ]
    Enc_B4 = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15] 
    G4 = [0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j, 0.0333342 + 0.0331912*1j, 0.0333342 - 0.0331912*1j, 0.0133398 + 0.0349029*1j, 0.0492855 - 0.0197268*1j, 0.00287885 + 0.0300281*1j, 0.0377259 - 0.0309699*1j, 0.0102176 + 0.0339645*1j, 0.0520245 - 0.0145977*1j, 0.00105207 + 0.0285336*1j]

    A5=np.array([ [0.5,0,0,10], 
    [0,2,0,10], 
    [0,0,5,10], 
    [0,0,0,10], ]) 
    p5=[ 0.2, 0.04, 0.01, 0.1 ]
    b5=[ 0, 0, 0, 3.63 ]
    Enc_B5 = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15] 
    G5 = [0.00105207 - 0.0285336*1j, 0.0520245 + 0.0145977*1j, 0.0102176 - 0.0339645*1j, 0.0377259 + 0.0309699*1j, 0.00287885 - 0.0300281*1j, 0.0492855 + 0.0197268*1j, 0.0133398 - 0.0349029*1j, 0.0333342 + 0.0331912*1j, 0.0333342 - 0.0331912*1j, 0.0133398 + 0.0349029*1j, 0.0492855 - 0.0197268*1j, 0.00287885 + 0.0300281*1j, 0.0377259 - 0.0309699*1j, 0.0102176 + 0.0339645*1j, 0.0520245 - 0.0145977*1j, 0.00105207 + 0.0285336*1j]

    Q = np.identity(n_dim)

    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q,expnt_p = p1,enumeration_B=Enc_B1,enumeration_G=convertListGToTableG(G1,n_dim))  
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q,expnt_p = p2,enumeration_B=Enc_B2,enumeration_G=convertListGToTableG(G2,n_dim))  
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q,expnt_p = p3,enumeration_B=Enc_B3,enumeration_G=convertListGToTableG(G3,n_dim))  
    term4 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A4,expnt_b=b4,expnt_Q = Q,expnt_p = p4,enumeration_B=Enc_B4,enumeration_G=convertListGToTableG(G4,n_dim))  
    term5 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A5,expnt_b=b5,expnt_Q = Q,expnt_p = p5,enumeration_B=Enc_B5,enumeration_G=convertListGToTableG(G5,n_dim))  

    fz = calcfz([term1,term2,term3,term4,term5])

    #pdf = findSymbolic_UCPDF([term1,term2,term3,term4,term5])

    print(fz)

    #plot2d_from_4d(pdf,fz)



def test_coalignment():
    A = np.array([[1,1],[1,1],[1,-1]])
    b1 = [0.3,0]
    Q1 = np.identity(2)
    p = [0.1,0.2,0.3]
    G = [[['a']],[['b']],[['g']],[['h']]]
    B = [0,1,6,7]
    term1 = Term(parent=0,ndim=2,m_hyperplanes=3,A_hplane_arr=A,expnt_b=b1,expnt_Q = Q1,expnt_p = p,enumeration_B=B,enumeration_G=G )    

    coaligned_ind,p_adjusted = term1.coalignmentCheck()
    newTerm = term1.coalignmentAdjustment(coaligned_ind,p_adjusted)

    print(newTerm.enumeration_G)
    print(newTerm.A_hplane_arr)

def test_evalAtX():
    bundle1 = Bundle(1,3,[1,1])
    bundle2 = Bundle(-15,1,[0,0])
    bundle3 = Bundle(-1,3,[1,2])
    bundle4 = Bundle(1,1,[-1,1])

    pdf = [[bundle1,bundle2],[bundle3,bundle4]]
    evaluateAtX(pdf,[1,2])

def test_1D():
    Phi = 1
    Gam = 1
    H = 1
    
    alpha1 = 0.1
    gamma = 0.1
    beta = 0.1
    z1 = 0.1

    A1 = np.array([[-Phi],[Gam]])
    p1 = [gamma, beta]
    b1 = [Phi*z1];  
    Q1 = np.identity(1)
    B1 = [0,1,2,3]
    G1 = [
        [[1/(2*pi) *(1/(1j*z1+alpha1+gamma) - 1/(1j*z1-alpha1+gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1+gamma) - 1/(1j*z1-alpha1+gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1-gamma) - 1/(1j*z1-alpha1-gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1-gamma) - 1/(1j*z1-alpha1-gamma))]]
        ]
    
    A2 = np.array([[Phi],[Gam]])
    p2 = [alpha1,beta]
    b2 = [0]
    Q2 = np.identity(1)
    B2 = [0,1,2,3]
    G2 = [
        [[1/(2*pi) *(1/(1j*z1+gamma+alpha1) - 1/(1j*z1-gamma+alpha1))]],
        [[1/(2*pi) *(1/(1j*z1+gamma+alpha1) - 1/(1j*z1-gamma+alpha1))]],
        [[1/(2*pi) *(1/(1j*z1+gamma-alpha1) - 1/(1j*z1-gamma-alpha1))]],
        [[1/(2*pi) *(1/(1j*z1+gamma-alpha1) - 1/(1j*z1-gamma-alpha1))]]
        ]
    
    n_dim = 1
    m_i = 2


    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B1,enumeration_G=G1 )    
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B2,enumeration_G=G2)

    resultingint = findSymbolic_UCPDF([term1,term2])
    #print(resultingint)

    print_UCPDF(resultingint)

    # for i in range(len(resultingint)):
    #     print(f"product {i}")
    #     for j in range(len(resultingint[i])):
    #         print(f"num: {resultingint[i][j].const_num}, const: {resultingint[i][j].const_den}, coeff: {resultingint[i][j].coeff_den}")
    
def test_fz_1D():
    
    Phi = 1
    Gam = 1
    H = 1
    
    alpha1 = 0.1
    gamma = 0.1
    beta = 0.1
    z1 = 0.1

    n_dim = 1
    m_i = 2
    A1 = np.array([[-Phi],[Gam]])
    p1 = [gamma, beta]
    b1 = [Phi*z1];  
    Q1 = np.identity(1)
    B1 = [0,1,2,3]
    G1 = [
        [[1/(2*pi) *(1/(1j*z1+alpha1+gamma) - 1/(1j*z1-alpha1+gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1+gamma) - 1/(1j*z1-alpha1+gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1-gamma) - 1/(1j*z1-alpha1-gamma))]],
        [[1/(2*pi) *(1/(1j*z1+alpha1-gamma) - 1/(1j*z1-alpha1-gamma))]]
        ]

    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B1,enumeration_G=convertConstGToBundle(G1,2) )    
    
    print(calcfz([term1]))
    print(G1[0][3])


def test_fz_2d():
    # constants
    n_dim = 2 # number of dimensions
    m_i = 2 # number of hyperplanes 
    alpha_1 = 0.1
    alpha_2 = 0.2
    gamma = 0.3
    z1 = 0.1

    # term components
    G1 = [
      [[1/(2*pi)*(1/(1j*z1+alpha_1+alpha_2+gamma)-1/(1j*z1-alpha_1+alpha_2+gamma))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1+alpha_2-gamma)-1/(1j*z1-alpha_1+alpha_2-gamma))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1-alpha_2+gamma)-1/(1j*z1-alpha_1-alpha_2+gamma))]],
      [[1/(2*pi)*(1/(1j*z1+alpha_1-alpha_2-gamma)-1/(1j*z1-alpha_1-alpha_2-gamma))]] 
      ]
    A1 = np.array([[-1,0], [-1, 1]])
    b1 = [z1,0]
    p1 = [gamma,alpha_2]
    Q1 = np.identity(2)

    # child 2
    G2 = [
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1+gamma)-1/(1j*z1-alpha_2+alpha_1+gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1-gamma)-1/(1j*z1-alpha_2+alpha_1-gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1+gamma)-1/(1j*z1-alpha_2-alpha_1+gamma))]],
        [[ 1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1-gamma)-1/(1j*z1-alpha_2-alpha_1-gamma))]]
        ]
    A2 = np.array([[1,-1], [0, -1]])
    b2 = [0,z1]
    p2 = [alpha_1,gamma]
    Q2 = np.identity(2)

    G3 = [
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1+alpha_2)-1/(1j*z1-gamma+alpha_1+alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1-alpha_2)-1/(1j*z1-gamma+alpha_1-alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma-alpha_1+alpha_2)-1/(1j*z1-gamma-alpha_1+alpha_2))]],
        [[ 1/(2*pi)*(1/(1j*z1+gamma-alpha_1-alpha_2)-1/(1j*z1-gamma-alpha_1-alpha_2))]]
            ]
    A3 = np.array([[1,0], [0, 1]])
    b3 = [0,0]
    p3 = [alpha_1,alpha_2]
    Q3 = np.identity(2)

    B = [0,1,2,3]

    # define terms
    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B,enumeration_G=convertConstGToBundle(G1,2) )    
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B,enumeration_G=convertConstGToBundle(G2,2))
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q3,expnt_p = p3,enumeration_B=B,enumeration_G=convertConstGToBundle(G3,2))
    
    print(calcfz([term1,term2,term3]))

def test_enumerating():
    A1=np.array([ [0.479452,0,0.205479,0], 
    [0.731707,-1.21951,0.731707,-0], 
    [0.558659,-0.27933,0.111732,-0], 
    [0,0,0,-10], 
    [0.37037,1.11111,-0.740741,1.48148], ]) 

    num_tries = 100000
    collector = []
    #rand_x = np.array([-0.38,-0.6,-0.59,-0.66]).reshape(4,1)
    for pertry in range(0,num_tries):
        rand_x = np.array([random.uniform(-1,1) for i in range(4)]).reshape(4,1)
        sign_seq =  np.copysign(1,np.matmul(A1,rand_x)).reshape(5)
        sign_bin_list = ['0' if sign==1 else '1' for sign in sign_seq]
        sign_bin = ''.join(sign_bin_list)
        #print(f" vector = {rand_x.reshape(4)}, sign_bin = {sign_bin}, int = {int(sign_bin,2)}, sign_bin_short = {sign_bin[0:-1]}, int = {int(sign_bin[0:-1],2)}") 

        if int(sign_bin,2) not in collector:
            collector.append(int(sign_bin,2))
        
    print(collector)

def test_indices_rearr():
    MU1_terms = splitMUtxt("MU1.txt")
    priority_indicies = [0,2]
    rearrangeIndices(MU1_terms[0:2],priority_indicies)


def test_compare():
    file1 = "nainaMU1.npy"
    file2 = "natMU1.npy"

    arr1 = np.load(file1)
    arr2 = np.load(file2)

    print(arr1-arr2)



def test_4d_fz():
    
    if True: 
        MU1_terms = splitMUtxt("MU1.txt")
        print(f"number of terms after MU1: {len(MU1_terms)}")
        MU1_term = MU1_terms[3:4]
        fz1=calcfz(MU1_terms)
        mult_d = True
        if mult_d:
            ucpdf1 = findSymbolicMarginal_UCPDF(MU1_terms,[0,1])
            ucpdf11 = findSymbolicMarginal_UCPDF_eti(MU1_terms,[0,1])
            plot2d_from_2d(ucpdf1,fz1,orig_dim=4)
            plot2d_from_2d(ucpdf11,fz1,orig_dim=2)
        else:
            ucpdf1_1d_x1 = findSymbolicMarginal_UCPDF_1d(MU1_terms,[0,1])
            ucpdf1_1d_x0 = findSymbolicMarginal_UCPDF_1d(MU1_terms,[1,0])
            plot1d_from_2d(ucpdf1_1d_x1,fz1,4)
            plot1d_from_2d(ucpdf1_1d_x0,fz1,4)

        print("now start plotting MU1")
        # plot2d_from_2d(ucpdf1,fz1,orig_dim=4)
        # plot2d_from_2d(ucpdf11,fz1,orig_dim=2)
        # plot1d_from_2d(ucpdf1_1d_x1,fz1,4)
        # plot1d_from_2d(ucpdf1_1d_x0,fz1,4)


    if True:
        MU2_terms = splitMUtxt("MU2.txt")
        print(f"number of terms after MU2: {len(MU2_terms)}")
        fz2=calcfz(MU2_terms)
        #ucpdf2 = findSymbolic_UCPDF(MU2_terms)
        MU2_term = MU2_terms[0:1]
        ucpdf2 = findSymbolicMarginal_UCPDF(MU2_terms,[0,1])
        ucpdf22 = findSymbolicMarginal_UCPDF_eti(MU2_term,[0,1])
        ucpdf2_1d = findSymbolicMarginal_UCPDF_1d(MU2_terms,[0,1])
        print("now start plotting MU1")
        plot2d_from_2d(ucpdf2,fz2,orig_dim=4)
        plot2d_from_2d(ucpdf22,fz2,orig_dim=2)
        plot1d_from_2d(ucpdf2_1d,fz2,4)


    if False:
        MU3_terms = splitMUtxt("MU3.txt")
        print(f"number of terms after MU3: {len(MU3_terms)}")
        fz3 = calcfz(MU3_terms)
        MU3_term = MU3_terms[-3:-2]
        ucpdf3 = findSymbolicMarginal_UCPDF(MU3_terms,[0,1])
        print("now start plotting MU3")
        plot2d_from_2d(ucpdf3,fz3,orig_dim=4)

    if False:
        MU4_terms = splitMUtxt("MU4.txt")
        print(f"number of terms after MU3: {len(MU4_terms)}")
        fz4 = calcfz(MU4_terms)
        MU4_term = MU4_terms[-3:-2]
        ucpdf4 = findSymbolicMarginal_UCPDF(MU4_terms,[0,1])
        print("now start plotting MU4")
        plot2d_from_2d(ucpdf4,fz4,orig_dim=4)

    if False:
        MU5_terms = splitMUtxt("MU5.txt")
        print(f"number of terms after MU5: {len(MU5_terms)}")
        fz5 = calcfz(MU5_terms)
        MU5_term = MU5_terms[-3:-2]
        ucpdf5 = findSymbolicMarginal_UCPDF(MU5_terms,[0,1])
        print("now start plotting MU5")
        plot2d_from_2d(ucpdf5,fz5,orig_dim=4)
    
    if False:
        MU6_terms = splitMUtxt("MU6.txt")
        print(f"number of terms after MU6: {len(MU6_terms)}")
        # fz6 = calcfz(MU6_terms)
        # MU6_term = MU6_terms[-3:-2]
        # ucpdf6 = findSymbolicMarginal_UCPDF(MU6_terms,[0,1])
        # print("now start plotting MU6")
        # plot2d_from_2d(ucpdf6,fz6,orig_dim=4)


if __name__ == "__main__":
    #test_1D()
    test_2d()
    #test_4d()
    #test_2d_new()
    #test_2d_diffH()
    #test_enumerating()

    # test_4d_fz()
    #test_indices_rearr()

    # test_compare()

