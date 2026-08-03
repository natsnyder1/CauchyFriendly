import numpy as np
import math
pi = math.pi
from inv_cpdf_fileimport import *
from inv_cpdf_fullscript import *

def make_g_term(first_p,p1,p2,p3,sign_seq,z1):
    denom_sum = 0
    p_order = [first_p,p1,p2,p3]
    for el_i,el in enumerate(sign_seq):
        if el == "+":
            denom_sum += p_order[el_i]
        else:
            denom_sum -= p_order[el_i]
    return denom_sum + 1j*z1


def test_3d():
    # constants
    n_dim = 3 # number of dimensions
    m_i = 3 # number of hyperplanes 
    alpha_1 = 0.12
    alpha_2 = 0.23
    alpha_3 = 0.34
    gamma = 0.45
    z1 = 0.56

    G1 = [
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"++++",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"-+++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+++-",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"-++-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"++-+",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"-+-+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"++--",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"-+--",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+-++",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"--++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+-+-",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"--+-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+--+",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"---+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+---",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"----",z1))]],
      ]
    A1 = np.array([[-1,0,0], [-1, 1,0],[-1,0,1]])
    b1 = [z1,0,0]
    p1 = [gamma,alpha_2,alpha_3]
    Q1 = np.identity(3)
    
    G2 = [
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"++++",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"-+++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+++-",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"-++-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"++-+",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"-+-+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"++--",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"-+--",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+-++",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"--++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+-+-",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"--+-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+--+",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"---+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+---",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"----",z1))]],
      ]
    A2 = np.array([[1,-1,0], [0, -1,0],[0,-1,1]])
    b2 = [0,z1,0]
    p2 = [alpha_1,gamma,alpha_3]
    Q2 = np.identity(3)
    
    G3 = [
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"++++",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"-+++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+++-",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"-++-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"++-+",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"-+-+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"++--",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"-+--",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+-++",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"--++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+-+-",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"--+-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+--+",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"---+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+---",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"----",z1))]],
      ]
    A3 = np.array([[1,0,-1], [0, 1,-1],[0,0,-1]])
    b3 = [0,0,z1]
    p3 = [alpha_1,alpha_2,gamma]
    Q3 = np.identity(3)
    
    G4 = [
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"++++",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"-+++",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+++-",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"-++-",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"++-+",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"-+-+",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"++--",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"-+--",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+-++",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"--++",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+-+-",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"--+-",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+--+",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"---+",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+---",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"----",z1))]],
      ]
    A4 = np.array([[1,0,0], [0, 1,0],[0,0,1]])
    b4 = [0,0,0]
    p4 = [alpha_1,alpha_2,alpha_3]
    Q4 = np.identity(3)

    B = [0,1,2,3,4,5,6,7]

    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B,enumeration_G=convertConstGToBundle(G1,2) )    
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B,enumeration_G=convertConstGToBundle(G2,2))
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q3,expnt_p = p3,enumeration_B=B,enumeration_G=convertConstGToBundle(G3,2))
    term4 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A4,expnt_b=b4,expnt_Q = Q4,expnt_p = p4,enumeration_B=B,enumeration_G=convertConstGToBundle(G4,2))
    
    MU1_terms = [term1,term2,term3,term4]
    fz1=calcfz(MU1_terms)
    ucpdf1 = findSymbolicMarginal_UCPDF(MU1_terms,[1,2])
    fz1_ite,x_hat_ite,P_ite = calc_fz_moment(MU1_terms)

    x2 = 0.5
    x3 = 0.6
    ite_evaluated = evaluateAtX(ucpdf1,[0,x2,x3],fz1) * 2*pi

    print(f"integrate then evaluate: {np.round(ite_evaluated,3)}")
    # print_UCPDF(ucpdf1)

    #####

    n_dim = 2
    m_i = 2 # number of hyperplanes 

    A11 = np.array([[1,0],[0,1]])
    B11 = [0,1,2,3]
    # G11 = [G1[0],G1[1],G1[6],G1[7]]
    # G11 = [G1[0],G1[5],G1[2],G1[7]]
    # G11 = [G1[4],G1[5],G1[2],G1[3]]
    # G11 = [G1[4],G1[1],G1[6],G1[3]]
     

    # G11 = [G1[4],G1[5],G1[6],G1[7]]
    # G11 = [G1[0],G1[1],G1[2],G1[3]]

    p11 = [alpha_2,alpha_3]
    b11 = [0,0]
    Q11 = np.eye(2)

    A22 = np.array([[-1,0],[-1,0],[-1,1]])
    B22 = [0,1,6,7]
    G22 = [G2[0],G2[1],G2[6],G2[7]]
    p22 = [alpha_1,gamma,alpha_3]
    b22 = [z1,0]
    Q22 = np.eye(2)

    A33 = np.array([[0,-1],[1,-1],[0,-1]])
    B33 = [0,2,5,7]
    G33 = [G3[0],G3[2],G3[5],G3[7]]
    p33 = [alpha_1,alpha_2,gamma]
    b33 = [0,z1]
    Q33 = np.eye(2)

    A44 = np.array([[1,0],[0,1]])
    B44 = [0,1,2,3]
    # G44 = [G4[0],G4[1],G4[6],G4[7]]
    # G44 = [G4[0],G4[5],G4[2],G4[7]]
    # G44 = [G4[4],G4[5],G4[2],G4[3]]
    # G44 = [G4[4],G4[1],G4[6],G4[3]]
    
    # G44 =[G4[0],G4[1],G4[2],G4[3]]
    # G44 =[G4[4],G4[5],G4[6],G4[7]]

    p44 = [alpha_2,alpha_3]
    b44 = [0,0]
    Q44 = np.eye(2)

    G_opt = 31

    if G_opt ==11: 
        G11 = [G1[4],G1[5],G1[2],G1[3]] 
        G44 = [G4[0],G4[1],G4[6],G4[7]]
    elif G_opt ==12: 
        G11 = [G1[4],G1[1],G1[6],G1[3]]
        G44 = [G4[0],G4[5],G4[2],G4[7]]
    elif G_opt ==21: 
        G11 = [G1[0],G1[1],G1[6],G1[7]]
        G44 = [G4[4],G4[5],G4[2],G4[3]]
    elif G_opt ==22: 
        G11 = [G1[0],G1[5],G1[2],G1[7]]
        G44 = [G4[4],G4[1],G4[6],G4[3]]
    elif G_opt ==31: 
        G11 = [G1[0],G1[1],G1[2],G1[3]]
        G44 = [G4[4],G4[5],G4[6],G4[7]]
    elif G_opt ==32: 
        G11 = [G1[4],G1[5],G1[6],G1[7]]
        G44 = [G4[0],G4[1],G4[2],G4[3]]
    elif G_opt == 41:
        # DOES NOT WORK
        G11 = [G1[4],G1[5],G1[2],G1[3]]
        G44 = [G4[0],G4[1],G4[2],G4[3]]


    term11 = Term(parent=0,ndim=n_dim,m_hyperplanes=2,A_hplane_arr=A11,expnt_b=b11,expnt_Q = Q11,expnt_p = p11,enumeration_B=B11,enumeration_G=convertConstGToBundle(G11,2) )    
    term22 = Term(parent=0,ndim=n_dim,m_hyperplanes=3,A_hplane_arr=A22,expnt_b=b22,expnt_Q = Q22,expnt_p = p22,enumeration_B=B22,enumeration_G=convertConstGToBundle(G22,2))
    term33 = Term(parent=0,ndim=n_dim,m_hyperplanes=3,A_hplane_arr=A33,expnt_b=b33,expnt_Q = Q33,expnt_p = p33,enumeration_B=B33,enumeration_G=convertConstGToBundle(G33,2))
    term44 = Term(parent=0,ndim=n_dim,m_hyperplanes=2,A_hplane_arr=A44,expnt_b=b44,expnt_Q = Q44,expnt_p = p44,enumeration_B=B44,enumeration_G=convertConstGToBundle(G44,2))
    
    MU11_terms = [term11,term22,term33,term44]
    fz11=calcfz(MU11_terms)
    fz11_y,x_hat_eti,P_eti = calc_fz_moment(MU11_terms)
    ucpdf11 = findSymbolic_UCPDF(MU11_terms)
    eti_evaluated = evaluateAtX(ucpdf11,[x2,x3],fz11)
    print(f"evaluate then integrate: {np.round(eti_evaluated,3)}")
    # print_UCPDF(ucpdf11)
    print(f"fz from 3d: {np.round(fz1,3)}, fz from 2d: {np.round(fz11,3)}")

    print(f"x_hat from 2d: {np.round(x_hat_eti,3)}")
    print(f"P from 2d:\n{np.round(P_eti,3)}")


    # print(eti_evaluated/ite_evaluated)
    # print(fz1)
    # print(fz11)

def test_rotation():
    A1 = np.array([[-1,0,0], [-1, 1,0],[-1,0,1]])

    th = 0.001
    Rth = np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th), 0,np.cos(th)]])

    A1_rot = np.matmul(A1,Rth.T)

    A4 = np.array([[1,0,0], [0, 1,0],[0,0,1]])
    A4_rot = np.matmul(A4,Rth.T)

    pass

def test_3d_manual():


    alpha_1 = 0.1
    alpha_2 = 0.22
    alpha_3 = 0.3
    gamma = 0.45
    z1 = 0.56

    x2 = 0.12
    x3 = 0.33
    

    G1 = [
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"++++",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"-+++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+++-",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"-++-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"++-+",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"-+-+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"++--",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"-+--",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+-++",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"--++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+-+-",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"--+-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+--+",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"---+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"+---",z1)-1/make_g_term(alpha_1,gamma,alpha_2,alpha_3,"----",z1))]],
      ]
    
    G11 = [G1[4][0][0],G1[1][0][0],G1[6][0][0],G1[7][0][0]]
    
    G2 = [
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"++++",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"-+++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+++-",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"-++-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"++-+",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"-+-+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"++--",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"-+--",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+-++",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"--++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+-+-",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"--+-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+--+",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"---+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"+---",z1)-1/make_g_term(alpha_2,alpha_1,gamma,alpha_3,"----",z1))]],
      ]
    
    G22 = [G2[0][0][0],G2[1][0][0],G2[6][0][0],G2[7][0][0]]
    
    G3 = [
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"++++",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"-+++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+++-",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"-++-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"++-+",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"-+-+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"++--",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"-+--",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+-++",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"--++",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+-+-",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"--+-",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+--+",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"---+",z1))]],
      [[1/(2*pi)*(1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"+---",z1)-1/make_g_term(alpha_3,alpha_1,alpha_2,gamma,"----",z1))]],
      ]
    
    G33 = [G3[0][0][0],G3[2][0][0],G3[5][0][0],G3[7][0][0]]

    
    G4 = [
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"++++",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"-+++",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+++-",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"-++-",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"++-+",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"-+-+",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"++--",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"-+--",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+-++",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"--++",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+-+-",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"--+-",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+--+",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"---+",z1))]],
      [[1/(2*pi)*(1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"+---",z1)-1/make_g_term(gamma,alpha_1,alpha_2,alpha_3,"----",z1))]],
      ]
    
    G44 = [G4[0][0][0],G4[1][0][0],G4[6][0][0],G4[7][0][0]]

    marg_t1 = G11[3]/((alpha_2-1j*x3)*(alpha_3-1j*x3)) - G11[2]/((alpha_2-1j*x3)*(-alpha_3-1j*x3)) + G11[1]/((-alpha_2-1j*x3)*(alpha_3-1j*x3)) - G11[0]/((-alpha_2-1j*x3)*(-alpha_3-1j*x3))
    marg_t2 = G22[1]/((alpha_3-1j*x3)*(alpha_1+gamma+1j*(z1-x2-x3))) - G22[0]/((-alpha_3-1j*x3)*(alpha_1+gamma+1j*(z1-x2-x3))) + G22[3]/((alpha_3-1j*x3)*(-alpha_1-gamma+1j*(z1-x2-x3))) - G22[2]/((-alpha_3-1j*x3)*(-alpha_1-gamma+1j*(z1-x2-x3)))
    marg_t3 = G33[1]/((alpha_2-1j*x2)*(alpha_1+gamma+1j*(z1-x2-x3))) - G33[0]/((-alpha_2-1j*x2)*(alpha_1+gamma+1j*(z1-x2-x3))) + G33[3]/((alpha_2-1j*x2)*(-alpha_1-gamma+1j*(z1-x2-x3))) - G33[2]/((-alpha_2-1j*x2)*(-alpha_1-gamma+1j*(z1-x2-x3)))
    marg_t4 = G44[3]/((alpha_2-1j*x3)*(alpha_3-1j*x3)) - G44[2]/((alpha_2-1j*x3)*(-alpha_3-1j*x3)) + G44[1]/((-alpha_2-1j*x3)*(alpha_3-1j*x3)) - G44[0]/((-alpha_2-1j*x3)*(-alpha_3-1j*x3))

    print(marg_t1)
    print(marg_t2)
    print(marg_t3)
    print(marg_t4)

def test_2d():
    # constants
    n_dim = 2 # number of dimensions
    m_i = 2 # number of hyperplanes 
    alpha_1 = 0.1
    alpha_2 = 0.2
    gamma = 0.3
    z1 = 0.1

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

    
    x1 = 0.1
    x2 = 0.8

    term1 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A1,expnt_b=b1,expnt_Q = Q1,expnt_p = p1,enumeration_B=B,enumeration_G=convertConstGToBundle(G1,2) )    
    term2 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A2,expnt_b=b2,expnt_Q = Q2,expnt_p = p2,enumeration_B=B,enumeration_G=convertConstGToBundle(G2,2))
    term3 = Term(parent=0,ndim=n_dim,m_hyperplanes=m_i,A_hplane_arr=A3,expnt_b=b3,expnt_Q = Q3,expnt_p = p3,enumeration_B=B,enumeration_G=convertConstGToBundle(G3,2))

    MU1_terms = [term1,term2,term3]
    fz1=calcfz(MU1_terms)
    ucpdf1 = findSymbolic_UCPDF(MU1_terms)
    ite_evaluated = evaluateAtX(ucpdf1,[0,x2],fz1)

    print(f"integrate then evaluate: {np.round(ite_evaluated,3)}")

    #######

    marg_ite_p1_t1 = G1[3][0][0]/(alpha_2-1j*x2) - G1[2][0][0]/(-alpha_2-1j*x2)
    marg_ite_p1_t2 = G2[1][0][0]/(alpha_1-gamma+1j*(z1-x2)) - G2[3][0][0]/(-alpha_1-gamma+1j*(z1-x2)) + G2[0][0][0]/(alpha_1+gamma+1j*(z1-x2)) - G2[1][0][0]/(alpha_1-gamma+1j*(z1-x2))
    marg_ite_p1_t3 = G3[1][0][0]/(alpha_2-1j*x2) - G3[0][0][0]/(-alpha_2-1j*x2)

    marg_ite_p1 = marg_ite_p1_t1+marg_ite_p1_t2+marg_ite_p1_t3
    
    marg_ite_n1_t1 = G1[1][0][0]/(alpha_2-1j*x2) - G1[0][0][0]/(-alpha_2-1j*x2)
    marg_ite_n1_t2 = G2[0][0][0]/(alpha_1+gamma+1j*(z1-x2)) - G2[2][0][0]/(-alpha_1+gamma+1j*(z1-x2)) + G2[2][0][0]/(-alpha_1+gamma+1j*(z1-x2)) - G2[3][0][0]/(-alpha_1-gamma+1j*(z1-x2))
    marg_ite_n1_t3 = G3[3][0][0]/(alpha_2-1j*x2) - G3[2][0][0]/(-alpha_2-1j*x2)

    marg_ite_n1 = marg_ite_n1_t1+marg_ite_n1_t2+marg_ite_n1_t3

    print(f"integrate then evaluate, pos: {np.round(marg_ite_p1/(2*pi*fz1),3)}")
    print(f"integrate then evaluate, neg: {np.round(marg_ite_n1/(2*pi*fz1),3)}")

    #######

    marg_eti_p1_t1 = G1[1][0][0]/(alpha_2-1j*x2) - G1[2][0][0]/(-alpha_2-1j*x2)
    marg_eti_p1_t2 = G2[0][0][0]/(alpha_1+gamma+1j*(z1-x2)) - G2[3][0][0]/(-alpha_1-gamma+1j*(z1-x2))
    marg_eti_p1_t3 = G3[3][0][0]/(alpha_2-1j*x2) - G3[0][0][0]/(-alpha_2-1j*x2)

    marg_eti_p1 = marg_eti_p1_t1+marg_eti_p1_t2+marg_eti_p1_t3

    marg_eti_n1_t1 = G1[3][0][0]/(alpha_2-1j*x2) - G1[0][0][0]/(-alpha_2-1j*x2)
    marg_eti_n1_t2 = G2[0][0][0]/(alpha_1+gamma+1j*(z1-x2)) - G2[3][0][0]/(-alpha_1-gamma+1j*(z1-x2))
    marg_eti_n1_t3 = G3[1][0][0]/(alpha_2-1j*x2) - G3[2][0][0]/(-alpha_2-1j*x2)

    marg_eti_n1 = marg_eti_n1_t1+marg_eti_n1_t2+marg_eti_n1_t3

    print(f"evaluate then integrate, pos: {np.round(marg_eti_p1,3)}")
    print(f"evaluate then integrate, neg: {np.round(marg_eti_n1,3)}")


if __name__ == "__main__":
    test_3d()
    # test_3d_manual()
    # test_2d()

    # test_rotation()