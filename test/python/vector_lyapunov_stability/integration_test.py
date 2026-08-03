import numpy as np
import math
pi = math.pi


def test_2d():
    # constants
    n_dim = 2 # number of dimensions
    m_i = 2 # number of hyperplanes 
    alpha_1 = 0.1
    alpha_2 = 0.2
    gamma = 0.3
    z1 = 1
    
    b1 = 1
    b2 = 2

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
    #b1 = [z1,0]
    #p1 = [gamma,alpha_2]
    Q1 = np.identity(2)

    # child 2
    G2 = [
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1+gamma)-1/(1j*z1-alpha_2+alpha_1+gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2+alpha_1-gamma)-1/(1j*z1-alpha_2+alpha_1-gamma))]],
        [[1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1+gamma)-1/(1j*z1-alpha_2-alpha_1+gamma))]],
        [[ 1/(2*pi)*(1/(1j*z1+alpha_2-alpha_1-gamma)-1/(1j*z1-alpha_2-alpha_1-gamma))]]
        ]
    A2 = np.array([[1,-1], [0, -1]])
    # b2 = [0,z1]
    #p2 = [alpha_1,gamma]
    Q2 = np.identity(2)

    G3 = [
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1+alpha_2)-1/(1j*z1-gamma+alpha_1+alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma+alpha_1-alpha_2)-1/(1j*z1-gamma+alpha_1-alpha_2))]],
        [[1/(2*pi)*(1/(1j*z1+gamma-alpha_1+alpha_2)-1/(1j*z1-gamma-alpha_1+alpha_2))]],
        [[ 1/(2*pi)*(1/(1j*z1+gamma-alpha_1-alpha_2)-1/(1j*z1-gamma-alpha_1-alpha_2))]]
            ]
    A3 = np.array([[1,0], [0, 1]])
    # b3 = [0,0]
    # p3 = [alpha_1,alpha_2]
    Q3 = np.identity(2)

    B = [0,1,2,3]

    lambda1 = 1
    lambda2 = 2
    p1_coeff = 1/math.sqrt(lambda1)
    p2_coeff = 1/math.sqrt(lambda2)

    # t = 1:
    t1_gm1_1_p = p2_coeff
    t1_gm1_1_q = alpha_2 + 1j*b2

    t1_gm1_2_p = p2_coeff
    t1_gm1_2_q = -alpha_2 + 1j*b2

    t1_gm2_1_p = p2_coeff
    t1_gm2_1_q = -alpha_2 + 1j*b2

    t1_gm2_2_p = -p2_coeff
    t1_gm2_2_q = -alpha_2 + 1j*b2
    ##
    t1_gp1_1_p = -p2_coeff
    t1_gp1_1_q = alpha_2 + 1j*b2

    t1_gp1_2_p = -p2_coeff
    t1_gp1_2_q = -alpha_2 + 1j*b2

    t1_gp2_1_p = p2_coeff
    t1_gp2_1_q = alpha_2 + 1j*b2

    t1_gp2_2_p = -p2_coeff
    t1_gp2_2_q = alpha_2 + 1j*b2
    ##
    
    t1_m_1_p = p1_coeff+p2_coeff
    t1_m_1_q = gamma+1j*(b1+b2)

    t1_m_2_p = p1_coeff
    t1_m_2_q = alpha_2+gamma+1j*b1

    t1_p_1_p = -p1_coeff-p2_coeff
    t1_p_1_q = -gamma+1j*(b1+b2)

    t1_p_2_p = -p1_coeff
    t1_p_2_q = -alpha_2-gamma+1j*b1

    # t = 2:
    t2_gm1_1_p = p2_coeff
    t2_gm1_1_q = -alpha_1 + gamma + 1j*b2

    t2_gm1_2_p = -p2_coeff
    t2_gm1_2_q = -alpha_1 - gamma + 1j*b2

    t2_gm2_1_p = p2_coeff
    t2_gm2_1_q = alpha_1 + gamma + 1j*b2

    t2_gm2_2_p = p2_coeff
    t2_gm2_2_q = -alpha_1 + gamma + 1j*b2
    ##
    t2_gp1_1_p = p2_coeff
    t2_gp1_1_q = alpha_1 + gamma + 1j*b2

    t2_gp1_2_p = -p2_coeff
    t2_gp1_2_q = alpha_1 - gamma + 1j*b2

    t2_gp2_1_p = -p2_coeff
    t2_gp2_1_q = alpha_1 - gamma + 1j*b2

    t2_gp2_2_p = -p2_coeff
    t2_gp2_2_q = -alpha_1 - gamma + 1j*b2
    ##
    
    t2_m_1_p = p1_coeff
    t2_m_1_q = alpha_1+1j*(b1)

    t2_m_2_p = p1_coeff + p2_coeff
    t2_m_2_q = gamma+1j*(b1+b2)

    t2_p_1_p = -p1_coeff
    t2_p_1_q = -alpha_1+1j*b1

    t2_p_2_p = -p1_coeff - p2_coeff
    t2_p_2_q = -gamma+1j*(b1+b2)

    # t = 3:
    t3_gm1_1_p = p2_coeff
    t3_gm1_1_q = alpha_2 + 1j*b2

    t3_gm1_2_p = -p2_coeff
    t3_gm1_2_q = -alpha_2 + 1j*b2

    ##
    t3_gp1_1_p = p2_coeff
    t3_gp1_1_q = alpha_2 + 1j*b2

    t3_gp1_2_p = -p2_coeff
    t3_gp1_2_q = -alpha_2 + 1j*b2

    ##
    
    t3_m_1_p = p1_coeff
    t3_m_1_q = alpha_1+1j*(b1)

    t3_p_1_p = -p1_coeff
    t3_p_1_q = -alpha_1+1j*b1


    ##
    t1_gm1_1_p
    t1_gm1_1_q

    t1_m_1_p
    t1_m_1_q

    first_num = G1[1][0][0]/(t1_gm1_1_q*t1_m_1_q)

    g_pp = G3[0][0][0]
    g_pm = G3[1][0][0]
    g_mp = G3[2][0][0]
    g_mm = G3[3][0][0]
    p1 = p1_coeff
    p2 = p2_coeff
    q1 = alpha_1+1j*b1
    q2 = alpha_2 + 1j*b2
    q3 = -alpha_1 + 1j*b1
    q4 = -alpha_2 + 1j*b2

    e = (g_pp+g_pm+g_mp+g_mm)/(p1*p2)
    d = ((p1*q2*(-g_mm-g_pm) + p2*q1*(-g_mm-g_mp) + p1*q4*(g_mp+g_pp) + p2*q3*(g_pm + g_pp))/(p1*p2)**2)
    c = g_mm*((p1*q2+p2*q1)**2/(p1*p2)**3 - (q1*q2)/(p1*p2)**2) + g_mp*((p1*q4-p2*q1)**2/(p1*p2)**3 + (q1*q4)/(p1*p2)**2) + g_pm*((p2*q3-p1*q2)**2/(p1*p2)**3 + (q2*q3)/(p1*p2)**2) + g_pp*((p1*q4+p2*q3)**2/(p1*p2)**3 - (q3*q4)/(p1*p2)**2)

    c1 = ((p1/q1+p2/q2)**2/(p1*p2/(q1*q2))**3 - 1/(p1*p2/(q1*q2))**2)
    c2 = ((p1/q1-p2/q4)**2/-(p1*p2/(q1*q4))**3 - 1/(p1*p2/(q1*q4))**2)
    c3 = ((p2/q2-p1/q3)**2/-(p1*p2/(q2*q3))**3 - 1/(p1*p2/(q2*q3))**2)
    c4 = ((-p1/q3-p2/q4)**2/(p1*p2/(q3*q4))**3 - 1/(p1*p2/(q3*q4))**2)

    a1 = p1/q1; bb1 = p2/q2
    a2 = p1/q1; bb2 = -p2/q4
    a3 = -p1/q3; bb3 = p2/q2
    a4 = -p1/q3; bb4 = -p2/q4

    d1 = -(a1+bb1)/(a1*bb1)**2
    d2 = -(a2+bb2)/(a2*bb2)**2
    d3 = -(a3+bb3)/(a3*bb3)**2
    d4 = -(a4+bb4)/(a4*bb4)**2


    test_c = g_mm*c1/(q1*q2) - g_mp*c2/(q1*q4) - g_pm*c3/(q2*q3) + g_pp*c4/(q3*q4)
    test_d = g_mm*d1/(q1*q2) - g_mp*d2/(q1*q4) - g_pm*d3/(q2*q3) + g_pp*d4/(q3*q4)
    

    f1 = c1*bb1/(a1-bb1) + d1/(a1-bb1)
    f2 = c2*bb2/(a2-bb2) + d2/(a2-bb2)
    f3 = c3*bb3/(a3-bb3) + d3/(a3-bb3)
    f4 = c4*bb4/(a4-bb4) + d4/(a4-bb4)

    g1 = -c1-f1
    g2 = -c2-f2
    g3 = -c3-f3
    g4 = -c4-f4

    ff1 = f1*g_mm/(q1*q2)
    ff2 = f2*g_mp/(q1*q4)
    ff3 = f3*g_pm/(q2*q3)
    ff4 = f4*g_pp/(q3*q4)

    gg1 = g1*g_mm/(q1*q2)
    gg2 = g2*g_mp/(q1*q4)
    gg3 = g3*g_pm/(q2*q3)
    gg4 = g4*g_pp/(q3*q4)

    

    
    # test_f = g_mm*f1/(q1*q2) - g_mp*f2/(q1*q4) - g_pm*f3/(q2*q3) + g_pp*f4/(q3*q4)
    # test_g = g_mm*g1/(q1*q2) - g_mp*g2/(q1*q4) - g_pm*g3/(q2*q3) + g_pp*g4/(q3*q4)


    pass

    




if __name__ == "__main__":
    test_2d()