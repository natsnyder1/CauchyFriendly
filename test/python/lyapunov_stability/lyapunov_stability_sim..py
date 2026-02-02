import numpy as np
from scipy.stats import cauchy 
import matplotlib.pyplot as plt
import math
from term import Term
pi=math.pi

def advance_simulation_truth(Phi,B,Gamma,xk,beta):
    wk = cauchy.rvs(loc=0,scale=beta,size=1)
    xk1 = Phi*xk + Gamma*wk

    return xk1

def advance_simulation_measurement(H,xk,gamma):
    vk = cauchy.rvs(loc=0,scale=gamma,size=1)
    zk = H*xk+vk

    return zk

def run_simulation():
    define_simulation()

def define_simulation():
    Phi = 0.8
    B = 1
    Gamma = 1
    H = 1

    alpha = 0.1 # 
    gamma = 0.2 # meas noise
    beta = 0.3 # process noise

    P = 1 # Weighting constant in Lyapunov function 

    x0 = 0 
    x = np.linspace(-10,10,1000)
    pdf_val = cauchy.pdf(x,loc=x0,scale=alpha)
    x_init = cauchy.rvs(loc=x0,scale=alpha,size=1)

    time = 10
    saved_data = np.zeros((time,2))
    saved_estimator = np.zeros([time,1])
    
    xk = x_init
    z0 = advance_simulation_measurement(H,xk,gamma)
    saved_data[0,0] = xk
    saved_data[0,1] = z0
    
    list_of_estimator_terms_0_0 = define_estimator(alpha,gamma,z0,H)
    x0_hat = calc_estimate_from_listofterms(0,list_of_estimator_terms_0_0)
    saved_estimator[0,0] = x0_hat

    list_of_estimator_terms_km1_km1 = list_of_estimator_terms_0_0
    for k in range(1,time):
        saved_data[k,0] = xk
        zk = advance_simulation_measurement(H,xk,gamma)
        saved_data[k,1] = zk

        #list_of_estimaor_terms_0_0 = define_estimator(alpha,gamma,zk,H)

        xk1 = advance_simulation_truth(Phi,B,Gamma,xk,beta)

        list_of_tp_terms_k_km1 = estimator_tp(Phi,beta,list_of_estimator_terms_km1_km1)
        list_of_mu_terms_k_k = estimator_mu(k,gamma,H,zk,list_of_tp_terms_k_km1)
        xk_hat = calc_estimate_from_listofterms(k,list_of_mu_terms_k_k)
        saved_estimator[k,0] = xk_hat


        list_of_estimator_terms_km1_km1 = list_of_mu_terms_k_k
        xk = xk1
    

    plt.plot(range(0,time),saved_data[:,0],color='blue')
    plt.plot(range(0,time),saved_data[:,1],color='red')

    # plt.plot(x,pdf_val,color = 'blue')
    # plt.plot(x,cauchy.pdf(x,loc=x0,scale=0.4),color ='red')
    # plt.xlabel('x')
    # plt.ylabel('Probability Density')

    plt.show()

def define_estimator(alpha,gamma,z0,H):
    omega1_0 = alpha
    omega2_0 = gamma/abs(H)
    sigma2_0 = z0/H
    sigma1_0 = 0

    c1_0 = np.real(1/(2*pi*abs(H)) *(1/(omega1_0 + omega2_0 + 1j*sigma2_0)-1/(omega1_0 - omega2_0 + 1j*sigma2_0)))
    d1_0 = np.imag(1/(2*pi*abs(H)) *(1/(omega1_0 + omega2_0 + 1j*sigma2_0)-1/(omega1_0 - omega2_0 + 1j*sigma2_0)))

    c2_0 = np.real(1/(2*pi*abs(H)) *(1/(omega1_0 - omega2_0 + 1j*sigma2_0)-1/(omega1_0 + omega2_0 - 1j*sigma2_0)))
    d2_0 = np.imag(1/(2*pi*abs(H)) *(1/(omega1_0 - omega2_0 + 1j*sigma2_0)-1/(omega1_0 + omega2_0 - 1j*sigma2_0)))

    term1_0 = Term(c1_0,d1_0,omega1_0,sigma1_0)
    term2_0 = Term(c2_0,d2_0,omega2_0,sigma2_0)

    return [term1_0, term2_0]


def sign(x):
    return math.copysign(1,x)

def estimator_term_tp(Phi,beta,term):
    omegai = term.omegai
    sigmai = term.sigmai
    ci = term.ci
    di = term.di

    omegai_1 = abs(Phi)*omegai+beta
    sigmai_1 = Phi*sigmai
    ci_1 = ci
    di_1 = di*sign(Phi)

    termi_1 = Term(ci_1,di_1,omegai_1,sigmai_1)

    return termi_1

def estimator_tp(Phi,beta,list_of_terms):
    list_of_tp_terms = []
    for termi in list_of_tp_terms:
        termi_1 = estimator_term_tp(Phi,beta,termi)
        list_of_tp_terms.append(termi_1)
    return list_of_tp_terms

def estimator_mu(k,gamma,H,zk,list_of_past_terms):
    omega_k2 = gamma/abs(H)
    sigma_k2 = zk/H

    ab_bigarr = np.zeros((2,k+1))
    list_of_new_terms = []

    for i,term in enumerate(list_of_past_terms): 
        omegai = term.omegai
        sigmai = term.sigmai


        delti = (sigma_k2 -sigmai)^2 + omega_k2^2-omegai^2
        thetai = 2*omegai*(sigma_k2-sigmai)

        Delti = pi*abs(H)/(omega_k2) *(delti^2 + thetai^2)
        
        Fi = 1/(Delti) * np.array([[delti, -thetai], [thetai, delti]])
        
        ab_arr_mat = np.array([[-omegai/omega_k2, (sigma_k2 - sigmai)/(omega_k2)],[0,1]])

        past_cd_arr = np.array([[term.ci],[term.di]])
        cd_arr = Fi * past_cd_arr
        ab_arr = ab_arr_mat* cd_arr
        ab_bigarr[:,i] = ab_arr

        newterm_i = Term(cd_arr[0],cd_arr[1],omegai,sigmai)
        list_of_new_terms.append(newterm_i)

    ab_sum = np.sum(ab_bigarr,axis=1)
    c_k2 = ab_sum[0]
    d_k2 = ab_sum[1]

    newterm_k2 = Term(c_k2,d_k2,omega_k2,sigma_k2)
    list_of_new_terms.append(newterm_k2)

    return list_of_new_terms

def calc_estimate_from_listofterms(k,list_of_terms):
    if len(list_of_terms) != k+2:
        return "ERROR"
    fyk = sum([termi.ci for termi in list_of_terms])
    xk_hat = sum([termi.ci*termi.sigmai - termi.di*termi.omegai for termi in list_of_terms])
    return xk_hat

if __name__ == "__main__":
    define_simulation()