import numpy as np
from scipy.stats import cauchy 
import matplotlib.pyplot as plt
import math
from term import Term
pi=math.pi
import random

def advance_simulation_truth(Phi,B,Gamma,beta,xk_bar,xk_tild,uk):
    wk = cauchy.rvs(loc=0,scale=beta,size=1)[0]
    xk1_bar = Phi*xk_bar + B*uk
    xk1_tild = Phi*xk_tild + Gamma*wk

    return xk1_bar,xk1_tild,wk

def advance_simulation_measurement(H,xk,gamma):
    vk = cauchy.rvs(loc=0,scale=gamma,size=1)[0]
    zk = H*xk+vk

    return zk,vk

def run_simulation_MC():
    runs = 100
    time = 300
    
    Phi = 0.8
    B = 1
    Gamma = 1
    H = 1

    alpha = 0.1 # x initilization 
    gamma = 0.2 # meas noise
    beta = 0.3 # process noise

    P = 1 # Weighting constant in Lyapunov function 

    saved_expec = np.zeros(time)
    rhok = alpha
    for k in range(time): 
        saved_expec[k] = 2*math.log(1+rhok*math.sqrt(P))

        rho_kp1 = abs(Phi)*rhok + Gamma*beta
        rhok = rho_kp1

    saved_lyap_all = np.zeros(shape=(runs,time,2))
    for run in range(0,runs):
        saved_lyap_func = define_simulation(time,Phi,B,Gamma,H,alpha,gamma,beta,P)
        saved_lyap_all[run,:,:] = saved_lyap_func

    lyap_avg = np.mean(saved_lyap_all,axis=0)

    plt.subplot(311)
    plt.plot(range(0,time),lyap_avg[:,0],color='blue',label="E[V(xk)|y(k)]")
    plt.plot(range(0,time-1),saved_expec[:-1],color='red',label="E[V(xk)]")

    plt.legend()
    plt.xlabel('time')

    plt.subplot(312)
    plt.plot(range(0,time-1),lyap_avg[:-1,1],color='blue',label="E[V(xk+1)|y(k)]")
    plt.plot(range(0,time-1),saved_expec[:-1],color='red',label="E[V(xk)]")

    plt.legend()
    plt.xlabel('time')

    plt.subplot(313)
    plt.plot(range(0,time-1),lyap_avg[:-1,1]-lyap_avg[:-1,0],color='green',label="E[V(xk+1)|y(k)]-E[V(xk)|y(k)]")
    plt.legend()
    plt.xlabel('time')

    plt.show()


def define_simulation(time=100,Phi=0.8,B=1,Gamma=1,H=1,alpha=0.1,gamma=0.1,beta=0.1,P=1):
    
    #check parameters
    if (abs(Phi) < 1):
        if math.pi*gamma *abs( (beta/(1-abs(Phi))* H/gamma)**2 -1) < 1:
            return "Pick different parameters"

    x0_bar = 50*(random.random()-0.5)
    x0_tild = cauchy.rvs(loc=0,scale=alpha,size=1)[0]

    saved_data = np.zeros((time,5)) #x_bar, x_tild, zk, vk, wk 
    saved_estimator = np.zeros([time,2]) #xk_hat, fyk
    saved_lyap_functions = np.zeros([time,2]) 
    
    xk = x0_bar + x0_tild
    z0,v0 = advance_simulation_measurement(H,xk,gamma)
    saved_data[0,0] = x0_bar
    saved_data[0,1] = x0_tild
    saved_data[0,2] = z0
    saved_data[0,3] = v0
    
    list_of_estimator_terms_0_0 = define_estimator(alpha,gamma,z0,H)
    x0_hat,fy0 = calc_estimate_from_listofterms(0,list_of_estimator_terms_0_0)
    lyap_val0 = calc_Lyap_function(0,P,list_of_estimator_terms_0_0,fy0)

    # save estimate and inital lyapunov function value
    saved_estimator[0,0] = x0_hat
    saved_estimator[0,1] = fy0
    saved_lyap_functions[0,0] = lyap_val0

    # update simulation time k-1 _ k
    list_of_estimator_terms_km1_km1 = list_of_estimator_terms_0_0
    list_of_estimator_terms_k_km1 = list_of_estimator_terms_0_0
    x_km1 = xk
    xk_bar = x0_bar
    xk_tild = x0_tild
    fy_km1 = fy0


    for k in range(1,time):
        
        saved_data[k,0] = xk_bar
        saved_data[k,1] = xk_tild

        #Control
        uk = 0

        #Advance simulation measurement @ k 
        zk,vk = advance_simulation_measurement(H,xk_bar+xk_tild,gamma)
        saved_data[k,2] = zk
        saved_data[k,3] = vk

        # #Estimator time propogation (k-1 | k-1) -> (k | k-1)
        # list_of_tp_terms_k_km1 = estimator_tp(Phi,beta,list_of_estimator_terms_km1_km1)
        # lyap_val_k1 = calc_second_Lyap_function(k-1,P,list_of_tp_terms_k_km1,xk_bar,fy_km1)
        # saved_lyap_functions[k-1,1] = lyap_val_k1
        
        #Estimator measuremnt update (k | k-1) -> (k |k)
        #list_of_mu_terms_k_k = estimator_mu(k,gamma,H,zk,list_of_tp_terms_k_km1)
        list_of_mu_terms_k_k = estimator_mu(k,gamma,H,zk,list_of_estimator_terms_k_km1)

        # calculate x hat and lyapunov function values
        xk_hat, fyk = calc_estimate_from_listofterms(k,list_of_mu_terms_k_k)
        # if fyk < 1e-100: 
        #      print("check?")
        lyap_val = calc_Lyap_function(k,P,list_of_mu_terms_k_k,fyk)

        #save x hat and lyap function values
        saved_estimator[k,0] = xk_hat
        saved_estimator[k,1] = fyk
        saved_lyap_functions[k,0] = lyap_val

        #Advance truth k -> k+1
        xkp1_bar,xkp1_tild,wk = advance_simulation_truth(Phi,B,Gamma,beta,xk_bar,xk_tild,uk)
        saved_data[k,4] = wk

        #Estimator time propogation (k|k) -> (k+1 | k)
        list_of_tp_terms_kp1_k = estimator_tp(Phi,beta,list_of_mu_terms_k_k)
        lyap_val_k1 = calc_second_Lyap_function(k,P,list_of_tp_terms_kp1_k,xk_bar,fyk)
        saved_lyap_functions[k,1] = lyap_val_k1


        # Reset truth 
        x_km1 = xk
        xk_bar = xkp1_bar
        xk_tild = xkp1_tild
        fy_km1 = fyk

        # Reset estimate (k|k) -> (km1 | km1)
        #list_of_estimator_terms_km1_km1 = list_of_mu_terms_k_k
        list_of_estimator_terms_k_km1 = list_of_tp_terms_kp1_k
        
    show = True
    if show:
        plt.figure()
        plt.subplot(511)
        plt.plot(range(0,time),saved_data[:,0]+saved_data[:,1],color='blue',label="truth")
        plt.plot(range(0,time),saved_data[:,2],color='red',label="measurement")
        plt.plot(range(0,time),saved_estimator[:,0],color='green',label = "estimate")
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('xk, zk, xk_hat')

        plt.subplot(512)
        plt.plot(range(0,time),saved_lyap_functions[:,0],color='blue',label="E[V(xk)|y(k)]")
        plt.legend()
        plt.xlabel('time')

        plt.subplot(513)
        plt.plot(range(0,time-1),saved_lyap_functions[:-1,1],color='red',label="E[V(xk+1)|y(k)]")
        plt.legend()
        plt.xlabel('time')

        plt.subplot(514)
        plt.plot(range(0,time-1),saved_lyap_functions[:-1,1]-saved_lyap_functions[:-1,0],color='green',label="E[V(xk+1)|y(k)]-E[V(xk)|y(k)]")
        plt.legend()
        plt.xlabel('time')

        plt.subplot(515)
        plt.plot(range(0,time),saved_estimator[:,1],color='blue',label="fyk")
        plt.legend()
        plt.xlabel('time')
        


        plt.show()

    return saved_lyap_functions

def define_estimator(alpha,gamma,z0,H):
    omega1_0 = alpha
    omega2_0 = gamma/abs(H)
    sigma2_0 = z0/H
    sigma1_0 = 0

    c1_0 = np.real(1/(2*pi*abs(H)) *((1/(omega1_0 + omega2_0 + 1j*sigma2_0))-(1/(omega1_0 - omega2_0 + 1j*sigma2_0))))
    d1_0 = np.imag(1/(2*pi*abs(H)) *(1/(omega1_0 + omega2_0 + 1j*sigma2_0)-1/(omega1_0 - omega2_0 + 1j*sigma2_0)))

    c2_0 = np.real(1/(2*pi*abs(H)) *(1/(omega1_0 - omega2_0 + 1j*sigma2_0)+1/(omega1_0 + omega2_0 - 1j*sigma2_0)))
    d2_0 = np.imag(1/(2*pi*abs(H)) *(1/(omega1_0 - omega2_0 + 1j*sigma2_0)+1/(omega1_0 + omega2_0 - 1j*sigma2_0)))

    term1_0 = Term(c1_0,d1_0,omega1_0,sigma1_0)
    term2_0 = Term(c2_0,d2_0,omega2_0,sigma2_0)

    cd_bigarr = np.array([[c1_0,c2_0],[d1_0,d2_0]])
    print(cd_bigarr)

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
    for termi in list_of_terms:
        termi_1 = estimator_term_tp(Phi,beta,termi)
        list_of_tp_terms.append(termi_1)
    return list_of_tp_terms

def estimator_mu(k,gamma,H,zk,list_of_past_terms):
    omega_k2 = gamma/abs(H)
    sigma_k2 = zk/H

    ab_bigarr = np.zeros((2,len(list_of_past_terms)))
    cd_bigarr = np.zeros((2,len(list_of_past_terms)+1))
    cd_bigarr_savedsums = np.zeros((2,len(list_of_past_terms)+1))

    #ab_bigarr = np.zeros((2,k+1))
    #cd_bigarr = np.zeros((2,k+2))

    list_of_new_terms = []
    k_truncate=0
    for i,term in enumerate(list_of_past_terms): 
        omegai = term.omegai
        sigmai = term.sigmai

        delti = (sigma_k2 -sigmai)**2 + omega_k2**2 - omegai**2
        thetai = 2*omegai*(sigma_k2-sigmai)

        Delti = pi*abs(H)/(omega_k2) *(delti**2 + thetai**2)
        
        Fi = 1/(Delti) * np.array([[delti, -thetai], [thetai, delti]])
        
        ab_arr_mat = np.array([[-omegai/omega_k2, (sigma_k2 - sigmai)/(omega_k2)],[0,-1]])

        past_cd_arr = np.array([[term.ci],[term.di]])
        cd_arr = np.matmul(Fi,past_cd_arr)
        cd_bigarr[:,i:i+1] = cd_arr
        cd_bigarr_savedsums[:,i]=np.sum(cd_bigarr,axis=1)
        
        ab_arr = np.matmul(ab_arr_mat, cd_arr)
        ab_bigarr[:,i:i+1] = ab_arr
        ab_bigarr[0:2,i:i+1]=ab_arr

        newterm_i = Term(cd_arr[0,0],cd_arr[1,0],omegai,sigmai)
        list_of_new_terms.append(newterm_i)

    ab_sum = np.sum(ab_bigarr,axis=1)
    c_k2 = ab_sum[0]
    d_k2 = ab_sum[1]

    cd_bigarr[0,-1] = c_k2
    cd_bigarr[1,-1] = d_k2

    # print(cd_bigarr)
    # print(k_truncate)

    newterm_k2 = Term(c_k2,d_k2,omega_k2,sigma_k2)
    list_of_new_terms.append(newterm_k2)

    return list_of_new_terms

def calc_estimate_from_listofterms(k,list_of_terms):
    # if len(list_of_terms) != k+2:
    #     return "ERROR"
    fyk = sum([termi.ci for termi in list_of_terms])
    xk_hat = sum([termi.ci*termi.sigmai - termi.di*termi.omegai for termi in list_of_terms])

    return xk_hat/fyk, fyk

def calc_Lyap_function(k,p,list_of_terms,fyk):
    lyap_vec = np.zeros(k+2)

    for i,termi in enumerate(list_of_terms):
        ci = termi.ci
        di = termi.di
        omegai = termi.omegai
        sigmai = termi.sigmai

        lyap_element = 0.5*ci*math.log( (1+ math.sqrt(p)*omegai)**2 + p *sigmai**2) + di*math.atan(math.sqrt(p)*sigmai/(1+math.sqrt(p)*omegai))
        lyap_vec[i] = lyap_element
     
    lyap_function_val = 2/fyk *(np.sum(lyap_vec))

    return lyap_function_val

def calc_second_Lyap_function(k,p,list_of_tp_terms,xk1_bar,fyk):
    lyap_vec_k1 = np.zeros(len(list_of_tp_terms))
    # lyap_vec_k1 = np.zeros(k+2)

    for i,termi in enumerate(list_of_tp_terms):
        ci = termi.ci
        di = termi.di
        omegai = termi.omegai
        sigmai = termi.sigmai

        lyap_element = 0.5*ci*math.log( (1+ math.sqrt(p)*omegai)**2 + p *(sigmai+ xk1_bar)**2) + di*math.atan(math.sqrt(p)*(sigmai+ xk1_bar)/(1+math.sqrt(p)*omegai))
        lyap_vec_k1[i] = lyap_element
    
    #lyap_function_val_k1 = 2*(np.sum(lyap_vec_k1))
    lyap_function_val_k1 = 2/fyk *(np.sum(lyap_vec_k1))

    return lyap_function_val_k1



if __name__ == "__main__":
    np.random.seed(seed=233423)    
    #rs = RandomState(MT19937(SeedSequence(123456789)))
    #run_simulation()
    #og_set = np.seterr({'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'})
    np.seterr(over='raise')
    define_simulation(time=300,Phi=0.95, H=1)