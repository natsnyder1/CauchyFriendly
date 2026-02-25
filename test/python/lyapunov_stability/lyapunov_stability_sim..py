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

def run_simulation_MC(runs=200,time=200,print_individ_plots=False,control_steps=1,save_data_to_text=False):
    
    Phi = 0.95
    B = 1
    Gamma = 1
    H = 1

    alpha = 0.5 # x initilization 
    gamma = 0.1 # meas noise
    beta = 0.02 # process noise

    P = 1 # Weighting constant in Lyapunov function 

    saved_expec = np.zeros((time,3))
    rhok = alpha
    for k in range(time): 
        saved_expec[k,0] = 2*math.log(1+rhok*math.sqrt(P))

        rho_kp1 = abs(Phi)*rhok + Gamma*beta
        saved_expec[k,1] = 2*math.log(1+rho_kp1*math.sqrt(P))
        saved_expec[k,2] = math.log((1+rho_kp1*math.sqrt(P))/(1+rhok*math.sqrt(P)))
        rhok = rho_kp1

    saved_lyap_all = np.zeros(shape=(runs,time,2))
    for run in range(0,runs):
        saved_lyap_func = define_simulation(time,Phi,B,Gamma,H,alpha,gamma,beta,P,print_individ_plots=print_individ_plots,control_steps = control_steps,print_control_cost=False)
        saved_lyap_all[run,:,:] = saved_lyap_func
        print(run)

    lyap_avg = np.mean(saved_lyap_all,axis=0)

    if save_data_to_text:
        lyap_diff = lyap_avg[:-1,1]-lyap_avg[:-1,0]
        lyap_to_save = np.append(lyap_avg[:-1,:],lyap_diff.reshape(len(lyap_diff),1),axis=1)
        file = "saved_MC.txt"
        np.savetxt(file, lyap_to_save, delimiter=' ')

        # Read the text file back into a NumPy array
        #loaded_array = np.loadtxt(text_file_path, delimiter=' ')

    plt.subplot(311)
    plt.plot(range(0,time),lyap_avg[:,0],color='blue',label="E[V(xk)|y(k)]")
    if control_steps ==0: 
        plt.plot(range(0,time-1),saved_expec[:-1,0],color='red',label="E[V(xk)]")
    plt.legend()
    plt.xlabel('time')

    plt.subplot(312)
    plt.plot(range(0,time-1),lyap_avg[:-1,1],color='blue',label="E[V(xk+1)|y(k)]")
    if control_steps ==0:
        plt.plot(range(0,time-1),saved_expec[:-1,1],color='red',label="E[V(xk+1)]")
    plt.legend()
    plt.xlabel('time')

    plt.subplot(313)
    plt.plot(range(0,time-1),np.zeros(len(range(0,time-1))),color='red',linestyle="dashed")
    plt.plot(range(0,time-1),lyap_avg[:-1,1]-lyap_avg[:-1,0],color='green',label="E[V(xk+1)|y(k)]-E[V(xk)|y(k)]")
    if control_steps ==0:
        plt.plot(range(0,time-1),saved_expec[:-1,2],color='red',label="deltV")
    plt.legend()
    plt.xlabel('time')

    plt.show()


def define_simulation(time=100,Phi=0.8,B=1,Gamma=1,H=1,alpha=0.1,gamma=0.1,beta=0.1,P=1,eta_r =0.7,theta_i=8,print_individ_plots=True,control_steps = 1,print_control_cost = True):
    #check parameters
    if (abs(Phi) < 1):
        if math.pi*gamma *abs( (beta/(1-abs(Phi))* H/gamma)**2 -1) < 1:
            print("Pick different parameters")
            return 

    x0_bar = 0
    x0_tild = cauchy.rvs(loc=0,scale=alpha,size=1)[0]

    saved_data = np.zeros((time,6)) #x_bar, x_tild, zk, vk, wk, ul
    saved_estimator = np.zeros([time,3]) #xk_hat, fyk
    saved_lyap_functions = np.zeros([time,2]) 
    
    xk = x0_bar + x0_tild
    z0,v0 = advance_simulation_measurement(H,x0_tild,gamma)
    saved_data[0,0] = x0_bar
    saved_data[0,1] = x0_tild
    saved_data[0,2] = z0 + x0_bar
    saved_data[0,3] = v0
    
    list_of_estimator_terms_0_0 = define_estimator(alpha,gamma,z0,H)
    x0_hat,fy0 = calc_estimate_from_listofterms(0,list_of_estimator_terms_0_0)
    lyap_val0 = calc_Lyap_function(0,P,list_of_estimator_terms_0_0,x0_bar,fy0)
    saved_lyap_functions[0,0] = lyap_val0

    if control_steps>0:
        u0 = calc_control(1,0,control_steps,Phi,B,beta,Gamma,eta_r,theta_i,x0_bar,fy0,list_of_estimator_terms_0_0,show=print_control_cost)
        #print(uk)
    else: 
        u0 = 0
    saved_data[0,5] = u0

    xp1_bar,xp1_tild,wk = advance_simulation_truth(Phi,B,Gamma,beta,x0_bar,x0_tild,u0)
    saved_data[0,4] = wk
    list_of_estimator_terms_1_0 = estimator_tp(Phi,beta,list_of_estimator_terms_0_0,xp1_bar)
    lyap_val_k1_0 = calc_second_Lyap_function(0,P,list_of_estimator_terms_1_0,xp1_bar,fy0)
    saved_lyap_functions[0,1] = lyap_val_k1_0

    # save estimate and inital lyapunov function value
    saved_estimator[0,0] = x0_hat + x0_bar
    saved_estimator[0,1] = fy0
    saved_lyap_functions[0,0] = lyap_val0

    # update simulation time k-1 _ k
    list_of_estimator_terms_k_km1 = list_of_estimator_terms_1_0
    xk_bar = xp1_bar
    xk_tild = xp1_tild


    for k in range(1,time):
        
        saved_data[k,0] = xk_bar
        saved_data[k,1] = xk_tild

        #Advance simulation measurement @ k 
        zk_tild,vk = advance_simulation_measurement(H,xk_tild,gamma)
        saved_data[k,2] = zk_tild + H*xk_bar
        saved_data[k,3] = vk
        
        #Estimator measuremnt update (k | k-1) -> (k |k)
        list_of_mu_terms_k_k = estimator_mu(k,gamma,H,zk_tild,Phi,list_of_estimator_terms_k_km1)

        # calculate x hat and lyapunov function values
        xk_hat, fyk = calc_estimate_from_listofterms(k,list_of_mu_terms_k_k)
        lyap_val = calc_Lyap_function(k,P,list_of_mu_terms_k_k,xk_bar,fyk)

        #save x hat and lyap function values
        saved_estimator[k,0] = xk_hat + xk_bar
        saved_estimator[k,1] = fyk
        saved_lyap_functions[k,0] = lyap_val

        #Control
        if control_steps>0:
            uk = calc_control(1,k,control_steps,Phi,B,beta,Gamma,eta_r,theta_i,xk_bar,fyk,list_of_mu_terms_k_k,show=print_control_cost)
            saved_data[k,5] = uk
            #print(uk)
        else: 
            uk = 0


        #Advance truth k -> k+1
        xkp1_bar,xkp1_tild,wk = advance_simulation_truth(Phi,B,Gamma,beta,xk_bar,xk_tild,uk)
        saved_data[k,4] = wk

        #Estimator time propogation (k|k) -> (k+1 | k)
        list_of_tp_terms_kp1_k = estimator_tp(Phi,beta,list_of_mu_terms_k_k,xk_bar)
        xk1_hat,fyk1 = calc_estimate_from_listofterms(k,list_of_tp_terms_kp1_k)
        saved_estimator[k,2] = xk1_hat
        lyap_val_k1 = calc_second_Lyap_function(k,P,list_of_tp_terms_kp1_k,xkp1_bar,fyk)
        saved_lyap_functions[k,1] = lyap_val_k1

        # Reset truth 
        xk_bar = xkp1_bar
        xk_tild = xkp1_tild

        # Reset estimate (k|k) -> (km1 | km1)
        list_of_estimator_terms_k_km1 = list_of_tp_terms_kp1_k
        
    show = print_individ_plots
    if show:
        plt.figure()
        plt.subplot(311)
        plt.plot(range(0,time),saved_data[:,0]+saved_data[:,1],color='blue',label="truth")
        # plt.plot(range(0,time),saved_data[:,0],color='purple',label="xk_bar")
        plt.plot(range(0,time),saved_data[:,2],color='red',label="measurement")
        plt.plot(range(0,time),saved_estimator[:,0],color='green',label = "estimate k|k ")
        # plt.plot(range(0,time),saved_estimator[:,2],color='orange',label = "estimate k+1|k",linestyle="dashed")
        # plt.plot(range(0,time),np.zeros((len(range(0,time)))),linestyle="dashed")
        plt.legend()
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')
        plt.ylabel('xk, zk, xk_hat')


        plt.subplot(312)
        plt.plot(range(0,time),saved_data[:,3],color='blue',label="meas noise")
        plt.plot(range(0,time),saved_data[:,4],color='green',label="proc noise")
        plt.plot(range(0,time),np.zeros((len(range(0,time)))),linestyle="dashed")
        plt.legend()
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')

        plt.subplot(313)
        plt.plot(range(0,time),saved_data[:,5],color='blue',label="uk")
        plt.plot(range(0,time),np.zeros((len(range(0,time)))),linestyle="dashed")
        plt.legend()
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')
        

        plt.figure()
        plt.subplot(311)
        plt.plot(range(0,time),saved_lyap_functions[:,0],color='blue',label="E[V(xk)|y(k)]")
        plt.legend()
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')

        plt.subplot(312)
        plt.plot(range(0,time-1),saved_lyap_functions[:-1,1],color='red',label="E[V(xk+1)|y(k)]")
        plt.legend()
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')

        plt.subplot(313)
        plt.plot(range(0,time-1),saved_lyap_functions[:-1,1]-saved_lyap_functions[:-1,0],color='green',label="E[V(xk+1)|y(k)]-E[V(xk)|y(k)]")
        plt.legend()
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
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
    #print(cd_bigarr)

    return [term1_0, term2_0]


def sign(x):
    return math.copysign(1,x)

def estimator_term_tp(Phi,beta,term,xkp1_bar):
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

def estimator_tp(Phi,beta,list_of_terms,xkp1_bar):
    list_of_tp_terms = []
    for termi in list_of_terms:
        termi_1 = estimator_term_tp(Phi,beta,termi,xkp1_bar)
        list_of_tp_terms.append(termi_1)
    return list_of_tp_terms

def estimator_mu(k,gamma,H,zk,Phi,list_of_past_terms):
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
        Gi = Fi*np.array([[1,0],[0,sign(Phi)]])
        
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

    #print(cd_bigarr)
    c_sum = np.sum(cd_bigarr,axis=1)[0]
    d_sum = np.sum(cd_bigarr,axis=1)[1]
    #print(c_sum)
    #print(d_sum)
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

def calc_control(n,k,control_steps,Phi,B,beta,Gamma,eta_r,theta,xk_bar,fyk,list_of_terms,show):
    len_u = 300
    range = 5
    
    if control_steps==1:
        u_arr = np.linspace(-range,range,len_u)
        cost_val = np.zeros((len_u,len(list_of_terms)))
        cost_val_der = np.zeros((len_u,len(list_of_terms)))

        for i,term in enumerate(list_of_terms):
            omegai = term.omegai
            sigmai = term.sigmai
            ci = term.ci
            di = term.di
            
            num = ci*(2*omegai*Phi+2*eta_r+2*beta*Gamma) + di*(-2*sigmai*Phi - 2*(Phi*xk_bar + B*u_arr))
            den = omegai**2*Phi**2+2*omegai*Phi*eta_r+2*omegai*Phi*beta*Gamma+sigmai**2*Phi**2 + 2*sigmai*Phi*(Phi*xk_bar+B*u_arr) +eta_r**2+2*eta_r*beta*Gamma+(Phi*xk_bar+B*u_arr)**2+beta**2*Gamma**2
            term_cost = num/den 
            term_cost_2 = (ci-1j*di)/(omegai*Phi+1j*sigmai*Phi+eta_r+1j*(Phi*xk_bar+B*u_arr)+beta*Gamma) + (ci+1j*di)/(omegai*Phi-1j*sigmai*Phi+eta_r-1j*(Phi*xk_bar+B*u_arr)+beta*Gamma)

            M_cost = theta/pi /(u_arr**2+theta**2)
                
            cost_val[:,i] = term_cost * M_cost
            
            der_term = (ci-1j*di)*(1j*B)/(omegai*Phi+1j*sigmai*Phi+eta_r+1j*(Phi*xk_bar+B*u_arr)+beta*Gamma)**2+ (ci+1j*di)*(-1j*B)/(omegai*Phi-1j*sigmai*Phi+eta_r-1j*(Phi*xk_bar+B*u_arr)+beta*Gamma)**2
            der_M = theta/pi * 2*u_arr /(u_arr**2+theta**2)**2
                
            cost_val_der[:,i] = der_M* term_cost + der_term*M_cost
            
        cost = np.sum(cost_val,axis=1) /((2*pi)**n * fyk)
        ind_max = np.argmax(cost)

        cost_der = np.sum(cost_val_der,axis=1)

        if show:
            plt.figure()
            plt.subplot(211)
            plt.plot(u_arr,cost,color='blue',label="cost")
            plt.legend()
            plt.xlabel('u')

            plt.subplot(212)
            plt.plot(u_arr,cost_der,color='blue',label="cost derivative")
            plt.legend()
            plt.xlabel('u')

            plt.show
        
        return u_arr[ind_max]
    
    elif control_steps ==2:

        uk_arr = np.linspace(-range,range,len_u)
        uk1_arr = np.linspace(-range,range,len_u)
        cost_val = np.zeros((len_u,len_u,len(list_of_terms)))

        uk_arr, uk1_arr = np.meshgrid(uk_arr, uk1_arr)

        for i,term in enumerate(list_of_terms):
            omegai = term.omegai
            sigmai = term.sigmai
            ci = term.ci
            di = term.di
            
            term_cost = (ci-1j*di)/(omegai*Phi**2 + 1j*sigmai*Phi**2 + eta_r + 1j*(Phi**2*xk_bar + Phi*B*uk_arr + B*uk1_arr) + beta*Gamma + beta*Phi*Gamma) + (ci+1j*di)/(omegai*Phi**2-1j*sigmai*Phi**2+eta_r-1j*(Phi**2*xk_bar+Phi*B*uk_arr+B*uk1_arr)+beta*Gamma+beta*Phi*Gamma)
            M_cost = theta/pi /(uk_arr**2+theta**2) * theta/pi /(uk1_arr**2+theta**2)
            cost_val[:,:,i] = term_cost*M_cost

        cost = np.sum(cost_val,axis=2) /((2*pi)**n * fyk)
        ind_max = np.unravel_index(np.argmax(cost, axis=None), cost.shape)

        if show and k>45:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_wireframe(uk_arr, uk1_arr, cost, rstride=10, cstride=10)
            ax.set_xlabel("u k")
            ax.set_ylabel("u k+1")
            ax.set_zlabel("cost")

            plt.show()

        return uk_arr[ind_max]


def calc_Lyap_function(k,p,list_of_terms,xk_bar,fyk):
    lyap_vec = np.zeros(k+2)

    for i,termi in enumerate(list_of_terms):
        ci = termi.ci
        di = termi.di
        omegai = termi.omegai
        sigmai = termi.sigmai

        lyap_element = 0.5*ci*math.log( (1+ math.sqrt(p)*omegai)**2 + p *(sigmai+xk_bar)**2) + di*math.atan(math.sqrt(p)*(sigmai+xk_bar)/(1+math.sqrt(p)*omegai))
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

        #lyap_element = 0.5*ci*math.log( (1+ math.sqrt(p)*omegai)**2 + p *(sigmai)**2) + di*math.atan(math.sqrt(p)*(sigmai)/(1+math.sqrt(p)*omegai))
        lyap_element = 0.5*ci*math.log( (1+ math.sqrt(p)*omegai)**2 + p *(sigmai + xk1_bar)**2) + di*math.atan(math.sqrt(p)*(sigmai+ xk1_bar)/(1+math.sqrt(p)*omegai))
        
        lyap_vec_k1[i] = lyap_element
    
    #lyap_function_val_k1 = 2*(np.sum(lyap_vec_k1))
    lyap_function_val_k1 = 2/fyk *(np.sum(lyap_vec_k1))

    return lyap_function_val_k1



if __name__ == "__main__":
    np.seterr(over='raise')
    multiple_sim = True
    
    if multiple_sim:
        run_simulation_MC(runs=2,time=100,print_individ_plots=False,control_steps=2,save_data_to_text=True)
    else:
        np.random.seed(seed=233423) 
        define_simulation(time=100,Phi=0.95, H=1,alpha=0.5,beta=0.02,gamma=0.1,eta_r=0.7,control_steps=2,print_control_cost=False)
   
    #og_set = np.seterr({'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'})
    