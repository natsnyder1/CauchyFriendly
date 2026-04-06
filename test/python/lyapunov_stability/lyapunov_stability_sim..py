import numpy as np
from scipy.stats import cauchy 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
from term import Term
pi=math.pi
import random
from scipy.optimize import minimize

def advance_simulation_truth(k,Phi,B,Gamma,beta,xk_bar,xk_tild,uk):
    wk = cauchy.rvs(loc=0,scale=beta,size=1)[0]

    xk1_bar = Phi*xk_bar + B*uk
    xk1_tild = Phi*xk_tild + Gamma*wk

    return xk1_bar,xk1_tild,wk

def advance_simulation_measurement(H,xk,gamma):
    vk = cauchy.rvs(loc=0,scale=gamma,size=1)[0]
    zk = H*xk+vk

    return zk,vk

def run_simulation_MC(runs=200,time=200,print_individ_plots=False,control_steps=1,save_data_to_text=False,plot_all_runs=False,filename="saved_MC"):
    
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
        saved_expec[k,2] = math.log(((1+rho_kp1*math.sqrt(P))/(1+rhok*math.sqrt(P)))**2)
        rhok = rho_kp1

    saved_lyap_all = np.zeros(shape=(runs,time,2))
    for run in range(0,runs):
        saved_lyap_func = define_simulation(time,Phi,B,Gamma,H,alpha,gamma,beta,P,print_individ_plots=print_individ_plots,control_steps = control_steps,print_control_cost=False)
        saved_lyap_all[run,:,:] = saved_lyap_func

        if save_data_to_text:
            file = f"{filename}_allruns.npy"
            np.save(file,saved_lyap_all[:run+1,:,:])
        
        print(run)

    lyap_avg = np.mean(saved_lyap_all,axis=0)

    if save_data_to_text:
        lyap_diff = lyap_avg[:-1,1]-lyap_avg[:-1,0]
        lyap_to_save = np.append(lyap_avg[:-1,:],lyap_diff.reshape(len(lyap_diff),1),axis=1)
        file = f"{filename}_avg.txt"
        np.savetxt(file, lyap_to_save, delimiter=' ')

        file = f"{filename}_expectation.txt"
        np.savetxt(file,saved_expec,delimiter=' ')

        # Read the text file back into a NumPy array
        #loaded_array = np.loadtxt(text_file_path, delimiter=' ')
    
    if plot_all_runs:
        plt.figure()
        plt.title("Lyapunov Functions for all runs")
        plt.subplot(211)
        plt.plot(range(0,time),np.transpose(saved_lyap_all[:,:,0]),color='blue')
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')
        plt.ylabel("E[V(xk)|y(k)]")
    
        plt.subplot(212)
        plt.plot(range(0,time),np.transpose(saved_lyap_all[:,:,1]),color='blue')
        plt.minorticks_on()
        plt.grid(which="minor")
        plt.grid(which="major")
        plt.xlabel('time')
        plt.ylabel("E[V(xk+1)|y(k)]")

    plt.figure()
    plt.title("Averaged Lyapunov Functions")
    plt.subplot(311)
    plt.plot(range(0,time),lyap_avg[:,0],color='green',label="E[V(xk)|y(k)]")
    if control_steps ==0: 
        plt.plot(range(0,time-1),saved_expec[:-1,0],color='red',label="E[V(xk)]")
    plt.legend()
    plt.minorticks_on()
    plt.grid(which="minor")
    plt.grid(which="major")
    plt.xlabel('time')
    plt.ylim(0,1)

    plt.subplot(312)
    plt.plot(range(0,time-1),lyap_avg[:-1,1],color='green',label="E[V(xk+1)|y(k)]")
    if control_steps ==0:
        plt.plot(range(0,time-1),saved_expec[:-1,1],color='red',label="E[V(xk+1)]")
    plt.legend()
    plt.minorticks_on()
    plt.grid(which="minor")
    plt.grid(which="major")
    plt.xlabel('time')
    plt.ylim(0,1)

    plt.subplot(313)
    plt.plot(range(0,time-1),np.zeros(len(range(0,time-1))),color='red',linestyle="dashed")
    plt.plot(range(0,time-1),lyap_avg[:-1,1]-lyap_avg[:-1,0],color='green',label="E[V(xk+1)|y(k)]-E[V(xk)|y(k)]")
    if control_steps ==0:
        plt.plot(range(0,time-1),saved_expec[:-1,2],color='red',label="deltV")
    plt.legend()
    plt.minorticks_on()
    plt.grid(which="minor")
    plt.grid(which="major")
    plt.xlabel('time')

    plt.show()


def define_simulation(time=100,Phi=0.8,B=1,Gamma=1,H=1,alpha=0.1,gamma=0.1,beta=0.1,P=1,eta_r =0.7,theta_i=8,print_individ_plots=True,control_steps = 1,print_control_cost = True):
    #check parameters
    if (abs(Phi) < 1):
        if math.pi*gamma *abs( (beta/(1-abs(Phi))* H/gamma)**2 -1) < 1:
            print("Pick different parameters")
            return 

    x0_bar = 0
    #x0_tild = 100 #cauchy.rvs(loc=0,scale=alpha,size=1)[0]
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

    high_wk_ind = -5
    high_wk = 0
    if  abs(x0_tild) > 5:
        high_wk_ind=0
        high_wk = x0_tild

    if control_steps>0:
        u0 = calc_control(1,0,control_steps,Phi,B,beta,Gamma,eta_r,theta_i,x0_bar,fy0,list_of_estimator_terms_0_0,high_wk_ind,high_wk,show=print_control_cost)
        #print(uk)
    else: 
        u0 = 0
    saved_data[0,5] = u0

    xp1_bar,xp1_tild,wk = advance_simulation_truth(0,Phi,B,Gamma,beta,x0_bar,x0_tild,u0)
    saved_data[0,4] = wk
    if abs(wk)> 5: 
        high_wk_ind = 0
        high_wk = wk
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
            uk = calc_control(1,k,control_steps,Phi,B,beta,Gamma,eta_r,theta_i,xk_bar,fyk,list_of_mu_terms_k_k,high_wk_ind,high_wk,show=print_control_cost)
            saved_data[k,5] = uk
            #print(uk)
        else: 
            uk = 0


        #Advance truth k -> k+1
        xkp1_bar,xkp1_tild,wk = advance_simulation_truth(k,Phi,B,Gamma,beta,xk_bar,xk_tild,uk)
        saved_data[k,4] = wk
        if abs(wk) > 5:
            high_wk_ind = k
            high_wk = wk

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
        lw = 5
        lfz = 20
        lfzl = 15
        lt = 20

        plt.figure(figsize=(13,20)).subplots(5,1,height_ratios=[4,2.5,2,4,3])
        plt.subplot(511)
        plt.plot(range(0,time),saved_data[:,0]+saved_data[:,1],color='midnightblue',label="truth",lw=lw)
        # plt.plot(range(0,time),saved_data[:,0],color='purple',label="xk_bar")
        plt.plot(range(0,time),saved_data[:,2],color='red',label="measurement",lw=lw-1)
        plt.plot(range(0,time),saved_estimator[:,0],color='green',label = "estimate k|k ",lw=lw-2)
        # plt.plot(range(0,time),saved_estimator[:,2],color='orange',label = "estimate k+1|k",linestyle="dashed")
        # plt.plot(range(0,time),np.zeros((len(range(0,time)))),linestyle="dashed")
        plt.legend(fontsize=lfzl,frameon=False,loc=3,bbox_to_anchor=(0,-0.1))
        plt.minorticks_on()
        plt.grid(which="minor",color="gainsboro")
        plt.grid(which="major")
        plt.ylabel(r'$ x_k, z_k, \hat{x}_{k|k} $',fontsize=lfz)
        plt.xlim(0,100)
        plt.ylim(-6.1,3.4)
        plt.tick_params(axis='y', which='major', labelsize=lt)
        plt.tick_params(axis='x',labelbottom='off')
        plt.gca().axes.xaxis.set_ticklabels([])


        plt.subplot(512)
        plt.plot(range(0,time),saved_data[:,3],color='midnightblue',label=r"$w_k$",lw=lw)
        plt.plot(range(0,time),saved_data[:,4],color='red',label=r"$v_k$",lw=lw-1.5)
        #plt.plot(range(0,time),np.zeros((len(range(0,time)))),linestyle="dashed")
        plt.legend(fontsize=lfzl,frameon=False,loc=3,bbox_to_anchor=(0,-0.15))
        plt.minorticks_on()
        plt.grid(which="minor",color="gainsboro")
        plt.grid(which="major")
        plt.ylabel(r"$w_k, v_k$",fontsize=lfz)
        plt.xlim(0,100)
        plt.ylim(-6,3)
        plt.tick_params(axis='y', which='major', labelsize=lt,labelbottom=False,)
        plt.gca().axes.xaxis.set_ticklabels([])
        

        plt.subplot(513)
        plt.plot(range(0,time),saved_data[:,5],color='midnightblue',lw=lw)
        plt.legend(fontsize=lfzl,frameon=False)
        plt.minorticks_on()
        plt.grid(which="minor",color="gainsboro")
        plt.grid(which="major")
        #plt.xlabel('Time step (k)',fontsize=lt)
        plt.ylabel(r"$u_k$",fontsize=lfz)
        plt.xlim(0,100)
        plt.tick_params(axis='y', which='major', labelsize=lt,labelbottom=False,)
        plt.gca().axes.xaxis.set_ticklabels([])
        

        #plt.figure()
        plt.subplot(514)
        plt.plot(range(0,time),saved_lyap_functions[:,0],color='midnightblue',lw=lw,label=r"$E[V(x_k)|y_k]$")
        plt.plot(range(0,time-1),saved_lyap_functions[:-1,1],color='red',lw=lw-1.5,label=r"$E[V(x_{k+1})|y_k]$")
        plt.legend(fontsize=lfzl,frameon=False)
        plt.minorticks_on()
        plt.grid(which="minor",color="gainsboro")
        plt.grid(which="major")
        plt.ylabel("Lyapunov \n Expectation",fontsize=lfz-2)
        plt.tick_params(axis='y', which='major', labelsize=lt,labelbottom=False,)
        plt.xlim(0,100)
        plt.gca().axes.xaxis.set_ticklabels([])

        # plt.subplot(615)
        # plt.plot(range(0,time-1),saved_lyap_functions[:-1,1],color='midnightblue',lw=lw)
        # plt.legend(fontsize=lfzl,frameon=False)
        # plt.minorticks_on()
        # plt.grid(which="minor",color="gainsboro")
        # plt.grid(which="major")
        # plt.ylabel(r"$E[V(x_{k+1})|y_k]$",fontsize=lfz-2)
        # plt.tick_params(axis='y', which='major', labelsize=lt)

        plt.subplot(515)
        plt.plot(range(0,time-1),saved_lyap_functions[:-1,1]-saved_lyap_functions[:-1,0],color='midnightblue',lw=lw)
        plt.legend(fontsize=lfzl,frameon=False)
        plt.minorticks_on()
        plt.grid(which="minor",color="gainsboro")
        plt.grid(which="major")
        plt.xlabel('Time step (k)',fontsize=lt)
        #plt.ylabel(r"$E[V(x_{k+1})|y_k]-E[V(x_k)|y_k]$",fontsize=lfz)
        plt.ylabel("Drift",fontsize=lfz)
        plt.tick_params(axis='both', which='major', labelsize=lt)
        plt.xlim(0,100)

        #plt.figure(figsize=(10,6))
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

def calc_secondmoment_from_listofterms(k,list_of_terms,fyk,xk_hat):

    E2 = sum([t1.ci*(t1.sigmai**2-t1.omegai**2)-2*t1.di*t1.sigmai*t1.omegai for t1 in list_of_terms])

    return E2/fyk - xk_hat**2

def calc_control(n,k,control_steps,Phi,B,beta,Gamma,eta_r,theta,xk_bar,fyk,list_of_terms,high_wk_ind,high_wk,show):
    if k-high_wk_ind < 10:
        len_u = min(max(1000,3*int(high_wk)),10000)
        range = max(50,1.5*abs(high_wk))
    else:
        len_u = 200
        range = 10
    
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
                
            cost_val_der[:,i] = np.real(der_M* term_cost + der_term*M_cost)
            
        cost = np.sum(cost_val,axis=1) /((2*pi)**n * fyk)
        ind_max = np.argmax(cost)

        cost_der = np.sum(cost_val_der,axis=1)

        u_guess = u_arr[ind_max]

        res = optimize_1d(u_guess,list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta)

        if show :
            plt.figure()
            plt.subplot(211)
            plt.plot(u_arr,cost,color='blue',label="cost")
            plt.scatter(res.x,-res.fun/(2*pi*fyk),c='r',marker='o')
            plt.legend()
            plt.xlabel('u')

            plt.subplot(212)
            plt.plot(u_arr,cost_der,color='blue',label="cost derivative")
            plt.legend()
            plt.xlabel('u')

            plt.show
        
        
        return res.x
    
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
        
        uk_ballpark = uk_arr[ind_max]
        uk1_ballpark = uk1_arr[ind_max]

        u_guess = np.array([uk_ballpark,uk1_ballpark])
        res=optimize_2d(u_guess,list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta)

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot_wireframe(uk_arr, uk1_arr, cost, rstride=10, cstride=10)
            ax.scatter(res.x[0],res.x[1],-res.fun/((2*pi)**n * fyk),marker='o',c='r')
            ax.set_xlabel("u k")
            ax.set_ylabel("u k+1")
            ax.set_zlabel("cost")

            plt.show()

        return res.x[0]


def optimize_2d(u_guess,list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta):
    #res = minimize(eval_cost, u0, method='nelder-mead',args=(list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta), options={'xatol': 1e-8, 'disp': True})
    #rest = minimize(eval_2dcost, u_guess, method='BFGS', jac=eval_2dgradient, args = (list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta),options={'disp': False})
    rest = minimize(eval_2dgradient, u_guess, method='BFGS', jac=True, args = (list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta),options={'disp': False})
    
    return rest

def optimize_1d(u_guess,list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta):
    rest = minimize(eval_1dcostandgradient, u_guess, method='BFGS', jac=True, args = (list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta),options={'disp': False})
    return rest

def eval_2dcost(u_vec,list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta):
    uk = u_vec[0]
    uk1 = u_vec[1]
    cost_vec = np.zeros((len(list_of_terms)))
    for i,term in enumerate(list_of_terms):
        omegai = term.omegai
        sigmai = term.sigmai
        ci = term.ci
        di = term.di
            
        term_cost = (ci-1j*di)/(omegai*Phi**2 + 1j*sigmai*Phi**2 + eta_r + 1j*(Phi**2*xk_bar + Phi*B*uk + B*uk1) + beta*Gamma + beta*Phi*Gamma) + (ci+1j*di)/(omegai*Phi**2-1j*sigmai*Phi**2+eta_r-1j*(Phi**2*xk_bar+Phi*B*uk+B*uk1)+beta*Gamma+beta*Phi*Gamma)
        M_cost = theta/pi /(uk**2+theta**2) * theta/pi /(uk1**2+theta**2)

        cost_vec[i] = np.real(term_cost)*M_cost
    cost = np.sum(cost_vec,axis=0)

    return -cost

def eval_1dcostandgradient(u,list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta):
    cost_vec = np.zeros((len(list_of_terms)))
    g_vec = np.zeros((len(list_of_terms)))
    for i,term in enumerate(list_of_terms):
        omegai = term.omegai
        sigmai = term.sigmai
        ci = term.ci
        di = term.di
            
        num = ci*(2*omegai*Phi+2*eta_r+2*beta*Gamma) + di*(-2*sigmai*Phi - 2*(Phi*xk_bar + B*u))
        den = omegai**2*Phi**2+2*omegai*Phi*eta_r+2*omegai*Phi*beta*Gamma+sigmai**2*Phi**2 + 2*sigmai*Phi*(Phi*xk_bar+B*u) +eta_r**2+2*eta_r*beta*Gamma+(Phi*xk_bar+B*u)**2+beta**2*Gamma**2
        term_cost = num/den 
        term_cost_2 = (ci-1j*di)/(omegai*Phi+1j*sigmai*Phi+eta_r+1j*(Phi*xk_bar+B*u)+beta*Gamma) + (ci+1j*di)/(omegai*Phi-1j*sigmai*Phi+eta_r-1j*(Phi*xk_bar+B*u)+beta*Gamma)

        M_cost = theta/pi /(u**2+theta**2)
                
        cost_vec[i] = np.real(term_cost * M_cost)[0]
            
        der_term = (ci-1j*di)*(1j*B)/(omegai*Phi+1j*sigmai*Phi+eta_r+1j*(Phi*xk_bar+B*u)+beta*Gamma)**2+ (ci+1j*di)*(-1j*B)/(omegai*Phi-1j*sigmai*Phi+eta_r-1j*(Phi*xk_bar+B*u)+beta*Gamma)**2
        der_M = theta/pi * 2*u /(u**2+theta**2)**2
                
        g_vec[i] = np.real(der_M* term_cost + der_term*M_cost)[0]
        
    cost = np.sum(cost_vec)
    cost_der = np.sum(g_vec)

    return (-cost,-cost_der)

def eval_2dgradient(u_vec,list_of_terms,Phi,eta_r,xk_bar,B,beta,Gamma,theta):

    cost_vec = np.zeros((len(list_of_terms)))
    g_vec = np.zeros((2,len(list_of_terms)))
    uk = u_vec[0]
    uk1 =  u_vec[1]
    for i,term in enumerate(list_of_terms):
        omegai = term.omegai
        sigmai = term.sigmai
        ci = term.ci
        di = term.di
            
        term_cost = (ci-1j*di)/(omegai*Phi**2 + 1j*sigmai*Phi**2 + eta_r + 1j*(Phi**2*xk_bar + Phi*B*uk + B*uk1) + beta*Gamma + beta*Phi*Gamma) + (ci+1j*di)/(omegai*Phi**2-1j*sigmai*Phi**2+eta_r-1j*(Phi**2*xk_bar+Phi*B*uk+B*uk1)+beta*Gamma+beta*Phi*Gamma)
        M_cost = theta/pi /(uk**2+theta**2) * theta/pi /(uk1**2+theta**2)

        dM_duk = theta/pi*-2*uk /(uk**2+theta**2)**2 * theta/pi /(uk1**2+theta**2)
        dM_duk1 = theta/pi /(uk**2+theta**2) * theta/pi *-2*uk1/(uk1**2+theta**2)**2

        dterm_duk = (-1j*Phi*B)*(ci-1j*di)/(omegai*Phi**2 + 1j*sigmai*Phi**2 + eta_r + 1j*(Phi**2*xk_bar + Phi*B*uk + B*uk1) + beta*Gamma + beta*Phi*Gamma)**2 + (1j*Phi*B)*(ci+1j*di)/(omegai*Phi**2-1j*sigmai*Phi**2+eta_r-1j*(Phi**2*xk_bar+Phi*B*uk+B*uk1)+beta*Gamma+beta*Phi*Gamma)**2
        dterm_duk1 = (-1j*B)*(ci-1j*di)/(omegai*Phi**2 + 1j*sigmai*Phi**2 + eta_r + 1j*(Phi**2*xk_bar + Phi*B*uk + B*uk1) + beta*Gamma + beta*Phi*Gamma)**2 + (1j*B)*(ci+1j*di)/(omegai*Phi**2-1j*sigmai*Phi**2+eta_r-1j*(Phi**2*xk_bar+Phi*B*uk+B*uk1)+beta*Gamma+beta*Phi*Gamma)**2

        g_vec[0,i] = dM_duk*term_cost+M_cost*dterm_duk
        g_vec[1,i] = dM_duk1*term_cost+M_cost*dterm_duk1
        cost_vec[i] = np.real(term_cost)*M_cost

    cost = np.sum(cost_vec)
    g_vec = np.sum(g_vec,axis=1)
    return (-cost,-g_vec)


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
    multiple_sim = False
    
    if multiple_sim:
        run_simulation_MC(runs=10000,time=100,print_individ_plots=False,control_steps=0,save_data_to_text=True,plot_all_runs=True,filename = 'MC_nocontrol_stable')
    else:
        np.random.seed(seed=233423)
        define_simulation(time=100,Phi=1.05, H=1,alpha=0.5,beta=0.02,gamma=0.1,eta_r=0.7,control_steps=2,print_control_cost=False)

    #og_set = np.seterr({'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'})
    