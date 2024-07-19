import cauchy_estimator as ce 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg', force=True)
import sys, os
import pickle
# The linear target tracking scenario with poorly modelled noises using Nominal KF, Optimal KF, and Cauchy Estimator 
# Nonlnear case also
np.random.seed()

SUM_INIT_H = False

def nonlin_h(x):
    return np.array( [(x[0]**2 + x[2]**2)**0.5, np.arctan2(x[2], x[0])] )    

# H Summation Reinitialization
def H_reinit_func(cauchyEsts, zk, best_idx, idx_min, step_k, other_params):

    U = other_params

    # Get Max / Min Windows
    cauchyEstFull = cauchyEsts[best_idx]
    cauchyEstReinit = cauchyEsts[idx_min]
    
    # Grab their pyhandles
    pyducFull = cauchyEstFull.get_pyduc()
    pyducReinit = cauchyEstReinit.get_pyduc()

    # Store old and Create new dynamics for reinitialization
    xhat,Phat = cauchyEstFull.get_last_mean_cov()
    H_orig = U @ np.array([1,0,0,0, 0,0,1.0,0]).reshape((2,4))
    Hsum = np.vstack(( np.zeros((1,4)), np.sum(H_orig, axis=0) ))
    gamma_orig = pyducFull.cget_gamma()
    sum_gamma = np.sum( gamma_orig )
    gamsum = np.array([gamma_orig[0], sum_gamma])

    # Set the dynamics temporarily for the min window, and then back
    zksum = np.sum(zk)
    A0,p0,b0 = ce.speyers_window_init(xhat, Phat, Hsum[1], gamsum[1], zksum ) # Create
    pyducReinit.cset_H(Hsum)
    pyducReinit.cset_gamma(gamsum)
    cauchyEstReinit.reset_with_last_measurement( zksum, A0, p0, b0, None)  # Run 
    pyducReinit.cset_H(H_orig)     # Reset
    pyducReinit.cset_gamma(gamma_orig)    # Reset


# H Summation Reinitialization
def H_reinit_func_nl(cauchyEsts, zk, best_idx, idx_min, step_k, other_params):

    U = other_params

    # Get Max / Min Windows
    cauchyEstFull = cauchyEsts[best_idx]
    cauchyEstReinit = cauchyEsts[idx_min]
    
    # Grab their pyhandles
    pyducFull = cauchyEstFull.get_pyduc()
    pyducReinit = cauchyEstReinit.get_pyduc()

    # Store old and Create new dynamics for reinitialization
    xhat,Phat = cauchyEstFull.get_last_mean_cov()
    H_orig = ce.cd4_gvf(xhat, nonlin_h)
    H_orig[0,1] = 0
    H_orig[0,3] = 0
    H_orig[1,1] = 0
    H_orig[1,3] = 0
    Hsum = np.vstack(( np.zeros((1,4)), np.sum(H_orig, axis=0) ))
    gamma_orig = pyducFull.cget_gamma()
    sum_gamma = np.sum( gamma_orig )
    gamsum = np.array([gamma_orig[0], sum_gamma])

    xbar = cauchyEstFull._xbar[4:]
    zbar = nonlin_h(xbar)
    dz = zk - zbar
    dx = xhat - xbar
    
    # Set the dynamics temporarily for the min window, and then back
    dzksum = np.sum(dz)
    zksum = np.sum(zk)
    A0,p0,b0 = ce.speyers_window_init(dx, Phat, Hsum[1], gamsum[1], dzksum ) # Create
    pyducReinit.cset_H(Hsum)
    pyducReinit.cset_gamma(gamsum)
    global SUM_INIT_H
    SUM_INIT_H = True
    cauchyEstReinit.reset_with_last_measurement( zksum, A0, p0, b0, xbar)  # Run
    SUM_INIT_H = False 
    pyducReinit.cset_H(H_orig)     # Reset
    pyducReinit.cset_gamma(gamma_orig)    # Reset
    
# OVERALL THE ESTIMATOR IS SEEN TO PERFORM FINE FOR THE ESTIMATION TASK, HOWEVER
# THIS FUNCTION HAS A VERY INTERESTING ERROR WITH REGARDS TO THE 2D MARGINAL CPDF of the 4D System 
# Keeping V = [1e4,1e2;1e2,1e4] yields errors on step 6 
# Keeping V = [1e3,1e2;1e2,1e4] yields errors on step 4 for states (3,4) but not states (1,2)....at next step both marginals have errors (NEGATIVE CPDF VALUES)
def linear_target_track(N, use_scen, num_windows, V0_scale_factor, KNOB_Q, KNOB_V, fig_path):
    T = 1.0 # seconds
    W1 = 1.0 # PSD of noise channel 1
    W2 = 1.0 # PSD of noise channel 2
    #N = 300 # steps

    # Blocks
    T2 = np.array([1.0, T, 0, 1.0]).reshape((2,2))
    z2 = np.zeros(2)
    Z2 = np.zeros((2,2))
    QT2 = np.array([T**3/3, T**2/2, T**2/2, T]).reshape((2,2))
    z4 = np.zeros(4)
    I4 = np.eye(4)

    # True Initial Simulation State
    x0 = np.array([5, -50, 10, -75])  # [m, m/s, m, m/s] -> [x,vx,y,vy]-> [posx,velx,posy,vely]
    
    # Nominal Kalman Filter
    Phi = np.vstack(( np.hstack((T2,Z2)), np.hstack((Z2,T2))  ))
    H = np.array([[1.0,0,0,0], [0,0,1,0]])
    Q0 = np.vstack(( np.hstack(( W1 * QT2, Z2)), np.hstack((Z2, W2 * QT2))  ))
    V0 = np.array([1e4,1e2,1e2,1e4]).reshape((2,2)) / V0_scale_factor 
    P0_kf = np.diag(np.array([100,1,100,1.0]))
    x0_kf = np.random.multivariate_normal(x0, P0_kf)

    # Cauchy Estimator 
    #Gamc = np.array([[0,1.0,0,0],[0,0,0,1.0]]).T
    Gamk = np.array([[T**2/2,T,0,0],[0,0,T**2/2,T]]).T
    beta0 = np.array([W1**0.5 / 1.3898, W2**0.5 / 1.3898 ]) / 1.7
    _, U = np.linalg.eig(V0)
    Hbar = U @ H 
    Vbar = U @ V0 @ U.T 
    gambar0 = np.diag(Vbar)**0.5/1.3898 / 1.7
    b0 = x0_kf.copy()
    p0 = np.diag(P0_kf)**0.5 / 1.3898
    A0 = np.eye(4)
    ce.set_tr_search_idxs_ordering([3,1,2,0])
    swm_debug_print = True
    win_debug_print = True
    #cauchyEst = ce.PySlidingWindowManager('lti', num_windows, swm_debug_print=swm_debug_print, win_debug_print=win_debug_print)
    #cauchyEst.initialize_lti(A0, p0, b0, Phi, None, Gamk, beta0, Hbar, gambar0, reinit_func = None) # ADD REINIT FUNC -> H_reinit_func
    cauchyEst = ce.PyCauchyEstimator('lti', 6, win_debug_print)
    cauchyEst.initialize_lti(A0,p0,b0,Phi,None,Gamk,beta0,Hbar,gambar0)

    # Run Simulation 

    # Nominal KF Log
    xs_kfn = [] 
    Ps_kfn = [] 
    # Optimal KF Log
    xs_kfo = [] 
    Ps_kfo = [] 
    # Nominal Cauchy Log
    xs_ce = [] 
    Ps_ce = []
    xs_ce_avg = [] 
    Ps_ce_avg = []

    x_kfn = x0_kf.copy() 
    P_kfn = P0_kf.copy()
    x_kfo = x0_kf.copy() 
    P_kfo = P0_kf.copy() 

    # Scenarios 
    s0_q = lambda k : Q0.copy()
    s0_v = lambda k : V0.copy()
    s1_q = lambda k : np.abs(1 + KNOB_Q*np.sin( (np.pi * k) / N))*Q0 # Q0.copy() #
    s2_q = lambda k : Q0.copy() if k < 100 else KNOB_Q*Q0 if k < 200 else Q0.copy() 
    s1_v = lambda k : np.abs(1 + KNOB_V*(1-np.sin( (np.pi * k) / N)))*V0 #V0.copy() #
    s2_v = lambda k : V0.copy() if k < 200 else KNOB_V*V0 

    if( use_scen == "0" ):
        qscen = s0_q
        vscen = s0_v
    elif(use_scen == "1a"):
        qscen = s1_q
        vscen = s0_v
    elif(use_scen == "1b"):
        qscen = s0_q
        vscen = s1_v
    elif(use_scen == "1c"):
        qscen = s1_q
        vscen = s1_v
    elif(use_scen == "2"):
        qscen = s2_q
        vscen = s2_v
    else:
        print("Use Scenario {} not valid [0,1a,1b,1c,2]. Exiting!".format(use_scen))
        exit(1)

    xks = [x0.copy()] 
    vks = [np.random.multivariate_normal(z2, vscen(0))] 
    zks = [H @ xks[0] + vks[0]]
    wks = []
    # Simulation Realization Loop
    xk = x0.copy()
    for k in range(N):
        wk = np.random.multivariate_normal( z4, qscen(k) ) #np.random.standard_cauchy(4) * np.diag(qscen(k))**0.5 #
        xk = Phi @ xk + wk
        vk = np.random.multivariate_normal( z2, vscen(k+1) )
        zk = H @ xk + vk 
        xks.append(xk)
        zks.append(zk)
        vks.append(vk)
        wks.append(wk)

    # Estimation Loop
    show_best_win = True # FALSE!!
    Qn = Q0.copy() 
    Vn = V0.copy()
    for k in range(N+1):
        Qo = qscen(k-1)
        Vo = vscen(k)
        zk = zks[k]
        #x_ce, P_ce, x_ce_avg, P_ce_avg = cauchyEst.step( U @ zk, reinit_args=U)
        x_ce, P_ce = cauchyEst.step( U @ zk)
        x_ce_avg, P_ce_avg = x_ce, P_ce
        
        xs_ce.append(x_ce)
        Ps_ce.append(P_ce)
        xs_ce_avg.append(x_ce_avg)
        Ps_ce_avg.append(P_ce_avg)
        
        # Time Prop KFs
        if k > 0: 
            x_kfn = Phi @ x_kfn
            P_kfn = Phi @ P_kfn @ Phi.T + Qn
            x_kfo = Phi @ x_kfo
            P_kfo = Phi @ P_kfo @ Phi.T + Qo
        # Measurement Update
        K_kfn = P_kfn @ H.T @ np.linalg.inv(H @ P_kfn @ H.T + Vn)
        x_kfn += K_kfn @ (zk - H @ x_kfn)
        P_kfn = (I4 - K_kfn @ H) @ P_kfn @ (I4 - K_kfn @ H).T + K_kfn @ Vn @ K_kfn.T
        K_kfo = P_kfo @ H.T @ np.linalg.inv(H @ P_kfo @ H.T + Vo)
        x_kfo += K_kfo @ (zk - H @ x_kfo)
        P_kfo = (I4 - K_kfo @ H) @ P_kfo @ (I4 - K_kfo @ H).T + K_kfo @ Vo @ K_kfo.T
        # Log Step
        xs_kfn.append(x_kfn)
        Ps_kfn.append(P_kfn)
        xs_kfo.append(x_kfo)
        Ps_kfo.append(P_kfo)

        if show_best_win:
            if k == 0:
                P_ce += np.eye(4)
            #best_win_idx = np.argmax(cauchyEst.win_counts)
            PX,VX,FXZ = cauchyEst.get_marginal_2D_pointwise_cpdf( #cauchyEst.cauchyEsts[best_win_idx].get_marginal_2D_pointwise_cpdf( #cauchyEst.get_marginal_2D_pointwise_cpdf(
                0,1, 
                x_ce[0] - 3*P_ce[0,0]**0.5, 
                x_ce[0] + 3*P_ce[0,0]**0.5, 
                6*P_ce[0,0]**0.5 / 400, 
                x_ce[1] - 3*P_ce[1,1]**0.5, 
                x_ce[1] + 3*P_ce[1,1]**0.5, 
                6*P_ce[1,1]**0.5 / 400, 
                log_dir = None)
            PY,VY,FYZ = cauchyEst.get_marginal_2D_pointwise_cpdf( #cauchyEst.cauchyEsts[best_win_idx].get_marginal_2D_pointwise_cpdf( #cauchyEst.get_marginal_2D_pointwise_cpdf(
                2,3, 
                x_ce[2] - 3*P_ce[2,2]**0.5, 
                x_ce[2] + 3*P_ce[2,2]**0.5, 
                6*P_ce[2,2]**0.5 / 400, 
                x_ce[3] - 3*P_ce[3,3]**0.5, 
                x_ce[3] + 3*P_ce[3,3]**0.5, 
                6*P_ce[3,3]**0.5 / 400, 
                log_dir = None)
            z_low = -0.2
            '''
            fig1 = plt.figure() 
            ax1 = fig1.add_subplot(projection='3d')
            ax1.set_title("Position/Velocity in X")
            ax1.set_xlabel('Position in x (meters)')
            ax1.set_ylabel('Velocity in x (meters)')
            ax1.plot_wireframe(PX,VX, FXZ, color='k') 
            ell_px_ce, ell_vx_ce = ce.get_2d_covariance_ellipsoid(x_ce[0:2], P_ce[0:2,0:2], 0.95, num_points = 100)
            ell_px_kfo, ell_vx_kfo = ce.get_2d_covariance_ellipsoid(x_kfo[0:2], P_kfo[0:2,0:2], 0.95, num_points = 100)
            ell_px_kfn, ell_vx_kfn = ce.get_2d_covariance_ellipsoid(x_kfn[0:2], P_kfn[0:2,0:2], 0.95, num_points = 100)
            ax1.plot(ell_px_ce, ell_vx_ce, z_low, color='k')
            ax1.plot(ell_px_kfo, ell_vx_kfo, z_low, color='g')
            ax1.plot(ell_px_kfn, ell_vx_kfn, z_low, color='b')
            ax1.scatter(xks[k][0], xks[k][1], z_low, color='r', marker='*', s=50)

            fig2 = plt.figure() 
            ax2 = fig2.add_subplot(projection='3d')
            ax2.set_title("Position/Velocity in Y")
            ax2.set_xlabel('Position in y (meters)')
            ax2.set_ylabel('Velocity in y (meters)')
            ax2.plot_wireframe(PY,VY, FYZ, color='k') 
            quantile = 0.70
            ell_py_ce, ell_vy_ce = ce.get_2d_covariance_ellipsoid(x_ce[2:4], P_ce[2:4,2:4], quantile, num_points = 100)
            ell_py_kfo, ell_vy_kfo = ce.get_2d_covariance_ellipsoid(x_kfo[2:4], P_kfo[2:4,2:4], quantile, num_points = 100)
            ell_py_kfn, ell_vy_kfn = ce.get_2d_covariance_ellipsoid(x_kfn[2:4], P_kfn[2:4,2:4], quantile, num_points = 100)
            ax2.plot(ell_py_ce, ell_vy_ce, z_low, color='k')
            ax2.plot(ell_py_kfo, ell_vy_kfo, z_low, color='g')
            ax2.plot(ell_py_kfn, ell_vy_kfn, z_low, color='b')
            ax2.scatter(xks[k][2], xks[k][3], z_low, color='r', marker='*', s=50)
            plt.show()
            plt.close('all')
            '''
    
    # Lists -> NP Arrays 
    xks = np.array(xks)
    zks = np.array(zks)
    vks = np.array(vks)
    wks = np.array(wks)
    xs_kfn = np.array(xs_kfn)
    Ps_kfn = np.array(Ps_kfn)
    xs_kfo = np.array(xs_kfo)
    Ps_kfo = np.array(Ps_kfo)

    xs_ce = np.array(xs_ce)
    Ps_ce = np.array(Ps_ce)
    xs_ce_avg = np.array(xs_ce_avg)
    Ps_ce_avg = np.array(Ps_ce_avg)

    # Now Plot Results
    Ts = np.arange(N+1)*T 
    # State History Plot 
    '''
    plt.figure()
    plt.suptitle("State History Plot\nTrue State=Red, Opt KF=green, Nom KF=blue, Nom Cauchy=magenta")
    plt.subplot(2,1,1)
    plt.plot(xks[:, 0], xks[:, 2], 'r')
    #plt.scatter(xks[:, 0], xks[:, 1], color='r')
    plt.plot(xs_kfn[:, 0], xs_kfn[:, 2], 'b')
    #plt.scatter(xs_kfn[:, 0], xs_kfn[:, 1], color='g')
    plt.plot(xs_kfo[:, 0], xs_kfo[:, 2], 'g')
    #plt.scatter(xs_kfo[:, 0], xs_kfo[:, 1], color='b')
    plt.ylabel("Pos Y")
    plt.xlabel("Pos X")
    plt.subplot(2,1,2)
    plt.plot(xks[:, 1], xks[:, 3], 'r')
    #plt.scatter(xks[:, 2], xks[:, 3], color='r')
    plt.plot(xs_kfn[:, 1], xs_kfn[:, 3], 'b')
    #plt.scatter(xs_kfn[:, 2], xs_kfn[:, 3], color='g')
    plt.plot(xs_kfo[:, 1], xs_kfo[:, 3], 'g')
    #plt.scatter(xs_kfo[:, 2], xs_kfo[:, 3], color='b')
    plt.ylabel("Vel Y")
    plt.xlabel("Vel X")
    '''

    # State Error Plot
    scale = 1
    plt.figure(figsize=(12,12))
    plt.suptitle("State Error Plot\nOpt KF=green, Nom KF=blue, Nom Cauchy=magenta")
    one_sig_kfn = np.array([scale*np.diag(P)**0.5 for P in Ps_kfn])
    one_sig_kfo = np.array([scale*np.diag(P)**0.5 for P in Ps_kfo])
    one_sig_ce = np.array([scale*np.diag(P)**0.5 for P in Ps_ce])
    one_sig_ce_avg = np.array([scale*np.diag(P)**0.5 for P in Ps_ce_avg])

    # Some Analysis 
    states = 4
    conf_percent = 0.70
    from scipy.stats import chi2 
    s = chi2.ppf(conf_percent, states)
    ae_kfo = np.mean(np.abs(xks[2:] - xs_kfo[2:]), axis = 0)
    ae_kfn = np.mean(np.abs(xks[2:] - xs_kfn[2:]), axis = 0)
    ae_ce = np.mean(np.abs(xks[2:] - xs_ce[2:]), axis = 0)
    ae_cea = np.mean(np.abs(xks[2:] - xs_ce_avg[2:]), axis = 0)
    kfo_bound_realiz = np.sum([ ( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s for xt, xh, P in zip(xks[2:], xs_kfo[2:], Ps_kfo[2:]) ]) / (xks.shape[0]-2)
    kfn_bound_realiz = np.sum([ ( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s for xt, xh, P in zip(xks[2:], xs_kfn[2:], Ps_kfn[2:]) ]) / (xks.shape[0]-2)
    ce_bound_realiz = np.sum([ (( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s) if np.all(np.linalg.eig(P)[0]>0) else 0 for xt, xh, P in zip(xks[2:], xs_ce[2:], Ps_ce[2:]) ]) / (xks.shape[0]-2)
    cea_bound_realiz = np.sum([ (( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s) if np.all(np.linalg.eig(P)[0]>0) else 0 for xt, xh, P in zip(xks[2:], xs_ce_avg[2:], Ps_ce_avg[2:]) ]) / (xks.shape[0]-2)

    print("The Optimal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_kfo, kfo_bound_realiz) )
    print("The Nominal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_kfn, kfn_bound_realiz) )
    print("The (Best Window) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_ce, ce_bound_realiz) )
    print("The (Averaged) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_cea, cea_bound_realiz) )

    plt.subplot(4,1,1)
    plt.plot(Ts, xks[:,0] - xs_kfo[:,0], 'g')
    plt.plot(Ts, one_sig_kfo[:,0], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,0], 'g--')
    plt.plot(Ts, xks[:,0] - xs_kfn[:,0], 'b')
    plt.plot(Ts, one_sig_kfn[:,0], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,0], 'b--')
    #plt.plot(Ts, xks[:,0] - xs_ce[:,0], 'm')
    #plt.plot(Ts, one_sig_ce[:,0], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,0], 'm--')
    plt.plot(Ts, xks[:,0] - xs_ce_avg[:,0], 'k')
    plt.plot(Ts, one_sig_ce_avg[:,0], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,0], 'k--')
    plt.ylabel("Pos X")
    plt.subplot(4,1,2)
    plt.plot(Ts, xks[:,1] - xs_kfo[:,1], 'g')
    plt.plot(Ts, one_sig_kfo[:,1], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,1], 'g--')
    plt.plot(Ts, xks[:,1] - xs_kfn[:,1], 'b')
    plt.plot(Ts, one_sig_kfn[:,1], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,1], 'b--')
    #plt.plot(Ts, xks[:,1] - xs_ce[:,1], 'm')
    #plt.plot(Ts, one_sig_ce[:,1], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,1], 'm--')
    plt.plot(Ts, xks[:,1] - xs_ce_avg[:,1], 'k')
    plt.plot(Ts, one_sig_ce_avg[:,1], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,1], 'k--')
    plt.ylabel("Vel X")
    plt.subplot(4,1,3)
    plt.plot(Ts, xks[:,2] - xs_kfo[:,2], 'g')
    plt.plot(Ts, one_sig_kfo[:,2], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,2], 'g--')
    plt.plot(Ts, xks[:,2] - xs_kfn[:,2], 'b')
    plt.plot(Ts, one_sig_kfn[:,2], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,2], 'b--')
    #plt.plot(Ts, xks[:,2] - xs_ce[:,2], 'm')
    #plt.plot(Ts, one_sig_ce[:,2], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,2], 'm--')
    plt.plot(Ts, xks[:,2] - xs_ce_avg[:,2], 'k')
    plt.plot(Ts, one_sig_ce_avg[:,2], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,2], 'k--')
    plt.ylabel("Pos Y")
    plt.subplot(4,1,4)
    plt.plot(Ts, xks[:,3] - xs_kfo[:,3], 'g')
    plt.plot(Ts, one_sig_kfo[:,3], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,3], 'g--')
    plt.plot(Ts, xks[:,3] - xs_kfn[:,3], 'b')
    plt.plot(Ts, one_sig_kfn[:,3], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,3], 'b--')
    #plt.plot(Ts, xks[:,3] - xs_ce[:,3], 'm')
    #plt.plot(Ts, one_sig_ce[:,3], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,3], 'm--')
    plt.plot(Ts, xks[:,3] - xs_ce_avg[:,3], 'k')
    plt.plot(Ts, one_sig_ce_avg[:,3], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,3], 'k--')
    plt.ylabel("Vel Y")
    plt.xlabel("Time (sec)")
    plt.show()
    plt.savefig(fname = fig_path)
    
    ret_dic = {"xks": xks,
               "zks" : zks,
               "vks" : vks,
               "wks" : wks,
               "xs_kfo" : xs_kfo, 
               "Ps_kfo" : Ps_kfo,
               "ae_kfo" : ae_kfo,
               "kfo_bound_realiz" : kfo_bound_realiz,
               "xs_kfn" : xs_kfn, 
               "Ps_kfn" : Ps_kfn,
               "ae_kfn" : ae_kfn,
               "kfn_bound_realiz" : kfn_bound_realiz,
               "xs_ce" : xs_ce, 
               "Ps_ce" : Ps_ce,
               "ae_ce" : ae_ce,
               "ce_bound_realiz" : ce_bound_realiz,
               "xs_ce_avg" : xs_ce_avg, 
               "Ps_ce_avg" : Ps_ce_avg,
               "ae_cea" : ae_cea,
               "cea_bound_realiz" : cea_bound_realiz
               }
    return ret_dic


# Eh..
class VBayes():

    def __init__(self, x0, P0, f, F, h, H, Qhat = np.eye(4), V = 10000 * np.eye(2), rho_R = 0.98, tau_P = 3.0, tau_R = 3.0, is_lin = True):
        self.f = f 
        self.F = F 
        self.h = h 
        self.H = H 

        self.n = f(x0).size
        self.m = h(x0).size
        
        self.x = x0.copy() 
        self.P = P0.copy()
        self.v = tau_R + self.m + 1
        self.u = self.n + 2
        self.Qhat = Qhat.copy()
        self.V = V.copy() * tau_R
        self.rho_R = rho_R
        self.tau_P = tau_P
        self.tau_R = tau_R
        self.is_lin = is_lin
        self.eps = 1e-5

    # Time Prop
    def pxk_g_ykm1(self):
        self.last_x = self.x.copy() 
        self.last_P = self.P.copy() 
        Phi = self.F(self.x)
        self.Phi = Phi.copy() 

        self.x = self.f(self.x)
        self.P = Phi @ self.P @ Phi.T + self.Qhat
        self.u = self.n + self.tau_P + 1
        self.U = self.tau_P * self.P.copy()
        self.v = self.rho_R*(self.v - self.m - 1) + self.m + 1 
        self.V = self.rho_R*self.V.copy()

    def lin_nat_param_update(self, yk):
        # Get lam_R 
        H = self.H(self.x) 

        it = 0
        xbar = self.x.copy()
        Pbar = self.P.copy()
        vbar = np.array([self.v]).reshape(-1).copy()
        ubar = np.array([self.u]).reshape(-1).copy()
        Ubar = self.U.copy()
        Vbar = self.V.copy()

        for it in range(25):
            EPkk_I = (self.u - self.n - 1) * np.linalg.inv(self.U)
            ERk_I = (self.v - self.m - 1) * np.linalg.inv(self.V)
            # Form \lambda^x_k(1) and \lambda^x_k(2)
            lxk1 = EPkk_I @ xbar + H.T @ ERk_I @ yk 
            lxk2 = -0.5 * (EPkk_I +  H.T @ ERk_I @ H)
            # recover Pkk and xkk 
            self.P = np.linalg.inv(lxk2 / -0.5)
            self.x = self.P @ lxk1 
            # Form \lambda^P_k(1) and \lambda^P_k(2)
            ex = (self.x - xbar).reshape((self.x.size, 1))
            Ck = 0.5 * ex @ ex.T + self.P 
            #lPk1 = -0.5 * (ubar + self.n + 2)
            lPk2 = -0.5 * (Ubar + Ck)
            # recover uk and Uk 
            self.u = ubar + 1
            self.U = lPk2 / -0.5 
            # Form \lambda^R_k(1) and \lambda^R_k(2)
            resid = (yk - H @ self.x ).reshape((yk.size,1))
            Ak = resid @ resid.T + H @ self.P @ H.T
            #lRk1 = -0.5 * (vbar + self.m + 2)
            lRk2 = -0.5 * (Vbar + Ak)
            # recover 
            self.v = vbar + 1
            self.V = lRk2 / -0.5 
            #print("\nLam-k {}".format(lxk1))
            #print("Ck-k {}".format(np.diag(Ck)))
            #print("A-k {}".format(np.diag(Ak)))
        Pbar = np.linalg.inv( EPkk_I )
        R = np.linalg.inv( ERk_I )
        Q = Pbar - self.Phi @ self.last_P @ self.Phi.T

        return self.x.copy(), self.P.copy(), Q, R

    def nonlin_nat_param_update(self): 
        pass
    
    def step(self, yk, with_tp = True):
        # Time Prop
        if with_tp:
            self.pxk_g_ykm1()
        if self.is_lin:
            _x, _P, Q, R = self.lin_nat_param_update(yk)
        else:
            _x, _P = self.nonlin_nat_param_update(yk)
        return _x, _P, Q, R

def indep_linear_target_track(N, use_scen, num_windows, V0_scale_factor, KNOB_Q, KNOB_V, fig_path):
    T = 1.0 # seconds
    W1 = 1.0 # PSD of noise channel 1
    W2 = 1.0 # PSD of noise channel 2
    #N = 300 # steps

    # Blocks
    T2 = np.array([1.0, T, 0, 1.0]).reshape((2,2))
    z2 = np.zeros(2)
    Z2 = np.zeros((2,2))
    QT2 = np.array([T**3/3, T**2/2, T**2/2, T]).reshape((2,2))
    z4 = np.zeros(4)
    I4 = np.eye(4)

    # True Initial Simulation State
    x0 = np.array([5, -50, 10, -75])  # [m, m/s, m, m/s] -> [x,vx,y,vy]-> [posx,velx,posy,vely]
    
    # Nominal Kalman Filter
    Phi = np.vstack(( np.hstack((T2,Z2)), np.hstack((Z2,T2))  ))
    H = np.array([[1.0,0,0,0], [0,0,1,0]])
    Q0 = np.vstack(( np.hstack(( W1 * QT2, Z2)), np.hstack((Z2, W2 * QT2))  ))
    V0 = np.array([1e4,1e2,1e2,1e4]).reshape((2,2)) / V0_scale_factor 
    P0_kf = np.diag(np.array([100,1,100,1.0]))
    x0_kf = np.random.multivariate_normal(x0, P0_kf)

    # Cauchy Estimator 
    Phik = T2.copy()
    Gamk = np.array([[T**2/2,T]]).T
    Hk = H[0,0:2]
    beta0_1 = np.array([W1**0.5 / 1.3898]) / 1.7
    gambar0_1 = V0[0,0]**0.5/1.3898 / 1.7
    b0_1 = x0_kf[0:2].copy()
    p0_1 = np.diag(P0_kf)[0:2]**0.5 / 1.3898
    A0_1 = np.eye(2)

    beta0_2 = np.array([W2**0.5 / 1.3898]) / 1.7
    gambar0_2 = V0[1,1]**0.5/1.3898 / 1.7
    b0_2 = x0_kf[2:4].copy()
    p0_2 = np.diag(P0_kf)[2:4]**0.5 / 1.3898
    A0_2 = np.eye(2)

    ce.set_tr_search_idxs_ordering([1,0])
    swm_debug_print = False
    win_debug_print = False
    steps = 16
    reinit_func = None #H_reinit_func
    num_windows = 10 #HighJack
    show_marginals = False # FALSE!!

    cauchyEst1 = ce.PySlidingWindowManager('lti', num_windows, swm_debug_print)
    cauchyEst1.initialize_lti(A0_1,p0_1,b0_1,Phik.copy(),None,Gamk.copy(),beta0_1.copy(),Hk,gambar0_1, reinit_func = reinit_func)
    cauchyEst2 = ce.PySlidingWindowManager('lti', num_windows, swm_debug_print)
    cauchyEst2.initialize_lti(A0_2,p0_2,b0_2,Phik.copy(),None,Gamk.copy(),beta0_2.copy(),Hk,gambar0_2, reinit_func = reinit_func)
    
    #cauchyEst1 = ce.PyCauchyEstimator('lti', steps, win_debug_print)
    #cauchyEst1.initialize_lti(A0_1,p0_1,b0_1,Phik.copy(),None,Gamk.copy(),beta0_1.copy(),Hk,gambar0_1)
    #cauchyEst2 = ce.PyCauchyEstimator('lti', steps, win_debug_print)
    #cauchyEst2.initialize_lti(A0_2,p0_2,b0_2,Phik.copy(),None,Gamk.copy(),beta0_2.copy(),Hk,gambar0_2)

    fk = lambda x : Phi @ x
    Fk = lambda x : Phi
    hk = lambda x : H @ x
    Hk = lambda x : H 
    vb = VBayes(x0_kf.copy(), P0_kf.copy(), fk, Fk, hk, Hk, Qhat = 1*np.eye(4), V = 10000*np.eye(2) )


    # Nominal KF Log
    xs_kfn = [] 
    Ps_kfn = [] 
    # Optimal KF Log
    xs_kfo = [] 
    Ps_kfo = [] 
    # Nominal Cauchy Log
    xs_ce = [] 
    Ps_ce = []
    xs_ce_avg = [] 
    Ps_ce_avg = []

    vb_xs = [vb.x.copy()] 
    vb_Ps = [vb.P.copy()] 

    x_kfn = x0_kf.copy() 
    P_kfn = P0_kf.copy()
    x_kfo = x0_kf.copy() 
    P_kfo = P0_kf.copy() 

    # Scenarios 
    s0_q = lambda k : Q0.copy()
    s0_v = lambda k : V0.copy()
    s1_q = lambda k : (10 + 5*np.sin(np.pi*k/N))*Q0 #np.abs(1 + KNOB_Q*np.sin( (np.pi * k) / N))*Q0 #V0.copy() # np.abs(1 + KNOB_Q*np.sin( (np.pi * k) / N))*Q0 # Q0.copy() #
    s2_q = lambda k : Q0.copy() if k < 100 else KNOB_Q/5*Q0 if k < 200 else Q0.copy() 
    s1_v = lambda k : (1 + .5*np.sin(np.pi*k/N))*V0 #np.abs(1 + KNOB_V*np.sin( (np.pi * k) / N))*V0 #V0.copy() #
    s2_v = lambda k : V0.copy() if k < 200 else 15*V0

    if( use_scen == "0" ):
        qscen = s0_q
        vscen = s0_v
    elif(use_scen == "1a"):
        qscen = s1_q
        vscen = s0_v
    elif(use_scen == "1b"):
        qscen = s0_q
        vscen = s1_v
    elif(use_scen == "1c"):
        qscen = s1_q
        vscen = s1_v
    elif(use_scen == "2"):
        qscen = s2_q
        vscen = s2_v
    else:
        print("Use Scenario {} not valid [0,1a,1b,1c,2]. Exiting!".format(use_scen))
        exit(1)

    xks = [x0.copy()] 
    vks = [np.random.multivariate_normal(z2, vscen(0))] 
    zks = [H @ xks[0] + vks[0]]
    wks = []
    # Simulation Realization Loop
    xk = x0.copy()
    for k in range(N):
        wk = np.random.multivariate_normal( z4, qscen(k) ) #np.random.standard_cauchy(4) * np.diag(qscen(k))**0.5 #
        xk = Phi @ xk + wk
        vk = np.random.multivariate_normal( z2, vscen(k+1) )
        zk = H @ xk + vk 
        xks.append(xk)
        zks.append(zk)
        vks.append(vk)
        wks.append(wk)

    # Estimation Loop
    if show_marginals:
        fig1 = plt.figure() 
        fig2 = plt.figure() 

    Qn = Q0.copy() 
    Vn = V0.copy()
    for k in range(N+1):
        Qo = qscen(k-1)
        Vo = vscen(k)
        zk = zks[k]
        print("Step {}/{}".format(k+1, N))
        
        # COMMENT OPT 1 or OPT 2 depending on whether cauchyEst is SLIDINGWINDOWMANGER or CAUCHYESTIMATOR class

        # OPT1: SWM
        x_ce1, P_ce1, x_ce_avg1, P_ce_avg1 = cauchyEst1.step( zk[0] )
        x_ce2, P_ce2, x_ce_avg2, P_ce_avg2 = cauchyEst2.step( zk[1] )

        if k > 0:
            vb_x, vb_P, vb_Q, vb_R = vb.step(zk, True)
            vb_xs.append(vb_x)
            vb_Ps.append(vb_P)

        # OPT2: DEBUG SINGLE WINDOW
        #x_ce1, P_ce1 = cauchyEst1.step( zk[0] )
        #x_ce2, P_ce2 = cauchyEst2.step( zk[1] )
        #x_ce_avg1, P_ce_avg1 = x_ce1, P_ce1
        #x_ce_avg2, P_ce_avg2 = x_ce2, P_ce2

        x_ce = np.concatenate((x_ce1,x_ce2))
        P_ce = np.zeros((4,4))
        P_ce[0:2,0:2] = P_ce1.copy()
        P_ce[2:4,2:4] = P_ce2.copy()
        x_ce_avg = np.concatenate((x_ce_avg1,x_ce_avg2))
        P_ce_avg = np.zeros((4,4))
        P_ce_avg[0:2,0:2] = P_ce_avg1.copy()
        P_ce_avg[2:4,2:4] = P_ce_avg2.copy()

        xs_ce.append(x_ce)
        Ps_ce.append(P_ce)
        xs_ce_avg.append(x_ce_avg)
        Ps_ce_avg.append(P_ce_avg)
        
        # Time Prop KFs
        if k > 0: 
            x_kfn = Phi @ x_kfn
            P_kfn = Phi @ P_kfn @ Phi.T + Qn
            x_kfo = Phi @ x_kfo
            P_kfo = Phi @ P_kfo @ Phi.T + Qo
        # Measurement Update
        K_kfn = P_kfn @ H.T @ np.linalg.inv(H @ P_kfn @ H.T + Vn)
        x_kfn += K_kfn @ (zk - H @ x_kfn)
        P_kfn = (I4 - K_kfn @ H) @ P_kfn @ (I4 - K_kfn @ H).T + K_kfn @ Vn @ K_kfn.T
        K_kfo = P_kfo @ H.T @ np.linalg.inv(H @ P_kfo @ H.T + Vo)
        x_kfo += K_kfo @ (zk - H @ x_kfo)
        P_kfo = (I4 - K_kfo @ H) @ P_kfo @ (I4 - K_kfo @ H).T + K_kfo @ Vo @ K_kfo.T
        # Log Step
        xs_kfn.append(x_kfn)
        Ps_kfn.append(P_kfn)
        xs_kfo.append(x_kfo)
        Ps_kfo.append(P_kfo)

        if show_marginals:
            if k == 0:
                P_ce += np.eye(4)
            best_win_idx = np.argmax(cauchyEst1.win_counts)
            sig=3
            points_per_xaxis = 200
            points_per_yaxis = points_per_xaxis
            PX,VX,FXZ = cauchyEst1.cauchyEsts[best_win_idx].get_2D_pointwise_cpdf( #cauchyEst.cauchyEsts[best_win_idx].get_2D_pointwise_cpdf( #cauchyEst.get_2D_pointwise_cpdf(
                x_ce[0] - sig*P_ce[0,0]**0.5, 
                x_ce[0] + sig*P_ce[0,0]**0.5, 
                2*sig*P_ce[0,0]**0.5 / points_per_xaxis, 
                x_ce[1] - sig*P_ce[1,1]**0.5, 
                x_ce[1] + sig*P_ce[1,1]**0.5, 
                2*sig*P_ce[1,1]**0.5 / points_per_xaxis, 
                log_dir = None)
            PY,VY,FYZ = cauchyEst2.cauchyEsts[best_win_idx].get_2D_pointwise_cpdf( #cauchyEst.cauchyEsts[best_win_idx].get_2D_pointwise_cpdf( #cauchyEst.get_2D_pointwise_cpdf(
                x_ce[2] - sig*P_ce[2,2]**0.5, 
                x_ce[2] + sig*P_ce[2,2]**0.5, 
                2*sig*P_ce[2,2]**0.5 / points_per_yaxis, 
                x_ce[3] - sig*P_ce[3,3]**0.5, 
                x_ce[3] + sig*P_ce[3,3]**0.5, 
                2*sig*P_ce[3,3]**0.5 / points_per_yaxis, 
                log_dir = None)
            #'''
            z_low1 = -np.max(FXZ) / 10
            z_low2 = -np.max(FYZ) / 10

            _x_ce1, _P_ce1 = cauchyEst1.cauchyEsts[best_win_idx].get_last_mean_cov()
            _x_ce2, _P_ce2 = cauchyEst2.cauchyEsts[best_win_idx].get_last_mean_cov()

            fig1.clf()
            fig2.clf()
            quantile = 0.70
            ax1 = fig1.add_subplot(projection='3d')
            ax1.set_title("Position/Velocity in X")
            ax1.set_xlabel('Position in x (meters)')
            ax1.set_ylabel('Velocity in x (meters)')
            ax1.plot_wireframe(PX,VX, FXZ, color='k', alpha=0.7) 
            ell_px_ce, ell_vx_ce = ce.get_2d_covariance_ellipsoid(_x_ce1, _P_ce1, quantile, num_points = 100)
            ell_px_kfo, ell_vx_kfo = ce.get_2d_covariance_ellipsoid(x_kfo[0:2], P_kfo[0:2,0:2], quantile, num_points = 100)
            ell_px_kfn, ell_vx_kfn = ce.get_2d_covariance_ellipsoid(x_kfn[0:2], P_kfn[0:2,0:2], quantile, num_points = 100)
            ax1.plot(ell_px_ce, ell_vx_ce, z_low1, color='k')
            ax1.plot(ell_px_kfo, ell_vx_kfo, z_low1, color='g')
            ax1.plot(ell_px_kfn, ell_vx_kfn, z_low1, color='b')
            ax1.scatter(xks[k][0], xks[k][1], z_low1, color='r', marker='*', s=50)
            ax1.scatter(x_kfo[0], x_kfo[1], z_low1, color='g', marker='*', s=50)
            ax1.scatter(x_kfn[0], x_kfn[1], z_low1, color='b', marker='*', s=50)
            ax1.scatter(x_ce[0], x_ce[1], z_low1, color='k', marker='*', s=50)

            ax2 = fig2.add_subplot(projection='3d')
            ax2.set_title("Position/Velocity in Y")
            ax2.set_xlabel('Position in y (meters)')
            ax2.set_ylabel('Velocity in y (meters)')
            ax2.plot_wireframe(PY,VY, FYZ, color='k', alpha=0.7) 
            ell_py_ce, ell_vy_ce = ce.get_2d_covariance_ellipsoid(_x_ce2, _P_ce2, quantile, num_points = 100)
            ell_py_kfo, ell_vy_kfo = ce.get_2d_covariance_ellipsoid(x_kfo[2:4], P_kfo[2:4,2:4], quantile, num_points = 100)
            ell_py_kfn, ell_vy_kfn = ce.get_2d_covariance_ellipsoid(x_kfn[2:4], P_kfn[2:4,2:4], quantile, num_points = 100)
            ax2.plot(ell_py_ce, ell_vy_ce, z_low2, color='k')
            ax2.plot(ell_py_kfo, ell_vy_kfo, z_low2, color='g')
            ax2.plot(ell_py_kfn, ell_vy_kfn, z_low2, color='b')
            ax2.scatter(xks[k][2], xks[k][3], z_low2, color='r', marker='*', s=50)
            ax2.scatter(x_kfo[2], x_kfo[3], z_low2, color='g', marker='*', s=50)
            ax2.scatter(x_kfn[2], x_kfn[3], z_low2, color='b', marker='*', s=50)
            ax2.scatter(x_ce[2], x_ce[3], z_low2, color='k', marker='*', s=50)
            plt.pause(1)
            #plt.close('all')
            ax1.clear()
            ax2.clear()
            #'''
    
    # Lists -> NP Arrays 
    xks = np.array(xks)
    zks = np.array(zks)
    vks = np.array(vks)
    wks = np.array(wks)
    xs_kfn = np.array(xs_kfn)
    Ps_kfn = np.array(Ps_kfn)
    xs_kfo = np.array(xs_kfo)
    Ps_kfo = np.array(Ps_kfo)

    xs_ce = np.array(xs_ce)
    Ps_ce = np.array(Ps_ce)
    xs_ce_avg = np.array(xs_ce_avg)
    Ps_ce_avg = np.array(Ps_ce_avg)

    vb_xs = np.array(vb_xs)
    vb_Ps = np.array(vb_Ps)

    # Now Plot Results
    Ts = np.arange(N+1)*T 
    # State History Plot 
    '''
    plt.figure()
    plt.suptitle("State History Plot\nTrue State=Red, Opt KF=green, Nom KF=blue, Nom Cauchy=magenta")
    plt.subplot(2,1,1)
    plt.plot(xks[:, 0], xks[:, 2], 'r')
    #plt.scatter(xks[:, 0], xks[:, 1], color='r')
    plt.plot(xs_kfn[:, 0], xs_kfn[:, 2], 'b')
    #plt.scatter(xs_kfn[:, 0], xs_kfn[:, 1], color='g')
    plt.plot(xs_kfo[:, 0], xs_kfo[:, 2], 'g')
    #plt.scatter(xs_kfo[:, 0], xs_kfo[:, 1], color='b')
    plt.ylabel("Pos Y")
    plt.xlabel("Pos X")
    plt.subplot(2,1,2)
    plt.plot(xks[:, 1], xks[:, 3], 'r')
    #plt.scatter(xks[:, 2], xks[:, 3], color='r')
    plt.plot(xs_kfn[:, 1], xs_kfn[:, 3], 'b')
    #plt.scatter(xs_kfn[:, 2], xs_kfn[:, 3], color='g')
    plt.plot(xs_kfo[:, 1], xs_kfo[:, 3], 'g')
    #plt.scatter(xs_kfo[:, 2], xs_kfo[:, 3], color='b')
    plt.ylabel("Vel Y")
    plt.xlabel("Vel X")
    '''

    # State Error Plot
    scale = 1
    plt.figure(figsize=(12,12))
    plt.suptitle("State Error Plot\nOpt KF=green, Nom KF=blue, Nom Cauchy=magenta/black")
    one_sig_kfn = np.array([scale*np.diag(P)**0.5 for P in Ps_kfn])
    one_sig_kfo = np.array([scale*np.diag(P)**0.5 for P in Ps_kfo])
    one_sig_ce = np.array([scale*np.diag(P)**0.5 for P in Ps_ce])
    one_sig_ce_avg = np.array([scale*np.diag(P)**0.5 for P in Ps_ce_avg])

    one_sig_vb = np.array([scale*np.diag(P)**0.5 for P in vb_Ps])

    # Some Analysis 
    states = 4
    conf_percent = 0.70
    from scipy.stats import chi2 
    s = chi2.ppf(conf_percent, states)
    # ARMSE
    ae_kfo = np.mean( (xks[1:] - xs_kfo[1:])**2, axis=0)
    ae_kfo = np.array([ (ae_kfo[0]+ae_kfo[2])**0.5, (ae_kfo[1]+ae_kfo[3])**0.5 ])
    ae_kfn = np.mean( (xks[1:] - xs_kfn[1:])**2, axis=0)
    ae_kfn = np.array([ (ae_kfn[0]+ae_kfn[2])**0.5, (ae_kfn[1]+ae_kfn[3])**0.5 ])
    ae_ce = np.mean( (xks[1:] - xs_ce[1:])**2, axis=0)
    ae_ce = np.array([ (ae_ce[0]+ae_ce[2])**0.5, (ae_ce[1]+ae_ce[3])**0.5 ])
    ae_cea = np.mean( (xks[1:] - xs_ce_avg[1:])**2, axis=0)
    ae_cea = np.array([ (ae_cea[0]+ae_cea[2])**0.5, (ae_cea[1]+ae_cea[3])**0.5 ])
    
    ae_vb = np.mean( (xks[1:] - vb_xs[1:])**2, axis=0)
    ae_vb = np.array([ (ae_vb[0]+ae_vb[2])**0.5, (ae_vb[1]+ae_vb[3])**0.5 ])

    kfo_bound_realiz = np.mean([ ( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s for xt, xh, P in zip(xks[1:], xs_kfo[1:], Ps_kfo[1:]) ])
    kfn_bound_realiz = np.mean([ ( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s for xt, xh, P in zip(xks[1:], xs_kfn[1:], Ps_kfn[1:]) ])
    ce_bound_realiz = np.mean([ (( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s) if np.all(np.linalg.eig(P)[0]>0) else 0 for xt, xh, P in zip(xks[1:], xs_ce[1:], Ps_ce[1:]) ]) 
    cea_bound_realiz = np.mean([ (( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s) if np.all(np.linalg.eig(P)[0]>0) else 0 for xt, xh, P in zip(xks[1:], xs_ce_avg[1:], Ps_ce_avg[1:]) ]) 
    vb_bound_realiz = np.mean([ (( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s) if np.all(np.linalg.eig(P)[0]>0) else 0 for xt, xh, P in zip(xks[1:], vb_xs[1:], vb_Ps[1:]) ]) 

    print("The Optimal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_kfo, kfo_bound_realiz) )
    print("The Nominal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_kfn, kfn_bound_realiz) )
    print("The (Best Window) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_ce, ce_bound_realiz) )
    print("The (Averaged) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_cea, cea_bound_realiz) )
    print("The VB KF Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_vb, vb_bound_realiz) )

    plt.subplot(4,1,1)
    plt.plot(Ts, xks[:,0] - xs_kfo[:,0], 'g')
    plt.plot(Ts, one_sig_kfo[:,0], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,0], 'g--')
    plt.plot(Ts, xks[:,0] - xs_kfn[:,0], 'b')
    plt.plot(Ts, one_sig_kfn[:,0], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,0], 'b--')
    #plt.plot(Ts, xks[:,0] - xs_ce[:,0], 'm')
    #plt.plot(Ts, one_sig_ce[:,0], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,0], 'm--')
    plt.plot(Ts, xks[:,0] - xs_ce_avg[:,0], 'm')
    plt.plot(Ts, one_sig_ce_avg[:,0], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,0], 'k--')
    plt.plot(Ts, xks[:,0] - vb_xs[:,0], 'y')
    plt.plot(Ts, one_sig_vb[:,0], 'y--')
    plt.plot(Ts, -one_sig_vb[:,0], 'y--')

    plt.ylabel("Pos X")
    plt.subplot(4,1,2)
    plt.plot(Ts, xks[:,1] - xs_kfo[:,1], 'g')
    plt.plot(Ts, one_sig_kfo[:,1], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,1], 'g--')
    plt.plot(Ts, xks[:,1] - xs_kfn[:,1], 'b')
    plt.plot(Ts, one_sig_kfn[:,1], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,1], 'b--')
    #plt.plot(Ts, xks[:,1] - xs_ce[:,1], 'm')
    #plt.plot(Ts, one_sig_ce[:,1], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,1], 'm--')
    plt.plot(Ts, xks[:,1] - xs_ce_avg[:,1], 'm')
    plt.plot(Ts, one_sig_ce_avg[:,1], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,1], 'k--')
    plt.plot(Ts, xks[:,1] - vb_xs[:,1], 'y')
    plt.plot(Ts, one_sig_vb[:,1], 'y--')
    plt.plot(Ts, -one_sig_vb[:,1], 'y--')
    plt.ylabel("Vel X")

    plt.subplot(4,1,3)
    plt.plot(Ts, xks[:,2] - xs_kfo[:,2], 'g')
    plt.plot(Ts, one_sig_kfo[:,2], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,2], 'g--')
    plt.plot(Ts, xks[:,2] - xs_kfn[:,2], 'b')
    plt.plot(Ts, one_sig_kfn[:,2], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,2], 'b--')
    #plt.plot(Ts, xks[:,2] - xs_ce[:,2], 'm')
    #plt.plot(Ts, one_sig_ce[:,2], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,2], 'm--')
    plt.plot(Ts, xks[:,2] - xs_ce_avg[:,2], 'm')
    plt.plot(Ts, one_sig_ce_avg[:,2], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,2], 'k--')
    plt.plot(Ts, xks[:,2] - vb_xs[:,2], 'y')
    plt.plot(Ts, one_sig_vb[:,2], 'y--')
    plt.plot(Ts, -one_sig_vb[:,2], 'y--')
    plt.ylabel("Pos Y")


    plt.subplot(4,1,4)
    plt.plot(Ts, xks[:,3] - xs_kfo[:,3], 'g')
    plt.plot(Ts, one_sig_kfo[:,3], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,3], 'g--')
    plt.plot(Ts, xks[:,3] - xs_kfn[:,3], 'b')
    plt.plot(Ts, one_sig_kfn[:,3], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,3], 'b--')
    #plt.plot(Ts, xks[:,3] - xs_ce[:,3], 'm')
    #plt.plot(Ts, one_sig_ce[:,3], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,3], 'm--')
    plt.plot(Ts, xks[:,3] - xs_ce_avg[:,3], 'm')
    plt.plot(Ts, one_sig_ce_avg[:,3], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,3], 'k--')
    plt.plot(Ts, xks[:,3] - vb_xs[:,3], 'y')
    plt.plot(Ts, one_sig_vb[:,3], 'y--')
    plt.plot(Ts, -one_sig_vb[:,3], 'y--')
    plt.ylabel("Vel Y")
    plt.xlabel("Time (sec)")
    
    ret_dic = {"xks": xks,
               "zks" : zks,
               "vks" : vks,
               "wks" : wks,
               "xs_kfo" : xs_kfo, 
               "Ps_kfo" : Ps_kfo,
               "ae_kfo" : ae_kfo,
               "kfo_bound_realiz" : kfo_bound_realiz,
               "xs_kfn" : xs_kfn, 
               "Ps_kfn" : Ps_kfn,
               "ae_kfn" : ae_kfn,
               "kfn_bound_realiz" : kfn_bound_realiz,
               "xs_ce" : xs_ce, 
               "Ps_ce" : Ps_ce,
               "ae_ce" : ae_ce,
               "ce_bound_realiz" : ce_bound_realiz,
               "xs_ce_avg" : xs_ce_avg, 
               "Ps_ce_avg" : Ps_ce_avg,
               "ae_cea" : ae_cea,
               "cea_bound_realiz" : cea_bound_realiz,
               "ae_vb" : ae_vb,
               "vb_bound_realiz" : vb_bound_realiz
               }
    plt.pause(10)
    plt.savefig(fname = fig_path)
    plt.close('all')
    return ret_dic

def emce_f(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    T = 1.0
    xk = pyduc.cget_x()
    
    T2 = np.array([1.0, T, 0, 1.0]).reshape((2,2))
    Z2 = np.zeros((2,2))
    Phik = np.vstack(( np.hstack((T2,Z2)), np.hstack((Z2,T2))  ))
    Gamk = np.array([[T**2/2,T,0,0], [0,0,T**2/2,T]]).T
    pyduc.cset_Phi(Phik)
    pyduc.cset_Gamma(Gamk)
    # Propagate and set x
    xbar = Phik @ xk
    pyduc.cset_x(xbar)
    pyduc.cset_is_xbar_set_for_ece()

def emce_h(c_duc, c_zbar):
    global SUM_INIT_H
    if SUM_INIT_H:
        pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
        xbar = pyduc.cget_x() # xbar
        zbar = nonlin_h(xbar)
        zbar[1] += zbar[0]
        pyduc.cset_zbar(c_zbar, zbar) 
    else:
        pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
        xbar = pyduc.cget_x() # xbar
        zbar = nonlin_h(xbar)
        pyduc.cset_zbar(c_zbar, zbar)

def emce_H(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    H = ce.cd4_gvf(xbar, nonlin_h)
    H[0,1] = 0
    H[0,3] = 0
    H[1,1] = 0
    H[1,3] = 0
    global SUM_INIT_H
    if SUM_INIT_H:
        H[1] = np.sum(H, axis = 0)
        pyduc.cset_H(H)
    else:
        pyduc.cset_H(H)

def non_linear_target_track(N, use_scen, num_windows, V0_scale_factor, KNOB_Q, KNOB_V, fig_path):
    T = 1.0 # seconds
    W1 = 1.0 # PSD of noise channel 1
    W2 = 1.0 # PSD of noise channel 2
    #N = 300 # steps

    # Blocks
    T2 = np.array([1.0, T, 0, 1.0]).reshape((2,2))
    z2 = np.zeros(2)
    Z2 = np.zeros((2,2))
    QT2 = np.array([T**3/3, T**2/2, T**2/2, T]).reshape((2,2))
    z4 = np.zeros(4)
    I4 = np.eye(4)

    # True Initial Simulation State
    x0 = np.array([1e4, -100, 1e4, -100])  # [m, m/s, m, m/s] -> [x,vx,y,vy]-> [posx,velx,posy,vely]
    
    # Nominal Kalman Filter
    Phi = np.vstack(( np.hstack((T2,Z2)), np.hstack((Z2,T2))  ))
    Q0 = np.vstack(( np.hstack(( W1 * QT2, Z2)), np.hstack((Z2, W2 * QT2))  ))
    V0 = np.array([100 / V0_scale_factor,0,0,1e-6]).reshape((2,2)) 
    P0_kf = np.diag(np.array([100.0**2,10**2,100*2,10**2]))
    x0_kf = np.random.multivariate_normal(x0, P0_kf)

    # Cauchy Estimator 
    beta = np.array([W1**0.5 / 1.3898, W2**0.5 / 1.3898,]) / 1.3
    gamma = np.array([V0[0,0]**0.5/1.3898, V0[1,1]**0.5/1.3898 ]) / 1.3
    b0  = np.zeros(4)
    p0 = np.diag(P0_kf)**0.5 / 1.3898
    A0 = np.eye(4)
    x0_ce = x0_kf.copy()


    ce.set_tr_search_idxs_ordering([1,3,0,2])
    swm_debug_print = True
    win_debug_print = False
    steps = 16
    reinit_func =  H_reinit_func_nl #None
    show_marginals = False # FALSE!!

    cauchyEst = ce.PySlidingWindowManager('nonlin', num_windows, swm_debug_print, win_debug_print= win_debug_print)
    cauchyEst.initialize_nonlin(x0_ce, A0, p0, b0, beta, gamma, emce_f, emce_h, emce_H, 0, 0, 0, reinit_func = reinit_func)

    #cauchyEst = ce.PyCauchyEstimator('nonlin', num_windows, win_debug_print)
    #cauchyEst.initialize_nonlin(x0_ce, A0, p0, b0, beta0.copy(),gambar0.copy(), emce_f, emce_h, emce_H, 0, 0, 0)

    # Run Simulation 


    # Nominal KF Log
    xs_kfn = [] 
    Ps_kfn = [] 
    # Optimal KF Log
    xs_kfo = [] 
    Ps_kfo = [] 
    # Nominal Cauchy Log
    xs_ce = [] 
    Ps_ce = []
    xs_ce_avg = [] 
    Ps_ce_avg = []

    x_kfn = x0_kf.copy() 
    P_kfn = P0_kf.copy()
    x_kfo = x0_kf.copy() 
    P_kfo = P0_kf.copy() 

    # Scenarios 
    s0_q = lambda k : Q0.copy()
    s0_v = lambda k : V0.copy()
    s1_q = lambda k : (100 + 50*np.cos(np.pi*k/N))*Q0 #np.abs(1 + KNOB_Q*np.sin( (np.pi * k) / N))*Q0 #V0.copy() # np.abs(1 + KNOB_Q*np.sin( (np.pi * k) / N))*Q0 # Q0.copy() #
    s2_q = lambda k : Q0.copy() if k < 100 else 100*Q0 if k < 200 else Q0.copy() 
    s1_v = lambda k : (1 + 0.5*np.cos(np.pi*k/N))*V0 #np.abs(1 + KNOB_V*np.sin( (np.pi * k) / N))*V0 #V0.copy() #
    s2_v = lambda k : (1 + 0.5*np.cos(np.pi*k/N))*V0

    if( use_scen == "0" ):
        qscen = s0_q
        vscen = s0_v
    elif(use_scen == "1a"):
        qscen = s1_q
        vscen = s0_v
    elif(use_scen == "1b"):
        qscen = s0_q
        vscen = s1_v
    elif(use_scen == "1c"):
        qscen = s1_q
        vscen = s1_v
    elif(use_scen == "2"):
        qscen = s2_q
        vscen = s2_v
    else:
        print("Use Scenario {} not valid [0,1a,1b,1c,2]. Exiting!".format(use_scen))
        exit(1)

    xks = [x0.copy()] 
    vks = [np.random.multivariate_normal(z2, vscen(0))] 
    zks = [nonlin_h(xks[0]) + vks[0]]
    wks = []
    # Simulation Realization Loop
    xk = x0.copy()
    for k in range(N):
        wk = np.random.multivariate_normal( z4, qscen(k) ) #np.random.standard_cauchy(4) * np.diag(qscen(k))**0.5 #
        xk = Phi @ xk + wk
        vk = np.random.multivariate_normal( z2, vscen(k+1) )
        zk = nonlin_h(xk) + vk 
        xks.append(xk)
        zks.append(zk)
        vks.append(vk)
        wks.append(wk)

    # Estimation Loop
    if show_marginals:
        fig1 = plt.figure() 
        fig2 = plt.figure() 

    Qn = Q0.copy() 
    Vn = V0.copy()
    for k in range(N+1):
        Qo = qscen(k-1)
        Vo = vscen(k)
        zk = zks[k]
        print("Step {}/{}".format(k+1, N))
        
        # COMMENT OPT 1 or OPT 2 depending on whether cauchyEst is SLIDINGWINDOWMANGER or CAUCHYESTIMATOR class

        # OPT1: SWM
        x_ce, P_ce, x_ce_avg, P_ce_avg = cauchyEst.step( zk )

        # OPT2: DEBUG SINGLE WINDOW
        #x_ce, P_ce = cauchyEst.step( zk[0] )
        #x_ce_avg, P_ce_avg = x_ce, P_ce

        xs_ce.append(x_ce)
        Ps_ce.append(P_ce)
        xs_ce_avg.append(x_ce_avg)
        Ps_ce_avg.append(P_ce_avg)
        
        # Time Prop KFs
        if k > 0: 
            x_kfn = Phi @ x_kfn
            P_kfn = Phi @ P_kfn @ Phi.T + Qn
            x_kfo = Phi @ x_kfo
            P_kfo = Phi @ P_kfo @ Phi.T + Qo
        # Measurement Update
        Hn = ce.cd4_gvf(x_kfn, nonlin_h)
        Ho = ce.cd4_gvf(x_kfo, nonlin_h)
        K_kfn = P_kfn @ Hn.T @ np.linalg.inv(Hn @ P_kfn @ Hn.T + Vn)
        x_kfn += K_kfn @ (zk - nonlin_h(x_kfn) )
        P_kfn = (I4 - K_kfn @ Hn) @ P_kfn @ (I4 - K_kfn @ Hn).T + K_kfn @ Vn @ K_kfn.T
        K_kfo = P_kfo @ Ho.T @ np.linalg.inv(Ho @ P_kfo @ Ho.T + Vo)
        x_kfo += K_kfo @ (zk - nonlin_h(x_kfo))
        P_kfo = (I4 - K_kfo @ Ho) @ P_kfo @ (I4 - K_kfo @ Ho).T + K_kfo @ Vo @ K_kfo.T
        # Log Step
        xs_kfn.append(x_kfn)
        Ps_kfn.append(P_kfn)
        xs_kfo.append(x_kfo)
        Ps_kfo.append(P_kfo)

        if show_marginals:
            if k == 0:
                P_ce += np.eye(4)
            best_win_idx = np.argmax(cauchyEst.win_counts)
            sig=3
            points_per_xaxis = 200
            points_per_yaxis = points_per_xaxis
            PX,VX,FXZ = cauchyEst.cauchyEsts[best_win_idx].get_marginal_2D_pointwise_cpdf(  #cauchyEst.cauchyEsts[best_win_idx].get_marginal_2D_pointwise_cpdf( #cauchyEst.get_marginal_2D_pointwise_cpdf(
                0,1,
                -sig*P_ce[0,0]**0.5, 
                sig*P_ce[0,0]**0.5, 
                2*sig*P_ce[0,0]**0.5 / points_per_xaxis, 
                -sig*P_ce[1,1]**0.5, 
                sig*P_ce[1,1]**0.5, 
                2*sig*P_ce[1,1]**0.5 / points_per_xaxis, 
                log_dir = None)
            PY,VY,FYZ = cauchyEst.cauchyEsts[best_win_idx].get_marginal_2D_pointwise_cpdf( #cauchyEst.cauchyEsts[best_win_idx].get_marginal_2D_pointwise_cpdf( #cauchyEst.get_marginal_2D_pointwise_cpdf(
                2,3,
                -sig*P_ce[2,2]**0.5, 
                sig*P_ce[2,2]**0.5, 
                2*sig*P_ce[2,2]**0.5 / points_per_yaxis, 
                -sig*P_ce[3,3]**0.5, 
                sig*P_ce[3,3]**0.5, 
                2*sig*P_ce[3,3]**0.5 / points_per_yaxis, 
                log_dir = None)
            PX += x_ce[0]
            VX += x_ce[1]
            PY += x_ce[2]
            VY += x_ce[3]

            #'''
            z_low1 = -np.max(FXZ) / 10
            z_low2 = -np.max(FYZ) / 10

            _x_ce, _P_ce = cauchyEst.cauchyEsts[best_win_idx].get_last_mean_cov()

            fig1.clf()
            fig2.clf()
            quantile = 0.70
            ax1 = fig1.add_subplot(projection='3d')
            ax1.set_title("Position/Velocity in X")
            ax1.set_xlabel('Position in x (meters)')
            ax1.set_ylabel('Velocity in x (meters)')
            ax1.plot_wireframe(PX,VX, FXZ, color='k', alpha=0.7) 
            ell_px_ce, ell_vx_ce = ce.get_2d_covariance_ellipsoid(_x_ce[0:2], _P_ce[0:2,0:2], quantile, num_points = 100)
            ell_px_kfo, ell_vx_kfo = ce.get_2d_covariance_ellipsoid(x_kfo[0:2], P_kfo[0:2,0:2], quantile, num_points = 100)
            ell_px_kfn, ell_vx_kfn = ce.get_2d_covariance_ellipsoid(x_kfn[0:2], P_kfn[0:2,0:2], quantile, num_points = 100)
            ax1.plot(ell_px_ce, ell_vx_ce, z_low1, color='k')
            ax1.plot(ell_px_kfo, ell_vx_kfo, z_low1, color='g')
            ax1.plot(ell_px_kfn, ell_vx_kfn, z_low1, color='b')
            ax1.scatter(xks[k][0], xks[k][1], z_low1, color='r', marker='*', s=50)
            ax1.scatter(x_kfo[0], x_kfo[1], z_low1, color='g', marker='*', s=50)
            ax1.scatter(x_kfn[0], x_kfn[1], z_low1, color='b', marker='*', s=50)
            ax1.scatter(x_ce[0], x_ce[1], z_low1, color='k', marker='*', s=50)

            ax2 = fig2.add_subplot(projection='3d')
            ax2.set_title("Position/Velocity in Y")
            ax2.set_xlabel('Position in y (meters)')
            ax2.set_ylabel('Velocity in y (meters)')
            ax2.plot_wireframe(PY,VY, FYZ, color='k', alpha=0.7) 
            ell_py_ce, ell_vy_ce = ce.get_2d_covariance_ellipsoid(_x_ce[2:4], _P_ce[2:4,2:4], quantile, num_points = 100)
            ell_py_kfo, ell_vy_kfo = ce.get_2d_covariance_ellipsoid(x_kfo[2:4], P_kfo[2:4,2:4], quantile, num_points = 100)
            ell_py_kfn, ell_vy_kfn = ce.get_2d_covariance_ellipsoid(x_kfn[2:4], P_kfn[2:4,2:4], quantile, num_points = 100)
            ax2.plot(ell_py_ce, ell_vy_ce, z_low2, color='k')
            ax2.plot(ell_py_kfo, ell_vy_kfo, z_low2, color='g')
            ax2.plot(ell_py_kfn, ell_vy_kfn, z_low2, color='b')
            ax2.scatter(xks[k][2], xks[k][3], z_low2, color='r', marker='*', s=50)
            ax2.scatter(x_kfo[2], x_kfo[3], z_low2, color='g', marker='*', s=50)
            ax2.scatter(x_kfn[2], x_kfn[3], z_low2, color='b', marker='*', s=50)
            ax2.scatter(x_ce[2], x_ce[3], z_low2, color='k', marker='*', s=50)
            plt.pause(1)
            #plt.close('all')
            ax1.clear()
            ax2.clear()
            #'''
    
    # Lists -> NP Arrays 
    xks = np.array(xks)
    zks = np.array(zks)
    vks = np.array(vks)
    wks = np.array(wks)
    xs_kfn = np.array(xs_kfn)
    Ps_kfn = np.array(Ps_kfn)
    xs_kfo = np.array(xs_kfo)
    Ps_kfo = np.array(Ps_kfo)

    xs_ce = np.array(xs_ce)
    Ps_ce = np.array(Ps_ce)
    xs_ce_avg = np.array(xs_ce_avg)
    Ps_ce_avg = np.array(Ps_ce_avg)

    # Now Plot Results
    Ts = np.arange(N+1)*T 
    # State History Plot 
    '''
    plt.figure()
    plt.suptitle("State History Plot\nTrue State=Red, Opt KF=green, Nom KF=blue, Nom Cauchy=magenta")
    plt.subplot(2,1,1)
    plt.plot(xks[:, 0], xks[:, 2], 'r')
    #plt.scatter(xks[:, 0], xks[:, 1], color='r')
    plt.plot(xs_kfn[:, 0], xs_kfn[:, 2], 'b')
    #plt.scatter(xs_kfn[:, 0], xs_kfn[:, 1], color='g')
    plt.plot(xs_kfo[:, 0], xs_kfo[:, 2], 'g')
    #plt.scatter(xs_kfo[:, 0], xs_kfo[:, 1], color='b')
    plt.ylabel("Pos Y")
    plt.xlabel("Pos X")
    plt.subplot(2,1,2)
    plt.plot(xks[:, 1], xks[:, 3], 'r')
    #plt.scatter(xks[:, 2], xks[:, 3], color='r')
    plt.plot(xs_kfn[:, 1], xs_kfn[:, 3], 'b')
    #plt.scatter(xs_kfn[:, 2], xs_kfn[:, 3], color='g')
    plt.plot(xs_kfo[:, 1], xs_kfo[:, 3], 'g')
    #plt.scatter(xs_kfo[:, 2], xs_kfo[:, 3], color='b')
    plt.ylabel("Vel Y")
    plt.xlabel("Vel X")
    '''

    # State Error Plot
    scale = 1
    plt.figure(figsize=(12,12))
    plt.suptitle("State Error Plot\nOpt KF=green, Nom KF=blue, Nom Cauchy=magenta/black")
    one_sig_kfn = np.array([scale*np.diag(P)**0.5 for P in Ps_kfn])
    one_sig_kfo = np.array([scale*np.diag(P)**0.5 for P in Ps_kfo])
    one_sig_ce = np.array([scale*np.diag(P)**0.5 for P in Ps_ce])
    one_sig_ce_avg = np.array([scale*np.diag(P)**0.5 for P in Ps_ce_avg])

    # Some Analysis 
    states = 4
    conf_percent = 0.70
    from scipy.stats import chi2 
    s = chi2.ppf(conf_percent, states)
    # ARMSE
    ae_kfo = np.mean( (xks[1:] - xs_kfo[1:])**2, axis=0)
    ae_kfo = np.array([ (ae_kfo[0]+ae_kfo[2])**0.5, (ae_kfo[1]+ae_kfo[3])**0.5 ])
    ae_kfn = np.mean( (xks[1:] - xs_kfn[1:])**2, axis=0)
    ae_kfn = np.array([ (ae_kfn[0]+ae_kfn[2])**0.5, (ae_kfn[1]+ae_kfn[3])**0.5 ])
    ae_ce = np.mean( (xks[1:] - xs_ce[1:])**2, axis=0)
    ae_ce = np.array([ (ae_ce[0]+ae_ce[2])**0.5, (ae_ce[1]+ae_ce[3])**0.5 ])
    ae_cea = np.mean( (xks[1:] - xs_ce_avg[1:])**2, axis=0)
    ae_cea = np.array([ (ae_cea[0]+ae_cea[2])**0.5, (ae_cea[1]+ae_cea[3])**0.5 ])
    kfo_bound_realiz = np.mean([ ( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s for xt, xh, P in zip(xks[1:], xs_kfo[1:], Ps_kfo[1:]) ])
    kfn_bound_realiz = np.mean([ ( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s for xt, xh, P in zip(xks[1:], xs_kfn[1:], Ps_kfn[1:]) ])
    ce_bound_realiz = np.mean([ (( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s) if np.all(np.linalg.eig(P)[0]>0) else 0 for xt, xh, P in zip(xks[1:], xs_ce[1:], Ps_ce[1:]) ]) 
    cea_bound_realiz = np.mean([ (( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s) if np.all(np.linalg.eig(P)[0]>0) else 0 for xt, xh, P in zip(xks[1:], xs_ce_avg[1:], Ps_ce_avg[1:]) ]) 

    print("The Optimal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_kfo, kfo_bound_realiz) )
    print("The Nominal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_kfn, kfn_bound_realiz) )
    print("The (Best Window) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_ce, ce_bound_realiz) )
    print("The (Averaged) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_cea, cea_bound_realiz) )

    plt.subplot(4,1,1)
    plt.plot(Ts, xks[:,0] - xs_kfo[:,0], 'g')
    plt.plot(Ts, one_sig_kfo[:,0], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,0], 'g--')
    plt.plot(Ts, xks[:,0] - xs_kfn[:,0], 'b')
    plt.plot(Ts, one_sig_kfn[:,0], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,0], 'b--')
    #plt.plot(Ts, xks[:,0] - xs_ce[:,0], 'm')
    #plt.plot(Ts, one_sig_ce[:,0], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,0], 'm--')
    plt.plot(Ts, xks[:,0] - xs_ce_avg[:,0], 'm')
    plt.plot(Ts, one_sig_ce_avg[:,0], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,0], 'k--')
    plt.ylabel("Pos X")
    plt.subplot(4,1,2)
    plt.plot(Ts, xks[:,1] - xs_kfo[:,1], 'g')
    plt.plot(Ts, one_sig_kfo[:,1], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,1], 'g--')
    plt.plot(Ts, xks[:,1] - xs_kfn[:,1], 'b')
    plt.plot(Ts, one_sig_kfn[:,1], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,1], 'b--')
    #plt.plot(Ts, xks[:,1] - xs_ce[:,1], 'm')
    #plt.plot(Ts, one_sig_ce[:,1], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,1], 'm--')
    plt.plot(Ts, xks[:,1] - xs_ce_avg[:,1], 'm')
    plt.plot(Ts, one_sig_ce_avg[:,1], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,1], 'k--')
    plt.ylabel("Vel X")
    plt.subplot(4,1,3)
    plt.plot(Ts, xks[:,2] - xs_kfo[:,2], 'g')
    plt.plot(Ts, one_sig_kfo[:,2], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,2], 'g--')
    plt.plot(Ts, xks[:,2] - xs_kfn[:,2], 'b')
    plt.plot(Ts, one_sig_kfn[:,2], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,2], 'b--')
    #plt.plot(Ts, xks[:,2] - xs_ce[:,2], 'm')
    #plt.plot(Ts, one_sig_ce[:,2], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,2], 'm--')
    plt.plot(Ts, xks[:,2] - xs_ce_avg[:,2], 'm')
    plt.plot(Ts, one_sig_ce_avg[:,2], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,2], 'k--')
    plt.ylabel("Pos Y")
    plt.subplot(4,1,4)
    plt.plot(Ts, xks[:,3] - xs_kfo[:,3], 'g')
    plt.plot(Ts, one_sig_kfo[:,3], 'g--')
    plt.plot(Ts, -one_sig_kfo[:,3], 'g--')
    plt.plot(Ts, xks[:,3] - xs_kfn[:,3], 'b')
    plt.plot(Ts, one_sig_kfn[:,3], 'b--')
    plt.plot(Ts, -one_sig_kfn[:,3], 'b--')
    #plt.plot(Ts, xks[:,3] - xs_ce[:,3], 'm')
    #plt.plot(Ts, one_sig_ce[:,3], 'm--')
    #plt.plot(Ts, -one_sig_ce[:,3], 'm--')
    plt.plot(Ts, xks[:,3] - xs_ce_avg[:,3], 'm')
    plt.plot(Ts, one_sig_ce_avg[:,3], 'k--')
    plt.plot(Ts, -one_sig_ce_avg[:,3], 'k--')
    plt.ylabel("Vel Y")
    plt.xlabel("Time (sec)")
    plt.savefig(fname = fig_path)
    
    ret_dic = {"xks": xks,
               "zks" : zks,
               "vks" : vks,
               "wks" : wks,
               "xs_kfo" : xs_kfo, 
               "Ps_kfo" : Ps_kfo,
               "ae_kfo" : ae_kfo,
               "kfo_bound_realiz" : kfo_bound_realiz,
               "xs_kfn" : xs_kfn, 
               "Ps_kfn" : Ps_kfn,
               "ae_kfn" : ae_kfn,
               "kfn_bound_realiz" : kfn_bound_realiz,
               "xs_ce" : xs_ce, 
               "Ps_ce" : Ps_ce,
               "ae_ce" : ae_ce,
               "ce_bound_realiz" : ce_bound_realiz,
               "xs_ce_avg" : xs_ce_avg, 
               "Ps_ce_avg" : Ps_ce_avg,
               "ae_cea" : ae_cea,
               "cea_bound_realiz" : cea_bound_realiz
               }
    plt.pause(10)
    plt.close('all')
    return ret_dic

# Calls the above for different scenarios and subscenarios
def call_target_track(root_dir = None, subdir_name = "VB_target_track_single_reals"):
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    N = 300
    KNOB_Q = 50 # ENTER ABOVE KNOB_VAL = 5 -> KNOB_QV = KNOB_VAL
    KNOB_V = 50 
    #scenarios = ["0", "1a", "1b", "1c", "2"]
    #V0S_HELS = {"high" : 1, "equal" : 10000, "low" : 1000000} # V0 Scaling High Equal Low #{"high" : 10, "equal" : 1000, "low" : 100000}
    scenarios = ["1c"]
    V0S_HELS = {"high" : 1} # V0 Scaling High Equal Low #{"high" : 10, "equal" : 1000, "low" : 100000}
    num_windows = 5
    for use_scen in scenarios:
        for setting, V0_scale_factor in V0S_HELS.items():
            sub_dir = root_dir+ "/" + "w{}_s{}_".format(num_windows, KNOB_Q) + subdir_name
            if not os.path.isdir(sub_dir):
                os.mkdir(sub_dir)
            scen_tag = "/scen{}_V{}".format(use_scen,setting[0].upper())
            scen_dir = sub_dir+scen_tag
            fig_path = scen_dir + "/state_est_errs.png"
            # Create Read me File for the directory
            if not os.path.isdir(scen_dir):
                os.mkdir(scen_dir)
            data_dic = indep_linear_target_track(N, use_scen, num_windows, V0_scale_factor, KNOB_Q, KNOB_V, fig_path) #linear_target_track
            pickle_path = scen_dir + "/data.pickle"
            with open(pickle_path, "wb") as handle:
                pickle.dump(data_dic, handle)
            user_read_path = scen_dir + "/summary.txt"
            with open(user_read_path, "w") as handle:
                ae_kfo = data_dic["ae_kfo"]
                kfo_bound_realiz = data_dic["kfo_bound_realiz"]
                ae_kfn = data_dic["ae_kfn"]
                kfn_bound_realiz = data_dic["kfn_bound_realiz"]
                ae_cea = data_dic["ae_cea"]
                cea_bound_realiz = data_dic["cea_bound_realiz"]
                ae_ce = data_dic["ae_ce"]
                ce_bound_realiz = data_dic["ce_bound_realiz"]
                ae_vb = data_dic["ae_vb"]
                vb_bound_realiz = data_dic["vb_bound_realiz"]

                l1 = "The Optimal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time\n".format(np.round(ae_kfo,5), np.round(kfo_bound_realiz,8))
                l2 = "The Nominal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time\n".format(np.round(ae_kfn,5), np.round(kfn_bound_realiz,8))
                l3 = "The (Best Window) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time\n".format(np.round(ae_ce,5), np.round(ce_bound_realiz,8))
                l4 = "The (Averaged) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time\n".format(np.round(ae_cea,5), np.round(cea_bound_realiz,8))
                l5 = "The VBayes Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time\n".format(np.round(ae_vb,5), np.round(vb_bound_realiz,8))
                handle.writelines([l1,l2,l3,l4,l5])
            foo = 3
            
def view_data(root_dir = None):
    if root_dir is None:
        root_dir = os.path.dirname(os.path.abspath(__file__))
    subdir_name = "w6_s50_foo_target_track_single_reals"
    N = 300
    T = 1
    KNOB_Q = 50 # ENTER ABOVE KNOB_VAL = 5 -> KNOB_QV = KNOB_VAL
    KNOB_V = 50 
    scenarios = ["0", "1a", "1b", "1c", "2"]
    V0S_HELS = {"high" : 1, "equal" : 10000, "low" : 1000000} # V0 Scaling High Equal Low #{"high" : 10, "equal" : 1000, "low" : 100000}
    #scenarios = ["2"]
    #V0S_HELS = {"high" : 1} # V0 Scaling High Equal Low #{"high" : 10, "equal" : 1000, "low" : 100000}
    
    subdir = root_dir + "/" + subdir_name
    for (root, subsubdir_names, files) in os.walk(subdir): 
        for subsubdir_name in subsubdir_names: 
            print("Loading " + subsubdir_name)
            subsub_dir = subdir + "/" + subsubdir_name
            with open(subsub_dir + "/" + "data.pickle", "rb") as handle:
                data = pickle.load(handle)
            xks = data["xks"]
            xs_kfo = data["xs_kfo"]
            Ps_kfo = data["Ps_kfo"]
            ae_kfo = data["ae_kfo"]
            kfo_bound_realiz = data["kfo_bound_realiz"]
            xs_kfn = data["xs_kfn"]
            Ps_kfn =  data["Ps_kfn"]
            ae_kfn = data["ae_kfn"]
            kfn_bound_realiz = data["kfn_bound_realiz"]
            xs_ce = data["xs_ce"]
            Ps_ce = data["Ps_ce"]
            xs_ce_avg = data["xs_ce_avg"]
            Ps_ce_avg = data["Ps_ce_avg"]
            ae_cea = data["ae_cea"]
            cea_bound_realiz = data["cea_bound_realiz"]
            ce_bound_realiz = data["ce_bound_realiz"]


            Ts = np.arange(N+1)*T 
            # State Error Plot
            scale = 1
            plt.figure(figsize=(12,12))
            plt.suptitle("State Error Plot\nOpt KF=green, Nom KF=blue, Nom Cauchy=magenta/black")
            one_sig_kfn = np.array([scale*np.diag(P)**0.5 for P in Ps_kfn])
            one_sig_kfo = np.array([scale*np.diag(P)**0.5 for P in Ps_kfo])
            one_sig_ce = np.array([scale*np.diag(P)**0.5 for P in Ps_ce])
            one_sig_ce_avg = np.array([scale*np.diag(P)**0.5 for P in Ps_ce_avg])

            # Some Analysis 
            states = 4
            conf_percent = 0.70
            from scipy.stats import chi2 
            s = chi2.ppf(conf_percent, states)
            # ARMSE
            ae_kfo = np.mean( (xks[1:] - xs_kfo[1:])**2, axis=0)
            ae_kfo = np.array([ (ae_kfo[0]+ae_kfo[2])**0.5, (ae_kfo[1]+ae_kfo[3])**0.5 ])
            ae_kfn = np.mean( (xks[1:] - xs_kfn[1:])**2, axis=0)
            ae_kfn = np.array([ (ae_kfn[0]+ae_kfn[2])**0.5, (ae_kfn[1]+ae_kfn[3])**0.5 ])
            ae_ce = np.mean( (xks[1:] - xs_ce[1:])**2, axis=0)
            ae_ce = np.array([ (ae_ce[0]+ae_ce[2])**0.5, (ae_ce[1]+ae_ce[3])**0.5 ])
            ae_cea = np.mean( (xks[1:] - xs_ce_avg[1:])**2, axis=0)
            ae_cea = np.array([ (ae_cea[0]+ae_cea[2])**0.5, (ae_cea[1]+ae_cea[3])**0.5 ])
            kfo_bound_realiz = np.mean([ ( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s for xt, xh, P in zip(xks[1:], xs_kfo[1:], Ps_kfo[1:]) ])
            kfn_bound_realiz = np.mean([ ( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s for xt, xh, P in zip(xks[1:], xs_kfn[1:], Ps_kfn[1:]) ])
            ce_bound_realiz = np.mean([ (( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s) if np.all(np.linalg.eig(P)[0]>0) else 0 for xt, xh, P in zip(xks[1:], xs_ce[1:], Ps_ce[1:]) ]) 
            cea_bound_realiz = np.mean([ (( (xt - xh) @ np.linalg.inv(P) @ (xt - xh) ) < s) if np.all(np.linalg.eig(P)[0]>0) else 0 for xt, xh, P in zip(xks[1:], xs_ce_avg[1:], Ps_ce_avg[1:]) ]) 

            print("The Optimal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_kfo, kfo_bound_realiz) )
            print("The Nominal KF has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_kfn, kfn_bound_realiz) )
            print("The (Best Window) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_ce, ce_bound_realiz) )
            print("The (Averaged) Cauchy Est has average error of {} and its estimate was inside the 70% Conf. Shell {} percent of the time".format(ae_cea, cea_bound_realiz) )

            plt.subplot(4,1,1)
            plt.plot(Ts, xks[:,0] - xs_kfo[:,0], 'g')
            plt.plot(Ts, one_sig_kfo[:,0], 'g--')
            plt.plot(Ts, -one_sig_kfo[:,0], 'g--')
            plt.plot(Ts, xks[:,0] - xs_kfn[:,0], 'b')
            plt.plot(Ts, one_sig_kfn[:,0], 'b--')
            plt.plot(Ts, -one_sig_kfn[:,0], 'b--')
            #plt.plot(Ts, xks[:,0] - xs_ce[:,0], 'm')
            #plt.plot(Ts, one_sig_ce[:,0], 'm--')
            #plt.plot(Ts, -one_sig_ce[:,0], 'm--')
            plt.plot(Ts, xks[:,0] - xs_ce_avg[:,0], 'm')
            plt.plot(Ts, one_sig_ce_avg[:,0], 'k--')
            plt.plot(Ts, -one_sig_ce_avg[:,0], 'k--')
            plt.ylabel("Pos X")
            plt.subplot(4,1,2)
            plt.plot(Ts, xks[:,1] - xs_kfo[:,1], 'g')
            plt.plot(Ts, one_sig_kfo[:,1], 'g--')
            plt.plot(Ts, -one_sig_kfo[:,1], 'g--')
            plt.plot(Ts, xks[:,1] - xs_kfn[:,1], 'b')
            plt.plot(Ts, one_sig_kfn[:,1], 'b--')
            plt.plot(Ts, -one_sig_kfn[:,1], 'b--')
            #plt.plot(Ts, xks[:,1] - xs_ce[:,1], 'm')
            #plt.plot(Ts, one_sig_ce[:,1], 'm--')
            #plt.plot(Ts, -one_sig_ce[:,1], 'm--')
            plt.plot(Ts, xks[:,1] - xs_ce_avg[:,1], 'm')
            plt.plot(Ts, one_sig_ce_avg[:,1], 'k--')
            plt.plot(Ts, -one_sig_ce_avg[:,1], 'k--')
            plt.ylabel("Vel X")
            plt.subplot(4,1,3)
            plt.plot(Ts, xks[:,2] - xs_kfo[:,2], 'g')
            plt.plot(Ts, one_sig_kfo[:,2], 'g--')
            plt.plot(Ts, -one_sig_kfo[:,2], 'g--')
            plt.plot(Ts, xks[:,2] - xs_kfn[:,2], 'b')
            plt.plot(Ts, one_sig_kfn[:,2], 'b--')
            plt.plot(Ts, -one_sig_kfn[:,2], 'b--')
            #plt.plot(Ts, xks[:,2] - xs_ce[:,2], 'm')
            #plt.plot(Ts, one_sig_ce[:,2], 'm--')
            #plt.plot(Ts, -one_sig_ce[:,2], 'm--')
            plt.plot(Ts, xks[:,2] - xs_ce_avg[:,2], 'm')
            plt.plot(Ts, one_sig_ce_avg[:,2], 'k--')
            plt.plot(Ts, -one_sig_ce_avg[:,2], 'k--')
            plt.ylabel("Pos Y")
            plt.subplot(4,1,4)
            plt.plot(Ts, xks[:,3] - xs_kfo[:,3], 'g')
            plt.plot(Ts, one_sig_kfo[:,3], 'g--')
            plt.plot(Ts, -one_sig_kfo[:,3], 'g--')
            plt.plot(Ts, xks[:,3] - xs_kfn[:,3], 'b')
            plt.plot(Ts, one_sig_kfn[:,3], 'b--')
            plt.plot(Ts, -one_sig_kfn[:,3], 'b--')
            #plt.plot(Ts, xks[:,3] - xs_ce[:,3], 'm')
            #plt.plot(Ts, one_sig_ce[:,3], 'm--')
            #plt.plot(Ts, -one_sig_ce[:,3], 'm--')
            plt.plot(Ts, xks[:,3] - xs_ce_avg[:,3], 'm')
            plt.plot(Ts, one_sig_ce_avg[:,3], 'k--')
            plt.plot(Ts, -one_sig_ce_avg[:,3], 'k--')
            plt.ylabel("Vel Y")
            plt.xlabel("Time (sec)")

            plt.show()
            foobar = 2

if __name__ == "__main__":
    #test_linear_target_track()
    call_target_track(root_dir = None)
    #view_data(root_dir = None)