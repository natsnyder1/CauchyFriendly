from distutils.log import debug
import numpy as np 
import cauchy_estimator as ce 
import math, os, pickle
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg',force=True)
import pycauchy
from gmat_sat import *
import numba as nb 

# Process Noise Model 
def leo6_process_noise_model(dt):
    q = 8e-15; # Process noise ncertainty in the process position and velocity
    W = np.zeros((6,6))
    W[0:3,0:3] = q * np.eye(3) * dt**3 / 3 
    W[0:3,3:6] = q * np.eye(3) * dt**2 / 2 
    W[3:6,0:3] = q * np.eye(3) * dt**2 / 2 
    W[3:6,3:6] = q * np.eye(3) * dt 
    return W

# Process Noise Model 
def leo6_process_noise_model2(dt):
    q = 8e-21; # Process noise ncertainty in the process position and velocity
    W = np.zeros((6,6))
    W[0:3,0:3] = q * np.eye(3) * dt**3 / 3 
    W[0:3,3:6] = q * np.eye(3) * dt**2 / 2 
    W[3:6,0:3] = q * np.eye(3) * dt**2 / 2 
    W[3:6,3:6] = q * np.eye(3) * dt 
    return W

# Process Noise Model 
def leo6_process_noise_model3(dt, qs):
    W = np.zeros((6,6))
    W[0:3,0:3] = np.diag(qs) * dt**3 / 3 
    W[0:3,3:6] = np.diag(qs) * dt**2 / 2 
    W[3:6,0:3] = np.diag(qs) * dt**2 / 2 
    W[3:6,3:6] = np.diag(qs) * dt 
    return W

### Testing Cauchy ###
global_STM_taylor_order = 4
global_leo = None
global_date = None
INITIAL_H = False

#mce_naive_p0 = np.array([1,1,1, 0.001,0.001,0.001, 0.001])
mce_naive_p0 = np.array([.1,.1,.1,.001,.001,.001,0.01])
#mce_naive_p0 = 3*np.array([.1,.1,.1,.0006,.0006,.0006,0.0033])
#mce_naive_p0 = np.array([.01,.01,.01,.01,.01,.01,.01])


FACT_LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@nb.njit(cache=True)
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return FACT_LOOKUP_TABLE[n]

# Gets Powers of A**i for i \in [0,taylor_order]
# So if Taylor Order = 2 -> returns [eye(n), A, A @ A]
@nb.njit(cache=True)
def Get_Power_Jacobians(A, taylor_order):
    n = A.shape[0]
    pow_Jacs = [np.eye(n), A.copy()]
    for i in range(2,taylor_order+1):
        pow_Jacs.append(A @ pow_Jacs[-1])
    return pow_Jacs

# Power Series Approximation of the matrix exponential \sum_{i=0}^L Jac**i * dt**i / factorial(i)
@nb.njit(cache=True)
def Get_Matrix_Exponential(pow_Jacs, dt):
    N = len(pow_Jacs)
    Phi = pow_Jacs[0] + pow_Jacs[1] * dt 
    for i in range(2,N):
        Phi += pow_Jacs[i] * dt**i / fast_factorial(i)
    return Phi 

# Power Series Approximation of the integral of the matrix exponential \sum_{i=0}^L Jac**i * dt**(i+1) / factorial(i+1)
@nb.njit(cache=True)
def Get_Integral_Of_Matrix_Exponential(pow_Jacs, dt):
    N = len(pow_Jacs)
    IPhi = pow_Jacs[0] * dt + pow_Jacs[1] * dt**2 / 2 
    for i in range(2,N):
        IPhi += pow_Jacs[i] * dt**(i+1) / fast_factorial(i+1)
    return IPhi 

@nb.njit(cache=True)
def Large_DT_Get_Power_Jacobians_Fast4(As, taylor_order):
    pow_Jack1 = Get_Power_Jacobians(As[0], taylor_order)
    pow_Jack2 = Get_Power_Jacobians(As[1], taylor_order)
    pow_Jack3 = Get_Power_Jacobians(As[2], taylor_order)
    pow_Jack4 = Get_Power_Jacobians(As[3], taylor_order)
    return (pow_Jack1, pow_Jack2, pow_Jack3, pow_Jack4)
    
# Assumes sub_dt is constant over each substep -> 4*sub_dt == dt
@nb.njit(cache=True)
def Large_DT_Phi_Fast4(pow_Jacks4, sub_dt):
    pow_Jack1, pow_Jack2, pow_Jack3, pow_Jack4 = pow_Jacks4
    Phi1 = Get_Matrix_Exponential(pow_Jack1, sub_dt)
    Phi2 = Get_Matrix_Exponential(pow_Jack2, sub_dt)
    Phi3 = Get_Matrix_Exponential(pow_Jack3, sub_dt)
    Phi4 = Get_Matrix_Exponential(pow_Jack4, sub_dt)
    return Phi4 @ Phi3 @ Phi2 @ Phi1
    
# Assumes sub_dt is constant over each substep -> 4*sub_dt == dt
@nb.njit(cache=True)
def Large_DT_Gamma_Fast4(pow_Jacks4, sub_dt):
    num_subs = 4
    assert(len(pow_Jacks4) == num_subs)
    assert(num_subs == 4)
    n = pow_Jacks4[0][0].shape[0]
    tmp = np.zeros(n)
    taylor_order = len(pow_Jacks4[0])-1
    pow_Jack1, pow_Jack2, pow_Jack3, pow_Jack4 = pow_Jacks4
    for i in range(taylor_order+1):
        cache_Ai = pow_Jack4[i]
        cache_sub_dt_powi = float(i+1)
        cache_facti = fast_factorial(i)
        for j in range(taylor_order+1):
            cache_Aj = cache_Ai @ pow_Jack3[j]
            cache_sub_dt_powj = cache_sub_dt_powi + j 
            cache_factj = cache_facti * fast_factorial(j)
            for k in range(taylor_order+1):
                cache_Ak = cache_Aj @ pow_Jack2[k]
                cache_sub_dt_powk = cache_sub_dt_powj + k
                cache_factk = cache_factj * fast_factorial(k)
                for l in range(taylor_order+1):
                    tmp += ( cache_Ak @ pow_Jack1[l][:,-1] ) * sub_dt**(cache_sub_dt_powk + l) \
                        / ( cache_factk * fast_factorial(l) * (cache_sub_dt_powk + l) )
    return num_subs * tmp

# As must be provided as a list in the order [A1, A2, A3, A4]
# where Phi_Total = Phi(A4) @ Phi(A3) @ Phi(A2) @ Phi(A1)
# Assumes DT is constant over each substep
@nb.njit(cache=True)
def work_Large_DT_Get_Phi_Gam_Fast4(As, sub_dt, taylor_order):
    pow_Jacks4 = Large_DT_Get_Power_Jacobians_Fast4(As, taylor_order)
    Phi_k = Large_DT_Phi_Fast4(pow_Jacks4, sub_dt)
    Gamma_k = Large_DT_Gamma_Fast4(pow_Jacks4, sub_dt)
    return Phi_k, Gamma_k 

def Regular_DT_Get_xbar_Phi_Gam():
    # Set Phi and Gamma  
    Jac = global_leo.get_jacobian_matrix()
    Jac[3:6,6] *= 1000 # km to m
    dt = global_leo.dt
    pow_Jacs = Get_Power_Jacobians(Jac, global_STM_taylor_order)
    Phi_k = Get_Matrix_Exponential(pow_Jacs, dt)
    IPhi_k = Get_Integral_Of_Matrix_Exponential(pow_Jacs, dt)
    Gamma_k = IPhi_k[:, -1]
    x_bar = global_leo.step()
    x_bar[0:6] *= 1000 # km -> m
    return x_bar, Phi_k, Gamma_k

def Large_DT_Get_xbar_Phi_Gam_Fast4():
    global global_leo
    dt = global_leo.dt
    num_substeps = 4 
    assert(num_substeps == 4)
    sub_dt = dt / num_substeps
    global_leo.dt = sub_dt
    As = []  # Jacobians
    for i in range(num_substeps):
        A = global_leo.get_jacobian_matrix()
        A[3:6,6] *= 1000 # km to m
        As.append(A)
        x_bar = global_leo.step()
    Phi_k, Gamma_k = work_Large_DT_Get_Phi_Gam_Fast4(As, sub_dt, global_STM_taylor_order)
    global_leo.dt = dt
    x_bar[0:6] *= 1000 # km -> m
    return x_bar, Phi_k, Gamma_k

def ece_dynamics_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    global global_leo
    if global_leo.dt < 70:
        x_bar, Phi_k, Gamma_k = Regular_DT_Get_xbar_Phi_Gam()
    else:
        x_bar, Phi_k, Gamma_k = Large_DT_Get_xbar_Phi_Gam_Fast4() # divys the large dt into four pieces
    pyduc.cset_Phi(Phi_k)
    pyduc.cset_Gamma(Gamma_k)
    pyduc.cset_x(x_bar)
    pyduc.cset_is_xbar_set_for_ece()

def ece_nonlinear_msmt_model(c_duc, c_zbar):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    global INITIAL_H
    if(INITIAL_H):
        zbar = np.array([0, 0, xbar[0] + xbar[1] + xbar[2]])
    else:
        zbar = xbar[0:3]
    pyduc.cset_zbar(c_zbar, zbar)

def ece_extended_msmt_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    #xbar = pyduc.cget_x() # xbar
    global INITIAL_H
    if INITIAL_H:
        H = np.zeros((3,7))
        H[2,0] = 1
        H[2,1] = 1
        H[2,2] = 1
        global global_leo 
        gam = global_leo.gamma0[2]
        gamma = np.array([gam, gam, 3*gam])
        pyduc.cset_gamma(gamma)
    else:
        H = np.hstack(( np.eye(3), np.zeros((3,4)) ))
    pyduc.cset_H(H)

def ece_gps_ebf_h_model(c_duc, c_zbar): #state_eci_to_ebf(self, date):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    global global_leo, global_date, INITIAL_H 
    state = pyduc.cget_x()
    # Create Transformed GPS Msmt in Earth MJ2000Eq Coordinates
    zkbar = transform_coordinate_system(state[0:3], global_date, mode="ei2b", sat_handle = global_leo)
    if(INITIAL_H):
        zkbar = np.array([0, 0, np.sum(zkbar)])
    pyduc.cset_zbar(c_zbar, zkbar)

def ece_gps_ebf_H_model(c_duc):
    global global_date, global_leo, INITIAL_H
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    # Create Transformed GPS Msmt in Earth MJ2000Eq Coordinates
    #state = pyduc.cget_x()
    #_H = transform_coordinate_system_jacobian_H(state[0:3], global_date, mode ="ei2b", sat_handle = global_leo)
    H = np.hstack( (np.array(list(global_leo.csConverter.GetLastRotationMatrix().GetDataVector())).reshape((3,3)), np.zeros((3,4)) ) )
    if INITIAL_H:
        H[2,:] = np.sum(H, axis=0)
        H[:2,:] *= 0
        gam = global_leo.gamma0
        gamma = np.array([0, 0, np.sum(gam)])
        pyduc.cset_gamma(gamma)
    pyduc.cset_H(H)

def plot_all_windows(win_moms, xs_true, e_hats_kf, one_sigs_kf, best_idx, idx_min):
    W = len(win_moms)
    Ts_kf = np.arange(e_hats_kf.shape[0])
    for win_idx in range(W):
        if len(win_moms[win_idx]) > 1: #k > win_idx:
            x_hats = np.array([ win_moms[win_idx][i][0] for i in range(len(win_moms[win_idx])) ])
            P_hats = np.array([ win_moms[win_idx][i][1] for i in range(len(win_moms[win_idx])) ])
            T_cur = win_idx + x_hats.shape[0] + 1
            one_sigs = np.array([np.sqrt(np.diag(P_hat)) for P_hat in P_hats])
            e_hats = np.array([xt - xh for xt,xh in zip(xs_true[win_idx+1:T_cur], x_hats)])
            
            plt.figure()
            plt.subplot(711)
            plt.title("Win Err" + str(win_idx) + " PosX/PosY/VelX/VelY")
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,0], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,0], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,0], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 0], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 0], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 0], 'm')
            # Plot a black star indicating this window reinitialized the other
            if win_idx == best_idx:
                plt.scatter(Ts_kf[T_cur-1], one_sigs[-1,0], color='k', marker='*')
                plt.scatter(Ts_kf[T_cur-1], -one_sigs[-1,0], color='k', marker='*')
            if win_idx == idx_min:
                plt.scatter(Ts_kf[T_cur-1], one_sigs[-1,0], color='k', marker='o')
                plt.scatter(Ts_kf[T_cur-1], -one_sigs[-1,0], color='k', marker='o')
            plt.subplot(712)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,1], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,1], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,1], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 1], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 1], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 1], 'm')
            plt.subplot(713)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,2], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,2], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,2], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 2], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 2], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 2], 'm')
            plt.subplot(714)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,3], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,3], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,3], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 3], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 3], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 3], 'm')
            plt.subplot(715)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,4], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,4], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,4], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 4], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 4], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 4], 'm')
            plt.subplot(716)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,5], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,5], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,5], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 5], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 5], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 5], 'm')
            plt.subplot(717)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,6], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,6], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,6], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 6], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 6], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 6], 'm')
    plt.show()
    plt.close('all')

def plot_against_kf(mce_msmt_idxs, xs, xs_kf, Ps_kf, xs_mce, Ps_mce, xs_avg_mce, Ps_avg_mce, kf_dt, mce_dt, sig = 1, title_prefix = ''):
    true_x_mce = xs[mce_msmt_idxs]
    true_x_kf = xs[mce_msmt_idxs[0]:]


    xt_kf = xs_kf[mce_msmt_idxs[0]:]
    Pt_kf = Ps_kf[mce_msmt_idxs[0]:]
    sig_kf = np.array([np.diag(P)**0.5 for P in Pt_kf]) * sig
    es_kf = true_x_kf - xt_kf

    with_mce = False
    if (Ps_mce is not None ) and (xs_mce is not None):
        with_mce = True
        sig_mce = np.array([np.diag(P)**0.5 for P in Ps_mce]) * sig
        es_mce = true_x_mce - xs_mce
    with_avg_mce = False 
    if (Ps_avg_mce is not None ) and (xs_avg_mce is not None):
        with_avg_mce = True
        sig_avg_mce = np.array([np.diag(P)**0.5 for P in Ps_avg_mce]) * sig
        es_avg_mce = true_x_mce - xs_avg_mce

    kf_msmt_idxs = np.arange(mce_msmt_idxs[0], mce_msmt_idxs[-1]+1, 1)
    N = xs.shape[1]

    plt.figure()
    plt.suptitle(title_prefix + " Estimation Errors Vs {}-Sigma Bounds\nKF (g/m) vs MCE (b/r) vs Weighted Avg MCE (b--/r--)".format(sig))
    ylabels = ['Pos X (km)', 'Pos Y (km)', 'Pos Z (km)', 'Vel X (km/s)', 'Vel Y (km/s)', 'Vel Z (km/s)', 'Change \nAtms Dens']
    for i in range(N):
        plt.subplot(N, 1, i+1)
        # Kalman Filter
        plt.plot(kf_msmt_idxs, es_kf[:,i], 'g')
        plt.scatter(kf_msmt_idxs, es_kf[:,i], color='g')
        plt.plot(kf_msmt_idxs, sig_kf[:,i], 'm')
        plt.plot(kf_msmt_idxs, -sig_kf[:,i], 'm')
        if with_mce:
            # MCE
            plt.plot(mce_msmt_idxs, es_mce[:,i], 'b')
            plt.scatter(mce_msmt_idxs, es_mce[:,i], color='b')
            plt.plot(mce_msmt_idxs, sig_mce[:,i], 'r')
            plt.plot(mce_msmt_idxs, -sig_mce[:,i], 'r')
        if with_avg_mce:
            # Weighted Avg MCE 
            plt.plot(mce_msmt_idxs, es_avg_mce[:,i], 'b--')
            plt.scatter(mce_msmt_idxs, es_avg_mce[:,i], color='b', linestyle='dashed')
            plt.plot(mce_msmt_idxs, sig_avg_mce[:,i], 'r--')
            plt.plot(mce_msmt_idxs, -sig_avg_mce[:,i], 'r--')
        plt.ylabel(ylabels[i])
    plt.xlabel('Time Step k (KF dt={}, MCE dt={})'.format(kf_dt, mce_dt))
    plt.show()
    foobar=2
    
def reinitialize_func_speyer(cauchyEsts, zk, best_idx, idx_min, step_k, other_params):
    # using speyer's start method
    speyer_restart_idx = -1
    cauchyEsts[idx_min].reset_about_estimator(cauchyEsts[best_idx], msmt_idx = speyer_restart_idx)

def reinitialize_func_init_cond(cauchyEsts, zk, best_idx, idx_min, step_k, other_params):
    xhat, Phat = cauchyEsts[best_idx].get_last_mean_cov()
    _A0 = cauchyEsts[best_idx]._Phi.copy().reshape((7,7)).T # np.eye(5) # This will not be correct for the "deterministic prop" + stochastic prop -> fix gmat MCE to do so
    global mce_naive_p0
    _p0 = mce_naive_p0.copy()
    _b0 = np.zeros(7)
    #cauchyEsts[idx_min].reset(_A0, _p0, _b0, xhat)
    #cauchyEsts[idx_min].step(zk)
    cauchyEsts[idx_min].reset_with_last_measurement(zk[2], _A0, _p0, _b0, xhat)
    
def reinitialize_func_H_summation(cauchyEsts, zk, best_idx, idx_min, step_k, other_params):
    assert other_params is not None
    # Both H channels concatenated
    xhat, Phat = cauchyEsts[best_idx].get_last_mean_cov()
    _H = np.array([1.0, 1.0, 1.0, 0, 0, 0, 0])
    _gamma = 3 * cauchyEsts[best_idx]._gamma[0]
    _xbar = cauchyEsts[best_idx]._xbar[14:]
    _dz = (zk[0] + zk[1] + zk[2]) - (_xbar[0] + _xbar[1] + _xbar[2])
    _dx = xhat - _xbar
    _P = Phat.copy()
    #'''
    Ps_kf = other_params
    P_kf = Ps_kf[step_k].copy() # The KF is in km^2 and we are in m^2
    P_kf[0:6,0:6] *= 1000**2
    ratios = np.ones(6)
    for i in range(6):
        pkf = P_kf[i,i] #* 1000**2
        pce = Phat[i,i]
        ratios[i] = pkf/pce
        if( ratios[i] > 1):
            _P[i,i] *= ratios[i] * 4
    #'''
    # Reset
    _A0, _p0, _b0 = ce.speyers_window_init(_dx, _P, _H, _gamma, _dz)
    global INITIAL_H
    INITIAL_H = True
    cauchyEsts[idx_min].reset_with_last_measurement(zk[0] + zk[1] + zk[2], _A0, _p0, _b0, _xbar)
    pyduc = cauchyEsts[idx_min].get_pyduc()
    pyduc.cset_gamma(cauchyEsts[best_idx]._gamma)
    INITIAL_H = False

def reinitialize_func_H_summation_ebf(cauchyEsts, zk, best_idx, idx_min, step_k, other_params):
    global global_leo, global_date 
    assert other_params is not None
    #Ps_kf, xhat_avg, Phat_avg = other_params # can reinitialize with the averaged data
    #xhat = xhat_avg
    #Phat = Phat_avg
    Ps_kf, _, _ = other_params

    # Both H channels concatenated
    xhat, Phat = cauchyEsts[best_idx].get_last_mean_cov()
    transform_coordinate_system(xhat[0:3]/1000, global_date, mode="ei2b", sat_handle = global_leo)
    H = np.hstack( (np.array(list(global_leo.csConverter.GetLastRotationMatrix().GetDataVector())).reshape((3,3)), np.zeros((3,4)) ) )
    _H = np.sum(H, axis = 0)
    _gamma = np.sum(cauchyEsts[best_idx]._gamma)
    _xbar = cauchyEsts[best_idx]._xbar[14:]
    _dz = np.sum(zk) - np.sum( H @ _xbar )
    _dx = xhat - _xbar
    _P = Phat.copy()
    #'''
    P_kf = Ps_kf[step_k].copy() # The KF is in km^2 and we are in m^2
    P_kf[0:6,0:6] *= 1000**2
    ratios = np.ones(6)
    for i in range(6):
        pkf = P_kf[i,i] #* 1000**2
        pce = Phat[i,i]
        ratios[i] = pkf/pce
        if( ratios[i] > 1):
            _P[i,i] *= ratios[i] * 4
    #'''
    # Reset
    #print("THIS FUNCTION IS NOT FINISHED AND MUST BE WORKED FURTHER!")
    #assert(False)
    _A0, _p0, _b0 = ce.speyers_window_init(_dx, _P, _H, _gamma, _dz)
    global INITIAL_H
    INITIAL_H = True
    cauchyEsts[idx_min].reset_with_last_measurement(np.sum(zk), _A0, _p0, _b0, _xbar)
    pyduc = cauchyEsts[idx_min].get_pyduc()
    pyduc.cset_gamma(cauchyEsts[best_idx]._gamma)
    INITIAL_H = False

class GmatMCE():

    def __init__(self, num_windows, t0, x0, dt, A0, p0, b0, beta, gamma, 
                 Cd_dist="gauss", std_Cd = 0.0013, tau_Cd = 21600, Cd_nominal = 2.1,
                 EDIT_CHNG_ATM_DENS_LOW = -0.98, EDIT_CHNG_ATM_DENS_HIGH = 10,
                 win_reinitialize_func=None, win_reinitialize_params=None, debug_print = True, mce_print = False, with_ei2b = False):
        # Print Out Some Info 
        print("Note that the MCE uses meters for the state and measurements, and not KM")
        Cd_dist = Cd_dist.lower()
        assert Cd_dist in ["gauss", "sas"]
        self.num_windows = num_windows
        self.dt = dt 
        self.win_moms = { i : [] for i in range(num_windows) }
        self.debug_print = debug_print
        self.mce_print = mce_print
        self.Cd_dist = Cd_dist
        self.std_Cd = std_Cd 
        self.tau_Cd = tau_Cd
        self.Cd_nominal = Cd_nominal

        self.xhat = x0.copy() 
        self.Phat = np.diag(p0 * 1.3898)**2 
        self.partial_step_boolean = False
        self.best_idx = 0
        self.with_ei2b = with_ei2b
        self.EDIT_STATE_LOW = EDIT_CHNG_ATM_DENS_LOW
        self.EDIT_STATE_HIGH = EDIT_CHNG_ATM_DENS_HIGH
        self.EDIT_STATE_INDEX = 6
        assert(EDIT_CHNG_ATM_DENS_LOW >= -0.999)
        assert(EDIT_CHNG_ATM_DENS_HIGH > 0)

        # Setup GMAT Fermi Satellite Object -- internally GMAT uses KM, so need conversions
        gmat.Clear()
        global global_leo
        _x0 = x0.copy() # meters
        _x0[0:6] /= 1000 # meters to km
        global_leo = FermiSatelliteModel(t0, _x0[0:6], dt, gmat_print = debug_print)
        global_leo.create_model(with_jacchia=True, with_SRP = True, Cd0 = self.Cd_nominal)
        global_leo.set_solve_for(field="Cd", dist=Cd_dist, scale=std_Cd, tau=tau_Cd, alpha=2.0 if Cd_dist == "gauss" else 1.3)
        global_leo.reset_state(_x0, 0)
        ce.set_tr_search_idxs_ordering([5,4,6,3,2,1,0])
        global_leo.gamma0 = gamma.copy() 
        global_leo.beta0 = beta.copy() 
        
        # Setup Windows
        self.cauchyEsts = [ce.PyCauchyEstimator("nonlin", num_windows, mce_print) for _ in range(num_windows)]
        for i in range(num_windows):
            if with_ei2b:
                self.cauchyEsts[i].initialize_nonlin(x0, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_gps_ebf_h_model, ece_gps_ebf_H_model, 0) 
            else:
                self.cauchyEsts[i].initialize_nonlin(x0, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, 0)
            self.cauchyEsts[i].set_window_number(i)

        # Set window reinitialization function if one has not been provided
        if win_reinitialize_func is None:
            self.win_reinitialize_func = reinitialize_func_speyer
        else:    
            self.win_reinitialize_func = win_reinitialize_func
        self.win_reinitialize_params = win_reinitialize_params

        # Setup counters 
        self.first_mu = True
        self.win_idxs = np.arange(num_windows)
        self.win_counts = np.zeros(num_windows, dtype=np.int64)
        self.k = 0 # step counter

        # Setup data storage
        self.xhats = [] 
        self.Phats = [] 
        self.avg_xhats = [] 
        self.avg_Phats = [] 
    
    def _best_window_est(self):
        W = self.num_windows
        okays = np.zeros(W, dtype=np.bool8)
        idxs = []
        for i in range(W):
            if(self.win_counts[i] > 0):
                err = self.cauchyEsts[i]._err_code
                if (err[2] & (1<<1)) or (err[2] & (1<<3)):
                    pass
                else:
                    idxs.append( (i, self.win_counts[i]) )
                    okays[i] = True
        if(len(idxs) == 0):
            print("No window is available without an error code!")
            exit(1)
        sorted_idxs = list(reversed(sorted(idxs, key = lambda x : x[1])))
        return sorted_idxs[0][0], okays
    
    def _edit_means(self, EDIT_STATE_INDEX, EDIT_STATE_LOW, EDIT_STATE_HIGH):
        n = self.cauchyEsts[0].n
        assert(EDIT_STATE_INDEX < self.cauchyEsts[0].n)
        for win_num in range(self.num_windows):
            if self.win_counts[win_num] > 0:
                cauchyEst = self.cauchyEsts[win_num]
                x = cauchyEst.moment_info["x"][-1]
                if (x[EDIT_STATE_INDEX] < EDIT_STATE_LOW) or (x[EDIT_STATE_INDEX] > EDIT_STATE_HIGH):
                    xi_corrected = np.clip(x[EDIT_STATE_INDEX], EDIT_STATE_LOW, EDIT_STATE_HIGH)
                    print("Window {}/{} Underwent Editing! Atms. Density Edited from {} to {}".format(win_num+1, self.num_windows, cauchyEst.moment_info["x"][-1][EDIT_STATE_INDEX], xi_corrected))
                    cauchyEst.moment_info["x"][-1][EDIT_STATE_INDEX] = xi_corrected
                    # Now deal with pyduc
                    pyduc = cauchyEst.get_pyduc()
                    x = pyduc.cget_x()
                    x[EDIT_STATE_INDEX] = xi_corrected
                    pyduc.cset_x(x)
                    # Now deal with xhats and xbars
                    lxh = cauchyEst._x.size
                    c = EDIT_STATE_INDEX
                    while c < lxh:
                        cauchyEst._x[c] = xi_corrected
                        c += n
                    lxb = cauchyEst._xbar.size
                    c = EDIT_STATE_INDEX
                    while c < lxb:
                        cauchyEst._xbar[c] = xi_corrected
                        c += n
                    self.win_moms[win_num][-1][0][EDIT_STATE_INDEX] = xi_corrected
    
    def _weighted_average_win_est(self, usable_wins):
            win_avg_mean = np.zeros(7)
            win_avg_cov = np.zeros((7,7))
            win_norm_fac = 0.0
            for i in range(self.num_windows):
                win_count = self.win_counts[i]
                if win_count > 0:
                    win_okay = usable_wins[i]
                    if win_okay:
                        norm_fac = win_count / self.num_windows
                        win_norm_fac += norm_fac
                        win_avg_mean += self.win_moms[i][-1][0] * norm_fac
                        win_avg_cov += self.win_moms[i][-1][1] * norm_fac
            win_avg_mean /= win_norm_fac
            win_avg_cov /= win_norm_fac
            return win_avg_mean, win_avg_cov

    # zk is assumed to be GPS coordinates in the ECI (Earth Centered Inertial) frame for this function
    # ellapsed time and dt_step are used for actual GPS measurements, where a coorfinate frame transformation needs to happen
    def step(self, zk, x_truth = None, is_inputs_meters = True, last_step = False, ellapsed_time = None, dt_step = None, reset_mean = None, print_state_innovation = True):
        # Need both ellapsed_time and dt_step for GPS measurement processing
        if( (ellapsed_time is not None) or (dt_step is not None) ):
            assert( (ellapsed_time is not None) and (dt_step is not None) )

        if not is_inputs_meters:
            _zk = zk.copy() * 1000 # Convert km to meters
            if x_truth is not None:
                _x_truth = x_truth.copy()
                _x_truth[0:6] *= 1000
        else:
            _zk = zk.copy() # otherwise OK
            if x_truth is not None:
                _x_truth = x_truth.copy()
        
        global global_leo
        if self.first_mu:
            self.first_mu = False 
            xw,Pw = self.cauchyEsts[0].step(_zk, None, False) # window state, window covariance
            if print_state_innovation:
                z_resid = _zk - self.cauchyEsts[0]._zbar
                x_resid = xw - self.cauchyEsts[0]._xbar[0:7] #xw == self.cauchyEsts[win_idx]._x[14:]
                print("Win{} Innov:\n  xhat-xbar={} (meters)\n  z-zbar={} (meters)".format(1, np.round(x_resid, decimals=6), np.round(z_resid, decimals=3)))
            self.win_moms[0].append( (xw,Pw) )
            self.win_counts[0] += 1
            best_idx = 0
            usable_wins = np.zeros(self.num_windows, dtype=np.bool8)
            usable_wins[0] = True
        else:
            # find max and min indices
            idx_max = np.argmax(self.win_counts)
            idx_min = np.argmin(self.win_counts)
            # Step all windows that are not uninitialized
            for win_idx, win_count in zip(self.win_idxs, self.win_counts):
                if(win_count > 0):
                    if self.debug_print:
                        print("  Window {} is on step {}/{}".format(win_idx+1, win_count+1, self.num_windows) )
                    # reset state of window for Fermi GMAT propagator
                    if reset_mean is None:
                        xwr = self.win_moms[win_idx][-1][0].copy() 
                        xwr[0:6] /= 1000 # convert from meters to km
                        xwr[6] = np.clip(xwr[self.EDIT_STATE_INDEX], self.EDIT_STATE_LOW, self.EDIT_STATE_HIGH) # make sure that change in atms. density is in bounds and reasonable
                    else:
                        xwr = reset_mean.copy()
                        assert(False) # CANNOT USE UNTIL DEBUGGED FURTHER

                    if ellapsed_time is not None:
                        self.cauchyEsts[win_idx].get_pyduc().cset_dt(dt_step)
                        global_leo.dt = dt_step
                        global_leo.reset_state_with_ellapsed_time(xwr, ellapsed_time)
                    else:
                        global_leo.reset_state(xwr, self.k)
                    xw, Pw = self.cauchyEsts[win_idx].step(_zk, None, False)
                    if self.partial_step_boolean:
                        self.win_moms[win_idx].pop()
                    self.win_moms[win_idx].append( (xw, Pw ) )
                    if self.debug_print:
                        print("    x_k|k:   ", xw)
                    if print_state_innovation:
                        z_resid = _zk - self.cauchyEsts[win_idx]._zbar
                        x_resid = xw - self.cauchyEsts[win_idx]._xbar[0:7] #xw == self.cauchyEsts[win_idx]._x[14:]
                        print("Win{} Innov:\n  xhat-xbar={} (meters)\n  z-zbar={} (meters)".format(win_idx+1, np.round(x_resid, decimals=6), np.round(z_resid, decimals=3)))
                    if x_truth is not None:
                        print("    e_k|k:   ", _x_truth - xw)
                    self.win_counts[win_idx] += 1
            # Now reinitialize the empty window about the best estimate
            best_idx, usable_wins = self._best_window_est()
            self.usable_wins = usable_wins
            self.best_idx = best_idx
            #best_idx, usable_wins = idx_max, np.zeros(self.num_windows, dtype=np.bool) 
            #for _ in range(self.num_windows):
            #    if _ <= self.k:
            #        usable_wins[_] = True
            # Increment step count
            self.k += 1

        if self.debug_print:
            print("Best Window Index For Reinit is: Window {}/{}, which has undergone {}/{} steps".format(best_idx+1, self.num_windows, self.win_counts[best_idx], self.num_windows) )
        
        # Edit all windows which ran to assure that their atms. density are OK and not under or above the limits
        EDIT_STATE_INDEX = self.EDIT_STATE_INDEX
        EDIT_STATE_LOW = self.EDIT_STATE_LOW #-0.05
        EDIT_STATE_HIGH = self.EDIT_STATE_HIGH #0.05
        self._edit_means(EDIT_STATE_INDEX, EDIT_STATE_LOW, EDIT_STATE_HIGH)
        
        # Store away the best moment info
        self.xhat, self.Phat = self.cauchyEsts[best_idx].get_last_mean_cov()
        self.xhats.append(self.xhat.copy())
        self.Phats.append(self.Phat.copy())

        # Compute Weighted Average Window Estimate
        self.avg_xhat, self.avg_Phat = self._weighted_average_win_est(usable_wins)
        self.avg_xhats.append(self.avg_xhat)
        self.avg_Phats.append(self.avg_Phat) 

        # Reinitialize empty window using the chosen window reinitialization strategy
        if self.k > 0:
            if self.with_ei2b:
                self.win_reinitialize_func(self.cauchyEsts, _zk, best_idx, idx_min, self.k, (self.win_reinitialize_params, self.avg_xhat, self.avg_Phat)  )
            else:
                self.win_reinitialize_func(self.cauchyEsts, _zk, best_idx, idx_min, self.k, self.win_reinitialize_params )
            self.win_counts[idx_min] += 1
            xw,Pw = self.cauchyEsts[idx_min].get_last_mean_cov()
            if self.partial_step_boolean and (self.k > idx_min):
                self.win_moms[idx_min][-1] = (xw, Pw )
            else:
                self.win_moms[idx_min].append( (xw, Pw ) )
            if self.debug_print:
                print("Window {}/{} was reinitialized!".format(idx_min+1, self.num_windows))
            # Tear down most full window
            if self.win_counts[idx_max] == self.num_windows:
                if not last_step:
                    self.cauchyEsts[idx_max].reset()
                    self.win_counts[idx_max] = 0
        if self.partial_step_boolean:
            self.partial_step_boolean = False # reset partial flag boolean if turned to True --> This doesnt need an if condition but keeping for clarity
            
    # Propagate the denisty function to TCA
    def pred_to_tca(self, pred_t0, pred_dt, i_star_lhs, t_lhs, t_c, max_terms = np.inf, with_propagate_drag_estimate=True, xhat_pred_t0 = None):
        # Find the estimator which has less terms than the max specified, and is 
        best_win_terms = -1
        best_win_idx = -1
        for i in range(self.num_windows):
            if self.usable_wins[i] and (self.win_counts[i] < self.num_windows):
                win_i_terms = self.cauchyEsts[i].get_num_CF_terms()
                if (win_i_terms < max_terms) and (win_i_terms > best_win_terms):
                    best_win_terms = win_i_terms
                    best_win_idx = i
        if best_win_idx == -1:
            print("[GMAT MCE: pred_to_tca] All Characteristic Functions have number of terms greater than the max allowable")
            exit(1)
        
        # Deterministic piece to propagate using the GMATSat class
        cauchyEst = self.cauchyEsts[best_win_idx]
        if xhat_pred_t0 is None:
            xhat, _ = cauchyEst.get_last_mean_cov()
            xhat[0:6] /= 1000 # m to km
        else:
            xhat = xhat_pred_t0.copy() 
        xhat[6] *= with_propagate_drag_estimate

        # Need to accumulate sequence of Phis from pred_t0 to TCA so we can propagate the denisty function itself to TCA
        global global_leo
        global_leo.clear_model()
        global_leo = FermiSatelliteModel(pred_t0, xhat[0:6], pred_dt, gmat_print=False)
        global_leo.create_model(True, True, Cd0 = self.Cd_nominal)
        global_leo.set_solve_for(field="Cd", dist=self.Cd_dist, scale=self.std_Cd, tau=self.tau_Cd, alpha = 2.0 if self.Cd_dist == "gauss" else 1.3)
        global_leo.reset_state(xhat, 0)

        # Set running Phi from start to i_star_lhs (t_lhs) 
        Phi_total = np.eye(7)
        for i in range(i_star_lhs):
            Jac = global_leo.get_jacobian_matrix()
            Jac[3:6,6] *= 1000 # km to m
            Phi_k = np.eye(7) + Jac * global_leo.dt
            for i in range(2,global_STM_taylor_order+1):
                Phi_k += np.linalg.matrix_power(Jac, i) * global_leo.dt**i / math.factorial(i)
            Phi_total = Phi_k @ Phi_total
            global_leo.step()

        # Set running Phi for last partial step t_lhs to t_c
        new_step_dt = t_c - t_lhs 
        Jac = global_leo.get_jacobian_matrix()
        Jac[3:6,6] *= 1000 # km to m
        Phi_k = np.eye(7) + Jac * new_step_dt
        for i in range(2,global_STM_taylor_order+1):
            Phi_k += np.linalg.matrix_power(Jac, i) * new_step_dt**i / math.factorial(i)
        Phi_total = Phi_k @ Phi_total
        
        old_dt = global_leo.dt
        global_leo.dt = new_step_dt
        cauchyEst.tca_xhat = global_leo.step()
        global_leo.dt = old_dt
        cauchyEst.tca_xhat[0:6] *= 1000 # km to m
        cauchyEst.deterministic_transform(Phi_total, cauchyEst.tca_xhat)
        cauchyEst.tca_Phat = Phi_total @ cauchyEst.get_last_mean_cov()[1] @ Phi_total.T
        return best_win_idx

    def no_msmt_update_moment_append(self, i, xprop, P_prop, with_window_append = True):
        self.win_moms[i].append( (xprop.copy(), P_prop.copy()) )
        if with_window_append:
            self.cauchyEsts[i].moment_info["fz"].append(1)
            self.cauchyEsts[i].moment_info["x"].append(xprop)
            self.cauchyEsts[i].moment_info["P"].append(P_prop)
            self.cauchyEsts[i].moment_info["cerr_x"].append(0)
            self.cauchyEsts[i].moment_info["cerr_P"].append(0)
            self.cauchyEsts[i].moment_info["cerr_fz"].append(0)
            self.cauchyEsts[i].moment_info["err_code"].append(0)
        if i == self.best_idx:
            # Store away the best moment info
            self.xhats.append(xprop.copy())
            self.Phats.append(P_prop.copy())
            self.avg_xhats.append(xprop.copy())
            self.avg_Phats.append(P_prop.copy()) 
            self.xhat = xprop.copy() 
            self.Phat = P_prop.copy()
            self.avg_xhat = xprop.copy() 
            self.avg_Phat = P_prop.copy()

    def no_msmt_update_moment_overwrite(self, i, xprop, P_prop):
        self.cauchyEsts[i].moment_info["x"][-1] = xprop.copy()
        self.cauchyEsts[i].moment_info["P"][-1] = P_prop.copy()
        self.win_moms[i][-1] = (xprop.copy(), P_prop.copy())
        if i == self.best_idx:
            self.xhats[-1] = xprop.copy()
            self.Phats[-1] = P_prop.copy()
            self.avg_xhats[-1] = xprop.copy()
            self.avg_Phats[-1] = P_prop.copy()
            self.xhat = xprop.copy() 
            self.Phat = P_prop.copy()
            self.avg_xhat = xprop.copy()
            self.avg_Phat = P_prop.copy()
    
    def reset_beta(self, beta):
        for i in self.win_counts:
            pyduc = self.cauchyEsts[i].get_pyduc()
            pyduc.cset_beta(beta)

    # Propagates all windows deterministically 
    # if with_append_prop = True -> appends propagated to lists
    # elif with_overwrite_last = True -> modifies propagated to lists
    # else partial det TP + partial step -> appends to win_moms temporarily, sets temp win_mom boolean -> signals deletion after reloading
    def deterministic_time_prop(self, ellapsed_time, dt_sub_steps, with_append_prop = True, with_overwrite_last = False):
        assert( not (with_append_prop and with_overwrite_last) )
        global global_leo
        if self.first_mu:
            x_reset = self.xhat.copy()
            x_reset[0:6] /= 1000
            global_leo.reset_state_with_ellapsed_time(x_reset, ellapsed_time)
            Phi_total = np.eye(7)
            for dtss in dt_sub_steps:
                global_leo.dt = dtss 
                Jac = global_leo.get_jacobian_matrix()
                Jac[3:6,6] *= 1000 # km to m
                Phi_k = np.eye(7) + Jac * global_leo.dt
                for i in range(2,global_STM_taylor_order+1):
                    Phi_k += np.linalg.matrix_power(Jac, i) * global_leo.dt**i / math.factorial(i)
                Phi_total = Phi_k @ Phi_total
                xprop = global_leo.step()
            self.cauchyEsts[0].deterministic_transform(Phi_total, np.zeros(7))
            pyduc = self.cauchyEsts[0].get_pyduc()
            xprop[0:6] *= 1000
            pyduc.cset_x(xprop)

            if with_append_prop:
                P_prop = Phi_total @ np.diag(self.cauchyEsts[0]._p0 * 1.3898)**2 @ Phi_total.T
                self.no_msmt_update_moment_append(0, xprop, P_prop)
            elif with_overwrite_last:
                _, P_prop = self.win_moms[0][-1]
                P_prop = Phi_total @ P_prop @ Phi_total.T
                self.no_msmt_update_moment_overwrite(0, xprop, P_prop)
        else:
            for i in range(self.num_windows):
                if self.k >= i: #(self.win_counts[i] > 0):
                    xhat, Phat = self.cauchyEsts[i].get_last_mean_cov()
                    xhat[0:6] /= 1000
                    global_leo.reset_state_with_ellapsed_time(xhat, ellapsed_time)
                    Phi_total = np.eye(7)
                    for dtss in dt_sub_steps:
                        global_leo.dt = dtss 
                        Jac = global_leo.get_jacobian_matrix()
                        Jac[3:6,6] *= 1000 # km to m
                        Phi_k = np.eye(7) + Jac * global_leo.dt
                        for j in range(2, global_STM_taylor_order+1):
                            Phi_k += np.linalg.matrix_power(Jac, j) * global_leo.dt**j / math.factorial(j)
                        Phi_total = Phi_k @ Phi_total
                        xprop = global_leo.step()
                    self.cauchyEsts[i].deterministic_transform( Phi_total, np.zeros(7) )
                    pyduc = self.cauchyEsts[i].get_pyduc()
                    xprop[0:6] *= 1000
                    pyduc.cset_x(xprop)
                    P_prop = Phi_total @ Phat @ Phi_total.T
                    if with_append_prop:
                        self.no_msmt_update_moment_append(i, xprop, P_prop)
                    elif with_overwrite_last:
                        self.no_msmt_update_moment_overwrite(i, xprop, P_prop)
                    else:
                        self.win_moms[i].append((xprop, P_prop))
                        self.partial_step_boolean = True

    def teardown_except_selected_estimators(self, mce_idxs):
        if type(mce_idxs) == int:
            _mce_idxs = [mce_idxs]
        else:
            _mce_idxs = list(mce_idxs)
        global global_leo
        global_leo.clear_model()
        for i in range(self.num_windows):
            if i not in _mce_idxs:
                self.cauchyEsts[i].__del__()
        print("GMAT MCE Torn down")
 
    def teardown(self):
        global global_leo
        global_leo.clear_model()
        for cauchyEst in self.cauchyEsts:
            cauchyEst.__del__()
        print("GMAT MCE Torn down")

def form_short_encounter_contour_plot(s_mce_tca, p_mce_tca, xlow, xhigh, delta_x, ylow, yhigh, delta_y, APPROX_EPS=1e-12):
    # Form conjunction plane using short encounter assumption
    s_tca_xhat = s_mce_tca.tca_xhat
    p_tca_xhat = p_mce_tca.tca_xhat
    rv = s_tca_xhat[3:6] - p_tca_xhat[3:6]
    _, _, Vt = np.linalg.svd( rv.reshape((1,3)) )
    Trel = Vt[1:,:]
    Trel = np.hstack((Trel, np.zeros((2,4)) ))
    _Trel = Trel.copy().reshape(-1)
    r2d = Trel @ (s_tca_xhat - p_tca_xhat)
    
    # Form 2D System and use approximation, extend caching mechanism to threading...it will take a long time
    rsys2d_fz, rsys2d_mean, rsys2d_var, \
    rsys2d_cerr_fz, rsys2d_cerr_mean, rsys2d_cerr_var, \
    cpdf_points, num_gridx, num_gridy = pycauchy.pycauchy_single_step_eval_2d_rsys_cpdf(
        _Trel, s_mce_tca.py_handle, p_mce_tca.py_handle, float(APPROX_EPS),
        float(xlow), float(xhigh), float(delta_x),
        float(ylow), float(yhigh), float(delta_y) )
    cpdf_points = cpdf_points.reshape(num_gridx*num_gridy, 3)
    # Meters to KM
    X = cpdf_points[:,0].reshape( (num_gridy, num_gridx) ) / 1000
    Y = cpdf_points[:,1].reshape( (num_gridy, num_gridx) ) / 1000
    Z = cpdf_points[:,2].reshape( (num_gridy, num_gridx) )
    rsys2d_mean /= 1000
    rsys2d_var /= 1e6
    return X,Y,Z, rsys2d_mean, rsys2d_var.reshape((2,2))