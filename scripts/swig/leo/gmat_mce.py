from distutils.log import debug
from glob import glob
import numpy as np 
import cauchy_estimator as ce 
import math, os, pickle
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg',force=True)
from gmat_sat import *

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

### Testing Cauchy ###
global_leo = None
INITIAL_H = False

#mce_naive_p0 = np.array([1,1,1, 0.001,0.001,0.001, 0.001])
mce_naive_p0 = np.array([.1,.1,.1,.001,.001,.001,0.01])
#mce_naive_p0 = np.array([.01,.01,.01,.01,.01,.01,.01])

def ece_dynamics_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    # Set Phi and Gamma
    x = pyduc.cget_x()
    Jac = global_leo.get_jacobian_matrix()
    Jac[3:6,6] *= 1000 # km to m
    taylor_order = 3
    Phi_k = np.eye(7) + Jac * global_leo.dt
    for i in range(2,taylor_order+1):
        Phi_k += np.linalg.matrix_power(Jac, i) * global_leo.dt**i / math.factorial(i)
    Gamma_k = np.zeros((7,1))
    Gamma_c = np.zeros((7,1)) # continous time Gamma 
    Gamma_c[6,0] = 1.0
    for i in range(taylor_order+1):
        Gamma_k += ( np.linalg.matrix_power(Jac, i) * global_leo.dt**(i+1) / math.factorial(i+1) ) @ Gamma_c

    pyduc.cset_Phi(Phi_k)
    pyduc.cset_Gamma(Gamma_k)
    # Propagate and set x
    xbar = global_leo.step() 
    xbar[0:6] *= 1000
    pyduc.cset_x(xbar)
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
    sig_mce = np.array([np.diag(P)**0.5 for P in Ps_mce]) * sig
    sig_avg_mce = np.array([np.diag(P)**0.5 for P in Ps_avg_mce]) * sig

    es_kf = true_x_kf - xt_kf
    es_mce = true_x_mce - xs_mce
    es_avg_mce = true_x_mce - xs_avg_mce

    kf_msmt_idxs = np.arange(mce_msmt_idxs[0], mce_msmt_idxs[-1]+1, 1)
    N = xs.shape[1]

    plt.figure()
    plt.suptitle(title_prefix + " Estimation Errors Vs {}-Sigma Bounds\nKF (g/m) vs MCE (b/r) vs Weighted Avg MCE (b--/r--)".format(sig))
    ylabels = ['Pos X (km)', 'Pos Y (km)', 'Pos Z (km)', 'Position Z (km)', 'Vel X (km/s)', 'Vel Y (km/s)', 'Vel Z (km/s)', 'Change Dens']
    for i in range(N):
        plt.subplot(N, 1, i+1)
        # Kalman Filter
        plt.plot(kf_msmt_idxs, es_kf[:,i], 'g')
        plt.scatter(kf_msmt_idxs, es_kf[:,i], color='g')
        plt.plot(kf_msmt_idxs, sig_kf[:,i], 'm')
        plt.plot(kf_msmt_idxs, -sig_kf[:,i], 'm')
        # MCE
        plt.plot(mce_msmt_idxs, es_mce[:,i], 'b')
        plt.scatter(mce_msmt_idxs, es_mce[:,i], color='b')
        plt.plot(mce_msmt_idxs, sig_mce[:,i], 'r')
        plt.plot(mce_msmt_idxs, -sig_mce[:,i], 'r')
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
    _A0 = cauchyEsts[best_idx]._Phi.copy().reshape((7,7)).T # np.eye(5)
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
    P_kf[0:6,0:6] * 1000**2
    ratios = np.ones(6)
    for i in range(6):
        pkf = P_kf[i,i] * 1000**2
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

class GmatMCE():

    def __init__(self, num_windows, t0, x0, dt, A0, p0, b0, beta, gamma, Cd_dist="gauss", std_Cd = 0.0013, tau_Cd = 21600, win_reinitialize_func=None, win_reinitialize_params=None, debug_print = True, mce_print = False):
        # Print Out Some Info 
        print("Note that the MCE uses meters for the state and measurements, and not KM")
        Cd_dist = Cd_dist.lower()
        assert Cd_dist in ["gauss", "sas"]
        self.num_windows = num_windows
        self.dt = dt 
        self.win_moms = { i : [] for i in range(num_windows) }
        self.debug_print = debug_print
        self.mce_print = mce_print

        # Setup GMAT Fermi Satellite Object -- internally GMAT uses KM, so need conversions
        gmat.Clear()
        global global_leo
        _x0 = x0.copy() # meters
        _x0[0:6] /= 1000 # meters to km
        global_leo = FermiSatelliteModel(t0, _x0[0:6], dt) 
        global_leo.create_model(with_jacchia=True, with_SRP=True)
        global_leo.set_solve_for(field="Cd", dist=Cd_dist, scale=std_Cd, tau=tau_Cd, alpha=2.0 if Cd_dist == "gauss" else 1.3)
        global_leo.reset_state(_x0, 0)
        ce.set_tr_search_idxs_ordering([5,4,6,3,2,1,0])
        global_leo.gamma0 = gamma.copy() 
        global_leo.beta0 = beta.copy() 
        
        # Setup Windows
        self.cauchyEsts = [ce.PyCauchyEstimator("nonlin", num_windows, mce_print) for _ in range(num_windows)]
        for i in range(num_windows):
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

    # zk is assumed to be GPS coordinates in the ECI frame for this function
    def sim_step(self, zk, x_truth = None, is_inputs_meters = True):
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
                    xwr = self.win_moms[win_idx][-1][0].copy() 
                    xwr[0:6] /= 1000 # convert from meters to km
                    xwr[6] = np.clip(xwr[6], -.85, 10) # make sure that change in atms. density is in bounds and reasonable
                    global_leo.reset_state(xwr, self.k)
                    xw, Pw = self.cauchyEsts[win_idx].step(_zk, None, False)
                    self.win_moms[win_idx].append( (xw, Pw ) )
                    if self.debug_print:
                        print("    x_k|k:   ", xw)
                    if x_truth is not None:
                        print("    e_k|k:   ", _x_truth - xw)
                    self.win_counts[win_idx] += 1
            # Now reinitialize the empty window about the best estimate
            best_idx, usable_wins = self._best_window_est()
            #best_idx, usable_wins = idx_max, np.zeros(self.num_windows, dtype=np.bool) 
            #for _ in range(self.num_windows):
            #    if _ <= self.k:
            #        usable_wins[_] = True
            # Increment step count
            self.k += 1

        if self.debug_print:
            print("Best Window Index For Reinit is: Window {}/{}, which has undergone {}/{} steps".format(best_idx+1, self.num_windows, self.win_counts[best_idx], self.num_windows) )
        
        # Edit all windows which ran to assure that their atms. density are OK and not under or above the limits
        EDIT_STATE_INDEX = 6
        EDIT_STATE_LOW = -0.85
        EDIT_STATE_HIGH = 10
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
            self.win_reinitialize_func(self.cauchyEsts, _zk, best_idx, idx_min, self.k, self.win_reinitialize_params)
            self.win_counts[idx_min] += 1
            xw,Pw = self.cauchyEsts[idx_min].get_last_mean_cov()
            self.win_moms[idx_min].append( (xw, Pw ) )
            if self.debug_print:
                print("Window {}/{} was reinitialized!".format(idx_min+1, self.num_windows))
            # Tear down most full window
            if self.win_counts[idx_max] == self.num_windows:
                self.cauchyEsts[idx_max].reset()
                self.win_counts[idx_max] = 0
    
    def real_step():
        pass
    
    def clear_gmat(self):
        global global_leo
        global_leo.clear_model()
        for cauchyEst in self.cauchyEsts:
            cauchyEst.__del__()
        print("GMAT MCE Torn down")