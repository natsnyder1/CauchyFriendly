#!/usr/bin/env python3

import numpy as np 
import math 
import matplotlib.pyplot as plt
import time

def prop_state_ctime(state, Phi, Gam, control, dt = .025):
    dx_dt = Phi @ state + Gam @ control
    return dx_dt * dt + state

def prop_state_dtime(state, Phi, Gam, control):
    return Phi @ state + Gam @ control

def prop_states_ctime(state, Phi, Gam, control, dt = .025):
    dx_dt = state @ Phi.T + control @ Gam.T 
    return dx_dt * dt + state

def prop_states_dtime(state, Phi, Gam, control):
    return state @ Phi.T + control @ Gam.T

def get_simulated_states_dtime(x0, Phi, Gam, controls):
    xs = []
    xs.append(x0)
    x = x0
    for u in controls:
        x = prop_state_dtime(x, Phi, Gam, u)
        xs.append(x.reshape(-1))
    xs = np.array(xs)
    #plt.figure()
    #plt.subplot(211)
    #plt.title("True States Pos/Vel")
    #plt.plot([i for i in range(xs.shape[0])], xs[:,0])
    #plt.subplot(212)
    #plt.plot([i for i in range(xs.shape[0])], xs[:,1])
    #plt.show()
    return xs 

def get_simulated_states_ctime(x0, Phi, Gam, controls, dt = .025):
    xs = []
    xs.append(x0)
    x = x0
    for u in controls:
        x = prop_state_ctime(x, Phi, Gam, u, dt)
        xs.append(x.reshape(-1))
    return np.array(xs)

# Get proposal particles
def proposal_particle_sample(states, W):
    if W.shape[0] > 1:
        new_states = np.array([np.random.multivariate_normal(s.reshape(-1), np.linalg.cholesky(W) ) for s in states])
    else:
        new_states = np.array([np.random.normal(s.reshape(-1), np.sqrt(W) ) for s in states])
    return new_states

# Use the D-Time Dynamics
# 1) propogate the states and sample around each of the prop states using W
## return propogated particles
def get_pf_prop(states, Phi, Gam, control, W):
    prop_states = prop_states_dtime(states,Phi,Gam, control)
    # If process noise is small / large I wonder how much of a difference sampling here makes
    sampled_prop_states = proposal_particle_sample(prop_states, W)
    return sampled_prop_states

# 2) Using V, find the probability of each state, normalize weights so the sum in 1
## return normalized weights
def get_pf_importance_sample_weights(states, msmt, H, V):
    n = states[0].size
    resid = states @ H - msmt.T
    if resid.ndim == 1:
        exponent = -.50*resid **2 / V + 1e-5
    else:
        resid = np.atleast_3d(resid)
        exponent = -.50* np.moveaxis(resid,1,2) @ np.linalg.inv(V + 1e-5) @ resid
    weights = 1.0 / ( (2.0 * np.pi)**(n / 2.0) ) * np.exp(exponent)
    wn = np.linalg.norm(weights)
    weights /= wn #np.sum(weights)
    weights /= np.sum(weights)
    assert np.abs(np.sum(weights) - 1.0) < 1e-5
    return weights

# 3) Using the importance weights, resample new particle set
def resample_particles(states, weights):
    s = np.arange(states.shape[0])
    return np.random.choice(s,s.size,p=weights)

# Runs a full step of the particle filter
def particle_filter(states, Phi, Gam, H, control, msmt, W, V):
    prop_states = get_pf_prop(states, Phi, Gam, control, W)
    weights = get_pf_importance_sample_weights(prop_states, msmt, H, V)
    # Either conditional mean is computed using weights
    conditional_mean = np.sum( prop_states * weights.reshape((weights.size,1)), axis = 0 )
    ehat = np.atleast_3d(prop_states - conditional_mean)
    conditional_variance = np.sum(np.matmul(ehat,np.moveaxis(ehat,1,2) ), axis = 0)
    resampled_states = prop_states[resample_particles(prop_states, weights)]
    # or mean is computed as an average over the new states found ( i dont think so tho -- biased?)
    #conditional_mean = np.mean(resampled_states, axis = 0)
    print("Mean is: ", conditional_mean, "Var is: ", conditional_variance)
    return resampled_states, conditional_mean, conditional_variance

# Runs a simulation of a linear time invariant particle filter
def run_particle_filter(init_states, Phi, Gam, H, controls, msmts, W, V):
    states = init_states
    #state_hist = [states]
    state_means = []
    state_vars = []
    count = 0
    for u, z in zip(controls,msmts):
        states, conditional_mean, conditional_variance = particle_filter(states, Phi, Gam, H, u, z, W, V)
        #state_hist.append(states)
        state_means.append(conditional_mean)
        state_vars.append(conditional_variance)
        print("Finished Step ", count + 1, " / ", controls.shape[0])
        count += 1
    return np.array(state_means), np.array(state_vars)

# Runs a full step of the kalman filter
def kalman_filter(x, u, msmt, P, Phi, B, Gam, H, W, V):
    # Propogate Dynamics
    xbar = Phi @ x + B @ u 
    # A Priori Covariance Matrix
    M = Phi @ P @ Phi.T + Gam @ np.atleast_2d(W) @ Gam.T 
    # Update Kalman Gain 
    K = M @ H.T @ np.linalg.inv( H @ M @ H.T + V )
    # Find the conditional mean estimate 
    r = msmt - H @ xbar
    xhat = xbar + K @ r
    # Posteriori Covariance Matrix
    I = np.eye(x.size)
    P = (I - K @ H ) @ M @ (I - K @ H).T + K @ V @ K.T
    return xhat, P, r, K

# Runs a simulation of a linear time (in)variant kalman filter
def run_kalman_filter(x0, us, msmts, P0, Phi, B, Gam, H, W, V, dynamics_update_callback = None, other_params = None, is_debug = False):
    xs = [x0.copy()]
    Ps = [P0.copy()]
    rs = []
    Ks = []
    P = P0 
    x = x0
    if us is None:
        assert( B is None )
        us = np.zeros((msmts.shape[0], 1))
        B = np.zeros((x0.shape[0],1))
    else:
        assert(us.shape[0] == msmts.shape[0])
        assert( B is not None )
        B = B.reshape((x0.size, us.shape[1]))
    W = np.atleast_2d(W)
    V = np.atleast_2d(V)
    Gam = Gam.reshape((x0.size, W.shape[0]))
    H = H.reshape((V.shape[0], x0.size))

    for u, z in zip(us, msmts):
        if dynamics_update_callback is not None:
            dynamics_update_callback(x, u, Phi, B, Gam, H, W, V, other_params)
        x, P, r, K = kalman_filter(x, u, z, P, Phi, B, Gam, H, W, V)
        xs.append(x.reshape(-1))
        Ps.append(P)
        rs.append(r.reshape(-1))
        Ks.append(K)
    if not is_debug:
        return np.array(xs), np.array(Ps)
    else:
        return np.array(xs), np.array(Ps), np.array(rs), np.array(Ks)


# Runs a full step of the extended kalman filter: updates from k-1|k-1 to k|k
# x is the state vector (of step k-1)
# u is the control vector (of step k-1)
# msmt is the newest measurement (of step k)
# f is the discrete time non-linear system dynamics (maps k-1|k-1 -> k|k-1)
# h is the (possibly) non-linear measurement matrix (used in updating k|k-1 -> k|k)
# callback_Phi_Gam is a callback function which returns the state transition matrix and noise gain matrix, with arguments x (of step k-1) and control (of step k-1)
# callback_H = forms the H matrix -> FUNCTION with argument xbar (of step k|k-1)
# P is the posterior covariance of k-1|k-1, to be updated now to k|k
# W is the process noise matrix describing the current step k-1|k-1 -> k|k-1
# V is the measurement noise matrix describing the step k|k-1 -> k|k
def extended_kalman_filter(x, u, msmt, f, h, callback_Phi_Gam, callback_H, P, W, V, other_params = None):
    assert W.ndim == 2
    assert V.ndim == 2
    Phi, Gam = callback_Phi_Gam(x, u, other_params)
    # Propogate Dynamics
    xbar = f(x, u, other_params) #k|k-1
    
    # A Priori Covariance Matrix
    M = Phi @ P @ Phi.T + Gam @ W @ Gam.T 

    H = callback_H(xbar, other_params)
    # Update Kalman Gain 
    K = M @ H.T @ np.linalg.inv( H @ M @ H.T + V )
    
    # Find the conditional mean estimate 
    r = msmt - h(xbar, other_params)
    xhat = xbar + K @ r
    
    # Posteriori Covariance Matrix
    I = np.eye(x.size)
    P = (I - K @ H ) @ M @ (I - K @ H).T + K @ V @ K.T
    return xhat, P

# Runs a simulation of a linear time invariant kalman filter
# P0 is initial covariance of system
def run_extended_kalman_filter(x0, us, msmts, f, h, callback_Phi_Gam, callback_H, P0, W, V, other_params = None):
    xs = [x0.copy()]
    Ps = [P0.copy()]
    P = P0 
    x = x0
    if us is None:
        us = np.zeros((msmts.shape[0], 1))
    else:
        assert(us.shape[0] == msmts.shape[0])

    for u, z in zip(us, msmts):
        x, P = extended_kalman_filter(x, u, z, f, h, callback_Phi_Gam, callback_H, P, W, V, other_params)
        xs.append(x)
        Ps.append(P)
    return np.array(xs), np.array(Ps)

# This function smooths the results of the extended kalman filter by applying a backward recursive smoother
# xs_kf is a T x n array of state estimates {x1, x2, ..., xT}
# Ps_kf is a T x n x n array of posteriori state covariance estimates {P1, P2, ..., PT}
# Ms_kf is a T x n x n array of apriori state covariance estimates {M1, M2, ..., MT}
# Phis_kf is a (T-1) x n x n of state transition matrices {Phi1, Phi2, ...Phi_{T-1} }
# nonlin_transition_model is the nonlinear propagation model x_k+1 = f(x_k)
# Returns on output: xs_smoothed (T x n array), Ps_smoothed (T x n x n array) of smoothes estimates
# NOTE: This function can easily be changed to accomadate controls or linear dynamics...
def ekf_smoother(xs_kf, Ps_kf, Ms_kf, Phis_kf, nonlin_transition_model):
    assert xs_kf.shape[0] == Ps_kf.shape[0] == Ms_kf.shape[0] == (Phis_kf.shape[0]+1)
    assert xs_kf.shape[1] == Ps_kf.shape[1] == Ms_kf.shape[1] == Phis_kf.shape[1]
    assert nonlin_transition_model is not None

    T = len(xs_kf)
    x_smoothed = xs_kf[-1].copy()
    P_smoothed = Ps_kf[-1].copy()
    xs_smoothed = [x_smoothed.copy()]
    Ps_smoothed = [P_smoothed.copy()]
    for i in reversed(range(T-1)):
        Phi = Phis_kf[i]
        x_kf = xs_kf[i]
        P_kf = Ps_kf[i]
        M_kf = Ms_kf[i+1]

        C = P_kf @ Phi.T @ np.linalg.inv(M_kf)
        x_smoothed = x_kf + C @ (x_smoothed - nonlin_transition_model(x_kf) )
        P_smoothed = P_kf + C @ ( P_smoothed - M_kf ) @ C.T
        print("Smoothed Diff is: ", x_smoothed - x_kf)
        xs_smoothed.append(x_smoothed.copy())
        Ps_smoothed.append(P_smoothed.copy())
    xs_smoothed.reverse()
    Ps_smoothed.reverse()
    xs_smoothed = np.array(xs_smoothed)
    Ps_smoothed = np.array(Ps_smoothed)
    return xs_smoothed, Ps_smoothed