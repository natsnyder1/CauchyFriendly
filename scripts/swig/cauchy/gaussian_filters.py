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
    #plt.figure()
    #plt.subplot(211)
    #times = np.array([i for i in range(xs.shape[0])]) * dt
    #plt.plot(times, xs[:,0])
    #plt.subplot(212)
    #plt.plot(times, xs[:,1])
    #plt.show()

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

def test_particle_filter_kalman_filter():
    # True Start State of the Time Invariant System
    x0 = np.array([.5, 0])
    # Particle Filter State Set
    x0pf = np.random.uniform(-5.0, 5.0, size = (1000,2))
    # Kalman Filter Start State  
    x0kf = x0 + 2*np.random.randn(2)
    P0kf = 2*np.eye(x0.size)

    # Simple Mass Spring Damper Dynamics w/ Gaussian Process and Msmt Noise
    Phi = np.array([ [0, 1], [-1.5, -.25] ])
    Gam = np.array([[0],[1]])
    H = np.array([1,0])
    V = np.array([.01])
    W = np.array([.02])
    
    # C-Time to D-Time Conversion
    dt = 1.0 / 20.0
    Phi_dt = np.sum([ (np.linalg.matrix_power(Phi,i) * dt**i) / math.factorial(i) for i in range(3)], axis = 0)
    Gam_dt = Phi_dt @ Gam * dt
    SIM_LENGTH = 200
    times = np.arange(0, SIM_LENGTH) * dt
    zero_control = np.zeros((SIM_LENGTH,1))
    
    # Generate True States, True Measurements, Noisy Measurements
    true_states = get_simulated_states_dtime(x0, Phi_dt, Gam_dt, zero_control )
    true_msmts = (true_states @ H).reshape((SIM_LENGTH + 1,1))
    noisy_msmts = true_msmts[1:] + np.random.normal(0, np.sqrt(V), SIM_LENGTH).reshape((SIM_LENGTH,1))
    
    # Run Particle filter simulation
    tic = time.time()
    pf_ests, pf_vars = run_particle_filter(x0pf, Phi_dt, Gam_dt, H, zero_control, noisy_msmts, W, V)
    el_t = time.time() - tic 
    print("PF Time Taken: ", el_t)
    print("PF Time / Step: ", el_t / times.size)

    # Run Kalman filter simulation 
    tic = time.time()
    kf_ests, _ = run_kalman_filter(x0kf, zero_control, noisy_msmts, P0kf, Phi_dt, Gam_dt, H.reshape((1,2)), np.atleast_2d(W), np.atleast_2d(V) )
    el_t = time.time() - tic 
    print("KF Time Taken: ", el_t)
    print("KF Time / Step: ", el_t / times.size)

    # Plot Particle Simulation Results
    plt.figure(1)
    plt.subplot(211)
    plt.title("Particle Filter Means (b) vs Truth (g)")
    plt.plot(times, pf_ests[:,0], 'b')
    plt.plot(times, true_states[1:,0], 'g')
    plt.subplot(212)
    plt.plot(times, pf_ests[:,1], 'b')
    plt.plot(times, true_states[1:,1], 'g')
    # Plot Kalman Simulation Results
    plt.figure(2)
    plt.subplot(211)
    plt.title("Kalman Filter Means (b) vs Truth (g)")
    plt.plot(times, kf_ests[:,0], 'b')
    plt.plot(times, true_states[1:,0], 'g')
    plt.subplot(212)
    plt.plot(times, kf_ests[:,1], 'b')
    plt.plot(times, true_states[1:,1], 'g')
    # Display
    plt.show()


def test_4state_kalman():
    DT = 0.1
    Phi = np.array([1,DT,0,0,0,1,0,0,0,0,1,DT,0,0,0,1]).reshape((4,4))
    Gamma = np.array([0.5*DT*DT, 0, DT, 0, 0, 0.5*DT*DT, 0, DT]).reshape((4,2))
    H = np.array([1,0,0,0, 0,0,1,0.0]).reshape((2,4))
    W = np.array([1.5, 0.1, 0.1, 1.5]).reshape((2,2))
    V = np.array([1.2, 0.2, 0.2, 1.2]).reshape((2,2))

    x = np.array([0,0,0,0])
    u = np.array([1, 2.])
    P = np.array([1, 0, 0,  0, 0,  1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1]).reshape((4,4))

    msmts = np.array([0.154914, 0.215172, 
                    -1.54821, 0.0707649, 
                    0.144694, 0.805929, 
                    0.0107711, 0.780828, 
                    1.28027, 0.628823, 
                    -1.0378, 0.728563, 
                    -0.999518, -0.0609128, 
                    1.92718, 1.16976, 
                    -0.440777, 1.83967, 
                    0.149059, 0.335838, 
                    0.0650556, 2.6097, 
                    1.13946, 2.80649, 
                    0.920578, 2.10558, 
                    0.582518, 1.76105, 
                    -0.0648976, 3.89787, 
                    1.32872, 2.04418, 
                    1.40127, 2.83793, 
                    3.95089, 1.24606, 
                    2.79387, 4.07122, 
                    2.77409, 4.95685]).reshape((20,2))

    us = np.repeat(u.reshape((1,2)), 20, axis = 0).reshape((20,2))
    xs, Ps = run_kalman_filter(x, us, msmts, P, Phi, Gamma, H, W, V, is_debug = False)
    print("Connditional Means:")
    for x in xs:
        print(x)
    print("Connditional Covariances:")
    for P in Ps:
        print(P)

# KF on Moshe's 3-state problem
def test_simple_kalman():
    steps = 7
    Phi = np.array([1.4000, -0.6000, -1.0000, -0.2000, 1.0000, 0.5000, 0.6000, -0.6000, -0.2000]).reshape((3,3))
    Gamma = np.array([0.1000, 0.3000, -0.2000]).reshape((3,1))
    H = np.array([1.0000, 0.5000, 0.2000]).reshape((1,3))
    W = np.array([[(0.1*1.3898)**2]])
    V = np.array([[(0.2*1.3898)**2]])

    x = np.array([-0.4644, 0.2079, 0.1394])
    P = np.array([(0.10 * 1.3898)**2, 0, 0,
         0, (0.08 * 1.3898)**2, 0, 
         0, 0, (0.05 * 1.3898)**2]).reshape((3,3))
    
    us = np.repeat(0, steps).reshape((steps, 1))
    msmts = np.array([-1.0780, -1.3594, 1.8163, 3.0419, 3.2480, 3.3815, 3.3265]).reshape((steps,1))

    xs, Ps = run_kalman_filter(x, us, msmts, P, Phi, Gamma, H, W, V)
    print("Connditional Means:")
    for x in xs:
        print(x)
    print("Connditional Covariances:")
    for P in Ps:
        print(P)

if __name__ == "__main__":
    test_simple_kalman()