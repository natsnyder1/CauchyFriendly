from cmath import log
from tkinter.tix import Tree
from turtle import goto
import numpy as np
import os, sys 
file_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(file_dir + "/../cauchy")
import cauchy_estimator as ce
import gaussian_filters as gf
import math

count = 0

def lookup_air_density(r_sat):
    if(r_sat == 550e3):
        return 2.384e-13
    elif(r_sat == 500e3):
        return 5.125e-13
    elif(r_sat == 450e3):
        return 1.184e-12
    elif(r_sat == 400e3):
        return 2.803e-12
    elif(r_sat == 350e3):
        return 7.014e-12
    elif(r_sat == 300e3):
        return 1.916e-11
    elif(r_sat == 250e3):
        return 6.073e-11
    elif(r_sat == 200e3):
        return 2.541e-10
    elif(r_sat == 150e3):
        return 2.076e-9
    elif(r_sat == 100e3):
        return 5.604e-7
    else:
        print("Lookup air density function does not have value for {}...please add! Exiting!\n", r_sat)
        exit(1)


class leo_satellite_5state():
    # Size of simulation dynamics
    n = 5
    num_satellites = 2 # number of sattelites to talk to (measurements)
    p = num_satellites
    cmcc = 1
    # Orbital distances
    r_earth = 6378.1e3 # spherical approximation of earths radius (meters)
    r_sat = 550e3 # orbit distance of satellite above earths surface (meters)
    
    # Satellite parameter specifics
    M = 5.9722e24 # Mass of earth (kg)
    G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  #Nm^2/kg^2
    m = 5000.0 # kg
    rho = lookup_air_density(r_sat) # kg/m^3
    C_D = 2.0 #drag coefficient
    A = 64.0 #m^2
    tau = 21600.0 # 1/(m*sec)
    # Parameters for runge kutta ODE integrator
    dt = 60 #time step in sec
    sub_steps_per_dt = 60 # so sub intervals are dt / sub_steps_dt 
    # Initial conditions
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), v0/np.sqrt(2), -v0/np.sqrt(2), 0.01])
    omega0 = v0/r0 # rad/sec (angular rate of orbit)
    orbital_period = 2.0*np.pi / omega0 #Period of orbit in seconds
    time_steps_per_period = (int)(orbital_period / dt + 0.50) # number of dt's until 1 revolution is made
    num_revolutions = 10
    num_simulation_steps = num_revolutions * time_steps_per_period
    # Satellite parameters for measurement update
    satellite_positions = np.array([ [-7e6, -7e6], [7e6, 7e6] ])
    dt_R = 0.0 # bias time of sattelite clocks, for now its zero
    b = np.zeros(2)
    std_dev_gps = 2.0 # uncertainty in GPS measurement (meters)
    V = np.array([ [pow(std_dev_gps,2), 0], [0, pow(std_dev_gps,2)] ])
    cholV = np.linalg.cholesky(V)
    # Conversion Parameters 
    SAS_alpha = 1.3
    CAUCHY_TO_GAUSS = 1.3898
    GAUSS_TO_CAUCHY = 1.0 / CAUCHY_TO_GAUSS
    beta_drag = 0.0013
    beta_gauss = (beta_drag * CAUCHY_TO_GAUSS) / (tau * (1.0 - np.exp(-dt/tau)))
    beta_cauchy = beta_gauss * GAUSS_TO_CAUCHY
    # Satellite parameters for process noise
    q = 8e-15; # Process noise ncertainty in the process position and velocity
    W = np.zeros(25)
    W[0] = pow(dt,3)/3*q
    W[6] = pow(dt,3)/3*q
    W[2] = pow(dt,2)/2*q
    W[8] = pow(dt,2)/2*q
    W[10] = pow(dt,2)/2*q
    W[16] = pow(dt,2)/2*q
    W[12] = dt*q
    W[18] = dt*q
    W[24] = pow( beta_drag * CAUCHY_TO_GAUSS, 2)
    W = W.reshape((5,5))
    cholW = np.linalg.cholesky(W)
    Wd = np.array([[beta_gauss**2]])
    # Initial uncertainty in position
    alpha_density_cauchy = 0.0039 # Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
    alpha_density_gauss = alpha_density_cauchy * CAUCHY_TO_GAUSS # Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
    alpha_pv_gauss = 0.001 # Initial Gaussian standard deviation in position and velocity of satellite
    alpha_pv_cauchy = alpha_pv_gauss * GAUSS_TO_CAUCHY # Initial converted uncertainty parameter in position and velocity of satellite converted for Cauchy Estimator
    P0 = np.zeros(25)
    P0[0] = pow(alpha_pv_gauss,2); P0[6] = pow(alpha_pv_gauss,2)
    P0[12] = pow(alpha_pv_gauss,2); P0[18] = pow(alpha_pv_gauss,2)
    P0[24] = pow(alpha_density_gauss,2)
    P0 = P0.reshape((5,5))
    cholP0 = np.linalg.cholesky(P0)
    # For Kalman Schmidt Recursion
    last_kf_est = x0[2:4].copy()
    
leo = leo_satellite_5state()
count = 0
xs_callback = None

def leo5_ode(x):
    global leo 
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    v = np.sqrt(x[2]*x[2] + x[3]*x[3])
    dx_dt = np.zeros(5)
    dx_dt[0] = x[2] 
    dx_dt[1] = x[3]
    dx_dt[2] = -(leo.mu)/pow(r,3) * x[0] - 0.5*leo.A*leo.C_D/leo.m*leo.rho*(1+x[4])*v*x[2]
    dx_dt[3] = -(leo.mu)/pow(r,3) * x[1] - 0.5*leo.A*leo.C_D/leo.m*leo.rho*(1+x[4])*v*x[3]
    dx_dt[4] = -1.0 / leo.tau * x[4]
    return dx_dt

def leo_5state_transition_model(x):
    global leo 
    x_new = x.copy()
    dt_sub = leo.dt / leo.sub_steps_per_dt
    for _ in range(leo.sub_steps_per_dt):
        x_new = ce.runge_kutta4(leo5_ode, x_new, dt_sub)
    return x_new 


def leo_5state_transition_model_jacobians(x):
    Jac = ce.cd4_gvf(x, leo5_ode) # Jacobian matrix
    taylor_order = 6
    Phi_k = np.zeros((x.size,x.size))
    for i in range(taylor_order+1):
        Phi_k += np.linalg.matrix_power(Jac, i) * leo.dt**i / math.factorial(i)
    Gamma_k = np.zeros((x.size,1))
    Gamma_c = np.zeros((x.size,1)) # continous time Gamma 
    Gamma_c[4,0] = 1.0
    for i in range(taylor_order+1):
        Gamma_k += ( np.linalg.matrix_power(Jac, i) * leo.dt**(i+1) / math.factorial(i+1) ) @ Gamma_c
    return Phi_k, Gamma_k

def leo_5state_measurement_model_jacobian(x):
    H = np.zeros((2,5))
    H[0,0] = 1.0
    H[1,1] = 1.0
    return H

def leo_5state_measurement_model(x):
    global leo
    return x[:2].copy()

def ekf_f3(x, u, other_params):
    global count 
    x5 = np.concatenate((xs_callback[count, 0:2], x))
    return leo_5state_transition_model(x5)[2:]

def ekf_h3(xbar, other_params):
    global count 
    count += 1
    return leo_5state_measurement_model(xbar)

def ekf_callback3_Phi_Gam(x, u, other_params):
    global count 
    x5 = np.concatenate((xs_callback[count, 0:2], x))
    Phi, _ = leo_5state_transition_model_jacobians(x5)
    Gamma = np.eye(x.size)
    return Phi[2:,2:], Gamma

def ekf_callback3_H(xbar, other_params):
    return leo_5state_measurement_model_jacobian(xbar)[:,0:3]

def ece_dynamics_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    # Set Phi and Gamma
    #print("Hello from dyn update!")
    step = pyduc.cget_step()
    #print("Step is ", step)
    x = pyduc.cget_x().copy()
    #print("x is ", x)
    x5 = np.hstack((xs_callback[step, 0:2], x.copy()))
    #print("x5 is ", x5)
    Phi, Gamma = leo_5state_transition_model_jacobians(x5)
    pyduc.cset_Phi(Phi[2:,2:].copy())
    pyduc.cset_Gamma(Gamma[2:].copy())
    # Propagate and set x
    xbar = leo_5state_transition_model(x5.copy()) 
    #print("xbar is ", xbar)
    pyduc.cset_x(xbar[2:].copy())
    pyduc.cset_is_xbar_set_for_ece()
    # Set H
    H = leo_5state_measurement_model_jacobian(xbar)
    pyduc.cset_H(H[:,:3])

def ece_nonlinear_msmt_model(c_duc, c_zbar):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    zbar = xbar[0:2].copy()#leo_5state_measurement_model(xbar)
    pyduc.cset_zbar(c_zbar, zbar)

def ece_extended_msmt_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    H = leo_5state_measurement_model_jacobian(xbar)
    pyduc.cset_H(H[:,0:3])

def simulate_leo5_state(sim_steps, with_sas_density = True, with_added_jumps = True):
    global leo 
    xs = [] 
    zs = [] 
    vs = [] 
    ws = [] 
    two_zeros = np.zeros(2)
    five_zeros = np.zeros(5)

    x0_noise =  np.hstack((leo.alpha_pv_gauss * np.random.randn(4), leo.alpha_density_gauss * np.random.randn() ))
    x0_truth = leo.x0 + x0_noise
    v0 = np.random.multivariate_normal(two_zeros, leo.V)
    z0 = leo_5state_measurement_model(x0_truth) + v0
    xs.append(x0_truth)
    zs.append(z0)
    vs.append(v0)
    xk = x0_truth.copy()
    for i in range(sim_steps):
        wk = np.random.multivariate_normal(five_zeros, leo.W)
        if with_sas_density:
            wk[4] = ce.random_symmetric_alpha_stable(leo.SAS_alpha, leo.beta_drag, 0)
        if with_added_jumps:
            if i == 20:
                wk[4] = 3.5
            if i == 60:
                wk[4] = -1.0
            if i == 100:
                wk[4] = -1.75
        xk = leo_5state_transition_model(xk) + wk
        vk = np.random.multivariate_normal(two_zeros, leo.V)
        zk = leo_5state_measurement_model(xk) + vk
        xs.append(xk.copy())
        zs.append(zk.copy())
        ws.append(wk[4])
        vs.append(vk)
    xs = np.array(xs)
    zs = np.array(zs)
    ws = np.array(ws).reshape((sim_steps, 1))
    vs = np.array(vs).reshape((sim_steps+1, 2))
    return (xs, zs, ws, vs)


def test_leo3():
    global xs_callback, leo
    np.random.seed(18)
    noise_std = 0.001  # velocity noise level
    num_windows = 8
    log_dir = file_dir + "/pylog/leo3/w"+str(num_windows)+"/" + str(int(leo.r_sat/1000 + 0.99)) + "km_noise" + str(noise_std)
    sim_steps = 55
    xs, zs, ws, vs = simulate_leo5_state(sim_steps, with_sas_density = True, with_added_jumps = True)
    xs_callback = xs[:,0:2]
    vels_true = xs[:,2:4]
    vels_noise = noise_std * np.random.randn(*vels_true.shape)
    vels_msmt = vels_true + vels_noise
    ce.log_sim_truth(log_dir, xs[:,2:], vels_true, ws, vs)
    I3 = np.eye(3)

    with_kf = True 
    kf_atm_boost_val = 100000 #100000
    with_ce = True

    # Experimentally using Kalman estimates for cauchy
    with_cascade = False
    if with_cascade:
        assert(with_kf)
        assert(not with_ce)

    if with_kf:
        # Run EKF3
        x0_kf = xs[0][2:].copy() + np.random.randn(3) * np.array([leo.alpha_pv_gauss, leo.alpha_pv_gauss, leo.alpha_density_gauss])
        P0_kf = leo.P0.copy()[2:,2:]
        xs_kf = [x0_kf.copy()] 
        Ps_kf = [P0_kf.copy()]
        x_kf = x0_kf.copy()
        P_kf = P0_kf.copy()
        W_kf = leo.W[2:,2:]
        W_kf[2,2] *= kf_atm_boost_val
        V_kf = np.diag([noise_std,noise_std])**2
        for i in range(1, sim_steps+1):
            zk = vels_msmt[i]
            x_kf = ekf_f3(x_kf, None, None)
            Phi, Gam = ekf_callback3_Phi_Gam(x_kf, None, None)
            P_kf = Phi @ P_kf @ Phi.T + Gam @ W_kf @ Gam.T
            zbar = ekf_h3(x_kf, None)
            H_kf = ekf_callback3_H(x_kf, None)
            K_kf = P_kf @ H_kf.T @ np.linalg.inv(H_kf @ P_kf @ H_kf.T + V_kf)
            r_kf = zk - zbar
            x_kf = x_kf + K_kf @ r_kf
            P_kf = (I3 - K_kf @ H_kf) @ P_kf @ (I3 - K_kf @ H_kf).T + K_kf @ V_kf @ K_kf.T
            xs_kf.append(x_kf.copy())
            Ps_kf.append(P_kf.copy())
        xs_kf = np.array(xs_kf)
        Ps_kf = np.array(Ps_kf)
        if kf_atm_boost_val == 1:
            ce.log_kalman(log_dir, xs_kf, Ps_kf)
            ce.log_kalman(log_dir+"/kf_untuned", xs_kf, Ps_kf)
        else:
            boost_dir = "/kf_tuned_" + str(kf_atm_boost_val)
            ce.log_kalman(log_dir+boost_dir, xs_kf, Ps_kf)

    if with_ce:
        #P_ce0 = 50*Ps_kf[25]
        #x_ce0 = xs_kf[1]
        #x_bar0 = leo_5state_transition_model(leo.x0)[2:]
        #z_bar0 = leo_5state_measurement_model(x_bar0)
        #dz0 = (vels_msmt[1] - z_bar0)[0]
        #dx0 = x_ce0 - x_bar0
        #H = np.array([1.0, 0 ,0])
        #A0, p0, b0 = ce.speyers_window_init(dx0, P_ce0, H, noise_std * leo.GAUSS_TO_CAUCHY, dz0)

        x_bar0 = leo_5state_transition_model(leo.x0)[2:]
        A0 = leo_5state_transition_model_jacobians(xs[1])[0][2:,2:].T 
        p0 = np.array([noise_std, noise_std, leo.alpha_density_cauchy])
        b0 = np.zeros(3)
        beta = np.array([leo.beta_cauchy])
        gamma = 3 * np.array([noise_std * leo.GAUSS_TO_CAUCHY, noise_std * leo.GAUSS_TO_CAUCHY])
        ce.set_tr_search_idxs_ordering([2,1,0])
        #cauchy_steps = 8
        #cauchyEst = ce.PyCauchyEstimator("nonlin", cauchy_steps, True)
        #cauchyEst.initialize_nonlin(x0, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, 0)
        
        #win_var_boost = np.array([3*noise_std,3*noise_std,0])
        #win_var_boost = np.array([noise_std**2,noise_std**2,0])
        win_var_boost = None
        cauchyEst = ce.PySlidingWindowManager("nonlin", num_windows, sim_steps, log_dir, log_full=True)
        cauchyEst.initialize_nonlin(x_bar0, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, 0, 0, 0, win_var_boost)
        for i in range(1,sim_steps+1):
            vk = vels_msmt[i]
            cauchyEst.step(vk, None)
        cauchyEst.shutdown()

    if with_cascade:
        x0 = xs[1][2:].copy()
        A0 = leo_5state_transition_model_jacobians(xs[0])[0][2:,2:].T #np.array([1,.1,.1,.1,1,.1,.1,.1,1]).reshape((3,3)) #leo_5state_transition_model_jacobians(xs[1])[0][2:,2:].T
        p0 = 3*np.array([noise_std, noise_std, leo.alpha_density_cauchy])
        b0 = np.zeros(3)
        beta = np.array([leo.beta_cauchy])
        ce_noise_std1 = .0005 * leo.GAUSS_TO_CAUCHY #noise_std * leo.GAUSS_TO_CAUCHY # Ps_kf[-1][0,0]**0.5 * leo.GAUSS_TO_CAUCHY
        ce_noise_std2 = .0005 * leo.GAUSS_TO_CAUCHY #noise_std * leo.GAUSS_TO_CAUCHY # Ps_kf[-1][1,1]**0.5 * leo.GAUSS_TO_CAUCHY
        gamma = np.array([ce_noise_std1, ce_noise_std2])
        ce.set_tr_search_idxs_ordering([2,1,0])
        #cauchy_steps = 8
        #cauchyEst = ce.PyCauchyEstimator("nonlin", cauchy_steps, True)
        #cauchyEst.initialize_nonlin(x0, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, 0)
        
        win_var_boost = None # 2*np.array([ce_noise_std1, 0, 0])
        log_dir = None
        cauchyEst = ce.PySlidingWindowManager("nonlin", num_windows, sim_steps, log_dir)
        cauchyEst.initialize_nonlin(x0, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, 0, 0, 0, win_var_boost)
        for i in range(1,sim_steps+1):
            vk = xs_kf[i][0:2]
            cauchyEst.step(vk, None)
        cauchyEst.shutdown()


    if with_kf:
        kf_tuple = (xs_kf[1:], Ps_kf[1:])
    else:
        kf_tuple = None
    if with_ce or with_cascade:
        ce_tuple = cauchyEst.moment_info 
    else:
        ce_tuple = None
    ce.plot_simulation_history(ce_tuple, (xs[1:,2:], zs[1:], ws[1:], vs[1:]), kf_tuple)
    foo = 2
    




if __name__ == '__main__':
    test_leo3()