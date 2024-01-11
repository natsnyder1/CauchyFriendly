import numpy as np
import os, sys 
import matplotlib.pyplot as plt
file_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(file_dir + "/../cauchy")
import cauchy_estimator as ce
import gaussian_filters as gf
import math

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
    r_sat = 200e3 # orbit distance of satellite above earths surface (meters)
    
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
    sub_steps_per_dt = 90 # so sub intervals are dt / sub_steps_dt 
    # Initial conditions
    r0 = r_earth + r_sat # orbit distance from center of earth
    v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
    x0 = np.array([r0/np.sqrt(2), r0/np.sqrt(2), v0/np.sqrt(2), -v0/np.sqrt(2), 0.0])
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
    alpha_pv_gauss = 0.01 # Initial Gaussian standard deviation in position and velocity of satellite
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

#psuedo range
def _leo_5state_measurement_model_jacobian(x):
    global leo
    m = len(leo.satellite_positions)
    n = x.size
    H = np.zeros((m,n))
    for i in range(m):
        sp = leo.satellite_positions[i]
        dr = np.linalg.norm(x[0:2] - sp)
        H[i,0] = (x[0] - sp[0]) / dr
        H[i,1] = (x[1] - sp[1]) / dr
    return H 

# pseudo range
def _leo_5state_measurement_model(x):
    global leo
    n = x.size
    pseudo_ranges = np.zeros(len(leo.satellite_positions))
    for i in range(len(leo.satellite_positions)):
        pseudo_ranges[i] = np.linalg.norm(x[0:n//2] - leo.satellite_positions[i])
    return pseudo_ranges

# 'gps'
def leo_5state_measurement_model_jacobian(x):
    
    H = np.zeros((2,5))
    H[0,0] = 1.0
    H[1,1] = 1.0
    return H

# 'gps'
def leo_5state_measurement_model(x):
    global leo
    return x[:2].copy()

def ekf_f(x, u, other_params):
    return leo_5state_transition_model(x)

def ekf_h(xbar, other_params):
    return leo_5state_measurement_model(xbar)

def ekf_callback_Phi_Gam(x, u, other_params):
    Phi, _ = leo_5state_transition_model_jacobians(x)
    Gamma = np.eye(x.size)
    return Phi, Gamma 

def ekf_callback_H(xbar, other_params):
    return leo_5state_measurement_model_jacobian(xbar)

def ece_dynamics_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    # Set Phi and Gamma
    x = pyduc.cget_x()
    Phi, Gamma = leo_5state_transition_model_jacobians(x)
    pyduc.cset_Phi(Phi)
    pyduc.cset_Gamma(Gamma)
    # Propagate and set x
    xbar = leo_5state_transition_model(x) 
    pyduc.cset_x(xbar)
    pyduc.cset_is_xbar_set_for_ece()
    # Set H
    H = leo_5state_measurement_model_jacobian(xbar)
    pyduc.cset_H(H)

def ece_nonlinear_msmt_model(c_duc, c_zbar):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    zbar = leo_5state_measurement_model(xbar)
    pyduc.cset_zbar(c_zbar, zbar)

def ece_extended_msmt_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    H = leo_5state_measurement_model_jacobian(xbar)
    pyduc.cset_H(H)

H_scale = 1e-8
def ltv_scalar_ce_update(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    global leo 
    v = leo.last_kf_est[2:4]
    norm_v = np.linalg.norm(v)
    H = np.array([0.5 * leo.C_D * leo.A / leo.m * leo.rho * norm_v]) / H_scale
    pyduc.cset_H(H)
    

def simulate_leo5_state(sim_steps, with_sas_density = True, with_added_jumps = True, return_full_proc_noise = False):
    global leo 
    xs = [] 
    zs = [] 
    vs = [] 
    ws = [] 
    two_zeros = np.zeros(2)
    five_zeros = np.zeros(5)

    x0_noise =  np.hstack((leo.alpha_pv_gauss * np.random.randn(4), leo.alpha_density_gauss * np.random.randn() ))
    x0_truth = leo.x0.copy()# + x0_noise
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
                wk[4] = 7.5
            if i == 60:
                wk[4] = -2.0
            if i == 100:
                wk[4] = -2.75
        xk = leo_5state_transition_model(xk) + wk
        vk = np.random.multivariate_normal(two_zeros, leo.V)
        zk = leo_5state_measurement_model(xk) + vk
        xs.append(xk.copy())
        zs.append(zk.copy())
        if return_full_proc_noise:
            ws.append(wk.copy()) #[4])
        else:
            ws.append(wk[4])
        vs.append(vk)
    xs = np.array(xs)
    zs = np.array(zs)
    if return_full_proc_noise:
        ws = np.array(ws) #.reshape((sim_steps, 1))
    else:
        ws = np.array(ws).reshape((sim_steps, 1))
    vs = np.array(vs).reshape((sim_steps+1, 2))
    return (xs, zs, ws, vs)

def test_leo5():
    global leo
    # 2124125479 -- no huge jumps
    seed = 2124125479 #int(np.random.rand() * (2**32 -1)) #3872826552#
    print("Seeding with seed: ", seed)
    np.random.seed(seed)
    '''
    num_steps = 4
    total_steps = num_steps
    print_debug = True
    cauchyEst = ce.PyCauchyEstimator("nonlin", total_steps, print_debug)
    cauchyEst.initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
    zs = np.array([ [16467650.909971617, 3370417.0852888753], 
                    [16437796.96138626, 3513168.4008487226], 
                    [16388080.65038866, 3738274.4835428493], 
                    [16318568.603603812, 4030929.8436910328] ])
    '''

    # Get Ground Truth and Measurements
    #zs = np.genfromtxt(file_dir + "/../../../log/leo5/dense/w5/msmts.txt", delimiter= ' ')
    #zs = zs[1:,:]
    prop_steps = 150
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=True, with_added_jumps=True)
    zs_without_z0 = zs[1:,:]

    # Run EKF 
    W_kf = leo.W.copy()
    V_kf = leo.V.copy()
    W_kf[4,4] *= 1000
    xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    
    ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf))

    # Run Cauchy Estimator
    #'''
    beta = np.array([leo.beta_cauchy])
    gamma = np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])
    #beta_scale = 50
    #beta = np.array([leo.beta_cauchy / beta_scale])
    #gamma_scale = 5
    #gamma = gamma_scale*np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])
    
    # Create Phi.T as A0, start at propagated x0
    # Initialize Initial Hyperplanes
    Phi, _ = leo_5state_transition_model_jacobians(leo.x0)
    xbar = leo_5state_transition_model(leo.x0)
    A0 = Phi.T.copy()
    p0 = np.repeat(leo.alpha_pv_cauchy, 5)
    p0[4] = leo.alpha_density_cauchy
    b0 = np.zeros(5)
    num_controls = 0

    num_windows = 6
    total_steps = prop_steps
    ce.set_tr_search_idxs_ordering([3,2,4,1,0])
    log_dir = file_dir + "/pylog/w"+str(6) + "_" + str(int(leo.r_sat/1000)) + "km"
    #log_dir = file_dir + "/pylog/w"+str(6) + "_" + str(int(leo.r_sat/1000)) + "km" + "_bsd" + str(int(beta_scale)) + "_gsu" + str(int(beta_scale))
    cauchyEst = ce.PySlidingWindowManager("nonlin", num_windows, total_steps, log_dir=log_dir, log_seq=True, log_full=True)
    cauchyEst.initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
    
    for zk in zs_without_z0:
        cauchyEst.step(zk, None)
    cauchyEst.shutdown()
    #'''

    ce.plot_simulation_history(cauchyEst.moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True)
    foobar = 2
    
def test_kalman_schmidt_recursion_cascade():
    global leo
    # 2124125479 -- no huge jumps
    # 178211974 -- one mild jump
    seed = int(np.random.rand() * (2**32 -1)) #3872826552#
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Get Ground Truth and Measurements
    #zs = np.genfromtxt(file_dir + "/../../../log/leo5/dense/w5/msmts.txt", delimiter= ' ')
    #zs = zs[1:,:]
    prop_steps = 500
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=True, with_added_jumps=False)
    zs_without_z0 = zs[1:,:]

    # Run EKF 
    W_kf = leo.W.copy() #* 100#000
    V_kf = leo.V.copy()
    #W_kf[4,4] *= 100000#000
    xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    '''
    W_kf = leo.W.copy() #* 100000
    V_kf = leo.V.copy()
    W_kf[4,4] *= 100#000#000#000
    xs_kf2, Ps_kf2 = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    foo = np.zeros(xs.shape[0])
    moment_info = {"x": xs_kf2, "P": Ps_kf2, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf))
    foobar = 2
    '''
    #'''
    # Set-up Scalar Cauchy Estimator
    Phi = np.array([[np.exp(-leo.dt/leo.tau)]])
    Gamma = np.array([1])
    H = np.array([0.5 * leo.C_D * leo.A / leo.m * leo.rho * np.linalg.norm(leo.x0[2:4])]) / H_scale 
    beta = np.array([10*leo.beta_drag])
    gamma = np.array([10*leo.beta_drag]) # Shouldnt this be set to the largest sing. val of the KF posterior covariance?
    A0 = np.array([[1.0]])
    p0 = np.array([leo.alpha_density_cauchy]) #* 1000
    b0 = np.array([0.0])
    cauchyEst = ce.PyCauchyEstimator("ltv", prop_steps, True)
    cauchyEst.initialize_ltv(A0, p0, b0, Phi, None, Gamma, beta, H, gamma, ltv_scalar_ce_update)

    # Now do Kalman Schmidt Recursion 
    x_ksr = leo.x0.copy() + np.array([0,0,0,0,0.001])
    P_ksr = leo.P0.copy()
    W_ksr = leo.W.copy()
    V_ksr = leo.V.copy()
    I = np.eye(5)
    xs_ksr = [x_ksr.copy()]
    Ps_ksr = [P_ksr.copy()]
    for i in range(1, zs.shape[0]):
        zk = zs[i]
        # EKF performs time update
        Phi_ksr, Gamma_ksr = ekf_callback_Phi_Gam(x_ksr, None, None)
        x_ksr =  ekf_f(x_ksr, None, None)
        M_ksr = Phi_ksr @ P_ksr @ Phi_ksr.T + Gamma_ksr @ (W_ksr) @ Gamma_ksr.T 

        # EKF performs measurement update 
        H_ksr = ekf_callback_H(x_ksr, None)
        zbar_ksr = ekf_h(x_ksr, None)
        K_ksr = M_ksr @ H_ksr.T @ np.linalg.inv( H_ksr @ M_ksr @ H_ksr.T + V_ksr)
        r_k = zk - zbar_ksr
        #x_ksr[0:4] = x_ksr[0:4] + K_ksr[0:4] @ r_k
        x_ksr = x_ksr + K_ksr @ r_k

        # Cauchy runs time prop and measurement update 
        leo.last_kf_est =  x_ksr.copy() #xs[i].copy() # make velocity available to cauchy in its callback to update H
        v_norm = np.linalg.norm(leo.last_kf_est[2:4])
        z_ksr_psuedo = 0.5 * leo.C_D * leo.A / leo.m * leo.rho * v_norm * leo.last_kf_est[4] / H_scale #x_ksr[4] / H_scale #xs[i,4] / H_scale
        cauchyEst.step(z_ksr_psuedo, None)
        
        # Cauchy rectifies EKF density: (this needs some work)
        ce_cond_mean = cauchyEst.moment_info["x"][-1]
        ce_cond_var = cauchyEst.moment_info["P"][-1]
        print("Norm Vel:", v_norm, "True Drag is: ", xs[i, 4], "Estimated Drag is: ", ce_cond_mean.item())

        x_ksr[4] = ce_cond_mean
        Pss = M_ksr[:4, :4]
        Psc = M_ksr[:4, 4]
        Pcs = M_ksr[4, :4]
        Ks = K_ksr[:4,:]
        Kc =  K_ksr[4, :].reshape((1,2))
        Hs = H_ksr[:,:4]
        Is = I[:4,:4]

        P_ksr[:4,:4] = (Is - Ks @ Hs) @ Pss #@ (Is - Ks @ Hs).T + Ks @ V_ksr @ Ks.T
        P_ksr[:4,4] = (Is - Ks @ Hs) @ Psc # @ (Is - Ks @ Hs).T + Kc @ V_ksr @ Ks.T
        P_ksr[4,:4] = P_ksr[:4,4].T
        P_ksr[4,4] = ce_cond_var

        #HPHTV_inv = np.linalg.inv( H_ksr @ M_ksr @ H_ksr.T + V_ksr)
        #P_ksr[:4,:4] = Pss - Ks @ H_ksr @ M_ksr[:,:4] - M_ksr[:,:4].T @ H_ksr.T @ Ks.T + Ks @ HPHTV_inv @ Ks.T
        #P_ksr[4,:4] = Pcs - Kc @ H_ksr @ M_ksr[:,:4] - M_ksr[:,4].T @ H_ksr.T @ Ks.T + Kc @ HPHTV_inv @ Ks.T
        #P_ksr[:4,4] = P_ksr[4,:4].T
        #P_ksr[4,4] = ce_cond_var


        min_eig = np.min(np.linalg.eig(P_ksr)[0])
        if min_eig < 0:
            P_ksr += np.eye(5)*1e-3
            print("KF going semidef!")
        # Save results
        xs_ksr.append(x_ksr.copy())
        Ps_ksr.append(P_ksr.copy())

    #xs_ksr = np.array(xs_ksr)
    #Ps_ksr = np.array(Ps_ksr)
    foo = np.zeros(xs.shape[0])
    moment_info = {"x": xs_ksr, "P": Ps_ksr, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf))
    foobar=2
    #'''
def test_kalman_schmidt_recursion_russell():
    global leo
    # 2124125479 -- no huge jumps
    # 178211974 -- one mild jump
    seed = 2124125479 #int(np.random.rand() * (2**32 -1)) #3872826552#
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Get Ground Truth and Measurements
    prop_steps = 500
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=True, with_added_jumps=True)
    zs_without_z0 = zs[1:,:]

    # Run EKF 
    W_kf = leo.W.copy() #* 100#000
    V_kf = leo.V.copy()
    #W_kf[4,4] *= 100000#000
    xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf, W_kf)

    '''
    W_kf = leo.W.copy() #* 100000
    V_kf = leo.V.copy()
    W_kf[4,4] *= 100#000#000#000
    xs_kf2, Ps_kf2 = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    foo = np.zeros(xs.shape[0])
    moment_info = {"x": xs_kf2, "P": Ps_kf2, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf))
    foobar = 2
    '''

    '''
    # Set-up Scalar Cauchy Estimator
    Phi = np.array([[np.exp(-leo.dt/leo.tau)]])
    Gamma = np.array([1])
    H = np.array([1]) 
    beta = np.array([leo.beta_drag])
    gamma = np.array([leo.beta_drag]) # Shouldnt this be set to the largest sing. val of the KF posterior covariance?
    A0 = np.array([[1.0]])
    p0 = np.array([leo.alpha_density_cauchy]) #* 1000
    b0 = np.array([0.0])
    cauchyEst = ce.PyCauchyEstimator("lti", prop_steps, True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, None, Gamma, beta, H, gamma)

    # Now do Kalman Schmidt Recursion 
    x_ksr = leo.x0.copy() + np.array([0,0,0,0,0.001])
    P_ksr = leo.P0.copy()
    W_ksr = leo.W.copy()
    V_ksr = leo.V.copy()
    I_ksr = np.eye(5)
    xs_ksr = [x_ksr.copy()]
    Ps_ksr = [P_ksr.copy()]
    for i in range(1, zs.shape[0]):
        zk = zs[i]
        # EKF performs time update
        Phi_ksr, Gamma_ksr = ekf_callback_Phi_Gam(x_ksr, None, None)
        
        # Opton 0: Take leo.W as W_ksr
        #Jac = ce.cd4_gvf(x_ksr, leo5_ode)
        # Option 1: Solve power series formula as W_ksr
        #_Phi,_Gamma, W_ksr = ce.discretize_nl_sys(Jac, np.array([[0,0,0,0,1.0]]).T, np.array([[(leo.beta_drag*leo.CAUCHY_TO_GAUSS)**2]]), leo.dt, 6)
        # Option 2: Use Lyapunov solver as W_ksr
        #Q = np.zeros((5,5)) * leo.q
        #Q[4,4] = (leo.beta_drag * leo.CAUCHY_TO_GAUSS)**2
        #W_ksr = ce.discretize_ctime_process_noise(Jac, Q, leo.dt, 6)
        # Option 3: Take rank 1 W_ksr
        #W_ksr = _Gamma @ np.array([[leo.beta_gauss**2]]) @ _Gamma.T


        x_ksr =  ekf_f(x_ksr, None, None)
        M_ksr = Phi_ksr @ P_ksr @ Phi_ksr.T + Gamma_ksr @ (W_ksr) @ Gamma_ksr.T 
        M_ksr = (M_ksr + M_ksr.T)/2

        z_ksr_psuedo = xs[i,4] + np.random.randn() * gamma[0] * leo.CAUCHY_TO_GAUSS
        # Cauchy runs time prop and measurement update 
        cauchyEst.step(z_ksr_psuedo, None)
        # Cauchy rectifies EKF density: (this needs some work)
        ce_cond_mean = cauchyEst.moment_info["x"][-1]
        ce_cond_var = cauchyEst.moment_info["P"][-1]
        print("Norm Vel:", np.linalg.norm(x_ksr[2:4]), "True Drag is: ", xs[i, 4], "Estimated Drag is: ", ce_cond_mean.item())

        # Rectify EKF
        x_ksr[4] = ce_cond_mean
        M_ksr[:4,4] = M_ksr[:4,4] * ce_cond_var**.5 / M_ksr[4,4]**.5
        M_ksr[4,:4] = M_ksr[:4,4].T
        M_ksr[4,4] = ce_cond_var

        H_ksr = ekf_callback_H(x_ksr, None)
        K_ksr = M_ksr @ H_ksr.T @ np.linalg.inv( H_ksr @ M_ksr @ H_ksr.T + V_ksr)
        K_ksr[4,:] = 0
        zbar_ksr = ekf_h(x_ksr, None)
        r_k = zk - zbar_ksr
        x_ksr = x_ksr + K_ksr @ r_k
        P_ksr = (I_ksr - K_ksr @ H_ksr) @ M_ksr @ (I_ksr - K_ksr @ H_ksr).T + K_ksr @ V_ksr @ K_ksr.T

        min_eig = np.min(np.linalg.eig(P_ksr)[0])
        if min_eig < 0:
            P_ksr += np.eye(5)*1e-3
            print("KF going semidef!")
        # Save results
        xs_ksr.append(x_ksr.copy())
        Ps_ksr.append(P_ksr.copy())

    #xs_ksr = np.array(xs_ksr)
    #Ps_ksr = np.array(Ps_ksr)
    foo = np.zeros(xs.shape[0])
    moment_info = {"x": xs_ksr, "P": Ps_ksr, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf))
    foobar=2
    '''

def test_kalman_filter_smoother():

    global leo
    #seed = 2124125479 #-- no huge jumps
    seed = 178211974 #-- one mild jump
    #seed = int(np.random.rand() * (2**32 -1)) #3872826552#
    with_sas_density=True
    with_added_jumps=True
    density_scale = 10000
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Get Ground Truth and Measurements
    prop_steps = 500
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=with_sas_density, with_added_jumps=with_added_jumps)

    # Run EKF 
    x_kf = leo.x0.copy()
    P_kf = leo.P0.copy()
    W_kf = leo.W.copy() 
    W_kf[4,4] *= density_scale
    V_kf = leo.V.copy()
    I = np.eye(5)
    xs_kf = [x_kf.copy()]
    Ms_kf = [P_kf.copy()]
    Ps_kf = [P_kf.copy()]
    Phis_kf = []
    for i in range(1, zs.shape[0]):
        zk = zs[i]
        # EKF performs time update
        Phi, Gamma = ekf_callback_Phi_Gam(x_kf, None, None)
        Phis_kf.append(Phi.copy())
        
        # Opton 0: Take leo.W as W_kf
        # Option 1: Solve power series formula as W_ksr
        #Jac = ce.cd4_gvf(x_ksr, leo5_ode)
        #_Phi,_Gamma, W_kf = ce.discretize_nl_sys(Jac, np.array([[0,0,0,0,1.0]]).T, np.array([[(leo.beta_drag*leo.CAUCHY_TO_GAUSS)**2]]), leo.dt, 6)
        # Option 2: Use Lyapunov solver as W_ksr
        #Q = np.zeros((5,5)) * leo.q
        #Q[4,4] = (leo.beta_drag * leo.CAUCHY_TO_GAUSS)**2
        #W_kf = ce.discretize_ctime_process_noise(Jac, Q, leo.dt, 6)
        # Option 3: Take rank 1 W_ksr
        #W_kf = _Gamma @ np.array([[leo.beta_gauss**2]]) @ _Gamma.T


        x_kf =  ekf_f(x_kf, None, None)
        M_kf = Phi @ P_kf @ Phi.T + Gamma @ W_kf @ Gamma.T 
        M_kf = (M_kf + M_kf.T)/2
        Ms_kf.append(M_kf.copy())

        H = ekf_callback_H(x_kf, None)
        K = M_kf @ H.T @ np.linalg.inv( H @ M_kf @ H.T + V_kf)
        zbar = ekf_h(x_kf, None)
        r_k = zk - zbar
        x_kf = x_kf + K @ r_k
        P_kf = (I - K @ H) @ M_kf @ (I - K @ H).T + K @ V_kf @ K.T

        min_eig = np.min(np.linalg.eig(P_kf)[0])
        if min_eig < 0:
            P_kf += np.eye(5)*1e-3
            print("KF going semidef!")
        # Save results
        xs_kf.append(x_kf.copy())
        Ps_kf.append(P_kf.copy())
    xs_kf = np.array(xs_kf)
    Ps_kf = np.array(Ps_kf)
    Ms_kf = np.array(Ms_kf)
    Phis_kf = np.array(Phis_kf)

    # Run Smoother 
    xs_smoothed, Ps_smoothed = gf.ekf_smoother(xs_kf, Ps_kf, Ms_kf, Phis_kf, leo_5state_transition_model)

    # Plot EKF vs Smoothed EKF
    foo = np.zeros(xs.shape[0])
    moment_info = {"x": xs_smoothed, "P": Ps_smoothed, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf,Ps_kf))
    foo=2

def test_kf_ce_smoothing():
    log_dir = file_dir + "/../../../log/leo5/dense/gps/w8/400km"
    sim_history = ce.load_sim_truth_log_folder(log_dir)
    ce_moment_info = ce.load_cauchy_log_folder(log_dir)
    xs_ce = ce_moment_info["x"]
    Ps_ce = ce_moment_info["P"]
    use_stored_kf_data = False
    attempt_to_smooth_cauchy=False
    _xs_kf, _Ps_kf = ce.load_kalman_log_folder(log_dir)

    if use_stored_kf_data:
        # Just load in 
        xs_kf = _xs_kf
        Ps_kf = _Ps_kf
        W_cov = leo.W.copy()
        # Create Phis and Ms for Kalman Smoother
        Ms_kf = [Ps_kf[0].copy()] 
        Phis_kf = []
        for i in range(xs_kf.shape[0]-1):
            x_kf = xs_kf[i]
            P_kf = Ps_kf[i]
            Phi_kf, _ = leo_5state_transition_model_jacobians(x_kf)
            M_kf = Phi_kf @ P_kf @ Phi_kf.T + W_cov
            M_kf = (M_kf + M_kf.T)/2
            Phis_kf.append(Phi_kf.copy())
            Ms_kf.append(M_kf.copy())
        Phis_kf = np.array(Phis_kf)
        Ms_kf = np.array(Ms_kf)
    else:
        # (Re)Run EKF to test things
        zs = sim_history[1]
        x0_true = sim_history[0][0]
        x_kf = x0_true + np.random.multivariate_normal(np.zeros(5), _Ps_kf[0])
        P_kf = _Ps_kf[0]
        W_kf = leo.W.copy()
        W_kf[4,4] *= 1000
        V_kf = leo.V.copy()
        I = np.eye(5)
        zs = sim_history[1]
        xs_kf = [x_kf.copy()]
        Ms_kf = [P_kf.copy()]
        Ps_kf = [P_kf.copy()]
        Phis_kf = []
        for i in range(1, zs.shape[0]):
            zk = zs[i]
            # EKF performs time update
            Phi, Gamma = ekf_callback_Phi_Gam(x_kf, None, None)
            Phis_kf.append(Phi.copy())
            
            # Opton 0: Take leo.W as W_kf
            # Option 1: Solve power series formula as W_ksr
            #Jac = ce.cd4_gvf(x_ksr, leo5_ode)
            #_Phi,_Gamma, W_kf = ce.discretize_nl_sys(Jac, np.array([[0,0,0,0,1.0]]).T, np.array([[(leo.beta_drag*leo.CAUCHY_TO_GAUSS)**2]]), leo.dt, 6)
            # Option 2: Use Lyapunov solver as W_ksr
            #Q = np.zeros((5,5)) * leo.q
            #Q[4,4] = (leo.beta_drag * leo.CAUCHY_TO_GAUSS)**2
            #W_kf = ce.discretize_ctime_process_noise(Jac, Q, leo.dt, 6)
            # Option 3: Take rank 1 W_ksr
            #W_kf = _Gamma @ np.array([[leo.beta_gauss**2]]) @ _Gamma.T

            x_kf =  ekf_f(x_kf, None, None)
            M_kf = Phi @ P_kf @ Phi.T + Gamma @ W_kf @ Gamma.T 
            M_kf = (M_kf + M_kf.T)/2
            Ms_kf.append(M_kf.copy())

            H = ekf_callback_H(x_kf, None)
            K = M_kf @ H.T @ np.linalg.inv( H @ M_kf @ H.T + V_kf)
            zbar = ekf_h(x_kf, None)
            r_k = zk - zbar
            x_kf = x_kf + K @ r_k
            P_kf = (I - K @ H) @ M_kf @ (I - K @ H).T + K @ V_kf @ K.T

            min_eig = np.min(np.linalg.eig(P_kf)[0])
            if min_eig < 0:
                P_kf += np.eye(5)*1e-3
                print("KF going semidef!")
            # Save results
            xs_kf.append(x_kf.copy())
            Ps_kf.append(P_kf.copy())
        xs_kf = np.array(xs_kf)
        Ps_kf = np.array(Ps_kf)
        Ms_kf = np.array(Ms_kf)
        Phis_kf = np.array(Phis_kf)

    # Smooth Kalman Estimates
    #print("Kalman Filter (green/magenta) vs Kalman Filter Smoothed (blue/red) and there 1-sigma bounds")
    xs_kf_smoothed, Ps_kf_smoothed = gf.ekf_smoother(xs_kf, Ps_kf, Ms_kf, Phis_kf, leo_5state_transition_model)
    #foo = np.zeros(xs_kf.shape[0])
    #moment_info = {"x": xs_kf_smoothed, "P": Ps_kf_smoothed, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    #ce.plot_simulation_history(moment_info, sim_history, (xs_kf, Ps_kf))

    if attempt_to_smooth_cauchy:
        # This does not work very well...
        # Smooth Cauchy Estimates?
        # Create Phis and Ms for Cauchy 'Smoothed'
        Ms_ce = [Ps_ce[0].copy()]
        Phis_ce = []
        W_cov[4,4] *= 500
        for i in range(xs_ce.shape[0]-1):
            x_ce = xs_ce[i]
            P_ce = Ps_ce[i]
            Phi_ce, _ = leo_5state_transition_model_jacobians(x_ce)
            M_ce = Phi_ce @ P_ce @ Phi_ce.T + W_cov
            M_ce = (M_ce + M_ce.T)/2
            Phis_ce.append(Phi_ce.copy())
            Ms_ce.append(M_ce.copy())
        Phis_ce = np.array(Phis_ce)
        Ms_ce = np.array(Ms_ce)

        # Smooth Cauchy Estimates?
        print("Cauchy Estimator (green/magenta) vs Cauchy Estimator Smoothed (blue/red) and there 1-sigma bounds")
        xs_ce_smoothed, Ps_ce_smoothed = gf.ekf_smoother(xs_ce, Ps_ce, Ms_ce, Phis_ce, leo_5state_transition_model)

        foo = np.zeros(xs_ce.shape[0])
        moment_info = {"x": xs_ce_smoothed, "P": Ps_ce_smoothed, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
        xs,zs,ws,vs = sim_history
        #ce.plot_simulation_history( moment_info, (xs[1:], zs[1:], ws[1:], vs[1:]), (xs_ce, Ps_ce))
    

    # Compare Non-smoothed estimates:
    foo = np.zeros(xs_ce.shape[0])
    moment_info = {"x": xs_ce, "P": Ps_ce, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history( moment_info, sim_history, (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True )
    foobar = 2

    # Compare Smoothed estimates:
    if attempt_to_smooth_cauchy:
        moment_info = {"x": xs_ce_smoothed, "P": Ps_ce_smoothed, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    else:
        moment_info = {"x": xs_ce, "P": Ps_ce, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history( moment_info, sim_history, (xs_kf_smoothed, Ps_kf_smoothed), with_partial_plot=True, with_cauchy_delay=True, scale=1)
    foobar = 2

def one_sigma_data(xs_truth, xs_est, Ps_est, scale=1):
    err = xs_truth - xs_est
    sig1 = np.array([np.diag(pest)**0.5 for pest in Ps_est])
    up = sig1 * scale
    down = -sig1 * scale
    return err, up, down

def test_leo5_windows():
    global leo
    # 2124125479 -- no huge jumps
    seed = 2124125479 #int(np.random.rand() * (2**32 -1)) #3872826552#
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Get Ground Truth and Measurements
    prop_steps = 10
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=True, with_added_jumps=False)
    zs_without_z0 = zs[1:,:]

    # Run EKF 
    W_kf = leo.W.copy()
    V_kf = leo.V.copy()
    W_kf[4,4] *= 1000
    xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    #ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf))

    # Run Cauchy Estimator
    #'''
    beta = np.array([leo.beta_cauchy])
    gamma = np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])
    
    # Create Phi.T as A0, start at propagated x0
    # Initialize Initial Hyperplanes
    Phi, _ = leo_5state_transition_model_jacobians(leo.x0)
    xbar = leo_5state_transition_model(leo.x0)
    A0 = Phi.T.copy()
    p0 = np.repeat(leo.alpha_pv_cauchy, 5)
    p0[4] = leo.alpha_density_cauchy
    b0 = np.zeros(5)
    num_controls = 0

    num_windows = 5
    total_steps = prop_steps
    ce.set_tr_search_idxs_ordering([3,2,4,1,0])
    
    cauchyEsts = [ce.PyCauchyEstimator("nonlin", num_windows, True) for _ in range(4)]
    cauchyEsts[0].initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
    win0_xs = []
    win0_Ps = []
    for i in range(1, num_windows+1):
        zk = zs[i]
        xs_ce, Ps_ce = cauchyEsts[0].step(zk, None, full_info=True)
        win0_xs.append(xs_ce)
        win0_Ps.append(Ps_ce)
    cauchyEsts[0].shutdown()
    final_win0_xs = np.array([wxs[1] for wxs in win0_xs])
    final_win0_Ps = np.array([wPs[1] for wPs in win0_Ps])

    # Now start up window 1 using info of window 0
    win1_xbar = leo_5state_transition_model(win0_xs[0][1])
    win1_dx = win0_xs[1][0] - win1_xbar
    win1_dz = zs[2] - leo_5state_measurement_model(win1_xbar)
    win1_A0, win1_p0, win1_b0 = ce.speyers_window_init(win1_dx, win0_Ps[1][0], np.array([1.0, 0, 0, 0, 0]), gamma[0], win1_dz[0])
    
    # Overriding Win1_A0,p0,b0
    Phi1, _ = leo_5state_transition_model_jacobians(win0_xs[0][1])
    win1_A0 = Phi1.T
    win1_p0 = p0.copy()
    win1_b0 = b0.copy()


    cauchyEsts[1].initialize_nonlin(win1_xbar, win1_A0, win1_p0, win1_b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
    win1_xs = []
    win1_Ps = []
    for i in range(2, num_windows+2):
        zk = zs[i]
        xs_ce, Ps_ce = cauchyEsts[1].step(zk, None, full_info=True)
        win1_xs.append(xs_ce)
        win1_Ps.append(Ps_ce)
    cauchyEsts[1].shutdown()
    final_win1_xs = np.array([wxs[1] for wxs in win1_xs])
    final_win1_Ps = np.array([wPs[1] for wPs in win1_Ps])

    # Now start up window 2 using info of window 0
    win2_xbar = leo_5state_transition_model(win0_xs[1][1])
    win2_dx = win0_xs[2][0] - win2_xbar
    win2_dz = zs[3] - leo_5state_measurement_model(win2_xbar)
    win2_A0, win2_p0, win2_b0 = ce.speyers_window_init(win2_dx, win0_Ps[2][0], np.array([1.0, 0, 0, 0, 0]), gamma[0], win2_dz[0])
    
    # Overriding Win2_A0,p0,b0
    Phi2, _ = leo_5state_transition_model_jacobians(win0_xs[1][1])
    win2_A0 = Phi2.T
    win2_p0 = p0.copy()
    win2_b0 = b0.copy()

    cauchyEsts[2].initialize_nonlin(win2_xbar, win2_A0, win2_p0, win2_b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
    win2_xs = []
    win2_Ps = []
    for i in range(3, num_windows+3):
        zk = zs[i]
        xs_ce, Ps_ce = cauchyEsts[2].step(zk, None, full_info=True)
        win2_xs.append(xs_ce)
        win2_Ps.append(Ps_ce)
    cauchyEsts[2].shutdown()
    final_win2_xs = np.array([wxs[1] for wxs in win2_xs])
    final_win2_Ps = np.array([wPs[1] for wPs in win2_Ps])

    # Now start up window 3 using info of window 0
    win3_xbar = leo_5state_transition_model(win0_xs[2][1])
    win3_dx = win0_xs[3][0] - win3_xbar
    win3_dz = zs[4] - leo_5state_measurement_model(win3_xbar)
    win3_A0, win3_p0, win3_b0 = ce.speyers_window_init(win3_dx, win0_Ps[3][0], np.array([1.0, 0, 0, 0, 0]), gamma[0], win3_dz[0])
    
    # Overriding Win3_A0,p0,b0
    Phi3, _ = leo_5state_transition_model_jacobians(win0_xs[2][1])
    win3_A0 = Phi3.T
    win3_p0 = p0.copy()
    win3_b0 = b0.copy()

    cauchyEsts[3].initialize_nonlin(win3_xbar, win3_A0, win3_p0, win3_b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
    win3_xs = []
    win3_Ps = []
    for i in range(4, num_windows+4):
        zk = zs[i]
        xs_ce, Ps_ce = cauchyEsts[3].step(zk, None, full_info=True)
        win3_xs.append(xs_ce)
        win3_Ps.append(Ps_ce)
    cauchyEsts[3].shutdown()
    final_win3_xs = np.array([wxs[1] for wxs in win3_xs])
    final_win3_Ps = np.array([wPs[1] for wPs in win3_Ps])


    # Plot KF Bound vs Cauchy Bounds
    T = xs_kf.shape[0]
    Ts = np.arange(T)
    kf_err, kf_up, kf_down = one_sigma_data(xs[:T,:], xs_kf, Ps_kf, scale=1)
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(Ts, kf_err[:,i], 'g')
        plt.plot(Ts, kf_up[:,i], 'g--')
        plt.plot(Ts, kf_down[:,i], 'g--')
    
    # 
    win0_T = final_win0_xs.shape[0]
    win0_err, win0_up, win0_down = one_sigma_data(xs[1:win0_T+1,:], final_win0_xs, final_win0_Ps, scale=1)
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(Ts[1:win0_T+1], win0_err[:,i], 'b')
        plt.plot(Ts[1:win0_T+1], win0_up[:,i], 'b--')
        plt.plot(Ts[1:win0_T+1], win0_down[:,i], 'b--')

    win1_T = final_win1_xs.shape[0]
    win1_err, win1_up, win1_down = one_sigma_data(xs[2:win1_T+2,:], final_win1_xs, final_win1_Ps, scale=1)
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(Ts[2:win1_T+2], win1_err[:,i], 'r')
        plt.plot(Ts[2:win1_T+2], win1_up[:,i], 'r--')
        plt.plot(Ts[2:win1_T+2], win1_down[:,i], 'r--')
    
    win2_T = final_win2_xs.shape[0]
    win2_err, win2_up, win2_down = one_sigma_data(xs[3:win2_T+3,:], final_win2_xs, final_win2_Ps, scale=1)
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(Ts[3:win2_T+3], win2_err[:,i], 'm')
        plt.plot(Ts[3:win2_T+3], win2_up[:,i], 'm--')
        plt.plot(Ts[3:win2_T+3], win2_down[:,i], 'm--')
    
    win3_T = final_win3_xs.shape[0]
    win3_err, win3_up, win3_down = one_sigma_data(xs[4:win3_T+4,:], final_win3_xs, final_win3_Ps, scale=1)
    for i in range(5):
        plt.subplot(5,1,i+1)
        plt.plot(Ts[4:win3_T+4], win3_err[:,i], 'k')
        plt.plot(Ts[4:win3_T+4], win3_up[:,i], 'k--')
        plt.plot(Ts[4:win3_T+4], win3_down[:,i], 'k--')

    plt.show()
    foobar = 2

def test_innovation():
    global leo
    #seed = 2124125479 #-- no huge jumps
    seed = 178211974 #-- one mild jump
    #seed = int(np.random.rand() * (2**32 -1)) #3872826552#
    with_sas_density=False
    with_added_jumps=True
    return_full_proc_noise=True
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Get Ground Truth and Measurements
    prop_steps = 80
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=with_sas_density, with_added_jumps=with_added_jumps, return_full_proc_noise=return_full_proc_noise)

    xk = xs[0].copy() # x0_truth
    xs2 = [xk.copy()]
    ws2 = ws.copy()
    ws2[20,4] = 0
    #ws2[60,4] = 0
    #ws2[100,4] = 0
    T = xs.shape[0]-1
    for i in range(T):
        xk = leo_5state_transition_model(xk)+ ws2[i]
        xs2.append(xk.copy())
    xs2 = np.array(xs2)

    #plt.title("Innovation Test:")
    Ts = np.arange(T+1)
    diff = xs - xs2
    r1 = np.linalg.norm(xs[:,0:2], axis = 1)
    r2 = np.linalg.norm(xs2[:,0:2], axis = 1)
    v1 = np.linalg.norm(xs[:,2:4], axis = 1)
    v2 = np.linalg.norm(xs2[:,2:4], axis = 1)
    plt.figure(1)
    for i in range(5):
        plt.subplot(5,1, i+1)
        if i == 0:
            plt.title("Differences in magnitude")
        plt.plot(Ts, diff[:,i])
    max_diff = np.max(np.abs(diff))
    plt.figure(2)
    plt.subplot(2,2,1)
    plt.title("Norm, Position")
    plt.plot(Ts, r1, 'b')
    plt.plot(Ts, r2, 'g')
    plt.subplot(2,2,2)
    plt.title("Diff Norm, Position")
    plt.plot(Ts, r1-r2, 'b')
    plt.subplot(2,2,3)
    plt.title("Norm, Velocity")
    plt.plot(Ts, v1, 'b')
    plt.plot(Ts, v2, 'g')
    plt.subplot(2,2,4)
    plt.title("Diff Norm, Velocity")
    plt.plot(Ts, v1-v2, 'b')

    plt.show()
    print("max_diff is", max_diff)
    
def test_innovation2():
    global leo
    #seed = 2124125479 #-- no huge jumps
    seed = 178211974 #-- one mild jump
    #seed = int(np.random.rand() * (2**32 -1)) #3872826552#
    with_sas_density=True
    with_added_jumps=False
    return_full_proc_noise=True
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Get Ground Truth and Measurements
    prop_steps = 720
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=with_sas_density, with_added_jumps=with_added_jumps, return_full_proc_noise=return_full_proc_noise)
    xs2, zs, ws2, vs = simulate_leo5_state(prop_steps, with_sas_density=with_sas_density, with_added_jumps=with_added_jumps, return_full_proc_noise=return_full_proc_noise)

    #plt.title("Innovation Test:")
    T = xs.shape[0]-1
    Ts = np.arange(T+1)
    diff = xs - xs2
    r1 = np.linalg.norm(xs[:,0:2], axis = 1)
    r2 = np.linalg.norm(xs2[:,0:2], axis = 1)
    v1 = np.linalg.norm(xs[:,2:4], axis = 1)
    v2 = np.linalg.norm(xs2[:,2:4], axis = 1)
    diff_r = np.sqrt( np.sum((xs[:,0:2] - xs2[:,0:2])**2, axis=1))
    diff_v = np.sqrt( np.sum((xs[:,2:4] - xs2[:,2:4])**2, axis=1))
    plt.figure(1)
    for i in range(5):
        plt.subplot(5,1, i+1)
        if i == 0:
            plt.title("Differences in magnitude")
        plt.plot(Ts, diff[:,i])
    max_diff = np.max(np.abs(diff))
    plt.figure(2)
    plt.subplot(2,2,1)
    plt.title("Norm, Position")
    plt.plot(Ts, r1, 'b')
    plt.plot(Ts, r2, 'g')
    plt.subplot(2,2,2)
    plt.title("Diff Norm, Position")
    plt.plot(Ts, diff_r, 'b')
    plt.subplot(2,2,3)
    plt.title("Norm, Velocity")
    plt.plot(Ts, v1, 'b')
    plt.plot(Ts, v2, 'g')
    plt.subplot(2,2,4)
    plt.title("Diff Norm, Velocity")
    plt.plot(Ts, diff_v, 'b')

    plt.show()
    print("max_diff is", max_diff)
    
def test_python_debug_window_manager():
    global leo
    # 2124125479 -- no huge jumps
    seed = 2124125479 #int(np.random.rand() * (2**32 -1)) #3872826552#
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Get Ground Truth and Measurements
    #zs = np.genfromtxt(file_dir + "/../../../log/leo5/dense/w5/msmts.txt", delimiter= ' ')
    #zs = zs[1:,:]
    prop_steps = 150
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=True, with_added_jumps=True)
    zs_without_z0 = zs[1:,:]

    # Run EKF 
    '''
    W_kf = leo.W.copy()
    V_kf = leo.V.copy()
    W_kf[4,4] *= 1000
    xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    
    ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf))
    '''

    # Run Cauchy Estimator
    #'''
    beta = np.array([leo.beta_cauchy])
    gamma = np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])
    #beta_scale = 50
    #beta = np.array([leo.beta_cauchy / beta_scale])
    #gamma_scale = 5
    #gamma = gamma_scale*np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])
    
    # Create Phi.T as A0, start at propagated x0
    # Initialize Initial Hyperplanes
    Phi, _ = leo_5state_transition_model_jacobians(leo.x0)
    xbar = leo_5state_transition_model(leo.x0)
    A0 = Phi.T.copy()
    p0 = np.repeat(leo.alpha_pv_cauchy, 5)
    p0[4] = leo.alpha_density_cauchy
    b0 = np.zeros(5)
    num_controls = 0

    num_windows = 6
    total_steps = prop_steps
    ce.set_tr_search_idxs_ordering([3,2,4,1,0])
    log_dir = file_dir + "/pylog/debug_w"+str(6) + "_" + str(int(leo.r_sat/1000)) + "km"
    debug_print = True
    #cauchyEsts = [ce.PyCauchyEstimator("nonlin", num_windows, debug_print) for _ in range(num_windows)]#ce.PySlidingWindowManager("nonlin", num_windows, total_steps, log_dir=log_dir, log_seq=True, log_full=True)
    #cauchyEsts[0].initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)

    cauchyEst = ce.PyCauchyEstimator("nonlin", num_windows, debug_print)
    cauchyEst.initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
    for i in range(1, num_windows+1):
        zk = zs[i]
        xhat, Phat = cauchyEst.step(zk, None)
        xtrue = xs[i]

        std00 = 3*np.sqrt(Phat[0,0])
        std11 = 3*np.sqrt(Phat[1,1])
        std22 = 3*np.sqrt(Phat[2,2])
        std33 = 3*np.sqrt(Phat[3,3])
        std44 = 3*np.sqrt(Phat[4,4])
        x0, y0 = cauchyEst.get_marginal_1D_pointwise_cpdf(0, -std00, std00, 0.001)
        x1, y1 = cauchyEst.get_marginal_1D_pointwise_cpdf(1, -std11, std11, 0.001)
        x2, y2 = cauchyEst.get_marginal_1D_pointwise_cpdf(2, -std22, std22, 0.001)
        x3, y3 = cauchyEst.get_marginal_1D_pointwise_cpdf(3, -std33, std33, 0.001)
        x4, y4 = cauchyEst.get_marginal_1D_pointwise_cpdf(4, -std44, std44, 0.001)
        plt.subplot(511)
        plt.plot(x0 + xhat[0], y0)
        plt.scatter(xtrue[0], 0, color='r', marker='x')
        plt.scatter(xhat[0], 0, color='b', marker='x')
        plt.subplot(512)
        plt.plot(x1 + xhat[1], y1)
        plt.scatter(xtrue[1], 0, color='r', marker='x')
        plt.scatter(xhat[1], 0, color='b', marker='x')
        plt.subplot(513)
        plt.plot(x2 + xhat[2], y2)
        plt.scatter(xtrue[2], 0, color='r', marker='x')
        plt.scatter(xhat[2], 0, color='b', marker='x')
        plt.subplot(514)
        plt.plot(x3 + xhat[3], y3)
        plt.scatter(xtrue[3], 0, color='r', marker='x')
        plt.scatter(xhat[3], 0, color='b', marker='x')
        plt.subplot(515)
        plt.plot(x4 + xhat[4], y4)
        plt.scatter(xtrue[4], 0, color='r', marker='x')
        plt.scatter(xhat[4], 0, color='b', marker='x')
        plt.show()
        foobar = 3



    #for zk in zs_without_z0:
    #    cauchyEst.step(zk, None)
    #cauchyEst.shutdown()
    #'''

    #ce.plot_simulation_history(cauchyEst.moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True)
    foobar = 2


if __name__ == '__main__':
    #test_leo5()
    #test_kalman_schmidt_recursion_cascade()
    #test_kalman_schmidt_recursion_russell()
    #test_kalman_filter_smoother()
    #test_kf_ce_smoothing()
    #test_leo5_windows()
    #test_innovation()
    #test_innovation2()
    test_python_debug_window_manager()