import numpy as np
import os, sys 
import matplotlib.pyplot as plt
file_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(file_dir + "/../cauchy")
import cauchy_estimator as ce
import gaussian_filters as gf
import math
import pickle
import matplotlib
matplotlib.use('TkAgg',force=True)

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

    def __init__(self, leo5_alt = 200e3, leo5_A = 64.0, leo5_m = 5000, leo5_gps_std_dev = 2.0, leo5_dt = 60.0):
        # Size of simulation dynamics
        self.n = 5
        self.num_satellites = 2 # number of sattelites to talk to (measurements)
        self.p = self.num_satellites
        self.pncc = 1
        self.cmcc = 0
        # Orbital distances
        self.r_earth = 6378.1e3 # spherical approximation of earths radius (meters)
        self.r_sat = leo5_alt #550e3 # orbit distance of satellite above earths surface (meters)
        
        # Satellite parameter specifics
        self.M = 5.9722e24 # Mass of earth (kg)
        self.G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
        self.mu = self.M*self.G  #Nm^2/kg^2
        self.m = leo5_m # kg
        self.rho = lookup_air_density(self.r_sat) # kg/m^3
        self.C_D = 2.0 #drag coefficient
        self.A = leo5_A #m^2
        self.tau = 21600.0 # 1/(m*sec)
        # Parameters for runge kutta ODE integrator
        self.dt = leo5_dt #time step in sec
        self.sub_steps_per_dt = int(leo5_dt) # so sub intervals are 1 second
        # Initial conditions
        self.r0 = self.r_earth + self.r_sat # orbit distance from center of earth
        self.v0 = np.sqrt(self.mu/self.r0) # speed of the satellite in orbit for distance r0
        self.x0 = np.array([self.r0/np.sqrt(2), self.r0/np.sqrt(2), self.v0/np.sqrt(2), -self.v0/np.sqrt(2), 0.0])
        self.omega0 = self.v0/self.r0 # rad/sec (angular rate of orbit)
        self.orbital_period = 2.0*np.pi / self.omega0 #Period of orbit in seconds
        self.time_steps_per_period = (int)(self.orbital_period / self.dt + 0.50) # number of dt's until 1 revolution is made
        self.num_revolutions = 10
        self.num_simulation_steps = self.num_revolutions * self.time_steps_per_period
        # Satellite parameters for measurement update
        self.satellite_positions = np.array([ [-7e6, -7e6], [7e6, 7e6] ])
        self.dt_R = 0.0 # bias time of sattelite clocks, for now its zero
        self.b = np.zeros(2)
        self.std_dev_gps = leo5_gps_std_dev# 2.0 # uncertainty in GPS measurement (meters)
        self.V = np.array([ [pow(self.std_dev_gps,2), 0], [0, pow(self.std_dev_gps,2)] ])
        self.cholV = np.linalg.cholesky(self.V)
        # Conversion Parameters 
        self.SAS_alpha = 1.3
        self.CAUCHY_TO_GAUSS = 1.3898
        self.GAUSS_TO_CAUCHY = 1.0 / self.CAUCHY_TO_GAUSS
        self.beta_drag = 0.0013
        self.beta_gauss = (self.beta_drag * self.CAUCHY_TO_GAUSS) / (self.tau * (1.0 - np.exp(-self.dt/self.tau)))
        self.beta_cauchy = self.beta_gauss * self.GAUSS_TO_CAUCHY
        # Satellite parameters for process noise
        self.q = 8e-15; # Process noise ncertainty in the process position and velocity
        self.W = np.zeros(25)
        self.W[0] = pow(self.dt,3)/3*self.q
        self.W[6] = pow(self.dt,3)/3*self.q
        self.W[2] = pow(self.dt,2)/2*self.q
        self.W[8] = pow(self.dt,2)/2*self.q
        self.W[10] = pow(self.dt,2)/2*self.q
        self.W[16] = pow(self.dt,2)/2*self.q
        self.W[12] = self.dt*self.q
        self.W[18] = self.dt*self.q
        self.W[24] = pow( self.beta_drag * self.CAUCHY_TO_GAUSS, 2)
        self.W = self.W.reshape((5,5))
        self.cholW = np.linalg.cholesky(self.W)
        self.Wd = np.array([[self.beta_gauss**2]])
        # Initial uncertainty in position
        self.alpha_density_cauchy = 0.0039 # Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
        self.alpha_density_gauss = self.alpha_density_cauchy * self.CAUCHY_TO_GAUSS # Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
        self.alpha_pv_gauss = 0.01 # Initial Gaussian standard deviation in position and velocity of satellite
        self.alpha_pv_cauchy = self.alpha_pv_gauss * self.GAUSS_TO_CAUCHY # Initial converted uncertainty parameter in position and velocity of satellite converted for Cauchy Estimator
        self.P0 = np.zeros(25)
        self.P0[0] = pow(self.alpha_pv_gauss,2); self.P0[6] = pow(self.alpha_pv_gauss,2)
        self.P0[12] = pow(self.alpha_pv_gauss,2); self.P0[18] = pow(self.alpha_pv_gauss,2)
        self.P0[24] = pow(self.alpha_density_gauss,2)
        self.P0 = self.P0.reshape((5,5))
        self.cholP0 = np.linalg.cholesky(self.P0)
        # For Kalman Schmidt Recursion
        self.last_kf_est = self.x0[2:4].copy()
    
leo = leo_satellite_5state()
INITIAL_H = False

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
    taylor_order = 3
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
    global INITIAL_H
    if(INITIAL_H):
        zbar = np.array([0, xbar[0] + xbar[1]])
    else:
        zbar = leo_5state_measurement_model(xbar)
    pyduc.cset_zbar(c_zbar, zbar)

def ece_extended_msmt_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    global INITIAL_H
    if INITIAL_H:
        H = np.zeros((2,5))
        H[0,0] = 0
        H[1,0] = 1
        H[1,1] = 1
        global leo 
        gam = leo.std_dev_gps * leo.GAUSS_TO_CAUCHY
        gamma = np.array([gam, 2*gam])
        pyduc.cset_gamma(gamma)
    else:
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
            if i == 30:
                wk[4] = 7.5
            if i == 100:
                wk[4] = -2.0
            if i == 160:
                wk[4] = -1.0
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
    prop_steps = 900
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

# Sliding Window Debugger
def test_single_sliding_window():
    global leo
    # 2124125479 -- no huge jumps
    seed = 2124125479 #int(np.random.rand() * (2**32 -1)) #3872826552#
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Leo Satelite Parameters
    leo5_alt = 550e3 # meters
    leo5_A = 14 # meters^2
    leo5_m = 5000 # kg
    leo5_gps_std_dev = 7.5 # meters
    leo = leo_satellite_5state(leo5_alt, leo5_A, leo5_m, leo5_gps_std_dev)

    # Cauchy and Kalman Tunables
    num_window_steps = 8
    prop_steps = num_window_steps # Number of time steps to run sim
    gamma_scale = 1 # scaling gamma up by .... (1 is normal)
    beta_scale = 1 # scaling beta down by ... (1 is normal)

    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=True, with_added_jumps=False)
    zs_without_z0 = zs[1:,:]

    # Run Cauchy Estimator
    #'''
    beta = np.array([leo.beta_cauchy])
    gamma = np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])
    beta /= beta_scale
    gamma /= gamma_scale

    # Create Phi.T as A0, start at propagated x0
    # Initialize Initial Hyperplanes
    Phi, _ = leo_5state_transition_model_jacobians(leo.x0)
    xbar = leo_5state_transition_model(leo.x0)
    A0 = Phi.T.copy() # np.linalg.eig(Ps_kf[35])[1].T #Phi.T.copy()
    p0 = np.repeat(leo.alpha_pv_cauchy, 5)
    p0[4] = leo.alpha_density_cauchy
    b0 = np.zeros(5)
    num_controls = 0

    ce.set_tr_search_idxs_ordering([3,2,4,1,0])
    debug_print = True
    cauchyEst = ce.PyCauchyEstimator("nonlin", num_window_steps, debug_print) 
    cauchyEst.initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    for i in range(num_window_steps):
        xhat, Phat = cauchyEst.step(zs_without_z0[i], None)

# choose window with last estimate's covariance defined
def best_window_est(cauchyEsts, window_counts):
    W = len(cauchyEsts)
    okays = np.zeros(W, dtype=np.bool8)
    idxs = []
    for i in range(W):
        if(window_counts[i] > 0):
            err = cauchyEsts[i]._err_code
            if (err[1] & (1<<1)) or (err[1] & (1<<3)):
                pass
            else:
                idxs.append((i, window_counts[i]))
                okays[i] = True
    if(len(idxs) == 0):
        print("No window is available without an error code!")
        exit(1)
    sorted_idxs = list(reversed(sorted(idxs, key = lambda x : x[1])))
    return sorted_idxs[0][0], okays

def weighted_average_win_est(win_moms, win_counts, usable_wins):
        num_windows = len(win_moms)
        win_avg_mean = np.zeros(5)
        win_avg_cov = np.zeros((5,5))
        win_norm_fac = 0.0
        for i in range(num_windows):
            win_count = win_counts[i]
            if win_counts[i] > 1:
                win_okay = usable_wins[i]
                if win_okay:
                    norm_fac = win_count / num_windows
                    win_norm_fac += norm_fac
                    win_avg_mean += win_moms[i][-1][0] * norm_fac
                    win_avg_cov += win_moms[i][-1][1] * norm_fac
        win_avg_mean /= win_norm_fac
        win_avg_cov /= win_norm_fac
        return win_avg_mean, win_avg_cov

def edit_means(cauchyEsts, window_counts, state_idx, low, high):
    W = len(cauchyEsts)
    for i in range(W):
        if window_counts[i] > 1:
            xhat, _ = cauchyEsts[i].get_last_mean_cov()
            if (xhat[state_idx] < low) or (xhat[state_idx] > high):
                xhat[state_idx] = np.clip(xhat[state_idx], low, high)
                pyduc = cauchyEsts[i].get_pyduc()
                pyduc.cset_x(xhat)
                print("Window", i+1, "underwent mean editing!")

def corr(P):
    N = P.shape[0]
    C = np.eye(N,N)
    for i in range(N):
        for j in range(i+1,N):
            C[i,j] = P[i,j]/(P[i,i]**0.5 * P[j,j]**0.5)
    return C

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
            plt.subplot(511)
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
            plt.subplot(512)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,1], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,1], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,1], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 1], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 1], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 1], 'm')
            plt.subplot(513)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,2], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,2], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,2], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 2], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 2], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 2], 'm')
            plt.subplot(514)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,3], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,3], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,3], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 3], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 3], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 3], 'm')
            plt.subplot(515)
            plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,4], 'b')
            plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,4], 'r')
            plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,4], 'r')
            plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 4], 'g')
            plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 4], 'm')
            plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 4], 'm')
    plt.show()
    plt.close('all')

def test_python_debug_window_manager():
    global leo
    seed = 2124125479 #int(np.random.rand() * (2**32 -1))
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Leo Satelite Parameters
    leo5_alt = 200e3 # kmeters
    leo5_A = 64 # meters^2
    leo5_m = 5000 # kg
    leo5_gps_std_dev = 2.0 # meters
    leo5_dt = 60 # sec
    leo = leo_satellite_5state(leo5_alt, leo5_A, leo5_m, leo5_gps_std_dev, leo5_dt)

    # Log or Load Setting
    LOAD_RESULTS_AND_EXIT = True
    WITH_LOG = False
    assert(not (LOAD_RESULTS_AND_EXIT and WITH_LOG))

    # Cauchy and Kalman Tunables
    WITH_PLOT_ALL_WINDOWS = True
    WITH_SAS_DENSITY = True
    WITH_ADDED_DENSITY_JUMPS = True
    WITH_PLOT_MARG_DENSITY = False
    reinit_methods = ["speyer", "init_cond", "H2", "H2Boost", "H2Boost2", "H2_KF"]
    reinit_method = reinit_methods[4]
    prop_steps = 300 # Number of time steps to run sim
    num_windows = 8 # Number of Cauchy Windows
    ekf_scale = 10000 # Scaling factor for EKF atmospheric density
    gamma_scale = 1 # scaling gamma up by .... (1 is normal)
    beta_scale = 1 # scaling beta down by ... (1 is normal)
    time_tag = False

    alt_and_std = str(int(leo.r_sat/1000)) + "km" + "_A" + str(int(10*leo.A)) + "_m" + str(int(leo5_m)) + "_std" + str(int(10*leo.std_dev_gps)) + "_dt" + str(int(leo5_dt))
    ekf_scaled = "_ekfs" + str(ekf_scale)
    beta_scaled = "_bs" + str(beta_scale)
    gamma_scaled = "_gs" + str(gamma_scale)
    density_type = "_sas" if WITH_SAS_DENSITY else "_gauss"
    added_jumps = "_wj" if WITH_ADDED_DENSITY_JUMPS else "_nj"
    #time_id = str(time.time()) if time_tag else "" ### ADD SEEDING LOAD/LOG LOGIC!!

    # Log Files
    if WITH_LOG:
        log_dir = file_dir + "/pylog/leo5/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        log_dir += alt_and_std + "/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        log_dir += reinit_method + "/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        log_dir += "w" + str(num_windows) + density_type + added_jumps + ekf_scaled + beta_scaled + gamma_scaled + "/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        with open(log_dir + "seed.txt", "w") as handle:
            handle.write( "Seeded with: " + str(seed) )
    # Load Files
    if LOAD_RESULTS_AND_EXIT:
        log_dir = file_dir + "/pylog/leo5/"
        log_dir += alt_and_std + "/"
        log_dir += reinit_method + "/"
        log_dir += "w" + str(num_windows) + density_type + added_jumps + ekf_scaled + beta_scaled + gamma_scaled + "/"
    
    # Possibly only plot logged simulation results and exit
    if LOAD_RESULTS_AND_EXIT:
        scale = 1
        ce_moments = ce.load_cauchy_log_folder(log_dir, False)
        xs_kf, Ps_kf = ce.load_kalman_log_folder(log_dir)
        xs, zs, ws, vs = ce.load_sim_truth_log_folder(log_dir)
        weighted_ce_hist_path = log_dir + "weighted_ce.pickle"
        found_pickle = False
        if os.path.isfile(weighted_ce_hist_path):
            with open(weighted_ce_hist_path, "rb") as handle:
                found_pickle = True
                avg_ce_xhats, avg_ce_Phats, win_moms = pickle.load(handle)
                foo = np.zeros(avg_ce_xhats.shape[0])
                avgd_moment_info = {"x": avg_ce_xhats, "P": avg_ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
                print("All Window Cauchy Estimator History:")
                one_sigs_kf = np.array([ np.sqrt( np.diag(P_kf)) for P_kf in Ps_kf ])
                e_hats_kf = np.array([xt - xh for xt,xh in zip(xs,xs_kf) ])
                plot_all_windows(win_moms, xs, e_hats_kf, one_sigs_kf, 0, 1)
        print("Full Window History:")
        ce.plot_simulation_history(ce_moments, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True, scale=scale)
        if found_pickle:
            print("Weighted Cauchy Estimator History:")
            ce.plot_simulation_history(avgd_moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True, scale=scale)
        foobar=2
        exit(1)

    # Get Ground Truth and Measurements
    #zs = np.genfromtxt(file_dir + "/../../../log/leo5/dense/w5/msmts.txt", delimiter= ' ')
    #zs = zs[1:,:]
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=WITH_SAS_DENSITY, with_added_jumps=WITH_ADDED_DENSITY_JUMPS)
    zs_without_z0 = zs[1:,:]
    if WITH_LOG:
        ce.log_sim_truth(log_dir, xs, zs, ws, vs)

    # Run EKF 
    #'''
    W_kf = leo.W.copy()
    V_kf = leo.V.copy()
    # EKF NO SCALING
    #W_kf[0:4,0:4] *= 1000
    _xs_kf, _Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    # EKF WITH SCALING
    W_kf[4,4] *= ekf_scale
    xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    if WITH_LOG:
        ce.log_kalman(log_dir, xs_kf, Ps_kf)
    #ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf))
    #exit(1)
    #'''

    # Compute 1-sigma bounds for KF for Window Plot Compares  
    if WITH_PLOT_ALL_WINDOWS:
        one_sigs_kf = np.array([ np.sqrt( np.diag(P_kf)) for P_kf in Ps_kf ])
        e_hats_kf = np.array([xt - xh for xt,xh in zip(xs,xs_kf) ])

    # Run Cauchy Estimator
    #'''
    beta = np.array([leo.beta_cauchy])
    gamma = np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])
    beta /= beta_scale
    gamma *= gamma_scale

    # Create Phi.T as A0, start at propagated x0
    # Initialize Initial Hyperplanes
    Phi, _ = leo_5state_transition_model_jacobians(leo.x0)
    xbar = leo_5state_transition_model(leo.x0)
    A0 = Phi.T.copy() # np.linalg.eig(Ps_kf[35])[1].T #Phi.T.copy()
    p0 = np.repeat(leo.alpha_pv_cauchy, 5)
    p0[4] = leo.alpha_density_cauchy
    b0 = np.zeros(5)
    num_controls = 0

    total_steps = prop_steps
    ce.set_tr_search_idxs_ordering([3,2,4,1,0])
    debug_print = False

    win_idxs = np.arange(num_windows)
    win_counts = np.zeros(num_windows, dtype=np.int64)
    cauchyEsts = [ce.PyCauchyEstimator("nonlin", num_windows, debug_print) for _ in range(num_windows)]#ce.PySlidingWindowManager("nonlin", num_windows, total_steps, log_dir=log_dir, log_seq=True, log_full=True)
    for i in range(num_windows):
        cauchyEsts[i].initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
        cauchyEsts[i].set_window_number(i)
    win_moms = { i : [] for i in range(num_windows) }
    win_moms[0].append( cauchyEsts[0].step(zs_without_z0[0], None, False) )
    win_counts[0] = 1
    N = zs_without_z0.shape[0]

    ce_xhats = [win_moms[0][-1][0].copy()]
    ce_Phats = [win_moms[0][-1][1].copy()]

    avg_ce_xhats = [win_moms[0][-1][0].copy()]
    avg_ce_Phats = [win_moms[0][-1][1].copy()]
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    #win_marginals = { i : [] for i in range(num_windows) }
    for k in range(1, N):
        print("---- Step {}/{} -----".format(k+2, N+1))
        zk = zs_without_z0[k] 
        # find max and min indices
        idx_max = np.argmax(win_counts)
        idx_min = np.argmin(win_counts)
        # Step all windows that are not uninitialized
        for win_idx, win_count in zip(win_idxs, win_counts):
            if(win_count > 0):
                print("  Window {} is on step {}/{}".format(win_idx+1, win_count+1, num_windows) )
                win_moms[win_idx].append( cauchyEsts[win_idx].step(zk, None, False) )
                print("    x_k|k:   ", win_moms[win_idx][-1][0] )
                print("    e_k|k:   ", xs[k+1] - win_moms[win_idx][-1][0] )
                win_counts[win_idx] += 1
        
        best_idx, usable_wins = best_window_est(cauchyEsts, win_counts)
        xhat, Phat = cauchyEsts[best_idx].get_last_mean_cov()
        ce_xhats.append(xhat.copy())
        ce_Phats.append(Phat.copy())
        print("Best Window Index For Reinit is: Window ", best_idx+1)
        
        # Mean edit for safety of values
        #if leo5_alt > 300e3:
        #edit_means(cauchyEsts, win_counts, 4, -.05, 0.05)
        #else:
        edit_means(cauchyEsts, win_counts, 4, -.85, 10)
        

        # Compute Weighted Average Window Estimate
        avg_xhat, avg_Phat = weighted_average_win_est(win_moms, win_counts, usable_wins)
        avg_ce_xhats.append(avg_xhat)
        avg_ce_Phats.append(avg_Phat)

        # Reinitialize empty estimator
        if(reinit_method == "speyer"):
            # using speyer's start method
            speyer_restart_idx = 1
            xreset, Preset = cauchyEsts[idx_min].reset_about_estimator(cauchyEsts[best_idx], msmt_idx = speyer_restart_idx)
            print("  Window {} is on step {}/{} and has mean:\n  {}".format(idx_min+1, win_counts[idx_min]+1, num_windows, np.around(xreset,4)) )
        elif(reinit_method == "init_cond"):
            _A0 = cauchyEsts[best_idx]._Phi.copy().reshape((5,5)).T # np.eye(5)
            _p0 = p0.copy() #np.sqrt(np.diag(Ps_kf[k+1]))
            win_moms[idx_min].append( cauchyEsts[idx_min].reset_with_last_measurement(zk[1], _A0, _p0, b0, xhat) )
        elif("H2" in reinit_method):
            # Both H channels concatenated
            _H = np.array([1.0, 1.0, 0, 0, 0])
            _gamma = 2 * gamma[0]
            _xbar = cauchyEsts[best_idx]._xbar[5:]
            _dz = zk[0] + zk[1] - _xbar[0] - _xbar[1]
            _dx = xhat - _xbar
            
            # Covariance Selection
            if("KF" in reinit_method):
                _P = _Ps_kf[k+1].copy() # KF COVAR DOUBLES LOOKS GOOD
            else:
                _P = Phat.copy() # CAUCHY COVAR LOOKS GOOD

            if("Boost" in reinit_method):
                # Boost
                _pos_scale = np.ones(4)
                _P_kf = _Ps_kf[k+1].copy()
                _P_cauchy = Phat
                for i in range(4):
                    if( (_P_kf[i,i] / _P_cauchy[i,i]) > 1):
                        _pos_scale[i] = (_P_kf[i,i] / _P_cauchy[i,i]) * 1.3898
                        _P[i,i] *= _pos_scale[i]
                if "Boost2" in reinit_method:
                    _P *= 2
            # Reset
            _A0, _p0, _b0 = ce.speyers_window_init(_dx, _P, _H, _gamma, _dz)
            global INITIAL_H
            INITIAL_H = True
            win_moms[idx_min].append( cauchyEsts[idx_min].reset_with_last_measurement(zk[0] + zk[1], _A0, _p0, _b0, _xbar) )
            pyduc = cauchyEsts[idx_min].get_pyduc()
            pyduc.cset_gamma(gamma)
            INITIAL_H = False
            foobar=2
        else:
            print("Reinitialization Scheme ", reinit_method, "Not Implemented! Please Add! Exiting!")
            exit(1)
        # Increment Initialized Estimator Count
        win_counts[idx_min] += 1

        # Now plot all windows 
        if WITH_PLOT_ALL_WINDOWS and (k==(N-1)):
            plot_all_windows(win_moms, xs, e_hats_kf, one_sigs_kf, best_idx, idx_min)

        # Density Marginals
        if WITH_PLOT_MARG_DENSITY:
            if k > 18:
                plt.figure(figsize = (8,12))
                print("  Window Counts at Step {} are:\n  {}\n  Marginal 1D CPDFs of Atmospheric Density are:".format(k+2, win_counts) )
                print("---------------------------------------------------")
                #y_avg = np.zeros(10001)
                #weight_avg = 0.0
                x_true = xs[k+1]
                top = min(k+1, num_windows)
                for win_idx in range(top):
                    win_xhat, _ = cauchyEsts[win_idx].get_last_mean_cov()
                    wgl = -1 - win_xhat[4]
                    wgh = 9 - win_xhat[4]
                    wx, wy = cauchyEsts[win_idx].get_marginal_1D_pointwise_cpdf(4, wgl, wgh, 0.001)
                    plt.subplot(top, 1, win_idx+1)
                    plt.plot(win_xhat[4] + wx, wy, 'b')
                    plt.scatter(x_true[4], 0, color='r', marker = 'x')
                    plt.scatter(win_xhat[4], 0, color='b', marker = 'x')
                    plt.ylabel("Win"+str(win_idx+1))
                    if win_idx == 0:
                        plt.title("Densities at Step {}/{}".format(k+2,N+1))
                    #weight_avg += (win_counts[win_idx] / num_windows)
                    #y_avg += wy[:10001] * (win_counts[win_idx] / num_windows)
                plt.xlabel("Change in Atms. Density State")
                print("---------------------------------------------------")
                #y_avg /= weight_avg
                #weights = y_avg / np.sum(y_avg)
                #plt.figure()
                #plt.plot(win_xhat[4] + wx, y_avg)
                #plt.scatter(x_true[4], 0, color='r', marker = 'x')
                #plt.scatter(np.sum(weights * (win_xhat[4] + wx) ), 0, color='b', marker = 'x')
                plt.show()
                foobar = 2

        # reset full estimator
        if(win_counts[idx_max] == num_windows):
            cauchyEsts[idx_max].reset()
            win_counts[idx_max] = 0
            
    ce_xhats = np.array(ce_xhats)
    ce_Phats = np.array(ce_Phats)
    avg_ce_xhats = np.array(avg_ce_xhats)
    avg_ce_Phats = np.array(avg_ce_Phats)

    foo = np.zeros(ce_xhats.shape[0])
    moment_info = {"x": ce_xhats, "P": ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    avg_moment_info = {"x": avg_ce_xhats, "P": avg_ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }

    if WITH_LOG:
        ce.log_cauchy(log_dir, moment_info)
        with open(log_dir + "weighted_ce.pickle", "wb") as handle:
            pickle.dump((avg_ce_xhats, avg_ce_Phats, win_moms), handle)
    
    print("Full Window History:")
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True)
    print("Weighted Cauchy Estimator History:")
    ce.plot_simulation_history(avg_moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True)
    foobar = 2

def load_and_rerun_kf():
    global leo
    seed = 2124125479 #int(np.random.rand() * (2**32 -1))
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Leo Satelite Parameters
    leo5_alt = 200e3 # kmeters
    leo5_A = 64 # meters^2
    leo5_m = 5000 # kg
    leo5_gps_std_dev = 2.0 # meters
    leo5_dt = 60 # sec
    leo = leo_satellite_5state(leo5_alt, leo5_A, leo5_m, leo5_gps_std_dev, leo5_dt)

    # Log or Load Setting
    LOAD_RESULTS_AND_EXIT = True
    #WITH_LOG = False
    #assert(not (LOAD_RESULTS_AND_EXIT and WITH_LOG))

    # Cauchy and Kalman Tunables
    WITH_PLOT_ALL_WINDOWS = True
    WITH_SAS_DENSITY = True
    WITH_ADDED_DENSITY_JUMPS = True
    WITH_PLOT_MARG_DENSITY = False
    reinit_methods = ["speyer", "init_cond", "H2", "H2Boost", "H2Boost2", "H2_KF"]
    reinit_method = reinit_methods[4]
    prop_steps = 300 # Number of time steps to run sim
    num_windows = 8 # Number of Cauchy Windows
    ekf_scale = 1 # Scaling factor for EKF atmospheric density
    gamma_scale = 1 # scaling gamma up by .... (1 is normal)
    beta_scale = 1 # scaling beta down by ... (1 is normal)
    time_tag = False

    alt_and_std = str(int(leo.r_sat/1000)) + "km" + "_A" + str(int(10*leo.A)) + "_m" + str(int(leo5_m)) + "_std" + str(int(10*leo.std_dev_gps)) + "_dt" + str(int(leo5_dt))
    ekf_scaled = "_ekfs" + str(ekf_scale)
    beta_scaled = "_bs" + str(beta_scale)
    gamma_scaled = "_gs" + str(gamma_scale)
    density_type = "_sas" if WITH_SAS_DENSITY else "_gauss"
    added_jumps = "_wj" if WITH_ADDED_DENSITY_JUMPS else "_nj"
    #time_id = str(time.time()) if time_tag else "" ### ADD SEEDING LOAD/LOG LOGIC!!

    # Log Files
    '''
    if WITH_LOG:
        log_dir = file_dir + "/pylog/leo5/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        log_dir += alt_and_std + "/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        log_dir += reinit_method + "/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        log_dir += "w" + str(num_windows) + density_type + added_jumps + ekf_scaled + beta_scaled + gamma_scaled + "/"
        if( not os.path.isdir(log_dir)):
            os.mkdir(log_dir)
        with open(log_dir + "seed.txt", "w") as handle:
            handle.write( "Seeded with: " + str(seed) )
    '''

    # Load Files
    if LOAD_RESULTS_AND_EXIT:
        log_dir = file_dir + "/pylog/leo5/"
        log_dir += alt_and_std + "/"
        log_dir += reinit_method + "/"
        log_dir += "w" + str(num_windows) + density_type + added_jumps + ekf_scaled + beta_scaled + gamma_scaled + "/"
    
    # Possibly only plot logged simulation results and exit
    if LOAD_RESULTS_AND_EXIT:
        scale = 1
        ce_moments = ce.load_cauchy_log_folder(log_dir, False)
        xs_kf, Ps_kf = ce.load_kalman_log_folder(log_dir)
        xs, zs, ws, vs = ce.load_sim_truth_log_folder(log_dir)
        weighted_ce_hist_path = log_dir + "weighted_ce.pickle"
        found_pickle = False
        if os.path.isfile(weighted_ce_hist_path):
            with open(weighted_ce_hist_path, "rb") as handle:
                found_pickle = True
                avg_ce_xhats, avg_ce_Phats, win_moms = pickle.load(handle)
                foo = np.zeros(avg_ce_xhats.shape[0])
                avgd_moment_info = {"x": avg_ce_xhats, "P": avg_ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
                print("All Window Cauchy Estimator History:")
                one_sigs_kf = np.array([ np.sqrt( np.diag(P_kf)) for P_kf in Ps_kf ])
                e_hats_kf = np.array([xt - xh for xt,xh in zip(xs,xs_kf) ])
                #plot_all_windows(win_moms, xs, e_hats_kf, one_sigs_kf, 0, 1)
        print("Full Window History:")
        #ce.plot_simulation_history(ce_moments, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True, scale=scale)
        if found_pickle:
            print("Weighted Cauchy Estimator History:")
            #ce.plot_simulation_history(avgd_moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True, scale=scale)
        foobar=2
        #exit(1)

    # Get Ground Truth and Measurements
    #zs = np.genfromtxt(file_dir + "/../../../log/leo5/dense/w5/msmts.txt", delimiter= ' ')
    #zs = zs[1:,:]
    #xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=WITH_SAS_DENSITY, with_added_jumps=WITH_ADDED_DENSITY_JUMPS)
    #zs_without_z0 = zs[1:,:]
    #if WITH_LOG:
    #    ce.log_sim_truth(log_dir, xs, zs, ws, vs)

    # Run EKF 
    #'''
    W_kf = leo.W.copy()
    V_kf = leo.V.copy()

    # EKF WITH SCALING
    W_kf[4,4] *= 10000
    xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs[1:], ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    #if WITH_LOG:
    #    ce.log_kalman(log_dir, xs_kf, Ps_kf)
    #ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf))
    #exit(1)
    #'''

    print("Weighted Cauchy Estimator History:")
    ce.plot_simulation_history(avgd_moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True)
    plt.figure()
    Ts = np.arange(xs.shape[0])
    plt.plot(Ts, xs[:,4], 'r')
    plt.plot(Ts, xs_kf[:,4], 'g')
    plt.plot(Ts[1:], avgd_moment_info['x'][:, 4], 'b')
    plt.show()
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
    #test_single_sliding_window()
    #test_python_debug_window_manager()
    load_and_rerun_kf()
