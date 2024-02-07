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

    def __init__(self, leo5_alt = 200e3, leo5_A = 64.0, leo5_gps_std_dev = 2.0):
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
        self.m = 5000.0 # kg
        self.rho = lookup_air_density(self.r_sat) # kg/m^3
        self.C_D = 2.0 #drag coefficient
        self.A = leo5_A #m^2
        self.tau = 21600.0 # 1/(m*sec)
        # Parameters for runge kutta ODE integrator
        self.dt = 60.0 #time step in sec
        self.sub_steps_per_dt = int(self.dt) # so sub intervals are dt / sub_steps_dt 
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

def leo4_ode(x):
    global leo 
    r = np.sqrt(x[0]*x[0] + x[1]*x[1])
    v = np.sqrt(x[2]*x[2] + x[3]*x[3])
    dx_dt = np.zeros(4)
    dx_dt[0] = x[2] 
    dx_dt[1] = x[3]
    dx_dt[2] = -(leo.mu)/pow(r,3) * x[0] - 0.5*leo.A*leo.C_D/leo.m*leo.rho*v*x[2]
    dx_dt[3] = -(leo.mu)/pow(r,3) * x[1] - 0.5*leo.A*leo.C_D/leo.m*leo.rho*v*x[3]
    return dx_dt

def leo_5state_transition_model(x):
    global leo 
    x_new = x.copy()
    dt_sub = leo.dt / leo.sub_steps_per_dt
    for _ in range(leo.sub_steps_per_dt):
        x_new = ce.runge_kutta4(leo5_ode, x_new, dt_sub)
    return x_new 

def leo_4state_transition_model(x):
    global leo 
    x_new = x.copy()
    dt_sub = leo.dt / leo.sub_steps_per_dt
    for _ in range(leo.sub_steps_per_dt):
        x_new = ce.runge_kutta4(leo4_ode, x_new, dt_sub)
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

def leo_4state_transition_model_jacobians(x):
    Jac = ce.cd4_gvf(x, leo4_ode) # Jacobian matrix
    taylor_order = 3
    Phi_k = np.zeros((x.size,x.size))
    for i in range(taylor_order+1):
        Phi_k += np.linalg.matrix_power(Jac, i) * leo.dt**i / math.factorial(i)
    Gamma_k = np.zeros((x.size,1))
    Gamma_c = np.zeros((x.size,1)) # continous time Gamma 
    Gamma_c[2,0] = 1.0
    Gamma_c[3,0] = 1.0
    for i in range(taylor_order+1):
        Gamma_k += ( np.linalg.matrix_power(Jac, i) * leo.dt**(i+1) / math.factorial(i+1) ) @ Gamma_c
    return Phi_k, Gamma_k

# 'gps'
def leo_5state_measurement_model_jacobian(x):
    H = np.zeros((2,5))
    H[0,0] = 1.0
    H[1,1] = 1.0
    return H

def leo_4state_measurement_model_jacobian(x):
    H = np.zeros((2,4))
    H[0,0] = 1.0
    H[1,1] = 1.0
    return H

# 'gps'
def leo_5state_measurement_model(x):
    global leo
    return x[:2].copy()

def leo_4state_measurement_model(x):
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
    xbar = leo_4state_transition_model(x) 
    print("    x_k-1:   ", x)
    Phi, Gamma = leo_4state_transition_model_jacobians(x)
    pyduc.cset_Phi(Phi)
    pyduc.cset_Gamma(Gamma)
    # Propagate and set x
    pyduc.cset_x(xbar)
    pyduc.cset_is_xbar_set_for_ece()

def ece_nonlinear_msmt_model(c_duc, c_zbar):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    
    global INITIAL_H
    if(INITIAL_H):
        zbar = np.array([0, xbar[0] + xbar[1]])
    else:
        zbar = leo_4state_measurement_model(xbar)
    pyduc.cset_zbar(c_zbar, zbar)

def ece_extended_msmt_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    global INITIAL_H
    if INITIAL_H:
        H = np.zeros((2,4))
        H[0,0] = 0
        H[1,0] = 1
        H[1,1] = 1
        global leo 
        gam = leo.std_dev_gps * leo.GAUSS_TO_CAUCHY
        gamma = np.array([gam, 2*gam])
        pyduc.cset_gamma(gamma)
    else:
        H = leo_4state_measurement_model_jacobian(xbar)
    print("    x_k|k-1: ", xbar)
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
            if i == 70:
                wk[4] = -0.0
            if i == 200:
                wk[4] = -0.0
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

# Sliding Window Debugger
def test_single_sliding_window():
    global leo
    # 2124125479 -- no huge jumps
    seed = 2124125479 #int(np.random.rand() * (2**32 -1)) #3872826552#
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Leo Satelite Parameters
    leo5_alt = 550e3 # meters
    leo5_A = 64 # meters^2
    leo5_gps_std_dev = 2.0 # meters
    leo = leo_satellite_5state(leo5_alt, leo5_A, leo5_gps_std_dev)

    # Cauchy and Kalman Tunables
    num_window_steps = 5
    prop_steps = num_window_steps # Number of time steps to run sim
    gamma_scale = 1 # scaling gamma up by .... (1 is normal)
    beta_scale = 1 # scaling beta down by ... (1 is normal)

    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=False, with_added_jumps=False)
    zs_without_z0 = zs[1:,:]

    # Run Cauchy Estimator
    #'''
    beta = np.array([leo.beta_cauchy/1000, leo.beta_cauchy/1000])
    gamma = np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])
    #beta /= beta_scale
    #gamma /= gamma_scale

    # Create Phi.T as A0, start at propagated x0
    # Initialize Initial Hyperplanes
    Phi, _ = leo_4state_transition_model_jacobians(leo.x0[0:4])
    xbar = leo_4state_transition_model(leo.x0[0:4])
    A0 = Phi.T.copy() # np.linalg.eig(Ps_kf[35])[1].T #Phi.T.copy()
    p0 = np.repeat(leo.alpha_pv_cauchy, 4)
    b0 = np.zeros(4)
    num_controls = 0

    ce.set_tr_search_idxs_ordering([2,3,1,0])
    debug_print = True
    cauchyEst = ce.PyCauchyEstimator("nonlin", num_window_steps, debug_print) 
    cauchyEst.initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    for i in range(num_window_steps):
        xhat, Phat = cauchyEst.step(zs_without_z0[i], None)

# choose window with last estimate's covariance defined
def best_window_est(cauchyEsts, window_counts):
    W = len(cauchyEsts)
    idxs = []
    for i in range(W):
        if(window_counts[i] > 0):
            err = cauchyEsts[i]._err_code
            if (err[1] & (1<<1)) or (err[1] & (1<<3)):
                pass
            else:
                idxs.append((i, window_counts[i]))
    if(len(idxs) == 0):
        print("No window is available without an error code!")
        exit(1)
    sorted_idxs = list(reversed(sorted(idxs, key = lambda x : x[1])))
    return sorted_idxs[0][0]

def corr(P):
    N = P.shape[0]
    C = np.eye(N,N)
    for i in range(N):
        for j in range(i+1,N):
            C[i,j] = P[i,j]/(P[i,i]**0.5 * P[j,j]**0.5)
    return C

def obsv(Phi, H):
    assert(H.ndim == 2)
    return np.vstack([H @ np.linalg.matrix_power(Phi, i) for i in range(Phi.shape[0])])

def obsv_Gram_nls(JacA, H, dt, order):
    assert(H.ndim == 2)
    OGram = np.zeros_like(JacA)
    for i in range(order+1):
        for j in range(order+1):
            tmp_i = np.linalg.matrix_power(JacA, i) / math.factorial(i)
            tmp_j = np.linalg.matrix_power(JacA, j) / math.factorial(j)
            Tk_coef = dt**(i+j+1) / (i+j+1)
            OGram += tmp_i.T @ H.T @ H @ tmp_j * Tk_coef
    return OGram

def test_python_debug_window_manager():
    global leo
    # 2124125479 -- no huge jumps
    seed = 2124125479 #int(np.random.rand() * (2**32 -1)) #3872826552#
    print("Seeding with seed: ", seed)
    np.random.seed(seed)

    # Leo Satelite Parameters
    leo5_alt = 550e3 # meters
    leo5_A = 14 # meters^2
    leo5_gps_std_dev = 7.5 # meters
    leo = leo_satellite_5state(leo5_alt, leo5_A, leo5_gps_std_dev)

    # Cauchy and Kalman Tunables
    WITH_LOG = False
    WITH_PLOT_DENSITY = False
    prop_steps = 15 # Number of time steps to run sim
    num_windows = 5 # Number of Cauchy Windows
    ekf_scale = 10000 # Scaling factor for EKF atmospheric density
    gamma_scale = 1 # scaling gamma up by .... (1 is normal)
    beta_scale = 1 # scaling beta down by ... (1 is normal)
    use_speyer_restart = True
    speyer_restart_idx = 1
    with_jumps = False

    alt = str(int(leo.r_sat/1000)) + "km_"
    win_init_prob = "wip_"  if use_speyer_restart else "nwip_"
    ekf_scaled = "ekfs" + str(ekf_scale)
    beta_scaled = "_bs" + str(beta_scale)
    gamma_scaled = "_gs" + str(gamma_scale)
    added_jumps = "wj_" if with_jumps else "nj_"

    # Directory name
    if (leo.A == 64) and (leo.std_dev_gps == 2):
        dir_base_name = "leo4/debug"
    elif (leo.A == 14) and (leo.std_dev_gps == 7.5):
        dir_base_name = "leo4/fermi"
    else:
        if(WITH_LOG):
            print("Please setup directory for new parameters leo.A={} and leo.std_dev_gps={}".format(leo.A, leo.std_dev_gps) )
            exit(1)
        else:
            dir_base_name = "foobar"
    # Forming log directory
    if (beta_scale == 1) and (gamma_scale == 1):
        log_dir = file_dir + "/pylog/" + dir_base_name  + str(num_windows) + "/" + alt + added_jumps + win_init_prob + ekf_scaled
    else:
        log_dir = file_dir + "/pylog/" + dir_base_name + str(num_windows) + "_ce_scaled" + "/" + alt + added_jumps + win_init_prob + ekf_scaled + beta_scaled + gamma_scaled

    # Possibly only plot logged simulation results and exit
    plot_debug = False
    if plot_debug:
        ce_moments = ce.load_cauchy_log_folder(log_dir, False)
        xs_kf, Ps_kf = ce.load_kalman_log_folder(log_dir)
        xs, zs, ws, vs = ce.load_sim_truth_log_folder(log_dir)
        scale = 1
        ce.plot_simulation_history(ce_moments, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True, scale=scale)
        foobar=2
        exit(1)

    # Get Ground Truth and Measurements
    #zs = np.genfromtxt(file_dir + "/../../../log/leo5/dense/w5/msmts.txt", delimiter= ' ')
    #zs = zs[1:,:]
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=False, with_added_jumps=with_jumps)
    zs_without_z0 = zs[1:,:]
    if WITH_LOG:
        ce.log_sim_truth(log_dir, xs, zs, ws, vs)

    # Run EKF 
    #'''
    W_kf = leo.W.copy()
    V_kf = leo.V.copy()
    W_kf[4,4] *= ekf_scale
    xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    #xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    if WITH_LOG:
        ce.log_kalman(log_dir, xs_kf, Ps_kf)
    #ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf))
    #exit(1)
    #'''

    # Run Cauchy Estimator
    #'''
    beta = np.array([leo.beta_cauchy/10, leo.beta_cauchy/10])
    gamma = np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])
    beta /= beta_scale
    gamma *= gamma_scale

    # Create Phi.T as A0, start at propagated x0
    # Initialize Initial Hyperplanes
    Phi, _ = leo_4state_transition_model_jacobians(leo.x0[0:4])
    xbar = leo_4state_transition_model(leo.x0[0:4])
    A0 = np.eye(4) #Phi.T.copy() # np.linalg.eig(Ps_kf[35])[1].T #Phi.T.copy()
    p0 = np.repeat(leo.alpha_pv_cauchy, 4)
    b0 = np.zeros(4)
    num_controls = 0

    # Grammian / Obsv computation
    _H = np.zeros((1,4))
    _H[0,0] = 1
    _H[0,1] = 1
    _JacA = ce.cd4_gvf(leo.x0[0:4], leo4_ode)
    _O = obsv(Phi, _H)
    _Gram = obsv_Gram_nls(_JacA, _H, leo.dt, order=2)

    total_steps = prop_steps
    ce.set_tr_search_idxs_ordering([3,2,1,0])
    debug_print = False

    win_idxs = np.arange(num_windows)
    win_counts = np.zeros(num_windows, dtype=np.int64)
    cauchyEsts = [ce.PyCauchyEstimator("nonlin", num_windows, debug_print) for _ in range(num_windows)]#ce.PySlidingWindowManager("nonlin", num_windows, total_steps, log_dir=log_dir, log_seq=True, log_full=True)
    for i in range(num_windows):
        cauchyEsts[i].initialize_nonlin(xbar, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls)
        cauchyEsts[i].set_window_number(i)
    xhat, Phat = cauchyEsts[0].step(zs_without_z0[0], None)
    win_counts[0] = 1
    N = zs_without_z0.shape[0]

    ce_xhats = [xhat.copy()]
    ce_Phats = [Phat.copy()]
    
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
                win_xhat, win_Phat = cauchyEsts[win_idx].step(zk, None)
                print("  Window {} is on step {}/{} and has mean:\n  {}".format(win_idx+1, win_count+1, num_windows, np.around(win_xhat,4)) )
                win_counts[win_idx] += 1
        
        best_idx = best_window_est(cauchyEsts, win_counts)
        xhat, Phat = cauchyEsts[best_idx].get_last_mean_cov()
        ce_xhats.append(xhat.copy())
        ce_Phats.append(Phat.copy())
        print("Best Window Index For Reinit is: Window ", best_idx+1)
        # Reinitialize empty estimator
        if(use_speyer_restart):
            # using speyer's start method
            #cauchyEsts[best_idx]._P[16:] = Ps_kf[k][:4,:4].copy().reshape(-1)
            xreset, Preset = cauchyEsts[idx_min].reset_about_estimator(cauchyEsts[best_idx], msmt_idx = speyer_restart_idx)
            print("  Window {} is on step {}/{} and has mean:\n  {}".format(idx_min+1, win_counts[idx_min]+1, num_windows, np.around(xreset,4)) )
        else:
            # Possibly reinitialize about start params
            #PKF_K = _Ps_kf[k+1]
            #EVECS_KF_T = np.linalg.eig(_Ps_kf[k+1])[1].T
            #STD_KF = np.sqrt(np.diag(PKF_K)) * ce.GAUSSIAN_TO_CAUCHY_NOISE
            #cauchyEsts[idx_min].reset(EVECS_KF_T, STD_KF, b0, xhat)
            #cauchyEsts[idx_min].step(zk)
            #scalar_msmt_idx = 1
            #cauchyEsts[idx_min].step_scalar_msmt(zk[scalar_msmt_idx], scalar_msmt_idx, True)
            
            _A0 = np.eye(4) #cauchyEsts[best_idx]._Phi.copy().reshape((4,4)).T # np.eye(5)
            _p0 = p0.copy() #np.sqrt(np.diag(Ps_kf[k+1]))
            cauchyEsts[idx_min].reset_with_last_measurement(zk[1], _A0, _p0, b0, xhat)

            # Modify restart slightly
            #_A0, _p0, _b0, _xbar = cauchyEsts[idx_max].get_reinitialization_statistics()
            ##_p0[[0,2]] = _p0[[1,3]]*8
            #cauchyEsts[idx_min].reset_with_single_measurement(zk[1], _A0, _p0, _b0, _xbar)

        print("Correlation of Cauchy")
        print( np.around(corr(Phat),4) )
        print("Covariance of Cauchy")
        print( np.around(Phat,4) )
        print("Correlation of EKF")
        print( np.around(corr(Ps_kf[k+1]),4) )
        print("Covariance of EKF")
        print( np.around(Ps_kf[k+1],4) )
        print("Phi of Cauchy is ")
        print(cauchyEsts[idx_max]._Phi.reshape((4,4)))

        win_counts[idx_min] += 1

        # Density Marginals
        if WITH_PLOT_DENSITY:
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
    foo = np.zeros(ce_xhats.shape[0])
    moment_info = {"x": ce_xhats, "P": ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    if WITH_LOG:
        ce.log_cauchy(log_dir, moment_info)

    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True)
    foobar = 2

def test_leo4_windows():
    global leo
    # 2124125479 -- no huge jumps
    #seed = 2124125479 #int(np.random.rand() * (2**32 -1)) #3872826552#
    #print("Seeding with seed: ", seed)
    #np.random.seed(seed)

    # Leo Satelite Parameters
    leo5_alt = 550e3 # meters
    leo5_A = 14 # meters^2
    leo5_gps_std_dev = 7.5 # meters
    leo = leo_satellite_5state(leo5_alt, leo5_A, leo5_gps_std_dev)

    # Get Ground Truth and Measurements
    prop_steps = 100
    USE_SPEYER_RESTART = False
    PLOT_WINDOW = True
    WITH_SAS_DENSITY=True
    WITH_ADDED_JUMPS=False
    speyer_restart_idx = 1
    beta_scale = 10
    xs, zs, ws, vs = simulate_leo5_state(prop_steps, with_sas_density=WITH_SAS_DENSITY, with_added_jumps=WITH_ADDED_JUMPS)
    zs_without_z0 = zs[1:,:]

    # Run EKF 
    W_kf = leo.W.copy()
    V_kf = leo.V.copy()
    W_kf[4,4] *= 1000
    xs_kf, Ps_kf = gf.run_extended_kalman_filter(leo.x0, None, zs_without_z0, ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, leo.P0, W_kf, V_kf)
    
    one_sigs_kf = np.array([ np.sqrt( np.diag(P_kf)) for P_kf in Ps_kf ])
    e_hats_kf = np.array([xt - xh for xt,xh in zip(xs,xs_kf) ])
    Ts_kf = np.arange(one_sigs_kf.shape[0])
    #ce.plot_simulation_history(None, (xs, zs, ws, vs), (xs_kf, Ps_kf))

    # Run Cauchy Estimator
    #'''
    num_windows = 7
    #beta = np.array([4*leo.beta_cauchy, 4*leo.beta_cauchy])
    beta = np.array([leo.beta_cauchy / beta_scale])
    gamma = np.array([leo.std_dev_gps * leo.GAUSS_TO_CAUCHY, leo.std_dev_gps * leo.GAUSS_TO_CAUCHY])

    # Create Phi.T as A0, start at propagated x0
    # Initialize Initial Hyperplanes
    xbar = leo_4state_transition_model(leo.x0[0:4])
    A0 = leo_4state_transition_model_jacobians(leo.x0[0:4])[0].T.copy() # np.linalg.eig(Ps_kf[35])[1].T #Phi.T.copy()
    p0 = np.repeat(leo.alpha_pv_cauchy, 4)
    b0 = np.zeros(4)
    num_controls = 0

    total_steps = prop_steps
    ce.set_tr_search_idxs_ordering([3,2,1,0])
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

    ce_xhats = [win_moms[0][-1][0].copy()]
    ce_Phats = [win_moms[0][-1][1].copy()]

    N = zs_without_z0.shape[0]
    np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
    for k in range(1, N):
        print("---- Step {}/{} -----".format(k+1, N))
        print("  xtrue:     ", xs[k+1])
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
                print("    e_k|k:   ", xs[k+1][:-1] - win_moms[win_idx][-1][0] )
                win_counts[win_idx] += 1
        
        best_idx = best_window_est(cauchyEsts, win_counts)
        xhat, Phat = cauchyEsts[best_idx].get_last_mean_cov()
        print("Best Window Index For Reinit is: Window ", best_idx+1)
        ce_xhats.append(win_moms[best_idx][-1][0].copy())
        ce_Phats.append(win_moms[best_idx][-1][1].copy())
        # Reinitialize empty estimator
        if USE_SPEYER_RESTART:
            win_moms[idx_min].append( cauchyEsts[idx_min].reset_about_estimator(cauchyEsts[best_idx], msmt_idx = speyer_restart_idx) )
            print("  Window {} is on step {}/{} and has mean:\n  {}".format(idx_min+1, win_counts[idx_min]+1, num_windows, np.around(win_moms[idx_min][-1][0],4)) )
        else:
            #_A0 = cauchyEsts[best_idx]._Phi.copy().reshape((4,4)).T # np.eye(5)
            #_p0 = p0.copy() #np.sqrt(np.diag(Ps_kf[k+1]))
            #win_moms[idx_min].append( cauchyEsts[idx_min].reset_with_last_measurement(zk[1], _A0, _p0, b0, xhat) )
            #'''
            _H = np.array([1.0, 1.0, 0, 0])
            _gamma = 2 * gamma[0]
            _xbar = cauchyEsts[best_idx]._xbar[4:]
            _dz = zk[0] + zk[1] - _xbar[0] - _xbar[1]
            _dx = xhat - _xbar
            _P = Phat.copy()
            _pos_scale = np.ones(2)
            if(Ps_kf[k+1][0,0] / Phat[0,0] > 1):
                _pos_scale[0] = Ps_kf[k+1][0,0] / Phat[0,0] * 1.3898
            if(Ps_kf[k+1][1,1] / Phat[1,1] > 1):
                _pos_scale[1] = Ps_kf[k+1][1,1] / Phat[1,1] * 1.3898
            _P[0,0] *= _pos_scale[0]
            _P[1,1] *= _pos_scale[1]
            #_P[0:2,0:2] += np.eye(2) * 2
            _P *= 2
            _A0, _p0, _b0 = ce.speyers_window_init(_dx, _P, _H, _gamma, _dz)
            global INITIAL_H
            INITIAL_H = True
            win_moms[idx_min].append( cauchyEsts[idx_min].reset_with_last_measurement(zk[0] + zk[1], _A0, _p0, _b0, _xbar) )
            pyduc = cauchyEsts[idx_min].get_pyduc()
            pyduc.cset_gamma(gamma)
            INITIAL_H = False
            foobar=2
            #'''
        win_counts[idx_min] += 1


        # Now plot all windows 
        if PLOT_WINDOW and k==(N-1):
            for win_idx in win_idxs:
                if k > win_idx:
                    x_hats = np.array([ win_moms[win_idx][i][0] for i in range(len(win_moms[win_idx])) ])
                    P_hats = np.array([ win_moms[win_idx][i][1] for i in range(len(win_moms[win_idx])) ])
                    T_cur = win_idx + x_hats.shape[0] + 1
                    one_sigs = np.array([np.sqrt(np.diag(P_hat)) for P_hat in P_hats])
                    e_hats = np.array([xt[:-1] - xh for xt,xh in zip(xs[win_idx+1:T_cur], x_hats)])
                    
                    plt.figure()
                    plt.subplot(411)
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
                    plt.subplot(412)
                    plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,1], 'b')
                    plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,1], 'r')
                    plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,1], 'r')
                    plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 1], 'g')
                    plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 1], 'm')
                    plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 1], 'm')
                    plt.subplot(413)
                    plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,2], 'b')
                    plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,2], 'r')
                    plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,2], 'r')
                    plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 2], 'g')
                    plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 2], 'm')
                    plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 2], 'm')
                    plt.subplot(414)
                    plt.plot(Ts_kf[win_idx+1:T_cur], e_hats[:,3], 'b')
                    plt.plot(Ts_kf[win_idx+1:T_cur], one_sigs[:,3], 'r')
                    plt.plot(Ts_kf[win_idx+1:T_cur], -one_sigs[:,3], 'r')
                    plt.plot(Ts_kf[:T_cur], e_hats_kf[:T_cur, 3], 'g')
                    plt.plot(Ts_kf[:T_cur], one_sigs_kf[:T_cur, 3], 'm')
                    plt.plot(Ts_kf[:T_cur], -one_sigs_kf[:T_cur, 3], 'm')
            plt.show()
            plt.close('all')

        # reset full estimator
        if(win_counts[idx_max] == num_windows):
            cauchyEsts[idx_max].reset()
            win_counts[idx_max] = 0
    ce_xhats = np.array(ce_xhats)
    ce_Phats = np.array(ce_Phats)
    foo = np.zeros(ce_xhats.shape[0])
    moment_info = {"x": ce_xhats, "P": ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
    ce.plot_simulation_history(moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf), with_partial_plot=True, with_cauchy_delay=True)
    foobar = 2

if __name__ == '__main__':
    #test_single_sliding_window()
    #test_python_debug_window_manager()
    test_leo4_windows()