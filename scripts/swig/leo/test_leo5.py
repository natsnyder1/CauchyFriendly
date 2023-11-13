import numpy as np
import os, sys 
file_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(file_dir + "/../cauchy")
import cauchy_estimator as ce
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
    sub_steps_per_dt = 60 # so sub intervals are dt / sub_steps_dt 
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
    Wd = np.array([[np.pow(beta_gauss, 2)]])
    # Initial uncertainty in position
    alpha_density_cauchy = 0.0039 # Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
    alpha_density_gauss = alpha_density_cauchy * CAUCHY_TO_GAUSS # Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
    alpha_pv_gauss = 1.0 # Initial Gaussian standard deviation in position and velocity of satellite
    alpha_pv_cauchy = alpha_pv_gauss * GAUSS_TO_CAUCHY # Initial converted uncertainty parameter in position and velocity of satellite converted for Cauchy Estimator
    P0 = np.zeros(25)
    P0[0] = pow(alpha_pv_gauss,2); P0[6] = pow(alpha_pv_gauss,2)
    P0[12] = pow(alpha_pv_gauss,2); P0[18] = pow(alpha_pv_gauss,2)
    P0[24] = pow(alpha_density_gauss,2)
    P0 = P0.reshape((5,5))
    cholP0 = np.linalg.cholesky(P0)
    
leo = leo_satellite_5state()

def leo5_ode(x, parmas):
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

def trans_model_5state_rk4(x):
    global leo 
    x_new = x.copy()
    dt_sub = leo.dt / leo.sub_steps_dt
    for _ in range(leo.sub_steps_dt):
        x_new = ce.runge_kutta4(leo5_ode, x_new, dt_sub)
    return x_new 


def leo_5state_transition_model_jacobians(x):
    Jac = ce.cd4_gvf(x, leo5_ode) # Jacobian matrix
    taylor_order = 6
    Phi_k = np.zeros((x.size,x.size))
    for i in range(taylor_order+1):
        Phi_k += np.linalg.matrix_power(Jac, i) * leo.dt**i / math.factorial(i)
    Gamma_k = np.zeros((x.size,1))
    Gamma_c = np.zeros((x.size)) # continous time Gamma 
    Gamma_c[4] = 1.0
    for i in range(taylor_order+1):
        Gamma_k += ( np.linalg.matrix_power(Jac, i) * leo.dt**(i+1) / math.factorial(i+1) ) @ Gamma_c
    return Phi_k, Gamma_k

def leo_5state_measurement_model_jacobian(x):
        global leo
        pass