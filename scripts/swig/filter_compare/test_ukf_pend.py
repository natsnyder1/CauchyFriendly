import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg',force=True)
import sys, os 
import cauchy_estimator as ce 
import ukf 

seed = 950 #np.random.randint(0,100000)
np.random.seed( seed )
print("Seed ", seed)

class PendulumParams:
    L = 0.3 # meters
    g = 9.81 # meters / second^2
    c = 0.6 # 1/seconds (damping)
    w_PSD = 0.01 # power spectral density of c-time process noise
    dt = 0.05 # integration step time

pend = PendulumParams() # Lets just make a simple globally viewable object to get ahold of these parameters when we want them

# The ODE
def pend_ode(x):
    dx_dt = np.zeros(2)
    dx_dt[0] = x[1]
    dx_dt[1] = -pend.g / pend.L * np.sin(x[0]) - pend.c * x[1]
    return dx_dt 

# Nonlinear transition model from t_k to t_k+1...ie: dt
def nonlin_transition_model(x, u = 0):
    return ce.runge_kutta4(pend_ode, x, pend.dt)

# This is the callback function correpsonding to the decription for point 1.) above 
def dynamics_update(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    ## Propagate x 
    xk = pyduc.cget_x()
    xbar = nonlin_transition_model(xk) # propagate from k -> k+1
    pyduc.cset_x(xbar)
    pyduc.cset_is_xbar_set_for_ece() # need to call this!
    ## Phi, Gamma, beta may update
    Jac_F = jacobian_pendulum_ode(xk)
    Phi_k, Gam_k = ce.discretize_nl_sys(Jac_F, Gamma_c, None, pend.dt, taylor_order, with_Gamk=True, with_Wk=False)
    pyduc.cset_Phi(Phi_k)
    pyduc.cset_Gamma(Gam_k)
    #pyduc.cset_beta(beta)

# This is the callback function correpsonding to the decription for point 2.) above 
def nonlinear_msmt_model(c_duc, c_zbar):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    ## Set zbar
    xbar = pyduc.cget_x() # xbar
    zbar = H @ xbar # for other systems, call your nonlinear h(x) function
    pyduc.cset_zbar(c_zbar, zbar)

# This is the callback function correpsonding to the decription for point 3.) above 
def msmt_model_jacobian(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    ## Set H: for other systems, call your nonlinear jacobian function H(x)
    pyduc.cset_H(H) # we could write some if condition to only set this once, but its such a trivial overhead, who cares

def ukf_h(x):
    return H @ x 

def jacobian_pendulum_ode(x):
    Jac = np.zeros((2,2))
    Jac[0,1] = 1
    Jac[1,0] = -pend.g/pend.L*np.cos(x[0])
    Jac[1,1] = -pend.c
    return Jac

theta_vec0 = np.array([np.pi/4, 0]) # initial angle of 45 degrees at 0 radians/sec
theta_k = theta_vec0.copy()
thetas = [theta_k]
propagations = 160
for k in range(propagations):
    theta_k = nonlin_transition_model(theta_k)
    thetas.append(theta_k)
thetas = np.array(thetas)
Ts = np.arange(propagations+1) * pend.dt

# Creating the dynamic simulation
V = np.array([[0.003]]) # measurement noise on theta
H = np.array([1.0,0.0]) # meausrement model
xk = theta_vec0.copy()
xs = [xk] # State vector history
ws = []   # Process noise history
vs = [V[0]**0.5 * np.random.randn()] # Measurement noise history
zs = [H @ xk + vs[0]] # Measurement history
propagations = 160
for k in range(propagations):
    wk = pend.dt * pend.w_PSD**0.5 * np.random.randn(1)
    xk[1] += wk
    xk = nonlin_transition_model(xk)
    xs.append(xk)
    ws.append(wk)
    vk = V[0]**0.5 * np.random.randn(1)
    zk = H @ xk + vk
    vs.append(vk)
    zs.append(zk)
xs = np.array(xs)
zs = np.array(zs)
ws = np.array(ws)
vs = np.array(vs)
#ce.plot_simulation_history(None, (xs,zs,ws,vs), None)

# Continuous time Gamma (\Gamma_c)
Gamma_c = np.array([[0.0,1.0]]).T
W_c = np.array([[pend.w_PSD]])
I2 = np.eye(2)
H = H.reshape((1,2))
taylor_order = 2

# Setting up and running the EKF
# The gaussian_filters module has a "run_ekf" function baked in, but we'll just show the whole thing here
P0_kf = np.eye(2) * 0.3
x0_kf = np.random.multivariate_normal(theta_vec0, P0_kf) # lets initialize the Kalman filter slightly off from the true state position

xs_kf = [x0_kf.copy()] 
Ps_kf = [P0_kf.copy()] 
x_kf = x0_kf.copy()
P_kf = P0_kf.copy() 
x_ukf = x0_kf.copy()
P_ukf = P0_kf.copy()
for k in range(propagations):
    Jac_F = jacobian_pendulum_ode(x_kf)
    Phi_k, W_k = ce.discretize_nl_sys(Jac_F, Gamma_c, W_c, pend.dt, taylor_order, with_Gamk = False, with_Wk = True)
    # Propagate covariance and state estimates
    P_kf = Phi_k @ P_kf @ Phi_k.T + W_k
    x_kf = nonlin_transition_model(x_kf)
    # Form Kalman Gain, update estimate and covariance
    K = P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
    zbar = H @ x_kf
    r = zs[k+1] - zbar
    x_kf += K @ r 
    P_kf = (I2 - K @ H) @ P_kf @ (I2 - K @ H).T + K @ V @ K.T
    # Store estimates
    xs_kf.append(x_kf.copy())
    Ps_kf.append(P_kf.copy())
xs_kf = np.array(xs_kf)
Ps_kf = np.array(Ps_kf)

# Run the UKF 
kappa = 0.0
alpha = 1e-3
beta = 2.0
nx = 2
lam, W_m0, W_c0, W_mci = ukf.ukf_weights(nx, kappa, alpha, beta)
x_ukf = x0_kf.copy()
P_ukf = P0_kf.copy()
xs_ukf = [x_ukf.copy()] 
Ps_ukf = [P_ukf.copy()] 
for k in range(propagations):
    Jac_F = jacobian_pendulum_ode(x_ukf)
    Phi_k, W_k = ce.discretize_nl_sys(Jac_F, Gamma_c, W_c, pend.dt, taylor_order, with_Gamk = False, with_Wk = True)
    sig_points_kk = ukf.ukf_get_sigma_points(x_ukf, P_ukf, lam)
    sig_points_k1k = ukf.ukf_propagate_sigma_points(sig_points_kk, nonlin_transition_model, 0)
    x_ukf_prior, P_ukf_prior = ukf.ukf_compute_apriori_mean_cov(sig_points_k1k, W_m0, W_c0, W_mci, W_k)
    #print("Step: {}, Lambda: {}, P_apriori: {}, sig_points_kk: {}, sig_points_k1k: {}".format(i, lam, P_ukf_prior, sig_points_kk.copy().reshape(-1), sig_points_k1k.copy().reshape(-1)))
    z = zs[k+1]
    zbar, Pzz, K, zbar_sig_points = ukf.ukf_compute_msmt_model_and_kalman_gain(sig_points_k1k, ukf_h, x_ukf_prior, W_m0, W_c0, W_mci, V)
    x_ukf, P_ukf = ukf.ukf_compute_posterior_mean_cov(x_ukf_prior, P_ukf_prior, Pzz, z, zbar, K)
    xs_ukf.append(x_ukf.copy())
    Ps_ukf.append(P_ukf.copy())
xs_kf = np.array(xs_kf)
Ps_kf = np.array(Ps_kf)


es_kf = np.array([xt - xh[0:6] for xt,xh in zip(xs,xs_kf)])
sigs_kf = np.array([np.diag(P)**0.5 for P in Ps_kf])
es_ukf = np.array([xt - xh[0:6] for xt,xh in zip(xs,xs_ukf)])
sigs_ukf = np.array([np.diag(P)**0.5 for P in Ps_ukf])
'''
ylabels = ['Angle Error (rad)', 'Rate Error (rad/sec)']
tks = np.arange(xs.shape[0]) * pend.dt 
plt.figure(figsize=(14,11))
color_scheme = "State Errors of EKF (blue) + UKF (green)"
plt.suptitle("State Error Plot of Position/Velocity\nSolid Lines = State Errors, Dashed Lines = One Sigma Bounds\n"+color_scheme)
for i in range(2):
    plt.subplot(2,1,i+1)
    plt.ylabel(ylabels[i])
    plt.plot(tks, es_kf[:,i], color='b')
    plt.plot(tks, sigs_kf[:,i], color='b', linestyle='--')
    plt.plot(tks, -sigs_kf[:,i], color='b', linestyle='--')
    
    plt.plot(tks, es_ukf[:,i], color='g')
    plt.plot(tks, sigs_ukf[:,i], color='g', linestyle='--')
    plt.plot(tks, -sigs_ukf[:,i], color='g', linestyle='--')
plt.xlabel("Time (sec)")
plt.show()
'''

# Plot Simulation results 
#ce.plot_simulation_history( None, (xs,zs,ws,vs), (xs_kf, Ps_kf) )

scale_g2c = 1.0 / 1.3898 # scale factor to fit the cauchy to the gaussian
beta = np.array([pend.w_PSD / pend.dt])**0.5 * scale_g2c
gamma = np.array([V[0,0]**0.5]) * scale_g2c
x0_ce = x0_kf.copy()
A0 = np.eye(2)
p0 = np.diag(P0_kf)**0.5 * scale_g2c 
b0 = np.zeros(2)
num_controls = 0


theta0 = 2*theta_vec0
# Creating the dynamic simulation
xk = theta0.copy()
xs = [xk] # State vector history
ws = []   # Process noise history
vs = [V[0]**0.5 * np.random.randn()] # Measurement noise history
zs = [H @ xk + vs[0]] # Measurement history
propagations = 160
c0 = pend.c
for k in range(propagations):
    wk = pend.dt * pend.w_PSD**0.5 * np.random.randn(1)
    xk[1] += wk
    xk = nonlin_transition_model(xk)
    xs.append(xk)
    ws.append(wk)
    vk = V[0]**0.5 * np.random.randn(1)
    zk = H @ xk + vk
    vs.append(vk)
    zs.append(zk)
    if k == 40:
        pend.c = c0 * 4 #* 0.25 # all of a sudden, the pendulum changes its damping greatly
pend.c = c0
xs = np.array(xs)
zs = np.array(zs)
ws = np.array(ws)
vs = np.array(vs)

# Setting up and running the EKF
x0_kf = np.random.multivariate_normal(theta0, P0_kf) # lets initialize the Kalman filter slightly off from the true state position

xs_kf = [x0_kf.copy()] 
Ps_kf = [P0_kf.copy()] 
x_kf = x0_kf.copy()
P_kf = P0_kf.copy() 
for k in range(propagations):
    Jac_F = jacobian_pendulum_ode(x_kf)
    Phi_k, W_k = ce.discretize_nl_sys(Jac_F, Gamma_c, W_c, pend.dt, taylor_order, with_Gamk = False, with_Wk = True)
    # Propagate covariance and state estimates
    P_kf = Phi_k @ P_kf @ Phi_k.T + W_k
    x_kf = nonlin_transition_model(x_kf)
    # Form Kalman Gain, update estimate and covariance
    K = P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
    zbar = H @ x_kf
    r = zs[k+1] - zbar
    x_kf += K @ r 
    P_kf = (I2 - K @ H) @ P_kf @ (I2 - K @ H).T + K @ V @ K.T
    # Store estimates
    xs_kf.append(x_kf.copy())
    Ps_kf.append(P_kf.copy())
xs_kf = np.array(xs_kf)
Ps_kf = np.array(Ps_kf)

x_ukf = x0_kf.copy()
P_ukf = P0_kf.copy()
xs_ukf = [x_ukf.copy()] 
Ps_ukf = [P_ukf.copy()] 
for k in range(propagations):
    Jac_F = jacobian_pendulum_ode(x_ukf)
    _, W_k = ce.discretize_nl_sys(Jac_F, Gamma_c, W_c, pend.dt, taylor_order, with_Gamk = False, with_Wk = True)
    sig_points_kk = ukf.ukf_get_sigma_points(x_ukf, P_ukf, lam)
    sig_points_k1k = ukf.ukf_propagate_sigma_points(sig_points_kk, nonlin_transition_model, 0)
    x_ukf_prior, P_ukf_prior = ukf.ukf_compute_apriori_mean_cov(sig_points_k1k, W_m0, W_c0, W_mci, W_k)
    #print("Step: {}, Lambda: {}, P_apriori: {}, sig_points_kk: {}, sig_points_k1k: {}".format(i, lam, P_ukf_prior, sig_points_kk.copy().reshape(-1), sig_points_k1k.copy().reshape(-1)))
    z = zs[k+1]
    zbar, Pzz, K, zbar_sig_points = ukf.ukf_compute_msmt_model_and_kalman_gain(sig_points_k1k, ukf_h, x_ukf_prior, W_m0, W_c0, W_mci, V)
    x_ukf, P_ukf = ukf.ukf_compute_posterior_mean_cov(x_ukf_prior, P_ukf_prior, Pzz, z, zbar, K)
    xs_ukf.append(x_ukf.copy())
    Ps_ukf.append(P_ukf.copy())
xs_kf = np.array(xs_kf)
Ps_kf = np.array(Ps_kf)


# Run the Cauchy Estimator
x0_ce = x0_kf.copy()
swm_print_debug = False 
win_print_debug = False
num_windows = 10
cauchyEst = ce.PySlidingWindowManager("nonlin", num_windows, swm_print_debug, win_print_debug)
cauchyEst.initialize_nonlin(x0_ce, A0, p0, b0, beta/3.5, gamma/1.5, dynamics_update, nonlinear_msmt_model, msmt_model_jacobian, num_controls, pend.dt)
for zk in zs:
    cauchyEst.step(zk, None)
cauchyEst.shutdown()
xs_mce = cauchyEst.avg_moment_info['x']
Ps_mce = cauchyEst.avg_moment_info['P']
Ps_mce[0][1,1] = Ps_kf[0][1,1] # undefined at setp 0 since H @ A orthog

es_kf = np.array([xt - xh[0:6] for xt,xh in zip(xs,xs_kf)])
sigs_kf = np.array([np.diag(P)**0.5 for P in Ps_kf])
es_ukf = np.array([xt - xh[0:6] for xt,xh in zip(xs,xs_ukf)])
sigs_ukf = np.array([np.diag(P)**0.5 for P in Ps_ukf])
es_mce = np.array([xt - xh[0:6] for xt,xh in zip(xs,xs_mce)])
sigs_mce = np.array([np.diag(P)**0.5 for P in Ps_mce])
ylabels = ['Angle Error (rad)', 'Rate Error (rad/sec)']
tks = np.arange(xs.shape[0]) * pend.dt 
plt.figure(figsize=(14,11))
color_scheme = "State Errors of EKF (blue) + UKF (green)"
plt.suptitle("State Error Plot of Position/Velocity\nSolid Lines = State Errors, Dashed Lines = One Sigma Bounds\n"+color_scheme)
for i in range(2):
    plt.subplot(2,1,i+1)
    plt.ylabel(ylabels[i])

    plt.plot(tks, es_ukf[:,i], color='tab:olive')
    plt.plot(tks, sigs_ukf[:,i], color='tab:brown', linestyle='--')
    plt.plot(tks, -sigs_ukf[:,i], color='tab:brown', linestyle='--')

    plt.plot(tks, es_kf[:,i], color='g')
    plt.plot(tks, sigs_kf[:,i], color='m', linestyle='--')
    plt.plot(tks, -sigs_kf[:,i], color='m', linestyle='--')

    plt.plot(tks, es_mce[:,i], color='b')
    plt.plot(tks, sigs_mce[:,i], color='r', linestyle='--')
    plt.plot(tks, -sigs_mce[:,i], color='r', linestyle='--')

plt.xlabel("Time (sec)")
plt.show()
foobar = 2

#from scipy import io
#io.savemat('pend.mat', {"es_kf" : es_kf, "sigs_kf" : sigs_kf, "es_ukf" : es_ukf, "sigs_ukf": sigs_ukf, "es_mce" : es_mce, "sigs_mce" : sigs_mce })
