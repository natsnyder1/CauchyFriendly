import numpy as np 
import cauchy_estimator as ce 
import gaussian_filters as gf
import math
import matplotlib
matplotlib.use('TkAgg',force=True)

class PendulumParams:
    L = 0.3 # meters
    g = 9.81 # meters / second^2
    damp = 0.3 # 1/seconds (damping)
    dt = 0.05
    PSD = 1.0 # power spectral density of c-time process
    V = 0.10 # additive discrete time measurement noise
    H = np.array([1.0,0.0]) # observe the angle

pend = PendulumParams()

def pend_ode(x):
    dx_dt = np.zeros(2)
    dx_dt[0] = x[1]
    dx_dt[1] = -pend.g / pend.L * np.sin(x[0]) - pend.damp * x[1]
    return dx_dt

def nonlin_transition_model(x):
    return ce.runge_kutta4(pend_ode, x, pend.dt)

def c2d_linearized_dynamics(x):
    Jac = ce.cd4_gvf(x, pend_ode) # Jacobian matrix
    taylor_order = 2
    Phi_k = np.zeros((x.size,x.size))
    for i in range(taylor_order+1):
        Phi_k += np.linalg.matrix_power(Jac, i) * pend.dt**i / math.factorial(i)
    Gamma_k = np.zeros(x.size)
    Gamma_c = np.zeros(x.size) # continous time Gamma 
    Gamma_c[1] = 1.0
    for i in range(taylor_order+1):
        Gamma_k += ( np.linalg.matrix_power(Jac, i) * pend.dt**(i+1) / math.factorial(i+1) ) @ Gamma_c
    return Phi_k, Gamma_k.reshape((2,1))

def simulate_pendulum(x0, steps):
    xk = x0.copy()
    xs = [xk.copy()]
    ws = []
    vs = []
    zs = []

    v0 = np.random.randn() * np.sqrt(pend.V)
    z0 = pend.H @ xk + v0
    vs.append(v0)
    zs.append(z0)
    for i in range(steps):
        _, Gamma_k = c2d_linearized_dynamics(xk)
        # Some different ways of creating the process noise matrix, or sampling from it 
        #wk = np.random.standard_cauchy() * np.sqrt(pend.PSD/pend.dt) * ce.GAUSSIAN_TO_CAUCHY_NOISE

        #Wk = 1.0/pend.dt * Gamma_k @ np.atleast_2d(pend.PSD) @ Gamma_k.T #1
        #wk = Gamma_k.reshape(-1) * np.random.randn() * np.sqrt(pend.PSD/pend.dt) #1a
        #wk = np.random.multivariate_normal(np.zeros(2), Wk) #1b

        # Qk and Wk come out very similar, except that Qk is full rank and Wk is not
        _,_,Qk = ce.discretize_nl_sys( ce.cd4_gvf(xk, pend_ode), np.array([[0.0,1.0]]).T, np.atleast_2d(pend.PSD), pend.dt, 2) #2
        wk = np.random.multivariate_normal(np.zeros(2), Qk)
        
        xk = nonlin_transition_model(xk) + wk
        xs.append(xk)
        ws.append(wk)
        vk = np.random.randn() * np.sqrt(pend.V)
        zk = pend.H @ xk + vk
        vs.append(vk)
        zs.append(zk)
    
    ws = np.array(ws)
    pncc = 1 if ws.ndim == 1 else 2
    return ( np.array(xs), np.array(zs).reshape((steps+1,1)), ws.reshape((steps, pncc)), np.array(vs).reshape((steps+1,1)) )

def ekf_f(x, u, other_params):
    return ce.runge_kutta4(pend_ode, x, pend.dt)

def ekf_h(x, other_params):
    return pend.H @ x

def ekf_callback_Phi_Gam(x, u, other_params):
    return c2d_linearized_dynamics(x)

def msmt_model(x):
    return pend.H @ x

def ekf_callback_H(x, other_params):
    return pend.H.copy().reshape((1,2))


def ece_dynamics_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    x = pyduc.cget_x()
    Phi_k, Gamma_k = c2d_linearized_dynamics(x)
    xbar = nonlin_transition_model(x)
    pyduc.cset_x(xbar)
    pyduc.cset_is_xbar_set_for_ece()
    pyduc.cset_Phi(Phi_k)
    pyduc.cset_Gamma(Gamma_k)

def ece_nonlinear_msmt_model(c_duc, c_zbar):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    x = pyduc.cget_x() # xbar
    zbar = msmt_model(x)
    pyduc.cset_zbar(c_zbar, zbar)

def ece_extended_msmt_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    pyduc.cset_H(pend.H)



np.random.seed(10)
x0 = np.array([np.pi/2, 0])
steps = 200
xs, zs, ws, vs = simulate_pendulum(x0, steps)

P0 = np.diag(np.array([0.05, 0.05]))
x0_kf = np.random.multivariate_normal(x0, P0)
W = np.atleast_2d(pend.PSD/pend.dt)
V = np.atleast_2d(pend.V)
xs_kf, Ps_kf = gf.run_extended_kalman_filter(x0_kf, None, zs[1:], ekf_f, ekf_h, ekf_callback_Phi_Gam, ekf_callback_H, P0, W, V, None)

# Cauchy Estimator
A0 = np.eye(2)
p0 = np.sqrt(np.diag(P0)) * ce.GAUSSIAN_TO_CAUCHY_NOISE
b0 = np.zeros(2)
beta = (np.sqrt(W) * ce.GAUSSIAN_TO_CAUCHY_NOISE).reshape(-1) / 5 #/ 8 # tuning
gamma = (np.sqrt(V) * ce.GAUSSIAN_TO_CAUCHY_NOISE).reshape(-1) #/ np.sqrt(2) # tuning


#for i in range(6):
#print("Testing windows of {}!".format(3+i))
ce.set_tr_search_idxs_ordering([1,0])
num_windows = 6 #3 + i
num_controls = 0
swm_print_debug = True
win_print_debug = False
cauchyEst = ce.PySlidingWindowManager("nonlin", num_windows, swm_print_debug, win_print_debug)
cauchyEst.initialize_nonlin(x0_kf, A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, num_controls, pend.dt, 0, None)
for zk in zs:
    cauchyEst.step(zk, None)
cauchyEst.shutdown()
ce.plot_simulation_history(cauchyEst.moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf) )
#ce.plot_simulation_history(cauchyEst.avg_moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf) )

foo = 5

#Ts = np.arange(xs.shape[0]) * pend.dt
#plt.subplot(211)
#plt.plot(Ts, xs[:,0], 'r')
#plt.plot(Ts, xs_kf[:,0], 'g')
#plt.subplot(212)
#plt.plot(Ts, xs[:,1], 'r')
#plt.plot(Ts, xs_kf[:,1], 'g')
#plt.show()



foo = 0
