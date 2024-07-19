import ukf as ukf 
import numpy as np 
import matplotlib.pyplot as plt
import cauchy_estimator as ce 

# FILTERS ALL USE THIS
def f(x,u):
    return 3.5 * np.exp(-0.1 * x**2) + np.sin(np.pi*(4*np.e-2)*u) + 1 #0.9 * x #

# FOR SYS MIS-IDENTIFICATION -- SIMULATION
#def _f(x,u):
#    return 3.1 * np.exp(-0.15 * x**2) + np.sin(np.pi*(4.3*np.e-2.1)*u) + 1.1 #0.9 * x #


def h(x):
    return 0.2*x**2 #0.4 * x #

def ece_dynamics_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    # Set Phi and Gamma
    x = pyduc.cget_x()
    step = pyduc.cget_step()-1
    Phi = ce.cd4_gvf(x, f, (step,))
    pyduc.cset_Phi(Phi)
    pyduc.cset_Gamma(np.array([[1.0]]))
    # Propagate and set x
    xbar = f(x, step) 
    pyduc.cset_x(xbar)
    pyduc.cset_is_xbar_set_for_ece()

def ece_nonlinear_msmt_model(c_duc, c_zbar):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    zbar = h(xbar)
    pyduc.cset_zbar(c_zbar, zbar)

def ece_extended_msmt_update_callback(c_duc):
    pyduc = ce.Py_CauchyDynamicsUpdateContainer(c_duc)
    xbar = pyduc.cget_x() # xbar
    H = ce.cd4_gvf(xbar, h)
    pyduc.cset_H(H)


def test_ukf():
    np.random.seed(10)
    std_Q = 0.053
    Q = np.array([[std_Q**2]])
    std_V = 0.253
    V = np.array([[std_V**2]])
    std_P = 0.1
    P = np.array([[std_P**2]])

    num_props = 100
    Ts = np.arange(num_props+1)
    x0 = np.array([0.5]) 
    _x = x0 + std_P * np.random.randn()
    xs = [_x] 
    zs = [h(_x)]
    for i in range(num_props):
        _x = f(_x,i) + std_Q * np.random.randn() #std_Q * np.clip(np.random.standard_cauchy(), -50, 50) # 
        _z = h(_x) + std_V * np.random.randn() #std_V * np.clip(np.random.standard_cauchy(), -50,50) # 
        xs.append(_x)
        zs.append(_z)
    xs = np.array(xs)
    zs = np.array(zs)
    print("True State (b) and Measurement (r)")
    plt.plot(Ts, xs, 'b')
    plt.plot(Ts, zs, 'r')
    plt.show()
    
    WITH_PLOT_INTRM = False
    WITH_FILTERPY_UKF_DEBUG = False
    WITH_KF = True
    WITH_CAUCHY = True

    x_ukf = x0.copy() 
    P_ukf = P.copy()
    xs_ukf = [x_ukf] 
    Ps_ukf = [P_ukf]
    if WITH_KF:
        x_kf = x0.copy()
        P_kf = P.copy()
        xs_kf = [x_kf]
        Ps_kf = [P_kf]
    
    if WITH_CAUCHY:
        xs_ce = []
        Ps_ce = []
        cauchyEst = ce.PyCauchyEstimator("nonlin", num_props+1, False)
        A0 = np.array([[1.0]])
        p0 = np.array([std_P]) * ce.GAUSSIAN_TO_CAUCHY_NOISE
        b0 = np.array([0]) 
        beta = np.array([std_Q]) * ce.GAUSSIAN_TO_CAUCHY_NOISE 
        gamma = np.array([std_V]) * ce.GAUSSIAN_TO_CAUCHY_NOISE 
        cauchyEst.initialize_nonlin(x0.copy(),A0, p0, b0, beta, gamma, ece_dynamics_update_callback, ece_nonlinear_msmt_model, ece_extended_msmt_update_callback, 0, 0, step = 0)
        x_ce, P_ce = cauchyEst.step(zs[0])
        xs_ce.append(x_ce)
        Ps_ce.append(P_ce)

    kappa = 0.0
    alpha = 0.30
    beta = 2.0
    nx = 1
    lam, W_m0, W_c0, W_mci = ukf.ukf_weights(nx, kappa, alpha, beta)
    for i in range(num_props):
        # UKF Direct Call
        #x_ukf, P_ukf = ukf.ukf(x_ukf, i, zs[i+1], P_ukf, f, h, Q, V, lam, W_m0, W_c0, W_mci)
        # UKF Steps
        sig_points_kk = ukf.ukf_get_sigma_points(x_ukf, P_ukf, lam)
        sig_points_k1k = ukf.ukf_propagate_sigma_points(sig_points_kk, f, i)
        x_ukf_prior, P_ukf_prior = ukf.ukf_compute_apriori_mean_cov(sig_points_k1k, W_m0, W_c0, W_mci, Q)
        #print("Step: {}, Lambda: {}, P_apriori: {}, sig_points_kk: {}, sig_points_k1k: {}".format(i, lam, P_ukf_prior, sig_points_kk.copy().reshape(-1), sig_points_k1k.copy().reshape(-1)))
        z = zs[i+1]
        zbar, Pzz, K, zbar_sig_points = ukf.ukf_compute_msmt_model_and_kalman_gain(sig_points_k1k, h, x_ukf_prior, W_m0, W_c0, W_mci, V)
        x_ukf, P_ukf = ukf.ukf_compute_posterior_mean_cov(x_ukf_prior, P_ukf_prior, Pzz, z, zbar, K)
        xs_ukf.append(x_ukf)
        Ps_ukf.append(P_ukf)
        
        # (E)KF
        if WITH_KF:
            I = np.eye(1)
            Phi = ce.cd4_gvf(x_kf, f, (i,))
            P_kf = Phi @ P_kf @ Phi.T + Q 
            x_kf = f(x_kf, i)
            H = ce.cd4_gvf(x_kf, h)
            K_kf = P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
            zbar_kf = h(x_kf)
            x_kf = x_kf + K_kf @ (z - zbar_kf)
            P_kf = (I - K_kf @ H) @ P_kf @ (I - K_kf @ H).T + K_kf @ V @ K_kf.T
            foobar = 2
            xs_kf.append(x_kf)
            Ps_kf.append(P_kf)
        
        # Run Cauchy 
        if WITH_CAUCHY:
            x_ce, P_ce = cauchyEst.step(z)
            xs_ce.append(x_ce)
            Ps_ce.append(P_ce)
        
        # Intermediate Plotting
        if WITH_PLOT_INTRM:
            plt.figure()
            plt.scatter(xs[i], 0, color='k', marker = "v")
            plt.scatter(xs[i+1], 0, color='m', marker = "^")
            plt.scatter(sig_points_kk, np.zeros_like(sig_points_kk), color='k', marker = "+")
            plt.scatter(sig_points_k1k, np.zeros_like(sig_points_k1k), color='b', marker = "+")

            std_P_ukf_prior = P_ukf_prior**0.5
            plt.scatter(x_ukf_prior, 0, color='b', marker = "*")
            plt.scatter(x_ukf_prior + std_P_ukf_prior, 0, color='b', marker = "s")
            plt.scatter(x_ukf_prior - std_P_ukf_prior, 0, color='b', marker = "s")
            std_P_ukf = P_ukf**0.5
            plt.scatter(x_ukf, 0, color='m', marker = "*")
            plt.scatter(x_ukf + std_P_ukf, 0, color='m', marker = "s")
            plt.scatter(x_ukf - std_P_ukf, 0, color='m', marker = "s")

            if WITH_KF:
                std_P_kf = P_kf**0.5
                plt.scatter(x_kf, 0, color='y', marker = "o")
                plt.scatter(x_kf + std_P_kf, 0, color='y', marker = "s")
                plt.scatter(x_kf - std_P_kf, 0, color='y', marker = "s")

            plt.scatter(z, 0, color='r', marker = "*")
            plt.scatter(zbar_sig_points, np.zeros_like(zbar_sig_points), color='r', marker = "+")

            plt.show()
            plt.close('all')

    #'''
    if WITH_FILTERPY_UKF_DEBUG:
        from filterpy.kalman import UnscentedKalmanFilter as UKF_FP
        from filterpy.kalman import MerweScaledSigmaPoints
        points = MerweScaledSigmaPoints(1, alpha=alpha, beta=beta, kappa=kappa)
        ukf_fp = UKF_FP(dim_x=1,dim_z=1,dt=0, fx=f, hx=h, points=points)
        ukf_fp.x = x0.copy()
        ukf_fp.P = P.copy()
        ukf_fp.R = V.copy()
        ukf_fp.Q = Q.copy()
        debug_xs_ukf = [x0.copy()]
        debug_Ps_ukf = [P.copy()]
        for i in range(1, num_props+1):
            ukf_fp.predict(dt = i-1)
            ukf_fp.update(zs[i])
            debug_xs_ukf.append(ukf_fp.x.copy())
            debug_Ps_ukf.append(ukf_fp.P.copy())
    #'''

    xs_ukf = np.array(xs_ukf)
    Ps_ukf = np.array(Ps_ukf).reshape(-1)
    plt.plot(Ts, xs - xs_ukf, 'b')
    plt.plot(Ts, Ps_ukf**0.5, 'b--')
    plt.plot(Ts, -Ps_ukf**0.5, 'b--')
    if WITH_KF:
        xs_kf = np.array(xs_kf)
        Ps_kf = np.array(Ps_kf).reshape(-1)
        plt.plot(Ts, xs - xs_kf, 'g')
        plt.plot(Ts, Ps_kf**0.5, 'g--')
        plt.plot(Ts, -Ps_kf**0.5, 'g--')
    if WITH_FILTERPY_UKF_DEBUG:
        debug_xs_ukf = np.array(debug_xs_ukf)
        debug_Ps_ukf = np.array(debug_Ps_ukf).reshape(-1)
        plt.plot(Ts, xs - debug_xs_ukf, 'm')
        plt.plot(Ts, debug_Ps_ukf**0.5, 'm--')
        plt.plot(Ts, -debug_Ps_ukf**0.5, 'm--')
    if WITH_CAUCHY:
        xs_ce = np.array(xs_ce)
        Ps_ce = np.array(Ps_ce).reshape(-1)
        plt.plot(Ts, xs - xs_ce, 'r')
        plt.plot(Ts, Ps_ce**0.5, 'r--')
        plt.plot(Ts, -Ps_ce**0.5, 'r--')
    plt.show()
    foobar = 2

if __name__ == "__main__":
    test_ukf()