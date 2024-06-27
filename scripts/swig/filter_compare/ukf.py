import numpy as np 

# Guide
# https://www.eecs.yorku.ca/course_archive/2010-11/W/4421/lectures/upf.pdf

# Returns weights for UKF
# nx == number of states
# kappa >= 0
# alpha <= 1 and alpha >= 0
# beta == 2 for gaussian (controls kurtosis)
def ukf_weights(nx, kappa, alpha, beta):
    lam = alpha**2 * (nx + kappa) - nx
    W_m0 = lam / (nx + lam)
    W_c0 = lam / (nx + lam) + (1 - alpha**2 + beta)
    W_mci = 1.0 / (2*(nx + lam))
    return lam, W_m0, W_c0, W_mci

def ukf_get_sigma_points(x, P, lam):
    nx = x.size
    sqrt_nxlam = np.sqrt(nx + lam)
    P_sqrt_scaled = np.linalg.cholesky(P) * sqrt_nxlam
    # Sigma Points k-1 | k-1
    sig_points = np.zeros((2*nx+1, nx))
    sig_points[0] = x
    for i in range(nx):
        sig_points[i+1] = x + P_sqrt_scaled[:,i]
        sig_points[i+1+nx] = x - P_sqrt_scaled[:,i]
    return sig_points

def ukf_propagate_sigma_points(sig_points_x, f, u):
    nx = sig_points_x[0].size
    prop_sig_points_x = np.zeros((2*nx+1, nx))
    prop_sig_points_x[0] = f(sig_points_x[0], u)
    for i in range(nx):
        prop_sig_points_x[i+1] = f(sig_points_x[i+1], u)
        prop_sig_points_x[i+1+nx] = f(sig_points_x[i+1+nx], u)
    return prop_sig_points_x

def ukf_compute_apriori_mean_cov(prop_sig_points, W_m0, W_c0, W_mci, Q):
    nx = prop_sig_points[0].size
    # Mean k | k-1
    x = W_m0 * prop_sig_points[0].copy()
    for i in range(1,2*nx+1):
        x += prop_sig_points[i] * W_mci
    # Covariance k | k-1
    ei = prop_sig_points[0] - x
    P = W_c0 * np.outer(ei,ei)
    for i in range(1,2*nx+1):
        ei = prop_sig_points[i] - x
        P += W_mci * np.outer(ei,ei)
    return x, P + Q

def ukf_compute_msmt_model_and_kalman_gain(sig_points_x, h, x_bar, W_m0, W_c0, W_mci, V):
    nx = x_bar.size
    # Measurement Model
    zbar_sig0 = h(sig_points_x[0])
    p = zbar_sig0.size
    zbar_sigs = np.zeros((2*nx+1, p))
    zbar_sigs[0] = zbar_sig0
    for i in range(nx):
        zbar_sigs[i+1] = h(sig_points_x[i+1])
        zbar_sigs[i+1+nx] = h(sig_points_x[i+1+nx])
    # Mean Estimate of zbar 
    zbar = W_m0 * zbar_sigs[0]
    for i in range(1,2*nx+1):
        zbar += W_mci * zbar_sigs[i]
    # Covariance of zbar Pzz
    ei = zbar_sigs[0] - zbar
    Pzz = W_c0 * np.outer(ei, ei)
    for i in range(1,2*nx+1):
        ei = zbar_sigs[i] - zbar
        Pzz += W_mci * np.outer(ei,ei)
    Pzz += V
    # Cross Covariance of x and z 
    ex0 = sig_points_x[0] - x_bar
    ez0 = zbar_sigs[0] - zbar
    Pxz = W_c0 * np.outer(ex0, ez0)
    for i in range(1,2*nx+1):
        exi = sig_points_x[i] - x_bar 
        ezi = zbar_sigs[i] - zbar
        Pxz += W_mci * np.outer(exi,ezi)
    # Form Kalman Gain: K = Pxz @ Pzz.I
    K = Pxz @ np.linalg.inv(Pzz) #np.linalg.solve(Pzz.T, Pxz.T).T
    return zbar, Pzz, K, zbar_sigs

def ukf_compute_posterior_mean_cov(x, P, Pzz, z, zbar, K):
    x_hat = x + K @ (z - zbar)
    P_hat = P - K @ Pzz @ K.T
    return x_hat, P_hat

def ukf(x, u, z, P, f, h, Q, V, lam, W_m0, W_c0, W_mci):
    # Sigma points k-1 | k-1
    sig_points_x = ukf_get_sigma_points(x, P, lam)
    # Propagate Sigma Points k | k -1
    sig_points_prop = ukf_propagate_sigma_points(sig_points_x, f, u)
    # Construct a-proiri mean and covariance 
    x_bar, P_bar = ukf_compute_apriori_mean_cov(sig_points_prop, W_m0, W_c0, W_mci, Q)
    # Form measurement model and kalman gain 
    zbar, Pzz, K, _ = ukf_compute_msmt_model_and_kalman_gain(sig_points_prop, h, x_bar, W_m0, W_c0, W_mci, V)
    # Construct state and covariance
    x_hat, P_hat = ukf_compute_posterior_mean_cov(x_bar, P_bar, Pzz, z, zbar, K)
    return x_hat, P_hat