import os
import numpy as np
import cauchy_estimator as ce
import gaussian_filters as gf
file_dir = os.path.dirname(os.path.abspath(__file__))

# Creates a single cauchy estimator instance and runs the estimator for several steps
def test_3state_lti_single_window():
    ndim = 3
    Phi = np.array([ [1.4, -0.6, -1.0], 
                     [-0.2,  1.0,  0.5],  
                     [0.6, -0.6, -0.2]] )
    Gamma = np.array([.1, 0.3, -0.2])
    H = np.array([1.0, 0.5, 0.2])
    beta = np.array([0.1]) # Cauchy process noise scaling parameter(s)
    gamma = np.array([0.2]) # Cauchy measurement noise scaling parameter(s)
    A0 = np.eye(ndim) # Unit directions of the initial state uncertainty
    p0 = np.array([0.10, 0.08, 0.05]) # Initial state uncertainty cauchy scaling parameter(s)
    b0 = np.zeros(ndim) # Initial median of system state

    np.random.seed(10)
    num_steps = 7

    # Simulate Dynamic System -- Either in Cauchy Noise or Gaussian Noise
    # Applying arbitrary controls to the estimator via 'B @ u'
    num_controls = 0
    B = np.random.randn(ndim,num_controls) if num_controls > 0 else  None
    us = np.random.randn(num_steps, num_controls) if num_controls > 0 else  None
    x0_truth = p0 * np.random.randn(ndim)
    (xs, zs, ws, vs) = ce.simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)

    # Testing Single Cauchy Estimator Instance
    cauchyEst = ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma, 0, 0)
    z0 = zs[0]
    u0 = np.zeros(num_controls) if num_controls > 0 else None 
    cauchyEst.step(z0, u0) # initial step has no propagation
    for i in range(1,len(zs)):
        zk1 = zs[i]
        uk = us[i-1] if num_controls > 0 else None 
        cauchyEst.step(zk1, uk)
    ce.plot_simulation_history( cauchyEst.moment_info, (xs, zs, ws, vs), None )
    cauchyEst.shutdown()

# Creates a single cauchy estimator instance and runs the estimator for several steps
def test_2state_lti_single_window():
    # x_{k+1} = \Phi_k @ x_k + B_k @ u_k + \Gamma_k @ w_k
    # z_k = H @ x_k + v_k
    ndim = 2
    Phi = np.array([ [0.9, 0.1], [-0.2, 1.1] ])
    Gamma = np.array([.1, 0.3])
    H = np.array([1.0, 0.0])
    beta = np.array([0.1]) # Cauchy process noise scaling parameter(s)
    gamma = np.array([0.2]) # Cauchy measurement noise scaling parameter(s)
    A0 = np.eye(ndim) # Unit directions of the initial state uncertainty
    p0 = np.array([0.10, 0.05]) # Initial state uncertainty cauchy scaling parameter(s)
    b0 = np.zeros(ndim) # Initial median of system state

    np.random.seed(15)
    num_steps = 10

    # Applying (arbitrary) controls to the estimator via 'B @ u'
    num_controls = 0
    B = np.random.randn(ndim,num_controls) if num_controls > 0 else  None
    us = np.random.randn(num_steps, num_controls) if num_controls > 0 else  None
    
    # Simulate system states and measurements
    x0_truth = p0 * np.random.randn(ndim)
    (xs, zs, ws, vs) = ce.simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)

    # Run Cauchy Estimator
    cauchyEst = ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma, 0, 0)
    z0 = zs[0]
    u0 = np.zeros(num_controls) if num_controls > 0 else None 
    cauchyEst.step(z0, u0) # initial step has no propagation, and no "u" control
    for i in range(1,len(zs)):
        zk1 = zs[i]
        uk = us[i-1] if num_controls > 0 else None 
        cauchyEst.step(zk1, uk)
    ce.plot_simulation_history( cauchyEst.moment_info, (xs, zs, ws, vs), None )
    cauchyEst.shutdown()

def test_1state_lti():
    #n = 1
    #cmcc = 0
    #pncc = 1
    #p = 1
    Phi = np.array([[0.9]])
    B = None
    Gamma = np.array([0.4])
    H = np.array([2.0])
    beta = np.array([0.1]) # Cauchy process noise scaling parameter(s)
    gamma = np.array([0.2]) # Cauchy measurement noise scaling parameter(s)
    A0 = np.array([[1.0]]) # Unit directions of the initial state uncertainty
    p0 = np.array([0.10]) # Initial state uncertainty cauchy scaling parameter(s)
    b0 = np.zeros(1) # Initial median of system state

    zs = np.array([-0.44368369151309078, 0.42583824213752575, -0.33410810748025471, -0.50758511396868289, 
                -0.21567892215326886, 0.22514658508547963, 0.49585892022310135, 0.7119460882715376, 
                -2.7235055765981881, -2.7488835688860456, 0.6978978132016932])
    cauchyEst = ce.PyCauchyEstimator("lti", zs.size, True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma)

    estimates = []
    for i in range(zs.size):
        cauchyEst.step(zs[i], None)
        estimates.append(cauchyEst.moment_info["x"][i])
    print("State Estimates are:\n", estimates) # matches 1 state c++ cauchy example

# Creates several estimators and looks at their cpdfs over time
def test_2state_cpdfs():
    ndim = 2
    Phi = np.array([ [0.9, 0.1], [-0.2, 1.1] ])
    Gamma = np.array([.1, 0.3])
    H = np.array([1.0, 0.5])
    beta = np.array([0.1]) # Cauchy process noise scaling parameter(s)
    gamma = np.array([0.2]) # Cauchy measurement noise scaling parameter(s)
    A0 = np.eye(ndim) # Unit directions of the initial state uncertainty
    p0 = np.array([0.10, 0.05]) # Initial state uncertainty cauchy scaling parameter(s)
    b0 = np.zeros(ndim) # Initial median of system state

    num_steps = 9
    # Simulate system states and measurements
    #zs = np.array([0.022356919463887182, -0.22675889756491788, 0.42133397996398181, 
    #                  -1.7507202433585822, -1.3984154994099112, -1.7541436172809546, -1.8796017689052031, 
    #                  -1.9279807448991575, -1.9071129520752277, -2.0343612017356922])

    # Simulate system states and measurements
    x0_truth = p0 * np.random.randn(ndim)
    #(xs, zs, ws, vs) = ce.simulate_cauchy_ltiv_system(num_steps, x0_truth, None, Phi, None, Gamma, beta, H, gamma, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)
    (xs, zs, ws, vs) = ce.simulate_gaussian_ltiv_system(num_steps, x0_truth, None, Phi, None, Gamma, np.array([[(1.3898*beta)**2]]), H, np.array([[(1.3898*gamma)**2]]), with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)

    '''
    # Plot only the evolution of a single cpdf
    cauchyEst = ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, None, Gamma, beta, H, gamma)
    for i in range(num_steps+1):
        cauchyEst.step(zs[i], None)
        xhat, _ = cauchyEst.get_last_mean_cov()
        X,Y,Z = cauchyEst.get_2D_pointwise_cpdf(xhat[0]-1.5, xhat[0]+1.5, 0.02, xhat[1]-1.5, xhat[1]+1.5, 0.02)
        cauchyEst.plot_2D_pointwise_cpdf(X,Y,Z)
    '''

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    '''
    plot_oldest_newest = True # setting this to false plots all the cpdfs
    cauchyEsts = [ ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True) for _ in range(num_steps) ]
    cauchyEsts[0].initialize_lti(A0, p0, b0, Phi, None, Gamma, beta, H, gamma)
    for i in range(num_steps+1):
        cpdf_list = []
        cmean_list = []
        for j in range(i+1):
            cauchyEsts[j].step(zs[i], None)
            xhatj, _ = cauchyEsts[j].get_last_mean_cov()
            if(plot_oldest_newest):
                if (j == 0) or (j == i):
                    cmean_list.append(xhatj)
                    XYZ = cauchyEsts[j].get_2D_pointwise_cpdf(xhatj[0]-0.5, xhatj[0]+0.5, 0.004, xhatj[1]-0.5, xhatj[1]+0.5, 0.004)
                    cpdf_list.append(XYZ)
            else:
                cmean_list.append(xhatj)
                XYZ = cauchyEsts[j].get_2D_pointwise_cpdf(xhatj[0]-0.5, xhatj[0]+0.5, 0.004, xhatj[1]-0.5, xhatj[1]+0.5, 0.004)
                cpdf_list.append(XYZ)
            if (i > 0) and (j==0):
                _A0, _p0, _b0 = cauchyEsts[0].get_reinitialization_statistics()
                cauchyEsts[i].initialize_lti(_A0, _p0, _b0, Phi, None, Gamma, beta, H, gamma) 

        ce.plot_2D_pointwise_cpdfs(cpdf_list, cmean_list, colors[:i+1])
    '''
    cauchyEsts = [ ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True) for _ in range(2) ]
    cauchyEsts[0].initialize_lti(A0, p0, b0, Phi, None, Gamma, beta, H, gamma)
    cauchyEsts[0].step(zs[0], None)
    cauchyEsts[0].step(zs[1], None)
    cauchyEsts[0].step(zs[2], None)
    _A0, _p0, _b0 = cauchyEsts[0].get_reinitialization_statistics()
    cauchyEsts[1].initialize_lti(_A0, _p0, _b0, Phi, None, Gamma, beta, H, gamma)
    cauchyEsts[1].step(zs[2], None)
    xhat1, _ = cauchyEsts[0].get_last_mean_cov()
    XYZ1 = cauchyEsts[0].get_2D_pointwise_cpdf(xhat1[0]-0.5, xhat1[0]+0.5, 0.004, xhat1[1]-0.5, xhat1[1]+0.5, 0.004)
    xhat2, _ = cauchyEsts[1].get_last_mean_cov()
    XYZ2 = cauchyEsts[1].get_2D_pointwise_cpdf(xhat2[0]-0.5, xhat2[0]+0.5, 0.004, xhat2[1]-0.5, xhat2[1]+0.5, 0.004)
    cpdf_list = [XYZ1, XYZ2]
    cmean_list = [xhat1, xhat2]
    ce.plot_2D_pointwise_cpdfs(cpdf_list, cmean_list, colors[:2])

    for i in range(3,7):
        cauchyEsts[0].step(zs[i], None)
        cauchyEsts[1].step(zs[i], None)
    xhat1, _ = cauchyEsts[0].get_last_mean_cov()
    xhat2, _ = cauchyEsts[1].get_last_mean_cov()
    XYZ1 = cauchyEsts[0].get_2D_pointwise_cpdf(xhat1[0]-0.5, xhat1[0]+0.5, 0.004, xhat1[1]-0.5, xhat1[1]+0.5, 0.004)
    XYZ2 = cauchyEsts[1].get_2D_pointwise_cpdf(xhat2[0]-0.5, xhat2[0]+0.5, 0.004, xhat2[1]-0.5, xhat2[1]+0.5, 0.004)
    cpdf_list = [XYZ1, XYZ2]
    cmean_list = [xhat1, xhat2]
    ce.plot_2D_pointwise_cpdfs(cpdf_list, cmean_list, colors[:2])
    foo = 9

# Runs the estimator for a long estimation horizon, given the chosen window bank size
def test_3state_lti_window_manager():
    # x_{k+1} = \Phi_k @ x_k + B_k @ u_k + \Gamma_k @ w_k
    # z_k = H @ x_k + v_k
    ndim = 3
    Phi = np.array([ [1.4, -0.6, -1.0], 
                     [-0.2,  1.0,  0.5],  
                     [0.6, -0.6, -0.2]] )
    Gamma = np.array([.1, 0.3, -0.2])
    H = np.array([1.0, 0.5, 0.2])
    beta = np.array([0.1]) # Cauchy process noise scaling parameter(s)
    gamma = np.array([0.2]) # Cauchy measurement noise scaling parameter(s)
    A0 = np.eye(ndim) # Unit directions of the initial state uncertainty
    p0 = np.array([0.10, 0.08, 0.05]) # Initial state uncertainty cauchy scaling parameter(s)
    b0 = np.zeros(ndim) # Initial median of system state

    # Gaussian Noise Model Equivalent for KF
    W = np.diag( (beta * ce.CAUCHY_TO_GAUSSIAN_NOISE)**2 )
    V = np.diag( (gamma * ce.CAUCHY_TO_GAUSSIAN_NOISE)**2 )
    P0 = np.diag( (p0 * ce.CAUCHY_TO_GAUSSIAN_NOISE)**2 )

    np.random.seed(10)
    num_steps = 150

    # Simulate Dynamic System -- Either in Cauchy Noise or Gaussian Noise
    # Applying arbitrary controls to the estimator via 'B @ u'
    num_controls = 0
    B = np.random.randn(ndim,num_controls) if num_controls > 0 else  None
    us = np.random.randn(num_steps, num_controls) if num_controls > 0 else  None

    # Simulate system states and measurements
    x0_truth = p0 * np.random.randn(ndim)
    (xs, zs, ws, vs) = ce.simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)
    #(xs, zs, ws, vs) = ce.simulate_gaussian_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)

    # Run Cauchy Estimator
    num_windows = 8
    log_dir = file_dir + "/../../../log/python/swig_3state_lti"
    debug_print = True
    log_sequential = False 
    log_full = False
    cauchyEst = ce.PySlidingWindowManager("lti", num_windows, num_steps + 1, log_dir, debug_print, log_sequential, log_full)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma) 
    
    z0 = zs[0]
    u0 = np.zeros(num_controls) if num_controls > 0 else None 
    cauchyEst.step(z0, u0) # initial step has no propagation
    for i in range(1,len(zs)):
        zk1 = zs[i]
        uk = us[i-1] if num_controls > 0 else None 
        cauchyEst.step(zk1, uk)
    cauchyEst.shutdown()

    # Run Kalman Filter
    x0_kf = b0.copy() # start the KF at the median of the Cauchy
    zs_kf = zs[1:] # No measurement processed at initial step (however this could be added)
    xs_kf, Ps_kf = gf.run_kalman_filter(x0_kf, us, zs_kf, P0, Phi, B, Gamma, H, W, V)

    # Plot Cauchy (and Kalman Filter) performance in relation to the true simulation state history
    ce.plot_simulation_history( cauchyEst.moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf) )
    #foobar = 0 # hold plots up in debug mode by breaking here

# Runs the estimator for a long estimation horizon, given the chosen window bank size
def test_2state_lti_window_manager():
    # x_{k+1} = \Phi_k @ x_k + B_k @ u_k + \Gamma_k @ w_k
    # z_k = H @ x_k + v_k
    ndim = 2
    Phi = np.array([ [0.9, 0.1], [-0.2, 1.1] ])
    Gamma = np.array([.1, 0.3])
    H = np.array([1.0, 0.5])
    beta = np.array([0.1]) # Cauchy process noise scaling parameter(s)
    gamma = np.array([0.2]) # Cauchy measurement noise scaling parameter(s)
    A0 = np.eye(ndim) # Unit directions of the initial state uncertainty
    p0 = np.array([0.10, 0.08]) # Initial state uncertainty cauchy scaling parameter(s)
    b0 = np.zeros(ndim) # Initial median of system state

    # Gaussian Noise Model Equivalent for KF
    W = np.diag( (beta * ce.CAUCHY_TO_GAUSSIAN_NOISE)**2 )
    V = np.diag( (gamma * ce.CAUCHY_TO_GAUSSIAN_NOISE)**2 )
    P0 = np.diag( (p0 * ce.CAUCHY_TO_GAUSSIAN_NOISE)**2 )

    np.random.seed(15)
    num_steps = 150

    # Simulate Dynamic System -- Either in Cauchy Noise or Gaussian Noise
    # Applying arbitrary controls to the estimator via 'B @ u'
    num_controls = 0
    B = np.random.randn(ndim,num_controls) if num_controls > 0 else  None
    us = np.random.randn(num_steps, num_controls) if num_controls > 0 else  None
    x0_truth = p0 * np.random.randn(ndim)
    #(xs, zs, ws, vs) = ce.simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)
    (xs, zs, ws, vs) = ce.simulate_gaussian_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)

    # Run Cauchy Estimator
    num_windows = 6
    log_dir = file_dir + "/../../../log/python/swig_2state_lti"
    debug_print = True
    log_sequential = False 
    log_full = False
    cauchyEst = ce.PySlidingWindowManager("lti", num_windows, num_steps + 1, log_dir, debug_print, log_sequential, log_full)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma) 
    
    z0 = zs[0]
    u0 = np.zeros(num_controls) if num_controls > 0 else None 
    cauchyEst.step(z0, u0) # initial step has no propagation
    for i in range(1,len(zs)):
        zk1 = zs[i]
        uk = us[i-1] if num_controls > 0 else None 
        cauchyEst.step(zk1, uk)
    cauchyEst.shutdown()

    # Run Kalman Filter
    x0_kf = b0.copy() # start the KF at the median of the Cauchy
    zs_kf = zs[1:] # No measurement processed at initial step (however this could be added)
    xs_kf, Ps_kf = gf.run_kalman_filter(x0_kf, us, zs_kf, P0, Phi, B, Gamma, H, W, V)

    # Plot Cauchy (and Kalman Filter) performance in relation to the true simulation state history
    ce.plot_simulation_history( cauchyEst.moment_info, (xs, zs, ws, vs), (xs_kf, Ps_kf) )
    #foobar = 0 # hold plots up in debug mode by breaking here



if __name__ == "__main__":
    test_1state_lti()
    #test_2state_cpdfs()
    #test_2state_lti_single_window()
    #test_3state_lti_single_window()
    #test_2state_lti_window_manager()
    #test_3state_lti_window_manager()