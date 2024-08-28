import os
import numpy as np
import cauchy_estimator as ce
import gaussian_filters as gf
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg',force=True)

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
    H = np.array([1.0, 2.0])
    beta = np.array([0.1]) # Cauchy process noise scaling parameter(s)
    gamma = np.array([0.2]) # Cauchy measurement noise scaling parameter(s)
    A0 = np.eye(ndim) # Unit directions of the initial state uncertainty
    p0 = np.array([0.10, 0.08]) # Initial state uncertainty cauchy scaling parameter(s)
    b0 = np.zeros(ndim) # Initial median of system state

    np.random.seed(19)
    num_steps = 10

    # Applying (arbitrary) controls to the estimator via 'B @ u'
    num_controls = 0
    B = np.random.randn(ndim,num_controls) if num_controls > 0 else  None
    us = np.random.randn(num_steps, num_controls) if num_controls > 0 else  None
    
    # Simulate system states and measurements
    x0_truth = p0 * np.random.randn(ndim)
    (xs, zs, ws, vs) = ce.simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)
    
    # Run Cauchy Estimator
    ce.set_tr_search_idxs_ordering([1,0])
    cauchyEst = ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma, 0, 0)
    z0 = zs[0]
    u0 = np.zeros(num_controls) if num_controls > 0 else None 
    cauchyEst.step(z0, u0) # initial step has no propagation, and no "u" control
    for i in range(1,len(zs)):
        zk1 = zs[i]
        uk = us[i-1] if num_controls > 0 else None 
        cauchyEst.step(zk1, uk)
        X,Y,Z = cauchyEst.get_2D_pointwise_cpdf(-3,3,0.05, -3,3,0.05)
        cauchyEst.plot_2D_pointwise_cpdf(X,Y,Z)
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
    cauchyEsts[0].plot_2D_pointwise_cpdf(*XYZ1)
    cauchyEsts[1].plot_2D_pointwise_cpdf(*XYZ2)
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
    #(xs, zs, ws, vs) = ce.simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)
    (xs, zs, ws, vs) = ce.simulate_gaussian_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)

    # Run Cauchy Estimator
    num_windows = 8
    #log_dir = file_dir + "/../../../log/python/swig_3state_lti"
    swm_debug_print = True
    win_debug_print = False
    #log_sequential = False 
    #log_full = False
    cauchyEst = ce.PySlidingWindowManager("lti", num_windows, swm_debug_print, win_debug_print)
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
    foobar = 0 # hold plots up in debug mode by breaking here

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
    #log_dir = file_dir + "/../../../log/python/swig_2state_lti"
    swm_debug_print = True
    win_debug_print = False
    #log_sequential = False 
    #log_full = False
    cauchyEst = ce.PySlidingWindowManager("lti", num_windows, swm_debug_print, win_debug_print)
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
    foobar = 0 # hold plots up in debug mode by breaking here

# Runs the 3-state dummy problem and looks at their marginals
def test_3state_marginal_cpdfs():
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

    zs = [0.022172011200334241, -0.11943271347277583, -1.22353301003957098, 
            -1.4055389648301792, -1.34053610027255954, 0.4580483915838776, 
            0.65152999529515989, 0.52378648722334, 0.75198272983]
    num_steps = len(zs)
    
    # 2D Grid Params
    g2lx = -2
    g2hx = 2
    g2rx = 0.025
    g2ly = -2
    g2hy = 2
    g2ry = 0.025

    # 1D Grid Params
    g1lx = -4
    g1hx = 4
    g1rx = 0.001
    
    # Testing Single Cauchy Estimator Instance
    cauchyEst = ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, None, Gamma, beta, H, gamma)
    for i in range(len(zs)-1):
        zk1 = zs[i]
        xs, Ps = cauchyEst.step(zk1, None)
        X01, Y01, Z01 = cauchyEst.get_marginal_2D_pointwise_cpdf(0, 1, g2lx, g2hx, g2rx, g2ly, g2hy, g2ry)
        X02, Y02, Z02 = cauchyEst.get_marginal_2D_pointwise_cpdf(0, 2, g2lx, g2hx, g2rx, g2ly, g2hy, g2ry)
        X12, Y12, Z12 = cauchyEst.get_marginal_2D_pointwise_cpdf(1, 2, g2lx, g2hx, g2rx, g2ly, g2hy, g2ry)
        x0, y0 = cauchyEst.get_marginal_1D_pointwise_cpdf(0, g1lx, g1hx, g1rx)
        x1, y1 = cauchyEst.get_marginal_1D_pointwise_cpdf(1, g1lx, g1hx, g1rx)
        x2, y2 = cauchyEst.get_marginal_1D_pointwise_cpdf(2, g1lx, g1hx, g1rx)

        # set up a figure three times as wide as it is tall
        fig1 = plt.figure(figsize = (18,5))
        ax12 = fig1.add_subplot(1,3,1,projection='3d')
        ax13 = fig1.add_subplot(1,3,2,projection='3d')
        ax23 = fig1.add_subplot(1,3,3,projection='3d')
        plt.tight_layout()
        # Marg 2D
        # Marg (0,1)
        ax12.set_title("Marginal of States 1 and 2", pad=-15)
        ax12.plot_wireframe(X01, Y01, Z01, zorder=2, color='b')
        ax12.set_xlabel("x-axis (State-1)")
        ax12.set_ylabel("y-axis (State-2)")
        ax12.set_zlabel("z-axis (CPDF Probability)")
        # Marg (0,2)
        ax13.set_title("Marginal of States 1 and 3", pad=-8)
        ax13.plot_wireframe(X02, Y02, Z02, zorder=2, color='g')
        ax13.set_xlabel("x-axis (State-1)")
        ax13.set_ylabel("y-axis (State-3)")
        ax13.set_zlabel("z-axis (CPDF Probability)")
        # Marg (1,2)
        ax23.set_title("Marginal of States 2 and 3", pad=-8)
        ax23.plot_wireframe(X12, Y12, Z12, zorder=2, color='r')
        ax23.set_xlabel("x-axis (State-2)")
        ax23.set_ylabel("y-axis (State-3)")
        ax23.set_zlabel("z-axis (CPDF Probability)")

        # set up a figure three times as wide as it is tall
        fig2 = plt.figure(figsize = (18,4))
        ax1 = fig2.add_subplot(1,3,1)
        ax2 = fig2.add_subplot(1,3,2)
        ax3 = fig2.add_subplot(1,3,3)
        # Marg 1D
        # Marg 1
        ax1.set_title("1D Marg of State 1")
        ax1.plot(x0,y0)
        ax1.set_xlabel("State 1")
        ax1.set_ylabel("CPDF Probability")
        # Marg 2
        ax2.set_title("1D Marg of State 2")
        ax2.plot(x1,y1)
        ax2.set_xlabel("State 2")
        ax2.set_ylabel("CPDF Probability")
        # # Marg 3
        ax3.set_title("1D Marg of State 3")
        ax3.plot(x2,y2)
        ax3.set_xlabel("State 3")
        ax3.set_ylabel("CPDF Probability")
        plt.show()
        plt.close()
        
# Runs the 3-state dummy problem and looks at their marginals
def test_3state_reset():
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

    zs = [0.022172011200334241, -0.11943271347277583]
    num_steps = len(zs)
    
    # Testing Single Cauchy Estimator Instance
    cauchyEst = ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, None, Gamma, beta, H, gamma)

    cauchyEst.step(zs[0])
    cauchyEst.step(zs[1])

    cauchyEst2 = ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True)
    cauchyEst2.initialize_lti(A0, p0, b0, Phi, None, Gamma, beta, H, gamma)
    # Automatic
    cauchyEst2.reset_about_estimator(cauchyEst)
    # Manual
    #A0,p0,b0 = cauchyEst.get_reinitialization_statistics()
    #cauchyEst2.initialize_lti(A0, p0, b0, Phi, None, Gamma, beta, H, gamma)
    #cauchyEst2.step(zs[1])
    foobar = 2

# Creates a single cauchy instance and then 
def test_2state_smoothing():
    # x_{k+1} = \Phi_k @ x_k + B_k @ u_k + \Gamma_k @ w_k
    # z_k = H @ x_k + v_k
    ndim = 2
    Phi = np.array([ [0.9, 0.1], [-0.2, 1.1] ])
    Gamma = np.array([.1, 0.3])
    H = np.array([1.0, 2.0])
    beta = np.array([0.1]) # Cauchy process noise scaling parameter(s)
    gamma = np.array([0.2]) # Cauchy measurement noise scaling parameter(s)
    A0 = np.eye(ndim) # Unit directions of the initial state uncertainty
    p0 = np.array([0.10, 0.08]) # Initial state uncertainty cauchy scaling parameter(s)
    b0 = np.zeros(ndim) # Initial median of system state

    np.random.seed(1)
    num_steps = 7

    # Applying (arbitrary) controls to the estimator via 'B @ u'
    num_controls = 0
    B = np.random.randn(ndim,num_controls) if num_controls > 0 else  None
    us = np.random.randn(num_steps, num_controls) if num_controls > 0 else  None
    
    # Simulate system states and measurements
    x0_truth = p0 * np.random.randn(ndim)
    (xs, zs, ws, vs) = ce.simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma, with_zeroth_step_msmt=True, dynamics_update_callback=None, other_params=None)
    
    cauchyEst = ce.PyCauchyEstimator("lti", num_steps+1, debug_print=True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma, 0, 0)
    xs_ce = [] 
    Ps_ce = [] 
    for i in range(num_steps+1):
        x_ce, P_ce = cauchyEst.step(zs[i], None) # Step over all measurements simultaneously 
        xs_ce.append(x_ce)
        Ps_ce.append(P_ce)
    xs_ce = np.array(xs_ce)
    Ps_ce = np.array(Ps_ce)
    cauchyEst.shutdown()

    # Run Cauchy Smoother
    ndim = ndim + num_steps
    A0 = np.eye(ndim) # Unit directions of the initial state uncertainty
    p0 = np.concatenate(( p0, np.repeat(beta, num_steps) )) # Initial state uncertainty cauchy scaling parameter(s)
    b0 = np.zeros(ndim) # Initial median of system state
    beta = np.array([]) # Cauchy process noise scaling parameter(s)
    gamma = np.repeat(gamma, num_steps+1) # Cauchy measurement noise scaling parameter(s)
    H_smooth = np.zeros((num_steps+1, ndim))
    n = 2
    p = 1
    r = 1
    for i in range(num_steps+1):
        for j in range(i+1):
            if j == 0:
                H_smooth[i*p:(i+1)*p, j*n:(j+1)*n] = H @ np.linalg.matrix_power(Phi, i)
            else:
                H_smooth[i*p : (i+1)*p, n+(j-1)*r : n+j*r] = H @ np.linalg.matrix_power(Phi, i-j) @ Gamma 
    _Phi = Phi 
    _Gamma = Gamma 
    Phi = np.eye(ndim) # not used
    Gamma = np.zeros((ndim,0))

    #H_smooth = np.flip(H_smooth, axis = 0)
    #zs = np.flip(zs)
    #rev_idxs = list(reversed(np.arange(ndim)))
    #ce.set_tr_search_idxs_ordering( rev_idxs )

    cauchyEst = ce.PyCauchyEstimator("lti", 1, debug_print=True)
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H_smooth, gamma, 0, 0)
    cauchyEst.step(zs, None) # Step over all measurements simultaneously 
    cauchyEst.shutdown()
    x_hat_large, P_hat_large = cauchyEst.get_last_mean_cov()
    print("State History:\n", xs)
    print("Process Noise History:\n", ws)
    print("Large state est vector:", x_hat_large)
    # Turn Smoother state vector into smoothed set of state estimates
    xk = x_hat_large[0:n]
    x_hats = [xk]
    P11 = P_hat_large[0:2,0:2]
    P_hat = P11.copy()
    P_hats = [P11] 
    n = 2
    pncc = 1
    G = _Gamma.reshape((n,pncc))
    from numpy.linalg import matrix_power as MPOW
    for idx in range(num_steps):
        # Propagation of the conditional smoothed mean
        xk = _Phi @ xk + _Gamma * x_hat_large[n+idx]
        # Propagation of the conditional smoothed covariance
        k = idx+2
        km1 = k-1
        km1Phi = MPOW(_Phi, km1)
        t1 = km1Phi @ P11 @ km1Phi.T 
        t2 = np.zeros((2,2))
        t3 = np.zeros((2,2))
        t4 = np.zeros((2,2))
        for i in range(1,k):
            wi_idx_start = n+(i-1)*pncc
            wi_idx_stop = wi_idx_start + pncc
            Pwix1 = P_hat_large[wi_idx_start:wi_idx_stop,0:n]
            Px1wi = Pwix1.T
            kmim1Phi = MPOW(_Phi, k-i-1)
            _t = km1Phi @ Px1wi @ G.T @ kmim1Phi.T
            t2 += _t 
            t3 += _t.T #kmim1Phi @ G @ Pwix1 @ km1Phi.T
        for i in range(1,k):
            wi_idx_start = n+(i-1)*pncc
            wi_idx_stop = wi_idx_start + pncc
            for j in range(1,k):
                wj_idx_start = n+(j-1)*pncc
                wj_idx_stop = wj_idx_start + pncc
                Pwiwj = P_hat_large[wi_idx_start:wi_idx_stop,wj_idx_start:wj_idx_stop]
                kmim1Phi = MPOW(_Phi, k-i-1)
                kmjm1Phi = MPOW(_Phi, k-j-1)
                t4 += kmim1Phi @ G @ Pwiwj @ G.T @ kmjm1Phi.T
        P_hat = t1 + t2 + t3 + t4
        # Store
        x_hats.append(xk)
        P_hats.append(P_hat)
    x_hats = np.array(x_hats) 
    P_hats = np.array(P_hats) 
    print("Estimator Means:\n", xs_ce)
    print("Estimator Covs:\n", Ps_ce)

    print("Smoothed x_hats:\n", x_hats)
    print("Smoothed Covs:\n", P_hats)

    print("Last Est Cov:\n", Ps_ce[-1])
    print("Last Smoothed Cov:\n", P_hats[-1])

if __name__ == "__main__":
    #test_1state_lti()
    #test_2state_cpdfs()
    #test_2state_lti_single_window()
    #test_3state_lti_single_window()
    #test_2state_lti_window_manager()
    test_3state_lti_window_manager()
    #test_3state_marginal_cpdfs()
    #test_3state_reset()
    #test_2state_smoothing()