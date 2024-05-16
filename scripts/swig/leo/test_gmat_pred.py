from cProfile import run
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg',force=True)
import cauchy_estimator as ce 
import math 
import copy 
import time
import sys, os, pickle
from datetime import datetime 
from datetime import timedelta 
from gmat_sat import *
import pc_prediction as pc 
import gmat_mce as gmce

def find_cross_radial_error_mc_vs_density():
    # Given an initial condition, simulate out different realizations of change in atms density and look at the monte carlo avg of error vs nominal (no density change) over a week lookahead
    x0 = np.array([4996.245288270519, 3877.946463086103, 2736.0432364171807, -5.028093574446193, 5.575921341999267, 1.2698611722905329])
    P0 = np.array([ [ 1.05408888e-05, -8.97284021e-06, -1.89319050e-06, 1.10789874e-08,  6.77331750e-09,  5.31534425e-09],
                    [-8.97284021e-06,  9.49191574e-06,  1.31370671e-06, -9.52652261e-09, -7.09634679e-09, -5.12992550e-09],
                    [-1.89319050e-06,  1.31370671e-06,  1.91941294e-06, -1.62654495e-09, -1.54468399e-09, -9.88335840e-10],
                    [ 1.10789874e-08, -9.52652261e-09, -1.62654495e-09, 1.21566585e-11,  7.54301726e-12,  5.41815567e-12],
                    [ 6.77331750e-09, -7.09634679e-09, -1.54468399e-09, 7.54301726e-12,  6.14186208e-12,  3.36503199e-12],
                    [ 5.31534425e-09, -5.12992550e-09, -9.88335840e-10, 5.41815567e-12,  3.36503199e-12,  4.76717851e-12]])
    t0 = '11 Feb 2023 23:47:55.0'
    dt = 120.0
    mc_trials = 30
    days_lookahead = 7
    mode = 'gauss'
    orbit_period = 2*np.pi*np.linalg.norm(x0[0:3])/np.linalg.norm(x0[3:6]) # seconds in orbit
    prop_time = (days_lookahead * 24 * 60 * 60) # seconds in days_lookahead days 
    num_orbits = int( np.ceil( prop_time / orbit_period) )
    total_steps = int( prop_time / dt )
    step_lookouts = np.array( [ int( (i*24*60*60)/dt ) -1 for i in range(1,days_lookahead+1)] )


    # Take KF and get filtered result after a day 
    dir_path = file_dir + "/pylog/gmat7/pred/" + "mcdata_sastrials_2_1713828844.pickle"
    print("Reading MC Data From: ", dir_path)
    with open(dir_path, "rb") as handle:
        mc_dic = pickle.load(handle)
    start_idx = 1000
    xs_kf,Ps_kf = mc_dic["ekf_runs"][0]
    # Now store cov of estimator at the end of filter period
    x0 = xs_kf[start_idx][0:6]
    P0 = Ps_kf[start_idx][0:6,0:6]

    # Now propagate the state estimate days_lookahead days into the future assuming no atms. density change 
    fermiSat = FermiSatelliteModel(t0,x0,dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    xkf_preds = [] 
    Pkf_preds = [] 
    P_kf = P0.copy()
    for idx in range(total_steps):
        Phi = fermiSat.get_transition_matrix(mc_dic["STM_order"])
        x_kf = fermiSat.step()
        P_kf = Phi @ P_kf @ Phi.T
        xkf_preds.append(x_kf)
        Pkf_preds.append(P_kf)
    xkf_preds = np.array(xkf_preds)
    Pkf_preds = np.array(Pkf_preds)
    # Store xhats each day
    xbar_lookouts = xkf_preds[step_lookouts]
    Pbar_lookouts = Pkf_preds[step_lookouts]
    Ts = [pc.get_along_cross_radial_transformation(x) for x in xbar_lookouts]
    Pbar_Ts = np.array([T @ P[0:3,0:3] @ T.T for T,P in zip(Ts,Pbar_lookouts)])
    std_dev_Pbar_Ts = np.array([np.diag(P)**0.5 for P in Pbar_Ts])
    fermiSat.clear_model()

    # Now use a monte carlo and propagate the state estimate days_lookahead days into the future assuming atms. denisty change on the given distribution
    mc_data = np.zeros((mc_trials, len(step_lookouts), 6))
    fermiSat = FermiSatelliteModel(t0,x0,dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    fermiSat.set_solve_for(field="Cd", dist=mode, scale = 0.0013, tau = 21600, alpha = 2.0 if mode == "gauss" else 1.3)
    for i in range(mc_trials):
        print("Finished Trial", i)
        xks, _, _, ws = fermiSat.simulate(num_orbits, 0, None, False)
        mc_data[i, :,:] = xks[step_lookouts+1,0:6]
        x07 = np.concatenate((x0,np.array([0.0])))
        fermiSat.reset_initial_state(x07)
    # See via MC what the error bound is for cross and radial error after one day to seven days vs. Kalman filter 
    mc_data -= xbar_lookouts#.reshape((1,xbar_lookouts.size,1))
    mc_data = mc_data.transpose((1,2,0))
    Ppred_mcs = [np.cov(mc_data[i])[0:3,0:3] for i in range(days_lookahead)]
    Ts = [pc.get_along_cross_radial_transformation(x) for x in xbar_lookouts]
    Ppred_Ts = np.array([T @ P @ T.T for T,P in zip(Ts,Ppred_mcs)])
    std_dev_Ppred_Ts = np.array([np.diag(P)**0.5 for P in Ppred_Ts])

    # Now plot the covariances as a function of look ahead and vs KF
    plt.suptitle("Along Cross Radial (ACR)-Track KF Variance Projected 7 days (b) vs 7-day ACR-Track MC Variance (r)")
    plt.subplot(311)
    plt.plot(np.arange(days_lookahead)+1, std_dev_Pbar_Ts[:,0], 'b')
    plt.plot(np.arange(days_lookahead)+1, std_dev_Ppred_Ts[:,0], 'r')
    plt.subplot(312)
    plt.plot(np.arange(days_lookahead)+1, std_dev_Pbar_Ts[:,1], 'b')
    plt.plot(np.arange(days_lookahead)+1, std_dev_Ppred_Ts[:,1], 'r')
    plt.subplot(313)
    plt.plot(np.arange(days_lookahead)+1, std_dev_Pbar_Ts[:,2], 'b')
    plt.plot(np.arange(days_lookahead)+1, std_dev_Ppred_Ts[:,2], 'r')
    plt.show()
    foobar=2

def test_sat_pc():
    x0 = np.array([4996.245288270519, 3877.946463086103, 2736.0432364171807, -5.028093574446193, 5.575921341999267, 1.2698611722905329])
    P0 = np.array([ [ 1.05408888e-05, -8.97284021e-06, -1.89319050e-06, 1.10789874e-08,  6.77331750e-09,  5.31534425e-09],
                    [-8.97284021e-06,  9.49191574e-06,  1.31370671e-06, -9.52652261e-09, -7.09634679e-09, -5.12992550e-09],
                    [-1.89319050e-06,  1.31370671e-06,  1.91941294e-06, -1.62654495e-09, -1.54468399e-09, -9.88335840e-10],
                    [ 1.10789874e-08, -9.52652261e-09, -1.62654495e-09, 1.21566585e-11,  7.54301726e-12,  5.41815567e-12],
                    [ 6.77331750e-09, -7.09634679e-09, -1.54468399e-09, 7.54301726e-12,  6.14186208e-12,  3.36503199e-12],
                    [ 5.31534425e-09, -5.12992550e-09, -9.88335840e-10, 5.41815567e-12,  3.36503199e-12,  4.76717851e-12]])
    t0 = '11 Feb 2023 23:47:55.0'
    dt = 60.0
    # Construct start conditions for primary
    x0_prim = x0.copy()
    R_prim = 0.003 #km
    P0_prim = P0.copy()
    # Construct start conditions for secondary
    x0_sec = x0.copy()
    x0_sec[3:] *= -1.0
    R_sec = 0.003 #km
    P0_sec = P0.copy()
    # Number of steps
    steps = int(55*(60/dt) + 0.50)

    # Create Satellite Model for Primary and Propagate 
    fermi_t0 = "{} {} {} {}:{}:{}.{}".format(t0.day, MonthDic2[t0.month], t0.year,t0.hour,t0.minute,t0.second, int(t0.microsecond/1000) )
    fermiSat = FermiSatelliteModel(t0, x0_prim, dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    p_pks = [] # Primary Pos 3-Vec
    p_vks = [] # Primary Vel 3-Vec
    p_aks = [] # Primary Acc 3-Vec
    p_Pks = []
    # Propagate Primary and Store
    xk = x0_prim.copy()
    Pk = P0_prim.copy()
    for i in range(steps):
        dxk_dt = fermiSat.get_state6_derivatives() 
        p_pks.append(xk[0:3])
        p_vks.append(xk[3:6])
        p_aks.append(dxk_dt[3:6])
        p_Pks.append(Pk)
        xk = fermiSat.step()
        #Phik = fermiSat.get_transition_matrix(taylor_order=3)
        #Pk = Phik @ Pk @ Phik.T
    p_pks = np.array(p_pks)
    p_vks = np.array(p_vks)
    p_aks = np.array(p_aks)
    p_Pks = np.array(p_Pks)
    fermiSat.clear_model()

    # Create Satellite Model for Secondary and Propagate
    fermiSat = FermiSatelliteModel(t0, x0_sec, dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    s_pks = [] # Secondary Pos 3-Vec
    s_vks = [] # Secondary Vel 3-Vec
    s_aks = [] # Secondary Acc 3-Vec
    s_Pks = []
    # Propagate Secondary and Store
    xk = x0_sec.copy()
    Pk = P0_sec.copy()
    for i in range(steps):
        dxk_dt = fermiSat.get_state6_derivatives() 
        s_pks.append(xk[0:3])
        s_vks.append(xk[3:6])
        s_aks.append(dxk_dt[3:6])
        s_Pks.append(Pk)
        xk = fermiSat.step()
        #Phik = fermiSat.get_transition_matrix(taylor_order=3)
        #Pk = Phik @ Pk @ Phik.T
    s_pks = np.array(s_pks)
    s_vks = np.array(s_vks)
    s_aks = np.array(s_aks)
    s_Pks = np.array(s_Pks)
    fermiSat.clear_model()

    # Now test the closest approach method
    start_idx = 25
    
    '''
    tks = dt * np.arange(steps)
    i, troot, t_c, pp_c, sp_c = pc.closest_approach_info(tks[start_idx:], 
        (p_pks[start_idx:,:],p_vks[start_idx:,:],p_aks[start_idx:,:]), 
        (s_pks[start_idx:,:],s_vks[start_idx:,:],s_aks[start_idx:,:]))
    i += start_idx
    print("Step dt: ", dt)
    print("Tc: {}, Idx of Tc: {}".format(t_c, i) )
    print("Primary at Tc: ", pp_c)
    print("Secondary at Tc: ", sp_c)
    print("Pos Diff is: ", pp_c-sp_c)
    print("Pos Norm is: ", np.linalg.norm(pp_c-sp_c))
    # Plot orbits of primary and secondary
    fig = plt.figure() 
    ax = fig.gca(projection='3d')
    plt.title("Leo Trajectory over Time")
    ax.plot(p_pks[:,0], p_pks[:,1], p_pks[:,2], color = 'r')
    ax.plot(s_pks[:,0], s_pks[:,1], s_pks[:,2], color = 'b')
    # Plot relative difference in position
    fig2 = plt.figure()
    plt.title("Pos Norm Diff over Time")
    #plt.plot(tks, p_pks[:,0] - s_pks[:,0], 'r')
    #plt.plot(tks, p_pks[:,1] - s_pks[:,1], 'g')
    #plt.plot(tks, p_pks[:,2] - s_pks[:,2], 'b')
    plt.plot(tks, np.linalg.norm(p_pks-s_pks,axis=1))
    plt.show()
    foo=3
    '''
    #'''
    # GMAT iterative closest time of approach
    i_star_lhs, t_lhs, t_c, pp_c, sp_c = pc.iterative_time_closest_approach(
        dt, t0, 
        (p_pks,p_vks,p_aks), 
        (s_pks,s_vks,s_aks), 
        start_idx = start_idx,
        with_plot=False
        )
    #'''

    # Propagate the Primary Covariance to Point of Closest Approach
    x0_prim = np.concatenate((p_pks[i_star_lhs],p_vks[i_star_lhs]))
    fermiSat = FermiSatelliteModel(t0 + timedelta(t_lhs), x0_prim, t_c - t_lhs)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    p_Phi = fermiSat.get_transition_matrix(taylor_order=3)
    p_Ptc = p_Phi @ p_Pks[i_star_lhs] @ p_Phi.T
    p_xtc = fermiSat.step()
    print("XPrim at TCA: ", p_xtc[0:3])
    print("Diff XPrim at TCA (meters): ", 1000*(pp_c - p_xtc[0:3]) )
    fermiSat.clear_model()

    # Propagate the Secondary Covariance to Point of Closest Approach 
    x0_sec = np.concatenate((s_pks[i_star_lhs],s_vks[i_star_lhs]))
    fermiSat = FermiSatelliteModel(t0 + timedelta(t_lhs), x0_sec, t_c - t_lhs)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    s_Phi = fermiSat.get_transition_matrix(taylor_order=3)
    s_Ptc = s_Phi @ s_Pks[i_star_lhs] @ s_Phi.T
    s_xtc = fermiSat.step()
    print("XSec at TCA: ", s_xtc[0:3])
    print("Diff XSec at TCA (meters): ", 1000*(sp_c - s_xtc[0:3]) )
    fermiSat.clear_model()


    # Form Relative System and Project onto Encounter Plane
    rx_tc = s_xtc - p_xtc
    rP_tc = s_Ptc + p_Ptc
    rR = R_prim + R_sec
    
    # plane normal to relative velocity vector 
    _,_,T1 = np.linalg.svd(rx_tc[3:].reshape((1,3)))
    T1 = np.hstack((T1[1:], np.zeros((2,3))))
    rx_ep = T1 @ rx_tc # mean 
    rP_ep = T1 @ rP_tc @ T1.T #variance 
    rxx = rx_ep[0]
    rxy = rx_ep[1]

    
    # Possibly another way to do this 
    T2 = pc.get_along_cross_radial_transformation(s_xtc)
    T2 = np.hstack((T2[1:,:], np.zeros((2,3))))
    rx_ep2 = T2 @ rx_tc # mean
    rP_ep2 = T2 @ rP_tc @ T2.T # variance 

    # Take Integral over 2D projection 
    int_coeff = 1.0 / (2.0*np.pi * np.linalg.det(rP_ep)**0.5 )
    int_PI = np.linalg.inv(rP_ep)

    from scipy.integrate import dblquad
    area = dblquad(lambda x, y: int_coeff*np.exp(-1.0/2.0 * np.array([x-rxx,y-rxy]) @ int_PI @  np.array([x-rxx,y-rxy]) ), -rR, rR, lambda x: -(rR**2-x**2), lambda x: (rR**2-x**2) )
    print("Prob Collision Stat is: ", area)

    foobar = 3

def test_sat_crossing():
    # Load out data 
    t0 = '11 Feb 2023 23:47:55.0'
    x0 = np.array([550+6378, 0, 0, 0, 7.585175924227056, 0])
    dt = 120.0
    # Construct start conditions for primary
    x0_prim = x0.copy()
    R_prim = 0.003 #km
    #P0_prim = P0.copy()
    # Construct start conditions for secondary
    x0_sec = x0.copy()
    x0_sec[3:] *= -1.0
    R_sec = 0.003 #km
    # Number of steps
    steps = 5040

    # Create Satellite Model for Primary and Propagate 
    fermiSat = FermiSatelliteModel(t0, x0_prim, dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    p_pks = [] # Primary Pos 3-Vec
    p_vks = [] # Primary Vel 3-Vec
    p_aks = [] # Primary Acc 3-Vec
    #p_Pks = []
    # Propagate Primary and Store
    xk = x0_prim.copy()
    #Pk = P0_prim.copy()
    for i in range(steps):
        dxk_dt = fermiSat.get_state6_derivatives() 
        p_pks.append(xk[0:3])
        p_vks.append(xk[3:6])
        p_aks.append(dxk_dt[3:6])
        #p_Pks.append(Pk)
        xk = fermiSat.step()
        #Phik = fermiSat.get_transition_matrix(taylor_order=3)
        #Pk = Phik @ Pk @ Phik.T
    p_pks = np.array(p_pks)
    p_vks = np.array(p_vks)
    p_aks = np.array(p_aks)
    #p_Pks = np.array(p_Pks)
    fermiSat.clear_model()

    # Create Satellite Model for Secondary and Propagate
    fermiSat = FermiSatelliteModel(t0, x0_sec, dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    s_pks = [] # Secondary Pos 3-Vec
    s_vks = [] # Secondary Vel 3-Vec
    s_aks = [] # Secondary Acc 3-Vec
    s_Pks = []
    # Propagate Secondary and Store
    xk = x0_sec.copy()
    #Pk = P0_sec.copy()
    for i in range(steps):
        dxk_dt = fermiSat.get_state6_derivatives() 
        s_pks.append(xk[0:3])
        s_vks.append(xk[3:6])
        s_aks.append(dxk_dt[3:6])
        #s_Pks.append(Pk)
        xk = fermiSat.step()
        #Phik = fermiSat.get_transition_matrix(taylor_order=3)
        #Pk = Phik @ Pk @ Phik.T
    s_pks = np.array(s_pks)
    s_vks = np.array(s_vks)
    s_aks = np.array(s_aks)
    #s_Pks = np.array(s_Pks)
    fermiSat.clear_model()

    #'''
    # Plot Trajectories of both satellites 
    fig = plt.figure() 
    ax = fig.gca(projection='3d')
    plt.title("Leo Trajectory over Time")
    ax.plot(p_pks[:,0], p_pks[:,1], p_pks[:,2], color = 'r')
    ax.plot(s_pks[:,0], s_pks[:,1], s_pks[:,2], color = 'b')
    ax.scatter(p_pks[0,0], p_pks[0,1], p_pks[0,2], color = 'k', s=80)
    fig2 = plt.figure()
    r_norms = np.linalg.norm(s_pks - p_pks, axis=1)
    plt.plot(np.arange(steps), r_norms)
    plt.show()
    foobar = 5
    #'''

    # Now test the closest approach method
    #'''
    start_idx = 4920
    end_idx = 4950 #p_pks.shape[0]
    # GMAT iterative closest time of approach
    t0 =  datetime.strptime(t0, "%d %b %Y %H:%M:%S.%f")
    i_star_lhs, t_lhs, t_c, pp_c, sp_c = pc.iterative_time_closest_approach(
        dt, t0, 
        (p_pks[:end_idx],p_vks[:end_idx],p_aks[:end_idx]), 
        (s_pks[:end_idx],s_vks[:end_idx],s_aks[:end_idx]), 
        start_idx = start_idx,
        with_plot=False
        )
    
    # Propagate the Primary Covariance to Point of Closest Approach
    x0_prim = np.concatenate((p_pks[i_star_lhs],p_vks[i_star_lhs]))
    fermiSat = FermiSatelliteModel(t0 + timedelta(seconds = t_lhs), x0_prim, t_c - t_lhs)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    #p_Phi = fermiSat.get_transition_matrix(taylor_order=3)
    #p_Ptc = p_Phi @ p_Pks[i_star_lhs] @ p_Phi.T
    p_xtc = fermiSat.step()
    print("XPrim at TCA: ", p_xtc[0:3])
    print("Diff XPrim at TCA (meters): ", 1000*(pp_c - p_xtc[0:3]) )
    fermiSat.clear_model()
    
    # Propagate the Secondary Covariance to Point of Closest Approach 
    x0_sec = np.concatenate((s_pks[i_star_lhs],s_vks[i_star_lhs]))
    fermiSat = FermiSatelliteModel(t0 + timedelta(seconds = t_lhs), x0_sec, t_c - t_lhs)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    s_xtc = fermiSat.step()
    print("XSec at TCA: ", s_xtc[0:3])
    print("Diff XSec at TCA (meters): ", 1000*(sp_c - s_xtc[0:3]) )
    fermiSat.clear_model()
    #'''

    # Now draw 

    '''
    # Form Relative System and Project onto Encounter Plane, Compute Prob
    rx_tc = s_xtc - p_xtc
    rP_tc = s_Ptc + p_Ptc
    rR = R_prim + R_sec
    '''

def test_sat_pc_mc():
    n = 7 # states
    data_path = file_dir + "/pylog/gmat7/pred/pc/"
    
    # Runtime Options
    mode = "sas" #"sas"
    # "bar.pickle" #"foo.pickle"
    cached_dir = "" #"sas_okay.pickle" # "sas_okay.pickle" #mode + "_realiz.pickle" #"sas_realiz.pickle" #"gauss_realiz.pickle" # "" # if set to something, loads and adds to this dir, if set to nothing, creates a new directory 
    with_filter_plots = False
    with_pred_plots = False
    with_mc_from_x0 = False # True
    with_density_jumps = False

    if cached_dir is "":
        # Some Initial Data and Declarations
        t0 = '11 Feb 2023 23:47:55.0'
        #x0 = np.array([550+6378, 0, 0, 0, 7.585175924227056, 0])
        
        # replacing x0 with 
        '''
        r_sat = 550e3
        r_earth = 6378.1e3
        M = 5.9722e24 # Mass of earth (kg)
        G = 6.674e-11 # m^3/(s^2 * kg) Universal Gravitation Constant
        mu = M*G  #Nm^2/kg^2
        #rho = lookup_air_density(r_sat)
        r0 = r_earth + r_sat # orbit distance from center of earth
        v0 = np.sqrt(mu/r0) # speed of the satellite in orbit for distance r0
        r0 /= 1000 
        v0 /= 1000
        x0 = np.array([r0/np.sqrt(3), r0/np.sqrt(3), r0/np.sqrt(3), -0.57735027*v0, 0.78867513*v0, -0.21132487*v0])
        '''
        x0 = np.array([4996.245288270519, 3877.946463086103, 2736.0432364171807, -5.028093574446193, 5.575921341999267, 1.2698611722905329])

        filt_dt = 60
        filt_orbits = 12
        pred_dt = 120.0
        pred_steps = 5040
        R_prim = 0.003 #km
        R_sec = 0.003 #km
        std_gps_noise = .0075 # kilometers
        std_Cd = 0.0013
        tau_Cd = 21600
        sas_Cd = mode

        cache_dic = {
            'mode' : mode,
            'with_density_jumps' : with_density_jumps,
            'R_prim' : R_prim, 'R_sec' : R_sec,
            't0' : t0,
            'x0' : x0,
            'pred_steps' : pred_steps,
            'std_gps_noise' : std_gps_noise,
            'filt_orbits' : filt_orbits,
            'std_Cd' : std_Cd,
            'tau_Cd' : tau_Cd,
            'x0_P0_prim' : None,
            'x0_P0_sec' : None,
            'kf_prim_sim' : None,
            'kf_sec_sim' : None,
            'kf_prim' : None,
            'kf_sec' : None,
            'filt_dt' : filt_dt,
            'prim_pred_hist' : None,
            'sec_pred_hist' : None,
            'pred_dt' : pred_dt,
            'itca_window_idxs' : None,
            'itca_data' : None, 
            'nom_prim_tca' : None,
            'nom_sec_tca' : None,
            'mc_prim_tcas' : None, 
            'mc_sec_tcas' : None,
            'mc_prim_sample_tcas' : None,
            'mc_sec_sample_tcas' : None,
            # MCE Cache
            'mce_prim' : None,
            'mce_sec' : None,

        }
    else:
        with open(data_path + cached_dir, 'rb') as handle:
            cache_dic = pickle.load(handle)
        t0 = cache_dic['t0']
        x0 = cache_dic['x0']
        filt_dt = cache_dic['filt_dt']
        filt_orbits = cache_dic['filt_orbits']
        pred_dt = cache_dic['pred_dt']
        pred_steps = cache_dic['pred_steps']
        R_prim = cache_dic['R_prim']
        R_sec = cache_dic['R_sec']
        std_gps_noise = cache_dic['std_gps_noise']
        std_Cd = cache_dic['std_Cd']
        tau_Cd = cache_dic['tau_Cd']
        mode = cache_dic['mode']
        sas_Cd = mode
    
    # Process Noise Model
    W6 = gmce.leo6_process_noise_model2(filt_dt)
    Wn = np.zeros((n,n))
    if sas_Cd == "gauss":
        scale_pv = 1000
        scale_d = 20.0
        sas_alpha = 2.0
    else:
        scale_pv = 10000
        scale_d = 10000
        sas_alpha = 1.3
        #scale_pv = 500
        #scale_d = 250
    Wn[0:6,0:6] = W6.copy()
    Wn[0:6,0:6] *= scale_pv
    # Process Noise for changes in Cd
    if sas_Cd != "gauss":
        Wn[6,6] = (1.3898 * std_Cd)**2 # tune to cauchy LSF
    else:
        Wn[6,6] = std_Cd**2
    Wn[6,6] *= scale_d #0 # Tunable w/ altitude
    V = np.eye(3) * std_gps_noise**2
    I7 = np.eye(7)
    H = np.hstack((np.eye(3), np.zeros((3,4))))
    STM_order = 3

    # Filtering for primary satellite
    if cached_dir is "":
        # Create Satellite Model of Primary Satellite
        p_x0 = x0.copy()
        fermiSat = FermiSatelliteModel(t0, p_x0, filt_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)

        # Simulate Primary, Setup its Kalman Filter, and Run
        p_xs, p_zs, p_ws, p_vs = fermiSat.simulate(filt_orbits, std_gps_noise, W=None, with_density_jumps=with_density_jumps)
        cache_dic['kf_prim_sim'] = (p_xs.copy(), p_zs.copy(), p_ws.copy(), p_vs.copy())
        P_kf = np.eye(n) * (0.001)**2
        P_kf[6,6] = .01
        x_kf = np.random.multivariate_normal(p_xs[0], P_kf)
        x_kf[6] = 0
        cache_dic['x0_P0_prim'] = (x_kf.copy(), P_kf.copy())
        fermiSat.reset_state(x_kf, 0)
        p_xs_kf = [x_kf.copy()]
        p_Ps_kf = [P_kf.copy()]
        N = p_zs.shape[0]
        for i in range(1, N):
            # Time Prop
            Phi_k = fermiSat.get_transition_matrix(STM_order)
            P_kf = Phi_k @ P_kf @ Phi_k.T + Wn
            x_kf = fermiSat.step() 
            # Measurement Update
            K = np.linalg.solve(H @ P_kf @ H.T + V, H @ P_kf).T #P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
            zbar = H @ x_kf
            zk = p_zs[i]
            r = zk - zbar 
            print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(p_xs[i] - x_kf))
            x_kf = x_kf + K @ r 
            # Make sure changes in Cd/Cr are within bounds
            x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)
            fermiSat.reset_state(x_kf, i) #/1000)
            P_kf = (I7 - K @ H) @ P_kf @ (I7 - K @ H).T + K @ V @ K.T 
            # Log
            p_xs_kf.append(x_kf.copy())
            p_Ps_kf.append(P_kf.copy())
        p_xs_kf = np.array(p_xs_kf)
        p_Ps_kf = np.array(p_Ps_kf)
        fermiSat.clear_model()
        cache_dic['kf_prim'] = (p_xs_kf.copy(), p_Ps_kf.copy())
    else:
        p_xs, p_zs, p_ws, p_vs = cache_dic['kf_prim_sim']
        p_x0_kf, p_P0_kf = cache_dic['x0_P0_prim']
        p_xs_kf, p_Ps_kf = cache_dic['kf_prim']
    if with_filter_plots:
        # Plot Primary 
        print("Primary Sateliite KF Run:")
        ce.plot_simulation_history(None, (p_xs, p_zs, p_ws, p_vs), (p_xs_kf, p_Ps_kf), scale=1)

    # Now Repeat for secondary Satellite
    if cached_dir is "":
        s_x0 = x0.copy()
        s_x0[3:] *= -1
        fermiSat = FermiSatelliteModel(t0, s_x0, filt_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        # Simulate Secondary, Setup its Kalman Filter, and Run
        s_xs, s_zs, s_ws, s_vs = fermiSat.simulate(filt_orbits, std_gps_noise, W=None, with_density_jumps=with_density_jumps)
        cache_dic['kf_sec_sim'] = (s_xs.copy(), s_zs.copy(), s_ws.copy(), s_vs.copy())
        P_kf = np.eye(n) * (0.001)**2
        P_kf[6,6] = .01
        x_kf = np.random.multivariate_normal(s_xs[0], P_kf)
        x_kf[6] = 0
        cache_dic['x0_P0_sec'] = (x_kf.copy(), P_kf.copy())
        fermiSat.reset_state(x_kf, 0)
        s_xs_kf = [x_kf.copy()]
        s_Ps_kf = [P_kf.copy()]
        N = s_zs.shape[0]
        for i in range(1, N):
            # Time Prop
            Phi_k = fermiSat.get_transition_matrix(STM_order)
            P_kf = Phi_k @ P_kf @ Phi_k.T + Wn
            x_kf = fermiSat.step() 
            # Measurement Update
            K = np.linalg.solve(H @ P_kf @ H.T + V, H @ P_kf).T #P_kf @ H.T @ np.linalg.inv(H @ P_kf @ H.T + V)
            zbar = H @ x_kf
            zk = s_zs[i]
            r = zk - zbar 
            print("Norm residual: ", np.linalg.norm(r), " Norm State Diff:", np.linalg.norm(s_xs[i] - x_kf))
            x_kf = x_kf + K @ r 
            # Make sure changes in Cd/Cr are within bounds
            x_kf[6:] = np.clip(x_kf[6:], -0.98, np.inf)
            fermiSat.reset_state(x_kf, i) #/1000)
            P_kf = (I7 - K @ H) @ P_kf @ (I7 - K @ H).T + K @ V @ K.T 
            # Log
            s_xs_kf.append(x_kf.copy())
            s_Ps_kf.append(P_kf.copy())
        s_xs_kf = np.array(s_xs_kf)
        s_Ps_kf = np.array(s_Ps_kf)
        cache_dic['kf_sec'] = (s_xs_kf.copy(), s_Ps_kf.copy())
        fermiSat.clear_model()
    else:
        s_xs, s_zs, s_ws, s_vs = cache_dic['kf_sec_sim']
        s_x0_kf, s_P0_kf = cache_dic['x0_P0_sec']
        s_xs_kf, s_Ps_kf = cache_dic['kf_sec']
    if with_filter_plots:
        # Plot Secondary 
        print("Secondary Satelite KF Run:")
        ce.plot_simulation_history(None, (s_xs, s_zs, s_ws, s_vs), (s_xs_kf, s_Ps_kf), scale=1)
    
    # Time at the start of prediction
    filt_time = (p_xs_kf.shape[0]-1) * filt_dt # Number of filtering steps * filt_dt
    t0_pred = datetime.strptime(t0, "%d %b %Y %H:%M:%S.%f") + timedelta(seconds = filt_time)

    # Prediction for primary and secondary satellites 7-days into future
    if cached_dir is "":
        # Propagate Primary Satellite 7 days into future + its covariance
        p_xpred = p_xs_kf[-1]
        p_Ppred = p_Ps_kf[-1]
        fermiSat = FermiSatelliteModel(t0_pred, p_xpred[0:6], pred_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(p_xpred, 0)
        p_pks = [] # Primary Pos 3-Vec
        p_vks = [] # Primary Vel 3-Vec
        p_cdks = [] # Primary change in atms. drag
        p_aks = [] # Primary Acc 3-Vec
        p_Pks = [] # Primary Covariance 
        # Propagate Primary and Store
        xk = p_xpred.copy()
        Pk = p_Ppred.copy()
        for i in range(pred_steps):
            dxk_dt = fermiSat.get_state6_derivatives() 
            p_pks.append(xk[0:3])
            p_vks.append(xk[3:6])
            p_cdks.append(xk[6])
            p_aks.append(dxk_dt[3:6])
            p_Pks.append(Pk)
            Phik = fermiSat.get_transition_matrix(taylor_order=3)
            Pk = Phik @ Pk @ Phik.T
            xk = fermiSat.step()
        p_pks = np.array(p_pks)
        p_vks = np.array(p_vks)
        p_aks = np.array(p_aks)
        p_Pks = np.array(p_Pks)
        fermiSat.clear_model()
        cache_dic['prim_pred_hist'] = (p_pks, p_vks, p_cdks, p_aks, p_Pks)

        # Propagate Secondary Satellite 7 days into future + its covariance
        s_xpred = s_xs_kf[-1]
        s_Ppred = s_Ps_kf[-1]
        fermiSat = FermiSatelliteModel(t0_pred, s_xpred[0:6], pred_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(s_xpred, 0)
        s_pks = [] # Secondary Pos 3-Vec
        s_vks = [] # Secondary Vel 3-Vec
        s_cdks = [] # Secondary change in atms. drag
        s_aks = [] # Secondary Acc 3-Vec
        s_Pks = [] # Secondary Covariance 
        xk = s_xpred.copy()
        Pk = s_Ppred.copy()
        for i in range(pred_steps):
            dxk_dt = fermiSat.get_state6_derivatives() 
            s_pks.append(xk[0:3])
            s_vks.append(xk[3:6])
            s_cdks.append(xk[6])
            s_aks.append(dxk_dt[3:6])
            s_Pks.append(Pk)
            Phik = fermiSat.get_transition_matrix(taylor_order=3)
            Pk = Phik @ Pk @ Phik.T
            xk = fermiSat.step()
        s_pks = np.array(s_pks)
        s_vks = np.array(s_vks)
        s_cdks = np.array(s_cdks)
        s_aks = np.array(s_aks)
        s_Pks = np.array(s_Pks)
        fermiSat.clear_model()
        cache_dic['sec_pred_hist'] = (s_pks, s_vks, s_cdks, s_aks, s_Pks)
    else:
        p_pks, p_vks, p_cdks, p_aks, p_Pks = cache_dic['prim_pred_hist']
        s_pks, s_vks, s_cdks, s_aks, s_Pks = cache_dic['sec_pred_hist']


    # Here, store the data before moving on to the second part 
    if cached_dir is "":
        input_ok = False
        while not input_ok:
            input_ok = True
            ui = input("Would you like to store your data? (Enter y or n):").lower()
            if( ui == 'y' ):
                timestamp = str( time.time() ) + ".pickle"
                fpath = data_path + timestamp
                with open(fpath, "wb") as handle:
                    pickle.dump(cache_dic, handle)
                print("Stored data to:", fpath)
            elif( ui == 'n' ):
                print("Data not stored!")
            else:
                print("Unrecognized input")
                input_ok = False
    
    # Plot relative differences and choose a window of time where both satellite are very close to each other, 7-days out in future
    if with_pred_plots:
        # Plot Trajectories of both satellites 
        fig = plt.figure() 
        ax = fig.gca(projection='3d')
        plt.title("Predicted trajectories (primary=red, secondary=blue) over 7-day lookahead:")
        ax.plot(p_pks[:,0], p_pks[:,1], p_pks[:,2], color = 'r')
        ax.plot(s_pks[:,0], s_pks[:,1], s_pks[:,2], color = 'b')
        ax.scatter(p_pks[0,0], p_pks[0,1], p_pks[0,2], color = 'k', s=80)
        ax.set_xlabel("x-axis (km)")
        ax.set_ylabel("y-axis (km)")
        ax.set_zlabel("z-axis (km)")
        fig2 = plt.figure()
        plt.suptitle("Norm of predicted satellite seperation over 7-day lookahead")
        r_norms = np.linalg.norm(s_pks - p_pks, axis=1)
        plt.plot( np.arange(r_norms.size), r_norms) #(np.arange(r_norms.size) * pred_dt) / (24*60*60),
        plt.ylabel("Seperation (km)")
        plt.xlabel("# days lookahead")
        plt.show()

    # If user wishes to see prediction plot, ask if they would like to create MC over other window
    run_itca = False 
    if cache_dic['itca_data'] is None:
        cache_dic['itca_window_idxs'] = [4650, 4670] # Could manually reset this here
        run_itca = True
    if (cache_dic['itca_data'] is not None) and with_pred_plots:
        print("Old ITCA Left and Right Hand Side Window Indices: ", cache_dic['itca_window_idxs'][0], cache_dic['itca_window_idxs'][1] )
        while True:
            is_run = input("Would you like to rerun Iterative Time of Closest Approach (ITCA)? (Enter y or n): ")
            if is_run == 'y':
                run_itca = True
                print("Re-running ITCA!")
                valid_range = (0, r_norms.size-1)
                is_ok = False 
                while True:
                    cache_dic['itca_window_idxs'][0] = int( input("   Enter index for itca window start: i.e., a value between [{},{}]".format(valid_range[0], valid_range[1]) ) )
                    if( (cache_dic['itca_window_idxs'][0] >= valid_range[0]) and (cache_dic['itca_window_idxs'][0] <= valid_range[1]) ):
                        break
                    else:
                        print("Invalid Entery of {}. Try Again!".format(cache_dic['itca_window_idxs'][0]) )
                valid_range = (cache_dic['itca_window_idxs'][0]+1, r_norms.size-1)
                while True:
                    cache_dic['itca_window_idxs'][1] = int( input("   Enter index for itca window end: i.e., a value between [{},{}]".format(valid_range[0], valid_range[1]) ) )
                    if( (cache_dic['itca_window_idxs'][1] >= valid_range[0]) and (cache_dic['itca_window_idxs'][1] <= valid_range[1]) ):
                        break
                    else:
                        print("Invalid Entery of {}. Try Again!".format(cache_dic['itca_window_idxs'][1]) )
                print("Re-running ITCA with LHS/RHS indices of ", cache_dic['itca_window_idxs'])
                break
            elif is_run == 'n':
                run_itca = False
                print("Not rerunning ITCA!")
                break
            else: 
                print("Invalid entery. Try again!")
        
    # Now run the iterative time of closest approach algorithm if desired
    if run_itca:
        # Run iterative time of closest approach over this window, find exact point of closest approach
        start_idx = cache_dic['itca_window_idxs'][0]
        end_idx = cache_dic['itca_window_idxs'][1]
        # GMAT iterative closest time of approach
        i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c = pc.iterative_time_closest_approach(
            pred_dt, t0_pred, 
            (p_pks[:end_idx],p_vks[:end_idx],p_aks[:end_idx]), 
            (s_pks[:end_idx],s_vks[:end_idx],s_aks[:end_idx]), 
            start_idx = start_idx,
            with_plot=with_pred_plots
            )
        cache_dic['itca_data'] = (i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c)

        # Step both the primary covariance and secondary covariance to the time of closest approach 
        xlhs_prim = np.concatenate(( p_pks[i_star_lhs], p_vks[i_star_lhs], np.array([p_cdks[i_star_lhs]]) ))
        fermiSat = FermiSatelliteModel( t0_pred + timedelta(seconds = t_lhs), xlhs_prim[0:6], t_c - t_lhs )
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(xlhs_prim, 0)
        p_Phi = fermiSat.get_transition_matrix(taylor_order=3)
        p_Ptc = p_Phi @ p_Pks[i_star_lhs] @ p_Phi.T
        p_xtc = fermiSat.step()
        print("XPrim at TCA: ", p_xtc[0:3])
        print("Pos Diff XPrim at TCA (meters): ", 1000*(pp_c - p_xtc[0:3]) )
        print("Vel Diff XPrim at TCA (meters/sec): ", 1000*(pv_c - p_xtc[3:6]) )
        fermiSat.clear_model()
        p_xtc[0:3] = pp_c.copy()
        p_xtc[3:6] = pv_c.copy()
        cache_dic['nom_prim_tca'] = (p_xtc, p_Ptc)

        # Propagate the Secondary Covariance to Point of Closest Approach 
        xlhs_sec = np.concatenate((s_pks[i_star_lhs],s_vks[i_star_lhs], np.array([s_cdks[i_star_lhs]]) ))
        fermiSat = FermiSatelliteModel(t0_pred + timedelta(seconds = t_lhs), xlhs_sec[0:6], t_c - t_lhs)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(xlhs_sec, 0)
        s_Phi = fermiSat.get_transition_matrix(taylor_order=3)
        s_Ptc = s_Phi @ s_Pks[i_star_lhs] @ s_Phi.T
        s_xtc = fermiSat.step()
        print("XSec at TCA: ", s_xtc[0:3])
        print("Pos Diff XSec at TCA (meters): ", 1000*(sp_c - s_xtc[0:3]) )
        print("Vel Diff XSec at TCA (meters/sec): ", 1000*(sv_c - s_xtc[3:6]) )
        fermiSat.clear_model()
        s_xtc[0:3] = sp_c.copy()
        s_xtc[3:6] = sv_c.copy()
        cache_dic['nom_sec_tca'] = (s_xtc, s_Ptc)

        input_ok = False
        while not input_ok:
            input_ok = True
            ui = input("Would you like to store your data? (y/n)").lower()
            if( ui == 'y' ):
                if cached_dir is "":
                    fpath = data_path + timestamp
                else:
                    fpath = data_path + cached_dir
                with open(fpath, "wb") as handle:
                    pickle.dump(cache_dic, handle)
                print("Stored data to:", fpath)
            elif( ui == 'n' ):
                print("Data not stored!")
            else:
                print("Unrecognized input")
                input_ok = False
    else:
        i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c = cache_dic['itca_data']
        p_xtc, p_Ptc = cache_dic['nom_prim_tca']
        s_xtc, s_Ptc = cache_dic['nom_sec_tca']

    # Repeat the following two steps for a select number of monte carlos ... caching the mc trial data as you go... this is expensive
    mc_trials = 0 #int( input("How many MC trials would you like to add: (i.e, 0 to 10000): ") )
    sub_trials_per_log = 5
    mc_count = 0
    prim_count = 0 
    sec_count = 0
    new_mc_runs_prim = []
    new_mc_runs_sec = []
    while mc_count < mc_trials: 
        inner_loop = mc_trials if mc_trials < sub_trials_per_log else sub_trials_per_log
        if (inner_loop + mc_count) > mc_trials:
            inner_loop = mc_trials - mc_count
        # Simulate a new atms. density realization under the current atms. distribution starting at primary at end of filtration. 
        print("Running MC for Primary:")
        inner_mc_runs_prim = []
        for mc_it in range(inner_loop):
            print( "Primary trial {}/{}:".format(prim_count+1, mc_trials) )
            if with_mc_from_x0:
                p_xpred = p_xs_kf[-1].copy()
            else:
                p_xpred = np.zeros(n)
                p_xpred[0:6] = np.random.multivariate_normal(p_xs_kf[-1][0:6], p_Ps_kf[-1][0:6,0:6])
            fermiSat = FermiSatelliteModel(t0_pred, p_xpred[0:6], pred_dt)
            fermiSat.create_model(with_jacchia=True, with_SRP=True)
            fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
            xs, _, ws, _ = fermiSat.simulate(None, std_gps_noise, W=None, with_density_jumps = False, num_steps = i_star_lhs)
            fermiSat.clear_model()
            fermiSat = FermiSatelliteModel(t0_pred, xs[-1][0:6].copy(), t_c - t_lhs)
            fermiSat.create_model(with_jacchia=True, with_SRP=True)
            fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
            fermiSat.reset_state(xs[-1].copy(), 0)
            xmc = fermiSat.step()
            inner_mc_runs_prim.append(xmc.copy())
            fermiSat.clear_model()
            prim_count += 1
    
        print("Running MC for Secondary:")
        inner_mc_runs_sec = []
        for mc_it in range(inner_loop):
            print( "Secondary trial {}/{}:".format(sec_count+1, mc_trials) )
            if with_mc_from_x0:
                s_xpred = s_xs_kf[-1].copy()
            else:
                s_xpred = np.zeros(n)
                s_xpred[0:6] = np.random.multivariate_normal(s_xs_kf[-1][0:6], s_Ps_kf[-1][0:6,0:6])
            fermiSat = FermiSatelliteModel(t0_pred, s_xpred[0:6], pred_dt)
            fermiSat.create_model(with_jacchia=True, with_SRP=True)
            fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
            xs, _, ws, _ = fermiSat.simulate(None, std_gps_noise, W=None, with_density_jumps = False, num_steps = i_star_lhs)
            fermiSat.clear_model()
            fermiSat = FermiSatelliteModel(t0_pred, xs[-1][0:6].copy(), t_c - t_lhs)
            fermiSat.create_model(with_jacchia=True, with_SRP=True)
            fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
            fermiSat.reset_state(xs[-1].copy(), 0)
            xmc = fermiSat.step()
            inner_mc_runs_sec.append(xmc.copy())
            fermiSat.clear_model()
            sec_count += 1
        new_mc_runs_prim += inner_mc_runs_prim 
        new_mc_runs_sec += inner_mc_runs_sec
        mc_count += inner_loop

        print("Caching {} runs, total done is {}, total left is {}".format(inner_loop, mc_count, mc_trials - mc_count) )
        if with_mc_from_x0:
            if cache_dic['mc_prim_tcas'] is None:
                cache_dic['mc_prim_tcas'] = inner_mc_runs_prim
            else:
                cache_dic['mc_prim_tcas'] += inner_mc_runs_prim
            
            if cache_dic['mc_sec_tcas'] is None:
                cache_dic['mc_sec_tcas'] = inner_mc_runs_sec
            else:
                cache_dic['mc_sec_tcas'] += inner_mc_runs_sec
        else:
            if 'mc_prim_sample_tcas' not in cache_dic:
                cache_dic['mc_prim_sample_tcas'] = None
            if 'mc_sec_sample_tcas' not in cache_dic:
                cache_dic['mc_sec_sample_tcas'] = None
            if cache_dic['mc_prim_sample_tcas'] is None:
                cache_dic['mc_prim_sample_tcas'] = inner_mc_runs_prim
            else:
                cache_dic['mc_prim_sample_tcas'] += inner_mc_runs_prim
            
            if cache_dic['mc_sec_sample_tcas'] is None:
                cache_dic['mc_sec_sample_tcas'] = inner_mc_runs_sec
            else:
                cache_dic['mc_sec_sample_tcas'] += inner_mc_runs_sec
        if cached_dir is "":
            fpath = data_path + timestamp
        else:
            fpath = data_path + cached_dir
        with open(fpath, "wb") as handle:
            pickle.dump(cache_dic, handle)
        print("Stored data to:", fpath)
        
    print("Logged ", len(new_mc_runs_prim), " total new Primary MC runs")
    print("Logged ", len(new_mc_runs_sec), " total new Secondary MC runs")

    plot_only_new_mcs = False 
    if plot_only_new_mcs:
        prim_mc = new_mc_runs_sec
        sec_mc = new_mc_runs_sec
    else:
        prim_mc = cache_dic['mc_prim_tcas']
        sec_mc = cache_dic['mc_sec_tcas']
    
    # Run the MCE for ~50 steps before the end of filtering for both the primary and the secondary, 
    # Take the obtained 7x7 covariance, and project it to the TCA point
    if ('mce_prim' not in cache_dic):
        run_mce = True
    else:
        if cache_dic['mce_prim'] is None:
            run_mce = True
        else:
            while True:
                is_run = input("Would you like to rerun the MCE? (Enter y or n): ")
                if is_run == 'y':
                    run_mce = True
                    print("Re-running MCE!")
                    break
                elif is_run == 'n':
                    run_mce = False
                    print("Not rerunning MCE!")
                    break
                else: 
                    print("Invalid entery. Try again!")
    if run_mce:
        mce_steps = 15
        mce_num_windows = 3
        sim_steps = p_zs.shape[0]
        mce_msmt_idx = sim_steps - mce_steps
        mce_t0 = gmat_time_string_2_datetime(t0) + timedelta( seconds = (mce_msmt_idx-1) * filt_dt )
        # Begin Primary
        mce_p_zs = p_zs[mce_msmt_idx:]
        mce_p_xs = p_xs[mce_msmt_idx:]
        
        # MAY WANT TO ADD PHI^T as way of starting off the MCE
        '''
        mce_p_x0 = p_xs_kf[mce_msmt_idx].copy()
        mce_p_x0[0:6] *= 1000
        mce_p_P0 = p_Ps_kf[mce_msmt_idx].copy()
        mce_p_P0[0:6,0:6] *= 1000**2
        mce_p_P0[0:6,6] *= 1000
        mce_p_P0[6,0:6] *= 1000
        mce_p_dx0 = np.random.randn(7)*0.001
        mce_p_x0bar = mce_p_x0-mce_p_dx0
        mce_p_dz0 = 1000*mce_p_zs[0][2] - H[2] @ mce_p_x0bar
        mce_gamma = np.ones(3) * std_gps_noise * 1000 / 1.3898
        mce_beta = np.array([std_Cd])
        mce_A0, mce_p0, mce_b0 = ce.speyers_window_init(mce_p_dx0, mce_p_P0, H[2], mce_gamma[2], mce_p_dz0)
        '''
        mce_x0 = p_xs_kf[mce_msmt_idx-1].copy()
        mce_x0[6] = 0.1
        fermSat = FermiSatelliteModel(mce_t0 - timedelta(seconds = filt_dt), mce_x0.copy(), filt_dt)
        fermSat.create_model()
        fermSat.set_solve_for(field="Cd", dist=mode, scale=std_Cd, tau=tau_Cd, alpha=2.0 if mode == "gauss" else 1.3)
        fermSat.reset_state(mce_x0, 0)
        mce_p_Jac = fermSat.get_jacobian_matrix()
        mce_p_Jac[0:6,6] *= 1000
        mce_Phi0 = (np.eye(7) + mce_p_Jac * filt_dt + mce_p_Jac @ mce_p_Jac * filt_dt**2 / 2 + mce_p_Jac @ mce_p_Jac @ mce_p_Jac * filt_dt**3 / 6)
        mce_A0 = mce_Phi0.T
        mce_p0 = np.array([1,1,1,.001,.001,.001,0.001])
        mce_b0 = np.zeros(7)
        mce_p_x0bar = fermSat.step().copy()
        mce_p_x0bar[0:6] *= 1000
        mce_gamma = np.ones(3) * std_gps_noise * 1000 / 1.3898
        mce_beta = np.array([std_Cd])    
        mce_other_params = p_Ps_kf[mce_msmt_idx:]
        fermSat.clear_model()

        # Begin Primary MCE Estimation
        mce_other_params = p_Ps_kf[mce_msmt_idx:]
        p_cauchyEst = gmce.GmatMCE(mce_num_windows, mce_t0, mce_p_x0bar, filt_dt, mce_A0, mce_p0, mce_b0, mce_beta, mce_gamma, Cd_dist = mode, win_reinitialize_func=gmce.reinitialize_func_init_cond, win_reinitialize_params=mce_other_params, debug_print = True, mce_print = True)
        for zk, xk in zip(mce_p_zs, mce_p_xs):
            _zk = zk.copy() * 1000 # km -> m
            _xk = xk.copy()
            _xk[0:6] *= 1000 # km -> m
            p_cauchyEst.sim_step( zk = _zk, x_truth = _xk, is_inputs_meters = True)

        # Convert the data from meters to Km 
        p_ce_xhats = np.array(p_cauchyEst.xhats)
        p_ce_xhats[:,0:6] /= 1000
        p_ce_avg_xhats = np.array(p_cauchyEst.avg_xhats)
        p_ce_avg_xhats[:,0:6] /= 1000
        p_ce_Phats = np.array(p_cauchyEst.Phats)
        p_ce_avg_Phats = np.array(p_cauchyEst.avg_Phats)
        for pHat, apHat in zip(p_ce_Phats, p_ce_avg_Phats):
            pHat[0:6,0:6] /= 1000**2
            pHat[0:6,6] /= 1000
            pHat[6,0:6] /= 1000
            apHat[0:6,0:6] /= 1000**2
            apHat[6,0:6] /= 1000
            apHat[0:6,6] /= 1000

        cache_dic['mce_prim'] = (mce_steps, p_ce_xhats, p_ce_Phats, p_ce_avg_xhats, p_ce_avg_Phats)
        # Now plot out the results -- 
        if with_filter_plots:
            # could plot the best windows
            #foo = np.zeros(p_ce_xhats.shape[0])
            #p_cauchy_info = {"x" : p_ce_xhats, "P" : p_ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
            #ce.plot_simulation_history(p_cauchy_info, 
            #    (p_xs[mce_msmt_idx:], p_zs[mce_msmt_idx:], p_ws[mce_msmt_idx:], p_vs[mce_msmt_idx:]), 
            #    (p_xs_kf[mce_msmt_idx:], p_Ps_kf[mce_msmt_idx:]), 
            #    scale=1)
            
            # or could plot the avg windows
            foo = np.zeros(p_ce_xhats.shape[0])
            p_cauchy_avg_info = {"x" : p_ce_avg_xhats, "P" : p_ce_avg_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
            ce.plot_simulation_history(p_cauchy_avg_info, 
                (p_xs[mce_msmt_idx:], p_zs[mce_msmt_idx:], p_ws[mce_msmt_idx:], p_vs[mce_msmt_idx:]), 
                (p_xs_kf[mce_msmt_idx:], p_Ps_kf[mce_msmt_idx:]), 
                scale=1)
        p_cauchyEst.clear_gmat()

        # Here, for this experiment at least, we do not care yet about the "uncertainty" at the end of filtration.
        # That is, we assume, like above for the KF, the filtration error is zero 
        # Take the MCE Covariance and propagate it to the expected TCA like we did before with the KF
        p_xpred = p_xs_kf[-1] # p_ce_xhats # p_ce_avg_xhats
        p_Ppred = p_ce_Phats[-1] # p_ce_avg_Phats
        fermiSat = FermiSatelliteModel(t0_pred, p_xpred[0:6], pred_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(p_xpred, 0) # do not wish to use Cauchy's estimate of the density
        Pk = p_Ppred.copy()
        for i in range(i_star_lhs):
            Phik = fermiSat.get_transition_matrix(taylor_order=STM_order)
            Pk = Phik @ Pk @ Phik.T
            fermiSat.step()
        fermiSat.clear_model()
        # Now, adjust the time step and propagate the primary covariance exactly to time of closest approach 
        xlhs_prim = np.concatenate(( p_pks[i_star_lhs], p_vks[i_star_lhs], np.array([p_cdks[i_star_lhs]]) ))
        fermiSat = FermiSatelliteModel( t0_pred + timedelta(seconds = t_lhs), xlhs_prim[0:6], t_c - t_lhs )
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(xlhs_prim, 0)
        p_Phi = fermiSat.get_transition_matrix(taylor_order=3)
        p_mce_Ptc = p_Phi @ Pk @ p_Phi.T
        cache_dic['mce_prim_cov_tca'] = p_mce_Ptc.copy()
        fermiSat.clear_model()


        # Begin Secondary MCE Estimation
        mce_s_zs = s_zs[mce_msmt_idx:]
        mce_s_xs = s_xs[mce_msmt_idx:]
        '''
        # MAY WANT TO ADD PHI^T as way of starting off the MCE
        mce_s_x0 = s_xs_kf[mce_msmt_idx].copy()
        mce_s_x0[0:6] *= 1000
        mce_s_P0 = s_Ps_kf[mce_msmt_idx].copy()
        mce_s_P0[0:6,0:6] *= 1000**2
        mce_s_P0[0:6,6] *= 1000
        mce_s_P0[6,0:6] *= 1000
        mce_s_dx0 = np.random.randn(7)*0.001
        mce_s_x0bar = mce_s_x0-mce_s_dx0
        mce_s_dz0 = 1000*mce_s_zs[0][2] - H[2] @ mce_s_x0bar
        mce_A0, mce_p0, mce_b0 = ce.speyers_window_init(mce_s_dx0, mce_s_P0, H[2], mce_gamma[2], mce_s_dz0)
        '''
        mce_x0 = s_xs_kf[mce_msmt_idx-1].copy()
        mce_x0[6] = 0.1
        fermSat = FermiSatelliteModel(mce_t0 - timedelta(seconds = filt_dt), mce_x0.copy(), filt_dt)
        fermSat.create_model()
        fermSat.set_solve_for(field="Cd", dist=mode, scale=std_Cd, tau=tau_Cd, alpha=2.0 if mode == "gauss" else 1.3)
        fermSat.reset_state(mce_x0, 0)
        mce_s_Jac = fermSat.get_jacobian_matrix()
        mce_s_Jac[0:6,6] *= 1000        
        mce_Phi0 = (np.eye(7) + mce_s_Jac * filt_dt + mce_s_Jac @ mce_s_Jac * filt_dt**2 / 2 + mce_s_Jac @ mce_s_Jac @ mce_s_Jac * filt_dt**3 / 6)
        mce_A0 = mce_Phi0.T
        mce_p0 = np.array([1,1,1,.001,.001,.001,0.001])
        mce_b0 = np.zeros(7)
        mce_s_x0bar = fermSat.step().copy()
        mce_s_x0bar[0:6] *= 1000
        mce_gamma = np.ones(3) * std_gps_noise * 1000 / 1.3898
        mce_beta = np.array([std_Cd])    
        mce_other_params = s_Ps_kf[mce_msmt_idx:]
        fermSat.clear_model()

        # Begin Secondary MCE Estimation
        mce_other_params = s_Ps_kf[mce_msmt_idx:]
        s_cauchyEst = gmce.GmatMCE(mce_num_windows, mce_t0, mce_s_x0bar, filt_dt, mce_A0, mce_p0, mce_b0, mce_beta, mce_gamma, Cd_dist = mode, win_reinitialize_func=gmce.reinitialize_func_init_cond, win_reinitialize_params=mce_other_params, debug_print = True, mce_print = True)
        for zk, xk in zip(mce_s_zs, mce_s_xs):
            _zk = zk.copy() * 1000 # km -> m
            _xk = xk.copy()
            _xk[0:6] *= 1000 # km -> m
            s_cauchyEst.sim_step( zk = _zk, x_truth = _xk, is_inputs_meters = True)
        
        # Convert the data from meters to Km 
        s_ce_xhats = np.array(s_cauchyEst.xhats)
        s_ce_xhats[:,0:6] /= 1000
        s_ce_avg_xhats = np.array(s_cauchyEst.avg_xhats)
        s_ce_avg_xhats[:,0:6] /= 1000
        s_ce_Phats = np.array(s_cauchyEst.Phats)
        s_ce_avg_Phats = np.array(s_cauchyEst.avg_Phats)
        for pHat, apHat in zip(s_ce_Phats, s_ce_avg_Phats):
            pHat[0:6,0:6] /= 1000**2
            pHat[0:6,6] /= 1000
            pHat[6,0:6] /= 1000
            apHat[0:6,0:6] /= 1000**2
            apHat[6,0:6] /= 1000
            apHat[0:6,6] /= 1000
    
        cache_dic['mce_sec'] = (mce_steps, s_ce_xhats, s_ce_Phats, s_ce_avg_xhats, s_ce_avg_Phats)
        # Now plot out the results -- 
        if with_filter_plots:
            # could plot the best windows
            #foo = np.zeros(s_ce_xhats.shape[0])
            #s_cauchy_info = {"x" : s_ce_xhats, "P" : s_ce_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
            #ce.plot_simulation_history(s_cauchy_info, 
            #    (s_xs[mce_msmt_idx:], s_zs[mce_msmt_idx:], s_ws[mce_msmt_idx:], s_vs[mce_msmt_idx:]), 
            #    (s_xs_kf[mce_msmt_idx:], s_Ps_kf[mce_msmt_idx:]), 
            #    scale=1)
            # or could plot the avg windows
            foo = np.zeros(s_ce_xhats.shape[0])
            s_cauchy_avg_info = {"x" : s_ce_avg_xhats, "P" : s_ce_avg_Phats, "err_code" : foo, "fz" : foo, "cerr_fz" : foo, "cerr_x" : foo, "cerr_P": foo }
            ce.plot_simulation_history(s_cauchy_avg_info, 
                (s_xs[mce_msmt_idx:], s_zs[mce_msmt_idx:], s_ws[mce_msmt_idx:], s_vs[mce_msmt_idx:]), 
                (s_xs_kf[mce_msmt_idx:], s_Ps_kf[mce_msmt_idx:]), 
                scale=1)
        s_cauchyEst.clear_gmat()

        # Here, for this experiment at least, we do not care yet about the "uncertainty" at the end of filtration.
        # That is, we assume, like above for the KF, the filtration error is zero 
        # Take the MCE Covariance and propagate it to the expected TCA like we did before with the KF
        s_xpred = s_xs_kf[-1] # p_ce_xhats # p_ce_avg_xhats
        s_Ppred = s_ce_Phats[-1] # p_ce_avg_Phats
        fermiSat = FermiSatelliteModel(t0_pred, s_xpred[0:6], pred_dt)
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(s_xpred, 0) # do not wish to use Cauchy's estimate of the density
        Pk = s_Ppred.copy()
        for i in range(i_star_lhs):
            Phik = fermiSat.get_transition_matrix(taylor_order=STM_order)
            Pk = Phik @ Pk @ Phik.T
            fermiSat.step()
        fermiSat.clear_model()
        # Now, adjust the time step and propagate the primary covariance exactly to time of closest approach 
        xlhs_sec = np.concatenate(( s_pks[i_star_lhs], s_vks[i_star_lhs], np.array([s_cdks[i_star_lhs]]) ))
        fermiSat = FermiSatelliteModel( t0_pred + timedelta(seconds = t_lhs), xlhs_sec[0:6], t_c - t_lhs )
        fermiSat.create_model(with_jacchia=True, with_SRP=True)
        fermiSat.set_solve_for("Cd", sas_Cd, std_Cd, tau_Cd, alpha=sas_alpha)
        fermiSat.reset_state(xlhs_sec, 0)
        s_Phi = fermiSat.get_transition_matrix(taylor_order=3)
        s_mce_Ptc = s_Phi @ Pk @ s_Phi.T
        cache_dic['mce_sec_cov_tca'] = s_mce_Ptc.copy()
        fermiSat.clear_model()

        # Save Data 
        input_ok = False
        while not input_ok:
            input_ok = True
            ui = input("Would you like to store your data? (y/n)").lower()
            if( ui == 'y' ):
                if cached_dir is "":
                    fpath = data_path + timestamp
                else:
                    fpath = data_path + cached_dir
                with open(fpath, "wb") as handle:
                    pickle.dump(cache_dic, handle)
                print("Stored data to:", fpath)
            elif( ui == 'n' ):
                print("Data not stored!")
            else:
                print("Unrecognized input")
                input_ok = False
    else:
        p_mce_Ptc = cache_dic['mce_prim_cov_tca']
        s_mce_Ptc = cache_dic['mce_sec_cov_tca']
    
    kf_quant = 0.9999
    mce_quant = 0.70
    # Plot this out in 3D
    #pc.draw_3d_encounter_plane(s_xtc, p_xtc, s_Ptc[0:3,0:3], p_Ptc[0:3,0:3], 
    #   mc_runs_prim = prim_mc, mc_runs_sec = sec_mc, 
    #   s_mce_Ptc = s_mce_Ptc, p_mce_Ptc = p_mce_Ptc)

    # Plot this out in 2D
    pc.draw_2d_projected_encounter_plane(kf_quant, s_xtc, p_xtc, s_Ptc, p_Ptc, 
        cache_dic['mc_prim_tcas'], cache_dic['mc_sec_tcas'], 
        quantile_mce = mce_quant, s_mce_Ptc = s_mce_Ptc, p_mce_Ptc = p_mce_Ptc,
        mc_sec_sample_tcas = cache_dic['mc_sec_sample_tcas'], mc_prim_sample_tcas = cache_dic['mc_prim_sample_tcas'])
    
    kf_quant = 0.9999
    mce_quant = 0.9999

    # End of Filtration
    p_P_mce_filtend = cache_dic['mce_prim'][2][-1] # [4][-1] # [4][-1] is average, [2][-1] is 'best'
    s_P_mce_filtend = cache_dic['mce_sec'][2][-1] # [4][-1] # [4][-1] is average, [2][-1] is 'best'
    pc.analyze_3d_statistics(kf_quant, mce_quant,
        s_xs_kf[-1], p_xs_kf[-1], s_Ps_kf[-1], p_Ps_kf[-1],
        s_P_mce_filtend, p_P_mce_filtend,
        cache_dic['mc_prim_tcas'], cache_dic['mc_sec_tcas'],
        cache_dic['mc_sec_sample_tcas'], cache_dic['mc_prim_sample_tcas'])

    # At Time of Closest Approach
    pc.analyze_3d_statistics(kf_quant, mce_quant,
        s_xtc, p_xtc, s_Ptc, p_Ptc,
        s_mce_Ptc, p_mce_Ptc, 
        cache_dic['mc_prim_tcas'], cache_dic['mc_sec_tcas'],
        cache_dic['mc_sec_sample_tcas'], cache_dic['mc_prim_sample_tcas'])
    foobar = 2


if __name__ == "__main__":
    #test_sat_pc()
    #find_cross_radial_error_mc_vs_density()
    #test_sat_crossing()
    test_sat_pc_mc()