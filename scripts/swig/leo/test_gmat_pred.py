import pickle
import pc_prediction as pc
from datetime import timedelta
from datetime import datetime
import cauchy_estimator as ce
import gmat_mce as gmce
from gmat_sat import *
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg', force=True)


def store_data(cache_dic, fpath, auto_yes=False):
    if auto_yes:
        with open(fpath, "wb") as handle:
            pickle.dump(cache_dic, handle)
            print("Stored data to:", fpath)
            return
    input_ok = False
    while not input_ok:
        input_ok = True
        ui = input("Would you like to store your data? (y/n)").lower()
        if (ui == 'y'):
            with open(fpath, "wb") as handle:
                pickle.dump(cache_dic, handle)
            print("Stored data to:", fpath)
        elif (ui == 'n'):
            print("Data not stored!")
        else:
            print("Unrecognized input")
            input_ok = False


def vel_for_rad(orbit, _rad_dir):
    M = 5.9722e24  # Mass of earth (kg)
    G = 6.674e-11  # m^3/(s^2 * kg) Universal Gravitation Constant
    mu = M*G  # meter
    r_earth = 6378
    r0 = 1000*(orbit + r_earth)
    v0 = (mu/r0)**0.5
    r0 /= 1000  # to KM
    v0 /= 1000  # to KM
    rad_dir = _rad_dir / np.linalg.norm(_rad_dir)
    _, _, vel_dirs = np.linalg.svd(rad_dir.reshape((1, 3)))
    return r0, v0, vel_dirs[1:], rad_dir


def test_sat_pc_mc():
    n = 7  # states
    data_path = file_dir + "/pylog/gmat7/pred/pc/"

    # Runtime Options
    # "gauss_realiz.pickle"
    # "sas_okay.pickle"
    # "1715885643.9563792.pickle"
    # "1715889415.040428.pickle"
    # "init_cond_w5_dt300.pickle"
    # '''''''''''''''''''''''''' #
    # "ic_w5_dt300_b300.pickle"
    # "ic_w5_dt300_b60.pickle"
    # "ic_w5_dt60_b60.pickle"
    # "hs_w5_dt300_b300.pickle"
    # "hs_w5_dt300_b60.pickle"
    # "hs_w5_dt60_b60.pickle"

    # if set to something, loads and adds to this dir, if set to nothing, creates a new directory
    mode = "sas"  # "sas"
    cached_dir = "foo_sas_180_debug2.pickle"
    with_filter_plots = False
    with_pred_plots = False
    with_density_jumps = False
    with_mc_from_x0 = True  # True
    with_auto_save_yes = False
    with_no_initial_pred_error = True
    with_mce_contour_plot = True
    RSYS_MAX_TERMS_ESTM = 20000 # np.inf

    if cached_dir is "":
        # Some Initial Data and Declarations
        t0 = '11 Feb 2023 23:47:55.0'
        # x0 = np.array([550+6378, 0, 0, 0, 7.585175924227056, 0])

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

        # Primary
        x0_prim = np.array([
            4996.245288270519, 3877.946463086103, 2736.0432364171807,  # km
            -5.028093574446193, 5.575921341999267, 1.2698611722905329])  # km/s
        # Secondary 
        x0_sec = np.array([6267.308106510068, -1065.2042361596507, 2694.104501686107, -1.8179384332000135, -7.250275016073654, 1.3748775570376515])
        
        # x0 = np.array([
        #    -4.1687684994514830e+03,5.3553338507134049e+03,1.2393355399955583e+03,
        #    -5.8486240699389134e+00,-3.8472966137675524e+00,-2.9777449709740309e+00
        #    ])

        # orbit_height = 550 # km
        # rad_dir = [0,.2,.8]
        # p0n, v0n, v0ds, r0d = vel_for_rad(orbit_height, rad_dir)
        # v0d_idx = 0
        # x0 = np.concatenate( (p0n * r0d, v0n*v0ds[v0d_idx] ) )

        filt_dt = 60
        filt_orbits = 12
        pred_dt = 120.0
        pred_steps = 5040
        R_prim = 0.003  # km
        R_sec = 0.003  # km
        std_gps_noise = .0075  # kilometers
        std_Cd = 0.0013
        tau_Cd = 21600
        sas_Cd = mode

        cache_dic = {
            'mode': mode,
            'with_density_jumps': with_density_jumps,
            'with_no_initial_pred_error': with_no_initial_pred_error,
            'R_prim': R_prim, 'R_sec': R_sec,
            't0': t0,
            'x0_prim': x0_prim,
            'x0_sec': x0_sec,
            'std_Cd': std_Cd,
            'tau_Cd': tau_Cd,
            'std_gps_noise': std_gps_noise,

            # --- Simulation Length and True Trajectories --- #
            # Number of orbits to run the simulation, the KF filter, and the time between steps
            'filt_orbits': filt_orbits,
            'filt_dt': filt_dt,
            # State, measurement and noise histories of the primary and secondary satellite
            'prim_sim': None,  # 'kf_prim_sim' : None,
            'sec_sim': None,  # 'kf_sec_sim' : None,
            # Number of steps for prediction, and the time between steps
            'pred_steps': pred_steps,
            'pred_dt': pred_dt,
            # Trajectory of the satellites starting at truth after filtering period, under noise expectation of zero, for pred_steps at pred_dt per step
            'prim_pred': None,  # 'prim_pred_hist' : None,
            'sec_pred': None,  # 'sec_pred_hist' : None,
            # Start / end indices for time of closest approach for truth under noise expectation of zero, and its relevant data regarding expected tca starting at truth after filtering period
            'itca_window_idxs': None,
            'itca_data': None,
            # The position, velocity and acceleration for truth under noise expectation of zero at TCA
            'prim_tca': None,
            'sec_tca': None,

            # --- Kalman Filter Data --- #
            # State estimates and covariances of the primary / secondary satellite during the filtering simulation period
            'kf_prim_sim': None,
            'kf_sec_sim': None,
            # Trajectory of the satellites starting at KF estimate after filtering period, under noise expectation of zero, for pred_steps at pred_dt per step
            # These will be equal to prim_pred/sec_pred, respectively if with_no_initial_pred_error == True
            # That is -> \hat{x}_k is set equal to x_k for prim/sec if with_no_initial_pred_error == True
            'kf_prim_pred': None,
            'kf_sec_pred': None,
            # Relevant data regarding tca starting at estimate after filtering period
            # This will be equal to itca_data if with_no_initial_pred_error == True
            'kf_itca_data': None,
            # The position, velocity and acceleration for KF estimates under noise expectation of zero at TCA
            # This will be equal to prim_tca/sec_itca if with_no_initial_pred_error == True
            'kf_prim_tca': None,
            'kf_sec_tca': None,

            # --- Monte Carlo Realizations of the Prediction Phase, Starting at End of Filtration, Under Gauss/SaS Noise Realizations --- #
            # Starting at xk under gauss/sas density forcing realizations
            'mc_prim_tcas': None,
            'mc_sec_tcas': None,
            # Starting at the sampled N(xk,Pk_kf) value under gauss/sas density forcing realizations
            'mc_prim_sample_tcas': None,  # taken around KF
            'mc_sec_sample_tcas': None,  # taken around KF
            'mc_itca_info': None,

            # --- MCE Data --- #
            # Launch Settings for the MCE
            'mce_info': None,
            # State estimates and covariances of the primary / secondary satellite during the filtering simulation period
            'mce_prim_sim': None,
            'mce_sec_sim': None,
            # Trajectory of the satellites starting at MCE estimate after filtering period, under noise expectation of zero, for pred_steps at pred_dt per step
            # These will be equal to prim_pred/sec_pred, respectively if with_no_initial_pred_error == True
            # That is -> \hat{x}_k is set equal to x_k for prim/sec if with_no_initial_pred_error == True
            'mce_prim_pred': None,
            'mce_sec_pred': None,
            # Relevant data regarding tca starting at estimate after filtering period
            # This will be equal to itca_data if with_no_initial_pred_error == True
            'mce_itca_data': None,
            # The position, velocity and acceleration for MCE estimates under noise expectation of zero at TCA
            # This will be equal to prim_tca/sec_itca if with_no_initial_pred_error == True
            'mce_prim_tca': None,
            'mce_sec_tca': None,

            # --- Relative MCE System projected onto encounter plane ---- #
            'mce_rsys' : None
        }
        timestamp = str(time.time()) + ".pickle"
        fpath = data_path + timestamp

    else:
        fpath = data_path + cached_dir
        with open(fpath, 'rb') as handle:
            cache_dic = pickle.load(handle)
            if cache_dic['x0_prim'] is None:
                cache_dic['x0_prim'] = cache_dic['x0'].copy()
                cache_dic['x0_sec'] = np.zeros(7)
            if 'mce_rsys' not in cache_dic:
                cache_dic['mce_rsys'] = None


        t0 = cache_dic['t0']
        x0_prim = cache_dic['x0_prim']
        x0_sec = cache_dic['x0_sec']
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
    Wn = np.zeros((n, n))
    if sas_Cd == "gauss":
        scale_pv = 1000
        scale_d = 20.0
        sas_alpha = 2.0
    else:
        scale_pv = 10000
        scale_d = 10000
        sas_alpha = 1.3
        # scale_pv = 500
        # scale_d = 250
    Wn[0:6, 0:6] = W6.copy()
    Wn[0:6, 0:6] *= scale_pv
    # Process Noise for changes in Cd
    if sas_Cd != "gauss":
        Wn[6, 6] = (1.3898 * std_Cd)**2  # tune to cauchy LSF
    else:
        Wn[6, 6] = std_Cd**2
    Wn[6, 6] *= scale_d  # 0 # Tunable w/ altitude
    V = np.eye(3) * std_gps_noise**2
    I7 = np.eye(7)
    H = np.hstack((np.eye(3), np.zeros((3, 4))))
    STM_order = 3
    P0_kf = np.eye(n) * (0.001)**2
    P0_kf[6, 6] = .01
    p_x0 = np.zeros(n)
    p_x0[0:6] = x0_prim.copy()
    s_x0 = np.zeros(n)
    if x0_sec is None:
        s_x0[0:6] = x0_prim.copy() 
        s_x0[3:6] *= -1
    else:
        s_x0[0:6] = x0_sec.copy()

    # Filtering for primary satellite
    if cached_dir is "":
        print("Running EKF For Primary")
        (p_xs, p_zs, p_ws, p_vs), (p_xs_kf, p_Ps_kf) = pc.simulate_then_run_ekf(t0, p_x0, P0_kf, filt_dt,
                                                                                sas_Cd, std_Cd, tau_Cd, sas_alpha, filt_orbits, std_gps_noise, with_density_jumps, STM_order, Wn, H, V)
        cache_dic['prim_sim'] = (
            p_xs.copy(), p_zs.copy(), p_ws.copy(), p_vs.copy())
        cache_dic['kf_prim_sim'] = (p_xs_kf.copy(), p_Ps_kf.copy())
    else:
        p_xs, p_zs, p_ws, p_vs = cache_dic['prim_sim']
        p_xs_kf, p_Ps_kf = cache_dic['kf_prim_sim']
    if with_filter_plots:
        # Plot Primary
        print("Primary Sateliite KF Run:")
        ce.plot_simulation_history(
            None, (p_xs, p_zs, p_ws, p_vs), (p_xs_kf, p_Ps_kf), scale=1)

    # Now Repeat for secondary Satellite
    if cached_dir is "":
        print("Running EKF For Secondary")
        (s_xs, s_zs, s_ws, s_vs), (s_xs_kf, s_Ps_kf) = pc.simulate_then_run_ekf(t0, s_x0, P0_kf, filt_dt,
                                                                                sas_Cd, std_Cd, tau_Cd, sas_alpha, filt_orbits, std_gps_noise, with_density_jumps, STM_order, Wn, H, V)
        cache_dic['sec_sim'] = (s_xs.copy(), s_zs.copy(),
                                s_ws.copy(), s_vs.copy())
        cache_dic['kf_sec_sim'] = (s_xs_kf.copy(), s_Ps_kf.copy())
    else:
        s_xs, s_zs, s_ws, s_vs = cache_dic['sec_sim']
        s_xs_kf, s_Ps_kf = cache_dic['kf_sec_sim']
    if with_filter_plots:
        # Plot Secondary
        print("Secondary Satelite KF Run:")
        ce.plot_simulation_history(
            None, (s_xs, s_zs, s_ws, s_vs), (s_xs_kf, s_Ps_kf), scale=1)

    # Time at the start of prediction
    # Number of filtering steps * filt_dt
    filt_time = (p_xs_kf.shape[0]-1) * filt_dt
    t0_pred = datetime.strptime(
        t0, "%d %b %Y %H:%M:%S.%f") + timedelta(seconds=filt_time)

    # Prediction for primary and secondary satellites 7-days into future
    if cached_dir is "":
        if with_no_initial_pred_error:
            print("Running Prediction For Primary")
            # The KF of the primary will be the same as the expected primary due to removing filtering error at end of filtering period
            p_xpred = p_xs[-1].copy()
            p_Ppred = p_Ps_kf[-1].copy()
            p_pks, p_vks, p_cdks, p_aks, p_Pks_kf = pc.sat_prediction(
                t0_pred, p_xpred, p_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            cache_dic['kf_prim_pred'] = (p_pks.copy(), p_vks.copy(
            ), p_cdks.copy(), p_aks.copy(), p_Pks_kf.copy())
            cache_dic['prim_pred'] = (
                p_pks.copy(), p_vks.copy(), p_cdks.copy(), p_aks.copy())
            # The KF of the secondary will be the same as the expected secondary due to removing filtering error at end of filtering period
            print("Running Prediction For Secondary")
            s_xpred = s_xs[-1].copy()
            s_Ppred = s_Ps_kf[-1].copy()
            s_pks, s_vks, s_cdks, s_aks, s_Pks_kf = pc.sat_prediction(
                t0_pred, s_xpred, s_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            cache_dic['kf_sec_pred'] = (s_pks.copy(), s_vks.copy(
            ), s_cdks.copy(), s_aks.copy(), s_Pks_kf.copy())
            cache_dic['sec_pred'] = (
                s_pks.copy(), s_vks.copy(), s_cdks.copy(), s_aks.copy())
        else:
            print("Running Prediction + EKF Prediction For Primary")
            # The KF of the primary will not be the same as the expected primary due to filtering error at end of filtering period
            p_xpred = p_xs_kf[-1].copy()
            p_Ppred = p_Ps_kf[-1].copy()
            p_pks_kf, p_vks_kf, p_cdks_kf, p_aks_kf, p_Pks_kf = pc.sat_prediction(
                t0_pred, p_xpred, p_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            cache_dic['kf_prim_pred'] = (p_pks_kf.copy(), p_vks_kf.copy(
            ), p_cdks_kf.copy(), p_aks_kf.copy(), p_Pks_kf.copy())
            p_xpred = p_xs[-1].copy()
            p_Ppred = None
            p_pks, p_vks, p_cdks, p_aks = pc.sat_prediction(
                t0_pred, p_xpred, p_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            cache_dic['prim_pred'] = (
                p_pks.copy(), p_vks.copy(), p_cdks.copy(), p_aks.copy())
            # The KF of the secondary will not be the same as the expected primary due to filtering error at end of filtering period
            print("Running Prediction + EKF Prediction For Secondary")
            s_xpred = s_xs_kf[-1].copy()
            s_Ppred = s_Ps_kf[-1].copy()
            s_pks, s_vks, s_cdks, s_aks, s_Pks_kf = pc.sat_prediction(
                t0_pred, s_xpred, s_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            cache_dic['kf_sec_pred'] = (s_pks.copy(), s_vks.copy(
            ), s_cdks.copy(), s_aks.copy(), s_Pks_kf.copy())
            s_xpred = s_xs[-1].copy()
            s_Ppred = None
            s_pks, s_vks, s_cdks, s_aks = pc.sat_prediction(
                t0_pred, s_xpred, s_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            cache_dic['sec_pred'] = (
                s_pks.copy(), s_vks.copy(), s_cdks.copy(), s_aks.copy())
        store_data(cache_dic, fpath, with_auto_save_yes)
    else:
        p_pks, p_vks, p_cdks, p_aks = cache_dic['prim_pred']
        p_pks_kf, p_vks_kf, p_cdks_kf, p_aks_kf, p_Pks_kf = cache_dic['kf_prim_pred']
        s_pks, s_vks, s_cdks, s_aks = cache_dic['sec_pred']
        s_pks_kf, s_vks_kf, s_cdks_kf, s_aks_kf, s_Pks_kf = cache_dic['kf_sec_pred']

    # Plot relative differences and choose a window of time where both satellite are very close to each other, 7-days out in future
    if with_pred_plots:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.title(
            "Predicted trajectories (primary=red, secondary=blue) over 7-day lookahead:")
        ax.plot(p_pks[:, 0], p_pks[:, 1], p_pks[:, 2], color='r')
        ax.plot(s_pks[:, 0], s_pks[:, 1], s_pks[:, 2], color='b')
        ax.scatter(p_pks[0, 0], p_pks[0, 1], p_pks[0, 2], color='k', s=80)
        ax.set_xlabel("x-axis (km)")
        ax.set_ylabel("y-axis (km)")
        ax.set_zlabel("z-axis (km)")
        fig2 = plt.figure()
        plt.suptitle(
            "Norm of predicted satellite seperation over 7-day lookahead")
        r_norms = np.linalg.norm(s_pks - p_pks, axis=1)
        # (np.arange(r_norms.size) * pred_dt) / (24*60*60),
        plt.plot(np.arange(r_norms.size), r_norms)
        plt.ylabel("Seperation (km)")
        plt.xlabel("# days lookahead")
        plt.show()

    # If user wishes to see prediction plot, ask if they would like to create MC over another window
    run_itca = False
    if cache_dic['itca_data'] is None:
        # [4650, 4670] # Could manually reset this here
        cache_dic['itca_window_idxs'] = [4652, 4675]
        run_itca = True
        # Prediction Step for ITCA to be apart of 
        # Use this code snippet to take the nominal predicted trajectory, and come up with the expected TCA
        #preffered_itca_step = 4656
        #t0_filt = time_string_2_datetime(t0)
        #tf_pred = t0_pred + timedelta(seconds=pred_dt * preffered_itca_step)
        #xf_pred = np.concatenate((p_pks[preffered_itca_step], -p_vks[preffered_itca_step]))
        #s_x_back_prop = pc.backprop_sat_from_xftf_to_x0t0(xf_pred, tf_pred, t0_filt)
        #foo = 3
    if (cache_dic['itca_data'] is not None) and with_pred_plots:
        print("Old ITCA Left and Right Hand Side Window Indices: ",
              cache_dic['itca_window_idxs'][0], cache_dic['itca_window_idxs'][1])
        while True:
            is_run = input(
                "Would you like to rerun Iterative Time of Closest Approach (ITCA)? (Enter y or n): ")
            if is_run == 'y':
                run_itca = True
                print("Re-running ITCA!")
                valid_range = (0, r_norms.size-1)
                is_ok = False
                while True:
                    cache_dic['itca_window_idxs'][0] = int(input(
                        "   Enter index for itca window start: i.e., a value between [{},{}]".format(valid_range[0], valid_range[1])))
                    if ((cache_dic['itca_window_idxs'][0] >= valid_range[0]) and (cache_dic['itca_window_idxs'][0] <= valid_range[1])):
                        break
                    else:
                        print("Invalid Entery of {}. Try Again!".format(
                            cache_dic['itca_window_idxs'][0]))
                valid_range = (
                    cache_dic['itca_window_idxs'][0]+1, r_norms.size-1)
                while True:
                    cache_dic['itca_window_idxs'][1] = int(input(
                        "   Enter index for itca window end: i.e., a value between [{},{}]".format(valid_range[0], valid_range[1])))
                    if ((cache_dic['itca_window_idxs'][1] >= valid_range[0]) and (cache_dic['itca_window_idxs'][1] <= valid_range[1])):
                        break
                    else:
                        print("Invalid Entery of {}. Try Again!".format(
                            cache_dic['itca_window_idxs'][1]))
                print("Re-running ITCA with LHS/RHS indices of ",
                      cache_dic['itca_window_idxs'])
                break
            elif is_run == 'n':
                run_itca = False
                print("Not rerunning ITCA!")
                break
            else:
                print("Invalid entery. Try again!")

    # Now run the iterative time of closest approach algorithm if desired
    if run_itca:
        print("Running the Iterative Time of Closest Approach Algorithm for Expected Encounter Plane")
        # Run iterative time of closest approach over this window, find exact point of closest approach
        start_idx = cache_dic['itca_window_idxs'][0]
        end_idx = cache_dic['itca_window_idxs'][1]
        # below the KF covariance is used in the nominal calculation...which is harmless
        # if we do not use with_no_initial_pred_error, p_Ptc_kf and s_Ptc_kf are changed
        (i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c), \
            (p_xtc, p_Ptc_kf), \
            (s_xtc, s_Ptc_kf) = pc.find_tca_and_prim_sec_stats(t0_pred, start_idx, end_idx, pred_dt,
                                                               p_pks, p_vks, p_aks, p_cdks, p_Pks_kf,
                                                               s_pks, s_vks, s_aks, s_cdks, s_Pks_kf,
                                                               sas_Cd, std_Cd, tau_Cd, sas_alpha, with_pred_plots)
        cache_dic['itca_data'] = (
            i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c)
        cache_dic['prim_tca'] = (p_xtc, None)
        cache_dic['sec_tca'] = (s_xtc, None)
        # If we have no initial prediction error, kf will match the expected true encounter plane
        if with_no_initial_pred_error:
            cache_dic['kf_itca_data'] = (
                i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c)
            cache_dic['kf_prim_tca'] = (p_xtc, p_Ptc_kf)
            cache_dic['kf_sec_tca'] = (s_xtc, s_Ptc_kf)
        # If we have initial prediction error, run iterative time of closest approach over this window using the KFs predicted state trajectory and variances
        else:
            print(
                "Running the Iterative Time of Closest Approach Algorithm for EKF Expected Encounter Plane")
            (i_star_lhs_kf, t_lhs_kf, t_c_kf, pp_c_kf, pv_c_kf, sp_c_kf, sv_c_kf), \
                (p_xtc_kf, p_Ptc_kf), \
                (s_xtc_kf, s_Ptc_kf) = pc.find_tca_and_prim_sec_stats(t0_pred, start_idx, end_idx, pred_dt,
                                                                      p_pks_kf, p_vks_kf, p_aks_kf, p_cdks_kf, p_Pks_kf,
                                                                      s_pks_kf, s_vks_kf, s_aks_kf, s_cdks_kf, s_Pks_kf,
                                                                      sas_Cd, std_Cd, tau_Cd, sas_alpha, with_pred_plots)
            cache_dic['kf_itca_data'] = (
                i_star_lhs_kf, t_lhs_kf, t_c_kf, pp_c_kf, pv_c_kf, sp_c_kf, sv_c_kf)
            cache_dic['kf_prim_tca'] = (p_xtc_kf, p_Ptc_kf)
            cache_dic['kf_sec_tca'] = (s_xtc_kf, s_Ptc_kf)

        store_data(cache_dic, fpath, with_auto_save_yes)
    else:
        i_star_lhs, t_lhs, t_c, pp_c, pv_c, sp_c, sv_c = cache_dic['itca_data']
        p_xtc, _ = cache_dic['prim_tca']
        s_xtc, _ = cache_dic['sec_tca']
        p_xtc_kf, p_Ptc_kf = cache_dic['kf_prim_tca']
        s_xtc_kf, s_Ptc_kf = cache_dic['kf_sec_tca']
        start_idx = cache_dic['itca_window_idxs'][0]
        end_idx = cache_dic['itca_window_idxs'][1]

    # Repeat the following two steps for a select number of monte carlos ... caching the mc trial data as you go... this is expensive
    # int( input("How many MC trials would you like to add: (i.e, 0 to 10000): ") )
    mc_trials = 0
    # pc.log_mc_trials(mc_trials, with_mc_from_x0, t0_pred, pred_dt,
    #    p_xs[-1].copy(), p_Ps_kf[-1].copy(), s_xs[-1].copy(), s_Ps_kf[-1].copy(),
    #    sas_Cd, std_Cd, tau_Cd, sas_alpha, std_gps_noise,
    #    i_star_lhs, t_c, t_lhs,
    #    cache_dic, fpath)

    pc.log_mc_trials2(mc_trials, with_mc_from_x0, t0_pred, pred_dt,
                      p_xs[-1].copy(), p_Ps_kf[-1].copy(), s_xs[-1].copy(), s_Ps_kf[-1].copy(),
                      sas_Cd, std_Cd, tau_Cd, sas_alpha, start_idx, end_idx, cache_dic, fpath)

    # Run the MCE for ~50 steps before the end of filtering for both the primary and the secondary,
    # Take the obtained 7x7 covariance, and project it to the TCA point
    if cache_dic['mce_info'] is None:
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
        mce_steps = 60
        mce_num_windows = 3  # 5
        mce_dt = 300.0  # sec
        beta_dt = 60  # mce_dt # filt_dt # 60 # sec
        # gmce.reinitialize_func_init_cond # reinitialize_func_H_summation # reinitialize_func_speyer
        mce_init_func = gmce.reinitialize_func_H_summation
        mce_start_by_time_prop = True

        cache_dic['mce_info'] = (
            ('mce_steps', mce_steps),
            ('mce_num_windows', mce_num_windows),
            ('mce_dt', mce_dt),
            ('beta_dt', beta_dt),
            ('mce_init_func', mce_init_func.__name__),
            ('mce_start_by_time_prop', mce_start_by_time_prop)
        )

        assert (mce_dt % filt_dt) == 0
        mce_dt_factor = int(mce_dt / filt_dt)
        sim_steps = p_zs.shape[0]
        mce_msmt_idx = (sim_steps-1) - ((mce_steps-1) * mce_dt_factor)
        mce_t0 = time_string_2_datetime(
            t0) + timedelta(seconds=mce_msmt_idx * filt_dt)

        mce_gamma = np.ones(3) * std_gps_noise * 1000 * \
            ce.GAUSSIAN_TO_CAUCHY_NOISE
        # The idea is to model the additive change in atmospheric density, which Russell uses with
        # the continuous time model of: dx/dt = -\frac{1}{\tau}x + w, where x is the change in atmospheric density from its nominal as atms = atms_nom*(1+x) and w is a white noise increment
        # when fit to f10.7 data, russell gets back a scaling parameter value of 0.0013 and an alpha of approx 1.3 and uses this for scalar Cauchy at dt=60
        # The MCE uses the representation "x_k+1 = \Phi_k x_k + \Gamma_k w_k"
        # The idea now is to get what w_k (beta) above should be knowing that we can compute Gamma_k easily for the scalar LTI system above (Gamma_k = Gamma for the atmospheric dens state)
        # Now, for a Gaussian, the covariance of the scalar change in atmospheric density would be W_k = (\sigma_c2g * 0.0013)**2 where sigma_c2g converts cauchy to Gaussian -> close enough for alpha=1.3
        # Solving for \Gamma_k above by holding w_k constant across the time interval we get \Gamma_k = tau * ( 1 - exp(-dt/tau) )
        # Therefore, we want to set E[(\Gamma_k w_k)^2] = (\sigma_c2g * 0.0013)**2 = W_k
        # The std dev of w_k is therefore std(w_k) = (\sigma_c2g * 0.0013) / Gamma_k
        # The cauchy parameter to use is therefore beta = std(w_k) * ce.GAUSSIAN_TO_CAUCHY_NOISE
        # Note that below if beta_dt > 60 sec, we are underbounding the noise level...which may still be appropriate
        beta_gauss = (std_Cd * ce.CAUCHY_TO_GAUSSIAN_NOISE) / \
            (tau_Cd * (1.0 - np.exp(-beta_dt/tau_Cd)))
        beta_cauchy = beta_gauss * ce.GAUSSIAN_TO_CAUCHY_NOISE
        mce_beta = np.array([beta_cauchy])

        # Begin Primary
        print("Running MCE for Primary Satellite!")
        mce_msmt_idxs = np.arange(mce_msmt_idx, sim_steps, mce_dt_factor)
        mce_p_zs = p_zs[mce_msmt_idxs]
        mce_p_xs = p_xs[mce_msmt_idxs]

        # USING TIME PROPAGATION START
        if mce_start_by_time_prop:
            mce_x0 = p_xs_kf[mce_msmt_idx-mce_dt_factor].copy()
            mce_x0[6] = 0.01
            fermSat = FermiSatelliteModel(
                mce_t0 - timedelta(seconds=mce_dt), mce_x0.copy(), mce_dt, gmat_print=False)
            fermSat.create_model()
            fermSat.set_solve_for(field="Cd", dist=mode, scale=std_Cd,
                                  tau=tau_Cd, alpha=2.0 if mode == "gauss" else 1.3)
            fermSat.reset_state(mce_x0, 0)
            mce_p_Jac = fermSat.get_jacobian_matrix()
            mce_p_Jac[0:6, 6] *= 1000
            # mce_Phi0 = (np.eye(7) + mce_p_Jac * filt_dt + mce_p_Jac @ mce_p_Jac * filt_dt**2 / 2 + mce_p_Jac @ mce_p_Jac @ mce_p_Jac * filt_dt**3 / 6)
            mce_Phi0 = (np.eye(7) + mce_p_Jac * mce_dt + mce_p_Jac @ mce_p_Jac *
                        mce_dt**2 / 2 + mce_p_Jac @ mce_p_Jac @ mce_p_Jac * mce_dt**3 / 6)
            mce_A0 = mce_Phi0.T
            mce_p0 = gmce.mce_naive_p0.copy()
            mce_b0 = np.zeros(7)
            mce_p_x0bar = fermSat.step().copy()
            mce_p_x0bar[0:6] *= 1000
            fermSat.clear_model()
        # USE THE KF To Initialize the MCE
        else:
            mce_p_x0 = p_xs_kf[mce_msmt_idx].copy()
            mce_p_x0[0:6] *= 1000
            mce_p_P0 = p_Ps_kf[mce_msmt_idx].copy()
            mce_p_P0[0:6, 0:6] *= 1000**2
            mce_p_P0[0:6, 6] *= 1000
            mce_p_P0[6, 0:6] *= 1000
            mce_p_dx0 = np.random.multivariate_normal(np.zeros(7), mce_p_P0/8)
            mce_p_dx0[6] = 0.01
            mce_p_x0bar = mce_p_x0-mce_p_dx0
            mce_p_dz0 = 1000*mce_p_zs[0][0] - H[0] @ mce_p_x0bar
            mce_A0, mce_p0, mce_b0 = ce.speyers_window_init(
                mce_p_dx0, mce_p_P0, H[0], mce_gamma[0], mce_p_dz0)

        # Begin Primary MCE Estimation
        mce_other_params = p_Ps_kf[mce_msmt_idxs]
        p_cauchyEst = gmce.GmatMCE(mce_num_windows, mce_t0, mce_p_x0bar, mce_dt,
                                   mce_A0, mce_p0, mce_b0, mce_beta, mce_gamma,
                                   Cd_dist=mode, std_Cd=std_Cd, tau_Cd=tau_Cd,
                                   win_reinitialize_func=mce_init_func,
                                   win_reinitialize_params=mce_other_params,
                                   debug_print=True, mce_print=True)

        len_mce_p_zxs = mce_p_zs.shape[0]
        for i in range(len_mce_p_zxs):
            zk = mce_p_zs[i]
            xk = mce_p_xs[i]
            _zk = zk.copy() * 1000  # km -> m
            _xk = xk.copy()
            _xk[0:6] *= 1000  # km -> m
            p_cauchyEst.sim_step(zk=_zk, x_truth=_xk, is_inputs_meters=True, last_step = ( i == (len_mce_p_zxs-1) ) )

        # If with_mce_contour_plot is set, propagate the density function of the primary to expected TCA
        if with_mce_contour_plot:
            p_mce_tca_idx = p_cauchyEst.pred_to_tca(t0_pred, pred_dt, i_star_lhs, t_lhs, t_c, max_terms = RSYS_MAX_TERMS_ESTM, with_propagate_drag_estimate=True, xhat_pred_t0 = p_xs[-1])
            p_mce_tca = p_cauchyEst.cauchyEsts[p_mce_tca_idx]
            p_cauchyEst.teardown_except_selected_estimators(p_mce_tca_idx)

        # Convert the data from meters to Km
        p_xs_mce = np.array(p_cauchyEst.xhats)
        p_xs_mce[:, 0:6] /= 1000
        p_xs_mce_avg = np.array(p_cauchyEst.avg_xhats)
        p_xs_mce_avg[:, 0:6] /= 1000
        p_Ps_mce = np.array(p_cauchyEst.Phats)
        p_Ps_mce_avg = np.array(p_cauchyEst.avg_Phats)
        for pHat, apHat in zip(p_Ps_mce, p_Ps_mce_avg):
            pHat[0:6, 0:6] /= 1000**2
            pHat[0:6, 6] /= 1000
            pHat[6, 0:6] /= 1000
            apHat[0:6, 0:6] /= 1000**2
            apHat[6, 0:6] /= 1000
            apHat[0:6, 6] /= 1000
        cache_dic['mce_prim_sim'] = (
            mce_msmt_idxs, p_xs_mce, p_Ps_mce, p_xs_mce_avg, p_Ps_mce_avg)
        if not with_mce_contour_plot:
            p_cauchyEst.teardown()

        # Begin Secondary MCE Estimation
        print("Running MCE for Secondary Satellite!")
        mce_s_zs = s_zs[mce_msmt_idxs]
        mce_s_xs = s_xs[mce_msmt_idxs]

        # USING TIME PROPAGATION START
        if mce_start_by_time_prop:
            mce_x0 = s_xs_kf[mce_msmt_idx-mce_dt_factor].copy()
            mce_x0[6] = 0.01
            fermSat = FermiSatelliteModel(
                mce_t0 - timedelta(seconds=mce_dt), mce_x0.copy(), mce_dt, gmat_print=False)
            fermSat.create_model()
            fermSat.set_solve_for(field="Cd", dist=mode, scale=std_Cd,
                                  tau=tau_Cd, alpha=2.0 if mode == "gauss" else 1.3)
            fermSat.reset_state(mce_x0, 0)
            mce_s_Jac = fermSat.get_jacobian_matrix()
            mce_s_Jac[0:6, 6] *= 1000
            # mce_Phi0 = (np.eye(7) + mce_s_Jac * filt_dt + mce_s_Jac @ mce_s_Jac * filt_dt**2 / 2 + mce_s_Jac @ mce_s_Jac @ mce_s_Jac * filt_dt**3 / 6)
            mce_Phi0 = (np.eye(7) + mce_s_Jac * mce_dt + mce_s_Jac @ mce_s_Jac *
                        mce_dt**2 / 2 + mce_s_Jac @ mce_s_Jac @ mce_s_Jac * mce_dt**3 / 6)
            mce_A0 = mce_Phi0.T
            mce_p0 = gmce.mce_naive_p0.copy()
            mce_b0 = np.zeros(7)
            mce_s_x0bar = fermSat.step().copy()
            mce_s_x0bar[0:6] *= 1000
            fermSat.clear_model()
        # USE THE KF To Initialize the MCE
        else:
            mce_s_x0 = s_xs_kf[mce_msmt_idx].copy()
            mce_s_x0[0:6] *= 1000
            mce_s_P0 = s_Ps_kf[mce_msmt_idx].copy()
            mce_s_P0[0:6, 0:6] *= 1000**2
            mce_s_P0[0:6, 6] *= 1000
            mce_s_P0[6, 0:6] *= 1000
            mce_s_dx0 = np.random.multivariate_normal(np.zeros(7), mce_s_P0/8)
            mce_s_dx0[6] = 0.01
            mce_s_x0bar = mce_s_x0-mce_s_dx0
            mce_s_dz0 = 1000*mce_s_zs[0][0] - H[0] @ mce_s_x0bar
            mce_A0, mce_p0, mce_b0 = ce.speyers_window_init(
                mce_s_dx0, mce_s_P0, H[0], mce_gamma[0], mce_s_dz0)

        # Begin Secondary MCE Estimation
        mce_other_params = s_Ps_kf[mce_msmt_idxs]
        s_cauchyEst = gmce.GmatMCE(mce_num_windows, mce_t0, mce_s_x0bar, mce_dt,
                                   mce_A0, mce_p0, mce_b0, mce_beta, mce_gamma,
                                   Cd_dist=mode, std_Cd=std_Cd, tau_Cd=tau_Cd,
                                   win_reinitialize_func=mce_init_func,
                                   win_reinitialize_params=mce_other_params,
                                   debug_print=True, mce_print=True)
        
        len_mce_s_zxs = mce_s_zs.shape[0]
        for i in range(len_mce_s_zxs):
            zk = mce_s_zs[i]
            xk = mce_s_xs[i]
            _zk = zk.copy() * 1000  # km -> m
            _xk = xk.copy()
            _xk[0:6] *= 1000  # km -> m
            s_cauchyEst.sim_step(zk=_zk, x_truth=_xk, is_inputs_meters=True, last_step = ( i == (len_mce_s_zxs-1) ) )
        
        # If with_mce_contour_plot is set, propagate the density function of the secondary to expected TCA
        if with_mce_contour_plot:
            s_mce_tca_idx = s_cauchyEst.pred_to_tca(t0_pred, pred_dt, i_star_lhs, t_lhs, t_c, max_terms = RSYS_MAX_TERMS_ESTM, with_propagate_drag_estimate=True, xhat_pred_t0 = s_xs[-1])
            s_mce_tca = s_cauchyEst.cauchyEsts[s_mce_tca_idx]
            s_cauchyEst.teardown_except_selected_estimators(s_mce_tca_idx)

        # Convert the data from meters to Km
        s_xs_mce = np.array(s_cauchyEst.xhats)
        s_xs_mce[:, 0:6] /= 1000
        s_xs_mce_avg = np.array(s_cauchyEst.avg_xhats)
        s_xs_mce_avg[:, 0:6] /= 1000
        s_Ps_mce = np.array(s_cauchyEst.Phats)
        s_Ps_mce_avg = np.array(s_cauchyEst.avg_Phats)
        for pHat, apHat in zip(s_Ps_mce, s_Ps_mce_avg):
            pHat[0:6, 0:6] /= 1000**2
            pHat[0:6, 6] /= 1000
            pHat[6, 0:6] /= 1000
            apHat[0:6, 0:6] /= 1000**2
            apHat[6, 0:6] /= 1000
            apHat[0:6, 6] /= 1000
        cache_dic['mce_sec_sim'] = (
            mce_msmt_idxs, s_xs_mce, s_Ps_mce, s_xs_mce_avg, s_Ps_mce_avg)
        if not with_mce_contour_plot:
            s_cauchyEst.teardown()

        # The MCE of the primary/secondary will be the same as the expected primary/secondary (and the KF's) due to removing filtering error at end of filtering period
        if with_no_initial_pred_error:
            p_xpred = p_xs[-1]
            p_Ppred = p_Ps_mce[-1]  # p_Ps_mce_avg[-1]
            s_xpred = s_xs[-1]
            s_Ppred = s_Ps_mce[-1]  # s_Ps_mce_avg[-1]
            # p_pks_mce, p_vks_mce, p_cdks_mce, p_aks_mce, p_Pks_mce = pc.sat_prediction(t0_pred, p_xpred, p_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            # (p_pks_mce, p_vks_mce, p_cdks_mce, p_aks_mce, p_Pks_mce)
            cache_dic['mce_prim_pred'] = None
            # s_pks_mce, s_vks_mce, s_cdks_mce, s_aks_mce, s_Pks_mce = pc.sat_prediction(t0_pred, s_xpred, s_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            # (s_pks_mce, s_vks_mce, s_cdks_mce, s_aks_mce, s_Pks_mce)
            cache_dic['mce_sec_pred'] = None
            # is same as expected: kf == mce == expected
            cache_dic['mce_itca_data'] = cache_dic['kf_itca_data']
            print("Stepping End of MCEs Filtration to Expected TCA For Primary Satellite")
            p_xtc_mce, p_Ptc_mce = pc.step_sat_stats_to_tca(
                t0_pred, p_xpred, p_Ppred, pred_dt, i_star_lhs, t_lhs, t_c, sas_Cd, std_Cd, tau_Cd, sas_alpha, STM_order)
            cache_dic['mce_prim_tca'] = (p_xtc_mce, p_Ptc_mce)
            print(
                "Stepping End of MCEs Filtration to Expected TCA For Secondary Satellite")
            s_xtc_mce, s_Ptc_mce = pc.step_sat_stats_to_tca(
                t0_pred, s_xpred, s_Ppred, pred_dt, i_star_lhs, t_lhs, t_c, sas_Cd, std_Cd, tau_Cd, sas_alpha, STM_order)
            cache_dic['mce_sec_tca'] = (s_xtc_mce, s_Ptc_mce)
        else:
            p_xpred = p_xs_mce[-1]  # p_xs_mce_avg[-1]
            p_Ppred = p_Ps_mce[-1]  # p_Ps_mce_avg[-1]
            s_xpred = s_xs_mce[-1]  # s_Ps_mce_avg[-1]
            s_Ppred = s_Ps_mce[-1]  # s_Ps_mce_avg[-1]
            print("Running MCE Prediction For Primary")
            p_pks_mce, p_vks_mce, p_cdks_mce, p_aks_mce, p_Pks_mce = pc.sat_prediction(
                t0_pred, p_xpred, p_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            cache_dic['mce_prim_pred'] = (
                p_pks_mce, p_vks_mce, p_cdks_mce, p_aks_mce, p_Pks_mce)
            print("Running MCE Prediction For Secondary")
            s_pks_mce, s_vks_mce, s_cdks_mce, s_aks_mce, s_Pks_mce = pc.sat_prediction(
                t0_pred, s_xpred, s_Ppred, pred_dt, pred_steps, sas_Cd, std_Cd, tau_Cd, sas_alpha)
            cache_dic['mce_sec_pred'] = (
                s_pks_mce, s_vks_mce, s_cdks_mce, s_aks_mce, s_Pks_mce)
            print(
                "Running the Iterative Time of Closest Approach Algorithm for MCE Expected Encounter Plane")
            (i_star_lhs_mce, t_lhs_mce, t_c_mce, pp_c_mce, pv_c_mce, sp_c_mce, sv_c_mce), \
                (p_xtc_mce, p_Ptc_mce), \
                (s_xtc_mce, s_Ptc_mce) = pc.find_tca_and_prim_sec_stats(t0_pred, start_idx, end_idx, pred_dt,
                                                                        p_pks_mce, p_vks_mce, p_aks_mce, p_cdks_mce, p_Pks_mce,
                                                                        s_pks_mce, s_vks_mce, s_aks_mce, s_cdks_mce, s_Pks_mce,
                                                                        sas_Cd, std_Cd, tau_Cd, sas_alpha, with_pred_plots)
            cache_dic['mce_itca_data'] = (
                i_star_lhs_mce, t_lhs_mce, t_c_mce, pp_c_mce, pv_c_mce, sp_c_mce, sv_c_mce)
            cache_dic['mce_prim_tca'] = (p_xtc_mce, p_Ptc_mce)
            cache_dic['mce_sec_tca'] = (s_xtc_mce, s_Ptc_mce)
        
        # If we are forming the contour plot of the encounter plane, go ahead and do so now
        if with_mce_contour_plot:
            print("Forming Contour Plot of the 2D RSYS Projected onto Conjunction Plane at TCA")
            # CHANGE THESE BASED ON THE PLOTS BELOW FIRST 
            # In Meters
            xlow = -50.0
            xhigh = 50.0
            delta_x = 0.2
            ylow = -50.0
            yhigh = 50.0
            delta_y = 0.2
            rsys_Xs, rsys_Ys, rsys_Zs, rsys_mean, rsys_var = gmce.form_short_encounter_contour_plot(s_mce_tca, p_mce_tca, xlow, xhigh, delta_x, ylow, yhigh, delta_y)
            cache_dic['mce_rsys'] = (rsys_Xs, rsys_Ys, rsys_Zs, rsys_mean, rsys_var)

        # Save Data
        store_data(cache_dic, fpath, with_auto_save_yes) #with_auto_save_yes # False
    else:
        p_xtc_mce, p_Ptc_mce = cache_dic['mce_prim_tca']# if cache_dic['mce_prim_tca'] else None, None
        s_xtc_mce, s_Ptc_mce = cache_dic['mce_sec_tca']# if cache_dic['mce_prim_tca'] else None, None
        mce_msmt_idxs, p_xs_mce, p_Ps_mce, p_xs_mce_avg, p_Ps_mce_avg = cache_dic['mce_prim_sim']# if cache_dic['mce_prim_tca'] else None, None, None, None, None
        mce_msmt_idxs, s_xs_mce, s_Ps_mce, s_xs_mce_avg, s_Ps_mce_avg = cache_dic['mce_sec_sim']# if cache_dic['mce_prim_tca'] else None, None, None, None, None
        mce_steps = cache_dic['mce_info'][0][1]# if cache_dic['mce_prim_tca'] else None
        mce_dt = cache_dic['mce_info'][2][1]# if cache_dic['mce_prim_tca'] else None

    if True:
        gmce.plot_against_kf(mce_msmt_idxs, p_xs, p_xs_kf, p_Ps_kf, 
                            None, None,
                            #p_xs_mce, p_Ps_mce,
                            p_xs_mce_avg, p_Ps_mce_avg, 
                            filt_dt, mce_dt, sig=1, title_prefix='Primary Satellite')
        gmce.plot_against_kf(mce_msmt_idxs, s_xs, s_xs_kf, s_Ps_kf, 
                             None, None,
                             #s_xs_mce, s_Ps_mce, 
                             s_xs_mce_avg, s_Ps_mce_avg, 
                             filt_dt, mce_dt, sig=1, title_prefix='Secondary Satellite')

    kf_quant = 0.9999
    mce_quant = 0.9999
    # Plot this out in 3D
    #pc.draw_3d_encounter_plane(
    #    s_xtc, p_xtc, s_Ptc_kf, p_Ptc_kf,
    #    mc_runs_prim=cache_dic['mc_prim_tcas'], mc_runs_sec=cache_dic['mc_sec_tcas'],
    #    s_mce_Ptc=s_Ptc_mce, p_mce_Ptc=p_Ptc_mce)

    # Plot this out in 2D
    pc.draw_2d_projected_encounter_plane(kf_quant,
                                         s_xtc, p_xtc,
                                         s_Ptc_kf, p_Ptc_kf,
                                         s_mce_Ptc=s_Ptc_mce, p_mce_Ptc=p_Ptc_mce, quantile_mce=mce_quant,
                                         mc_prim=cache_dic['mc_prim_tcas'], mc_sec=cache_dic['mc_sec_tcas'],
                                         mc_sec_sample_tcas=cache_dic['mc_sec_sample_tcas'], mc_prim_sample_tcas=cache_dic['mc_prim_sample_tcas'])

    pc.draw_ensemble_encounter_plane(
        kf_quant,
        s_xtc, p_xtc,
        s_Ptc_kf, p_Ptc_kf,
        s_Ptc_mce, p_Ptc_mce, mce_quant,
        cache_dic['mc_prim_tcas'],
        cache_dic['mc_sec_tcas'],
        rsys_info = cache_dic['mce_rsys'])

    # End of Filtration
    # [4][-1] # [4][-1] is average, [2][-1] is 'best'
    p_P_mce_filtend = cache_dic['mce_prim_sim'][2][-1]
    # [4][-1] # [4][-1] is average, [2][-1] is 'best'
    s_P_mce_filtend = cache_dic['mce_sec_sim'][2][-1]
    pc.analyze_3d_statistics(kf_quant, mce_quant,
                             s_xs_kf[-1], p_xs_kf[-1], 
                             s_Ps_kf[-1], p_Ps_kf[-1],
                             s_P_mce_filtend, p_P_mce_filtend,
                             mc_prim_tcas=None, mc_sec_tcas=None,
                             mc_prim_sample_tcas=None, mc_sec_sample_tcas=None,
                             plot_title="at the end of filtering and start of prediction",
                             scale_vel_prim=0.03,
                             scale_vel_sec=0.03)

    # At Time of Closest Approach
    pc.analyze_3d_statistics(kf_quant, mce_quant,
                             s_xtc, p_xtc,
                             s_Ptc_kf, p_Ptc_kf,
                             s_Ptc_mce, p_Ptc_mce,
                             #mc_prim_tcas=cache_dic['mc_prim_tcas'], mc_sec_tcas=cache_dic['mc_sec_tcas'],
                             #mc_sec_sample_tcas=None, mc_prim_sample_tcas=None)
                             mc_prim_tcas = cache_dic['mc_prim_tcas'], mc_sec_tcas = cache_dic['mc_sec_tcas'],
                             mc_sec_sample_tcas = cache_dic['mc_sec_sample_tcas'], mc_prim_sample_tcas = cache_dic['mc_prim_sample_tcas'])
    foobar = 2


if __name__ == "__main__":
    # test_sat_pc()
    # find_cross_radial_error_mc_vs_density()
    # test_sat_crossing()
    test_sat_pc_mc()

# Old Codes
'''
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
    
    #
    #tks = dt * np.arange(steps)
    #i, troot, t_c, pp_c, sp_c = pc.closest_approach_info(tks[start_idx:], 
    #    (p_pks[start_idx:,:],p_vks[start_idx:,:],p_aks[start_idx:,:]), 
    #    (s_pks[start_idx:,:],s_vks[start_idx:,:],s_aks[start_idx:,:]))
    #i += start_idx
    #print("Step dt: ", dt)
    #print("Tc: {}, Idx of Tc: {}".format(t_c, i) )
    #print("Primary at Tc: ", pp_c)
    #print("Secondary at Tc: ", sp_c)
    #print("Pos Diff is: ", pp_c-sp_c)
    #print("Pos Norm is: ", np.linalg.norm(pp_c-sp_c))
    ## Plot orbits of primary and secondary
    #fig = plt.figure() 
    #ax = fig.gca(projection='3d')
    #plt.title("Leo Trajectory over Time")
    #ax.plot(p_pks[:,0], p_pks[:,1], p_pks[:,2], color = 'r')
    #ax.plot(s_pks[:,0], s_pks[:,1], s_pks[:,2], color = 'b')
    # Plot relative difference in position
    #fig2 = plt.figure()
    #plt.title("Pos Norm Diff over Time")
    #plt.plot(tks, p_pks[:,0] - s_pks[:,0], 'r')
    ##plt.plot(tks, p_pks[:,1] - s_pks[:,1], 'g')
    ##plt.plot(tks, p_pks[:,2] - s_pks[:,2], 'b')
    #plt.plot(tks, np.linalg.norm(p_pks-s_pks,axis=1))
    #plt.show()
    #foo=3
    #
    #
    # GMAT iterative closest time of approach
    i_star_lhs, t_lhs, t_c, pp_c, sp_c = pc.iterative_time_closest_approach(
        dt, t0, 
        (p_pks,p_vks,p_aks), 
        (s_pks,s_vks,s_aks), 
        start_idx = start_idx,
        with_plot=False
        )
    #

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

    #
    # Plot Trajectories of both satellites 
    #fig = plt.figure() 
    #ax = fig.gca(projection='3d')
    #plt.title("Leo Trajectory over Time")
    #ax.plot(p_pks[:,0], p_pks[:,1], p_pks[:,2], color = 'r')
    #ax.plot(s_pks[:,0], s_pks[:,1], s_pks[:,2], color = 'b')
    #ax.scatter(p_pks[0,0], p_pks[0,1], p_pks[0,2], color = 'k', s=80)
    #fig2 = plt.figure()
    #r_norms = np.linalg.norm(s_pks - p_pks, axis=1)
    #plt.plot(np.arange(steps), r_norms)
    #plt.show()
    #foobar = 5
    #

    # Now test the closest approach method
    #
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
    #
'''
