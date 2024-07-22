
import test_gmat_real as tgr 
import os
import numpy as np
gmat_data_dir = gmat_data_dir = tgr.file_dir + "/gmat_data/2023-05-18T19.52.25-auto/inputs/" # CHANGE THIS TO DIRECTORY WHERE EOP FILE and SPACE WEATHER FILE LIVE!
if not os.path.isdir(gmat_data_dir):
    print("gmat_data_dir DNE! Check again!")
    exit(1)

def stm_hardcode():
    # Load data from input file paths
    gps_path =  gmat_data_dir + "G_navsol_from_gseqprt_2023122_00_thinned.txt.navsol"
    restart_ekf_path = gmat_data_dir + "Sat_GLAST_Restart_20230502_090241.csv" #"Sat_GLAST_Restart_20230212_094850.csv"
    gps_msmts = tgr.load_gps_from_txt(gps_path)
    inputted_ekf = tgr.load_glast_file(restart_ekf_path)
    day_t0, day_x0, day_P0, labels_x0, labels_P0 = tgr.find_restart_point(restart_ekf_path, gps_msmts[0][0])
    dt = (inputted_ekf[0][1] - inputted_ekf[0][0]).total_seconds()
    Cd0 = day_x0[6]

    fermiSat = tgr.FermiSatelliteModel(day_t0, day_x0[0:6], dt, gmat_print = True)
    fermiSat.create_model(with_jacchia=True, with_SRP=True, Cd0=Cd0, Cr0=0.75)

    STM_GMAT = np.array([1.003034385738636, 0.002031626475282531,-0.000729540253611561,57.05677242340163,0.03930977151368073,-0.01424258269519773,                
                            0.002031764086920329, 0.9988466639018433, -0.0002964118288430859, 0.0393111184375006, 56.97881004832186, -0.005945956158486117,              
                            -0.000729606252658715, -0.0002964185478580219, 0.9981228723229191, -0.01424322983557232, -0.005946022226398968, 56.96446221349366,           
                            0.0001049370279345411, 7.26044306760972e-05, -2.630381755629913e-05, 1.002942441514051, 0.002105658350439001, -0.0007693785505276875,         
                            7.261628213660584e-05, -3.907161459983431e-05, -1.099303617158858e-05, 0.002105795981925901, 0.9989243825042977, -0.00032999736805977,        
                            -2.630949405191991e-05, -1.099361296326399e-05, -6.55903388699953e-05, -0.0007694445428907903, -0.0003300040852407521, 0.9981370913679972]).reshape((6,6))
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:.6f}'.format}, linewidth=160)

    TAYLOR_ORDER = 4
    STM = fermiSat.get_transition_matrix(taylor_order=TAYLOR_ORDER, use_units_km=True)
    _STM = STM.copy()
    diffs = (STM - STM_GMAT)
    perc_diffs = (STM - STM_GMAT) / STM_GMAT * 100
    print("STM_GMAT:\n", STM_GMAT)
    print("STM_PS{}_x0:\n".format(TAYLOR_ORDER), STM)
    print("STM_PS{}_x0 Diffs:\n".format(TAYLOR_ORDER), diffs)
    print("STM_PS{}_x0 Percent Diffs:\n".format(TAYLOR_ORDER), perc_diffs)
    print("STM_PS{}_x0 Max Diff:".format(TAYLOR_ORDER), np.max(np.abs((diffs))) )
    print("STM_PS{}_x0 Max Percent Diffs:".format(TAYLOR_ORDER), np.max(np.abs(perc_diffs)))
    print("\n\n")

    x1 = fermiSat.step()
    STM = fermiSat.get_transition_matrix(taylor_order=TAYLOR_ORDER, use_units_km=True)
    diffs = (STM - STM_GMAT)
    perc_diffs = (STM - STM_GMAT) / STM_GMAT * 100
    print("STM_GMAT:\n", STM_GMAT)
    print("STM_PS{}_x1:\n".format(TAYLOR_ORDER), STM)
    print("STM_PS{}_x1 Diffs:\n".format(TAYLOR_ORDER), diffs)
    print("STM_PS{}_x1 Percent Diffs:\n".format(TAYLOR_ORDER), perc_diffs)
    print("STM_PS{}_x1 Max Diff:".format(TAYLOR_ORDER), np.max(np.abs((diffs))) )
    print("STM_PS{}_x1 Max Percent Diffs:".format(TAYLOR_ORDER), np.max(np.abs(perc_diffs)))
    print("\n\n")

    STM = (_STM + STM) / 2.0
    diffs = (STM - STM_GMAT)
    perc_diffs = (STM - STM_GMAT) / STM_GMAT * 100
    print("STM_GMAT:\n", STM_GMAT)
    print("AVG_STM_PS{}:\n".format(TAYLOR_ORDER), STM)
    print("AVG_STM_PS{} Diffs:\n".format(TAYLOR_ORDER), diffs)
    print("AVG_STM_PS{} Percent Diffs:\n".format(TAYLOR_ORDER), perc_diffs)
    print("AVG_STM_PS{} Max Diff:".format(TAYLOR_ORDER), np.max(np.abs((diffs))) )
    print("AVG_STM_PS{} Max Percent Diffs:".format(TAYLOR_ORDER), np.max(np.abs(perc_diffs)))
    print("\n\n")

    # LEERS METHOD
    fermiSat.reset_state(day_x0[0:6], 0)
    fermiSat.dt = dt
    Jac1 = fermiSat.get_jacobian_matrix()
    fermiSat.step()
    Jac2 = fermiSat.get_jacobian_matrix()
    STM = np.eye(6) + (Jac1 + Jac2)/2 * dt + 0.5 * Jac1 @ Jac2 * dt**2
    diffs = (STM - STM_GMAT)
    perc_diffs = (STM - STM_GMAT) / STM_GMAT * 100
    print("STM_GMAT:\n", STM_GMAT)
    print("STM_LEER:\n", STM)
    print("STM_LEER Diffs:\n", diffs)
    print("STM_LEER Percent Diffs:\n", perc_diffs)
    print("STM_LEER Max Diff:", np.max(np.abs((diffs))) )
    print("STM_LEER Max Percent Diffs:", np.max(np.abs(perc_diffs)))
    print("\n\n")

    # RK4 Testings
    fermiSat.reset_state(day_x0[0:6], 0)
    sub_steps = 3
    dt_sub = dt / sub_steps
    STM = np.eye(6)
    fermiSat.dt = dt_sub
    for i in range(sub_steps):
        Jac = fermiSat.get_jacobian_matrix()
        fPhi = lambda Phi : Jac @ Phi
        STM = tgr.ce.runge_kutta4(fPhi, STM, dt_sub)
        fermiSat.step()
    diffs = (STM - STM_GMAT)
    perc_diffs = (STM - STM_GMAT) / STM_GMAT * 100
    print("STM_GMAT:\n", STM_GMAT)
    print("STM_RK4:\n", STM)
    print("STM_RK4 Diffs:\n", diffs)
    print("STM_RK4 Percent Diffs:\n", perc_diffs)
    print("STM_RK4 Max Diff:", np.max(np.abs((diffs))) )
    print("STM_RK4 Max Percent Diffs:", np.max(np.abs(perc_diffs)))
    foobar = 2

def stm_loop():
    input_path = tgr.file_dir + "/gmat_data/2023-05-18T19.52.25-auto/inputs/"
    output_path = tgr.file_dir + "/gmat_data/2023-05-18T19.52.25-auto/outputs/"
    # Load data from input file paths
    gps_path =  input_path + "G_navsol_from_gseqprt_2023122_00_thinned.txt.navsol"
    ekf_out_path = output_path + "Sat_GLAST_Restart_20230502_090241.csv" #"Sat_GLAST_Restart_20230212_094850.csv"
    gps_msmts = tgr.load_gps_from_txt(gps_path)
    ekf_out = tgr.load_glast_file(ekf_out_path)
    day_t0, day_x0, day_P0, labels_x0, labels_P0 = tgr.find_restart_point(ekf_out_path, gps_msmts[0][0])
    dt = (ekf_out[0][1] - ekf_out[0][0]).total_seconds()
    Cd0 = day_x0[6] if day_x0.size > 6 else 2.1

    # STM ON FIRST GPS MSMT
    STM_GMAT = np.array([1.003034385738636, 0.002031626475282531,-0.000729540253611561,57.05677242340163,0.03930977151368073,-0.01424258269519773,                
                            0.002031764086920329, 0.9988466639018433, -0.0002964118288430859, 0.0393111184375006, 56.97881004832186, -0.005945956158486117,              
                            -0.000729606252658715, -0.0002964185478580219, 0.9981228723229191, -0.01424322983557232, -0.005946022226398968, 56.96446221349366,           
                            0.0001049370279345411, 7.26044306760972e-05, -2.630381755629913e-05, 1.002942441514051, 0.002105658350439001, -0.0007693785505276875,         
                            7.261628213660584e-05, -3.907161459983431e-05, -1.099303617158858e-05, 0.002105795981925901, 0.9989243825042977, -0.00032999736805977,        
                            -2.630949405191991e-05, -1.099361296326399e-05, -6.55903388699953e-05, -0.0007694445428907903, -0.0003300040852407521, 0.9981370913679972]).reshape((6,6))
    
    # STM ON FOURTH GPS MSMT
    #STM_GMAT = np.array([1.001542542806007, -0.001964477247012069, 0.00108563400156658, 51.02692484552533, -0.03315726568645618, 0.01837433210181619,                  
    #                       -0.001964399691677653, 0.9996623412891866, -0.0006843060394828077, -0.03315658972210785, 50.99368391715793, -0.01134778024751283,            
    #                       0.001085596831226614, -0.0006843096080460666, 0.9987976056255934, 0.0183740077310293, -0.01134781119382756, 50.97941623944995,                
    #                       6.217243909237916e-05, -7.650208632160013e-05, 4.239482555480801e-05, 1.00162576105938, -0.001935988215109958, 0.001075880176660856,          
    #                       -7.649460122810276e-05, -1.452360869372734e-05, -2.618867042366496e-05, -0.001935910680841804, 0.9995954936931657, -0.0006509720020160885,    
    #                       4.239124075424695e-05, -2.618901595667391e-05, -4.745364272950013e-05, 0.001075843014701189, -0.0006509755700655249, 0.9987812317143444]).reshape((6,6))
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:.6f}'.format}, linewidth=160)

    sub_steps = 4
    dt_sub = dt / sub_steps
    TAYLOR_ORDER = 4
    fermiSat = tgr.FermiSatelliteModel(day_t0, day_x0[0:6], dt_sub, gmat_print = False)
    fermiSat.create_model(with_jacchia=True, with_SRP=True, Cd0=Cd0, Cr0=0.75)
    #fermiSat.dt = dt_sub 
    #fermiSat.reset_state(day_x0[0:6], 0)

    # Begin STM Loop
    STM_PS = np.eye(6) # 1.) STM by using Power Series Expansion about time step i
    STM_AVG_STM = np.eye(6) # 2.) STM by averaging STM (as explained above) of time step i and i+1
    STM_AVG_JAC = np.eye(6) # 3.) STM by averaging Jacobian of time step i and i+1 and using power series expansion (as explained above) 
    STM_RK4 = np.eye(6) # 4.) STM by integrating \dot{STM} = Jacobian @ \STM from time step i to i+1 using Runge Kutta 4
    STM_AVG_RK4 = np.eye(6) # 5.) STM by integrating \dot{STM} = Jacobian @ \STM from time step i to i+1 using Runge Kutta 4 and the averaged jacobian
    STM_LEER = np.eye(6) # 6.) Leers Method -- essentially an RK2
    for i in range(sub_steps):
        # Get Jacobians and STMs over time step DT_SUB
        Jac_i = fermiSat.get_jacobian_matrix()
        STM_i = fermiSat.get_transition_matrix(taylor_order=TAYLOR_ORDER, use_units_km=True)
        fermiSat.step()
        Jac_ip1 = fermiSat.get_jacobian_matrix()
        STM_ip1 = fermiSat.get_transition_matrix(taylor_order=TAYLOR_ORDER, use_units_km=True)
        Jac_avg = (Jac_i+Jac_ip1)/2
        
        # UPDATE STATE TRANSITION MATRICES
        STM_PS = STM_i @ STM_PS
        STM_AVG_STM = (STM_ip1 + STM_i) / 2 @ STM_AVG_STM
        STM_AVG_JAC = tgr.gmce.get_transition_matrix( Jac_avg, dt_sub, TAYLOR_ORDER) @ STM_AVG_JAC
        # for RK4 integration
        fPhi = lambda Phi : Jac_i @ Phi
        STM_RK4 = tgr.ce.runge_kutta4(fPhi, STM_RK4, dt_sub)
        # for averaged RK4 integration
        fPhi_avg = lambda Phi : Jac_avg @ Phi
        STM_AVG_RK4 = tgr.ce.runge_kutta4(fPhi_avg, STM_AVG_RK4, dt_sub)
        # Leer 
        STM_LEER = (np.eye(6) + Jac_avg * dt_sub + 0.5 * Jac_i @ Jac_ip1 * dt_sub**2) @ STM_LEER
    
    print("\nRun Settings: DT={}, SUB_STEPS={}, DT_SUB={}, TAYLOR_ORDER={} (For Power Series Based Methods)".format(dt, sub_steps, dt_sub, TAYLOR_ORDER))
    # 1.)
    diffs = (STM_PS - STM_GMAT)
    perc_diffs = diffs / STM_GMAT * 100
    print("1.) STM_PS       Max Diff:", np.max(np.abs((diffs))) )
    print("    STM_PS       Max Percent Diffs:", np.max(np.abs(perc_diffs)))
    # 2.)
    diffs = (STM_AVG_STM - STM_GMAT)
    perc_diffs = diffs / STM_GMAT * 100
    print("\n2.) STM_AVG_STM  Max Diff:", np.max(np.abs((diffs))) )
    print("    STM_AVG_STM  Max Percent Diffs:", np.max(np.abs(perc_diffs)))
    # 3.)
    diffs = (STM_AVG_JAC - STM_GMAT)
    perc_diffs = diffs / STM_GMAT * 100
    print("\n3.) STM_AVG_JAC  Max Diff:", np.max(np.abs((diffs))) )
    print("    STM_AVG_JAC  Max Percent Diffs:", np.max(np.abs(perc_diffs)))
    # 4.)
    diffs = (STM_RK4 - STM_GMAT)
    perc_diffs = diffs / STM_GMAT * 100
    print("\n4.) STM_RK4      Max Diff:", np.max(np.abs((diffs))) )
    print("    STM_RK4      Max Percent Diffs:", np.max(np.abs(perc_diffs)))
    # 5.)
    diffs = (STM_AVG_RK4 - STM_GMAT)
    perc_diffs = diffs / STM_GMAT * 100
    print("\n5.) STM_AVG_RK4  Max Diff:", np.max(np.abs((diffs))) )
    print("    STM_AVG_RK4  Max Percent Diffs:", np.max(np.abs(perc_diffs)))
    # 6.)
    diffs = (STM_LEER - STM_GMAT)
    perc_diffs = diffs / STM_GMAT * 100
    print("\n6.) STM_LEER     Max Diff:", np.max(np.abs((diffs))) )
    print("    STM_LEER     Max Percent Diffs:", np.max(np.abs(perc_diffs)))
    print("\nThats all, folks!")

def stm_load_and_loop():
    input_path = tgr.file_dir + "/gmat_data/2023-05-18T19.52.25-auto/inputs/"
    output_path = tgr.file_dir + "/gmat_data/2023-05-18T19.52.25-auto/outputs/"
    # Load data from input file paths
    gps_path =  input_path + "G_navsol_from_gseqprt_2023122_00_thinned.txt.navsol"
    ekf_out_path = output_path + "Sat_GLAST_Restart_20230502_090241.csv" #"Sat_GLAST_Restart_20230212_094850.csv"
    stm_path = output_path + "STM_Log.txt"
    gps_msmts = tgr.load_gps_from_txt(gps_path)
    ekf_out = tgr.load_glast_file(ekf_out_path)
    stm_ts, stm_dts, stm_xs, STMs = tgr.load_STMs(stm_path, ekf_out)
    for i in range(len(stm_ts)-1):
        IDX_GPS = i # Pick a gps msmt idx and it will load out the STM which was used to get to this GPS MSMT after TP
        gps_t0 = gps_msmts[IDX_GPS][0]
        IDX_EKF = tgr.find_time_match(gps_t0, ekf_out[0], start_idx = 0)-1 # we want the state before the gps time stamp in creating the STM for this measurement (time prop to MU)
        IDX_STM = tgr.find_time_match(gps_t0, stm_ts, start_idx = 0)-1

        t0 = ekf_out[0][IDX_EKF]
        x0 = ekf_out[1][IDX_EKF]
        dt = stm_dts[IDX_STM]
        Cd0 = x0[6] if x0.size > 6 else 2.1

        STM_GMAT = STMs[IDX_STM]
        # STM ON FIRST GPS MSMT
        #STM_GMAT = np.array([1.003034385738636, 0.002031626475282531,-0.000729540253611561,57.05677242340163,0.03930977151368073,-0.01424258269519773,                
        #                        0.002031764086920329, 0.9988466639018433, -0.0002964118288430859, 0.0393111184375006, 56.97881004832186, -0.005945956158486117,              
        #                        -0.000729606252658715, -0.0002964185478580219, 0.9981228723229191, -0.01424322983557232, -0.005946022226398968, 56.96446221349366,           
        #                        0.0001049370279345411, 7.26044306760972e-05, -2.630381755629913e-05, 1.002942441514051, 0.002105658350439001, -0.0007693785505276875,         
        #                        7.261628213660584e-05, -3.907161459983431e-05, -1.099303617158858e-05, 0.002105795981925901, 0.9989243825042977, -0.00032999736805977,        
        #                        -2.630949405191991e-05, -1.099361296326399e-05, -6.55903388699953e-05, -0.0007694445428907903, -0.0003300040852407521, 0.9981370913679972]).reshape((6,6))
        
        # STM ON FOURTH GPS MSMT
        #STM_GMAT = np.array([1.001542542806007, -0.001964477247012069, 0.00108563400156658, 51.02692484552533, -0.03315726568645618, 0.01837433210181619,                  
        #                       -0.001964399691677653, 0.9996623412891866, -0.0006843060394828077, -0.03315658972210785, 50.99368391715793, -0.01134778024751283,            
        #                       0.001085596831226614, -0.0006843096080460666, 0.9987976056255934, 0.0183740077310293, -0.01134781119382756, 50.97941623944995,                
        #                       6.217243909237916e-05, -7.650208632160013e-05, 4.239482555480801e-05, 1.00162576105938, -0.001935988215109958, 0.001075880176660856,          
        #                       -7.649460122810276e-05, -1.452360869372734e-05, -2.618867042366496e-05, -0.001935910680841804, 0.9995954936931657, -0.0006509720020160885,    
        #                       4.239124075424695e-05, -2.618901595667391e-05, -4.745364272950013e-05, 0.001075843014701189, -0.0006509755700655249, 0.9987812317143444]).reshape((6,6))
        np.set_printoptions(suppress=True, formatter={'float_kind':'{:.6f}'.format}, linewidth=160)

        dt_nom_step = 5
        sub_steps = int(dt + dt_nom_step - 1) // int(dt_nom_step)
        #sub_steps = 12
        dt_sub = dt / sub_steps
        TAYLOR_ORDER = 4
        print("TIME {} FOR GPS IDX {}, STM IDX {}, EKF_IDX {}, DT={}, DT_SUB={}, NUM_STEPS={}".format(gps_t0, IDX_GPS, IDX_STM, IDX_EKF, dt, dt_sub, sub_steps))

        fermiSat = tgr.FermiSatelliteModel(t0, x0[0:6], dt_sub, gmat_print = False)
        fermiSat.create_model(with_jacchia=True, with_SRP=True, Cd0=Cd0, Cr0=0.75)
        #fermiSat.dt = dt_sub 
        #fermiSat.reset_state(day_x0[0:6], 0)

        # Begin STM Loop
        STM_PS = np.eye(6) # 1.) STM by using Power Series Expansion about time step i
        STM_AVG_STM = np.eye(6) # 2.) STM by averaging STM (as explained above) of time step i and i+1
        STM_AVG_JAC = np.eye(6) # 3.) STM by averaging Jacobian of time step i and i+1 and using power series expansion (as explained above) 
        STM_RK4 = np.eye(6) # 4.) STM by integrating \dot{STM} = Jacobian @ \STM from time step i to i+1 using Runge Kutta 4
        STM_AVG_RK4 = np.eye(6) # 5.) STM by integrating \dot{STM} = Jacobian @ \STM from time step i to i+1 using Runge Kutta 4 and the averaged jacobian
        STM_LEER = np.eye(6) # 6.) Leers Method -- essentially an RK2
        for i in range(sub_steps):
            # Get Jacobians and STMs over time step DT_SUB
            Jac_i = fermiSat.get_jacobian_matrix()
            STM_i = fermiSat.get_transition_matrix(taylor_order=TAYLOR_ORDER, use_units_km=True)
            fermiSat.step()
            Jac_ip1 = fermiSat.get_jacobian_matrix()
            STM_ip1 = fermiSat.get_transition_matrix(taylor_order=TAYLOR_ORDER, use_units_km=True)
            Jac_avg = (Jac_i+Jac_ip1)/2
            
            # UPDATE STATE TRANSITION MATRICES
            STM_PS = STM_i @ STM_PS
            STM_AVG_STM = (STM_ip1 + STM_i) / 2 @ STM_AVG_STM
            STM_AVG_JAC = tgr.gmce.get_transition_matrix( Jac_avg, dt_sub, TAYLOR_ORDER) @ STM_AVG_JAC
            # for RK4 integration
            fPhi = lambda Phi : Jac_i @ Phi
            STM_RK4 = tgr.ce.runge_kutta4(fPhi, STM_RK4, dt_sub)
            # for averaged RK4 integration
            fPhi_avg = lambda Phi : Jac_avg @ Phi
            STM_AVG_RK4 = tgr.ce.runge_kutta4(fPhi_avg, STM_AVG_RK4, dt_sub)
            # Leer 
            STM_LEER = (np.eye(6) + Jac_avg * dt_sub + 0.5 * Jac_i @ Jac_ip1 * dt_sub**2) @ STM_LEER
        
        print("\nRun Settings: DT={}, SUB_STEPS={}, DT_SUB={}, TAYLOR_ORDER={} (For Power Series Based Methods)".format(dt, sub_steps, dt_sub, TAYLOR_ORDER))
        # 1.)
        diffs = (STM_PS - STM_GMAT)
        perc_diffs = diffs / STM_GMAT * 100
        print("1.) STM_PS       Max Diff:", np.max(np.abs((diffs))) )
        print("    STM_PS       Max Percent Diffs:", np.max(np.abs(perc_diffs)))
        # 2.)
        diffs = (STM_AVG_STM - STM_GMAT)
        perc_diffs = diffs / STM_GMAT * 100
        print("\n2.) STM_AVG_STM  Max Diff:", np.max(np.abs((diffs))) )
        print("    STM_AVG_STM  Max Percent Diffs:", np.max(np.abs(perc_diffs)))
        # 3.)
        diffs = (STM_AVG_JAC - STM_GMAT)
        perc_diffs = diffs / STM_GMAT * 100
        print("\n3.) STM_AVG_JAC  Max Diff:", np.max(np.abs((diffs))) )
        print("    STM_AVG_JAC  Max Percent Diffs:", np.max(np.abs(perc_diffs)))
        # 4.)
        diffs = (STM_RK4 - STM_GMAT)
        perc_diffs = diffs / STM_GMAT * 100
        print("\n4.) STM_RK4      Max Diff:", np.max(np.abs((diffs))) )
        print("    STM_RK4      Max Percent Diffs:", np.max(np.abs(perc_diffs)))
        # 5.)
        diffs = (STM_AVG_RK4 - STM_GMAT)
        perc_diffs = diffs / STM_GMAT * 100
        print("\n5.) STM_AVG_RK4  Max Diff:", np.max(np.abs((diffs))) )
        print("    STM_AVG_RK4  Max Percent Diffs:", np.max(np.abs(perc_diffs)))
        # 6.)
        diffs = (STM_LEER - STM_GMAT)
        perc_diffs = diffs / STM_GMAT * 100
        print("\n6.) STM_LEER     Max Diff:", np.max(np.abs((diffs))) )
        print("    STM_LEER     Max Percent Diffs:", np.max(np.abs(perc_diffs)))
        fermiSat.clear_model()
    print("\nThats all, folks!")


if __name__ == '__main__':
    #stm_hardcode()
    #stm_loop()
    stm_load_and_loop()