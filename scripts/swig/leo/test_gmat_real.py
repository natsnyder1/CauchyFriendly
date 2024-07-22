import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg',force=True)
import os, pickle
from datetime import datetime 
from gmat_sat import *
import gmat_mce as gmce
from scipy.stats import chi2
import cauchy_estimator as ce
import numba as nb
import time 
import pprint 

KM_TO_M = 1000.0
M_TO_KM = 1.0 / KM_TO_M

# Time conversion function using GMAT
def time_convert(time_in, type_in, type_out):
    if type(time_in) == datetime:
        millisec = str(np.round(time_in.microsecond / 1e6, 3)).split(".")[1]
        _time_in = time_in.strftime("%d %b %Y %H:%M:%S") + "." + millisec
        is_in_gregorian = True
    elif type(time_in) == str:
        _time_in = time_in
        is_in_gregorian = True
    elif type(time_in) == float:
        _time_in = time_in
        is_in_gregorian = False
    else:
        print("Time In Type: ", type(time_in), " Not Supported! Input was", time_in)
        exit(1)
    timecvt = gmat.TimeSystemConverter.Instance()
    if is_in_gregorian:
        time_in_greg = _time_in
        time_in_mjd = timecvt.ConvertGregorianToMjd(_time_in)
    else:
        time_in_mjd = _time_in
        time_in_greg = timecvt.ConvertMjdToGregorian(_time_in)
    time_types = {"A1": timecvt.A1, "TAI": timecvt.TAI, "UTC" : timecvt.UTC, "TDB": timecvt.TDB, "TT": timecvt.TT}
    assert type_in in time_types.keys()
    assert type_out in time_types.keys()
    time_code_in = time_types[type_in]
    time_code_out = time_types[type_out]
    time_out_mjt = timecvt.Convert(time_in_mjd, time_code_in, time_code_out)
    time_out_greg = timecvt.ConvertMjdToGregorian(time_out_mjt)
    time_dic = {"in_greg" : time_in_greg, 
                "in_mjd" : time_in_mjd, 
                "out_greg": time_out_greg, 
                "out_mjd": time_out_mjt}
    return time_dic

# Load in GPS Data and time stamps 
def load_gps_from_txt(fpath):
    # See if cached pickle file already exists 
    fprefix, fname = fpath.rsplit("/", 1)
    name_substrs = fname.split(".")
    name_prefix = name_substrs[0]
    name_suffix = name_substrs[2]
    assert(name_suffix in ["navsol", "gmd"])
    with_UTC_format = name_suffix == "navsol" # otherwise gmd
    if with_UTC_format:
        pickle_fpath = fprefix + "/" + name_prefix + "_navsol" + ".pickle"
    else:
        pickle_fpath = fprefix + "/" + name_prefix + "_gmd" + ".pickle"
    if os.path.isfile(pickle_fpath):
        print("Reading Cached GPS Data From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "rb") as handle:
            gps_msmts = pickle.load(handle)
        return gps_msmts
    # Read in gps and pickle (so it doesnt have to be done again)
    else:
        gps_msmts = []
        with open(fpath, 'r') as handle:
            lines = handle.readlines()
            for line in lines:
                cols = line.split()
                if with_UTC_format:
                    year = int(cols[1])
                    month = int(cols[2])
                    day = int(cols[3])
                    hour = int(cols[4])
                    minute = int(cols[5])
                    second = int(cols[6])//1000
                    microsecond = (int(cols[6]) % 1000) * 1000
                    date_time = datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
                    pos_x = float(cols[9])
                    pos_y = float(cols[10])
                    pos_z = float(cols[11])
                    pos = np.array([pos_x, pos_y, pos_z])
                    gps_msmts.append((date_time, pos))
                else:
                    print("Enter Formatting for TAI times")
                    exit(1)

        print("Writing Cached GPS Data To Pickle File at: ", pickle_fpath)
        with open(pickle_fpath, "wb") as handle:
            pickle.dump(gps_msmts, handle)
        return gps_msmts 

# Load in Stored GMAT EKF Run
def load_glast_file(fpath):
    fprefix, fname = fpath.rsplit("/", 1)
    name_prefix = fname.split(".")[0]
    pickle_fpath = fprefix + "/" + name_prefix + ".pickle"
    if os.path.isfile(pickle_fpath):
        print("Reading Cached GLast Data From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "rb") as handle:
            times, means, covars = pickle.load(handle)
        return times, means, covars
    else:
        with open(fpath, 'r') as handle:
            lines = handle.readlines()

            # Find state size
            header = lines[0]
            header_cols = header.split(",")
            len_line = len(header_cols)
            c = -2*(len_line-1)
            n = -1.5 + (9 - 4*c)**0.5 / 2
            n = int(n + 0.99)
            # Find covariance indices
            idxs = []
            for i in range(n+1, len_line):
                label = header_cols[i]
                str_cov = label.split("_")
                idxs.append( (int(str_cov[1])-1, int(str_cov[2])-1) ) 

            times = [] 
            means = []
            covars = []

            for line in lines[1:]:
                # Time Creation
                sub_strs = line.split(",")
                str_date = sub_strs[0]
                date_list = str_date.split()
                day = int(date_list[0])
                month = MonthDic[date_list[1]]
                year = int(date_list[2])
                time_list = date_list[3].split(":")
                hour = int(time_list[0])
                minute = int(time_list[1])
                str_second, str_millisec = time_list[2].split(".")
                second = int(str_second)
                microsecond = int(str_millisec) * 1000
                date_time = datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
                times.append(date_time)
                # Append Means 
                xk = np.zeros(n)
                for i in range(1,n+1):
                    xk[i-1] = float(sub_strs[i])
                means.append(xk)
                cholPk = np.zeros((n,n))
                for k, idx in enumerate(idxs):
                    i = idx[0]
                    j = idx[1]
                    cPij = float(sub_strs[k+n+1])
                    cholPk[i,j] = cPij
                Pk = cholPk @ cholPk.T
                covars.append(Pk)
        print("Writing Cached GLast Data From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "wb") as handle:
            pickle.dump((times, means, covars), handle)
        return times, means, covars

# Load in Stored GMAT Smoother Ephem
def load_ephem_file(fpath, glast_info = None):
    fprefix, fname = fpath.rsplit("/", 1)
    name_prefix = fname.split(".")[0]
    pickle_fpath = fprefix + "/" + name_prefix + ".pickle"
    if os.path.isfile(pickle_fpath):
        print("Reading Cached GLast Data From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "rb") as handle:
            times, PosVels, PosVelCovars = pickle.load(handle)
    else:
        times = []
        PosVels = [] 
        PosVelCovars = [] 
        if glast_info is not None:
            glast_ts, glast_xs, glast_Ps = glast_info
            times = glast_ts
        with open(fpath, 'r') as handle:
            lines = handle.readlines()
            if glast_info is None:
                # Initial Time Creation
                line = lines[6]
                date_list = line.split()[1:]
                day = int(date_list[0])
                month = MonthDic[date_list[1]]
                year = int(date_list[2])
                time_list = date_list[3].split(":")
                hour = int(time_list[0])
                minute = int(time_list[1])
                str_second, str_millisec = time_list[2].split(".")
                second = int(str_second)
                microsecond = int(str_millisec) * 1000
                t0 = datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
            
            # Get Means and Times  
            start_state_est_idx = 13 
            count = start_state_est_idx
            while lines[count] != "\n":
                time_posvel = lines[count].split()
                assert( len(time_posvel) == 7 )
                if glast_info is None:
                    # Append Times
                    ellapsed_time_from_t0 = float(time_posvel[0])
                    tdelta = timedelta( seconds=ellapsed_time_from_t0 )
                    if tdelta.microseconds > 999000:
                        tdelta = timedelta( seconds= np.ceil(ellapsed_time_from_t0 ) )
                    tk = t0 + tdelta
                    times.append(tk)
                # Append Means 
                xk = np.zeros(6)
                for i in range(1, 6+1):
                    xk[i-1] = float(time_posvel[i])
                PosVels.append(xk)
                count += 1
            print("Read in {} Estimates from Ephemeris file!".format(count))
            assert(len(times) == len(PosVels))
            # Get Covariances and Times 
            count += 3
            while lines[count] != "\n":
                cov_pt1 = lines[count].split()
                cov_pt2 = lines[count+1].split()
                cov_pt3 = lines[count+2].split()
                cov_str = cov_pt1[1:] + cov_pt2 + cov_pt3
                assert( len(cov_str) == 21 )
                lt_covar = [float(cs) for cs in cov_str]
                Covar = np.zeros((6,6))
                i = 0
                j = 0
                for k in range(21):
                    Covar[i,j] = lt_covar[k]
                    if i == j:
                        i += 1
                        j = -1
                    else:
                        Covar[j,i] = lt_covar[k]
                    j+=1
                count +=3
                PosVelCovars.append(Covar)
        print("Writing Cached GLast Data From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "wb") as handle:
            pickle.dump((times, PosVels, PosVelCovars), handle)
    return times, PosVels, PosVelCovars 

def load_STMs(stm_path, ekf_times_means):

    sp_root, _ = stm_path.rsplit(".", 1)#split(".") 
    cache_path = sp_root + ".pickle"
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as handle:
            print("Loading Cached STMS from ", cache_path)
            ts, dts, xs, STMs = pickle.load(handle)
            return ts, dts, xs, STMs

    with open(stm_path, 'r') as handle:
        lines = handle.readlines()
        t0_sep = lines[1].split()[0:4]
        t_last = t0_sep[0] + ' ' + t0_sep[1] + ' ' + t0_sep[2] + ' ' + t0_sep[3]
        t_last = time_string_2_datetime(t_last)
        x_tlast_idx = find_time_match(t_last, ekf_times_means[0], start_idx = 0)
        x_tlast = ekf_times_means[1][x_tlast_idx]
        blank_new_entry = ['1','0','0','0','0','0']
        # Get state size 
        n = 1
        while lines[n+1][0] == ' ':
            n += 1
        count = 1+n
        num_lines = len(lines)
        
        # In the STM Log File, the STM for x_k+1 = \Phi_k x_k + w_k is associated with the time t_k+1, which corresponds to the time of the measurement z_k+1
        # Here, however we associate the time t_k with \Phi_k and x_k
        ts = [t_last]
        xs = [x_tlast]
        STMs = []
        dts = []
        while count < num_lines:
            line = lines[count]
            assert(line[0] != ' ')
            line_split = line.split()
            tk_sep = line_split[0:4]
            is_new_entry = line_split[4:] != blank_new_entry
            if is_new_entry:
                stm_row_sep = line_split[4:]
                t_new = tk_sep[0] + ' ' + tk_sep[1] + ' ' + tk_sep[2] + ' ' + tk_sep[3]
                t_new = time_string_2_datetime(t_new)
                STM_k = np.zeros((n,n))
                for i in range(n):
                    STM_k[0, i] = float(stm_row_sep[i])
                for i in range(1, n):
                    line = lines[count + i].split()
                    for j in range(n):
                        STM_k[i, j] = float(line[j])
                # Find the EKF state associated with t_last
                x_tlast_idx = find_time_match(t_new, ekf_times_means[0], start_idx = x_tlast_idx)
                x_tnew = ekf_times_means[1][x_tlast_idx]
                # Save and continue
                ts.append(t_new)
                xs.append(x_tnew)
                STMs.append(STM_k)
                dts.append((t_new-t_last).total_seconds())
                t_last = t_new
            else:
                t_prop = tk_sep[0] + ' ' + tk_sep[1] + ' ' + tk_sep[2] + ' ' + tk_sep[3]
                t_prop = time_string_2_datetime(t_prop)
                x_tlast_idx = find_time_match(t_prop, ekf_times_means[0], start_idx = x_tlast_idx)
                x_tprop = ekf_times_means[1][x_tlast_idx]
                dts.append((t_prop-t_last).total_seconds())
                ts.append(t_prop)
                xs.append(x_tprop)
                STMs.append(None)
                t_last = t_prop
            count += n
    handle.close()
    STMs.append(None)
    dts.append(None)

    # Cache if you get here
    with open(cache_path, 'wb') as handle:
        pickle.dump((ts, dts, xs, STMs),handle)
    return ts, dts, xs, STMs
                    
# If provided, scan GLAST csv to find the a-priori state/covariance closest to the first GPS reading (returns state before first GPS reading)
def find_restart_point(fpath, gps_datetime):
    fprefix, fname = fpath.rsplit("/", 1)
    name_prefix = fname.split(".")[0]
    gps_timetag = "_gps_{}_{}_{}_{}_{}_{}_{}".format(gps_datetime.year,gps_datetime.month,gps_datetime.day,gps_datetime.hour,gps_datetime.minute,gps_datetime.second,gps_datetime.microsecond)
    pickle_fpath = fprefix + "/" + name_prefix + gps_timetag + ".pickle"
    if os.path.isfile(pickle_fpath):
        print("Reading Cached Restart Point From Pickled File at: ", pickle_fpath)
        with open(pickle_fpath, "rb") as handle:
            date_time, x0, P0, state_labels, cov_labels = pickle.load(handle)
        return date_time, x0, P0, state_labels, cov_labels
    else:
        with open(fpath, 'r') as handle:
            lines = handle.readlines()
            count = 0
            for line in lines:
                if count == 0:
                    count +=1 
                    continue
                str_date = line.split(",")[0]
                dt_list = str_date.split()
                day = int(dt_list[0])
                month = MonthDic[dt_list[1]]
                year = int(dt_list[2])
                time_list = dt_list[3].split(":")
                hour = int(time_list[0])
                minute = int(time_list[1])
                str_second, str_millisec = time_list[2].split(".")
                second = int(str_second)
                microsecond = int(str_millisec) * 1000
                date_time = datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
                time_delt = gps_datetime - date_time
                if time_delt.days < 0:
                    break
                count += 1
            # Take the warm-start right before count's current index
            assert(count > 2) # count would need to be larger than 2
            count -= 2
            count = int( np.clip(count, 1, np.inf) )
            final_choice = lines[count]
            str_date = final_choice.split(",")[0]
            dt_list = str_date.split()
            day = int(dt_list[0])
            month = MonthDic[dt_list[1]]
            year = int(dt_list[2])
            time_list = dt_list[3].split(":")
            hour = int(time_list[0])
            minute = int(time_list[1])
            str_second, str_millisec = time_list[2].split(".")
            second = int(str_second)
            microsecond = int(str_millisec) * 1000
            date_time = datetime(year, month, day, hour=hour, minute=minute, second=second, microsecond=microsecond)
            # Find state size
            len_line = len(final_choice.split(","))
            c = -2*(len_line-1)
            n = -1.5 + (9 - 4*c)**0.5 / 2
            n = int(n + 0.99)
            substrs = final_choice.split(",")
            # Read in the state and cholesky of the covariance matrix
            x0 = np.zeros(n)
            cholP0 = np.zeros((n,n))
            labels = lines[0].split(",")
            for i in range(1, n+1):
                x0[i-1] = float( substrs[i] )
            idxs = []
            for i in range(n+1, len_line):
                label = labels[i]
                str_cov = label.split("_")
                idxs.append( (int(str_cov[1])-1, int(str_cov[2])-1) ) 
            for k, idx in enumerate(idxs):
                i = idx[0]
                j = idx[1]
                cPij = float(substrs[k+n+1])
                cholP0[i,j] = cPij
            # Recreate Covariance matrix from its cholesky 
            P0 = cholP0 @ cholP0.T
    state_labels = labels[1:n+1]
    cov_labels = labels[n+1:] 
    # Log this start state to a pickled file
    # Enter code
    print("Writing Cached Restart Point From Pickled File at: ", pickle_fpath)
    with open(pickle_fpath, "wb") as handle:
            pickle.dump((date_time, x0, P0, state_labels, cov_labels), handle)
    return date_time, x0, P0, state_labels, cov_labels

# finds a time match between a query time and the glast times
def find_time_match(t_query, ts_list, start_idx = 0):
    count = 0
    for t_list in ts_list[start_idx:]:
        dt = t_query - t_list
        #if (dt.days == 0) and (dt.seconds == 0):
        if dt.total_seconds() == 0:
            return count + start_idx
        count +=1
    print("No Time Found to Be Same!")
    assert(False)

def find_approximate_time_match(t_query, ts_list, start_idx = 0):
    count = 0
    min_diff = 1e100
    min_diff_idx = -1
    for t_list in ts_list[start_idx:]:
        dt = abs( (t_query - t_list).total_seconds() )
        if dt < min_diff:
            min_diff = dt
            min_diff_idx = count + start_idx
        count += 1
    return min_diff_idx

# If GLAST csv not provided, we may need to run a small nonlinear least squares to find a passible initial state hypothesis            
def estimate_restart_stats(gps_msmts):
    exit(1)

# Utility functions
def cov_km_to_m(_P):
    P = _P.copy()
    if P.shape[0] == 7:
        P[0:6, 0:6] *= 1000**2
        P[0:6, 6] *= 1000
        P[6, 0:6] *= 1000
    elif P.shape[0] == 6:
        P *= 1000**2
    return P 

def cov_m_to_km(_P):
    P = _P.copy()
    if P.shape[0] == 7:
        P[0:6, 0:6] /= 1000**2
        P[0:6, 6] /= 1000
        P[6, 0:6] /= 1000
    elif P.shape[0] == 6:
        P /= 1000**2
    return P 

# Run EKF (as a method to compare as to how GMAT does it)
def run_fermi_ekf(gps_msmts, t0, _x0, _P0, run_dic,
    gmat_ekf_data = None, gmat_smoother_data=None):
    # Take the first seven states if we have an 8 state estimator
    if _x0.size > 7:
        x0 = _x0[0:7].copy()
        P0 = np.delete(_P0, 7, axis=0)
        P0 = np.delete(P0, 7, axis=1)
    else:
        x0 = _x0.copy() 
        P0 = _P0.copy()
    Cd_nominal = x0[6]
    x0[6] = 0
    _T = np.eye(7)
    _T[6,6] /= Cd_nominal
    P0 = _T @ P0 @ _T.T 
    
    fermi_x0 = x0.copy()
    fermi_P0 = P0.copy()

    # Some Fermi OD Constants and initial variables
    fermi_Cd_tau = run_dic['kf_Cd_half_life'] / np.log(2)
    fermi_PSD_Cd = 2 / fermi_Cd_tau * run_dic['kf_Cd_steady_state_sigma']**2
    fermi_dt = run_dic['kf_dt']
    fermi_qs = run_dic['kf_qs']
    fermi_gps_std_dev = run_dic['kf_gps_std_dev']
    EKF_taylor_order = run_dic['kf_taylor_order']
    msmt_sigma_edit_percentile = run_dic['kf_msmt_sigma_edit_percentile']

    fermiSat = FermiSatelliteModel(t0, fermi_x0[0:6], fermi_dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True, Cd0 = Cd_nominal)
    fermiSat.set_solve_for(field="Cd", dist="sas", scale=fermi_PSD_Cd, tau = fermi_Cd_tau, alpha=1.3)
    fermiSat.reset_state(x0.copy(), 0)
    
    # Run Kalman Filter Forwards 
    I7 = np.eye(7)
    H = np.zeros((3,7))
    H[0:3,0:3] = np.eye(3)
    V = np.eye(3) * fermi_gps_std_dev**2

    Wk = np.zeros((7,7))
    Wk[:6,:6] = gmce.leo6_process_noise_model3(fermi_dt, fermi_qs**2)
    Wk[6,6] = (fermi_Cd_tau * fermi_PSD_Cd) / 2 * (1 - np.exp(-2*fermi_dt/fermi_Cd_tau))
    x_kf = fermi_x0.copy() 
    P_kf = fermi_P0.copy() 
    xs_kf = [fermi_x0.copy()]
    xbars_kf = [fermi_x0.copy()]
    Ps_kf = [fermi_P0.copy()]
    msmt_sigma_edit_value = chi2.ppf(msmt_sigma_edit_percentile, 3)
    t_idx_gmat_ekf = 0
    t_idx_gmat_sm = 0

    ts = [t0] 
    if gmat_ekf_data is not None:
        t_idx_gmat_ekf = find_time_match(t0, gmat_ekf_data[0], t_idx_gmat_ekf)
        xs_kf_gmat = [ gmat_ekf_data[1][t_idx_gmat_ekf] ]
        Ps_kf_gmat = [ gmat_ekf_data[2][t_idx_gmat_ekf] ]
    else:
        xs_kf_gmat = [  ]
        Ps_kf_gmat = [  ]
    if gmat_smoother_data is not None:
        t_idx_gmat_sm = find_time_match(t0, gmat_smoother_data[0], t_idx_gmat_sm)
        xs_sm_gmat = [ gmat_smoother_data[1][t_idx_gmat_sm] ]
        Ps_sm_gmat = [ gmat_smoother_data[2][t_idx_gmat_sm] ]
    else:
        xs_sm_gmat = [  ]
        Ps_sm_gmat = [  ]

    tkm1 = t0
    count = 0
    step_count = 0.0
    N = len(gps_msmts)
    for gps_msmt in gps_msmts:
        count += 1
        tk, _zk = gps_msmt
        zk = _zk.copy() / 1000 # m -> km 
        t_delta = tk - tkm1 
        dt_step = t_delta.total_seconds() #+ (t_delta.microseconds / 1e6)
        # Conduct prop_steps of propagation to next time step (given by GPS)
        prop_steps = int( dt_step / fermi_dt )
        for i in range(prop_steps):
            Phi_k = fermiSat.get_transition_matrix(taylor_order=EKF_taylor_order)
            P_kf = Phi_k @ P_kf @ Phi_k.T + Wk
            x_kf = fermiSat.step()
            step_count += 1
        # Conduct sub-propagation steps to next time step (given by GPS)
        sub_dt = dt_step % fermi_dt
        if sub_dt != 0:
            Wk_sub = np.zeros((7,7))
            Wk_sub[0:6,0:6] = gmce.leo6_process_noise_model3(sub_dt, qs=fermi_qs**2)
            Wk_sub[6,6] = (fermi_Cd_tau * fermi_PSD_Cd) / 2 * (1 - np.exp(-2*sub_dt/fermi_Cd_tau)) #(fermi_Cd_sigma * sub_dt/fermi_dt)**2 * fermi_Cd_sigma_scale
            fermiSat.dt = sub_dt
            Phi_k = fermiSat.get_transition_matrix(taylor_order=EKF_taylor_order)
            P_kf = Phi_k @ P_kf @ Phi_k.T + Wk_sub
            x_kf = fermiSat.step()
            fermiSat.dt = fermi_dt
            step_count += sub_dt / fermi_dt
        # Conduct Measurement Update 
        with_ei2b = True
        if with_ei2b:
            zbar = transform_coordinate_system(x_kf[0:3], tk, "ei2b", sat_handle = fermiSat) #H @ x_kf
            #H = transform_coordinate_system_jacobian_H(x_kf[0:3], tk, "ei2b", sat_handle = fermiSat)
            H = np.hstack( (np.array(list(fermiSat.csConverter.GetLastRotationMatrix().GetDataVector())).reshape((3,3)), np.zeros((3,4)) ) )
        else:
            zbar = H @ x_kf
            zk = transform_coordinate_system(zk, tk, "eb2i", sat_handle = fermiSat)
        xbars_kf.append(x_kf.copy())
        rk = zk - zbar
        rk_norm = np.linalg.norm(rk)
        chi_square_val = rk @ np.linalg.inv(V) @ rk
        #is_cs_lt_2sig = chi_square_val < twosig_s3
        is_cs_lt_3sig = chi_square_val < msmt_sigma_edit_value
        print("MU {}/{} Stats:\n  Residual: {}\n  Norm residual: {} km -> {} m\n  Chi Square {} of {} limit".format(count, len(gps_msmts), rk, rk_norm, 1000*rk_norm, chi_square_val, msmt_sigma_edit_value))
        # Reject rk if we have the case rk is way to large (bad msmt time stamp)
        if(is_cs_lt_3sig):
            K_kf = np.linalg.solve(H @ P_kf @ H.T + V, H @ P_kf).T
            x_kf = x_kf + K_kf @ rk
            P_kf = (I7 - K_kf @ H) @ P_kf @ (I7 - K_kf @ H).T + K_kf @ V @ K_kf.T 
            # Make sure changes in Cd/Cr are within bounds
            x_kf[6:] = np.clip(x_kf[6:], -0.98, 12)
            fermiSat.reset_state(x_kf, step_count)
        else:
            print("At GPS Measurement Time: {} has rk_norm = {} too large! resid.T @ V.I @ resid {} > SIG_VAL={}, Rejecting Measurement!".format(tk, rk_norm, chi_square_val, msmt_sigma_edit_value))
        # Append 
        xs_kf.append(x_kf.copy())
        Ps_kf.append(P_kf.copy())

        if gmat_ekf_data is not None:
            t_idx_gmat_ekf = find_time_match(tk, gmat_ekf_data[0], start_idx = t_idx_gmat_ekf)
            # GMAT EKF Compare
            gmat_ekf_mean = gmat_ekf_data[1][t_idx_gmat_ekf]
            gmat_ekf_cov = gmat_ekf_data[2][t_idx_gmat_ekf]
            if with_ei2b:
                gmat_ekf_zbar = transform_coordinate_system(gmat_ekf_mean[0:3], tk, "ei2b", sat_handle = fermiSat)
            else:
                gmat_ekf_zbar = gmat_ekf_mean[0:3].copy()
            gmat_ekf_resid = zk - gmat_ekf_zbar
            xs_kf_gmat.append(gmat_ekf_mean)
            Ps_kf_gmat.append(gmat_ekf_cov)
            # GMAT Smoother Compare
            if gmat_smoother_data is not None: 
                t_idx_gmat_sm = find_time_match(tk, gmat_smoother_data[0], start_idx = t_idx_gmat_sm)
                gmat_smoother_mean = gmat_smoother_data[1][t_idx_gmat_sm]
                gmat_smoother_cov = gmat_smoother_data[2][t_idx_gmat_sm]
                if with_ei2b:
                    gmat_smoother_zbar = transform_coordinate_system(gmat_smoother_mean[0:3], tk, "ei2b", sat_handle = fermiSat)
                else:
                    gmat_smoother_zbar = gmat_smoother_mean[0:3].copy()
                gmat_smoother_resid = zk - gmat_smoother_zbar

                xs_sm_gmat.append(gmat_smoother_mean)
                Ps_sm_gmat.append(gmat_smoother_cov)

            print("Step {}/{} Compare:".format(count, N) )
            #print("  GPS Says Fermi is at {}".format(zk) )
            if gmat_smoother_data is not None: 
                print("  Smooth  Says Fermi is at {}".format(gmat_smoother_mean[0:3]) )
            print("  GLAST   Diff is {} (meters)".format( 1000*(gmat_smoother_mean[0:3]-gmat_ekf_mean[0:3])) )
            print("  KF EXP  Diff is {} (meters)".format( 1000*(gmat_smoother_mean[0:3]-x_kf[0:3])) )
            if gmat_smoother_data is not None: 
                print("  Smooth Residual is {}, norm: {}".format(gmat_smoother_resid, np.linalg.norm(gmat_smoother_resid)) )
            print("  GLAST  Residual is {}, norm: {}".format(gmat_ekf_resid, np.linalg.norm(gmat_ekf_resid)) )
            print("  KF EXP residual is {}, norm: {}".format(rk, rk_norm) )
            
        # Update last time step to point to the current time instance
        tkm1 = tk
        ts.append(tk)
    # Now run the smoother backwards
    # assert(False) -> Not yet, just use the GMAT ONE
    fermiSat.clear_model()
    return ts, xs_kf, xbars_kf, Ps_kf, xs_kf_gmat, Ps_kf_gmat, xs_sm_gmat, Ps_sm_gmat

# Run MCE over same interval as EKF was run
def run_fermi_mce(gps_msmts, t0, _x0, _P0, run_dic,
    tup_exp_ekf = None, tup_gmat_ekf = None, tup_sm = None):
    # Take the first seven states (pos,vel,Cd) -> convert Cd and Cd Covariance to change in atms. dens and its covariance
    if _x0.size > 7:
        x0 = _x0[0:7].copy()
        P0 = np.delete(_P0, 7, axis=0)
        P0 = np.delete(P0, 7, axis=1)
    else:
        x0 = _x0.copy() 
        P0 = _P0.copy()
    Cd_nominal = x0[6]
    x0[6] = 0
    _T = np.eye(7)
    _T[6,6] /= Cd_nominal
    P0 = _T @ P0 @ _T.T

    gamma_cauchy_scaled = run_dic['gamma_cauchy'] * run_dic['gamma_scale']
    beta_cauchy_scaled = np.array([run_dic['beta_cauchy']]).reshape(-1) * run_dic['beta_scale']
    msmt_sigma_edit_percentile = run_dic['mce_msmt_sigma_edit_percentile']
    Cd_half_life = run_dic['mce_Cd_half_life']
    mce_num_windows = run_dic['mce_num_windows']
    mce_dt_nominal = run_dic['mce_dt_nominal']
    mce_dt_max = run_dic['mce_dt_max_stoch_step']
    mce_change_atms_dens_low_thresh = run_dic['mce_change_atms_dens_low_thresh']
    mce_change_atms_dens_high_thresh = run_dic['mce_change_atms_dens_high_thresh']

    with_kf_init = run_dic['with_mce_kf_init'] 
    with_reset_mean = run_dic['with_reset_mean_to_kf_for_mce']

    # Gaussian Msmt Noise and outlier checker (this can probably be deleted, cauchy will be robust)
    V = np.eye(3) * (gamma_cauchy_scaled * ce.CAUCHY_TO_GAUSSIAN_NOISE / 1000)**2
    msmt_sigma_edit_value = chi2.ppf(msmt_sigma_edit_percentile, 3)
    mce_Cd_tau = Cd_half_life / np.log(2)
    mce_gamma = np.ones(3) * gamma_cauchy_scaled
    mce_beta = beta_cauchy_scaled.copy()
    mce_foo_std_Cd = 1.0 # Unused
    
    mce_t0 = t0
    mce_x0 = x0.copy()
    mce_x0[0:6] *= 1000
    mce_dt0 = (gps_msmts[0][0] - mce_t0).total_seconds()
    fermSat = FermiSatelliteModel(t0, x0[0:6].copy(), mce_dt0, gmat_print=True)
    fermSat.create_model(with_jacchia=True, with_SRP=True, Cd0 = Cd_nominal)
    fermSat.set_solve_for(field="Cd", dist="sas", scale=mce_foo_std_Cd, tau=mce_Cd_tau, alpha=1.3)
    fermSat.reset_state(x0.copy(), 0)
    if with_kf_init:
        kf_x1 = tup_gmat_ekf[1][1].copy() 
        kf_P1 = tup_gmat_ekf[2][1].copy()
        kf_P1 = _T @ kf_P1 @ _T.T 
        kf_P1 = cov_km_to_m(kf_P1)
        kf_P1 *= 3
        kf_x1[0:6] *= 1000 # km 2 m
        kf_x1[6] = 1e-4 # very close to zero but not zero...just to give a small update
        x0bar = fermSat.step().copy()
        mce_x0bar = x0bar.copy()
        mce_x0bar[0:6] *= 1000 # km 2 m
        mce_x1 = mce_x0bar.copy()
        mce_dx1 = kf_x1 - mce_x1
        time_z1_utc = gps_msmts[0][0] 
        z1_ebf = gps_msmts[0][1].copy() 
        z1bar_ebf = transform_coordinate_system(x0bar[0:3], time_z1_utc, mode = "ei2b", sat_handle = fermSat)
        z1bar_ebf *= 1000 # km 2 m
        H_ebf = np.hstack( (np.array(list(fermSat.csConverter.GetLastRotationMatrix().GetDataVector())).reshape((3,3)), np.zeros((3,4)) ) )
        mce_dz1 = z1_ebf - z1bar_ebf
        mce_A0, mce_p0, mce_b0 = ce.speyers_window_init(mce_dx1, kf_P1, H_ebf[0], mce_gamma[0], mce_dz1[0])
    else:
        mce_Phi0 = fermSat.get_transition_matrix(taylor_order=gmce.global_STM_taylor_order, use_units_km=False)
        # CHANGE BELOW
        #mce_A0 = mce_Phi0.T 
        #mce_p0 = gmce.mce_naive_p0.copy() 
        # TO THIS AFTER RUN OF 20/30 IS COMPLETE
        mce_A0 = mce_Phi0.T / np.linalg.norm(mce_Phi0.T,ord=1, axis=1).reshape((7,1))
        mce_p0 = np.linalg.norm(mce_Phi0.T,ord=1, axis=1) * np.diag(cov_km_to_m(P0))**0.5 / 1.3898
        mce_b0 = np.zeros(7)
        mce_x0bar = fermSat.step().copy()
        mce_x0bar[0:6] *= 1000 # km 2 m
    fermSat.clear_model()
    mce_init_func = gmce.reinitialize_func_H_summation_ebf #gmce.reinitialize_func_speyer #gmce.reinitialize_func_H_summation_ebf
    mce_other_params = tup_gmat_ekf[2]
    mce_t0_start = gps_msmts[0][0]
    cauchyEst = gmce.GmatMCE(mce_num_windows, mce_t0_start, mce_x0bar, mce_dt_nominal,
                    mce_A0, mce_p0, mce_b0, mce_beta, mce_gamma,
                    Cd_dist="sas", std_Cd=mce_foo_std_Cd, tau_Cd=mce_Cd_tau, Cd_nominal=Cd_nominal,
                    EDIT_CHNG_ATM_DENS_LOW = mce_change_atms_dens_low_thresh,
                    EDIT_CHNG_ATM_DENS_HIGH = mce_change_atms_dens_high_thresh,
                    win_reinitialize_func=mce_init_func,
                    win_reinitialize_params=mce_other_params,
                    debug_print=True, mce_print=False, with_ei2b = True)
    # Append the prior 'warm start' condition -- Keeps consistent with KF
    cauchyEst.no_msmt_update_moment_append( 0, mce_x0.copy(), P0.copy(), with_window_append=False) 

    tkm1 = gps_msmts[0][0]
    count = 0
    ellapsed_time = mce_dt0
    ellapsed_time_since_mu = mce_dt0
    N = len(gps_msmts)
    for gps_msmt in gps_msmts:
        tk, _zk = gps_msmt
        zk = _zk.copy() / 1000 # m -> km 
        t_delta = tk - tkm1 
        dt_step = t_delta.total_seconds()
        gmce.global_date = tk

        # Measurement Outlier Rejection Testing
        with_ei2b = True
        if count > 0:
            xhat = cauchyEst.xhat.copy()
            xhat[0:6] /= 1000
            gmce.global_leo.reset_state_with_ellapsed_time(xhat, ellapsed_time)
            gmce.global_leo.dt = dt_step
            xhat_prop = gmce.global_leo.step()
        else:
            xhat_prop = mce_x0bar.copy() 
            xhat_prop[0:6] /= 1000
        zbar = transform_coordinate_system(tup_gmat_ekf[1][count+1][0:3], tk, "ei2b", sat_handle = gmce.global_leo) # PUT THIS IN AND CHANGE CHI_SQUARE_VAL BACK TO KFS!!! MAY NEED TO RERUN 20/30
        #zbar = transform_coordinate_system(xhat_prop[0:3], tk, "ei2b", sat_handle = gmce.global_leo)
        rk = zk - zbar
        rk_norm = np.linalg.norm(rk)
        chi_square_val = rk @ np.linalg.inv(V) @ rk
        is_cs_lt_3sig = chi_square_val < msmt_sigma_edit_value
        print("MU {}/{} Stats:\n  Residual: {}\n  Norm residual: {} km -> {} m\n  Chi Square {} of {} limit".format(count+1, len(gps_msmts), rk, rk_norm, 1000*rk_norm, chi_square_val, msmt_sigma_edit_value))
        
        # Regardless if we reject the measurement or not, we need to update the CF
        # To keep Phi well approximated, lets chunk up the dt between last step and this step
        dt_sub_steps = []
        if dt_step < mce_dt_nominal:
            dt_sub_steps.append(dt_step)
            dt_step = 0
        else: #dt_step >= mce_dt_nom:
            while dt_step > 0:
                if dt_step <= mce_dt_max:
                    dt_sub_steps.append(dt_step)
                    dt_step = 0
                else:
                    dt_sub_steps.append(mce_dt_nominal)
                    dt_step -= mce_dt_nominal
        dt_step_last = dt_sub_steps[-1]
        if len(dt_sub_steps) > 1:
            time_substeps = dt_sub_steps if cauchyEst.first_mu else dt_sub_steps[:-1] # if we are at first measurement update, need to deterministically propagate all the way, otherwise, all but last chunk
            cauchyEst.deterministic_time_prop(ellapsed_time, time_substeps, with_append_prop = (not is_cs_lt_3sig) ) # append only if sigma edited out
            ellapsed_time += np.sum(time_substeps)
            ellapsed_time_since_mu += np.sum(time_substeps)
        # Reject rk if we have the case rk is way to large (bad msmt time stamp)
        if(is_cs_lt_3sig):
            # Scale up beta for the last substep, as we will 
            if len(dt_sub_steps) > 1:
                mce_beta = np.array([ beta_cauchy_scaled ]) if cauchyEst.first_mu else np.array([ ( (1 - np.exp(-ellapsed_time_since_mu/mce_Cd_tau)) / (1 - np.exp(-dt_step_last/mce_Cd_tau)) ) * beta_cauchy_scaled ])
            else:
                mce_beta = np.array([ beta_cauchy_scaled ])
            cauchyEst.reset_beta(mce_beta)
            was_first_mu = cauchyEst.first_mu
            if with_reset_mean == False:
                reset_val = None
            else:
                # NEED A METHOD TO EXPLICITY SET CD
                # THERE IS A BUG WHEN WE DO DETERMINISTIC PROP AFTER SKIPPING A MEASUREMENT WITH THIS METHOD -- FIX!!!
                reset_val = tup_gmat_ekf[1][count].copy()
                reset_val[6] = reset_val[6] / Cd_nominal - 1
                gmce.global_leo.dt = ellapsed_time_since_mu
                gmce.global_leo.reset_state_with_ellapsed_time(reset_val, ellapsed_time - ellapsed_time_since_mu)
                reset_val = gmce.global_leo.step().copy()
                gmce.global_leo.dt = dt_step_last
                assert(False) # DONT ALLOW UNTIL DEBUGGED BY ME

            cauchyEst.step(zk, is_inputs_meters=False, ellapsed_time = ellapsed_time, dt_step = dt_step_last, reset_mean = reset_val, print_state_innovation=True)
            ellapsed_time += dt_step_last if not was_first_mu else 0
            ellapsed_time_since_mu = 0
        else:
            print("At GPS Measurement Time: {} has rk_norm = {} too large! resid.T @ V.I @ resid {} > SIG_VAL={}, Rejecting Measurement!".format(tk, rk_norm, chi_square_val, msmt_sigma_edit_value))
            if dt_step_last > 0: # should only NOT hit for cauchyEst.first_mu == True
                cauchyEst.deterministic_time_prop(ellapsed_time, [dt_step_last], with_append_prop=False, with_overwrite_last=True)
                ellapsed_time += dt_step_last
                ellapsed_time_since_mu += dt_step_last
            else: # first step, sigma edit
                xprop = mce_x0bar.copy()
                P_prop = np.diag(mce_p0 * 1.3898)**2
                cauchyEst.no_msmt_update_moment_append(0, xprop, P_prop)
        
        # Update last time step to point to the current time instance
        tkm1 = tk
        count += 1

        if tup_exp_ekf is not None:
            # MCE + AVG Compare 
            if with_ei2b:
                mce_zbar = transform_coordinate_system(cauchyEst.xhat[0:3]/1000, tk, "ei2b", sat_handle = gmce.global_leo)
                avg_mce_zbar = transform_coordinate_system(cauchyEst.avg_xhat[0:3]/1000, tk, "ei2b", sat_handle = gmce.global_leo)
            else:
                mce_zbar = cauchyEst.xhat[0:3][0:3].copy() / 1000
                avg_mce_zbar = cauchyEst.avg_xhat[0:3][0:3].copy() / 1000
            
            # GMAT EKF Compare
            gmat_ekf_mean = tup_gmat_ekf[1][count]
            #gmat_ekf_cov = tup_gmat_ekf[2][count]
            if with_ei2b:
                gmat_ekf_zbar = transform_coordinate_system(gmat_ekf_mean[0:3], tk, "ei2b", sat_handle = gmce.global_leo)
            else:
                gmat_ekf_zbar = gmat_ekf_mean[0:3].copy()
            # Compare to Smoother 

            # EKF Experimental Compare
            gmat_exp_mean = tup_exp_ekf[1][count]
            #gmat_exp_cov = tup_exp_ekf[2][count]
            if with_ei2b:
                gmat_exp_zbar = transform_coordinate_system(gmat_exp_mean[0:3], tk, "ei2b", sat_handle = gmce.global_leo)
            else:
                gmat_exp_zbar = gmat_exp_mean[0:3].copy()

            gmat_smoother_mean = tup_sm[1][count].copy()
            # Calculate error against smoother, the measurement residual, the measurement residual norm, and (possibly) the state innovation xhat - xbar
            mce_err_sm = np.round(1000*gmat_smoother_mean[0:3]-cauchyEst.xhat[0:3],3)
            mce_resid = np.round( 1000*(zk - mce_zbar), 3)
            mce_resid_norm = np.round(np.linalg.norm(mce_resid), 3)

            avg_mce_err_sm = np.round(1000*gmat_smoother_mean[0:3]-cauchyEst.avg_xhat[0:3],3)
            avg_mce_resid = np.round( 1000*(zk - avg_mce_zbar), 3)
            avg_mce_resid_norm = np.round(np.linalg.norm(avg_mce_resid), 3)
            
            gmat_ekf_err_sm = np.round(1000*(gmat_smoother_mean[0:3]-gmat_ekf_mean[0:3]),3)
            gmat_ekf_resid = np.round( 1000*(zk - gmat_ekf_zbar), 3)
            gmat_ekf_resid_norm = np.round(np.linalg.norm(gmat_ekf_resid), 3)
            
            gmat_exp_err_sm = np.round(1000*(gmat_smoother_mean[0:3]-gmat_exp_mean[0:3]),3)
            gmat_exp_resid =  np.round( 1000*(zk - gmat_exp_zbar), 3)
            gmat_exp_resid_norm = np.round(np.linalg.norm(gmat_exp_resid), 3)
            if len(tup_exp_ekf) == 4:
                gmat_exp_xbar = tup_exp_ekf[3][count]
                gmat_exp_innov = gmat_exp_mean - gmat_exp_xbar
                gmat_exp_innov[0:6] *= 1000
                gmat_exp_innov = np.round(gmat_exp_innov, 6)
                print("KF EXP X_INNOV {} (meters)".format(gmat_exp_innov) )
            if with_ei2b:
                gmat_smoother_zbar = transform_coordinate_system(gmat_smoother_mean[0:3], tk, "ei2b", sat_handle = gmce.global_leo)
            else:
                gmat_smoother_zbar = gmat_smoother_mean[0:3].copy()
            
            gmat_smoother_mean = np.round(gmat_smoother_mean[0:3],6)
            gmat_smoother_resid = np.round(1000*(zk - gmat_smoother_zbar),3)
            gmat_smoother_resid_norm = np.round(np.linalg.norm(gmat_smoother_resid), 3)
            print("Step {}/{} Compare:".format(count, N) )
            #print("  GPS Says Fermi is at {}".format(zk) )
            print("  Smooth  STATE {}(km) ZResid {}(meters) ZResNorm {}(meters)".format(gmat_smoother_mean, gmat_smoother_resid, gmat_smoother_resid_norm) )
            print("  GLAST   ERR_SM {} ZResid {} ZResNorm {} (meters)".format(gmat_ekf_err_sm,  gmat_ekf_resid, gmat_ekf_resid_norm) )
            print("  KF EXP  ERR_SM {} ZResid {} ZResNorm {}".format(gmat_exp_err_sm, gmat_exp_resid, gmat_exp_resid_norm) )
            print("  MCE     ERR_SM {} ZResid {} ZResNorm {} (meters)".format(mce_err_sm, mce_resid,  mce_resid_norm) )
            print("  AVGMCE  ERR_SM {} ZResid {} ZResNorm {} (meters)".format(avg_mce_err_sm, avg_mce_resid, avg_mce_resid_norm) )
    
    gmce.global_leo.clear_model()
    return cauchyEst.xhats, cauchyEst.Phats, cauchyEst.avg_xhats, cauchyEst.avg_Phats

def plot_ekf_mce_sigma_bound_and_errors(
        ts, xs_kf_exp, Ps_kf_exp, 
        xs_kf_gmat, Ps_kf_gmat, 
        xs_mce, Ps_mce,
        xs_mce_avg, Ps_mce_avg,
        xs_sm, Ps_sm, log_path = None):
    # Deal with data in order: MCE, KF_EXP, KF_GMAT -- SMOOTHER IS TRUTH
    t0 = ts[0]
    tks = np.array([(tk-t0).total_seconds() for tk in ts])
    if xs_kf_exp is not None:
        es_kf_exp = np.array([xt - xh[0:6] for xt,xh in zip(xs_sm,xs_kf_exp)])
        sigs_kf_exp = np.array([np.diag(P)**0.5 for P in Ps_kf_exp])
    if xs_kf_gmat is not None:
        es_kf_gmat = np.array([xt - xh[0:6] for xt,xh in zip(xs_sm,xs_kf_gmat)])
        sigs_kf_gmat = np.array([np.diag(P)**0.5 for P in Ps_kf_gmat])
    if xs_mce is not None:
        es_mce = np.array([xt - (xh[0:6]/1000) for xt,xh in zip(xs_sm,xs_mce)])
        sigs_mce = np.array([np.diag(cov_m_to_km(P))**0.5 for P in Ps_mce])
    if xs_mce_avg is not None:
        es_mce_avg = np.array([xt - (xh[0:6]/1000) for xt,xh in zip(xs_sm,xs_mce_avg)])
        sigs_mce_avg = np.array([np.diag(cov_m_to_km(P))**0.5 for P in Ps_mce_avg])

    sigs_sm = np.array([np.diag(P)**0.5 for P in Ps_sm])
    ylabels = ['PosX (km)', 'PoxY (km)', 'PosZ (km)', 'VelX (km/s)', 'VelY (km/s)', 'VelZ (km/s)']
    plt.figure(figsize=(14,11))
    color_scheme = "KF EXP (blue) + KF GMAT (green) + MCE (magenta) + MCE Avg (black)"
    plt.suptitle("State Error Plot of Position/Velocity\nSolid Lines = State Errors, Dashed Lines = One Sigma Bounds\n"+color_scheme)
    for i in range(6):
        plt.subplot(6,1,i+1)
        plt.ylabel(ylabels[i])
        if xs_kf_exp is not None:
            plt.plot(tks, es_kf_exp[:,i], color='b')
            plt.plot(tks, sigs_kf_exp[:,i], color='b', linestyle='--')
            plt.plot(tks, -sigs_kf_exp[:,i], color='b', linestyle='--')
        if xs_kf_gmat is not None:
            plt.plot(tks, es_kf_gmat[:,i], color='g')
            plt.plot(tks, sigs_kf_gmat[:,i], color='g', linestyle='--')
            plt.plot(tks, -sigs_kf_gmat[:,i], color='g', linestyle='--')
        if xs_mce is not None:
            plt.plot(tks, es_mce[:,i], color='m')
            plt.plot(tks, sigs_mce[:,i], color='m', linestyle='--')
            plt.plot(tks, -sigs_mce[:,i], color='m', linestyle='--')
        if xs_mce_avg is not None:
            plt.plot(tks, es_mce_avg[:,i], color='k')
            plt.plot(tks, sigs_mce_avg[:,i], color='k', linestyle='--')
            plt.plot(tks, -sigs_mce_avg[:,i], color='k', linestyle='--')
        plt.plot(tks, -sigs_sm[:,i], color='r', linestyle='--')
        plt.plot(tks, sigs_sm[:,i], color='r', linestyle='--')
    plt.xlabel("Time (sec)")
    if log_path is not None:
        plt.savefig(log_path)
        plt.close()
    else:
        plt.show()

def get_propagated_filter_estimates_and_compare_to_smoother_ephemeris(t0, dt_pred, x_kf, P_kf, _x_mce, _P_mce, _x_mce_avg, _P_mce_avg, ts_sm, _xs_sm, dt_lookahead, kf_Cd_half_life, mce_Cd_half_life, Cd_nominal, sigma_scale = 3):
        # 0.) Find starting point in smoother ephemeris
    # 1.) Take t0 + dt_lookahead and find the closest_point in the smoother ephemeris 
    # 2.) Propagate KF, MCE deterministically 
    s = sigma_scale # The sigma level (1-sig ~ 68%ish to 3-sig is 99.7%ish)
    t0_idx = find_time_match(t0, ts_sm)
    tf = t0 + timedelta(seconds=dt_lookahead) # dt_lookahead given in seconds 
    tf_idx = find_approximate_time_match(tf, ts_sm, t0_idx)
    tf = ts_sm[tf_idx]
    time_to_go = (tf-t0).total_seconds()
    ts = [0] # List of ellaspsed smoother times
    print("Propagating State Estimates from t0={} to tf={}, time_span={} seconds.\nThis will match smoother ephemeris time steps perfectly!".format( t0, tf, time_to_go ))

    # Setup Fermi Sat Object for KF
    kf_Cd_tau = kf_Cd_half_life / np.log(2)
    fermi_sat = FermiSatelliteModel(t0, x_kf[0:6], dt_pred, gmat_print=False)
    fermi_sat.create_model(with_jacchia=True, with_SRP=True, Cd0 = Cd_nominal)
    fermi_sat.set_solve_for("Cd", "sas", scale=0.0013, tau=kf_Cd_tau, alpha=1.3) # Nothing other than Cd_tau is needed here for input
    fermi_sat.reset_state(x_kf, 0)
    
    xk = x_kf.copy()
    Pk = P_kf.copy()
    xs_kf_pred = [xk.copy()] 
    Ps_kf_pred = [Pk.copy()] 

    # Propagate the KF 
    t_idx = t0_idx
    while time_to_go > 0: # t_idx < tf_idx would also work
        _dt_sm = (ts_sm[t_idx+1] - ts_sm[t_idx]).total_seconds()
        ts.append( _dt_sm if t_idx == t0_idx else _dt_sm + ts[-1])
        dt_sm = _dt_sm
        while dt_sm > 0:
            if dt_sm < dt_pred:
                fermi_sat.dt = dt_sm
            #Phik = fermi_sat.get_transition_matrix(taylor_order = gmce.global_STM_taylor_order, use_units_km = True)
            Phik = fermi_sat.get_simple_transition_matrix(taylor_order = gmce.global_STM_taylor_order, use_units_km = True)
            xk = fermi_sat.step()
            dt_sm -= dt_pred
        xs_kf_pred.append( xk.copy( ) )
        Ps_kf_pred.append( Pk.copy( ) )
        fermi_sat.dt = dt_pred
        time_to_go -= _dt_sm
        t_idx += 1

    fermi_sat.clear_model()
    
    # Rinse and repeat for the MCE
    for i in range(2):
        Cd_std_foo = 1 # Unused
        x_mce = _x_mce.copy() if i == 0 else _x_mce_avg.copy()
        x_mce[0:6] /= 1000
        mce_Cd_tau = mce_Cd_half_life / np.log(2)
        fermi_sat = FermiSatelliteModel(t0, x_mce[0:6], dt_pred, gmat_print=False)
        fermi_sat.create_model(with_jacchia=True, with_SRP=True, Cd0 = Cd_nominal)
        fermi_sat.set_solve_for("Cd", "sas", scale=Cd_std_foo, tau=mce_Cd_tau, alpha=1.3) # Nothing other than Cd_tau is needed here for input
        fermi_sat.reset_state(x_mce, 0)
        P_mce = cov_m_to_km(_P_mce if i == 0 else _P_mce_avg)
        time_to_go = (tf-t0).total_seconds()
        xk = x_mce.copy()
        Pk = P_mce.copy()
        if i == 0:
            xs_mce_pred = [xk.copy()]
            Ps_mce_pred = [Pk.copy()]
        else:
            xs_mce_avg_pred = [xk.copy()]
            Ps_mce_avg_pred = [Pk.copy()]
        # Propagate the MCE 
        t_idx = t0_idx
        while time_to_go > 0: # t_idx < tf_idx would also work
            _dt_sm = (ts_sm[t_idx+1] - ts_sm[t_idx]).total_seconds()
            dt_sm = _dt_sm
            while dt_sm > 0:
                if dt_sm < dt_pred:
                    fermi_sat.dt = dt_sm
                Phik = fermi_sat.get_simple_transition_matrix(taylor_order = gmce.global_STM_taylor_order, use_units_km = True)
                Pk = Phik @ Pk @ Phik.T
                xk = fermi_sat.step()
                dt_sm -= dt_pred
            # Convert MCE estimates to KM
            if i == 0:
                xs_mce_pred.append( xk.copy() )
                Ps_mce_pred.append( Pk.copy() )
            else:
                xs_mce_avg_pred.append( xk.copy() )
                Ps_mce_avg_pred.append( Pk.copy() )
            fermi_sat.dt = dt_pred
            time_to_go -= _dt_sm
            t_idx += 1
        fermi_sat.clear_model()

    N = len(xs_kf_pred)
    xs_sm = np.array(_xs_sm)[t0_idx:tf_idx+1,:]
    xs_kf_pred = np.array(xs_kf_pred)[:,0:6]
    Ps_kf_pred = np.array(Ps_kf_pred)[:, 0:3, 0:3]
    xs_mce_pred = np.array(xs_mce_pred)[:,0:6]
    Ps_mce_pred = np.array(Ps_mce_pred)[:,0:3, 0:3]
    xs_mce_avg_pred = np.array(xs_mce_avg_pred)[:,0:6]
    Ps_mce_avg_pred = np.array(Ps_mce_avg_pred)[:,0:3, 0:3]
    ts = np.array(ts) / 3600 # convert to hours
    # KF/MCE Errors
    es_kf_pred = (xs_sm - xs_kf_pred)[:, 0:3]
    sigs_kf_pred = np.array([np.diag(P)**0.5 for P in Ps_kf_pred])

    es_mce_pred = (xs_sm - xs_mce_pred)[:, 0:3]
    sigs_mce_pred = np.array([np.diag(P)**0.5 for P in Ps_mce_pred])

    es_mce_avg_pred = (xs_sm - xs_mce_avg_pred)[:, 0:3]
    sigs_mce_avg_pred = np.array([np.diag(P)**0.5 for P in Ps_mce_avg_pred])
    
    # Now convert these errors to along cross and radial track
    es_acr_kf_pred = np.zeros((N,3))
    sigs_acr_kf_pred = np.zeros((N,3))
    es_acr_mce_pred = np.zeros((N,3))
    sigs_acr_mce_pred = np.zeros((N,3))
    es_acr_mce_avg_pred = np.zeros((N,3))
    sigs_acr_mce_avg_pred = np.zeros((N,3))
    count = 0
    for xkf,Pkf, xmce,Pmce, xmce_avg,Pmce_avg, xsm in zip(xs_kf_pred,Ps_kf_pred, xs_mce_pred,Ps_mce_pred, xs_mce_avg_pred,Ps_mce_avg_pred, xs_sm):
        e_acr_kf, P_acr_kf, _ = get_along_cross_radial_errors_cov(xkf, Pkf, xsm)
        e_acr_mce, P_acr_mce, _ = get_along_cross_radial_errors_cov(xmce, Pmce, xsm)
        e_acr_mce_avg, P_acr_mce_avg, _ = get_along_cross_radial_errors_cov(xmce_avg, Pmce_avg, xsm)
        es_acr_kf_pred[count, :] = e_acr_kf
        sigs_acr_kf_pred[count, :] = s * np.diag(P_acr_kf)**0.5
        es_acr_mce_pred[count, :] = e_acr_mce
        sigs_acr_mce_pred[count, :] = s * np.diag(P_acr_mce)**0.5
        es_acr_mce_avg_pred[count, :] = e_acr_mce_avg
        sigs_acr_mce_avg_pred[count, :] = s * np.diag(P_acr_mce_avg)**0.5
        count += 1
    return ts, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, es_mce_avg_pred, sigs_mce_avg_pred, es_acr_kf_pred, sigs_acr_kf_pred, es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred

def plot_propagated_filter_estimates_and_compare_to_smoother_ephemeris(
        ts, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, 
        es_mce_avg_pred, sigs_mce_avg_pred, es_acr_kf_pred, sigs_acr_kf_pred, 
        es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred, 
        sigma_scale = 3, log_path_cartesian = None, log_path_along_cross_radial = None):
    s = sigma_scale
    # Plot cartesian on left hand side subplots
    plt.figure(figsize=(14,11))
    plt.suptitle("Predictive Estimation ERRORS \n KF (left green) + MCE (center blue) + MCE Avg (right red) \nusing KF Smoother Solution as Definitive Ephem")
    ylabels = ['x (km)', 'y (km)', 'z (km)']
    for i in range(3):
        #plt.subplot(3,2,2*i+1)
        plt.subplot(3,3,3*i+1)
        plt.plot(ts, es_kf_pred[:,i], 'g')
        plt.plot(ts, sigs_kf_pred[:,i], 'g--')
        plt.plot(ts, -sigs_kf_pred[:,i], 'g--')
        plt.ylabel(ylabels[i])
        if i == 2:
            plt.xlabel('Hours Ellapsed Since Last Measurement Update')
        plt.subplot(3,3,3*i+2)
        plt.plot(ts, es_mce_pred[:,i], 'b')
        plt.plot(ts, sigs_mce_pred[:,i], 'b--')
        plt.plot(ts, -sigs_mce_pred[:,i], 'b--')
        if i == 2:
            plt.xlabel('Hours Ellapsed Since Last Measurement Update')
        plt.subplot(3,3,3*i+3)
        plt.plot(ts, es_mce_avg_pred[:,i], 'r')
        plt.plot(ts, sigs_mce_avg_pred[:,i], 'r--')
        plt.plot(ts, -sigs_mce_avg_pred[:,i], 'r--')
        if i == 2:
            plt.xlabel('Hours Ellapsed Since Last Measurement Update')
    if log_path_cartesian is not None:
        plt.savefig(log_path_cartesian)
        plt.close()
    
    plt.figure(figsize=(14,11))
    plt.suptitle("ACR Predictive Estimation ERRORS\n KF (left green) + MCE (center blue) + MCE Avg (right red) \nusing KF Smoother Solution as Definitive Ephem")
    ylabels = ['Along (km)', 'Cross (km)', 'Radial (km)']
    for i in range(3):
        #plt.subplot(3,2,2*i+2)
        plt.subplot(3,3,3*i+1)
        plt.plot(ts, es_acr_kf_pred[:,i], 'g')
        plt.plot(ts, sigs_acr_kf_pred[:,i], 'g--')
        plt.plot(ts, -sigs_acr_kf_pred[:,i], 'g--')
        plt.ylabel(ylabels[i])
        if i == 2:
            plt.xlabel('Hours Ellapsed Since Last Measurement Update')
        plt.subplot(3,3,3*i+2)
        plt.plot(ts, es_acr_mce_pred[:,i], 'b')
        plt.plot(ts, sigs_acr_mce_pred[:,i], 'b--')
        plt.plot(ts, -sigs_acr_mce_pred[:,i], 'b--')
        if i == 2:
            plt.xlabel('Hours Ellapsed Since Last Measurement Update')
        plt.subplot(3,3,3*i+3)
        plt.plot(ts, es_acr_mce_avg_pred[:,i], 'r')
        plt.plot(ts, sigs_acr_mce_avg_pred[:,i], 'r--')
        plt.plot(ts, -sigs_acr_mce_avg_pred[:,i], 'r--')
        if i == 2:
            plt.xlabel('Hours Ellapsed Since Last Measurement Update')
    if log_path_along_cross_radial is not None:
        plt.savefig(log_path_along_cross_radial)
        plt.close()
    if (log_path_cartesian is not None) and (log_path_along_cross_radial is not None):
        plt.show()
    foobar = 2

# Run KF over subintervals of definitive ephem
# Run MCE over subintervals of definitive ephem
# Predict and compare each subinterval
def test_pred_def_overlap_3day(gps_path, restart_ekf_path, restart_smoother_path, ensemble_sub_dir_name, with_plotting = True):
    # Load data from input file paths
    all_gps_msmts = load_gps_from_txt(gps_path)
    outputted_ekf = load_glast_file(restart_ekf_path)
    outputted_smoother = load_ephem_file(restart_smoother_path)
    
    # Output directory and subdirectories
    ensemble_root_dir = gmat_data_dir + "pred_def_ensemble/"
    if not os.path.isdir(ensemble_root_dir):
        os.mkdir(ensemble_root_dir)
    ensemble_sub_dir_name = ensemble_sub_dir_name if ensemble_sub_dir_name[-1] == '/' else ensemble_sub_dir_name + "/"
    ensemble_fpath = ensemble_root_dir + ensemble_sub_dir_name
    fpath_run_dic_human = ensemble_fpath + "run_dic.txt"
    fpath_run_dic_pickle = ensemble_fpath + "run_dic.pickle"
    ensembles_previously_logged = 0 # for continuing data collection
    if not os.path.isdir(ensemble_fpath):
        os.mkdir(ensemble_fpath)
    else: # Check to make sure subdirectory is OK
        not_okay = True
        while not_okay:
            print("Subdirectory {} given already exists in {}! This will append data to the chosen directory, but could corrput your previous data if not careful...\nDo you really wish to continue? You may override/corrupt data...".format(ensemble_sub_dir_name, ensemble_root_dir))
            answer = input("Enter (y/n) -> 'y'=yes (continue) or 'n'=no (exit)").lower()
            if answer == 'y':
                not_okay = False
                print("Continuing!")
                with open(fpath_run_dic_pickle, 'rb') as handle:
                    old_run_dic = pickle.load(handle)
                    ensembles_previously_logged = old_run_dic["days_ensemble"]
            elif answer == 'n':
                print("Not Continuing! Goodbye!")
                exit(1)
            else:
                print("Incorrect Response! Enter (y/n) -> 'y'=yes (continue) or 'n'=no (exit)")
    
    print("Logging to ", ensemble_fpath)

    # Some helpful pieces of the smoother
    ts_sm = outputted_smoother[0] # Should change and cache restart point for day run but whats a few seconds waiting for the function to find again...
    xs_sm = outputted_smoother[1] # Should change and cache restart point for day run but whats a few seconds waiting for the function to find again...
    
    # Dynamic Constants
    fermi_Cd_half_life = 21600.0 # from Russells paper - half life of change in atms. density -- 6 hours
    fermi_gps_std_dev = 0.0075 # km

    # Kalman Filtering constants
    kf_dt = 30.0 # seconds
    kf_Cd_steady_state_sigma = 0.6 # from OD script
    kf_Cd_half_life = fermi_Cd_half_life
    kf_qs = np.array([1e-9,1e-8,1e-8]) # Square root of acceleration PSD from OD script
    kf_gps_std_dev = fermi_gps_std_dev # GPS std devs
    kf_taylor_order = 4 # for construction of STM
    kf_msmt_sigma_edit_percentile = 0.998 # Percentile/100 at which to reject measurement

    # MCE Filtering Constants
    mce_num_windows = 4
    mce_dt_max_stoch_step = 150.0 #120.0 # max stochastic propagation step for forming Phi(dt_stoch),Gam(dt_stoch)
    mce_dt_nominal = 30.0 # nominal deterministic propagation step (rotates Char Func by Phi(dt_nom))
    mce_Cd_half_life = 21600.0
    mce_taylor_order = gmce.global_STM_taylor_order    
    mce_msmt_sigma_edit_percentile = 0.998 # Percentile/100 at which to reject measurement
    mce_change_atms_dens_low_thresh = -0.05 # Treat change atms. dens. as consider state -- mean stays close to nominal but covariance can vary greatly, which is what we are after
    mce_change_atms_dens_high_thresh = 0.05 # Treat change atms. dens. as consider state -- mean stays close to nominal but covariance can vary greatly, which is what we are after
    beta_scale = 1.0 #2 # multiplies the nominal beta_cauchy param by this and gives it to the MCE 
    # TO
    beta_russell = 0.0013# change back to using russels number from best fit
    beta_russell_dt = 60.0
    Gam_density_multiplier = mce_Cd_half_life/np.log(2) * (1.0 - np.exp(-2*beta_russell_dt/mce_Cd_half_life) )
    beta_cauchy = beta_russell / Gam_density_multiplier
    gamma_scale = 1.0 # 1.0/1.35 # 1.0/1.5 # multiplies the nominal gamma_cauchy param by this and gives it to the MCE
    gamma_cauchy = fermi_gps_std_dev * 1000 * ce.GAUSSIAN_TO_CAUCHY_NOISE 
    with_mce_kf_init = False # False
    with_reset_mean_to_kf_for_mce = False # FAILS CURRENTLY FOR WHEN GPS ERRORS ARE REJECTED -- BUG!
    mce_naive_p0 = None if with_mce_kf_init else gmce.mce_naive_p0.copy()

    # Number of steps to run the EKF/MCE before running prediction
    filter_run_steps = 30

    # Number of Ensemble runs
    days_ensemble = 5

    # Prediction Constants
    days_lookahead = 3
    dt_pred = 60.0
    sigma_scale = 1.0 # for forming the sigma bounds on return from 

    # Some constants and defines for the filtering and prediction run
    t0_day1, _, _, _, _ = find_restart_point(restart_ekf_path, all_gps_msmts[0][0])
    days_seconds = 24.0*60.0*60.0
    tstart_idx_gps_msmts = 0
    all_gps_times = [gps[0] for gps in all_gps_msmts]
    dt_lookahead = days_lookahead * days_seconds

    # Make dic of things that went on
    run_dic = {
        # Dynamic Constants
        'fermi_Cd_half_life' : fermi_Cd_half_life,
        'fermi_gps_std_dev' : fermi_gps_std_dev,

        # Kalman Filtering constants
        'kf_dt' : kf_dt,
        'kf_Cd_steady_state_sigma' : kf_Cd_steady_state_sigma,
        'kf_Cd_half_life' : kf_Cd_half_life,
        'kf_qs' : kf_qs,
        'kf_gps_std_dev' : kf_gps_std_dev,
        'kf_taylor_order' : kf_taylor_order,
        'kf_msmt_sigma_edit_percentile' : kf_msmt_sigma_edit_percentile,

        # MCE Filtering Constants
        'mce_num_windows' : mce_num_windows,
        'mce_dt_max_stoch_step' : mce_dt_max_stoch_step,
        'mce_dt_nominal' : mce_dt_nominal,
        'mce_Cd_half_life' : mce_Cd_half_life,
        'mce_taylor_order' : mce_taylor_order,
        'mce_msmt_sigma_edit_percentile' : mce_msmt_sigma_edit_percentile,
        'mce_change_atms_dens_low_thresh' : mce_change_atms_dens_low_thresh,
        'mce_change_atms_dens_high_thresh' : mce_change_atms_dens_high_thresh,
        'beta_scale' : beta_scale,
        'beta_cauchy' : beta_cauchy,
        'gamma_scale' : gamma_scale,
        'gamma_cauchy' : gamma_cauchy,
        'with_mce_kf_init' : with_mce_kf_init,
        'with_reset_mean_to_kf_for_mce' : with_reset_mean_to_kf_for_mce,
        'mce_naive_p0' : mce_naive_p0,

        # Number of steps to run the EKF/MCE before running prediction
        'filter_run_steps' : filter_run_steps,

        # Number of Ensemble runs
        'ensembles_previously_logged' : ensembles_previously_logged,
        'days_ensemble' : days_ensemble + ensembles_previously_logged,

        # Prediction Constants
        'days_seconds' : days_seconds,
        'days_lookahead' : days_lookahead,
        'dt_pred' : dt_pred,
        'sigma_scale' : sigma_scale
    }

    # Store constants into human readable dic as well as pickle them
    with open(fpath_run_dic_human, 'w') as handle:
        for k, v in run_dic.items():
            handle.writelines("{} : {},\n".format(str(k), str(v)) )
    with open(fpath_run_dic_pickle, 'wb') as handle:
        pickle.dump(run_dic, handle)

    # Begin ensemble loop -- collect data for each run and log
    # Can corrupt start of range below to append data to an existing directory
    for day in range(1+ensembles_previously_logged,ensembles_previously_logged+days_ensemble+1): #range(1,days+1):
        day_t0_approx = t0_day1 + timedelta(seconds=(day-1)*days_seconds) # Start of day 1 would be index 0
        # Find the GPS Measurement closest to day_t0
        tstart_idx_gps_msmts = find_approximate_time_match(day_t0_approx, all_gps_times, tstart_idx_gps_msmts)
        # Use only the GPS data for this filtering run
        gps_msmts = all_gps_msmts[tstart_idx_gps_msmts:tstart_idx_gps_msmts+filter_run_steps]
        day_t0, day_x0, day_P0, labels_x0, labels_P0 = find_restart_point(restart_ekf_path, gps_msmts[0][0])
        Cd_nominal = day_x0[6]
        t0_pred = gps_msmts[filter_run_steps-1][0]
        # Run the Python GMAT EKF (should match GLAST from GMAT)
        ts, xs_kf, xbars_kf, Ps_kf, xs_kf_gmat, Ps_kf_gmat, xs_sm_gmat, Ps_sm_gmat = \
            run_fermi_ekf(gps_msmts, day_t0, day_x0, day_P0, run_dic,
                gmat_ekf_data=outputted_ekf, gmat_smoother_data=outputted_smoother) # currently no smoother... load this from GMAT
        # Run the MCE
        xs_mce, Ps_mce, xs_mce_avg, Ps_mce_avg = \
            run_fermi_mce(gps_msmts, day_t0, day_x0, day_P0, run_dic, 
                tup_exp_ekf = (ts, xs_kf, Ps_kf, xbars_kf), tup_gmat_ekf = (ts, xs_kf_gmat, Ps_kf_gmat), tup_sm = (ts, xs_sm_gmat, Ps_sm_gmat))

        # Store the the filtering outputs into a pickled file
        day_filt_fname = "day" + str(day) + "_filt.pickle"
        pickle_fpath = ensemble_fpath + day_filt_fname
        with open(pickle_fpath, 'wb') as handle:
            pickle.dump((
                tstart_idx_gps_msmts, day_t0, 
                ts, xs_kf, Ps_kf, xs_kf_gmat, Ps_kf_gmat, 
                xs_mce, Ps_mce, xs_mce_avg, Ps_mce_avg, 
                xs_sm_gmat, Ps_sm_gmat, Cd_nominal), handle)

        # Plot the filtering results against the smoother, forming state error plots
        # Maybe turn this off when confident
        if with_plotting:
            plot_ekf_mce_sigma_bound_and_errors(
                ts, 
                xs_kf, Ps_kf, 
                xs_kf_gmat, Ps_kf_gmat, 
                xs_mce, Ps_mce,
                xs_mce_avg, Ps_mce_avg,
                xs_sm_gmat, Ps_sm_gmat)

        # Return the 3 day prediction errors, the ACR errors, all against the smoother
        # These below are end of filtering and therefore the start of the prediction point
        x_kf_pred = xs_kf[-1]
        P_kf_pred = Ps_kf[-1]
        x_mce_pred = xs_mce[-1]
        P_mce_pred = Ps_mce[-1]
        x_mce_avg_pred = xs_mce_avg[-1]
        P_mce_avg_pred = Ps_mce_avg[-1]
        ts_pred, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, es_mce_avg_pred, sigs_mce_avg_pred, \
        es_acr_kf_pred, sigs_acr_kf_pred, es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred = \
            get_propagated_filter_estimates_and_compare_to_smoother_ephemeris(
                t0_pred, dt_pred, x_kf_pred, P_kf_pred, x_mce_pred, P_mce_pred, x_mce_avg_pred, P_mce_avg_pred, 
                ts_sm, xs_sm, dt_lookahead, kf_Cd_half_life, mce_Cd_half_life, Cd_nominal, sigma_scale)
        
        # Store the the prediction outputs into a pickled file
        day_pred_fname = "day" + str(day) + "_pred.pickle"
        pickle_fpath = ensemble_fpath + day_pred_fname
        with open(pickle_fpath, 'wb') as handle:
            pickle.dump((ts_pred, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, es_mce_avg_pred, sigs_mce_avg_pred, \
                es_acr_kf_pred, sigs_acr_kf_pred, es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred), handle)
        
        # Plot the prediction results against the smoother, forming state error plots for Cartesian and Along-Cross-Radial Error
        if with_plotting:
            plot_propagated_filter_estimates_and_compare_to_smoother_ephemeris(
                ts_pred, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, es_mce_avg_pred, sigs_mce_avg_pred,
                es_acr_kf_pred, sigs_acr_kf_pred, es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred, sigma_scale)
        foobar = 2
    print("Thats all folks!")
    return ensemble_fpath

def plot_pred_def_overlap_3day(ensemble_sub_dir_name, sigma_override = None, with_plot_mce_best = True, with_plot_mce_avg = True):
    # Form ensemble_fpath
    ensemble_root_dir = gmat_data_dir + "pred_def_ensemble/"
    ensemble_sub_dir_name = ensemble_sub_dir_name if ensemble_sub_dir_name[-1] == '/' else ensemble_sub_dir_name + "/"
    ensemble_fpath = ensemble_root_dir + ensemble_sub_dir_name

    # Load constants from pickle file and print them as well as human readable dic
    fpath_run_dic_pickle = ensemble_fpath + "run_dic.pickle"
    with open(fpath_run_dic_pickle, 'rb') as handle:
        run_dic = pickle.load(handle)
    print("Loaded Run From: ", ensemble_fpath)
    print("Parameters of Run Are: ")
    pprint.pprint(run_dic)
    
    days_ensemble = run_dic['days_ensemble']
    days_lookahead = run_dic['days_lookahead']
    _sigma_scale = run_dic['sigma_scale']
    # Sigma override allows us to change the sigma to whatever sigma value we want here
    sigma_override = _sigma_scale if sigma_override is None else sigma_override
    sigma_scale = sigma_override / _sigma_scale

    ylabels_cart = ['Error X (km)', 'Error Y (km)', 'error Z (km)']
    ylabels_log_cart = ['Error X Log10(km)', 'Error Y Log10(km)', 'Error Z Log10(km)']
    ylabels_acr = ['Error Along (km)', 'Cross (km)', 'Radial (km)']
    ylabels_log_acr = ['Along Log10(km)', 'Cross Log10(km)', 'Radial Log10(km)']
    ylabels_zscore = ['Z-score X', 'Z-Score Y', 'Z-Score Z']
    ylabels_zscore_acr = ['Z-score Along', 'Z-Score Cross', 'Z-Score Radial']

    xlabel = 'Hours Ellapsed Since Last Measurement Update'

    # Plot cartesian on left hand side subplots
    
    mce_err_cov_label = "+ MCE (blue/red) + MCE Avg (brown/magenta)" if (with_plot_mce_avg and with_plot_mce_best) else "+ MCE (blue/red)" if with_plot_mce_best else  "+ MCE Avg (brown/magenta)" if with_plot_mce_avg else ""
    fig1, ax1 = plt.subplots(3)
    fig1.suptitle("{}-Day Ensemble of Cartesian {}-Day Predictive Estimation Errors/{}-SigmaBounds\nKF (green/orange) {} \nusing KF Smoother Solution as Definitive Ephem".format(days_ensemble,days_lookahead,sigma_scale,mce_err_cov_label))

    fig2, ax2 = plt.subplots(3)
    fig2.suptitle("{}-Day Ensemble of Along/Cross/Radial {}-Day Predictive Estimation Errors/{}-SigmaBounds\nKF (green/orange) {} \nusing KF Smoother Solution as Definitive Ephem".format(days_ensemble,days_lookahead,sigma_scale,mce_err_cov_label))
    '''
    fig3, ax3 = plt.subplots(3)
    fig3.suptitle("Log of {}-Day Ensemble of Cartesian {}-Day Predictive Estimation Errors/{}-SigmaBounds\nKF (green/orange) {} \nusing KF Smoother Solution as Definitive Ephem".format(days_ensemble,days_lookahead,sigma_scale,mce_err_cov_label))

    fig4, ax4 = plt.subplots(3)
    fig4.suptitle("Log of {}-Day Ensemble of Along/Cross/Radial {}-Day Predictive Estimation Errors/{}-SigmaBounds\nKF (green/orange) {} \nusing KF Smoother Solution as Definitive Ephem".format(days_ensemble,days_lookahead,sigma_scale,mce_err_cov_label))
    '''
    mce_err_label = "+ MCE (blue) + MCE Avg (brown)" if (with_plot_mce_avg and with_plot_mce_best) else "+ MCE (blue)" if with_plot_mce_best else  "+ MCE Avg (brown)" if with_plot_mce_avg else ""
    fig5, ax5 = plt.subplots(3)
    fig5.suptitle("{}-Day Ensemble of {}-Day Predictive Estimation Errors\nKF (green) {} \nusing KF Smoother Solution as Definitive Ephem".format(days_ensemble,days_lookahead,mce_err_label))

    fig6, ax6 = plt.subplots(3)
    fig6.suptitle("{}-Day Ensemble of Along/Cross/Radial {}-Day Predictive Estimation Errors\nKF (green) {} \nusing KF Smoother Solution as Definitive Ephem".format(days_ensemble,days_lookahead,mce_err_label))

    fig7, ax7 = plt.subplots(3)
    fig7.suptitle("{}-Day Ensemble of {}-Day Predictive Z-Scores\nKF (green) {} \nusing KF Smoother Solution as Definitive Ephem".format(days_ensemble,days_lookahead,mce_err_label))

    fig8, ax8 = plt.subplots(3)
    fig8.suptitle("{}-Day Ensemble of Along/Cross/Radial {}-Day Predictive Z-Scores\nKF (green) {} \nusing KF Smoother Solution as Definitive Ephem".format(days_ensemble,days_lookahead,mce_err_label))

    # Load in the pickled data
    for day in range(1, days_ensemble+1):
        fpath_pred = ensemble_fpath + "day" + str(day) + "_pred.pickle"
        with open(fpath_pred, "rb") as handle:
            ts_pred, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, es_mce_avg_pred, sigs_mce_avg_pred, \
                es_acr_kf_pred, sigs_acr_kf_pred, es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred  = \
                pickle.load(handle)
        
        # Plot cartesian
        for i in range(3):
            if day == days_ensemble:
                ax1[i].set_ylabel(ylabels_cart[i])
                if i == 2:
                    ax1[i].set_xlabel(xlabel)
            if with_plot_mce_best:
                ax1[i].plot(ts_pred, es_mce_pred[:,i], 'b')
                ax1[i].plot(ts_pred, sigma_scale*sigs_mce_pred[:,i], 'r')
                ax1[i].plot(ts_pred, -sigma_scale*sigs_mce_pred[:,i], 'r')
            if with_plot_mce_avg:
                ax1[i].plot(ts_pred, es_mce_avg_pred[:,i], color='brown')
                ax1[i].plot(ts_pred, sigma_scale*sigs_mce_avg_pred[:,i], 'm')
                ax1[i].plot(ts_pred, -sigma_scale*sigs_mce_avg_pred[:,i], 'm')
            ax1[i].plot(ts_pred, es_kf_pred[:,i], 'g')
            ax1[i].plot(ts_pred, sigma_scale*sigs_kf_pred[:,i], color='orange')
            ax1[i].plot(ts_pred, -sigma_scale*sigs_kf_pred[:,i], color='orange')
    
        # Plot ACR
        for i in range(3):
            if day == days_ensemble:
                ax2[i].set_ylabel(ylabels_acr[i])
                if i == 2:
                    ax2[i].set_xlabel('Hours Ellapsed Since Last Measurement Update')
            if with_plot_mce_best:
                ax2[i].plot(ts_pred, es_acr_mce_pred[:,i], 'b')
                ax2[i].plot(ts_pred, sigma_scale*sigs_acr_mce_pred[:,i], 'r')
                ax2[i].plot(ts_pred, -sigma_scale*sigs_acr_mce_pred[:,i], 'r')
            if with_plot_mce_avg:
                ax2[i].plot(ts_pred, es_acr_mce_avg_pred[:,i], color='brown')
                ax2[i].plot(ts_pred, sigma_scale*sigs_acr_mce_avg_pred[:,i], 'm')
                ax2[i].plot(ts_pred, -sigma_scale*sigs_acr_mce_avg_pred[:,i], 'm')
            ax2[i].plot(ts_pred, es_acr_kf_pred[:,i], 'g')
            ax2[i].plot(ts_pred, sigma_scale*sigs_acr_kf_pred[:,i], color='orange')
            ax2[i].plot(ts_pred, -sigma_scale*sigs_acr_kf_pred[:,i], color='orange')

        '''
        # Plot Log caresian
        for i in range(3):
            if day == days_ensemble:
                ax3[i].set_ylabel(ylabels_log_cart[i])
                if i == 2:
                    ax3[i].set_xlabel('Hours Ellapsed Since Last Measurement Update')
            ax3[i].plot(ts_pred, np.log10(np.abs(es_kf_pred[:,i])), 'g')
            ax3[i].plot(ts_pred, np.log10(np.abs(sigma_scale*sigs_kf_pred[:,i])), color='orange')
            ax3[i].plot(ts_pred, np.log10(np.abs(es_mce_pred[:,i])), 'b')
            ax3[i].plot(ts_pred, np.log10(np.abs(sigma_scale*sigs_mce_pred[:,i])), 'r')
        
        # Plot Log ACR
        for i in range(3):
            if day == days_ensemble:
                ax4[i].set_ylabel(ylabels_log_acr[i])
                if i == 2:
                    ax4[i].set_xlabel('Hours Ellapsed Since Last Measurement Update')
            ax4[i].plot(ts_pred, np.log10(np.abs(es_acr_kf_pred[:,i])), 'g')
            ax4[i].plot(ts_pred, np.log10(np.abs(sigma_scale*sigs_acr_kf_pred[:,i])), color='orange')
            ax4[i].plot(ts_pred, np.log10(np.abs(es_acr_mce_pred[:,i])), 'r')
            ax4[i].plot(ts_pred, np.log10(np.abs(sigma_scale*sigs_acr_mce_pred[:,i])), 'r')
        '''

        # Plot Errors Only in Cartesian
        for i in range(3):
            if day == days_ensemble:
                ax5[i].set_ylabel(ylabels_cart[i])
                if i == 2:
                    ax5[i].set_xlabel(xlabel)
            if with_plot_mce_best:
                ax5[i].plot(ts_pred, es_mce_pred[:,i], 'b')
            if with_plot_mce_avg:
                ax5[i].plot(ts_pred, es_mce_avg_pred[:,i], color='brown')
            ax5[i].plot(ts_pred, es_kf_pred[:,i], 'g')

        
        # Plot Errors Only in ACR
        for i in range(3):
            if day == days_ensemble:
                ax6[i].set_ylabel(ylabels_acr[i])
                if i == 2:
                    ax6[i].set_xlabel(xlabel)
            if with_plot_mce_best:
                ax6[i].plot(ts_pred, es_acr_mce_pred[:,i], 'b')
            if with_plot_mce_avg:
                ax6[i].plot(ts_pred, es_acr_mce_avg_pred[:,i], color='brown')
            ax6[i].plot(ts_pred, es_acr_kf_pred[:,i], 'g')
        
        # Plot Z-Scores Cartesian -- need the sigmas to be 1-sig
        scale_sigs = 1.0 / _sigma_scale
        for i in range(3):
            if day == days_ensemble:
                ax7[i].set_ylabel(ylabels_zscore[i])
                if i == 2:
                    ax7[i].set_xlabel(xlabel)
            if with_plot_mce_best:
                ax7[i].plot(ts_pred, es_mce_pred[:,i]/(sigs_mce_pred[:,i]*scale_sigs), 'b')
            if with_plot_mce_avg:
                ax7[i].plot(ts_pred, es_mce_avg_pred[:,i]/(sigs_mce_avg_pred[:,i]*scale_sigs), color='brown')
            ax7[i].plot(ts_pred, es_kf_pred[:,i]/(sigs_kf_pred[:,i]*scale_sigs), 'g')
        
        # Plot Z-Scores ACR
        scale_sigs = 1.0 / _sigma_scale
        for i in range(3):
            if day == days_ensemble:
                ax8[i].set_ylabel(ylabels_zscore_acr[i])
                if i == 2:
                    ax8[i].set_xlabel(xlabel)
            if with_plot_mce_best:
                ax8[i].plot(ts_pred, es_acr_mce_pred[:,i]/(sigs_acr_mce_pred[:,i]*scale_sigs), 'b')
            if with_plot_mce_avg:
                ax8[i].plot(ts_pred, es_acr_mce_avg_pred[:,i]/(sigs_acr_mce_avg_pred[:,i]*scale_sigs), color='brown')
            ax8[i].plot(ts_pred, es_acr_kf_pred[:,i]/(sigs_acr_kf_pred[:,i]*scale_sigs), 'g')

    plt.show()
    foobar = 2

def store_daily_plots(ensemble_sub_dir_name):
    # Form ensemble_fpath
    ensemble_root_dir = gmat_data_dir + "pred_def_ensemble/"
    ensemble_sub_dir_name = ensemble_sub_dir_name if ensemble_sub_dir_name[-1] == '/' else ensemble_sub_dir_name + "/"
    ensemble_fpath = ensemble_root_dir + ensemble_sub_dir_name

    # Load run settings
    fpath_run_dic_pickle = ensemble_fpath + "run_dic.pickle"
    with open(fpath_run_dic_pickle, 'rb') as handle:
        run_dic = pickle.load(handle)
    
    daily_plots_root_dir = ensemble_fpath + "daily_plots/"
    if not os.path.isdir(daily_plots_root_dir):
        os.mkdir(daily_plots_root_dir) 

    days_ensemble = run_dic['days_ensemble']
    sigma_scale = run_dic['sigma_scale']
    for day in range(1,days_ensemble+1):
        # Create day subdirectory
        daily_plot_subdir = daily_plots_root_dir + "day{}/".format(day)
        day_est_errors_fpath = daily_plot_subdir + "est_errors.png"
        day_cart_pred_errors_fpath = daily_plot_subdir + "cart_pred_errors.png"
        day_acr_pred_errors_fpath = daily_plot_subdir + "acr_pred_errors.png"

        if not os.path.isdir(daily_plot_subdir):
            os.mkdir(daily_plot_subdir) 

        # Load filter run data 
        day_filt_fname = "day" + str(day) + "_filt.pickle"
        pickle_fpath = ensemble_fpath + day_filt_fname
        with open(pickle_fpath, 'rb') as handle:
            tstart_idx_gps_msmts, day_t0, \
            ts, xs_kf, Ps_kf, xs_kf_gmat, Ps_kf_gmat, \
            xs_mce, Ps_mce, xs_mce_avg, Ps_mce_avg, \
            xs_sm_gmat, Ps_sm_gmat, Cd_nominal = pickle.load(handle)
        
        # Plot filter run data
        plot_ekf_mce_sigma_bound_and_errors(
            ts, 
            xs_kf, Ps_kf, 
            xs_kf_gmat, Ps_kf_gmat, 
            xs_mce, Ps_mce, 
            xs_mce_avg, Ps_mce_avg, 
            xs_sm_gmat, Ps_sm_gmat, day_est_errors_fpath)
        
        # Load prediction run data 
        day_filt_fname = "day" + str(day) + "_pred.pickle"
        pickle_fpath = ensemble_fpath + day_filt_fname
        with open(pickle_fpath, 'rb') as handle:
            ts_pred, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, \
            es_mce_avg_pred, sigs_mce_avg_pred, es_acr_kf_pred, sigs_acr_kf_pred, \
            es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred = pickle.load(handle)
        
        # Plot prediction run data
        plot_propagated_filter_estimates_and_compare_to_smoother_ephemeris(
                ts_pred, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, es_mce_avg_pred, sigs_mce_avg_pred,
                es_acr_kf_pred, sigs_acr_kf_pred, es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred, 
                sigma_scale, day_cart_pred_errors_fpath, day_acr_pred_errors_fpath)

def view_selected_daily_plot(ensemble_sub_dir_name, day):
    # Form ensemble_fpath
    ensemble_root_dir = gmat_data_dir + "pred_def_ensemble/"
    ensemble_sub_dir_name = ensemble_sub_dir_name if ensemble_sub_dir_name[-1] == '/' else ensemble_sub_dir_name + "/"
    ensemble_fpath = ensemble_root_dir + ensemble_sub_dir_name

    # Load run settings
    fpath_run_dic_pickle = ensemble_fpath + "run_dic.pickle"
    with open(fpath_run_dic_pickle, 'rb') as handle:
        run_dic = pickle.load(handle)
    sigma_scale = run_dic['sigma_scale']
    # Load filter run data 
    day_filt_fname = "day" + str(day) + "_filt.pickle"
    pickle_fpath = ensemble_fpath + day_filt_fname
    with open(pickle_fpath, 'rb') as handle:
        tstart_idx_gps_msmts, day_t0, \
        ts, xs_kf, Ps_kf, xs_kf_gmat, Ps_kf_gmat, \
        xs_mce, Ps_mce, xs_mce_avg, Ps_mce_avg, \
        xs_sm_gmat, Ps_sm_gmat, Cd_nominal = pickle.load(handle)
    # Plot filter run data
    plot_ekf_mce_sigma_bound_and_errors(
        ts, 
        xs_kf, Ps_kf, 
        xs_kf_gmat, Ps_kf_gmat, 
        xs_mce, Ps_mce, 
        xs_mce_avg, Ps_mce_avg, 
        xs_sm_gmat, Ps_sm_gmat)
    # Load prediction run data 
    day_filt_fname = "day" + str(day) + "_pred.pickle"
    pickle_fpath = ensemble_fpath + day_filt_fname
    with open(pickle_fpath, 'rb') as handle:
        ts_pred, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, \
        es_mce_avg_pred, sigs_mce_avg_pred, es_acr_kf_pred, sigs_acr_kf_pred, \
        es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred = pickle.load(handle)
    # Plot prediction run data
    plot_propagated_filter_estimates_and_compare_to_smoother_ephemeris(
            ts_pred, es_kf_pred, sigs_kf_pred, es_mce_pred, sigs_mce_pred, es_mce_avg_pred, sigs_mce_avg_pred,
            es_acr_kf_pred, sigs_acr_kf_pred, es_acr_mce_pred, sigs_acr_mce_pred, es_acr_mce_avg_pred, sigs_acr_mce_avg_pred, 
            sigma_scale)


if __name__ == "__main__":
    #test_time_convert()
    #test_STM_Gamma_approximations()
    #test_leo_STM_Gamma_approximations()
    
    # Input File Paths
    gps_path =  gmat_data_dir + "G_navsol_from_gseqprt_2023043_2023137_thinned_stitched.txt.navsol"
    restart_ekf_path = gmat_data_dir + "Sat_GLAST_Restart_ekf.csv" #"Sat_GLAST_Restart_20230212_094850.csv"
    restart_smoother_path = gmat_data_dir + "Sat_GLAST_Smooth_Def_ephem.e"
    
    # USER: Choose Your Subdirectory Name...
    # Either ... Create a new Subdirectory 
    #        ... or Append Data to an Existing Subdirectory
    ensemble_sub_dir_name = "run_foo" #"run_beta_small30" # your choice

    # Run Ensemble
    #'''
    with_plotting = False
    ensemble_fpath = test_pred_def_overlap_3day(gps_path, restart_ekf_path, restart_smoother_path, ensemble_sub_dir_name, with_plotting=with_plotting)
    print("Data Was Stored In Directory: ", ensemble_fpath)
    #'''

    #'''
    # Plot Ensemble
    override_sigma = None # set to [1,2,3...] if you have a hankering to see plots with various sigmas besides the set sigma
    with_plot_mce_best = False
    with_plot_mce_avg = True
    plot_pred_def_overlap_3day(ensemble_sub_dir_name, override_sigma, with_plot_mce_best, with_plot_mce_avg)
    foobar=2
    #'''

    # Store Daily Plots
    store_daily_plots(ensemble_sub_dir_name)

    # View Single Plot on Chosen Day
    #view_selected_daily_plot(ensemble_sub_dir_name, day = 3)
    foo = 2



#--------Nat Stuff----------#
# Nice Debug Examples
'''
# Functions which were used to debug certain issues
def test_gps_transformation(t0, x0, gps_msmts, inputted_glast_data):
    test_len = 10
    #mod = gmat.Moderator.Instance()
    #ss = mod.GetDefaultSolarSystem()
    #earth = ss.GetBody('Earth')
    #eop_file_path = gmat_data_dir + "eop_file.txt"
    #earth.SetField('EopFileName', eop_file_path)
    fixedState = gmat.Rvector6()
    ecf = gmat.Construct("CoordinateSystem","ECF","Earth","BodyFixed")
    eci = gmat.Construct("CoordinateSystem","ECI","Earth","MJ2000Eq")
    csConverter = gmat.CoordinateConverter()
    #gmat.Initialize()

    fermi_t0 = "{} {} {} {}:{}:{}.{}".format(t0.day, MonthDic2[t0.month], t0.year,t0.hour,t0.minute,t0.second, int(t0.microsecond/1000) )
    fermi_dt = 65.0
    fermi_x0 = x0.copy() 
    fermi_gps_std_dev = 7.5 / 1e3 # m -> km
    fermiSat = FermiSatelliteModel(fermi_t0, fermi_x0, fermi_dt)
    fermiSat.create_model(with_jacchia=True, with_SRP=True)
    x_next = fermiSat.step()

    t_idx = 0
    glast_times = inputted_glast_data[0]
    glast_means = inputted_glast_data[1]
    for i in range(test_len):
        # Find the closest glast vector. These time stamps should be identical
        gps_msmt = gps_msmts[i]
        date = gps_msmt[0]
        msmt = gps_msmt[1]
        t_idx = find_time_match(date, glast_times, start_idx = t_idx)
        cmp_vec = glast_means[t_idx]
        #cmp_vec = x_next.copy()

        # Create Transformed GPS Msmt in Earth MJ2000Eq Coordinates
        epoch = gmat.UtcDate(date.year, date.month, date.day, date.hour, date.minute, date.second + date.microsecond / 1e6)
        #_rvec = [*(msmt/1000), 0,0,0]
        _rvec = list(cmp_vec[0:6])
        rvec = gmat.Rvector6( *_rvec )
        
        # Foo Loop
        #for i in range(0,1000,5):
        #    time_a1mjd = epoch.ToA1Mjd() + 1.1574065865715965e-08*i
        #    csConverter.Convert(time_a1mjd, rvec, ecf, fixedState, eci)
        #    outvec = np.array([fixedState[0], fixedState[1], fixedState[2], fixedState[3], fixedState[4], fixedState[5]])
        #    # Print difference between transformation in meters
        #    diff_vec = (outvec[0:3] - cmp_vec[0:3]) * 1000
        #    print("Difference: i= ",i, ": ", diff_vec)
        #time_dic_tai = time_convert(date, "UTC", "TAI")
        time_dic_a1 = time_convert(date, "UTC", "A1")
        time_a1mjd = time_dic_a1["out_mjd"] #time_a1mjd = epoch.ToA1Mjd()
        #time_a1mjd += 1.1572410585358739e-08*210
        #csConverter.Convert(time_a1mjd, rvec, ecf, fixedState, eci)
        csConverter.Convert(time_a1mjd, rvec, eci, fixedState, ecf)



        outvec = np.array([fixedState[0], fixedState[1], fixedState[2], fixedState[3], fixedState[4], fixedState[5]])
        # Print difference between transformation in meters
        #diff_vec = (outvec[0:3] - cmp_vec[0:3]) * 1000
        diff_vec = (outvec[0:3]* 1000 - msmt[0:3])
        print("Difference: ", diff_vec)
    return outvec

def test_time_convert():
    time_in = datetime(2023, 2, 12, 0, 30, 29, 1000) #"11 Feb 2023 23:49:00.000" # 29987.521168993055
    type_in = "UTC"
    type_out = "A1"
    time_dic = time_convert(time_in, type_in, type_out)
    print("In {} Greg: {}".format(type_in, time_dic["in_greg"]) )
    print("In {} MJD: {}".format(type_in, time_dic["in_mjd"]) )
    print("Out {} Greg: {}".format(type_out, time_dic["out_greg"]) )
    print("In {} MJD: {}".format(type_out, time_dic["out_mjd"]) )

def test_single_gps_msmt():
    mod = gmat.Moderator.Instance()
    ss = mod.GetDefaultSolarSystem()
    earth = ss.GetBody('Earth')
    eop_file_path = gmat_data_dir + "eop_file.txt"
    earth.SetField('EopFileName', eop_file_path)
    in_pos_vec = [4656.8449747241457, 4230.0931453206676, 2811.4539689390904, 0, 0, 0]
    time_a1mjd = 29987.492789749787
    in_gps_vec = gmat.Rvector6(*in_pos_vec)
    ecf = gmat.Construct("CoordinateSystem","ECF","Earth","BodyFixed")
    eci = gmat.Construct("CoordinateSystem","ECI","Earth","MJ2000Eq")
    csConverter = gmat.CoordinateConverter()
    gmat.Initialize()
    tmp_gps_vec = gmat.Rvector6()
    csConverter.Convert(time_a1mjd, in_gps_vec, eci, tmp_gps_vec, ecf)
    out_gps_vec = np.array([tmp_gps_vec[0],tmp_gps_vec[1],tmp_gps_vec[2],tmp_gps_vec[3],tmp_gps_vec[4],tmp_gps_vec[5]])
    print("Computed Value: ", out_gps_vec)
    known_gps_vec = np.array([-705.78640593524074, -6246.8126277142010, 2821.9433196988944, 0, 0, 0])
    print("True Value: ", known_gps_vec)
    print("Residual", 1000*(known_gps_vec[0:3] - out_gps_vec[0:3]) )

def test_STM_approximations():
    #Dynamic System 
    A = np.array([0,1, -10, -.3]).reshape((2,2))
    I = np.eye(2)
    Phi0 = I.copy() 
    dt = 1.0
    fPhi = lambda Phi : A @ Phi
    Phi_ps1 = I + A * dt
    Phi_ps2 = I + A * dt + A @ A * dt**2 / 2
    Phi_ps3 = I + A * dt + A @ A * dt**2 / 2 + A @ A @ A * dt**3 / 6
    Phi_ps4 = I + A * dt + A @ A * dt**2 / 2 + A @ A @ A * dt**3 / 6 + A @ A @ A @ A * dt**4 / 24
    Phi_ps5 = I + A * dt + A @ A * dt**2 / 2 + A @ A @ A * dt**3 / 6 + A @ A @ A @ A * dt**4 / 24 + A @ A @ A @ A @ A * dt**5 / 120
    Phi_rk4 = ce.runge_kutta4(fPhi, Phi0, dt)
    
    # Analytic State Transition Matrix:
    t = dt
    b = A[0,1]
    c = A[1,0]
    d = A[1,1]
    s1 = ( d + (d**2 + 4*b*c + 0j)**0.5 ) / 2
    s2 = ( d - (d**2 + 4*b*c + 0j)**0.5 ) / 2
    Phi_analytic = np.zeros((2,2))
    Phi_analytic[0,0] = (s1-d)/(s1-s2) * np.exp(s1*t) + (d-s2)/(s1-s2)*np.exp(s2*t)
    Phi_analytic[0,1] = b/(s1-s2) * (np.exp(s1*t) - np.exp(s2*t))
    Phi_analytic[1,0] = c/(s1-s2) * (np.exp(s1*t) - np.exp(s2*t))
    Phi_analytic[1,1] = s1/(s1-s2) * np.exp(s1*t) + s2/(s2-s1)*np.exp(s2*t)
    
    print("For DT={}, the STM is approximated as: ".format(dt))
    print("Diff PS1: ", np.sum( (Phi_analytic-Phi_ps1)**2 ) )
    print("Diff PS2: ", np.sum( (Phi_analytic-Phi_ps2)**2 ) )
    print("Diff PS3: ", np.sum( (Phi_analytic-Phi_ps3)**2 ) )
    print("Diff PS4: ", np.sum( (Phi_analytic-Phi_ps4)**2 ) )
    print("Diff PS5: ", np.sum( (Phi_analytic-Phi_ps5)**2 ) )
    print("Diff RK4: ", np.sum( (Phi_analytic-Phi_rk4)**2 ) )

    # Repeat again
    sub_steps = 5
    dt_sub = dt / sub_steps
    Phi_ps1 = Phi0.copy()
    Phi_ps2 = Phi0.copy()
    Phi_ps3 = Phi0.copy()
    Phi_ps4 = Phi0.copy()
    Phi_ps5 = Phi0.copy()
    Phi_rk4 = Phi0.copy()
    for _ in range(sub_steps):
        Phi_ps1 = (I + A * dt_sub) @ Phi_ps1
        Phi_ps2 = (I + A * dt_sub + A @ A * dt_sub**2 / 2) @ Phi_ps2
        Phi_ps3 = (I + A * dt_sub + A @ A * dt_sub**2 / 2 + A @ A @ A * dt_sub**3 / 6) @ Phi_ps3
        Phi_ps4 = (I + A * dt_sub + A @ A * dt_sub**2 / 2 + A @ A @ A * dt_sub**3 / 6 + A @ A @ A @ A * dt_sub**4 / 24) @ Phi_ps4
        Phi_ps5 = (I + A * dt_sub + A @ A * dt_sub**2 / 2 + A @ A @ A * dt_sub**3 / 6 + A @ A @ A @ A * dt_sub**4 / 24 + A @ A @ A @ A @ A * dt_sub**5 / 120) @ Phi_ps5
        Phi_rk4 = ce.runge_kutta4(fPhi, Phi_rk4, dt_sub)
    print("\nFor DT={}, using {} substeps each DT_SUB={}, the STM is approximated as: ".format(dt, sub_steps, dt_sub))
    print("Diff PS1_IT: ", np.sum( (Phi_analytic-Phi_ps1)**2 ) )
    print("Diff PS2_IT: ", np.sum( (Phi_analytic-Phi_ps2)**2 ) )
    print("Diff PS3_IT: ", np.sum( (Phi_analytic-Phi_ps3)**2 ) )
    print("Diff PS4_IT: ", np.sum( (Phi_analytic-Phi_ps4)**2 ) )
    print("Diff PS5_IT: ", np.sum( (Phi_analytic-Phi_ps5)**2 ) )
    print("Diff RK4_IT: ", np.sum( (Phi_analytic-Phi_rk4)**2 ) )
    foobar = 2

def Gamma_SubSec3(A, B, dt, taylor_order=4):
    num_subs = 3
    sub_dt = dt / num_subs
    tmp = np.zeros((2,2))
    pow_Jack = [np.linalg.matrix_power(A, i) for i in range(taylor_order+1)]
    for i in range(taylor_order+1):
        for j in range(taylor_order+1):
            for k in range(taylor_order+1):
                tmp += ( pow_Jack[i] @ pow_Jack[j] @ pow_Jack[k] ) * sub_dt**(i+j+k+1) \
                    / ( math.factorial(i) * math.factorial(j) * math.factorial(k) * (i+j+k+1) )
    return num_subs * tmp @ B 

def Gamma_SubSec4(A, B, dt, taylor_order=5):
    num_subs = 4
    sub_dt = dt / num_subs
    tmp = np.zeros((2,2))
    pow_Jack = [np.linalg.matrix_power(A, i) for i in range(taylor_order+1)]
    for i in range(taylor_order+1):
        for j in range(taylor_order+1):
            for k in range(taylor_order+1):
                for l in range(taylor_order+1):
                    tmp += ( pow_Jack[i] @ pow_Jack[j] @ pow_Jack[k] @ pow_Jack[l] ) * sub_dt**(i+j+k+l+1) \
                        / ( math.factorial(i) * math.factorial(j) * math.factorial(k) * math.factorial(l) * (i+j+k+l+1) )
    return num_subs * tmp @ B     

def test_STM_Gamma_approximations():
    # Dynamic System 
    A = np.array([0,1, -10, -.3]).reshape((2,2))
    B = np.array([0,1.0])
    I = np.eye(2)
    Phi0 = I.copy() 
    dt = 1.0
    fPhi = lambda Phi : A @ Phi
    Phi_ps1 = I + A * dt
    Phi_ps2 = I + A * dt + A @ A * dt**2 / 2
    Phi_ps3 = I + A * dt + A @ A * dt**2 / 2 + A @ A @ A * dt**3 / 6
    Phi_ps4 = I + A * dt + A @ A * dt**2 / 2 + A @ A @ A * dt**3 / 6 + A @ A @ A @ A * dt**4 / 24
    Phi_ps5 = I + A * dt + A @ A * dt**2 / 2 + A @ A @ A * dt**3 / 6 + A @ A @ A @ A * dt**4 / 24 + A @ A @ A @ A @ A * dt**5 / 120
    Phi_rk4 = ce.runge_kutta4(fPhi, Phi0, dt)
    
    # Analytic State Transition Matrix:
    t = dt
    b = A[0,1]
    c = A[1,0]
    d = A[1,1]
    s1 = ( d + (d**2 + 4*b*c + 0j)**0.5 ) / 2
    s2 = ( d - (d**2 + 4*b*c + 0j)**0.5 ) / 2
    Phi_analytic = np.zeros((2,2))
    Phi_analytic[0,0] = (s1-d)/(s1-s2) * np.exp(s1*t) + (d-s2)/(s1-s2)*np.exp(s2*t)
    Phi_analytic[0,1] = b/(s1-s2) * (np.exp(s1*t) - np.exp(s2*t))
    Phi_analytic[1,0] = c/(s1-s2) * (np.exp(s1*t) - np.exp(s2*t))
    Phi_analytic[1,1] = s1/(s1-s2) * np.exp(s1*t) + s2/(s2-s1)*np.exp(s2*t)

    #Gamma_analytic = integral( Phi_analytic @ B , dt)
    Gamma_analytic = np.zeros(2)
    Gamma_analytic[0] = b/(s1-s2) * 1/s1 * (np.exp(s1*t)-1) - b/(s1-s2) * 1/s2 * (np.exp(s2*t)-1)
    Gamma_analytic[1] = s1/(s1-s2) * 1/s1 * (np.exp(s1*t)-1) + s2/(s2-s1) * 1/s2 * (np.exp(s2*t)-1)
    
    print("For DT={}, the STM is approximated as: ".format(dt))
    print("Diff PS1: ", np.sum( (Phi_analytic-Phi_ps1)**2 ) )
    print("Diff PS2: ", np.sum( (Phi_analytic-Phi_ps2)**2 ) )
    print("Diff PS3: ", np.sum( (Phi_analytic-Phi_ps3)**2 ) )
    print("Diff PS4: ", np.sum( (Phi_analytic-Phi_ps4)**2 ) )
    print("Diff PS5: ", np.sum( (Phi_analytic-Phi_ps5)**2 ) )
    print("Diff RK4: ", np.sum( (Phi_analytic-Phi_rk4)**2 ) )

    Gamk1 = np.sum([np.linalg.matrix_power(A, i) * dt**(i+1) / math.factorial(i+1) for i in range(1+1) ], axis = 0) @ B
    Gamk2 = np.sum([np.linalg.matrix_power(A, i) * dt**(i+1) / math.factorial(i+1) for i in range(2+1) ], axis = 0) @ B
    Gamk3 = np.sum([np.linalg.matrix_power(A, i) * dt**(i+1) / math.factorial(i+1) for i in range(3+1) ], axis = 0) @ B
    Gamk4 = np.sum([np.linalg.matrix_power(A, i) * dt**(i+1) / math.factorial(i+1) for i in range(4+1) ], axis = 0) @ B
    Gamk5 = np.sum([np.linalg.matrix_power(A, i) * dt**(i+1) / math.factorial(i+1) for i in range(5+1) ], axis = 0) @ B
    Gam_Pieces3 = Gamma_SubSec3(A, B, dt, taylor_order=5)
    Gam_Pieces4 = Gamma_SubSec4(A, B, dt, taylor_order=5)

    print("Diff Gam1: ", np.sum( (Gamma_analytic-Gamk1)**2 ) )
    print("Diff Gam2: ", np.sum( (Gamma_analytic-Gamk2)**2 ) )
    print("Diff Gam3: ", np.sum( (Gamma_analytic-Gamk3)**2 ) )
    print("Diff Gam4: ", np.sum( (Gamma_analytic-Gamk4)**2 ) )
    print("Diff Gamk5: ", np.sum( (Gamma_analytic-Gamk5)**2 ) )
    print("Diff GamPieces3: ", np.sum( (Gamma_analytic - Gam_Pieces3 )**2 ) )
    print("Diff GamPieces4: ", np.sum( (Gamma_analytic - Gam_Pieces4 )**2 ) )

    # Repeat again
    sub_steps = 5
    dt_sub = dt / sub_steps
    Phi_ps1 = Phi0.copy()
    Phi_ps2 = Phi0.copy()
    Phi_ps3 = Phi0.copy()
    Phi_ps4 = Phi0.copy()
    Phi_ps5 = Phi0.copy()
    Phi_rk4 = Phi0.copy()
    for _ in range(sub_steps):
        Phi_ps1 = (I + A * dt_sub) @ Phi_ps1
        Phi_ps2 = (I + A * dt_sub + A @ A * dt_sub**2 / 2) @ Phi_ps2
        Phi_ps3 = (I + A * dt_sub + A @ A * dt_sub**2 / 2 + A @ A @ A * dt_sub**3 / 6) @ Phi_ps3
        Phi_ps4 = (I + A * dt_sub + A @ A * dt_sub**2 / 2 + A @ A @ A * dt_sub**3 / 6 + A @ A @ A @ A * dt_sub**4 / 24) @ Phi_ps4
        Phi_ps5 = (I + A * dt_sub + A @ A * dt_sub**2 / 2 + A @ A @ A * dt_sub**3 / 6 + A @ A @ A @ A * dt_sub**4 / 24 + A @ A @ A @ A @ A * dt_sub**5 / 120) @ Phi_ps5
        Phi_rk4 = ce.runge_kutta4(fPhi, Phi_rk4, dt_sub)

    print("\nFor DT={}, using {} substeps each DT_SUB={}, the STM is approximated as: ".format(dt, sub_steps, dt_sub))
    print("Diff PS1_IT: ", np.sum( (Phi_analytic-Phi_ps1)**2 ) )
    print("Diff PS2_IT: ", np.sum( (Phi_analytic-Phi_ps2)**2 ) )
    print("Diff PS3_IT: ", np.sum( (Phi_analytic-Phi_ps3)**2 ) )
    print("Diff PS4_IT: ", np.sum( (Phi_analytic-Phi_ps4)**2 ) )
    print("Diff PS5_IT: ", np.sum( (Phi_analytic-Phi_ps5)**2 ) )
    print("Diff RK4_IT: ", np.sum( (Phi_analytic-Phi_rk4)**2 ) )
    foobar = 2

def test_leo_STM_Gamma_approximations():
    x0 = np.array([ 4.99624529e+03,  3.87794646e+03,  2.73604324e+03, -5.02809357e+00, 5.57592134e+00,  1.26986117e+00, -7.07238385e-05])
    dt = 30
    t0 = datetime(2023, 2, 11, 23, 47, 55)
    tau_Cd = 21600 / np.log(2)
    fermSat = FermiSatelliteModel(t0, x0[0:6].copy(), dt, gmat_print=True)
    fermSat.create_model()
    fermSat.set_solve_for(field="Cd", dist="sas", scale=0.0013, tau=tau_Cd, alpha=1.3)
    fermSat.reset_state(x0.copy(), 0)

    # Lets Form Phi and Gamma in One Step 
    taylor_order = 4
    Jack = fermSat.get_jacobian_matrix()
    Jack[3:6,6] *= 1000 # km -> m
    Phik = fermSat.get_transition_matrix(taylor_order=taylor_order, use_units_km=False)
    Gamc = np.zeros(7)
    Gamc[6] = 1.0
    Gamk = np.sum([np.linalg.matrix_power(Jack, i) * dt**(i+1) / math.factorial(i+1) for i in range(taylor_order+1) ], axis = 0) @ Gamc

    # Lets Form Phi and Gamma in num_substeps
    fermSat.reset_state(x0.copy(), 0)
    num_substeps = 3 # This cant change
    sub_dt = dt / num_substeps
    fermSat.dt = sub_dt
    Phi2k = np.eye(7)
    Jacks = []
    for i in range(num_substeps):
        Jack = fermSat.get_jacobian_matrix()
        Jack[3:6, 6] *= 1000
        Jacks.append(Jack)
        Phi2k = fermSat.get_transition_matrix(taylor_order=taylor_order, use_units_km=False) @ Phi2k
        fermSat.step()
    tmp = np.zeros((7,7))
    pow_Jack1 = [np.linalg.matrix_power(Jacks[0], i) for i in range(taylor_order+1)]
    pow_Jack2 = [np.linalg.matrix_power(Jacks[1], i) for i in range(taylor_order+1)]
    pow_Jack3 = [np.linalg.matrix_power(Jacks[2], i) for i in range(taylor_order+1)]
    for i in range(taylor_order+1):
        for j in range(taylor_order+1):
            for k in range(taylor_order+1):
                tmp += ( pow_Jack1[i] @ pow_Jack2[j] @ pow_Jack3[k] ) * sub_dt**(i+j+k+1) \
                    / ( math.factorial(i) * math.factorial(j) * math.factorial(k) * (i+j+k+1) )
    Gam2k = tmp @ Gamc
    foobar=2
'''
# End Nice Debug Examples