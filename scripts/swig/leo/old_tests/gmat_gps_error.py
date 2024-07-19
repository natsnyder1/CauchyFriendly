import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg',force=True)
import os, sys 
#import cauchy_estimator as ce 
import math 

file_dir = os.path.dirname(os.path.abspath(__file__))
gmat_root_dir = '/home/natsubuntu/Desktop/SysControl/estimation/CauchyCPU/CauchyEst_Nat/GMAT/application'
gmat_bin_dir = gmat_root_dir + '/bin'
gmat_api_dir = gmat_root_dir + '/api'
gmat_py_dir = gmat_bin_dir + '/gmatpy'
gmat_startup_file = gmat_bin_dir + '/api_startup_file.txt'
sys.path.append(gmat_bin_dir)
sys.path.append(gmat_api_dir)
sys.path.append(gmat_py_dir)
if os.path.exists(gmat_startup_file):
   import gmat_py as gmat
   gmat.Setup(gmat_startup_file)
from datetime import datetime 
import pickle 

MonthDic = {"Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6, "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12}
MonthDic2 = {v:k for k,v in MonthDic.items()}

# 0.) Time conversion function using GMAT
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

# 1.) Load in GPS Data and time stamps 
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

# 2.) Load in warm start ephemeris
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
        
# 2.) Find the a-priori covariance closest to the first GPS reading (returns state before the first GPS reading)
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

# Convert any combination of UTC, TAI, A1 (GREG/JULIAN)
def find_time_match(t_gps, ts_glast, start_idx = 0):
    count = 0
    for t_glast in ts_glast[start_idx:]:
        dt = t_gps - t_glast
        if (dt.days == 0) and (dt.seconds == 0):
            return count + start_idx
        count +=1
    print("No GLast Time Found to Be Same!")
    assert(False)

# Debugging the GPS transformation required
def test_gps_transformation(t0, x0, gps_msmts, inputted_glast_data):
    test_len = 10
    mod = gmat.Moderator.Instance()
    ss = mod.GetDefaultSolarSystem()
    earth = ss.GetBody('Earth')
    eop_file_path = file_dir + "/gmat_data/gps_2_11_23/eop_file.txt"
    earth.SetField('EopFileName', eop_file_path)
    fixedState = gmat.Rvector6()
    ecf = gmat.Construct("CoordinateSystem","ECF","Earth","BodyFixed")
    eci = gmat.Construct("CoordinateSystem","ECI","Earth","MJ2000Eq")
    csConverter = gmat.CoordinateConverter()
    gmat.Initialize()

    #fermi_t0 = "{} {} {} {}:{}:{}.{}".format(t0.day, MonthDic2[t0.month], t0.year,t0.hour,t0.minute,t0.second, int(t0.microsecond/1000) )
    #fermi_dt = 65.0
    #fermi_x0 = x0.copy() 
    #fermi_gps_std_dev = 7.5 / 1e3 # m -> km
    #fermiSat = FermiSatelliteModel(fermi_t0, fermi_x0, fermi_dt, fermi_gps_std_dev)
    #fermiSat.create_model(with_jacchia=True, with_SRP=True)
    #x_next = fermiSat.step()

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
        time_dic_tai = time_convert(date, "UTC", "TAI")
        time_dic_a1 = time_convert(date, "UTC", "A1")
        time_a1mjd = time_dic_a1["out_mjd"] #time_a1mjd = epoch.ToA1Mjd()
        #time_a1mjd += 1.1572410585358739e-08*210 # time bias correction for now?
        csConverter.Convert(time_a1mjd, rvec, eci, fixedState, ecf)
        #csConverter.Convert(time_a1mjd, rvec, eci, fixedState, ecf)

        outvec = np.array([fixedState[0], fixedState[1], fixedState[2], fixedState[3], fixedState[4], fixedState[5]])
        # Print difference between transformation in meters
        #diff_vec = (outvec[0:3] - cmp_vec[0:3]) * 1000
        diff_vec = (outvec[0:3]* 1000 - msmt[0:3])

        print("Difference: ", diff_vec)
    return outvec

# Test functions and find GPS residual
def test_all():
    gps_path = file_dir + "/gmat_data/gps_2_11_23/G_navsol_from_gseqprt_2023043_2023137_thinned_stitched.txt.navsol"
    restart_path = file_dir + "/gmat_data/gps_2_11_23/Sat_GLAST_Restart_20230212_094850.csv"
    gps_msmts = load_gps_from_txt(gps_path)
    inputted_glast = load_glast_file(restart_path)
    t0, x0, P0, labels_x0, labels_P0 = find_restart_point(restart_path, gps_msmts[0][0])
    #run_fermi_kalman_filter_and_smoother(gps_msmts, t0, x0[:7], P0[:7,:7])
    test_gps_transformation(t0, x0, gps_msmts, inputted_glast)
    print("Thats all folks!")

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
    eop_file_path = file_dir + "/gmat_data/gps_2_11_23/eop_file.txt"
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

if __name__ == "__main__":
    test_all()
    #test_time_convert()
    #test_single_gps_msmt()
