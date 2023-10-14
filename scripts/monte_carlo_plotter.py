#/usr/bin/env python3
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 

with_heavy_tailed_performance = False # This can be added if necessary
with_gauss_performance = False
with_cauchy_offset = False

print("MONTE_CARLO_PLOTTER.PY USAGE:\n\tONOPTIONAL ARG: Gaussian or Heavy Tailed Noise Simulation (g/h)\nEnter l=path_to_data to change log folder" )
print("MONTE_CARLO_PLOTTER.PY INFO: \n\tUses a arithmetic mean to calculate averaged performance at each time_step for gaussian noise. \n\tUses the log of the geometric mean to calculate averaged performance at each time_step for heavy tailed noise.")
print("Example: python3 monte_carlo_plotter.py h -> This calculates the monte carlo performance for heavy tailed noise (h)")
print("Example: python3 monte_carlo_plotter.py g l=../log/monte_carlo/foo-> This calculates the monte carlo performance for gaussian noise (g) and changes the log directory (l) to ../log/monte_carlo/foo (from ../log/monte_carlo/)")
print("Note: the g or h option must be provided, but not both!")

log_dirs = ["../log/monte_carlo/"]

for i in range(1, len(sys.argv)):
    if sys.argv[i] == "g":
        with_gauss_performance = True
    elif sys.argv[i] == "h":
        with_heavy_tailed_performance = True
    elif(sys.argv[i].startswith("l")):
        log_dirs = sys.argv[i].split("=")[1].split(";")
        for log_dir,count in zip(log_dirs,range(len(log_dirs))):
            if log_dir[-1] != "/":
                log_dir += "/"
            log_dirs[count] = log_dir
            print("New monte carlo log directory path is ", log_dir)
    elif sys.argv[i].startswith("d"):
        with_cauchy_offset = int(sys.argv[i].split("=")[1])
    else:
        print("UNKNOWN CMD LINE OPTION ", sys.argv[i], ". SEE ABOVE USAGE FOR HELP!")
        exit(1)
assert(with_gauss_performance ^ with_heavy_tailed_performance)
for log_dir in log_dirs:
    assert(os.path.isdir(log_dir))

#plt.rcParams['figure.dpi'] = 300
#plt.rcParams['savefig.dpi'] = 300
leg_loc = 1
leg_prop={'size': 12}

colors = ["b","g","o", "p", "y", "k"]
line_types = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]

f_means = "cond_means.txt"
#_covars = "cond_covars.txt"
#f_err_means = "cerr_cond_means.txt"
#f_err_covars = "cerr_cond_covars.txt"
#f_err_norm_factors = "cerr_norm_factors.txt"
#f_norm_factor = "norm_factors.txt" # Semilog plot 

f_true_states = "true_states.txt"
#f_msmts = "msmts.txt"
#f_msmt_noises = "msmt_noises.txt"
#f_proc_noises = "proc_noises.txt"
f_kf_cond_means = "kf_cond_means.txt"
#f_kf_cond_covars = "kf_cond_covars.txt"
#f_kf_residuals = "kf_residuals.txt"

#f_xbars = "x_bars.txt"
#f_residuals = "residuals.txt"
#f_full_residuals = "full_residuals.txt"

# Reads and parses window data 
def load_window_data(f_win_data):
    file = open(f_win_data, 'r')
    lines = file.readlines()
    return np.array([[float(f) for f in line.split(":")[1].split(" ")] for line in lines])

# Reads and parses non-window data
def load_data(f_data):
    file = open(f_data, 'r')
    lines = file.readlines()
    return np.array([[float(f) for f in line.split(" ")] for line in lines])

FONT_SIZE = 16
fig = plt.figure(1)
for log_dir,count in zip(log_dirs,range(len(log_dirs))):
    means = []
    #covars = []
    kf_means = []
    #kf_covars = []
    true_states = []
    dirs = next(os.walk(log_dir))[1]
    dirs = sorted(dirs, key=lambda x : int(x.split("_")[1]) ) 
    for d in dirs:
        path2data = log_dir + d + "/" + f_true_states
        true_states.append( np.loadtxt( path2data) )
        path2data = log_dir + d + "/" + f_kf_cond_means
        kf_means.append( np.loadtxt( path2data) )
        path2data = log_dir + d + "/" + f_means
        means.append(load_window_data(path2data))
    print("Loaded ", len(dirs), "MC Trials!")
    # Log of the geometric mean of the squared error [over all trials in Monte Carlo Sim] (only variances -- not covariances)
    if(with_heavy_tailed_performance):
        # Number of steps 
        T = means[0].shape[0]
        # Delay / offset checking
        if (T + with_cauchy_offset != kf_means[0].shape[0]):
            print("T + with_cauchy_offset != kf_means[0].shape[0]. Please use the d=offset option!")
            print("T=", T)
            print("with_cauchy_offset=",with_cauchy_offset)
            print("kf_means[0].shape[0]=", kf_means[0].shape[0])
            assert( T + with_cauchy_offset == kf_means[0].shape[0] )
        # Number of states
        n = means[0].shape[1]
        # Number of trials
        n_mc = len(true_states) + 0.0
        # Data as 3-D tensors
        true_states = np.array(true_states)
        kf_means = np.array(kf_means)
        means = np.array(means)
        # Log of the squared errors 
        lse_kf = np.log((true_states - kf_means)**2)
        lse_cauchy = np.log((true_states[:,with_cauchy_offset:,:] - means)**2)
        # Performances of the Sum of the Log of the squared errors for Kalman and Cauchy 
        P_kf = 1.0 / n_mc * np.sum(lse_kf, axis = 0)
        P_cauchy = 1.0 / n_mc * np.sum(lse_cauchy, axis = 0)

        P_kf = np.sqrt(np.exp(P_kf))
        P_cauchy = np.sqrt(np.exp(P_cauchy))

        # Plotting the performances of Kalman and Cauchy 
        plt.suptitle("Monte Carlo (Geom-Mean Error) Performance", fontsize=FONT_SIZE+4)
        for i in range(n):
            if len(log_dirs) == 1:
                label_ce = "Cauchy Est." if i == 0 else ""
                label_kf = "Kalman Filter" if i== 0 else ""
            if len(log_dirs) == 3:
                # This is to be removed:
                if count == 0:
                    label_ce = "Cauchy (" + r"$\alpha=1.7$" + ")" if i == 0 else ""
                    label_kf = "Kalman (" + r"$\alpha=1.7$" + ")" if i== 0 else ""
                if count == 1:
                    label_ce = "Cauchy (" + r"$\alpha=1.5$" + ")" if i == 0 else ""
                    label_kf = "Kalman (" + r"$\alpha=1.5$" + ")" if i== 0 else ""
                if count == 2:
                    label_ce = "Cauchy (" + r"$\alpha=1.3$" + ")" if i == 0 else ""
                    label_kf = "Kalman (" + r"$\alpha=1.3$" + ")" if i== 0 else ""
            if len(log_dirs) == 5:
                # This is to be removed:
                if count == 0:
                    label_ce = "Cauchy (" + r"$\alpha=2.0$" + ")" if i == 0 else ""
                    label_kf = "Kalman (" + r"$\alpha=2.0$" + ")" if i== 0 else ""
                if count == 1:
                    label_ce = "Cauchy (" + r"$\alpha=1.7$" + ")" if i == 0 else ""
                    label_kf = "Kalman (" + r"$\alpha=1.7$" + ")" if i== 0 else ""
                if count == 2:
                    label_ce = "Cauchy (" + r"$\alpha=1.5$" + ")" if i == 0 else ""
                    label_kf = "Kalman (" + r"$\alpha=1.5$" + ")" if i== 0 else ""
                if count == 3:
                    label_ce = "Cauchy (" + r"$\alpha=1.3$" + ")" if i == 0 else ""
                    label_kf = "Kalman (" + r"$\alpha=1.3$" + ")" if i== 0 else ""
                if count == 4:
                    label_ce = "Cauchy (" + r"$\alpha=1.0$" + ")" if i == 0 else ""
                    label_kf = "Kalman (" + r"$\alpha=1.0$" + ")" if i== 0 else ""
            plt.subplot(str(n) + "1" + str(i+1))
            #if i == 0:
            #    plt.ylabel("Position (ft)", fontsize=FONT_SIZE)
            #if i == 1:
            #    plt.ylabel("Velocity (ft/sec)", fontsize=FONT_SIZE)
            #if i == 2:
            #    plt.ylabel("Acceleration (ft/sec^2)", fontsize=FONT_SIZE)
            plt.plot(np.arange(T), P_cauchy[:,i], colors[0], linestyle=line_types[count], label=label_ce) #colors[2*count]
            plt.plot(np.arange(T), P_kf[with_cauchy_offset:,i], colors[1], linestyle=line_types[count], label=label_kf) #colors[2*count+1]
            plt.xticks(fontsize=FONT_SIZE)
            plt.yticks(fontsize=FONT_SIZE)
        plt.xlabel("Time-Step k", fontsize=FONT_SIZE)
        print("Data in directory ", log_dir, " uses line-type format:", "'"+str(line_types[count])+"'")

    # Arithmetic mean of the squared error [over all trials in Monte Carlo Sim] (only variances -- not covariances)
    else:
        # Number of steps 
        T = means[0].shape[0]
        # Number of states
        n = means[0].shape[1]
        # Number of trials
        n_mc = len(true_states) + 0.0
        # Data as 3-D tensors
        true_states = np.array(true_states)
        kf_means = np.array(kf_means)
        means = np.array(means)
        # mean of errors (e_bar)
        me_kf = np.mean(true_states - kf_means, axis = 0)
        me_cauchy = np.mean(true_states - means, axis = 0)
        # errors (e)
        e_kf = true_states - kf_means
        e_cauchy = true_states - means
        # squared difference of errors (e-ebar)
        sde_kf = (e_kf - me_kf)**2
        sde_cauchy = (e_cauchy - me_cauchy)**2
        # (Unbiased) Variances for Kalman and Cauchy 
        P_kf = 1.0 / (n_mc-1.0) * np.sum(sde_kf, axis = 0)
        P_cauchy = 1.0 / (n_mc-1.0) * np.sum(sde_cauchy, axis = 0)

        # Plotting the performances of Kalman and Cauchy 
        plt.suptitle("Monte Carlo (Arithmetic Mean) Performance")
        for i in range(n):
            label_ce = "Cauchy Est" if i == 0 else ""
            label_kf = "Kalman Filter" if i== 0 else ""
            plt.subplot(str(n) + "1" + str(i+1))
            plt.plot(np.arange(T), P_cauchy[:,i], colors[0]+line_types[count], label=label_ce)#colors[2*count]
            plt.plot(np.arange(T), P_kf[:,i], colors[1]+line_types[count], label=label_kf)#colors[2*count+1]
        print("Data in directory ", log_dir, " uses line-type format: '", line_types[count], "'")

leg = fig.legend(loc=leg_loc, prop={'size' : FONT_SIZE}) #leg_prop)
leg.draggable(state=True)
plt.show()