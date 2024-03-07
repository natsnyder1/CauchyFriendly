import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg',force=True)
import os
import pickle
file_dir = os.path.dirname(os.path.abspath(__file__))

# Reads and parses window data 
def load_window_data(f_win_data):
    file = open(f_win_data, 'r')
    lines = file.readlines()
    return np.array([[float(f) for f in line.split(":")[1].split(" ")] for line in lines])

def view_manually_mct_run():
    # MC LOG DIR to look into
    log_base_dir = file_dir + "/../log/homing_missile/monte_carlo/w7_bs5_sas13/mct5/"

    kf_means = np.genfromtxt(log_base_dir + "kf_cond_means.txt")
    kf_covars = np.genfromtxt(log_base_dir + "kf_cond_covars.txt")
    kf_controls = np.genfromtxt(log_base_dir + "kf_controls.txt")
    kf_msmts = np.genfromtxt(log_base_dir + "kf_with_controller_msmts.txt")
    kf_trues = np.genfromtxt(log_base_dir + "kf_with_controller_true_states.txt")

    ce_means = load_window_data(log_base_dir + "cond_means.txt")
    ce_covars = load_window_data(log_base_dir + "cond_covars.txt")
    ce_controls = np.genfromtxt(log_base_dir + "cauchy_controls.txt")
    ce_msmts = np.genfromtxt(log_base_dir + "cauchy_with_controller_msmts.txt")
    ce_trues = np.genfromtxt(log_base_dir + "cauchy_with_controller_true_states.txt")

    proc_noises = np.genfromtxt(log_base_dir + "proc_noises.txt")
    msmt_noises = np.genfromtxt(log_base_dir + "msmt_noises.txt")


    T = kf_means.shape[0]
    n = kf_means.shape[1]
    Ts = np.arange(T)
    kf_covars = kf_covars.reshape((T,n,n))
    ce_covars = ce_covars.reshape((T-1,n,n))

    kf_one_sig = np.array([np.sqrt(np.diag(P)) for P in kf_covars])
    kf_errs = kf_trues - kf_means 

    ce_one_sig = np.array([np.sqrt(np.diag(P)) for P in ce_covars])
    ce_errs = ce_trues[1:] - ce_means 

    plt.figure()
    plt.subplot(311)
    plt.title("State Errs. (pos/ve/acc)")
    plt.plot(Ts, kf_errs[:,0], 'g--')
    plt.plot(Ts, kf_one_sig[:,0], 'm--')
    plt.plot(Ts, -kf_one_sig[:,0], 'm--')
    plt.plot(Ts[1:], ce_errs[:,0], 'b')
    plt.plot(Ts[1:], ce_one_sig[:,0], 'r')
    plt.plot(Ts[1:], -ce_one_sig[:,0], 'r')
    plt.subplot(312)
    plt.plot(Ts, kf_errs[:,1], 'g--')
    plt.plot(Ts, kf_one_sig[:,1], 'm--')
    plt.plot(Ts, -kf_one_sig[:,1], 'm--')
    plt.plot(Ts[1:], ce_errs[:,1], 'b')
    plt.plot(Ts[1:], ce_one_sig[:,1], 'r')
    plt.plot(Ts[1:], -ce_one_sig[:,1], 'r')
    plt.subplot(313)
    plt.plot(Ts, kf_errs[:,2], 'g--')
    plt.plot(Ts, kf_one_sig[:,2], 'm--')
    plt.plot(Ts, -kf_one_sig[:,2], 'm--')
    plt.plot(Ts[1:], ce_errs[:,2], 'b')
    plt.plot(Ts[1:], ce_one_sig[:,2], 'r')
    plt.plot(Ts[1:], -ce_one_sig[:,2], 'r')

    plt.figure()
    plt.subplot(311)
    plt.title("True States (g/b) vs Est States (g--/b--) (pos/vel/acc) (KF/CE)")
    plt.plot(Ts, kf_trues[:,0], 'g')
    plt.plot(Ts, kf_means[:,0], 'g--')
    plt.plot(Ts, ce_trues[:,0], 'b')
    plt.plot(Ts[1:], ce_means[:,0], 'b--')
    plt.subplot(312)
    plt.plot(Ts, kf_trues[:,1], 'g')
    plt.plot(Ts, kf_means[:,1], 'g--')
    plt.plot(Ts, ce_trues[:,1], 'b')
    plt.plot(Ts[1:], ce_means[:,1], 'b--')
    plt.subplot(313)
    plt.plot(Ts, kf_trues[:,2], 'g')
    plt.plot(Ts, kf_means[:,2], 'g--')
    plt.plot(Ts, ce_trues[:,2], 'b')
    plt.plot(Ts[1:], ce_means[:,2], 'b--')

    plt.figure()
    plt.subplot(311)
    plt.title("Controls u(x_est) (g/b), Msmts (g/b) / Msmt Noises (m--), Proc Noises (k) (KF/CE)")
    plt.plot(Ts[:-1], kf_controls, 'g')
    plt.plot(Ts[:-1], ce_controls, 'b')
    plt.subplot(312)
    plt.plot(Ts, kf_msmts, 'g')
    plt.plot(Ts, ce_msmts, 'b')
    plt.plot(Ts, msmt_noises, 'm--')
    plt.subplot(313)
    plt.plot(Ts[:-1], proc_noises, 'k')

    plt.show()
    foobar=2

def draw_mc_plot(gsd_Pces, gsd_Pkfs, mde_ces, mde_kfs, label_ces, label_kfs):
    # Plot 
    solid = 'solid'
    dashed = 'dashed'
    dashdot = 'dashdot'
    densly_dashdotted = (0,(3,1,1,1))
    dotted = 'dotted'
    loosely_dotted = (0,(1,10))
    #line_types = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
    line_types = [solid, dashed, dashdot, densly_dashdotted, dotted, loosely_dotted]
    len_kf = gsd_Pkfs[0].shape[0]
    len_ce = gsd_Pces[0].shape[0]
    delay = len_kf - len_ce 
    assert( (delay == 0) or (delay == 1))
    Ts = np.arange(len_kf)
    fig = plt.figure()
    count = 0
    for P_cauchy, P_kf, label_ce, label_kf in zip(gsd_Pces, gsd_Pkfs, label_ces, label_kfs):
        plt.subplot(311)
        if count == 0:
            plt.title("Geometric Standard Deviation Cauchy Vs Kalman")
        plt.plot(Ts[delay:], P_cauchy[:, 0], 'b', linestyle=line_types[count], label=label_ce) 
        plt.plot(Ts[delay:], P_kf[delay:, 0], 'g', linestyle=line_types[count], label=label_kf) 
        if(count == 0):
            plt.ylabel("Pos. Error")
        plt.subplot(312)
        plt.plot(Ts[delay:], P_cauchy[:, 1], 'b', linestyle=line_types[count], label="") 
        plt.plot(Ts[delay:], P_kf[delay:, 1], 'g', linestyle=line_types[count], label="") 
        if(count == 0):
            plt.ylabel("Vel. Error")
        plt.subplot(313)
        plt.plot(Ts[delay:], P_cauchy[:, 2], 'b', linestyle=line_types[count], label="") 
        plt.plot(Ts[delay:], P_kf[delay:, 2], 'g', linestyle=line_types[count], label="") 
        if(count == 0):
            plt.ylabel("Accel. Error")
        count += 1
    leg = fig.legend(loc=1, prop={'size' : 16}) #leg_prop)
    leg.set_draggable(state=True)
    plt.show()

    count = 1
    bin_low = -350
    bin_high = 350
    bin_width = 5
    num_bins = int( (bin_high - bin_low) / bin_width ) + 1
    fig = plt.figure()
    fig2 = plt.figure()
    #plt.tight_layout()
    bins = np.linspace(bin_low,bin_high, num_bins)
    fig.suptitle("Histograms of Target Miss Distance as a Function of SAS Noise Severity\n(Bin Width={} feet)".format(bin_width), fontsize = 24)
    fig2.suptitle("CDF of Target Miss Distance as a Function of SAS Noise Severity\n(Bin Width={} feet)".format(bin_width), fontsize = 24)
    for mde_ce, mde_kf, label_ce, label_kf in zip(mde_ces, mde_kfs, label_ces, label_kfs):
        ax = fig.add_subplot(3,2,count) #plt.subplot(3,2,count)
        ax2 = fig2.add_subplot(3,2,count)
        ax.set_ylabel("Bin Counts", fontsize = 20)
        ax2.set_ylabel("CDF", fontsize = 20)
        if count > 4:
            ax.set_xlabel("Miss Distance (Feet)", fontsize = 20)
            ax2.set_xlabel("Miss Distance (Feet)", fontsize = 20)
        ax.set_title("Alpha=" + label_ce[3:], fontsize = 20)
        ax2.set_title("Alpha=" + label_ce[3:], fontsize = 20)

        ce_hist = ax.hist(mde_ce, bins, alpha=0.5, label=label_ce)
        ce_hits = ce_hist[0]
        ce_bins = ce_hist[1]
        kf_hist = ax.hist(mde_kf, bins, alpha=0.5, label=label_kf)
        kf_hits = kf_hist[0]
        ax.legend(loc=1, prop={'size' : 12}, fontsize = 18) #leg_prop)

        n = ce_hits.size
        assert(n % 2 == 0)
        cdf_bins = ce_bins[n//2:]
        ce_cdf = np.cumsum(np.flip(ce_hits[0:n//2]) + ce_hits[n//2:])
        kf_cdf = np.cumsum(np.flip(kf_hits[0:n//2]) + kf_hits[n//2:])
        ce_cdf /= ce_cdf[-1]
        kf_cdf /= kf_cdf[-1]
        ax2.plot(cdf_bins[1:], ce_cdf, 'b', label=label_ce)
        ax2.plot(cdf_bins[1:], kf_cdf, color='orange', label=label_kf)
        x_major_ticks = np.arange(0, 341, 20)
        #x_minor_ticks = np.arange(0, 101, 5)
        y_major_ticks = np.arange(0, 1.1, .10)
        #y_minor_ticks = np.arange(0, 1.1, .5)
        ax2.set_xticks(x_major_ticks)
        #ax2.set_xticks(x_minor_ticks, minor=True)
        ax2.set_yticks(y_major_ticks)
        #ax2.set_yticks(y_minor_ticks, minor=True)
        # And a corresponding grid
        ax2.grid(which='both')
        ax2.legend(loc=1, prop={'size' : 12}) #leg_prop)
        
        count += 1
    plt.show()
    '''
    for mde_ce, mde_kf, label_ce, label_kf in zip(mde_ces, mde_kfs, label_ces, label_kfs):
        bins = np.linspace(-250,250, 51)
        fig = plt.figure()
        plt.title("Miss Distance Error Histogram")
        plt.hist(mde_ce, bins, alpha=0.5, label=label_ce)
        plt.hist(mde_kf, bins, alpha=0.5, label=label_kf)
        plt.ylabel("Bin Counts")
        plt.xlabel("Miss Distance")
        leg = fig.legend(loc=1, prop={'size' : 16}) #leg_prop)
        leg.set_draggable(state=True)
        plt.show()
    '''
    foobar = 2

def extract_monte_carlo_dir_data_and_pickle(mc_dir):
    alphas_dir = next(os.walk(mc_dir))[1]

    list_alpha_labels = []
    list_ce_trues = [] 
    list_kf_trues = [] 
    list_ce_means = [] 
    list_kf_means = [] 

    for alpha_dir in alphas_dir:
        log_base_dir = mc_dir + alpha_dir + "/"
        print("Reading MC Trials from ", log_base_dir)
        dirs = next(os.walk(log_base_dir))[1]
        kf_means = [] 
        kf_trues = [] 
        ce_means = [] 
        ce_trues = []
        count = 0
        for d in dirs:
            path = log_base_dir + d
            kf_mean = np.genfromtxt(path + "/kf_cond_means.txt")
            kf_true = np.genfromtxt(path + "/kf_with_controller_true_states.txt")
            ce_mean = load_window_data(path + "/cond_means.txt")
            ce_true = np.genfromtxt(path + "/cauchy_with_controller_true_states.txt")
            kf_means.append(kf_mean)
            kf_trues.append(kf_true)
            ce_means.append(ce_mean)
            ce_trues.append(ce_true)
            count += 1
            if(count % 200) == 0:
                print("Read ", count, " directories!")
        alpha_val = float(log_base_dir[-3:-1])/10
        print("Loaded ", len(ce_means), " monte carlo trials for alpha=", alpha_val)
        # Data as 3-D tensors
        ce_trues = np.array(ce_trues)
        kf_trues = np.array(kf_trues)
        ce_means = np.array(ce_means)
        kf_means = np.array(kf_means)
        # Add numpy data to lists
        list_alpha_labels.append(alpha_val)
        list_ce_trues.append(ce_trues)
        list_kf_trues.append(kf_trues)
        list_ce_means.append(ce_means)
        list_kf_means.append(kf_means)
    
    # Sort into descending order (alpha=2,1.7,..etc)
    zipped = zip(list_alpha_labels, list_ce_trues, list_kf_trues, list_ce_means, list_kf_means)
    szip = list(reversed( sorted(zipped, key = lambda x : x[0]) ))
    list_alpha_labels = [] 
    list_ce_trues = []
    list_kf_trues = []
    list_ce_means = [] 
    list_kf_means = []
    for sz in szip:
        list_alpha_labels.append(sz[0])
        list_ce_trues.append(sz[1])
        list_kf_trues.append(sz[2])
        list_ce_means.append(sz[3])
        list_kf_means.append(sz[4])

    # Pickle Results 
    pickle_path = mc_dir + "raw_numpy_data.pickle"
    with open(pickle_path, 'wb') as handle:
        pickle.dump((list_alpha_labels, list_ce_trues, list_kf_trues, list_ce_means, list_kf_means), handle)
        print("Results data has been stored at", pickle_path, ". These can be loaded in directly for quick plotting!")
    
    return list_alpha_labels, list_ce_trues, list_kf_trues, list_ce_means, list_kf_means

def plot_monte_carlo_averages(mc_dir):
    raw_dat = "raw_numpy_data.pickle"
    
    # Data is stored in tuple 
    # as (list_alpha_labels, list_ce_trues, list_kf_trues, list_ce_means, list_kf_means)
    if(os.path.isfile(mc_dir + raw_dat)):
        print("LOADING NUMPY DATA FROM PICKLED FILE: ", mc_dir + raw_dat)
        with open(mc_dir + raw_dat, 'rb') as handle:
            list_alpha_labels, list_ce_trues, list_kf_trues, list_ce_means, list_kf_means \
                = pickle.load(handle)
    else:
        print("LOADING NUMPY DATA FROM MONTE CARLO DIRECTORY", mc_dir, "AND PICKLING TO: ", mc_dir + raw_dat)
        list_alpha_labels, list_ce_trues, list_kf_trues, list_ce_means, list_kf_means \
            = extract_monte_carlo_dir_data_and_pickle(mc_dir)
    
    gsd_Pkfs = [] 
    gsd_Pces = [] 
    mde_ces = []
    mde_kfs = [] 
    label_ces = [] 
    label_kfs = [] 


    for alpha_val, ce_trues, kf_trues, ce_means, kf_means in zip(list_alpha_labels, list_ce_trues, list_kf_trues, list_ce_means, list_kf_means):
        len_kf = kf_means[0].shape[0]
        len_ce = ce_means[0].shape[0]
        delay = len_kf - len_ce 
        
        # Number of trials
        n_mc = ce_means.shape[0] + 0.0
        
        # Log of the squared errors 
        lse_kf = np.log((kf_trues - kf_means)**2)
        lse_cauchy = np.log((ce_trues[:,delay:,:] - ce_means)**2)
        # Performances of the Sum of the Log of the squared errors for Kalman and Cauchy 
        P_kf = 1.0 / n_mc * np.sum(lse_kf, axis = 0)
        P_cauchy = 1.0 / n_mc * np.sum(lse_cauchy, axis = 0)
        P_kf = np.sqrt(np.exp(P_kf))
        P_cauchy = np.sqrt(np.exp(P_cauchy))

        label_ce = "ce-"+str(alpha_val)
        label_kf = "kf-"+str(alpha_val)

        # Used For Histograms of the True Terminal State Miss Distance Error
        mde_ce = ce_trues[:,-1,0].reshape(-1)
        mde_kf = kf_trues[:,-1,0].reshape(-1)

        # Store Results
        gsd_Pces.append(P_cauchy.copy())
        gsd_Pkfs.append(P_kf.copy())
        mde_ces.append(mde_ce.copy())
        mde_kfs.append(mde_kf.copy())
        label_ces.append(label_ce)
        label_kfs.append(label_kf)

        print("Finished Processing alpha={} ({} Trials)".format(alpha_val, mde_ce.size) )

    # Pickle Results 
    pickle_path = mc_dir + "results.pickle"
    with open(pickle_path, 'wb') as handle:
        pickle.dump((gsd_Pces, gsd_Pkfs, mde_ces, mde_kfs, label_ces, label_kfs), handle)
        print("Results data has been stored at", pickle_path, ". These can be loaded in directly for quick plotting!")
    
    # Plot 
    draw_mc_plot(gsd_Pces, gsd_Pkfs, mde_ces, mde_kfs, label_ces, label_kfs)
    foobar = 2

def plot_cached_monte_carlo_averages(mc_dir):
    pickle_path = mc_dir + "results.pickle"
    with open(pickle_path, 'rb') as handle:
       gsd_Pces, gsd_Pkfs, mde_ces, mde_kfs, label_ces, label_kfs = pickle.load(handle)
    draw_mc_plot(gsd_Pces, gsd_Pkfs, mde_ces, mde_kfs, label_ces, label_kfs)

if __name__ == "__main__":
    mc_dir = file_dir + "/../log/homing_missile/monte_carlo/"
    #view_mct_run()
    plot_monte_carlo_averages(mc_dir)
    #plot_cached_monte_carlo_averages(mc_dir)
