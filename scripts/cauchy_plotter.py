#/usr/bin/env python3

from cmath import log
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 

log_dir = "../log/"
f_true_states = "true_states.txt"
f_means = "cond_means.txt"
f_covars = "cond_covars.txt"

#f_norm_factor = "norm_factors.txt" # Semilog plot 
f_msmts = "msmts.txt"
f_msmt_noises = "msmt_noises.txt"
f_proc_noises = "proc_noises.txt"

f_err_means = "cerr_cond_means.txt"
f_err_covars = "cerr_cond_covars.txt"
f_err_norm_factors = "cerr_norm_factors.txt"

f_kf_cond_means = "kf_cond_means.txt"
f_kf_cond_covars = "kf_cond_covars.txt"

f_kf_residuals = "kf_residuals.txt"
f_residuals = "residuals.txt"

print(sys.argv)
print("CAUCHY PLOTTER HELPER: Command Line Options Below") 
print("Enter 'kf' (kalman_filter) to plot against kalman filter")
print("Enter 'e' (extended) to plot the residuals if the system logged was nonlinear")
print("Enter l=path_to_data to change log folder")
print("Enter d=# steps to plot the cauchy estimator with a delay of # steps at beginning")
print("Enter 'p' to partially plot a (failed) run of the cauchy estimator....i.e: if you had an error")
print("Enter 's=2' or 's=3' to scale the covariances to +/- [2,3]-sigma value. Default is s=1")

with_kf = False
with_extended = False 
with_cauchy_delay = False
with_partial_plot=False
with_no_win_logging=True
scale=1
cauchy_offset = 0
plot_kf_key = "kf"
plot_extended_key = "e"
partial_plot_key = "p"

for i in range(1, len(sys.argv)):
    if(sys.argv[i] == plot_kf_key):
        with_kf = True
    elif(sys.argv[i] == plot_extended_key):
        with_extended = True
    elif(sys.argv[i] == "nwl"):
        with_no_win_logging = False
    elif(sys.argv[i] == partial_plot_key):
        with_partial_plot = True
    elif(sys.argv[i].startswith("s=")):
        scale = int(sys.argv[i].split("=")[1])
    elif(sys.argv[i].startswith("l=")):
        log_dir = sys.argv[i].split("=")[1]
        if log_dir[-1] != "/":
            log_dir += "/"
        print("New log directory path is ", log_dir)
    elif(sys.argv[i].startswith("d=")):
        with_cauchy_delay = int(sys.argv[i].split("=")[1])
    else:
        print("UNKNOWN CMD LINE OPTION ", sys.argv[i], ". SEE ABOVE USAGE FOR HELP!")
        exit(1)

# Reads and parses window data 
def load_window_data(f_win_data):
    file = open(f_win_data, 'r')
    lines = file.readlines()
    return np.array([[float(f) for f in line.split(":")[1].split(" ")] for line in lines])

# Reads and parses Kalman Filter Data
def load_data(f_data):
    file = open(f_data, 'r')
    lines = file.readlines()
    return np.array([[float(f) for f in line.split(" ")] for line in lines])

# Load in the data points
means = load_window_data(log_dir + f_means) if with_no_win_logging else load_data(log_dir + f_means)
print("Means: ", means.shape)

covars = load_window_data(log_dir + f_covars) if with_no_win_logging else load_data(log_dir + f_covars)
print("Covars: ", covars.shape)
n = int(np.sqrt(covars.shape[1]))
covars = covars.reshape((covars.shape[0], n,n))
print("Covars after Reshaping: ", covars.shape)

true_states = np.loadtxt(log_dir + f_true_states)
if true_states.ndim == 1:
    true_states = true_states.reshape((true_states.size,1))
print("True States: ", true_states.shape)

msmts = np.loadtxt(log_dir + f_msmts)
msmt_noises = np.loadtxt(log_dir + f_msmt_noises)
proc_noises = np.loadtxt(log_dir + f_proc_noises)
if(msmts.ndim == 1):
    msmts = msmts.reshape((msmts.size,1))
if(proc_noises.ndim == 1):
    proc_noises = proc_noises.reshape((proc_noises.size,1))
if(msmt_noises.ndim == 1):
    msmt_noises = msmt_noises.reshape((msmt_noises.size,1))
print("Msmts: ", msmts.shape)
print("Msmt Noises: ", msmt_noises.shape)
print("Proc Noises: ", proc_noises.shape)

cerr_means = load_window_data(log_dir + f_err_means) if with_no_win_logging else load_data(log_dir + f_err_means)
print("Cerr Means: ", cerr_means.shape)
cerr_covars = load_window_data(log_dir + f_err_covars) if with_no_win_logging else load_data(log_dir + f_err_covars)
print("Cerr Covar: ", cerr_covars.shape)
cerr_norm_factor = load_window_data(log_dir + f_err_norm_factors) if with_no_win_logging else load_data(log_dir + f_err_norm_factors)
print("Cerr Means: ", cerr_norm_factor.shape)


kf_cond_means = None
kf_cond_covars = None
if(with_kf):
    kf_cond_means = load_data(log_dir + f_kf_cond_means)
    kf_cond_covars = load_data(log_dir + f_kf_cond_covars)
    kf_cond_covars = kf_cond_covars.reshape((kf_cond_covars.shape[0], n, n))

# Check array lengths, cauchy_delay, partial plot parameters
T = np.arange(0, msmts.shape[0])
cd = with_cauchy_delay 
#plot_len variable has been introduced so that runs which fail can still be partially plotted
if(not with_partial_plot and with_cauchy_delay):
    plot_len = covars.shape[0] + cd
    if(plot_len != T.size):
        print("[ERROR]: covars.shape[0] + with_cauchy_delay != T.size. You have mismatch in array lengths")
        print("Cauchy Covars size: ", covars.shape)
        print("with_cauchy_delay: ", with_cauchy_delay)
        print("T size: ", T.size)
        print("Please fix appropriately!")
        assert(False)
elif(with_partial_plot or with_cauchy_delay):
    plot_len = covars.shape[0] + cd
    if(plot_len > T.size):
        print("[ERROR]: covars.shape[0] + with_cauchy_delay > T.size. You have mismatch in array lengths")
        print("Cauchy Covars size: ", covars.shape)
        print("with_cauchy_delay: ", with_cauchy_delay)
        print("T size: ", T.size)
        print("Please fix appropriately!")
        assert(False)
else:
    if(covars.shape[0] + cd != T.size):
        print("[ERROR]: covars.shape[0] + with_cauchy_delay != T.size. You have mismatch in array lengths")
        print("Cauchy Covars size: ", covars.shape)
        print("with_cauchy_delay: ", with_cauchy_delay)
        print("T size: ", T.size)
        print("Please toggle on 'p' option for partial plotting or set 'd' to lag cauchy estimator appropriately")
        assert(False)
    plot_len = T.size


# 1.) Plot the true state history vs the conditional mean estimate  
# 2.) Plot the state error and one-sigma bound of the covariance 
# 3.) Plot the msmts, and the msmt and process noise 
# 4.) Plot the max complex error in the mean/covar and norm factor 
fig = plt.figure(1)
if with_kf:
    fig.suptitle("True States (r) vs Cauchy (b) vs Kalman (g--)")
else:
    fig.suptitle("True States (r) vs Cauchy Estimates (b)")
for i in range(covars.shape[1]):
    plt.subplot(str(n) + "1" + str(i+1))
    plt.plot(T[:plot_len], true_states[:plot_len,i], 'r')
    plt.plot(T[cd:plot_len], means[:,i], 'b')
    if with_kf:
        plt.plot(T[:plot_len], kf_cond_means[:plot_len,i], 'g--')

fig = plt.figure(2)
if with_kf:
    fig.suptitle("Cauchy 1-Sig (b/r) vs Kalman 1-Sig (g-/m-)")
else:
    fig.suptitle("State Error (b) vs One Sigma Bound (r)")
for i in range(covars.shape[1]):
    plt.subplot(str(n) + "1" + str(i+1))
    plt.plot(T[cd:plot_len], true_states[cd:plot_len,i] - means[:,i], 'b')
    plt.plot(T[cd:plot_len], scale*np.sqrt(covars[:,i,i]), 'r')
    plt.plot(T[cd:plot_len], -scale*np.sqrt(covars[:,i,i]), 'r')
    if with_kf:
        plt.plot(T[:plot_len], true_states[:plot_len,i] - kf_cond_means[:plot_len,i], 'g--')
        plt.plot(T[:plot_len], scale*np.sqrt(kf_cond_covars[:plot_len,i,i]), 'm--')
        plt.plot(T[:plot_len], -scale*np.sqrt(kf_cond_covars[:plot_len,i,i]), 'm--')

line_types = ['-', '--', '-.', ':', '-']
fig = plt.figure(3)
fig.suptitle("Msmts (m), Msmt Noise (g), Proc Noise (b)")
m = 3 #proc_noises.shape[1] + msmt_noises.shape[1] + msmts.shape[1]
count = 1
plt.subplot(str(m) + "1" + str(count))
for i in range(msmts.shape[1]):
    plt.plot(T[:plot_len], msmts[:plot_len,i], "m" + line_types[i])
count += 1
plt.subplot(str(m) + "1" + str(count))
for i in range(msmt_noises.shape[1]):
    plt.plot(T[:plot_len], msmt_noises[:plot_len,i], "g" + line_types[i])
count += 1
plt.subplot(str(m) + "1" + str(count))
for i in range(proc_noises.shape[1]):
    plt.plot(T[1:plot_len], proc_noises[:plot_len-1,i], "b" + line_types[i])

fig = plt.figure(4)
fig.suptitle("Complex Errors (mean,covar,norm factor) in Semi-Log")
plt.subplot(311)
plt.semilogy(T[cd:plot_len], cerr_means, 'g')
plt.subplot(312)
plt.semilogy(T[cd:plot_len], cerr_covars, 'g')
plt.subplot(313)
plt.semilogy(T[cd:plot_len], cerr_norm_factor, 'g')

residuals = None
kf_residuals = None 

#If the user specifies "e" (extended) on command line, plot residuals
#if they also specify "kf", plot residuals against the kalman filters

if(with_extended):
    residuals = load_window_data(log_dir + f_residuals)
    if(residuals.ndim == 1):
        residuals  = residuals.reshape((residuals.size,1))
    
    if(with_kf):
        kf_residuals = load_data(log_dir + f_kf_residuals)
        if(kf_residuals.ndim == 1):
            kf_residuals  = kf_residuals.reshape((kf_residuals.size,1))
    
    fig = plt.figure(5)
    if(with_kf):
        fig.suptitle("Residuals: ECE (blue) vs EKF (green)")
    else:
        fig.suptitle("Residuals: ECE")
    p = msmts.shape[1]
    for i in range(p):
        plt.subplot(str(p) + "1" + str(i+1))
        plt.plot(T[:plot_len], residuals, 'b')
        if(with_kf):
            plt.plot(T[1:], kf_residuals, 'g--')

plt.show()
