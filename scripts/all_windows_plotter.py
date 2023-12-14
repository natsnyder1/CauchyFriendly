#/usr/bin/env python3
from functools import partial
import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 

print("ALL WINDOWS PLOTTER HELPER:\nFIRST ARG: max window number, example: 7=[0,1,2,3,4,5,6]\nOPTIONAL: e=extended\nOPTIONAL: kf=with kalman filter compare\nOPTIONAL: a=average windows together\nOPTIONAL: c=compare to the full window data\nEnter l=path_to_data to change log folder")
print("Example: python3 window_plotter.py 7 e kf-> This plots the windows [0,...,6] with additional info for the nonlinear (extended) system, and compares to the ekf simulation")
print("Example: python3 window_plotter.py 3 kf-> This plots the [0,...,2]-nd windows with compare to the kf simulation\n\n\n")
print("Example: python3 window_plotter.py 7 e kf a c-> This plots the windows [0,...,6] with additional info for the nonlinear (extended) system, compares to the ekf simulation, averages the windows and overlays the full window plots")

log_dir="../log/"
window_sub_dir="windows/win"
window_log_dir = log_dir + window_sub_dir
assert(len(sys.argv) > 1)
win_high_num = int(sys.argv[1])
colors = ["tab:blue", "tab:orange", "hotpink", "yellow", "tab:purple", "tab:brown", "aquamarine", "tab:pink", "tab:gray", "tab:olive"]


with_extended = False # Show Residual plots for extended estimator
with_kf = False # Compare the Cauchy to the Kalman Filter
with_averaged = False # Average the Windows Together
with_full_window_compare = False # Compare the (possibly averaged) Cauchy Windows to the cauchy windowing technique
with_single_window_only = False # Only plot single window (whichever number is indicated above)

for i in range(2, len(sys.argv)):   
    if(sys.argv[i] == "e"):
        with_extended = True
    elif(sys.argv[i] == "kf"):
        with_kf = True
    elif(sys.argv[i] == "a"):
        with_averaged = True
    elif(sys.argv[i] == "c"):
        with_full_window_compare = True
    elif(sys.argv[i].startswith("l=")):
        log_dir = sys.argv[i].split("=")[1]
        if log_dir[-1] != "/":
            log_dir += "/"
        window_log_dir = log_dir + window_sub_dir
        print("New log directory path is ", log_dir)
    else:
        print("UNKNOWN CMD LINE OPTION ", sys.argv[i], ". SEE ABOVE USAGE FOR HELP!")
        exit(1)
assert(os.path.isdir(window_log_dir[:-4]))


if(not with_averaged):
    if(with_kf):
        if(with_extended):
            print("Plotting (extended) windows ", 0, "through ", win_high_num, " against the extended kalman filter")
        else:
            print("Plotting Windows ", 0, "through ", win_high_num, " against the kalman filter")
    else:
        if(with_extended):
            print("Plotting (extended) windows ", 0, "through ", win_high_num)
        else:
            print("Plotting Windows ", 0, "through ", win_high_num)
else:
    if(with_kf):
        if(with_extended):
            print("Plotting the average of (extended) windows ", 0, "through ", win_high_num, " against the extended kalman filter")
        else:
            print("Plotting the average of windows ", 0, "through ", win_high_num, " against the kalman filter")
    else:
        if(with_extended):
            print("Plotting the average of (extended) windows ", 0, "through ", win_high_num)
        else:
            print("Plotting the average of windows ", 0, "through ", win_high_num)

if with_full_window_compare:
    print("Full window compare also requested! Full windows are plotted in black!")

f_means = "cond_means.txt"
f_covars = "cond_covars.txt"
f_err_means = "cerr_cond_means.txt"
f_err_covars = "cerr_cond_covars.txt"
f_err_norm_factors = "cerr_norm_factors.txt"
#f_norm_factor = "norm_factors.txt" # Semilog plot 

f_true_states = "true_states.txt"
f_msmts = "msmts.txt"
f_msmt_noises = "msmt_noises.txt"
f_proc_noises = "proc_noises.txt"
f_kf_cond_means = "kf_cond_means.txt"
f_kf_cond_covars = "kf_cond_covars.txt"
f_kf_residuals = "kf_residuals.txt"

f_xbars = "x_bars.txt"
f_residuals = "residuals.txt"
f_full_residuals = "full_residuals.txt"

partial_run = True 
cd = 1

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
means = []
for i in range(win_high_num):
    means.append( load_data(window_log_dir + str(i) + "/" + f_means) )
n = int(means[0].shape[1])
sim_len = int(means[0].shape[0])
covars = []
for i in range(win_high_num):
    covars.append( load_data(window_log_dir + str(i) + "/" + f_covars) )
    covars[i] = covars[i].reshape((covars[i].shape[0], n,n))
    print("Covars of win", i, " after Reshaping: ", covars[i].shape)

cerr_means = []
cerr_covars = []
cerr_norm_factors = []
for i in range(win_high_num):
    cerr_means.append( load_data(window_log_dir + str(i) + "/" + f_err_means) ) 
    #print("Cerr Means: ", cerr_means[i].shape)
    cerr_covars.append( load_data(window_log_dir + str(i) + "/" + f_err_covars) )
    #print("Cerr Covar: ", cerr_covars[i].shape)
    cerr_norm_factors.append( load_data(window_log_dir + str(i) + "/" + f_err_norm_factors) )
    #print("Cerr Means: ", cerr_norm_factors[i].shape)

true_states = np.loadtxt(log_dir + f_true_states)
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


#If the user specifies "kf" on command line, plot kf as well
kf_cond_means = None
kf_cond_covars = None
kf_residuals = None
if(with_kf):
    kf_cond_means = load_data(log_dir + f_kf_cond_means)
    kf_cond_covars = load_data(log_dir + f_kf_cond_covars)
    kf_cond_covars = kf_cond_covars.reshape((kf_cond_covars.shape[0], n, n))
    if(with_extended):
        kf_residuals = load_data(log_dir + f_kf_residuals)
        if(kf_residuals.ndim == 1):
            kf_residuals = kf_residuals.reshape((kf_residuals.size,1))

# Load Extended Data
x_bars = [] 
resids = [] 
full_resids = []
if(with_extended):
    for i in range(win_high_num):    
        x_bars.append( load_data(window_log_dir + str(i) + "/" + f_xbars) )
        resids.append( load_data(window_log_dir + str(i) + "/" + f_residuals) )
        if(resids[i].ndim == 1):
            resids[i] = resids[i].reshape(resids[i].size, 1)
        full_resids.append( load_data(window_log_dir + str(i) + "/" + f_full_residuals) )
        if(full_resids[i].ndim == 1):
            full_resids[i] = full_resids[i].reshape(full_resids[i].size, 1)

# Load full windowed data
full_window_means = None 
full_window_covars = None
full_window_cerr_means = None 
full_window_cerr_covars = None 
full_window_cerr_norm_factors = None 
#full_window_x_bars = None 
#full_window_resids = None 
if with_full_window_compare:
    full_window_means = load_window_data(log_dir + f_means)
    full_window_covars = load_window_data(log_dir + f_covars)
    full_window_covars = full_window_covars.reshape((full_window_covars.shape[0], n, n))
    full_window_cerr_means = load_window_data(log_dir + f_err_means)
    full_window_cerr_covars = load_window_data(log_dir + f_err_covars)
    full_window_cerr_norm_factors = load_window_data(log_dir + f_err_norm_factors)

# Need to average all windows means and covariances together
if(with_averaged):
    for i in range(win_high_num-1):
        means[0][i+1:,:] += means[i+1]
        means[0][i,:] /= (i+1)
        covars[0][i+1:,:] += covars[i+1]
        covars[0][i,:] /= (i+1)
        cerr_means[0][i+1:] += cerr_means[i+1]
        cerr_means[0][i,:] /= (i+1)
        cerr_covars[0][i+1:] += cerr_covars[i+1]
        cerr_covars[0][i,:] /= (i+1)
        cerr_norm_factors[0][i+1:] += cerr_norm_factors[i+1]
        cerr_norm_factors[0][i,:] /= (i+1)
        if(with_extended):
            x_bars[0][i+1:,:] += x_bars[i+1]
            x_bars[0][i,:] /= (i+1)
            resids[0][i+1:,:] += resids[i+1]
            resids[0][i,:] /= (i+1)
            full_resids[0][i+1:,:] += full_resids[i+1]
            full_resids[0][i,:] /= (i+1)
    means[0][win_high_num-1:,:] /= win_high_num
    means = means[0:1]
    covars[0][win_high_num-1:,:] /= win_high_num
    covars = covars[0:1]
    cerr_means[0][win_high_num-1:,:] /= win_high_num
    cerr_means = cerr_means[0:1]
    cerr_covars[0][win_high_num-1:,:] /= win_high_num
    cerr_covars = cerr_covars[0:1]
    cerr_norm_factors[0][win_high_num-1:,:] /= win_high_num
    cerr_norm_factors = cerr_norm_factors[0:1]
    if(with_extended):
        x_bars[0][win_high_num-1:,:] /= win_high_num
        x_bars = x_bars[0:1]
        resids[0][win_high_num-1:,:] /= win_high_num
        resids = resids[0:1]
        full_resids[0][win_high_num-1:,:] /= win_high_num
        full_resids = full_resids[0:1]


# 1.) Plot the true state history vs the conditional mean estimate  
# 2.) Plot the state error and one-sigma bound of the covariance 
# 3.) Plot the msmts, and the msmt and process noise 
# 4.) Plot the max complex error in the mean/covar and norm factor 

T = np.arange(0, msmts.shape[0])
fig = plt.figure(1)
if with_kf:
    if with_averaged:
        fig.suptitle("True States (r) vs Kalman (g--) vs Averaged Cauchy Windows (b)")
    else:
        fig.suptitle("True States (r) vs Kalman (g--) vs Cauchy Windows (all other)")
else:
    if with_averaged:
        fig.suptitle("True States (r) vs Averaged Cauchy Windows (b)")
    else:
        fig.suptitle("True States (r) vs Cauchy Estimates (all other)")
for i in range(n):
    plt.subplot(str(n) + "1" + str(i+1))
    plt.plot(T, true_states[:,i], 'r')
    if with_kf:
        plt.plot(T, kf_cond_means[:,i], 'g--')
    for j in range(len(means)):
        plt.plot(T[j+cd:sim_len+cd], means[j][:,i], colors[j])
    if with_full_window_compare:
        plt.plot(T[cd:sim_len+cd], full_window_means[:, i], 'k')
        

fig = plt.figure(2)
if with_kf:
    if with_averaged:
        fig.suptitle("Averaged CE 1-Sig (solid/dotted) vs KF 1-sig (g--/m--)")
    else:
        fig.suptitle("CE 1-Sig (solid/dotted) vs KF 1-sig (g--/m--)")
else:
    if with_averaged:
        fig.suptitle("Averaged CE 1-Sig (solid/dotted)")
    else:
        fig.suptitle("CE 1-Sig (solid/dotted)")
for i in range(n):
    plt.subplot(str(n) + "1" + str(i+1))
    for j in range(len(covars)):
        plt.plot(T[j+cd:sim_len+cd], true_states[j+cd:sim_len+cd,i] - means[j][:,i], colors[j])
        plt.plot(T[j+cd:sim_len+cd], np.sqrt(covars[j][:,i,i]), colors[j], linestyle=":")
        plt.plot(T[j+cd:sim_len+cd], -1*np.sqrt(covars[j][:,i,i]), colors[j], linestyle=":")
    if with_kf:
        plt.plot(T, true_states[:,i] - kf_cond_means[:,i], 'g--')
        plt.plot(T, np.sqrt(kf_cond_covars[:,i,i]), 'm--')
        plt.plot(T, -1.0*np.sqrt(kf_cond_covars[:,i,i]), 'm--')
    if with_full_window_compare:
        plt.plot(T[cd:sim_len+cd], true_states[cd:sim_len+cd,i] - full_window_means[:,i], 'k')
        plt.plot(T[cd:sim_len+cd], np.sqrt(full_window_covars[:,i,i]), 'k--')
        plt.plot(T[cd:sim_len+cd], -1.0*np.sqrt(full_window_covars[:,i,i]), 'k--')

line_types = ['-', '--', '-.', ':', '-']
fig = plt.figure(3)
fig.suptitle("Msmts (m), Msmt Noise (g), Proc Noise (b)")
m = 3 #proc_noises.shape[1] + msmt_noises.shape[1] + msmts.shape[1]
count = 1
plt.subplot(str(m) + "1" + str(count))
for i in range(msmts.shape[1]):
    plt.plot(T, msmts[:,i], "m" + line_types[i])
count += 1
plt.subplot(str(m) + "1" + str(count))
for i in range(msmt_noises.shape[1]):
    plt.plot(T, msmt_noises[:,i], "g" + line_types[i])
count += 1
plt.subplot(str(m) + "1" + str(count))
for i in range(proc_noises.shape[1]):
    plt.plot(T[1:], proc_noises[:,i], "b" + line_types[i])

fig = plt.figure(4)
if with_averaged:
    fig.suptitle("Averaged Complex Errors (mean,covar,norm factor) in Semi-Log")
else:
    fig.suptitle("Complex Errors (mean,covar,norm factor) in Semi-Log")
plt.subplot(311)
for i in range(len(cerr_means)):
    plt.semilogy(T[i+cd:sim_len+cd], cerr_means[i], colors[i])
if with_full_window_compare:
    plt.semilogy(T[cd:sim_len+cd], full_window_cerr_means, 'k')
plt.subplot(312)
for i in range(len(cerr_means)):    
    plt.semilogy(T[i+cd:sim_len+cd], cerr_covars[i], colors[i])
if with_full_window_compare:
    plt.semilogy(T[cd:sim_len+cd], full_window_cerr_covars, 'k')
plt.subplot(313)
for i in range(len(cerr_means)):    
    plt.semilogy(T[i+cd:sim_len+cd], cerr_norm_factors[i], colors[i])
if with_full_window_compare:
    plt.semilogy(T[cd:sim_len+cd], full_window_cerr_norm_factors, 'k')

# Now if the user supplies the "e" cmd line arg, plot the extended information too
if(with_extended):
    fig = plt.figure(5)
    if with_averaged:
        fig.suptitle("Averaged delta_x = x_hat-x_bar (left); e_bar = x_truth - x_bar (right)")
    else:
        fig.suptitle("delta_x = x_hat-x_bar (left); e_bar = x_truth - x_bar (right)")
    for i in range(n):
        plt.subplot(int(str(n) + "2" + str(i+1)))
        for j in range(len(means)):
            plt.plot(T[j+cd:sim_len+cd], means[j][:,i] - x_bars[j][:,i], colors[j])
    for i in range(n):
        plt.subplot(int(str(n) + "2" + str(i+1+n)))
        for j in range(len(means)):
            plt.plot(T[j+cd:sim_len+cd], true_states[j+cd:sim_len+cd,i] - x_bars[j][:,i], colors[j])

    fig = plt.figure(6)
    if with_averaged:
        fig.suptitle("Averaged [z_k - h(x_bar) (left)]; [z_k - h(x_bar) - H*(x_truth-x_bar) (right)]")
    else:
        fig.suptitle("[z_k - h(x_bar) (left)]; [z_k - h(x_bar) - H*(x_truth-x_bar) (right)]")
    p = msmts.shape[1]
    for i in range(p):
        plt.subplot(str(p) + "2" + str(i+1))
        for j in range(len(resids)):
            plt.plot(T[j+cd:sim_len+cd], resids[j][:,i], colors[j])
        if with_kf:
            plt.plot(T[1:], kf_residuals[:, i], 'g--')
    for i in range(p):
        plt.subplot(str(p) + "2" + str(i+1+p))
        for j in range(len(resids)):
            plt.plot(T[j+cd:sim_len+cd], full_resids[j][:,i], colors[j])

plt.show()


