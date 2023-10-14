#/usr/bin/env python3

import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import os 

print("Command Line Argument Helper:\n --- Option 1 (Default): No cmd line arg given -- plot the KF txt files in 'log/kf_log' folder\n --- Option 2: enter 'm' (main) to plot KF text files in 'log' folder\nEnter l=path_to_data to change log folder")
print('Enter c to plot control history')
print('Enter s={1,2,3} to set the sigma bound to +/-{1,2,3} standard deviations...default is s=1')
log_dir = "../log/kf_log/"
with_control_plot = False
sig = 1
for i in range(1, len(sys.argv)):
    if(sys.argv[i] == 'm'):
        log_dir = "../log/"
    elif(sys.argv[i] == 'c'):
        with_control_plot = True
    elif(sys.argv[i].startswith("s=")):
        sig_str = sys.argv[i].split("=")[1]
        sig = int(sig_str)
    elif(sys.argv[i].startswith("l=")):
        log_dir = sys.argv[i].split("=")[1]
        if log_dir[-1] != "/":
            log_dir += "/"
    else:
        print("UNKNOWN CMD LINE OPTION ", sys.argv[i], ". SEE ABOVE USAGE FOR HELP!")
        exit(1)
print("Plotting data in the ", log_dir, " sub-directory!")
assert(os.path.isdir(log_dir))

f_true_states = "true_states.txt"
f_msmts = "msmts.txt"
f_msmt_noises = "msmt_noises.txt"
f_proc_noises = "proc_noises.txt"

f_kf_cond_means = "kf_cond_means.txt"
f_kf_cond_covars = "kf_cond_covars.txt"
f_kf_controls = "kf_controls.txt"

# Reads and parses Kalman Filter Data
def load_data(f_data):
    file = open(f_data, 'r')
    lines = file.readlines()
    return np.array([[float(f) for f in line.split(" ")] for line in lines])

# Loads KF Means 
kf_cond_means = load_data(log_dir + f_kf_cond_means)
print("Means: ", kf_cond_means.shape)

# Loads KF Covariances
kf_cond_covars = load_data(log_dir + f_kf_cond_covars)
n = int(np.sqrt(kf_cond_covars.shape[1]))
kf_cond_covars = kf_cond_covars.reshape((kf_cond_covars.shape[0], n, n))
print("Covars after Reshaping: ", kf_cond_covars.shape)

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

# 1.) Plot the true state history vs the conditional mean estimate  
# 2.) Plot the state error and one-sigma bound of the covariance 
# 3.) Plot the msmts, and the msmt and process noise 
# 4.) Plot the max complex error in the mean/covar and norm factor 
msmt_at_idx_zero_offset = 0
if(msmts.shape[0] == kf_cond_means.shape[0]):
    msmt_at_idx_zero_offset -= 1
T = np.arange(0, kf_cond_means.shape[0])
fig = plt.figure(1)
fig.suptitle("True States (r) vs Kalman Estimates (g--)")
for i in range(kf_cond_means.shape[1]):
    plt.subplot(str(n) + "1" + str(i+1))
    plt.plot(T, true_states[:,i], 'r')
    plt.plot(T, kf_cond_means[:,i], 'g--')


fig = plt.figure(2)
fig.suptitle("Kalman State Error (b) vs One Sigma Bound (r)")
for i in range(kf_cond_means.shape[1]):
    plt.subplot(str(n) + "1" + str(i+1))
    plt.plot(T, true_states[:,i] - kf_cond_means[:,i], 'b')
    plt.plot(T, sig*np.sqrt(kf_cond_covars[:,i,i]), 'r')
    plt.plot(T, -1.0*sig*np.sqrt(kf_cond_covars[:,i,i]), 'r')

line_types = ['-', '--', '-.', ':', '-']
fig = plt.figure(3)
fig.suptitle("Msmts (m), Msmt Noise (g), Proc Noise (b)")
m = 3 #proc_noises.shape[1] + msmt_noises.shape[1] + msmts.shape[1]
count = 1
plt.subplot(str(m) + "1" + str(count))
for i in range(msmts.shape[1]):
    plt.plot(T[1+msmt_at_idx_zero_offset:], msmts[:,i], "m" + line_types[i])
count += 1
plt.subplot(str(m) + "1" + str(count))
for i in range(msmt_noises.shape[1]):
    plt.plot(T[1+msmt_at_idx_zero_offset:], msmt_noises[:,i], "g" + line_types[i])
count += 1
plt.subplot(str(m) + "1" + str(count))
for i in range(proc_noises.shape[1]):
    plt.plot(T[1:], proc_noises[:,i], "b" + line_types[i])

if with_control_plot:
    kf_controls = np.loadtxt(log_dir + f_kf_controls)
    fig = plt.figure(4)
    plt.title("KF State Dependent Control History")
    plt.plot(T[1:], kf_controls, 'b')

plt.show()


