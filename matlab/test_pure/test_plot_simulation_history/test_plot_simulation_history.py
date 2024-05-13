# file: test_plot_simulation_history.py

import scipy.io
import subprocess
import sys

sys.path.append('/Users/nishadelias/Documents/GitHub/CauchyFriendly/scripts/tutorial')
import cauchy_estimator as ce 

gs_outputs = scipy.io.loadmat('../test_simulate_gaussian_ltiv_system/gaussian_simulation_outputs.mat')
xs = gs_outputs['xs_py']
zs = gs_outputs['zs_py']
ws = gs_outputs['ws_py']
vs = gs_outputs['vs_py']

kf_outputs = scipy.io.loadmat('../test_run_kalman_filter/kalman_filter_outputs.mat')
xs_kf = kf_outputs['xs_kf_py']
Ps_kf = kf_outputs['Ps_kf_py']

# uncomment the below line to see the python graphs
#ce.plot_simulation_history(None, (xs,zs,ws,vs), (xs_kf, Ps_kf))

matlab_executable = "/Applications/MATLAB_R2024a.app/bin/matlab"
matlab_script_path = 'test_plot_simulation_history.m'
matlab_command = f'"{matlab_executable}" -batch "run(\'{matlab_script_path}\')"'

subprocess.run(matlab_command, shell=True)

