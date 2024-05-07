# file: test_run_kalman_filter.py
import numpy as np
import scipy.io
import subprocess
import os
import sys

sys.path.append('/Users/nishadelias/Documents/GitHub/CauchyFriendly/scripts/tutorial')
import gaussian_filters as gf

def run_python_kalman():
    # Define input parameters
    Phi = np.array([ [0.9, 0.1], [-0.2, 1.1] ])
    Gamma = np.array([.1, 0.3])
    B = None # No control matrix, since no controls
    us = None # No controls
    H = np.array([1.0, 0.5])
    W = np.array([[0.01]])
    V = np.array([[0.02]])

    x0_kf = np.zeros(2)
    P0_kf = np.eye(2) * 0.05


    gs_outputs = scipy.io.loadmat('../test_simulate_gaussian_ltiv_system/gaussian_simulation_outputs.mat')
    zs = gs_outputs['zs_py']

    # Call Python function
    xs_kf, Ps_kf = gf.run_kalman_filter(x0_kf, us, zs[1:], P0_kf, Phi, B, Gamma, H, W, V)

    return xs_kf, Ps_kf


def run_matlab_kalman():
    # Construct the MATLAB command to execute the script with the full MATLAB path
    matlab_executable = "/Applications/MATLAB_R2024a.app/bin/matlab"
    matlab_script_path = 'test_run_kalman_filter.m'
    matlab_command = f'"{matlab_executable}" -batch "run(\'{matlab_script_path}\')"'

    # Execute MATLAB script using subprocess
    subprocess.run(matlab_command, shell=True)

    # Load MATLAB outputs from the saved .mat file
    matlab_outputs = scipy.io.loadmat('matlab_outputs.mat')
    xs_kf = matlab_outputs['xs_kf']
    Ps_kf = matlab_outputs['Ps_kf']

    # Clean up: Delete the MATLAB outputs file
    if os.path.exists('matlab_outputs.mat'):
        os.remove('matlab_outputs.mat')

    return xs_kf, Ps_kf


# Run both versions
xs_kf_py, Ps_kf_py = run_python_kalman()
xs_kf_matlab, Ps_kf_matlab = run_matlab_kalman()

xs_kf_py = xs_kf_py.T
Ps_kf_py = Ps_kf_py.T

print("\nPython Shapes:\nxs_kf:", xs_kf_py.shape, "\nPs_kf:", Ps_kf_py.shape, "\n")
print("\nMatlab Shapes:\nxs_kf:", xs_kf_matlab.shape, "\nPs_kf:", Ps_kf_matlab.shape, "\n")
print("Note that the Python Shapes in this test have been transposed, because numpy arrays are row-major and matlab arrays are column-major\n")

# Compare outputs
assert np.allclose(xs_kf_py, xs_kf_matlab)
assert np.allclose(Ps_kf_py, Ps_kf_matlab)

print("Outputs from Python and MATLAB versions match.\n")

scipy.io.savemat('kalman_filter_outputs.mat', {
    'xs_kf_py': xs_kf_py.T,
    'Ps_kf_py': Ps_kf_py.T,
    'xs_kf_matlab': xs_kf_matlab,
    'Ps_kf_matlab': Ps_kf_matlab.T})