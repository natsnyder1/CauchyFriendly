# file : test_simulate_gaussian_ltiv_system.py 

import numpy as np
import scipy.io
import subprocess
import os

def generate_random_vectors(num_steps, W, V, x0_truth_mu, x0_truth_cov, with_zeroth_step_msmt = True):
    np.random.seed(0)  # Seed for reproducibility
    ws = np.random.multivariate_normal(np.zeros(W.shape[0]), W, num_steps).T
    vs = np.random.multivariate_normal(np.zeros(V.shape[0]), V, num_steps + (1 if with_zeroth_step_msmt else 0)).T
    x0_truth = np.random.multivariate_normal(x0_truth_mu, x0_truth_cov)
    
    # Saving the vectors to a .mat file for MATLAB to read
    scipy.io.savemat('random_noise_vectors.mat', {'ws': ws, 'vs': vs, 'x0_truth': x0_truth.reshape(-1,1)})

# This function is a copy of the original simulate_gaussian_ltiv_system, except it loads in noise from random_noise_vectors.mat
def simulate_gaussian_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, with_zeroth_step_msmt = True, dynamics_update_callback = None, other_params = None):
    mat_data = scipy.io.loadmat('random_noise_vectors.mat')
    ws = mat_data['ws']
    vs = mat_data['vs']
    if(B is not None):
        assert(us is not None)
        assert(us.shape[0] == num_steps)
        if(B.size == x0_truth.size):
            us = us.reshape((num_steps,1))
        else:
            assert(us.shape[1] == B.shape[1])
    else:
        assert(us is None)
        B = np.zeros((x0_truth.size,1))
        us = np.zeros((num_steps,1))
    if(W.size == 1):
        W = W.reshape((1,1))
    else:
        assert(W.shape[0] == W.shape[1])
    if(V.size == 1):
        V = V.reshape((1,1))
    else:
        assert(V.shape[0] == V.shape[1])
    assert(Gamma is not None)
    assert(W is not None)
    Gamma = Gamma.reshape((x0_truth.size, W.shape[0]))
    v_zero_vec = np.zeros(V.shape[0])
    w_zero_vec = np.zeros(W.shape[0])
    xk = x0_truth.reshape(-1, 1)
    xs = [xk]
    zs = [] 

    if(with_zeroth_step_msmt):
        v0 = vs[:, 0]
        z0 = H @ xk + v0.reshape(-1, 1)
        zs.append(z0)
    
    for i in range(num_steps):
        if dynamics_update_callback is not None:
            dynamics_update_callback(Phi, B, Gamma, H, W, V, i, other_params)
        uk = us[i, :].reshape(-1, 1)
        wk = ws[:, i].reshape(-1, 1)
        xk = Phi @ xk + B @ uk + Gamma @ wk 
        xs.append(xk)
        if(with_zeroth_step_msmt):
            vk = vs[:, i+1].reshape(-1, 1)
        else:
            vk = vs[:, i].reshape(-1, 1)
        zk = H @ xk + vk  # Measurement
        zs.append(zk)
    # Convert lists to numpy arrays for output
    xs = np.hstack(xs)
    zs = np.hstack(zs) if zs else np.array([]).reshape(H.shape[0], 0)
    ws = ws.copy()
    vs = vs[:, :num_steps + (1 if with_zeroth_step_msmt else 0)]

    return xs, zs, ws, vs

def run_python_simulation():
    
    # Define input parameters
    Phi = np.array([ [0.9, 0.1], [-0.2, 1.1] ])
    Gamma = np.array([.1, 0.3])
    B = None # No control matrix, since no controls
    us = None # No controls
    H = np.array([1.0, 0.5])
    W = np.array([[0.01]])
    V = np.array([[0.02]])
    num_propagations = 7

    x0_kf = np.zeros(2)
    P0_kf = np.eye(2) * 0.05

    generate_random_vectors(num_propagations, W, V, x0_kf, P0_kf, with_zeroth_step_msmt = True)

    mat_data = scipy.io.loadmat('random_noise_vectors.mat')
    x0_truth = mat_data['x0_truth']  # Extract random initial state
    
    # Call Python function
    xs_py, zs_py, ws_py, vs_py = simulate_gaussian_ltiv_system(num_propagations, x0_truth, us, Phi, B, Gamma, W, H, V)
    
    return xs_py, zs_py, ws_py, vs_py

def run_matlab_simulation():
    # Construct the MATLAB command to execute the script with the full MATLAB path
    matlab_executable = "/Applications/MATLAB_R2024a.app/bin/matlab"
    matlab_script_path = 'test_simulate_gaussian_ltiv_system.m'
    matlab_command = f'"{matlab_executable}" -batch "run(\'{matlab_script_path}\')"'

    # Execute MATLAB script using subprocess
    subprocess.run(matlab_command, shell=True)

    # Load MATLAB outputs from the saved .mat file
    matlab_outputs = scipy.io.loadmat('matlab_outputs.mat')
    xs_matlab = matlab_outputs['xs']
    zs_matlab = matlab_outputs['zs']
    ws_matlab = matlab_outputs['ws']
    vs_matlab = matlab_outputs['vs']

    # Clean up: Delete the MATLAB outputs file
    if os.path.exists('matlab_outputs.mat'):
        os.remove('matlab_outputs.mat')

    return xs_matlab, zs_matlab, ws_matlab, vs_matlab

# Run both versions
xs_py, zs_py, ws_py, vs_py = run_python_simulation()
xs_matlab, zs_matlab, ws_matlab, vs_matlab = run_matlab_simulation()

xs_py = xs_py.T
zs_py = zs_py.T
ws_py = ws_py.T
vs_py = vs_py.T

if os.path.exists('random_noise_vectors.mat'):
    os.remove('random_noise_vectors.mat')

print("\nPython Shapes:\nState Hist Shape:", xs_py.shape, "\nMeasurement Hist Shape:", zs_py.shape, "\nProcess Noise Hist Shape:",ws_py.shape, "\nMeasurement Noise Hist Shape:", vs_py.shape, "\n")
print("Matlab Shapes:\nState Hist Shape:", xs_matlab.shape, "\nMeasurement Hist Shape:", zs_matlab.shape, "\nProcess Noise Hist Shape:",ws_matlab.shape, "\nMeasurement Noise Hist Shape:", vs_matlab.shape, "\n")

# Compare outputs
print("Running asserts...")
assert np.allclose(xs_py, xs_matlab)
assert np.allclose(zs_py, zs_matlab)
assert np.allclose(ws_py, ws_matlab)
assert np.allclose(vs_py, vs_matlab)

print("Outputs from Python and MATLAB versions match.\n")

# Save outputs to a separate file that can be used by other functions
scipy.io.savemat('gaussian_simulation_outputs.mat', {
    'xs_py': xs_py,
    'zs_py': zs_py,
    'ws_py': ws_py,
    'vs_py': vs_py,
    'xs_matlab': xs_matlab,
    'zs_matlab': zs_matlab,
    'ws_matlab': ws_matlab,
    'vs_matlab': vs_matlab})