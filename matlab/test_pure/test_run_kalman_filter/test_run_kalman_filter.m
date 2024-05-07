% file: test_run_kalman_filter.m

gs_outputs = load('../test_simulate_gaussian_ltiv_system/gaussian_simulation_outputs.mat');
zs = gs_outputs.zs_matlab;

% Define input parameters
num_steps = 7;
us = []; % No control inputs
Phi = [0.9, 0.1; -0.2, 1.1];
Gamma = [0.1, 0.3];
B = [];
H = [1.0, 0.5];
W = 0.01;
V = 0.02;
x0_kf = zeros(2, 1);
P0_kf = eye(2) * 0.05;


% Call the simulation function
addpath('../../matlab_pure');
[xs_kf, Ps_kf] = run_kalman_filter(x0_kf, us, zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);
rmpath('../../matlab_pure');

% Save the outputs to a .mat file
save('matlab_outputs.mat', 'xs_kf', 'Ps_kf');