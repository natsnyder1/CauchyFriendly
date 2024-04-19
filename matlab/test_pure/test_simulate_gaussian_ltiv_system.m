% MATLAB testing script (test_simulate_gaussian_ltiv_system.m)

% Load the saved random noise vectors
noise = load('../random_noise_vectors.mat');
W_noise = noise.W_noise;
V_noise = noise.V_noise;

% Define the systems initial conditions and parameters, match the ones used in python
num_steps = 10;
x0_truth = [1; 1];
us = ones(num_steps, 1);
Phi = eye(2);
B = [0.1; 0.1];
Gamma = eye(2);
W = eye(2);
H = [1, 0];
V = 1;
% Call the MATLAB function with the noise vectors
[xs, zs, ws, vs] = simulate_gaussian_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, true, [], [], W_noise, V_noise);

% Since external noise is used, these results can now be compared directly with the Python results.