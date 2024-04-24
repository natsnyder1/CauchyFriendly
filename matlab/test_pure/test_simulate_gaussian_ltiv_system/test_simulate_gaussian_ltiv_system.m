% file : test_simulate_gaussian_ltiv_system.m 

% Load random initial state from the .mat file
data = load('random_noise_vectors.mat');
x0_truth = data.x0_truth; % Extract random initial state

% Define input parameters
num_steps = 7;
us = []; % No control inputs
Phi = [0.9, 0.1; -0.2, 1.1];
Gamma = [0.1, 0.3];
B = [];
H = [1.0, 0.5];
W = 0.01;
V = 0.02;

% Call the simulation function
[xs, zs, ws, vs] = test_function(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V);

% Save the outputs to a .mat file
save('matlab_outputs.mat', 'xs', 'zs', 'ws', 'vs');