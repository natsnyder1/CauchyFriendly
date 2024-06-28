addpath("matlab_pure");
addpath("mex_files");

num_propagations = 7;
us = [];
Phi = [0.9, 0.1; -0.2, 1.1];
Gamma = [0.1; 0.3];
B = [];
H = [1.0; 0.5];
W = 0.01;
V = 0.02;

x0_kf = zeros(2, 1);
P0_kf = eye(2) * 0.05;
x0_truth = x0_kf + chol(P0_kf, 'lower') * randn(2, 1);

[xs, zs, ws, vs] = simulate_gaussian_ltiv_system(num_propagations, x0_truth, us, Phi, B, Gamma, W, H, V);
fprintf('State Hist Shape: %s\nMeasurement Hist Shape: %s\nProcess Noise Hist Shape: %s\nMeasurement Noise Hist Shape: %s\n\n', ...
        mat2str(size(xs)), mat2str(size(zs)), mat2str(size(ws)), mat2str(size(vs)));

[xs_kf, Ps_kf] = run_kalman_filter(x0_kf, us, zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);
fprintf('xs_kf shape: %s\nPs_kf shape: %s\n', ...
        mat2str(size(xs_kf)), mat2str(size(Ps_kf)));

%plot_simulation_history([], {xs,zs,ws,vs}, {xs_kf, Ps_kf});


scale_g2c = 1.0 / 1.3898; % Scale the Gaussian's standard deviation
beta = sqrt(W(1,1)) * scale_g2c; % Process noise scale parameter for the Cauchy Estimator
gamma = sqrt(V(1,1)) * scale_g2c; % Measurement noise scale parameter for the Cauchy Estimator
p0 = sqrt(diag(P0_kf)) * scale_g2c; % Initial state uncertainty scale parameters for the Cauchy Estimator
A0 = eye(2); % unit directions (row-wise) which describe the directions of initial uncertainty for p0
b0 = x0_kf; % The 'median' vector of the Cauchy Estimator's initial state hypothesis

steps = num_propagations + 1;
print_debug = false;

cauchyEst = MCauchyEstimator("lti", steps, print_debug);
cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);

% Give it the measurements
u = []; % no controls
for k = 1:length(zs)
    % returns the conditional mean xk and covariance matrix Pk at each estimation step
    % this object also has an internal structure / property moment_info which stores this (and more) as well
    zk = zs(k);
    [xk, Pk] = cauchyEst.step(zk, u); % Call the step function with the current measurement
end

%plot_simulation_history(cauchyEst.moment_info, {xs, zs, ws, vs}, {xs_kf, Ps_kf});

cauchyEst.shutdown();

% Create a longer simulation
num_propagations = 200;
x0_truth = x0_kf + chol(P0_kf, 'lower') * randn(2, 1); 
[xs, zs, ws, vs] = simulate_gaussian_ltiv_system(num_propagations, x0_truth, us, Phi, B, Gamma, W, H, V);

% Run the Kalman filter
[xs_kf, Ps_kf] = run_kalman_filter(x0_kf, us, zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);

% Running the Cauchy estimator for long time horizons
num_windows = 8;
swm_print_debug = false; % Turn this on to see basic window information
win_print_debug = false; % Turn this on to see indivdual window (estimator) information

cauchyEst = MSlidingWindowManager("lti", num_windows, swm_print_debug, win_print_debug);
cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
u = []; % no controls
for k = 1:length(zs)
    % Below, xk and Pk are the conditional mean and covariance given by a window (estimator)...
    % ...which has processed the most measurements at the current time step
    % Below, wavg_xk and wavg_Pk are the weighted averages of the conditional mean and covariance across all windows...
    % ...which typically varies litlle from xk and Pk, but in some cases, can yield much smoother trajectories

    % The sliding window manager is explained in more detail below the generated figures
    zk = zs(k);
    [xk, Pk, wavg_xk, wavg_Pk] = cauchyEst.step(zk, u);
end
cauchyEst.shutdown();
% Plot results
plot_simulation_history(cauchyEst.moment_info, {xs, zs, ws, vs}, {xs_kf, Ps_kf});

rmpath("matlab_pure");
rmpath("mex_files");