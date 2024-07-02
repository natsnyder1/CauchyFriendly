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
x0_truth = mvnrnd(x0_kf, P0_kf, 1)';

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
x0_truth = mvnrnd(x0_kf, P0_kf, 1)';
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
% plot_simulation_history(cauchyEst.moment_info, {xs, zs, ws, vs}, {xs_kf, Ps_kf});



[xs, zs, ws, vs] = simulate_cauchy_ltiv_system(num_propagations, x0_truth, us, Phi, B, Gamma, W, H, V);
[xs_kf, Ps_kf] = run_kalman_filter(x0_kf, us, zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);

cauchyEst = MSlidingWindowManager("lti", num_windows, swm_print_debug, win_print_debug);
cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
u = [];
for k = 1:length(zs)
    zk = zs(k);
    [xk, Pk, wavg_xk, wavg_Pk] = cauchyEst.step(zk, u);
end
cauchyEst.shutdown();
% Plot results
% plot_simulation_history(cauchyEst.moment_info, {xs, zs, ws, vs}, {xs_kf, Ps_kf});


% % 1.) Lets hand craft a small simulation where a large process noise enters the system
% steps = 7;
% ws = [0.05;-0.03; 4.0; 0.1; -0.07; 0.08]; % large process noise jump between k=2 and k=3
% vs = [0.08; 0.1; -0.03; -0.07; 0.09; 0.13; 0.02];
% xk = [0.0;0.0];
% xs = xk';
% zs = H' * x0_truth + vs(1);
% for i = 1:(steps-1)
%     xk = Phi * xk + Gamma * ws(i);
%     xs = [xs; xk'];
%     zs = [zs; H' * xk + vs(i+1)];
% end
% 
% % Run KF 
% [xs_kf, Ps_kf] = run_kalman_filter(x0_kf, [], zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);
% 
% % Parameters for creating the CPDF grid
% grid_lowx = -0.50; % CPDF Grid Low X
% grid_highx = 1.25; % CPDF Grid High X
% grid_resx = 0.025; % CPDF Resolution in X
% grid_lowy = -0.50; % CPDF Grid Low Y
% grid_highy = 2.0;  % CPDF Grid High Y
% grid_resy = 0.025; % CPDF Resolution in Y
% 
% % Create mesh grid for CPDF
% [Xs, Ys] = meshgrid(grid_lowx:grid_resx:grid_highx, grid_lowy:grid_resy:grid_highy);
% 
% % Run CE and Plot its CPDF along the way
% cauchyEst = MCauchyEstimator("lti", steps, false);
% cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
% 
% for i = 1:size(zs, 1)
%     zk = zs(i);
%     [xk, Pk] = cauchyEst.step(zk);
%     if i > 2 && i < 7
%         % Get CPDF Grid over X and Y (f_XY)
%         [Xs, Ys, f_XY] = cauchyEst.get_2D_pointwise_cpdf(grid_lowx, grid_highx, grid_resx, grid_lowy, grid_highy, grid_resy); 
%         % Plot 
%         figure;
%         title(sprintf('Cauchy CPDF (Blue Wiremesh) at Step k=%d\nKF Mean (k*), CE Mean (b*), Truth (r*)', i));
%         ax = gca;
%         mesh(ax, Xs, Ys, f_XY, 'EdgeColor', 'b'); % Wiremesh plot
%         hold on;
%         % Plot conditional mean of Cauchy Estimator
%         scatter3(ax, xk(1), xk(2), 0, 'b', 'filled', 'MarkerEdgeColor', 'k', 'Marker', '*');
%         % Plot conditional mean of Kalman Filter
%         kfStateAtI = xs_kf(i, :); % Obtain Kalman Filter state estimate at step i
%         scatter3(ax, kfStateAtI(1), kfStateAtI(2), 0, 'k', '*', 'LineWidth', 2);
%         % Plot True State Location
%         trueStateAtI = xs(i, :); % Obtain true state at step i
%         scatter3(ax, trueStateAtI(1), trueStateAtI(2), 0, 'r', '*');
%         % Set labels
%         xlabel('x-axis (State-1)');
%         ylabel('y-axis (State-2)');
%         zlabel('z-axis (CPDF Probability)');
%         view(3);
% 
%         hold off;
%     end
% end
% 
% 
% % 1b.) Lets hand craft a small simulation where a process noise enter the system in one of two channels
% steps = 6;
% Gamma = eye(2); % two independent process noise channels
% beta = [0.02; 0.02];
% scale_c2g = 1/scale_g2c;
% W = diag(beta) * scale_c2g; % scale Gaussian to fit Cauchy distribution (in least squares sense)
% 
% ws = [[0.05, 1.50,  0.10, -0.07, 0.10];
%               [0.03, 0.08, -0.12,  0.02, -0.09]]'; % large process noise jump between k=2 and k=3
% vs = [0.08; 0.1; -0.03; -0.07; 0.09; 0.1];
% xk = [0.0; 0.0];
% xs = xk';
% zs = H' * x0_truth + vs(1);
% for i = 1:(steps-1)
%     xk = Phi * xk + Gamma * ws(i,:)';
%     xs = [xs; xk'];
%     zs = [zs; H' * xk + vs(i+1)];
% end
% 
% % Run KF 
% [xs_kf, Ps_kf] = run_kalman_filter(x0_kf, [], zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);
% 
% % Run CE and Plot its CPDF along the way
% cauchyEst = MCauchyEstimator("lti", steps, false);
% cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
% 
% grid_lowx = -0.25; % CPDF Grid Low X
% grid_highx = 2.25; % CPDF Grid High X
% grid_resx = 0.05; % CPDF Resolution in X
% grid_lowy = -1.0; % CPDF Grid Low Y
% grid_highy = 3.5;  % CPDF Grid High Y
% grid_resy = 0.05; % CPDF Resolution in Y
% 
% for i = 1:size(zs, 1)
%     zk = zs(i);
%     [xk, Pk] = cauchyEst.step(zk);
%     if i > 1 && i < 6
%         % Get CPDF Grid over X and Y (f_XY)
%         [Xs, Ys, f_XY] = cauchyEst.get_2D_pointwise_cpdf(grid_lowx, grid_highx, grid_resx, grid_lowy, grid_highy, grid_resy); 
%         % Plot 
%         figure;
%         title(sprintf('Cauchy CPDF (Blue Wiremesh) at Step k=%d\nKF Mean (k*), CE Mean (b*), Truth (r*)', i));
%         ax = gca;
%         mesh(ax, Xs, Ys, f_XY, 'EdgeColor', 'b'); % Wiremesh plot
%         hold on;
%         % Plot conditional mean of Cauchy Estimator
%         scatter3(ax, xk(1), xk(2), 0, 'b', 'filled', 'MarkerEdgeColor', 'k', 'Marker', '*');
%         % Plot conditional mean of Kalman Filter
%         kfStateAtI = xs_kf(i, :); % Obtain Kalman Filter state estimate at step i
%         scatter3(ax, kfStateAtI(1), kfStateAtI(2), 0, 'k', '*', 'LineWidth', 2);
%         % Plot True State Location
%         trueStateAtI = xs(i, :); % Obtain true state at step i
%         scatter3(ax, trueStateAtI(1), trueStateAtI(2), 0, 'r', '*');
%         % Set labels
%         xlabel('x-axis (State-1)');
%         ylabel('y-axis (State-2)');
%         zlabel('z-axis (CPDF Probability)');
%         view(3);
% 
%         hold off;
%     end
% end
% cauchyEst.shutdown();
% 
% 
% % 2.) Lets hand craft a small simulation where a large measurement noise enters the system
% steps = 6;
% ws = [[0.05, 0.07,  0.10, -0.07, 0.10];
%               [0.03, 0.08, -0.12,  0.02, -0.09]]';
% vs = [0.08; -0.05; 4.5; -0.07; 0.09; 0.1]; % large measurement noise at k=2
% xk = [0.0; 0.0];
% xs = xk';
% zs = H' * x0_truth + vs(1);
% for i = 1:(steps-1)
%     xk = Phi * xk + Gamma * ws(i,:)';
%     xs = [xs; xk'];
%     zs = [zs; H' * xk + vs(i+1)];
% end
% 
% % Run KF 
% [xs_kf, Ps_kf] = run_kalman_filter(x0_kf, [], zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);
% 
% % Run CE and Plot its CPDF along the way
% cauchyEst = MCauchyEstimator("lti", steps, false);
% cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
% 
% grid_lowx = -2; % CPDF Grid Low X
% grid_highx = 6; % CPDF Grid High X
% grid_resx = 0.001; % CPDF Resolution in X
% 
% for i = 1:size(zs, 1)
%     zk = zs(i);
%     [xk, Pk] = cauchyEst.step(zk);
%     if i > 1 && i < 6
%         % Get Marginal CPDF Grid over X
%         state1_idx = 0;
%         [X1, f_X1] = cauchyEst.get_marginal_1D_pointwise_cpdf(state1_idx, grid_lowx, grid_highx, grid_resx);
%         state2_idx = 1;
%         [X2, f_X2] = cauchyEst.get_marginal_1D_pointwise_cpdf(state2_idx, grid_lowx, grid_highx, grid_resx);
%         % Plot 
%         figure;
%         sgtitle(sprintf('Cauchy Marginal CPDF (State1 top, State2 bottom) at Step k=%d\nKF Mean (k*), CE Mean (b*), Truth (r*)', i));
%         % Plot Marginal CPDF of State 1, conditional mean of Cauchy Estimator, Kalman filter, and True State Location
%         subplot(2, 1, 1); % 2 rows, 1 column, 1st plot
%         plot(X1, f_X1, 'b');
%         hold on; % Retain plots so that new plot does not delete existing ones
%         scatter(xk(1), 0, 260, 'b', '*');
%         scatter(xs_kf(i, 1), 0, 260, 'k', '*');
%         scatter(xs(i, 1), 0, 260, 'r', '*');
%         xlabel('x1');
%         ylabel('Marginal CPDF f(x1)');
%         hold off; % Release hold for next plot
%         % Plot Marginal CPDF of State 2, conditional mean of Cauchy Estimator, Kalman filter, and True State Location
%         subplot(2, 1, 2); % 2 rows, 1 column, 2nd plot
%         plot(X2, f_X2, 'b');
%         hold on;
%         scatter(xk(2), 0, 260, 'b', '*', 'LineWidth', 1);
%         scatter(xs_kf(i, 2), 0, 260, 'k', '*', 'LineWidth', 1);
%         scatter(xs(i, 2), 0, 260, 'r', '*', 'LineWidth', 1);
%         xlabel('x2');
%         ylabel('Marginal CPDF f(x2)');
%         hold off;
%     end
% end
% cauchyEst.shutdown();

