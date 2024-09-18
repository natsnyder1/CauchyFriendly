addpath("matlab_pure");
addpath("mex_files");

%test_state_spaces() % for compare against python
%test_lti_systems_tutorial();
%test_3state_lti_window_manager();
%test_2state_lti_window_manager();
%test_2state_lti_single_window();
%test_3state_lti_single_window();
test_3state_marginal_cpdfs();

function test_lti_systems_tutorial()
    num_propagations = 7;
    us = [];
    Phi = [0.9, 0.1; -0.2, 1.1];
    Gamma = [0.1; 0.3];
    B = [];
    H = [1.0 0.5];
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


    % 1.) Lets hand craft a small simulation where a large process noise enters the system
    steps = 7;
    ws = [0.05;-0.03; 4.0; 0.1; -0.07; 0.08]; % large process noise jump between k=2 and k=3
    vs = [0.08; 0.1; -0.03; -0.07; 0.09; 0.13; 0.02];
    xk = [0.0;0.0];
    xs = xk';
    zs = H * x0_truth + vs(1);
    for i = 1:(steps-1)
        xk = Phi * xk + Gamma * ws(i);
        xs = [xs; xk'];
        zs = [zs; H * xk + vs(i+1)];
    end

    % Run KF 
    [xs_kf, Ps_kf] = run_kalman_filter(x0_kf, [], zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);

    % Parameters for creating the CPDF grid
    grid_lowx = -0.50; % CPDF Grid Low X
    grid_highx = 1.25; % CPDF Grid High X
    grid_resx = 0.025; % CPDF Resolution in X
    grid_lowy = -0.50; % CPDF Grid Low Y
    grid_highy = 2.0;  % CPDF Grid High Y
    grid_resy = 0.025; % CPDF Resolution in Y

    % Create mesh grid for CPDF
    [Xs, Ys] = meshgrid(grid_lowx:grid_resx:grid_highx, grid_lowy:grid_resy:grid_highy);

    % Run CE and Plot its CPDF along the way
    cauchyEst = MCauchyEstimator("lti", steps, false);
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);

    for i = 1:size(zs, 1)
        zk = zs(i);
        [xk, Pk] = cauchyEst.step(zk);
        if i > 2 && i < 7
            % Get CPDF Grid over X and Y (f_XY)
            [Xs, Ys, f_XY] = cauchyEst.get_2D_pointwise_cpdf(grid_lowx, grid_highx, grid_resx, grid_lowy, grid_highy, grid_resy); 
            % % Plot 
            % figure;
            % title(sprintf('Cauchy CPDF (Blue Wiremesh) at Step k=%d\nKF Mean (k*), CE Mean (b*), Truth (r*)', i));
            % ax = gca;
            % mesh(ax, Xs, Ys, f_XY, 'EdgeColor', 'b'); % Wiremesh plot
            % hold on;
            % % Plot conditional mean of Cauchy Estimator
            % scatter3(ax, xk(1), xk(2), 0, 500, 'g', '*', 'LineWidth', 4, 'DisplayName', 'CE Mean');
            % % Plot conditional mean of Kalman Filter
            % kfStateAtI = xs_kf(i, :); % Obtain Kalman Filter state estimate at step i
            % scatter3(ax, kfStateAtI(1), kfStateAtI(2), 0, 500, 'k', '*', 'LineWidth', 4, 'DisplayName', 'KF Mean');
            % % Plot True State Location
            % trueStateAtI = xs(i, :); % Obtain true state at step i
            % scatter3(ax, trueStateAtI(1), trueStateAtI(2), 0, 500, 'r', '*', 'LineWidth', 4, 'DisplayName', 'Truth');
            % % Set labels
            % xlabel('x-axis (State-1)');
            % ylabel('y-axis (State-2)');
            % zlabel('z-axis (CPDF Probability)');
            % view(-60, 20);
            % hold off;
        end
    end


    % 1b.) Lets hand craft a small simulation where a process noise enter the system in one of two channels
    steps = 6;
    Gamma = eye(2); % two independent process noise channels
    beta = [0.02; 0.02];
    scale_c2g = 1/scale_g2c;
    W = diag(beta) * scale_c2g; % scale Gaussian to fit Cauchy distribution (in least squares sense)

    ws = [[0.05, 1.50,  0.10, -0.07, 0.10];
                  [0.03, 0.08, -0.12,  0.02, -0.09]]'; % large process noise jump between k=2 and k=3
    vs = [0.08; 0.1; -0.03; -0.07; 0.09; 0.1];
    xk = [0.0; 0.0];
    xs = xk';
    zs = H * x0_truth + vs(1);
    for i = 1:(steps-1)
        xk = Phi * xk + Gamma * ws(i,:)';
        xs = [xs; xk'];
        zs = [zs; H * xk + vs(i+1)];
    end

    % Run KF 
    [xs_kf, Ps_kf] = run_kalman_filter(x0_kf, [], zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);

    % Run CE and Plot its CPDF along the way
    cauchyEst = MCauchyEstimator("lti", steps, false);
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);

    grid_lowx = -0.25; % CPDF Grid Low X
    grid_highx = 2.25; % CPDF Grid High X
    grid_resx = 0.05; % CPDF Resolution in X
    grid_lowy = -1.0; % CPDF Grid Low Y
    grid_highy = 3.5;  % CPDF Grid High Y
    grid_resy = 0.05; % CPDF Resolution in Y

    for i = 1:size(zs, 1)
        zk = zs(i);
        [xk, Pk] = cauchyEst.step(zk);
        if i > 1 && i < 6
            % Get CPDF Grid over X and Y (f_XY)
            [Xs, Ys, f_XY] = cauchyEst.get_2D_pointwise_cpdf(grid_lowx, grid_highx, grid_resx, grid_lowy, grid_highy, grid_resy); 
            % % Plot 
            % figure;
            % title(sprintf('Cauchy CPDF (Blue Wiremesh) at Step k=%d\nKF Mean (k*), CE Mean (b*), Truth (r*)', i));
            % ax = gca;
            % mesh(ax, Xs, Ys, f_XY, 'EdgeColor', 'b', 'EdgeAlpha', 0.4, 'FaceColor', 'b', 'FaceAlpha', 0.01);
            % hold on;
            % % Plot conditional mean of Cauchy Estimator
            % %scatter3(ax, xk(1), xk(2), 0, 'b', 'filled', 'MarkerEdgeColor', 'k', 'Marker', '*');
            % scatter3(ax, xk(1), xk(2), 0, 500, 'g', '*', 'LineWidth', 4, 'DisplayName', 'CE Mean');
            % % Plot conditional mean of Kalman Filter
            % kfStateAtI = xs_kf(i, :); % Obtain Kalman Filter state estimate at step i
            % scatter3(ax, kfStateAtI(1), kfStateAtI(2), 0, 500, 'k', '*', 'LineWidth', 4, 'DisplayName', 'KF Mean');
            % % Plot True State Location
            % trueStateAtI = xs(i, :); % Obtain true state at step i
            % scatter3(ax, trueStateAtI(1), trueStateAtI(2), 0, 500, 'r', '*', 'LineWidth', 4, 'DisplayName', 'Truth');
            % % Set labels
            % xlabel('x-axis (State-1)');
            % ylabel('y-axis (State-2)');
            % zlabel('z-axis (CPDF Probability)');
            % view(-60, 20);
            % hold off;
        end
    end
    cauchyEst.shutdown();


    % 2.) Lets hand craft a small simulation where a large measurement noise enters the system
    steps = 6;
    ws = [[0.05, 0.07,  0.10, -0.07, 0.10];
                  [0.03, 0.08, -0.12,  0.02, -0.09]]';
    vs = [0.08; -0.05; 4.5; -0.07; 0.09; 0.1]; % large measurement noise at k=2
    xk = [0.0; 0.0];
    xs = xk';
    zs = H * x0_truth + vs(1);
    for i = 1:(steps-1)
        xk = Phi * xk + Gamma * ws(i,:)';
        xs = [xs; xk'];
        zs = [zs; H * xk + vs(i+1)];
    end

    % Run KF 
    [xs_kf, Ps_kf] = run_kalman_filter(x0_kf, [], zs(2:end), P0_kf, Phi, B, Gamma, H, W, V);

    % Run CE and Plot its CPDF along the way
    cauchyEst = MCauchyEstimator("lti", steps, false);
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);

    grid_lowx = -2; % CPDF Grid Low X
    grid_highx = 6; % CPDF Grid High X
    grid_resx = 0.001; % CPDF Resolution in X

    for i = 1:size(zs, 1)
        zk = zs(i);
        [xk, Pk] = cauchyEst.step(zk);
        if i > 1 && i < 6
            % Get Marginal CPDF Grid over X
            state1_idx = 1;
            [X1, f_X1] = cauchyEst.get_marginal_1D_pointwise_cpdf(state1_idx, grid_lowx, grid_highx, grid_resx);
            state2_idx = 2;
            [X2, f_X2] = cauchyEst.get_marginal_1D_pointwise_cpdf(state2_idx, grid_lowx, grid_highx, grid_resx);
            % % Plot 
            % figure;
            % sgtitle(sprintf('Cauchy Marginal CPDF (State1 top, State2 bottom) at Step k=%d\nKF Mean (k*), CE Mean (b*), Truth (r*)', i));
            % % Plot Marginal CPDF of State 1, conditional mean of Cauchy Estimator, Kalman filter, and True State Location
            % subplot(2, 1, 1); % 2 rows, 1 column, 1st plot
            % plot(X1, f_X1, 'b');
            % hold on; % Retain plots so that new plot does not delete existing ones
            % scatter(xk(1), 0, 260, 'b', '*');
            % scatter(xs_kf(i, 1), 0, 260, 'k', '*');
            % scatter(xs(i, 1), 0, 260, 'r', '*');
            % xlabel('x1');
            % ylabel('Marginal CPDF f(x1)');
            % hold off; % Release hold for next plot
            % % Plot Marginal CPDF of State 2, conditional mean of Cauchy Estimator, Kalman filter, and True State Location
            % subplot(2, 1, 2); % 2 rows, 1 column, 2nd plot
            % plot(X2, f_X2, 'b');
            % hold on;
            % scatter(xk(2), 0, 260, 'b', '*', 'LineWidth', 1);
            % scatter(xs_kf(i, 2), 0, 260, 'k', '*', 'LineWidth', 1);
            % scatter(xs(i, 2), 0, 260, 'r', '*', 'LineWidth', 1);
            % xlabel('x2');
            % ylabel('Marginal CPDF f(x2)');
            % hold off;
        end
    end
    cauchyEst.shutdown();
end

function test_state_spaces()
    zs = reshape([1,2,3,4,5,6,7,8,9,10],2,5)';

    Phi2 = [[0.9, 0.1];[-0.2, 1.1]];
    Gam2  = [ [0.1, -0.15] ; [0.3, 0.2]];
    B2 = [];
    H2 = [[1.0,0.5];[0.7, -0.3]];
    beta2 = [0.20, 0.25];
    gamma2 = [0.15,0.10];
    A02 = eye(2);
    p02 = ones(2,1)*0.1;
    b02 = zeros(2,1);
    
    % Running the Cauchy estimator for long time horizons
    num_windows = 3;
    swm_print_debug = false; % Turn this on to see basic window information
    win_print_debug = false; % Turn this on to see indivdual window (estimator) information
    cauchyEst = MSlidingWindowManager("lti", num_windows, swm_print_debug, win_print_debug);
    cauchyEst.initialize_lti(A02, p02, b02, Phi2, B2, Gam2, beta2, H2, gamma2);
    for k = 1:length(zs)
        zk = zs(k,:);
        [xk, Pk, xavgk, Pavgk] = cauchyEst.step(zk,[]);
        fprintf("Step %d, x=[%f, %f]\n", k, xk(1), xk(2));
    end
    cauchyEst.shutdown()
    
    Phi3 = [[1.4, -0.6, -1.0];[-0.2,  1.0,  0.5];[0.6, -0.6, -0.2]];
    Gam3 = [[0.1,-0.2]; [0.3,0.4]; [-0.2,1.0]];
    B3 = [];
    H3 = [[1.0, 0.5, 0.2];[0.2, -0.3, 0.8]];
    beta3 = [0.1, 0.15];
    gamma3 = [0.2, 0.25];
    A03 = eye(3);
    p03 = ones(3,1)*0.1;
    b03 = zeros(3,1);
    num_windows = 3;
    swm_print_debug = false; % Turn this on to see basic window information
    win_print_debug = false; % Turn this on to see indivdual window (estimator) information
    cauchyEst = MSlidingWindowManager("lti", num_windows, swm_print_debug, win_print_debug);
    cauchyEst.initialize_lti(A03, p03, b03, Phi3, B3, Gam3, beta3, H3, gamma3);
    for k = 1:length(zs)
        zk = zs(k,:);
        [xk, Pk, xavgk, Pavgk] = cauchyEst.step(zk,[]);
        fprintf("Step %d, x=[%f, %f, %f]\n", k, xk(1), xk(2), xk(3));
    end
end

% Creates a single Cauchy estimator instance and runs the estimator for 1D case
function test_1state_lti()
    rng(105);
    Phi = 0.9;
    B = 1.0;
    Gamma = 0.4;
    H = 1.0;
    beta = 0.1; % Cauchy process noise scaling parameter(s)
    gamma = 0.2; % Cauchy measurement noise scaling parameter(s)
    A0 = 1.0; % Unit directions of the initial state uncertainty
    p0 = 0.10; % Initial state uncertainty cauchy scaling parameter(s)
    b0 = 0.0; % Initial median of system state
    x0_truth = randn() * CAUCHY_TO_GAUSSIAN_NOISE * p0;
    num_steps = 100;
    us_truth = sin(2 * pi * (0:num_steps-1) / num_steps)';

    [xs, zs, ws, vs] = simulate_cauchy_ltiv_system(num_steps, x0_truth, us_truth, Phi, B, Gamma, beta, H, gamma);
    
    % Cauchy Estimator for 1D
    cauchyEst = MSlidingWindowManager("lti", length(zs), true);
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);

    xs_ce = [];
    Ps_ce = [];

    for i = 1:length(zs)
        if i > 1
            [x_ce, P_ce] = cauchyEst.step(zs(i), us_truth(i-1));
        else
            [x_ce, P_ce] = cauchyEst.step(zs(i), 0.0);
        end
        xs_ce = [xs_ce, x_ce];
        Ps_ce = [Ps_ce, P_ce];
    end

    % Plotting logic can be added here
    % plot_simulation(xs, xs_ce, Ps_ce, xs_kf, Ps_kf, num_steps);

    cauchyEst.shutdown();
end

% Creates a single Cauchy estimator instance and runs the estimator for several steps
function test_3state_lti_single_window()
    ndim = 3;
    Phi = [1.4, -0.6, -1.0; 
           -0.2,  1.0,  0.5;  
            0.6, -0.6, -0.2];
    Gamma = [0.1; 0.3; -0.2];
    H = [1.0, 0.5, 0.2];
    beta = [0.1]; % Cauchy process noise scaling parameter(s)
    gamma = [0.2]; % Cauchy measurement noise scaling parameter(s)
    A0 = eye(ndim); % Unit directions of the initial state uncertainty
    p0 = [0.10; 0.08; 0.05]; % Initial state uncertainty cauchy scaling parameter(s)
    b0 = zeros(ndim,1); % Initial median of system state

    rng(10);
    num_steps = 7;

    % Simulate Dynamic System
    num_controls = 0;
    if num_controls > 0 
        B = randn(ndim, num_controls) ;
        us = randn(num_steps, num_controls);
    else
        B = [];
        us = []
    end

    x0_truth = p0 .* randn(ndim,1);
    [xs, zs, ws, vs] = simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma);

    % Testing Single Cauchy Estimator Instance
    cauchyEst = MSlidingWindowManager("lti", num_steps+1, false, false);
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
    z0 = zs(:,1);
    if num_controls > 0 
        u0 = zeros(num_controls, 1);
    else
        u0 = [];
    end
    cauchyEst.step(z0, uk); % Initial step without propagation

    for i = 2:length(zs)
        zk1 = zs(:, i);
        if num_controls > 0 
            uk = us(:, i-1);
        else
            uk = [];
        end
        cauchyEst.step(zk1, uk);
    end

    % Plot the results (if implemented in MATLAB)
    % plot_simulation_history(cauchyEst.moment_info, xs, zs, ws, vs)
    cauchyEst.shutdown();
end

% Creates a single Cauchy estimator instance and runs the estimator for several steps
function test_2state_lti_single_window()
    ndim = 2;
    Phi = [0.9, 0.1; -0.2, 1.1];
    Gamma = [0.1; 0.3];
    H = [1.0, 2.0];
    beta = [0.1]; % Cauchy process noise scaling parameter(s)
    gamma = [0.2]; % Cauchy measurement noise scaling parameter(s)
    A0 = eye(ndim); % Unit directions of the initial state uncertainty
    p0 = [0.10; 0.08]; % Initial state uncertainty cauchy scaling parameter(s)
    b0 = zeros(ndim, 1); % Initial median of system state

    rng(19);
    num_steps = 10;

    % Applying arbitrary controls to the estimator
    num_controls = 0;
    B = randn(ndim, num_controls) 
    if num_controls > 0 
        B = randn(ndim, num_controls);
        us = randn(num_steps, num_controls);
    else
        B = [];
        us = [];
    end

    % Simulate system states and measurements
    x0_truth = p0 .* randn(ndim,1);
    [xs, zs, ws, vs] = simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma);

    % Run Cauchy Estimator
    set_tr_search_idxs_ordering([1, 0]);
    cauchyEst = MSlidingWindowManager("lti", num_steps+1, false, false);
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
    z0 = zs(:,1);
    if num_controls > 0
        u0 = zeros(num_controls, 1);
    else
        u0 = [];
    end
    cauchyEst.step(z0, u0); % initial step

    for i = 2:length(zs)
        zk1 = zs(:, i);
        if num_controls > 0 
           uk = us(:, i-1);
        else
            uk = [];
        end
        cauchyEst.step(zk1, uk);
    end

    % Plotting function calls (implement depending on MATLAB plotting routines)
    % plot_2D_pointwise_cpdf(X, Y, Z);
    % plot_simulation_history(cauchyEst.moment_info, xs, zs, ws, vs);
    cauchyEst.shutdown();
end

function test_3state_lti_window_manager()
    % x_{k+1} = \Phi_k * x_k + B_k * u_k + \Gamma_k * w_k
    % z_k = H * x_k + v_k
    CAUCHY_TO_GAUSSIAN_NOISE = 1.3898;
    ndim = 3;
    Phi = [1.4, -0.6, -1.0; 
          -0.2, 1.0, 0.5;  
           0.6, -0.6, -0.2];
    Gamma = [0.1; 0.3; -0.2];
    H = [1.0, 0.5, 0.2];
    beta = 0.1; % Cauchy process noise scaling parameter(s)
    gamma = 0.2; % Cauchy measurement noise scaling parameter(s)
    A0 = eye(ndim); % Unit directions of the initial state uncertainty
    p0 = [0.10, 0.08, 0.05]; % Initial state uncertainty Cauchy scaling parameters
    b0 = zeros(ndim, 1); % Initial median of system state
    
    % Gaussian Noise Model Equivalent for KF
    W = diag((beta * CAUCHY_TO_GAUSSIAN_NOISE).^2);
    V = diag((gamma * CAUCHY_TO_GAUSSIAN_NOISE).^2);
    P0 = diag((p0 * CAUCHY_TO_GAUSSIAN_NOISE).^2);
    
    rng(10); % Set random seed
    num_steps = 150;
    
    % Simulate Dynamic System (Gaussian Noise)
    num_controls = 0;
    if num_controls > 0
        B = randn(ndim, num_controls);
        us = randn(num_steps, num_controls);
    else
        B = [];
        us = [];
    end
    
    x0_truth = p0' .* randn(ndim,1);
    [xs, zs, ws, vs] = simulate_gaussian_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, true, [], []);
    
    % Run Cauchy Estimator
    num_windows = 8;
    swm_debug_print = true;
    win_debug_print = false;
    cauchyEst = MSlidingWindowManager("lti", num_windows, swm_debug_print, win_debug_print);
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
    
    z0 = zs(1, :);
    if num_controls > 0
        u0 = zeros(1, num_controls);
    else
        u0 = [];
    end
    cauchyEst.step(z0, u0); % Initial step without propagation
    for i = 2:length(zs)
        zk1 = zs(i, :);
        if num_controls > 0
            uk = us(i - 1, :);
        else
            uk = [];
        end
        cauchyEst.step(zk1, uk);
    end
    cauchyEst.shutdown();
    
    % Run Kalman Filter
    x0_kf = b0;
    zs_kf = zs(2:end, :);
    [xs_kf, Ps_kf] = run_kalman_filter(x0_kf, us, zs_kf, P0, Phi, B, Gamma, H, W, V);
    
    % Plot results
    plot_simulation_history(cauchyEst.moment_info, {xs, zs, ws, vs}, {xs_kf, Ps_kf});
end

function test_2state_lti_window_manager()
    % x_{k+1} = \Phi_k * x_k + B_k * u_k + \Gamma_k * w_k
    % z_k = H * x_k + v_k
    ndim = 2;
    Phi = [0.9, 0.1; 
          -0.2, 1.1];
    Gamma = [0.1; 0.3];
    H = [1.0, 0.5];
    beta = 0.1; % Cauchy process noise scaling parameter(s)
    gamma = 0.2; % Cauchy measurement noise scaling parameter(s)
    A0 = eye(ndim); % Unit directions of the initial state uncertainty
    p0 = [0.10, 0.08]; % Initial state uncertainty Cauchy scaling parameters
    b0 = zeros(ndim, 1); % Initial median of system state
    CAUCHY_TO_GAUSSIAN_NOISE = 1.3898;
    % Gaussian Noise Model Equivalent for KF
    W = diag((beta * CAUCHY_TO_GAUSSIAN_NOISE).^2);
    V = diag((gamma * CAUCHY_TO_GAUSSIAN_NOISE).^2);
    P0 = diag((p0 * CAUCHY_TO_GAUSSIAN_NOISE).^2);
    
    rng(15); % Set random seed
    num_steps = 150;
    
    % Simulate Dynamic System (Gaussian Noise)
    num_controls = 0;
    if num_controls > 0
        B = randn(ndim, num_controls);
        us = randn(num_steps, num_controls);
    else
        B = [];
        us = [];
    end
    
    x0_truth = p0' .* randn(ndim,1);
    [xs, zs, ws, vs] = simulate_gaussian_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, true, [], []);
    
    % Run Cauchy Estimator
    num_windows = 6;
    swm_debug_print = true;
    win_debug_print = false;
    cauchyEst = MSlidingWindowManager("lti", num_windows, swm_debug_print, win_debug_print);
    cauchyEst.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
    
    z0 = zs(1, :);
    if num_controls > 0
        u0 = zeros(1, num_controls);
    else
        u0 = [];
    end
    cauchyEst.step(z0, u0); % Initial step without propagation
    for i = 2:length(zs)
        zk1 = zs(i, :);
        if num_controls > 0
            uk = us(i - 1, :);
        else
            uk = [];
        end
        cauchyEst.step(zk1, uk);
    end
    cauchyEst.shutdown();
    
    % Run Kalman Filter
    x0_kf = b0;
    zs_kf = zs(2:end, :);
    [xs_kf, Ps_kf] = run_kalman_filter(x0_kf, us, zs_kf, P0, Phi, B, Gamma, H, W, V);
    
    % Plot results
    plot_simulation_history(cauchyEst.moment_info, {xs, zs, ws, vs}, {xs_kf, Ps_kf});
end

function test_3state_marginal_cpdfs()
    % Define parameters
    ndim = 3;
    Phi = [1.4, -0.6, -1.0; 
           -0.2,  1.0,  0.5;  
            0.6, -0.6, -0.2];
    Gamma = [0.1; 0.3; -0.2];
    H = [1.0, 0.5, 0.2];
    beta = [0.1]; % Cauchy process noise scaling parameter(s)
    gamma = [0.2]; % Cauchy measurement noise scaling parameter(s)
    A0 = eye(ndim); % Unit directions of the initial state uncertainty
    p0 = [0.10; 0.08; 0.05]; % Initial state uncertainty cauchy scaling parameter(s)
    b0 = zeros(ndim,1); % Initial median of system state

    % Measurement sequence
    zs = [0.022172011200334241, -0.11943271347277583, -1.22353301003957098, ...
          -1.4055389648301792, -1.34053610027255954, 0.4580483915838776, ...
           0.65152999529515989, 0.52378648722334, 0.75198272983];
    num_steps = length(zs);

    % 2D Grid Parameters
    g2lx = -2; g2hx = 2; g2rx = 0.025;
    g2ly = -2; g2hy = 2; g2ry = 0.025;

    % 1D Grid Parameters
    g1lx = -4; g1hx = 4; g1rx = 0.001;
    
    % Initialize the Cauchy Estimator
    cauchyEst = MCauchyEstimator("lti", num_steps+1, true);
    cauchyEst.initialize_lti(A0, p0, b0, Phi, [], Gamma, beta, H, gamma);
    
    % Step through the measurements
    for i = 1:num_steps-2
        zk1 = zs(i);
        [xs, Ps] = cauchyEst.step(zk1, []);

        % Get 2D marginals
        [X01, Y01, Z01] = cauchyEst.get_marginal_2D_pointwise_cpdf(1, 2, g2lx, g2hx, g2rx, g2ly, g2hy, g2ry);
        [X02, Y02, Z02] = cauchyEst.get_marginal_2D_pointwise_cpdf(1, 3, g2lx, g2hx, g2rx, g2ly, g2hy, g2ry);
        [X12, Y12, Z12] = cauchyEst.get_marginal_2D_pointwise_cpdf(2, 3, g2lx, g2hx, g2rx, g2ly, g2hy, g2ry);

        % Get 1D marginals
        [x0, y0] = cauchyEst.get_marginal_1D_pointwise_cpdf(1, g1lx, g1hx, g1rx);
        [x1, y1] = cauchyEst.get_marginal_1D_pointwise_cpdf(2, g1lx, g1hx, g1rx);
        [x2, y2] = cauchyEst.get_marginal_1D_pointwise_cpdf(3, g1lx, g1hx, g1rx);
        
        % Plot 2D marginals
        figure('Position', [100, 100, 1800, 500]);
        subplot(1,3,1);
        surf(X01, Y01, Z01, 'EdgeColor', 'b');
        title('Marginal of States 1 and 2');
        xlabel('State 1'); ylabel('State 2'); zlabel('CPDF Probability');
        
        subplot(1,3,2);
        surf(X02, Y02, Z02, 'EdgeColor', 'g');
        title('Marginal of States 1 and 3');
        xlabel('State 1'); ylabel('State 3'); zlabel('CPDF Probability');

        subplot(1,3,3);
        surf(X12, Y12, Z12, 'EdgeColor', 'r');
        title('Marginal of States 2 and 3');
        xlabel('State 2'); ylabel('State 3'); zlabel('CPDF Probability');
        
        % Plot 1D marginals
        figure('Position', [100, 100, 1800, 400]);
        subplot(1,3,1);
        plot(x0, y0);
        title('1D Marginal of State 1');
        xlabel('State 1'); ylabel('CPDF Probability');

        subplot(1,3,2);
        plot(x1, y1);
        title('1D Marginal of State 2');
        xlabel('State 2'); ylabel('CPDF Probability');

        subplot(1,3,3);
        plot(x2, y2);
        title('1D Marginal of State 3');
        xlabel('State 3'); ylabel('CPDF Probability');
        waitfor(gcf)
        
    end
end
