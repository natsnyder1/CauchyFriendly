addpath("matlab_pure");
addpath("mex_files");
addpath(pwd);


% PendulumParams
global pend; % Lets just make a simple globally viewable object to get ahold of these parameters when we want them
pend = struct(...
    'L', 0.3, ...        % meters
    'g', 9.81, ...       % meters / second^2
    'c', 0.6, ...        % 1/seconds (damping)
    'w_PSD', 0.01, ...   % power spectral density of continuous-time process noise
    'dt', 0.05 ...       % integration step time
);

theta_vec0 = [pi/4; 0]; % initial angle of 45 degrees at 0 radians/sec
theta_k = theta_vec0;
thetas = theta_k';
propagations = 160;
for k = 1:propagations
    theta_k = nonlin_transition_model(theta_k);
    thetas = [thetas; theta_k'];
end
Ts = ((0:propagations) * pend.dt)';
% figure;
% sgtitle('Pendulum Trajectory (angle: top), (angular rate: bottom)');
% subplot(2, 1, 1);
% plot(Ts, thetas(:, 1));
% subplot(2, 1, 2);
% plot(Ts, thetas(:, 2));


% Creating the dynamic simulation
V = 0.003; % measurement noise on theta
H = [1.0, 0.0]; % meausrement model
xk = theta_vec0;
xs = xk'; % State vector history
ws = [];   % Process noise history
vs = sqrt(V(1)) * randn(); % Measurement noise history
zs = H * xk + vs(1); % Measurement history
propagations = 160;
for k = 1:propagations
    wk = pend.dt * sqrt(pend.w_PSD) * randn();
    xk(2) = xk(2) + wk;
    xk = nonlin_transition_model(xk);
    xs = [xs; xk'];
    ws = [ws; wk];
    vk = sqrt(V(1)) * randn();
    zk = H * xk + vk;
    vs = [vs; vk];
    zs = [zs; zk];
end
%plot_simulation_history([], {xs,zs,ws,vs}, [])
% Now we have our simulation data!


% Continuous time Gamma (\Gamma_c)
Gamma_c = [0.0; 1.0];
W_c = pend.w_PSD;
I2 = eye(2);
taylor_order = 2;

% Setting up and running the EKF
% The gaussian_filters module has a "run_ekf" function baked in, but we'll just show the whole thing here
P0_kf = eye(2) * 0.003;
%x0_kf = mvnrnd(theta_vec0, P0_kf); % lets initialize the Kalman filter slightly off from the true state position
x0_kf = zeros(size(theta_vec0))';

xs_kf = x0_kf;
Ps_kf = zeros(propagations+1, 2, 2);
Ps_kf(1, :, :) = P0_kf;
x_kf = x0_kf';
P_kf = P0_kf;
for k = 1:propagations
    Jac_F = jacobian_pendulum_ode(x_kf);
    [Phi_k, W_k] = discretize_nl_sys(Jac_F, Gamma_c, W_c, pend.dt, taylor_order, false, true);
    % Propagate covariance and state estimates
    P_kf = Phi_k * P_kf * Phi_k' + W_k;
    x_kf = nonlin_transition_model(x_kf);
    % Form Kalman Gain, update estimate and covariance
    K = (H * P_kf * H' + V) \ (H * P_kf)';
    zbar = H * x_kf;
    r = zs(k+1) - zbar;
    x_kf = x_kf + K * r;
    P_kf = (I2 - K * H) * P_kf * (I2 - K * H)' + K * V * K';
    % Store estimates
    xs_kf = [xs_kf; x_kf'];
    Ps_kf(k, :, :) = P_kf;
end
% Plot Simulation results 
%plot_simulation_history([], {xs,zs,ws,vs}, {xs_kf, Ps_kf});

scale_g2c = 1.0 / 1.3898; % scale factor to fit the cauchy to the gaussian
beta = sqrt(pend.w_PSD / pend.dt) * scale_g2c;
gamma = sqrt(V(1, 1)) * scale_g2c;
x0_ce = x0_kf;

A0 = eye(2);
p0 = sqrt(diag(P0_kf)) * scale_g2c;
b0 = zeros(2, 1);
steps = 5;
num_controls = 0;
print_debug = true;
cauchyEst = MCauchyEstimator("nonlin", steps, print_debug);
cauchyEst.initialize_nonlin(x0_ce, A0, p0, b0, beta, gamma, 'dynamics_update', 'nonlinear_msmt_model', 'msmt_model_jacobian', num_controls, pend.dt)
cauchyEst.step(zs(1));
cauchyEst.step(zs(2));
cauchyEst.step(zs(3));
cauchyEst.step(zs(4));
cauchyEst.step(zs(5));
cauchyEst.shutdown();

swm_print_debug = false; 
win_print_debug = false;
num_windows = 6;
new_beta = beta / 5; % tuned down
cauchyEst = MSlidingWindowManager("nonlin", num_windows, swm_print_debug, win_print_debug);
cauchyEst.initialize_nonlin(x0_ce, A0, p0, b0, new_beta, gamma, 'dynamics_update', 'nonlinear_msmt_model', 'msmt_model_jacobian', num_controls, pend.dt);

for k = 1:length(zs)
    zk = zs(k);
    cauchyEst.step(zk, []);
end
cauchyEst.shutdown()

plot_simulation_history(cauchyEst.moment_info, {xs,zs,ws,vs}, {xs_kf, Ps_kf} )
%ce.plot_simulation_history( cauchyEst.avg_moment_info, (xs,zs,ws,vs), (xs_kf, Ps_kf) )

% The ODE
function dx_dt = pend_ode(x)
    global pend;
    dx_dt = zeros(2, 1);
    dx_dt(1) = x(2);
    dx_dt(2) = -pend.g / pend.L * sin(x(1)) - pend.c * x(2);
end

% Nonlinear transition model from t_k to t_k+1...ie: dt
function x_new = nonlin_transition_model(x)
    global pend;
    x_new = runge_kutta4(@pend_ode, x, pend.dt);
end

% This is the callback function correpsonding to the decription for point 1.) above 
function foobar_dynamics_update(c_duc)
    mduc = M_CauchyDynamicsUpdateContainer(c_duc);
    %% Propagate x 
    %x = mduc.cget_x()
    %u = mduc.cget_u()
    %xbar <- f(x,u)
    %mduc.cset_x(xbar)
    %mduc.cset_is_xbar_set_for_ece() % need to call this!
    %% Phi, Gamma, beta may update
    %mduc.cset_Phi(Phi)
    %mduc.cset_Gamma(Gamma)
    %mduc.cset_beta(beta)
end

% This is the callback function correpsonding to the decription for point 2.) above 
function foobar_nonlinear_msmt_model(c_duc, c_zbar)
    mduc = M_CauchyDynamicsUpdateContainer(c_duc);
    %% Set zbar
    %x = mduc.cget_x() % xbar
    %zbar = msmt_model(x)
    %mduc.cset_zbar(c_zbar, zbar)
end

% This is the callback function correpsonding to the decription for point 3.) above 
function foobar_msmt_model_jacobian(c_duc)
    mduc = M_CauchyDynamicsUpdateContainer(c_duc);
    %% H and gamma may update
    %x = mduc.cget_x() % xbar
    % H <- jacobian( h(x) )
    %mduc.cset_H(H)
    %mduc.cset_gamma(gamma)
end

% This is the callback function correpsonding to the decription for point 1.) above 
function dynamics_update(c_duc)
    global pend;
    mduc = M_CauchyDynamicsUpdateContainer(c_duc);
    %% Propagate x 
    xk = mduc.cget_x();
    xbar = nonlin_transition_model(xk); % propagate from k -> k+1
    mduc.cset_x(xbar);
    mduc.cset_is_xbar_set_for_ece(); % need to call this!
    %% Phi, Gamma, beta may update
    Jac_F = jacobian_pendulum_ode(xk);
    [Phi_k, Gam_k] = discretize_nl_sys(Jac_F, Gamma_c, [], pend.dt, taylor_order, true, false);
    mduc.cset_Phi(Phi_k);
    mduc.cset_Gamma(Gam_k);
    %mduc.cset_beta(beta)
end

% This is the callback function correpsonding to the decription for point 2.) above 
function nonlinear_msmt_model(c_duc, c_zbar)
    mduc = M_CauchyDynamicsUpdateContainer(c_duc);
    %% Set zbar
    xbar = mduc.cget_x(); % xbar
    zbar = H @ xbar; % for other systems, call your nonlinear h(x) function
    mduc.cset_zbar(c_zbar, zbar);
end

% This is the callback function correpsonding to the decription for point 3.) above 
function msmt_model_jacobian(c_duc)
    mduc = M_CauchyDynamicsUpdateContainer(c_duc);
    %% Set H: for other systems, call your nonlinear jacobian function H(x)
    mduc.cset_H(H); % we could write some if condition to only set this once, but its such a trivial overhead, who cares
end

function Jac = jacobian_pendulum_ode(x)
    global pend;
    Jac = zeros(2);
    Jac(1,2) = 1;
    Jac(2,1) = -pend.g/pend.L*cos(x(1));
    Jac(2,2) = -pend.c;
end
