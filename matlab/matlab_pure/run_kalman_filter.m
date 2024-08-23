% file : run_kalman_filter.m 

% Runs a simulation of a linear time (in)variant kalman filter
function [xs_kf, Ps_kf, rs_kf, Ks_kf] = run_kalman_filter(x0, us, msmts, P0, Phi, B, Gam, H, W, V, dynamics_update_callback, other_params, is_debug)
    % Default values for optional arguments
    if nargin < 11
        dynamics_update_callback = [];
    end
    if nargin < 12
        other_params = [];
    end
    if nargin < 13
        is_debug = false;
    end

    Gam = Gam';
    
    number_of_steps = size(msmts, 1);
    state_dimension = numel(x0);
    
    % Pre-allocate storage for results
    xs_kf = zeros(number_of_steps + 1, state_dimension); % +1 for initial state
    Ps_kf = zeros(number_of_steps + 1, state_dimension, state_dimension); 
    % Only allocate rs_kf and Ks_kf if debugging is enabled
    rs_kf = [];
    Ks_kf = [];
    if is_debug
        rs_kf = zeros(number_of_steps, size(H, 2));
        Ks_kf = zeros(number_of_steps, state_dimension, size(H, 2)); % Account for multiple dimensions
    end
    
    % Intial conditions
    xs_kf(1, :) = x0'; % Initial state transposed
    Ps_kf(1, :, :) = P0; 
    x = x0;
    P = P0;
    
    % Handle the case where the control input 'us' is empty
    if isempty(us)
        us = zeros(size(B, 2), number_of_steps); % Control input is zero if empty
    end
    if isempty(B)
        B = zeros(state_dimension, size(us, 1)); % Control matrix is zero if empty, size fixed
    end

    for i = 1:number_of_steps
        u = us(:,i);
        z = msmts(i, :); % Measurement for current step

        % Optional callback for updating dynamics matrices
        if ~isempty(dynamics_update_callback)
            dynamics_update_callback(x, u, Phi, B, Gam, H, W, V, other_params);
        end

        % Kalman filter update step
        [x, P, r, K] = kalman_filter(x, u, z', P, Phi, B, Gam, H, W, V);
        xs_kf(i + 1, :) = x'; % State transposed
        Ps_kf(i + 1, :, :) = P;
        if is_debug
            rs_kf(i, :) = r'; % Residual transposed
            Ks_kf(i, :, :) = K'; % Kalman gain transposed
        end
    end
end