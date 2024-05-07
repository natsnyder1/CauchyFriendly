% file : run_kalman_filter.m 

% Runs a simulation of a linear time (in)variant kalman filter
function [xs_kf, Ps_kf, rs_kf, Ks_kf] = run_kalman_filter(x0, us, msmts, P0, Phi, B, Gam, H, W, V, dynamics_update_callback, other_params, is_debug)
    if nargin < 11
        dynamics_update_callback = [];
    end
    if nargin < 12
        other_params = [];
    end
    if nargin < 13
        is_debug = false;
    end

    number_of_steps = size(msmts, 2);
    state_dimension = numel(x0);
    
    % Pre-allocate storage for results
    xs_kf = zeros(state_dimension, number_of_steps + 1); % +1 for initial state
    Ps_kf = zeros(state_dimension, state_dimension, number_of_steps + 1); % +1 for initial state
    if is_debug
        rs_kf = zeros(size(msmts, 1), number_of_steps);
        Ks_kf = zeros(state_dimension, size(msmts, 1), number_of_steps);
    end
    
    % Intial conditions
    xs_kf(:, 1) = x0;
    Ps_kf(:, :, 1) = P0;
    x = x0;
    P = P0;
    
    % Handle the case where the control input 'us' is empty
    if isempty(us) && ~isempty(B)
        us = zeros(size(B, 2), number_of_steps); % Control input is zero if empty
    elseif isempty(B)
        B = zeros(state_dimension, 1); % Control matrix is zero if empty
    end

    for i = 1:number_of_steps
        % Extract control input 'u' if available, otherwise use zero input
        if isempty(us)
            u = zeros(size(B, 2), 1);
        else
            u = us(:, i);
        end
        z = msmts(:, i); % Measurement for current step

        % Optional callback for updating dynamics matrices
        if ~isempty(dynamics_update_callback)
            dynamics_update_callback(x, u, Phi, B, Gam, H, W, V, other_params);
        end

        % Kalman filter update step
        [x, P, r, K] = kalman_filter(x, u, z, P, Phi, B, Gam, H, W, V);

        % Store results
        xs_kf(:, i + 1) = x;
        Ps_kf(:, :, i + 1) = P;
        
        if is_debug
            rs_kf(:, i) = r;
            Ks_kf(:, :, i) = K;
        end
    end

    if ~is_debug
        % If debug information is not required, discard 'rs_kf' and 'Ks_kf'.
        rs_kf = [];
        Ks_kf = [];
    end
end