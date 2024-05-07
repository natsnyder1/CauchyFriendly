% file : kalman_filter.m 

function [xhat, P, r, K] = kalman_filter(x, u, msmt, P, Phi, B, Gam, H, W, V)
    % Ensure vectors are columns and matrices are at least 2D if necessary
    x = reshape(x, numel(x), 1);
    u = reshape(u, numel(u), 1);
    msmt = reshape(msmt, numel(msmt), 1);

    W = reshape(W, size(W, 1), size(W, 2));
    V = reshape(V, size(V, 1), size(V, 2));
    
    % Propagate Dynamics
    xbar = Phi * x + B * u;
   
    % A Priori Covariance Matrix
    M = Phi * P * Phi' + Gam' * W * Gam;
    
    % Update Kalman Gain
    K = M * H' / (H * M * H' + V);
    
    % Find the conditional mean estimate
    r = msmt - H * xbar;
    xhat = xbar + K * r;
    
    % Posteriori Covariance Matrix
    I = eye(length(x));
    P = (I - K * H) * M * (I - K * H)' + K * V * K';
end

