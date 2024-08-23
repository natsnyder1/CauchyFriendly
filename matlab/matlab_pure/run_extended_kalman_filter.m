% file: run_extended_kalman_filter

% Runs a simulation of a linear time invariant kalman filter
% P0 is initial covariance of system
function [xs, Ps] = run_extended_kalman_filter(x0, us, msmts, f, h, callback_Phi_Gam, callback_H, P0, W, V, other_params)
    x = x0;
    P = P0;
    xs = x0;
    Ps = P0;
    
    if isempty(us)
        us = zeros(size(msmts, 1), 1);
    else
        assert(size(us, 1) == size(msmts, 1));
    end
    
    for i = 1:size(msmts, 1)
        u = us(i, :).';
        z = msmts(i, :).';
        [x, P] = extended_kalman_filter(x, u, z, f, h, callback_Phi_Gam, callback_H, P, W, V, other_params);
        xs = [xs, x];
        Ps = cat(3, Ps, P);
    end
end