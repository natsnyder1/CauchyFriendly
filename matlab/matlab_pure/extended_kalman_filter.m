% file: extended_kalman_filter

% Runs a full step of the extended kalman filter: updates from k-1|k-1 to k|k
% x is the state vector (of step k-1)
% u is the control vector (of step k-1)
% msmt is the newest measurement (of step k)
% f is the discrete time non-linear system dynamics (maps k-1|k-1 -> k|k-1)
% h is the (possibly) non-linear measurement matrix (used in updating k|k-1 -> k|k)
% callback_Phi_Gam is a callback function which returns the state transition matrix and noise gain matrix, with arguments x (of step k-1) and control (of step k-1)
% callback_H = forms the H matrix -> FUNCTION with argument xbar (of step k|k-1)
% P is the posterior covariance of k-1|k-1, to be updated now to k|k
% W is the process noise matrix describing the current step k-1|k-1 -> k|k-1
% V is the measurement noise matrix describing the step k|k-1 -> k|k
function [xhat, P] = extended_kalman_filter(x, u, msmt, f, h, callback_Phi_Gam, callback_H, P, W, V, other_params)
    assert(all(size(W) == [length(x), length(x)]));
    assert(all(size(V) == [size(msmt, 1), size(msmt, 1)]));

    [Phi, Gam] = callback_Phi_Gam(x, u, other_params);
    xbar = f(x, u, other_params);

    M = Phi * P * Phi' + Gam * W * Gam';

    H = callback_H(xbar, other_params);
    K = M * H' / (H * M * H' + V);

    r = msmt - h(xbar, other_params);
    xhat = xbar + K * r;

    I = eye(length(x));
    P = (I - K * H) * M * (I - K * H)' + K * V * K';
end