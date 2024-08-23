% file: simulate_cauchy_ltiv_system.m

function [xs, zs, ws, vs] = simulate_cauchy_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, beta, H, gamma, with_zeroth_step_msmt, dynamics_update_callback, other_params)
    if nargin < 10 
        with_zeroth_step_msmt = true;
    end
    if nargin < 11
        dynamics_update_callback = [];
    end
    if nargin < 12
        other_params = [];
    end
    
    if ~isempty(B)
        assert(~isempty(us), 'us must not be empty');
        assert(size(us, 1) == num_steps, 'us should have num_steps rows');
        if numel(B) == numel(x0_truth)
            us = reshape(us, num_steps, 1);
            B = reshape(B, numel(x0_truth), 1);
        else
            assert(size(us, 2) == size(B, 2), 'The second dimension of us and B must match.');
        end
    else
        assert(isempty(us), 'us must be empty');
        B = zeros(numel(x0_truth), 1);
        us = zeros(num_steps, 1);
    end
    assert(~isempty(Gamma), 'Gamma must not be empty');
    assert(~isempty(beta), 'beta must not be empty');
    Gamma = reshape(Gamma, numel(x0_truth), numel(beta));

    xs = x0_truth;
    zs = [];
    ws = [];
    vs = [];

    xk = x0_truth;
    if with_zeroth_step_msmt
        v0 = arrayfun(@(g) random('Stable', 1, 0, g, 0), gamma);
        zs = H * xk + v0;
        vs = v0;
    end
   
    for i = 1:num_steps
        if ~isempty(dynamics_update_callback)
            dynamics_update_callback(Phi, B, Gamma, H, beta, gamma, i, other_params);
        end
        uk = us(i, :)';
        wk = arrayfun(@(b) random('Stable', 1, 0, b, 0), beta);
        xk = Phi * xk + B * uk + Gamma * wk;
        xs(:, end+1) = xk;
        ws(:, end+1) = wk;
        vk = arrayfun(@(g) random('Stable', 1, 0, g, 0), gamma);
        zs(:, end+1) = H * xk + vk;
        vs(:, end+1) = vk;
    end

    xs = xs';
    zs = zs';
    ws = ws';
    vs = vs';
end
