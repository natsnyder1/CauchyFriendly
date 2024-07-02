% file : test_function.m 
% this test function is an alternate version of simulate_gaussian_ltiv_system that is only used for testing

function [xs, zs, ws, vs] = test_function(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, with_zeroth_step_msmt, dynamics_update_callback, other_params)
    data = load('random_noise_vectors.mat');
    ws = data.ws;
    vs = data.vs;
    if nargin < 10
        with_zeroth_step_msmt = true;
    end
    if nargin < 11
        dynamics_update_callback = [];
    end
    
    if ~isempty(B)
        assert(~isempty(us));
        assert(size(us,1) == num_steps);
        if numel(B) == numel(x0_truth)
            us = reshape(us, num_steps, 1);
        else
            assert(size(us,2) == size(B,2));
        end
    else
        assert(isempty(us));
        B = zeros(length(x0_truth), 1);
        us = zeros(num_steps, 1);
    end
    
    if isscalar(W)
        W = reshape(W, 1, 1);
    else
        assert(size(W,1) == size(W,2));
    end
    
    if isscalar(V)
        V = reshape(V, 1, 1);
    else
        assert(size(V,1) == size(V,2));
    end
    
    assert(~isempty(Gamma));
    assert(~isempty(W));
    Gamma = reshape(Gamma, numel(x0_truth), size(W,1));
    xk = x0_truth;
    
    xs = xk'; % Initialize with the first state
    zs = []; % Initialize measurement history
    
    if with_zeroth_step_msmt
        v0 = vs(:, 1); % Use the first column (Python vs[:, 0] equivalent)
        z0 = H' * xk + v0;
        zs = [zs, z0];
    end

    for i = 1:num_steps
        if ~isempty(dynamics_update_callback)
            dynamics_update_callback(Phi, B, Gamma, H, W, V, i - 1, other_params);
        end
        uk = us(i,:)';
        wk = ws(:, i); % Use the preloaded process noise
        xk = Phi * xk + B * uk + Gamma * wk;

        xs = [xs; xk']; % Concatenate to states matrix
        if with_zeroth_step_msmt
            vk = vs(:, i+1); 
        else
            vk = vs(:, i); 
        end
        zk = H' * xk + vk;
        
        zs = [zs; zk]; % Concatenate to measurement matrix

    end
    ws = ws';
    vs = vs';
end