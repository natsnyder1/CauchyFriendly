% file : simulate_gaussian_ltiv_system.m 

function [xs, zs, ws, vs] = simulate_gaussian_ltiv_system(num_steps, x0_truth, us, Phi, B, Gamma, W, H, V, with_zeroth_step_msmt, dynamics_update_callback, other_params)
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
    
    if numel(W) == 1
        W = reshape(W, 1, 1);
    else
        assert(size(W,1) == size(W,2));
    end
    
    if numel(V) == 1
        V = reshape(V, 1, 1);
    else
        assert(size(V,1) == size(V,2));
    end
    
    assert(~isempty(Gamma));
    assert(~isempty(W));
    Gamma = reshape(Gamma, numel(x0_truth), size(W,1));
    v_zero_vec = zeros(size(V,1), 1);
    w_zero_vec = zeros(size(W,1), 1);
    xs = {x0_truth};
    zs = {};
    ws = {};
    vs = {};
    
    xk = x0_truth;
    
    if with_zeroth_step_msmt
        v0 = mvnrnd(v_zero_vec, V)';
        zs{end+1} = H * xk + v0;
        vs{end+1} = v0;
    end
    
    for i = 1:num_steps
        if ~isempty(dynamics_update_callback)
            dynamics_update_callback(Phi, B, Gamma, H, W, V, i - 1, other_params);
        end
        uk = us(i,:)';
        wk = mvnrnd(w_zero_vec, W)';
        xk = Phi * xk + B * uk + Gamma * wk;
        xs{end+1} = xk;
        ws{end+1} = wk;
        vk = mvnrnd(v_zero_vec, V)';
        zs{end+1} = H * xk + vk;
        vs{end+1} = vk;
    end
    
    % Convert cell arrays to matrices
    xs = cell2mat(xs')';
    zs = cell2mat(zs')';
    ws = cell2mat(ws')';
    vs = cell2mat(vs')';
end