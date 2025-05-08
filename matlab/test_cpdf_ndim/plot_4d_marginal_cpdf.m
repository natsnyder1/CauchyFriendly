zs = [-0.2630, -0.9829, -0.9332, -0.8131, -0.2414, 0.0140, -0.4842, -0.7607];
marg_state_idxs = [0, 1];  % x1 and x2

% Grid2D: shape = [2 x N] (row: axis, col: point)
grid2D = [ -0.5 -0.5 -0.5 -0.4 -0.4 -0.4 -0.3 -0.3 -0.3 -0.2 0.0;
           -0.1  0.0  0.1 -0.1  0.0  0.1 -0.1  0.0  0.1  0.0 0.0 ];

real_fx = test_4d_marginal_cpdf(zs, grid2D, marg_state_idxs);  % [points x steps-1]

num_points = size(grid2D, 2);
num_steps = length(zs) - 1;

for t = 1:num_steps
    fx_t = real_fx(:, t);

    figure;
    scatter3(grid2D(1,:), grid2D(2,:), fx_t, 50, fx_t, 'filled');
    xlabel(sprintf('x_{%d}', marg_state_idxs(1)+1));
    ylabel(sprintf('x_{%d}', marg_state_idxs(2)+1));
    zlabel('CPDF');
    title(sprintf('Step %d', t));
    colorbar;
    axis tight;
    view(45, 25);
end