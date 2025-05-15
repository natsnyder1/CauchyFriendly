grid_size = 50;
x1_low = -1.0;
x1_high = 0.5;
x2_low = -0.5;
x2_high = 1.0;

zs = [-0.2630, -0.9829, -0.9332, -0.8131, -0.2414, 0.0140, -0.4842, -0.7607];
marg_state_idxs = [0, 1];  % x1 and x2

x1_vals = linspace(x1_low, x1_high, grid_size);
x2_vals = linspace(x2_low, x2_high, grid_size);

[X1, X2] = meshgrid(x1_vals, x2_vals);

grid2D = [X1(:)'; X2(:)'];  % Shape: [2 x 2500]


real_fx = test_4d_marginal_cpdf(zs, grid2D, marg_state_idxs);  % [points x steps-1]

num_points = size(grid2D, 2);
num_steps = length(zs) - 1;
num_x1 = length(x1_vals);
num_x2 = length(x2_vals);

for t = 1:num_steps
    fx_t = reshape(real_fx(:,t), [num_x2, num_x1]);

    figure;
    surf(x1_vals, x2_vals, fx_t, 'EdgeColor', 'none');
    xlabel(sprintf('x_{%d}', marg_state_idxs(1)+1));
    ylabel(sprintf('x_{%d}', marg_state_idxs(2)+1));
    zlabel('CPDF');
    title(sprintf('Step %d', t));
    colorbar;
    view(45, 25);  % 3D view
end