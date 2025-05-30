grid_size = 70; % Grid is grid_size x grid_size
x1_low = -2.5;
x1_high = 1.0;
x2_low = -2.0;
x2_high = 1.5;

% Measurements
% No double peak
%zs = [-0.3630, -0.5829, -0.6332, -0.5131, -0.4414, -0.3140, -0.4842];

% Double peak at step 2
zs = [-0.3630, -1.4829, -0.6332, -0.5131, -0.4414, 0.3140, -0.4842];

marg_state_idxs = [0, 1];  % x1 and x2

% Define the grid
x1_vals = linspace(x1_low, x1_high, grid_size);
x2_vals = linspace(x2_low, x2_high, grid_size);

[X1, X2] = meshgrid(x1_vals, x2_vals);

grid2D = [X1(:)'; X2(:)'];  % Shape: [2 x 2500]


% input parameters
n = 4; cmcc = 0; pncc = 1; p = 1;

Phi = [1.4, -0.6, -1.0, 0.0;
      -0.2,  1.0,  0.5, 0.0;
       0.6, -0.6, -0.2, 0.0;
       0.0,  0.0,  0.0, 0.5];

Gamma = [0.1; 0.3; -0.2; 0.4];
H = [2.0; 0.5; 0.2; -0.1];
beta = 0.1;
gamma = 0.1;
A0 = eye(n);
p0 = [0.1; 0.08; 0.05; 0.2];
b0 = zeros(n, 1);

real_fx = test_4d_marginal_cpdf(zs, grid2D, marg_state_idxs, ...
    Phi, Gamma, H, beta, gamma, A0, p0, b0, n, cmcc, pncc, p);


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
    view(75, 25);  % 3D view
end