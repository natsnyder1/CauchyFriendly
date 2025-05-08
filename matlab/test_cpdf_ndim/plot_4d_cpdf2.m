Phi = [1.4, -0.6, -1.0, 0.0; 
       -0.2, 1.0, 0.5, 0.0; 
        0.6, -0.6, -0.2, 0.0; 
        0, 0, 0, 0.5];

Gamma = [.1; 0.3; -0.2; 0.4];
H = [2.0; 0.5; 0.2; -0.1];
beta = 0.1;
gamma = 0.2;
A0 = eye(4);
p0 = [0.1; 0.08; 0.05; 0.2];
b0 = [0; 0; 0; 0];
zs = [-0.2630, -0.9829, -0.9332, -0.8131, -0.2414, 0.0140];
grid4D = [
    0,0,0,0;
   -0.05,0,0,0;
    0.05,0,0,0;
    0,-0.05,0,0;
    0,0.05,0,0;
    0,0,-0.05,0;
    0,0,0.05,0;
    0,0,0,0.05;
    0,0,0,-0.05
];


[real_fx, ~] = test_4d_cpdf(Phi, Gamma, H, beta, gamma, A0, p0, b0, zs, grid4D);
fx_mat = reshape(real_fx, [9, 5]);  % (grid_points, steps)

x_vals = [-0.05, 0, 0.05];

% Indices for perturbations per variable
idx_x1 = [2,1,3];
idx_x2 = [4,1,5];
idx_x3 = [6,1,7];
idx_x4 = [9,1,8];

figure('Units','normalized','Position',[0.1, 0.1, 0.9, 0.8]);

for t = 1:5
    % Row 1: x1
    subplot(4,5,t);
    plot(x_vals, fx_mat(idx_x1, t), 'o-', 'LineWidth', 2);
    title(['Step ', num2str(t)]);
    ylabel('x_1'); grid on;
    
    % Row 2: x2
    subplot(4,5,t + 5);
    plot(x_vals, fx_mat(idx_x2, t), 's-', 'LineWidth', 2);
    ylabel('x_2'); grid on;
    
    % Row 3: x3
    subplot(4,5,t + 10);
    plot(x_vals, fx_mat(idx_x3, t), '^-', 'LineWidth', 2);
    ylabel('x_3'); grid on;
    
    % Row 4: x4
    subplot(4,5,t + 15);
    plot(x_vals, fx_mat(idx_x4, t), 'd-', 'LineWidth', 2);
    xlabel('Perturbation'); ylabel('x_4'); grid on;
end

sgtitle('CPDF Evolution: Each Axis Perturbation Across Time Steps');
