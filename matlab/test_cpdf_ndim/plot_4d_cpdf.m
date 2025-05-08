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
fx_mat = reshape(real_fx, [9, 5]); % (grid_points, steps)

% Grid perturbations for axis labels
x_vals = [-0.05, 0, 0.05];

% Define indices for each axis perturbation
x1_idx = [2,1,3];
x2_idx = [4,1,5];
x3_idx = [6,1,7];
x4_idx = [9,1,8];

figure;
for t = 1:5
    subplot(1,5,t);
    plot(x_vals, fx_mat(x1_idx, t), '-o', 'DisplayName','x₁');
    hold on;
    plot(x_vals, fx_mat(x2_idx, t), '-s', 'DisplayName','x₂');
    plot(x_vals, fx_mat(x3_idx, t), '-^', 'DisplayName','x₃');
    plot(x_vals, fx_mat(x4_idx, t), '-d', 'DisplayName','x₄');
    title(['Step ', num2str(t)]);
    xlabel('Perturbation Value');
    ylabel('CPDF');
    legend;
    grid on;
end
sgtitle('CPDF evolution over perturbations and time');

