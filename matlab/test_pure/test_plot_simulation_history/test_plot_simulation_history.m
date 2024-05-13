% file: test_plot_simulation_history.m

gs_outputs = load('../test_simulate_gaussian_ltiv_system/gaussian_simulation_outputs.mat');
xs = gs_outputs.xs_matlab;
zs = gs_outputs.zs_matlab;
ws = gs_outputs.ws_matlab;
vs = gs_outputs.vs_matlab;

kf_outputs = load('../test_run_kalman_filter/kalman_filter_outputs.mat');
xs_kf = kf_outputs.xs_kf_matlab;
Ps_kf = kf_outputs.Ps_kf_matlab;

addpath('../../matlab_pure');
plot_simulation_history([], {xs,zs,ws,vs}, {xs_kf, Ps_kf})
rmpath('../../matlab_pure');