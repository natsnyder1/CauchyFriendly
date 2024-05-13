% file: plot_simulation_history.m

function plot_simulation_history(cauchy_moment_info, simulation_history, kf_history, with_partial_plot, with_cauchy_delay, scale)

    if nargin < 4
        with_partial_plot = false;
    end
    
    if nargin < 5
        with_cauchy_delay = false;
    end
    
    if nargin < 6
        scale = 1;
    end
    
    with_sim = ~isempty(simulation_history);
    with_kf = ~isempty(kf_history);
    with_ce = ~isempty(cauchy_moment_info);
    
    if with_sim
        n = size(simulation_history{1}, 2);
        T = 0:size(simulation_history{2}, 1)-1;
    elseif with_kf
        n = size(kf_history{1}, 2);
        T = 0:size(kf_history{1}, 1)-1;
    elseif with_ce
        n = numel(cauchy_moment_info.x{1});
        T = 0:numel(cauchy_moment_info.x)-1;
        if with_cauchy_delay
            T = T + with_cauchy_delay;
        end
    else
        fprintf('Must provide simulation data, kalman filter data or cauchy estimator data!\nExiting function with no plotting (Nothing Given!)\n');
        return;
    end
    
    % Simulation history
    if with_sim
        true_states = simulation_history{1};
        msmts = simulation_history{2};
        proc_noises = simulation_history{3};
        msmt_noises = simulation_history{4};
    end
    
    % Cauchy Estimator
    if with_ce
        means = cell2mat(cauchy_moment_info.x);
        covars = cell2mat(cauchy_moment_info.P);
        cerr_norm_factors = cell2mat(cauchy_moment_info.cerr_fz);
        cerr_means = cell2mat(cauchy_moment_info.cerr_x);
        cerr_covars = cell2mat(cauchy_moment_info.cerr_P);
        n = size(means, 2);
    end
    
    % Kalman filter
    if with_kf
        kf_cond_means = kf_history{1};
        kf_cond_covars = kf_history{2};
    end
    
    % Check array lengths, cauchy_delay, partial plot parameters
    cd = 0;
    if with_ce
        cd = with_cauchy_delay;
        %plot_len variable has been introduced so that runs which fail can still be partially plotted
        plot_len = size(covars, 1) + cd;
        if ~with_partial_plot && with_cauchy_delay && plot_len ~= numel(T)
            fprintf('[ERROR]: covars.shape[0] + with_cauchy_delay != T.size. You have mismatch in array lengths\n');
            fprintf('Cauchy Covars size: '); disp(size(covars));
            fprintf('with_cauchy_delay: %d\n', with_cauchy_delay);
            fprintf('T size: %d\n', numel(T));
            fprintf('Please fix appropriately!\n');
            assert(false);
        elseif (with_partial_plot || with_cauchy_delay) && plot_len > numel(T)
            fprintf('[ERROR]: covars.shape[0] + with_cauchy_delay > T.size. You have mismatch in array lengths\n');
            fprintf('Cauchy Covars size: '); disp(size(covars));
            fprintf('with_cauchy_delay: %d\n', with_cauchy_delay);
            fprintf('T size: %d\n', numel(T));
            fprintf('Please fix appropriately!\n');
            assert(false);
        elseif (size(covars, 1) + cd) ~= numel(T)
            fprintf('[ERROR]: covars.shape[0] + with_cauchy_delay != T.size. You have mismatch in array lengths\n');
            fprintf('Cauchy Covars size: '); disp(size(covars));
            fprintf('with_cauchy_delay: %d\n', with_cauchy_delay);
            fprintf('T size: %d\n', numel(T));
            fprintf('Please toggle on ''p'' option for partial plotting or set ''d'' to lag cauchy estimator appropriately\n');
            assert(false);
        else
            plot_len = numel(T);
        end
    else
        plot_len = numel(T);
    end
    
    % 1.) Plot the true state history vs the conditional mean estimate  
    % 2.) Plot the state error and one-sigma bound of the covariance 
    % 3.) Plot the msmts, and the msmt and process noise 
    % 4.) Plot the max complex error in the mean/covar and norm factor 
    figure;
    hold on;
    title('True States (r) vs Cauchy (b) vs Kalman (g--)');
    for i = 1:n
        subplot(n, 1, i);
        hold on;
        if with_sim
            plot(T(1:plot_len), true_states(1:plot_len, i), 'r');
        end
        if with_ce
            plot(T((cd+1):plot_len), means(:, i), 'b');
        end
        if with_kf
            plot(T(1:plot_len), kf_cond_means(1:plot_len, i), 'g--');
        end
        hold off;
    end
    
    % More blocks similar to the first one would follow, dealing with plotting errors, covariances, measurements, etc.
    
    end
    