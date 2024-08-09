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
        T = 0:(size(simulation_history{2}, 1)-1);
    elseif with_kf
        n = size(kf_history{1}, 2);
        T = 0:(size(kf_history{1}, 1)-1);
    elseif with_ce
        n = numel(cauchy_moment_info.x(1));
        T = 0:(length(cauchy_moment_info.x) - 1 + with_cauchy_delay);
    else
        fprintf('Must provide simulation data, kalman filter data or cauchy estimator data!\nExiting function with no plotting (Nothing Given!)\n');
        return;
    end

    if with_sim
        true_states = simulation_history{1};
        msmts = simulation_history{2};
        proc_noises = simulation_history{3};
        msmt_noises = simulation_history{4};
    end

    if with_ce
        means = cat(1, cauchy_moment_info.x);
        covars = cat(1, cauchy_moment_info.P);
        cerr_norm_factors = cat(1, cauchy_moment_info.cerr_fz);
        cerr_means = cat(1, cauchy_moment_info.cerr_x);
        cerr_covars = cat(1, cauchy_moment_info.cerr_P);
        n = size(means, 2);
    end

    if with_kf
        kf_cond_means = kf_history{1};
        kf_cond_covars = kf_history{2};
    end

    if with_ce
        cd = with_cauchy_delay;
        if ~with_partial_plot && with_cauchy_delay
            plot_len = size(covars, 1) + cd;
            assert(plot_len == numel(T), 'Mismatch in array lengths');
        elseif with_partial_plot || with_cauchy_delay
            plot_len = size(covars, 1) + cd;
            assert(plot_len <= numel(T), 'Mismatch in array lengths');
        else
            assert(size(covars, 1) + cd == numel(T), 'Mismatch in array lengths');
            plot_len = numel(T);
        end
    else
        plot_len = numel(T);
    end



    % Plot the true state history vs the conditional mean estimate
    figure;
    if with_kf
        sgtitle('True States (red) vs Cauchy (blue) vs Kalman (black--)');
    else
        sgtitle('True States (red) vs Cauchy Estimates (blue)');
    end
    for i = 1:n
        subplot(n, 1, i);
        hold on;

        if with_sim
            plot(T(1:plot_len), true_states(1:plot_len,i), 'r');
        end
        if with_ce
            plot(T(cd+1:plot_len), means(:,i), 'b');
        end
        if with_kf
            plot(T(1:plot_len), kf_cond_means(1:plot_len,i), 'k--');
        end
        hold off;
    end

    % Plot the state error and one-sigma bound of the covariance
    figure;
    if with_kf
        sgtitle('Cauchy 1-Sig (blue/red) vs Kalman 1-Sig (black-/purple-)');
    else
        sgtitle('State Error (b) vs One Sigma Bound (r)');
    end
    for i = 1:n
        subplot(n, 1, i);
        hold on;
        if with_ce
            plot(T(cd+1:plot_len), true_states(cd+1:plot_len,i) - means(:,i), 'b');
            plot(T(cd+1:plot_len), scale*sqrt(covars(:,i,i)), 'r');
            plot(T(cd+1:plot_len), -scale*sqrt(covars(:,i,i)), 'r');
        end
        if with_kf
            plot(T(1:plot_len), true_states(1:plot_len,i) - kf_cond_means(1:plot_len,i), 'k--');
            plot(T(1:plot_len), scale*sqrt(kf_cond_covars(1:plot_len,i,i)), 'm--');
            plot(T(1:plot_len), -scale*sqrt(kf_cond_covars(1:plot_len,i,i)), 'm--');
        end
        hold off;
    end

    % Plot the measurements, and the measurement and process noise
    if with_sim
        line_types = {'-', '--', '-.', ':', '-', '--', '-.', ':'};
        figure;
        sgtitle('Msmts (purple), Msmt Noise (black), Proc Noise (blue)');
        m = 3;
        count = 1;

        % Plot measurements (msmts)
        subplot(m, 1, count);
        hold on;
        for i = 1:size(msmts, 2)
            plot(T(1:plot_len), msmts(1:plot_len, i), strcat('m', line_types{i}));
        end
        hold off;
        count = count + 1;
        
        % Plot measurement noise (msmt_noises)
        subplot(m, 1, count);
        hold on;
        for i = 1:size(msmt_noises, 2)
            plot(T(1:plot_len), msmt_noises(1:plot_len, i), strcat('k', line_types{i}));
        end
        hold off;
        count = count + 1;
        
        % Plot process noise (proc_noises)
        subplot(m, 1, count);
        hold on;
        for i = 1:size(proc_noises, 2)
            plot(T(2:plot_len), proc_noises(1:plot_len-1, i), strcat('b', line_types{i}));
        end
        hold off;
    end
    
    % Plot the max complex error in the mean/covariance and norm factor
    if with_ce
        figure;
        sgtitle('Complex Errors (mean,covar,norm factor) in Semi-Log');
        subplot(3, 1, 1);
        semilogy(T(cd+1:plot_len), cerr_means, 'k');
        
        subplot(3, 1, 2);
        semilogy(T(cd+1:plot_len), cerr_covars, 'k');
        
        abs_cerr_norm_factors = abs(cerr_norm_factors);
        subplot(3, 1, 3);
        semilogy(T(cd+1:plot_len), abs_cerr_norm_factors, 'k');
    end
    
end
