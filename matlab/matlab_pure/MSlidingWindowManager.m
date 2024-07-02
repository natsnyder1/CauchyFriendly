% file: MSlidingWindowManager

classdef MSlidingWindowManager < handle
    properties
        num_windows;
        mode;
        is_initialized;
        moment_info;
        avg_moment_info;
        debug_print;
        step_idx;
        win_idxs;
        win_counts;
        cauchyEsts;
        n;
        cmcc;
        pncc;
        p;
        reinit_func;
    end
    
    methods
        function obj = MSlidingWindowManager(mode, num_windows, swm_debug_print, win_debug_print)
            if nargin < 3
                swm_debug_print = true;
            end
            if nargin < 4
                win_debug_print = false;
            end
            obj.num_windows = num_windows;
            assert(num_windows < 20);
            modes = ["lti", "ltv", "nonlin"];
            if ~any(strcmpi(mode, modes))
                fprintf("[Error MSlidingWindowManager:] chosen mode %s invalid. Please choose one of the following: ", mode);
                disp(modes)
                return;
            else
                obj.mode = lower(mode);
                fprintf("Set Sliding Window Manager Mode to: %s\n", obj.mode);
            end
            obj.is_initialized = false;
            fields = ["x", "P", "cerr_x", "cerr_P", "fz", "cerr_fz", "win_idx", "err_code"];
            for i = 1:length(fields)
                obj.moment_info.(fields(i)) = [];
                obj.avg_moment_info.(fields(i)) = [];
            end
            obj.debug_print = swm_debug_print;
            obj.step_idx = 0;
            obj.win_idxs = 0:(num_windows-1);
            obj.win_counts = zeros(num_windows, 1, 'int64');
            for i = 1:num_windows
                obj.cauchyEsts{i} = MCauchyEstimator(obj.mode, num_windows, win_debug_print);
            end
        end
        
        function set_dimensions(obj)
            obj.n = obj.cauchyEsts{1}.n;
            obj.cmcc = obj.cauchyEsts{1}.cmcc;
            obj.pncc = obj.cauchyEsts{1}.pncc;
            obj.p = obj.cauchyEsts{1}.p;
        end
        
        function initialize_lti(obj, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dt, step, reinit_func)
            if nargin < 11
                dt = 0;
            end
            if nargin < 12
                step = 0;
            end
            if nargin < 13
                reinit_func = [];
            end
            if numel(p0) == 1
                fprintf("[Error MSlidingWindowManager:] Do not use this class for scalar systems! This is only for systems of dimension >1 Use the MCauchyEstimator class instead!\n");
                return;
            end
            if ~strcmp(obj.mode, "lti")
                fprintf("Attempting to call initialize_lti method when mode was set to %s is not allowed! You must call initialize_%s ... or reset the mode altogether!\n", obj.mode, obj.mode);
                fprintf("NonLin initialization not successful!\n");
                return;
            end
            obj.reinit_func = reinit_func;
            i_step = int32(step);
            d_dt = double(dt);
            for i = 1:obj.num_windows
                obj.cauchyEsts{i}.initialize_lti(A0, p0, b0, Phi, B, Gamma, beta, H, gamma, i_step + i, d_dt);
                obj.cauchyEsts{i}.set_window_number(i);
            end
            obj.is_initialized = true;
            set_dimensions(obj);
            fprintf("LTI initialization successful! You can use the step(msmts, controls) method to run the estimator now!\n");
            fprintf("Note: Conditional Mean/Variance will be a function of the last %d time-steps, %d measurements per step == %d total!\n", obj.num_windows, obj.p, obj.p * obj.num_windows);
        end
        
        function initialize_ltv(obj, varargin)
            % Placeholder for initialize_ltv method
        end
        
        function initialize_nonlin(obj, varargin)
            % Placeholder for initialize_nonlin method
        end
        
        function [best_idx, okays] = best_window_est(obj)
            W = obj.num_windows;
            okays = false(W, 1);
            idxs = [];
            check_idx = obj.p;
            COV_UNSTABLE = 2;
            COV_DNE = 8;
            for i = 1:W
                if obj.win_counts(i) > 0
                    err = obj.cauchyEsts{i}.err_code(check_idx);
                    if bitand(err, COV_UNSTABLE) || bitand(err, COV_DNE)
                        % pass (do nothing)
                    else
                        idxs = [idxs; i, obj.win_counts(i)]; % Append to idxs
                        okays(i) = true;
                    end
                end
            end
            
            if obj.step_idx == 0
                best_idx = 1;
                okays(1) = true;
            else
                if isempty(idxs)
                    fprintf('No window is available without an error code!\n');
                    errorId = 'MSlidingWindowManager:NoValidWindow';
                    errorMsg = 'No window is available without an error code!';
                    error(errorId, errorMsg);
                end
                [~, sorting_indices] = sort(idxs(:, 2), 'descend');
                sorted_idxs = idxs(sorting_indices, :);
                best_idx = sorted_idxs(1, 1);
            end
            
            n = obj.n;
            best_estm = obj.cauchyEsts{best_idx};
            
            obj.moment_info.fz = [obj.moment_info.fz; best_estm.fz(check_idx)];
            temp = best_estm.x((check_idx - 1) * n + (1:n));
            obj.moment_info.x = cat(1, obj.moment_info.x, reshape(temp, [1 size(temp)]));

            P_flat = best_estm.P((check_idx - 1) * n * n + (1:n * n));
            temp = reshape(P_flat, n, n);
            obj.moment_info.P = cat(1, obj.moment_info.P, reshape(temp, [1 size(temp)]));
            
            obj.moment_info.cerr_x = [obj.moment_info.cerr_x; best_estm.cerr_x(check_idx)];
            obj.moment_info.cerr_P = [obj.moment_info.cerr_P; best_estm.cerr_P(check_idx)];
            obj.moment_info.cerr_fz = [obj.moment_info.cerr_fz; best_estm.cerr_fz(check_idx)];
            obj.moment_info.win_idx = [obj.moment_info.win_idx; best_idx];
            obj.moment_info.err_code = [obj.moment_info.err_code; best_estm.err_code(check_idx)];
        end
        
        
        function [win_avg_mean, win_avg_cov] = weighted_average_win_est(obj, usable_wins)
            n = obj.n;
            last_idx = obj.p;
            win_avg_mean = zeros(n, 1);
            win_avg_cov = zeros(n, n);
            win_avg_fz = 0;
            win_avg_cerr_fz = 0;
            win_avg_cerr_x = 0;
            win_avg_cerr_P = 0;
            win_avg_err_code = 0;
            win_norm_fac = 0.0;
            
            for i = 1:obj.num_windows
                win_count = double(obj.win_counts(i));
                if win_count > 0 && usable_wins(i)
                    est = obj.cauchyEsts{i};
                    norm_fac = win_count / obj.num_windows;
                    win_norm_fac = win_norm_fac + norm_fac;
                    [x, P] = est.get_last_mean_cov(); 
                    win_avg_mean = win_avg_mean + x * norm_fac;
                    win_avg_cov = win_avg_cov + P * norm_fac;         
                    win_avg_fz = win_avg_fz + est.fz(last_idx) * norm_fac;
                    win_avg_cerr_fz = win_avg_cerr_fz + est.cerr_fz(last_idx) * norm_fac;
                    win_avg_cerr_x = win_avg_cerr_x + est.cerr_x(last_idx) * norm_fac;
                    win_avg_cerr_P = win_avg_cerr_P + est.cerr_P(last_idx) * norm_fac;
                    win_avg_err_code = bitor(win_avg_err_code, est.err_code(last_idx));
                end
            end
            
            if win_norm_fac ~= 0
                win_avg_mean = win_avg_mean / win_norm_fac;
                win_avg_cov = win_avg_cov / win_norm_fac;
                win_avg_fz = win_avg_fz / win_norm_fac;
                win_avg_cerr_fz = win_avg_cerr_fz / win_norm_fac;
                win_avg_cerr_x = win_avg_cerr_x / win_norm_fac;
                win_avg_cerr_P = win_avg_cerr_P / win_norm_fac;
            
                obj.avg_moment_info.fz = [obj.avg_moment_info.fz, win_avg_fz];
                obj.avg_moment_info.x = cat(1, obj.avg_moment_info.x, reshape(win_avg_mean, [1 size(win_avg_mean)]));
                obj.avg_moment_info.P = cat(1, obj.avg_moment_info.P, reshape(win_avg_cov, [1 size(win_avg_cov)]));
                obj.avg_moment_info.cerr_x = [obj.avg_moment_info.cerr_x, win_avg_cerr_x];
                obj.avg_moment_info.cerr_P = [obj.avg_moment_info.cerr_P, win_avg_cerr_P];
                obj.avg_moment_info.cerr_fz = [obj.avg_moment_info.cerr_fz, win_avg_cerr_fz];
                obj.avg_moment_info.win_idx = [obj.avg_moment_info.win_idx, -1];
                obj.avg_moment_info.err_code = [obj.avg_moment_info.err_code, win_avg_err_code];
            else
                error('No valid windows to compute the weighted average estimate.');
            end
        end
        
        
        function [xhat, Phat, wavg_xhat, wavg_Phat] = step(obj, msmts, controls, reinit_args)
            if nargin < 4
                reinit_args = [];
            end
            if nargin < 3
                controls = [];
            end
            
            if ~obj.is_initialized
                fprintf('Estimator is not initialized yet. Mode set to %s. Please call method initialize_%s before running step()!\n', obj.mode, obj.mode);
                fprintf('Not stepping! Please call correct method / fix mode!\n');
                return;
            end
            if obj.debug_print
                fprintf('SWM: Step %d\n', obj.step_idx);
            end
            
            if obj.step_idx == 0
                if obj.debug_print
                    fprintf('  Window %d is on step %d/%d\n', 1, 1, obj.num_windows);
                end
                obj.cauchyEsts{1}.step(msmts, controls);
                obj.win_counts(1) = obj.win_counts(1) + 1;
            else
                idx_max = find(obj.win_counts == max(obj.win_counts), 1, 'first');
                idx_min = find(obj.win_counts == min(obj.win_counts), 1, 'first');
                
                for win_idx = 1:length(obj.win_idxs)
                    win_count = obj.win_counts(win_idx);
                    if win_count > 0
                        if obj.debug_print
                            fprintf('  Window %d is on step %d/%d\n', win_idx, win_count + 1, obj.num_windows);
                        end
                        obj.cauchyEsts{win_idx}.step(msmts, controls);
                        obj.win_counts(win_idx) = obj.win_counts(win_idx) + 1;
                    end
                end
            end
            
            [best_idx, usable_wins] = obj.best_window_est();
            [wavg_xhat, wavg_Phat] = obj.weighted_average_win_est(usable_wins);
            [xhat, Phat] = obj.cauchyEsts{best_idx}.get_last_mean_cov();
            
            if obj.step_idx > 0
                if ~isempty(obj.reinit_func)
                    reinit_args.copied_win_counts = obj.win_counts;
                    obj.reinit_func(obj.cauchyEsts, best_idx, usable_wins, reinit_args);
                else
                    if ~isempty(reinit_args)
                        fprintf(['  [Warn MSlidingWindowManager:] Providing reinit_args ' ...
                                 'with no reinit_func given will do nothing!\n']);
                    end
                    speyer_restart_idx = obj.p;
                    obj.cauchyEsts{idx_min}.reset_about_estimator(obj.cauchyEsts{best_idx}, speyer_restart_idx);
                    if obj.debug_print
                        fprintf('  Window %d reinitializes Window %d\n', best_idx, idx_min);
                    end
                    obj.win_counts(idx_min) = obj.win_counts(idx_min) + 1;
                end
                
                if obj.win_counts(idx_max) == obj.num_windows
                    obj.cauchyEsts{idx_max}.reset();
                    obj.win_counts(idx_max) = 0;
                end
            end
            
            obj.step_idx = obj.step_idx + 1;
        end
        
        
        function shutdown(obj)
            if ~obj.is_initialized
                fprintf('Cannot shutdown Sliding Window Manager before it has been initialized!\n');
                return;
            end
            for i = 1:obj.num_windows
                obj.cauchyEsts{i}.shutdown();
            end
            obj.win_counts = zeros(obj.num_windows, 1, 'int64');
            fprintf('Sliding Window Manager has been shutdown!\n');
            obj.is_initialized = false;
            obj.step_idx = 0;
        end
        
        function delete(obj)
            if obj.is_initialized
                obj.shutdown();
                obj.is_initialized = false;
            end
        end
    end
end
