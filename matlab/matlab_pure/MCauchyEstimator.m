% file: MCauchyEstimator

classdef MCauchyEstimator < handle
    properties
        % MCauchyEstimator
        modes = {'lti', 'ltv', 'nonlin'}
        mode
        num_steps
        debug_print = true
        % ndim_input_checker
        n
        pncc
        cmcc
        p
        % call_step
        moment_info = struct('x', zeros(0, 0), 'P', zeros(0, 0, 0), 'cerr_x', [], 'cerr_P', [], 'fz', [], 'cerr_fz', [], 'err_code', [])
        fz
        step_count = 0
        x
        P
        xbar
        zbar
        cerr_fz
        cerr_x
        cerr_P
        err_code
        % initialize_lti
        A0
        p0
        b0
        Phi
        Gamma
        beta
        B
        H
        gamma
        mcauchy_handle
        is_initialized = false
        % initialize_nonlin
        x0
        f_duc_ptr1
        f_duc_ptr2
        f_duc_ptr3
        % step
        msmts
        controls
        free_names_ptr
    end
    
    methods (Access = public)
        function obj = MCauchyEstimator(mode, num_steps, debug_print)
            if nargin < 3
                debug_print = true;
            end

            mode = lower(mode);
            if ~ismember(mode, obj.modes)
                fprintf('[Error MCauchyEstimator:] chosen mode %s invalid. Please choose one of the following: {%s}\n', mode, strjoin(obj.modes, ', '));
            else
                obj.mode = mode;
                fprintf('Set Cauchy Estimator Mode to: %s\n', obj.mode);
            end
            
            obj.num_steps = int32(num_steps);
            assert(obj.num_steps > 0, 'Number of steps must be positive.');

            obj.debug_print = logical(debug_print);

        end

        
        function n = init_params_checker(obj, A0, p0, b0)
            assert(size(A0, 1) == size(A0, 2), 'A0 must be a square matrix.');
            assert(size(A0, 1) == length(p0), 'Dimension mismatch between A0 and p0.');
            assert(size(A0, 1) == length(b0), 'Dimension mismatch between A0 and b0.');
            
            n = size(A0, 1);
            assert(rank(A0) == n, 'A0 must have full rank.');
            assert(all(p0 >= 0), 'All elements of p0 must be non-negative.');
        end


        function ndim_input_checker(obj, A0, p0, b0, Phi, B, Gamma, beta, H, gamma)
            n = obj.init_params_checker(A0, p0, b0);
            
            assert(size(Phi, 1) == size(Phi, 2) && size(Phi, 1) == n, 'Phi must be a square matrix and match dimensions with A0.');
            assert(rank(Phi) == n, 'Phi must have full rank.');
            assert(size(Gamma, 1) == n, 'Gamma must have the same number of rows as dimension of A0.');
        
            pncc = 0;
            if ~isempty(Gamma)
                if ismatrix(Gamma)
                    assert(numel(beta) == size(Gamma, 2), 'Dimension mismatch between Gamma and beta.');
                else
                    assert(numel(beta) == 1, 'If Gamma is a vector, beta must be scalar.');
                end
                pncc = numel(beta);
            else
                assert(isempty(beta), 'If Gamma is empty, beta must also be empty.');
            end
            
            cmcc = 0;
            if ~isempty(B)
                if ismatrix(B)
                    cmcc = size(B, 2);
                else
                    cmcc = 1;
                end
                assert(size(B, 1) == n, 'B must have the same number of rows as dimension of A0.');
            end
            
            if iscolumn(H) || isrow(H)
                assert(numel(gamma) == 1, 'If H is a vector, gamma must be scalar.');
                p = 1;
            else
                assert(size(H, 2) == n, 'Second dimension of H must match the dimension of A0.');
                assert(numel(gamma) == size(H, 1), 'Dimension mismatch between H and gamma.');
                p = size(H, 1);
            end
            
            if any(abs(H * Gamma) < 1e-12, 'all')
                warning('Warning MCauchyEstimator: | H * Gamma | < eps for some input / output channels. This may result in undefined moments!');
            end
            
            obj.n = n;
            obj.pncc = pncc;
            obj.cmcc = cmcc;
            obj.p = p;
        end


        function [msmts, controls] = msmts_controls_checker(obj, msmts, controls)
            if obj.p == 1
                if ~isa(msmts, 'double')
                    msmts = double(msmts(:));
                else
                    msmts = msmts(:);
                end
                assert(numel(msmts) == obj.p, 'Mismatched measurements size.');
            else
                msmts = double(msmts(:));
                assert(numel(msmts) == obj.p, 'Mismatched measurements size.');
            end
            if obj.cmcc == 0
                assert(isempty(controls), 'Controls should be none if cmcc is zero.');
                controls = double([]);
            else
                assert(~isempty(controls), 'Controls should not be none if cmcc is non-zero.');
                if ~isa(controls, 'double')
                    controls = double(controls(:));
                else
                    controls = controls(:);
                end
                assert(numel(controls) == obj.cmcc, 'Mismatched controls size.');
            end
        end


        function [xs, Ps] = call_step(obj, msmts, controls, full_info)

            if strcmp(obj.mode, 'lti')
                [~, ~, ~, ~, ~, ~, ...
                obj.fz, obj.x, obj.P, ...
                obj.cerr_fz, obj.cerr_x, obj.cerr_P, obj.err_code] = ...
                    mcauchy_step(obj.mcauchy_handle, msmts, controls);
            elseif strcmp(obj.mode, 'ltv')
                [obj.Phi, obj.Gamma, obj.B, obj.H, obj.beta, obj.gamma, ...
                obj.fz, obj.x, obj.P, ...
                obj.cerr_fz, obj.cerr_x, obj.cerr_P, obj.err_code] = ...
                    mcauchy_single_step_ltiv(obj.mcauchy_handle, msmts, controls);
            else
                [obj.Phi, obj.Gamma, obj.B, obj.H, obj.beta, obj.gamma, ...
                obj.fz, obj.x, obj.P, obj.xbar, obj.zbar, ...
                obj.cerr_fz, obj.cerr_x, obj.cerr_P, obj.err_code] = ...
                    mcauchy_single_step_nonlin(obj.mcauchy_handle, msmts, controls, obj.step_count ~= 0);
            end
        
            if full_info
                xs = cell(obj.p, 1);
                Ps = cell(obj.p, 1);
                for i = 1:obj.p
                    xs{i} = obj.x((i-1)*obj.n+1 : i*obj.n);
                    Ps{i} = reshape(obj.P((i-1)*obj.n^2+1 : i*obj.n^2), obj.n, obj.n);
                end
            else
                xs = obj.x(end-obj.n+1:end);
                Ps = reshape(obj.P(end-obj.n^2+1:end), obj.n, obj.n);
            end
        
            fz = obj.fz(end);
            x = obj.x(end-obj.n+1:end);
            P = reshape(obj.P((end-obj.n^2+1):end), obj.n, obj.n);
            cerr_fz = obj.cerr_fz(end);
            cerr_x = obj.cerr_x(end);
            cerr_P = obj.cerr_P(end);
            err_code = obj.err_code(end);
            
            if size(obj.moment_info.fz, 1) == 0
                obj.moment_info.fz = fz;
                obj.moment_info.x = reshape(x, [1 size(x)]);
                obj.moment_info.P = reshape(P, [1 size(P)]);
                obj.moment_info.cerr_x = cerr_x;
                obj.moment_info.cerr_P = cerr_P;
                obj.moment_info.cerr_fz = cerr_fz;
                obj.moment_info.err_code = err_code;
            else 
                obj.moment_info.fz = cat(1,obj.moment_info.fz, fz);
                obj.moment_info.x = cat(1, obj.moment_info.x, reshape(x, [1 size(x)]));
                obj.moment_info.P = cat(1, obj.moment_info.P, reshape(P, [1 size(P)]));
                obj.moment_info.cerr_x = cat(1,obj.moment_info.cerr_x, cerr_x);
                obj.moment_info.cerr_P = cat(1,obj.moment_info.cerr_P, cerr_P);
                obj.moment_info.cerr_fz = cat(1,obj.moment_info.cerr_fz, cerr_fz);
                obj.moment_info.err_code = cat(1,obj.moment_info.err_code, err_code);
            end
            
            obj.step_count = obj.step_count + 1;
        end


        function initialize_lti(obj, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, init_step, dt)
            if ~strcmp(obj.mode, "lti")
                fprintf(['Attempting to call initialize_lti method when mode was set to %s is not allowed! ' ...
                         'You must call initialize_%s ... or reset the mode altogether!\n'], obj.mode, obj.mode);
                disp('LTI initialization not successful!');
                return;
            end

            if (nargin < 12)
                dt = 0;
            end
            if (nargin < 11)
                init_step = 0;
            end
            
            obj.ndim_input_checker(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
            
            % Reshape and ensure all inputs are column vectors

            obj.A0 = reshape(A0.', [], 1);
            obj.p0 = reshape(p0.', [], 1);
            obj.b0 = reshape(b0.', [], 1);
            obj.Phi = reshape(Phi.', [], 1);
            obj.Gamma = [];
            if ~isempty(Gamma)
                obj.Gamma = reshape(Gamma.', [], 1);
            end
            obj.beta = [];
            if ~isempty(beta)
                obj.beta = reshape(beta.', [], 1);
            end
            obj.B = [];
            if ~isempty(B)
                obj.B = reshape(B.', [], 1);
            end
            obj.H = reshape(H.', [], 1);
            obj.gamma = reshape(gamma.', [], 1);
            
            % Typecast as needed
            init_step = int32(init_step);
            dt = double(dt);
            
            % Instantiate and initialize the mcauchy object
            % FIX THIS LINE
            obj.mcauchy_handle = mcauchy_initialize_lti(obj.num_steps, obj.A0, obj.p0, obj.b0, obj.Phi, obj.Gamma, obj.B, ...
                                                obj.beta, obj.H, obj.gamma, dt, init_step, obj.debug_print);
            obj.is_initialized = true;
    
            fprintf('LTI initialization successful! You can use the step(msmts, controls) method to run the estimator now!\n');
            fprintf('Note: You can call the step function %d time-steps, %d measurements per step == %d total times!\n', ...
                    obj.num_steps, obj.p, obj.num_steps * obj.p);
        end


        function [xk, Pk] = step(obj, msmts, controls, full_info)
            if nargin < 3
                controls = []; % Default value if controls are not provided
            end
            if nargin < 4
                full_info = false; % Default value if full_info is not provided
            end

            full_info = logical(full_info);

            if ~obj.is_initialized
                fprintf("Estimator is not initialized yet. Mode set to %s. Please call method initialize_%s before running step()!\n", obj.mode, obj.mode);
                fprintf("Not stepping! Please call correct method / fix mode!\n");
                xk = [];
                Pk = [];
                return;
            end
            
            if obj.step_count == obj.num_steps
                fprintf("[Error:] Cannot step estimator again, you have already stepped the estimator the initialized number of steps\n");
                fprintf("Not stepping! Please shut estimator down or reset it!\n");
                xk = [];
                Pk = [];
                return;
            end
            
            [msmts, controls] = obj.msmts_controls_checker(msmts, controls);
            obj.msmts = msmts;
            obj.controls = controls;
            
            [xk, Pk] = obj.call_step(msmts, controls, full_info);
        end


        function [last_mean, last_cov] = get_last_mean_cov(obj)
            last_mean = squeeze(obj.moment_info.x(end,:))';
            last_cov = squeeze(obj.moment_info.P(end,:,:));
        end


        function [A0, p0, b0, reinit_xbar] = get_reinitialization_statistics(obj, msmt_idx)
            if nargin < 2
                msmt_idx = -1;
            end
            A0 = []; p0 = []; b0 = []; reinit_xbar = []; % Default empty outputs
            
            if obj.step_count == 0 || ~obj.is_initialized
                fprintf('[Error get_reinitialization_statistics]: Cannot find reinitialization stats of an estimator not initialized, or that has not processed at least one measurement! Please correct!\n');
                return;
            end
            
            if msmt_idx-1 >= obj.p || msmt_idx-1 < -obj.p
                fprintf('[Error get_reinitialization_statistics]: Cannot find reinitialization stats for msmt_idx=%d. The index is out of range -%d <= msmt_idx < %d...(max index is p-1=%d)! Please correct!\n', msmt_idx, -obj.p, obj.p, obj.p-1);
                return;
            end
            
            msmt_idx = int32(msmt_idx);
            if msmt_idx-1 < 0
                msmt_idx = msmt_idx + obj.p;
            end
            
            reinit_msmt = obj.msmts(msmt_idx);
            reinit_xhat = obj.x(((msmt_idx-1)*obj.n+1) : (msmt_idx)*obj.n); 
            reinit_Phat = obj.P(((msmt_idx-1)*obj.n*obj.n+1) : (msmt_idx)*obj.n*obj.n);
            reinit_H = obj.H(((msmt_idx-1)*obj.n+1) : (msmt_idx)*obj.n);
            reinit_gamma = obj.gamma(msmt_idx);

            if ~strcmp(obj.mode, "nonlin")
                [A0,p0,b0] = mcauchy_get_reinitialization_statistics(obj.mcauchy_handle, reinit_msmt, reinit_xhat, reinit_Phat, reinit_H, reinit_gamma);
                A0 = reshape(A0, [obj.n, obj.n]);
            else
                reinit_xbar = obj.xbar(((msmt_idx-1)*obj.n)+1:(msmt_idx)*obj.n);
                reinit_zbar = obj.zbar(msmt_idx);
                dx = reinit_xhat - reinit_xbar;
                dz = reinit_msmt - reinit_zbar;
                [A0, p0, b0] = mcauchy_get_reinitialization_statistics(obj.mcauchy_handle, dz, dx, reinit_Phat, reinit_H, reinit_gamma);
                A0 = reshape(A0, [obj.n, obj.n]);
            end
        end
        
        function reset(obj, A0, p0, b0, xbar)
            if nargin < 5
                xbar = [];
            end
            if nargin < 4
                b0 = [];
            end
            if nargin < 3
                p0 = [];
            end
            if nargin < 2
                A0 = [];
            end

            if ~obj.is_initialized
                fprintf('Cannot reset estimator before it has been initialized (or after shutdown has been called)!\n');
                return;
            end
            
            obj.step_count = 0; % Reset step count
            
            % Check the size and assign default values if arguments not supplied
            A0 = obj.size_checker(A0, [obj.n, obj.n], 'A0');
            p0 = obj.size_checker(p0, [obj.n, 1], 'p0');
            b0 = obj.size_checker(b0, [obj.n, 1], 'b0');
            xbar = obj.size_checker(xbar, [obj.n, 1], 'xbar');

            if ~isempty(A0), obj.A0 = reshape(A0, [], 1); end
            if ~isempty(p0), obj.p0 = reshape(p0, [], 1); end
            if ~isempty(b0), obj.b0 = reshape(b0, [], 1); end
            if ~isempty(xbar), obj.xbar = reshape(xbar, [], 1); end

            if ~strcmp(obj.mode, 'nonlin') && ~isempty(xbar)
                fprintf("Note to user: Setting xbar for any mode besides 'nonlinear' will have no effect!\n");
            end

            mcauchy_reset(obj.mcauchy_handle, obj.A0, obj.p0, obj.b0, obj.xbar);
        end


        function reset_about_estimator(obj, other_estimator, msmt_idx)
            if nargin < 3
                msmt_idx = -1;
            end
            
            if ~obj.is_initialized
                error('[Error reset_about_estimator:] This estimator is not initialized! Must initialize the estimator before using this function!');
            end
            if ~other_estimator.is_initialized
                error('[Error reset_about_estimator:] Other estimator is not initialized! Must initialize the other estimator (and step it) before using this function!');
            end
            if other_estimator.step_count == 0
                error('[Error reset_about_estimator:] Other estimator has step_count == 0 (step_count must be > 0). The inputted estimator must be stepped before using this function!');
            end
            if eq(obj, other_estimator)
                error('[Error reset_about_estimator:] Other estimator cannot be this estimator itself!');
            end
            if obj.p ~= other_estimator.p
                error('[Error reset_about_estimator:] Both estimators must process the same number of measurements! this=%d, other=%d', obj.p, other_estimator.p);
            end
            if ~strcmp(obj.mode, other_estimator.mode)
                error('[Error reset_about_estimator:] Both estimators must have same mode! this=%s, other=%s', obj.mode, other_estimator.mode);
            end
            % indeces start at 1 in matlab instead of 0 in python, so must
            % check msmt_idx-1 inst
            if msmt_idx-1 >= obj.p || msmt_idx-1 < -obj.p
                error('[Error reset_about_estimator:] Specified msmt_idx=%d. The index is out of range -%d <= msmt_idx < %d...(max index is p-1=%d)! Please correct!', msmt_idx, -obj.p, obj.p-1, obj.p-1);
            end
            if msmt_idx-1 < 0
                msmt_idx = msmt_idx + obj.p;
            end
            msmts = other_estimator.msmts(msmt_idx:end);
            obj.msmts = msmts;
            if ~strcmp(obj.mode, 'nonlin')
                [A0, p0, b0] = other_estimator.get_reinitialization_statistics(msmt_idx);
                obj.reset(A0, p0, b0);
            else
                [A0, p0, b0, xbar] = other_estimator.get_reinitialization_statistics(msmt_idx);
                obj.reset(A0, p0, b0, xbar);
            end
            
            [xs, Ps] = obj.call_step(msmts, [], false);
            mcauchy_set_master_step(obj.mcauchy_handle, obj.p);
        end
        

        function [xs, Ps] = reset_with_last_measurement(obj, z_scalar, A0, p0, b0, xbar)
            z_scalar = reshape(z_scalar, [], 1);
            if ~strcmp(obj.mode, 'nonlin')
                obj.reset(A0, p0, b0);
            else
                obj.reset(A0, p0, b0, xbar);
            end
            [xs, Ps] = obj.call_step(z_scalar, []);
            mcauchy_set_master_step(obj.mcauchy_handle, obj.p);
        end
    

        function set_window_number(obj, win_num)
            win_num = int32(win_num);
            mcauchy_set_window_number(obj.mcauchy_handle, win_num);
        end
    
        function shutdown(obj)
            if obj.free_names_ptr ~= 0
                mcauchy_free_names(obj.free_names_ptr);
            end
            obj.free_names_ptr = 0;
            if ~obj.is_initialized
                error('Cannot shutdown estimator before it has been initialized!');
            end
            mcauchy_shutdown(obj.mcauchy_handle);
            obj.mcauchy_handle = [];
            fprintf('Estimator backend has been shutdown!\n');
            obj.is_initialized = false;
        end

        function delete(obj)
            if obj.is_initialized
                obj.shutdown();
                obj.is_initialized = false;
            end
            if obj.free_names_ptr ~= 0
                mcauchy_free_names(obj.free_names_ptr);
            end
            obj.free_names_ptr = 0;
        end


        function [X, Y, Z] = get_marginal_2D_pointwise_cpdf(obj, marg_idx1, marg_idx2, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution, log_dir, reset_cache)
    
            if ~obj.is_initialized
                error('Estimator must be initialized before computing CPDFs.');
            end
        
            if obj.step_count < 1
                error('Estimator must have performed at least one step before computing CPDFs.');
            end
            
            if ~( (marg_idx1>0) && (marg_idx2 > marg_idx1) && (marg_idx2 <= obj.n) )
                error('Calling marginal 2d cpdf required marg_idx1>= 1 and marg_idx2 <=n with marg_idx1 < marg_idx2');
            end
        
            marg_idx1 = int32(marg_idx1);
            marg_idx2 = int32(marg_idx2);
            gridx_low = double(gridx_low);
            gridx_high = double(gridx_high);
            gridx_resolution = double(gridx_resolution);
            gridy_low = double(gridy_low);
            gridy_high = double(gridy_high);
            gridy_resolution = double(gridy_resolution);
        
            assert(gridx_high > gridx_low, 'Grid x-axis high limit must be greater than low limit.');
            assert(gridy_high > gridy_low, 'Grid y-axis high limit must be greater than low limit.');
            assert(gridx_resolution > 0, 'Grid x-axis resolution must be positive.');
            assert(gridy_resolution > 0, 'Grid y-axis resolution must be positive.');
            
            if nargin < 11
                reset_cache = true;
            end
            
            if(nargin < 10) || isempty(log_dir)
                log_dir = 'n'; % Default log directory
            else
                log_dir = char(log_dir); % Convert to char array if not empty
            end
        
            [cpdf_points, num_gridx, num_gridy] = mcauchy_get_marginal_2D_pointwise_cpdf(obj.mcauchy_handle, marg_idx1-1, marg_idx2-1, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution, log_dir, reset_cache);
            X = squeeze(cpdf_points(1,:,:));
            Y = squeeze(cpdf_points(2,:,:));
            Z = squeeze(cpdf_points(3,:,:));
        end

        function [X, Y, Z] = get_2D_pointwise_cpdf(obj, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution, log_dir, reset_cache)
            
            if nargin < 9
                reset_cache = true;
            end
            
            if (nargin < 8) || isempty(log_dir)
                log_dir = [];
            end
            
            X = []; Y = []; Z = []; % Initialize empty outputs in case of early return
            
            if ~obj.is_initialized
                error('Cannot evaluate Cauchy Estimator CPDF before it has been initialized (or after shutdown has been called)!');
            end
            
            if obj.n ~= 2
                error('Cannot evaluate Cauchy Estimator 2D CPDF for a %d-state system!', obj.n);
            end
        
            if obj.step_count < 1
                error('Cannot evaluate Cauchy Estimator 2D CPDF before it has been stepped!');
            end
            
            % marginal indices fixed to 0 and 1 since the function is only for 2-state systems
            [X, Y, Z] = obj.get_marginal_2D_pointwise_cpdf(1, 2, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution, log_dir, reset_cache);
        end


        function [X, Y] = get_marginal_1D_pointwise_cpdf(obj, marg_idx, gridx_low, gridx_high, gridx_resolution, log_dir)
            if ~obj.is_initialized
                error('Cannot evaluate Cauchy Estimator CPDF before it has been initialized (or after shutdown has been called)!');
            end
            
            if obj.step_count < 1
                error('Cannot evaluate Cauchy Estimator 1D Marginal CPDF before it has been stepped!');
            end
            
            if ~( (marg_idx>0) && (marg_idx <= obj.n) )
                error('Calling marginal 1d cpdf required marg_idx >= 1 and marg_idx <=n');
            end
        
            marg_idx = int32(marg_idx);
            gridx_low = double(gridx_low);
            gridx_high = double(gridx_high);
            gridx_resolution = double(gridx_resolution);
            assert(gridx_high > gridx_low, 'Grid x-axis high limit must be greater than low limit.');
            assert(gridx_resolution > 0, 'Grid x-axis resolution must be positive.');
            
            if nargin < 6 || isempty(log_dir)
                log_dir = 'n'; % No log directory
            else
                log_dir = char(log_dir); % Convert to char array
            end
        
            [cpdf_points, num_gridx] = mcauchy_get_marginal_1D_pointwise_cpdf(obj.mcauchy_handle, marg_idx-1, gridx_low, gridx_high, gridx_resolution, log_dir);
            X = squeeze(cpdf_points(1, :))';
            Y = squeeze(cpdf_points(2, :))';
        end
        

        function [X, Y] = get_1D_pointwise_cpdf(obj, gridx_low, gridx_high, gridx_resolution, log_dir)
            if ~obj.is_initialized
                error('Cannot evaluate Cauchy Estimator CPDF before it has been initialized (or after shutdown has been called)!');
            end
            
            if obj.n ~= 1
                error('Cannot evaluate Cauchy Estimator 1D CPDF for a %d-state system!', obj.n);
            end
        
            if obj.step_count < 1
                error('Cannot evaluate Cauchy Estimator 1D CPDF before it has been stepped!');
            end
            
            if nargin < 5 || isempty(log_dir)
                log_dir = []; % Default log directory
            end
            
            [X, Y] = obj.get_marginal_1D_pointwise_cpdf(1, gridx_low, gridx_high, gridx_resolution, log_dir);
        end


        function plot_2D_pointwise_cpdf(X, Y, Z, state_labels)
            if nargin < 4
                state_labels = [1, 2];
            end
            
            figure;
            
            % Create 3D axes
            ax = axes('NextPlot', 'add', 'DataAspectRatio', [1 1 1], 'PlotBoxAspectRatio', [2 2 1]);
            view(3); % Set the view to 3D
            
            % Plot the wireframe
            wireframe = mesh(ax, X, Y, Z, 'EdgeColor', 'b');
            
            xlabel(ax, sprintf('x-axis (State-%d)', state_labels(1)));
            ylabel(ax, sprintf('y-axis (State-%d)', state_labels(2)));
            zlabel(ax, 'z-axis (CPDF Probability)');
            % set(ax, 'FontSize', 14);
            grid(ax, 'on');
            title(ax, sprintf('Cauchy Estimator''s CPDF for States %d and %d', state_labels(1), state_labels(2)));
        
            % makes the plot interactive
            % rotate3d on;
        end


        function plot_1D_pointwise_cpdf(x, y, state_label)
            if nargin < 3
                state_label = 1;
            end
            
            figure;
            
            plot(x, y, 'b-'); 
            
            xlabel(sprintf('State-%d', state_label));
            ylabel('CPDF Probability');
            grid on;
            title(sprintf('Cauchy Estimator''s 1D CPDF for State %d', state_label));
        end
        
        

        % NEED TO IMPLEMENT THE FUNCTIONS BELOW
        
        % Placeholder for 'initialize_ltv' method
        function initialize_ltv(obj, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, dynamics_update_callback, init_step, dt)
            disp('initialize_ltv method is not yet implemented.');
            % Insert code for LTV initialization
        end

        function initialize_nonlin(obj, x0, A0, p0, b0, beta, gamma, dynamics_update_callback, nonlinear_msmt_model, extended_msmt_update_callback, cmcc, dt, step)
            if nargin < 13
                step = 0;
            end
            if nargin < 12
                dt = 0;
            end
            
            if ~strcmp(obj.mode, "nonlin")
                fprintf('Attempting to call initialize_lti method when mode was set to %s is not allowed! You must call initialize_%s ... or reset the mode altogether!\n', obj.mode, obj.mode);
                fprintf('NonLin initialization not successful!\n');
                return;
            end

            if isstring(dynamics_update_callback)
                f_duc_name1 = char(dynamics_update_callback);
            else
                f_duc_name1 = dynamics_update_callback;
            end
            if isstring(nonlinear_msmt_model)
                f_duc_name2 = char(nonlinear_msmt_model);
            else
                f_duc_name2 = nonlinear_msmt_model;
            end
            if isstring(extended_msmt_update_callback)
                f_duc_name3 = char(extended_msmt_update_callback);
            else
                f_duc_name3 = extended_msmt_update_callback;
            end
            
            obj.n = obj.init_params_checker(A0, p0, b0);  
            
            if isempty(beta)
                obj.pncc = 0;
            else
                assert(isvector(beta), 'beta must be a vector.');
                obj.pncc = length(beta);
            end
            assert(isvector(gamma), 'gamma must be a vector.');
            obj.p = length(gamma);
            obj.cmcc = int32(cmcc);
            assert(numel(x0) == obj.n, 'Size of x0 must match the system dimension.');

            if obj.free_names_ptr ~= 0
                mcauchy_free_names(obj.free_names_ptr);
            end
            obj.free_names_ptr = 0;

            [obj.f_duc_ptr1, obj.f_duc_ptr2, obj.f_duc_ptr3, obj.free_names_ptr] = ...
                mcauchy_get_nonlin_function_pointers(f_duc_name1, f_duc_name2, f_duc_name3);

            obj.x0 = double(x0(:));
            obj.A0 = double(A0(:));
            obj.p0 = double(p0(:));
            obj.b0 = double(b0(:));
            if ~isempty(beta) 
                obj.beta = double(beta(:));
            else 
                obj.beta = [];
            end
            obj.gamma = double(gamma(:));
            
            dt = double(dt);
            step = int32(step);
            
            obj.mcauchy_handle = mcauchy_initialize_nonlin(obj.num_steps, obj.x0, obj.A0, obj.p0, obj.b0, obj.beta, obj.gamma, obj.f_duc_ptr1, obj.f_duc_ptr2, obj.f_duc_ptr3, obj.cmcc, dt, step, obj.debug_print);
            
            obj.is_initialized = true;
            
            fprintf('Nonlin initialization successful! You can use the step(msmts, controls) method now to run the Cauchy Estimator!\n');
            fprintf('Note: You can call the step function %d time-steps, %d measurements per step == %d total times!\n', ...
                    obj.num_steps, obj.p, obj.num_steps * obj.p);
        end

        % Placeholder for 'step_asynchronous' method
        function step_asynchronous(obj, msmts, controls)
            disp('step_asynchronous method is not yet implemented.');
            % Insert code for performing asynchronous step
        end

    end

    methods (Access = private)
        function out = size_checker(obj, in, expected_size, varName)
            % Helper method to check the size of input variables and set to empty if not provided
            if nargin < 2 || isempty(in) % Default to empty if not provided
                out = [];
            else % Given a variable, ensure it has the correct size
                if isequal(size(in), expected_size)
                    out = in;
                else
                    error('Size of %s is incorrect. Expected [%d, %d].', varName, expected_size(1), expected_size(2));
                end
            end
        end
    end

    
end

