% file: MatCauchyEstimator

classdef MatCauchyEstimator < handle
    properties
        modes = {'lti', 'ltv', 'nonlin'}
        mode
        num_steps
        debug_print = true

        n
        pncc
        cmcc
        p

        moment_info = struct('x', {}, 'P', {}, 'cerr_x', {}, 'cerr_P', {}, 'fz', {}, 'cerr_fz', {}, 'err_code', {}) % Cell arrays instead of lists
        fz
        step_count = 0

        A0
        p0
        b0
        Phi
        Gamma
        beta
        B
        H
        gamma

        matcauchy_handle
        is_initialized = false

        msmts
        controls

        x
        P

        xbar
        zbar
    end
    
    methods (Access = public)
        function obj = MatCauchyEstimator(mode, num_steps, debug_print)
            if nargin < 3
                debug_print = true;
            end

            mode = lower(mode);
            if ~ismember(mode, obj.modes)
                fprintf('[Error MatCauchyEstimator:] chosen mode %s invalid. Please choose one of the following: {%s}\n', mode, strjoin(obj.modes, ', '));
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
                if ndims(Gamma) == 2
                    assert(size(beta, 2) == size(Gamma, 2), 'Dimension mismatch between Gamma and beta.');
                else
                    assert(length(beta) == 1, 'If Gamma is a vector, beta must be scalar.');
                end
                pncc = length(beta);
            else
                assert(isempty(beta), 'If Gamma is empty, beta must also be empty.');
            end
            
            cmcc = 0;
            if ~isempty(B)
                if ndims(B) == 2
                    cmcc = size(B, 2);
                else
                    cmcc = 1;
                end
                assert(size(B, 1) == n, 'B must have the same number of rows as dimension of A0.');
            end
            
            p = size(H, 1);
            if size(H, 2) == 1
                assert(length(gamma) == 1, 'If H is a vector, gamma must be scalar.');
            else
                assert(length(gamma) == p, 'Dimension mismatch between H and gamma.');
            end
            
            if any(abs(H * Gamma) < 1e-12, 'all')
                warning('Warning MatCauchyEstimator: | H @ Gamma | < eps for some input / output channels. This may result in undefined moments!');
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
            % Placeholder for Python pycauchy call, assuming a similar interface is available in MATLAB
            if strcmp(obj.mode, 'lti')
                [~, ~, ~, ~, ~, ~, ...
                obj.fz, obj.x, obj.P, ...
                obj.moment_info.cerr_fz, obj.moment_info.cerr_x, obj.moment_info.cerr_P, obj.moment_info.err_code] = ...
                    matcauchy_single_step_ltiv(obj.matcauchy_handle, msmts, controls);
            elseif strcmp(obj.mode, 'ltv')
                [obj.Phi, obj.Gamma, obj.B, obj.H, obj.beta, obj.gamma, ...
                obj.fz, obj.x, obj.P, ...
                obj.moment_info.cerr_fz, obj.moment_info.cerr_x, obj.moment_info.cerr_P, obj.moment_info.err_code] = ...
                    matcauchy_single_step_ltiv(obj.matcauchy_handle, msmts, controls);
            else
                [obj.Phi, obj.Gamma, obj.B, obj.H, obj.beta, obj.gamma, ...
                obj.fz, obj.x, obj.P, obj.xbar, obj.zbar, ...
                obj.moment_info.cerr_fz, obj.moment_info.cerr_x, obj.moment_info.cerr_P, obj.moment_info.err_code] = ...
                    matcauchy_single_step_nonlin(obj.matcauchy_handle, msmts, controls, obj.step_count ~= 0);
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
            P = reshape(obj.P(end-obj.n^2+1:end), obj.n, obj.n);
            cerr_fz = obj.moment_info.cerr_fz(end);
            cerr_x = obj.moment_info.cerr_x(end);
            cerr_P = obj.moment_info.cerr_P(end);
            err_code = obj.moment_info.err_code(end);
        
            obj.moment_info.fz(end+1) = fz;
            obj.moment_info.x(end+1) = {x};
            obj.moment_info.P(end+1) = {P};
            obj.moment_info.cerr_x(end+1) = cerr_x;
            obj.morning_info.cerr_P(end+1) = {cerr_P};
            obj.moment_info.cerr_fz(end+1) = cerr_fz;
            obj.moment_info.err_code(end+1) = err_code;
        
            obj.step_count = obj.step_count + 1;
        end


        function initialize_lti(obj, A0, p0, b0, Phi, B, Gamma, beta, H, gamma, init_step, dt)
            if ~strcmp(obj.mode, "lti")
                fprintf(['Attempting to call initialize_lti method when mode was set to %s is not allowed! ' ...
                         'You must call initialize_%s ... or reset the mode altogether!\n'], obj.mode, obj.mode);
                disp('LTI initialization not successful!');
                return;
            end
            
            obj.ndim_input_checker(A0, p0, b0, Phi, B, Gamma, beta, H, gamma);
            
            % Reshape and ensure all inputs are column vectors
            obj.A0 = A0(:);
            obj.p0 = p0(:);
            obj.b0 = b0(:);
            obj.Phi = Phi(:);
            obj.Gamma = [];
            if ~isempty(Gamma)
                obj.Gamma = Gamma(:);
            end
            obj.beta = [];
            if ~isempty(beta)
                obj.beta = beta(:);
            end
            obj.B = [];
            if ~isempty(B)
                obj.B = B(:);
            end
            obj.H = H(:);
            obj.gamma = gamma(:);
            
            % Typecast as needed
            init_step = int32(init_step);
            dt = double(dt);
            
            % Instantiate and initialize the matcauchy object
            obj.matcauchy_handle = matcauchy();
            obj.matcauchy_handle.initialize_lti(obj.A0, obj.p0, obj.b0, obj.Phi, obj.B, obj.Gamma, ...
                                                obj.beta, obj.H, obj.gamma, init_step, dt);
            obj.is_initialized = true;
    
            fprintf('LTI initialization successful! You can use the step(msmts, controls) method to run the estimator now!\n');
            fprintf('Note: You can call the step function %d time-steps, %d measurements per step == %d total times!\n', ...
                    obj.num_steps, obj.p, obj.num_steps * obj.p);
        end



        % add placeholders here for missing functions


        function result = step(obj, msmts, controls, full_info)
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
                result = [];
                return;
            end
            
            if obj.step_count == obj.num_steps
                fprintf("[Error:] Cannot step estimator again, you have already stepped the estimator the initialized number of steps\n");
                fprintf("Not stepping! Please shut estimator down or reset it!\n");
                result = [];
                return;
            end
            
            [msmts, controls] = obj.msmts_controls_checker(msmts, controls);
            obj.msmts = msmts; % In MATLAB, no need to explicitly copy
            obj.controls = controls;
            
            result = obj.call_step(msmts, controls, full_info);
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
            
            if msmt_idx >= obj.p || msmt_idx < -obj.p
                fprintf('[Error get_reinitialization_statistics]: Cannot find reinitialization stats for msmt_idx=%d. The index is out of range -%d <= msmt_idx < %d...(max index is p-1=%d)! Please correct!\n', msmt_idx, -obj.p, obj.p, obj.p-1);
                return;
            end
            
            msmt_idx = int32(msmt_idx);
            if msmt_idx < 0
                msmt_idx = msmt_idx + obj.p;
            end
            
            reinit_msmt = obj.msmts(msmt_idx+1); % Assuming _msmts is 1-indexed in MATLAB
            reinit_xhat = obj.x((msmt_idx*obj.n)+1 : (msmt_idx+1)*obj.n); % Copying is implicit in MATLAB
            reinit_Phat = obj.P((msmt_idx*obj.n*obj.n)+1 : (msmt_idx+1)*obj.n*obj.n);
            reinit_H = obj.H((msmt_idx*obj.n)+1 : (msmt_idx+1)*obj.n);
            reinit_gamma = obj.gamma(msmt_idx+1);

            if ~strcmp(obj.mode, "nonlin")
                A0 = obj.matcauchy_handle.get_reinitialization_statistics(reinit_msmt, reinit_xhat, reinit_Phat, reinit_H, reinit_gamma);
                A0 = reshape(A0, [obj.n, obj.n]);
            else
                reinit_xbar = obj.xbar((msmt_idx*obj.n)+1:(msmt_idx+1)*obj.n);
                reinit_zbar = obj.zbar(msmt_idx+1);
                dx = reinit_xhat - reinit_xbar;
                dz = reinit_msmt - reinit_zbar;
                [A0, p0, b0] = obj.matcauchy_handle.get_reinitialization_statistics(dz, dx, reinit_Phat, reinit_H, reinit_gamma);
                A0 = reshape(A0, [obj.n, obj.n]);
            end
        end
        
        function reset(obj, A0, p0, b0, xbar)
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

            obj.matcauchy_handle.reset(obj.A0, obj.p0, obj.b0, obj.xbar);
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
            if msmt_idx >= obj.p || msmt_idx < -obj.p
                error('[Error reset_about_estimator:] Specified msmt_idx=%d. The index is out of range -%d <= msmt_idx < %d...(max index is p-1=%d)! Please correct!', msmt_idx, -obj.p, obj.p-1, obj.p-1);
            end
            if msmt_idx < 0
                msmt_idx = msmt_idx + obj.p;
            end
            msmt_idx = msmt_idx + 1; % Adjusting for MATLAB 1-based indexing
            msmts = other_estimator.msmts(msmt_idx:end);
            obj.msmts = msmts;
            if ~strcmp(obj.mode, 'nonlin')
                [A0, p0, b0] = other_estimator.get_reinitialization_statistics(msmt_idx);
                obj.reset(A0, p0, b0);
            else
                [A0, p0, b0, xbar] = other_estimator.get_reinitialization_statistics(msmt_idx);
                obj.reset(A0, p0, b0, xbar);
            end
            
            [xs, Ps] = obj.call_step(msmts, []);
            obj.matcauchy_handle.set_master_step(obj.p);
            % return xs, Ps; % Not needed in MATLAB since the variables are in the workspace
        end
        

        function [xs, Ps] = reset_with_last_measurement(obj, z_scalar, A0, p0, b0, xbar)
            z_scalar = reshape(z_scalar, [], 1);
            if ~strcmp(obj.mode, 'nonlin')
                obj.reset(A0, p0, b0);
            else
                obj.reset(A0, p0, b0, xbar);
            end
            [xs, Ps] = obj.call_step(z_scalar, []);
            obj.matcauchy_handle.set_master_step(obj.p);
        end
    

        function set_window_number(obj, win_num)
            win_num = int32(win_num);
            obj.matcauchy_handle.set_window_number(win_num);
        end
    
        function shutdown(obj)
            if ~obj.is_initialized
                error('Cannot shutdown estimator before it has been initialized!');
            end
            obj.matcauchy_handle.shutdown();
            obj.is_initialized = false;
            fprintf('Estimator backend has been shutdown!\n');
        end
        
        % add placeholders for the rest of the functions

    end

    methods (Access = public)
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

