
% input: jacobian of nonliinear dynamics matrix f(x,u) w.r.t x, continous time control matrix G, power spectral density Q of ctime process, change in time dt (time of step k to k+1), order of taylor approximation
% returns: discrete state transition matrix, discrete control matrix, and discrete process noise matrix, given the gradient of f(x,u) w.r.t x
% This function essentially gives you the process model parameter matrices required for the EKF
function varargout = discretize_nl_sys(JacA, G, Q, dt, order, with_Gamk, with_Wk)
    if nargin < 6
        with_Gamk = true;
    end
    if nargin < 7
        with_Wk = true;
    end

    assert(ndims(JacA) == 2, 'JacA must be a 2D matrix.');
    assert(ndims(G) == 2, 'G must be a 2D matrix.');
    if with_Wk
        assert(ndims(Q) == 2, 'Q must be a 2D matrix.')
    end
    assert(dt > 0, 'dt must be positive.');
    assert(order > 0, 'order must be positive.')
    
    n = size(JacA, 1);
    Phi_k = zeros(n, n);
    Gam_k = zeros(size(G));
    W_k = zeros(n, n);
    
    I_n = eye(n); % Identity matrix

    % Form Discrete Time State Transition Matrices Phi_k and Control Gain Matrix Gam_k
    for i = 0:order
        Phi_k = Phi_k + (JacA^i) * (dt^i) / factorial(i);
        if with_Gamk
            Gam_k = Gam_k + (JacA^i * G * dt^(i+1)) / factorial(i+1);
        end
    end

    if with_Wk
        % Form Discrete Time Noise Matrix Qk
        for i = 0:order
            for j = 0:order
                Tk_coef = dt^(i + j + 1) / (i + j + 1);
                tmp_i = (JacA^i / factorial(i)) * G;
                tmp_j = (JacA^j / factorial(j)) * G;
                W_k = W_k + (tmp_i * Q * tmp_j.') * Tk_coef;
            end
        end
    end
    
    % Return variables as requested
    varargout{1} = Phi_k; % Always return Phi_k
    
    if with_Gamk && ~with_Wk
        varargout{2} = Gam_k; % Return Gam_k as the second argument
    elseif ~with_Gamk && with_Wk
        varargout{2} = W_k; % Return W_k as the second argument
    elseif with_Gamk && with_Wk
        varargout{2} = Gam_k; % Return Gam_k as the second argument
        varargout{3} = W_k;  % Return W_k as the third argument
    end
    % Note: If neither with_Gamk nor with_Wk is true, only Phi_k will be returned
end