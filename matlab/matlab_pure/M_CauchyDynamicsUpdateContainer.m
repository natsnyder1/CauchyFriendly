% file: M_CauchyDynamicsUpdateContainer

classdef M_CauchyDynamicsUpdateContainer < handle
    properties
        cduc;
        n;
        pncc;
        cmcc;
        p;
    end
    
    methods
        function obj = M_CauchyDynamicsUpdateContainer(cduc)
            obj.cduc = cduc;
            obj.n = mcauchy_dynamics_get_n(obj.cduc);
            obj.pncc = mcauchy_dynamics_get_pncc(obj.cduc);
            obj.cmcc = mcauchy_dynamics_get_cmcc(obj.cduc);
            obj.p = mcauchy_dynamics_get_p(obj.cduc);
        end

        function step = cget_step(obj)
            step = mcauchy_dynamics_get_step(obj.cduc);
        end
        
        function dt = cget_dt(obj)
            dt = mcauchy_dynamics_get_dt(obj.cduc);
        end

        function x = cget_x(obj)
            x = mcauchy_dynamics_get_x(obj.cduc);
        end
        
        function obj = cset_x(obj, x)
            assert(isvector(x) && length(x) == obj.n);
            mcauchy_dynamics_set_x(obj.cduc, x);
        end

        function u = cget_u(obj)
            u = mcauchy_dynamics_get_u(obj.cduc);
        end

        function Phi = cget_Phi(obj)
            Phi = mcauchy_dynamics_get_Phi(obj.cduc);
        end

        function obj = cset_Phi(obj, Phi)
            assert(ismatrix(Phi) && all(size(Phi) == [obj.n obj.n]));
            mcauchy_dynamics_set_Phi(obj.cduc, Phi);
        end

        function Gamma = cget_Gamma(obj)
            Gamma = mcauchy_dynamics_get_Gam(obj.cduc);
        end

        function obj = cset_Gamma(obj, Gamma)
            assert(ismatrix(Gamma) && all(size(Gamma) == [obj.n obj.pncc]));
            mcauchy_dynamics_set_Gam(obj.cduc, Gamma);
        end

        function B = cget_B(obj)
            B = mcauchy_dynamics_get_B(obj.cduc);
        end

        function obj = cset_B(obj, B)
            assert(ismatrix(B) && all(size(B) == [obj.n obj.cmcc]));
            mcauchy_dynamics_set_B(obj.cduc, B);
        end

        function obj = cset_beta(obj, beta)
            assert(isvector(beta) && length(beta) == obj.pncc);
            mcauchy_dynamics_set_beta(obj.cduc, beta);
        end

        function H = cget_H(obj)
            H = mcauchy_dynamics_get_H(obj.cduc);
            H = H';
        end

        function obj = cset_H(obj, H)
            assert(ismatrix(H) && all(size(H) == [obj.p obj.n]));
            mcauchy_dynamics_set_H(obj.cduc, H);
        end

        function gamma = cget_gamma(obj)
            gamma = mcauchy_dynamics_get_gamma(obj.cduc);
        end

        function obj = cset_gamma(obj, gamma)
            assert(isvector(gamma) && length(gamma) == obj.p, 'The input must be a vector of length p.');
            mcauchy_dynamics_set_gamma(obj.cduc, gamma);
        end

        function obj = cset_is_xbar_set_for_ece(obj)
            mcauchy_dynamics_set_is_xbar_set_for_ece(obj.cduc, true);
        end
        
        function obj = cset_zbar(obj, c_zbar, zbar)
            assert(isvector(zbar) && length(zbar) == obj.p, 'zbar must be a vector of length p.');
            mcauchy_dynamics_set_zbar(obj.cduc, c_zbar, zbar, obj.p);
        end
        
    end
end