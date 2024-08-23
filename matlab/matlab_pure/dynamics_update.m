% This is the callback function correpsonding to the decription for point 1.) above 
function dynamics_update(c_duc)
    global pend;
    Gamma_c = [0.0; 1.0];
    taylor_order = 2;
    mduc = M_CauchyDynamicsUpdateContainer(c_duc);
    %% Propagate x 
    xk = mduc.cget_x();
    xbar = nonlin_transition_model(xk); % propagate from k -> k+1
    mduc.cset_x(xbar);
    mduc.cset_is_xbar_set_for_ece(); % need to call this!
    %% Phi, Gamma, beta may update
    Jac_F = jacobian_pendulum_ode(xk);
    [Phi_k, Gam_k] = discretize_nl_sys(Jac_F, Gamma_c, [], pend.dt, taylor_order, true, false);
    mduc.cset_Phi(Phi_k);
    mduc.cset_Gamma(Gam_k);
    %mduc.cset_beta(beta)
end