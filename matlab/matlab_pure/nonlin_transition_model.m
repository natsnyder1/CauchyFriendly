% Nonlinear transition model from t_k to t_k+1...ie: dt
function x_new = nonlin_transition_model(x)
    global pend;
    x_new = runge_kutta4(@pend_ode, x, pend.dt);
end
