% The ODE
function dx_dt = pend_ode(x)
    global pend;
    dx_dt = zeros(2, 1);
    dx_dt(1) = x(2);
    dx_dt(2) = -pend.g / pend.L * sin(x(1)) - pend.c * x(2);
end