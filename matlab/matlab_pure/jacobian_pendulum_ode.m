function Jac = jacobian_pendulum_ode(x)
    global pend;
    Jac = zeros(2);
    Jac(1,2) = 1;
    Jac(2,1) = -pend.g/pend.L*cos(x(1));
    Jac(2,2) = -pend.c;
end