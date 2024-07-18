% file: runge_kutta4.m

% runge kutta integrator
function x_new = runge_kutta4(f, x, dt)
    k1 = f(x);
    k2 = f(x + dt*k1/2.0);
    k3 = f(x + dt*k2/2.0);
    k4 = f(x + dt*k3);
    x_new = x + 1.0 / 6.0 * (k1 + 2*k2 + 2*k3 + k4) * dt;
end

