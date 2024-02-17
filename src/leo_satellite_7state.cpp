#include"../include/cauchy_windows.hpp"
#include"../include/kalman_filter.hpp"

int count;

double lookup_air_density(double r_sat)
{
    if(r_sat == 550e3)
        return 2.384e-13;
    else if(r_sat == 500e3)
        return 5.125e-13;
    else if(r_sat == 450e3)
        return 1.184e-12;
    else if(r_sat == 400e3)
        return 2.803e-12;
    else if(r_sat == 350e3)
        return 7.014e-12;
    else if(r_sat == 300e3)
        return 1.916e-11;
    else if(r_sat == 250e3)
        return 6.073e-11;
    else if(r_sat == 200e3)
        return 2.541e-10;
    else if(r_sat == 150e3)
        return 2.076e-9;
    else if(r_sat == 100e3)
        return 5.604e-7;
    else
    {
        printf("Lookup air density function does not have value for %lf...please add! Exiting!\n", r_sat);
        exit(1);
    }
}

struct leo_satellite_7state
{   
    // Size of simulation dynamics
    int n;
    int p;
    int pncc;
    // Satellite parameter specifics
    double M; // Mass of earth (kg)
    double G; // Universal gravitation constant (m^3/(kg*s^2))
    double mu; // Specific gravitational constant for orbit around earth (m^3/s^2)
    double m; // # Mass of Satellite (kg)
    double rho; //kg/m^3 desnity of air at orbit height
    double C_D; // drag constant
    double A; //Area of satellite perpendicular to travel (m^2)
    double tau; // parameter of drag increment ODE 1/(m*sec)
    // Parameters for runge kutta ODE integrator
    double dt; // time step in sec
    int sub_steps_per_dt; // so sub intervals for ODE integrator are dt / sub_steps_dt 
    // Initial conditions and simulation length
    double r_earth; // spherical approximation of earths radius (m)
    double r_sat; // orbit distance of satellite above earths surface (m)
    double r0; // orbit distance from center of earth (m)
    double v0; // speed of the satellite in orbit for distance r0 (m/s)
    double x0[7]; // The initial true state of the satellite
    double omega0; // rad/sec (angular rate of orbit)
    double orbital_period; // time it takes to make one revolution
    int time_steps_per_period; // number of dt's until 1 revolution is made
    int num_revolutions; // number of times around the earth
    int num_simulation_steps;
    // Satellite parameters for measurement update
    int num_satellites; // number of sattelites to talk to
    double satellite_positions[9]; // location of satellites (treated as beacons)
    double dt_R; // bias time of sattelite clocks, for now its zero
    double b[3]; // bias time of the sattelites, for now its zero
    double std_dev_gps;
    double V[9];
    double cholV[9];
    // Satellite parameters for process noise
    double Wd[1]; // Continous time spectral density matrix held constant over interval dt
    double SAS_alpha; // the SAS alpha value of 1.3 for the density increment pdf
    double beta_drag; // Drag noise given in D-time for intervals of 60 seconds and w.r.t cauchy scale param
    double beta_cauchy; // Cauchy scale param for use in MCE for system x_k+1 = \Phi * x_k + \Gamma_k * w_k, where beta_cauchy is scale param of w_k
    double beta_gauss; // Gaussian std dev param for use in EKF for system x_k+1 = \Phi * x_k + \Gamma_k * w_k, where beta_gauss is std dev of w_k 
    // Initial uncertainty in position
    double P0[49];
    double alpha_density_cauchy; // Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
    double alpha_density_gauss; // Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
    double alpha_pv_gauss; // Initial Gaussian standard deviation in position and velocity of satellite
    double alpha_pv_cauchy; // Initial converted uncertainty parameter in position and velocity of satellite converted for Cauchy Estimator
    double cholP0[49];
    // PDF conversions for scale parameters of one density to another
    double ALPHA13_TO_CAUCHY;
    double ALPHA13_TO_GAUSS;
    double CAUCHY_TO_GAUSS;
    double GAUSS_TO_CAUCHY;

    leo_satellite_7state()
    {
        // Size of simulation dynamics
        n = 7;
        num_satellites = 3; // number of sattelites to talk to (measurements)
        p = num_satellites;
        pncc = 1;
        // Orbital distances
        r_earth = 6378.1e3; // spherical approximation of earths radius (meters)
        r_sat = 200e3; // orbit distance of satellite above earths surface (meters)
        
        // Satellite parameter specifics
        M = 5.9722e24; // Mass of earth (kg)
        G = 6.674e-11; // m^3/(s^2 * kg) Universal Gravitation Constant
        mu = M*G ; //Nm^2/kg^2
        m = 5000.0; // # kg
        rho = lookup_air_density(r_sat); //kg/m^3
        C_D = 2.0; //drag coefficient
        A = 64.0; //m^2
        tau = 21600.0; // 1/(m*sec)
        // Parameters for runge kutta ODE integrator
        dt = 60; // time step in sec
        sub_steps_per_dt = 60; // so sub intervals are dt / sub_steps_dt 
        // Initial conditions
        r0 = r_earth + r_sat; // orbit distance from center of earth
        v0 = sqrt(mu/r0); // speed of the satellite in orbit for distance r0
        x0[0] = sqrt(1)*r0/sqrt(6); x0[1] = sqrt(2)*r0/sqrt(6); x0[2] = sqrt(3)*r0/sqrt(6); x0[3] = -v0/sqrt(3); x0[4] = -v0/sqrt(3); x0[5] = v0/sqrt(3); x0[6] = 0.0;
        omega0 = v0/r0; // rad/sec (angular rate of orbit)
        orbital_period = 2.0*PI / omega0; // Period of orbit in seconds
        time_steps_per_period = (int)(orbital_period / dt + 0.50); // number of dt's until 1 revolution is made
        num_revolutions = 10;
        num_simulation_steps =  num_revolutions * time_steps_per_period;
        // Satellite parameters for measurement update
        satellite_positions[0] = -1e6; satellite_positions[1] = 2e6; satellite_positions[2] = 3e6;
        satellite_positions[3] = 3e6; satellite_positions[4] = -2e6; satellite_positions[5] = 1e6;
        satellite_positions[6] = -6e6; satellite_positions[7] = -5e6; satellite_positions[8] = -2e6;
        dt_R = 0.0; // bias time of sattelite clocks, for now its zero
        b[0] = 0; b[1] = 0; b[2] = 0; // bias time of the sattelites, for now its zero
        std_dev_gps = 2.0; // uncertainty in GPS measurement
        memset(V,0,p*p*sizeof(double));
        V[0] = pow(std_dev_gps,2); V[4] = pow(std_dev_gps,2); V[8] = pow(std_dev_gps,2);
        memcpy(cholV, V, p*p*sizeof(double));
        cholesky(cholV, p, true);
        // Conversion Parameters 
        SAS_alpha = 1.3;
        CAUCHY_TO_GAUSS = 1.3898;
        GAUSS_TO_CAUCHY = 1.0 / CAUCHY_TO_GAUSS;
        beta_drag = 0.0013;
        beta_gauss = (beta_drag * CAUCHY_TO_GAUSS) / (tau * (1.0 - exp(-dt/tau)));
        beta_cauchy = beta_gauss * GAUSS_TO_CAUCHY;
        // Satellite parameters for process noise
        Wd[0] = pow(beta_gauss, 2);
        // Initial uncertainty in position
        alpha_density_cauchy = 0.0039; // Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
        alpha_density_gauss = alpha_density_cauchy * CAUCHY_TO_GAUSS; // Cauchy uncertainty parameter of initial density coefficient (given by Carpenter)
        alpha_pv_gauss = 1.0; // Initial Gaussian standard deviation in position and velocity of satellite
        alpha_pv_cauchy = alpha_pv_gauss * GAUSS_TO_CAUCHY; // Initial converted uncertainty parameter in position and velocity of satellite converted for Cauchy Estimator
        memset(P0, 0, n*n*sizeof(double));
        for(int i = 0; i < 6; i++)
            P0[i*n+i] = pow(alpha_pv_gauss,2);
        P0[n*n-1] = pow(alpha_density_gauss,2);
        memcpy(cholP0, P0, n*n*sizeof(double));
        cholesky(cholP0, n, true);
    }
};

// k1,2,3,4 are size n
// work1,2,3,4 are size n
// x_new is size n
// x_old is size n
// params is a void pointer to a data container needed by the function (pointer) f
void runge_kutta4(double* k1, double* k2, double* k3, double* k4, double* work1, double* work2, double* work3,
                  void (*f)(double* dx_dt, double* x, void* params), int n, double* x_new, double* x_old, double dt, void* params)
{
    // Compute k1
    (*f)(k1, x_old, params);
    // Compute k2
    memcpy(work1, k1, n*sizeof(double));
    scale_vec(work1, dt/2.0, n);
    add_vecs(work1, x_old, n);
    (*f)(k2, work1, params);
    // Compute k3
    memcpy(work2, k2, n*sizeof(double));
    scale_vec(work2, dt/2.0, n);
    add_vecs(work2, x_old, n);
    (*f)(k3, work2, params);
    // Compute k4
    memcpy(work3, k3, n*sizeof(double));
    scale_vec(work3, dt, n);
    add_vecs(work3, x_old, n);
    (*f)(k4, work3, params);
    // Form x_new
    scale_vec(k2, 2.0, n);
    scale_vec(k3, 2.0, n);
    // x_new = x + 1.0 / 6.0 * (k1 + 2*k2 + 2*k3 + k4) * dt 
    // Use work1 as tmp space
    add_vecs(k1,k4,work1, n);
    add_vecs(work1,k2,n);
    add_vecs(work1,k3,n);
    scale_vec(work1, dt/6.0, n);
    add_vecs(x_old, work1, x_new, n);
}
// returns Central Difference Gradient of vector f, the matrix Jacobian, 4th Order expansion
// m is the output size of the vector function f, i.e f(x).size
// n is the size of x
// eps is the step size of the numerical difference to compute derivative (1e-5 is good)
// params is a void pointer to a data container needed by the function (pointer) f
// G is pointer to a row-major matrix of size m x n. Gradients returned as [df(x)/dx1; df(x)/dx2; ... ; df(x)/dxn]
void cd4_gvf(void(*f)(double* y, double* x, void* params), int m, int n, double eps, double* G, double* x, void* params)
{
    //numerical gradient
    double work1[m];
    double work2[m];
    double work3[m];
    double work4[m];
    for(int i = 0; i < n; i++)
    {
        // f(x + 2.0*ep*ei)
        x[i] += 2.0*eps;
        (*f)(work1, x, params);
        // f(x + ep*ei)
        x[i] -= eps;
        (*f)(work2, x, params);
        // f(x - ep*ei)
        x[i] -= 2.0*eps;
        (*f)(work3, x, params);
        // f(x - 2.0*ep*ei)
        x[i] -= eps;
        (*f)(work4, x, params);
        // Reset x
        x[i] += 2.0*eps;
        // G[:,i] = (-1.0 * f(x + 2.0*ep*ei) + 8.0*f(x + ep*ei) - 8.0 * f(x - ep*ei) + f(x - 2.0*ep*ei) ) / (12.0*ep)
        scale_vec(work1, -1.0, m);
        scale_vec(work2, 8.0, m);
        scale_vec(work3, -8.0, m);
        add_vecs(work1, work2, m);
        add_vecs(work1, work3, m);
        add_vecs(work1, work4, m);
        scale_vec(work1, 1.0 / (12.0 * eps), n);
        for(int j = 0; j < m; j++)
            G[i + j*n] = work1[j];
    }
}

void leo_ode_7state(double* dx_dt, double* x, void* params)
{
    leo_satellite_7state* leo = (leo_satellite_7state*)(params);
    double mu = leo->mu;
    double A = leo->A;
    double C_D = leo->C_D;
    double m = leo->m;
    double rho = leo->rho;
    double tau = leo->tau;
    double r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]); //np.linalg.norm(pos)
    double v = sqrt(x[3]*x[3] + x[4]*x[4] + x[5]*x[5]); //np.linalg.norm(vel)
    dx_dt[0] = x[3]; 
    dx_dt[1] = x[4];
    dx_dt[2] = x[5];
    dx_dt[3] = -(mu)/pow(r,3) * x[0] - 0.5*A*C_D/m*rho*(1+x[6])*v*x[3];
    dx_dt[4] = -(mu)/pow(r,3) * x[1] - 0.5*A*C_D/m*rho*(1+x[6])*v*x[4];
    dx_dt[5] = -(mu)/pow(r,3) * x[2] - 0.5*A*C_D/m*rho*(1+x[6])*v*x[5];
    dx_dt[6] = -1.0 / tau * x[6];
}

void leo_7state_transition_model(leo_satellite_7state* leo, double* xk1, double* x)
{
    int n = leo->n;
    int rk4_steps = leo->sub_steps_per_dt;
    double rk4_dt = leo->dt / rk4_steps;
    double k1[n];
    double k2[n];
    double k3[n];
    double k4[n];
    double work1[n];
    double work2[n];
    double work3[n];
    double x_old[n];
    double x_new[n];
    double* xo = x_old;
    double* xn = x_new;
    memcpy(x_old, x, n*sizeof(double));
    for(int i = 0; i < rk4_steps; i++)
    {
        runge_kutta4(k1, k2, k3, k4, work1, work2, work3, leo_ode_7state, leo->n, xn, xo, rk4_dt, leo);
        ptr_swap(&xn, &xo);
    }
    memcpy(xk1, xo, n*sizeof(double));
}

void leo_7state_measurement_model(leo_satellite_7state* leo, double* x, double* z)
{
    double c = 299792458; // spped of light (m/s)
    int p = leo->p;
    double* sat_pos = leo->satellite_positions;
    for(int i = 0; i < p; i++)
        z[i] = sqrt(pow(x[0] - sat_pos[p*i],2) + pow(x[1] - sat_pos[p*i + 1],2) + pow(x[2] - sat_pos[p*i + 2],2)) - c*leo->dt_R + leo->b[i];
}

void leo_7state_transition_model_jacobians(double* Phi_k, double* Gamma_k, double* x, leo_satellite_7state* leo)
{
    int n = 7;
    int pncc = 1;
    double dt = leo->dt;
    double eps = 1e-5;
    int taylor_order = 6;
    double fact;
    double G[n*n];
    double work1[n*n];
    double work2[n*n];
    double work3[n*n];
    double* w1;
    double* w2; 
    cd4_gvf(&leo_ode_7state, n, n, eps, G, x, leo);
    // Form Discrete Time Transition Matrix
    memset(work3, 0, n*n*sizeof(double));
    for(int i = 0; i < n; i++)
        work3[i*n+i] = 1.0;
    memcpy(work1, G, n*n*sizeof(double));
    add_mat(work3, work1, dt, n, n);
    w1 = work1;
    w2 = work2; 
    fact = 1;
    for(int i = 2; i <= taylor_order; i++)
    {
        fact *= i;
        matmatmul(w1, G, w2, n, n, n, n);
        add_mat(work3, w2, pow(dt,i)/fact, n, n); // G^i*dt^i/factorial(i)
        ptr_swap(&w1, &w2);
    }
    memcpy(Phi_k, work3, n*n*sizeof(double));
    // Form Discrete Time Control Matrix
    memset(work3, 0, n*n*sizeof(double));
    for(int i = 0; i < n; i++)
        work3[i*n+i] = dt;
    fact = 2;
    memcpy(work1, G, n*n*sizeof(double));
    add_mat(work3, work1, dt*dt/fact, n, n); 
    w1 = work1;
    w2 = work2; 
    for(int i = 2; i <= taylor_order; i++)
    {
        fact *= (i+1);
        matmatmul(w1, G, w2, n, n, n, n);
        add_mat(work3, w2, pow(dt,i+1)/fact, n, n); // G^i*dt^(i+1)/factorial(i+1)
        ptr_swap(&w1, &w2);
    }
    double Gamma[n*pncc];
    memset(Gamma, 0, n*pncc*sizeof(double));
    Gamma[n*pncc-1] = 1;
    matmatmul(work3, Gamma, Gamma_k, n, n, n, pncc);
}

void leo_7state_measurement_model_jacobian(double* H_k, double* x, leo_satellite_7state* leo)
{
    int n = 7;
    int p = leo->p;
    double* sp; // satellite positions
    memset(H_k, 0, p*n*sizeof(double));
    for(int i = 0; i < p; i++)
    {
        sp = leo->satellite_positions + i*p;
        double dr = sqrt(pow(x[0] - sp[0],2) + pow(x[1] - sp[1],2) + pow(x[2] - sp[2],2));
        H_k[i*n] = (x[0] - sp[0]) / dr;
        H_k[i*n+1] = (x[1] - sp[1]) / dr;
        H_k[i*n+2] = (x[2] - sp[2]) / dr;
    }
}

void leo_7state_simulation_transition_model(KalmanDynamicsUpdateContainer* duc, double* xk1, double* w)
{
    leo_satellite_7state* leo = (leo_satellite_7state*)(duc->other_stuff);
    leo_7state_transition_model(leo, xk1, duc->x);
    int n = leo->n;
    //w[0] = random_normal(0, leo->beta_drag * leo->CAUCHY_TO_GAUSS);
    w[0] = random_symmetric_alpha_stable(leo->SAS_alpha, leo->beta_drag, 0);
    w[0] = fabs(w[0]) < 10 ? w[0] : 10*sgn(w[0]);
    
    if(count == 20)
        w[0] = 1.50;
    else if (count == 60)
        w[0] = -1.00;
    else if (count == 100)
        w[0] = -0.75;
    count += 1;
    
    xk1[n-1] += w[0]; // f(x) + w
}

void leo_7state_simulation_measurement_model(KalmanDynamicsUpdateContainer* duc, double* z, double* v)
{
    leo_satellite_7state* leo = (leo_satellite_7state*)(duc->other_stuff);
    leo_7state_measurement_model(leo, duc->x, z);
    int p = duc->p;
    double work1[p];    
    double work2[p]; 
    memset(work1, 0, p*sizeof(double));
    multivariate_random_normal(v, work1, leo->cholV, work2, p);
    add_vecs(z,v,p); //h(x) + v
}

// EKF Simulation Callbacks
void ekf_leo_7state_transition_model(KalmanDynamicsUpdateContainer* duc, double* xk1)
{
    leo_satellite_7state* leo = (leo_satellite_7state*)(duc->other_stuff);
    leo_7state_transition_model(leo, xk1, duc->x);
}
// Constructs the estimate of the measurement for the EKF: \bar{z_k} = h(\bar{x_k|k-1}) (using the nonlinear measurement model)
void ekf_leo_7state_msmt_model(KalmanDynamicsUpdateContainer* duc, double* z)
{
    leo_satellite_7state* leo = (leo_satellite_7state*)(duc->other_stuff);
    leo_7state_measurement_model(leo, duc->x, z);
}

void ekf_leo_7state_transition_model_jacobians(KalmanDynamicsUpdateContainer* duc)
{
    leo_satellite_7state* leo = (leo_satellite_7state*)(duc->other_stuff);
    leo_7state_transition_model_jacobians(duc->Phi, duc->Gamma, duc->x, leo);
}

void ekf_leo_7state_measurement_model_jacobian(KalmanDynamicsUpdateContainer* duc)
{
    leo_satellite_7state* leo = (leo_satellite_7state*)(duc->other_stuff);
    leo_7state_measurement_model_jacobian(duc->H, duc->x, leo);
}

// Cauchy Simulation Callbacks
void ece_leo_7state_transition_model_and_jacobians(CauchyDynamicsUpdateContainer* duc)
{
    int n = duc->n;
    double xbar[n];
    leo_satellite_7state* leo = (leo_satellite_7state*)(duc->other_stuff);
    // Set Phi_k, Gamma_k
    leo_7state_transition_model_jacobians(duc->Phi, duc->Gamma, duc->x, leo);
    // Set xbar
    leo_7state_transition_model(leo, xbar, duc->x);
    // Copy xbar to duc->x and set 'is_xbar_set_for_ece' boolean (assures user sets xbar)
    memcpy(duc->x, xbar, n*sizeof(double));
    duc->is_xbar_set_for_ece = true;
    // Set H_k
    leo_7state_measurement_model_jacobian(duc->H, duc->x, leo);
}

void ece_leo_7state_measurement_model(CauchyDynamicsUpdateContainer* duc, double* zbar)
{
    leo_satellite_7state* leo = (leo_satellite_7state*)(duc->other_stuff);
    leo_7state_measurement_model(leo, duc->x, zbar);
}

void ece_leo_7state_measurement_jacobian(CauchyDynamicsUpdateContainer* duc)
{
    leo_satellite_7state* leo = (leo_satellite_7state*)(duc->other_stuff);
    leo_7state_measurement_model_jacobian(duc->H, duc->x, leo);
}

void test_7state_leo()
{
    unsigned int seed = 10; //time(NULL);
    printf("Seeding with %u \n", seed);
    srand ( seed );
    char log_dir[50] = "../log/leo7/gtable/w4";

    count = 0;
    leo_satellite_7state leo;
    // Dimensions of dynamics
    const int n = leo.n;    
    const int cmcc = 0;   
    int pncc = leo.pncc; // sample from 5x5 W first in simulation
    const int p = leo.p;
    const int sim_num_steps = 300; //leo.num_simulation_steps;

    // Initializing Dynamics for the Telegraph Simulation
    double Phi[n*n]; 
    double Gamma[n*leo.pncc]; 
    double H[p*n];

    // Declare the starting state for the simulation
    double x0[n];
    for(int i = 0; i < n-1; i++)
        x0[i] = random_normal( leo.x0[i], leo.alpha_pv_gauss ); 
    x0[n-1] = random_normal( leo.x0[n-1], leo.alpha_density_gauss ); 

    // Initializing the Kalman Filter noises and covariances for the nonlinear simulation
    double x0_kf[n];
    memcpy(x0_kf, leo.x0, n*sizeof(double));
    double P0[n*n];
    memcpy(P0, leo.P0, n*n*sizeof(double));
    double* W = leo.Wd;
    //W[0] *= 10000;
    KalmanDynamicsUpdateContainer kduc;
        kduc.n = n; kduc.pncc = pncc; 
        kduc.cmcc = cmcc; kduc.p = p;
        kduc.dt = leo.dt; kduc.step = 0;
        kduc.Phi = Phi; kduc.Gamma = Gamma; 
        kduc.B = NULL; kduc.H = H;
        kduc.W = W; kduc.V = leo.V;
        kduc.x = x0_kf;
        kduc.u = NULL;
        kduc.other_stuff = &leo;
    assert_correct_kalman_dynamics_update_container_setup(&kduc);

    SimulationLogger sim_log(log_dir, sim_num_steps, x0, &kduc,  &leo_7state_simulation_transition_model, &leo_7state_simulation_measurement_model);
    sim_log.run_simulation_and_log();
    
    // Run the EKF Simulation
    double kf_state_history[(sim_num_steps+1)*n];
    double kf_covar_history[(sim_num_steps+1)*n*n];
    double kf_Ks_history[sim_num_steps*n*p];
    double kf_residual_history[sim_num_steps*p];
    run_extended_kalman_filter(sim_num_steps, 
        NULL, sim_log.msmt_history + p, // starts at x0, runs time prop, then computes E[x1|z1]
        kf_state_history, kf_covar_history,
        kf_Ks_history, kf_residual_history,
        x0_kf, P0,
        Phi, Gamma, Gamma,
        H, W, leo.V,
        n, cmcc, pncc, p,
        &ekf_leo_7state_transition_model,
        &ekf_leo_7state_msmt_model,
        &ekf_leo_7state_transition_model_jacobians,
        &ekf_leo_7state_measurement_model_jacobian, 
        &kduc, 
        false
        );
    log_kf_data(log_dir, kf_state_history, kf_covar_history, kf_residual_history, sim_num_steps+1, n, p);
    
    ///*
    // Now run the Cauchy Simulation
    double x0_ce[n];
    //memcpy(x0_ce, leo.x0, n*sizeof(double));
    memcpy(x0_ce, sim_log.true_state_history, n*sizeof(double));
    double beta[pncc];
    double gamma[p];
    beta[0] = leo.beta_cauchy; // / 100
    for(int i = 0; i < p; i++)
        gamma[i] = leo.std_dev_gps * leo.GAUSS_TO_CAUCHY; // *50

    // Create Phi.T as A0
    // Initialize Initial Hyperplanes
    leo_7state_transition_model_jacobians(Phi, Gamma, x0_ce, &leo);
    double A0[n*n];
    memcpy(A0, Phi, n*n*sizeof(double));
    reflect_array(A0, n, n); // eye(n) @ Phi.T
    double p0[n];
    for(int i = 0; i < n-1; i++)
        p0[i] = leo.alpha_pv_cauchy;
    p0[n-1] = leo.alpha_density_cauchy;
    double b0[n];
    memset(b0, 0, n * sizeof(double) );

    CauchyDynamicsUpdateContainer duc;
    duc.n = n; duc.cmcc = cmcc; duc.p = p; duc.pncc = pncc;
    duc.x = x0_ce; duc.dt = leo.dt; duc.step = 0; //tp_start; // step will be 0 if tp_start is false and 1 if tp_start is true
    duc.Phi = Phi; duc.Gamma = Gamma;
    duc.u = NULL; duc.B = NULL;
    duc.H = H; duc.beta = beta; duc.gamma = gamma;
    duc.other_stuff = &leo;
    assert_correct_cauchy_dynamics_update_container_setup(&duc);
    //*/

    int ftr_idx_ordering[7] = {5,4,3,6,2,1,0};
    set_tr_search_idxs_ordering(ftr_idx_ordering, 7);

    ///*
    int foo_steps = 4;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, foo_steps, n, cmcc, pncc, p, print_basic_info);
    for(int i = 0; i < foo_steps; i++)
    {
        double* msmts = sim_log.msmt_history + (i+1)*p;
        ece_leo_7state_transition_model_and_jacobians(&duc);
        for(int j = 0; j < p; j++)
        {
            double zbar[p];
            ece_leo_7state_measurement_jacobian(&duc);
            ece_leo_7state_measurement_model(&duc, zbar);
            double msmt = msmts[j] - zbar[j];
            printf("Processing measurement z=%.4lf, which is #%d/%d at step %d/%d\n", msmt, j+1, p, i+1, foo_steps);
            cauchyEst.step(msmt, Phi, Gamma, beta, H + j*n, gamma[j], NULL, NULL); 
            cauchyEst.finalize_extended_moments(duc.x);

            printf("True State is:\n");
            print_mat(sim_log.true_state_history + (i+1)*n, 1, n);
            printf("Cauchy Conditional Mean is:\n");
            print_cmat(cauchyEst.conditional_mean, 1, n, 5);
        }
    }
    //*/

    /*
    const bool is_extended = true;
    ece_leo_7state_transition_model_and_jacobians(&duc);
    ece_leo_7state_measurement_jacobian(&duc);
    const int total_steps = sim_num_steps+1;
    const int num_windows = 4; 
    const bool WINDOW_PRINT_DEBUG = true;
    const bool WINDOW_LOG_SEQUENTIAL = true;
    const bool WINDOW_LOG_FULL = true;
    double window_var_boost[7] = {1,1,1,1,1,1,1};
    SlidingWindowManager swm( num_windows, total_steps-1, // We skip msmt at time step 0 since starting with "propagation A/p/b" 
            A0, p0, b0, &duc, 
            WINDOW_PRINT_DEBUG, WINDOW_LOG_SEQUENTIAL, WINDOW_LOG_FULL, 
            is_extended, ece_leo_7state_transition_model_and_jacobians, 
            ece_leo_7state_measurement_model, ece_leo_7state_measurement_jacobian, 
            window_var_boost, log_dir);
    for(int i = 1; i < total_steps; i++)
        swm.step(sim_log.msmt_history + i*p, NULL);
    swm.shutdown();
    */
}

int main()
{
    test_7state_leo();
    return 0;
}
