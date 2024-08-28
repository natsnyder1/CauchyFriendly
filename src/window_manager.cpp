#include "../include/cauchy_windows.hpp"
#include "../include/kalman_filter.hpp"

// Moshes Three State Problem
void test_3state_window_manager()
{
    // Cauchy Settings
    const int n = 3;
    const int pncc = 1;
    const int cmcc = 0;
    const int p = 1;
    double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double Gamma[n*pncc] = {.1, 0.3, -0.2};
    double H[n] = {1.0, 0.5, 0.2};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; //{-0.63335359, -0.74816241, -0.19777826, -0.7710082 ,  0.63199184,  0.07831134, -0.06640465, -0.20208744,  0.97711365}; 
    double p0[n] = {0.10, 0.08, 0.05}; //{0.0, 0.0, 0.0}; //
    double b0[n] = {0, 0, 0};
    //char log_dir[50] = "../log/window_manager";
    char* log_dir = NULL;
    // Kalman Settings
    double x0_kf[n];
    double W[pncc*pncc] = { pow(beta[0]*CAUCHY_TO_GAUSS_NOISE, 2) };
    double V[p*p] = { pow(gamma[0]*CAUCHY_TO_GAUSS_NOISE, 2) };
    double P0[n*n] = { pow(p0[0]*CAUCHY_TO_GAUSS_NOISE, 2), 0, 0, 
                       0, pow(p0[1]*CAUCHY_TO_GAUSS_NOISE, 2), 0, 
                       0, 0, pow(p0[2]*CAUCHY_TO_GAUSS_NOISE, 2)  };
    memcpy(x0_kf, b0, n * sizeof(double));

    // Possibly Set a Seed
    ///* 
    //unsigned int seed = time(NULL);
    //printf("Seeding with %u \n", seed);
    srand ( 11 );
    //*/

    bool is_gaussian_sim = false;
    // Container for Cauchy Estimator
    CauchyDynamicsUpdateContainer duc;
    duc.n = n; duc.pncc = pncc; duc.p = p; duc.cmcc = cmcc;
    duc.Phi = Phi; duc.Gamma = Gamma; duc.H = H; duc.B = NULL; duc.u = NULL;
    duc.beta = beta; duc.gamma = gamma;
    duc.step = 0; duc.dt = 0; duc.other_stuff = NULL; duc.x = NULL;
    // Container for Kalman Estimator
    KalmanDynamicsUpdateContainer kduc;
    kduc.n = n; kduc.pncc = pncc; kduc.p = p; kduc.cmcc = cmcc;
    kduc.Phi = Phi; kduc.Gamma = Gamma; kduc.H = H; kduc.B = NULL; kduc.u = NULL;
    kduc.W = W; kduc.V = V;
    kduc.step = 0; kduc.dt = 0; kduc.other_stuff = NULL; kduc.x = NULL;

    int num_steps = 200;
    SimulationLogger* sim_log;
    if(is_gaussian_sim)
        sim_log = new SimulationLogger(log_dir, num_steps, x0_kf, &kduc, gaussian_lti_transition_model, gaussian_lti_measurement_model);
    else 
        sim_log = new SimulationLogger(log_dir, num_steps, b0, &duc, cauchy_lti_transition_model, cauchy_lti_measurement_model);
    sim_log->run_simulation_and_log();
    
    
    // New Sliding Window Manager
    int total_steps = num_steps+1; // including estimation for x0 (i.e, z0 -> first MU)
    int num_windows = 8;
    const bool WINDOW_PRINT_DEBUG = false;
    const bool WINDOW_LOG_SEQUENTIAL = false;
    const bool WINDOW_LOG_FULL = false;
    const bool is_extended = false;
    double* window_var_boost = NULL;
    SlidingWindowManager swm(num_windows, total_steps, A0, p0, b0, &duc, 
        WINDOW_PRINT_DEBUG, WINDOW_LOG_SEQUENTIAL, WINDOW_LOG_FULL, 
        is_extended, NULL, NULL, NULL, window_var_boost, log_dir);

    // Iterate over each step, window manager iterates over the individual measurements of each steps measurement vector
    for(int i = 0; i < total_steps; i++)
    {
        double* zs = sim_log->msmt_history + i*p;
        swm.step(zs, NULL);
    }
    swm.shutdown();

    // Add Kalman Filter and its logging here
    // Run the KF Simulation
    double kf_state_history[total_steps*n];
    double kf_covar_history[total_steps*n*n];
    double kf_Ks_history[total_steps*n*p];
    double kf_residual_history[num_steps*p];
    run_kalman_filter(num_steps, NULL, sim_log->msmt_history + p,
        kf_state_history, kf_covar_history,
        kf_Ks_history, kf_residual_history,
        x0_kf, P0,
        Phi, NULL, Gamma,
        H, W, V,
        n, cmcc, pncc, p,
        NULL,
        NULL,
        NULL);
    // Write the KF's states, variances, and residuals to the log folder
    if(log_dir != NULL)
        log_kf_data(log_dir, kf_state_history, kf_covar_history, kf_residual_history, total_steps, n, p);

    delete sim_log;
}

// Moshes Three State Problem with 3 Measurement Updates
void test_3state_3msmts_window_manager()
{
    // Cauchy Settings
    const int n = 3;
    const int pncc = 1;
    const int cmcc = 0;
    const int p = 3;
    double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double Gamma[n*pncc] = {.1, 0.3, -0.2};
    double H[p*n] = {1.0, 0.5, 0.2, 
                     0.2, 0.5, 1.0, 
                     -0.5, 1.0, -0.2};
    double beta[pncc] = {0.02};
    double gamma[p] = {0.1, 0.08, 0.06};
    double A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; //{-0.63335359, -0.74816241, -0.19777826, -0.7710082 ,  0.63199184,  0.07831134, -0.06640465, -0.20208744,  0.97711365}; 
    double p0[n] = {0.10, 0.08, 0.05}; //{0.0, 0.0, 0.0}; //
    double b0[n] = {0, 0, 0};
    char log_dir[50] = "../log/window_manager";
    // Kalman Settings
    double x0_kf[n];
    double W[pncc*pncc] = { pow(beta[0]*CAUCHY_TO_GAUSS_NOISE, 2) };
    double V[p*p] = { pow(gamma[0]*CAUCHY_TO_GAUSS_NOISE, 2), 0, 0,
                      0, pow(gamma[1]*CAUCHY_TO_GAUSS_NOISE, 2), 0,
                      0, 0, pow(gamma[2]*CAUCHY_TO_GAUSS_NOISE, 2)};
    double P0[n*n] = { pow(p0[0]*CAUCHY_TO_GAUSS_NOISE, 2), 0, 0, 
                       0, pow(p0[1]*CAUCHY_TO_GAUSS_NOISE, 2), 0, 
                       0, 0, pow(p0[2]*CAUCHY_TO_GAUSS_NOISE, 2)  };
    memcpy(x0_kf, b0, n * sizeof(double));

    // Possibly Set a Seed
    /* 
    unsigned int seed = time(NULL);
    printf("Seeding with %u \n", seed);
    srand ( seed );
    */

    bool is_gaussian_sim = true;
    // Container for Cauchy Estimator
    CauchyDynamicsUpdateContainer duc;
    duc.n = n; duc.pncc = pncc; duc.p = p; duc.cmcc = cmcc;
    duc.Phi = Phi; duc.Gamma = Gamma; duc.H = H; 
    duc.B = NULL; duc.u = NULL;
    duc.beta = beta; duc.gamma = gamma;
    duc.step = 0; duc.dt = 0; duc.other_stuff = NULL; duc.x = NULL;
    // Container for Kalman Estimator
    KalmanDynamicsUpdateContainer kduc;
    kduc.n = n; kduc.pncc = pncc; kduc.p = p; kduc.cmcc = cmcc;
    kduc.Phi = Phi; kduc.Gamma = Gamma; kduc.H = H; 
    kduc.B = NULL; kduc.u = NULL;
    kduc.W = W; kduc.V = V;
    kduc.step = 0; kduc.dt = 0; kduc.other_stuff = NULL; kduc.x = NULL;

    int num_steps = 350;
    SimulationLogger* sim_log;
    if(is_gaussian_sim)
        sim_log = new SimulationLogger(log_dir, num_steps, x0_kf, &kduc, gaussian_lti_transition_model, gaussian_lti_measurement_model);
    else 
        sim_log = new SimulationLogger(log_dir, num_steps, b0, &duc, cauchy_lti_transition_model, cauchy_lti_measurement_model);
    sim_log->run_simulation_and_log();
    
    // New Sliding Window Manager
    int total_steps = num_steps+1; // including estimation for x0 (i.e, z0 -> first MU)
    int num_windows = 4;
    const bool WINDOW_PRINT_DEBUG = true;
    const bool WINDOW_LOG_SEQUENTIAL = false;
    const bool WINDOW_LOG_FULL = true;
    const bool is_extended = false;
    double* window_var_boost = NULL;
    SlidingWindowManager swm(num_windows, total_steps, A0, p0, b0, &duc, 
        WINDOW_PRINT_DEBUG, WINDOW_LOG_SEQUENTIAL, WINDOW_LOG_FULL, 
        is_extended, NULL, NULL, NULL, window_var_boost, log_dir);

    // Iterate over each step, window manager iterates over the individual measurements of each steps measurement vector
    for(int i = 0; i < total_steps; i++)
    {
        double* zs = sim_log->msmt_history + i*p;
        swm.step(zs, NULL);
    }
    swm.shutdown();

    // Add Kalman Filter and its logging here
    // Run the KF Simulation
    double kf_state_history[total_steps*n];
    double kf_covar_history[total_steps*n*n];
    double kf_Ks_history[total_steps*n*p];
    double kf_residual_history[num_steps*p];
    run_kalman_filter(num_steps, NULL, sim_log->msmt_history + p,
        kf_state_history, kf_covar_history,
        kf_Ks_history, kf_residual_history,
        x0_kf, P0,
        Phi, NULL, Gamma,
        H, W, V,
        n, cmcc, pncc, p,
        NULL,
        NULL,
        NULL);
    // Write the KF's states, variances, and residuals to the log folder
    log_kf_data(log_dir, kf_state_history, kf_covar_history, kf_residual_history, total_steps, n, p);


    delete sim_log;
} 

// Moshes Three State Problem
void test_2state_window_manager()
{
    // Cauchy Settings
    const int n = 2;
    const int pncc = 1;
    const int cmcc = 0;
    const int p = 1;
    double Phi[n*n] = {0.9, 0.1, -0.2, 1.1};
    double Gamma[n*pncc] = {1, 0.3};
    double H[p*n] = {1.0, 1.0};
    double beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double A0[n*n] = {1,0,0,1}; // Unit directions of the initial state uncertainty
    double p0[n] = {.10, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double b0[n] = {0,0}; // Initial median of system state
    //char log_dir[50] = "../log/window_manager";
    char* log_dir = NULL;
    // Kalman Settings
    double x0_kf[n];
    double W[pncc*pncc] = { pow(beta[0]*CAUCHY_TO_GAUSS_NOISE, 2) };
    double V[p*p] = { pow(gamma[0]*CAUCHY_TO_GAUSS_NOISE, 2) };
    double P0[n*n] = { pow(p0[0]*CAUCHY_TO_GAUSS_NOISE, 2), 0, 
                       0, pow(p0[1]*CAUCHY_TO_GAUSS_NOISE, 2)};
    memcpy(x0_kf, b0, n * sizeof(double));

    // Possibly Set a Seed
    ///* 
    //unsigned int seed = time(NULL);
    //printf("Seeding with %u \n", seed);
    srand ( 10 );
    //*/

    bool is_gaussian_sim = false;
    // Container for Cauchy Estimator
    CauchyDynamicsUpdateContainer duc;
    duc.n = n; duc.pncc = pncc; duc.p = p; duc.cmcc = cmcc;
    duc.Phi = Phi; duc.Gamma = Gamma; duc.H = H; duc.B = NULL; duc.u = NULL;
    duc.beta = beta; duc.gamma = gamma;
    duc.step = 0; duc.dt = 0; duc.other_stuff = NULL; duc.x = NULL;
    // Container for Kalman Estimator
    KalmanDynamicsUpdateContainer kduc;
    kduc.n = n; kduc.pncc = pncc; kduc.p = p; kduc.cmcc = cmcc;
    kduc.Phi = Phi; kduc.Gamma = Gamma; kduc.H = H; kduc.B = NULL; kduc.u = NULL;
    kduc.W = W; kduc.V = V;
    kduc.step = 0; kduc.dt = 0; kduc.other_stuff = NULL; kduc.x = NULL;

    int num_steps = 1050;
    SimulationLogger* sim_log;
    if(is_gaussian_sim)
        sim_log = new SimulationLogger(log_dir, num_steps, x0_kf, &kduc, gaussian_lti_transition_model, gaussian_lti_measurement_model);
    else 
        sim_log = new SimulationLogger(log_dir, num_steps, b0, &duc, cauchy_lti_transition_model, cauchy_lti_measurement_model);
    sim_log->run_simulation_and_log();
    
    
    // New Sliding Window Manager
    int total_steps = num_steps+1; // including estimation for x0 (i.e, z0 -> first MU)
    int num_windows = 4;
    const bool WINDOW_PRINT_DEBUG = false;
    const bool WINDOW_LOG_SEQUENTIAL = false;
    const bool WINDOW_LOG_FULL = false;
    const bool is_extended = false;
    double* window_var_boost = NULL;
    SlidingWindowManager swm(num_windows, total_steps, A0, p0, b0, &duc, 
        WINDOW_PRINT_DEBUG, WINDOW_LOG_SEQUENTIAL, WINDOW_LOG_FULL, 
        is_extended, NULL, NULL, NULL, window_var_boost, log_dir);

    // Iterate over each step, window manager iterates over the individual measurements of each steps measurement vector
    for(int i = 0; i < total_steps; i++)
    {
        double* zs = sim_log->msmt_history + i*p;
        swm.step(zs, NULL);
    }
    swm.shutdown();

    // Add Kalman Filter and its logging here
    // Run the KF Simulation
    double kf_state_history[total_steps*n];
    double kf_covar_history[total_steps*n*n];
    double kf_Ks_history[total_steps*n*p];
    double kf_residual_history[num_steps*p];
    run_kalman_filter(num_steps, NULL, sim_log->msmt_history + p,
        kf_state_history, kf_covar_history,
        kf_Ks_history, kf_residual_history,
        x0_kf, P0,
        Phi, NULL, Gamma,
        H, W, V,
        n, cmcc, pncc, p,
        NULL,
        NULL,
        NULL);
    // Write the KF's states, variances, and residuals to the log folder
    if(log_dir != NULL)
        log_kf_data(log_dir, kf_state_history, kf_covar_history, kf_residual_history, total_steps, n, p);
    delete sim_log;
}


int main()
{
    //test_2state_window_manager();
    test_3state_window_manager();
    //test_3state_3msmts_window_manager();
    return 0;
}