#include "../include/cauchy_windows.hpp"

// Moshes Three State Problem
void test_3state_window_manager()
{
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
    char log_dir[50] = "../log/window_manager/";
    // Possibly Set a Seed
    /* 
    unsigned int seed = time(NULL);
    printf("Seeding with %u \n", seed);
    srand ( seed );
    */

    CauchyDynamicsUpdateContainer duc;
    duc.n = n; duc.pncc = pncc; duc.p = p; duc.cmcc = cmcc;
    duc.Phi = Phi; duc.Gamma = Gamma; duc.H = H; duc.B = NULL;
    duc.beta = beta; duc.gamma = gamma;
    duc.step = 0; duc.dt = 0; duc.other_stuff = NULL; duc.x = NULL;

    double x0[n] = {0,0,0};
    int num_steps = 100;
    SimulationLogger sim_log(log_dir, num_steps, x0, &duc, cauchy_lti_transition_model, cauchy_lti_measurement_model);
    sim_log.run_simulation_and_log();
    
    // New Sliding Window Manager
    int total_steps = num_steps+1; // including estimation for x0 (first MU)
    int num_windows = 8;
    const bool WINDOW_PRINT_DEBUG = false;
    const bool WINDOW_LOG_SEQUENTIAL = false;
    const bool WINDOW_LOG_FULL = true;
    const bool is_extended = false;
    double* window_var_boost = NULL;
    SlidingWindowManager swm(num_windows, total_steps, A0, p0, b0, &duc, 
        WINDOW_PRINT_DEBUG, WINDOW_LOG_SEQUENTIAL, WINDOW_LOG_FULL, 
        is_extended, NULL, NULL, NULL, window_var_boost, log_dir);

    // Iterate over each step, and each measurement at each step
    for(int i = 0; i < total_steps; i++)
    {
        double* zs = sim_log.msmt_history + i*p;
        for(int j = 0; j < p; j++)
            swm.step(zs + j, NULL);
    }
    swm.shutdown();

    // Add Kalman Filter and its logging here

}


int main()
{
    test_3state_window_manager();
    return 0;
}