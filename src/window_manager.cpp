#include "../include/cauchy_windows.hpp"

// Moshes Three State Problem
void test_3state_window_manager()
{
    const int n = 3;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double Gamma[n*pncc] = {.1, 0.3, -0.2};
    double H[n] = {1.0, 0.5, 0.2};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; //{-0.63335359, -0.74816241, -0.19777826, -0.7710082 ,  0.63199184,  0.07831134, -0.06640465, -0.20208744,  0.97711365}; 
    double p0[n] = {0.10, 0.08, 0.05}; //{0.0, 0.0, 0.0}; //
    double b0[n] = {0, 0, 0};
    const int steps = 12;

    // Define some measurements
    double zs[steps] = {-1.2172011200334241, -0.35943271347277583, -0.52353301003957098, 0.5855389648301792, 
    -0.8048243525901404, 0.34053610027255954, 1.0580483915838776, -0.55152999529515989,
    -0.72879029737003309, -0.82415138330170357, -0.63794753995479381, -0.50437372151915394};

    // New Sliding Window Manager
    int num_windows = 3;
    CauchyDynamicsUpdateContainer duc;
    duc.n = n; duc.pncc = pncc; duc.p = p; duc.cmcc = 0;
    duc.Phi = Phi; duc.Gamma = Gamma; duc.H = H; duc.B = NULL;
    duc.beta = beta; duc.gamma = gamma;
    duc.step = 0; duc.dt = 0; duc.other_stuff = NULL; duc.x = NULL;
    const bool WINDOW_PRINT_DEBUG = true;
    const bool WINDOW_LOG_SEQUENTIAL = true;
    const bool WINDOW_LOG_FULL = true;
    const bool is_extended = false;
    double* window_var_boost = NULL;
    char log_dir[50] = "../log/window_manager";
    SlidingWindowManager swm(num_windows, steps, A0, p0, b0, &duc, 
        WINDOW_PRINT_DEBUG, WINDOW_LOG_SEQUENTIAL, WINDOW_LOG_FULL, 
        is_extended, NULL, NULL, NULL, window_var_boost, log_dir);

    for(int i = 0; i < steps; i++)
        swm.step(zs + i, NULL);
    swm.shutdown();
}


int main()
{
    test_3state_window_manager();
    return 0;
}