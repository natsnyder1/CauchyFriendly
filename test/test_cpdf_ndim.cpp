#include "../include/cpdf_ndim.hpp"
#include "../include/cpdf_2d.hpp"

void test_1d_cpdf()
{
    // Seed generator
    unsigned int seed = time(NULL);
    printf("Seeding with %u \n", seed);
    srand ( seed );

    // Scalar problem
    const int n = 1;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {0.9};
    double Gamma[n*pncc] = {0.4};
    double H[n] = {2.0};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] =  {1.0}; 
    double p0[n] = {0.10};
    double b0[n] = {0};
    CauchyDynamicsUpdateContainer duc;
    duc.n = n; duc.pncc = pncc; duc.p = p; duc.cmcc = cmcc;
    duc.Phi = Phi; duc.Gamma = Gamma; duc.H = H; 
    duc.B = NULL; duc.u = NULL; duc.x = NULL;
    duc.beta = beta; duc.gamma = gamma;
    duc.step = 0; duc.dt = 0; duc.other_stuff = NULL; 

    int sim_steps = 40;
    int total_steps = sim_steps + 1;
    char sim_log_dir[20] = "one_state";
    SimulationLogger sim_log(sim_log_dir, sim_steps, b0, &duc, cauchy_lti_transition_model, cauchy_lti_measurement_model);
    sim_log.run_simulation_and_log();

    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, total_steps, n, cmcc, pncc, p, print_basic_info);

    PointWiseNDimCauchyCPDF cpdf_1d(&cauchyEst);
    double grid_low = -4;
    double grid_high = 3;
    double grid_res = 0.005;
    char cpdf_log_dir[20] = "one_state/log_1d";
    CauchyCPDFGridDispatcher1D grid_1d(&cpdf_1d, grid_low, grid_high, grid_res, cpdf_log_dir);
    int num_threads = 1;
    bool with_print = false;
    double mean_estimates[total_steps];
    double cov_estimates[total_steps];
    for(int i = 0; i < total_steps-SKIP_LAST_STEP; i++)
    {  
        cauchyEst.step(sim_log.msmt_history[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        mean_estimates[i] = creal(cauchyEst.conditional_mean[0]);
        cov_estimates[i] = creal(cauchyEst.conditional_variance[0]);
        grid_1d.evaluate_point_grid(0, num_threads);
        grid_1d.log_point_grid();
        if(with_print)
        {
            for(int j = 0; j < grid_1d.num_grid_points; j++)
                printf("x=%.2lf, fx=%.9lf\n", grid_1d.points[j].x, grid_1d.points[j].y);
        }
    }
    for(int i = 0; i < total_steps-SKIP_LAST_STEP; i++)
    {
        printf("Step %d: True State: %.4lf, Estimate: %.4lf, Msmt %.4lf, Msmt Noise: %.4lf, Proc Noise: %.4lf\n", 
            i, sim_log.true_state_history[i], mean_estimates[i], 
            sim_log.msmt_history[i], sim_log.msmt_noise_history[i], i > 0 ? sim_log.proc_noise_history[i-1] : 0);
    }
    char path_mean_ests[30] = "one_state/cond_means.txt";
    char path_cov_ests[30] = "one_state/cond_covars.txt";
    log_double_array_to_file(path_mean_ests, mean_estimates, total_steps-SKIP_LAST_STEP, 1);
    log_double_array_to_file(path_cov_ests, cov_estimates, total_steps-SKIP_LAST_STEP, 1);
}

void test_2d_cpdf()
{
    const int n = 2;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {0.9, 0.1, -0.2, 1.1};
    double Gamma[n*pncc] = {.1, 0.3};
    double H[p*n] = {1.0, 0.5};
    double beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double A0[n*n] = {1,0,0,1}; // Unit directions of the initial state uncertainty
    double p0[n] = {.10, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double b0[n] = {0,0}; // Initial median of system state
    const int steps = 9;
    bool print_basic_info = true;
    char* log_dir = NULL;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWise2DCauchyCPDF cpdf_2d(log_dir, -1, 1, 0.25, -1, 1, 0.25);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    double zs[steps] = {0.022356919463887182, -0.22675889756491788, 0.42133397996398181, 
                        -1.7507202433585822, -1.3984154994099112, -1.7541436172809546, 
                        -1.8796017689052031, -1.9279807448991575, -1.9071129520752277}; 
    for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // Just to make sure that if last step is set to SKIP, we dont run...8 steps good enough 
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        cpdf_2d.evaluate_point_wise_cpdf(&cauchyEst);
        for(uint j = 0; j < cpdf_2d.num_gridx * cpdf_2d.num_gridy; j++)
        {    
            printf("\n2D: x=%.2lf, y=%.2lf, fx=%.9lf\n", cpdf_2d.cpdf_points[j].x, cpdf_2d.cpdf_points[j].y, cpdf_2d.cpdf_points[j].z);
            double xk[2] = {cpdf_2d.cpdf_points[j].x, cpdf_2d.cpdf_points[j].y};
            bool with_time_print = (i == (steps-SKIP_LAST_STEP-1) ) && (j==(cpdf_2d.num_gridx * cpdf_2d.num_gridy-1));
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_cpdf(xk, with_time_print);
            printf("ND: x=%.2lf, y=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", cpdf_2d.cpdf_points[j].x, cpdf_2d.cpdf_points[j].y, creal(fx), cimag(fx));            
        }
    }
}

void test_3d_cpdf()
{
    const int n = 3;
    const int cmcc = 0;
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
    const int steps = 8;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    double zs[steps] = {0.022172011200334241, -0.05943271347277583, -1.12353301003957098, -1.4055389648301792, 
    -0.9048243525901404, -1.34053610027255954, -2.0580483915838776, -0.55152999529515989};

    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    const int grid_points = 7;
    double grid3D[grid_points][3] = 
    {
        {0,0,0},
        {-0.05,0,0},
        {0.05,0,0},
        {0,-0.05,0},
        {0, 0.05,0},
        {0,0,-0.05},
        {0,0,0.05}
    };


    for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        for(uint j = 0; j < grid_points; j++)
        {    
            double* xk = grid3D[j];
            bool with_time_print = (i == (steps-2) ) && (j==(grid_points-1));
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_cpdf(xk, with_time_print);
            printf("ND: x=%.2lf, y=%.2lf, z=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk[0], xk[1], xk[2], creal(fx), cimag(fx));
        }
    }
}

void test_4d_cpdf()
{
    const int n = 4;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1; 
    double Phi[n*n] = {1.4, -0.6, -1.0, 0.0,  
    -0.2,  1.0,  0.5, 0.0,  
    0.6, -0.6, -0.2, 0.0, 
    0, 0, 0, 0.5};
    double Gamma[n*pncc] = {.1, 0.3, -0.2, 0.4};
    double H[n] = {2.0, 0.5,  0.2, -0.1};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] = {1, 0, 0, 0, 
    0, 1, 0, 0, 
    0, 0, 1, 0, 
    0, 0, 0, 1};
    double p0[n] = {0.1, 0.08, 0.05, 0.2};
    double b0[n] = {0, 0, 0, 0};
    const int steps = 6;
    bool print_basic_info = true;
    double zs[steps] = {-0.26300165310514712, -0.98289343232730964, -0.93317363235517392, -0.81311530427193779, 
            -0.24140673945883995, 0.013971096637110103}; //, -3.4842328985975715, -3.1607056967588112};
    const int grid_points = 9;
    double grid4D[grid_points][4] = 
        {
            {0,0,0,0},
            {-0.05,0,0,0},
            {0.05,0,0,0},
            {0,-0.05,0,0},
            {0, 0.05,0,0},
            {0,0,-0.05,0},
            {0,0,0.05,0},
            {0,0,0,0.05},
            {0,0,0,-0.05},
        };
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    // Runs estimator step by step
    for(int i = 0; i < steps-SKIP_LAST_STEP; i++)
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        for(uint j = 0; j < grid_points; j++)
        {    
            double* xk = grid4D[j];
            bool with_time_print = (i == (steps-2) ) && (j==(grid_points-1));
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_cpdf(xk, with_time_print);
            printf("ND: x1=%.2lf, x2=%.2lf, x3=%.2lf, x4=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk[0], xk[1], xk[2], xk[3], creal(fx), cimag(fx));
        }
    }

}

void test_2d_marginal_cpdf()
{
    const int n = 2;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {0.9, 0.1, -0.2, 1.1};
    double Gamma[n*pncc] = {.1, 0.3};
    double H[p*n] = {1.0, 0.5};
    double beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double A0[n*n] = {1,0,0,1}; // Unit directions of the initial state uncertainty
    double p0[n] = {.10, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double b0[n] = {0,0}; // Initial median of system state
    const int steps = 9;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    double zs[steps] = {0.022356919463887182, -0.22675889756491788, 0.42133397996398181, 
                        -1.7507202433585822, -1.3984154994099112, -1.7541436172809546, 
                        -1.8796017689052031, -1.9279807448991575, -1.9071129520752277}; 

    const int num_1d_grid_points = 21;
    double grid1D[num_1d_grid_points] = {
        -2,-1.8,-1.6,-1.4,-1.2,
        -1,-0.8,-0.6,-0.4,-0.2,
        0,
        0.2,0.4,0.6,0.8,1.0,
        1.2,1.4,1.6,1.8,2.0};
    const int num_marg_state_idxs = 1;
    int marg_state_idxs[num_marg_state_idxs] = {0};
    bool with_caching = true;
    for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // Just to make sure that if last step is set to SKIP, we dont run...8 steps good enough 
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        for(uint j = 0; j < num_1d_grid_points; j++)
        {
            double xk_marginal[1] = {grid1D[j]};
            bool with_time_print = (i==(steps-SKIP_LAST_STEP-1) && (j==(num_1d_grid_points-1)));
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print);
            printf("\nND: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx), cimag(fx));
            C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_1D_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print, with_caching);
            printf("1D: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx2), cimag(fx2));
        }
    }
}

void test_3d_marginal_cpdf()
{
    const int n = 3;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double Gamma[n*pncc] = {.1, 0.3, -0.2};
    double H[n] = {1.0, 0.5, 0.2};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] =  {
        1.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 
        0.0, 0.0, 1.0};
    double p0[n] = {0.10, 0.08, 0.05}; //{0.0, 0.0, 0.0}; //
    double b0[n] = {0, 0, 0};
    const int steps = 9;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    double zs[steps] = {0.022172011200334241, -0.05943271347277583, -1.12353301003957098, -1.4055389648301792, 
    -0.9048243525901404, -1.34053610027255954, -2.0580483915838776, -0.55152999529515989};

    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    bool test_2d = false;
    if(test_2d) 
    {
        const int num_2d_grid_points = 11;
        const int num_marg_state_idxs = 2;
        double grid2D[num_2d_grid_points][2] = 
            { 
                {-.4, -.3},
                {-.2,-.2},
                {0,-.2},
                {-.2,0},
                {0,0},
                {-.8,0},
                {-.8,-.2},
                {-.8,-.3},
                {-.8,-.4},
                {-1,-.2},
                {-1,-.4}
            };
        int marg_state_idxs[num_marg_state_idxs] = {0,1};
        bool with_caching = true;
        for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
        {
            cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
            for(uint j = 0; j < num_2d_grid_points; j++)
            {   
                double* xk_marginal = grid2D[j];
                bool with_time_print = (i == (steps-SKIP_LAST_STEP-1) ) && (j==(num_2d_grid_points-1));
                C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print);
                printf("\nND: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_state_idxs[0]+1, xk_marginal[0], marg_state_idxs[1]+1, xk_marginal[1], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_2D_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print, with_caching);
                printf("2D: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_state_idxs[0]+1, xk_marginal[0], marg_state_idxs[1]+1, xk_marginal[1], creal(fx2), cimag(fx2));
            }
        }
    }
    // 1D testing
    else
    {
        const int num_1d_grid_points = 11;
        const int num_marg_state_idxs = 1;
        int marg_state_idxs[num_marg_state_idxs] = {0};
        double grid1D[num_1d_grid_points] = {-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0};
        bool with_caching = true;
        for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
        {
            cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
            for(uint j = 0; j < num_1d_grid_points; j++)
            {    
                double xk_marginal[1] = {grid1D[j]};
                bool with_time_print = (i == (steps-SKIP_LAST_STEP-1) ) && (j==(num_1d_grid_points-1));
                C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print);
                printf("\nND: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_1D_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print, with_caching);
                printf("1D: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx2), cimag(fx2));
            }
        }
    }
}

void test_4d_marginal_cpdf()
{
    const int n = 4;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1; 
    double Phi[n*n] = {1.4, -0.6, -1.0, 0.0,  
    -0.2,  1.0,  0.5, 0.0,  
    0.6, -0.6, -0.2, 0.0, 
    0, 0, 0, 0.5};
    double Gamma[n*pncc] = {.1, 0.3, -0.2, 0.4};
    double H[n] = {2.0, 0.5,  0.2, -0.1};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] = {
        1, 0, 0, 0, 
        0, 1, 0, 0, 
        0, 0, 1, 0, 
        0, 0, 0, 1};
    double p0[n] = {0.1, 0.08, 0.05, 0.2};
    double b0[n] = {0, 0, 0, 0};
    const int steps = 8;
    bool print_basic_info = true;
    double zs[steps] = {-0.26300165310514712, -0.98289343232730964, -0.93317363235517392, -0.81311530427193779, 
            -0.24140673945883995, 0.013971096637110103, -0.4842328985975715, -0.7607056967588112};
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    bool test_2d = false;
    if(test_2d) 
    {
        const int num_2d_grid_points = 11;
        const int num_marg_state_idxs = 2;
        double grid2D[num_2d_grid_points][2] = 
            { 
                {-.5,-0.1},
                {-.5,0.0},
                {-.5,0.1},
                {-.4,-0.1},
                {-.4,0.0},
                {-.4,0.1},
                {-.3,-0.1},
                {-.3, 0},
                {-.3, 0.1},
                {-.2,0},
                {0,0}
            };
        bool with_caching=true;
        int marg_state_idxs[num_marg_state_idxs] = {0,1};
        for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
        {
            cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
            for(uint j = 0; j < num_2d_grid_points; j++)
            {   
                double* xk_marginal = grid2D[j];
                bool with_time_print = (i == (steps-SKIP_LAST_STEP-1) ) && (j==(num_2d_grid_points-1));
                C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print);
                printf("\nND: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_state_idxs[0]+1, xk_marginal[0], marg_state_idxs[1]+1, xk_marginal[1], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_2D_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print, with_caching);
                printf("2D: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_state_idxs[0]+1, xk_marginal[0], marg_state_idxs[1]+1, xk_marginal[1], creal(fx2), cimag(fx2));
            }
        }
    }
    // 1D testing
    else
    {
        const int num_1d_grid_points = 11;
        const int num_marg_state_idxs = 1;
        int marg_state_idxs[num_marg_state_idxs] = {0};
        double grid1D[num_1d_grid_points] = {-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0};
        bool with_caching = true;
        for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
        {
            cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
            for(uint j = 0; j < num_1d_grid_points; j++)
            {   
                double xk_marginal[1] = {grid1D[j]};
                bool with_time_print = (i == (steps-SKIP_LAST_STEP-1) ) && (j==(num_1d_grid_points-1));
                C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print);
                printf("\nND: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_1D_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print, with_caching);
                printf("1D: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx2), cimag(fx2));
            }
        }
    }
}

void test_3d_threaded_marginal_cpdf_eval()
{
    const int n = 3;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double Gamma[n*pncc] = {.1, 0.3, -0.2};
    double H[n] = {1.0, 0.5, 0.2};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] =  {
        1.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 
        0.0, 0.0, 1.0};
    double p0[n] = {0.10, 0.08, 0.05}; //{0.0, 0.0, 0.0}; //
    double b0[n] = {0, 0, 0};
    const int steps = 9;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    double zs[steps] = {
        0.022172011200334241, -0.05943271347277583, -1.12353301003957098, -1.4055389648301792, 
       -0.9048243525901404,   -1.34053610027255954, -2.0580483915838776,  -1.55152999529515989, -1.8782362786388};

    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    bool test_2d = false;
    if(test_2d) 
    {
        double x_low = -1;
        double x_high = 1;
        double x_res = 0.5;
        double y_low = -1;
        double y_high = 1;
        double y_res = 0.5;
        //char log_dir[12] = "log_marg2d";
        char* log_dir = NULL;
        CauchyCPDFGridDispatcher2D marg_cpdf(&cpdf_ndim, x_low, x_high, x_res, y_low, y_high, y_res, log_dir);
        int marg_idxs_group1[2] = {0,1};
        int marg_idxs_group2[2] = {0,2};
        int marg_idxs_group3[2] = {1,2};
        bool with_caching = true;
        bool with_timing = true;
        int num_threads = 8;

        // Test Marg Density Functionality
        for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
        {
            cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
            printf("------- Testing Marg Group 1: (%d,%d) -------\n", marg_idxs_group1[0], marg_idxs_group1[1]);
            marg_cpdf.evaluate_point_grid(marg_idxs_group1[0], marg_idxs_group1[1], num_threads, false);
            //cpdf_ndim.master_step_of_cached_2d_terms = -1;
            for(int j = 0; j < marg_cpdf.num_grid_points; j++)
            {   
                double xk_marginal[2] = {marg_cpdf.points[j].x, marg_cpdf.points[j].y};
                //C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_idxs_group1, 2, false);
                //printf("\nND: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_idxs_group1[0], xk_marginal[0], marg_idxs_group1[1], xk_marginal[1], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_2D_marginal_cpdf(xk_marginal, marg_idxs_group1, 2, false, with_caching);
                printf("2D: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_idxs_group1[0], xk_marginal[0], marg_idxs_group1[1], xk_marginal[1], creal(fx2), cimag(fx2));
                printf("2T: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n\n", 
                    marg_idxs_group1[0], marg_cpdf.points[j].x, 
                    marg_idxs_group1[1], marg_cpdf.points[j].y, 
                    marg_cpdf.points[j].z, 0.0);
            }
            printf("\n------- Testing Marg Group 2: (%d,%d) -------\n", marg_idxs_group2[0], marg_idxs_group2[1]);
            marg_cpdf.evaluate_point_grid(marg_idxs_group2[0], marg_idxs_group2[1], num_threads, false);
            //cpdf_ndim.master_step_of_cached_2d_terms = -1;
            for(int j = 0; j < marg_cpdf.num_grid_points; j++)
            {   
                double xk_marginal[2] = {marg_cpdf.points[j].x, marg_cpdf.points[j].y};
                //C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_idxs_group2, 2, false);
                //printf("\nND: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_idxs_group2[0], xk_marginal[0], marg_idxs_group2[1], xk_marginal[1], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_2D_marginal_cpdf(xk_marginal, marg_idxs_group2, 2, false, with_caching);
                printf("2D: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_idxs_group2[0], xk_marginal[0], marg_idxs_group2[1], xk_marginal[1], creal(fx2), cimag(fx2));
                printf("2T: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n\n", 
                    marg_idxs_group2[0], marg_cpdf.points[j].x, 
                    marg_idxs_group2[1], marg_cpdf.points[j].y, 
                    marg_cpdf.points[j].z, 0.0);
            }
            printf("\n------- Testing Marg Group 3: (%d,%d) -------\n", marg_idxs_group3[0], marg_idxs_group3[1]);
            marg_cpdf.evaluate_point_grid(marg_idxs_group3[0], marg_idxs_group3[1], num_threads, false);
            //cpdf_ndim.master_step_of_cached_2d_terms = -1;
            for(int j = 0; j < marg_cpdf.num_grid_points; j++)
            {   
                double xk_marginal[2] = {marg_cpdf.points[j].x, marg_cpdf.points[j].y};
                //C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_idxs_group3, 2, false);
                //printf("\nND: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_idxs_group3[0], xk_marginal[0], marg_idxs_group3[1], xk_marginal[1], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_2D_marginal_cpdf(xk_marginal, marg_idxs_group3, 2, false, with_caching);
                printf("2D: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", marg_idxs_group3[0], xk_marginal[0], marg_idxs_group3[1], xk_marginal[1], creal(fx2), cimag(fx2));
                printf("2T: x%d=%.2lf,  x%d=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n\n", 
                    marg_idxs_group3[0], marg_cpdf.points[j].x, 
                    marg_idxs_group3[1], marg_cpdf.points[j].y, 
                    marg_cpdf.points[j].z, 0.0);
            }
        }
        cauchyEst.reset();
        printf("------- Serial vs Distributed Time Test: (Using Marg Group 1) -------\n");
        // Time Test:
        x_res = 0.025;
        y_res = 0.025;
        marg_cpdf.reset_grid(x_low, x_high, x_res, y_low, y_high, y_res);
        cauchyEst.print_basic_info = false;
        for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
        {
            cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
            marg_cpdf.evaluate_point_grid(marg_idxs_group1[0], marg_idxs_group1[1], num_threads, with_timing);
            marg_cpdf.log_point_grid();
            cpdf_ndim.master_step_of_cached_2d_terms = -1;
            CPUTimer tmr;
            tmr.tic();
            for(int j = 0; j < marg_cpdf.num_grid_points; j++)
            {   
                double xk_marginal[2] = {marg_cpdf.points[j].x, marg_cpdf.points[j].y};
                cpdf_ndim.evaluate_2D_marginal_cpdf(xk_marginal, marg_idxs_group1, 2, false, with_caching);
            }            
            tmr.toc(false);
            printf("Serial Evaluation took %d ms for master step %d\n\n", tmr.cpu_time_used, cauchyEst.master_step);
        }
    }
    // 1D testing
    else
    {
        double grid_low = -1;
        double grid_high = 1;
        double grid_res = 0.15;
        char* log_dir = NULL;
        //char log_dir[12] = "log_marg1d";
        CauchyCPDFGridDispatcher1D marg_cpdf(&cpdf_ndim, grid_low, grid_high, grid_res, log_dir);
        int marg_group1[1] = {0};
        int marg_group2[1] = {1};
        int marg_group3[1] = {2};
        bool with_caching = true;
        bool with_timing = true;
        int num_threads = 8;
        CPUTimer tmr;

        // Marg Test For All States
        for(int i = 0; i < steps-SKIP_LAST_STEP-3; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
        {
            cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
            printf("\n------- Testing Marg Group 1: (%d) -------\n", marg_group1[0]);
            // Testing Marg of state 1
            marg_cpdf.evaluate_point_grid(marg_group1[0], num_threads, false);
            cpdf_ndim.master_step_of_cached_2d_terms = -1;
            for(int j = 0; j < marg_cpdf.num_grid_points; j++)
            {    
                double xk_marginal[1] = {marg_cpdf.points[j].x};
                C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_group1, 1, false);
                printf("\nND: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_1D_marginal_cpdf(xk_marginal, marg_group1, 1, false, with_caching);
                printf("1D: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx2), cimag(fx2));
                printf("1T: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n\n", xk_marginal[0], marg_cpdf.points[j].y, 0.0);
            }
            printf("\n------- Testing Marg Group 1: (%d) -------\n", marg_group2[0]);
            // Testing Marg of state 2
            marg_cpdf.evaluate_point_grid(marg_group2[0], num_threads, false);
            cpdf_ndim.master_step_of_cached_2d_terms = -1;
            for(int j = 0; j < marg_cpdf.num_grid_points; j++)
            {    
                double xk_marginal[1] = {marg_cpdf.points[j].x};
                C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_group2, 1, false);
                printf("\nND: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_1D_marginal_cpdf(xk_marginal, marg_group2, 1, false, with_caching);
                printf("1D: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx2), cimag(fx2));
                printf("1T: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n\n", xk_marginal[0], marg_cpdf.points[j].y, 0.0);
            }
            printf("\n------- Testing Marg Group 3: (%d) -------\n", marg_group3[0]);
            // Testing Marg of state 3
            marg_cpdf.evaluate_point_grid(marg_group3[0], num_threads, false);
            cpdf_ndim.master_step_of_cached_2d_terms = -1;
            for(int j = 0; j < marg_cpdf.num_grid_points; j++)
            {    
                double xk_marginal[1] = {marg_cpdf.points[j].x};
                C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_group3, 1, false);
                printf("\nND: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx), cimag(fx));
                C_COMPLEX_TYPE fx2 = cpdf_ndim.evaluate_1D_marginal_cpdf(xk_marginal, marg_group3, 1, false, with_caching);
                printf("1D: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx2), cimag(fx2));
                printf("1T: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n\n", xk_marginal[0], marg_cpdf.points[j].y, 0.0);
            }
        }
        printf("------ Testing Times of 1D Marginalization ------\n");
        grid_res = 0.01;
        marg_cpdf.reset_grid(grid_low, grid_high, grid_res);
        cauchyEst.reset();
        cauchyEst.print_basic_info = false;
        // Timing Test
        for(int i = 0; i < steps-SKIP_LAST_STEP-3; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
        {
            cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
            // Testing Marg of state 1
            marg_cpdf.evaluate_point_grid(marg_group1[0], num_threads, with_timing);
            marg_cpdf.log_point_grid();
            cpdf_ndim.master_step_of_cached_2d_terms = -1;
            tmr.tic();
            for(int j = 0; j < marg_cpdf.num_grid_points; j++)
            {    
                double xk_marginal[1] = {marg_cpdf.points[j].x};
                cpdf_ndim.evaluate_1D_marginal_cpdf(xk_marginal, marg_group1, 1, false, with_caching);
            }
            tmr.toc(false);
            printf("Serial Evaluation took %d ms for master step %d\n\n", tmr.cpu_time_used, cauchyEst.master_step);
        }
    }
}

void test_2d_cpdf_and_marginals()
{
    const int n = 2;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {0.9, 0.1, -0.2, 1.1};
    double Gamma[n*pncc] = {.1, 0.3};
    double H[p*n] = {1.0, 0.5};
    double beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double A0[n*n] = {1,0,0,1}; // Unit directions of the initial state uncertainty
    double p0[n] = {.10, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double b0[n] = {0,0}; // Initial median of system state
    const int steps = 9;
    bool print_basic_info = false;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    double zs[steps] = {0.022356919463887182, -0.22675889756491788, 0.42133397996398181, 
                        -1.7507202433585822, -1.3984154994099112, -1.7541436172809546, 
                        -1.8796017689052031, -1.9279807448991575, -1.9071129520752277}; 
    // CPDF Structures
    double grid2d_low_x = -1.5;
    double grid2d_high_x = 1;
    double grid2d_res_x = 0.025;
    double grid2d_low_y = -3.5;
    double grid2d_high_y = 1;
    double grid2d_res_y = 0.025;
    // For compare
    //PointWise2DCauchyCPDF cpdf_2d(NULL, 
    //    grid2d_low_x, grid2d_high_x, grid2d_res_x,
    //    grid2d_low_y, grid2d_high_y, grid2d_res_y);
    // Main driver
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    // New 2D Grid wrapper
    char log_2d[20] = "two_state/log_2d";
    CauchyCPDFGridDispatcher2D cpdf_grid(&cpdf_ndim, 
        grid2d_low_x, grid2d_high_x, grid2d_res_x,
        grid2d_low_y, grid2d_high_y, grid2d_res_y, 
        log_2d);
    // Marg 1D Grid Wrapper
    double grid1d_low = -3;
    double grid1d_high = 3;
    double grid1d_res = 0.001;
    char log_1d[20] = "two_state/log_1d";
    CauchyCPDFGridDispatcher1D marg1_grid(&cpdf_ndim, 
        grid1d_low, grid1d_high, grid1d_res, 
        log_1d);

    int num_threads = 8;
    bool with_value_print = false;
    CPUTimer tmr;
    for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // Just to make sure that if last step is set to SKIP, we dont run...8 steps good enough 
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        // Old 2D Eval
        //tmr.tic();
        //cpdf_2d.evaluate_point_wise_cpdf(&cauchyEst, num_threads);
        //tmr.toc(false);
        //printf("Old 2D Step %d: Took %d ms for %d points (%d terms/point)\n", i+1, tmr.cpu_time_used, cpdf_grid.num_grid_points, cauchyEst.Nt);
        // New 2D Eval
        tmr.tic();
        cpdf_grid.evaluate_point_grid(0, 1, num_threads);
        tmr.toc(false);
        printf("New 2D Step %d: Took %d ms for %d points (%d terms/point)\n", i+1, tmr.cpu_time_used, cpdf_grid.num_grid_points, cauchyEst.Nt);
        cpdf_grid.log_point_grid();
        // 1D Marg Eval
        tmr.tic();
        marg1_grid.evaluate_point_grid(0, num_threads);
        tmr.toc(false);
        printf("MargS0 Step %d: Took %d ms for %d points (%d terms/point)\n", i+1, tmr.cpu_time_used, marg1_grid.num_grid_points, cauchyEst.Nt);
        marg1_grid.log_point_grid();
        tmr.tic();
        marg1_grid.evaluate_point_grid(1, num_threads);
        tmr.toc(false);
        printf("MargS1 Step %d: Took %d ms for %d points (%d terms/point)\n\n", i+1, tmr.cpu_time_used, marg1_grid.num_grid_points, cauchyEst.Nt);
        marg1_grid.log_point_grid();

        if(with_value_print)
        {
            printf(" ------ Step %d ------\n", i+1);
            for(int j = 0; j < cpdf_grid.num_grid_points; j++)
            {    
                //C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_cpdf(xk, with_time_print);
                //printf("ND: x=%.2lf, y=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", cpdf_2d.cpdf_points[j].x, cpdf_2d.cpdf_points[j].y, creal(fx), cimag(fx));
                //printf("\n2D: x=%.2lf, y=%.2lf, fx=%.9lf\n", cpdf_2d.cpdf_points[j].x, cpdf_2d.cpdf_points[j].y, cpdf_2d.cpdf_points[j].z);
                printf("2+: x=%.2lf, y=%.2lf, fx=%.9lf\n", cpdf_grid.points[j].x, cpdf_grid.points[j].y, cpdf_grid.points[j].z);
            }
            for(int j = 0; j < marg1_grid.num_grid_points; j++)
            {
                printf("S1: x=%.2lf, fx=%.9lf\n", marg1_grid.points[j].x, marg1_grid.points[j].y);
            }
        }
    }
}

void test_3d_cpdf_and_marginals()
{
    const int n = 3;
    const int cmcc = 0;
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
    const int steps = 7;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    double zs[steps] = {0.022172011200334241, -0.11943271347277583, -1.22353301003957098, 
                       -1.4055389648301792, -1.34053610027255954, 0.4580483915838776, 
                        0.65152999529515989}; //, 0.52378648722334, 0.75198272983};

    // CPDF Grid Sizes
    double grid2d_low_x = -2;
    double grid2d_high_x = 2;
    double grid2d_res_x = 0.025;
    double grid2d_low_y = -2;
    double grid2d_high_y = 2;
    double grid2d_res_y = 0.025;
    
    // Main driver
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    // New 2D Grid wrapper
    char log_2d[20] = "three_state/log_2d";
    CauchyCPDFGridDispatcher2D marg2_grid(&cpdf_ndim, 
        grid2d_low_x, grid2d_high_x, grid2d_res_x,
        grid2d_low_y, grid2d_high_y, grid2d_res_y, 
        log_2d);
    // Marg 1D Grid Wrapper
    double grid1d_low = -4;
    double grid1d_high = 4;
    double grid1d_res = 0.001;
    char log_1d[20] = "three_state/log_1d";
    CauchyCPDFGridDispatcher1D marg1_grid(&cpdf_ndim, 
        grid1d_low, grid1d_high, grid1d_res, 
        log_1d);

    int num_threads = 8;
    CPUTimer tmr;
    for(int i = 0; i < steps-SKIP_LAST_STEP; i++) // Just to make sure that if last step is set to SKIP, we dont run...8 steps good enough 
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        
        // Marg 2D Eval
        tmr.tic();
        marg2_grid.evaluate_point_grid(0, 1, num_threads);
        tmr.toc(false);
        printf("New 2D Step %d (0,1): Took %d ms for %d points (%d terms/point)\n", i+1, tmr.cpu_time_used, marg2_grid.num_grid_points, cauchyEst.Nt);
        marg2_grid.log_point_grid();
        // Marg 2D Eval
        tmr.tic();
        marg2_grid.evaluate_point_grid(0, 2, num_threads);
        tmr.toc(false);
        printf("New 2D Step %d (0,2): Took %d ms for %d points (%d terms/point)\n", i+1, tmr.cpu_time_used, marg2_grid.num_grid_points, cauchyEst.Nt);
        marg2_grid.log_point_grid();
        // Marg 2D Eval
        tmr.tic();
        marg2_grid.evaluate_point_grid(1, 2, num_threads);
        tmr.toc(false);
        printf("New 2D Step %d (0,2): Took %d ms for %d points (%d terms/point)\n", i+1, tmr.cpu_time_used, marg2_grid.num_grid_points, cauchyEst.Nt);
        marg2_grid.log_point_grid();

        // 1D Marg Eval
        tmr.tic();
        marg1_grid.evaluate_point_grid(0, num_threads);
        tmr.toc(false);
        printf("MargS0 Step %d: Took %d ms for %d points (%d terms/point)\n", i+1, tmr.cpu_time_used, marg1_grid.num_grid_points, cauchyEst.Nt);
        marg1_grid.log_point_grid();
        // 1D Marg Eval
        tmr.tic();
        marg1_grid.evaluate_point_grid(1, num_threads);
        tmr.toc(false);
        printf("MargS1 Step %d: Took %d ms for %d points (%d terms/point)\n", i+1, tmr.cpu_time_used, marg1_grid.num_grid_points, cauchyEst.Nt);
        marg1_grid.log_point_grid();
        // 1D Marg Eval
        tmr.tic();
        marg1_grid.evaluate_point_grid(2, num_threads);
        tmr.toc(false);
        printf("MargS2 Step %d: Took %d ms for %d points (%d terms/point)\n\n", i+1, tmr.cpu_time_used, marg1_grid.num_grid_points, cauchyEst.Nt);
        marg1_grid.log_point_grid();
    }
}

int main()
{
    // Basic Functionality Testing
    //test_1d_cpdf();
    //test_2d_cpdf(); // Is verified to be working (has been compared to the explicit 2D CPDF code)
    //test_3d_cpdf(); // Is (possibly) verified to be working (real and positive values returned...no errors)
    //test_4d_cpdf(); // Is (possibly) verified to be working (real and positive values returned...no errors)
    //test_2d_marginal_cpdf(); // Is (possibly) verified to be working (real and positive values returned...no errors)
    //test_3d_marginal_cpdf(); // Is (possibly) verified to be working (real and positive values returned...no errors)
    //test_4d_marginal_cpdf(); // Is (possibly) verified to be working (real and positive values returned...no errors)

    // Threaded Functionality Testing
    //test_3d_threaded_marginal_cpdf_eval();
    test_2d_cpdf_and_marginals();
    //test_3d_cpdf_and_marginals();
    return 0;
}