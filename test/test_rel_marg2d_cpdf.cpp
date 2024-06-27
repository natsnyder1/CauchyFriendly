#include "../include/cauchy_prediction.hpp"


void test_reltrans_2d()
{
    //time_t seed = time(NULL);
    //srand ( seed );
    //printf("Seeding with %u\n", (uint)seed);
    // CPDF Grid 
    double xlow = -2.00;
    double xhigh = 2.00;
    double xdelta = 0.025;
    double ylow = -2.00;
    double yhigh = 2.00;
    double ydelta = 0.025;
    char rel_sys_log_dir[100] = "./rel_cpdf/2d/temp_realiz/";
    char sys1_log_dir[100]; 
    char sys2_log_dir[100];
    sprintf(sys1_log_dir, "%s/sys1/", rel_sys_log_dir);
    sprintf(sys2_log_dir, "%s/sys2/", rel_sys_log_dir);
    
    // Two 2D systems, offset from one another
    const int n = 2;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double sys1_Phi[n*n] = {0.9, 0.1, -0.2, 1.1};
    double sys1_Gamma[n*pncc] = {1, 0.3};
    double* sys1_B = NULL;
    double sys1_H[p*n] = {1.0, 1.0};
    double sys1_beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double sys1_gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double sys1_A0[n*n] = {1,0,0,1}; // Unit directions of the initial state uncertainty
    double sys1_p0[n] = {0.10, 0.08}; // Initial state uncertainty cauchy scaling parameter(s)
    double sys1_b0[n] = {0.18,0.18}; // Initial median of system state
    double* sys1_u = NULL;
    
    double sys2_Phi[n*n] = {0.9, 0.1, -0.2, 1.1};
    double sys2_Gamma[n*pncc] = {1, 0.3};
    double* sys2_B = NULL;
    double sys2_H[p*n] = {1.0, 1.0};
    double sys2_beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double sys2_gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double sys2_A0[n*n] = {1,0,0,1}; // Unit directions of the initial state uncertainty
    double sys2_p0[n] = {0.10, 0.08}; // Initial state uncertainty cauchy scaling parameter(s)    
    double sys2_b0[n] = {-0.18, -0.18}; // Initial median of system state
    double* sys2_u = NULL;

    // MCE Steps and Debud Print Settings
    const int steps = 5;
    bool print_basic_info = false;

    // Dynamic System 1
    CauchyEstimator sys1_mce(sys1_A0, sys1_p0, sys1_b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWiseNDimCauchyCPDF sys1_cpdf(&sys1_mce);
    CauchyCPDFGridDispatcher2D sys1_cpdf_grid_dispatcher(&sys1_cpdf, xlow, xhigh, xdelta, ylow, yhigh, ydelta, sys1_log_dir);
    CauchyDynamicsUpdateContainer sys1_cduc;
    sys1_cduc.Phi = sys1_Phi; sys1_cduc.Gamma = sys1_Gamma; sys1_cduc.H = sys1_H; sys1_cduc.beta = sys1_beta; 
    sys1_cduc.gamma = sys1_gamma; sys1_cduc.step = 0; 
    sys1_cduc.B = sys1_B; sys1_cduc.u = sys1_u; sys1_cduc.n = n; sys1_cduc.pncc = pncc; sys1_cduc.cmcc = cmcc; sys1_cduc.p = p;
    SimulationLogger sys1_sim_log(sys1_log_dir, steps, sys1_b0, &sys1_cduc, cauchy_lti_transition_model, cauchy_lti_measurement_model);
    sys1_sim_log.run_simulation_and_log();
    double sys1_cond_means[steps*n];
    double sys1_cond_vars[steps*n*n];

    // Dynamic System 2
    CauchyEstimator sys2_mce(sys2_A0, sys2_p0, sys2_b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWiseNDimCauchyCPDF sys2_cpdf(&sys2_mce);
    CauchyCPDFGridDispatcher2D sys2_cpdf_grid_dispatcher(&sys2_cpdf, xlow, xhigh, xdelta, ylow, yhigh, ydelta, sys2_log_dir);
    CauchyDynamicsUpdateContainer sys2_cduc;
    sys2_cduc.Phi = sys1_Phi; sys2_cduc.Gamma = sys1_Gamma; sys2_cduc.H = sys1_H; sys2_cduc.beta = sys1_beta; 
    sys2_cduc.gamma = sys1_gamma; sys2_cduc.step = 0; 
    sys2_cduc.B = sys1_B; sys2_cduc.u = sys1_u; sys2_cduc.n = n; sys2_cduc.pncc = pncc; sys2_cduc.cmcc = cmcc; sys2_cduc.p = p;
    SimulationLogger sys2_sim_log(sys2_log_dir, steps, sys2_b0, &sys2_cduc, cauchy_lti_transition_model, cauchy_lti_measurement_model);
    sys2_sim_log.run_simulation_and_log();
    double sys2_cond_means[steps*n];
    double sys2_cond_vars[steps*n*n];

    // Relative System 
    double rsys_cond_means[steps*n];
    double rsys_cond_vars[steps*n*n];
    
    // Foolin Around on Step 1
    sys1_sim_log.msmt_history[0] += 0.35;
    sys2_sim_log.msmt_history[0] -= 1.5;

    // The relative transformation between the systems is, initially, identity
    double Trel[2*n] = {1,0,0,1};
    int cpdf_threads = 8;
    bool cpdf_timing = true;
    int rel_cpdf_threads = 8;
    bool rel_cpdf_timing = true;

    // Run Both Systems 
    // Loop over each step
    for(int i = 0; i < steps-1; i++)
    {
        // Loop over each measurement, per estimation step
        for(int j = 0; j < p; j++)
        {
            double sys1_z = sys1_sim_log.msmt_history[i*p + j];
            double sys2_z = sys2_sim_log.msmt_history[i*p + j];
            sys1_mce.step(sys1_z, sys1_Phi, sys1_Gamma, sys1_beta, sys1_H + (j%p)*n, sys1_gamma[j%p], sys1_B, sys1_u);
            sys2_mce.step(sys2_z, sys2_Phi, sys2_Gamma, sys2_beta, sys2_H + (j%p)*n, sys2_gamma[j%p], sys2_B, sys2_u);
        }
        // Store conditional mean and variance
        convert_complex_array_to_real(sys1_mce.conditional_mean, sys1_cond_means + i*n, n);
        convert_complex_array_to_real(sys1_mce.conditional_variance, sys1_cond_vars + i*n*n, n*n);
        convert_complex_array_to_real(sys2_mce.conditional_mean, sys2_cond_means + i*n, n);
        convert_complex_array_to_real(sys2_mce.conditional_variance, sys2_cond_vars + i*n*n, n*n);
        
        // Evaluate the CPDF over the grid of points for each system 
        int marg_idx0 = 0; int marg_idx1 = 1;  
        sys1_cpdf_grid_dispatcher.reset_grid(
            xlow + sys1_cond_means[i*n], xhigh + sys1_cond_means[i*n], xdelta,
            ylow + sys1_cond_means[i*n+1], yhigh + sys1_cond_means[i*n+1], ydelta);
        sys1_cpdf_grid_dispatcher.evaluate_point_grid(marg_idx0, marg_idx1, cpdf_threads, cpdf_timing);
        sys1_cpdf_grid_dispatcher.log_point_grid();
        sys2_cpdf_grid_dispatcher.reset_grid(
            xlow + sys2_cond_means[i*n], xhigh + sys2_cond_means[i*n], xdelta,
            ylow + sys2_cond_means[i*n+1], yhigh + sys2_cond_means[i*n+1], ydelta);
        sys2_cpdf_grid_dispatcher.evaluate_point_grid(marg_idx0, marg_idx1, cpdf_threads, cpdf_timing);
        sys2_cpdf_grid_dispatcher.log_point_grid();
        // Form the relative system and repeat 
        const bool FULL_RSYS_SOLVE = false; // false // CAN CHANGE
        const bool GET_RSYS_MOMENTS = true; // DONT CHANGE
        C_COMPLEX_TYPE rsys_norm_factor[1];
        C_COMPLEX_TYPE rsys_cond_mean[2];
        C_COMPLEX_TYPE rsys_cond_covar[4];
        Cached2DCPDFTermContainer* rel_trans_2d_cached_terms = get_marg2d_relative_and_transformed_cpdf(
            &sys1_mce, &sys2_mce, Trel, FULL_RSYS_SOLVE, rel_cpdf_timing,
            GET_RSYS_MOMENTS, rsys_norm_factor, rsys_cond_mean, rsys_cond_covar);
        // Store conditional mean and variance
        convert_complex_array_to_real(rsys_cond_mean, rsys_cond_means + i*2, 2);
        convert_complex_array_to_real(rsys_cond_covar, rsys_cond_vars + i*4, 4);
        // Evaluate relative system grid
        int num_points_x, num_points_y;
        CauchyPoint3D* points = grid_eval_marg2d_relative_and_transformed_cpdf(rel_trans_2d_cached_terms, 
            xlow + rsys_cond_means[i*2], xhigh + rsys_cond_means[i*2], xdelta, 
            ylow + rsys_cond_means[i*2+1], yhigh + rsys_cond_means[i*2+1], ydelta,
            creal(sys1_mce.fz), creal(sys2_mce.fz),
            &num_points_x, &num_points_y, rel_cpdf_threads, FULL_RSYS_SOLVE, rel_cpdf_timing);
        log_marg2d_relative_and_transformed_cpdf(rel_sys_log_dir, i+1, points, num_points_x, num_points_y);
        rel_trans_2d_cached_terms->deinit();
        free(rel_trans_2d_cached_terms);
        free(points);
    }
    char sys1_fpath_cond_mean[200];
    sprintf(sys1_fpath_cond_mean, "%scond_means.txt", sys1_log_dir);
    char sys1_fpath_cond_var[200];
    sprintf(sys1_fpath_cond_var, "%scond_covars.txt", sys1_log_dir);
    char sys2_fpath_cond_mean[200];
    sprintf(sys2_fpath_cond_mean, "%scond_means.txt", sys2_log_dir);
    char sys2_fpath_cond_var[200];
    sprintf(sys2_fpath_cond_var, "%scond_covars.txt", sys2_log_dir);
    char rsys_fpath_cond_mean[200];
    sprintf(rsys_fpath_cond_mean, "%scond_means.txt", rel_sys_log_dir);
    char rsys_fpath_cond_var[200];
    sprintf(rsys_fpath_cond_var, "%scond_covars.txt", rel_sys_log_dir);
    char rsys_fpath_reltrans[200];
    sprintf(rsys_fpath_reltrans, "%sreltrans.txt", rel_sys_log_dir);
    log_double_array_to_file(sys1_fpath_cond_mean, sys1_cond_means, steps-1, n);
    log_double_array_to_file(sys1_fpath_cond_var, sys1_cond_vars, steps-1, n*n);
    log_double_array_to_file(sys2_fpath_cond_mean, sys2_cond_means, steps-1, n);
    log_double_array_to_file(sys2_fpath_cond_var, sys2_cond_vars, steps-1, n*n);
    log_double_array_to_file(rsys_fpath_cond_mean, rsys_cond_means, steps-1, 2);
    log_double_array_to_file(rsys_fpath_cond_var, rsys_cond_vars, steps-1, 4);
    log_double_array_to_file(rsys_fpath_reltrans, Trel, 2, n);
}

void test_reltrans_3d()
{
    time_t seed = 1717026144;//time(NULL);
    srand ( seed );
    printf("Seeding with %u\n", (uint)seed);
    // CPDF Grid 
    double xlow = -2.00;
    double xhigh = 2.00;
    double xdelta = 0.025;
    double ylow = -2.00;
    double yhigh = 2.00;
    double ydelta = 0.025;
    double scale_fac = 1.0; // makes grid smaller / larger for systems
    double rscale_fac = 1.0; // makes grid smaller / larger for rsys
    char rel_sys_log_dir[100] = "./rel_cpdf/3d/rel_trans_unit/";

    // Two 2D systems, offset from one another
    const int n = 3;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;

    // The relative transformation between the systems
    double Trel[2*n] = {1,0,0,0,1,0};//{0.5, 0.25, 0.25,
                        //0.25, 0.5, 0.25};

    char sys1_log_dir[100]; 
    char sys2_log_dir[100];
    sprintf(sys1_log_dir, "%s/sys1/", rel_sys_log_dir);
    sprintf(sys2_log_dir, "%s/sys2/", rel_sys_log_dir);

    double sys1_Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double sys1_Gamma[n*pncc] = {.1, 0.3, -0.2};
    double* sys1_B = NULL;
    double sys1_H[n] = {1.0, 0.5, 0.2};
    double sys1_beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double sys1_gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double sys1_A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; // Unit directions of the initial state uncertainty
    double sys1_p0[n] = {0.10, 0.08, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double sys1_b0[n] = {0.18,0.18, 0.18}; // Initial median of system state
    double* sys1_u = NULL;
    
    double sys2_Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double sys2_Gamma[n*pncc] = {.1, 0.3, -0.2};
    double* sys2_B = NULL;
    double sys2_H[n] = {1.0, 0.5, 0.2};
    double sys2_beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double sys2_gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double sys2_A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; // Unit directions of the initial state uncertainty
    double sys2_p0[n] = {0.10, 0.08, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double sys2_b0[n] = {-0.18,-0.18, -0.18}; // Initial median of system state
    double* sys2_u = NULL;

    // MCE Steps and Debud Print Settings
    const int steps = 5;
    bool print_basic_info = false;

    // Dynamic System 1
    CauchyEstimator sys1_mce(sys1_A0, sys1_p0, sys1_b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWiseNDimCauchyCPDF sys1_cpdf(&sys1_mce);
    CauchyCPDFGridDispatcher2D sys1_cpdf_grid_dispatcher(&sys1_cpdf, xlow, xhigh, xdelta, ylow, yhigh, ydelta, sys1_log_dir);
    CauchyDynamicsUpdateContainer sys1_cduc;
    sys1_cduc.Phi = sys1_Phi; sys1_cduc.Gamma = sys1_Gamma; sys1_cduc.H = sys1_H; sys1_cduc.beta = sys1_beta; 
    sys1_cduc.gamma = sys1_gamma; sys1_cduc.step = 0; 
    sys1_cduc.B = sys1_B; sys1_cduc.u = sys1_u; sys1_cduc.n = n; sys1_cduc.pncc = pncc; sys1_cduc.cmcc = cmcc; sys1_cduc.p = p;
    SimulationLogger sys1_sim_log(sys1_log_dir, steps, sys1_b0, &sys1_cduc, cauchy_lti_transition_model, cauchy_lti_measurement_model);
    sys1_sim_log.run_simulation_and_log();
    double sys1_cond_means[steps*n];
    double sys1_cond_vars[steps*n*n];

    // Dynamic System 2
    CauchyEstimator sys2_mce(sys2_A0, sys2_p0, sys2_b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWiseNDimCauchyCPDF sys2_cpdf(&sys2_mce);
    CauchyCPDFGridDispatcher2D sys2_cpdf_grid_dispatcher(&sys2_cpdf, xlow, xhigh, xdelta, ylow, yhigh, ydelta, sys2_log_dir);
    CauchyDynamicsUpdateContainer sys2_cduc;
    sys2_cduc.Phi = sys1_Phi; sys2_cduc.Gamma = sys1_Gamma; sys2_cduc.H = sys1_H; sys2_cduc.beta = sys1_beta; 
    sys2_cduc.gamma = sys1_gamma; sys2_cduc.step = 0; 
    sys2_cduc.B = sys1_B; sys2_cduc.u = sys1_u; sys2_cduc.n = n; sys2_cduc.pncc = pncc; sys2_cduc.cmcc = cmcc; sys2_cduc.p = p;
    SimulationLogger sys2_sim_log(sys2_log_dir, steps, sys2_b0, &sys2_cduc, cauchy_lti_transition_model, cauchy_lti_measurement_model);
    sys2_sim_log.run_simulation_and_log();
    double sys2_cond_means[steps*n];
    double sys2_cond_vars[steps*n*n];

    // Relative System 
    double rsys_cond_means[steps*n];
    double rsys_cond_vars[steps*n*n];
    
    // Foolin Around on Step 1
    sys1_sim_log.msmt_history[0] += 1.00;
    sys2_sim_log.msmt_history[0] += 1.65;

    int cpdf_threads = 8;
    bool cpdf_timing = true;
    int rel_cpdf_threads = 8;
    bool rel_cpdf_timing = true;

    // Run Both Systems 
    // Loop over each step
    for(int i = 0; i < steps-1; i++)
    {
        // Loop over each measurement, per estimation step
        for(int j = 0; j < p; j++)
        {
            double sys1_z = sys1_sim_log.msmt_history[i*p + j];
            double sys2_z = sys2_sim_log.msmt_history[i*p + j];
            sys1_mce.step(sys1_z, sys1_Phi, sys1_Gamma, sys1_beta, sys1_H + (j%p)*n, sys1_gamma[j%p], sys1_B, sys1_u);
            sys2_mce.step(sys2_z, sys2_Phi, sys2_Gamma, sys2_beta, sys2_H + (j%p)*n, sys2_gamma[j%p], sys2_B, sys2_u);
        }
        // Store conditional mean and variance
        convert_complex_array_to_real(sys1_mce.conditional_mean, sys1_cond_means + i*n, n);
        convert_complex_array_to_real(sys1_mce.conditional_variance, sys1_cond_vars + i*n*n, n*n);
        convert_complex_array_to_real(sys2_mce.conditional_mean, sys2_cond_means + i*n, n);
        convert_complex_array_to_real(sys2_mce.conditional_variance, sys2_cond_vars + i*n*n, n*n);
        
        // Evaluate the CPDF over the grid of points for each system 
        int marg_idx0 = 0; int marg_idx1 = 1;  
        sys1_cpdf_grid_dispatcher.reset_grid(
            scale_fac*xlow + sys1_cond_means[i*n], scale_fac*xhigh + sys1_cond_means[i*n], scale_fac*xdelta,
            scale_fac*ylow + sys1_cond_means[i*n+1], scale_fac*yhigh + sys1_cond_means[i*n+1], scale_fac*ydelta);
        sys1_cpdf_grid_dispatcher.evaluate_point_grid(marg_idx0, marg_idx1, cpdf_threads, cpdf_timing);
        sys1_cpdf_grid_dispatcher.log_point_grid();
        sys2_cpdf_grid_dispatcher.reset_grid(
            scale_fac*xlow + sys2_cond_means[i*n], scale_fac*xhigh + sys2_cond_means[i*n], scale_fac*xdelta,
            scale_fac*ylow + sys2_cond_means[i*n+1], scale_fac*yhigh + sys2_cond_means[i*n+1], scale_fac*ydelta);
        sys2_cpdf_grid_dispatcher.evaluate_point_grid(marg_idx0, marg_idx1, cpdf_threads, cpdf_timing);
        sys2_cpdf_grid_dispatcher.log_point_grid();
        // Form the relative system and repeat 
        const bool FULL_RSYS_SOLVE = false; // false // CAN CHANGE
        const bool GET_RSYS_MOMENTS = true; // DONT CHANGE
        const bool WITH_REL_APPROX = false; // false // CAN CHANGE
        const double EPS_REL_APPROX = 1e-8; // 1e-12 // CAN CHANGE
        C_COMPLEX_TYPE rsys_norm_factor[1];
        C_COMPLEX_TYPE rsys_cond_mean[2];
        C_COMPLEX_TYPE rsys_cond_covar[4];
        Cached2DCPDFTermContainer* rel_trans_2d_cached_terms = get_marg2d_relative_and_transformed_cpdf(
            &sys1_mce, &sys2_mce, Trel, FULL_RSYS_SOLVE, rel_cpdf_timing,
            GET_RSYS_MOMENTS, rsys_norm_factor, rsys_cond_mean, rsys_cond_covar,
            WITH_REL_APPROX, EPS_REL_APPROX);
        // Store conditional mean and variance
        convert_complex_array_to_real(rsys_cond_mean, rsys_cond_means + i*2, 2);
        convert_complex_array_to_real(rsys_cond_covar, rsys_cond_vars + i*4, 4);
        // Evaluate relative system grid
        int num_points_x, num_points_y;
        CauchyPoint3D* points = grid_eval_marg2d_relative_and_transformed_cpdf(rel_trans_2d_cached_terms, 
            xlow*rscale_fac + rsys_cond_means[i*2], xhigh*rscale_fac + rsys_cond_means[i*2], xdelta*rscale_fac, 
            ylow*rscale_fac + rsys_cond_means[i*2+1], yhigh*rscale_fac + rsys_cond_means[i*2+1], ydelta*rscale_fac,
            creal(sys1_mce.fz), creal(sys2_mce.fz),
            &num_points_x, &num_points_y, rel_cpdf_threads, FULL_RSYS_SOLVE, rel_cpdf_timing);
        log_marg2d_relative_and_transformed_cpdf(rel_sys_log_dir, i+1, points, num_points_x, num_points_y);
        rel_trans_2d_cached_terms->deinit();
        free(rel_trans_2d_cached_terms);
        free(points);
    }
    char sys1_fpath_cond_mean[200];
    sprintf(sys1_fpath_cond_mean, "%scond_means.txt", sys1_log_dir);
    char sys1_fpath_cond_var[200];
    sprintf(sys1_fpath_cond_var, "%scond_covars.txt", sys1_log_dir);
    char sys2_fpath_cond_mean[200];
    sprintf(sys2_fpath_cond_mean, "%scond_means.txt", sys2_log_dir);
    char sys2_fpath_cond_var[200];
    sprintf(sys2_fpath_cond_var, "%scond_covars.txt", sys2_log_dir);
    char rsys_fpath_cond_mean[200];
    sprintf(rsys_fpath_cond_mean, "%scond_means.txt", rel_sys_log_dir);
    char rsys_fpath_cond_var[200];
    sprintf(rsys_fpath_cond_var, "%scond_covars.txt", rel_sys_log_dir);
    char rsys_fpath_reltrans[200];
    sprintf(rsys_fpath_reltrans, "%sreltrans.txt", rel_sys_log_dir);
    log_double_array_to_file(sys1_fpath_cond_mean, sys1_cond_means, steps-1, n);
    log_double_array_to_file(sys1_fpath_cond_var, sys1_cond_vars, steps-1, n*n);
    log_double_array_to_file(sys2_fpath_cond_mean, sys2_cond_means, steps-1, n);
    log_double_array_to_file(sys2_fpath_cond_var, sys2_cond_vars, steps-1, n*n);
    log_double_array_to_file(rsys_fpath_cond_mean, rsys_cond_means, steps-1, 2);
    log_double_array_to_file(rsys_fpath_cond_var, rsys_cond_vars, steps-1, 4);
    log_double_array_to_file(rsys_fpath_reltrans, Trel, 2, n);
}

// When x = b, complex part is zero and real part has a derivative of zero (is extremized, but we dont know how)
C_COMPLEX_TYPE upper_bound_contribution1(Cached2DCPDFTerm* term, const bool FULL_SOLVE)
{
    return eval_rel_marg2d_cached_term_for_cpdf(term, term->b[0], term->b[1], FULL_SOLVE);
}

double ubc2_f(double* A, double* B, double C, double* X)
{
    return C + B[0]*X[0] + B[1]*X[1] + A[0] * X[0] * X[0] + A[1] * X[0] * X[1] + A[2] * X[0] * X[1] + A[3] * X[1] * X[1];
}
void ubc2_gradf(double* grad, double* A, double* B, double* X)
{
    grad[0] = B[0] + 2 * (A[0] * X[0] + A[1] * X[1]);
    grad[1] = B[1] + 2 * (A[2] * X[0] + A[3] * X[1]);
}

// Newton Raphson On Approximate polynomial  
C_COMPLEX_TYPE upper_bound_contribution2(Cached2DCPDFTerm* term, const bool FULL_SOLVE)
{
    // Only looking at the lower half of the integration of each cell right now
    double total_contrib = 0;
    const double EPS = 1e-4;
    for(int i = 0; i < term->m; i++)
    {
        double a1 = term->gam1_reals[i];
        double a2 = term->gam2_reals[i];
        double b1 = term->b[0];
        double b2 = term->b[1];
        double c = term->cos_thetas[i];
        double s = term->sin_thetas[i];

        double sb0 = c*a1*a1 - c*b1*b1 + s*a1*a2 - b1*b2*s;
        double sh0 = 2*a1*c*b1 + s*a2*b1 + s*a1*b2; 
        double sb1 = 2*c*b1 + b2*s;
        double sh1 = -2*a1*c - s*a2;
        double sb2 = s*b1;
        double sh2 = -s*a1;
        double sb3 = -s;
        double sb4 = -c;

        double C = sgn(sb0)*cabs(sb0 + I*sh0);
        double B[2] = { sgn(sb1)*cabs(sb1 + I*sh1), sgn(sb2)*cabs(sb2 + I*sh2) };
        double A[4] = {sb4, 0.5*sb3, 0.5*sb3, 0};
        double X[2] = {term->b[0], term->b[1]};
        double DX, FVAL, GRADVAL[2];
        int iteration = 0;
        DX = 1e8;
        while( fabs(DX) > EPS)
        {
            FVAL = ubc2_f(A, B, C, X);
            ubc2_gradf(GRADVAL, A, B, X);
            int j = iteration % 2;
            DX = -FVAL / GRADVAL[j];
            X[j] += DX;
            iteration += 1;
        }
        // Max value of contribution will be GVAL[i] / FVAL
        double contrib = cabs(term->g_vals[i]) / FVAL;
        total_contrib += contrib;
    }
    return total_contrib + 0*I;
}

// cr + br @ x + x @ A @ x
double ubc3_alpha1(double cr, double* br, double* Ar, double* x)
{
    double val = cr;
    double work[2];
    val += dot_prod(br, x, 2);
    matvecmul(Ar, x, work, 2, 2);
    val += dot_prod(x, work, 2);
    return val;
}
// br + 2 * A @ x
void ubc3_grad_alpha1(double* grad, double* br, double* Ar, double* x)
{
    double work[2];
    matvecmul(Ar,x,work,2,2);
    add_vecs(br, work, grad, 2, 2.0);
}

// ci + bi @ x
double ubc3_alpha2(double ci, double* bi, double* x)
{
    return ci + dot_prod(bi, x, 2);
}

// bi
void ubc3_grad_alpha2(double* grad, double* bi)
{
    memcpy(grad, bi, 2 * sizeof(double));
}

double ubc_g(double cr, double ci, double* br, double* bi, double* Ar, double* x)
{
    return pow(ubc3_alpha1(cr,br,Ar,x),2) + pow(ubc3_alpha2(ci,bi,x),2);
}

void ubc3_grad_g(double* grad, double cr, double ci, double* br, double* bi, double* Ar, double* x)
{
    double ga1[2]; // Gradient of alpha1 function 
    double* ga2; // Gradient of alpha2 function 
    double a1, a2; // Values of alpha1 and alpha2
    a1 = ubc3_alpha1(cr, br, Ar, x);
    a2 = ubc3_alpha2(ci, bi, x);
    ubc3_grad_alpha1(ga1, br, Ar, x);
    ga2 = bi; //ubc3_grad_alpha2(ga2, bi);
    grad[0] = 0; grad[1] = 0;
    add_vecs(grad, ga1, 2, 2*a1);
    add_vecs(grad, ga2, 2, 2*a2);
}

// H = BGFS Hessian, see Wiki for y and s definitions, or python code
void ubc3_hess_bfgs_g(double* H, double* y, double* s, double* work1, double* work2)
{
    double ys = y[0]*s[0] + y[1]*s[1];
    double sHs = H[0] * s[0] * s[0] + 2 * H[1] * s[1] * s[0] + H[3] * s[1] * s[1];
    // Hess = Hess + np.outer(y,y)/(y @ s) - Hess @ np.outer(s,s) @ Hess.T / (s @ Hess @ s)
    double _H[4];
    memcpy(_H, H, 4*sizeof(double));
    // outer(y,y)/(y @ s)
    work1[0] = y[0] * y[0] / ys;
    work1[1] = y[0] * y[1] / ys;
    work1[2] = work1[1];
    work1[3] = y[1] * y[1] / ys;
    add_vecs(H, work1, 4);
    // np.outer(s,s) / (s @ Hess @ s)
    work1[0] = s[0] * s[0] / sHs;
    work1[1] = s[0] * s[1] / sHs;
    work1[2] = work1[1];
    work1[3] = s[1] * s[1] / sHs;
    matmatmul(work1, _H, work2, 2,2,2,2, false, true); // work2 = np.outer(s,s) / (s @ Hess @ s) @ Hess.T
    matmatmul(_H, work2, work1, 2,2,2,2, false, false); // work1 = Hess @ work2
    sub_vecs(H, work1, 4);
}

// Real Hessian Check (do not use for optimization purposes)
void ubc3_hess_g(double* H, double cr, double ci, double* br, double* bi, double* Ar, double* x, double* work1, double* work2)
{
    double alpha1 = ubc3_alpha1(cr, br, Ar, x);
    double alpha2 = ubc3_alpha2(ci, bi, x);
    double GA1[2];
    double GA2[2];
    ubc3_grad_alpha1(GA1, br, Ar, x);
    ubc3_grad_alpha2(GA2, bi);
    outer_mat_prod(GA1, work1, 2, 1);
    add_vecs(work1, Ar, 4, 2*alpha1);
    outer_mat_prod(GA2, work2, 2, 1);
    add_vecs(work1,work2,H,4);
    scale_vec(H, 2.0, 4);
}

bool ubc3_check_hess_g(double* H)
{
    double evals[2];
    double evecs[4];
    memset(evals,0, 2*sizeof(double));
    memset(evecs,0, 4*sizeof(double));
    lapacke_sym_eig(H, evals, evecs, 2);
    if( (evals[0] >= -1e-15) && (evals[1]>= -1e-15) )
        return true;
    else
    {
        printf("HESS CHECK FAILED WITH EIGS: %.3E and %.3E\n", evals[0], evals[1]);
        return false;
    }
}

// BFGS on quadratic minimization of the real + imaginary parts of the denom
C_COMPLEX_TYPE upper_bound_contribution3(Cached2DCPDFTerm* term, const bool FULL_SOLVE)
{
    // Only looking at the lower half of the integration of each cell right now
    double total_contrib = 0;
    const double EPS = 1e-6;
    const double alpha = 0.5;
    const double beta = 0.8;
    const int BT_IT_LIMIT = 50;
    const int NEWT_IT_LIMIT = 25;
    const C_COMPLEX_TYPE BAD_RETURN = 1e8 + 1e8*I;
    const bool WITH_PRINT = true;
    for(int _i = 0; _i < 2*term->m; _i++)
    {
        int i = _i / 2;
        int j = _i % 2;
        double a1 = term->gam1_reals[i];
        double a2 = term->gam2_reals[i];
        double b1 = term->b[0];
        double b2 = term->b[1];
        double c = term->cos_thetas[i+j];
        double s = term->sin_thetas[i+j];

        double sb0 = c*a1*a1 - c*b1*b1 + s*a1*a2 - b1*b2*s;
        double sh0 = 2*a1*c*b1 + s*a2*b1 + s*a1*b2; 
        double sb1 = 2*c*b1 + b2*s;
        double sh1 = -2*a1*c - s*a2;
        double sb2 = s*b1;
        double sh2 = -s*a1;
        double sb3 = -s;
        double sb4 = -c;

        double Cr = sb0;
        double Ci = sh0;
        double Br[2] = {sb1, sb2};
        double Bi[2] = {sh1, sh2};
        double A[4] = {sb4, 0.5*sb3, 0.5*sb3, 0};
        double X[2] = {term->b[0], term->b[1]};
        double WORK[4]; double WORK2[4];
        double DX[2], GRADVAL[2], HESSVAL[4];
        double S[2], Y[2];

        int iteration = 0;
        bool abandon_loop = false;
        double GX = ubc_g(Cr, Ci, Br, Bi, A, X);
        ubc3_grad_g(GRADVAL, Cr, Ci, Br, Bi, A, X);
        ubc3_hess_g(HESSVAL, Cr, Ci, Br, Bi, A, X, WORK, WORK2);
        assert( ubc3_check_hess_g(HESSVAL) );
        HESSVAL[0] = 1; HESSVAL[1] = 0;
        HESSVAL[2] = 0; HESSVAL[3] = 1;
        while( (GRADVAL[0]*GRADVAL[0] + GRADVAL[1]*GRADVAL[1])  > EPS )
        {
            if(WITH_PRINT)
                printf("Iter %d, Grad Norm=%.3E, Func=%.3E\n", iteration, sqrt(dot_prod(GRADVAL, GRADVAL, 2)), GX);
            // Solves DX = - HESS^-1 GRAD (omits the negative until below)
            memcpy(WORK, HESSVAL, 4*sizeof(double));
            memcpy(WORK2, GRADVAL, 2*sizeof(double));
            solve_pd(WORK, WORK2, DX, 2, 1);
            DX[0] *= -1; DX[1] *= -1;
            // Backtrack on result
            double t = 1.0;
            int bt_iteration = 0;
            bool continue_backtracking = true;
            double agdx = alpha * (GRADVAL[0]*DX[0]+GRADVAL[1]*DX[1]);
            double GXT, GLIN;
            while(continue_backtracking)
            {
                // G(x + t*dx)
                WORK[0] = X[0] + t * DX[0];
                WORK[1] = X[1] + t * DX[1];
                GXT = ubc_g(Cr, Ci, Br, Bi, A, WORK);
                // G(x) + t*alpha*Grad @ dx
                GLIN = GX + t*agdx;
                // Test G(x + t*dx) > G(x) + t*alpha*Grad @ dx:
                if(GXT > GLIN)
                    t *= beta;
                else
                    continue_backtracking = false;
                if(bt_iteration > BT_IT_LIMIT)
                {
                    printf("Backtracked %d steps! Exiting! (||Grad||_2=%.3lf, FUNC=%.3lf)\n", BT_IT_LIMIT, sqrt(dot_prod(GRADVAL, GRADVAL, 2)), GX);
                    abandon_loop = true;
                    break;
                }
                bt_iteration++;
            }
            S[0] = DX[0] * t; 
            S[1] = DX[1] * t;
            X[0] += S[0];
            X[1] += S[1];
            // Update function value 
            GX = ubc_g(Cr, Ci, Br, Bi, A, X);
            // Update Gradient vector
            ubc3_grad_g(GRADVAL, Cr, Ci, Br, Bi, A, X);
            // Change in gradient: new - old
            Y[0] = GRADVAL[0] - WORK2[0]; // WORK2 contains old GRADVAL
            Y[1] = GRADVAL[1] - WORK2[1]; // WORK2 contains old GRADVAL
            // Use BFGS Approx of the Hessian matrix
            ubc3_hess_bfgs_g(HESSVAL, Y, S, WORK, WORK2);
            if(iteration > NEWT_IT_LIMIT)
            {
                printf("Optimized %d steps! Exiting! (||Grad||_2=%.3lf, FUNC=%.3lf)\n", NEWT_IT_LIMIT, sqrt(dot_prod(GRADVAL, GRADVAL, 2)), ubc_g(Cr, Ci, Br, Bi, A, X) );
                return BAD_RETURN;
            }
            if(abandon_loop)
                return BAD_RETURN;
            iteration++;
        }
        // Max value of contribution will be GVAL[i] / GX
        double contrib = cabs(term->g_vals[i]) / sqrt( fabs(GX) );
        total_contrib += contrib;
    }
    return 2*total_contrib + 0*I;
}

// BFGS on quadratic minimization of the real + imaginary parts of the denom
C_COMPLEX_TYPE upper_bound_contribution4(Cached2DCPDFTerm* term, const bool FULL_SOLVE)
{
    // Only looking at the lower half of the integration of each cell right now
    double total_contrib = 0;
    const C_COMPLEX_TYPE BAD_RETURN = 1e8 + 1e8*I;
    const double EPS = 1e-14;
    const bool WITH_PRINT = true;
    for(int i = 0; i < term->m; i++)
    {
        double a1 = term->gam1_reals[i];
        double a2 = term->gam2_reals[i];
        double c1 = term->cos_thetas[i];
        double s1 = term->sin_thetas[i];
        double c2 = term->cos_thetas[i+1];
        double s2 = term->sin_thetas[i+1];
        double den1 = a1*(a1*c1 + a2*s1);
        double den2 = a1*(a1*c2 + a2*s2);
        if( fabs(den1) < EPS )
            return BAD_RETURN;
        if( fabs(den2) < EPS )
            return BAD_RETURN;
        double g_val_r = cabs(term->g_vals[i]);
        // Max value of contribution will be GVAL[i] / GX
        double contrib = fabs( (g_val_r * s2) / den2 ) + fabs( (g_val_r * s1) / den1 );
        total_contrib += contrib;
    }
    return fabs(2*total_contrib) + 0*I;
}


// Testing possible approximation 
void attempt_relsys_approximation(double approx_eps, double importance_thresh, Cached2DCPDFTermContainer* cached_terms, CauchyPoint3D* points, int num_points, double sys1_fz, double sys2_fz, const bool FULL_SOLVE, const bool with_timing)
{
    assert(FULL_SOLVE);
    assert((approx_eps > 0) && (importance_thresh > 0));
    double norm_factor = RECIPRICAL_TWO_PI * RECIPRICAL_TWO_PI / (sys1_fz * sys2_fz);
    int num_rel_terms = cached_terms->current_term_idx;
    Cached2DCPDFTerm** approx_terms = (Cached2DCPDFTerm**) malloc(num_rel_terms * sizeof(Cached2DCPDFTerm*));
    bool* term_found_negligable = (bool*) malloc( num_rel_terms * sizeof(bool) );
    memset(term_found_negligable, 1, num_rel_terms * sizeof(bool) );
    // Loop over cached terms. Test approximate out condition. Collect Stats.
    int num_terms_keep = 0;
    int num_terms_discard = 0;
    Cached2DCPDFTerm* terms = cached_terms->cached_terms;
    
    for(int i = 0; i < num_rel_terms; i++)
    {
        Cached2DCPDFTerm* term = terms + i;
        // Run Approx Condition 
        C_COMPLEX_TYPE contrib = upper_bound_contribution4(term, FULL_SOLVE) * norm_factor;
        bool is_term_keep = (fabs(creal(contrib)) > approx_eps);
        if( is_term_keep )
        {
            approx_terms[i] = term;
            num_terms_keep++;
        }
        else
        {
            approx_terms[i] = NULL;
            num_terms_discard++;
        }
    }
    printf("Approx_Eps=%.3E leaves us with %d/%d terms -> %d were removed\n", approx_eps, num_terms_keep, num_rel_terms, num_terms_discard);
    
    if(num_terms_discard > 0)
    {
        double max_percent_diff_real = 0;
        double max_percent_diff_real_true_calc = 0;
        double max_percent_diff_real_approx_calc = 0;
        int max_percent_diff_real_term_idx = 0;

        double max_diff_real = 0;
        double max_diff_real_true_calc = 0;
        double max_diff_real_approx_calc = 0;
        int max_diff_real_term_idx = 0;

        double max_imag_approx_calc = 0;
        int max_imag_approx_idx = 0; 

        double cpdf_max_diff_real_above_thresh = 0;
        double cpdf_max_diff_real_above_thresh_true_calc = 0;
        double cpdf_max_diff_real_above_thresh_approx_calc = 0;
        int cpdf_max_diff_real_above_thresh_term_idx = -1;

        // Run diagnostics if terms were seen to be removed
        printf("Running Diagnostics - Terms (%d/%d) were found to be approximated out\n", num_terms_discard, num_rel_terms);
        printf("Grid Size is %d...Calculating...\n", num_points);
        for(int i = 0; i < num_points; i++)
        {
            CauchyPoint3D* point = points + i;
            C_COMPLEX_TYPE cpdf_val = 0;
            for(int j = 0; j < num_rel_terms; j++)
            {
                C_COMPLEX_TYPE term_val = eval_rel_marg2d_cached_term_for_cpdf(terms+j, point->x, point->y, FULL_SOLVE);
                
                bool is_negligable = ( (fabs(creal(term_val))*norm_factor) < approx_eps);
                term_found_negligable[j] *= is_negligable;
                // Evaluate the j-th term if it is not NULL
                if(approx_terms[j] != NULL)
                    cpdf_val += term_val;
                //if( (approx_terms[j] == NULL) && is_negligable == false ) 
                //    printf("Found One! Term Index %d, Point Index %d\n", j, i);
            }
            cpdf_val *= norm_factor;
            // Run Diagnostics 
            double cvr = fabs(creal(cpdf_val)); // cpdf val real 
            double cvi = fabs(cimag(cpdf_val)); // cpdf val real 
            double tvr = fabs(point->z); // true val real -- should be positive but just to ensure ~1e-18 are flipped positive

            double dcvr = fabs(cvr - tvr); // diff cvr
            double pdcvr = 100*dcvr/tvr; // percent dcvr
            // max_percent_diff_real
            if(pdcvr > max_percent_diff_real)
            {
                max_percent_diff_real = pdcvr;
                max_percent_diff_real_true_calc = tvr;
                max_percent_diff_real_approx_calc = cvr;
                max_percent_diff_real_term_idx = i;
            }
            // max_diff_real 
            if(dcvr > max_diff_real)
            {
                max_diff_real = dcvr;
                max_diff_real_true_calc = tvr;
                max_diff_real_approx_calc = cvr;
                max_diff_real_term_idx = i;
            }
            // max_imag_approx_calc
            if(cvi > max_imag_approx_calc)
            {
                max_imag_approx_calc = cvi;
                max_imag_approx_idx = i;
            }
            // cpdf_importance_thresh
            if(tvr > importance_thresh)
            {
                if(dcvr > cpdf_max_diff_real_above_thresh)
                {
                    cpdf_max_diff_real_above_thresh = dcvr;
                    cpdf_max_diff_real_above_thresh_true_calc = tvr;
                    cpdf_max_diff_real_above_thresh_approx_calc = cvr;
                    cpdf_max_diff_real_above_thresh_term_idx = i;
                }
            }
        }
        // Find whether terms marked using term_found_negligable were classified by our bound correctly on the grid
        int good_approx_out = 0;
        int bad_approx_out = 0;
        int good_keep = 0;
        int bad_keep = 0;
        for(int j = 0; j < num_rel_terms; j++)
        {
            // Four Cases:
            // Case 1: Term was Approxed out and term_found_negligable[i] == True  --> Good! Means we were smart in approxing it out
            if( (approx_terms[j] == NULL) && (term_found_negligable[j] == true) )
                good_approx_out++;
            // Case 2: Term was Approxed out and term_found_negligable[i] == False  --> Bad! Means we may be dumb in approxing it out
            else if( (approx_terms[j] == NULL) && (term_found_negligable[j] == false) )
                bad_approx_out++;
            // Case 3: Term was not Approxed out and term_found_negligable[i] == True  --> Bad! Means we may be dumb in keeping it
            else if( (approx_terms[j] != NULL) && (term_found_negligable[j] == true) )
                bad_keep++;
            // Case 4: Term was not Approxed out and term_found_negligable[i] == False  --> Good! Means we were smart in keeping it
            else
                good_keep++;
        } 
        // Print out summary statistics
        printf("Approx Bound Stats:\n  Good Approx:%d/%d\n  Bad Approx: %d/%d\n  Good Keep: %d/%d\n  Bad Keep: %d/%d\n", good_approx_out, num_rel_terms, bad_approx_out, num_rel_terms, good_keep, num_rel_terms, bad_keep, num_rel_terms);
        printf("Maximum Percentage Difference Between CPDF Values:\n  Max Percentage: %.2lf\n  True CPDF Value: %.3E\n  Approx CPDF Value: %.3E\n  Index: %d (Point x=%.3E, Point y=%.3E)\n",
            max_percent_diff_real, max_percent_diff_real_true_calc, max_percent_diff_real_approx_calc, max_percent_diff_real_term_idx, 
            points[max_percent_diff_real_term_idx].x, points[max_percent_diff_real_term_idx].y);
        printf("Maximum Difference Between CPDF Values:\n  Max Diff: %.3E\n  True CPDF Value: %.3E\n  Approx CPDF Value: %.3E\n  Index: %d (Point x = %.3E, Point y = %.3E)\n", 
            max_diff_real, max_diff_real_true_calc, max_diff_real_approx_calc, max_diff_real_term_idx,
            points[max_diff_real_term_idx].x, points[max_diff_real_term_idx].y);
        printf("Maximum Imaginary Value Observed:\n  Value: %.3E\n  Index: %d (Point x = %.3E, Point y = %.3E)\n", 
            max_imag_approx_calc, max_imag_approx_idx,
            points[max_imag_approx_idx].x, points[max_imag_approx_idx].y);
        printf("Maximum Difference Between CPDF Values Above Threshold of %.3E\n  Max Diff: %.3E\n  True CPDF Value: %.3E\n  Approx CPDF Value: %.3E\n  Index: %d (Point x = %.3E, Point y = %.3E)\n", 
            importance_thresh, cpdf_max_diff_real_above_thresh, cpdf_max_diff_real_above_thresh_true_calc, cpdf_max_diff_real_above_thresh_approx_calc, cpdf_max_diff_real_above_thresh_term_idx,
            points[cpdf_max_diff_real_above_thresh_term_idx].x, points[cpdf_max_diff_real_above_thresh_term_idx].y);
    }
    else 
    {
        printf("Not Running Diagnostics - No terms were found to be approximated out\n");
    }
    free(approx_terms);
    free(term_found_negligable);
}

void test_reltrans_3d_approx()
{
    time_t seed = 1717026144;//time(NULL);
    srand ( seed );
    printf("Seeding with %u\n", (uint)seed);
    // CPDF Grid 
    double xlow = -1.00;
    double xhigh = 1.00;
    double xdelta = 0.05;
    double ylow = -1.00;
    double yhigh = 1.00;
    double ydelta = 0.05;

    // Two 2D systems, offset from one another
    const int n = 3;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    
    // The relative transformation between the systems
    double Trel[2*n] = {1,0,0,0,1,0};//{0.5,  0.25,0.25,
                                     // 0.25, 0.5, 0.25};

    double sys1_Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double sys1_Gamma[n*pncc] = {.1, 0.3, -0.2};
    double* sys1_B = NULL;
    double sys1_H[n] = {1.0, 0.5, 0.2};
    double sys1_beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double sys1_gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double sys1_A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; // Unit directions of the initial state uncertainty
    double sys1_p0[n] = {0.10, 0.08, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double sys1_b0[n] = {0.18,0.18, 0.18}; // Initial median of system state
    double* sys1_u = NULL;
    
    double sys2_Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double sys2_Gamma[n*pncc] = {.1, 0.3, -0.2};
    double* sys2_B = NULL;
    double sys2_H[n] = {1.0, 0.5, 0.2};
    double sys2_beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double sys2_gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double sys2_A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; // Unit directions of the initial state uncertainty
    double sys2_p0[n] = {0.10, 0.08, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double sys2_b0[n] = {-0.18,-0.18, -0.18}; // Initial median of system state
    double* sys2_u = NULL;
    // MCE Steps and Debud Print Settings
    const int steps = 6;
    bool print_basic_info = false;
    // Dynamic System 1
    CauchyEstimator sys1_mce(sys1_A0, sys1_p0, sys1_b0, steps, n, cmcc, pncc, p, print_basic_info);
    // Dynamic System 2
    CauchyEstimator sys2_mce(sys2_A0, sys2_p0, sys2_b0, steps, n, cmcc, pncc, p, print_basic_info);
    // Relative System 
    double rsys_cond_means[steps*n];
    double rsys_cond_vars[steps*n*n];
    // Measurements 
    double sys1_zs[steps] = {0.25, 0.53, 0.43, 0.10, 0.31, 0.1672}; //, 0.5218};
    double sys2_zs[steps] = {-0.25, -0.53, -0.43, -0.10, -0.31, -0.8187}; //, 0.23983};
    assert(p==1);

    int num_eval_threads = 8;
    int num_cache_threads = num_eval_threads;
    bool rel_cpdf_timing = true;

    // Run Both Systems 
    // Loop over each step
    for(int i = 0; i < steps-1; i++)
    {
        printf("Step %d/%d\n", i+1,steps-1);
        // Loop over each measurement, per estimation step
        for(int j = 0; j < p; j++)
        {
            double sys1_z = sys1_zs[i];
            double sys2_z = sys2_zs[i];
            sys1_mce.step(sys1_z, sys1_Phi, sys1_Gamma, sys1_beta, sys1_H + (j%p)*n, sys1_gamma[j%p], sys1_B, sys1_u);
            sys2_mce.step(sys2_z, sys2_Phi, sys2_Gamma, sys2_beta, sys2_H + (j%p)*n, sys2_gamma[j%p], sys2_B, sys2_u);
        }
        
        // Form the relative system and repeat 
        const bool FULL_RSYS_SOLVE = false; // false // CAN CHANGE
        const bool GET_RSYS_MOMENTS = true; // DONT CHANGE
        C_COMPLEX_TYPE rsys_norm_factor[1];
        C_COMPLEX_TYPE rsys_cond_mean[2];
        C_COMPLEX_TYPE rsys_cond_covar[4];
        Cached2DCPDFTermContainer* rel_trans_2d_cached_terms = get_marg2d_relative_and_transformed_cpdf(
            &sys1_mce, &sys2_mce, Trel, FULL_RSYS_SOLVE, rel_cpdf_timing,
            GET_RSYS_MOMENTS, rsys_norm_factor, rsys_cond_mean, rsys_cond_covar);
        // Store conditional mean and variance
        convert_complex_array_to_real(rsys_cond_mean, rsys_cond_means + i*2, 2);
        convert_complex_array_to_real(rsys_cond_covar, rsys_cond_vars + i*4, 4);
        // Evaluate relative system grid
        int num_points_x, num_points_y;
        CauchyPoint3D* points = grid_eval_marg2d_relative_and_transformed_cpdf(rel_trans_2d_cached_terms, 
            xlow + rsys_cond_means[i*2], xhigh + rsys_cond_means[i*2], xdelta, 
            ylow + rsys_cond_means[i*2+1], yhigh + rsys_cond_means[i*2+1], ydelta,
            creal(sys1_mce.fz), creal(sys2_mce.fz),
            &num_points_x, &num_points_y, num_eval_threads, FULL_RSYS_SOLVE, rel_cpdf_timing);

        // Now attempt evaluation with approximation and compare to non-approximated
        ///*
        const bool WITH_TERM_APPROX = true;
        const double TERM_APPROX_EPS = 1e-8;
        Cached2DCPDFTermContainer* rel_trans_2d_cached_terms2 = get_marg2d_relative_and_transformed_cpdf(
            &sys1_mce, &sys2_mce, Trel, FULL_RSYS_SOLVE, rel_cpdf_timing,
            GET_RSYS_MOMENTS, rsys_norm_factor, rsys_cond_mean, rsys_cond_covar, WITH_TERM_APPROX, TERM_APPROX_EPS, num_cache_threads);
        CauchyPoint3D* points2 = grid_eval_marg2d_relative_and_transformed_cpdf(rel_trans_2d_cached_terms2, 
            xlow + rsys_cond_means[i*2], xhigh + rsys_cond_means[i*2], xdelta, 
            ylow + rsys_cond_means[i*2+1], yhigh + rsys_cond_means[i*2+1], ydelta,
            creal(sys1_mce.fz), creal(sys2_mce.fz),
            &num_points_x, &num_points_y, num_eval_threads, FULL_RSYS_SOLVE, rel_cpdf_timing);
        double max_diff = 0;
        int arg_max_diff = -1;
        for(int i = 0; i < num_points_x*num_points_y; i++)
        {
            double diff = fabs(points[i].z - points2[i].z);
            if( diff > max_diff )
            {
                max_diff = diff;
                arg_max_diff = i;
            }
        }
        printf("Max Difference is: %.3E at index: %d\n\n\n", max_diff, arg_max_diff);
        rel_trans_2d_cached_terms2->deinit();
        free(rel_trans_2d_cached_terms2);
        free(points2);
        //*/

        // Diagnostics
        // Run the approximation function and calculate the difference in terms and the approximation error
        //const double REL_APPROX_EPS = 1e-8;
        //const double IMPORTANCE_THRESH = 1e-5;
        //attempt_relsys_approximation(REL_APPROX_EPS, IMPORTANCE_THRESH, rel_trans_2d_cached_terms, points, num_points_x*num_points_y, creal(sys1_mce.fz), creal(sys2_mce.fz), FULL_RSYS_SOLVE, rel_cpdf_timing);
        rel_trans_2d_cached_terms->deinit();
        free(rel_trans_2d_cached_terms);
        free(points);
    }
}

void test_time_propagations()
{
    // Two 2D systems, offset from one another
    const int n = 3;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;

    double sys1_Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double sys1_Gamma[n*pncc] = {.1, 0.3, -0.2};
    double* sys1_B = NULL;
    double sys1_H[n] = {1.0, 0.5, 0.2};
    double sys1_beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double sys1_gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double sys1_A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; // Unit directions of the initial state uncertainty
    double sys1_p0[n] = {0.10, 0.08, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double sys1_b0[n] = {0.18,0.18, 0.18}; // Initial median of system state
    double* sys1_u = NULL;

    double sys1_Phi_inv[n*n];
    double work[n*n];
    inv(sys1_Phi, sys1_Phi_inv, work, n);

    double foo1[n*n] = {1.1, -0.8, -1.2,  -0.2,  0.7,  -0.25,  0.47, -0.9, 0.8};
    double foo2[n*n] = {1.2, -0.7, -1.0,  -0.4,  0.2,  0.25,  0.16, -0.7, -0.25};
    double prod_foo[n*n];
    double inv_prod_foo[n*n];
    matmatmul(foo2, foo1, prod_foo, 3, 3, 3, 3);
    inv(prod_foo, inv_prod_foo, work, n);
    // MCE Steps and Debud Print Settings
    const int steps = 6;
    bool print_basic_info = true;
    // Dynamic System 1
    CauchyEstimator sys1_mce(sys1_A0, sys1_p0, sys1_b0, steps, n, cmcc, pncc, p, print_basic_info);
    // Measurements 
    double sys1_zs[steps] = {0.25, 0.53, 0.43, 0.10, 0.31, 0.1672}; //, 0.5218};
    for(int i = 0; i < steps-1; i++)
    {
        double sys1_z = sys1_zs[i];
        sys1_mce.step(sys1_z, sys1_Phi, sys1_Gamma, sys1_beta, sys1_H, sys1_gamma[0], sys1_B, sys1_u);
        // Now Multiply By Phi and Get Moments
        //sys1_mce.deterministic_time_prop(sys1_Phi, NULL, NULL);
        //sys1_mce.compute_moments(false);
        // Now Multiply By Inv Phi and Get Original Moments back
        //sys1_mce.deterministic_time_prop(sys1_Phi_inv, NULL, NULL);
        //sys1_mce.compute_moments(false);

        // compound Phi testing
        sys1_mce.deterministic_time_prop(foo1, NULL, NULL);
        sys1_mce.compute_moments(false);
        sys1_mce.deterministic_time_prop(foo2, NULL, NULL);
        sys1_mce.compute_moments(false);
        sys1_mce.deterministic_time_prop(inv_prod_foo, NULL, NULL);
        sys1_mce.compute_moments(false);
        sys1_mce.deterministic_time_prop(prod_foo, NULL, NULL);
        sys1_mce.compute_moments(false);
        sys1_mce.deterministic_time_prop(inv_prod_foo, NULL, NULL);
        sys1_mce.compute_moments(false);
    }
}

int main()
{
    //test_reltrans_2d();
    //test_reltrans_3d();
    test_reltrans_3d_approx();
    //test_time_propagations();
}