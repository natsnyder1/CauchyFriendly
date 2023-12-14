#ifndef _PY_CAUCHY_HPP_
#define _PY_CAUCHY_HPP_

#include "../../../include/cauchy_windows.hpp"
#include "../../../include/cpdf_2d.hpp"

CauchyDynamicsUpdateContainer* cduc;
SlidingWindowManager* swm;


void allocate_cduc_memory(int n, int pncc, int cmcc, int p)
{
    cduc = (CauchyDynamicsUpdateContainer*) malloc( sizeof(CauchyDynamicsUpdateContainer));
    null_ptr_check(cduc);
    cduc->n = n;
    cduc->pncc = pncc;
    cduc->cmcc = cmcc;
    cduc->p = p;
    cduc->x = (double*) malloc(n * sizeof(double));
    null_ptr_check(cduc->x);
    cduc->Phi = (double*) malloc(n * n * sizeof(double));
    null_ptr_check(cduc->Phi);
    cduc->Gamma = (double*) malloc(n * pncc * sizeof(double));
    if(pncc > 0)
        null_ptr_check(cduc->Gamma);
    cduc->B = (double*) malloc(n * cmcc * sizeof(double));
    if(cmcc > 0)
        null_ptr_check(cduc->B);
    cduc->H = (double*) malloc(p * n * sizeof(double));
    null_ptr_check(cduc->H);
    cduc->beta = (double*) malloc(pncc * sizeof(double));
    if(pncc > 0)
        null_ptr_check(cduc->beta);
    cduc->gamma = (double*) malloc(p * sizeof(double));
    null_ptr_check(cduc->gamma);
    cduc->other_stuff = NULL;
}

void deallocate_cduc_memory()
{
    free(cduc->Phi);
    free(cduc->Gamma);
    free(cduc->B);
    free(cduc->H);
    free(cduc->beta);
    free(cduc->gamma);
    free(cduc->x);
    free(cduc);
}

void transfer_pyparams_to_cduc(double* Phi, double* Gamma, double* B, double* beta, double* H, double* gamma, double* x)
{
    int n = cduc->n;
    int pncc = cduc->pncc;
    int cmcc = cduc->cmcc;
    int p = cduc->p;
    if(Phi != NULL)
        memcpy(cduc->Phi, Phi, n*n*sizeof(double));
    if(Gamma != NULL)
        memcpy(cduc->Gamma, Gamma, n * pncc*sizeof(double));
    if(B != NULL)
        memcpy(cduc->B, B, n * cmcc * sizeof(double));
    if(beta != NULL)
        memcpy(cduc->beta, beta, pncc * sizeof(double));
    if(H != NULL)
        memcpy(cduc->H, H, p * n * sizeof(double));
    if(gamma != NULL)
        memcpy(cduc->gamma, gamma, p * sizeof(double) );
    if(x != NULL)
        memcpy(cduc->x, x, n * sizeof(double));
}

void pycauchy_initialize_lti_window_manager(
    int num_windows, int num_sim_steps,
    double* A0, int size_A0,
    double* p0, int size_p0,
    double* b0, int size_b0,
    double* Phi, int size_Phi,
    double* Gamma, int size_Gamma,
    double* B, int size_B,
    double* beta, int size_beta,
    double* H, int size_H,
    double* gamma, int size_gamma,
    bool debug_print,
    bool log_seq,
    bool log_full, 
    char* log_dir,
    double dt, 
    int init_step, 
    double* win_var_boost, int size_wvb)
{
    const int n = size_b0;
    const int pncc = size_beta;
    const int cmcc = size_B / n;
    const int p = size_gamma;

    allocate_cduc_memory(n, pncc, cmcc, p);
    transfer_pyparams_to_cduc(Phi, Gamma, B, beta, H, gamma, NULL);
    cduc->dt = dt;
    cduc->step = init_step;
    assert_correct_cauchy_dynamics_update_container_setup(cduc);
    win_var_boost = (size_wvb == 0) ? NULL : win_var_boost;

    swm = new SlidingWindowManager(num_windows, num_sim_steps, A0, p0, b0, cduc, debug_print, log_seq, log_full, false, NULL, NULL, NULL, win_var_boost, log_dir);
    null_ptr_check(swm);
}


void pycauchy_initialize_ltv_window_manager(
    int num_windows, int num_sim_steps,
    double* A0, int size_A0,
    double* p0, int size_p0,
    double* b0, int size_b0,
    double* Phi, int size_Phi,
    double* Gamma, int size_Gamma,
    double* B, int size_B,
    double* beta, int size_beta,
    double* H, int size_H,
    double* gamma, int size_gamma,
    void (*f_dyn_update_callback)(CauchyDynamicsUpdateContainer*),
    bool debug_print,
    bool log_seq,
    bool log_full, 
    char* log_dir,
    double dt, 
    int init_step, 
    double* win_var_boost, int size_wvb)
{
    const int n = size_b0;
    const int pncc = size_beta;
    const int cmcc = size_B / n;
    const int p = size_gamma;

    allocate_cduc_memory(n, pncc, cmcc, p);
    transfer_pyparams_to_cduc(Phi, Gamma, B, beta, H, gamma, NULL);
    cduc->dt = dt;
    cduc->step = init_step;
    assert_correct_cauchy_dynamics_update_container_setup(cduc);
    win_var_boost = (size_wvb == 0) ? NULL : win_var_boost;

    swm = new SlidingWindowManager(num_windows, num_sim_steps, A0, p0, b0, cduc, debug_print, log_seq, log_full, false, f_dyn_update_callback, NULL, NULL, win_var_boost, log_dir);
    null_ptr_check(swm);
}

void pycauchy_initialize_nonlin_window_manager(
    int num_windows, int num_sim_steps,
    double* x0, int size_x0, // xbar_0
    double* A0, int size_A0,
    double* p0, int size_p0,
    double* b0, int size_b0,
    double* beta, int size_beta,
    double* gamma, int size_gamma,
    void (*f_dyn_update_callback)(CauchyDynamicsUpdateContainer*),
    void (*f_nonlinear_msmt_model)(CauchyDynamicsUpdateContainer*, double*),
    void (*f_extended_msmt_update_callback)(CauchyDynamicsUpdateContainer*),
    int cmcc,
    bool debug_print, 
    bool log_seq, 
    bool log_full, 
    char* log_dir, 
    double dt, 
    int init_step,
    double* win_var_boost, int size_wvb)
{
    const int n = size_b0;
    const int pncc = size_beta;
    const int p = size_gamma;
    allocate_cduc_memory(n, pncc, cmcc, p);
    transfer_pyparams_to_cduc(NULL, NULL, NULL, beta, NULL, gamma, x0);
    cduc->dt = dt;
    cduc->step = init_step;
    //assert_correct_cauchy_dynamics_update_container_setup(cduc);
    win_var_boost = (size_wvb == 0) ? NULL : win_var_boost;

    swm = new SlidingWindowManager(num_windows, num_sim_steps, A0, p0, b0, cduc, debug_print, log_seq, log_full, true, f_dyn_update_callback, f_nonlinear_msmt_model, f_extended_msmt_update_callback, win_var_boost, log_dir);
    null_ptr_check(swm);
}


void pycauchy_step(
    double* msmts, int size_msmts,
    double* controls, int size_controls,
    double* out_fz, 
    double** out_x, int* size_out_x,
    double** out_P, int* size_out_P, 
    double* out_cerr_fz, double* out_cerr_x, double* out_cerr_P, 
    int* out_win_idx, int* out_err_code)
{
    controls = (size_controls == 0) ? NULL : controls;
    swm->step(msmts, controls);
    // Now take the data back to python
    int data_idx = swm->msmt_count-1;
    if(data_idx < 0)
    {
        printf(RED "[ERROR pycauchy_step:] swm->msmt_count-1 is less than 0, this implies a bug! Please gdb here!\n");
        exit(1);
    }
    int n = swm->n;

    *size_out_x = n;
    *size_out_P = n*n;
    *out_x = (double*) malloc( n * sizeof(double) );
    *out_P = (double*) malloc( n * n * sizeof(double) );
    memcpy(*out_x, swm->full_window_means + data_idx * n, n * sizeof(double) );
    memcpy(*out_P, swm->full_window_variances + data_idx * n * n, n * n * sizeof(double) );
    *out_fz = swm->full_window_norm_factors[data_idx];
    *out_cerr_fz = swm->full_window_cerr_norm_factors[data_idx];
    *out_cerr_x = swm->full_window_cerr_means[data_idx];
    *out_cerr_P = swm->full_window_cerr_variances[data_idx];
    *out_win_idx = swm->full_window_idxs[data_idx];
    *out_err_code = swm->full_window_numeric_errors[data_idx];
}


void pycauchy_shutdown()
{
    swm->shutdown();
    deallocate_cduc_memory();
    delete swm;
}


// Single Cauchy Estimator instance handles 
struct PyCauchyDataHandler
{
    CauchyDynamicsUpdateContainer* duc;
    CauchyEstimator* cauchyEst;
    PointWise2DCauchyCPDF* cpdf;
    void (*f_dyn_update_callback)(CauchyDynamicsUpdateContainer*);
    void (*f_nonlinear_msmt_model)(CauchyDynamicsUpdateContainer*, double*);
    void (*f_extended_msmt_update_callback)(CauchyDynamicsUpdateContainer*);

    PyCauchyDataHandler()
    {
        duc = NULL;
        cauchyEst = NULL;
        cpdf = NULL;
        f_dyn_update_callback = NULL;
        f_nonlinear_msmt_model = NULL;
        f_extended_msmt_update_callback = NULL;
    }
};

void allocate_duc_memory(CauchyDynamicsUpdateContainer** duc, int n, int pncc, int cmcc, int p)
{
    *duc = (CauchyDynamicsUpdateContainer*) malloc(sizeof(CauchyDynamicsUpdateContainer));
    null_ptr_check(*duc);
    (*duc)->n = n;
    (*duc)->pncc = pncc;
    (*duc)->cmcc = cmcc;
    (*duc)->p = p;
    (*duc)->x = (double*) malloc(n * sizeof(double));
    null_ptr_check((*duc)->x);
    (*duc)->Phi = (double*) malloc(n * n * sizeof(double));
    null_ptr_check((*duc)->Phi);
    (*duc)->Gamma = (double*) malloc(n * pncc * sizeof(double));
    if(pncc > 0)
        null_ptr_check((*duc)->Gamma);
    (*duc)->B = (double*) malloc(n * cmcc * sizeof(double));
    if(cmcc > 0)
        null_ptr_check((*duc)->B);
    (*duc)->H = (double*) malloc(p * n * sizeof(double));
    null_ptr_check((*duc)->H);
    (*duc)->beta = (double*) malloc(pncc * sizeof(double));
    if(pncc > 0)
        null_ptr_check((*duc)->beta);
    (*duc)->gamma = (double*) malloc(p * sizeof(double));
    null_ptr_check((*duc)->gamma);
    (*duc)->other_stuff = NULL;
}

void deallocate_duc_memory(CauchyDynamicsUpdateContainer* duc)
{
    free(duc->Phi);
    free(duc->Gamma);
    free(duc->B);
    free(duc->H);
    free(duc->beta);
    free(duc->gamma);
    free(duc->x);
}

void* pycauchy_initialize_lti(
    int num_steps, 
    double* A0, int size_A0, 
    double* p0, int size_p0, 
    double* b0, int size_b0, 
    double* Phi, int size_Phi,
    double* Gamma, int size_Gamma,
    double* B, int size_B,
    double* beta, int size_beta, 
    double* H, int size_H,
    double* gamma, int size_gamma,
    double dt, 
    int init_step, 
    bool debug_print)
{
    const int n = size_b0;
    const int pncc = size_beta;
    const int cmcc = size_B / n;
    const int p = size_gamma;

    PyCauchyDataHandler* pcdh = new PyCauchyDataHandler();
    allocate_duc_memory( &(pcdh->duc), n, pncc, cmcc, p);
    CauchyDynamicsUpdateContainer* duc = pcdh->duc;
    memcpy(duc->Phi, Phi, size_Phi * sizeof(double));
    memcpy(duc->Gamma, Gamma, size_Gamma * sizeof(double));
    memcpy(duc->B, B, size_B * sizeof(double) );
    memcpy(duc->H, H, size_H * sizeof(double) );
    memcpy(duc->beta, beta, size_beta * sizeof(double) );
    memcpy(duc->gamma, gamma, size_gamma * sizeof(double) );
    duc->dt = dt;
    duc->step = init_step;
    pcdh->cauchyEst = new CauchyEstimator(A0, p0, b0, num_steps, n, cmcc, pncc, p, debug_print);
    return pcdh;
}

void* pycauchy_initialize_ltv(
    int num_steps, 
    double* A0, int size_A0, 
    double* p0, int size_p0, 
    double* b0, int size_b0, 
    double* Phi, int size_Phi,
    double* Gamma, int size_Gamma,
    double* B, int size_B,
    double* beta, int size_beta, 
    double* H, int size_H,
    double* gamma, int size_gamma,
    void (*f_dyn_update_callback)(CauchyDynamicsUpdateContainer*), 
    double dt, 
    int init_step, 
    bool debug_print)
{
    void* pyCauchyObj = pycauchy_initialize_lti
    (
        num_steps, 
        A0, size_A0, 
        p0, size_p0, 
        b0, size_b0, 
        Phi, size_Phi,
        Gamma, size_Gamma,
        B, size_B,
        beta, size_beta, 
        H, size_H,
        gamma, size_gamma,
        dt, init_step, debug_print
    );
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) pyCauchyObj;
    pcdh->f_dyn_update_callback = f_dyn_update_callback;
    return pcdh;
}

void* pycauchy_initialize_nonlin(
    int num_steps, 
    double* x0, int size_x0,
    double* A0, int size_A0, 
    double* p0, int size_p0,
    double* b0, int size_b0,
    double* beta, int size_beta,
    double* gamma, int size_gamma,
    void (*f_dyn_update_callback)(CauchyDynamicsUpdateContainer*),
    void (*f_nonlinear_msmt_model)(CauchyDynamicsUpdateContainer*, double*),
    void (*f_extended_msmt_update_callback)(CauchyDynamicsUpdateContainer*),
    int cmcc,  
    double dt, int init_step, bool debug_print)
{
    const int n = size_b0;
    const int pncc = size_beta;
    const int p = size_gamma;

    PyCauchyDataHandler* pcdh = new PyCauchyDataHandler();
    allocate_duc_memory( &(pcdh->duc), n, pncc, cmcc, p);
    CauchyDynamicsUpdateContainer* duc = pcdh->duc;
    memcpy(duc->x, x0, n * sizeof(double));
    memcpy(duc->beta, beta, size_beta * sizeof(double) );
    memcpy(duc->gamma, gamma, size_gamma * sizeof(double) );
    duc->dt = dt;
    duc->step = init_step;
    pcdh->cauchyEst = new CauchyEstimator(A0, p0, b0, num_steps, n, cmcc, pncc, p, debug_print);
    pcdh->f_dyn_update_callback = f_dyn_update_callback;
    pcdh->f_nonlinear_msmt_model = f_nonlinear_msmt_model;
    pcdh->f_extended_msmt_update_callback = f_extended_msmt_update_callback;
    return pcdh;
}

void pycauchy_single_step_ltiv(
    void* _pcdh,
    double* msmts, int size_msmts,
    double* controls, int size_controls,
    bool full_info,
    double* out_fz, 
    double** out_x, int* size_out_x,
    double** out_P, int* size_out_P, 
    double* out_cerr_fz, double* out_cerr_x, double* out_cerr_P, 
    int* out_err_code)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    CauchyDynamicsUpdateContainer* duc = pcdh->duc;
    assert(duc->p == size_msmts);
    duc->u = controls;

    if(full_info)
    {
        int num_moments = duc->p;
        *size_out_x = duc->n * num_moments;
        *size_out_P = duc->n*duc->n * num_moments;
        *out_x = (double*) malloc( num_moments * duc->n * sizeof(double) );
        *out_P = (double*) malloc( num_moments * duc->n * duc->n * sizeof(double) );
    }

    // Update dynamics if time varying, before the call to step
    if(pcdh->f_dyn_update_callback != NULL)
    {
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_mean, duc->x, duc->n);
        pcdh->f_dyn_update_callback(duc);
    }

    for(int i = 0; i < duc->p; i++)
    {
        pcdh->cauchyEst->step(msmts[i], duc->Phi, duc->Gamma, duc->beta, duc->H, duc->gamma[i], duc->B, duc->u);
        if(full_info)
        {
            convert_complex_array_to_real(pcdh->cauchyEst->conditional_mean, *out_x + i*duc->n, duc->n);
            convert_complex_array_to_real(pcdh->cauchyEst->conditional_variance, *out_P + i*duc->n*duc->n, duc->n*duc->n);
        }
    }
    duc->step += 1;
    
    // Return output data
    if(!full_info)
    {
        *size_out_x = duc->n;
        *size_out_P = duc->n*duc->n;
        *out_x = (double*) malloc( duc->n * sizeof(double) );
        *out_P = (double*) malloc( duc->n * duc->n * sizeof(double) );
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_mean, *out_x, duc->n);
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_variance, *out_P, duc->n*duc->n);
    }
    *out_fz = creal(pcdh->cauchyEst->fz);
    *out_cerr_fz = cimag(pcdh->cauchyEst->fz);
    *out_cerr_x = max_abs_imag_carray(pcdh->cauchyEst->conditional_mean, duc->n);
    *out_cerr_P = max_abs_imag_carray(pcdh->cauchyEst->conditional_variance, duc->n * duc->n);
    *out_err_code = pcdh->cauchyEst->numeric_moment_errors;
}

void pycauchy_single_step_nonlin(
    void* _pcdh,
    double* msmts, int size_msmts,
    double* controls, int size_controls,
    bool with_propagate,
    bool full_info,
    double* out_fz, 
    double** out_x, int* size_out_x,
    double** out_P, int* size_out_P, 
    double* out_cerr_fz, double* out_cerr_x, double* out_cerr_P, 
    int* out_err_code)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    CauchyEstimator* cauchyEst = pcdh->cauchyEst;
    CauchyDynamicsUpdateContainer* duc = pcdh->duc;
    const int n = duc->n;
    const int p = duc->p;
    assert(p == size_msmts);
    double* z_bar = (double*) malloc(p*sizeof(double));    
    double* zs = (double*) malloc(p*sizeof(double));  
    memcpy(zs, msmts, p * sizeof(double));
    duc->u = controls;

    if(full_info)
    {
        int num_moments = duc->p;
        *size_out_x = duc->n * num_moments;
        *size_out_P = duc->n*duc->n * num_moments;
        *out_x = (double*) malloc( num_moments * duc->n * sizeof(double) );
        *out_P = (double*) malloc( num_moments * duc->n * duc->n * sizeof(double) );
    }

    // propagate system forwards (i.e, create \bar{x}_{k+1})
    // update the Phi, Gamma, H matrices for the differential system
    // this is called at all time steps except time step 0, here only measurement update occurs
    if(with_propagate) 
    {
        duc->is_xbar_set_for_ece = false;
        pcdh->f_dyn_update_callback(duc);
        assert(duc->is_xbar_set_for_ece == true);
    }

    for(int i = 0; i < p; i++)
    {
        pcdh->f_nonlinear_msmt_model(duc, z_bar); // duc->x == x_bar on i==0 and x_hat on i>0
        zs[i] -= z_bar[i];
        pcdh->f_extended_msmt_update_callback(duc);
        int window_numeric_errors = cauchyEst->step(zs[i], duc->Phi, duc->Gamma, duc->beta, duc->H + i*n, duc->gamma[i], NULL, NULL);
        // Shifts bs in CF by -\delta{x_k}. Sets conditional_mean=\delta{x_k} + duc->x (which is x_bar). Then sets (duc->x) x_bar = creal(conditional_mean)
        cauchyEst->finalize_extended_moments(duc->x);

        if(full_info)
        {
            convert_complex_array_to_real(pcdh->cauchyEst->conditional_mean, *out_x + i*duc->n, duc->n);
            convert_complex_array_to_real(pcdh->cauchyEst->conditional_variance, *out_P + i*duc->n*duc->n, duc->n*duc->n);
        }
    }
    duc->step += 1;

    // Return output data
    if(!full_info)
    {
        *size_out_x = duc->n;
        *size_out_P = duc->n*duc->n;
        *out_x = (double*) malloc( duc->n * sizeof(double) );
        *out_P = (double*) malloc( duc->n * duc->n * sizeof(double) );
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_mean, *out_x, duc->n);
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_variance, *out_P, duc->n*duc->n);
    }
    *out_fz = creal(pcdh->cauchyEst->fz);
    *out_cerr_fz = cimag(pcdh->cauchyEst->fz);
    *out_cerr_x = max_abs_imag_carray(pcdh->cauchyEst->conditional_mean, duc->n);
    *out_cerr_P = max_abs_imag_carray(pcdh->cauchyEst->conditional_variance, duc->n * duc->n);
    *out_err_code = pcdh->cauchyEst->numeric_moment_errors;
    free(zs);
    free(z_bar);
}

void pycauchy_single_step_shutdown(void* _pcdh)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    deallocate_duc_memory(pcdh->duc);
    free(pcdh->duc);
    delete pcdh->cauchyEst;
    if(pcdh->cpdf != NULL)
        delete pcdh->cpdf;
    delete pcdh;
}


// Constructing 2D cpdf 
void pycauchy_get_2D_pointwise_cpdf(
    void* _pcdh, 
    double gridx_low, double gridx_high, double gridx_resolution, 
    double gridy_low, double gridy_high, double gridy_resolution, 
    char* log_dir, 
    double** out_cpdf_data, int* size_out_cpdf_data, int* out_num_gridx, int* out_num_gridy)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    CauchyEstimator* cauchyEst = pcdh->cauchyEst;
    if(pcdh->cpdf == NULL)
    {
        pcdh->cpdf = new PointWise2DCauchyCPDF(log_dir, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution);
        null_ptr_check(pcdh->cpdf);
    }
    else
        pcdh->cpdf->reset_grid(gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution);
    
    if( pcdh->cpdf->evaluate_point_wise_cpdf(pcdh->cauchyEst, NUM_CPUS) )
    {
        // If something goes wrong...
        *out_num_gridx = 0;
        *out_num_gridy = 0;
        *size_out_cpdf_data = 0;
        *out_cpdf_data = (double*) malloc(0);
    }

    double* cpdf_points = (double*) pcdh->cpdf->cpdf_points;
    int num_gridx = pcdh->cpdf->num_gridx;
    int num_gridy = pcdh->cpdf->num_gridy;
    BYTE_COUNT_TYPE size_gridvals = num_gridx * num_gridy * 3 * sizeof(double);
    *out_num_gridx = num_gridx;
    *out_num_gridy = num_gridy;
    *size_out_cpdf_data = 3 * num_gridx * num_gridy;
    *out_cpdf_data = (double*) malloc(size_gridvals);
    memcpy(*out_cpdf_data, cpdf_points, size_gridvals);
    if(log_dir != NULL)
        pcdh->cpdf->store_2d_cpdf(pcdh->cauchyEst->master_step-1);
}

// Gets reinitialization statistics to restart an estimator about the other

void pycauchy_get_reinitialization_statistics(
    void* _pcdh, double z,
    double** out_A0, int* size_out_A0,
    double** out_p0, int* size_out_p0,
    double** out_b0, int* size_out_b0 )
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    CauchyDynamicsUpdateContainer* duc = pcdh->duc;
    CauchyEstimator* cauchyEst = pcdh->cauchyEst;
    const int d = cauchyEst->d;
    double* x_hat = (double*) malloc( d * sizeof(double)); 
    double* P_hat = (double*) malloc( d * d * sizeof(double));
    convert_complex_array_to_real(cauchyEst->conditional_mean, x_hat, d); 
    convert_complex_array_to_real(cauchyEst->conditional_variance, P_hat, d*d); 

    *out_A0 = (double*) malloc(d * d * sizeof(double) );
    *size_out_A0 = d*d;
    *out_p0 = (double*) malloc(d * sizeof(double) );
    *size_out_p0 = d;
    *out_b0 = (double*) malloc(d * sizeof(double) );
    *size_out_b0 = d;

    int gam_idx = (cauchyEst->master_step-1) % duc->p;
    speyers_window_init(cauchyEst->d, x_hat, P_hat, duc->H + gam_idx*d, duc->gamma[gam_idx], z, *out_A0, *out_p0, *out_b0, 0, 1, NULL);

    free(x_hat);
    free(P_hat);
}

void pycauchy_speyers_window_init(
                double* x1_hat, int size_x1_hat,
                double* Var, int size_Var,
                double* H, int size_H,
                double gamma, double z,  
                double** out_A0, int* size_out_A0,
                double** out_p0, int* size_out_p0,
                double** out_b0, int* size_out_b0)
{
    int n = size_x1_hat;
    assert(size_H == n);

    *out_A0 = (double*) malloc(n*n*sizeof(double));
    *size_out_A0 = n*n;
    *out_p0 = (double*) malloc(n*sizeof(double));
    *size_out_p0 = n;
    *out_b0 = (double*) malloc(n*sizeof(double));
    *size_out_b0 = n;
    speyers_window_init(n, x1_hat, Var, H, gamma, z, *out_A0, *out_p0, *out_b0, 0, 1, NULL);
}

int pycauchy_set_tr_search_idxs_ordering(int* ordering, int size_ordering)
{
    if(size_ordering > 12)
    {    
        printf(RED "[ERROR pycauchy_set_tr_search_idxs_ordering:]\n\tCannot run estimation problems over 12 dimensions currently! Please fix the tr_search_idxs_ordering array in cauchy_constants.hpp to fix this and recompile!" NC "\n");
        return 1;
    }
    for(int i = 0; i < size_ordering; i++)
        TR_SEARCH_IDXS_ORDERING[i] = ordering[i];
    printf("tr_search_idxs_ordering has successfully been set to:\n");
    print_mat(TR_SEARCH_IDXS_ORDERING, 1, size_ordering);
    return 0;
}

#endif //_PY_CAUCHY_HPP_