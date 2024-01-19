#ifndef _PY_CAUCHY_HPP_
#define _PY_CAUCHY_HPP_

#include "../../../include/cauchy_windows.hpp"
#include "../../../include/cpdf_ndim.hpp"
//#include "../../../include/cpdf_2d.hpp"

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
    double* out_swm_fz, 
    double** out_xhat, int* size_out_xhat,
    double** out_Phat, int* size_out_Phat, 
    double* out_swm_cerr_fz, 
    double* out_swm_cerr_xhat, 
    double* out_swm_cerr_Phat, 
    int* out_swm_win_idx, 
    int* out_swm_err_code)
{
    controls = (size_controls == 0) ? NULL : controls;
    swm->step(msmts, controls);
    // Now take the data back to python
    int data_idx = swm->msmt_count-1;
    if(data_idx < 0)
    {
        printf(RED "[ERROR pycauchy_step:] swm->msmt_count-1 is less than 0, this implies a bug! Please gdb here!" NC "\n");
        exit(1);
    }
    int n = swm->n;

    *size_out_xhat = n;
    *size_out_Phat = n*n;
    *out_xhat = (double*) malloc( n * sizeof(double) );
    *out_Phat = (double*) malloc( n * n * sizeof(double) );
    memcpy(*out_xhat, swm->full_window_means + data_idx * n, n * sizeof(double) );
    memcpy(*out_Phat, swm->full_window_variances + data_idx * n * n, n * n * sizeof(double) );
    *out_swm_fz = swm->full_window_norm_factors[data_idx];
    *out_swm_cerr_fz = swm->full_window_cerr_norm_factors[data_idx];
    *out_swm_cerr_xhat = swm->full_window_cerr_means[data_idx];
    *out_swm_cerr_Phat = swm->full_window_cerr_variances[data_idx];
    *out_swm_win_idx = swm->full_window_idxs[data_idx];
    *out_swm_err_code = swm->full_window_numeric_errors[data_idx];
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
    PointWiseNDimCauchyCPDF* cpdf;
    CauchyCPDFGridDispatcher1D* grid_1d;
    CauchyCPDFGridDispatcher2D* grid_2d;
    void (*f_dyn_update_callback)(CauchyDynamicsUpdateContainer*);
    void (*f_nonlinear_msmt_model)(CauchyDynamicsUpdateContainer*, double*);
    void (*f_extended_msmt_update_callback)(CauchyDynamicsUpdateContainer*);

    PyCauchyDataHandler()
    {
        duc = NULL;
        cauchyEst = NULL;
        cpdf = NULL;
        grid_1d = NULL;
        grid_2d = NULL;
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

void single_step_dynamics_allocate(bool is_lti,
    int n, int pncc, int cmcc, int p,
    double** out_Phi, int* size_out_Phi,
    double** out_Gamma, int* size_out_Gamma, 
    double** out_B, int* size_out_B,
    double** out_H, int* size_out_H,
    double** out_beta, int* size_out_beta,
    double** out_gamma, int* size_out_gamma)
{
    // No need to return the LTI dynamics as they dont change
    if(is_lti)
    {
        *size_out_Phi = 0; *out_Phi = (double*) malloc( 0 );
        *size_out_Gamma = 0; *out_Gamma = (double*) malloc( 0 );
        *size_out_B = 0; *out_B = (double*) malloc( 0 );
        *size_out_H = 0; *out_H = (double*) malloc( 0 );
        *size_out_beta = 0; *out_beta = (double*) malloc( 0 );
        *size_out_gamma = 0; *out_gamma = (double*) malloc( 0 );
    }
    // Return the LTV dynamics as they do change
    else 
    {
        *size_out_Phi = n*n; 
        *out_Phi = (double*) malloc( n*n*sizeof(double) );
        *size_out_Gamma = n*pncc; 
        *out_Gamma = (double*) malloc( n*pncc*sizeof(double) );
        *size_out_B = n*cmcc; 
        *out_B = (double*) malloc( n*cmcc*sizeof(double) );
        *size_out_H = p*n; 
        *out_H = (double*) malloc( p*n*sizeof(double) );
        *size_out_beta = pncc; 
        *out_beta = (double*) malloc( pncc * sizeof(double) );
        *size_out_gamma = p; 
        *out_gamma = (double*) malloc( p * sizeof(double) );
    }
}

void single_step_moment_info_allocate(
    int n, int num_moments,
    double** out_fz, int* size_out_fz,
    double** out_xhat, int* size_out_xhat,
    double** out_Phat, int* size_out_Phat, 
    double** out_cerr_fz, int* size_out_cerr_fz,
    double** out_cerr_xhat, int* size_out_cerr_xhat,
    double** out_cerr_Phat, int* size_out_cerr_Phat,
    int** out_err_code, int* size_out_err_code)
{
    *size_out_fz = num_moments;
    *out_fz = (double*) malloc( num_moments * sizeof(double) );
    *size_out_xhat = num_moments * n;
    *out_xhat = (double*) malloc(num_moments * n * sizeof(double) );
    *size_out_Phat = num_moments * n * n;
    *out_Phat = (double*) malloc( num_moments * n * n * sizeof(double) );
    *size_out_cerr_fz = num_moments;
    *out_cerr_fz = (double*) malloc( num_moments * sizeof(double) );
    *size_out_cerr_xhat = num_moments;
    *out_cerr_xhat = (double*) malloc( num_moments * sizeof(double) );
    *size_out_cerr_Phat = num_moments;
    *out_cerr_Phat = (double*) malloc( num_moments * sizeof(double) );
    *size_out_err_code = num_moments;
    *out_err_code = (int*) malloc( num_moments * sizeof(int) );
}

void pycauchy_single_step_ltiv(
    void* _pcdh,
    double* msmts, int size_msmts,
    double* controls, int size_controls,
    double** out_Phi, int* size_out_Phi,
    double** out_Gamma, int* size_out_Gamma, 
    double** out_B, int* size_out_B,
    double** out_H, int* size_out_H,
    double** out_beta, int* size_out_beta,
    double** out_gamma, int* size_out_gamma,
    double** out_fz, int* size_out_fz,
    double** out_xhat, int* size_out_xhat,
    double** out_Phat, int* size_out_Phat, 
    double** out_cerr_fz, int* size_out_cerr_fz,
    double** out_cerr_xhat, int* size_out_cerr_xhat,
    double** out_cerr_Phat, int* size_out_cerr_Phat,
    int** out_err_code, int* size_out_err_code)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    CauchyDynamicsUpdateContainer* duc = pcdh->duc;
    assert(duc->p >= size_msmts); // size_msmts can be smaller, used for reinitialization return values
    duc->u = controls;
    // Defining return sizes
    int p = duc->p;
    int n = duc->n;
    int cmcc = duc->cmcc;
    int pncc = duc->pncc;
    int num_moments = size_msmts;
    bool is_lti = pcdh->f_dyn_update_callback == NULL;
    bool is_ltv = pcdh->f_dyn_update_callback != NULL;

    single_step_dynamics_allocate(is_lti,
        n, pncc, cmcc, size_msmts,
        out_Phi, size_out_Phi,
        out_Gamma, size_out_Gamma, 
        out_B, size_out_B,
        out_H, size_out_H,
        out_beta, size_out_beta,
        out_gamma, size_out_gamma);
    
    single_step_moment_info_allocate(
        n, num_moments,
        out_fz, size_out_fz,
        out_xhat, size_out_xhat,
        out_Phat, size_out_Phat, 
        out_cerr_fz, size_out_cerr_fz,
        out_cerr_xhat, size_out_cerr_xhat,
        out_cerr_Phat, size_out_cerr_Phat,
        out_err_code, size_out_err_code);

    int msmt_start_idx = p - size_msmts; // ADDED
    // Update dynamics if linear time varying, before the call to step
    if(is_ltv)
    {
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_mean, duc->x, n);
        pcdh->f_dyn_update_callback(duc);
        // Store Updated Dynamics
        memcpy(*out_Phi, duc->Phi, n * n * sizeof(double));
        memcpy(*out_Gamma, duc->Gamma, n * pncc * sizeof(double));
        memcpy(*out_B, duc->B, n * cmcc * sizeof(double));
        memcpy(*out_H, duc->H + msmt_start_idx*n, size_msmts * n * sizeof(double));
        memcpy(*out_beta, duc->beta, pncc * sizeof(double));
        memcpy(*out_gamma, duc->gamma + msmt_start_idx, size_msmts * sizeof(double));
    }

    // Overloading this so it can be used for speyer reinitialization as well
    // when size_msmts == duc->p, behavior is normal
    // when size_msmts < duc->p, only processes size_msmts, of indices [p-size_msmts, p)
    //for(int i = 0; i < p; i++)
    for(int i = 0; i < size_msmts; i++)
    {
        int idx = msmt_start_idx + i; // ADDED
        pcdh->cauchyEst->step(msmts[i], duc->Phi, duc->Gamma, duc->beta, duc->H + idx*n, duc->gamma[idx], duc->B, duc->u);
        // Store moment info after i-th measurement update
        (*out_fz)[i] = creal(pcdh->cauchyEst->fz);
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_mean, *out_xhat + i*n, n);
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_variance, *out_Phat + i*n*n, n*n);
        (*out_cerr_fz)[i] = cimag(pcdh->cauchyEst->fz);
        (*out_cerr_xhat)[i] = max_abs_imag_carray(pcdh->cauchyEst->conditional_mean, n);
        (*out_cerr_Phat)[i] = max_abs_imag_carray(pcdh->cauchyEst->conditional_variance, n * n);
        (*out_err_code)[i] = pcdh->cauchyEst->numeric_moment_errors;
    }
}

void pycauchy_single_step_nonlin(
    void* _pcdh,
    double* msmts, int size_msmts,
    double* controls, int size_controls,
    bool with_propagate, 
    double** out_Phi, int* size_out_Phi,
    double** out_Gamma, int* size_out_Gamma, 
    double** out_B, int* size_out_B,
    double** out_H, int* size_out_H,
    double** out_beta, int* size_out_beta,
    double** out_gamma, int* size_out_gamma,
    double** out_fz, int* size_out_fz,
    double** out_xhat, int* size_out_xhat,
    double** out_Phat, int* size_out_Phat,
    double** out_xbar, int* size_out_xbar,
    double** out_zbar, int* size_out_zbar, 
    double** out_cerr_fz, int* size_out_cerr_fz,
    double** out_cerr_xhat, int* size_out_cerr_xhat,
    double** out_cerr_Phat, int* size_out_cerr_Phat,
    int** out_err_code, int* size_out_err_code)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    CauchyEstimator* cauchyEst = pcdh->cauchyEst;
    CauchyDynamicsUpdateContainer* duc = pcdh->duc;
    assert(duc->p >= size_msmts); // size_msmts can be smaller, used for reinitialization return values
    int n = duc->n;
    int pncc = duc->pncc;
    int cmcc = duc->cmcc;
    int p = duc->p;
    int num_moments = size_msmts;
    duc->u = controls;
    double zbar[p];
    
    single_step_dynamics_allocate(false,
        n, pncc, cmcc, size_msmts,
        out_Phi, size_out_Phi,
        out_Gamma, size_out_Gamma, 
        out_B, size_out_B,
        out_H, size_out_H,
        out_beta, size_out_beta,
        out_gamma, size_out_gamma);
    
    single_step_moment_info_allocate(
        n, num_moments,
        out_fz, size_out_fz,
        out_xhat, size_out_xhat,
        out_Phat, size_out_Phat, 
        out_cerr_fz, size_out_cerr_fz,
        out_cerr_xhat, size_out_cerr_xhat,
        out_cerr_Phat, size_out_cerr_Phat,
        out_err_code, size_out_err_code);
    *size_out_xbar = num_moments * n;
    *out_xbar = (double*) malloc( num_moments * n * sizeof(double) );
    *size_out_zbar = num_moments;
    *out_zbar = (double*) malloc( num_moments * sizeof(double) ); 

    // propagate system forwards (i.e, create \bar{x}_{k+1})
    // update the Phi, Gamma, H matrices for the differential system
    // this is called at all time steps except time step 0, here only measurement update occurs
    if(with_propagate) 
    {
        duc->is_xbar_set_for_ece = false;
        pcdh->f_dyn_update_callback(duc);
        assert(duc->is_xbar_set_for_ece == true);
        // Store Updated Dynamics (nonlinear info regarding time prop of k|k-1)
        memcpy(*out_Phi, duc->Phi, n * n * sizeof(double));
        memcpy(*out_B, duc->B, n * cmcc * sizeof(double));
        memcpy(*out_Gamma, duc->Gamma, n * pncc * sizeof(double));
        memcpy(*out_beta, duc->beta, pncc * sizeof(double));
    }
    int msmt_start_idx = p - size_msmts; // ADDED
    // Overloading this so it can be used for speyer reinitialization as well
    // when size_msmts == duc->p, behavior is normal
    // when size_msmts < duc->p, only processes size_msmts, of indices [p-size_msmts, p)
    //for(int i = 0; i < p; i++)
    for(int i = 0; i < size_msmts; i++)
    {
        int idx = msmt_start_idx + i; // ADDED
        // Run state variation estimator
        pcdh->f_nonlinear_msmt_model(duc, zbar); // duc->x == x_bar on i==0 and x_hat on i>0
        pcdh->f_extended_msmt_update_callback(duc);
        
        // Store nonlinear info regarding k|k-1
        memcpy(*out_xbar + i*n, duc->x, n*sizeof(double));
        (*out_zbar)[i] = zbar[idx];
        double dz = msmts[i] - zbar[idx];
        memcpy(*out_H + i*n, duc->H + idx*n, n*sizeof(double));
        (*out_gamma)[i] = duc->gamma[idx];

        cauchyEst->step(dz, duc->Phi, duc->Gamma, duc->beta, duc->H + idx*n, duc->gamma[idx], NULL, NULL);
        // Shifts bs in CF by -\delta{x_k}. Sets conditional_mean=\delta{x_k} + duc->x (which is x_bar). Then sets (duc->x) x_bar = creal(conditional_mean)
        cauchyEst->finalize_extended_moments(duc->x);

        // Store moment info after i-th measurement update
        (*out_fz)[i] = creal(pcdh->cauchyEst->fz);
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_mean, *out_xhat + i*n, n);
        convert_complex_array_to_real(pcdh->cauchyEst->conditional_variance, *out_Phat + i*n*n, n*n);
        (*out_cerr_fz)[i] = cimag(pcdh->cauchyEst->fz);
        (*out_cerr_xhat)[i] = max_abs_imag_carray(pcdh->cauchyEst->conditional_mean, n);
        (*out_cerr_Phat)[i] = max_abs_imag_carray(pcdh->cauchyEst->conditional_variance, n * n);
        (*out_err_code)[i] = pcdh->cauchyEst->numeric_moment_errors;
    }
    duc->step += 1;
}

void pycauchy_single_step_shutdown(void* _pcdh)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    deallocate_duc_memory(pcdh->duc);
    free(pcdh->duc);
    delete pcdh->cauchyEst;
    if(pcdh->cpdf != NULL)
        delete pcdh->cpdf;
    if(pcdh->grid_2d != NULL)
        delete pcdh->grid_2d;
    if(pcdh->grid_1d != NULL)
        delete pcdh->grid_1d;
    delete pcdh;
}

void pycauchy_single_step_set_master_step(void* _pcdh, int step)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    if(pcdh->cauchyEst == NULL)
    {
        printf(RED "[ERROR pycauchy_single_step_set_master_step:] pcdh->cauchyEst == NULL! Debug here! Exiting!\n");
        exit(1);
    }
    else
        pcdh->cauchyEst->master_step = step;
}

void pycauchy_single_step_set_window_number(void* _pcdh, int win_num)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    if(pcdh->cauchyEst == NULL)
    {
        printf(RED "[ERROR pycauchy_single_step_set_window_number:] pcdh->cauchyEst == NULL! Debug here! Exiting!\n");
        exit(1);
    }
    else
        pcdh->cauchyEst->win_num = win_num;
}


void pycauchy_single_step_reset(
    void* _pcdh, 
    double* A0, int size_A0, 
    double* p0, int size_p0, 
    double* b0, int size_b0, 
    double* xbar, int size_xbar)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    int n = pcdh->cauchyEst->d;
    if(size_A0 > 0)
        memcpy(pcdh->cauchyEst->A0_init, A0, n*n*sizeof(double));
    if(size_p0 > 0)
        memcpy(pcdh->cauchyEst->p0_init, p0, n*sizeof(double));
    if(size_b0 > 0)
        memcpy(pcdh->cauchyEst->b0_init, b0, n*sizeof(double));
    if(size_xbar > 0)
        memcpy(pcdh->duc->x, xbar, n*sizeof(double));
    if(pcdh->cauchyEst->master_step != 0)
        pcdh->cauchyEst->reset();
    else
        setup_first_term(&(pcdh->cauchyEst->childterms_workspace), pcdh->cauchyEst->terms_dp[n], A0, p0, b0, n);
}

// Construct marginal 2D cpdf 
void pycauchy_get_marginal_2D_pointwise_cpdf(
    void* _pcdh, 
    int marg_idx1, int marg_idx2,
    double gridx_low, double gridx_high, double gridx_resolution, 
    double gridy_low, double gridy_high, double gridy_resolution, 
    char* log_dir, 
    double** out_cpdf_data, int* size_out_cpdf_data, int* out_num_gridx, int* out_num_gridy)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    CauchyEstimator* cauchyEst = pcdh->cauchyEst;
    if(pcdh->cpdf == NULL)
    {
        pcdh->cpdf = new PointWiseNDimCauchyCPDF(cauchyEst); //PointWise2DCauchyCPDF(log_dir, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution);
        null_ptr_check(pcdh->cpdf);
    }
    if(pcdh->grid_2d == NULL)
    {
        pcdh->grid_2d = new CauchyCPDFGridDispatcher2D(pcdh->cpdf, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution, log_dir);
        null_ptr_check(pcdh->grid_2d);
    }
    else
        pcdh->grid_2d->reset_grid(gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution);
    
    if( pcdh->grid_2d->evaluate_point_grid(marg_idx1, marg_idx2, NUM_CPUS, true) )
    {
        // If something goes wrong...
        *out_num_gridx = 0;
        *out_num_gridy = 0;
        *size_out_cpdf_data = 0;
        *out_cpdf_data = (double*) malloc(0);
        return;
    }

    double* cpdf_points = (double*) pcdh->grid_2d->points;
    int num_gridx = pcdh->grid_2d->num_points_x;
    int num_gridy = pcdh->grid_2d->num_points_y;
    BYTE_COUNT_TYPE size_gridvals = num_gridx * num_gridy * 3 * sizeof(double);
    *out_num_gridx = num_gridx;
    *out_num_gridy = num_gridy;
    *size_out_cpdf_data = 3 * num_gridx * num_gridy;
    *out_cpdf_data = (double*) malloc(size_gridvals);
    memcpy(*out_cpdf_data, cpdf_points, size_gridvals);
    if(log_dir != NULL)
        pcdh->grid_2d->log_point_grid();
}

// Constructing 2D cpdf 
void pycauchy_get_2D_pointwise_cpdf(
    void* _pcdh, 
    double gridx_low, double gridx_high, double gridx_resolution, 
    double gridy_low, double gridy_high, double gridy_resolution, 
    char* log_dir, 
    double** out_cpdf_data, int* size_out_cpdf_data, int* out_num_gridx, int* out_num_gridy)
{
    pycauchy_get_marginal_2D_pointwise_cpdf(
        _pcdh, 0,1,
        gridx_low, gridx_high, gridx_resolution, 
        gridy_low, gridy_high, gridy_resolution, log_dir, 
        out_cpdf_data, size_out_cpdf_data, out_num_gridx, out_num_gridy);
}

// Construct marginal 1D cpdf 
void pycauchy_get_marginal_1D_pointwise_cpdf(
    void* _pcdh, 
    int marg_idx1,
    double gridx_low, double gridx_high, double gridx_resolution, 
    char* log_dir, 
    double** out_cpdf_data, int* size_out_cpdf_data, int* out_num_gridx)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    CauchyEstimator* cauchyEst = pcdh->cauchyEst;
    if(pcdh->cpdf == NULL)
    {
        pcdh->cpdf = new PointWiseNDimCauchyCPDF(cauchyEst); //PointWise2DCauchyCPDF(log_dir, gridx_low, gridx_high, gridx_resolution, gridy_low, gridy_high, gridy_resolution);
        null_ptr_check(pcdh->cpdf);
    }
    if(pcdh->grid_1d == NULL)
    {
        pcdh->grid_1d = new CauchyCPDFGridDispatcher1D(pcdh->cpdf, gridx_low, gridx_high, gridx_resolution, log_dir);
        null_ptr_check(pcdh->grid_1d);
    }
    else
        pcdh->grid_1d->reset_grid(gridx_low, gridx_high, gridx_resolution);
    
    if( pcdh->grid_1d->evaluate_point_grid(marg_idx1, NUM_CPUS, true) )
    {
        // If something goes wrong...
        *out_num_gridx = 0;
        *size_out_cpdf_data = 0;
        *out_cpdf_data = (double*) malloc(0);
        return;
    }

    double* cpdf_points = (double*) pcdh->grid_1d->points;
    int num_gridx = pcdh->grid_1d->num_grid_points;
    BYTE_COUNT_TYPE size_gridvals = num_gridx * 2 * sizeof(double);
    *out_num_gridx = num_gridx;
    *size_out_cpdf_data = 2 * num_gridx;
    *out_cpdf_data = (double*) malloc(size_gridvals);
    memcpy(*out_cpdf_data, cpdf_points, size_gridvals);
    if(log_dir != NULL)
        pcdh->grid_1d->log_point_grid();
}

// Construct 1D cpdf
void pycauchy_get_1D_pointwise_cpdf(
    void* _pcdh,
    double gridx_low, double gridx_high, double gridx_resolution, 
    char* log_dir, 
    double** out_cpdf_data, int* size_out_cpdf_data, int* out_num_gridx)
{
    pycauchy_get_marginal_1D_pointwise_cpdf(
        _pcdh, 0,
        gridx_low, gridx_high, gridx_resolution, 
        log_dir, out_cpdf_data, size_out_cpdf_data, out_num_gridx);
}


// Gets reinitialization statistics to restart a new LTI estimator about this LTI estimator
void pycauchy_get_reinitialization_statistics(
    void* _pcdh, 
    double z, 
    double* xhat, int size_xhat,
    double* Phat, int size_Phat,
    double* H, int size_H,
    double gamma,
    double** out_A0, int* size_out_A0,
    double** out_p0, int* size_out_p0,
    double** out_b0, int* size_out_b0)
{
    PyCauchyDataHandler* pcdh = (PyCauchyDataHandler*) _pcdh;
    CauchyDynamicsUpdateContainer* duc = pcdh->duc;
    CauchyEstimator* cauchyEst = pcdh->cauchyEst;
    const int d = cauchyEst->d;

    *out_A0 = (double*) malloc(d * d * sizeof(double) );
    *size_out_A0 = d*d;
    *out_p0 = (double*) malloc(d * sizeof(double) );
    *size_out_p0 = d;
    *out_b0 = (double*) malloc(d * sizeof(double) );
    *size_out_b0 = d;
    speyers_window_init(cauchyEst->d, xhat, Phat, H, gamma, z, *out_A0, *out_p0, *out_b0, 0, 1, NULL);
}

void pycauchy_speyers_window_init(
                double* xhat, int size_xhat,
                double* Phat, int size_Phat,
                double* H, int size_H,
                double gamma, double z,  
                double** out_A0, int* size_out_A0,
                double** out_p0, int* size_out_p0,
                double** out_b0, int* size_out_b0)
{
    int n = size_xhat;
    assert(size_H == n);

    *out_A0 = (double*) malloc(n*n*sizeof(double));
    *size_out_A0 = n*n;
    *out_p0 = (double*) malloc(n*sizeof(double));
    *size_out_p0 = n;
    *out_b0 = (double*) malloc(n*sizeof(double));
    *size_out_b0 = n;
    speyers_window_init(n, xhat, Phat, H, gamma, z, *out_A0, *out_p0, *out_b0, 0, 1, NULL);
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