#ifndef _PYCAUCHY_HPP_
#define _PY_CAUCHY_HPP_

#include "../../../include/cauchy_windows.hpp"


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
        memcpy(cduc->Gamma, Gamma, n*pncc*sizeof(double));
    if(B != NULL)
        memcpy(cduc->B, B, n * pncc * sizeof(double));
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
    int num_windows, 
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

    swm = new SlidingWindowManager(num_windows, REMOVE_SIM_NUM_STEPS, A0, p0, b0, cduc, debug_print, log_seq, log_full, false, NULL, NULL, NULL, win_var_boost, log_dir);
    null_ptr_check(swm);
}

void pycauchy_initialize_nonlin_window_manager(
    int num_windows, 
    double* x0, int size_x0, // xbar_0
    double* A0, int size_A0,
    double* p0, int size_p0,
    double* b0, int size_b0,
    double* beta, int size_beta,
    double* gamma, int size_gamma,
    int cmcc,
    void (*f_dyn_update_callback)(CauchyDynamicsUpdateContainer*),
    void (*f_nonlinear_msmt_model)(CauchyDynamicsUpdateContainer*, double*),
    void (*f_extended_msmt_update_callback)(CauchyDynamicsUpdateContainer*),
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
    assert_correct_cauchy_dynamics_update_container_setup(cduc);

    swm = new SlidingWindowManager(num_windows, REMOVE_SIM_NUM_STEPS, A0, p0, b0, cduc, debug_print, log_seq, log_full, true, f_dyn_update_callback, f_nonlinear_msmt_model, f_extended_msmt_update_callback, win_var_boost, log_dir);

}


void pycauchy_step(double* msmts, double* controls)
{
    swm->step(msmts, controls);
}


void pycauchy_shutdown()
{
    swm->shutdown();
    deallocate_cduc_memory();
    delete swm;
}


#endif //_PY_CAUCHY_HPP_