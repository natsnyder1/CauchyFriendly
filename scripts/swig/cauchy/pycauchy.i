/* file : pycauchy.i */
  
/* name of module to use*/
%module pycauchy 
%{ 
    #define SWIG_FILE_WITH_INIT
    /* Every thing in this file is being copied in  
     wrapper file. We include the C header file necessary 
     to compile the interface */
    #include "pycauchy.hpp" 
  
    /* variable declaration*/
%} 

%include "typemaps.i"
%include "numpy.i"
%init %{
import_array();
%}

// Python Wrapper to tear down the C-side Sliding Window Manager
void pycauchy_shutdown();
void pycauchy_single_step_shutdown(void *_pcdh);

// Python Wrapper to step the C-side Sliding Window Manager
%apply (double* IN_ARRAY1, int DIM1) {(double* msmts, int size_msmts)};
%apply (double* IN_ARRAY1, int DIM1) {(double* controls, int size_controls)};
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double **out_x, int *size_out_x)};
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double **out_P, int *size_out_P)};
%apply double* OUTPUT { double *out_fz, double *out_cerr_x, double* out_cerr_P, double* out_cerr_fz };
%apply int* OUTPUT { int* out_win_idx, int* out_err_code };
void pycauchy_step(
    double* msmts, int size_msmts,
    double* controls, int size_controls,
    double* out_fz, 
    double** out_x, int* size_out_x,
    double** out_P, int* size_out_P, 
    double* out_cerr_fz, double* out_cerr_x, double* out_cerr_P, 
    int* out_win_idx, int* out_err_code);

void pycauchy_single_step_ltiv(
    void* _pcdh,
    double* msmts, int size_msmts,
    double* controls, int size_controls,
    bool full_info,
    double* out_fz, 
    double** out_x, int* size_out_x,
    double** out_P, int* size_out_P, 
    double* out_cerr_fz, double* out_cerr_x, double* out_cerr_P, 
    int* out_err_code);

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
    int* out_err_code);

%clear (double* msmts, int size_msmts);
%clear (double* controls, int size_controls);
%clear (double **out_x, int *size_out_x);
%clear (double **out_P, int *size_out_P);
%clear (double *out_fz, double *out_cerr_x, double* out_cerr_P, double* out_cerr_fz);
%clear (int* out_win_idx, int* out_err_code);

// Python Wrapper to initialize the C-side Sliding Window Manager for LTI systems
%apply (double* IN_ARRAY1, int DIM1) {(double* A0, int size_A0)};
%apply (double* IN_ARRAY1, int DIM1) {(double* p0, int size_p0)};
%apply (double* IN_ARRAY1, int DIM1) {(double* b0, int size_b0)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Phi, int size_Phi)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Gamma, int size_Gamma)};
%apply (double* IN_ARRAY1, int DIM1) {(double* B, int size_B)};
%apply (double* IN_ARRAY1, int DIM1) {(double* beta, int size_beta)};
%apply (double* IN_ARRAY1, int DIM1) {(double* H, int size_H)};
%apply (double* IN_ARRAY1, int DIM1) {(double* gamma, int size_gamma)};
%apply (double* IN_ARRAY1, int DIM1) {(double* win_var_boost, int size_wvb)};

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
    double* win_var_boost, int size_wvb);


%typemap(in) void (*f_dyn_update_callback)(CauchyDynamicsUpdateContainer*) {
    $1 = (void (*)(CauchyDynamicsUpdateContainer*))PyLong_AsVoidPtr($input);;
}
%typemap(in) void (*f_nonlinear_msmt_model)(CauchyDynamicsUpdateContainer*, double*) {
    $1 = (void (*)(CauchyDynamicsUpdateContainer*, double*))PyLong_AsVoidPtr($input);;
}
%typemap(in) void (*f_extended_msmt_update_callback)(CauchyDynamicsUpdateContainer*) {
    $1 = (void (*)(CauchyDynamicsUpdateContainer*))PyLong_AsVoidPtr($input);;
}

%{
    // Python Wrapper to initialize the C-side Sliding Window Manager for Nonlinear systems
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
        double* win_var_boost, int size_wvb);
    

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
        double* win_var_boost, int size_wvb);

    // Single Cauchy Estimator Instance 

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
        double dt, int init_step, bool debug_print);

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
        bool debug_print);

%}

%apply (double* IN_ARRAY1, int DIM1) {(double* x0, int size_x0)};
// Python Wrapper to initialize the C-side Sliding Window Manager for Nonlinear systems
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
    double* win_var_boost, int size_wvb);

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
    double* win_var_boost, int size_wvb);


// Single Cauchy Estimator Instance 
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
    double dt, int init_step, bool debug_print);

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
    bool debug_print);

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
    bool debug_print);

//void pycauchy_single_step_shutdown(void* _pcdh); // this was not placed in here but still is okay

%apply (double* IN_ARRAY1, int DIM1) {(double* xbar, int size_xbar)};
void pycauchy_single_step_reset(
    void* _pcdh, 
    double* A0, int size_A0, 
    double* p0, int size_p0, 
    double* b0, int size_b0, 
    double* xbar, int size_xbar);

// CPDF Wrappers
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double **out_cpdf_data, int *size_out_cpdf_data)};
%apply int* OUTPUT { int* out_num_gridx, int* out_num_gridy };

void pycauchy_get_marginal_2D_pointwise_cpdf(void* _pcdh, 
    int marg_idx1, int marg_idx2,
    double gridx_low, double gridx_high, double gridx_resolution, 
    double gridy_low, double gridy_high, double gridy_resolution, 
    char* log_dir, 
    double** out_cpdf_data, int* size_out_cpdf_data, int* out_num_gridx, int* out_num_gridy);

void pycauchy_get_2D_pointwise_cpdf(void* _pcdh, 
    double gridx_low, double gridx_high, double gridx_resolution, 
    double gridy_low, double gridy_high, double gridy_resolution, 
    char* log_dir, 
    double** out_cpdf_data, int* size_out_cpdf_data, int* out_num_gridx, int* out_num_gridy);

void pycauchy_get_marginal_1D_pointwise_cpdf(
    void* _pcdh, 
    int marg_idx1,
    double gridx_low, double gridx_high, double gridx_resolution, 
    char* log_dir, 
    double** out_cpdf_data, int* size_out_cpdf_data, int* out_num_gridx);

void pycauchy_get_1D_pointwise_cpdf(
    void* _pcdh,
    double gridx_low, double gridx_high, double gridx_resolution, 
    char* log_dir, 
    double** out_cpdf_data, int* size_out_cpdf_data, int* out_num_gridx);

// Helper Wrappers
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double **out_A0, int *size_out_A0)};
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double **out_p0, int *size_out_p0)};
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) {(double **out_b0, int *size_out_b0)};

void pycauchy_get_reinitialization_statistics(
    void* _pcdh, double z,
    double** out_A0, int* size_out_A0,
    double** out_p0, int* size_out_p0,
    double** out_b0, int* size_out_b0 );

%apply (double* IN_ARRAY1, int DIM1) {(double* x1_hat, int size_x1_hat)};
%apply (double* IN_ARRAY1, int DIM1) {(double* Var, int size_Var)};

void pycauchy_speyers_window_init(
                double* x1_hat, int size_x1_hat,
                double* Var, int size_Var,
                double* H, int size_H,
                double gamma, double z,  
                double** out_A0, int* size_out_A0,
                double** out_p0, int* size_out_p0,
                double** out_b0, int* size_out_b0);

%apply (int* IN_ARRAY1, int DIM1) {(int* ordering, int size_ordering)};
int pycauchy_set_tr_search_idxs_ordering(int* ordering, int size_ordering);