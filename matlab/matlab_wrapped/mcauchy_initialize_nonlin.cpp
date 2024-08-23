#include "mex.h"
#include "pycauchy.hpp"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 14) {
        mexErrMsgIdAndTxt("MATLAB:validation:nrhs",
                          "Fourteen input arguments required.");
    }
    int num_steps = static_cast<int>(mxGetScalar(prhs[0]));
    double *x0 = mxGetPr(prhs[1]);
    double *A0 = mxGetPr(prhs[2]);
    double *p0 = mxGetPr(prhs[3]);
    double *b0 = mxGetPr(prhs[4]);
    double *beta = mxGetPr(prhs[5]);
    double *gamma = mxGetPr(prhs[6]);

    uint64_t ptr1 = *((uint64_t*)mxGetData(prhs[7]));
    void (*f_dyn_update_callback)(CauchyDynamicsUpdateContainer*) = (void (*)(CauchyDynamicsUpdateContainer*))ptr1;
    uint64_t ptr2 = *((uint64_t*)mxGetData(prhs[8]));
    void (*f_nonlinear_msmt_model)(CauchyDynamicsUpdateContainer*, double*) = (void (*)(CauchyDynamicsUpdateContainer*, double*))ptr2;
    uint64_t ptr3 = *((uint64_t*)mxGetData(prhs[9]));
    void (*f_extended_msmt_update_callback)(CauchyDynamicsUpdateContainer*) = (void (*)(CauchyDynamicsUpdateContainer*))ptr3;

    int cmcc = static_cast<int>(mxGetScalar(prhs[10]));
    double dt = mxGetScalar(prhs[11]);
    int init_step = static_cast<int>(mxGetScalar(prhs[12]));
    bool debug_print = static_cast<bool>(mxGetScalar(prhs[13]));

    // Call the function from the pycauchy library
    void* result = pycauchy_initialize_nonlin(
        num_steps,
        x0, mxGetNumberOfElements(prhs[1]),
        A0, mxGetNumberOfElements(prhs[2]),
        p0, mxGetNumberOfElements(prhs[3]),
        b0, mxGetNumberOfElements(prhs[4]),
        beta, mxGetNumberOfElements(prhs[5]),
        gamma, mxGetNumberOfElements(prhs[6]),
        f_dyn_update_callback,
        f_nonlinear_msmt_model,
        f_extended_msmt_update_callback,
        cmcc,
        dt, init_step, debug_print
    );

    // Return the pointer as a MATLAB uint64
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*)mxGetData(plhs[0])) = (uint64_t)result;
}
