#include "mex.h"
#include "pycauchy.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 13) {
        mexErrMsgIdAndTxt("MATLAB:validation:nrhs",
                          "Thirteen input arguments required.");
    }
    int num_steps = static_cast<int>(mxGetScalar(prhs[0]));
    double *data_A0 = mxGetPr(prhs[1]);
    double *data_p0 = mxGetPr(prhs[2]);
    double *data_b0 = mxGetPr(prhs[3]);
    double *data_Phi = mxGetPr(prhs[4]);
    double *data_Gamma = mxGetPr(prhs[5]);
    double *data_B = mxGetPr(prhs[6]); 
    double *data_beta = mxGetPr(prhs[7]);
    double *data_H = mxGetPr(prhs[8]);
    double *data_gamma = mxGetPr(prhs[9]);
    double dt = mxGetScalar(prhs[10]);
    int init_step = static_cast<int>(mxGetScalar(prhs[11]));
    bool debug_print = static_cast<bool>(mxGetScalar(prhs[12]));

    // Call the function from the pycauchy library
    void* result = pycauchy_initialize_lti(
        num_steps,
        data_A0, mxGetNumberOfElements(prhs[1]),
        data_p0, mxGetNumberOfElements(prhs[2]),
        data_b0, mxGetNumberOfElements(prhs[3]),
        data_Phi, mxGetNumberOfElements(prhs[4]),
        data_Gamma, mxGetNumberOfElements(prhs[5]),
        data_B, mxGetNumberOfElements(prhs[6]),
        data_beta, mxGetNumberOfElements(prhs[7]),
        data_H, mxGetNumberOfElements(prhs[8]),
        data_gamma, mxGetNumberOfElements(prhs[9]),
        dt,
        init_step,
        debug_print
    );

    // Return the pointer as a MATLAB uint64
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*)mxGetData(plhs[0])) = (uint64_t)result;
}
