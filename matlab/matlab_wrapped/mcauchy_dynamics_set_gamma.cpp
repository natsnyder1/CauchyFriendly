// file: mcauchy_dynamics_set_gamma.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Input validation
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_gamma:nrhs", "Two inputs required.");
    }
    if (nlhs > 0) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_gamma:nlhs", "No outputs expected.");
    }
    
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)mxGetData(prhs[0]);
    double *input_gamma = mxGetPr(prhs[1]);
    mwSize numElements = mxGetNumberOfElements(prhs[1]);
    
    // Ensure input_gamma has the correct size
    if (numElements != cduc->p) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_gamma:lengthMismatch",
                          "The input vector length must match the value of p.");
    }

    for (int i = 0; i < numElements; ++i) {
        cduc->gamma[i] = input_gamma[i];
    }
}