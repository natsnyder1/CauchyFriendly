// file: mcauchy_dynamics_set_zbar.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_zbar:nrhs", "Four inputs required.");
    }
    if (nlhs > 0) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_zbar:nlhs", "No output expected.");
    }

    uint64_t pointerValue0 = *((uint64_t*)mxGetData(prhs[0]));
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)pointerValue0;

    uint64_t pointerValue1 = *((uint64_t*)mxGetData(prhs[1]));
    double *c_zbar = (double *)pointerValue1;
    double *input_zbar = mxGetPr(prhs[2]);
    mwSize numElements = mxGetNumberOfElements(prhs[2]);
    double p = mxGetScalar(prhs[3]);

    // Ensure input_zbar has the correct length
    if (numElements != p) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_zbar:lengthMismatch",
                          "Input vector length must match the value of p.");
    }

    for (int i = 0; i < numElements; ++i) {
        c_zbar[i] = input_zbar[i];
    }
}