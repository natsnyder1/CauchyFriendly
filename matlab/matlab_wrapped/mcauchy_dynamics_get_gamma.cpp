// file: mcauchy_dynamics_get_gamma.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_gamma:nrhs", "One input required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_gamma:nlhs", "One output required.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)pointerValue;
    
    int p = cduc->p;
    plhs[0] = mxCreateDoubleMatrix(p, 1, mxREAL);
    double *gamma_out = mxGetPr(plhs[0]);

    for (int i = 0; i < p; ++i) {
        gamma_out[i] = cduc->gamma[i];
    }
}