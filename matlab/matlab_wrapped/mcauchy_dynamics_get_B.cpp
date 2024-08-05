// file: mcauchy_dynamics_get_B.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_B:nrhs", "One input required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_B:nlhs", "One output required.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)pointerValue;
    int n = cduc->n;
    int cmcc = cduc->cmcc;
    plhs[0] = mxCreateDoubleMatrix(n, cmcc, mxREAL);
    double *B_out = mxGetPr(plhs[0]);
    
    int size_B = n * cmcc;
    for (int i = 0; i < size_B; i++) {
        B_out[i] = cduc->B[i];
    }
}