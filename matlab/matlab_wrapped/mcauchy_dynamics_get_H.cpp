// file: mcauchy_dynamics_get_H.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_H:nrhs", "One input required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_H:nlhs", "One output required.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)pointerValue;
    
    int n = cduc->n;
    int p = cduc->p;

    // Transpose the dimensions because MATLAB uses column-major order
    plhs[0] = mxCreateDoubleMatrix(p, n, mxREAL);
    double *H_out = mxGetPr(plhs[0]);

    // column-major
    // CHECK THIS AGAIN LATER
    int size_H = n * p;
    for (int i = 0; i < size_H; ++i) {
        H_out[i] = cduc->H[i];
    }
}