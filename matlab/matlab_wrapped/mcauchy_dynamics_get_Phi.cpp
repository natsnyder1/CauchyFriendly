// file: mcauchy_dynamics_get_Phi.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Input validation
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_Phi:nrhs", "One input required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_Phi:nlhs", "One output required.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)pointerValue;
    
    int n = cduc->n;
    plhs[0] = mxCreateDoubleMatrix(n, n, mxREAL);
    double *Phi_out = mxGetPr(plhs[0]);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Phi_out[j * n + i] = cduc->Phi[i * n + j];
            // do this because matlab arrays are stored in memory in column-major order
        }
    }
}