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

    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)mxGetData(prhs[0]);
    int n = cduc->n;
    plhs[0] = mxCreateDoubleMatrix(n, n, mxREAL);
    double *Phi_out = mxGetPr(plhs[0]);
    
    // Assuming column-major order for MATLAB arrays
    int size_Phi = n * n;
    for (int i = 0; i < size_Phi; ++i) {
        Phi_out[i] = cduc->Phi[i];
    }
}