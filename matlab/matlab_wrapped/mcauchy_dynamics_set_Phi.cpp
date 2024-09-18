// file: mcauchy_dynamics_set_Phi.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_Phi:nrhs", "Two inputs required.");
    }
    if (nlhs > 0) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_Phi:nlhs", "No outputs expected.");
    }
    
    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)pointerValue;
    
    double *input_Phi = mxGetPr(prhs[1]);
    mwSize cols = mxGetN(prhs[1]);
    mwSize rows = mxGetM(prhs[1]);
    
    if (rows != cduc->n || cols != cduc->n) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_Phi:sizeMismatch",
                          "Input matrix must be of size n x n.");
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < rows; j++ ) {
            cduc->Phi[i * cols + j] = input_Phi[j * rows + i];
            // store this way because matlab arrays are stored in memory in column major
        }
    }
}