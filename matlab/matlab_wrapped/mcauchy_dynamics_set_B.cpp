// file: mcauchy_dynamics_set_B.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_B:nrhs", "Two inputs required.");
    }
    if (nlhs > 0) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_B:nlhs", "No output expected.");
    }
    
    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)pointerValue;
    
    double *input_B = mxGetPr(prhs[1]);
    mwSize cols = mxGetN(prhs[1]);
    mwSize rows = mxGetM(prhs[1]);
    
    // Ensure the input_B has the correct dimensions
    if (rows != cduc->n || cols != cduc->cmcc) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_B:sizeMismatch",
                          "The input matrix must have dimensions n x cmcc.");
    }

    int size_B = rows * cols;
    for (int i = 0; i < size_B; i++) {
        cduc->B[i] = input_B[i];
    }
}