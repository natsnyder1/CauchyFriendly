// file: mcauchy_dynamics_set_beta.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_beta:nrhs", "Two inputs required.");
    }
    if (nlhs > 0) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_beta:nlhs", "No output expected.");
    }
    
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)mxGetData(prhs[0]);
    double *input_beta = mxGetPr(prhs[1]);
    mwSize numElements = mxGetNumberOfElements(prhs[1]);
    
    // Ensure input_beta has the correct length
    if (numElements != cduc->pncc) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_beta:lengthMismatch",
                          "Length of input vector must match pncc.");
    }

    for (int i = 0; i < numElements; i++) {
        cduc->beta[i] = input_beta[i];
    }
}