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
    
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)mxGetData(prhs[0]);
    double *input_Phi = mxGetPr(prhs[1]);
    mwSize cols = mxGetN(prhs[1]);
    mwSize rows = mxGetM(prhs[1]);
    
    if (rows != cduc->n || cols != cduc->n) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_Phi:sizeMismatch",
                          "Input matrix must be of size n x n.");
    }

    int size_Phi = rows * cols;
    for (int i = 0; i < size_Phi; ++i) {
        cduc->Phi[i] = input_Phi[i];
    }
}