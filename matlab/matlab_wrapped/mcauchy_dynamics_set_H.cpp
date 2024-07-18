// file: mcauchy_dynamics_set_H.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_H:nrhs", "Two inputs required.");
    }
    if (nlhs > 0) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_H:nlhs", "No output expected.");
    }

    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)mxGetData(prhs[0]);
    double *input_H = mxGetPr(prhs[1]);
    mwSize input_rows = mxGetM(prhs[1]);
    mwSize input_cols = mxGetN(prhs[1]);

    // Ensure input_H has the correct dimensions
    if (input_rows != (mwSize)cduc->p || input_cols != (mwSize)cduc->n) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_H:sizeMismatch",
                          "The input matrix must match the dimensions (p x n).");
    }

    mwSize size_H = input_rows * input_cols;
    for (mwSize i = 0; i < size_H; i++) {
        cduc->H[i] = input_H[i];
    }
}
