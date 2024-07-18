// file: mcauchy_dynamics_set_Gam.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_Gam:nrhs", "Two inputs required.");
    }
    if (nlhs > 0) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_Gam:nlhs", "No output expected.");
    }
    
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)mxGetData(prhs[0]);
    double *input_Gamma = mxGetPr(prhs[1]);
    mwSize cols = mxGetN(prhs[1]);
    mwSize rows = mxGetM(prhs[1]);
    
    if (rows != cduc->n || cols != cduc->pncc) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_Gam:sizeMismatch",
                          "The input matrix must have dimensions n x pncc.");
    }

    int size_Gamma = rows * cols;
    for (int i = 0; i < size_Gamma; i++) {
        cduc->Gamma[i] = input_Gamma[i];
    }
}