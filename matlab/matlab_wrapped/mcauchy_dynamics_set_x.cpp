// file: mcauchy_dynamics_set_x.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_x:nrhs", "Two inputs required.");
    }
    if (nlhs > 0) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_x:nlhs", "No output expected.");
    }
    
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)mxGetData(prhs[0]);
    double *input_x = mxGetPr(prhs[1]);
    mwSize numElements = mxGetNumberOfElements(prhs[1]);

    if (numElements != cduc->n) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_x:sizeMismatch", 
                          "Size of input does not match size of original x array.");
    }
    
    for (int i = 0; i < cduc->n; i++) {
        cduc->x[i] = input_x[i];
    }
}