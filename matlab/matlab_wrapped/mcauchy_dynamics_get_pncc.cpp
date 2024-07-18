// file: mcauchy_dynamics_get_pncc.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_get_pncc:nrhs", "One input required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_get_pncc:nlhs", "One output required.");
    }
    
    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)pointerValue;
    
    // Create the output mxArray and set its value
    plhs[0] = mxCreateDoubleScalar((double) cduc->pncc);
}