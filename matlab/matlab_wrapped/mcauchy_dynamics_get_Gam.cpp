// file: mcauchy_dynamics_get_Gam.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_Gam:nrhs", "One input required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_Gam:nlhs", "One output required.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)pointerValue;
    
    int n = cduc->n;
    int pncc = cduc->pncc;
    plhs[0] = mxCreateDoubleMatrix(n, pncc, mxREAL);
    
    double *Gamma_out = mxGetPr(plhs[0]);
    
    int size_Gamma = n * pncc;
    for (int i = 0; i < size_Gamma; i++) {
        Gamma_out[i] = cduc->Gamma[i];
    }
}