// mcauchy_dynamics_get_x.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_x:nrhs", "One input required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_get_x:nlhs", "One output required.");
    }

    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)mxGetData(prhs[0]);
    plhs[0] = mxCreateDoubleMatrix(1, cduc->n, mxREAL);

    double *output_array = mxGetPr(plhs[0]);
    for (int i = 0; i < cduc->n; i++) {
        output_array[i] = cduc->x[i];
    }
}