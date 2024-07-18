// file: mcauchy_dynamics_set_is_xbar_set_for_ece.cpp

#include "mex.h"
#include "dynamic_models.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_is_xbar_set_for_ece:nrhs", "Two inputs required.");
    }
    if (nlhs > 0) {
        mexErrMsgIdAndTxt("MATLAB:mcauchy_dynamics_set_is_xbar_set_for_ece:nlhs", "No outputs expected.");
    }

    CauchyDynamicsUpdateContainer *cduc = (CauchyDynamicsUpdateContainer *)mxGetData(prhs[0]);
    mxLogical *is_set = mxGetLogicals(prhs[1]);

    cduc->is_xbar_set_for_ece = is_set[0];
}