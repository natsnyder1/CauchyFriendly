#include "mex.h"
#include "pycauchy.hpp"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:validation:nrhs",
                          "Two input arguments required.");
    }

    // Extract the pointer from the first input argument
    if (!mxIsUint64(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1) {
        mexErrMsgIdAndTxt("MATLAB:validation:invalidPointer",
                          "First input must be a valid uint64 pointer.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    void* _pcdh = (void*)pointerValue;

    int step = static_cast<int>(mxGetScalar(prhs[1]));

    pycauchy_single_step_set_master_step(_pcdh, step);
}