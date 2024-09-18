#include "mex.h"
#include "pycauchy.hpp"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("MATLAB:validation:nrhs",
                          "Five input arguments required.");
    }

    // Extract the pointer from the first input argument
    if (!mxIsUint64(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1) {
        mexErrMsgIdAndTxt("MATLAB:validation:invalidPointer",
                          "First input must be a valid uint64 pointer.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    void* _pcdh = (void*)pointerValue;

    double* A0 = mxGetPr(prhs[1]);
    double* p0 = mxGetPr(prhs[2]);
    double* b0 = mxGetPr(prhs[3]);
    double* xbar = mxGetPr(prhs[4]);
    
    // Added 
    int n = mxGetNumberOfElements(prhs[2]);
    double* _A0 = (double*) malloc( n * n * sizeof(double));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            _A0[i*n+j] = A0[j*n+i];

    pycauchy_single_step_reset(
    _pcdh, 
    _A0, mxGetNumberOfElements(prhs[1]), 
    p0, mxGetNumberOfElements(prhs[2]), 
    b0, mxGetNumberOfElements(prhs[3]), 
    xbar, mxGetNumberOfElements(prhs[4])
    );
    free(_A0);
}