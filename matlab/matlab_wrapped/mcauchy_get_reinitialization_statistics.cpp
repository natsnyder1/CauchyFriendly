#include "mex.h"
#include "pycauchy.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 6) {
        mexErrMsgIdAndTxt("MATLAB:validation:nrhs",
                          "Six input arguments required.");
    }

    // Extract the pointer from the first input argument
    if (!mxIsUint64(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1) {
        mexErrMsgIdAndTxt("MATLAB:validation:invalidPointer",
                          "First input must be a valid uint64 pointer.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    void* _pcdh = (void*)pointerValue;
    
    double z = mxGetScalar(prhs[1]);
    double *xhat = mxGetPr(prhs[2]);
    double *Phat = mxGetPr(prhs[3]);
    double *H = mxGetPr(prhs[4]);
    double gamma = mxGetScalar(prhs[5]);

    double* out_A0 = nullptr;
    double* out_p0 = nullptr;
    double* out_b0 = nullptr;

    int size_out_A0, size_out_p0, size_out_b0;

    pycauchy_get_reinitialization_statistics(
    _pcdh, 
    z, 
    xhat, mxGetNumberOfElements(prhs[2]),
    Phat, mxGetNumberOfElements(prhs[3]),
    H, mxGetNumberOfElements(prhs[4]),
    gamma,
    &out_A0, &size_out_A0,
    &out_p0, &size_out_p0,
    &out_b0, &size_out_b0
    );

    plhs[0] = mxCreateDoubleMatrix(size_out_A0, 1, mxREAL);
    memcpy(mxGetPr(plhs[0]), out_A0, size_out_A0 * sizeof(double));
    free(out_A0);

    plhs[1] = mxCreateDoubleMatrix(size_out_p0, 1, mxREAL);
    memcpy(mxGetPr(plhs[1]), out_p0, size_out_p0 * sizeof(double));
    free(out_p0);

    plhs[2] = mxCreateDoubleMatrix(size_out_b0, 1, mxREAL);
    memcpy(mxGetPr(plhs[2]), out_b0, size_out_b0 * sizeof(double));
    free(out_b0);
}