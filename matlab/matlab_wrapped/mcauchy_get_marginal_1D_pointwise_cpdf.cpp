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
    
    int marg_idx1 = static_cast<int>(mxGetScalar(prhs[1]));
    double gridx_low = mxGetScalar(prhs[2]);
    double gridx_high = mxGetScalar(prhs[3]);
    double gridx_resolution = mxGetScalar(prhs[4]);
    char *log_dir = mxArrayToString(prhs[5]);
    if (log_dir == NULL) {
        mexErrMsgIdAndTxt("MATLAB:conversionFailed", "Could not convert input to string.");
    }
    if(*log_dir == 'n')
    {
        log_dir = nullptr;
        //printf("Successfully set log dir to NULL 1D!\n");
    }

    double* out_cpdf_data = nullptr;
    int size_out_cpdf_data, out_num_gridx;

    pycauchy_get_marginal_1D_pointwise_cpdf(
    _pcdh, 
    marg_idx1,
    gridx_low, gridx_high, gridx_resolution, 
    log_dir, 
    &out_cpdf_data, &size_out_cpdf_data, &out_num_gridx
    );

    const mwSize dims[] = {2, mwSize(out_num_gridx), 1};

    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    memcpy(mxGetPr(plhs[0]), out_cpdf_data, size_out_cpdf_data * sizeof(double));
    free(out_cpdf_data);

    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int* out_num_gridx_ptr = (int*)mxGetData(plhs[1]);
    *out_num_gridx_ptr = out_num_gridx;

    mxFree(log_dir);
}