#include "mex.h"
#include "pycauchy.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 11) {
        mexErrMsgIdAndTxt("MATLAB:validation:nrhs",
                          "Eleven input arguments required.");
    }

    // Extract the pointer from the first input argument
    if (!mxIsUint64(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1) {
        mexErrMsgIdAndTxt("MATLAB:validation:invalidPointer",
                          "First input must be a valid uint64 pointer.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    void* _pcdh = (void*)pointerValue;
    
    int marg_idx1 = static_cast<int>(mxGetScalar(prhs[1]));
    int marg_idx2 = static_cast<int>(mxGetScalar(prhs[2]));
    double gridx_low = mxGetScalar(prhs[3]);
    double gridx_high = mxGetScalar(prhs[4]);
    double gridx_resolution = mxGetScalar(prhs[5]);
    double gridy_low = mxGetScalar(prhs[6]);
    double gridy_high = mxGetScalar(prhs[7]);
    double gridy_resolution = mxGetScalar(prhs[8]);
    char *log_dir = mxArrayToString(prhs[9]);
    bool reset_cache = static_cast<int>(mxGetScalar(prhs[10]));
    if (log_dir == NULL) {
        mexErrMsgIdAndTxt("MATLAB:conversionFailed", "Could not convert input to string.");
    }
    if(*log_dir == 'n')
    {
        log_dir = nullptr;
        //printf("Successfully set log dir to NULL 2D!\n");
    }

    double* out_cpdf_data = nullptr;
    int size_out_cpdf_data, out_num_gridx, out_num_gridy;

    pycauchy_get_marginal_2D_pointwise_cpdf(
    _pcdh, 
    marg_idx1, marg_idx2,
    gridx_low, gridx_high, gridx_resolution, 
    gridy_low, gridy_high, gridy_resolution, 
    log_dir, 
    reset_cache,
    &out_cpdf_data, &size_out_cpdf_data, &out_num_gridx, &out_num_gridy
    );

    const mwSize dims[] = {3, mwSize(out_num_gridy), mwSize(out_num_gridx)};

    plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
    memcpy(mxGetPr(plhs[0]), out_cpdf_data, size_out_cpdf_data * sizeof(double));
    free(out_cpdf_data);

    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int* out_num_gridx_ptr = (int*)mxGetData(plhs[1]);
    *out_num_gridx_ptr = out_num_gridx;

    plhs[2] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int* out_num_gridy_ptr = (int*)mxGetData(plhs[2]);
    *out_num_gridy_ptr = out_num_gridy;

    mxFree(log_dir);
}