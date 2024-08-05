#include "mex.h"
#include "pycauchy.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("MATLAB:validation:nrhs",
                          "Three input arguments required.");
    }

    // Extract the pointer from the first input argument
    if (!mxIsUint64(prhs[0]) || mxGetNumberOfElements(prhs[0]) != 1) {
        mexErrMsgIdAndTxt("MATLAB:validation:invalidPointer",
                          "First input must be a valid uint64 pointer.");
    }

    uint64_t pointerValue = *((uint64_t*)mxGetData(prhs[0]));
    void* _pcdh = (void*)pointerValue;

    double *msmts = mxGetPr(prhs[1]);
    double *controls = mxGetPr(prhs[2]);
    bool with_propagate = static_cast<bool>(mxGetScalar(prhs[3]));

    // Define output variables
    double *out_Phi = nullptr;
    double *out_Gamma = nullptr;
    double *out_B = nullptr;
    double *out_H = nullptr;
    double *out_beta = nullptr;
    double *out_gamma = nullptr;
    double *out_fz = nullptr;
    double *out_xhat = nullptr;
    double *out_Phat = nullptr;
    double *out_xbar = nullptr;
    double *out_zbar = nullptr;
    double *out_cerr_fz = nullptr;
    double *out_cerr_xhat = nullptr;
    double *out_cerr_Phat = nullptr;
    int *out_err_code = nullptr;

    int size_out_Phi, size_out_Gamma, size_out_B, size_out_H;
    int size_out_beta, size_out_gamma, size_out_fz, size_out_xhat;
    int size_out_Phat, size_out_xbar, size_out_zbar;
    int size_out_cerr_fz, size_out_cerr_xhat;
    int size_out_cerr_Phat, size_out_err_code;

    pycauchy_single_step_nonlin(
        _pcdh,
        msmts, mxGetNumberOfElements(prhs[1]),   
        controls, mxGetNumberOfElements(prhs[2]), 
        with_propagate,
        &out_Phi, &size_out_Phi,
        &out_Gamma, &size_out_Gamma,
        &out_B, &size_out_B,
        &out_H, &size_out_H,
        &out_beta, &size_out_beta,
        &out_gamma, &size_out_gamma,
        &out_fz, &size_out_fz,
        &out_xhat, &size_out_xhat,
        &out_Phat, &size_out_Phat,
        &out_xbar, &size_out_xbar,
        &out_zbar, &size_out_zbar,
        &out_cerr_fz, &size_out_cerr_fz,
        &out_cerr_xhat, &size_out_cerr_xhat,
        &out_cerr_Phat, &size_out_cerr_Phat,
        &out_err_code, &size_out_err_code
    );


    // // Create mxArrays for outputs and assign them to plhs[]
    plhs[0] = mxCreateDoubleMatrix(size_out_Phi, 1, mxREAL);
    memcpy(mxGetPr(plhs[0]), out_Phi, size_out_Phi * sizeof(double));
    free(out_Phi);

    plhs[1] = mxCreateDoubleMatrix(size_out_Gamma, 1, mxREAL);
    memcpy(mxGetPr(plhs[1]), out_Gamma, size_out_Gamma * sizeof(double));
    free(out_Gamma);

    plhs[2] = mxCreateDoubleMatrix(size_out_B, 1, mxREAL);
    memcpy(mxGetPr(plhs[2]), out_B, size_out_B * sizeof(double));
    free(out_B);

    plhs[3] = mxCreateDoubleMatrix(size_out_H, 1, mxREAL);
    memcpy(mxGetPr(plhs[3]), out_H, size_out_H * sizeof(double));
    free(out_H);

    plhs[4] = mxCreateDoubleMatrix(size_out_beta, 1, mxREAL);
    memcpy(mxGetPr(plhs[4]), out_beta, size_out_beta * sizeof(double));
    free(out_beta);

    plhs[5] = mxCreateDoubleMatrix(size_out_gamma, 1, mxREAL);
    memcpy(mxGetPr(plhs[5]), out_gamma, size_out_gamma * sizeof(double));
    free(out_gamma);

    plhs[6] = mxCreateDoubleMatrix(size_out_fz, 1, mxREAL);
    memcpy(mxGetPr(plhs[6]), out_fz, size_out_fz * sizeof(double));
    free(out_fz);

    plhs[7] = mxCreateDoubleMatrix(size_out_xhat, 1, mxREAL);
    memcpy(mxGetPr(plhs[7]), out_xhat, size_out_xhat * sizeof(double));
    free(out_xhat);

    plhs[8] = mxCreateDoubleMatrix(size_out_Phat, 1, mxREAL);
    memcpy(mxGetPr(plhs[8]), out_Phat, size_out_Phat * sizeof(double));
    free(out_Phat);

    plhs[9] = mxCreateDoubleMatrix(size_out_xbar, 1, mxREAL);
    memcpy(mxGetPr(plhs[9]), out_xbar, size_out_xbar * sizeof(double));
    free(out_xbar);

    plhs[10] = mxCreateDoubleMatrix(size_out_zbar, 1, mxREAL);
    memcpy(mxGetPr(plhs[10]), out_zbar, size_out_zbar * sizeof(double));
    free(out_zbar);

    plhs[11] = mxCreateDoubleMatrix(size_out_cerr_fz, 1, mxREAL);
    memcpy(mxGetPr(plhs[11]), out_cerr_fz, size_out_cerr_fz * sizeof(double));
    free(out_cerr_fz);

    plhs[12] = mxCreateDoubleMatrix(size_out_cerr_xhat, 1, mxREAL);
    memcpy(mxGetPr(plhs[12]), out_cerr_xhat, size_out_cerr_xhat * sizeof(double));
    free(out_cerr_xhat);

    plhs[13] = mxCreateDoubleMatrix(size_out_cerr_Phat, 1, mxREAL);
    memcpy(mxGetPr(plhs[13]), out_cerr_Phat, size_out_cerr_Phat * sizeof(double));
    free(out_cerr_Phat);

    plhs[14] = mxCreateNumericMatrix(size_out_err_code, 1, mxINT32_CLASS, mxREAL);
    memcpy(mxGetData(plhs[14]), out_err_code, size_out_err_code * sizeof(int));
    free(out_err_code);
}
