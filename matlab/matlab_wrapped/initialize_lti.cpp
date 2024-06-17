#include "mex.h"
#include "pycauchy.hpp"
#include <cmath>

void checkSquareAndFullRank(const mxArray* matrix, const char* matrixName) {
    mwSize rows = mxGetM(matrix);
    mwSize cols = mxGetN(matrix);
    if (rows != cols) {
        mexErrMsgIdAndTxt("MATLAB:validation:square",
                          "%s must be a square matrix.", matrixName);
    }

    // Call MATLAB's rank function to check for full rank
    mxArray *lhs[1], *rhs[1];
    rhs[0] = const_cast<mxArray*>(matrix);
    mexCallMATLAB(1, lhs, 1, rhs, "rank");
    double rank = mxGetScalar(lhs[0]);
    mxDestroyArray(lhs[0]);

    if (rank != rows) {
        mexErrMsgIdAndTxt("MATLAB:validation:notFullRank",
                          "%s is not full rank.", matrixName);
    }
}


void checkVectorSize(const mxArray* vector, mwSize expectedSize, const char* vectorName) {
    if (!mxIsDouble(vector) || mxIsComplex(vector) || mxGetN(vector) > 1) {
        mexErrMsgIdAndTxt("MATLAB:validation:vector",
                          "%s must be a real double vector.", vectorName);
    }
    if (mxGetM(vector) != expectedSize) {
        mexErrMsgIdAndTxt("MATLAB:validation:size",
                          "%s must have a size of %d.", vectorName, expectedSize);
    }
}

void checkNonNegativity(const mxArray* vector, const char* vectorName) {
    mwSize numElements = mxGetNumberOfElements(vector);
    double *data = mxGetPr(vector);
    for (mwSize i = 0; i < numElements; ++i) {
        if (data[i] < 0) {
            mexErrMsgIdAndTxt("MATLAB:validation:nonNegative",
                              "%s must contain only non-negative values.", vectorName);
        }
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 9) {
        mexErrMsgIdAndTxt("MATLAB:validation:nrhs",
                          "Nine input arguments required.");
    }
//  initialize_lti(A0_correct, p0_correct, b0_correct, Phi_correct, B_correct, Gamma_correct, beta_correct, H_correct, gamma_correct);
    const mxArray *A0 = prhs[0];
    const mxArray *p0 = prhs[1];
    const mxArray *b0 = prhs[2];
    const mxArray *Phi = prhs[3];
    const mxArray *B = prhs[4];
    const mxArray *Gamma = prhs[5];
    const mxArray *beta = prhs[6];
    const mxArray *H = prhs[7];
    const mxArray *gamma = prhs[8];

    // _init_params_checker logic
    //checkSquareAndFullRank(A0, "A0");
    //checkVectorSize(p0, mxGetM(A0), "p0");
    //checkVectorSize(b0, mxGetM(A0), "b0");
    //checkNonNegativity(p0, "p0");

    // _ndim_input_checker logic
    mwSize n = mxGetM(A0);
    // checkSquareAndFullRank(Phi, "Phi");
    // if (mxGetM(Gamma) != n) {
    //     mexErrMsgIdAndTxt("MATLAB:validation:dimensionMismatch",
    //                       "Gamma must have %d rows.", n);
    // }
    // 
    // // Checks for vector beta
    // if (mxGetN(Gamma) == 1) {  // Gamma is a column vector
    //     if (mxGetNumberOfElements(beta) != 1) {
    //         mexErrMsgIdAndTxt("MATLAB:beta:wrongSize",
    //                           "Beta must have exactly one element when Gamma is a column vector.");
    //     }
    // } 
    // else {
    //     if (mxGetM(beta) != mxGetN(Gamma)) {
    //         mexErrMsgIdAndTxt("MATLAB:beta:dimensionMismatch",
    //                           "Beta must have as many rows as Gamma has columns.");
    //     }
    // }
    // 
    // if (mxGetNumberOfElements(B) > 0 && mxGetM(B) != n) {
    //     mexErrMsgIdAndTxt("MATLAB:validation:dimensionMismatch",
    //                       "B must have %d rows.", n);
    // }
    // 
    // if (mxGetN(H) != n) {
    //     mexErrMsgIdAndTxt("MATLAB:validation:dimensionMismatch",
    //                       "H must have %d columns.", n);
    // }
    // 
    // if (mxGetN(gamma) != 1) {
    //     mexErrMsgIdAndTxt("MATLAB:validation:notColumnVector",
    //                       "Gamma must be a column vector.");
    // }

    double *data_A0 = mxGetPr(prhs[0]);
    double *data_p0 = mxGetPr(prhs[1]);
    double *data_b0 = mxGetPr(prhs[2]);
    double *data_Phi = mxGetPr(prhs[3]);
    double *data_B = mxGetPr(prhs[4]);
    double *data_Gamma = mxGetPr(prhs[5]); // Assuming Gamma can never be empty by this point
    double *data_beta = mxGetPr(prhs[6]);  // Assuming beta can never be empty by this point
    double *data_H = mxGetPr(prhs[7]);
    double *data_gamma = mxGetPr(prhs[8]);

    // Assume some integers for the additional parameters required by pycauchy_initialize_lti
    int num_steps = 3;   // Example value, replace with actual if needed
    int init_step = 0; // defaults to 0
    double dt = 0;      // Time step, example value
    bool debug_print = true;  // Debugging flag

    // Call the function from the pycauchy library
    void* result = pycauchy_initialize_lti(
        num_steps,
        data_A0, mxGetNumberOfElements(prhs[0]),
        data_p0, mxGetNumberOfElements(prhs[1]),
        data_b0, mxGetNumberOfElements(prhs[2]),
        data_Phi, mxGetNumberOfElements(prhs[3]),
        data_Gamma, mxGetNumberOfElements(prhs[5]), // SWITCH NUMBERING TODO
        data_B, mxGetNumberOfElements(prhs[4]),
        data_beta, mxGetNumberOfElements(prhs[6]),
        data_H, mxGetNumberOfElements(prhs[7]),
        data_gamma, mxGetNumberOfElements(prhs[8]),
        dt,
        init_step,
        debug_print
    );

    // Return the pointer as a MATLAB uint64
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*)mxGetData(plhs[0])) = (uint64_t)result;
}
