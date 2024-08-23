#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:validation:nrhs",
                          "One input argument required.");
    }

    uint64_t ptr = *((uint64_t*)mxGetData(prhs[0]));
    void (*free_names)() = (void (*)())ptr;

    free_names();
}