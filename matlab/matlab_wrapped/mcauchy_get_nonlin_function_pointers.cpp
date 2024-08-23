#include "mex.h"
#include <iostream>
#include "dynamic_models.hpp"

char *name1;
char *name2;
char *name3;

void callFunction1(CauchyDynamicsUpdateContainer *cduc) {
    int nrhs = 1;
    int nlhs = 0;
    mxArray** prhs = (mxArray**)malloc(nrhs * sizeof(mxArray*));
    mxArray** plhs = (mxArray**)malloc(nlhs * sizeof(mxArray*));
    uint64_t uint_cduc = (uint64_t)cduc;
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    memcpy(mxGetPr(prhs[0]), &uint_cduc, sizeof(uint64_t));

    mexCallMATLAB(nlhs, plhs, nrhs, prhs, name1);

    free(prhs);
    free(plhs);
}

void callFunction2(CauchyDynamicsUpdateContainer *cduc, double* c_zbar) {
    int nrhs = 2;
    int nlhs = 0;
    mxArray** prhs = (mxArray**)malloc(nrhs * sizeof(mxArray*));
    mxArray** plhs = (mxArray**)malloc(nlhs * sizeof(mxArray*));
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    prhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
    uint64_t uint_cduc = (uint64_t)cduc;
    uint64_t uint_c_zbar = (uint64_t)c_zbar;
    memcpy(mxGetPr(prhs[0]), &uint_cduc, sizeof(uint64_t));
    memcpy(mxGetPr(prhs[1]), &uint_c_zbar, sizeof(uint64_t));
    mexCallMATLAB(nlhs, plhs, nrhs, prhs, name2);

    free(prhs);
    free(plhs);
}

void callFunction3(CauchyDynamicsUpdateContainer *cduc) {
    int nrhs = 1;
    int nlhs = 0;
    mxArray** prhs = (mxArray**)malloc(nrhs * sizeof(mxArray*));
    mxArray** plhs = (mxArray**)malloc(nlhs * sizeof(mxArray*));
    uint64_t uint_cduc = (uint64_t)cduc;
    prhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
    memcpy(mxGetPr(prhs[0]), &uint_cduc, sizeof(uint64_t));

    mexCallMATLAB(nlhs, plhs, nrhs, prhs, name3);

    free(prhs);
    free(plhs);
}


char* duplicate_string(const char* source) {
    if (source == nullptr) {
        return nullptr;
    }
    size_t length = strlen(source) + 1; 
    char* destination = (char*)malloc(length); 
    if (destination) {
        strcpy(destination, source);
    }
    return destination;
}

void free_names() {
    free(name1);
    name1 = NULL;
    free(name2);
    name2 = NULL;
    free(name2);
    name2 = NULL;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("mcauchy_get_nonlin_function_pointers:input", "Three inputs required.");
    }

    char* name1_temp = mxArrayToString(prhs[0]);
    char* name2_temp = mxArrayToString(prhs[1]);
    char* name3_temp = mxArrayToString(prhs[2]);

    name1 = duplicate_string(name1_temp);
    name2 = duplicate_string(name2_temp);
    name3 = duplicate_string(name3_temp);
    
    void (*funcPtr1)(CauchyDynamicsUpdateContainer*) = &callFunction1;
    void (*funcPtr2)(CauchyDynamicsUpdateContainer*, double*) = &callFunction2;
    void (*funcPtr3)(CauchyDynamicsUpdateContainer*) = &callFunction3;
    void (*free_names_ptr)() = &free_names;

    // printf("The address of funcPtr1 is: %p\n", funcPtr1);
    // printf("The address of funcPtr2 is: %p\n", funcPtr2);
    // printf("The address of funcPtr3 is: %p\n", funcPtr3);

    // Return the pointers as a MATLAB uint64's
    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*)mxGetData(plhs[0])) = (uint64_t)funcPtr1;

    plhs[1] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*)mxGetData(plhs[1])) = (uint64_t)funcPtr2;

    plhs[2] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*)mxGetData(plhs[2])) = (uint64_t)funcPtr3;

    plhs[3] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t*)mxGetData(plhs[3])) = (uint64_t)free_names_ptr;

    mxFree(name1_temp);
    mxFree(name2_temp);
    mxFree(name3_temp);

    std::cout << "All function pointers set" << std::endl;
}