#include "mex.h"
#include <vector>
#include "../../include/cpdf_ndim.hpp"
#include "../../include/cpdf_2d.hpp"

// CauchyEstimator, PointWiseNDimCauchyCPDF, and C_COMPLEX_TYPE are defined in included headers

void run_test_4d_cpdf(std::vector<double>& real_out, std::vector<double>& imag_out) {
    const int n = 4;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    const int steps = 6;
    const int grid_points = 9;

    // Same as original function
    double Phi[n*n] = {1.4, -0.6, -1.0, 0.0, -0.2, 1.0, 0.5, 0.0, 0.6, -0.6, -0.2, 0.0, 0, 0, 0, 0.5};
    double Gamma[n*pncc] = {.1, 0.3, -0.2, 0.4};
    double H[n] = {2.0, 0.5, 0.2, -0.1};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    double p0[n] = {0.1, 0.08, 0.05, 0.2};
    double b0[n] = {0, 0, 0, 0};
    double zs[steps] = {-0.26300165310514712, -0.98289343232730964, -0.93317363235517392, -0.81311530427193779, 
            -0.24140673945883995, 0.013971096637110103};
    double grid4D[grid_points][4] = 
        {
            {0,0,0,0}, 
            {-0.05,0,0,0}, 
            {0.05,0,0,0}, 
            {0,-0.05,0,0}, 
            {0,0.05,0,0}, 
            {0,0,-0.05,0},
            {0,0,0.05,0}, 
            {0,0,0,0.05}, 
            {0,0,0,-0.05}
        };

    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, false);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);

    for (int i = 0; i < steps - 1; i++) {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        for (int j = 0; j < grid_points; j++) {
            double* xk = grid4D[j];
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_cpdf(xk, false);
            real_out.push_back(creal(fx));
            imag_out.push_back(cimag(fx));
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    std::vector<double> real_fx;
    std::vector<double> imag_fx;

    run_test_4d_cpdf(real_fx, imag_fx);

    size_t len = real_fx.size();
    plhs[0] = mxCreateDoubleMatrix(len, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(len, 1, mxREAL);

    double* real_ptr = mxGetPr(plhs[0]);
    double* imag_ptr = mxGetPr(plhs[1]);
    std::copy(real_fx.begin(), real_fx.end(), real_ptr);
    std::copy(imag_fx.begin(), imag_fx.end(), imag_ptr);
}
