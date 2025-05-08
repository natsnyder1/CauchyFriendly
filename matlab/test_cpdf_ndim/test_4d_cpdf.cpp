#include "mex.h"
#include <vector>
#include "../../include/cpdf_ndim.hpp"
#include "../../include/cpdf_2d.hpp"

// CauchyEstimator, PointWiseNDimCauchyCPDF, and C_COMPLEX_TYPE are defined in included headers

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 10) {
        mexErrMsgTxt("Expected 10 input arguments: Phi, Gamma, H, beta, gamma, A0, p0, b0, zs, grid4D");
    }

    const int n = 4;
    const int steps = 6;
    const int grid_points = 9;

    double* Phi = mxGetPr(prhs[0]);
    double* Gamma = mxGetPr(prhs[1]);
    double* H = mxGetPr(prhs[2]);
    double* beta = mxGetPr(prhs[3]);
    double gamma = mxGetScalar(prhs[4]);
    double* A0 = mxGetPr(prhs[5]);
    double* p0 = mxGetPr(prhs[6]);
    double* b0 = mxGetPr(prhs[7]);
    double* zs = mxGetPr(prhs[8]);
    double* grid4D = mxGetPr(prhs[9]);

    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;

    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, false);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);

    std::vector<double> real_fx;
    std::vector<double> imag_fx;

    for (int i = 0; i < steps - 1; i++) {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma, NULL, NULL);
        for (int j = 0; j < grid_points; j++) {
            double xk[4];
            for (int d = 0; d < n; d++) {
                xk[d] = grid4D[d * grid_points + j];
            }
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_cpdf(xk, false);
            real_fx.push_back(creal(fx));
            imag_fx.push_back(cimag(fx));
        }
    }

    size_t len = real_fx.size();
    plhs[0] = mxCreateDoubleMatrix(len, 1, mxREAL);
    plhs[1] = mxCreateDoubleMatrix(len, 1, mxREAL);

    std::copy(real_fx.begin(), real_fx.end(), mxGetPr(plhs[0]));
    std::copy(imag_fx.begin(), imag_fx.end(), mxGetPr(plhs[1]));
}