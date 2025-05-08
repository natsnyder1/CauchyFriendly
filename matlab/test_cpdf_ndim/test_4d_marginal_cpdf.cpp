#include "mex.h"
#include <vector>
#include<complex>
#include "../../include/cpdf_ndim.hpp"
#include "../../include/cpdf_2d.hpp"

void run_marginal_cpdf(
    const std::vector<double>& zs,
    const std::vector<double>& grid2D,  // column-major: size = [2 x grid_points]
    int grid_points,
    std::vector<int>& marg_state_idxs,
    std::vector<double>& real_out
) {
    const int n = 4, cmcc = 0, pncc = 1, p = 1;
    const int steps = zs.size();

    double Phi[n * n] = {1.4, -0.6, -1.0, 0.0,
                        -0.2, 1.0, 0.5, 0.0,
                         0.6, -0.6, -0.2, 0.0,
                         0, 0, 0, 0.5};
    double Gamma[n * pncc] = {0.1, 0.3, -0.2, 0.4};
    double H[n] = {2.0, 0.5, 0.2, -0.1};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n * n] = {1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1};
    double p0[n] = {0.1, 0.08, 0.05, 0.2};
    double b0[n] = {0, 0, 0, 0};

    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, false);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);

    for (int t = 0; t < steps - 1; t++) {
        cauchyEst.step(zs[t], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        for (int i = 0; i < grid_points; i++) {
            double xk[2] = {
                grid2D[i * 2 + 0],  // row 0, column i
                grid2D[i * 2 + 1]   // row 1, column i
            };
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_2D_marginal_cpdf(xk, marg_state_idxs.data(), 2, false, true);
            real_out.push_back(creal(fx));
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 3) {
        mexErrMsgTxt("Usage: [real_fx] = test_4d_marginal(zs, grid2D, marg_state_idxs)");
    }

    // Input: zs
    double* zs_ptr = mxGetPr(prhs[0]);
    size_t steps = mxGetNumberOfElements(prhs[0]);
    std::vector<double> zs(zs_ptr, zs_ptr + steps);

    // Input: grid2D (2 x N)
    double* grid_ptr = mxGetPr(prhs[1]);
    size_t rows = mxGetM(prhs[1]);
    size_t cols = mxGetN(prhs[1]);
    if (rows != 2) mexErrMsgTxt("grid2D must be a 2xN matrix.");
    std::vector<double> grid2D(grid_ptr, grid_ptr + 2 * cols);

    // Input: marg_state_idxs (2-element integer vector)
    double* marg_ptr = mxGetPr(prhs[2]);
    std::vector<int> marg_state_idxs = { static_cast<int>(marg_ptr[0]), static_cast<int>(marg_ptr[1]) };

    // Output: real part of CPDF
    std::vector<double> real_fx;
    run_marginal_cpdf(zs, grid2D, cols, marg_state_idxs, real_fx);

    plhs[0] = mxCreateDoubleMatrix(cols, steps - 1, mxREAL);
    double* out_ptr = mxGetPr(plhs[0]);
    std::copy(real_fx.begin(), real_fx.end(), out_ptr);
}
