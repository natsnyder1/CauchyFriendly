#include "mex.h"
#include <vector>
#include <complex>
#include <cmath>
#include "../../include/cpdf_ndim.hpp"
#include "../../include/cpdf_2d.hpp"

// The main computation function
void run_marginal_cpdf(
    const std::vector<double>& zs,
    const std::vector<double>& grid2D,  // column-major: size = [2 x grid_points]
    int grid_points,
    const std::vector<int>& marg_state_idxs,
    const std::vector<double>& Phi,
    const std::vector<double>& Gamma,
    const std::vector<double>& H,
    const std::vector<double>& beta,
    double gamma,
    const std::vector<double>& A0,
    const std::vector<double>& p0,
    const std::vector<double>& b0,
    int n, int cmcc, int pncc, int p,
    std::vector<double>& real_out
) {
    const int steps = zs.size();

    CauchyEstimator cauchyEst(
        const_cast<double*>(A0.data()),
        const_cast<double*>(p0.data()),
        const_cast<double*>(b0.data()),
        steps, n, cmcc, pncc, p, false);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);

    for (int t = 0; t < steps - 1; t++) {
        cauchyEst.step(
            zs[t],
            const_cast<double*>(Phi.data()),
            const_cast<double*>(Gamma.data()),
            const_cast<double*>(beta.data()),
            const_cast<double*>(H.data()),
            gamma,
            nullptr,
            nullptr);

        for (int i = 0; i < grid_points; i++) {
            double xk[2] = {
                grid2D[i * 2 + 0],  // row 0, col i
                grid2D[i * 2 + 1]   // row 1, col i
            };
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_2D_marginal_cpdf(
                xk, const_cast<int*>(marg_state_idxs.data()), 2, false, true);
            real_out.push_back(creal(fx));
        }
    }
}

// The MATLAB MEX gateway
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 15) {
        mexErrMsgTxt("Usage: real_fx = test_4d_marginal_cpdf(zs, grid2D, marg_state_idxs, Phi, Gamma, H, beta, gamma, A0, p0, b0, n, cmcc, pncc, p)");
    }

    // Extract scalars first (inputs 11–14)
    int n     = static_cast<int>(mxGetScalar(prhs[11]));
    int cmcc  = static_cast<int>(mxGetScalar(prhs[12]));
    int pncc  = static_cast<int>(mxGetScalar(prhs[13]));
    int p     = static_cast<int>(mxGetScalar(prhs[14]));

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

    // Input: marg_state_idxs (2-element vector)
    double* marg_ptr = mxGetPr(prhs[2]);
    std::vector<int> marg_state_idxs = {
        static_cast<int>(marg_ptr[0]),
        static_cast<int>(marg_ptr[1])
    };

    // Matrix/Vector Inputs: Phi, Gamma, H, beta, gamma, A0, p0, b0
    std::vector<double> Phi(mxGetPr(prhs[3]), mxGetPr(prhs[3]) + n * n);
    std::vector<double> Gamma(mxGetPr(prhs[4]), mxGetPr(prhs[4]) + n * pncc);
    std::vector<double> H(mxGetPr(prhs[5]), mxGetPr(prhs[5]) + n);
    std::vector<double> beta(mxGetPr(prhs[6]), mxGetPr(prhs[6]) + pncc);
    double gamma = mxGetScalar(prhs[7]);
    std::vector<double> A0(mxGetPr(prhs[8]), mxGetPr(prhs[8]) + n * n);
    std::vector<double> p0(mxGetPr(prhs[9]), mxGetPr(prhs[9]) + n);
    std::vector<double> b0(mxGetPr(prhs[10]), mxGetPr(prhs[10]) + n);

    // Output: real part of CPDF
    std::vector<double> real_fx;
    run_marginal_cpdf(zs, grid2D, cols, marg_state_idxs,
                      Phi, Gamma, H, beta, gamma,
                      A0, p0, b0, n, cmcc, pncc, p, real_fx);

    // MATLAB output matrix: [points x (steps-1)]
    plhs[0] = mxCreateDoubleMatrix(cols, steps - 1, mxREAL);
    double* out_ptr = mxGetPr(plhs[0]);
    std::copy(real_fx.begin(), real_fx.end(), out_ptr);
}
