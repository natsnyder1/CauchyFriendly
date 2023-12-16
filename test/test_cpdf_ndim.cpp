#include "../include/cpdf_ndim.hpp"
#include "../include/cpdf_2d.hpp"


void test_2d_cpdf()
{
    const int n = 2;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {0.9, 0.1, -0.2, 1.1};
    double Gamma[n*pncc] = {.1, 0.3};
    double H[p*n] = {1.0, 0.5};
    double beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double A0[n*n] = {1,0,0,1}; // Unit directions of the initial state uncertainty
    double p0[n] = {.10, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double b0[n] = {0,0}; // Initial median of system state
    const int steps = 9;
    bool print_basic_info = true;
    char* log_dir = NULL;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWise2DCauchyCPDF cpdf_2d(log_dir, -1, 1, 0.25, -1, 1, 0.25);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    double zs[steps] = {0.022356919463887182, -0.22675889756491788, 0.42133397996398181, 
                        -1.7507202433585822, -1.3984154994099112, -1.7541436172809546, 
                        -1.8796017689052031, -1.9279807448991575, -1.9071129520752277}; 
    for(int i = 0; i < steps-2; i++) // Just to make sure that if last step is set to SKIP, we dont run...8 steps good enough 
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        cpdf_2d.evaluate_point_wise_cpdf(&cauchyEst);
        for(uint j = 0; j < cpdf_2d.num_gridx * cpdf_2d.num_gridy; j++)
        {    
            printf("\n2D: x=%.2lf, y=%.2lf, fx=%.9lf\n", cpdf_2d.cpdf_points[j].x, cpdf_2d.cpdf_points[j].y, cpdf_2d.cpdf_points[j].z);
            double xk[2] = {cpdf_2d.cpdf_points[j].x, cpdf_2d.cpdf_points[j].y};
            bool with_time_print = (i == (steps-3) ) && (j==(cpdf_2d.num_gridx * cpdf_2d.num_gridy-1));
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_cpdf(xk, with_time_print);
            printf("ND: x=%.2lf, y=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", cpdf_2d.cpdf_points[j].x, cpdf_2d.cpdf_points[j].y, creal(fx), cimag(fx));
        }
    }
}

void test_3d_cpdf()
{
    const int n = 3;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double Gamma[n*pncc] = {.1, 0.3, -0.2};
    double H[n] = {1.0, 0.5, 0.2};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; //{-0.63335359, -0.74816241, -0.19777826, -0.7710082 ,  0.63199184,  0.07831134, -0.06640465, -0.20208744,  0.97711365}; 
    double p0[n] = {0.10, 0.08, 0.05}; //{0.0, 0.0, 0.0}; //
    double b0[n] = {0, 0, 0};
    const int steps = 8;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    double zs[steps] = {0.022172011200334241, -0.05943271347277583, -1.12353301003957098, -1.4055389648301792, 
    -0.9048243525901404, -1.34053610027255954, -2.0580483915838776, -0.55152999529515989};

    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    const int grid_points = 7;
    double grid3D[grid_points][3] = 
    {
        {0,0,0},
        {-0.05,0,0},
        {0.05,0,0},
        {0,-0.05,0},
        {0, 0.05,0},
        {0,0,-0.05},
        {0,0,0.05}
    };


    for(int i = 0; i < steps-1; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        for(uint j = 0; j < grid_points; j++)
        {    
            double* xk = grid3D[j];
            bool with_time_print = (i == (steps-2) ) && (j==(grid_points-1));
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_cpdf(xk, with_time_print);
            printf("ND: x=%.2lf, y=%.2lf, z=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk[0], xk[1], xk[2], creal(fx), cimag(fx));
        }
    }
}

void test_4d_cpdf()
{
    const int n = 4;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1; 
    double Phi[n*n] = {1.4, -0.6, -1.0, 0.0,  
    -0.2,  1.0,  0.5, 0.0,  
    0.6, -0.6, -0.2, 0.0, 
    0, 0, 0, 0.5};
    double Gamma[n*pncc] = {.1, 0.3, -0.2, 0.4};
    double H[n] = {2.0, 0.5,  0.2, -0.1};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] = {1, 0, 0, 0, 
    0, 1, 0, 0, 
    0, 0, 1, 0, 
    0, 0, 0, 1};
    double p0[n] = {0.1, 0.08, 0.05, 0.2};
    double b0[n] = {0, 0, 0, 0};
    const int steps = 6;
    bool print_basic_info = true;
    double zs[steps] = {-0.26300165310514712, -0.98289343232730964, -0.93317363235517392, -0.81311530427193779, 
            -0.24140673945883995, 0.013971096637110103}; //, -3.4842328985975715, -3.1607056967588112};
    const int grid_points = 9;
    double grid4D[grid_points][4] = 
        {
            {0,0,0,0},
            {-0.05,0,0,0},
            {0.05,0,0,0},
            {0,-0.05,0,0},
            {0, 0.05,0,0},
            {0,0,-0.05,0},
            {0,0,0.05,0},
            {0,0,0,0.05},
            {0,0,0,-0.05},
        };
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    // Runs estimator step by step
    for(int i = 0; i < steps-1; i++)
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        for(uint j = 0; j < grid_points; j++)
        {    
            double* xk = grid4D[j];
            bool with_time_print = (i == (steps-2) ) && (j==(grid_points-1));
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_cpdf(xk, with_time_print);
            printf("ND: x1=%.2lf, x2=%.2lf, x3=%.2lf, x4=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk[0], xk[1], xk[2], xk[3], creal(fx), cimag(fx));
        }
    }

}

void test_2d_marginal_cpdf()
{
    const int n = 2;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {0.9, 0.1, -0.2, 1.1};
    double Gamma[n*pncc] = {.1, 0.3};
    double H[p*n] = {1.0, 0.5};
    double beta[pncc] = {0.1}; // Cauchy process noise scaling parameter(s)
    double gamma[p] = {0.2}; // Cauchy measurement noise scaling parameter(s)
    double A0[n*n] = {1,0,0,1}; // Unit directions of the initial state uncertainty
    double p0[n] = {.10, 0.05}; // Initial state uncertainty cauchy scaling parameter(s)
    double b0[n] = {0,0}; // Initial median of system state
    const int steps = 9;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    double zs[steps] = {0.022356919463887182, -0.22675889756491788, 0.42133397996398181, 
                        -1.7507202433585822, -1.3984154994099112, -1.7541436172809546, 
                        -1.8796017689052031, -1.9279807448991575, -1.9071129520752277}; 

    const int num_1d_grid_points = 21;
    double grid1D[num_1d_grid_points] = {
        -2,-1.8,-1.6,-1.4,-1.2,
        -1,-0.8,-0.6,-0.4,-0.2,
        0,
        0.2,0.4,0.6,0.8,1.0,
        1.2,1.4,1.6,1.8,2.0};
    const int num_marg_state_idxs = 1;
    int marg_state_idxs[num_marg_state_idxs] = {0};

    for(int i = 0; i < steps-2; i++) // Just to make sure that if last step is set to SKIP, we dont run...8 steps good enough 
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        for(uint j = 0; j < num_1d_grid_points; j++)
        {
            double xk_marginal[1] = {grid1D[j]};
            bool with_time_print = (i==(steps-3) && (j==(num_1d_grid_points-1)));
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print);
            printf("ND: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx), cimag(fx));
        }
    }
}

void test_3d_marginal_cpdf()
{
    const int n = 3;
    const int cmcc = 0;
    const int pncc = 1;
    const int p = 1;
    double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
    double Gamma[n*pncc] = {.1, 0.3, -0.2};
    double H[n] = {1.0, 0.5, 0.2};
    double beta[pncc] = {0.1};
    double gamma[p] = {0.2};
    double A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; //{-0.63335359, -0.74816241, -0.19777826, -0.7710082 ,  0.63199184,  0.07831134, -0.06640465, -0.20208744,  0.97711365}; 
    double p0[n] = {0.10, 0.08, 0.05}; //{0.0, 0.0, 0.0}; //
    double b0[n] = {0, 0, 0};
    const int steps = 8;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
    double zs[steps] = {0.022172011200334241, -0.05943271347277583, -1.12353301003957098, -1.4055389648301792, 
    -0.9048243525901404, -1.34053610027255954, -2.0580483915838776, -0.55152999529515989};

    PointWiseNDimCauchyCPDF cpdf_ndim(&cauchyEst);
    const int num_1d_grid_points = 11;
    double grid1D[num_1d_grid_points] = {-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0};
    const int num_marg_state_idxs = 1;
    int marg_state_idxs[num_marg_state_idxs] = {0};


    for(int i = 0; i < steps-1; i++) // To make sure we dont seg fault (at this point) if skip last step is turned on
    {
        cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
        for(uint j = 0; j < num_1d_grid_points; j++)
        {    
            double xk_marginal[1] = {grid1D[j]};
            bool with_time_print = (i == (steps-2) ) && (j==(num_1d_grid_points-1));
            C_COMPLEX_TYPE fx = cpdf_ndim.evaluate_marginal_cpdf(xk_marginal, marg_state_idxs, num_marg_state_idxs, with_time_print);
            printf("ND: x=%.2lf, fx=%.9lf, imag(fx)=%.9lf\n", xk_marginal[0], creal(fx), cimag(fx));
        }
    }
}


int main()
{
    //test_2d_cpdf(); // Is verified to be working (has been compared to the explicit 2D CPDF code)
    //test_3d_cpdf(); // Is (possibly) verified to be working (real and positive values returned...no errors)
    //test_4d_cpdf(); // Is (possibly) verified to be working (real and positive values returned...no errors)
    test_2d_marginal_cpdf();
    //test_3d_marginal_cpdf();
    return 0;
}