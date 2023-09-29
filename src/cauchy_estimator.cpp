#include "../include/cauchy_estimator.hpp"


// Moshes Three State Problem
void test_cauchy_3_state_moshe()
{
  const int n = 3;
  const int cmcc = 1;
  const int p = 1;
  double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
  double Gamma[n*cmcc] = {.1, 0.3, -0.2};
  double H[n] = {1.0, 0.5, 0.2};
  double beta[cmcc] = {0.1};
  double gamma[p] = {0.2};
  double A0[n*n] =  {1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0}; //{-0.63335359, -0.74816241, -0.19777826, -0.7710082 ,  0.63199184,  0.07831134, -0.06640465, -0.20208744,  0.97711365}; 
  double p0[n] = {0.10, 0.08, 0.05}; //{0.0, 0.0, 0.0}; //
  double b0[n] = {0, 0, 0};
  const int steps = 8;
  bool print_basic_info = true;
  CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, p, print_basic_info);
  double zs[steps] = {-1.2172011200334241, -0.35943271347277583, -0.52353301003957098, 0.5855389648301792, 
  -0.8048243525901404, 0.34053610027255954, 1.0580483915838776, -0.55152999529515989};
  //-0.72879029737003309, -0.82415138330170357, -0.63794753995479381, -0.50437372151915394};
  for(int j = 0; j < 1; j++)
  {
    for(int i = 0; i < steps; i++)
    {
      cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0]);
    }
    cauchyEst.reset();
  }
}

// Moshes Three State Problem
void test_cauchy_4_state_moshe()
{
  const int n = 4;
  const int cmcc = 1;
  const int p = 1; 
  double Phi[n*n] = {1.4, -0.6, -1.0, 0.0,  
    -0.2,  1.0,  0.5, 0.0,  
    0.6, -0.6, -0.2, 0.0, 
    0, 0, 0, 0.5};
  double Gamma[n*cmcc] = {.1, 0.3, -0.2, 0.4};
  double H[n] = {2.0, 0.5,  0.2, -0.1};
  double beta[cmcc] = {0.1};
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

  CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, p, print_basic_info);

  // Runs estimator step by step
  for(int j = 0; j < 1; j++)
  {
    for(int i = 0; i < steps; i++)
    {
      cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0]);
    }
    cauchyEst.reset();
  }

}

void test_cauchy_3_state_moshe_3msmts()
{
  const int n = 3;
  const int cmcc = 1; 
  const int p = 3;
  double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
  double Gamma[n*cmcc] = {.1, 0.3, -0.2};
  double H[n*p] = {1,0,0, 0,1,0, 0,0,1}; //{1.0, 0.5, 0.2, 0.2, 0.5, 1.0, -0.54, 0.2, 0.33};
  double beta[cmcc] = {0.1};
  double gamma[p] = {0.2, 0.15, 0.10};
  double A0[n*n] = {1.0, 0, 0, 
                        0, 1.0, 0, 
                        0, 0, 1.0};
  double p0[n] = {0.1, 0.08, 0.05};
  double b0[n] = {0, 0, 0};
  const int steps = 5;
  bool print_basic_info = true;
  CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, p, print_basic_info);
  // zs for reg conditions
  double zs[n*p] = {};
  
  // zs for horth conditions
  /*
  double zs[n*p] = {};
  */
  // Runs estimator step by step
  for(int j = 0; j < 1; j++)
  {
    for(int i = 0; i < steps*p; i++)
    {
      cauchyEst.step(zs[i], Phi, Gamma, beta, H + (i%p)*n, gamma[i % p]);
    }
    cauchyEst.reset();
  }
}


int main()
{
    printf("Size of Cauchy Term is %lu\n", sizeof(CauchyTerm));
    printf("Size of Cauchy Estimator is %lu\n", sizeof(CauchyEstimator));
    //test_cauchy_3_state_moshe();
    test_cauchy_4_state_moshe();
    return 0;
}
