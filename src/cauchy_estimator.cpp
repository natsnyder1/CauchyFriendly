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
  const int steps = 8; //8; //12;
  bool print_basic_info = false;
  CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, p, print_basic_info);
  double zs[steps] = {0.056658570969158, -0.142753984516164, -1.205346279653962, 1.378810275944929, -0.531562515031748, -1.019512573307872, -0.597247860048719, -0.710687942924685}; //, -0.325576, -0.05943287, 0.65387723, 0.438622377};
  for(int j = 0; j < 10; j++)
  {
    for(int i = 0; i < steps; i++)
    {
      cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0]);
    }
    cauchyEst.reset();
  }
}

int main()
{
    printf("Size of Cauchy Term is %lu\n", sizeof(CauchyTerm));
    printf("Size of Cauchy Estimator is %lu\n", sizeof(CauchyEstimator));
    test_cauchy_3_state_moshe();
    return 0;
}
