#include "../include/cauchy_estimator.hpp"
#include "../include/cpdf_2d.hpp"

// Moshes Two State Problem
void test_cauchy_2_state_moshe()
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
  const int steps = 10;
  bool print_basic_info = true;
  char log_dir[10] = "foobar";
  CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
  PointWise2DCauchyCPDF cpdf(log_dir,-1.5, 1.5, 0.25, -3, 1, 0.25);
  double zs[steps] = {0.022356919463887182, -0.22675889756491788, 0.42133397996398181, 
                      -1.7507202433585822, -1.3984154994099112, -1.7541436172809546, -1.8796017689052031, 
                      -1.9279807448991575, -1.9071129520752277, -2.0343612017356922};
  for(int i = 0; i < steps; i++)
  {
    cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
    cpdf.evaluate_point_wise_cpdf(&cauchyEst, NUM_CPUS);
  }
}

// Moshes Three State Problem
void test_cauchy_3_state_moshe()
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
  double zs[steps] = {-1.2172011200334241, -0.35943271347277583, -0.52353301003957098, 0.5855389648301792, 
  -0.8048243525901404, 0.34053610027255954, 1.0580483915838776, -0.55152999529515989};
  //-0.72879029737003309, -0.82415138330170357}; //, -0.63794753995479381, -0.50437372151915394};
  for(int j = 0; j < 2; j++)
  {
    for(int i = 0; i < steps; i++)
    {
      cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
    }
    cauchyEst.reset();
  }
}

// Moshes Three State Problem
void test_cauchy_4_state_moshe()
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

  CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
  // Runs estimator step by step
  for(int j = 0; j < 1; j++)
  {
    for(int i = 0; i < steps; i++)
    {
      cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
    }
    cauchyEst.reset();
  }

}

void test_cauchy_3_state_moshe_3msmts()
{
  const int n = 3;
  const int cmcc = 0;
  const int pncc = 1; 
  const int p = 3;
  double Phi[n*n] = {1.4, -0.6, -1.0,  -0.2,  1.0,  0.5,  0.6, -0.6, -0.2};
  double Gamma[n*pncc] = {.1, 0.3, -0.2};
  //double H[n*p] = {1.0, 0.5, 0.2, 0.2, 0.5, 1.0, -0.54, 0.2, 0.33}; // reg
  double H[n*p] = {1,0,0, 0,1,0, 0,0,1}; // horth
  double beta[pncc] = {0.1};
  double gamma[p] = {0.2, 0.15, 0.10};
  double A0[n*n] = {1.0, 0, 0, 
                    0, 1.0, 0, 
                    0, 0, 1.0};
  double p0[n] = {0.1, 0.08, 0.05};
  double b0[n] = {0, 0, 0};
  const int steps = 5;
  bool print_basic_info = true;
  CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);
  // zs for reg conditions
  /*
  double zs[steps*p] = 
    {-0.035615041591695096, -0.073027082761234374, -0.1265331454364314, 
    0.61549398789878296, 0.23591755422092489, -0.34317367463390247, 
    0.37558191996841117, 0.37444478312568541, -1.3027453832861597, 
    2.6961335310150134, 0.54281681104100954, -0.18694512563047694, 
    0.53210372672050144, 1.623969984120111, 0.2132162054128228};
  */
  ///*
  // zs for horth conditions
  double zs[steps*p] = 
    {0.10943250903225685, 0.32131358116921616, -0.39352816664526724, 
    0.76258687662854907, -0.25344840215960657, 0.1578820974338809, 
    0.52543601367678883, -0.67309502187832315, -0.37267005411252474, 
    2.7335350536863903, -0.3754139600950176, 0.6986657326616188, 
    0.52558307773279223, 0.82802377147093642, 0.98211422248186553};
  //*/
  // Runs estimator step by step
  for(int j = 0; j < 2; j++)
  {
    for(int i = 0; i < steps*p; i++)
    {
      cauchyEst.step(zs[i], Phi, Gamma, beta, H + (i%p)*n, gamma[i % p], NULL, NULL);
    }
    cauchyEst.reset();
  }
}

// Moshes Five State Problem
void test_cauchy_5_state_moshe()
{
  /* initialize random seed: */
  //srand ( time(NULL) );
  // PREAMBLE
  const int n = 5;
  const int cmcc = 0;
  const int pncc = 1;
  const int p = 1;
  double Phi[n*n] = {0.9 , -0.6 , -0.5 ,  0.5 ,  0.5 ,  
                         0.05,  1.  ,  0.25, -0.25, -0.25,  
                         0.25, -0.6 ,  0.15,  0.55,  0.35,  
                         0.1 ,  0.  , -0.1 , 0.4 , -0.1 ,  
                         0.05,  0.  , -0.05,  0.45,  0.75};
  double Gamma[n*pncc] = {0.1, 0.3, -0.2, 0.4, -0.15};
  double H[n] = {2.0, 0.5,  0.2, -0.1, 0.4};
  double beta[pncc] = {0.1};
  double gamma[p] = {0.2};
  double A0[n*n] =  {1, 0, 0, 0, 0,
                         0, 1, 0, 0, 0, 
                         0, 0, 1, 0, 0,
                         0, 0, 0, 1, 0,
                         0, 0, 0, 0, 1};
  double p0[n] = {0.1, 0.08, 0.05, 0.2, 0.3}; //{0.0, 0.0, 0.0}; //
  double b0[n] = {0, 0, 0, 0, 0};
  const int steps = 9;
  bool print_basic_info = true;
  CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);

  double zs[steps] = {-0.23451108940494397, -1.6969544453434717, -1.2059957432220769, 
                      -2.7147437491778885, -1.3823660959009454, -0.29548720069792656, 
                      -1.4872516450002977, -1.5538990428366166, -1.7101257926043105};

  for(int j = 0; j < 1; j++)
  {
    for(int i = 0; i < steps; i++)
    {
      cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
    }
    cauchyEst.reset();
  }
  
}

void test_cauchy_four_state_two_pnoise()
{
  const int n = 4;
  const int cmcc = 0;
  const int pncc = 2;
  const int p = 1;

  double Phi[n*n] = {1.4, -0.6, -1.0, 0.0,  
                        -0.2, 1.0, 0.5, 0.0,  
                        0.6, -0.6, -0.2, 0.0,  
                        0.0, 0.0, 0.0, 1.0};
  double Gamma[n*pncc] = {0.1,0.0,  
                          0.3,0.0,  
                          -0.2, 0.0,  
                          0.0, 1.0};
  double H[n] = {0.4165285461783826, -0.60, -1.0, 1.0};
  double beta[pncc] = {0.1, 0.001};
  double gamma[p] = {0.2};
  double A0[n*n] = {1,0,0,0,  
                    0,1,0,0,  
                    0,0,1,0,  
                    0,0,0,1}; 
  double p0[n] = {0.4, 0.5, 0.6, 0.70};
  double b0[n] = {0.0, 0.0, 0.0, 0.0};

  const int steps = 5;
  bool print_basic_info = true;
  CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_basic_info);

  double zs[steps] = {-5.3335189550166655, -4.4110988021211845, -3.6610012492599329, 
                      -2.5741683288219699, -6.5109959475268671};

  // Runs estimator step by step
  for(int j = 0; j < 1; j++)
  {
    for(int i = 0; i < steps; i++)
    {
      cauchyEst.step(zs[i], Phi, Gamma, beta, H, gamma[0], NULL, NULL);
    }
    cauchyEst.reset();
  }

}


int main()
{
    printf("Size of Cauchy Term is %lu\n", sizeof(CauchyTerm));
    printf("Size of Cauchy Estimator is %lu\n", sizeof(CauchyEstimator));
    test_cauchy_2_state_moshe();
    //test_cauchy_3_state_moshe();
    //test_cauchy_4_state_moshe();
    //test_cauchy_3_state_moshe_3msmts();
    //test_cauchy_5_state_moshe();
    //test_cauchy_four_state_two_pnoise();
    return 0;
}
