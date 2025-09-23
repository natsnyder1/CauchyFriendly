/*******************************************************************
 * Cauchy Estimator API Test Program
 *
 * Written by: Andrew Gao - agao06@ucla.edu
 * Rev: 2025
 * 
 * This is a test and example program for the Cauchy Estimator API 
 * 
 *******************************************************************/

#include "cauchy_estimator_api.hpp"

 int main() {
  // Setup config

  // NOTE: Config must a named lvalue, do not pass rvalues into initializers or constructor
  CauchyEstimatorConfig config;
  
  // Test initialize from CauchyEstimatorConfig struct
  auto testAPI = CauchyAPI::initializeFromJSON("./config.json");
  auto cfg = testAPI.config();

  std::vector<double> one_msmt(cfg.msmt_dim_p, 2); // size of vector is p for p measurements and all values are 2

  testAPI.step(one_msmt);

  std::cout << "TEST HAS COMPLETED" << std::endl;

  
  // Test initialize from JSON

  return 0;
 }