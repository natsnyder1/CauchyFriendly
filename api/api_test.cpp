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

  // NOTE: Config must a named lvalue, do not pass rvalues into initializers or constructor due to std::move
  CauchyEstimatorConfig config;
  
  // Test initialize from CauchyEstimatorConfig struct
  // Test initialize from JSON
  auto testAPI = CauchyAPI::initializeFromJSON("./config.json");
  auto cfg = testAPI.config();

  std::vector<double> one_msmt(cfg.msmt_dim_p, 2); // size of vector is p for p measurements and all values are 2

  // IN A LOOP
  // Some get measurement function
  testAPI.step(one_msmt);
  // Get statistics
  const auto& stats = testAPI.getStatistics();
  // Print statistics
  printStatistics(testAPI.getStatistics());

  
  std::cout << "TEST HAS COMPLETED" << std::endl;

  

  return 0;
 }