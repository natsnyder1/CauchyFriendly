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

  // Measurements
  // One measurement
  std::vector<double> one_msmt(cfg.msmt_dim_p, 2); // size of vector is p for p measurements and all values are 2
  // Many measurements
  double zs[10] = {-1.2172011200334241, -0.35943271347277583, -0.52353301003957098, 0.5855389648301792, 
  -0.8048243525901404, 0.34053610027255954, 1.0580483915838776, -0.55152999529515989,
  -0.72879029737003309}; 

//   // IN A LOOP
//   // Some get measurement function
//   testAPI.step(one_msmt);
//   // Get statistics
//   const auto& stats = testAPI.getStatistics();
//   // Print statistics
//   printStatistics(testAPI.getStatistics());

    for(int i = 0; i < 10; i++) {
        std::vector<double> curr_msmt(cfg.msmt_dim_p, zs[i]);
        testAPI.step(curr_msmt);
        const auto& stats = testAPI.getStatistics();
        // std::cout << stats.mean[0] << std::endl;
    }

  
  std::cout << "TEST HAS COMPLETED" << std::endl;

  

  return 0;
 }