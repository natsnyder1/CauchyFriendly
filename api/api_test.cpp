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

  // Test initialize from JSON

  return 0;
 }