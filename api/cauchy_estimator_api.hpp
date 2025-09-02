/*******************************************************************
 * Cauchy Estimator API Header
 *
 * Written by: Andrew Gao - agao06@ucla.edu
 * Rev: 2025
 * 
 * This header declares the Cauchy Estimator API classes, functions, types, and templates
 * 
 *******************************************************************/

#pragma once

#include "cauchy_estimator.hpp"
#include <vector>
#include <string>

enum class StatusCode : uint8_t {
    Ok = 0,
    InitializeError = 1
};

struct Status {
    StatusCode code;
    std::string message;

    static Status ok() { return {StatusCode::Ok, ""}; }
    static Status error(StatusCode c, const std::string& msg) { return {c, msg}; }
};

struct CauchyEstimatorConfig {
    bool valid = false;

    int state_dim_n;
    int msmt_dim_p;
    int process_noise_dim_q;
    int control_dim;

    int num_steps;
    int num_windows;

    std::vector<double> Phi, Gamma, H, beta, gamma, A0, p0, b0;    
};

class CauchyAPI {
    Status initialize(CauchyEstimatorConfig cfg); // configure estimator: dimensions, time step, intial CF/mean/covariance, noise stuff
    Status intializeFromJSON();
    Status step(); // advances estimator by one time-step, take in new measurements (and optionally controls)
    Status getConditionals(); // get conditional mean/covariance after update
    Status reset(); // discard/reinitialize state for sliding windows
};

/**********************
 * TODO: 
 *  -> simulink plugin
 *  -> LTV matlab
 *  -> noise models conversion (uniform, gaussian, cauchy)
 *  -> docker image for environment
 * 
 *  -> Kalman filter on/off
 *  -> setup: Cauchy dist. settings, noise settings, WINDOW SIZE
 *  -> simulation: timestep stuff, dynamics ("nonlin", "lin" flag), initial condition
 *  -> logging/data plotting: logging flags, plotting flags
 */