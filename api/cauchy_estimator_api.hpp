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
#include "cauchy_windows.hpp"
#include "json.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <optional>

enum class StatusCode : uint8_t {
    Ok = 0,
    ConfigError = 1,
    InitializeError = 2
};

struct Status {
    StatusCode code;
    std::string message;

    static Status ok() { return {StatusCode::Ok, ""}; }
    static Status error(StatusCode c, const std::string& msg) { return {c, msg}; }
};

struct CauchyEstimatorConfig {
    bool valid = false;
    void validate() const {
        if (!valid) throw std::runtime_error("Invalid config");
    }

    // private: TODO: Should this be private to avoid config changes?
    int state_dim_n;
    int msmt_dim_p;
    int process_noise_dim_q;
    int control_dim;

    int num_steps;
    int num_windows;

    std::vector<double> Phi, Gamma, H, beta, gamma, A0, p0, b0;
    std::vector<double> B, u;

    // TODO: Figure out what these do (keeping example settings for now)
    bool WINDOW_PRINT_DEBUG = false;
    bool WINDOW_LOG_SEQUENTIAL = false;
    bool WINDOW_LOG_FULL = false;
    bool is_extended = false;
    std::string log_dir; // Should not set this to NULL apparently?
    std::vector<double> window_var_boost;

};

class CauchyAPI {
    // Private fields
    private:
    bool initialized_ = false;
    // Care declaration order for member init lsit
    CauchyEstimatorConfig cfg_;
    CauchyDynamicsUpdateContainer duc_;
    SlidingWindowManager swm_;

    static CauchyDynamicsUpdateContainer makeCDUC(CauchyEstimatorConfig& cfg);
    static SlidingWindowManager makeSWM(CauchyEstimatorConfig& cfg, CauchyDynamicsUpdateContainer& duc);
    
    /***
     * TODO:
     * Kalman settings, KalmanDynamicsUpdateContainer, SimulationLogger*,
     * KF Simulation and Logging
     ***/

    // Public functions
    public:
    // API Constructor from config
    explicit CauchyAPI(CauchyEstimatorConfig cfg);

    // General API functions
    static CauchyAPI initialize(CauchyEstimatorConfig& cfg); // configure estimator: dimensions, time step, intial CF/mean/covariance, noise stuff
    static CauchyAPI intializeFromJSON(std::string path = "config.json");
    Status step(); // advances estimator by one time-step, take in new measurements (and optionally controls)
    Status getConditionals(); // get conditional mean/covariance after update
    Status reset(); // discard/reinitialize state for sliding windows

    const CauchyEstimatorConfig& config() noexcept;
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