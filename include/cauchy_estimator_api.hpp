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
#include <span>
#include <iostream>
#include <iomanip>

/*******************************************************************
 * StatusCode and Status structs for communication with API
 *******************************************************************/
enum class StatusCode : uint8_t {
    Ok = 0,
    ConfigError = 1,
    InitializeError = 2,
    StepError = 3
};

struct Status {
    StatusCode code;
    std::string message;

    static Status ok() { return {StatusCode::Ok, ""}; }
    static Status error(StatusCode c, const std::string& msg) { return {c, msg}; }
};

/*******************************************************************
 * Config struct for setting up CauchyEstimator
 *******************************************************************/
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

/*******************************************************************
 * Statistics Struct for storing data from estimator and using in control
 *******************************************************************/
struct CauchyStatistics {
    std::span<const double> mean;
    std::span<const double> covariance;
    int n = 0;
    int window_index = -1;
    int step_index = -1;

};

/*******************************************************************
 * CauchyAPI class declarations and definitions
 *******************************************************************/
class CauchyAPI {
    // Private fields
    private:

    // TEMPORARY MEMBER WHILE I FIGURE OUT child_window_loop and malloc for num_steps
    int curr_step_ = 0;

    bool initialized_ = false;
    // Care declaration order for member init list
    CauchyEstimatorConfig cfg_;
    CauchyDynamicsUpdateContainer duc_;
    SlidingWindowManager swm_;

    CauchyStatistics latest_stats_;

    static CauchyDynamicsUpdateContainer makeCDUC(CauchyEstimatorConfig& cfg) {
        CauchyDynamicsUpdateContainer duc;
        duc.n = cfg.state_dim_n;
        duc.pncc = cfg.process_noise_dim_q;
        duc.p = cfg.msmt_dim_p;
        duc.cmcc = cfg.control_dim;
        duc.Phi = cfg.Phi.data();
        duc.Gamma = cfg.Gamma.data();
        duc.H = cfg.H.data();
        duc.B = NULL;
        duc.u = NULL;
        duc.beta = cfg.beta.data();
        duc.gamma = cfg.gamma.data();

        // TODO: CHECK THIS STUFF
        duc.step = 0;
        duc.dt = 0;
        duc.other_stuff = NULL;
        duc.x = NULL;

        return duc;
    }

    static SlidingWindowManager makeSWM(CauchyEstimatorConfig& cfg, CauchyDynamicsUpdateContainer& duc) {
        return SlidingWindowManager(cfg.num_windows, cfg.num_steps+1, cfg.A0.data(), cfg.p0.data(), cfg.b0.data(),
            &duc, cfg.WINDOW_PRINT_DEBUG, cfg.WINDOW_LOG_SEQUENTIAL, cfg.WINDOW_LOG_FULL, 
            cfg.is_extended, NULL, NULL, NULL, cfg.window_var_boost.empty() ? nullptr : cfg.window_var_boost.data(), cfg.log_dir.empty() ? nullptr: cfg.log_dir.c_str());
    }

    void updateStatistics() noexcept {
        latest_stats_ = CauchyStatistics{};

        const int i = swm_.msmt_count - 1;
        const int n = swm_.n;

        // Compute pointers into the “selected/winner” buffers at step i
        const double* mean_i = swm_.full_window_means
                            + static_cast<size_t>(i) * static_cast<size_t>(n);
        const double* cov_i  = swm_.full_window_variances
                            + static_cast<size_t>(i) * static_cast<size_t>(n) * static_cast<size_t>(n);

        // Fill the view (no copies)
        latest_stats_.n           = n;
        latest_stats_.step_index  = i;
        latest_stats_.window_index = swm_.full_window_idxs ? swm_.full_window_idxs[i] : -1;
        latest_stats_.mean        = std::span<const double>(mean_i, static_cast<size_t>(n));
        latest_stats_.covariance  = std::span<const double>(cov_i,  static_cast<size_t>(n) * static_cast<size_t>(n));
    }

/*******************************************************************
 *  TODO:
 *  Kalman settings, KalmanDynamicsUpdateContainer, SimulationLogger*,
 *  KF Simulation and Logging
 *******************************************************************/

    // Public functions
    public:
    // API Constructor from config
    explicit CauchyAPI(CauchyEstimatorConfig cfg)
    : cfg_((cfg.validate(), std::move(cfg))),
      duc_(makeCDUC(cfg_)), swm_(makeSWM(cfg_, duc_))
    {
        initialized_ = true;
    }

    // General API functions
    // Configure estimator: dimensions, time step, intial CF/mean/covariance, noise stuff
    static CauchyAPI initialize(CauchyEstimatorConfig& cfg) {
        return CauchyAPI(cfg);
    }

    static CauchyAPI initializeFromJSON(std::string path = "config.json") {
        using json = nlohmann::json;
        std::ifstream f(path);
        if (!f) throw std::runtime_error("Could not open JSON");

        json j;
        f >> j;
        
        CauchyEstimatorConfig cfg;
        
        // Populate config from JSON
        cfg.state_dim_n = j["state_dim_n"];
        cfg.msmt_dim_p = j["msmt_dim_p"];
        cfg.process_noise_dim_q = j["process_noise_dim_q"];
        cfg.control_dim = j["control_dim"];
        cfg.num_steps = j["num_steps"];
        cfg.num_windows = j["num_windows"];
        cfg.Phi = j["Phi"].get<std::vector<double>>();
        cfg.Gamma = j["Gamma"].get<std::vector<double>>();
        cfg.H = j["H"].get<std::vector<double>>();
        cfg.beta = j["beta"].get<std::vector<double>>();
        cfg.gamma = j["gamma"].get<std::vector<double>>();
        cfg.A0 = j["A0"].get<std::vector<double>>();
        cfg.p0 = j["p0"].get<std::vector<double>>();
        cfg.b0 = j["b0"].get<std::vector<double>>();
        cfg.WINDOW_PRINT_DEBUG = j["WINDOW_PRINT_DEBUG"];
        cfg.WINDOW_LOG_SEQUENTIAL = j["WINDOW_LOG_SEQUENTIAL"];
        cfg.WINDOW_LOG_FULL = j["WINDOW_LOG_FULL"];
        cfg.is_extended = j["is_extended"];

        if (j.contains("log_dir") && !j["log_dir"].is_null()) {
            cfg.log_dir = j["log_dir"].get<std::string>();
        }
        else {
            cfg.log_dir.clear();
        }
        if (j.contains("window_var_boost") && !j["window_var_boost"].is_null()) {
            cfg.window_var_boost = j["window_var_boost"].get<std::vector<double>>();
        }
        else {
            cfg.window_var_boost.clear();
        }

        cfg.valid = true;
        return CauchyAPI(cfg);
    }

    // Advances estimator by one time-step, take in new measurements (and optionally controls)
    Status step(std::span<double> z, std::span<double> u = {}) {
        // TODO: Write step, based on window_manager.cpp
        // MAIN QUESTION: What if we don't have a total step? SWM is constructed with it
        // but what if we don't want one? see how to edit SWM i guess...
        // // Iterate over each step, window manager iterates over the individual measurements of each steps measurement vector
        // for(int i = 0; i < total_steps; i++)
        // {
        //     double* zs = sim_log->msmt_history + i*p;
        //     swm.step(zs, NULL);
        // }
        // swm.shutdown();
        if (z.size() != static_cast<size_t>(cfg_.msmt_dim_p)) {
            return Status::error(StatusCode::StepError, "check measurement size");
        }
        if (!u.empty() && u.size() != static_cast<size_t>(cfg_.control_dim)) {
            return Status::error(StatusCode::StepError, "check control size");
        }
        if (curr_step_ > cfg_.num_steps) {
            return Status::error(StatusCode::StepError, "Outside the number of possible steps (WILL BE PATCHED)");
        }
        swm_.step(z.data(), u.empty() ? nullptr : u.data());
        ++curr_step_;

        updateStatistics();
        return Status::ok();
    } 


    // get conditional mean/covariance after update
    const CauchyStatistics& getStatistics() const noexcept {
        return latest_stats_;
    } 
    const CauchyStatistics* getStatisticsPtr() const noexcept {
        return &latest_stats_;
    }

    // TODO: Write reset for the API
    Status reset(); 

    const CauchyEstimatorConfig& config() noexcept {
        return cfg_;
    }
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



inline void printStatistics(const CauchyStatistics& stats) {
    std::cout << std::endl;
    std::cout << "Step: " << stats.step_index
              << " | Best Window Number: " << stats.window_index
              << " | Dimension n = " << stats.n << "\n";

    // Print mean
    std::cout << "Mean: [ ";
    for (int i = 0; i < stats.n; ++i) {
        std::cout << std::fixed << std::setprecision(4) << stats.mean[i];
        if (i < stats.n - 1) std::cout << ", ";
    }
    std::cout << " ]\n";

    // Print covariance as a matrix
    std::cout << "Covariance:\n";
    for (int r = 0; r < stats.n; ++r) {
        for (int c = 0; c < stats.n; ++c) {
            std::cout << std::setw(10) << std::setprecision(4)
                      << stats.covariance[r * stats.n + c] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}
