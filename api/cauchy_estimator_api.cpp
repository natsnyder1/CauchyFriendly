/*******************************************************************
 * Cauchy Estimator API Source
 *
 * Written by: Andrew Gao - agao06@ucla.edu
 * Rev: 2025
 * 
 * This class defines the functions and class methods for the Cauchy Estimator API
 * 
 * TEMPORARILY REMOVE. KEEP EVERYTHING HEADER ONLY
 * 
 *******************************************************************/

#include "cauchy_estimator_api.hpp"

// Container for Cauchy Estimator
CauchyDynamicsUpdateContainer CauchyAPI::makeCDUC(CauchyEstimatorConfig& cfg)
{
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

// New Sliding Window Manager
SlidingWindowManager CauchyAPI::makeSWM(CauchyEstimatorConfig& cfg, CauchyDynamicsUpdateContainer& duc)
{
    return SlidingWindowManager(cfg.num_windows, cfg.num_steps+1, cfg.A0.data(), cfg.p0.data(), cfg.b0.data(),
        &duc, cfg.WINDOW_PRINT_DEBUG, cfg.WINDOW_LOG_SEQUENTIAL, cfg.WINDOW_LOG_FULL, 
        cfg.is_extended, NULL, NULL, NULL, cfg.window_var_boost.empty() ? nullptr : cfg.window_var_boost.data(), cfg.log_dir.empty() ? nullptr: cfg.log_dir.c_str());
}

CauchyAPI::CauchyAPI(CauchyEstimatorConfig cfg)
    : cfg_((cfg.validate(), std::move(cfg))),
      duc_(makeCDUC(cfg_)), swm_(makeSWM(cfg_, duc_))
{
    initialized_ = true;
}

CauchyAPI CauchyAPI::initialize(CauchyEstimatorConfig& cfg)
{
    return CauchyAPI(cfg);
}

CauchyAPI CauchyAPI::initializeFromJSON(std::string path)
{
    using json = nlohmann::json;
    try {
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
        return CauchyAPI(cfg);
    }
    catch (const std::exception& e) {
        std::runtime_error("Failed to load from JSON");
    }
}

Status CauchyAPI::step()
{
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
    return Status();
}

const CauchyEstimatorConfig &CauchyAPI::config() noexcept
{
    return cfg_;
}
