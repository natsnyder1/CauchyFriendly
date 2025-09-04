/*******************************************************************
 * Cauchy Estimator API Source
 *
 * Written by: Andrew Gao - agao06@ucla.edu
 * Rev: 2025
 * 
 * This class defines the functions and class methods for the Cauchy Estimator API
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
        cfg.is_extended, NULL, NULL, NULL, cfg.window_var_boost, cfg.log_dir);
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

CauchyAPI CauchyAPI::intializeFromJSON()
{
    return Status();
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
