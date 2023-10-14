#ifndef _KALMAN_FILTER_HPP_
#define _KALMAN_FILTER_HPP_

#include "cauchy_types.hpp"
#include "dynamic_models.hpp"

// Discrete Time Kalman Filter, for LTI and LTV systems
// Row Major Ordering,
// u[cmcc]; // R^cmcc - deterministic actuator values (controls),
// z[p]; // R^p - meausurement from sensor,
// x[n]; // R^n - conditional state of system,
// P[n*n]; // R^(n x n) - conditional covariance of system,
// K[n*p];  // R^(n x p) - Kalman Gain,
// Phi[n*n]; // R^(n x n) - Dynamics Matrix,
// B[n*cmcc]; // R^(n x cmcc) - Control Matrix,
// Gamma[n*pncc]; // R^(n x pncc) - Process Noise Gains,
// H[p*n]; // R^(p x n) - Measurement Model,
// W[pncc*pncc]; // R^(pncc x pncc) - Variance of Process,
// V[p*p]; // R^(p x p) - Variance of Measurement,
// n - state dimention
// cmcc - control matrix (B) column count,
// pncc - process noise matrix (Gamma) column count,
// p - dimention of measurement model,
// work[n*n]; // R^(n x n) - memory workspace,
// work2[n*n]; // R^(n x n) - memory workspace #2,
// NOTE: work2 contains the residual r = z - H @ x upon returning from this function call,
// NOTE: The user MUST provide callback function(s) if the dynamics/msmt model(s) are Linear Time Varying (LTV),
// NOTE: If LTV (as mentioned above), "duc" MUST have its associated fields and member pointers correctly set to the TV data,
// NOTE: If the state transition dynamics are LTV, dynamics_update_callback must not be NULL (this callback possibly updates Phi,B,Gamma,W),
// NOTE: If the measurement model is LTV, msmt_update_callback must not be NULL (this callback updates H and V).
void kalman_filter(double* u, const double* z,
                   double* x, double* P, double* K,
                   const double* Phi, const double* B, const double* Gamma, 
                   const double* H, const double* W, const double* V,
                   const int n, const int cmcc, const int pncc, const int p, 
                   double* work, double* work2, 
                   void (*dynamics_update_callback)(KalmanDynamicsUpdateContainer*) = NULL,
                   void (*msmt_update_callback)(KalmanDynamicsUpdateContainer*) = NULL, 
                   KalmanDynamicsUpdateContainer* duc = NULL)
{
    // For LTV Systems, NOTE: There are no NULL checks and the user MUST setup their duc appropriately

    // Step 0: If LTV, update the transition dynamics (Phi_k, B_k, Gamma_k) and noise (W_k) about x_(k|k), u_k, and 'other_stuff' (our void pointer, for miscellaneous params)
    if(duc != NULL)
    {
        duc->x = x;
        duc->u = u;
    }
    if(dynamics_update_callback != NULL)
        (*dynamics_update_callback)(duc); // called for current step: duc->step
    if(duc != NULL)
        duc->step += 1; // duc->step is updated to match the indexing xk+1|k
        

    // -- A PRIORI STEPS -- //
    // 1. Propogate Model: \bar{x_k+1|k} = Phi_k @ \hat{x_k|k} + B @ u 
    matvecmul(Phi, x, work, n, n);
    matvecmul(B, u, work2, n, cmcc);
    add_vecs(work, work2, x, n, 1.0); // x now contains \bar{x_k+1|k}

    // 2. Propogate Variance: M_k+1|k = Phi_k @ P_k|k @ Phi_k.T + Gamma_k @ W_k @ Gamma_k.T
    matmatmul(Phi, P, work, n, n, n, n, false, false);
    matmatmul(work, Phi, P, n, n, n, n, false, true);

    matmatmul(Gamma, W, work, n, pncc, pncc, pncc, false, false);
    matmatmul(work, Gamma, work2, n, pncc, n, pncc, false, true);
    add_mat(P, work2, 1.0, n, n); // P now contains M_k+1|k

    // -- A POSTERIORI STEPS --
    // If LTV, update the measurement model (H_k, V_k) about x_k|k-1
    if(msmt_update_callback != NULL)
        (*msmt_update_callback)(duc);

    // 3. Calculate Kalman Gain K = M @ H.T @ (H @ M @ H.T + V).I
    matmatmul(H, P, work, p, n, n, n, false, false);
    matmatmul(work, H, work2, p, n, p, n, false, true);
    add_mat(work2, V, 1.0, p, p); // work2 contains (H @ M @ H.T + V)
    matmatmul(P, H, work, n, n, p, n, false, true); // work conatins M @ H.T \in R^(n x p)
    solve_pd(work2, work, K, p, n); // 'K' now holds M @ H.T @ (H @ M @ H.T + V).I

    // 4. Find Posterior (conditional) covariance of the state: P_k|k = (I - K @ H) @ M_k|k-1 @ (I - K @ H).T + K @ V @ K.T
    matmatmul(K, H, work, n, p, p, n);
    scale_mat(work, -1, n, n);
    for(int i = 0; i < n; i++)
        work[i*n + i] += 1; // work contains (I - K @ H)
    matmatmul(work, P, work2, n, n, n, n, false, false);
    matmatmul(work2, work, P, n, n, n, n, false, true); // P holds (I - K @ H) @ M @ (I - K @ H).T
    matmatmul(K, V, work, n, p, p, p, false, false);
    matmatmul(work, K, work2, n, p, n, p, false, true); // work2 contains K @ V @ K.T
    add_mat(P, work2, 1, n, n); // P now contains (I - K @ H) @ M @ (I - K @ H).T + K @ V @ K.T

    // 5. Find Posterior (conditional) state: \hat{x_k|k} = \bar{x_k|k-1} + K @ (z - H @ \bar{x_k|k-1})
    // Note: work2 contains the residual r = z - H @ x upon returning from this function call
    matvecmul(H, x, work, p, n);
    add_vecs(z, work, work2, p, -1); // work2 contains the residual: z - H @ x_k|k-1
    matvecmul(K, work2, work, n, p); // work contains K @ (z - H @ x_k|k-1)
    add_vecs(x, work, n, 1); // x now contains \hat{x_k|k}
}

// runs the kalman filter over 'num_steps' intervals, using the provided us and zs,
// us \in R^(num_steps x cmcc) -- the controls,
// zs \in R^(num_steps x p) -- the measurements,
// kf_state_history \in R^(num_steps+1 x n x 1) -- OUTPUT -- the state history (including x0),
// kf_covariance_history \in R^(num_steps+1 x n x n) -- OUTPUT -- the covariance history (including P0),
// kf_Ks_history \in R^(num_steps x n x p) -- OUTPUT -- the Kalman Gain history,
// kf_rs_history \in R^(num_steps x p x 1) -- OUTPUT -- the residual history,
// NOTE: x MUST be initialized with x0,
// NOTE: P MUST be initialized with P0,
void run_kalman_filter(const int num_steps, double* us, const double* zs,
                   double* kf_state_history, double* kf_covariance_history,
                   double* kf_Ks_history, double* kf_residual_history,
                   const double* x0, const double* P0,
                   double* Phi, double* B, double* Gamma, 
                   double* H, double* W, double* V,
                   const int n, const int cmcc, const int pncc, const int p, 
                   void (*dynamics_update_callback)(KalmanDynamicsUpdateContainer*) = NULL,
                   void (*msmt_update_callback)(KalmanDynamicsUpdateContainer*) = NULL,  
                   KalmanDynamicsUpdateContainer* duc = NULL)
{
    double work[n*n];
    double work2[n*n];
    double K[n*p];
    double P[n*n];
    double x[n];

    // Memcpy the initial states into the state / variance histories
    memcpy(x, x0, n*sizeof(double));
    memcpy(P, P0, n*n*sizeof(double));
    memcpy(kf_state_history, x0, n*sizeof(double));
    memcpy(kf_covariance_history, P0, n*n*sizeof(double));

    if( (dynamics_update_callback != NULL) || (msmt_update_callback != NULL) )
    {
        assert(duc != NULL);
        //assert_correct_kalman_dynamics_update_container_setup(duc);
        duc->step = 0;
    }    

    for(int i = 0; i < num_steps; i++)
    {   
        kalman_filter(us + i*cmcc, 
                zs + i*p,
                x, P, K,
                Phi, B, Gamma, 
                H, W, V,
                n, cmcc, pncc, p, 
                work, work2, 
                dynamics_update_callback, 
                msmt_update_callback,
                duc);
        memcpy(kf_state_history + (i+1)*n, x, n*sizeof(double));
        memcpy(kf_covariance_history + (i+1)*n*n, P, n*n*sizeof(double));
        memcpy(kf_Ks_history + i*n*p, K, n*p*sizeof(double));  
        memcpy(kf_residual_history + i*p, work2, p*sizeof(double));
    }

}

// KF LTI / LTV MONTE CARLO
// monte_carlo_num_steps -- the number of steps the monte carlo should average,
// sim_num_steps -- the number of steps the simulation should run,
// us \in R^(sim_num_steps x cmcc)-- actuator control history,
// x0 \in R^(n) -- The true initial state of the system,
// P0 \in R^(n x n) -- The initial covariance surrounding the initial state,
// NOTE: "us" is treated as constant, though its typing of "double*" may not make it appear so, 
// ... there are points where duc will need to point to the individual controls within "us", 
// ... this is as to why "us" is not typed a "const double*", but rather a "double*",
void monte_carlo_ltiv_kalman_filter(const int monte_carlo_num_steps,
                const int sim_num_steps,
                double* us,
                const double* x0, const double* P0,
                double* mc_covariance_history,
                double* kf_covariance_history,
                KalmanDynamicsUpdateContainer* duc,
                void (*transition_model)(KalmanDynamicsUpdateContainer*, double* xk1, double* w),
                void (*msmt_model)(KalmanDynamicsUpdateContainer*, double* z, double* v),
                void (*dynamics_update_callback)(KalmanDynamicsUpdateContainer*) = NULL, 
                void (*msmt_update_callback)(KalmanDynamicsUpdateContainer*) = NULL)
{
    // Assert that our duc has been properly set-up here
    assert_correct_kalman_dynamics_update_container_setup(duc);

    // Setting System Size Constants from duc
    const int n = duc->n;
    const int cmcc = duc->cmcc;
    const int pncc = duc->pncc;
    const int p = duc->p;

    // Allocate the Simulated State, Msmt and Noise Histories
    double* true_state_history = (double*) malloc( (sim_num_steps+1) * n * sizeof(double));
    null_ptr_check(true_state_history);
    double* msmt_history = (double*) malloc( sim_num_steps * p * sizeof(double));
    null_ptr_check(msmt_history);
    double* process_noise_history = (double*) malloc( sim_num_steps * pncc * sizeof(double));
    null_ptr_check(process_noise_history);
    double* msmt_noise_history = (double*) malloc(sim_num_steps * p * sizeof(double));
    null_ptr_check(msmt_noise_history);

    // Allocate the Kalman Filters State and Covariance Histories
    double* kf_state_history = (double*) malloc((sim_num_steps+1) * n * sizeof(double));
    null_ptr_check(kf_state_history);
    double* kf_Ks_history = (double*) malloc(sim_num_steps * n * p * sizeof(double));
    null_ptr_check(kf_Ks_history);
    double* kf_residual_history = (double*) malloc(sim_num_steps * p * sizeof(double));
    null_ptr_check(kf_residual_history);

    // Allocate the Monte Carlo error histories accross all trials 
    double* mc_trial_i_error_history = (double*) malloc(monte_carlo_num_steps * (sim_num_steps+1) * n * sizeof(double*));
    null_ptr_check(mc_trial_i_error_history);
    //for(int i = 0; i < monte_carlo_num_steps; i++)
    //    mc_trial_i_error_history[i] = (double*) malloc( (sim_num_steps+1) * n * sizeof(double) );
    // Allocate the Monte Carlo averaged error history (for averaging across all trials) 
    double* mc_averaged_error_history = (double*) malloc( (sim_num_steps+1) * n * sizeof(double) );
    null_ptr_check(mc_averaged_error_history);
    // Allocate the Monte Carlo trial i error covariance 
    double* mc_trial_i_covariance_history = (double*) malloc( (sim_num_steps+1) * n * n * sizeof(double) );
    null_ptr_check(mc_trial_i_covariance_history);
    // Set the mc_averaged_error_history to zero
    memset(mc_averaged_error_history, 0, (sim_num_steps+1) * n * sizeof(double));
    // Set the mc_covariance_history to zero
    memset(mc_covariance_history, 0, (sim_num_steps+1) * n * n * sizeof(double));
    

    // Declare helper variables for the monte carlo 
    double work[n];
    double trial_i_x0[n];
    double chol_P0[n*n];
    memcpy(chol_P0, P0, n*n*sizeof(double));
    cholesky(chol_P0, n);

    for(int i = 0; i < monte_carlo_num_steps; i++)
    {
        printf("Monte Carlo Step %d!\n", i);

        // Step 1: Draw a starting trial_i_x0 from x0 and P0 for the KF.
        multivariate_random_normal(trial_i_x0, x0, chol_P0, work, n);

        // Step 2. Simulate the dynamic system for sim_num_steps using the user-provided dynamics_update_callback, model_update_callback
        simulate_dynamic_system(
            sim_num_steps, 
            n, cmcc, pncc, p,
            trial_i_x0, us, 
            true_state_history, 
            msmt_history,
            process_noise_history,
            msmt_noise_history,
            duc,
            transition_model,
            msmt_model, 
            false);

        // Step 3. Run the LTI / LTV Kalman Filter Simulation
        run_kalman_filter(sim_num_steps, us, msmt_history,
                kf_state_history, kf_covariance_history,
                kf_Ks_history, kf_residual_history,
                x0, P0,
                duc->Phi, duc->B, duc->Gamma, 
                duc->H, duc->W, duc->V,
                n, cmcc, pncc, p,
                dynamics_update_callback,
                msmt_update_callback,  
                duc);

        // 3. Take the true state history and the kf_state_history and form the trial_i error history
        add_vecs(true_state_history, kf_state_history, mc_trial_i_error_history + i * (sim_num_steps+1) * n, (sim_num_steps+1) * n, -1);
        // Sum the trial_i error history to the running sum (to form the averaged error history across the full monte carlo)
        add_vecs(mc_averaged_error_history, mc_trial_i_error_history + i * (sim_num_steps+1) * n, (sim_num_steps+1) * n, 1);
    }
    printf("Finished Monte Carlo Simulation! ... Calculating Monte Carlo Error Covariance\n");
    // Form the mc_averaged_error_history by dividing by (one over) the number of trials
    double one_over_mc_steps = 1.0 / ((double) monte_carlo_num_steps);
    scale_vec(mc_averaged_error_history, one_over_mc_steps, (sim_num_steps+1) * n);

    printf("Monte Carlo Averaged Error History\n");
    print_mat(mc_averaged_error_history, sim_num_steps+1, n);

    // Now find the covariance across all simulation steps
    for(int i = 0; i < monte_carlo_num_steps; i++)
    {
        for(int j = 0; j < sim_num_steps+1; j++)
        {
            covariance
            (
                mc_trial_i_covariance_history + j*n*n,
                mc_averaged_error_history + j*n,
                mc_trial_i_error_history + i * (sim_num_steps+1) * n + j*n, 
                work, 
                n
            );
        }
        add_vecs(mc_covariance_history, mc_trial_i_covariance_history, (sim_num_steps+1) * n * n, 1);
    }
    double one_over_mc_steps_minus_one = 1.0 / ((double) monte_carlo_num_steps - 1);
    scale_vec(mc_covariance_history, one_over_mc_steps_minus_one, (sim_num_steps+1) * n * n);
    printf("Finished Calculating Monte Carlo Error Covariance!\n");

    // Free Arrays
    // Free Simulation
    free(true_state_history);
    free(msmt_history);
    free(process_noise_history);
    free(msmt_noise_history);
    // Free Kalman Filter
    free(kf_state_history);
    free(kf_Ks_history);
    free(kf_residual_history);
    // Free Monte Carlo
    //for(int i = 0; i < monte_carlo_num_steps; i++)
    //    free(mc_trial_i_error_history[i]);
    free(mc_trial_i_error_history);
    free(mc_averaged_error_history);
}


// Discrete Time Extended Kalman Filter. For nonlinear systems,
// All inputs same as LTI/LTV KF, except the following,
// nonlinear_transition_model: calls the user-provided nonlinear dynamics model and propogates state to return xbar,
// nonlinear_msmt_model: calls the user-provided nonlinear msmt model and returns estimate zbar of the measurement,
// Please see function kalman filter for full list of parameters.
void extended_kalman_filter(double* u, const double* z,
                   double* x, double* P, double* K,
                   const double* Phi, const double* B, const double* Gamma, 
                   const double* H, const double* W, const double* V,
                   const int n, const int cmcc, const int pncc, const int p, 
                   double* work, double* work2,
                   void  (*nonlinear_transition_model)(KalmanDynamicsUpdateContainer*, double* _xbar) = NULL,
                   void  (*nonlinear_msmt_model)(KalmanDynamicsUpdateContainer*, double* _zbar) = NULL,
                   void (*dynamics_update_callback)(KalmanDynamicsUpdateContainer*) = NULL,
                   void (*msmt_update_callback)(KalmanDynamicsUpdateContainer*) = NULL, 
                   KalmanDynamicsUpdateContainer* duc = NULL,
                   bool only_msmt_update = false,
                   bool only_time_prop = false)
{
    // For nonlinear systems, NOTE: There are no NULL checks and the user MUST setup their duc appropriately

    // Step 0: If nonlinear or LTV, update the linearized transition dynamics (Phi_k, B_k, Gamma_k) and noise (W_k) about x_(k|k), u_k, and 'other_stuff' (our void pointer, for miscellaneous params)
    if(duc != NULL)
    {
        duc->x = x;
        duc->u = u;
    }

    // -- A PRIORI STEPS -- //
    // Run the time propogation segement if the estimator does not wish to only run a measurement update
    if(!only_msmt_update)
    {
        if(dynamics_update_callback != NULL)
            (*dynamics_update_callback)(duc); // called for current step: duc->step, sets Phi and Gamma and W_k (if need be)
        if(duc != NULL)
            duc->step += 1; // duc->step is updated to match the indexing xk+1|k

        // 1. Propogate Model: \bar{x_k+1|k} = Phi_k @ \hat{x_k|k} + B @ u
        if(nonlinear_transition_model == NULL)
        {
            matvecmul(Phi, x, work, n, n);
            matvecmul(B, u, work2, n, cmcc);
            add_vecs(work, work2, x, n, 1.0); // x now contains \bar{x_k+1|k}
        }
        // if transition model is nonlinear, compute \bar{x_k+1|k} = f(\hat{x_k|k}, u_k)
        else
            (*nonlinear_transition_model)(duc, x); // x now contains \bar{x_k+1|k}
        // 2. Propogate Variance: M_k+1|k = Phi_k @ P_k|k @ Phi_k.T + Gamma_k @ W_k @ Gamma_k.T
        matmatmul(Phi, P, work, n, n, n, n, false, false);
        matmatmul(work, Phi, P, n, n, n, n, false, true);

        matmatmul(Gamma, W, work, n, pncc, pncc, pncc, false, false);
        matmatmul(work, Gamma, work2, n, pncc, n, pncc, false, true);
        add_mat(P, work2, 1.0, n, n); // P now contains M_k+1|k
    }


    // -- A POSTERIORI STEPS -- //
    if(!only_time_prop)
    {
        // If nonlinear, update the measurement model (H_k, V_k) about x_k|k-1, and 'other_stuff' (our void pointer, for miscellaneous params)
        if(msmt_update_callback != NULL)
            (*msmt_update_callback)(duc); // sets H_k, V_k (if need be)

        // 3. Calculate Kalman Gain K = M @ H.T @ (H @ M @ H.T + V).I
        matmatmul(H, P, work, p, n, n, n, false, false);
        matmatmul(work, H, work2, p, n, p, n, false, true);
        add_mat(work2, V, 1.0, p, p); // work2 contains (H @ M @ H.T + V)
        matmatmul(P, H, work, n, n, p, n, false, true); // work conatins M @ H.T \in R^(n x p)
        solve_pd(work2, work, K, p, n); // 'K' now holds M @ H.T @ (H @ M @ H.T + V).I

        // 4. Find Posterior (conditional) covariance of the state: P_k|k = (I - K @ H) @ M_k|k-1 @ (I - K @ H).T + K @ V @ K.T
        matmatmul(K, H, work, n, p, p, n);
        scale_mat(work, -1, n, n);
        for(int i = 0; i < n; i++)
            work[i*n + i] += 1; // work contains (I - K @ H)
        matmatmul(work, P, work2, n, n, n, n, false, false);
        matmatmul(work2, work, P, n, n, n, n, false, true); // P holds (I - K @ H) @ M @ (I - K @ H).T
        matmatmul(K, V, work, n, p, p, p, false, false);
        matmatmul(work, K, work2, n, p, n, p, false, true); // work2 contains K @ V @ K.T
        add_mat(P, work2, 1, n, n); // P now contains (I - K @ H) @ M @ (I - K @ H).T + K @ V @ K.T

        // 5. Find Posterior (conditional) state: \hat{x_k|k} = \bar{x_k|k-1} + K @ (z - H @ \bar{x_k|k-1})
        // Note: work2 contains the residual r = z - H @ x upon returning from this function call
        if(nonlinear_msmt_model == NULL)
            matvecmul(H, x, work, p, n);
        // If the msmt model is nonlinear, compute: \hat{x_k|k} = \bar{x_k|k-1} + K @ (z - h(\bar{x_k|k-1}))
        else
            (*nonlinear_msmt_model)(duc, work);
        add_vecs(z, work, work2, p, -1); // work2 contains the residual: z - H @ x_k|k-1 or z - h(\bar{x_k|k-1})
        matvecmul(K, work2, work, n, p); // work contains K @ (z - H @ x_k|k-1) or K @ (z - h(\bar{x_k|k-1}))
        add_vecs(x, work, n, 1); // x now contains \hat{x_k|k}
    }
}

// runs the extended kalman filter over 'num_steps' intervals, using the provided us and zs,
// see run_kalman_filter and extended_kalman_filter functions for help with input arguments.
void run_extended_kalman_filter(const int num_steps, double* us, const double* zs,
                   double* kf_state_history, double* kf_covariance_history,
                   double* kf_Ks_history, double* kf_residual_history,
                   const double* x0, const double* P0,
                   double* Phi, double* B, double* Gamma,
                   double* H, double* W, double* V,
                   const int n, const int cmcc, const int pncc, const int p,
                   void  (*nonlinear_transition_model)(KalmanDynamicsUpdateContainer*, double* _xbar) = NULL,
                   void  (*nonlinear_msmt_model)(KalmanDynamicsUpdateContainer*, double* _zbar) = NULL,
                   void (*dynamics_update_callback)(KalmanDynamicsUpdateContainer*) = NULL,
                   void (*msmt_update_callback)(KalmanDynamicsUpdateContainer*) = NULL,  
                   KalmanDynamicsUpdateContainer* duc = NULL,
                   bool with_msmt_update_at_step_zero=false)
{
    double work[n*n];
    double work2[n*n];
    double K[n*p];
    double P[n*n];
    double x[n];
    int msmt_offset = with_msmt_update_at_step_zero ? 1 : 0; // If msmt_offset=1, kf_state_history,kf_covariance_history get overridden for step 0
    // Memcpy the initial states into the state / variance histories
    memcpy(x, x0, n*sizeof(double));
    memcpy(P, P0, n*n*sizeof(double));
    memcpy(kf_state_history, x0, n*sizeof(double));
    memcpy(kf_covariance_history, P0, n*n*sizeof(double));

    if( (dynamics_update_callback != NULL) || (msmt_update_callback != NULL) )
    {
        assert(duc != NULL);
        //assert_correct_kalman_dynamics_update_container_setup(duc);
        duc->step = 0;
    }    

    for(int i = 0; i < num_steps; i++)
    {   
        int control_offset = (i != 0)*msmt_offset; // if msmt_offset is set, delay the counter i by one step when indexing 'us' since only MU runs at i=0.
        bool msmt_update_only = (!i)*msmt_offset; // will evaluate to 1 on first step if msmt_offset is set to 1, 0 otherwise
        extended_kalman_filter(us + (i - control_offset)*cmcc, 
            zs + i*p,
            x, P, K,
            Phi, B, Gamma, 
            H, W, V,
            n, cmcc, pncc, p, 
            work, work2,
            nonlinear_transition_model,
            nonlinear_msmt_model,
            dynamics_update_callback, 
            msmt_update_callback,
            duc,
            msmt_update_only
            );
        memcpy(kf_state_history + (i+1-msmt_offset)*n, x, n*sizeof(double)); 
        memcpy(kf_covariance_history + (i+1-msmt_offset)*n*n, P, n*n*sizeof(double));
        memcpy(kf_Ks_history + i*n*p, K, n*p*sizeof(double));  
        memcpy(kf_residual_history + i*p, work2, p*sizeof(double));
    }

}



#endif //_KALMAN_FILTER_HPP_