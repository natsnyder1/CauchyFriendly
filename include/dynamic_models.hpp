#ifndef _DYNAMIC_MODELS_HPP_
#define _DYNAMIC_MODELS_HPP_

#include <cstring>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <assert.h>

#include "cpu_linalg.hpp"
#include "random_variables.hpp"

// Container used to update the Dynamics of the Kalman Filter when the model is LTI, LTV, or non-linear
// This structure is very helpful when writing a general callback function which updates the dynamics of the KF from step to step
struct KalmanDynamicsUpdateContainer
{
  // Based on the dynamical model x_k+1 = Phi @ x + B @ u + Gamma @ w
  // Based on the measurement model z_k = H @ x_k + v_k
  // Generally, B == Gamma 
  // Generally, pncc and cmcc are the same, as there is a single noise associated with each control. 
  double* x; // State Vector \in R^(n)
  double* u; // Actuator Control Vector \in R^{cmcc}
  double dt; // Discrete Time Step (if one is defined)
  int step; // Current Step: simulation time so far would be equal to dt * step
  int n; // number of states
  int cmcc; // number of columns of B 
  int pncc; // number of columns of Gamma
  int p; // number of measurements (rows of H)
  double* Phi; // Discrete Time Transition Matrix
  double* Gamma; // Discrete Time Process Noise Matrix 
  double* B; // Discrete Time Control Matrix (This is typically the same as Gamma)
  double* H; // Discrete Time Measurement Model
  double* W; // Discrete Time Process Noise
  double* V; // Discrete Time Measurement Noise
  void* other_stuff; // cast as required to include other non-basic parameters in the update to the dynamics or the measurement

  // Default Constructor
  // The parameters below must be set by user.
  // See "assert_correct_kalman_dynamics_update_container_setup" for rules of correct setup
  KalmanDynamicsUpdateContainer()
  {
    step = 0;
    dt = 0;
    n = 0;
    cmcc = 0;
    pncc = 0;
    p = 0;
    Phi = NULL;
    Gamma = NULL;
    B = NULL;
    H = NULL;
    W = NULL;
    V = NULL;
    other_stuff = NULL;
  }

};

// Structure used for updating CauchyDynamics at the user level
// Rename this the cauchy dynamics update container
struct CauchyDynamicsUpdateContainer
{
  double* x; 
  double* u;
  double dt;
  int step;
  int n; 
  int cmcc;
  int pncc;
  int p;
  double* Phi;
  double* Gamma;
  double* B;
  double* H;
  double* beta;
  double* gamma;
  bool is_xbar_set_for_ece;
  void* other_stuff; // cast as required to include other non-basic parameters in the update to the dynamics or the measurement

  CauchyDynamicsUpdateContainer()
  {
    step = 0;
    dt = 0;
    n = 0;
    cmcc = 0;
    pncc = 0;
    p = 0;
    // x = NULL; // This should probably be set too, can test later.
    u = NULL;
    Phi = NULL;
    Gamma = NULL;
    B = NULL;
    H = NULL;
    beta = NULL;
    gamma = NULL;
    other_stuff = NULL;
  }

};

struct SAS_noise_container
{
    double alpha; // exponential decay parameter of the SAS pdf
    double beta; // scaling parameter for process noise SAS pdf
    double gamma; // scaling parameter for msmt noise SAS pdf
    double max_beta_realization; // for clipping process noise in worst case
    double max_gamma_realization; // for clipping process noise in worst case

    SAS_noise_container()
    {
        alpha = -1;
        beta = -1;
        gamma = -1;
        max_beta_realization = -1;
        max_gamma_realization = -1;
    }
};

void assert_correct_sas_noises(SAS_noise_container* sas_noises)
{
    assert( (sas_noises->alpha >= 1) && (sas_noises->alpha <= 2) );
    assert( sas_noises->beta > 0);
    assert( sas_noises->max_beta_realization > 0);
    assert( sas_noises->max_gamma_realization > 0);
}

void assert_correct_kalman_dynamics_update_container_setup(KalmanDynamicsUpdateContainer* duc)
{
    assert(duc != NULL);
    assert(duc->n > 0);
    assert(duc->cmcc >= 0);
    assert(duc->pncc > 0);
    assert(duc->p > 0);
    assert(duc->Phi != NULL);
    assert(duc->Gamma != NULL);
    assert(duc->H != NULL);
    assert(duc->W != NULL);
    assert(duc->V != NULL);
    if(duc->cmcc > 0)
        assert(duc->B != NULL);
    else 
        assert(duc->B == NULL);
}

void assert_correct_cauchy_dynamics_update_container_setup(CauchyDynamicsUpdateContainer* duc)
{
    assert(duc != NULL);
    assert(duc->n > 0);
    assert(duc->pncc > 0);
    assert(duc->cmcc >= 0);
    assert(duc->p > 0);
    assert(duc->Phi != NULL);
    assert(duc->Gamma != NULL);
    assert(duc->H != NULL);
    assert(duc->beta != NULL);
    assert(duc->gamma != NULL);
    if(duc->cmcc > 0)
        assert(duc->B != NULL);
    else 
        assert(duc->B == NULL);
}

/*
// A base class that is used to construct simulations which use feedback control
class DynamicSimulation
{
    public:
    virtual int step_simulation(double* xk_est, double* u_feedback, KalmanDynamicsUpdateContainer* duc, double* xk1_true, double* w, double* zk1, double* v)
    {
        printf("Please fill in the virtual function to run your simulation!\n");
        return 0;
    }
    virtual int step_simulation(double* xk_est, double* u_feedback, CauchyDynamicsUpdateContainer* duc, double* xk1_true, double* w, double* zk1, double* v)
    {
        printf("Please fill in the virtual function to run your simulation!\n");
        return 0;
    }
    virtual int reset_simulation(void)
    {
        return 0;
    }
};
*/

// Propogates the system one step forwards, then simulates the measurement about this state
// xk1[n]; // R^n - state of system propogated one step,
// z[p]; // R^p - meausurement from sensor (constructed using xk1),
// w[pncc]; // R^pncc - process noise drawn using W (really from chol_W),
// v[p]; // R^p - measurement noise drawn from V (really from chol_V),
// xk[n]; // R^n - state of system at current step,
// u[cmcc]; // R^cmcc - deterministic actuator values (controls),
// Phi[n*n]; // R^(n x n) - Dynamics Matrix,
// B[n*cmcc]; // R^(n x cmcc) - Control Matrix,
// Gamma[n*pncc]; // R^(n x pncc) - Process Noise Matrix,
// H[p*n]; // R^(p x n) - Measurement Model,
// mean_W[pncc]; // R^pncc - mean of process noise,
// mean_V[p]; // R^p - mean of measurement noise,
// chol_W[pncc*pncc]; // R^(pncc x pncc) - Cholesky of Variance of Process Noise W,
// chol_V[p*p]; // R^(p x p) - Cholesky of Variance of Process Noise V,
// n - state dimention
// cmcc - control matrix (B) column count,
// pncc - process noise matrix (Gamma) column count,
// p - dimention of measurement model,
// work[n*n]; // R^(n x n) - memory workspace,
// work2[n*n]; // R^(n x n) - memory workspace #2,
//old name: simulate_gaussian_dynamic_system
void one_step_gaussian_dynamic_system(double* xk1, double* z, double* w, double* v, 
    const double* xk, const double* u, 
    const double* Phi, const double* B, const double* Gamma, const double* H, 
    const double* mean_W, const double* mean_V, const double* chol_W, const double* chol_V, 
    const int n, const int cmcc, const int pncc, const int p, double* work, double* work2)
{
    // get simulated process noise:
    multivariate_random_normal(w, mean_W, chol_W, work, pncc);
    // get simulated measurement noise:
    multivariate_random_normal(v, mean_V, chol_V, work, p);

    // Form x_k+1 = Phi @ x + B @ u + Gamma @ w
    matvecmul(Phi, xk, work, n, n);
    matvecmul(B, u, work2, n, cmcc);
    add_vecs(work, work2, n, 1.0);
    matvecmul(Gamma, w, work2, n, pncc);
    add_vecs(work, work2, xk1, n, 1.0); // xk1 now contains Phi @ x + B @ u + Gamma @ w
    // Form z = H @ x + v
    matvecmul(H, xk1, z, p, n);
    add_vecs(z, v, p, 1.0); // z now contains H @ x + v
}


// Implements the Standard LTI Transition model:
// x_k+1 = Phi @ x_k + B @ u + Gamma @ w
// w is assumed zero meaned white noise here
// xk1[n]; // --OUTPUT -- R^n - state of system propogated one step,
// Phi[n*n]; // R^(n x n) - Dynamics Matrix,
// xk[n]; // R^n - state of system at current step,
// B[n*cmcc]; // R^(n x cmcc) - Control Matrix,
// Gamma[n*pncc]; // R^(n x pncc) - Process Noise Matrix,
// w[pncc]; // --OUTPUT -- R^pncc - process noise drawn using W (really from chol_W),
void gaussian_lti_transition_model(KalmanDynamicsUpdateContainer* duc, double* xk1, double* w)
{
    const int pncc = duc->pncc;
    const int n = duc->n;
    const int cmcc = duc->cmcc;
    double chol_W[pncc*pncc];
    double mean_W[pncc];
    double work[n*n];
    double work2[n*n];
    if(pncc > 1)
    {
        // get simulated process noise:
        memcpy(chol_W, duc->W, pncc*pncc * sizeof(double) );
        memset(mean_W, 0, pncc*sizeof(double));
        cholesky(chol_W, pncc);
        multivariate_random_normal(w, mean_W, chol_W, work, pncc);
    }
    else
        w[0] = random_normal(0, sqrt(duc->W[0]) );
    
    // Form x_k+1 = Phi @ x + B @ u + Gamma @ w
    matvecmul(duc->Phi, duc->x, work, n, n);
    matvecmul(duc->B, duc->u, work2, n, cmcc);
    add_vecs(work, work2, n, 1.0);
    matvecmul(duc->Gamma, w, work2, n, pncc);
    add_vecs(work, work2, xk1, n, 1.0);
    // xk1 now contains Phi @ x + B @ u + Gamma @ w
}

// Implements the Standard LTI measurement model:
// z = H @ x + v
void gaussian_lti_measurement_model(KalmanDynamicsUpdateContainer* duc, double* z, double* v)
{
    const int n = duc->n;
    const int p = duc->p;

    double chol_V[p*p];
    double mean_V[p];
    double work[p];

    if(p > 1)
    {
        // get simulated process noise:
        memcpy(chol_V, duc->V, p*p*sizeof(double) );
        memset(mean_V, 0, p*sizeof(double));
        cholesky(chol_V, p);
        multivariate_random_normal(v, mean_V, chol_V, work, p);
    }
    else
        v[0] = random_normal(0, sqrt(duc->V[0]) );

    // Form z = H @ x + v
    matvecmul(duc->H, duc->x, z, p, n);
    add_vecs(z, v, p, 1.0); // z now contains H @ x + v
}

// Implements the LTI Transition model:
// x_k+1 = Phi @ x_k + Gamma @ w
// w is assumed zero meaned cauchy noise here
// xk1[n]; // --OUTPUT -- R^n - state of system propogated one step,
// Phi[n*n]; // R^(n x n) - Dynamics Matrix,
// xk[n]; // R^n - state of system at current step,
// B[n*cmcc]; // R^(n x cmcc) - Control Matrix,
// Gamma[n*pncc]; // R^(n x pncc) - Process Noise Matrix,
// w[pncc]; // --OUTPUT -- R^pncc - process noise drawn using W (really from chol_W),
void cauchy_lti_transition_model(CauchyDynamicsUpdateContainer* duc, double* xk1, double* w)
{
    const int n = duc->n;
    const int pncc = duc->pncc;
    double work[n*n];
    double work2[n*n];

    for(int i = 0; i < pncc; i++)
        w[i] = random_cauchy(duc->beta[i]);
    
    // Form x_k+1 = Phi @ x + Gamma @ w
    matvecmul(duc->Phi, duc->x, work, n, n);
    //add_vecs(work, work2, n, 1.0);
    matvecmul(duc->Gamma, w, work2, n, pncc);
    add_vecs(work, work2, xk1, n, 1.0);
    // xk1 now contains Phi @ x + B @ u + Gamma @ w
}

// Implements the Standard LTI measurement model:
// z = H @ x + v,
// with v cauchy distributed
void cauchy_lti_measurement_model(CauchyDynamicsUpdateContainer* duc, double* z, double* v)
{
    for(int i = 0; i < duc->p; i++)
        v[i] = random_cauchy(duc->gamma[i]);

    // Form z = H @ x + v
    matvecmul(duc->H, duc->x, z, duc->p, duc->n);
    add_vecs(z, v, duc->p, 1.0); // z now contains H @ x + v
}

void sas_lti_transition_model(CauchyDynamicsUpdateContainer* duc, double* xk1, double* w)
{
    const int n = duc->n;
    const int pncc = duc->pncc;
    double work[n*n];
    double work2[n*n];
    SAS_noise_container* sas_noises = (SAS_noise_container*)duc->other_stuff;
    assert(sas_noises != NULL);
    double beta = (sas_noises->alpha == 2) ? sas_noises->beta / sqrt(2) : sas_noises->beta;
    for(int i = 0; i < pncc; i++)
    {    
        w[i] = random_symmetric_alpha_stable(sas_noises->alpha, beta, 0);
        if(fabs(w[i]) > sas_noises->max_beta_realization)
            w[i] = sgn(w[i]) * sas_noises->max_beta_realization;
    }    
    // Form x_k+1 = Phi @ x + Gamma @ w
    matvecmul(duc->Phi, duc->x, work, n, n);
    //add_vecs(work, work2, n, 1.0);
    matvecmul(duc->Gamma, w, work2, n, pncc);
    add_vecs(work, work2, xk1, n, 1.0);
    // xk1 now contains Phi @ x + B @ u + Gamma @ w
}

// Implements the Standard LTI measurement model:
// z = H @ x + v,
// with v cauchy distributed
void sas_lti_measurement_model(CauchyDynamicsUpdateContainer* duc, double* z, double* v)
{
    SAS_noise_container* sas_noises = (SAS_noise_container*)duc->other_stuff;
    assert(sas_noises != NULL);
    double gamma = (sas_noises->alpha == 2) ? sas_noises->gamma / sqrt(2) : sas_noises->gamma;
    for(int i = 0; i < duc->p; i++)
    {
        v[i] = random_symmetric_alpha_stable(sas_noises->alpha, gamma, 0);
        if(fabs(v[i]) > sas_noises->max_gamma_realization)
            v[i] = sgn(v[i]) * sas_noises->max_gamma_realization;
    }
    // Form z = H @ x + v
    matvecmul(duc->H, duc->x, z, duc->p, duc->n);
    add_vecs(z, v, duc->p, 1.0); // z now contains H @ x + v
}



// General function which uses some user-provided dynamical model to simulate State/Msmt/Noise Histories over num_steps,
// this function can be used to simulate any type of system as the transition and measurement models are left up to the user,
// NOTE: It is completely up to the user to implement the transition and measurement models correctly (Regardless of LTI/LTV),
// NOTE: For both LTI or LTV models, the user MUST use the Kalman Update Dynamics Container to keep track of parameters,
// NOTE: If the simulation requires deterministic control, use "us \in R^(num_steps x cmcc)" to bring these controls in. 
// NOTE: This function will NOT change the values of "us", though its data pointed to is not declared as constant.
// NOTE MAY NEED TO CHANGE THIS: This function WILL increment the duc's step count after running the transition model (and reset it to zero before starting the simulation)
void simulate_dynamic_system(const int num_steps, 
    const int n, const int cmcc, const int pncc, const int p,
    const double* x0, double* us, 
    double* true_state_history, 
    double* msmt_history,
    double* process_noise_history,
    double* msmt_noise_history,
    KalmanDynamicsUpdateContainer* duc,
    void (*transition_model)(KalmanDynamicsUpdateContainer*, double* xk1, double* w),
    void (*msmt_model)(KalmanDynamicsUpdateContainer*, double* z, double* v), 
    bool with_msmt_on_first_step = false
    )
{
    assert(duc != NULL);
    //assert_correct_kalman_dynamics_update_container_setup(duc);
    
    double* x = (double*) malloc(n * sizeof(double));
    null_ptr_check(x);
    double* xk1 = (double*) malloc(n * sizeof(double));
    null_ptr_check(xk1);
    double z[p];
    double w[pncc];
    double v[p];

    duc->step = 0;
    memcpy(x, x0, n*sizeof(double));
    memcpy(true_state_history, x, n*sizeof(double));

    // This is for the case where we need to simulate z_0 = H @ x_0 + v_0, in addition to {z_1,z_2,...}
    int first_step_offset = 0;
    if(with_msmt_on_first_step)
    {
        duc->x = x;
        (*msmt_model)(duc, z, v);
        memcpy(msmt_history, z, p*sizeof(double));
        memcpy(msmt_noise_history, v, p*sizeof(double));
        first_step_offset += 1;
    }

    // Simulating {z_1, z_2, ...z_n}
    for(int i = 0; i < num_steps; i++)
    {
        duc->x = x;
        duc->u = us + i*cmcc;
        (*transition_model)(duc, xk1, w);
        memcpy(true_state_history + i*n, x, n * sizeof(double) ); // if any (after-thhe-fact) updates to x must be made (this is rare, tho..telegraph does use it)
        memcpy(true_state_history + (i+1)*n, xk1, n * sizeof(double) );
        memcpy(process_noise_history + i*pncc, w, pncc * sizeof(double) );
        
        duc->step += 1;
        duc->x = xk1;
        (*msmt_model)(duc, z, v);
        memcpy(msmt_history + (i+first_step_offset)*p, z, p*sizeof(double));
        memcpy(msmt_noise_history + (i + first_step_offset)*p, v, p*sizeof(double));
        ptr_swap(&x, &xk1);
    }
    free(x);
    free(xk1);
}

// General function which uses some user-provided dynamical model to simulate State/Msmt/Noise Histories over num_steps,
// this function can be used to simulate any type of system as the transition and measurement models are left up to the user,
// NOTE: It is completely up to the user to implement the transition and measurement models correctly (Regardless of LTI/LTV),
// NOTE: For both LTI or LTV models, the user MUST use the Kalman Update Dynamics Container to keep track of parameters,
// NOTE: If the simulation requires deterministic control, use "us \in R^(num_steps x pncc)" to bring these controls in. 
// NOTE: This function will NOT change the values of "us", though its data pointed to is not declared as constant.
// NOTE MAY NEED TO CHANGE THIS: This function WILL increment the duc's step count after running the transition model (and reset it to zero before starting the simulation)
void simulate_dynamic_system(const int num_steps, 
    const int n, const int pncc, const int p,
    const double* x0, 
    double* true_state_history, 
    double* msmt_history,
    double* process_noise_history,
    double* msmt_noise_history,
    CauchyDynamicsUpdateContainer* duc,
    void (*transition_model)(CauchyDynamicsUpdateContainer*, double* xk1, double* w),
    void (*msmt_model)(CauchyDynamicsUpdateContainer*, double* z, double* v), 
    bool with_msmt_on_first_step = true
    )
{
    assert(duc != NULL);
    //assert_correct_kalman_dynamics_update_container_setup(duc);
    
    double* x = (double*) malloc(n * sizeof(double));
    null_ptr_check(x);
    double* xk1 = (double*) malloc(n * sizeof(double));
    null_ptr_check(xk1);
    double z[p];
    double w[pncc];
    double v[p];

    duc->step = 0;
    memcpy(x, x0, n*sizeof(double));
    memcpy(true_state_history, x, n*sizeof(double));

    // This is for the case where we need to simulate z_0 = H @ x_0 + v_0, in addition to {z_1,z_2,...}
    int first_step_offset = 0;
    if(with_msmt_on_first_step)
    {
        duc->x = x;
        (*msmt_model)(duc, z, v);
        memcpy(msmt_history, z, p*sizeof(double));
        memcpy(msmt_noise_history, v, p*sizeof(double));
        first_step_offset += 1;
    }

    // Simulating {z_1, z_2, ...z_n}
    for(int i = 0; i < num_steps; i++)
    {
        duc->x = x;
        (*transition_model)(duc, xk1, w);
        memcpy(true_state_history + i*n, x, n * sizeof(double) ); // if any (after-thhe-fact) updates to x must be made (this is rare, tho..telegraph does use it)
        memcpy(true_state_history + (i+1)*n, xk1, n * sizeof(double) );
        memcpy(process_noise_history + i*pncc, w, pncc * sizeof(double) );
        
        duc->step += 1;
        duc->x = xk1;
        (*msmt_model)(duc, z, v);
        memcpy(msmt_history + (i+first_step_offset)*p, z, p*sizeof(double));
        memcpy(msmt_noise_history + (i+first_step_offset)*p, v, p*sizeof(double));
        ptr_swap(&x, &xk1);
    }
    free(x);
    free(xk1);
}

#endif // _DYNAMIC_MODELS_HPP_