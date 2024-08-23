#ifndef _BALLISTIC_MISSILE_HPP_
#define _BALLISTIC_MISSILE_HPP_

#include "../dynamic_models.hpp"

const double _RADAR_TO_GAUSS_NOISE = 1.4112;
const double _GAUSS_TO_CAUCHY_NOISE = 1.0 / 1.3898;
const double _RADAR_TO_CAUCHY_NOISE = 1.0; //1.0723;
const double _GAMMA_SCALE_FACTOR = 1.00; // this is to scale the measurement model pdf skinnier/fatter for the cauchy est.

// Three State Ballistic Missie Parameter Structure
// This structure is easily referenced by the dynamical models through the duc->"other_stuff" void pointer
// See "kf_msmt_update_callback" for how this is done
struct BallisticMissileConstants
{
    double dt;
    double t_f;  
    double tau;
    double Vc;
    double R1;
    double R2;
    double E_yt;
    double E_vt;
    double E_at;
    double E_yt2; // variance of initial position
    double E_vt2; // variance of initial velocity
    double E_at2; // variance of acceleration (also +/- the height of the telegraph wave)
    int counter;
    // Parameters For Telegraph Simulation
    double lambda;
    int tele_switch_idx; // the step index at which the telegraph wave switches 
    double tele_switch_sgn; // whether the tele wave is +/-
    double tele_height_2;
    double lambda_second; // poisson parameter that controls which telegraph signal we have (first or second)
    double prob_of_second_tele_wave;
    double tele_wave_height;
    double RADAR_SAS_PARAM;

    // These parameters are adopted from Dr. Speyer's 271B Stochastic Estimation Project
    BallisticMissileConstants(double _RADAR_SAS_PARAM)
    {
        RADAR_SAS_PARAM = _RADAR_SAS_PARAM;
        dt = 0.10;
        t_f = 10.0;
        Vc = 300.0; // 300 // 100
        R1 = 15e-6; // DEBUG: 15e-12; 
        R2 = 1.67e-3; // DEBUG: 1.67e-9;
        E_yt = 0;
        E_vt = 0;
        E_at = 0;
        E_yt2 = pow(1, 2); //1 // 0.1
        E_vt2 = pow(200,2);//200 // 2 or 5 for NL
        E_at2 = pow(100,2); //100 // 10

        // Init Telegraph Params
        lambda = 0.750; // 2.0
        tau = 1.0 / (2.0 * lambda); // tau can be set however when not telegraph....but must be 1.0 / (2.0 * lambda) when conducting telegraph simulation
        tele_switch_sgn = (random_uniform() > 0.5) ? 1.0 : -1.0;
        tele_switch_idx = get_next_tele_idx();

        tele_height_2 = 3*sqrt(E_at2);
        lambda_second = 0.1; // poisson parameter that controls which telegraph signal we have (first or second)
        prob_of_second_tele_wave = 1.0 - exp(-lambda_second);
        tele_wave_height = sqrt(E_at2);
        counter = 0;
    }

    int get_next_tele_idx()
    {
        double draw = log( random_uniform() );
        double inc = 1.0/( lambda * dt );
        double next_idx = -1.0 * inc * draw;
        int integer_next_idx = (int) next_idx;
        return integer_next_idx + 2;
    }

    void reinit_tele_stats()
    {
        tele_switch_sgn = (random_uniform() > 0.5) ? 1.0 : -1.0; // *= -1;
        tele_switch_idx = get_next_tele_idx();
    }

    void compute_telegraph_auto_correlation()
    {
        reinit_tele_stats();
        const int len_signal = 1000000;
        double* signal = (double*) malloc(len_signal * sizeof(double));
        
        // create signal 
        for(int i = 0; i < len_signal; i++)
        {
            if(i >= tele_switch_idx)
            {
                tele_switch_sgn *= -1;
                tele_switch_idx += get_next_tele_idx();
            }
            signal[i] = tele_switch_sgn * sqrt(E_at2);
        }
        const int lag = 200;
        double* auto_corr = (double*) malloc(lag * sizeof(double));
        memset(auto_corr, 0, lag * sizeof(double));
        wss_acorr(auto_corr, signal, len_signal, lag);
        //store_float_elements(signal, len_signal, 1, 1, "SIG_TEST.txt");
        //store_float_elements(auto_corr, lag, 1, 1, "AUTO_COR_TEST.txt");

        free(auto_corr);
        free(signal);
    }

};

// Callback to update the measurement model of the Ballistic Missile for the kalman filter in gaussian noise
void ballistic_missile_gaussian_msmt_update_callback(KalmanDynamicsUpdateContainer* duc)
{
    BallisticMissileConstants* bsc = (BallisticMissileConstants*)(duc->other_stuff);
    const double t_k = duc->step * duc->dt;
    duc->H[0] = 1.0 / (bsc->Vc * (bsc->t_f - t_k + 1e-6));
    const double psd_v = bsc->R1 + bsc->R2 / pow(bsc->t_f - t_k + 1e-6, 2);
    duc->V[0] = psd_v / duc->dt;
}
// Callback to update the measurement model of the Ballistic Missile for the cauchy estimator in gaussian noise
void ballistic_missile_gaussian_msmt_update_callback(CauchyDynamicsUpdateContainer* duc)
{
  BallisticMissileConstants* bsc = (BallisticMissileConstants*) duc->other_stuff;
  const double t_k = duc->step * duc->dt;
  duc->H[0] = 1.0 / (bsc->Vc * (bsc->t_f - t_k + 1e-6));
  const double psd_v = bsc->R1 + bsc->R2 / pow(bsc->t_f - t_k + 1e-6, 2);
  duc->gamma[0] = sqrt(psd_v / bsc->dt) * _GAUSS_TO_CAUCHY_NOISE / _GAMMA_SCALE_FACTOR;
}

// Callback to update the measurement model of the Ballistic Missile for the kalman filter in radar noise
void ballistic_missile_radar_msmt_update_callback(KalmanDynamicsUpdateContainer* duc)
{
    BallisticMissileConstants* bsc = (BallisticMissileConstants*)(duc->other_stuff);
    const double t_k = duc->step * duc->dt;
    duc->H[0] = 1.0 / (bsc->Vc * (bsc->t_f - t_k + 1e-6));
    const double psd_v = bsc->R1 + bsc->R2 / pow(bsc->t_f - t_k + 1e-6, 2);
    duc->V[0] = psd_v / duc->dt * pow(_RADAR_TO_GAUSS_NOISE,2);
}

// Callback to update the measurement model of the Ballistic Missile for the cauchy estimator in radar noise
void ballistic_missile_radar_msmt_update_callback(CauchyDynamicsUpdateContainer* duc)
{
  BallisticMissileConstants* bsc = (BallisticMissileConstants*) duc->other_stuff;
  const double t_k = duc->step * duc->dt;
  duc->H[0] = 1.0 / (bsc->Vc * (bsc->t_f - t_k + 1e-6));
  const double psd_v = bsc->R1 + bsc->R2 / pow(bsc->t_f - t_k + 1e-6, 2);
  duc->gamma[0] = sqrt(psd_v / bsc->dt) * _RADAR_TO_CAUCHY_NOISE / _GAMMA_SCALE_FACTOR;
}

// Model which propogates the state one step into the future using standard gaussian forcing 
void ballistic_missile_gaussian_transition_model(KalmanDynamicsUpdateContainer* duc, double* xk1, double* w)
{
    gaussian_lti_transition_model(duc, xk1, w);
}

// Model which propogates the state one step into the future using telegraph forcing 
void ballistic_missile_telegraph_transition_model(KalmanDynamicsUpdateContainer* duc, double* xk1, double* w)
{
    BallisticMissileConstants* bsc = (BallisticMissileConstants*)(duc->other_stuff);

    double Phi_tele[4] = {1.0, bsc->dt, 
                              0.0, 1.0};
    double Gamma_tele[2] = {-1.0 * pow(bsc->dt,2) / 2.0, 
                                -1.0 * bsc->dt};
    double x_tele[2];
    double work[2];
    double work2[2];

    // The height of the telegraph wave is modelled as the expected std dev of the target acceleration 
    double tele_wave_height = sqrt(bsc->E_at2);

    // If this is the first step of the simulation, call bsc->reinit_tele_stats()
    if(duc->step == 0)
    {
        bsc->reinit_tele_stats();
        duc->x[2] = bsc->tele_switch_sgn * tele_wave_height;
    }

    // If the simulation step is (greater than or) equal to the tele_switch_idx ...
    // ... re-increment the tele_switch_idx w.r.t the underlying poisson distribution ...
    // ... the poisson distribution is a function of the underlying variable "lambda"
    if( duc->step >= bsc->tele_switch_idx )
    {
        bsc->tele_switch_idx += bsc->get_next_tele_idx();
        bsc->tele_switch_sgn *= -1.0;
    }
    // Set w as the telegraph height times +/-1 (which side of the tele wave)
    //duc->x[2] = bsc->tele_switch_sgn * tele_wave_height; // set x[2]_0 to +/- tele_wave_height
    w[0] = bsc->tele_switch_sgn * tele_wave_height;

    // Set x_tele equal to the current position and velocity
    x_tele[0] = duc->x[0];
    x_tele[1] = duc->x[1];

    // Now set xk1[0:2] = Phi_tele @ x_tele + Gamma_tele @ w
    matvecmul(Phi_tele, x_tele, work, 2, 2);
    matvecmul(Gamma_tele, w, work2, 2, 1);
    add_vecs(work, work2, xk1, 2);

    // Set xk1[2] = w[0]
    xk1[2] = w[0];
}

// Model which propogates the state one step into the future using two telegraphs forcing
// one of the telegraphs is deemed the primary telegraph, the other (stronger) telegraph is the secondary, occuring with low probability
void ballistic_missile_double_telegraph_transition_model(KalmanDynamicsUpdateContainer* duc, double* xk1, double* w)
{
    assert(duc->other_stuff != NULL);
    BallisticMissileConstants* bsc = (BallisticMissileConstants*)(duc->other_stuff);

    double Phi_tele[4] = {1.0, bsc->dt, 
                              0.0, 1.0};
    double Gamma_tele[2] = {-1.0 * pow(bsc->dt,2) / 2.0, 
                                -1.0 * bsc->dt};
    double x_tele[2];
    double work[2];
    double work2[2];

    // The nominal height of the telegraph wave is the expected std dev of the target acceleration 
    // The telegraph height can also switch to tele_height_2 with probability prob_of_second_tele_wave

    // If this is the first step of the simulation, call bsc->reinit_tele_stats()
    if(duc->step == 0)
    {
        // Reinitialize the start statistics (the sign and the next tele switch idx)
        bsc->reinit_tele_stats();
        // Use a poisson distribution to determine the next height of the wave
        if( random_uniform() < bsc->prob_of_second_tele_wave )
            bsc->tele_wave_height = bsc->tele_height_2;
        else
            bsc->tele_wave_height = sqrt(bsc->E_at2);
        // Set the start acceleration x_0[2]
        duc->x[2] = bsc->tele_switch_sgn * bsc->tele_wave_height;
    }

    // If the simulation step is (greater than or) equal to the tele_switch_idx ...
    // ... re-increment the tele_switch_idx w.r.t the underlying poisson distribution ...
    // ... the poisson distribution is a function of the underlying variable "lambda"
    if( duc->step >= bsc->tele_switch_idx )
    {
        // Set the next switch idx and flip the telegraph switch sign
        bsc->tele_switch_idx += bsc->get_next_tele_idx();
        bsc->tele_switch_sgn *= -1.0;
        // Use a poisson distribution to determine the next height of the wave
        if( random_uniform() < bsc->prob_of_second_tele_wave )
            bsc->tele_wave_height = bsc->tele_height_2;
        else
            bsc->tele_wave_height = sqrt(bsc->E_at2);
    }
    // Set w as the telegraph height times +/-1 (which side of the tele wave)
    w[0] = bsc->tele_switch_sgn * bsc->tele_wave_height;

    // Set x_tele equal to the current position and velocity
    x_tele[0] = duc->x[0];
    x_tele[1] = duc->x[1];

    // Now set xk1[0:2] = Phi_tele @ x_tele + Gamma_tele @ w
    matvecmul(Phi_tele, x_tele, work, 2, 2);
    matvecmul(Gamma_tele, w, work2, 2, 1);
    add_vecs(work, work2, xk1, 2);

    // Set xk1[2] = w[0]
    xk1[2] = w[0];
}

// standard gaussian measurement update function
void ballistic_missile_gaussian_msmt_model(KalmanDynamicsUpdateContainer* duc, double* z, double* v)
{
    ballistic_missile_gaussian_msmt_update_callback(duc);
    gaussian_lti_measurement_model(duc, z, v);
}

// Linear model which uses the symmetric alpha stable noise (\alpha=1.7) generator to generate measurements 
void ballistic_missile_radar_msmt_model(KalmanDynamicsUpdateContainer* duc, double* z, double* v)
{
    BallisticMissileConstants* bsc = (BallisticMissileConstants*) duc->other_stuff;
    // since this is a radar simulation, we "pretend" the msmt update parameters dont need any scaling 
    // This is beccause here we are generating the simulation, whereas when we run the kalman and the cauchy ... 
    // ... we need to appropriately update the noise terms 
    // so, thats why we call the 'gaussian' update callback here
    ballistic_missile_gaussian_msmt_update_callback(duc); 
    //gaussian_lti_measurement_model(duc, z, v);
    //Note that the scaling param for gaussians is related to the general scaling param by gauss / sqrt(2) = reg
    const int p = duc->p;
    const int n = duc->n;
    // TODDO: This model needs some method to scale the sas-sample drawn.
    for(int i = 0; i < p; i++)
        v[i] = random_symmetric_alpha_stable(bsc->RADAR_SAS_PARAM, sqrt(duc->V[i*p+i]), 0.0);
    matvecmul(duc->H, duc->x, z, p, n);
    add_vecs(z, v, p);
}

// Nonlinear model which uses the symmetric alpha stable noise (\alpha=1.7) generator to generate measurements 
void ballistic_missile_nonlinear_radar_msmt_model(KalmanDynamicsUpdateContainer* duc, double* z, double* v)
{
    // Set the msmt noise scaling param (squared)
    BallisticMissileConstants* bsc = (BallisticMissileConstants*)(duc->other_stuff);
    const double t_k = duc->step * duc->dt;
    const double psd_v = bsc->R1 + bsc->R2 / pow(bsc->t_f - t_k + 1e-6, 2);
    duc->V[0] = psd_v / duc->dt;

    // Construct the measurement 
    v[0] = random_symmetric_alpha_stable(bsc->RADAR_SAS_PARAM, sqrt(duc->V[0]), 0.0);

    double h_x = atan( duc->x[0] / (bsc->Vc * (bsc->t_f - t_k + 1e-6)) ); 
    if( (h_x + v[0]) > (PI / 2.0) ) // Msmt cannot execeed PI/2.0
    {
        v[0] = PI / 2.0 - h_x;
        z[0] = PI / 2.0 - 1e-10;
    }
    else if( (h_x + v[0]) < -(PI / 2.0) ) // Msmt cannot execeed -PI/2.0
    {    
        v[0] = -PI / 2.0 - h_x;
        z[0] = -PI / 2.0 + 1e-10;
    }
    else
        z[0] = h_x + v[0]; // h(x_bar_k) + v_k
}

// Constructs the estimate of the measurement for the EKF: \bar{z_k} = h(\bar{x_k|k-1}) (using the nonlinear measurement model)
void ballistic_missile_ekf_msmt_model(KalmanDynamicsUpdateContainer* duc, double* z)
{
    // Set the msmt noise scaling param (squared)
    BallisticMissileConstants* bsc = (BallisticMissileConstants*)(duc->other_stuff);
    const double t_k = duc->step * duc->dt;
    // Construct the estimate of the measurement 
    z[0] = atan( duc->x[0] / (bsc->Vc * (bsc->t_f - t_k + 1e-6)) ); // h(x_bar_k)
}

// Constructs the estimate of the measurement for the ECE (extended cauchy estimator): \bar{z_k} = h(\bar{x_k|k-1}) (using the nonlinear measurement model)
void ballistic_missile_ece_msmt_model(CauchyDynamicsUpdateContainer* duc, double* z)
{
    // Set the msmt noise scaling param (squared)
    BallisticMissileConstants* bsc = (BallisticMissileConstants*)(duc->other_stuff);
    const double t_k = duc->step * duc->dt;
    // Construct the estimate of the measurement
    z[0] = atan( duc->x[0] / (bsc->Vc * (bsc->t_f - t_k + 1e-6)) ); // h(x_bar_k)
}

// Callback to update (and linearize) the nonlinear measurement model of the Ballistic Missile for the extended kalman filter in radar noise
void ballistic_missile_nonlinear_radar_msmt_update_callback(KalmanDynamicsUpdateContainer* duc)
{
    // Set H_k using the derivative of h(x) = atan(eta), where eta = x[0] / (Vc * (t_f - t)) evaluated at \bar{x}_k|k-1
    BallisticMissileConstants* bsc = (BallisticMissileConstants*)(duc->other_stuff);
    const double t_k = duc->step * duc->dt;
    const double eta = duc->x[0] / (bsc->Vc * (bsc->t_f - t_k + 1e-6));
    const double deriv_eta_dx = 1.0 / (bsc->Vc * (bsc->t_f - t_k + 1e-6)); // w.r.t x
    duc->H[0] = 1.0 / (1.0 + pow(eta,2)) * deriv_eta_dx;
    // Set V[0]
    const double psd_v = bsc->R1 + bsc->R2 / pow(bsc->t_f - t_k + 1e-6,2);
    duc->V[0] = psd_v / duc->dt * pow(_RADAR_TO_GAUSS_NOISE,2); // convert radar scale param to gaussian 
}

// Callback to linearize the nonlinear measurement model of the Ballistic Missile for the extended cauchy estimator in radar noise
void ballistic_missile_nonlinear_radar_msmt_update_callback(CauchyDynamicsUpdateContainer* duc)
{
    // Set H_k using the derivative of h(x) = atan(eta), where eta = x[0] / (Vc * (t_f - t)) evaluated at \bar{x}_k|k-1
    BallisticMissileConstants* bsc = (BallisticMissileConstants*) duc->other_stuff;
    const double t_k = duc->step * duc->dt; // the step is correctly set to 'k' for the following operations 
    const double eta = duc->x[0] / (bsc->Vc * (bsc->t_f - t_k + 1e-6)); // where x_hat == duc->x
    const double deriv_eta_dx = 1.0 / (bsc->Vc * (bsc->t_f - t_k + 1e-6)); // w.r.t x
    duc->H[0] = 1.0 / (1.0 + pow(eta,2)) * deriv_eta_dx; 
    // Set V[0]
    const double psd_v = bsc->R1 + bsc->R2 / pow(bsc->t_f - t_k + 1e-6,2);
    duc->gamma[0] = sqrt(psd_v / bsc->dt) * _RADAR_TO_CAUCHY_NOISE / _GAMMA_SCALE_FACTOR; //* _GAUSS_TO_CAUCHY_NOISE / _GAMMA_SCALE_FACTOR;
}

// Callback to propogate the dynamics forwards (for x_bar) and linearize the measurement model of the Ballistic Missile for the extended cauchy estimator in radar noise
void ballistic_missile_nonlinear_radar_full_update_callback(CauchyDynamicsUpdateContainer* duc)
{
    // In the cauchy case, we are given \bar{x}_k-1|k-1, as the estimator does not explictly construct \bar{x}_k|k-1.
    // Propogate the state forwards using the state dynamics to get \bar{x}_k|k-1
    // Note that x_bar should be saved back into the duc
    const int n = duc->n;
    double x_bar[n];
    assert(duc->x != NULL);
    matvecmul(duc->Phi, duc->x, x_bar, n, n);
    // If a controller is provided
    if(duc->u != NULL)
    {
        assert(duc->B != NULL);
        double work[n];
        matvecmul(duc->B, duc->u, work, n, duc->cmcc);
        add_vecs(x_bar, work, n);
    }
    memcpy(duc->x, x_bar, n*sizeof(double) );
    duc->is_xbar_set_for_ece = true; // we have set duc->x to be x_bar.

    // Set H_k using the derivative of h(x) = atan(eta), where eta = x[0] / (Vc * (t_f - t)) evaluated at \bar{x}_k|k-1
    BallisticMissileConstants* bsc = (BallisticMissileConstants*) duc->other_stuff;
    const double t_k = duc->step * duc->dt; // the step is correctly set to 'k' for the following operations 
    const double eta = duc->x[0] / (bsc->Vc * (bsc->t_f - t_k + 1e-6));
    const double deriv_eta_dx = 1.0 / (bsc->Vc * (bsc->t_f - t_k + 1e-6)); // w.r.t x
    duc->H[0] = 1.0 / (1.0 + pow(eta,2)) * deriv_eta_dx;
    
    // Set V[0]
    const double psd_v = bsc->R1 + bsc->R2 / pow(bsc->t_f - t_k + 1e-6, 2);
    duc->gamma[0] = sqrt(psd_v / bsc->dt) * _RADAR_TO_CAUCHY_NOISE / _GAMMA_SCALE_FACTOR; //_GAUSS_TO_CAUCHY_NOISE;
}
 
// Function to create Phi and Gamma (given tau and dt) for the missile problem
void init_nonlinear_missile_dynamics(double* x, double* Phi, double* Gamma, double* H, const double tau, const double Vc, const double tf, const double DT, const int n)
{
  Phi[0] = 1.0;
  Phi[1] = DT;
  Phi[2] = pow(tau,2)*(1.0-exp(-DT/tau)) - tau*DT;
  Phi[3] = 0.0;
  Phi[4] = 1.0;
  Phi[5] = tau*(exp(-DT/tau) - 1.0);
  Phi[6] = 0.0;
  Phi[7] = 0.0;
  Phi[8] = exp(-DT/tau); 

  Gamma[0] = pow(tau,2)*DT + pow(tau,3)*(exp(-DT/tau)-1.0) - 0.5*tau*DT*DT;
  Gamma[1] = -pow(tau,2)*(exp(-DT/tau)-1.0) - tau*DT;
  Gamma[2] = -tau*(exp(-DT/tau)-1.0);

  H[0] = 1.0 / (1.0 + pow(x[0] / (Vc*tf), 2) ) * 1.0 / (Vc*tf);
  H[1] = 0;
  H[2] = 0;
}

// Function to create Phi and Gamma (given tau and dt) for the missile problem
void init_nonlinear_missile_dynamics(double* x, double* Phi, double* B, double* Gamma, double* H, const double tau, const double Vc, const double tf, const double DT, const int n)
{
  Phi[0] = 1.0;
  Phi[1] = DT;
  Phi[2] = pow(tau,2)*(1.0-exp(-DT/tau)) - tau*DT;
  Phi[3] = 0.0;
  Phi[4] = 1.0;
  Phi[5] = tau*(exp(-DT/tau) - 1.0);
  Phi[6] = 0.0;
  Phi[7] = 0.0;
  Phi[8] = exp(-DT/tau); 

  B[0] = 0.5 * DT * DT;
  B[1] = DT;
  B[2] = 0;

  Gamma[0] = pow(tau,2)*DT + pow(tau,3)*(exp(-DT/tau)-1.0) - 0.5*tau*DT*DT;
  Gamma[1] = -pow(tau,2)*(exp(-DT/tau)-1.0) - tau*DT;
  Gamma[2] = -tau*(exp(-DT/tau)-1.0);

  H[0] = 1.0 / (1.0 + pow(x[0] / (Vc*tf), 2) ) * 1.0 / (Vc*tf);
  H[1] = 0;
  H[2] = 0;
}

// Function to create Phi, Gamma and H (given tau and dt) for the linear missile problem
void init_missile_dynamics(double* Phi, double* Gamma, double* H, const double tau, const double Vc, const double tf, const double DT, const int n)
{
  Phi[0] = 1.0;
  Phi[1] = DT;
  Phi[2] = pow(tau,2)*(1.0-exp(-DT/tau)) - tau*DT;
  Phi[3] = 0.0;
  Phi[4] = 1.0;
  Phi[5] = tau*(exp(-DT/tau) - 1.0);
  Phi[6] = 0.0;
  Phi[7] = 0.0;
  Phi[8] = exp(-DT/tau); 

  Gamma[0] = pow(tau,2)*DT + pow(tau,3)*(exp(-DT/tau)-1.0) - 0.5*tau*DT*DT;
  Gamma[1] = -pow(tau,2)*(exp(-DT/tau)-1.0) - tau*DT;
  Gamma[2] = -tau*(exp(-DT/tau)-1.0);

  H[0] = 1.0 / (Vc * tf);
  H[1] = 0;
  H[2] = 0;
}

#endif // _BALLISTIC_MISSILE_HPP_