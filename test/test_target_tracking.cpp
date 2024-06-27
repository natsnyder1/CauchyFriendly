#include "../include/cauchy_estimator.hpp"



void test_2d_target_track()
{
    const int n = 2;
    const int p = 1;
    const int pncc = 1;
    const int cmcc = 0;

    double T = 1.0;
    double Phi[n*n] = {1, T, 0, 1};
    double Gamma[n*pncc] = {T*T/2, T};
    double H[p*n] = {1, 0};
    double* B = NULL;
    double* u = NULL;

    double x0[n] = {5, -100};
    double V[p*p] = {pow(0.0421*1.3898,2)};
    double W[pncc*pncc] = {pow(0.423*1.3898,2)};
    double gamma[p] = {sqrt(V[0])/1.3898};
    double beta[pncc] = {sqrt(W[0])/1.3898};

    double A0[n*n] = {1,0,0,1};
    double p0[n] = {7.20, 0.72};
    double b0[n];
    memcpy(b0, x0, n*sizeof(double));

    const int steps = 10;
    int ordering[n] = {1,0};
    set_tr_search_idxs_ordering(ordering, n);
    const bool print_debug = true;
    CauchyEstimator cauchyEst(A0, p0, b0, steps, n, cmcc, pncc, p, print_debug);

    KalmanDynamicsUpdateContainer kduc; 
    kduc.n = n; kduc.p = p; kduc.pncc = pncc; kduc.cmcc = 0;
    kduc.dt = 0; kduc.other_stuff = NULL; kduc.B = NULL; kduc.u = NULL; 
    kduc.H = H;  kduc.Gamma = Gamma; kduc.Phi = Phi; kduc.step = 0; 
    kduc.V = V; kduc.W = W; //kduc.x = x0;
    assert_correct_kalman_dynamics_update_container_setup(&kduc);

    SimulationLogger logger(NULL, steps, x0, &kduc, &gaussian_lti_transition_model, gaussian_lti_measurement_model);
    logger.run_simulation_and_log();

    for(int i = 0; i < 10; i++)
    {
        double zk = logger.msmt_history[i];
        cauchyEst.step(zk, Phi, Gamma, beta, H, gamma[0], B, u);
    }

}

int main()
{
    test_2d_target_track();
    return 0;
}