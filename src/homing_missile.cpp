#include "../include/models/ballistic_missile.hpp"
#include "../include/kalman_filter.hpp"
#include "../include/cauchy_windows.hpp"
#include <cstdlib>
#include <cstring>
#include <dirent.h>

// This controls whether the simulation runs in state estimate depenedent feedback or with the true dynamic simulation state
const bool IS_FEEDBACK_STATE_DEPENDENT = true; 

// Model which propogates the state one step into the future using two telegraphs forcing
// one of the telegraphs is deemed the primary telegraph, the other (stronger) telegraph is the secondary, occuring with low probability
void evader_double_telegraph_transition_model(KalmanDynamicsUpdateContainer* duc, double* xk1, double* w)
{
    assert(duc->other_stuff != NULL);
    BallisticMissileConstants* bsc = (BallisticMissileConstants*)(duc->other_stuff);

    double Phi_tele[4] = {1.0, bsc->dt, 
                              0.0, 1.0};
    double Gamma_tele[2] = {1.0 * pow(bsc->dt,2) / 2.0, 
                                1.0 * bsc->dt};
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

void control_feedback_law(double* u, const double* x, const int n, const int cmcc, void* other_stuff)
{
    BallisticMissileConstants* bsc = (BallisticMissileConstants*) other_stuff;
    assert(bsc != NULL);
    double t_k = bsc->counter * bsc->dt;
    double range_to_go = bsc->Vc * (bsc->t_f - t_k + 1e-6);
    double K_e = 5.0;
    u[0] = -bsc->Vc * K_e * (range_to_go * x[1] + bsc->Vc * x[0]) / pow(range_to_go,2);
    bsc->counter += 1;
}

class HomingSimulation : public DynamicSimulation
{

    public:
    // Constants which define the measurement and process statistics
    int max_num_steps;
    int sim_num_steps;
    BallisticMissileConstants* bsc;

    double* evader_telegraph_wave;
    double* ekf_true_relative_state_history;
    double* ekf_control_history;
    double* ekf_msmt_history; // msmt is influenced by the control history (state dependent)
    double* cauchy_true_relative_state_history;
    double* cauchy_control_history;
    double* cauchy_msmt_history; // msmt is influenced by the control history (state dependent)
    double* msmt_noise_history; // this is declared once for both the cauchy and EKF simulations

    KalmanDynamicsUpdateContainer rduc; // relative simulation dynamics container for evader-pursuer simulation
    // Three state state estimator system dynamics dimentions
    int n;
    int cmcc;
    int pncc;
    int p;
    // Two state state evader pursuer system dynamics dimentions
    int n_r;
    int cmcc_r;
    int pncc_r;
    int p_r;
    double zero_val; // 0
    double junk_val; // 0
    // Relative dynamics of the pursuer and evader
    double* Phi_r;
    double* B_r;

    HomingSimulation(int _max_num_steps, BallisticMissileConstants* _bsc)
    {
        max_num_steps = _max_num_steps;
        sim_num_steps = max_num_steps - 1;
        bsc = _bsc;

        // Three state state estimator system dynamics dimentions
        n = 3;
        cmcc = 1;
        pncc = 1;
        p = 1;
        // Two state state evader pursuer system dynamics dimentions
        n_r = 2;
        cmcc_r = cmcc;
        pncc_r = pncc;
        p_r = p;
        // Scratch values of zero
        zero_val = 0;
        junk_val = 0;

        // Simulation history
        evader_telegraph_wave = (double*) malloc(max_num_steps*cmcc*sizeof(double));
        msmt_noise_history = (double*) malloc(max_num_steps*p*sizeof(double));
        ekf_true_relative_state_history = (double*) malloc(max_num_steps*n*sizeof(double));
        ekf_control_history = (double*) malloc(sim_num_steps*cmcc*sizeof(double));
        ekf_msmt_history = (double*) malloc(max_num_steps*p*sizeof(double));
        cauchy_true_relative_state_history = (double*) malloc(max_num_steps*n*sizeof(double));
        cauchy_control_history = (double*) malloc(sim_num_steps*cmcc*sizeof(double));
        cauchy_msmt_history = (double*) malloc(max_num_steps*p*sizeof(double));

        Phi_r = (double*) malloc(n_r*n_r*sizeof(double));
        B_r = (double*) malloc(n_r*cmcc_r*sizeof(double));

        Phi_r[0] = 1;
        Phi_r[1] = bsc->dt;
        Phi_r[2] = 0;
        Phi_r[3] = 1;
        B_r[0] = bsc->dt*bsc->dt/2.0;
        B_r[1] = bsc->dt;

        rduc.n = n_r; rduc.pncc = pncc_r; 
        rduc.cmcc = cmcc_r; rduc.p = p_r;
        rduc.dt = bsc->dt;
        rduc.Phi = Phi_r; rduc.Gamma = B_r; 
        rduc.B = B_r; rduc.H = &zero_val;
        rduc.W = &zero_val; rduc.V = &zero_val;
        rduc.other_stuff = bsc;
        // rduc.x, rduc.u -- need to be set still
        //simulate_telegraph_process_noise_and_measurement_noise();
        
    }
    
    void simulate_telegraph_process_noise_and_measurement_noise()
    {
        // Initialize the simulation at the same place for both the cauchy and ekf 
        memset(ekf_true_relative_state_history, 0, max_num_steps*n*sizeof(double));
        memset(cauchy_true_relative_state_history, 0, max_num_steps*n*sizeof(double));
        ekf_true_relative_state_history[0] = random_normal( 0, sqrt(bsc->E_yt2) );
        ekf_true_relative_state_history[1] = random_normal( 0, sqrt(bsc->E_vt2) );
        cauchy_true_relative_state_history[0] = ekf_true_relative_state_history[0];
        cauchy_true_relative_state_history[1] = ekf_true_relative_state_history[1];
        // Initialize the measurement for the starting state k=0 to be zero (this measurement is unused)
        memset(ekf_msmt_history, 0, max_num_steps*p*sizeof(double));
        memset(cauchy_msmt_history, 0, max_num_steps*p*sizeof(double));
        memset(msmt_noise_history, 0, max_num_steps*p*sizeof(double));

        // Create telegraph wave realization
        double tele_sample;
        double work[3] = {0,0,0};
        rduc.x = work; // not needed
        rduc.step = 0;
        for(int i = 0; i < max_num_steps; i++)
        {
            evader_double_telegraph_transition_model(&rduc, work, &tele_sample);
            evader_telegraph_wave[i] = tele_sample;
            rduc.step += 1;
        }
        cauchy_true_relative_state_history[max_num_steps*n-1] = evader_telegraph_wave[max_num_steps-1];
        ekf_true_relative_state_history[max_num_steps*n-1] = evader_telegraph_wave[max_num_steps-1];
        
        // Create radar measurement noise realization
        rduc.step = 1;
        for(int i = 0; i < sim_num_steps; i++)
        {
            const double t_k = rduc.step * rduc.dt;
            const double psd_v = bsc->R1 + bsc->R2 / pow(bsc->t_f - t_k + 1e-6, 2);
            double V = psd_v / rduc.dt;
            msmt_noise_history[i+1] = random_symmetric_alpha_stable(bsc->RADAR_SAS_PARAM, sqrt(V), 0.0);
            //if(i == 74)
            //    msmt_noise_history[i+1] = 0.95;
            rduc.step += 1;
        }
        rduc.step = 0;
        bsc->counter = 0; // used inside the control law and is incremented after every control
    }

    int step_simulation(double* xk_est, double* u_feedback, KalmanDynamicsUpdateContainer* duc, double* xk1_true, double* w, double* zk1, double* v)
    {
        // Propogate the true relative state dynamics of the evader and pursuer
        int i = rduc.step;
 
        // Take state estimate and form the feedback control
        if(IS_FEEDBACK_STATE_DEPENDENT)
            control_feedback_law(u_feedback, xk_est, n, cmcc, bsc); // State based control
        else
            control_feedback_law(u_feedback, ekf_true_relative_state_history + i*n, n, cmcc, bsc); // Simulation based control
        //u_feedback[0] = 0;

        // Propogate the true state dynamics
        rduc.x = ekf_true_relative_state_history + i*n;
        double u_r = u_feedback[0] - evader_telegraph_wave[i]; // relative acceleration between evader and pursuer
        ekf_true_relative_state_history[i*n + 2] = evader_telegraph_wave[i]; //u_r;
        ekf_control_history[i] = u_feedback[0];
        rduc.u = &u_r;
        gaussian_lti_transition_model(&rduc, ekf_true_relative_state_history + (i+1)*n, &junk_val);
        rduc.step += 1;

        // Form the measurement for the true relative state between the evader and pursuer
        rduc.x = ekf_true_relative_state_history + (i+1)*n;
        double v_k = msmt_noise_history[i+1];
        double t_k = rduc.dt * rduc.step;
        double h_x = atan( rduc.x[0] / (bsc->Vc * (bsc->t_f - t_k + 1e-6)) ); 

        if( (h_x + v_k) > (PI / 2.0) ) // Msmt cannot execeed PI/2.0
        {
            //v_k = PI / 2.0 - h_x;
            //msmt_noise_history[i+1] = v_k;
            zk1[0] = PI / 2.0 - 1e-10;
        }
        else if( (h_x + v_k) < -(PI / 2.0) ) // Msmt cannot execeed -PI/2.0
        {
            //v_k = -PI / 2.0 - h_x;
            //msmt_noise_history[i+1] = v_k;
            zk1[0] = -PI / 2.0 + 1e-10;
        }
        else
            zk1[0] = h_x + v_k; // h(x_bar_k) + v_k

        ekf_msmt_history[i+1] = zk1[0];
        return 0;
    }

    int step_simulation(double* xk_est, double* u_feedback, CauchyDynamicsUpdateContainer* duc, double* xk1_true, double* w, double* zk1, double* v)
    {
        // Propogate the true relative state dynamics of the evader and pursuer
        int i = rduc.step;

        // Take state estimate and form the feedback control
        if(IS_FEEDBACK_STATE_DEPENDENT)
            control_feedback_law(u_feedback, xk_est, n, cmcc, bsc); // State based control
        else
            control_feedback_law(u_feedback, cauchy_true_relative_state_history + i*n, n, cmcc, bsc); // Simulation based control
        //u_feedback[0] = 0;

        // Propogate the true state dynamics
        rduc.x = cauchy_true_relative_state_history + i*n;
        double u_r = u_feedback[0] - evader_telegraph_wave[i]; // relative acceleration between evader and pursuer
        cauchy_true_relative_state_history[i*n + 2] = evader_telegraph_wave[i];
        cauchy_control_history[i] = u_feedback[0];
        rduc.u = &u_r;
        gaussian_lti_transition_model(&rduc, cauchy_true_relative_state_history + (i+1)*n, &junk_val);
        rduc.step += 1;

        // Form the measurement for the true relative state between the evader and pursuer
        rduc.x = cauchy_true_relative_state_history + (i+1)*n;
        double v_k = msmt_noise_history[i+1];
        double t_k = rduc.dt * rduc.step;
        double h_x = atan( rduc.x[0] / (bsc->Vc * (bsc->t_f - t_k + 1e-6)) ); 

        if( (h_x + v_k) > (PI / 2.0) ) // Msmt cannot execeed PI/2.0
        {
            //v_k = PI / 2.0 - h_x;
            //msmt_noise_history[i+1] = v_k;
            zk1[0] = PI / 2.0 - 1e-10;
        }
        else if( (h_x + v_k) < -(PI / 2.0) ) // Msmt cannot execeed -PI/2.0
        {
            //v_k = -PI / 2.0 - h_x;
            //msmt_noise_history[i+1] = v_k;
            zk1[0] = -PI / 2.0 + 1e-10;
        }
        else
            zk1[0] = h_x + v_k; // h(x_bar_k) + v_k

        cauchy_msmt_history[i+1] = zk1[0];
        return 0;
    }

    int reset_counters(void)
    {
        rduc.step = 0;
        bsc->counter = 0;
        return 0;
    }


    ~HomingSimulation()
    {
        free(evader_telegraph_wave);
        free(ekf_true_relative_state_history);
        free(ekf_control_history);
        free(ekf_msmt_history);
        free(cauchy_true_relative_state_history);
        free(cauchy_control_history);
        free(cauchy_msmt_history);
        free(msmt_noise_history);

        free(Phi_r);
        free(B_r);
    }   
};

int count_subdirs_with_prefix(char* path, char* prefix)
{
    DIR *dir;
    struct dirent *entry;
    int count = 0;

    if (!(dir = opendir(path))) {  
        printf("count_subdirs: opendir %s not found! Correct the path! Exiting!\n", path);
        exit(1);
    }
    int len_prefix = strlen(prefix);
    while ((entry = readdir(dir)) != NULL) {  
        char *name = entry->d_name;
        if (entry->d_type == DT_DIR) {
            if (!strcmp(name, ".") || !strcmp(name, ".."))
                continue;
            if( strncmp(prefix, name, len_prefix) == 0 )
                count++;
            //printf("%s/%s\n", path, name);
        }
    }
    printf("\nCounted: %u subdirectories in %s starting with %s\n", count, path, prefix);
    closedir (dir); 
    return count;
}

void test_single_window()
{
    // Cauchy estimator CMD LINE PREAMBLE
    int NUM_WINDOWS = 9;
    double SCALE_BETA = 5.0; // 1.0 seems to be pretty good until the end // 5 (may...) be winner // 25 pretty good // 50 pretty drifty
    double RADAR_SAS_PARAM = 1.3;
    // PRINT OUT SETTINGS 
    printf("Running 3-State Test with Settings:\n");
    printf("Number of windows: %d\n", NUM_WINDOWS);
    printf("BETA SCALE: %lf\n", SCALE_BETA);
    printf("RADAR_SAS_PARAM: %lf\n", RADAR_SAS_PARAM);

    const int n = 3; 
    const int p = 1;
    const int cmcc = 1;
    const int pncc = 1;
    const int sim_num_steps = 99;
    const int total_steps = sim_num_steps+1;
    BallisticMissileConstants bsc(RADAR_SAS_PARAM);
    
    // Arrays for dynamics
    double Phi[n*n], Gamma[n*pncc], B[n*cmcc], H[p*n]; double u_feedback[cmcc];
    const double sigma_w0 = sqrt( (2.0/bsc.tau*bsc.E_at2) / bsc.dt );// std dev of equivalent gaussian process noise
    const double sigma_v0 = sqrt( (bsc.R1 + bsc.R2 / pow(bsc.t_f, 2)) / bsc.dt);// std dev of equivalent gaussian msmt noise

    ///*
    // Arrays For Cauchy Estimator 
    // Initializing the Cauchy Estimator's noises and covariances for the Telegraph Simulation
    int ftr_ordering[3] = {1, 2, 0};
    set_tr_search_idxs_ordering(ftr_ordering, n);
    double xhat_ce[n];
    double x_ce[n];
    double beta[pncc] = {sigma_w0 * GAUSS_TO_CAUCHY_NOISE / SCALE_BETA};
    double gamma[p]; 
    double A0[n*n];
    double p0[n] = { sqrt(bsc.E_yt2) * GAUSS_TO_CAUCHY_NOISE, sqrt(bsc.E_vt2) * GAUSS_TO_CAUCHY_NOISE, sqrt(bsc.E_at2) * GAUSS_TO_CAUCHY_NOISE }; 
    double b0[n] = {0.0, 0.0, 0.0};
    CauchyDynamicsUpdateContainer duc;
        duc.n = n; duc.cmcc = cmcc; 
        duc.p = p; duc.pncc = pncc;
        duc.dt = bsc.dt; duc.step = 1; // step starts at 1 since we assume initial time propogation in the parameters
        duc.Phi = Phi; duc.B = B; duc.Gamma = Gamma; 
        duc.H = H; duc.beta = beta; duc.gamma = gamma;
        duc.x = x_ce;
        duc.u = u_feedback; 
        duc.other_stuff = &bsc;
    assert_correct_cauchy_dynamics_update_container_setup(&duc);
    //*/

    // Create the DynamicSimulation class object
    HomingSimulation hs(total_steps, &bsc);
    // Create temporary work spaces
    double z[p];

    // Seed Trials
    unsigned int seed = 1658778374; //time(NULL); //
    printf("Seeding with %u \n", seed);
    srand ( seed ); //seed // 1658964656 -- no crazy cov error since msmts stay about zero //1658778374 -- a very nice example of EKF vs EMCE // another good example 1658966894

    // Reset the dynamic simulation counters and generate realizations
    hs.reset_counters();
    hs.simulate_telegraph_process_noise_and_measurement_noise();
    memset(u_feedback, 0, cmcc * sizeof(double));

    ///*
    // Reset the Cauchy Estimator
    memset(x_ce, 0, n * sizeof(double));
    memset(xhat_ce, 0, n * sizeof(double));
    init_nonlinear_missile_dynamics(x_ce, Phi, B, Gamma, H, bsc.tau, bsc.Vc, bsc.t_f, bsc.dt, n);
    memcpy(A0, Phi, n*n*sizeof(double));
    reflect_array(A0, n, n); // eye(n) @ Phi.T
    gamma[0] = sigma_v0 * _RADAR_TO_CAUCHY_NOISE;
    duc.step = 1; // we start in the time propagation phase
    
    int foo_steps = NUM_WINDOWS;
    bool print_basic_info = true;
    CauchyEstimator cauchyEst(A0, p0, b0, foo_steps, n, 0, pncc, p, print_basic_info);
    for(int i = 0; i < foo_steps; i++)
    {
        hs.step_simulation(xhat_ce, u_feedback, &duc, NULL, NULL, z, NULL);
        if(i > 0)
            ballistic_missile_nonlinear_radar_full_update_callback(&duc);
        for(int j = 0; j < p; j++)
        {
            double zbar[p];
            ballistic_missile_ece_msmt_model(&duc, zbar);
            ballistic_missile_nonlinear_radar_msmt_update_callback(&duc);
            double msmt = z[j] - zbar[j];
            printf("Processing measurement z=%.4lf, which is #%d/%d at step %d/%d\n", msmt, j+1, p, i+1, foo_steps);
            cauchyEst.step(msmt, Phi, Gamma, beta, H + j*n, gamma[j], NULL, NULL); 
            cauchyEst.finalize_extended_moments(duc.x);

            printf("True State is:\n");
            print_mat(hs.cauchy_true_relative_state_history + (i+1)*n, 1, n);
            printf("Cauchy Conditional Mean is:\n");
            print_mat(duc.x, 1, 3);
        }
    }
}

void test_homing_missile(int argc, char** argv)
{
    char default_log_dir[40] = "../log/homing_missile/default";
    // Cauchy estimator CMD LINE PREAMBLE
    int NUM_WINDOWS = 8;
    char* BASE_LOG_DIR;
    double SCALE_BETA = 5.0; // 1.0 seems to be pretty good until the end // 5 (may...) be winner // 25 pretty good // 50 pretty drifty
    double RADAR_SAS_PARAM = 1.3;
    int num_mc_trials = 3;
    printf("Command Line Arguments are: DEFAULT_NUM_WINDOWS, LOG_DIR_NAME, SCALE_BETA, RADAR_SAS_PARAM, NUM_TRIALS\n");
    
    if(argc == 1)
    {
        printf("No command line arguments given! Using default arguments!\n");
        BASE_LOG_DIR = (char*) malloc(strlen(default_log_dir)+1);
        strcpy(BASE_LOG_DIR, default_log_dir);
    }
    else if(argc == 6)
    {
      // Set number of windows
      NUM_WINDOWS = atoi(argv[1]);
      assert( (NUM_WINDOWS >= 2) && (NUM_WINDOWS <= 10));
      // Set log directory 
      BASE_LOG_DIR = (char*) malloc(strlen(argv[2]) + 10);
      null_ptr_check(BASE_LOG_DIR);
      strcpy(BASE_LOG_DIR, argv[2]);
      // Set Beta Scaling
      SCALE_BETA = std::strtod(argv[3],NULL);
      assert(SCALE_BETA > 0 && SCALE_BETA < 200);
      // Set RADAR SAS PARAM
      RADAR_SAS_PARAM = std::strtod(argv[4],NULL);
      assert(RADAR_SAS_PARAM >= 1 && RADAR_SAS_PARAM <= 2);
      // Set number fo trials
      num_mc_trials = std::atoi(argv[5]);
      assert( (num_mc_trials > 0) && (num_mc_trials < 20000) );
    }
    else
    {
        printf("Too many cmd line arguments given! Provide (up to): DEFAULT_NUM_WINDOWS LOG_DIR_PATH_PREFIX SCALE_BETA RADAR_SAS_PARAM NUM_TRIALS\n");
        exit(1);
    }
    // PRINT OUT SETTINGS 
    printf("Running 3-State Test with Settings:\n");
    printf("Number of windows: %d\n", NUM_WINDOWS);
    printf("BETA SCALE: %lf\n", SCALE_BETA);
    printf("RADAR_SAS_PARAM: %lf\n", RADAR_SAS_PARAM);
    printf("LOG_PATH_PREFIX: %s\n", BASE_LOG_DIR);

    // To avoid re-logging old trials, find the number of subdirectories, and use this count as the base integer for new trials
    int len_mc_subdir = strlen(BASE_LOG_DIR) + 100;
    char* mc_subdir_base = (char*) malloc( len_mc_subdir );
    null_ptr_check(mc_subdir_base);
    char mc_dir_prefix[100];
    sprintf(mc_dir_prefix, "w%d_bs%d_sas%d", NUM_WINDOWS, (int)SCALE_BETA, (int)(10*RADAR_SAS_PARAM));
    if(BASE_LOG_DIR[strlen(BASE_LOG_DIR)-1] == '/')
        sprintf(mc_subdir_base, "%s%s", BASE_LOG_DIR, mc_dir_prefix );
    else
        sprintf(mc_subdir_base, "%s/%s", BASE_LOG_DIR, mc_dir_prefix );
    check_dir_and_create(mc_subdir_base);

    char mtc_prefix[4] = "mct";
    int mc_start_idx = count_subdirs_with_prefix(mc_subdir_base, mtc_prefix) + 1;

    const int n = 3; 
    const int p = 1;
    const int cmcc = 1;
    const int pncc = 1;
    const int sim_num_steps = 99;
    const int total_steps = sim_num_steps+1;
    BallisticMissileConstants bsc(RADAR_SAS_PARAM);
    
    // Arrays for dynamics
    double Phi[n*n], Gamma[n*pncc], B[n*cmcc], H[p*n]; double u_feedback[cmcc];
    
    // Arrays for Kalman Filter
    double kf_state_history[(sim_num_steps+1)*n];
    double kf_covar_history[(sim_num_steps+1)*n*n];
    double kf_Ks_history[(sim_num_steps)*n*p];
    double kf_residual_history[sim_num_steps*p];
    const double sigma_w0 = sqrt( (2.0/bsc.tau*bsc.E_at2) / bsc.dt );// std dev of equivalent gaussian process noise
    const double sigma_v0 = sqrt( (bsc.R1 + bsc.R2 / pow(bsc.t_f, 2)) / bsc.dt);// std dev of equivalent gaussian msmt noise
    double V[p*p]; // = {pow(sigma_v*_RADAR_TO_GAUSS_NOISE,2)};
    double W[pncc*pncc] = {sigma_w0*sigma_w0};
    double P_kf[n*n];
    double K[n*p];
    double x_kf[n];
    KalmanDynamicsUpdateContainer kduc;
        kduc.n = n; kduc.pncc = pncc; 
        kduc.cmcc = cmcc; kduc.p = p;
        kduc.dt = bsc.dt; kduc.step = 0;
        kduc.Phi = Phi; kduc.Gamma = Gamma; 
        kduc.B = B; kduc.H = H;
        kduc.W = W; kduc.V = V;
        kduc.x = x_kf;
        kduc.u = u_feedback;
        kduc.other_stuff = &bsc;

    ///*
    // Arrays For Cauchy Estimator 
    // Initializing the Cauchy Estimator's noises and covariances for the Telegraph Simulation
    int ftr_ordering[3] = {1, 2, 0};
    set_tr_search_idxs_ordering(ftr_ordering, n);
    double xhat_ce[n];
    double x_ce[n];
    double beta[pncc] = {sigma_w0 * GAUSS_TO_CAUCHY_NOISE / SCALE_BETA};
    double gamma[p]; 
    double A0[n*n];
    double p0[n] = { sqrt(bsc.E_yt2) * GAUSS_TO_CAUCHY_NOISE, sqrt(bsc.E_vt2) * GAUSS_TO_CAUCHY_NOISE, sqrt(bsc.E_at2) * GAUSS_TO_CAUCHY_NOISE }; 
    double b0[n] = {0.0, 0.0, 0.0};
    CauchyDynamicsUpdateContainer duc;
        duc.n = n; duc.cmcc = cmcc; 
        duc.p = p; duc.pncc = pncc;
        duc.dt = bsc.dt; duc.step = 1; // step starts at 1 since we assume initial time propogation in the parameters
        duc.Phi = Phi; duc.B = B; duc.Gamma = Gamma; 
        duc.H = H; duc.beta = beta; duc.gamma = gamma;
        duc.x = x_ce;
        duc.u = u_feedback; 
        duc.other_stuff = &bsc;
    assert_correct_cauchy_dynamics_update_container_setup(&duc);
    //*/

    // Create the DynamicSimulation class object
    HomingSimulation hs(total_steps, &bsc);
    // Create temporary work spaces
    double work[n*n];
    double work2[n*n];
    double z[p];

    // Seed Trials
    unsigned int seed = time(NULL); //1658778374
    printf("Seeding with %u \n", seed);
    srand ( seed ); //seed // 1658964656 -- no crazy cov error since msmts stay about zero //1658778374 -- a very nice example of EKF vs EMCE // another good example 1658966894
    
    // Create subdirectory for this monte carlo trial 
    char* mc_sub_dir = (char*) malloc( strlen(mc_subdir_base) + 10 );
    null_ptr_check(mc_sub_dir);
    // Create path array to log control dependent data
    char* sim_logpath = (char*) malloc( strlen(mc_subdir_base) + 250 );
    null_ptr_check(sim_logpath);
    for(int mc_trial = 0; mc_trial < num_mc_trials; mc_trial++)
    {   
        printf("MC Trial %d/%d:\n", mc_trial+1, num_mc_trials);

        // Set mc_sub_dir for this trial 
        sprintf(mc_sub_dir, "%s/%s%d", mc_subdir_base, mtc_prefix, mc_start_idx + mc_trial);
        check_dir_and_create(mc_sub_dir);

        // Reset the dynamic simulation counters and generate realizations
        hs.reset_counters();
        hs.simulate_telegraph_process_noise_and_measurement_noise();
        memset(u_feedback, 0, cmcc * sizeof(double));

        // Log Process and Measurement Noise realizations 
        sprintf(sim_logpath, "%s/msmt_noises.txt", mc_sub_dir);
        log_double_array_to_file(sim_logpath, hs.msmt_noise_history, total_steps, p);
        sprintf(sim_logpath, "%s/proc_noises.txt", mc_sub_dir);
        log_double_array_to_file(sim_logpath, hs.evader_telegraph_wave, sim_num_steps, cmcc);

        // Reset the Kalman Filter
        memset(x_kf, 0, n * sizeof(double));
        memset(P_kf, 0, n * n * sizeof(double));
        P_kf[0] = bsc.E_yt2; P_kf[4] = bsc.E_vt2; P_kf[8] = bsc.E_at2;
        V[0] = pow(sigma_v0 *_RADAR_TO_GAUSS_NOISE,2);
        memcpy(kf_state_history, x_kf, n*sizeof(double));
        memcpy(kf_covar_history, P_kf, n*n*sizeof(double));
        init_nonlinear_missile_dynamics(x_kf, Phi, B, Gamma, H, bsc.tau, bsc.Vc, bsc.t_f, bsc.dt, n);
        kduc.step = 0;

        // Loop Kalman Filter
        for(int i = 0; i < sim_num_steps; i++)
        {
            // Step the true underlying simulation one step, given we have the new control (which is dependent on the state estimate)
            hs.step_simulation(x_kf, u_feedback, &kduc, NULL, NULL, z, NULL);
            // Single step the extended Kalman filter
            extended_kalman_filter(u_feedback, z,
                x_kf, P_kf, K,
                Phi, B, Gamma, 
                H, W, V,
                n, cmcc, pncc, p, 
                work, work2,
                NULL,
                &ballistic_missile_ekf_msmt_model,
                NULL,
                &ballistic_missile_nonlinear_radar_msmt_update_callback,
                &kduc);
            // Store results of the KF simulation (x_kf, P, K, r)
            memcpy(kf_state_history + (i+1)*n, x_kf, n*sizeof(double) );
            memcpy(kf_covar_history + (i+1)*n*n, P_kf, n*n*sizeof(double) );
            memcpy(kf_Ks_history + i*n*p, K, n*p*sizeof(double) );
            memcpy(kf_residual_history + i*p, work2, p*sizeof(double) );
        }
        // Log true states dependent on controller for EKF
        sprintf(sim_logpath, "%s/kf_with_controller_true_states.txt", mc_sub_dir);
        log_double_array_to_file(sim_logpath, hs.ekf_true_relative_state_history, total_steps, n);
        sprintf(sim_logpath, "%s/kf_with_controller_msmts.txt", mc_sub_dir);
        log_double_array_to_file(sim_logpath, hs.ekf_msmt_history, total_steps, p);
        sprintf(sim_logpath, "%s/kf_controls.txt", mc_sub_dir);
        log_double_array_to_file(sim_logpath, hs.ekf_control_history, sim_num_steps, cmcc);

        // Log Kalman Filter
        log_kf_data(
            mc_sub_dir, 
            kf_state_history, 
            kf_covar_history, 
            kf_residual_history, 
            total_steps, n, p);
        
        // Reset counters for EMCE
        hs.reset_counters();
        memset(u_feedback, 0, cmcc * sizeof(double));

        ///*
        // Reset the Cauchy Estimator
        memset(x_ce, 0, n * sizeof(double));
        memset(xhat_ce, 0, n * sizeof(double));
        init_nonlinear_missile_dynamics(x_ce, Phi, B, Gamma, H, bsc.tau, bsc.Vc, bsc.t_f, bsc.dt, n);
        memcpy(A0, Phi, n*n*sizeof(double));
        reflect_array(A0, n, n); // eye(n) @ Phi.T
        gamma[0] = sigma_v0 * _RADAR_TO_CAUCHY_NOISE;
        duc.step = 1; // we start in the time propagation phase
        const bool WINDOW_PRINT_DEBUG = false;
        const bool WINDOW_LOG_SEQUENTIAL = false;
        const bool WINDOW_LOG_FULL = false;
        const bool is_extended = true;
        double* window_var_boost = NULL;
        SlidingWindowManager swm(NUM_WINDOWS, sim_num_steps, A0, p0, b0, &duc, 
            WINDOW_PRINT_DEBUG, WINDOW_LOG_SEQUENTIAL, WINDOW_LOG_FULL, 
            is_extended, 
            ballistic_missile_nonlinear_radar_full_update_callback,
            ballistic_missile_ece_msmt_model, 
            ballistic_missile_nonlinear_radar_msmt_update_callback, 
            window_var_boost, mc_sub_dir);
        // Loop EMCE
        for(int i = 0; i < sim_num_steps; i++)
        {
            // Step the true underlying simulation one step, given we have the new control (which is dependent on the state estimate)
            hs.step_simulation(xhat_ce, u_feedback, &duc, NULL, NULL, z, NULL);
            // Step the sliding window manager
            swm.step(z, u_feedback);
            // Store best estimate at current step in xhat_ce
            memcpy(xhat_ce, swm.full_window_means + (swm.msmt_count-1) * n, n * sizeof(double) );
        }
        swm.shutdown(); // logs EMCE data

        // Log true states dependent on controller for EMCE
        sprintf(sim_logpath, "%s/cauchy_with_controller_true_states.txt", mc_sub_dir);
        log_double_array_to_file(sim_logpath, hs.cauchy_true_relative_state_history, total_steps, n);
        sprintf(sim_logpath, "%s/cauchy_with_controller_msmts.txt", mc_sub_dir);
        log_double_array_to_file(sim_logpath, hs.cauchy_msmt_history, total_steps, p);
        sprintf(sim_logpath, "%s/cauchy_controls.txt", mc_sub_dir);
        log_double_array_to_file(sim_logpath, hs.cauchy_control_history, sim_num_steps, cmcc);
        //*/
    }
    
    // Free Path Variables
    free(BASE_LOG_DIR);
    free(mc_subdir_base);
    free(mc_sub_dir);
    free(sim_logpath);
}

int main(int argc, char** argv)
{
    //test_single_window();
    test_homing_missile(argc, argv);
    return 0;
}