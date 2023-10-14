#ifndef _CAUCHY_WINDOWS_H_
#define _CAUCHY_WINDOWS_H_

#include "cauchy_constants.hpp"
#include "cauchy_estimator.hpp"
#include "cauchy_types.hpp"
#include "cpu_linalg.hpp"
#include "cpu_timer.hpp"
#include "dynamic_models.hpp"

//#include <cstdio>
//#include <sys/types.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <sys/wait.h>


enum ACK_MSGS{SLAVE_SEG_FAULT, SLAVE_OKAY, SLAVE_WINDOW_FULL};

// Measurements, controls, are serialized before sending
struct MsmtMessage
{
    // Serialized data to be sent / recieved
    char* serial_data; 
    int size_bytes_serial_data;
    // Message is serialized as follows:
    // int num_msmts
    // int num_controls
    // double* msmts (of length num_msmts)
    // double* u_feedback (of length num_controls)
    int num_msmts;
    int num_controls;
    double* msmts;
    double* u_feedback;

    void init(int _num_msmts, int _num_controls)
    {
        num_msmts = _num_msmts;
        num_controls = _num_controls;
        msmts = (double*) malloc( num_msmts * sizeof(double) );
        null_ptr_check(msmts);
        u_feedback = (double*) malloc( num_controls * sizeof(double));
        if(num_controls != 0)
            null_ptr_check(u_feedback);
        const int int_size = sizeof(int);
        size_bytes_serial_data = 2 * int_size + num_msmts * sizeof(double) + num_controls * sizeof(double);
        serial_data = (char*) malloc( size_bytes_serial_data );
        null_ptr_check(serial_data);
        memcpy(serial_data, &num_msmts, sizeof(int));
        memcpy(serial_data + int_size, &num_controls, sizeof(int));
    }

    void serialize_data()
    {
        int ser_byte_count = 2*sizeof(int);
        memcpy(serial_data + ser_byte_count, msmts, num_msmts * sizeof(double));
        ser_byte_count += num_msmts * sizeof(double);
        memcpy(serial_data + ser_byte_count, u_feedback, num_controls * sizeof(double));
    }

    void deserialize_data()
    {
        int ser_byte_count = 0;
        int _num_msmts;
        int _num_controls;
        memcpy(&_num_msmts, serial_data, sizeof(int) );
        ser_byte_count += sizeof(int);
        memcpy(&_num_controls, serial_data + ser_byte_count, sizeof(int) );
        ser_byte_count += sizeof(int);
        // Need to make sure thse match (for now)
        if(_num_msmts != num_msmts)
        {
            printf(RED "[ERROR in MsmtMessage::deserialize_data] Variable number of measurements is not implemented!\n"
                   RED " Constructed MsmtMessage for %d msmts, got %d. This can be fixed by changing source code!\n"
                   RED " Exiting for now! Goodbye!"
                   NC "\n", num_msmts, _num_msmts);
            exit(1);
        }
        if(_num_controls != num_controls)
        {
            printf(RED "[ERROR in MsmtMessage::deserialize_data] Variable number of controls is not implemented!\n"
                   RED " Constructed MsmtMessage for %d controls, got %d. This can be fixed by changing source code!\n"
                   RED " Exiting for now! Goodbye!"
                   NC "\n", num_controls, _num_controls);
            exit(1);
        }
        memcpy(msmts, serial_data + ser_byte_count, num_msmts * sizeof(double));
        ser_byte_count += num_msmts * sizeof(double);
        memcpy(u_feedback, serial_data + ser_byte_count, num_controls * sizeof(double));
        ser_byte_count += num_controls * sizeof(double);
        if(ser_byte_count != size_bytes_serial_data)
        {
            printf(RED "[ERROR in MsmtMessage::deserialize_data] Serialized byte count does not match expected byte count!\n"
                   RED " Constructed MsmtMessage for %d bytes, got %d. This can be fixed by changing source code!\n"
                   RED " Exiting for now! Goodbye!"
                   NC "\n", size_bytes_serial_data, ser_byte_count);
            exit(1);
        }

    }

    void deinit()
    {
        free(serial_data);
        free(msmts);
        free(u_feedback);
    }


};

struct WindowMessage
{
    char* serial_data;
    int size_bytes_serial_data;

    int state_dim; // Dimension of state vector
    bool is_extended; // whether or not the cauchy is running in extended mode

    // Serialization order is as below
    int win_num; // Window Number
    int numeric_moment_errors; // bits indicate stability results of moments generated
    enum ACK_MSGS ack_msg; // Slave Seg Fault, Slave Processed Measurement, Slave Reports Estimate
    C_COMPLEX_TYPE fz; // Normalization factor
    C_COMPLEX_TYPE* x_hat; // Conditional Mean (size is state_dim)
    C_COMPLEX_TYPE* P_hat; // Conditional Variance (size is state dim*state_dim))
    double* x_bar; // For Extended Cauchy Estimation
    
    void init(int _win_num, int _state_dim, bool _is_extended)
    {
        win_num = _win_num;
        state_dim = _state_dim;
        is_extended = _is_extended;
        numeric_moment_errors = 0;
        size_bytes_serial_data = 2*sizeof(int) + sizeof(ACK_MSGS) + sizeof(C_COMPLEX_TYPE) + 
                                 sizeof(C_COMPLEX_TYPE) * state_dim + sizeof(C_COMPLEX_TYPE) * state_dim * state_dim;
        if(is_extended)
            size_bytes_serial_data += sizeof(double) * state_dim;
        else
            x_bar = NULL;

        serial_data = (char*) malloc( size_bytes_serial_data );
        null_ptr_check(serial_data);
        fz = 0;
        x_hat = (C_COMPLEX_TYPE*) calloc( state_dim , sizeof(C_COMPLEX_TYPE) );
        null_ptr_check(x_hat);
        P_hat = (C_COMPLEX_TYPE*) calloc( state_dim * state_dim , sizeof(C_COMPLEX_TYPE) );
        null_ptr_check(P_hat);
        if(is_extended)
        {
            x_bar = (double*) calloc( state_dim , sizeof(double) );
            null_ptr_check(x_bar);
        }
    }
    
    void serialize_data()
    {
        int ser_byte_count = 0;
        memcpy(serial_data + ser_byte_count, &win_num, sizeof(int));
        ser_byte_count += sizeof(int);
        memcpy(serial_data + ser_byte_count, &numeric_moment_errors, sizeof(int));
        ser_byte_count += sizeof(int);
        memcpy(serial_data + ser_byte_count, &ack_msg, sizeof(ACK_MSGS));
        ser_byte_count += sizeof(ACK_MSGS);
        memcpy(serial_data + ser_byte_count, &fz, sizeof(C_COMPLEX_TYPE));
        ser_byte_count += sizeof(C_COMPLEX_TYPE);
        memcpy(serial_data + ser_byte_count, x_hat, state_dim*sizeof(C_COMPLEX_TYPE));
        ser_byte_count += state_dim*sizeof(C_COMPLEX_TYPE);
        memcpy(serial_data + ser_byte_count, P_hat, state_dim*state_dim*sizeof(C_COMPLEX_TYPE));
        ser_byte_count += state_dim*state_dim*sizeof(C_COMPLEX_TYPE);
        if(is_extended)
        {
            memcpy(serial_data + ser_byte_count, x_bar, state_dim*sizeof(double));
            ser_byte_count += state_dim*sizeof(double);
        }
        if(ser_byte_count != size_bytes_serial_data)
        {
            printf(RED "[ERROR in WindowMessage::serialize_data] Serialized byte count does not match expected byte count!\n"
                   RED " Constructed WindowMessage for %d bytes, got %d when serializing. This can be fixed by changing source code!\n"
                   RED " Exiting for now! Goodbye!"
                   NC "\n", size_bytes_serial_data, ser_byte_count);
            exit(1);
        }
    }

    void deserialize_data()
    {
        int ser_byte_count = 0;
        memcpy(&win_num, serial_data + ser_byte_count, sizeof(int));
        ser_byte_count += sizeof(int);
        memcpy(&numeric_moment_errors, serial_data + ser_byte_count, sizeof(int));
        ser_byte_count += sizeof(int);
        memcpy(&ack_msg, serial_data + ser_byte_count, sizeof(ACK_MSGS));
        ser_byte_count += sizeof(ACK_MSGS);
        memcpy(&fz, serial_data + ser_byte_count, sizeof(C_COMPLEX_TYPE));
        ser_byte_count += sizeof(C_COMPLEX_TYPE);
        memcpy(x_hat, serial_data + ser_byte_count, state_dim*sizeof(C_COMPLEX_TYPE));
        ser_byte_count += state_dim*sizeof(C_COMPLEX_TYPE);
        memcpy(P_hat, serial_data + ser_byte_count, state_dim*state_dim*sizeof(C_COMPLEX_TYPE));
        ser_byte_count += state_dim*state_dim*sizeof(C_COMPLEX_TYPE);
        if(is_extended)
        {
            memcpy(x_bar, serial_data + ser_byte_count, state_dim*sizeof(double));
            ser_byte_count += state_dim*sizeof(double);
        }
        if(ser_byte_count != size_bytes_serial_data)
        {
            printf(RED "[ERROR in WindowMessage::deserialize_data] Serialized byte count does not match expected byte count!\n"
                   RED " Constructed WindowMessage for %d bytes, got %d when deserializing. This can be fixed by changing source code!\n"
                   RED " Exiting for now! Goodbye!"
                   NC "\n", size_bytes_serial_data, ser_byte_count);
            exit(1);
        }
    }

    void deinit()
    {
        free(serial_data);
        free(x_hat);
        free(P_hat);
        if(is_extended)
            free(x_bar);
    }


};

struct WindowInitializationMessage
{
    char* serial_data;
    int size_bytes_serial_data;

    int state_dim; // Dimension of state vector
    bool is_extended; // whether or not the cauchy is running in extended mode

    // Serialization is constructed as follows    
    double* A_0;
    double* p_0;
    double* b_0;
    double* x; // For nonlinear systems, this is x_bar (k|k-1). For LTV systems, this is x_hat (k|k)

    void init(int _state_dim, int _is_extended)
    {
        state_dim = _state_dim;
        is_extended = _is_extended;

        size_bytes_serial_data = state_dim*state_dim*sizeof(double) + 2*state_dim*sizeof(double) + state_dim*sizeof(double);
        serial_data = (char*) malloc(size_bytes_serial_data);
        null_ptr_check(serial_data);
        A_0 = (double*) malloc(state_dim * state_dim * sizeof(double));
        null_ptr_check(A_0);
        p_0 = (double*) malloc(state_dim * sizeof(double));
        null_ptr_check(p_0);
        b_0 = (double*) malloc(state_dim * sizeof(double));
        null_ptr_check(b_0);
        x = (double*) malloc(state_dim * sizeof(double));
        null_ptr_check(x);
    }

    void serialize_data()
    {
        int ser_byte_count = 0;
        memcpy(serial_data + ser_byte_count, A_0, state_dim*state_dim*sizeof(double));
        ser_byte_count += state_dim*state_dim*sizeof(double);
        memcpy(serial_data + ser_byte_count, p_0, state_dim*sizeof(double));
        ser_byte_count += state_dim*sizeof(double);
        memcpy(serial_data + ser_byte_count, b_0, state_dim*sizeof(double));
        ser_byte_count += state_dim*sizeof(double);
        memcpy(serial_data + ser_byte_count, x, state_dim*sizeof(double));
        ser_byte_count += state_dim*sizeof(double);
        if(ser_byte_count != size_bytes_serial_data)
        {
            printf(RED "[ERROR in WindowInitializationMessage::serialize_data] Serialized byte count does not match expected byte count!\n"
                   RED " Constructed WindowInitializationMessage for %d bytes, got %d when serializing. This can be fixed by changing source code!\n"
                   RED " Exiting for now! Goodbye!"
                   NC "\n", size_bytes_serial_data, ser_byte_count);
            exit(1);
        }
    }

    void deserialize_data()
    {
        int ser_byte_count = 0;
        memcpy(A_0, serial_data + ser_byte_count, state_dim*state_dim*sizeof(double));
        ser_byte_count += state_dim*state_dim*sizeof(double);
        memcpy(p_0, serial_data + ser_byte_count, state_dim*sizeof(double));
        ser_byte_count += state_dim*sizeof(double);
        memcpy(b_0, serial_data + ser_byte_count, state_dim*sizeof(double));
        ser_byte_count += state_dim*sizeof(double);
        memcpy(x, serial_data + ser_byte_count, state_dim*sizeof(double));
        ser_byte_count += state_dim*sizeof(double);
        if(ser_byte_count != size_bytes_serial_data)
        {
            printf(RED "[ERROR in WindowInitializationMessage::deserialize_data] Serialized byte count does not match expected byte count!\n"
                   RED " Constructed WindowInitializationMessage for %d bytes, got %d when deserializing. This can be fixed by changing source code!\n"
                   RED " Exiting for now! Goodbye!"
                   NC "\n", size_bytes_serial_data, ser_byte_count);
            exit(1);
        }
    }

    void deinit()
    {
        free(serial_data);
        free(A_0);
        free(p_0);
        free(b_0);
        free(x);
    }

};

struct CauchyWindow
{
    int window_number; // window index
    //int device_number; // gpu dev number
    int pid; // process id
    int p2c_fd; // reading pipe
    int c2p_fd; // writing pipe
    int msmt_count; // counter to keep track of how many msmts have been processed
    int win_size; // How many windows are in the bank (equivalently, how many msmts a window processes before restarting)
}; 

struct CauchyWindowManager
{
    int num_windows; // Number of Windows To Manage
    int** window_p2c_fds; // parent to child communication channel. File descriptors (fd)
    int** window_c2p_fds; // child to parent communication channel. File descriptors (fd)
    //int num_gpus_available; // This will setup how many GPUs run each window -- currently deprecated


};

// Speyer's initialization routine assumes that Var is a positive definite matrix
// This is made sure of by the numerical checks each window conducts on its computed covariance matrix
// It may be the case that Var has very small eigenvalues, and is very close to being positive semidefinite
// It may also be the case that Var was computed to have positive eigenvalues, but due to eigenvalue solver instability ...
// ... they may come out (small) negative or zero now when checked in this function
// If any eigenvalue of the covariance is detected to be less than COV_EIGENVALUE_TOLERANCE ...
// ... the diagonal of the covariance gets boosted by COV_EIGENVALUE_TOLERANCE - min(eig(Var))
// On enterance of this function, if window_var_boost != NULL ... 
// ... the diagonal of the covariance is boosted by this vector. This helps make the covariance less correlated (more Pos Def) for initialization
// This function could also implement other boosting strategies, such as multiplicative boosting (this is not yet implemented, however).
void speyers_window_init(const int N, double* x1_hat, double* Var, 
                  double* H, double gamma, double z_1, 
                  double* A_0, double* p_0, double* b_0, 
                  const int window_initializer_idx, const int window_initializee_idx, double* window_var_boost)
{
    double work[N*N];
    double work2[N*N];
    double eigs[N];
    memset(eigs, 0, N*sizeof(double));

    // Boost variance on enterance to this function, if window_var_boost != NULL
    bool print_var = false;
    if(window_var_boost != NULL)
        for(int i = 0; i < N; i++)
            Var[i*N+i] += window_var_boost[i];
    
    // Check for positive definiteness of Var
    memcpy(work, Var, N*N*sizeof(double));
    memset(work2, 0, N*sizeof(double));
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', N, work, N, work2);
    double min_eig = work2[0];
    for(int i = 0; i < N; i++)
    {    
	    if(work2[i] < COV_EIGENVALUE_TOLERANCE)
        {    
            printf(YEL "[WARN (Window manager) Speyer Initialization:] Initializer: Win %d, Initializee: Win %d, eig[%d]=%.3E of Covariance very small or negative!"
                NC "\n", window_initializer_idx, window_initializee_idx, i, work2[i]);
            print_var = true;
        }
        if( work2[i] < min_eig )
	  min_eig = work2[i];
    }
    if(print_var)
    {
      double min_eig = array_min(work2, N);
      double min_boost = COV_EIGENVALUE_TOLERANCE - min_eig;
      printf("Eig-Values of Covar:\n");
      print_mat(work2, 1, N, 16);
      printf("Covar of initializer is\n");
      print_mat(Var, N, N, 16);
      printf("Making Covariance more positive definate for initialization!\n");
      for(int i = 0; i < N; i++)
          Var[i*N+i] += min_boost;
    }

    // Begin Speyer Initialization
    inner_mat_prod(H, work, 1, N);
    matmatmul(Var, work, work2, N, N, N, N);
    matmatmul(work2, Var, work, N, N, N, N);
    double Hx1_hat = dot_prod(H, x1_hat, N);
    double scale = gamma * gamma + (z_1 - Hx1_hat) * (z_1 - Hx1_hat);
    scale_mat(work, 1.0 / scale, N, N);
    add_mat(A_0, Var, work, 1.0, N, N);

    // var_work2 = Var + Var @ H^T @ H @ Var / ( gamma^2 + (z_1 - H @ x1_hat)^2)

    // Form A_0 using lapack's (d)ouble (sy)metric (ev)al/vec routine 
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', N, A_0, N, eigs);

    // Form b_0
    double scale2 = (z_1 - Hx1_hat) / scale;
    matvecmul(Var, H, work, N, N);
    scale_mat(work, scale2, N, N);
    sub_vecs(x1_hat, work, b_0, N);

    // Form p_0
    matvecmul(A_0, H, work2, N, N, true); // H @ A = A_T @ H_T
    matvecmul(Var, H, work, N, N);
    double HVarH_T = dot_prod(H, work, N);
    double scale3 = (scale + HVarH_T) / gamma;
    for(int i = 0; i < N; i++)
    {
        p_0[i] = eigs[i] / scale3; // p_i_bar / h_i_bar
        p_0[i] *= work2[i]; // p_i_bar
        p_0[i] /= sgn(work2[i]); // p_i
    }
    reflect_array(A_0,N,N);
}

void print_window_settings(CauchyWindow* cw)
{
    char* str = (char*) malloc(4096);
    sprintf(str, "Hello from Window Number: %d\n  pid: %d\n  p2c_fd: %d, c2p_fd: %d\n  win_size: %d\n", cw->window_number, cw->pid, cw->p2c_fd, cw->c2p_fd, cw->win_size);
    printf("%s\n", str);
    free(str);
}

void init_cauchy_window(CauchyWindow* cw, int window_number, int pid, int* p2c_fd, int* c2p_fd, int num_windows)
{
    // This needs to change such that the manager already knows num_gpus vs. num_windows 
    // The manager can then divide up the work amongst all gpus
    // cw->device_number = 0; (currently deprecated)
    cw->window_number = window_number;
    cw->pid = pid;
    cw->p2c_fd = p2c_fd[0]; // child needs read side of p2c pipe
    cw->c2p_fd = c2p_fd[1]; // child needs write side of c2p pipe
    close(p2c_fd[1]); // child closes write side of p2c pipe 
    close(c2p_fd[0]); // child closes read side of c2p pipe
    cw->win_size = num_windows;
}

// Forks and sets up each window
bool init_cauchy_windows(CauchyWindowManager* cwm, CauchyWindow* cw)
{
    int num_windows = cwm->num_windows;
    bool am_i_manager = true;
    for(int win_num = 0; win_num < num_windows; win_num++)
    {
        if(am_i_manager)
        {
            int pid = fork();
            if(pid == 0) // form return 0 for the child proccess
            {
                pid = getpid();
                init_cauchy_window(cw, win_num, pid, cwm->window_p2c_fds[win_num], cwm->window_c2p_fds[win_num], num_windows);
                return !am_i_manager; // child exit
            }
            else
            {   
                close(cwm->window_p2c_fds[win_num][0]); // Parent closes the read side side of the p2c pipe
                close(cwm->window_c2p_fds[win_num][1]); // Parent closes the write side of the c2p pipe
            }
        }
    }
    return am_i_manager; // parent exit
}

void init_cauchy_window_manager(CauchyWindowManager* cwm, int num_windows)
{
    assert( num_windows > 1 );
    cwm->window_c2p_fds = (int**) malloc(num_windows * sizeof(int*));
    cwm->window_p2c_fds = (int**) malloc(num_windows * sizeof(int*));
    for(int i = 0; i < num_windows; i++)
    {
        cwm->window_c2p_fds[i] = (int*) malloc(2 * sizeof(int));
        cwm->window_p2c_fds[i] = (int*) malloc(2 * sizeof(int));
    }
    cwm->num_windows = num_windows;
    for(int i = 0; i < num_windows; i++)
    {
        pipe(cwm->window_p2c_fds[i]);
        pipe(cwm->window_c2p_fds[i]);
    }
}


int child_window_loop(CauchyWindow cw, 
        int num_simulation_steps,
        double* A0, double* p0, double* b0,
        CauchyDynamicsUpdateContainer* duc, 
        bool WINDOW_PRINT_DEBUG,
        bool is_extended, 
        void (*dynamics_update_callback)(CauchyDynamicsUpdateContainer* _duc), 
        void (*nonlinear_msmt_model)(CauchyDynamicsUpdateContainer* _duc, double* _zbar),
        void (*extended_msmt_update_callback)(CauchyDynamicsUpdateContainer* _duc)
    )
{

    //printf("Child %d has pid %d: Sleeping for GDB attach!\n", cw.window_number, cw.pid);
    //sleep(60);
    //printf("Child %d is done waiting!\n", cw.window_number);
    
    // Adding a step counter variable;
    assert(duc->step == 0 || duc->step == 1);
    int step_count = duc->step;
    const int n = duc->n;
    const int p = duc->p;
    const int pncc = duc->pncc;
    const int cmcc = duc->cmcc;
    double z_bar[p]; // extended helper: computes \bar{z}_{k+1} = h(\bar{x}_{k+1})
    double x_bar[n]; // extended helper: needs to be sent to the empty window, if we are the full window

    const int num_windows = cw.win_size;

    // Print Child Window
    print_window_settings(&cw);
    
    // Setup the Estimator 
    CauchyEstimator cauchyEst(A0, p0, b0, num_windows, n, pncc, p, false);
    cauchyEst.set_win_num(cw.window_number);

    WindowMessage win_msg;
    win_msg.init(cw.window_number, duc->n, is_extended);
    WindowInitializationMessage init_msg;
    init_msg.init(duc->n, is_extended);
    MsmtMessage msmt_msg;
    msmt_msg.init(duc->p, duc->cmcc);
    duc->u = msmt_msg.u_feedback;


    // read in dispatcher's measurements
    //double msmt;
    int msmt_count = 0;
    int win_msmt_count = 0; // resets when it hits num_windows
    bool is_first_loop = true;
    while(msmt_count < num_simulation_steps)
    {
        // Reset the ACK -- tells manager we are okay and to keep the msmts coming
        win_msg.ack_msg = SLAVE_OKAY;

        // Read in the msmt from the window manager
        read(cw.p2c_fd, msmt_msg.serial_data, msmt_msg.size_bytes_serial_data);
        msmt_msg.deserialize_data();

        if(WINDOW_PRINT_DEBUG)
        {
            printf("Child %d recieved msmt # %d: ", cw.window_number, msmt_count);
            for(int i = 0; i < p; i++)
                printf("%.4lf, ", msmt_msg.msmts[i]);
            printf("\n");
        }

        // Only Window #0 reports until first loop is over
        if(is_first_loop)
        {
            msmt_count += 1;
            if(msmt_count > cw.window_number)
            {
                // Increase window measurement count
                win_msmt_count += 1;
                if(WINDOW_PRINT_DEBUG)
                    printf("FIRST LOOP: Window %d has count %d\n", cw.window_number, win_msmt_count);

                // Update the dynamics and system matrices if window_msmt_count != 1
                // This is because either we are the initializee in the speyer start case, or we do not yet have an estimate to linearize around
                if( win_msmt_count != 1)
                {
                    if(dynamics_update_callback != NULL)
                    {
                        if(is_extended)
                            duc->is_xbar_set_for_ece = false;
                        else
                            convert_complex_array_to_real(cauchyEst.conditional_mean, duc->x, n); // this is already set in the extended case
                        duc->step = step_count; // step_count is k+1 here 
                        dynamics_update_callback(duc); // \bar{x}_{k+1} = f(x_k,u_k), with construction of Phi_k, Gamma_k, H_{k+1}, beta_k, gamma_{k+1}
                        if(is_extended)
                            assert(duc->is_xbar_set_for_ece);
                    }
                    // If we have a Linear System, and cmcc is larger than 0, we need to shift the CF by the control bias
                    // If we have a nonlinear system, this is done above within dynamics_update_callback, and becomes apart of \bar{x}_{k+1} = f(x_k,u_k)
                    if(!is_extended && (cmcc > 0) )
                    {
                        double bias[n];
                        matvecmul(duc->B, duc->u, bias, n, cmcc);
                        cauchyEst.shift_cf_by_bias(bias);
                    }
                }
                
                bool speyer_init_was_run = false;
                // If we are a window other than win_num 0 (which is the initiallizer of the other windows)
                if( (cw.window_number != 0) && (win_msmt_count == 1) )
                {
                    if(WINDOW_PRINT_DEBUG)
                        printf("Window %d Waiting on new speyer start params!\n", cw.window_number);
                    // Wait on pipe again for master to provide the A_0, p_0, b_0 which will recreate window 0's (the largest window) estimate with a 1-step CF
                    read(cw.p2c_fd, init_msg.serial_data, init_msg.size_bytes_serial_data);
                    init_msg.deserialize_data();

                    if(WINDOW_PRINT_DEBUG)
                        printf("Window %d reports: Parent sent new A/p/b speyer start params!\n", cw.window_number);
                    // reset the cDynam's to reflect these new parameters
                    cauchyEst.reinitialize_start_statistics(init_msg.A_0, init_msg.p_0, init_msg.b_0);
                    duc->step = step_count;
                    // If Non-linear (is_extended), init_msg.x contains x_bar
                    if(is_extended)
                        memcpy( duc->x, init_msg.x, n * sizeof(double) );
                    // If LTV, init_msg.x contains x_hat
                    if(!is_extended && dynamics_update_callback != NULL)
                    {
                        memcpy( duc->x, init_msg.x, n * sizeof(double) );
                        dynamics_update_callback(duc); // Possibly update Phi, Gamma, beta, H, gamma (only H and gamma will be used on this step tho)
                    }
                    speyer_init_was_run = true;
                }

                int window_numeric_errors;
                for(int i = speyer_init_was_run ? p-1 : 0; i < p; i++)
                {
                    if(is_extended)
                    {
                        if( i == (p-1) )
                            memcpy(x_bar, duc->x, n*sizeof(double)); // this only needs to be run on the last measurement update
                        (*nonlinear_msmt_model)(duc, z_bar); // duc->x == x_bar on i==0 and x_hat on i>0
                        msmt_msg.msmts[i] -= z_bar[i];
                        extended_msmt_update_callback(duc);
                    }
                    window_numeric_errors = cauchyEst.step(msmt_msg.msmts[i], duc->Phi, duc->Gamma, duc->beta, duc->H + i*n, duc->gamma[i] );
                    // Shifts bs in CF by -\delta{x_k}. Sets conditional_mean=\delta{x_k} + duc->x (which is x_bar). Then sets (duc->x) x_bar = creal(conditional_mean)
                    if(is_extended)
                        cauchyEst.finalize_extended_moments(duc->x); 
                }
                if(speyer_init_was_run)
                    cauchyEst.master_step = p;

                
                // Check the estimates after speyer's init (only if print debug is on and we just initialized an estimate)
                if(WINDOW_PRINT_DEBUG && speyer_init_was_run)
                {
                    printf("(This should match FL) Window %d Conditional Mean/Variance:\n", cw.window_number);
                    cauchyEst.print_conditional_mean_variance();
                    //print_conditional_mean_variance(cauchyEst.conditional_mean, cauchyEst.conditional_variance, n);
                }
                
                // If we are window 0, write estimate to standard out
                if(cw.window_number == 0)
                {
                    if(WINDOW_PRINT_DEBUG)
                    {
                        printf("FL: Window %d Conditional Mean/Variance:\n", cw.window_number);
                        cauchyEst.print_conditional_mean_variance();
                        //print_conditional_mean_variance(cauchyEst.conditional_mean, cauchyEst.conditional_variance, n);
                    }
                    // Fill msg with the estimates for the manager 
                    win_msg.ack_msg = SLAVE_WINDOW_FULL; // tag means this windows estimate should be reported
                }

                // In this new implementation, all windows send their conditional information to the window manager
                win_msg.numeric_moment_errors = window_numeric_errors;
                win_msg.fz = cauchyEst.fz;
                memcpy(win_msg.x_hat, cauchyEst.conditional_mean, n*sizeof(C_COMPLEX_TYPE));
                memcpy(win_msg.P_hat, cauchyEst.conditional_variance, n*n*sizeof(C_COMPLEX_TYPE));
                if(is_extended)
                    memcpy(win_msg.x_bar, x_bar, n*sizeof(double));

                // If we are full, report and reset 
                if(win_msmt_count == num_windows)
                {
                    win_msmt_count = 0;
                    // reset estimator
                    cauchyEst.reset();
                }
            }
            else 
            {
                // lag one step
            }
            
            if(msmt_count == num_windows )
            {
                is_first_loop = false;
            }

        }
        // Regular looping, each window's win_msmt_count is lagged one from the others
        else 
        {
            // Increment win_msmt_count, msmt count
            msmt_count += 1;
            win_msmt_count += 1;
            if(WINDOW_PRINT_DEBUG)
                printf("REG LOOP: Window %d has count %d\n", cw.window_number, win_msmt_count);

            // Update the dynamics and system matrices if window_msmt_count != 1
            // This is because either we are the initializee in the speyer start case, or we do not yet have an estimate to linearize around
            if( win_msmt_count != 1)
            {
                if(dynamics_update_callback != NULL)
                {
                    if(is_extended)
                        duc->is_xbar_set_for_ece = false;
                    convert_complex_array_to_real(cauchyEst.conditional_mean, duc->x, n);
                    duc->step = step_count; // step_count is k+1 here 
                    dynamics_update_callback(duc); // \bar{x}_{k+1} = f(x_k,u_k), with construction of Phi_k, Gamma_k, H_{k+1}, beta_k, gamma_{k+1}
                    if(is_extended)
                        assert(duc->is_xbar_set_for_ece);
                }
                // If we have a Linear System, and cmcc is larger than 0, we need to shift the CF by the control bias
                // If we have a nonlinear system, this is done above within dynamics_update_callback, and becomes apart of x_bar = f(x_k,u_k)
                if(!is_extended && (cmcc > 0) )
                {
                    double bias[n];
                    matvecmul(duc->B, duc->u, bias, n, cmcc);
                    cauchyEst.shift_cf_by_bias(bias);
                }
            }
            // If we wish to use speyer's reinitialization method
            bool speyer_init_was_run = false;
            if(win_msmt_count == 1)
            {
                if(WINDOW_PRINT_DEBUG)
                    printf("Window %d Waiting on new speyer start params!\n", cw.window_number);
                // Wait on pipe again for master to provide the A_0, p_0, b_0 which will recreate window 0's (the largest window) estimate with a 1-step CF
                read(cw.p2c_fd, init_msg.serial_data, init_msg.size_bytes_serial_data);
                init_msg.deserialize_data();

                if(WINDOW_PRINT_DEBUG)
                    printf("Window %d reports: Parent sent new A/p/b speyer start params!\n", cw.window_number);
                // reset the cDynam's to reflect these new parameters
                cauchyEst.reinitialize_start_statistics(init_msg.A_0, init_msg.p_0, init_msg.b_0);
                duc->step = step_count;
                // If Non-linear (is_extended), init_msg.x contains x_bar
                if(is_extended)
                    memcpy( duc->x, init_msg.x, n * sizeof(double) );
                // If LTV, init_msg.x contains x_hat
                if(!is_extended && dynamics_update_callback != NULL)
                {
                    memcpy( duc->x, init_msg.x, n * sizeof(double) );
                    dynamics_update_callback(duc); // Possibly update Phi, Gamma, beta, H, gamma (only H and gamma will be used on this step tho)
                }
                speyer_init_was_run = true; 
            }

            int window_numeric_errors;
            for(int i = speyer_init_was_run ? p-1 : 0; i < p; i++)
            {
                if(is_extended)
                {
                    if( ( i == (p-1) ) && ( win_msmt_count == num_windows ) )
                        memcpy(x_bar, duc->x, n*sizeof(double)); // this only needs to be run on the last measurement update
                    (*nonlinear_msmt_model)(duc, z_bar); // duc->x == x_bar on i==0 and x_hat on i>0
                    msmt_msg.msmts[i] -= z_bar[i];
                    extended_msmt_update_callback(duc);
                }
                window_numeric_errors = cauchyEst.step(msmt_msg.msmts[i], duc->Phi, duc->Gamma, duc->beta, duc->H + i*n, duc->gamma[i] );
                // Shifts bs in CF by -\delta{x_k}. Sets conditional_mean=\delta{x_k} + duc->x (which is x_bar). Then sets (duc->x) x_bar = creal(conditional_mean)
                if(is_extended)
                    cauchyEst.finalize_extended_moments(duc->x);
            }
            if(speyer_init_was_run)
                    cauchyEst.master_step = p;

            // Check the estimates after speyer's init (only if print debug is on and we just initialized an estimate)
            if(WINDOW_PRINT_DEBUG && speyer_init_was_run)
            {
                printf("(This should match the Full Window) Window %d Conditional Mean/Variance:\n", cw.window_number);
                cauchyEst.print_conditional_mean_variance();
            }

            // If we are full, report and reset
            if(win_msmt_count == num_windows)
            {
                if(WINDOW_PRINT_DEBUG)
                {
                    printf("Window number %d is full!\n", cw.window_number);
                    printf("Window %d Conditional Mean/Variance:\n", cw.window_number);
                    cauchyEst.print_conditional_mean_variance();
                }
                // Fill msg with the estimates for the manager 
                win_msg.ack_msg = SLAVE_WINDOW_FULL; // tag means this windows estimate should be reported
                // reset the estimator here
                win_msmt_count = 0;
                cauchyEst.reset();
            }
            // In this new implementation, all windows send their conditional information to the window manager
            win_msg.numeric_moment_errors = window_numeric_errors;
            win_msg.fz = cauchyEst.fz;
            memcpy(win_msg.x_hat, cauchyEst.conditional_mean, n*sizeof(C_COMPLEX_TYPE));
            memcpy(win_msg.P_hat, cauchyEst.conditional_variance, n*n*sizeof(C_COMPLEX_TYPE));
            if(is_extended)
                memcpy(win_msg.x_bar, x_bar, n*sizeof(double));
        }
        step_count++;
        // Write back to the parent the msg and to send the next measurement
        win_msg.serialize_data();
        write(cw.c2p_fd, win_msg.serial_data, win_msg.size_bytes_serial_data);
    }
    printf("Simulation has finished! Shutting down Window (child) %d\n", cw.window_number);
    // closing p2c and c2p pipe 
    close(cw.p2c_fd);
    close(cw.c2p_fd);
    // Free up Message Structures
    win_msg.deinit();
    init_msg.deinit();
    msmt_msg.deinit();
    return -1; // use return code -1 to indicate a child is exiting and the process should be killed upon return
}


struct WinCountStruct
{
    int win_idx;
    int win_count;
};

// Descending order sort
int compare_func_win_counts(const void* p1, const void* p2)
{
    WinCountStruct* w1 = (WinCountStruct*) p1;
    WinCountStruct* w2 = (WinCountStruct*) p2;
    return w2->win_count -  w1->win_count;
}

// dynamics_update_callback: updates Phi, Gamma, beta, H, gamma (and possibly B), propagates x_hat to x_bar...This is used on a TP/MU step of the filter in extended mode
// nonlinear_msmt_model: calls z_bar = h(x_bar), this is run both on a TP/MU and solely MU step of the filter in extended mode
// extended_msmt_update_callback: updates H, gam on a solely MU step of the filter in extended mode
// Notes: If we have nonlinear dynamics (_dynamics_update_callback != NULL), then both _nonlinear_msmt_model and _extended_msmt_update_callback must be set, or both must not be set
//      : If we do not have nonlinear dynamics (_dynamics_update_callback == NULL), then both _nonlinear_msmt_model and _extended_msmt_update_callback must be set in order to run in extended mode
//      : If the problem is Linear Time Variant, then _nonlinear_msmt_model == NULL, and _dynamics_update_callback and/or _extended_msmt_update_callback should be set, but is not enforced. 
//      : If the problem is Linear Time Invariant, none of these callbacks need to be set
struct SlidingWindowManager
{
    // Manager Internal Variables
    void (*dynamics_update_callback)(CauchyDynamicsUpdateContainer*) = NULL;
    void (*nonlinear_msmt_model)(CauchyDynamicsUpdateContainer*, double* _zbar) = NULL;
    void (*extended_msmt_update_callback)(CauchyDynamicsUpdateContainer*) = NULL;
    CauchyWindowManager cwm;
    CauchyWindow cw;
    WindowMessage win_msg;
    WindowInitializationMessage init_msg;
    MsmtMessage msmt_msg;
    int window_initializer_idx; // Window index whose CF is a function of the most mesurements
    int window_initializee_idx; // Window index whose CF was just reset, and thus needs the start params of the fullest window (to recreate its estimate with 1 msmt)

    // Launch Parameters
    int num_windows;
    int num_sim_steps;
    double* A0;
    double* p0; 
    double* b0;
    int n;
    int p;
    int cmcc;
    int pncc;

    CauchyDynamicsUpdateContainer* duc;
    bool WINDOW_PRINT_DEBUG;
    bool WINDOW_LOG_SEQUENTIAL;
    bool WINDOW_LOG_FULL;
    bool is_extended;
    bool am_i_manager;
    int msmt_count;

    // Logging variables / variables to reset empty window
    bool* active_windows;
    int* active_window_counts;
    int* active_window_numeric_errors;

    int* full_window_idxs; // Stores the full window index at each MU, or the window used to report the mean/cov in the case of numeric instability
    double* full_window_means; // stores the full window estimates at each estimation step
    double* full_window_variances;
    double* full_window_norm_factors;
    double* full_window_cerr_means;
    double* full_window_cerr_variances;
    double* full_window_cerr_norm_factors;

    // Used for logging all window information
    int* window_step_counts;
    double** window_means;
    double** window_variances;
    double** window_norm_factors;
    double** window_cerr_means;
    double** window_cerr_variances;
    double** window_cerr_norm_factors;
    double** window_x_bars;

    // Timer for clocking the Sliding Window Approximation hertz rate
    CPUTimer win_tmr;

    // Directory for logging information
    char* log_dir;
    char* win_log_dir;
    // Log File Names for the full windows
    FILE* f_fw_means;
    FILE* f_fw_cerr_means;
    FILE* f_fw_variances;
    FILE* f_fw_cerr_variances;
    FILE* f_fw_norm_factors;
    FILE* f_fw_cerr_norm_factors;

    FILE** f_aw_means;
    FILE** f_aw_cerr_means;
    FILE** f_aw_variances;
    FILE** f_aw_cerr_variances;
    FILE** f_aw_norm_factors;
    FILE** f_aw_cerr_norm_factors;

    // Window Var Boost used to boost up the diagonal of the initializer window's covariance matrix, when initializing the initializee window
    // This is usually set to NULL, however it can be helpful (as seen in the LEO problems)
    double* window_var_boost;

    SlidingWindowManager(
                        const int _num_windows, // number of windows the sliding window manager is to manage
                        const int _num_sim_steps, // number of steps the simulation should be run
                        double* _A0, double* _p0, double* _b0, // Initialization parameters for the first window at the first step
                        CauchyDynamicsUpdateContainer* _duc, // contains pointers to all parameters required to run the estimator
                        const bool _WINDOW_PRINT_DEBUG, // Whether to display useful output at each estimation step
                        const bool _WINDOW_LOG_SEQUENTIAL, // Whether to log as we go, or log all data at the end of the simulation
                        const bool _WINDOW_LOG_FULL, // Whether to log all of the windows information at each step or just the full windows at each step
                        const bool _is_extended, // whether to run in extended mode
                        void (*_dynamics_update_callback)(CauchyDynamicsUpdateContainer* _duc), // used for LTV systems and extended estimator
                        void (*_nonlinear_msmt_model)(CauchyDynamicsUpdateContainer* _duc, double* _zbar), // used for extenedd estimator
                        void (*_extended_msmt_update_callback)(CauchyDynamicsUpdateContainer* _duc), // used for extended estimator
                        double* _window_var_boost, // boosts the diagonal of the variance of the initializee window, reduces initial correlation
                        char* _log_dir // if set to NULL, no logging occurs
                        )
    {
        num_windows = _num_windows;
        num_sim_steps = _num_sim_steps;
        A0 = _A0;
        p0 = _p0;
        b0 = _b0;
        duc = _duc;
        n = duc->n;
        p = duc->p;
        cmcc = duc->cmcc;
        pncc = duc->pncc;
        WINDOW_PRINT_DEBUG = _WINDOW_PRINT_DEBUG;
        WINDOW_LOG_SEQUENTIAL = _WINDOW_LOG_SEQUENTIAL;
        WINDOW_LOG_FULL = _WINDOW_LOG_FULL;
        is_extended = _is_extended;
        window_var_boost = _window_var_boost;
        // Create logging directory
        // If _log_dir == NULL, no logging takes place
        // If _log_dir != NULL and WINDOW_LOG_FULL=true, create a 'win' subdirectory
        if(_log_dir == NULL)
        {
            if( (WINDOW_LOG_FULL == true) || (WINDOW_LOG_SEQUENTIAL == true) )
            {
                printf(RED " Error SlidingWindowManager: Either WINDOW_LOG_FULL or WINDOW_LOG_SEQUENTIAL were set to true, but no log directory was provided (log_dir==NULL was given)!\n"
                RED "Either set WINDOW_LOG_FULL=WINDOW_LOG_SEQUENTIAL=false, or provide a log directory\n"
                RED "Exiting! Please fix!"
                NC "\n");
                exit(1);
            }
        }
        check_make_log_dir( _log_dir); 

        assert_correct_cauchy_dynamics_update_container_setup(duc);
        assert( (duc->step == 0) || (duc->step == 1) );
        memset(&cwm, 0, sizeof(CauchyWindowManager));
        memset(&cw, 0, sizeof(CauchyWindow));
        init_cauchy_window_manager(&cwm, num_windows);
        // If extended is true, x0 = _xbar must not be NULL
        if(is_extended)
        {    
            if(duc->x == NULL)
            {
                printf(RED "[Error SlidingWindowManager:] _xbar (which is x0) must be provided to run in extended mode.\nPlease fix and re-compile! Exiting!\n");
                exit(1);
            }
            // If we do not have nonlinear dynamics, then the nonlinear msmt functions must be set...or else this isnt an "extended problem"
            if( (_dynamics_update_callback == NULL) || (_nonlinear_msmt_model == NULL) || (_extended_msmt_update_callback == NULL) )
            {
                printf(RED "[Error SlidingWindowManager:] is_extended was set to true but not all callbacks necessary are provided!\n " 
                        RED "  Provide callback (_dynamics_update_callback) to update dynamics as x_bar = x_{k+1} = f(x_k,u_k) (...and update Phi_k, Gamma_k, beta_k, H_k, gamma_k!)\n"
                        RED "  If transition dynamics are linear, you need to wrap the linear update x_{k+1} = Phi_k * x_k + B_k * u_k in _dynamics_update_callback!\n"
                        RED "  Provide callback (_nonlinear_msmt_model) to compute z_bar = h(x_k|k-1). If this portion of the nonlinear system is indeed linear (i.e., return z_bar = H_k @ x_k|k-1)!\n"
                        RED "  Provide callback (_extended_msmt_update_callback) to compute H_k = grad(h)_(x_k|k-1), even if this portion of the nonlinear system is indeed linear (i.e., return H_k )!\n"
                        RED "  If no callbacks are needed, turn is_extended=false and run in LTI (or LTV mode). Otherwise, fix this! Consult documentation for further detail. Exiting!\n");
                exit(1);
            }
        }
        else
        {
            if((_nonlinear_msmt_model != NULL) || (_extended_msmt_update_callback != NULL))
            {
                printf(RED "[Error SlidingWindowManager:] LTI/LTV mode requested but _nonlinear_msmt_model != NULL or _extended_msmt_update_callback != NULL. These are not used in LTI/LTV mode, only in extended mode!. Please fix (set to NULL), or reconsider the structure of your program! Exiting!\n");
                exit(1);
            }
            if( _dynamics_update_callback != NULL )
            {
                if(duc->x == NULL)
                {
                    printf(RED "[Error SlidingWindowManager:] LTV mode requested (i.e, some or all callbacks are not NULL), but duc->x == NULL. The memory address of duc->x must be set to a memory space of size equal to the state dimension. Please fix! Exiting!\n");
                    exit(1);
                }
            }
        }
        // If SKIP_LAST_STEP=false, request for the user to turn this to true -- there is no point in doing this if set to false
        if(SKIP_LAST_STEP==false)
        {
            printf(RED "[ERROR: SlidingWindowManager] Boolean SKIP_LAST_STEP=false in cauchy_constants.hpp!\n"
                   RED "This should be set to true when running the sliding window manager!"
                   RED "Please set this variable to true and recompile the program! Exiting!\n");
            exit(1);
        }

        am_i_manager = init_cauchy_windows(&cwm, &cw);
        if(!am_i_manager)
        {
            printf("Child Window %d was created!\n", cw.window_number);
            // Launch Child Process Loop
            child_window_loop(cw, _num_sim_steps,
                _A0, _p0, _b0,
                _duc, 
                _WINDOW_PRINT_DEBUG,
                _is_extended, 
                _dynamics_update_callback, 
                _nonlinear_msmt_model,
                _extended_msmt_update_callback
            );
            printf("Child Window %d exiting! Goodbye!\n", cw.window_number);
            exit(0);
        }
        else
        {
            dynamics_update_callback = _dynamics_update_callback;
            nonlinear_msmt_model = _nonlinear_msmt_model;
            extended_msmt_update_callback = _extended_msmt_update_callback;
            win_msg.init(-1, duc->n, is_extended);
            init_msg.init(duc->n, is_extended);
            msmt_msg.init(duc->p, duc->cmcc);
            duc->u = msmt_msg.u_feedback;

            // Based on window settings, setup logging information here
            active_windows = (bool*) malloc( num_windows * sizeof(bool) );
            null_ptr_check(active_windows);
            active_window_counts = (int*) calloc( num_windows , sizeof(int) );
            null_ptr_check(active_window_counts);
            active_window_numeric_errors = (int*) malloc( num_windows * sizeof(int) );
            null_ptr_check(active_window_numeric_errors);
            
            full_window_idxs = (int*) malloc(num_sim_steps * sizeof(int));
            null_ptr_check(full_window_idxs);
            full_window_means = (double*) malloc( num_sim_steps * n * sizeof(double) );
            null_ptr_check(full_window_means);
            full_window_variances = (double*) malloc( num_sim_steps * n * n * sizeof(double) );
            null_ptr_check(full_window_variances);
            full_window_norm_factors = (double*) malloc( num_sim_steps * sizeof(double) );
            null_ptr_check(full_window_norm_factors);
            full_window_cerr_means = (double*) malloc( num_sim_steps * sizeof(double) );
            null_ptr_check(full_window_cerr_means);
            full_window_cerr_variances = (double*) malloc( num_sim_steps * sizeof(double) );
            null_ptr_check(full_window_cerr_variances);
            full_window_cerr_norm_factors = (double*) malloc( num_sim_steps * sizeof(double) );
            null_ptr_check(full_window_cerr_norm_factors);

            // Used for logging all window information
            window_step_counts = (int*) calloc(num_windows , sizeof(int));
            null_ptr_check(window_step_counts);

            window_means = (double**) malloc( num_windows * sizeof(double*) );
            null_dptr_check((void**) window_means );
            window_variances = (double**) malloc( num_windows * sizeof(double*) );
            null_dptr_check((void**) window_variances );
            window_norm_factors = (double**) malloc( num_windows * sizeof(double*) );
            null_dptr_check((void**) window_norm_factors );
            window_cerr_means = (double**) malloc( num_windows * sizeof(double*) );
            null_dptr_check((void**) window_cerr_means );
            window_cerr_variances = (double**) malloc( num_windows * sizeof(double*) );
            null_dptr_check((void**) window_cerr_variances );
            window_cerr_norm_factors = (double**) malloc( num_windows * sizeof(double*) );
            null_dptr_check((void**) window_cerr_norm_factors );
            if(is_extended)
            {
                window_x_bars = (double**) malloc( num_windows * sizeof(double*) );
                null_dptr_check((void**) window_x_bars );
            }


            for(int i = 0; i < num_windows; i++)
            {
                window_means[i] = (double*) malloc( num_sim_steps * n * sizeof(double) );
                null_ptr_check(window_means[i]);
                window_variances[i] = (double*) malloc( num_sim_steps * n * n * sizeof(double) );
                null_ptr_check(window_variances[i]);
                window_norm_factors[i] = (double*) malloc( num_sim_steps * sizeof(double) );
                null_ptr_check(window_norm_factors[i]);
                window_cerr_means[i] = (double*) malloc( num_sim_steps * sizeof(double) );
                null_ptr_check(window_cerr_means[i]);
                window_cerr_variances[i] = (double*) malloc( num_sim_steps * sizeof(double) );
                null_ptr_check(window_cerr_variances[i]);
                window_cerr_norm_factors[i] = (double*) malloc( num_sim_steps * sizeof(double) );
                null_ptr_check(window_cerr_norm_factors[i]);
                if(is_extended)
                {
                    window_x_bars[i] = (double*) malloc( num_sim_steps * n * sizeof(double) );
                    null_ptr_check(window_x_bars[i]);
                }
            }
            window_initializer_idx = 0; // Window index whose CF is a function of the most mesurements
            window_initializee_idx = 0; // Window index whose CF was just reset

            // Launch window manager
            msmt_count = 0;
            sleep(1);
            printf("Parent has pid: %d\n", (int) getpid() );
            printf("Start your engines, racers...\n"); 
            sleep(1);
            // add tmr here
            win_tmr.tic();
        }
    }


    void step(double* msmts, double* controls)
    {
        // hold the window mean/covariance used to restart the empty window
        double x_hat_restart[n]; 
        double x_bar_restart[n]; 
        double P_hat_restart[n*n];

        // Adding a step counter variable;
        //printf("Parent has pid: %d\n", (int) getpid() );
        //sleep(1);

        if(WINDOW_PRINT_DEBUG)
        {
            for(int j = 0; j < p; j++)
                printf("\nParent Sends Measurements #%d, msmt=%lf\n", msmt_count, msmts[j]);
        }
        memcpy(msmt_msg.msmts, msmts, p*sizeof(double));
        //msmt_msg.num_msmts = duc->p;
        memcpy(msmt_msg.u_feedback, controls, cmcc*sizeof(double));
        //msmt_msg.num_controls = duc->cmcc;
        msmt_msg.serialize_data();
        // Send each window the i-th generated measurements from cDynam
        for(int j = 0; j < num_windows; j++)
            write(cwm.window_p2c_fds[j][1], msmt_msg.serial_data, msmt_msg.size_bytes_serial_data);

        // If we wish to run speyers initialization routine, compute the initializer/initializee window indices
        // In the first step, window 0 uses the initial A/p/b. Only after the first step do we conduct window reinitialization
        if(msmt_count > 0)
        {
            // If the measurement count is less than W, manager uses window 0's mean/covar to create the A/p/b for Window msmt_count-1 (to recreate window 0's mean/covar). 
            if(msmt_count < num_windows)
            {
                // Increment the initializee by 1
                window_initializee_idx += 1;
            }
            else 
            {
                // Configure the initializer and the initializee
                val_swap<int>(&window_initializer_idx, &window_initializee_idx);
                window_initializer_idx += 2;
                if(window_initializer_idx >= num_windows)
                    window_initializer_idx -= num_windows;
            }
        }

        // Now wait for each window to reply with a their msg 
        bool children_okay = true;
        bool child_okay;
        memset(active_windows, 0, num_windows * sizeof(bool));
        for(int j = 0; j < num_windows; j++)
        {   
            // If we are about to read the initializee (i.e j == initializee_idx), we need to skip...
            // ... as the initializee is expecting a write of init_msg (to collect new A/p/b), not a manager read (to assert SLAVE_OKAY and collect moments)
            if( (j == window_initializee_idx) && (msmt_count > 0) ) // (window_initializer_idx > window_initializee_idx)
                continue;

            // Read the SLAVE response
            read(cwm.window_c2p_fds[j][0], win_msg.serial_data, win_msg.size_bytes_serial_data);
            win_msg.deserialize_data();

            if(j <= msmt_count)  
            {
                // Store this window's mean / covariance / normalization factor
                save_window_data(j, win_msg.x_hat, win_msg.P_hat, win_msg.fz, win_msg.x_bar);
                active_windows[j] = true;
                active_window_counts[j] += 1;
                active_window_numeric_errors[j] = win_msg.numeric_moment_errors;
            }
            
            child_okay = ((win_msg.ack_msg == SLAVE_OKAY) || (win_msg.ack_msg == SLAVE_WINDOW_FULL)) ? 1 : 0;
            children_okay *= child_okay;
        }

        // The best window index for reinitialization should be the initializer_idx, but in lieu of numerical error, it could be different
        int best_win_idx = select_best_stable_mean_cov(); 

        // Run Speyers Initialization method to restart the (newly) reset window (initializee) around the estimate of the (newly) full window (initializer)
        if( msmt_count > 0 )
        {
            // Run Speyers start, write out the estimate to initializee, and read back OKAY from initializee
            // Create A_0, p_0, b_0 for window window_initializee_idx
            int bwsc_idx = (window_step_counts[best_win_idx]-1) * n; // best window step count index
            memcpy(x_hat_restart, window_means[best_win_idx] + bwsc_idx, n * sizeof(double));
            memcpy(P_hat_restart, window_variances[best_win_idx] + bwsc_idx*n, n * n * sizeof(double));
            if(is_extended)
                memcpy(x_bar_restart, window_x_bars[best_win_idx] + bwsc_idx, n * sizeof(double));
            
            if(!is_extended)
            {
                if(dynamics_update_callback != NULL)
                {
                    duc->x = x_hat_restart;
                    dynamics_update_callback(duc);
                }
                speyers_window_init(n, x_hat_restart, P_hat_restart, duc->H + (p-1)*n, duc->gamma[p-1], msmt_msg.msmts[p-1], init_msg.A_0, init_msg.p_0, init_msg.b_0, best_win_idx, window_initializee_idx, window_var_boost);
                memcpy(init_msg.x, x_hat_restart, n*sizeof(double));
            }
            else
            {
                duc->x = x_bar_restart;
                // convert x_hat into the differential dx = x_hat - x_bar
                sub_vecs(x_hat_restart, x_bar_restart, n);
                // set x_bar, the step, and update H and gamma
                extended_msmt_update_callback(duc);
                // convert the measurement into the differential dz = z - h(x_bar)
                double z_bar[p];
                (*nonlinear_msmt_model)(duc, z_bar); // z_bar = h(x_bar)
                msmt_msg.msmts[p-1] -= z_bar[p-1];
                speyers_window_init(n, x_hat_restart, P_hat_restart, duc->H + (p-1)*n, duc->gamma[p-1], msmt_msg.msmts[p-1], init_msg.A_0, init_msg.p_0, init_msg.b_0, best_win_idx, window_initializee_idx, window_var_boost);
                // copy x_bar over to the initializee msg
                memcpy(init_msg.x, x_bar_restart, n*sizeof(double));
            }                  
            
            // Write the init_msg to the initializee window 
            init_msg.serialize_data();
            write(cwm.window_p2c_fds[window_initializee_idx][1], init_msg.serial_data, init_msg.size_bytes_serial_data);
            
            // Read from the initializee and make sure its okay (they ack SLAVE_OKAY)
            read(cwm.window_c2p_fds[window_initializee_idx][0], win_msg.serial_data, win_msg.size_bytes_serial_data);
            win_msg.deserialize_data();
            
            // Store this windows first step information
            save_window_data(window_initializee_idx, win_msg.x_hat, win_msg.P_hat, win_msg.fz, win_msg.x_bar);
            active_window_numeric_errors[window_initializee_idx] = win_msg.numeric_moment_errors;
            active_window_counts[window_initializee_idx] += 1;

            child_okay = ((win_msg.ack_msg == SLAVE_OKAY) || (win_msg.ack_msg == SLAVE_WINDOW_FULL)) ? 1 : 0; 
            children_okay *= child_okay;
        }
        // Log the best window data from this time-step, as well as all window data if WINDOWW_LOG_FULL is set to true
        save_best_window_data(best_win_idx);

        msmt_count += 1;
        duc->step++;
        // Assert all children repsonded they are OK
        assert(children_okay);
    }

    void shutdown()
    {
        win_tmr.toc();
        double elapsed_time = ((double)win_tmr.cpu_time_used) / 1000; // seconds
        printf("Parent Reports: The Simulation of %d measurements took %lf seconds; rate = %lf hz\n", num_sim_steps,  elapsed_time, ((double)num_sim_steps) / elapsed_time );

        // Wait for each child to exit
        for(int i = 0; i < num_windows; i++)
        { 
            wait(NULL);
            close(cwm.window_p2c_fds[i][1]);
            close(cwm.window_c2p_fds[i][0]);
        }
        // If user requests batch log configuration, do so here
        if( (!WINDOW_LOG_SEQUENTIAL) && (log_dir != NULL) )
        {
            printf("Logging 'full' (or best) window history in batch mode!\n");
            batch_logger_best_window_history();
        }
        if( (WINDOW_LOG_FULL) && (!WINDOW_LOG_SEQUENTIAL) && (win_log_dir != NULL) )
        {
            printf("Logging all windows history in batch mode!\n");
            batch_logger_all_windows_history();
        }
        printf("All children have exited. Parent says goodbye!\n");
    }

    // Creates the logging directory if one has not been created already
    void check_make_log_dir(char* _log_dir)
    {
        if(_log_dir == NULL)
        {
            log_dir = NULL;
            win_log_dir = NULL;
            return;
        }
        int len_log_dir = strlen(_log_dir);
        if(_log_dir[len_log_dir-1] == '/')
        {
            log_dir = (char*) malloc(len_log_dir * sizeof(char));
            null_ptr_check(log_dir);
            strncpy(log_dir, _log_dir, len_log_dir-1);
            log_dir[len_log_dir-1] = '\0';
            len_log_dir -= 1;
        }
        else 
        {
            log_dir = (char*) malloc( (len_log_dir+1) * sizeof(char));
            null_ptr_check(log_dir);
            strncpy(log_dir, _log_dir, len_log_dir+1);
        }
        check_dir_and_create(log_dir);
        // Create logging file name
        create_full_window_log_files();

        if(WINDOW_LOG_FULL)
        {
            int len_win_log_dir = len_log_dir + 8;
            win_log_dir = (char*) malloc( (len_win_log_dir+1) * sizeof(char) );
            null_ptr_check(win_log_dir);
            sprintf(win_log_dir, "%s/windows", log_dir);
            check_dir_and_create(win_log_dir);
            create_all_window_log_files();
        }
        else 
        {
            win_log_dir = NULL;
        }
    }

    void create_full_window_log_files()
    {
        assert(strlen(log_dir) < 4000);
        char* temp_path = (char*) malloc( 4096 * sizeof(char) );
        null_ptr_check(temp_path);
        sprintf(temp_path, "%s/%s", log_dir, "cond_means.txt");
        f_fw_means = fopen(temp_path, "w");
        if(f_fw_means == NULL)
        {
            printf("cond_means.txt file path has failed! Debug here!\n");
            exit(1);
        }
        sprintf(temp_path, "%s/%s", log_dir, "cerr_cond_means.txt");
        f_fw_cerr_means = fopen(temp_path, "w");
        if(f_fw_cerr_means == NULL)
        {
            printf("cerr_cond_means.txt file path has failed! Debug here!\n");
            exit(1);
        }
        sprintf(temp_path, "%s/%s", log_dir, "cond_covars.txt");
        f_fw_variances = fopen(temp_path, "w");
        if(f_fw_variances == NULL)
        {
            printf("cond_covars.txt file path has failed! Debug here!\n");
            exit(1);
        }
        sprintf(temp_path, "%s/%s", log_dir, "cerr_cond_covars.txt");
        f_fw_cerr_variances = fopen(temp_path, "w");
        if(f_fw_cerr_variances == NULL)
        {
            printf("cerr_cond_covars.txt file path has failed! Debug here!\n");
            exit(1);
        }
        sprintf(temp_path, "%s/%s", log_dir, "norm_factors.txt");
        f_fw_norm_factors = fopen(temp_path, "w");
        if(f_fw_norm_factors == NULL)
        {
            printf("norm_factors.txt file path has failed! Debug here!\n");
            exit(1);
        }
        sprintf(temp_path, "%s/%s", log_dir, "cerr_norm_factors.txt");
        f_fw_cerr_norm_factors = fopen(temp_path, "w");
        if(f_fw_cerr_norm_factors == NULL)
        {
            printf("norm_factors.txt file path has failed! Debug here!\n");
            exit(1);
        }
        free(temp_path);
    }

    void create_all_window_log_files()
    {
        assert(strlen(win_log_dir) < 4000);
        char* temp_win_dir_path = (char*) malloc( 4096 * sizeof(char) );
        null_ptr_check(temp_win_dir_path);
        char* temp_file_path = (char*) malloc( 4096 * sizeof(char) );
        null_ptr_check(temp_file_path);

        f_aw_means = (FILE**) malloc( num_windows * sizeof(FILE*) );
        f_aw_cerr_means = (FILE**) malloc( num_windows * sizeof(FILE*) );
        f_aw_variances = (FILE**) malloc( num_windows * sizeof(FILE*) );
        f_aw_cerr_variances = (FILE**) malloc( num_windows * sizeof(FILE*) );
        f_aw_norm_factors = (FILE**) malloc( num_windows * sizeof(FILE*) );
        f_aw_cerr_norm_factors = (FILE**) malloc( num_windows * sizeof(FILE*) );

        for(int win_idx = 0; win_idx < num_windows; win_idx++)
        {
            // Create window 'win_idx' lod directory in the windows subfolder of log_dir
            sprintf(temp_win_dir_path, "%s/win%d", win_log_dir, win_idx);
            check_dir_and_create(temp_win_dir_path);
            // Create file pointers to each of these directories

            // Condition Means of window win_idx
            sprintf(temp_file_path, "%s/%s", temp_win_dir_path, "cond_means.txt");
            f_aw_means[win_idx] = fopen(temp_file_path, "w");
            if(f_aw_means[win_idx] == NULL)
            {
                printf("cond_means.txt file path for window %d has failed! Debug here!\n", win_idx);
                exit(1);
            }
            // Max error of conditional means of window win_idx
            sprintf(temp_file_path, "%s/%s", temp_win_dir_path, "cerr_cond_means.txt");
            f_aw_cerr_means[win_idx] = fopen(temp_file_path, "w");
            if(f_aw_cerr_means[win_idx] == NULL)
            {
                printf("cerr_cond_means.txt file path for window %d has failed! Debug here!\n", win_idx);
                exit(1);
            }
            // Condition Variances of window win_idx
            sprintf(temp_file_path, "%s/%s", temp_win_dir_path, "cond_covars.txt");
            f_aw_variances[win_idx] = fopen(temp_file_path, "w");
            if(f_aw_variances[win_idx] == NULL)
            {
                printf("cond_covars.txt file path for window %d has failed! Debug here!\n", win_idx);
                exit(1);
            }
            // Max error of conditional variances of window win_idx
            sprintf(temp_file_path, "%s/%s", temp_win_dir_path, "cerr_cond_covars.txt");
            f_aw_cerr_variances[win_idx] = fopen(temp_file_path, "w");
            if(f_aw_cerr_variances[win_idx] == NULL)
            {
                printf("cerr_cond_covars.txt file path for window %d has failed! Debug here!\n", win_idx);
                exit(1);
            }
            // Normalization factors of window win_idx
            sprintf(temp_file_path, "%s/%s", temp_win_dir_path, "norm_factors.txt");
            f_aw_norm_factors[win_idx] = fopen(temp_file_path, "w");
            if(f_aw_norm_factors[win_idx] == NULL)
            {
                printf("norm_factors.txt file path for window %d has failed! Debug here!\n", win_idx);
                exit(1);
            }
            // Complex error of normalization factors of window win_idx
            sprintf(temp_file_path, "%s/%s", temp_win_dir_path, "cerr_norm_factors.txt");
            f_aw_cerr_norm_factors[win_idx] = fopen(temp_file_path, "w");
            if(f_aw_cerr_norm_factors[win_idx] == NULL)
            {
                printf("cerr_norm_factors.txt file path for window %d has failed! Debug here!\n", win_idx);
                exit(1);
            }
        }
        free(temp_win_dir_path);
        free(temp_file_path);
    }

    void sequential_logger_best_window(int best_win_idx)
    {   
        if(log_dir == NULL)
            return;
        log_double_array_to_file(f_fw_means, best_win_idx, full_window_means + msmt_count * n, n);
        log_double_array_to_file(f_fw_variances, best_win_idx, full_window_variances + msmt_count * n * n, n * n);
        log_double_array_to_file(f_fw_norm_factors, best_win_idx, full_window_norm_factors + msmt_count, 1);
        log_double_array_to_file(f_fw_cerr_means, best_win_idx, full_window_cerr_means + msmt_count, 1);
        log_double_array_to_file(f_fw_cerr_variances, best_win_idx, full_window_cerr_variances + msmt_count, 1);
        log_double_array_to_file(f_fw_cerr_norm_factors, best_win_idx, full_window_cerr_norm_factors + msmt_count, 1);
        fflush(f_fw_means);
        fflush(f_fw_variances);
        fflush(f_fw_norm_factors);
        fflush(f_fw_cerr_means);
        fflush(f_fw_cerr_variances);
        fflush(f_fw_cerr_norm_factors);
    }

    void sequential_logger_all_windows()
    {
        if(win_log_dir == NULL)
            return;
        for(int win_idx = 0; win_idx < num_windows; win_idx++)
        {
            int win_step_count = window_step_counts[win_idx]-1;
            if(win_step_count >= 0)
            {
                log_double_array_to_file(f_aw_means[win_idx], window_means[win_idx] + win_step_count * n, n);
                log_double_array_to_file(f_aw_variances[win_idx], window_variances[win_idx] + win_step_count * n * n, n * n);
                log_double_array_to_file(f_aw_cerr_means[win_idx], window_cerr_means[win_idx] + win_step_count, 1);
                log_double_array_to_file(f_aw_cerr_variances[win_idx], window_cerr_variances[win_idx] + win_step_count, 1);
                log_double_array_to_file(f_aw_norm_factors[win_idx], window_norm_factors[win_idx] + win_step_count, 1);
                log_double_array_to_file(f_aw_cerr_norm_factors[win_idx], window_cerr_norm_factors[win_idx] + win_step_count, 1);
                fflush(f_aw_means[win_idx]);
                fflush(f_aw_variances[win_idx]);
                fflush(f_aw_cerr_means[win_idx]);
                fflush(f_aw_cerr_variances[win_idx]);
                fflush(f_aw_norm_factors[win_idx]);
                fflush(f_aw_cerr_norm_factors[win_idx]);
            }
        }
    }
    
    void batch_logger_best_window_history()
    {
        if(log_dir == NULL)
            return;
        for(int i = 0; i < msmt_count; i++)
        {
            int best_win_idx = full_window_idxs[i];
            log_double_array_to_file(f_fw_means, best_win_idx, full_window_means + i * n, n);
            log_double_array_to_file(f_fw_variances, best_win_idx, full_window_variances + i * n * n, n * n);
            log_double_array_to_file(f_fw_norm_factors, best_win_idx, full_window_norm_factors + i, 1);
            log_double_array_to_file(f_fw_cerr_means, best_win_idx, full_window_cerr_means + i, 1);
            log_double_array_to_file(f_fw_cerr_variances, best_win_idx, full_window_cerr_variances + i, 1);
            log_double_array_to_file(f_fw_cerr_norm_factors, best_win_idx, full_window_cerr_norm_factors + i, 1);
        }
        fflush(f_fw_means);
        fflush(f_fw_variances);
        fflush(f_fw_norm_factors);
        fflush(f_fw_cerr_means);
        fflush(f_fw_cerr_variances);
        fflush(f_fw_cerr_norm_factors);
    }

    void batch_logger_all_windows_history()
    {
        if(win_log_dir == NULL)
            return;

        for(int win_idx = 0; win_idx < num_windows; win_idx++)
        {
            int step_count = window_step_counts[win_idx];
            for(int i = 0; i < step_count; i++)
            {
                log_double_array_to_file(f_aw_means[win_idx], window_means[win_idx] + i * n, n);
                log_double_array_to_file(f_aw_variances[win_idx], window_variances[win_idx] + i * n * n, n * n);
                log_double_array_to_file(f_aw_cerr_means[win_idx], window_cerr_means[win_idx] + i, 1);
                log_double_array_to_file(f_aw_cerr_variances[win_idx], window_cerr_variances[win_idx] + i, 1);
                log_double_array_to_file(f_aw_norm_factors[win_idx], window_norm_factors[win_idx] + i, 1);
                log_double_array_to_file(f_aw_cerr_norm_factors[win_idx], window_cerr_norm_factors[win_idx] + i, 1);
            }
            fflush(f_aw_means[win_idx]);
            fflush(f_aw_variances[win_idx]);
            fflush(f_aw_cerr_means[win_idx]);
            fflush(f_aw_cerr_variances[win_idx]);
            fflush(f_aw_norm_factors[win_idx]);
            fflush(f_aw_cerr_norm_factors[win_idx]);
        }
    }
    
    // store window win_idx mean and covariance
    void save_window_data(int win_idx, C_COMPLEX_TYPE* x_hat, C_COMPLEX_TYPE* P_hat, C_COMPLEX_TYPE fz, double* x_bar)
    {
        int wsc = window_step_counts[win_idx]++;
        // Save Norm Factor
        window_norm_factors[win_idx][wsc] = creal(fz);
        window_cerr_norm_factors[win_idx][wsc] = cimag(fz);
        
        // Save Conditional Mean
        double* window_mean = window_means[win_idx] + wsc * n;
        convert_complex_array_to_real(x_hat, window_mean, n);
        window_cerr_means[win_idx][wsc] = max_abs_imag_carray(x_hat, n);
        
        // Save Conditional Variance
        double* window_variance = window_variances[win_idx] + wsc * n * n;
        convert_complex_array_to_real(P_hat, window_variance, n * n);
        window_cerr_variances[win_idx][wsc] =  max_abs_imag_carray(P_hat, n*n);
        
        // Save x_bar if extended
        if(is_extended)
        {
            double* window_x_bar = window_x_bars[win_idx] + wsc * n;
            memcpy( window_x_bar, x_bar, n * sizeof(double) );
        }
    }

    // store the best window (most full or most numerically stable) in the full_window_... variables
    void save_best_window_data(int best_win_idx)
    {
        full_window_idxs[msmt_count] = best_win_idx;
        int best_win_step_count = window_step_counts[best_win_idx]-1;

        // Log best mean/covar at this step
        double* fw_mean = full_window_means + msmt_count * n;
        double* fw_var = full_window_variances + msmt_count * n * n;
        double* w_mean = window_means[best_win_idx] +  best_win_step_count * n;
        double* w_var = window_variances[best_win_idx] + best_win_step_count * n * n;
        memcpy(fw_mean, w_mean, n * sizeof(double));
        memcpy(fw_var, w_var, n * n * sizeof(double));
        // Log error stats of best mean / covar at this step
        full_window_cerr_means[msmt_count] = window_cerr_means[best_win_idx][best_win_step_count];
        full_window_cerr_variances[msmt_count] = window_cerr_variances[best_win_idx][best_win_step_count];
        full_window_norm_factors[msmt_count] = window_norm_factors[best_win_idx][best_win_step_count];
        full_window_cerr_norm_factors[msmt_count] = window_cerr_norm_factors[best_win_idx][best_win_step_count];

        // Log Windowing information if log setting is set to sequential
        if(WINDOW_LOG_SEQUENTIAL)
            sequential_logger_best_window(best_win_idx);

        // Log moment information of all windows at this step if both are set to true
        if(WINDOW_LOG_SEQUENTIAL && WINDOW_LOG_FULL)
            sequential_logger_all_windows();
    }

    int select_best_stable_mean_cov()
    {
        // From the active windows, sort them based on who is most full
        WinCountStruct win_counts[num_windows];
        int num_active_windows = 0;
        for(int i = 0; i < num_windows; i++)
        {
            if(active_windows[i])
            {
                win_counts[num_active_windows].win_idx = i;
                win_counts[num_active_windows].win_count = active_window_counts[i];
                num_active_windows++;
                // Reset the 'full' active window count, which is now stored in win_counts[num_active_windows]
                if(active_window_counts[i] == num_windows)
                    active_window_counts[i] = 0;
            }
        }
        qsort(win_counts, num_active_windows, sizeof(WinCountStruct), compare_func_win_counts);

        // Select best window option to reset the empty window, based on the following:
        // All estimators return tags indicating
        // 1.) Whether the covariance is numerically stable and therefore usable from the set of measurements sent to the estimator
        // 2.) Whether the mean is numerically stable and therefore usable from the set of measurements sent to the estimator
        // The strategy chosen uses the best window to reinitialize the empty window, given a chosen strategy
        int best_win_idx = stratgey_choose_fullest_window_first(win_counts, num_active_windows);

        if(best_win_idx == -1)
        {
            printf(RED "ERROR SLIDING WINDOW MANAGER: in select_best_stable_mean_cov() function, best_win_idx returns as -1. This is unrecoverable! Exiting!\n");
            exit(1);
        }

        return best_win_idx;
    }

    // This windowing strategy abides by the following set of rules
    // Start with the most full window
    // If fz has flipped negative, abandon this window
    // If the current steps covariance error flag is set to DNE (no stable covariances computed), abandon window
    // If the current steps mean error flag is set to DNE (no stable covariances computed), abandon window
    // if we are abandoning a window, print message saying so and go to the next most full window.
    // If every possible window is found unusable, then return -1, indicating the sliding window approximation should be shut down
    int stratgey_choose_fullest_window_first(WinCountStruct* win_counts, const int num_active_windows)
    {
        for(int i = 0; i < num_active_windows; i++)
        {
            const int win_idx = win_counts[i].win_idx;
            const int win_count = win_counts[i].win_count;
            const int win_moment_errors = active_window_numeric_errors[win_idx];

            // Check 1: If fz has flipped negative, abandone
            if(win_moment_errors & (1 << ERROR_FZ_NEGATIVE) )
                continue;
            if(win_moment_errors & (1 << ERROR_COVARIANCE_AT_CURRENT_STEP_DNE) )
            {
                printf(YEL "[Warn Sliding Window Approximation: Window Selection]"
                       YEL "  Window %d, which has undergone %d/%d Time Propagations is unusable at current step due to COVARIANCE DNE ERROR!\n"
                       YEL "  Attempting to use next best window!"
                       NC  "\n", win_idx, win_count, num_windows);
                continue;
            }
            if(win_moment_errors & (1 << ERROR_MEAN_AT_CURRENT_STEP_DNE) )
            {
                printf(YEL "[Warn Sliding Window Approximation: Window Selection]"
                       YEL "  Window %d, which has undergone %d/%d Time Propagations is unusable at current step due to MEAN DNE ERROR!\n"
                       YEL "  Attempting to use next best window!"
                       NC  "\n", win_idx, win_count, num_windows);
                continue;
            }
            if(WINDOW_PRINT_DEBUG)
                printf("Sliding Window Approximation: Window selection stratgey chose window %d, which has %d/%d Time Propagations!\n", win_idx, win_count, num_windows);
            return win_idx;
        }
        return -1;
    }

    // ... Enter other windowing strategies if desired here ... //
    // ...
    // ... //

    ~SlidingWindowManager()
    {
        if(am_i_manager)
        {
            for(int i = 0; i < num_windows; i++)
            {
                free(cwm.window_c2p_fds[i]);
                free(cwm.window_p2c_fds[i]);
            }
            free(cwm.window_c2p_fds);
            free(cwm.window_p2c_fds);
            win_msg.deinit();
            init_msg.deinit();
            msmt_msg.deinit();


            free(active_windows);
            free(active_window_counts);
            free(active_window_numeric_errors);

            free(full_window_idxs);
            free(full_window_means);
            free(full_window_variances);
            free(full_window_norm_factors);
            free(full_window_cerr_means);
            free(full_window_cerr_variances);
            free(full_window_cerr_norm_factors);
            // Used for logging all window information
            free(window_step_counts);
            for(int i = 0; i < num_windows; i++)
            {
                free(window_means[i]);
                free(window_variances[i]);
                free(window_norm_factors[i]);
                free(window_cerr_means[i]);
                free(window_cerr_variances[i]);
                free(window_cerr_norm_factors[i]);
            }
            free(window_means);
            free(window_variances);
            free(window_norm_factors);
            free(window_cerr_means);
            free(window_cerr_variances);
            free(window_cerr_norm_factors);

            if(log_dir != NULL)
            {
                free(log_dir);
                fclose(f_fw_means);
                fclose(f_fw_cerr_means);
                fclose(f_fw_variances);
                fclose(f_fw_cerr_variances);
                fclose(f_fw_norm_factors);
                fclose(f_fw_cerr_norm_factors);
            }
            if(win_log_dir != NULL)
            {
                free(win_log_dir);
                
                for(int i = 0; i < num_windows; i++)
                {
                    fclose(f_aw_means[i]);
                    fclose(f_aw_variances[i]);
                    fclose(f_aw_cerr_means[i]);
                    fclose(f_aw_cerr_variances[i]);
                    fclose(f_aw_norm_factors[i]);
                    fclose(f_aw_cerr_norm_factors[i]);
                }
                free(f_aw_means);
                free(f_aw_variances);
                free(f_aw_cerr_means);
                free(f_aw_cerr_variances);
                free(f_aw_norm_factors);
                free(f_aw_cerr_norm_factors);
            }
        }
    }
};

#endif // _CAUCHY_WINDOWS_H_
