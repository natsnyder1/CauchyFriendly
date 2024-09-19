#ifndef _CAUCHY_ESTIMATOR_HPP_
#define _CAUCHY_ESTIMATOR_HPP_


#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cauchy_util.hpp"
#include "cell_enumeration.hpp"
#include "cauchy_linalg.hpp"
#include "eval_gs.hpp"
#include "gtable.hpp"
#include "random_variables.hpp"
#include "term_reduction.hpp"
#include "flattening.hpp"
#include "cpu_timer.hpp"

// Function/Structure Prototypes
struct CauchyEstimator;
void* distributed_step_tp_to_muc(void* args);


struct DIST_TP_TO_MUC_STRUCT
{
    CauchyEstimator* cauchyEst;
    CoalignmentElemStorage* coalign_store;
    ChunkedPackedTableStorage* gb_tables;
    DiffCellEnumHelper* dce_helper;
    C_COMPLEX_TYPE fz_chunk;
    C_COMPLEX_TYPE* cond_mean_chunk;
    C_COMPLEX_TYPE* cond_var_chunk;
    CauchyTerm** new_terms_dp;
    int* new_terms_per_shape;
    double* Phi;
    double* H;
    double* processed_Gamma;
    double* processed_beta;
    int processed_pncc;
    double msmt;
    double gamma;
    double* B;
    double* u;
    int tid;
    int n_tids;
};

struct CauchyEstimator
{
    int d; // state dimension
    int cmcc; // control matrix column count (columns of B)
    int pncc; // process noise column count (columns of Gamma)
    int p; // number of measurements processed per step
    int Nt; // total number of terms
    int num_estimation_steps; // total number of estimation steps this estimator (window) will conduct
    int master_step; // when multiple measurements are in play, this keep track of the number of times msmt_update is called (child generation)
    CauchyTerm** terms_dp; // terms of the cauchy estimator, organized by shape
    double* A0_init;
    double* p0_init;
    double* b0_init;
    int num_gtable_chunks; 
    int num_btable_chunks;
    int* terms_per_shape;
    int shape_range;
    int* B_dense;
    double* root_point;
    double G_SCALE_FACTOR;
    C_COMPLEX_TYPE* conditional_mean;
    C_COMPLEX_TYPE* conditional_variance;
    C_COMPLEX_TYPE fz;
    FastTermRedHelper ftr_helper; // Fast term reduction
    DiffCellEnumHelper* dce_helper; // DCE method
    ChildTermWorkSpace childterms_workspace; // Used to store the terms temporarily during TP/TPC/MU/MUC
    CoalignmentElemStorage* coalign_store; // Memory manager for storing terms after MUC
    ReductionElemStorage* reduce_store; // Memory manager for storing terms after term reduction
    ChunkedPackedTableStorage* gb_tables; // Memory manager for storing g and b-tables
    CauchyStats stats; // Used to Gather Memory Stats
    bool print_basic_info;
    bool skip_post_mu;
    int num_threads_tp_to_muc;
    int num_threads_make_gtables;
    int win_num;
    int numeric_moment_errors;
    C_COMPLEX_TYPE last_fz;
    C_COMPLEX_TYPE* last_conditional_mean;
    C_COMPLEX_TYPE* last_conditional_variance;

    CauchyEstimator(double* _A0, double* _p0, double* _b0, int _steps, int _d, int _cmcc, int _pncc, int _p, const bool _print_basic_info)
    {
        // Init state dimensions and shape / term counters
        Nt = 1;
        master_step = 0;
        d = _d;
        cmcc = _cmcc;
        pncc = _pncc;
        p = _p;
        num_estimation_steps = p*_steps;
        int max_hp_shape = d > 1 ? (_steps-1) * pncc + d : d + pncc;
        shape_range = max_hp_shape + 1;
        terms_per_shape = (int*) calloc(shape_range , sizeof(int));
        null_ptr_check(terms_per_shape);
        terms_per_shape[d] = 1;
        terms_dp = (CauchyTerm**) malloc( shape_range*sizeof(CauchyTerm*) );
        null_dptr_check((void**)terms_dp);
        for(int i = 0; i < shape_range; i++)
        {
            if( i == d )
            {
                terms_dp[d] = (CauchyTerm*) malloc((d+1) * sizeof(CauchyTerm));
                null_ptr_check(terms_dp[d]);
            }
            else 
                terms_dp[i] = (CauchyTerm*) malloc(0); 
        }
        // Init Moments
        conditional_mean = (C_COMPLEX_TYPE*) malloc(d * sizeof(C_COMPLEX_TYPE));
        null_ptr_check(conditional_mean);
        conditional_variance = (C_COMPLEX_TYPE*) malloc(d * d * sizeof(C_COMPLEX_TYPE));
        null_ptr_check(conditional_variance);
        last_conditional_mean = (C_COMPLEX_TYPE*) malloc(d * sizeof(C_COMPLEX_TYPE));
        null_ptr_check(last_conditional_mean);
        last_conditional_variance = (C_COMPLEX_TYPE*) malloc(d * d * sizeof(C_COMPLEX_TYPE));
        null_ptr_check(last_conditional_variance);

        // Init Gtable helpers
        root_point = (double*) malloc( d * sizeof(double));
        null_ptr_check(root_point);
        for(int i = 0; i < d; i++)
            root_point[i] = 1.0 + random_uniform();
        if(DENSE_STORAGE)
        {
            int two_to_max_shape = (1<<max_hp_shape);
            B_dense = (int*) malloc(two_to_max_shape * sizeof(int) );
            null_ptr_check(B_dense);
            for(int i = 0; i < two_to_max_shape; i++)
                B_dense[i] = i;
        }
        // Set the function pointers necessary to run all methods
        set_function_pointers();
        error_checks();

        dce_helper = (DiffCellEnumHelper*) malloc( NUM_CPUS * sizeof(DiffCellEnumHelper) );
        null_ptr_check(dce_helper);
        gb_tables = (ChunkedPackedTableStorage*) malloc( NUM_CPUS * sizeof(ChunkedPackedTableStorage) );
        null_ptr_check(gb_tables);
        coalign_store = (CoalignmentElemStorage*) malloc( NUM_CPUS * sizeof(CoalignmentElemStorage) );
        null_ptr_check(coalign_store);
        reduce_store = (ReductionElemStorage*) malloc(NUM_CPUS * sizeof(ReductionElemStorage));
        null_ptr_check(reduce_store);

        dce_helper->init(shape_range-1, d, DCE_STORAGE_MULT);
        gb_tables->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        coalign_store->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        reduce_store->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        // Until needed, initialize the other gbtables / coalign store containers to 0
        for(int i = 1; i < NUM_CPUS; i++)
        {
            dce_helper[i].init(shape_range-1, d, DCE_STORAGE_MULT);
            gb_tables[i].init(0, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
            coalign_store[i].init(0, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
            reduce_store[i].init(0, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        }
        ftr_helper.init(d, 1<<d);
        
        childterms_workspace.init(shape_range-1, d);
        print_basic_info = _print_basic_info;

        // Initialize the first term 
        setup_first_term(&childterms_workspace, terms_dp[d], _A0, _p0, _b0, d);
        // Create memory for storing the initialization parameters / resetting these when the estimator "resets"
        A0_init = (double*) malloc(d * d * sizeof(double)); //_A0;
        null_ptr_check(A0_init);
        memcpy(A0_init, _A0, d * d * sizeof(double));
        p0_init = (double*) malloc(d * sizeof(double)); //_p0;
        null_ptr_check(p0_init);
        memcpy(p0_init, _p0, d * sizeof(double));
        b0_init = (double*) malloc(d * sizeof(double)); //_b0;
        null_ptr_check(b0_init);
        memcpy(b0_init, _b0, d * sizeof(double));

        // Set threading arguments, window number, and initial error code
        num_threads_tp_to_muc = 1;
        num_threads_make_gtables = 1;
        win_num = 0;
        numeric_moment_errors = 0;
    }

    void set_win_num(int _win_num)
    {
        win_num = _win_num;
    }

    void set_function_pointers()
    {
        // THIS NEEDS TO BE RE-IMPLEMENTED
        switch (GTABLE_STORAGE_METHOD)
        {
            case GTABLE_HASHTABLE_STORAGE:
                lookup_g_numerator = (LOOKUP_G_NUMERATOR_TYPE)g_num_hashtable;
                gtable_insert = (GTABLE_INSERT_TYPE) g_insert_hashtable;
                gtable_add = (GTABLE_ADD_TYPE) gs_add_hashtable;
                gtable_p_find = (GTABLE_P_FIND_TYPE) gp_find_hashtable;
                gtable_p_get_keys = (GTABLE_P_GET_KEYS_TYPE) gtable_p_get_keys_hashtable;
                break;
            case GTABLE_BINSEARCH_STORAGE:
                lookup_g_numerator = (LOOKUP_G_NUMERATOR_TYPE)g_num_binsearch;
                gtable_insert = (GTABLE_INSERT_TYPE) g_insert_binsearch;
                gtable_add = (GTABLE_ADD_TYPE) gs_add_binsearch;
                gtable_p_find = (GTABLE_P_FIND_TYPE) gp_find_binsearch;
                gtable_p_get_keys = (GTABLE_P_GET_KEYS_TYPE) gtable_p_get_keys_binsearch;
                break;
            case GTABLE_DENSE_STORAGE:
                lookup_g_numerator = (LOOKUP_G_NUMERATOR_TYPE)g_num_dense;
                gtable_insert = (GTABLE_INSERT_TYPE) g_insert_dense;
                gtable_add = (GTABLE_ADD_TYPE) gs_add_dense;
                gtable_p_find = NULL;
                gtable_p_get_keys = NULL;
                break;
            default:
                printf("CHOSEN GTABLE/BTABLE METHOD NOT IMPLEMENTED YET! EXITING!\n");
                exit(1);
        }
    }

    void error_checks()
    {
        if(GAMMA_PERTURB_EPS <= 0 )
        {
            printf(RED "ERROR: GAMMA_PERTURB_EPS=%lf in cauchy_constants.hpp must be positive!\nExiting!" NC "\n", GAMMA_PERTURB_EPS);
            exit(1);
        }
        if( (shape_range-1) > 31)
        {
            printf(RED "ERROR: Until tested, max HP (%d) shape cannot exceed 31!\nExiting!" NC "\n", shape_range-1);
            exit(1);
        }
        // Make sure the gtables are setup correctly
        if(HASHTABLE_STORAGE)
        {
            if((GTABLE_SIZE_MULTIPLIER < 1))
            {
                printf(RED "ERROR: GTABLE_SIZE_MULTIPLIER in cauchy_types.hpp must be defined as >1 when using HASHTABLE_STORAGE method!\nExiting!" NC "\n");
                exit(1);
            }
            if(GTABLE_SIZE_MULTIPLIER == 1)
            {
                printf(YEL "WARNING! You are using GTABLE_SIZE_MULTIPLIER==1! This is unoptimal! Consider using GTABLE_SIZE_MULTIPLIER>1!" NC "\n");
                sleep(2);
            }
        }
        else
        {
            if(GTABLE_SIZE_MULTIPLIER != 1)
            {
                printf(RED "ERROR: GTABLE_SIZE_MULTIPLIER in cauchy_types.hpp must be defined as 1 when using BINSEARCH_STORAGE or DENSE_STORAGE method!\nExiting!" NC "\n");
                exit(1);
            }
        }
        if(DENSE_STORAGE)
        {
            if(FULL_STORAGE)
            {
                printf(YEL "WARNING! You are using DENSE_STORAGE WITH FULL STORAGE set on! This is VERY expensive! Consider using FULL_STORAGE=false method!" NC "\n");
                sleep(2);
            }
        }
        // Make sure the largest gtable size will fit into a single page
        if(DENSE_STORAGE)
        {
            int max_cell_count = 1 << (shape_range-1);
            BYTE_COUNT_TYPE bytes_max_hp_gtable = (BYTE_COUNT_TYPE)(GTABLE_SIZE_MULTIPLIER * sizeof(GTABLE_TYPE) * max_cell_count / (1 + HALF_STORAGE) + 10); // a little extra padding for floating point calcs
            if( bytes_max_hp_gtable > CP_STORAGE_PAGE_SIZE)
            {
                printf(RED "[Error Cauchy Estimator Initialization:]\nCP_STORAGE_PAGE_SIZE is smaller than the memory required to store the largest HPA's gtable (which requires %llu bytes)\nIncrease CP_STORAGE_PAGE_SIZE in cauchy_constants.hpp!\nExiting!" NC"\n", bytes_max_hp_gtable );
                exit(1);
            }
        }
        else 
        {
            int max_cell_count = cell_count_central(shape_range-1, d);
            BYTE_COUNT_TYPE bytes_max_hp_gtable = (BYTE_COUNT_TYPE)(GTABLE_SIZE_MULTIPLIER * sizeof(GTABLE_TYPE) * max_cell_count / (1 + HALF_STORAGE) + 10); // a little extra padding for floating point calcs
            if( bytes_max_hp_gtable > CP_STORAGE_PAGE_SIZE)
            {
                printf(RED "[Error Cauchy Estimator Initialization:]\nCP_STORAGE_PAGE_SIZE is smaller than the memory required to store the largest HPA's gtable (which requires %llu bytes)\nIncrease CP_STORAGE_PAGE_SIZE in cauchy_constants.hpp!\nExiting!" NC"\n", bytes_max_hp_gtable );
                exit(1);
            }
        }
        // Make sure that the first d+1 gtables will fit into a single page
        BYTE_COUNT_TYPE bytes_first_gtables = (BYTE_COUNT_TYPE)(GTABLE_SIZE_MULTIPLIER * sizeof(GTABLE_TYPE) * (d+1) * (1<<d) / (1 + HALF_STORAGE) + 10); // a little extra padding for floating point calcs
        if(bytes_first_gtables > CP_STORAGE_PAGE_SIZE)
        {
            printf(RED "[Error Cauchy Estimator Initialization:]\nCP_STORAGE_PAGE_SIZE is smaller than the memory required to store the first STATE_DIM+1 gtables after MU#1 (which requires %llu bytes)\nIncrease CP_STORAGE_PAGE_SIZE in cauchy_constants.hpp!\nExiting!" NC"\n", bytes_first_gtables );
            exit(1);
        }
        BYTE_COUNT_TYPE bytes_max_hpa = (shape_range-1) * d * sizeof(double);
        if(bytes_max_hpa > CP_STORAGE_PAGE_SIZE)
        {
            printf(RED "[Error Cauchy Estimator Initialization:]\nCP_STORAGE_PAGE_SIZE is smaller than the memory required to store the largest hyperplane arrangement (which requires %llu bytes)\nIncrease CP_STORAGE_PAGE_SIZE in cauchy_constants.hpp!\nExiting!" NC"\n", bytes_max_hpa );
            exit(1);
        }
        if(DCE_STORAGE_MULT <= 1)
        {
            printf(RED "[Error Cauchy Estimator Initialization:]\nDCE_STORAGE_MULT MUST BE LARGE THAN 1...Set to 2,3,4..etc\n Exiting!" NC"\n");
            exit(1);
        }
    }

    void cache_moments(CauchyTerm* parent, CauchyTerm* children, int num_children)
    {
        C_COMPLEX_TYPE g_val;
		#if _WIN32
			C_COMPLEX_TYPE yei[MAX_HP_SIZE];
		#else 
			C_COMPLEX_TYPE yei[d];
		#endif 

        // Cache parent term
        g_val = parent->eval_g_yei(root_point, yei, false);
        fz += g_val;
        for(int j = 0; j < d; j++)
        {
            C_COMPLEX_TYPE y = yei[j];
            conditional_mean[j] += g_val * y;
            for(int k = 0; k < d; k++)
                conditional_variance[j*d + k] -= g_val * y * yei[k];
        }
        for(int i = 0; i < num_children; i++)
        {
            g_val = children[i].eval_g_yei(root_point, yei, false);
            fz += g_val;
            for(int j = 0; j < d; j++)
            {
                C_COMPLEX_TYPE y = yei[j];
                conditional_mean[j] += g_val * y;
                for(int k = 0; k < d; k++)
                    conditional_variance[j*d + k] -= g_val * y * yei[k];
            }
        }
    }

    void finalize_cached_moments()
    {
        G_SCALE_FACTOR = RECIPRICAL_TWO_PI / creal(fz);
        //C_COMPLEX_TYPE Ifz = I*fz; // imaginary fz
        C_COMPLEX_TYPE Ifz = MAKE_CMPLX(0, creal(fz)); //MAKE_CMPLX(cimag(fz), creal(fz)); // imaginary fz
        for(int i = 0; i < d; i++)
            conditional_mean[i] /= Ifz;

        for(int i = 0; i < d; i++)
        {
            for(int j = 0; j < d; j++)
            {
                conditional_variance[i*d + j] = (conditional_variance[i*d+j] / fz) - conditional_mean[i] * conditional_mean[j];
            }
        }
        // Run numeric checks on fz, conditional mean, conditional variance
        moments_numerical_check();
    }

    void moments_numerical_check()
    {
        // if this is the first measurement update of the current step, 
        bool first_msmt_of_current_step = (master_step % p) == 0;
        bool not_last_msmt_of_current_step = ( (master_step % p) != (p-1) ); // These flags will not be set when p=1 (first msmt == last)
        bool last_msmt_of_current_step = (master_step % p) == (p-1);
        if(first_msmt_of_current_step)
        {
            // Reset the Current Step Error Messages 
            // Do Not Exist (DNE) flags get pulled high
            // Calculating valid mean/covariance information at the current step pulls the DNE flags low
            numeric_moment_errors |= (1<<ERROR_MEAN_AT_CURRENT_STEP_DNE);
            numeric_moment_errors |= (1<<ERROR_COVARIANCE_AT_CURRENT_STEP_DNE);
            // Current step error flags get pulled low
            // Calculating invalid mean/covariance sets these flags to high
            numeric_moment_errors &= ~(1<<ERROR_MEAN_UNSTABLE_CURRENT_STEP_FINAL_MSMT);
            numeric_moment_errors &= ~(1<<ERROR_MEAN_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT);
            numeric_moment_errors &= ~(1<<ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT);
            numeric_moment_errors &= ~(1<<ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_FINAL_MSMT);
        }
        
        // Set new bit flag array to store results of the moments at this estimation step
        int new_numeric_moment_errors = 0;

        // Check fz: For the normalization factor, two checks must be met
        //         : Check 1.) fz must be strictly positive
        //         : Check 2.) Ratio of imaginary to real value must be smaller than the allowable threshold ratio
        if( creal(fz) <= 0 )
            new_numeric_moment_errors |= (1 << ERROR_FZ_NEGATIVE); // catastrohpic. estimator has failed
        if( fabs( cimag(fz) / ( 1e-15 + creal(fz) ) ) > THRESHOLD_FZ_IMAG_TO_REAL )
            new_numeric_moment_errors |= (1 << ERROR_FZ_UNSTABLE);

        bool mean_okay = true;
        // Check mean: For the conditional mean, two checks must be met
        //        : Check 1.) Ratio of all imaginary to real values of the conditional mean must be smaller than the allowable threshold ratio
        //        : Check 2.) All imaginary values of the conditional mean must be smaller than the predefined hard limit
        for(int i = 0; i < d; i++)
        {
            double mean_i_real = fabs( creal(conditional_mean[i]) );
            double mean_i_imag = fabs( cimag(conditional_mean[i]) );
            double ratio = mean_i_imag / (1e-15 + mean_i_real);
            if( (ratio > THRESHOLD_MEAN_IMAG_TO_REAL) || (mean_i_imag > HARD_LIMIT_IMAGINARY_MEAN) )
            {
                new_numeric_moment_errors |= (1<<ERROR_MEAN_UNSTABLE_ANY_STEP);
                if(not_last_msmt_of_current_step)
                    new_numeric_moment_errors |= (1<<ERROR_MEAN_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT);
                if(last_msmt_of_current_step)
                    new_numeric_moment_errors |= (1<<ERROR_MEAN_UNSTABLE_CURRENT_STEP_FINAL_MSMT);
                mean_okay = false;
            }
        }
        bool cov_okay = true;
        // Check variance: For the conditional variance, four checks must be met
        //        : 1.) Eigenvalues must all be positive
        //        : 2.) Correlation matrix must contain values between +/- 1
        //        : 3.) Ratio of all imaginary to real values of the conditional variance must be less than threshold
        //        : 4.) All imaginary parts of the conditional variance must be lower than the predefined hard limit
        int cov_error_flags = covariance_checker(conditional_variance, d, win_num, master_step+1, num_estimation_steps, true);
        if( cov_error_flags )
        {
            new_numeric_moment_errors |= (1<<ERROR_COVARIANCE_UNSTABLE_ANY_STEP);
            if(not_last_msmt_of_current_step)
                new_numeric_moment_errors |= (1<<ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT);
            if(last_msmt_of_current_step)
                new_numeric_moment_errors |= (1<<ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_FINAL_MSMT);
            cov_okay = false;
        }
        // Add new moment errors to the running bit flag array
        numeric_moment_errors |= new_numeric_moment_errors;
        // If the mean / covariance estimates passes checks at the current step, we know there does exists usable moments
        // This implies that the moments may have become numerically unstable at a point this step, but there does exists usable information
        // Its the job of the window manager to decide whether to use this information, if errors have occured
        if(mean_okay)
            numeric_moment_errors &= ~( 1 << ERROR_MEAN_AT_CURRENT_STEP_DNE ); // Set DNE Mean Bit low
        if(cov_okay)
            numeric_moment_errors &= ~( 1 << ERROR_COVARIANCE_AT_CURRENT_STEP_DNE ); // Set DNE Variance Bit low

        if( (new_numeric_moment_errors != 0) && print_basic_info )
        {
            print_current_step_numeric_moment_errors(new_numeric_moment_errors, cov_error_flags);
        }

        // It is the job of the estimator to keep the best information available for the window manager to use
        // If the mean is okay at current step, transfer "conditional_mean" to the "last_conditional_mean" memory space
        if(mean_okay)
            memcpy(last_conditional_mean, conditional_mean, d*sizeof(C_COMPLEX_TYPE));
        // If the mean has become unusable at the current step, transfer "last_conditional_mean" to the "conditional_mean" memory space
        else
            memcpy(conditional_mean, last_conditional_mean, d*sizeof(C_COMPLEX_TYPE));
        // If the covariance is okay at current step, transfer "conditional_variance" to the "last_conditional_varaince" memory space
        if(cov_okay)
            memcpy(last_conditional_variance, conditional_variance, d*d*sizeof(C_COMPLEX_TYPE));
        // If the covariance has become unusable at the current step, transfer "last_conditional_variance" to the "conditional_variance" memory space
        else
            memcpy(conditional_variance, last_conditional_variance, d*d*sizeof(C_COMPLEX_TYPE));
        
        // If both the mean and covariance are not okay, we use purely the last measurements information. Revert fz to last_fz
        if( (!mean_okay) && (!cov_okay) )
            fz = last_fz;
        // Otherwise we are using some of the information from this MU. Store fz into last_fz
        else 
            last_fz = fz;
    }

    void print_current_step_numeric_moment_errors(int current_step_numeric_moment_error_flags, int cov_error_flags)
    {
        printf(YEL "[WARN: Numeric Moment Errors] Window=%d, Step=%d/%d, MU %d/%d\n", win_num, (master_step / p) + 1, num_estimation_steps/p, (master_step % p), p);
        if( current_step_numeric_moment_error_flags & (1 << ERROR_FZ_NEGATIVE) )
        {
            printf(YEL "  ERROR_FZ_NEGATIVE has been set! fz=%.4E + %.4Ej\n", creal(fz), cimag(fz));
        }
        if( current_step_numeric_moment_error_flags & (1 << ERROR_FZ_UNSTABLE) )
        {
            printf(YEL "  ERROR_FZ_UNSTABLE has been set! fz=%.4E + %.4Ej\n", creal(fz), cimag(fz));
        }
        if( current_step_numeric_moment_error_flags & (1 << ERROR_MEAN_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT) )
        {
            printf(YEL "  ERROR_MEAN_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT has been set!\n");
            print_cvec(conditional_mean, d, 4);
        }
        if( current_step_numeric_moment_error_flags & (1 << ERROR_MEAN_UNSTABLE_CURRENT_STEP_FINAL_MSMT) )
        {
            printf(YEL "  ERROR_MEAN_UNSTABLE_CURRENT_STEP_FINAL_MSMT has been set!\n");
            print_cvec(conditional_mean, d, 4);
        }
        if( current_step_numeric_moment_error_flags & (1 << ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT) )
        {
            printf(YEL "  ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_NOT_FINAL_MSMT has been set!\n");
            print_cmat(conditional_variance, d, d, 16);
            print_covariance_error_flags(cov_error_flags);
        }
        if( current_step_numeric_moment_error_flags & (1 << ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_FINAL_MSMT) )
        {
            printf(YEL "  ERROR_COVARIANCE_UNSTABLE_CURRENT_STEP_FINAL_MSMT has been set!\n");
            print_cmat(conditional_variance, d, d, 16);
            print_covariance_error_flags(cov_error_flags);
        }
        printf(NC "\n");
    }

    void print_covariance_error_flags(int cov_error_flags)
    {
        if(cov_error_flags == 0)
            return;
        if(cov_error_flags & (1<<COV_ERROR_FLAGS_INVALID_EIGENVALUES))
            printf("Covariance has invalid (negative) eigenvalues!\n");
        if(cov_error_flags & (1<<COV_ERROR_FLAGS_INVALID_CORRELATION))
            printf("Covariance has invalid correlations!\n");
        if(cov_error_flags & (1<<COV_ERROR_FLAGS_INVALID_I2R_RATIO))
            printf("Covariance maximum imaginary to real ratio is too large!\n");
        if(cov_error_flags & (1<<COV_ERROR_FLAGS_INVALID_IMAGINARY_VALUE))
            printf("Covariances maximum imaginary value is too large!\n");
    }

    void print_conditional_mean_variance()
    {
        const int precision = 16;
        printf("Moment Information (after MU) at step %d, MU %d/%d\n", (master_step+1) / p, (master_step % p)+1, p);
        printf("fz: %.*lf + %.*lfj\n", precision, creal(fz), precision, cimag(fz));
        printf("Conditional Mean:\n");
        print_cmat(conditional_mean, 1, d, precision);
        printf("Conditional Variance:\n");
        print_cmat(conditional_variance, d, d, precision);
    }

    void compute_moments(const bool before_ftr = true)
    {
        if(!INTEGRABLE_FLAG)
        {
            printf(RED "[WARNING COMPUTE MOMENTS:] NON INTEGRABLE FLAG WAS RAISED!\n"
                   RED "THIS INDICATES THAT H IS ORTHOGONAL TO GAMMA.\n"
                   RED "THE CHARACTERISTIC FUNCTION WILL NOT HAVE MOMENTS AT THIS TIME STEP\n"
                   RED "THE MOMENTS SHOULD COME OUT COMPLEX BELOW!"
                   NC "\n");
        }
        CPUTimer tmr;
        tmr.tic();
        bool first_step = (master_step == 0);
        C_COMPLEX_TYPE g_val;
		#if _WIN32 
			C_COMPLEX_TYPE yei[MAX_HP_SIZE];
		#else 
			C_COMPLEX_TYPE yei[d];
		#endif
        fz = 0;
        memset(conditional_mean, 0, d*sizeof(C_COMPLEX_TYPE));
        memset(conditional_variance, 0, d*d*sizeof(C_COMPLEX_TYPE));
        for(int m = 1; m < shape_range; m++)
        {
            int Nt_shape = terms_per_shape[m];
            CauchyTerm* terms = terms_dp[m];
            for(int i = 0; i < Nt_shape; i++)
            {
                if(before_ftr)
                    g_val = terms[i].eval_g_yei(root_point, yei, first_step);
                else
                    g_val = terms[i].eval_g_yei_after_ftr(root_point, yei);
                fz += g_val;
                for(int j = 0; j < d; j++)
                {
                    C_COMPLEX_TYPE y = yei[j];
                    conditional_mean[j] += g_val * y;
                    for(int k = 0; k < d; k++)
                        conditional_variance[j*d + k] -= g_val * y * yei[k];
                }
            }
        }
        assert(creal(fz) > 0);
        G_SCALE_FACTOR = RECIPRICAL_TWO_PI / creal(fz);

        C_COMPLEX_TYPE Ifz = MAKE_CMPLX(0, creal(fz)); //I*fz; //MAKE_CMPLX(cimag(fz), creal(fz)); // imaginary fz
        for(int i = 0; i < d; i++)
            conditional_mean[i] /= Ifz;

        for(int i = 0; i < d; i++)
        {
            for(int j = 0; j < d; j++)
            {
                conditional_variance[i*d + j] = (conditional_variance[i*d+j] / fz) - conditional_mean[i] * conditional_mean[j];
            }
        }
        tmr.toc(false);
        if( before_ftr && print_basic_info )
        {
            const int precision = 16;
            printf("Moment Information (after MU) at step %d, MU %d/%d (took %d ms)\n", (master_step+1) / p, (master_step % p)+1, p, tmr.cpu_time_used);
            printf("fz: %.*lf + %.*lfj\n", precision, creal(fz), precision, cimag(fz));
            printf("Conditional Mean:\n");
            print_cmat(conditional_mean, 1, d, precision);
            printf("Conditional Variance:\n");
            print_cmat(conditional_variance, d, d, precision);
        }
        if( (!before_ftr) && print_basic_info) 
        {
            const int precision = 16;
            printf("Moment Information (after FTR) at step %d, MU %d/%d (took %d ms)\n", (master_step+1) / p, (master_step % p)+1, p, tmr.cpu_time_used);
            printf("fz: %.*lf + %.*lfj\n", precision, creal(fz), precision, cimag(fz));
            printf("Conditional Mean:\n");
            print_cmat(conditional_mean, 1, d, precision);
            printf("Conditional Variance:\n");
            print_cmat(conditional_variance, d, d, precision);
        }
        INTEGRABLE_FLAG = true; // reset
    }

    void step_tp_to_muc(double msmt, double* Phi, double* Gamma, double* beta, double* H, double gamma, double* B, double* u)
    {
		#if _WIN32
			double tmp_Gamma[MAX_HP_SIZE*MAX_HP_SIZE];
			double tmp_beta[MAX_HP_SIZE];
		#else
			double tmp_Gamma[d*pncc];
			double tmp_beta[pncc];
		#endif
		int tmp_pncc = 0;
        const bool with_tp = ((master_step % p) == 0);
        fz = 0;
        memset(conditional_mean, 0, d * sizeof(C_COMPLEX_TYPE));
        memset(conditional_variance, 0, d*d*sizeof(C_COMPLEX_TYPE));
        CPUTimer tmr_mu;
        tmr_mu.tic();
        // Transpose, normalize, and pre-coalign Gamma and beta
        if(with_tp)
            tmp_pncc = precoalign_Gamma_beta(Gamma, beta, pncc, d, tmp_Gamma, tmp_beta);
        
        // Allocate structures for the maximum number of new terms we'd generate at this step
        int* new_terms_per_shape = (int*) calloc(shape_range, sizeof(int));
        null_ptr_check(new_terms_per_shape);
        CauchyTerm* new_child_terms;
        if(skip_post_mu)
            new_child_terms = (CauchyTerm*) malloc(shape_range * sizeof(CauchyTerm));
        else
        {
            int Nt_alloc = 0;
            for(int m = 1; m < shape_range; m++)
                if(terms_per_shape[m] > 0)
                    Nt_alloc += terms_per_shape[m] * (m + tmp_pncc);
            new_child_terms = (CauchyTerm*) malloc(Nt_alloc * sizeof(CauchyTerm));
        }
        null_ptr_check(new_child_terms);
        int Nt_new = 0;

        for(int m = 1; m < shape_range; m++)
        {
            int Nt_shape = terms_per_shape[m];
            if(Nt_shape > 0)
            {
                // Allocate Memory for new terms
                BYTE_COUNT_TYPE new_shape;
                if(with_tp)
                {
                    new_shape = m + tmp_pncc;
                    if( (!DENSE_STORAGE) && (!skip_post_mu) )
                    {
                        BYTE_COUNT_TYPE bytes_max_cells = (((BYTE_COUNT_TYPE)dce_helper->cell_counts_cen[new_shape]) / (1 + HALF_STORAGE)) * Nt_shape * sizeof(BKEYS_TYPE);
                        gb_tables->extend_bp_tables(bytes_max_cells);
                    }
                }
                else
                    new_shape = m;
                
                if(!skip_post_mu)
                {
                    BYTE_COUNT_TYPE new_num_terms = ((BYTE_COUNT_TYPE)Nt_shape) * (new_shape+1);
                    BYTE_COUNT_TYPE ps_bytes = new_num_terms * new_shape * sizeof(double);
                    BYTE_COUNT_TYPE bs_bytes = new_num_terms * d * sizeof(double);
                    coalign_store->extend_storage(ps_bytes, bs_bytes, d);
                }
                // End of memory allocation
                CauchyTerm* terms = terms_dp[m];
                for(int i = 0; i < Nt_shape; i++)
                {
                    CauchyTerm* parent = terms + i;
                    transfer_term_to_workspace(&childterms_workspace, parent);
                    // Run Time Propagation Routines
                    if( with_tp )
                    {
                        parent->time_prop(Phi, B, u, cmcc);
                        int m_tp = parent->tp_coalign(tmp_Gamma, tmp_beta, tmp_pncc);
                        if( (!DENSE_STORAGE) && (!skip_post_mu) )
                        {
                            if(parent->m == parent->phc)
                            {
                                BKEYS B_parent = parent->enc_B;
                                gb_tables->set_term_bp_table_pointer( &(parent->enc_B), parent->cells_gtable_p, true);
                                memcpy(parent->enc_B, B_parent, parent->cells_gtable_p * sizeof(BKEYS_TYPE));
                            }
                            else
                            {
                                gb_tables->set_term_bp_table_pointer( &(parent->enc_B), dce_helper->cell_counts_cen[m_tp] / (1 + HALF_STORAGE), false);
                                if(FAST_TP_DCE)
                                    make_time_prop_btable_fast(parent, dce_helper);
                                else
                                    make_time_prop_btable(parent, dce_helper);
                                gb_tables->incr_chunked_bp_table_ptr(parent->cells_gtable);
                                // Nats check
                                /*
                                if(  (dce_helper->cell_counts_cen[m_tp] - dce_helper->cell_counts_cen[m]) != ((parent->cells_gtable - parent->cells_gtable_p) * (1 + HALF_STORAGE) ) )
                                {
                                    printf(YEL "--For term %d, m_tp=%d (after TP), m_old=%d (before tp), the following has occured:\n", i, m_tp, m);
                                    printf(YEL "  Max cells cen m_tp: %d\n", dce_helper->cell_counts_cen[m_tp]);
                                    printf(YEL "  Max cells cen m: %d\n", dce_helper->cell_counts_cen[m]);
                                    printf(YEL "  Cells gtable: %d\n", parent->cells_gtable * (1 + HALF_STORAGE));
                                    printf(YEL "  Cells gtable_p: %d\n", parent->cells_gtable_p * (1 + HALF_STORAGE));
                                    printf(YEL "  Max cells diff: %d\t cells gtable diff: %d\n", dce_helper->cell_counts_cen[m_tp] - dce_helper->cell_counts_cen[m],  (parent->cells_gtable - parent->cells_gtable_p) * (1 + HALF_STORAGE) );
                                    printf(NC "--\n");
                                }
                                */
                            }
                        }
                    }
                    int m_precoalign = parent->m; // HPs pre-MU Coalign
                    // Run Measurement Update Routines
                    //sort_encoded_B(parent->enc_B, parent->cells_gtable);
                    CauchyTerm* children = skip_post_mu ? new_child_terms : new_child_terms + Nt_new;
                    int num_children = parent->msmt_update(children, msmt, H, gamma, false, skip_post_mu, &childterms_workspace);
                    //sort_encoded_B(parent->enc_B, parent->cells_gtable);
                    Nt_new += num_children;
                    // Cache moment results here -- evaluate g / yei
                    cache_moments(parent, children, num_children);
                    if(!skip_post_mu)
                    {
                        // Normalize the parent, coalign the children, increment new terms per shape count,
                        // assign parent and child term elements to the coaligned storage buffers  
                        parent->normalize_hps(true);
                        new_terms_per_shape[parent->m]++;
                        coalign_store->set_term_ptrs(parent, m_precoalign);
                        for(int j = 0; j < num_children; j++)
                        {
                            new_terms_per_shape[children[j].mu_coalign()]++;
                            coalign_store->set_term_ptrs(children+j, m_precoalign);
                        }
                    }
                    else
                        new_terms_per_shape[parent->m] += num_children + 1;
                }
            }
        }
        Nt += Nt_new;
        // Finalize cached moment information
        finalize_cached_moments();
        // Aggregate terms of similar shape into contiguous arrays
        if(!skip_post_mu)
        {    
            // Now coalesce terms into array of array storage for term reduction
            CauchyTerm** new_terms_dp = (CauchyTerm**) malloc( shape_range * sizeof(CauchyTerm*) );
            null_dptr_check((void**)new_terms_dp);
            for(int m = 0; m < shape_range; m++)
            {
                if(new_terms_per_shape[m] > 0)
                {
                    new_terms_dp[m] = (CauchyTerm*) malloc( new_terms_per_shape[m] * sizeof(CauchyTerm) );
                    null_ptr_check(new_terms_dp[m]);
                }
                else 
                    new_terms_dp[m] = (CauchyTerm*) malloc(0);
            }   
            // Add parent terms 
            memset(new_terms_per_shape, 0, shape_range * sizeof(int));
            for(int shape = 0; shape < shape_range; shape++)
            {
                int Nt_shape = terms_per_shape[shape];
                CauchyTerm* terms = terms_dp[shape];
                for(int i = 0; i < Nt_shape; i++)
                {
                    int m = terms[i].m;
                    new_terms_dp[m][new_terms_per_shape[m]++] = terms[i];
                }
                free(terms_dp[shape]);
            }
            // Add new child terms 
            for(int i = 0; i < Nt_new; i++)
            {
                int m = new_child_terms[i].m;
                new_terms_dp[m][new_terms_per_shape[m]++] = new_child_terms[i];
            }
            // Swap terms_dp w/ new_terms_dp and terms_per_shape w/ new_terms_per_shape
            ptr_swap<CauchyTerm*>(&terms_dp, &new_terms_dp);
            ptr_swap<int>(&terms_per_shape, &new_terms_per_shape);
            free(new_terms_dp);
            free(new_terms_per_shape);
            free(new_child_terms);
            tmr_mu.toc(false);

            // Print Stats
            if(print_basic_info)
            {
                printf("Step %d/%d:\n", master_step+1, num_estimation_steps);
                if(with_tp)
                    printf("TP to MUC: Took %d ms\n", tmr_mu.cpu_time_used);
                else 
                    printf("MU to MUC: Took %d ms\n", tmr_mu.cpu_time_used);
                printf("Total Terms after MUC: %d\n", Nt);
                for(int m = 1; m < shape_range; m++)
                    if(terms_per_shape[m] > 0)
                        printf("Shape %d has %d terms\n", m, terms_per_shape[m]);
                print_conditional_mean_variance();
                //stats.print_total_estimator_memory(gb_tables, coalign_store, reduce_store, Nt, true, num_threads_tp_to_muc, num_threads_make_gtables);
                //stats.print_cell_count_histograms(terms_dp, shape_range, terms_per_shape, dce_helper->cell_counts_cen);
            }
            // Free unused memory
            coalign_store->unallocate_unused_space();
            if(with_tp && !DENSE_STORAGE)
                gb_tables->swap_btables();
            for(int i = 0; i < num_threads_make_gtables; i++)
                reduce_store[i].reset();
        }
        else 
        {
            tmr_mu.toc(false);
            // Print Stats
            ptr_swap<int>(&terms_per_shape, &new_terms_per_shape);
            free(new_child_terms);
            free(new_terms_per_shape);
            if(print_basic_info)
            {
                printf("Step %d/%d:\n", master_step+1, num_estimation_steps);
                if(with_tp)
                    printf("TP to MU: Took %d ms\n", tmr_mu.cpu_time_used);
                else 
                    printf("MU to MU: Took %d ms\n", tmr_mu.cpu_time_used);
                printf("Total Terms after MU: %d\n", Nt);
                printf("Note: No added memory pressure (SKIP_LAST_STEP=true)\n");
                for(int m = 1; m < shape_range; m++)
                    if(terms_per_shape[m] > 0)
                        printf("Shape %d has %d terms\n", m, terms_per_shape[m]);
                print_conditional_mean_variance();
                //stats.print_total_estimator_memory(gb_tables, coalign_store, reduce_store, Nt, true, num_threads_tp_to_muc, num_threads_make_gtables);
            }
        }
        // I dont think this needs to be set, but if we do enter here after threading was used, threads used should be set to 1
        num_threads_tp_to_muc = 1;
        num_threads_make_gtables = 1;
    }

    void threaded_step_tp_to_muc(double msmt, double* Phi, double* Gamma, double* beta, double* H, double gamma, double* B, double* u)
    {
		#if _WIN32
			double tmp_Gamma[MAX_HP_SIZE*MAX_HP_SIZE];
			double tmp_beta[MAX_HP_SIZE];
		#else 
			double tmp_Gamma[d*pncc];
			double tmp_beta[pncc];
		#endif
        int tmp_pncc = 0;
        const bool with_tp = ((master_step % p) == 0);
        fz = 0;
        memset(conditional_mean, 0, d * sizeof(C_COMPLEX_TYPE));
        memset(conditional_variance, 0, d*d*sizeof(C_COMPLEX_TYPE));
        CPUTimer tmr_mu;
        tmr_mu.tic();
        // Transpose, normalize, and pre-coalign Gamma and beta
        if(with_tp)
            tmp_pncc = precoalign_Gamma_beta(Gamma, beta, pncc, d, tmp_Gamma, tmp_beta);
        
        int num_chunks = (Nt + MIN_TERMS_PER_THREAD_TP_TO_MUC -1) / MIN_TERMS_PER_THREAD_TP_TO_MUC;
        num_threads_tp_to_muc = num_chunks > NUM_CPUS ? NUM_CPUS : num_chunks;

        pthread_t* tids = (pthread_t*)malloc(num_threads_tp_to_muc*sizeof(pthread_t));
        DIST_TP_TO_MUC_STRUCT* tid_args = (DIST_TP_TO_MUC_STRUCT*) malloc(num_threads_tp_to_muc * sizeof(DIST_TP_TO_MUC_STRUCT) );
        for(int i = 0; i < num_threads_tp_to_muc; i++)
        {
            tid_args[i].cauchyEst = this;
            tid_args[i].coalign_store = coalign_store + i;
            tid_args[i].gb_tables = gb_tables + i;
            tid_args[i].dce_helper = dce_helper + i;
            tid_args[i].fz_chunk = 0; // set on return
            tid_args[i].cond_mean_chunk = NULL; // set on return 
            tid_args[i].cond_var_chunk = NULL; // set on return
            tid_args[i].new_terms_dp = NULL; // set on return
            tid_args[i].new_terms_per_shape = NULL; // set on return
            tid_args[i].Phi = Phi;
            tid_args[i].H = H;
            tid_args[i].processed_Gamma = with_tp ? tmp_Gamma : NULL;
            tid_args[i].processed_beta = with_tp ? tmp_beta : NULL;
            tid_args[i].processed_pncc = tmp_pncc;
            tid_args[i].msmt = msmt;
            tid_args[i].gamma = gamma;
            tid_args[i].B = B;
            tid_args[i].u = u;
            tid_args[i].tid = i;
            tid_args[i].n_tids = num_threads_tp_to_muc;
            pthread_create(tids + i, NULL, distributed_step_tp_to_muc, tid_args + i);
        }
        for(int i = 0; i < num_threads_tp_to_muc; i++)
            pthread_join(tids[i], NULL);
        
        // Sum moment chunks
        int d_squared = d*d;
        for(int i = 0; i < num_threads_tp_to_muc; i++)
        {
            fz += tid_args[i].fz_chunk;
            add_vecs(conditional_mean, tid_args[i].cond_mean_chunk, d);
            add_vecs(conditional_variance, tid_args[i].cond_var_chunk, d_squared);
            free(tid_args[i].cond_mean_chunk);
            free(tid_args[i].cond_var_chunk);
        }
        finalize_cached_moments();

        // Take the partial sorted term array of arrays and collect them together
        memset(terms_per_shape, 0, shape_range * sizeof(int));
        for(int m = 1; m < shape_range; m++)
            for(int j = 0; j < num_threads_tp_to_muc; j++)
                terms_per_shape[m] += tid_args[j].new_terms_per_shape[m];
        Nt = 0;
        if(!skip_post_mu)
        {
            // This part could be threaded if its seen to be slow
            CauchyTerm** new_terms_dp = (CauchyTerm**) malloc( shape_range * sizeof(CauchyTerm*) );
            for(int m = 0; m < shape_range; m++)
            {
                free(terms_dp[m]);
                new_terms_dp[m] = (CauchyTerm*) malloc(terms_per_shape[m] * sizeof(CauchyTerm));
                if(terms_per_shape[m] > 0)
                {
                    Nt += terms_per_shape[m];
                    int count_m = 0;
                    for(int j = 0; j < num_threads_tp_to_muc; j++)
                    {
                        int terms_chunk_m = tid_args[j].new_terms_per_shape[m];
                        if( terms_chunk_m > 0 )
                        {
                            memcpy(new_terms_dp[m] + count_m, tid_args[j].new_terms_dp[m], terms_chunk_m * sizeof(CauchyTerm) );
                            count_m += terms_chunk_m;
                        }
                    }
                }
                // Clean up terms of terms array of each threads
                for(int j = 0; j < num_threads_tp_to_muc; j++)
                    free(tid_args[j].new_terms_dp[m]);
            }
            for(int i = 0; i < num_threads_tp_to_muc; i++)
            {
                free(tid_args[i].new_terms_per_shape);
                free(tid_args[i].new_terms_dp);
            }
            ptr_swap<CauchyTerm*>(&new_terms_dp, &terms_dp);
            free(new_terms_dp);
        }
        else 
        {
            for(int m = 0; m < shape_range; m++)
                Nt += terms_per_shape[m];
            for(int i = 0; i < num_threads_tp_to_muc; i++)
                free(tid_args[i].new_terms_per_shape);
        }
        free(tid_args);
		free(tids);
        // Print Stats
        tmr_mu.toc(false);
        if(print_basic_info)
        {
            printf("Step %d/%d:\n", master_step+1, num_estimation_steps);
            if(with_tp)
                if(skip_post_mu)
                    printf("TP to MU [Threaded %d]: Took %d ms\n", num_threads_tp_to_muc, tmr_mu.cpu_time_used);
                else
                    printf("TP to MUC [Threaded %d]: Took %d ms\n", num_threads_tp_to_muc, tmr_mu.cpu_time_used);
            else
                if(skip_post_mu) 
                    printf("MU [Threaded %d]: Took %d ms\n", num_threads_tp_to_muc, tmr_mu.cpu_time_used);
                else
                    printf("MU to MUC [Threaded %d]: Took %d ms\n", num_threads_tp_to_muc, tmr_mu.cpu_time_used);
            if(skip_post_mu)
            {
                printf("Total Terms after MU: %d\n", Nt);
                printf("Note: No added memory pressure (SKIP_LAST_STEP=true)\n");
            }
            else
                printf("Total Terms after MUC: %d\n", Nt);
            for(int m = 1; m < shape_range; m++)
                if(terms_per_shape[m] > 0)
                    printf("Shape %d has %d terms\n", m, terms_per_shape[m]);
            print_conditional_mean_variance();
            //stats.print_total_estimator_memory(gb_tables, coalign_store, reduce_store, Nt, true, num_threads_tp_to_muc, num_threads_make_gtables);
            //stats.print_cell_count_histograms(terms_dp, shape_range, terms_per_shape, dce_helper->cell_counts_cen);
        }
        // Unallocate the reduce_storage
        for(int i = 0; i < num_threads_make_gtables; i++)
            reduce_store[i].reset();
    }

    void fast_term_reduction_and_create_gtables()
    {
        if(skip_post_mu)
            return;
        // bs[i] are the bs list (array) of all terms with i hyperplanes
        // shape_idxs[i][j] is the index of the "bs[i] + d*j" vector in the term list
        CPUTimer ftr_tmr;
        ftr_tmr.tic();
        CPUTimer tmr;
        tmr.tic();
        int max_Nt_shape = array_max<int>(terms_per_shape, shape_range);
        if(max_Nt_shape > ftr_helper.max_num_terms)
            ftr_helper.realloc_helpers(max_Nt_shape);
        tmr.toc(false);
        if(print_basic_info)
            printf("[FTR/Gtables step %d/%d:] Preprocessing took %d ms\n", master_step+1, num_estimation_steps, tmr.cpu_time_used);
        int Nt_reduced = 0; // Total number of terms after term reduction has finished
        int Nt_removed = 0; // Total number of terms removed after term approximation
        CauchyTerm** ftr_terms_dp = (CauchyTerm**) malloc( shape_range * sizeof(CauchyTerm*) );
        null_dptr_check((void**)ftr_terms_dp);
        for(int m = 0; m < shape_range; m++)
        {
            if(terms_per_shape[m] > 0)
            {
                int Nt_shape = terms_per_shape[m];
                CauchyTerm* terms = terms_dp[m];
                memcpy(ftr_helper.F_TR, ftr_helper.F, Nt_shape * sizeof(int) );
                
                // helper variables
                int bopm_cpu_time;
                int ftr_cpu_time;
                bool is_ftr_threaded = false;
                bool is_make_gtables_threaded = false;
                int num_gtable_tids = 1;
                
                // Distributed Construction of helper arrays for FTR, and FTR itself
                if( (NUM_CPUS_FTR > 1) && (Nt_shape > MIN_TERMS_PER_THREAD_FTR) )
                {
                    is_ftr_threaded = true;
                    const int term_chunk_per_cpu = Nt_shape / NUM_CPUS_FTR; // This should be chosen is a wiser way depending on term size
                    tmr.tic();
                    threaded_build_ordered_point_maps(terms, 
                        ftr_helper.ordered_points, ftr_helper.forward_map, 
                        ftr_helper.backward_map, Nt_shape, d, NUM_CPUS_FTR);
                    tmr.toc(false);
                    bopm_cpu_time = tmr.cpu_time_used;

                    // Run threaded FTR
                    tmr.tic();
                    threaded_fast_term_reduction(
                        terms,
                        ftr_helper.F_TR,
                        ftr_helper.ordered_points, 
                        ftr_helper.forward_map, 
                        ftr_helper.backward_map,
                        REDUCTION_EPS, Nt_shape, 
                        m, d,
                        NUM_CPUS_FTR, term_chunk_per_cpu);
                    tmr.toc(false);
                    ftr_cpu_time = tmr.cpu_time_used;
                }
                // Serially Construct of helper arrays for FTR, and FTR itself
                else 
                {
                    tmr.tic();
                    build_ordered_point_maps(
                        terms,
                        ftr_helper.ordered_points, 
                        ftr_helper.forward_map, 
                        ftr_helper.backward_map, 
                        Nt_shape,
                        d, false);
                    tmr.toc(false);
                    bopm_cpu_time = tmr.cpu_time_used;

                    // Run FTR
                    tmr.tic();
                    fast_term_reduction(
                        terms,
                        ftr_helper.F_TR, 
                        ftr_helper.ordered_points,
                        ftr_helper.forward_map, 
                        ftr_helper.backward_map,
                        REDUCTION_EPS, Nt_shape, m, d);
                    tmr.toc(false);
                    ftr_cpu_time = tmr.cpu_time_used;
                }

                // Build the term reduction lists: 
                tmr.tic();
                ForwardFlagArray ffa(ftr_helper.F_TR, Nt_shape, NUM_CPUS);
                tmr.toc(false);
                int ffa_cpu_time = tmr.cpu_time_used;

                // Build Gtables for shape m
                tmr.tic();
                int Nt_reduced_shape = 0;
                int Nt_removed_shape = 0;
                ftr_terms_dp[m] = (CauchyTerm*) malloc( ffa.num_terms_after_reduction * sizeof(CauchyTerm) );
                null_ptr_check(ftr_terms_dp[m]);
                CauchyTerm* ftr_terms = ftr_terms_dp[m];
                if( (NUM_CPUS > 1) && (Nt_shape > MIN_TERMS_PER_THREAD_GTABLE) )
                {
                    is_make_gtables_threaded = true;
                    num_gtable_tids = threaded_make_gtables(
                        &Nt_reduced_shape, &Nt_removed_shape,
                        terms, ftr_terms, &ffa, dce_helper, 
                        gb_tables, reduce_store, ftr_helper.F_TR,
                        B_dense, G_SCALE_FACTOR, Nt_shape, m, d,
                        win_num, master_step+1, num_estimation_steps);
                    // Up the count of num_threads_make_gtable if this shape used more chunked packed space than previously
                    num_threads_make_gtables = num_gtable_tids > num_threads_make_gtables ? num_gtable_tids : num_threads_make_gtables;
                }
                else 
                {
                    make_gtables(
                        &Nt_reduced_shape, &Nt_removed_shape,
                        terms, ftr_terms, &ffa, dce_helper, 
                        gb_tables, reduce_store, ftr_helper.F_TR,
                        B_dense, G_SCALE_FACTOR, 
                        Nt_shape, ffa.num_terms_after_reduction,
                        m, d);
                }
                // After term reduction and g-evaluation 
                Nt_reduced += Nt_reduced_shape;
                Nt_removed += Nt_removed_shape;
                terms_per_shape[m] = Nt_reduced_shape;
                tmr.toc(false);
                int gtable_cpu_time = tmr.cpu_time_used;
                if(print_basic_info)
                {
                    printf("Shape %d: (%d/%d remain)\n", m, ffa.num_terms_after_reduction, Nt_shape);
                    if(is_ftr_threaded)
                    {
                        printf("  Built ordered maps in %d ms (threaded %d)\n", bopm_cpu_time, NUM_CPUS_FTR);
                        printf("  FTR took %d ms (threaded %d)\n", ftr_cpu_time, NUM_CPUS_FTR);
                    }
                    else 
                    {
                        printf("  Built ordered maps in %d ms\n", bopm_cpu_time);
                        printf("  FTR took %d ms\n", ftr_cpu_time);
                    }
                    printf("  Built Forward Flag Array in %d ms\n", ffa_cpu_time);
                    if(is_make_gtables_threaded)
                        printf("  Built Gtables in %d ms (threaded %d)\n", gtable_cpu_time, num_gtable_tids);
                    else
                        printf("  Built Gtables in %d ms\n", gtable_cpu_time);
                    if(WITH_TERM_APPROXIMATION)
                        printf("  Term Approx removes %d terms, %d/%d remain\n", Nt_removed_shape, Nt_reduced_shape, ffa.num_terms_after_reduction);
                }
            }
            else 
            {
                ftr_terms_dp[m] = (CauchyTerm*) malloc(0);
            }
            free(terms_dp[m]);
        }
        // For all terms not reduced out or approximated out, keep these terms
        Nt = Nt_reduced;
        ptr_swap<CauchyTerm*>(&terms_dp, &ftr_terms_dp);
        free(ftr_terms_dp);

        ftr_tmr.toc(false);
        if(print_basic_info)
        {
            printf("FTR/Gtables took %d ms total!\n", ftr_tmr.cpu_time_used);     
            printf("Total Terms after FTR: %d\n", Nt);
            for(int i = 0; i < shape_range; i++)
                if(terms_per_shape[i] > 0)
                    printf("After FTR: Shape %d has %d terms\n", i, terms_per_shape[i]);
            //stats.print_total_estimator_memory(gb_tables, coalign_store, reduce_store, Nt, false, num_threads_tp_to_muc, num_threads_make_gtables);
            //stats.print_cell_count_histograms(terms_dp, shape_range, terms_per_shape, dce_helper->cell_counts_cen);
        }
        
        // Deallocate unused or unneeded memory
        tmr.tic();
        for(int i = 0; i < num_threads_tp_to_muc; i++)
            coalign_store[i].reset();
        for(int i = 0; i < num_threads_make_gtables; i++)
        { 
            reduce_store[i].unallocate_unused_space();
            gb_tables[i].swap_gtables();
        }
        tmr.toc(false);

        // Compute moments after FTR
        if(print_basic_info)
        {
            printf("Deallocating memory after FTR took %d ms\n", tmr.cpu_time_used);   
            compute_moments(false);
        }
        else
        {
            //fz = 1 + 0*I;
            fz = MAKE_CMPLX(1,0);
        }
    }

    void step_first(double msmt, double* H, double gamma)
    {
        CauchyTerm* terms = terms_dp[d];
        Nt = terms[0].msmt_update(terms+1, msmt, H, gamma, true, false, &childterms_workspace) + 1;
        terms_per_shape[d] = Nt;
        compute_moments(true);
        for(int i = 0; i < Nt; i++)
        {
            reduce_store->set_term_ptrs(terms + i);
            terms[i].cells_gtable = dce_helper->cell_counts_cen[d] / (1 + HALF_STORAGE);
            if(!DENSE_STORAGE)
                gb_tables->set_term_btable_pointer( &(terms[i].enc_B), terms[i].cells_gtable, true);
            else
                terms[i].enc_B = B_dense;
            gb_tables->set_term_gtable_pointer(&(terms[i].gtable), terms[i].cells_gtable, true);
            make_gtable_first(terms + i, G_SCALE_FACTOR);
            terms[i].become_parent();
        }
        gb_tables->swap_gtables();
        if(print_basic_info)
            compute_moments(false);
        else
        {
            //fz = 1 + 0*I;
            fz = MAKE_CMPLX(1,0);
        }
        memcpy(last_conditional_mean, conditional_mean, d*sizeof(C_COMPLEX_TYPE));
        memcpy(last_conditional_variance, conditional_variance, d*d*sizeof(C_COMPLEX_TYPE));
        last_fz = fz;
    }

    // Main function that is called
    int step(double msmt, double* Phi, double* Gamma, double* beta, double* H, double gamma, double* B, double* u)
    {
        set_function_pointers();
        if( numeric_moment_errors & (1<<ERROR_FZ_NEGATIVE) )
        {
            printf(RED "[Window %d:] ERROR_FZ_NEGATIVE triggered. Cannot continue stepping until this estimator has been reset!"
                   NC "\n", win_num);
            return numeric_moment_errors;
        }
        if(master_step == num_estimation_steps)
        {
            printf(RED "[Window %d:] ERROR MASTER STEP. master_step == num_estimation_steps (max measurements=%d)!\nCannot continue stepping until this estimator has been reset!"
                   NC "\n", win_num, master_step);
            exit(1);
        }

        CPUTimer tmr;
        tmr.tic();
        skip_post_mu = SKIP_LAST_STEP && (master_step == (num_estimation_steps-1)); 
        if(master_step == 0)
            step_first(msmt, H, gamma);
        else
        {
            if( (NUM_CPUS == 1) || (Nt < MIN_TERMS_PER_THREAD_TP_TO_MUC) )
                step_tp_to_muc(msmt, Phi, Gamma, beta, H, gamma, B, u);
            else
                threaded_step_tp_to_muc(msmt, Phi, Gamma, beta, H, gamma, B, u);
            fast_term_reduction_and_create_gtables();
        }
        master_step++;
        tmr.toc(false);
        if(print_basic_info)
            printf("Step %d took %d ms\n", master_step, tmr.cpu_time_used);
        return numeric_moment_errors;
    }

    void reset()
    {
        CPUTimer tmr;
        tmr.tic();
        for(int i = 0; i < NUM_CPUS; i++)
        {
            //dce_helper[i].deinit();
            gb_tables[i].deinit(); // g and b-tables
            coalign_store[i].deinit();
            reduce_store[i].deinit();
        }
        ftr_helper.deinit();

        // Deallocate terms 
        for(int i = 0; i < shape_range; i++)
            free(terms_dp[i]);
        for(int i = 0; i < shape_range; i++)
        {
            if( i == d )
            {
                terms_dp[d] = (CauchyTerm*) malloc((d+1) * sizeof(CauchyTerm));
                null_ptr_check(terms_dp[d]);
            }
            else 
                terms_dp[i] = (CauchyTerm*) malloc(0); 
        }
        memset(terms_per_shape, 0, shape_range * sizeof(int));
        terms_per_shape[d] = 1;
        Nt = 1;
        master_step = 0;
        numeric_moment_errors = 0;

        // Re-init first term
        setup_first_term(&childterms_workspace, terms_dp[d], A0_init, p0_init, b0_init, d);

        //Re-init helpers
        //dce_helper->init(shape_range-1, d, DCE_STORAGE_MULT);
        gb_tables->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        coalign_store->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        reduce_store->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        for(int i = 1; i < NUM_CPUS; i++)
        {
            //dce_helper[i].init(shape_range-1, d, DCE_STORAGE_MULT);
            gb_tables[i].init(0, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
            coalign_store[i].init(0, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
            reduce_store[i].init(0, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        }
        ftr_helper.init(d, 1<<d);

        num_threads_tp_to_muc = 1;
        tmr.toc(false);
        if(print_basic_info)
            printf("Resetting CF Took %d ms\n", tmr.cpu_time_used);
    }

    void reinitialize_start_statistics(double* A_0, double* p_0, double* b_0)
    {
        memcpy(A0_init, A_0, d*d*sizeof(double));
        memcpy(p0_init, p_0, d*sizeof(double));
        memcpy(b0_init, b_0, d*sizeof(double));
        // Re-init first term
        setup_first_term(&childterms_workspace, terms_dp[d], A0_init, p0_init, b0_init, d);
    }
    
    // Shifts all b-vectors of the CF by the bias of size R^d
    void shift_cf_by_bias(double* bias)
    {
        if(!skip_post_mu)
        {
            for(int m = 1; m < shape_range; m++)
            {
                int Nt_shape = terms_per_shape[m];
                if( Nt_shape > 0 ) 
                {
                    CauchyTerm* terms = terms_dp[m];
                    for(int i = 0; i < Nt_shape; i++)
                        for(int j = 0; j < d; j++)
                        terms[i].b[j] += bias[j];
                }
            }
        }
    }

    // Deterministically propagates the CF, using either (Phi,B,u) or just (Phi,None,None)
    void deterministic_time_prop(double* Phi, double* B, double* u)
    {
        int _cmcc;
        //bool Bu_okay = !( (B==NULL) ^ (u==NULL) ); // Both False or Both True Evaluate to True
        //assert(Bu_okay);
        if( (B == NULL) && (u == NULL) )
            _cmcc = 0;
        else if( (B != NULL) && (u != NULL) )
            _cmcc = cmcc;
        else
        {
            printf("Illegal use of arguments B and u! Either B or u set, both not both!\n");
            assert(false);
        }

        for(int m = 1; m < shape_range; m++)
        {
            int Nt_shape = terms_per_shape[m];
            for(int i = 0; i < Nt_shape; i++)
            {
                CauchyTerm* term = terms_dp[m] + i;
                term->time_prop(Phi, B, u, _cmcc);
            }
        }
    }

    // Shifts bs in CF by -\delta{x_k}. Sets conditional_mean=\delta{x_k} + duc->x (which is x_bar). Then sets (duc->x) x_bar = creal(conditional_mean)
    void finalize_extended_moments(double* x_bar)
    {
        // Shifts bs in CF by -\delta{x_k}.
        // We do not need to do this on the last MU
        // This could be threaded, if its slow
        CPUTimer tmr_extnd;
        tmr_extnd.tic();
        if(!skip_post_mu)
        {
			#if _WIN32
				double delta_xk[MAX_HP_SIZE];
			#else
				double delta_xk[d];
			#endif

            convert_complex_array_to_real(conditional_mean, delta_xk, d);
            for(int m = 1; m < shape_range; m++)
            {
                int Nt_shape = terms_per_shape[m];
                if( Nt_shape > 0 ) 
                {
                    CauchyTerm* terms = terms_dp[m];
                    for(int i = 0; i < Nt_shape; i++)
                        sub_vecs(terms[i].b, delta_xk, d);
                }
            }
        }
        // Sets conditional_mean=\delta{x_k} + duc->x (which is x_bar).
        for(int i = 0; i < d; i++)
            conditional_mean[i] += x_bar[i];
        // Then sets (duc->x) x_bar = creal(conditional_mean)
        for(int i = 0; i < d; i++)
            x_bar[i] = creal(conditional_mean[i]);
        tmr_extnd.toc(false);
        if(print_basic_info)
            printf("finalize_extended_moments took %d ms\n", tmr_extnd.cpu_time_used);
    }

    ~CauchyEstimator()
    {
        free(terms_per_shape);
        free(root_point);
        free(conditional_mean);
        free(conditional_variance);
        free(last_conditional_mean);
        free(last_conditional_variance);

        for(int i = 0; i < NUM_CPUS; i++)
        {
            dce_helper[i].deinit();
            gb_tables[i].deinit(); // g and b-tables
            coalign_store[i].deinit();
            reduce_store[i].deinit();
        }
        // Deallocate terms 
        for(int i = 0; i < shape_range; i++)
            free(terms_dp[i]);
        free(terms_dp);
        
        ftr_helper.deinit();
        childterms_workspace.deinit();
        if(DENSE_STORAGE)
            free(B_dense);
        
        free(dce_helper);
        free(gb_tables);
        free(coalign_store);
        free(reduce_store);

        free(A0_init);
        free(p0_init);
        free(b0_init);
    }

};



void cache_moments(CauchyTerm* parent, CauchyTerm* children, int num_children, 
    double* root_point, C_COMPLEX_TYPE* fz, C_COMPLEX_TYPE* conditional_mean, C_COMPLEX_TYPE* conditional_variance, const int d)
{
    C_COMPLEX_TYPE g_val;
	#if _WIN32
		C_COMPLEX_TYPE yei[MAX_HP_SIZE];
	#else
		C_COMPLEX_TYPE yei[d];
	#endif

    // Cache parent term
    g_val = parent->eval_g_yei(root_point, yei, false);
    *fz += g_val;
    for(int j = 0; j < d; j++)
    {
        C_COMPLEX_TYPE y = yei[j];
        conditional_mean[j] += g_val * y;
        for(int k = 0; k < d; k++)
            conditional_variance[j*d + k] -= g_val * y * yei[k];
    }
    for(int i = 0; i < num_children; i++)
    {
        g_val = children[i].eval_g_yei(root_point, yei, false);
        *fz += g_val;
        for(int j = 0; j < d; j++)
        {
            C_COMPLEX_TYPE y = yei[j];
            conditional_mean[j] += g_val * y;
            for(int k = 0; k < d; k++)
                conditional_variance[j*d + k] -= g_val * y * yei[k];
        }
    }
}

void* distributed_step_tp_to_muc(void* args)
{
    CPUTimer tmr_mu;
    tmr_mu.tic();

    DIST_TP_TO_MUC_STRUCT* dist_args = (DIST_TP_TO_MUC_STRUCT*) args;
    CauchyEstimator* cauchyEst = dist_args->cauchyEst;

    int d = cauchyEst->d;
    int cmcc = cauchyEst->cmcc;
    int shape_range = cauchyEst->shape_range;
    int* terms_per_shape = cauchyEst->terms_per_shape;
    double* root_point = cauchyEst->root_point;
    const bool skip_post_mu = cauchyEst->skip_post_mu;
    const bool with_tp = ((cauchyEst->master_step % cauchyEst->p) == 0);
    int tid = dist_args->tid;
    int n_tids = dist_args->n_tids;

    ChunkedPackedTableStorage* gb_tables = NULL;
    CoalignmentElemStorage* coalign_store = NULL;
    DiffCellEnumHelper* dce_helper = NULL;
    ChildTermWorkSpace* childterms_workspace = NULL;
    // Initialize gb_tables, and dce_helper if we need to run TP DCE, not using dense storage, and not the last step 
    // Only the parent btables need to be initialized in this routine
    // This is why everything is set to null at the begining
    if(with_tp && (!DENSE_STORAGE) && (!skip_post_mu) )
    {
        gb_tables = dist_args->gb_tables;
        dce_helper = dist_args->dce_helper;
    }
    // If its not the last step (or if SKIP_LAST_STEP was set to true)
    if(!skip_post_mu)
        coalign_store = dist_args->coalign_store;
    // Initialize child term workspace
    childterms_workspace = (ChildTermWorkSpace*) malloc(sizeof(ChildTermWorkSpace));
    null_ptr_check(childterms_workspace);
    childterms_workspace->init(shape_range-1, d);

    // Initialize partial moment workspace
    C_COMPLEX_TYPE fz_chunk = 0;
    dist_args->cond_mean_chunk = (C_COMPLEX_TYPE*) malloc(d*sizeof(C_COMPLEX_TYPE));
    dist_args->cond_var_chunk = (C_COMPLEX_TYPE*) malloc(d*d*sizeof(C_COMPLEX_TYPE));
    C_COMPLEX_TYPE* cond_mean_chunk = dist_args->cond_mean_chunk;
    C_COMPLEX_TYPE* cond_var_chunk = dist_args->cond_var_chunk;
    null_ptr_check(cond_mean_chunk);
    null_ptr_check(cond_var_chunk);
    memset(cond_mean_chunk, 0, d * sizeof(C_COMPLEX_TYPE));
    memset(cond_var_chunk, 0, d*d*sizeof(C_COMPLEX_TYPE));
    
    // Bring in Gamma, beta matrices if its a TP step
    // If not a TP step, these are NULL and tmp_pncc is 0
    double* tmp_Gamma = dist_args->processed_Gamma;
    double* tmp_beta = dist_args->processed_beta;
    int tmp_pncc = dist_args->processed_pncc;
    double* Phi = dist_args->Phi;
    // Bring in H, gamma, msmt
    double* H = dist_args->H;
    double gamma = dist_args->gamma;
    double msmt = dist_args->msmt;
    double* B = dist_args->B;
    double* u = dist_args->u;

    // Allocate structures for the maximum number of new terms we'd generate at this step
    int* new_terms_per_shape = (int*) calloc(shape_range, sizeof(int));
    null_ptr_check(new_terms_per_shape);
    int* tid_terms_per_shape = (int*) calloc(shape_range, sizeof(int));
    null_ptr_check(tid_terms_per_shape);
    CauchyTerm* new_child_terms;
    int Nt_alloc = 0;
    for(int m = 1; m < shape_range; m++)
    {
        if(terms_per_shape[m] > 0)
        {
            if(terms_per_shape[m] < n_tids)
            {
                if(tid < terms_per_shape[m])
                {
                    Nt_alloc += (m + tmp_pncc);
                    tid_terms_per_shape[m] = 1;
                }
            }
            else
            {
                int added_terms = terms_per_shape[m] / n_tids;
                int modulo_terms = terms_per_shape[m] % n_tids;
                if(tid < modulo_terms)
                    added_terms += 1;
                tid_terms_per_shape[m] = added_terms;
                Nt_alloc += added_terms * (m + tmp_pncc); // if not a TP step, processed_pncc is 0
            }
        }
    }
    if(cauchyEst->skip_post_mu)
        new_child_terms = (CauchyTerm*) malloc(shape_range * sizeof(CauchyTerm));
    else
        new_child_terms = (CauchyTerm*) malloc(Nt_alloc * sizeof(CauchyTerm));
    null_ptr_check(new_child_terms);

    int two_to_d = 1<<cauchyEst->d;
    int B_trivial[two_to_d];
    for(int i = 0; i < two_to_d; i++)
        B_trivial[i] = i;

    int Nt_new = 0;
    int old_term_count = 0;
    for(int m = 1; m < shape_range; m++)
    {
        int Nt_shape_tid = tid_terms_per_shape[m];
        if(Nt_shape_tid > 0)
        {
            // Allocate Memory for new terms
            BYTE_COUNT_TYPE new_shape;
            if(with_tp)
            {
                new_shape = m + tmp_pncc;
                if( (!DENSE_STORAGE) && (!skip_post_mu) )
                {
                    BYTE_COUNT_TYPE bytes_max_cells = (((BYTE_COUNT_TYPE)dce_helper->cell_counts_cen[new_shape]) / (1 + HALF_STORAGE)) * Nt_shape_tid * sizeof(BKEYS_TYPE);
                    gb_tables->extend_bp_tables(bytes_max_cells);
                }
            }
            else
                new_shape = m;
            
            if(!skip_post_mu)
            {
                BYTE_COUNT_TYPE new_num_terms = ((BYTE_COUNT_TYPE)Nt_shape_tid) * (new_shape+1);
                BYTE_COUNT_TYPE ps_bytes = new_num_terms * new_shape * sizeof(double);
                BYTE_COUNT_TYPE bs_bytes = new_num_terms * d * sizeof(double);
                coalign_store->extend_storage(ps_bytes, bs_bytes, d);
            }
            // End of memory allocation
            int Nt_shape = terms_per_shape[m]; 
            CauchyTerm* terms = cauchyEst->terms_dp[m];
            for(int i = tid; i < Nt_shape; i += n_tids)
            {
                old_term_count++;
                CauchyTerm* parent = terms + i;
                transfer_term_to_workspace(childterms_workspace, parent);
                // Run Time Propagation Routines
                if( with_tp )
                {
                    parent->time_prop(Phi, B, u, cmcc);
                    int m_tp = parent->tp_coalign(tmp_Gamma, tmp_beta, tmp_pncc);
                    if( (!DENSE_STORAGE) && (!skip_post_mu) )
                    {
                        if(parent->m <= parent->d)
                        {
                            gb_tables->set_term_bp_table_pointer( &(parent->enc_B), parent->cells_gtable_p, true);
                            memcpy(parent->enc_B, B_trivial, parent->cells_gtable_p * sizeof(BKEYS_TYPE));
                        }
                        else if(parent->m == parent->phc)
                        {
                            BKEYS B_parent = parent->enc_B;
                            gb_tables->set_term_bp_table_pointer( &(parent->enc_B), parent->cells_gtable_p, true);
                            memcpy(parent->enc_B, B_parent, parent->cells_gtable_p * sizeof(BKEYS_TYPE));
                        }
                        else
                        {
                            gb_tables->set_term_bp_table_pointer( &(parent->enc_B), dce_helper->cell_counts_cen[m_tp] / (1 + HALF_STORAGE), false);
                            if(FAST_TP_DCE)
                                make_time_prop_btable_fast(parent, dce_helper);
                            else
                                make_time_prop_btable(parent, dce_helper);
                            gb_tables->incr_chunked_bp_table_ptr(parent->cells_gtable);
                        }
                    }
                }
                int m_precoalign = parent->m; // HPs pre-MU Coalign
                // Run Measurement Update Routines
                CauchyTerm* children = skip_post_mu ? new_child_terms : new_child_terms + Nt_new;
                int num_children = parent->msmt_update(children, msmt, H, gamma, false, skip_post_mu, childterms_workspace);
                Nt_new += num_children;
                // Cache moment results here -- evaluate g / yei
                cache_moments(parent, children, num_children, root_point, &fz_chunk, cond_mean_chunk, cond_var_chunk, d);

                if(!skip_post_mu)
                {
                    // Normalize the parent, coalign the children, increment new terms per shape count,
                    // assign parent and child term elements to the coaligned storage buffers  
                    parent->normalize_hps(true);
                    new_terms_per_shape[parent->m]++;
                    coalign_store->set_term_ptrs(parent, m_precoalign);
                    for(int j = 0; j < num_children; j++)
                    {
                        new_terms_per_shape[children[j].mu_coalign()]++;
                        coalign_store->set_term_ptrs(children+j, m_precoalign);
                    }
                }
                else 
                    new_terms_per_shape[parent->m] += num_children + 1;
            }
        }
    }
    //printf("Thread %d has processed %d old terms, generating %d new_terms, %d total!\n", tid, old_term_count, Nt_new, Nt_new + old_term_count);
    dist_args->fz_chunk = fz_chunk;
    // Aggregate terms of similar shape into contiguous arrays
    if(!skip_post_mu)
    {    
        // Now coalesce terms into array of array storage for term reduction
        CauchyTerm** new_terms_dp = (CauchyTerm**) malloc( shape_range * sizeof(CauchyTerm*) );
        null_dptr_check((void**)new_terms_dp);
        for(int m = 0; m < shape_range; m++)
        {
            if(new_terms_per_shape[m] > 0)
            {
                new_terms_dp[m] = (CauchyTerm*) malloc( new_terms_per_shape[m] * sizeof(CauchyTerm) );
                null_ptr_check(new_terms_dp[m]);
            }
            else 
                new_terms_dp[m] = (CauchyTerm*) malloc(0);
        }   
        // Add parent terms 
        memset(new_terms_per_shape, 0, shape_range * sizeof(int));
        for(int shape = 0; shape < shape_range; shape++)
        {
            int Nt_shape_tid = tid_terms_per_shape[shape];
            if(Nt_shape_tid > 0)
            {
                int Nt_shape = terms_per_shape[shape];
                CauchyTerm* terms = cauchyEst->terms_dp[shape];
                for(int i = tid; i < Nt_shape; i += n_tids)
                {
                    int m = terms[i].m;
                    new_terms_dp[m][new_terms_per_shape[m]++] = terms[i];
                }
            }
        }
        // Add new child terms 
        for(int i = 0; i < Nt_new; i++)
        {
            int m = new_child_terms[i].m;
            new_terms_dp[m][new_terms_per_shape[m]++] = new_child_terms[i];
        }
        free(new_child_terms); // free new child term array ( now stored in DP )
        tmr_mu.toc(false);

        // Pass the new term pointers and counts to dist_args
        dist_args->new_terms_dp = new_terms_dp;
        dist_args->new_terms_per_shape = new_terms_per_shape;

        // Free unused memory
        if( with_tp && (!DENSE_STORAGE) )
            gb_tables->swap_btables();
        coalign_store->unallocate_unused_space();
    }
    else 
    {
        tmr_mu.toc(false);
        dist_args->new_terms_per_shape = new_terms_per_shape;
        free(new_child_terms);
    }

    free(tid_terms_per_shape);
    childterms_workspace->deinit();
    free(childterms_workspace);
    return NULL;
}




#endif //_CAUCHY_ESTIMATOR_HPP_


 //Compare code for FAST TP DCE vs REG TP DCE
/*
    int* tmp_benc_ptr = parent->enc_B;
    int cells_fast = parent->cells_gtable;
    int* b_enc2 = (int*) malloc(dce_helper.cell_counts_cen[m_tp] / (1 + HALF_STORAGE) * sizeof(int));
    parent->enc_B = b_enc2;
    parent->cells_gtable = 0;
    make_time_prop_btable(parent, &dce_helper);
    // Sort both benc arrays 
    qsort(tmp_benc_ptr, cells_fast, sizeof(int), sort_func_B_enc);
    qsort(parent->enc_B, parent->cells_gtable, sizeof(int), sort_func_B_enc);
    // Check the results of both b methods
    if(cells_fast != parent->cells_gtable)
    {
        printf("Shape m=%d, term %i\n", parent->m, i);
        printf("Btables do not match! Cell counts differ! (fast=%d cells, vs reg=%d)\n", cells_fast, parent->cells_gtable);
        printf("B from Fast method:\n");
        print_mat(tmp_benc_ptr, 1, cells_fast);
        printf("B from Reg method:\n");
        print_mat(parent->enc_B, 1, parent->cells_gtable);
        exit(1);
    }
    if(is_Bs_different(tmp_benc_ptr, parent->enc_B, cells_fast))
    {
        printf("Shape m=%d, term %i\n", parent->m, i);
        printf("Btables do not match! Counts are same (i.e, %d cells), but have different values!\n", cells_fast);
        printf("B from Fast method:\n");
        print_mat(tmp_benc_ptr, 1, cells_fast);
        printf("B from Reg method:\n");
        print_mat(parent->enc_B, 1, parent->cells_gtable);
        exit(1);
    }
    parent->enc_B = tmp_benc_ptr;
    parent->cells_gtable = cells_fast;
    free(b_enc2);
*/