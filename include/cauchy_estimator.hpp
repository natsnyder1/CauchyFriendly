#ifndef _CAUCHY_ESTIMATOR_HPP_
#define _CAUCHY_ESTIMATOR_HPP_


#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cauchy_util.hpp"
#include "cell_enumeration.hpp"
#include "cpu_linalg.hpp"
#include "eval_gs.hpp"
#include "gtable.hpp"
#include "random_variables.hpp"
#include "term_reduction.hpp"
#include "flattening.hpp"
#include "cpu_timer.hpp"
#include <cstdlib>
#include <pthread.h>

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
    int processed_cmcc;
    double msmt;
    double gamma;
    int tid;
    int n_tids;
};

struct CauchyEstimator
{
    int d; // state dimension
    int cmcc; // control matrix column count (columns of Gamma)
    int p; // number of measurements processed per step
    int Nt; // total number of terms
    int num_estimation_steps;
    int current_estimation_step;
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
    C_COMPLEX_TYPE* conditional_mean;
    C_COMPLEX_TYPE* conditional_variance;
    C_COMPLEX_TYPE fz;
    double G_SCALE_FACTOR;
    FastTermRedHelper ftr_helper; // Fast term reduction
    DiffCellEnumHelper* dce_helper; // DCE method
    ChunkedPackedTableStorage* gb_tables; // g and b-tables
    CoalignmentElemStorage* coalign_store;
    ReductionElemStorage reduce_store;
    ChildTermWorkSpace childterms_workspace;
    CauchyStats stats; // Used to Gather Memory Stats
    bool print_basic_info;
    bool skip_post_mu;
    int num_threads_tp_to_muc;


    CauchyEstimator(double* _A0, double* _p0, double* _b0, int _steps, int _d, int _cmcc, int _p, const bool _print_basic_info = true)
    {
        // Init state dimensions and shape / term counters
        Nt = 1;
        master_step = 0;
        d = _d;
        cmcc = _cmcc;
        p = _p;
        num_estimation_steps = p*_steps;
        int max_hp_shape = (_steps-1) * cmcc + d;
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
        
        dce_helper->init(shape_range-1, d, DCE_STORAGE_MULT);
        gb_tables->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        coalign_store->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        // Until needed, initialize the other gbtables / coalign store containers to 0
        for(int i = 1; i < NUM_CPUS; i++)
        {
            dce_helper[i].init(shape_range-1, d, DCE_STORAGE_MULT);
            gb_tables[i].init(0, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
            coalign_store[i].init(0, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        }

        ftr_helper.init(d, 1<<d);
        reduce_store.init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        
        childterms_workspace.init(shape_range-1, d);
        print_basic_info = _print_basic_info;
        // Initialize the first term 
        setup_first_term(&childterms_workspace, terms_dp[d], _A0, _p0, _b0, d);
        A0_init = _A0;
        p0_init = _p0;
        b0_init = _b0;

        // Set threading arguments
        num_threads_tp_to_muc = 1;
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
        C_COMPLEX_TYPE yei[d];

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
        assert(creal(fz) > 0);
        G_SCALE_FACTOR = RECIPRICAL_TWO_PI / creal(fz);
        C_COMPLEX_TYPE Ifz = I*fz; //CMPLX(cimag(fz), creal(fz)); // imaginary fz
        for(int i = 0; i < d; i++)
            conditional_mean[i] /= Ifz;

        for(int i = 0; i < d; i++)
        {
            for(int j = 0; j < d; j++)
            {
                conditional_variance[i*d + j] = (conditional_variance[i*d+j] / fz) - conditional_mean[i] * conditional_mean[j];
            }
        }
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
        C_COMPLEX_TYPE yei[d];
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

        C_COMPLEX_TYPE Ifz = I*fz; //CMPLX(cimag(fz), creal(fz)); // imaginary fz
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

    void step_tp_to_muc(double msmt, double* Phi, double* Gamma, double* beta, double* H, double gamma)
    {
        double tmp_Gamma[d*cmcc];
        double tmp_beta[cmcc];
        int tmp_cmcc = 0;
        const bool with_tp = ((master_step % p) == 0);
        fz = 0;
        memset(conditional_mean, 0, d * sizeof(C_COMPLEX_TYPE));
        memset(conditional_variance, 0, d*d*sizeof(C_COMPLEX_TYPE));
        CPUTimer tmr_mu;
        tmr_mu.tic();
        // Transpose, normalize, and pre-coalign Gamma and beta
        if(with_tp)
            tmp_cmcc = precoalign_Gamma_beta(Gamma, beta, cmcc, d, tmp_Gamma, tmp_beta);
        
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
                    Nt_alloc += terms_per_shape[m] * (m + tmp_cmcc);
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
                    new_shape = m + tmp_cmcc;
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
                        parent->time_prop(Phi);
                        int m_tp = parent->tp_coalign(tmp_Gamma, tmp_beta, tmp_cmcc);
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
                            }
                        }
                    }
                    int m_precoalign = parent->m; // HPs pre-MU Coalign
                    // Run Measurement Update Routines
                    CauchyTerm* children = skip_post_mu ? new_child_terms : new_child_terms + Nt_new;
                    int num_children = parent->msmt_update(children, msmt, H, gamma, false, skip_post_mu, &childterms_workspace);
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
                stats.print_total_estimator_memory(gb_tables, coalign_store, &reduce_store, Nt, true, num_threads_tp_to_muc, 1);
                //stats.print_cell_count_histograms(terms_dp, shape_range, terms_per_shape, dce_helper->cell_counts_cen);
            }
            // Free unused memory
            if(with_tp && !DENSE_STORAGE)
                gb_tables->swap_btables();
            reduce_store.reset();
            coalign_store->unallocate_unused_space();
        }
        else 
        {
            tmr_mu.toc(false);
            // Print Stats
            if(print_basic_info)
            {
                printf("Step %d/%d: (SKIP_LAST_STEP is used!)\n", master_step+1, num_estimation_steps);
                if(with_tp)
                    printf("TP to MU: Took %d ms\n", tmr_mu.cpu_time_used);
                else 
                    printf("MU to MU: Took %d ms\n", tmr_mu.cpu_time_used);
                printf("Total Terms after MUC: %d\n", Nt);
                printf("Note: No added memory pressure (SKIP_LAST_STEP=true)\n");
                print_conditional_mean_variance();
            }
            free(new_child_terms);
            free(new_terms_per_shape);
        }
        // I dont think this needs to be set, but if we do enter here after threading was used, threads used should be set to 1
        num_threads_tp_to_muc = 1;
    }

    void threaded_step_tp_to_muc(double msmt, double* Phi, double* Gamma, double* beta, double* H, double gamma)
    {
        double tmp_Gamma[d*cmcc];
        double tmp_beta[cmcc];
        int tmp_cmcc = 0;
        const bool with_tp = ((master_step % p) == 0);
        fz = 0;
        memset(conditional_mean, 0, d * sizeof(C_COMPLEX_TYPE));
        memset(conditional_variance, 0, d*d*sizeof(C_COMPLEX_TYPE));
        CPUTimer tmr_mu;
        tmr_mu.tic();
        // Transpose, normalize, and pre-coalign Gamma and beta
        if(with_tp)
            tmp_cmcc = precoalign_Gamma_beta(Gamma, beta, cmcc, d, tmp_Gamma, tmp_beta);
        
        int num_chunks = (Nt + MIN_TERMS_PER_THREAD_TP_TO_MUC -1) / MIN_TERMS_PER_THREAD_TP_TO_MUC;
        num_threads_tp_to_muc = num_chunks > NUM_CPUS ? NUM_CPUS : num_chunks;

        pthread_t tids[num_threads_tp_to_muc];
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
            tid_args[i].processed_cmcc = tmp_cmcc;
            tid_args[i].msmt = msmt;
            tid_args[i].gamma = gamma;
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
        free(tid_args);
        // Print Stats
        tmr_mu.toc(false);
        if(print_basic_info)
        {
            printf("Step %d/%d:\n", master_step+1, num_estimation_steps);
            if(with_tp)
                printf("TP to MUC [Threaded %d]: Took %d ms\n", num_threads_tp_to_muc, tmr_mu.cpu_time_used);
            else 
                printf("MU to MUC [Threaded %d]: Took %d ms\n", num_threads_tp_to_muc, tmr_mu.cpu_time_used);
            printf("Total Terms after MUC: %d\n", Nt);
            for(int m = 1; m < shape_range; m++)
                if(terms_per_shape[m] > 0)
                    printf("Shape %d has %d terms\n", m, terms_per_shape[m]);
            print_conditional_mean_variance();
            stats.print_total_estimator_memory(gb_tables, coalign_store, &reduce_store, Nt, true, num_threads_tp_to_muc, 1);
            //stats.print_cell_count_histograms(terms_dp, shape_range, terms_per_shape, dce_helper->cell_counts_cen);
        }
        // Unallocate the reduce_storage
        reduce_store.reset();
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
                tmr.tic();
                CauchyTerm* terms = terms_dp[m];
                int Nt_shape = terms_per_shape[m];
                memcpy(ftr_helper.F_TR, ftr_helper.F, Nt_shape * sizeof(int) );

                build_ordered_point_maps(
                    terms,
                    ftr_helper.ordered_points, 
                    ftr_helper.forward_map, 
                    ftr_helper.backward_map, 
                    Nt_shape,
                    d, false);
      
                fast_term_reduction(
                    terms,
                    ftr_helper.F_TR, 
                    ftr_helper.ordered_points,
                    ftr_helper.forward_map, 
                    ftr_helper.backward_map,
                    REDUCTION_EPS, Nt_shape, m, d);
                // Build the term reduction lists: 
                // example ffa.Fs[3] = [7,9,11] means terms at indices 7,9,11 reduce with the term at index 3
                ForwardFlagArray ffa(ftr_helper.F_TR, Nt_shape);
                tmr.toc(false);
                if(print_basic_info)
                    printf("[FTR Shape %d:], %d/%d terms remain (Took %d ms)\n", m, ffa.num_terms_after_reduction, Nt_shape, tmr.cpu_time_used);

                tmr.tic();
                int max_Nt_reduced_shape = ffa.num_terms_after_reduction; // Max number of terms after reduction (before term approximation)
                int Nt_reduced_shape = 0;
                int Nt_removed_shape = 0;
                int max_cells_shape = !DENSE_STORAGE ? dce_helper->cell_counts_cen[m] / (1 + HALF_STORAGE) : (1<<m) / (1+HALF_STORAGE);
                int dce_temp_hashtable_size = max_cells_shape * dce_helper->storage_multiplier;
                gb_tables->extend_gtables(max_cells_shape, max_Nt_reduced_shape);
                BYTE_COUNT_TYPE ps_bytes = ((BYTE_COUNT_TYPE)max_Nt_reduced_shape) * m * sizeof(double);
                BYTE_COUNT_TYPE bs_bytes = ((BYTE_COUNT_TYPE)max_Nt_reduced_shape) * d * sizeof(double);
                reduce_store.extend_storage(ps_bytes, bs_bytes, d);
                if(!DENSE_STORAGE)
                    gb_tables->extend_btables(max_cells_shape, max_Nt_reduced_shape);
                ftr_terms_dp[m] = (CauchyTerm*) malloc( max_Nt_reduced_shape * sizeof(CauchyTerm) );
                null_ptr_check(ftr_terms_dp[m]);
                CauchyTerm* ftr_terms = ftr_terms_dp[m];
                // Now make B-Tables, G-Tables, for each reduction group
                int** forward_F = ffa.Fs;
                int* forward_F_counts = ffa.F_counts;
                int* backward_F = ftr_helper.F_TR;             
                for(int j = 0; j < Nt_shape; j++)
                {
                    // Check whether we need to process term j (if it has reductions or is unique)
                    if(backward_F[j] == j)
                    {
                        int rt_idx = j;
                        CauchyTerm* child_j = terms + rt_idx;
                        // Make the Btable if not an old term

                        if(DENSE_STORAGE)
                        {
                            child_j->enc_B = B_dense;
                            child_j->cells_gtable = max_cells_shape;
                        }
                        else
                        {
                            if( (child_j->is_new_child) )
                            {
                                // Set the child btable memory position
                                BKEYS parent_B = child_j->enc_B;
                                int num_cells_parent = child_j->cells_gtable;
                                gb_tables->set_term_btable_pointer(&(child_j->enc_B), max_cells_shape, false);
                                make_new_child_btable(child_j, 
                                    parent_B, num_cells_parent,
                                    dce_helper->B_mu_hash, num_cells_parent * dce_helper->storage_multiplier,
                                    dce_helper->B_coal_hash, dce_temp_hashtable_size,
                                    dce_helper->B_uncoal, dce_helper->F);
                            }
                        }
                        //printf("B%d is:\n", Nt_reduced_shape);
                        //print_B_encoded(child_j->enc_B, child_j->cells_gtable, child_j->m, true);
                        
                        // set memory position of the child gtable
                        // Make the g-table of the root term
                        gb_tables->set_term_gtable_pointer(&(child_j->gtable), child_j->cells_gtable, false);
                        // If the term is negligable (is approximated out), we need to search for a new "root"
                        if( make_gtable(child_j, G_SCALE_FACTOR) )
                            rt_idx = -1;
                        else
                        {
                            gb_tables->incr_chunked_gtable_ptr(child_j->cells_gtable);
                            if(!DENSE_STORAGE)
                                if(child_j->is_new_child)
                                    gb_tables->incr_chunked_btable_ptr(child_j->cells_gtable);
                        }
                        
                        int num_term_combos = forward_F_counts[j];
                        int k = 0;
                        // If the root term has been approximated out, we need to search through its term combinations to find a new term to take the place as root
                        if(rt_idx == -1)
                        {
                            int num_cells_of_red_group = child_j->cells_gtable;
                            BKEYS btable_for_red_group = child_j->enc_B;
                            double* A_lfr = child_j->A; // HPA of the last failed root
                            while(k < num_term_combos)
                            {
                                int cp_idx = forward_F[j][k++];
                                CauchyTerm* child_k = terms + cp_idx;
                                // The btable of all terms in this reduction group are similar
                                // The only difference is the orientation of their hyperplanes
                                // Update the Btable of the last potential root for child_k
                                // Use the memory space currently pointed to by child_j only if child_k is not a parent 
                                if(DENSE_STORAGE)
                                {
                                    child_k->enc_B = btable_for_red_group;
                                    child_k->cells_gtable = num_cells_of_red_group;
                                    child_k->gtable = child_j->gtable;
                                }
                                else 
                                {
                                    if(child_k->is_new_child)
                                    {
                                        child_k->enc_B = btable_for_red_group;
                                        child_k->cells_gtable = num_cells_of_red_group;
                                        update_btable(A_lfr, child_k->enc_B, child_k->A, NULL, child_k->cells_gtable, m, d);
                                        // Set memory position of child gtable k here
                                        // This child can use the gtable memory position of child_j (since it was approximated out)
                                        child_k->gtable = child_j->gtable;
                                    }
                                    // If child_k is a parent, its B is already in memory, no need to use new space
                                    else 
                                    {   
                                        btable_for_red_group = child_k->enc_B;
                                        if(child_k->cells_gtable == num_cells_of_red_group)
                                        {
                                            // Set memory position of child gtable k here
                                            // This child can use the gtable memory position of child_j (since it was approximated out)
                                            child_k->gtable = child_j->gtable;
                                        }
                                        // Only in the case of numerical round off error can two reducing terms
                                        // have the same HPA (up to +/- direction of their normals)
                                        // but a different numbers of cells. 
                                        // So in the case where the two have different cell counts, do the following:

                                        // if the cells are less, 
                                        // update cell_count of red group
                                        // set gtable to child_j memory range (it will fit with certainty)
                                        else if(child_k->cells_gtable < num_cells_of_red_group)
                                        {
                                            num_cells_of_red_group = child_k->cells_gtable;
                                            child_k->gtable = child_j->gtable;
                                        }
                                        // if the cells are greater
                                        // update cell_count of red group
                                        // recheck gtable pointer memory address
                                        else
                                        {
                                            num_cells_of_red_group = child_k->cells_gtable; 
                                            gb_tables->set_term_gtable_pointer(&(child_k->gtable), num_cells_of_red_group, false); 
                                        }
                                    }
                                }
                                // If child term k is not approximated out, it becomes root
                                if( !make_gtable(child_k, G_SCALE_FACTOR) )
                                {
                                    rt_idx = cp_idx;
                                    if(!DENSE_STORAGE)
                                        if(child_k->is_new_child)
                                            gb_tables->incr_chunked_btable_ptr(child_k->cells_gtable);
                                    gb_tables->incr_chunked_gtable_ptr(child_k->cells_gtable);
                                    child_j = child_k;
                                    break;
                                }
                                else
                                    A_lfr = child_k->A;
                            }
                        }
                        // For all terms combinations left, create thier g-table. If it is not approximated out, add it to the root g-table
                        while(k < num_term_combos)
                        {
                            int cp_idx = forward_F[j][k++];
                            CauchyTerm* child_k = terms + cp_idx;
                            // Set memory position of child gtable k here
                            // Make the Btable if not an old term

                            if(DENSE_STORAGE)
                            {
                                child_k->cells_gtable = child_j->cells_gtable;
                                child_k->enc_B = B_dense;
                            }
                            else
                            {
                                if(child_k->is_new_child)
                                {
                                    // Set the child btable memory position
                                    child_k->cells_gtable = child_j->cells_gtable;
                                    gb_tables->set_term_btable_pointer(&(child_k->enc_B), child_k->cells_gtable, false);
                                    update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, child_k->cells_gtable, m, d);
                                }
                                else
                                {
                                    // To deal with the case where numerical instability causes cell counts to be different, 
                                    // If the cell counts are different (due to instability), update child_k's btable to be compatible with the root
                                    if(child_k->cells_gtable != child_j->cells_gtable)
                                    {
                                        printf(RED"[BIG WARN FTR/Make Gtables:] child_k->cells_gtable != child_j->cells_gtable. We have code below to fix this! But EXITING now until this is commented out!" NC "\n");
                                        exit(1);
                                        // If child_k has more than child_j's cells,
                                        // Downgrade child_k to be equal to child_j
                                        if(child_k->cells_gtable > child_j->cells_gtable)
                                        {
                                            child_k->cells_gtable = child_j->cells_gtable;
                                            update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, child_k->cells_gtable, m, d);
                                        }
                                        // If child_k has less than child_j's cells, 
                                        // Upgrade child_k to be equal to child_j
                                        // Here we need to abandon the old btable memory location of child_k and acquire a new one 
                                        // This is to keep the btables consistent, 
                                        // child_k's btable then be set to the (re-oriented) child_j's btable.
                                        else
                                        {
                                            child_k->cells_gtable = child_j->cells_gtable;
                                            gb_tables->set_term_btable_pointer(&(child_k->enc_B), child_k->cells_gtable, false);
                                            update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, child_k->cells_gtable, m, d);
                                        }
                                    }
                                }
                            }
                            gb_tables->set_term_gtable_pointer(&(child_k->gtable), child_k->cells_gtable, false);
                            // If child term k is not approximated out, we can add the gtables together
                            if( !make_gtable(child_k, G_SCALE_FACTOR) )
                            { 
                                add_gtables(child_j, child_k);
                                // No need to re-increment gb_table pointers
                                // This is because for the remainder of the terms to be combined, we can use this memory space again
                            }
                        }
                        // If we found a root term, then increase the (reduced) term count
                        if(rt_idx != -1)
                        {
                            child_j->become_parent();
                            reduce_store.set_term_ptrs(child_j);
                            ftr_terms[Nt_reduced_shape++] = *child_j;
                        }
                        else
                            Nt_removed_shape++;

                        //terms[j].become_null();
                        //for(int l = 0; l < num_term_combos; l++)
                        //    terms[forward_F[j][l]].become_null();
                    }
                }
                // After term reduction and g-evaluation 
                Nt_reduced += Nt_reduced_shape;
                Nt_removed += Nt_removed_shape;
                terms_per_shape[m] = Nt_reduced_shape;
                if(WITH_TERM_APPROXIMATION)
                {
                    // may not be worth the expense of shrinking...well see
                    ftr_terms_dp[m] = (CauchyTerm*) realloc(ftr_terms_dp[m], Nt_reduced * sizeof(CauchyTerm) );
                    null_ptr_check(ftr_terms_dp[m]);
                }
                tmr.toc(false);
                if(print_basic_info)
                    printf("[Eval Gtables shape %d:] Took %d ms\n", m, tmr.cpu_time_used);
    
                if(print_basic_info && WITH_TERM_APPROXIMATION)
                        printf("[Term Approx Shape %d:] %d/%d (%d were removed)\n", m, Nt_reduced_shape, max_Nt_reduced_shape, Nt_removed_shape);
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
            stats.print_total_estimator_memory(gb_tables, coalign_store, &reduce_store, Nt, false, num_threads_tp_to_muc, 1);
            stats.print_cell_count_histograms(terms_dp, shape_range, terms_per_shape, dce_helper->cell_counts_cen);
        }
        
        // Deallocate unused or unneeded memory
        tmr.tic();
        for(int i = 0; i < num_threads_tp_to_muc; i++)
            coalign_store[i].reset(); 
        reduce_store.unallocate_unused_space();
        gb_tables->swap_gtables();
        tmr.toc(false);

        // Compute moments after FTR
        if(print_basic_info)
        {
            printf("Deallocating memory after FTR took %d ms\n", tmr.cpu_time_used);   
            compute_moments(false);
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
            reduce_store.set_term_ptrs(terms + i);
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
    }

    void step(double msmt, double* Phi, double* Gamma, double* beta, double* H, double gamma)
    {
        CPUTimer tmr;
        tmr.tic();
        skip_post_mu = SKIP_LAST_STEP && (master_step == (num_estimation_steps-1)); 
        if(master_step == 0)
            step_first(msmt, H, gamma);
        else
        {
            if( (NUM_CPUS == 1) || (Nt < MIN_TERMS_PER_THREAD_TP_TO_MUC) )
                step_tp_to_muc(msmt, Phi, Gamma, beta, H, gamma);
            else
                threaded_step_tp_to_muc(msmt, Phi, Gamma, beta, H, gamma);
            // Now thread fast term reduction    
            fast_term_reduction_and_create_gtables();
        }
        master_step++;
        tmr.toc(false);
        printf("Step %d took %d ms\n", master_step, tmr.cpu_time_used);
    }

    void reset()
    {
        CPUTimer tmr;
        tmr.tic();
        for(int i = 0; i < NUM_CPUS; i++)
        {
            dce_helper[i].deinit();
            gb_tables[i].deinit(); // g and b-tables
            coalign_store[i].deinit();
        }

        ftr_helper.deinit();
        reduce_store.deinit();

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

        // Re-init first term
        setup_first_term(&childterms_workspace, terms_dp[d], A0_init, p0_init, b0_init, d);

        //Re-init helpers
        dce_helper->init(shape_range-1, d, DCE_STORAGE_MULT);
        gb_tables->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        coalign_store->init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        for(int i = 1; i < NUM_CPUS; i++)
        {
            dce_helper[i].init(shape_range-1, d, DCE_STORAGE_MULT);
            gb_tables[i].init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
            coalign_store[i].init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        }

        ftr_helper.init(d, 1<<d);
        reduce_store.init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);

        num_threads_tp_to_muc = 1;
        tmr.toc(false);
        printf("Resetting CF Took %d ms\n", tmr.cpu_time_used);
    }

    ~CauchyEstimator()
    {
        free(terms_per_shape);
        free(root_point);
        free(conditional_mean);
        free(conditional_variance);

        for(int i = 0; i < NUM_CPUS; i++)
        {
            dce_helper[i].deinit();
            gb_tables[i].deinit(); // g and b-tables
            coalign_store[i].deinit();
        }
        // Deallocate terms 
        for(int i = 0; i < shape_range; i++)
            free(terms_dp[i]);
        free(terms_dp);
        
        ftr_helper.deinit();
        reduce_store.deinit();
        childterms_workspace.deinit();
        if(DENSE_STORAGE)
            free(B_dense);
        
        free(dce_helper);
        free(gb_tables);
        free(coalign_store);
    }

};



void cache_moments(CauchyTerm* parent, CauchyTerm* children, int num_children, 
    double* root_point, C_COMPLEX_TYPE* fz, C_COMPLEX_TYPE* conditional_mean, C_COMPLEX_TYPE* conditional_variance, const int d)
{
    C_COMPLEX_TYPE g_val;
    C_COMPLEX_TYPE yei[d];

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
    // If not a TP step, these are NULL and tmp_cmcc is 0
    double* tmp_Gamma = dist_args->processed_Gamma;
    double* tmp_beta = dist_args->processed_beta;
    int tmp_cmcc = dist_args->processed_cmcc;
    double* Phi = dist_args->Phi;
    // Bring in H, gamma, msmt
    double* H = dist_args->H;
    double gamma = dist_args->gamma;
    double msmt = dist_args->msmt;

    // Allocate structures for the maximum number of new terms we'd generate at this step
    int* new_terms_per_shape = (int*) calloc(shape_range, sizeof(int));
    null_ptr_check(new_terms_per_shape);
    int* tid_terms_per_shape = (int*) calloc(shape_range, sizeof(int));
    null_ptr_check(tid_terms_per_shape);
    CauchyTerm* new_child_terms;
    if(cauchyEst->skip_post_mu)
        new_child_terms = (CauchyTerm*) malloc(shape_range * sizeof(CauchyTerm));
    else
    {
        int Nt_alloc = 0;
        for(int m = 1; m < shape_range; m++)
        {
            if(terms_per_shape[m] > 0)
            {
                if(terms_per_shape[m] < n_tids)
                {
                    if(tid < terms_per_shape[m])
                    {
                        Nt_alloc += (m + tmp_cmcc);
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
                    Nt_alloc += added_terms * (m + tmp_cmcc); // if not a TP step, processed_cmcc is 0
                }
            }
        }
        new_child_terms = (CauchyTerm*) malloc(Nt_alloc * sizeof(CauchyTerm));
    }
    null_ptr_check(new_child_terms);
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
                new_shape = m + tmp_cmcc;
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
                    parent->time_prop(Phi);
                    int m_tp = parent->tp_coalign(tmp_Gamma, tmp_beta, tmp_cmcc);
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
        free(new_child_terms);
        free(new_terms_per_shape);
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