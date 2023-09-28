#ifndef _CAUCHY_ESTIMATOR_HPP_
#define _CAUCHY_ESTIMATOR_HPP_


#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cauchy_util.hpp"
#include "cpu_linalg.hpp"
#include "eval_gs.hpp"
#include "gtable.hpp"
#include "random_variables.hpp"
#include "term_reduction.hpp"
#include "flattening.hpp"
#include "cpu_timer.hpp"
#include <cstdlib>
#include <cstring>

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
    DiffCellEnumHelper dce_helper; // DCE method
    ChunkedPackedTableStorage gb_tables; // g and b-tables
    CoalignmentElemStorage coalign_store;
    ReductionElemStorage reduce_store;
    ChildTermWorkSpace childterms_workspace;
    CauchyStats stats; // Used to Gather Memory Stats
    bool print_basic_info;
    bool skip_post_mu;

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
        ftr_helper.init(d, 1<<d);
        dce_helper.init(shape_range-1, d, DCE_STORAGE_MULT, cmcc);
        gb_tables.init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        coalign_store.init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        reduce_store.init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        childterms_workspace.init(shape_range-1, d);
        print_basic_info = _print_basic_info;
        // Initialize the first term 
        setup_first_term(&childterms_workspace, terms_dp[d], _A0, _p0, _b0, d);
        A0_init = _A0;
        p0_init = _p0;
        b0_init = _b0;
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
                break;
            case GTABLE_BINSEARCH_STORAGE:
                lookup_g_numerator = (LOOKUP_G_NUMERATOR_TYPE)g_num_binsearch;
                gtable_insert = (GTABLE_INSERT_TYPE) g_insert_binsearch;
                gtable_add = (GTABLE_ADD_TYPE) gs_add_binsearch;
                gtable_p_find = (GTABLE_P_FIND_TYPE) gp_find_binsearch;
                break;
            case GTABLE_DENSE_STORAGE:
                lookup_g_numerator = (LOOKUP_G_NUMERATOR_TYPE)g_num_dense;
                gtable_insert = (GTABLE_INSERT_TYPE) g_insert_dense;
                gtable_add = (GTABLE_ADD_TYPE) gs_add_dense;
                gtable_p_find = NULL;
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
        int max_cell_count = cell_count_central(shape_range-1, d);
        BYTE_COUNT_TYPE bytes_max_hp_gtable = (BYTE_COUNT_TYPE)(GTABLE_SIZE_MULTIPLIER * sizeof(GTABLE_TYPE) * max_cell_count / (1 + HALF_STORAGE) + 10); // a little extra padding for floating point calcs
        if( bytes_max_hp_gtable > CP_STORAGE_PAGE_SIZE)
        {
            printf(RED "[Error Cauchy Estimator Initialization:]\nCP_STORAGE_PAGE_SIZE is smaller than the memory required to store the largest HPA's gtable (which requires %llu bytes)\nIncrease STORAGE_PAGE_SIZE in cauchy_util.hpp!\nExiting!" NC"\n", bytes_max_hp_gtable );
            exit(1);
        }
        // Make sure that the first d+1 gtables will fit into a single page
        BYTE_COUNT_TYPE bytes_first_gtables = (BYTE_COUNT_TYPE)(GTABLE_SIZE_MULTIPLIER * sizeof(GTABLE_TYPE) * (d+1) * (1<<d) / (1 + HALF_STORAGE) + 10); // a little extra padding for floating point calcs
        if(bytes_first_gtables > CP_STORAGE_PAGE_SIZE)
        {
            printf(RED "[Error Cauchy Estimator Initialization:]\nCP_STORAGE_PAGE_SIZE is smaller than the memory required to store the first STATE_DIM+1 gtable (which requires %llu bytes)\nIncrease STORAGE_PAGE_SIZE in cauchy_util.hpp!\nExiting!" NC"\n", bytes_first_gtables );
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
        if( print_basic_info )
        {
            const int precision = 16;
            printf("Moment Information (after MU) at step %d, MU %d/%d\n", (master_step+1) / p, (master_step % p)+1, p);
            printf("fz: %.*lf + %.*lfj\n", precision, creal(fz), precision, cimag(fz));
            printf("Conditional Mean:\n");
            print_cmat(conditional_mean, 1, d, precision);
            printf("Conditional Variance:\n");
            print_cmat(conditional_variance, d, d, precision);
        }
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
        // Transpose, normalize, and pre-coalign Gamma and beta
        if(with_tp)
            tmp_cmcc = precoalign_Gamma_beta(Gamma, beta, cmcc, d, tmp_Gamma, tmp_beta);
        
        // Allocate structures for the maximum number of new terms we'd generate at this step
        int* new_terms_per_shape = (int*) calloc(shape_range, sizeof(int));
        int Nt_alloc = 0;
        for(int m = 1; m < shape_range; m++)
            if(terms_per_shape[m] > 0)
                Nt_alloc += terms_per_shape[m] * (m + tmp_cmcc);
        CauchyTerm* new_child_terms = (CauchyTerm*) malloc(Nt_alloc * sizeof(CauchyTerm));
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
                    if(!DENSE_STORAGE)
                    {
                        BYTE_COUNT_TYPE bytes_max_cells = (((BYTE_COUNT_TYPE)dce_helper.cell_counts_cen[new_shape]) / (1 + HALF_STORAGE)) * Nt_shape * sizeof(BKEYS_TYPE);
                        gb_tables.extend_bp_tables(bytes_max_cells);
                    }
                }
                else
                    new_shape = m;
                BYTE_COUNT_TYPE new_num_terms = ((BYTE_COUNT_TYPE)Nt_shape) * (new_shape+1);
                BYTE_COUNT_TYPE ps_bytes = new_num_terms * new_shape * sizeof(double);
                BYTE_COUNT_TYPE bs_bytes = new_num_terms * d * sizeof(double);
                coalign_store.extend_storage(ps_bytes, bs_bytes, d);
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
                        parent->tp_coalign(tmp_Gamma, tmp_beta, tmp_cmcc);
                        if(parent->m == parent->phc)
                        {
                            BKEYS B_parent = parent->enc_B;
                            gb_tables.set_term_bp_table_pointer( &(parent->enc_B), parent->cells_gtable_p, true);
                            memcpy(parent->enc_B, B_parent, parent->cells_gtable_p * sizeof(BKEYS_TYPE));
                        }
                        else
                        {
                            gb_tables.set_term_bp_table_pointer( &(parent->enc_B), dce_helper.cell_counts_cen[m] / (1 + HALF_STORAGE), false);
                            make_time_prop_btable(terms + i, &dce_helper);
                            gb_tables.incr_chunked_bp_table_ptr(parent->cells_gtable);        
                        }
                    }
                    int m_precoalign = parent->m; // HPs pre-MU Coalign
                    // Run Measurement Update Routines
                    CauchyTerm* children = new_child_terms + Nt_new;
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
                        coalign_store.set_term_ptrs(parent, m_precoalign);
                        for(int j = 0; j < num_children; j++)
                        {
                            new_terms_per_shape[children[j].mu_coalign()]++;
                            coalign_store.set_term_ptrs(children+j, m_precoalign);
                        }
                    }
                }
            }
        }
        // Finalize cached moment information
        finalize_cached_moments();
        // Aggregate terms of similar shape into contiguous arrays
        if(!skip_post_mu)
        {    
            // Swap bp_tables to btable memory pointer if we time proped
            if(with_tp)
                gb_tables.swap_btables();
            // Free unused memory
            reduce_store.reset();
            coalign_store.unallocate_unused_space();
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
            Nt += Nt_new;
            // Swap terms_dp and terms_per_shape
            ptr_swap<CauchyTerm*>(&terms_dp, &new_terms_dp);
            ptr_swap<int>(&terms_per_shape, &new_terms_per_shape);
            free(new_terms_dp);
            free(new_terms_per_shape);
            free(new_child_terms);
        }
    }

    void fast_term_reduction_and_create_gtables()
    {
        if(skip_post_mu)
            return;
        // bs[i] are the bs list (array) of all terms with i hyperplanes
        // shape_idxs[i][j] is the index of the "bs[i] + d*j" vector in the term list
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
                int max_cells_shape = !DENSE_STORAGE ? dce_helper.cell_counts_cen[m] / (1 + HALF_STORAGE) : (1<<m) / (1+HALF_STORAGE);
                int dce_temp_hashtable_size = max_cells_shape * dce_helper.storage_multiplier;
                gb_tables.extend_gtables(max_cells_shape, max_Nt_reduced_shape);
                BYTE_COUNT_TYPE ps_bytes = ((BYTE_COUNT_TYPE)max_Nt_reduced_shape) * m * sizeof(double);
                BYTE_COUNT_TYPE bs_bytes = ((BYTE_COUNT_TYPE)max_Nt_reduced_shape) * d * sizeof(double);
                reduce_store.extend_storage(ps_bytes, bs_bytes, d);
                if(!DENSE_STORAGE)
                    gb_tables.extend_btables(max_cells_shape, max_Nt_reduced_shape);
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
                                gb_tables.set_term_btable_pointer(&(child_j->enc_B), max_cells_shape, false);
                                make_new_child_btable(child_j, 
                                    parent_B, num_cells_parent,
                                    dce_helper.B_mu_hash, num_cells_parent * dce_helper.storage_multiplier,
                                    dce_helper.B_coal_hash, dce_temp_hashtable_size,
                                    dce_helper.B_uncoal, dce_helper.F);
                            }
                        }
                        //printf("B%d is:\n", Nt_reduced_shape);
                        //print_B_encoded(child_j->enc_B, child_j->cells_gtable, child_j->m, true);
                        
                        // set memory position of the child gtable
                        // Make the g-table of the root term
                        gb_tables.set_term_gtable_pointer(&(child_j->gtable), child_j->cells_gtable, false);
                        // If the term is negligable (is approximated out), we need to search for a new "root"
                        if( make_gtable(child_j, G_SCALE_FACTOR) )
                            rt_idx = -1;
                        else
                        {
                            gb_tables.incr_chunked_gtable_ptr(child_j->cells_gtable);
                            if(!DENSE_STORAGE)
                                if(child_j->is_new_child)
                                    gb_tables.incr_chunked_btable_ptr(child_j->cells_gtable);
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
                                            gb_tables.set_term_gtable_pointer(&(child_k->gtable), num_cells_of_red_group, false); 
                                        }
                                    }
                                }
                                // If child term k is not approximated out, it becomes root
                                if( !make_gtable(child_k, G_SCALE_FACTOR) )
                                {
                                    rt_idx = cp_idx;
                                    if(!DENSE_STORAGE)
                                        if(child_k->is_new_child)
                                            gb_tables.incr_chunked_btable_ptr(child_k->cells_gtable);
                                    gb_tables.incr_chunked_gtable_ptr(child_k->cells_gtable);
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
                                    gb_tables.set_term_btable_pointer(&(child_k->enc_B), child_k->cells_gtable, false);
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
                                            gb_tables.set_term_btable_pointer(&(child_k->enc_B), child_k->cells_gtable, false);
                                            update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, child_k->cells_gtable, m, d);
                                        }
                                    }
                                }
                            }
                            gb_tables.set_term_gtable_pointer(&(child_k->gtable), child_k->cells_gtable, false);
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
        tmr.tic();
        coalign_store.reset(); 
        reduce_store.unallocate_unused_space();
        ptr_swap<CauchyTerm*>(&terms_dp, &ftr_terms_dp);
        free(ftr_terms_dp);
        if(print_basic_info)
            stats.print_total_estimator_memory(&gb_tables, Nt, shape_range, terms_per_shape, d);
        gb_tables.swap_gtables();
        Nt = Nt_reduced;
        // Deallocate this round of FTR helpers
        tmr.toc(false);
        // Compute moments after FTR
        if(print_basic_info)
        {
            printf("Deallocating memory after FTR took %d ms\n", tmr.cpu_time_used);
            printf("Total Terms after FTR: %d\n", Nt);
            for(int i = 0; i < shape_range; i++)
                if(terms_per_shape[i] > 0)
                    printf("After FTR: Shape %d has %d terms\n", i, terms_per_shape[i]);
            stats.print_cell_count_histograms(terms_dp, shape_range, terms_per_shape, dce_helper.cell_counts_cen);
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
            terms[i].cells_gtable = dce_helper.cell_counts_cen[d] / (1 + HALF_STORAGE);
            if(!DENSE_STORAGE)
                gb_tables.set_term_btable_pointer( &(terms[i].enc_B), terms[i].cells_gtable, true);
            else
                terms[i].enc_B = B_dense;
            gb_tables.set_term_gtable_pointer(&(terms[i].gtable), terms[i].cells_gtable, true);
            make_gtable_first(terms + i, G_SCALE_FACTOR);
            terms[i].become_parent();
        }
        gb_tables.swap_gtables();
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
            step_tp_to_muc(msmt, Phi, Gamma, beta, H, gamma);
            fast_term_reduction_and_create_gtables();
        }
        master_step++;
        tmr.toc(false);
        printf("Step %d took %d ms\n", master_step, tmr.cpu_time_used);
    }

    void reset()
    {
        ftr_helper.deinit();
        dce_helper.deinit();
        gb_tables.deinit(); // g and b-tables
        reduce_store.deinit();
        coalign_store.deinit();
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
        ftr_helper.init(d, 1<<d);
        dce_helper.init(shape_range-1, d, DCE_STORAGE_MULT, cmcc);
        gb_tables.init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        reduce_store.init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
        coalign_store.init(1, CP_STORAGE_PAGE_SIZE, CP_STORAGE_ALLOC_METHOD);
    }

    ~CauchyEstimator()
    {
        free(terms_per_shape);
        free(root_point);
        free(conditional_mean);
        free(conditional_variance);
        ftr_helper.deinit();
        dce_helper.deinit();
        gb_tables.deinit(); // g and b-tables

        // Deallocate terms 
        for(int i = 0; i < shape_range; i++)
            free(terms_dp[i]);
        free(terms_dp);
        coalign_store.deinit();
        reduce_store.deinit();
        childterms_workspace.deinit();
        if(DENSE_STORAGE)
            free(B_dense);
    }

};

#endif //_CAUCHY_ESTIMATOR_HPP_