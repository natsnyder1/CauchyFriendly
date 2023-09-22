#ifndef _CAUCHY_ESTIMATOR_HPP_
#define _CAUCHY_ESTIMATOR_HPP_


#include "cauchy_constants.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cauchy_util.hpp"
#include "cpu_linalg.hpp"
#include "eval_gs.hpp"
#include "random_variables.hpp"
#include "term_reduction.hpp"
#include "flattening.hpp"
#include "cpu_timer.hpp"
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
    CauchyTerm* terms; // terms of the cauchy estimator
    double* A0_init;
    double* p0_init;
    double* b0_init;
    int num_gtable_chunks; 
    int num_btable_chunks;
    int* terms_per_shape;
    int shape_range;
    double* root_point;
    C_COMPLEX_TYPE* conditional_mean;
    C_COMPLEX_TYPE* conditional_variance;
    C_COMPLEX_TYPE fz;
    double G_SCALE_FACTOR;
    FastTermRedHelper ftr_helper;
    DiffCellEnumHelper dce_helper;
    ChunkedPackedTableStorage gb_tables; // g and b-tables
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
        terms = (CauchyTerm*) malloc( (d+1)*sizeof(CauchyTerm) );
        null_ptr_check(terms);
        terms_per_shape = (int*) calloc(shape_range , sizeof(int));
        null_ptr_check(terms_per_shape);
        
        // Init Moments
        conditional_mean = (C_COMPLEX_TYPE*) malloc(d * sizeof(C_COMPLEX_TYPE));
        null_ptr_check(conditional_mean);
        conditional_variance = (C_COMPLEX_TYPE*) malloc(d * d * sizeof(C_COMPLEX_TYPE));
        null_ptr_check(conditional_variance);
        
        // Init first term
        terms->init(d, d);
        memcpy(terms->A, _A0, d*d*sizeof(double));
        memcpy(terms->p, _p0, d*sizeof(double));
        memcpy(terms->b, _b0, d*sizeof(double));
        A0_init = _A0;
        p0_init = _p0;
        b0_init = _b0;

        // Init Gtable helpers
        root_point = (double*) malloc( d * sizeof(double));
        null_ptr_check(root_point);
        for(int i = 0; i < d; i++)
            root_point[i] = 1.0 + random_uniform();
        // Set the function pointers necessary to run all methods
        set_function_pointers();
        error_checks();
        ftr_helper.init(d, 1<<d);
        dce_helper.init(shape_range-1, d, DCE_STORAGE_MULT, cmcc);
        BYTE_COUNT_TYPE gb_init_bytes = (BYTE_COUNT_TYPE)((d+1) * GTABLE_SIZE_MULTIPLIER * (1<<d) * sizeof(GTABLE_TYPE));
        int start_pages = (gb_init_bytes + GB_TABLE_STORAGE_PAGE_SIZE - 1) / GB_TABLE_STORAGE_PAGE_SIZE;
        gb_tables.init(start_pages, GB_TABLE_STORAGE_PAGE_SIZE, GB_TABLE_ALLOC_METHOD);
        print_basic_info = _print_basic_info;
    }

    void set_function_pointers()
    {
        // THIS NEEDS TO BE RE-IMPLEMENTED
        switch (GTABLE_STORAGE_METHOD)
        {
            case GTABLE_HASHTABLE_STORAGE:
                //lookup_g_numerator = (G_NUM_TYPE)g_num_hashtable;
                break;
            case GTABLE_BINSEARCH_STORAGE:
                //lookup_g_numerator = (G_NUM_TYPE)g_num_binsearch;
                printf("GTABLE_BINSEARCH_STORAGE not implemented yet! Exiting\n");
                exit(1);
                break;
            case GTABLE_DENSE_STORAGE:
                //lookup_g_numerator = (G_NUM_TYPE)g_num_dense;
                printf("GTABLE_DENSE_STORAGE not implemented yet! Exiting\n");
                exit(1);
                break;
            default:
                printf("GTABLE/BTABLE METHOD NOT IMPLEMENTED YET! EXITING!\n");
                exit(1);
        }
    }

    void error_checks()
    {
        if(GAMMA_PERTURB_EPS <= 0 )
        {
            printf(RED "ERROR: GAMMA_PERTURB_EPS=%lf in cauchy_constants.hpp must be positive!" NC "\n", GAMMA_PERTURB_EPS);
            exit(1);
        }
        if( (shape_range-1) > 31)
        {
            printf(RED "ERROR: Until tested, max HP (%d) shape cannot exceed 31!" NC "\n", shape_range-1);
            exit(1);
        }
        // Make sure the gtables are setup correctly
        if(HASHTABLE_STORAGE)
        {
            if((GTABLE_SIZE_MULTIPLIER < 1))
            {
                printf(RED "ERROR: GTABLE_SIZE_MULTIPLIER in cauchy_types.hpp must be defined as >1 when using HASHTABLE_STORAGE method!" NC "\n");
                exit(1);
            }
        }
        else
        {
            if(GTABLE_SIZE_MULTIPLIER != 1)
            {
                printf(RED "ERROR: GTABLE_SIZE_MULTIPLIER in cauchy_types.hpp must be defined as 1 when using BINSEARCH_STORAGE or DENSE_STORAGE method!" NC "\n");
                exit(1);
            }
        }
        if(DENSE_STORAGE)
        {
            if(FULL_STORAGE)
                printf(YEL "WARNING! You are using DENSE_STORAGE WITH FULL STORAGE set on! This is VERY expensive! Consider using FULL_STORAGE=false method!" NC "\n");
        }
        // Make sure the largest gtable size will fit into a single page
        int max_cell_count = cell_count_central(shape_range-1, d);
        BYTE_COUNT_TYPE bytes_max_hp_gtable = (BYTE_COUNT_TYPE)(GTABLE_SIZE_MULTIPLIER * sizeof(GTABLE_TYPE) * max_cell_count / (1 + HALF_STORAGE) + 10); // a little extra padding for floating point calcs
        if( bytes_max_hp_gtable > GB_TABLE_STORAGE_PAGE_SIZE)
        {
            printf(RED "[Error Cauchy Estimator Initialization:]\nGB_TABLE_STORAGE_PAGE_SIZE is smaller than the memory required to store the largest HPA's gtable (which requires %llu bytes)\nIncrease GB_TABLE_STORAGE_PAGE_SIZE in cauchy_util.hpp!" NC"\n", bytes_max_hp_gtable );
            exit(1);
        }
        // Make sure that the first d+1 gtables will fit into a single page
        BYTE_COUNT_TYPE bytes_first_gtables = (BYTE_COUNT_TYPE)(GTABLE_SIZE_MULTIPLIER * sizeof(GTABLE_TYPE) * (d+1) * (1<<d) / (1 + HALF_STORAGE) + 10); // a little extra padding for floating point calcs
        if(bytes_first_gtables > GB_TABLE_STORAGE_PAGE_SIZE)
        {
            printf(RED "[Error Cauchy Estimator Initialization:]\nGB_TABLE_STORAGE_PAGE_SIZE is smaller than the memory required to store the first STATE_DIM+1 gtable (which requires %llu bytes)\nIncrease GB_TABLE_STORAGE_PAGE_SIZE in cauchy_util.hpp!" NC"\n", bytes_first_gtables );
            exit(1);
        }
    }

    void time_prop(double* Phi, double* Gamma, double* beta)
    {
        double tmp_Gamma[d*cmcc];
        double tmp_beta[cmcc];
        CPUTimer tmr;
        tmr.tic();
        // Transpose, normalize, and pre-coalign Gamma and beta
        int tmp_cmcc = precoalign_Gamma_beta(Gamma, beta, cmcc, d, tmp_Gamma, tmp_beta);
        memset(terms_per_shape, 0, shape_range * sizeof(int));
        for(int i = 0; i < Nt; i++)
        {
            terms[i].time_prop(Phi);
            int m_new = terms[i].tp_coalign(tmp_Gamma, tmp_beta, tmp_cmcc);
            terms_per_shape[m_new]++;
        }
        tmr.toc(false);
        if(print_basic_info)
            printf("TP/TPC Step %d/%d took %d ms\n", master_step+1, num_estimation_steps, tmr.cpu_time_used);
        // No need to update / build the Btables on the last step, if SKIP_LAST_STEP is true
        if(!skip_post_mu)
        {
            tmr.tic();
            // Now build the new Btable after adding and possibly coaligning Gamma
            BYTE_COUNT_TYPE bytes_max_cells = 0;
            for(int i = 0; i < shape_range; i++)
                if(terms_per_shape[i] > 0)
                    bytes_max_cells += (((BYTE_COUNT_TYPE)dce_helper.cell_counts_cen[i]) / (1 + HALF_STORAGE)) * terms_per_shape[i] * sizeof(BKEYS_TYPE);
            gb_tables.extend_bp_tables(bytes_max_cells);

            for(int i = 0; i < Nt; i++)
            { 
                int m = terms[i].m;
                int phc = terms[i].phc;
                BKEYS B_parent = terms[i].enc_B;
                
                //printf("-------\nTerm %d's HPA is:\n", i);
                //print_mat(terms[i].A, m, d);
                //printf("Term %d's parent B is:\n", i);
                //print_B_encoded(B_parent, terms[i].cells_gtable_p, phc, true);
                
                // If m == phc, the Gammas have coaligned with previous hps, simply just copy the B over to Bp
                if(m == phc)
                {
                    gb_tables.set_term_bp_table_pointer( &(terms[i].enc_B), terms[i].cells_gtable_p, true);
                    memcpy(terms[i].enc_B, B_parent, terms[i].cells_gtable_p * sizeof(BKEYS_TYPE));
                }
                else
                {
                    gb_tables.set_term_bp_table_pointer( &(terms[i].enc_B), dce_helper.cell_counts_cen[m] / (1 + HALF_STORAGE), false);
                    make_time_prop_btable(B_parent, terms + i, &dce_helper);
                    gb_tables.incr_chunked_bp_table_ptr(terms[i].cells_gtable);
                    
                    //printf("Term %d's B_TP is:\n", i);
                    //print_B_encoded(terms[i].enc_B, terms[i].cells_gtable, m, true);
                    //printf("-------\n");
                }
            }
            //printf("-------\n");
            // Now swap btable memory locations and clear btables_p;
            // btable_ps are now pointed to by btables and memory in btables_p is unallocated
            gb_tables.swap_btables();
            tmr.toc(false);
            if(print_basic_info)
                printf("TP Cell Enum Took %d ms\n", tmr.cpu_time_used);
        }

        if(print_basic_info)
            for(int i = 0; i < shape_range; i++)
                if(terms_per_shape[i] > 0)
                    printf("TP/TPC: Shape %d has %d terms\n", i, terms_per_shape[i]);
    }

    // true msmt update
    void msmt_update(double msmt, double* H, double gamma)
    {

        if(master_step == 0)
        {
            Nt = terms[0].msmt_update(terms+1, msmt, H, gamma, true, false) + 1;
            terms_per_shape[d] = Nt;
            if(print_basic_info)
                printf("MU step %d/%d: Shape %d now has %d terms\n", master_step+1, num_estimation_steps, d, terms_per_shape[d]);
            compute_moments(true);
        }
        else
        {
            CPUTimer tmr;
            tmr.tic();
            // Get new maximum term count
            int Nt_alloc = 0;
            for(int i = 0; i < shape_range; i++)
                if(terms_per_shape[i] > 0)
                    Nt_alloc += terms_per_shape[i]*(i+1);
            // Allocate memory for new child terms
            terms = (CauchyTerm*) realloc(terms,  Nt_alloc * sizeof(CauchyTerm) );
            null_ptr_check(terms);
            // Reset terms per shape counts before entering MU
            memset(terms_per_shape, 0, shape_range * sizeof(int) ); // reset term counts
            int Nt_new = Nt;
            for(int i = 0; i < Nt; i++)
            {
                CauchyTerm* child_terms = terms + Nt_new;
                int num_new_children = terms[i].msmt_update(child_terms, msmt, H, gamma, false, skip_post_mu);
                Nt_new += num_new_children;
                terms_per_shape[terms[i].m] += num_new_children + 1;
            }
            tmr.toc(false);

            if(print_basic_info)
            {
                printf("MU step %d/%d took %d ms:\n", master_step+1, num_estimation_steps, tmr.cpu_time_used);
                for(int i = 0; i < shape_range; i++)
                    if(terms_per_shape[i] > 0)
                        printf("MU: Shape %d now has %d terms\n", i, terms_per_shape[i]);
            }
            Nt = Nt_new;
            //terms = (CauchyTerm*) realloc(terms,  Nt * sizeof(CauchyTerm) );
            //null_ptr_check(terms);
            // Compute the moments after MU
            compute_moments(true);
            // Run the measurement update coalignment (Possibly skip if last step)
            memset(terms_per_shape, 0, shape_range * sizeof(int) ); // reset term counts
            if(!skip_post_mu)
            {
                tmr.tic();
                for(int i = 0; i < Nt; i++)
                {   
                    // No MUC for old terms 
                    if(terms[i].parent == NULL)
                    {
                        terms[i].normalize_hps(true);
                        terms_per_shape[terms[i].m]++;
                    }
                    else 
                    {
                        terms_per_shape[terms[i].mu_coalign()]++;
                    }
                }
                tmr.toc(false);
                if(print_basic_info)
                {
                    printf("MUC took %d ms:\n", tmr.cpu_time_used);
                    for(int i = 0; i < shape_range; i++)
                        if(terms_per_shape[i] > 0)
                            printf("MUC: Shape %d now has %d terms\n", i, terms_per_shape[i]);
                }
            }
        }
    }

    // dumb msmt update for debugging against old code -- has flaws currently.
    void _msmt_update(double msmt, double* H, double gamma)
    {

        if(master_step == 0)
        {
            CauchyTerm old_term = terms[0];
            Nt = old_term.msmt_update(terms, msmt, H, gamma, true, false) + 1;
            terms[d] = old_term;
            terms_per_shape[d] = Nt;
            if(print_basic_info)
                printf("MU step %d/%d: Shape %d now has %d terms\n", master_step+1, num_estimation_steps, d, terms_per_shape[d]);
            compute_moments(true);
        }
        else
        {
            // Get new maximum term count
            int Nt_alloc = 0;
            for(int i = 0; i < shape_range; i++)
                if(terms_per_shape[i] > 0)
                    Nt_alloc += terms_per_shape[i]*(i+1);
            // Allocate memory for new child terms
            CauchyTerm* terms_mu = (CauchyTerm*) malloc(Nt_alloc * sizeof(CauchyTerm) );
            null_ptr_check(terms_mu);
            terms = (CauchyTerm*) realloc(terms,  Nt_alloc * sizeof(CauchyTerm) );
            null_ptr_check(terms);
            int Nt_new = 0;
            memset(terms_per_shape, 0, shape_range * sizeof(int) ); // reset term counts
            for(int i = 0; i < Nt; i++)
            {
                CauchyTerm* child_terms = terms_mu + Nt_new;
                CauchyTerm old_term = terms[i];
                int num_new_children = old_term.msmt_update(child_terms, msmt, H, gamma, false, skip_post_mu);
                terms_mu[Nt_new + num_new_children] = old_term;
                for(int j = 0; j < num_new_children; j++)
                    terms_mu[Nt_new + j].parent = &(terms_mu[Nt_new + num_new_children]);
                Nt_new += num_new_children + 1;
                terms_per_shape[old_term.m] += num_new_children + 1;
            }
            if(print_basic_info)
                for(int i = 0; i < shape_range; i++)
                    if(terms_per_shape[i] > 0)
                        printf("MU step %d/%d: Shape %d now has %d terms\n", master_step+1, num_estimation_steps, i, terms_per_shape[i]);
            memset(terms_per_shape, 0, shape_range * sizeof(int) ); // reset term counts
            Nt = Nt_new;
            ptr_swap<CauchyTerm>(&terms, &terms_mu);
            // Compute the moments after MU
            compute_moments(true);

            // Run the measurement update coalignment (Possibly skip if last step)
            if(!skip_post_mu)
            {
                for(int i = 0; i < Nt; i++)
                {   
                    // No MUC for old terms 
                    if(terms[i].parent == NULL)
                    {
                        terms[i].normalize_hps(true);
                        terms_per_shape[terms[i].m]++;
                    }
                    else 
                    {
                        terms_per_shape[terms[i].mu_coalign()]++;
                    }
                }
                if(print_basic_info)
                    for(int i = 0; i < shape_range; i++)
                        if(terms_per_shape[i] > 0)
                            printf("MUC step %d/%d: Shape %d now has %d terms\n", master_step+1, num_estimation_steps, i, terms_per_shape[i]);
            }
            free(terms_mu);
        }
    }

    void compute_moments(const bool before_ftr = true)
    {
        CPUTimer tmr;
        tmr.tic();
        bool first_step = (master_step == 0);
        C_COMPLEX_TYPE g_val;
        C_COMPLEX_TYPE yei[d];
        fz = 0;
        memset(conditional_mean, 0, d*sizeof(C_COMPLEX_TYPE));
        memset(conditional_variance, 0, d*d*sizeof(C_COMPLEX_TYPE));
        for(int i = 0; i < Nt; i++)
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
    }

    void first_create_gtables()
    {
        for(int i = 0; i < Nt; i++)
        {
            terms[i].cells_gtable = dce_helper.cell_counts_cen[d] / (1 + HALF_STORAGE);
            gb_tables.set_term_btable_pointer( &(terms[i].enc_B), terms[i].cells_gtable, true);
            gb_tables.set_term_gtable_pointer(&(terms[i].gtable), terms[i].cells_gtable, true);
            make_gtable_first(terms + i, G_SCALE_FACTOR);
            terms[i].become_parent();
        }
        gb_tables.swap_gtables();
        if(print_basic_info)
            compute_moments(false);
    }

    void fast_term_reduction_and_create_gtables()
    {
        if(skip_post_mu)
            return;
        // bs[i] are the bs list (array) of all terms with i hyperplanes
        // shape_idxs[i][j] is the index of the "bs[i] + d*j" vector in the term list
        CPUTimer tmr;
        tmr.tic();
        double** bs;
        int** shape_idxs;
        get_contiguous_bs_from_term_list(&bs, &shape_idxs, terms, Nt, terms_per_shape, shape_range, d);

        int max_Nt_shape = array_max<int>(terms_per_shape, shape_range);
        if(max_Nt_shape > ftr_helper.max_num_terms)
            ftr_helper.realloc_helpers(max_Nt_shape);
        tmr.toc(false);
        if(print_basic_info)
            printf("[FTR/Gtables step %d/%d:] Preprocessing took %d ms\n", master_step+1, num_estimation_steps, tmr.cpu_time_used);
        int Nt_reduced = 0; // Total number of terms after term reduction has finished
        int Nt_removed = 0; // Total number of terms removed after term approximation
        bool* F_removed = (bool*) malloc(Nt * sizeof(bool)); // boolean flag of terms that are removed after FTR
        null_ptr_check(F_removed);
        memset(F_removed, 1, Nt * sizeof(bool));
        for(int m = 0; m < shape_range; m++)
        {
            if(terms_per_shape[m] > 0)
            {
                tmr.tic();
                int Nt_shape = terms_per_shape[m];
                memcpy(ftr_helper.F_TR, ftr_helper.F, Nt_shape * sizeof(int) );
                int* shape_idx = shape_idxs[m]; 

                build_ordered_point_maps(
                    bs[m],
                    ftr_helper.ordered_points, 
                    ftr_helper.forward_map, 
                    ftr_helper.backward_map, 
                    Nt_shape,
                    d, false);
      
                fast_term_reduction(
                    bs[m], 
                    terms,
                    shape_idxs[m],
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
                int max_cells_shape = dce_helper.cell_counts_cen[m] / (1 + HALF_STORAGE);
                int dce_temp_hashtable_size = max_cells_shape * dce_helper.storage_multiplier;
                gb_tables.extend_gtables(max_cells_shape, max_Nt_reduced_shape);
                gb_tables.extend_btables(max_cells_shape, max_Nt_reduced_shape);

                // Now make B-Tables, G-Tables, for each reduction group
                int** forward_F = ffa.Fs;
                int* forward_F_counts = ffa.F_counts;
                int* backward_F = ftr_helper.F_TR;             
                for(int j = 0; j < Nt_shape; j++)
                {
                    // Check whether we need to process term j (if it has reductions or is unique)
                    if(backward_F[j] == j)
                    {
                        int rt_idx = shape_idx[j];
                        CauchyTerm* child_j = terms + rt_idx;
                        // Make the Btable if not an old term
                        if(child_j->parent != NULL)
                        {
                            // Set the child btable memory position
                            gb_tables.set_term_btable_pointer(&(child_j->enc_B), max_cells_shape, false);
                            // Make the Btable if not an old term
                            make_new_child_btable(child_j, 
                                dce_helper.B_mu_hash, child_j->parent->cells_gtable * dce_helper.storage_multiplier,
                                dce_helper.B_coal_hash, dce_temp_hashtable_size,
                                dce_helper.B_uncoal, dce_helper.F);
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
                            if(child_j->parent != NULL)
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
                                int cp_idx = shape_idx[forward_F[j][k++]];
                                CauchyTerm* child_k = terms + cp_idx;
                                // The btable of all terms in this reduction group are similar
                                // The only difference is the orientation of their hyperplanes
                                // Update the Btable of the last potential root for child_k
                                // Use the memory space currently pointed to by child_j only if child_k is not a parent 
                                if(child_k->parent != NULL)
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
                                // If child term k is not approximated out, it becomes root
                                if( !make_gtable(child_k, G_SCALE_FACTOR) )
                                {
                                    rt_idx = cp_idx;
                                    if(child_k->parent != NULL)
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
                            int cp_idx = shape_idx[forward_F[j][k++]];
                            CauchyTerm* child_k = terms + cp_idx;
                            // Set memory position of child gtable k here
                            // Make the Btable if not an old term
                            if(child_k->parent != NULL)
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
                                    printf(RED"[BIG WARN FTR/Make Gtables:] child_k->cells_gtable != child_j->cells_gtable. We have code to fix this! But EXITING now until this is commented out!" NC "\n");
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
                            F_removed[rt_idx] = false;
                            Nt_reduced_shape++;
                            child_j->become_parent();
                        }
                        else
                            Nt_removed_shape++;
                    }
                }
                // After term reduction and g-evaluation 
                Nt_reduced += Nt_reduced_shape;
                Nt_removed += Nt_removed_shape;
                terms_per_shape[m] = Nt_reduced_shape;
                tmr.toc(false);
                if(print_basic_info)
                    printf("[Eval Gtables shape %d:] Took %d ms\n", m, tmr.cpu_time_used);
    
                if(print_basic_info && WITH_TERM_APPROXIMATION)
                        printf("[Term Approx Shape %d:] %d/%d (%d were removed)\n", m, Nt_reduced_shape, max_Nt_reduced_shape, Nt_removed_shape);
            }
        }
        // For all terms not reduced out or approximated out, keep these terms 
        tmr.tic();
        CauchyTerm* terms_after_reduction = (CauchyTerm*) malloc(Nt_reduced * sizeof(CauchyTerm));
        int count = 0;
        for(int i = 0; i < Nt; i++)
        {
            if(F_removed[i])
                terms[i].deinit();
            else
                terms_after_reduction[count++] = terms[i]; 
        }
        Nt = Nt_reduced;
        ptr_swap<CauchyTerm>(&terms, &terms_after_reduction); // terms now points to coalesced term count
        free(terms_after_reduction);
        gb_tables.swap_gtables();

        // Deallocate this round of FTR helpers
        for(int i = 0; i < shape_range; i++)
        {
            free(bs[i]);
            free(shape_idxs[i]);
        }        
        free(bs);
        free(shape_idxs);
        free(F_removed);
        tmr.toc(false);
        // Compute moments after FTR
        if(print_basic_info)
        {
            printf("Deallocating memory after FTR took %d ms\n", tmr.cpu_time_used);
            compute_moments(false);
        }
    }

    void step(double msmt, double* Phi, double* Gamma, double* beta, double* H, double gamma)
    {
        CPUTimer tmr;
        tmr.tic();
        skip_post_mu = SKIP_LAST_STEP && (master_step == (num_estimation_steps-1)); 
        if(master_step == 0)
        {
            msmt_update(msmt, H, gamma);
            first_create_gtables();
        }
        else
        {
            if( (master_step % p) == 0 )
                time_prop(Phi, Gamma, beta);
            msmt_update(msmt, H, gamma);
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
        // Deallocate terms 
        for(int i = 0; i < Nt; i++)
            terms[i].deinit();
        free(terms);

        terms = (CauchyTerm*) malloc( (d+1)*sizeof(CauchyTerm) );
        null_ptr_check(terms);
        memset(terms_per_shape, 0, shape_range * sizeof(int));
        Nt = 1;
        master_step = 0;

        // Re-init first term
        terms->init(d, d);
        memcpy(terms->A, A0_init, d*d*sizeof(double));
        memcpy(terms->p, p0_init, d*sizeof(double));
        memcpy(terms->b, b0_init, d*sizeof(double));

        //Re-init helpers
        ftr_helper.init(d, 1<<d);
        dce_helper.init(shape_range-1, d, DCE_STORAGE_MULT, cmcc);
        BYTE_COUNT_TYPE gb_init_bytes = (BYTE_COUNT_TYPE)((d+1) * GTABLE_SIZE_MULTIPLIER * (1<<d) * sizeof(GTABLE_TYPE));
        int start_pages = (gb_init_bytes + GB_TABLE_STORAGE_PAGE_SIZE - 1) / GB_TABLE_STORAGE_PAGE_SIZE;
        gb_tables.init(start_pages, GB_TABLE_STORAGE_PAGE_SIZE, GB_TABLE_ALLOC_METHOD);
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
        for(int i = 0; i < Nt; i++)
            terms[i].deinit();
        free(terms);
    }

};

#endif //_CAUCHY_ESTIMATOR_HPP_