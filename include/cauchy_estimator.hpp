#ifndef _CAUCHY_ESTIMATOR_HPP_
#define _CAUCHY_ESTIMATOR_HPP_


#include "cauchy_term.hpp"
#include "cauchy_util.hpp"
#include "eval_gs.hpp"
#include "random_variables.hpp"
#include "term_reduction.hpp"
#include "flattening.hpp"

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

    CauchyEstimator(double* A0, double* p0, double* b0, int _steps, int _d, int _cmcc, int _p)
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
        memcpy(terms->A, A0, d*d*sizeof(double));
        memcpy(terms->p, p0, d*sizeof(double));
        memcpy(terms->b, b0, d*sizeof(double));

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
        BYTE_COUNT_TYPE bytes_max_hp_gtable = (BYTE_COUNT_TYPE)(GTABLE_SIZE_MULTIPLIER * sizeof(GTABLE_TYPE) * max_cell_count);
        if( bytes_max_hp_gtable > GB_TABLE_STORAGE_PAGE_SIZE)
            printf(RED "[Error Cauchy Estimator Initialization:]\nGB_TABLE_STORAGE_PAGE_SIZE is smaller than the memory required to store the largest HPA's gtable (which requires %llu bytes)\nIncrease GB_TABLE_STORAGE_PAGE_SIZE in cauchy_util.hpp!" NC"\n", bytes_max_hp_gtable );
    }

    void time_prop(double* Phi, double* Gamma, double* beta)
    {
        double tmp_Gamma[d*cmcc];
        double tmp_beta[cmcc];
        
        // Transpose, normalize, and pre-coalign Gamma and beta
        int tmp_cmcc = precoalign_Gamma_beta(Gamma, beta, cmcc, d, tmp_Gamma, tmp_beta);
        const bool skip_post_mu = SKIP_LAST_STEP && (master_step == (num_estimation_steps-1)); 
        memset(terms_per_shape, 0, shape_range * sizeof(int));
        for(int i = 0; i < Nt; i++)
        {
            terms[i].time_prop(Phi);
            int m_new = terms[i].tp_coalign(tmp_Gamma, tmp_beta, tmp_cmcc);
            terms_per_shape[m_new]++;
        }
        // No need to update / build the Btables on the last step, if SKIP_LAST_STEP is true
        if(!skip_post_mu)
        {
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
                // If m == phc, the Gammas have coaligned with previous hps, simply just copy the B over to Bp
                if(m == phc)
                {
                    gb_tables.set_term_bp_table_pointer( &(terms[i].enc_B), terms[i].cells_gtable, true);
                    memcpy(terms[i].enc_B, B_parent, terms[i].cells_gtable * sizeof(BKEYS_TYPE));
                }
                else
                {
                    gb_tables.set_term_bp_table_pointer( &(terms[i].enc_B), dce_helper.cell_counts_cen[m] / (1 + HALF_STORAGE), false);
                    make_time_prop_btable(B_parent, terms + i, &dce_helper);
                    gb_tables.incr_chunked_bp_table_ptr(terms[i].cells_gtable);
                }
            }
            // Now swap btable memory locations and clear btables_p;
            // btable_ps are now pointed to by btables and memory in btables_p is unallocated
            gb_tables.swap_btables();
        }
    }

    // true msmt update
    void _msmt_update(double msmt, double* H, double gamma)
    {

        if(master_step == 0)
        {
            Nt = terms[0].msmt_update(terms+1, msmt, H, gamma, true, false);
            terms_per_shape[d] = Nt+1;
        }
        else
        {
            // Do not run coalignment or Btable formation if its the last step and directed unnecessary
            const bool skip_post_mu = SKIP_LAST_STEP && (master_step == (num_estimation_steps-1)); 
            // Get new hypothesized term count
            int Nt_alloc = 0;
            for(int i = 0; i < shape_range; i++)
            {
                if(terms_per_shape[i] > 0)
                {
                    int terms_in_shape = (i+1) * terms_per_shape[i];
                    Nt_alloc += terms_in_shape;
                }
            }
            memset(terms_per_shape, 0, shape_range * sizeof(int) ); // reset term count
            // Allocate memory for new child terms
            terms = (CauchyTerm*) realloc(terms,  Nt_alloc * sizeof(CauchyTerm) );
            null_ptr_check(terms);
            int Nt_new = Nt;
            for(int i = 0; i < Nt; i++)
            {
                CauchyTerm* child_terms = terms + Nt_new;
                int num_new_children = terms[i].msmt_update(child_terms, msmt, H, gamma, false, skip_post_mu);
                terms_per_shape[terms[i].m]++;
                if(skip_post_mu)
                {
                    for(int j = 0; j < num_new_children; j++)
                        terms_per_shape[child_terms[j].m]++;
                }
                else
                {   
                    // Oldest term does not coalign. Just normalize HP
                    terms[i].normalize_hps(true);
                    // Coalign new children
                    for(int j = 0; j < num_new_children; j++)
                        terms_per_shape[child_terms[j].mu_coalign()]++;
                }
                Nt_new += num_new_children;
            }
            Nt = Nt_new;
            terms = (CauchyTerm*) realloc(terms,  Nt * sizeof(CauchyTerm) );
            null_ptr_check(terms);
        }
    }

    // dumb msmt update for debugging against old code
    void msmt_update(double msmt, double* H, double gamma)
    {
        // For debugging purposes only, preserve the order of terms w.r.t old code
        if(master_step == 0)
        {
            CauchyTerm old_term = terms[0];
            Nt = old_term.msmt_update(terms, msmt, H, gamma, true, false);
            terms[d] = old_term;
            terms_per_shape[d] = Nt+1;
        }
        else
        {
            // Do not run coalignment or Btable formation if its the last step and directed unnecessary
            const bool skip_post_mu = SKIP_LAST_STEP && (master_step == (num_estimation_steps-1)); 
            // Get new hypothesized term count
            int Nt_alloc = 0;
            for(int i = 0; i < shape_range; i++)
            {
                if(terms_per_shape[i] > 0)
                {
                    int terms_in_shape = (i+1) * terms_per_shape[i];
                    Nt_alloc += terms_in_shape;
                }
            }
            memset(terms_per_shape, 0, shape_range * sizeof(int) ); // reset term count
            // Allocate memory for new child terms
            CauchyTerm* terms_mu = (CauchyTerm*) malloc(Nt_alloc * sizeof(CauchyTerm) );
            
            null_ptr_check(terms_mu);
            int Nt_new = 0;
            for(int i = 0; i < Nt; i++)
            {
                CauchyTerm* child_terms = terms_mu + Nt_new;
                CauchyTerm old_term = terms[i];
                int num_new_children = terms[i].msmt_update(child_terms, msmt, H, gamma, false, skip_post_mu);
                terms_mu[Nt_new + num_new_children] = old_term;
                terms_per_shape[old_term.m]++;
                if(skip_post_mu)
                {
                    for(int j = 0; j < num_new_children; j++)
                        terms_per_shape[child_terms[j].m]++;
                }
                else
                {   
                    // Oldest term does not coalign. Just normalize HP
                    terms_mu[Nt_new + num_new_children].normalize_hps(true);
                    // Coalign new children
                    for(int j = 0; j < num_new_children; j++)
                        terms_per_shape[child_terms[j].mu_coalign()]++;
                }
                Nt_new += (num_new_children+1);
            }
            Nt = Nt_new;
            terms = (CauchyTerm*) realloc(terms,  Nt * sizeof(CauchyTerm) );
            null_ptr_check(terms);
            // Suppppper dumb but it should work to keep terms in order with old code
            int count = 0;
            for(int j = 0; j < shape_range; j++)
            {
                if(terms_per_shape[j] > 0)
                {
                    for(int i = 0; i < Nt; i++)
                        if(terms_mu[i].m == j)
                            terms[count++] = terms_mu[i];
                }
            }
            assert(count == Nt);
            free(terms_mu);

        }
    }

    void compute_moments(const bool before_ftr = true)
    {
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
        G_SCALE_FACTOR = 1.0 / creal(fz);

        C_COMPLEX_TYPE Ifz = CMPLX(cimag(fz), creal(fz)); // imaginary fz
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

    void first_create_gtables()
    {
        for(int i = 0; i < Nt; i++)
        {
            terms[i].cells_gtable = dce_helper.cell_counts_cen[d];
            gb_tables.set_term_btable_pointer( &(terms[i].enc_B), terms[i].cells_gtable, true);
            make_gtable_first(terms + i, G_SCALE_FACTOR);
            terms[i].become_parent();
        }
    }

    void fast_term_reduction_and_create_gtables()
    {
        const bool optional_TA_print = true;
        const bool skip_post_mu = SKIP_LAST_STEP && (master_step == (num_estimation_steps-1)); 
        if(skip_post_mu)
            return;
        // bs[i] are the bs list (array) of all terms with i hyperplanes
        // shape_idxs[i][j] is the index of the "bs[i] + d*j" vector in the term list
        double** bs;
        int** shape_idxs;
        get_contiguous_bs_from_term_list(&bs, &shape_idxs, terms, Nt, terms_per_shape, shape_range, d);

        int max_Nt_shape = array_max<int>(terms_per_shape, shape_range);
        if(max_Nt_shape > ftr_helper.max_num_terms)
            ftr_helper.realloc_helpers(max_Nt_shape);

        int Nt_reduced = 0; // Total number of terms after term reduction has finished
        int Nt_removed = 0; // Total number of terms removed after term approximation
        bool* F_removed = (bool*) malloc(Nt * sizeof(bool)); // boolean flag of terms that are removed after FTR
        null_ptr_check(F_removed);
        memset(F_removed, 1, Nt * sizeof(bool));
        for(int m = 0; m < shape_range; m++)
        {
            int Nt_shape = terms_per_shape[m];
            if(Nt_shape > 0)
            {
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
                ForwardFlagArray ffa(ftr_helper.F_TR, Nt);
                int max_Nt_reduced_shape = ffa.num_terms_after_reduction; // Max number of terms after reduction (before term approximation)
                int Nt_reduced_shape = 0;
                int Nt_removed_shape = 0;
                int max_cells_shape = dce_helper.cell_counts_cen[m] / (1 + HALF_STORAGE);
                int dce_temp_hashtable_sizes = max_cells_shape * dce_helper.storage_multiplier;
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
                                dce_helper.B_mu_hash, dce_temp_hashtable_sizes,
                                dce_helper.B_coal_hash, dce_temp_hashtable_sizes,
                                dce_helper.B_uncoal, dce_helper.F);
                        }
                        int num_cells_of_red_group = child_j->cells_gtable;
                        BKEYS btable_for_red_group = child_j->enc_B;
                        // set memory position of the child gtable
                        // Make the g-table of the root term
                        gb_tables.set_term_gtable_pointer(&(child_j->gtable), child_j->cells_gtable, false);
                        // If the term is negligable (is approximated out), we need to search for a new "root"
                        if( make_gtable(child_j, G_SCALE_FACTOR) )
                            rt_idx = -1;
                        else
                        {
                            if(child_j->parent != NULL)
                                gb_tables.incr_chunked_btable_ptr(child_j->cells_gtable);
                        }
                        
                        int num_term_combos = forward_F_counts[j];
                        int k = 0;
                        // If the root term has been approximated out, we need to search through its term combinations to find a new term to take the place as root
                        if(rt_idx == -1)
                        {
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
                                child_k->cells_gtable = num_cells_of_red_group;
                                gb_tables.set_term_btable_pointer(&(child_k->enc_B), child_k->cells_gtable, false);
                                update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, num_cells_of_red_group, m, d);
                            }
                            else
                            {
                                // To deal with the case where numerical instability causes cell counts to be different, 
                                // If the cell counts are different (due to instability), update child_k's btable to be compatible with the root
                                if(child_k->cells_gtable != child_j->cells_gtable)
                                {
                                    // If child_k has more than child_j's cells,
                                    // Downgrade child_k to be equal to child_j
                                    if(child_k->cells_gtable > child_j->cells_gtable)
                                    {
                                        child_k->cells_gtable = child_j->cells_gtable;
                                        update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, num_cells_of_red_group, m, d);
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
                                        update_btable(child_j->A, child_j->enc_B, child_k->A, child_k->enc_B, num_cells_of_red_group, m, d);
                                    }
                                }
                            }
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
                
                if(WITH_TERM_APPROXIMATION && optional_TA_print)
                    printf("[Flattening Shape %d]: There were %d/%d terms seen to be under eps -- these were removed!\n", m, Nt_removed_shape, max_Nt_reduced_shape);
                // reduced_terms = (CauchyTerm*) realloc(reduced_terms, final_reduction_count * sizeof(CauchyTerm));
                // ptr_swap<CauchyTerm>(&terms, &reduced_terms);
                // deallocate 
            }
        }
        // For all terms not reduced out or approximated out, keep these terms 
        CauchyTerm* terms_after_reduction = (CauchyTerm*) malloc(Nt_reduced * sizeof(CauchyTerm));
        int count = 0;
        for(int i = 0; i < Nt; i++)
        {
            if(F_removed[i])
                terms[i].deinit();
            else
                terms_after_reduction[count++] = terms[i]; 
        }
        if(WITH_TERM_APPROXIMATION && optional_TA_print)
            printf("[Flattening Summary]: There were %d/%d terms total under term approx. eps -- these were removed!\n", Nt_removed, Nt);

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
    }

    void step(double msmt, double* Phi, double* Gamma, double* beta, double* H, double gamma)
    {
        
        if(master_step == 0)
        {
            msmt_update(msmt, H, gamma);
            compute_moments(true);
            first_create_gtables();
            compute_moments(false);
        }
        else
        {
            if( (master_step % p) == 0 )
                time_prop(Phi, Gamma, beta);
            msmt_update(msmt, H, gamma);
            compute_moments(true);
            fast_term_reduction_and_create_gtables();
            compute_moments(false);
        }
        master_step++;
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
    }

};

#endif //_CAUCHY_ESTIMATOR_HPP_