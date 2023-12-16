#ifndef _CPDF_NDIM_HPP_
#define _CPDF_NDIM_HPP_

#include "cauchy_constants.hpp"
#include "cauchy_estimator.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cpu_linalg.hpp"
#include "cpu_timer.hpp"
#include "random_variables.hpp"

struct PointWiseNDimCauchyCPDF
{
    ChildTermWorkSpace lowered_children_workspace; // Used to store the terms temporarily during TP/TPC/MU/MUC
    CoalignmentElemStorage coalign_store; // Memory manager for storing terms after Coalignment
    ReductionElemStorage reduce_store; // Memory manager for storing terms after Term Reduction
    ChunkedPackedTableStorage gb_tables; // Memory manager for storing g and b-tables
    CauchyEstimator* cauchyEst;
    double ZERO_EPSILON;


    PointWiseNDimCauchyCPDF(CauchyEstimator* _cauchyEst)
    {
        ZERO_EPSILON = MU_EPS;
        cauchyEst = _cauchyEst;
        lowered_children_workspace.init(cauchyEst->shape_range-1, cauchyEst->d);
        coalign_store.init(1, CP_STORAGE_PAGE_SIZE);
        gb_tables.init(1, CP_STORAGE_PAGE_SIZE);
        reduce_store.init(1, CP_STORAGE_PAGE_SIZE);
    }

    void construct_lowered_parent_elements(
        CauchyTerm* parent, 
        double* A_lp,
        double* A_ldim,
        bool* F_A_ldim_not_zero,
        double* p_lp,
        double* b_lp,
        double* tild_b_lp,
        int* enc_lambda_bar,
        double* xk,
        int m, int ldim)
    {
        int ldim_m1 = ldim - 1;
        int _enc_lambda_bar = 0;
        for(int i = 0; i < m; i++)
        {
            A_ldim[i] = -parent->A[i*ldim + ldim - 1]; // -\tilde{a}^{k|k}_il (equation 8)
            // If A_ldim[i] is not zero, the hyperplane enters the integration formula (F_A_ldim_not_zero[i] = 1)
            // If A_ldim[i] is effectively zero, the hyperplane does not enter the integration formula (F_A_ldim_not_zero[i] = 0)
            F_A_ldim_not_zero[i] = (fabs(A_ldim[i]) > ZERO_EPSILON) ? 1 : 0;
        }
        for(int i = 0; i < m; i++)
        {
            if(F_A_ldim_not_zero[i])
            {
                for(int j = 0; j < ldim_m1; j++)
                    A_lp[i*ldim_m1 + j] = parent->A[i*ldim + j] / A_ldim[i];
                if(A_ldim[i] < 0)
                    _enc_lambda_bar |= (1 << i);
                p_lp[i] = parent->p[i] * fabs(A_ldim[i]);
            }   
            else 
            {
                memcpy(A_lp + i*ldim_m1, parent->A + i*ldim, ldim_m1 * sizeof(double));
                p_lp[i] = parent->p[i];
                //_enc_lambda_bar |= (0<<i) // does not change...since the sign element is 1 (for no flip)
            }
        }
        *enc_lambda_bar = _enc_lambda_bar;
        if(xk != NULL)
        {
            //assert(ldim == cauchyEst->d);
            for(int i = 0; i < ldim_m1; i++)
                b_lp[i] = parent->b[i] - xk[i];
            *tild_b_lp = parent->b[ldim_m1] - xk[ldim_m1];
        }
        else 
        {
            memcpy(b_lp, parent->b, ldim_m1 * sizeof(double));
            *tild_b_lp = parent->b[ldim_m1];
        }
    }

    void construct_lowered_marginal_parent_elements(
        CauchyTerm* parent, 
        double* A_lp,
        double* A_ldim,
        bool* F_A_ldim_not_zero,
        double* p_lp,
        double* b_lp,
        double* tild_b_lp,
        int* enc_lambda_bar,
        double* marg_xk, 
        int* marg_reorient_idxs,
        int num_marg_states,
        int m, int d)
    {
        double reorient_parent_A[m*d];
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < d; j++)
            {
                int j_reindex = marg_reorient_idxs[j];
                reorient_parent_A[i*d + j_reindex] = parent->A[i*d + j];
            }
        }
        int ldim = d;
        int ldim_m1 = d - 1;
        int _enc_lambda_bar = 0;
        for(int i = 0; i < m; i++)
        {
            A_ldim[i] = -reorient_parent_A[i*ldim + ldim - 1]; // -\tilde{a}^{k|k}_il (equation 8)
            // If A_ldim[i] is not zero, the hyperplane enters the integration formula (F_A_ldim_not_zero[i] = 1)
            // If A_ldim[i] is effectively zero, the hyperplane does not enter the integration formula (F_A_ldim_not_zero[i] = 0)
            F_A_ldim_not_zero[i] = (fabs(A_ldim[i]) > ZERO_EPSILON) ? 1 : 0;
        }
        for(int i = 0; i < m; i++)
        {
            if(F_A_ldim_not_zero[i])
            {
                for(int j = 0; j < ldim_m1; j++)
                    A_lp[i*ldim_m1 + j] = reorient_parent_A[i*ldim + j] / A_ldim[i];
                if(A_ldim[i] < 0)
                    _enc_lambda_bar |= (1 << i);
                p_lp[i] = parent->p[i] * fabs(A_ldim[i]);
            }   
            else 
            {
                memcpy(A_lp + i*ldim_m1, reorient_parent_A + i*ldim, ldim_m1 * sizeof(double));
                p_lp[i] = parent->p[i];
                //_enc_lambda_bar |= (0<<i) // does not change...since the sign element is 1 (for no flip)
            }
        }
        *enc_lambda_bar = _enc_lambda_bar;
        // form b[marg_reorient_idxs] - [0;marg_xk]
        int num_nu_zeros = d - num_marg_states;
        double tmp_b[ldim];
        memcpy(tmp_b, parent->b, ldim * sizeof(double));
        for(int i = 0; i < num_marg_states; i++)
        {
            int nu_idx_wrt_xk_i = marg_reorient_idxs[num_nu_zeros+i]; // index corresponding to xk[i] 
            tmp_b[nu_idx_wrt_xk_i] -= marg_xk[i];
        }
        for(int i = 0; i < ldim_m1; i++)
            b_lp[i] = tmp_b[marg_reorient_idxs[i]];
        *tild_b_lp = tmp_b[marg_reorient_idxs[ldim_m1]];
    }


    // generates lowered children from parent term -> parent -> lowered parent -> lowered chilren
    int construct_lowered_children(CauchyTerm* parent, CauchyTerm* lowered_children, double* xk = NULL, int* marg_reorient_idxs = NULL, int num_marg_states = -1)
    {
        // Lowered Parent Term Formation
        // Shape of the parent is m x ldim
        // Shape of the lowered parent is m x ldim_m1
        // Shape of the lowered child terms is m_m1 x ldim_m1 (before coalignement)
        int m = parent->m;
        int ldim = parent->d;
        int m_m1 = m-1;
        int ldim_m1 = ldim - 1;
        // Lowered Parent Term is constructed as follows
        // A_lp = A[:,:-1] / -A[:,-1] (equation 8) -> size m x ldim_m1
        // A_ldim = -A[:,-1] (equation 8)
        // F_A_ldim_not_zero -> holds indices of which A_ldim elements != 0
        // p_lp = p * |A[:,-1]| (equation 9) -> size m
        // b_lp = b[:-1] - x[:-1] (equation 6)
        // tild_b_lp = b[-1] - x[-1] (equation 6)
        // (Note that only for ldim == d do we need to subtract x from the bs)
        // lambda_bar = sgn(-A[:,-1]) (equation 18)
        double A_lp[m * ldim_m1];
        double A_ldim[m];
        bool F_A_ldim_not_zero[m]; // This is much like the HA orthogonality flag in the cauchy estimator
        double p_lp[m];
        double b_lp[ldim_m1];
        double tild_b_lp;
        int enc_lambda_bar; 

        // If we are integrating over all states, ordering does not matter
        if(marg_reorient_idxs == NULL)
        {
            construct_lowered_parent_elements(
                parent, A_lp, A_ldim, 
                F_A_ldim_not_zero, p_lp, b_lp, 
                &tild_b_lp, &enc_lambda_bar, 
                xk, m, ldim);
        }
        // If we are integrating to build a marginal cpdf, reorder the HPA and b-elements appropriately
        // Ordering is [\nu_1,...,\nu_p, \nu_p+1,...,\nu_n], where
        // [1...p] are states you dont care about
        // [p+1,n] are the states you want to retrieve a marginal pdf for
        // integration is carried out n-p times over only those states
        else 
        {
            assert(xk != NULL);
            assert(cauchyEst->d == ldim);
            assert(num_marg_states > 0);
            construct_lowered_marginal_parent_elements(
                parent, A_lp, A_ldim, 
                F_A_ldim_not_zero, p_lp, b_lp, 
                &tild_b_lp, &enc_lambda_bar, 
                xk, marg_reorient_idxs, num_marg_states, m, ldim);
        }

        // Lowered Child Term Formation 
        // create m child terms each of m-1 x ldim-1 dimension
        // if |\tilde{a}^{k|k}_il| for index l, do not create that child
        int child_term_count = 0;
        for(int t = 0; t < m; t++)
        {
            if(F_A_ldim_not_zero[t])
            {
                CauchyTerm* child = lowered_children + child_term_count;
                child->init_mem(&lowered_children_workspace, child_term_count, m_m1, ldim_m1);
                // Equations 14 and 16
                double* At = child->A;
                double* pt = child->p;
                double* bt = child->b;
                double* mu_it = A_lp + t*ldim_m1;
                double p_it = p_lp[t];
                int _l = 0;
                int enc_unintegrable_Atls = 0;
                for(int l = 0; l < m; l++)
                {
                    if(l != t)
                    {
                        double* mu_il = A_lp + l*ldim_m1;
                        double* a_tl = At + _l * ldim_m1;
                        // a_t(_l) = \mu_il - \mu_it, for all l != t
                        if(F_A_ldim_not_zero[l])
                            sub_vecs(mu_il, mu_it, a_tl, ldim_m1);
                        else 
                        {
                            memcpy(a_tl, mu_il, ldim_m1 * sizeof(double));
                            // Additionally, we need to mark this hyperplane as not entering integration formula
                            // Marking the hyperplane is crucial for g-evaluation and therefore matters in coalignment too
                            enc_unintegrable_Atls |= (1 << _l);
                        }
                        pt[_l] = p_lp[l];
                        _l += 1;
                    }
                }
                add_vecs(b_lp, mu_it, bt, ldim_m1, tild_b_lp); // Equation 15
                // Set denominator values for the child g
                child->c_val = tild_b_lp; // Equation 23
                child->d_val = p_it; // Equation 24
                child->z = t; // storing that this is the t-th child
                child->Horthog_flag = enc_unintegrable_Atls; // overriding this variable, has exactly same use case
                child->is_new_child = true;
                // Set lower child gtable's parent g/b-table pointers and variables
                child->enc_lhp = enc_lambda_bar; // Equation 18
                child->gtable_p = parent->gtable_p; // Equation 19 numerator g-table for child
                child->cells_gtable_p = parent->cells_gtable_p; // number of gs in the numerator g-table
                // The pairing enc_B, cells_gtable are taken as input into DCE. On output, they are overridden with this childs enum matrix 
                child->enc_B = parent->enc_B; // Pointer to parent B table, which generates the child B table
                child->cells_gtable = parent->cells_gtable_p;
                child->phc = m;
                child->m = m_m1;
                child->d = ldim_m1;
                child_term_count += 1;
            }
        }
        return child_term_count;
    }

    C_COMPLEX_TYPE evaluate_1d_cf(CauchyTerm* term)
    {
        // Equations 29 and 30
        int enc_lp;
        int enc_lm;
        if(term->A[0] > 0)
        {
            enc_lp = 1; // A * vu^- < 0 -> sgn(A * vu^-) = -1 -> enc_sgn = 1
            enc_lm = 0; // A * vu^+ > 0 -> sgn(A * vu^+) = 1 -> enc_sgn = 0
        }
        else 
        {
            enc_lp = 0; // A * vu^- > 0 -> sgn(A * vu^-) = 1 -> enc_sgn = 0
            enc_lm = 1; // A * vu^+ < 0 -> sgn(A * vu^+) = -1 -> enc_sgn = 1
        }
        const int two_to_phc_minus1 = 1; //(1<<(phc-1)); // since phc = 1
        const int rev_phc_mask = 1; //(1<<phc) - 1;  // since phc = 1
        int size_gtable_p = GTABLE_SIZE_MULTIPLIER * term->cells_gtable_p;
        C_COMPLEX_TYPE gp = lookup_g_numerator(enc_lp, two_to_phc_minus1, rev_phc_mask, term->gtable_p, size_gtable_p, true);
        C_COMPLEX_TYPE gm = lookup_g_numerator(enc_lm, two_to_phc_minus1, rev_phc_mask, term->gtable_p, size_gtable_p, false);
        C_COMPLEX_TYPE g_val = gp / (term->p[0] + I*term->b[0]) - gm / (-term->p[0] + I*term->b[0]);
        return g_val;
    }

    int find_shape_range_lim(int* terms_per_shape, int shape_range)
    {
        int top = 0;
        for(int i = 0; i < shape_range; i++)
            if(terms_per_shape[i] > 0)
                top = i;
        return top+1;
    }

    C_COMPLEX_TYPE evaluate_cpdf(double* xk, bool with_timing_print = false)
    {
        // For now, no fancy stuff, if the cauchy has an issue do not evaluate
        assert(cauchyEst->numeric_moment_errors == 0);
        // If cauchy has not been run a step, cant evaluate.
        assert(cauchyEst->master_step > 0);

        double norm_factor = creal(cauchyEst->fz);
        int dim = cauchyEst->d;
        int shape_range = find_shape_range_lim(cauchyEst->terms_per_shape, cauchyEst->shape_range);
        int* terms_per_shape = (int*) malloc(shape_range * sizeof(int)); 
        null_ptr_check(terms_per_shape);
        CauchyTerm** parent_terms_dp = cauchyEst->terms_dp; // Initially set to these
        CauchyTerm** lowered_terms_dp = (CauchyTerm**) malloc( shape_range * sizeof(CauchyTerm*) );
        null_dptr_check((void**)lowered_terms_dp);
        memcpy(terms_per_shape, cauchyEst->terms_per_shape, shape_range * sizeof(int) );
        int* new_terms_per_shape = (int*) calloc(shape_range , sizeof(int));
        null_ptr_check(new_terms_per_shape);
        double G_SCALE_FACTOR = 1.0;
        CPUTimer tmr_outer_loop;
        tmr_outer_loop.tic();
        CPUTimer tmr;
        for(int d = dim; d > 1; d--)
        {
            tmr.tic();
            int d_lowered = d-1; // lowered dimention of spectral variable
            double* point_xk = (d == dim) ? xk : NULL;
            // Lowering from d to d-1
            int Nt_lowered_apriori = 0;
            for(int i = d; i < shape_range; i++)
                Nt_lowered_apriori += terms_per_shape[i] * i;
            CauchyTerm* lowered_children = (CauchyTerm*) malloc(Nt_lowered_apriori * sizeof(CauchyTerm));
            null_ptr_check(lowered_children);
            int Nt_lowered_aposteriori = 0;
            for(int m = d; m < shape_range; m++)
            {
                if(terms_per_shape[m] > 0)
                {
                    // Shape and dimension sizes for the current lowering step
                    int Nt_shape = terms_per_shape[m];
                    int m_lowered = m-1; // Nominal number of HPs a lowered child has
                    assert(m_lowered > 0); // Do not have code yet to do when m < dim
                    // Allocate space for terms after coalignment
                    BYTE_COUNT_TYPE Nt_lowered_apriori_shape = ((BYTE_COUNT_TYPE)Nt_shape) * m;
                    BYTE_COUNT_TYPE ps_bytes = Nt_lowered_apriori_shape * m_lowered * sizeof(double);
                    BYTE_COUNT_TYPE bs_bytes = Nt_lowered_apriori_shape * d_lowered * sizeof(double);
                    coalign_store.extend_storage(ps_bytes, bs_bytes, d_lowered);
                    // Build all lowered children of parent terms
                    CauchyTerm* parent_terms = parent_terms_dp[m];
                    for(int j = 0; j < Nt_shape; j++)
                    {
                        // Parent -> Lowered Parent -> Lowered Children function
                        CauchyTerm* parent = parent_terms + j;
                        CauchyTerm* children = lowered_children + Nt_lowered_aposteriori;
                        int Nt_children_of_parent = construct_lowered_children(parent, children, point_xk);
                        Nt_lowered_aposteriori += Nt_children_of_parent;
                        // Coalign each of the lowered children of the parent
                        // If the lowered HPA does not coalign...
                        // coalign_store->set_term_ptrs() does not use space for c_map and cs_map
                        // c_map and cs_map are then set to NULL by coalign_store->set_term_ptrs()
                        for(int k = 0; k < Nt_children_of_parent; k++)
                        {
                            new_terms_per_shape[ children[k].mu_coalign() ]++; // Function works for lowering just the same!
                            coalign_store.set_term_ptrs(children+k, m_lowered);
                        }
                    }
                }
            }
            tmr.toc(false);
            if(with_timing_print)
            {
                printf("Parent to Lower Child Generation for %d -> %d dimensions took: %d ms\n", d, d_lowered, tmr.cpu_time_used);
                for(int m = 1; m < shape_range; m++)
                    if(new_terms_per_shape[m] > 0)
                        printf("  Shape %d has %d lowered children!\n", m, new_terms_per_shape[m]);
                printf("  Total Terms: %d\n", sum_vec(new_terms_per_shape, shape_range));
                printf("Term Reduction / Make Gtables:\n");
            }
            // Coalescing the coaligned lowered terms into an array of array storage for term reduction
            // Allocate memory for the array of array storage container
            for(int m = 0; m < shape_range; m++)
            {
                if(new_terms_per_shape[m] > 0)
                {
                    lowered_terms_dp[m] = (CauchyTerm*) malloc( new_terms_per_shape[m] * sizeof(CauchyTerm) );
                    null_ptr_check(lowered_terms_dp[m]);
                }
                else 
                    lowered_terms_dp[m] = (CauchyTerm*) malloc(0);
            }   
            // Organize lowered children into array of arrays
            memset(new_terms_per_shape, 0, shape_range * sizeof(int));
            for(int i = 0; i < Nt_lowered_aposteriori; i++)
            {
                int m = lowered_children[i].m;
                lowered_terms_dp[m][new_terms_per_shape[m]++] = lowered_children[i];
            }
            // Lowered children are now organized.
            free(lowered_children); // Free up lower child temp array.
            coalign_store.unallocate_unused_space(); // free up unused space
            reduce_store.reset_page_idxs();
            memset(terms_per_shape, 0, shape_range * sizeof(int) ); // reset counter
            // If d == dim, create new parent double pointer array
            if(d == dim)
            {
                parent_terms_dp = (CauchyTerm**) malloc( shape_range * sizeof(CauchyTerm*) );
                null_dptr_check((void**)parent_terms_dp);
            }
            // If d < dim, free parent pointer arrays
            else 
            {
                for(int i = 0; i < shape_range; i++)
                    free(parent_terms_dp[i]);
            }

            // Run Term Reduction and then make B/Gtables
            // For each lowered term, we need to build its b-table and the associated g-table
            // We need to test whether term reduction works for the lowering process
            int max_Nt_shape = array_max<int>(new_terms_per_shape, shape_range);
            FastTermRedHelper ftr_helper;
            ftr_helper.init(d_lowered, max_Nt_shape);
            DiffCellEnumHelper dce_helper;
            dce_helper.init(shape_range-1, d_lowered, DCE_STORAGE_MULT);
            int Nt_reduced = 0; // Total number of terms after term reduction has finished
            int Nt_removed = 0; // Total number of terms removed after term approximation
            int total_time = 0;
            for(int m = 0; m < shape_range; m++)
            {
                // Only reduce and carry through terms with m >= d_lowered (Terms with m < d_lowered evaluate to 0)
                if( (new_terms_per_shape[m] > 0) && (m >= d_lowered) )
                {
                    tmr.tic();
                    CauchyTerm* terms = lowered_terms_dp[m];
                    int Nt_shape = new_terms_per_shape[m];
                    memcpy(ftr_helper.F_TR, ftr_helper.F, Nt_shape * sizeof(int) );
                    // Build FTR Maps
                    //tmr.tic();
                    build_ordered_point_maps(
                        terms,
                        ftr_helper.ordered_points, 
                        ftr_helper.forward_map, 
                        ftr_helper.backward_map, 
                        Nt_shape,
                        d_lowered, false);
                    //tmr.toc(false);
                    //bopm_cpu_time = tmr.cpu_time_used;
                    // Run FTR
                    //tmr.tic();
                    fast_term_reduction(
                        terms,
                        ftr_helper.F_TR, 
                        ftr_helper.ordered_points,
                        ftr_helper.forward_map, 
                        ftr_helper.backward_map,
                        REDUCTION_EPS, Nt_shape, m, d_lowered);
                    //tmr.toc(false);
                    //ftr_cpu_time = tmr.cpu_time_used;
                    ForwardFlagArray ffa(ftr_helper.F_TR, Nt_shape);
                    int max_Nt_reduced_shape = ffa.num_terms_after_reduction;
                    int Nt_reduced_shape = 0;
                    int Nt_removed_shape = 0;
                    parent_terms_dp[m] = (CauchyTerm*) malloc( max_Nt_reduced_shape * sizeof(CauchyTerm) );
                    null_ptr_check(parent_terms_dp[m]);
                    CauchyTerm* ftr_terms = parent_terms_dp[m];
                    make_gtables(
                        &Nt_reduced_shape, &Nt_removed_shape,
                        terms,
                        ftr_terms,
                        &ffa,
                        &dce_helper, 
                        &gb_tables,
                        &reduce_store,
                        ftr_helper.F_TR,
                        cauchyEst->B_dense, 
                        G_SCALE_FACTOR,
                        Nt_shape, max_Nt_reduced_shape, 
                        m, d_lowered,
                        -1,-1, false);
                    Nt_reduced += Nt_reduced_shape;
                    Nt_removed += Nt_removed_shape;
                    terms_per_shape[m] = Nt_reduced_shape;
                    
                    tmr.toc(false);
                    total_time += tmr.cpu_time_used;
                    if(with_timing_print)
                        printf("  Shape %d: Reduced to %d terms in %d ms (%d approxed out)\n", m, Nt_reduced_shape, tmr.cpu_time_used, Nt_removed_shape);
                }
                else
                {
                    parent_terms_dp[m] = (CauchyTerm*) malloc(0);
                }
                free(lowered_terms_dp[m]);
                new_terms_per_shape[m] = 0;
            }
            if(with_timing_print)
                printf("  Total Terms: %d, Total Time: %d ms\n", sum_vec(terms_per_shape, shape_range), total_time);

            // Deallocate unused or unneeded memory
            //tmr.tic();
            reduce_store.unallocate_unused_space();
            coalign_store.reset_page_idxs();
            gb_tables.swap_gtables();
            ftr_helper.deinit();
            dce_helper.deinit();
            //tmr.toc(false);
            shape_range -= 1;
        }
        // At d=1, conduct final integration
        // At this point, we should only have terms of 1D, as everything collapses
        int Nt_1d = terms_per_shape[1];
        CauchyTerm* terms = parent_terms_dp[1];
        C_COMPLEX_TYPE unnormalized_f_x = 0;
        tmr.tic();
        for(int i = 0; i < Nt_1d; i++)
            unnormalized_f_x += evaluate_1d_cf(terms + i);
        C_COMPLEX_TYPE normalized_f_x =  unnormalized_f_x / norm_factor * pow(RECIPRICAL_TWO_PI, dim);
        tmr.toc(false);
        tmr_outer_loop.toc(false);
        if(with_timing_print)
        {
            printf("Final 1D integration took: %d ms\n", tmr.cpu_time_used);
            printf("Total Evaluation Time: %d ms\n", tmr_outer_loop.cpu_time_used);
        }
        // Free parent terms
        if(dim > 1)
        {
            for(int i = 0; i < shape_range+1; i++)
                free(parent_terms_dp[i]);
            free(parent_terms_dp);
        }
        free(lowered_terms_dp);
        free(terms_per_shape);
        free(new_terms_per_shape);
        return normalized_f_x;
    }

    // Evaluates marginal of a particular state, or several states
    // xk_marginal must match the indexing of marg_state_idxs
    // That is, if we had spectral vector [nu_1, nu_2, nu_3], but wanted the marginal of states 1 and 3,
    // marg_state_idxs = [0,2], num_marg_state_idxs = 2, xk_marginal=[xk[0], xk[2]]
    C_COMPLEX_TYPE evaluate_marginal_cpdf(double* xk_marginal, int* marg_state_idxs, int num_marg_state_idxs, bool with_timing_print = false)
    {
        // For now, no fancy stuff, if the cauchy has an issue do not evaluate
        assert(cauchyEst->numeric_moment_errors == 0);
        // If cauchy has not been run a step, cant evaluate.
        assert(cauchyEst->master_step > 0);
        assert(cauchyEst->d > 1);

        // Marginal indices checking (marginal state indices should be in ascending order)
        assert(marg_state_idxs != NULL);
        assert(num_marg_state_idxs < cauchyEst->d);
        for(int i = 0; i < num_marg_state_idxs-1; i++)
            assert(marg_state_idxs[i] < marg_state_idxs[i+1]);

        // Reorient the hyperplane arrangements 
        int dim = cauchyEst->d;
        int num_nu_zeros = dim - num_marg_state_idxs;
        int reorient_indices[dim];
        bool F_reorient_indices[dim];
        memset(F_reorient_indices, 1, dim * sizeof(bool));
        for(int i = 0; i < num_marg_state_idxs; i++)
        {
            assert(marg_state_idxs[i] < dim);
            reorient_indices[num_nu_zeros+i] = marg_state_idxs[i];
            F_reorient_indices[marg_state_idxs[i]] = 0;
        }
        int count = 0;
        for(int i = 0; i < dim; i++)
            if(F_reorient_indices[i])
                reorient_indices[count++] = i;
        // indices array now contains how the hyperplanes and b-vectors should be reoriented

        double norm_factor = creal(cauchyEst->fz);
        int shape_range = find_shape_range_lim(cauchyEst->terms_per_shape, cauchyEst->shape_range);
        int* terms_per_shape = (int*) malloc(shape_range * sizeof(int)); 
        null_ptr_check(terms_per_shape);
        CauchyTerm** parent_terms_dp = cauchyEst->terms_dp; // Initially set to these
        CauchyTerm** lowered_terms_dp = (CauchyTerm**) malloc( shape_range * sizeof(CauchyTerm*) );
        null_dptr_check((void**)lowered_terms_dp);
        memcpy(terms_per_shape, cauchyEst->terms_per_shape, shape_range * sizeof(int) );
        int* new_terms_per_shape = (int*) calloc(shape_range , sizeof(int));
        null_ptr_check(new_terms_per_shape);
        double G_SCALE_FACTOR = 1.0;
        CPUTimer tmr_outer_loop;
        tmr_outer_loop.tic();
        CPUTimer tmr;
        for(int d = dim; d > num_nu_zeros; d--)
        {
            tmr.tic();
            int d_lowered = d-1; // lowered dimention of spectral variable
            
            // At the first step, reorient the columns of the parent HPA, and reorient b-vector
            // These pointers and variables below control the program flow to do just this
            double* point_xk_marginal = (d == dim) ? xk_marginal : NULL;
            int* marg_reorient_idxs = (d==dim) ? reorient_indices : NULL;
            int  num_marg_states = (d==dim) ? num_marg_state_idxs : -1;

            // Lowering from d to d-1
            int Nt_lowered_apriori = 0;
            for(int i = d; i < shape_range; i++)
                Nt_lowered_apriori += terms_per_shape[i] * i;
            CauchyTerm* lowered_children = (CauchyTerm*) malloc(Nt_lowered_apriori * sizeof(CauchyTerm));
            null_ptr_check(lowered_children);
            int Nt_lowered_aposteriori = 0;
            for(int m = d; m < shape_range; m++)
            {
                if(terms_per_shape[m] > 0)
                {
                    // Shape and dimension sizes for the current lowering step
                    int Nt_shape = terms_per_shape[m];
                    int m_lowered = m-1; // Nominal number of HPs a lowered child has
                    assert(m_lowered > 0); // Do not have code yet to do when m < dim
                    // Allocate space for terms after coalignment
                    BYTE_COUNT_TYPE Nt_lowered_apriori_shape = ((BYTE_COUNT_TYPE)Nt_shape) * m;
                    BYTE_COUNT_TYPE ps_bytes = Nt_lowered_apriori_shape * m_lowered * sizeof(double);
                    BYTE_COUNT_TYPE bs_bytes = Nt_lowered_apriori_shape * d_lowered * sizeof(double);
                    coalign_store.extend_storage(ps_bytes, bs_bytes, d_lowered);
                    // Build all lowered children of parent terms
                    CauchyTerm* parent_terms = parent_terms_dp[m];
                    for(int j = 0; j < Nt_shape; j++)
                    {
                        // Parent -> Lowered Parent -> Lowered Children function
                        CauchyTerm* parent = parent_terms + j;
                        CauchyTerm* children = lowered_children + Nt_lowered_aposteriori;
                        int Nt_children_of_parent = construct_lowered_children(parent, children, point_xk_marginal, marg_reorient_idxs, num_marg_states);
                        Nt_lowered_aposteriori += Nt_children_of_parent;
                        // Coalign each of the lowered children of the parent
                        // If the lowered HPA does not coalign...
                        // coalign_store->set_term_ptrs() does not use space for c_map and cs_map
                        // c_map and cs_map are then set to NULL by coalign_store->set_term_ptrs()
                        for(int k = 0; k < Nt_children_of_parent; k++)
                        {
                            new_terms_per_shape[ children[k].mu_coalign() ]++; // Function works for lowering just the same!
                            coalign_store.set_term_ptrs(children+k, m_lowered);
                        }
                    }
                }
            }
            // Coalescing the coaligned lowered terms into an array of array storage for term reduction
            // Allocate memory for the array of array storage container
            for(int m = 0; m < shape_range; m++)
            {
                if(new_terms_per_shape[m] > 0)
                {
                    lowered_terms_dp[m] = (CauchyTerm*) malloc( new_terms_per_shape[m] * sizeof(CauchyTerm) );
                    null_ptr_check(lowered_terms_dp[m]);
                }
                else 
                    lowered_terms_dp[m] = (CauchyTerm*) malloc(0);
            }   
            tmr.toc(false);
            if(with_timing_print)
            {
                printf("Parent to Lower Child Generation for %d -> %d dimensions took: %d ms\n", d, d_lowered, tmr.cpu_time_used);
                for(int m = 1; m < shape_range; m++)
                    if(new_terms_per_shape[m] > 0)
                        printf("  Shape %d has %d lowered children!\n", m, new_terms_per_shape[m]);
                printf("  Total Terms: %d\n", sum_vec(new_terms_per_shape, shape_range));
                printf("Term Reduction / Make Gtables:\n");
            }

            // Organize lowered children into array of arrays
            memset(new_terms_per_shape, 0, shape_range * sizeof(int));
            for(int i = 0; i < Nt_lowered_aposteriori; i++)
            {
                int m = lowered_children[i].m;
                lowered_terms_dp[m][new_terms_per_shape[m]++] = lowered_children[i];
            }
            // Lowered children are now organized.
            free(lowered_children); // Free up lower child temp array.
            coalign_store.unallocate_unused_space(); // free up unused space
            reduce_store.reset_page_idxs();
            memset(terms_per_shape, 0, shape_range * sizeof(int) ); // reset counter
            // If d == dim, create new parent double pointer array
            if(d == dim)
            {
                parent_terms_dp = (CauchyTerm**) malloc( shape_range * sizeof(CauchyTerm*) );
                null_dptr_check((void**)parent_terms_dp);
            }
            // If d < dim, free parent pointer arrays
            else 
            {
                for(int i = 0; i < shape_range; i++)
                    free(parent_terms_dp[i]);
            }

            // Run Term Reduction and then make B/Gtables
            // For each lowered term, we need to build its b-table and the associated g-table
            // We need to test whether term reduction works for the lowering process
            int max_Nt_shape = array_max<int>(new_terms_per_shape, shape_range);
            FastTermRedHelper ftr_helper;
            ftr_helper.init(d_lowered, max_Nt_shape);
            DiffCellEnumHelper dce_helper;
            dce_helper.init(shape_range-1, d_lowered, DCE_STORAGE_MULT);
            int Nt_reduced = 0; // Total number of terms after term reduction has finished
            int Nt_removed = 0; // Total number of terms removed after term approximation
            int total_time = 0;
            for(int m = 0; m < shape_range; m++)
            {
                // Only reduce and carry through terms with m >= d_lowered (Terms with m < d_lowered evaluate to 0)
                if( (new_terms_per_shape[m] > 0) && (m >= d_lowered) )
                {
                    tmr.tic();
                    CauchyTerm* terms = lowered_terms_dp[m];
                    int Nt_shape = new_terms_per_shape[m];
                    memcpy(ftr_helper.F_TR, ftr_helper.F, Nt_shape * sizeof(int) );
                    // Build FTR Maps
                    //tmr.tic();
                    build_ordered_point_maps(
                        terms,
                        ftr_helper.ordered_points, 
                        ftr_helper.forward_map, 
                        ftr_helper.backward_map, 
                        Nt_shape,
                        d_lowered, false);
                    //tmr.toc(false);
                    //bopm_cpu_time = tmr.cpu_time_used;
                    // Run FTR
                    //tmr.tic();
                    fast_term_reduction(
                        terms,
                        ftr_helper.F_TR, 
                        ftr_helper.ordered_points,
                        ftr_helper.forward_map, 
                        ftr_helper.backward_map,
                        REDUCTION_EPS, Nt_shape, m, d_lowered);
                    //tmr.toc(false);
                    //ftr_cpu_time = tmr.cpu_time_used;
                    ForwardFlagArray ffa(ftr_helper.F_TR, Nt_shape);
                    int max_Nt_reduced_shape = ffa.num_terms_after_reduction;
                    int Nt_reduced_shape = 0;
                    int Nt_removed_shape = 0;
                    parent_terms_dp[m] = (CauchyTerm*) malloc( max_Nt_reduced_shape * sizeof(CauchyTerm) );
                    null_ptr_check(parent_terms_dp[m]);
                    CauchyTerm* ftr_terms = parent_terms_dp[m];
                    make_gtables(
                        &Nt_reduced_shape, &Nt_removed_shape,
                        terms,
                        ftr_terms,
                        &ffa,
                        &dce_helper, 
                        &gb_tables,
                        &reduce_store,
                        ftr_helper.F_TR,
                        cauchyEst->B_dense, 
                        G_SCALE_FACTOR,
                        Nt_shape, max_Nt_reduced_shape, 
                        m, d_lowered,
                        -1,-1, false);
                    Nt_reduced += Nt_reduced_shape;
                    Nt_removed += Nt_removed_shape;
                    terms_per_shape[m] = Nt_reduced_shape;

                    tmr.toc(false);
                    total_time += tmr.cpu_time_used;
                    if(with_timing_print)
                        printf("  Shape %d: Reduced to %d terms in %d ms (%d approxed out)\n", m, Nt_reduced_shape, tmr.cpu_time_used, Nt_removed_shape);
                }
                else
                {
                    parent_terms_dp[m] = (CauchyTerm*) malloc(0);
                }
                free(lowered_terms_dp[m]);
                new_terms_per_shape[m] = 0;
            }
            if(with_timing_print)
                printf("  Total Terms: %d, Total Time: %d ms\n", sum_vec(terms_per_shape, shape_range), total_time);

            // Deallocate unused or unneeded memory
            //tmr.tic();
            reduce_store.unallocate_unused_space();
            coalign_store.reset_page_idxs();
            gb_tables.swap_gtables();
            ftr_helper.deinit();
            dce_helper.deinit();
            //tmr.toc(false);
            shape_range -= 1;
        }
        // At d=num_nu_zeros, evaluate the parent gs at a chosen value \bar{\nu[1:num_nu_zeros]}
        tmr.tic();
        int ldim = num_nu_zeros;
        double nu_bar[ldim];
        for(int i = 0; i < ldim; i++)
            nu_bar[i] = 2*random_uniform() - 1; // [-1, 1]
        // For each term, compute sign vector w.r.t \bar{\nu[1:num_nu_zeros]}, then lookup g
        C_COMPLEX_TYPE unnormalized_f_x = 0;
        double work[shape_range-1];
        for(int m = ldim; m < shape_range; m++)
        {
            if(terms_per_shape[m] > 0)
            {
                int Nt_shape = terms_per_shape[m];
                int two_to_m_minus1 = (1<<(m-1));
                int rev_m_mask = (1<<m) - 1;
                CauchyTerm* terms = parent_terms_dp[m];
                for(int i = 0; i < Nt_shape; i++)
                {
                    // Compute sign vector of term w.rt \bar{\nu[1:num_nu_zeros]},
                    CauchyTerm* term = terms + i;
                    // Encode sign vector
                    matvecmul(term->A, nu_bar, work, m, ldim);
                    int enc_sv = 0;
                    for(int k = 0; k < m; k++)
                        if(work[k] < 0)
                            enc_sv |= (1<<k);
                    // Look-up the G-value, add it to the unnormalized pdf value at xk_marginal
                    unnormalized_f_x += lookup_g_numerator(enc_sv, two_to_m_minus1, rev_m_mask, term->gtable_p, term->cells_gtable_p * GTABLE_SIZE_MULTIPLIER, true);
                }
            }
        }
        C_COMPLEX_TYPE normalized_f_x =  unnormalized_f_x / norm_factor * pow(RECIPRICAL_TWO_PI, num_marg_state_idxs);
        tmr.toc(false);
        tmr_outer_loop.toc(false);
        if(with_timing_print)
        {
            printf("Final G-Evaluation took: %d ms\n", tmr.cpu_time_used);
            printf("Total Evaluation Time: %d ms\n", tmr_outer_loop.cpu_time_used);
        }
        // Free parent terms
        for(int i = 0; i < shape_range+1; i++)
            free(parent_terms_dp[i]);
        free(parent_terms_dp);
        free(lowered_terms_dp);
        free(terms_per_shape);
        free(new_terms_per_shape);
        return normalized_f_x;
    }

    ~PointWiseNDimCauchyCPDF()
    {
        cauchyEst = NULL;
        coalign_store.deinit();
        reduce_store.deinit();
        gb_tables.deinit();
        lowered_children_workspace.deinit();
    }

};

#endif // _CPDF_NDIM_HPP_