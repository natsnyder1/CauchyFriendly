#ifndef _PREDICTION_HPP_
#define _PREDICTION_HPP_

#include "cauchy_constants.hpp"
#include "cauchy_estimator.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cauchy_util.hpp"
#include "cauchy_linalg.hpp"
#include "cpdf_ndim.hpp"
#include "cpu_timer.hpp"
#include "random_variables.hpp"
#include <complex.h>
#include <cstdlib>
#include <math.h>
#include <pthread.h>


// Unfinished
/*
struct CauchyPredictor
{


    private:
    double* Phi_list;
    double* Gam_list;
    double* Phi_prod;
    double* HPA_extension;
    int size_lists;
    int size_allocd_lists;
    int ALLOC_SIZE;
    int d;
    int pncc;
    CauchyEstimator* cauchyEst;
    bool is_hpa_extension_set;

    // Reallocates the arrays used to generate the HPA extension
    void realloc_lists(int realloc_size)
    {
        size_allocd_lists = realloc_size;
        Phi_list = (double*) realloc(Phi_list, size_allocd_lists * d * d * sizeof(double));
        null_ptr_check(Phi_list);
        Gam_list = (double*) realloc(Gam_list, size_allocd_lists * pncc * d * sizeof(double));
        null_ptr_check(Gam_list);
        HPA_extension = (double*) realloc(HPA_extension, size_allocd_lists * pncc * d * sizeof(double));
        null_ptr_check(HPA_extension);
    }

    // Checks whether to call realloc_lists
    void incr_and_check_list_alloc()
    {
        size_lists++;
        if(size_lists == size_allocd_lists)
            realloc_lists(size_allocd_lists*ALLOC_SIZE);
    }

    // Uses Phi_list and Gam_list to construct Phi_prod and HPA_extension
    // Phi_prod = \Prod_{i=0}^{size_lists-1}Phi_i^T
    void construct_phi_prod_and_hpa_extension()
    {
        assert( size_lists > 0 );
        // Place Pointers at the back of these lists
        double* Gam_ptr = Gam_list + (size_lists-1)*pncc*d;
        double* HPA_ext_ptr = HPA_extension + (size_lists-1)*pncc*d;
        double* Phi_ptr = Phi_list + (size_lists-1)*d*d;
        memset(Phi_prod, 0, d*d*sizeof(double));
        for(int i = 0; i < d; i++)
            Phi_prod[i*d+i] = 1; // Set Phi_prod to identity to begin
        double work[d*d];

        for(int i = 0; i < size_lists; i++)
        {   
            // HPA_EXT[END--] = \GamPtr_{END--} @ \PhiProd^T
            matmatmul(Gam_ptr, Phi_prod, HPA_ext_ptr, pncc, d, d, d, false, true);
            // \PhiProd <- \PhiProd @ PhiPtr_{END--}
            memcpy(work, Phi_prod, d*d*sizeof(double));
            matmatmul(work, Phi_ptr, Phi_prod, pncc, d, d, d, false, false);
            Phi_ptr -= d*d;
            Gam_ptr -= pncc*d;
            HPA_ext_ptr -= pncc*d;
        }
        // \PhiProd is \Phi_N @ \Phi_N-1 @ ... @ Phi_1
        // reflect_array(Phi_prod, d, d); // \PhiProd^T -- better to keep it as stated above
    }


    public:

    CauchyPredictor(CauchyEstimator* _cauchyEst)
    {
        cauchyEst = _cauchyEst;
        d = cauchyEst->d;
        pncc = cauchyEst->pncc;
        Phi_prod = (double*) malloc( d * d * sizeof(double) );
        null_ptr_check(Phi_prod);
        is_hpa_extension_set = false;
        // Initial list size is ALLOC_SIZE
        ALLOC_SIZE = 10;
        size_lists = 0;
        Phi_list = NULL;
        Gam_list = NULL;
        HPA_extension = NULL;
        realloc_lists(ALLOC_SIZE);
    }

    void propagate(double* Phi_k, double* Gamma_k)
    {
        is_hpa_extension_set = false;
        memcpy(Phi_list + size_lists * d * d, Phi_k, d * d * sizeof(double));
        // Store Gamma row-wise, i.e, store [gam_col_1,...,gam_col_n] as [gam_col_1^T;...;gam_col_n^T]
        if(pncc > 1)
        {
            double Gam_tmp[pncc*d];
            transpose(Gamma_k, Gam_tmp, d, pncc);
            memcpy(Gam_list + size_lists * pncc * d, Gamma_k, pncc * d * sizeof(double));
        }
        else
            memcpy(Gam_list + size_lists * pncc * d, Gamma_k, pncc * d * sizeof(double));
        incr_and_check_list_alloc();
    }

    void print_debug_dynamic_matrices()
    {
        construct_phi_prod_and_hpa_extension();
        printf("Product of Phis: Phi_N @ Phi_N-1 @ Phi_N-2 ... @ Phi_1\n");
        print_mat(Phi_prod, d, d);
        printf("HPA Extension [Gamma^T_k @ Phi_Prod_N^T ; Gamma^T_k+1 @ Phi_Prod_{N-1}^T ; ... ; Gamma^T_N");
        print_mat(HPA_extension, size_lists * pncc, d);
    }
};
*/

// Unfinished
/*
CauchyEstimator* get_relative_CF(CauchyEstimator* primary, CauchyEstimator* secondary)
{
    assert( (primary->d) == (secondary->d) );
    // Iterate over all terms of primary 
    int Ntp = primary->Nt;
    int Nts = secondary->Nt;
    int N = Ntp * Nts;
    int d = primary->d;

    CauchyTerm* new_terms = (CauchyTerm*) malloc(N * sizeof(CauchyTerm) );
    null_ptr_check(new_terms);
    ChildTermWorkSpace workspace;
    workspace.init( (primary->shape_range-1) + (secondary->shape_range-1), d);
    int count_new = 0;

    for(int mp = 1; mp < primary->shape_range; mp++)
    {
        int Nt_shape_prim = primary->terms_per_shape[mp];
        if(Nt_shape_prim > 0)
        {
            for(int tp = 0; tp < Nt_shape_prim; tp++)
            {
                // Select a term of the primary
                CauchyTerm* term_prim = primary->terms_dp[mp] + tp;
                // Now loop over all of the secondary
                for(int ms = 1; ms < secondary->shape_range; ms++)
                {
                    int Nt_shape_sec = secondary->terms_per_shape[ms];
                    if(Nt_shape_sec > 0)
                    {
                        for(int ts = 0; ts < Nt_shape_sec; ts++)
                        {
                            // Select a term of the secondary
                            CauchyTerm* term_sec = secondary->terms_dp[ms] + ts;
                            // Now we need to allocate memory for the new term
                            CauchyTerm* new_term = new_terms + count_new;
                            new_term->init_mem(&workspace, 0, mp + ms, d);
                            // Subtract the bs of primary from secondary, store in new term
                            sub_vecs(term_sec->b, term_prim->b, new_term->b, d);
                            // Now transfer (negative of) primary's A to new term and conjoin secondarys to it 
                            memcpy(new_term->A, term_prim->A, mp * d * sizeof(double)); //transfer
                            scale_vec(new_term->A, -1, mp * d); //(negative of)
                            memcpy(new_term->A + mp*d, term_sec->A, ms * d * sizeof(double)); // conjoin
                            // Now transfer primary's p to new term and conjoin secondarys p to it 
                            memcpy(new_term->p, term_prim->p, mp * sizeof(double)); //transfer
                            memcpy(new_term->p + mp, term_sec->A, ms * sizeof(double)); //transfer
                            
                            // We now need to coalign new_term->A
                            new_term->mu_coalign();
                            // Now we need to find the cell enumeration of the new term

                        }
                    }
                }

            }
        }
    }


    for(int tp = 0; tp < Ntp; tp++)
    {
        
        //for(int i = 0; )
        for(int j = 0; j < Nts; j++)
        {

        }
    }
}
*/

// Evaluates the conditional mean and covariance of the relative system
// This function adds the contribution of the inputted term to  rel_norm_factor/rel_cond_mean/rel_cond_covar
// all arguments are for dimension "d"
void eval_rel_sys_moments_for_term(
    C_COMPLEX_TYPE* rel_norm_factor, C_COMPLEX_TYPE* rel_cond_mean, C_COMPLEX_TYPE* rel_cond_covar, 
    double* bar_nu, int d, double* rel_A, double* rel_p, double* rel_b, 
    double* A_parent, int* coalign_map, double* bar_nu_full, const int full_state_dim, const int ZERO_HP_MARKER_VALUE, 
    int mp, int ms, GTABLE prim_gtable, int cells_prim_gtable, GTABLE sec_gtable, int cells_sec_gtable)
{
    C_COMPLEX_TYPE yei[d];
    double tmp_yei[d];
    memset(tmp_yei, 0, d*sizeof(double));

    // Use bar_nu to evaluate a cell (sign-vector) of the HPA, which defines the location g-value
    // Get sign-vector for primary 
    double* A = rel_A;
    double* p = rel_p;
    int* c_map = coalign_map;
    int enc_sv_prim = 0;
    for(int i = 0; i < mp; i++)
    {
        double _s; 
        if(c_map[i] != ZERO_HP_MARKER_VALUE)
            _s = dot_prod(A + i*d, bar_nu, d);
        else
            _s = dot_prod(A_parent + i*full_state_dim, bar_nu_full, full_state_dim);
        double s = sgn( _s );
        add_vecs(tmp_yei, A + i*d, d, p[i] * s);
        if( s < 0 )
            enc_sv_prim |= (1 << i);
    }
    // Get sign-vector for secondary 
    A = rel_A + mp*d;
    p = rel_p + mp;
    c_map = coalign_map + mp;
    int enc_sv_sec = 0;
    for(int i = 0; i < ms; i++)
    {
        double _s; 
        if(c_map[i] != ZERO_HP_MARKER_VALUE)
            _s = dot_prod(A + i*d, bar_nu, d);
        else
            _s = dot_prod(A_parent + (i+mp)*full_state_dim, bar_nu_full, full_state_dim);
        double s = sgn( _s );
        add_vecs(tmp_yei, A + i*d, d, p[i] * s);
        if( s < 0 )
            enc_sv_sec |= (1 << i);
    }
    
    // Evaluate G-Value of Primary
    int two_to_mp_minus1 = 1<<(mp-1);
    int rev_mp_mask = (1<<mp)-1;
    C_COMPLEX_TYPE g_prim = lookup_g_numerator(enc_sv_prim, two_to_mp_minus1, rev_mp_mask, prim_gtable, cells_prim_gtable, true);
    // Evaluate G-Value of Secondary
    int two_to_ms_minus1 = 1<<(ms-1);
    int rev_ms_mask = (1<<ms)-1;
    C_COMPLEX_TYPE g_sec = lookup_g_numerator(enc_sv_sec, two_to_ms_minus1, rev_ms_mask, sec_gtable, cells_sec_gtable, true);
    // Get Aggregate G-Value
    C_COMPLEX_TYPE g_val = g_prim * g_sec;


    for(int i = 0; i < d; i++)
        yei[i] = -tmp_yei[i] + I*rel_b[i]; //yei[j] = CMPLX(-tmp_yei[j], b[j]);

    // Evaluate contribution to the conditional mean and covariance
    *rel_norm_factor += g_val;
    for(int j = 0; j < d; j++)
    {
        C_COMPLEX_TYPE y = yei[j];
        rel_cond_mean[j] += g_val * y;
        for(int k = 0; k < d; k++)
            rel_cond_covar[j*d + k] -= g_val * y * yei[k];
    }    
}   

void finalize_rel_sys_moments(C_COMPLEX_TYPE* rel_norm_factor, C_COMPLEX_TYPE* rel_cond_mean, C_COMPLEX_TYPE* rel_cond_covar, int d)
{
    C_COMPLEX_TYPE fz = (*rel_norm_factor);
    C_COMPLEX_TYPE Ifz = I * fz; //CMPLX(cimag(fz), creal(fz)); // imaginary fz
    for(int i = 0; i < d; i++)
        rel_cond_mean[i] /= Ifz;

    for(int i = 0; i < d; i++)
    {
        for(int j = 0; j < d; j++)
        {
            rel_cond_covar[i*d + j] = (rel_cond_covar[i*d+j] / fz) - rel_cond_mean[i] * rel_cond_mean[j];
        }
    }
}

// Caches the "relative-then-transformed" term using the 2D cpdf transform caching structure 
void cache_rel_marg2d_term_for_cpdf(
    Cached2DCPDFTermContainer* cached_2d_terms,
    double* relA_2d, double* relp_2d, double* relb_2d, const int rel_m,
    int* c_map_2d, int* cs_map_2d,
    GTABLE gtable_prim, const int gtable_prim_size, 
    GTABLE gtable_sec, const int gtable_sec_size,
    double* A_parent, const int mpp, const int mps, const int state_dim, double* bar_nu_full,
    bool full_solve = false)
{
    int ZERO_HP_MARKER_VALUE = 32;
    const int rel_mp = mpp + mps;
    // Values for constructing sign vectors of the parent g-tables
    // The first mpp signs are for the primary
    int mask_mpp = (1<<mpp)-1; 
    int two_to_mpp_minus1 = 1<<(mpp-1);
    int rev_mpp_mask = mask_mpp;
    // The other signs (up to mpp+mps) are apart of the secondary
    // Can just clear out these signs through bit shifting
    int two_to_mps_minus1 = 1<<(mps-1);
    int rev_mps_mask = (1<<mps)-1;

    // Variables for running 2D cpdf formula
    const int d = 2;
    const int two_m = 2*rel_m;
    int enc_sv;
    double thetas[two_m + full_solve];
    double SVs[(1+full_solve)*rel_m*rel_m];
    double* SV;
    double A_scaled[rel_m*d];
    double* a;
    marg2d_get_cell_wall_angles(thetas, relA_2d, rel_m); // angles of cell walls
    if(full_solve)
        thetas[2*rel_m] = thetas[0] + 2*PI;
    marg2d_get_SVs(SVs, relA_2d, thetas, rel_m, full_solve); // SVs corresponding to cells within the above (sequential) cell walls
    for(int i = 0; i < rel_m; i++)
        for(int j = 0; j < d; j++)
            A_scaled[i*d + j] = relA_2d[i*d + j] * relp_2d[i];

    int cells_to_visit = rel_m * (1 + full_solve);
    cached_2d_terms->set_term_ptrs(rel_m);
    Cached2DCPDFTerm* cached_term = cached_2d_terms->cached_terms + cached_2d_terms->current_term_idx;
    cached_term->b[0] = relb_2d[0];
    cached_term->b[1] = relb_2d[1];
    cached_term->m = rel_m;
    
    double gam1_real;
    double gam2_real;
    C_COMPLEX_TYPE g_val;
    C_COMPLEX_TYPE g_prim;
    C_COMPLEX_TYPE g_sec;
    double theta1;
    double sin_t1;
    double cos_t1;
    for(int i = 0; i < cells_to_visit; i++)
    {

        // 1.) Encode SV and extract G in the i-th cell
        SV = SVs + i*rel_m;
        enc_sv = 0;
        if(c_map_2d == NULL)
        {
            for(int j = 0; j < rel_m; j++)
                if( SV[j] < 0 )
                    enc_sv |= (1 << j);
        }
        else 
        {
            for(int j = 0; j < rel_mp; j++)
            {
                if(c_map_2d[j] == ZERO_HP_MARKER_VALUE)
                {
                    double* ap_j = A_parent + j*state_dim;
                    double in_prod = dot_prod(ap_j, bar_nu_full, state_dim);
                    if( in_prod < 0 )
                        enc_sv |= (1<<j);
                }
                else 
                {
                    int b_idx = c_map_2d[j];
                    int flip = cs_map_2d[j];
                    int s = SV[b_idx] * flip;
                    if(s < 0)
                        enc_sv |= (1<<j);
                }
            }
        }
        // The first mpp signs are for the primary
        int enc_sv_prim = enc_sv & mask_mpp;
        // Overloading numerator lookup function to lookup gtable Gs (just replace "phc" definitions with "m" definitions, which is done here)
        g_prim = lookup_g_numerator(enc_sv_prim, two_to_mpp_minus1, rev_mpp_mask, gtable_prim, gtable_prim_size, true);

        // The other signs (up to mpp+mps) are apart of the secondary
        // Can just clear out these signs through bit shifting
        int enc_sv_sec = enc_sv >> mpp;
        // Overloading numerator lookup function to lookup gtable Gs (just replace "phc" definitions with "m" definitions, which is done here)
        g_sec = lookup_g_numerator(enc_sv_sec, two_to_mps_minus1, rev_mps_mask, gtable_sec, gtable_sec_size, true);
        
        // The contribution of this cell is the g-value of the primary and secondary multiplied
        g_val = g_prim * g_sec;

        // 2.) Evaluate the real part of gamma1 parameter and real part of gamma2 parameter
        gam1_real = 0;
        gam2_real = 0;
        for(int j = 0; j < rel_m; j++)
        {
            a = A_scaled + j*d;
            gam1_real -= a[0] * SV[j];
            gam2_real -= a[1] * SV[j];
        }

        // Solution of integral:  sin(theta) / (gamma1*gamma1*cos(theta) + gamma1*gamma2*sin(theta));
        theta1 = thetas[i];
        sin_t1 = sin(theta1);
        cos_t1 = cos(theta1);

        // Cache all relevant components
        cached_term->sin_thetas[i] = sin_t1;
        cached_term->cos_thetas[i] = cos_t1;
        cached_term->gam1_reals[i] = gam1_real;
        cached_term->gam2_reals[i] = gam2_real;
        cached_term->g_vals[i] = g_val;
    }
    cached_term->sin_thetas[cells_to_visit] = sin(thetas[cells_to_visit]);
    cached_term->cos_thetas[cells_to_visit] = cos(thetas[cells_to_visit]);
    cached_2d_terms->incr_cached_term_idx();
}

bool approx_out_rel_marg2d_term_for_cpdf(Cached2DCPDFTerm* term, const double norm_fact, const double TERM_EPS)
{
    // Only looking at the lower half of the integration of each cell right now
    double total_contrib = 0;
    const double DEN_EPS = 1e-14;
    for(int i = 0; i < term->m; i++)
    {
        double a1 = term->gam1_reals[i];
        double a2 = term->gam2_reals[i];
        double c1 = term->cos_thetas[i];
        double s1 = term->sin_thetas[i];
        double c2 = term->cos_thetas[i+1];
        double s2 = term->sin_thetas[i+1];
        double den1 = a1*(a1*c1 + a2*s1);
        double den2 = a1*(a1*c2 + a2*s2);
        if( fabs(den1) < DEN_EPS )
            return false;
        if( fabs(den2) < DEN_EPS )
            return false;
        double g_val_r = cabs(term->g_vals[i]);
        // Max value of contribution will be GVAL[i] / GX
        double contrib = fabs( (g_val_r * s2) / den2 ) + fabs( (g_val_r * s1) / den1 );
        total_contrib += contrib;
    }
    total_contrib *= (2*norm_fact);
    return fabs(total_contrib) < TERM_EPS;
}

struct ThreadedCachedRSysArgs
{
    // Inputs
    CauchyEstimator* primary;
    CauchyEstimator* secondary;
    double* Trel;
    bool get_relative_moments;
    double approx_norm_factor;
    bool with_term_approx;
    double term_approx_eps;
    double* bar_nu;
    double* bar_nu_full;
    int num_threads;
    int tid;
    bool full_solve;
    // Outputs 
    Cached2DCPDFTermContainer* cached_2d_terms;
    C_COMPLEX_TYPE chunk_rel_norm_factor;
    C_COMPLEX_TYPE chunk_rel_cond_mean[2];
    C_COMPLEX_TYPE chunk_rel_cond_covar[4];
    BYTE_COUNT_TYPE terms_approxed_out; // delete
    BYTE_COUNT_TYPE num_terms_approxed_out;
};

void* call_threaded_marg2d_relative_and_transformed_cpdf(void* args)
{
    const double ZERO_EPSILON = MU_EPS;
    const int ZERO_HP_MARKER_VALUE = 32;

    ThreadedCachedRSysArgs* rsa = (ThreadedCachedRSysArgs*) args; // Relative system arguments
    rsa->cached_2d_terms = (Cached2DCPDFTermContainer*) malloc(sizeof(Cached2DCPDFTermContainer));
    Cached2DCPDFTermContainer* cached_2d_terms = rsa->cached_2d_terms;
    
    CauchyEstimator* primary = rsa->primary;
    CauchyEstimator* secondary = rsa->secondary;
    double* Trel = rsa->Trel;
    const double approx_norm_factor = rsa->approx_norm_factor;
    const bool get_relative_moments = rsa->get_relative_moments;
    const bool with_term_approx = rsa->with_term_approx;
    const double term_approx_eps = rsa->term_approx_eps;
    const bool FULL_SOLVE = rsa->full_solve;
    cached_2d_terms->init(1, FULL_SOLVE);

    int tid = rsa->tid;
    int num_threads = rsa->num_threads;

    double* bar_nu = rsa->bar_nu; 
    double* bar_nu_full = rsa->bar_nu_full;
    
    C_COMPLEX_TYPE* rel_norm_factor = &(rsa->chunk_rel_norm_factor);
    *rel_norm_factor = 0;
    C_COMPLEX_TYPE* rel_cond_mean = rsa->chunk_rel_cond_mean;
    memset(rel_cond_mean, 0, 2 * sizeof(C_COMPLEX_TYPE) );
    C_COMPLEX_TYPE* rel_cond_covar = rsa->chunk_rel_cond_covar;
    memset(rel_cond_covar, 0, 4 * sizeof(C_COMPLEX_TYPE) );

    // Total terms that will be created before any approximations
    const int Ntp = primary->Nt;
    const int Nts = secondary->Nt;
    const BYTE_COUNT_TYPE N = ((BYTE_COUNT_TYPE)Ntp) * Nts;
    //printf("The resulting relative CF will have %d*%d=%d terms before any approximations!\n", Ntp,Nts,N);
    const int d = primary->d;
    const int max_m = ((primary->shape_range-1) + (secondary->shape_range-1));

    // Holds a temporary relative term of dimension d 
    double A_rel[max_m*d];
    double p_rel[max_m];
    double b_rel[d];

    // The temporary transformed relative 2D term
    double A_2d[max_m*2];
    double* p_2d;
    double Ac_2d[max_m*2];
    double pc_2d[max_m];
    double b_2d[2];
    int c_map_2d[max_m];
    int cs_map_2d[max_m];
    
    BYTE_COUNT_TYPE num_terms_approxed_out = 0;
    const BYTE_COUNT_TYPE TERM_ALLLOC_CAP = 500000;
    // Loop over all shapes of the primary
    for(int mp = 1; mp < primary->shape_range; mp++)
    {
        int Nt_shape_prim = primary->terms_per_shape[mp];
        // Loop over all of the secondary
        for(int ms = 1; ms < secondary->shape_range; ms++)
        {
            int mps = mp + ms;
            int Nt_shape_sec = secondary->terms_per_shape[ms];
            // Loop over all terms of a given shape of the primary 
            for(int tp = 0; tp < Nt_shape_prim; tp++)
            {
                // Select a term of the primary
                CauchyTerm* term_prim = primary->terms_dp[mp] + tp;
                // Loop over all terms of a given shape of the secondary 
                for(int ts = tid; ts < Nt_shape_sec; ts += num_threads)
                {
                    if( (tp == 0) && (ts == tid) )
                    {
                        int tid_new_terms;
                        // Cap new term alloc to TERM_ALLLOC_CAP -> most of these will get approxed out
                        BYTE_COUNT_TYPE true_new_num_terms = ( (BYTE_COUNT_TYPE) Nt_shape_prim ) * Nt_shape_sec;
                        if(true_new_num_terms > TERM_ALLLOC_CAP)
                            tid_new_terms = TERM_ALLLOC_CAP;
                        else
                            tid_new_terms = Nt_shape_prim * (Nt_shape_sec + num_threads - 1 / num_threads);
                        //tid_new_terms = (Nt_shape_prim * Nt_shape_sec + num_threads - 1) / num_threads;
                        cached_2d_terms->extend_storage_by(tid_new_terms, mps);
                    }
                    // Select a term of the secondary
                    CauchyTerm* term_sec = secondary->terms_dp[ms] + ts;
                    // Subtract the bs of primary from secondary, store in new term
                    sub_vecs(term_sec->b, term_prim->b, b_rel, d);
                    // Now transfer (negative of) primary's A to new term and conjoin secondarys to it 
                    memcpy(A_rel, term_prim->A, mp * d * sizeof(double)); //transfer
                    scale_vec(A_rel, -1, mp * d); //(negative of)
                    memcpy(A_rel + mp*d, term_sec->A, ms * d * sizeof(double)); // conjoin
                    // Now transfer primary's p to new term and conjoin secondarys p to it 
                    memcpy(p_rel, term_prim->p, mp * sizeof(double)); //transfer
                    memcpy(p_rel + mp, term_sec->p, ms * sizeof(double)); //transfer

                    // Transform A and b using the transformation Trel as: A2D = A @ Trel.T and b2d = Trel @ b, and p2d = p_rel (unchanged until A2D is normalized)
                    matmatmul(A_rel, Trel, A_2d, mps, d, 2, d, false, true);
                    matvecmul(Trel, b_rel, b_2d, 2, d, false);
                    p_2d = p_rel; // transformation doesnt change this...before coalignments

                    // We now need to coalign workspace.A = [-A_prim; A_sec]
                    int rel_mc = marg2d_remove_zeros_and_coalign(A_2d, p_2d, Ac_2d, pc_2d, c_map_2d, cs_map_2d, mps, ZERO_EPSILON, ZERO_HP_MARKER_VALUE);
                    
                    // Evaluate the moments. In the presence of zero hyperplanes after the transformation, work with the full, non-lowered system
                    if(get_relative_moments)
                        eval_rel_sys_moments_for_term(rel_norm_factor, rel_cond_mean, rel_cond_covar, bar_nu, 2, A_2d, p_2d, b_2d, A_rel, c_map_2d, bar_nu_full, d, ZERO_HP_MARKER_VALUE, mp, ms, term_prim->gtable_p, term_prim->cells_gtable_p, term_sec->gtable_p, term_sec->cells_gtable_p);
                    
                    // Now we need to aquire the sign vectors of the 2D arrangement, and build out the cached term's g-array
                    cache_rel_marg2d_term_for_cpdf(
                        cached_2d_terms,
                        Ac_2d, pc_2d, b_2d, rel_mc,
                        c_map_2d, cs_map_2d,
                        // these are really the gtables, after FTR, the Gtable gets swapped to gtable_parent
                        term_prim->gtable_p, term_prim->cells_gtable_p, 
                        term_sec->gtable_p, term_sec->cells_gtable_p,
                        A_rel, mp, ms, d, bar_nu_full,
                        FULL_SOLVE);
                    // As this will produce so many terms, we may wish to run some approximation step here and remove super inconsequential terms
                    if(with_term_approx)
                    {
                        int new_cti = cached_2d_terms->current_term_idx - 1; // cached term index
                        Cached2DCPDFTerm* cached_term = cached_2d_terms->cached_terms + new_cti;
                        if( approx_out_rel_marg2d_term_for_cpdf(cached_term, approx_norm_factor, term_approx_eps) )
                        {
                            num_terms_approxed_out++;
                            cached_2d_terms->pop_term_ptrs(); // removes the last set term
                        }
                    }
                }
            }
        }
    }
    rsa->num_terms_approxed_out = num_terms_approxed_out;
    return NULL;
}

// Relative system: rel = secondary - primary 
// Transformed system: Trel @ rel
// Trel must be 2 x D
// Returns all "relative-then-transformed" terms using the 2D cpdf transform caching container structure 
// If get_relative_moments==true, sets rel_norm_factor/rel_cond_mean/rel_cond_covar
Cached2DCPDFTermContainer* get_marg2d_relative_and_transformed_cpdf(
    CauchyEstimator* primary, CauchyEstimator* secondary, double* Trel, 
    const bool FULL_SOLVE, const bool with_timing,
    const bool get_relative_moments = false, 
    C_COMPLEX_TYPE* rel_norm_factor = NULL, C_COMPLEX_TYPE* rel_cond_mean = NULL, C_COMPLEX_TYPE* rel_cond_covar = NULL, 
    const bool with_term_approx = false, const double term_approx_eps = 1e-8, int num_threads = 1)
{
    assert( (primary->d) == (secondary->d) );
    if(SKIP_LAST_STEP == true)
    {
        if( (primary->master_step / primary->p) == primary->num_estimation_steps )
        {
            printf("Cannot run relative marginalization process for last step if SKIP_LAST_STEP == true!\n");
            return NULL;
        }
        if( (secondary->master_step / secondary->p) == secondary->num_estimation_steps )
        {
            printf("Cannot run relative marginalization process on last step if SKIP_LAST_STEP == true!\n");
            return NULL;
        }
    }
    assert(num_threads > 0);
    const double approx_norm_factor = RECIPRICAL_TWO_PI * RECIPRICAL_TWO_PI / creal(primary->fz) / creal(secondary->fz);
    int num_terms_approxed_out = 0;

    // Evaluates the conditional mean and covariance of the relative system, if the arguments are provided
    if(get_relative_moments)
    {
        assert(rel_norm_factor != NULL);
        *rel_norm_factor = 0;
        assert(rel_cond_mean != NULL);
        memset(rel_cond_mean, 0, 2*sizeof(C_COMPLEX_TYPE));
        assert(rel_cond_covar != NULL);
        memset(rel_cond_covar, 0, 4*sizeof(C_COMPLEX_TYPE));
    }
    double ZERO_EPSILON = MU_EPS;
    int ZERO_HP_MARKER_VALUE = 32;

    // Total terms that will be created before any approximations
    BYTE_COUNT_TYPE Ntp = primary->Nt;
    BYTE_COUNT_TYPE Nts = secondary->Nt;
    BYTE_COUNT_TYPE N = Ntp * Nts;
    if(num_threads > N)
        num_threads = 1;

    //printf("The resulting relative CF will have %d*%d=%d terms before any approximations!\n", Ntp,Nts,N);
    int d = primary->d;
    int max_m = ((primary->shape_range-1) + (secondary->shape_range-1));
    assert(max_m < 32);

    // create a random point \bar\nu, in the event A_2d has a singular HPA 
    double bar_nu_full[d];
    double bar_nu[2];
    for(int i = 0; i < d; i++)
        bar_nu_full[i] = 2*random_uniform_open() -1;
    //bar_nu_full[0] = 0.1982739812; bar_nu_full[1] = -0.278786; bar_nu_full[2] = 0.781236; 

    matvecmul(Trel, bar_nu_full, bar_nu, 2, d);
    Cached2DCPDFTermContainer* cached_2d_terms = (Cached2DCPDFTermContainer*) malloc(sizeof(Cached2DCPDFTermContainer));
    CPUTimer tmr;
    if(num_threads == 1)
    {
        tmr.tic();
        cached_2d_terms->init(1, FULL_SOLVE);
        // Holds a temporary relative term of dimension d
        double A_rel[max_m*d];
        double p_rel[max_m];
        double b_rel[d];

        // The temporary transformed relative 2D term
        double A_2d[max_m*2];
        double* p_2d;
        double Ac_2d[max_m*2];
        double pc_2d[max_m];
        double b_2d[2];
        int c_map_2d[max_m];
        int cs_map_2d[max_m];
        //int count_new = 0;

        // Loop over all shapes of the primary
        for(int mp = 1; mp < primary->shape_range; mp++)
        {
            int Nt_shape_prim = primary->terms_per_shape[mp];
            // Loop over all of the secondary
            for(int ms = 1; ms < secondary->shape_range; ms++)
            {
                int mps = mp + ms;
                int Nt_shape_sec = secondary->terms_per_shape[ms];
                // Loop over all terms of a given shape of the primary 
                for(int tp = 0; tp < Nt_shape_prim; tp++)
                {
                    // Select a term of the primary
                    CauchyTerm* term_prim = primary->terms_dp[mp] + tp;
                    // Loop over all terms of a given shape of the secondary 
                    for(int ts = 0; ts < Nt_shape_sec; ts++)
                    {
                        if( (tp == 0) && (ts == 0) )
                            cached_2d_terms->extend_storage_by(Nt_shape_prim*Nt_shape_sec, mps);
                        // Select a term of the secondary
                        CauchyTerm* term_sec = secondary->terms_dp[ms] + ts;
                        // Subtract the bs of primary from secondary, store in new term
                        sub_vecs(term_sec->b, term_prim->b, b_rel, d);
                        // Now transfer (negative of) primary's A to new term and conjoin secondarys to it 
                        memcpy(A_rel, term_prim->A, mp * d * sizeof(double)); //transfer
                        scale_vec(A_rel, -1, mp * d); //(negative of)
                        memcpy(A_rel + mp*d, term_sec->A, ms * d * sizeof(double)); // conjoin
                        // Now transfer primary's p to new term and conjoin secondarys p to it 
                        memcpy(p_rel, term_prim->p, mp * sizeof(double)); //transfer
                        memcpy(p_rel + mp, term_sec->p, ms * sizeof(double)); //transfer

                        // Transform A and b using the transformation Trel as: A2D = A @ Trel.T and b2d = Trel @ b, and p2d = p_rel (unchanged until A2D is normalized)
                        matmatmul(A_rel, Trel, A_2d, mps, d, 2, d, false, true);
                        matvecmul(Trel, b_rel, b_2d, 2, d, false);
                        p_2d = p_rel; // transformation doesnt change this...before coalignments

                        // We now need to coalign workspace.A = [-A_prim; A_sec]
                        int rel_mc = marg2d_remove_zeros_and_coalign(A_2d, p_2d, Ac_2d, pc_2d, c_map_2d, cs_map_2d, mps, ZERO_EPSILON, ZERO_HP_MARKER_VALUE);
                        
                        // Evaluate the moments. In the presence of zero hyperplanes after the transformation, work with the full, non-lowered system
                        if(get_relative_moments)
                            eval_rel_sys_moments_for_term(rel_norm_factor, rel_cond_mean, rel_cond_covar, bar_nu, 2, A_2d, p_2d, b_2d, A_rel, c_map_2d, bar_nu_full, d, ZERO_HP_MARKER_VALUE, mp, ms, term_prim->gtable_p, term_prim->cells_gtable_p, term_sec->gtable_p, term_sec->cells_gtable_p);
                        
                        // Now we need to aquire the sign vectors of the 2D arrangement, and build out the cached term's g-array
                        cache_rel_marg2d_term_for_cpdf(
                            cached_2d_terms,
                            Ac_2d, pc_2d, b_2d, rel_mc,
                            c_map_2d, cs_map_2d,
                            // these are really the gtables, after FTR, the Gtable gets swapped to gtable_parent
                            term_prim->gtable_p, term_prim->cells_gtable_p, 
                            term_sec->gtable_p, term_sec->cells_gtable_p,
                            A_rel, mp, ms, d, bar_nu_full,
                            FULL_SOLVE);
                        // As this will produce so many terms, we may wish to run some approximation step here and remove super inconsequential terms
                        if(with_term_approx)
                        {
                            int new_cti = cached_2d_terms->current_term_idx - 1; // cached term index
                            Cached2DCPDFTerm* cached_term = cached_2d_terms->cached_terms + new_cti;
                            if( approx_out_rel_marg2d_term_for_cpdf(cached_term, approx_norm_factor, term_approx_eps) )
                            {
                                num_terms_approxed_out++;
                                cached_2d_terms->pop_term_ptrs(); // removes the last set term
                            }
                        }
                    }
                }
            }
        }
        if(get_relative_moments)
            finalize_rel_sys_moments(rel_norm_factor, rel_cond_mean, rel_cond_covar, 2);
        tmr.toc(false);
        if(with_timing)
        {
            printf("Relative CPDF System Caching:\n  Caching %llu x %llu = %llu terms took %d ms\n", Ntp, Nts, N, tmr.cpu_time_used);
            if(with_term_approx)
                printf("  Removed %d/%llu terms using epsilon=%.3E. Term total=%llu\n", num_terms_approxed_out, N, term_approx_eps, N-num_terms_approxed_out);
        }
        return cached_2d_terms;
    }
    else 
    {
        tmr.tic();
        cached_2d_terms->init(0, FULL_SOLVE);
        // Special multithreaded evaluation of terms to be cached
        pthread_t tids[num_threads];
        ThreadedCachedRSysArgs args[num_threads];

        // setup the multithreaded args and launch thread 
        for(int i = 0; i < num_threads; i++)
        {
            args[i].primary = primary;
            args[i].secondary = secondary;
            args[i].Trel = Trel;
            args[i].get_relative_moments = get_relative_moments;
            args[i].approx_norm_factor = approx_norm_factor;
            args[i].with_term_approx = with_term_approx;
            args[i].term_approx_eps = term_approx_eps;
            args[i].bar_nu = bar_nu;
            args[i].bar_nu_full = bar_nu_full;
            args[i].num_threads = num_threads;
            args[i].tid = i;
            args[i].full_solve = FULL_SOLVE;
            pthread_create(tids + i, NULL, call_threaded_marg2d_relative_and_transformed_cpdf, args + i);
        }

        for(int i = 0; i < num_threads; i++)
            pthread_join(tids[i], NULL);

        // thread by thread, realloc the main container, merge the thread's container into it, then delete the thread's container
        BYTE_COUNT_TYPE num_pages_thetas = 0;
        BYTE_COUNT_TYPE num_pages_gams = 0;
        BYTE_COUNT_TYPE num_pages_gvals = 0;
        BYTE_COUNT_TYPE num_terms_total = 0;
        for(int i = 0; i < num_threads; i++)
        {
            num_pages_thetas += args[i].cached_2d_terms->chunked_sin_thetas.current_page_idx+1;
            num_pages_gams += args[i].cached_2d_terms->chunked_gam1_reals.current_page_idx+1;
            num_pages_gvals += args[i].cached_2d_terms->chunked_g_vals.current_page_idx+1;
            num_terms_total += args[i].cached_2d_terms->current_term_idx;
            num_terms_approxed_out += args[i].num_terms_approxed_out;
        }
        cached_2d_terms->current_term_idx = num_terms_total;
        cached_2d_terms->cached_terms = (Cached2DCPDFTerm*) realloc(cached_2d_terms->cached_terms, num_terms_total * sizeof(Cached2DCPDFTerm) );
        cached_2d_terms->terms_alloc_total = num_terms_total;

        cached_2d_terms->chunked_cos_thetas.chunked_elems = (double**) realloc(cached_2d_terms->chunked_cos_thetas.chunked_elems, num_pages_thetas * sizeof(double*) );
        cached_2d_terms->chunked_cos_thetas.used_elems_per_page = (BYTE_COUNT_TYPE*) realloc(cached_2d_terms->chunked_cos_thetas.used_elems_per_page, num_pages_thetas * sizeof(BYTE_COUNT_TYPE) );
        cached_2d_terms->chunked_cos_thetas.page_limit = num_pages_thetas;
        
        cached_2d_terms->chunked_sin_thetas.chunked_elems = (double**) realloc(cached_2d_terms->chunked_sin_thetas.chunked_elems, num_pages_thetas * sizeof(double*) );
        cached_2d_terms->chunked_sin_thetas.used_elems_per_page = (BYTE_COUNT_TYPE*) realloc(cached_2d_terms->chunked_sin_thetas.used_elems_per_page, num_pages_thetas * sizeof(BYTE_COUNT_TYPE) );
        cached_2d_terms->chunked_sin_thetas.page_limit = num_pages_thetas;

        cached_2d_terms->chunked_gam1_reals.chunked_elems = (double**) realloc(cached_2d_terms->chunked_gam1_reals.chunked_elems, num_pages_gams * sizeof(double*) );
        cached_2d_terms->chunked_gam1_reals.used_elems_per_page = (BYTE_COUNT_TYPE*) realloc(cached_2d_terms->chunked_gam1_reals.used_elems_per_page, num_pages_gams * sizeof(BYTE_COUNT_TYPE) );
        cached_2d_terms->chunked_gam1_reals.page_limit = num_pages_gams;

        cached_2d_terms->chunked_gam2_reals.chunked_elems = (double**) realloc(cached_2d_terms->chunked_gam2_reals.chunked_elems, num_pages_gams * sizeof(double*) );
        cached_2d_terms->chunked_gam2_reals.used_elems_per_page = (BYTE_COUNT_TYPE*) realloc(cached_2d_terms->chunked_gam2_reals.used_elems_per_page, num_pages_gams * sizeof(BYTE_COUNT_TYPE) );
        cached_2d_terms->chunked_gam2_reals.page_limit = num_pages_gams;

        cached_2d_terms->chunked_g_vals.chunked_elems = (C_COMPLEX_TYPE**) realloc(cached_2d_terms->chunked_g_vals.chunked_elems, num_pages_gvals * sizeof(C_COMPLEX_TYPE*) );
        cached_2d_terms->chunked_g_vals.used_elems_per_page = (BYTE_COUNT_TYPE*) realloc(cached_2d_terms->chunked_g_vals.used_elems_per_page, num_pages_gvals * sizeof(BYTE_COUNT_TYPE) );
        cached_2d_terms->chunked_g_vals.page_limit = num_pages_gvals;

        // Now fill thetas / gams / gvals / total 
        num_pages_thetas = 0;
        num_pages_gams = 0;
        num_pages_gvals = 0;
        num_terms_total = 0;
        for(int i = 0; i < num_threads; i++)
        {
            // Get individual container counts
            int argi_num_pages_thetas = args[i].cached_2d_terms->chunked_sin_thetas.current_page_idx+1;
            int argi_num_pages_gams = args[i].cached_2d_terms->chunked_gam1_reals.current_page_idx+1;
            int argi_num_pages_gvals = args[i].cached_2d_terms->chunked_g_vals.current_page_idx+1;
            int argi_num_terms_total = args[i].cached_2d_terms->current_term_idx;
            // Now concatenate this into the single container
            memcpy(cached_2d_terms->cached_terms + num_terms_total, args[i].cached_2d_terms->cached_terms, argi_num_terms_total * sizeof(Cached2DCPDFTerm) );

            memcpy(cached_2d_terms->chunked_cos_thetas.chunked_elems + num_pages_thetas, args[i].cached_2d_terms->chunked_cos_thetas.chunked_elems, argi_num_pages_thetas * sizeof(double*) );
            memcpy(cached_2d_terms->chunked_cos_thetas.used_elems_per_page + num_pages_thetas, args[i].cached_2d_terms->chunked_cos_thetas.used_elems_per_page, argi_num_pages_thetas * sizeof(int) );
            
            memcpy(cached_2d_terms->chunked_sin_thetas.chunked_elems + num_pages_thetas, args[i].cached_2d_terms->chunked_sin_thetas.chunked_elems, argi_num_pages_thetas * sizeof(double*) );
            memcpy(cached_2d_terms->chunked_sin_thetas.used_elems_per_page + num_pages_thetas, args[i].cached_2d_terms->chunked_sin_thetas.used_elems_per_page, argi_num_pages_thetas * sizeof(int) );

            memcpy(cached_2d_terms->chunked_gam1_reals.chunked_elems + num_pages_gams, args[i].cached_2d_terms->chunked_gam1_reals.chunked_elems, argi_num_pages_gams * sizeof(double*) );
            memcpy(cached_2d_terms->chunked_gam1_reals.used_elems_per_page + num_pages_gams, args[i].cached_2d_terms->chunked_gam1_reals.used_elems_per_page, argi_num_pages_gams * sizeof(int) );

            memcpy(cached_2d_terms->chunked_gam2_reals.chunked_elems + num_pages_gams, args[i].cached_2d_terms->chunked_gam2_reals.chunked_elems, argi_num_pages_gams * sizeof(double*) );
            memcpy(cached_2d_terms->chunked_gam2_reals.used_elems_per_page + num_pages_gams, args[i].cached_2d_terms->chunked_gam2_reals.used_elems_per_page, argi_num_pages_gams * sizeof(int) );

            memcpy(cached_2d_terms->chunked_g_vals.chunked_elems + num_pages_gvals, args[i].cached_2d_terms->chunked_g_vals.chunked_elems, argi_num_pages_gvals * sizeof(C_COMPLEX_TYPE*) );
            memcpy(cached_2d_terms->chunked_g_vals.used_elems_per_page + num_pages_gvals, args[i].cached_2d_terms->chunked_g_vals.used_elems_per_page, argi_num_pages_gvals * sizeof(int) );

            num_pages_thetas += argi_num_pages_thetas;
            num_pages_gams += argi_num_pages_gams;
            num_pages_gvals += argi_num_pages_gvals;
            num_terms_total += argi_num_terms_total;
            free(args[i].cached_2d_terms->cached_terms);
            free(args[i].cached_2d_terms->chunked_cos_thetas.chunked_elems);
            free(args[i].cached_2d_terms->chunked_cos_thetas.used_elems_per_page);
            free(args[i].cached_2d_terms->chunked_sin_thetas.chunked_elems);
            free(args[i].cached_2d_terms->chunked_sin_thetas.used_elems_per_page);
            free(args[i].cached_2d_terms->chunked_gam1_reals.chunked_elems);
            free(args[i].cached_2d_terms->chunked_gam1_reals.used_elems_per_page);
            free(args[i].cached_2d_terms->chunked_gam2_reals.chunked_elems);
            free(args[i].cached_2d_terms->chunked_gam2_reals.used_elems_per_page);
            free(args[i].cached_2d_terms->chunked_g_vals.chunked_elems);
            free(args[i].cached_2d_terms->chunked_g_vals.used_elems_per_page);
            free(args[i].cached_2d_terms);
        }
        // THIS IS OKAY
        // Now need to add all the chunked rsys moments together and then finalize the result
        if(get_relative_moments)
        {
            for(int i = 0; i < num_threads; i++)
            {
                *rel_norm_factor += args[i].chunk_rel_norm_factor;
                for(int j = 0; j < 2; j++)
                    rel_cond_mean[j] += args[i].chunk_rel_cond_mean[j];
                for(int j = 0; j < 4; j++)
                    rel_cond_covar[j] += args[i].chunk_rel_cond_covar[j];
            }
        }
        finalize_rel_sys_moments(rel_norm_factor, rel_cond_mean, rel_cond_covar, 2);
        tmr.toc(false);
        if(with_timing)
        {
            printf("Relative CPDF System Caching (%d threads):\n  Caching %llu x %llu = %llu terms took %d ms\n", num_threads, Ntp, Nts, N, tmr.cpu_time_used);
            if(with_term_approx)
                printf("  Removed %d/%llu terms using epsilon=%.3E. Term total=%llu\n", num_terms_approxed_out, N, term_approx_eps, N-num_terms_approxed_out);
        }

        return cached_2d_terms;
    }
}

// Evaluates the "relative-then-transformed" term using the 2D cpdf transform caching structure 
C_COMPLEX_TYPE eval_rel_marg2d_cached_term_for_cpdf(Cached2DCPDFTerm* cached_term, double x1, double x2, const bool FULL_SOLVE)
{
    double sin_t1;
    double cos_t1;
    double sin_t2;
    double cos_t2;
    C_COMPLEX_TYPE gamma1;
    C_COMPLEX_TYPE gamma2;
    C_COMPLEX_TYPE integral_low_lim;
    C_COMPLEX_TYPE integral_high_lim;
    C_COMPLEX_TYPE integral_over_cell;
    const int m = cached_term->m;
    double gam1_imag = cached_term->b[0] - x1;
    double gam2_imag = cached_term->b[1] - x2;
    double* sin_thetas = cached_term->sin_thetas;
    double* cos_thetas = cached_term->cos_thetas;
    double* gam1_reals = cached_term->gam1_reals;
    double* gam2_reals = cached_term->gam2_reals;
    C_COMPLEX_TYPE* g_vals = cached_term->g_vals;
    C_COMPLEX_TYPE term_integral = 0;
    const int cells = (1+FULL_SOLVE)*m;

    // Variables to check which integration form we should use
    // Both methods work fine for gamma2 == 0
    // If gamma1 == 0, then we need to use the slower int method 
    // If gamma1 and gamma2 == 0, assert false and exit 
    const bool check_gamma1 = fabs(gam1_imag) < INTEGRAL_GAMMA_EPS;
    for(int i = 0; i < cells; i++)
    {
        gamma1 = gam1_reals[i] + I*gam1_imag;
        gamma2 = gam2_reals[i] + I*gam2_imag;
        sin_t1 = sin_thetas[i];
        cos_t1 = cos_thetas[i];
        sin_t2 = sin_thetas[i+1];
        cos_t2 = cos_thetas[i+1];
        // Check which method
        bool fast_int_method = true;
        if( check_gamma1 )
        {
            if( fabs(gam1_reals[i]) < INTEGRAL_GAMMA_EPS )
            {
                fast_int_method = false; // Gamma1 is effectively zero
                // Now need to check gamma2 for singularity 
                if( (fabs(gam2_reals[i]) < INTEGRAL_GAMMA_EPS) && (fabs(gam2_imag) < INTEGRAL_GAMMA_EPS) )
                {
                    printf(RED "[Error eval_rel_marg2d_cached_term_for_cpdf:] Possible singularity error, gamma1=%.4E+%.4Ej and gamma2=%.4E+%.4Ej\n Until resolved, exiting! Debug here! Goodbye!" NC "\n", gam1_reals[i], gam1_imag, gam2_reals[i], gam2_imag);
                    exit(1);
                }
            }
        }
        // Faster integral
        if(fast_int_method)
        {
            gamma2 *= gamma1;
            gamma1 *= gamma1;
            integral_low_lim = sin_t1 / (gamma1*cos_t1 + gamma2*sin_t1);
            integral_high_lim = sin_t2 / (gamma1*cos_t2 + gamma2*sin_t2);
            integral_over_cell = integral_high_lim - integral_low_lim;
            integral_over_cell *= g_vals[i];
            term_integral += integral_over_cell;
        }
        // New way that automatically deals with gamma1==0 or gamma2==0 issues
        else
        {
            integral_low_lim = (gamma1 * sin_t1 - gamma2 * cos_t1) / (gamma1 * cos_t1 + gamma2 * sin_t1);
            integral_high_lim = (gamma1 * sin_t2 - gamma2 * cos_t2) / (gamma1 * cos_t2 + gamma2 * sin_t2);
            integral_over_cell = integral_high_lim - integral_low_lim;
            gamma1 *= gamma1;
            gamma2 *= gamma2;
            integral_over_cell *= g_vals[i]/(gamma1 + gamma2);
            term_integral += integral_over_cell;
        }
    }
    return term_integral;
}

struct ThreadRelMargEvalCPDF2D 
{
    int points_per_thread;
    Cached2DCPDFTermContainer* cached_terms;
    CauchyPoint3D* points;
    double norm_factor_prim; 
    double norm_factor_sec; 
    bool FULL_SOLVE;
};

// Evaluates the cpdf for a point [x1, x2] given that the Cached2DCPDFTermContainer has been constructed using get_marg2d_relative_and_transformed_cpdf
C_COMPLEX_TYPE eval_marg2d_relative_and_transformed_cpdf(Cached2DCPDFTermContainer* cached_terms, double x1, double x2, double norm_factor_prim, double norm_factor_sec, const bool FULL_SOLVE)
{
    int num_terms = cached_terms->current_term_idx;
    C_COMPLEX_TYPE unnormalized_m2d_rt_cpdf_val = 0;
    for(int i = 0; i < num_terms; i++)
    {
        C_COMPLEX_TYPE term_component = eval_rel_marg2d_cached_term_for_cpdf(cached_terms->cached_terms + i, x1, x2, FULL_SOLVE);
        unnormalized_m2d_rt_cpdf_val += term_component;
        //printf("Term Component %d: %.4E + %.4Ej\n", i, creal(term_component), cimag(term_component) );
    }
    C_COMPLEX_TYPE normalized_m2d_rt_cpdf_val = (2-FULL_SOLVE)*unnormalized_m2d_rt_cpdf_val * RECIPRICAL_TWO_PI * RECIPRICAL_TWO_PI / (norm_factor_prim * norm_factor_sec);
    return normalized_m2d_rt_cpdf_val;
}

void* call_threaded_eval_marg2d_relative_and_transformed_cpdf(void* args)
{
    ThreadRelMargEvalCPDF2D* tid_arg = (ThreadRelMargEvalCPDF2D*) args;
    int ppt = tid_arg->points_per_thread;
    CauchyPoint3D* points = tid_arg->points;
    Cached2DCPDFTermContainer* cached_terms = tid_arg->cached_terms;
    double norm_factor_prim = tid_arg->norm_factor_prim;
    double norm_factor_sec = tid_arg->norm_factor_sec;
    const bool FULL_SOLVE = tid_arg->FULL_SOLVE;
    for(int i = 0; i < ppt; i++)
        points[i].z = creal( eval_marg2d_relative_and_transformed_cpdf(cached_terms, points[i].x, points[i].y, norm_factor_prim, norm_factor_sec, FULL_SOLVE) );
    return NULL;
}

// Evaluates the cpdf for a grid of points, possibly using multiple threads on the cached cpdf structure 
CauchyPoint3D* grid_eval_marg2d_relative_and_transformed_cpdf(
    Cached2DCPDFTermContainer* cached_terms, 
    double xlow, double xhigh, double delta_x, double ylow, double yhigh, double delta_y, 
    double norm_factor_prim, double norm_factor_sec, 
    int* ret_num_points_x, int* ret_num_points_y, 
    int num_threads, const bool FULL_SOLVE, const bool with_timing)
{
    CPUTimer tmr;
    tmr.tic();
    // Form set of points x
    assert(xhigh > xlow);
    int num_xpoints = (int) ceil( (xhigh - xlow) / delta_x + 1 );
    assert(num_xpoints > 1);
    assert(num_threads > 0);
    double* xs = (double*) malloc(num_xpoints * sizeof(double));
    for(int i = 0; i < num_xpoints; i++)
        xs[i] = xlow + delta_x * i;
    xs[num_xpoints-1] = xhigh;
    *ret_num_points_x = num_xpoints;

    // Form set of points x
    assert(yhigh > ylow);
    int num_ypoints = (int) ceil( (yhigh - ylow) / delta_y + 1 );
    assert(num_ypoints > 1);
    double* ys = (double*) malloc(num_ypoints * sizeof(double));
    for(int i = 0; i < num_ypoints; i++)
        ys[i] = ylow + delta_y * i;
    ys[num_ypoints-1] = yhigh;
    *ret_num_points_y = num_ypoints;
    if( num_threads > (num_xpoints * num_ypoints) )
        num_threads = num_xpoints * num_ypoints;
    
    CauchyPoint3D* points = (CauchyPoint3D*) malloc((num_xpoints * num_ypoints) * sizeof(CauchyPoint3D));
    int grid_points = num_xpoints * num_ypoints;
    for(int i = 0; i < num_xpoints; i++)
    {
        for(int j = 0; j < num_ypoints; j++)
        {
            int count = i*num_ypoints + j;
            points[count].x = xs[i];
            points[count].y = ys[j];
            points[count].z = -1;
        }
    }

    if(num_threads == 1)
    {
        C_COMPLEX_TYPE cpdf_eval;
        for(int i = 0; i < grid_points; i++)
        {
            cpdf_eval = eval_marg2d_relative_and_transformed_cpdf(cached_terms, points[i].x, points[i].y, norm_factor_prim, norm_factor_sec, FULL_SOLVE);
            points[i].z = creal(cpdf_eval); 
            printf("x:%.2lf, y:%.2lf, z:%.4E+%.4Ej\n", points[i].x, points[i].y, creal(cpdf_eval), cimag(cpdf_eval) );
            if(FULL_SOLVE)
            {
                double imag_cpdf = cimag(cpdf_eval);
                if( (fabs(imag_cpdf) > 1e-10) || isnan(imag_cpdf) )
                    printf("We have a large-ish imaginary part: x=%.2E, y=%.2E, cpdf_eval=%.4E+%.4Ej\n", points[i].x, points[i].y, creal(cpdf_eval), cimag(cpdf_eval));
            }
        }
    }
    else 
    {
        int grid_points = num_xpoints * num_ypoints;
        if(grid_points < num_threads)
            num_threads = grid_points;
        int points_per_thread = grid_points / num_threads;
        pthread_t tids[num_threads];
        ThreadRelMargEvalCPDF2D tid_args[num_threads];
        for(int i = 0; i < num_threads; i++)
        {
            int ppt = points_per_thread;
            if(i == (num_threads-1))
                ppt += grid_points % num_threads;
            tid_args[i].points_per_thread = ppt;
            tid_args[i].points = points + i * points_per_thread;
            tid_args[i].cached_terms = cached_terms;
            tid_args[i].norm_factor_prim = norm_factor_prim;
            tid_args[i].norm_factor_sec = norm_factor_sec;
            tid_args[i].FULL_SOLVE = FULL_SOLVE;
            pthread_create(tids + i, NULL, call_threaded_eval_marg2d_relative_and_transformed_cpdf, tid_args + i);
        }
        for(int i = 0; i < num_threads; i++)
            pthread_join(tids[i], NULL);
        //for(int i = 0; i < grid_points; i++)
        //    printf("x:%.2lf, y:%.2lf, z:%.4E\n", points[i].x, points[i].y, points[i].z );
    }
    free(xs);
    free(ys);
    tmr.toc(false);
    if(with_timing)
        printf("Relative CPDF System Evaluation (%d threads):\n  Evaluating %d points each over %d cached terms took %d ms\n", num_threads, num_xpoints * num_ypoints, cached_terms->current_term_idx, tmr.cpu_time_used);
    return points;
}

int log_marg2d_relative_and_transformed_cpdf(char* log_dir, int step, CauchyPoint3D* points, int num_points_x, int num_points_y)
{
    assert( step > 0 );
    FILE* data_file;
    FILE* dims_file;
    // Create path character array
    int len_log_dir = strlen(log_dir);
    char path[len_log_dir + 30];

    sprintf(path, "%s/grid_elems_%d%d.txt", log_dir, 0,1);
    // First log if tag_count == 1
    // Open (possibly overwrite old) grid dims file
    if(step == 1)
        dims_file = fopen(path, "w");
    else 
        dims_file = fopen(path, "a");
    if(dims_file == NULL)
    {
        printf(RED "[ERROR log_marg2d_relative_and_transformed_cpdf:]\n  Could not open grid dim file!\n  Path: %s\n  Please check path and try again! Exiting!" NC "\n", path);
        exit(1);
    }
    fprintf(dims_file, "%d,%d\n", num_points_x, num_points_y);
    
    // write out binary data stream
    sprintf(path, "%s/cpdf_%d%d_%d.bin", log_dir, 0, 1, step);
    data_file = fopen(path, "wb");
    if(data_file == NULL)
    {
        printf(RED "[ERROR log_marg2d_relative_and_transformed_cpdf:]\n  Could not open binary data file!\n  Path: %s\n  Please check path and try again! Exiting!" NC "\n", path);
        exit(1);
    }
    int num_grid_points = num_points_x * num_points_y;
    fwrite(points, sizeof(CauchyPoint3D), num_grid_points, data_file);
    fclose(data_file);
    fclose(dims_file);
    return 0;
}

#endif // _PREDICTION_HPP_