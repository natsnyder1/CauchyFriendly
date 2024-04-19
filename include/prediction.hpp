#ifndef _PREDICTION_HPP_
#define _PREDICTION_HPP_

#include "cauchy_estimator.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cpu_linalg.hpp"
#include <cstring>

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
        
        for(int i = 0; )


        for(int j = 0; j < Nts; j++)
        {

        }
    }
}


#endif // _PREDICTION_HPP_