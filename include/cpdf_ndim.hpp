#ifndef _CPDF_NDIM_HPP_
#define _CPDF_NDIM_HPP_

#include "array_logging.hpp"
#include "cauchy_constants.hpp"
#include "cauchy_estimator.hpp"
#include "cauchy_term.hpp"
#include "cauchy_types.hpp"
#include "cauchy_util.hpp"
#include "cauchy_linalg.hpp"
#include "cpu_timer.hpp"
#include "random_variables.hpp"
#include "term_reduction.hpp"

void* evaluate_2d_marginal_grid_points(void* marg_args);
void* evaluate_1d_marginal_grid_points(void* marg_args);

int sort_dless(const void* p1, const void* p2)
{
    double a = *((double*)p1);
    double b = *((double*)p2);
    if(a > b)
        return 1;
    else if(a < b)
        return -1;
    else 
        return 0;
} 

// Extracts columns of HPA for 2D marginalization process
void marg2d_extract_2D_HPA(double* A_ndim, double* b_ndim, double* A2, double* b2, int m, int d, int marg_idx1, int marg_idx2)
{
    for(int i = 0; i < m; i++)
    {
        A2[i*2 + 0] = A_ndim[i*d + marg_idx1];
        A2[i*2 + 1] = A_ndim[i*d + marg_idx2];
    }
    b2[0] = b_ndim[marg_idx1];
    b2[1] = b_ndim[marg_idx2];
}

int marg2d_remove_zeros_and_coalign(double* work_A, double* work_p, double* work_A2, double* work_p2, int* c_map, int* cs_map, int m, double ZERO_EPSILON, int ZERO_HP_MARKER_VALUE)
{
    bool F[m];
    memset(F, 1, m * sizeof(bool));
    int F_idxs[m];

    // Normalize all HPs in work_A using 1-norm
    // Scale p[i] by the 1-norm of work_A[i,:]
    //checking for zero HPs, and marking these accordingly
    for(int i = 0; i < m; i++)
    {
        double* ai = work_A + i*2;
        double fabs_ai0 = fabs(ai[0]);
        double fabs_ai1 = fabs(ai[1]);
        bool is_zero = (fabs_ai0 < ZERO_EPSILON) && (fabs_ai1 < ZERO_EPSILON);
        // Mark with special zero index of ZERO_HP_MARKER_VALUE in F_Map if this is the case
        if(is_zero)
        {
            c_map[i] = ZERO_HP_MARKER_VALUE;
            cs_map[i] = ZERO_HP_MARKER_VALUE;
            F_idxs[i] = ZERO_HP_MARKER_VALUE;
            F[i] = 0;
            continue;
        }
        else 
        {
            double sum_a = fabs_ai0 + fabs_ai1;
            ai[0] /= sum_a;
            ai[1] /= sum_a;
            work_p[i] *= sum_a;
        }
    }

    // Loop over all non-zero rows of work_A
    int unique_count = 0;
    for(int i = 0; i < m; i++)
    {
        if(F[i])
        {
            c_map[i] = unique_count;
            cs_map[i] = 1;
            F_idxs[i] = i;
            work_p2[unique_count] = work_p[i];
            // Zeros check
            double* ai = work_A + i*2;    
            for(int j = i+1; j < m; j++)
            {
                if(F[j])
                {
                    double* aj = work_A + j*2;
                    // Check for positive coals first
                    bool pos_coal = (fabs(ai[0] - aj[0]) < COALIGN_MU_EPS) && (fabs(ai[1] - aj[1]) < COALIGN_MU_EPS);
                    if(pos_coal)
                    {
                        c_map[j] = unique_count;
                        cs_map[j] = 1;
                        F[j] = 0;
                        F_idxs[j] = i;
                        work_p2[unique_count] += work_p[j];
                        continue;
                    }
                    bool neg_coal = (fabs(ai[0] + aj[0]) < COALIGN_MU_EPS) && (fabs(ai[1] + aj[1]) < COALIGN_MU_EPS);
                    if(neg_coal)
                    {
                        c_map[j] = unique_count;
                        cs_map[j] = -1;
                        F[j] = 0;
                        F_idxs[j] = i;
                        work_p2[unique_count] += work_p[j];
                        continue;
                    }
                }
            }
            unique_count++;
        }
    }
    if(unique_count < m)
    {
        unique_count = 0;
        for(int i = 0; i < m; i++)
        {
            if(F_idxs[i] == i)
            {
                work_A2[unique_count*2+0] = work_A[i*2+0];
                work_A2[unique_count*2+1] = work_A[i*2+1];
                unique_count++;
            }
        }
    }
    else
        memcpy(work_A2, work_A, m * 2 * sizeof(double));

    return unique_count;
} 

// returns the (2*m) angles of the cell walls for all cells encompassing A
void marg2d_get_cell_wall_angles(double* thetas, double* A, const int m)
{
    const int d = 2;
    double point[d]; // point on hyperplane i
    double* a;
    int count = 0;
    for(int i = 0; i < m; i++)
    {
        a = A + i*d;
        if( fabs(a[0]) < fabs(a[1]) )
        {
            point[0] = 1;
            point[1] = -a[0] / a[1];
        }
        else 
        {
            point[0] = - a[1] / a[0];
            point[1] = 1;
        }
        double t1 = atan2(point[1], point[0]);
        if(t1 < 0)
            t1 += PI;
        double t2 = t1 + PI;
        thetas[count++] = t1;
        thetas[count++] = t2;
    }
    // Now sort the thetas, the theta at idx m corresponds to a half rotation around the arrangement
    qsort(thetas, 2*m, sizeof(double), &sort_dless);
}

// Returns half of the SVs ( i.e, m/(2m) ),
// to get all the SVs, we simply flip these
// SVs is size 2*m x m 
// thetas 
void marg2d_get_SVs(double* SVs, double* A, double* thetas, const int m, const bool flip_svs = true)
{
    const int d = 2;
    double point[d];
    double* SV;
    for(int i = 0; i < m; i++)
    {
        double t = (thetas[i+1] + thetas[i]) / 2.0;
        point[0] = cos(t); point[1] = sin(t);
        SV = SVs + i*m;
        for(int j = 0; j < m; j++)
        {
            double* a = A + j*d;
            double sum = a[0] * point[0] + a[1] * point[1];
            SV[j] = sum > 0 ? 1 : -1;
        }
    }

    // Now flip these signs, they now correspond to full arrangement
    if(flip_svs)
    {
        SV = SVs + m*m;
        for(int i = 0; i < m*m; i++)
            SV[i] = -1*SVs[i];
    }
}


struct Cached2DCPDFTerm
{
    double* sin_thetas;
    double* cos_thetas;
    double* gam1_reals;
    double* gam2_reals;
    C_COMPLEX_TYPE* g_vals;
    double b[2];
    int m;
};

struct Cached1DCPDFTerm
{
    double a;
    double b;
    double w;
    double s;
};

struct Cached2DCPDFTermContainer
{
    Cached2DCPDFTerm* cached_terms;
    ChunkedPackedElement<double> chunked_sin_thetas;
    ChunkedPackedElement<double> chunked_cos_thetas;
    ChunkedPackedElement<double> chunked_gam1_reals;
    ChunkedPackedElement<double> chunked_gam2_reals;
    ChunkedPackedElement<C_COMPLEX_TYPE> chunked_g_vals;
    int current_term_idx;
    int terms_alloc_total;
    bool full_solve;


    void init(int pages_at_start = 0, bool _full_solve = false)
    {
        chunked_sin_thetas.init(pages_at_start, CP_STORAGE_PAGE_SIZE);
        chunked_cos_thetas.init(pages_at_start, CP_STORAGE_PAGE_SIZE);
        chunked_gam1_reals.init(pages_at_start, CP_STORAGE_PAGE_SIZE);
        chunked_gam2_reals.init(pages_at_start, CP_STORAGE_PAGE_SIZE);
        chunked_g_vals.init(pages_at_start, CP_STORAGE_PAGE_SIZE);
        cached_terms = (Cached2DCPDFTerm*) malloc(0);
        current_term_idx = 0;
        terms_alloc_total = 0;
        full_solve = _full_solve;
    }

    void extend_storage(int* terms_per_shape, int shape_range)
    {
        BYTE_COUNT_TYPE theta_bytes = 0;
        BYTE_COUNT_TYPE gam_bytes = 0;
        BYTE_COUNT_TYPE terms_total = 0;
        for(int m = 1; m < shape_range; m++)
        {
            BYTE_COUNT_TYPE Nt_shape = terms_per_shape[m];
            terms_total += Nt_shape;
            theta_bytes += Nt_shape * (m+1);
            gam_bytes += Nt_shape * m;
        }
        terms_alloc_total = terms_total;
        cached_terms = (Cached2DCPDFTerm*)realloc(cached_terms, terms_total * sizeof(Cached2DCPDFTerm));
        chunked_sin_thetas.extend_elems(theta_bytes * sizeof(double));
        chunked_cos_thetas.extend_elems(theta_bytes * sizeof(double));
        chunked_gam1_reals.extend_elems(gam_bytes * sizeof(double));
        chunked_gam2_reals.extend_elems(gam_bytes * sizeof(double));
        chunked_g_vals.extend_elems(gam_bytes * sizeof(C_COMPLEX_TYPE));
        current_term_idx = 0;
    }
    
    void extend_storage_by(BYTE_COUNT_TYPE num_terms, int m)
    {
        BYTE_COUNT_TYPE theta_bytes = num_terms * ( (1+full_solve) * m + 1 );
        BYTE_COUNT_TYPE gam_bytes = num_terms * (1+full_solve) * m;
        terms_alloc_total += num_terms;
        cached_terms = (Cached2DCPDFTerm*)realloc(cached_terms, terms_alloc_total * sizeof(Cached2DCPDFTerm));
        chunked_sin_thetas.extend_elems(theta_bytes * sizeof(double));
        chunked_cos_thetas.extend_elems(theta_bytes * sizeof(double));
        chunked_gam1_reals.extend_elems(gam_bytes * sizeof(double));
        chunked_gam2_reals.extend_elems(gam_bytes * sizeof(double));
        chunked_g_vals.extend_elems(gam_bytes * sizeof(C_COMPLEX_TYPE));
        //current_term_idx = 0;
    }

    void reset_page_idxs()
    {
        chunked_sin_thetas.reset_page_idxs();
        chunked_cos_thetas.reset_page_idxs();
        chunked_gam1_reals.reset_page_idxs();
        chunked_gam2_reals.reset_page_idxs();
        chunked_g_vals.reset_page_idxs();
        current_term_idx = 0;
    }

    void reset()
    {
        chunked_sin_thetas.reset();
        chunked_cos_thetas.reset();
        chunked_gam1_reals.reset();
        chunked_gam2_reals.reset();
        chunked_g_vals.reset();
        current_term_idx = 0;
    }

    void deinit()
    {
        chunked_sin_thetas.deinit();
        chunked_cos_thetas.deinit();
        chunked_gam1_reals.deinit();
        chunked_gam2_reals.deinit();
        chunked_g_vals.deinit();
        current_term_idx = 0;
        free(cached_terms);
    }

    void set_term_ptrs(BYTE_COUNT_TYPE m)
    {
        Cached2DCPDFTerm* cached_term = cached_terms + current_term_idx;
        int cells_to_visit = (1+full_solve)*m;
        chunked_sin_thetas.set_elem_ptr(&(cached_term->sin_thetas), cells_to_visit+1);
        chunked_cos_thetas.set_elem_ptr(&(cached_term->cos_thetas), cells_to_visit+1);
        chunked_gam1_reals.set_elem_ptr(&(cached_term->gam1_reals), cells_to_visit);
        chunked_gam2_reals.set_elem_ptr(&(cached_term->gam2_reals), cells_to_visit);
        chunked_g_vals.set_elem_ptr(&(cached_term->g_vals), cells_to_visit);
    }
    void incr_cached_term_idx()
    {
        current_term_idx++;
    }

    // can call to remove last term 
    void pop_term_ptrs()
    {
        if(current_term_idx > 0)
        {
            Cached2DCPDFTerm* cached_term = cached_terms + (current_term_idx-1);
            int cells_to_visit = (1+full_solve)*cached_term->m;
            chunked_sin_thetas.pop_elem_ptr(cells_to_visit+1);
            chunked_cos_thetas.pop_elem_ptr(cells_to_visit+1);
            chunked_gam1_reals.pop_elem_ptr(cells_to_visit);
            chunked_gam2_reals.pop_elem_ptr(cells_to_visit);
            chunked_g_vals.pop_elem_ptr(cells_to_visit);
            current_term_idx -= 1;
        }
    }
};

struct PointWiseNDimCauchyCPDF
{
    ChildTermWorkSpace lowered_children_workspace; // Used to store the terms temporarily during TP/TPC/MU/MUC
    CoalignmentElemStorage coalign_store; // Memory manager for storing terms after Coalignment
    ReductionElemStorage reduce_store; // Memory manager for storing terms after Term Reduction
    ChunkedPackedTableStorage gb_tables; // Memory manager for storing g and b-tables
    CauchyEstimator* cauchyEst;
    double ZERO_EPSILON;
    int ZERO_HP_MARKER_VALUE;
    double* bar_nu;
    // Special structure for caching 2D CPDF / Marginal 2D CPDF
    Cached2DCPDFTermContainer cached_2d_terms;
    int master_step_of_cached_2d_terms;
    int marg_idxs_of_cached_2d_terms[2];
    // Special structure for caching 1D CPDF / Marginal 1D CPDF
    Cached1DCPDFTerm* cached_1d_terms;
    int master_step_of_cached_1d_terms;
    int marg_idx_of_cached_1d_terms;

    PointWiseNDimCauchyCPDF(CauchyEstimator* _cauchyEst)
    {
        ZERO_EPSILON = MU_EPS;
        ZERO_HP_MARKER_VALUE = 32; // currently the code cannot support more than 32 HPs in a HPA. So, this index can signify a unique event.
        cauchyEst = _cauchyEst;
        lowered_children_workspace.init(cauchyEst->shape_range-1, cauchyEst->d);
        coalign_store.init(1, CP_STORAGE_PAGE_SIZE);
        gb_tables.init(1, CP_STORAGE_PAGE_SIZE);
        reduce_store.init(1, CP_STORAGE_PAGE_SIZE);
        cached_2d_terms.init(0);
        master_step_of_cached_2d_terms = -1;
        marg_idxs_of_cached_2d_terms[0] = -1; 
        marg_idxs_of_cached_2d_terms[1] = -1; 
        cached_1d_terms = NULL;
        master_step_of_cached_1d_terms = -1;
        marg_idx_of_cached_1d_terms = -1;

        // Draw random nu_bar for case when marginal coefficients A[i, marg_idx] i=[1,...,m] is zero:
        bar_nu = (double*) malloc(cauchyEst->d * sizeof(double));
        null_ptr_check(bar_nu);
        for(int i = 0; i < cauchyEst->d; i++)
            bar_nu[i] = 2*random_uniform_open();
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
                reorient_parent_A[i*d + j] = parent->A[i*d + j_reindex];
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
        coalign_store.reset_page_idxs();
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
        
        // This code currently does not support when m (hyperplanes) < d (dimension), throw error if so
        for(int i = 1; i < cauchyEst->d; i++)
        {
            if(cauchyEst->terms_per_shape[i] > 0)
            {
                printf(RED "[Error Eval Marginal CPDF:] Estimator has HPAs of(mxd) = (%d x %d)...The # HPS < d...\n"
                           "This needs to be coded for integrate then evaluate method still...Exiting for now. Please fix!" 
                           NC "\n", i, cauchyEst->d);
                exit(1);
            }
        }

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
        coalign_store.reset_page_idxs();
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

    C_COMPLEX_TYPE evaluate_1D_marginal_cpdf(double* xk_marginal, int* marg_state_idxs, int num_marg_state_idxs, bool with_timing_print = false, bool with_caching = false)
    {
        // For now, no fancy stuff, if the cauchy has an issue do not evaluate
        //assert(cauchyEst->numeric_moment_errors == 0);
        // If cauchy has not been run a step, cant evaluate.
        assert(num_marg_state_idxs == 1);
        assert(cauchyEst->master_step > 0);

        bool setup_cache = false;
        bool run_with_cache = false;
        if(with_caching)
        {
            bool recache_condition_master_step = (master_step_of_cached_1d_terms != cauchyEst->master_step);
            bool recache_condition_marg_idx1 = (marg_idx_of_cached_1d_terms != marg_state_idxs[0]);
            if(recache_condition_master_step || recache_condition_marg_idx1)
            {
                setup_cache = true;
                master_step_of_cached_1d_terms = cauchyEst->master_step;
                marg_idx_of_cached_1d_terms = marg_state_idxs[0];
                cached_1d_terms = (Cached1DCPDFTerm*) realloc(cached_1d_terms, cauchyEst->Nt * sizeof(Cached1DCPDFTerm));
                null_ptr_check(cached_1d_terms);
            }
            else
                run_with_cache = true;
        }

        CPUTimer tmr;
        tmr.tic();
        int marg_idx = marg_state_idxs[0];
        double x1 = xk_marginal[0];
        double norm_factor = creal(cauchyEst->fz);
        double fx_unnormalized = 0;

        // Cached run
        if(run_with_cache)
        {
            int Nt_total = cauchyEst->Nt;
            for(int i = 0; i < Nt_total; i++)
            {   
                Cached1DCPDFTerm* cached_term = cached_1d_terms + i;
                double a = cached_term->a;
                double b = cached_term->b;
                double w2 = cached_term->w*cached_term->w;
                double x1ms = x1 - cached_term->s;
                x1ms *= x1ms;
                double term_integral = (a*x1 + b) / (w2 + x1ms);
                fx_unnormalized += term_integral;
            }
        }
        else 
        {
            int shape_range = find_shape_range_lim(cauchyEst->terms_per_shape, cauchyEst->shape_range);
            int d = cauchyEst->d;
            int* terms_per_shape = cauchyEst->terms_per_shape; 
            int count = 0;
            for(int m = 1; m < shape_range; m++)
            {
                int Nt_shape = terms_per_shape[m];
                if(Nt_shape > 0)
                {
                    CauchyTerm* parents = cauchyEst->terms_dp[m];
                    int two_to_m_minus1 = (1<<(m-1));
                    int rev_m_mask = (1<<m) - 1;
                    for(int i = 0; i < Nt_shape; i++)
                    {
                        CauchyTerm* parent = parents + i;
                        double b_c = parent->b[marg_idx] - xk_marginal[0];
                        double p_cc = 0; // p for the "cut" then coaligned hyperplane arrangement
                        int enc_sv_num_rhs = 0;
                        int enc_sv_num_lhs = 0;
                        for(int j = 0; j < m; j++)
                        {
                            double A_cj = parent->A[j*d + marg_idx];
                            double fabs_A_cj = fabs(A_cj);
                            p_cc += parent->p[j] * fabs_A_cj;
                            if(fabs_A_cj > 1e-15)
                            {
                                if(A_cj > 0)
                                    enc_sv_num_lhs |= (1 << j);
                                else
                                    enc_sv_num_rhs |= (1 << j);
                            }
                            else 
                            {
                                double in_prod = dot_prod(parent->A + j*d, bar_nu, d);
                                if(in_prod > 0)
                                    enc_sv_num_lhs |= (1 << j);
                                else
                                    enc_sv_num_rhs |= (1 << j);
                            }
                        }
                        C_COMPLEX_TYPE g_num_lhs = lookup_g_numerator(enc_sv_num_lhs, two_to_m_minus1, rev_m_mask, parent->gtable_p, parent->cells_gtable_p * GTABLE_SIZE_MULTIPLIER, true);
                        C_COMPLEX_TYPE g_num_rhs = lookup_g_numerator(enc_sv_num_rhs, two_to_m_minus1, rev_m_mask, parent->gtable_p, parent->cells_gtable_p * GTABLE_SIZE_MULTIPLIER, true);
                        C_COMPLEX_TYPE g_val = g_num_lhs / (I*b_c + p_cc) - g_num_rhs / (I*b_c - p_cc);
                        fx_unnormalized += creal(g_val);
                        if(setup_cache)
                        {
                            Cached1DCPDFTerm* cached_term = cached_1d_terms + count;
                            cached_term->s = parent->b[marg_idx];
                            cached_term->w = p_cc;
                            double c_kk = creal(g_num_rhs);
                            double d_kk = cimag(g_num_rhs);
                            cached_term->a = d_kk / PI;
                            cached_term->b = (c_kk*cached_term->w - d_kk*cached_term->s) / PI;
                            count++;
                        }
                    }
                }
            }
        }
        tmr.toc(false);
        if(with_timing_print)
            printf("Marginal 1D CPDF for state %d took %d ms to process %d terms\n", marg_idx, tmr.cpu_time_used, cauchyEst->Nt);
        
        C_COMPLEX_TYPE fx_normalized;
        if(run_with_cache) 
            fx_normalized = fx_unnormalized / norm_factor;
        else
            fx_normalized = fx_unnormalized * RECIPRICAL_TWO_PI / norm_factor;
        return fx_normalized;
    }

    C_COMPLEX_TYPE evaluate_2D_marginal_cpdf(double* xk_marginal, int* marg_state_idxs, int num_marg_state_idxs, bool with_timing_print = false, bool with_caching = false)
    {
        assert(num_marg_state_idxs == 2);
        assert(cauchyEst->master_step > 0);
        //assert(cauchyEst->numeric_moment_errors == 0);

        const int shape_range = find_shape_range_lim(cauchyEst->terms_per_shape, cauchyEst->shape_range);
        int* terms_per_shape = cauchyEst->terms_per_shape;
        int d = cauchyEst->d;
        bool setup_cache = false;
        bool run_with_cache = false;

        if(with_caching)
        {
            bool recache_condition_master_step = (master_step_of_cached_2d_terms != cauchyEst->master_step);
            bool recache_condition_marg_idx1 = (marg_idxs_of_cached_2d_terms[0] != marg_state_idxs[0]);
            bool recache_condition_marg_idx2 = (marg_idxs_of_cached_2d_terms[1] != marg_state_idxs[1]);
            if(recache_condition_master_step || recache_condition_marg_idx1 || recache_condition_marg_idx2)
            {
                setup_cache = true;
                master_step_of_cached_2d_terms = cauchyEst->master_step;
                marg_idxs_of_cached_2d_terms[0] = marg_state_idxs[0];
                marg_idxs_of_cached_2d_terms[1] = marg_state_idxs[1];
                cached_2d_terms.reset_page_idxs();
                cached_2d_terms.extend_storage(terms_per_shape, shape_range);
            }
            else
                run_with_cache = true;
        }

        CPUTimer tmr;
        tmr.tic();
        int marg_idx1 = marg_state_idxs[0];
        int marg_idx2 = marg_state_idxs[1];
        double x1 = xk_marginal[0];
        double x2 = xk_marginal[1];
        double norm_factor = creal(cauchyEst->fz);
        double fx_unnormalized = 0;
        // Cached run
        if(run_with_cache)
        {
            int Nt_total = cauchyEst->Nt;
            for(int i = 0; i < Nt_total; i++)
            {    
                double term_integral = marg2d_cached_eval_term_for_cpdf(cached_2d_terms.cached_terms + i, x1, x2);
                fx_unnormalized += term_integral;
            }
        }
        // Regular run, possibly with caching setup
        else 
        {
            int c_map[shape_range-1]; // coalignment index + zero map
            int cs_map[shape_range-1]; // coalignment sign orientation + zero map
            double work_A[(shape_range-1) * 2];
            double work_A2[(shape_range-1) * 2];
            double work_p[shape_range-1];
            double work_p2[shape_range-1];
            double b[2];
            for(int m = 1; m < shape_range; m++)
            {
                CauchyTerm* terms = cauchyEst->terms_dp[m];
                int Nt_shape = terms_per_shape[m];
                for(int i = 0; i < Nt_shape; i++)
                {
                    CauchyTerm* term = terms + i;
                    // Extract marginal idx columns
                    memcpy(work_p, term->p, m * sizeof(double));
                    marg2d_extract_2D_HPA(term->A, term->b, work_A, b, m, d, marg_idx1, marg_idx2);
                    // Remove zero rows, keeping list of where these occur
                    int m_new = marg2d_remove_zeros_and_coalign(work_A, work_p, work_A2, work_p2, c_map, cs_map, m, ZERO_EPSILON, ZERO_HP_MARKER_VALUE);
                    // Now this term can be evaluated using the (modified) 2D cpdf routine
                    // If we have coalignments and/or zero HPs, we need to use the modified version of the 2D evaluation 
                    double term_integral;
                    if(m_new == m)
                        term_integral = marg2d_eval_term_for_cpdf(x1, x2, work_A2, work_p2, b, m_new, NULL, NULL, term->gtable_p, term->cells_gtable_p * GTABLE_SIZE_MULTIPLIER, NULL, m, -1, setup_cache);
                    // If there are no coalignments or zero HPs, we can procced with the original version of the 2D evaluation
                    else
                        term_integral = marg2d_eval_term_for_cpdf(x1, x2, work_A2, work_p2, b, m_new, c_map, cs_map, term->gtable_p, term->cells_gtable_p * GTABLE_SIZE_MULTIPLIER, term->A, m, d, setup_cache);
                    fx_unnormalized += term_integral;
                }
            }
        }

        tmr.toc(false);
        if(with_timing_print)
            printf("Marginal 2D CPDF for states (%d,%d) took %d ms to process %d terms!\n", marg_idx1, marg_idx2, tmr.cpu_time_used, cauchyEst->Nt);
        
        double fx_normalized = 2 * fx_unnormalized * RECIPRICAL_TWO_PI * RECIPRICAL_TWO_PI / norm_factor;
        return fx_normalized + 0*I;
    }

    void reset_2D_marginal_cpdf()
    {
        master_step_of_cached_2d_terms = -1;
        marg_idxs_of_cached_2d_terms[0] = -1;
        marg_idxs_of_cached_2d_terms[1] = -1;
    }
    /*
    C_COMPLEX_TYPE evaluate_ND_marginal_cpdf(double* xk_marginal, int* marg_state_idxs, int num_marg_state_idxs, bool with_timing_print = false, bool with_caching = false)
    {
        // Needs to be implemented
        assert(false);
        return 0;
    }
    */

    // This routine evaluates a 2D hyperplane arrangement for its contribution to the marginal cpdf
    // Can be used for marginalization of a Ndim arrangement down to two dimensions, or to for a 2D arrangement itself
    double marg2d_eval_term_for_cpdf(
        double x1, double x2, 
        double* A, double* p, double* b, const int m,
        int* c_map, int* cs_map, 
        GTABLE gtable_parent, const int gtable_parent_size, 
        double* A_parent,
        const int m_parent, const int state_dim, const bool setup_cache)
    {
        // Values for constructing sign vectors of the parent g-table
        const int two_to_m_parent_minus1 = 1 << (m_parent-1);
        const int two_to_m_parent = (1<<m_parent);
        const int rev_m_parent_mask = two_to_m_parent - 1;
        // Variables for running 2D cpdf formula
        const int d = 2;
        const int two_m = 2*m;
        int enc_sv;
        double thetas[two_m];
        double SVs[m*m];
        double* SV;
        double A_scaled[m*d];
        double* a;
        marg2d_get_cell_wall_angles(thetas, A, m); // angles of cell walls
        marg2d_get_SVs(SVs, A, thetas, m, false); // SVs corresponding to cells within the above (sequential) cell walls
        for(int i = 0; i < m; i++)
            for(int j = 0; j < d; j++)
                A_scaled[i*d + j] = A[i*d + j] * p[i];

        Cached2DCPDFTerm* cached_term;
        if(setup_cache)
        {
            cached_2d_terms.set_term_ptrs(m);
            cached_term = cached_2d_terms.cached_terms + cached_2d_terms.current_term_idx;
            cached_term->b[0] = b[0];
            cached_term->b[1] = b[1];
            cached_term->m = m;
        }
        double gam1_real;
        double gam2_real;
        double gam1_imag = b[0] - x1;
        double gam2_imag = b[1] - x2;
        C_COMPLEX_TYPE g_val;
        C_COMPLEX_TYPE gamma1;
        C_COMPLEX_TYPE gamma2;
        C_COMPLEX_TYPE integral_low_lim;
        C_COMPLEX_TYPE integral_high_lim;
        C_COMPLEX_TYPE integral_over_cell;
        double term_integral = 0;
        double theta1;
        double theta2;
        double sin_t1;
        double cos_t1;
        double sin_t2;
        double cos_t2;
        // Variables to check which integration form we should use
        // Both methods work fine for gamma2 == 0
        // If gamma1 == 0, then we need to use the slower int method 
        // If gamma1 and gamma2 == 0, assert false and exit 
        const bool check_gamma1 = fabs(gam1_imag) < INTEGRAL_GAMMA_EPS;
        
        for(int i = 0; i < m; i++)
        {

            // 1.) Encode SV and extract G in the i-th cell
            SV = SVs + i*m;
            enc_sv = 0;
            if(c_map == NULL)
            {
                for(int j = 0; j < m; j++)
                    if( SV[j] < 0 )
                        enc_sv |= (1 << j);
            }
            else 
            {
                for(int j = 0; j < m_parent; j++)
                {
                    if(c_map[j] == ZERO_HP_MARKER_VALUE)
                    {
                        double* ap_j = A_parent + j*state_dim;
                        double in_prod = dot_prod(ap_j, bar_nu, state_dim);
                        if( in_prod < 0 )
                            enc_sv |= (1<<j);
                    }
                    else 
                    {
                        int b_idx = c_map[j];
                        int flip = cs_map[j];
                        int s = SV[b_idx] * flip;
                        if(s < 0)
                            enc_sv |= (1<<j);
                    }
                }
            }
            // Overloading numerator lookup function to lookup gtable Gs (just replace "phc" definitions with "m" definitions, which is done here)
            g_val = lookup_g_numerator(enc_sv, two_to_m_parent_minus1, rev_m_parent_mask, gtable_parent, gtable_parent_size, true);
            
            // 2.) Evaluate the real part of gamma1 parameter and real part of gamma2 parameter
            gam1_real = 0;
            gam2_real = 0;
            for(int j = 0; j < m; j++)
            {
                a = A_scaled + j*d;
                gam1_real -= a[0] * SV[j];
                gam2_real -= a[1] * SV[j];
            }
            theta1 = thetas[i];
            theta2 = thetas[i+1];
            sin_t1 = sin(theta1);
            cos_t1 = cos(theta1);
            sin_t2 = sin(theta2);
            cos_t2 = cos(theta2);
            gamma1 = gam1_real + I*gam1_imag;
            gamma2 = gam2_real + I*gam2_imag;
            if(setup_cache)
            {
                cached_term->sin_thetas[i] = sin_t1;
                cached_term->cos_thetas[i] = cos_t1;
                cached_term->gam1_reals[i] = gam1_real;
                cached_term->gam2_reals[i] = gam2_real;
                cached_term->g_vals[i] = g_val;
            }

            // Evaluating the piece-wise integral within this cell
            bool fast_int_method = true;
            if( check_gamma1 )
            {
                if( fabs(gam1_real) < INTEGRAL_GAMMA_EPS )
                {
                    fast_int_method = false; // Gamma1 is effectively zero
                    // Now need to check gamma2 for singularity 
                    if( (fabs(gam2_real) < INTEGRAL_GAMMA_EPS) && (fabs(gam2_imag) < INTEGRAL_GAMMA_EPS) )
                    {
                        printf(RED "[Error marg2d_eval_term_for_cpdf:] Possible singularity error, gamma1=%.4E+%.4Ej and gamma2=%.4E+%.4Ej\n Until resolved, exiting! Debug here! Goodbye!" NC"\n", gam1_real, gam1_imag, gam2_real, gam2_imag);
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
                integral_over_cell *= g_val;
                term_integral += creal(integral_over_cell);
            }
            // New way that automatically deals with gamma1==0 or gamma2==0 issues
            else
            {
                integral_low_lim = (gamma1 * sin_t1 - gamma2 * cos_t1) / (gamma1 * cos_t1 + gamma2 * sin_t1);
                integral_high_lim = (gamma1 * sin_t2 - gamma2 * cos_t2) / (gamma1 * cos_t2 + gamma2 * sin_t2);
                integral_over_cell = integral_high_lim - integral_low_lim;
                gamma1 *= gamma1;
                gamma2 *= gamma2;
                integral_over_cell *= g_val / (gamma1 + gamma2);
                term_integral += creal(integral_over_cell);
            }
        }
        if(setup_cache)
        {
            cached_term->sin_thetas[m] = sin_t2;
            cached_term->cos_thetas[m] = cos_t2;
            cached_2d_terms.incr_cached_term_idx();
        }
        return term_integral;
    }

    double marg2d_cached_eval_term_for_cpdf(Cached2DCPDFTerm* cached_term, double x1, double x2)
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
        double term_integral = 0;

        // Variables to check which integration form we should use
        // Both methods work fine for gamma2 == 0
        // If gamma1 == 0, then we need to use the slower int method 
        // If gamma1 and gamma2 == 0, assert false and exit 
        const bool check_gamma1 = fabs(gam1_imag) < INTEGRAL_GAMMA_EPS;

        for(int i = 0; i < m; i++)
        {
            sin_t1 = sin_thetas[i];
            cos_t1 = cos_thetas[i];
            sin_t2 = sin_thetas[i+1];
            cos_t2 = cos_thetas[i+1];
            gamma1 = gam1_reals[i] + I*gam1_imag;
            gamma2 = gam2_reals[i] + I*gam2_imag;

            // Evaluating the piece-wise integral within this cell
            bool fast_int_method = true;
            if( check_gamma1 )
            {
                if( fabs(gam1_reals[i]) < INTEGRAL_GAMMA_EPS )
                {
                    fast_int_method = false; // Gamma1 is effectively zero
                    // Now need to check gamma2 for singularity 
                    if( (fabs(gam2_reals[i]) < INTEGRAL_GAMMA_EPS) && (fabs(gam2_imag) < INTEGRAL_GAMMA_EPS) )
                    {
                        printf(RED "[Error marg2d_cached_eval_term_for_cpdf:] Possible singularity error, gamma1=%.4E+%.4Ej and gamma2=%.4E+%.4Ej\n Until resolved, exiting! Debug here! Goodbye!" NC"\n", gam1_reals[i], gam1_imag, gam2_reals[i], gam2_imag);
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
                term_integral += creal(integral_over_cell);
            }
            // New way that automatically deals with gamma1==0 or gamma2==0 issues
            else
            {
                integral_low_lim = (gamma1 * sin_t1 - gamma2 * cos_t1) / (gamma1 * cos_t1 + gamma2 * sin_t1);
                integral_high_lim = (gamma1 * sin_t2 - gamma2 * cos_t2) / (gamma1 * cos_t2 + gamma2 * sin_t2);
                integral_over_cell = integral_high_lim - integral_low_lim;
                gamma1 *= gamma1;
                gamma2 *= gamma2;
                integral_over_cell *= g_vals[i] / (gamma1 + gamma2);
                term_integral += creal(integral_over_cell);
            }

            /* 
            // Old Code
            gamma2 *= gamma1;
            gamma1 *= gamma1;
            integral_low_lim = sin_t1 / (gamma1*cos_t1 + gamma2*sin_t1);
            integral_high_lim = sin_t2 / (gamma1*cos_t2 + gamma2*sin_t2);
            integral_over_cell = integral_high_lim - integral_low_lim;
            integral_over_cell *= g_vals[i];
            term_integral += creal(integral_over_cell);
            */
        }
        return term_integral;
    }

    ~PointWiseNDimCauchyCPDF()
    {
        cauchyEst = NULL;
        coalign_store.deinit();
        reduce_store.deinit();
        gb_tables.deinit();
        lowered_children_workspace.deinit();
        free(bar_nu);
        cached_2d_terms.deinit();
        if(cached_1d_terms != NULL)
            free(cached_1d_terms);
    }

};

struct ThreadEvalCPDF2D
{
    PointWiseNDimCauchyCPDF* cpdf;
    CauchyPoint3D* points;
    int num_grid_points;
    int marg_idx1;
    int marg_idx2;
};

struct CauchyCPDFGridDispatcher2D
{
    PointWiseNDimCauchyCPDF* cpdf;
    CauchyPoint3D* points;
    int num_grid_points;
    int num_points_x;
    int num_points_y;
    // Logging of 25 marginals supported, can make this more robust later
    char* log_dir;
    int tags[25*2]; 
    int tag_counts[25]; 
    int num_tags;

    CauchyCPDFGridDispatcher2D(
        PointWiseNDimCauchyCPDF* _cpdf, 
        double grid_low_x,
        double grid_high_x,
        double grid_res_x,
        double grid_low_y,
        double grid_high_y,
        double grid_res_y,
        char* _log_dir = NULL)
    {
        points = NULL;
        reset_grid(grid_low_x, grid_high_x, grid_res_x, grid_low_y, grid_high_y, grid_res_y);
        cpdf = _cpdf;
        if(_log_dir != NULL)
        {
            int len_logdir = strlen(_log_dir);
            log_dir = (char*) malloc( (strlen(_log_dir)+1) * sizeof(char));
            strcpy(log_dir, _log_dir);
            if(log_dir[len_logdir-1] == '/')
            {
                log_dir[len_logdir-1] = '\0';
            }
            check_dir_and_create(log_dir);
        }
        else 
            log_dir = NULL;
        num_tags = 0;
    }

    void reset_grid(
        double grid_low_x, double grid_high_x, double grid_res_x,
        double grid_low_y, double grid_high_y, double grid_res_y)
    {
        assert(grid_high_x > grid_low_x);
        assert(grid_high_y > grid_low_y);
        assert(grid_res_x > 0);
        assert(grid_res_y > 0);

        num_points_x = (int) ( (grid_high_x - grid_low_x + grid_res_x - 1e-15) / grid_res_x ) + 1;
        num_points_y = (int) ( (grid_high_y - grid_low_y + grid_res_y - 1e-15) / grid_res_y ) + 1;
        num_grid_points = num_points_x * num_points_y;
        points = (CauchyPoint3D*) realloc(points, num_grid_points * sizeof(CauchyPoint3D));
        null_ptr_check(points);
        for(int i = 0; i < num_points_y; i++)
        {
            double grid_point_y = (grid_low_y + i * grid_res_y);
            if(grid_point_y > grid_high_y)
                grid_point_y = grid_high_y;
            for(int j = 0; j < num_points_x; j++)
            {
                double grid_point_x = (grid_low_x + j * grid_res_x);
                if(grid_point_x > grid_high_x)
                    grid_point_x = grid_high_x;
                
                int idx = i*num_points_x + j; 
                points[idx].x = grid_point_x;
                points[idx].y = grid_point_y;
                points[idx].z = -1;
            }
        }
    }

    // Grid points on x correspond to marg_idx1, and grid points on y correspond to marg_idx2
    int evaluate_point_grid(int marg_idx1, int marg_idx2, int num_threads, bool with_timing = false)
    {
        assert(marg_idx1 < marg_idx2);
        assert(marg_idx2 < cpdf->cauchyEst->d);
        assert(marg_idx1 > -1);
        assert(marg_idx2 > 0);
        assert(num_threads > 0);
        if( (cpdf->cauchyEst->master_step == cpdf->cauchyEst->num_estimation_steps) && (SKIP_LAST_STEP == true) )
        {
            printf(YEL "[WARN CauchyCPDFGridDispatcher2D:] Cannot evaluate cauchy estimator cpdf for the last step since SKIP_LAST_STEP == true! (The G Tables were not created, as they were skipped!)" NC "\n");
            return 1;
        }
        if(num_grid_points < num_threads)
            num_threads = 1; //num_grid_points-1;
        // Evaluate with first point and setup caching
        CPUTimer tmr;
        tmr.tic();
        int marg_state_idxs[2] = {marg_idx1, marg_idx2};
        double point[2] = {points[0].x, points[0].y};
        points[0].z = creal( cpdf->evaluate_2D_marginal_cpdf(point, marg_state_idxs, 2, false, true) );
        tmr.toc(false);
        int cache_time = tmr.cpu_time_used;
        if(with_timing)
            printf("2D Grid Eval Step %d:\n  Caching took: %d ms (%d CF terms)\n", cpdf->cauchyEst->master_step, tmr.cpu_time_used, cpdf->cauchyEst->Nt);
        tmr.tic();
        if(num_threads < 2)
        {
            for(int i = 1; i < num_grid_points; i++)
            {
                point[0] = points[i].x; point[1] = points[i].y; 
                points[i].z = creal( cpdf->evaluate_2D_marginal_cpdf(point, marg_state_idxs, 2, false, true) );
                //printf("Point %d: x=%.2lf, y=%.2lf, z=%.4E\n", i, points[i].x, points[i].y, points[i].z);
                if( points[i].z < 0 )
                    printf(YEL"[WARN evaluate_point_grid:]" NC " Negative CPDF Value of %.4E at x=%.3E, y=%.3E\n", points[i].z, points[i].x, points[i].y);
            }
        }
        else 
        {
            // Evaluate points 2 to num_grid_points using threaded and cached structure
            pthread_t tids[num_threads];
            ThreadEvalCPDF2D args[num_threads];
            int points_per_thread = (num_grid_points - 1) / num_threads;
            for(int i = 0; i < num_threads; i++)
            {
                args[i].cpdf = cpdf;
                args[i].marg_idx1 = marg_idx1;
                args[i].marg_idx2 = marg_idx2;
                args[i].points = points + 1 + i * points_per_thread;
                args[i].num_grid_points = points_per_thread;
                if( i == (num_threads-1) )
                    args[i].num_grid_points += (num_grid_points - 1) % num_threads;
                pthread_create(tids + i, NULL, evaluate_2d_marginal_grid_points, args+i);
            }
            for(int i = 0; i < num_threads; i++)
                pthread_join(tids[i], NULL);
        }
        tmr.toc(false);
        int compute_time = tmr.cpu_time_used;
        if(with_timing)
        {
            printf("  Computing %d gridpoints took: %d ms (used %d threads)\n", num_grid_points, tmr.cpu_time_used,num_threads);
            printf("  Total Time: %d ms\n", compute_time + cache_time);
        }
        return 0;
    }

    // File binary data is logged to: {log_dir}/cpdf_{marg_idx1}{marg_idx2}_{log_count}
    // File binary data dimensions is logged to {log_dir}/grid_elems_{marg_idx1}{marg_idx2}.txt
    // File row format for binary data dimensions: {num_points_x},{num_points_y}
    int log_point_grid()
    {
        FILE* data_file;
        FILE* dims_file;
        if(log_dir == NULL)
        {
            printf(YEL "[WARN CauchyCPDFGridDispatcher2D:]\n  Cannot Log! The log directory was not set!" NC "\n");
            return 1;
        }
        if( (cpdf->cauchyEst->master_step == cpdf->cauchyEst->num_estimation_steps) && (SKIP_LAST_STEP == true) )
        {
            printf(YEL "[WARN CauchyCPDFGridDispatcher2D:] Cannot log cpdf for the last step since SKIP_LAST_STEP == true! (The G Tables were not created, as they were skipped!)" NC "\n");
            return 1;
        }

        // Create path character array
        int len_log_dir = strlen(log_dir);
        char path[len_log_dir + 30];

        // Check if this marginal index pair has been logged yet
        int* tag = cpdf->marg_idxs_of_cached_2d_terms;
        int tag_idx = does_tag_exist(tag);
        int tag_count;
        // Marginal index pair has not been logged yet
        if(tag_idx == -1)
        {
            tag_idx = num_tags;
            tags[2*tag_idx] = tag[0];
            tags[2*tag_idx+1] = tag[1];
            tag_counts[tag_idx] = 0;
            num_tags++;
        }
        tag_count = ++tag_counts[tag_idx];
        sprintf(path, "%s/grid_elems_%d%d.txt", log_dir, tag[0], tag[1]);
        // First log if tag_count == 1
        // Open (possibly overwrite old) grid dims file
        if(tag_count == 1)
            dims_file = fopen(path, "w");
        else 
            dims_file = fopen(path, "a");
        if(dims_file == NULL)
        {
            printf(RED "[ERROR CauchyCPDFGridDispatcher2D:]\n  Could not open grid dim file!\n  Path: %s\n  Please check path and try again! Exiting!" NC "\n", path);
            exit(1);
        }
        fprintf(dims_file, "%d,%d\n", num_points_x, num_points_y);
        
        // write out binary data stream
        sprintf(path, "%s/cpdf_%d%d_%d.bin", log_dir, tag[0], tag[1], tag_count);
        data_file = fopen(path, "wb");
        if(data_file == NULL)
        {
            printf(RED "[ERROR CauchyCPDFGridDispatcher2D:]\n  Could not open binary data file!\n  Path: %s\n  Please check path and try again! Exiting!" NC "\n", path);
            exit(1);
        }
        fwrite(points, sizeof(CauchyPoint3D), num_grid_points, data_file);
        fclose(data_file);
        fclose(dims_file);
        return 0;
    }

    // returns index of tag in tag array if the tag exists
    // returns -1 if tag DNE
    int does_tag_exist(int* tag)
    {
        int tag_idx = -1;
        for(int i = 0; i < num_tags; i++)
        {    
            if( (tags[2*i] == tag[0]) && (tags[2*i+1] == tag[1]) )
            {
                tag_idx = i;
                break;
            }
        }
        return tag_idx;
    }

    ~CauchyCPDFGridDispatcher2D()
    {
        cpdf = NULL;
        free(points);
        if(log_dir != NULL)
            free(log_dir);
    }

};

struct ThreadEvalCPDF1D
{
    PointWiseNDimCauchyCPDF* cpdf;
    CauchyPoint2D* points;
    int num_grid_points;
    int marg_idx;
};

struct CauchyCPDFGridDispatcher1D
{
    PointWiseNDimCauchyCPDF* cpdf;
    CauchyPoint2D* points;
    int num_grid_points;
    // Logging of 10 marginals supported, can make this more robust later
    char* log_dir;
    int tags[10]; 
    int tag_counts[10]; 
    int num_tags;

    CauchyCPDFGridDispatcher1D(
        PointWiseNDimCauchyCPDF* _cpdf, 
        double grid_low,
        double grid_high,
        double grid_res,
        char* _log_dir = NULL)
    {
        points = NULL;
        reset_grid(grid_low, grid_high, grid_res);
        cpdf = _cpdf;
        if(_log_dir != NULL)
        {
            int len_logdir = strlen(_log_dir);
            log_dir = (char*) malloc( (strlen(_log_dir)+1) * sizeof(char));
            strcpy(log_dir, _log_dir);
            if(log_dir[len_logdir-1] == '/')
            {
                log_dir[len_logdir-1] = '\0';
            }
            // If the directory already exists, clear the directories
            check_dir_and_create(log_dir);
        }
        else 
            log_dir = NULL;
        num_tags = 0;
    }

    void reset_grid(double grid_low, double grid_high, double grid_res)
    {
        assert(grid_high > grid_low);
        assert(grid_res > 0);

        num_grid_points = (int) ( (grid_high - grid_low + grid_res - 1e-15) / grid_res ) + 1;
        points = (CauchyPoint2D*) realloc(points, num_grid_points * sizeof(CauchyPoint2D));
        null_ptr_check(points);
        for(int i = 0; i < num_grid_points; i++)
        {
            double grid_point = grid_low + i * grid_res;
            if(grid_point > grid_high)
                grid_point = grid_high;
            points[i].x = grid_point;
            points[i].y = -1;
        }
    }

    // Grid points on x correspond to marg_idx1, and grid points on y correspond to marg_idx2
    int evaluate_point_grid(int marg_idx, int num_threads, bool with_timing = false)
    {
        assert(marg_idx < cpdf->cauchyEst->d);
        assert(marg_idx > -1);
        assert(num_threads > 0);
        if( (cpdf->cauchyEst->master_step == cpdf->cauchyEst->num_estimation_steps) && (SKIP_LAST_STEP == true) )
        {
            printf(YEL "[WARN CauchyCPDFGridDispatcher1D:] Cannot evaluate cauchy estimator cpdf for the last step since SKIP_LAST_STEP == true! (The G Tables were not created, as they were skipped!)" NC "\n");
            return 1;
        }
        
        if(num_grid_points < num_threads)
            num_threads = num_grid_points-1;
        // Evaluate first point and setup caching
        CPUTimer tmr;
        tmr.tic();
        int marg_state_idxs[1] = {marg_idx};
        double point[1] = {points[0].x};
        points[0].y = creal( cpdf->evaluate_1D_marginal_cpdf(point, marg_state_idxs, 1, false, true) );
        tmr.toc(false);
        int cache_time = tmr.cpu_time_used;
        if(with_timing)
            printf("1D Grid Eval Step %d:\n  Caching took: %d ms (%d CF terms)\n", cpdf->cauchyEst->master_step, tmr.cpu_time_used, cpdf->cauchyEst->Nt);

        tmr.tic();
        if( (num_threads < 2) )
        {
            for(int i = 1; i < num_grid_points; i++)
            {
                point[0] = points[i].x;
                points[i].y = creal( cpdf->evaluate_1D_marginal_cpdf(point, marg_state_idxs, 1, false, true) );
            }
        }
        else
        {
            // Evaluate points 2 to num_grid_points using threaded and cached structure
            pthread_t tids[num_threads];
            ThreadEvalCPDF1D args[num_threads];
            int points_per_thread = (num_grid_points - 1) / num_threads;
            for(int i = 0; i < num_threads; i++)
            {
                args[i].cpdf = cpdf;
                args[i].marg_idx = marg_idx;
                args[i].points = points + 1 + i * points_per_thread;
                args[i].num_grid_points = points_per_thread;
                if( i == (num_threads-1) )
                    args[i].num_grid_points += (num_grid_points - 1) % num_threads;
                pthread_create(tids + i, NULL, evaluate_1d_marginal_grid_points, args+i);
            }
            for(int i = 0; i < num_threads; i++)
                pthread_join(tids[i], NULL);
        }
        tmr.toc(false);
        int compute_time = tmr.cpu_time_used;
        if(with_timing)
        {
            printf("  Computing %d gridpoints took: %d ms (used %d threads)\n", num_grid_points, tmr.cpu_time_used,num_threads);
            printf("  Total Time: %d ms\n", compute_time + cache_time);
        }
        return 0;
    }

    // File binary data is logged to: {log_dir}/cpdf_{marg_idx1}{marg_idx2}_{log_count}
    // File binary data dimensions is logged to {log_dir}/grid_elems_{marg_idx1}{marg_idx2}.txt
    // File row format for binary data dimensions: {num_points_x},{num_points_y}
    int log_point_grid()
    {
        FILE* data_file;
        FILE* dims_file;
        if(log_dir == NULL)
        {
            printf(YEL "[WARN CauchyCPDFGridDispatcher1D:]\n  Cannot Log! The log directory was not set!" NC "\n");
            return 1;
        }
        if( (cpdf->cauchyEst->master_step == cpdf->cauchyEst->num_estimation_steps) && (SKIP_LAST_STEP == true) )
        {
            printf(YEL "[WARN CauchyCPDFGridDispatcher2D:] Cannot log cpdf for the last step since SKIP_LAST_STEP == true! (The G Tables were not created, as they were skipped!)" NC "\n");
            return 1;
        }

        // Create path character array
        int len_log_dir = strlen(log_dir);
        char path[len_log_dir + 30];

        // Check if this marginal index pair has been logged yet
        int tag = cpdf->marg_idx_of_cached_1d_terms;
        int tag_idx = does_tag_exist(tag);
        int tag_count;
        // Marginal index pair has not been logged yet
        if(tag_idx == -1)
        {
            tag_idx = num_tags;
            tags[tag_idx] = tag;
            tag_counts[tag_idx] = 0;
            num_tags++;
        }
        tag_count = ++tag_counts[tag_idx];
        sprintf(path, "%s/grid_elems_%d.txt", log_dir, tag);
        // First log if tag_count == 1
        // Open (possibly overwrite old) grid dims file
        if(tag_count == 1)
            dims_file = fopen(path, "w");
        else 
            dims_file = fopen(path, "a");
        if(dims_file == NULL)
        {
            printf(RED "[ERROR CauchyCPDFGridDispatcher1D:]\n  Could not open grid dim file!\n  Path: %s\n  Please check path and try again! Exiting!" NC "\n", path);
            exit(1);
        }
        fprintf(dims_file, "%d\n", num_grid_points);
        
        // write out binary data stream
        sprintf(path, "%s/cpdf_%d_%d.bin", log_dir, tag, tag_count);
        data_file = fopen(path, "wb");
        if(data_file == NULL)
        {
            printf(RED "[ERROR CauchyCPDFGridDispatcher1D:]\n  Could not open binary data file!\n  Path: %s\n  Please check path and try again! Exiting!" NC "\n", path);
            exit(1);
        }
        fwrite(points, sizeof(CauchyPoint2D), num_grid_points, data_file);
        fclose(data_file);
        fclose(dims_file);
        return 0;
    }

    // returns index of tag in tag array if the tag exists
    // returns -1 if tag DNE
    int does_tag_exist(int tag)
    {
        int tag_idx = -1;
        for(int i = 0; i < num_tags; i++)
        {    
            if( tags[i] == tag )
            {
                tag_idx = i;
                break;
            }
        }
        return tag_idx;
    }

    ~CauchyCPDFGridDispatcher1D()
    {
        cpdf = NULL;
        free(points);
        if(log_dir != NULL)
            free(log_dir);
    }

};

void* evaluate_2d_marginal_grid_points(void* marg_args)
{
    ThreadEvalCPDF2D* tec2d = (ThreadEvalCPDF2D*) marg_args;
    PointWiseNDimCauchyCPDF* cpdf = tec2d->cpdf;
    int num_grid_points = tec2d->num_grid_points;
    CauchyPoint3D* points = tec2d->points;
    int marg_state_idxs[2] = {tec2d->marg_idx1, tec2d->marg_idx2};
    double xk_marginal[2];
    bool with_timing = false;
    bool with_caching = true;
    for(int i = 0; i < num_grid_points; i++)
    {
        xk_marginal[0] = points[i].x;
        xk_marginal[1] = points[i].y;
        C_COMPLEX_TYPE fx = cpdf->evaluate_2D_marginal_cpdf(xk_marginal, marg_state_idxs, 2, with_timing, with_caching);
        points[i].z = creal(fx);
        if( points[i].z < -1e-10 )
            printf(YEL"[WARN evaluate_2d_marginal_grid_points:]" NC " Negative CPDF Value of %.4E at x=%.3E, y=%.3E\n", points[i].z, points[i].x, points[i].y);
    }
    return NULL;
}

void* evaluate_1d_marginal_grid_points(void* marg_args)
{
    ThreadEvalCPDF1D* tec1d = (ThreadEvalCPDF1D*) marg_args;
    PointWiseNDimCauchyCPDF* cpdf = tec1d->cpdf;
    int num_grid_points = tec1d->num_grid_points;
    CauchyPoint2D* points = tec1d->points;
    int marg_state_idxs[1] = {tec1d->marg_idx};
    double xk_marginal[1];
    bool with_timing = false;
    bool with_caching = true;
    for(int i = 0; i < num_grid_points; i++)
    {
        xk_marginal[0] = points[i].x;
        C_COMPLEX_TYPE fx = cpdf->evaluate_1D_marginal_cpdf(xk_marginal, marg_state_idxs, 1, with_timing, with_caching);
        points[i].y = creal(fx);
    }
    return NULL;
}

#endif // _CPDF_NDIM_HPP_