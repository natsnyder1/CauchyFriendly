#ifndef _INC_ENU_GLPK_H_
#define _INC_ENU_GLPK_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <glpk.h>
#include <assert.h>

// Log Error Warnings and Colors
#define RED "\e[0;31m"
#define NC "\e[0m"
#define YEL  "\e[0;33m" 

// NULL Pointer checker
void inc_enu_null_ptr_check(void* ptr)
{
    if(ptr == NULL)
    {
        printf("Pointer allocated by malloc has returned with NULL, indicating FAILURE! Please Debug Further!\n");
        exit(1);
    }
}

void inc_enu_print_simplex_info_meaning(int info)
{
    if(info == 0)
        printf("[info == 0]: The LP problem instance has been successfully solved. (This code does not necessarily mean that the solver has found optimal solution. It only means that the solution process was successful.)\n");
    else if(info == GLP_EBADB) 
        printf("[info == GLP_EBADB]: Unable to start the search, because the initial basis specified in the problem object is invalid—the number of basic (auxiliary and structural) variables is not the same as the number of rows in the problem object.\n");
    else if(info == GLP_ESING)
        printf("[info == GLP_ESING]: Unable to start the search, because the basis matrix corresponding to the initial basis is singular within the working precision.\n");
    else if(info == GLP_ECOND)
        printf("[info == GLP_ECOND]: Unable to start the search, because the basis matrix corresponding to the initial basis is ill-conditioned, i.e. its condition number is too large.\n");
    else if(info == GLP_EBOUND)
        printf("[info == GLP_EBOUND]: Unable to start the search, because some double-bounded (auxiliary or structural) variables have incorrect bounds.\n");
    else if(info == GLP_EFAIL)
        printf("[info == GLP_EFAIL]: The search was prematurely terminated due to the solver failure.\n");
    else if(info == GLP_EOBJLL)
        printf("[info == GLP_EOBJLL]: The search was prematurely terminated, because the objective function being maximized has reached its lower limit and continues decreasing (the dual simplex only).\n");
    else if(info == GLP_EOBJUL)
        printf("[info == GLP_EOBJUL]: The search was prematurely terminated, because the objective function being minimized has reached its upper limit and continues increasing (the dual simplex only).\n");
    else if(info == GLP_EITLIM) 
        printf("[info == GLP_EITLIM]: The search was prematurely terminated, because the simplex iteration limit has been exceeded.\n");
    else if(info == GLP_ETMLIM)
        printf("[info == GLP_ETMLIM]: The search was prematurely terminated, because the time limit has been exceeded.\n");
    else if(info == GLP_ENOPFS) 
        printf("[info == GLP_ETMLIM]: The LP problem instance has no primal feasible solution (only if the LP presolver is used).\n");
    else if(info == GLP_ENODFS) 
        printf("[info == GLP_ENODFS]: The LP problem instance has no dual feasible solution (only if the LP presolver is used).\n");
}

void inc_enu_print_simplex_status_meaning(int info)
{
    if(info == GLP_OPT)
        printf("[info == GLP_OPT]: Solution is optimal\n");
    if(info == GLP_FEAS)
        printf("[info == GLP_FEAS]: Solution is feasible\n");
    if(info == GLP_INFEAS)
        printf("[info == GLP_INFEAS]: Solution is infeasible\n");
    if(info == GLP_NOFEAS)
        printf("[info == GLP_NOFEAS]: Problem has no feasible solution\n");
    if(info == GLP_UNBND)
        printf("[info == GLP_UNBND]: Problem has unbounded solution\n");
    if(info == GLP_UNDEF)
        printf("[info == GLP_UNDEF]: Solution is undefined\n");
}

int inc_enu_is_cell(glp_prob* lp, int* ja, double* ar, double* s, double* x, int nS, int m, int n)
{
    const int nr = glp_get_num_rows(lp);
    if(nr > nS)
    {
        // Remove Rows (this can be improved)
        int rows[nr-nS+1];
        for(int i = 1; i <= nr-nS; i++)
            rows[i] = nS + i;
        glp_del_rows(lp, nr-nS, rows);
    }
    if(nr < nS)
        glp_add_rows(lp, nS-nr);
    int min = nS < nr ? nS : nr;
    min = min - 1 < 0 ? min : min - 1;
    for(int i = min; i < nS; i++)
    {
        if(s[i] > 0)
            glp_set_row_bnds(lp, i+1, GLP_LO, 1.0, 0.0);
        else
            glp_set_row_bnds(lp, i+1, GLP_UP, 0.0, -1.0);
        glp_set_mat_row(lp, i+1, n, ja, ar + i*n);
    }
    int attempts = 0;
    glp_smcp parm;
    glp_init_smcp(&parm);
    parm.msg_lev = GLP_MSG_OFF; //GLP_MSG_ERR; //GLP_MSG_OFF
    RESTART:
    glp_std_basis(lp);
    int simplex_info = glp_simplex(lp, &parm); 
    //inc_enu_print_simplex_info_meaning(simplex_info);
    int simplex_status = glp_get_status(lp);
    //inc_enu_print_simplex_status_meaning(simplex_status);
    if( (simplex_status == GLP_OPT) || (simplex_status == GLP_FEAS) || (simplex_status == GLP_UNBND) )
    {
        for(int i = 0; i < n; i++)
            x[i] = glp_get_col_prim(lp, i+1);
        return 1;
    }    
    else if( (simplex_status == GLP_NOFEAS) || (simplex_status == GLP_INFEAS)  )
        return 0;
    else
    {
        // This can be modified if solver reports undefined solution (numerical error), and the tolerances can be modified
        if(attempts > -1)
        {
            //inc_enu_print_simplex_info_meaning(simplex_info);
            //inc_enu_print_simplex_status_meaning(simplex_status);
            //printf(RED "[WARNING IN inc_enu_is_cell FUNCTION:]"
            //       YEL "Solution is undefined! MAX_ATTEMPTS=%d Reached. Declaring this an infeasible cell..."
            //       NC "\n", attempts);
            return 0;
        }
        //printf("[WARN IN inc_enu_is_cell FUNCTION: Solve Attempts=%d] Solution is undefined! Solver Failure...attempting to make solution more sensitive...\n", attempts+1);
        //parm.tol_bnd /= 10;
        //parm.tol_dj /= 10;
        parm.tol_piv /= 10; // This is the param which is important to make smaller...so I think
        //parm.msg_lev = GLP_MSG_DBG; // can mute output by commenting this
        attempts += 1;
        goto RESTART;
    }
}

void inc_enu(glp_prob* lp, int* ja, double* ar, int* B, double* s, double* x, int* cell_count, int nS, int m, int n)
{
    if(nS == m)
    {
        for(int i = 0; i < nS; i++)
            B[*cell_count * m + i] = (int)s[i];
        *cell_count += 1;
    }   
    else 
    {
        double dp = 0.0;
        for(int i = 0; i < n; i++)
            dp += ar[1 + nS*n + i] * x[i];
        if(dp >= 0)
        {
            s[nS] = 1;
            inc_enu(lp, ja, ar, B, s, x, cell_count, nS+1, m, n);
            s[nS] = -1;
            if(inc_enu_is_cell(lp, ja, ar, s, x, nS+1, m, n) )
                inc_enu(lp, ja, ar, B, s, x, cell_count, nS+1, m, n);
        }
        else if( dp < 0)
        {   
            s[nS] = -1;
            inc_enu(lp, ja, ar, B, s, x, cell_count, nS+1, m, n);
            s[nS] = 1;
            if(inc_enu_is_cell(lp, ja, ar, s, x, nS+1, m, n) )
                inc_enu(lp, ja, ar, B, s, x, cell_count, nS+1, m, n);
        }
    }
}

void inc_enu_setup_feasibility_lp(glp_prob** lp, int* ja, int m, int n)
{
    for(int i = 0; i <= n; i++)
        ja[i] = i;
    *lp = glp_create_prob(); //creates a problem object
    glp_set_prob_name(*lp, "inc_enu"); //assigns a symbolic name to the problem object
    glp_set_obj_dir(*lp, GLP_MIN);
    glp_add_cols(*lp, n);
    char x_name[4];
    x_name[0] = 'x';
    for(int i = 0; i < n; i++)
    {
        snprintf(x_name+1, 3, "%d",i+1);
        glp_set_col_name(*lp, i+1, x_name); //assigns name x1 to first column
        glp_set_col_bnds(*lp, i+1, GLP_FR, 0.0, 0.0); //sets the type and bounds of the first row,
    }
    // Sets objective to be a feasibility problem
    for(int i = 0; i < n; i++)
        glp_set_obj_coef(*lp, i+1, 0);
}

void inc_enu_flip_hyperplanes_wrt_root_point(double* ar, double* rp, int* flips, double* sign_seq, double* x_feas, int m, int n)
{
    for(int i = 0; i < m; i++)
    {
        double hs = 0; // halfspace
        for(int j = 0; j < n; j++)
            hs += ar[1 + i*n + j] * rp[j];
        if( hs == 0 )
        {
            printf("Root point falls exactly on hyperplane %d. Must choose new root point!\n", i);
            exit(1);
        }
        else if(hs < 0)
        {
            for(int j = 0; j < n; j++)
                ar[1 + i*n + j] *= -1;
            flips[i] = -1;
        }
        else
            flips[i] = 1;
    }
    sign_seq[0] = 1;
    memcpy(x_feas, rp, n*sizeof(double));
}

int inc_enu_post_process(int* B, int* flips, int cell_counts, int m)
{
    for(int i = 0; i < cell_counts; i++)
    {    
        for(int j = 0; j < m; j++)
        {
            B[i*m + j] *= flips[j];
            B[cell_counts*m + i*m + j] = -B[i*m + j];
        }
    }
    return 2*cell_counts;
}   

void inc_enu_encode(int* B_enc, int* B, int cell_count, int m)
{
    // The GPU code uses 0 as "sign 1" and 1 as "sign -1"
    // This is to use bitwise operations on the encoded sign vectors (the xor) 
    // For example: -1 * -1 = 1.. which translate to 1 ^ 1 = 0 
    // Another example 1 * 1 = 1.. which translate to 0 ^ 0 = 0
    // Another example -1 * 1 = -1..which translate to 1 ^ 0 = 1
    for(int i = 0; i < cell_count; i++)
    {
        B_enc[i] = 0;
        for(int j = 0; j < m; j++)
            if(B[i*m + j] == -1)
                B_enc[i] |= (1 << j);
    }
}

void inc_enu_decode(int* B, int* B_enc, int cell_count, int m)
{
    // See inc_enu_encode for sign convention
    for(int i = 0; i < cell_count; i++)
    {
        int b_enc = B_enc[i];
        for(int j = 0; j < m; j++)
        {
            if( ((b_enc >> j) & 0x01) )
                B[i*m + j] = -1;
            else
                B[i*m + j] = 1;
        }
    }
}

void inc_enu_reset(glp_prob* lp)
{
    const int nr = glp_get_num_rows(lp);
    int rows[nr+1];
    for(int i = 0; i < nr; i++)
        rows[i+1] = i+1;
    glp_del_rows(lp, nr, rows);
}

int inc_enu_sort_compare_int(const void* p1, const void* p2)
{
  return *((int*)p1) - *((int*)p2);
}

void inc_enu_sort_Bs(int* feas_counts, int* enum_array, int N, int fcc_cen)
{
  for(int i = 0; i < N; i++)
    qsort(enum_array + i*fcc_cen, feas_counts[i], sizeof(int), &inc_enu_sort_compare_int);
}

void inc_enu_print(int* B, int cell_count, int m)
{
    printf("Found %d cells in Arrangement A:\n", cell_count);
    printf("B is:\n");
    for(int i = 0; i < cell_count; i++)
    {    for(int j = 0; j < m; j++)
            printf("%d, ", B[i*m+j]);
        printf("\n");
    }
}

void inc_enu_print_encoded(int* B_enc, int cell_count, bool with_sort = false)
{
    if(with_sort)
    {    
        qsort(B_enc, cell_count, sizeof(int), &inc_enu_sort_compare_int);
        printf("B_enc (with sorted sign-vectors) is:\n");
    }
    else
        printf("B_enc is:\n");
    for(int i = 0; i < cell_count; i++)
        printf("%d, ", B_enc[i]);
    printf("\n");
}

void inc_enu_print_encoded(int* B_enc, int cell_count, int m, bool with_sort = false)
{
    if(with_sort)
    {    
        qsort(B_enc, cell_count, sizeof(int), &inc_enu_sort_compare_int);
        printf("B_enc (with sorted sign-vectors) is:\n");
    }
    else
        printf("B_enc is:\n");
    for(int i = 0; i < cell_count; i++)
    {
        for(int j = 0; j < m; j++)
        {
            if( ((B_enc[i] >> j) & 0x01) == 1)
                printf("-1, ");
            else
                printf("1, ");
        }
        printf("\n");
    }
    printf("\n");
}

int inc_enu_binomial_coeff(int n, int k)
{
    int res = 1;
    // Since C(n, k) = C(n, n-k)
    if (k > n - k)
        k = n - k;
 
    // Calculate value of
    // [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
    for (int i = 0; i < k; ++i) {
      res *= (n - i);
      res /= (i + 1);
    }
 
    return res;
}

int inc_enu_cell_count_central(int num_hyp, int dim)
{
  if(num_hyp < dim)
    return 1 << num_hyp;
  int fc = 0;
  for( int i = 0; i < dim; i++)
    fc += inc_enu_binomial_coeff(num_hyp-1, i);
  return 2*fc;
}

// Bs is a N x inc_enu_cell_count_central(m, n) x m integer allocated array (IF is_encoded_B = False), otherwise is a N x inc_enu_cell_count_central(m, n) integer allocated array (IF is_encode_B = True). On return, this array is filled
// cell_counts is a N integer allocated array (On return, this is filled)
// As is a N x m x n double allocated arry filled with the N hyperplanes arrangements of size m x n to be enumerated
// root_point is a double allocated array of size n, which is a point in n-space used to define the "all postive halfspace" of each hyperplane
// N is the number of hyperplane arrangements to be enumerated
// m is the number of hyperplanes
// n is the dimention of each hyperplane
// is_encode_B is the boolean which decides whether the enumeration matrix should be returned encoded as integers or should be returned unencoded. Encoded inc-enu is only supported up to 32 hyperplanes (or sizeof(int)*8)
// is_sort_B is the boolean which decides whether the enumeration matrix should be returned sorted 
void run_inc_enu(int* Bs, int* cell_counts, double* As, double* root_point, const int N, const int m, const int n, const bool is_encode_B = true, const bool is_sort_B = false)
{
    if(is_encode_B)
        assert( sizeof(int) * 8 > (int)m );
    glp_prob *lp; // LP pointer
    int ja[1 + n]; // Column indices of each element of constraint matrix (GLPK Starts all indexing at 1. Index 0 is ignored.)
    inc_enu_setup_feasibility_lp(&lp, ja, m, n);
    double ar[1 + m*n]; // memory space for constraint matrix of inc-enu LP (GLPK Starts all indexing at 1. Index 0 is ignored.)
    ar[0]=0;
    double sign_seq[m]; // storage space for sign sequence being tested for inc-enu LP
    int flips[m]; // Initial flips (of hyperplane coefficients) such that the root point is in all positive half-space
    double x_feas[n]; // storage space to retrieve LP (feasibility) solution
    int fcc_cen = inc_enu_cell_count_central(m, n);
    int* _B = (int*) malloc( fcc_cen * m * sizeof(int) ); // Use this memory space to build B's if Bs is to be returned encoded
    inc_enu_null_ptr_check(_B);
    for(int i = 0; i < N; i++)
    {
        // Run inc-enu
        double* A = As + i * m * n;
        int* B = is_encode_B ? _B : Bs + i * fcc_cen * m;
        int* cell_count = cell_counts + i;
        // Load program data into ar
        memcpy(ar + 1, A, m*n*sizeof(double));
        // Preprocess
        inc_enu_flip_hyperplanes_wrt_root_point(ar, root_point, flips, sign_seq, x_feas, m, n);
        *cell_count = 0;
        // Run inc-enu
        inc_enu(lp, ja, ar, B, sign_seq, x_feas, cell_count, 1, m, n);
        // Postprocess
        *cell_count = inc_enu_post_process(B, flips, *cell_count, m);
        assert( (*cell_count) <= fcc_cen);
        // Encode result if desired 
        if(is_encode_B)
            inc_enu_encode(Bs + i*fcc_cen, B, *cell_count, m);
        // Reset problem
        inc_enu_reset(lp);
    }
    if(is_sort_B && is_encode_B)
        inc_enu_sort_Bs(cell_counts, Bs, N, fcc_cen);
    if(is_sort_B && !is_encode_B)
        assert(false); // Not implemented yet
    free(_B);
    glp_delete_prob(lp);
}

#endif // _INC_ENU_GLPK_H_
