
#ifndef _ENU_HPP_
#define _ENU_HPP_
#include "../../../../include/inc_enu_glpk.hpp"
#include "../../../../include/cell_enumeration.hpp"

/*
double* precoalign_check(double* A, int m, int n, int* m_coal)
{
    double* A_coal = (double*) malloc( (m*n) * sizeof(double) );
}
*/

void call_inc_enu(double* A, int size_A,
                  int m, int n,
                  int** out_B, int* size_out_B, 
                  const bool is_encoded = true,
                  const bool is_sorted = true)
{
    if( (!is_encoded && is_sorted) )
    {
        printf("Cannot Sort Unencoded Cell Enumeration! Exiting!\n");
        *out_B = (int*) malloc(0);
        *size_out_B = 0;
        return;
    }
    int ccA = inc_enu_cell_count_central(m, n);
    double root_point[n];
    for(int i = 0; i < n; i++)
        root_point[i] = random_uniform();

    if(is_encoded)
        *out_B = (int*) malloc( ccA * sizeof(int) );
    else 
        *out_B = (int*) malloc( ccA * m * sizeof(int) );
    run_inc_enu(*out_B, size_out_B, A, root_point, 1, m, n, is_encoded, is_sorted);
    if(!is_encoded)
        *size_out_B *= m;
}

void call_nat_enu(double* A, int size_A,
                  int m, int n,
                  int** out_B, int* size_out_B, 
                  const bool is_encoded = true,
                  const bool is_sorted = true)
{
    if( (!is_encoded && is_sorted) )
    {
        printf("Cannot Sort Unencoded Cell Enumeration! Exiting!\n");
        *out_B = (int*) malloc(0);
        *size_out_B = 0;
        return;
    }
    bool warn_checks = true;
    int* B = make_enumeration_matrix(A, m, n, size_out_B, NULL, false, warn_checks);
    if(!is_encoded)
    {
        *out_B = (int*) malloc((*size_out_B)*m*sizeof(int));
        inc_enu_decode(*out_B, B, *size_out_B, m);
        *size_out_B *= m;
    }
    else
    {
        if(is_sorted)
            inc_enu_sort_Bs(size_out_B, B, 1, *size_out_B);
        *out_B = (int*) malloc((*size_out_B)*sizeof(int));
        memcpy(*out_B, B, *size_out_B * sizeof(int));
    }
    free(B);
}
#endif