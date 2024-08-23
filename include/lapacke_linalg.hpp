#ifndef _LAPACKE_LINALG_HPP_
#define _LAPACKE_LINALG_HPP_

#include "cauchy_types.hpp"
#include <lapacke.h>

// BEGIN MKL DEFINE
/*
typedef double __complex__ C_COMPLEX_TYPE;
#define MKL_Complex16 C_COMPLEX_TYPE
#include<mkl_lapacke.h>
#include <iomanip>
#include <limits>
*/
// END MKL DEFINE

/*
------  BEGIN HELPER FUNCTIONS ---------
These functions currently only support double, can be easily modified to arbitrary type using templates
*/

// A is size n x n,
// evals is size n,
// evecs is size n x n
int lapacke_sym_eig(double* A, double* evals, double* evecs, const int n)
{
    memcpy(evecs, A, n*n*sizeof(double)); // evecs starts with A and is returned as its eigenvectors
    int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, evecs, n, evals);
    if(info > 0)
        printf("[ERROR:] Eigenvalues not computed. Info = %d. \
            If info = i, the algorithm failed to converge. \
            i off-diagnoal elements of an intermediate \
            tridiagonal form did not converge to zero.", info);
    return info;
}

// A is size n x n,
// evals is size n,
// evecs is size n x n
int lapacke_gen_eigs(double* A, C_COMPLEX_TYPE* evals, C_COMPLEX_TYPE* evecs, const int n)
{
    double* V = (double*) malloc(n*n*sizeof(double));
    null_ptr_check(V);
    double eig_rp[n]; // real part of eigenvalues
    double eig_ip[n]; // complex part of eigenvalues
    char jobvl = 'N'; // left_esys
    char jobvr = 'V'; // right_esys
    const int lda = n;
    const int ldvl = n;
    const int ldvr = n; // leading dimention of right eigenvectors
    double* vl = NULL; // No left eigenvalues
    int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, jobvl, jobvr, n, A, lda, eig_rp, eig_ip, vl, ldvl, V, ldvr);
    if(info > 0)
        printf("[Error:] INFO = %d, the QR algorithm failed to compute all the \
                eigenvalues, and no eigenvectors have been computed; \
                elements %d+1:N of WR and WI contain eigenvalues which \
                have converged.\n", info, info);
    else if(info < 0)
        printf("[Error:] INFO = %d, the %d-th argument had an illegal value.\n", info, -info);
    else 
    {
        const double eps = 1e-15; // epsilon which determines if the cmplx part is non-machine zero on return
        for(int i = 0; i < n; i++)
        {
            //evals[i] = eig_rp[i] + I*eig_ip[i];
            evals[i] = MAKE_CMPLX(eig_rp[i], eig_ip[i]);
        }
        for(int i = 0; i < n; i++)
        {
            if( fabs(eig_ip[i]) > eps )
            {
                for(int j = 0; j < n; j++)
                {
                    //evecs[i + j*n] = V[i + j*n] + I*V[i+1 + j*n];
                    evecs[i + j*n] = MAKE_CMPLX(V[i + j*n], V[i+1 + j*n]);
                }
                for(int j = 0; j < n; j++)
                {
                    //evecs[i+1 + j*n] = V[i + j*n] - I*V[i+1 + j*n];
                    evecs[i+1 + j*n] = MAKE_CMPLX(V[i + j*n], -V[i+1 + j*n]);
                }
                i++;
            }
            else 
            {
                for(int j = 0; j < n; j++)
                {
                    //evecs[i + j*n] = V[i + j*n] + 0.0*I;
                    evecs[i + j*n] = MAKE_CMPLX(V[i + j*n], 0.0);
                }
            }
        }
    }
    free(V);
    return info;
}

// Assumes Row-Major Data-Layout,
// U \in R^(m x m),
// S \in R^(min(m,n)),
// Vt \in R^(n,n)
int lapacke_svd(double* A, double* U, double* S, double* Vt, int m, int n)
{
    const int lda = n; // Since using ROW_MAJOR, lda is n and not m
    const int ldu = m;
    const int ldvt = n;
    const int superb_size = (n < m) ? n-1 : m-1;
    double superb[superb_size]; // On stack
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, Vt, ldvt, superb);
    if(info > 0)
        printf("[Error:] Info = %d. SVD not complete. If info > 0, info specifies \
        how many superdiagonals of an intermediate bidiagonal form B did not \
         converge to zero.\n", info);
    return info;
}
// Assumes Row-Major Data-Layout,
// U \in C^(m x m),
// S \in C^(min(m,n)),
// Vt \in C^(n,n)
int lapacke_svd(C_COMPLEX_TYPE* A, C_COMPLEX_TYPE* U, double* S, C_COMPLEX_TYPE* Vt, int m, int n)
{
    const int lda = n; // Since using ROW_MAJOR, lda is n and not m
    const int ldu = m;
    const int ldvt = n;
    const int superb_size = (n < m) ? n-1 : m-1;
    double superb[superb_size]; // On stack
    int info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, S, U, ldu, Vt, ldvt, superb);
    if(info > 0)
        printf("[Error:] Info = %d. SVD not complete. If info > 0, info specifies \
        how many superdiagonals of an intermediate bidiagonal form B did not \
         converge to zero.\n", info);
    return info;
}

// A is an n x n matrix
// Returns condition number and info (from lapack)
int lapacke_cond(double* A, int n, double* cond_num)
{
    double s[n];
    double* _A = (double*) malloc(n*n*sizeof(double));
    null_ptr_check(_A);
    memcpy(_A,A,n*n*sizeof(double));
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', n, n, _A, n, s, NULL, n, NULL, n);
    if(info != 0)
    {
        printf("Error in the condition number call\n");
        free(_A);
        return info;
    }
    *cond_num = s[0] / s[n-1];
    free(_A);
    return info;
}
// A is an n x n matrix
// work_A is a declared n * n sized memory chunk
// work_sv is a declared n sized memory chunk
// Returns condition number and info (from lapack)
int lapacke_cond(double* A, double* work_A, double* work_sv, int n, double* cond_num)
{
    memcpy(work_A, A, n*n*sizeof(double));
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', n, n, work_A, n, work_sv, NULL, n, NULL, n);
    if(info != 0)
    {
        printf("Error in the condition number call\n");
        return info;
    }
    *cond_num = work_sv[0] / work_sv[n-1];
    return info;
}

// A is an m x n matrix
// work_A is a declared m * n sized memory chunk
// Returns condition number and info (from lapack)
int lapacke_cond(double* A, double* work_A, int m, int n, double* cond_num)
{
    const int lda = n; // Since using ROW_MAJOR, lda is n and not m
    const int ldu = m;
    const int ldvt = n;
    const int min_mn = m < n ? m : n;
    double s[min_mn]; // On stack
    memcpy(work_A, A, m*n*sizeof(double));
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', m, n, work_A, lda, s, NULL, ldu, NULL, ldvt);
    if(info > 0)
        printf("[ERROR in condition number:] Info = %d. SVD not complete. If info > 0, info specifies \
        how many superdiagonals of an intermediate bidiagonal form B did not \
         converge to zero.\n", info);
    *cond_num = s[0] / s[min_mn-1];
    return info;
}

// Permutation matrix P is size min_mn = m < n ? m : n;
int lapacke_plu(double* A, int* P, int m, int n, const bool with_warning = true)
{
    const int lda = n; // row major
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, A, lda, P);
    if(with_warning)
        if(info > 0)
            printf("[PLU Warning:] U[%d,%d] is singular \
            and its inverse will not be able to be computed.\n", info, info);
    return info;
}

// Permutation matrix P is size min_mn = m < n ? m : n;
int lapacke_plu(C_COMPLEX_TYPE* A, int* P, int m, int n)
{
    const int lda = n; // row major
    int info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, m, n, A, lda, P);
    if(info > 0)
        printf("[Warning:] U%d%d is singular \
        and its inverse will not be able to be computed.\n", info, info);
    return info;
}

// A is size n x n
int lapacke_inv(double* A, int n)
{
    const int lda = n; // row major
    int P[n];
    int inv_info;
    int plu_info;
    plu_info = lapacke_plu(A, P, n, n);
    if(plu_info == 0)
    {
        inv_info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, lda, P);
        return inv_info;
    }
    else
    {
        printf("[Error:] PLU Factorization Failed. Inverse could not be completed! Rank deficient matrix found at pivit %d\n", plu_info);
        return plu_info;
    }
}

// A is size n x n
int lapacke_inv(double* A, double* A_inv, int n)
{
    memcpy(A_inv, A, n*n*sizeof(double));
    const int lda = n; // row major
    int P[n];
    int inv_info;
    int plu_info;
    plu_info = lapacke_plu(A_inv, P, n, n);
    if(plu_info == 0)
    {
        inv_info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A_inv, lda, P);
        return inv_info;
    }
    else
    {
        printf("[Error:] PLU Factorization Failed. Inverse could not be completed! Rank deficient matrix found at pivit %d\n", plu_info);
        return plu_info;
    }
}

// A is size n x n
int lapacke_inv(C_COMPLEX_TYPE* A, int n)
{
    const int lda = n; // row major
    int P[n];
    int inv_info;
    int plu_info;
    plu_info = lapacke_plu(A, P, n, n);
    if(plu_info == 0)
    {
        inv_info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, A, lda, P);
        return inv_info;
    }
    else
    {
        printf("[Error:] PLU Factorization Failed. Inverse could not be completed! Rank deficient matrix found at pivit %d\n", plu_info);
        return plu_info;
    }
}

// A is size n x n
int lapacke_inv(C_COMPLEX_TYPE* A, C_COMPLEX_TYPE* A_inv, int n)
{
    memcpy(A_inv, A, n*n*sizeof(C_COMPLEX_TYPE));
    const int lda = n; // row major
    int P[n];
    int inv_info;
    int plu_info;
    plu_info = lapacke_plu(A_inv, P, n, n);
    if(plu_info == 0)
    {
        inv_info = LAPACKE_zgetri(LAPACK_ROW_MAJOR, n, A_inv, lda, P);
        return inv_info;
    }
    else
    {
        printf("[Error:] PLU Factorization Failed. Inverse could not be completed! Rank deficient matrix found at pivit %d\n", plu_info);
        return plu_info;
    }
}

// A is size m x n,
// B is size max(m,n) x nrhs, but inputted as m x nrhs,
// Solution of size (n x nrhs) is returned in B
int lapacke_lstsq(double* A, double* B, const int m, const int n, const int nrhs, bool is_A_transposed = false)
{
    char TRANS = is_A_transposed ? 'T' : 'N'; // sys involves A
    int lda = n; 
    int ldb = nrhs;
    int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, TRANS, m, n, nrhs, A, lda, B, ldb);
    if(info < 0)
    {
        printf("[Error:] INFO = %d, the %d-th argument had an illegal value\n", info, -info);
    }
    if(info > 0)
    {
         printf("INFO = %d, the %d-th diagonal element of the \
                triangular factor of A is zero, so that A does not have \
                full rank; the least squares solution could not be computed.\n", info, info);
    }
    return info;
}


// A is size m x n,
// B is size m x nrhs,
// X is size max(m,n) x nrhs,
// Solution is of size n x nrhs returned in X
int lapacke_lstsq(double* A, double* B, double* X, const int m, const int n, const int nrhs, bool is_A_transposed = false)
{
    double* _A = (double*) malloc(m*n*sizeof(double));
    null_ptr_check(_A);
    memcpy(_A, A, m*n*sizeof(double));
    memcpy(X, B, m * nrhs * sizeof(double));
    char TRANS = is_A_transposed ? 'T' : 'N'; // sys involves A
    int lda = n; 
    int ldb = nrhs;
    int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, TRANS, m, n, nrhs, _A, lda, X, ldb);
    if(info < 0)
    {
        printf("[Error:] INFO = %d, the %d-th argument had an illegal value\n", info, -info);
    }
    if(info > 0)
    {
         printf("INFO = %d, the %d-th diagonal element of the \
                triangular factor of A is zero, so that A does not have \
                full rank; the least squares solution could not be computed.\n", info, info);
    }
    free(_A);
    return info;
}

// Solves system A*X = B by LU factorization and returns solution in X,
// A \in R^{n x n},
// B \in R^{n x nrhs} where each column of B is a RHS, for which there are NRHS
// X \in R^{n x nrhs} and contains solution on return
int lapacke_solve(double* A, double* B, double* X, int n, int nrhs)
{
    double* _A = (double*) malloc(n*n*sizeof(double));
    null_ptr_check(_A);
    memcpy(_A, A, n*n*sizeof(double));
    memcpy(X, B, n*nrhs*sizeof(double));
    int P[n];
    int lda = n;
    int ldb = nrhs;
    int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, _A, lda, P, X, ldb);
    if(info < 0)
        printf("[ERROR:] The %d-th argument has an illegal value. Solution not computed.\n", -1*info);
    if(info > 0)
        printf("[ERROR:] U(%d,%d) is exactly zero.  The factorization \
                has been completed, but the factor U is exactly \
                singular, so the solution could not be computed..\n", info, info);
    free(_A);
    return info;
}

// Solves system A*X = B by PLU factorization and returns solution in B,
// A \in R^{n x n} and returns the LU factors in A,
// B \in R^{n x nrhs} where each column of B is a RHS, for which there are NRHS,
// P \in Z^n is the permutation matrix for PLU factorization
// X \in R^{n x nrhs} and contains solution on return
int lapacke_solve(double* A, double* B, int* P, int n, int nrhs)
{
    int lda = n;
    int ldb = nrhs;
    int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, A, lda, P, B, ldb);
    if(info < 0)
        printf("[ERROR:] The %d-th argument has an illegal value. Solution not computed.\n", -1*info);
    if(info > 0)
        printf("[ERROR:] U(%d,%d) is exactly zero.  The factorization \
                has been completed, but the factor U is exactly \
                singular, so the solution could not be computed..\n", info, info);
    return info;
}

// Solves system A*X = B by LU factorization and returns solution in X,
// A \in R^{n x n},
// work_A is a n*n sized memory chunk
// B \in R^{n x nrhs} where each column of B is a RHS, for which there are NRHS
// X \in R^{n x nrhs} and contains solution on return
int lapacke_solve(double* A, double* B, double* X, int* P, double* work_A, int n, int nrhs)
{
    memcpy(work_A, A, n*n*sizeof(double));
    memcpy(X, B, n*nrhs*sizeof(double));
    int lda = n;
    int ldb = nrhs;
    int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs, work_A, lda, P, X, ldb);
    if(info < 0)
        printf("[ERROR:] The %d-th argument has an illegal value. Solution not computed.\n", -1*info);
    if(info > 0)
        printf("[ERROR:] U(%d,%d) is exactly zero.  The factorization \
                has been completed, but the factor U is exactly \
                singular, so the solution could not be computed..\n", info, info);
    return info;
}

// Solves system (LU) @ X = B and returns solution in B,
// LU is the LU factorized matrix A = LU by first calling lapacke_plu (dgetrf) and where A is n x n
// P is the n array permutation matrix
// B is the n x nrhs matrix that is returned with the solution
// is_transposed dictates whether LU(.T) @ X = B should be solved (i.e, A(.T) @ X = B)
int lapacke_solve_trf(double* LU, int* P, double* B, const int n, const int nrhs, bool is_transposed = false)
{
    char trans = is_transposed ? 'T' : 'N';
    int info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, trans, n, nrhs, LU, n, P, B, nrhs);  
    if(info != 0)
    {
        printf("[ERROR in lapacke_solve_trf:] when calling LAPACKE_dgetrs, INFO is %d,\n \
          INFO = 0:  indicates successful exit,\n \
          INFO < 0:  if INFO = -i, the i-th argument had an illegal value\n", info);
    }
    return info;
}

// rank function 
int lapacke_matrix_rank(double* A, double* work_A, int m, int n, int* matrix_rank)
{
    const int lda = n; // Since using ROW_MAJOR, lda is n and not m
    const int ldu = m;
    const int ldvt = n;
    const int max_mn = m > n ? m : n;
    const int min_mn = m < n ? m : n;
    double s[min_mn]; // On stack
    memcpy(work_A, A, m*n*sizeof(double));
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'N', m, n, work_A, lda, s, NULL, ldu, NULL, ldvt);
    if(info > 0)
        printf("[ERROR in Matrix Rank:] Info = %d. SVD not complete. If info > 0, info specifies \
        how many superdiagonals of an intermediate bidiagonal form B did not \
         converge to zero.\n", info);
    double eps = 2.220446049250313e-16;
    double tol = s[0] * max_mn * eps;
    int rank_count = 0;
    for(int i = 0; i < min_mn; i++)
        if(s[i] > tol)
            rank_count += 1;
    *matrix_rank = rank_count;
    return info;
}

// Determinant of matrix
// A is n x n
// work_A is a n*n sized memory chunk
// Returns determinant via the determinant pointer
int lapacke_det(double* A, double* work_A, int n, double* determinant)
{
    int P[n];
    memcpy(work_A, A, n*n*sizeof(double));
    int info = lapacke_plu(work_A, P, n, n);
    if(info != 0)
        printf("[Determinant Warning:] PLU (which solves determinant) has issued a warning. See PLU Warning Output.\n");
    double det = 1;
    for(int i = 0; i < n; i++)
        det *= work_A[i*n+i];
    // Find how many swaps occured in the Permutation matrix
    int swaps = 0;
    for(int i = 0; i < n; i++)
        if(P[i] != i+1)
            swaps += 1;
    int swap_sign = 1 - 2*(swaps % 2);
    *determinant = det * swap_sign;
    return info;
}

// Determinant of matrix
// A is n x n and gets overrided by this version
// Returns determinant via the determinant pointer
int lapacke_det(double* A, int n, double* determinant)
{
    int P[n];
    int info = lapacke_plu(A, P, n, n);
    if(info != 0)
        printf("[Determinant Warning:] PLU (which solves determinant) has issued a warning. See PLU Warning Output.\n");
    double det = 1;
    for(int i = 0; i < n; i++)
        det *= A[i*n+i];
    // Find how many swaps occured in the Permutation matrix
    int swaps = 0;
    for(int i = 0; i < n; i++)
        if(P[i] != i+1)
            swaps += 1;
    int swap_sign = 1 - 2*(swaps % 2);
    *determinant = det * swap_sign;
    return info;
}

// Calculate COND = ( norm(A) * norm(inv(A)) )
// This function should be used in conjunction with a linear system solver, where this checks the condition number before solving
// Computes the condition number estimate using a 1 norm or infinity norm
// norm type is either '1' or 'I' for 1-norm or infinity norm, respectively
// A is n x n matrix
// P is n array for permutation of PLU
int lapacke_cond(char norm_type, double* A, int* P, double* cond, const int n, const bool with_warnings = true)
{
    double anorm;
    int info;
    double rcond;
    // Computes the norm of A
    anorm = LAPACKE_dlange(LAPACK_ROW_MAJOR, norm_type, n, n, A, n);
    // Modifies A in place with an LU decomposition
    info = lapacke_plu(A, P, n, n, with_warnings);
    if(with_warnings)
        if (info != 0) printf("failure in lapacke_cond due to PLU with error %d\n", info);
    // Computes the reciprocal norm
    info = LAPACKE_dgecon(LAPACK_ROW_MAJOR, norm_type, n, A, n, anorm, &rcond);
    if (info != 0) printf("failure in lapacke_cond due to LAPACKE_dgecon with error %d\n", info);
    *cond = 1.0 / rcond;
    return info;
}


/*
------  END OF LIBRARY FUNCTIONS ---------
*/
#endif //_LAPACKE_LINALG_HPP_
