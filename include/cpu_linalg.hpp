#ifndef _CPU_LINALG_HPP_
#define _CPU_LINALG_HPP_

#include"cauchy_types.hpp"
#include<float.h>
#include<lapacke.h>
#include<assert.h>
//*/
/*
typedef double __complex__ C_COMPLEX_TYPE;
#define MKL_Complex16 C_COMPLEX_TYPE
#include<mkl_lapacke.h>
#include <iomanip>
#include <limits>
*/
/*
------  BEGIN HELPER FUNCTIONS ---------
These functions currently only support double, can be easily modified to arbitrary type using templates
*/

void val_swap(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

void val_swap(double *a, double *b)
{
    double tmp = *a;
    *a = *b;
    *b = tmp;
}

void print_mat(double **A, const int M, const int N)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            double val = A[i][j];
            if( fabs(val) < 1e-14 )
                val = 0.0;
            std::cout << val << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_mat(double *A, const int M, const int N)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            double val = A[i*N + j];
            if( fabs(val) < 1e-14 )
                val = 0.0;
            std::cout << val << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_mat(double *A, const int M, const int N, const int precision)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            printf("%.*E, ", precision, A[i*N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void print_mats(double *A, const int Nt, const int M, const int N, const int precision)
{
    for(int k = 0; k < Nt; k++)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                printf("%.*E, ", precision, A[k*M*N + i*N + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

void print_mats(double *A, const int Nt, const int M, const int N)
{
    for(int k = 0; k < Nt; k++)
    {
        for(int i = 0; i < M; i++)
        {
            for(int j = 0; j < N; j++)
            {
                printf("%lf, ", A[k*M*N + i*N + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

void print_cmat(C_COMPLEX_TYPE *A, const int M, const int N, const int precision = 16)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            double rval = creal(A[i*N + j]);
            double ival = cimag(A[i*N + j]);
            printf("%.*lf + %.*lfj, ", precision, rval, precision, ival);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void print_mat(int *A, const int M, const int N)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            std::cout << A[i*N + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Prints A assuming data is stored in column major ordering
void print_mat_colmaj(double* A, const int m, const int n)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            printf("%lf, ", A[j*m + i]);
        }
        printf("\n");
    }
    printf("\n");
}


void print_vec(double *x, const int M)
{
    for(int i = 0; i < M; i++)
    {
        double val = x[i];
        if( fabs(val) < 1e-14 )
            val = 0.0;
        if(i < M - 1)
            std::cout << val << ", ";
        else
            std::cout << val;
    }
    std::cout << std::endl;
    std::cout << std::endl;
}

void print_vec(double *x, const int N, const int precision)
{
    for(int j = 0; j < N; j++)
    {
        printf("%.*E, ", precision, x[j]);
    }
    printf("\n");
}

void rand_mat(double **mat, const int M, const int N)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            mat[i][j] = (double) (rand())  / (double) (RAND_MAX);
        }
    }
}

void rand_mat(double *mat, const int M, const int N)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            mat[i*N + j] = (double) (rand()+1)  / (double) (RAND_MAX);
        }
    }
}

void rand_vec(double *vec, const int M)
{
    for(int i = 0; i < M; i++)
    {
        vec[i] = (double) (rand())  / (double) (RAND_MAX);
    }
}

double rand_num()
{
    return (double) (rand())  / (double) (RAND_MAX);
}

void clear_strictly_lower(double *A, const int M, const int N)
{
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            if(i > j)
                A[i*N + j] = 0.0;
}

void clear_strictly_upper(double *A, const int M, const int N)
{
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            if(i < j)
                A[i*N + j] = 0.0;
}



/*
------  END OF HELPER FUNCTIONS ---------


------  START OF LIBRARY FUNCTIONS ---------
*/

/*
// A very nice cholesky which returns the lower triangular L matrix explicitly
double *cholesky(double *A, int n) 
{
    double *L = (double*)calloc(n * n, sizeof(double));
    if (L == NULL)
        exit(EXIT_FAILURE);
 
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i+1); j++) {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (i == j) ?
                           sqrt(A[i * n + i] - s) :
                           (1.0 / L[j * n + j] * (A[i * n + j] - s));
        }
 
    return L;
}
*/

// overwrites the lower part of the A matrix, clears upper if clear_upper set to true
void cholesky(double *A, const int n, bool clear_upper = false) 
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i+1); j++) 
        {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += A[i * n + k] * A[j * n + k];
            A[i * n + j] = (i == j) ?
                           sqrt(A[i * n + i] - s) :
                           (1.0 / A[j * n + j] * (A[i * n + j] - s));
        }
    if(clear_upper)
    {
        for(int i = 0; i < n; i++)
            for(int j = i+1; j < n; j++)
                A[i*n + j] = 0.0;
    }
}

// B = A.T @ A
void inner_mat_prod(double *A, double *B, const int M, const int N)
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            double sum = 0.0;
            for(int k = 0; k < M; k++)
            {
                sum += A[i+ k*N] * A[j + k*N];
            }
            B[i*N + j] = sum;
        }
    }
}

// B = A @ A.T
void outer_mat_prod(double *A, double *B, const int M, const int N)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < M; j++)
        {
            double sum = 0.0;
            for(int k = 0; k < N; k++)
            {
                sum += A[i*N + k] * A[j*N + k];
            }
            B[i*M + j] = sum;
        }
    }
}

// C = A @ B.T
void outer_mat_prod(double *A, double *B, double *C, const int M, const int N)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < M; j++)
        {
            double sum = 0.0;
            for(int k = 0; k < N; k++)
            {
                sum += A[i*N + k] * B[j*N + k];
            }
            C[i*M + j] = sum;
        }
    }
}

// Computes C = A @ B
// To compute C = A @ B.T, set the is_B_trasnposed flag
// To compute C = A.T @ B, set the is_A_transposed flag
// To compute C = A.T @ B.T, set both is_A_transposed and is_B_transposed
void matmatmul(const double *A, const double *B, double *C, const int MA, const int NA, const int MB, const int NB, bool is_A_transposed = false, bool is_B_transposed = false)
{
    // Four Cases Here:
    // Case 1: C = A @ B -> Matrix multiplication with neither A nor B transposed
    if(!is_A_transposed && !is_B_transposed)
    {
        assert(NA == MB);
        for(int i = 0; i < MA; i++) // MA
        {
            for(int j = 0; j < NB; j++) // NB
            {
                double sum = 0.0;
                for(int k = 0; k < NA; k++)
                {
                    sum += A[i * NA + k] * B[j + k * NB];
                }
                C[i * NB + j] = sum;
            }
        }
    }
    // Case 2: C = A @ B.T -> Matrix multiplication with B transposed
    if(!is_A_transposed && is_B_transposed)
    {
        assert(NA == NB);
        for(int i = 0; i < MA; i++) 
        {
            for(int j = 0; j < MB; j++) 
            {
                double sum = 0.0;
                for(int k = 0; k < NA; k++)
                {
                    sum += A[i * NA + k] * B[k + j * NB];
                }
                C[i * MB + j] = sum;
            }
        }
    }
    // Case 3: C = A.T @ B -> Matrix multiplication with A transposed
    if(is_A_transposed && !is_B_transposed)
    {
        assert(MA == MB);
        for(int i = 0; i < NA; i++)
        {
            for(int j = 0; j < NB; j++)
            {
                double sum = 0.0;
                for(int k = 0; k < MA; k++)
                {
                    sum += A[k * NA + i] * B[j + k * NB];
                }
                C[i * NB + j] = sum;
            }
        }
    }
    // Case 4: C = A.T @ B.T -> Matrix multiplication with A and B transposed
    if(is_A_transposed && is_B_transposed)
    {
        assert(MA == NB);
        for(int i = 0; i < NA; i++) // MA
        {
            for(int j = 0; j < MB; j++) // NB
            {
                double sum = 0.0;
                for(int k = 0; k < MA; k++)
                {
                    sum += A[k * NA + i] * B[k + j * NB];
                }
                C[i * MB + j] = sum;
            }
        }
    }
}

// Computes C = A @ B
// To compute C = A @ B.T, set the is_B_trasnposed flag
// To compute C = A.T @ B, set the is_A_transposed flag
// To compute C = A.T @ B.T, set both is_A_transposed and is_B_transposed
void matmatmul(const C_COMPLEX_TYPE *A, const C_COMPLEX_TYPE *B, C_COMPLEX_TYPE *C, const int MA, const int NA, const int MB, const int NB, bool is_A_transposed = false, bool is_B_transposed = false)
{
    // Four Cases Here:
    // Case 1: C = A @ B -> Matrix multiplication with neither A nor B transposed
    if(!is_A_transposed && !is_B_transposed)
    {
        assert(NA == MB);
        for(int i = 0; i < MA; i++) // MA
        {
            for(int j = 0; j < NB; j++) // NB
            {
                C_COMPLEX_TYPE sum = 0.0;
                for(int k = 0; k < NA; k++)
                {
                    sum += A[i * NA + k] * B[j + k * NB];
                }
                C[i * NB + j] = sum;
            }
        }
    }
    // Case 2: C = A @ B.T -> Matrix multiplication with B transposed
    if(!is_A_transposed && is_B_transposed)
    {
        assert(NA == NB);
        for(int i = 0; i < MA; i++) 
        {
            for(int j = 0; j < MB; j++) 
            {
                C_COMPLEX_TYPE sum = 0.0;
                for(int k = 0; k < NA; k++)
                {
                    sum += A[i * NA + k] * B[k + j * NB];
                }
                C[i * MB + j] = sum;
            }
        }
    }
    // Case 3: C = A.T @ B -> Matrix multiplication with A transposed
    if(is_A_transposed && !is_B_transposed)
    {
        assert(MA == MB);
        for(int i = 0; i < NA; i++)
        {
            for(int j = 0; j < NB; j++)
            {
                C_COMPLEX_TYPE sum = 0.0;
                for(int k = 0; k < MA; k++)
                {
                    sum += A[k * NA + i] * B[j + k * NB];
                }
                C[i * NB + j] = sum;
            }
        }
    }
    // Case 4: C = A.T @ B.T -> Matrix multiplication with A and B transposed
    if(is_A_transposed && is_B_transposed)
    {
        assert(MA == NB);
        for(int i = 0; i < NA; i++) // MA
        {
            for(int j = 0; j < MB; j++) // NB
            {
                C_COMPLEX_TYPE sum = 0.0;
                for(int k = 0; k < MA; k++)
                {
                    sum += A[k * NA + i] * B[k + j * NB];
                }
                C[i * MB + j] = sum;
            }
        }
    }
}

// Computes y = A @ x 
// To compute y = A.T @ x, set the is_A_transposed flag 
void matvecmul(const double *A, const double *x, double *y, const int M, const int N, bool is_A_transposed = false)
{
    // Regular y = A x
    if(!is_A_transposed)
    {
        for(int i = 0; i < M; i++)
        {
            double sum = 0.0;
            for(int j = 0; j < N; j++)
            {
                sum += A[i * N + j] * x[j];
            }
            y[i] = sum;
        }
    }
    // Case y = A.T @ x -> index A as if it is transposed. A in (MxN), index as NxM
    else
    {
        for(int i = 0; i < N; i++)
        {
            double sum = 0.0;
            for(int j = 0; j < M; j++)
            {
                sum += A[i + j*N] * x[j];
            }
            y[i] = sum;
        }
    }
}

// Computes y = A @ x 
// To compute y = A.T @ x, set the is_A_transposed flag 
void matvecmul(const C_COMPLEX_TYPE *A, const C_COMPLEX_TYPE *x, C_COMPLEX_TYPE *y, const int M, const int N, bool is_A_transposed = false)
{
    // Regular y = A x
    if(!is_A_transposed)
    {
        for(int i = 0; i < M; i++)
        {
            C_COMPLEX_TYPE sum = 0.0;
            for(int j = 0; j < N; j++)
            {
                sum += A[i * N + j] * x[j];
            }
            y[i] = sum;
        }
    }
    // Case y = A.T @ x -> index A as if it is transposed. A in (MxN), index as NxM
    else
    {
        for(int i = 0; i < N; i++)
        {
            C_COMPLEX_TYPE sum = 0.0;
            for(int j = 0; j < M; j++)
            {
                sum += A[i + j*N] * x[j];
            }
            y[i] = sum;
        }
    }
}

// Multiplies U @ x and stores in b for an upper triangular matrix U (i.e, the upper triangular part)
// U is N x N
// x and b are N
void trmv_DZ_mul(double* U, C_COMPLEX_TYPE *x, C_COMPLEX_TYPE *b, const int N)
{
    for(int i = 0; i < N; i++)
    {
        b[i] = 0;
        for(int j = i; j < N; j++)
            b[i] += U[i*N + j] * x[j];
    }
}

// Computes x^T*y
double dot_prod(const double* x, const double* y, const int n)
{
    double z = 0.0;
    for(int i = 0; i < n; i++)
        z += x[i] * y[i];
    return z;
}

// ||x||_2
double norm(double* x, const int n)
{
    return sqrt(dot_prod(x, x, n));
}

// x *= scale
void scale_vec(double* x, const double scale, const int n)
{
    for(int i = 0; i < n; i++)
        x[i] *= scale;
}

// x *= scale
void scale_vec(C_COMPLEX_TYPE* x, const double scale, const int n)
{
    for(int i = 0; i < n; i++)
        x[i] *= scale;
}

// y = scale*x
void scale_vec(double* y, double* x, const double scale, const int n)
{
    for(int i = 0; i < n; i++)
        y[i] = x[i]*scale;
}

// x = x \circ scale (scale is a vec as well)
void scale_vec(double* x, double* scale_vec, const int n)
{
    for(int i = 0; i < n; i++)
        x[i] *= scale_vec[i];
}

// z = x + scale*y
void add_vecs(const double* x, const double* y, double* z, const int n, double scale = 1.0)
{
    for(int i = 0; i < n; i++)
        z[i] = x[i] + scale*y[i];
}

// x += scale*y
void add_vecs(double* x, const double* y, const int n, double scale = 1.0)
{
    for(int i = 0; i < n; i++)
        x[i] += scale*y[i];
}
// z = x - y
void sub_vecs(const double* x, const double* y, double* z, const int n)
{
    for(int i = 0; i < n; i++)
        z[i] = x[i] - y[i];
}

// x -= y
void sub_vecs(double* x, const double* y, const int n)
{
    for(int i = 0; i < n; i++)
        x[i] -= y[i];
}

// x -= y
void sub_vecs(C_COMPLEX_TYPE* x, const C_COMPLEX_TYPE* y, const int n)
{
    for(int i = 0; i < n; i++)
        x[i] -= y[i];
}

double sum_vec(double* x, const int n)
{
    double sum = 0;
    for(int i = 0; i < n; i++)
        sum += x[i];
    return sum;
}

template<typename T>
T array_max(T* arr, const int N)
{
    if(N==1)
        return arr[0];
    else
    {
        T max_val = arr[0];
        for(int i = 1; i < N; i++)
            if(arr[i] > max_val)
                max_val = arr[i];
        return max_val;
    }
}

template<typename T>
T array_min(T* arr, const int N)
{
    if(N==1)
        return arr[0];
    else
    {
        T min_val = arr[0];
        for(int i = 1; i < N; i++)
            if(arr[i] < min_val)
                min_val = arr[i];
        return min_val;
    }
}

// Finds the maximum imaginary part value of a complex array
double carray_max_imag(C_COMPLEX_TYPE* arr, const int N)
{
    if(N==1)
        return cimag(arr[0]);
    else
    {
        double imag_max = cimag(arr[0]);
        double val;
        for(int i = 1; i < N; i++)
        {
            val = cimag(arr[i]);
            if(val > imag_max)
                imag_max = val;
        }
        return imag_max;
    }
}

int carray_argmax(C_COMPLEX_TYPE* arr, const int N)
{
    if(N==1)
        return 0;
    else
    {
        double max_val = cabs(arr[0]);
        int arg_max_val = 0;
        double val;
        for(int i = 1; i < N; i++)
        {
            val = cabs(arr[i]);
            if(val > max_val)
            {
                arg_max_val = i;
                max_val = val;
            }
        }
        return arg_max_val;
    }
}

int carray_vector_argmax(C_COMPLEX_TYPE* arr, const int N, const int m)
{
    if(N==1)
        return 0;
    else
    {
        int arg_max_sum = 0;
        double max_sum = 0;
        for(int j = 0; j < m; j++)
            max_sum += cabs(arr[j]);
        double val_sum;
        for(int i = 1; i < N; i++)
        {
            val_sum = 0;
            C_COMPLEX_TYPE* vec = arr + i*m;
            for(int j = 0; j < m; j++)
            {
                val_sum += cabs(vec[j]);
            }
            if(val_sum > max_sum)
            {
                arg_max_sum = i;
                max_sum = val_sum;
            }
        }
        return arg_max_sum;
    }
}

int carray_argmin(C_COMPLEX_TYPE* arr, const int N)
{
    if(N==1)
        return 0;
    else
    {
        double min_val = cabs(arr[0]);
        int arg_min_val = 0;
        double val;
        for(int i = 1; i < N; i++)
        {
            val = cabs(arr[i]);
            if(val < min_val)
            {
                arg_min_val = i;
                min_val = val;
            }
        }
        return arg_min_val;
    }
}

int carray_vector_argmin(C_COMPLEX_TYPE* arr, const int N, const int m)
{
    if(N==1)
        return 0;
    else
    {
        int arg_min_sum = 0;
        double min_sum = 0;
        for(int j = 0; j < m; j++)
            min_sum += cabs(arr[j]);
        double val_sum;
        for(int i = 1; i < N; i++)
        {
            val_sum = 0;
            C_COMPLEX_TYPE* vec = arr + i*m;
            for(int j = 0; j < m; j++)
            {
                val_sum += cabs(vec[j]);
            }
            if(val_sum < min_sum)
            {
                arg_min_sum = i;
                min_sum = val_sum;
            }
        }
        return arg_min_sum;
    }
}


void scale_mat(double *A, const double scale, const int M, const int N)
{
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            A[i*N + j] *= scale;
}

// B = A \circ scale_vec.
// \circ is the hadamard product.
// Can scale the rows / columns of A with scale_vec and store in B
void scale_mat(double* B, double *A, const double* scale_vec, const int M, const int N, bool is_scale_rows = true)
{
    if(is_scale_rows)
        for(int i = 0; i < M; i++)
            for(int j = 0; j < N; j++)
                B[i*N+j] =  A[i*N+j] * scale_vec[j];
    else
        for(int i = 0; i < N; i++)
            for(int j = 0; j < M; j++)
                B[i+j*N] =  A[i+j*N] * scale_vec[j];
}

// A = A \circ scale_vec.
// \circ is the hadamard product.
// Can scale the rows / columns of A with scale_vec and store in A
void scale_mat(double *A, const double* scale_vec, const int M, const int N, bool is_scale_rows = true)
{
    if(is_scale_rows)
        for(int i = 0; i < M; i++)
            for(int j = 0; j < N; j++)
                A[i*N+j] *= scale_vec[j];
    else
        for(int i = 0; i < N; i++)
            for(int j = 0; j < M; j++)
                A[i+j*N] *= scale_vec[j];
}

// A = A \circ scale_vec.
// \circ is the hadamard product.
// Can scale the rows / columns of A with scale_vec and store in A
void scale_mat(C_COMPLEX_TYPE *A, const C_COMPLEX_TYPE* scale_vec, const int M, const int N, bool is_scale_rows = true)
{
    if(is_scale_rows)
        for(int i = 0; i < M; i++)
            for(int j = 0; j < N; j++)
                A[i*N+j] *= scale_vec[j];
    else
        for(int i = 0; i < N; i++)
            for(int j = 0; j < M; j++)
                A[i+j*N] *= scale_vec[j];
}

// B += scale*A
void add_mat(double* B, const double* A, const double scale, const int M, const int N)
{
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            B[i*N+j] += A[i*N+j] * scale;
}

// C = B + scale*A
void add_mat(double* C, const double* B, const double* A, double scale, const int M, const int N)
{
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            C[i*N+j] = B[i*N+j] + A[i*N+j] * scale;
}

// C = scale_B*B + scale_A*A
void add_mat(double* C, const double* B, const double* A, const double scaleB, const double scaleA, const int M, const int N)
{
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            C[i*N+j] = scaleB*B[i*N+j] + A[i*N+j] * scaleA;
}

// Computes sign function 
double sgn(double x)
{
    return (x > 0) - (x < 0);
}

// Explicit Transpose of the matrix A. Returns B = A.T
double* transpose(double *A, const int M, const int N)
{
    double* A_T = (double*)malloc(M*N*sizeof(double));
    null_ptr_check(A_T);
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            A_T[j*M+i] = A[i*N+j];
    return A_T;
}

// Explicit Transpose of the matrix A. Returns B = A.T
void transpose(double *A, double* A_T, const int M, const int N)
{
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            A_T[j*M+i] = A[i*N+j];
}

// In place transpose
void reflect_array(double *A, const int M, const int N)
{
    for(int i = 0; i < M; i++)
        for(int j = i; j < N; j++)
            val_swap( &(A[i*N+j]), &(A[j*N + i]) );
}

// Embeds A \in (MAxNA) into B \in (MBxNB) at starting point eR,eC
void embed_matrix(double* B, double* A, int MA, int NA, int MB, int NB, int eR, int eC)
{
    // Make sure the matrix being inserted does not overflow
    //assert(eR*NB + eC + MA*NB + NA < MB*NB);
    assert(eC+NA <= NB);
    assert(eR+MA <= MB);

    for(int i = 0; i < MA; i++)
        memcpy(B + eR*NB + eC + i*NB, A + i*NA, NA*sizeof(double));
        //for(int j = 0; j < NA; j++)
        //    B[eR*NB + eC + i*NB +j] = scale*A[i*NA + j];
        

}

// slices A \in (MAxNA) out of B \in (MBxNB) at starting point eR,eC
void unembed_matrix(double* A, double* B, int MA, int NA, int MB, int NB, int eR, int eC)
{
    assert(eC+NA <= NB);
    assert(eR+MA <= MB);
    for(int i = 0; i < MA; i++)
        memcpy(A + i*NA, B + eR*NB + eC + i*NB, NA*sizeof(double));
        //memcpy(B + eR*NB + eC + i*NB, A + i*NA, NA*sizeof(double));
        //for(int j = 0; j < NA; j++)
        //    A[i*NA + j] = scale * B[eR*NB + eC + i*NB +j];
}

// forward solves the system of equations b = Lx 
// Note: L can (possibly) be in row-wise upper triag form -> ie: the matrix is stored in memory as a U and not a L.
// This is for the case the user wishes to solve b = U.T @ x but does not wish to explicitly transpose U
// If this is so, is_lower must be set to false
void forward_solve(double *L, double *b, double *x, const int M, const int N, bool is_lower = true)
{
    // Normal Case. L is stored row-wise lower triangular in memory
    if(is_lower)
    {
        int top = M >= N ? N : M;
        for(int i = 0; i < top; i++) // M
        {
            double sol = b[i];
            if(i > 0)
            {
                for(int j = 0; j < i; j++)
                    sol -= L[i*N + j] * x[j];
            }
            x[i] = sol / L[i*N + i];
        }
        for(int i = M; i < N; i++)
            x[i] = 0.0;
    }

    // If the lower traingular matrix L is stored row-wise upper-triangular in memory -> ie stored as a "U". 
    // This is so a transpose operation is not needed to bring U into lower traigular form L.
    // we perform column wise forward sub from top row to bottom row (down the columns) and from left to right.
    if(!is_lower)
    {
        int top = M >= N ? N : M;
        for(int i = 0; i < top; i++)
        {
            double sol = b[i];
            if(i > 0)
            {
                for(int k = 0; k < i; k++)
                    sol -= L[i + N*k] * x[k];
            }
            x[i] = sol / L[i*N + i];
        }
        for(int i = N; i < M; i++)
            x[i] = 0.0;
    }

}

// back solves the system of equations b = Ux 
// Note: U can (possibly) be in row-wise lower triag form -> ie: the matrix is stored in memory as a L and not a U
// This is for the case the user wishes to solve b = L.T @ x but does not wish to explicitly transpose L
// If this is so, is_upper must be set to false
void back_solve(double *U, double *b, double *x, const int M, const int N, bool is_upper = true)
{
    //if(is_upper)
    //    assert(M <= N);
    //if(!is_upper)
    //    assert(M >= N);

    if(is_upper)
    {
        int top = M >= N ? N-1 : M-1;
        for(int i = top; i >= 0; i--)
        {
            double sol = b[i];
            if(i < top)
                for(int j = top; j > i; j--)
                    sol -= U[i * N + j] * x[j];
            x[i] = sol / U[i * N + i];
        }
        for(int i = M; i < N; i++)
            x[i] = 0.0;
    }
    if(!is_upper)
    {
        int top = M >= N ? N-1 : M-1;
        for(int i = top; i >= 0; i--)
        {
            double sol = b[i];
            if(i < top)
                for(int j = top; j > i; j--)
                    sol -= U[i + N*j] * x[j];
            x[i] = sol / U[i*N + i];
        }
        for(int i = N; i < M; i++)
            x[i] = 0.0;
    }
}

// If permutations were involved in forming the upper triagular matrix due to column swaps in gaussian elimination, then use this version of backsolve
void back_solve(double *U, double *b, double *x, int* P, const int M, const int N, bool is_upper = true)
{
    assert(is_upper); // until transpose is tested

    int top = M >= N ? N-1 : M -1;
    double tmp[N];
    for(int i = top; i >= 0; i--)
    {
        double sol = b[i];
        if(i < top)
            for(int j = top; j > i; j--)
                sol -= U[i * N + j] * tmp[j];
        tmp[i] = sol / U[i * N + i];
    }
    for(int i = M; i < N; i++)
        tmp[i] = 0.0;    
    for(int i = 0; i < N; i++)
        x[P[i]] = tmp[i];

}

// Gaussian Elimination of the system b = Ax
// the matrix A and vector b are explicitly modified into reduced row-echelon form
// this version allows matrix A to be passed in as a regular pointer
// if row swap criterion is needed, the two rows are explicitly swapped via val_swap function
int gauss_elim(double *A, double *b, int* P, const int M, const int N, double eps = 1e-12)
{
    int piv_idx;
    double piv_val;
    double a;
    assert(M <= N);
    for(int i = 0; i < N; i++)
        P[i] = i; // Permutation Matrix

    for(int i = 0; i < M; i++)
    {
        //row-swap criterion here
        piv_idx = -1;
        piv_val = eps;
        for(int j = i; j < N; j++)
        {
            a = fabs(A[i*N+j]);
            if( a > piv_val )
            {    
                piv_idx = j;
                piv_val = a;
            }
        }
        if(piv_idx == -1)
            return 1;
        if(piv_idx != i)
        {
            for(int j = 0; j < M; j++)
                val_swap(&A[j*N + i], &A[j*N + piv_idx]);
            val_swap(&P[i], &P[piv_idx]);
        }
        for(int j = i+1; j < M; j++)
        {
            double ratio = A[j*N + i] / A[i*N + i];
            b[j] -= ratio * b[i];
            for(int k = i; k < N; k++)
                A[j*N + k] -=  (ratio * A[i*N + k]);
        }
    }
    return 0;
}

// Gaussian Elimination of the system b = Ax
// the matrix A and vector b are explicitly modified into reduced row-echelon form
// this version allows matrix A to be passed in as a regular pointer
// if row swap criterion is needed, the two rows are explicitly swapped via val_swap function
int gauss_elim(double *A, double *b, const int n, double eps = 1e-12)
{
    int piv_idx;
    double piv_val, a;
    double work[n];
    for(int i = 0; i < n; i++)
    {
        //row-swap criterion here
        piv_idx = -1;
        piv_val = eps;
        for(int j = i; j < n; j++)
        {
            a = fabs(A[j*n+i]);
            if( a > piv_val )
            {    
                piv_idx = j;
                piv_val = a;
            }
        }
        if(piv_idx == -1)
            return 1;
        if(piv_idx != i)
        {
            // Full row swap below not needed 
            // Only partial row swap, but since overhead is so small...
            // the readibility of the section increases
            memcpy(work, A + piv_idx*n, n*sizeof(double));
            memcpy(A + piv_idx*n, A + i*n, n*sizeof(double));
            memcpy(A + i*n, work, n*sizeof(double));
            val_swap(&b[i], &b[piv_idx]);
        }
        for(int j = i+1; j < n; j++)
        {
            double ratio = A[j*n + i] / A[i*n + i];
            b[j] -= ratio * b[i];
            for(int k = i; k < n; k++)
                A[j*n + k] -=  (ratio * A[i*n + k]);
        }
    }
    return 0;
}

// Solves A @ x = b for the overdetermined system (i.e: M > N)
// Via gram equations
void solve_least_square(double *A, double *b, double *x, const int M, const int N)
{
    assert(M >= N);
    // Helper matrix B and vectors b_bar, y
    double* B = (double*) malloc(N*N*sizeof(double));
    null_ptr_check(B);
    double* b_bar = (double*) malloc(N*sizeof(double));
    null_ptr_check(b_bar);
    double* y = (double*) malloc(N*sizeof(double));
    null_ptr_check(y);

    // Solving overdetermined system b = Ax
    //printf("b_bar = A.T @ b is:\n");
    matvecmul(A, b, b_bar, M, N, true);
    //print_vec(b_bar, N);
    //printf("Inner Matrix Product B = A.T @ A\n");
    inner_mat_prod(A, B, M, N);
    //print_mat(B, N, N);
    //printf("Cholesky B = L @ L.T is:\n");
    cholesky(B, N, true);
    //print_mat(B, N, N);
    //printf("Forward solve b_bar = L @ y is:\n");
    forward_solve(B, b_bar, y, N, N);
    //print_vec(y, N);
    //printf("Back solve y = B.T @ x yields our solution x: \n");
    back_solve(B, y, x, N, N, false);
    //print_vec(x, N);
    free(B);
    free(b_bar);
    free(y);
}

// Solves A @ x = b for the underdetermined system (i.e: M < N)
// via gram equations
void solve_least_norm(double *A, double *b, double *x, const int M, const int N)
{
    assert(M <= N);
    // helper vectors for intermediate steps
    double *y = (double*) malloc(M * sizeof(double));
    null_ptr_check(y);
    double *t = (double*) malloc(M * sizeof(double));
    null_ptr_check(t);
    double *B = (double*) malloc(M * M * sizeof(double)); //[M*M];
    null_ptr_check(B);

    //printf("Outer Product B = A @ A.T is:\n");
    outer_mat_prod(A, B, M, N);    
    //printf("Cholesky of B = L @ L.T is:\n");
    cholesky(B, M, true);    
    //printf("Forward Solve of b = L @ y is:\n");
    forward_solve(B, b, y, M, M); // b = B @ y    
    //printf("Back Solve of y = L.T @ t is:\n");
    back_solve(B, y, t, M, M, false); // y = B.T @ t    
    //printf("The Solution x = A.T @ t is:\n");
    matvecmul(A, t, x, M, N, true); // x = A.T @ t
    free(y);
    free(t);
    free(B);
}

// Factors A as A = PLU, returning P in lapacke permutation indexing.. ie. [2,2,2] means [2,0,1]
int PLU(double* A, int *P, const int n, double tol = 1e-16) 
{
  int i, j, k, pivot_ind;
  double pivot;
  double temp;
  double temp_row[n]; 

  for (j = 0; j < n; ++j) 
  {
    pivot = tol;
    pivot_ind = -1;
    for (i = j; i < n; ++i)
    {
      if (fabs(A[i*n+j]) > fabs(pivot))
      {
        pivot = A[i*n+j];
        pivot_ind = i;
      }
    }
    if(pivot_ind == -1)
        return 1;
    if(pivot_ind != j)
    {
        memcpy(temp_row, A+j*n, n*sizeof(double));
        memcpy(A+j*n, A+pivot_ind*n, n*sizeof(double));
        memcpy(A+pivot_ind*n, temp_row, n*sizeof(double));
    }

    P[j] = pivot_ind;

    for(k = j+1; k < n; ++k) 
    {
        A[k*n+j] /= A[j*n+j];
        temp = A[k*n+j];
        for(int q=j+1;q<n;q++)
        {
            A[k*n+q] -= temp*A[j*n+q];
        }
    }
  }
  return 0;
}

// Use these for PLU: Simple backsolve using packed L and U
void back_solve(const double *LU, double *b, const int n)
{
    for(int i = n-1; i >= 0; i--)
    {
        double sol = b[i];
        for(int j = n-1; j > i; j--)
            sol -= LU[i * n + j] * b[j];
        b[i] = sol / LU[i * n + i];
    }
}

// Use these for PLU: Simple forward solve using packed L and U (i.e, enteries of L on diagonal are 1)
void forward_solve(const double *LU, double *b, int n)
{
    for(int i = 0; i < n; i++) 
    {
        double sol = b[i];
        for(int j = 0; j < i; j++)
            sol -= LU[i*n + j] * b[j];
        b[i] = sol;
    }
}

// Takes a lapack-like permutation index array and forms the regular permutation array... ie: [2,2,2] is now [2,0,1]
void lapack_to_regular_permutation_array(const int* P, int* Preg, const int n)
{
    for(int i = 0; i < n; i++)
        Preg[i] = i;
    // Preg now holds the Permutation in regular form
    for(int i = 0; i < n; i++)
        val_swap(&Preg[i], &Preg[P[i]]);
}
// Takes a regular permutation index array and transposes it... i.e: [2,0,1] is now [1,2,0]
void transpose_permutation_array(const int* P, int* P_T, const int n)
{
    // P_T now hold the transpose of the permutation matrix P
    for(int i = 0; i < n; i++)
        P_T[P[i]] = i;
}

// Solves PLU @ x = b
// Inputs are the outputs of PLU (LU packed and P), along with a single right hand side b
// LU is n x n packed PLU decomposition
// P is the permutation array, given in lapack form
// b is the right hand side vector
// x is the solution vector
void solve_trf(const double* LU, const int* P, const double* b, double* x, const int n)
{
    int Preg[n];
    int P_T[n];
    lapack_to_regular_permutation_array(P, Preg, n);
    transpose_permutation_array(Preg, P_T, n);
    for(int i = 0; i < n; i++)
        x[P_T[i]] = b[i];
    forward_solve(LU, x, n);
    back_solve(LU, x, n);
}

double matrix_one_norm(double* A, const int m, const int n)
{
    double norm_val, max_norm_val;
    max_norm_val = -1;
    for(int i = 0; i < n; i++)
    {   
        norm_val = 0;
        for(int j = 0; j < m; j++)
            norm_val += fabs(A[j*n + i]);
        if(norm_val > max_norm_val)
            max_norm_val = norm_val;
    }
    return max_norm_val;
}

double matrix_inf_norm(double* A, const int m, const int n)
{
    double norm_val, max_norm_val;
    max_norm_val = -1;
    for(int i = 0; i < m; i++)
    {   
        norm_val = 0;
        for(int j = 0; j < n; j++)
            norm_val += fabs(A[i*n + j]);
        if(norm_val > max_norm_val)
            max_norm_val = norm_val;
    }
    return max_norm_val;
}

// Finds the condition number of a matrix by one norm or infinity norm approximation
// norm indicates whether the one ('1') or infinity ('inf') norm should be used. 
// A is the n x n matrix on input for condition number testing and the PLU factorization (packed) on the output
// work is a declared n x n memory array space, which holds the inverse of A on exit
// P is an n-array permutation matrix returned in lapack formation from the PLU factorization call
// tol indicates what number for PLU factorization is considered too small...and DBL_MAX is returned.
double cond(char norm, double* A, double* work, int* P, const int n, double tol)
{
    int Preg[n]; int P_T[n];
    int p, offset;
    double MAX_COND_NUM = DBL_MAX;
    double norm_val, inv_norm_val;
    // Find the 1 norm or inf norm of A
    if(norm == '1')
        norm_val = matrix_one_norm(A, n, n);
    else if(norm == 'I')
        norm_val = matrix_inf_norm(A, n, n);
    else
    {
        printf("Norm symbol entered was %c.\nEnter norm symbol of '1' for one norm and 'I' for infinity norm. Exiting!\n", norm);
        exit(1);
    }
    
    // Find A_inv and store it in work
    int info_plu = PLU(A, P, n, tol);
    if(info_plu == 1)
        return MAX_COND_NUM;
    memset(work, 0, n*n*sizeof(double));
    lapack_to_regular_permutation_array(P, Preg, n);
    transpose_permutation_array(Preg, P_T, n);
    for(int i = 0; i < n; i++)
    {
        p = P_T[i];
        offset = i*n + p;
        work[offset] = 1;
        forward_solve(A, work + i*n, n);
        back_solve(A, work + i*n, n);
    }
    reflect_array(work, n, n);
    
    // Find the 1 or inf norm of A_inv
    if(norm == '1')
        inv_norm_val = matrix_one_norm(work, n, n);
    else
        inv_norm_val = matrix_inf_norm(work, n, n);

    return norm_val * inv_norm_val;
}

// Computes the determinant of a matrix
// Can be done with PLU or gaussian elimination
double determinant(double* A, const int n, const bool use_gauss_elim = false)
{
    if(n == 1)
        return A[0];
    else if(n == 2)
        return A[0] * A[3] - A[1] * A[2];
    else if(n == 3)
    {
        double r1,r2,r3;
        r1 = 0;
        r2 = 0;
        r3 = 0;
        r1 = A[0] * ((A[4] * A[8]) 
        - (A[7] * A[5]));

        r2 = A[1] * ((A[3] * A[8]) 
        - (A[6] * A[5]));

        r3 = A[2] * ((A[3] * A[7]) 
        - (A[6] * A[4]));

        return r1 - r2 + r3;
    }
    else
    {
        if(use_gauss_elim)
        {
            double fake_b[n];
            //double* A_ge = (double*) malloc(n*n*sizeof(double)); // chhange this if we have large matrices
            double A_ge[n*n];
            int P[n];
            memcpy(A_ge, A, n*n*sizeof(double));
            gauss_elim(A_ge, fake_b, P, n, n);
            double sum = 1.0; 
            for(int i = 0; i < n; i++)
                sum *= A_ge[i*n + i];
            //free(A_ge);
            return sum;
        }
        else 
        {
            int P[n];
            PLU(A, P, n, 0);
            double det = 1;
            for(int i = 0; i < n; i++)
                det *= A[i*n+i];
            // Find how many swaps occured in the Permutation matrix
            int swaps = 0;
            for(int i = 0; i < n; i++)
                if(P[i] != i)
                    swaps += 1;
            int swap_sign = 1 - 2*(swaps % 2);
            return det * swap_sign;
        }

    }

}

// matrix inverse
int inv(const double* A, double* A_inv, double* work, const int n)
{
    int P[n];
    int P2[n];
    int p, offset;
    double tol = 1e-16;

    // Find A_inv and store it in work
    memcpy(work, A, n*n*sizeof(double));
    int info_plu = PLU(work, P, n, tol); // Factor A as A = P L U
    if(info_plu == 1)
        return info_plu;
    memset(A_inv, 0, n*n*sizeof(double));
    lapack_to_regular_permutation_array(P, P2, n);
    transpose_permutation_array(P2, P, n);
    for(int i = 0; i < n; i++)
    {
        p = P[i];
        offset = i*n + p;
        A_inv[offset] = 1;
        forward_solve(work, A_inv + i*n, n);
        back_solve(work, A_inv + i*n, n);
    }
    reflect_array(A_inv, n, n);
    return 0;
}

// Computes H_k @ x where H_k = I_k - 2v_k @ v_k.T
// overwrites solution in x
void householder_matvecmul(double* v, double* x, const int N, const int k)
{
    double dp = 0.0;
    for(int i = 0; i < N-k; i++)
        dp += v[i] * x[i+k];
    for(int i = 0; i < N-k; i++)
        x[i+k] -= 2*dp*v[i];
}

// Computes H_k @ A where H_k = I_k - 2v_k @ v_k.T across columns of A
// overwrites A 
void householder_matmatmul(double* v, double* A, const int M, const int N, const int k, bool is_A_transposed = false)
{
    if(!is_A_transposed)
    {
        for(int i = 0; i < N-k; i++)
        {
            double dp = 0.0;
            for(int j = 0; j < M-k; j++)
                dp += v[j]*A[k*N+k + i +j*N];
            for(int j = 0; j < M-k; j++)
                A[k*N+k+i+j*N] -= 2*dp*v[j];
        }
    }
    else 
    {
        for(int i = 0; i < M-k; i++)
        {
            double dp = 0.0;
            for(int j = 0; j < N-k; j++)
                dp += v[j]*A[k*N+k + i*N +j];
            for(int j = 0; j < N-k; j++)
                A[k*N+k+i*N+j] -= 2*dp*v[j];
        }
    }

}

// TODO SECTION: 
// (Householder) QR: https://rosettacode.org/wiki/QR_decomposition
// Takes the householder QR of an M by N matrix 
// Two cases: M >= N and M < N
// Both cases: overwrite A to be R
// If M >= N, performs the QR factorization 
    // R is [R;-0-] and reflectors Q are stored N x M
// If N > M, performs A = R.T @ Q.T
    // R.T is [R.T 0] and reflectors Q are stored M x N
void QR(double* A, double* Q, int M, int N)
{
    if(M >= N)
    {
        double v[M];
        double norm_v;
        for(int i = 0; i < N; i++)
        {
            // Create the M-i length reflector
            for(int j = 0; j < M-i; j++)
                v[j] = A[i*N + i + j*N];
            //printf("y:\n");
            //print_mat(v,M-i,1);
            norm_v = sqrt(dot_prod(v,v,M-i));
            v[0] += sgn(v[0]) * norm_v; 
            //printf("w:\n");
            //print_mat(v,M-i,1);
            norm_v = sqrt(dot_prod(v,v,M-i));
            for(int j = 0; j < M-i; j++)
            {
                v[j] /= norm_v;
                Q[i*M + j] = v[j]; 
            }
            for(int j = M-i; j < M; j++)
                Q[i*M + j] = 0.0;
            //printf("v:\n");
            //print_mat(v,M-i,1);
            householder_matmatmul(v, A, M, N, i);
            //printf("A:\n");
            //print_mat(A,M,N);
        }
        //printf("R:\n");
        //print_mat(A,M,N);
        //printf("Q:\n");
        //print_mat(Q,N,M);
    }
    else 
    {
        double v[N];
        double norm_v;
        for(int i = 0; i < M; i++)
        {
            // Create the M-i length reflector
            for(int j = 0; j < N-i; j++)
                v[j] = A[i*N + i + j];
            //printf("y:\n");
            //print_mat(v,N-i,1);
            norm_v = sqrt(dot_prod(v,v,N-i));
            v[0] += sgn(v[0]) * norm_v; 
            //printf("w:\n");
            //print_mat(v,N-i,1);
            norm_v = sqrt(dot_prod(v,v,N-i));
            for(int j = 0; j < N-i; j++)
            {
                v[j] /= norm_v;
                Q[i*N + j] = v[j];
            }
            for(int j = N-i; j < N; j++)
                Q[i*N + j] = 0.0;
            //printf("v:\n");
            //print_mat(v,N-i,1);
            householder_matmatmul(v, A, M, N, i, true);
            //printf("A:\n");
            //print_mat(A,M,N);
        }
        //printf("R.T:\n");
        //print_mat(A,M,N);
        //printf("Q:\n");
        //print_mat(Q,M,N);
    }

}

// QR factorization + solves of A
// solve: A @ X = B -> solves for X for many RHS Bs
// based on house holder QR and forward / backward subs
// handles underdetermined, overdetermined, and square systems using QR 
// A in M x N
// X in NRHS x N
// B in NRHS x M
void qr_solve(double* A, double* B, double* X, int M, int N, int NRHS)
{
    if(M >= N)
    {
        double* Q = (double*) malloc(M*N*sizeof(double));
        null_ptr_check(Q);
        QR(A, Q, M, N);
        for(int i = 0; i < NRHS; i++)
        {
            for(int j = 0; j < N; j++)
                householder_matvecmul(Q + M*j, B + i*M, M, j);
            back_solve(A, B + i*M, X + i*N , M, N);
        }
        free(Q);    
    }
    // Solve A^T = Q_1 @ R_1, then R_1_T @ z = b by back sub, then x = Q_1 @ z via householder muls
    else 
    { 
        double* Q = (double*) malloc(M*N*sizeof(double));
        null_ptr_check(Q);
        QR(A, Q, M, N);
        for(int i = 0; i < NRHS; i++)
        {
            forward_solve(A, B + i*M, X + i*N , M, N);
            for(int j = M-1; j >= 0; j--)
                householder_matvecmul(Q + N*j, X + i*N, N, j);
        }
        free(Q);
    }

}


// LDLT: https://sites.ualberta.ca/~xzhuang/Math381/Lab5.pdf (translate the matlab code)
void LDLT(double* A, double* L, double* D, const int N)
{
    memset(L, 0, N*N*sizeof(double));
    memset(D, 0, N*sizeof(double));
    for(int i = 0; i < N; i++)
        L[i*N + i] = 1;
    
    for(int i = 0; i < N; i++)
    {
        D[i] = A[i*N+i];
        for(int j = 0; j < i; j++)
            D[i] -=  L[i*N + j]*L[i*N + j]*D[j];
        for(int j = i+1; j < N; j++)
        {
            L[j*N + i] = A[j*N + i]/D[i];
            double tmp = 0.0;
            for(int k = 0; k < i; k++)
                tmp -= L[j*N + k]*L[i*N+k]*D[k];
            L[j*N + i] += tmp/D[i];
        }
    }
}

void ldlt_solve(double* A, double* B, double* X, const int N, const int NRHS)
{
    double* L = (double*) malloc(N*N*sizeof(double));
    null_ptr_check(L);
    double D[N];
    double z[N];

    // Factor
    LDLT(A, L, D, N);
    //printf("L:\n");
    //print_mat(L,N,N);
    //printf("D:\n");
    //print_mat(D,1,N);

    // Non-Singular Assertion of D before solve
    for(int j = 0; j < N; j++)
    {
        if(fabs(D[j]) < 1e-12)
        {
            printf("Singular System (j=%d). Cannot Solve!\n", j);
            return;
        }
    }    
    // Solve
    for(int i = 0; i < NRHS; i++)
    {
        forward_solve(L, B + i*N, z, N, N); // Lz = B[i]
        for(int j = 0; j < N; j++) // diag(D)w = z
            z[j] /= D[j];
        // L.T x = w
        back_solve(L, z, X + i*N, N, N, false);
    }
    free(L);  
}

// Solves a positive definite system of equations A @ X = B with 'A' positive definite and (possibly) multiple RHS 'B',
// A \in R^(n x n) - must be Positive Definite (pd),
// B \in R^(nrhs x n) - nrhs = number of right hand sides,
// X \in R^(nrhs x n) - each solution is stored in a row,
// Note that B and X are declared row-wise,
// This means that B should be transposed (so columns are rows) before calling this function,
// This also means that X is returned in transposed form (and needs transposing to yield A @ X = B for column-wise solutions X[:,i])
void solve_pd(double* A, double* B, double* X, const int n, const int nrhs)
{
    double y[n];
    cholesky(A, n);
    for(int i = 0; i < nrhs; i++)
    {
        forward_solve(A, B + i*n, y, n, n, true);
        back_solve(A, y, X + i*n, n, n, false);
    }
}

// Computes the right hand sided Auto-Correlation for wide-sense stationary processes
// The lag should be much smaller than the signal length
void wss_acorr(double* auto_corr, double* signal, const int len_signal, const int lag)
{
    assert(lag > 0);
    assert(len_signal > lag);
    for(int k = 0; k <= lag; k++)
    {
        const int n = len_signal - k;
        double sum = 0.0;
        for(int q = 0; q < n; q++)
        {
            sum += signal[k + q] * signal[q];
        }
        sum /= n;
        auto_corr[k] = sum;
    }
}


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
            //evals[i] = CMPLX(eig_rp[i], eig_ip[i]);
            evals[i] = eig_rp[i] + I*eig_ip[i];
        }
        for(int i = 0; i < n; i++)
        {
            if( fabs(eig_ip[i]) > eps )
            {
                for(int j = 0; j < n; j++)
                {
                    evecs[i + j*n] = V[i + j*n] + I*V[i+1 + j*n];
                    //evecs[i + j*n] = CMPLX(V[i + j*n], V[i+1 + j*n]);
                }
                for(int j = 0; j < n; j++)
                {
                    evecs[i+1 + j*n] = V[i + j*n] - I*V[i+1 + j*n];
                    //evecs[i + j*n] = CMPLX(V[i + j*n], -V[i+1 + j*n]);
                }
                i++;
            }
            else 
            {
                for(int j = 0; j < n; j++)
                {
                    evecs[i + j*n] = V[i + j*n] + 0.0*I;
                    //evecs[i + j*n] = CMPLX(V[i + j*n], 0.0);
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
#endif //_CPU_LINALG_HPP_
