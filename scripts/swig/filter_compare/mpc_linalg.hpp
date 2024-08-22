#ifndef _MPC_LINALG_HPP_
#define _MPC_LINALG_HPP_

#include "../../../include/cauchy_linalg.hpp"
#include "../../../include/lapacke_linalg.hpp"
#include "../../../include/random_variables.hpp"

class Mat;
class Tensor;
class MatArray;
void print_wrong_dim_mat(Mat* A, Mat* B);
void print_wrong_dim_matmul(Mat* A, Mat* B);
void print_wrong_dim_mat(const Mat& A, const Mat& B);
void print_wrong_dim_matmul(const Mat& A, const Mat& B);

double max(double x, double y)
{
    return x > y ? x : y;
}

double min(double x, double y)
{
    return x < y ? x : y;
}


struct SharedPtr
{
    private:

    public:

    int* count;
    double* ptr;

    SharedPtr()
    {
        count = NULL;
        ptr = NULL;
    }

    void init(double* data)
    {
        ptr = data;
        count = (int*) malloc( sizeof(int) );
        *count = 1;
    }

    void reinit(double* data)
    {
        ptr = data;
        if(count == NULL)
            count = (int*) malloc( sizeof(int) );
        *count = 1;
    }

    void null_init()
    {
        ptr = NULL;
        count = NULL;
    }
    
    void inc()
    {
        if(count != NULL)
        {
            if(ptr != NULL)
                *count += 1;
            else
            {
                printf("ATTEMPTING TO INC COUNT FOR NULL POINTER DATA!\n");
            }
        }
        else 
        {
            printf("ATTEMPTING TO INC NULL POINTER COUNT!\n");
        }
    }

    void dec()
    {
        if(count != NULL)
        {
            *count -= 1;
            if( *count <= 0 )
            {
                if(ptr != NULL)
                {
                    free(ptr);
                    ptr = NULL;
                }
                free(count);
                count = NULL;
            }
        }
        //else 
        //{
        //    printf("ATTEMPTING TO DEC NULL POINTER COUNT!\n");
        //}
    }

    ~SharedPtr()
    {
        dec();
    }
};

class Mat
{
    private:
    SharedPtr sh_ptr;
    friend class Tensor;
    friend class MatArray;
    public:
    int m, n;
    double* data;
    

    Mat(int m, int n = 1)
    {
        assert(  (m > 0) && (n > 0) );
        this->m = m;
        this->n = n;
        data = (double*) malloc(m*n*sizeof(double));
        sh_ptr.init(data);
        //memset(this->data, 0, this->m*this->n);
    }
    
    Mat()
    {
        this->m = 0;
        this->n = 0;
        data = NULL; //(double*) malloc(0);
        sh_ptr.null_init();
    }

    Mat(Mat* A)
    {
        this->m = A->m;
        this->n = A->n;
        data = (double*) malloc(m*n*sizeof(double));
        null_ptr_check(data);
        memcpy(data, A->data, this->m*this->n*sizeof(double));
        sh_ptr.init(data);
    }

    Mat(const Mat& A)
    {
        //std::cout << "Copy Constructor Called\n";
        this->m = A.m;
        this->n = A.n;
        data = (double*) malloc(m*n*sizeof(double));
        null_ptr_check(data);
        memcpy(data, A.data, this->m*this->n*sizeof(double));
        sh_ptr.init(data);
    }

    Mat(const double* c_data, const int m, const int n)
    {
        this->m = m;
        this->n = n;
        this->data = (double*) malloc(m*n*sizeof(double));
        null_ptr_check(data);
        memcpy(this->data, c_data, m * n * sizeof(double));
        sh_ptr.init(data);
    }

    Mat(const double* c_data, const int m)
    {
        this->m = m;
        this->n = 1;
        this->data = (double*) malloc(m*sizeof(double));
        null_ptr_check(data);
        memcpy(this->data, c_data, m * sizeof(double));
        sh_ptr.init(data);
    }

    void copy(const Mat& A)
    {
        if(&A == this)
            return;
        this->m = A.m;
        this->n = A.n;

        if(sh_ptr.count == NULL)
        {
            data = (double*) malloc(this->m*this->n*sizeof(double));
            memcpy(data, A.data, this->m*this->n*sizeof(double));
            null_ptr_check(data);
            sh_ptr.init(data);
        }
        else if(*sh_ptr.count == 1)
        {
            data = (double*) realloc(data, this->m*this->n*sizeof(double));
            memcpy(data, A.data, this->m*this->n*sizeof(double));
            null_ptr_check(data);
            sh_ptr.ptr = data;
        }
        else 
        {
            sh_ptr.dec();
            data = (double*) malloc(this->m*this->n*sizeof(double));
            memcpy(data, A.data, this->m*this->n*sizeof(double));
            null_ptr_check(data);
            sh_ptr.init(data);
        }
    }

    void init(int m, int n)
    {
        this->m = m;
        this->n = n;
        this->data = (double*) malloc(m*n*sizeof(double));
        sh_ptr.init(data);
    }

    Mat copy()
    {
        Mat A(this);
        return A;
    }

    // C = this.TF1 @ B.TF2
    Mat multiply(Mat& B, bool trans_flag1 = false, bool trans_flag2 = false)
    {
        int m = this->m;
        int n = B.n;
        int p = this->n;
        if(this->n != B.m)
            print_wrong_dim_mat(this, &B);
        Mat C(m,n);
        matmatmul(this->data, B.data, C.data, m,p,p,n, trans_flag1, trans_flag2);
        return C;
    }

     // this = this.TF1 @ B.TF2
    void multiply_inplace(Mat&B, bool trans_flag1 = false, bool trans_flag2 = false)
    {
        Mat C = multiply(B, trans_flag1, trans_flag2);
        this->copy(C);
    }

    Mat outer()
    {
        Mat B(this->m, this->m);
        outer_mat_prod(this->data, B.data, this->m, this->n);
        return B;
    }

    Mat inner()
    {
        Mat B(this->n, this->n);
        inner_mat_prod(this->data, B.data, this->m, this->n);
        return B;
    }

    // inner product assuming columns of matrix are orthogonal
    Mat inner_orth()
    {
        Mat A(this->n);
        A.zero();
        for(int j = 0; j < n; j++)
        {
            for(int i = 0; i < m; i++)
            {
                double val = this->data[i*n+j];
                A.data[j] += val*val;
            }
        } 
        return A;
    }
    
    // inner product assuming columns of matrix are orthogonal
    Mat outer_orth()
    {
        Mat A(this->m);
        A.zero();
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                double val = this->data[i*n+j];
                A.data[i] += val*val;
            }
        } 
        return A;
    }

    Mat solve(Mat B)
    {
        return (*this) / B;
    }

    Mat solve_triangular(Mat& B, bool is_lower)
    {
        assert(this->m == this->n);
        assert(this->m == B.m);
        Mat X(&B);
        if(is_lower)
            forward_solve_nrhs(this->data, B.data, X.data, n, B.n);
        else
            back_solve_nrhs(this->data, B.data, X.data, n, B.n);
        return X;
    }

    Mat chol()
    {
        assert(this->m == this->n);
        Mat A(this);
        // Assumes a PD matrix 
        cholesky(A.data, A.n, true);
        return A;
    }

    void chol_inplace()
    {
        assert(this->m == this->n);
        // Assumes a PD matrix 
        cholesky(this->data, this->n, true);
    }

    Mat PLU(Mat& P)
    {
        assert( P.m*P.n == min(this->m,this->n) );
        const int size_P = P.m*P.n;
        int _P[size_P];
        memset(_P, 0, size_P * sizeof(int) );
        Mat LU(this);
        lapacke_plu(LU.data, _P, LU.m, LU.n);
        for(int i = 0; i < size_P; i++)
            P.data[i] = (double)_P[i];
        return LU;
    }

    Mat PLU_inplace()
    {
        const int min_mn = min(this->m,this->n);
        Mat P(min_mn,1);
        int _P[min_mn];
        memset(_P, 0, min_mn * sizeof(int) );
        lapacke_plu(this->data, _P, this->m, this->n);
        for(int i = 0; i < min_mn; i++)
            P.data[i] = (double)_P[i];
        return P;
    }

    void eye(int _n)
    {
        bool size_okay = (_n*_n) == (this->m * this->n);

        if(sh_ptr.count == NULL)
            this->zeros(_n, _n);
        else if( *(sh_ptr.count) == 1)
        {
            this->m = _n;
            this->n = _n;
            if(!size_okay)
                data = (double*) realloc(data, _n*_n*sizeof(double));
        }
        else // > 1
        {
            sh_ptr.dec();
            this->m = _n;
            this->n = _n;
            data = (double*) malloc(_n*_n*sizeof(double));
            free(sh_ptr.count);
            sh_ptr.init(data);
        }
        this->zero();
        for(int i = 0; i < _n; i++)
                this->data[i*_n+i] = 1.0;
        this->m = _n;
        this->n = _n;
    }

    void scale_diag(double d)
    {
        assert(this->m == this->n);
        for(int i = 0; i < n; i++)
            this->data[i*n+i] *= d;
    }

    void embed(int m_start, int n_start, Mat& A)
    {
        if( (this->m < A.m) ||  (this->n < A.n) )
            print_wrong_dim_mat(this, &A);
        if( (this->m < m_start + A.m) || (this->n < n_start + A.n) )
        {
            printf("embed error: m_start,n_start=(%d,%d), A=(%d x %d)\n", m_start, n_start, A.m,A.n);
            print_wrong_dim_mat(this, &A);
        }
        int m_end = m_start + A.m;
        int n_end = n_start + A.n;
        for(int i = m_start; i < m_end; i++)
            for(int j = n_start; j < n_end; j++)
            {
                int ai = i-m_start;
                int aj = j-n_start;
                this->data[i*this->n + j] = A.data[ai*A.n + aj];
            }
    }
    void embed(int m_start, int n_start, int m_end, int n_end, Mat& A)
    {
        if( (this->m < A.m) ||  (this->n < A.n) )
            print_wrong_dim_mat(this, &A);
        if( (this->m <= m_start + A.m) || (this->n <= n_start + A.n) )
        {
            printf("embed error: m_start,n_start=(%d,%d), A=(%d x %d)\n", m_start, n_start, A.m,A.n);
            print_wrong_dim_mat(this, &A);
        }
        assert(m_end == (m_start + A.m) );
        assert(n_end == (n_start + A.n) );
        for(int i = m_start; i < m_end; i++)
            for(int j = n_start; j < n_end; j++)
            {
                int ai = i-m_start;
                int aj = j-n_start;
                this->data[i*this->n + j] = A.data[ai*A.n + aj];
            }
    }

    Mat extract(int m_start, int m_end, int n_start, int n_end)
    {
        return (*this)(m_start, m_end, n_start, n_end); // using operator overloaded below
    }
    
    Mat extract(int m_start, int m_end)
    {
        assert(m_start <= m_end);
        assert(m_end <= this->m*this->n);
        assert(m_start >= 0);
        int wm = m_end - m_start;
        Mat A(wm);
        memcpy(A.data, this->data + m_start, wm * sizeof(double));
        return A;// using operator overloaded below
    }

    void reshape(int _m, int _n)
    {
        assert( (_m * _n) == (this->m * this->n) );
        this->m = _m;
        this->n = _n;
    }
    void zeros(int _m, int _n)
    {
        if(sh_ptr.count == NULL)
        {
            this->data = (double*) malloc(_m * _n * sizeof(double));
            null_ptr_check(this->data);
            this->sh_ptr.init(data);
        }
        else if( *(sh_ptr.count) == 1)
        {
            if(_m*_n != this->m*this->n)
            {
                this->data = (double*) realloc(data, _m * _n * sizeof(double));
                this->sh_ptr.ptr = this->data;
            }
        }
        else // > 1
        {
            if( (_m*_n) != this->m*this->n)
            {
                this->sh_ptr.dec();
                this->data = (double*) malloc(_m * _n * sizeof(double));
                null_ptr_check(this->data);
                this->sh_ptr.init(data);
            }
        }
        this->zero();
        this->m = _m;
        this->n = _n;
    }

    void zero()
    {
        memset(this->data, 0, this->m*this->n*sizeof(double));
    }

    void vec_insert(int idx, const Mat& A)
    {
        memcpy(this->data + idx, A.data, A.m*A.n*sizeof(double));
    }

    Mat subvec(int start_idx, int length)
    {
        assert( (this->n == 1) || (this->m == 1) );
        Mat vec(length);
        vec.data = this->data + start_idx;
        return vec;
    }

    Mat trans()
    {
        Mat A(this->n, this->m);
        transpose(this->data, A.data, this->m, this->n);
        return A;
    }

    Mat inv()
    {
        assert(this->m == this->n);
        Mat A(this->m, this->n);
        lapacke_inv(this->data, A.data, this->n);
        return A;
    }

    void inv_inplace()
    {
        assert(this->m == this->n);
        lapacke_inv(this->data, this->n);
    }

    Mat inv_diag()
    {
        assert(this->m == this-> n);
        Mat A(this->m, this->n);
        A.zero();
        for(int i = 0; i < this->n; i++)
            A.data[i*n+i] = 1.0 / this->data[i*n+i];
        return A;
    }

    void trans_inplace()
    {
        if(this->m == this->n)
        {
            reflect_array(this->data, this->m, this->n);
        }
        else
        {
            Mat A = this->trans();
            this->copy(A);
        }
    }

    void add_diag(Mat& D)
    {
        assert(this->m == this->n);
        assert(D.m*D.n == this->m);
        if(D.is_1d())
        {
            for(int i = 0; i < n; i++)
                this->data[i*n+i] += D.data[i];
        }
        else 
        {
            assert(D.m == D.n);
            for(int i = 0; i < n; i++)
                this->data[i*n+i] += D.data[i*n*i];
        }
    }

    void print(int decimals, bool scientific_notation)
    {
        for(int i = 0; i < this->m; i++)
        {
            for(int j = 0; j < this->n; j++)
            {
                if(scientific_notation)
                    printf("%.*E, ", decimals, this->data[i*n+j]);
                else
                    printf("%.*f, ", decimals, this->data[i*n+j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    void print()
    {
        for(int i = 0; i < this->m; i++)
        {
            for(int j = 0; j < this->n; j++)
            {
                printf("%.6f, ", this->data[i*n+j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    void shape()
    {
        printf("(%d x %d)\n", this->m, this->n);
    }

    double norm2()
    {
        return norm(this->data, this->n);
    }

    int size()
    {
        return this->m * this-> n;
    }
    
    bool is_1d()
    {
        return (this->m ==1) || (this->n==1);
    }
    bool is_diag(double eps)
    {
        assert(this->m == this->n);
        for(int i = 0; i < m; i++)
        {
            for(int j = 0; j < n; j++)
            {
                if(i != j)
                    if( abs(this->data[i*n+j]) > eps ) 
                        return false;
            }
        }
        return true;
    }
    bool is_zeros(double eps)
    {
        for(int i = 0; i < m; i++)
            for(int j = 0; j < n; j++)
                if(abs(data[i]) > eps)
                    return false;
        return true;

    }

    friend Mat operator+(const Mat& A, const Mat& B)
    {
        if(A.m*A.n != B.m*B.n)
            print_wrong_dim_mat(A, B);
        Mat C(A.m, A.n);
        add_vecs(A.data,B.data,C.data,A.m*A.n);
        return C;
    }

    friend Mat operator+(const Mat& A, double a)
    {
        Mat B(A);
        int mn = B.m + B.n;
        for(int i = 0; i < mn; i++)
            B.data[i] += A.data[i] + a;
        return B;
    }

    friend Mat operator+(double a, const Mat& A)
    {
        Mat B(A);
        int mn = B.m + B.n;
        for(int i = 0; i < mn; i++)
            B.data[i] += A.data[i] + a;
        return B;
    }

    void operator+=(const Mat& A)
    {
        if(A.m*A.n != this->m*this->n)
            print_wrong_dim_mat(this, A);
        add_vecs(this->data,A.data,A.m*A.n);
    }

    friend Mat operator-(const Mat& A, const Mat& B)
    {
        if(A.m*A.n != B.m*B.n)
            print_wrong_dim_mat(A, B);
        Mat C(A.m, A.n);
        sub_vecs(A.data,B.data,C.data,A.m*A.n);
        return C;
    }
    void operator-=(const Mat& A)
    {
        if(A.m*A.n != this->m*this->n)
            print_wrong_dim_mat(this, A);
        sub_vecs(this->data,A.data,A.m*A.n);
    }

    // horizontal concatenate inplace
    void operator<=(const Mat& A)
    {
        if( this->m != A.m )
            print_wrong_dim_mat(this, A);
        int m = this->m;
        int n = this->n;
        int na = A.n;

        double* new_data = (double*) malloc( m * (n+na) * sizeof(double) );
        for(int i = 0; i < m; i++)
        {
            int offset_new =  i*(n+na);
            memcpy(new_data + offset_new, this->data + i*n, n * sizeof(double) );
            memcpy(new_data + offset_new + n, A.data + i*na, na * sizeof(double) );
        }
        ptr_swap(&this->data, &new_data);
        this->sh_ptr.ptr = this->data;
        free(new_data);
        this->m = m;
        this->n = n+na;
    }

    // vertical concatenate inplace
    void operator^=(const Mat& A)
    {
        if( this->n != A.n )
            print_wrong_dim_mat(this, A);
        int m = this->m;
        int n = this->n;
        int ma = A.m;
        if(*sh_ptr.count == 1)
        {
            this->data = (double*) realloc(this->data, (ma + m) * n * sizeof(double) );
            null_ptr_check(this->data);
            this->sh_ptr.ptr = this->data;
            memcpy(this->data + n*m, A.data, ma * n * sizeof(double));
        }
        else  // > 1
        {   
            this->sh_ptr.dec();
            double* old_data = this->data;
            this->data = (double*) malloc((ma+m)*n*sizeof(double));
            memcpy(this->data, old_data, m*n*sizeof(double));
            memcpy(this->data + n*m, A.data, ma * n * sizeof(double));
            this->sh_ptr.init(this->data);
        }
        this->m = m + ma;
        this->n = n;
    }

    void operator^=(double power)
    {
        int mn = this->m * this->n;
        for(int i = 0; i < mn; i++)
            this->data[i] = pow(this->data[i], power);
    }

    void operator^=(int power)
    {
        int mn = this->m * this->n;
        for(int i = 0; i < mn; i++)
            this->data[i] = pow(this->data[i], power);
    }

    // horizontal concatenate return 
    friend Mat operator<<(const Mat& A, const Mat& B)
    {
        if( A.m != B.m )
            print_wrong_dim_mat(A, B);
        int m = A.m;
        int na = A.n;
        int nb = B.n;
        Mat C(m, na+nb);
        for(int i = 0; i < m; i++)
        {
            int offset_new =  i*(nb+na);
            memcpy(C.data + offset_new, A.data + i*na, na * sizeof(double) );
            memcpy(C.data + offset_new + na, B.data + i*nb, nb * sizeof(double) );
        }
        return C;
    }

    // vertical concatenate return
    friend Mat operator^(const Mat& A, const Mat& B)
    {
        if( A.n != B.n )
            print_wrong_dim_mat(A, B);
        int ma = A.m;
        int mb = B.m;
        int n = A.n;
        Mat C(ma+mb,n);
        memcpy(C.data, A.data, ma * n * sizeof(double));
        memcpy(C.data + n*ma, B.data, mb * n * sizeof(double));
        return C;
    }

    friend Mat operator*(const Mat& A, const Mat& B)
    {
        int m_max = max(A.m,B.m);
        int n_max = max(A.n, B.n);
        Mat C(m_max, n_max);
        if(A.m*A.n == B.m*B.n)
        {
            int mn = A.m*A.n;
            for(int i = 0; i < mn; i++)
                C.data[i] = A.data[i] * B.data[i];
        }
        else if( (A.m == B.m) && (A.n != B.n) )
        {
            const Mat& _A = A.n > B.n ? A : B;
            const Mat& _B = A.n < B.n ? A : B;
            int m = _A.m;
            int n = _A.n;
            assert(_B.n == 1);
            for(int i = 0; i < m; i++)
                for(int j = 0; j < n; j++)
                    C.data[i*n + j] = _A.data[i*n + j] * _B.data[i];
        }
        else if( (A.m != B.m) && (A.n == B.n) )
        {
            const Mat& _A = A.m > B.m ? A : B;
            const Mat& _B = A.m < B.m ? A : B;
            int m = _A.m;
            int n = _A.n;
            assert(_B.m == 1);
            for(int i = 0; i < m; i++)
                for(int j = 0; j < n; j++)
                    C.data[i*n + j] = _A.data[i*n + j] * _B.data[j];
        }
        else 
        {
            print_wrong_dim_mat(A,B);
        }
        return C;
    }

    void operator==(const Mat& A)
    {
        this->copy(A);
    }

    friend Mat operator*(const Mat& A, double a)
    {
        Mat B(A);
        scale_vec(B.data, a, B.m * B.n);
        return B;
    }

    friend Mat operator*(double a, const Mat& A)
    {
        Mat B(A);
        scale_vec(B.data, a, B.m * B.n);
        return B;
    }

    void operator*=(const Mat& B)
    {
        if( (B.m > this->m) || (B.n > this->n) )
        {
            print_wrong_dim_mat(this, B);
        }
        if( (this->m*this->n) == (B.m*B.n) )
        {
            int mn = this->m*this->n;
            for(int i = 0; i < mn; i++)
                this->data[i] *= B.data[i];
        }
        else if( (this->m == B.m) && (this->n != B.n) )
        {
            assert(B.n == 1);
            for(int i = 0; i < this->m; i++)
                for(int j = 0; j < this->n; j++)
                    this->data[i*this->n + j] *= B.data[i];
        }
        else if( (this->m != B.m) && (this->n == B.n) )
        {
            assert(B.m == 1);
            for(int i = 0; i < this->m; i++)
                for(int j = 0; j < this->n; j++)
                    this->data[i*n + j] *= B.data[i];
        }
        else 
        {
            print_wrong_dim_mat(this, B);
        }
    }

    void operator*=(double a)
    {
        scale_vec(this->data, a, this->m * this-> n);
    }

    friend Mat operator&(const Mat& A, const Mat& B)
    {
        int m = A.m;
        int n = B.n;
        int p = A.n;
        if(A.n != B.m)
            print_wrong_dim_mat(A, B);
        Mat C(m,n);
        matmatmul(A.data, B.data, C.data, m,p,p,n, false, false);
        return C;
    }

    void operator&=(const Mat& B)
    {
        int m = this->m;
        int n = this->n;
        int p = this->n;
        if(n != B.m)
            print_wrong_dim_mat(this, B);
        Mat C(m,n);
        matmatmul(this->data, B.data, C.data, m,p,p,n, false, false);
        this->copy(C);
    }

    double operator()(int row)
    {
        assert(row < this->n*this->m);
        return this->data[row];
    }

    double operator()(int row, int col)
    {
        return this->data[row*this->n + col];
    }

    Mat operator()(int yl, int yh, int xl, int xh)
    {
        assert( (yl < yh) && (xl < xh) );
        assert( (yh <= this->m) && (xh <= this->n) );

        int wy = yh - yl;
        int wx = xh - xl;
        Mat A(wy,wx);
        for(int i = 0; i < wy; i++)
            for(int j = 0; j < wx; j++)
                A.data[i*wx + j] = this->data[ (yl+i)*this->n + xl + j];
        return A;
    }


    friend Mat operator/(const Mat& A, const Mat& B)
    {
        if ((A.n*A.m == 0) || (B.n*B.m == 0) )
            print_wrong_dim_mat(A, B);
        if ((A.m != B.m) )
            print_wrong_dim_mat(A, B);
        
        int max_mn = max(A.m,A.n);
        Mat X(max_mn, B.n);
        // least square problem
        if(A.n != A.m)
            lapacke_lstsq(A.data, B.data, X.data, A.m, A.n, X.n, false);
        else 
            lapacke_solve(A.data, B.data, X.data, A.n, B.n);
        return X;
    }

    
    Mat& operator=(const Mat& other) {
        //std::cout << "Copy assignment called\n";
        if(&other != this)
        {
            this->m = other.m;
            this->n = other.n; 
            if(this->data != NULL)
                this->sh_ptr.dec();
            this->data = other.data;
            this->sh_ptr = other.sh_ptr;
            this->sh_ptr.inc();
        }
        return *this;
    }
    
    ~Mat()
    {
        //printf("Goodbye from Mat!\n");
        //if(data != NULL)
        //{
        //    free(data);
        //    data = NULL;
        //}
    }
};

class Tensor
{
    public:
    int N;
    int m;
    int n;
    Mat* datas;
    Tensor(int N, int m, int n)
    {
        this->N = N;
        this->m = m;
        this->n = n;
        datas = (Mat*) malloc(N * sizeof(Mat));
        memset(datas, 0, N * sizeof(Mat));
    }

    Mat& operator[](int idx)
    {
        assert(idx < this->N);
        if(idx >= 0)
            return datas[idx];
        else
        {
            assert(idx > -this->N-1);
            return datas[this->N+idx];
        }
    }

    void print()
    {
        for(int i = 0; i < N; i++)
            datas[i].print();
    }

    ~Tensor()
    {
        if(datas != NULL)
        {
            for(int i = 0; i < N; i++)
            {
                if(datas[i].data != NULL)
                {
                    //free(datas[i].data);
                    //datas->data = NULL;
                    datas[i].sh_ptr.dec();
                }
            }
            free(datas);
        }
    }
};

class MatArray
{
    public:
    int N;
    Mat* datas;
    MatArray(int N)
    {
        this->N = N;
        datas = (Mat*) malloc(N * sizeof(Mat));
        memset(datas, 0, N * sizeof(Mat));
    }

    Mat& operator[](int idx)
    {
        assert(idx < this->N);
        if(idx >= 0)
            return datas[idx];
        else
        {
            assert(idx > -this->N-1);
            return datas[this->N+idx];
        }
    }

    void print()
    {
        for(int i = 0; i < N; i++)
            datas[i].print();
    }

    ~MatArray()
    {
        if(datas != NULL)
        {
            for(int i = 0; i < N; i++)
            {
                if(datas[i].data != NULL)
                {
                    //free(datas->data);
                    //datas->data = NULL;
                    datas[i].sh_ptr.dec();
                }
            }
            free(datas);
        }
    }
};

Mat randn(int m, int n)
{
    Mat A(m,n);
    int mn = m*n;
    for(int i = 0; i < mn; i ++)
        A.data[i] = random_normal(0, 1);
    return A;
}

void print_wrong_dim_mat(Mat* A, Mat* B)
{
    printf("Wrong dimensions for matrix: (%d x %d) for input (%d X %d)", A->m, A->n, B->m, B->n);
    exit(1);
}
void print_wrong_dim_matmul(Mat* A, Mat* B)
{
    printf("Wrong dimensions for matrix multiplication: (%d x %d) for input (%d X %d)", A->m, A->n, B->m, B->n);
    exit(1);
}
void print_wrong_dim_mat(const Mat& A, const Mat& B)
{
    printf("Wrong dimensions for matrix: (%d x %d) for input (%d X %d)", A.m, A.n, B.m, B.n);
    exit(1);
}
void print_wrong_dim_matmul(const Mat& A, const Mat& B)
{
    printf("Wrong dimensions for matrix multiplication: (%d x %d) for input (%d X %d)", A.m, A.n, B.m, B.n);
    exit(1);
}

#endif // _MPC_LINALG_HPP_