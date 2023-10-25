#ifndef _HELLO_SWIG_HPP_
#define _HELLO_SWIG_HPP_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
int pow_2(int x);
struct structFooBar;
void structFooBar_init();
void structFooBar_deinit();
void structFooBar_set(int);
*/

double foobar;

int pow_2(int x)
{
    return x*x;
}

double get_array_sum(double* x, int n)
{
    double sum = x[0];
    for(int i = 1; i < n; i++)
        sum += x[i]; 
    return sum;
}

void fill_array_cumsum(int m, int **x_out, int *nx_out)
{
    (*x_out) = (int*) malloc( m * sizeof(int) );
    *nx_out = m;
    double cumsum = 0;
    for(int i = 0; i < m; i++)
    {
        (*x_out)[i] = cumsum;
        cumsum += (i+1);
    }
}

void double_it(double* x, int n, double* y, int m)
{
    for(int i = 0; i < n; i++)
        y[i] = 2*x[i];
}

struct structFooBar
{
    int x;
    
    void set_x(int _x)
    {
        x = _x;
        printf("x now equals %d\n", x);
    }

    int get_pow2_x()
    {
        return pow_2(x);
    }
};

structFooBar* my_foo;

void structFooBar_init()
{
    my_foo = (structFooBar*) malloc(sizeof(structFooBar));
    printf("Struct FooBar has been initialized!\n");
}

void structFooBar_deinit()
{
    free(my_foo);
    printf("Struct FooBar has been de-initialized!\n");
}

void structFooBar_set(int x)
{
    my_foo->set_x(x);
    printf("my_foo x-value is now: %d. Calling struct's get_pow2 function yields: %d\n", my_foo->x, my_foo->get_pow2_x() );
}

// a C function that accepts a callback
void use_callback(void (*f)(int i, const char* str))
{
    printf("Hello From C Code! Running callback function\n");
    f(100, "callback arg");
    printf("Goodbye From C Code!\n");
}

void print_mat(double* A, const int m, const int n)
{
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            printf("%lf, ", A[i*n + j]);
        printf("\n");
    }
    printf("\n");
}

struct DynUpdateLight
{
    int n;
    int pncc;
    int p;
    double* Phi;
    double* Gamma;
    double* H;
    double* beta;
    double* gamma;
    double* x;
    int step;
    void* other;
};

struct DynamicsTest
{
    void(*dynam_callback1)(DynUpdateLight*);
    int n;
    int p;
    int pncc;
    DynUpdateLight* duc;

    void init(DynUpdateLight* _duc)
    {
        duc = _duc;
    }
};

DynUpdateLight* duc;
DynamicsTest* dtest;

void DynUpdateLight_init(const int n, const int pncc, const int p)
{
    duc = (DynUpdateLight*) malloc( sizeof(DynUpdateLight) );
    duc->n = n;
    duc->pncc = pncc;
    duc->p = p;
    duc->Phi = (double*) calloc(n * n , sizeof(double));
    duc->Gamma = (double*) calloc(n * pncc , sizeof(double));
    duc->H = (double*) calloc(n * p , sizeof(double));
    duc->beta =  (double*) calloc(pncc , sizeof(double));
    duc->gamma =  (double*) calloc(p , sizeof(double));
    duc->x = (double*) calloc(n , sizeof(double));
    duc->step = 0;
    duc->other = NULL;
    printf("Setting all Values of duc to zero:"); 
}

DynUpdateLight* get_DynUpdateLight_ptr(const int n, const int pncc, const int p)
{
    DynUpdateLight* _duc = (DynUpdateLight*) malloc( sizeof(DynUpdateLight) );
    _duc->n = n;
    _duc->pncc = pncc;
    _duc->p = p;
    _duc->Phi = (double*) calloc(n * n , sizeof(double));
    _duc->Gamma = (double*) calloc(n * pncc , sizeof(double));
    _duc->H = (double*) calloc(n * p , sizeof(double));
    _duc->beta =  (double*) calloc(pncc , sizeof(double));
    _duc->gamma =  (double*) calloc(p , sizeof(double));
    _duc->x = (double*) calloc(n , sizeof(double));
    _duc->step = 0;
    _duc->other = NULL;
    return _duc;
}


void DynUpdateLight_print()
{
    printf("----- Dynamics Update Container Printer -----\n");
    printf("Phi is:\n");
    print_mat(duc->Phi, duc->n, duc->n);
    printf("Gamma is:\n");
    print_mat(duc->Gamma, duc->n, duc->pncc);
    printf("H is:\n");
    print_mat(duc->H, duc->n, duc->p);
    printf("beta is:\n");
    print_mat(duc->beta, 1, duc->pncc);
    printf("gamma is:\n");
    print_mat(duc->gamma, 1, duc->p);
    printf("x is:\n");
    print_mat(duc->x, 1, duc->n);
    printf("----- End of Printe -----\n");
}


void initialize_LTI_system(
    int n, int p, int pncc,
    double* Phi, int size_Phi, 
    double* Gamma,  int size_Gamma, 
    double* H,  int size_H, 
    double* beta,  int size_beta, 
    double* gamma,  int size_gamma
)
{
    // Bring these into sliding window manager function
    DynUpdateLight_init(n, pncc, p);
    // Copy params to the initialized duc
    memcpy(duc->Phi, Phi, n*n*sizeof(double));
    memcpy(duc->Gamma, Gamma, n*n*sizeof(double));
    memcpy(duc->H, H, n*n*sizeof(double));
    memcpy(duc->beta, beta, n*n*sizeof(double));
    memcpy(duc->gamma, gamma, n*n*sizeof(double));
}

void initialize_LTV_system()
{

}

void initialize_NL_system(int n, int pncc, int p,
    double* x, int size_x,
    void (*f_dyn)(DynUpdateLight*) )
{
    DynUpdateLight_init(n, pncc, p); 
    printf("C: Before the duc was filled, it is:\n");
    DynUpdateLight_print();
    memcpy(duc->x, x, n * sizeof(double));
    printf("C: Handing Allocated but Uninitialized DynUpdateLight pointer to callback!\n");
    f_dyn(duc);
    printf("C: After the duc was filled, it is:\n");
    DynUpdateLight_print();
}

void init_dynamic_test(DynUpdateLight* duc)
{
    dtest = (DynamicsTest*) malloc(sizeof(DynamicsTest));
    dtest->init(duc);
}

struct FooPoint
{
    int x;
    int y;
    int z;
};

void initialize_foo_system(int n, void (*f_dyn)(FooPoint*) )
{
    printf("Hello from foo system! n=%d\n", n);
    FooPoint fp; fp.x = 0; fp.y = 0; fp.z = 0; 
    printf("Foo Point Initialized as: x=%d, y=%d, z=%d\n", fp.x, fp.y, fp.z);
    f_dyn(&fp);
    printf("After Provided Python Callback, Foo Point is: x=%d, y=%d, z=%d\n", fp.x, fp.y, fp.z);
}




#endif // _HELLO_SWIG_HPP_