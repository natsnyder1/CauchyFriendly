/* file : hello_swig.i */
  
/* name of module to use*/
%module hello_swig 
%{ 
    #define SWIG_FILE_WITH_INIT
    /* Every thing in this file is being copied in  
     wrapper file. We include the C header file necessary 
     to compile the interface */
    #include "hello_swig.hpp" 
  
    /* variable declaration*/
%} 
%include "typemaps.i"
%include "numpy.i"
%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* x, int n)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int n, double* x)};

  
/* explicitly list functions and variables to be interfaced */
double foobar;
int pow_2(int x);
void structFooBar_init();
void structFooBar_deinit();
void structFooBar_set(int);

/* Define some functions that take python array input and returns scalar */
%apply (double* IN_ARRAY1, int DIM1) {(double* x, int n)};
double get_array_sum(double* x, int n);
%clear (double* x, int n);

/* Define some functions that take numpy array input and returns numpy array output*/
%apply (double* IN_ARRAY1, int DIM1) {(double* x, int n)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double* y, int m)};
void double_it(double* x, int n, double* y, int m);
%clear (double* x, int n);
%clear (double* y, int m);

/* Define some functions that take scalar input and returns python array output */
%apply (int ** ARGOUTVIEWM_ARRAY1, int *DIM1) {(int **X, int *out_n)};

void fill_array_cumsum(int m, int **X, int *out_n);
%clear (int **X, int *out_n);

%apply (int ** ARGOUTVIEWM_ARRAY1, int *DIM1) {(int **out_cumarray, int *size_cumarray)};
%apply (int ** ARGOUTVIEWM_ARRAY1, int *DIM1) {(int **out_doubled_cumarray, int *size_doubled_cumarray)};
%apply int *OUTPUT { int *out_cumarray_sum, int *out_doubled_cumarray_sum };
void return_multiple_things(int m, int **out_cumarray, int *size_cumarray, int** out_doubled_cumarray, int* size_doubled_cumarray, int* out_cumarray_sum, int* out_doubled_cumarray_sum);
%clear (int **out_cumarray, int *size_cumarray);
%clear (int **out_doubled_cumarray, int *size_doubled_cumarray);
%clear (int **out_cumarray_sum, int *out_doubled_cumarray_sum);

DynUpdateLight* get_DynUpdateLight_ptr(const int n, const int pncc, const int p);


/* Define some function which takes a 2D array into python and returns two one-D arrays */

/* Define some function which takes a 2D array into python and returns a 2D array */

/* Define python functions that can be sent as callback functions within the C code */

// a typemap for the callback, it expects the argument to be an integer
// whose value is the address of an appropriate callback function
// a typemap for the callback, it expects the argument to be an integer
// whose value is the address of an appropriate callback function
%typemap(in) void (*f)(int, const char*) {
    $1 = (void (*)(int, const char*))PyLong_AsVoidPtr($input);;
}
%typemap(in) void (*f2)(FooPoint*) {
    $1 = (void (*)(FooPoint*))PyLong_AsVoidPtr($input);;
}
%typemap(in) void (*f3)(DynUpdateLight*) {
    $1 = (void (*)(DynUpdateLight*))PyLong_AsVoidPtr($input);;
}

%{
    void use_callback(void (*f)(int i, const char* str));
    void initialize_foo_system(int n, void (*f2)(FooPoint*) );
    void initialize_NL_system(int n, int pncc, int p, double* x, int size_x, void (*f3)(DynUpdateLight*) );
    void initialize_NL_system_v2(int n, int pncc, int p, double* x, int size_x, void (*f)(int, const char*), void (*f_dyn)(DynUpdateLight*) );
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* x, int size_x)};
void use_callback(void (*f)(int i, const char* str));
void initialize_foo_system(int n, void (*f2)(FooPoint*) );
void initialize_NL_system(int n, int pncc, int p, double* x, int size_x, void (*f3)(DynUpdateLight*) );
void initialize_NL_system_v2(int n, int pncc, int p, double* x, int size_x, void (*f)(int, const char*), void (*f3)(DynUpdateLight*) );
%clear (double* x, int size_x);



/* or if we want to interface all functions then we can simply 
   include header file like this -  
   %include "hello_swig.hpp" 
*/