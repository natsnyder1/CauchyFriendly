/* file : _enu.i */
  
/* name of module to use*/
%module _enu 
%{ 
    #define SWIG_FILE_WITH_INIT
    /* Every thing in this file is being copied in  
     wrapper file. We include the C header file necessary 
     to compile the interface */
    #include "_enu.hpp" 
  
    /* variable declaration*/
%} 

%include "typemaps.i"
%include "numpy.i"
%init %{
import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* A, int size_A)};
%apply (int** ARGOUTVIEWM_ARRAY1, int* DIM1) {(int **out_B, int *size_out_B)};

void call_inc_enu(double* A, int size_A,
                  int m, int n,
                  int** out_B, int* size_out_B, 
                  const bool is_encoded = true,
                  const bool is_sorted = true);

void call_nat_enu(double* A, int size_A,
                  int m, int n,
                  int** out_B, int* size_out_B, 
                  const bool is_encoded = true,
                  const bool is_sorted = true);