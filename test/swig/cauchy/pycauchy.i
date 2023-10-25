/* file : pycauchy.i */
  
/* name of module to use*/
%module hello_swig 
%{ 
    #define SWIG_FILE_WITH_INIT
    /* Every thing in this file is being copied in  
     wrapper file. We include the C header file necessary 
     to compile the interface */
    #include "pycauchy.hpp" 
  
    /* variable declaration*/
%} 

%include "numpy.i"
%init %{
import_array();
%}

