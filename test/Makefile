CC=g++

GCC_DEBUG = -g -Wall -Wextra
GCC_RELEASE = -O3 # It is seen that -O3 compilation sometimes causes a seg fault due to the compiler optimizations attempted

CFLAGS=""

#INC_LAPACK=-I/usr/local/opt/lapack/include -L/usr/local/opt/lapack/lib

LIB_LAPACK = -llapacke -llapack -lblas -lm -lpthread
LIB_GLPK = -lglpk -lpthread
LIB_MATH_PTHREAD = -lm -lpthread


D ?= 0
ifeq ($(D), 1)
	CFLAGS = $(GCC_DEBUG)
else
	CFLAGS = $(GCC_RELEASE)
endif

bin/gamma : gamma.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)  

bin/binsearch : binsearch.cpp
	$(CC) $(CFLAGS) $^ -o $@

bin/test_cpdf_ndim : test_cpdf_ndim.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)  

bin/test_enum : test_enumerations.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD) $(LIB_GLPK)
#$(LIB_LAPACK)

bin/test_coalign : test_coalign_with_zeros.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)  

bin/test_pred : test_rel_marg2d_cpdf.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)

bin/test_target_tracking : test_target_tracking.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)

bin/test_eig : test_eig_solver.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_LAPACK)

cpdf : bin/test_cpdf_ndim

enum : bin/test_enum

coalign : bin/test_coalign

pred : bin/test_pred

track : bin/test_target_tracking

eig : bin/test_eig

all : bin/gamma bin/binsearch cpdf enum coalign track eig

clean : 
	rm -f bin/gamma bin/binsearch bin/test_cpdf_ndim bin/test_enum bin/test_coalign bin/test_pred bin/test_target_tracking bin/test_eig
