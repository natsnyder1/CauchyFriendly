CC=g++

GCC_DEBUG = -g -Wall -Wextra
GCC_RELEASE = -O3 # It is seen that -O3 compilation sometimes causes a seg fault due to the compiler optimizations attempted

CFLAGS=""

INC_LAPACK=-I/usr/local/opt/lapack/include
LIB_LAPACK = -L/usr/local/opt/lapack/lib -llapacke -llapack -lblas -lm -lpthread
LIB_GLPK = -lglpk -lpthread

D ?= 0
ifeq ($(D), 1)
	CFLAGS = $(GCC_DEBUG)
else
	CFLAGS = $(GCC_RELEASE)
endif

bin/gamma : gamma.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(INC_LAPACK) $(LIB_LAPACK)

bin/binsearch : binsearch.cpp
	$(CC) $(CFLAGS) $^ -o $@

bin/test_cpdf_ndim : test_cpdf_ndim.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(INC_LAPACK) $(LIB_LAPACK)

bin/test_enum : test_enumerations.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(INC_LAPACK) $(LIB_LAPACK) $(LIB_GLPK)

bin/test_coalign : test_coalign_with_zeros.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(INC_LAPACK) $(LIB_LAPACK)

cpdf : bin/test_cpdf_ndim

enum : bin/test_enum

coalign : bin/test_coalign

all : bin/gamma bin/binsearch cpdf enum coalign

clean : 
	rm -f bin/gamma bin/binsearch bin/test_cpdf_ndim bin/test_enum bin/test_coalign