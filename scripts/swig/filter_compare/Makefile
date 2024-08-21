CC=g++

GCC_DEBUG = -g -Wall -Wextra
GCC_RELEASE = -O3 -Wall -Wextra # It is seen that -O3 compilation sometimes causes a seg fault due to the compiler optimizations attempted

CFLAGS=""

# MAC SETTING COMMENTED
#INC_LAPACK=-I/usr/local/opt/lapack/include
#LIB_LAPACK = -L/usr/local/opt/lapack/lib -llapacke -llapack -lblas -lm -lpthread
#$(INC_LAPACK) $(LIB_LAPACK)

LIB_LAPACK = -llapacke -llapack -lblas -lm -lpthread
#LIB_LAPACK= -Xlinker -start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Xlinker -end-group -lgomp -lpthread -lm -ldl


D ?= 0
ifeq ($(D), 1)
	CFLAGS = $(GCC_DEBUG)
else
	CFLAGS = $(GCC_RELEASE)
endif

mpc : mpc.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_LAPACK) 

all : mpc

clean : 
	rm mpc
