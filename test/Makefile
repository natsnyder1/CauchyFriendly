CC=g++

GCC_DEBUG = -g -Wall -Wextra
GCC_RELEASE = -O3 # It is seen that -O3 compilation sometimes causes a seg fault due to the compiler optimizations attempted

CFLAGS=""

INC_LAPACK=-I/usr/local/opt/lapack/include

LIB_LAPACK = -L/usr/local/opt/lapack/lib -llapacke -llapack -lblas -lm -lpthread

D ?= 0
ifeq ($(D), 1)
	CFLAGS = $(GCC_DEBUG)
else
	CFLAGS = $(GCC_RELEASE)
endif

bin/gamma : gamma.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(INC_LAPACK) $(LIB_LAPACK)

gamma : bin/gamma

clean : 
	rm -f bin/gamma 