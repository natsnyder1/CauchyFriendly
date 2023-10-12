CC=g++

GCC_DEBUG = -g -Wall -Wextra
GCC_RELEASE = -O3 # It is seen that -O3 compilation sometimes causes a seg fault due to the compiler optimizations attempted

CFLAGS=""

# MAC SETTING COMMENTED
#INC_LAPACK=-I/usr/local/opt/lapack/include
#LIB_LAPACK = -L/usr/local/opt/lapack/lib -llapacke -llapack -lblas -lm -lpthread

LIB_LAPACK = -llapacke -llapack -lblas -lm -lpthread

D ?= 0
ifeq ($(D), 1)
	CFLAGS = $(GCC_DEBUG)
else
	CFLAGS = $(GCC_RELEASE)
endif

bin/cauchy_estimator : src/cauchy_estimator.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_LAPACK) 

bin/window_manager : src/window_manager.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_LAPACK)  

#$(INC_LAPACK) $(LIB_LAPACK)

cauchy : bin/cauchy_estimator
window : bin/window_manager

all :
	cauchy window

clean : 
	rm -f bin/cauchy_estimator bin/window_manager
