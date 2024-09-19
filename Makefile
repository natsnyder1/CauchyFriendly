CC=g++
#g++

GCC_DEBUG = -g -Wall -Wextra
GCC_RELEASE = -O3 # It is seen that -O3 compilation sometimes causes a seg fault due to the compiler optimizations attempted

CFLAGS=""

# MAC SETTING COMMENTED
#INC_LAPACK=-I/usr/local/opt/lapack/include
#LIB_LAPACK = -L/usr/local/opt/lapack/lib -llapacke -llapack -lblas -lm -lpthread
#$(INC_LAPACK) $(LIB_LAPACK)

LIB_LAPACK = -llapacke -llapack -lblas -lm -lpthread
LIB_MATH_PTHREAD = -lm -lpthread
#LIB_LAPACK= -Xlinker -start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Xlinker -end-group -lgomp -lpthread -lm -ldl


D ?= 0
ifeq ($(D), 1)
	CFLAGS = $(GCC_DEBUG)
else
	CFLAGS = $(GCC_RELEASE)
endif

bin/cauchy_estimator : src/cauchy_estimator.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK) 

bin/window_manager : src/window_manager.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)  

bin/leo_satellite_5state : src/leo_satellite_5state.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)  

bin/homing_missile : src/homing_missile.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)

bin/leo_satellite_7state : src/leo_satellite_7state.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)  

bin/leo_satellite_5state_gps : src/leo_satellite_5state_gps.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)  

bin/leo_satellite_7state_gps : src/leo_satellite_7state_gps.cpp
	$(CC) $(CFLAGS) $^ -o $@ $(LIB_MATH_PTHREAD)
#$(LIB_LAPACK)  

cauchy : bin/cauchy_estimator
window : bin/window_manager
home : bin/homing_missile
leo5 : bin/leo_satellite_5state
leo5_gps : bin/leo_satellite_5state_gps
leo7 : bin/leo_satellite_7state
leo7_gps : bin/leo_satellite_7state_gps

all: cauchy window home leo5 leo5_gps leo7 leo7_gps

clean : 
	rm -f bin/cauchy_estimator bin/window_manager bin/homing_missile bin/leo_satellite_5state bin/leo_satellite_7state bin/leo_satellite_5state_gps bin/leo_satellite_7state_gps
