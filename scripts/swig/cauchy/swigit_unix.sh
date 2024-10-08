#!/bin/bash

printf "This script wraps a C++ header only file to have a python interface through swig\nModify the contents of this file judiciously\n"

FILE_NAME="pycauchy" 
SWIG_FILE=${FILE_NAME}.i

# Include + Library path symbols
LIB_MATH_PTHREAD="-lm -lpthread"
INC_PYTHON=-I"/usr/local/include/python3.7m"
LIB_PYTHON=-L"/usr/local/lib -lpython3.7m"
INC_NUMPY=-I"/usr/local/lib/python3.7/site-packages/numpy/core/include"


# For cluster
#INC_PYTHON="-I/home/natsnyder1/.local/lib/python3.7/site-packages/numpy/core/include -I/cm/local/apps/python37/include/python3.7m"
#LIB_LAPACK="-Xlinker -start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Xlinker -end-group -lgomp -lpthread -lm -ldl"


rm _${FILE_NAME}.so
rm ${FILE_NAME}_wrap.cxx
rm ${FILE_NAME}_wrap.o
rm ${FILE_NAME}.py
rm -rf __pycache__

echo "All temp files / libraries initially deleted"
#sleep 1
echo "Creating new temp files / libraries..."

/home/natsubuntu/Desktop/SysControl/estimation/CauchyCPU/CauchyEst_Nat/CauchyFriendly/scripts/swig/swig_download/install_swig/bin/swig -c++ -python ${SWIG_FILE}
if [ $? -eq 1 ]; then 
    echo "[ERROR:] swig -c++ -python ${SWIG_FILE} command returned with failure!"
    exit 1
fi
g++ -O3 -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_NUMPY
if [ $? -eq 1 ]; then 
    echo "[ERROR:] clanclang++ -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_NUMPY command returned with failure!"
    exit 1
fi
g++ $LIB_PYTHON -shared -lstdc++ $LIB_MATH_PTHREAD ${FILE_NAME}_wrap.o -o _${FILE_NAME}.so
if [ $? -eq 1 ]; then 
    echo "[ERROR:] clanclang++ -shared ${FILE_NAME}_wrap.o -o _${FILE_NAME}.so -lstdc++ command returned with failure!"
    exit 1
fi
printf "All temp files / libraries (re)created!\nModule ${FILE_NAME}.py is now ready for use!\n"