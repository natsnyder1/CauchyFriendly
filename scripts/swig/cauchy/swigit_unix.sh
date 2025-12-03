#!/bin/bash

printf "This script wraps a C++ header-only file to have a python interface through swig\nModify the contents of this file judiciously\n"

# First SWIG Configuration (pycauchy)
FILE_NAME="pycauchy" 
SWIG_FILE=${FILE_NAME}.i

# Include + Library path symbols
LIB_MATH_PTHREAD="-lm -lpthread"
INC_PYTHON=-I"/usr/include/python3.12"
LIB_PYTHON=-L"/usr/lib/x86_64-linux-gnu -lpython3.12"
INC_NUMPY=-I"/usr/lib/python3/dist-packages/numpy/core/include"
INC_GLPK=-I"/usr/include/"
LIB_GLPK=-L"/usr/lib -lm -lglpk -lpthread"

rm _${FILE_NAME}.so
rm ${FILE_NAME}_wrap.cxx
rm ${FILE_NAME}_wrap.o
rm ${FILE_NAME}.py
rm -rf __pycache__

echo "All temp files / libraries initially deleted"
echo "Creating new temp files / libraries..."

/home/agao0/Documents/research/CauchyRepo/CauchyFriendly/scripts/swig/swig_download/install_swig/bin/swig -c++ -python ${SWIG_FILE}
if [ $? -eq 1 ]; then 
    echo "[ERROR:] swig -c++ -python ${SWIG_FILE} command returned with failure!"
    exit 1
fi

g++ -O3 -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_NUMPY
if [ $? -eq 1 ]; then 
    echo "[ERROR:] g++ -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_NUMPY command returned with failure!"
    exit 1
fi

g++ $LIB_PYTHON -shared -lstdc++ $LIB_MATH_PTHREAD ${FILE_NAME}_wrap.o -o _${FILE_NAME}.so
if [ $? -eq 1 ]; then 
    echo "[ERROR:] g++ -shared ${FILE_NAME}_wrap.o -o _${FILE_NAME}.so -lstdc++ command returned with failure!"
    exit 1
fi

printf "All temp files / libraries (re)created!\nModule ${FILE_NAME}.py is now ready for use!\n"

# Second SWIG Configuration (enumeration/_enu)
FILE_NAME="enumeration/_enu" 
SWIG_FILE=${FILE_NAME}.i


rm enumeration/__enu.so
rm ${FILE_NAME}_wrap.cxx
rm ${FILE_NAME}_wrap.o
rm ${FILE_NAME}.py
rm -rf __pycache__

echo "All temp files / libraries initially deleted"
echo "Creating new temp files / libraries..."

/home/agao0/Documents/research/CauchyRepo/CauchyFriendly/scripts/swig/swig_download/install_swig/bin/swig -c++ -python ${SWIG_FILE}
if [ $? -eq 1 ]; then 
    echo "[ERROR:] swig -c++ -python ${SWIG_FILE} command returned with failure!"
    exit 1
fi

g++ -O3 -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_GLPK $INC_NUMPY -o ${FILE_NAME}_wrap.o
if [ $? -eq 1 ]; then 
    echo "[ERROR:] g++ -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_GLPK $INC_NUMPY -o ${FILE_NAME}_wrap.o command returned with failure!"
    exit 1
fi

g++ $LIB_PYTHON -shared -lstdc++ $LIB_GLPK ${FILE_NAME}_wrap.o -o enumeration/__enu.so
if [ $? -eq 1 ]; then 
    echo "[ERROR:] g++ $LIB_PYTHON -shared -lstdc++ $LIB_GLPK ${FILE_NAME}_wrap.o -o enumeration/__enu.so command returned with failure!"
    exit 1
fi

printf "All temp files / libraries (re)created!\nModule ${FILE_NAME}.py is now ready for use!\n"
