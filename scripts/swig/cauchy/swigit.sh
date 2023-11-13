#!/bin/bash

printf "This script wraps a C++ header only file to have a python interface through swig\nModify the contents of this file judiciously\n"

FILE_NAME="pycauchy" 
PYTHON_INC_PATH="-I/usr/local/include/python3.7m/"

SWIG_FILE=${FILE_NAME}.i
INCLUDE_FILE=${FILE_NAME}.hpp
LIB_LAPACK="-llapacke -llapack -lblas -lm -lpthread"


rm _${FILE_NAME}.so
rm ${FILE_NAME}_wrap.cxx
rm ${FILE_NAME}_wrap.o
rm ${FILE_NAME}.py
rm -rf __pycache__

echo "All temp files / libraries initially deleted"
#sleep 1
echo "Creating new temp files / libraries..."

swig -c++ -python ${SWIG_FILE}
if [ $? -eq 1 ]; then 
    echo "[ERROR:] swig -c++ -python ${SWIG_FILE} command returned with failure!"
    exit 1
fi
g++ -g -fpic -c ${FILE_NAME}_wrap.cxx $PYTHON_INC_PATH
if [ $? -eq 1 ]; then 
    echo "[ERROR:] g++ -fpic -c ${FILE_NAME}_wrap.cxx $PYTHON_INC_PATH command returned with failure!"
    exit 1
fi
g++ -shared ${FILE_NAME}_wrap.o -lstdc++ $LIB_LAPACK -o _${FILE_NAME}.so
if [ $? -eq 1 ]; then 
    echo "[ERROR:] g++ -shared ${FILE_NAME}_wrap.o -o _${FILE_NAME}.so -lstdc++ command returned with failure!"
    exit 1
fi
printf "All temp files / libraries (re)created!\nModule ${FILE_NAME}.py is now ready for use!\n"
