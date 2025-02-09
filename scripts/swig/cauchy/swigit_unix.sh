#!/bin/bash

printf "This script wraps a C++ header only file to have a python interface through swig\nModify the contents of this file judiciously\n"

FILE_NAME="pycauchy" 
SWIG_FILE=${FILE_NAME}.i

# Include + Library path symbols
LIB_MATH_PTHREAD="-lm -lpthread"
INC_PYTHON=-I"/Library/Frameworks/Python.framework/Versions/3.12/include/python3.12"
LIB_PYTHON=-L"/Library/Frameworks/Python.framework/Versions/3.12/lib -lpython3.12"
INC_NUMPY=-I"/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/numpy/core/include"


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

/Users/nishadelias/Documents/GitHub/CauchyFriendly/scripts/swig/swig_download/install_swig/bin/swig -c++ -python ${SWIG_FILE}
if [ $? -eq 1 ]; then 
    echo "[ERROR:] swig -c++ -python ${SWIG_FILE} command returned with failure!"
    exit 1
fi
clang++ -O3 -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_NUMPY
if [ $? -eq 1 ]; then 
    echo "[ERROR:] clanclang++ -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_NUMPY command returned with failure!"
    exit 1
fi
clang++ $LIB_PYTHON -dynamiclib -lstdc++ $LIB_MATH_PTHREAD ${FILE_NAME}_wrap.o -o _${FILE_NAME}.so
if [ $? -eq 1 ]; then 
    echo "[ERROR:] clanclang++ -dynamiclib ${FILE_NAME}_wrap.o -o _${FILE_NAME}.so -lstdc++ command returned with failure!"
    exit 1
fi
printf "All temp files / libraries (re)created!\nModule ${FILE_NAME}.py is now ready for use!\n"


FILE_NAME="enumeration/_enu" 
SWIG_FILE=${FILE_NAME}.i
# Nats computer
# LIB_LAPACK="-llapacke -llapack -lblas -lm -lpthread"
# LIB_GLPK="-lm -lglpk -lpthread"

# Nishad's Computer
INC_GLPK="-I/opt/homebrew/Cellar/glpk/5.0/include/"
LIB_GLPK="-L/opt/homebrew/Cellar/glpk/5.0/lib -lm -lglpk -lpthread"

# For cluster
#PYTHON_INC_PATH="-I/home/natsnyder1/.local/lib/python3.7/site-packages/numpy/core/include -I/cm/local/apps/python37/include/python3.7m"
#LIB_LAPACK="-Xlinker -start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Xlinker -end-group -lgomp -lpthread -lm -ldl"


rm enumeration/__enu.so
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
clang++ -O3 -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_GLPK $INC_NUMPY -o ${FILE_NAME}_wrap.o
if [ $? -eq 1 ]; then 
    echo "[ERROR:] clang++ -fpic -c ${FILE_NAME}_wrap.cxx $INC_PYTHON $INC_GLPK $INC_NUMPY -o ${FILE_NAME}_wrap.o command returned with failure!"
    exit 1
fi
clang++ $LIB_PYTHON -dynamiclib -lstdc++ ${LIB_GLPK} ${FILE_NAME}_wrap.o -o enumeration/__enu.so
if [ $? -eq 1 ]; then 
    echo "[ERROR:] clang++ $LIB_PYTHON -dynamiclib -lstdc++ ${LIB_GLPK} ${FILE_NAME}_wrap.o -o enumeration/__enu.so command returned with failure!"
    exit 1
fi
printf "All temp files / libraries (re)created!\nModule ${FILE_NAME}.py is now ready for use!\n"