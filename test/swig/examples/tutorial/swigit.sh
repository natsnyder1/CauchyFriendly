swig -python gfg.i
gcc -fpic -c gfg_wrap.c gfg.c -I/usr/local/include/python3.7m
gcc -shared gfg.o gfg_wrap.o -o _gfg.so