delete cpp_mexapi_version.o mcauchy_step.mexmaca64 mcauchy_step.o minitialize_lti.mexmaca64 minitialize_lti.o

setenv('CFLAGS', '-fno-omit-frame-pointer -fsanitize=address');
setenv('LDFLAGS', '-fsanitize=protect-initialized-data -fsanitize=leak -fsanitize=address');

mex -g -I/Users/nishadelias/Documents/GitHub/CauchyFriendly/scripts/swig/cauchy -I/opt/homebrew/opt/lapack/include -L/opt/homebrew/opt/lapack/lib -llapacke -llapack -lblas -lm -lpthread matlab_wrapped/minitialize_lti.cpp
mex -g -I/Users/nishadelias/Documents/GitHub/CauchyFriendly/scripts/swig/cauchy -I/opt/homebrew/opt/lapack/include -L/opt/homebrew/opt/lapack/lib -llapacke -llapack -lblas -lm -lpthread matlab_wrapped/mcauchy_step.cpp
mex -g -I/Users/nishadelias/Documents/GitHub/CauchyFriendly/scripts/swig/cauchy -I/opt/homebrew/opt/lapack/include -L/opt/homebrew/opt/lapack/lib -llapacke -llapack -lblas -lm -lpthread matlab_wrapped/mshutdown.cpp