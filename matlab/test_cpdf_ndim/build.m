mex -setup c++
delete 'cpp_*_version.o' test_4d_cpdf.o test_4d_cpdf.mexmaca64

% The following two lines are specific to Nishad's computer. Please change depending on your own locations
includePath = '-I/Users/nishadelias/Documents/GitHub/CauchyFriendly/scripts/swig/cauchy -I/Users/nishadelias/Documents/GitHub/CauchyFriendly/include';
libraryPath = '-lm -lpthread';

eval(['mex -g ', includePath, ' ', libraryPath, ' test_4d_cpdf.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' test_4d_marginal_cpdf.cpp']);