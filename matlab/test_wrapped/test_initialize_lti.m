% Test script for validating the MEX function
% when building the cpp file, use the following command
%% mex -g -I/path/to/cauchy/CauchyFriendly/scripts/swig/cauchy -I/path/to/lapack/include -L/path/to/lapack/lib -llapacke -llapack -lblas initialize_lti.cpp
% replace t
% Define correct matrices and vectors
A0_correct = eye(2); % Identity matrix, square and full rank
p0_correct = [0.16089135 0.16089135]; % Non-negative vector matching the size of A0
b0_correct = [0. 0.]; % Matching size vector
Phi_correct = [.9, .1; -.2, 1.1]; % Diagonal matrix, square and full rank
B_correct = [];
Gamma_correct = [.1 .3]; % Another identity matrix
beta_correct = [0.02 0.02]; % Proper size vector
H_correct = [1 0.5]; % Row vector, should have matching columns
gamma_correct = [0.10175662]; % Scalar gamma


result_ptr = initialize_lti(A0_correct, p0_correct, b0_correct, Phi_correct, B_correct, Gamma_correct, beta_correct, H_correct, gamma_correct);
% Print the returned pointer
fprintf('Returned pointer: %lu\n', result_ptr);