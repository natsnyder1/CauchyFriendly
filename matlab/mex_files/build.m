mex -setup c++
delete 'cpp_*_version.o' mcauchy_*

% Allow lldb to watch for memory leaks
setenv('CFLAGS', '-fno-omit-frame-pointer -fsanitize=address');
setenv('LDFLAGS', '-fsanitize=protect-initialized-data -fsanitize=leak -fsanitize=address');

% The following two lines are specific to Nishad's computer. Please change depending on your own locations
includePath = '-I/Users/natsnyder/Desktop/CauchyFriendly/scripts/swig/cauchy -I/Users/natsnyder/Desktop/CauchyFriendly/include';
libraryPath = '-lm -lpthread';


eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_initialize_lti.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_step.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_shutdown.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_set_window_number.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_reset.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_set_master_step.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_get_reinitialization_statistics.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_get_marginal_2D_pointwise_cpdf.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_get_marginal_1D_pointwise_cpdf.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_get_nonlin_function_pointers.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_initialize_nonlin.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_single_step_nonlin.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_free_names.cpp']);



eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_n.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_pncc.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_cmcc.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_p.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_step.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_dt.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_x.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_set_x.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_u.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_Phi.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_set_Phi.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_Gam.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_set_Gam.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_B.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_set_B.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_set_beta.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_H.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_set_H.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_get_gamma.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_set_gamma.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_set_is_xbar_set_for_ece.cpp']);
eval(['mex -g ', includePath, ' ', libraryPath, ' ../matlab_wrapped/mcauchy_dynamics_set_zbar.cpp']);


