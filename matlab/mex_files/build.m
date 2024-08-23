mex -setup c++
delete cpp_mexapi_version.o ...
    mcauchy_free_names.mexmaca64 mcauchy_free_names.o ...
    mcauchy_get_reinitialization_statistics.mexmaca64 mcauchy_get_reinitialization_statistics.o ...
    mcauchy_get_marginal_2D_pointwise_cpdf.mexmaca64 mcauchy_get_marginal_2D_pointwise_cpdf.o ...
    mcauchy_get_marginal_1D_pointwise_cpdf.mexmaca64 mcauchy_get_marginal_1D_pointwise_cpdf.o ...
    mcauchy_get_nonlin_function_pointers.mexmaca64 mcauchy_get_nonlin_function_pointers.o...
    mcauchy_initialize_lti.mexmaca64 mcauchy_initialize_lti.o ...
    mcauchy_initialize_nonlin.mexmaca64 mcauchy_initialize_nonlin.o ...
    mcauchy_reset.mexmaca64 mcauchy_reset.o ...
    mcauchy_set_window_number.mexmaca64 mcauchy_set_window_number.o ...
    mcauchy_set_master_step.mexmaca64 mcauchy_set_master_step.o ...
    mcauchy_shutdown.mexmaca64 mcauchy_shutdown.o ...
    mcauchy_single_step_nonlin.mexmaca64 mcauchy_single_step_nonlin.o ...
    mcauchy_step.mexmaca64 mcauchy_step.o
delete mcauchy_dynamics_get_n.mexmaca64 mcauchy_dynamics_get_n.o ...
    mcauchy_dynamics_get_pncc.mexmaca64 mcauchy_dynamics_get_pncc.o ...
    mcauchy_dynamics_get_cmcc.mexmaca64 mcauchy_dynamics_get_cmcc.o ...
    mcauchy_dynamics_get_p.mexmaca64 mcauchy_dynamics_get_p.o ...
    mcauchy_dynamics_get_step.mexmaca64 mcauchy_dynamics_get_step.o ...
    mcauchy_dynamics_get_dt.mexmaca64 mcauchy_dynamics_get_dt.o ...
    mcauchy_dynamics_get_x.mexmaca64 mcauchy_dynamics_get_x.o ...
    mcauchy_dynamics_set_x.mexmaca64 mcauchy_dynamics_set_x.o ...
    mcauchy_dynamics_get_u.mexmaca64 mcauchy_dynamics_get_u.o ...
    mcauchy_dynamics_get_Phi.mexmaca64 mcauchy_dynamics_get_Phi.o ...
    mcauchy_dynamics_set_Phi.mexmaca64 mcauchy_dynamics_set_Phi.o ...
    mcauchy_dynamics_get_Gam.mexmaca64 mcauchy_dynamics_get_Gam.o ...
    mcauchy_dynamics_set_Gam.mexmaca64 mcauchy_dynamics_set_Gam.o ...
    mcauchy_dynamics_get_B.mexmaca64 mcauchy_dynamics_get_B.o ...
    mcauchy_dynamics_set_B.mexmaca64 mcauchy_dynamics_set_B.o ...
    mcauchy_dynamics_set_beta.mexmaca64 mcauchy_dynamics_set_beta.o ...
    mcauchy_dynamics_get_H.mexmaca64 mcauchy_dynamics_get_H.o ...
    mcauchy_dynamics_set_H.mexmaca64 mcauchy_dynamics_set_H.o ...
    mcauchy_dynamics_get_gamma.mexmaca64 mcauchy_dynamics_get_gamma.o ...
    mcauchy_dynamics_set_gamma.mexmaca64 mcauchy_dynamics_set_gamma.o ...
    mcauchy_dynamics_set_is_xbar_set_for_ece.mexmaca64 mcauchy_dynamics_set_is_xbar_set_for_ece.o...
    mcauchy_dynamics_set_zbar.mexmaca64 mcauchy_dynamics_set_zbar.o


% Allow lldb to watch for memory leaks
setenv('CFLAGS', '-fno-omit-frame-pointer -fsanitize=address');
setenv('LDFLAGS', '-fsanitize=protect-initialized-data -fsanitize=leak -fsanitize=address');

% The following two lines are specific to Nishad's computer. Please change depending on your own locations
%includePath = '-I/home/natsubuntu/Desktop/SysControl/estimation/CauchyCPU/CauchyEst_Nat/CauchyFriendly/scripts/swig/cauchy -I/home/natsubuntu/Desktop/SysControl/estimation/CauchyCPU/CauchyEst_Nat/CauchyFriendly/include';
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


